# Mixture-of-Experts Models

Some modern models, such as Qwen3.5-MoE, replace the standard feed-forward network inside each transformer layer with a **Mixture-of-Experts** (MoE) block. Instead of a single FFN, there are many "expert" FFNs and a lightweight router that picks which experts to activate for each token. Only a fraction of the experts run on each forward pass, so the model can have many more total parameters without proportionally increasing compute.

For Chiquito, MoE models introduce two new challenges:

1. **Composite model naming** — the on-disk weight keys do not match the model-object attribute paths.
2. **Fused expert tensors** — expert weights are stored as 3-D tensors that do not fit the standard `nn.Linear` quantization path.

This chapter covers how Chiquito handles both.

## What changes in a MoE layer

In a standard transformer layer, the FFN has two linear projections (gate/up and down). In a MoE layer, these projections are replicated across many experts. HuggingFace stores the expert weights as **fused 3-D tensors**: instead of separate 2-D `(out_features, in_features)` weights per expert, the checkpoint contains a single `(num_experts, out_features, in_features)` tensor.

For example, Qwen3.5-MoE has 128 experts per layer. Its `gate_up_proj` weight has shape `(128, 9216, 2048)` — all 128 experts fused into one tensor. At fp16, a single fused tensor like this occupies ~4.8 GB.

## The composite model problem

Many MoE models (and vision-language models in general) are **composite**: the checkpoint stores text-model weights under a prefix like `model.language_model.*`, but `AutoModelForCausalLM.from_config(text_config)` produces a model whose attributes live directly under `model.*`. Chiquito's standard `ChiquitoModel` assumes that on-disk key names and model-object attribute paths match. For composite models, they do not.

`ChiquitoCompositeModel` bridges this gap by maintaining two separate name mappings ([`composite_model.py:27-43`](https://github.com/elcapo/chiquito/blob/main/src/chiquito/composite_model.py#L27-L43)):

```python
class ChiquitoCompositeModel(ChiquitoModel):
    # On-disk weight key prefixes (used by the splitter)
    LAYER_NAMES = {
        "embed": "model.language_model.embed_tokens",
        "layer_prefix": "model.language_model.layers",
        "norm": "model.language_model.norm",
        "lm_head": "lm_head",
    }

    # Model-object attribute paths (for getattr on the meta model)
    _MODEL_LAYER_NAMES = {
        "embed": "model.embed_tokens",
        "layer_prefix": "model.layers",
        "norm": "model.norm",
        "lm_head": "lm_head",
    }

    _DISK_PREFIX = "model.language_model."
    _MODEL_PREFIX = "model."
```

`LAYER_NAMES` tells the splitter where to find weights on disk. `_MODEL_LAYER_NAMES` tells `_build_layers()` how to walk the actual model object. The subclass overrides `_build_layers()` to use the model-object paths while storing the disk paths in `self.layer_names`.

### Key remapping

When a layer's weights are loaded from disk, they arrive with keys like `model.language_model.layers.5.self_attn.q_proj.weight`. But the model object expects `model.layers.5.self_attn.q_proj.weight`. The `_remap_state_dict_keys()` override translates between the two ([`composite_model.py:95-106`](https://github.com/elcapo/chiquito/blob/main/src/chiquito/composite_model.py#L95-L106)):

```python
def _remap_state_dict_keys(self, state_dict):
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith(self._DISK_PREFIX):
            new_key = self._MODEL_PREFIX + key[len(self._DISK_PREFIX):]
            remapped[new_key] = value
        else:
            remapped[key] = value
    return remapped
```

This method is called in the forward pass (in the base class) right after loading weights and before moving them to the GPU, so the remapping is transparent to the rest of the engine.

## Eager expert dispatch

HuggingFace's MoE implementations can dispatch experts in two modes: an optimized "fused" mode that operates on the full 3-D tensor, and an "eager" mode that accesses experts via scalar indexing (`self.gate_up_proj[expert_idx]`). Chiquito needs the eager mode because it replaces the fused tensors with lazy wrappers (described below). The composite model forces this during `_build_layers()` ([`composite_model.py:54-59`](https://github.com/elcapo/chiquito/blob/main/src/chiquito/composite_model.py#L54-L59)):

```python
config = getattr(self, "config", None)
if config is not None:
    if hasattr(config, "_experts_implementation"):
        config._experts_implementation = "eager"
    if hasattr(config, "_experts_implementation_internal"):
        config._experts_implementation_internal = "eager"
```

## Quantizing fused expert tensors

The standard quantization path in `_quantize_state_dict()` handles 2-D `nn.Linear` weights. Fused expert weights are 3-D. To support them, the quantizer reshapes 3-D+ tensors to 2-D before calling bitsandbytes, and stores the original shape so the loader can restore it ([`splitter.py:82-106`](https://github.com/elcapo/chiquito/blob/main/src/chiquito/splitter.py#L82-L106)):

```python
# Flatten to 2-D for bitsandbytes; save original shape for 3-D+.
original_shape = tensor.shape
flat = tensor.reshape(-1, tensor.shape[-1])

gpu_tensor = flat.to(torch.float16).to(device).contiguous()
packed, quant_state = F.quantize_4bit(gpu_tensor, quant_type="nf4", blocksize=64)

result[name] = packed.cpu()
for qs_key, qs_val in quant_state.as_dict(packed=True).items():
    result[f"{name}.{qs_key}"] = qs_val.cpu() if isinstance(qs_val, torch.Tensor) else qs_val

# Store original shape so the loader can reshape after dequantization.
if len(original_shape) > 2:
    result[f"{name}.original_shape"] = torch.tensor(original_shape, dtype=torch.int64)
```

The `original_shape` entry is what tells the loader that this is a fused tensor rather than a standard linear weight.

## Lazy per-expert dequantization

The most interesting part of MoE support is how quantized expert weights are loaded at inference time. Dequantizing all 128 experts from 4-bit to fp16 at once would require ~4.8 GB of VRAM just for a single weight tensor — defeating the purpose of layer-by-layer inference on low-VRAM hardware.

The solution is `LazyDequantExperts` ([`lazy_experts.py`](https://github.com/elcapo/chiquito/blob/main/src/chiquito/lazy_experts.py)): a wrapper that keeps the packed 4-bit data on GPU (~1.2 GB) and dequantizes individual expert slices on-demand when indexed.

```python
class LazyDequantExperts:
    def __init__(self, packed, absmax, code, quant_type, blocksize,
                 source_dtype, original_shape, dtype, device):
        self.device = torch.device(device)
        self.num_experts = original_shape[0]
        self._expert_rows = original_shape[1]
        self._expert_cols = original_shape[2]
        self._expert_numel = self._expert_rows * self._expert_cols
        self._expert_packed_bytes = self._expert_numel // 2
        self._expert_blocks = self._expert_numel // blocksize

        self._packed = packed.to(self.device)
        self._absmax = absmax.to(self.device)
        self._code = code.to(self.device)
        # ...
```

The key method is `__getitem__`, which dequantizes a single expert's 2-D weight matrix:

```python
def __getitem__(self, expert_idx: int) -> torch.Tensor:
    from bitsandbytes.functional import QuantState, dequantize_4bit

    p_start = expert_idx * self._expert_packed_bytes
    p_end = p_start + self._expert_packed_bytes
    a_start = expert_idx * self._expert_blocks
    a_end = a_start + self._expert_blocks

    partial_qs = QuantState(
        absmax=self._absmax[a_start:a_end],
        shape=torch.Size([self._expert_rows, self._expert_cols]),
        blocksize=self._blocksize,
        code=self._code,
        quant_type=self._quant_type,
        dtype=self._source_dtype,
    )

    return dequantize_4bit(self._packed[p_start:p_end], partial_qs).to(self.dtype)
```

Because HuggingFace's eager MoE forward loop accesses experts via `self.gate_up_proj[expert_idx]` with a scalar integer index, `__getitem__` is triggered automatically. Each call dequantizes only one expert's 2-D slice, keeping peak VRAM proportional to `1 expert` instead of `num_experts`.

### How the wrapper is installed

The composite model's `_move_quantized_layer_to_device()` override detects fused expert tensors (by checking for `original_shape` and `.experts.` in the key name) and wraps them in `LazyDequantExperts` instead of creating standard `Params4bit` objects ([`composite_model.py:128-175`](https://github.com/elcapo/chiquito/blob/main/src/chiquito/composite_model.py#L128-L175)):

```python
if name in qs_map and "original_shape" in qs_map[name] and ".experts." in name:
    wrapper = LazyDequantExperts.from_quantized(
        packed=tensor,
        qs_entries=qs_map.pop(name),
        dtype=self._dtype,
        device=self._device,
    )
    # Replace the nn.Parameter with the lazy wrapper
    attr_parts = name.split(".")
    module = self.model
    for s in attr_parts[:-1]:
        module = getattr(module, s)
    if attr_name in module._parameters:
        del module._parameters[attr_name]
    setattr(module, attr_name, wrapper)
```

The wrapper is set as a plain attribute — not as `nn.Parameter` — because `nn.Parameter` would try to include it in gradient computation and move it to meta device during cleanup in a way that does not work for non-tensor objects.

### Cleanup

After the layer executes, the lazy wrappers need to be cleaned up differently from regular parameters. The composite model overrides `_cleanup_moved_params()` to detect wrappers and replace them with empty meta parameters ([`composite_model.py:177-197`](https://github.com/elcapo/chiquito/blob/main/src/chiquito/composite_model.py#L177-L197)):

```python
def _cleanup_moved_params(self, moved):
    for param_name in moved:
        parts = param_name.split(".")
        module = self.model
        for s in parts[:-1]:
            module = getattr(module, s)
        attr_name = parts[-1]
        current = getattr(module, attr_name, None)

        if isinstance(current, LazyDequantExperts):
            delattr(module, attr_name)
            module._parameters[attr_name] = torch.nn.Parameter(
                torch.empty(current.shape, device="meta"),
                requires_grad=False,
            )
        else:
            set_module_tensor_to_device(self.model, param_name, "meta")

    clean_gpu_memory()
```

## VRAM impact

The lazy dequantization strategy has a dramatic effect on peak VRAM usage for quantized MoE models:

| Strategy | VRAM for `gate_up_proj` (128 experts) |
|----------|--------------------------------------|
| Full dequantize to fp16 | ~4.8 GB |
| Keep packed 4-bit + dequant one expert | ~1.2 GB + ~37 MB per active expert |

Since the router typically activates only a few experts per token, peak VRAM stays close to the packed size.

## Registration

The composite model is registered in `__init__.py` so that `AutoModel.from_pretrained()` automatically selects it for Qwen3.5-MoE architectures:

```python
from .auto_model import AutoModel
from .composite_model import ChiquitoCompositeModel

AutoModel.register("Qwen3_5Moe", ChiquitoCompositeModel)
```

When the config's architecture string contains `"Qwen3_5Moe"`, the factory returns a `ChiquitoCompositeModel` instead of the base `ChiquitoModel`. This is the same registry mechanism described in [Architecture Extensibility](11-extensibility.md).

## Usage

From the user's perspective, MoE models work exactly like any other model:

```python
from chiquito import AutoModel

model = AutoModel.from_pretrained(
    "Qwen/Qwen3.5-MoE-A3B-Instruct",
    quantization="4bit",
)

tokens = model.tokenizer("Hello, world!", return_tensors="pt")
output = model.generate(tokens["input_ids"].cuda(), max_new_tokens=50)
print(model.tokenizer.decode(output[0], skip_special_tokens=True))
```

The `AutoModel` factory detects the MoE architecture, selects `ChiquitoCompositeModel`, and everything — checkpoint splitting, key remapping, lazy expert dequantization — happens automatically.

## Summary

| Component | Purpose | Where |
|-----------|---------|-------|
| `ChiquitoCompositeModel` | Dual name mappings (disk vs model-object) + MoE overrides | [`composite_model.py`](https://github.com/elcapo/chiquito/blob/main/src/chiquito/composite_model.py) |
| `LazyDequantExperts` | On-demand per-expert 4-bit dequantization | [`lazy_experts.py`](https://github.com/elcapo/chiquito/blob/main/src/chiquito/lazy_experts.py) |
| `_quantize_state_dict` (3-D path) | Reshape + quantize fused tensors, store `original_shape` | [`splitter.py`](https://github.com/elcapo/chiquito/blob/main/src/chiquito/splitter.py) |
| `AutoModel.register("Qwen3_5Moe", ...)` | Auto-detection for MoE architectures | [`__init__.py`](https://github.com/elcapo/chiquito/blob/main/src/chiquito/__init__.py) |

MoE support builds on every layer of the engine — the splitter, the quantizer, the extensibility system, and the load-execute-free cycle — while keeping the user-facing API unchanged.
