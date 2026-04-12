# Pre-Quantized Weight Caching with bitsandbytes

In layer-by-layer inference, the bottleneck is transferring weights from CPU to GPU. Quantization reduces the size of those weights, directly speeding up the transfer. Chiquito supports 4-bit and 8-bit quantization via the **bitsandbytes** library. Weights are quantized once on the first run and cached to disk so that subsequent runs load the already-quantized data directly.

## Why quantization helps here

The speed gain from quantization in Chiquito is not about reducing computation (the GPU is fast enough) — it is about **reducing data transfer**:

| Precision | Size per layer (32B model) | PCIe transfer time | Reduction |
|-----------|--------------------------|-------------------|-----------|
| fp16 | ~1 GB | ~80 ms | baseline |
| 8-bit | ~500 MB | ~40 ms | 2x |
| 4-bit | ~250 MB | ~20 ms | 4x |

For a 67-layer model generating 20 tokens, this means:
- fp16: 67 x 20 x 80 ms = ~107 seconds of transfer
- 4-bit: 67 x 20 x 20 ms = ~27 seconds of transfer

That is a ~4x speedup in the dominant cost.

## The two quantization formats

### 8-bit: block-wise absmax

Each weight block (64 elements) is stored as 8-bit integers plus a per-block scaling factor (absmax). During computation, weights are dequantized back to fp16 on-the-fly using optimized CUDA kernels.

### 4-bit: NF4 (Normal Float 4)

NF4 is an information-theoretically optimal 4-bit format for normally distributed values (which neural network weights tend to be). Each weight is stored as a 4-bit code that indexes into a 16-value codebook derived from the normal distribution.

Both formats are provided by bitsandbytes and integrated into HuggingFace transformers.

## Per-quantization split directories

Each quantization level gets its own directory alongside the base fp16 split:

| Quantization | Directory | Contents |
|---|---|---|
| None (fp16) | `chiquito_split/` | Original fp16 weights per layer |
| 4-bit | `chiquito_split_4bit/` | Packed NF4 data + quant state |
| 8-bit | `chiquito_split_8bit/` | Block-wise int8 data + quant state |

The naming is handled by `split_dir_name()` in `splitter.py`:

```python
def split_dir_name(quantization: str | None = None) -> str:
    if quantization:
        return f"{SPLIT_DIR_NAME}_{quantization}"
    return SPLIT_DIR_NAME
```

## How Chiquito creates the quantized split

### Step 1: Ensure the base (fp16) split exists

On the first run, `split_and_save_layers()` creates the base `chiquito_split/` directory with per-layer safetensors files — the same as in non-quantized mode.

### Step 2: Quantize and save each layer

When a quantization level is requested, `split_and_save_layers()` reads each layer from the base split, quantizes the 2-D+ weight tensors with bitsandbytes, and writes the result to the quantized directory:

```python
fp16_data = load_safetensors(layer_file_path(base_dir, layer_name))
quantized = _quantize_state_dict(fp16_data, quantization)
save_safetensors(quantized, layer_file_path(split_dir, layer_name))
```

The `_quantize_state_dict()` function calls `bitsandbytes.functional.quantize_4bit()` (or `quantize_blockwise()` for 8-bit) and serializes the result using `QuantState.as_dict(packed=True)`, which converts the quantization metadata into tensors suitable for safetensors:

- The packed weight data (uint8 for 4-bit, int8 for 8-bit) is saved under the original parameter name.
- The quant-state entries (absmax, quant\_map, etc.) are saved with suffixed keys like `weight.absmax`, `weight.quant_map`.

### Step 3: On subsequent runs, skip everything

`split_and_save_layers()` checks for `.done` markers per layer. If the entire quantized split is complete, it returns immediately:

```
All layers already split in /path/to/model/chiquito_split_4bit
```

## How Chiquito loads pre-quantized weights

### Step 1: Configure bitsandbytes modules

During model initialization, `_init_model()` still creates a `BitsAndBytesConfig` and calls `preprocess_model()` to replace `nn.Linear` with `bnb.nn.Linear4bit` (or `Linear8bitLt`). This is needed so the forward pass uses the quantized CUDA kernels.

### Step 2: Load and reconstruct quantized parameters

When loading a layer, `_move_quantized_layer_to_device()` uses `parse_quantized_state_dict()` to separate packed weight data from quant-state entries:

```python
base_params, qs_map = parse_quantized_state_dict(raw_state_dict)
```

For each quantized weight, it reconstructs a `Params4bit` (or `Int8Params`) with `bnb_quantized=True` and places it on the GPU directly — bypassing `set_module_tensor_to_device()` which would attempt to re-quantize:

```python
qs = QuantState.from_dict(qs_map[name], device=self._device)
param = bnb.nn.Params4bit(
    tensor, requires_grad=False,
    quant_state=qs, quant_type=qs.quant_type,
    blocksize=qs.blocksize, bnb_quantized=True,
)
module._parameters[splits[-1]] = param.to(self._device)
```

Non-quantized tensors (biases, norm weights) are loaded normally via `set_module_tensor_to_device()`.

### Step 3: Lightweight model reset

Pre-quantized models use the lightweight `_reset_model_to_meta()` between forward passes. The bnb module types (`Linear4bit`, `Linear8bitLt`) survive the meta round-trip, and `_move_quantized_layer_to_device()` replaces parameters directly on each layer load.

This avoids the expensive `del model; _init_model()` cycle that was previously needed for quantized models.

## Pre-quantized HuggingFace models

Some HuggingFace models are distributed already quantized (the weights on disk are already in a quantized format). Chiquito detects this via `config.quantization_config` and handles it with the `pre_quantized=True` flag on the quantizer.

## Usage

```python
from chiquito import AutoModel

# 4-bit quantization
model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    quantization="4bit",
)

# 8-bit quantization
model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    quantization="8bit",
)
```

## Summary

| Aspect | Detail |
|--------|--------|
| **Why** | Smaller weights = faster CPU→GPU transfer (the bottleneck) |
| **How** | Weights quantized once, cached to `chiquito_split_{4bit,8bit}/`, loaded pre-quantized on subsequent runs |
| **4-bit** | NF4 format, ~4x transfer speedup, ~4x less RAM with `preload_to_ram=True` |
| **8-bit** | Block-wise int8, ~2x transfer speedup, ~2x less RAM |
| **Reset** | Pre-quantized models use lightweight `_reset_model_to_meta()` — no full reinit needed |

Quantization is the last major optimization. Together with RAM preloading and the sliding window cache, it makes layer-by-layer inference practical for large models on consumer hardware.
