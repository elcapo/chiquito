# On-the-Fly Quantization with bitsandbytes

In layer-by-layer inference, the bottleneck is transferring weights from CPU to GPU. Quantization reduces the size of those weights, directly speeding up the transfer. Chiquito supports 4-bit and 8-bit quantization via the **bitsandbytes** library, applied on-the-fly as weights are loaded onto the GPU.

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

### 8-bit: LLM.int8

Each weight is stored as an 8-bit integer plus a per-group scaling factor. During computation, weights are dequantized back to fp16 on-the-fly using optimized CUDA kernels.

### 4-bit: NF4 (Normal Float 4)

NF4 is an information-theoretically optimal 4-bit format for normally distributed values (which neural network weights tend to be). Each weight is stored as a 4-bit code that indexes into a 16-value codebook derived from the normal distribution.

Both formats are provided by bitsandbytes and integrated into HuggingFace transformers.

## How Chiquito applies quantization

### Step 1: Configure bitsandbytes

During model initialization, `_init_model()` creates a `BitsAndBytesConfig` and applies it via `AutoHfQuantizer` ([`model.py:211-237`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L211-L237)):

```python
if self._quantization is not None:
    from transformers import BitsAndBytesConfig
    from transformers.quantizers import AutoHfQuantizer

    if self._quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self._dtype,
            bnb_4bit_quant_type="nf4",
        )
    elif self._quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(...)

    self.hf_quantizer = AutoHfQuantizer.from_config(bnb_config)
    device_map = self.hf_quantizer.update_device_map(None)
    self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)
```

The `preprocess_model()` call replaces standard `nn.Linear` modules with bitsandbytes equivalents:
- 4-bit: `bnb.nn.Linear4bit`
- 8-bit: `bnb.nn.Linear8bitLt`

These are still on the meta device at this point — no quantization has happened yet. The modules are just **prepared** to accept quantized weights.

### Step 2: Quantize on-the-fly during weight loading

The quantization happens inside `set_module_tensor_to_device()` (from the `accelerate` library), which Chiquito calls in `_move_layer_to_device()` ([`model.py:318-328`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L318-L328)):

```python
def _move_layer_to_device(self, state_dict):
    moved = list(state_dict.keys())
    for param_name in moved:
        set_module_tensor_to_device(
            self.model, param_name, self._device,
            value=state_dict[param_name], dtype=self._dtype,
        )
    return moved
```

When `set_module_tensor_to_device` detects that the target module is a bitsandbytes linear layer, it automatically quantizes the fp16 tensor as it places it on the GPU. The fp16 data goes in, and the stored result is 4-bit (or 8-bit). This is transparent to our code.

### Step 3: Forward computation with quantized weights

During the forward pass, the bitsandbytes linear layers use optimized CUDA kernels to dequantize weights on-the-fly during matrix multiplication. The dequantization cost is small compared to the transfer time savings.

## The reset problem

There is one important complication. Bitsandbytes modules maintain internal **quantization state** — absmax values, codebooks, and other metadata computed during quantization. This state cannot survive a round-trip to the meta device. If we move a `Linear4bit` module to meta and back, the quantization state is lost and the module breaks.

This is why quantized models need a full `del model` + `_init_model()` reset instead of the lightweight `_reset_model_to_meta()` ([`model.py:427-434`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L427-L434)):

```python
if self.hf_quantizer is not None:
    # Quantized models need full reinit
    del self.model
    clean_memory()
    self._init_model()
else:
    self._reset_model_to_meta()
    clean_gpu_memory()
```

This is more expensive (creates a new model object from scratch), but it is the only way to get clean bitsandbytes modules ready for fresh quantization. The heavy `clean_memory()` call (gc + malloc_trim + empty_cache) ensures the old model is fully freed.

## Pre-quantized models

Some HuggingFace models are distributed already quantized (the weights on disk are already in 4-bit or 8-bit format). Chiquito detects this via `config.quantization_config` and handles it similarly ([`model.py:230-237`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L230-L237)):

```python
elif quantization_config is not None:
    from transformers.quantizers import AutoHfQuantizer

    self.hf_quantizer = AutoHfQuantizer.from_config(
        quantization_config, pre_quantized=True
    )
    device_map = self.hf_quantizer.update_device_map(None)
    self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)
```

The `pre_quantized=True` flag tells the quantizer that the weights are already quantized, so it should not re-quantize them during loading.

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
| **How** | bitsandbytes replaces `nn.Linear` with quantized versions; `set_module_tensor_to_device` quantizes on placement |
| **4-bit** | NF4 format, ~4x transfer speedup, `BitsAndBytesConfig(load_in_4bit=True)` |
| **8-bit** | LLM.int8, ~2x transfer speedup, `BitsAndBytesConfig(load_in_8bit=True)` |
| **Caveat** | Quantized models cannot use `_reset_model_to_meta()` — they need full reinit |

Quantization is the last major optimization. Together with RAM preloading and the sliding window cache, it makes layer-by-layer inference practical for large models on consumer hardware.
