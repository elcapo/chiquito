# Quantization

## Why quantize?

In layer-by-layer inference, the dominant cost is transferring each layer's weights from CPU to GPU. A 32B fp16 model has ~1 GB per transformer layer. At PCIe 3.0 speeds (~12 GB/s practical), that's ~80ms per layer just for the transfer — multiplied by 67 layers and 20 tokens, it adds up fast.

Quantization reduces the weight size:

| Precision | Size per layer (32B model) | Transfer time |
|---|---|---|
| fp16 | ~1 GB | ~80ms |
| 8-bit | ~500 MB | ~40ms |
| 4-bit | ~250 MB | ~20ms |

The 4x reduction in transfer time from 4-bit quantization directly translates to faster inference, since weight loading is the bottleneck.

Additionally, a 32B model at 4-bit (~16 GB) fits in 64 GB of RAM with `preload_to_ram=True`, which wasn't possible at fp16 (~65 GB).

## How it works in Chiquito

Chiquito uses [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) for on-the-fly quantization. The fp16 weights are stored on disk (in the split safetensors files) and quantized as they are loaded onto the GPU.

```python
model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    quantization="4bit",  # or "8bit"
)
```

### What happens under the hood

1. **Model init**: `_init_model()` creates the model with `init_empty_weights()`, then applies a `BitsAndBytesConfig` via transformers' `AutoHfQuantizer`. This replaces `nn.Linear` modules with `bnb.nn.Linear4bit` (or `Linear8bitLt` for 8-bit).

2. **Weight loading**: `set_module_tensor_to_device()` from accelerate detects the bnb module type and quantizes the fp16 weight tensor on the fly as it's placed on the GPU. The quantized weight uses ~4x less GPU memory.

3. **Forward pass**: The bnb modules dequantize on the fly during matrix multiplication, using optimized CUDA kernels.

4. **Model reset**: Because bnb modules have internal quantization state that can't survive a round-trip to the meta device, quantized models use the full `del model; _init_model()` reset path instead of the lightweight `_reset_model_to_meta()`.

### 4-bit vs 8-bit

| | 4-bit (NF4) | 8-bit (LLM.int8) |
|---|---|---|
| **Size reduction** | ~4x | ~2x |
| **Quality impact** | Small — NF4 is information-theoretically optimal for normal distributions | Minimal — dynamic quantization preserves outliers |
| **Speed** | Faster transfers, slight dequantization overhead | Moderate transfer improvement |
| **Best for** | Maximum speed, models that barely fit in RAM | When quality matters more than speed |

### Quantization vs. pre-quantized models

Chiquito quantizes **on the fly** from fp16 weights. This is different from using pre-quantized model files (GPTQ, AWQ), which store already-quantized weights and need specialized libraries (`gptqmodel`, `autoawq`) for their custom CUDA kernels.

The on-the-fly approach:
- Works with any fp16 HuggingFace model — no need to find a pre-quantized variant.
- Adds quantization overhead to each layer load (small, dominated by transfer time).
- Only requires `bitsandbytes` as an extra dependency.

Pre-quantized models could in principle be faster (skip quantization step, smaller files on disk), but supporting them requires additional dependencies that are harder to install and maintain. This may be added in the future.
