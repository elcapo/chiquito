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

Chiquito uses [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) for quantization. When you specify a quantization level, Chiquito creates a dedicated split directory with pre-quantized weights so that subsequent runs skip the quantization step entirely.

```python
model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    quantization="4bit",  # or "8bit"
)
```

### Per-quantization split directories

Each quantization level gets its own split directory:

| Quantization | Directory |
|---|---|
| None (fp16) | `chiquito_split/` |
| 4-bit | `chiquito_split_4bit/` |
| 8-bit | `chiquito_split_8bit/` |

On the first run with a given quantization setting:

1. The base fp16 split (`chiquito_split/`) is created if it doesn't exist.
2. Each layer's fp16 weights are quantized using bitsandbytes and saved into the quantized split directory (e.g. `chiquito_split_4bit/`).
3. A `.done` marker is written per layer so interrupted runs can resume.

On subsequent runs, Chiquito detects that the quantized split already exists and skips both splitting and quantization.

### What happens under the hood

1. **Model init**: `_init_model()` creates the model with `init_empty_weights()`, then applies a `BitsAndBytesConfig` via transformers' `AutoHfQuantizer`. This replaces `nn.Linear` modules with `bnb.nn.Linear4bit` (or `Linear8bitLt` for 8-bit).

2. **Weight loading**: Pre-quantized safetensors are loaded from the quantized split directory. The packed weight data and quant-state metadata are reconstructed into `Params4bit` (or `Int8Params`) objects and placed directly on the GPU — no re-quantization needed.

3. **Forward pass**: The bnb modules dequantize on the fly during matrix multiplication, using optimized CUDA kernels.

4. **Model reset**: Pre-quantized models use the lightweight `_reset_model_to_meta()` reset between forward passes, since the bnb module types survive the meta round-trip and weights are reconstructed from the pre-quantized files on each layer load.

### 4-bit vs 8-bit

| | 4-bit (NF4) | 8-bit (block-wise) |
|---|---|---|
| **Size reduction** | ~4x | ~2x |
| **Quality impact** | Small — NF4 is information-theoretically optimal for normal distributions | Minimal — block-wise quantization preserves accuracy |
| **Speed** | Faster transfers, slight dequantization overhead | Moderate transfer improvement |
| **Best for** | Maximum speed, models that barely fit in RAM | When quality matters more than speed |

### Pre-quantized vs. on-the-fly

Previous versions of Chiquito quantized on the fly from fp16 weights on every forward pass. The current approach pre-quantizes once and caches the result:

- **Smaller files on disk and in RAM** — the quantized split stores packed data (~4x smaller for 4-bit), so `preload_to_ram=True` uses much less memory.
- **Faster CPU→GPU transfers** — packed weights are smaller, reducing PCIe bandwidth usage.
- **No per-token quantization overhead** — weights arrive on the GPU already quantized.
- **Works with any fp16 HuggingFace model** — no need to find a pre-quantized variant.
