# The Problem: VRAM as a Bottleneck

## Why large models don't fit on consumer GPUs

A transformer model's memory footprint is dominated by its weights. A 7B-parameter model in fp16 uses ~14 GB just for weights, and a 32B model uses ~65 GB. Consumer GPUs typically have 8-24 GB of VRAM, so models above ~4B parameters cannot be loaded entirely.

The standard approach, quantization, reduces precision (for instance, to 4-bit) to shrink the footprint. But even quantized, a 32B model needs ~16 GB, which is tight for an 8 GB card.

## The layer by layer approach

A transformer model is a sequence of identical layers. During a forward pass, data flows through them one at a time: layer 0 finishes before layer 1 starts. This means we never need more than one layer's weights in VRAM simultaneously.

A single transformer layer of a 7B model weighs ~500 MB. A single layer of a 32B model weighs ~1 GB. Both fit comfortably in 8 GB of VRAM.

The idea:

```python
# Don't take this literally
for layer in layers:
    load_to_gpu(layer.weights)
    run(layer)
    free(memory)
```

This is the core insight behind [AirLLM](https://github.com/lyogavin/airllm) and Chiquito. The trade-off is speed: loading weights from storage for every layer on every token is slow. Chiquito mitigates this by keeping weights in system RAM (which is much larger than VRAM on most machines) and using [pinned memory](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/) for faster CPU to GPU transfers.

## Where time goes

For a model with N layers generating T tokens, the layer by layer approach processes N layers per token. With KV cache, the first token (prefill) processes the full input sequence, while subsequent tokens process a single token each. But the weight loading happens on every token regardless:

| Step | Layer loads | Compute per layer |
|------|-----------|-------------------|
| Prefill (1st token) | N | Full input sequence |
| Decode (each subsequent token) | N | 1 token (KV cache reuses prior computation) |

The weight loading time (CPU to GPU transfer) tends to dominate over the compute time, especially for decode steps where the actual matrix multiplications are small. This is why optimizing the loading path matters.
