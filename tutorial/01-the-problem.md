# The Problem: VRAM as a Bottleneck

## How much memory does a model need?

A neural network's memory footprint is dominated by its weights (the learned numbers). The memory requirement is straightforward arithmetic:

```
memory = number_of_parameters x bytes_per_parameter
```

Most models store weights in float16 (2 bytes per parameter). So:

| Model size | Parameters | Memory (fp16) |
|-----------|-----------|---------------|
| 1B | 1,000,000,000 | ~2 GB |
| 7B | 7,000,000,000 | ~14 GB |
| 32B | 32,000,000,000 | ~65 GB |
| 70B | 70,000,000,000 | ~140 GB |

A consumer GPU (NVIDIA RTX 4090) has 24 GB of VRAM. That means anything above ~12B parameters in fp16 simply does not fit.

## The standard solution: quantization

The usual approach is to reduce precision. Instead of storing each weight as a 16-bit float, you can use 8 bits or even 4 bits:

| Precision | Bytes/param | 32B model size |
|-----------|-----------|----------------|
| fp16 | 2 | ~65 GB |
| 8-bit | 1 | ~32 GB |
| 4-bit | 0.5 | ~16 GB |

This helps a lot, but even a 4-bit 32B model barely fits on a 24 GB card, and there is no room left for activations or KV cache. For truly large models, quantization alone is not enough.

## The key insight: transformers are sequential

An LLM (specifically, a transformer-based language model) is structured as a stack of layers. During inference, data flows through them one at a time:

```
input -> layer 0 -> layer 1 -> ... -> layer N -> output
```

Layer 0 must finish before layer 1 starts. Layer 1 must finish before layer 2 starts. At no point do we need two layers' weights in memory simultaneously.

A single layer of a 7B model weighs about 500 MB. A single layer of a 32B model weighs about 1 GB. Both fit comfortably in 8 GB of VRAM.

This is the core idea:

```python
# Pseudocode — don't take this literally
for layer in model.layers:
    load_weights_to_gpu(layer)
    output = layer(output)
    free_gpu_memory(layer)
```

Instead of loading the entire model, we load one layer at a time, run it, free the GPU memory, and move on to the next.

## The trade-off: speed

There is a catch. For every generated token, we must iterate through all N layers. A 32B model has 64 transformer layers (plus embedding, norm, and lm_head — roughly 67 layers total). Generating 20 tokens means 67 x 20 = 1,340 layer loads.

At PCIe 3.0 speeds, transferring a 1 GB layer from system RAM to GPU takes about 80 ms. That is:

```
67 layers x 20 tokens x 80 ms = ~107 seconds
```

The weight transfer completely dominates the actual computation. This is not fast by any standard, but it makes the impossible merely slow: you can run a 32B model on an 8 GB GPU.

## Where Chiquito fits

[AirLLM](https://github.com/lyogavin/airllm) was the first implementation of this idea. Chiquito is a rewrite that improves on it in one critical way: instead of reading weights from disk (NVMe) on every layer load, Chiquito preloads them into system RAM and uses **pinned memory** for faster CPU-to-GPU transfers via DMA (Direct Memory Access).

The three loading strategies (which we will build step by step in this tutorial) are:

| Mode | Where weights live | Speed | RAM required |
|------|-------------------|-------|-------------|
| Full preload | System RAM (pinned) | Fastest | Full model size |
| Sliding window | System RAM (N layers) | Fast | N layers worth |
| Disk only | NVMe/SSD | Slowest | Minimal |

## What we will build

Over the course of this tutorial, we will build the complete Chiquito library from scratch. By the end, you will have a working inference engine that can run models much larger than your GPU's VRAM.

The only prerequisites are:
- Python proficiency
- Understanding that an LLM is organized as layers with weights
- Understanding that inference means feeding input through the model to get output

Everything else — PyTorch mechanics, HuggingFace conventions, attention masks, KV cache — we will explain as we need it.
