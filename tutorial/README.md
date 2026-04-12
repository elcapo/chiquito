# Chiquito Tutorial

A step-by-step tutorial that teaches every concept needed to understand and recreate the Chiquito source code (~980 lines of Python).

## Prerequisites

- Python proficiency
- Know what an LLM is (a neural network organized as layers with weights)
- Know what inference is (feeding input through the model to get output)

Everything else is explained from scratch.

## Contents

### Foundations (concepts)

1. [The Problem: VRAM as a Bottleneck](01-the-problem.md) — Why large models don't fit on consumer GPUs and how layer-by-layer inference solves it.
2. [PyTorch Foundations](02-pytorch-foundations.md) — Tensors, devices, `nn.Module`, `state_dict`, the meta device, and `set_module_tensor_to_device`.
3. [How HuggingFace Stores Models](03-huggingface-models.md) — `AutoConfig`, `AutoTokenizer`, safetensors, weight maps, and `snapshot_download`.
4. [Transformer Architecture](04-transformer-architecture.md) — The four blocks (embedding, transformer layers, norm, lm_head) and how to navigate them.

### Building the engine

5. [Splitting Checkpoints](05-checkpoint-splitting.md) — Parsing weight maps and creating per-layer safetensors files. Covers `splitter.py`.
6. [The Forward Pass](06-forward-pass.md) — The load-execute-free cycle, causal attention masks, and position IDs. Covers the core of `model.py`.
7. [KV Cache and Generation](07-kv-cache-and-generation.md) — Avoiding redundant computation, `DynamicCache`, and `GenerationMixin` integration.

### Optimizations

8. [Memory Management](08-memory-management.md) — `gc.collect`, `malloc_trim`, `empty_cache`, and pinned memory. Covers `utils.py`.
9. [RAM Preloading and Sliding Window](09-ram-preloading.md) — Three loading strategies and the producer-consumer pattern.
10. [Quantization](10-quantization.md) — Pre-quantized 4-bit/8-bit weight caching with bitsandbytes.

### Extensibility and integration

11. [Architecture Extensibility](11-extensibility.md) — The `AutoModel` factory, `LAYER_NAMES`, and override points. Covers `auto_model.py`.
12. [Putting It All Together](12-putting-it-together.md) — End-to-end flow, the dependency graph, and design decisions.
