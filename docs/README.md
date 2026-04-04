# Chiquito Developer Documentation

This documentation explains the concepts behind Chiquito and how the code implements them. It is intended for developers who want to understand, modify, or extend the library.

## Contents

1. [The Problem: VRAM as a Bottleneck](01-the-problem.md): Why layer by layer inference exists and what trade-offs it makes.
2. [Architecture Overview](02-architecture.md): How the code is organized and how data flows through it.
3. [Layer Splitting](03-layer-splitting.md): How HuggingFace checkpoints are broken into per-layer files.
4. [RAM Preloading and the Sliding Window](04-ram-preloading.md): The three loading modes and the producer-consumer pattern behind the sliding window.
5. [The Forward Pass](05-forward-pass.md): Layer by layer inference, KV cache, and the interaction with `GenerationMixin`.
6. [Quantization](06-quantization.md): On-the-fly 4-bit and 8-bit quantization via bitsandbytes.
7. [Extending Chiquito](07-extending.md): How to add support for new model architectures.
