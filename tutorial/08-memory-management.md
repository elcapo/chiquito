# Memory Management: gc, malloc_trim, and Pinned Memory

Layer-by-layer inference creates and destroys large tensors constantly. Without explicit memory management, GPU memory accumulates leaked allocations, Python's heap grows unboundedly, and CPU-to-GPU transfers are slower than they need to be.

This unit covers the memory utilities in [`utils.py`](../src/chiquito/utils.py) and the pinned memory technique used for fast preloading.

## The three levels of memory cleanup

### Level 1: `torch.cuda.empty_cache()` — lightweight GPU cleanup

PyTorch uses a caching allocator for GPU memory. When you delete a GPU tensor, the memory is not returned to CUDA — it stays in PyTorch's cache for reuse. This is normally fine, but in our case we load a new layer's weights on every iteration. The cached blocks from the previous layer are useless, and they prevent the new layer's weights from fitting.

`torch.cuda.empty_cache()` tells PyTorch to release all cached GPU memory blocks back to CUDA:

```python
def clean_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

([`utils.py:20-22`](../src/chiquito/utils.py#L20-L22))

This is **called after every layer** in the forward pass ([`model.py:519`](../src/chiquito/model.py#L519)) because it is cheap (microseconds) and essential to prevent GPU memory from growing.

### Level 2: `gc.collect()` — Python garbage collection

Python's garbage collector normally runs periodically, but it can leave reference cycles alive for a while. In our case, large tensor objects might linger after `del` due to circular references (e.g., autograd graphs, closures).

`gc.collect()` forces an immediate collection cycle:

```python
import gc
gc.collect()
```

This is heavier than `empty_cache` (it scans all Python objects), so we only use it during initialization and after large operations, not on every layer.

### Level 3: `malloc_trim()` — return heap memory to the OS

On Linux, when Python (via CPython's memory allocator, which uses glibc's `malloc`) frees memory, the freed space often stays in the process's heap. The OS still sees the process as using that memory (high RSS). Over time, this fragmentation can consume gigabytes of apparently "leaked" memory.

`malloc_trim(0)` tells glibc to return freed heap memory to the OS:

```python
import ctypes
ctypes.CDLL("libc.so.6").malloc_trim(0)
```

This is Linux-specific (it uses glibc directly) and is wrapped in a try/except for portability.

### The combined cleanup function

[`utils.py:10-17`](../src/chiquito/utils.py#L10-L17) combines all three:

```python
def clean_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

This is the "heavy" cleanup — used during model initialization, after splitting, and when reinitializing quantized models. It is too expensive to call on every layer but essential for preventing memory bloat during setup.

## When each function is used

| Function | Cost | When used | Where |
|---------|------|-----------|-------|
| `clean_gpu_memory()` | Microseconds | After every layer in forward pass | [`model.py:519`](../src/chiquito/model.py#L519) |
| `clean_memory()` | Milliseconds | During init, after splitting, quantized model reset | [`model.py:430`](../src/chiquito/model.py#L430), [`splitter.py:67`](../src/chiquito/splitter.py#L67) |

## Pinned memory: faster CPU-to-GPU transfers

When you copy a tensor from CPU to GPU, CUDA must first ensure the source memory is **page-locked** (pinned) — meaning the OS cannot swap it out to disk during the transfer. If the tensor is in regular (pageable) memory, CUDA copies it to an internal pinned buffer first, then transfers to GPU. This double-copy adds overhead.

If we pre-pin the memory ourselves, CUDA can transfer directly via DMA (Direct Memory Access) over PCIe, skipping the intermediate copy:

```
Regular memory:   CPU tensor → [CUDA pinned buffer] → GPU    (2 copies)
Pinned memory:    CPU tensor → GPU                            (1 copy, DMA)
```

PyTorch makes this easy:

```python
tensor = tensor.pin_memory()  # page-lock this tensor in RAM
```

Chiquito pins all tensors when preloading to RAM ([`model.py:294-301`](../src/chiquito/model.py#L294-L301)):

```python
def _preload_all_layers(self):
    self._ram_cache = {}
    pin = torch.cuda.is_available()
    for name in tqdm(self.layer_names, desc="Preloading layers to RAM"):
        state_dict = load_safetensors(layer_file_path(self._split_dir, name))
        if pin:
            state_dict = {k: v.pin_memory() for k, v in state_dict.items()}
        self._ram_cache[name] = state_dict
```

The result: CPU-to-GPU transfers for preloaded layers use DMA, which is 2-5x faster than pageable transfers for large tensors at PCIe 3.0/4.0 speeds.

## The cost of pinned memory

Pinned memory has a trade-off: it **cannot be swapped out**. If you pin 60 GB of tensors, your system must have 60 GB of physical RAM available. This is fine for the "full preload" mode where you have enough RAM, but it is why the sliding window and disk-only modes exist — for when RAM is limited.

## Summary

| Technique | What it does | Why it matters |
|-----------|-------------|----------------|
| `empty_cache()` | Release cached GPU blocks | Prevents GPU memory growth between layers |
| `gc.collect()` | Force Python GC cycle | Frees tensor reference cycles |
| `malloc_trim(0)` | Return heap to OS | Prevents RSS bloat on Linux |
| `pin_memory()` | Page-lock CPU tensors | 2-5x faster CPU→GPU DMA transfers |

These are small utilities, but they are essential for making layer-by-layer inference work in practice. Without `empty_cache`, the GPU runs out of memory after a few layers. Without `pin_memory`, preloaded weights transfer slowly. Without `gc.collect` and `malloc_trim`, initialization can leak gigabytes.
