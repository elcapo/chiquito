# RAM Preloading and the Sliding Window

## Three loading modes

The `preload_to_ram` parameter controls how layer weights are loaded during inference:

### `preload_to_ram=True` — Full preload

At initialization, all per-layer safetensors files are loaded into a Python dict (`self._ram_cache`) as CPU tensors. If CUDA is available, tensors are [pinned](https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers) via `tensor.pin_memory()`, which enables faster DMA transfers to GPU.

During inference, `_load_layer_to_cpu()` is a dict lookup — effectively instant.

**Trade-off**: Uses RAM equal to the full model size (e.g., ~14 GB for a 7B fp16 model). Not viable when the model exceeds available RAM.

### `preload_to_ram=False` — Disk-only

No preloading. Each `_load_layer_to_cpu()` call reads from disk via `safetensors.torch.load_file()`. A `ThreadPoolExecutor` prefetches the next layer in a background thread while the current layer executes on GPU, overlapping I/O with compute.

In practice, the OS [page cache](https://www.kernel.org/doc/html/latest/admin-guide/mm/concepts.html) often keeps recently-read files in memory, so "disk" reads may actually come from kernel-managed RAM. This is why the performance gap between `True` and `False` can be smaller than expected on machines with ample free memory.

**Trade-off**: Minimal RAM usage, but no pinned memory and no guarantee of cache hits.

### `preload_to_ram=N` (integer) — Sliding window

Keeps at most N layers in RAM at a time, using a producer-consumer pattern. This is the middle ground for models that don't fit entirely in RAM.

## The sliding window: `_SlidingWindowCache`

The `_SlidingWindowCache` class in `model.py` implements a bounded buffer with a background loader thread.

### Initialization

```python
cache = _SlidingWindowCache(layer_names, split_dir, window_size=10)
cache.start()
```

`start()` synchronously loads the first N layers (with a progress bar), then spawns a daemon thread that continues loading upcoming layers as slots free up.

### Runtime flow

```
Main thread                          Background thread
───────────                          ─────────────────
get("layer.0") → instant             (idle, buffer full)
  move to GPU, execute
release("layer.0") → frees slot      wakes up, loads layer.10
get("layer.1") → instant
  move to GPU, execute
release("layer.1") → frees slot      loads layer.11
...
```

### Synchronization

The cache uses a `threading.Condition` (which wraps a `Lock`) for coordination:

- **`get(layer_name)`** blocks (via `wait()`) until the requested layer is in the cache.
- **`release(layer_name)`** removes the layer and calls `notify_all()`, waking the background thread if it was waiting for a free slot.
- **The background thread** waits when the cache is full (`len(cache) >= window_size`), and loads the next layer when a slot opens.

This is a standard [bounded buffer](https://en.wikipedia.org/wiki/Producer%E2%80%93consumer_problem) pattern. The GPU never stalls as long as disk I/O for one layer completes within the time it takes the GPU to execute `window_size - 1` layers.

### Forward pass restart

Since `forward()` is called once per generated token, the sliding window cache must restart at the beginning of each forward call. The cache is stopped, recreated, and `start()` is called again to preload the first N layers.
