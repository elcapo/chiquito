# RAM Preloading and the Sliding Window Cache

So far, our forward pass loads each layer from disk on every call. For 67 layers x 20 tokens = 1,340 disk reads, this is painfully slow. This unit introduces the three loading strategies that trade RAM for speed, implemented in [`model.py`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py).

## The three modes

Chiquito's `preload_to_ram` parameter controls where weights live between forward calls:

| `preload_to_ram` | Weights live in | RAM usage | Speed |
|-----------------|----------------|-----------|-------|
| `True` | System RAM (pinned) | Full model | Fastest |
| `False` | Disk (read on demand) | Minimal | Slowest |
| Integer N | System RAM (N layers) | N layers | Fast |

The constructor interprets this parameter at [`model.py:129-135`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L129-L135):

```python
if preload_to_ram is True:
    self._window_size = None  # load all
elif preload_to_ram is False:
    self._window_size = 0     # disk only
else:
    self._window_size = int(preload_to_ram)
```

## Mode 1: Full preload (`preload_to_ram=True`)

At initialization, load every layer's weights into RAM and pin them for fast DMA transfers. We saw this in [Unit 08](08-memory-management.md):

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

([`model.py:294-301`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L294-L301))

During the forward pass, `_load_layer_to_cpu()` is just a dictionary lookup ([`model.py:311-313`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L311-L313)):

```python
def _load_layer_to_cpu(self, layer_name):
    if self._ram_cache is not None:
        return self._ram_cache[layer_name]
```

This is the fastest mode because:
1. No disk I/O during inference
2. Pinned memory enables DMA transfers (2-5x faster than pageable)

The limitation: the entire model must fit in RAM. A 32B model at fp16 needs ~65 GB of RAM.

## Mode 2: Disk-only (`preload_to_ram=False`)

When RAM is extremely limited, weights are read from disk on every layer load. `_load_layer_to_cpu()` falls through to the disk path ([`model.py:316`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L316)):

```python
return load_safetensors(layer_file_path(self._split_dir, layer_name))
```

This is the slowest mode but requires almost no RAM (just enough for one layer at a time).

### Prefetching with ThreadPoolExecutor

Even in disk-only mode, we can overlap disk reads with GPU computation. While the GPU is executing layer N, a background thread reads layer N+1 from disk.

The forward pass sets this up at [`model.py:468-475`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L468-L475):

```python
use_executor = self._prefetch and self._window_cache is None

with torch.inference_mode():
    executor = ThreadPoolExecutor(max_workers=1) if use_executor else None
    future = None

    if executor:
        future = executor.submit(self._load_layer_to_cpu, self.layer_names[0])
```

And inside the loop ([`model.py:484-491`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L484-L491)):

```python
if executor and future:
    state_dict = future.result()           # wait for current layer
    if i + 1 < len(self.layer_names):
        future = executor.submit(          # start loading next layer
            self._load_layer_to_cpu, self.layer_names[i + 1]
        )
else:
    state_dict = self._load_layer_to_cpu(layer_name)
```

This simple one-step-ahead prefetch can hide most of the disk I/O latency if the GPU computation takes longer than the read.

## Mode 3: Sliding window (`preload_to_ram=N`)

The sweet spot between full preload and disk-only. Keep N layers in RAM at any time, using a background thread to load upcoming layers as slots free up.

This is a classic **producer-consumer** pattern implemented in the `_SlidingWindowCache` class ([`model.py:27-94`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L27-L94)).

### The data structure

```python
class _SlidingWindowCache:
    def __init__(self, layer_names, split_dir, window_size):
        self._layer_names = layer_names
        self._split_dir = split_dir
        self._window_size = window_size
        self._cache = {}                        # {layer_name: state_dict}
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._loader_thread = None
        self._next_to_load = 0
        self._stop = False
```

The cache is a dictionary that never holds more than `window_size` entries. Synchronization uses a `threading.Condition`, which combines a lock with a wait/notify mechanism.

### Startup: preload the first N layers

Before the background thread starts, the first N layers are loaded synchronously ([`model.py:47-58`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L47-L58)):

```python
def start(self):
    initial_count = min(self._window_size, len(self._layer_names))
    for i in tqdm(range(initial_count), desc="Preloading window to RAM"):
        name = self._layer_names[i]
        self._cache[name] = load_safetensors(
            layer_file_path(self._split_dir, name)
        )
    self._next_to_load = initial_count
    self._loader_thread = threading.Thread(
        target=self._background_loader, daemon=True
    )
    self._loader_thread.start()
```

This guarantees that when the forward pass starts, the first N layers are already available.

### The consumer: `get()` and `release()`

The forward pass calls `get()` to retrieve a layer's weights and `release()` when done:

```python
def get(self, layer_name):
    with self._ready:
        while layer_name not in self._cache:
            self._ready.wait()       # block until the layer is loaded
        return self._cache[layer_name]

def release(self, layer_name):
    with self._ready:
        self._cache.pop(layer_name, None)   # free the slot
        self._ready.notify_all()             # wake the producer
```

([`model.py:60-69`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L60-L69))

`get()` blocks if the requested layer is not yet in the cache. `release()` removes the layer and wakes the background thread, which may now have room to load the next layer.

### The producer: background loader

```python
def _background_loader(self):
    while True:
        with self._ready:
            if self._stop or self._next_to_load >= len(self._layer_names):
                return
            while len(self._cache) >= self._window_size:
                if self._stop:
                    return
                self._ready.wait()       # wait for a free slot
            name = self._layer_names[self._next_to_load]
            self._next_to_load += 1

        # Load outside the lock (disk I/O is slow)
        data = load_safetensors(layer_file_path(self._split_dir, name))

        with self._ready:
            self._cache[name] = data
            self._ready.notify_all()     # wake get() if it's waiting
```

([`model.py:78-94`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L78-L94))

Key points:
- The disk read (`load_safetensors`) happens **outside the lock** — this is critical for performance, otherwise the consumer would block on the lock while waiting for I/O
- The thread checks two exit conditions: `_stop` flag or all layers loaded
- When the cache is full, the thread waits for a `release()` to free a slot

### Cache restart

Because `forward()` is called once per token and iterates through all layers, the sliding window cache must be restarted for each forward call ([`model.py:437-438`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L437-L438)):

```python
if self._window_size is not None and self._window_size > 0:
    self._start_window_cache()
```

`_start_window_cache()` stops the old cache and creates a fresh one ([`model.py:303-309`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L303-L309)):

```python
def _start_window_cache(self):
    if self._window_cache is not None:
        self._window_cache.stop()
    self._window_cache = _SlidingWindowCache(
        self.layer_names, self._split_dir, self._window_size
    )
    self._window_cache.start()
```

### Stopping

```python
def stop(self):
    with self._ready:
        self._stop = True
        self._ready.notify_all()
    if self._loader_thread is not None:
        self._loader_thread.join(timeout=5)
```

([`model.py:71-76`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L71-L76))

Sets the stop flag, wakes the background thread so it can exit, and joins with a timeout.

## How the forward pass selects a loading strategy

The `_load_layer_to_cpu()` method dispatches based on which cache is active ([`model.py:311-316`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L311-L316)):

```python
def _load_layer_to_cpu(self, layer_name):
    if self._ram_cache is not None:
        return self._ram_cache[layer_name]        # Full preload
    if self._window_cache is not None:
        return self._window_cache.get(layer_name)  # Sliding window
    return load_safetensors(...)                    # Disk only
```

## Choosing the right window size

| If you have... | Use |
|----------------|-----|
| RAM ≥ model size | `preload_to_ram=True` — fastest, everything pinned |
| Some extra RAM | `preload_to_ram=N` — set N so that N layers fit in available RAM |
| Almost no extra RAM | `preload_to_ram=False` — slowest but works anywhere |

A good rule of thumb for the window size: if each layer is ~1 GB and you have 16 GB of free RAM, set `preload_to_ram=10`. The background thread will stay ahead of the forward pass unless disk I/O is extremely slow.

## Summary

The three loading modes trade RAM for speed. Full preload is fastest but requires the most RAM. Disk-only is slowest but works with minimal RAM. The sliding window is the middle ground, using a producer-consumer pattern to keep a bounded number of layers in RAM while a background thread loads ahead.
