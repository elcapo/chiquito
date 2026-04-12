# Splitting Checkpoints into Per-Layer Files

Now we start building. The first practical piece of Chiquito is the **splitter**: code that takes a HuggingFace model checkpoint and produces one safetensors file per layer. It also handles pre-quantizing those layers when a quantization level is requested. This is implemented in [`splitter.py`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/splitter.py).

## Why we need to split

As we saw in [How HuggingFace Stores Models](03-huggingface-models.md), HuggingFace models are stored in shard files that mix parameters from many layers. A single shard might contain weights from layers 0-7. To load just layer 5, we would have to deserialize the entire shard and discard the rest.

We want the opposite: one file per layer, so that loading layer 5 means reading only layer 5's ~500 MB file.

## The target structure

Given a model at `/path/to/model/`, we create a `chiquito_split/` subdirectory with per-layer files. When a quantization level is requested, we additionally create a `chiquito_split_{4bit,8bit}/` directory with pre-quantized weights:

```
/path/to/model/
├── model.safetensors.index.json
├── model-00001-of-00004.safetensors
├── ...
├── chiquito_split/                          # base fp16 split
│   ├── model.embed_tokens.safetensors
│   ├── model.layers.0.safetensors
│   ├── model.layers.1.safetensors
│   ├── ...
│   ├── model.layers.21.safetensors
│   ├── model.norm.safetensors
│   ├── lm_head.safetensors
│   ├── model.embed_tokens.safetensors.done  # completion markers
│   ├── model.layers.0.safetensors.done
│   └── ...
├── chiquito_split_4bit/                     # pre-quantized (4-bit)
│   ├── model.embed_tokens.safetensors
│   ├── model.layers.0.safetensors
│   ├── ...
│   └── lm_head.safetensors
└── chiquito_split_8bit/                     # pre-quantized (8-bit)
    └── ...
```

The split directory name is computed by `split_dir_name()`:

```python
def split_dir_name(quantization: str | None = None) -> str:
    if quantization:
        return f"{SPLIT_DIR_NAME}_{quantization}"
    return SPLIT_DIR_NAME
```

Each per-layer file contains only the parameters whose names start with that layer's prefix. For example, `model.layers.5.safetensors` contains keys like `model.layers.5.self_attn.q_proj.weight`, `model.layers.5.mlp.gate_proj.weight`, etc.

## File paths and completion markers

The first thing we need are helper functions for computing file paths ([`splitter.py:13-22`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/splitter.py#L13-L22)):

```python
def layer_file_path(split_dir: Path, layer_name: str) -> Path:
    return split_dir / (layer_name + ".safetensors")

def done_marker_path(split_dir: Path, layer_name: str) -> Path:
    return split_dir / (layer_name + ".safetensors.done")

def is_layer_split(split_dir: Path, layer_name: str) -> bool:
    return (layer_file_path(split_dir, layer_name).exists() and
            done_marker_path(split_dir, layer_name).exists())
```

The `.done` marker is a crash-safety mechanism. If the process is interrupted while writing `model.layers.5.safetensors`, the file might be incomplete. The marker is written only after the file is fully saved, so on restart we know which layers need to be re-split.

## The splitting algorithm

The core function is `split_and_save_layers()` ([`splitter.py:25-111`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/splitter.py#L25-L111)). It handles two cases:

### Case 1: Single-file model

When the model has a single `model.safetensors` file (no index), we load it entirely and extract each layer:

```python
if weight_map is None:
    state_dict = load_file(str(single_file), device="cpu")
    for layer_name in tqdm(layer_names, desc="Splitting layers"):
        if is_layer_split(split_dir, layer_name):
            continue
        prefix = layer_name + "."
        layer_state = {k: v for k, v in state_dict.items()
                       if k.startswith(prefix)}
        if layer_state:
            save_safetensors(layer_state, layer_file_path(split_dir, layer_name))
            done_marker_path(split_dir, layer_name).touch()
    del state_dict
    clean_memory()
```

This is straightforward but uses a lot of RAM (the entire model must fit in memory at once). For small models this is fine.

You might wonder: if we have to load the entire file to split it, what's the point? The key is that **splitting is a one-time preparation step, not part of inference**. You pay the full-memory cost once, during the split. After that, every inference run loads only one layer at a time (~500 MB instead of ~14 GB), and thanks to the `.done` markers the split is never repeated.

### Case 2: Multi-shard model

For sharded models, we work incrementally to minimize RAM usage ([`splitter.py:70-111`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/splitter.py#L70-L111)):

```python
loaded_shards: set[str] = set()
state_dict: dict = {}

for layer_name in tqdm(layer_names, desc="Splitting layers"):
    if is_layer_split(split_dir, layer_name):
        continue

    prefix = layer_name + "."
    # Find which shards contain this layer's parameters
    needed_shards = {
        shard for param, shard in weight_map.items()
        if param.startswith(prefix)
    }

    # Load any shards we haven't loaded yet
    for shard in needed_shards:
        if shard not in loaded_shards:
            state_dict.update(load_file(str(model_path / shard), device="cpu"))
            loaded_shards.add(shard)

    # Extract this layer's weights
    layer_state = {k: v for k, v in state_dict.items()
                   if k.startswith(prefix)}
    if layer_state:
        save_safetensors(layer_state, layer_file_path(split_dir, layer_name))
        done_marker_path(split_dir, layer_name).touch()

        # Free extracted weights from state_dict
        for k in layer_state:
            del state_dict[k]
        del layer_state
        clean_memory()
```

The key optimization is that we **accumulate shard data in `state_dict`** and delete extracted parameters as we go. Since layers are ordered and shards tend to contain contiguous layers, we naturally work through the shards front to back without holding the entire model in memory.

Note the lazy shard downloading ([`splitter.py:89-95`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/splitter.py#L89-L95)): if a shard file does not exist locally (because the user only downloaded the index), we download just that shard on demand:

```python
if not shard_path.exists() and repo_id:
    import huggingface_hub
    huggingface_hub.snapshot_download(
        repo_id, allow_patterns=[shard], token=hf_token,
    )
```

## Early exit

Before doing any work, the function checks if all layers have already been split ([`splitter.py:33-36`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/splitter.py#L33-L36)):

```python
if split_dir.exists() and all(is_layer_split(split_dir, name)
                               for name in layer_names):
    print(f"All layers already split in {split_dir}")
    return split_dir
```

This means subsequent runs of Chiquito on the same model skip the splitting step entirely.

## The entry point: find_or_create_split

The function called by `ChiquitoModel.__init__` is `find_or_create_split()` ([`splitter.py:241-259`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/splitter.py#L241-L259)):

```python
def find_or_create_split(model_id_or_path, layer_names, hf_token=None, quantization=None):
    model_path = resolve_model_path(model_id_or_path, hf_token)
    repo_id = None if Path(model_id_or_path).is_dir() else model_id_or_path
    split_dir = split_and_save_layers(
        model_path, layer_names, hf_token=hf_token, repo_id=repo_id,
        quantization=quantization,
    )
    return model_path, split_dir
```

It resolves the model path (downloading from HuggingFace if needed), determines the `repo_id` for lazy shard downloading, runs the split (and quantization, if requested), and returns both the model path and the split directory.

When `quantization` is provided, `split_and_save_layers()` first ensures the base fp16 split exists, then quantizes each layer and saves the result to the quantized directory (e.g., `chiquito_split_4bit/`). See [Pre-Quantized Weight Caching](10-quantization.md) for details.

This is called at [`model.py:139-145`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L139-L145):

```python
self._model_path, self._split_dir = find_or_create_split(
    model_id_or_path, all_layer_names, hf_token=hf_token,
    quantization=quantization,
)
```

## Summary

The splitter converts HuggingFace's multi-shard checkpoint into per-layer files that Chiquito can load individually. It handles single-file and sharded models, uses completion markers for crash recovery, and caches splits so subsequent runs are instant. When a quantization level is requested, it also pre-quantizes each layer and stores the result in a dedicated directory — see [Pre-Quantized Weight Caching](10-quantization.md) for the full quantization story.

After splitting, our model directory contains one `.safetensors` file per layer, and the forward pass can load them one at a time — which is exactly what we build next.
