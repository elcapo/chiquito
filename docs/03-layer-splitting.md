# Layer Splitting

## Why split?

HuggingFace models are stored as one or more large [safetensors](https://huggingface.co/docs/safetensors) shard files (e.g., `model-00001-of-00004.safetensors`). Each shard contains weights from multiple layers mixed together. To load a single layer, we would need to open a shard, deserialize all of it, and extract the subset we need.

Chiquito pre-splits the checkpoint into one file per layer. This way, loading layer N means reading exactly one small file with no wasted I/O.

## How splitting works

The splitting logic lives in `splitter.py`. The entry point is `find_or_create_split()`:

1. **Resolve the model path** — either a local directory or a HuggingFace Hub download via `huggingface_hub.snapshot_download()`.
2. **Check for existing splits** — if all `.safetensors` + `.done` marker pairs exist in `chiquito_split/`, skip splitting.
3. **Parse the weight map** — `model.safetensors.index.json` maps each parameter name (e.g., `model.layers.5.self_attn.q_proj.weight`) to its shard file. For single-file models without an index, the entire file is loaded once.
4. **Extract and save per-layer** — for each layer name, collect all parameters with that prefix, save them as `{layer_name}.safetensors`, and write a `.done` marker.

## Layer naming

Layer names follow the HuggingFace model's attribute hierarchy:

| Layer | Name | Example file |
|-------|------|-------------|
| Embedding | `model.embed_tokens` | `model.embed_tokens.safetensors` |
| Transformer 0 | `model.layers.0` | `model.layers.0.safetensors` |
| Transformer N | `model.layers.N` | `model.layers.N.safetensors` |
| Final norm | `model.norm` | `model.norm.safetensors` |
| LM head | `lm_head` | `lm_head.safetensors` |

These names are defined in `ChiquitoModel.LAYER_NAMES` and can be overridden by subclasses for architectures with different naming conventions.

## Completion markers

Each layer file is accompanied by a `.done` marker (e.g., `model.layers.0.safetensors.done`). Splitting writes the safetensors file first, then touches the marker. If the process is interrupted, incomplete layers (missing marker) will be re-split on the next run.

## Pre-quantized splits

When a quantization level is requested (e.g. `quantization="4bit"`), the splitter creates an additional directory alongside the base fp16 split:

| Quantization | Directory |
|---|---|
| None (fp16) | `chiquito_split/` |
| 4-bit | `chiquito_split_4bit/` |
| 8-bit | `chiquito_split_8bit/` |

The quantized split is built from the base fp16 split: each layer's fp16 weights are quantized using bitsandbytes and saved to the quantized directory. Only decoder layers (`model.layers.*`) are quantized — non-decoder layers (embedding, norm, lm_head) are kept in fp16 as they are small. See [Quantization](06-quantization.md) for details.

## Disk usage

The base fp16 split files are roughly the same size as the original model. For a 7B fp16 model, that's ~14 GB of split files alongside the ~14 GB original. Quantized splits are smaller (~4x for 4-bit, ~2x for 8-bit). The original model files can be deleted manually if disk space is tight.
