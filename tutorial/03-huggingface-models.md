# How HuggingFace Stores and Distributes Models

HuggingFace is the de facto standard for distributing open-source LLMs. Chiquito relies heavily on HuggingFace conventions: model configs, tokenizers, weight files, and the Hub download infrastructure. This unit explains what we need to know to work with them.

## `AutoConfig`: the model's blueprint

Every HuggingFace model has a `config.json` file that describes its architecture without containing any weights. You can load it with `AutoConfig`:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(config.num_hidden_layers)   # 22
print(config.hidden_size)         # 2048
print(config.vocab_size)          # 32000
print(config.architectures)       # ["LlamaForCausalLM"]
```

This is lightweight — it downloads only `config.json`, not the weights. Chiquito uses the config at two critical points:

1. **Counting layers** to build the layer name list ([`model.py:167-189`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L167-L189)):
   ```python
   def _build_layer_name_list_from_config(self, model_id_or_path, hf_token):
       config = AutoConfig.from_pretrained(model_id_or_path, ...)
       with init_empty_weights():
           model = AutoModelForCausalLM.from_config(config, ...)
       # Count layers by accessing the module tree
       module = model
       for attr in self.LAYER_NAMES["layer_prefix"].split("."):
           module = getattr(module, attr)
       n_layers = len(module)
   ```

2. **Detecting architecture** in the AutoModel factory ([`auto_model.py:34`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/auto_model.py#L34)):
   ```python
   arch = config.architectures[0]  # e.g., "LlamaForCausalLM"
   ```

## `AutoTokenizer`: text to numbers and back

Tokenizers convert text into sequences of integer IDs that the model can process, and decode the output IDs back into text:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Encode
tokens = tokenizer("Hello, world!", return_tensors="pt")
print(tokens["input_ids"])   # tensor([[1, 15043, 29892, 3186, 29991]])

# Decode
text = tokenizer.decode(tokens["input_ids"][0])
print(text)                  # "Hello, world!"
```

Chiquito loads the tokenizer once during initialization ([`model.py:150-152`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L150-L152)) and exposes it as `model.tokenizer` for the user to encode prompts and decode outputs.

## The safetensors format

Weights are stored in **safetensors** files; a simple, fast, memory-mappable binary format. Each file is a flat mapping from parameter name to tensor:

```python
from safetensors.torch import load_file, save_file

# Load
state_dict = load_file("model.safetensors", device="cpu")
# {"model.layers.0.self_attn.q_proj.weight": tensor(...), ...}

# Save
save_file(state_dict, "output.safetensors")
```

Chiquito wraps these in small utility functions ([`utils.py:25-30`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/utils.py#L25-L30)):

```python
def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path), device="cpu")

def save_safetensors(state_dict: dict[str, torch.Tensor], path: Path):
    save_file(state_dict, str(path))
```

## Sharded models and the weight map

Small models may have a single `model.safetensors` file. Larger models are **sharded**: split across multiple files like `model-00001-of-00004.safetensors`, `model-00002-of-00004.safetensors`, etc.

When a model is sharded, there is an index file called `model.safetensors.index.json` that maps every parameter name to the shard file that contains it:

```json
{
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
    "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00004.safetensors",
    ...
    "model.layers.31.mlp.up_proj.weight": "model-00004-of-00004.safetensors",
    "lm_head.weight": "model-00004-of-00004.safetensors"
  }
}
```

This weight map is what makes checkpoint splitting possible: we can look up which shard contains a given layer's parameters without loading everything.

Chiquito parses this map in [`splitter.py:44-53`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/splitter.py#L44-L53):

```python
index_path = model_path / "model.safetensors.index.json"
single_file = model_path / "model.safetensors"

if index_path.exists():
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
elif single_file.exists():
    weight_map = None   # single-file model, no map needed
else:
    raise FileNotFoundError(...)
```

## Downloading models from the Hub

HuggingFace provides `snapshot_download()` to download model files to a local cache:

```python
import huggingface_hub

cache_path = huggingface_hub.snapshot_download(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    token=None,                   # set if the model is gated
    ignore_patterns=["*.bin"],    # skip old-format weight files
)
```

This returns the local path where the files were cached. Subsequent calls with the same model ID return the cached path without re-downloading.

Chiquito's `resolve_model_path()` function ([`utils.py:33-45`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/utils.py#L33-L45)) wraps this logic: if the path is a local directory with model files, use it directly; otherwise, download from the Hub:

```python
def resolve_model_path(model_id_or_path: str, hf_token=None) -> Path:
    path = Path(model_id_or_path)
    if path.is_dir():
        if (path / "model.safetensors.index.json").exists() or \
           (path / "model.safetensors").exists():
            return path

    cache_path = huggingface_hub.snapshot_download(
        model_id_or_path, token=hf_token, ignore_patterns=["*.bin"],
    )
    return Path(cache_path)
```

## Summary

| Concept | Library | What Chiquito uses it for |
|---------|---------|--------------------------|
| `AutoConfig` | transformers | Read layer count, architecture name |
| `AutoTokenizer` | transformers | Encode/decode text |
| safetensors | safetensors | Load/save weight files |
| Weight map | JSON file | Find which shard has which parameters |
| `snapshot_download` | huggingface_hub | Download models to local cache |

With this knowledge, we can now look at how a transformer model is structured internally.
