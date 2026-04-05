# Extensibility: AutoModel and the Registry

Everything we have built so far assumes Llama-style naming: `model.embed_tokens`, `model.layers.N`, `model.norm`, `lm_head`. But not all models follow this convention. ChatGLM uses `transformer.encoder.layers`, Phi uses `model.layers` but with a different position embedding setup, and so on.

Chiquito handles this with two mechanisms: a **factory pattern with registry** for architecture detection, and **override points** for customizing layer execution. The factory is in [`auto_model.py`](../src/chiquito/auto_model.py) (~40 lines), and the override points are in [`model.py`](../src/chiquito/model.py).

## The LAYER_NAMES dict

Each `ChiquitoModel` subclass defines a class variable mapping logical block names to the actual module paths used by that architecture ([`model.py:99-104`](../src/chiquito/model.py#L99-L104)):

```python
class ChiquitoModel(GenerationMixin):
    LAYER_NAMES = {
        "embed": "model.embed_tokens",
        "layer_prefix": "model.layers",
        "norm": "model.norm",
        "lm_head": "lm_head",
    }
```

These four keys are the only thing that varies between most architectures. A ChatGLM subclass might look like:

```python
class ChiquitoChatGLMModel(ChiquitoModel):
    LAYER_NAMES = {
        "embed": "transformer.embedding.word_embeddings",
        "layer_prefix": "transformer.encoder.layers",
        "norm": "transformer.encoder.final_layernorm",
        "lm_head": "transformer.output_layer",
    }
```

The entire initialization and forward pass logic uses these names through `self.LAYER_NAMES`, so changing the dict is enough to support a new architecture's naming.

## The AutoModel factory

The `AutoModel` class provides a `from_pretrained()` method that automatically selects the right `ChiquitoModel` subclass ([`auto_model.py:11-41`](../src/chiquito/auto_model.py#L11-L41)):

```python
class AutoModel:
    _REGISTRY: dict[str, type[ChiquitoModel]] = {}

    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated using "
            "AutoModel.from_pretrained(pretrained_model_name_or_path)"
        )

    @classmethod
    def register(cls, arch_name: str, model_class: type[ChiquitoModel]):
        cls._REGISTRY[arch_name] = model_class

    @classmethod
    def from_pretrained(cls, model_id_or_path: str, **kwargs) -> ChiquitoModel:
        from .model import ChiquitoModel

        hf_token = kwargs.get("hf_token")
        token_kwargs = {"token": hf_token} if hf_token else {}
        config = AutoConfig.from_pretrained(
            model_id_or_path, trust_remote_code=True, **token_kwargs
        )

        arch = config.architectures[0] if getattr(config, "architectures", None) else ""

        for key, model_cls in cls._REGISTRY.items():
            if key in arch:
                return model_cls(model_id_or_path, **kwargs)

        return ChiquitoModel(model_id_or_path, **kwargs)
```

The logic:
1. Load the model's config (lightweight — just `config.json`)
2. Extract the architecture name (e.g., `"LlamaForCausalLM"`, `"ChatGLMModel"`)
3. Check if any registered key is a substring of the architecture name
4. If a match is found, instantiate that subclass; otherwise, fall back to the base `ChiquitoModel`

### Registering a custom architecture

```python
from chiquito import AutoModel

AutoModel.register("ChatGLM", ChiquitoChatGLMModel)

# Now this automatically uses ChiquitoChatGLMModel
model = AutoModel.from_pretrained("THUDM/chatglm3-6b")
```

The substring matching (`"ChatGLM" in "ChatGLMModel"`) is intentionally loose: it catches variations like `ChatGLMForCausalLM`, `ChatGLMModel`, etc.

## Override points for layer execution

Sometimes different `LAYER_NAMES` are not enough — some architectures have different layer signatures or different position embedding schemes. Chiquito provides four override methods:

### `_compute_position_embeddings`

Computes rotary position embeddings after the embedding layer ([`model.py:340-346`](../src/chiquito/model.py#L340-L346)):

```python
def _compute_position_embeddings(self, hidden_states, position_ids):
    rotary_emb = getattr(getattr(self.model, "model", None), "rotary_emb", None)
    if rotary_emb is not None:
        return rotary_emb(hidden_states, position_ids=position_ids)
    return None
```

Override this if the model's position embedding module is at a different path or uses different arguments.

### `_run_transformer_layer`

Calls a single transformer layer ([`model.py:348-368`](../src/chiquito/model.py#L348-L368)):

```python
def _run_transformer_layer(self, layer, hidden_states, attention_mask,
                            position_ids, position_embeddings=None,
                            past_key_values=None, use_cache=False):
    kwargs = {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": use_cache,
    }
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    out = layer(hidden_states, **kwargs)
    return out[0] if isinstance(out, tuple) else out
```

Override this if the model's transformer layers expect different keyword arguments.

### `_run_norm` and `_run_lm_head`

Simple wrappers ([`model.py:370-374`](../src/chiquito/model.py#L370-L374)):

```python
def _run_norm(self, layer, hidden_states):
    return layer(hidden_states)

def _run_lm_head(self, layer, hidden_states):
    return layer(hidden_states).float()
```

Override if the norm or LM head layers need special handling (e.g., additional arguments or different output processing).

## Putting it together: a complete custom architecture

```python
from chiquito import AutoModel
from chiquito.model import ChiquitoModel

class ChiquitoChatGLMModel(ChiquitoModel):
    LAYER_NAMES = {
        "embed": "transformer.embedding.word_embeddings",
        "layer_prefix": "transformer.encoder.layers",
        "norm": "transformer.encoder.final_layernorm",
        "lm_head": "transformer.output_layer",
    }

    def _compute_position_embeddings(self, hidden_states, position_ids):
        rotary_emb = self.model.transformer.rotary_pos_emb
        return rotary_emb(max_seq_len=self.max_seq_len)

    def _run_transformer_layer(self, layer, hidden_states, attention_mask,
                                position_ids, position_embeddings=None,
                                past_key_values=None, use_cache=False):
        out = layer(hidden_states, attention_mask=attention_mask,
                    rotary_pos_emb=position_embeddings,
                    kv_cache=past_key_values, use_cache=use_cache)
        return out[0]

AutoModel.register("ChatGLM", ChiquitoChatGLMModel)

# Now works automatically
model = AutoModel.from_pretrained("THUDM/chatglm3-6b")
```

## Why this design works

The base `ChiquitoModel` handles all the complex logic: checkpoint splitting, meta device initialization, the load-execute-free cycle, KV cache management, sliding window, quantization. Architecture-specific code only needs to specify:

1. **Where things are** (the `LAYER_NAMES` dict)
2. **How to call them** (the override methods)

This keeps architecture plugins small (typically under 30 lines) while reusing all the infrastructure.

## The default covers most models

Many modern architectures follow the Llama naming convention: Llama itself, Mistral, Mixtral, Qwen2, DeepSeek, CodeLlama, and others. These all work with the base `ChiquitoModel` without any registration. The registry is only needed for architectures that deviate from this convention.

## Summary

| Mechanism | Purpose | Where |
|-----------|---------|-------|
| `LAYER_NAMES` dict | Map logical blocks to module paths | [`model.py:99-104`](../src/chiquito/model.py#L99-L104) |
| `AutoModel._REGISTRY` | Map architecture names to classes | [`auto_model.py:12`](../src/chiquito/auto_model.py#L12) |
| `AutoModel.from_pretrained()` | Auto-detect architecture and instantiate | [`auto_model.py:25-40`](../src/chiquito/auto_model.py#L25-L40) |
| Override methods | Customize layer execution | [`model.py:340-374`](../src/chiquito/model.py#L340-L374) |
