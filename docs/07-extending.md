# Extending Chiquito

## Supporting new architectures

Chiquito's base `ChiquitoModel` works for any model that follows the Llama-style naming convention:

- `model.embed_tokens` — embedding layer
- `model.layers.N` — transformer layers
- `model.norm` — final normalization
- `lm_head` — output projection

This covers Llama, Mistral, Mixtral, Qwen2, and most modern causal LMs on HuggingFace.

For models with different naming (e.g., ChatGLM uses `transformer.embedding`, `transformer.encoder.layers`, etc.), create a subclass:

```python
from chiquito import ChiquitoModel, AutoModel

class ChiquitoChatGLMModel(ChiquitoModel):
    LAYER_NAMES = {
        "embed": "transformer.embedding.word_embeddings",
        "layer_prefix": "transformer.encoder.layers",
        "norm": "transformer.encoder.final_layernorm",
        "lm_head": "transformer.output_layer",
    }

# Register so AutoModel picks it up
AutoModel.register("ChatGLM", ChiquitoChatGLMModel)
```

After registration, `AutoModel.from_pretrained()` will check if the model's `config.architectures[0]` contains `"ChatGLM"` and use your subclass automatically.

## Override points

Beyond `LAYER_NAMES`, `ChiquitoModel` provides methods that subclasses can override:

### `_compute_position_embeddings(hidden_states, position_ids)`

Computes rotary position embeddings. The default implementation looks for `self.model.model.rotary_emb` and calls it. Override if your model uses a different rotary embedding location or format.

### `_run_transformer_layer(layer, hidden_states, attention_mask, position_ids, position_embeddings, past_key_values, use_cache)`

Calls a single transformer layer. Override if your model's layers expect different keyword arguments. For example, some models pass position embeddings differently or use a custom attention mask format.

### `_run_norm(layer, hidden_states)`

Runs the final normalization layer. Override if your model applies normalization differently (e.g., with additional arguments).

### `_run_lm_head(layer, hidden_states)`

Runs the language model head. The default calls `layer(hidden_states).float()`. The `.float()` ensures logits are in fp32 for numerical stability during sampling.

## Example: custom position embeddings

Some models compute position embeddings differently. For instance, a model that passes `position_ids` directly to each layer instead of precomputed cos/sin:

```python
class ChiquitoCustomModel(ChiquitoModel):

    def _compute_position_embeddings(self, hidden_states, position_ids):
        # This model doesn't use precomputed rotary embeddings
        return None

    def _run_transformer_layer(self, layer, hidden_states, attention_mask,
                                position_ids, position_embeddings=None,
                                past_key_values=None, use_cache=False):
        # Pass position_ids directly, no position_embeddings
        return layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )[0]
```
