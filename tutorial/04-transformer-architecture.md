# Transformer Architecture from the Outside

We do not need to understand the internal mathematics of self-attention or feed-forward networks to build Chiquito. What we need is a high-level understanding of the four logical components of a causal language model and how data flows through them.

## The four blocks

A causal language model (the kind used for text generation) has four sequential components:

```
token IDs  -->  [Embedding]  -->  [Transformer Layer] x N  -->  [Norm]  -->  [LM Head]  -->  logits
```

### 1. Embedding (`model.embed_tokens`)

Converts integer token IDs into dense vectors (embeddings). If the vocabulary has 32,000 tokens and the hidden dimension is 2048, the embedding layer is a lookup table of shape `(32000, 2048)`.

**Input**: tensor of token IDs, shape `(batch, seq_len)` — e.g., `(1, 10)` for a 10-token prompt
**Output**: tensor of embeddings, shape `(batch, seq_len, hidden_dim)` — e.g., `(1, 10, 2048)`

### 2. Transformer layers (`model.layers.0` through `model.layers.N-1`)

The core of the model. Each transformer layer takes a hidden state and produces a hidden state of the same shape. Internally, each layer contains:

- **Self-attention**: looks at all previous tokens to decide what information is relevant
- **Feed-forward network (FFN/MLP)**: processes each position independently through two linear layers with an activation function in between

For our purposes, each transformer layer is a black box:

**Input**: hidden states `(batch, seq_len, hidden_dim)` + attention mask + position information
**Output**: hidden states `(batch, seq_len, hidden_dim)`

A 7B model (like TinyLlama) has 22 layers. A 32B model has 64 layers. This is the repeated block that makes up the bulk of the model's parameters and the bulk of the loading time.

### 3. Final normalization (`model.norm`)

A normalization layer (typically RMSNorm) applied after the last transformer layer. It stabilizes the hidden states before the final projection.

**Input**: hidden states `(batch, seq_len, hidden_dim)`
**Output**: normalized hidden states, same shape

### 4. LM Head (`lm_head`)

A linear projection from the hidden dimension to the vocabulary size. The output is a vector of **logits** — one score per token in the vocabulary. The highest-scoring token is (roughly) the model's prediction for the next token.

**Input**: hidden states `(batch, seq_len, hidden_dim)`
**Output**: logits `(batch, seq_len, vocab_size)` — e.g., `(1, 10, 32000)`

## The naming convention

HuggingFace models use a consistent naming scheme for their modules. For Llama-style models (which includes Llama, Mistral, Qwen2, and many others), the names are:

| Block | Module path |
|-------|------------|
| Embedding | `model.embed_tokens` |
| Transformer layer i | `model.layers.i` |
| Final norm | `model.norm` |
| LM head | `lm_head` |

These names are what appear as prefixes in the `state_dict()`. For example, `model.layers.5.self_attn.q_proj.weight` is the query projection weight in transformer layer 5.

Chiquito encodes these names in a class variable ([`model.py:99-104`](../src/chiquito/model.py#L99-L104)):

```python
class ChiquitoModel(GenerationMixin):
    LAYER_NAMES = {
        "embed": "model.embed_tokens",
        "layer_prefix": "model.layers",
        "norm": "model.norm",
        "lm_head": "lm_head",
    }
```

Different model architectures use different names (e.g., ChatGLM uses `transformer.encoder.layers` instead of `model.layers`). We will see in [Unit 11](11-extensibility.md) how to handle this.

## Creating a model on the meta device

As we saw in [Unit 02](02-pytorch-foundations.md), we can create a full model architecture without allocating any memory by using the meta device:

```python
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
```

This model has all four blocks with correct shapes, but every parameter is on the meta device (zero memory). We can inspect its structure:

```python
# Embedding: (32000, 2048)
print(model.model.embed_tokens.weight.shape)

# Transformer layers: 22
print(len(model.model.layers))

# Norm: (2048,)
print(model.model.norm.weight.shape)

# LM head: (32000, 2048)
print(model.lm_head.weight.shape)
```

## Navigating the module tree

To access sub-modules programmatically (by string name), we walk the attribute tree:

```python
# Access "model.layers" by splitting the dotted path
module = model
for attr in "model.layers".split("."):
    module = getattr(module, attr)
# module is now the nn.ModuleList containing all transformer layers
```

This is how Chiquito builds its layer list in [`model.py:260-292`](../src/chiquito/model.py#L260-L292). The `_build_layers()` method walks the module tree for each of the four block types and collects references:

```python
def _build_layers(self):
    names = self.LAYER_NAMES
    self.layers = []
    self.layer_names = []

    # Embedding
    module = self.model
    for attr in names["embed"].split("."):
        module = getattr(module, attr)
    self.layers.append(module)
    self.layer_names.append(names["embed"])

    # Transformer layers
    module = self.model
    for attr in names["layer_prefix"].split("."):
        module = getattr(module, attr)
    for i, layer in enumerate(module):
        self.layers.append(layer)
        self.layer_names.append(f'{names["layer_prefix"]}.{i}')

    # Norm and LM head follow the same pattern...
```

After this, `self.layers` is an ordered list of modules `[embed, layer.0, layer.1, ..., norm, lm_head]` and `self.layer_names` is the corresponding list of name strings. The forward pass iterates over both in lockstep.

## Rotary Position Embeddings (RoPE)

Transformers need to know the position of each token in the sequence (otherwise "the cat sat on the mat" and "mat the on sat cat the" would look identical). Most modern models use **Rotary Position Embeddings** (RoPE), which encode position information by rotating the query and key vectors.

For our purposes, the key fact is:
- RoPE embeddings are computed once, after the embedding layer, from the position IDs
- They are passed to every transformer layer as a `(cos, sin)` tuple
- The RoPE module itself is a **buffer** (not a learned parameter), so it lives on the GPU permanently and does not need to be loaded per-layer

Chiquito computes position embeddings in [`model.py:340-346`](../src/chiquito/model.py#L340-L346):

```python
def _compute_position_embeddings(self, hidden_states, position_ids):
    rotary_emb = getattr(getattr(self.model, "model", None), "rotary_emb", None)
    if rotary_emb is not None:
        return rotary_emb(hidden_states, position_ids=position_ids)
    return None
```

## Summary

For Chiquito, a transformer model is a pipeline of four named blocks:

1. **Embedding** — token IDs to vectors
2. **N transformer layers** — the repeated self-attention + FFN block
3. **Norm** — stabilize final hidden states
4. **LM head** — vectors to vocabulary scores

Each block has a known name prefix in the `state_dict`, and we can access any of them by walking the module tree. The forward pass will iterate over them in order, loading weights one block at a time.
