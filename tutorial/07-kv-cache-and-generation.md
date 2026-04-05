# KV Cache and Text Generation

A single forward pass produces logits for the next token. To generate text, we need to call `forward()` repeatedly in a loop, each time feeding the newly generated token back as input. The **KV cache** makes this efficient, and HuggingFace's **GenerationMixin** provides the loop itself.

## The problem without a cache

Consider generating 20 tokens from a 10-token prompt. Without caching, each forward call would process the entire sequence:

| Step | Input length | Layers processed | Attention cost |
|------|-------------|-----------------|----------------|
| 1 | 10 tokens | all N | 10 x 10 |
| 2 | 11 tokens | all N | 11 x 11 |
| 3 | 12 tokens | all N | 12 x 12 |
| ... | ... | ... | ... |
| 20 | 29 tokens | all N | 29 x 29 |

The attention computation grows quadratically. More importantly, we are recomputing the same attention results for tokens 0-9 on every step — wasteful.

## What the KV cache stores

In each transformer layer, the self-attention mechanism computes three projections from the hidden states:

- **Query (Q)**: what the current token is looking for
- **Key (K)**: what each token offers to be found
- **Value (V)**: what each token offers as content

The attention output for a token depends on its Q dotted against all previous K's, weighted to retrieve previous V's. The crucial insight: **K and V for a given token at a given layer never change once computed**. Only Q changes (for the new token).

The KV cache stores the K and V tensors from all previous tokens. On subsequent forward calls:
- We only compute Q, K, V for the **new** token
- We append the new K and V to the cache
- We compute attention using the new Q against all cached K's and V's

This reduces the attention cost from quadratic to linear:

| Phase | Input | What's computed | Attention cost |
|-------|-------|----------------|----------------|
| **Prefill** | Full prompt (10 tokens) | Q, K, V for all 10 tokens | 10 x 10 |
| **Decode** (each subsequent token) | 1 new token | Q, K, V for 1 token | 1 x (10+n) |

## DynamicCache from transformers

HuggingFace provides `DynamicCache`, a container that stores K and V tensors for each layer. Each layer writes to its own index.

Chiquito creates the cache at the start of `forward()` if none exists ([`model.py:444-448`](../src/chiquito/model.py#L444-L448)):

```python
if past_key_values is None:
    past_key_values = DynamicCache()
past_len = past_key_values.get_seq_length()
is_prefill = past_len == 0
total_len = past_len + seq_len
```

The cache is passed to each transformer layer via the `past_key_values` argument. Internally, each layer calls `cache.update(new_K, new_V, layer_idx)` to append its K and V. This happens inside the transformer layer's attention module — we do not need to implement it ourselves.

The cache object **lives on the GPU between forward calls**. It is relatively small compared to layer weights (it stores per-token vectors, not weight matrices), so keeping it in VRAM is not a problem.

Note how `is_prefill` drives the attention mask shape:
- `is_prefill = True` (first call, `past_len == 0`): triangular mask of shape `(seq_len, seq_len)`
- `is_prefill = False` (subsequent calls, `past_len > 0`): all-True mask of shape `(1, total_len)`

## GenerationMixin: the autoregressive loop

Instead of writing our own generation loop, Chiquito inherits from HuggingFace's `GenerationMixin` ([`model.py:97`](../src/chiquito/model.py#L97)):

```python
class ChiquitoModel(GenerationMixin):
    ...
```

`GenerationMixin` provides a `.generate()` method that handles:
- The autoregressive loop (call forward, sample token, append, repeat)
- Sampling strategies (greedy, top-k, top-p, etc.)
- Stopping conditions (max length, EOS token)
- KV cache creation and passing

To use it, our model must implement three things:

### 1. `can_generate()` — tell GenerationMixin we support generation

```python
def can_generate(self) -> bool:
    return True
```

([`model.py:378-379`](../src/chiquito/model.py#L378-L379))

### 2. `prepare_inputs_for_generation()` — prepare each step's inputs

This is called before each `forward()` call. Its main job is to **trim the input** to only the new token when a cache exists:

```python
def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
    past_key_values = kwargs.get("past_key_values")

    if past_key_values is not None:
        past_len = past_key_values.get_seq_length()
        if input_ids.shape[1] > past_len:
            input_ids = input_ids[:, past_len:]
        else:
            input_ids = input_ids[:, -1:]

    position_ids = None
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values is not None:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
        "use_cache": True,
    }
```

([`model.py:381-406`](../src/chiquito/model.py#L381-L406))

On the first call, `past_key_values` is None, so `input_ids` is the full prompt. On subsequent calls, we slice to only the new token(s). Position IDs are computed from the attention mask using `cumsum`.

### 3. `forward()` — the layer-by-layer forward pass

This is what we built in [Unit 06](06-forward-pass.md). `GenerationMixin` calls it, gets back logits and the updated cache, samples the next token, and loops.

### 4. `__call__` — make the model callable

`GenerationMixin` calls the model as a function, so we need `__call__` to route to `forward()` ([`model.py:408-409`](../src/chiquito/model.py#L408-L409)):

```python
def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)
```

## End-to-end generation flow

Here is what happens when the user calls `model.generate(input_ids, max_new_tokens=20)`:

```
1. GenerationMixin.generate()
   ├─ Creates empty DynamicCache
   ├─ Step 1 (prefill):
   │  ├─ prepare_inputs_for_generation(full prompt, cache=None)
   │  │  └─ Returns full input_ids (no trimming)
   │  ├─ forward(input_ids=[10 tokens], past_key_values=empty cache)
   │  │  ├─ Build triangular mask (10x10)
   │  │  ├─ Iterate all layers: load → run → free
   │  │  └─ Return logits + cache (now contains K/V for 10 tokens)
   │  └─ Sample next token from logits
   │
   ├─ Steps 2-20 (decode):
   │  ├─ prepare_inputs_for_generation(all tokens so far, cache)
   │  │  └─ Trims input_ids to just the last token
   │  ├─ forward(input_ids=[1 token], past_key_values=cache)
   │  │  ├─ Build all-True mask (1 x total_len)
   │  │  ├─ Iterate all layers: load → run → free
   │  │  └─ Return logits + updated cache
   │  └─ Sample next token from logits
   │
   └─ Return all generated token IDs
```

Note that **every decode step iterates through all layers** — this is the speed cost of layer-by-layer inference. The KV cache saves us the quadratic attention cost, but the weight loading cost is linear in the number of layers and happens on every token.

## Summary

The KV cache stores key and value tensors from previous tokens so we only compute attention for new tokens. `GenerationMixin` provides the autoregressive generation loop. Our model just needs to implement `forward()`, `prepare_inputs_for_generation()`, and `can_generate()`.

At this point we have a **functionally complete** layer-by-layer inference engine. It can load any Llama-style model and generate text. But it always reads weights from disk on every layer load, which is slow. The next units add the optimizations that make it practical: memory management, RAM preloading, and quantization.
