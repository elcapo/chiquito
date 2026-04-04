# The Forward Pass

## Overview

`ChiquitoModel.forward()` runs the model layer by layer: load weights, execute, free, repeat. It is called once per generated token by `GenerationMixin.generate()`.

## The four layer types

Each forward pass iterates over all layers in order. The layer name determines what happens:

| Layer | What it does |
|-------|-------------|
| **Embedding** (`model.embed_tokens`) | Converts token IDs to hidden states. Also triggers computation of [rotary position embeddings](https://arxiv.org/abs/2104.09864). |
| **Transformer** (`model.layers.N`) | Self-attention + feed-forward. The core of the model. Receives attention mask, position embeddings, and optionally a KV cache. |
| **Norm** (`model.norm`) | [RMSNorm](https://arxiv.org/abs/1910.07467) applied to the final hidden states. |
| **LM Head** (`lm_head`) | Linear projection from hidden size to vocabulary size, producing logits. |

## Weight loading and GPU lifecycle

For each layer:

```python
# 1. Load weights to CPU
state_dict = self._load_layer_to_cpu(layer_name)

# 2. Copy to GPU
self._move_layer_to_device(state_dict)  # uses set_module_tensor_to_device()

# 3. Execute
hidden_states = layer(hidden_states, ...)

# 4. Free GPU memory
layer.to("meta")
clean_gpu_memory()  # torch.cuda.empty_cache()
```

`_move_layer_to_device()` uses accelerate's `set_module_tensor_to_device()`, which replaces a meta-device parameter with a real tensor on the target device. This is the inverse of `layer.to("meta")`.

## KV cache

### The problem without KV cache

Transformer attention computes `softmax(Q @ K^T) @ V` where Q, K, V come from the current input. During text generation, each new token must attend to all previous tokens. Without caching, the model reprocesses the entire sequence on every token — O(T * N * S) total work for T tokens, N layers, S sequence length.

### How KV cache helps

The key insight: K and V states for previous tokens don't change between generation steps. By caching them, each decode step only computes K and V for the new token, then concatenates with the cached values.

Chiquito uses transformers' [`DynamicCache`](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.DynamicCache):

```python
# Created once at the start of generation
cache = DynamicCache()

# Each attention layer calls internally:
key_states, value_states = cache.update(new_key, new_value, layer_idx)
# Returns concatenation: [cached_K; new_K], [cached_V; new_V]
```

The cache is a mutable object passed through all layers. Each attention layer writes to its own index via `.update()`. The cache lives on GPU between forward calls — for typical sequence lengths it uses only a few MB.

### Prefill vs. decode

| | Prefill (1st forward) | Decode (subsequent forwards) |
|---|---|---|
| **Input** | Full prompt (S tokens) | 1 new token |
| **Cache state** | Empty (past_len = 0) | Contains S + previous tokens |
| **Attention mask** | Causal (S x S triangular) | All-ones (1 x total_len) |
| **Position IDs** | [0, 1, ..., S-1] | [total_len - 1] |
| **Compute** | Full attention over S tokens | Attention over 1 query vs. all cached K/V |

The detection is based on `past_key_values.get_seq_length()`: if 0, it's a prefill.

### Interaction with GenerationMixin

`GenerationMixin.generate()` manages the generation loop:

1. Calls `_prepare_cache_for_generation()` — creates an empty `DynamicCache`.
2. Calls `prepare_inputs_for_generation()` — our implementation trims `input_ids` to only new tokens when a cache exists, and passes `use_cache=True`.
3. Calls `forward()` — we pass the cache to each transformer layer.
4. The returned `CausalLMOutputWithPast(past_key_values=cache)` feeds the cache back to step 2 on the next iteration.

### Where time goes with KV cache

KV cache eliminates redundant compute but does not eliminate weight loading. Each decode step still loads all N layers from RAM/disk to GPU. For large models, this I/O dominates:

```
32B model, 67 layers, ~1.4s per layer load:
  Per decode token: 67 × 1.4s ≈ 94s
  20 tokens: 94s × 20 ≈ 31 min
```

The compute per decode step (attention over 1 token) is negligible compared to the weight transfer. This is the fundamental limitation of layer by layer inference: it trades VRAM for time.

## Memory cleanup

After each layer, two things are freed:

1. **GPU weights** — `layer.to("meta")` moves parameters back to the meta device. If an HF quantizer is active, parameters are moved individually via `set_module_tensor_to_device(model, param, "meta")`.
2. **CUDA cache** — `clean_gpu_memory()` calls `torch.cuda.empty_cache()`. This is the lightweight variant; the full `clean_memory()` (which includes `gc.collect()` and `malloc_trim()`) is only used during model initialization.

## Model reset between forward calls

At the start of each `forward()`, `_reset_model_to_meta()` moves all parameters back to meta device and re-places buffers (like rotary embedding frequencies) on the target device. This is much cheaper than the original approach of deleting and recreating the entire model.
