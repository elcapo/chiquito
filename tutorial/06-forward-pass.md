# The Layer-by-Layer Forward Pass

This is the heart of Chiquito: the `forward()` method that iterates through layers, loading each one's weights to the GPU, running it, and freeing the memory. This unit builds the core loop implemented in [`model.py:413-531`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L413-L531).

## The load-execute-free cycle

The fundamental pattern for each layer is:

```python
# 1. Load weights from file to CPU
state_dict = load_safetensors("chiquito_split/model.layers.5.safetensors")

# 2. Move weights from CPU to GPU
for param_name, tensor in state_dict.items():
    set_module_tensor_to_device(model, param_name, "cuda:0",
                                 value=tensor, dtype=torch.float16)

# 3. Execute the layer
output = layer(input)

# 4. Free GPU memory — move weights back to meta
for param_name in state_dict:
    set_module_tensor_to_device(model, param_name, "meta")
torch.cuda.empty_cache()
```

In Chiquito, steps 1, 2, and 4 are encapsulated in helper methods:

- Step 1 is `_load_layer_to_cpu()` ([`model.py:311-316`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L311-L316)).
- Step 2 is `_move_layer_to_device()` ([`model.py:318-328`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L318-L328)).
- Step 4 happens inline in the forward loop ([`model.py:517-519`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L517-L519)).

## Resetting the model before each forward call

Remember that `forward()` is called once per generated token and each call iterates through all layers. Before starting, we need a clean model: all parameters back on meta.

For non-quantized models, this is a lightweight operation ([`model.py:251-258`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L251-L258)):

```python
def _reset_model_to_meta(self):
    for name, _ in self.model.named_parameters():
        set_module_tensor_to_device(self.model, name, "meta")
    # Re-place buffers on device (these stay permanently on GPU)
    for buffer_name, buffer in self.model.named_buffers():
        set_module_tensor_to_device(
            self.model, buffer_name, self._device,
            value=buffer, dtype=self._dtype
        )
```

Notice that **buffers** (like RoPE frequencies) are moved back to the GPU. They are small and needed by every layer, so they live on the GPU permanently.

The reset happens at the start of `forward()` ([`model.py:426-434`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L426-L434)):

```python
def forward(self, input_ids, ...):
    if self.hf_quantizer is not None:
        # Quantized models need full reinit (see "Pre-Quantized Weight Caching")
        del self.model
        clean_memory()
        self._init_model()
    else:
        self._reset_model_to_meta()
        clean_gpu_memory()
```

## Causal attention masks

Transformers use **attention masks** to control which tokens can attend to which. For a causal (left-to-right) language model, each token can only attend to itself and previous tokens, never to future tokens.

### Prefill (first forward call)

During prefill, we process the entire input prompt at once. The mask is a triangular matrix — token i can attend to tokens 0 through i:

```
Token:  0  1  2  3  4
    0 [ T  F  F  F  F ]   token 0 sees only itself
    1 [ T  T  F  F  F ]   token 1 sees tokens 0-1
    2 [ T  T  T  F  F ]   token 2 sees tokens 0-2
    3 [ T  T  T  T  F ]   token 3 sees tokens 0-3
    4 [ T  T  T  T  T ]   token 4 sees tokens 0-4
```

In code ([`model.py:451-453`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L451-L453)):

```python
if is_prefill:
    causal_mask = torch.ones(seq_len, seq_len, device=self._device)
    causal_mask = causal_mask.triu(diagonal=1)[None, None, ...] == 0
```

`triu(diagonal=1)` creates an upper triangular matrix with ones above the diagonal. Comparing with `== 0` inverts it to get the lower-triangular-plus-diagonal pattern we want.

### Decode (subsequent forward calls)

During decode, we process a single new token. This token can attend to all previous tokens (which are in the KV cache) plus itself. The mask is simply all `True`:

```python
else:
    causal_mask = torch.ones(1, 1, seq_len, total_len,
                              dtype=torch.bool, device=self._device)
```

Where `total_len = past_len + seq_len` (all cached tokens plus the new one).

## Position IDs

Each token needs to know its position in the sequence. Position IDs are a simple range:

```python
if position_ids is None:
    position_ids = torch.arange(
        past_len, total_len, dtype=torch.long, device=self._device
    )[None, :]
```

During prefill: `[0, 1, 2, ..., seq_len-1]`. During decode: `[past_len]` (a single position for the single new token).

These position IDs are used by the RoPE module to compute position embeddings (see [Transformer Architecture](04-transformer-architecture.md)).

## The forward loop

With all the setup done, the core loop iterates through layers ([`model.py:477-528`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L477-L528)):

```python
with torch.inference_mode():
    for i, (layer_name, layer) in enumerate(
        zip(self.layer_names, self.layers)
    ):
        # Load weights to CPU, then move to GPU
        state_dict = self._load_layer_to_cpu(layer_name)
        moved = self._move_layer_to_device(state_dict)

        # Run layer based on its type
        if layer_name == names["embed"]:
            hidden_states = layer(input_ids)
            position_embeddings = self._compute_position_embeddings(
                hidden_states, position_ids
            )
        elif layer_name == names["norm"]:
            hidden_states = self._run_norm(layer, hidden_states)
        elif layer_name == names["lm_head"]:
            hidden_states = self._run_lm_head(layer, hidden_states)
        else:
            hidden_states = self._run_transformer_layer(
                layer, hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=True,
            )

        # Free GPU memory
        for param_name in moved:
            set_module_tensor_to_device(self.model, param_name, "meta")
        clean_gpu_memory()
```

The layer dispatch is straightforward: check the layer name against the four known types and call the appropriate function. Embedding gets special treatment because it also triggers RoPE computation. The transformer layers get the full set of arguments (mask, positions, cache).

## Layer execution methods

The transformer layer, norm, and LM head are called through dedicated methods ([`model.py:348-374`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L348-L374)) that serve as override points for different architectures:

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

def _run_norm(self, layer, hidden_states):
    return layer(hidden_states)

def _run_lm_head(self, layer, hidden_states):
    return layer(hidden_states).float()  # logits in float32
```

## The return value

After all layers have run, the final hidden states are the logits. We return them wrapped in a `CausalLMOutputWithPast` object ([`model.py:530-531`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L530-L531)):

```python
logits = hidden_states
return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)
```

This is the format that `GenerationMixin` (HuggingFace's text generation loop) expects. We will see how this integrates with text generation in the next unit.

## Summary

The forward pass implements the core promise of Chiquito:

1. Reset all parameters to meta (free everything)
2. Build attention mask and position IDs
3. For each layer: load weights → move to GPU → execute → free GPU
4. Return logits and KV cache

At this point we have a working layer-by-layer forward pass. What we do not yet have is the ability to generate text token by token — that requires the KV cache and the autoregressive generation loop, which is the subject of the next unit.
