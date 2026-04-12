# Putting It All Together

Over the previous chapters we have built every piece of Chiquito: the splitter, the meta device model, the layer-by-layer forward pass, KV cache, memory management, three loading strategies, quantization, and architecture extensibility. This final chapter shows how all the pieces connect and traces the complete flow from construction to text generation.

## The file map

| File | Responsibility |
|------|---------------|
| [`__init__.py`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/__init__.py) | Package exports |
| [`auto_model.py`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/auto_model.py) | Factory + registry ([Architecture Extensibility](11-extensibility.md)) |
| [`model.py`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py) | Core engine ([The Forward Pass](06-forward-pass.md) through [Quantization](10-quantization.md)) |
| [`splitter.py`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/splitter.py) | Checkpoint splitting + pre-quantization ([Splitting Checkpoints](05-checkpoint-splitting.md), [Quantization](10-quantization.md)) |
| [`utils.py`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/utils.py) | Memory + I/O helpers ([HuggingFace Models](03-huggingface-models.md), [Memory Management](08-memory-management.md)) |

## The initialization sequence

When the user calls `AutoModel.from_pretrained("some-model")`, here is what happens:

### 1. Architecture detection ([`auto_model.py:25-40`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/auto_model.py#L25-L40))

```
AutoModel.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
  ├─ Load config.json → architectures[0] = "Qwen2ForCausalLM"
  ├─ Check registry: no match for "Qwen2"
  └─ Fall back to ChiquitoModel("Qwen/Qwen2.5-Coder-7B-Instruct", ...)
```

### 2. Constructor ([`model.py:106-165`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L106-L165))

```
ChiquitoModel.__init__
  ├─ Store hyperparameters (device, dtype, max_seq_len, ...)
  ├─ Interpret preload_to_ram: True → None, False → 0, int → N
  │
  ├─ Build layer name list from config
  │  ├─ Load config (no weights)
  │  ├─ Create meta model, count layers
  │  └─ Return ["model.embed_tokens", "model.layers.0", ..., "model.norm", "lm_head"]
  │
  ├─ Split checkpoint (and pre-quantize if quantization requested)
  │  ├─ resolve_model_path() — download from Hub if needed
  │  └─ split_and_save_layers() — create chiquito_split/ (+ chiquito_split_{4bit,8bit}/)
  │
  ├─ Load config, tokenizer, generation_config
  │
  ├─ _init_model()
  │  ├─ Create meta model (with SDPA attention if supported)
  │  ├─ Apply quantization if requested (replace nn.Linear with bnb modules)
  │  ├─ model.eval() + model.tie_weights()
  │  ├─ _build_layers() — extract module references for the 4 block types
  │  └─ Move buffers to device (RoPE frequencies, etc.)
  │
  └─ Preload weights
     ├─ If preload_to_ram=True:  _preload_all_layers() — load all + pin memory
     ├─ If preload_to_ram=N:    _start_window_cache() — start sliding window
     └─ If preload_to_ram=False: (nothing — read from disk on demand)
```

## The generation flow

When the user calls `model.generate(input_ids, max_new_tokens=20)`:

```
GenerationMixin.generate()
  │
  ├─ Step 1: PREFILL (process the full prompt)
  │  ├─ prepare_inputs_for_generation(input_ids, cache=None)
  │  │  └─ Returns full prompt, no trimming
  │  │
  │  └─ forward(input_ids=[prompt], past_key_values=empty)
  │     ├─ Reset model (meta for pre-quantized/fp16, full reinit for on-the-fly quantized)
  │     ├─ Restart sliding window cache if applicable
  │     ├─ Build triangular causal mask (seq_len x seq_len)
  │     ├─ For each layer:
  │     │  ├─ Load weights (from RAM cache / sliding window / disk)
  │     │  ├─ Move to GPU (quantize if bnb)
  │     │  ├─ Execute (embed / transformer / norm / lm_head)
  │     │  ├─ Move weights to meta
  │     │  └─ empty_cache()
  │     └─ Return logits + KV cache (now populated)
  │
  ├─ Steps 2-20: DECODE (one token at a time)
  │  ├─ prepare_inputs_for_generation(all_ids, cache)
  │  │  └─ Trim to last token only
  │  │
  │  └─ forward(input_ids=[1 token], past_key_values=cache)
  │     ├─ Reset model
  │     ├─ Build all-True mask (1 x total_len)
  │     ├─ For each layer: same load-run-free cycle
  │     └─ Return logits + updated KV cache
  │
  └─ Return all generated token IDs
```

## Where time goes

For a 32B model with 67 layers generating 20 tokens:

```
Total layer loads: 67 layers x 20 tokens = 1,340
```

| Mode | Time per load | Total transfer | Notes |
|------|--------------|---------------|-------|
| Disk (NVMe) | ~200 ms | ~268 s | Read from SSD each time |
| RAM (pinned) | ~80 ms | ~107 s | DMA via PCIe 3.0 |
| RAM + 4-bit | ~20 ms | ~27 s | 4x less data to transfer |

The actual GPU computation for each layer takes only a few milliseconds during decode (a single token through attention + FFN). Weight transfer dominates by 10-50x.

## The benchmark script

[`benchmark.py`](../benchmark.py) measures inference across multiple models, preload modes, and quantization levels. Before running benchmarks, it prints model info (parameter counts, per-layer and total sizes in fp16/8-bit/4-bit) for each model. Then it runs all combinations and produces a comparison table:

```python
def run_once(model_id, preload, quantization=None):
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(model_id, preload_to_ram=preload,
                                       quantization=quantization)
    load_time = time.perf_counter() - t0

    tokens = model.tokenizer(PROMPT, return_tensors="pt")
    input_ids = tokens["input_ids"].cuda()

    t1 = time.perf_counter()
    output = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False)
    gen_time = time.perf_counter() - t1
    ...
```

Usage:
```bash
# Single model, default preload modes, no quantization
uv run benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Multiple models with quantization comparison
uv run benchmark.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 Qwen/Qwen2.5-7B \
  --preload true false 5 \
  --quantization false 4bit 8bit
```

## The complete dependency graph

```
User code
  └─ AutoModel.from_pretrained()        [auto_model.py]
      └─ ChiquitoModel.__init__()       [model.py]
          ├─ resolve_model_path()        [utils.py]
          ├─ find_or_create_split()      [splitter.py]
          │   ├─ split_and_save_layers() [splitter.py]
          │   │   ├─ load_file()         [safetensors]
          │   │   ├─ save_safetensors()  [utils.py]
          │   │   └─ _quantize_state_dict()  [splitter.py → bitsandbytes]
          │   └─ clean_memory()          [utils.py]
          ├─ AutoConfig / AutoTokenizer  [transformers]
          ├─ init_empty_weights()        [accelerate]
          └─ _preload_all_layers() or _SlidingWindowCache
              └─ load_safetensors()      [utils.py]

  └─ model.generate()                   [GenerationMixin from transformers]
      └─ model.forward()                [model.py]
          ├─ _load_layer_to_cpu()        [model.py → utils.py or cache]
          ├─ _move_layer_to_device()     [model.py → accelerate]
          │   └─ _move_quantized_layer_to_device()  [model.py → bitsandbytes]
          │       └─ parse_quantized_state_dict()    [splitter.py]
          ├─ Layer execution             [transformers model internals]
          ├─ set_module_tensor_to_device("meta")  [accelerate]
          └─ clean_gpu_memory()          [utils.py]
```

## Design decisions recap

| Decision | Rationale |
|----------|-----------|
| Inherit from `GenerationMixin` | Reuse HuggingFace's battle-tested generation loop instead of writing our own |
| Split checkpoints to disk | One-time cost; avoids re-parsing shards on every run |
| Meta device for model skeleton | Zero-memory initialization; weight loading is controlled per-layer |
| Pinned memory for RAM cache | 2-5x faster DMA transfers over PCIe |
| Sliding window as bounded buffer | Enables models that exceed RAM capacity with predictable memory usage |
| Pre-quantized weight caching | Quantize once, cache to disk; subsequent runs load packed weights directly — faster transfers, less RAM |
| Override points instead of config | Subclasses can change behavior (how to call a layer) not just data (what names to use) |

## What you now know

Starting from just Python, the concept of an LLM, and the idea of inference, you have learned:

1. **The VRAM problem** — why large models do not fit, and how sequential layer processing works around it
2. **PyTorch fundamentals** — tensors, devices, modules, state dicts, and the meta device
3. **HuggingFace ecosystem** — configs, tokenizers, safetensors, weight maps, and the Hub
4. **Transformer structure** — the four blocks (embedding, layers, norm, lm_head) and their naming
5. **Checkpoint splitting** — converting sharded checkpoints into per-layer files
6. **The forward pass** — the load-execute-free cycle with attention masks and position IDs
7. **KV cache** — avoiding redundant computation during autoregressive generation
8. **Memory management** — GPU cache cleanup, Python GC, malloc_trim, and pinned memory
9. **Loading strategies** — full preload, sliding window (producer-consumer), and disk-only with prefetch
10. **Quantization** — 4-bit/8-bit compression to reduce transfer bottleneck
11. **Extensibility** — factory pattern and override points for new architectures

These concepts enable running models that need 140 GB of VRAM on a GPU with 8 GB.
