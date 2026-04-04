# Architecture Overview

## File structure

The program source code is in [src/chiquito](../src/chiquito/) and it consists in just a bunch of source files:

- **__init__.py**: Exports the main classes (`ChiquitoModel` and `AutoModel`).
- **auto_model.py**: Factory with architecture registry.
- **model.py**: `ChiquitoModel`, `_SlidingWindowCache`.
- **splitter.py**: Checkpoint splitting into per-layer files.
- **utils.py**: Memory cleanup, safetensors I/O, HuggingFace path resolution.

## Dependencies

Chiquito builds on top of the HuggingFace ecosystem:

- [PyTorch](https://pytorch.org/) — Tensor operations and GPU compute.
- [transformers](https://huggingface.co/docs/transformers) — Model configs, tokenizers, `GenerationMixin` for text generation, and `DynamicCache` for KV caching.
- [accelerate](https://huggingface.co/docs/accelerate) — `init_empty_weights()` to create models on [the meta device](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference) without allocating memory, and `set_module_tensor_to_device()` to place individual parameters on a device.
- [safetensors](https://huggingface.co/docs/safetensors) — Fast, safe serialization format for tensors. Used for both reading HuggingFace checkpoints and writing per-layer splits.
- [huggingface-hub](https://huggingface.co/docs/huggingface_hub) — Downloads model files from the HuggingFace Hub.

## Data flow

```mermaid
flowchart TD
    A["<b>AutoModel.from_pretrained</b>('model_id')"] --> B[Detect architecture from config]
    B --> C[Instantiate <b>ChiquitoModel</b>]

    C -->|"1st: split"| D[<b>find_or_create_split</b>]
    D --> D1[<b>resolve_model_path</b><br><i>download or locate local model</i>]
    D1 --> D2[<b>split_and_save_layers</b><br><i>one .safetensors file per layer</i>]

    C -->|"2nd: init"| E[<b>_init_model</b>]
    E --> E1["<b>init_empty_weights</b><br><i>model on meta device (0 bytes)</i>"]
    E1 --> E2[<b>AutoModelForCausalLM.from_config</b>]
    E2 --> E3[Move buffers to GPU<br><i>rotary embeddings, etc.</i>]

    C -->|"3rd: preload"| F{preload_to_ram?}
    F -->|True| F1[<b>_preload_all_layers</b><br><i>all layers into RAM dict</i>]
    F -->|int N| F2[<b>_start_window_cache</b><br><i>sliding window + background thread</i>]
    F -->|False| F3[No preload<br><i>disk on each call</i>]

    C -->|"4th: generate"| G["<b>model.generate</b>(input_ids, max_new_tokens=N)"]
    G --> H["<b>forward</b>() — called once per token"]

    H --> I[<b>_reset_model_to_meta</b><br><i>clean parameter state</i>]
    I --> J[For each layer]

    J --> K[<b>_load_layer_to_cpu</b><br><i>from RAM, window, or disk</i>]
    K --> L[<b>_move_layer_to_device</b><br><i>CPU → GPU</i>]
    L --> M{Layer type?}

    M -->|embed| N1[Embedding + rotary pos emb]
    M -->|transformer| N2[Self-attention + FFN<br><i>with KV cache</i>]
    M -->|norm| N3[RMSNorm]
    M -->|lm_head| N4[Linear → logits]

    N1 --> O["<b>set_module_tensor_to_device</b>(meta)<br><i>free GPU memory</i>"]
    N2 --> O
    N3 --> O
    N4 --> O

    O --> P[<b>clean_gpu_memory</b>]
    P --> J

    style A fill:#4a6fa5,color:#fff
    style G fill:#4a6fa5,color:#fff
    style F fill:#e8a838,color:#000
    style M fill:#e8a838,color:#000
    style O fill:#c0392b,color:#fff
    style K fill:#27ae60,color:#fff
    style L fill:#27ae60,color:#fff
```

## The meta device

The [meta device](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference) is a PyTorch concept that allows creating tensors with a shape and dtype but no actual data. A model on the meta device describes the full architecture (layer types, parameter shapes, buffer values) without using any GPU or CPU memory for weights.

Chiquito uses this as follows:

1. **Init**: Create the full model on meta device. This gives us the correct module hierarchy and shapes.
2. **Per-layer load**: Use `set_module_tensor_to_device()` to replace a meta parameter with a real tensor loaded from a safetensors file.
3. **After execution**: Move the layer back to meta to free GPU memory.

This cycle repeats for every layer on every forward pass.
