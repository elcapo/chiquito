# Chiquito

Chiquito offers layer by layer LLM inference for machines with limited VRAM but plenty of RAM. It loads HuggingFace transformer models one layer at a time onto the GPU, making it possible to run large models on consumer hardware. It is a clean, minimal rewrite inspired by [AirLLM](https://github.com/lyogavin/airllm).

**The twist:** while AirLLM loads weights from disk on every forward pass, Chiquito preloads all layer weights into system RAM by default (`preload_to_ram=True`). Copying from RAM to GPU over PCIe is 2-5x faster than reading from even a fast NVMe SSD, so inference is noticeably quicker if you have the RAM to spare.

## How it works

On initialization, Chiquito splits the HuggingFace checkpoint into one `.safetensors` file per layer and loads them into system RAM as CPU tensors.

During inference, each layer is copied from RAM to GPU, executed, and immediately freed. Only one layer lives on the GPU at any given time.

For models that don't fit in RAM, a **sliding window** mode keeps only N layers loaded at a time. A background thread loads upcoming layers into the freed slots, so the GPU never stalls as long as disk I/O keeps up.

| Mode | RAM usage | Speed |
|---|---|---|
| `preload_to_ram=True` | Full model size | Fastest |
| `preload_to_ram=10` | ~10 layers | Fast |
| `preload_to_ram=False` | Minimal | Slower |

## Installation

```bash
uv sync
```

## Usage

```python
from chiquito import AutoModel

model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tokens = model.tokenizer("The meaning of life is", return_tensors="pt")
output = model.generate(tokens["input_ids"].cuda(), max_new_tokens=50)

print(model.tokenizer.decode(output[0], skip_special_tokens=True))
```

For models that don't fit in RAM, use a sliding window (e.g. 10 layers):

```python
model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    preload_to_ram=10,
)
```

To fall back to disk-based loading (minimal RAM usage, slower inference):

```python
model = AutoModel.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    preload_to_ram=False,
)
```

## Acknowledgments

The layer-by-layer inference idea comes from [AirLLM](https://github.com/lyogavin/airllm) by [Gavin Li](https://github.com/lyogavin). Chiquito is a simplified rewrite with the RAM-preloading twist, not a fork.
