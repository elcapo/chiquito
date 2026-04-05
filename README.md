# Chiquito

![Chiquito](./resources/cover.png)

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

To enable on-the-fly 4-bit quantization (requires [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)):

```python
model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    preload_to_ram=True,
    quantization="4bit",  # or "8bit"
)
```

This reduces weight transfer times by ~4x (4-bit) or ~2x (8-bit) and makes large models fit in RAM. A 32B model goes from ~65 GB to ~16 GB with 4-bit quantization.

## Benchmarks

Test system: Intel Core i9-10980HK, 64 GB RAM, NVIDIA RTX 2080 Super (8 GB VRAM).

Run the benchmark script to compare modes on any model:

```bash
uv run python benchmark.py --model <model_id> --preload true false 5 10
```

### TinyLlama 1.1B

| preload_to_ram | load (s) | gen (s) | tokens | tok/s |
|---|---|---|---|---|
| `True` | 7.91 | 55.10 | 20 | 0.36 |
| `False` | 1.74 | 54.58 | 20 | 0.37 |
| `5` | 1.74 | 55.85 | 20 | 0.36 |
| `10` | 1.77 | 57.00 | 20 | 0.35 |

All modes produce identical output. On a small model that fits in VRAM, generation speed is similar across modes. The preload overhead shows up in load time. In this case, differences are not visible as disk I/O isn't the bottleneck.

### Qwen2.5-Coder 7B

| preload_to_ram | load (s) | gen (s) | tokens | tok/s |
|---|---|---|---|---|
| `True` | 44.45 | 361.67 | 20 | 0.06 |
| `False` | 1.74 | 391.50 | 20 | 0.05 |
| `5` | 2.91 | 377.37 | 20 | 0.05 |
| `10` | 2.82 | 373.87 | 20 | 0.05 |

All modes produce identical output. With a larger model, the preload advantage starts to show: `preload_to_ram=True` is ~8% faster in generation time (361s vs 391s) thanks to pinned memory DMA transfers.

### Qwen2.5-Coder 32B

| preload_to_ram | load (s) | gen (s) | tokens | tok/s |
|---|---|---|---|---|
| `True` | — | — | — | — |
| `False` | 5.16 | 1828.81 | 20 | 0.01 |
| `5` | 5.20 | 1857.62 | 20 | 0.01 |
| `10` | 4.53 | 1857.56 | 20 | 0.01 |
| `34` | 5.22 | 1871.65 | 20 | 0.01 |

`preload_to_ram=True` could not be tested — the model weighs ~65 GB in fp16, which exceeds the 64 GB of available RAM. This is the scenario the sliding window mode was designed for. All tested modes produce identical output and perform similarly, confirming that the disk prefetch keeps up with GPU execution even at this scale.

## Development

Check code formatting and lint errors:

```bash
uv run ruff check src/           # lint (import order, unused vars, common bugs, ...)
uv run ruff format --check src/  # formatting (reports diffs without modifying files)
```

Auto-fix both:

```bash
uv run ruff check --fix src/
uv run ruff format src/
```

Run tests:

```bash
uv run pytest
```

Run type checking:

```bash
uv run mypy
```

## Documentation

See [docs/](docs/README.md) for developer documentation covering the concepts behind the code: layer splitting, RAM preloading, the sliding window, KV cache, and how to extend Chiquito for new architectures.

Also, see [tutorial/](tutorial/README.md) for an introduction to the concepts involved in the source code and a step by step guide on how the project was built.

## Acknowledgments

The layer by layer inference idea comes from [AirLLM](https://github.com/lyogavin/airllm) by [Gavin Li](https://github.com/lyogavin). Chiquito is a simplified rewrite with the RAM-preloading twist, not a fork.
