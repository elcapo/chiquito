# Chiquito

![Chiquito](./resources/cover.png)

Chiquito offers layer by layer LLM inference for machines with limited VRAM but plenty of RAM. It loads HuggingFace transformer models one layer at a time onto the GPU, making it possible to run large models on consumer hardware. It is a clean, minimal rewrite inspired by [AirLLM](https://github.com/lyogavin/airllm).

**The twist:** while AirLLM loads weights from disk on every forward pass, Chiquito preloads all layer weights into system RAM by default (`preload_to_ram=True`). Copying from RAM to GPU over PCIe is 2-5x faster than reading from even a fast NVMe SSD, so inference is noticeably quicker if you have the RAM to spare.

> [!WARNING]
> Chiquito is an educational project. Its goal is to make large models accessible from consumer hardware so you can study how they work, not to serve them efficiently. Inference is extremely slow (see [benchmarks](#benchmarks)). Do not use this in production.

## Documentation

See [docs/](docs/README.md) for developer documentation covering the concepts behind the code: layer splitting, RAM preloading, the sliding window, KV cache, and how to extend Chiquito for new architectures.

Also, see [tutorial/](tutorial/README.md) for an introduction to the concepts involved in the source code and a step by step guide on how the project was built.

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

To enable 4-bit quantization (requires [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)):

```python
model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    preload_to_ram=True,
    quantization="4bit",  # or "8bit"
)
```

On the first run, Chiquito pre-quantizes the weights and stores them in a dedicated split directory (`chiquito_split_4bit/`). Subsequent runs load the already-quantized files directly, avoiding re-quantization overhead. This reduces weight transfer times by ~4x (4-bit) or ~2x (8-bit) and makes large models fit in RAM. A 32B model goes from ~65 GB to ~16 GB with 4-bit quantization.

## Benchmarks

Test system: Intel Core i9-10980HK, 63 GB RAM, NVIDIA RTX 2080 Super (8 GB VRAM).

All benchmarks were run after the models had already been downloaded, split by layers, and quantized (where applicable), so first-run overhead (downloading, splitting, pre-quantization) is not reflected in the times below.

Prompt: `"The reason why we need local AI is"`, 20 tokens generated.

Run the benchmark script to compare modes on any model:

```bash
uv run python benchmark.py --model <model_id> --preload [true|false|<n_layers>] --quantization [false|4bit|8bit]
```

### TinyLlama 1.1B

| preload_to_ram | quantization | load (s) | gen (s) | tok/s |
|---|---|---|---|---|
| `False` | — | 1.97 | 173.86 | 0.12 |
| `False` | 4bit | 8.09 | 170.99 | 0.12 |
| `True` | 4bit | 10.58 | 162.22 | 0.12 |

On a small model (1.1B parameters, 22 layers), generation speed is virtually identical across all modes at 0.12 tok/s. The model fits easily in both RAM and VRAM, so neither preloading nor quantization has a meaningful impact on throughput.

The load time without quantization (1.97 s) is much lower than with 4-bit quantization (8–10 s) because loading pre-quantized weights involves extra deserialization work. Preloading to RAM adds a bit more load time (10.58 s vs 8.09 s) since all layers must be copied into memory upfront, but this overhead is negligible for such a small model.

### Qwen2.5-Coder 32B

| preload_to_ram | quantization | load (s) | gen (s) | tok/s |
|---|---|---|---|---|
| `False` | — | 3.72 | 2063.69 | 0.01 |
| `False` | 4bit | 11.03 | 655.26 | 0.03 |
| `True` | 4bit | 70.89 | 531.95 | 0.04 |

With a 32B-parameter model (64 layers), the effect of quantization and preloading becomes evident:

- **Without quantization**, each layer is loaded in fp16 (~2 GB per layer). Generation takes over 34 minutes because every layer transfer from disk to GPU is large and slow.
- **4-bit quantization** reduces the per-layer size by ~4x, cutting generation time to ~11 minutes — a **3.1x speedup**.
- **Preloading to RAM** with 4-bit quantization gives a further **19% speedup** (532 s vs 655 s) because layer weights are served from RAM over PCIe instead of being read from disk. The tradeoff is a much higher load time (71 s) as all 64 quantized layers are copied into RAM upfront. This is the sweet spot for models that fit in RAM after quantization: a 32B model goes from ~65 GB (fp16) to ~16 GB (4-bit), well within the 63 GB of available RAM.

### Qwen3.5-122B-A10B (MoE)

| preload_to_ram | quantization | load (s) | gen (s) | tok/s |
|---|---|---|---|---|
| `False` | 4bit | 16.48 | 2602.25 | 0.01 |
| `5` | 4bit | 16.98 | 1870.26 | 0.01 |

A 122B-parameter mixture-of-experts model (10B active per token, 256 experts) running on an 8 GB GPU thanks to two specialized paths in `ChiquitoCompositeModel`: composite-config handling for multimodal architectures, and **lazy per-expert dequantization** that keeps the fused expert weights packed in 4-bit on the GPU and dequantizes only the ~8 selected experts per token (peak VRAM ~1.5 GB per layer instead of ~5.8 GB). See [`composite_model.py`](src/chiquito/composite_model.py) and [`lazy_experts.py`](src/chiquito/lazy_experts.py).

Even with 4-bit quantization, this model is far too large to fit entirely in RAM. The **sliding window** (`preload_to_ram=5`) keeps 5 layers buffered in RAM at a time, and a background thread prefetches upcoming layers into freed slots. This yields a **28% speedup** in generation time (1870 s vs 2602 s) compared to pure disk-based loading, while keeping RAM usage bounded. Load times are nearly identical (~17 s) since only a handful of layers are preloaded in either case.

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

## Acknowledgments

The layer by layer inference idea comes from [AirLLM](https://github.com/lyogavin/airllm) by [Gavin Li](https://github.com/lyogavin). Chiquito is a simplified rewrite with the RAM-preloading twist, not a fork.
