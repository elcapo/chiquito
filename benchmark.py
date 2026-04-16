#!/usr/bin/env python
"""Benchmark Chiquito inference."""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch

from chiquito import AutoModel

PROMPT = "The reason why we need local AI is"
MAX_NEW_TOKENS = 20


def _format_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{math.ceil(n / 1024**3)} GB"
    return f"{math.ceil(n / 1024**2)} MB"


def _get_cpu_name() -> str:
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "Unknown"


def _get_total_ram() -> str:
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return _format_bytes(kb * 1024)
        except OSError:
            pass
    return "Unknown"


def _get_system_info() -> dict:
    gpu = "not available"
    vram = "not available"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = props.name
        vram = _format_bytes(props.total_memory)
    return {
        "processor": _get_cpu_name(),
        "ram": _get_total_ram(),
        "gpu": gpu,
        "vram": vram,
    }


def _print_system_info(info: dict) -> None:
    print("System")
    for key, value in info.items():
        label = key.upper() if len(key) <= 4 else key.capitalize()
        print(f"  {label}: {value}")
    print()


BENCHMARKS_PATH = Path(__file__).resolve().parent / "resources" / "benchmarks.json"


def _save_benchmark(
    system: dict,
    model_id: str,
    preload: bool | int,
    quantization: bool | str,
    result: dict,
) -> None:
    entries: list = []
    if BENCHMARKS_PATH.exists():
        entries = json.loads(BENCHMARKS_PATH.read_text())

    entry = {
        "system": system,
        "benchmark": {
            "model": model_id,
            "prompt": PROMPT,
            "max_tokens": MAX_NEW_TOKENS,
            "preload": preload,
            "quantization": quantization,
        },
        "results": {
            "output": result["text"],
            "load_time": round(result["load_time"], 4),
            "generation_time": round(result["generation_time"], 4),
            "tokens_per_second": round(result["tokens_per_second"], 2),
        },
    }
    entries.append(entry)

    BENCHMARKS_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")


def parse_preload_value(v: str) -> bool | int:
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        return int(v)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"invalid value '{v}': must be 'true', 'false', or an integer"
        ) from err


def parse_quantization_value(v: str) -> str | None:
    if v.lower() == "false":
        return None
    return v


def run_once(
    model_id: str, preload: bool | int, quantization: str | None = None
) -> dict:
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_id, preload_to_ram=preload, quantization=quantization
    )
    load_time = time.perf_counter() - t0

    tokens = model.tokenizer(PROMPT, return_tensors="pt")
    input_ids = tokens["input_ids"].cuda()

    torch.manual_seed(42)
    t1 = time.perf_counter()
    output = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    generation_time = time.perf_counter() - t1

    text = model.tokenizer.decode(output[0], skip_special_tokens=True)
    n_tokens = output.shape[1] - input_ids.shape[1]

    del model
    torch.cuda.empty_cache()

    return {
        "preload": preload,
        "load_time": load_time,
        "generation_time": generation_time,
        "tokens": n_tokens,
        "tokens_per_second": n_tokens / generation_time if generation_time > 0 else 0,
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--preload",
        default="false",
        type=parse_preload_value,
        help="preload_to_ram values to benchmark (true, false, or integer window sizes)",
    )
    parser.add_argument(
        "--quantization",
        default="false",
        choices=["false", "4bit", "8bit"],
        help="on-the-fly quantization via bitsandbytes (requires bitsandbytes installed)",
    )
    args = parser.parse_args()

    preload_value = args.preload
    quantization_value = parse_quantization_value(args.quantization)

    system_info = _get_system_info()
    _print_system_info(system_info)
    print(f"Model: {args.model}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Maximum tokens: {MAX_NEW_TOKENS}")
    print(f"Preload: {preload_value}")
    print(f"Quantization: {quantization_value}")
    print()

    result = run_once(
        args.model, preload=preload_value, quantization=quantization_value
    )
    print()

    print(f"Output: {result['text']}")
    print(f"Load time (s): {result['load_time']}")
    print(f"Generation time (s): {result['generation_time']}")
    print(f"Tokens per second: {result['tokens_per_second']}")

    _save_benchmark(system_info, args.model, preload_value, quantization_value, result)
    print(f"\nBenchmark saved to {BENCHMARKS_PATH}")


if __name__ == "__main__":
    main()
