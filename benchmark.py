#!/usr/bin/env python
"""Benchmark Chiquito inference across different models, preload modes, and quantization levels."""

import argparse
import itertools
import time
from collections import defaultdict

import torch
from transformers import AutoConfig

from chiquito import AutoModel

PROMPT = "The reason why we need local AI is"
MAX_NEW_TOKENS = 20


def parse_preload_value(v: str) -> bool | int:
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    return int(v)


def parse_quantization_value(v: str) -> str | None:
    if v.lower() == "false":
        return None
    return v


# ---------------------------------------------------------------------------
# Model info helpers (merged from model-info.py)
# ---------------------------------------------------------------------------


def per_block_params(cfg: AutoConfig) -> int:
    H = cfg.hidden_size
    intermediate = cfg.intermediate_size
    h = cfg.num_attention_heads
    kv = getattr(cfg, "num_key_value_heads", h)
    d = getattr(cfg, "head_dim", None) or H // h
    attn = 2 * H * H + 2 * H * kv * d
    mlp = 3 * H * intermediate
    norms = 2 * H
    return attn + mlp + norms


def total_params(cfg: AutoConfig) -> int:
    block = per_block_params(cfg)
    V, H = cfg.vocab_size, cfg.hidden_size
    tied = getattr(cfg, "tie_word_embeddings", False)
    embed = V * H
    lm_head = 0 if tied else V * H
    return cfg.num_hidden_layers * block + embed + lm_head + H


def fmt(n_bytes: float) -> str:
    gb = n_bytes / 1024**3
    if gb >= 1:
        return f"{gb:.2f} GB"
    return f"{n_bytes / 1024**2:.0f} MB"


def get_model_info(model_ids: list[str]) -> dict[str, dict[str, str | int]]:
    """Return size info per model for embedding in the results table."""
    info: dict[str, dict[str, str | int]] = {}
    for model_id in model_ids:
        cfg = AutoConfig.from_pretrained(model_id)
        block = per_block_params(cfg)
        total = total_params(cfg)
        info[model_id] = {
            "layers": cfg.num_hidden_layers,
            "layer_fp16": fmt(block * 2),
            "total_fp16": fmt(total * 2),
            "total_int8": fmt(total * 1),
            "total_nf4": fmt(total * 0.5),
        }
    return info


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


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
    gen_time = time.perf_counter() - t1

    text = model.tokenizer.decode(output[0], skip_special_tokens=True)
    n_tokens = output.shape[1] - input_ids.shape[1]

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_id,
        "preload": preload,
        "quantization": quantization,
        "load_time": load_time,
        "gen_time": gen_time,
        "tokens": n_tokens,
        "tok_per_s": n_tokens / gen_time if gen_time > 0 else 0,
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        nargs="+",
        required=True,
        help="HuggingFace model ID(s) or local path(s)",
    )
    parser.add_argument(
        "--preload",
        nargs="+",
        default=["true", "false", "5"],
        help="preload_to_ram values to benchmark (true, false, or integer window sizes)",
    )
    parser.add_argument(
        "--quantization",
        nargs="+",
        default=["false"],
        choices=["false", "4bit", "8bit"],
        help="quantization levels to benchmark (false for none, 4bit, 8bit)",
    )
    args = parser.parse_args()

    preload_values = [parse_preload_value(v) for v in args.preload]
    quantization_values = [parse_quantization_value(v) for v in args.quantization]

    # Gather model info (parameter counts and sizes)
    model_info = get_model_info(args.model)

    print(f"Prompt:        {PROMPT!r}")
    print(f"Models:        {args.model}")
    print(f"Preload modes: {preload_values}")
    print(f"Quantization:  {[q or 'none' for q in quantization_values]}")
    print()

    # Run all combinations
    results = []
    combos = list(itertools.product(args.model, quantization_values, preload_values))
    for model_id, quant, preload in combos:
        quant_label = quant or "none"
        print(f"--- {model_id}  preload={preload}  quant={quant_label} ---")
        r = run_once(model_id, preload, quantization=quant)
        results.append(r)
        print(f"    Output: {r['text']}")
        print()

    # Print results table
    headers = [
        "model",
        "layers",
        "layer (fp16)",
        "total (fp16)",
        "total (int8)",
        "total (nf4)",
        "quantization",
        "preload_to_ram",
        "load (s)",
        "gen (s)",
        "tokens",
        "tok/s",
    ]

    def fmt_row(r: dict) -> list[str]:
        info = model_info[r["model"]]
        return [
            r["model"],
            str(info["layers"]),
            str(info["layer_fp16"]),
            str(info["total_fp16"]),
            str(info["total_int8"]),
            str(info["total_nf4"]),
            r["quantization"] or "none",
            str(r["preload"]),
            f"{r['load_time']:.2f}",
            f"{r['gen_time']:.2f}",
            str(r["tokens"]),
            f"{r['tok_per_s']:.2f}",
        ]

    rows = [fmt_row(r) for r in results]
    col_w = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def print_row(vals: list[str]) -> None:
        print("  ".join(v.rjust(w) for v, w in zip(vals, col_w, strict=True)))

    print_row(headers)
    print_row(["-" * w for w in col_w])
    for row in rows:
        print_row(row)

    # Check consistency per (model, quantization) group
    by_group: dict[tuple[str, str | None], list[dict]] = defaultdict(list)
    for r in results:
        by_group[(r["model"], r["quantization"])].append(r)

    for (model_id, quant), group_results in by_group.items():
        quant_label = quant or "none"
        texts = [r["text"] for r in group_results]
        if len(set(texts)) == 1:
            print(
                f"\n{model_id} (quant={quant_label}): all modes produced identical output."
            )
        else:
            print(
                f"\nWARNING: {model_id} (quant={quant_label}) outputs differ between modes!"
            )
            for r in group_results:
                print(f"  preload={r['preload']}: {r['text']}")


if __name__ == "__main__":
    main()
