#!/usr/bin/env python
"""Benchmark Chiquito inference across different preload_to_ram modes."""

import argparse
import time

import torch

from chiquito import AutoModel


PROMPT = "The reason why we need local AI is"
MAX_NEW_TOKENS = 20


def parse_preload_value(v: str) -> bool | int:
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    return int(v)


def run_once(model_id: str, preload: bool | int, quantization: str | None = None) -> dict:
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(model_id, preload_to_ram=preload, quantization=quantization)
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
        "preload": preload,
        "load_time": load_time,
        "gen_time": gen_time,
        "tokens": n_tokens,
        "tok_per_s": n_tokens / gen_time if gen_time > 0 else 0,
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument(
        "--preload",
        nargs="+",
        default=["true", "false", "5"],
        help="preload_to_ram values to benchmark (true, false, or integer window sizes)",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        choices=["4bit", "8bit"],
        help="on-the-fly quantization via bitsandbytes (requires bitsandbytes installed)",
    )
    args = parser.parse_args()

    preload_values = [parse_preload_value(v) for v in args.preload]

    print(f"Model:  {args.model}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Modes:  {preload_values}")
    if args.quantization:
        print(f"Quant:  {args.quantization}")
    print()

    results = []
    for preload in preload_values:
        print(f"--- preload_to_ram={preload} ---")
        r = run_once(args.model, preload, quantization=args.quantization)
        results.append(r)
        print(f"    Output: {r['text']}")
        print()

    # Print table
    hdr_preload = "preload_to_ram"
    hdr_load = "load (s)"
    hdr_gen = "gen (s)"
    hdr_toks = "tokens"
    hdr_tps = "tok/s"

    col_w = [
        max(len(hdr_preload), max(len(str(r["preload"])) for r in results)),
        max(len(hdr_load), 8),
        max(len(hdr_gen), 8),
        max(len(hdr_toks), 6),
        max(len(hdr_tps), 8),
    ]

    def row(vals: list[str]):
        return "  ".join(v.rjust(w) for v, w in zip(vals, col_w))

    print(row([hdr_preload, hdr_load, hdr_gen, hdr_toks, hdr_tps]))
    print(row(["-" * w for w in col_w]))
    for r in results:
        print(
            row([
                str(r["preload"]),
                f"{r['load_time']:.2f}",
                f"{r['gen_time']:.2f}",
                str(r["tokens"]),
                f"{r['tok_per_s']:.2f}",
            ])
        )

    # Check consistency
    texts = [r["text"] for r in results]
    if len(set(texts)) == 1:
        print("\nAll modes produced identical output.")
    else:
        print("\nWARNING: outputs differ between modes!")
        for r in results:
            print(f"  preload_to_ram={r['preload']}: {r['text']}")


if __name__ == "__main__":
    main()
