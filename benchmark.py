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


def parse_quantization_value(v: str) -> bool | str:
    if v.lower() == "false":
        return False
    return v


def run_once(model_id: str, preload: bool | int, quantization: str | None = None) -> dict:
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(model_id, preload_to_ram=preload, quantization=quantization)
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
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument(
        "--preload",
        default="false",
        choices=["true", "false", "5"],
        help="preload_to_ram values to benchmark (true, false, or integer window sizes)",
    )
    parser.add_argument(
        "--quantization",
        default="false",
        choices=["false", "4bit", "8bit"],
        help="on-the-fly quantization via bitsandbytes (requires bitsandbytes installed)",
    )
    args = parser.parse_args()

    preload_value = parse_preload_value(args.preload)
    quantization_value = parse_quantization_value(args.quantization)

    print(f"Model: {args.model}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Maximum tokens: {MAX_NEW_TOKENS}")
    print(f"Preload: {preload_value}")
    print(f"Quantization: {quantization_value}")
    print()

    result = run_once(args.model, preload=preload_value, quantization=quantization_value)
    print()

    print(f"Output: {result['text']}")
    print(f"Load time (s): {result['load_time']}")
    print(f"Generation time (s): {result['generation_time']}")
    print(f"Tokens per second: {result['tokens_per_second']}")


if __name__ == "__main__":
    main()
