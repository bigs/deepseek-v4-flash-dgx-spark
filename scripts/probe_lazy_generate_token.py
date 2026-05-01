#!/usr/bin/env python3
"""Generate one token id with the lazy-expert official DeepSeek-V4 runtime."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--inference-dir", required=True, type=Path)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-seq-len", type=int, default=1_048_576)
    parser.add_argument("--input-token-id", type=int, default=0)
    parser.add_argument("--prompt", help="Optional text prompt. Overrides --input-token-id.")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer
    from lazy_official_runtime import build_lazy_model

    print("mem_initial", torch.cuda.mem_get_info())
    model, model_args, counts, weight_store = build_lazy_model(
        args.model_dir,
        args.inference_dir,
        args.manifest_csv,
        args.config,
        args.max_seq_len,
    )
    print("max_seq_len", model_args.max_seq_len)
    print("load_counts", counts)
    print("mem_after_load", torch.cuda.mem_get_info())

    torch.set_default_device("cuda")
    tokenizer = None
    if args.prompt is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        prompt_token_ids = tokenizer.encode(args.prompt)
    else:
        prompt_token_ids = [args.input_token_id]

    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device="cuda")
    generated: list[int] = []
    with torch.inference_mode():
        logits = model(input_ids, 0)
        for step in range(args.max_new_tokens):
            next_token = int(logits.argmax(dim=-1).item())
            generated.append(next_token)
            if step + 1 < args.max_new_tokens:
                decode_input = torch.tensor([[next_token]], dtype=torch.long, device="cuda")
                logits = model(decode_input, len(prompt_token_ids) + step)
    print("prompt", repr(args.prompt) if args.prompt is not None else None)
    print("input_token_ids", prompt_token_ids)
    print("generated_token_ids", generated)
    if tokenizer is not None:
        print("generated_text", repr(tokenizer.decode(generated)))
        print("prompt_plus_generated_text", repr(tokenizer.decode(prompt_token_ids + generated)))
    print("logits_shape", tuple(logits.shape), logits.dtype)
    print("mem_after_forward", torch.cuda.mem_get_info())
    weight_store.close()


if __name__ == "__main__":
    main()
