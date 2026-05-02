#!/usr/bin/env python3
"""Analyze decode-time optimization ceilings from telemetry JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry_jsonl", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--target-tok-s", type=float, default=20.0)
    return parser.parse_args()


def speed(tokens: int, seconds: float) -> float | None:
    return tokens / seconds if seconds > 0 else None


def main() -> None:
    args = parse_args()
    rows = []
    for line in args.telemetry_jsonl.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") != "engine_generate":
            continue
        tokens = int(event["generated_tokens"])
        if tokens <= 1:
            continue
        timings = event["timings"]
        cache_delta = event["expert_cache"]["delta"]
        decode = float(timings["decode_seconds"])
        read = float(cache_delta.get("packed_read_seconds") or 0.0)
        materialize = float(cache_delta.get("materialize_seconds") or 0.0)
        residual = max(0.0, decode - read - materialize)
        target_decode = tokens / args.target_tok_s
        rows.append(
            {
                "request_id": event["request_id"],
                "tokens": tokens,
                "decode_seconds": decode,
                "observed_tok_s": speed(tokens, decode),
                "packed_read_seconds": read,
                "materialize_seconds": materialize,
                "residual_seconds": residual,
                "packed_read_fraction": read / decode if decode else None,
                "materialize_fraction": materialize / decode if decode else None,
                "residual_fraction": residual / decode if decode else None,
                "tok_s_if_residual_zero": speed(tokens, read + materialize),
                "tok_s_if_materialize_zero": speed(tokens, read + residual),
                "tok_s_if_read_zero": speed(tokens, materialize + residual),
                "tok_s_if_read_and_materialize_halved": speed(
                    tokens, read * 0.5 + materialize * 0.5 + residual
                ),
                "target_tok_s": args.target_tok_s,
                "target_decode_seconds": target_decode,
                "required_speedup": decode / target_decode if target_decode else None,
            }
        )
    output = {
        "telemetry_jsonl": str(args.telemetry_jsonl),
        "target_tok_s": args.target_tok_s,
        "results": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
