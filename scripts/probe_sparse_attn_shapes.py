#!/usr/bin/env python3
"""Probe official sparse_attn kernel viability for DeepSeek-V4 decode/prefill shapes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", type=Path, default=Path("/model/inference"))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.inference_dir))

    import torch
    from kernel import sparse_attn

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)

    cases = [
        {"name": "decode_window128_top128", "b": 1, "m": 1, "h": 64, "d": 512, "n": 128, "topk": 128},
        {"name": "decode_window128_top512", "b": 1, "m": 1, "h": 64, "d": 512, "n": 128, "topk": 512},
        {"name": "prefill_m8_n8_top128", "b": 1, "m": 8, "h": 64, "d": 512, "n": 8, "topk": 128},
        {"name": "prefill_m16_n16_top128", "b": 1, "m": 16, "h": 64, "d": 512, "n": 16, "topk": 128},
    ]
    results = []
    for case in cases:
        row = dict(case)
        try:
            q = torch.randn(
                case["b"], case["m"], case["h"], case["d"], device="cuda", dtype=torch.bfloat16
            )
            kv = torch.randn(case["b"], case["n"], case["d"], device="cuda", dtype=torch.bfloat16)
            attn_sink = torch.zeros(case["h"], device="cuda", dtype=torch.float32)
            idx = torch.arange(case["topk"], device="cuda", dtype=torch.int32)
            idx = torch.where(idx < case["n"], idx, torch.full_like(idx, -1))
            topk_idxs = idx.view(1, 1, case["topk"]).expand(case["b"], case["m"], case["topk"]).contiguous()
            sparse_attn(q, kv, attn_sink, topk_idxs, 1.0 / (case["d"] ** 0.5))
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.repeat):
                start = time.perf_counter()
                out = sparse_attn(q, kv, attn_sink, topk_idxs, 1.0 / (case["d"] ** 0.5))
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - start)
            row.update(
                {
                    "ok": True,
                    "out_shape": list(out.shape),
                    "median_seconds": sorted(samples)[len(samples) // 2],
                    "samples": samples,
                }
            )
        except Exception as exc:
            row.update({"ok": False, "error": repr(exc)})
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    output = {"repeat": args.repeat, "results": results}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
