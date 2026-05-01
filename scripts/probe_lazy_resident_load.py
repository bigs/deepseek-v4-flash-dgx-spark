#!/usr/bin/env python3
"""Instantiate official DeepSeek-V4 with lazy routed experts and load resident weights."""

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
    args = parser.parse_args()

    import torch
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
    print("parameter_count", sum(1 for _ in model.parameters()))
    print("buffer_count", sum(1 for _ in model.buffers()))
    print("mem_after_load", torch.cuda.mem_get_info())
    weight_store.close()


if __name__ == "__main__":
    main()
