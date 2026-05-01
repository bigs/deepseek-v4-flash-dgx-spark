#!/usr/bin/env python3
"""Allocate the planned resident tensors for a full-context custom runtime.

This does not load checkpoint values or run inference. It instantiates the
official model on `meta`, then allocates CUDA tensors matching:

- all non-routed-expert parameters,
- all official runtime buffers, including full-context KV/cache buffers,
- an optional per-layer routed-expert scratch budget.

The probe answers whether the residency plan fits before we build the loader.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from inspect_official_model import load_model_module, tensor_nbytes


def gib(nbytes: int) -> float:
    return nbytes / 1024**3


def is_routed_expert_param(name: str) -> bool:
    return ".ffn.experts." in name or ".layers." in name and ".experts." in name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-seq-len", type=int, default=1_048_576)
    parser.add_argument("--expert-scratch-gib", type=float, default=2.0)
    args = parser.parse_args()

    import json
    import torch

    model_mod = load_model_module(args.inference_dir)
    config = json.loads(args.config.read_text())
    config["max_seq_len"] = args.max_seq_len
    model_args = model_mod.ModelArgs(**config)

    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    with torch.device("meta"):
        model = model_mod.Transformer(model_args)

    allocated = []
    planned_bytes = 0
    routed_skipped_bytes = 0

    print(f"cuda_initial_mem_get_info={torch.cuda.mem_get_info()}")
    print(f"max_seq_len={args.max_seq_len}")

    for name, param in model.named_parameters():
        nbytes = tensor_nbytes(param)
        if is_routed_expert_param(name):
            routed_skipped_bytes += nbytes
            continue
        allocated.append(torch.empty_strided(param.shape, param.stride(), dtype=param.dtype, device="cuda"))
        planned_bytes += nbytes

    for _, buffer in model.named_buffers():
        nbytes = tensor_nbytes(buffer)
        allocated.append(torch.empty_strided(buffer.shape, buffer.stride(), dtype=buffer.dtype, device="cuda"))
        planned_bytes += nbytes

    scratch_bytes = int(args.expert_scratch_gib * 1024**3)
    if scratch_bytes:
        allocated.append(torch.empty(scratch_bytes, dtype=torch.uint8, device="cuda"))
        planned_bytes += scratch_bytes

    torch.cuda.synchronize()
    print(f"resident_planned_bytes={planned_bytes} ({gib(planned_bytes):.2f} GiB)")
    print(f"routed_expert_skipped_bytes={routed_skipped_bytes} ({gib(routed_skipped_bytes):.2f} GiB)")
    print(f"cuda_after_allocate_mem_get_info={torch.cuda.mem_get_info()}")

    allocated.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"cuda_after_free_mem_get_info={torch.cuda.mem_get_info()}")


if __name__ == "__main__":
    main()
