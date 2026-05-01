#!/usr/bin/env python3
"""Print Python/runtime support relevant to DeepSeek-V4-Flash on DGX Spark."""

from __future__ import annotations

import importlib
import sys


def main() -> None:
    print("python", sys.version.replace("\n", " "))
    for name in ["torch", "vllm", "sglang", "transformers", "safetensors", "tilelang", "fast_hadamard_transform"]:
        try:
            mod = importlib.import_module(name)
            print(name, getattr(mod, "__version__", "ok"))
        except Exception as exc:
            print(name, "MISSING", repr(exc))

    try:
        import torch

        print("torch_cuda_available", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("torch_cuda_device_name", torch.cuda.get_device_name(0))
            print("torch_cuda_capability", torch.cuda.get_device_capability(0))
        for dtype_name in [
            "float4_e2m1fn_x2",
            "float8_e4m3fn",
            "float8_e8m0fnu",
            "float8_e5m2",
        ]:
            print(f"torch_has_{dtype_name}", hasattr(torch, dtype_name), getattr(torch, dtype_name, None))
    except Exception as exc:
        print("torch_probe_error", repr(exc))


if __name__ == "__main__":
    main()
