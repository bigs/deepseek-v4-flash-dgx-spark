#!/usr/bin/env python3
"""Inspect official DeepSeek-V4 inference model shapes without importing kernels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from collections import Counter
from pathlib import Path


def install_kernel_stub() -> None:
    kernel = types.ModuleType("kernel")
    for name in [
        "act_quant",
        "fp4_act_quant",
        "fp8_gemm",
        "fp4_gemm",
        "sparse_attn",
        "hc_split_sinkhorn",
    ]:
        setattr(kernel, name, lambda *args, **kwargs: None)
    sys.modules["kernel"] = kernel


def load_model_module(inference_dir: Path):
    install_kernel_stub()
    spec = importlib.util.spec_from_file_location("deepseek_v4_official_model", inference_dir / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load model.py from {inference_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_nbytes(tensor) -> int:
    return tensor.numel() * tensor.element_size()


def gib(nbytes: int) -> float:
    return nbytes / 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-seq-len", type=int, help="Override config max_seq_len.")
    parser.add_argument(
        "--torch-default-dtype",
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="Default dtype active while instantiating the official model.",
    )
    args = parser.parse_args()

    model_mod = load_model_module(args.inference_dir)
    config = json.loads(args.config.read_text())
    if args.max_seq_len is not None:
        config["max_seq_len"] = args.max_seq_len
    model_args = model_mod.ModelArgs(**config)

    import torch

    torch.set_default_dtype(
        {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[args.torch_default_dtype]
    )
    with torch.device("meta"):
        model = model_mod.Transformer(model_args)

    param_bytes = 0
    buffer_bytes = 0
    param_dtypes: Counter[str] = Counter()
    buffer_dtypes: Counter[str] = Counter()
    param_prefixes: Counter[str] = Counter()
    buffer_prefixes: Counter[str] = Counter()

    for name, param in model.named_parameters():
        nbytes = tensor_nbytes(param)
        param_bytes += nbytes
        param_dtypes[str(param.dtype)] += nbytes
        param_prefixes[name.split(".")[0]] += nbytes

    for name, buffer in model.named_buffers():
        nbytes = tensor_nbytes(buffer)
        buffer_bytes += nbytes
        buffer_dtypes[str(buffer.dtype)] += nbytes
        buffer_prefixes[name.split(".")[0]] += nbytes

    print(f"max_seq_len={model.max_seq_len}")
    print(f"n_layers={len(model.layers)}")
    print(f"torch_default_dtype={torch.get_default_dtype()}")
    print(f"parameter_logical_bytes={param_bytes} ({gib(param_bytes):.2f} GiB)")
    print(f"buffer_logical_bytes={buffer_bytes} ({gib(buffer_bytes):.2f} GiB)")
    print("parameter_bytes_by_dtype:")
    for dtype, nbytes in param_dtypes.most_common():
        print(f"  {dtype}: {nbytes} ({gib(nbytes):.2f} GiB)")
    print("buffer_bytes_by_dtype:")
    for dtype, nbytes in buffer_dtypes.most_common():
        print(f"  {dtype}: {nbytes} ({gib(nbytes):.2f} GiB)")
    print("parameter_bytes_by_top_prefix:")
    for prefix, nbytes in param_prefixes.most_common():
        print(f"  {prefix}: {nbytes} ({gib(nbytes):.2f} GiB)")
    print("buffer_bytes_by_top_prefix:")
    for prefix, nbytes in buffer_prefixes.most_common():
        print(f"  {prefix}: {nbytes} ({gib(nbytes):.2f} GiB)")

    print("layer_buffer_bytes:")
    for idx, layer in enumerate(model.layers):
        layer_bytes = sum(tensor_nbytes(buffer) for _, buffer in layer.named_buffers())
        print(f"  layer.{idx}: {layer_bytes} ({gib(layer_bytes):.3f} GiB)")


if __name__ == "__main__":
    main()
