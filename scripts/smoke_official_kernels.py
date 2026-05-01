#!/usr/bin/env python3
"""Smoke test official DeepSeek-V4 TileLang kernels on CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", required=True, type=Path)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(args.inference_dir))

    import torch
    from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("mem_initial", torch.cuda.mem_get_info())

    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    q, s = act_quant(x, 128)
    print("act_quant", q.shape, q.dtype, s.shape, s.dtype)

    x4 = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    q4, s4 = fp4_act_quant(x4, 32)
    print("fp4_act_quant", q4.shape, q4.dtype, s4.shape, s4.dtype)

    a, a_s = act_quant(torch.randn(2, 128, device="cuda", dtype=torch.bfloat16), 128)
    b, _ = act_quant(torch.randn(64, 128, device="cuda", dtype=torch.bfloat16), 128)
    b_s = torch.ones(1, 1, device="cuda", dtype=torch.float32)
    y = fp8_gemm(a, a_s, b, b_s)
    print("fp8_gemm", y.shape, y.dtype, float(y.float().abs().mean()))

    a4, a4_s = act_quant(
        torch.randn(2, 128, device="cuda", dtype=torch.bfloat16),
        128,
        "ue8m0",
        torch.float8_e8m0fnu,
    )
    b4, _ = fp4_act_quant(torch.randn(64, 128, device="cuda", dtype=torch.bfloat16), 32)
    b4_s = torch.ones(64, 4, device="cuda", dtype=torch.float8_e8m0fnu)
    y4 = fp4_gemm(a4, a4_s, b4, b4_s, torch.float8_e8m0fnu)
    print("fp4_gemm", y4.shape, y4.dtype, float(y4.float().abs().mean()))

    q = torch.randn(1, 2, 16, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, 64, 512, device="cuda", dtype=torch.bfloat16)
    attn_sink = torch.zeros(16, device="cuda", dtype=torch.float32)
    topk_idxs = torch.arange(64, device="cuda", dtype=torch.int32).view(1, 1, 64).expand(1, 2, 64).contiguous()
    o = sparse_attn(q, kv, attn_sink, topk_idxs, 512**-0.5)
    print("sparse_attn", o.shape, o.dtype, float(o.float().abs().mean()))

    mixes = torch.randn(1, 2, 24, device="cuda", dtype=torch.float32)
    hc_scale = torch.ones(3, device="cuda", dtype=torch.float32)
    hc_base = torch.zeros(24, device="cuda", dtype=torch.float32)
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, 4, 2, 1e-6)
    print("hc_split_sinkhorn", pre.shape, post.shape, comb.shape, float(comb.sum(-1).mean()))

    torch.cuda.synchronize()
    print("mem_final", torch.cuda.mem_get_info())


if __name__ == "__main__":
    main()
