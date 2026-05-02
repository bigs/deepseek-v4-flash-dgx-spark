# Experiment 016: Native Kernel Smoke

Date: 2026-05-01

Host: `spark-66c9`

Goal: verify which official native FP8/FP4 DeepSeek kernels can execute on GB10 before
we invest in custom replacements.

## Command

```bash
python3 scripts/guarded_docker_run.py \
  --name official-kernel-smoke-exp016 \
  --image deepseek-v4-flash-spark:tilelang \
  --log-file ~/runs/deepseek-v4-flash/official-kernel-smoke-exp016.log \
  --memory 32g \
  --memory-swap 40g \
  --min-mem-available-gib 32 \
  --timeout-seconds 600 \
  --mount ~/models/deepseek-v4-flash/hf:/model:ro \
  --mount ~/deepseek-v4-flash-dgx-spark:/repo:ro \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 \
  --recover-nvidia-driver-on-exit \
  -- /repo/scripts/smoke_official_kernels.py --inference-dir /model/inference
```

## Artifact

- `results/spark-66c9/official-kernel-smoke-exp016.log`

## Result

The smoke test passed.

Environment:

```text
torch 2.10.0+cu130 cuda 13.0
device NVIDIA GB10 (12, 1)
```

Successful kernels:

| Kernel / Function | Output |
| --- | --- |
| `act_quant` | `torch.float8_e4m3fn` activations + `torch.float32` scale |
| `fp4_act_quant` | `torch.float4_e2m1fn_x2` activations + `torch.float8_e8m0fnu` scale |
| `fp8_gemm` | BF16 output |
| `fp4_gemm` | BF16 output |
| `sparse_attn` small shape | BF16 output |
| `hc_split_sinkhorn` | valid combine weights |

Representative output:

```text
fp8_gemm torch.Size([2, 64]) torch.bfloat16 1620.5224609375
fp4_gemm torch.Size([2, 64]) torch.bfloat16 18.909317016601562
sparse_attn torch.Size([1, 2, 16, 512]) torch.bfloat16 0.1591687649488449
```

## Interpretation

GB10 can execute the official native FP8/FP4 TileLang primitives at small shapes. That
means our immediate bottleneck is not "no native kernels run"; it is the runtime data path
around those kernels:

- repeated expert materialization;
- safetensor tensor-level loading;
- many host-to-device transfers;
- sparse attention shape/shared-memory limits in the full runtime.

The custom-kernel track should therefore be targeted, not broad:

1. Keep using official FP8/FP4 GEMM kernels where they work.
2. Replace the sparse-attention fallback with a GB10-compatible shape-specialized kernel.
3. Write custom expert-load/dequant/fuse kernels only after packed misses and cache policy
   are integrated.

## Next Step

Profile the full-runtime sparse attention fallback separately. It is likely still a
correctness bottleneck for long contexts, but it is not the current largest measured cost
in the short decode traces.

