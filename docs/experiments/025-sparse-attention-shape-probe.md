# Experiment 025: Sparse Attention Shape Probe

Date: 2026-05-01

Host: `spark-66c9`

Goal: check whether the official Tilelang `sparse_attn` kernel can run at real
DeepSeek-V4 attention shapes on GB10.

## Change

Added `scripts/probe_sparse_attn_shapes.py`.

The probe imports the official `/model/inference/kernel.py` implementation and tests
decode/prefill-like shapes with `h=64` and `d=512`.

## Command

```bash
python3 scripts/guarded_docker_run.py \
  --name sparse-shapes-e025 \
  --image deepseek-v4-flash-spark:tilelang \
  --entrypoint /bin/bash \
  --log-file /home/cole/runs/deepseek-v4-flash/sparse-shapes-e025.log \
  --memory 48g \
  --memory-swap 56g \
  --min-mem-available-gib 24 \
  --timeout-seconds 900 \
  --mount /home/cole/deepseek-v4-flash-dgx-spark:/repo:ro \
  --mount /home/cole/models/deepseek-v4-flash/hf:/model:ro \
  --mount /home/cole/runs/deepseek-v4-flash:/runs \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --recover-nvidia-driver-on-exit \
  -- -lc "cd /repo && python3 /repo/scripts/probe_sparse_attn_shapes.py --inference-dir /model/inference --out-json /runs/sparse-shapes-e025.json --repeat 3"
```

## Result

All tested real-shape cases failed:

- `decode_window128_top128`
- `decode_window128_top512`
- `prefill_m8_n8_top128`
- `prefill_m16_n16_top128`

Failure:

```text
InternalError('Failed to set the allowed dynamic shared memory size to 141312')
```

## Interpretation

The official sparse attention kernel can run at the small smoke-test shape from
Experiment 016, but it does not run at the production `h=64,d=512` shape on GB10. The
runtime should keep the correctness-first fallback until we write a GB10-specific sparse
attention kernel with a smaller shared-memory footprint.

