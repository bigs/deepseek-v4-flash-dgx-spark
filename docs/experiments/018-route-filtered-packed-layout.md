# Experiment 018: Route-Filtered Packed Expert Layout

Date: 2026-05-01

Host: `spark-66c9`

Goal: build fixed-offset packed expert layouts for both fast route-specific validation
and full all-expert deployment.

## Change

Added `scripts/build_packed_expert_layout.py`.

The builder reads `results/spark-66c9/weight-manifest.csv` and writes fixed-offset
per-layer binary files plus `layout.json`. Each expert block stores the six routed
expert tensors with target-name metadata so the runtime can reconstruct the official
parameters without a safetensors lookup on the miss path.

The script supports:

- full all-layer/all-expert packing;
- layer and expert filters;
- `--route-jsonl` to pack only experts observed in a route trace;
- `--top-per-layer` to cap a trace-derived working set.

## Command

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/build_packed_expert_layout.py \
  --model-dir /home/cole/models/deepseek-v4-flash/hf \
  --manifest-csv /home/cole/deepseek-v4-flash-dgx-spark/results/spark-66c9/weight-manifest.csv \
  --route-jsonl /home/cole/deepseek-v4-flash-dgx-spark/results/spark-66c9/route-cache1024-t8-routes.jsonl \
  --out-dir /home/cole/runs/deepseek-v4-flash/packed-route-e018 \
  --overwrite
```

## Route-Filtered Result

- Packed 1,232 observed routed experts across 43 layers.
- Output size: `16,471,031,808` bytes, about `15.3 GiB`.
- Final layer completed at about `8.4s` elapsed.
- Layout file: `/home/cole/runs/deepseek-v4-flash/packed-route-e018/layout.json`.

## Full Result

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/build_packed_expert_layout.py \
  --model-dir /home/cole/models/deepseek-v4-flash/hf \
  --manifest-csv /home/cole/deepseek-v4-flash-dgx-spark/results/spark-66c9/weight-manifest.csv \
  --out-dir /home/cole/runs/deepseek-v4-flash/packed-full-e018 \
  --overwrite
```

Result:

- Packed all 11,008 routed experts across 43 layers.
- Output size: `147,169,738,752` bytes, about `137.1 GiB`.
- Directory size on disk: `138G`.
- Final layer completed at about `79.3s` elapsed.
- Layout file: `/home/cole/runs/deepseek-v4-flash/packed-full-e018/layout.json`.
- Layout metadata size: `21M`.

## Interpretation

The builder is fast enough to use iteratively. A route-filtered working set is useful for
runtime validation and A/B measurements, and the full all-expert packed artifact is
practical on Spark local storage.
