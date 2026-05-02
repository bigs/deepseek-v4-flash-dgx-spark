# Experiment 027: Full Packed Layout Runtime

Date: 2026-05-02

Host: `spark-66c9`

Goal: check whether the full all-expert packed layout is a practical deployment artifact
under the current best runtime settings.

## Command

This repeated the best E026 runtime configuration, but changed the packed layout from the
route-filtered test artifact to the full all-expert layout.

```bash
python3 scripts/guarded_docker_run.py \
  --name full-layout-e027 \
  --image deepseek-v4-flash-spark:tilelang \
  --entrypoint /bin/bash \
  --log-file /home/cole/runs/deepseek-v4-flash/full-layout-e027.log \
  --memory 112g \
  --memory-swap 120g \
  --min-mem-available-gib 12 \
  --timeout-seconds 1800 \
  --mount /home/cole/deepseek-v4-flash-dgx-spark:/repo:ro \
  --mount /home/cole/models/deepseek-v4-flash/hf:/model:ro \
  --mount /home/cole/runs/deepseek-v4-flash:/runs \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env DEEPSEEK_SPARK_MODEL_DIR=/model \
  --env DEEPSEEK_SPARK_MANIFEST_CSV=/repo/results/spark-66c9/weight-manifest.csv \
  --env DEEPSEEK_SPARK_TELEMETRY_JSONL=/runs/full-layout-e027.jsonl \
  --env DEEPSEEK_SPARK_POSTFILL_MODE=deferred \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES=1024 \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_POLICY=layer_lru \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_COUNT=43 \
  --env DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT=/runs/packed-full-e018/layout.json \
  --recover-nvidia-driver-on-exit \
  -- -lc "cd /repo && python3 /repo/scripts/probe_server_http.py --prompt Hello --max-tokens 8 --skip-chat"
```

## Artifacts

- `results/spark-66c9/full-layout-e027.log`
- `results/spark-66c9/full-layout-e027.jsonl`

## Results

| Scenario | Layout | Prefill | Decode | Decode tok/s | Packed read | Materialize | Packed misses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E026 best | route-filtered | 91.662s | 6.434s | 1.24 | 2.157s | 4.322s | 0 |
| E027 | full | 92.612s | 8.283s | 0.97 | 3.964s | 6.190s | 0 |

Full-layout decode step walls:

```text
1.863s, 1.357s, 1.418s, 1.124s, 0.866s, 0.828s, 0.819s, 0.008s
```

The cached continuation reused 9 prompt tokens but still had a `0.851s` one-token
prefill/resume because deferred postfill had not completed.

## Interpretation

The full packed layout is viable and avoids the safetensors fallback path, but it is
slower than the route-filtered artifact on this short probe. That points at physical
layout and filesystem locality as real variables, not just the packed block format.

The next layout work should optimize access locality: per-layer files remain useful, but
hot expert ordering, route-trace clustering, and smaller metadata footprints are now
worth measuring.
