# Experiment 021: Layer-Aware Expert Cache

Date: 2026-05-01

Host: `spark-66c9`

Goal: reduce decode-time expert churn by preventing late routed layers from evicting
early routed layers before the next token reaches them.

## Change

Added an opt-in cache policy:

```bash
DEEPSEEK_SPARK_EXPERT_CACHE_POLICY=layer_lru
DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_COUNT=43
DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_QUOTA=24
```

If `DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_QUOTA` is not set, the default quota is:

```text
floor(DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES / DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_COUNT)
```

The old behavior remains the default:

```bash
DEEPSEEK_SPARK_EXPERT_CACHE_POLICY=global_lru
```

## Rationale

The decode loop visits routed layers in order for every generated token. A single global
LRU can fill with late-layer experts and evict early-layer experts that will be needed
again at the start of the next token. A per-layer quota should trade a little total cache
flexibility for much lower cross-layer churn.

## Command

```bash
python3 scripts/guarded_docker_run.py \
  --name layer-lru-e021 \
  --image deepseek-v4-flash-spark:tilelang \
  --entrypoint /bin/bash \
  --log-file /home/cole/runs/deepseek-v4-flash/layer-lru-e021.log \
  --memory 104g \
  --memory-swap 112g \
  --min-mem-available-gib 16 \
  --timeout-seconds 1800 \
  --mount /home/cole/deepseek-v4-flash-dgx-spark:/repo:ro \
  --mount /home/cole/models/deepseek-v4-flash/hf:/model:ro \
  --mount /home/cole/runs/deepseek-v4-flash:/runs \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env DEEPSEEK_SPARK_MANIFEST_CSV=/repo/results/spark-66c9/weight-manifest.csv \
  --env DEEPSEEK_SPARK_TELEMETRY_JSONL=/runs/layer-lru-e021.jsonl \
  --env DEEPSEEK_SPARK_POSTFILL_MODE=deferred \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES=1024 \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_POLICY=layer_lru \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_COUNT=43 \
  --env DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT=/runs/packed-route-e018/layout.json \
  -- -lc "cd /repo && python3 /repo/scripts/probe_server_http.py --prompt Hello --max-tokens 8 --skip-chat"
```

## Artifacts

- `results/spark-66c9/layer-lru-e021.log`
- `results/spark-66c9/layer-lru-e021.jsonl`

## Results

| Scenario | Prefill | Decode | Decode step walls |
| --- | ---: | ---: | --- |
| `packed-route-e020` global LRU | 95.067s | 15.642s | 2.448s, 1.421s, 2.796s, 2.791s, 2.274s, 2.063s, 1.840s, 0.008s |
| `layer-lru-e021` layer LRU | 94.988s | 12.750s | 1.608s, 1.071s, 2.451s, 2.381s, 1.887s, 1.691s, 1.653s, 0.008s |

Decode throughput for the 8 generated tokens improved from about `0.51 tok/s` to about
`0.63 tok/s`.

The layer-aware run used the default quota of 23 experts per layer:

- final entries after first completion: 961
- cache hits: 913
- cache misses: 1,151
- evictions: 190
- packed read seconds: 9.308s
- materialize seconds: 11.160s

## Interpretation

Layer-aware eviction helped, but not by eliminating misses. The miss count was similar
to global LRU, and evictions were actually higher because the default 23/layer quota
kept the cache below the nominal 1024-entry budget. The improvement appears to come from
less damaging cross-layer churn and somewhat lower materialization overhead.

The next cache experiment should raise the budget to 2048 entries, where this short route
should fit without evictions.
