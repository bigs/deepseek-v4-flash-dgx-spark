# Experiment 026: Allocator Cache on Expert Eviction

Date: 2026-05-01

Host: `spark-66c9`

Goal: test whether `torch.cuda.empty_cache()` on every expert eviction is suppressing
the packed miss path.

## Change

Changed `LazyRoutedExpert._drop_cached()` so cached-expert evictions do not call
`torch.cuda.empty_cache()` by default.

The new environment control is:

```bash
DEEPSEEK_SPARK_EMPTY_CACHE_ON_EVICT=uncached
```

Modes:

- `uncached`: default; keep old empty-cache behavior only when the expert cache is off.
- `always`, `true`, or `1`: force old empty-cache behavior for every eviction.
- any other value: do not empty the CUDA allocator cache on eviction.

## Command

The run repeated the best prior configuration, layer-LRU cache with 1024 entries and the
route-filtered packed layout.

```bash
python3 scripts/guarded_docker_run.py \
  --name no-emptycache-e026 \
  --image deepseek-v4-flash-spark:tilelang \
  --entrypoint /bin/bash \
  --log-file /home/cole/runs/deepseek-v4-flash/no-emptycache-e026.log \
  --memory 112g \
  --memory-swap 120g \
  --min-mem-available-gib 12 \
  --timeout-seconds 1800 \
  --mount /home/cole/deepseek-v4-flash-dgx-spark:/repo:ro \
  --mount /home/cole/models/deepseek-v4-flash/hf:/model:ro \
  --mount /home/cole/runs/deepseek-v4-flash:/runs \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env DEEPSEEK_SPARK_MANIFEST_CSV=/repo/results/spark-66c9/weight-manifest.csv \
  --env DEEPSEEK_SPARK_TELEMETRY_JSONL=/runs/no-emptycache-e026.jsonl \
  --env DEEPSEEK_SPARK_POSTFILL_MODE=deferred \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES=1024 \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_POLICY=layer_lru \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_COUNT=43 \
  --env DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT=/runs/packed-route-e018/layout.json \
  --recover-nvidia-driver-on-exit \
  -- -lc "cd /repo && python3 /repo/scripts/probe_server_http.py --prompt Hello --max-tokens 8 --skip-chat"
```

## Artifacts

- `results/spark-66c9/no-emptycache-e026.log`
- `results/spark-66c9/no-emptycache-e026.jsonl`

## Results

| Scenario | Prefill | Decode | Decode tok/s | Packed read | Materialize |
| --- | ---: | ---: | ---: | ---: | ---: |
| layer-LRU 1024, empty cache on eviction | 94.988s | 12.750s | 0.63 | 9.308s | 11.160s |
| layer-LRU 1024, keep allocator cache | 91.662s | 6.434s | 1.24 | 2.157s | 4.322s |

Decode step walls with allocator caching:

```text
1.441s, 1.044s, 1.001s, 0.885s, 0.734s, 0.658s, 0.663s, 0.008s
```

The cached continuation also improved: its single-token prefill/resume path fell from
`1.916s` to `0.689s`.

## Interpretation

This crossed the sprint target. The current best observed decode rate is about
`1.24 tok/s` on the 8-token probe.

The lesson is sharp: on GB10 unified memory, repeatedly forcing CUDA allocator cache
release destroys the packed streaming path. We should keep allocator-reserved memory hot
while using our own expert-cache accounting and the guarded process memory limits for
safety.

