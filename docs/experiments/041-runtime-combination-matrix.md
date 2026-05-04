# Experiment 041: Runtime Combination Matrix

Date: 2026-05-03

Host: `spark-66c9`

Goal: run the controlled runtime combination matrix instead of comparing one-off runs
from different days and cache states.

## Matrix

Axes:

- layout: route-filtered packed layout, full all-expert packed layout
- generated tokens: 8, 32
- packed loader: reusable Python staging, native pinned/direct-copy loader
- expert cache entries: 1024, 2048

Common settings:

```bash
DEEPSEEK_SPARK_POSTFILL_MODE=deferred
DEEPSEEK_SPARK_EXPERT_CACHE_POLICY=layer_lru
DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_COUNT=43
DEEPSEEK_SPARK_NATIVE_BUILD_DIR=/runs/torch-extensions/native-packed-loader
```

Native-loader scenarios additionally used:

```bash
DEEPSEEK_SPARK_PACKED_NATIVE_LOADER=1
DEEPSEEK_SPARK_PACKED_NATIVE_PINNED_STAGING=1
DEEPSEEK_SPARK_DIRECT_PARAM_COPY=1
DEEPSEEK_SPARK_PARAM_COPY_NON_BLOCKING=1
```

## Artifacts

- `results/spark-66c9/matrix-e041-manifest.json`
- `results/spark-66c9/matrix-e041-driver.log`
- `results/spark-66c9/matrix-e041-summary.json`
- `results/spark-66c9/matrix-e041-summary.csv`
- `results/spark-66c9/matrix-e041-summary.md`
- `results/spark-66c9/matrix-*.jsonl`
- `results/spark-66c9/matrix-*.log`

## Results

See `results/spark-66c9/matrix-e041-summary.md` for the complete table. The best
deployable rows were:

| Run | Layout | Tokens | Cache | Decode | Tok/s | Read | Materialize | Packed misses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full-t8-native-c1024` | full | 8 | 1024 | 5.213s | 1.535 | 1.343s | 2.839s | 0 |
| `full-t32-native-c1024` | full | 32 | 1024 | 21.022s | 1.522 | 6.076s | 9.198s | 0 |
| `full-t32-reuse-c2048` | full | 32 | 2048 | 23.231s | 1.377 | 5.818s | 11.171s | 0 |

The route-filtered layout remained useful for short-route probes, but it failed the
32-token deployment test:

| Run | Decode | Tok/s | Packed misses | Materialize |
| --- | ---: | ---: | ---: | ---: |
| `route-t32-native-c1024` | 183.245s | 0.175 | 1438 | 170.807s |
| `route-t32-reuse-c1024` | 195.935s | 0.163 | 1438 | 183.692s |
| `route-t32-native-c2048` | 182.333s | 0.176 | 1319 | 168.351s |
| `route-t32-reuse-c2048` | 189.792s | 0.169 | 1319 | 177.187s |

## Interpretation

The same-day matrix replaced the previous one-off best of 5.036s for 8 tokens with a
more conservative controlled best of 5.213s. That is the number to compare against for
future matrix runs.

The full packed layout is mandatory for publishable deployment claims. The route-filtered
layout can silently fall back to safetensors for unseen routed experts, which destroys the
32-token run.

The 2048-entry cache did not automatically win. It reduced evictions, but the 32-token
native full-layout run regressed from 21.022s at 1024 entries to 24.982s at 2048 entries.
The next cache work should be policy and object reuse, not simply making the cache bigger.
