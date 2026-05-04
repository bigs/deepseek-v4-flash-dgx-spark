# Experiment 046: Hot-First Layout Runtime

Date: 2026-05-03

Host: `spark-66c9`

Goal: measure the E045 hot-first full packed layout in the current best full-layout
runtime recipe.

## Configuration

```bash
DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT=/runs/packed-full-hot-e045/layout.json
DEEPSEEK_SPARK_PACKED_NATIVE_LOADER=1
DEEPSEEK_SPARK_PACKED_NATIVE_PINNED_STAGING=1
DEEPSEEK_SPARK_DIRECT_PARAM_COPY=1
DEEPSEEK_SPARK_PARAM_COPY_NON_BLOCKING=1
DEEPSEEK_SPARK_NATIVE_MATERIALIZER=1
DEEPSEEK_SPARK_EXPERT_ARENA_SLOTS=256
DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES=1024
DEEPSEEK_SPARK_EXPERT_CACHE_POLICY=layer_lru
DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_COUNT=43
```

## Artifacts

- `results/spark-66c9/hot-full-t32-e046.log`
- `results/spark-66c9/hot-full-t32-e046.jsonl`

## Results

| Run | Layout | Decode | Tok/s | Packed read | Materialize | Packed misses |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E043 | original full | 16.516s | 1.937 | 3.660s | 5.435s | 0 |
| E046 | hot-first full | 30.899s | 1.036 | 16.096s | 18.482s | 0 |

The hot-first run had the same arena behavior as E043:

| Arena allocations | Arena reuses | Loads | Hits | Evictions |
| ---: | ---: | ---: | ---: | ---: |
| 990 | 2249 | 3239 | 5017 | 2250 |

## Interpretation

The hot-first physical layout is not the next recipe. It regressed the 32-token decode by
87.1% versus E043 while preserving the same cache and arena behavior. The regression is
visible in packed read and materialization accounting, so future layout experiments need
longer route traces, replay screening, and explicit cache-state controls before full
inference runs.
