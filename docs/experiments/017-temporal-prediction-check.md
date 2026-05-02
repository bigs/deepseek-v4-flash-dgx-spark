# Experiment 017: Temporal Prediction Check

Date: 2026-05-01

Input trace: `results/spark-66c9/route-cache1024-t8-routes.jsonl`

Goal: test whether the simplest speculative prefetch policy is promising enough to
implement now: predict each layer's next routed experts from that layer's previous routed
experts.

## Method

For every layer in the Experiment 012 route trace:

1. Treat the previous event's routed expert set as the prediction.
2. Compare it with the current event's routed expert set.
3. Compute:
   - slot hit rate: fraction of actual routed expert slots found in the previous set;
   - all-hit rate: fraction of layer events where every actual routed expert was present
     in the previous set.

The local analysis output is:

- `results/spark-66c9/temporal-prediction-exp017.json`

## Result

Overall:

| Metric | Value |
| --- | ---: |
| layer-to-layer prediction pairs | 344 |
| average routed experts per event | 6 |
| slot hit rate | 38.8% |
| all-hit rate | 0.0% |

Early layers are especially poor:

| Layer | Slot Hit Rate | All-Hit Rate |
| ---: | ---: | ---: |
| 0 | 4.2% | 0.0% |
| 1 | 4.2% | 0.0% |
| 2 | 4.2% | 0.0% |

Some middle layers are better but still not good enough for exact prefetch:

| Layer | Slot Hit Rate | All-Hit Rate |
| ---: | ---: | ---: |
| 3 | 54.2% | 0.0% |
| 4 | 45.8% | 0.0% |
| 9 | 47.9% | 0.0% |

## Interpretation

This reproduces the Flash-MoE lesson in our own trace: naive temporal prediction is not a
good next step.

A 38.8% slot hit rate means most prefetched expert blocks would not be used. A 0% all-hit
rate means the runtime would still need synchronous miss handling on every layer event.
That is exactly the failure mode Flash-MoE documented: imperfect prediction can waste
storage and memory bandwidth while not removing the critical wait.

## Decision

Do not implement speculative expert prediction yet.

The better combination is:

1. packed expert files to make misses cheap;
2. layer-aware expert cache to reduce repeated misses;
3. deterministic parallel reads for remaining misses;
4. only then revisit prediction with a better model and strict end-to-end tokens/sec
   acceptance criteria.

