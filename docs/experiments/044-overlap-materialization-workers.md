# Experiment 044: Overlap Materialization Workers

Date: 2026-05-03

Host: `spark-66c9`

Goal: test whether Python-threaded expert materialization/prefetch becomes viable after
native packed loading and expert arena reuse.

## Configuration

Same as E043, plus:

```bash
DEEPSEEK_SPARK_EXPERT_MATERIALIZE_WORKERS=2
```

## Artifacts

- `results/spark-66c9/overlap-full-t32-e044.log`
- `results/spark-66c9/overlap-full-t32-e044.jsonl`

## Results

| Run | Decode | Tok/s | Packed read | Materialize | Hits | Loads | Arena reuses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E043 arena 256 | 16.516s | 1.937 | 3.660s | 5.435s | 5017 | 3239 | 2249 |
| E044 workers 2 | 50.415s | 0.635 | 5.201s | 9.827s | 13327 | 3185 | 2194 |

Cached continuation:

| Tokens | Prefill | Decode | Read | Materialize |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.454s | 0.00836s | 0.055s | 0.069s |

## Interpretation

This is a negative result. Python-threaded materialization increased cache hits, but it
made the full request much slower. The extra hits came from prefetch/materialization work
touching experts, not from useful end-to-end overlap.

Do not use `DEEPSEEK_SPARK_EXPERT_MATERIALIZE_WORKERS` in the current runtime. The next
overlap attempt should be below Python: deterministic CPU I/O, explicit staging queues,
CUDA streams, and native handoff.
