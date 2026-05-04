# Experiment 043: Expert Arena

Date: 2026-05-03

Host: `spark-66c9`

Goal: test whether reusing evicted expert module objects reduces allocator and parameter
materialization overhead.

## Change

Added an opt-in expert arena:

```bash
DEEPSEEK_SPARK_EXPERT_ARENA_SLOTS=256
```

When an expert is evicted from the routed-expert cache, the runtime can place the
`torch.nn.Module` in the arena instead of dropping it. A later miss can reuse the module
object and overwrite its parameters from packed storage.

Telemetry now records:

- `arena_allocations`
- `arena_reuses`

## Artifacts

- `results/spark-66c9/arena-full-t32-e043.log`
- `results/spark-66c9/arena-full-t32-e043.jsonl`

## Results

Compared against E042:

| Run | Decode | Tok/s | Packed read | Materialize | Arena allocations | Arena reuses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E042 materializer v2 | 21.917s | 1.460 | 6.102s | 9.051s | 3239 | 0 |
| E043 arena 256 | 16.516s | 1.937 | 3.660s | 5.435s | 990 | 2249 |

Cached continuation:

| Tokens | Prefill | Decode | Read | Materialize | Arena allocations | Arena reuses |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.366s | 0.00824s | 0.036s | 0.044s | 0 | 31 |

## Interpretation

This is the strongest result in the batch. The arena reduced the full-layout 32-token
decode from 21.022s in the E041 matrix to 16.516s, a 21.4% wall-time reduction, while
keeping the deployable full packed layout.

The win is not just Python object allocation. Read accounting also improved, likely
because fewer module/allocator side effects compete with the packed miss path. The next
materializer work should assume an arena is part of the baseline.
