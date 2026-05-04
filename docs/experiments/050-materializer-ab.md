# Experiment 050: Materializer A/B

Date: 2026-05-04

Host: `spark-66c9`

Goal: answer whether the native CUDA materializer should be combined with the previous
best full-layout expert arena recipe.

## Change

Added `scripts/run_materializer_ab.py`, a same-container A/B harness that runs repeated
`probe_server_http.py` invocations with different materializer flags while preserving the
same container, page-cache state, model mounts, extension build directory, and driver
state between rows.

The tested sequence was:

```text
arena,cuda,arena,cuda
```

Common runtime recipe:

```bash
DEEPSEEK_SPARK_POSTFILL_MODE=deferred
DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES=1024
DEEPSEEK_SPARK_EXPERT_CACHE_POLICY=layer_lru
DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_COUNT=43
DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT=/runs/packed-full-e018/layout.json
DEEPSEEK_SPARK_PACKED_NATIVE_LOADER=1
DEEPSEEK_SPARK_PACKED_NATIVE_PINNED_STAGING=1
DEEPSEEK_SPARK_DIRECT_PARAM_COPY=1
DEEPSEEK_SPARK_PARAM_COPY_NON_BLOCKING=1
DEEPSEEK_SPARK_NATIVE_MATERIALIZER=1
DEEPSEEK_SPARK_EXPERT_ARENA_SLOTS=256
DEEPSEEK_SPARK_NATIVE_WITH_CUDA=1
```

The `cuda` rows additionally enabled:

```bash
DEEPSEEK_SPARK_NATIVE_COPY_PLAN=1
DEEPSEEK_SPARK_NATIVE_FUSED_MATERIALIZER=1
DEEPSEEK_SPARK_NATIVE_CUDA_MEMCPY=1
```

## Artifacts

- `results/spark-66c9/materializer-ab-e050.guard.log`
- `results/spark-66c9/materializer-ab-e050-summary.md`
- `results/spark-66c9/materializer-ab-e050-summary.csv`
- `results/spark-66c9/materializer-ab-e050-summary.json`
- `results/spark-66c9/materializer-ab-e050-01-arena.jsonl`
- `results/spark-66c9/materializer-ab-e050-02-cuda.jsonl`
- `results/spark-66c9/materializer-ab-e050-03-arena.jsonl`
- `results/spark-66c9/materializer-ab-e050-04-cuda.jsonl`
- matching `.driver.log` files for each row

## Correctness

All four rows generated 32 tokens and produced the same completion token IDs as the first
row. The harness records this as `token_ids_match_first=true` for each row.

## Results

Prefill is intentionally not compared here. The first row paid a cold-ish `92.819s`
prefill, while rows 2-4 were warmed by the same-container sequence and prefilling took
about `4.5s` to `5.0s`. This experiment is a decode-path A/B.

| Run | Variant | Decode | Tok/s | Packed read | Materialize | Packed loads | Arena alloc/reuse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E050-01 | arena | 21.489s | 1.489 | 7.807s | 10.237s | 3239 | 990/2249 |
| E050-02 | cuda | 16.671s | 1.919 | 4.335s | 5.846s | 3239 | 990/2249 |
| E050-03 | arena | 18.593s | 1.721 | 4.598s | 6.352s | 3239 | 990/2249 |
| E050-04 | cuda | 16.239s | 1.971 | 3.767s | 5.246s | 3239 | 990/2249 |

Variant means:

| Variant | Mean decode | Mean tok/s | Mean packed read | Mean materialize |
| --- | ---: | ---: | ---: | ---: |
| arena | 20.041s | 1.605 | 6.203s | 8.295s |
| cuda | 16.455s | 1.945 | 4.051s | 5.546s |

Adjacent paired deltas:

| Pair | Decode win | Packed-read win | Materialize win |
| --- | ---: | ---: | ---: |
| E050-01 arena -> E050-02 cuda | 4.818s / 22.4% | 3.472s / 44.5% | 4.391s / 42.9% |
| E050-03 arena -> E050-04 cuda | 2.354s / 12.7% | 0.832s / 18.1% | 1.106s / 17.4% |

## Interpretation

Yes: the native CUDA materializer and the expert arena should be combined.

This is the cleanest comparison so far because it alternates the two variants inside one
guarded container. Page-cache effects still exist, but the second arena row is the
important control: after a CUDA row has warmed the run, returning to arena-only is still
slower than returning to CUDA.

The best observed 32-token deployable decode row is now E050-04 at `16.239s`, or
`1.971 tok/s`.
That is only a small absolute improvement over the older E043 `16.516s` arena row, but it
is the first controlled result showing the current best arena recipe and the native CUDA
materializer work together.

## Decision

Promote `DEEPSEEK_SPARK_NATIVE_COPY_PLAN=1`,
`DEEPSEEK_SPARK_NATIVE_FUSED_MATERIALIZER=1`, and
`DEEPSEEK_SPARK_NATIVE_CUDA_MEMCPY=1` into the current best recipe.

Next materializer work should start from this combined recipe, then reduce the remaining
per-parameter `cudaMemcpyAsync` calls into fewer larger transfers and batch a layer's
routed expert misses into one native call.
