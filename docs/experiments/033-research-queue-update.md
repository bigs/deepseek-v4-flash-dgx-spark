# Experiment 033: Research Queue Update

Date: 2026-05-02

Goal: convert the second flash-moe pass and new measurements into a concrete queue.

## Artifact

- `docs/research-queue.md`

## Updated Priority

1. Native loader/materialization extension.
2. Route-trace replay as the screening gate.
3. Threaded or async packed reads below Python.
4. Layout locality sweep.
5. Persistent KV postfill semantics.
6. GB10 sparse-attention kernel.

## Experiment Coverage

| Queue option | Evidence |
| --- | --- |
| Native loader/materialization extension | E032 slab microbenchmark and E036 full reusable-staging runtime. |
| Route-trace replay as screening gate | E031 replay harness and cache/read-order variants. |
| Threaded or async packed reads below Python | E031 threaded replay variant. |
| Layout locality sweep | E031 offset-order replay variant; full physical reordering remains next. |
| Persistent KV postfill semantics | E028 inline postfill result; use deferred postfill plus wait/reuse boundary. |
| GB10 sparse-attention kernel | E025 production-shape failure plus E034 budget analysis. |

## Interpretation

The kernel question is narrower now. We should write custom GB10 code, but the next custom
code should target expert loading/materialization first. Broad FP8/FP4 GEMM replacement is
not justified by current measurements.
