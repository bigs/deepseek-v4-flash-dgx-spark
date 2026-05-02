# Experiment 022: Parallel Expert Materialization

Date: 2026-05-01

Host: `spark-66c9`

Goal: test whether routed experts for the current layer can be materialized in parallel
before running the layer's expert compute loop.

## Change

Added an opt-in materialization thread pool:

```bash
DEEPSEEK_SPARK_EXPERT_MATERIALIZE_WORKERS=4
```

When enabled, `LazyMoE.forward()` finds the active experts for the layer after routing and
submits their `materialize()` calls to a `ThreadPoolExecutor` before the normal expert
compute loop.

## Result

This is a regression.

| Scenario | Prefill | Decode | Decode tok/s | Packed read | Materialize |
| --- | ---: | ---: | ---: | ---: | ---: |
| best serial path, no empty-cache | 91.662s | 6.434s | 1.24 | 2.157s | 4.322s |
| 4 materialization workers | 93.517s | 16.170s | 0.49 | 3.100s | 16.721s |

The worker run reported many more cache hits because the pre-materialization step touches
experts before the compute loop sees them, but the actual wall time became much worse.

## Interpretation

Python-threaded CUDA materialization is not the right parallelism boundary. It likely
creates allocator and copy contention while giving us no useful overlap with compute,
because the layer still waits for every active expert before doing the expert matmuls.

Do not enable `DEEPSEEK_SPARK_EXPERT_MATERIALIZE_WORKERS` in the current runtime. A better
E022/E024 design would use a dedicated CPU I/O prefetch stage and a persistent staging
pool, leaving CUDA copies on a controlled stream or moving the whole miss path into a
custom extension.

