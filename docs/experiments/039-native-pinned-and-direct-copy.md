# Experiment 039: Native Pinned Staging and Direct Copy Prep

Date: 2026-05-02

Host: `spark-66c9`

Goal: check whether native tensor staging should use pinned CPU memory before testing a
full runtime run with direct CPU-to-parameter copies.

## Change

Extended `scripts/bench_packed_slab_materialization.py` so `--include-native` compares:

- native pageable CPU tensor staging;
- native pinned CPU tensor staging.

Added runtime controls for the materializer:

```bash
DEEPSEEK_SPARK_DIRECT_PARAM_COPY=1
DEEPSEEK_SPARK_PARAM_COPY_NON_BLOCKING=1
```

`DEEPSEEK_SPARK_DIRECT_PARAM_COPY=1` copies CPU tensor views directly into destination
expert parameters instead of first allocating an intermediate GPU tensor with
`tensor.to(...)` and then copying into the parameter.

## Artifacts

- `results/spark-66c9/native-materializer-e039.log`
- `results/spark-66c9/native-materializer-e039.json`

## Results

64 route-filtered packed blocks, 855,638,016 bytes total, 5 repeats.

| Method | Median elapsed | Throughput |
| --- | ---: | ---: |
| Current bytes/bytearray | 0.650s | 1.23 GiB/s |
| Reusable `preadv` bytearray slab | 0.0997s | 7.99 GiB/s |
| Native pageable tensor | 0.101s | 7.92 GiB/s |
| Native pinned tensor | 0.0851s | 9.36 GiB/s |

## Interpretation

Pinned native staging is worth enabling for full runtime validation. This reverses the
earlier E023 result because the older pinned experiment allocated pinned memory per miss,
while the native path keeps a reusable thread-local tensor.
