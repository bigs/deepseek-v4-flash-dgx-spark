# Experiment 038: Native Runtime

Date: 2026-05-02

Host: `spark-66c9`

Goal: validate the native packed reader in full inference before adding pinned staging and
direct parameter copies.

## Command

Same route-filtered 8-token probe as E026/E036, plus:

```bash
DEEPSEEK_SPARK_PACKED_NATIVE_LOADER=1
DEEPSEEK_SPARK_NATIVE_BUILD_DIR=/runs/torch-extensions/native-packed-loader
```

## Artifacts

- `results/spark-66c9/native-runtime-e038.log`
- `results/spark-66c9/native-runtime-e038.jsonl`

## Results

| Run | Decode | Tok/s | Packed read | Materialize |
| --- | ---: | ---: | ---: | ---: |
| E036 reusable staging | 6.408s | 1.248 | 2.265s | 4.448s |
| E038 native reader | 5.341s | 1.498 | 1.266s | 3.212s |

Decode step walls:

```text
1.181s, 0.863s, 0.840s, 0.731s, 0.615s, 0.546s, 0.557s, 0.008s
```

## Interpretation

The native reader survives full inference and becomes a clear improvement over reusable
Python staging. The remaining miss time is still split between reads and materialization,
which motivated E039/E040: pinned native staging and direct CPU-to-parameter copies.
