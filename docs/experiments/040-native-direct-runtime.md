# Experiment 040: Native Direct Runtime

Date: 2026-05-02

Host: `spark-66c9`

Goal: validate the native loader/materializer stack in real 8-token inference.

## Command

Same route-filtered 8-token probe as E026/E036, plus:

```bash
DEEPSEEK_SPARK_PACKED_NATIVE_LOADER=1
DEEPSEEK_SPARK_PACKED_NATIVE_PINNED_STAGING=1
DEEPSEEK_SPARK_DIRECT_PARAM_COPY=1
DEEPSEEK_SPARK_PARAM_COPY_NON_BLOCKING=1
DEEPSEEK_SPARK_NATIVE_BUILD_DIR=/runs/torch-extensions/native-packed-loader
```

## Artifacts

- `results/spark-66c9/native-runtime-e038.log`
- `results/spark-66c9/native-runtime-e038.jsonl`
- `results/spark-66c9/native-direct-e040.log`
- `results/spark-66c9/native-direct-e040.jsonl`

## Results

| Run | Extra flags | Prefill | Decode | Tok/s | Packed read | Materialize |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E026 no empty cache | none | 91.662s | 6.434s | 1.243 | 2.157s | 4.322s |
| E036 reusable staging | reusable `preadv` | 92.056s | 6.408s | 1.248 | 2.265s | 4.448s |
| E038 native reader | native pageable tensor | 91.345s | 5.341s | 1.498 | 1.266s | 3.212s |
| E040 native direct | pinned native + direct copy | 91.760s | 5.036s | 1.589 | 1.120s | 2.608s |

E040 decode step walls:

```text
1.083s, 0.811s, 0.801s, 0.693s, 0.586s, 0.520s, 0.533s, 0.008s
```

The cached continuation also improved:

| Run | Continuation prefill | Continuation decode |
| --- | ---: | ---: |
| E038 native reader | 0.739s | 0.0083s |
| E040 native direct | 0.534s | 0.0082s |

## Interpretation

The native loader/materializer is the new best route-filtered 8-token recipe. Compared
with E036, decode improved from 6.408s to 5.036s, a 21.4% wall-time reduction. Packed
read accounting fell by about 50.5%, and materialization accounting fell by about 41.4%.

We are still far from 20 tok/s, but the direction is correct: the fastest path came from
cutting Python allocation and redundant GPU staging in the miss path, not from changing
the model math.
