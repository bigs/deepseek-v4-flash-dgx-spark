# Experiment 042: Native Materializer V2

Date: 2026-05-03

Host: `spark-66c9`

Goal: move packed-storage slicing and parameter copy scheduling deeper into the native
extension, then measure whether it improves the full-layout 32-token runtime.

## Change

Added native extension function:

```text
copy_storage_to_tensors(storage, targets, offsets, non_blocking)
```

When `DEEPSEEK_SPARK_NATIVE_MATERIALIZER=1` and the packed storage is a native CPU
`torch.uint8` tensor, the runtime avoids constructing Python tensor views for every
parameter before the copy. The extension builds CPU `from_blob` views and copies them
into the target parameter tensors.

## Artifacts

- `results/spark-66c9/matv2-full-t32-e042.log`
- `results/spark-66c9/matv2-full-t32-e042.jsonl`

## Results

Compared against the E041 `full-t32-native-c1024` matrix row:

| Run | Decode | Tok/s | Packed read | Materialize | Arena allocations | Arena reuses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E041 full native c1024 | 21.022s | 1.522 | 6.076s | 9.198s | n/a | n/a |
| E042 materializer v2 | 21.917s | 1.460 | 6.102s | 9.051s | 3239 | 0 |

Cached continuation:

| Tokens | Prefill | Decode | Read | Materialize |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.409s | 0.00825s | 0.041s | 0.062s |

## Interpretation

Native materializer v2 by itself is not a wall-time win. It slightly reduced measured
materialization time, but the full request regressed from 21.022s to 21.917s. Keep the
primitive because it enables lower-Python-copy paths, but do not count it as an
optimization unless combined results prove it.
