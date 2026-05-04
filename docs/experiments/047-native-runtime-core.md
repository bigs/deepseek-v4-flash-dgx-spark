# Experiment 047: Native Runtime Core

Date: 2026-05-04

Host: `spark-66c9`

Goal: move the packed expert miss path further into the native extension and validate
correctness and performance in full inference.

## Change

Added `ExpertCopyPlan` to the native extension. A plan stores the byte offsets and sizes
for one expert module's parameters. The runtime can cache this plan and avoid rebuilding
offset lists on every miss.

New opt-in flags:

```bash
DEEPSEEK_SPARK_NATIVE_COPY_PLAN=1
DEEPSEEK_SPARK_NATIVE_FUSED_MATERIALIZER=1
DEEPSEEK_SPARK_NATIVE_WITH_CUDA=1
DEEPSEEK_SPARK_NATIVE_CUDA_MEMCPY=1
```

There are two native materializer variants:

- copy-plan fused read/copy: `read_into_and_copy`
- CUDA fused read/copy: `read_into_and_copy_cuda`

The CUDA variant uses `cudaMemcpyAsync` from pinned packed staging into CUDA parameter
storage on the current CUDA stream. This bypasses `target.copy_` for the planned tensor
copies.

## Artifacts

- `results/spark-66c9/native-copy-plan-smoke-e047.log`
- `results/spark-66c9/native-core-full-t32-e047.log`
- `results/spark-66c9/native-core-full-t32-e047.jsonl`
- `results/spark-66c9/arena-rerun-full-t32-e048.log`
- `results/spark-66c9/arena-rerun-full-t32-e048.jsonl`
- `results/spark-66c9/native-cuda-copy-smoke-e049.log`
- `results/spark-66c9/native-cuda-full-t32-e049.log`
- `results/spark-66c9/native-cuda-full-t32-e049.jsonl`

## Correctness

Both native smoke tests passed:

- CPU copy-plan storage copy.
- Fused file read plus copy-plan storage copy.
- CUDA copy-plan storage copy into CUDA tensors.
- Fused file read plus CUDA copy into CUDA tensors.

Full inference generated the same 32 completion token IDs as the E043 baseline for:

- E047 copy-plan fused materializer.
- E048 same-code arena rerun.
- E049 CUDA memcpy materializer.

## Results

| Run | Path | Decode | Tok/s | Packed read | Materialize | Token IDs |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| E043 | historical arena best | 16.516s | 1.937 | 3.660s | 5.435s | baseline |
| E047 | copy-plan fused | 27.345s | 1.170 | 11.663s | 14.235s | match |
| E048 | same-code arena rerun | 26.271s | 1.218 | 10.899s | 13.353s | match |
| E049 | CUDA memcpy fused | 21.901s | 1.461 | 7.921s | 10.255s | match |

E049 versus the same-window E048 baseline:

- Decode improved by 4.369s, or 16.6%.
- Packed read accounting improved by 2.978s, or 27.3%.
- Materialization accounting improved by 3.098s, or 23.2%.

## Interpretation

The native copy-plan object is correct, but by itself it is not enough. The first fused
copy-plan run was slightly slower than the same-code arena rerun.

The CUDA memcpy variant is the useful primitive. It still does not beat the older E043
absolute best, because the measured full-layout read path was much slower in this run
window than it was during E043. But against the same-window baseline, bypassing PyTorch
`copy_` with planned `cudaMemcpyAsync` moved the right counters and produced a real
end-to-end decode win.

Next work should keep the CUDA copy path and attack the remaining overhead:

- reduce one `cudaMemcpyAsync` per parameter to fewer larger transfers where layout allows;
- add explicit CUDA event timing around native copy work;
- batch all routed misses for a layer into one native call;
- make page-cache state an explicit part of performance comparisons.
