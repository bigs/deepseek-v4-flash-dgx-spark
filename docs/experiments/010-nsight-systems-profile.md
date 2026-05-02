# Experiment 010: Nsight Systems Profile

Date: 2026-05-01

Host: `spark-66c9`

Goal: capture a CUDA/NVTX timeline for the known slow HTTP inference path and determine
whether the immediate bottleneck is kernel compute, CUDA API overhead, or host/device movement.

## Command

The new harness mounts the host Nsight Systems installation into the runtime container and
runs the usual guarded Docker probe under `nsys profile`:

```bash
python3 scripts/run_nsys_profile.py \
  --name nsys-hello \
  --manifest-csv /repo/results/spark-66c9/weight-manifest.csv \
  --prompt Hello \
  --max-tokens 2 \
  --wait-postfill-before-cached \
  --timeout-seconds 2400
```

The profiled application was:

```bash
python3 /repo/scripts/probe_server_http.py \
  --prompt Hello \
  --max-tokens 2 \
  --wait-postfill-before-cached \
  --skip-chat
```

## Artifacts

- `results/spark-66c9/nsys-hello.nsys-rep`
- `results/spark-66c9/nsys-hello.sqlite`
- `results/spark-66c9/nsys-hello-stats.txt`
- `results/spark-66c9/nsys-hello-summary.md`
- `results/spark-66c9/nsys-hello.jsonl`
- `results/spark-66c9/nsys-hello.log`

## API / Engine Result

The first completion generated `World =` from prompt `Hello`.

Telemetry for the first request:

| Metric | Value |
| --- | ---: |
| HTTP elapsed | 225.710s |
| engine total | 225.706s |
| prompt tokens | 1 |
| completion tokens | 2 |
| prefill wall | 144.948s |
| decode wall | 28.228s |
| first decode step wall | 28.220s |
| second decode step wall | 0.008s |

After deferred postfill completed, the cached continuation reused all three prefix tokens
and returned in 0.0028s HTTP wall time.

## Nsight Summary

The decisive result is that CUDA kernel execution is not the bottleneck in this run.

| Signal | Value |
| --- | ---: |
| CUDA kernels | 23,015 calls |
| CUDA kernel time | 0.355s |
| CUDA memcpy operations | 9,508 copies |
| CUDA memcpy time | 92.716s |
| Total memcpy volume | 33.250 GiB |
| CUDA runtime API time | 98.126s |

Memcpy by kind:

| Kind | Calls | GPU Time | Volume |
| --- | ---: | ---: | ---: |
| Host-to-Device | 4,117 | 92.589s | 16.622 GiB |
| Device-to-Device | 4,609 | 0.126s | 16.627 GiB |
| Device-to-Host | 782 | 0.001s | ~0 GiB |

Dominant CUDA runtime API:

| API | Calls | Total |
| --- | ---: | ---: |
| `cudaMemcpyAsync_v3020` | 9,508 | 92.736s |
| `cuLibraryLoadData` | 47 | 2.728s |
| `cuMemCreate` | 2,119 | 1.796s |
| `cudaLaunchKernel_v7000` | 19,028 | 0.356s |

## NVTX Breakdown

| Range | Wall | HtoD Volume | HtoD Time | Kernel Time |
| --- | ---: | ---: | ---: | ---: |
| `engine.load` | 51.962s | 11.318 GiB | 43.943s | 0.154s |
| `engine.prefill` | 144.949s | 3.212 GiB | 30.485s | 0.116s |
| `engine.decode` | 28.228s | 2.092 GiB | 18.162s | 0.075s |
| `engine.decode_step step=0` | 28.220s | 2.092 GiB | 18.162s | 0.075s |

The biggest host-to-device copy-size buckets were:

| Copy size | Calls | HtoD Time |
| --- | ---: | ---: |
| 4 MiB | 1,364 | 44.498s |
| 32 MiB | 88 | 21.578s |
| 8 MiB | 193 | 9.535s |
| 1010 MiB | 1 | 7.594s |
| 0.25 MiB | 1,300 | 5.848s |

## Interpretation

This run is overwhelmingly movement-bound, not compute-bound. The official kernels visible
in the profile, including `fp4_gemm_kernel__kernel` and `fp8_gemm_kernel__kernel`, account
for a tiny fraction of wall time. The slow path is repeated host-to-device transfer of
resident weights during load and lazy routed-expert weights during prefill/decode.

This matches the current correctness-first runtime shape:

- resident weights are loaded from CPU safetensor tensors into CUDA parameters at startup;
- each routed expert is materialized on demand;
- each materialized expert copies its weights to device;
- `LazyRoutedExpert.forward()` evicts the expert immediately after use.

That immediate eviction policy is useful for not OOMing while proving correctness, but it
guarantees repeated HtoD copies on every token. The profile shows that this policy is now
the first optimization target.

## Next Optimization Target

Build and measure a routed-expert cache before writing custom math kernels.

The first cache should be deliberately simple:

1. Keep `LazyRoutedExpert` modules alive across calls instead of evicting immediately.
2. Add an LRU budget by estimated expert bytes or max resident experts.
3. Track expert cache hits/misses and HtoD bytes/copies in telemetry.
4. Rerun this exact Nsight experiment and compare:
   - HtoD GiB/time during prefill
   - HtoD GiB/time during decode
   - cached continuation behavior
   - host `MemAvailable`

If a small expert cache collapses decode from ~28s toward actual kernel time, that gives us
a much better baseline before deeper GB10 kernel work. Custom kernels still matter, but the
profile says data movement and eviction policy are ahead of kernel math in the queue.
