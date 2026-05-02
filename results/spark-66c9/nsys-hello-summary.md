# Nsight Summary: `results/spark-66c9/nsys-hello.sqlite`

## Telemetry Events

- `engine_load`
- `engine_generate` request_id=a9804b49b074485fb434e6aa85b09b90 prompt=1 generated=2 reused=0 new=1 prefill=144.948s decode=28.228s cache_hit=False
- `engine_deferred_postfill`
- `engine_generate` request_id=f8c1c94d52fb41ebbfc99efbc2a2baac prompt=3 generated=1 reused=3 new=0 prefill=0.000s decode=0.000s cache_hit=True

## Totals

- CUDA kernels: 23015 calls, 0.355s total
- CUDA memcpy: 9508 copies, 92.716s total, 33.250 GiB
- CUDA runtime APIs: 62430 calls, 98.126s total

## Memcpy By Kind

| Kind | Calls | GPU Time | GiB | Avg MiB | Max MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Host-to-Device | 4117 | 92.589s | 16.622 | 4.134 | 2020.000 |
| Device-to-Device | 4609 | 0.126s | 16.627 | 3.694 | 2020.000 |
| Device-to-Host | 782 | 0.001s | 0.000 | 0.000 | 0.002 |

## Engine NVTX Ranges

| Range | Wall | HtoD GiB | HtoD GPU Time | DtoD GiB | Kernels | Kernel Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `engine.ensure_loaded request_id=a9804b49b074485fb434e6aa85b09b90` | 52.200s | 11.318 | 43.943s | 11.318 | 461 | 0.154s |
| `engine.load` | 51.962s | 11.318 | 43.943s | 11.318 | 461 | 0.154s |
| `engine.encode request_id=a9804b49b074485fb434e6aa85b09b90` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.prefill request_id=a9804b49b074485fb434e6aa85b09b90,new_tokens=1,reused_tokens=0` | 144.949s | 3.212 | 30.485s | 3.216 | 13653 | 0.116s |
| `engine.decode request_id=a9804b49b074485fb434e6aa85b09b90,max_tokens=2` | 28.228s | 2.092 | 18.162s | 2.094 | 8900 | 0.075s |
| `engine.decode_step request_id=a9804b49b074485fb434e6aa85b09b90,step=0` | 28.220s | 2.092 | 18.162s | 2.094 | 8900 | 0.075s |
| `engine.decode_step request_id=a9804b49b074485fb434e6aa85b09b90,step=1` | 0.008s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.postfill request_id=a9804b49b074485fb434e6aa85b09b90,mode=deferred` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.deferred_postfill session_id=http-probe` | 18.313s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.ensure_loaded request_id=f8c1c94d52fb41ebbfc99efbc2a2baac` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.encode request_id=f8c1c94d52fb41ebbfc99efbc2a2baac` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.prefill request_id=f8c1c94d52fb41ebbfc99efbc2a2baac,new_tokens=0,reused_tokens=3` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.decode request_id=f8c1c94d52fb41ebbfc99efbc2a2baac,max_tokens=1` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.decode_step request_id=f8c1c94d52fb41ebbfc99efbc2a2baac,step=0` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.postfill request_id=f8c1c94d52fb41ebbfc99efbc2a2baac,mode=deferred` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |

## Top CUDA Runtime APIs

| API | Calls | Total | Avg ms | Max s |
| --- | ---: | ---: | ---: | ---: |
| `cudaMemcpyAsync_v3020` | 9508 | 92.736s | 9.753 | 7.594 |
| `cuLibraryLoadData` | 47 | 2.728s | 58.040 | 0.326 |
| `cuMemCreate` | 2119 | 1.796s | 0.848 | 0.001 |
| `cudaLaunchKernel_v7000` | 19028 | 0.356s | 0.019 | 0.037 |
| `cuMemSetAccess` | 366 | 0.215s | 0.587 | 0.009 |
| `cudaMemGetInfo_v3020` | 3 | 0.212s | 70.809 | 0.212 |
| `cudaStreamSynchronize_v3020` | 4900 | 0.026s | 0.005 | 0.000 |
| `cuLaunchKernel` | 3931 | 0.022s | 0.006 | 0.000 |
| `cuMemMap` | 2118 | 0.009s | 0.004 | 0.000 |
| `cudaLaunchKernelExC_v11060` | 56 | 0.008s | 0.139 | 0.007 |
