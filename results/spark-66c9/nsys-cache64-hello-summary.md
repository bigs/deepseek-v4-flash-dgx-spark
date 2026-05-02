# Nsight Summary: `/home/cole/runs/deepseek-v4-flash/nsys-cache64-hello.sqlite`

## Telemetry Events

- `engine_load`
- `engine_generate` request_id=7383057429c54b78a4edc688e2f00731 prompt=1 generated=2 reused=0 new=1 prefill=145.455s decode=28.397s cache_hit=False
- `engine_deferred_postfill`
- `engine_generate` request_id=a41a66c1fcb14c47b941454721722d9d prompt=3 generated=1 reused=3 new=0 prefill=0.000s decode=0.000s cache_hit=True

## Totals

- CUDA kernels: 23016 calls, 0.358s total
- CUDA memcpy: 9508 copies, 91.435s total, 33.250 GiB
- CUDA runtime APIs: 62429 calls, 96.857s total

## Memcpy By Kind

| Kind | Calls | GPU Time | GiB | Avg MiB | Max MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Host-to-Device | 4117 | 91.298s | 16.622 | 4.134 | 2020.000 |
| Device-to-Device | 4609 | 0.136s | 16.627 | 3.694 | 2020.000 |
| Device-to-Host | 782 | 0.001s | 0.000 | 0.000 | 0.002 |

## Engine NVTX Ranges

| Range | Wall | HtoD GiB | HtoD GPU Time | DtoD GiB | Kernels | Kernel Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `engine.ensure_loaded request_id=7383057429c54b78a4edc688e2f00731` | 51.531s | 11.318 | 43.266s | 11.318 | 461 | 0.154s |
| `engine.load` | 51.294s | 11.318 | 43.266s | 11.318 | 461 | 0.154s |
| `engine.encode request_id=7383057429c54b78a4edc688e2f00731` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.prefill request_id=7383057429c54b78a4edc688e2f00731,new_tokens=1,reused_tokens=0` | 145.455s | 3.212 | 29.970s | 3.216 | 13653 | 0.118s |
| `engine.decode request_id=7383057429c54b78a4edc688e2f00731,max_tokens=2` | 28.397s | 2.092 | 18.062s | 2.094 | 8901 | 0.077s |
| `engine.decode_step request_id=7383057429c54b78a4edc688e2f00731,step=0` | 28.389s | 2.092 | 18.062s | 2.094 | 8901 | 0.077s |
| `engine.decode_step request_id=7383057429c54b78a4edc688e2f00731,step=1` | 0.008s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.postfill request_id=7383057429c54b78a4edc688e2f00731,mode=deferred` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.deferred_postfill session_id=http-probe` | 18.752s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.ensure_loaded request_id=a41a66c1fcb14c47b941454721722d9d` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.encode request_id=a41a66c1fcb14c47b941454721722d9d` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.prefill request_id=a41a66c1fcb14c47b941454721722d9d,new_tokens=0,reused_tokens=3` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.decode request_id=a41a66c1fcb14c47b941454721722d9d,max_tokens=1` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.decode_step request_id=a41a66c1fcb14c47b941454721722d9d,step=0` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.postfill request_id=a41a66c1fcb14c47b941454721722d9d,mode=deferred` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |

## Top CUDA Runtime APIs

| API | Calls | Total | Avg ms | Max s |
| --- | ---: | ---: | ---: | ---: |
| `cudaMemcpyAsync_v3020` | 9508 | 91.449s | 9.618 | 7.611 |
| `cuLibraryLoadData` | 47 | 2.784s | 59.236 | 0.327 |
| `cuMemCreate` | 2143 | 1.741s | 0.812 | 0.001 |
| `cudaLaunchKernel_v7000` | 19029 | 0.367s | 0.019 | 0.034 |
| `cudaMemGetInfo_v3020` | 3 | 0.216s | 72.062 | 0.216 |
| `cuMemSetAccess` | 390 | 0.215s | 0.551 | 0.009 |
| `cudaStreamSynchronize_v3020` | 4900 | 0.028s | 0.006 | 0.000 |
| `cuLaunchKernel` | 3931 | 0.023s | 0.006 | 0.000 |
| `cuMemMap` | 2142 | 0.008s | 0.004 | 0.000 |
| `cudaLaunchKernelExC_v11060` | 56 | 0.008s | 0.136 | 0.007 |
