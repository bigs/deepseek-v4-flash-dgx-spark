# Nsight Summary: `/home/cole/runs/deepseek-v4-flash/nsys-cache1024-hello.sqlite`

## Telemetry Events

- `engine_load`
- `engine_generate` request_id=d9f80ffe8eb44265bb6043ae653fd815 prompt=1 generated=2 reused=0 new=1 prefill=144.576s decode=28.583s cache_hit=False
  - expert cache enabled=True entries=495 resident_gib=6.163 hits=21 misses=495 evictions=0 copied_gib=6.163
- `engine_deferred_postfill`
- `engine_generate` request_id=d1a5a272e6644d8c82512a480afebdc0 prompt=3 generated=1 reused=3 new=0 prefill=0.000s decode=0.000s cache_hit=True
  - expert cache enabled=True entries=650 resident_gib=8.093 hits=0 misses=0 evictions=0 copied_gib=0.000

## Totals

- CUDA kernels: 22918 calls, 0.359s total
- CUDA memcpy: 9363 copies, 95.402s total, 32.951 GiB
- CUDA runtime APIs: 62745 calls, 100.889s total

## Memcpy By Kind

| Kind | Calls | GPU Time | GiB | Avg MiB | Max MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Host-to-Device | 4045 | 95.267s | 16.473 | 4.170 | 2020.000 |
| Device-to-Device | 4536 | 0.133s | 16.478 | 3.720 | 2020.000 |
| Device-to-Host | 782 | 0.001s | 0.000 | 0.000 | 0.002 |

## Engine NVTX Ranges

| Range | Wall | HtoD GiB | HtoD GPU Time | DtoD GiB | Kernels | Kernel Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `engine.ensure_loaded request_id=d9f80ffe8eb44265bb6043ae653fd815` | 55.547s | 11.318 | 46.981s | 11.318 | 461 | 0.155s |
| `engine.load` | 55.320s | 11.318 | 46.981s | 11.318 | 461 | 0.155s |
| `engine.encode request_id=d9f80ffe8eb44265bb6043ae653fd815` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.prefill request_id=d9f80ffe8eb44265bb6043ae653fd815,new_tokens=1,reused_tokens=0` | 144.577s | 3.212 | 29.974s | 3.216 | 13653 | 0.119s |
| `engine.decode request_id=d9f80ffe8eb44265bb6043ae653fd815,max_tokens=2` | 28.583s | 1.942 | 18.312s | 1.945 | 8803 | 0.077s |
| `engine.decode_step request_id=d9f80ffe8eb44265bb6043ae653fd815,step=0` | 28.575s | 1.942 | 18.312s | 1.945 | 8803 | 0.077s |
| `engine.decode_step request_id=d9f80ffe8eb44265bb6043ae653fd815,step=1` | 0.008s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.postfill request_id=d9f80ffe8eb44265bb6043ae653fd815,mode=deferred` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.deferred_postfill session_id=http-probe` | 19.081s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.ensure_loaded request_id=d1a5a272e6644d8c82512a480afebdc0` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.encode request_id=d1a5a272e6644d8c82512a480afebdc0` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.prefill request_id=d1a5a272e6644d8c82512a480afebdc0,new_tokens=0,reused_tokens=3` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.decode request_id=d1a5a272e6644d8c82512a480afebdc0,max_tokens=1` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.decode_step request_id=d1a5a272e6644d8c82512a480afebdc0,step=0` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |
| `engine.postfill request_id=d1a5a272e6644d8c82512a480afebdc0,mode=deferred` | 0.000s | 0.000 | 0.000s | 0.000 | 0 | 0.000s |

## Top CUDA Runtime APIs

| API | Calls | Total | Avg ms | Max s |
| --- | ---: | ---: | ---: | ---: |
| `cudaMemcpyAsync_v3020` | 9363 | 95.409s | 10.190 | 7.824 |
| `cuLibraryLoadData` | 47 | 2.708s | 57.628 | 0.324 |
| `cuMemCreate` | 2422 | 1.877s | 0.775 | 0.001 |
| `cudaLaunchKernel_v7000` | 18949 | 0.350s | 0.018 | 0.033 |
| `cuMemSetAccess` | 669 | 0.252s | 0.376 | 0.009 |
| `cudaMemGetInfo_v3020` | 3 | 0.207s | 68.951 | 0.207 |
| `cudaStreamSynchronize_v3020` | 4827 | 0.030s | 0.006 | 0.000 |
| `cuLaunchKernel` | 3913 | 0.024s | 0.006 | 0.000 |
| `cuMemMap` | 2421 | 0.010s | 0.004 | 0.000 |
| `cudaLaunchKernelExC_v11060` | 56 | 0.007s | 0.131 | 0.007 |
