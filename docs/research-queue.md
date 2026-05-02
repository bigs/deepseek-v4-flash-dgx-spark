# Research Queue

Date: 2026-05-02

This queue reflects the second flash-moe pass and experiments 030-040.

## Current Best Recipe

- Route-filtered packed layout for short-route experiments.
- Full all-expert packed layout for publishable deployment experiments.
- `DEEPSEEK_SPARK_EXPERT_CACHE_POLICY=layer_lru`.
- `DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES=1024`.
- Keep CUDA allocator cache warm on cached evictions.
- `DEEPSEEK_SPARK_PACKED_NATIVE_LOADER=1`.
- `DEEPSEEK_SPARK_PACKED_NATIVE_PINNED_STAGING=1`.
- `DEEPSEEK_SPARK_DIRECT_PARAM_COPY=1`.
- `DEEPSEEK_SPARK_PARAM_COPY_NON_BLOCKING=1`.

Best observed 8-token decode is now the native pinned/direct-copy route-filtered packed
layout:

| Run | Decode | Tok/s |
| --- | ---: | ---: |
| E026 no empty cache | 6.434s | 1.243 |
| E036 reusable staging | 6.408s | 1.248 |
| E038 native reader | 5.341s | 1.498 |
| E040 native direct | 5.036s | 1.589 |

## Updated Queue

1. **Second-stage native materializer**
   - Move tensor slicing and parameter copy scheduling deeper into native code.
   - The next target is fewer Python parameter loops and a batched/fused copy plan per
     expert block.

2. **Route-trace replay as the screening gate**
   - Use `scripts/replay_packed_expert_trace.py` before full inference runs.
   - It quickly rejects cache/read-order ideas that only look good on paper.

3. **Threaded or async packed reads below Python**
   - Replay showed warm threaded reads improved from 11.92 GiB/s to 13.18 GiB/s.
   - Python-threaded CUDA materialization still regresses; the candidate is lower-level
     I/O plus explicit handoff, not Python workers doing active CUDA work.

4. **Layout locality sweep**
   - E027 showed the full layout is slower than the route-filtered layout.
   - Test hot-expert ordering, route-clustered ordering, and per-layer metadata/layout
     variants with replay first, then full inference.

5. **Persistent KV postfill semantics**
   - Inline postfill makes cache resume nearly free but puts postfill on the request path.
   - Server policy should complete deferred postfill before allowing a session-reuse
     request to run.

6. **GB10 sparse-attention kernel**
   - Still needed for long-context quality/performance because the official production
     shape fails on GB10.
   - It is not the first decode-throughput lever: E034 shows non-loader residual decode
     time is only about 5.5% of the 32-token full-layout decode.

## Deprioritized

- Global LRU for the current runtime. Replay saw fewer misses, but full inference
  regressed badly.
- Broad replacement of official FP8/FP4 GEMM kernels. Existing small-shape kernels run,
  and current decode is dominated by packed read plus materialization.
- Python-threaded expert materialization. It was a regression in E022.
- Inline postfill as the default serving policy.
