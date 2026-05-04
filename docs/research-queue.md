# Research Queue

Date: 2026-05-04

This queue reflects the second flash-moe pass and experiments 030-050.

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
- `DEEPSEEK_SPARK_NATIVE_MATERIALIZER=1`.
- `DEEPSEEK_SPARK_EXPERT_ARENA_SLOTS=256`.
- `DEEPSEEK_SPARK_NATIVE_WITH_CUDA=1`.
- `DEEPSEEK_SPARK_NATIVE_COPY_PLAN=1`.
- `DEEPSEEK_SPARK_NATIVE_FUSED_MATERIALIZER=1`.
- `DEEPSEEK_SPARK_NATIVE_CUDA_MEMCPY=1`.

Best observed controlled deployable 32-token decode is now the E050 full-layout arena
plus native CUDA materializer recipe:

| Run | Decode | Tok/s |
| --- | ---: | ---: |
| E041 full native c1024 | 21.022s | 1.522 |
| E042 native materializer v2 | 21.917s | 1.460 |
| E043 expert arena 256 | 16.516s | 1.937 |
| E044 materialize workers 2 | 50.415s | 0.635 |
| E046 hot-first full layout | 30.899s | 1.036 |
| E049 native CUDA materializer | 21.901s | 1.461 |
| E050 arena plus native CUDA A/B best | 16.239s | 1.971 |

The fastest controlled 8-token matrix row is `full-t8-native-c1024` at 5.213s
and 1.535 tok/s. The older E040 5.036s route-filtered run remains useful history, but
E041 is the cleaner same-day comparison point.

## Updated Queue

1. **Native materializer beyond tensor-copy scheduling**
   - E042 showed the first native materializer primitive is only a weak/noisy win alone.
   - E047 showed native copy plans are correct but not enough by themselves.
   - E049 showed planned `cudaMemcpyAsync` is a useful primitive against a same-window
     baseline, but not yet a new absolute best.
   - E050 showed the CUDA copy path and expert arena combine into the new best observed
     32-token deployable row.
   - The next target is fewer larger native transfers and one native call per layer's
     routed misses.

2. **Route-trace replay as the screening gate**
   - Use `scripts/replay_packed_expert_trace.py` before full inference runs.
   - It quickly rejects cache/read-order ideas that only look good on paper.

3. **Threaded or async packed reads below Python**
   - Replay showed warm threaded reads improved from 11.92 GiB/s to 13.18 GiB/s.
   - E044 reconfirmed that Python-threaded CUDA materialization is wrong even after
     native loading plus arena reuse.
   - The candidate is lower-level I/O plus explicit handoff, not Python workers doing
     active CUDA work.

4. **Layout locality sweep**
   - E041 proved route-filtered layouts are invalid for longer deployment probes because
     unseen experts fall back to safetensors.
   - E046 showed simple hot-first ordering from an 8-token trace regresses badly.
   - Future layout variants need longer route traces, replay screening, and page-cache
     state controls before full inference.

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
- Route-filtered packed layouts for deployment claims. They are useful short-probe tools,
  but E041 showed large packed-miss fallback at 32 tokens.
- Broad replacement of official FP8/FP4 GEMM kernels. Existing small-shape kernels run,
  and current decode is dominated by packed read plus materialization.
- Python-threaded expert materialization. It was a regression in E022 and E044.
- Inline postfill as the default serving policy.
- Simple hot-first full-layout ordering based on an 8-token route trace.
