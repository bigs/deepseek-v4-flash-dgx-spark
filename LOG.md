# Optimization Sprint Log

Goal: push DeepSeek-V4-Flash decode and prefill as fast as possible on DGX Spark, with an initial sprint target of 1 token/sec decode.

Current integrated baseline from the checked-in measurements:

- `nsys-hello`: prefill 144.948s, first decode step 28.220s.
- `route-cache1024-t8`: prefill 125.083s, decode steps 26.873s, 17.828s, 17.410s, 18.022s, 13.525s, 11.917s, 10.021s, 0.008s.
- CUDA kernel time is not the dominant cost yet; miss materialization and host-to-device movement dominate.

## Sprint Experiments

1. E018: Build a full fixed-offset packed routed-expert layout from the HF checkpoint.
2. E019: Integrate packed expert miss loading into the lazy runtime.
3. E020: Measure packed miss loading with expert cache sizes 1024 and 2048.
4. E021: Test layer-aware cache policy instead of one global LRU.
5. E022: Add persistent I/O workers for packed misses.
6. E023: Test pinned or page-aligned host staging for packed blocks.
7. E024: Add async miss prefetch for routed experts once a layer's routing decision is known.
8. E025: Probe native sparse-attention / kernel replacement paths at runtime shapes.
9. E026: Reduce Python object overhead in expert materialization and decode loops.

## Entries

### 2026-05-02

- Started second flash-moe pass and follow-on research sprint.
  - Flash-MoE reinforces packed per-layer expert files, reusable staging buffers, route replay, and skeptical treatment of custom caches/prediction.
  - Spark differs from Apple Silicon: device-resident expert caching is useful here, but whole-system memory pressure still invalidates simple miss-count reasoning.
- E031 route-trace replay added and run against the full packed layout.
  - Warm layer-LRU route-order replay: 1.303s for 1,248 blocks / 16.68GB, 11.92 GiB/s.
  - Threaded read-only replay improved to 1.179s / 13.18 GiB/s.
  - Global LRU had fewer simulated misses, but this needed full-runtime validation.
- E032 reusable slab materialization microbenchmark completed.
  - Current bytes/bytearray path: 0.176s median for 64 blocks, 4.54 GiB/s.
  - Reusable `preadv` bytearray slab: 0.112s median, 7.10 GiB/s.
  - Added opt-in runtime flag `DEEPSEEK_SPARK_PACKED_REUSE_STAGING=1`.
- E034 kernel budget analysis completed from E029 telemetry.
  - 32-token full-layout decode was 30.307s.
  - Packed reads were 11.527s and materialization was 17.106s; residual non-loader time was only 1.674s.
  - Broad custom kernels cannot get us near 20 tok/s; custom GB10 work should first target native loader/materialization, then sparse attention.
- E035 global LRU recheck completed.
  - Despite slightly fewer misses than layer-LRU, full decode regressed to 20.154s / 0.40 tok/s.
  - Keep layer-LRU as the current runtime policy.
- E036 reusable staging full-runtime run completed.
  - Decode improved slightly from 6.434s to 6.408s for 8 tokens, about 1.248 tok/s.
  - Full-runtime impact is small, but the primitive is worth keeping for the native loader track.
- E037 native packed loader added and microbenchmarked.
  - Added a lazy PyTorch C++ extension that reads packed expert blocks directly into reusable CPU uint8 tensors.
  - Median 64-block throughput: current bytes path 1.25 GiB/s, reusable `preadv` 7.82 GiB/s, native tensor 7.96 GiB/s.
- E039 native pinned staging and direct-copy materializer controls added.
  - Native pinned tensor staging reached 9.36 GiB/s in the 64-block microbenchmark.
  - Added `DEEPSEEK_SPARK_DIRECT_PARAM_COPY=1` and `DEEPSEEK_SPARK_PARAM_COPY_NON_BLOCKING=1`.
- E040 native direct runtime completed.
  - Best 8-token decode is now 5.036s, about 1.589 tok/s.
  - Packed read accounting fell to 1.120s and materialization accounting fell to 2.608s.
  - Recipe: native loader, pinned native staging, direct CPU-to-parameter copy, non-blocking copy, layer-LRU cache, route-filtered packed layout.

- Starting follow-on experiments after the 1 tok/s sprint:
  - E027: run the best config against the full all-expert packed layout.
  - E028: test inline postfill for persistent session resume latency.
  - E029: run a longer decode to measure steady-state throughput beyond 8 tokens.
- E027 full-layout runtime completed.
  - Full packed layout had zero packed misses and worked as the deployment artifact.
  - 8-token decode was 8.283s, about 0.97 tok/s, versus 6.434s / 1.24 tok/s with the route-filtered layout.
  - Full layout packed read accounting was 3.964s versus 2.157s route-filtered; this points to locality/cache effects or larger metadata/filesystem footprint.
- E028 inline postfill completed.
  - Initial 8-token request regressed to 16.173s decode plus 2.332s inline postfill.
  - Cached continuation had a real cache resume hit: 0.000035s prefill and 0.000122s decode.
  - The same cached continuation still spent 9.307s in inline postfill before returning.
  - Inline mode proves persistent KV can remove resume prefill, but it is the wrong latency policy; deferred background postfill plus a wait/reuse boundary is the better server behavior.
- E029 longer decode started against the full all-expert packed layout.
  - This uses the full layout to avoid route-filter fallback misses during a 32-token generation.
- E029 first two attempts failed due an incomplete harness `PYTHONPATH`.
  - The patched HTTP probe now surfaces response bodies, which exposed `ModuleNotFoundError("No module named 'fast_hadamard_transform'")`.
  - Rerunning with `PYTHONPATH=/repo:/repo/spark_runtime:/model/inference` fixed the harness issue.
- E029 completed as `long-decode-e029c`.
  - Full layout, 32 generated tokens: prefill 92.102s, decode 30.307s, about 1.06 tok/s.
  - Packed miss path stayed complete: 3,239 packed loads, 0 packed misses.
  - Packed read time was 11.527s and materialization time was 17.106s inside decode.
  - Non-final decode steps averaged 0.977s; after the first 8 tokens they averaged 0.938s.
  - This confirms sustained decode is still dominated by synchronous expert miss service, not raw token-loop overhead.
- Follow-on learning from E027-E029:
  - Full layout is publishable but locality-sensitive.
  - Inline postfill proves persistent KV resume can be near-free, but pushes full-forward cost into request latency.
  - Longer decode sustains roughly 1 tok/s with current full-layout runtime, so the next three experiments should be route-trace replay, slab-backed materialization, and layout-locality sweeps.

### 2026-05-01

- Started sprint log and anchored the current performance baseline.
- First target is E018/E019 because packed expert reads were already measured much faster than safetensor tensor-by-tensor materialization, and the runtime bottleneck is weight miss handling rather than GPU arithmetic.
- E018 route-filtered builder added and run on Spark against `route-cache1024-t8-routes.jsonl`.
  - Packed 1,232 observed routed experts across 43 layers.
  - Output size: 16,471,031,808 bytes, about 15.3 GiB.
  - Build wall from script progress: about 8.4s.
  - Layout: `/home/cole/runs/deepseek-v4-flash/packed-route-e018/layout.json`.
- E018 full packed layout also built on Spark.
  - Packed all 11,008 routed experts across 43 layers.
  - Output size: 147,169,738,752 bytes, about 137.1 GiB; directory reports 138G.
  - Final layer completed at about 79.3s elapsed.
  - Layout: `/home/cole/runs/deepseek-v4-flash/packed-full-e018/layout.json`.
- E019 initial runtime integration added behind `DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT`.
  - It reads a whole packed expert block once, reconstructs CPU tensor views by target name, and falls back to safetensors for missing experts.
  - Next step: validate packed tensors against safetensors, then measure generation with the packed layout enabled.
- E019 validation passed inside the runtime container for representative experts in layers 0, 15, and 42. All six tensors per expert matched safetensors exactly.
- E020 packed-layout generation probe completed on Spark with expert cache 1024 and the route-filtered packed layout.
  - Prefill improved from 125.083s to 95.067s.
  - Decode improved from 115.604s to 15.642s for 8 generated tokens, about 0.51 tok/s.
  - Decode step walls dropped from `[26.873, 17.828, 17.410, 18.022, 13.525, 11.917, 10.021, 0.008]` to `[2.448, 1.421, 2.796, 2.791, 2.274, 2.063, 1.840, 0.008]`.
  - Packed loads: 1,237 total across completion plus cached continuation; packed misses: 0.
- E021 layer-aware cache policy implemented locally behind `DEEPSEEK_SPARK_EXPERT_CACHE_POLICY=layer_lru`; it will be measured after the packed-layout baseline completes.
- E021 layer-aware cache run completed on Spark with the same 1024-entry cache budget.
  - Decode improved again from 15.642s to 12.750s for 8 generated tokens, about 0.63 tok/s.
  - Misses stayed similar, so this is not a full cache-reuse win yet.
  - Driver memory was recovered after the run; MemAvailable returned to about 118 GiB.
- E020 2048-entry packed-cache run completed.
  - Evictions dropped to 0, but decode was 13.900s for 8 tokens, slower than layer-LRU 1024.
  - This says remaining first-use packed miss materialization dominates more than capacity misses for the short probe.
- E026 allocator overhead tweak implemented locally: cached expert evictions no longer call `torch.cuda.empty_cache()` by default. Uncached mode still empties by default.
- E026 measured on Spark with layer-LRU 1024 and packed layout.
  - Prefill improved from 94.988s to 91.662s.
  - Decode improved from 12.750s to 6.434s for 8 generated tokens.
  - Current best decode: about 1.24 tok/s.
  - Packed read accounting dropped from 9.308s to 2.157s, and materialization from 11.160s to 4.322s.
  - This crossed the sprint's 1 tok/s decode target.
- E023 pinned staging smoke test passed correctness but was a performance regression.
  - One packed expert block read with per-miss pinned allocation took about 0.340s versus roughly 0.011-0.016s in earlier pageable validation.
  - Do not use per-miss pinned allocation; revisit only with a persistent pinned staging pool.
- E022 parallel expert materialization tested with 4 workers.
  - Decode regressed to 16.170s for 8 generated tokens, about 0.49 tok/s.
  - Materialization accounting jumped to 16.721s.
  - Do not use Python-threaded CUDA materialization; future overlap needs CPU-only I/O prefetch or a custom extension.
- E025 sparse-attention shape probe run on Spark.
  - Official Tilelang sparse attention fails at production-like `h=64,d=512` shapes on GB10 with dynamic shared memory request 141,312 bytes.
  - Keep the Python fallback until we write a GB10-specific kernel.
- E024 prefetch feasibility written up.
  - Naive temporal prediction had 0% all-slot hit rate, and Python-threaded active materialization regressed badly.
  - Next credible prefetch path is a lower-level CPU I/O plus explicit CUDA stream/custom extension design, not more Python threading.
