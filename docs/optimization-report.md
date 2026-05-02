# Optimization Report: Flash-MoE-Inspired Spark Experiments

Date: 2026-05-01

Scope: Experiments 011-017, based on the Nsight baseline in Experiment 010 and the
Flash-MoE source read.

## Executive Summary

The highest-impact finding is that packed expert misses matter more than a naive cache by
itself.

Measured impact ranking:

1. **Packed expert miss format**: strongest direct win. Fresh-process packed block to CUDA
   was `6.5x` faster than safetensor tensor fetches/transfers for 16 experts.
2. **Packed on-disk layout**: strong and stable. Packed reads reached about `3.0 GiB/s`
   in cold-ish tests and `13.6 GiB/s` warm, with simpler one-block-per-expert offsets.
3. **Expert cache**: useful but not sufficient. A 1024-entry cache got real hits, but
   the two-token Nsight benchmark remained dominated by first-use misses.
4. **Parallel packed reads**: modest win. 4-8 thread reads improved packed read throughput
   by about `1.2x-1.4x`.
5. **Native kernels**: viable but not the current top bottleneck. Official FP8/FP4
   TileLang kernels run on GB10 at small shapes.
6. **Temporal prediction**: not worth implementing yet. Naive previous-token prediction
   had `38.8%` slot hit rate and `0%` all-hit rate.

## Experiment Map

| Experiment | Question | Result | Decision |
| --- | --- | --- | --- |
| 011 | Does a global routed-expert cache fix decode? | 64 entries: 0 hits. 1024 entries: hits but no 2-token wall-time win. | Keep cache, but make misses cheaper and policy smarter. |
| 012 | How much expert reuse exists? | 806 experts for 80% coverage across observed layers in an 8-token trace. | Use layer-aware cache sizing. |
| 013 | Is fixed-offset packed layout better than HF ranges? | Packed layout was best or tied, up to `2.3x` faster cold-ish. | Build packed expert files. |
| 014 | Is packed miss materialization faster? | Packed block to CUDA: `0.367s`; safetensor tensors to CUDA: `2.375s`. | Wire packed miss loader into runtime. |
| 015 | Does deterministic parallel read help? | `1.2x-1.4x` faster than serial packed reads. | Use persistent I/O pool after packed layout. |
| 016 | Do native FP8/FP4 kernels execute on GB10? | Official `fp8_gemm`, `fp4_gemm`, quant, small sparse attention all passed. | Reuse working kernels; target only blockers. |
| 017 | Should we implement temporal prediction now? | Previous-token prediction: `38.8%` slot hit, `0%` all-hit. | Defer prediction. |

## What Had The Most Impact

### 1. Packed Miss Materialization

Experiment 014 is the clearest single result:

| Path | Payload | Time | Throughput |
| --- | ---: | ---: | ---: |
| safetensor tensors to CUDA | 204 MiB | 2.375s | 85.9 MiB/s |
| packed block to CUDA | 204 MiB | 0.367s | 556.1 MiB/s |

This attacks the same failure Nsight showed in Experiment 010: enormous CUDA memcpy/API
time around lazy expert materialization.

The current runtime miss path does too much:

- safetensor lookup;
- six tensor materializations per expert;
- six HtoD transfers per expert;
- PyTorch object handling around every miss.

The packed path reduces this to one expert block and one transfer.

### 2. Packed On-Disk Layout

Experiment 013 showed packed files are better even before runtime integration:

| Layer | HF Exact Cold-ish | Packed Cold-ish | Speedup |
| ---: | ---: | ---: | ---: |
| 15 | 2.97 GiB/s | 2.99 GiB/s | 1.0x |
| 18 | 1.30 GiB/s | 3.02 GiB/s | 2.3x |
| 39 | 1.47 GiB/s | 3.06 GiB/s | 2.1x |

The earlier Experiment 002 was still useful: HF layout is not catastrophically bad. But
packed layout is more predictable and better aligned with a custom miss loader.

### 3. Expert Cache

Experiment 011 and 012 together show the cache is real but not magic.

The 1024-entry cache in the 8-token trace reached:

- `13.69 GiB` resident expert bytes;
- `1085` hits;
- `1237` misses;
- `213` evictions.

Decode step times improved as reuse accumulated:

```text
26.87s, 17.83s, 17.41s, 18.02s, 13.52s, 11.92s, 10.02s, 0.008s
```

But the cache cannot reduce first-use miss cost. That is why it must be combined with
packed misses.

### 4. Parallel I/O

Experiment 015 showed deterministic parallel reads help, but they are incremental:

| Layer | Serial | Parallel 8 | Speedup |
| ---: | ---: | ---: | ---: |
| 15 | 7.81 GiB/s | 9.96 GiB/s | 1.27x |
| 18 | 7.70 GiB/s | 10.19 GiB/s | 1.32x |
| 39 | 7.20 GiB/s | 10.24 GiB/s | 1.42x |

This should be implemented as a persistent pool, not ad hoc per request.

## Recommended Combination

The next runtime architecture should combine the pieces in this order:

1. **Packed expert files**
   - One file per layer.
   - Fixed expert block size.
   - Layout file maps tensor slices inside each block.

2. **Packed miss loader**
   - Read one expert block.
   - Use pinned/page-aligned host staging.
   - Transfer one block to CUDA or slice/copy parameters with fewer transfers.

3. **Layer-aware expert cache**
   - Keep device-resident experts across decode.
   - Reserve a small minimum per routed layer.
   - Allocate extra entries to layers with strong concentration.
   - Track hits, misses, evictions, and resident bytes by layer.

4. **Persistent I/O pool**
   - Parallelize remaining packed misses.
   - Avoid per-step thread setup.
   - Dispatch only deterministic known misses.

5. **Targeted kernels**
   - Keep official FP8/FP4 GEMM where it works.
   - Replace sparse attention fallback for GB10 full shapes.
   - Consider fused expert load/dequant/matmul only after packed misses are integrated.

## What Not To Do Yet

Do not implement temporal prediction yet. Experiment 017 showed:

- `38.8%` slot hit rate;
- `0%` all-hit rate.

That would waste bandwidth and still leave synchronous misses on every layer. This matches
Flash-MoE's result.

Do not start with a broad custom-kernel rewrite. Experiment 016 showed native FP8/FP4
kernels run, and Experiment 010 showed CUDA kernel time was tiny compared with movement.

Do not rely on a tiny global LRU. Experiment 011 showed 64 entries produced zero hits.

## Next Concrete Engineering Step

Implement the packed miss loader behind an opt-in environment variable:

```bash
DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT=/model/packed_experts/layout.json
```

Acceptance criteria:

- correctness matches current safetensor-backed `LazyRoutedExpert`;
- first-use miss telemetry reports packed block bytes and load seconds;
- Nsight shows fewer HtoD copies per expert miss;
- Experiment 011/012 reruns show lower decode wall time for long decode.

This is the best next step because it directly combines the two strongest findings:
packed miss format plus expert cache.

