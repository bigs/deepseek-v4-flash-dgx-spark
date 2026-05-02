# Optimization Sprint Summary

Date: 2026-05-02

Host: `spark-66c9`

Scope: experiments 018-029.

## Result Table

| Experiment | Change / Question | Tokens | Prefill | Decode | Decode tok/s | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline | safetensors lazy route cache, 1024 entries | 8 | 125.083s | 115.604s | 0.07 | Replace scattered miss loading. |
| E020 | route-filtered packed layout, global LRU 1024 | 8 | 95.067s | 15.642s | 0.51 | Packed layout is the main win. |
| E020b | route-filtered packed layout, global LRU 2048 | 8 | 95.830s | 13.900s | 0.58 | More capacity helps less than miss cost. |
| E021 | layer-aware LRU 1024 | 8 | 94.988s | 12.750s | 0.63 | Keep layer-aware cache policy. |
| E022 | Python threaded materialization | 8 | 93.517s | 16.170s | 0.49 | Keep opt-in only; do not enable. |
| E023 | per-miss pinned staging | N/A | N/A | N/A | N/A | Correct but slower; needs persistent staging pool. |
| E025 | official sparse attention at production shapes | N/A | N/A | N/A | N/A | Fails on GB10 shape; keep fallback. |
| E026 | avoid `torch.cuda.empty_cache()` on cached evictions | 8 | 91.662s | 6.434s | 1.24 | New short-probe best. |
| E027 | full all-expert packed layout | 8 | 92.612s | 8.283s | 0.97 | Viable deployment artifact, slower locality. |
| E028 | inline postfill | 8 | 95.424s | 16.173s | 0.49 | Not a default latency policy. |
| E029 | full layout, 32-token decode | 32 | 92.102s | 30.307s | 1.06 | Sustained decode remains miss-bound. |

## Integrated Changes

- Full and route-filtered packed expert layout builder.
- Packed expert loader behind `DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT`.
- Packed tensor correctness validation against safetensors.
- Layer-aware expert cache policy.
- Allocator-cache-safe eviction default: cached evictions no longer call
  `torch.cuda.empty_cache()` unless explicitly requested.
- Opt-in probes for parallel materialization, pinned staging, sparse-attention shape
  testing, and long-decode serving measurements.
- HTTP probe error reporting now includes response bodies, which exposed E029's initial
  harness-path failure.

## Main Finding

Packed layout was the largest impact because it attacks the actual bottleneck: synchronous
expert miss service. The best short probe moved from `0.07 tok/s` to `1.24 tok/s`, and the
full-layout 32-token run sustained `1.06 tok/s`.

That is a large improvement, but still far from 20 tok/s. The packed loader still pays too
much per expert: the 32-token run spent `11.527s` in packed reads and `17.106s` in
materialization during `30.307s` of decode.

## Next Experiments

1. Route-trace replay harness: replay expert miss/hit sequences without the full model loop
   so loader/cache/layout changes can be benchmarked quickly.
2. Slab-backed materialization: one reusable host slab and one reusable device slab per
   expert block, with tensor views instead of per-tensor allocations.
3. Layout-locality sweep: compare full layout variants by physical ordering, including
   layer-local hot experts first and route-trace-clustered expert order.
