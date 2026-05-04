# Optimization Sprint Summary

Date: 2026-05-04

Host: `spark-66c9`

Scope: experiments 018-050.

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
| E041 | controlled full matrix, native full c1024 | 32 | 91.366s | 21.022s | 1.52 | Use full layout for deployment claims. |
| E042 | native materializer v2 | 32 | 91.926s | 21.917s | 1.46 | Primitive is useful, but not a wall-time win alone. |
| E043 | 256-slot expert arena | 32 | 91.150s | 16.516s | 1.94 | Current best deployable recipe. |
| E044 | Python materialization workers | 32 | 92.488s | 50.415s | 0.63 | Disable; overlap must move below Python. |
| E046 | hot-first full layout | 32 | 91.791s | 30.899s | 1.04 | Simple hot-first layout is a regression. |
| E049 | native CUDA materializer | 32 | 92.029s | 21.901s | 1.46 | Correct and faster than same-window baseline, but not a new absolute best. |
| E050 | arena plus native CUDA A/B best | 32 | 4.550s warm | 16.239s | 1.97 | New best observed decode recipe. |

E050 is a same-container decode-path A/B. Its later-row prefill timing is warm and is not
comparable to the cold-ish prefill rows above.

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
- Controlled runtime matrix runner and summarizer.
- Native materializer primitive for packed storage-to-parameter copies.
- Expert arena for reusing evicted expert modules.
- Hot-first full-layout repacker for physical-order experiments.
- Native copy-plan and CUDA `cudaMemcpyAsync` materializer path.
- Same-container materializer A/B harness.

## Main Finding

Packed layout was the largest early impact because it attacks the actual bottleneck:
synchronous expert miss service. The best short probe moved from `0.07 tok/s` to more than
`1 tok/s`, and the full-layout 32-token run improved from E029's `1.06 tok/s` to E050's
`1.97 tok/s`.

That is a large improvement, but still far from 20 tok/s. The best current deployable
32-token run spent `3.767s` in packed reads and `5.246s` in materialization during
`16.239s` of decode, so the next work still belongs in the native materializer and
staging path.

## Next Experiments

1. Native materializer v3: reduce planned CUDA copies to fewer larger transfers and
   batch a layer's routed misses into one native call.
2. Route-trace replay gate: screen cache/layout/order variants on longer traces before
   full inference.
3. Native deterministic overlap: CPU I/O plus copy scheduling below Python; do not use
   Python materialization workers.
4. Layout-locality sweep with explicit page-cache controls; hot-first from an 8-token
   trace already failed.
