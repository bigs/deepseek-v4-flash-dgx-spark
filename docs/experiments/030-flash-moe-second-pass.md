# Experiment 030: Flash-MoE Second Pass

Date: 2026-05-02

Source: `/Users/bigs/Code/flash-moe`

Goal: take another pass over flash-moe and extract Spark-specific research ideas.

## Transferable Ideas

| Flash-MoE idea | Spark interpretation | Decision |
| --- | --- | --- |
| Fixed per-layer packed expert files | Already our largest win. Keep pushing layout and loader. | Keep |
| Avoid framework overhead in hot path | Our miss path still creates Python/PyTorch objects per expert. | Prioritize native loader/materialization |
| Reusable aligned/staged I/O buffers | Our current packed path allocated bytes/bytearray per miss. | Test reusable staging |
| Trust OS page cache over giant custom cache | Spark differs because we cache device-resident experts, but cache pressure still matters. | Test cache policy with replay plus full inference |
| Parallel `pread` for active experts | Replay can test read scheduling without full model. | Add route replay harness |
| FMA-optimized dequant kernels | Useful once data path is no longer dominant. | Defer broad kernels |
| Sparse, shape-specialized kernels | Official sparse attention fails at production shape on GB10. | Keep as targeted kernel task |
| Speculative routing/prefetch | Flash-MoE found prediction loses because all slots must hit. Our E017 agrees. | Deprioritize |
| Compression | Flash-MoE found decompression overhead beat I/O savings; Spark already uses native FP8/FP4. | Deprioritize for now |

## Main Difference From Apple Silicon

Flash-MoE's Mac result is built around trusting the OS page cache and streaming from very
fast local SSD. On Spark, the strongest measured wins so far come from keeping selected
experts device-resident and avoiding CUDA allocator churn. The common lesson is not "no
cache"; it is "measure whole-system memory pressure and avoid clever caches that steal
from a better layer of the stack."

## Resulting Experiments

- E031: route-trace packed-read replay.
- E032: reusable slab materialization microbenchmark.
- E034: decode budget and GB10 custom-kernel priority.
- E035: full global-LRU recheck after allocator-cache fix.
- E036: full reusable-staging runtime probe.
