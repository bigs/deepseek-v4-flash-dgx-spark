# Experiment 024: Prefetch Feasibility

Date: 2026-05-01

Host: `spark-66c9`

Goal: decide whether to add async expert prefetch to the current Python runtime.

## Evidence

Two prior experiments constrain the obvious prefetch designs:

- Experiment 017 tested naive temporal expert prediction and found `0%` all-slot hit rate.
  That means prefetching the next token's full expert set from the previous token's slots
  is not reliable enough.
- Experiment 022 tested active-expert pre-materialization with Python worker threads. It
  regressed decode from `6.434s` to `16.170s`, which means Python-threaded CUDA
  materialization is the wrong overlap boundary.

The current runtime only knows exact routed experts after each layer's gate runs. At that
point, prefetching those same experts must either:

- happen before the layer's expert compute and add latency;
- overlap with shared-expert compute, which still risks CUDA allocator/copy contention;
- or move into a lower-level CPU I/O plus CUDA stream design.

## Decision

Do not add async prefetch to the Python runtime yet.

The next credible version is a custom extension or worker pipeline that keeps prefetch
CPU-only until a controlled CUDA copy point:

1. persistent file descriptors;
2. persistent pageable or pinned staging buffers;
3. route-aware CPU reads after gate output;
4. explicit CUDA stream copies or fused unpack/copy kernels;
5. no Python-threaded `torch.nn.Module` construction on the hot path.

