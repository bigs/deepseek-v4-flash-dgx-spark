# Flash-MoE Lessons for DGX Spark

Date: 2026-05-01

Source repo: `/Users/bigs/Code/flash-moe`

Goal: mine implementation ideas from Flash-MoE, which runs a large MoE by streaming
experts from flash, and translate them into experiments for DeepSeek V4 Flash on DGX
Spark / GB10.

## What Flash-MoE Actually Optimizes

Flash-MoE's core win is a deliberately simple hot path:

1. Non-expert weights are resident and memory mapped at startup.
2. Expert weights are repacked into one fixed-layout binary file per layer.
3. Each routed expert is a single large `pread` from a known offset.
4. The K active expert reads run in parallel.
5. GPU work is arranged into a small number of command buffers, with the expert command
   buffer submitted asynchronously so the CPU can prepare the next layer.
6. The OS page cache is trusted for reuse instead of building a large application-level
   cache by default.

The per-layer expert file format is the most directly portable idea. Flash-MoE's
`repack_experts.py` writes `packed_experts/layer_XX.bin`, where expert `E` begins at:

```text
offset = E * expert_size
```

Within each expert block, gate/up/down weights, scales, and biases are stored in fixed
order. This removes safetensor metadata traversal, name lookup, tensor slicing, and many
small reads from the inference path.

## Findings Worth Copying

### Packed Per-Layer Expert Files

This should be one of our first serious layout experiments. Our current runtime still
discovers and copies individual tensors through safetensors/PyTorch. Flash-MoE shows that
a flat per-layer expert blob enables large, predictable I/O and simple offset math.

For DeepSeek V4 Flash, the packed format should include the native hybrid FP8/FP4 expert
tensors without dequantizing to BF16. The immediate target is not a perfect final format;
it is a measurable format that lets us compare:

- safetensors tensor fetch + `.to(cuda)`
- packed expert `pread` into host memory + one HtoD transfer
- packed expert `pread` into pinned/page-aligned host staging + one HtoD transfer

### Large Parallel Reads Beat Fancy Read APIs

Flash-MoE repeatedly found that plain parallel `pread` beat more elaborate APIs in the
end-to-end pipeline. `mmap` was especially bad for cold expert reads because it turned one
large expert read into many page faults.

For Spark, the bus and memory architecture are different enough that we need to remeasure,
but the default experiment should still be:

- fixed offsets;
- large reads;
- a small persistent I/O thread pool;
- explicit alignment;
- no hot-path safetensor object work.

### Measure Cache Pressure, Do Not Assume

Flash-MoE's best default disabled its Metal LRU expert cache and trusted the OS page cache.
That result may not transfer directly to GB10. Our Nsight profile shows repeated
host-to-device expert materialization is currently the dominant cost, so some form of
GPU/device-resident expert cache is likely useful for us.

The lesson is narrower: every cache must be budgeted and measured against the memory it
steals from the page cache, KV cache, and runtime. Cache telemetry should track hits,
misses, evictions, reuse distance, and resident bytes.

### Pipeline the Deterministic Work First

Flash-MoE experimented with prediction and speculative expert prefetching, but the durable
wins came from deterministic scheduling: overlap expert reads with independent work,
batch GPU encoders, defer waits, and avoid unnecessary CPU round trips.

For our runtime, deterministic wins likely come before prediction:

- keep routed experts resident across decode steps;
- batch or fuse expert copies;
- pre-stage known next operations while the GPU is working;
- avoid creating/destroying PyTorch modules on each expert call;
- avoid forcing synchronization around every tiny operation.

### FMA Dequant Is Real, But Later

Flash-MoE's Metal kernel rewrites:

```text
(q * scale + bias) * x
```

as:

```text
fma(q, scale * x, bias * x)
```

That reduced arithmetic in the dequantized matvec. This is relevant when we write custom
FP4/FP8 kernels, but our current profile says CUDA kernel time is tiny compared with
host-to-device movement. We should not spend the next step on math kernels before reducing
weight movement.

### Persistent Prompt / KV State Matters

Flash-MoE snapshots system prompt KV and linear-attention state in serve mode. This matches
our direction: serving needs persistent KV/session state so repeated requests do not pay
full prefill every time. The persistent cache should be a first-class API/server concept,
not a side effect hidden in a benchmark harness.

## Findings To Treat Carefully

### "Trust the OS Cache"

On Apple Silicon, the OS page cache and unified memory are unusually strong. On Spark,
Linux page cache, CUDA HtoD behavior, GPU memory allocation, and unified memory behavior
will differ. We should copy the experiment, not the conclusion.

### Prediction / Speculative Prefetch

Flash-MoE's prediction attempts often lost because imperfect predictions wasted SSD and
memory bandwidth. We should only revisit prediction after the deterministic baseline is
fast enough that idle windows and routing stability are visible in traces.

### Compression

Flash-MoE's LZ4 expert compression reduced cold read size but lost end-to-end because
decompression cost and pipeline effects dominated. Compression is a later experiment, and
only if Spark measurements show flash bandwidth is the limiting resource after HtoD and
module materialization are fixed.

## Spark Experiment Backlog

### 011: Routed Expert Cache

Add a simple budgeted cache for `LazyRoutedExpert` materializations.

Measure:

- expert cache hit/miss/eviction counts;
- HtoD volume and time under Nsight;
- decode wall time;
- host available memory;
- CUDA allocation count.

This directly attacks the current immediate-eviction bottleneck.

### 012: Expert Reuse / Frequency Telemetry

Record routed expert IDs by layer and token during a prompt/decode run.

Analyze:

- unique experts per layer;
- top-N coverage per layer;
- reuse distance;
- cache size needed for 50%, 80%, and 90% hit rates.

This tells us whether a GPU expert cache can fit in Spark's memory budget.

### 013: Packed Expert Layout Microbench

Build a DeepSeek-specific packed expert format for a small layer subset.

Compare:

- current safetensors tensor fetch;
- fixed-offset `pread`;
- `mmap`;
- aligned destination buffers;
- pinned host staging;
- one large HtoD copy per expert vs current tensor-by-tensor copies.

This should run before we rewrite kernels, because layout controls everything downstream.

### 014: Packed Expert Runtime Path

Use the packed format in inference for routed experts while preserving correctness against
the current official-module path.

The first version can still call PyTorch kernels after loading, but it should remove
hot-path safetensor lookup and reduce HtoD copy fragmentation.

### 015: Custom Expert Kernel Prototype

Once movement is under control, prototype a custom routed expert path:

- read native FP4/FP8 expert blocks;
- keep activations resident;
- fuse dequant, matmul, activation, and combine where practical;
- use the FMA-style dequant rearrangement where it maps cleanly to CUDA.

## Current Priority

The next scientific step is still the expert cache, not a custom kernel. Flash-MoE's
results reinforce that the first objective is to make the weight path predictable and
measurable. Our Nsight run shows only 0.355s in CUDA kernels but 92.716s in CUDA memcpy,
so optimizing math before reducing movement would be premature.

