# Experiment Plan

Date: 2026-05-01

Goal: get `deepseek-ai/DeepSeek-V4-Flash` running on a single DGX Spark in native
FP4/FP8 mixed precision, then optimize toward useful throughput by exploiting
local NVMe and MoE expert sparsity.

Assumption: upstream GB10 support is immature. Stock frameworks are useful for
smoke tests, kernel references, and correctness comparisons, but this project is
explicitly allowed to write custom CUDA kernels, custom FP4/FP8 layouts, custom
weight conversion tools, and a minimal purpose-built runtime if that is the most
direct path to a good Spark result.

## Stop Conditions

Call a path "not currently viable" when one of these is true:

- It requires multiple GPUs in code paths that cannot be disabled or shimmed.
- It expands the native checkpoint into BF16/FP16 enough to exceed memory before
  generation starts.
- It requires x86_64-only CUDA kernels or containers with no practical arm64
  build path.
- It enters sustained swap-heavy execution for normal single-user inference.
- It cannot produce a correct short completion after focused debugging.

Call a path "initially viable" when it can:

- Load or stream the native FP4/FP8 checkpoint on one Spark.
- Produce deterministic short completions from a tiny prompt.
- Run with one sequence without sustained swap, then scale context length while
  measuring KV-cache and expert-cache pressure separately.
- Preserve FP4 expert storage rather than silently converting all experts upward.

## Phase 0: Host Preparation

1. Create a dedicated workspace on Spark, for example:

   ```bash
   mkdir -p ~/models/deepseek-v4-flash ~/runs/deepseek-v4-flash
   ```

2. Confirm the live baseline before each serious run:

   ```bash
   uname -a
   free -h
   nvidia-smi
   df -h /
   docker info --format 'Default={{.DefaultRuntime}} Runtimes={{json .Runtimes}}'
   ```

3. Install or confirm basic tools:

   ```bash
   python3 -m pip --version
   git --version
   git lfs version
   huggingface-cli --help
   ```

## Phase 1: Download and Inspect Checkpoint

Download with Hugging Face tooling, preserving the upstream format:

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir ~/models/deepseek-v4-flash/hf \
  --local-dir-use-symlinks False
```

Inspect without loading tensors:

```bash
du -sh ~/models/deepseek-v4-flash/hf
find ~/models/deepseek-v4-flash/hf -name '*.safetensors' -maxdepth 1 -print | wc -l
jq '.weight_map | length' ~/models/deepseek-v4-flash/hf/model.safetensors.index.json
jq '.expert_dtype, .quantization_config, .n_routed_experts, .num_experts_per_tok' \
  ~/models/deepseek-v4-flash/hf/config.json
```

Write a small inspector later to summarize tensor names, dtypes, shapes, and
per-shard byte counts from safetensor metadata only. This should answer how much
of the repo is dense/router payload versus expert payload before we try to run it.

## Phase 2: Stock Engine Smoke Tests

Test stock paths first because support is moving quickly.

Risk control: any stock-engine launch that can load weights must run with a
Docker memory limit, host `MemAvailable` watchdog, timeout, and persistent log.
On DGX Spark, `nvidia-smi` cannot report memory usage and a bad unified-memory
load can make the host unreachable before a clean Python exception is logged.

### SGLang

Questions:

- Is there a working `linux/arm64` image for the documented Grace-Blackwell path?
- If not, does source build work on GB10 with CUDA 13.1?
- Does the DeepSeek-V4 path hard-require multiple visible GPUs?

Expected risk: documentation targets four GPUs for V4-Flash.

### vLLM

Questions:

- Does current vLLM support GB10 `sm_121` and arm64 for the DeepSeek-V4 kernels?
- Can it start with `--data-parallel-size 1` and no expert parallelism, even if slow?
- Can `--kv-cache-dtype fp8` and `--max-num-seqs 1` keep memory inside the Spark
  envelope while context length is scaled upward?

Expected risk: the documented recipe is for larger multi-GPU systems and uses
expert parallelism.

### Official DeepSeek Inference Code

Questions:

- Does conversion work with `MP=1`?
- Does generation work with `torchrun --nproc-per-node 1`?
- Does it preserve native FP4 experts?

Expected risk: the example uses `MP=4`; `MP=1` may expose tensor-parallel
assumptions or memory pressure.

## Phase 3: Memory and IO Characterization

Run these before attempting custom streaming:

- KV-cache allocation behavior at 4K, 32K, 128K, 512K, and 1M token limits.
- Sequential NVMe read throughput against a large local file.
- Random read throughput at block sizes matching candidate expert/tile chunks.
- Page-cache behavior during repeated reads of the same shard subset.
- Major/minor page faults during metadata-only safetensor scans and real loads.
- CUDA allocation behavior on GB10 using small probes, because `nvidia-smi` memory
  accounting is unavailable.

Useful tools:

```bash
fio
iostat -x 1
pidstat -r -d 1
perf stat -e page-faults,major-faults
/usr/bin/time -v
cat /proc/meminfo
dmesg -T | tail -200
```

## Phase 4: Streaming Design Spike

If stock engines cannot load the model, build a minimal prototype around the
official architecture:

1. Parse safetensor metadata and map tensor names to layer/expert ownership.
2. Identify the exact expert tensor layout and FP4 packing.
3. Build a read-only mmap-backed expert tensor cache.
4. Keep dense weights resident if possible.
5. Route one layer at a time and fetch only top-k expert weights.
6. Add prefetch once router decisions for the next layer/token are visible.
7. Measure whether IO, FP4 decode, tensor-core compute, or Python overhead is the
   dominant bottleneck.

The first prototype can be ugly if it proves the core residency model. Production
quality only matters after correctness and rough throughput are measurable.

## Phase 4.5: Custom Kernel Track

Start this track as soon as metadata inspection identifies the expert tensor
layout. It does not need to wait for every stock-engine test to fail.

Candidate kernels/components:

- FP4 expert weight unpack/decode path matching DeepSeek's native checkpoint
  representation.
- Fused routed-expert matmul for top-6 experts with small active batch sizes.
- Cache-aware expert block loader with explicit prefetch and eviction hooks.
- FP8 dense/attention helpers only where framework kernels are missing or
  silently promote precision.
- Microbenchmarks for GB10 `sm_121` that isolate FP4 tensor-core throughput,
  unpack overhead, memory bandwidth, and NVMe-fed cache refill cost.

Success for this track is not a full engine at first. It is a measured kernel or
loader primitive that beats the framework fallback, preserves native precision,
or makes a previously impossible memory layout possible.

## Phase 5: Optimize

Optimization order:

1. Avoid format expansion.
2. Improve on-disk layout for expert-local reads.
3. Pin or prefetch hot experts.
4. Replace framework fallbacks with custom kernels when they block GB10 support
   or waste memory bandwidth.
5. Tune expert cache size against resident KV cache and context length.
6. Batch prompts only after single-sequence decode works.
7. Revisit CUDA Graphs / compiled execution after the IO and kernel paths are stable.

## First Concrete Milestones

- `M1`: checkpoint downloaded and metadata summarized.
- `M2`: one stock engine smoke-tested with exact failure or success log.
  SGLang/V3-overlay has a negative result in
  `docs/experiments/003-sglang-v3-overlay-probe.md`.
- `M3`: official inference conversion tested with `MP=1`.
- `M4`: memory/IO baseline captured on Spark.
- `M5`: minimal expert metadata map created.
- `M6`: first custom GB10 microbenchmark for native FP4/FP8 expert work.
- `M7`: first correct short generation, regardless of speed. Completed in
  `docs/experiments/004-lazy-runtime-first-tokens.md`.
- `M8`: first measured tokens/sec with native FP4/FP8 and no sustained swap.

## Current Custom-Runtime Direction

The stock SGLang probe shows that normal framework loading is not viable on one
Spark: it tries to materialize too much of the checkpoint before any useful
offload policy can act.

The official DeepSeek inference code is the best architecture reference. A
full-context meta inspection and CUDA allocation probe show that this resident
set fits:

- full 1M-context official buffers / sparse KV state,
- all non-routed-expert parameters,
- temporary routed-expert scratch.

The routed expert payload must not be permanently resident. The next runtime
prototype should load dense weights normally, replace routed experts with a lazy
top-k expert loader, and start with slow correctness-first kernels before
optimizing.

## Serving Direction

The next product shape is a narrow custom inference server rather than waiting
for full SGLang/vLLM support. It should expose an OpenAI-compatible API, but keep
the engine small and Spark-specific so we can own GB10 kernels, lazy FP4 expert
loading, and persistent full-context cache behavior.

Persistent KV/compressed cache reuse is a hard requirement. The first server can
support one active cached session plus a FIFO queue, then grow into multiple
sessions only after the single-session path is reliable and measured.

See `docs/serving-design.md`.
