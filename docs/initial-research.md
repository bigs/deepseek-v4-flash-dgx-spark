# DeepSeek-V4-Flash on DGX Spark: Initial Research

Date: 2026-05-01

This repo is for investigating whether `deepseek-ai/DeepSeek-V4-Flash` can run
on a single NVIDIA DGX Spark in its native FP4/FP8 mixed checkpoint format, and
how far we can push flash-backed weight streaming if the full native checkpoint
does not fit in the 128 GB unified memory envelope.

## Current Model Facts

Primary model: [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

Observed/reported facts:

- Hugging Face lists the repo at about 160 GB, with 46 safetensor shards.
- License is MIT.
- Architecture is `DeepseekV4ForCausalLM`.
- It is a fine-grained MoE model: 284B total parameters, 13B activated per token.
- It supports a 1,048,576 token context window.
- The instruct checkpoint is mixed precision: MoE expert weights are FP4, while
  most other parameters are FP8.
- Config details include 43 layers, 256 routed experts plus 1 shared expert,
  top-6 expert routing, `hidden_size=4096`, `head_dim=512`, `num_attention_heads=64`,
  `num_key_value_heads=1`, `sliding_window=128`, and YaRN rope scaling.
- The official model card says there is no Jinja chat template; prompt encoding
  is handled by the repository's `encoding/` Python code.
- The official `inference/` folder provides a simple conversion and `torchrun`
  path. Its default example uses `MP=4`, which is a warning sign for a single
  GB10 because Spark is one GPU, not a four-GPU data-center node.

Useful source links:

- Model card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/README.md
- Config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json
- Inference folder: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/inference
- Encoding folder: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/encoding

## Runtime Ecosystem Snapshot

The model is new enough that support should be treated as moving quickly.

Known current routes:

- **Official DeepSeek inference code**: best source of architectural truth, but
  appears oriented around multi-process/model-parallel execution and conversion
  rather than production serving on a constrained UMA machine.
- **vLLM recipe**: documents `deepseek-ai/DeepSeek-V4-Flash` as a 284B/13B,
  1M-context MoE and recommends FP8 KV cache plus expert parallelism. The recipe
  is written for larger multi-GPU systems such as GB200/H200/B200/B300 rather
  than one Spark.
- **SGLang recipe**: documents DeepSeek-V4 support and provides separate
  Blackwell/Grace-Blackwell/Hopper container images. Its own hardware guidance
  lists V4-Flash as a four-GPU single-node serving target on B200/GB200/GB300/H200.
- **NVIDIA NeMo AutoModel**: has a DeepSeek V4 Flash coverage page, but the full
  43-layer schedule is documented as requiring multi-node guidance for fine-tuning.

Useful source links:

- vLLM recipe: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
- SGLang DeepSeek-V4 docs: https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4
- NVIDIA NeMo AutoModel page: https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/llm/deepseek-ai/dsv4-flash.html

## DGX Spark / GB10 Facts That Matter

Primary hardware target: NVIDIA DGX Spark with GB10 Grace Blackwell.

Relevant public specs:

- 20-core Arm CPU, reported by NVIDIA as 10 Cortex-X925 plus 10 Cortex-A725.
- Integrated Blackwell GPU with fifth-generation Tensor Cores and FP4 support.
- 128 GB coherent unified LPDDR5x system memory.
- 256-bit memory interface, 273 GB/s memory bandwidth.
- Up to 1 PFLOP / 1,000 TOPS at FP4 precision with sparsity.
- 6,144 CUDA cores.
- Two copy engines.
- 1 TB or 4 TB NVMe M.2 storage.
- NVIDIA positions one Spark for models up to 200B parameters, or two Sparks for
  up to 405B parameters.

Important caveat: parameter count alone is a rough marketing proxy. DeepSeek-V4-Flash
is 284B total but sparse with 13B active and a 160 GB native mixed-precision
checkpoint. That puts it outside the normal one-Spark "up to 200B" comfort zone
but close enough to justify experiments.

Useful source links:

- DGX Spark product page: https://www.nvidia.com/en-gb/products/workstations/dgx-spark/
- DGX Spark hardware guide: https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- DGX Spark porting guide: https://docs.nvidia.com/dgx/dgx-spark-porting-guide/dgx-spark-porting-guide.pdf

## Initial Assessment

The native checkpoint is larger than available unified memory after OS/runtime
overhead. A naive load of all weights plus runtime workspace is unlikely to work
on one Spark.

The KV cache is a different story. Because the model has 43 layers, one
key/value head, and 512-dimensional KV heads, a conventional per-token KV cache
estimate is:

```text
43 layers * 1 KV head * 512 values * 2 tensors (K,V) * dtype_size
```

At the full 1,048,576-token context this is roughly:

- FP8 KV cache: 46.2 GB decimal / 43.0 GiB.
- BF16 or FP16 KV cache: 92.3 GB decimal / 86.0 GiB.

That is large but not absurd on a 128 GB unified-memory machine, especially if
the rest of the runtime avoids keeping the full 160 GB checkpoint resident. This
means the central memory problem is weight residency and expert movement, not KV
cache capacity by itself.

We should not wait for upstream libraries to make GB10 first-class. Current
vLLM, SGLang, PyTorch, Triton, and TensorRT-LLM behavior should be treated as
useful probes and reference implementations, not as hard boundaries. If the best
path requires custom CUDA kernels, custom FP4 unpack/layout code, custom
safetensor-to-runtime conversion, or a small purpose-built inference loop, that
is in scope.

The most promising path is not "fit the whole model"; it is to exploit the MoE
structure:

1. Keep dense weights, routing, active KV cache, and hot expert working sets in
   unified memory.
2. Keep cold expert weights on local NVMe.
3. Use a streaming or demand-paged expert cache so each token/layer only pays for
   the top-6 routed experts, with prefetch based on router output where possible.
4. Start with short contexts and one sequence to debug weight residency first,
   then scale context length once the expert-cache behavior is understood.

The bottlenecks will probably be memory bandwidth and page movement rather than
raw FP4 tensor-core throughput. GB10 has excellent capacity for its size, but
273 GB/s LPDDR5x bandwidth is much closer to a high-end integrated-memory system
than to HBM-equipped B200/H200 systems.

## What Makes This Plausible

- Native FP4 experts are already compact; no lossy re-quantization is required
  for the largest portion of the model.
- MoE sparsity means only a fraction of expert weights are used per token.
- The local NVMe has enough capacity for the checkpoint and converted formats.
- UMA eliminates explicit CPU-RAM-to-VRAM copies for CPU-produced metadata and
  can make some memory-management strategies simpler than on discrete GPUs.
- GB10 supports FP4 tensor cores, so the target precision is architecturally
  aligned with the machine.

## What Makes This Hard

- Stock serving engines currently document V4-Flash as a multi-GPU target.
- The official inference path assumes model parallelism in examples.
- Full native checkpoint size is larger than practical available memory on one
  Spark.
- KV cache is feasible enough to keep in memory, but it still competes directly
  with resident dense weights, hot experts, allocator overhead, page cache, and
  OS memory. FP8 KV remains the preferred target where supported.
- Direct storage-to-GPU fast paths may not behave like discrete-GPU GDS systems
  on GB10. NVIDIA documents that GPUDirect RDMA-style mechanisms are not supported
  on DGX Spark because memory from CUDA device allocators cannot be coherently
  accessed by the CPU complex or PCIe devices.
- Any custom streaming path has to preserve FP4/FP8 kernel compatibility and avoid
  accidentally expanding experts to BF16/FP16 in memory.

Relevant storage / IO source links:

- NVIDIA GPUDirect RDMA note for DGX Spark: https://nvidia.custhelp.com/app/answers/detail/a_id/5780
- GPUDirect Storage overview: https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html

## Working Hypotheses

1. **Stock vLLM/SGLang will probably not run the native checkpoint on one Spark
   without modification.** They are still worth testing first because support is
   moving quickly and failure modes will tell us exactly where assumptions break.
2. **Custom GB10-specific kernels and layouts are likely necessary for the best
   result.** Upstream library lag should not block the project; the goal is to
   learn the machine and model deeply enough to build missing pieces.
3. **Linux page cache plus mmap may be the first practical "stream from flash"
   mechanism.** It is less exotic than true direct IO, fits UMA better, and can
   be profiled with ordinary tools.
4. **Expert-level caching is the central optimization target.** Streaming whole
   layers or whole safetensor shards will be too coarse; the useful unit is likely
   expert block/tile granularity, constrained by the kernel format.
5. **The first success criterion should be correctness at terrible speed.** Once
   one-token or short-response generation works, optimize residency, prefetch, and
   batching.
6. **A converted runtime layout may be necessary.** Safetensors are a distribution
   format, not necessarily the right on-disk layout for low-latency expert paging.

## Immediate Next Questions

- Can current vLLM or SGLang containers run at all on GB10/aarch64 with their
  DeepSeek-V4 kernels?
- Do those containers provide `linux/arm64` images, or do we need local source
  builds?
- Does the official DeepSeek inference code support `MP=1` after conversion, or
  are there hard-coded tensor/model parallel assumptions?
- What is the smallest context and batch shape that can initialize without OOM?
- What is the actual KV-cache layout used by each runtime, and does it match the
  simple one-KV-head estimate or add hidden padding/alignment overhead?
- How much of the 160 GB checkpoint is expert weight payload versus dense/router
  payload?
- Are FP4 experts stored in a layout that can be sliced per expert without
  heavyweight decompression?
- Can we observe page-cache behavior and major faults clearly enough on Spark to
  guide a streaming cache design?
