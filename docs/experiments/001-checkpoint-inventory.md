# Experiment 001: Checkpoint Inventory

Goal: inspect `deepseek-ai/DeepSeek-V4-Flash` metadata on DGX Spark without
loading tensor payloads.

Host:

- `spark-66c9`
- Ubuntu 24.04.4 aarch64
- NVIDIA GB10
- Model path: `/home/cole/models/deepseek-v4-flash/hf`

Command:

```bash
python3 scripts/inspect_checkpoint.py \
  --model-dir ~/models/deepseek-v4-flash/hf \
  --out-dir results/spark-66c9
```

Expected outputs:

- `results/spark-66c9/checkpoint-inventory.summary.json`
- `results/spark-66c9/checkpoint-inventory.md`
- `results/spark-66c9/checkpoint-tensors.csv`

Questions answered:

- How many bytes are routed expert payload versus dense/router/attention payload?
- Which dtypes are actually present in the safetensors?
- Are expert tensors contiguous enough for direct mmap/page-cache reads?
- How much expert payload exists per layer?
- Which shard owns each expert tensor?

## Result

Run completed on `spark-66c9` at `2026-05-01T17:43:21Z`.

Artifacts:

- `results/spark-66c9/checkpoint-inventory.summary.json`
- `results/spark-66c9/checkpoint-inventory.md`
- `results/spark-66c9/checkpoint-tensors.csv`

High-level findings:

- Total tensor payload is `148.65 GiB` across `69,187` tensors and `46`
  safetensor shards.
- Core routed experts dominate the checkpoint: `137.06 GiB`.
- Core routed expert payload is exactly `3.19 GiB` per layer for all `43`
  layers.
- Core shared experts are only `1.01 GiB` total and should be treated as
  resident dense-ish weights, not streamed cold expert state.
- Core attention dense weights are `5.03 GiB`.
- Embedding plus output head are about `1.97 GiB` total.
- Packed expert weights appear as `I8` payloads with `F8_E8M0` scales.
- No tensors are left unclassified by the current inspector.

Initial conclusion:

The first custom runtime target should be a layer/expert-local reader and cache
for `core_routed_experts`. The Hugging Face shard layout is already strongly
layer-local: one core layer's routed experts are `3.19 GiB`, and the complete
core routed expert component spans `43` files. That is favorable for a first
streaming/cache prototype, even if we later convert to a more tile-local runtime
layout.
