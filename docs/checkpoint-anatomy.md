# Checkpoint Anatomy Workflow

The first scientific milestone is to understand the checkpoint layout before
trying to serve it. `scripts/inspect_checkpoint.py` reads only safetensor headers
and `model.safetensors.index.json`; it does not load tensor payloads into memory.

Run on DGX Spark after downloading the model:

```bash
python3 scripts/inspect_checkpoint.py \
  --model-dir ~/models/deepseek-v4-flash/hf \
  --out-dir results/spark-66c9
```

Outputs:

- `checkpoint-inventory.summary.json`: structured summary by dtype, component,
  layer, shard, and contiguity.
- `checkpoint-inventory.md`: human-readable report.
- `checkpoint-tensors.csv`: one row per tensor with dtype, shape, component,
  shard, and byte offsets.

The most important question is whether routed expert tensors are stored in a
layout that allows expert-local or layer-local reads without dragging unrelated
weights through memory. That answer determines whether we can rely on mmap/page
cache over the Hugging Face layout or need a converted runtime layout.

The first Spark run is checked into `results/spark-66c9/`. The headline result:
`137.06 GiB` of the `148.65 GiB` tensor payload is core routed experts, with
`3.19 GiB` per core layer. This makes routed expert cache design the dominant
problem.

Experiment 002 adds read benchmarks over these offsets. The current Hugging Face
layout is usable for a first expert-cache prototype: top-6 expert selections read
`76.50 MiB`, coalesce to 2-12 ranges depending on expert order, and show near
`1.00x` read amplification.

The inspector classifies names into broad components:

- `core_routed_experts`
- `mtp_routed_experts`
- `core_shared_experts`
- `mtp_shared_experts`
- `core_router`
- `mtp_router`
- `core_attention_dense`
- `mtp_attention_dense`
- `core_norms`
- `mtp_norms`
- `final_norm`
- `core_hyper_correction`
- `mtp_hyper_correction`
- `embeddings`
- `output_head`
- `auxiliary_heads`
- `mtp_projection`
- `other`

This is intentionally conservative. If new tensor names land in `other`, inspect
them before changing the runtime plan.
