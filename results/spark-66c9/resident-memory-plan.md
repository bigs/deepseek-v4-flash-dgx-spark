# Resident Memory Plan Probe

Date: 2026-05-01  
Host: `spark-66c9`

Goal: prove that the intended full-context custom runtime residency plan can
fit on one DGX Spark if routed experts are not permanently resident.

Command source:

```bash
python3 scripts/probe_resident_memory_plan.py \
  --inference-dir /model/inference \
  --config /model/inference/config.json \
  --max-seq-len 1048576 \
  --expert-scratch-gib 2.0
```

Guard:

- Docker image: `vllm/vllm-openai:v0.17.1-cu130`
- Docker memory: `80g`
- Docker memory+swap: `88g`
- Host kill threshold: `MemAvailable < 32 GiB`

Result:

```text
cuda_initial_mem_get_info=(124706619392, 130663661568)
max_seq_len=1048576
resident_planned_bytes=43765422920 (40.76 GiB)
routed_expert_skipped_bytes=150592290816 (140.25 GiB)
cuda_after_allocate_mem_get_info=(80192946176, 130663661568)
cuda_after_free_mem_get_info=(80875372544, 130663661568)
```

Interpretation:

The planned resident set fits with full 1M context:

- all official runtime buffers, including full-context sparse/compressed KV
  caches,
- all non-routed-expert parameters,
- plus `2 GiB` of routed expert scratch.

This validates the first custom-runtime shape: keep dense and cache state
resident, leave routed experts off resident memory, and materialize only the
top-k expert tensors needed for the current layer/token.

After the probe exited, the GB10 driver retained memory until the NVIDIA modules
were unloaded/reloaded. Future large CUDA probes should use
`scripts/guarded_docker_run.py --recover-nvidia-driver-on-exit`.
