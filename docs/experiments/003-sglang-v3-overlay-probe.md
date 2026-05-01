# Experiment 003: SGLang V3 Overlay Probe

Date: 2026-05-01

Goal: determine whether current SGLang can be used as a short path to run
`DeepSeek-V4-Flash` on one DGX Spark by presenting the checkpoint as a
DeepSeek-V3-style model.

## Attempts

### Original Hugging Face Config

Image: `lmsysorg/sglang:spark`

The original checkpoint failed before loading weights because Transformers in
the image did not recognize `model_type: deepseek_v4`.

Failure:

```text
ValueError: The checkpoint you are trying to load has model type `deepseek_v4`
but Transformers does not recognize this architecture.
```

### V3 Overlay

We created a symlink overlay with:

- `model_type: deepseek_v3`
- `architectures: ["DeepseekV3ForCausalLM"]`

Without `--skip-tokenizer-init`, SGLang selected the wrong tokenizer path and
failed before model construction.

With `--skip-tokenizer-init`, SGLang reached model construction but failed
because V3 config defaults did not include `moe_layer_freq`.

### Sparse V3 Overlay

We added sparse MoE and attention-shape fields:

- `first_k_dense_replace: 0`
- `moe_layer_freq: 1`
- `qk_nope_head_dim: 448`
- `v_head_dim: 512`
- `kv_lora_rank: null`
- `n_group: 8`
- `topk_group: 4`

This failed because SGLang's `DeepseekV2AttentionMLA` path assumes
`kv_lora_rank` is an integer.

### Sparse V3 Overlay With `kv_lora_rank=448`

We then set `kv_lora_rank: 448`, matching the apparent split implied by
`wkv.weight` shape `[512, 4096]` and `qk_rope_head_dim: 64`.

This got through argument processing and into SGLang weight loading:

```text
Load weight begin. avail mem=74.42 GB
```

The host stopped accepting SSH shortly after this point. After reboot, no
previous-boot kernel journal was available to the unprivileged user and
`dmesg` access was denied, so the exact kernel failure is unconfirmed. Given the
last log line and the model's `148.65 GiB` native payload, the working
assumption is host memory exhaustion or a unified-memory pressure path that
wedged the machine.

## Interpretation

The V3 overlay path is not a clean compatibility shim:

- Current SGLang/Transformers in `lmsysorg/sglang:spark` does not directly know
  `deepseek_v4`.
- The V3/V2 MLA code expects `kv_b_proj`-style attention internals, while the V4
  checkpoint exposes V4-specific tensors such as `wkv`, `wq_a`, `wq_b`, `wo_a`,
  and `wo_b`.
- Even a small-context launch can still try to materialize enough weights to
  exceed the Spark memory envelope unless offload or custom loading is active
  early enough.

## Safety Rule

Do not repeat a full SGLang load without guardrails. Future risky launches
should run under:

- a Docker memory cgroup limit,
- a host `MemAvailable` watchdog,
- an outer timeout,
- tmux logging,
- and page-cache eviction before the run.

`scripts/guarded_docker_run.py` provides the host-side wrapper for this.
`scripts/launch_sglang_v3kv448_guarded.sh` is the bounded repro script for this
specific overlay. It intentionally defaults to a small context and a `72g`
container memory limit; it is not a final serving recipe.

## Guarded Repro Result

After reboot, we reran the `kv_lora_rank=448` overlay with:

- `context_length=4096`
- `max_total_tokens=4096`
- `mem_fraction_static=0.35`
- Docker `--memory 72g`
- Docker `--memory-swap 80g`
- host kill threshold `MemAvailable < 48 GiB`

The run reached weight loading:

```text
Load weight begin. avail mem=115.84 GB
Detected fp8 checkpoint.
```

Then host `MemAvailable` dropped from `115.84 GiB` to `82.78 GiB` to
`36.78 GiB` across two watchdog polls. The guard killed the container:

```text
docker exited return_code=137 reason=MemAvailable 36.78 GiB below 48.00 GiB
```

The host stayed reachable, but memory did not fully recover after the container
exited:

- `free -h`: `83Gi used`, `37Gi available`
- process RSS and cgroup memory did not account for the missing memory
- `nvidia-smi` showed no running GPU processes
- `torch.cuda.mem_get_info()` reported approximately `39.1 GiB` free out of
  `130.7 GiB`
- `sudo nvidia-smi --gpu-reset -i 0` failed because the GB10 is the primary GPU

The memory was recovered without a reboot by unloading and reloading the NVIDIA
driver after stopping persistence:

```bash
sudo systemctl stop nvidia-persistenced
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo systemctl start nvidia-persistenced
```

After reload:

```text
free -h: 121Gi total, 3.5Gi used, 118Gi available
torch.cuda.mem_get_info(): (125522702336, 130663661568)
```

This is captured as `scripts/recover_nvidia_driver_memory.sh`.

Logs:

- `results/spark-66c9/sglang-v3kv448-guarded.log`
- `results/spark-66c9/torch-mem-probe-after-guard.log`
- `results/spark-66c9/torch-mem-probe-after-driver-reload.log`

## Current Status

This experiment satisfies the first stock-engine smoke test with a useful
negative result. SGLang remains useful as a source of NSA/FP8/MoE code, but the
next high-probability path is a purpose-built V4 loader/runtime or a focused
SGLang V4 port, not more blind config overlays.
