# SGLang V3KV448 Guarded Repro

Date: 2026-05-01  
Host: `spark-66c9`

Command source: `scripts/launch_sglang_v3kv448_guarded.sh`

This run repeated the previous SGLang V3 overlay probe with a small context and
host guardrails:

- Docker image: `lmsysorg/sglang:spark`
- Overlay: `/home/cole/models/deepseek-v4-flash/overlay-deepseek-v3-sparse-kv448`
- Context length: `4096`
- Max total tokens: `4096`
- KV cache dtype: `fp8_e4m3`
- Docker memory: `72g`
- Docker memory+swap: `80g`
- Host kill threshold: `MemAvailable < 48 GiB`
- Timeout: `900s`

Result:

```text
Load weight begin. avail mem=115.84 GB
Detected fp8 checkpoint.
guard: elapsed=44.0s MemAvailable=82.78 GiB
guard: elapsed=46.0s MemAvailable=36.78 GiB
guard: docker exited return_code=137 reason=MemAvailable 36.78 GiB below 48.00 GiB
```

Post-run state:

```text
free -h: 121Gi total, 83Gi used, 37Gi available
torch.cuda.mem_get_info(): (39137210368, 130663661568)
nvidia-smi: no running GPU processes
```

Recovery without reboot:

```bash
sudo systemctl stop nvidia-persistenced
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo systemctl start nvidia-persistenced
```

After driver reload:

```text
free -h: 121Gi total, 3.5Gi used, 118Gi available
torch.cuda.mem_get_info(): (125522702336, 130663661568)
```

Interpretation:

SGLang's stock weight loader is not a viable single-Spark path for this
checkpoint without a much earlier streaming/offload intervention. Even with a
4K context, startup weight loading rapidly consumes enough unified memory to
threaten host stability. The guard prevented an SSH-loss reboot, but the NVIDIA
driver still retained or reserved memory after container kill.
