# DGX Spark Inventory

Date: 2026-05-01

Host: `spark-66c9`  
SSH user: `cole`

This is a light inventory captured over SSH. It is intended to anchor the first
round of experiments, not to be a complete benchmark.

## OS and Runtime

- Hostname: `spark-66c9`
- OS: Ubuntu 24.04.4 LTS (`noble`)
- Kernel: `6.17.0-1008-nvidia`
- Architecture: `aarch64`
- Python: `Python 3.12.3`
- Docker: `Docker version 29.3.0, build 5927d80`
- NVIDIA container runtime: present
- NVIDIA driver: `590.48.01`
- CUDA reported by `nvidia-smi`: `13.1`

## CPU and Memory

`lscpu` reports:

- Architecture: `aarch64`
- CPUs: 20
- CPU models shown: `Cortex-X925` and `Cortex-A725`
- NUMA nodes: 1

`free -h` at capture time:

```text
Mem: 121Gi total, 45Gi used, 32Gi free, 45Gi buff/cache, 76Gi available
Swap: 15Gi total, 40Mi used
```

The 121 GiB Linux-visible memory figure is consistent with a 128 GB unified
memory system after platform reservations and unit conversion.

## GPU

`nvidia-smi` reports:

```text
GPU 0: NVIDIA GB10
Driver Version: 590.48.01
CUDA Version: 13.1
Persistence-M: On
Display: Off
Temperature: 37C
Perf state: P8
GPU-Util: 0%
Compute mode: Default
Memory-Usage: Not Supported
MIG: N/A
```

`Memory-Usage: Not Supported` is expected on DGX Spark. NVIDIA documents this as
a known reporting behavior for the unified memory architecture.

## Storage

Root filesystem:

```text
/dev/nvme0n1p2  3.7T  1.4T  2.2T  38%  /
```

NVMe device:

```text
Node: /dev/nvme0n1
Model: SAMSUNG MZALC4T0HBL1-00B07
Usage: 1.44 TB / 4.10 TB
Format: 512 B + 0 B
Firmware: NXHB202Q
```

The machine has enough free local NVMe capacity for the Hugging Face checkpoint,
one or more converted layouts, and trace/benchmark artifacts.

## Experiment Implications

- Use native arm64 containers or build from source on-device. x86_64 CUDA
  containers are not viable without unacceptable emulation assumptions.
- Do not rely on `nvidia-smi` memory accounting for OOM diagnosis. Use process
  RSS/PSS, `/proc/meminfo`, framework allocator logs, CUDA errors, and kernel OOM
  logs.
- Keep swap enabled for system survivability, but do not treat swap-backed
  inference as success. A useful run should avoid sustained swap pressure.
- Treat page-cache behavior as part of the design. The checkpoint is larger than
  practical RAM, so Linux cache pressure is unavoidable in streaming experiments.
