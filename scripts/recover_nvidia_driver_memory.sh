#!/usr/bin/env bash
set -euo pipefail

# Recover GB10 unified memory retained by the NVIDIA driver after a killed CUDA
# process. Run on the DGX Spark host over SSH. This does not reboot the machine.

echo "before:"
free -h
nvidia-smi || true

echo "stopping nvidia-persistenced"
sudo -n systemctl stop nvidia-persistenced

echo "unloading NVIDIA kernel modules"
sudo -n modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia

echo "after unload:"
free -h
lsmod | egrep '^nvidia' || true

echo "reloading NVIDIA kernel modules"
sudo -n modprobe nvidia
sudo -n modprobe nvidia_uvm
sudo -n systemctl start nvidia-persistenced

echo "after reload:"
free -h
nvidia-smi
