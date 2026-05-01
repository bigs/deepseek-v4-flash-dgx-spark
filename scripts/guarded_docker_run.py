#!/usr/bin/env python3
"""Run a GPU Docker command with host-memory guardrails.

This script is intended to run on the DGX Spark host, usually inside tmux. It
keeps stdout/stderr in a log file, applies a Docker memory cgroup limit, and
kills the container if host MemAvailable falls below a configured threshold.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
import time
from pathlib import Path


GIB = 1024**3


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def mem_available_gib() -> float:
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                kib = int(line.split()[1])
                return kib * 1024 / GIB
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def docker_kill(name: str, log) -> None:
    print(f"[{timestamp()}] guard: docker rm -f {name}", file=log, flush=True)
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Docker container name.")
    parser.add_argument("--image", required=True, help="Docker image to run.")
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--memory", default="96g", help="Docker --memory value.")
    parser.add_argument(
        "--memory-swap",
        default="104g",
        help="Docker --memory-swap value. Use empty string to omit.",
    )
    parser.add_argument(
        "--min-mem-available-gib",
        type=float,
        default=16.0,
        help="Kill container if host MemAvailable drops below this GiB value.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=1800.0,
        help="Kill container after this many seconds.",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument(
        "--mount",
        action="append",
        default=[],
        metavar="HOST:CONTAINER[:MODE]",
        help="Bind mount passed to docker -v. May be repeated.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable passed to docker -e. May be repeated.",
    )
    parser.add_argument("--entrypoint", help="Optional Docker entrypoint.")
    parser.add_argument(
        "--recover-nvidia-driver-on-exit",
        action="store_true",
        help="After Docker exits, unload/reload NVIDIA modules to release retained GB10 unified memory.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("container command is required after --")
    return args


def build_docker_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        args.name,
        "--runtime=nvidia",
        "--gpus",
        "all",
    ]
    if args.memory:
        cmd.extend(["--memory", args.memory])
    if args.memory_swap:
        cmd.extend(["--memory-swap", args.memory_swap])
    for mount in args.mount:
        cmd.extend(["-v", mount])
    for env in args.env:
        cmd.extend(["-e", env])
    if args.entrypoint:
        cmd.extend(["--entrypoint", args.entrypoint])
    cmd.append(args.image)
    cmd.extend(args.command)
    return cmd


def recover_nvidia_driver(log) -> None:
    commands = [
        ["sudo", "-n", "systemctl", "stop", "nvidia-persistenced"],
        ["sudo", "-n", "modprobe", "-r", "nvidia_uvm", "nvidia_drm", "nvidia_modeset", "nvidia"],
        ["sudo", "-n", "modprobe", "nvidia"],
        ["sudo", "-n", "modprobe", "nvidia_uvm"],
        ["sudo", "-n", "systemctl", "start", "nvidia-persistenced"],
    ]
    print(f"[{timestamp()}] guard: recovering NVIDIA driver memory", file=log, flush=True)
    for cmd in commands:
        print(f"[{timestamp()}] guard: {shlex.join(cmd)}", file=log, flush=True)
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            print(
                f"[{timestamp()}] guard: recovery command failed rc={result.returncode}",
                file=log,
                flush=True,
            )
            break
    subprocess.run(["free", "-h"], stdout=log, stderr=subprocess.STDOUT, check=False)
    subprocess.run(["nvidia-smi"], stdout=log, stderr=subprocess.STDOUT, check=False)


def main() -> int:
    args = parse_args()
    docker_cmd = build_docker_command(args)
    rendered = shlex.join(docker_cmd)

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(rendered)
        return 0

    with args.log_file.open("a", buffering=1) as log:
        print(f"==== {timestamp()} guarded docker run start ====", file=log)
        print(f"command: {rendered}", file=log)
        print(
            f"guard: memory={args.memory} memory_swap={args.memory_swap} "
            f"min_mem_available_gib={args.min_mem_available_gib} "
            f"timeout_seconds={args.timeout_seconds}",
            file=log,
            flush=True,
        )
        try:
            print(
                f"[{timestamp()}] guard: initial MemAvailable={mem_available_gib():.2f} GiB",
                file=log,
                flush=True,
            )
        except Exception as exc:
            print(f"[{timestamp()}] guard: meminfo read failed: {exc!r}", file=log)

        proc = subprocess.Popen(docker_cmd, stdout=log, stderr=subprocess.STDOUT)
        start = time.monotonic()
        reason: str | None = None

        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                try:
                    available = mem_available_gib()
                    print(
                        f"[{timestamp()}] guard: elapsed={elapsed:.1f}s "
                        f"MemAvailable={available:.2f} GiB",
                        file=log,
                        flush=True,
                    )
                    if available < args.min_mem_available_gib:
                        reason = (
                            f"MemAvailable {available:.2f} GiB below "
                            f"{args.min_mem_available_gib:.2f} GiB"
                        )
                        docker_kill(args.name, log)
                        break
                except Exception as exc:
                    print(f"[{timestamp()}] guard: meminfo read failed: {exc!r}", file=log)

                if elapsed > args.timeout_seconds:
                    reason = f"timeout after {elapsed:.1f}s"
                    docker_kill(args.name, log)
                    break

                time.sleep(args.poll_interval)
        except KeyboardInterrupt:
            reason = "keyboard interrupt"
            docker_kill(args.name, log)

        return_code = proc.wait()
        print(
            f"[{timestamp()}] guard: docker exited return_code={return_code} "
            f"reason={reason or 'process exited'}",
            file=log,
            flush=True,
        )
        if args.recover_nvidia_driver_on_exit:
            recover_nvidia_driver(log)

    if reason is None:
        return return_code
    if reason.startswith("timeout"):
        return 124
    return 137


if __name__ == "__main__":
    sys.exit(main())
