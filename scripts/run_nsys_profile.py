#!/usr/bin/env python3
"""Run a guarded Docker inference probe under Nsight Systems.

This script is intended to run on the DGX Spark host from the repo root. It uses
the host Nsight Systems installation by bind-mounting it into the runtime
container, so the base runtime image does not need to include `nsys`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="nsys-hello")
    parser.add_argument("--image", default="deepseek-v4-flash-spark:tilelang")
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--model-dir", type=Path, default=home / "models/deepseek-v4-flash/hf")
    parser.add_argument("--run-dir", type=Path, default=home / "runs/deepseek-v4-flash")
    parser.add_argument("--manifest-csv", default="/repo/weight-manifest.csv")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--memory", default="104g")
    parser.add_argument("--memory-swap", default="112g")
    parser.add_argument("--min-mem-available-gib", type=float, default=16.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--postfill-mode", choices=["deferred", "inline", "off"], default="deferred")
    parser.add_argument("--expert-cache-entries", type=int, default=0)
    parser.add_argument("--wait-postfill-before-cached", action="store_true")
    parser.add_argument("--skip-chat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recover-nvidia-driver-on-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace", default="cuda,nvtx,osrt")
    parser.add_argument(
        "--nsys-install-root",
        type=Path,
        help="Nsight Systems install root to mount. Auto-detected from PATH by default.",
    )
    parser.add_argument("--extra-nsys-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def nsys_install_mount(explicit_root: Path | None) -> tuple[Path, Path]:
    if explicit_root is not None:
        install_root = explicit_root
        return install_root, install_root / "target-linux-sbsa-armv8/nsys"
    nsys = shutil.which("nsys")
    if not nsys:
        raise RuntimeError("nsys was not found on PATH")
    resolved = Path(nsys).resolve()
    parts = resolved.parts
    if "target-linux-sbsa-armv8" not in parts:
        raise RuntimeError(f"unexpected nsys layout: {resolved}")
    install_root = resolved.parents[1]
    container_nsys = install_root / "target-linux-sbsa-armv8/nsys"
    return install_root, container_nsys


def guarded_command(args: argparse.Namespace, install_root: Path, container_nsys: Path) -> list[str]:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    report_base = f"/runs/{args.name}"
    telemetry_path = f"/runs/{args.name}.jsonl"
    probe_args = [
        "/repo/scripts/probe_server_http.py",
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
    ]
    if args.wait_postfill_before_cached:
        probe_args.append("--wait-postfill-before-cached")
    if args.skip_chat:
        probe_args.append("--skip-chat")

    nsys_args = [
        str(container_nsys),
        "profile",
        f"--trace={args.trace}",
        "--sample=none",
        "--cpuctxsw=none",
        "--stats=false",
        "--force-overwrite=true",
        "--output",
        report_base,
        *args.extra_nsys_arg,
        "python3",
        *probe_args,
    ]

    cmd = [
        sys.executable,
        "scripts/guarded_docker_run.py",
        "--name",
        args.name,
        "--image",
        args.image,
        "--log-file",
        str(args.run_dir / f"{args.name}.log"),
        "--memory",
        args.memory,
        "--memory-swap",
        args.memory_swap,
        "--min-mem-available-gib",
        str(args.min_mem_available_gib),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--mount",
        f"{args.model_dir}:/model:ro",
        "--mount",
        f"{args.repo_dir}:/repo:ro",
        "--mount",
        f"{args.run_dir}:/runs",
        "--mount",
        f"{install_root}:{install_root}:ro",
        "--env",
        "PYTHONPATH=/repo:/repo/spark_runtime:/model/inference",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        f"DEEPSEEK_SPARK_MANIFEST_CSV={args.manifest_csv}",
        "--env",
        f"DEEPSEEK_SPARK_TELEMETRY_JSONL={telemetry_path}",
        "--env",
        "DEEPSEEK_SPARK_NVTX=1",
        "--env",
        "DEEPSEEK_SPARK_CUDA_EVENTS=0",
        "--env",
        "DEEPSEEK_SPARK_TELEMETRY_TOKEN_CAP=64",
        "--env",
        f"DEEPSEEK_SPARK_POSTFILL_MODE={args.postfill_mode}",
        "--env",
        f"DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES={args.expert_cache_entries}",
        "--entrypoint",
        str(container_nsys),
    ]
    if args.recover_nvidia_driver_on_exit:
        cmd.append("--recover-nvidia-driver-on-exit")
    cmd.extend(["--", *nsys_args[1:]])
    return cmd


def run_stats(args: argparse.Namespace) -> None:
    report = args.run_dir / f"{args.name}.nsys-rep"
    if not report.exists():
        print(f"nsys report not found, skipping stats: {report}", file=sys.stderr)
        return
    stats_path = args.run_dir / f"{args.name}-stats.txt"
    cmd = [
        "nsys",
        "stats",
        "--force-export=true",
        "--force-overwrite=true",
        "--report",
        "cuda_gpu_kern_sum,nvtx_sum,nvtx_gpu_proj_sum,osrt_sum",
        str(report),
    ]
    print("+", shlex.join(cmd), ">", stats_path)
    with stats_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)


def main() -> int:
    args = parse_args()
    install_root, container_nsys = nsys_install_mount(args.nsys_install_root)
    cmd = guarded_command(args, install_root, container_nsys)
    print("+", shlex.join(cmd))
    if args.dry_run:
        return 0
    result = subprocess.run(cmd, check=False)
    run_stats(args)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
