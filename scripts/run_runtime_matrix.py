#!/usr/bin/env python3
"""Run a guarded Spark runtime matrix for packed expert loader combinations."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    name: str
    layout_name: str
    layout_path: str
    max_tokens: int
    loader: str
    cache_entries: int

    @property
    def telemetry_name(self) -> str:
        return f"matrix-{self.name}.jsonl"

    @property
    def log_name(self) -> str:
        return f"matrix-{self.name}.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/cole/deepseek-v4-flash-dgx-spark"))
    parser.add_argument("--model-dir", type=Path, default=Path("/home/cole/models/deepseek-v4-flash/hf"))
    parser.add_argument("--run-dir", type=Path, default=Path("/home/cole/runs/deepseek-v4-flash"))
    parser.add_argument("--image", default="deepseek-v4-flash-spark:tilelang")
    parser.add_argument("--layouts", default="route,full")
    parser.add_argument("--tokens", default="8,32")
    parser.add_argument("--loaders", default="reuse,native")
    parser.add_argument("--cache-entries", default="1024,2048")
    parser.add_argument("--memory", default="112g")
    parser.add_argument("--memory-swap", default="120g")
    parser.add_argument("--min-mem-available-gib", type=float, default=12.0)
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--only", action="append", default=[], help="Scenario name to run.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def scenarios(args: argparse.Namespace) -> list[Scenario]:
    layout_paths = {
        "route": "/runs/packed-route-e018/layout.json",
        "full": "/runs/packed-full-e018/layout.json",
        "hot": "/runs/packed-full-hot-e045/layout.json",
    }
    result = []
    for layout_name, max_tokens, loader, cache_entries in itertools.product(
        csv_values(args.layouts),
        [int(value) for value in csv_values(args.tokens)],
        csv_values(args.loaders),
        [int(value) for value in csv_values(args.cache_entries)],
    ):
        if layout_name not in layout_paths:
            raise ValueError(f"unknown layout {layout_name!r}")
        if loader not in {"reuse", "native"}:
            raise ValueError(f"unknown loader {loader!r}")
        name = f"{layout_name}-t{max_tokens}-{loader}-c{cache_entries}"
        result.append(
            Scenario(
                name=name,
                layout_name=layout_name,
                layout_path=layout_paths[layout_name],
                max_tokens=max_tokens,
                loader=loader,
                cache_entries=cache_entries,
            )
        )
    if args.only:
        wanted = set(args.only)
        result = [scenario for scenario in result if scenario.name in wanted]
        missing = wanted - {scenario.name for scenario in result}
        if missing:
            raise ValueError(f"unknown --only scenario(s): {sorted(missing)}")
    return result


def env_for_scenario(scenario: Scenario) -> list[str]:
    env = [
        "PYTHONPATH=/repo:/repo/spark_runtime:/model/inference",
        "PYTHONDONTWRITEBYTECODE=1",
        "DEEPSEEK_SPARK_MODEL_DIR=/model",
        "DEEPSEEK_SPARK_MANIFEST_CSV=/repo/results/spark-66c9/weight-manifest.csv",
        f"DEEPSEEK_SPARK_TELEMETRY_JSONL=/runs/{scenario.telemetry_name}",
        "DEEPSEEK_SPARK_POSTFILL_MODE=deferred",
        f"DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES={scenario.cache_entries}",
        "DEEPSEEK_SPARK_EXPERT_CACHE_POLICY=layer_lru",
        "DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_COUNT=43",
        f"DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT={scenario.layout_path}",
        "DEEPSEEK_SPARK_NATIVE_BUILD_DIR=/runs/torch-extensions/native-packed-loader",
    ]
    if scenario.loader == "reuse":
        env.append("DEEPSEEK_SPARK_PACKED_REUSE_STAGING=1")
    elif scenario.loader == "native":
        env.extend(
            [
                "DEEPSEEK_SPARK_PACKED_NATIVE_LOADER=1",
                "DEEPSEEK_SPARK_PACKED_NATIVE_PINNED_STAGING=1",
                "DEEPSEEK_SPARK_DIRECT_PARAM_COPY=1",
                "DEEPSEEK_SPARK_PARAM_COPY_NON_BLOCKING=1",
            ]
        )
    return env


def command_for_scenario(args: argparse.Namespace, scenario: Scenario) -> list[str]:
    cmd = [
        "python3",
        str(args.repo / "scripts/guarded_docker_run.py"),
        "--name",
        f"matrix-{scenario.name}",
        "--image",
        args.image,
        "--entrypoint",
        "/bin/bash",
        "--log-file",
        str(args.run_dir / scenario.log_name),
        "--memory",
        args.memory,
        "--memory-swap",
        args.memory_swap,
        "--min-mem-available-gib",
        str(args.min_mem_available_gib),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--mount",
        f"{args.repo}:/repo:ro",
        "--mount",
        f"{args.model_dir}:/model:ro",
        "--mount",
        f"{args.run_dir}:/runs",
    ]
    for item in env_for_scenario(scenario):
        cmd.extend(["--env", item])
    cmd.extend(
        [
            "--recover-nvidia-driver-on-exit",
            "--",
            "-lc",
            (
                "cd /repo && python3 /repo/scripts/probe_server_http.py "
                f"--prompt {json.dumps(args.prompt)} "
                f"--max-tokens {scenario.max_tokens} "
                "--skip-chat"
            ),
        ]
    )
    return cmd


def main() -> int:
    args = parse_args()
    matrix = scenarios(args)
    manifest = {
        "scenarios": [scenario.__dict__ for scenario in matrix],
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_dir / "matrix-e041-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    for scenario in matrix:
        telemetry_path = args.run_dir / scenario.telemetry_name
        if args.skip_existing and telemetry_path.exists() and telemetry_path.stat().st_size > 0:
            print(f"skip existing {scenario.name}", flush=True)
            continue
        cmd = command_for_scenario(args, scenario)
        print("run", scenario.name, flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            print(f"scenario {scenario.name} failed rc={completed.returncode}", flush=True)
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
