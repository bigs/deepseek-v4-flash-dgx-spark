#!/usr/bin/env python3
"""Run same-container A/B probes for native materializer variants."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


NATIVE_FLAGS = [
    "DEEPSEEK_SPARK_NATIVE_COPY_PLAN",
    "DEEPSEEK_SPARK_NATIVE_FUSED_MATERIALIZER",
    "DEEPSEEK_SPARK_NATIVE_CUDA_MEMCPY",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/repo"))
    parser.add_argument("--run-dir", type=Path, default=Path("/runs"))
    parser.add_argument("--prefix", default="materializer-ab-e050")
    parser.add_argument("--sequence", default="arena,cuda,arena,cuda")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--sleep-between", type=float, default=3.0)
    return parser.parse_args()


def env_for_variant(base: dict[str, str], variant: str, telemetry: Path) -> dict[str, str]:
    env = dict(base)
    env["DEEPSEEK_SPARK_TELEMETRY_JSONL"] = str(telemetry)
    env["DEEPSEEK_SPARK_NATIVE_WITH_CUDA"] = "1"
    for flag in NATIVE_FLAGS:
        env.pop(flag, None)
    if variant == "arena":
        return env
    if variant == "copy_plan":
        env["DEEPSEEK_SPARK_NATIVE_COPY_PLAN"] = "1"
        env["DEEPSEEK_SPARK_NATIVE_FUSED_MATERIALIZER"] = "1"
        return env
    if variant == "cuda":
        env["DEEPSEEK_SPARK_NATIVE_COPY_PLAN"] = "1"
        env["DEEPSEEK_SPARK_NATIVE_FUSED_MATERIALIZER"] = "1"
        env["DEEPSEEK_SPARK_NATIVE_CUDA_MEMCPY"] = "1"
        return env
    raise ValueError(f"unknown variant: {variant}")


def load_first_generate(path: Path) -> dict[str, Any]:
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "engine_generate":
            return event
    raise ValueError(f"no engine_generate event in {path}")


def parse_probe_completion(log_text: str) -> dict[str, Any]:
    for line in log_text.splitlines():
        if line.startswith("completion "):
            return ast.literal_eval(line[len("completion "):])
    raise ValueError("probe output did not contain completion line")


def row_for_run(
    *,
    name: str,
    variant: str,
    telemetry_path: Path,
    driver_log_path: Path,
    returncode: int,
    elapsed_seconds: float,
    first_token_ids: list[int] | None,
) -> dict[str, Any]:
    event = load_first_generate(telemetry_path)
    completion = parse_probe_completion(driver_log_path.read_text())
    token_ids = completion["response"]["spark_metrics"]["generated_token_ids"]
    decode_seconds = float(event["timings"]["decode_seconds"])
    delta = event["expert_cache"]["delta"]
    return {
        "name": name,
        "variant": variant,
        "returncode": returncode,
        "driver_elapsed_seconds": elapsed_seconds,
        "prefill_seconds": event["timings"]["prefill_seconds"],
        "decode_seconds": decode_seconds,
        "tok_s": int(event["generated_tokens"]) / decode_seconds,
        "packed_read_seconds": delta["packed_read_seconds"],
        "materialize_seconds": delta["materialize_seconds"],
        "packed_loads": delta["packed_loads"],
        "packed_misses": delta.get("packed_misses", 0),
        "arena_allocations": delta.get("arena_allocations"),
        "arena_reuses": delta.get("arena_reuses"),
        "generated_tokens": event["generated_tokens"],
        "token_ids": token_ids,
        "token_ids_match_first": first_token_ids is None or token_ids == first_token_ids,
        "telemetry_path": str(telemetry_path),
        "driver_log_path": str(driver_log_path),
    }


def write_summary(prefix: str, run_dir: Path, rows: list[dict[str, Any]]) -> None:
    json_path = run_dir / f"{prefix}-summary.json"
    csv_path = run_dir / f"{prefix}-summary.csv"
    md_path = run_dir / f"{prefix}-summary.md"
    summary = {"rows": rows}
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    csv_fields = [
        "name",
        "variant",
        "returncode",
        "prefill_seconds",
        "decode_seconds",
        "tok_s",
        "packed_read_seconds",
        "materialize_seconds",
        "packed_loads",
        "packed_misses",
        "arena_allocations",
        "arena_reuses",
        "token_ids_match_first",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in csv_fields})

    lines = [
        f"# Materializer A/B: {prefix}",
        "",
        "| name | variant | decode | tok/s | read | mat | arena | token ids |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    row["variant"],
                    f"{row['decode_seconds']:.3f}s",
                    f"{row['tok_s']:.3f}",
                    f"{row['packed_read_seconds']:.3f}s",
                    f"{row['materialize_seconds']:.3f}s",
                    f"{row['arena_allocations']}/{row['arena_reuses']}",
                    "match" if row["token_ids_match_first"] else "DIFF",
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    variants = [item.strip() for item in args.sequence.split(",") if item.strip()]
    rows = []
    first_token_ids: list[int] | None = None
    for index, variant in enumerate(variants, start=1):
        name = f"{args.prefix}-{index:02d}-{variant}"
        telemetry_path = args.run_dir / f"{name}.jsonl"
        driver_log_path = args.run_dir / f"{name}.driver.log"
        telemetry_path.unlink(missing_ok=True)
        driver_log_path.unlink(missing_ok=True)
        env = env_for_variant(os.environ, variant, telemetry_path)
        command = [
            sys.executable,
            str(args.repo / "scripts/probe_server_http.py"),
            "--prompt",
            args.prompt,
            "--max-tokens",
            str(args.max_tokens),
            "--skip-chat",
        ]
        print(json.dumps({"event": "ab_run_start", "name": name, "variant": variant}), flush=True)
        started = time.monotonic()
        completed = subprocess.run(
            command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = time.monotonic() - started
        driver_log_path.write_text(completed.stdout, encoding="utf-8")
        row = row_for_run(
            name=name,
            variant=variant,
            telemetry_path=telemetry_path,
            driver_log_path=driver_log_path,
            returncode=completed.returncode,
            elapsed_seconds=elapsed,
            first_token_ids=first_token_ids,
        )
        if first_token_ids is None:
            first_token_ids = row["token_ids"]
        rows.append(row)
        write_summary(args.prefix, args.run_dir, rows)
        print(
            json.dumps(
                {
                    "event": "ab_run_done",
                    "name": name,
                    "variant": variant,
                    "returncode": completed.returncode,
                    "decode_seconds": row["decode_seconds"],
                    "tok_s": row["tok_s"],
                    "token_ids_match_first": row["token_ids_match_first"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if completed.returncode != 0:
            return completed.returncode
        if args.sleep_between > 0 and index < len(variants):
            time.sleep(args.sleep_between)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
