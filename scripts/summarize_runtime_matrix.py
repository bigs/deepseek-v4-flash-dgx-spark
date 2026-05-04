#!/usr/bin/env python3
"""Summarize Spark runtime matrix telemetry JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def load_generate_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "engine_generate":
            events.append(event)
    return events


def row_for_scenario(run_dir: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    telemetry_path = run_dir / scenario.get("telemetry_name", f"matrix-{scenario['name']}.jsonl")
    events = load_generate_events(telemetry_path)
    if not events:
        raise ValueError(f"no engine_generate events in {telemetry_path}")
    first = events[0]
    cached = events[1] if len(events) > 1 else {}
    tokens = int(first["generated_tokens"])
    decode_seconds = float(first["timings"]["decode_seconds"])
    delta = first["expert_cache"]["delta"]
    cached_timings = cached.get("timings", {})
    return {
        "name": scenario["name"],
        "layout": scenario["layout_name"],
        "max_tokens": scenario["max_tokens"],
        "loader": scenario["loader"],
        "cache_entries": scenario["cache_entries"],
        "prefill_seconds": first["timings"]["prefill_seconds"],
        "decode_seconds": decode_seconds,
        "tok_s": tokens / decode_seconds if decode_seconds else None,
        "packed_read_seconds": delta["packed_read_seconds"],
        "materialize_seconds": delta["materialize_seconds"],
        "packed_loads": delta["packed_loads"],
        "packed_misses": delta.get("packed_misses", 0),
        "cache_hits": delta["hits"],
        "cache_misses": delta["misses"],
        "evictions": delta["evictions"],
        "cached_prefill_seconds": cached_timings.get("prefill_seconds"),
        "cached_decode_seconds": cached_timings.get("decode_seconds"),
        "cached_cache_resume_hit": cached.get("cache_resume_hit"),
    }


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "name",
        "prefill",
        "decode",
        "tok/s",
        "read",
        "mat",
        "loads",
        "packed miss",
        "hits",
        "evict",
        "cached prefill",
    ]
    lines = [
        "# Experiment 041: Runtime Combination Matrix Results",
        "",
        "| " + " | ".join(headers) + " |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (item["layout"], item["max_tokens"], item["cache_entries"], item["loader"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    f"{row['prefill_seconds']:.3f}s",
                    f"{row['decode_seconds']:.3f}s",
                    f"{row['tok_s']:.3f}",
                    f"{row['packed_read_seconds']:.3f}s",
                    f"{row['materialize_seconds']:.3f}s",
                    str(row["packed_loads"]),
                    str(row["packed_misses"]),
                    str(row["cache_hits"]),
                    str(row["evictions"]),
                    (
                        f"{row['cached_prefill_seconds']:.3f}s"
                        if row["cached_prefill_seconds"] is not None
                        else ""
                    ),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = [row_for_scenario(args.run_dir, scenario) for scenario in manifest["scenarios"]]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_md(args.out_md, rows)


if __name__ == "__main__":
    main()
