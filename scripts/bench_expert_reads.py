#!/usr/bin/env python3
"""Benchmark DeepSeek-V4-Flash expert payload reads from safetensor offsets."""

from __future__ import annotations

import argparse
import csv
import json
import mmap
import os
import resource
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TensorRange:
    name: str
    file: str
    layer: int
    expert: int
    matrix: str
    kind: str
    start: int
    end: int
    bytes: int


@dataclass(frozen=True)
class ReadRange:
    file: str
    start: int
    end: int
    label: str

    @property
    def bytes(self) -> int:
        return self.end - self.start


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{n} B"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            out.extend(range(int(left), int(right) + 1))
        else:
            out.append(int(part))
    return out


def load_tensor_csv(path: Path) -> list[TensorRange]:
    tensors: list[TensorRange] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["component"] != "core_routed_experts":
                continue
            tensors.append(
                TensorRange(
                    name=row["name"],
                    file=row["file"],
                    layer=int(row["layer"]),
                    expert=int(row["expert"]),
                    matrix=row["matrix"],
                    kind=row["kind"],
                    start=int(row["file_data_start"]),
                    end=int(row["file_data_end"]),
                    bytes=int(row["bytes"]),
                )
            )
    return tensors


def file_order_experts(tensors: list[TensorRange], layer: int) -> list[int]:
    seen = set()
    experts = []
    for tensor in sorted((t for t in tensors if t.layer == layer), key=lambda t: (t.file, t.start)):
        if tensor.expert not in seen:
            seen.add(tensor.expert)
            experts.append(tensor.expert)
    return experts


def scenario_experts(tensors: list[TensorRange], layer: int) -> dict[str, list[int]]:
    by_file = file_order_experts(tensors, layer)
    return {
        "numeric_0_5": [0, 1, 2, 3, 4, 5],
        "numeric_mid_6": [125, 126, 127, 128, 129, 130],
        "spread_6": [0, 51, 102, 153, 204, 255],
        "file_adjacent_first_6": by_file[:6],
        "file_adjacent_mid_6": by_file[125:131],
    }


def ranges_for_experts(tensors: list[TensorRange], layer: int, experts: set[int]) -> list[ReadRange]:
    selected = [
        t
        for t in tensors
        if t.layer == layer and t.expert in experts and t.kind in {"weight", "scale"}
    ]
    selected.sort(key=lambda t: (t.file, t.start))
    return [ReadRange(t.file, t.start, t.end, t.name) for t in selected]


def ranges_for_layer(tensors: list[TensorRange], layer: int) -> list[ReadRange]:
    selected = [t for t in tensors if t.layer == layer]
    selected.sort(key=lambda t: (t.file, t.start))
    return [ReadRange(t.file, t.start, t.end, t.name) for t in selected]


def coalesce_ranges(ranges: Iterable[ReadRange], max_gap: int) -> list[ReadRange]:
    ordered = sorted(ranges, key=lambda r: (r.file, r.start))
    if not ordered:
        return []
    merged: list[ReadRange] = []
    cur = ordered[0]
    labels = [cur.label]
    for nxt in ordered[1:]:
        if nxt.file == cur.file and nxt.start <= cur.end + max_gap:
            cur = ReadRange(cur.file, cur.start, max(cur.end, nxt.end), f"{labels[0]}..{nxt.label}")
            labels.append(nxt.label)
        else:
            merged.append(cur)
            cur = nxt
            labels = [cur.label]
    merged.append(cur)
    return merged


def read_pread(model_dir: Path, ranges: list[ReadRange], chunk_size: int) -> int:
    checksum = 0
    fd_by_file: dict[str, int] = {}
    try:
        for read_range in ranges:
            fd = fd_by_file.get(read_range.file)
            if fd is None:
                fd = os.open(model_dir / read_range.file, os.O_RDONLY)
                fd_by_file[read_range.file] = fd
            remaining = read_range.bytes
            offset = read_range.start
            while remaining:
                n = min(chunk_size, remaining)
                data = os.pread(fd, n, offset)
                if not data:
                    raise OSError(f"short read at {read_range.file}:{offset}")
                checksum = (checksum + data[0] + data[-1] + len(data)) & 0xFFFFFFFF
                got = len(data)
                remaining -= got
                offset += got
    finally:
        for fd in fd_by_file.values():
            os.close(fd)
    return checksum


def read_mmap(model_dir: Path, ranges: list[ReadRange], chunk_size: int) -> int:
    checksum = 0
    mmap_by_file: dict[str, mmap.mmap] = {}
    fd_by_file: dict[str, int] = {}
    try:
        for read_range in ranges:
            mm = mmap_by_file.get(read_range.file)
            if mm is None:
                fd = os.open(model_dir / read_range.file, os.O_RDONLY)
                fd_by_file[read_range.file] = fd
                mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                mmap_by_file[read_range.file] = mm
            remaining = read_range.bytes
            offset = read_range.start
            while remaining:
                n = min(chunk_size, remaining)
                data = mm[offset : offset + n]
                if not data:
                    raise OSError(f"short mmap read at {read_range.file}:{offset}")
                checksum = (checksum + data[0] + data[-1] + len(data)) & 0xFFFFFFFF
                got = len(data)
                remaining -= got
                offset += got
    finally:
        for mm in mmap_by_file.values():
            mm.close()
        for fd in fd_by_file.values():
            os.close(fd)
    return checksum


def evict_ranges(model_dir: Path, ranges: list[ReadRange]) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    fd_by_file: dict[str, int] = {}
    try:
        for read_range in ranges:
            fd = fd_by_file.get(read_range.file)
            if fd is None:
                fd = os.open(model_dir / read_range.file, os.O_RDONLY)
                fd_by_file[read_range.file] = fd
            os.posix_fadvise(fd, read_range.start, read_range.bytes, os.POSIX_FADV_DONTNEED)
    finally:
        for fd in fd_by_file.values():
            os.close(fd)


def run_one(
    model_dir: Path,
    scenario: dict[str, object],
    method: str,
    chunk_size: int,
    repeat: int,
    evict_mode: str,
) -> list[dict[str, object]]:
    ranges = scenario["ranges"]
    assert isinstance(ranges, list)
    results = []
    if evict_mode == "scenario":
        evict_ranges(model_dir, ranges)
    for iteration in range(repeat):
        if evict_mode == "iteration":
            evict_ranges(model_dir, ranges)
        before = resource.getrusage(resource.RUSAGE_SELF)
        start = time.perf_counter()
        if method == "pread":
            checksum = read_pread(model_dir, ranges, chunk_size)
        elif method == "mmap":
            checksum = read_mmap(model_dir, ranges, chunk_size)
        else:
            raise ValueError(method)
        elapsed = time.perf_counter() - start
        after = resource.getrusage(resource.RUSAGE_SELF)
        read_bytes = sum(r.bytes for r in ranges)
        results.append(
            {
                "method": method,
                "iteration": iteration,
                "elapsed_s": elapsed,
                "read_bytes": read_bytes,
                "read_human": human_bytes(read_bytes),
                "throughput_gib_s": read_bytes / elapsed / (1024**3) if elapsed else None,
                "checksum": checksum,
                "minor_faults": after.ru_minflt - before.ru_minflt,
                "major_faults": after.ru_majflt - before.ru_majflt,
                "evict_mode": evict_mode,
            }
        )
    return results


def build_scenarios(tensors: list[TensorRange], layers: list[int], max_gap_values: list[int]) -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    for layer in layers:
        for name, experts in scenario_experts(tensors, layer).items():
            exact = ranges_for_experts(tensors, layer, set(experts))
            useful_bytes = sum(r.bytes for r in exact)
            for max_gap in max_gap_values:
                ranges = coalesce_ranges(exact, max_gap)
                read_bytes = sum(r.bytes for r in ranges)
                scenarios.append(
                    {
                        "name": f"layer{layer}_{name}_gap{max_gap}",
                        "kind": "topk_experts",
                        "layer": layer,
                        "experts": experts,
                        "max_gap": max_gap,
                        "tensor_ranges": len(exact),
                        "read_ranges": len(ranges),
                        "useful_bytes": useful_bytes,
                        "useful_human": human_bytes(useful_bytes),
                        "read_bytes_planned": read_bytes,
                        "read_human_planned": human_bytes(read_bytes),
                        "read_amplification": read_bytes / useful_bytes if useful_bytes else None,
                        "ranges": ranges,
                    }
                )

        full = ranges_for_layer(tensors, layer)
        ranges = coalesce_ranges(full, 0)
        read_bytes = sum(r.bytes for r in ranges)
        scenarios.append(
            {
                "name": f"layer{layer}_whole_layer_gap0",
                "kind": "whole_layer",
                "layer": layer,
                "experts": "all",
                "max_gap": 0,
                "tensor_ranges": len(full),
                "read_ranges": len(ranges),
                "useful_bytes": read_bytes,
                "useful_human": human_bytes(read_bytes),
                "read_bytes_planned": read_bytes,
                "read_human_planned": human_bytes(read_bytes),
                "read_amplification": 1.0,
                "ranges": ranges,
            }
        )
    return scenarios


def summarize_results(results: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in results:
        key = f"{row['name']}::{row['method']}"
        grouped.setdefault(key, []).append(row)
    summary: dict[str, dict[str, object]] = {}
    for key, rows in grouped.items():
        throughputs = [float(row["throughput_gib_s"]) for row in rows if row["throughput_gib_s"] is not None]
        elapsed = [float(row["elapsed_s"]) for row in rows]
        summary[key] = {
            "scenario": rows[0]["name"],
            "method": rows[0]["method"],
            "iterations": len(rows),
            "read_bytes": rows[0]["read_bytes"],
            "read_human": rows[0]["read_human"],
            "median_elapsed_s": statistics.median(elapsed),
            "median_throughput_gib_s": statistics.median(throughputs) if throughputs else None,
            "min_throughput_gib_s": min(throughputs) if throughputs else None,
            "max_throughput_gib_s": max(throughputs) if throughputs else None,
            "major_faults_total": sum(int(row["major_faults"]) for row in rows),
            "minor_faults_total": sum(int(row["minor_faults"]) for row in rows),
        }
    return summary


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    scenarios = payload["scenarios"]
    results_summary = payload["results_summary"]
    assert isinstance(scenarios, list)
    assert isinstance(results_summary, dict)

    lines = [
        "# Expert Read Benchmark",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## Setup",
        "",
        f"- Model dir: `{payload['model_dir']}`",
        f"- Tensor CSV: `{payload['tensor_csv']}`",
        f"- Layers: `{payload['layers']}`",
        f"- Methods: `{payload['methods']}`",
        f"- Chunk size: `{payload['chunk_size_human']}`",
        f"- Repeat: `{payload['repeat']}`",
        f"- Evict mode: `{payload.get('evict_mode', 'never')}`",
        "",
        "## Planned Scenarios",
        "",
        "| scenario | kind | experts | tensor ranges | read ranges | useful | planned read | amplification |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in scenarios:
        experts = scenario["experts"]
        lines.append(
            "| {name} | {kind} | {experts} | {tensor_ranges} | {read_ranges} | {useful_human} | {read_human_planned} | {amp:.2f} |".format(
                name=scenario["name"],
                kind=scenario["kind"],
                experts=experts if isinstance(experts, str) else ",".join(str(e) for e in experts),
                tensor_ranges=scenario["tensor_ranges"],
                read_ranges=scenario["read_ranges"],
                useful_human=scenario["useful_human"],
                read_human_planned=scenario["read_human_planned"],
                amp=float(scenario["read_amplification"]),
            )
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| scenario | method | read | median seconds | median GiB/s | major faults | minor faults |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in sorted(results_summary):
        row = results_summary[key]
        throughput = row["median_throughput_gib_s"]
        lines.append(
            "| {scenario} | {method} | {read_human} | {elapsed:.4f} | {throughput:.3f} | {major} | {minor} |".format(
                scenario=row["scenario"],
                method=row["method"],
                read_human=row["read_human"],
                elapsed=float(row["median_elapsed_s"]),
                throughput=float(throughput) if throughput is not None else 0.0,
                major=row["major_faults_total"],
                minor=row["minor_faults_total"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--tensor-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--layers", default="0,18,42", help="Comma/range layer list, e.g. 0,18,42")
    parser.add_argument("--methods", default="pread,mmap", help="Comma-separated methods: pread,mmap")
    parser.add_argument("--chunk-size-mib", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-gaps", default="0,1048576", help="Comma-separated coalescing gaps in bytes")
    parser.add_argument(
        "--evict-mode",
        choices=["never", "scenario", "iteration"],
        default="never",
        help="Use posix_fadvise DONTNEED before each scenario or iteration when available.",
    )
    parser.add_argument("--output-prefix", default="expert-read-benchmark")
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    tensor_csv = args.tensor_csv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    layers = parse_int_list(args.layers)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    max_gaps = parse_int_list(args.max_gaps)
    chunk_size = args.chunk_size_mib * 1024 * 1024

    tensors = load_tensor_csv(tensor_csv)
    scenarios = build_scenarios(tensors, layers, max_gaps)
    results: list[dict[str, object]] = []
    for scenario in scenarios:
        ranges = scenario["ranges"]
        assert isinstance(ranges, list)
        if not ranges:
            continue
        for method in methods:
            for row in run_one(model_dir, scenario, method, chunk_size, args.repeat, args.evict_mode):
                enriched = {k: v for k, v in scenario.items() if k != "ranges"}
                enriched.update(row)
                results.append(enriched)
                print(
                    f"{enriched['name']} {method} iter={row['iteration']} "
                    f"{row['read_human']} {row['elapsed_s']:.4f}s "
                    f"{row['throughput_gib_s']:.3f} GiB/s "
                    f"majflt={row['major_faults']}",
                    flush=True,
                )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "tensor_csv": str(tensor_csv),
        "layers": layers,
        "methods": methods,
        "chunk_size": chunk_size,
        "chunk_size_human": human_bytes(chunk_size),
        "repeat": args.repeat,
        "evict_mode": args.evict_mode,
        "max_gaps": max_gaps,
        "scenarios": [
            {k: v for k, v in scenario.items() if k != "ranges"}
            for scenario in scenarios
        ],
        "results": results,
        "results_summary": summarize_results(results),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.output_prefix}.json"
    md_path = out_dir / f"{args.output_prefix}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, payload)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
