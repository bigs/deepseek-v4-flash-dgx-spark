#!/usr/bin/env python3
"""Build and benchmark a small fixed-offset packed expert layout."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import time


@dataclass(frozen=True)
class TensorRange:
    source_name: str
    shard: str
    layer: int
    expert: int
    offset: int
    size: int

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True)
class ReadRange:
    file: Path
    offset: int
    size: int
    label: str

    @property
    def end(self) -> int:
        return self.offset + self.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--route-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", default="15,18,39")
    parser.add_argument("--top-per-layer", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--evict", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def parse_layer_expert(name: str) -> tuple[int, int] | None:
    parts = name.split(".")
    if len(parts) < 5 or parts[0] != "layers" or parts[2] != "ffn" or parts[3] != "experts":
        return None
    return int(parts[1]), int(parts[4])


def load_manifest(path: Path) -> dict[tuple[int, int], list[TensorRange]]:
    by_expert: dict[tuple[int, int], list[TensorRange]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] != "lazy_routed_expert":
                continue
            parsed = parse_layer_expert(row["source_name"])
            if parsed is None:
                continue
            layer, expert = parsed
            by_expert[(layer, expert)].append(
                TensorRange(
                    source_name=row["source_name"],
                    shard=row["source_shard"],
                    layer=layer,
                    expert=expert,
                    offset=int(row["source_offset"]),
                    size=int(row["source_bytes"]),
                )
            )
    for ranges in by_expert.values():
        ranges.sort(key=lambda item: item.source_name)
    return by_expert


def top_experts(route_jsonl: Path, layers: list[int], top_per_layer: int) -> dict[int, list[int]]:
    counts: dict[int, Counter[int]] = {layer: Counter() for layer in layers}
    for line in route_jsonl.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") != "expert_route":
            continue
        layer = int(row["layer_id"])
        if layer not in counts:
            continue
        for expert_group in row["expert_indices"]:
            counts[layer].update(int(expert) for expert in expert_group)
    return {
        layer: [expert for expert, _ in counts[layer].most_common(top_per_layer)]
        for layer in layers
    }


def read_all(ranges: list[ReadRange]) -> int:
    checksum = 0
    fd_by_file: dict[Path, int] = {}
    try:
        for read_range in ranges:
            fd = fd_by_file.get(read_range.file)
            if fd is None:
                fd = os.open(read_range.file, os.O_RDONLY)
                fd_by_file[read_range.file] = fd
            data = os.pread(fd, read_range.size, read_range.offset)
            if len(data) != read_range.size:
                raise OSError(f"short read: {read_range.file}:{read_range.offset}")
            checksum = (checksum + data[0] + data[-1] + len(data)) & 0xFFFFFFFF
    finally:
        for fd in fd_by_file.values():
            os.close(fd)
    return checksum


def evict_all(ranges: list[ReadRange]) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    fd_by_file: dict[Path, int] = {}
    try:
        for read_range in ranges:
            fd = fd_by_file.get(read_range.file)
            if fd is None:
                fd = os.open(read_range.file, os.O_RDONLY)
                fd_by_file[read_range.file] = fd
            os.posix_fadvise(fd, read_range.offset, read_range.size, os.POSIX_FADV_DONTNEED)
    finally:
        for fd in fd_by_file.values():
            os.close(fd)


def coalesce(ranges: list[ReadRange]) -> list[ReadRange]:
    ordered = sorted(ranges, key=lambda item: (item.file, item.offset))
    merged: list[ReadRange] = []
    for read_range in ordered:
        if merged and merged[-1].file == read_range.file and merged[-1].end == read_range.offset:
            prev = merged[-1]
            merged[-1] = ReadRange(prev.file, prev.offset, prev.size + read_range.size, prev.label)
        else:
            merged.append(read_range)
    return merged


def build_packed(
    *,
    model_dir: Path,
    out_dir: Path,
    by_expert: dict[tuple[int, int], list[TensorRange]],
    selections: dict[int, list[int]],
) -> dict[int, dict[str, object]]:
    packed_dir = out_dir / "packed-expert-layout"
    packed_dir.mkdir(parents=True, exist_ok=True)
    layout: dict[int, dict[str, object]] = {}
    for layer, experts in selections.items():
        expert_ranges = [by_expert[(layer, expert)] for expert in experts]
        expert_size = sum(item.size for item in expert_ranges[0])
        path = packed_dir / f"layer_{layer:02d}.bin"
        offsets = {}
        with path.open("wb") as out:
            for slot, expert in enumerate(experts):
                offsets[str(expert)] = slot * expert_size
                for tensor_range in by_expert[(layer, expert)]:
                    src = model_dir / tensor_range.shard
                    fd = os.open(src, os.O_RDONLY)
                    try:
                        data = os.pread(fd, tensor_range.size, tensor_range.offset)
                    finally:
                        os.close(fd)
                    if len(data) != tensor_range.size:
                        raise OSError(f"short read while packing {tensor_range.source_name}")
                    out.write(data)
        layout[layer] = {
            "path": str(path),
            "expert_size": expert_size,
            "experts": experts,
            "offsets": offsets,
            "bytes": path.stat().st_size,
        }
    (packed_dir / "layout.json").write_text(json.dumps(layout, indent=2, sort_keys=True))
    return layout


def benchmark(name: str, ranges: list[ReadRange], repeat: int, evict: bool) -> dict[str, object]:
    samples = []
    for iteration in range(repeat):
        if evict:
            evict_all(ranges)
        before = resource.getrusage(resource.RUSAGE_SELF)
        start = time.perf_counter()
        checksum = read_all(ranges)
        elapsed = time.perf_counter() - start
        after = resource.getrusage(resource.RUSAGE_SELF)
        samples.append(
            {
                "iteration": iteration,
                "elapsed_s": elapsed,
                "checksum": checksum,
                "minor_faults": after.ru_minflt - before.ru_minflt,
                "major_faults": after.ru_majflt - before.ru_majflt,
            }
        )
    read_bytes = sum(item.size for item in ranges)
    median_elapsed = sorted(sample["elapsed_s"] for sample in samples)[len(samples) // 2]
    return {
        "name": name,
        "ranges": len(ranges),
        "read_bytes": read_bytes,
        "median_elapsed_s": median_elapsed,
        "median_throughput_gib_s": read_bytes / median_elapsed / 1024**3,
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    layers = parse_int_list(args.layers)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_expert = load_manifest(args.manifest_csv)
    selections = top_experts(args.route_jsonl, layers, args.top_per_layer)
    layout = build_packed(
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        by_expert=by_expert,
        selections=selections,
    )

    results = []
    for layer, experts in selections.items():
        hf_ranges = [
            ReadRange(args.model_dir / tensor.shard, tensor.offset, tensor.size, tensor.source_name)
            for expert in experts
            for tensor in by_expert[(layer, expert)]
        ]
        packed_info = layout[layer]
        packed_ranges = [
            ReadRange(
                Path(str(packed_info["path"])),
                int(packed_info["offsets"][str(expert)]),
                int(packed_info["expert_size"]),
                f"layer{layer}.expert{expert}",
            )
            for expert in experts
        ]
        results.append(benchmark(f"layer{layer}.hf_exact", hf_ranges, args.repeat, args.evict))
        results.append(benchmark(f"layer{layer}.hf_coalesced", coalesce(hf_ranges), args.repeat, args.evict))
        results.append(benchmark(f"layer{layer}.packed", packed_ranges, args.repeat, args.evict))

    output = {
        "layers": layers,
        "top_per_layer": args.top_per_layer,
        "evict": args.evict,
        "selections": selections,
        "layout": layout,
        "results": results,
    }
    json_path = args.out_dir / "packed-expert-layout-benchmark.json"
    md_path = args.out_dir / "packed-expert-layout-benchmark.md"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    with md_path.open("w", encoding="utf-8") as out:
        out.write("# Packed Expert Layout Benchmark\n\n")
        out.write(f"- layers: {layers}\n")
        out.write(f"- top experts per layer: {args.top_per_layer}\n")
        out.write(f"- evict before iterations: {args.evict}\n\n")
        out.write("| Scenario | Ranges | MiB | Median s | Median GiB/s |\n")
        out.write("| --- | ---: | ---: | ---: | ---: |\n")
        for result in results:
            out.write(
                f"| `{result['name']}` | {result['ranges']} | "
                f"{result['read_bytes'] / 1024**2:.1f} | "
                f"{result['median_elapsed_s']:.4f} | "
                f"{result['median_throughput_gib_s']:.2f} |\n"
            )


if __name__ == "__main__":
    main()

