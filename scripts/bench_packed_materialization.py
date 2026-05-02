#!/usr/bin/env python3
"""Benchmark packed-block versus safetensor expert miss materialization."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time


@dataclass(frozen=True)
class TensorRow:
    source_name: str
    shard: str
    offset: int
    size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--experts", required=True, help="Comma-separated expert ids")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--method",
        choices=["both", "safetensor_tensors_to_device", "packed_block_to_device"],
        default="both",
    )
    return parser.parse_args()


def parse_experts(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def load_rows(path: Path, layer: int, experts: list[int]) -> dict[int, list[TensorRow]]:
    wanted_prefixes = {
        expert: f"layers.{layer}.ffn.experts.{expert}."
        for expert in experts
    }
    rows = {expert: [] for expert in experts}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] != "lazy_routed_expert":
                continue
            for expert, prefix in wanted_prefixes.items():
                if row["source_name"].startswith(prefix):
                    rows[expert].append(
                        TensorRow(
                            source_name=row["source_name"],
                            shard=row["source_shard"],
                            offset=int(row["source_offset"]),
                            size=int(row["source_bytes"]),
                        )
                    )
    for expert in experts:
        rows[expert].sort(key=lambda item: item.source_name)
        if not rows[expert]:
            raise ValueError(f"no rows for expert {expert}")
    return rows


def pack_bytes(model_dir: Path, rows: list[TensorRow]) -> bytes:
    parts = []
    for row in rows:
        fd = os.open(model_dir / row.shard, os.O_RDONLY)
        try:
            data = os.pread(fd, row.size, row.offset)
        finally:
            os.close(fd)
        if len(data) != row.size:
            raise OSError(f"short read: {row.source_name}")
        parts.append(data)
    return b"".join(parts)


def build_packed_file(model_dir: Path, out_path: Path, experts: list[int], rows_by_expert: dict[int, list[TensorRow]]) -> dict[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    offsets = {}
    with out_path.open("wb") as handle:
        for expert in experts:
            offsets[expert] = handle.tell()
            handle.write(pack_bytes(model_dir, rows_by_expert[expert]))
    return offsets


def main() -> None:
    args = parse_args()
    import torch
    from safetensors.torch import safe_open

    experts = parse_experts(args.experts)
    rows_by_expert = load_rows(args.manifest_csv, args.layer, experts)
    handles = {}
    packed_path = args.out_json.with_suffix(".packed.bin")
    packed_offsets = build_packed_file(args.model_dir, packed_path, experts, rows_by_expert)
    expert_size = sum(row.size for row in rows_by_expert[experts[0]])

    def safetensor_once(expert: int) -> int:
        total = 0
        gpu_tensors = []
        for row in rows_by_expert[expert]:
            handle = handles.get(row.shard)
            if handle is None:
                handle = safe_open(args.model_dir / row.shard, framework="pt", device="cpu")
                handles[row.shard] = handle
            tensor = handle.get_tensor(row.source_name)
            gpu_tensors.append(tensor.to(args.device))
            total += tensor.numel() * tensor.element_size()
        if args.device == "cuda":
            torch.cuda.synchronize()
        return total + len(gpu_tensors)

    def packed_once(expert: int) -> int:
        fd = os.open(packed_path, os.O_RDONLY)
        try:
            data = os.pread(fd, expert_size, packed_offsets[expert])
        finally:
            os.close(fd)
        if len(data) != expert_size:
            raise OSError(f"short packed read for expert {expert}")
        cpu = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        gpu = cpu.to(args.device)
        if args.device == "cuda":
            torch.cuda.synchronize()
        return int(gpu.numel())

    results = []
    methods = [("safetensor_tensors_to_device", safetensor_once), ("packed_block_to_device", packed_once)]
    if args.method != "both":
        methods = [(name, fn) for name, fn in methods if name == args.method]
    for method, fn in methods:
        for iteration in range(args.repeat):
            start = time.perf_counter()
            checksum = 0
            for expert in experts:
                checksum += fn(expert)
            elapsed = time.perf_counter() - start
            results.append(
                {
                    "method": method,
                    "iteration": iteration,
                    "elapsed_s": elapsed,
                    "experts": len(experts),
                    "bytes": sum(sum(row.size for row in rows_by_expert[expert]) for expert in experts),
                    "checksum": checksum,
                }
            )

    summary = {}
    for method in {row["method"] for row in results}:
        samples = [row["elapsed_s"] for row in results if row["method"] == method]
        median = sorted(samples)[len(samples) // 2]
        summary[method] = {
            "median_elapsed_s": median,
            "median_mib_s": results[0]["bytes"] / median / 1024**2,
        }

    output = {
        "layer": args.layer,
        "experts": experts,
        "repeat": args.repeat,
        "device": args.device,
        "packed_path": str(packed_path),
        "method": args.method,
        "expert_bytes": {str(expert): sum(row.size for row in rows_by_expert[expert]) for expert in experts},
        "summary": summary,
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
