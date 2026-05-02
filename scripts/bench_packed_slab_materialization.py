#!/usr/bin/env python3
"""Benchmark packed expert block materialization strategies.

This compares the current Python bytes/bytearray path, a reusable bytearray slab filled by
preadv, and an optional native tensor reader. It measures the substrate underneath tensor
reconstruction: read one packed expert block, wrap or fill a uint8 CPU tensor, transfer it
to CUDA, and synchronize.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time


@dataclass(frozen=True)
class ExpertBlock:
    layer: int
    expert: int
    path: Path
    offset: int
    size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-json", type=Path, required=True)
    parser.add_argument("--route-jsonl", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--max-blocks", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--include-native",
        action="store_true",
        help="Include the native C++ extension that reads directly into CPU uint8 tensors.",
    )
    return parser.parse_args()


def load_layout(path: Path) -> dict[tuple[int, int], ExpertBlock]:
    layout = json.loads(path.read_text())
    root = path.parent
    blocks: dict[tuple[int, int], ExpertBlock] = {}
    for layer_key, layer_entry in layout["layers"].items():
        layer = int(layer_key)
        layer_path = root / str(layer_entry["path"])
        for expert_key, expert_entry in layer_entry["experts"].items():
            expert = int(expert_key)
            blocks[(layer, expert)] = ExpertBlock(
                layer=layer,
                expert=expert,
                path=layer_path,
                offset=int(expert_entry["offset"]),
                size=int(expert_entry["bytes"]),
            )
    return blocks


def select_blocks(layout: dict[tuple[int, int], ExpertBlock], route_jsonl: Path, max_blocks: int) -> list[ExpertBlock]:
    selected: list[ExpertBlock] = []
    seen: set[tuple[int, int]] = set()
    for line in route_jsonl.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") != "expert_route":
            continue
        layer = int(row["layer_id"])
        for group in row["expert_indices"]:
            for value in group:
                key = (layer, int(value))
                if key in seen or key not in layout:
                    continue
                seen.add(key)
                selected.append(layout[key])
                if len(selected) >= max_blocks:
                    return selected
    return selected


def open_fds(blocks: list[ExpertBlock]) -> dict[Path, int]:
    return {path: os.open(path, os.O_RDONLY) for path in sorted({block.path for block in blocks})}


def close_fds(fds: dict[Path, int]) -> None:
    for fd in fds.values():
        os.close(fd)


def bench_current_bytes(blocks: list[ExpertBlock], fds: dict[Path, int], device: str) -> int:
    import torch

    checksum = 0
    for block in blocks:
        data = os.pread(fds[block.path], block.size, block.offset)
        if len(data) != block.size:
            raise OSError(f"short read: {block}")
        cpu = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        gpu = cpu.to(device)
        torch.cuda.synchronize()
        checksum = (checksum + int(gpu[0].item()) + int(gpu[-1].item()) + gpu.numel()) & 0xFFFFFFFF
    return checksum


def bench_preadv_bytearray(blocks: list[ExpertBlock], fds: dict[Path, int], device: str) -> int:
    import torch

    if not hasattr(os, "preadv"):
        raise RuntimeError("os.preadv is not available on this platform")
    max_size = max(block.size for block in blocks)
    slab = bytearray(max_size)
    checksum = 0
    for block in blocks:
        view = memoryview(slab)[: block.size]
        read = os.preadv(fds[block.path], [view], block.offset)
        if read != block.size:
            raise OSError(f"short read: {block}")
        cpu = torch.frombuffer(view, dtype=torch.uint8)
        gpu = cpu.to(device)
        torch.cuda.synchronize()
        checksum = (checksum + int(gpu[0].item()) + int(gpu[-1].item()) + gpu.numel()) & 0xFFFFFFFF
    return checksum


def bench_native_tensor(
    blocks: list[ExpertBlock],
    fds: dict[Path, int],
    device: str,
    *,
    pinned: bool = False,
) -> int:
    import torch

    from spark_runtime.native_packed_loader import load_native_packed_loader

    native = load_native_packed_loader()
    max_size = max(block.size for block in blocks)
    staging = torch.empty(max_size, dtype=torch.uint8, device="cpu", pin_memory=pinned)
    checksum = 0
    for block in blocks:
        cpu = staging[: block.size]
        read = native.read_into(fds[block.path], cpu, block.size, block.offset)
        if read != block.size:
            raise OSError(f"short read: {block}")
        gpu = cpu.to(device)
        torch.cuda.synchronize()
        checksum = (checksum + int(gpu[0].item()) + int(gpu[-1].item()) + gpu.numel()) & 0xFFFFFFFF
    return checksum


def bench_native_tensor_pageable(blocks: list[ExpertBlock], fds: dict[Path, int], device: str) -> int:
    return bench_native_tensor(blocks, fds, device, pinned=False)


def bench_native_tensor_pinned(blocks: list[ExpertBlock], fds: dict[Path, int], device: str) -> int:
    return bench_native_tensor(blocks, fds, device, pinned=True)


def main() -> None:
    args = parse_args()
    import torch

    layout = load_layout(args.layout_json)
    blocks = select_blocks(layout, args.route_jsonl, args.max_blocks)
    if not blocks:
        raise ValueError("no blocks selected")
    methods = [
        ("current_bytes", bench_current_bytes),
        ("preadv_bytearray_slab", bench_preadv_bytearray),
    ]
    if args.include_native:
        from spark_runtime.native_packed_loader import load_native_packed_loader

        load_native_packed_loader()
        methods.append(("native_tensor", bench_native_tensor_pageable))
        methods.append(("native_pinned_tensor", bench_native_tensor_pinned))
    results = []
    for method, fn in methods:
        for iteration in range(args.repeat):
            fds = open_fds(blocks)
            try:
                torch.cuda.synchronize()
                start = time.perf_counter()
                checksum = fn(blocks, fds, args.device)
                elapsed = time.perf_counter() - start
                torch.cuda.synchronize()
            finally:
                close_fds(fds)
            row = {
                "method": method,
                "iteration": iteration,
                "elapsed_s": elapsed,
                "blocks": len(blocks),
                "bytes": sum(block.size for block in blocks),
                "throughput_gib_s": sum(block.size for block in blocks) / elapsed / 1024**3,
                "checksum": checksum,
            }
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    summary = {}
    for method, _ in methods:
        samples = [row for row in results if row["method"] == method]
        median = sorted(samples, key=lambda row: row["elapsed_s"])[len(samples) // 2]
        summary[method] = median
    output = {
        "layout_json": str(args.layout_json),
        "route_jsonl": str(args.route_jsonl),
        "max_blocks": args.max_blocks,
        "repeat": args.repeat,
        "device": args.device,
        "block_count": len(blocks),
        "block_bytes": blocks[0].size,
        "summary": summary,
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
