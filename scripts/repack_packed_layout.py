#!/usr/bin/env python3
"""Repack an existing packed expert layout with a different physical expert order."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-layout", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--route-jsonl", type=Path)
    parser.add_argument("--order", choices=["expert_id", "hot_first"], default="hot_first")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def route_counts(path: Path | None) -> dict[int, Counter[int]]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    if path is None:
        return counts
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "expert_route":
            continue
        layer = int(row["layer_id"])
        for group in row["expert_indices"]:
            counts[layer].update(int(expert) for expert in group)
    return counts


def expert_order(experts: dict[str, object], layer: int, counts: dict[int, Counter[int]], order: str) -> list[int]:
    expert_ids = [int(expert) for expert in experts]
    if order == "expert_id":
        return sorted(expert_ids)
    counter = counts.get(layer, Counter())
    return sorted(expert_ids, key=lambda expert: (-counter.get(expert, 0), expert))


def read_exact(fd: int, size: int, offset: int) -> bytes:
    data = os.pread(fd, size, offset)
    if len(data) != size:
        raise OSError(f"short read size={size} offset={offset}")
    return data


def main() -> None:
    args = parse_args()
    input_layout = json.loads(args.input_layout.read_text())
    input_root = args.input_layout.parent
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_layout_path = args.out_dir / "layout.json"
    if output_layout_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_layout_path} exists; pass --overwrite")

    counts = route_counts(args.route_jsonl)
    started = time.monotonic()
    output_layout = {
        "version": input_layout.get("version", 1),
        "created_at_unix": time.time(),
        "source_layout": str(args.input_layout),
        "route_jsonl": str(args.route_jsonl) if args.route_jsonl else None,
        "order": args.order,
        "layers": {},
    }
    for layer_key, layer_entry in sorted(input_layout["layers"].items(), key=lambda item: int(item[0])):
        layer = int(layer_key)
        input_path = Path(layer_entry["path"])
        if not input_path.is_absolute():
            input_path = input_root / input_path
        output_path = args.out_dir / f"layer_{layer:02d}.bin"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite")
        ordered_experts = expert_order(layer_entry["experts"], layer, counts, args.order)
        output_layer = {
            "path": output_path.name,
            "bytes": 0,
            "experts": {},
        }
        fd = os.open(input_path, os.O_RDONLY)
        try:
            with output_path.open("wb") as out:
                for expert in ordered_experts:
                    input_expert = layer_entry["experts"][str(expert)]
                    block = read_exact(
                        fd,
                        int(input_expert["bytes"]),
                        int(input_expert["offset"]),
                    )
                    output_offset = out.tell()
                    out.write(block)
                    output_layer["experts"][str(expert)] = {
                        **input_expert,
                        "offset": output_offset,
                    }
        finally:
            os.close(fd)
        output_layer["bytes"] = output_path.stat().st_size
        output_layout["layers"][str(layer)] = output_layer
        print(
            json.dumps(
                {
                    "event": "repacked_layer",
                    "layer": layer,
                    "experts": len(ordered_experts),
                    "bytes": output_layer["bytes"],
                    "elapsed_s": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    tmp_path = output_layout_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(output_layout, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(output_layout_path)


if __name__ == "__main__":
    main()
