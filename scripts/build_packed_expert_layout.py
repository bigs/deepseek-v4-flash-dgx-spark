#!/usr/bin/env python3
"""Build a fixed-offset packed routed-expert layout from a checkpoint manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time


@dataclass(frozen=True)
class TensorRow:
    source_name: str
    target_name: str
    transform: str
    source_dtype: str
    source_shape: list[int]
    source_bytes: int
    source_shard: str
    source_offset: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated layer ids, or 'all' for all routed expert layers.",
    )
    parser.add_argument(
        "--experts",
        default="all",
        help="Comma-separated expert ids, or 'all' for all experts in each selected layer.",
    )
    parser.add_argument(
        "--route-jsonl",
        type=Path,
        help="Optional route trace; when set, only experts observed in the trace are packed.",
    )
    parser.add_argument(
        "--top-per-layer",
        type=int,
        help="With --route-jsonl, pack only the most frequent N experts per layer.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_shape(value: str) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.split("x") if part]


def parse_int_filter(value: str) -> set[int] | None:
    if value.strip().lower() == "all":
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def parse_layer_expert(name: str) -> tuple[int, int] | None:
    parts = name.split(".")
    if len(parts) < 6 or parts[0] != "layers" or parts[2] != "ffn" or parts[3] != "experts":
        return None
    return int(parts[1]), int(parts[4])


def tensor_sort_key(row: TensorRow) -> tuple[int, str]:
    component = row.target_name.rsplit(".", 2)[-2:]
    rank = {
        ("w1", "scale"): 0,
        ("w2", "scale"): 1,
        ("w3", "scale"): 2,
        ("w1", "weight"): 3,
        ("w2", "weight"): 4,
        ("w3", "weight"): 5,
    }.get(tuple(component), 99)
    return rank, row.target_name


def load_manifest(
    path: Path,
    layers: set[int] | None,
    experts: set[int] | None,
    route_selection: dict[int, set[int]] | None,
) -> dict[tuple[int, int], list[TensorRow]]:
    by_expert: dict[tuple[int, int], list[TensorRow]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] != "lazy_routed_expert":
                continue
            parsed = parse_layer_expert(row["source_name"])
            if parsed is None:
                continue
            layer, expert = parsed
            if layers is not None and layer not in layers:
                continue
            if experts is not None and expert not in experts:
                continue
            if route_selection is not None and expert not in route_selection.get(layer, set()):
                continue
            by_expert[(layer, expert)].append(
                TensorRow(
                    source_name=row["source_name"],
                    target_name=row["target_name"],
                    transform=row["transform"],
                    source_dtype=row["source_dtype"],
                    source_shape=parse_shape(row["source_shape"]),
                    source_bytes=int(row["source_bytes"]),
                    source_shard=row["source_shard"],
                    source_offset=int(row["source_offset"]),
                )
            )
    for rows in by_expert.values():
        rows.sort(key=tensor_sort_key)
    return dict(by_expert)


def load_route_selection(path: Path, top_per_layer: int | None) -> dict[int, set[int]]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") != "expert_route":
            continue
        layer = int(row["layer_id"])
        for expert_group in row["expert_indices"]:
            counts[layer].update(int(expert) for expert in expert_group)
    selection: dict[int, set[int]] = {}
    for layer, counter in counts.items():
        if top_per_layer is None:
            selection[layer] = set(counter)
        else:
            selection[layer] = {expert for expert, _ in counter.most_common(top_per_layer)}
    return selection


def copy_range(model_dir: Path, fd_cache: dict[Path, int], row: TensorRow) -> bytes:
    path = model_dir / row.source_shard
    fd = fd_cache.get(path)
    if fd is None:
        fd = os.open(path, os.O_RDONLY)
        fd_cache[path] = fd
    data = os.pread(fd, row.source_bytes, row.source_offset)
    if len(data) != row.source_bytes:
        raise OSError(f"short read while packing {row.source_name}")
    return data


def close_fds(fd_cache: dict[Path, int]) -> None:
    for fd in fd_cache.values():
        os.close(fd)
    fd_cache.clear()


def build_layout(
    *,
    model_dir: Path,
    out_dir: Path,
    manifest_csv: Path,
    by_expert: dict[tuple[int, int], list[TensorRow]],
    overwrite: bool,
) -> dict[str, object]:
    layers = sorted({layer for layer, _ in by_expert})
    out_dir.mkdir(parents=True, exist_ok=True)
    layout_path = out_dir / "layout.json"
    if layout_path.exists() and not overwrite:
        raise FileExistsError(f"{layout_path} already exists; pass --overwrite to replace it")

    fd_cache: dict[Path, int] = {}
    started = time.monotonic()
    layout: dict[str, object] = {
        "version": 1,
        "created_at_unix": time.time(),
        "model_dir": str(model_dir),
        "manifest_csv": str(manifest_csv),
        "layers": {},
    }
    try:
        for layer in layers:
            experts = sorted(expert for item_layer, expert in by_expert if item_layer == layer)
            layer_path = out_dir / f"layer_{layer:02d}.bin"
            if layer_path.exists() and not overwrite:
                raise FileExistsError(f"{layer_path} already exists; pass --overwrite to replace it")
            layer_entry: dict[str, object] = {
                "path": layer_path.name,
                "bytes": 0,
                "experts": {},
            }
            with layer_path.open("wb") as out:
                for expert in experts:
                    expert_offset = out.tell()
                    tensor_entries: dict[str, object] = {}
                    for row in by_expert[(layer, expert)]:
                        tensor_offset = out.tell() - expert_offset
                        out.write(copy_range(model_dir, fd_cache, row))
                        tensor_entries[row.target_name] = {
                            "offset": tensor_offset,
                            "bytes": row.source_bytes,
                            "source_dtype": row.source_dtype,
                            "source_shape": row.source_shape,
                            "transform": row.transform,
                        }
                    layer_entry["experts"][str(expert)] = {
                        "offset": expert_offset,
                        "bytes": out.tell() - expert_offset,
                        "tensors": tensor_entries,
                    }
            layer_entry["bytes"] = layer_path.stat().st_size
            layout["layers"][str(layer)] = layer_entry
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "event": "packed_layer_done",
                        "layer": layer,
                        "experts": len(experts),
                        "bytes": layer_entry["bytes"],
                        "elapsed_s": elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        close_fds(fd_cache)

    tmp_path = layout_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(layout, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(layout_path)
    return layout


def main() -> None:
    args = parse_args()
    route_selection = (
        load_route_selection(args.route_jsonl, args.top_per_layer)
        if args.route_jsonl is not None
        else None
    )
    by_expert = load_manifest(
        args.manifest_csv,
        layers=parse_int_filter(args.layers),
        experts=parse_int_filter(args.experts),
        route_selection=route_selection,
    )
    if not by_expert:
        raise ValueError("no routed expert tensors matched the requested layer/expert filter")
    layout = build_layout(
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        manifest_csv=args.manifest_csv,
        by_expert=by_expert,
        overwrite=args.overwrite,
    )
    total_bytes = sum(int(layer["bytes"]) for layer in layout["layers"].values())
    print(
        json.dumps(
            {
                "event": "packed_layout_done",
                "layers": len(layout["layers"]),
                "experts": len(by_expert),
                "bytes": total_bytes,
                "layout": str(args.out_dir / "layout.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
