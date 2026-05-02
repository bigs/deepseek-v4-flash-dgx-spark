#!/usr/bin/env python3
"""Replay routed expert traces against a packed expert layout.

This is a read-only microbenchmark for the part of decode that currently dominates:
expert cache misses and packed expert block reads. It intentionally does not run model
math; it replays the observed layer/expert sequence, applies a cache policy, and reads
missed packed expert blocks from disk.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import time
from typing import Iterable


@dataclass(frozen=True)
class ExpertBlock:
    layer: int
    expert: int
    path: Path
    offset: int
    size: int


class ExpertCache:
    def __init__(self, policy: str, max_entries: int, layer_count: int, layer_quota: int | None):
        self.policy = policy
        self.max_entries = max_entries
        self.layer_count = layer_count
        self.layer_quota = layer_quota or max(1, max_entries // max(1, layer_count))
        self.global_lru: OrderedDict[tuple[int, int], None] = OrderedDict()
        self.layer_lru: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def check(self, layer: int, expert: int) -> bool:
        if self.policy == "none" or self.max_entries <= 0:
            self.misses += 1
            return False
        if self.policy == "global_lru":
            key = (layer, expert)
            if key in self.global_lru:
                self.global_lru.move_to_end(key)
                self.hits += 1
                return True
            self.misses += 1
            self.global_lru[key] = None
            while len(self.global_lru) > self.max_entries:
                self.global_lru.popitem(last=False)
                self.evictions += 1
            return False
        if self.policy == "layer_lru":
            cache = self.layer_lru[layer]
            if expert in cache:
                cache.move_to_end(expert)
                self.hits += 1
                return True
            self.misses += 1
            cache[expert] = None
            while len(cache) > self.layer_quota:
                cache.popitem(last=False)
                self.evictions += 1
            return False
        raise ValueError(f"unknown cache policy: {self.policy}")

    @property
    def entries(self) -> int:
        if self.policy == "global_lru":
            return len(self.global_lru)
        if self.policy == "layer_lru":
            return sum(len(cache) for cache in self.layer_lru.values())
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-json", type=Path, required=True)
    parser.add_argument("--route-jsonl", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--cache-policy", choices=["none", "global_lru", "layer_lru"], default="layer_lru")
    parser.add_argument("--cache-entries", type=int, default=1024)
    parser.add_argument("--layer-count", type=int, default=43)
    parser.add_argument("--layer-quota", type=int)
    parser.add_argument("--read-order", choices=["route", "offset"], default="route")
    parser.add_argument("--io", choices=["serial", "threaded"], default="serial")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--evict-before-iteration", action="store_true")
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


def iter_route_events(path: Path, max_events: int | None) -> Iterable[tuple[int, list[int]]]:
    emitted = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") != "expert_route":
            continue
        layer = int(row["layer_id"])
        seen: set[int] = set()
        experts: list[int] = []
        for group in row["expert_indices"]:
            for value in group:
                expert = int(value)
                if expert not in seen:
                    seen.add(expert)
                    experts.append(expert)
        yield layer, experts
        emitted += 1
        if max_events is not None and emitted >= max_events:
            break


def open_fds(blocks: dict[tuple[int, int], ExpertBlock]) -> dict[Path, int]:
    return {path: os.open(path, os.O_RDONLY) for path in sorted({block.path for block in blocks.values()})}


def close_fds(fds: dict[Path, int]) -> None:
    for fd in fds.values():
        os.close(fd)


def evict_files(blocks: dict[tuple[int, int], ExpertBlock]) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    fds = open_fds(blocks)
    try:
        for path, fd in fds.items():
            os.posix_fadvise(fd, 0, path.stat().st_size, os.POSIX_FADV_DONTNEED)
    finally:
        close_fds(fds)


def read_block(fds: dict[Path, int], block: ExpertBlock) -> int:
    data = os.pread(fds[block.path], block.size, block.offset)
    if len(data) != block.size:
        raise OSError(f"short read: layer={block.layer} expert={block.expert}")
    return (data[0] + data[-1] + len(data)) & 0xFFFFFFFF


def replay_once(args: argparse.Namespace, blocks: dict[tuple[int, int], ExpertBlock]) -> dict[str, object]:
    cache = ExpertCache(args.cache_policy, args.cache_entries, args.layer_count, args.layer_quota)
    fds = open_fds(blocks)
    checksum = 0
    events = 0
    activations = 0
    read_blocks = 0
    read_bytes = 0
    missing_layout = 0
    before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=args.workers) if args.io == "threaded" else None
    try:
        for layer, experts in iter_route_events(args.route_jsonl, args.max_events):
            events += 1
            activations += len(experts)
            misses: list[ExpertBlock] = []
            for expert in experts:
                if cache.check(layer, expert):
                    continue
                block = blocks.get((layer, expert))
                if block is None:
                    missing_layout += 1
                    continue
                misses.append(block)
            if args.read_order == "offset":
                misses.sort(key=lambda block: (block.path, block.offset))
            read_blocks += len(misses)
            read_bytes += sum(block.size for block in misses)
            if executor is None:
                for block in misses:
                    checksum = (checksum + read_block(fds, block)) & 0xFFFFFFFF
            else:
                for value in executor.map(lambda item: read_block(fds, item), misses):
                    checksum = (checksum + value) & 0xFFFFFFFF
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        close_fds(fds)
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "elapsed_s": elapsed,
        "throughput_gib_s": read_bytes / elapsed / 1024**3 if elapsed else None,
        "events": events,
        "activations": activations,
        "read_blocks": read_blocks,
        "read_bytes": read_bytes,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "cache_evictions": cache.evictions,
        "cache_entries": cache.entries,
        "missing_layout": missing_layout,
        "checksum": checksum,
        "minor_faults": after.ru_minflt - before.ru_minflt,
        "major_faults": after.ru_majflt - before.ru_majflt,
    }


def main() -> None:
    args = parse_args()
    blocks = load_layout(args.layout_json)
    samples = []
    for iteration in range(args.repeat):
        if args.evict_before_iteration:
            evict_files(blocks)
        row = replay_once(args, blocks)
        row["iteration"] = iteration
        samples.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    median_elapsed = sorted(row["elapsed_s"] for row in samples)[len(samples) // 2]
    median = next(row for row in samples if row["elapsed_s"] == median_elapsed)
    output = {
        "layout_json": str(args.layout_json),
        "route_jsonl": str(args.route_jsonl),
        "cache_policy": args.cache_policy,
        "cache_entries": args.cache_entries,
        "layer_count": args.layer_count,
        "layer_quota": args.layer_quota or max(1, args.cache_entries // max(1, args.layer_count)),
        "read_order": args.read_order,
        "io": args.io,
        "workers": args.workers,
        "repeat": args.repeat,
        "median": median,
        "samples": samples,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
