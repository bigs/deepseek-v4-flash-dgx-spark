#!/usr/bin/env python3
"""Summarize routed expert trace JSONL files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_jsonl", type=Path)
    return parser.parse_args()


def threshold_count(counter: Counter[int], total: int, threshold: float) -> int:
    running = 0
    for idx, (_, count) in enumerate(counter.most_common(), start=1):
        running += count
        if total and running / total >= threshold:
            return idx
    return 0


def main() -> None:
    args = parse_args()
    layer_counts: dict[int, Counter[int]] = defaultdict(Counter)
    layer_seen: dict[int, set[int]] = defaultdict(set)
    layer_reuse_distances: dict[int, list[int]] = defaultdict(list)
    layer_last_seen: dict[int, dict[int, int]] = defaultdict(dict)
    events = 0
    routed_tokens = 0
    routed_activations = 0

    for line in args.trace_jsonl.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") != "expert_route":
            continue
        events += 1
        layer = int(row["layer_id"])
        routed_tokens += int(row["token_count"])
        routed_activations += int(row["activation_count"])
        experts = [int(value) for group in row["expert_indices"] for value in group]
        for expert_id in experts:
            layer_counts[layer][expert_id] += 1
            seen = layer_seen[layer]
            last_seen = layer_last_seen[layer]
            if expert_id in last_seen:
                layer_reuse_distances[layer].append(len(seen) - last_seen[expert_id])
            seen.add(expert_id)
            last_seen[expert_id] = len(seen)

    print(f"# Expert Route Summary: `{args.trace_jsonl}`")
    print()
    print("## Totals")
    print()
    print(f"- route events: {events}")
    print(f"- routed tokens: {routed_tokens}")
    print(f"- routed activations: {routed_activations}")
    print(f"- layers observed: {len(layer_counts)}")
    print()

    print("## Per-Layer Coverage")
    print()
    print("| Layer | Activations | Unique Experts | Top-1 | Top-5 Cov | Top-10 Cov | 50%@ | 80%@ | 90%@ | Avg Reuse Distance |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    experts_for_80 = 0
    for layer in sorted(layer_counts):
        counts = layer_counts[layer]
        total = counts.total()
        top = counts.most_common(10)
        top1 = top[0][0] if top else -1
        top5_cov = sum(count for _, count in top[:5]) / total if total else 0.0
        top10_cov = sum(count for _, count in top[:10]) / total if total else 0.0
        at50 = threshold_count(counts, total, 0.50)
        at80 = threshold_count(counts, total, 0.80)
        at90 = threshold_count(counts, total, 0.90)
        experts_for_80 += at80
        reuse = layer_reuse_distances[layer]
        avg_reuse = mean(reuse) if reuse else 0.0
        print(
            f"| {layer} | {total} | {len(counts)} | {top1} | "
            f"{top5_cov:.1%} | {top10_cov:.1%} | {at50} | {at80} | {at90} | "
            f"{avg_reuse:.1f} |"
        )

    print()
    if layer_counts:
        avg_80 = experts_for_80 / len(layer_counts)
        print("## Cache Sizing Signal")
        print()
        print(f"- experts needed for 80% coverage across observed layers: {experts_for_80}")
        print(f"- average experts per observed layer for 80% coverage: {avg_80:.1f}")


if __name__ == "__main__":
    main()

