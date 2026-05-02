#!/usr/bin/env python3
"""Summarize key Nsight Systems SQLite signals for Spark inference runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--jsonl", type=Path)
    return parser.parse_args()


def ns_to_s(value: int | float | None) -> float:
    return float(value or 0) / 1e9


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    conn = sqlite3.connect(args.sqlite)
    conn.row_factory = sqlite3.Row

    print(f"# Nsight Summary: `{args.sqlite}`")
    print()
    if args.jsonl and args.jsonl.exists():
        print("## Telemetry Events")
        print()
        for row in [json.loads(line) for line in args.jsonl.read_text().splitlines()]:
            if row["event"] != "engine_generate":
                print(f"- `{row['event']}`")
                continue
            timings = row["timings"]
            print(
                "- `engine_generate` "
                f"request_id={row['request_id']} "
                f"prompt={row['prompt_tokens']} generated={row['generated_tokens']} "
                f"reused={row['reused_prefix_tokens']} new={row['newly_prefilled_tokens']} "
                f"prefill={fmt(timings.get('prefill_seconds', 0))}s "
                f"decode={fmt(timings.get('decode_seconds', 0))}s "
                f"cache_hit={row['cache_resume_hit']}"
            )
            expert_cache = row.get("expert_cache")
            if expert_cache:
                delta = expert_cache.get("delta", {})
                after = expert_cache.get("after", {})
                print(
                    "  - expert cache "
                    f"enabled={after.get('enabled')} entries={after.get('entries')} "
                    f"resident_gib={fmt(after.get('resident_bytes', 0) / 1024**3)} "
                    f"hits={delta.get('hits', 0)} misses={delta.get('misses', 0)} "
                    f"evictions={delta.get('evictions', 0)} "
                    f"copied_gib={fmt(delta.get('copied_bytes', 0) / 1024**3)}"
                )
        print()

    kernel_count, kernel_ns = conn.execute(
        "select count(*), coalesce(sum(end-start), 0) from CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    memcpy_count, memcpy_ns, memcpy_bytes = conn.execute(
        "select count(*), coalesce(sum(end-start), 0), coalesce(sum(bytes), 0) "
        "from CUPTI_ACTIVITY_KIND_MEMCPY"
    ).fetchone()
    runtime_count, runtime_ns = conn.execute(
        "select count(*), coalesce(sum(end-start), 0) from CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchone()

    print("## Totals")
    print()
    print(f"- CUDA kernels: {kernel_count} calls, {fmt(ns_to_s(kernel_ns))}s total")
    print(
        f"- CUDA memcpy: {memcpy_count} copies, {fmt(ns_to_s(memcpy_ns))}s total, "
        f"{fmt(memcpy_bytes / 1024**3)} GiB"
    )
    print(f"- CUDA runtime APIs: {runtime_count} calls, {fmt(ns_to_s(runtime_ns))}s total")
    print()

    print("## Memcpy By Kind")
    print()
    print("| Kind | Calls | GPU Time | GiB | Avg MiB | Max MiB |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in conn.execute(
        """
        select coalesce(e.label, cast(m.copyKind as text)) as label,
               count(*) as calls,
               coalesce(sum(m.end-m.start), 0) as ns,
               coalesce(sum(m.bytes), 0) as bytes,
               avg(m.bytes) as avg_bytes,
               max(m.bytes) as max_bytes
        from CUPTI_ACTIVITY_KIND_MEMCPY m
        left join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind
        group by m.copyKind
        order by ns desc
        """
    ):
        print(
            f"| {row['label']} | {row['calls']} | {fmt(ns_to_s(row['ns']))}s | "
            f"{fmt(row['bytes'] / 1024**3)} | {fmt(row['avg_bytes'] / 1024**2)} | "
            f"{fmt(row['max_bytes'] / 1024**2)} |"
        )
    print()

    print("## Engine NVTX Ranges")
    print()
    print("| Range | Wall | HtoD GiB | HtoD GPU Time | DtoD GiB | Kernels | Kernel Time |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in conn.execute(
        """
        select coalesce(n.text, s.value) as label, n.start, n.end
        from NVTX_EVENTS n
        left join StringIds s on s.id=n.textId
        where coalesce(n.text, s.value) like '%engine.%' and n.end is not null
        order by n.start
        """
    ):
        start = row["start"]
        end = row["end"]
        htod_bytes, htod_ns = conn.execute(
            """
            select coalesce(sum(bytes), 0), coalesce(sum(end-start), 0)
            from CUPTI_ACTIVITY_KIND_MEMCPY
            where copyKind=1 and start>=? and end<=?
            """,
            (start, end),
        ).fetchone()
        dtod_bytes = conn.execute(
            """
            select coalesce(sum(bytes), 0)
            from CUPTI_ACTIVITY_KIND_MEMCPY
            where copyKind=8 and start>=? and end<=?
            """,
            (start, end),
        ).fetchone()[0]
        kernels, kernel_ns_for_range = conn.execute(
            """
            select count(*), coalesce(sum(end-start), 0)
            from CUPTI_ACTIVITY_KIND_KERNEL
            where start>=? and end<=?
            """,
            (start, end),
        ).fetchone()
        print(
            f"| `{row['label']}` | {fmt(ns_to_s(end - start))}s | "
            f"{fmt(htod_bytes / 1024**3)} | {fmt(ns_to_s(htod_ns))}s | "
            f"{fmt(dtod_bytes / 1024**3)} | {kernels} | {fmt(ns_to_s(kernel_ns_for_range))}s |"
        )
    print()

    print("## Top CUDA Runtime APIs")
    print()
    print("| API | Calls | Total | Avg ms | Max s |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in conn.execute(
        """
        select s.value as api, count(*) as calls, sum(r.end-r.start) as ns,
               avg(r.end-r.start) as avg_ns, max(r.end-r.start) as max_ns
        from CUPTI_ACTIVITY_KIND_RUNTIME r
        join StringIds s on s.id=r.nameId
        group by r.nameId
        order by ns desc
        limit 10
        """
    ):
        print(
            f"| `{row['api']}` | {row['calls']} | {fmt(ns_to_s(row['ns']))}s | "
            f"{fmt(row['avg_ns'] / 1e6)} | {fmt(ns_to_s(row['max_ns']))} |"
        )


if __name__ == "__main__":
    main()
