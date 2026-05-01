#!/usr/bin/env python3
"""Inspect DeepSeek-V4-Flash safetensor metadata without loading tensor data."""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LAYER_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\.(?P<matrix>w[123])\.(?P<kind>weight|scale)$"
)
MTP_EXPERT_RE = re.compile(
    r"^mtp\.(?P<mtp>\d+)\.ffn\.experts\.(?P<expert>\d+)\.(?P<matrix>w[123])\.(?P<kind>weight|scale)$"
)
LAYER_SHARED_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.shared_experts\.(?P<matrix>w[123])\.(?P<kind>weight|scale)$"
)
MTP_SHARED_EXPERT_RE = re.compile(
    r"^mtp\.(?P<mtp>\d+)\.ffn\.shared_experts\.(?P<matrix>w[123])\.(?P<kind>weight|scale)$"
)
LAYER_RE = re.compile(r"^layers\.(?P<layer>\d+)\.")
MTP_RE = re.compile(r"^mtp\.(?P<mtp>\d+)\.")


@dataclass(frozen=True)
class TensorInfo:
    name: str
    file: str
    dtype: str
    shape: list[int]
    bytes: int
    data_start: int
    data_end: int
    file_data_start: int
    file_data_end: int
    component: str
    layer: int | None
    expert: int | None
    matrix: str | None
    kind: str | None


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


def classify(name: str) -> tuple[str, int | None, int | None, str | None, str | None]:
    match = LAYER_EXPERT_RE.match(name)
    if match:
        return (
            "core_routed_experts",
            int(match.group("layer")),
            int(match.group("expert")),
            match.group("matrix"),
            match.group("kind"),
        )

    match = MTP_EXPERT_RE.match(name)
    if match:
        return (
            "mtp_routed_experts",
            None,
            int(match.group("expert")),
            match.group("matrix"),
            match.group("kind"),
        )

    match = LAYER_SHARED_EXPERT_RE.match(name)
    if match:
        return (
            "core_shared_experts",
            int(match.group("layer")),
            None,
            match.group("matrix"),
            match.group("kind"),
        )

    match = MTP_SHARED_EXPERT_RE.match(name)
    if match:
        return (
            "mtp_shared_experts",
            None,
            None,
            match.group("matrix"),
            match.group("kind"),
        )

    layer = None
    match = LAYER_RE.match(name)
    if match:
        layer = int(match.group("layer"))

    if name == "embed.weight":
        return "embeddings", None, None, None, None
    if name == "head.weight":
        return "output_head", None, None, None, None
    if name == "norm.weight":
        return "final_norm", None, None, None, None
    if name.startswith("hc_head"):
        return "auxiliary_heads", None, None, None, None
    if name.startswith("mtp."):
        if ".ffn.gate." in name:
            return "mtp_router", None, None, None, None
        if ".attn." in name:
            return "mtp_attention_dense", None, None, None, None
        if ".ffn_norm." in name or ".attn_norm." in name or ".norm." in name or ".enorm." in name or ".hnorm." in name:
            return "mtp_norms", None, None, None, None
        if ".hc_" in name or name.startswith("mtp.0.hc_"):
            return "mtp_hyper_correction", None, None, None, None
        if ".e_proj." in name or ".eh_proj." in name or ".h_proj." in name:
            return "mtp_projection", None, None, None, None
        return "mtp_other", None, None, None, None

    if ".ffn.gate." in name:
        return "core_router", layer, None, None, None
    if ".attn." in name:
        return "core_attention_dense", layer, None, None, None
    if ".ffn_norm." in name or ".attn_norm." in name:
        return "core_norms", layer, None, None, None
    if ".hc_" in name:
        return "core_hyper_correction", layer, None, None, None
    return "other", layer, None, None, None


def read_safetensor_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path} is too small to be a safetensors file")
        header_len = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(header_len))
    return header_len, header


def load_tensors(model_dir: Path) -> tuple[list[TensorInfo], dict[str, dict[str, Any]]]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing index: {index_path}")

    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    shard_headers: dict[str, dict[str, Any]] = {}
    tensors: list[TensorInfo] = []

    for shard_name in sorted(set(weight_map.values())):
        shard_path = model_dir / shard_name
        header_len, header = read_safetensor_header(shard_path)
        data_base = 8 + header_len
        shard_headers[shard_name] = {
            "header_len": header_len,
            "file_size": shard_path.stat().st_size,
            "tensor_count": sum(1 for key in header if key != "__metadata__"),
        }
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            expected_file = weight_map.get(name)
            if expected_file != shard_name:
                raise ValueError(f"{name} found in {shard_name}, index maps it to {expected_file}")
            data_start, data_end = meta["data_offsets"]
            component, layer, expert, matrix, kind = classify(name)
            tensors.append(
                TensorInfo(
                    name=name,
                    file=shard_name,
                    dtype=meta["dtype"],
                    shape=list(meta["shape"]),
                    bytes=int(data_end - data_start),
                    data_start=int(data_start),
                    data_end=int(data_end),
                    file_data_start=int(data_base + data_start),
                    file_data_end=int(data_base + data_end),
                    component=component,
                    layer=layer,
                    expert=expert,
                    matrix=matrix,
                    kind=kind,
                )
            )

    if len(tensors) != len(weight_map):
        raise ValueError(f"header tensor count {len(tensors)} != index weight_map count {len(weight_map)}")
    return tensors, shard_headers


def summarize_contiguity(tensors: list[TensorInfo], component: str) -> dict[str, Any]:
    selected = [t for t in tensors if t.component == component]
    by_file: dict[str, list[TensorInfo]] = defaultdict(list)
    for tensor in selected:
        by_file[tensor.file].append(tensor)

    gaps = []
    contiguous_runs = 0
    for file_tensors in by_file.values():
        ordered = sorted(file_tensors, key=lambda t: t.file_data_start)
        if ordered:
            contiguous_runs += 1
        for prev, cur in zip(ordered, ordered[1:]):
            gap = cur.file_data_start - prev.file_data_end
            if gap:
                gaps.append(gap)
                contiguous_runs += 1

    return {
        "component": component,
        "tensor_count": len(selected),
        "files": len(by_file),
        "bytes": sum(t.bytes for t in selected),
        "nonzero_gaps": len(gaps),
        "gap_bytes": sum(gaps),
        "max_gap": max(gaps) if gaps else 0,
        "contiguous_runs": contiguous_runs,
    }


def build_summary(model_dir: Path, tensors: list[TensorInfo], shard_headers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text())
    dtype_bytes: dict[str, int] = defaultdict(int)
    dtype_counts: Counter[str] = Counter()
    component_bytes: dict[str, int] = defaultdict(int)
    component_counts: Counter[str] = Counter()
    component_dtype_bytes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    layer_bytes: dict[int, int] = defaultdict(int)
    layer_expert_bytes: dict[int, int] = defaultdict(int)
    expert_matrix_bytes: dict[str, int] = defaultdict(int)
    shard_bytes: dict[str, int] = defaultdict(int)

    for tensor in tensors:
        dtype_bytes[tensor.dtype] += tensor.bytes
        dtype_counts[tensor.dtype] += 1
        component_bytes[tensor.component] += tensor.bytes
        component_counts[tensor.component] += 1
        component_dtype_bytes[tensor.component][tensor.dtype] += tensor.bytes
        shard_bytes[tensor.file] += tensor.bytes
        if tensor.layer is not None:
            layer_bytes[tensor.layer] += tensor.bytes
        if tensor.component == "core_routed_experts" and tensor.layer is not None:
            layer_expert_bytes[tensor.layer] += tensor.bytes
        if tensor.component.endswith("routed_experts") and tensor.matrix:
            expert_matrix_bytes[f"{tensor.component}.{tensor.matrix}.{tensor.kind}"] += tensor.bytes

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "config": {
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "n_routed_experts": config.get("n_routed_experts"),
            "n_shared_experts": config.get("n_shared_experts"),
            "num_experts_per_tok": config.get("num_experts_per_tok"),
            "expert_dtype": config.get("expert_dtype"),
            "torch_dtype": config.get("torch_dtype"),
            "quantization_config": config.get("quantization_config"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "head_dim": config.get("head_dim"),
            "moe_intermediate_size": config.get("moe_intermediate_size"),
        },
        "totals": {
            "tensor_count": len(tensors),
            "payload_bytes": sum(t.bytes for t in tensors),
            "payload_human": human_bytes(sum(t.bytes for t in tensors)),
            "shard_count": len(shard_headers),
            "shard_file_bytes": sum(v["file_size"] for v in shard_headers.values()),
            "shard_file_human": human_bytes(sum(v["file_size"] for v in shard_headers.values())),
        },
        "by_dtype": {
            dtype: {"count": dtype_counts[dtype], "bytes": bytes_, "human": human_bytes(bytes_)}
            for dtype, bytes_ in sorted(dtype_bytes.items(), key=lambda item: item[1], reverse=True)
        },
        "by_component": {
            component: {
                "count": component_counts[component],
                "bytes": bytes_,
                "human": human_bytes(bytes_),
                "dtypes": {
                    dtype: {"bytes": b, "human": human_bytes(b)}
                    for dtype, b in sorted(component_dtype_bytes[component].items(), key=lambda item: item[1], reverse=True)
                },
            }
            for component, bytes_ in sorted(component_bytes.items(), key=lambda item: item[1], reverse=True)
        },
        "by_layer_bytes": {str(k): v for k, v in sorted(layer_bytes.items())},
        "by_core_expert_layer_bytes": {str(k): v for k, v in sorted(layer_expert_bytes.items())},
        "by_expert_matrix": {
            key: {"bytes": value, "human": human_bytes(value)}
            for key, value in sorted(expert_matrix_bytes.items(), key=lambda item: item[0])
        },
        "shards": {
            name: {
                **shard_headers[name],
                "payload_bytes": shard_bytes[name],
                "payload_human": human_bytes(shard_bytes[name]),
                "file_human": human_bytes(shard_headers[name]["file_size"]),
            }
            for name in sorted(shard_headers)
        },
        "contiguity": {
            component: summarize_contiguity(tensors, component)
            for component in [
                "core_routed_experts",
                "mtp_routed_experts",
                "core_shared_experts",
                "mtp_shared_experts",
                "core_attention_dense",
            ]
        },
    }


def write_csv(path: Path, tensors: list[TensorInfo]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "file",
                "component",
                "dtype",
                "shape",
                "bytes",
                "data_start",
                "data_end",
                "file_data_start",
                "file_data_end",
                "layer",
                "expert",
                "matrix",
                "kind",
            ]
        )
        for tensor in sorted(tensors, key=lambda t: (t.file, t.file_data_start, t.name)):
            writer.writerow(
                [
                    tensor.name,
                    tensor.file,
                    tensor.component,
                    tensor.dtype,
                    "x".join(str(dim) for dim in tensor.shape),
                    tensor.bytes,
                    tensor.data_start,
                    tensor.data_end,
                    tensor.file_data_start,
                    tensor.file_data_end,
                    "" if tensor.layer is None else tensor.layer,
                    "" if tensor.expert is None else tensor.expert,
                    "" if tensor.matrix is None else tensor.matrix,
                    "" if tensor.kind is None else tensor.kind,
                ]
            )


def table_lines(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for index, row in enumerate(rows):
        lines.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")
        if index == 0:
            lines.append("| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |")
    return lines


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    totals = summary["totals"]
    cfg = summary["config"]
    lines = [
        "# Checkpoint Inventory",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Model",
        "",
        f"- Model directory: `{summary['model_dir']}`",
        f"- Architecture: `{', '.join(cfg.get('architectures') or [])}`",
        f"- Layers: `{cfg.get('num_hidden_layers')}`",
        f"- Routed experts: `{cfg.get('n_routed_experts')}`",
        f"- Shared experts: `{cfg.get('n_shared_experts')}`",
        f"- Experts per token: `{cfg.get('num_experts_per_tok')}`",
        f"- Expert dtype: `{cfg.get('expert_dtype')}`",
        f"- Quantization: `{json.dumps(cfg.get('quantization_config'), sort_keys=True)}`",
        "",
        "## Totals",
        "",
        f"- Tensor payload: `{totals['payload_human']}` (`{totals['payload_bytes']}` bytes)",
        f"- Shard files: `{totals['shard_file_human']}` across `{totals['shard_count']}` safetensors",
        f"- Tensor count: `{totals['tensor_count']}`",
        "",
        "## Bytes By Dtype",
        "",
    ]
    rows = [["dtype", "count", "bytes", "human"]]
    for dtype, data in summary["by_dtype"].items():
        rows.append([dtype, str(data["count"]), str(data["bytes"]), data["human"]])
    lines.extend(table_lines(rows))
    lines.extend(["", "## Bytes By Component", ""])
    rows = [["component", "count", "bytes", "human", "largest dtypes"]]
    for component, data in summary["by_component"].items():
        largest_dtypes = ", ".join(
            f"{dtype} {dtype_data['human']}" for dtype, dtype_data in list(data["dtypes"].items())[:3]
        )
        rows.append([component, str(data["count"]), str(data["bytes"]), data["human"], largest_dtypes])
    lines.extend(table_lines(rows))

    lines.extend(["", "## Expert Matrix Bytes", ""])
    rows = [["matrix", "bytes", "human"]]
    for key, data in summary["by_expert_matrix"].items():
        rows.append([key, str(data["bytes"]), data["human"]])
    lines.extend(table_lines(rows))

    lines.extend(["", "## Contiguity Checks", ""])
    rows = [["component", "files", "tensors", "bytes", "runs", "nonzero gaps", "gap bytes", "max gap"]]
    for component, data in summary["contiguity"].items():
        rows.append(
            [
                component,
                str(data["files"]),
                str(data["tensor_count"]),
                human_bytes(data["bytes"]),
                str(data["contiguous_runs"]),
                str(data["nonzero_gaps"]),
                human_bytes(data["gap_bytes"]),
                human_bytes(data["max_gap"]),
            ]
        )
    lines.extend(table_lines(rows))

    lines.extend(["", "## Core Expert Bytes By Layer", ""])
    rows = [["layer", "bytes", "human"]]
    for layer, bytes_ in summary["by_core_expert_layer_bytes"].items():
        rows.append([layer, str(bytes_), human_bytes(bytes_)])
    lines.extend(table_lines(rows))

    lines.extend(
        [
            "",
            "## Initial Interpretation",
            "",
            "- The routed expert payload is the dominant checkpoint component and is the main target for streaming/caching.",
            "- Expert weights are stored as `I8` payloads with `F8_E8M0` scales; this matches a packed low-bit expert representation rather than ordinary BF16 weights.",
            "- Shared experts are small enough to keep resident with the dense weights.",
            "- Each full core layer's routed expert payload is roughly the same size, making layer/expert-local cache accounting straightforward.",
            "- The contiguity table shows whether component-local reads can be satisfied by long sequential ranges or require many small reads.",
            "",
            "The companion CSV contains one row per tensor with file offsets and component classification.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    tensors, shard_headers = load_tensors(model_dir)
    summary = build_summary(model_dir, tensors, shard_headers)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoint-inventory.summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(out_dir / "checkpoint-tensors.csv", tensors)
    write_markdown(out_dir / "checkpoint-inventory.md", summary)

    print(f"wrote {out_dir / 'checkpoint-inventory.summary.json'}")
    print(f"wrote {out_dir / 'checkpoint-tensors.csv'}")
    print(f"wrote {out_dir / 'checkpoint-inventory.md'}")


if __name__ == "__main__":
    main()
