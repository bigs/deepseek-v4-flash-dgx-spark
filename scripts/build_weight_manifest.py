#!/usr/bin/env python3
"""Map HF DeepSeek-V4-Flash tensors to official inference tensor names."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from inspect_official_model import load_model_module, tensor_nbytes


MAPPING = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "lm_head": ("head", 0),
    "embed": ("embed", 0),
    "wq_b": ("wq_b", 0),
    "wo_a": ("wo_a", 0),
    "wo_b": ("wo_b", 1),
    "head": ("head", 0),
    "attn_sink": ("attn_sink", 0),
    "weights_proj": ("weights_proj", 0),
}


def converted_name(source_name: str) -> tuple[str | None, str]:
    name = source_name
    if name.startswith("model."):
        name = name[len("model.") :]
    if name.startswith("mtp.") and ("emb" in name or name.endswith("head.weight")):
        return None, "skip_mtp_embed_or_head"

    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    if any(x in name for x in ["hc", "attn_sink", "tie2eid", "ape"]):
        key = name.split(".")[-1]
    else:
        key = name.split(".")[-2]
    new_key, _ = MAPPING.get(key, (key, None))
    return name.replace(key, new_key), "mapped"


def role_for_target(name: str) -> str:
    if ".ffn.experts." in name:
        return "lazy_routed_expert"
    if ".ffn.shared_experts." in name:
        return "resident_shared_expert"
    if name.endswith(".scale") and ".attn.wo_a." in name:
        return "consumed_by_wo_a_convert"
    return "resident"


def transform_for(source_name: str, target_name: str | None, dtype: str, role: str) -> str:
    if target_name is None:
        return "skip"
    if role == "consumed_by_wo_a_convert":
        return "consume_scale_for_wo_a_bf16"
    if target_name.endswith(".attn.wo_a.weight") or ".attn.wo_a.weight" in target_name:
        return "fp8_scaled_to_bf16"
    if role == "lazy_routed_expert" and dtype == "I8" and target_name.endswith(".weight"):
        return "reinterpret_int8_as_fp4"
    return "copy"


def load_official_targets(inference_dir: Path, config_path: Path, max_seq_len: int) -> dict[str, dict[str, Any]]:
    import torch

    model_mod = load_model_module(inference_dir)
    config = json.loads(config_path.read_text())
    config["max_seq_len"] = max_seq_len
    model_args = model_mod.ModelArgs(**config)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = model_mod.Transformer(model_args)
    targets: dict[str, dict[str, Any]] = {}
    for name, param in model.named_parameters():
        targets[name] = {
            "kind": "parameter",
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "bytes": tensor_nbytes(param),
            "role": role_for_target(name),
        }
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-csv", required=True, type=Path)
    parser.add_argument("--inference-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-seq-len", type=int, default=1_048_576)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()

    targets = load_official_targets(args.inference_dir, args.config, args.max_seq_len)
    rows: list[dict[str, Any]] = []
    source_count = 0

    with args.tensor_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_count += 1
            source_name = row["name"]
            target_name, status = converted_name(source_name)
            role = role_for_target(target_name) if target_name else "skipped"
            transform = transform_for(source_name, target_name, row["dtype"], role)
            target = targets.get(target_name or "")
            rows.append(
                {
                    "source_name": source_name,
                    "target_name": target_name or "",
                    "status": status,
                    "role": role,
                    "transform": transform,
                    "source_dtype": row["dtype"],
                    "source_shape": row["shape"],
                    "source_bytes": row["bytes"],
                    "source_shard": row["file"],
                    "source_offset": row["file_data_start"],
                    "target_known": bool(target),
                    "target_dtype": target["dtype"] if target else "",
                    "target_shape": json.dumps(target["shape"]) if target else "",
                    "target_bytes": target["bytes"] if target else "",
                }
            )

    mapped_targets = {row["target_name"] for row in rows if row["target_name"]}
    missing_targets = sorted(set(targets) - mapped_targets)
    unknown_targets = sorted({row["target_name"] for row in rows if row["target_name"] and row["target_name"] not in targets})

    summary = {
        "source_tensors": source_count,
        "manifest_rows": len(rows),
        "official_target_parameters": len(targets),
        "mapped_known_targets": sum(1 for row in rows if row["target_known"]),
        "missing_official_targets": len(missing_targets),
        "unknown_mapped_targets": len(unknown_targets),
        "bytes_by_role": {},
        "rows_by_transform": {},
        "missing_targets_sample": missing_targets[:100],
        "unknown_targets_sample": unknown_targets[:100],
    }
    for row in rows:
        summary["bytes_by_role"][row["role"]] = summary["bytes_by_role"].get(row["role"], 0) + int(row["source_bytes"])
        summary["rows_by_transform"][row["transform"]] = summary["rows_by_transform"].get(row["transform"], 0) + 1

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
