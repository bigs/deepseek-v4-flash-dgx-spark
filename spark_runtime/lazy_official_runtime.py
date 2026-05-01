"""Lazy-expert wrapper around the official DeepSeek-V4 inference model."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import safe_open


@dataclass(frozen=True)
class ManifestRow:
    source_name: str
    target_name: str
    role: str
    transform: str
    source_shard: str


class WeightStore:
    def __init__(self, model_dir: Path, manifest_csv: Path):
        self.model_dir = model_dir
        self.rows_by_target: dict[str, ManifestRow] = {}
        self.rows_by_source: dict[str, ManifestRow] = {}
        with manifest_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                manifest_row = ManifestRow(
                    source_name=row["source_name"],
                    target_name=row["target_name"],
                    role=row["role"],
                    transform=row["transform"],
                    source_shard=row["source_shard"],
                )
                self.rows_by_source[manifest_row.source_name] = manifest_row
                if manifest_row.target_name:
                    self.rows_by_target[manifest_row.target_name] = manifest_row
        self._handles: dict[str, Any] = {}

    def close(self) -> None:
        self._handles.clear()

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = safe_open(self.model_dir / shard, framework="pt", device="cpu")
        return self._handles[shard]

    def source_tensor(self, source_name: str) -> torch.Tensor:
        row = self.rows_by_source[source_name]
        return self._handle(row.source_shard).get_tensor(source_name)

    def tensor_for_row(self, row: ManifestRow) -> torch.Tensor:
        return self._handle(row.source_shard).get_tensor(row.source_name)

    def tensor_for_target(self, target_name: str) -> tuple[ManifestRow, torch.Tensor]:
        row = self.rows_by_target[target_name]
        return row, self.tensor_for_row(row)


class LazyRoutedExpert(torch.nn.Module):
    def __init__(self, official_model, layer_id: int, expert_id: int, args, weight_store: WeightStore):
        super().__init__()
        self.official_model = official_model
        self.layer_id = layer_id
        self.expert_id = expert_id
        self.args = args
        self.weight_store = weight_store
        self._expert: torch.nn.Module | None = None

    @property
    def prefix(self) -> str:
        return f"layers.{self.layer_id}.ffn.experts.{self.expert_id}"

    def materialize(self, device: torch.device) -> torch.nn.Module:
        if self._expert is not None:
            return self._expert
        dtype = torch.float4_e2m1fn_x2 if self.args.expert_dtype == "fp4" else None
        expert = self.official_model.Expert(
            self.args.dim,
            self.args.moe_inter_dim,
            dtype=dtype,
            swiglu_limit=self.args.swiglu_limit,
        ).to(device)
        for name, param in expert.named_parameters():
            target = f"{self.prefix}.{name}"
            row, tensor = self.weight_store.tensor_for_target(target)
            if row.transform == "reinterpret_int8_as_fp4":
                tensor = tensor.view(torch.float4_e2m1fn_x2)
            tensor = tensor.to(device=param.device, dtype=param.dtype)
            param.data.copy_(tensor)
        self._expert = expert
        return expert

    def evict(self) -> None:
        self._expert = None
        torch.cuda.empty_cache()

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        expert = self.materialize(x.device)
        try:
            return expert(x, weights)
        finally:
            self.evict()


class LazyMoE(torch.nn.Module):
    def __init__(self, official_model, layer_id: int, args, weight_store: WeightStore):
        super().__init__()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.layer_id = layer_id
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = official_model.Gate(layer_id, args)
        self.experts = torch.nn.ModuleList(
            [
                LazyRoutedExpert(official_model, layer_id, i, args, weight_store)
                if self.experts_start_idx <= i < self.experts_end_idx
                else None
                for i in range(self.n_routed_experts)
            ]
        )
        assert args.n_shared_experts == 1
        self.shared_experts = official_model.Expert(
            args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit
        )

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(y)
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)


def import_official_model(inference_dir: Path):
    sys.path.insert(0, str(inference_dir))
    import model as official_model

    official_model.sparse_attn = sparse_attn_fallback
    return official_model


def sparse_attn_fallback(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Correctness-first sparse attention fallback for GB10 shared-memory limits."""
    bsz, seqlen, n_heads, head_dim = q.shape
    out = torch.empty_like(q)
    for b in range(bsz):
        for s in range(seqlen):
            idxs = topk_idxs[b, s]
            valid = idxs >= 0
            if valid.any():
                gathered = kv[b, idxs[valid].long()]
                scores = torch.einsum("hd,kd->hk", q[b, s].float(), gathered.float()) * softmax_scale
                max_score = torch.maximum(scores.max(dim=-1).values, attn_sink.float())
                exp_scores = torch.exp(scores - max_score[:, None])
                denom = exp_scores.sum(dim=-1) + torch.exp(attn_sink.float() - max_score)
                probs = exp_scores / denom[:, None]
                value = torch.einsum("hk,kd->hd", probs, gathered.float())
            else:
                value = torch.zeros(n_heads, head_dim, device=q.device, dtype=torch.float32)
            out[b, s] = value.to(q.dtype)
    return out


def patch_lazy_moe(official_model, weight_store: WeightStore):
    class BoundLazyMoE(LazyMoE):
        def __init__(self, layer_id: int, args):
            super().__init__(official_model, layer_id, args, weight_store)

    official_model.MoE = BoundLazyMoE


def load_args(official_model, config_path: Path, max_seq_len: int | None):
    config = json.loads(config_path.read_text())
    if max_seq_len is not None:
        config["max_seq_len"] = max_seq_len
    return official_model.ModelArgs(**config)


def _copy_tensor_to_param(param: torch.nn.Parameter, tensor: torch.Tensor) -> None:
    if tensor.dtype != param.dtype:
        tensor = tensor.to(dtype=param.dtype)
    tensor = tensor.to(device=param.device)
    param.data.copy_(tensor)


def _load_wo_a_bf16(param: torch.nn.Parameter, row: ManifestRow, weight_store: WeightStore) -> None:
    weight = weight_store.tensor_for_row(row)
    scale_target = row.target_name.replace("weight", "scale")
    scale_row = weight_store.rows_by_target[scale_target]
    scale = weight_store.tensor_for_row(scale_row)
    converted = (
        weight.unflatten(0, (-1, 128))
        .unflatten(-1, (-1, 128))
        .float()
        * scale[:, None, :, None].float()
    )
    converted = converted.flatten(2, 3).flatten(0, 1).bfloat16()
    _copy_tensor_to_param(param, converted)


def load_resident_weights(model: torch.nn.Module, weight_store: WeightStore) -> dict[str, int]:
    counts = defaultdict(int)
    with torch.no_grad():
        for name, param in model.named_parameters():
            row = weight_store.rows_by_target.get(name)
            if row is None:
                counts["missing"] += 1
                continue
            if row.role == "lazy_routed_expert":
                counts["skipped_lazy_routed_expert"] += 1
                continue
            if row.transform == "fp8_scaled_to_bf16":
                _load_wo_a_bf16(param, row, weight_store)
            elif row.transform == "copy":
                _copy_tensor_to_param(param, weight_store.tensor_for_row(row))
            else:
                raise ValueError(f"unexpected resident transform for {name}: {row.transform}")
            counts["loaded"] += 1
    return dict(counts)


def build_lazy_model(
    model_dir: Path,
    inference_dir: Path,
    manifest_csv: Path,
    config_path: Path,
    max_seq_len: int,
):
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_num_threads(8)
    official_model = import_official_model(inference_dir)
    weight_store = WeightStore(model_dir, manifest_csv)
    patch_lazy_moe(official_model, weight_store)
    args = load_args(official_model, config_path, max_seq_len)
    with torch.device("cuda"):
        model = official_model.Transformer(args)
    counts = load_resident_weights(model, weight_store)
    return model, args, counts, weight_store
