"""Lazy-expert wrapper around the official DeepSeek-V4 inference model."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import safe_open

from spark_runtime.native_packed_loader import load_native_packed_loader


@dataclass(frozen=True)
class ManifestRow:
    source_name: str
    target_name: str
    role: str
    transform: str
    source_shard: str


@dataclass(frozen=True)
class PackedExpertBatch:
    tensors: dict[str, torch.Tensor]
    tensor_infos: dict[str, dict[str, Any]]
    read_seconds: float
    read_bytes: int
    _storage: bytearray | memoryview | torch.Tensor


class PackedExpertStore:
    def __init__(self, layout_path: Path):
        self.layout_path = layout_path
        self.base_dir = layout_path.parent
        self.layout = json.loads(layout_path.read_text())
        self.layers: dict[str, Any] = self.layout.get("layers", {})
        self.pinned_staging = os.getenv("DEEPSEEK_SPARK_PACKED_PINNED_STAGING", "0") in {
            "1",
            "true",
            "yes",
        }
        self.reuse_staging = os.getenv("DEEPSEEK_SPARK_PACKED_REUSE_STAGING", "0") in {
            "1",
            "true",
            "yes",
        }
        self.native_loader = os.getenv("DEEPSEEK_SPARK_PACKED_NATIVE_LOADER", "0") in {
            "1",
            "true",
            "yes",
        }
        self.native_pinned_staging = os.getenv(
            "DEEPSEEK_SPARK_PACKED_NATIVE_PINNED_STAGING", "0"
        ) in {
            "1",
            "true",
            "yes",
        }
        self.native_materializer = os.getenv("DEEPSEEK_SPARK_NATIVE_MATERIALIZER", "0") in {
            "1",
            "true",
            "yes",
        }
        self._native_module = load_native_packed_loader() if self.native_loader else None
        self._fds: dict[Path, int] = {}
        self._lock = threading.Lock()
        self._thread_local = threading.local()

    @classmethod
    def from_env(cls) -> "PackedExpertStore | None":
        value = os.getenv("DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT")
        if not value:
            return None
        return cls(Path(value))

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()

    def _fd(self, path: Path) -> int:
        with self._lock:
            fd = self._fds.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY)
                self._fds[path] = fd
            return fd

    def tensors_for_expert(self, layer_id: int, expert_id: int) -> PackedExpertBatch | None:
        layer = self.layers.get(str(layer_id))
        if layer is None:
            return None
        expert = layer.get("experts", {}).get(str(expert_id))
        if expert is None:
            return None
        path = Path(layer["path"])
        if not path.is_absolute():
            path = self.base_dir / path
        offset = int(expert["offset"])
        size = int(expert["bytes"])
        fd = self._fd(path)
        start = time.monotonic()
        storage = self._read_block(fd, size, offset)
        read_seconds = time.monotonic() - start
        if len(storage) != size:
            raise OSError(f"short packed expert read: layer={layer_id} expert={expert_id}")
        tensor_infos = expert["tensors"]
        tensors = (
            {}
            if self.native_materializer and isinstance(storage, torch.Tensor)
            else {
                target_name: self._tensor_from_storage(storage, tensor_info)
                for target_name, tensor_info in tensor_infos.items()
            }
        )
        return PackedExpertBatch(
            tensors=tensors,
            tensor_infos=tensor_infos,
            read_seconds=read_seconds,
            read_bytes=size,
            _storage=storage,
        )

    def copy_storage_to_tensors(
        self,
        storage: torch.Tensor,
        targets: list[torch.Tensor],
        offsets: list[int],
        *,
        non_blocking: bool,
    ) -> int:
        if self._native_module is None:
            raise RuntimeError("native packed loader is not enabled")
        return int(
            self._native_module.copy_storage_to_tensors(
                storage,
                targets,
                offsets,
                non_blocking,
            )
        )

    def _read_block(self, fd: int, size: int, offset: int) -> bytearray | memoryview | torch.Tensor:
        if self.native_loader:
            staging = getattr(self._thread_local, "packed_native_tensor", None)
            if (
                staging is None
                or staging.numel() < size
                or bool(getattr(self._thread_local, "packed_native_pinned", False))
                != self.native_pinned_staging
            ):
                staging = torch.empty(
                    size,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=self.native_pinned_staging,
                )
                self._thread_local.packed_native_tensor = staging
                self._thread_local.packed_native_pinned = self.native_pinned_staging
            view = staging[:size]
            read_bytes = self._native_module.read_into(fd, view, size, offset)
            if read_bytes != size:
                return view[:read_bytes]
            return view
        if not self.pinned_staging:
            if self.reuse_staging and hasattr(os, "preadv"):
                slab = getattr(self._thread_local, "packed_read_slab", None)
                if slab is None or len(slab) < size:
                    slab = bytearray(size)
                    self._thread_local.packed_read_slab = slab
                view = memoryview(slab)[:size]
                read_bytes = os.preadv(fd, [view], offset)
                if read_bytes != size:
                    return view[:read_bytes]
                return view
            return bytearray(os.pread(fd, size, offset))
        staging = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
        view = memoryview(staging.numpy())
        read_bytes = os.preadv(fd, [view], offset)
        if read_bytes != size:
            return staging[:read_bytes]
        return staging

    @staticmethod
    def _tensor_from_storage(
        storage: bytearray | memoryview | torch.Tensor,
        tensor_info: dict[str, Any],
    ) -> torch.Tensor:
        offset = int(tensor_info["offset"])
        size = int(tensor_info["bytes"])
        dtype = _source_dtype_to_torch(tensor_info["source_dtype"])
        shape = [int(dim) for dim in tensor_info["source_shape"]]
        if isinstance(storage, torch.Tensor):
            return storage[offset : offset + size].view(dtype).reshape(shape)
        view = memoryview(storage)[offset : offset + size]
        return torch.frombuffer(view, dtype=dtype).reshape(shape)


def _source_dtype_to_torch(dtype_name: str) -> torch.dtype:
    mapping = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F8_E8M0": torch.float8_e8m0fnu,
        "I8": torch.int8,
        "I32": torch.int32,
        "I64": torch.int64,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        raise ValueError(f"unsupported packed source dtype: {dtype_name}") from exc


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
        self.packed_store = PackedExpertStore.from_env()

    def close(self) -> None:
        self._handles.clear()
        if self.packed_store is not None:
            self.packed_store.close()

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

    def packed_tensors_for_expert(self, layer_id: int, expert_id: int) -> PackedExpertBatch | None:
        if self.packed_store is None:
            return None
        return self.packed_store.tensors_for_expert(layer_id, expert_id)


@dataclass
class ExpertCacheStats:
    enabled: bool = False
    max_entries: int = 0
    policy: str = "global_lru"
    layer_quota: int = 0
    entries: int = 0
    resident_bytes: int = 0
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    evictions: int = 0
    materialize_seconds: float = 0.0
    copied_bytes: int = 0
    routed_calls: int = 0
    routed_tokens: int = 0
    routed_activations: int = 0
    packed_loads: int = 0
    packed_misses: int = 0
    packed_read_seconds: float = 0.0
    packed_read_bytes: int = 0
    arena_reuses: int = 0
    arena_allocations: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_entries": self.max_entries,
            "policy": self.policy,
            "layer_quota": self.layer_quota,
            "entries": self.entries,
            "resident_bytes": self.resident_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "inserts": self.inserts,
            "evictions": self.evictions,
            "materialize_seconds": self.materialize_seconds,
            "copied_bytes": self.copied_bytes,
            "routed_calls": self.routed_calls,
            "routed_tokens": self.routed_tokens,
            "routed_activations": self.routed_activations,
            "packed_loads": self.packed_loads,
            "packed_misses": self.packed_misses,
            "packed_read_seconds": self.packed_read_seconds,
            "packed_read_bytes": self.packed_read_bytes,
            "arena_reuses": self.arena_reuses,
            "arena_allocations": self.arena_allocations,
        }


def _stats_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if isinstance(after_value, bool):
            delta[key] = after_value
        elif isinstance(after_value, (int, float)) and isinstance(before_value, (int, float)):
            delta[key] = after_value - before_value
        else:
            delta[key] = after_value
    return delta


class ExpertCacheManager:
    def __init__(self, max_entries: int):
        self.max_entries = max(0, max_entries)
        self.enabled = self.max_entries > 0
        self.policy = os.getenv("DEEPSEEK_SPARK_EXPERT_CACHE_POLICY", "global_lru")
        if self.policy not in {"global_lru", "layer_lru"}:
            raise ValueError(f"unknown expert cache policy: {self.policy}")
        layer_count = max(1, int(os.getenv("DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_COUNT", "43")))
        default_layer_quota = max(1, self.max_entries // layer_count) if self.enabled else 0
        self.layer_quota = int(
            os.getenv("DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_QUOTA", str(default_layer_quota))
        )
        trace_path = os.getenv("DEEPSEEK_SPARK_EXPERT_ROUTE_TRACE_JSONL")
        self.route_trace_path = Path(trace_path) if trace_path else None
        self._lock = threading.Lock()
        self._entries: dict[tuple[int, int], LazyRoutedExpert] = {}
        self._access_counter = 0
        self._resident_bytes = 0
        self.stats = ExpertCacheStats(
            enabled=self.enabled,
            max_entries=self.max_entries,
            policy=self.policy,
            layer_quota=self.layer_quota,
        )
        if self.route_trace_path is not None:
            self.route_trace_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "ExpertCacheManager":
        return cls(int(os.getenv("DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES", "0")))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self.stats.entries = len(self._entries)
            self.stats.resident_bytes = self._resident_bytes
            return self.stats.as_dict()

    def touch(self, expert: "LazyRoutedExpert") -> None:
        with self._lock:
            self._access_counter += 1
            expert._cache_last_used = self._access_counter
            self.stats.hits += 1

    def record_miss(self) -> None:
        with self._lock:
            self.stats.misses += 1

    def record_materialize(self, *, seconds: float, copied_bytes: int) -> None:
        with self._lock:
            self.stats.materialize_seconds += seconds
            self.stats.copied_bytes += copied_bytes

    def record_packed_load(self, *, hit: bool, seconds: float = 0.0, read_bytes: int = 0) -> None:
        with self._lock:
            if hit:
                self.stats.packed_loads += 1
                self.stats.packed_read_seconds += seconds
                self.stats.packed_read_bytes += read_bytes
            else:
                self.stats.packed_misses += 1

    def record_arena_acquire(self, *, reused: bool) -> None:
        with self._lock:
            if reused:
                self.stats.arena_reuses += 1
            else:
                self.stats.arena_allocations += 1

    def insert(self, expert: "LazyRoutedExpert", resident_bytes: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            key = expert.cache_key
            if key in self._entries:
                return
            self._access_counter += 1
            expert._cache_last_used = self._access_counter
            expert._cache_resident_bytes = resident_bytes
            self._entries[key] = expert
            self._resident_bytes += resident_bytes
            self.stats.inserts += 1
            self._evict_locked()

    def record_routing(
        self,
        *,
        layer_id: int,
        input_ids: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> None:
        flat_indices = expert_indices.flatten()
        token_count = input_ids.numel()
        activation_count = flat_indices.numel()
        with self._lock:
            self.stats.routed_calls += 1
            self.stats.routed_tokens += token_count
            self.stats.routed_activations += activation_count
        if self.route_trace_path is not None:
            self._write_route_trace(
                layer_id=layer_id,
                input_ids=input_ids,
                expert_indices=expert_indices,
                flat_indices=flat_indices,
            )

    def _write_route_trace(
        self,
        *,
        layer_id: int,
        input_ids: torch.Tensor,
        expert_indices: torch.Tensor,
        flat_indices: torch.Tensor,
    ) -> None:
        ids = flat_indices.detach().cpu().tolist()
        histogram: dict[int, int] = {}
        for expert_id in ids:
            histogram[int(expert_id)] = histogram.get(int(expert_id), 0) + 1
        nested_indices = expert_indices.detach().cpu().tolist()
        if nested_indices and not isinstance(nested_indices[0], list):
            nested_indices = [nested_indices]
        payload = {
            "timestamp": time.time(),
            "event": "expert_route",
            "layer_id": layer_id,
            "token_ids": [int(value) for value in input_ids.flatten().detach().cpu().tolist()],
            "token_count": int(input_ids.numel()),
            "activation_count": int(flat_indices.numel()),
            "expert_indices": [
                [int(value) for value in row]
                for row in nested_indices
            ],
            "expert_histogram": {str(key): value for key, value in sorted(histogram.items())},
        }
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self.route_trace_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _evict_locked(self) -> None:
        if self.policy == "layer_lru" and self.layer_quota > 0:
            by_layer: dict[int, list[LazyRoutedExpert]] = defaultdict(list)
            for expert in self._entries.values():
                by_layer[expert.layer_id].append(expert)
            for experts in by_layer.values():
                while len(experts) > self.layer_quota:
                    victim = min(experts, key=lambda expert: expert._cache_last_used)
                    experts.remove(victim)
                    self._drop_locked(victim)
        while len(self._entries) > self.max_entries:
            _, victim = min(
                self._entries.items(),
                key=lambda item: item[1]._cache_last_used,
            )
            self._drop_locked(victim)

    def evict(self, expert: "LazyRoutedExpert") -> None:
        with self._lock:
            self._drop_locked(expert)

    def _drop_locked(self, expert: "LazyRoutedExpert") -> None:
        key = expert.cache_key
        if key not in self._entries:
            return
        self._entries.pop(key, None)
        self._resident_bytes -= expert._cache_resident_bytes
        expert._drop_cached()
        self.stats.evictions += 1


_EXPERT_CACHE_MANAGER: ExpertCacheManager | None = None
_MATERIALIZE_EXECUTOR: ThreadPoolExecutor | None = None
_EXPERT_ARENA: "ExpertArena | None" = None


class ExpertArena:
    def __init__(self, max_slots: int):
        self.max_slots = max(0, max_slots)
        self.enabled = self.max_slots > 0
        self._lock = threading.Lock()
        self._pool: list[torch.nn.Module] = []

    @classmethod
    def from_env(cls) -> "ExpertArena":
        return cls(int(os.getenv("DEEPSEEK_SPARK_EXPERT_ARENA_SLOTS", "0")))

    def acquire(self, factory) -> tuple[torch.nn.Module, bool]:
        if not self.enabled:
            return factory(), False
        with self._lock:
            if self._pool:
                return self._pool.pop(), True
        return factory(), False

    def release(self, expert: torch.nn.Module) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            if len(self._pool) >= self.max_slots:
                return False
            self._pool.append(expert)
            return True


def configure_expert_cache(max_entries: int | None = None) -> ExpertCacheManager:
    global _EXPERT_CACHE_MANAGER
    _EXPERT_CACHE_MANAGER = (
        ExpertCacheManager.from_env()
        if max_entries is None
        else ExpertCacheManager(max_entries=max_entries)
    )
    return _EXPERT_CACHE_MANAGER


def expert_cache_manager() -> ExpertCacheManager:
    global _EXPERT_CACHE_MANAGER
    if _EXPERT_CACHE_MANAGER is None:
        _EXPERT_CACHE_MANAGER = ExpertCacheManager.from_env()
    return _EXPERT_CACHE_MANAGER


def expert_cache_snapshot() -> dict[str, Any]:
    return expert_cache_manager().snapshot()


def expert_cache_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return _stats_delta(before, after)


def expert_arena() -> ExpertArena:
    global _EXPERT_ARENA
    if _EXPERT_ARENA is None:
        _EXPERT_ARENA = ExpertArena.from_env()
    return _EXPERT_ARENA


def expert_materialize_executor() -> ThreadPoolExecutor | None:
    global _MATERIALIZE_EXECUTOR
    workers = int(os.getenv("DEEPSEEK_SPARK_EXPERT_MATERIALIZE_WORKERS", "0"))
    if workers <= 1:
        return None
    if _MATERIALIZE_EXECUTOR is None:
        _MATERIALIZE_EXECUTOR = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="expert-materialize",
        )
    return _MATERIALIZE_EXECUTOR


class LazyRoutedExpert(torch.nn.Module):
    def __init__(
        self,
        official_model,
        layer_id: int,
        expert_id: int,
        args,
        weight_store: WeightStore,
        cache_manager: ExpertCacheManager,
    ):
        super().__init__()
        self.official_model = official_model
        self.layer_id = layer_id
        self.expert_id = expert_id
        self.args = args
        self.weight_store = weight_store
        self.cache_manager = cache_manager
        self._expert: torch.nn.Module | None = None
        self._cache_last_used = 0
        self._cache_resident_bytes = 0
        self.direct_param_copy = os.getenv("DEEPSEEK_SPARK_DIRECT_PARAM_COPY", "0") in {
            "1",
            "true",
            "yes",
        }
        self.non_blocking_param_copy = os.getenv(
            "DEEPSEEK_SPARK_PARAM_COPY_NON_BLOCKING", "0"
        ) in {
            "1",
            "true",
            "yes",
        }
        self.native_materializer = os.getenv("DEEPSEEK_SPARK_NATIVE_MATERIALIZER", "0") in {
            "1",
            "true",
            "yes",
        }

    @property
    def prefix(self) -> str:
        return f"layers.{self.layer_id}.ffn.experts.{self.expert_id}"

    @property
    def cache_key(self) -> tuple[int, int]:
        return self.layer_id, self.expert_id

    def materialize(self, device: torch.device) -> torch.nn.Module:
        if self._expert is not None:
            self.cache_manager.touch(self)
            return self._expert
        self.cache_manager.record_miss()
        start = time.monotonic()
        dtype = torch.float4_e2m1fn_x2 if self.args.expert_dtype == "fp4" else None
        expert, arena_reused = expert_arena().acquire(
            lambda: self.official_model.Expert(
                self.args.dim,
                self.args.moe_inter_dim,
                dtype=dtype,
                swiglu_limit=self.args.swiglu_limit,
            ).to(device)
        )
        self.cache_manager.record_arena_acquire(reused=arena_reused)
        packed_batch = self.weight_store.packed_tensors_for_expert(self.layer_id, self.expert_id)
        if packed_batch is not None:
            self.cache_manager.record_packed_load(
                hit=True,
                seconds=packed_batch.read_seconds,
                read_bytes=packed_batch.read_bytes,
            )
        elif self.weight_store.packed_store is not None:
            self.cache_manager.record_packed_load(hit=False)
        copied_bytes = 0
        parameters = list(expert.named_parameters())
        if (
            self.native_materializer
            and packed_batch is not None
            and isinstance(packed_batch._storage, torch.Tensor)
            and self.weight_store.packed_store is not None
        ):
            native_targets = []
            native_offsets = []
            fallback_parameters = []
            for name, param in parameters:
                target = f"{self.prefix}.{name}"
                tensor_info = packed_batch.tensor_infos.get(target)
                if tensor_info is None:
                    fallback_parameters.append((name, param))
                    continue
                native_targets.append(param.data)
                native_offsets.append(int(tensor_info["offset"]))
            if native_targets:
                copied_bytes += self.weight_store.packed_store.copy_storage_to_tensors(
                    packed_batch._storage,
                    native_targets,
                    native_offsets,
                    non_blocking=self.non_blocking_param_copy,
                )
            parameters = fallback_parameters
        for name, param in parameters:
            target = f"{self.prefix}.{name}"
            row = self.weight_store.rows_by_target[target]
            if packed_batch is not None and target in packed_batch.tensors:
                tensor = packed_batch.tensors[target]
            elif packed_batch is not None and target in packed_batch.tensor_infos:
                tensor = PackedExpertStore._tensor_from_storage(
                    packed_batch._storage,
                    packed_batch.tensor_infos[target],
                )
            else:
                tensor = self.weight_store.tensor_for_row(row)
            if row.transform == "reinterpret_int8_as_fp4":
                tensor = tensor.view(torch.float4_e2m1fn_x2)
            if self.direct_param_copy and tensor.device.type == "cpu":
                param.data.copy_(tensor, non_blocking=self.non_blocking_param_copy)
            else:
                tensor = tensor.to(
                    device=param.device,
                    dtype=param.dtype,
                    non_blocking=self.non_blocking_param_copy,
                )
                param.data.copy_(tensor)
            copied_bytes += param.numel() * param.element_size()
        self._expert = expert
        resident_bytes = sum(
            param.numel() * param.element_size() for param in expert.parameters()
        )
        self.cache_manager.record_materialize(
            seconds=time.monotonic() - start,
            copied_bytes=copied_bytes,
        )
        self.cache_manager.insert(self, resident_bytes)
        return expert

    def evict(self) -> None:
        if self.cache_manager.enabled:
            self.cache_manager.evict(self)
        else:
            self._drop_cached()

    def _drop_cached(self) -> None:
        if self._expert is not None:
            expert_arena().release(self._expert)
        self._expert = None
        self._cache_resident_bytes = 0
        empty_cache_mode = os.getenv("DEEPSEEK_SPARK_EMPTY_CACHE_ON_EVICT", "uncached")
        should_empty_cache = empty_cache_mode in {"1", "true", "always"} or (
            empty_cache_mode == "uncached" and not self.cache_manager.enabled
        )
        if should_empty_cache:
            torch.cuda.empty_cache()

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        expert = self.materialize(x.device)
        try:
            return expert(x, weights)
        finally:
            if not self.cache_manager.enabled:
                self.evict()


class LazyMoE(torch.nn.Module):
    def __init__(
        self,
        official_model,
        layer_id: int,
        args,
        weight_store: WeightStore,
        cache_manager: ExpertCacheManager,
    ):
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
                LazyRoutedExpert(official_model, layer_id, i, args, weight_store, cache_manager)
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
        self.experts[self.experts_start_idx].cache_manager.record_routing(
            layer_id=self.layer_id,
            input_ids=input_ids.flatten(),
            expert_indices=indices,
        )
        active_experts = [
            i for i in range(self.experts_start_idx, self.experts_end_idx) if counts[i] != 0
        ]
        self._prefetch_active_experts(active_experts, x.device)
        for i in active_experts:
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(y)
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)

    def _prefetch_active_experts(self, active_experts: list[int], device: torch.device) -> None:
        executor = expert_materialize_executor()
        if executor is None or len(active_experts) <= 1:
            return
        futures = [
            executor.submit(self.experts[expert_id].materialize, device)
            for expert_id in active_experts
        ]
        for future in futures:
            future.result()


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


def patch_lazy_moe(
    official_model,
    weight_store: WeightStore,
    cache_manager: ExpertCacheManager,
) -> None:
    class BoundLazyMoE(LazyMoE):
        def __init__(self, layer_id: int, args):
            super().__init__(official_model, layer_id, args, weight_store, cache_manager)

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
    cache_manager = configure_expert_cache()
    patch_lazy_moe(official_model, weight_store, cache_manager)
    args = load_args(official_model, config_path, max_seq_len)
    with torch.device("cuda"):
        model = official_model.Transformer(args)
    counts = load_resident_weights(model, weight_store)
    return model, args, counts, weight_store
