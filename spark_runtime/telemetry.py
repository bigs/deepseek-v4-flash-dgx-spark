"""Low-overhead telemetry helpers for Spark inference experiments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TelemetryConfig:
    jsonl_path: Path | None = None
    nvtx: bool = True
    cuda_events: bool = False
    decode_step_limit: int = 256

    @classmethod
    def from_env(cls) -> "TelemetryConfig":
        jsonl = os.getenv("DEEPSEEK_SPARK_TELEMETRY_JSONL")
        return cls(
            jsonl_path=Path(jsonl) if jsonl else None,
            nvtx=_env_bool("DEEPSEEK_SPARK_NVTX", True),
            cuda_events=_env_bool("DEEPSEEK_SPARK_CUDA_EVENTS", False),
            decode_step_limit=int(os.getenv("DEEPSEEK_SPARK_TELEMETRY_TOKEN_CAP", "256")),
        )


class TelemetryRecorder:
    """Append-only JSONL telemetry plus optional NVTX ranges."""

    def __init__(self, config: TelemetryConfig | None = None):
        self.config = config or TelemetryConfig.from_env()
        self._lock = threading.Lock()
        if self.config.jsonl_path is not None:
            self.config.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "TelemetryRecorder":
        return cls(TelemetryConfig.from_env())

    def emit(self, event: dict[str, Any]) -> None:
        if self.config.jsonl_path is None:
            return
        payload = {
            "timestamp": time.time(),
            **event,
        }
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            with self.config.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    @contextmanager
    def nvtx(self, name: str, **attrs: Any) -> Iterator[None]:
        if not self.config.nvtx:
            yield
            return
        label = _nvtx_label(name, attrs)
        pushed = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push(label)
                pushed = True
        except Exception:
            pushed = False
        try:
            yield
        finally:
            if pushed:
                try:
                    import torch

                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass

    def cuda_timer(self, torch_module: Any) -> "_CudaTimer":
        return _CudaTimer(
            enabled=self.config.cuda_events and torch_module.cuda.is_available(),
            torch_module=torch_module,
        )


class _CudaTimer:
    def __init__(self, *, enabled: bool, torch_module: Any):
        self.enabled = enabled
        self._torch = torch_module
        self._start = None
        self._end = None
        self.elapsed_ms: float | None = None

    def __enter__(self) -> "_CudaTimer":
        if self.enabled:
            self._start = self._torch.cuda.Event(enable_timing=True)
            self._end = self._torch.cuda.Event(enable_timing=True)
            self._start.record()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self._start is not None and self._end is not None:
            self._end.record()
            self._torch.cuda.synchronize()
            self.elapsed_ms = float(self._start.elapsed_time(self._end))


def _nvtx_label(name: str, attrs: dict[str, Any]) -> str:
    if not attrs:
        return name
    suffix = ",".join(f"{key}={value}" for key, value in attrs.items())
    return f"{name} {suffix}"
