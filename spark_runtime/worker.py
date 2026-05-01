"""Dedicated single-thread engine worker for the HTTP server."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
import threading
from typing import Any

from spark_runtime.engine import CachePolicy, DeepSeekSparkEngine, GenerateResult


class DedicatedEngineWorker:
    """Run all model/CUDA work on one worker thread.

    The HTTP event loop stays responsive while long model calls run. Health
    reads return the latest snapshot produced by the worker rather than queueing
    behind a long CUDA call.
    """

    def __init__(self, engine: DeepSeekSparkEngine):
        self.engine = engine
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepseek-engine")
        self._snapshot_lock = threading.Lock()
        self._health_snapshot = self.engine.health()

    @property
    def loaded(self) -> bool:
        return bool(self.health().get("loaded", False))

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    def health(self) -> dict[str, Any]:
        with self._snapshot_lock:
            snapshot = deepcopy(self._health_snapshot)
        snapshot["worker"] = {
            "type": "dedicated-thread",
            "thread_name_prefix": "deepseek-engine",
        }
        return snapshot

    async def generate(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int = 16,
        session_id: str = "default",
        cache_policy: CachePolicy = "reuse",
        thinking_mode: str = "chat",
        reasoning_effort: str | None = None,
    ) -> GenerateResult:
        return await self._call(
            self.engine.generate_sync,
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            session_id=session_id,
            cache_policy=cache_policy,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )

    async def prepare_next_logits(self, session_id: str, expected_token_ids: list[int]) -> bool:
        return await self._call(
            self.engine.prepare_next_logits_sync,
            session_id,
            expected_token_ids,
        )

    async def reset_session(self, session_id: str) -> None:
        await self._call(self.engine.reset_session, session_id)

    async def close(self) -> None:
        try:
            await self._call(self.engine.close)
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)

    async def _call(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, partial(fn, *args, **kwargs))
        self._refresh_health_snapshot()
        return result

    def _refresh_health_snapshot(self) -> None:
        snapshot = self.engine.health()
        with self._snapshot_lock:
            self._health_snapshot = snapshot
