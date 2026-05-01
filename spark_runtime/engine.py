"""Long-lived DeepSeek-V4-Flash engine for DGX Spark serving."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from transformers import AutoTokenizer


CachePolicy = Literal["reuse", "reset", "none"]


@dataclass
class EngineConfig:
    model_dir: Path
    inference_dir: Path
    manifest_csv: Path
    config: Path
    max_seq_len: int = 1_048_576
    postfill_mode: Literal["deferred", "inline", "off"] = "deferred"


@dataclass
class SessionState:
    token_ids: list[int] = field(default_factory=list)
    cache_valid: bool = False
    next_logits: Any | None = None
    pending_postfill: bool = False
    last_access: float = field(default_factory=time.monotonic)


@dataclass
class GenerateResult:
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_text: str
    full_text: str
    finish_reason: Literal["stop", "length"]
    reused_prefix_tokens: int
    newly_prefilled_tokens: int
    timings: dict[str, float]
    memory: dict[str, tuple[int, int]]
    parsed_message: dict[str, Any] | None = None
    deferred_postfill_session_id: str | None = None
    deferred_postfill_token_ids: list[int] | None = None


class DeepSeekSparkEngine:
    """Single-worker engine with one active model and session KV cache."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.model = None
        self.model_args = None
        self.weight_store = None
        self.load_counts: dict[str, int] = {}
        self.tokenizer = None
        self.stop_token_ids: set[int] = set()
        self.sessions: dict[str, SessionState] = {}
        self.active_session_id: str | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.loaded:
            return
        import torch
        from spark_runtime.lazy_official_runtime import build_lazy_model

        start = time.monotonic()
        self.model, self.model_args, self.load_counts, self.weight_store = build_lazy_model(
            self.config.model_dir,
            self.config.inference_dir,
            self.config.manifest_csv,
            self.config.config,
            self.config.max_seq_len,
        )
        torch.set_default_device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_dir,
            trust_remote_code=True,
        )
        self.stop_token_ids = self._load_stop_token_ids()
        self.load_seconds = time.monotonic() - start

    def close(self) -> None:
        if self.weight_store is not None:
            self.weight_store.close()

    def reset_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        if self.active_session_id == session_id:
            self.active_session_id = None

    def health(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "max_seq_len": self.config.max_seq_len,
            "sessions": list(self.sessions),
            "active_session_id": self.active_session_id,
            "pending_postfill_sessions": [
                session_id
                for session_id, session in self.sessions.items()
                if session.pending_postfill
            ],
            "load_counts": self.load_counts,
        }

    async def generate(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int = 16,
        session_id: str = "default",
        cache_policy: CachePolicy = "reuse",
        thinking_mode: Literal["chat", "thinking"] = "chat",
        reasoning_effort: Literal["max", "high"] | None = None,
    ) -> GenerateResult:
        return await asyncio.to_thread(
            self.generate_sync,
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            session_id=session_id,
            cache_policy=cache_policy,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )

    def generate_sync(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int = 16,
        session_id: str = "default",
        cache_policy: CachePolicy = "reuse",
        thinking_mode: Literal["chat", "thinking"] = "chat",
        reasoning_effort: Literal["max", "high"] | None = None,
    ) -> GenerateResult:
        with self._lock:
            return self._generate_locked(
                prompt=prompt,
                messages=messages,
                max_tokens=max_tokens,
                session_id=session_id,
                cache_policy=cache_policy,
                thinking_mode=thinking_mode,
                reasoning_effort=reasoning_effort,
            )

    def _generate_locked(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        max_tokens: int,
        session_id: str,
        cache_policy: CachePolicy,
        thinking_mode: Literal["chat", "thinking"],
        reasoning_effort: Literal["max", "high"] | None,
    ) -> GenerateResult:
        import torch

        self.load()
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")

        timings: dict[str, float] = {}
        memory: dict[str, tuple[int, int]] = {"before": torch.cuda.mem_get_info()}
        prompt_token_ids = self.encode(
            prompt=prompt,
            messages=messages,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )
        if len(prompt_token_ids) > self.config.max_seq_len:
            raise ValueError(
                f"prompt has {len(prompt_token_ids)} tokens, max_seq_len={self.config.max_seq_len}"
            )

        session = self._select_session(session_id, cache_policy, prompt_token_ids)
        reused_prefix_tokens = self._reused_prefix_len(session, cache_policy, prompt_token_ids)
        newly_prefilled_tokens = len(prompt_token_ids) - reused_prefix_tokens

        prefill_start = time.monotonic()
        logits = None
        cache_resume_hit = False
        with torch.inference_mode():
            if newly_prefilled_tokens:
                input_ids = torch.tensor(
                    [prompt_token_ids[reused_prefix_tokens:]],
                    dtype=torch.long,
                    device="cuda",
                )
                logits = self.model(input_ids, reused_prefix_tokens)
            elif prompt_token_ids and session.next_logits is not None:
                logits = session.next_logits
                cache_resume_hit = True
            elif prompt_token_ids:
                # Fallback for a cache slot that has token ids but no stored
                # next-token logits. This consumes only the final cached token,
                # not the full prefix.
                input_ids = torch.tensor([[prompt_token_ids[-1]]], dtype=torch.long, device="cuda")
                logits = self.model(input_ids, len(prompt_token_ids) - 1)
        timings["prefill_seconds"] = time.monotonic() - prefill_start
        timings["cache_resume_hit"] = float(cache_resume_hit)

        generated: list[int] = []
        finish_reason: Literal["stop", "length"] = "length"
        decode_start = time.monotonic()
        with torch.inference_mode():
            for step in range(max_tokens):
                if logits is None:
                    break
                next_token = int(logits.argmax(dim=-1).item())
                generated.append(next_token)
                if next_token in self.stop_token_ids:
                    finish_reason = "stop"
                    break
                if step + 1 < max_tokens:
                    decode_input = torch.tensor([[next_token]], dtype=torch.long, device="cuda")
                    logits = self.model(decode_input, len(prompt_token_ids) + step)
        timings["decode_seconds"] = time.monotonic() - decode_start

        full_token_ids = prompt_token_ids + generated
        next_logits_valid = not generated
        deferred_postfill_token_ids = None
        if cache_policy != "none":
            postfill_start = time.monotonic()
            if generated and self.config.postfill_mode == "inline":
                if finish_reason != "stop":
                    with torch.inference_mode():
                        input_ids = torch.tensor([[generated[-1]]], dtype=torch.long, device="cuda")
                        logits = self.model(input_ids, len(prompt_token_ids) + len(generated) - 1)
                    next_logits_valid = True
            elif generated and self.config.postfill_mode == "deferred":
                if finish_reason != "stop":
                    deferred_postfill_token_ids = full_token_ids.copy()
            timings["postfill_seconds"] = time.monotonic() - postfill_start
            session.token_ids = full_token_ids
            session.cache_valid = True
            session.next_logits = logits if next_logits_valid else None
            session.pending_postfill = deferred_postfill_token_ids is not None
            session.last_access = time.monotonic()
            self.active_session_id = session_id
        else:
            timings["postfill_seconds"] = 0.0

        generated_text = self.tokenizer.decode(generated) if generated else ""
        full_text = self.tokenizer.decode(full_token_ids)
        parsed_message = (
            self._parse_completion_text(generated_text, thinking_mode)
            if messages is not None and generated_text
            else None
        )
        memory["after"] = torch.cuda.mem_get_info()
        return GenerateResult(
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated,
            generated_text=generated_text,
            full_text=full_text,
            finish_reason=finish_reason,
            reused_prefix_tokens=reused_prefix_tokens,
            newly_prefilled_tokens=newly_prefilled_tokens,
            timings=timings,
            memory=memory,
            parsed_message=parsed_message,
            deferred_postfill_session_id=session_id if deferred_postfill_token_ids else None,
            deferred_postfill_token_ids=deferred_postfill_token_ids,
        )

    async def prepare_next_logits(self, session_id: str, expected_token_ids: list[int]) -> bool:
        return await asyncio.to_thread(
            self.prepare_next_logits_sync,
            session_id,
            expected_token_ids,
        )

    def prepare_next_logits_sync(self, session_id: str, expected_token_ids: list[int]) -> bool:
        with self._lock:
            return self._prepare_next_logits_locked(session_id, expected_token_ids)

    def _prepare_next_logits_locked(self, session_id: str, expected_token_ids: list[int]) -> bool:
        import torch

        if not expected_token_ids:
            return False
        session = self.sessions.get(session_id)
        if session is None:
            return False
        if self.active_session_id != session_id:
            session.pending_postfill = False
            return False
        if session.token_ids != expected_token_ids:
            return False
        if session.next_logits is not None:
            session.pending_postfill = False
            return True

        with torch.inference_mode():
            input_ids = torch.tensor([[expected_token_ids[-1]]], dtype=torch.long, device="cuda")
            session.next_logits = self.model(input_ids, len(expected_token_ids) - 1)
        session.pending_postfill = False
        session.last_access = time.monotonic()
        return True

    def encode(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        thinking_mode: Literal["chat", "thinking"] = "chat",
        reasoning_effort: Literal["max", "high"] | None = None,
    ) -> list[int]:
        if messages is not None:
            rendered = self._render_messages(
                messages,
                thinking_mode=thinking_mode,
                reasoning_effort=reasoning_effort,
            )
            return self.tokenizer.encode(rendered)
        if prompt is None:
            prompt = ""
        return self.tokenizer.encode(prompt)

    def _render_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking_mode: Literal["chat", "thinking"],
        reasoning_effort: Literal["max", "high"] | None,
    ) -> str:
        encoding_dir = self.config.model_dir / "encoding"
        if str(encoding_dir) not in sys.path:
            sys.path.insert(0, str(encoding_dir))
        try:
            from encoding_dsv4 import encode_messages

            return encode_messages(
                messages,
                thinking_mode=thinking_mode,
                reasoning_effort=reasoning_effort,
            )
        except Exception:
            parts = []
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("assistant:")
            return "\n".join(parts)

    def _parse_completion_text(
        self,
        text: str,
        thinking_mode: Literal["chat", "thinking"],
    ) -> dict[str, Any] | None:
        encoding_dir = self.config.model_dir / "encoding"
        if str(encoding_dir) not in sys.path:
            sys.path.insert(0, str(encoding_dir))
        try:
            from encoding_dsv4 import parse_message_from_completion_text

            return parse_message_from_completion_text(text, thinking_mode=thinking_mode)
        except Exception:
            return None

    def _load_stop_token_ids(self) -> set[int]:
        stop_token_ids: set[int] = set()
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            stop_token_ids.add(int(eos_token_id))
        try:
            encoded = self.tokenizer.encode("<｜end▁of▁sentence｜>", add_special_tokens=False)
            if len(encoded) == 1:
                stop_token_ids.add(int(encoded[0]))
        except Exception:
            pass
        return stop_token_ids

    def _select_session(
        self,
        session_id: str,
        cache_policy: CachePolicy,
        prompt_token_ids: list[int],
    ) -> SessionState:
        if cache_policy == "none":
            return SessionState()
        if cache_policy == "reset":
            self.sessions[session_id] = SessionState()
        session = self.sessions.setdefault(session_id, SessionState())
        if cache_policy == "reuse" and session.cache_valid:
            if prompt_token_ids[: len(session.token_ids)] == session.token_ids:
                return session
            if session.token_ids[: len(prompt_token_ids)] == prompt_token_ids:
                # Caller is asking to continue from an earlier prefix. The
                # single cache slot cannot rewind cheaply, so reset.
                self.sessions[session_id] = SessionState()
                return self.sessions[session_id]
            self.sessions[session_id] = SessionState()
            return self.sessions[session_id]
        return session

    @staticmethod
    def _reused_prefix_len(
        session: SessionState,
        cache_policy: CachePolicy,
        prompt_token_ids: list[int],
    ) -> int:
        if cache_policy != "reuse" or not session.cache_valid:
            return 0
        if prompt_token_ids[: len(session.token_ids)] == session.token_ids:
            return len(session.token_ids)
        return 0


def config_from_env() -> EngineConfig:
    model_dir = Path(os.getenv("DEEPSEEK_SPARK_MODEL_DIR", "/model"))
    return EngineConfig(
        model_dir=model_dir,
        inference_dir=Path(os.getenv("DEEPSEEK_SPARK_INFERENCE_DIR", str(model_dir / "inference"))),
        manifest_csv=Path(
            os.getenv("DEEPSEEK_SPARK_MANIFEST_CSV", "/repo/results/spark-66c9/weight-manifest.csv")
        ),
        config=Path(os.getenv("DEEPSEEK_SPARK_CONFIG", str(model_dir / "inference/config.json"))),
        max_seq_len=int(os.getenv("DEEPSEEK_SPARK_MAX_SEQ_LEN", "1048576")),
        postfill_mode=os.getenv("DEEPSEEK_SPARK_POSTFILL_MODE", "deferred"),
    )
