from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time

from fastapi.testclient import TestClient

from spark_runtime.server import create_app
from spark_runtime.worker import DedicatedEngineWorker


@dataclass
class FakeResult:
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_text: str
    full_text: str
    finish_reason: str
    reused_prefix_tokens: int
    newly_prefilled_tokens: int
    timings: dict[str, float]
    memory: dict[str, tuple[int, int]]
    parsed_message: dict | None = None


class FakeEngine:
    loaded = True
    tokenizer = None

    def __init__(self):
        self.calls = []
        self.tokenizer = self

    def decode(self, token_ids):
        return "".join({21133: "World", 438: " ="}.get(token_id, "?") for token_id in token_ids)

    def close(self):
        pass

    def health(self):
        return {"loaded": True, "max_seq_len": 1048576, "sessions": ["default"], "load_counts": {}}

    def reset_session(self, session_id):
        self.calls.append(("reset", session_id))

    async def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return FakeResult(
            prompt_token_ids=[19923],
            generated_token_ids=[21133, 438],
            generated_text="World =",
            full_text="HelloWorld =",
            finish_reason="length",
            reused_prefix_tokens=1,
            newly_prefilled_tokens=0,
            timings={"prefill_seconds": 0.0, "decode_seconds": 0.1},
            memory={"before": (1, 2), "after": (1, 2)},
        )


def test_models_endpoint():
    client = TestClient(create_app(FakeEngine()))
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "deepseek-v4-flash-dgx-spark"


def test_chat_completion_response_shape():
    engine = FakeEngine()
    client = TestClient(create_app(engine))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-v4-flash-dgx-spark",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 2,
            "spark_session_id": "default",
            "spark_cache_policy": "reuse",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "World ="
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 2
    assert body["spark_metrics"]["reused_prefix_tokens"] == 1
    assert engine.calls[0][1]["messages"] == [{"role": "user", "content": "Hello"}]
    assert engine.calls[0][1]["thinking_mode"] == "chat"


def test_chat_completion_preserves_deepseek_encoding_fields():
    engine = FakeEngine()
    client = TestClient(create_app(engine))
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "You are terse."},
                {
                    "role": "assistant",
                    "content": "I can help.",
                    "reasoning_content": "User asked for help.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "search", "arguments": "{\"query\":\"x\"}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "{\"ok\":true}",
                },
                {"role": "latest_reminder", "content": "2026-05-01,US,English"},
                {"role": "user", "content": "Hello", "task": "action"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "response_format": {"type": "json_object"},
            "spark_thinking_mode": "thinking",
            "spark_reasoning_effort": "max",
        },
    )
    assert response.status_code == 200
    call = engine.calls[0][1]
    assert call["thinking_mode"] == "thinking"
    assert call["reasoning_effort"] == "max"
    messages = call["messages"]
    assert messages[0]["tools"][0]["function"]["name"] == "search"
    assert messages[0]["response_format"] == {"type": "json_object"}
    assert messages[1]["reasoning_content"] == "User asked for help."
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == "{\"query\":\"x\"}"
    assert messages[2]["tool_call_id"] == "call_1"
    assert messages[3]["role"] == "latest_reminder"
    assert messages[4]["task"] == "action"


def test_completion_response_shape():
    client = TestClient(create_app(FakeEngine()))
    response = client.post(
        "/v1/completions",
        json={"prompt": "Hello", "max_tokens": 2, "spark_cache_policy": "reset"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "World ="
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["spark_metrics"]["finish_reason"] == "length"
    assert body["spark_metrics"]["generated_token_ids"] == [21133, 438]
    assert body["spark_metrics"]["full_text"] == "HelloWorld ="


def test_generation_queue_limit(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_SPARK_MAX_QUEUE_SIZE", "0")
    client = TestClient(create_app(FakeEngine()))
    response = client.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 1})
    assert response.status_code == 429


def test_streaming_chat_completion_shape():
    client = TestClient(create_app(FakeEngine()))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 2, "stream": True},
    ) as response:
        assert response.status_code == 200
        text = response.read().decode()
    assert "chat.completion.chunk" in text
    assert "World" in text
    assert "[DONE]" in text


def test_dedicated_worker_health_does_not_queue_behind_postfill():
    class BlockingEngine:
        tokenizer = None

        def __init__(self):
            self.pending = False

        def health(self):
            return {"loaded": True, "pending_postfill_sessions": ["s"] if self.pending else []}

        def prepare_next_logits_sync(self, session_id, expected_token_ids):
            self.pending = True
            time.sleep(0.2)
            self.pending = False
            return True

        def close(self):
            pass

    async def run():
        worker = DedicatedEngineWorker(BlockingEngine())
        task = asyncio.create_task(worker.prepare_next_logits("s", [1]))
        await asyncio.sleep(0.05)
        start = time.monotonic()
        health = worker.health()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05
        assert health["worker"]["type"] == "dedicated-thread"
        await task
        await worker.close()

    asyncio.run(run())
