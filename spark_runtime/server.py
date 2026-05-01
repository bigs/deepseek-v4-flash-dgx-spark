"""OpenAI-compatible MVP server for the DGX Spark DeepSeek-V4 runtime."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import inspect
import os
import time
import uuid
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from spark_runtime import __version__
from spark_runtime.engine import DeepSeekSparkEngine, config_from_env
from spark_runtime.worker import DedicatedEngineWorker


MODEL_ID = "deepseek-v4-flash-dgx-spark"


class TextContentBlock(BaseModel):
    type: Literal["text"]
    text: str


class ToolResultContentBlock(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str | None = None
    content: str | list[TextContentBlock]


ContentBlock = Annotated[
    TextContentBlock | ToolResultContentBlock,
    Field(discriminator="type"),
]


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ChatTool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str
    arguments: str


class AssistantToolCall(BaseModel):
    id: str | None = None
    type: Literal["function"] = "function"
    function: FunctionCall


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "latest_reminder", "developer"]
    content: str | None = None
    content_blocks: list[ContentBlock] | None = None
    reasoning_content: str | None = None
    tool_calls: list[AssistantToolCall] | None = None
    tool_call_id: str | None = None
    tools: list[ChatTool] | None = None
    response_format: ResponseFormat | None = None
    task: Literal["action", "query", "authority", "domain", "title", "read_url"] | None = None
    wo_eos: bool | None = None
    mask: int | None = None


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    tools: list[ChatTool] | None = None
    response_format: ResponseFormat | None = None
    max_tokens: int = Field(default=16, ge=0, le=4096)
    temperature: float | None = 0.0
    stream: bool = False
    spark_session_id: str = "default"
    spark_cache_policy: Literal["reuse", "reset", "none"] = "reuse"
    spark_thinking_mode: Literal["chat", "thinking"] = "chat"
    spark_reasoning_effort: Literal["max", "high"] | None = None


class CompletionRequest(BaseModel):
    model: str = MODEL_ID
    prompt: str
    max_tokens: int = Field(default=16, ge=0, le=4096)
    temperature: float | None = 0.0
    stream: bool = False
    spark_session_id: str = "default"
    spark_cache_policy: Literal["reuse", "reset", "none"] = "reuse"


def create_app(engine: DeepSeekSparkEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(lifespan_app: FastAPI):
        yield
        for task in lifespan_app.state.background_tasks:
            task.cancel()
        if lifespan_app.state.background_tasks:
            await asyncio.gather(*lifespan_app.state.background_tasks, return_exceptions=True)
        await _call_engine_method(lifespan_app.state.engine, "close")

    app = FastAPI(
        title="DeepSeek V4 Flash DGX Spark",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.engine = engine or DedicatedEngineWorker(DeepSeekSparkEngine(config_from_env()))
    app.state.max_queue_size = int(os.getenv("DEEPSEEK_SPARK_MAX_QUEUE_SIZE", "8"))
    app.state.pending_requests = 0
    app.state.queue_lock = asyncio.Lock()
    app.state.background_tasks = set()
    app.state.postfill_delay_seconds = float(
        os.getenv("DEEPSEEK_SPARK_POSTFILL_DELAY_SECONDS", "0.25")
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "queue": {
                "pending": app.state.pending_requests,
                "max_size": app.state.max_queue_size,
            },
            **await _call_engine_method(app.state.engine, "health"),
        }

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        return {"ready": app.state.engine.loaded}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/spark/sessions/{session_id}/reset")
    async def reset_session(session_id: str) -> dict[str, Any]:
        await _call_engine_method(app.state.engine, "reset_session", session_id)
        return {"ok": True, "session_id": session_id}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        request: Request,
    ):
        messages = _encoding_messages(req)
        async with _queue_slot(app):
            result = await _generate_or_500(
                app.state.engine.generate(
                    messages=messages,
                    max_tokens=req.max_tokens,
                    session_id=req.spark_session_id,
                    cache_policy=req.spark_cache_policy,
                    thinking_mode=req.spark_thinking_mode,
                    reasoning_effort=req.spark_reasoning_effort,
                )
            )
        _schedule_deferred_postfill(app, result)
        if req.stream:
            return _stream_chat_response(result, request)
        return _chat_response(req.model, result)

    @app.post("/v1/completions")
    async def completions(
        req: CompletionRequest,
        request: Request,
    ):
        async with _queue_slot(app):
            result = await _generate_or_500(
                app.state.engine.generate(
                    prompt=req.prompt,
                    max_tokens=req.max_tokens,
                    session_id=req.spark_session_id,
                    cache_policy=req.spark_cache_policy,
                )
            )
        _schedule_deferred_postfill(app, result)
        if req.stream:
            return _stream_completion_response(result, request)
        return _completion_response(req.model, result)

    return app


def _encoding_messages(req: ChatCompletionRequest) -> list[dict[str, Any]]:
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    request_tools = [tool.model_dump(exclude_none=True) for tool in req.tools or []]
    request_response_format = (
        req.response_format.model_dump(exclude_none=True) if req.response_format else None
    )
    if not request_tools and request_response_format is None:
        return messages

    target_index = next(
        (idx for idx, message in enumerate(messages) if message.get("role") in {"system", "developer"}),
        None,
    )
    if target_index is None:
        messages.insert(0, {"role": "system", "content": ""})
        target_index = 0

    target = messages[target_index]
    if request_tools:
        target["tools"] = [*target.get("tools", []), *request_tools]
    if request_response_format is not None and "response_format" not in target:
        target["response_format"] = request_response_format
    return messages


def _schedule_deferred_postfill(app: FastAPI, result) -> None:
    session_id = getattr(result, "deferred_postfill_session_id", None)
    token_ids = getattr(result, "deferred_postfill_token_ids", None)
    if not session_id or not token_ids:
        return

    task = asyncio.create_task(_run_deferred_postfill(app, session_id, token_ids))
    app.state.background_tasks.add(task)
    task.add_done_callback(app.state.background_tasks.discard)


async def _run_deferred_postfill(app: FastAPI, session_id: str, token_ids: list[int]) -> None:
    await asyncio.sleep(app.state.postfill_delay_seconds)
    await _call_engine_method(app.state.engine, "prepare_next_logits", session_id, token_ids)


async def _call_engine_method(engine, method_name: str, *args, **kwargs):
    result = getattr(engine, method_name)(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


@asynccontextmanager
async def _queue_slot(app: FastAPI):
    async with app.state.queue_lock:
        if app.state.pending_requests >= app.state.max_queue_size:
            raise HTTPException(status_code=429, detail="generation queue is full")
        app.state.pending_requests += 1
    try:
        yield
    finally:
        async with app.state.queue_lock:
            app.state.pending_requests -= 1


async def _generate_or_500(coro):
    try:
        return await coro
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc


def _usage(result) -> dict[str, int]:
    return {
        "prompt_tokens": len(result.prompt_token_ids),
        "completion_tokens": len(result.generated_token_ids),
        "total_tokens": len(result.prompt_token_ids) + len(result.generated_token_ids),
    }


def _chat_response(model: str, result) -> JSONResponse:
    now = int(time.time())
    message = _assistant_message(result)
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": _usage(result),
            "spark_metrics": _metrics(result),
        }
    )


def _assistant_message(result) -> dict[str, Any]:
    if result.parsed_message is not None:
        return {
            key: value
            for key, value in result.parsed_message.items()
            if value not in (None, [], "")
        }
    return {"role": "assistant", "content": result.generated_text}


def _completion_response(model: str, result) -> JSONResponse:
    now = int(time.time())
    return JSONResponse(
        {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": now,
            "model": model,
            "choices": [
                {"index": 0, "text": result.generated_text, "finish_reason": result.finish_reason}
            ],
            "usage": _usage(result),
            "spark_metrics": _metrics(result),
        }
    )


def _metrics(result) -> dict[str, Any]:
    return {
        "reused_prefix_tokens": result.reused_prefix_tokens,
        "newly_prefilled_tokens": result.newly_prefilled_tokens,
        "generated_token_ids": result.generated_token_ids,
        "full_text": result.full_text,
        "finish_reason": result.finish_reason,
        "deferred_postfill_scheduled": bool(
            getattr(result, "deferred_postfill_token_ids", None)
        ),
        "timings": result.timings,
        "memory": {key: list(value) for key, value in result.memory.items()},
    }


def _stream_chat_response(result, request: Request) -> StreamingResponse:
    async def events():
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        for token_id in result.generated_token_ids:
            if await request.is_disconnected():
                break
            content = "" if token_id is None else request.app.state.engine.tokenizer.decode([token_id])
            yield (
                "data: "
                + _json_dumps(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                    }
                )
                + "\n\n"
            )
        yield (
            "data: "
            + _json_dumps(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": result.finish_reason}],
                    }
                )
            + "\n\n"
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def _stream_completion_response(result, request: Request) -> StreamingResponse:
    async def events():
        chunk_id = f"cmpl-{uuid.uuid4().hex}"
        for token_id in result.generated_token_ids:
            if await request.is_disconnected():
                break
            text = request.app.state.engine.tokenizer.decode([token_id])
            yield (
                "data: "
                + _json_dumps(
                    {
                        "id": chunk_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "text": text, "finish_reason": None}],
                    }
                )
                + "\n\n"
            )
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, separators=(",", ":"))


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("spark_runtime.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
