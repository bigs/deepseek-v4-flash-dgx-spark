#!/usr/bin/env python3
"""Start the MVP server locally and verify OpenAI-compatible HTTP inference."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def request_json(method: str, url: str, payload: dict | None = None, timeout: float = 10.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method)
    if payload is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc


def wait_ready(base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            request_json("GET", f"{base_url}/health", timeout=2.0)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("server did not become reachable")


def wait_no_pending_postfill(base_url: str, timeout_seconds: float) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_health = {}
    stats = {"polls": 0, "poll_errors": 0, "max_poll_seconds": 0.0}
    while time.monotonic() < deadline:
        try:
            poll_start = time.monotonic()
            last_health = request_json("GET", f"{base_url}/health", timeout=5.0)
            poll_seconds = time.monotonic() - poll_start
            stats["polls"] += 1
            stats["max_poll_seconds"] = max(stats["max_poll_seconds"], poll_seconds)
            if not last_health.get("pending_postfill_sessions"):
                last_health["postfill_poll_stats"] = stats
                return last_health
        except Exception as exc:
            stats["poll_errors"] += 1
            last_health = {"poll_error": repr(exc)}
        time.sleep(0.5)
    raise TimeoutError(f"postfill did not finish; last health={last_health}")


def timed_request(method: str, url: str, payload: dict | None = None, timeout: float = 10.0):
    start = time.monotonic()
    body = request_json(method, url, payload, timeout)
    return body, time.monotonic() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--wait-postfill-before-cached", action="store_true")
    parser.add_argument("--postfill-timeout", type=float, default=300.0)
    parser.add_argument("--skip-chat", action="store_true")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "spark_runtime.server:app",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--log-level",
            "info",
        ],
        env=os.environ.copy(),
    )
    try:
        wait_ready(base_url, args.startup_timeout)
        models, models_seconds = timed_request("GET", f"{base_url}/v1/models")
        print("models", {"elapsed_seconds": models_seconds, "response": models})
        completion, completion_seconds = timed_request(
            "POST",
            f"{base_url}/v1/completions",
            {
                "model": "deepseek-v4-flash-dgx-spark",
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "spark_session_id": "http-probe",
                "spark_cache_policy": "reset",
            },
            timeout=3600,
        )
        print("completion", {"elapsed_seconds": completion_seconds, "response": completion})
        if completion["usage"]["completion_tokens"] != args.max_tokens:
            raise RuntimeError("unexpected completion token count")
        if args.wait_postfill_before_cached:
            postfill_start = time.monotonic()
            health = wait_no_pending_postfill(base_url, args.postfill_timeout)
            print(
                "postfill_wait",
                {"elapsed_seconds": time.monotonic() - postfill_start, "health": health},
            )

        continued_prompt = completion["spark_metrics"]["full_text"]
        cached_completion, cached_seconds = timed_request(
            "POST",
            f"{base_url}/v1/completions",
            {
                "model": "deepseek-v4-flash-dgx-spark",
                "prompt": continued_prompt,
                "max_tokens": 1,
                "spark_session_id": "http-probe",
                "spark_cache_policy": "reuse",
            },
            timeout=3600,
        )
        print(
            "cached_completion",
            {"elapsed_seconds": cached_seconds, "response": cached_completion},
        )
        reused_tokens = cached_completion["spark_metrics"]["reused_prefix_tokens"]
        if reused_tokens <= 0:
            raise RuntimeError("session cache was not reused")

        if args.skip_chat:
            return 0

        chat, chat_seconds = timed_request(
            "POST",
            f"{base_url}/v1/chat/completions",
            {
                "model": "deepseek-v4-flash-dgx-spark",
                "messages": [{"role": "user", "content": args.prompt}],
                "max_tokens": 1,
                "spark_session_id": "http-chat-probe",
                "spark_cache_policy": "reset",
            },
            timeout=3600,
        )
        print("chat", {"elapsed_seconds": chat_seconds, "response": chat})
        if chat["object"] != "chat.completion":
            raise RuntimeError("unexpected chat response object")
        return 0
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


if __name__ == "__main__":
    sys.exit(main())
