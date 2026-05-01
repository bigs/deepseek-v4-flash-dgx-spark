#!/usr/bin/env python3
"""Start the server and run one chat request until DeepSeek emits EOS."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request


def request_json(method: str, url: str, payload: dict | None = None, timeout: float = 10.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method)
    if payload is not None:
        req.add_header("content-type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait_ready(base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            request_json("GET", f"{base_url}/health", timeout=2.0)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("server did not become reachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--message", default="Hello, how are you?")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
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
        payload = {
            "model": "deepseek-v4-flash-dgx-spark",
            "messages": [{"role": "user", "content": args.message}],
            "max_tokens": args.max_tokens,
            "spark_session_id": "chat-until-stop",
            "spark_cache_policy": "reset",
            "spark_thinking_mode": "chat",
        }
        start = time.monotonic()
        response = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            payload,
            timeout=args.request_timeout,
        )
        elapsed = time.monotonic() - start
        print("chat_until_stop", {"elapsed_seconds": elapsed, "response": response})
        finish_reason = response["choices"][0]["finish_reason"]
        if finish_reason != "stop":
            raise RuntimeError(f"expected finish_reason='stop', got {finish_reason!r}")
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
