#!/usr/bin/env python3
"""Search installed runtime packages for DeepSeek-V4-related code."""

from __future__ import annotations

import importlib
import os
from pathlib import Path


NEEDLES = [
    "DeepseekV4",
    "DeepSeekV4",
    "deepseek_v4",
    "DeepseekV3",
    "DeepSeek",
    "mxfp4",
    "nvfp4",
]


def scan_package(name: str) -> None:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        print(f"== {name}: missing {exc!r} ==")
        return

    base = Path(mod.__file__).resolve().parent
    print(f"== {name}: {base} ==")
    hits = 0
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git"}]
        for filename in files:
            if not filename.endswith((".py", ".json", ".toml", ".yaml", ".yml")):
                continue
            path = Path(root) / filename
            try:
                text = path.read_text(errors="ignore")
            except Exception:
                continue
            matched = [needle for needle in NEEDLES if needle in text]
            if matched:
                print(path, ",".join(matched))
                hits += 1
    print(f"hits={hits}")


def main() -> None:
    for name in ["sglang", "vllm", "transformers"]:
        scan_package(name)


if __name__ == "__main__":
    main()
