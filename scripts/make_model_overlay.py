#!/usr/bin/env python3
"""Create a symlink overlay of a Hugging Face model with patched JSON files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Patch an additional config key. Value is parsed as JSON.",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    for path in source.iterdir():
        target = dest / path.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if path.name == "config.json":
            config = json.loads(path.read_text())
            config["model_type"] = args.model_type
            config["architectures"] = [args.architecture]
            for item in args.set:
                key, raw_value = item.split("=", 1)
                config[key] = json.loads(raw_value)
            target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        else:
            os.symlink(path, target)

    print(f"created overlay {dest}")
    print(f"model_type={args.model_type} architecture={args.architecture}")


if __name__ == "__main__":
    main()
