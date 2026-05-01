#!/usr/bin/env python3
"""Ask Linux to drop page-cache entries for model files using posix_fadvise."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()

    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise SystemExit("posix_fadvise/POSIX_FADV_DONTNEED is not available")

    model_dir = args.model_dir.expanduser().resolve()
    files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    total = 0
    for path in files:
        size = path.stat().st_size
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        total += size
    print(f"advised_dontneed_files={len(files)} bytes={total}")


if __name__ == "__main__":
    main()
