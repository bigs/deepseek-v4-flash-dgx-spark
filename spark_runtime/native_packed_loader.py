"""Lazy loader for the native packed expert read extension."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "0").lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def load_native_packed_loader():
    """Compile and load the packed expert reader extension.

    The extension is intentionally loaded only when requested. It depends on the
    local PyTorch C++ extension toolchain, which is available in the Spark
    runtime container but should not be required for ordinary local unit tests.
    """

    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("native").joinpath("native_packed_loader.cpp")
    extra_cflags = ["-O3", "-std=c++17"]
    if os.getenv("DEEPSEEK_SPARK_NATIVE_EXTRA_CFLAGS"):
        extra_cflags.extend(os.getenv("DEEPSEEK_SPARK_NATIVE_EXTRA_CFLAGS", "").split())
    build_directory_value = os.getenv("DEEPSEEK_SPARK_NATIVE_BUILD_DIR")
    build_directory = Path(build_directory_value) if build_directory_value else None
    if build_directory is not None:
        build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name="deepseek_spark_native_packed_loader",
        sources=[str(source)],
        extra_cflags=extra_cflags,
        build_directory=str(build_directory) if build_directory is not None else None,
        verbose=_truthy_env("DEEPSEEK_SPARK_NATIVE_VERBOSE"),
    )
