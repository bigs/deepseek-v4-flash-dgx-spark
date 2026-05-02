# Experiment 037: Native Packed Loader

Date: 2026-05-02

Host: `spark-66c9`

Goal: move packed expert block reads from Python byte allocation into a native reader that
fills a reusable CPU `torch.uint8` tensor directly.

## Change

Added a lazy PyTorch C++ extension:

- `spark_runtime/native_packed_loader.py`
- `spark_runtime/native/native_packed_loader.cpp`

The extension exposes `read_into(fd, destination, size, offset)`, releases the GIL, and
loops around `pread` until the requested packed expert block has been read or EOF is hit.

Runtime integration is opt-in:

```bash
DEEPSEEK_SPARK_PACKED_NATIVE_LOADER=1
DEEPSEEK_SPARK_NATIVE_BUILD_DIR=/runs/torch-extensions/native-packed-loader
```

The benchmark script now supports:

```bash
scripts/bench_packed_slab_materialization.py --include-native
```

## Artifacts

- `results/spark-66c9/native-loader-e037.log`
- `results/spark-66c9/native-loader-e037.json`

## Results

64 route-filtered packed blocks, 855,638,016 bytes total, 5 repeats.

| Method | Median elapsed | Throughput |
| --- | ---: | ---: |
| Current bytes/bytearray | 0.636s | 1.25 GiB/s |
| Reusable `preadv` bytearray slab | 0.102s | 7.82 GiB/s |
| Native tensor reader | 0.100s | 7.96 GiB/s |

## Interpretation

The native tensor reader compiled and ran successfully in the Spark runtime container. It
is slightly faster than reusable `preadv` in this microbenchmark and much faster than the
old Python bytes path. The main value is that it gives us a native staging tensor that the
runtime can reuse without constructing Python byte objects.
