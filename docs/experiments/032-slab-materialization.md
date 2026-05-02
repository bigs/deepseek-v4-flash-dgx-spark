# Experiment 032: Reusable Slab Materialization

Date: 2026-05-02

Host: `spark-66c9`

Goal: test the flash-moe-inspired idea that reusable staging buffers beat per-miss
bytes/bytearray allocation.

## Change

Added `scripts/bench_packed_slab_materialization.py`.

The benchmark selects 64 packed expert blocks from the route trace and compares:

- `current_bytes`: `os.pread` -> `bytes` -> `bytearray` -> `torch.frombuffer` -> CUDA.
- `preadv_bytearray_slab`: reusable `bytearray` slab filled with `os.preadv` ->
  `torch.frombuffer` -> CUDA.

Also added an opt-in runtime path:

```bash
DEEPSEEK_SPARK_PACKED_REUSE_STAGING=1
```

## Artifacts

- `results/spark-66c9/slab-mat-e032.log`
- `results/spark-66c9/slab-mat-e032.json`

## Results

| Method | Median elapsed | Throughput |
| --- | ---: | ---: |
| current bytes/bytearray | 0.176s | 4.54 GiB/s |
| reusable `preadv` bytearray slab | 0.112s | 7.10 GiB/s |

Both methods read and transferred 64 blocks, 855,638,016 bytes total.

## Interpretation

The microbenchmark strongly supports reusable staging. The full-runtime effect is much
smaller because tensor reconstruction and parameter copies still dominate, but this is a
good primitive to keep. It is now integrated behind an environment flag and measured in
E036.
