# Experiment 023: Pinned Staging

Date: 2026-05-01

Host: `spark-66c9`

Goal: test pinned host staging for packed expert blocks.

## Change

Added optional packed-read staging behind:

```bash
DEEPSEEK_SPARK_PACKED_PINNED_STAGING=1
```

When enabled, the packed store allocates a CPU pinned `torch.uint8` tensor for each
expert block and reads into it with `os.preadv`.

## Validation

The pinned path reconstructed layer 0 expert 168 correctly. All six tensors matched
safetensors exactly:

- `w1.scale`
- `w1.weight`
- `w2.scale`
- `w2.weight`
- `w3.scale`
- `w3.weight`

## Result

This implementation is a regression. A single `13,369,344` byte packed expert block read
took about `0.340s` with per-miss pinned allocation. Earlier pageable packed validation
reads for the same block size were around `0.011-0.016s`.

## Interpretation

Per-miss pinned allocation is too expensive. Pinned staging remains worth revisiting only
with a persistent buffer pool, ideally paired with asynchronous copies or custom kernels.
The runtime should keep `DEEPSEEK_SPARK_PACKED_PINNED_STAGING` off for now.

