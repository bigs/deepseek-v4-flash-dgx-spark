# Experiment 015: Parallel Packed Reads

Date: 2026-05-01

Host: `spark-66c9`

Goal: test the deterministic I/O scheduling piece from Flash-MoE: once experts are in
fixed-offset packed files, does parallel read dispatch improve top-k expert fetches?

## Command

This used the packed files generated in Experiment 013 and compared serial reads with
4-thread and 8-thread `ThreadPoolExecutor` reads. Each layer scenario reads 16 packed
expert blocks, `204 MiB` total. Before each sample, the script used
`posix_fadvise(..., POSIX_FADV_DONTNEED)` on the packed file.

The raw one-off command is preserved in shell history; the output artifact is:

- `results/spark-66c9/parallel-packed-exp015.json`

## Results

| Layer | Mode | Median s | Median GiB/s | Speedup vs Serial |
| ---: | --- | ---: | ---: | ---: |
| 15 | serial | 0.0255 | 7.81 | 1.00x |
| 15 | parallel4 | 0.0217 | 9.19 | 1.18x |
| 15 | parallel8 | 0.0200 | 9.96 | 1.27x |
| 18 | serial | 0.0259 | 7.70 | 1.00x |
| 18 | parallel4 | 0.0206 | 9.66 | 1.25x |
| 18 | parallel8 | 0.0196 | 10.19 | 1.32x |
| 39 | serial | 0.0277 | 7.20 | 1.00x |
| 39 | parallel4 | 0.0208 | 9.60 | 1.33x |
| 39 | parallel8 | 0.0194 | 10.24 | 1.42x |

## Interpretation

Parallel packed reads help, but the gain is modest in this small benchmark:
approximately `1.2x` to `1.4x` for 16 experts / 204 MiB.

This result still supports a persistent I/O pool in the runtime because:

- it reduces cold-ish read wall time;
- it avoids per-request thread creation if implemented as a persistent pool;
- it composes naturally with one-block-per-expert packed layout;
- it is deterministic and does not risk prediction bandwidth waste.

The effect is much smaller than the packed materialization result in Experiment 014.
Parallel I/O is an incremental optimization; packed miss format is the larger win.

## Next Step

Combine packed expert files, a persistent I/O pool, and pinned host staging in one miss
loader. Prediction should still wait until this deterministic path is integrated.

