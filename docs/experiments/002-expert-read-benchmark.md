# Experiment 002: Expert Read Benchmark

Goal: measure whether the Hugging Face safetensor layout is usable for
expert-local reads on DGX Spark, before building a converted runtime layout.

This experiment uses `results/spark-66c9/checkpoint-tensors.csv` from Experiment
001. It reads tensor payload ranges directly from the safetensor shard files and
compares:

- top-k routed expert subsets
- expert sets that are numerically adjacent
- expert sets that are adjacent in file order
- spread expert sets
- whole-layer routed expert payload reads

The script reports useful bytes, planned read bytes, read amplification after
range coalescing, elapsed time, throughput, and major/minor page faults.

Command:

```bash
python3 scripts/bench_expert_reads.py \
  --model-dir ~/models/deepseek-v4-flash/hf \
  --tensor-csv results/spark-66c9/checkpoint-tensors.csv \
  --out-dir results/spark-66c9 \
  --layers 0,18,42 \
  --methods pread,mmap \
  --chunk-size-mib 8 \
  --repeat 2 \
  --max-gaps 0,1048576
```

Cold-ish per-iteration run:

```bash
python3 scripts/bench_expert_reads.py \
  --model-dir ~/models/deepseek-v4-flash/hf \
  --tensor-csv results/spark-66c9/checkpoint-tensors.csv \
  --out-dir results/spark-66c9 \
  --layers 0,18,42 \
  --methods pread,mmap \
  --chunk-size-mib 8 \
  --repeat 1 \
  --max-gaps 0,1048576 \
  --evict-mode iteration \
  --output-prefix expert-read-benchmark-evict-iteration
```

Expected outputs:

- `results/spark-66c9/expert-read-benchmark.json`
- `results/spark-66c9/expert-read-benchmark.md`
- `results/spark-66c9/expert-read-benchmark-evict-iteration.json`
- `results/spark-66c9/expert-read-benchmark-evict-iteration.md`

Interpretation rules:

- If top-k exact or zero-gap coalesced reads are fast enough and have low read
  amplification, the Hugging Face layout is good enough for the first expert
  cache prototype.
- If top-k reads are slow mostly because they require many small discontiguous
  reads, try a converted expert-local layout before writing complex kernels.
- If whole-layer reads are much faster per GiB but require `3.19 GiB` per layer,
  the first runtime cache may need layer-granular prefetching as a baseline.
- If repeated mmap runs become dramatically faster, page-cache residency is doing
  useful work and should be treated as an explicit part of the runtime design.

## Result

Runs completed on `spark-66c9` at `2026-05-01T17:48:48Z` and
`2026-05-01T17:50:16Z`.

The first run used normal page-cache behavior. `/usr/bin/time` reported
`File system inputs: 0`, so those numbers are warm-cache/layout overhead rather
than cold NVMe bandwidth.

The second run used `--evict-mode iteration`, which calls
`posix_fadvise(..., POSIX_FADV_DONTNEED)` before each measured iteration when
available. `/usr/bin/time` reported `File system inputs: 49110512`, so this run
did force real file input for a substantial part of the benchmark without root.

High-level findings:

- A top-6 expert set is `76.50 MiB`: 36 tensors covering 6 experts times
  `w1/w2/w3` weight and scale.
- File-adjacent top-6 expert sets coalesce to only 2 read ranges.
- Numeric `0..5` expert sets coalesce to 10 read ranges because safetensor order
  is lexicographic (`0, 1, 10, 100, ...`) rather than numeric.
- Spread expert sets coalesce to 12 read ranges.
- Read amplification stayed near `1.00x`; the Hugging Face layout does not force
  large unrelated reads for these top-k expert selections.
- Whole-layer routed expert reads are `3.19 GiB` and coalesce to 2 ranges.

Representative warm-cache medians:

- Layer 0 top-6 numeric `pread`: `3.34 GiB/s`; `mmap`: `3.63 GiB/s`.
- Layer 18 top-6 numeric `pread`: `3.58 GiB/s`; `mmap`: `2.82 GiB/s`.
- Layer 42 top-6 numeric `pread`: `4.48 GiB/s`; `mmap`: `4.71 GiB/s`.
- Whole-layer `pread`: `10.31-12.89 GiB/s`.
- Whole-layer `mmap`: `7.13-15.20 GiB/s`.

Representative `posix_fadvise` eviction-run results:

- Layer 0 top-6 numeric `pread`: `1.77 GiB/s`; `mmap`: `2.52 GiB/s`.
- Layer 18 top-6 numeric `pread`: `2.69 GiB/s`; `mmap`: `2.63 GiB/s`.
- Layer 42 top-6 numeric `pread`: `3.11 GiB/s`; `mmap`: `2.93 GiB/s`.
- Whole-layer `pread`: `5.73-8.08 GiB/s`.
- Whole-layer `mmap`: `5.08-5.40 GiB/s`.

Initial conclusion:

The Hugging Face layout is good enough for the first expert-cache prototype. We
do not need to start with a converted on-disk format just to avoid pathological
read amplification. A converted layout may still become useful later for kernel
tile order, prefetch scheduling, or avoiding lexicographic expert ordering, but
the next step should be a real expert cache/loader primitive over the current
safetensor offsets.
