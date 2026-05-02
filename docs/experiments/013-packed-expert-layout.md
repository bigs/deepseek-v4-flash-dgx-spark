# Experiment 013: Packed Expert Layout Microbenchmark

Date: 2026-05-01

Host: `spark-66c9`

Goal: test the Flash-MoE-inspired fixed-offset packed expert layout against the current
Hugging Face safetensor layout for routed expert reads.

## Change

Added `scripts/bench_packed_expert_layout.py`.

The script:

1. Reads the current weight manifest.
2. Reads the route trace from Experiment 012.
3. Selects top routed experts for chosen layers.
4. Builds small fixed-offset packed files for only those selected experts.
5. Benchmarks three read plans:
   - `hf_exact`: read each source tensor range individually from the HF shard.
   - `hf_coalesced`: coalesce adjacent HF source ranges before reading.
   - `packed`: read one fixed-offset expert block per selected expert.

This is a microbench, not a runtime path. It measures whether the packed layout is worth
building before we wire it into inference.

## Commands

Cold-ish run with page-cache eviction hints:

```bash
python3 scripts/bench_packed_expert_layout.py \
  --model-dir ~/models/deepseek-v4-flash/hf \
  --manifest-csv results/spark-66c9/weight-manifest.csv \
  --route-jsonl ~/runs/deepseek-v4-flash/route-cache1024-t8-routes.jsonl \
  --out-dir ~/runs/deepseek-v4-flash/packed-layout-exp013 \
  --layers 15,18,39 \
  --top-per-layer 16 \
  --repeat 3 \
  --evict
```

Warm-cache run:

```bash
python3 scripts/bench_packed_expert_layout.py \
  --model-dir ~/models/deepseek-v4-flash/hf \
  --manifest-csv results/spark-66c9/weight-manifest.csv \
  --route-jsonl ~/runs/deepseek-v4-flash/route-cache1024-t8-routes.jsonl \
  --out-dir ~/runs/deepseek-v4-flash/packed-layout-exp013-warm \
  --layers 15,18,39 \
  --top-per-layer 16 \
  --repeat 3
```

## Artifacts

- `results/spark-66c9/packed-layout-exp013/packed-expert-layout-benchmark.json`
- `results/spark-66c9/packed-layout-exp013/packed-expert-layout-benchmark.md`
- `results/spark-66c9/packed-layout-exp013-warm/packed-expert-layout-benchmark.json`
- `results/spark-66c9/packed-layout-exp013-warm/packed-expert-layout-benchmark.md`

## Results

Each scenario reads the top 16 experts for one layer, totaling `204.0 MiB`.

Eviction run:

| Scenario | Ranges | Median s | Median GiB/s |
| --- | ---: | ---: | ---: |
| `layer15.hf_exact` | 96 | 0.0671 | 2.97 |
| `layer15.hf_coalesced` | 26 | 0.1225 | 1.63 |
| `layer15.packed` | 16 | 0.0666 | 2.99 |
| `layer18.hf_exact` | 96 | 0.1536 | 1.30 |
| `layer18.hf_coalesced` | 30 | 0.1029 | 1.94 |
| `layer18.packed` | 16 | 0.0659 | 3.02 |
| `layer39.hf_exact` | 96 | 0.1354 | 1.47 |
| `layer39.hf_coalesced` | 28 | 0.1309 | 1.52 |
| `layer39.packed` | 16 | 0.0651 | 3.06 |

Warm-cache run:

| Scenario | Ranges | Median s | Median GiB/s |
| --- | ---: | ---: | ---: |
| `layer15.hf_exact` | 96 | 0.0161 | 12.40 |
| `layer15.hf_coalesced` | 26 | 0.0505 | 3.94 |
| `layer15.packed` | 16 | 0.0146 | 13.65 |
| `layer18.hf_exact` | 96 | 0.0157 | 12.69 |
| `layer18.hf_coalesced` | 30 | 0.0170 | 11.73 |
| `layer18.packed` | 16 | 0.0146 | 13.60 |
| `layer39.hf_exact` | 96 | 0.0167 | 11.92 |
| `layer39.hf_coalesced` | 28 | 0.0505 | 3.95 |
| `layer39.packed` | 16 | 0.0146 | 13.65 |

## Interpretation

The packed layout wins this microbench.

The win is not just fewer ranges. The current Hugging Face layout is already reasonably
layer-local, but fixed expert blocks make the read plan simpler and more predictable:

- 16 packed reads instead of 96 tensor reads;
- stable one-block-per-expert offsets;
- no safetensor key lookup or tensor metadata on the hot path;
- no dependence on lexicographic expert order in the source shard.

The eviction run is the most important signal. Packed reads were:

- essentially tied with HF exact on layer 15;
- `2.3x` faster than HF exact on layer 18;
- `2.1x` faster than HF exact on layer 39.

The warm-cache run also favors packed reads, though the gap is smaller because Linux page
cache hides much of the storage cost.

Coalescing adjacent HF ranges was inconsistent. It helped layer 18 in the eviction run but
hurt layer 15 and warm-cache layers 15 and 39. The packed layout is the simpler and more
stable plan.

## Next Step

Build a packed expert runtime path for misses:

1. Repack all routed experts, or start with a layer subset for correctness.
2. Read one packed expert block into host staging.
3. Materialize the official expert parameters from that block without safetensor lookup.
4. Measure whether first-use materialization time and HtoD copy fragmentation improve.

Packed layout should be combined with the expert cache. The cache reduces repeated
materialization; packed layout makes unavoidable misses cheaper.

