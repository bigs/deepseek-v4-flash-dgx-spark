# Experiment 011: Routed Expert Cache

Date: 2026-05-01

Host: `spark-66c9`

Goal: test whether keeping materialized routed experts resident reduces the repeated
host-to-device transfer bottleneck observed in Experiment 010.

## Change

Added an optional budgeted routed-expert cache controlled by:

```bash
DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES=N
```

Default remains `0`, which preserves the old immediate-eviction behavior. When enabled,
`LazyRoutedExpert` keeps materialized expert modules alive until a global LRU entry budget
is exceeded. Runtime telemetry now records:

- cache hits, misses, inserts, and evictions;
- resident expert bytes;
- materialization seconds;
- copied expert bytes;
- routed calls, routed tokens, and routed activations.

## Commands

The runs used the same Nsight Systems harness and prompt shape as Experiment 010.

Small cache:

```bash
python3 scripts/run_nsys_profile.py \
  --name nsys-cache64-hello \
  --manifest-csv /repo/results/spark-66c9/weight-manifest.csv \
  --prompt Hello \
  --max-tokens 2 \
  --expert-cache-entries 64 \
  --wait-postfill-before-cached \
  --timeout-seconds 2400
```

Larger cache:

```bash
python3 scripts/run_nsys_profile.py \
  --name nsys-cache1024-hello \
  --manifest-csv /repo/results/spark-66c9/weight-manifest.csv \
  --prompt Hello \
  --max-tokens 2 \
  --expert-cache-entries 1024 \
  --wait-postfill-before-cached \
  --timeout-seconds 2400
```

## Artifacts

Baseline from Experiment 010:

- `results/spark-66c9/nsys-hello-summary.md`
- `results/spark-66c9/nsys-hello.jsonl`
- `results/spark-66c9/nsys-hello.nsys-rep`
- `results/spark-66c9/nsys-hello.sqlite`

Cache runs:

- `results/spark-66c9/nsys-cache64-hello-summary.md`
- `results/spark-66c9/nsys-cache64-hello.jsonl`
- `results/spark-66c9/nsys-cache64-hello.nsys-rep`
- `results/spark-66c9/nsys-cache64-hello.sqlite`
- `results/spark-66c9/nsys-cache1024-hello-summary.md`
- `results/spark-66c9/nsys-cache1024-hello.jsonl`
- `results/spark-66c9/nsys-cache1024-hello.nsys-rep`
- `results/spark-66c9/nsys-cache1024-hello.sqlite`

## Results

| Run | Cache Entries | First Completion Wall | Prefill | Decode | HtoD GiB | HtoD GPU Time | Cache Hits | Misses | Evictions | Resident Expert GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nsys-hello` | 0 | 225.710s | 144.948s | 28.228s | 16.622 | 92.589s | n/a | n/a | n/a | 0.000 |
| `nsys-cache64-hello` | 64 | 225.722s | 145.455s | 28.397s | 16.622 | 91.298s | 0 | 774 | 710 | 0.797 |
| `nsys-cache1024-hello` | 1024 | 228.994s | 144.576s | 28.583s | 16.473 | 95.267s | 124 | 650 | 0 | 8.093 |

The 1024-entry run produced hits and avoided evictions, but this two-token probe did not
convert those hits into wall-time improvement. It only reduced total HtoD volume by about
0.149 GiB, and measured HtoD GPU time was worse due run-to-run noise or allocator effects.

The important telemetry detail is the reuse distance:

- 64 entries: 516 misses and 452 evictions during the first completion; 0 hits.
- 1024 entries: 495 misses and 21 hits during the first completion; 124 cumulative hits
  after deferred postfill.

This shows that a tiny global LRU cache is the wrong shape. A cache large enough to cover
the observed short-run reuse needed around 8.1 GiB for this prompt and still only helped
modestly because the benchmark mostly measures first-use expert materialization.

## Interpretation

This experiment partially falsifies the simplest cache theory. Keeping experts resident
does work mechanically, and it gives real hits when the budget is large enough, but the
two-token benchmark is dominated by first-use copies:

- the first prompt token must materialize its routed experts;
- the first decode step mostly uses experts that are still cold;
- the second generated token is sampled from logits and does not trigger another model
  forward because `max_tokens=2`;
- cached continuation reuses persistent KV/next-logits and therefore does not exercise
  routed experts at all.

So the cache is not the first big speedup by itself. It is still useful as telemetry and as
a building block for longer decode/session-reuse experiments, but it does not replace the
Flash-MoE layout lesson: we need to reduce first-use materialization cost.

## Next Step

Run a longer decode cache experiment before discarding expert caching. The next cache test
should use at least 8 to 16 generated tokens and report per-step cache hits/misses. If
longer decode still does not improve, the cache should stay small and the priority should
move to packed expert layout and fewer/larger HtoD transfers.

