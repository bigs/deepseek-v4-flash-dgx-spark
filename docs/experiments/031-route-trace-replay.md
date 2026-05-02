# Experiment 031: Route-Trace Packed-Read Replay

Date: 2026-05-02

Host: `spark-66c9`

Goal: add a fast flash-moe-style replay harness for packed expert reads and cache policy
screening.

## Change

Added `scripts/replay_packed_expert_trace.py`.

The script replays `expert_route` JSONL events against a packed layout, applies a cache
policy, and reads missed packed expert blocks. It does not run model math.

## Artifacts

- `results/spark-66c9/replay-layer-lru-route-e031.json`
- `results/spark-66c9/replay-layer-lru-route-warm-e031.json`
- `results/spark-66c9/replay-layer-lru-offset-e031.json`
- `results/spark-66c9/replay-layer-lru-threaded-e031.json`
- `results/spark-66c9/replay-global-lru-route-e031.json`
- `results/spark-66c9/replay-none-route-e031.json`

## Results

All runs used the full packed layout and `route-cache1024-t8-routes.jsonl`.

| Scenario | Elapsed | Throughput | Hits | Misses | Read blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| layer-LRU route order, first run | 4.840s | 3.21 GiB/s | 1,074 | 1,248 | 1,248 |
| layer-LRU route order, warm | 1.303s | 11.92 GiB/s | 1,074 | 1,248 | 1,248 |
| layer-LRU offset order | 1.333s | 11.65 GiB/s | 1,074 | 1,248 | 1,248 |
| layer-LRU threaded reads | 1.179s | 13.18 GiB/s | 1,074 | 1,248 | 1,248 |
| global LRU route order | 1.348s | 11.43 GiB/s | 1,085 | 1,237 | 1,237 |
| no cache | 2.674s | 10.81 GiB/s | 0 | 2,322 | 2,322 |

## Interpretation

The replay harness is useful, but it is a screening tool, not a full-performance oracle.
It predicted global LRU would have fewer misses than layer-LRU, which is true for this
trace, but E035 showed global LRU was much worse in full inference.

The reusable lessons:

- threaded read-only replay can move bytes faster when warm;
- offset sorting did not matter for the current full layout and trace;
- full inference remains necessary before accepting policy changes;
- replay is still the right first gate for layout and I/O variants.
