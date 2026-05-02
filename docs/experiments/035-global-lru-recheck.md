# Experiment 035: Global LRU Recheck

Date: 2026-05-02

Host: `spark-66c9`

Goal: test a replay-suggested cache policy candidate in full inference. Replay showed
global LRU had fewer misses than layer-LRU on the 8-token trace.

## Command

The run used the route-filtered packed layout, 1024 expert cache entries, global LRU, and
the allocator-cache fix.

## Artifacts

- `results/spark-66c9/global-lru-noempty-e035.log`
- `results/spark-66c9/global-lru-noempty-e035.jsonl`

## Results

| Scenario | Prefill | Decode | Tok/s | Packed read | Materialize | Misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E026 layer-LRU | 91.662s | 6.434s | 1.24 | 2.157s | 4.322s | 1,151 |
| E035 global LRU | 95.686s | 20.154s | 0.40 | 12.118s | 18.581s | 1,147 |

## Interpretation

Global LRU is rejected for the current runtime. It has slightly fewer misses, but far worse
materialization/read behavior in the full model. The likely issue is whole-system memory
and allocator behavior, not simple miss count.

Keep `layer_lru` as the default recommendation.
