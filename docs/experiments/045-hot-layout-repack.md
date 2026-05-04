# Experiment 045: Hot-First Full Layout Repack

Date: 2026-05-03

Host: `spark-66c9`

Goal: test whether physical hot-expert ordering in the full packed layout improves the
E043 arena baseline.

## Repack

Command:

```bash
python3 scripts/repack_packed_layout.py \
  --input-layout /home/cole/runs/deepseek-v4-flash/packed-full-e018/layout.json \
  --out-dir /home/cole/runs/deepseek-v4-flash/packed-full-hot-e045 \
  --route-jsonl /home/cole/runs/deepseek-v4-flash/route-cache1024-t8-routes.jsonl \
  --order hot_first \
  --overwrite
```

The repack preserves the full set of experts and changes only physical expert order
within each layer file.

## Repack Artifacts

- `results/spark-66c9/packed-full-hot-e045.log`
- `results/spark-66c9/packed-full-hot-e045-layout.json`

Repack output:

| Metric | Value |
| --- | ---: |
| Layers | 43 |
| Experts per layer | 256 |
| Layer bytes | 3,422,552,064 |
| Wall time | 1:32.40 |
| Max RSS | 193,976 KiB |

## Runtime Probe

E046 measured the hot-first layout with the current best full-layout flags:

```bash
DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT=/runs/packed-full-hot-e045/layout.json
DEEPSEEK_SPARK_PACKED_NATIVE_LOADER=1
DEEPSEEK_SPARK_PACKED_NATIVE_PINNED_STAGING=1
DEEPSEEK_SPARK_DIRECT_PARAM_COPY=1
DEEPSEEK_SPARK_PARAM_COPY_NON_BLOCKING=1
DEEPSEEK_SPARK_NATIVE_MATERIALIZER=1
DEEPSEEK_SPARK_EXPERT_ARENA_SLOTS=256
```

Runtime artifacts:

- `results/spark-66c9/hot-full-t32-e046.log`
- `results/spark-66c9/hot-full-t32-e046.jsonl`

## Results

| Run | Layout | Decode | Tok/s | Packed read | Materialize | Arena allocations | Arena reuses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E043 | original full | 16.516s | 1.937 | 3.660s | 5.435s | 990 | 2249 |
| E046 | hot-first full | 30.899s | 1.036 | 16.096s | 18.482s | 990 | 2249 |

Cached continuation on the hot layout remained fast:

| Tokens | Prefill | Decode | Read | Materialize |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.448s | 0.00845s | 0.084s | 0.098s |

## Interpretation

Hot-first physical ordering was a regression. It preserved packed correctness well enough
to serve the request, and arena behavior was identical to E043, but read and
materialization accounting grew sharply.

The likely lesson is that simple hot-first ordering from an 8-token route trace is too
weak a proxy for a 32-token full-layout request. Future layout work should be screened
with route-trace replay over longer traces and should compare page-cache state explicitly.
