# Experiment 012: Expert Reuse / Frequency Telemetry

Date: 2026-05-01

Host: `spark-66c9`

Goal: measure routed expert reuse directly so expert-cache sizes can be chosen from data
instead of guesses.

## Change

Added optional routed-expert trace output:

```bash
DEEPSEEK_SPARK_EXPERT_ROUTE_TRACE_JSONL=/runs/route-cache1024-t8-routes.jsonl
```

When enabled, each routed MoE layer emits:

- layer id;
- token ids for that MoE call;
- routed expert indices;
- per-event expert histogram.

This is intentionally off by default because it synchronizes route IDs back to CPU.

Added `scripts/summarize_expert_routes.py` to produce per-layer coverage and reuse
summaries from the trace.

## Command

This run used a longer decode than the Nsight probes and skipped Nsight overhead:

```bash
python3 scripts/guarded_docker_run.py \
  --name route-cache1024-t8 \
  --image deepseek-v4-flash-spark:tilelang \
  --log-file ~/runs/deepseek-v4-flash/route-cache1024-t8.log \
  --memory 104g \
  --memory-swap 112g \
  --min-mem-available-gib 16 \
  --timeout-seconds 1800 \
  --mount ~/models/deepseek-v4-flash/hf:/model:ro \
  --mount ~/deepseek-v4-flash-dgx-spark:/repo:ro \
  --mount ~/runs/deepseek-v4-flash:/runs \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env DEEPSEEK_SPARK_MANIFEST_CSV=/repo/results/spark-66c9/weight-manifest.csv \
  --env DEEPSEEK_SPARK_TELEMETRY_JSONL=/runs/route-cache1024-t8.jsonl \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES=1024 \
  --env DEEPSEEK_SPARK_EXPERT_ROUTE_TRACE_JSONL=/runs/route-cache1024-t8-routes.jsonl \
  --env DEEPSEEK_SPARK_POSTFILL_MODE=deferred \
  --entrypoint python3 \
  --recover-nvidia-driver-on-exit \
  -- /repo/scripts/probe_server_http.py \
      --prompt Hello \
      --max-tokens 8 \
      --wait-postfill-before-cached \
      --skip-chat
```

Summary command:

```bash
python3 scripts/summarize_expert_routes.py \
  results/spark-66c9/route-cache1024-t8-routes.jsonl \
  > results/spark-66c9/route-cache1024-t8-routes-summary.md
```

## Artifacts

- `results/spark-66c9/route-cache1024-t8.log`
- `results/spark-66c9/route-cache1024-t8.jsonl`
- `results/spark-66c9/route-cache1024-t8-routes.jsonl`
- `results/spark-66c9/route-cache1024-t8-routes-summary.md`

## Result

The first completion generated:

```text
World = function() {
    return "
```

Timing:

| Metric | Value |
| --- | ---: |
| completion wall | 297.608s |
| prefill | 125.083s |
| decode | 115.604s |
| deferred postfill | 12.384s |
| cached continuation wall | 0.003s |

Decode step walls:

| Step | Wall |
| ---: | ---: |
| 0 | 26.873s |
| 1 | 17.828s |
| 2 | 17.410s |
| 3 | 18.022s |
| 4 | 13.525s |
| 5 | 11.917s |
| 6 | 10.021s |
| 7 | 0.008s |

Expert cache after the first completion:

| Metric | Value |
| --- | ---: |
| entries | 1024 |
| resident expert bytes | 13.690 GiB |
| hits | 917 |
| misses | 1147 |
| evictions | 123 |
| copied expert bytes | 15.335 GiB |
| routed calls | 344 |
| routed activations | 2064 |

After deferred postfill:

| Metric | Value |
| --- | ---: |
| entries | 1024 |
| resident expert bytes | 13.690 GiB |
| hits | 1085 |
| misses | 1237 |
| evictions | 213 |
| routed activations | 2322 |

Route summary:

| Signal | Value |
| --- | ---: |
| route events | 387 |
| routed tokens | 387 |
| routed activations | 2322 |
| routed layers observed | 43 |
| experts for 80% coverage across observed layers | 806 |
| average experts per observed layer for 80% coverage | 18.7 |

Per-layer coverage is uneven. Early layers 0 and 1 are broad and need 39 experts each
for 80% coverage in this short run. Several middle layers are much more concentrated:
for example, layer 15 needs only 8 experts for 80% coverage and layer 18 needs 9.

## Interpretation

This explains Experiment 011:

- A 64-entry global cache is far below the observed reuse footprint.
- A 1024-entry global cache is large enough to get hits, but still evicts during longer
  decode.
- The observed 80% coverage estimate is 806 experts for only 43 routed layers and a tiny
  8-token sample, so real workloads will need either a larger cache, a smarter per-layer
  budget, or both.

The step timings suggest the cache starts helping as decode proceeds: step 0 is 26.9s,
steps 1-3 are about 17-18s, and steps 4-6 fall toward 10-13s. The final step does not run
a model forward because the probe already has logits for the sampled token.

## Next Step

Move from a single global LRU to a layer-aware cache policy. The data argues for:

- a minimum per-layer reserve for hot middle layers;
- larger budgets for broad early/late layers only when memory allows;
- telemetry segmented by prefill, decode, and postfill;
- a packed expert layout so misses are cheaper.

The cache is useful but cannot be the only optimization. It reduces repeated work after
reuse appears; it does not reduce first-use materialization cost.

