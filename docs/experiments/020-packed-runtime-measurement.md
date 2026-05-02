# Experiment 020: Packed Runtime Measurement

Date: 2026-05-01

Host: `spark-66c9`

Goal: measure the real lazy runtime with packed expert misses enabled.

## Command

```bash
python3 scripts/guarded_docker_run.py \
  --name packed-route-e020 \
  --image deepseek-v4-flash-spark:tilelang \
  --entrypoint /bin/bash \
  --log-file /home/cole/runs/deepseek-v4-flash/packed-route-e020.log \
  --memory 104g \
  --memory-swap 112g \
  --min-mem-available-gib 16 \
  --timeout-seconds 1800 \
  --mount /home/cole/deepseek-v4-flash-dgx-spark:/repo:ro \
  --mount /home/cole/models/deepseek-v4-flash/hf:/model:ro \
  --mount /home/cole/runs/deepseek-v4-flash:/runs \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env DEEPSEEK_SPARK_MANIFEST_CSV=/repo/results/spark-66c9/weight-manifest.csv \
  --env DEEPSEEK_SPARK_TELEMETRY_JSONL=/runs/packed-route-e020.jsonl \
  --env DEEPSEEK_SPARK_POSTFILL_MODE=deferred \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES=1024 \
  --env DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT=/runs/packed-route-e018/layout.json \
  -- -lc "cd /repo && python3 /repo/scripts/probe_server_http.py --prompt Hello --max-tokens 8 --skip-chat"
```

## Artifacts

- `results/spark-66c9/packed-route-e020.log`
- `results/spark-66c9/packed-route-e020.jsonl`

## Results

Compared with the previous `route-cache1024-t8` baseline:

| Scenario | Prefill | Decode | Decode step walls |
| --- | ---: | ---: | --- |
| `route-cache1024-t8` | 125.083s | 115.604s | 26.873s, 17.828s, 17.410s, 18.022s, 13.525s, 11.917s, 10.021s, 0.008s |
| `packed-route-e020` | 95.067s | 15.642s | 2.448s, 1.421s, 2.796s, 2.791s, 2.274s, 2.063s, 1.840s, 0.008s |
| `packed-cache2048-e020` | 95.830s | 13.900s | 1.416s, 1.006s, 2.796s, 2.752s, 2.179s, 1.974s, 1.769s, 0.008s |

Decode throughput for the 8 generated tokens was about `0.51 tok/s`.

Expert-cache/miss telemetry for the first completion:

- packed loads: 1,147
- packed misses: 0
- packed read bytes: 15,334,637,568
- packed read seconds: 10.293s
- materialize seconds: 14.164s
- cache hits: 917
- cache misses: 1,147
- evictions: 123

The 2048-entry cache eliminated evictions for this short route:

- entries after first completion: 1,145
- evictions: 0
- packed loads: 1,145
- packed read seconds: 10.292s
- materialize seconds: 12.617s

The cached one-token continuation completed in `2.118s`, with `2.109s` spent in the
single-token prefill/resume path and only `0.008s` in decode bookkeeping.

## Interpretation

Packed misses are the biggest runtime win so far. The route-filtered layout removed the
safetensors miss path completely for this probe and cut decode time by about `7.4x`.
Increasing the cache to 2048 entries helped less than expected, which means the first-use
miss path is still the dominant cost for this short generation.

We are still below the sprint target of 1 tok/s. The next suspects are:

- cache policy churn from global LRU evictions;
- remaining Python parameter-copy/object overhead on misses;
- the single-token prefill/resume path doing a full model forward when deferred postfill
  leaves `next_logits` unset.
