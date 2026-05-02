# Experiment 028: Inline Postfill

Date: 2026-05-02

Host: `spark-66c9`

Goal: test whether inline postfill makes persistent session resume latency acceptable.

## Command

This used the route-filtered packed layout to isolate postfill behavior from full-layout
locality effects.

```bash
python3 scripts/guarded_docker_run.py \
  --name inline-postfill-e028 \
  --image deepseek-v4-flash-spark:tilelang \
  --entrypoint /bin/bash \
  --log-file /home/cole/runs/deepseek-v4-flash/inline-postfill-e028.log \
  --memory 112g \
  --memory-swap 120g \
  --min-mem-available-gib 12 \
  --timeout-seconds 1800 \
  --mount /home/cole/deepseek-v4-flash-dgx-spark:/repo:ro \
  --mount /home/cole/models/deepseek-v4-flash/hf:/model:ro \
  --mount /home/cole/runs/deepseek-v4-flash:/runs \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env DEEPSEEK_SPARK_MODEL_DIR=/model \
  --env DEEPSEEK_SPARK_MANIFEST_CSV=/repo/results/spark-66c9/weight-manifest.csv \
  --env DEEPSEEK_SPARK_TELEMETRY_JSONL=/runs/inline-postfill-e028.jsonl \
  --env DEEPSEEK_SPARK_POSTFILL_MODE=inline \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES=1024 \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_POLICY=layer_lru \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_COUNT=43 \
  --env DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT=/runs/packed-route-e018/layout.json \
  --recover-nvidia-driver-on-exit \
  -- -lc "cd /repo && python3 /repo/scripts/probe_server_http.py --prompt Hello --max-tokens 8 --skip-chat"
```

## Artifacts

- `results/spark-66c9/inline-postfill-e028.log`
- `results/spark-66c9/inline-postfill-e028.jsonl`

## Results

| Request | Cache state | Prefill | Decode | Postfill | Total | Cache resume hit |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Initial 8-token completion | reset | 95.424s | 16.173s | 2.332s | 171.436s | false |
| Cached 1-token continuation | reuse | 0.000035s | 0.000122s | 9.307s | 9.309s | true |

The cached continuation shows that inline postfill can make the immediate resume path
effectively free:

```text
prefill 0.000035s, decode 0.000122s
```

But the request still paid `9.307s` in inline postfill before returning.

## Interpretation

Inline postfill proves that persistent KV cache can eliminate the repeated prefill cost,
but it is not acceptable as the default latency policy. It moves the full-forward cost
onto the critical path of every request.

The right serving policy is deferred postfill with an explicit completion/wait boundary:
serve the request quickly, finish postfill in the engine worker, and only let the next
request reuse the session once postfill has actually completed. That gives persistent KV
cache semantics without making every request wait for inline postfill.
