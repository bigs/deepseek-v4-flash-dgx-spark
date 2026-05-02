# Experiment 029: Longer Decode

Date: 2026-05-02

Host: `spark-66c9`

Goal: measure decode throughput past the 8-token probe using the full all-expert packed
layout.

## Command

The successful run was `long-decode-e029c`. The earlier `long-decode-e029` and
`long-decode-e029b` attempts failed because the harness used an incomplete `PYTHONPATH`.
The patched probe now includes HTTP error bodies, which exposed the real failure:
`ModuleNotFoundError("No module named 'fast_hadamard_transform'")`.

```bash
python3 scripts/guarded_docker_run.py \
  --name long-decode-e029c \
  --image deepseek-v4-flash-spark:tilelang \
  --entrypoint /bin/bash \
  --log-file /home/cole/runs/deepseek-v4-flash/long-decode-e029c.log \
  --memory 112g \
  --memory-swap 120g \
  --min-mem-available-gib 12 \
  --timeout-seconds 3600 \
  --mount /home/cole/deepseek-v4-flash-dgx-spark:/repo:ro \
  --mount /home/cole/models/deepseek-v4-flash/hf:/model:ro \
  --mount /home/cole/runs/deepseek-v4-flash:/runs \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env DEEPSEEK_SPARK_MODEL_DIR=/model \
  --env DEEPSEEK_SPARK_MANIFEST_CSV=/repo/results/spark-66c9/weight-manifest.csv \
  --env DEEPSEEK_SPARK_TELEMETRY_JSONL=/runs/long-decode-e029c.jsonl \
  --env DEEPSEEK_SPARK_POSTFILL_MODE=deferred \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES=1024 \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_POLICY=layer_lru \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_COUNT=43 \
  --env DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT=/runs/packed-full-e018/layout.json \
  --recover-nvidia-driver-on-exit \
  -- -lc "cd /repo && python3 /repo/scripts/probe_server_http.py --prompt Hello --max-tokens 32 --skip-chat"
```

## Artifacts

- `results/spark-66c9/long-decode-e029.log`
- `results/spark-66c9/long-decode-e029b.log`
- `results/spark-66c9/long-decode-e029c.log`
- `results/spark-66c9/long-decode-e029c.jsonl`

## Results

| Scenario | Layout | Tokens | Prefill | Decode | Decode tok/s | Packed read | Materialize | Packed misses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E027 | full | 8 | 92.612s | 8.283s | 0.97 | 3.964s | 6.190s | 0 |
| E029c | full | 32 | 92.102s | 30.307s | 1.06 | 11.527s | 17.106s | 0 |

The 32-token run loaded 3,239 packed experts and had 5,017 expert-cache hits. The cache
held 989 entries after the completion and evicted 2,250 entries during the run.

Decode step walls:

```text
1.996s, 1.303s, 1.176s, 1.077s, 0.879s, 0.740s, 0.748s, 0.803s,
1.133s, 1.044s, 0.909s, 0.919s, 1.098s, 0.738s, 0.912s, 0.849s,
0.722s, 1.115s, 0.768s, 0.763s, 0.877s, 1.177s, 1.484s, 1.385s,
0.766s, 0.776s, 0.926s, 0.898s, 0.811s, 0.835s, 0.671s, 0.008s
```

The non-final decode steps averaged `0.977s`. After the first 8 tokens, the non-final
decode steps averaged `0.938s`.

The cached one-token continuation reused 33 prompt tokens but still had a `0.519s`
prefill/resume fallback because deferred postfill had not completed before the follow-up
request.

## Interpretation

Longer decode is more stable than the 8-token full-layout result but still far from the
20 tok/s ambition. The current bottleneck remains synchronous expert miss servicing:
`11.527s` packed reads and `17.106s` materialization inside a `30.307s` decode.

This points to the next optimization boundary:

- replay real route traces without the full model loop;
- replace per-expert PyTorch object materialization with reusable slabs and views;
- make layout physical locality a first-class variable;
- only then add lower-level async I/O and CUDA-stream overlap.
