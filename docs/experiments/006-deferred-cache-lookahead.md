# Experiment 006: Deferred Cache Lookahead

Date: 2026-05-01

Host: `spark-66c9`

## Goal

Optimize the cached-continuation path measured in experiment 005. The previous
server reused the token prefix, but a cached continuation still spent about
31 seconds in `prefill_seconds` because the runtime had to consume the last
generated token before it could produce the next-token logits.

## Change

The engine now supports deferred postfill:

- A generation response records the logical session tokens immediately.
- If the final generated token has not been consumed by the model yet, the
  server schedules `prepare_next_logits()` after the response.
- During idle time, that deferred task consumes the final token and stores the
  next-token logits on the session.
- A later request whose prompt exactly matches the cached session tokens can
  start from those stored logits instead of running a resume forward.

The mode is controlled by:

```bash
DEEPSEEK_SPARK_POSTFILL_MODE=deferred
DEEPSEEK_SPARK_POSTFILL_DELAY_SECONDS=0.25
```

## Command

```bash
./scripts/guarded_docker_run.py \
  --name mvp-server-cache-lookahead-v3 \
  --image deepseek-v4-flash-spark:tilelang \
  --log-file ~/runs/deepseek-v4-flash/mvp-server-cache-lookahead-v3.log \
  --memory 104g \
  --memory-swap 112g \
  --timeout-seconds 3600 \
  --min-mem-available-gib 16 \
  --recover-nvidia-driver-on-exit \
  --mount ~/models/deepseek-v4-flash/hf:/model:ro \
  --mount "$PWD":/repo:ro \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env DEEPSEEK_SPARK_POSTFILL_MODE=deferred \
  --env DEEPSEEK_SPARK_POSTFILL_DELAY_SECONDS=0.25 \
  --entrypoint python3 \
  -- /repo/scripts/probe_server_http.py \
    --prompt Hello \
    --max-tokens 1 \
    --wait-postfill-before-cached \
    --skip-chat
```

## Result

Exit code: `0`

Baseline from experiment 005:

```text
cached completion prefill_seconds: 31.01734592899993
cached completion newly_prefilled_tokens: 0
cached completion reused_prefix_tokens: 2
```

Deferred lookahead run:

```text
first completion elapsed_seconds: 178.39523218700015
first completion prefill_seconds: 120.40532091400019
deferred postfill wait: 28.59614102500018
cached completion elapsed_seconds: 0.004936354998790193
cached completion prefill_seconds: 0.00003833599839708768
cached completion cache_resume_hit: 1.0
cached completion newly_prefilled_tokens: 0
cached completion reused_prefix_tokens: 2
```

The cached continuation improved from a measured 31-second resume forward to
about 5 ms response wall time after deferred postfill had completed.

## Notes

Deferred postfill currently runs as an asyncio task but still performs a
blocking CUDA/model call on the server event loop. That is acceptable for the
single-user MVP when there is idle think time between turns, but it means health
polls and new requests can wait while the background postfill is running. A
proper worker thread or dedicated engine loop is the next server-harness
cleanup.

Experiment 007 addresses this by moving model work onto a dedicated worker
thread.

Full log: `results/spark-66c9/mvp-server-cache-lookahead-v3.log`
