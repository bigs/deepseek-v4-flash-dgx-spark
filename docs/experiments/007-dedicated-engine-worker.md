# Experiment 007: Dedicated Engine Worker

Date: 2026-05-01

Host: `spark-66c9`

## Goal

Move blocking CUDA/model work off the FastAPI event loop while preserving the
single-worker model execution policy. Experiment 006 proved deferred cache
lookahead, but the deferred postfill still blocked health checks because it ran
inside the event loop.

## Change

The server now wraps the runtime in a dedicated single-thread engine worker:

- all generation and postfill CUDA/model calls run on one worker thread
- the HTTP event loop remains available for health checks and request handling
- health reads use a cached worker snapshot instead of queueing behind CUDA work
- generation and postfill are still serialized by one model worker

## Command

```bash
./scripts/guarded_docker_run.py \
  --name mvp-server-worker-lookahead-v4 \
  --image deepseek-v4-flash-spark:tilelang \
  --log-file ~/runs/deepseek-v4-flash/mvp-server-worker-lookahead-v4.log \
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

Key measurements:

```text
first completion elapsed_seconds: 176.1555534880008
first completion prefill_seconds: 120.48644326499925
deferred postfill wait: 28.12768760600011
health polls during postfill: 57
health poll errors: 0
max health poll latency: 0.0022077150006225565
cached completion elapsed_seconds: 0.0025925919999281177
cached completion prefill_seconds: 0.000032160000046133064
cached completion cache_resume_hit: 1.0
```

Compared with experiment 006, the cached continuation remains effectively
instant after deferred postfill, and health checks now stay responsive while the
background postfill is running.

## Interpretation

This establishes the right serving shape for the MVP:

- FastAPI owns HTTP, request queueing, and OpenAI-compatible response handling.
- The runtime owns model/cache state.
- A dedicated worker serializes all model work without blocking the event loop.

The next queueing improvement should make request cancellation and timeouts
explicit. Today, a queued request can wait behind a deferred postfill because
both use the same single worker, which is correct for cache consistency but
should be surfaced in metrics and policy.

Full log: `results/spark-66c9/mvp-server-worker-lookahead-v4.log`
