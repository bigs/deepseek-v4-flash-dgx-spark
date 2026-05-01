# Experiment 009: Telemetry Smoke Test

Date: 2026-05-01

Host: `spark-66c9`

Goal: verify that the serving runtime emits structured telemetry for real GPU inference while
keeping the existing memory guardrails in place.

## Command

The repo was synced to the Spark host and run with JSONL, NVTX, and CUDA event timing enabled:

```bash
./scripts/guarded_docker_run.py \
  --name telemetry-smoke \
  --image deepseek-v4-flash-spark:tilelang \
  --log-file ~/runs/deepseek-v4-flash/telemetry-smoke.log \
  --memory 104g \
  --memory-swap 112g \
  --timeout-seconds 1200 \
  --min-mem-available-gib 16 \
  --recover-nvidia-driver-on-exit \
  --mount ~/models/deepseek-v4-flash/hf:/model:ro \
  --mount "$PWD":/repo:ro \
  --mount ~/runs/deepseek-v4-flash:/runs \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env DEEPSEEK_SPARK_TELEMETRY_JSONL=/runs/telemetry-smoke.jsonl \
  --env DEEPSEEK_SPARK_NVTX=1 \
  --env DEEPSEEK_SPARK_CUDA_EVENTS=1 \
  --env DEEPSEEK_SPARK_TELEMETRY_TOKEN_CAP=16 \
  --env DEEPSEEK_SPARK_POSTFILL_MODE=deferred \
  --entrypoint python3 \
  -- /repo/scripts/probe_server_http.py \
    --prompt Hello \
    --max-tokens 2 \
    --wait-postfill-before-cached \
    --skip-chat
```

## Result

Status: pass.

Artifacts:

- `results/spark-66c9/telemetry-smoke.log`
- `results/spark-66c9/telemetry-smoke.jsonl`

JSONL emitted four events:

1. `engine_load`
2. first `engine_generate`
3. `engine_deferred_postfill`
4. cached `engine_generate`

The first request generated `World =` from prompt `Hello` with two completion tokens. Its total
request wall time was 216.76s, including 58.70s of model load, 124.38s prefill CUDA time, and
33.54s decode CUDA time. The first decode forward was the hot step at 33.53s CUDA time.

The deferred postfill took 21.91s CUDA time for the three-token cached sequence.

The cached continuation reused all three prompt tokens and returned in 0.0033s HTTP wall time.
Its engine telemetry showed a cache resume hit, 0 newly-prefilled tokens, 0.00017s prefill wall
time, and stable memory before/after.

The memory guard remained healthy. The lowest visible tail sample was 58.60 GiB available during
the run, and NVIDIA driver recovery returned the host to 118 GiB available afterward.

## Observations

The telemetry is already useful enough to guide the next optimization step:

- Cold-start load is a one-time cost, but it should be separated from steady-state latency in
  comparisons.
- Prefill for a one-token prompt is still dominated by the full lazy runtime path and weight
  movement, not prompt length.
- The decode forward is the obvious next kernel-level target: a single decode model forward took
  33.53s CUDA time in this run.
- Persistent KV/cache lookahead is working: cached continuation avoided prefill and returned
  effectively immediately.

Next experiment: capture Nsight Systems around the same request with the new NVTX ranges, then use
Nsight Compute on the decode-forward kernels identified in the timeline.
