# Instrumentation Plan

This repo uses a layered instrumentation stack for low-level inference work:

1. Always-on request telemetry: structured JSONL events for reproducible experiments.
2. NVTX ranges: timeline labels for NVIDIA Nsight Systems.
3. Optional CUDA event timers: synchronized GPU elapsed time for controlled measurements.
4. Targeted kernel profiling: Nsight Compute on the exact kernels or ranges that matter.
5. Service metrics: Prometheus/OpenTelemetry can sit above this once the runtime is stable.

That split keeps production-ish serving cheap while still making profiler captures legible when
we need them. NVIDIA documents NVTX as the annotation layer Nsight Systems can display in the
timeline, Nsight Compute as the kernel-level CUDA profiler, and CUPTI as the lower-level API used
by profiling tools. Prometheus/OpenTelemetry remain the normal service-metrics layer, but they do
not replace GPU timelines or kernel counters.

## Runtime controls

Set these environment variables in the container:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DEEPSEEK_SPARK_TELEMETRY_JSONL` | unset | Append structured events to this JSONL file. |
| `DEEPSEEK_SPARK_NVTX` | `1` | Emit NVTX ranges around load, encode, prefill, decode, decode steps, and postfill. |
| `DEEPSEEK_SPARK_CUDA_EVENTS` | `0` | Add synchronized CUDA event timings. Useful for experiments; adds synchronization overhead. |
| `DEEPSEEK_SPARK_TELEMETRY_TOKEN_CAP` | `256` | Maximum decode-step timings stored per request. |

The API response also includes `spark_metrics.request_id` and server timing fields under
`spark_metrics.timings`, including queue wait and request wall time.

## JSONL events

The engine currently emits:

- `engine_load`: model construction and tokenizer load timing, memory before/after, load counts.
- `engine_generate`: one event per generation with request id, token counts, cache hit/reuse data,
  finish reason, memory before/after, wall timings, optional CUDA timings, and per-token decode
  timing samples.
- `engine_deferred_postfill`: background postfill timing for persistent KV-cache lookahead.

## Nsight workflow

Use JSONL for every experiment, then take Nsight captures only around interesting cases:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  -o /runs/nsys-decode \
  python3 /repo/scripts/probe_server_http.py --prompt Hello --max-tokens 4 --skip-chat
```

Once a slow or hot kernel is identified in Nsight Systems, narrow down with Nsight Compute:

```bash
ncu --set full --target-processes all \
  -o /runs/ncu-decode \
  python3 /repo/scripts/probe_server_http.py --prompt Hello --max-tokens 2 --skip-chat
```

For this project, the first optimization loop should be:

1. Record JSONL for a baseline run.
2. Inspect prefill, decode, and postfill wall/CUDA timing splits.
3. Capture Nsight Systems with NVTX ranges for the same prompt.
4. Pick the slowest decode kernel family and profile it with Nsight Compute.
5. Change one thing, rerun the same JSONL experiment, and compare before adding more profiler cost.

## References

- NVIDIA Nsight Systems User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- NVIDIA Nsight Compute documentation: https://docs.nvidia.com/nsight-compute/
- NVIDIA CUPTI documentation: https://docs.nvidia.com/cupti/
- PyTorch profiler documentation: https://docs.pytorch.org/docs/stable/profiler.html
- Prometheus instrumentation practices: https://prometheus.io/docs/practices/instrumentation/
- OpenTelemetry metrics: https://opentelemetry.io/docs/concepts/signals/metrics/
