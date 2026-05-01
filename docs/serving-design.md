# Serving Design

Date: 2026-05-01

Goal: turn the first-token research harness into a useful single-Spark inference
service while preserving the freedom to use custom GB10 kernels and native
FP8/FP4 model storage.

This service is intentionally not a full vLLM/SGLang clone. It should provide
the OpenAI-compatible surface we need, queue requests safely, and keep the model
runtime narrow enough that we can keep changing kernels, expert loading, and
cache policy quickly.

## Non-Negotiable Constraints

- Keep one long-lived engine process with resident dense/shared weights.
- Keep full `1048576` context buffers allocated when configured for 1M context.
- Preserve native FP4 routed expert storage; do not expand all experts to BF16.
- Maintain persistent KV/compressed cache state so repeated turns do not prefill
  the full prefix again.
- Queue requests instead of running concurrent generations until batching is
  intentionally designed.
- Expose OpenAI-compatible APIs early enough that normal clients can test it.
- Keep HTTP/server code separate from model/runtime/kernel code.

## Initial Architecture

### Engine Layer

The engine owns model state and exposes a small API:

- `load(model_dir, max_seq_len=1048576)`
- `generate(messages_or_prompt, max_tokens, temperature, stop, stream, session_id, cache_policy)`
- `reset_session(session_id)`
- `health()`

The first implementation should wrap `spark_runtime.lazy_official_runtime`:

- official DeepSeek-V4 model structure
- resident non-routed weights
- lazy routed expert materialization from safetensors
- full-context cache buffers
- correctness-first sparse attention fallback until a GB10 kernel replaces it

### Server Layer

Use a simple Python HTTP server first, likely FastAPI/uvicorn.

Required endpoints:

- `GET /health`
- `GET /ready`
- `GET /v1/models`
- `POST /v1/chat/completions`

Useful soon after:

- `POST /v1/completions`
- `POST /spark/sessions/{id}/reset`
- `GET /spark/metrics`

The OpenAI-compatible response should support both:

- non-streaming `chat.completion`
- streaming SSE `chat.completion.chunk`

Sampling can start greedy-only. Add temperature/top-p once the queue and cache
semantics are stable.

## Queueing Model

Start with one generation worker:

- FIFO queue
- configurable max queue depth
- request timeout
- cancellation when the client disconnects
- one active CUDA/model forward path at a time

Do not implement vLLM-style continuous batching yet. Lazy expert routing makes
batching a real design problem, not a wrapper feature. Once single-request
serving is stable, use measurements to decide whether batching or smarter
session scheduling is worth it.

## Persistent KV / Session Cache

Persistent cache reuse is mandatory. Without it, long conversations will spend
most of their time repeatedly prefilling the same prefix.

The official model stores KV/compressed cache state inside model layers, so the
first server should support one active cache slot. That is enough for a
single-user Spark server and much simpler than multi-session cache paging.

Request extensions:

```json
{
  "spark_session_id": "default",
  "spark_cache_policy": "reuse"
}
```

Cache policies:

- `reuse`: append only tokens not already present in the session cache
- `reset`: clear the session then prefill the supplied prompt
- `none`: run as stateless as practical, without updating reusable session state

Session state:

- token prefix
- current `start_pos`
- generated history
- cache validity flag
- last access time

Prompt handling:

- If the new prompt extends the cached token prefix, only prefill the suffix.
- If the prompt diverges from the cached prefix, either reset or reject based on
  `spark_cache_policy`.
- If no `spark_session_id` is provided, use an ephemeral request or a default
  single-session policy configurable at server startup.

Future multi-session support can add:

- fixed cache slots
- LRU cache-slot eviction
- host/NVMe cache-state snapshots
- explicit session pinning

## Metrics

Every request should record:

- queue wait time
- prompt token count
- reused prefix token count
- newly-prefilled token count
- generated token count
- resident load status
- prefill time
- per-token decode time
- routed expert load count
- routed expert cache hit rate, once caching exists
- safetensor bytes read, once instrumented
- sparse attention fallback time
- CUDA `mem_get_info()` before and after request
- host `MemAvailable` before and after request

These metrics are the measurement surface for optimization. Avoid optimizing
before the server can report where time and memory go.

## Optimization Hooks

Keep these behind engine interfaces so HTTP behavior does not change:

- routed expert cache with configurable memory budget
- expert prefetch after gate decisions
- reusable FP4 expert buffers instead of module re-creation
- GB10-compatible sparse attention kernel
- FP4 routed expert fusion for small active batches
- explicit page-cache hints for safetensor reads
- optional converted expert layout if measurements show HF layout becomes a
  bottleneck

## First Serving Milestones

- `S1`: persistent CLI loads once and generates multiple turns without restart.
- `S2`: done, `/v1/models` and non-streaming `/v1/chat/completions` work.
- `S3`: implemented in the server harness; tested with a fake engine locally.
- `S4`: done, one-session persistent KV reuse works; second request appends to cached
  prefix without full-prefix prefill.
- `S5`: partial, single-worker FIFO behavior and max queue depth exist; request
  timeout and disconnect cancellation still need hardening. Model work now runs
  on a dedicated worker thread so the HTTP event loop remains responsive.
- `S6`: per-request metrics identify prefill, decode, expert load, and attention
  costs.
- `S7`: done, deferred cache lookahead replaces the measured cached-resume
  forward when there is idle time before the next turn.

See `docs/experiments/005-mvp-server-http.md` for the first successful HTTP
probe on `spark-66c9`.
See `docs/experiments/006-deferred-cache-lookahead.md` for the first measured
serving optimization.
See `docs/experiments/007-dedicated-engine-worker.md` for the worker-thread
server cleanup.
See `docs/chat-template.md` for the official DeepSeek-V4 message encoding
surface used by `/v1/chat/completions`.
See `docs/experiments/008-chat-until-stop.md` for the first API run that
decoded until the DeepSeek EOS token.
