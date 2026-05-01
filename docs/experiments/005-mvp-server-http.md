# Experiment 005: MVP HTTP Server Probe

Date: 2026-05-01

Host: `spark-66c9`

## Goal

Verify that the repo can serve DeepSeek-V4-Flash through an OpenAI-compatible
HTTP surface using the lazy official runtime, and prove that the first
session-cache path reuses model cache state across requests.

## Command

The server was launched inside the guarded Docker wrapper:

```bash
./scripts/guarded_docker_run.py \
  --name mvp-server-http-probe \
  --image deepseek-v4-flash-spark:tilelang \
  --log-file ~/runs/deepseek-v4-flash/mvp-server-http-probe.log \
  --memory 104g \
  --memory-swap 112g \
  --timeout-seconds 7200 \
  --min-mem-available-gib 16 \
  --recover-nvidia-driver-on-exit \
  --mount ~/models/deepseek-v4-flash/hf:/model:ro \
  --mount "$PWD":/repo:ro \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --entrypoint python3 \
  -- /repo/scripts/probe_server_http.py --prompt Hello --max-tokens 1
```

## Result

Exit code: `0`

The probe hit:

- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/completions` again with the same `spark_session_id` and an
  extended prompt
- `POST /v1/chat/completions`

The first completion produced:

```text
Hello -> World
generated_token_ids: [21133]
newly_prefilled_tokens: 1
reused_prefix_tokens: 0
prefill_seconds: 120.40
```

The cached follow-up produced:

```text
HelloWorld ->  =
generated_token_ids: [438]
newly_prefilled_tokens: 0
reused_prefix_tokens: 2
prefill_seconds: 31.02
```

The chat completion endpoint produced a valid `chat.completion` object:

```text
prompt_tokens: 5
generated_token_ids: [30594]
assistant text: 你好
prefill_seconds: 89.89
```

The host memory guard never tripped. `MemAvailable` started at `118.45 GiB`,
fell to a low of roughly `51.67 GiB` during the chat request, and returned to
`118 GiB` after the driver recovery hook.

## Interpretation

This establishes the first usable serving baseline:

- OpenAI-compatible non-streaming completions and chat completions work.
- The server runs through the repo-owned runtime rather than vLLM/SGLang.
- Requests are bounded by an explicit max-depth server queue and serialized by
  the engine's single CUDA worker.
- One-session persistent cache reuse works when the next prompt extends the
  cached token prefix.

The cached follow-up still spends about 31 seconds in the current `prefill`
phase because the MVP recomputes logits from the last cached token. The next
serving optimization should store the last logits or split the metric so this
single-token resume path is not counted as prefix prefill.

Full log: `results/spark-66c9/mvp-server-http-probe.log`
