# Experiment 008: Chat Until Stop Token

Date: 2026-05-01

Host: `spark-66c9`

## Goal

Run the OpenAI-compatible chat API with the official DeepSeek-V4 chat encoding
and let generation continue until the model emits its end-of-assistant-turn
token instead of stopping at an arbitrary one-token cap.

## Change

The engine now treats DeepSeek's EOS token as a stop token:

```text
<｜end▁of▁sentence｜>
```

When this token is generated, the API returns `finish_reason: "stop"` and does
not schedule deferred postfill.

## Command

```bash
./scripts/guarded_docker_run.py \
  --name chat-hello-until-stop \
  --image deepseek-v4-flash-spark:tilelang \
  --log-file ~/runs/deepseek-v4-flash/chat-hello-until-stop.log \
  --memory 104g \
  --memory-swap 112g \
  --timeout-seconds 9000 \
  --min-mem-available-gib 16 \
  --recover-nvidia-driver-on-exit \
  --mount ~/models/deepseek-v4-flash/hf:/model:ro \
  --mount "$PWD":/repo:ro \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env DEEPSEEK_SPARK_POSTFILL_MODE=off \
  --entrypoint python3 \
  -- /repo/scripts/probe_chat_until_stop.py \
    --message "Hello, how are you?" \
    --max-tokens 128 \
    --request-timeout 7200
```

## Result

Exit code: `0`

Input:

```text
Hello, how are you?
```

Assistant response:

```text
Hello! I'm just a computer program, so I don't have feelings, but I'm running smoothly and ready to help you. How can I assist you today?
```

The API returned:

```text
finish_reason: stop
prompt_tokens: 10
completion_tokens: 35
total_tokens: 45
generated_token_ids: [19923, 3, 342, 4571, 1438, 260, 6341, 2305, 14, 832, 342, 2090, 1664, 611, 13227, 14, 790, 342, 4571, 6934, 42379, 305, 7692, 304, 1694, 440, 16, 1730, 588, 342, 8233, 440, 4316, 33, 1]
```

The final token id was `1`, which decoded as
`<｜end▁of▁sentence｜>` in the full text.

## Timing

```text
wall time: 597.14s
prefill_seconds: 269.89
decode_seconds: 274.28
completion_tokens: 35
```

The observed decode phase was about `0.128 tokens/s` if counting all generated
tokens, including the EOS token. Because the first completion token comes from
prefill logits, the subsequent model-forward decode rate is about
`34 / 274.28 = 0.124 tokens/s`.

## Recovery

The host memory guard never tripped. After the run, the driver recovery hook
restored the host to:

```text
MemAvailable: 118 GiB
No running GPU processes found
```

Full log: `results/spark-66c9/chat-hello-until-stop.log`
