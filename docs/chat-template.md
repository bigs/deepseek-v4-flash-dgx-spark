# DeepSeek-V4 Chat Encoding

Date: 2026-05-01

DeepSeek-V4-Flash does not ship a Hugging Face/Jinja `chat_template`. The
official repository provides `encoding/encoding_dsv4.py`, which converts
OpenAI-style message dictionaries into the text prompt consumed by the model.

Primary references:

- `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/README.md`
- `/home/cole/models/deepseek-v4-flash/hf/encoding/README.md` on `spark-66c9`
- `/home/cole/models/deepseek-v4-flash/hf/encoding/encoding_dsv4.py` on
  `spark-66c9`

## Basic Format

Chat/non-thinking mode:

```text
<｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜></think>
```

Completed assistant turns in chat mode:

```text
<｜Assistant｜></think>{content}<｜end▁of▁sentence｜>
```

Thinking mode:

```text
<｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜><think>
```

Completed assistant turns in thinking mode:

```text
<｜Assistant｜><think>{reasoning}</think>{content}<｜end▁of▁sentence｜>
```

## API Mapping

`POST /v1/chat/completions` accepts Pydantic schemas for the fields consumed by
the official encoder:

- message roles: `system`, `user`, `assistant`, `tool`, `latest_reminder`,
  `developer`
- assistant history: `reasoning_content`, `tool_calls`
- tool results: `tool_call_id` on `tool` messages
- tool schemas: request-level `tools`, or message-level `tools`
- response format: request-level `response_format`, or message-level
  `response_format`
- quick instruction tasks: `task`
- DeepSeek-specific controls: `spark_thinking_mode` and
  `spark_reasoning_effort`

OpenAI-compatible request-level `tools` and `response_format` are normalized
onto the first `system` or `developer` message before calling
`encode_messages()`. If the request has no such message, the server inserts an
empty `system` message and attaches them there.

`POST /v1/completions` remains a raw prompt endpoint and does not apply the
chat encoder.
