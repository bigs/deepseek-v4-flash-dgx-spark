# Chat Until Stop Probe

Date: 2026-05-01

Source log: `chat-hello-until-stop.log`

## Summary

- Result: success, guarded Docker exit code `0`
- Endpoint: `POST /v1/chat/completions`
- Input user message: `Hello, how are you?`
- Max cap: `128` tokens
- Finish reason: `stop`
- Stop token id: `1`

## Response

```text
Hello! I'm just a computer program, so I don't have feelings, but I'm running smoothly and ready to help you. How can I assist you today?
```

## Measurements

| Measurement | Value |
| --- | ---: |
| Wall time | `597.1428210600006s` |
| Prompt tokens | `10` |
| Completion tokens | `35` |
| Total tokens | `45` |
| Prefill seconds | `269.89465479700084s` |
| Decode seconds | `274.275096708001s` |
| Completion-token decode rate | `0.128 tokens/s` |
| Forward-step decode rate | `0.124 tokens/s` |

The final generated token id was `1`, and `full_text` ended with
`<｜end▁of▁sentence｜>`.

## Recovery

The guard recovered the NVIDIA driver after the run. Final host status:

```text
MemAvailable: 118 GiB
No running GPU processes found
```
