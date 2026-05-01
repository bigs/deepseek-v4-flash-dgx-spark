# MVP Server HTTP Probe

Date: 2026-05-01

Source log: `mvp-server-http-probe.log`

## Summary

- Result: success, guarded Docker exit code `0`
- Image: `deepseek-v4-flash-spark:tilelang`
- Model mount: `/home/cole/models/deepseek-v4-flash/hf:/model:ro`
- Repo mount: `/home/cole/deepseek-v4-flash-dgx-spark:/repo:ro`
- Memory guard: `104g` cgroup memory, `112g` memory+swap,
  `16 GiB` minimum host `MemAvailable`

## HTTP Checks

- `GET /v1/models`: returned model id `deepseek-v4-flash-dgx-spark`
- `POST /v1/completions`: returned `World` for prompt `Hello`
- Cached `POST /v1/completions`: returned ` =` for prompt `HelloWorld`
- `POST /v1/chat/completions`: returned a valid OpenAI-style chat response

## Key Metrics

| Request | Prompt tokens | Completion tokens | Reused prefix | Newly prefilled | Prefill seconds | Generated text |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| completion | 1 | 1 | 0 | 1 | 120.40 | `World` |
| cached completion | 2 | 1 | 2 | 0 | 31.02 | ` =` |
| chat completion | 5 | 1 | 0 | 5 | 89.89 | `你好` |

The cached completion is the first end-to-end proof that the OpenAI-compatible
server can carry persistent session state across HTTP requests.

## Recovery

The `--recover-nvidia-driver-on-exit` hook ran after server shutdown. Final
host status:

```text
MemAvailable: 118 GiB
No running GPU processes found
```
