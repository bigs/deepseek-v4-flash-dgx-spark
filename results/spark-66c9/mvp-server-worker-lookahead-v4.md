# Dedicated Engine Worker Probe

Date: 2026-05-01

Source log: `mvp-server-worker-lookahead-v4.log`

## Summary

- Result: success, guarded Docker exit code `0`
- Mode: `DEEPSEEK_SPARK_POSTFILL_MODE=deferred`
- Worker: dedicated single model thread
- Health checks during postfill: responsive
- Cached continuation: still hits stored logits

## Measurements

| Measurement | Value |
| --- | ---: |
| First completion wall time | `176.1555534880008s` |
| First completion prefill | `120.48644326499925s` |
| Deferred postfill wait | `28.12768760600011s` |
| Health polls during postfill | `57` |
| Health poll errors | `0` |
| Max health poll latency | `0.0022077150006225565s` |
| Cached completion wall time | `0.0025925919999281177s` |
| Cached completion prefill | `0.000032160000046133064s` |
| Cached completion cache resume hit | `1.0` |
| Cached completion reused prefix tokens | `2` |
| Cached completion newly prefilled tokens | `0` |

## Recovery

The guard recovered the NVIDIA driver after the run. Final host status:

```text
MemAvailable: 118 GiB
No running GPU processes found
```
