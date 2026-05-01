# Deferred Cache Lookahead Probe

Date: 2026-05-01

Source log: `mvp-server-cache-lookahead-v3.log`

## Summary

- Result: success, guarded Docker exit code `0`
- Mode: `DEEPSEEK_SPARK_POSTFILL_MODE=deferred`
- First request: `Hello -> World`
- Deferred postfill: completed before the cached request
- Cached request: `HelloWorld ->  =`

## Measurements

| Measurement | Value |
| --- | ---: |
| First completion wall time | `178.39523218700015s` |
| First completion prefill | `120.40532091400019s` |
| Deferred postfill wait | `28.59614102500018s` |
| Cached completion wall time | `0.004936354998790193s` |
| Cached completion prefill | `0.00003833599839708768s` |
| Cached completion cache resume hit | `1.0` |
| Cached completion reused prefix tokens | `2` |
| Cached completion newly prefilled tokens | `0` |

Experiment 005 measured the same cached continuation at
`31.01734592899993s` of `prefill_seconds`. With deferred cache lookahead, the
cached request hits stored logits and returns in about 5 ms after postfill has
completed.

## Recovery

The guard recovered the NVIDIA driver after the run. Final host status:

```text
MemAvailable: 118 GiB
No running GPU processes found
```
