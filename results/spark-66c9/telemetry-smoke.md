# Telemetry Smoke Test

Date: 2026-05-01

Status: pass

Raw artifacts:

- `telemetry-smoke.log`
- `telemetry-smoke.jsonl`

## Event Summary

| Event | Count |
| --- | ---: |
| `engine_load` | 1 |
| `engine_generate` | 2 |
| `engine_deferred_postfill` | 1 |

## First Request

Prompt: `Hello`

Output: `World =`

Request id: `050bbeb4b4e840d0b53d89b7bc01dfef`

| Metric | Value |
| --- | ---: |
| HTTP elapsed | 216.7578s |
| engine total | 216.7537s |
| model load | 58.6991s |
| prompt tokens | 1 |
| completion tokens | 2 |
| prefill wall | 124.3763s |
| prefill CUDA | 124376.2812 ms |
| decode wall | 33.5415s |
| decode CUDA | 33541.5195 ms |
| first decode step wall | 33.5412s |
| first decode step CUDA | 33529.9805 ms |
| postfill scheduling wall | 0.00007s |
| cache resume hit | false |

Memory available inside CUDA dropped from 27.57 GiB before generation to 12.96 GiB afterward.

## Deferred Postfill

Session: `http-probe`

| Metric | Value |
| --- | ---: |
| tokens | 3 |
| wall | 21.9065s |
| CUDA | 21906.4473 ms |
| health polls | 46 |
| poll errors | 0 |
| max health poll latency | 0.0099s |

## Cached Continuation

Request id: `93b82109d4984afc943c225e40f1d0e7`

Output: ` function`

| Metric | Value |
| --- | ---: |
| HTTP elapsed | 0.0033s |
| engine total | 0.0011s |
| prompt tokens | 3 |
| completion tokens | 1 |
| reused prefix tokens | 3 |
| newly prefilled tokens | 0 |
| prefill wall | 0.00017s |
| decode wall | 0.00025s |
| cache resume hit | true |

Memory before and after cached generation was unchanged at 10.86 GiB available inside CUDA.

## Host Recovery

The guarded container exited with return code 0. Driver recovery completed and the host reported
118 GiB available memory afterward.
