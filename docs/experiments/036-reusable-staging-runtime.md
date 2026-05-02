# Experiment 036: Reusable Staging Runtime

Date: 2026-05-02

Host: `spark-66c9`

Goal: test whether the E032 reusable `preadv` staging win survives full inference.

## Command

Same as E026, plus:

```bash
DEEPSEEK_SPARK_PACKED_REUSE_STAGING=1
```

## Artifacts

- `results/spark-66c9/reuse-staging-e036.log`
- `results/spark-66c9/reuse-staging-e036.jsonl`

## Results

| Scenario | Prefill | Decode | Tok/s | Packed read | Materialize |
| --- | ---: | ---: | ---: | ---: | ---: |
| E026 no empty cache | 91.662s | 6.434s | 1.243 | 2.157s | 4.322s |
| E036 reusable staging | 92.056s | 6.408s | 1.248 | 2.265s | 4.448s |

Decode step walls:

```text
1.561s, 1.085s, 1.029s, 0.876s, 0.663s, 0.588s, 0.597s, 0.008s
```

## Interpretation

Reusable staging is a small full-runtime win and a large microbenchmark win. Keep it as an
opt-in primitive for future native loader work. It is not enough by itself because the
runtime still spends most miss time reconstructing tensors and copying parameters through
PyTorch.
