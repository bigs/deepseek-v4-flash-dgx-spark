# Experiment 034: Kernel Priority Budget

Date: 2026-05-02

Goal: evaluate whether broad custom GB10 kernels are the right next investment.

## Change

Added `scripts/analyze_decode_budget.py`.

The script parses telemetry and computes speed ceilings from measured decode components.

## Artifact

- `results/spark-66c9/kernel-budget-e034.json`

## Result

Input: E029 full-layout 32-token decode.

| Component | Seconds | Share |
| --- | ---: | ---: |
| Packed reads | 11.527s | 38.0% |
| Materialization | 17.106s | 56.4% |
| Residual / non-loader | 1.674s | 5.5% |
| Total decode | 30.307s | 100.0% |

Ceilings:

| Hypothetical | Tok/s |
| --- | ---: |
| Observed | 1.06 |
| Eliminate residual / non-loader time | 1.12 |
| Eliminate packed reads | 1.70 |
| Eliminate materialization | 2.42 |
| Halve reads and materialization | 2.00 |

Getting to 20 tok/s would require about a 19x decode speedup on this run.

## Interpretation

Broad custom kernels for ordinary matmul/attention cannot move the current short-decode
number much. The custom GB10 work should be:

1. native packed loader/materializer;
2. fused expert load/view/copy path;
3. GB10 sparse attention for long-context correctness/performance.

The sparse attention kernel is still important, but it is not the current largest decode
throughput bottleneck.
