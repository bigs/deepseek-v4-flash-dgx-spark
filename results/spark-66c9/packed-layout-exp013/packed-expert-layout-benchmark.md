# Packed Expert Layout Benchmark

- layers: [15, 18, 39]
- top experts per layer: 16
- evict before iterations: True

| Scenario | Ranges | MiB | Median s | Median GiB/s |
| --- | ---: | ---: | ---: | ---: |
| `layer15.hf_exact` | 96 | 204.0 | 0.0671 | 2.97 |
| `layer15.hf_coalesced` | 26 | 204.0 | 0.1225 | 1.63 |
| `layer15.packed` | 16 | 204.0 | 0.0666 | 2.99 |
| `layer18.hf_exact` | 96 | 204.0 | 0.1536 | 1.30 |
| `layer18.hf_coalesced` | 30 | 204.0 | 0.1029 | 1.94 |
| `layer18.packed` | 16 | 204.0 | 0.0659 | 3.02 |
| `layer39.hf_exact` | 96 | 204.0 | 0.1354 | 1.47 |
| `layer39.hf_coalesced` | 28 | 204.0 | 0.1309 | 1.52 |
| `layer39.packed` | 16 | 204.0 | 0.0651 | 3.06 |
