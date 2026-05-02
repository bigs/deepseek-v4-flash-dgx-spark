# Packed Expert Layout Benchmark

- layers: [15, 18, 39]
- top experts per layer: 16
- evict before iterations: False

| Scenario | Ranges | MiB | Median s | Median GiB/s |
| --- | ---: | ---: | ---: | ---: |
| `layer15.hf_exact` | 96 | 204.0 | 0.0161 | 12.40 |
| `layer15.hf_coalesced` | 26 | 204.0 | 0.0505 | 3.94 |
| `layer15.packed` | 16 | 204.0 | 0.0146 | 13.65 |
| `layer18.hf_exact` | 96 | 204.0 | 0.0157 | 12.69 |
| `layer18.hf_coalesced` | 30 | 204.0 | 0.0170 | 11.73 |
| `layer18.packed` | 16 | 204.0 | 0.0146 | 13.60 |
| `layer39.hf_exact` | 96 | 204.0 | 0.0167 | 11.92 |
| `layer39.hf_coalesced` | 28 | 204.0 | 0.0505 | 3.95 |
| `layer39.packed` | 16 | 204.0 | 0.0146 | 13.65 |
