# Experiment 019: Packed Runtime Integration

Date: 2026-05-01

Host: `spark-66c9`

Goal: make `LazyRoutedExpert` service expert-cache misses from a packed layout.

## Change

Added an opt-in packed store behind:

```bash
DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT=/path/to/layout.json
```

When the layout contains a requested `(layer, expert)`, the runtime now:

1. reads the whole packed expert block with one fixed-offset `pread`;
2. reconstructs CPU tensor views by target parameter name;
3. applies the same existing transforms, including `reinterpret_int8_as_fp4`;
4. copies into the existing official `Expert` module parameters.

Missing experts fall back to the safetensors path, which lets us use route-filtered
layouts safely during experiments.

## Validation

Representative packed experts were compared against safetensors inside the runtime
container.

Checked experts:

- layer 0, expert 168
- layer 15, expert 11
- layer 42, expert 120

For each expert, all six tensors matched safetensors exactly:

- `w1.scale`
- `w1.weight`
- `w2.scale`
- `w2.weight`
- `w3.scale`
- `w3.weight`

Observed packed block read times for these examples were about `11-12 ms` per
`13,369,344` byte expert block.

## Next Step

Run the actual generation probe with `DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT` enabled and
compare against the previous `route-cache1024-t8` baseline.

