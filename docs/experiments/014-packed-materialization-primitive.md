# Experiment 014: Packed Materialization Primitive

Date: 2026-05-01

Host: `spark-66c9`

Goal: test whether an expert-cache miss can be serviced more cheaply as one packed expert
block transferred to CUDA, compared with six safetensor tensor fetches/transfers per
expert.

This is not yet the full runtime path. It is the smallest CUDA-facing primitive behind
the proposed packed runtime path.

## Change

Added `scripts/bench_packed_materialization.py`.

The script benchmarks selected experts in two modes:

- `safetensor_tensors_to_device`: `safe_open(...).get_tensor()` for each expert tensor,
  then transfer each tensor to CUDA.
- `packed_block_to_device`: build a packed expert file once, then read one fixed-offset
  block per expert and transfer the block to CUDA as one `uint8` buffer.

Each selected DeepSeek V4 Flash routed expert in this layer is `13,369,344` bytes:

- 3 scale tensors of `262,144` bytes each;
- 3 packed weight tensors of `4,194,304` bytes each.

The benchmark uses the top 16 experts from layer 15 in Experiment 012.

## Commands

Fresh-process safetensor path:

```bash
python3 scripts/guarded_docker_run.py \
  --name packed-materialize-exp014-safe \
  --image deepseek-v4-flash-spark:tilelang \
  --log-file ~/runs/deepseek-v4-flash/packed-materialize-exp014-safe.log \
  --memory 32g \
  --memory-swap 40g \
  --min-mem-available-gib 32 \
  --timeout-seconds 600 \
  --mount ~/models/deepseek-v4-flash/hf:/model:ro \
  --mount ~/deepseek-v4-flash-dgx-spark:/repo:ro \
  --mount ~/runs/deepseek-v4-flash:/runs \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 \
  --recover-nvidia-driver-on-exit \
  -- /repo/scripts/bench_packed_materialization.py \
      --model-dir /model \
      --manifest-csv /repo/results/spark-66c9/weight-manifest.csv \
      --layer 15 \
      --experts 207,92,156,6,71,27,145,60,11,24,172,49,45,240,28,152 \
      --out-json /runs/packed-materialize-exp014-safe.json \
      --repeat 1 \
      --device cuda \
      --method safetensor_tensors_to_device
```

Fresh-process packed-block path:

```bash
python3 scripts/guarded_docker_run.py \
  --name packed-materialize-exp014-packed \
  --image deepseek-v4-flash-spark:tilelang \
  --log-file ~/runs/deepseek-v4-flash/packed-materialize-exp014-packed.log \
  --memory 32g \
  --memory-swap 40g \
  --min-mem-available-gib 32 \
  --timeout-seconds 600 \
  --mount ~/models/deepseek-v4-flash/hf:/model:ro \
  --mount ~/deepseek-v4-flash-dgx-spark:/repo:ro \
  --mount ~/runs/deepseek-v4-flash:/runs \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 \
  --recover-nvidia-driver-on-exit \
  -- /repo/scripts/bench_packed_materialization.py \
      --model-dir /model \
      --manifest-csv /repo/results/spark-66c9/weight-manifest.csv \
      --layer 15 \
      --experts 207,92,156,6,71,27,145,60,11,24,172,49,45,240,28,152 \
      --out-json /runs/packed-materialize-exp014-packed.json \
      --repeat 1 \
      --device cuda \
      --method packed_block_to_device
```

## Artifacts

- `results/spark-66c9/packed-materialize-exp014-safe.json`
- `results/spark-66c9/packed-materialize-exp014-safe.log`
- `results/spark-66c9/packed-materialize-exp014-packed.json`
- `results/spark-66c9/packed-materialize-exp014-packed.log`

Packed binary files were not copied into the repo; they are generated artifacts and were
`204 MiB` per run.

## Results

Both modes transfer the same useful payload for 16 experts: `213,909,504` bytes
(`204 MiB`).

| Method | Elapsed | Throughput |
| --- | ---: | ---: |
| safetensor tensors to CUDA | 2.375s | 85.9 MiB/s |
| packed block to CUDA | 0.367s | 556.1 MiB/s |

The packed-block primitive was `6.5x` faster in this fresh-process test.

## Interpretation

This is the first strong runtime-facing evidence that packed expert misses are worth
building, even though Experiment 002 showed the Hugging Face layout is not catastrophically
fragmented for raw reads.

The safetensor path pays for:

- opening/using safetensor handles;
- six separate tensor materializations per expert;
- six separate CPU-to-CUDA transfers per expert;
- dtype-aware tensor construction before transfer.

The packed path pays for:

- one fixed-offset read per expert;
- one CUDA transfer per expert block.

This benchmark still uses Python and a temporary `bytearray`, so it is not the final fast
path. A C++/CUDA or pinned-host implementation should improve it further.

## Next Step

Wire this primitive into `LazyRoutedExpert` behind an opt-in packed-layout path:

1. Add a layout file mapping `(layer, expert)` to packed file offsets and tensor slices.
2. Read one expert block into pinned host staging.
3. Slice the block into official parameter tensors.
4. Copy parameters to the existing official Expert module.
5. Re-run Experiment 011/012 with packed misses plus expert cache.

The expected combination is now clear: packed layout reduces miss cost, and the expert
cache reduces the number of misses.

