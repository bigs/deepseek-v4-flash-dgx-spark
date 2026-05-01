# Experiment 004: Lazy Runtime First Tokens

Date: 2026-05-01  
Host: `spark-66c9`

Goal: produce real decoded tokens from `deepseek-ai/DeepSeek-V4-Flash` on a
single DGX Spark with the runtime configured for full 1M context.

## Runtime Shape

This uses the official DeepSeek-V4 inference architecture with local patches:

- `max_seq_len=1048576`
- official V4 model classes and generation-style forward path
- official TileLang kernels for activation quantization, FP8 GEMM, FP4 GEMM,
  FP4 activation quantization, and HC split/Sinkhorn
- `spark_runtime.fast_hadamard_transform` as a correctness fallback because the
  upstream extension did not build cleanly in the tested arm64 CUDA 13 image
- `spark_runtime.lazy_official_runtime` to keep routed experts off resident
  memory and materialize only selected FP4 experts during forward
- PyTorch sparse-attention fallback for the `64 heads x 512 dim` shape because
  the official TileLang sparse-attention kernel requests too much dynamic shared
  memory on this GB10 path

The runtime image is built from `docker/Dockerfile.spark-runtime` as:

```bash
docker build -f docker/Dockerfile.spark-runtime \
  -t deepseek-v4-flash-spark:tilelang .
```

## Evidence

Command:

```bash
./scripts/guarded_docker_run.py \
  --name lazy-generate-prompt-2tok \
  --image deepseek-v4-flash-spark:tilelang \
  --log-file ~/runs/deepseek-v4-flash/lazy-generate-prompt-2tok.log \
  --memory 104g \
  --memory-swap 112g \
  --timeout-seconds 5400 \
  --min-mem-available-gib 16 \
  --recover-nvidia-driver-on-exit \
  --mount ~/models/deepseek-v4-flash/hf:/model:ro \
  --mount "$PWD:/repo:ro" \
  --env PYTHONPATH=/repo/spark_runtime:/model/inference \
  --entrypoint python3 \
  -- /repo/scripts/probe_lazy_generate_token.py \
    --model-dir /model \
    --inference-dir /model/inference \
    --manifest-csv /repo/results/spark-66c9/weight-manifest.csv \
    --config /model/inference/config.json \
    --max-seq-len 1048576 \
    --prompt Hello \
    --max-new-tokens 2
```

Result:

```text
max_seq_len 1048576
load_counts {'loaded': 1559}
mem_after_load (54670131200, 130663661568)
prompt 'Hello'
input_token_ids [19923]
generated_token_ids [21133, 438]
generated_text 'World ='
prompt_plus_generated_text 'HelloWorld ='
logits_shape (1, 129280) torch.float32
mem_after_forward (42789363712, 130663661568)
docker exited return_code=0 reason=process exited
```

The Spark stayed reachable and the guard recovered NVIDIA driver memory after
the run.

Logs:

- `results/spark-66c9/lazy-generate-prompt-2tok.log`
- `results/spark-66c9/lazy-generate-prompt.log`
- `results/spark-66c9/lazy-generate-token.log`
- `results/spark-66c9/lazy-resident-load.log`
- `results/spark-66c9/official-kernel-smoke.log`

## Interpretation

This proves the first correctness milestone: tokens can be generated on the DGX
Spark with the model configured for full 1M context, without loading all routed
experts resident. The path is intentionally slow and is not yet an optimized
serving stack.

Immediate optimization targets:

- replace the PyTorch sparse-attention fallback with a GB10-compatible kernel,
- cache hot routed experts instead of evicting every expert immediately,
- avoid repeated full startup loads for iterative generation,
- add a streaming CLI/server wrapper around the lazy runtime,
- measure per-layer expert load time and decode time separately.
