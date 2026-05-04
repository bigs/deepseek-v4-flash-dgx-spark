# DeepSeek V4 Flash on DGX Spark

Research code for running `deepseek-ai/DeepSeek-V4-Flash` on a single NVIDIA DGX Spark
in its native FP8/FP4 hybrid checkpoint format.

The project goal is not to wrap a general-purpose inference framework. It is to learn
how far a Spark can be pushed by keeping the model format native, keeping KV cache
persistent, streaming or lazily materializing routed experts, and replacing framework
fallbacks with Spark/GB10-specific kernels where measurement says that is warranted.

## Current Status

Working baseline:

- metadata inspection for the Hugging Face checkpoint
- weight manifest mapping from HF tensor names to the official DeepSeek inference layout
- correctness-first lazy official runtime
- OpenAI-compatible MVP server with `/v1/models`, `/v1/completions`, and
  `/v1/chat/completions`
- one active persistent session cache with deferred postfill lookahead
- guarded Docker runner to keep failed CUDA experiments from exhausting host memory
- structured JSONL telemetry, NVTX ranges, optional CUDA event timers, and per-request
  API metrics
- checked-in experiment logs and summaries under `docs/experiments/` and `results/`
- current best full-layout 32-token decode recipe: native packed loader, native CUDA
  materializer, and 256-slot expert arena at 16.239s for 32 tokens on the measured Spark

Known limitations:

- decode is currently very slow; this repo is at the measurement-and-optimization stage
- request handling is single-worker FIFO, not continuous batching
- sampling is effectively greedy-first
- the sparse attention path is a correctness fallback until a GB10 kernel replaces it
- installation is still operator-oriented rather than a polished release flow

## Hardware Target

Primary target:

- NVIDIA DGX Spark
- GB10 Grace Blackwell
- 128 GB unified memory
- local NVMe/flash storage
- CUDA 13.x era drivers/toolchain

The runtime is intentionally Spark-specific in places. It should still be useful as a
reference for other constrained unified-memory GPU systems, but the experiments and
guardrails are designed around DGX Spark behavior.

## Repository Layout

```text
spark_runtime/              Runtime, server, worker, telemetry, and lazy model path
scripts/                    Inspection, probing, benchmarking, and guarded run scripts
docker/                     Runtime container definition
docs/                       Research notes and design docs
docs/experiments/           Experiment writeups
results/                    Captured result artifacts and raw logs
tests/                      Local unit tests for server and telemetry behavior
```

Start with:

- `docs/initial-research.md`
- `docs/experiment-plan.md`
- `docs/serving-design.md`
- `docs/instrumentation.md`
- `docs/optimization-report.md`
- `spark_runtime/README.md`

## Local Development

This repo is a `uv` project.

```bash
uv sync
uv run pytest
uv run ruff check spark_runtime tests scripts
```

The local tests use fake engines and do not require the model checkpoint or a GPU.

## Model Download

On the Spark or another machine with enough disk:

```bash
mkdir -p ~/models/deepseek-v4-flash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir ~/models/deepseek-v4-flash/hf \
  --local-dir-use-symlinks False
```

The runtime expects the model to be mounted at `/model` in the container unless
overridden with environment variables.

## Container

Build the current runtime image:

```bash
docker build -f docker/Dockerfile.spark-runtime -t deepseek-v4-flash-spark:tilelang .
```

The Dockerfile currently layers `uv` and `tilelang` on top of the CUDA/vLLM image used
during the initial Spark probes.

## Manifest Generation

The lazy runtime needs a weight manifest that maps checkpoint tensors to the official
DeepSeek inference module names.

A typical flow is:

```bash
python3 scripts/inspect_checkpoint.py \
  --model-dir /model \
  --out-dir /repo/results/current

python3 scripts/build_weight_manifest.py \
  --tensor-csv /repo/results/current/checkpoint-tensors.csv \
  --inference-dir /model/inference \
  --config /model/inference/config.json \
  --max-seq-len 1048576 \
  --out-json /repo/results/current/weight-manifest.summary.json \
  --out-csv /repo/results/current/weight-manifest.csv
```

At runtime, set:

```bash
export DEEPSEEK_SPARK_MANIFEST_CSV=/repo/results/current/weight-manifest.csv
```

If unset, the runtime looks for `/repo/weight-manifest.csv`.

## Serving

Inside the container:

```bash
export DEEPSEEK_SPARK_MODEL_DIR=/model
export DEEPSEEK_SPARK_INFERENCE_DIR=/model/inference
export DEEPSEEK_SPARK_CONFIG=/model/inference/config.json
export DEEPSEEK_SPARK_MANIFEST_CSV=/repo/results/current/weight-manifest.csv
export DEEPSEEK_SPARK_MAX_SEQ_LEN=1048576
export DEEPSEEK_SPARK_POSTFILL_MODE=deferred
export DEEPSEEK_SPARK_PACKED_NATIVE_LOADER=1
export DEEPSEEK_SPARK_PACKED_NATIVE_PINNED_STAGING=1
export DEEPSEEK_SPARK_DIRECT_PARAM_COPY=1
export DEEPSEEK_SPARK_PARAM_COPY_NON_BLOCKING=1
export DEEPSEEK_SPARK_NATIVE_MATERIALIZER=1
export DEEPSEEK_SPARK_EXPERT_ARENA_SLOTS=256
export DEEPSEEK_SPARK_NATIVE_WITH_CUDA=1
export DEEPSEEK_SPARK_NATIVE_COPY_PLAN=1
export DEEPSEEK_SPARK_NATIVE_FUSED_MATERIALIZER=1
export DEEPSEEK_SPARK_NATIVE_CUDA_MEMCPY=1
export DEEPSEEK_SPARK_NATIVE_BUILD_DIR=/runs/torch-extensions/native-packed-loader

python3 -m uvicorn spark_runtime.server:app --host 127.0.0.1 --port 18080
```

Probe the HTTP API:

```bash
python3 scripts/probe_server_http.py --prompt Hello --max-tokens 1 --skip-chat
python3 scripts/probe_chat_until_stop.py --message "Hello, how are you?" --max-tokens 128
```

The server exposes:

- `GET /health`
- `GET /ready`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /spark/sessions/{session_id}/reset`

## Guarded Spark Runs

Large failed CUDA loads can consume enough unified memory to make the host difficult to
access. Use the guarded runner for serious experiments:

```bash
./scripts/guarded_docker_run.py \
  --name deepseek-smoke \
  --image deepseek-v4-flash-spark:tilelang \
  --log-file ~/runs/deepseek-v4-flash/smoke.log \
  --memory 104g \
  --memory-swap 112g \
  --min-mem-available-gib 16 \
  --timeout-seconds 1800 \
  --recover-nvidia-driver-on-exit \
  --mount ~/models/deepseek-v4-flash/hf:/model:ro \
  --mount "$PWD":/repo:ro \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env DEEPSEEK_SPARK_MANIFEST_CSV=/repo/results/current/weight-manifest.csv \
  --entrypoint python3 \
  -- /repo/scripts/probe_server_http.py --prompt Hello --max-tokens 1 --skip-chat
```

`--recover-nvidia-driver-on-exit` unloads and reloads the NVIDIA modules after the
container exits. That has been necessary on Spark after killed CUDA processes retain
unified memory.

## Telemetry

Runtime telemetry is controlled with environment variables:

```bash
export DEEPSEEK_SPARK_TELEMETRY_JSONL=/runs/telemetry.jsonl
export DEEPSEEK_SPARK_NVTX=1
export DEEPSEEK_SPARK_CUDA_EVENTS=0
export DEEPSEEK_SPARK_TELEMETRY_TOKEN_CAP=256
```

Use JSONL telemetry on every experiment. Enable CUDA event timing for controlled
measurements, because it synchronizes the device. Use NVTX ranges with Nsight Systems,
then use Nsight Compute only after a specific slow kernel family is identified.

See `docs/instrumentation.md` and `docs/experiments/009-telemetry-smoke.md`.

## Publishing Notes

The code and scripts avoid hardcoded private hostnames, SSH users, or local home paths.
Checked-in `results/` logs preserve experiment provenance and may contain local paths or
machine labels from the original run environment.

## Roadmap

Near-term optimization work:

- reduce the native CUDA materializer from several per-parameter copies to fewer larger
  transfers where the packed layout allows
- replace Python-threaded overlap attempts with native I/O/copy scheduling
- screen future packed-layout variants with longer route-trace replay before full
  inference
- prototype native FP4/FP8 expert kernels after weight movement and materialization are
  under control
- replace sparse attention fallback with a GB10-compatible kernel
- add stronger OpenAI-compatible request validation, cancellation, and queue controls
- expose service metrics once the runtime path stabilizes
