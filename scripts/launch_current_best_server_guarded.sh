#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root on a DGX Spark host.
# Starts the OpenAI-compatible runtime server with the current best measured recipe.

REPO_DIR="${REPO_DIR:-$PWD}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/deepseek-v4-flash/hf}"
RUN_DIR="${RUN_DIR:-$HOME/runs/deepseek-v4-flash}"
IMAGE="${IMAGE:-deepseek-v4-flash-spark:tilelang}"
CONTAINER_NAME="${CONTAINER_NAME:-deepseek-v4-flash-current-best}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
PUBLISH="${PUBLISH:-127.0.0.1:$PORT:$PORT}"
MEMORY="${MEMORY:-112g}"
MEMORY_SWAP="${MEMORY_SWAP:-120g}"
MIN_MEM_AVAILABLE_GIB="${MIN_MEM_AVAILABLE_GIB:-12}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-8}"
POSTFILL_DELAY_SECONDS="${POSTFILL_DELAY_SECONDS:-0.25}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1048576}"
EXPERT_CACHE_ENTRIES="${EXPERT_CACHE_ENTRIES:-1024}"
EXPERT_CACHE_LAYER_COUNT="${EXPERT_CACHE_LAYER_COUNT:-43}"
EXPERT_ARENA_SLOTS="${EXPERT_ARENA_SLOTS:-256}"
PACKED_LAYOUT_CONTAINER="${PACKED_LAYOUT_CONTAINER:-/runs/packed-full-e018/layout.json}"
MANIFEST_CSV_CONTAINER="${MANIFEST_CSV_CONTAINER:-/repo/results/spark-66c9/weight-manifest.csv}"
TELEMETRY_JSONL_CONTAINER="${TELEMETRY_JSONL_CONTAINER:-/runs/current-best-server.jsonl}"
NATIVE_BUILD_DIR_CONTAINER="${NATIVE_BUILD_DIR_CONTAINER:-/runs/torch-extensions/native-packed-loader-current-best}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/current-best-server.guard.log}"

mkdir -p "$RUN_DIR"

exec python3 "$REPO_DIR/scripts/guarded_docker_run.py" \
  --name "$CONTAINER_NAME" \
  --image "$IMAGE" \
  --log-file "$LOG_FILE" \
  --memory "$MEMORY" \
  --memory-swap "$MEMORY_SWAP" \
  --min-mem-available-gib "$MIN_MEM_AVAILABLE_GIB" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --recover-nvidia-driver-on-exit \
  --mount "$REPO_DIR:/repo:ro" \
  --mount "$MODEL_DIR:/model:ro" \
  --mount "$RUN_DIR:/runs" \
  --publish "$PUBLISH" \
  --env PYTHONPATH=/repo:/repo/spark_runtime:/model/inference \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env DEEPSEEK_SPARK_MODEL_DIR=/model \
  --env DEEPSEEK_SPARK_INFERENCE_DIR=/model/inference \
  --env DEEPSEEK_SPARK_CONFIG=/model/inference/config.json \
  --env DEEPSEEK_SPARK_MANIFEST_CSV="$MANIFEST_CSV_CONTAINER" \
  --env DEEPSEEK_SPARK_MAX_SEQ_LEN="$MAX_SEQ_LEN" \
  --env DEEPSEEK_SPARK_POSTFILL_MODE=deferred \
  --env DEEPSEEK_SPARK_MAX_QUEUE_SIZE="$MAX_QUEUE_SIZE" \
  --env DEEPSEEK_SPARK_POSTFILL_DELAY_SECONDS="$POSTFILL_DELAY_SECONDS" \
  --env DEEPSEEK_SPARK_TELEMETRY_JSONL="$TELEMETRY_JSONL_CONTAINER" \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_ENTRIES="$EXPERT_CACHE_ENTRIES" \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_POLICY=layer_lru \
  --env DEEPSEEK_SPARK_EXPERT_CACHE_LAYER_COUNT="$EXPERT_CACHE_LAYER_COUNT" \
  --env DEEPSEEK_SPARK_PACKED_EXPERT_LAYOUT="$PACKED_LAYOUT_CONTAINER" \
  --env DEEPSEEK_SPARK_PACKED_NATIVE_LOADER=1 \
  --env DEEPSEEK_SPARK_PACKED_NATIVE_PINNED_STAGING=1 \
  --env DEEPSEEK_SPARK_DIRECT_PARAM_COPY=1 \
  --env DEEPSEEK_SPARK_PARAM_COPY_NON_BLOCKING=1 \
  --env DEEPSEEK_SPARK_NATIVE_MATERIALIZER=1 \
  --env DEEPSEEK_SPARK_EXPERT_ARENA_SLOTS="$EXPERT_ARENA_SLOTS" \
  --env DEEPSEEK_SPARK_NATIVE_WITH_CUDA=1 \
  --env DEEPSEEK_SPARK_NATIVE_COPY_PLAN=1 \
  --env DEEPSEEK_SPARK_NATIVE_FUSED_MATERIALIZER=1 \
  --env DEEPSEEK_SPARK_NATIVE_CUDA_MEMCPY=1 \
  --env DEEPSEEK_SPARK_NATIVE_BUILD_DIR="$NATIVE_BUILD_DIR_CONTAINER" \
  --entrypoint python3 \
  -- -m uvicorn spark_runtime.server:app --host "$SERVER_HOST" --port "$PORT"
