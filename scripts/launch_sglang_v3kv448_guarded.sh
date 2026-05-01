#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root on the DGX Spark host.

MODEL_DIR="${MODEL_DIR:-$HOME/models/deepseek-v4-flash/hf}"
OVERLAY_DIR="${OVERLAY_DIR:-$HOME/models/deepseek-v4-flash/overlay-deepseek-v3-sparse-kv448}"
RUN_DIR="${RUN_DIR:-$HOME/runs/deepseek-v4-flash}"
IMAGE="${IMAGE:-lmsysorg/sglang:spark}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
MEMORY="${MEMORY:-72g}"
MEMORY_SWAP="${MEMORY_SWAP:-80g}"
MIN_MEM_AVAILABLE_GIB="${MIN_MEM_AVAILABLE_GIB:-48}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"

mkdir -p "$RUN_DIR"

python3 scripts/evict_model_cache.py --model-dir "$MODEL_DIR" \
  >"$RUN_DIR/evict-before-sglang-v3kv448-guarded.log" 2>&1 || true

python3 scripts/make_model_overlay.py \
  --source "$MODEL_DIR" \
  --dest "$OVERLAY_DIR" \
  --model-type deepseek_v3 \
  --architecture DeepseekV3ForCausalLM \
  --set first_k_dense_replace=0 \
  --set moe_layer_freq=1 \
  --set qk_nope_head_dim=448 \
  --set v_head_dim=512 \
  --set kv_lora_rank=448 \
  --set n_group=8 \
  --set topk_group=4

exec python3 scripts/guarded_docker_run.py \
  --name deepseek-sglang-v3kv448-guarded \
  --image "$IMAGE" \
  --log-file "$RUN_DIR/sglang-v3kv448-guarded.log" \
  --memory "$MEMORY" \
  --memory-swap "$MEMORY_SWAP" \
  --min-mem-available-gib "$MIN_MEM_AVAILABLE_GIB" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --recover-nvidia-driver-on-exit \
  --mount "$OVERLAY_DIR:/model:ro" \
  --entrypoint python3 \
  -- -m sglang.launch_server \
    --model-path /model \
    --tokenizer-path /model \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port 30000 \
    --tp-size 1 \
    --context-length "$CONTEXT_LENGTH" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --max-running-requests 1 \
    --chunked-prefill-size 1024 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.35 \
    --disable-cuda-graph \
    --skip-server-warmup \
    --skip-tokenizer-init \
    --log-level debug
