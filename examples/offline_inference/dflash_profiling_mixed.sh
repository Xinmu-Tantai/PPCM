#!/usr/bin/env bash
set -euo pipefail

ATTENTION_BACKEND="${1:-}"
ATTN_ARGS=()
if [[ -n "${ATTENTION_BACKEND}" ]]; then
  ATTN_ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
fi

python examples/offline_inference/dflash_profiling.py \
  --mode both \
  --tp-sizes 1 2 4 8 \
  --batch-sizes 1 2 4 8 16 \
  --num-runs 2 \
  --num-warmup-runs 1 \
  "${ATTN_ARGS[@]}"