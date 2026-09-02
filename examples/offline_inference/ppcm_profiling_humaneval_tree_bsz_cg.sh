#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
TP_SIZE="${TP_SIZE:-1}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profiler-dir)             PROFILER_DIR="$2"; shift 2 ;;
    --tp-size)                  TP_SIZE="$2"; shift 2 ;;
    --batch-size)               BATCH_SIZE="$2"; shift 2 ;;
    --max-num-seqs)            MAX_NUM_SEQS="$2"; shift 2 ;;
    *)                         EXTRA_ARGS+=("$1"); shift ;;
  esac
done

MAX_NUM_SEQS="${MAX_NUM_SEQS:-${BATCH_SIZE}}"
PROFILER_DIR="${PROFILER_DIR:-/path/to/output/ppcm-bsz${BATCH_SIZE}-cg-smoke-$(date +%m%d)}"

exec "${SCRIPT_DIR}/ppcm_profiling_humaneval_tree_unit_kvlayout.sh" \
  --profiler-dir "${PROFILER_DIR}" \
  --tp-size "${TP_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-samples 4 \
  --max-tokens 256 \
  --num-warmup-runs 0 \
  --profiler none \
  "${EXTRA_ARGS[@]}"
