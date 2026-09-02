#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

PROFILER_DIR="${PROFILER_DIR:-/path/to/output/ppcm-math500-8gpu-resume}"
RESUME_COMPLETED="${RESUME_COMPLETED:-1}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume-completed) RESUME_COMPLETED=1; shift ;;
    --no-resume-completed) RESUME_COMPLETED=0; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

RESUME_COMPLETED="${RESUME_COMPLETED}" \
TREE_KV_LAYOUT=logical \
TREE_DRAFT_MODE=accum_logp \
"${SCRIPT_DIR}/ppcm_profiling_math500_tree_budget_bsz_sweep.sh" \
  --profiler-dir "${PROFILER_DIR}" \
  --cuda-devices "0 1 2 3 4 5 6 7" \
  --parallel-workers 8 \
  "${EXTRA_ARGS[@]}"
