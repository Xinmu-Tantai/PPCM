#!/usr/bin/env bash
set -euo pipefail

DEBUG_DIR="${DFLASH_D0_DEBUG_DIR:-/tmp/debug_d0_$(date +%Y%m%d_%H%M%S)}"
DEFAULT_VENV_PYTHON="/home/i-hulanxiang/workspace/vllm-parallel-drafting/vllm_jacobi_venv/bin/python"
if [[ -x "${DEFAULT_VENV_PYTHON}" ]]; then
  DEFAULT_PYTHON_BIN="${DEFAULT_VENV_PYTHON}"
elif command -v python >/dev/null 2>&1; then
  DEFAULT_PYTHON_BIN="$(command -v python)"
else
  DEFAULT_PYTHON_BIN="$(command -v python3)"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
DRAFT_MODEL="${DRAFT_MODEL:-/mnt/specdec-dev/checkpoints/specforge/outputs/nemotron-780k-and-codealpaca20k-v2-causal-distill-lr1e-4-anchorcnt512/epoch_6_step_583488}"
TARGET_MODEL="${TARGET_MODEL:-/data/models/Qwen3-8B}"
TREE_ATTN_KERNEL="${TREE_ATTN_KERNEL:-optimus}"
SAMPLE_INDICES="${SAMPLE_INDICES:-0,1}"
MAX_DRAFT_PASSES="${MAX_DRAFT_PASSES:-0}"

export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

OPTIMUS_SRC="/home/i-hulanxiang/workspace/optimus_jit_local/src"
if [[ "${TREE_ATTN_KERNEL}" == "optimus" && -d "${OPTIMUS_SRC}" ]]; then
  export PYTHONPATH="${OPTIMUS_SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi

mkdir -p "${DEBUG_DIR}"

cd /home/i-hulanxiang/workspace/vllm-parallel-drafting

"${PYTHON_BIN}" tests/v1/spec_decode/run_d0_two_prompt_diagnostic.py \
  --debug-dir "${DEBUG_DIR}" \
  --python-bin "${PYTHON_BIN}" \
  --model "${TARGET_MODEL}" \
  --draft-model "${DRAFT_MODEL}" \
  --sample-indices "${SAMPLE_INDICES}" \
  --tree-attn-kernel "${TREE_ATTN_KERNEL}" \
  --max-draft-passes "${MAX_DRAFT_PASSES}"

echo
echo "D0 diagnostic artifacts:"
echo "  ${DEBUG_DIR}/d0_acceptance_summary.json"
