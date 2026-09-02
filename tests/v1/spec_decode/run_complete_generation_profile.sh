#!/usr/bin/env bash
set -euo pipefail

DEBUG_DIR="${DFLASH_COMPLETE_DEBUG_DIR:-/tmp/debug_complete_$(date +%Y%m%d_%H%M%S)}"
DEFAULT_VENV_PYTHON="/home/i-hulanxiang/workspace/vllm-parallel-drafting/vllm_jacobi_venv/bin/python"
if [[ -x "${DEFAULT_VENV_PYTHON}" ]]; then
  DEFAULT_PYTHON_BIN="${DEFAULT_VENV_PYTHON}"
elif command -v python >/dev/null 2>&1; then
  DEFAULT_PYTHON_BIN="$(command -v python)"
else
  DEFAULT_PYTHON_BIN="$(command -v python3)"
fi
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
REFERENCE_PYTHON_BIN="${REFERENCE_PYTHON_BIN:-${VLLM_PYTHON_BIN}}"
DRAFT_MODEL="${DRAFT_MODEL:-/mnt/specdec-dev/checkpoints/specforge/outputs/nemotron-780k-and-codealpaca20k-v2-causal-distill-lr1e-4-anchorcnt512/epoch_6_step_583488}"
TARGET_MODEL="${TARGET_MODEL:-/data/models/Qwen3-8B}"
TREE_ATTN_KERNEL="${TREE_ATTN_KERNEL:-optimus}"
SAMPLE_INDICES="${SAMPLE_INDICES:-0,1}"

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

"${REFERENCE_PYTHON_BIN}" \
  /home/i-hulanxiang/workspace/dflash/tests/drafting_mechanism/run_complete_generation_profile.py \
  --debug-dir "${DEBUG_DIR}/reference" \
  --model "${TARGET_MODEL}" \
  --draft-model "${DRAFT_MODEL}" \
  --sample-indices "${SAMPLE_INDICES}" \
  --attn-implementation sdpa \
  --tree-attn sdpa \
  --tree-draft crossproduct \
  --max-draft-passes 1

"${VLLM_PYTHON_BIN}" \
  /home/i-hulanxiang/workspace/vllm-parallel-drafting/tests/v1/spec_decode/run_complete_generation_profile.py \
  --debug-dir "${DEBUG_DIR}/vllm" \
  --model "${TARGET_MODEL}" \
  --draft-model "${DRAFT_MODEL}" \
  --sample-indices "${SAMPLE_INDICES}" \
  --enforce-eager \
  --tree-attn-kernel "${TREE_ATTN_KERNEL}" \
  --tree-draft accum_logp \
  --tree-construction breadth_first

"${VLLM_PYTHON_BIN}" \
  /home/i-hulanxiang/workspace/vllm-parallel-drafting/tests/v1/spec_decode/compare_complete_generation_profile.py \
  --reference-summary "${DEBUG_DIR}/reference/reference_complete_generation_summary.json" \
  --vllm-summary "${DEBUG_DIR}/vllm/vllm_complete_generation_summary.json" \
  --output-json "${DEBUG_DIR}/complete_generation_compare.json"

echo
echo "Complete-generation artifacts:"
echo "  ${DEBUG_DIR}/reference/reference_complete_generation_summary.json"
echo "  ${DEBUG_DIR}/vllm/vllm_complete_generation_summary.json"
echo "  ${DEBUG_DIR}/complete_generation_compare.json"
