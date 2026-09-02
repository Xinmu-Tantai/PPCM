#!/usr/bin/env bash
set -euo pipefail

DEBUG_DIR="/tmp/debug_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEBUG_DIR"
DRAFT_MODEL="/mnt/specdec-dev/checkpoints/specforge/outputs/nemotron-780k-and-codealpaca20k-v2-causal-distill-lr1e-4-anchorcnt512/epoch_6_step_583488"
TARGET_MODEL="/data/models/Qwen3-8B"
TREE_ATTN_KERNEL="${TREE_ATTN_KERNEL:-optimus}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

OPTIMUS_SRC="/home/i-hulanxiang/workspace/optimus_jit_local/src"
if [[ "${TREE_ATTN_KERNEL}" == "optimus" && -d "${OPTIMUS_SRC}" ]]; then
  export PYTHONPATH="${OPTIMUS_SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi

cd /home/i-hulanxiang/workspace/dflash &&
"${PYTHON_BIN}" tests/drafting_mechanism/test_runtime_step0_bundle.py \
  --debug-dir "$DEBUG_DIR" \
  --model "$TARGET_MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --attn-implementation sdpa \
  --tree-attn sdpa \
  --prompt-set humaneval \
  --sample-index 0

cd /home/i-hulanxiang/workspace/vllm-parallel-drafting &&
"${PYTHON_BIN}" tests/v1/spec_decode/test_dflash_runtime_bundle.py \
  --debug-dir "$DEBUG_DIR" \
  --model "$TARGET_MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --enforce-eager \
  --tree-attn-kernel "$TREE_ATTN_KERNEL" \
  --prompt-set humaneval \
  --sample-index 0

"${PYTHON_BIN}" tests/v1/spec_decode/compare_dflash_runtime_bundle.py \
  --reference-bundle "$DEBUG_DIR" \
  --vllm-bundle "$DEBUG_DIR" \
  --output-json "$DEBUG_DIR/runtime_compare.json"