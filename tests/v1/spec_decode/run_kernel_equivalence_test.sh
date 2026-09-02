#!/usr/bin/env bash
set -euo pipefail

DEBUG_DIR="${DFLASH_KERNEL_DEBUG_DIR:-/tmp/debug_kernel}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

OPTIMUS_SRC="/home/i-hulanxiang/workspace/optimus_jit_local/src"
if [[ -d "${OPTIMUS_SRC}" ]]; then
  export PYTHONPATH="${OPTIMUS_SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi

mkdir -p "${DEBUG_DIR}"
export DFLASH_KERNEL_DEBUG_DIR="${DEBUG_DIR}"

cd /home/i-hulanxiang/workspace/vllm-parallel-drafting

"${PYTHON_BIN}" -m pytest -q \
  tests/v1/spec_decode/test_optimus_kernel_integration.py \
  -k "optimus_root_posterior_matches_dense_sdpa_debug_dump or optimus_full_tree_matches_dense_sdpa_debug_dump" \
  -s

echo
echo "Debug artifacts:"
echo "  ${DEBUG_DIR}/optimus_root_posterior_equivalence.json"
echo "  ${DEBUG_DIR}/optimus_root_posterior_equivalence.pt"
echo "  ${DEBUG_DIR}/optimus_full_tree_equivalence.json"
echo "  ${DEBUG_DIR}/optimus_full_tree_equivalence.pt"
