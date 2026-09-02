#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/path/to/hf-datasets-cache}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

if [[ -z "${CUDA_HOME:-}" || ! -d "${CUDA_HOME}" ]]; then
  if [[ -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
  fi
fi
export CUDA_PATH="${CUDA_HOME:-${CUDA_PATH:-}}"
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

prepend_ld_path() {
  local path="$1"
  if [[ -d "$path" && ":${LD_LIBRARY_PATH:-}:" != *":${path}:"* ]]; then
    export LD_LIBRARY_PATH="${path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}
prepend_ld_path "${CUDA_HOME:-}/lib64"
prepend_ld_path "/usr/local/cuda/compat/lib.real"
prepend_ld_path "/usr/local/cuda/compat/lib"
prepend_ld_path "/usr/local/nvidia/lib64"
prepend_ld_path "/usr/local/nvidia/lib"

# Defaults
TARGET_MODEL="${TARGET_MODEL:-/path/to/target-model}"
DRAFT_MODEL="${DRAFT_MODEL:-/path/to/Qwen3-8B-PPCM}"
TREE_ATTN_KERNEL="${TREE_ATTN_KERNEL:-optimus}"

# For causal draft tree with width > 1, attention backend is automatically tree_attn.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"
PROFILER_DIR="${PROFILER_DIR:-}"
EXTRA_ARGS=()
REPORT_BATCH_SIZE=1

# Parse named arguments. This wrapper owns --tree-kv-layout so physical and
# logical runs stay comparable and cannot be accidentally collapsed to one mode.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)              TARGET_MODEL="$2";       shift 2 ;;
    --draft-model)        DRAFT_MODEL="$2";        shift 2 ;;
    --tree-attn-kernel)   TREE_ATTN_KERNEL="$2";   shift 2 ;;
    --attention-backend)  ATTENTION_BACKEND="$2";  shift 2 ;;
    --profiler-dir)       PROFILER_DIR="$2";       shift 2 ;;
    --batch-sizes)        REPORT_BATCH_SIZE="$2";  EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
    --tree-kv-layout)     echo "ERROR: this comparison wrapper runs both physical and logical; do not pass --tree-kv-layout"; exit 1 ;;
    *)                    EXTRA_ARGS+=("$1");      shift   ;;
  esac
done

is_placeholder_path() {
  local path="$1"
  [[ -z "${path}" || "${path}" == /path/to/* ]]
}

require_configured_path() {
  local label="$1"
  local path="$2"
  local hint="$3"
  if is_placeholder_path "${path}"; then
    echo "ERROR: please specify ${label} with ${hint}."
    exit 1
  fi
}

require_configured_path "target model path" "${TARGET_MODEL}" "--model or TARGET_MODEL"
require_configured_path "causal draft head path" "${DRAFT_MODEL}" "--draft-model or DRAFT_MODEL"
require_configured_path "profiler output directory" "${PROFILER_DIR}" "--profiler-dir or PROFILER_DIR"

if [[ ! -d "$TARGET_MODEL" ]]; then
  echo "ERROR: TARGET_MODEL path does not exist: $TARGET_MODEL"
  exit 1
fi

if [[ ! -d "$DRAFT_MODEL" ]]; then
  echo "ERROR: DRAFT_MODEL path does not exist: $DRAFT_MODEL"
  exit 1
fi

DRAFT_TAG="$(basename "${DRAFT_MODEL}")"
DATE_TAG="$(date +%m%d)"

TREE_WIDTH=7
TREE_DEPTH=16
MAX_TREE_BUDGET=255
NUM_CUDAGRAPH_TREE_CAPTURES=4
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

TREE_DRAFT_MODE="accum_logp"
ADDITIONAL_DRAFT_REFINEMENT_PASSES=0
TREE_PRUNE_RATIO=0.25
TREE_CONSTRUCTION="breadth_first"

if [[ -z "${PROFILER_DIR}" ]]; then
  PROFILER_DIR="/path/to/output/ppcm-humaneval-${DATE_TAG}-kvlayout-compare-${TREE_ATTN_KERNEL}"
fi
mkdir -p "$PROFILER_DIR"
RUN_LOG="${PROFILER_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1
echo "Run log:            $RUN_LOG"

OPTIMUS_SRC="${OPTIMUS_SRC:-}"

if [[ "${TREE_ATTN_KERNEL}" == "optimus" && -n "${OPTIMUS_SRC}" && -d "${OPTIMUS_SRC}" ]]; then
  export PYTHONPATH="${OPTIMUS_SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "Repo root:          $REPO_ROOT"
echo "HF datasets cache:  $HF_DATASETS_CACHE"
echo "Target model:       $TARGET_MODEL"
echo "Draft model:        $DRAFT_MODEL"
echo "Profiler dir:       $PROFILER_DIR"
echo "Tree attn kernel:   $TREE_ATTN_KERNEL"
echo "GPU memory util:    $GPU_MEMORY_UTILIZATION"
if [[ -n "${OPTIMUS_SRC}" ]]; then
  echo "Optimus source:     $OPTIMUS_SRC"
fi
echo "CUDA home:          ${CUDA_HOME:-unset}"
echo "CUDA path:          ${CUDA_PATH:-unset}"
echo "CUDA compiler:      ${CUDACXX:-unset}"
echo "Python:             $(command -v python)"
echo "V1 multiprocessing: ${VLLM_ENABLE_V1_MULTIPROCESSING}"
echo "Worker mp method:   ${VLLM_WORKER_MULTIPROC_METHOD}"

python - <<'PY'
import re
import shutil
import subprocess
import torch

print(f"Torch:              {torch.__version__} (CUDA build {torch.version.cuda})")
nvcc = shutil.which("nvcc")
print(f"NVCC:               {nvcc or 'not found'}")
if nvcc:
    nvcc_output = subprocess.check_output([nvcc, "--version"], text=True)
    match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)?)", nvcc_output)
    nvcc_cuda = match.group(1) if match else "unknown"
    print(f"NVCC CUDA release:  {nvcc_cuda}")
    if torch.version.cuda and nvcc_cuda != "unknown":
        torch_cuda = ".".join(torch.version.cuda.split(".")[:2])
        if torch_cuda != nvcc_cuda:
            print(
                "WARNING: NVCC CUDA release does not match PyTorch CUDA build; "
                "source builds such as flash-attn may fail."
            )
print(f"CUDA available:     {torch.cuda.is_available()}")
print(f"CUDA device count:  {torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("ERROR: PyTorch cannot initialize CUDA in this script environment")
print(f"CUDA device 0:      {torch.cuda.get_device_name(0)}")
PY

cd "$REPO_ROOT"

run_profile() {
  local label="$1"
  local mode="$2"
  local tree_kv_layout="${3:-}"
  local run_dir="${PROFILER_DIR}/${label}"

  echo "Running profiling label=${label} mode=${mode} tree_kv_layout=${tree_kv_layout:-n/a}"
  local layout_args=()
  if [[ -n "${tree_kv_layout}" ]]; then
    layout_args=(--tree-kv-layout "${tree_kv_layout}")
  fi

  python examples/offline_inference/dflash_profiling.py \
    --prompt-set humaneval \
    --mode "${mode}" \
    --head-type causal \
    --model "${TARGET_MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --max-tokens 2048 \
    --block-size ${TREE_DEPTH} \
    --tree-width ${TREE_WIDTH} \
    --max-tree-budget ${MAX_TREE_BUDGET} \
    --tree-draft ${TREE_DRAFT_MODE} \
    --max-draft-passes ${ADDITIONAL_DRAFT_REFINEMENT_PASSES} \
    --tree-prune-ratio ${TREE_PRUNE_RATIO} \
    --tree-construction "${TREE_CONSTRUCTION}" \
    --tree-attn-kernel "${TREE_ATTN_KERNEL}" \
    "${layout_args[@]}" \
    --num-cudagraph-tree-captures ${NUM_CUDAGRAPH_TREE_CAPTURES} \
    --attention-backend "${ATTENTION_BACKEND}" \
    --tp-sizes 1 \
    --batch-sizes 1 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-batched-tokens 51200 \
    --max-samples 4 \
    --max-num-seqs 1 \
    --num-runs 1 \
    --num-warmup-runs 1 \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    --torch-profiler-dir "${run_dir}"
}

run_profile "ar" "ar"
run_profile "tree_physical" "dflash" "physical"
run_profile "tree_logical" "dflash" "logical"

echo ""
echo "Comparison outputs:"
echo "  AR:            ${PROFILER_DIR}/ar/ar/tp1/bs${REPORT_BATCH_SIZE}/metrics_report.txt"
echo "  Tree physical: ${PROFILER_DIR}/tree_physical/dflash/tp1/bs${REPORT_BATCH_SIZE}/metrics_report.txt"
echo "  Tree logical:  ${PROFILER_DIR}/tree_logical/dflash/tp1/bs${REPORT_BATCH_SIZE}/metrics_report.txt"
echo ""
echo "Note:"
echo "  The causal draft head is ${DRAFT_MODEL}."
echo "  The vLLM target model is ${TARGET_MODEL}."
echo "  Hugging Face datasets cache is ${HF_DATASETS_CACHE}."
