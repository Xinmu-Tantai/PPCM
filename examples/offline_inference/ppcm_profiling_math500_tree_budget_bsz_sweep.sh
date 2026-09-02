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

TARGET_MODEL="${TARGET_MODEL:-/path/to/target-model}"
DRAFT_MODEL="${DRAFT_MODEL:-/path/to/Qwen3-8B-PPCM}"
TREE_ATTN_KERNEL="${TREE_ATTN_KERNEL:-optimus}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
OPTIMUS_SRC="${OPTIMUS_SRC:-}"

BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
TREE_BUDGETS="${TREE_BUDGETS:-16 32 64 128 256}"
TP_SIZE="${TP_SIZE:-1}"
TREE_KV_LAYOUT="${TREE_KV_LAYOUT:-logical}"
PROFILER_DIR="${PROFILER_DIR:-}"
CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-0}"
RESUME_COMPLETED="${RESUME_COMPLETED:-1}"
RUN_AR=1
RUN_DFLASH=1

TREE_WIDTH=7
TREE_DEPTH=16
NUM_CUDAGRAPH_TREE_CAPTURES="${NUM_CUDAGRAPH_TREE_CAPTURES:-4}"
TREE_DRAFT_MODE="${TREE_DRAFT_MODE:-accum_logp}"
ADDITIONAL_DRAFT_REFINEMENT_PASSES="${ADDITIONAL_DRAFT_REFINEMENT_PASSES:-0}"
TREE_PRUNE_RATIO="${TREE_PRUNE_RATIO:-0.25}"
TREE_CONSTRUCTION="${TREE_CONSTRUCTION:-breadth_first}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-51200}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NUM_RUNS="${NUM_RUNS:-1}"
NUM_WARMUP_RUNS="${NUM_WARMUP_RUNS:-1}"
PROFILER="${PROFILER:-none}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-default}"
SUMMARY_METRIC="${SUMMARY_METRIC:-e2e_throughput_tok_s}"
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra dflash_profiling.py args]

Runs Math-500 full-set vLLM DFlash sweeps over batch sizes and tree budgets.

Options:
  --model PATH                 Target model path (default: ${TARGET_MODEL})
  --draft-model PATH           DFlash draft model path (default: ${DRAFT_MODEL})
  --profiler-dir DIR           Output root (default: /path/to/output/...)
  --batch-sizes "1 2 ..."      Batch sizes to sweep (default: "${BATCH_SIZES}")
  --tree-budgets "16 32 ..."   Tree budgets to sweep (default: "${TREE_BUDGETS}")
  --tp-size N                  Tensor parallel size (default: ${TP_SIZE})
  --tree-kv-layout LAYOUT      physical or logical (default: ${TREE_KV_LAYOUT})
  --tree-attn-kernel KERNEL    optimus or triton (default: ${TREE_ATTN_KERNEL})
  --attention-backend NAME     vLLM attention backend (default: ${ATTENTION_BACKEND})
  --cuda-devices "0 1 ..."      CUDA devices to shard across (default: all visible GPUs)
  --parallel-workers N          Concurrent cells to run (default: floor(num_gpus / tp_size))
  --max-tokens N               Generation max tokens (default: ${MAX_TOKENS})
  --max-samples N              0 means full Math-500 set (default: ${MAX_SAMPLES})
  --num-runs N                 Timed runs per setting (default: ${NUM_RUNS})
  --num-warmup-runs N          Warmup batches per setting (default: ${NUM_WARMUP_RUNS})
  --profiler NAME              none, torch, or cuda (default: ${PROFILER})
  --cudagraph-mode MODE        vLLM CUDA graph mode (default: ${CUDAGRAPH_MODE})
  --summary-metric KEY         e2e_throughput_tok_s or benchmark_tok_s
  --no-resume                  Rerun cells even when metrics already exist
  --skip-ar                    Do not rerun AR baselines
  --skip-dflash                Do not rerun DFlash budget cells
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)                 TARGET_MODEL="$2"; shift 2 ;;
    --draft-model)           DRAFT_MODEL="$2"; shift 2 ;;
    --profiler-dir)          PROFILER_DIR="$2"; shift 2 ;;
    --batch-sizes)           BATCH_SIZES="$2"; shift 2 ;;
    --tree-budgets)          TREE_BUDGETS="$2"; shift 2 ;;
    --tp-size)               TP_SIZE="$2"; shift 2 ;;
    --tree-kv-layout)        TREE_KV_LAYOUT="$2"; shift 2 ;;
    --tree-attn-kernel)      TREE_ATTN_KERNEL="$2"; shift 2 ;;
    --attention-backend)     ATTENTION_BACKEND="$2"; shift 2 ;;
    --cuda-devices)          CUDA_DEVICE_LIST="$2"; shift 2 ;;
    --parallel-workers)      PARALLEL_WORKERS="$2"; shift 2 ;;
    --max-tokens)            MAX_TOKENS="$2"; shift 2 ;;
    --max-samples)           MAX_SAMPLES="$2"; shift 2 ;;
    --num-runs)              NUM_RUNS="$2"; shift 2 ;;
    --num-warmup-runs)       NUM_WARMUP_RUNS="$2"; shift 2 ;;
    --profiler)              PROFILER="$2"; shift 2 ;;
    --cudagraph-mode)        CUDAGRAPH_MODE="$2"; shift 2 ;;
    --summary-metric)        SUMMARY_METRIC="$2"; shift 2 ;;
    --no-resume)             RESUME_COMPLETED=0; shift ;;
    --skip-ar)               RUN_AR=0; shift ;;
    --skip-dflash)           RUN_DFLASH=0; shift ;;
    -h|--help)               usage; exit 0 ;;
    --)                      shift; EXTRA_ARGS+=("$@"); break ;;
    *)                       EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ "${TREE_KV_LAYOUT}" != "physical" && "${TREE_KV_LAYOUT}" != "logical" ]]; then
  echo "ERROR: --tree-kv-layout must be physical or logical"
  exit 1
fi

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
if [[ -z "${PROFILER_DIR}" ]]; then
  PROFILER_DIR="/path/to/output/ppcm-math500-${DATE_TAG}-${TREE_KV_LAYOUT}-${TREE_ATTN_KERNEL}"
fi

mkdir -p "$PROFILER_DIR"
RUN_LOG="${PROFILER_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1

if [[ "${TREE_ATTN_KERNEL}" == "optimus" && -n "${OPTIMUS_SRC}" && -d "${OPTIMUS_SRC}" ]]; then
  export PYTHONPATH="${OPTIMUS_SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi

read -r -a BATCH_SIZE_LIST <<< "${BATCH_SIZES//,/ }"
read -r -a TREE_BUDGET_LIST <<< "${TREE_BUDGETS//,/ }"

echo "Run log:            $RUN_LOG"
echo "Repo root:          $REPO_ROOT"
echo "HF datasets cache:  $HF_DATASETS_CACHE"
echo "Target model:       $TARGET_MODEL"
echo "Draft model:        $DRAFT_MODEL"
echo "Profiler dir:       $PROFILER_DIR"
echo "Prompt set:         math-500"
echo "Batch sizes:        ${BATCH_SIZES}"
echo "Tree budgets:       ${TREE_BUDGETS}"
echo "Tree KV layout:     ${TREE_KV_LAYOUT}"
echo "Tree attn kernel:   ${TREE_ATTN_KERNEL}"
echo "Max samples:        ${MAX_SAMPLES} (0 means full Math-500)"
echo "Max tokens:         ${MAX_TOKENS}"
echo "Profiler:           ${PROFILER}"
echo "CUDAGraph mode:     ${CUDAGRAPH_MODE}"
echo "CUDA home:          ${CUDA_HOME:-unset}"
echo "Python:             $(command -v python)"

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
    print(f"NVCC CUDA release:  {match.group(1) if match else 'unknown'}")
print(f"CUDA available:     {torch.cuda.is_available()}")
print(f"CUDA device count:  {torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("ERROR: PyTorch cannot initialize CUDA in this script environment")
print(f"CUDA device 0:      {torch.cuda.get_device_name(0)}")
PY

if [[ -z "${CUDA_DEVICE_LIST}" ]]; then
  CUDA_DEVICE_LIST="$(python - <<'PY'
import torch
print(" ".join(str(i) for i in range(torch.cuda.device_count())))
PY
)"
fi
read -r -a CUDA_DEVICE_ARRAY <<< "${CUDA_DEVICE_LIST//,/ }"
NUM_CUDA_DEVICES="${#CUDA_DEVICE_ARRAY[@]}"
if (( NUM_CUDA_DEVICES < 1 )); then
  echo "ERROR: no CUDA devices selected"
  exit 1
fi
if (( TP_SIZE < 1 )); then
  echo "ERROR: --tp-size must be >= 1"
  exit 1
fi
MAX_WORKER_GROUPS=$(( NUM_CUDA_DEVICES / TP_SIZE ))
if (( MAX_WORKER_GROUPS < 1 )); then
  echo "ERROR: selected CUDA devices (${CUDA_DEVICE_ARRAY[*]}) are fewer than TP_SIZE=${TP_SIZE}"
  exit 1
fi
if (( PARALLEL_WORKERS <= 0 )); then
  PARALLEL_WORKERS="${MAX_WORKER_GROUPS}"
elif (( PARALLEL_WORKERS > MAX_WORKER_GROUPS )); then
  echo "WARNING: capping --parallel-workers ${PARALLEL_WORKERS} to ${MAX_WORKER_GROUPS} for TP_SIZE=${TP_SIZE}"
  PARALLEL_WORKERS="${MAX_WORKER_GROUPS}"
fi
echo "CUDA devices:       ${CUDA_DEVICE_ARRAY[*]}"
echo "Parallel workers:   ${PARALLEL_WORKERS} (TP_SIZE=${TP_SIZE})"
echo "Resume completed:   ${RESUME_COMPLETED}"

cd "$REPO_ROOT"

run_profile() {
  local label="$1"
  local mode="$2"
  local batch_size="$3"
  local max_tree_budget="${4:-}"
  local run_dir="${PROFILER_DIR}/${label}"
  local cell_log="${run_dir}/cell_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p "$run_dir"

  local tree_args=()
  if [[ "${mode}" == "dflash" ]]; then
    tree_args=(
      --max-tree-budget "${max_tree_budget}"
      --tree-kv-layout "${TREE_KV_LAYOUT}"
    )
  fi

  echo ""
  echo "Running label=${label} mode=${mode} batch_size=${batch_size} max_tree_budget=${max_tree_budget:-n/a} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
  python examples/offline_inference/dflash_profiling.py \
    --prompt-set math-500 \
    --mode "${mode}" \
    --head-type causal \
    --model "${TARGET_MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --max-tokens "${MAX_TOKENS}" \
    --block-size "${TREE_DEPTH}" \
    --tree-width "${TREE_WIDTH}" \
    "${tree_args[@]}" \
    --tree-draft "${TREE_DRAFT_MODE}" \
    --max-draft-passes "${ADDITIONAL_DRAFT_REFINEMENT_PASSES}" \
    --tree-prune-ratio "${TREE_PRUNE_RATIO}" \
    --tree-construction "${TREE_CONSTRUCTION}" \
    --tree-attn-kernel "${TREE_ATTN_KERNEL}" \
    --num-cudagraph-tree-captures "${NUM_CUDAGRAPH_TREE_CAPTURES}" \
    --cudagraph-mode "${CUDAGRAPH_MODE}" \
    --attention-backend "${ATTENTION_BACKEND}" \
    --tp-sizes "${TP_SIZE}" \
    --batch-sizes "${batch_size}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-samples "${MAX_SAMPLES}" \
    --max-num-seqs "${batch_size}" \
    --num-runs "${NUM_RUNS}" \
    --num-warmup-runs "${NUM_WARMUP_RUNS}" \
    --profiler "${PROFILER}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    --torch-profiler-dir "${run_dir}" 2>&1 | sed -u "s/^/[${label}] /" | tee -a "$cell_log"
}

is_cell_complete() {
  local label="$1"
  local mode="$2"
  local batch_size="$3"
  local run_dir="${PROFILER_DIR}/${label}"
  [[ -s "${run_dir}/metrics_summary.txt" && -s "${run_dir}/${mode}/tp${TP_SIZE}/bs${batch_size}/metrics_report.txt" ]]
}

worker_cuda_devices() {
  local worker_idx="$1"
  local start=$(( worker_idx * TP_SIZE ))
  local devices=()
  local offset
  for ((offset = 0; offset < TP_SIZE; offset++)); do
    devices+=("${CUDA_DEVICE_ARRAY[$((start + offset))]}")
  done
  local joined="${devices[0]}"
  for ((offset = 1; offset < ${#devices[@]}; offset++)); do
    joined+=",${devices[$offset]}"
  done
  echo "$joined"
}

TASK_ROOT="${PROFILER_DIR}/.task_queue_$(date +%Y%m%d_%H%M%S)"
TASK_PENDING="${TASK_ROOT}/pending"
TASK_RUNNING="${TASK_ROOT}/running"
TASK_DONE="${TASK_ROOT}/done"
TASK_FAILED="${TASK_ROOT}/failed"
mkdir -p "$TASK_PENDING" "$TASK_RUNNING" "$TASK_DONE" "$TASK_FAILED"

TASK_COUNT=0
SKIP_COUNT=0
enqueue_task() {
  local label="$1"
  local mode="$2"
  local batch_size="$3"
  local max_tree_budget="${4:-}"
  if [[ "${RESUME_COMPLETED}" == "1" ]] && is_cell_complete "$label" "$mode" "$batch_size"; then
    echo "Skipping completed label=${label} mode=${mode} batch_size=${batch_size}"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return
  fi
  printf "%s|%s|%s|%s\n" "$label" "$mode" "$batch_size" "$max_tree_budget" \
    > "${TASK_PENDING}/$(printf "%04d" "$TASK_COUNT")_${label}.task"
  TASK_COUNT=$((TASK_COUNT + 1))
}

if [[ "${RUN_AR}" == "1" ]]; then
  for batch_size in "${BATCH_SIZE_LIST[@]}"; do
    enqueue_task "ar_bsz${batch_size}" "ar" "${batch_size}"
  done
fi

if [[ "${RUN_DFLASH}" == "1" ]]; then
  for batch_size in "${BATCH_SIZE_LIST[@]}"; do
    for budget in "${TREE_BUDGET_LIST[@]}"; do
      enqueue_task "budget${budget}_bsz${batch_size}_${TREE_KV_LAYOUT}" "dflash" "${batch_size}" "${budget}"
    done
  done
fi

echo "Queued cells:        ${TASK_COUNT}"
echo "Skipped completed:  ${SKIP_COUNT}"

run_worker() {
  local worker_idx="$1"
  local worker_devices="$2"
  local task_file=""
  local running_file=""
  local label=""
  local mode=""
  local batch_size=""
  local max_tree_budget=""

  export CUDA_VISIBLE_DEVICES="$worker_devices"
  echo "[worker ${worker_idx}] started CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

  while true; do
    task_file=""
    for candidate in "${TASK_PENDING}"/*.task; do
      [[ -e "$candidate" ]] || break
      running_file="${TASK_RUNNING}/$(basename "$candidate").worker${worker_idx}"
      if mv "$candidate" "$running_file" 2>/dev/null; then
        task_file="$running_file"
        break
      fi
    done
    [[ -n "$task_file" ]] || break

    IFS="|" read -r label mode batch_size max_tree_budget < "$task_file"
    echo "[worker ${worker_idx}] picked label=${label} mode=${mode} batch_size=${batch_size} max_tree_budget=${max_tree_budget:-n/a}"
    if run_profile "$label" "$mode" "$batch_size" "$max_tree_budget"; then
      mv "$task_file" "${TASK_DONE}/$(basename "$task_file")"
      echo "[worker ${worker_idx}] completed label=${label}"
    else
      mv "$task_file" "${TASK_FAILED}/$(basename "$task_file")"
      echo "[worker ${worker_idx}] FAILED label=${label}"
      return 1
    fi
  done

  echo "[worker ${worker_idx}] no more tasks"
}

if (( TASK_COUNT > 0 )); then
  WORKER_PIDS=()
  for ((worker_idx = 0; worker_idx < PARALLEL_WORKERS; worker_idx++)); do
    run_worker "$worker_idx" "$(worker_cuda_devices "$worker_idx")" &
    WORKER_PIDS+=("$!")
  done

  FAILED=0
  for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "$pid"; then
      FAILED=1
    fi
  done
  if (( FAILED != 0 )); then
    echo "ERROR: at least one worker failed. Failed task files are in ${TASK_FAILED}"
    exit 1
  fi
else
  echo "No cells to run; all requested cells are already complete."
fi

python examples/offline_inference/dflash_math500_sweep_summary.py \
  --profiler-dir "${PROFILER_DIR}" \
  --batch-sizes "${BATCH_SIZES}" \
  --tree-budgets "${TREE_BUDGETS}" \
  --tp-size "${TP_SIZE}" \
  --tree-kv-layout "${TREE_KV_LAYOUT}" \
  --metric "${SUMMARY_METRIC}"

echo ""
echo "Sweep outputs:"
echo "  CSV:   ${PROFILER_DIR}/run_summary.csv"
echo "  LaTeX: ${PROFILER_DIR}/run_summary_latex.tex"
