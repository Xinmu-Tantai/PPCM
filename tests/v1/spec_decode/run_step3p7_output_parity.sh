#!/usr/bin/env bash
# Run the Step-3.7-Flash AR vs DFlash output parity check.
# Usage:
#   bash tests/v1/spec_decode/run_step3p7_output_parity.sh
#   NUM_PROMPTS=8 MAX_TOKENS=512 bash tests/v1/spec_decode/run_step3p7_output_parity.sh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)

MODEL="${STEP3P7_MODEL:-/mnt/lanxiangh/models/Step-3.7-Flash}"
DRAFT="${STEP3P7_DRAFT:-/mnt/lanxiangh/checkpoints/specforge/ptd-step3p7-fkl-200k-epoch6-3e-4-no-gamma}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_TREE_BUDGET="${MAX_TREE_BUDGET:-127}"
GPU_MEM="${GPU_MEMORY_UTILIZATION:-0.80}"
NUM_PROMPTS="${NUM_PROMPTS:-4}"
OUTPUT_JSON="${OUTPUT_JSON:-/tmp/step3p7_parity_$(date +%Y%m%d_%H%M%S).json}"

OPTIMUS_SRC="${OPTIMUS_SRC:-/root/workspace/optimus_jit_local/src}"
export PYTHONPATH="${REPO_ROOT}${OPTIMUS_SRC:+:${OPTIMUS_SRC}}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export GPU_MEMORY_UTILIZATION="$GPU_MEM"

echo "Model:        $MODEL"
echo "Draft:        $DRAFT"
echo "TP size:      $TP_SIZE"
echo "Max tokens:   $MAX_TOKENS"
echo "Num prompts:  $NUM_PROMPTS"
echo "Output JSON:  $OUTPUT_JSON"
echo

cd "$REPO_ROOT"
python tests/v1/spec_decode/test_step3p7_output_parity.py \
    --model "$MODEL" \
    --draft-model "$DRAFT" \
    --tp-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --max-tree-budget "$MAX_TREE_BUDGET" \
    --gpu-memory-utilization "$GPU_MEM" \
    --num-prompts "$NUM_PROMPTS" \
    --output-json "$OUTPUT_JSON"

echo "Parity check complete. Results: $OUTPUT_JSON"
