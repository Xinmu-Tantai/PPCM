#!/usr/bin/env bash
# =============================================================================
# run_specforge_vllm_draft_parity.sh
#
# Direct SpecForge ↔ vLLM DFlash draft-logit parity test.
#
# Stage 1  (vllm_ptd_venv, TP=8):
#   Run vLLM in chain-spec mode with DFlash debug capture enabled at step 0.
#   Saves raw_target_hidden_states + draft_logits_req0 to a .pt file.
#
# Stage 2  (specforge_venv, single GPU):
#   Load the SAME raw_target_hidden_states from Stage 1.
#   Run SpecForge DFlashDraftModel forward with identical inputs.
#   Saves fc_output, fc_hidden_norm, draft_logits to a second .pt file.
#
# Stage 3  (any Python with torch):
#   compare_draft_logits.py loads both files, prints per-depth cosine /
#   top-1 match / KL report, and writes a JSON summary.
#
# Usage:
#   cd /root/workspace/vllm-parallel-drafting
#   bash tests/v1/spec_decode/run_specforge_vllm_draft_parity.sh
#
# Optional env overrides:
#   OUT_DIR                  output directory (default: /tmp/draft_parity_<ts>)
#   TARGET_MODEL             path to Step-3.7-Flash model
#   DRAFT_MODEL              path to DFlash draft checkpoint
#   TARGET_LM_HEAD_SHARD     path to safetensors shard with lm_head.weight
#   TP_SIZE                  tensor parallel size for vLLM (default: 8)
#   GPU_MEMORY_UTILIZATION   vLLM GPU mem util (default: 0.92)
#   VLLM_VENV                path to vllm_ptd_venv
#   SPECFORGE_VENV           path to specforge_venv
#   VLLM_ROOT                path to vllm-parallel-drafting repo
#   SPECFORGE_ROOT           path to specforge repo
#   SKIP_STAGE1              set to 1 to reuse existing vllm_draft_probe.pt
#   SKIP_STAGE2              set to 1 to reuse existing specforge_draft_probe.pt
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---- configurable paths / defaults ------------------------------------------
OUT_DIR="${OUT_DIR:-/tmp/draft_parity_probe_$(date +%Y%m%d_%H%M%S)}"

TARGET_MODEL="${TARGET_MODEL:-/mnt/lanxiangh/models/Step-3.7-Flash}"
DRAFT_MODEL="${DRAFT_MODEL:-/mnt/lanxiangh/checkpoints/specforge/ptd-step3p7-fkl-200k-epoch6-3e-4-no-gamma}"
TARGET_LM_HEAD_SHARD="${TARGET_LM_HEAD_SHARD:-${TARGET_MODEL}/model-00024.safetensors}"

TP_SIZE="${TP_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"   # block_size - 1 = 16 - 1

VLLM_VENV="${VLLM_VENV:-${REPO_ROOT}/vllm_ptd_venv}"
SPECFORGE_VENV="${SPECFORGE_VENV:-/root/workspace/specforge/specforge_venv}"
VLLM_ROOT="${VLLM_ROOT:-${REPO_ROOT}}"
SPECFORGE_ROOT="${SPECFORGE_ROOT:-/root/workspace/specforge}"

SKIP_STAGE1="${SKIP_STAGE1:-0}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"

VLLM_PROBE="${OUT_DIR}/vllm_draft_probe.pt"
SF_PROBE="${OUT_DIR}/specforge_draft_probe.pt"
REPORT="${OUT_DIR}/parity_report.json"

VLLM_PY="${VLLM_VENV}/bin/python"
SF_PY="${SPECFORGE_VENV}/bin/python"
CMP_PY="${VLLM_PY}"   # compare script uses only torch/json, either venv works

echo "================================================================"
echo " SpecForge ↔ vLLM DFlash Draft-Logit Parity Test"
echo "================================================================"
echo "  OUT_DIR  : ${OUT_DIR}"
echo "  TP       : ${TP_SIZE}"
echo "  spec     : ${NUM_SPECULATIVE_TOKENS}"
echo "  target   : ${TARGET_MODEL}"
echo "  draft    : ${DRAFT_MODEL}"
echo ""

mkdir -p "${OUT_DIR}"

# =============================================================================
# Stage 1 — vLLM probe
# =============================================================================
if [[ "${SKIP_STAGE1}" == "1" && -f "${VLLM_PROBE}" ]]; then
    echo "[stage1] Skipping (SKIP_STAGE1=1, file exists): ${VLLM_PROBE}"
else
    echo "[stage1] Running vLLM probe (TP=${TP_SIZE}) ..."
    PYTHONPATH="${VLLM_ROOT}:${PYTHONPATH:-}" \
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    "${VLLM_PY}" \
        "${SCRIPT_DIR}/probe_vllm_draft_logits.py" \
        --model "${TARGET_MODEL}" \
        --draft-model "${DRAFT_MODEL}" \
        --tp "${TP_SIZE}" \
        --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-51200}" \
        --enforce-eager \
        --disable-cascade-attn \
        --max-tokens 1 \
        --out "${VLLM_PROBE}" \
    2>&1 | tee "${OUT_DIR}/stage1_vllm_probe.log"

    if [[ ! -f "${VLLM_PROBE}" ]]; then
        echo "[stage1] ERROR: vllm probe file not created – check log above."
        exit 1
    fi
    echo "[stage1] Done → ${VLLM_PROBE}"
fi

# =============================================================================
# Stage 2 — SpecForge probe
# =============================================================================
if [[ "${SKIP_STAGE2}" == "1" && -f "${SF_PROBE}" ]]; then
    echo "[stage2] Skipping (SKIP_STAGE2=1, file exists): ${SF_PROBE}"
else
    echo "[stage2] Running SpecForge probe (single GPU) ..."
    # Use only the first GPU (the large target model was freed after stage 1).
    CUDA_VISIBLE_DEVICES="${SPECFORGE_GPU:-0}" \
    PYTHONPATH="${SPECFORGE_ROOT}:${VLLM_ROOT}:${PYTHONPATH:-}" \
    "${SF_PY}" \
        "${SCRIPT_DIR}/probe_specforge_draft_logits.py" \
        --vllm-probe "${VLLM_PROBE}" \
        --draft-model "${DRAFT_MODEL}" \
        --target-lm-head-shard "${TARGET_LM_HEAD_SHARD}" \
        --out "${SF_PROBE}" \
        --device "cuda:0" \
    2>&1 | tee "${OUT_DIR}/stage2_specforge_probe.log"

    if [[ ! -f "${SF_PROBE}" ]]; then
        echo "[stage2] ERROR: specforge probe file not created – check log above."
        exit 1
    fi
    echo "[stage2] Done → ${SF_PROBE}"
fi

# =============================================================================
# Stage 3 — Comparison
# =============================================================================
echo "[stage3] Comparing probe files ..."
PYTHONPATH="${VLLM_ROOT}:${PYTHONPATH:-}" \
"${CMP_PY}" \
    "${SCRIPT_DIR}/compare_draft_logits.py" \
    --specforge-probe "${SF_PROBE}" \
    --vllm-probe      "${VLLM_PROBE}" \
    --out             "${REPORT}" \
    --topk 10 \
2>&1 | tee "${OUT_DIR}/stage3_compare.log"

echo ""
echo "================================================================"
echo " Results"
echo "================================================================"
echo "  vLLM probe    : ${VLLM_PROBE}"
echo "  SpecForge probe: ${SF_PROBE}"
echo "  JSON report   : ${REPORT}"
echo "  Stage 1 log   : ${OUT_DIR}/stage1_vllm_probe.log"
echo "  Stage 2 log   : ${OUT_DIR}/stage2_specforge_probe.log"
echo "  Stage 3 log   : ${OUT_DIR}/stage3_compare.log"
if command -v python3 &>/dev/null; then
    python3 -c "
import json, sys
try:
    r = json.load(open('${REPORT}'))
    print('  Overall verdict  :', r.get('overall_verdict', 'N/A'))
    print('  FC output        :', r.get('fc_verdict', 'N/A'))
    print('  Draft logits     :', r.get('draft_logit_verdict', 'N/A'))
    s = r.get('sections',{}).get('draft_logit_parity',{}).get('summary',{})
    if s:
        print('  mean_cosine      :', s.get('mean_cosine'))
        print('  top1_match_rate  :', s.get('top1_match_rate'))
except Exception as e:
    print('  (could not parse report:', e, ')')
" 2>/dev/null || true
fi
echo "================================================================"
