#!/usr/bin/env bash
set -euo pipefail

DEBUG_DIR="${DFLASH_DRAFT_QUALITY_DEBUG_DIR:-/home/i-hulanxiang/workspace/vllm-parallel-drafting/tests/v1/spec_decode/_debug_draft_quality/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${DEBUG_DIR}"
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
SAMPLE_INDICES="${SAMPLE_INDICES:-0,1}"
MAX_STEPS="${MAX_STEPS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
TREE_WIDTH="${TREE_WIDTH:-7}"
MAX_TREE_BUDGET="${MAX_TREE_BUDGET:-255}"
TREE_DRAFT="${TREE_DRAFT:-accum_logp}"
TREE_HYBRID_ALPHA="${TREE_HYBRID_ALPHA:-1.0}"
MAX_DRAFT_PASSES="${MAX_DRAFT_PASSES:-0}"
TREE_PRUNE_RATIO="${TREE_PRUNE_RATIO:-0.25}"
TREE_CONSTRUCTION="${TREE_CONSTRUCTION:-breadth_first}"
TREE_ATTN_KERNEL="${TREE_ATTN_KERNEL:-optimus}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"
# HF-side transformers attn implementation. Default matches the production eval
# (dflash/eval_scripts/run_benchmark_causal_head_blk16w2_multi_draft_run.sh),
# which uses flash_attention_2. Earlier diagnostic runs accidentally used
# sdpa, which hit a layer-1+ all-zero mask bug in dflash/model/dflash.py that
# production never triggered. Use flash_attention_2 so HF matches training /
# production behavior and the HF<->vLLM comparison is apples-to-apples.
HF_ATTN_IMPLEMENTATION="${HF_ATTN_IMPLEMENTATION:-flash_attention_2}"
# Test N-2: opt-in second vLLM run with tree_width=1 / budget=1 / passes=1
# so per-iteration semantics match HF's chain-greedy target-forced path.
RUN_VLLM_CHAIN_SPEC="${RUN_VLLM_CHAIN_SPEC:-1}"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

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

CHAIN_SPEC_FLAG=()
if [[ "${RUN_VLLM_CHAIN_SPEC}" == "1" || "${RUN_VLLM_CHAIN_SPEC}" == "true" ]]; then
  CHAIN_SPEC_FLAG=("--run-vllm-chain-spec")
fi

"${PYTHON_BIN}" \
  /home/i-hulanxiang/workspace/vllm-parallel-drafting/tests/v1/spec_decode/run_draft_quality_diagnostic.py \
  --debug-dir "${DEBUG_DIR}" \
  --model "${TARGET_MODEL}" \
  --draft-model "${DRAFT_MODEL}" \
  --sample-indices "${SAMPLE_INDICES}" \
  --block-size "${BLOCK_SIZE}" \
  --tree-width "${TREE_WIDTH}" \
  --max-steps "${MAX_STEPS}" \
  --run-vllm-diagnostic \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-tree-budget "${MAX_TREE_BUDGET}" \
  --tree-draft "${TREE_DRAFT}" \
  --tree-hybrid-alpha "${TREE_HYBRID_ALPHA}" \
  --max-draft-passes "${MAX_DRAFT_PASSES}" \
  --tree-prune-ratio "${TREE_PRUNE_RATIO}" \
  --tree-construction "${TREE_CONSTRUCTION}" \
  --tree-attn-kernel "${TREE_ATTN_KERNEL}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --attn-implementation "${HF_ATTN_IMPLEMENTATION}" \
  "${CHAIN_SPEC_FLAG[@]}"

echo
echo "Draft-quality artifacts:"
echo "  ${DEBUG_DIR}/draft_quality_diagnostic_summary.json"
echo "  ${DEBUG_DIR}/vllm_draft_quality_diagnostic_summary.json"
echo "  ${DEBUG_DIR}/draft_quality_comparison_summary.json"
echo
echo "Draft-quality audit artifacts (checklist tests):"
echo "  ${DEBUG_DIR}/reference_capture.json                 # HF reference tensors/scalars"
echo "  ${DEBUG_DIR}/sample_000/vllm_draft_audit.json       # vLLM worker-side weight + parity capture"
echo "  ${DEBUG_DIR}/vllm_draft_weight_audit_report.json    # Test A: weight-loading audit"
echo "  ${DEBUG_DIR}/lm_head_embed_parity_report.json       # Tests B-D: lm_head/embed/mask/rope/aux parity"
echo "  ${DEBUG_DIR}/topk_log_divergence_report.json        # Test E: per-step top-k divergence"
echo "  ${DEBUG_DIR}/sample_000/vllm_target_aux_capture.json # vLLM worker-side raw pre-fc capture"
echo "  ${DEBUG_DIR}/target_hidden_parity_report.json        # Test F: per-layer target hidden parity + layer-id check"
echo "  ${DEBUG_DIR}/sample_000/vllm_draft_layer_capture.json # vLLM worker-side per-draft-layer d0 capture"
echo "  ${DEBUG_DIR}/draft_layer_parity_report.json           # Test G: per-draft-layer d0 hidden-state parity"
echo "  ${DEBUG_DIR}/sample_000/vllm_draft_layer1_bisect_capture.json # vLLM worker-side intra-layer taps at d0"
echo "  ${DEBUG_DIR}/draft_layer1_bisect_report.json           # Test H: intra-layer-1 bisect (input_ln/self_attn/post_attn_ln/mlp)"
echo "  ${DEBUG_DIR}/sample_000/vllm_draft_layer1_attn_bisect_capture.json # vLLM worker-side intra-self_attn taps at d0 + context K/V"
echo "  ${DEBUG_DIR}/draft_layer1_attn_bisect_report.json      # Test I+J: intra-self_attn bisect + I-2 context K/V parity + J manual-SDPA-vs-kernel cross-check"
echo "  ${DEBUG_DIR}/test_k_hf_sdpa_backend_report.json        # Test K: HF SDPA backend/dtype isolation at layer 1 d0"
echo "  ${DEBUG_DIR}/reference_capture_test_l.json             # HF multi-step per-iteration taps (sample 0)"
echo "  ${DEBUG_DIR}/sample_000/vllm_test_l_probe_capture.json # vLLM multi-step per-iteration taps"
echo "  ${DEBUG_DIR}/draft_layer1_multistep_report.json        # Test L: HF-vs-vLLM layer-1 self_attn parity across speculative iterations"
echo "  ${DEBUG_DIR}/draft_layer1_context_kv_alignment_report.json  # Test M: per-position HF-vs-vLLM context-K/V alignment + tail characterization"
echo "  ${DEBUG_DIR}/vllm_draft_quality_diagnostic_summary_chain_spec.json  # Test N-2: vLLM chain-spec (tree_width=1) top-1 summary (now populated via chain-spec topk probe)"
echo "  ${DEBUG_DIR}/test_n_per_iteration_report.json       # Test N-1 + N-2: per-iteration histograms, same-prefix parity, trajectory divergence"
echo "  ${DEBUG_DIR}/sample_000/vllm_test_l_probe_capture_chain.json  # Test O + P: chain-spec vLLM Test-L per-step taps (dense TEST_O_PROBE_STEPS set)"
echo "  ${DEBUG_DIR}/test_o_tree_vs_chain_layer1_report.json # Test O: tree-vs-chain-vs-HF A/B at layer-1 self_attn taps across iteration indices (step-matched)"
echo "  ${DEBUG_DIR}/test_p_position_matched_report.json    # Test P: tree-vs-chain layer-1 self_attn A/B at matched DECODED POSITIONS (position-aligned, trajectory-safe)"
echo "  ${DEBUG_DIR}/sample_000/vllm_test_q_probe_capture.json       # Test Q: tree-spec LAYER-0 entry/self_attn_out/exit hidden-state taps"
echo "  ${DEBUG_DIR}/sample_000/vllm_test_q_probe_capture_chain.json # Test Q: chain-spec LAYER-0 entry/self_attn_out/exit hidden-state taps"
echo "  ${DEBUG_DIR}/test_q_layer0_report.json              # Test Q: tree-vs-chain layer-0 A/B at matched decoded positions (scopes H1/H2/H3/H5-slot, refutes H4-kernel by probe placement)"
echo "  ${DEBUG_DIR}/test_v_branch_query_report.json        # Test V: tree-vs-chain layer-0 A/B for the first future/branch query row at matched decoded positions; checks whether the residual gap is depth>0 query-side."
echo "  ${DEBUG_DIR}/test_r_kv_position_audit.json          # Test R: offline per-iter context_positions / context_slot_mapping audit; efficient hypothesis ruling-out using capture-observed start-positions (scores H1/H2/H3/H5; reuses Test-L captures, no new run)"
echo "  ${DEBUG_DIR}/test_s_ctx_content_report.json        # Test S: per-position context K/V content A/B (tree vs chain) at matched decoded positions; uses ctx_per_position first_k fingerprint to score H1 (stale accepted-prefix content from rejected branches) vs H2 (tail-precompute content mismatch). Reuses Test-L captures."
echo "  ${DEBUG_DIR}/test_t_tail_layout_report.json        # Test T: offline tail-only tree-vs-chain layout audit at matched decoded positions; distinguishes H2 tail-content/write-back from H5 tail slot/position layout anomalies. Reuses Test-L captures."
echo "  ${DEBUG_DIR}/test_u_valid_tail_content_report.json # Test U: offline valid-tail-only tree-vs-chain content audit; only compares rows with slot != -1 on both sides to test whether written tail KV actually diverges. Reuses Test-L captures."
echo "  ${DEBUG_DIR}/test_w_effective_hidden_stream_report.json # Test W: tree-vs-chain layer-0 EFFECTIVE output parity (hidden+residual summed) at matched decoded positions; reconciles Test Q's split cosines with Test P's layer-1 parity by comparing the summed stream layer 1 actually consumes. Reuses Test-Q captures."
echo "  ${DEBUG_DIR}/test_x_branch_row_health_report.json # Test X: tree-internal branch-row health audit; per-iteration norms + row_k-vs-row_1 cosines across layer-0 taps for every captured branch row. Flags pathological norms / near-orthogonal input cosines without needing chain parity (replaces the structurally invalid Test V tree-vs-chain row-2 compare). Reuses Test-Q captures."
echo "  ${DEBUG_DIR}/test_y_branch_input_origin_report.json # Test Y: targeted tree-only branch-row input-origin audit; checks whether a bad branch row already differs before attention despite carrying the same query_input_id and expected query_position stride as the anchor row. Reuses the extended Test-Q capture metadata."
echo "  ${DEBUG_DIR}/test_z_actual_forward_input_report.json # Test Z: authoritative tree-only branch-row audit using the ACTUAL input_ids/positions passed into draft_inner.forward. Resolves whether low branch-row input cosine reflects real corruption or simply different token ids at the model boundary."
echo "  ${DEBUG_DIR}/test_aa_depth0_visible_prefix_report.json # Test AA: depth-0 tree-vs-chain visible-prefix audit using Test-Q forward-attn metadata plus Test-L context captures. Checks whether the tree run's actually visible context window is malformed or mismatched at the first divergent positions."
echo "  ${DEBUG_DIR}/sample_000/vllm_dflash_runtime_bundle.json # DFlash runtime bundles from selected executed tree iterations; first-pass context/query buffers plus seq_lens metadata used by Tests AB/AC."
echo "  ${DEBUG_DIR}/test_ab_first_pass_context_compaction_report.json # Test AB: first-pass accepted-context compaction audit across captured tree iterations. Verifies the visible accepted prefix has no PAD slots, no repeated positions, and matches the expected accepted-context length."
echo "  ${DEBUG_DIR}/test_ac_first_pass_metadata_consistency_report.json # Test AC: first-pass metadata/buffer consistency audit across captured tree iterations. Checks that seq_lens-derived visible context length agrees with the materialized context/query buffers and bonus/query continuity."
echo "  ${DEBUG_DIR}/test_ae_seq_len_derivation_report.json # Test AE: pre/post set_inputs_first_pass seq-lens derivation audit. Identifies whether emitted visible-context length follows logical cad.seq_lens-minus-rejected or the compacted accepted-prefix length from query_start_loc/front-valid rows."
echo "  ${DEBUG_DIR}/test_af_rejection_count_producer_report.json # Test AF: producer-side rejection accounting audit. Checks whether tiny compacted contexts are already implied by valid_sampled_tokens_count, num_rejected_tokens, and token_indices_to_sample before set_inputs_first_pass / copy_and_expand_dflash_inputs_kernel."
echo "  ${DEBUG_DIR}/sample_000/vllm_tree_attn_builder_probe.json # Tree-attention builder probe: raw per-step capture of CommonAttentionMetadata entering TreeAttentionMetadataBuilder.build_for_dflash_tree() and the emitted decode metadata."
echo "  ${DEBUG_DIR}/test_ag_tree_attn_builder_passthrough_report.json # Test AG: tree-attention metadata-builder pass-through audit. Verifies whether build_for_dflash_tree mutates seq_lens/max_seq_len/query_start_loc or simply forwards the caller's CommonAttentionMetadata into decode metadata unchanged."
echo "  ${DEBUG_DIR}/sample_000/vllm_drafter_first_pass_metadata_probe.json # Direct drafter-first-pass metadata probe: exact CommonAttentionMetadata seen by build_per_group_and_layer_attn_metadata(..., draft_index=0) before build_for_drafting."
echo "  ${DEBUG_DIR}/test_ah_drafter_first_pass_metadata_report.json # Test AH: direct drafter-first-pass metadata audit. Compares the drafter's actual first-pass CommonAttentionMetadata against the DFlash runtime bundle to determine whether the long logical seq_lens is already present before the drafter builder runs."
echo "  ${DEBUG_DIR}/test_ad_runtime_vs_forward_metadata_report.json # Test AD: offline runtime-bundle vs live-forward metadata join. Compares first-pass DFlash runtime metadata against the forward-attn metadata actually consumed by the layer-0 draft forward at the same tree step to localize handoff bugs in forward_context / attention-metadata building."
