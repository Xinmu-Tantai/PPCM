from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# run_native_profile -> llm.collective_rpc(_extract_topk_log) serializes a
# Python closure across the EngineCore IPC boundary. vLLM v1 refuses to pickle
# arbitrary callables unless VLLM_ALLOW_INSECURE_SERIALIZATION=1. The shell
# wrapper (run_draft_quality_diagnostic.sh:38) exports this, but when the
# Python entry point is invoked directly (or via a wrapper that forgets the
# export) the request encoder raises TypeError in the main process and the
# Test E topk_log extraction fails. Set it here (before any vLLM imports) so
# the diagnostic is robust regardless of launch path.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

_VLLM_ROOT = Path(
    os.environ.get(
        "VLLM_PARALLEL_DRAFTING_ROOT",
        Path(__file__).resolve().parents[3],
    )
)
_DFLASH_ROOT = Path(
    os.environ.get(
        "DFLASH_ROOT",
        "/root/workspace/causal_parallel_drafting_latest_eval",
    )
)
sys.path.insert(0, str(_VLLM_ROOT))
sys.path.insert(0, str(_DFLASH_ROOT))

from vllm import SamplingParams  # noqa: E402
from examples.offline_inference.dflash_profiling import (  # noqa: E402
    apply_chat_template,
    get_prompt_bank,
    run_native_profile,
    tokenizer_load_kwargs,
)
from model import DFlashDraftModel, extract_context_feature, sample_topk  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from draft_quality_audit import (  # noqa: E402
    build_draft_layer1_attn_bisect_report,
    build_draft_layer1_bisect_report,
    build_draft_layer1_context_kv_alignment_report,
    build_draft_layer1_multistep_report,
    build_draft_layer_parity_report,
    build_lm_head_embed_parity_report,
    build_target_hidden_parity_report,
    build_test_k_hf_sdpa_report,
    build_test_n_per_iteration_report,
    build_test_o_tree_vs_chain_layer1_report,
    build_test_p_position_matched_report,
    build_test_q_layer0_report,
    build_test_r_kv_position_audit,
    build_test_s_ctx_content_report,
    build_test_t_tail_layout_report,
    build_test_u_valid_tail_content_report,
    build_test_w_effective_hidden_stream_report,
    build_test_x_tree_branch_row_health_report,
    build_test_y_branch_input_origin_report,
    build_test_z_actual_forward_input_report,
    build_test_aa_depth0_visible_prefix_report,
    build_test_ab_first_pass_context_compaction_report,
    build_test_ac_first_pass_metadata_consistency_report,
    build_test_ae_seq_len_derivation_report,
    build_test_af_rejection_count_producer_report,
    build_test_ag_tree_attn_builder_passthrough_report,
    build_test_ah_drafter_first_pass_metadata_report,
    build_test_ad_runtime_vs_forward_metadata_report,
    build_topk_log_divergence_report,
    build_weight_audit_report,
    collect_vllm_draft_audit,
    install_vllm_dflash_runtime_bundle_probe,
    install_vllm_chain_spec_topk_log_probe,
    install_vllm_draft_layer1_attn_bisect_capture,
    install_vllm_draft_layer1_bisect_capture,
    install_vllm_draft_layer_capture,
    install_vllm_target_aux_capture,
    install_vllm_test_l_probe,
    install_vllm_test_q_probe,
    retrieve_vllm_draft_layer1_attn_bisect_capture,
    retrieve_vllm_draft_layer1_bisect_capture,
    retrieve_vllm_draft_layer_capture,
    retrieve_vllm_dflash_runtime_bundle,
    retrieve_vllm_drafter_first_pass_metadata_probe,
    retrieve_vllm_tree_attn_builder_probe,
    retrieve_vllm_target_aux_capture,
    retrieve_vllm_test_l_probe,
    retrieve_vllm_test_q_probe,
    uninstall_vllm_chain_spec_topk_log_probe,
    write_reference_capture,
)


TEST_H_BISECT_LAYER_IDX = 1
TEST_V_BRANCH_ROW = 2


class _TestLSkipTestK(Exception):
    """Sentinel raised inside the HF monkey-patch to short-circuit the
    Test K block when the patch is running under Test L (step_idx != 0).
    Must NOT inherit from any sentinel the K block's except clause catches
    except via a dedicated except arm."""


# Test L: multi-step layer-1 intra-attention parity.
#
# Rationale: Tests F/G/H/I/J/K established single-iteration (step 0) parity
# between HF and vLLM drafts is ≥0.9997 across every tap.  Yet end-to-end
# the gap is ~40 pt.  The remaining hypotheses (Pγ noise-K/V lifecycle,
# Pδ context-K/V refresh cadence) only manifest *across* speculative
# iterations, not at step 0.  Test L re-fires the Test I probe at a set of
# later iterations to observe whether layer-1 self_attn parity drifts as N
# grows (e.g. due to vLLM's paged cache retaining stale noise K/V that HF
# discards each step).
#
# Step semantics:
#   - HF reference: step_idx = number of teacher-forced tokens accepted so far
#     (one per outer-loop iteration).  At step N the draft attends over
#     num_prompt + N context tokens.
#   - vLLM: step index = number of speculative-iteration invocations (tied to
#     precompute_and_store_context_kv call count).
# These are NOT identical (vLLM may accept >1 token per iteration), but both
# count "number of speculative passes through the draft stack", so cross-step
# drift within each side is directly comparable.
TEST_L_STEPS: tuple[int, ...] = (5, 15, 30)
# For the chain-spec Test O run we capture step 0 as well because the legacy
# single-shot attn-bisect probe is not installed alongside Test L in chain
# mode.  The Test-L probe is standalone-capable (it monkey-patches
# ``precompute_and_store_context_kv`` and ``self_attn.forward``
# unconditionally for listed steps), so adding 0 just means one extra slot
# in ``per_step``.
TEST_O_STEPS: tuple[int, ...] = (0, 5, 15, 30)
# Test P (position-matched tree-vs-chain A/B): tree-spec advances many
# accepted tokens per iter while chain-spec advances 1, so matched decoded
# positions land at different iteration indices on each side.  Cover a
# dense iteration range so at least one tree iter's starting position
# coincides with a chain iter's starting position.  Numbers were picked
# to (a) fully span the first 20 tree iters (expected positions 0..~120),
# (b) probe a handful of later chain iters (positions >60 land at chain
# iter >=60 but tree iter ~15-20), and (c) keep the probe JSON under a
# few hundred KB per side.
TEST_P_CAPTURE_STEPS: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30,
    40, 50, 60, 80, 100, 120, 150, 180, 220,
)
# Union sets actually passed to the probes -- keep TEST_L_STEPS / TEST_O_STEPS
# as the public "report-configured" step sets (they are serialized into
# reference_capture_test_l.json and the Test-L / Test-O reports read them
# back), and install with the broader union so Test P also has data.
TEST_L_PROBE_STEPS: tuple[int, ...] = tuple(
    sorted(set(TEST_L_STEPS) | set(TEST_P_CAPTURE_STEPS))
)
TEST_O_PROBE_STEPS: tuple[int, ...] = tuple(
    sorted(set(TEST_O_STEPS) | set(TEST_P_CAPTURE_STEPS))
)

# Minimal tap subset Test L copies out of the full step-0 capture (plus
# replicates at step > 0).  Skip J/K variants — those are single-step-only.
TEST_L_ATTN_KEYS: tuple[str, ...] = (
    "i1_q_post_qproj",
    "k_noise_post_kproj",
    "v_noise_post_vproj",
    "i2_q_post_qnorm",
    "k_noise_post_knorm",
    "i3_q_post_rope",
    "k_noise_post_rope",
    "i4_attn_out_pre_oproj",
    "i5_attn_out_post_oproj",
)
TEST_L_CTX_KEYS: tuple[str, ...] = (
    "k_last_context_pre_rope",
    "k_last_context",
    "v_last_context",
)

TEST_M_FULL_CTX_KEYS: tuple[str, ...] = (
    "k_ctx_full_pre_rope",
    "k_ctx_full_post_rope",
    "v_ctx_full",
)


DEFAULT_TARGET_MODEL = "/data/models/Qwen3-8B"
DEFAULT_DRAFT_MODEL = (
    "/mnt/specdec-dev/checkpoints/specforge/outputs/"
    "nemotron-780k-and-codealpaca20k-v2-causal-distill-lr1e-4-anchorcnt512/"
    "epoch_6_step_583488"
)
_DEFAULT_DEBUG_DIR = Path("/tmp") / f"debug_draft_quality_{datetime.now():%Y%m%d_%H%M%S}"


def _parse_sample_indices(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",")]
    indices = [int(v) for v in values if v]
    if not indices:
        raise ValueError("At least one sample index must be provided.")
    return indices


def _get_debug_dir(path: str | None) -> Path:
    debug_dir = Path(path) if path else _DEFAULT_DEBUG_DIR
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _rank_of_token(logprobs: torch.Tensor, token_id: int) -> int:
    target_lp = logprobs[token_id]
    return int((logprobs > target_lp).sum().item()) + 1


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _capture_cuda_memory_snapshot(label: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"label": label, "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return snapshot

    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    snapshot.update(
        {
            "device": str(torch.cuda.current_device()),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "free_gib": float(free_bytes / (1024**3)),
            "total_gib": float(total_bytes / (1024**3)),
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    )
    return snapshot


def _cleanup_cuda_stage(debug_dir: Path | None = None, label: str | None = None) -> dict[str, Any]:
    gc.collect()
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        for device_idx in range(torch.cuda.device_count()):
            torch.cuda.set_device(device_idx)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        torch.cuda.set_device(current_device)

    snapshot = _capture_cuda_memory_snapshot(label or "post_cleanup")
    if debug_dir is not None and label is not None:
        _write_json(debug_dir / f"{label}.json", snapshot)
    return snapshot


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _extract_context_feature_on_device(
    hidden_states: list[torch.Tensor],
    layer_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    offset = 1
    selected_states = [
        hidden_states[int(layer_id) + offset].to(device)
        for layer_id in layer_ids
    ]
    return torch.cat(selected_states, dim=-1)


def _target_text_model(target: torch.nn.Module) -> torch.nn.Module:
    if hasattr(target, "language_model"):
        return target.language_model
    if hasattr(target, "model") and hasattr(target.model, "language_model"):
        return target.model.language_model
    if hasattr(target, "model"):
        return target.model
    return target


def _target_embed_tokens(target: torch.nn.Module) -> torch.nn.Module:
    text_model = _target_text_model(target)
    if hasattr(text_model, "embed_tokens"):
        return text_model.embed_tokens
    if hasattr(text_model, "get_input_embeddings"):
        return text_model.get_input_embeddings()
    raise AttributeError("Unable to locate target text embedding module")


def _target_lm_head(target: torch.nn.Module) -> torch.nn.Module:
    if hasattr(target, "lm_head"):
        return target.lm_head
    if hasattr(target, "get_output_embeddings"):
        lm_head = target.get_output_embeddings()
        if lm_head is not None:
            return lm_head
    raise AttributeError("Unable to locate target lm_head module")


def _target_text_config(target: torch.nn.Module) -> Any:
    config = target.config
    return getattr(config, "text_config", config)


def _target_vocab_size(target: torch.nn.Module) -> int:
    text_config = _target_text_config(target)
    if hasattr(text_config, "vocab_size"):
        return int(text_config.vocab_size)
    return int(_target_lm_head(target).weight.shape[0])


def _target_num_hidden_layers(target: torch.nn.Module) -> int:
    text_config = _target_text_config(target)
    return int(getattr(text_config, "num_hidden_layers", 0) or 0)


def _target_text_forward(
    target: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    past_key_values: DynamicCache,
    use_cache: bool,
    logits_to_keep: int,
    output_hidden_states: bool,
):
    text_model = _target_text_model(target)
    text_outputs = text_model(
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    lm_head = _target_lm_head(target)
    lm_head_device = _module_device(lm_head)
    hidden_for_logits = text_outputs.last_hidden_state
    if logits_to_keep:
        hidden_for_logits = hidden_for_logits[:, -int(logits_to_keep):]
    logits = lm_head(hidden_for_logits.to(lm_head_device))
    return SimpleNamespace(
        logits=logits,
        hidden_states=text_outputs.hidden_states,
        past_key_values=text_outputs.past_key_values,
        last_hidden_state=text_outputs.last_hidden_state,
    )


def _load_hf_target_model(
    model: str,
    attn_implementation: str,
    *,
    device: torch.device,
    device_map_auto: bool,
    device_map_gpus: int | None,
):
    kwargs: dict[str, Any] = {
        "attn_implementation": attn_implementation,
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if device_map_auto:
        visible_gpus = torch.cuda.device_count()
        if visible_gpus <= 0:
            raise RuntimeError("CUDA is required for target_device_map_auto")
        num_gpus = visible_gpus if device_map_gpus is None else device_map_gpus
        if num_gpus < 1 or num_gpus > visible_gpus:
            raise ValueError(
                f"target_device_map_gpus must be in [1, {visible_gpus}], "
                f"got {num_gpus}"
            )
        kwargs["device_map"] = "balanced" if num_gpus > 1 else "auto"
        kwargs["max_memory"] = {idx: "130GiB" for idx in range(num_gpus)}
    target = AutoModelForCausalLM.from_pretrained(model, **kwargs).eval()
    if not device_map_auto:
        target = target.to(device)
    return target


@torch.inference_mode()
def run_draft_quality_diagnostic(
    *,
    debug_dir: Path,
    model: str,
    draft_model_path: str,
    sample_indices: list[int],
    block_size: int = 16,
    tree_width: int = 7,
    max_steps: int = 256,
    attn_implementation: str = "flash_attention_2",
    seed: int = 0,
    target_device_map_auto: bool = False,
    target_device_map_gpus: int | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    tokenizer = None
    prompt_bank = None
    target = None
    draft = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            **tokenizer_load_kwargs(model),
        )
        prompt_bank = apply_chat_template(
            tokenizer,
            get_prompt_bank("humaneval"),
            model,
        )
        target = _load_hf_target_model(
            model,
            attn_implementation,
            device=device,
            device_map_auto=target_device_map_auto,
            device_map_gpus=target_device_map_gpus,
        )
        draft = DFlashDraftModel.from_pretrained(
            draft_model_path,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        ).to(device).eval()
        head_is_causal = draft.resolve_causal_head("causal")

        eos_token_ids = (
            {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
        )
        per_sample: list[dict[str, Any]] = []
        aggregate_top1: list[float] = []
        aggregate_topk: list[float] = []
        aggregate_rank: list[float] = []
        aggregate_lp: list[float] = []

        reference_capture_written = False
        # Test L: per-step HF-side payloads keyed by step_idx. Only the first
        # sample contributes; each entry has {"step", "context_len",
        # "attn_taps", "ctx_kv"}.
        hf_test_l_per_step: dict[int, dict[str, Any]] = {}
        # Sample that owns the HF reference capture (always sample_indices[0]).
        hf_reference_sample_index = sample_indices[0] if sample_indices else None

        for sample_index in sample_indices:
            prompt_text = prompt_bank[sample_index]
            sample_dir = debug_dir / f"sample_{sample_index:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            target_input_device = _module_device(_target_embed_tokens(target))
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(
                target_input_device
            )
            num_input_tokens = int(input_ids.shape[1])
            position_ids = torch.arange(
                num_input_tokens + max_steps + block_size + 2,
                device=target_input_device,
            ).unsqueeze(0)
            past_key_values_target = DynamicCache()
            past_key_values_draft = DynamicCache()

            output = _target_text_forward(
                target,
                input_ids,
                position_ids=position_ids[:, :num_input_tokens],
                past_key_values=past_key_values_target,
                use_cache=True,
                logits_to_keep=1,
                output_hidden_states=True,
            )
            target_hidden = _extract_context_feature_on_device(
                output.hidden_states, draft.target_layer_ids, device
            )
            root_token = int(output.logits[0, -1].argmax(dim=-1).item())
            generated_target_tokens = [root_token]
            start = num_input_tokens

            steps: list[dict[str, Any]] = []
            for step_idx in range(max_steps):
                if root_token in eos_token_ids:
                    break

                block_output_ids = torch.full(
                    (1, block_size),
                    draft.mask_token_id,
                    dtype=torch.long,
                    device=device,
                )
                block_output_ids[0, 0] = root_token
                draft_pos_ids = position_ids[
                    :, past_key_values_draft.get_seq_length() : start + block_size
                ]
                target_embed_tokens = _target_embed_tokens(target)
                embed_device = _module_device(target_embed_tokens)
                noise_embedding = target_embed_tokens(
                    block_output_ids.to(embed_device)
                ).to(device)

                # Test G: capture HF per-layer draft outputs at d0 row on the
                # very first draft step of sample 0. The d0 row in HF's
                # draft_output corresponds to the first mask position
                # (row index 1 in a block of size block_size).
                hf_capture_active = (
                    not reference_capture_written and step_idx == 0
                )
                # Test L: re-fire the intra-self_attn probe at later iterations
                # (only on the reference sample).  Unlike hf_capture_active,
                # this does NOT write per-layer / bisect / J / K captures —
                # only the minimal TEST_L_ATTN_KEYS + TEST_L_CTX_KEYS subset
                # is snapshotted.  Runs independently from hf_capture_active
                # so step 0 captures everything (Tests G/H/I/J/K via
                # hf_capture_active) and steps in TEST_L_STEPS additionally
                # capture the minimal subset into hf_test_l_per_step.
                hf_test_l_active = (
                    sample_index == hf_reference_sample_index
                    and int(step_idx) in TEST_L_PROBE_STEPS
                )
                hf_probe_active = hf_capture_active or hf_test_l_active
                hf_layer_outputs_at_d0: dict[int, torch.Tensor] = {}
                hf_layer_hook_handles: list[Any] = []
                hf_d0_row_index = 1

                def _make_hf_layer_hook(layer_idx_local: int):
                    def _hook(_module, _args, output):
                        if not hf_capture_active:
                            return None
                        if isinstance(output, torch.Tensor) and output.dim() >= 2:
                            row = hf_d0_row_index
                            if row < output.shape[-2]:
                                hf_layer_outputs_at_d0[layer_idx_local] = (
                                    output[..., row, :]
                                    .reshape(-1)
                                    .detach()
                                    .clone()
                                )
                        return None

                    return _hook

                if hf_capture_active:
                    for li, layer in enumerate(draft.layers):
                        hf_layer_hook_handles.append(
                            layer.register_forward_hook(_make_hf_layer_hook(li))
                        )

                hf_bisect_taps: dict[str, torch.Tensor] = {}
                hf_bisect_hook_handles: list[Any] = []
                if (
                    hf_capture_active
                    and 0 <= TEST_H_BISECT_LAYER_IDX < len(draft.layers)
                ):
                    bisect_layer = draft.layers[TEST_H_BISECT_LAYER_IDX]

                    def _row_clone(t: torch.Tensor) -> torch.Tensor | None:
                        if not isinstance(t, torch.Tensor) or t.dim() < 2:
                            return None
                        if hf_d0_row_index >= t.shape[-2]:
                            return None
                        return (
                            t[..., hf_d0_row_index, :]
                            .reshape(-1)
                            .detach()
                            .clone()
                        )

                    def _make_pre_hook(tap_name: str):
                        def _pre_hook(_module, args):
                            if not hf_capture_active:
                                return None
                            if not args:
                                return None
                            first = args[0]
                            if isinstance(first, torch.Tensor):
                                row = _row_clone(first)
                                if row is not None:
                                    hf_bisect_taps[tap_name] = row
                            return None

                        return _pre_hook

                    def _make_post_hook(tap_name: str):
                        def _post_hook(_module, _args, output):
                            if not hf_capture_active:
                                return None
                            tensor = (
                                output[0]
                                if isinstance(output, tuple) and output
                                else output
                            )
                            if isinstance(tensor, torch.Tensor):
                                row = _row_clone(tensor)
                                if row is not None:
                                    hf_bisect_taps[tap_name] = row
                            return None

                        return _post_hook

                    bisect_input_ln = getattr(
                        bisect_layer, "input_layernorm", None
                    )
                    bisect_self_attn = getattr(bisect_layer, "self_attn", None)
                    bisect_post_ln = getattr(
                        bisect_layer, "post_attention_layernorm", None
                    )
                    bisect_mlp = getattr(bisect_layer, "mlp", None)
                    if bisect_input_ln is not None:
                        hf_bisect_hook_handles.append(
                            bisect_input_ln.register_forward_pre_hook(
                                _make_pre_hook("layer_input")
                            )
                        )
                        hf_bisect_hook_handles.append(
                            bisect_input_ln.register_forward_hook(
                                _make_post_hook("post_input_ln")
                            )
                        )
                    if bisect_self_attn is not None:
                        hf_bisect_hook_handles.append(
                            bisect_self_attn.register_forward_hook(
                                _make_post_hook("self_attn_out")
                            )
                        )
                    if bisect_post_ln is not None:
                        hf_bisect_hook_handles.append(
                            bisect_post_ln.register_forward_pre_hook(
                                _make_pre_hook("post_attn_residual")
                            )
                        )
                        hf_bisect_hook_handles.append(
                            bisect_post_ln.register_forward_hook(
                                _make_post_hook("post_post_attn_ln")
                            )
                        )
                    if bisect_mlp is not None:
                        hf_bisect_hook_handles.append(
                            bisect_mlp.register_forward_hook(
                                _make_post_hook("mlp_out")
                            )
                        )

                hf_attn_taps: dict[str, torch.Tensor] = {}
                hf_attn_patch_info: dict[str, Any] = {}
                hf_test_k_variants: dict[str, torch.Tensor] = {}
                hf_test_k_metadata: dict[str, Any] = {}
                if (
                    hf_probe_active
                    and 0 <= TEST_H_BISECT_LAYER_IDX < len(draft.layers)
                ):
                    from model.dflash import apply_rotary_pos_emb

                    bisect_attn_mod = draft.layers[
                        TEST_H_BISECT_LAYER_IDX
                    ].self_attn
                    hf_attn_patch_info["module"] = bisect_attn_mod
                    hf_attn_patch_info["original_forward"] = (
                        bisect_attn_mod.forward
                    )
                    hf_attn_patch_info["num_input_tokens"] = int(num_input_tokens)

                    def _patched_hf_attn_forward(
                        hidden_states,
                        target_hidden,
                        position_embeddings,
                        attention_mask,
                        past_key_values=None,
                        cache_position=None,
                        **kwargs,
                    ):
                        bsz, q_len = hidden_states.shape[:-1]
                        ctx_len = target_hidden.shape[1]
                        is_causal = kwargs.pop("is_causal", None)
                        if is_causal is None:
                            is_causal = bisect_attn_mod.is_causal
                        head_dim = bisect_attn_mod.head_dim
                        noise_d0_seq_idx = ctx_len + hf_d0_row_index
                        last_ctx_idx = ctx_len - 1

                        q_proj_out = bisect_attn_mod.q_proj(hidden_states)
                        k_noise_proj = bisect_attn_mod.k_proj(hidden_states)
                        v_noise_proj = bisect_attn_mod.v_proj(hidden_states)

                        if hf_d0_row_index < q_len:
                            hf_attn_taps["i1_q_post_qproj"] = (
                                q_proj_out[0, hf_d0_row_index]
                                .detach()
                                .clone()
                            )
                            hf_attn_taps["k_noise_post_kproj"] = (
                                k_noise_proj[0, hf_d0_row_index]
                                .detach()
                                .clone()
                            )
                            hf_attn_taps["v_noise_post_vproj"] = (
                                v_noise_proj[0, hf_d0_row_index]
                                .detach()
                                .clone()
                            )

                        q = q_proj_out.view(bsz, q_len, -1, head_dim)
                        q = bisect_attn_mod.q_norm(q).transpose(1, 2)
                        if hf_d0_row_index < q_len:
                            # q now shape (bsz, num_heads, q_len, head_dim).
                            hf_attn_taps["i2_q_post_qnorm"] = (
                                q[0, :, hf_d0_row_index, :]
                                .reshape(-1)
                                .detach()
                                .clone()
                            )

                        k_ctx = bisect_attn_mod.k_proj(target_hidden)
                        v_ctx = bisect_attn_mod.v_proj(target_hidden)
                        k_cat = torch.cat([k_ctx, k_noise_proj], dim=1).view(
                            bsz, ctx_len + q_len, -1, head_dim
                        )
                        v_cat = torch.cat([v_ctx, v_noise_proj], dim=1).view(
                            bsz, ctx_len + q_len, -1, head_dim
                        )
                        k_normed = bisect_attn_mod.k_norm(k_cat).transpose(1, 2)
                        v_out = v_cat.transpose(1, 2)
                        if (
                            0 <= last_ctx_idx < k_normed.shape[-2]
                            and q_len > hf_d0_row_index
                        ):
                            hf_attn_taps["k_last_context_pre_rope"] = (
                                k_normed[0, :, last_ctx_idx, :]
                                .reshape(-1)
                                .detach()
                                .clone()
                            )
                            hf_attn_taps["v_last_context"] = (
                                v_out[0, :, last_ctx_idx, :]
                                .reshape(-1)
                                .detach()
                                .clone()
                            )
                            # Test M: capture FULL context window (all
                            # positions 0..ctx_len-1) pre-RoPE K and V for
                            # per-position HF-vs-vLLM alignment.  Flattened
                            # to (ctx_len, nkv*hd) and stashed raw; compact
                            # per-position stats are computed at serialization
                            # time to keep the capture path fast.
                            hf_attn_taps["k_ctx_full_pre_rope"] = (
                                k_normed[0, :, :ctx_len, :]
                                .permute(1, 0, 2)
                                .reshape(ctx_len, -1)
                                .detach()
                                .clone()
                            )
                            hf_attn_taps["v_ctx_full"] = (
                                v_out[0, :, :ctx_len, :]
                                .permute(1, 0, 2)
                                .reshape(ctx_len, -1)
                                .detach()
                                .clone()
                            )
                            if noise_d0_seq_idx < k_normed.shape[-2]:
                                hf_attn_taps["k_noise_post_knorm"] = (
                                    k_normed[0, :, noise_d0_seq_idx, :]
                                    .reshape(-1)
                                    .detach()
                                    .clone()
                                )

                        cos, sin = position_embeddings
                        q_rot, k_rot = apply_rotary_pos_emb(
                            q, k_normed, cos, sin
                        )
                        if hf_d0_row_index < q_rot.shape[-2]:
                            hf_attn_taps["i3_q_post_rope"] = (
                                q_rot[0, :, hf_d0_row_index, :]
                                .reshape(-1)
                                .detach()
                                .clone()
                            )
                        if 0 <= last_ctx_idx < k_rot.shape[-2]:
                            hf_attn_taps["k_last_context"] = (
                                k_rot[0, :, last_ctx_idx, :]
                                .reshape(-1)
                                .detach()
                                .clone()
                            )
                            # Test M: full post-RoPE K across all context
                            # positions.  Matches vLLM's
                            # ``all_k_final[bisect_layer_idx]`` layout.
                            hf_attn_taps["k_ctx_full_post_rope"] = (
                                k_rot[0, :, :ctx_len, :]
                                .permute(1, 0, 2)
                                .reshape(ctx_len, -1)
                                .detach()
                                .clone()
                            )
                            if noise_d0_seq_idx < k_rot.shape[-2]:
                                hf_attn_taps["k_noise_post_rope"] = (
                                    k_rot[0, :, noise_d0_seq_idx, :]
                                    .reshape(-1)
                                    .detach()
                                    .clone()
                                )

                        # Continue with the original attention path.
                        # NOTE: DynamicCache.get_seq_length() defaults to layer_idx=0.
                        # Since layer 0 runs first and updates its own cache before layer
                        # bisect_attn_mod.layer_idx (typically 1) runs, calling without a
                        # layer_idx would return layer 0's post-update length instead of
                        # THIS layer's empty cache. That bug caused
                        # _build_dflash_causal_attention_mask to produce an all-zero
                        # (non-causal) mask here — matching the bug we already fixed in
                        # dflash/model/dflash.py:140-144. Mirror that fix by passing
                        # bisect_attn_mod.layer_idx so the probe stays training-aligned.
                        cached_kv_len = (
                            past_key_values.get_seq_length(bisect_attn_mod.layer_idx)
                            if past_key_values is not None
                            else 0
                        )
                        if past_key_values is not None:
                            cache_kwargs = {
                                "sin": sin,
                                "cos": cos,
                                "cache_position": cache_position,
                            }
                            k_final, v_final = past_key_values.update(
                                k_rot,
                                v_out,
                                bisect_attn_mod.layer_idx,
                                cache_kwargs,
                            )
                        else:
                            k_final, v_final = k_rot, v_out

                        from model.dflash import (
                            _build_dflash_causal_attention_mask,
                            _to_additive_attention_mask,
                        )
                        from transformers.modeling_utils import (
                            ALL_ATTENTION_FUNCTIONS,
                        )

                        attn_backend = (
                            bisect_attn_mod.config._attn_implementation
                        )
                        use_explicit_dflash_causal_mask = bool(
                            is_causal
                        ) and attn_backend in {"eager", "sdpa"}
                        attn_mask_eff = attention_mask
                        if use_explicit_dflash_causal_mask:
                            dflash_causal_mask = (
                                _build_dflash_causal_attention_mask(
                                    query=q_rot,
                                    key=k_final,
                                    cached_kv_len=cached_kv_len,
                                    ctx_len=ctx_len,
                                )
                            )
                            if attn_mask_eff is not None:
                                dflash_causal_mask = (
                                    dflash_causal_mask
                                    + _to_additive_attention_mask(
                                        attn_mask_eff,
                                        query_dtype=q_rot.dtype,
                                        device=q_rot.device,
                                        key_len=k_final.shape[-2],
                                    )
                                )
                            attn_mask_eff = dflash_causal_mask
                            is_causal_for_call = False
                        else:
                            is_causal_for_call = is_causal

                        attn_fn = None
                        if attn_backend != "eager":
                            attn_fn = ALL_ATTENTION_FUNCTIONS.get(
                                attn_backend
                            )
                        if attn_fn is None:
                            from model.dflash import (
                                eager_attention_forward,
                            )

                            attn_fn = eager_attention_forward
                        kwargs["is_causal"] = is_causal_for_call
                        attn_output, attn_weights = attn_fn(
                            bisect_attn_mod,
                            q_rot,
                            k_final,
                            v_final,
                            attn_mask_eff,
                            dropout=0.0
                            if not bisect_attn_mod.training
                            else bisect_attn_mod.attention_dropout,
                            scaling=bisect_attn_mod.scaling,
                            sliding_window=bisect_attn_mod.sliding_window,
                            **kwargs,
                        )
                        attn_output = attn_output.reshape(bsz, q_len, -1)
                        if hf_d0_row_index < attn_output.shape[1]:
                            hf_attn_taps["i4_attn_out_pre_oproj"] = (
                                attn_output[0, hf_d0_row_index]
                                .detach()
                                .clone()
                            )

                        # ---------- Test J: manual SDPA reference ----------
                        # Recompute the exact same math (softmax(QK^T/sqrt(d) + mask) V)
                        # from the Q/K/V tensors that were just fed to the
                        # HF attention function. This establishes a
                        # backend-independent ground truth to compare both
                        # HF and vLLM kernels against.
                        #
                        # Only run for the single-shot reference capture
                        # (step 0).  Test-L re-installs this probe at later
                        # steps for tap capture only; skipping J keeps
                        # per-step overhead bounded.
                        if hf_capture_active:
                            try:
                                num_heads_ref = (
                                    bisect_attn_mod.config.num_attention_heads
                                )
                                num_kv_heads_ref = (
                                    bisect_attn_mod.config.num_key_value_heads
                                )
                                head_dim_ref = bisect_attn_mod.head_dim
                                scaling_ref = bisect_attn_mod.scaling
                                kv_len_ref = k_final.shape[-2]
                                ctx_len_ref = ctx_len
                                repeats = num_heads_ref // num_kv_heads_ref
                                k_exp = k_final.repeat_interleave(
                                    repeats, dim=1
                                )
                                v_exp = v_final.repeat_interleave(
                                    repeats, dim=1
                                )
                                scores_ref = (
                                    torch.einsum(
                                        "bhqd,bhkd->bhqk",
                                        q_rot.float(),
                                        k_exp.float(),
                                    )
                                    * scaling_ref
                                )
                                key_positions = torch.arange(
                                    kv_len_ref, device=q_rot.device
                                )
                                query_positions = ctx_len_ref + torch.arange(
                                    q_len, device=q_rot.device
                                )
                                mask_ref = (
                                    key_positions.unsqueeze(0)
                                    > query_positions.unsqueeze(1)
                                )
                                scores_ref = scores_ref.masked_fill(
                                    mask_ref.unsqueeze(0).unsqueeze(0),
                                    float("-inf"),
                                )
                                attn_w_ref = torch.softmax(scores_ref, dim=-1)
                                manual_out_ref = torch.einsum(
                                    "bhqk,bhkd->bhqd",
                                    attn_w_ref,
                                    v_exp.float(),
                                )
                                manual_out_ref = (
                                    manual_out_ref.transpose(1, 2)
                                    .reshape(bsz, q_len, -1)
                                    .to(q_rot.dtype)
                                )
                                if hf_d0_row_index < manual_out_ref.shape[1]:
                                    hf_attn_taps[
                                        "j_manual_attn_out_pre_oproj"
                                    ] = (
                                        manual_out_ref[0, hf_d0_row_index]
                                        .detach()
                                        .clone()
                                    )
                            except Exception as _e:
                                hf_attn_patch_info["manual_sdpa_error"] = str(
                                    _e
                                )

                        # ---------- Test K: HF SDPA backend / dtype isolation ----------
                        # Run the SAME q_rot / k_final / v_final / attn_mask_eff
                        # through multiple torch SDPA backends and dtypes, and
                        # capture the d0 row of each.  The audit then tells us
                        # (a) which backend torch dispatched to by default, and
                        # (b) whether any alternate backend / dtype combination
                        # reproduces either the attenuated HF kernel output or
                        # the textbook manual SDPA output.
                        #
                        # Only run for the step-0 single-shot reference
                        # capture; Test-L re-installs this probe at later
                        # steps where we only want the minimal tap subset.
                        try:
                            if not hf_capture_active:
                                raise _TestLSkipTestK
                            import torch.nn.functional as _K_F  # noqa: PLC0415
                            from torch.nn.attention import (  # noqa: PLC0415
                                SDPBackend,
                                sdpa_kernel,
                            )

                            # Inputs as fed to the actual kernel.
                            num_heads_k = (
                                bisect_attn_mod.config.num_attention_heads
                            )
                            num_kv_heads_k = (
                                bisect_attn_mod.config.num_key_value_heads
                            )
                            repeats_k = num_heads_k // num_kv_heads_k
                            q_k = q_rot
                            k_k = k_final.repeat_interleave(repeats_k, dim=1)
                            v_k = v_final.repeat_interleave(repeats_k, dim=1)
                            mask_k = attn_mask_eff
                            scaling_k = bisect_attn_mod.scaling
                            # If the actual backend supplied no explicit mask
                            # (e.g. flash_attention_2 uses its internal
                            # tail-aligned causal mask), synthesize the
                            # equivalent additive causal mask here so Test K
                            # variants run against the same causal semantics
                            # instead of silently degrading to non-causal.
                            if mask_k is None:
                                try:
                                    _kv_len_k = k_k.shape[-2]
                                    _q_len_k = q_k.shape[-2]
                                    _key_pos = torch.arange(
                                        _kv_len_k, device=q_k.device
                                    )
                                    _query_pos = (
                                        _kv_len_k - _q_len_k
                                    ) + torch.arange(
                                        _q_len_k, device=q_k.device
                                    )
                                    _cannot = (
                                        _key_pos.unsqueeze(0)
                                        > _query_pos.unsqueeze(1)
                                    )
                                    _built = torch.zeros(
                                        (1, 1, _q_len_k, _kv_len_k),
                                        dtype=q_k.dtype,
                                        device=q_k.device,
                                    )
                                    mask_k = _built.masked_fill(
                                        _cannot.unsqueeze(0).unsqueeze(0),
                                        torch.finfo(q_k.dtype).min,
                                    )
                                    hf_test_k_metadata[
                                        "mask_synthesized_causal"
                                    ] = True
                                except Exception as _mk_err:
                                    hf_test_k_metadata[
                                        "mask_synth_error"
                                    ] = str(_mk_err)

                            # Record metadata for post-hoc interpretation.
                            hf_test_k_metadata["q_shape"] = list(q_k.shape)
                            hf_test_k_metadata["k_shape"] = list(k_k.shape)
                            hf_test_k_metadata["v_shape"] = list(v_k.shape)
                            hf_test_k_metadata["q_dtype"] = str(q_k.dtype)
                            hf_test_k_metadata["k_dtype"] = str(k_k.dtype)
                            hf_test_k_metadata["v_dtype"] = str(v_k.dtype)
                            hf_test_k_metadata["scaling"] = float(scaling_k)
                            if isinstance(mask_k, torch.Tensor):
                                hf_test_k_metadata["mask_shape"] = list(
                                    mask_k.shape
                                )
                                hf_test_k_metadata["mask_dtype"] = str(
                                    mask_k.dtype
                                )
                                hf_test_k_metadata["mask_min"] = float(
                                    mask_k.float().min().item()
                                )
                                hf_test_k_metadata["mask_max"] = float(
                                    mask_k.float().max().item()
                                )
                                # Row at d0: how many keys are attended
                                # (mask value == 0) vs masked (-inf-like).
                                try:
                                    d0_mask_row = mask_k[
                                        0, 0, hf_d0_row_index
                                    ]
                                    hf_test_k_metadata[
                                        "d0_mask_num_attended"
                                    ] = int(
                                        (d0_mask_row == 0).sum().item()
                                    )
                                    hf_test_k_metadata[
                                        "d0_mask_num_masked"
                                    ] = int(
                                        (d0_mask_row < 0).sum().item()
                                    )
                                except Exception:
                                    pass
                            else:
                                hf_test_k_metadata["mask_shape"] = None

                            hf_test_k_metadata["sliding_window"] = getattr(
                                bisect_attn_mod, "sliding_window", None
                            )
                            hf_test_k_metadata["attn_implementation"] = str(
                                bisect_attn_mod.config._attn_implementation
                            )
                            hf_test_k_metadata["layer_idx"] = int(
                                TEST_H_BISECT_LAYER_IDX
                            )
                            hf_test_k_metadata["num_heads"] = int(num_heads_k)
                            hf_test_k_metadata["num_kv_heads"] = int(
                                num_kv_heads_k
                            )
                            try:
                                import torch.backends.cuda as _K_CUDA  # noqa: PLC0415

                                hf_test_k_metadata["sdp_flags"] = {
                                    "flash_sdp_enabled": bool(
                                        _K_CUDA.flash_sdp_enabled()
                                    ),
                                    "mem_efficient_sdp_enabled": bool(
                                        _K_CUDA.mem_efficient_sdp_enabled()
                                    ),
                                    "math_sdp_enabled": bool(
                                        _K_CUDA.math_sdp_enabled()
                                    ),
                                    "cudnn_sdp_enabled": bool(
                                        getattr(
                                            _K_CUDA, "cudnn_sdp_enabled", lambda: False
                                        )()
                                    ),
                                }
                            except Exception as _e:
                                hf_test_k_metadata["sdp_flags_error"] = str(_e)

                            def _k_capture(name: str, fn):
                                try:
                                    out = fn()
                                    if out is None:
                                        hf_test_k_metadata[
                                            f"{name}_error"
                                        ] = "returned None"
                                        return
                                    # out shape: (B, H, q_len, D).
                                    # Collapse to (B, q_len, H*D) then pick d0.
                                    out2 = (
                                        out.transpose(1, 2)
                                        .reshape(
                                            out.shape[0], out.shape[2], -1
                                        )
                                    )
                                    if hf_d0_row_index < out2.shape[1]:
                                        hf_test_k_variants[name] = (
                                            out2[0, hf_d0_row_index]
                                            .detach()
                                            .clone()
                                        )
                                except Exception as _e:
                                    hf_test_k_metadata[
                                        f"{name}_error"
                                    ] = str(_e)

                            # K1: MATH backend, bf16 inputs.
                            def _k1():
                                with sdpa_kernel([SDPBackend.MATH]):
                                    return _K_F.scaled_dot_product_attention(
                                        q_k,
                                        k_k,
                                        v_k,
                                        attn_mask=mask_k,
                                        is_causal=False,
                                        scale=scaling_k,
                                    )

                            _k_capture("k1_math_bf16", _k1)

                            # K2: EFFICIENT_ATTENTION backend, bf16.
                            def _k2():
                                with sdpa_kernel(
                                    [SDPBackend.EFFICIENT_ATTENTION]
                                ):
                                    return _K_F.scaled_dot_product_attention(
                                        q_k,
                                        k_k,
                                        v_k,
                                        attn_mask=mask_k,
                                        is_causal=False,
                                        scale=scaling_k,
                                    )

                            _k_capture("k2_efficient_bf16", _k2)

                            # K3: FLASH_ATTENTION backend, bf16.  Flash often
                            # rejects additive masks; the error is captured
                            # and reported.
                            def _k3():
                                with sdpa_kernel(
                                    [SDPBackend.FLASH_ATTENTION]
                                ):
                                    return _K_F.scaled_dot_product_attention(
                                        q_k,
                                        k_k,
                                        v_k,
                                        attn_mask=mask_k,
                                        is_causal=False,
                                        scale=scaling_k,
                                    )

                            _k_capture("k3_flash_bf16", _k3)

                            # K4: CUDNN_ATTENTION backend, bf16.
                            def _k4():
                                with sdpa_kernel(
                                    [SDPBackend.CUDNN_ATTENTION]
                                ):
                                    return _K_F.scaled_dot_product_attention(
                                        q_k,
                                        k_k,
                                        v_k,
                                        attn_mask=mask_k,
                                        is_causal=False,
                                        scale=scaling_k,
                                    )

                            _k_capture("k4_cudnn_bf16", _k4)

                            # K5: MATH backend with fp32-promoted inputs.
                            # Should reproduce textbook SDPA up to rounding.
                            def _k5():
                                with sdpa_kernel([SDPBackend.MATH]):
                                    out_fp32 = (
                                        _K_F.scaled_dot_product_attention(
                                            q_k.float(),
                                            k_k.float(),
                                            v_k.float(),
                                            attn_mask=(
                                                mask_k.float()
                                                if isinstance(
                                                    mask_k, torch.Tensor
                                                )
                                                else mask_k
                                            ),
                                            is_causal=False,
                                            scale=scaling_k,
                                        )
                                    )
                                    return out_fp32.to(q_k.dtype)

                            _k_capture("k5_math_fp32", _k5)

                            # K6: default backend selection (what torch
                            # picks automatically; should match the kernel
                            # output).
                            def _k6():
                                return _K_F.scaled_dot_product_attention(
                                    q_k,
                                    k_k,
                                    v_k,
                                    attn_mask=mask_k,
                                    is_causal=False,
                                    scale=scaling_k,
                                )

                            _k_capture("k6_default_bf16", _k6)

                            # K7: manual SDPA entirely in bf16 (no fp32
                            # promotion anywhere).  Probes whether the 6%
                            # attenuation comes purely from bf16 softmax
                            # precision independent of torch kernel choice.
                            def _k7():
                                scores = (
                                    torch.einsum(
                                        "bhqd,bhkd->bhqk", q_k, k_k
                                    )
                                    * scaling_k
                                )
                                if isinstance(mask_k, torch.Tensor):
                                    scores = scores + mask_k.to(scores.dtype)
                                attn_w = torch.softmax(scores, dim=-1)
                                out = torch.einsum(
                                    "bhqk,bhkd->bhqd", attn_w, v_k
                                )
                                return out

                            _k_capture("k7_manual_bf16", _k7)

                            # K8: manual SDPA with fp32 scores+softmax then
                            # bf16 matmul with V.  Isolates *softmax*
                            # precision specifically.
                            def _k8():
                                scores_fp32 = (
                                    torch.einsum(
                                        "bhqd,bhkd->bhqk",
                                        q_k.float(),
                                        k_k.float(),
                                    )
                                    * scaling_k
                                )
                                if isinstance(mask_k, torch.Tensor):
                                    scores_fp32 = scores_fp32 + mask_k.float()
                                attn_w_fp32 = torch.softmax(
                                    scores_fp32, dim=-1
                                )
                                out = torch.einsum(
                                    "bhqk,bhkd->bhqd",
                                    attn_w_fp32.to(v_k.dtype),
                                    v_k,
                                )
                                return out

                            _k_capture("k8_softmax_fp32_matmul_bf16", _k8)
                        except _TestLSkipTestK:
                            pass
                        except Exception as _e:
                            hf_attn_patch_info["test_k_error"] = str(_e)

                        o_out = bisect_attn_mod.o_proj(attn_output)
                        if hf_d0_row_index < o_out.shape[1]:
                            hf_attn_taps["i5_attn_out_post_oproj"] = (
                                o_out[0, hf_d0_row_index]
                                .detach()
                                .clone()
                            )
                        return o_out, attn_weights

                    bisect_attn_mod.forward = _patched_hf_attn_forward

                draft_output = draft(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=draft_pos_ids,
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=head_is_causal,
                )
                sample_hidden_states = draft_output[:, -block_size + 1 :, :]
                target_lm_head = _target_lm_head(target)
                lm_head_device = _module_device(target_lm_head)
                draft_logits = target_lm_head(
                    sample_hidden_states.to(lm_head_device)
                ).to(device)
                draft_logprobs = torch.log_softmax(
                    draft_logits[0, 0].float(), dim=-1
                )
                topk_tok, topk_lp = sample_topk(draft_logits, tree_width)

                for _h in hf_layer_hook_handles:
                    try:
                        _h.remove()
                    except Exception:
                        pass
                for _h in hf_bisect_hook_handles:
                    try:
                        _h.remove()
                    except Exception:
                        pass
                if hf_attn_patch_info:
                    try:
                        hf_attn_patch_info["module"].forward = (
                            hf_attn_patch_info["original_forward"]
                        )
                    except Exception:
                        pass

                # Test L: snapshot the minimal tap subset at every configured
                # step (step_idx in TEST_L_STEPS).  At step 0 the snapshot
                # shares the underlying tensors with the step-0 reference
                # capture; at later steps the probe re-ran the attention for
                # the purpose of tap capture only.  The detach().clone()s
                # happened inside _patched_hf_attn_forward so the tensors
                # here are safe to persist.
                if hf_test_l_active:
                    slot: dict[str, Any] = {
                        "step": int(step_idx),
                        # HF is teacher-forced: one accepted token per step.
                        "context_len": int(num_input_tokens + step_idx),
                        "attn_taps": {},
                        "ctx_kv": {},
                    }
                    for tap_name in TEST_L_ATTN_KEYS:
                        tap_tensor = hf_attn_taps.get(tap_name)
                        if tap_tensor is not None:
                            slot["attn_taps"][tap_name] = tap_tensor
                    for ctx_name in TEST_L_CTX_KEYS:
                        tap_tensor = hf_attn_taps.get(ctx_name)
                        if tap_tensor is not None:
                            slot["ctx_kv"][ctx_name] = tap_tensor
                    # Test M: also grab the (ctx_len, kv_dim) full-context
                    # snapshots.  These are routed through per-position
                    # fingerprinting at serialization time rather than dumped
                    # as raw float lists, to keep JSON size bounded.
                    for ctx_name in TEST_M_FULL_CTX_KEYS:
                        tap_tensor = hf_attn_taps.get(ctx_name)
                        if tap_tensor is not None:
                            slot["ctx_kv"][ctx_name] = tap_tensor
                    hf_test_l_per_step[int(step_idx)] = slot

                if not reference_capture_written and step_idx == 0:
                    draft_cfg = getattr(draft, "config", None)

                    def _maybe_attr(obj: Any, name: str) -> Any:
                        return getattr(obj, name, None) if obj is not None else None

                    draft_rope_theta = _maybe_attr(draft_cfg, "rope_theta")
                    if draft_rope_theta is None:
                        draft_rope_theta = _maybe_attr(draft_cfg, "rotary_base")
                    draft_vocab_size = (
                        _maybe_attr(draft_cfg, "draft_vocab_size")
                        or _maybe_attr(draft_cfg, "vocab_size")
                        or _target_vocab_size(target)
                    )
                    last_prompt_idx = num_input_tokens - 1
                    per_layer_hidden: dict[int, torch.Tensor] = {}
                    for lid in draft.target_layer_ids:
                        lid_int = int(lid)
                        hs_index = lid_int + 1
                        if hs_index < len(output.hidden_states):
                            per_layer_hidden[lid_int] = (
                                output.hidden_states[hs_index][0, last_prompt_idx]
                                .detach()
                                .clone()
                            )
                    concat_at_last_prompt = target_hidden[0, last_prompt_idx].detach().clone()

                    draft_num_hidden_layers = int(
                        getattr(draft.config, "num_hidden_layers", 0) or 0
                    )
                    draft_post_norm_at_d0 = (
                        draft_output[0, hf_d0_row_index].detach().clone()
                    )
                    last_layer_idx = (
                        max(hf_layer_outputs_at_d0.keys())
                        if hf_layer_outputs_at_d0
                        else None
                    )
                    draft_pre_norm_residual_at_d0 = (
                        hf_layer_outputs_at_d0[last_layer_idx].detach().clone()
                        if last_layer_idx is not None
                        else None
                    )
                    draft_noise_embed_at_d0 = (
                        noise_embedding[0, hf_d0_row_index].detach().clone()
                    )
                    draft_layer1_bisect_payload: dict[str, Any] | None = None
                    if hf_bisect_taps:
                        draft_layer1_bisect_payload = {
                            "layer_idx": int(TEST_H_BISECT_LAYER_IDX),
                            **hf_bisect_taps,
                        }
                    # Test I-1 + I-2 payloads (attn taps + context K/V).
                    draft_layer1_attn_bisect_payload: dict[str, Any] | None = None
                    draft_layer1_context_kv_payload: dict[str, Any] | None = None
                    if hf_attn_taps:
                        ctx_key_tensors = {
                            ck: hf_attn_taps.pop(ck)
                            for ck in (
                                "k_last_context",
                                "k_last_context_pre_rope",
                                "v_last_context",
                            )
                            if ck in hf_attn_taps
                        }
                        if hf_attn_taps:
                            draft_layer1_attn_bisect_payload = {
                                "layer_idx": int(TEST_H_BISECT_LAYER_IDX),
                                **hf_attn_taps,
                            }
                        if ctx_key_tensors:
                            draft_layer1_context_kv_payload = {
                                "layer_idx": int(TEST_H_BISECT_LAYER_IDX),
                                "last_context_index": int(
                                    int(num_input_tokens) - 1
                                ),
                                "num_context": int(num_input_tokens),
                                **ctx_key_tensors,
                            }
                    write_reference_capture(
                        output_path=debug_dir / "reference_capture.json",
                        target_lm_head_weight=_target_lm_head(target).weight,
                        target_embed_tokens_weight=_target_embed_tokens(target).weight,
                        sample_hidden_state=sample_hidden_states[0, 0],
                        query_ids=block_output_ids[0].tolist(),
                        target_embed_of_query_ids=noise_embedding[0],
                        draft_logprobs_on_sample_hidden=draft_logprobs,
                        topk_k=20,
                        draft_mask_token_id=int(draft.mask_token_id),
                        draft_target_layer_ids=list(draft.target_layer_ids),
                        draft_rope_theta=draft_rope_theta,
                        draft_vocab_size=int(draft_vocab_size),
                        target_vocab_size=_target_vocab_size(target),
                        num_prompt_tokens=int(num_input_tokens),
                        target_per_layer_hidden_at_last_prompt=per_layer_hidden,
                        target_concat_hidden_at_last_prompt=concat_at_last_prompt,
                        target_num_hidden_layers=int(
                            _target_num_hidden_layers(target)
                        ),
                        draft_num_hidden_layers=draft_num_hidden_layers,
                        draft_d0_row_index=int(hf_d0_row_index),
                        draft_layer_outputs_at_d0=hf_layer_outputs_at_d0,
                        draft_pre_norm_residual_at_d0=draft_pre_norm_residual_at_d0,
                        draft_post_norm_at_d0=draft_post_norm_at_d0,
                        draft_noise_embed_at_d0=draft_noise_embed_at_d0,
                        draft_layer1_bisect_at_d0=draft_layer1_bisect_payload,
                        draft_layer1_attn_bisect_at_d0=(
                            draft_layer1_attn_bisect_payload
                        ),
                        draft_layer1_context_kv=draft_layer1_context_kv_payload,
                        test_k_hf_sdpa_variants=(
                            hf_test_k_variants if hf_test_k_variants else None
                        ),
                        test_k_hf_sdpa_metadata=(
                            hf_test_k_metadata if hf_test_k_metadata else None
                        ),
                    )
                    reference_capture_written = True
                topk_tok_0 = topk_tok[0].detach().cpu().tolist()
                topk_lp_0 = topk_lp[0].detach().cpu().tolist()
                draft_top1_token = int(topk_tok[0, 0].item())
                past_key_values_draft.crop(start)

                target_step = _target_text_forward(
                    target,
                    torch.tensor(
                        [[root_token]], device=target_input_device, dtype=torch.long
                    ),
                    position_ids=position_ids[:, start : start + 1],
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    logits_to_keep=1,
                    output_hidden_states=True,
                )
                target_next_token = int(
                    target_step.logits[0, 0].argmax(dim=-1).item()
                )
                target_rank = _rank_of_token(draft_logprobs, target_next_token)
                target_logprob = float(draft_logprobs[target_next_token].item())
                target_in_topk = target_next_token in topk_tok_0
                draft_top1_match = draft_top1_token == target_next_token

                step_summary = {
                    "step": step_idx,
                    "position": start,
                    "root_token": root_token,
                    "target_next_token": target_next_token,
                    "draft_top1_token": draft_top1_token,
                    "draft_top1_match": draft_top1_match,
                    "target_in_topk": target_in_topk,
                    "target_rank": target_rank,
                    "target_logprob": target_logprob,
                    "topk_tok_0": topk_tok_0,
                    "topk_lp_0": topk_lp_0,
                }
                steps.append(step_summary)

                target_hidden = _extract_context_feature_on_device(
                    target_step.hidden_states,
                    draft.target_layer_ids,
                    device,
                )
                root_token = target_next_token
                generated_target_tokens.append(root_token)
                start += 1

            top1_values = [1.0 if step["draft_top1_match"] else 0.0 for step in steps]
            topk_values = [1.0 if step["target_in_topk"] else 0.0 for step in steps]
            rank_values = [float(step["target_rank"]) for step in steps]
            lp_values = [float(step["target_logprob"]) for step in steps]

            sample_summary = {
                "sample_index": sample_index,
                "prompt_text": prompt_text,
                "num_prompt_tokens": num_input_tokens,
                "num_steps": len(steps),
                "top1_hit_rate": _mean(top1_values),
                "topk_hit_rate": _mean(topk_values),
                "mean_target_rank": _mean(rank_values),
                "median_target_rank": _median(rank_values),
                "mean_target_logprob": _mean(lp_values),
                "generated_target_text": tokenizer.decode(
                    generated_target_tokens,
                    skip_special_tokens=True,
                ),
                "generated_target_tokens": generated_target_tokens,
                "steps": steps,
            }
            _write_json(sample_dir / "draft_quality_summary.json", sample_summary)

            # Test L: once the reference sample's per-step HF captures are
            # accumulated, serialize them to reference_capture_test_l.json so
            # the vLLM-side post-hook can read and cross-compare with the
            # per-step vLLM-side snapshots.  Only the reference sample
            # contributes (hf_reference_sample_index).
            if (
                sample_index == hf_reference_sample_index
                and hf_test_l_per_step
            ):
                # Serialize: tensors -> CPU float lists.
                def _to_list_l(t: Any) -> list[float] | None:
                    if t is None:
                        return None
                    if hasattr(t, "detach"):
                        return t.detach().cpu().float().tolist()
                    return list(t)

                def _per_position_stats_hf(
                    t: Any, k_first: int = 32
                ) -> dict[str, Any] | None:
                    """HF-side mirror of the vLLM probe's
                    ``_per_position_stats``.  Operates on a (num_positions,
                    kv_dim) or (num_positions, ...) tensor and returns the
                    same schema for the alignment report.
                    """
                    if t is None or not hasattr(t, "detach"):
                        return None
                    tf = t.detach().float()
                    if tf.dim() > 2:
                        tf = tf.reshape(tf.shape[0], -1)
                    elif tf.dim() == 1:
                        tf = tf.unsqueeze(0)
                    k = min(int(k_first), int(tf.shape[-1]))
                    return {
                        "num_positions": int(tf.shape[0]),
                        "kv_dim": int(tf.shape[-1]),
                        "first_k_width": int(k),
                        "norm": tf.norm(dim=-1).cpu().tolist(),
                        "abs_mean": tf.abs().mean(dim=-1).cpu().tolist(),
                        "first_k": tf[:, :k].cpu().tolist(),
                    }

                # Test M: which taps hold full (ctx_len, kv_dim) tensors and
                # need to be serialized as per-position fingerprints rather
                # than as flat float lists.
                _TEST_M_FULL_CTX_TAPS: tuple[str, ...] = (
                    "k_ctx_full_pre_rope",
                    "k_ctx_full_post_rope",
                    "v_ctx_full",
                )

                per_step_serialized: dict[str, Any] = {}
                for s_idx, slot in hf_test_l_per_step.items():
                    attn_taps_ser: dict[str, Any] = {}
                    ctx_kv_ser: dict[str, Any] = {}
                    ctx_per_position_ser: dict[str, Any] = {}
                    for tap_name, tensor in slot.get("attn_taps", {}).items():
                        vec = _to_list_l(tensor)
                        if vec is not None:
                            attn_taps_ser[tap_name] = vec
                    for ck_name, tensor in slot.get("ctx_kv", {}).items():
                        if ck_name in _TEST_M_FULL_CTX_TAPS:
                            # Route full ctx tensors through per-position
                            # fingerprinting; store under ``ctx_per_position``
                            # using a stable key name that matches vLLM's
                            # probe payload for the alignment report.
                            key_map = {
                                "k_ctx_full_pre_rope": "k_ctx_pre_rope",
                                "k_ctx_full_post_rope": "k_ctx_post_rope",
                                "v_ctx_full": "v_ctx",
                            }
                            stats = _per_position_stats_hf(tensor)
                            if stats is not None:
                                ctx_per_position_ser[key_map[ck_name]] = stats
                            continue
                        vec = _to_list_l(tensor)
                        if vec is not None:
                            ctx_kv_ser[ck_name] = vec
                    per_step_serialized[str(int(s_idx))] = {
                        "step": int(slot.get("step", s_idx)),
                        "context_len": int(slot.get("context_len", 0)),
                        "attn_taps": attn_taps_ser,
                        "ctx_kv": ctx_kv_ser,
                        "ctx_per_position": ctx_per_position_ser,
                    }
                test_l_payload = {
                    "sample_index": int(sample_index),
                    "bisect_layer_idx": int(TEST_H_BISECT_LAYER_IDX),
                    "d0_row_index": 1,
                    "steps_configured": [int(s) for s in TEST_L_PROBE_STEPS],
                    "per_step": per_step_serialized,
                }
                _write_json(
                    debug_dir / "reference_capture_test_l.json",
                    test_l_payload,
                )

            per_sample.append(sample_summary)
            aggregate_top1.extend(top1_values)
            aggregate_topk.extend(topk_values)
            aggregate_rank.extend(rank_values)
            aggregate_lp.extend(lp_values)

        summary = {
            "source": "draft_quality_target_forced",
            "note": (
                "Evaluates intrinsic draft-model quality along the target greedy "
                "trajectory using HF target+draft models, without the vLLM engine."
            ),
            "debug_dir": str(debug_dir),
            "config": {
                "model": model,
                "draft_model": draft_model_path,
                "block_size": block_size,
                "tree_width": tree_width,
                "max_steps": max_steps,
                "attn_implementation": attn_implementation,
                "seed": seed,
                "target_device_map_auto": target_device_map_auto,
                "target_device_map_gpus": target_device_map_gpus,
                "target_hf_device_map": getattr(target, "hf_device_map", None),
            },
            "sample_indices": sample_indices,
            "num_samples": len(per_sample),
            "num_steps": sum(int(sample["num_steps"]) for sample in per_sample),
            "top1_hit_rate": _mean(aggregate_top1),
            "topk_hit_rate": _mean(aggregate_topk),
            "mean_target_rank": _mean(aggregate_rank),
            "median_target_rank": _median(aggregate_rank),
            "mean_target_logprob": _mean(aggregate_lp),
            "samples": per_sample,
        }
        _write_json(debug_dir / "draft_quality_diagnostic_summary.json", summary)
        return summary
    finally:
        input_ids = None
        position_ids = None
        past_key_values_target = None
        past_key_values_draft = None
        output = None
        target_step = None
        target_hidden = None
        noise_embedding = None
        draft_output = None
        sample_hidden_states = None
        draft_logits = None
        draft_logprobs = None
        topk_tok = None
        topk_lp = None
        block_output_ids = None
        target_lm_head = None
        target_embed_tokens = None
        tokenizer = None
        prompt_bank = None
        target = None
        draft = None
        _cleanup_cuda_stage(
            debug_dir=debug_dir,
            label="post_target_forced_cleanup",
        )


def run_vllm_draft_quality_diagnostic(
    *,
    debug_dir: Path,
    model: str,
    draft_model_path: str,
    sample_indices: list[int],
    max_model_len: int = 32768,
    block_size: int = 16,
    tree_width: int = 7,
    max_steps: int = 256,
    max_tree_budget: int = 255,
    tree_draft: str = "accum_logp",
    tree_hybrid_alpha: float = 1.0,
    max_draft_passes: int = 1,
    tree_prune_ratio: float = 0.25,
    tree_construction: str = "breadth_first",
    tree_attn_kernel: str = "optimus",
    attention_backend: str = "FLASH_ATTN",
    seed: int = 0,
    tp_size: int = 1,
    gpu_memory_utilization: float = 0.5,
    max_num_batched_tokens: int = 51200,
    enable_expert_parallel: bool = False,
    disable_cascade_attn: bool = False,
    enforce_eager: bool = True,
    summary_filename: str = "vllm_draft_quality_diagnostic_summary.json",
    install_audit_hooks: bool = True,
    run_output_subdir_name: str = "vllm_draft_quality_profile",
    effective_num_speculative_tokens: int | None = None,
    install_chain_spec_topk_probe: bool = False,
    chain_spec_topk_width: int = 8,
    install_test_l_only_probe: bool = False,
    test_l_steps_override: tuple[int, ...] | None = None,
    test_l_capture_filename: str = "vllm_test_l_probe_capture.json",
    test_q_capture_filename: str = "vllm_test_q_probe_capture.json",
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        **tokenizer_load_kwargs(model),
    )
    prompt_bank = apply_chat_template(
        tokenizer,
        get_prompt_bank("humaneval"),
        model,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_steps)
    args = SimpleNamespace(
        model=model,
        draft_model=draft_model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=1,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        enable_expert_parallel=enable_expert_parallel,
        disable_cascade_attn=disable_cascade_attn,
        cudagraph_mode="none",
        attention_backend=attention_backend,
        head_type="causal",
        tree_width=tree_width,
        max_tree_budget=max_tree_budget,
        tree_draft=tree_draft,
        tree_hybrid_alpha=tree_hybrid_alpha,
        max_draft_passes=max_draft_passes,
        tree_prune_ratio=tree_prune_ratio,
        tree_construction=tree_construction,
        tree_attn_kernel=tree_attn_kernel,
        tree_kv_layout="physical",
        num_cudagraph_tree_captures=0,
        num_warmup_runs=0,
        num_runs=1,
        profiler="cuda",
        write_dflash_debug_artifacts=True,
        skip_engine_cleanup=False,
        sleep_after_stop=0.0,
    )

    per_sample: list[dict[str, Any]] = []
    aggregate_top1: list[float] = []
    aggregate_topk: list[float] = []
    aggregate_accept_d0: list[float] = []

    reference_capture_path = debug_dir / "reference_capture.json"
    reference_capture: dict[str, Any] | None = None
    if install_audit_hooks and reference_capture_path.exists():
        reference_capture = json.loads(reference_capture_path.read_text(encoding="utf-8"))

    audit_done = False

    for sample_index in sample_indices:
        prompt_text = prompt_bank[sample_index]
        sample_dir = debug_dir / f"sample_{sample_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        run_output_dir = sample_dir / run_output_subdir_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        pre_hook = None
        post_hook = None
        if install_chain_spec_topk_probe or install_test_l_only_probe:
            # This branch services the chain-spec (Test N-2 / Test O) run.
            # We may be asked to install one or both of:
            #   (a) chain-spec topk probe: populates ``_topk_log`` for
            #       tree_width=1 runs (see install_vllm_chain_spec_topk_log_probe
            #       docstring for the why).  This yields per-step top-k
            #       + target outcome entries that feed Test N-1's
            #       same-prefix comparison.
            #   (b) Test-L-only probe: captures layer-1 self-attn taps at
            #       TEST_O_STEPS (including step 0 since no legacy probe is
            #       installed here).  Feeds Test O's tree-vs-chain-vs-HF
            #       per-step comparison.
            _topk_width = int(chain_spec_topk_width)
            _install_topk = bool(install_chain_spec_topk_probe)
            _install_tl = bool(install_test_l_only_probe)
            _tl_steps = (
                tuple(int(s) for s in test_l_steps_override)
                if test_l_steps_override is not None
                else tuple(int(s) for s in TEST_O_PROBE_STEPS)
            )
            # Read reference's d0 row if it exists so Test-L's row
            # selection matches HF exactly (for chain-spec num_spec=1,
            # effective_d0_row = 0 * (1+1) + d0 = d0 directly; but we
            # still prefer the reference-authored value if present).
            _ref_d0 = 1
            _rc_path = debug_dir / "reference_capture.json"
            if _rc_path.exists():
                try:
                    _rc = json.loads(_rc_path.read_text(encoding="utf-8"))
                    _ref_d0 = int(_rc.get("draft_d0_row_index") or 1)
                except Exception:
                    _ref_d0 = 1
            _tl_filename = str(test_l_capture_filename)
            _tq_filename = str(test_q_capture_filename)
            _sample_dir = sample_dir

            def pre_hook(
                llm,
                _w=_topk_width,
                _do_topk=_install_topk,
                _do_tl=_install_tl,
                _steps=_tl_steps,
                _d0=_ref_d0,
            ):
                if _do_topk:
                    try:
                        llm.collective_rpc(
                            install_vllm_chain_spec_topk_log_probe,
                            args=(_w,),
                        )
                    except Exception as _e:
                        print(
                            "WARNING: install_vllm_chain_spec_topk_log_probe"
                            f" failed: {_e}"
                        )
                if _do_tl:
                    try:
                        llm.collective_rpc(
                            install_vllm_test_l_probe,
                            args=(
                                _steps,
                                TEST_H_BISECT_LAYER_IDX,
                                _d0,
                                0,
                                None,
                            ),
                        )
                    except Exception as _e:
                        print(
                            "WARNING: install_vllm_test_l_probe (chain) "
                            f"failed: {_e}"
                        )
                    # Test Q: layer-0 hidden-state probe, same capture
                    # steps as Test L.  Uses forward hooks (non-invasive)
                    # so no ordering dependency with the Test-L probe.
                    try:
                        llm.collective_rpc(
                            install_vllm_test_q_probe,
                            args=(
                                _steps,
                                _d0,
                                0,
                                None,
                                0,
                                (_d0, TEST_V_BRANCH_ROW),
                            ),
                        )
                    except Exception as _e:
                        print(
                            "WARNING: install_vllm_test_q_probe (chain) "
                            f"failed: {_e}"
                        )

            def post_hook(
                llm,
                _do_topk=_install_topk,
                _do_tl=_install_tl,
                _sd=_sample_dir,
                _fname=_tl_filename,
                _qfname=_tq_filename,
            ):
                # IMPORTANT retrieval order: retrieve Test-L first so its
                # monkey-patch is unwound before we uninstall topk probe
                # (topk probe touches drafter.propose / _greedy_sample,
                # which are independent of self_attn.forward -- but
                # retrieving Test-L first matches the tree-path order and
                # is therefore a safe default).
                if _do_tl:
                    try:
                        l_caps = llm.collective_rpc(
                            retrieve_vllm_test_l_probe
                        )
                        l_capture = l_caps[0] if l_caps else {}
                    except Exception as _e:
                        print(
                            "WARNING: retrieve_vllm_test_l_probe (chain) "
                            f"failed: {_e}"
                        )
                        l_capture = {"error": str(_e)}
                    try:
                        _write_json(_sd / _fname, l_capture)
                    except Exception as _we:
                        print(
                            "WARNING: writing chain Test-L capture failed:"
                            f" {_we}"
                        )
                    # Test Q (chain-spec side): retrieve layer-0 probe.
                    try:
                        q_caps = llm.collective_rpc(
                            retrieve_vllm_test_q_probe
                        )
                        q_capture = q_caps[0] if q_caps else {}
                    except Exception as _e:
                        print(
                            "WARNING: retrieve_vllm_test_q_probe (chain) "
                            f"failed: {_e}"
                        )
                        q_capture = {"error": str(_e)}
                    try:
                        _write_json(_sd / _qfname, q_capture)
                    except Exception as _we:
                        print(
                            "WARNING: writing chain Test-Q capture failed:"
                            f" {_we}"
                        )
                if _do_topk:
                    try:
                        result = llm.collective_rpc(
                            uninstall_vllm_chain_spec_topk_log_probe
                        )
                        if result and isinstance(result, list):
                            print(
                                "chain-spec topk probe uninstalled: "
                                f"{result[0]}"
                            )
                    except Exception as _e:
                        print(
                            "WARNING: uninstall_vllm_chain_spec_topk_log_probe"
                            f" failed: {_e}"
                        )
        if reference_capture is not None and not audit_done:
            ref_num_prompt = int(reference_capture.get("num_prompt_tokens") or 0)

            ref_draft_d0_row = int(
                reference_capture.get("draft_d0_row_index") or 1
            )

            def pre_hook(llm, _n=ref_num_prompt, _d0=ref_draft_d0_row):
                if _n <= 0:
                    return
                try:
                    llm.collective_rpc(
                        install_vllm_dflash_runtime_bundle_probe,
                        args=(tuple(int(s) for s in TEST_L_PROBE_STEPS),),
                    )
                except Exception as e:
                    print(
                        "WARNING: install_vllm_dflash_runtime_bundle_probe "
                        f"failed: {e}"
                    )
                try:
                    llm.collective_rpc(
                        install_vllm_target_aux_capture,
                        args=(_n, 4),
                    )
                except Exception as e:
                    print(f"WARNING: install_vllm_target_aux_capture failed: {e}")
                try:
                    llm.collective_rpc(
                        install_vllm_draft_layer_capture,
                        args=(_d0, 0, None),
                    )
                except Exception as e:
                    print(
                        f"WARNING: install_vllm_draft_layer_capture failed: {e}"
                    )
                try:
                    llm.collective_rpc(
                        install_vllm_draft_layer1_bisect_capture,
                        args=(_d0, 0, None, TEST_H_BISECT_LAYER_IDX),
                    )
                except Exception as e:
                    print(
                        "WARNING: install_vllm_draft_layer1_bisect_capture"
                        f" failed: {e}"
                    )
                try:
                    llm.collective_rpc(
                        install_vllm_draft_layer1_attn_bisect_capture,
                        args=(_d0, 0, None, TEST_H_BISECT_LAYER_IDX),
                    )
                except Exception as e:
                    print(
                        "WARNING: install_vllm_draft_layer1_attn_bisect_capture"
                        f" failed: {e}"
                    )
                # Test L: multi-step intra-self_attn probe.  Install AFTER
                # the legacy single-shot probe so Test-L's monkey-patch
                # wraps on top of the legacy patched function.  Legacy
                # captures at step 0 (its captured_attn flag short-circuits
                # subsequent calls); Test-L captures at steps > 0 only.
                # At step 0, Test-L's patched precompute/forward see
                # current_step not in {5,15,30} -> pass-through to legacy,
                # which captures.  At step > 0, Test-L captures into its
                # own per_step slot; legacy is pass-through.
                # The post-hook synthesizes a unified step-0 entry into
                # Test-L's per_step list from legacy's attn_taps/ctx_kv so
                # the multistep report sees step 0 on both sides.
                try:
                    llm.collective_rpc(
                        install_vllm_test_l_probe,
                        args=(
                            tuple(int(s) for s in TEST_L_PROBE_STEPS),
                            TEST_H_BISECT_LAYER_IDX,
                            _d0,
                            0,
                            None,
                        ),
                    )
                except Exception as e:
                    print(
                        f"WARNING: install_vllm_test_l_probe failed: {e}"
                    )
                # Test Q: layer-0 hidden-state probe (tree-spec side).
                # Registered via forward hooks on layer[0] and
                # layer[0].self_attn -- non-invasive, no interaction
                # with the Test-L monkey-patch on layer[1].self_attn
                # or on precompute_and_store_context_kv.  Captures at
                # the Test-P position-dense schedule so position
                # matching with chain-spec is maximally dense.
                try:
                    llm.collective_rpc(
                        install_vllm_test_q_probe,
                        args=(
                            tuple(int(s) for s in TEST_L_PROBE_STEPS),
                            _d0,
                            0,
                            None,
                            0,
                                (_d0, TEST_V_BRANCH_ROW),
                        ),
                    )
                except Exception as e:
                    print(
                        f"WARNING: install_vllm_test_q_probe failed: {e}"
                    )

            def post_hook(llm, _ref=reference_capture, _sample_dir=sample_dir):
                sample_hidden = _ref.get("sample_hidden_state") or []
                query_ids = _ref.get("query_ids") or []
                audits = llm.collective_rpc(
                    collect_vllm_draft_audit,
                    args=(sample_hidden, query_ids, 20),
                )
                audit = audits[0] if audits else {}
                _write_json(_sample_dir / "vllm_draft_audit.json", audit)
                _write_json(
                    debug_dir / "vllm_draft_weight_audit_report.json",
                    build_weight_audit_report(audit),
                )
                _write_json(
                    debug_dir / "lm_head_embed_parity_report.json",
                    build_lm_head_embed_parity_report(_ref, audit),
                )
                try:
                    caps = llm.collective_rpc(retrieve_vllm_target_aux_capture)
                    capture = caps[0] if caps else {}
                except Exception as e:
                    print(f"WARNING: retrieve_vllm_target_aux_capture failed: {e}")
                    capture = {"error": str(e)}
                _write_json(
                    _sample_dir / "vllm_target_aux_capture.json", capture
                )
                _write_json(
                    debug_dir / "target_hidden_parity_report.json",
                    build_target_hidden_parity_report(_ref, capture),
                )
                try:
                    lyr_caps = llm.collective_rpc(
                        retrieve_vllm_draft_layer_capture
                    )
                    draft_layer_capture = lyr_caps[0] if lyr_caps else {}
                except Exception as e:
                    print(
                        f"WARNING: retrieve_vllm_draft_layer_capture failed: {e}"
                    )
                    draft_layer_capture = {"error": str(e)}
                _write_json(
                    _sample_dir / "vllm_draft_layer_capture.json",
                    draft_layer_capture,
                )
                _write_json(
                    debug_dir / "draft_layer_parity_report.json",
                    build_draft_layer_parity_report(_ref, draft_layer_capture),
                )
                try:
                    bisect_caps = llm.collective_rpc(
                        retrieve_vllm_draft_layer1_bisect_capture
                    )
                    bisect_capture = bisect_caps[0] if bisect_caps else {}
                except Exception as e:
                    print(
                        "WARNING: retrieve_vllm_draft_layer1_bisect_capture"
                        f" failed: {e}"
                    )
                    bisect_capture = {"error": str(e)}
                _write_json(
                    _sample_dir / "vllm_draft_layer1_bisect_capture.json",
                    bisect_capture,
                )
                _write_json(
                    debug_dir / "draft_layer1_bisect_report.json",
                    build_draft_layer1_bisect_report(_ref, bisect_capture),
                )
                # IMPORTANT retrieval order: Test-L was installed AFTER the
                # legacy attn-bisect probe (install_vllm_test_l_probe is
                # called AFTER install_vllm_draft_layer1_attn_bisect_capture
                # in the pre-generation hook), so Test-L's monkey-patch
                # currently wraps legacy's.  retrieve_vllm_test_l_probe
                # restores self_attn.forward to the LEGACY patched function
                # (Test-L's original_self_attn_forward).  Only *then* can
                # retrieve_vllm_draft_layer1_attn_bisect_capture properly
                # unwind legacy's patch back to the real forward.  Running
                # legacy's retrieval first would jump straight to the real
                # forward and Test-L's subsequent retrieval would re-install
                # a stale legacy_patched reference on top.
                try:
                    l_caps = llm.collective_rpc(retrieve_vllm_test_l_probe)
                    l_capture = l_caps[0] if l_caps else {}
                except Exception as e:
                    print(
                        f"WARNING: retrieve_vllm_test_l_probe failed: {e}"
                    )
                    l_capture = {"error": str(e)}

                try:
                    attn_caps = llm.collective_rpc(
                        retrieve_vllm_draft_layer1_attn_bisect_capture
                    )
                    attn_capture = attn_caps[0] if attn_caps else {}
                except Exception as e:
                    print(
                        "WARNING: retrieve_vllm_draft_layer1_attn_bisect_capture"
                        f" failed: {e}"
                    )
                    attn_capture = {"error": str(e)}
                _write_json(
                    _sample_dir / "vllm_draft_layer1_attn_bisect_capture.json",
                    attn_capture,
                )
                _write_json(
                    debug_dir / "draft_layer1_attn_bisect_report.json",
                    build_draft_layer1_attn_bisect_report(_ref, attn_capture),
                )
                # Test K uses reference-only variants (already persisted in
                # reference_capture.json) plus the vLLM kernel output from
                # the attn_bisect capture for cross-comparison.
                try:
                    _write_json(
                        debug_dir / "test_k_hf_sdpa_backend_report.json",
                        build_test_k_hf_sdpa_report(_ref, attn_capture),
                    )
                except Exception as _tk_err:
                    _write_json(
                        debug_dir / "test_k_hf_sdpa_backend_report.json",
                        {"error": str(_tk_err)},
                    )
                try:
                    legacy_attn_taps = attn_capture.get("attn_taps") or {}
                    legacy_ctx_kv = attn_capture.get("ctx_kv") or {}
                    if legacy_attn_taps or legacy_ctx_kv:
                        step0_entry = {
                            "step": 0,
                            "context_len": int(
                                legacy_ctx_kv.get("num_context") or 0
                            ),
                            "attn_taps": {
                                k: v
                                for k, v in legacy_attn_taps.items()
                                if isinstance(v, list)
                            },
                            "ctx_kv": {
                                k: v
                                for k, v in legacy_ctx_kv.items()
                                if isinstance(v, list)
                            },
                            "attn_stats": attn_capture.get("attn_stats")
                            or {},
                            "source": (
                                "synthesized_from_legacy_attn_bisect_capture"
                            ),
                        }
                        existing = list(l_capture.get("per_step") or [])
                        if not any(
                            int(e.get("step", -1)) == 0 for e in existing
                        ):
                            l_capture = dict(l_capture)
                            l_capture["per_step"] = [step0_entry] + existing
                except Exception as _splice_err:
                    print(
                        "WARNING: splicing legacy step-0 into Test L failed:"
                        f" {_splice_err}"
                    )
                _write_json(
                    _sample_dir / "vllm_test_l_probe_capture.json",
                    l_capture,
                )
                # Test Q (tree-spec side): retrieve layer-0 hidden-state
                # probe.  Must be retrieved AFTER Test-L has been torn down
                # so any self_attn monkey-patch on layer[1] is fully
                # restored first (Test Q does not touch layer[1], so this
                # is not strictly required, but mirrors the safe order).
                try:
                    q_caps = llm.collective_rpc(retrieve_vllm_test_q_probe)
                    q_capture = q_caps[0] if q_caps else {}
                except Exception as _qe:
                    print(
                        f"WARNING: retrieve_vllm_test_q_probe failed: {_qe}"
                    )
                    q_capture = {"error": str(_qe)}
                _write_json(
                    _sample_dir / "vllm_test_q_probe_capture.json",
                    q_capture,
                )
                try:
                    rb_caps = llm.collective_rpc(
                        retrieve_vllm_dflash_runtime_bundle
                    )
                    runtime_bundle = rb_caps[0] if rb_caps else {}
                except Exception as _rbe:
                    print(
                        "WARNING: retrieve_vllm_dflash_runtime_bundle failed: "
                        f"{_rbe}"
                    )
                    runtime_bundle = {"error": str(_rbe)}
                _write_json(
                    _sample_dir / "vllm_dflash_runtime_bundle.json",
                    runtime_bundle,
                )
                try:
                    ag_caps = llm.collective_rpc(retrieve_vllm_tree_attn_builder_probe)
                    builder_probe = ag_caps[0] if ag_caps else {}
                except Exception as _age:
                    print(
                        "WARNING: retrieve_vllm_tree_attn_builder_probe failed: "
                        f"{_age}"
                    )
                    builder_probe = {"error": str(_age)}
                _write_json(
                    _sample_dir / "vllm_tree_attn_builder_probe.json",
                    builder_probe,
                )
                try:
                    ah_caps = llm.collective_rpc(
                        retrieve_vllm_drafter_first_pass_metadata_probe
                    )
                    drafter_first_pass_probe = ah_caps[0] if ah_caps else {}
                except Exception as _ahe:
                    print(
                        "WARNING: retrieve_vllm_drafter_first_pass_metadata_probe "
                        f"failed: {_ahe}"
                    )
                    drafter_first_pass_probe = {"error": str(_ahe)}
                _write_json(
                    _sample_dir / "vllm_drafter_first_pass_metadata_probe.json",
                    drafter_first_pass_probe,
                )
                # Load HF-side per-step payload (written at end of reference
                # sample) and cross-compare.
                hf_test_l_path = debug_dir / "reference_capture_test_l.json"
                hf_test_l_payload: dict[str, Any] = {}
                if hf_test_l_path.exists():
                    try:
                        hf_test_l_payload = json.loads(
                            hf_test_l_path.read_text(encoding="utf-8")
                        )
                    except Exception as _re:
                        hf_test_l_payload = {"error": str(_re)}
                try:
                    _write_json(
                        debug_dir / "draft_layer1_multistep_report.json",
                        build_draft_layer1_multistep_report(
                            hf_test_l_payload,
                            _ref,
                            l_capture,
                        ),
                    )
                except Exception as _lrerr:
                    _write_json(
                        debug_dir / "draft_layer1_multistep_report.json",
                        {"error": str(_lrerr)},
                    )
                # Test M: per-position context-K/V alignment across the
                # FULL context window.  Uses the ``ctx_per_position`` dicts
                # persisted by both the HF probe (into
                # ``reference_capture_test_l.json``) and the vLLM probe
                # (inside l_capture).  This report confirms whether
                # num_context=255 is real, whether overlap positions match
                # pointwise, and characterizes vLLM's tail beyond HF's
                # context window.
                try:
                    _write_json(
                        debug_dir / "draft_layer1_context_kv_alignment_report.json",
                        build_draft_layer1_context_kv_alignment_report(
                            hf_test_l_payload,
                            l_capture,
                        ),
                    )
                except Exception as _merr:
                    _write_json(
                        debug_dir / "draft_layer1_context_kv_alignment_report.json",
                        {"error": str(_merr)},
                    )

        _effective_spec_tokens = (
            effective_num_speculative_tokens
            if effective_num_speculative_tokens is not None
            else block_size - 1
        )
        result = run_native_profile(
            args=args,
            mode="dflash",
            prompt_batches=[[prompt_text]],
            sampling_params=sampling_params,
            tp_size=tp_size,
            batch_size=1,
            effective_num_speculative_tokens=_effective_spec_tokens,
            effective_max_num_seqs=1,
            profiler_config={"profiler": "cuda"},
            run_output_dir=run_output_dir,
            pre_generation_hook=pre_hook,
            post_generation_hook=post_hook,
        )
        if post_hook is not None:
            audit_done = True

        topk_path = run_output_dir / "topk_log.json"
        topk_entries: list[dict[str, Any]] = []
        if topk_path.exists():
            topk_entries = json.loads(topk_path.read_text(encoding="utf-8"))

        steps = [
            entry
            for entry in topk_entries
            if entry.get("req", 0) == 0 and entry.get("target_next_token") is not None
        ]
        top1_values = [
            1.0 for step in steps if bool(step.get("draft_top1_match"))
        ]
        top1_misses = [
            0.0 for step in steps if not bool(step.get("draft_top1_match"))
        ]
        topk_values = [
            1.0 for step in steps if bool(step.get("target_in_topk"))
        ]
        topk_misses = [
            0.0 for step in steps if not bool(step.get("target_in_topk"))
        ]
        d0_accept_values = [
            1.0 for step in steps if bool(step.get("accepted_d0"))
        ]
        d0_accept_misses = [
            0.0 for step in steps if not bool(step.get("accepted_d0"))
        ]

        outputs = result.get("outputs", [])
        sample_summary = {
            "sample_index": sample_index,
            "prompt_text": prompt_text,
            "output_text": outputs[0][1] if outputs else "",
            "num_steps": len(steps),
            "top1_hit_rate": _mean(top1_values + top1_misses),
            "topk_hit_rate": _mean(topk_values + topk_misses),
            "d0_acceptance_rate": _mean(d0_accept_values + d0_accept_misses),
            "acceptance_rate": float(result["acceptance_rate"]),
            "acceptance_length": float(result["acceptance_length"]),
            "per_pos_acceptance_rate": list(result.get("per_pos_acceptance_rate", [])),
            "topk_log_path": str(topk_path),
            "steps": steps,
        }
        _per_sample_summary_filename = (
            "vllm_draft_quality_summary.json"
            if run_output_subdir_name == "vllm_draft_quality_profile"
            else f"vllm_draft_quality_summary__{run_output_subdir_name}.json"
        )
        _write_json(sample_dir / _per_sample_summary_filename, sample_summary)
        per_sample.append(sample_summary)
        aggregate_top1.extend(top1_values + top1_misses)
        aggregate_topk.extend(topk_values + topk_misses)
        aggregate_accept_d0.extend(d0_accept_values + d0_accept_misses)

    summary = {
        "source": "vllm_depth0_verified",
        "note": (
            "Measures vLLM proposer depth-0 draft quality on the executed vLLM "
            "trajectory by combining proposer top-k logs with verifier-side target tokens."
        ),
        "debug_dir": str(debug_dir),
        "config": {
            "model": model,
            "draft_model": draft_model_path,
            "tp_size": tp_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_batched_tokens": max_num_batched_tokens,
            "enable_expert_parallel": enable_expert_parallel,
            "disable_cascade_attn": disable_cascade_attn,
            "max_model_len": max_model_len,
            "block_size": block_size,
            "tree_width": tree_width,
            "max_steps": max_steps,
            "max_tree_budget": max_tree_budget,
            "tree_draft": tree_draft,
            "tree_hybrid_alpha": tree_hybrid_alpha,
            "max_draft_passes": max_draft_passes,
            "tree_prune_ratio": tree_prune_ratio,
            "tree_construction": tree_construction,
            "tree_attn_kernel": tree_attn_kernel,
            "attention_backend": attention_backend,
            "enforce_eager": enforce_eager,
            "seed": seed,
        },
        "sample_indices": sample_indices,
        "num_samples": len(per_sample),
        "num_steps": sum(int(sample["num_steps"]) for sample in per_sample),
        "top1_hit_rate": _mean(aggregate_top1),
        "topk_hit_rate": _mean(aggregate_topk),
        "d0_acceptance_rate": _mean(aggregate_accept_d0),
        "samples": per_sample,
    }
    _write_json(debug_dir / summary_filename, summary)
    return summary


def build_draft_quality_comparison_summary(
    *,
    debug_dir: Path,
    target_forced_summary: dict[str, Any],
    vllm_summary: dict[str, Any],
) -> dict[str, Any]:
    target_by_index = {
        int(sample["sample_index"]): sample for sample in target_forced_summary["samples"]
    }
    vllm_by_index = {
        int(sample["sample_index"]): sample for sample in vllm_summary["samples"]
    }
    sample_indices = sorted(set(target_by_index).union(vllm_by_index))
    samples = []
    for sample_index in sample_indices:
        target_sample = target_by_index.get(sample_index, {})
        vllm_sample = vllm_by_index.get(sample_index, {})
        samples.append(
            {
                "sample_index": sample_index,
                "target_forced_top1_hit_rate": target_sample.get("top1_hit_rate"),
                "target_forced_topk_hit_rate": target_sample.get("topk_hit_rate"),
                "target_forced_num_steps": target_sample.get("num_steps"),
                "vllm_depth0_top1_hit_rate": vllm_sample.get("top1_hit_rate"),
                "vllm_depth0_topk_hit_rate": vllm_sample.get("topk_hit_rate"),
                "vllm_d0_acceptance_rate": vllm_sample.get("d0_acceptance_rate"),
                "vllm_num_steps": vllm_sample.get("num_steps"),
            }
        )

    summary = {
        "debug_dir": str(debug_dir),
        "target_forced": {
            "summary_path": str(debug_dir / "draft_quality_diagnostic_summary.json"),
            "top1_hit_rate": target_forced_summary["top1_hit_rate"],
            "topk_hit_rate": target_forced_summary["topk_hit_rate"],
            "num_steps": target_forced_summary["num_steps"],
        },
        "vllm_depth0_verified": {
            "summary_path": str(debug_dir / "vllm_draft_quality_diagnostic_summary.json"),
            "top1_hit_rate": vllm_summary["top1_hit_rate"],
            "topk_hit_rate": vllm_summary["topk_hit_rate"],
            "d0_acceptance_rate": vllm_summary["d0_acceptance_rate"],
            "num_steps": vllm_summary["num_steps"],
        },
        "samples": samples,
    }
    _write_json(debug_dir / "draft_quality_comparison_summary.json", summary)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose draft-model quality with the existing HF target-forced path "
            "and an optional vLLM proposer-side depth-0 diagnostic."
        ),
    )
    parser.add_argument("--debug-dir")
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--sample-indices", default="0,1")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tree-width", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help=(
            "HF transformers attn implementation for the reference draft/"
            "target. Default 'eager' is compatible with local Step-3.7; "
            "vLLM's tree attention kernel is controlled separately by "
            "--tree-attn-kernel."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-device-map-auto",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Shard the HF target model with device_map='auto'. This avoids "
            "loading the Step-3.7 target and draft on one GPU during the "
            "target-forced reference diagnostic."
        ),
    )
    parser.add_argument(
        "--target-device-map-gpus",
        type=int,
        default=None,
        help=(
            "Number of visible GPUs available to the HF target device map. "
            "Use with --target-device-map-auto, e.g. 4 to balance the target "
            "across cuda:0..cuda:3."
        ),
    )
    parser.add_argument(
        "--run-vllm-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tree-budget", type=int, default=255)
    parser.add_argument("--tree-draft", default="accum_logp")
    parser.add_argument("--tree-hybrid-alpha", type=float, default=1.0)
    parser.add_argument("--max-draft-passes", type=int, default=1)
    parser.add_argument("--tree-prune-ratio", type=float, default=0.25)
    parser.add_argument("--tree-construction", default="breadth_first")
    parser.add_argument("--tree-attn-kernel", default="optimus")
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument(
        "--vllm-tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for the vLLM diagnostic replay.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.5,
        help=(
            "GPU memory utilization for the vLLM diagnostic replay. "
            "Lower than throughput runs to leave room after HF diagnostics."
        ),
    )
    parser.add_argument(
        "--vllm-max-num-batched-tokens",
        type=int,
        default=16384,
        help=(
            "max_num_batched_tokens for the vLLM diagnostic replay. "
            "Default matches the known-good Step-3.7 profiling command."
        ),
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward --enable-expert-parallel into the vLLM diagnostic replay.",
    )
    parser.add_argument(
        "--disable-cascade-attn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward --disable-cascade-attn into the vLLM diagnostic replay.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run-vllm-chain-spec",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Test N-2: after the tree-spec vLLM run, run a second vLLM "
            "invocation with tree_width=1 / max_tree_budget=1 / "
            "max_draft_passes=1 (single-branch chain speculation).  "
            "This makes vLLM's per-iteration semantics identical to "
            "HF's chain-greedy target-forced reference, so any "
            "remaining aggregate top-1 gap is attributable to the "
            "draft stack rather than the speculative-tree machinery."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    debug_dir = _get_debug_dir(args.debug_dir)
    _write_json(
        debug_dir / "pre_target_forced_snapshot.json",
        _capture_cuda_memory_snapshot("pre_target_forced"),
    )
    target_forced_summary = run_draft_quality_diagnostic(
        debug_dir=debug_dir,
        model=args.model,
        draft_model_path=args.draft_model,
        sample_indices=_parse_sample_indices(args.sample_indices),
        block_size=args.block_size,
        tree_width=args.tree_width,
        max_steps=args.max_steps,
        attn_implementation=args.attn_implementation,
        seed=args.seed,
        target_device_map_auto=args.target_device_map_auto,
        target_device_map_gpus=args.target_device_map_gpus,
    )

    if not args.run_vllm_diagnostic:
        print(json.dumps(target_forced_summary, indent=2, sort_keys=True))
        return

    _write_json(
        debug_dir / "pre_vllm_snapshot.json",
        _capture_cuda_memory_snapshot("pre_vllm"),
    )
    vllm_summary = run_vllm_draft_quality_diagnostic(
        debug_dir=debug_dir,
        model=args.model,
        draft_model_path=args.draft_model,
        sample_indices=_parse_sample_indices(args.sample_indices),
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        tree_width=args.tree_width,
        max_steps=args.max_steps,
        max_tree_budget=args.max_tree_budget,
        tree_draft=args.tree_draft,
        tree_hybrid_alpha=args.tree_hybrid_alpha,
        max_draft_passes=args.max_draft_passes,
        tree_prune_ratio=args.tree_prune_ratio,
        tree_construction=args.tree_construction,
        tree_attn_kernel=args.tree_attn_kernel,
        attention_backend=args.attention_backend,
        seed=args.seed,
        tp_size=args.vllm_tp_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        enable_expert_parallel=args.enable_expert_parallel,
        disable_cascade_attn=args.disable_cascade_attn,
        enforce_eager=args.enforce_eager,
    )
    comparison = build_draft_quality_comparison_summary(
        debug_dir=debug_dir,
        target_forced_summary=target_forced_summary,
        vllm_summary=vllm_summary,
    )
    topk_divergence = build_topk_log_divergence_report(
        target_forced_summary=target_forced_summary,
        vllm_summary=vllm_summary,
    )
    _write_json(debug_dir / "topk_log_divergence_report.json", topk_divergence)

    # Test N-2 (optional): force chain-spec mode on vLLM so its
    # per-iteration semantics match HF's chain-greedy target-forced
    # trajectory exactly (one accepted token per round, single branch,
    # single depth).  Requires GPU cleanup after the tree-spec run.
    vllm_chain_summary: dict[str, Any] | None = None
    if args.run_vllm_chain_spec:
        _cleanup_cuda_stage(
            debug_dir=debug_dir,
            label="post_vllm_tree_pre_chain_cleanup",
        )
        try:
            vllm_chain_summary = run_vllm_draft_quality_diagnostic(
                debug_dir=debug_dir,
                model=args.model,
                draft_model_path=args.draft_model,
                sample_indices=_parse_sample_indices(args.sample_indices),
                max_model_len=args.max_model_len,
                block_size=args.block_size,
                tree_width=1,
                max_steps=args.max_steps,
                max_tree_budget=1,
                tree_draft=args.tree_draft,
                tree_hybrid_alpha=args.tree_hybrid_alpha,
                max_draft_passes=1,
                tree_prune_ratio=args.tree_prune_ratio,
                tree_construction=args.tree_construction,
                tree_attn_kernel=args.tree_attn_kernel,
                attention_backend=args.attention_backend,
                seed=args.seed,
                tp_size=args.vllm_tp_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_num_batched_tokens=args.vllm_max_num_batched_tokens,
                enable_expert_parallel=args.enable_expert_parallel,
                disable_cascade_attn=args.disable_cascade_attn,
                enforce_eager=args.enforce_eager,
                summary_filename=(
                    "vllm_draft_quality_diagnostic_summary_chain_spec.json"
                ),
                install_audit_hooks=False,
                run_output_subdir_name="vllm_draft_quality_profile_chain_spec",
                effective_num_speculative_tokens=1,
                install_chain_spec_topk_probe=True,
                chain_spec_topk_width=args.tree_width,
                install_test_l_only_probe=True,
                test_l_steps_override=TEST_O_PROBE_STEPS,
                test_l_capture_filename=(
                    "vllm_test_l_probe_capture_chain.json"
                ),
                test_q_capture_filename=(
                    "vllm_test_q_probe_capture_chain.json"
                ),
            )
        except Exception as _chain_err:
            print(
                f"WARNING: chain-spec vLLM run failed: {_chain_err}",
            )
            _write_json(
                debug_dir / "vllm_chain_spec_run_error.json",
                {"error": str(_chain_err)},
            )
        finally:
            _cleanup_cuda_stage(
                debug_dir=debug_dir,
                label="post_vllm_chain_cleanup",
            )

    # Test N: per-iteration histograms + trajectory/shared-prefix
    # decomposition of the aggregate top-1 gap.  Always runs (tree
    # + HF); includes chain-spec run if it was performed.
    try:
        n_report = build_test_n_per_iteration_report(
            target_forced_summary=target_forced_summary,
            vllm_tree_summary=vllm_summary,
            vllm_chain_summary=vllm_chain_summary,
        )
    except Exception as _nerr:
        n_report = {"error": str(_nerr)}
    _write_json(debug_dir / "test_n_per_iteration_report.json", n_report)

    # Test O: tree-vs-chain Test-L tap A/B at matched steps.  Only runs
    # when chain-spec was enabled AND the chain-spec Test-L capture was
    # saved alongside the tree-spec Test-L capture for the same sample.
    # We attempt the report for every sample that has both artifacts;
    # the canonical artifact lives under sample_<first-index>/ .
    if args.run_vllm_chain_spec:
        try:
            sample_indices_list = _parse_sample_indices(args.sample_indices)
            first_idx = sample_indices_list[0] if sample_indices_list else 0
            sample_dir_o = debug_dir / f"sample_{first_idx:03d}"
            tree_tl_path = sample_dir_o / "vllm_test_l_probe_capture.json"
            chain_tl_path = (
                sample_dir_o / "vllm_test_l_probe_capture_chain.json"
            )
            hf_test_l_path = debug_dir / "reference_capture_test_l.json"
            hf_reference_path = debug_dir / "reference_capture.json"
            missing = [
                str(p)
                for p in (
                    tree_tl_path,
                    chain_tl_path,
                    hf_test_l_path,
                    hf_reference_path,
                )
                if not p.exists()
            ]
            if missing:
                o_report: dict[str, Any] = {
                    "skipped": True,
                    "reason": "missing_inputs",
                    "missing": missing,
                }
            else:
                hf_test_l = json.loads(
                    hf_test_l_path.read_text(encoding="utf-8")
                )
                hf_reference = json.loads(
                    hf_reference_path.read_text(encoding="utf-8")
                )
                tree_tl = json.loads(
                    tree_tl_path.read_text(encoding="utf-8")
                )
                chain_tl = json.loads(
                    chain_tl_path.read_text(encoding="utf-8")
                )
                o_report = build_test_o_tree_vs_chain_layer1_report(
                    hf_test_l_payload=hf_test_l,
                    hf_reference_capture=hf_reference,
                    vllm_tree_capture=tree_tl,
                    vllm_chain_capture=chain_tl,
                )
        except Exception as _oerr:
            o_report = {"error": str(_oerr)}
        _write_json(
            debug_dir / "test_o_tree_vs_chain_layer1_report.json",
            o_report,
        )

        # Test P: position-matched tree-vs-chain internal-state A/B.
        # Reuses the same Test-L probe captures as Test O, but pairs
        # (tree_iter, chain_iter) by RECONSTRUCTED decoded position (from
        # topk_log accepted_len) instead of raw iteration index.  Requires
        # the tree-spec and chain-spec Test-L captures to each span a dense
        # set of iteration indices (see TEST_L_PROBE_STEPS / TEST_O_PROBE_STEPS);
        # matched pairs fall out post-hoc.
        try:
            if (
                chain_tl_path.exists()
                and tree_tl_path.exists()
                and vllm_chain_summary is not None
            ):
                tree_tl_for_p = json.loads(
                    tree_tl_path.read_text(encoding="utf-8")
                )
                chain_tl_for_p = json.loads(
                    chain_tl_path.read_text(encoding="utf-8")
                )
                hf_test_l_for_p: dict[str, Any] | None = None
                hf_reference_for_p: dict[str, Any] | None = None
                if hf_test_l_path.exists():
                    hf_test_l_for_p = json.loads(
                        hf_test_l_path.read_text(encoding="utf-8")
                    )
                if hf_reference_path.exists():
                    hf_reference_for_p = json.loads(
                        hf_reference_path.read_text(encoding="utf-8")
                    )
                p_report = build_test_p_position_matched_report(
                    vllm_tree_capture=tree_tl_for_p,
                    vllm_chain_capture=chain_tl_for_p,
                    vllm_tree_summary=vllm_summary,
                    vllm_chain_summary=vllm_chain_summary,
                    hf_test_l_payload=hf_test_l_for_p,
                    hf_reference_capture=hf_reference_for_p,
                    target_forced_summary=target_forced_summary,
                    sample_index=first_idx,
                )
            else:
                p_report = {
                    "skipped": True,
                    "reason": "missing_inputs_for_test_p",
                    "tree_tl_exists": tree_tl_path.exists(),
                    "chain_tl_exists": chain_tl_path.exists(),
                    "vllm_chain_summary_available": vllm_chain_summary
                    is not None,
                }
        except Exception as _perr:
            p_report = {"error": str(_perr)}
        _write_json(
            debug_dir / "test_p_position_matched_report.json",
            p_report,
        )

        # Test Q: layer-0 hidden-state A/B at matched decoded positions.
        # Uses the Test-Q probe outputs (layer[0] forward hooks on both
        # runs) plus the existing Test-L captures (for
        # context_positions/context_slot_mapping parity).  Same
        # position-matching logic as Test P; no new data-collection pass.
        try:
            tree_tq_path = sample_dir_o / "vllm_test_q_probe_capture.json"
            chain_tq_path = (
                sample_dir_o / "vllm_test_q_probe_capture_chain.json"
            )
            if (
                tree_tq_path.exists()
                and chain_tq_path.exists()
                and vllm_chain_summary is not None
            ):
                tree_tq_for_q = json.loads(
                    tree_tq_path.read_text(encoding="utf-8")
                )
                chain_tq_for_q = json.loads(
                    chain_tq_path.read_text(encoding="utf-8")
                )
                tree_tl_for_q: dict[str, Any] | None = None
                chain_tl_for_q: dict[str, Any] | None = None
                if tree_tl_path.exists():
                    tree_tl_for_q = json.loads(
                        tree_tl_path.read_text(encoding="utf-8")
                    )
                if chain_tl_path.exists():
                    chain_tl_for_q = json.loads(
                        chain_tl_path.read_text(encoding="utf-8")
                    )
                q_report = build_test_q_layer0_report(
                    vllm_tree_q_capture=tree_tq_for_q,
                    vllm_chain_q_capture=chain_tq_for_q,
                    vllm_tree_summary=vllm_summary,
                    vllm_chain_summary=vllm_chain_summary,
                    vllm_tree_l_capture=tree_tl_for_q,
                    vllm_chain_l_capture=chain_tl_for_q,
                    sample_index=first_idx,
                )
            else:
                q_report = {
                    "skipped": True,
                    "reason": "missing_inputs_for_test_q",
                    "tree_tq_exists": tree_tq_path.exists(),
                    "chain_tq_exists": chain_tq_path.exists(),
                    "vllm_chain_summary_available": (
                        vllm_chain_summary is not None
                    ),
                }
        except Exception as _qerr:
            q_report = {"error": str(_qerr)}
        _write_json(
            debug_dir / "test_q_layer0_report.json",
            q_report,
        )

        # Test V: same layer-0 tree-vs-chain A/B as Test Q, but for the
        # first future/branch query row (row-inside-block 2) instead of the
        # depth-0 row. Reuses the same Test-Q capture file because the probe
        # now records both rows in one pass.
        try:
            tree_tq_path = sample_dir_o / "vllm_test_q_probe_capture.json"
            chain_tq_path = (
                sample_dir_o / "vllm_test_q_probe_capture_chain.json"
            )
            if (
                tree_tq_path.exists()
                and chain_tq_path.exists()
                and vllm_chain_summary is not None
            ):
                tree_tq_for_v = json.loads(
                    tree_tq_path.read_text(encoding="utf-8")
                )
                chain_tq_for_v = json.loads(
                    chain_tq_path.read_text(encoding="utf-8")
                )
                tree_tl_for_v: dict[str, Any] | None = None
                chain_tl_for_v: dict[str, Any] | None = None
                if tree_tl_path.exists():
                    tree_tl_for_v = json.loads(
                        tree_tl_path.read_text(encoding="utf-8")
                    )
                if chain_tl_path.exists():
                    chain_tl_for_v = json.loads(
                        chain_tl_path.read_text(encoding="utf-8")
                    )
                v_report = build_test_q_layer0_report(
                    vllm_tree_q_capture=tree_tq_for_v,
                    vllm_chain_q_capture=chain_tq_for_v,
                    vllm_tree_summary=vllm_summary,
                    vllm_chain_summary=vllm_chain_summary,
                    vllm_tree_l_capture=tree_tl_for_v,
                    vllm_chain_l_capture=chain_tl_for_v,
                    sample_index=first_idx,
                    capture_row=TEST_V_BRANCH_ROW,
                    row_label="first future/branch row",
                )
            else:
                v_report = {
                    "skipped": True,
                    "reason": "missing_inputs_for_test_v",
                    "tree_tq_exists": tree_tq_path.exists(),
                    "chain_tq_exists": chain_tq_path.exists(),
                    "vllm_chain_summary_available": (
                        vllm_chain_summary is not None
                    ),
                }
        except Exception as _verr:
            v_report = {"error": str(_verr)}
        _write_json(
            debug_dir / "test_v_branch_query_report.json",
            v_report,
        )

        # Test W: effective (hidden + residual) layer-0 output parity at the
        # depth-0 row.  Reconciles Test Q's split cosines (layer0_output_hidden
        # vs layer0_output_residual) against Test P's layer-1 parity by
        # comparing the *summed* stream that layer 1 actually consumes.
        # Reuses the Test-Q capture files (tree + chain).
        try:
            tree_tq_path = sample_dir_o / "vllm_test_q_probe_capture.json"
            chain_tq_path = (
                sample_dir_o / "vllm_test_q_probe_capture_chain.json"
            )
            if (
                tree_tq_path.exists()
                and chain_tq_path.exists()
                and vllm_chain_summary is not None
            ):
                tree_tq_for_w = json.loads(
                    tree_tq_path.read_text(encoding="utf-8")
                )
                chain_tq_for_w = json.loads(
                    chain_tq_path.read_text(encoding="utf-8")
                )
                w_report = build_test_w_effective_hidden_stream_report(
                    vllm_tree_q_capture=tree_tq_for_w,
                    vllm_chain_q_capture=chain_tq_for_w,
                    vllm_tree_summary=vllm_summary,
                    vllm_chain_summary=vllm_chain_summary,
                    sample_index=first_idx,
                    capture_row=None,  # probe's d0 row (depth-0 mask)
                    row_label="depth-0 row",
                )
            else:
                w_report = {
                    "skipped": True,
                    "reason": "missing_inputs_for_test_w",
                    "tree_tq_exists": tree_tq_path.exists(),
                    "chain_tq_exists": chain_tq_path.exists(),
                    "vllm_chain_summary_available": (
                        vllm_chain_summary is not None
                    ),
                }
        except Exception as _werr:
            w_report = {"error": str(_werr)}
        _write_json(
            debug_dir / "test_w_effective_hidden_stream_report.json",
            w_report,
        )

        # Test X: tree-internal branch-row health audit.  Reuses the same
        # Test-Q capture (tree-side only) to surface per-iteration branch-row
        # norms and row_k-vs-row_1 cosines at each layer-0 tap.  Flags
        # pathological norms or near-orthogonal input cosines that would
        # indicate a branch-specific plumbing / mask bug.  Unlike Test V
        # (which was structurally invalid for tree-vs-chain due to shape
        # asymmetry), this is purely tree-internal and always well defined.
        try:
            tree_tq_path = sample_dir_o / "vllm_test_q_probe_capture.json"
            if tree_tq_path.exists():
                tree_tq_for_x = json.loads(
                    tree_tq_path.read_text(encoding="utf-8")
                )
                x_report = build_test_x_tree_branch_row_health_report(
                    vllm_tree_q_capture=tree_tq_for_x,
                    vllm_tree_summary=vllm_summary,
                    sample_index=first_idx,
                )
            else:
                x_report = {
                    "skipped": True,
                    "reason": "missing_tree_test_q_capture_for_test_x",
                    "tree_tq_exists": tree_tq_path.exists(),
                }
        except Exception as _xerr:
            x_report = {"error": str(_xerr)}
        _write_json(
            debug_dir / "test_x_branch_row_health_report.json",
            x_report,
        )

        # Test Y: targeted tree-only branch-row input-origin audit.
        # Reuses the extended Test-Q capture's per-row metadata
        # (query_input_id/query_position) to answer whether a bad branch row
        # is already malformed before attention despite apparently sane token
        # ids and positions.
        try:
            tree_tq_path = sample_dir_o / "vllm_test_q_probe_capture.json"
            if tree_tq_path.exists():
                tree_tq_for_y = json.loads(
                    tree_tq_path.read_text(encoding="utf-8")
                )
                y_report = build_test_y_branch_input_origin_report(
                    vllm_tree_q_capture=tree_tq_for_y,
                    vllm_tree_summary=vllm_summary,
                    sample_index=first_idx,
                )
            else:
                y_report = {
                    "skipped": True,
                    "reason": "missing_tree_test_q_capture_for_test_y",
                    "tree_tq_exists": tree_tq_path.exists(),
                }
        except Exception as _yerr:
            y_report = {"error": str(_yerr)}
        _write_json(
            debug_dir / "test_y_branch_input_origin_report.json",
            y_report,
        )

        # Test Z: targeted tree-only branch-row audit using the ACTUAL
        # input_ids/positions passed into draft_inner.forward. This is the
        # authoritative follow-up to Test Y and resolves whether a low-cos
        # branch row simply carried a different token id than the anchor row.
        try:
            tree_tq_path = sample_dir_o / "vllm_test_q_probe_capture.json"
            if tree_tq_path.exists():
                tree_tq_for_z = json.loads(
                    tree_tq_path.read_text(encoding="utf-8")
                )
                z_report = build_test_z_actual_forward_input_report(
                    vllm_tree_q_capture=tree_tq_for_z,
                    vllm_tree_summary=vllm_summary,
                    sample_index=first_idx,
                )
            else:
                z_report = {
                    "skipped": True,
                    "reason": "missing_tree_test_q_capture_for_test_z",
                    "tree_tq_exists": tree_tq_path.exists(),
                }
        except Exception as _zerr:
            z_report = {"error": str(_zerr)}
        _write_json(
            debug_dir / "test_z_actual_forward_input_report.json",
            z_report,
        )

        # Test AA: depth-0 visible-prefix audit. Reuses Test-Q forward-attn
        # metadata (seq_lens / slot_mapping summary) plus Test-L context
        # captures to check whether the tree run's *visible* context prefix for
        # the depth-0 row is malformed or mismatched versus chain.
        try:
            tree_tq_path = sample_dir_o / "vllm_test_q_probe_capture.json"
            chain_tq_path = (
                sample_dir_o / "vllm_test_q_probe_capture_chain.json"
            )
            if (
                tree_tq_path.exists()
                and chain_tq_path.exists()
                and tree_tl_path.exists()
                and chain_tl_path.exists()
                and vllm_chain_summary is not None
            ):
                tree_tq_for_aa = json.loads(
                    tree_tq_path.read_text(encoding="utf-8")
                )
                chain_tq_for_aa = json.loads(
                    chain_tq_path.read_text(encoding="utf-8")
                )
                tree_tl_for_aa = json.loads(
                    tree_tl_path.read_text(encoding="utf-8")
                )
                chain_tl_for_aa = json.loads(
                    chain_tl_path.read_text(encoding="utf-8")
                )
                aa_report = build_test_aa_depth0_visible_prefix_report(
                    vllm_tree_q_capture=tree_tq_for_aa,
                    vllm_chain_q_capture=chain_tq_for_aa,
                    vllm_tree_l_capture=tree_tl_for_aa,
                    vllm_chain_l_capture=chain_tl_for_aa,
                    vllm_tree_summary=vllm_summary,
                    vllm_chain_summary=vllm_chain_summary,
                    sample_index=first_idx,
                )
            else:
                aa_report = {
                    "skipped": True,
                    "reason": "missing_inputs_for_test_aa",
                    "tree_tq_exists": tree_tq_path.exists(),
                    "chain_tq_exists": chain_tq_path.exists(),
                    "tree_tl_exists": tree_tl_path.exists(),
                    "chain_tl_exists": chain_tl_path.exists(),
                    "vllm_chain_summary_available": (
                        vllm_chain_summary is not None
                    ),
                }
        except Exception as _aaerr:
            aa_report = {"error": str(_aaerr)}
        _write_json(
            debug_dir / "test_aa_depth0_visible_prefix_report.json",
            aa_report,
        )

        # Test AB / AC: targeted DFlash first-pass audits using the proposer
        # runtime bundles captured from the executed tree path at a selected
        # set of later tree iterations.
        runtime_bundle_path = sample_dir_o / "vllm_dflash_runtime_bundle.json"
        builder_probe_path = sample_dir_o / "vllm_tree_attn_builder_probe.json"
        drafter_probe_path = sample_dir_o / "vllm_drafter_first_pass_metadata_probe.json"
        q_capture_path = sample_dir_o / "vllm_test_q_probe_capture.json"
        try:
            if runtime_bundle_path.exists():
                runtime_bundle = json.loads(
                    runtime_bundle_path.read_text(encoding="utf-8")
                )
                ab_report = build_test_ab_first_pass_context_compaction_report(
                    runtime_bundle,
                    vllm_tree_summary=vllm_summary,
                    sample_index=0,
                )
                ac_report = build_test_ac_first_pass_metadata_consistency_report(
                    runtime_bundle,
                    vllm_tree_summary=vllm_summary,
                    sample_index=0,
                )
            else:
                ab_report = {
                    "skipped": True,
                    "reason": "missing_vllm_dflash_runtime_bundle",
                    "runtime_bundle_exists": runtime_bundle_path.exists(),
                }
                ac_report = dict(ab_report)
        except Exception as _abac_err:
            ab_report = {"error": str(_abac_err)}
            ac_report = {"error": str(_abac_err)}
        _write_json(
            debug_dir / "test_ab_first_pass_context_compaction_report.json",
            ab_report,
        )
        _write_json(
            debug_dir / "test_ac_first_pass_metadata_consistency_report.json",
            ac_report,
        )
        try:
            if runtime_bundle_path.exists():
                runtime_bundle_for_ae = json.loads(
                    runtime_bundle_path.read_text(encoding="utf-8")
                )
                ae_report = build_test_ae_seq_len_derivation_report(
                    runtime_bundle_for_ae,
                    vllm_tree_summary=vllm_summary,
                    sample_index=0,
                )
            else:
                ae_report = {
                    "skipped": True,
                    "reason": "missing_vllm_dflash_runtime_bundle",
                    "runtime_bundle_exists": runtime_bundle_path.exists(),
                }
        except Exception as _ae_err:
            ae_report = {"error": str(_ae_err)}
        _write_json(
            debug_dir / "test_ae_seq_len_derivation_report.json",
            ae_report,
        )
        try:
            if runtime_bundle_path.exists():
                runtime_bundle_for_af = json.loads(
                    runtime_bundle_path.read_text(encoding="utf-8")
                )
                af_report = build_test_af_rejection_count_producer_report(
                    runtime_bundle_for_af,
                    vllm_tree_summary=vllm_summary,
                    sample_index=0,
                )
            else:
                af_report = {
                    "skipped": True,
                    "reason": "missing_vllm_dflash_runtime_bundle",
                    "runtime_bundle_exists": runtime_bundle_path.exists(),
                }
        except Exception as _af_err:
            af_report = {"error": str(_af_err)}
        _write_json(
            debug_dir / "test_af_rejection_count_producer_report.json",
            af_report,
        )
        try:
            if builder_probe_path.exists():
                builder_probe_for_ag = json.loads(
                    builder_probe_path.read_text(encoding="utf-8")
                )
                runtime_bundle_for_ag = (
                    json.loads(runtime_bundle_path.read_text(encoding="utf-8"))
                    if runtime_bundle_path.exists()
                    else None
                )
                ag_report = build_test_ag_tree_attn_builder_passthrough_report(
                    builder_probe_for_ag,
                    runtime_bundle=runtime_bundle_for_ag,
                    vllm_tree_summary=vllm_summary,
                    sample_index=0,
                )
            else:
                ag_report = {
                    "skipped": True,
                    "reason": "missing_tree_attn_builder_probe",
                    "builder_probe_exists": builder_probe_path.exists(),
                }
        except Exception as _ag_err:
            ag_report = {"error": str(_ag_err)}
        _write_json(
            debug_dir / "test_ag_tree_attn_builder_passthrough_report.json",
            ag_report,
        )
        try:
            if drafter_probe_path.exists():
                drafter_probe_for_ah = json.loads(
                    drafter_probe_path.read_text(encoding="utf-8")
                )
                runtime_bundle_for_ah = (
                    json.loads(runtime_bundle_path.read_text(encoding="utf-8"))
                    if runtime_bundle_path.exists()
                    else None
                )
                ah_report = build_test_ah_drafter_first_pass_metadata_report(
                    drafter_probe_for_ah,
                    runtime_bundle=runtime_bundle_for_ah,
                    vllm_tree_summary=vllm_summary,
                    sample_index=0,
                )
            else:
                ah_report = {
                    "skipped": True,
                    "reason": "missing_drafter_first_pass_metadata_probe",
                    "drafter_probe_exists": drafter_probe_path.exists(),
                }
        except Exception as _ah_err:
            ah_report = {"error": str(_ah_err)}
        _write_json(
            debug_dir / "test_ah_drafter_first_pass_metadata_report.json",
            ah_report,
        )
        try:
            if runtime_bundle_path.exists() and q_capture_path.exists():
                runtime_bundle_for_ad = json.loads(
                    runtime_bundle_path.read_text(encoding="utf-8")
                )
                q_capture_for_ad = json.loads(
                    q_capture_path.read_text(encoding="utf-8")
                )
                ad_report = build_test_ad_runtime_vs_forward_metadata_report(
                    runtime_bundle_for_ad,
                    q_capture_for_ad,
                    vllm_tree_summary=vllm_summary,
                    sample_index=0,
                )
            else:
                ad_report = {
                    "skipped": True,
                    "reason": "missing_inputs_for_test_ad",
                    "runtime_bundle_exists": runtime_bundle_path.exists(),
                    "q_capture_exists": q_capture_path.exists(),
                }
        except Exception as _ad_err:
            ad_report = {"error": str(_ad_err)}
        _write_json(
            debug_dir / "test_ad_runtime_vs_forward_metadata_report.json",
            ad_report,
        )

        # Test R: purely offline audit of per-iter ``context_positions`` and
        # ``context_slot_mapping`` on both sides, cross-checked against the
        # HF oracle (``context_len``).  No new inference; reuses the Test-L
        # captures already written above.  Scores H1/H2/H3/H5 from the
        # offset / contiguity fingerprint and returns a single verdict.
        try:
            if (
                tree_tl_path.exists()
                and chain_tl_path.exists()
                and vllm_chain_summary is not None
            ):
                tree_tl_for_r = json.loads(
                    tree_tl_path.read_text(encoding="utf-8")
                )
                chain_tl_for_r = json.loads(
                    chain_tl_path.read_text(encoding="utf-8")
                )
                hf_tl_for_r: dict[str, Any] | None = None
                if hf_test_l_path.exists():
                    hf_tl_for_r = json.loads(
                        hf_test_l_path.read_text(encoding="utf-8")
                    )
                r_report = build_test_r_kv_position_audit(
                    vllm_tree_summary=vllm_summary,
                    vllm_chain_summary=vllm_chain_summary,
                    vllm_tree_l_capture=tree_tl_for_r,
                    vllm_chain_l_capture=chain_tl_for_r,
                    hf_reference_test_l=hf_tl_for_r,
                    sample_index=first_idx,
                )
            else:
                r_report = {
                    "skipped": True,
                    "reason": "missing_inputs_for_test_r",
                    "tree_tl_exists": tree_tl_path.exists(),
                    "chain_tl_exists": chain_tl_path.exists(),
                    "vllm_chain_summary_available": (
                        vllm_chain_summary is not None
                    ),
                }
        except Exception as _rerr:
            r_report = {"error": str(_rerr)}
        _write_json(
            debug_dir / "test_r_kv_position_audit.json",
            r_report,
        )

        # Test S: per-position context K/V content audit (tree vs chain at
        # matched decoded positions).  Directly scores H1 (stale content at
        # accepted-prefix paged slots) vs H2 (bad tail-precompute content).
        # Reuses the Test-L captures; no new inference.  Reads the
        # ``ctx_per_position`` fingerprint (norm / abs_mean / first_k) that
        # the probe already serializes.
        try:
            if (
                tree_tl_path.exists()
                and chain_tl_path.exists()
                and vllm_chain_summary is not None
            ):
                tree_tl_for_s = json.loads(
                    tree_tl_path.read_text(encoding="utf-8")
                )
                chain_tl_for_s = json.loads(
                    chain_tl_path.read_text(encoding="utf-8")
                )
                hf_tl_for_s: dict[str, Any] | None = None
                if hf_test_l_path.exists():
                    hf_tl_for_s = json.loads(
                        hf_test_l_path.read_text(encoding="utf-8")
                    )
                s_report = build_test_s_ctx_content_report(
                    vllm_tree_capture=tree_tl_for_s,
                    vllm_chain_capture=chain_tl_for_s,
                    vllm_tree_summary=vllm_summary,
                    vllm_chain_summary=vllm_chain_summary,
                    hf_test_l_payload=hf_tl_for_s,
                    sample_index=first_idx,
                )
            else:
                s_report = {
                    "skipped": True,
                    "reason": "missing_inputs_for_test_s",
                    "tree_tl_exists": tree_tl_path.exists(),
                    "chain_tl_exists": chain_tl_path.exists(),
                    "vllm_chain_summary_available": (
                        vllm_chain_summary is not None
                    ),
                }
        except Exception as _serr:
            s_report = {"error": str(_serr)}
        _write_json(
            debug_dir / "test_s_ctx_content_report.json",
            s_report,
        )

        # Test T: offline tail-only layout audit at matched decoded positions.
        # Reuses the same Test-L captures as R/S and narrows the residual tree-
        # vs-chain gap to either H2 (tail content/write-back only) or H5
        # (tail slot/position layout anomaly).
        try:
            if (
                tree_tl_path.exists()
                and chain_tl_path.exists()
                and vllm_chain_summary is not None
            ):
                tree_tl_for_t = json.loads(
                    tree_tl_path.read_text(encoding="utf-8")
                )
                chain_tl_for_t = json.loads(
                    chain_tl_path.read_text(encoding="utf-8")
                )
                t_report = build_test_t_tail_layout_report(
                    vllm_tree_capture=tree_tl_for_t,
                    vllm_chain_capture=chain_tl_for_t,
                    vllm_tree_summary=vllm_summary,
                    vllm_chain_summary=vllm_chain_summary,
                    sample_index=first_idx,
                )
            else:
                t_report = {
                    "skipped": True,
                    "reason": "missing_inputs_for_test_t",
                    "tree_tl_exists": tree_tl_path.exists(),
                    "chain_tl_exists": chain_tl_path.exists(),
                    "vllm_chain_summary_available": (
                        vllm_chain_summary is not None
                    ),
                }
        except Exception as _terr:
            t_report = {"error": str(_terr)}
        _write_json(
            debug_dir / "test_t_tail_layout_report.json",
            t_report,
        )

        # Test U: valid-tail-only content audit. Restricts the tree-vs-chain
        # tail comparison to rows with slot != -1 on both sides, i.e. rows that
        # are actually written into paged KV. This tells us whether the current
        # tail-only mismatch is causal or confined to masked rows.
        try:
            if (
                tree_tl_path.exists()
                and chain_tl_path.exists()
                and vllm_chain_summary is not None
            ):
                tree_tl_for_u = json.loads(
                    tree_tl_path.read_text(encoding="utf-8")
                )
                chain_tl_for_u = json.loads(
                    chain_tl_path.read_text(encoding="utf-8")
                )
                u_report = build_test_u_valid_tail_content_report(
                    vllm_tree_capture=tree_tl_for_u,
                    vllm_chain_capture=chain_tl_for_u,
                    vllm_tree_summary=vllm_summary,
                    vllm_chain_summary=vllm_chain_summary,
                    sample_index=first_idx,
                )
            else:
                u_report = {
                    "skipped": True,
                    "reason": "missing_inputs_for_test_u",
                    "tree_tl_exists": tree_tl_path.exists(),
                    "chain_tl_exists": chain_tl_path.exists(),
                    "vllm_chain_summary_available": (
                        vllm_chain_summary is not None
                    ),
                }
        except Exception as _uerr:
            u_report = {"error": str(_uerr)}
        _write_json(
            debug_dir / "test_u_valid_tail_content_report.json",
            u_report,
        )

    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
