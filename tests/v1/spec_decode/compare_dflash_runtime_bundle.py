from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

DEFAULT_REFERENCE_NAME = "reference_step0_runtime_bundle.pt"
DEFAULT_VLLM_NAME = "vllm_step0_runtime_bundle.pt"


def _resolve_bundle(path_or_dir: str, default_name: str) -> Path:
    path = Path(path_or_dir)
    if path.is_dir():
        return path / default_name
    return path


def _compare_tensor(name: str, reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if reference.shape != candidate.shape:
        result["shape_match"] = False
        return result

    result["shape_match"] = True
    if reference.numel() == 0:
        result["empty"] = True
        return result

    if reference.dtype.is_floating_point or candidate.dtype.is_floating_point:
        ref32 = reference.float()
        cand32 = candidate.float()
        diff = (ref32 - cand32).abs()
        result["max_abs_diff"] = float(diff.max().item())
        result["mean_abs_diff"] = float(diff.mean().item())
        result["reference_norm"] = float(ref32.norm().item())
        result["candidate_norm"] = float(cand32.norm().item())
        flat_ref = ref32.reshape(-1)
        flat_cand = cand32.reshape(-1)
        cosine = torch.nn.functional.cosine_similarity(
            flat_ref.unsqueeze(0),
            flat_cand.unsqueeze(0),
        )
        result["cosine_similarity"] = float(cosine.item())
        return result

    unequal = (reference != candidate).sum().item()
    result["exact_match"] = bool(unequal == 0)
    result["num_unequal"] = int(unequal)
    return result


def _as_int_list(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return [int(value.item())]
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        try:
            return [int(x) for x in value]
        except Exception:
            return None
    return None


def _first_index(values: list[int], target: int) -> int | None:
    for idx, value in enumerate(values):
        if int(value) == int(target):
            return idx
    return None


def _kv_visibility_audit(bundle: dict[str, Any]) -> dict[str, Any]:
    """Audit metadata-visible KV length against freshly-written context slots.

    DFlash first pass pre-inserts context K/V with ``context_slot_mapping`` and
    then runs query tokens with attention metadata ``seq_lens``. If
    ``seq_lens - query_len`` is larger than the front-valid copied context
    slots, attention is relying on older cache-resident prefix slots in addition
    to the freshly written context rows. That can be correct, but it is the
    exact seam where stale rejected-tail visibility can hide.
    """
    seq_lens = _as_int_list(bundle.get("seq_lens"))
    query_positions = _as_int_list(bundle.get("query_positions"))
    context_positions = _as_int_list(bundle.get("context_positions"))
    context_slot_mapping = _as_int_list(bundle.get("context_slot_mapping"))
    compact_src_slots = _as_int_list(bundle.get("compact_src_slots"))
    compact_dst_slots = _as_int_list(bundle.get("compact_dst_slots"))

    out: dict[str, Any] = {
        "available": bool(seq_lens and query_positions),
        "has_context_slot_mapping": context_slot_mapping is not None,
        "has_compaction_slots": (
            compact_src_slots is not None and compact_dst_slots is not None
        ),
    }
    if not seq_lens or not query_positions:
        out["reason"] = "missing_seq_lens_or_query_positions"
        return out

    query_len = len(query_positions)
    seq_len0 = int(seq_lens[0])
    visible_context_len = seq_len0 - query_len
    out.update(
        {
            "seq_len0": seq_len0,
            "query_len": query_len,
            "visible_context_len": visible_context_len,
            "query_positions_head": query_positions[:16],
        }
    )

    if context_positions is not None:
        out.update(
            {
                "num_context_rows": len(context_positions),
                "context_positions_head": context_positions[:16],
            }
        )
    if context_slot_mapping is not None:
        first_pad_idx = _first_index(context_slot_mapping, -1)
        front_valid_len = (
            first_pad_idx
            if first_pad_idx is not None
            else len(context_slot_mapping)
        )
        visible_slots = context_slot_mapping[: max(0, visible_context_len)]
        num_pad_slots_in_visible = sum(1 for s in visible_slots if int(s) == -1)
        refreshed_valid_slots = [s for s in context_slot_mapping if int(s) >= 0]
        out.update(
            {
                "first_pad_idx": first_pad_idx,
                "front_valid_context_len": front_valid_len,
                "refreshed_valid_slot_count": len(refreshed_valid_slots),
                "visible_exceeds_front_valid_context": (
                    visible_context_len > front_valid_len
                ),
                "visible_exceeds_refreshed_valid_slots": (
                    visible_context_len > len(refreshed_valid_slots)
                ),
                "visible_tail_not_refreshed_count": max(
                    0, visible_context_len - front_valid_len
                ),
                "num_pad_slots_inside_visible_window": num_pad_slots_in_visible,
                "context_slot_mapping_head": context_slot_mapping[:16],
                "visible_slot_mapping_head": visible_slots[:16],
            }
        )

    if compact_src_slots is not None and compact_dst_slots is not None:
        out.update(
            {
                "compact_src_slots_count": len(compact_src_slots),
                "compact_dst_slots_count": len(compact_dst_slots),
                "compact_src_slots_head": compact_src_slots[:16],
                "compact_dst_slots_head": compact_dst_slots[:16],
            }
        )

    if (
        context_slot_mapping is not None
        and out.get("visible_exceeds_front_valid_context") is True
    ):
        out["interpretation_hint"] = (
            "The metadata-visible context is longer than the front-valid "
            "freshly-written context rows. Attention therefore depends on "
            "older cache-resident prefix slots beyond this iteration's copied "
            "context rows; stale rejected-tail visibility must be ruled out at "
            "the cache/block-table level."
        )
    elif context_slot_mapping is not None:
        out["interpretation_hint"] = (
            "The metadata-visible context is covered by the freshly-written "
            "front-valid context rows for this bundle."
        )
    else:
        out["interpretation_hint"] = (
            "No context_slot_mapping was captured, so the audit cannot compare "
            "metadata-visible length to freshly-written context slots."
        )
    return out


def _compare_kv_visibility(
    reference_audit: dict[str, Any],
    vllm_audit: dict[str, Any],
) -> dict[str, Any]:
    keys = [
        "visible_context_len",
        "front_valid_context_len",
        "visible_tail_not_refreshed_count",
        "num_pad_slots_inside_visible_window",
        "refreshed_valid_slot_count",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        out[f"{key}_match"] = reference_audit.get(key) == vllm_audit.get(key)
        out[f"reference_{key}"] = reference_audit.get(key)
        out[f"vllm_{key}"] = vllm_audit.get(key)
    out["vllm_visible_exceeds_front_valid_context"] = vllm_audit.get(
        "visible_exceeds_front_valid_context"
    )
    out["reference_visible_exceeds_front_valid_context"] = reference_audit.get(
        "visible_exceeds_front_valid_context"
    )
    return out


def compare_runtime_bundles(
    *,
    reference_bundle: str,
    vllm_bundle: str,
    output_json: str | None = None,
) -> dict[str, Any]:
    ref_path = _resolve_bundle(reference_bundle, DEFAULT_REFERENCE_NAME)
    vllm_path = _resolve_bundle(vllm_bundle, DEFAULT_VLLM_NAME)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference bundle not found: {ref_path}")
    if not vllm_path.exists():
        raise FileNotFoundError(f"vLLM bundle not found: {vllm_path}")

    ref = torch.load(ref_path, map_location="cpu")
    cand = torch.load(vllm_path, map_location="cpu")

    tensor_keys = [
        "prompt_token_ids",
        "target_token_ids",
        "target_positions",
        "next_token_ids",
        "raw_target_hidden_states",
        "combined_target_hidden_states",
        "context_positions",
        "context_slot_mapping",
        "query_input_ids",
        "query_positions",
        "token_indices_to_sample",
        "seq_lens",
        "sample_hidden_states_req0",
        "draft_logits_req0",
        "draft_logprobs_req0",
        "topk_tok_0",
        "topk_lp_0",
        "builder_topk_tok",
        "builder_topk_lp",
        "builder_per_depth_entropy",
        "builder_tree_node_token_ids",
        "builder_tree_parent_indices",
        "builder_tree_depths",
        "builder_tree_node_child_ranks",
        "builder_tree_node_cum_logprobs",
        "builder_tree_node_expand_scores",
        "builder_tree_node_added_order",
        "tree_node_token_ids",
        "tree_parent_indices",
        "tree_depths",
        "verify_greedy_tokens",
        "accepted_path",
        "correction_token",
        "accepted_tokens",
        "emitted_tokens",
        "compact_src_slots",
        "compact_dst_slots",
        "post_compact_hidden_states",
        "post_compact_target_hidden_states",
        "post_verify_target_hidden_states",
    ]

    summary: dict[str, Any] = {
        "reference_bundle": str(ref_path),
        "vllm_bundle": str(vllm_path),
        "scalar_fields": {},
        "tensor_comparisons": {},
    }

    scalar_keys = [
        "source",
        "prompt_text",
        "model",
        "draft_model",
        "step",
        "dflash_is_causal",
        "parallel_drafting_token_id",
        "block_size",
        "tree_width",
        "tree_budget",
        "tree_num_nodes",
        "builder_tree_budget",
        "builder_tree_num_nodes",
        "builder_tree_construction",
        "builder_score_mode",
        "builder_hybrid_alpha",
        "accepted_len",
        "seed",
    ]
    for key in scalar_keys:
        summary["scalar_fields"][key] = {
            "reference": ref.get(key),
            "vllm": cand.get(key),
            "match": ref.get(key) == cand.get(key),
        }

    for key in tensor_keys:
        ref_value = ref.get(key)
        cand_value = cand.get(key)
        if not isinstance(ref_value, torch.Tensor) or not isinstance(cand_value, torch.Tensor):
            summary["tensor_comparisons"][key] = {
                "missing": True,
                "reference_type": type(ref_value).__name__,
                "candidate_type": type(cand_value).__name__,
            }
            continue
        summary["tensor_comparisons"][key] = _compare_tensor(key, ref_value, cand_value)

    if isinstance(ref.get("topk_tok_0"), torch.Tensor) and isinstance(cand.get("topk_tok_0"), torch.Tensor):
        ref_topk = ref["topk_tok_0"].tolist()
        cand_topk = cand["topk_tok_0"].tolist()
        summary["topk_overlap"] = {
            "reference": ref_topk,
            "vllm": cand_topk,
            "intersection": sorted(set(ref_topk).intersection(cand_topk)),
        }

    ref_kv_audit = _kv_visibility_audit(ref)
    vllm_kv_audit = _kv_visibility_audit(cand)
    summary["kv_visibility_audit"] = {
        "reference": ref_kv_audit,
        "vllm": vllm_kv_audit,
        "comparison": _compare_kv_visibility(ref_kv_audit, vllm_kv_audit),
    }

    if output_json is not None:
        output_path = Path(output_json)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare reference and vLLM DFlash runtime bundles.")
    parser.add_argument("--reference-bundle", required=True)
    parser.add_argument("--vllm-bundle", required=True)
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = compare_runtime_bundles(
        reference_bundle=args.reference_bundle,
        vllm_bundle=args.vllm_bundle,
        output_json=args.output_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
