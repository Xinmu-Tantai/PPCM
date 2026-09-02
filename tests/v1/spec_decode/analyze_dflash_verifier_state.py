#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze enhanced DFlash verifier debug bundles.

The companion runtime capture stores verifier slot mappings and small KV-cache
snapshots in ``dflash_runtime_verify_bundles.json``.  This script explains the
state for a selected node, or optionally runs the target-only AR parity probe to
find the first mismatching node and then explains that one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tests.v1.spec_decode.probe_step3p7_tree_target_parity import (  # noqa: E402
    _build_llm,
    _iter_probe_items,
    _load_prompt_token_ids,
    _run_ar_next_tokens,
)
from vllm.v1.spec_decode.dflash_tree import (  # noqa: E402
    _build_attention_bias_np,
    build_ancestor_matrix_np,
)


def _path_to_root(parents: list[int], node_idx: int) -> list[int]:
    path = []
    cur = node_idx
    while cur >= 0:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


def _safe_index(values: Any, idx: int) -> Any:
    if not isinstance(values, list) or idx >= len(values):
        return None
    return values[idx]


def _find_probe_slot(bundle: dict[str, Any], node_idx: int) -> dict[str, Any]:
    probe_nodes = bundle.get("verifier_probe_node_indices")
    if not isinstance(probe_nodes, list) or node_idx not in probe_nodes:
        return {"captured": False}
    probe_i = probe_nodes.index(node_idx)
    result: dict[str, Any] = {
        "captured": True,
        "probe_index": probe_i,
        "slot": _safe_index(bundle.get("verifier_probe_slots"), probe_i),
        "position": _safe_index(bundle.get("verifier_probe_positions"), probe_i),
        "model_position_kind": bundle.get("verifier_probe_model_position_kind"),
        "model_positions": (
            [
                row[probe_i]
                for row in bundle.get("verifier_probe_model_positions", [])
            ]
            if isinstance(bundle.get("verifier_probe_model_positions"), list)
            else None
        ),
        "depth": _safe_index(bundle.get("verifier_probe_depths"), probe_i),
        "parent": _safe_index(bundle.get("verifier_probe_parent_indices"), probe_i),
    }
    kv = bundle.get("verifier_probe_kv")
    if isinstance(kv, dict):
        result["kv_layers"] = {}
        for layer, payload in kv.items():
            if not isinstance(payload, dict):
                continue
            result["kv_layers"][layer] = {
                "key_norm": _safe_index(payload.get("key_norm"), probe_i),
                "value_norm": _safe_index(payload.get("value_norm"), probe_i),
                "key_shape": (
                    list(np.array(payload["key"]).shape)
                    if "key" in payload else None
                ),
                "value_shape": (
                    list(np.array(payload["value"]).shape)
                    if "value" in payload else None
                ),
            }
    return result


def _find_forward_probe(bundle: dict[str, Any], node_idx: int) -> dict[str, Any]:
    probe_nodes = bundle.get("verifier_forward_probe_node_indices")
    if not isinstance(probe_nodes, list) or node_idx not in probe_nodes:
        return {"captured": False}
    probe_i = probe_nodes.index(node_idx)
    return {
        "captured": True,
        "probe_index": probe_i,
        "token_id": _safe_index(
            bundle.get("verifier_forward_probe_token_ids"), probe_i
        ),
        "greedy_token_id": _safe_index(
            bundle.get("verifier_forward_probe_greedy_token_ids"), probe_i
        ),
        "hidden_norm": _safe_index(
            bundle.get("verifier_forward_probe_hidden_norm"), probe_i
        ),
        "logits_norm": _safe_index(
            bundle.get("verifier_forward_probe_logits_norm"), probe_i
        ),
        "topk_token_ids": _safe_index(
            bundle.get("verifier_forward_probe_topk_token_ids"), probe_i
        ),
        "topk_logits": _safe_index(
            bundle.get("verifier_forward_probe_topk_logits"), probe_i
        ),
        "candidate_token_ids": bundle.get(
            "verifier_forward_probe_candidate_token_ids"
        ),
        "candidate_logits": _safe_index(
            bundle.get("verifier_forward_probe_candidate_logits"), probe_i
        ),
    }


def _find_attention_probe(bundle: dict[str, Any], node_idx: int) -> list[dict[str, Any]]:
    records = bundle.get("verifier_attention_probe")
    if not isinstance(records, list):
        return []
    return [
        record
        for record in records
        if isinstance(record, dict) and record.get("node_index") == node_idx
    ]


def _analyze_node(
    bundle: dict[str, Any],
    *,
    node_idx: int,
    actual_token: int | None = None,
) -> dict[str, Any]:
    parents = [int(x) for x in bundle["tree_parent_indices"]]
    depths = [int(x) for x in bundle["tree_depths"]]
    tokens = [int(x) for x in bundle["tree_node_token_ids"]]
    greedy = [int(x) for x in bundle["verify_greedy_tokens"]]
    path = _path_to_root(parents, node_idx)
    neg_inf = float(torch.finfo(torch.float32).min)
    bias = _build_attention_bias_np(parents, neg_inf)
    ancestor = build_ancestor_matrix_np(parents)
    allowed_cols = np.where(bias[node_idx] == 0.0)[0].tolist()
    ancestor_cols = np.where(ancestor[node_idx] == 1)[0].tolist()

    slot_mapping = bundle.get("slot_mapping")
    slot = _safe_index(slot_mapping, node_idx)
    query_positions = bundle.get("query_positions")
    position = _safe_index(query_positions, node_idx)
    seq_lens = bundle.get("verifier_seq_lens")
    req_idx = int(bundle.get("req_idx", 0))
    seq_len = _safe_index(seq_lens, req_idx)
    qlen = int(bundle.get("tree_num_nodes", len(tokens)))
    context_len = int(seq_len) - qlen if seq_len is not None else None
    expected_position = (
        context_len + depths[node_idx] if context_len is not None else None
    )

    masked_prior_siblings = [
        idx for idx in range(node_idx)
        if idx not in allowed_cols and parents[idx] == parents[node_idx]
    ]
    return {
        "verify_step_index": bundle.get("verify_step_index"),
        "req_id": bundle.get("req_id"),
        "req_idx": req_idx,
        "node_idx": node_idx,
        "token": tokens[node_idx],
        "parent": parents[node_idx],
        "depth": depths[node_idx],
        "path": path,
        "path_token_ids": [tokens[idx] for idx in path],
        "verify_expected_token": greedy[node_idx],
        "ar_actual_token": actual_token,
        "allowed_query_cols": allowed_cols,
        "ancestor_cols": ancestor_cols,
        "masked_prior_siblings": masked_prior_siblings,
        "slot": slot,
        "position": position,
        "context_len": context_len,
        "expected_depth_position": expected_position,
        "position_matches_depth": (
            position == expected_position if expected_position is not None else None
        ),
        "query_start": bundle.get("query_start"),
        "query_end": bundle.get("query_end"),
        "tree_num_nodes": qlen,
        "compact_src_slots": bundle.get("compact_src_slots"),
        "compact_dst_slots": bundle.get("compact_dst_slots"),
        "probe": _find_probe_slot(bundle, node_idx),
        "forward_probe": _find_forward_probe(bundle, node_idx),
        "attention_probe": _find_attention_probe(bundle, node_idx),
        "context_probe": bundle.get("verifier_probe_context_kv"),
        "context_probe_positions": bundle.get("verifier_probe_context_positions"),
        "context_probe_slots": bundle.get("verifier_probe_context_slots"),
    }


def _find_first_mismatch(args: argparse.Namespace, bundles: list[dict[str, Any]]):
    _, prompt_token_ids = _load_prompt_token_ids(
        model=args.model,
        prompt_set=args.prompt_set,
        sample_start=args.sample_start,
        num_samples=args.num_samples,
    )
    items = _iter_probe_items(
        bundles,
        prompt_token_ids,
        max_steps=args.max_steps,
        max_nodes_per_step=args.max_nodes_per_step,
    )
    llm = _build_llm(args)
    try:
        actual_tokens, _ = _run_ar_next_tokens(
            llm,
            items,
            batch_size=args.batch_size,
        )
    finally:
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
    for item, actual_token in zip(items, actual_tokens):
        if actual_token != item.expected_token:
            return item, actual_token
    return None, None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze enhanced DFlash verifier-state debug bundles.",
    )
    parser.add_argument("--verify-bundles", required=True)
    parser.add_argument("--bundle-index", type=int, default=0)
    parser.add_argument("--node-index", type=int, default=6)
    parser.add_argument("--run-ar", action="store_true")
    parser.add_argument("--output-json")

    # Arguments used only with --run-ar.  They mirror the parity probe.
    parser.add_argument("--model", default="/mnt/lanxiangh/models/Step-3.7-Flash")
    parser.add_argument("--prompt-set", default="humaneval")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-nodes-per-step", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument(
        "--disable-cascade-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    bundles_path = Path(args.verify_bundles)
    bundles = json.loads(bundles_path.read_text(encoding="utf-8"))
    bundle_index = args.bundle_index
    node_idx = args.node_index
    actual_token = None
    if args.run_ar:
        item, actual_token = _find_first_mismatch(args, bundles)
        if item is None:
            report = {
                "passed": True,
                "verify_bundles": str(bundles_path),
                "message": "No mismatch found in selected AR replay window.",
            }
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        bundle_index = next(
            (
                idx for idx, bundle in enumerate(bundles)
                if bundle.get("req_id") == item.req_id
                and int(bundle.get("verify_step_index", -1))
                == item.verify_step_index
            ),
            args.bundle_index,
        )
        node_idx = item.node_idx

    if bundle_index >= len(bundles):
        raise IndexError(f"bundle index {bundle_index} out of range ({len(bundles)})")
    report = {
        "verify_bundles": str(bundles_path),
        "bundle_index": bundle_index,
        "analysis": _analyze_node(
            bundles[bundle_index],
            node_idx=node_idx,
            actual_token=actual_token,
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
