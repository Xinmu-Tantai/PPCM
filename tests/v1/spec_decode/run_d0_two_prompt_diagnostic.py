from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


_DEFAULT_DEBUG_DIR = Path("/tmp") / f"debug_d0_{datetime.now():%Y%m%d_%H%M%S}"
_VLLM_ROOT = Path("/home/i-hulanxiang/workspace/vllm-parallel-drafting")
_DFLASH_ROOT = Path("/home/i-hulanxiang/workspace/dflash")
_REFERENCE_CAPTURE = (
    _DFLASH_ROOT / "tests/drafting_mechanism/test_runtime_step0_bundle.py"
)
_VLLM_CAPTURE = (
    _VLLM_ROOT / "tests/v1/spec_decode/test_dflash_runtime_bundle.py"
)
_COMPARE_SCRIPT = (
    _VLLM_ROOT / "tests/v1/spec_decode/compare_dflash_runtime_bundle.py"
)

DEFAULT_TARGET_MODEL = "/data/models/Qwen3-8B"
DEFAULT_DRAFT_MODEL = (
    "/mnt/specdec-dev/checkpoints/specforge/outputs/"
    "nemotron-780k-and-codealpaca20k-v2-causal-distill-lr1e-4-anchorcnt512/"
    "epoch_6_step_583488"
)


def _parse_sample_indices(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",")]
    indices = [int(v) for v in values if v]
    if not indices:
        raise ValueError("At least one sample index must be provided.")
    return indices


def _run(cmd: list[str], *, env: dict[str, str], cwd: Path) -> None:
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def _tensor_to_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value)


def _root_children(bundle: dict[str, Any]) -> list[int]:
    token_ids = bundle.get("tree_node_token_ids")
    depths = bundle.get("tree_depths")
    if not isinstance(token_ids, torch.Tensor) or not isinstance(depths, torch.Tensor):
        return []
    mask = depths == 1
    return token_ids[mask].detach().cpu().tolist()


def _depth0_target_token(bundle: dict[str, Any]) -> int | None:
    greedy = bundle.get("verify_greedy_tokens")
    if not isinstance(greedy, torch.Tensor) or greedy.numel() == 0:
        return None
    return int(greedy[0].item())


def _first_diverging_builder_row(
    reference_bundle: dict[str, Any],
    vllm_bundle: dict[str, Any],
) -> dict[str, Any] | None:
    ref = reference_bundle.get("builder_topk_tok")
    cand = vllm_bundle.get("builder_topk_tok")
    if not isinstance(ref, torch.Tensor) or not isinstance(cand, torch.Tensor):
        return None
    if ref.shape != cand.shape or ref.ndim != 2:
        return {
            "shape_mismatch": True,
            "reference_shape": list(ref.shape),
            "vllm_shape": list(cand.shape),
        }
    for row_idx in range(ref.shape[0]):
        ref_row = ref[row_idx]
        cand_row = cand[row_idx]
        unequal = int((ref_row != cand_row).sum().item())
        if unequal > 0:
            return {
                "row_index": row_idx,
                "reference": ref_row.tolist(),
                "vllm": cand_row.tolist(),
                "num_unequal": unequal,
            }
    return None


def _make_sample_summary(
    *,
    sample_index: int,
    reference_bundle: dict[str, Any],
    vllm_bundle: dict[str, Any],
    compare_summary: dict[str, Any],
    sample_dir: Path,
) -> dict[str, Any]:
    ref_target = _depth0_target_token(reference_bundle)
    vllm_target = _depth0_target_token(vllm_bundle)
    ref_children = _root_children(reference_bundle)
    vllm_children = _root_children(vllm_bundle)
    first_builder_row = _first_diverging_builder_row(reference_bundle, vllm_bundle)

    tensor_cmp = compare_summary.get("tensor_comparisons", {})
    scalar_cmp = compare_summary.get("scalar_fields", {})
    topk_lp_cmp = tensor_cmp.get("topk_lp_0", {})
    builder_topk_tok_cmp = tensor_cmp.get("builder_topk_tok", {})
    kv_audit = compare_summary.get("kv_visibility_audit", {})
    vllm_kv_audit = kv_audit.get("vllm", {})
    kv_audit_cmp = kv_audit.get("comparison", {})

    return {
        "sample_index": sample_index,
        "sample_dir": str(sample_dir),
        "prompt_preview": str(reference_bundle.get("prompt_text", ""))[:200],
        "reference": {
            "accepted_len": int(reference_bundle.get("accepted_len", 0)),
            "d0_accepted": int(reference_bundle.get("accepted_len", 0)) >= 1,
            "depth0_target_token": ref_target,
            "depth0_target_in_root_children": (
                ref_target in ref_children if ref_target is not None else None
            ),
            "root_children": ref_children,
        },
        "vllm": {
            "accepted_len": int(vllm_bundle.get("accepted_len", 0)),
            "d0_accepted": int(vllm_bundle.get("accepted_len", 0)) >= 1,
            "depth0_target_token": vllm_target,
            "depth0_target_in_root_children": (
                vllm_target in vllm_children if vllm_target is not None else None
            ),
            "root_children": vllm_children,
        },
        "compare": {
            "accepted_len_match": scalar_cmp.get("accepted_len", {}).get("match"),
            "topk_tok_0_exact_match": tensor_cmp.get("topk_tok_0", {}).get(
                "exact_match"
            ),
            "topk_lp_0_max_abs_diff": topk_lp_cmp.get("max_abs_diff"),
            "topk_lp_0_mean_abs_diff": topk_lp_cmp.get("mean_abs_diff"),
            "builder_topk_tok_num_unequal": builder_topk_tok_cmp.get("num_unequal"),
            "first_diverging_builder_row": first_builder_row,
            "builder_tree_node_token_ids_num_unequal": tensor_cmp.get(
                "builder_tree_node_token_ids", {}
            ).get("num_unequal"),
            "builder_tree_parent_indices_num_unequal": tensor_cmp.get(
                "builder_tree_parent_indices", {}
            ).get("num_unequal"),
            "builder_tree_depths_num_unequal": tensor_cmp.get(
                "builder_tree_depths", {}
            ).get("num_unequal"),
            "kv_visible_context_len": vllm_kv_audit.get("visible_context_len"),
            "kv_front_valid_context_len": vllm_kv_audit.get(
                "front_valid_context_len"
            ),
            "kv_visible_tail_not_refreshed_count": vllm_kv_audit.get(
                "visible_tail_not_refreshed_count"
            ),
            "kv_visible_exceeds_front_valid_context": vllm_kv_audit.get(
                "visible_exceeds_front_valid_context"
            ),
            "kv_num_pad_slots_inside_visible_window": vllm_kv_audit.get(
                "num_pad_slots_inside_visible_window"
            ),
            "kv_visible_tail_not_refreshed_count_match": kv_audit_cmp.get(
                "visible_tail_not_refreshed_count_match"
            ),
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_bundle(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_diagnostic(
    *,
    debug_dir: Path,
    python_bin: str,
    model: str,
    draft_model: str,
    sample_indices: list[int],
    tree_attn_kernel: str,
    max_draft_passes: int,
) -> dict[str, Any]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)

    per_sample: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        sample_dir = debug_dir / f"sample_{sample_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        ref_cmd = [
            python_bin,
            str(_REFERENCE_CAPTURE),
            "--debug-dir",
            str(sample_dir),
            "--model",
            model,
            "--draft-model",
            draft_model,
            "--attn-implementation",
            "sdpa",
            "--tree-attn",
            "sdpa",
            "--prompt-set",
            "humaneval",
            "--sample-index",
            str(sample_index),
        ]
        _run(ref_cmd, env=env, cwd=_DFLASH_ROOT)

        vllm_cmd = [
            python_bin,
            str(_VLLM_CAPTURE),
            "--debug-dir",
            str(sample_dir),
            "--model",
            model,
            "--draft-model",
            draft_model,
            "--enforce-eager",
            "--tree-attn-kernel",
            tree_attn_kernel,
            "--max-draft-passes",
            str(max_draft_passes),
            "--prompt-set",
            "humaneval",
            "--sample-index",
            str(sample_index),
        ]
        _run(vllm_cmd, env=env, cwd=_VLLM_ROOT)

        compare_cmd = [
            python_bin,
            str(_COMPARE_SCRIPT),
            "--reference-bundle",
            str(sample_dir),
            "--vllm-bundle",
            str(sample_dir),
            "--output-json",
            str(sample_dir / "runtime_compare.json"),
        ]
        _run(compare_cmd, env=env, cwd=_VLLM_ROOT)

        reference_bundle = _load_bundle(sample_dir / "reference_step0_runtime_bundle.pt")
        vllm_bundle = _load_bundle(sample_dir / "vllm_step0_runtime_bundle.pt")
        compare_summary = _load_json(sample_dir / "runtime_compare.json")
        sample_summary = _make_sample_summary(
            sample_index=sample_index,
            reference_bundle=reference_bundle,
            vllm_bundle=vllm_bundle,
            compare_summary=compare_summary,
            sample_dir=sample_dir,
        )
        _write_json(sample_dir / "d0_summary.json", sample_summary)
        per_sample.append(sample_summary)

    aggregate = {
        "debug_dir": str(debug_dir),
        "sample_indices": sample_indices,
        "num_samples": len(per_sample),
        "reference_d0_acceptance_rate": (
            sum(1 for s in per_sample if s["reference"]["d0_accepted"]) / len(per_sample)
        ),
        "vllm_d0_acceptance_rate": (
            sum(1 for s in per_sample if s["vllm"]["d0_accepted"]) / len(per_sample)
        ),
        "samples": per_sample,
    }
    _write_json(debug_dir / "d0_acceptance_summary.json", aggregate)
    return aggregate


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a focused 2-prompt DFlash depth-0 diagnostic.",
    )
    parser.add_argument("--debug-dir", default=str(_DEFAULT_DEBUG_DIR))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--sample-indices", default="0,1")
    parser.add_argument("--tree-attn-kernel", default="optimus")
    parser.add_argument("--max-draft-passes", type=int, default=0)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = run_diagnostic(
        debug_dir=Path(args.debug_dir),
        python_bin=args.python_bin,
        model=args.model,
        draft_model=args.draft_model,
        sample_indices=_parse_sample_indices(args.sample_indices),
        tree_attn_kernel=args.tree_attn_kernel,
        max_draft_passes=args.max_draft_passes,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
