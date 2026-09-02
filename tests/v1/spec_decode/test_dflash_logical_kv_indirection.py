# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import torch
from datasets import load_dataset

from vllm import LLM, SamplingParams


DEFAULT_TARGET_MODEL = "/root/models/Qwen3-8B"
DEFAULT_DRAFT_MODEL = (
    "/root/data/outputs/dflash-qwen3-8b-causal-bs16-anc1-forwardkl-lr3e-4-gNone/"
    "epoch_6_step_291744_forward_kl"
)
DEFAULT_DEBUG_DIR = (
    Path("/root/data/vllm-ptd")
    / f"dflash_logical_kv_indirection_debug_{datetime.now():%m%d_%H%M%S}"
)


def _jsonify(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_humaneval_prompt(sample_index: int) -> str:
    dataset = load_dataset("openai/openai_humaneval", split="test")
    row = dataset[sample_index]
    return (
        "Write a solution to the following problem and make sure that it "
        f"passes the tests:\n```python\n{row['prompt']}\n```"
    )


def _maybe_apply_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _first_mismatch(values_a: list[Any], values_b: list[Any]) -> int | None:
    for idx, (a, b) in enumerate(zip(values_a, values_b)):
        if a != b:
            return idx
    if len(values_a) != len(values_b):
        return min(len(values_a), len(values_b))
    return None


def _head(value: Any, n: int = 16) -> Any:
    if isinstance(value, list):
        return value[:n]
    return value


def _slot_address_audit(
    physical_step0: dict[str, Any],
    logical_step0: dict[str, Any],
    logical_step1: dict[str, Any],
    *,
    block_size: int,
) -> dict[str, Any]:
    physical_src = physical_step0.get("compact_src_slots")
    physical_dst = physical_step0.get("compact_dst_slots")
    logical_slots = logical_step1.get("logical_kv_slots")
    logical_start = logical_step1.get("logical_kv_start")

    audit: dict[str, Any] = {
        "block_size": block_size,
        "physical_step0_compact_src_slots": physical_src,
        "physical_step0_compact_dst_slots": physical_dst,
        "logical_step0_commit_slots": logical_step0.get("logical_commit_slots"),
        "logical_step1_logical_kv_start": logical_start,
        "logical_step1_logical_kv_slots": logical_slots,
        "logical_slots_match_physical_src": logical_slots == physical_src,
        "logical_slots_match_physical_dst": logical_slots == physical_dst,
    }

    if isinstance(logical_start, int) and isinstance(logical_slots, list):
        rows = []
        for idx, slot in enumerate(logical_slots):
            logical_pos = logical_start + idx
            rows.append(
                {
                    "logical_pos": logical_pos,
                    "logical_block": logical_pos // block_size,
                    "logical_offset": logical_pos % block_size,
                    "physical_slot": slot,
                    "physical_block": int(slot) // block_size,
                    "physical_offset": int(slot) % block_size,
                    "noncanonical": (
                        physical_dst is not None
                        and idx < len(physical_dst)
                        and int(slot) != int(physical_dst[idx])
                    ),
                }
            )
        audit["logical_address_rows"] = rows
        audit["has_noncanonical_logical_slots"] = any(
            bool(row["noncanonical"]) for row in rows
        )
    return audit


def _compare_runs(
    physical: dict[str, Any],
    logical: dict[str, Any],
    *,
    block_size: int,
) -> dict[str, Any]:
    physical_bundles = physical["verify_bundles"]
    logical_bundles = logical["verify_bundles"]
    physical_topk = physical["topk_log"]
    logical_topk = logical["topk_log"]

    max_steps = min(len(physical_bundles), len(logical_bundles))
    step_reports = []
    first_verify_mismatch = None
    for step in range(max_steps):
        p = physical_bundles[step]
        l = logical_bundles[step]
        fields = [
            "tree_node_token_ids",
            "tree_parent_indices",
            "tree_depths",
            "query_positions",
            "verify_greedy_tokens",
            "accepted_path",
            "accepted_len",
            "correction_token",
        ]
        field_report = {}
        for field in fields:
            same = p.get(field) == l.get(field)
            field_report[field] = {
                "same": same,
                "physical_head": _head(p.get(field)),
                "logical_head": _head(l.get(field)),
            }
            if (
                first_verify_mismatch is None
                and field == "verify_greedy_tokens"
                and not same
            ):
                first_verify_mismatch = step

        step_reports.append(
            {
                "step": step,
                "fields": field_report,
                "physical_slots": {
                    "compact_src_slots": p.get("compact_src_slots"),
                    "compact_dst_slots": p.get("compact_dst_slots"),
                },
                "logical_slots": {
                    "logical_kv_start": l.get("logical_kv_start"),
                    "logical_kv_slots": l.get("logical_kv_slots"),
                    "logical_commit_accepted_slots": l.get(
                        "logical_commit_accepted_slots"
                    ),
                    "logical_commit_slots": l.get("logical_commit_slots"),
                },
            }
        )

    topk_first_mismatch = None
    for idx, (p, l) in enumerate(zip(physical_topk, logical_topk)):
        if p.get("verify_greedy_tokens") != l.get("verify_greedy_tokens"):
            topk_first_mismatch = idx
            break

    audit = {}
    if len(physical_bundles) > 1 and len(logical_bundles) > 1:
        audit = _slot_address_audit(
            physical_bundles[0],
            logical_bundles[0],
            logical_bundles[1],
            block_size=block_size,
        )

    return {
        "physical_num_bundles": len(physical_bundles),
        "logical_num_bundles": len(logical_bundles),
        "physical_num_topk_entries": len(physical_topk),
        "logical_num_topk_entries": len(logical_topk),
        "first_verify_bundle_mismatch_step": first_verify_mismatch,
        "first_topk_verify_mismatch_index": topk_first_mismatch,
        "slot_address_audit": audit,
        "step_reports": step_reports,
    }


def capture_layout_run(
    *,
    layout: str,
    model: str,
    draft_model: str,
    prompt_text: str,
    output_dir: Path,
    max_model_len: int,
    max_tokens: int,
    block_size: int,
    tree_width: int,
    max_tree_budget: int,
    seed: int,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    llm = LLM(
        model=model,
        trust_remote_code=True,
        seed=seed,
        max_model_len=max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=51200,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=True,
        enforce_eager=True,
        attention_backend="FLASH_ATTN",
        speculative_config={
            "method": "dflash",
            "model": draft_model,
            "num_speculative_tokens": block_size - 1,
            "max_model_len": max_model_len,
            "head_type": "causal",
            "tree_width": tree_width,
            "max_tree_budget": max_tree_budget,
            "tree_draft": "accum_logp",
            "max_draft_passes": 0,
            "tree_prune_ratio": 0.25,
            "tree_construction": "breadth_first",
            "tree_attn_kernel": "triton",
            "tree_kv_layout": layout,
            "num_cudagraph_tree_captures": 0,
        },
    )
    try:
        tokenizer = llm.get_tokenizer()
        formatted_prompt = _maybe_apply_chat_template(tokenizer, prompt_text)
        worker = llm.llm_engine.model_executor.driver_worker.worker
        model_runner = worker.model_runner
        drafter = model_runner.drafter

        if hasattr(drafter, "clear_topk_log"):
            drafter.clear_topk_log()
        if hasattr(drafter, "clear_runtime_bundles"):
            drafter.clear_runtime_bundles()
        model_runner._dflash_runtime_verify_bundles = []

        outputs = llm.generate(
            [formatted_prompt],
            SamplingParams(temperature=0.0, max_tokens=max_tokens),
            use_tqdm=False,
        )
        topk_log = drafter.get_topk_log() if hasattr(drafter, "get_topk_log") else []
        verify_bundles = getattr(model_runner, "_dflash_runtime_verify_bundles", [])
        payload = {
            "layout": layout,
            "generated_token_ids": outputs[0].outputs[0].token_ids,
            "generated_text": outputs[0].outputs[0].text,
            "topk_log": _jsonify(topk_log),
            "verify_bundles": _jsonify(verify_bundles),
        }
        (output_dir / f"{layout}_run.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        return payload
    finally:
        llm.llm_engine.engine_core.shutdown()


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = args.prompt or _load_humaneval_prompt(args.sample_index)

    physical = capture_layout_run(
        layout="physical",
        model=args.model,
        draft_model=args.draft_model,
        prompt_text=prompt_text,
        output_dir=debug_dir,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        block_size=args.block_size,
        tree_width=args.tree_width,
        max_tree_budget=args.max_tree_budget,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    logical = capture_layout_run(
        layout="logical",
        model=args.model,
        draft_model=args.draft_model,
        prompt_text=prompt_text,
        output_dir=debug_dir,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        block_size=args.block_size,
        tree_width=args.tree_width,
        max_tree_budget=args.max_tree_budget,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    report = _compare_runs(physical, logical, block_size=args.block_size)
    report.update(
        {
            "debug_dir": str(debug_dir),
            "model": args.model,
            "draft_model": args.draft_model,
            "sample_index": args.sample_index,
            "max_tokens": args.max_tokens,
            "block_size": args.block_size,
            "tree_width": args.tree_width,
            "max_tree_budget": args.max_tree_budget,
        }
    )
    (debug_dir / "logical_kv_indirection_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
def test_logical_kv_indirection_real_model() -> None:
    if os.environ.get("DFLASH_RUN_LOGICAL_KV_REAL_TEST") != "1":
        pytest.skip("Set DFLASH_RUN_LOGICAL_KV_REAL_TEST=1 to run this test.")
    args = _build_arg_parser().parse_args([])
    args.model = os.environ.get("DFLASH_TARGET_MODEL", args.model)
    args.draft_model = os.environ.get("DFLASH_DRAFT_MODEL", args.draft_model)
    args.debug_dir = os.environ.get("DFLASH_TEST_DEBUG_DIR", args.debug_dir)
    report = run_diagnostic(args)
    assert report["slot_address_audit"].get("has_noncanonical_logical_slots") is True


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run physical vs logical DFlash Triton real-model diagnostic and "
            "audit logical KV slot indirection at the first verification mismatch."
        )
    )
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--debug-dir", default=str(DEFAULT_DEBUG_DIR))
    parser.add_argument("--prompt")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tree-width", type=int, default=7)
    parser.add_argument("--max-tree-budget", type=int, default=127)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = run_diagnostic(args)
    print(json.dumps(report, indent=2))
    print(f"Wrote diagnostic artifacts to: {report['debug_dir']}")


if __name__ == "__main__":
    main()
