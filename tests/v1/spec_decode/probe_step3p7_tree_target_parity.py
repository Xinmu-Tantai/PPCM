#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay DFlash tree-verifier nodes through target-only AR.

This is a verifier parity probe, not an output-quality benchmark.  It consumes
``dflash_runtime_verify_bundles.json`` from a DFlash debug run, reconstructs the
AR prefix corresponding to each tree node, and asks the target model for one
greedy token from that prefix.  The result is compared against the verifier's
recorded ``verify_greedy_tokens[node_idx]``.

If these top-1 tokens disagree for a node, tree verification logits are not
matching target-only AR for the same token history.  That points at tree
attention / positions / KV state, rather than prompt formatting or drafter
quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class ProbeItem:
    item_index: int
    req_id: str
    prompt_index: int
    verify_step_index: int
    node_idx: int
    depth: int
    expected_token: int
    prefix_token_ids: list[int]
    path: list[int]
    path_token_ids: list[int]


def _is_step3p7_model_path(model_path: str) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False
    try:
        with config_path.open(encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("model_type") == "step3p7"


def _tokenizer_load_kwargs(model_path: str) -> dict[str, Any]:
    if _is_step3p7_model_path(model_path):
        return {"fix_mistral_regex": True}
    return {}


def _apply_step3p7_regen_template(tokenizer: Any, prompt: str) -> str:
    templated_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    think_prefix = "<think>\n"
    if not templated_prompt.endswith(think_prefix):
        raise ValueError(
            "Unexpected Step-3.7 chat template: generation prompt does not end "
            f"with {think_prefix!r}"
        )
    return templated_prompt + "\n</think>\n\n"


def _load_prompt_bank(prompt_set: str) -> list[str]:
    if prompt_set != "humaneval":
        raise ValueError(f"Unsupported prompt set: {prompt_set!r}")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    return [
        "Write a solution to the following problem and make sure that it "
        f"passes the tests:\n```python\n{row['prompt']}\n```"
        for row in dataset
    ]


def _load_prompt_token_ids(
    *,
    model: str,
    prompt_set: str,
    sample_start: int,
    num_samples: int,
) -> tuple[Any, list[list[int]]]:
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        **_tokenizer_load_kwargs(model),
    )
    prompts = _load_prompt_bank(prompt_set)[sample_start:sample_start + num_samples]
    prompt_token_ids = []
    for prompt in prompts:
        if _is_step3p7_model_path(model):
            prompt_text = _apply_step3p7_regen_template(tokenizer, prompt)
        else:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt_token_ids.append(
            tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )
    return tokenizer, prompt_token_ids


def _path_to_root(parent_indices: list[int], node_idx: int) -> list[int]:
    path = []
    cur = node_idx
    while cur >= 0:
        path.append(cur)
        cur = parent_indices[cur]
    path.reverse()
    return path


def _parse_int_csv(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "":
        return None
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if part:
            result.add(int(part))
    return result


def _iter_probe_items(
    bundles: list[dict[str, Any]],
    prompt_token_ids_by_prompt: list[list[int]],
    *,
    max_steps: int | None,
    max_nodes_per_step: int | None,
) -> list[ProbeItem]:
    committed_by_req: dict[str, list[int]] = {}
    prompt_index_by_req: dict[str, int] = {}
    seen_steps: set[tuple[str, int]] = set()
    items: list[ProbeItem] = []
    next_prompt_index = 0

    for bundle in bundles:
        req_id = str(bundle["req_id"])
        if req_id not in prompt_index_by_req:
            if next_prompt_index >= len(prompt_token_ids_by_prompt):
                raise ValueError(
                    "More request IDs in verify bundles than prompt token lists. "
                    "Increase --num-samples or adjust --sample-start."
                )
            prompt_index_by_req[req_id] = next_prompt_index
            committed_by_req[req_id] = []
            next_prompt_index += 1

        verify_step_index = int(bundle["verify_step_index"])
        step_key = (req_id, verify_step_index)
        if step_key in seen_steps:
            # TP workers each dump the same verifier metadata.  Replaying every
            # copy would incorrectly commit the same accepted tokens repeatedly.
            continue
        seen_steps.add(step_key)

        if max_steps is not None and verify_step_index >= max_steps:
            committed_by_req[req_id].extend(int(t) for t in bundle["emitted_tokens"])
            continue

        prompt_index = prompt_index_by_req[req_id]
        base_prefix = (
            prompt_token_ids_by_prompt[prompt_index] + committed_by_req[req_id]
        )
        tree_tokens = [int(t) for t in bundle["tree_node_token_ids"]]
        parent_indices = [int(p) for p in bundle["tree_parent_indices"]]
        depths = [int(d) for d in bundle["tree_depths"]]
        greedy = [int(t) for t in bundle["verify_greedy_tokens"]]

        node_count = len(tree_tokens)
        if max_nodes_per_step is not None:
            node_count = min(node_count, max_nodes_per_step)
        for node_idx in range(node_count):
            path = _path_to_root(parent_indices, node_idx)
            path_tokens = [tree_tokens[i] for i in path]
            items.append(
                ProbeItem(
                    item_index=len(items),
                    req_id=req_id,
                    prompt_index=prompt_index,
                    verify_step_index=verify_step_index,
                    node_idx=node_idx,
                    depth=depths[node_idx],
                    expected_token=greedy[node_idx],
                    prefix_token_ids=base_prefix + path_tokens,
                    path=path,
                    path_token_ids=path_tokens,
                )
            )

        committed_by_req[req_id].extend(int(t) for t in bundle["emitted_tokens"])

    return items


def _build_llm(args: argparse.Namespace) -> Any:
    from vllm import LLM

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tp_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": True,
        "language_model_only": True,
        "limit_mm_per_prompt": {"image": 0},
        "skip_mm_profiling": True,
        "mm_processor_cache_gb": 0,
    }
    if args.enable_expert_parallel:
        llm_kwargs["enable_expert_parallel"] = True
    if args.disable_cascade_attn:
        llm_kwargs["disable_cascade_attn"] = True
    return LLM(**llm_kwargs)


def _prompt_from_token_ids(token_ids: list[int]) -> dict[str, list[int]]:
    # vLLM accepts this prompt form and bypasses tokenizer re-encoding.
    return {"prompt_token_ids": token_ids}


def _run_ar_next_tokens(
    llm: Any,
    items: list[ProbeItem],
    *,
    batch_size: int,
) -> tuple[list[int | None], list[list[dict[str, Any]] | None]]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    actual: list[int | None] = []
    logprobs: list[list[dict[str, Any]] | None] = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = [_prompt_from_token_ids(item.prefix_token_ids) for item in batch]
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        for output in outputs:
            token_ids = list(output.outputs[0].token_ids)
            actual.append(int(token_ids[0]) if token_ids else None)
            logprobs.append(None)
    return actual, logprobs


def _logprob_rows(logprob_dict: Any) -> list[dict[str, Any]]:
    rows = []
    for token_id, value in logprob_dict.items():
        rows.append(
            {
                "token_id": int(token_id),
                "logprob": float(getattr(value, "logprob", value)),
                "decoded_token": getattr(value, "decoded_token", None),
            }
        )
    rows.sort(key=lambda row: row["logprob"], reverse=True)
    return rows


def _run_ar_next_tokens_with_logprobs(
    llm: Any,
    items: list[ProbeItem],
    *,
    batch_size: int,
    logprobs_k: int,
) -> tuple[list[int | None], list[list[dict[str, Any]] | None]]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=logprobs_k,
    )
    actual: list[int | None] = []
    logprobs: list[list[dict[str, Any]] | None] = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = [_prompt_from_token_ids(item.prefix_token_ids) for item in batch]
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        for output in outputs:
            completion = output.outputs[0]
            token_ids = list(completion.token_ids)
            actual.append(int(token_ids[0]) if token_ids else None)
            if completion.logprobs:
                logprobs.append(_logprob_rows(completion.logprobs[0]))
            else:
                logprobs.append(None)
    return actual, logprobs


def _json_dumpable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ProbeItem):
        return value.__dict__
    if isinstance(value, dict):
        return {str(k): _json_dumpable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_dumpable(v) for v in value]
    return value


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay DFlash tree verifier nodes through target-only AR.",
    )
    parser.add_argument("--model", default="/mnt/lanxiangh/models/Step-3.7-Flash")
    parser.add_argument(
        "--verify-bundles",
        default=(
            "/root/data/vllm-ptd/dflash_step3p7_verify_debug_0620/"
            "tree_logical/dflash/tp8/bs1/dflash_runtime_verify_bundles.json"
        ),
    )
    parser.add_argument("--prompt-set", default="humaneval")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-nodes-per-step", type=int)
    parser.add_argument(
        "--node-indices",
        help="Optional comma-separated tree node indices to replay.",
    )
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
    parser.add_argument(
        "--ar-logprobs-k",
        type=int,
        default=0,
        help="If positive, include target-only AR top logprobs in mismatch records.",
    )
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    bundles_path = Path(args.verify_bundles)
    bundles = json.loads(bundles_path.read_text(encoding="utf-8"))
    tokenizer, prompt_token_ids = _load_prompt_token_ids(
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
    selected_node_indices = _parse_int_csv(args.node_indices)
    if selected_node_indices is not None:
        items = [item for item in items if item.node_idx in selected_node_indices]
    if not items:
        raise RuntimeError("No probe items selected.")

    print(
        "[tree-target-parity] selected "
        f"{len(items)} node prefixes from {bundles_path}"
    )
    print(
        "[tree-target-parity] max_steps="
        f"{args.max_steps} max_nodes_per_step={args.max_nodes_per_step}"
    )

    llm = _build_llm(args)
    try:
        if args.ar_logprobs_k > 0:
            actual_tokens, actual_logprobs = _run_ar_next_tokens_with_logprobs(
                llm,
                items,
                batch_size=args.batch_size,
                logprobs_k=args.ar_logprobs_k,
            )
        else:
            actual_tokens, actual_logprobs = _run_ar_next_tokens(
                llm,
                items,
                batch_size=args.batch_size,
            )
    finally:
        try:
            llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass

    mismatches = []
    for item, actual_token, actual_logprob_rows in zip(
        items, actual_tokens, actual_logprobs
    ):
        if actual_token != item.expected_token:
            mismatches.append(
                {
                    **item.__dict__,
                    "actual_token": actual_token,
                    "expected_text": tokenizer.decode([item.expected_token]),
                    "actual_text": (
                        tokenizer.decode([actual_token])
                        if actual_token is not None else None
                    ),
                    "prefix_tail_text": tokenizer.decode(item.prefix_token_ids[-32:]),
                    "actual_top_logprobs": actual_logprob_rows,
                }
            )

    first_mismatch = mismatches[0] if mismatches else None
    report = {
        "passed": not mismatches,
        "num_items": len(items),
        "num_mismatches": len(mismatches),
        "mismatch_rate": len(mismatches) / len(items),
        "first_mismatch": first_mismatch,
        "verify_bundles": str(bundles_path),
        "model": args.model,
        "prompt_set": args.prompt_set,
        "sample_start": args.sample_start,
        "num_samples": args.num_samples,
        "max_steps": args.max_steps,
        "max_nodes_per_step": args.max_nodes_per_step,
        "node_indices": args.node_indices,
        "ar_logprobs_k": args.ar_logprobs_k,
    }

    print(json.dumps(_json_dumpable(report), indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_json_dumpable(report), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[tree-target-parity] wrote {output_path}")

    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
