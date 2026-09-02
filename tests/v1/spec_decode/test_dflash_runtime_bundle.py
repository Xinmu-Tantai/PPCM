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

_DEBUG_DIR_ENV = "DFLASH_TEST_DEBUG_DIR"
_RUN_REAL_TEST_ENV = "DFLASH_RUN_REAL_RUNTIME_TESTS"
_DEFAULT_DEBUG_DIR = Path("/tmp") / f"debug_{datetime.now():%Y%m%d_%H%M%S}"

DEFAULT_TARGET_MODEL = "/data/models/Qwen3-8B"
DEFAULT_DRAFT_MODEL = (
    "/mnt/specdec-dev/checkpoints/specforge/outputs/"
    "nemotron-780k-and-codealpaca20k-v2-causal-distill-lr1e-4-anchorcnt512/"
    "epoch_6_step_583488"
)
DEFAULT_PROMPT = "Write a Python function that returns the Fibonacci sequence."


def _get_debug_dir() -> Path:
    debug_dir = Path(os.environ.get(_DEBUG_DIR_ENV, str(_DEFAULT_DEBUG_DIR)))
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


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


def _load_prompt(prompt: str | None, prompt_set: str | None, sample_index: int) -> str:
    if prompt is not None:
        return prompt
    if prompt_set == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        row = dataset[sample_index]
        return (
            "Write a solution to the following problem and make sure that it "
            f"passes the tests:\n```python\n{row['prompt']}\n```"
        )
    return DEFAULT_PROMPT


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu()
    summary: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }
    if tensor.numel() == 0:
        summary["values"] = []
        return summary
    if tensor.numel() <= 128:
        summary["values"] = tensor.tolist()
        return summary

    flat = tensor.reshape(-1)
    preview_count = min(8, flat.numel())
    summary["norm"] = float(tensor.float().norm().item())
    summary["min"] = float(tensor.float().amin().item())
    summary["max"] = float(tensor.float().amax().item())
    summary["preview"] = flat[:preview_count].tolist()
    return summary


def _json_ready(payload: dict[str, Any]) -> dict[str, Any]:
    ready: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            ready[key] = _tensor_summary(value)
        else:
            ready[key] = value
    return ready


def _node_expand_score(
    cum_lp: float,
    depth: int,
    score_mode: str,
    per_depth_entropy: list[float] | None,
    hybrid_alpha: float,
    parent_expansion_entropy: float | None = None,
) -> float:
    if score_mode == "entropy":
        if parent_expansion_entropy is not None:
            return parent_expansion_entropy
        if per_depth_entropy is not None and depth < len(per_depth_entropy):
            return per_depth_entropy[depth]
        return 0.0
    if score_mode == "hybrid":
        if parent_expansion_entropy is not None:
            ent = parent_expansion_entropy
        elif per_depth_entropy is not None and depth < len(per_depth_entropy):
            ent = per_depth_entropy[depth]
        else:
            ent = 0.0
        return cum_lp + hybrid_alpha * ent
    return cum_lp


def _augment_tree_builder_diagnostics(bundle: dict[str, Any]) -> None:
    topk_tok = bundle.get("builder_topk_tok")
    topk_lp = bundle.get("builder_topk_lp")
    token_ids = bundle.get("builder_tree_node_token_ids")
    parent_indices = bundle.get("builder_tree_parent_indices")
    depths = bundle.get("builder_tree_depths")
    if not all(isinstance(v, torch.Tensor) for v in (
        topk_tok, topk_lp, token_ids, parent_indices, depths
    )):
        return

    per_depth_entropy_t = bundle.get("builder_per_depth_entropy")
    per_depth_entropy = None
    if isinstance(per_depth_entropy_t, torch.Tensor) and per_depth_entropy_t.numel() > 0:
        per_depth_entropy = per_depth_entropy_t.tolist()
    score_mode = str(bundle.get("builder_score_mode", "accum_logp"))
    hybrid_alpha = float(bundle.get("builder_hybrid_alpha", 1.0))

    num_nodes = int(token_ids.numel())
    child_ranks = torch.full((num_nodes,), -1, dtype=torch.int32)
    cum_logprobs = torch.zeros((num_nodes,), dtype=torch.float32)
    expand_scores = torch.zeros((num_nodes,), dtype=torch.float32)
    expand_scores[0] = float(
        _node_expand_score(0.0, 0, score_mode, per_depth_entropy, hybrid_alpha)
    )
    for node_idx in range(1, num_nodes):
        depth = int(depths[node_idx].item())
        parent_idx = int(parent_indices[node_idx].item())
        row_idx = depth - 1
        matches = (topk_tok[row_idx] == token_ids[node_idx]).nonzero(as_tuple=True)[0]
        if matches.numel() == 0:
            continue
        child_rank = int(matches[0].item())
        child_ranks[node_idx] = child_rank
        cum_lp = float(cum_logprobs[parent_idx].item() + topk_lp[row_idx, child_rank].item())
        cum_logprobs[node_idx] = cum_lp
        parent_entropy = None
        if per_depth_entropy is not None and row_idx < len(per_depth_entropy):
            parent_entropy = per_depth_entropy[row_idx]
        expand_scores[node_idx] = float(
            _node_expand_score(
                cum_lp,
                depth,
                score_mode,
                per_depth_entropy,
                hybrid_alpha,
                parent_expansion_entropy=parent_entropy,
            )
        )

    bundle["builder_tree_node_child_ranks"] = child_ranks
    bundle["builder_tree_node_cum_logprobs"] = cum_logprobs
    bundle["builder_tree_node_expand_scores"] = expand_scores
    bundle["builder_tree_node_added_order"] = torch.arange(
        num_nodes, dtype=torch.int32
    )


def _write_runtime_bundle(name: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    debug_dir = _get_debug_dir()
    pt_path = debug_dir / f"{name}.pt"
    json_path = debug_dir / f"{name}.json"
    torch.save(payload, pt_path)
    json_path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return pt_path, json_path


def capture_vllm_runtime_bundle(
    *,
    model: str,
    draft_model: str,
    prompt: str | None = None,
    prompt_set: str | None = None,
    sample_index: int = 0,
    max_model_len: int = 32768,
    block_size: int = 16,
    tree_width: int = 7,
    max_tree_budget: int = 255,
    tree_draft: str = "accum_logp",
    max_draft_passes: int = 1,
    tree_prune_ratio: float = 0.25,
    tree_construction: str = "breadth_first",
    tree_attn_kernel: str = "optimus",
    attention_backend: str = "FLASH_ATTN",
    max_tokens: int = 32,
    seed: int = 0,
    enforce_eager: bool = True,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-model vLLM runtime bundle.")
    if not Path(model).exists():
        raise FileNotFoundError(f"Target model path not found: {model}")
    if not Path(draft_model).exists():
        raise FileNotFoundError(f"Draft model path not found: {draft_model}")

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    llm_kwargs: dict[str, Any] = {
        "model": model,
        "trust_remote_code": True,
        "seed": seed,
        "max_model_len": max_model_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 51200,
        "disable_log_stats": True,
        # Keep the runtime bundle harness on the debug-safe eager path so
        # TorchInductor/CUDAGraph compilation does not mask the actual
        # tree-building mismatch we are trying to inspect.
        "enforce_eager": enforce_eager,
        "attention_backend": attention_backend,
        "speculative_config": {
            "method": "dflash",
            "model": draft_model,
            "num_speculative_tokens": block_size - 1,
            "max_model_len": max_model_len,
            "head_type": "causal",
            "tree_width": tree_width,
            "max_tree_budget": max_tree_budget,
            "tree_draft": tree_draft,
            "max_draft_passes": max_draft_passes,
            "tree_prune_ratio": tree_prune_ratio,
            "tree_construction": tree_construction,
            "tree_attn_kernel": tree_attn_kernel,
            "num_cudagraph_tree_captures": 0,
        },
    }
    if tree_attn_kernel == "optimus":
        llm_kwargs["block_size"] = 128

    llm = LLM(**llm_kwargs)
    try:
        tokenizer = llm.get_tokenizer()
        prompt_text = _maybe_apply_chat_template(
            tokenizer,
            _load_prompt(prompt, prompt_set, sample_index),
        )
        prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        worker = llm.llm_engine.model_executor.driver_worker.worker
        drafter = worker.model_runner.drafter
        assert hasattr(drafter, "clear_runtime_bundles")
        assert hasattr(drafter, "get_runtime_bundles")
        drafter.clear_runtime_bundles()
        worker.model_runner._dflash_runtime_verify_bundles = []

        llm.generate(
            [prompt_text],
            SamplingParams(temperature=0.0, max_tokens=max_tokens),
            use_tqdm=False,
        )

        runtime_bundles = drafter.get_runtime_bundles()
        if not runtime_bundles:
            raise RuntimeError("No DFlash runtime bundle was captured from the proposer.")

        bundle = dict(runtime_bundles[0])
        verify_bundles = getattr(
            worker.model_runner, "_dflash_runtime_verify_bundles", []
        )
        if verify_bundles:
            bundle.update(verify_bundles[0])
        bundle.update(
            {
                "source": "vllm",
                "prompt_text": prompt_text,
                "prompt_token_ids": prompt_token_ids,
                "model": model,
                "draft_model": draft_model,
                "block_size": block_size,
                "tree_width": tree_width,
                "tree_draft": tree_draft,
                "tree_construction": tree_construction,
                "tree_attn_kernel": tree_attn_kernel,
                "attention_backend": attention_backend,
                "seed": seed,
            }
        )
        _augment_tree_builder_diagnostics(bundle)
        pt_path, json_path = _write_runtime_bundle("vllm_step0_runtime_bundle", bundle)
        bundle["artifact_pt"] = str(pt_path)
        bundle["artifact_json"] = str(json_path)
        return bundle
    finally:
        llm.llm_engine.engine_core.shutdown()


def _default_enabled() -> bool:
    return os.environ.get(_RUN_REAL_TEST_ENV) == "1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
def test_capture_vllm_runtime_bundle_real_model() -> None:
    if not _default_enabled():
        pytest.skip(f"Set {_RUN_REAL_TEST_ENV}=1 to run the real-model capture.")
    bundle = capture_vllm_runtime_bundle(
        model=os.environ.get("DFLASH_TARGET_MODEL", DEFAULT_TARGET_MODEL),
        draft_model=os.environ.get("DFLASH_DRAFT_MODEL", DEFAULT_DRAFT_MODEL),
        prompt=os.environ.get("DFLASH_PROMPT"),
        prompt_set=os.environ.get("DFLASH_PROMPT_SET", "humaneval"),
        sample_index=int(os.environ.get("DFLASH_SAMPLE_INDEX", "0")),
    )
    assert Path(bundle["artifact_pt"]).exists()
    assert Path(bundle["artifact_json"]).exists()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a real vLLM DFlash step-0 runtime bundle.",
    )
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-set", choices=["humaneval"], default="humaneval")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tree-width", type=int, default=7)
    parser.add_argument("--max-tree-budget", type=int, default=255)
    parser.add_argument("--tree-draft", default="accum_logp")
    parser.add_argument("--max-draft-passes", type=int, default=1)
    parser.add_argument("--tree-prune-ratio", type=float, default=0.25)
    parser.add_argument("--tree-construction", default="breadth_first")
    parser.add_argument("--tree-attn-kernel", default="optimus")
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--debug-dir")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.debug_dir:
        os.environ[_DEBUG_DIR_ENV] = args.debug_dir
    bundle = capture_vllm_runtime_bundle(
        model=args.model,
        draft_model=args.draft_model,
        prompt=args.prompt,
        prompt_set=args.prompt_set,
        sample_index=args.sample_index,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        tree_width=args.tree_width,
        max_tree_budget=args.max_tree_budget,
        tree_draft=args.tree_draft,
        max_draft_passes=args.max_draft_passes,
        tree_prune_ratio=args.tree_prune_ratio,
        tree_construction=args.tree_construction,
        tree_attn_kernel=args.tree_attn_kernel,
        attention_backend=args.attention_backend,
        max_tokens=args.max_tokens,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
    )
    print(json.dumps(_json_ready(bundle), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
