"""
Output parity test for Step-3.7-Flash DFlash speculative decoding.

Verifies that AR, DFlash-logical, and DFlash-physical all produce
bit-identical output token sequences for the same greedy prompts.

Key invariant: speculative decoding must be *lossless* — the accepted
tokens are always what the target model would have produced under pure AR
(temperature=0).  Any divergence in token IDs indicates a verification bug.

Run as a standalone script (requires 8xGPU):
    PYTHONPATH=/root/workspace/vllm-parallel-drafting \
    python tests/v1/spec_decode/test_step3p7_output_parity.py \
        --model /mnt/lanxiangh/models/Step-3.7-Flash \
        --draft-model /mnt/lanxiangh/checkpoints/specforge/ptd-step3p7-fkl-200k-epoch6-3e-4-no-gamma

Or as a pytest (marks as large_gpu, skipped unless MODEL env var is set):
    pytest tests/v1/spec_decode/test_step3p7_output_parity.py -v \
        --model /mnt/lanxiangh/models/Step-3.7-Flash \
        --draft-model /mnt/lanxiangh/checkpoints/specforge/...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Prompt helpers (mirrors dflash_profiling._apply_step3p7_specforge_template)
# ---------------------------------------------------------------------------

_HUMANEVAL_PROBLEMS = [
    # HumanEval #0
    (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        "```python\n"
        "from typing import List\n\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        '    """ Check if in given list of numbers, are any two numbers closer to each\n'
        "    other than given threshold.\n"
        "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
        "    False\n"
        "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
        "    True\n"
        "    \"\"\"\n\n"
        "```"
    ),
    # HumanEval #1
    (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        "```python\n"
        "from typing import List\n\n\n"
        "def separate_paren_groups(paren_string: str) -> List[str]:\n"
        '    """ Input to this function is a string containing multiple groups of nested\n'
        "    parentheses. Your goal is to separate those group into separate strings and\n"
        "    return the list of those. Separate groups are balanced (each open brace is\n"
        "    properly closed) and not nested within each other. Ignore any spaces in the\n"
        "    input string.\n"
        "    >>> separate_paren_groups('( ) (( )) (( )( ))')\n"
        "    ['()', '(())', '(()())']\n"
        "    \"\"\"\n\n"
        "```"
    ),
    # HumanEval #2
    (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        "```python\n\n\n"
        "def truncate_number(number: float) -> float:\n"
        '    """ Given a positive floating point number, it can be decomposed into and\n'
        "    integer part (largest integer smaller than given number) and decimals\n"
        "    (leftover part always smaller than 1). Return the decimal part of the number.\n"
        "    >>> truncate_number(3.5)\n"
        "    0.5\n"
        "    \"\"\"\n\n"
        "```"
    ),
    # HumanEval #3
    (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        "```python\n"
        "from typing import List\n\n\n"
        "def below_zero(operations: List[int]) -> bool:\n"
        '    """ You\'re given a list of deposit and withdrawal operations on a bank\n'
        "    account that starts with zero balance. Your task is to detect if at any\n"
        "    point the balance of account falls below zero, and at that point function\n"
        "    should return True. Otherwise it should return False.\n"
        "    >>> below_zero([1, 2, 3])\n"
        "    False\n"
        "    >>> below_zero([1, 2, -4, 5])\n"
        "    True\n"
        "    \"\"\"\n\n"
        "```"
    ),
]


def _apply_step3p7_template(prompt: str) -> str:
    """Nothink template matching SpecForge training distribution.

    Matches dflash_profiling._apply_step3p7_specforge_template exactly.
    """
    return (
        "<|im_start|>system\n"
        "Reasoning: low\n\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


# ---------------------------------------------------------------------------
# Engine runner
# ---------------------------------------------------------------------------

def _build_llm(
    *,
    model: str,
    draft_model: str | None,
    tp_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    tree_kv_layout: str | None,
    max_tree_budget: int,
    enable_expert_parallel: bool,
    disable_cascade_attn: bool,
) -> Any:
    from vllm import LLM

    llm_kwargs: dict[str, Any] = dict(
        model=model,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=16384,
        max_num_seqs=1,
        max_model_len=max_model_len,
        enforce_eager=False,
        disable_log_stats=False,
        language_model_only=True,
        limit_mm_per_prompt={"image": 0},
        skip_mm_profiling=True,
        mm_processor_cache_gb=0,
    )
    if enable_expert_parallel:
        llm_kwargs["enable_expert_parallel"] = True
    if disable_cascade_attn:
        llm_kwargs["disable_cascade_attn"] = True

    if draft_model is not None:
        llm_kwargs["speculative_config"] = {
            "method": "dflash",
            "model": draft_model,
            "num_speculative_tokens": 16,
            "max_model_len": max_model_len,
            "head_type": "causal",
            "tree_width": 7,
            "max_tree_budget": max_tree_budget,
            "tree_draft": "accum_logp",
            "max_draft_passes": 0,
            "tree_prune_ratio": 0.25,
            "tree_construction": "breadth_first",
            "tree_attn_kernel": "triton",
            "tree_kv_layout": tree_kv_layout or "logical",
        }

    return LLM(**llm_kwargs)


def _generate_token_ids(llm: Any, prompts: list[str], max_tokens: int) -> list[list[int]]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    return [list(out.outputs[0].token_ids) for out in outputs]


def _cleanup_llm(llm: Any) -> None:
    import gc
    import torch

    if llm is not None:
        try:
            llm.sleep(level=2, mode="abort")
            llm.llm_engine.engine_core.shutdown(timeout=60)
            llm.llm_engine = None
        except Exception:
            pass

    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
    cleanup_dist_env_and_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def run_parity_check(
    *,
    model: str,
    draft_model: str,
    tp_size: int = 8,
    max_model_len: int = 3072,
    max_tokens: int = 256,
    max_tree_budget: int = 127,
    gpu_memory_utilization: float = 0.80,
    enable_expert_parallel: bool = True,
    disable_cascade_attn: bool = True,
    num_prompts: int = 4,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run AR, DFlash-logical, DFlash-physical and compare token IDs.

    Returns a summary dict with 'passed' bool and per-sample diffs.
    """
    prompts = [_apply_step3p7_template(p) for p in _HUMANEVAL_PROBLEMS[:num_prompts]]

    _common = dict(
        model=model,
        tp_size=tp_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_tree_budget=max_tree_budget,
        enable_expert_parallel=enable_expert_parallel,
        disable_cascade_attn=disable_cascade_attn,
    )

    results: dict[str, list[list[int]]] = {}

    for mode, draft, layout in [
        ("ar", None, None),
        ("dflash_logical", draft_model, "logical"),
        ("dflash_physical", draft_model, "physical"),
    ]:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running mode={mode} layout={layout}")
            print(f"{'='*60}")

        llm = _build_llm(draft_model=draft, tree_kv_layout=layout, **_common)
        # Warmup pass
        _generate_token_ids(llm, prompts[:1], max_tokens=32)
        # Actual run
        token_ids = _generate_token_ids(llm, prompts, max_tokens)
        results[mode] = token_ids
        _cleanup_llm(llm)

        if verbose:
            from vllm import SamplingParams
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
                for i, ids in enumerate(token_ids):
                    text = tok.decode(ids, skip_special_tokens=True)
                    print(f"  [sample {i}] {len(ids)} tokens: {text[:120]!r}")
            except Exception:
                for i, ids in enumerate(token_ids):
                    print(f"  [sample {i}] {len(ids)} tokens")

    # Compare
    ar_ids = results["ar"]
    logical_ids = results["dflash_logical"]
    physical_ids = results["dflash_physical"]

    divergences: list[dict[str, Any]] = []
    for i in range(num_prompts):
        ar = ar_ids[i]
        log = logical_ids[i]
        phy = physical_ids[i]
        min_len = min(len(ar), len(log), len(phy))

        first_ar_vs_log = next(
            (j for j in range(min_len) if ar[j] != log[j]), None
        )
        first_ar_vs_phy = next(
            (j for j in range(min_len) if ar[j] != phy[j]), None
        )
        first_log_vs_phy = next(
            (j for j in range(min_len) if log[j] != phy[j]), None
        )

        diverged = (
            first_ar_vs_log is not None
            or first_ar_vs_phy is not None
            or first_log_vs_phy is not None
            or len(ar) != len(log)
            or len(ar) != len(phy)
        )

        entry = {
            "sample_index": i,
            "ar_tokens": len(ar),
            "logical_tokens": len(log),
            "physical_tokens": len(phy),
            "first_divergence_ar_vs_logical": first_ar_vs_log,
            "first_divergence_ar_vs_physical": first_ar_vs_phy,
            "first_divergence_logical_vs_physical": first_log_vs_phy,
            "passed": not diverged,
        }
        divergences.append(entry)

        if verbose:
            status = "PASS" if not diverged else "FAIL"
            print(
                f"[sample {i}] {status}: "
                f"ar={len(ar)} logical={len(log)} physical={len(phy)} | "
                f"ar_vs_log_div={first_ar_vs_log} ar_vs_phy_div={first_ar_vs_phy} "
                f"log_vs_phy_div={first_log_vs_phy}"
            )

    all_passed = all(d["passed"] for d in divergences)
    summary = {
        "passed": all_passed,
        "num_prompts": num_prompts,
        "divergences": divergences,
    }
    if verbose:
        print(f"\n{'='*60}")
        print(f"Overall: {'PASSED' if all_passed else 'FAILED'}")
        print(f"{'='*60}")

    return summary


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------

_STEP3P7_MODEL = os.environ.get(
    "STEP3P7_MODEL", "/mnt/lanxiangh/models/Step-3.7-Flash"
)
_STEP3P7_DRAFT = os.environ.get(
    "STEP3P7_DRAFT",
    "/mnt/lanxiangh/checkpoints/specforge/ptd-step3p7-fkl-200k-epoch6-3e-4-no-gamma",
)


@pytest.mark.skipif(
    not (Path(_STEP3P7_MODEL).is_dir() and Path(_STEP3P7_DRAFT).is_dir()),
    reason=(
        "Step-3.7-Flash model and draft not available. "
        f"Set STEP3P7_MODEL={_STEP3P7_MODEL} and STEP3P7_DRAFT={_STEP3P7_DRAFT}"
    ),
)
@pytest.mark.parametrize("num_prompts", [4])
def test_step3p7_ar_dflash_output_parity(num_prompts: int) -> None:
    """AR and DFlash (logical + physical) must produce bit-identical outputs."""
    summary = run_parity_check(
        model=_STEP3P7_MODEL,
        draft_model=_STEP3P7_DRAFT,
        num_prompts=num_prompts,
        max_tokens=128,
        verbose=True,
    )
    divergences = [d for d in summary["divergences"] if not d["passed"]]
    assert summary["passed"], (
        f"{len(divergences)}/{num_prompts} samples diverged:\n"
        + "\n".join(json.dumps(d, indent=2) for d in divergences)
    )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step-3.7-Flash AR vs DFlash parity check")
    p.add_argument("--model", default=_STEP3P7_MODEL)
    p.add_argument("--draft-model", default=_STEP3P7_DRAFT)
    p.add_argument("--tp-size", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=3072)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-tree-budget", type=int, default=127)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument("--no-expert-parallel", action="store_true")
    p.add_argument("--output-json", default=None,
                   help="Write summary to this JSON file")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = run_parity_check(
        model=args.model,
        draft_model=args.draft_model,
        tp_size=args.tp_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        max_tree_budget=args.max_tree_budget,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_prompts=args.num_prompts,
        enable_expert_parallel=not args.no_expert_parallel,
        verbose=True,
    )
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"Summary written to {args.output_json}")
    sys.exit(0 if summary["passed"] else 1)
