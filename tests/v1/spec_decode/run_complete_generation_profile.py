from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, "/home/i-hulanxiang/workspace/vllm-parallel-drafting")

from transformers import AutoTokenizer

from vllm import SamplingParams

from examples.offline_inference.dflash_profiling import (
    apply_chat_template,
    get_prompt_bank,
    run_native_profile,
)


_DEFAULT_DEBUG_DIR = Path("/tmp") / f"debug_complete_vllm_{datetime.now():%Y%m%d_%H%M%S}"
DEFAULT_TARGET_MODEL = "/data/models/Qwen3-8B"
DEFAULT_DRAFT_MODEL = (
    "/mnt/specdec-dev/checkpoints/specforge/outputs/"
    "nemotron-780k-and-codealpaca20k-v2-causal-distill-lr1e-4-anchorcnt512/"
    "epoch_6_step_583488"
)


def _get_debug_dir(path: str | None) -> Path:
    debug_dir = Path(path) if path else _DEFAULT_DEBUG_DIR
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _parse_sample_indices(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",")]
    indices = [int(v) for v in values if v]
    if not indices:
        raise ValueError("At least one sample index must be provided.")
    return indices


def run_vllm_complete_generation(
    *,
    debug_dir: Path,
    model: str,
    draft_model: str,
    sample_indices: list[int],
    max_model_len: int = 32768,
    block_size: int = 16,
    tree_width: int = 7,
    max_tree_budget: int = 255,
    tree_draft: str = "accum_logp",
    tree_hybrid_alpha: float = 1.0,
    max_draft_passes: int = 1,
    tree_prune_ratio: float = 0.25,
    tree_construction: str = "breadth_first",
    tree_attn_kernel: str = "optimus",
    attention_backend: str = "FLASH_ATTN",
    max_tokens: int = 2048,
    seed: int = 0,
    enforce_eager: bool = True,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompt_bank = apply_chat_template(tokenizer, get_prompt_bank("humaneval"))
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    args = SimpleNamespace(
        model=model,
        draft_model=draft_model,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=51200,
        max_num_seqs=1,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
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
        num_cudagraph_tree_captures=0,
        num_warmup_runs=0,
        num_runs=1,
        sleep_after_stop=0.0,
    )

    per_sample = []
    for sample_index in sample_indices:
        prompt_text = prompt_bank[sample_index]
        sample_dir = debug_dir / f"sample_{sample_index:03d}"
        run_output_dir = sample_dir / "vllm_profile"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        result = run_native_profile(
            args=args,
            mode="dflash",
            prompt_batches=[[prompt_text]],
            sampling_params=sampling_params,
            tp_size=1,
            batch_size=1,
            effective_num_speculative_tokens=block_size - 1,
            effective_max_num_seqs=1,
            profiler_config={"profiler": "cuda"},
            run_output_dir=run_output_dir,
        )

        topk_path = run_output_dir / "topk_log.json"
        topk_log = []
        if topk_path.exists():
            topk_log = json.loads(topk_path.read_text(encoding="utf-8"))

        outputs = result.get("outputs", [])
        sample_summary = {
            "sample_index": sample_index,
            "prompt_text": prompt_text,
            "output_text": outputs[0][1] if outputs else "",
            "elapsed": float(result["elapsed"]),
            "total_output_tokens": int(result["total_output_tokens"]),
            "total_prompt_tokens": int(result["total_prompt_tokens"]),
            "throughput": float(result["throughput"]),
            "prefill_throughput": float(result["prefill_throughput"]),
            "decode_throughput": float(result["decode_throughput"]),
            "gpu_utilization": float(result["gpu_utilization"]),
            "draft_tokens": float(result["draft_tokens"]),
            "accepted_tokens": float(result["accepted_tokens"]),
            "drafts": float(result["drafts"]),
            "acceptance_rate": float(result["acceptance_rate"]),
            "acceptance_length": float(result["acceptance_length"]),
            "per_pos_acceptance_rate": list(result.get("per_pos_acceptance_rate", [])),
            "acceptance_length_histogram": list(
                result.get("acceptance_length_histogram", [])
            ),
            "avg_tree_nodes_per_depth": list(
                result.get("avg_tree_nodes_per_depth", [])
            ),
            "phase_cuda": dict(result.get("phase_cuda", {})),
            "mean_time_per_output_token_s": float(
                result["mean_time_per_output_token_s"]
            ),
            "d0_acceptance_rate": (
                float(result["per_pos_acceptance_rate"][0])
                if result.get("per_pos_acceptance_rate")
                else 0.0
            ),
            "topk_log_path": str(topk_path),
            "topk_log_entries": topk_log,
        }
        (sample_dir / "vllm_complete_generation.json").write_text(
            json.dumps(sample_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        per_sample.append(sample_summary)

    summary = {
        "source": "vllm",
        "debug_dir": str(debug_dir),
        "config": {
            "block_size": block_size,
            "tree_width": tree_width,
            "max_tree_budget": max_tree_budget,
            "max_tokens": max_tokens,
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
        "d0_acceptance_rate": (
            sum(s["d0_acceptance_rate"] for s in per_sample) / len(per_sample)
            if per_sample
            else 0.0
        ),
        "samples": per_sample,
    }
    (debug_dir / "vllm_complete_generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run vLLM complete-generation profiling for selected Humaneval prompts.",
    )
    parser.add_argument("--debug-dir")
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--sample-indices", default="0,1")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--tree-width", type=int, default=7)
    parser.add_argument("--max-tree-budget", type=int, default=255)
    parser.add_argument("--tree-draft", default="accum_logp")
    parser.add_argument("--tree-hybrid-alpha", type=float, default=1.0)
    parser.add_argument("--max-draft-passes", type=int, default=1)
    parser.add_argument("--tree-prune-ratio", type=float, default=0.25)
    parser.add_argument("--tree-construction", default="breadth_first")
    parser.add_argument("--tree-attn-kernel", default="optimus")
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    debug_dir = _get_debug_dir(args.debug_dir)
    summary = run_vllm_complete_generation(
        debug_dir=debug_dir,
        model=args.model,
        draft_model=args.draft_model,
        sample_indices=_parse_sample_indices(args.sample_indices),
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        tree_width=args.tree_width,
        max_tree_budget=args.max_tree_budget,
        tree_draft=args.tree_draft,
        tree_hybrid_alpha=args.tree_hybrid_alpha,
        max_draft_passes=args.max_draft_passes,
        tree_prune_ratio=args.tree_prune_ratio,
        tree_construction=args.tree_construction,
        tree_attn_kernel=args.tree_attn_kernel,
        attention_backend=args.attention_backend,
        max_tokens=args.max_tokens,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
