#!/usr/bin/env python3
"""Probe vLLM DFlash draft logits at decode step 0.

Run under vllm_ptd_venv.  Launches vLLM with chain-spec mode (num_speculative_tokens
= block_size - 1 = 15), enables the DFlash runtime-bundle capture at step 0, runs
one generation request, and saves the captured bundle to a .pt file.

The saved file is consumed by probe_specforge_draft_logits.py which feeds the EXACT
SAME raw_target_hidden_states tensor through SpecForge's DFlashDraftModel, allowing
a true apples-to-apples comparison of the two draft implementations.

Typical usage (from run_specforge_vllm_draft_parity.sh):

    PYTHONPATH=/root/workspace/vllm-parallel-drafting \\
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
    /root/workspace/vllm-parallel-drafting/vllm_ptd_venv/bin/python \\
        tests/v1/spec_decode/probe_vllm_draft_logits.py \\
        --out /tmp/vllm_draft_probe.pt \\
        --tp 8
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

_VLLM_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_VLLM_ROOT))

import torch
from vllm import LLM, SamplingParams  # noqa: E402

_DEFAULT_TARGET = "/mnt/lanxiangh/models/Step-3.7-Flash"
_DEFAULT_DRAFT  = (
    "/mnt/lanxiangh/checkpoints/specforge/"
    "ptd-step3p7-fkl-200k-epoch6-3e-4-no-gamma"
)
# block_size - 1 = 16 - 1 = 15
_DEFAULT_NUM_SPEC = 15   # block_size - 1 = 16 - 1
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 51200
_DEFAULT_PROMPT = (
    "<|im_start|>user\n"
    "Implement a Python function `truncate_number(number: float) -> float` that "
    "returns only the decimal part of the number. "
    "For example: truncate_number(3.5) == 0.5<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# ---------------------------------------------------------------------------
# Worker-side functions called via collective_rpc
# ---------------------------------------------------------------------------

def _enable_probe(worker, steps=(0,)):
    """Enable DFlash runtime-bundle capture at the given steps."""
    worker_obj = getattr(worker, "worker", worker)
    drafter = getattr(
        getattr(worker_obj, "model_runner", None), "drafter", None
    )
    if drafter is None:
        return {"error": "missing_drafter"}
    enable_fn = getattr(drafter, "enable_dflash_debug_artifacts", None)
    if enable_fn is not None:
        enable_fn()
    set_fn = getattr(drafter, "set_runtime_capture_steps", None)
    if set_fn is not None:
        set_fn(list(steps))
    clear_fn = getattr(drafter, "clear_runtime_bundles", None)
    if clear_fn is not None:
        clear_fn()
    return {"enabled": True, "steps": list(steps)}


def _save_probe_data(worker, save_path):
    """Save captured step-0 runtime bundle to disk from worker rank 0."""
    import torch  # noqa: PLC0415
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        is_rank0 = get_tensor_model_parallel_rank() == 0
    except Exception:
        is_rank0 = True  # single-process fallback

    worker_obj = getattr(worker, "worker", worker)
    drafter = getattr(
        getattr(worker_obj, "model_runner", None), "drafter", None
    )
    if drafter is None:
        return {"error": "missing_drafter", "rank0": is_rank0}

    bundles = list(drafter.get_runtime_bundles() or [])
    if not bundles:
        return {
            "error": "no_bundles_captured",
            "capture_steps": list(getattr(drafter, "_runtime_capture_steps", [])),
            "debug_enabled": getattr(drafter, "_dflash_debug_artifacts_enabled", False),
            "rank0": is_rank0,
        }

    b = bundles[0]

    def _maybe_tensor(key):
        v = b.get(key)
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v.detach().float().cpu()
        return v

    data = {
        "step": b.get("step", 0),
        "raw_target_hidden_states":    _maybe_tensor("raw_target_hidden_states"),
        "combined_target_hidden_states": _maybe_tensor("combined_target_hidden_states"),
        "draft_logits_req0":           _maybe_tensor("draft_logits_req0"),
        "query_input_ids":             _maybe_tensor("query_input_ids"),
        "query_positions":             _maybe_tensor("query_positions"),
        "target_positions":            _maybe_tensor("target_positions"),
        "target_token_ids":            _maybe_tensor("target_token_ids"),
        "next_token_ids":              _maybe_tensor("next_token_ids"),
    }
    data = {k: v for k, v in data.items() if v is not None}

    if is_rank0:
        torch.save(data, save_path)
        shapes = {
            k: list(v.shape) for k, v in data.items()
            if isinstance(v, torch.Tensor)
        }
        return {"saved": save_path, "step": data.get("step"), "shapes": shapes}
    return {"rank0": False}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Probe vLLM DFlash draft logits at step 0")
    p.add_argument("--model", default=_DEFAULT_TARGET)
    p.add_argument("--draft-model", default=_DEFAULT_DRAFT)
    p.add_argument("--out", required=True, help="Path to save .pt probe file")
    p.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    p.add_argument(
        "--num-speculative-tokens", type=int, default=_DEFAULT_NUM_SPEC,
        help="Number of draft tokens (block_size - 1)"
    )
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument(
        "--max-num-batched-tokens", type=int, default=_DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )
    p.add_argument(
        "--gpu-memory-utilization", type=float, default=0.92,
        help="GPU memory utilization for vLLM"
    )
    p.add_argument("--prompt", default=_DEFAULT_PROMPT)
    p.add_argument(
        "--max-tokens", type=int, default=1,
        help="Max tokens to generate (1 is enough to trigger one draft step)"
    )
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    p.add_argument(
        "--disable-cascade-attn", action="store_true", default=True,
        help="Disable cascade attention (recommended for single-req probing)"
    )
    p.add_argument("--enable-expert-parallel", action="store_true", default=False)
    p.add_argument(
        "--tree-attn-kernel", default="triton",
        help="DFlash tree attention kernel (triton or optimus)"
    )
    return p.parse_args()


def main():
    args = _parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[probe_vllm] Initialising vLLM: model={args.model}")
    print(f"[probe_vllm] draft={args.draft_model}  tp={args.tp}  spec={args.num_speculative_tokens}")

    num_spec = args.num_speculative_tokens
    llm_kwargs: dict = dict(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        # Step-3.7-Flash is a multimodal model class; disable MM to save memory
        language_model_only=True,
        limit_mm_per_prompt={"image": 0},
        skip_mm_profiling=True,
        mm_processor_cache_gb=0,
        speculative_config={
            "method": "dflash",
            "model": args.draft_model,
            "num_speculative_tokens": num_spec,
            "max_model_len": args.max_model_len,
            "head_type": "auto",
            "tree_width": num_spec,
            "max_tree_budget": num_spec,
            "tree_draft": "accum_logp",
            "tree_hybrid_alpha": 1.0,
            "max_draft_passes": 1,
            "tree_prune_ratio": 0.25,
            "tree_construction": "depth_first",
            "tree_attn_kernel": args.tree_attn_kernel,
            "tree_kv_layout": "logical",
            "num_cudagraph_tree_captures": 0,
        },
    )
    if args.disable_cascade_attn:
        llm_kwargs["disable_cascade_attn"] = True
    if args.enable_expert_parallel:
        llm_kwargs["enable_expert_parallel"] = True

    llm = LLM(**llm_kwargs)

    # Enable debug artifact capture at step 0 on all ranks
    print("[probe_vllm] Installing step-0 debug probe ...")
    results = llm.collective_rpc(_enable_probe, args=((0,),))
    print(f"[probe_vllm] Probe install result (rank 0): {results[0] if results else 'N/A'}")

    # Run one generation request — a single max_tokens=1 generation is enough to
    # trigger one speculative decode iteration and capture the draft bundle
    print(f"[probe_vllm] Running generation with prompt (len={len(args.prompt)}) ...")
    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        ignore_eos=True,
    )
    outputs = llm.generate([args.prompt], params)
    print(f"[probe_vllm] Generated: {repr(outputs[0].outputs[0].text[:80])}")

    # Save captured data from rank 0
    print(f"[probe_vllm] Saving probe data to {out_path} ...")
    save_results = llm.collective_rpc(_save_probe_data, args=(str(out_path),))
    r0 = save_results[0] if save_results else {}
    if "error" in r0:
        print(f"[probe_vllm] ERROR saving probe: {r0}")
        sys.exit(1)
    print(f"[probe_vllm] Saved. Tensor shapes: {r0.get('shapes', {})}")

    # Sanity-check the saved file
    probe = torch.load(out_path, weights_only=True)
    print("[probe_vllm] Verification of saved file:")
    for k, v in sorted(probe.items()):
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={list(v.shape)} dtype={v.dtype} "
                  f"norm={v.float().norm().item():.4f}")
        else:
            print(f"  {k}: {v}")

    print(f"[probe_vllm] Done → {out_path}")


if __name__ == "__main__":
    main()
