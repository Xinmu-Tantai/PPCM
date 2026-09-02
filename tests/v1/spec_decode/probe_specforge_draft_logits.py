#!/usr/bin/env python3
"""Probe SpecForge DFlash draft logits using the SAME target hidden states as vLLM.

Run under specforge_venv.  Loads the .pt file produced by probe_vllm_draft_logits.py,
extracts raw_target_hidden_states + noise token IDs + position IDs, then runs them
through SpecForge's DFlashDraftModel.  Saves a second .pt file with SpecForge's
draft_logits and intermediate activations.

Key design choices
------------------
* We use the target hidden states captured INSIDE vLLM (before the fc projection),
  ensuring both code paths receive identical input tensors and isolating any
  divergence to the draft model code itself.
* The lm_head (shared from the target in both SpecForge training and vLLM runtime)
  is loaded as a thin F.linear call from the target model's safetensors shard,
  so we never need to load the full 30 B target model.
* The DFlashDraftModel is run in CAUSAL mode (the model's causal_head=True setting
  is respected — we do NOT override is_causal=False as spec_generate does for its
  bidirectional pass).

Typical usage (from run_specforge_vllm_draft_parity.sh):

    PYTHONPATH=/root/workspace/specforge \\
    CUDA_VISIBLE_DEVICES=0 \\
    /root/workspace/specforge/specforge_venv/bin/python \\
        tests/v1/spec_decode/probe_specforge_draft_logits.py \\
        --vllm-probe /tmp/vllm_draft_probe.pt \\
        --draft-model /mnt/lanxiangh/checkpoints/specforge/ptd-step3p7-fkl-200k-epoch6-3e-4-no-gamma \\
        --target-lm-head-shard /mnt/lanxiangh/models/Step-3.7-Flash/model-00024.safetensors \\
        --out /tmp/specforge_draft_probe.pt
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

# ---- optional SpecForge dependency stubs (yunchang / USP) ------------------
# SpecForge's __init__ eagerly imports distributed attention code backed by
# yunchang.  For an inference-only probe we only need the draft model class.
for _mod in ("yunchang", "yunchang.globals", "yunchang.comm",
             "yunchang.ring", "yunchang.ulysses"):
    if _mod not in sys.modules:
        _stub = types.ModuleType(_mod)
        _stub.EXTRACT_FUNC_DICT = {}
        sys.modules[_mod] = _stub

# Stub the ProcessGroup attribute that yunchang.globals exposes
if hasattr(sys.modules.get("yunchang.globals", None), "PROCESS_GROUP") is False:
    _g = sys.modules["yunchang.globals"]

    class _PG:
        ULYSSES_PG = None
        RING_PG = None

    _g.PROCESS_GROUP = _PG
    _g.set_seq_parallel_pg = lambda *a, **k: None
# ---------------------------------------------------------------------------

import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def _load_specforge_draft_model(draft_model_path: str, device: torch.device):
    """Load SpecForge DFlashDraftModel from checkpoint."""
    spec_root = Path(__file__).resolve().parents[3]
    specforge_root = spec_root.parent / "specforge"

    # Insert specforge package on sys.path if not already present
    for p in (str(specforge_root), str(spec_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Import DFlashDraftModel directly from the module file to avoid the
    # specforge/__init__.py chain, which pulls in llama3_eagle.py → yunchang,
    # a dependency not always available in every venv.
    import importlib.util  # noqa: PLC0415
    _dflash_mod_path = specforge_root / "specforge" / "modeling" / "draft" / "dflash.py"
    _spec = importlib.util.spec_from_file_location("specforge.modeling.draft.dflash", _dflash_mod_path)
    _mod = importlib.util.module_from_spec(_spec)
    # Pre-register sub-package stubs so relative imports inside dflash.py resolve
    import types  # noqa: PLC0415
    for _pkg in ("specforge", "specforge.modeling", "specforge.modeling.draft"):
        if _pkg not in sys.modules:
            sys.modules[_pkg] = types.ModuleType(_pkg)
    sys.modules["specforge.modeling.draft.dflash"] = _mod
    _spec.loader.exec_module(_mod)
    DFlashDraftModel = _mod.DFlashDraftModel

    cfg_path = Path(draft_model_path) / "config.json"
    cfg = json.load(cfg_path.open())

    # Build a Qwen3Config-compatible object from the JSON.
    # The checkpoint config may have custom keys (block_size, dflash_config, …) and
    # a non-HF architectures value.  We filter out unknown init-params and then
    # manually restore the extra attrs via setattr so DFlashDraftModel can find them.
    from transformers import Qwen3Config  # noqa: PLC0415
    import inspect  # noqa: PLC0415
    _valid_qwen3_params = set(inspect.signature(Qwen3Config.__init__).parameters)
    _filtered_cfg = {
        k: v for k, v in cfg.items()
        if k in _valid_qwen3_params and k not in ("self", "kwargs")
    }
    config = Qwen3Config(**_filtered_cfg)
    # Restore custom keys as plain attributes so the draft model can access them
    for k, v in cfg.items():
        if not hasattr(config, k):
            setattr(config, k, v)

    # Ensure dflash_config is a plain dict on the config object
    dflash_cfg = cfg.get("dflash_config", {})
    config.dflash_config = dflash_cfg if isinstance(dflash_cfg, dict) else {}

    # Load model weights
    model = DFlashDraftModel(config).to(device=device, dtype=torch.bfloat16)
    model.eval()

    wt_path = Path(draft_model_path) / "model.safetensors"
    state = load_file(str(wt_path), device=str(device))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[specforge_probe] Missing keys (expected if no lm_head/embed): {missing}")
    if unexpected:
        print(f"[specforge_probe] Unexpected keys: {unexpected}")

    return model


def _load_target_lm_head_and_embed(shard_path: str, device: torch.device):
    """Load lm_head.weight and embed_tokens.weight from one safetensors shard."""
    print(f"[specforge_probe] Loading lm_head + embed_tokens from {shard_path} ...")
    shard = load_file(shard_path, device="cpu")
    lm_head_w = shard["lm_head.weight"].to(device=device, dtype=torch.bfloat16)
    embed_w   = shard.get("model.embed_tokens.weight")
    if embed_w is None:
        # Tied embeddings: lm_head == embed_tokens
        embed_w = lm_head_w
    else:
        embed_w = embed_w.to(device=device, dtype=torch.bfloat16)
    return lm_head_w, embed_w


@torch.inference_mode()
def run_specforge_draft_forward(
    draft_model,
    lm_head_weight: torch.Tensor,
    embed_weight: torch.Tensor,
    raw_target_hidden: torch.Tensor,   # [num_ctx, 6*H], float32 from vLLM probe
    noise_ids: torch.Tensor,            # [block_size], int64
    ctx_positions: torch.Tensor,        # [num_ctx], int64
    query_positions: torch.Tensor,      # [block_size], int64
    device: torch.device,
) -> dict:
    """Run one draft step and return intermediate tensors + final logits."""
    dtype = torch.bfloat16

    num_ctx   = raw_target_hidden.shape[0]
    block_size = noise_ids.shape[0]

    # Move inputs to device / dtype
    raw_target_hidden = raw_target_hidden.to(device=device, dtype=dtype)  # [num_ctx, 6*H]
    noise_ids         = noise_ids.to(device=device)
    ctx_positions     = ctx_positions.to(device=device)
    query_positions   = query_positions.to(device=device)

    # Build position_ids used by DFlashDraftModel: [1, num_ctx + block_size]
    all_positions = torch.cat([ctx_positions, query_positions], dim=0).unsqueeze(0)  # [1, L]

    # Add batch dim to target_hidden: [1, num_ctx, 6*H]
    target_hidden_batched = raw_target_hidden.unsqueeze(0)

    # Embed noise tokens: [1, block_size, H]
    noise_embedding = F.embedding(noise_ids, embed_weight).unsqueeze(0)

    # --- Intermediate: fc_output (same computation as vLLM combine_hidden_states) ---
    # DFlashDraftModel.forward() applies hidden_norm(fc(target_hidden)) internally.
    # We replicate those two steps here to capture both intermediate states.
    fc_output = F.linear(raw_target_hidden, draft_model.fc.weight)      # [num_ctx, H]
    fc_hidden_norm_output = draft_model.hidden_norm(fc_output)            # [num_ctx, H]

    # --- Run draft forward ---
    # NOTE: We do NOT pass is_causal=False here so the model uses its own
    # causal_head=True setting, matching vLLM's causal DFlash implementation.
    hidden_out = draft_model(
        position_ids=all_positions,
        noise_embedding=noise_embedding,
        target_hidden=target_hidden_batched,
        past_key_values=None,
        use_cache=False,
    )  # [1, block_size, H]

    # Take the last (block_size - 1) positions, matching SpecForge spec_generate's
    # `[:, -block_size + 1 :, :]` slice
    hidden_for_logits = hidden_out[:, -(block_size - 1):, :]  # [1, 15, H]

    # Compute logits using the target's lm_head (shared in both SpecForge and vLLM)
    draft_logits = F.linear(
        hidden_for_logits.float(),
        lm_head_weight.float()
    ).squeeze(0)  # [15, vocab_size]

    return {
        "fc_output":             fc_output.float().cpu(),          # [num_ctx, H]
        "fc_hidden_norm_output": fc_hidden_norm_output.float().cpu(),  # [num_ctx, H]
        "draft_logits":          draft_logits.float().cpu(),       # [15, vocab_size]
    }


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run SpecForge DFlash draft on vLLM-captured target hidden states"
    )
    p.add_argument(
        "--vllm-probe", required=True,
        help="Path to vllm_draft_probe.pt produced by probe_vllm_draft_logits.py"
    )
    p.add_argument(
        "--draft-model",
        default=(
            "/mnt/lanxiangh/checkpoints/specforge/"
            "ptd-step3p7-fkl-200k-epoch6-3e-4-no-gamma"
        ),
    )
    p.add_argument(
        "--target-lm-head-shard",
        default="/mnt/lanxiangh/models/Step-3.7-Flash/model-00024.safetensors",
        help="Safetensors shard containing lm_head.weight from the target model"
    )
    p.add_argument("--out", required=True, help="Path to save specforge_draft_probe.pt")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = _parse_args()
    device = torch.device(args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load vLLM probe data -------------------------------------------------
    print(f"[specforge_probe] Loading vLLM probe from {args.vllm_probe} ...")
    vllm_probe = torch.load(args.vllm_probe, weights_only=True)
    print("[specforge_probe] vLLM probe contents:")
    for k, v in sorted(vllm_probe.items()):
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {list(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {v}")

    raw_target_hidden = vllm_probe["raw_target_hidden_states"]  # [num_ctx, 6*H]
    query_input_ids   = vllm_probe["query_input_ids"]           # [block_size] or [block_size+1]
    query_positions   = vllm_probe["query_positions"]           # same length
    target_positions  = vllm_probe["target_positions"]          # [num_ctx]

    # Derive block_size from the query tensor length.
    # vLLM stores `num_query_per_req = 1 + num_speculative_tokens = block_size` query tokens.
    block_size = query_input_ids.shape[0]
    print(f"[specforge_probe] num_ctx={raw_target_hidden.shape[0]} block_size={block_size}")

    # ---- Load SpecForge draft model -------------------------------------------
    print(f"[specforge_probe] Loading DFlashDraftModel from {args.draft_model} ...")
    draft_model = _load_specforge_draft_model(args.draft_model, device)

    # ---- Load target lm_head + embed_tokens -----------------------------------
    lm_head_w, embed_w = _load_target_lm_head_and_embed(args.target_lm_head_shard, device)
    print(f"[specforge_probe] lm_head.weight shape={list(lm_head_w.shape)}")

    # ---- Run draft forward ----------------------------------------------------
    print("[specforge_probe] Running SpecForge draft forward ...")
    result = run_specforge_draft_forward(
        draft_model=draft_model,
        lm_head_weight=lm_head_w,
        embed_weight=embed_w,
        raw_target_hidden=raw_target_hidden,
        noise_ids=query_input_ids.long(),
        ctx_positions=target_positions.long(),
        query_positions=query_positions.long(),
        device=device,
    )

    # ---- Save -----------------------------------------------------------------
    save_dict = {
        **result,
        # Carry through the vLLM reference tensors for easy comparison
        "vllm_combined_target_hidden_states": (
            vllm_probe.get("combined_target_hidden_states")
        ),
        "vllm_draft_logits_req0": vllm_probe["draft_logits_req0"],
        "block_size": block_size,
        "raw_target_hidden_states": raw_target_hidden,
    }
    save_dict = {k: v for k, v in save_dict.items() if v is not None}

    torch.save(save_dict, str(out_path))
    print(f"[specforge_probe] Saved to {out_path}")
    print("[specforge_probe] Tensor shapes saved:")
    for k, v in sorted(save_dict.items()):
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {list(v.shape)}")

    # Quick self-check
    sf_logits   = result["draft_logits"]         # [15, V]
    vllm_logits = vllm_probe["draft_logits_req0"]  # [15, V]
    for depth in range(min(3, sf_logits.shape[0])):
        sf_top1   = int(sf_logits[depth].argmax())
        vllm_top1 = int(vllm_logits[depth].argmax())
        cos = F.cosine_similarity(sf_logits[depth].float().unsqueeze(0),
                                  vllm_logits[depth].float().unsqueeze(0)).item()
        print(f"[specforge_probe] depth={depth}  "
              f"sf_top1={sf_top1}  vllm_top1={vllm_top1}  "
              f"top1_match={sf_top1 == vllm_top1}  cos={cos:.6f}")


if __name__ == "__main__":
    main()
