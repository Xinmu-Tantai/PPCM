"""Audit helpers for the draft-quality diagnostic runner.

These helpers isolate the root-cause checklist items that distinguish
the HF reference draft path from the vLLM runtime draft path:

- Test A -- weight loading audit (looks for randomly initialized lm_head /
  embed_tokens / fc / hidden_norm and a missing draft_id_to_target_id).
- Test B -- lm_head parity on a pinned sample hidden state (target.lm_head
  vs draft.compute_logits + d2t scatter).
- Test C -- query embedding parity on a pinned list of token ids
  (target.model.embed_tokens vs draft.model.embed_tokens).
- Test D -- mask-token / rope-base / aux-layer-id consistency.
- Test E -- per-step top-k log divergence between the HF reference steps
  and the vLLM depth-0 steps aligned by token position.
- Test F -- target aux-hidden parity: compares the per-layer target hidden
  states captured at the last prompt token on the HF side with the raw
  pre-fc concat tensor captured inside the vLLM worker (via a
  forward_pre_hook on ``drafter.model.model.fc``), verifying both that the
  SAME 5 target layer outputs are selected AND that the numerical values
  agree layer-by-layer.
- Test G -- draft layer-by-layer parity at the d0 mask position on the very
  first draft step: compares HF's per-decoder-layer "full residual-stream"
  output at the d0 row against vLLM's per-decoder-layer output at the d0
  row (sum of the fused-residual tuple returned by vLLM decoder layers), and
  additionally the post-``norm`` hidden state that feeds the draft lm_head.
  This localizes where, inside the draft forward, the divergence first
  enters: layer 0 -> context-KV / embed / input mask; mid-layer k ->
  attention numerics at layer k; last-layer-only -> final norm / lm_head.
- Test H -- intra-layer bisect at d0 for the first diverging layer
  (default: layer 1). Captures six tap points (layer input, post-input-LN,
  self-attention output, post-attention residual, post-post-attention-LN,
  and MLP output) on HF and vLLM at the d0 row and reports per-tap parity.
  The first tap whose cosine drops below the threshold pinpoints whether
  the drift originates in input_layernorm, self_attn, post_attention_layernorm,
  or the MLP (H7b vs H7c sub-bisect).
- Test L -- multi-step layer-1 intra-self_attn probe: re-fires the Test I
  taps and the context-K/V capture at a configured set of speculative
  iterations (e.g. {0, 5, 15, 30}).  Tests F-K confirm step-0 parity is
  ≥0.9997; if the end-to-end gap is driven by a cross-iteration mechanism
  (e.g. vLLM retaining stale noise K/V in the paged cache, or context K/V
  not being refreshed each iteration), the per-step parity will drift as
  N grows on the vLLM side while HF stays flat.  Orthogonal to Tests
  A-K which are all step-0 snapshots.

All helpers return plain dicts that the runner serializes into JSON so the
audit can be inspected offline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _snapshot_attn_metadata(meta: Any) -> dict[str, Any]:
    """Best-effort structured snapshot of an attention metadata object.

    Extracts common vLLM v1 attn-metadata fields (``num_actual_tokens``,
    ``seq_lens``, ``query_start_loc``, ``slot_mapping``, ``block_table``,
    ``max_query_len``, ``max_seq_len``, causal/prefill flags) into plain
    python values so the caller can serialize them into JSON.
    """
    import torch  # noqa: PLC0415

    out: dict[str, Any] = {"type": type(meta).__name__}
    field_names = [
        "num_actual_tokens",
        "num_input_tokens",
        "max_query_len",
        "max_seq_len",
        "num_prefill_tokens",
        "num_decode_tokens",
        "num_prefills",
        "num_decodes",
        "is_causal",
        "prefill_metadata",
        "decode_metadata",
    ]
    for name in field_names:
        if not hasattr(meta, name):
            continue
        val = getattr(meta, name)
        if isinstance(val, torch.Tensor):
            out[name] = val.detach().cpu().tolist()
        elif val is None or isinstance(val, (int, float, bool, str)):
            out[name] = val
        else:
            out[name] = type(val).__name__
    tensor_fields = [
        "seq_lens",
        "query_start_loc",
        "slot_mapping",
        "block_table",
        "block_tables",
        "context_lens",
        "positions",
        "input_positions",
    ]
    for name in tensor_fields:
        if not hasattr(meta, name):
            continue
        val = getattr(meta, name)
        if isinstance(val, torch.Tensor):
            out[name] = {
                "shape": list(val.shape),
                "dtype": str(val.dtype),
                "values": val.detach().cpu().tolist()
                if val.numel() <= 4096
                else "<truncated>",
            }
    return out


def _stat_tensor_from_torch(t: Any) -> dict[str, Any]:
    import torch

    assert isinstance(t, torch.Tensor)
    ft = t.detach().float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "norm": float(ft.norm().item()),
        "abs_mean": float(ft.abs().mean().item()),
        "std": float(ft.std().item()) if ft.numel() > 1 else 0.0,
    }


def write_reference_capture(
    *,
    output_path: Path,
    target_lm_head_weight: Any,
    target_embed_tokens_weight: Any,
    sample_hidden_state: Any,
    query_ids: list[int],
    target_embed_of_query_ids: Any,
    draft_logprobs_on_sample_hidden: Any,
    topk_k: int,
    draft_mask_token_id: int,
    draft_target_layer_ids: list[int],
    draft_rope_theta: float | None,
    draft_vocab_size: int,
    target_vocab_size: int,
    num_prompt_tokens: int,
    target_per_layer_hidden_at_last_prompt: dict[int, Any],
    target_concat_hidden_at_last_prompt: Any,
    target_num_hidden_layers: int,
    draft_num_hidden_layers: int | None = None,
    draft_d0_row_index: int | None = None,
    draft_layer_outputs_at_d0: dict[int, Any] | None = None,
    draft_pre_norm_residual_at_d0: Any = None,
    draft_post_norm_at_d0: Any = None,
    draft_noise_embed_at_d0: Any = None,
    draft_layer1_bisect_at_d0: dict[str, Any] | None = None,
    draft_layer1_attn_bisect_at_d0: dict[str, Any] | None = None,
    draft_layer1_context_kv: dict[str, Any] | None = None,
    test_k_hf_sdpa_variants: dict[str, Any] | None = None,
    test_k_hf_sdpa_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dump the reference-side tensors + scalars needed for comparison.

    The reference models are torn down shortly after this call (to free GPU
    memory before the vLLM stage runs), so every tensor here must be CPU
    resident and serializable.
    """
    import torch

    top_lp, top_id = draft_logprobs_on_sample_hidden.topk(topk_k)
    payload: dict[str, Any] = {
        "target_lm_head_weight": _stat_tensor_from_torch(target_lm_head_weight),
        "target_embed_tokens_weight": _stat_tensor_from_torch(
            target_embed_tokens_weight
        ),
        "sample_hidden_state": sample_hidden_state.detach().cpu().float().tolist(),
        "sample_hidden_state_stats": _stat_tensor_from_torch(sample_hidden_state),
        "query_ids": [int(x) for x in query_ids],
        "target_embed_of_query_ids": (
            target_embed_of_query_ids.detach().cpu().float().tolist()
        ),
        "target_embed_of_query_ids_stats": _stat_tensor_from_torch(
            target_embed_of_query_ids
        ),
        "target_embed_of_query_ids_per_id_norm": (
            target_embed_of_query_ids.detach()
            .float()
            .norm(dim=-1)
            .cpu()
            .tolist()
        ),
        "lm_head_topk_tokens": [int(x) for x in top_id.tolist()],
        "lm_head_topk_logprobs": [float(x) for x in top_lp.tolist()],
        "draft_mask_token_id": int(draft_mask_token_id),
        "draft_target_layer_ids": [int(x) for x in draft_target_layer_ids],
        "draft_rope_theta": (
            None if draft_rope_theta is None else float(draft_rope_theta)
        ),
        "draft_vocab_size": int(draft_vocab_size),
        "target_vocab_size": int(target_vocab_size),
        "num_prompt_tokens": int(num_prompt_tokens),
        "target_num_hidden_layers": int(target_num_hidden_layers),
        "target_per_layer_hidden_at_last_prompt": {
            str(int(lid)): (
                t.detach().cpu().float().tolist()
                if hasattr(t, "detach")
                else list(t)
            )
            for lid, t in target_per_layer_hidden_at_last_prompt.items()
        },
        "target_per_layer_hidden_at_last_prompt_stats": {
            str(int(lid)): _stat_tensor_from_torch(t)
            for lid, t in target_per_layer_hidden_at_last_prompt.items()
            if hasattr(t, "detach")
        },
        "target_concat_hidden_at_last_prompt": (
            target_concat_hidden_at_last_prompt.detach().cpu().float().tolist()
            if hasattr(target_concat_hidden_at_last_prompt, "detach")
            else list(target_concat_hidden_at_last_prompt)
        ),
        "target_concat_hidden_at_last_prompt_stats": (
            _stat_tensor_from_torch(target_concat_hidden_at_last_prompt)
            if hasattr(target_concat_hidden_at_last_prompt, "detach")
            else None
        ),
    }

    if draft_num_hidden_layers is not None:
        payload["draft_num_hidden_layers"] = int(draft_num_hidden_layers)
    if draft_d0_row_index is not None:
        payload["draft_d0_row_index"] = int(draft_d0_row_index)

    def _to_list(vec: Any) -> list[float] | None:
        if vec is None:
            return None
        if hasattr(vec, "detach"):
            return vec.detach().cpu().float().tolist()
        return list(vec)

    if draft_layer_outputs_at_d0 is not None:
        payload["draft_layer_outputs_at_d0"] = {
            str(int(k)): _to_list(v) for k, v in draft_layer_outputs_at_d0.items()
        }
        payload["draft_layer_outputs_at_d0_stats"] = {
            str(int(k)): _stat_tensor_from_torch(v)
            for k, v in draft_layer_outputs_at_d0.items()
            if hasattr(v, "detach")
        }
    pre_norm_list = _to_list(draft_pre_norm_residual_at_d0)
    if pre_norm_list is not None:
        payload["draft_pre_norm_residual_at_d0"] = pre_norm_list
        if hasattr(draft_pre_norm_residual_at_d0, "detach"):
            payload["draft_pre_norm_residual_at_d0_stats"] = _stat_tensor_from_torch(
                draft_pre_norm_residual_at_d0
            )
    post_norm_list = _to_list(draft_post_norm_at_d0)
    if post_norm_list is not None:
        payload["draft_post_norm_at_d0"] = post_norm_list
        if hasattr(draft_post_norm_at_d0, "detach"):
            payload["draft_post_norm_at_d0_stats"] = _stat_tensor_from_torch(
                draft_post_norm_at_d0
            )
    noise_emb_list = _to_list(draft_noise_embed_at_d0)
    if noise_emb_list is not None:
        payload["draft_noise_embed_at_d0"] = noise_emb_list
        if hasattr(draft_noise_embed_at_d0, "detach"):
            payload["draft_noise_embed_at_d0_stats"] = _stat_tensor_from_torch(
                draft_noise_embed_at_d0
            )

    if draft_layer1_bisect_at_d0:
        bisect_payload: dict[str, Any] = {}
        bisect_stats: dict[str, Any] = {}
        for stage_name, tensor in draft_layer1_bisect_at_d0.items():
            if stage_name == "layer_idx":
                bisect_payload["layer_idx"] = int(tensor)
                continue
            vec = _to_list(tensor)
            if vec is not None:
                bisect_payload[stage_name] = vec
                if hasattr(tensor, "detach"):
                    bisect_stats[stage_name] = _stat_tensor_from_torch(tensor)
        if bisect_stats:
            bisect_payload["_stats"] = bisect_stats
        payload["draft_layer1_bisect_at_d0"] = bisect_payload

    # Test I-1: intra-self_attn taps (layer_idx + per-stage tensors).
    if draft_layer1_attn_bisect_at_d0:
        attn_payload: dict[str, Any] = {}
        attn_stats: dict[str, Any] = {}
        for stage_name, tensor in draft_layer1_attn_bisect_at_d0.items():
            if stage_name == "layer_idx":
                attn_payload["layer_idx"] = int(tensor)
                continue
            if not hasattr(tensor, "detach"):
                continue
            vec = _to_list(tensor)
            if vec is not None:
                attn_payload[stage_name] = vec
                if hasattr(tensor, "detach"):
                    attn_stats[stage_name] = _stat_tensor_from_torch(tensor)
        if attn_stats:
            attn_payload["_stats"] = attn_stats
        payload["draft_layer1_attn_bisect_at_d0"] = attn_payload

    # Test I-2: context K/V at the last context position for a chosen layer.
    if draft_layer1_context_kv:
        ctx_payload: dict[str, Any] = {}
        ctx_stats: dict[str, Any] = {}
        for k_name, tensor in draft_layer1_context_kv.items():
            if k_name in {"layer_idx", "last_context_index", "num_context"}:
                ctx_payload[k_name] = int(tensor)
                continue
            if not hasattr(tensor, "detach"):
                continue
            vec = _to_list(tensor)
            if vec is not None:
                ctx_payload[k_name] = vec
                if hasattr(tensor, "detach"):
                    ctx_stats[k_name] = _stat_tensor_from_torch(tensor)
        if ctx_stats:
            ctx_payload["_stats"] = ctx_stats
        payload["draft_layer1_context_kv"] = ctx_payload

    # Test K: HF-side SDPA backend-isolation variants.  Each entry is the
    # d0-row attention output produced under a specific torch SDPA backend /
    # dtype combination, using the SAME q/k/v/mask inputs that were fed to
    # the default HF kernel.  Purely reference-side; vLLM never computes
    # these variants.
    if test_k_hf_sdpa_variants:
        k_payload: dict[str, Any] = {}
        k_stats: dict[str, Any] = {}
        for variant_name, tensor in test_k_hf_sdpa_variants.items():
            if not hasattr(tensor, "detach"):
                continue
            vec = _to_list(tensor)
            if vec is not None:
                k_payload[variant_name] = vec
                k_stats[variant_name] = _stat_tensor_from_torch(tensor)
        if k_stats:
            k_payload["_stats"] = k_stats
        payload["test_k_hf_sdpa_variants"] = k_payload
    if test_k_hf_sdpa_metadata:
        # Metadata is already JSON-friendly (scalars / lists / dicts) because
        # it is assembled on the caller side.
        payload["test_k_hf_sdpa_metadata"] = test_k_hf_sdpa_metadata

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def collect_vllm_draft_audit(
    worker: Any,
    sample_hidden_state: list[float],
    query_ids: list[int],
    topk_k: int = 20,
) -> dict[str, Any]:
    """Runs inside the vLLM worker via ``llm.collective_rpc``.

    Must be top-level callable so it is picklable by ``collective_rpc``.
    Requires ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` on the launcher.
    """
    import torch

    worker_obj = getattr(worker, "worker", worker)
    model_runner = getattr(worker_obj, "model_runner", None)
    drafter = getattr(model_runner, "drafter", None)
    if drafter is None:
        return {"error": "no drafter on worker"}

    model = getattr(drafter, "model", None)
    if model is None:
        return {"error": "drafter has no .model"}

    def _stat(t):
        if t is None:
            return None
        ft = t.detach().float()
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "norm": float(ft.norm().item()),
            "abs_mean": float(ft.abs().mean().item()),
            "std": float(ft.std().item()) if t.numel() > 1 else 0.0,
        }

    lm_head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
    inner = getattr(model, "model", None)
    fc_weight = None
    hidden_norm_weight = None
    embed_tokens_weight = None
    if inner is not None:
        fc_weight = getattr(getattr(inner, "fc", None), "weight", None)
        hidden_norm_weight = getattr(
            getattr(inner, "hidden_norm", None), "weight", None
        )
        embed_tokens_weight = getattr(
            getattr(inner, "embed_tokens", None), "weight", None
        )

    weight_stats = {
        "lm_head.weight": _stat(lm_head_weight),
        "model.fc.weight": _stat(fc_weight),
        "model.hidden_norm.weight": _stat(hidden_norm_weight),
        "model.embed_tokens.weight": _stat(embed_tokens_weight),
    }

    d2t_tensor = getattr(model, "draft_id_to_target_id", None)
    d2t_info: dict[str, Any] = {"is_none": d2t_tensor is None}
    if d2t_tensor is not None:
        d2t_info.update(
            {
                "shape": list(d2t_tensor.shape),
                "dtype": str(d2t_tensor.dtype),
                "min": int(d2t_tensor.min().item()) if d2t_tensor.numel() else 0,
                "max": int(d2t_tensor.max().item()) if d2t_tensor.numel() else 0,
                "abs_mean": float(
                    d2t_tensor.detach().float().abs().mean().item()
                )
                if d2t_tensor.numel()
                else 0.0,
            }
        )

    rope_base: float | None = None
    try:
        rotary_emb = model.model.layers[0].self_attn.rotary_emb
        rope_base = float(getattr(rotary_emb, "base", None))
    except Exception:
        rope_base = None

    parallel_drafting_token_id: int | None = None
    try:
        parallel_drafting_token_id = int(drafter.parallel_drafting_token_id)
    except Exception:
        parallel_drafting_token_id = None

    aux_layer_ids = list(
        getattr(model.config, "eagle_aux_hidden_state_layer_ids", []) or []
    )

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    hs = torch.tensor(sample_hidden_state, device=device, dtype=dtype)
    if hs.dim() == 1:
        hs = hs.unsqueeze(0)

    logits_stats: dict[str, Any] | None = None
    topk_tokens: list[int] = []
    topk_lps: list[float] = []
    try:
        with torch.inference_mode():
            logits = model.compute_logits(hs)
        if logits is not None:
            logits_f = logits.float()
            logprobs = torch.log_softmax(logits_f[0], dim=-1)
            top_lp, top_id = logprobs.topk(topk_k)
            topk_tokens = [int(x) for x in top_id.tolist()]
            topk_lps = [float(x) for x in top_lp.tolist()]
            logits_stats = {
                "shape": list(logits_f.shape),
                "min": float(logits_f.min().item()),
                "max": float(logits_f.max().item()),
                "mean": float(logits_f.mean().item()),
            }
    except Exception as e:
        logits_stats = {"error": f"{type(e).__name__}: {e}"}

    q_emb_tokens: list[list[float]] = []
    q_emb_stats: dict[str, Any] = {}
    try:
        qids = torch.tensor(query_ids, device=device, dtype=torch.long)
        with torch.inference_mode():
            q_emb = model.model.embed_tokens(qids)
        q_emb_tokens = q_emb.detach().cpu().float().tolist()
        q_emb_stats = {
            "shape": list(q_emb.shape),
            "abs_mean": float(q_emb.detach().float().abs().mean().item()),
            "per_id_norm": q_emb.detach().float().norm(dim=-1).cpu().tolist(),
        }
    except Exception as e:
        q_emb_stats = {"error": f"{type(e).__name__}: {e}"}

    return {
        "weight_stats": weight_stats,
        "d2t": d2t_info,
        "rope_base": rope_base,
        "parallel_drafting_token_id": parallel_drafting_token_id,
        "aux_hidden_state_layer_ids": [int(x) for x in aux_layer_ids],
        "draft_vocab_size": int(
            getattr(model.config, "draft_vocab_size", model.config.vocab_size)
        ),
        "vocab_size": int(model.config.vocab_size),
        "sample_hidden_state_topk_tokens": topk_tokens,
        "sample_hidden_state_topk_logprobs": topk_lps,
        "sample_hidden_state_logits_stats": logits_stats,
        "query_embed_tokens": q_emb_tokens,
        "query_embed_stats": q_emb_stats,
    }


def install_vllm_target_aux_capture(
    worker: Any,
    num_prompt_tokens: int,
    num_calls_to_capture: int = 4,
) -> dict[str, Any]:
    """Installed via ``collective_rpc`` BEFORE generate().

    Registers a forward_pre_hook on ``drafter.model.model.fc`` that captures
    the raw pre-fc concat tensor ``[num_tokens, hidden_size * num_aux_layers]``
    for the first ``num_calls_to_capture`` calls. These tensors are saved on
    the worker for later retrieval.
    """
    import torch

    worker_obj = getattr(worker, "worker", worker)
    model_runner = getattr(worker_obj, "model_runner", None)
    if model_runner is None:
        return {"error": "no model_runner"}
    drafter = getattr(model_runner, "drafter", None)
    if drafter is None:
        return {"error": "no drafter"}
    draft_model = getattr(drafter, "model", None)
    if draft_model is None:
        return {"error": "no drafter.model"}
    draft_inner = getattr(draft_model, "model", None)
    if draft_inner is None:
        return {"error": "no drafter.model.model"}
    fc_module = getattr(draft_inner, "fc", None)
    if fc_module is None:
        return {"error": "no drafter.model.model.fc"}

    captures: list[dict[str, Any]] = []

    def pre_hook(_module, args):
        if len(captures) >= num_calls_to_capture:
            return None
        if not args:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor):
            return None
        entry: dict[str, Any] = {
            "call_index": len(captures),
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "num_tokens": int(x.shape[0]) if x.dim() >= 1 else 0,
            "feature_dim": int(x.shape[-1]) if x.dim() >= 1 else 0,
        }
        if x.dim() == 2:
            n = x.shape[0]
            last_prompt_row = num_prompt_tokens - 1
            if 0 <= last_prompt_row < n:
                entry["row_at_last_prompt"] = (
                    x[last_prompt_row].detach().cpu().float().tolist()
                )
            entry["row_at_last"] = x[-1].detach().cpu().float().tolist()
            try:
                xf = x.detach().float()
                entry["stats"] = {
                    "norm": float(xf.norm().item()),
                    "abs_mean": float(xf.abs().mean().item()),
                    "min": float(xf.min().item()),
                    "max": float(xf.max().item()),
                }
            except Exception:
                entry["stats"] = None
        captures.append(entry)
        return None

    handle = fc_module.register_forward_pre_hook(pre_hook)

    target_model = None
    try:
        target_model = model_runner.get_model()
    except Exception:
        target_model = getattr(model_runner, "model", None)
    installed_aux_layers: list[int] = []
    try:
        installed_aux_layers = [
            int(x) for x in target_model.model.aux_hidden_state_layers
        ]
    except Exception:
        installed_aux_layers = []

    dflash_target_layer_ids_0based: list[int] = []
    eagle_layer_ids_0based: list[int] = []
    try:
        dfc = getattr(draft_model.config, "dflash_config", {}) or {}
        if isinstance(dfc, dict):
            dflash_target_layer_ids_0based = [
                int(x) for x in dfc.get("target_layer_ids", []) or []
            ]
        egc = getattr(draft_model.config, "eagle_config", {}) or {}
        if isinstance(egc, dict):
            eagle_layer_ids_0based = [
                int(x) for x in egc.get("layer_ids", []) or []
            ]
    except Exception:
        pass

    target_hidden_size: int | None = None
    target_num_hidden_layers: int | None = None
    try:
        tcfg = target_model.config
        target_hidden_size = int(getattr(tcfg, "hidden_size", 0) or 0) or None
        target_num_hidden_layers = (
            int(getattr(tcfg, "num_hidden_layers", 0) or 0) or None
        )
    except Exception:
        pass

    worker_obj._draft_quality_target_aux_capture = {
        "captures": captures,
        "handle": handle,
        "installed_aux_layers": installed_aux_layers,
        "dflash_target_layer_ids_0based": dflash_target_layer_ids_0based,
        "eagle_layer_ids_0based": eagle_layer_ids_0based,
        "num_prompt_tokens": int(num_prompt_tokens),
        "target_hidden_size": target_hidden_size,
        "target_num_hidden_layers": target_num_hidden_layers,
    }
    return {
        "status": "installed",
        "installed_aux_layers": installed_aux_layers,
        "dflash_target_layer_ids_0based": dflash_target_layer_ids_0based,
        "eagle_layer_ids_0based": eagle_layer_ids_0based,
        "target_hidden_size": target_hidden_size,
        "target_num_hidden_layers": target_num_hidden_layers,
    }


def retrieve_vllm_target_aux_capture(worker: Any) -> dict[str, Any]:
    """Retrieve & uninstall the capture state installed by
    ``install_vllm_target_aux_capture``. Must be called from a post-generation
    ``collective_rpc`` so the hook sees at least one fc call."""
    worker_obj = getattr(worker, "worker", worker)
    state = getattr(worker_obj, "_draft_quality_target_aux_capture", None)
    if state is None:
        return {"error": "no capture state"}

    handle = state.get("handle")
    if handle is not None:
        try:
            handle.remove()
        except Exception:
            pass

    out = {
        "captures": state.get("captures", []),
        "installed_aux_layers": state.get("installed_aux_layers", []),
        "dflash_target_layer_ids_0based": state.get(
            "dflash_target_layer_ids_0based", []
        ),
        "eagle_layer_ids_0based": state.get("eagle_layer_ids_0based", []),
        "num_prompt_tokens": state.get("num_prompt_tokens"),
        "target_hidden_size": state.get("target_hidden_size"),
        "target_num_hidden_layers": state.get("target_num_hidden_layers"),
    }
    try:
        delattr(worker_obj, "_draft_quality_target_aux_capture")
    except Exception:
        pass
    return out


def _l2_diff(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def _max_abs_diff(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("inf")
    return max(abs(x - y) for x, y in zip(a, b))


def _mean_abs(a: list[float]) -> float:
    return (sum(abs(x) for x in a) / len(a)) if a else 0.0


def build_target_hidden_parity_report(
    reference: dict[str, Any],
    vllm_capture: dict[str, Any],
) -> dict[str, Any]:
    """Test F: verify both sides select identical target layers AND produce
    numerically identical aux-hidden vectors at the last prompt position.

    The report contains three nested sections:
    - ``layer_id_check``: is the same 5-layer set selected, and in the same
      concat order?
    - ``per_layer_parity``: per-layer cosine / L2 / max-abs between HF's
      per-layer hidden at the last prompt position and vLLM's deconcatenated
      slice of its first fc call's row at the last prompt position.
    - ``concat_parity``: whole-concat cosine / L2 / max-abs.
    """
    report: dict[str, Any] = {
        "layer_id_check": {},
        "per_layer_parity": {},
        "concat_parity": {},
        "vllm_num_fc_calls_captured": len(vllm_capture.get("captures", [])),
        "vllm_num_prompt_tokens": vllm_capture.get("num_prompt_tokens"),
        "ref_num_prompt_tokens": reference.get("num_prompt_tokens"),
    }

    ref_layer_ids = [int(x) for x in reference.get("draft_target_layer_ids", [])]
    vllm_dflash_ids_0based = [
        int(x) for x in vllm_capture.get("dflash_target_layer_ids_0based", [])
    ]
    vllm_installed_aux = [
        int(x) for x in vllm_capture.get("installed_aux_layers", [])
    ]

    expected_installed_from_ref = [lid + 1 for lid in ref_layer_ids]
    expected_installed_from_vllm_dflash = [lid + 1 for lid in vllm_dflash_ids_0based]

    ref_config_matches_vllm_config = (
        sorted(ref_layer_ids) == sorted(vllm_dflash_ids_0based)
    )
    ref_selects_same_target_outputs_as_vllm = (
        sorted(expected_installed_from_ref) == sorted(vllm_installed_aux)
    )
    vllm_shift_is_consistent = (
        sorted(expected_installed_from_vllm_dflash) == sorted(vllm_installed_aux)
    )
    concat_order_matches = (
        expected_installed_from_ref == vllm_installed_aux
    )

    report["layer_id_check"] = {
        "reference_draft_target_layer_ids_raw": ref_layer_ids,
        "vllm_dflash_target_layer_ids_0based": vllm_dflash_ids_0based,
        "vllm_installed_aux_hidden_state_layers": vllm_installed_aux,
        "expected_installed_from_reference_plus_1": expected_installed_from_ref,
        "expected_installed_from_vllm_dflash_plus_1": (
            expected_installed_from_vllm_dflash
        ),
        "ref_config_matches_vllm_config": ref_config_matches_vllm_config,
        "ref_selects_same_target_outputs_as_vllm": (
            ref_selects_same_target_outputs_as_vllm
        ),
        "vllm_shift_is_consistent": vllm_shift_is_consistent,
        "concat_order_matches_ascending": concat_order_matches,
    }

    ref_per_layer = reference.get("target_per_layer_hidden_at_last_prompt", {}) or {}
    ref_concat = list(
        reference.get("target_concat_hidden_at_last_prompt", []) or []
    )

    first_fc_call: dict[str, Any] | None = None
    ref_num_prompt = reference.get("num_prompt_tokens")
    for cap in vllm_capture.get("captures", []) or []:
        if (
            ref_num_prompt is not None
            and int(cap.get("num_tokens", -1)) == int(ref_num_prompt)
        ):
            first_fc_call = cap
            break
    if first_fc_call is None and vllm_capture.get("captures"):
        first_fc_call = vllm_capture["captures"][0]

    report["selected_vllm_fc_call"] = None
    if first_fc_call is not None:
        report["selected_vllm_fc_call"] = {
            "call_index": first_fc_call.get("call_index"),
            "num_tokens": first_fc_call.get("num_tokens"),
            "feature_dim": first_fc_call.get("feature_dim"),
            "matches_num_prompt_tokens": (
                int(first_fc_call.get("num_tokens", -1)) == int(ref_num_prompt or -1)
            ),
            "stats": first_fc_call.get("stats"),
        }

    vllm_concat = list((first_fc_call or {}).get("row_at_last_prompt", []) or [])

    target_hidden_size = vllm_capture.get("target_hidden_size")
    if target_hidden_size is None and ref_per_layer:
        any_vec = next(iter(ref_per_layer.values()))
        if isinstance(any_vec, list):
            target_hidden_size = len(any_vec)

    per_layer_parity: dict[str, Any] = {}
    if (
        vllm_concat
        and ref_per_layer
        and target_hidden_size
        and vllm_installed_aux
        and len(vllm_concat) == target_hidden_size * len(vllm_installed_aux)
    ):
        for slot_idx, vllm_layer_idx_plus1 in enumerate(vllm_installed_aux):
            ref_layer_id = vllm_layer_idx_plus1 - 1
            start = slot_idx * target_hidden_size
            vllm_slice = vllm_concat[start : start + target_hidden_size]
            ref_vec = ref_per_layer.get(str(ref_layer_id)) or ref_per_layer.get(
                ref_layer_id
            )
            if ref_vec is None:
                per_layer_parity[str(ref_layer_id)] = {
                    "status": "missing_in_reference",
                }
                continue
            ref_list = [float(x) for x in ref_vec]
            cos = _cosine(ref_list, vllm_slice)
            diff = [a - b for a, b in zip(ref_list, vllm_slice)]
            per_layer_parity[str(ref_layer_id)] = {
                "vllm_concat_slot": slot_idx,
                "vllm_layer_index_plus1": int(vllm_layer_idx_plus1),
                "cosine": float(cos),
                "l2_diff": _l2_diff(ref_list, vllm_slice),
                "max_abs_diff": _max_abs_diff(ref_list, vllm_slice),
                "mean_abs_diff": _mean_abs(diff),
                "ref_norm": math.sqrt(sum(x * x for x in ref_list)),
                "vllm_norm": math.sqrt(sum(x * x for x in vllm_slice)),
            }
    else:
        per_layer_parity["_skipped_reason"] = {
            "vllm_concat_len": len(vllm_concat),
            "ref_per_layer_keys": list(ref_per_layer.keys()),
            "target_hidden_size": target_hidden_size,
            "vllm_installed_aux_len": len(vllm_installed_aux),
        }
    report["per_layer_parity"] = per_layer_parity

    if ref_concat and vllm_concat and len(ref_concat) == len(vllm_concat):
        report["concat_parity"] = {
            "len": len(ref_concat),
            "cosine": float(_cosine(ref_concat, vllm_concat)),
            "l2_diff": _l2_diff(ref_concat, vllm_concat),
            "max_abs_diff": _max_abs_diff(ref_concat, vllm_concat),
            "mean_abs_diff": _mean_abs(
                [a - b for a, b in zip(ref_concat, vllm_concat)]
            ),
            "ref_norm": math.sqrt(sum(x * x for x in ref_concat)),
            "vllm_norm": math.sqrt(sum(x * x for x in vllm_concat)),
        }
    else:
        report["concat_parity"] = {
            "skipped": True,
            "ref_concat_len": len(ref_concat),
            "vllm_concat_len": len(vllm_concat),
        }

    return report


def build_weight_audit_report(audit: dict[str, Any]) -> dict[str, Any]:
    """Flag entries that look random / zero / missing.

    ``lm_head`` / ``fc`` / ``embed_tokens`` weights should have norms well
    above 1.0 in practice; ``hidden_norm`` is an RMSNorm affine weight
    initialized close to 1.0 so we only flag it if norm is truly tiny.
    """

    def _flag(name: str, stats: dict[str, Any] | None, min_norm: float) -> dict[str, Any]:
        if stats is None:
            return {"name": name, "status": "missing"}
        return {
            "name": name,
            "status": "present",
            "norm": stats["norm"],
            "abs_mean": stats["abs_mean"],
            "looks_random_or_zero": stats["norm"] < min_norm,
        }

    ws = audit.get("weight_stats", {})
    weights = [
        _flag("lm_head.weight", ws.get("lm_head.weight"), min_norm=1.0),
        _flag("model.fc.weight", ws.get("model.fc.weight"), min_norm=1.0),
        _flag(
            "model.hidden_norm.weight",
            ws.get("model.hidden_norm.weight"),
            min_norm=0.05,
        ),
        _flag(
            "model.embed_tokens.weight",
            ws.get("model.embed_tokens.weight"),
            min_norm=1.0,
        ),
    ]
    d2t_is_none = audit.get("d2t", {}).get("is_none", True)
    suspicious = [
        w["name"]
        for w in weights
        if w.get("looks_random_or_zero") or w.get("status") == "missing"
    ]
    if d2t_is_none:
        suspicious.append("draft_id_to_target_id")
    return {
        "weights": weights,
        "d2t_is_none": d2t_is_none,
        "d2t_stats": audit.get("d2t"),
        "aux_hidden_state_layer_ids": audit.get(
            "aux_hidden_state_layer_ids", []
        ),
        "draft_vocab_size": audit.get("draft_vocab_size"),
        "vocab_size": audit.get("vocab_size"),
        "suspicious_items": suspicious,
    }


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (na * nb)


def build_lm_head_embed_parity_report(
    reference: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    """Compare lm_head top-K and query embedding cosines across stacks."""
    ref_top = list(reference.get("lm_head_topk_tokens", []))
    vllm_top = list(audit.get("sample_hidden_state_topk_tokens", []))
    k = min(len(ref_top), len(vllm_top))
    overlap = sorted(set(ref_top[:k]) & set(vllm_top[:k]))

    ref_emb = reference.get("target_embed_of_query_ids") or []
    v_emb = audit.get("query_embed_tokens") or []
    cos_list: list[float] = []
    if ref_emb and v_emb and len(ref_emb) == len(v_emb):
        for a, b in zip(ref_emb, v_emb):
            cos_list.append(_cosine(a, b))

    ref_rope = reference.get("draft_rope_theta")
    v_rope = audit.get("rope_base")
    rope_match: bool | None
    if ref_rope is None or v_rope is None:
        rope_match = None
    else:
        rope_match = abs(ref_rope - v_rope) < 1e-3

    ref_aux = sorted([int(x) for x in reference.get("draft_target_layer_ids", [])])
    v_aux = sorted([int(x) for x in audit.get("aux_hidden_state_layer_ids", [])])

    return {
        "lm_head_top_k": k,
        "lm_head_topk_overlap": len(overlap),
        "lm_head_topk_overlap_ratio": (len(overlap) / k) if k else 0.0,
        "lm_head_ref_topk_tokens": ref_top,
        "lm_head_vllm_topk_tokens": vllm_top,
        "lm_head_ref_topk_logprobs": list(reference.get("lm_head_topk_logprobs", [])),
        "lm_head_vllm_topk_logprobs": list(
            audit.get("sample_hidden_state_topk_logprobs", [])
        ),
        "query_ids": list(reference.get("query_ids", [])),
        "query_embed_cosine_per_id": cos_list,
        "query_embed_cosine_mean": (sum(cos_list) / len(cos_list)) if cos_list else 0.0,
        "mask_token_id_match": (
            reference.get("draft_mask_token_id")
            == audit.get("parallel_drafting_token_id")
        ),
        "reference_mask_token_id": reference.get("draft_mask_token_id"),
        "vllm_parallel_drafting_token_id": audit.get("parallel_drafting_token_id"),
        "aux_layer_ids_match": ref_aux == v_aux,
        "reference_aux_layer_ids": ref_aux,
        "vllm_aux_layer_ids": v_aux,
        "rope_base_match": rope_match,
        "reference_draft_rope_theta": ref_rope,
        "vllm_rope_base": v_rope,
    }


def build_topk_log_divergence_report(
    target_forced_summary: dict[str, Any],
    vllm_summary: dict[str, Any],
) -> dict[str, Any]:
    """Align by token *position* and summarize agreement.

    The HF reference advances the target by exactly one token per step, so
    ``ref_step.position == num_prompt_tokens + step_idx``. vLLM's drafting
    rounds advance by ``accepted_len`` tokens each, so aligning by the raw
    drafting-round index is wrong. We instead reconstruct each vLLM step's
    position as ``num_prompt_tokens + sum_of_prior_accepted_len`` and match
    against the HF step at the same position.
    """
    ref_by_sample: dict[int, dict[str, Any]] = {}
    for s in target_forced_summary.get("samples", []):
        ref_by_sample[int(s["sample_index"])] = {
            "num_prompt_tokens": int(s.get("num_prompt_tokens", 0)),
            "steps_by_position": {
                int(st.get("position", -1)): st for st in s.get("steps", [])
            },
        }
    vllm_by_sample: dict[int, list[dict[str, Any]]] = {
        int(s["sample_index"]): list(s.get("steps", []))
        for s in vllm_summary.get("samples", [])
    }

    samples_report: list[dict[str, Any]] = []
    for sample_index in sorted(set(ref_by_sample) & set(vllm_by_sample)):
        ref_info = ref_by_sample[sample_index]
        num_prompt_tokens = ref_info["num_prompt_tokens"]
        ref_steps_by_position = ref_info["steps_by_position"]
        vllm_steps = sorted(
            vllm_by_sample[sample_index], key=lambda e: int(e.get("step", 0))
        )

        root_agree = 0
        top1_agree = 0
        topk_overlap_sum = 0
        topk_overlap_k = 0
        num_common = 0
        first_divergence: dict[str, Any] | None = None

        cumulative_accept = 0
        for v in vllm_steps:
            position = num_prompt_tokens + cumulative_accept
            r = ref_steps_by_position.get(position)
            if r is not None:
                num_common += 1
                root_match = r.get("root_token") == v.get("root_token")
                if root_match:
                    root_agree += 1
                ref_topk = list(r.get("topk_tok_0") or [])
                vllm_topk = list(v.get("topk_tok_0") or [])
                ref_t1 = ref_topk[0] if ref_topk else None
                vllm_t1 = vllm_topk[0] if vllm_topk else None
                t1_match = ref_t1 is not None and ref_t1 == vllm_t1
                if t1_match:
                    top1_agree += 1
                k = min(len(ref_topk), len(vllm_topk))
                if k > 0:
                    topk_overlap_sum += len(set(ref_topk[:k]) & set(vllm_topk[:k]))
                    topk_overlap_k += k
                if first_divergence is None and root_match and not t1_match:
                    first_divergence = {
                        "position": int(position),
                        "vllm_step": int(v.get("step", -1)),
                        "ref_step": int(r.get("step", -1)),
                        "root_token": r.get("root_token"),
                        "ref_top1": ref_t1,
                        "vllm_top1": vllm_t1,
                        "ref_topk": ref_topk,
                        "vllm_topk": vllm_topk,
                        "ref_topk_logprobs": list(r.get("topk_lp_0") or []),
                        "vllm_topk_logprobs": list(v.get("topk_lp_0") or []),
                    }
            # Spec-decode advance per round is ``accepted_len`` accepted
            # draft tokens + 1 bonus token from the target (full-accept
            # bonus, or the correction token when the first draft is wrong).
            try:
                cumulative_accept += int(v.get("accepted_len", 0)) + 1
            except (TypeError, ValueError):
                cumulative_accept += 1

        samples_report.append(
            {
                "sample_index": int(sample_index),
                "num_common_steps": num_common,
                "num_vllm_steps": len(vllm_steps),
                "num_ref_steps": len(ref_steps_by_position),
                "num_prompt_tokens": num_prompt_tokens,
                "root_token_agreement_rate": (
                    (root_agree / num_common) if num_common else 0.0
                ),
                "top1_agreement_rate": (
                    (top1_agree / num_common) if num_common else 0.0
                ),
                "topk_overlap_rate": (
                    (topk_overlap_sum / topk_overlap_k) if topk_overlap_k else 0.0
                ),
                "first_top1_divergence_with_same_root": first_divergence,
            }
        )

    aggregate_common = sum(s["num_common_steps"] for s in samples_report)
    aggregate_root = (
        sum(
            s["root_token_agreement_rate"] * s["num_common_steps"]
            for s in samples_report
        )
        / aggregate_common
        if aggregate_common
        else 0.0
    )
    aggregate_top1 = (
        sum(
            s["top1_agreement_rate"] * s["num_common_steps"]
            for s in samples_report
        )
        / aggregate_common
        if aggregate_common
        else 0.0
    )
    aggregate_topk = (
        sum(
            s["topk_overlap_rate"] * s["num_common_steps"]
            for s in samples_report
        )
        / aggregate_common
        if aggregate_common
        else 0.0
    )
    return {
        "per_sample": samples_report,
        "aggregate": {
            "num_common_steps": aggregate_common,
            "root_token_agreement_rate": aggregate_root,
            "top1_agreement_rate": aggregate_top1,
            "topk_overlap_rate": aggregate_topk,
        },
    }


def build_test_n_per_iteration_report(
    target_forced_summary: dict[str, Any],
    vllm_tree_summary: dict[str, Any],
    vllm_chain_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Test N: per-iteration draft-quality histograms + trajectory alignment.

    The pre-existing aggregated numbers (HF 86%, vLLM tree 44%) conflate two
    distinct effects:

      (i)  depth-0 draft quality at each sample's current position, and
      (ii) the trajectory the sample has taken to reach that position.

    vLLM's speculative-tree decoding and HF's chain-greedy decoding diverge
    the moment the first accepted token differs, after which they are
    scoring draft quality on *different prefixes*.  This report provides
    three slices to separate the two effects:

      - ``per_step_histogram_hf`` and ``per_step_histogram_vllm_tree``
        (and, when available, ``per_step_histogram_vllm_chain``): top-1
        hit-rate as a function of the 0-based iteration index, averaged
        across samples that reached that iteration.  Flat == healthy;
        monotonically-degrading curve == cumulative state corruption.

      - ``position_aligned``: for every (position, root_token) pair where
        HF and vLLM's current trajectory *coincide* (same accepted prefix),
        record whether each implementation's depth-0 draft matched its
        *own* next target token, and whether the two drafts agreed.  This
        is the cleanest like-for-like draft-quality comparison we can get
        without forcing identical input trajectories.

      - ``trajectory_divergence``: the step index at which each sample's
        HF and vLLM accepted trajectories first diverge, plus the
        position of the divergence.  Confirms/refutes the hypothesis that
        vLLM's aggregate gap is dominated by on-policy prefix drift.

    Also aggregates a chain-spec vLLM run (``vllm_chain_summary``) when
    provided; that run forces ``tree_width=1`` / ``max_tree_budget=1`` so
    vLLM advances exactly one token per iteration, making its per-step
    trajectory semantically identical to HF's and any remaining gap
    attributable to the kernel stack rather than the tree machinery.
    """

    def _sample_steps(summary: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
        return {
            int(s["sample_index"]): list(s.get("steps", []))
            for s in (summary.get("samples") or [])
        }

    def _num_prompt_tokens(summary: dict[str, Any]) -> dict[int, int]:
        return {
            int(s["sample_index"]): int(s.get("num_prompt_tokens", 0))
            for s in (summary.get("samples") or [])
        }

    hf_steps = _sample_steps(target_forced_summary)
    vllm_tree_steps = _sample_steps(vllm_tree_summary)
    vllm_chain_steps = (
        _sample_steps(vllm_chain_summary)
        if vllm_chain_summary is not None
        else {}
    )
    hf_num_prompt = _num_prompt_tokens(target_forced_summary)

    def _per_step_histogram(
        steps_by_sample: dict[int, list[dict[str, Any]]],
        max_steps: int | None = None,
    ) -> list[dict[str, Any]]:
        if not steps_by_sample:
            return []
        longest = max(len(steps) for steps in steps_by_sample.values())
        if max_steps is not None:
            longest = min(longest, max_steps)
        histogram: list[dict[str, Any]] = []
        for step_idx in range(longest):
            hits = 0
            total = 0
            for steps in steps_by_sample.values():
                if step_idx >= len(steps):
                    continue
                total += 1
                if bool(steps[step_idx].get("draft_top1_match")):
                    hits += 1
            histogram.append(
                {
                    "step": step_idx,
                    "num_samples_at_step": total,
                    "top1_match_rate": (hits / total) if total else 0.0,
                    "num_top1_matches": hits,
                }
            )
        return histogram

    hf_hist = _per_step_histogram(hf_steps)
    vllm_tree_hist = _per_step_histogram(vllm_tree_steps)
    vllm_chain_hist = (
        _per_step_histogram(vllm_chain_steps) if vllm_chain_steps else None
    )

    def _histogram_summary(h: list[dict[str, Any]]) -> dict[str, Any]:
        if not h:
            return {"num_bins": 0, "mean_rate": 0.0}
        rates = [bin_["top1_match_rate"] for bin_ in h]
        n = len(rates)
        head = rates[: max(1, n // 4)]
        tail = rates[-max(1, n // 4):]
        return {
            "num_bins": n,
            "mean_rate": sum(rates) / n,
            "first_quartile_mean_rate": sum(head) / len(head),
            "last_quartile_mean_rate": sum(tail) / len(tail),
            "step0_rate": rates[0],
            "step_last_rate": rates[-1],
        }

    hf_hist_summary = _histogram_summary(hf_hist)
    vllm_tree_hist_summary = _histogram_summary(vllm_tree_hist)
    vllm_chain_hist_summary = (
        _histogram_summary(vllm_chain_hist) if vllm_chain_hist else None
    )

    # ------------------------------------------------------------------
    # Position-aligned comparison (same-root-token, same-prefix probe).
    # ------------------------------------------------------------------
    per_sample_pos: list[dict[str, Any]] = []
    for sample_index in sorted(set(hf_steps) & set(vllm_tree_steps)):
        num_prompt = hf_num_prompt.get(sample_index, 0)

        hf_by_position: dict[int, dict[str, Any]] = {
            int(st.get("position", -1)): st for st in hf_steps[sample_index]
        }

        vllm_by_position: dict[int, dict[str, Any]] = {}
        cumulative = 0
        for v in sorted(
            vllm_tree_steps[sample_index], key=lambda e: int(e.get("step", 0))
        ):
            position = num_prompt + cumulative
            vllm_by_position[position] = v
            try:
                cumulative += int(v.get("accepted_len", 0)) + 1
            except (TypeError, ValueError):
                cumulative += 1

        same_position_count = 0
        same_root_count = 0
        hf_own_match_when_same_root = 0
        vllm_own_match_when_same_root = 0
        top1_agreement_when_same_root = 0

        shared_positions = sorted(set(hf_by_position) & set(vllm_by_position))
        for pos in shared_positions:
            same_position_count += 1
            h = hf_by_position[pos]
            v = vllm_by_position[pos]
            if h.get("root_token") != v.get("root_token"):
                continue
            same_root_count += 1
            if bool(h.get("draft_top1_match")):
                hf_own_match_when_same_root += 1
            if bool(v.get("draft_top1_match")):
                vllm_own_match_when_same_root += 1
            h_top1 = h.get("draft_top1_token")
            v_top1 = v.get("draft_top1_token")
            if h_top1 is not None and h_top1 == v_top1:
                top1_agreement_when_same_root += 1

        per_sample_pos.append(
            {
                "sample_index": sample_index,
                "num_shared_positions": same_position_count,
                "num_same_root_positions": same_root_count,
                "hf_top1_match_rate_same_root": (
                    hf_own_match_when_same_root / same_root_count
                    if same_root_count
                    else 0.0
                ),
                "vllm_top1_match_rate_same_root": (
                    vllm_own_match_when_same_root / same_root_count
                    if same_root_count
                    else 0.0
                ),
                "top1_agreement_rate_same_root": (
                    top1_agreement_when_same_root / same_root_count
                    if same_root_count
                    else 0.0
                ),
            }
        )

    def _weighted_rate(field: str, weight_field: str) -> float:
        total = sum(s[weight_field] for s in per_sample_pos)
        if not total:
            return 0.0
        return (
            sum(s[field] * s[weight_field] for s in per_sample_pos) / total
        )

    position_aligned = {
        "per_sample": per_sample_pos,
        "aggregate": {
            "num_same_root_positions": sum(
                s["num_same_root_positions"] for s in per_sample_pos
            ),
            "num_shared_positions": sum(
                s["num_shared_positions"] for s in per_sample_pos
            ),
            "hf_top1_match_rate_same_root": _weighted_rate(
                "hf_top1_match_rate_same_root", "num_same_root_positions"
            ),
            "vllm_top1_match_rate_same_root": _weighted_rate(
                "vllm_top1_match_rate_same_root", "num_same_root_positions"
            ),
            "top1_agreement_rate_same_root": _weighted_rate(
                "top1_agreement_rate_same_root", "num_same_root_positions"
            ),
        },
    }

    # ------------------------------------------------------------------
    # Trajectory divergence: where do HF and vLLM first disagree?
    # ------------------------------------------------------------------
    traj: list[dict[str, Any]] = []
    for sample_index in sorted(set(hf_steps) & set(vllm_tree_steps)):
        num_prompt = hf_num_prompt.get(sample_index, 0)
        hf_sorted = sorted(
            hf_steps[sample_index], key=lambda e: int(e.get("step", 0))
        )
        vllm_sorted = sorted(
            vllm_tree_steps[sample_index], key=lambda e: int(e.get("step", 0))
        )
        first_divergence_step = None
        first_divergence_position = None
        hf_root_at_divergence = None
        vllm_root_at_divergence = None

        cumulative = 0
        vllm_positions: list[tuple[int, dict[str, Any]]] = []
        for v in vllm_sorted:
            vllm_positions.append((num_prompt + cumulative, v))
            try:
                cumulative += int(v.get("accepted_len", 0)) + 1
            except (TypeError, ValueError):
                cumulative += 1

        max_iters = min(len(hf_sorted), len(vllm_positions))
        for step_idx in range(max_iters):
            h = hf_sorted[step_idx]
            position, v = vllm_positions[step_idx]
            if h.get("position") != position:
                first_divergence_step = step_idx
                first_divergence_position = int(position)
                hf_root_at_divergence = h.get("root_token")
                vllm_root_at_divergence = v.get("root_token")
                break
            if h.get("root_token") != v.get("root_token"):
                first_divergence_step = step_idx
                first_divergence_position = int(position)
                hf_root_at_divergence = h.get("root_token")
                vllm_root_at_divergence = v.get("root_token")
                break

        traj.append(
            {
                "sample_index": sample_index,
                "num_hf_steps": len(hf_sorted),
                "num_vllm_steps": len(vllm_sorted),
                "first_divergence_step": first_divergence_step,
                "first_divergence_position": first_divergence_position,
                "hf_root_token_at_divergence": hf_root_at_divergence,
                "vllm_root_token_at_divergence": vllm_root_at_divergence,
            }
        )

    # ------------------------------------------------------------------
    # Verdict / interpretation hint.
    # ------------------------------------------------------------------
    agg_hf_match = position_aligned["aggregate"]["hf_top1_match_rate_same_root"]
    agg_vllm_match = position_aligned["aggregate"][
        "vllm_top1_match_rate_same_root"
    ]
    verdict_lines: list[str] = []
    verdict_lines.append(
        f"HF aggregate top1: {target_forced_summary.get('top1_hit_rate'):.4f}; "
        f"vLLM tree aggregate top1: {vllm_tree_summary.get('top1_hit_rate'):.4f}."
    )
    if vllm_chain_summary is not None:
        verdict_lines.append(
            f"vLLM chain-spec aggregate top1: "
            f"{vllm_chain_summary.get('top1_hit_rate'):.4f}."
        )
    verdict_lines.append(
        f"Same-root positions examined: "
        f"{position_aligned['aggregate']['num_same_root_positions']}.  "
        f"HF top1-match-rate on shared-prefix positions: "
        f"{agg_hf_match:.4f};  vLLM top1-match-rate on same positions: "
        f"{agg_vllm_match:.4f}."
    )
    if (
        position_aligned["aggregate"]["num_same_root_positions"] >= 8
        and agg_hf_match > 0.0
    ):
        same_root_gap = agg_hf_match - agg_vllm_match
        if same_root_gap > 0.1:
            verdict_lines.append(
                "VERDICT: On shared-prefix positions, vLLM still trails HF "
                f"by {same_root_gap:.3f} -> genuine draft-kernel / state "
                "divergence, not just trajectory drift."
            )
        elif same_root_gap < 0.05:
            verdict_lines.append(
                "VERDICT: Draft quality on shared-prefix positions matches "
                "HF within tolerance; the aggregate gap is dominated by "
                "trajectory divergence (tree-spec explores a different "
                "on-policy prefix than HF's chain-greedy)."
            )
        else:
            verdict_lines.append(
                "VERDICT: Shared-prefix gap is moderate "
                f"(~{same_root_gap:.3f}); mixed contribution from "
                "trajectory and kernel."
            )
    else:
        verdict_lines.append(
            "VERDICT: Too few shared-prefix positions for a reliable "
            "kernel-vs-trajectory attribution."
        )

    if vllm_chain_summary is not None:
        chain_top1 = float(vllm_chain_summary.get("top1_hit_rate") or 0.0)
        hf_top1 = float(target_forced_summary.get("top1_hit_rate") or 0.0)
        chain_gap = hf_top1 - chain_top1
        if chain_gap > 0.1:
            verdict_lines.append(
                f"Chain-spec gap HF-vs-vLLM: {chain_gap:+.3f} (>0.10) -> "
                "even with identical chain-greedy semantics, vLLM's draft "
                "stack underperforms; issue is structural (kernel/cache/"
                "cross-attn plumbing), not tree-specific."
            )
        elif chain_gap < 0.03:
            verdict_lines.append(
                f"Chain-spec gap HF-vs-vLLM: {chain_gap:+.3f} (<0.03) -> "
                "vLLM matches HF in chain-greedy mode; the full "
                "tree-spec gap is attributable to speculative-tree "
                "semantics (context K/V pinning, tree masking, "
                "multi-draft-pass corruption, or branch selection)."
            )
        else:
            verdict_lines.append(
                f"Chain-spec gap HF-vs-vLLM: {chain_gap:+.3f} -> partial "
                "match; suggests a mixture of structural and tree-induced "
                "effects."
            )

    report = {
        "source": "test_n_per_iteration_and_trajectory",
        "aggregate_top1": {
            "hf": target_forced_summary.get("top1_hit_rate"),
            "vllm_tree": vllm_tree_summary.get("top1_hit_rate"),
            "vllm_chain": (
                vllm_chain_summary.get("top1_hit_rate")
                if vllm_chain_summary is not None
                else None
            ),
        },
        "per_step_histogram_hf": hf_hist,
        "per_step_histogram_vllm_tree": vllm_tree_hist,
        "per_step_histogram_vllm_chain": vllm_chain_hist,
        "histogram_summary_hf": hf_hist_summary,
        "histogram_summary_vllm_tree": vllm_tree_hist_summary,
        "histogram_summary_vllm_chain": vllm_chain_hist_summary,
        "position_aligned": position_aligned,
        "trajectory_divergence": traj,
        "verdict": verdict_lines,
    }
    return report


def install_vllm_chain_spec_topk_log_probe(
    worker: Any,
    tree_width_for_topk: int = 8,
) -> dict[str, Any]:
    """Make the chain-spec (tree_width=1) path populate ``_topk_log``.

    ``DFlashProposer.propose`` routes ``tree_width > 1`` batches through its
    tree-drafting code, which explicitly pushes one entry per request into
    ``self._topk_log`` (and queues its index in
    ``self._pending_topk_log_indices`` so the verifier-side
    ``record_topk_verify_outcome`` later fills in
    ``target_next_token`` / ``draft_top1_match`` / ``accepted_len``).
    When ``tree_width == 1`` the proposer short-circuits through
    ``SpecDecodeBaseProposer.propose`` -> ``_greedy_sample`` (the early-exit
    branch at eagle.py:484-486), which never touches ``_topk_log``.  That is
    why chain-spec runs produce an empty ``topk_log.json``.

    This probe monkey-patches two methods on the drafter:

      - ``propose``: wraps the outer call so we can stash ``next_token_ids``
        for the current speculative iteration.  ``next_token_ids`` is the
        already-accepted ``root_token`` for the round (identical to what the
        tree path records at dflash.py:696/703).
      - ``_greedy_sample``: after the drafter's own forward pass produces the
        depth-0 sample hidden states, recompute ``compute_logits`` to extract
        a (top-k) slice that mirrors the tree path's format, then push one
        entry per request into ``_topk_log`` and queue its index in
        ``_pending_topk_log_indices``.

    The verifier-side hook in ``gpu_model_runner.py`` already calls
    ``record_topk_verify_outcome`` for chain (non-tree) requests (see the
    ``CPU fallback for chain (non-tree) requests`` block at
    gpu_model_runner.py:3858-3876), so once we push into the pending queue,
    production fills in the rest of each entry automatically.

    Returns a small status dict describing what was installed.  The probe is
    idempotent: if it was already installed on this drafter, the second
    install is a no-op.  Call ``uninstall_vllm_chain_spec_topk_log_probe``
    to undo.
    """
    worker_obj = getattr(worker, "worker", worker)
    model_runner = getattr(worker_obj, "model_runner", None)
    drafter = getattr(model_runner, "drafter", None)
    if drafter is None:
        return {"installed": False, "reason": "no_drafter"}
    if not hasattr(drafter, "_topk_log") or not hasattr(
        drafter, "_pending_topk_log_indices"
    ):
        return {"installed": False, "reason": "drafter_missing_topk_state"}
    if getattr(drafter, "_chain_spec_probe_installed", False):
        return {"installed": True, "reason": "already_installed"}

    import torch

    drafter._chain_spec_step = 0
    drafter._chain_spec_pending_roots: list[int | None] = []

    orig_propose = drafter.propose
    orig_greedy = drafter._greedy_sample

    def patched_propose(*args: Any, **kwargs: Any) -> Any:
        ntids = kwargs.get("next_token_ids")
        if ntids is None:
            # positional signature (see eagle.py:405-422):
            # (target_token_ids, target_positions, target_hidden_states,
            #  next_token_ids, ...).  next_token_ids is the 4th positional.
            if len(args) >= 4:
                ntids = args[3]
        if isinstance(ntids, torch.Tensor):
            drafter._chain_spec_pending_roots = [
                int(x) for x in ntids.detach().cpu().tolist()
            ]
        else:
            drafter._chain_spec_pending_roots = []
        try:
            return orig_propose(*args, **kwargs)
        finally:
            drafter._chain_spec_pending_roots = []

    def patched_greedy(hidden_states: torch.Tensor) -> torch.Tensor:
        try:
            logits = drafter.model.compute_logits(hidden_states)
            lp = torch.log_softmax(logits.float(), dim=-1)
            k = int(min(tree_width_for_topk, lp.shape[-1]))
            topk_lp, topk_tok = lp.topk(k, dim=-1)
            batch_size = int(lp.shape[0])
            roots = list(drafter._chain_spec_pending_roots)
            # Pad / trim to match row count so we never crash on shape
            # mismatch (defensive: batch_size should equal len(roots)).
            while len(roots) < batch_size:
                roots.append(None)
            for req_idx in range(batch_size):
                drafter._pending_topk_log_indices.append(
                    len(drafter._topk_log)
                )
                drafter._topk_log.append(
                    {
                        "step": int(drafter._chain_spec_step),
                        "req": int(req_idx),
                        "root_token": roots[req_idx],
                        "topk_tok_0": [int(t) for t in topk_tok[req_idx].tolist()],
                        "topk_lp_0": [float(l) for l in topk_lp[req_idx].tolist()],
                        "source": "chain_spec_probe",
                    }
                )
            drafter._chain_spec_step += 1
            return logits.argmax(dim=-1)
        except Exception:
            import traceback
            traceback.print_exc()
            return orig_greedy(hidden_states)

    drafter.propose = patched_propose  # type: ignore[method-assign]
    drafter._greedy_sample = patched_greedy  # type: ignore[method-assign]
    drafter._chain_spec_probe_installed = True
    drafter._chain_spec_orig_propose = orig_propose
    drafter._chain_spec_orig_greedy = orig_greedy

    # Second patch: when tree_width=1, ``uses_tree_drafting`` returns
    # False, which makes the scheduler emit an empty
    # ``scheduled_spec_decode_tree_metadata``.  That in turn causes the
    # gpu_model_runner to pick ``SpecDecodeMetadata`` (not
    # ``DFlashTreeSpecDecodeMetadata``) and route the verify step through
    # ``self.rejection_sampler(...)`` instead of ``_sample_dflash_tree``.
    # Only ``_sample_dflash_tree`` calls ``record_topk_verify_outcome``;
    # the generic rejection path does not.  So we additionally monkey-
    # patch ``rejection_sampler.forward`` to replay the same bookkeeping
    # call after the real forward returns, using target greedy computed
    # from the target logits at the drafted positions.  Without this, all
    # chain-spec topk_log entries remain unfilled (no target_next_token,
    # so the diagnostic summary filters them out and the chain-spec
    # top1_hit_rate reports 0.0).
    rejection_sampler = getattr(model_runner, "rejection_sampler", None)
    drafter._chain_spec_rs_installed = False
    drafter._chain_spec_rs_obj = None
    drafter._chain_spec_orig_rs_forward = None
    if rejection_sampler is not None and hasattr(
        rejection_sampler, "forward"
    ):
        orig_rs_forward = rejection_sampler.forward

        def patched_rs_forward(
            metadata: Any,
            draft_probs: Any,
            logits: torch.Tensor,
            sampling_metadata: Any,
            *,
            _orig=orig_rs_forward,
            _drafter=drafter,
        ) -> Any:
            # Snapshot target greedy at drafted positions BEFORE the
            # sampler runs (the original implementation may mutate
            # logits in place).
            t_greedy_flat: torch.Tensor | None = None
            try:
                t_logits_idx = getattr(
                    metadata, "target_logits_indices", None
                )
                if t_logits_idx is not None and logits is not None:
                    t_greedy_flat = (
                        logits[t_logits_idx].argmax(dim=-1).detach().cpu()
                    )
            except Exception:
                t_greedy_flat = None

            output = _orig(metadata, draft_probs, logits, sampling_metadata)

            # Replay the bookkeeping call that _sample_dflash_tree does.
            try:
                num_draft_tokens = getattr(
                    metadata, "num_draft_tokens", None
                )
                draft_token_ids_all = getattr(
                    metadata, "draft_token_ids", None
                )
                if (
                    t_greedy_flat is not None
                    and num_draft_tokens is not None
                    and draft_token_ids_all is not None
                    and hasattr(_drafter, "record_topk_verify_outcome")
                ):
                    if isinstance(draft_token_ids_all, torch.Tensor):
                        draft_cpu = (
                            draft_token_ids_all.detach().cpu().tolist()
                        )
                    else:
                        draft_cpu = list(draft_token_ids_all)
                    greedy_list = [int(x) for x in t_greedy_flat.tolist()]
                    cursor = 0
                    for n_draft in num_draft_tokens:
                        n = int(n_draft)
                        if n <= 0:
                            continue
                        req_greedy = greedy_list[cursor : cursor + n]
                        req_draft = [
                            int(x) for x in draft_cpu[cursor : cursor + n]
                        ]
                        accepted = 0
                        while (
                            accepted < n
                            and req_draft[accepted] == req_greedy[accepted]
                        ):
                            accepted += 1
                        correction: int | None = (
                            int(req_greedy[accepted]) if accepted < n else None
                        )
                        try:
                            _drafter.record_topk_verify_outcome(
                                verify_greedy_tokens=req_greedy,
                                accepted_len=int(accepted),
                                correction_token=correction,
                                tree_num_nodes=int(n + 1),
                            )
                        except Exception:
                            import traceback
                            traceback.print_exc()
                        cursor += n
            except Exception:
                import traceback
                traceback.print_exc()
            return output

        rejection_sampler.forward = patched_rs_forward  # type: ignore[method-assign]
        drafter._chain_spec_rs_installed = True
        drafter._chain_spec_rs_obj = rejection_sampler
        drafter._chain_spec_orig_rs_forward = orig_rs_forward

    return {
        "installed": True,
        "tree_width_for_topk": int(tree_width_for_topk),
        "rs_forward_patched": bool(drafter._chain_spec_rs_installed),
    }


def uninstall_vllm_chain_spec_topk_log_probe(worker: Any) -> dict[str, Any]:
    """Restore the original ``propose`` and ``_greedy_sample`` methods.

    Safe to call even if the probe was never installed.  Returns the count
    of topk_log entries accumulated while the probe was active so the caller
    can sanity-check that the chain-spec run actually produced samples.
    """
    worker_obj = getattr(worker, "worker", worker)
    model_runner = getattr(worker_obj, "model_runner", None)
    drafter = getattr(model_runner, "drafter", None)
    if drafter is None:
        return {"uninstalled": False, "reason": "no_drafter"}
    if not getattr(drafter, "_chain_spec_probe_installed", False):
        return {"uninstalled": False, "reason": "not_installed"}
    try:
        drafter.propose = drafter._chain_spec_orig_propose
        drafter._greedy_sample = drafter._chain_spec_orig_greedy
    except Exception as e:
        return {"uninstalled": False, "reason": f"restore_failed: {e}"}
    if getattr(drafter, "_chain_spec_rs_installed", False):
        try:
            rs_obj = drafter._chain_spec_rs_obj
            if rs_obj is not None:
                rs_obj.forward = drafter._chain_spec_orig_rs_forward
        except Exception:
            pass
        drafter._chain_spec_rs_installed = False
    drafter._chain_spec_probe_installed = False
    count = len(getattr(drafter, "_topk_log", []) or [])
    return {"uninstalled": True, "topk_log_entries": int(count)}


def install_vllm_draft_layer_capture(
    worker: Any,
    d0_row: int = 1,
    req_idx: int = 0,
    num_query_per_req: int | None = None,
) -> dict[str, Any]:
    """Test G installer.

    Registers forward hooks on every ``drafter.model.model.layers[i]`` and on
    ``drafter.model.model.norm`` so that on the FIRST forward call we capture
    the d0-row hidden state per layer (plus the final post-norm hidden).

    The "full residual stream" output of layer i in vLLM's DFlashQwen3 is the
    sum of the fused-residual tuple returned by the layer: ``hidden_states``
    (post-mlp) plus ``residual`` (pre-mlp residual-out). We capture both
    components independently for inspection and also their sum at the d0 row
    (which is equal to HF's ``draft.layers[i](...)`` output at the same row).

    Args:
      d0_row: the row index inside the query portion that corresponds to the
        d0 (first mask) position for ``req_idx=0``. In vLLM this equals
        ``req_idx * num_query_per_req + 1`` (query_off=1 is the first mask).
        Default 1 covers ``req_idx=0, num_query_per_req>=2``.
    """
    import torch

    worker_obj = getattr(worker, "worker", worker)
    model_runner = getattr(worker_obj, "model_runner", None)
    if model_runner is None:
        return {"error": "no model_runner"}
    drafter = getattr(model_runner, "drafter", None)
    if drafter is None:
        return {"error": "no drafter"}
    draft_model = getattr(drafter, "model", None)
    if draft_model is None:
        return {"error": "no drafter.model"}
    draft_inner = getattr(draft_model, "model", None)
    if draft_inner is None:
        return {"error": "no drafter.model.model"}
    layers = getattr(draft_inner, "layers", None)
    norm_mod = getattr(draft_inner, "norm", None)
    if layers is None or norm_mod is None:
        return {"error": "no draft layers or norm"}

    if num_query_per_req is None:
        try:
            num_query_per_req = int(1 + drafter.num_speculative_tokens)
        except Exception:
            num_query_per_req = None
    effective_d0_row = (
        int(req_idx) * int(num_query_per_req) + int(d0_row)
        if num_query_per_req is not None
        else int(d0_row)
    )

    state: dict[str, Any] = {
        "per_layer": {},
        "post_norm": None,
        "pre_norm_residual": None,
        "layer_input_at_d0_layer0": None,
        "handles": [],
        "call_index": 0,
        "captured": False,
        "d0_row": effective_d0_row,
        "num_query_per_req": num_query_per_req,
        "num_layers": len(layers),
    }

    def _row_to_cpu(t: torch.Tensor, row: int) -> list[float] | None:
        if t is None or not isinstance(t, torch.Tensor):
            return None
        if t.dim() < 2:
            return None
        if row < 0 or row >= t.shape[0]:
            return None
        return t[row].detach().cpu().float().tolist()

    def _tensor_stats(t: torch.Tensor) -> dict[str, Any] | None:
        if t is None or not isinstance(t, torch.Tensor):
            return None
        tf = t.detach().float()
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "norm": float(tf.norm().item()),
            "abs_mean": float(tf.abs().mean().item()),
        }

    def make_layer_hook(layer_idx: int):
        def hook(_module, args, kwargs, output):
            if state["captured"]:
                return None
            row = state["d0_row"]
            if layer_idx == 0:
                hs_in: torch.Tensor | None = None
                if kwargs and "hidden_states" in kwargs:
                    cand = kwargs.get("hidden_states")
                    if (
                        isinstance(cand, torch.Tensor)
                        and cand.dim() >= 2
                        and cand.dtype.is_floating_point
                    ):
                        hs_in = cand
                if hs_in is None:
                    for cand in (list(args) + list((kwargs or {}).values())):
                        if (
                            isinstance(cand, torch.Tensor)
                            and cand.dim() >= 2
                            and cand.dtype.is_floating_point
                        ):
                            hs_in = cand
                            break
                if hs_in is not None:
                    state["layer_input_at_d0_layer0"] = _row_to_cpu(hs_in, row)
            hs_out: torch.Tensor | None = None
            res_out: torch.Tensor | None = None
            if isinstance(output, tuple):
                if len(output) >= 1 and isinstance(output[0], torch.Tensor):
                    hs_out = output[0]
                if len(output) >= 2 and isinstance(output[1], torch.Tensor):
                    res_out = output[1]
            elif isinstance(output, torch.Tensor):
                hs_out = output
            entry: dict[str, Any] = {
                "layer_idx": int(layer_idx),
                "hs_out_row": _row_to_cpu(hs_out, row) if hs_out is not None else None,
                "residual_row": (
                    _row_to_cpu(res_out, row) if res_out is not None else None
                ),
                "hs_out_stats": _tensor_stats(hs_out) if hs_out is not None else None,
                "residual_stats": _tensor_stats(res_out) if res_out is not None else None,
            }
            if (
                hs_out is not None
                and res_out is not None
                and hs_out.shape == res_out.shape
                and row < hs_out.shape[0]
            ):
                try:
                    full = (hs_out[row].detach().float() + res_out[row].detach().float())
                    entry["full_residual_row"] = full.cpu().tolist()
                    entry["full_residual_norm"] = float(full.norm().item())
                except Exception as e:
                    entry["full_residual_row"] = None
                    entry["full_residual_error"] = f"{type(e).__name__}: {e}"
            state["per_layer"][int(layer_idx)] = entry
            return None

        return hook

    def norm_hook(_module, args, output):
        if state["captured"]:
            return None
        row = state["d0_row"]
        pre_norm_residual: torch.Tensor | None = None
        if args:
            a0 = args[0] if len(args) >= 1 else None
            a1 = args[1] if len(args) >= 2 else None
            if isinstance(a0, torch.Tensor) and isinstance(a1, torch.Tensor):
                try:
                    pre_norm_residual = a0.detach().float() + a1.detach().float()
                except Exception:
                    pre_norm_residual = None
        post_norm: torch.Tensor | None = None
        if isinstance(output, tuple) and len(output) >= 1 and isinstance(output[0], torch.Tensor):
            post_norm = output[0]
        elif isinstance(output, torch.Tensor):
            post_norm = output
        if pre_norm_residual is not None and row < pre_norm_residual.shape[0]:
            try:
                state["pre_norm_residual"] = pre_norm_residual[row].cpu().tolist()
            except Exception:
                state["pre_norm_residual"] = None
        if post_norm is not None and row < post_norm.shape[0]:
            state["post_norm"] = _row_to_cpu(post_norm, row)
        state["captured"] = True
        state["call_index"] += 1
        return None

    for li, layer in enumerate(layers):
        state["handles"].append(
            layer.register_forward_hook(
                make_layer_hook(li), with_kwargs=True
            )
        )
    state["handles"].append(norm_mod.register_forward_hook(norm_hook))

    worker_obj._draft_quality_draft_layer_capture = state
    return {
        "status": "installed",
        "d0_row": effective_d0_row,
        "num_layers": state["num_layers"],
        "num_query_per_req": num_query_per_req,
    }


def retrieve_vllm_draft_layer_capture(worker: Any) -> dict[str, Any]:
    worker_obj = getattr(worker, "worker", worker)
    state = getattr(worker_obj, "_draft_quality_draft_layer_capture", None)
    if state is None:
        return {"error": "no capture state"}
    for h in state.get("handles", []):
        try:
            h.remove()
        except Exception:
            pass
    out = {
        "d0_row": state.get("d0_row"),
        "num_layers": state.get("num_layers"),
        "num_query_per_req": state.get("num_query_per_req"),
        "call_index_at_capture": state.get("call_index"),
        "captured": state.get("captured"),
        "layer_input_at_d0_layer0": state.get("layer_input_at_d0_layer0"),
        "per_layer": {
            str(int(k)): v for k, v in (state.get("per_layer") or {}).items()
        },
        "pre_norm_residual_at_d0": state.get("pre_norm_residual"),
        "post_norm_at_d0": state.get("post_norm"),
    }
    try:
        delattr(worker_obj, "_draft_quality_draft_layer_capture")
    except Exception:
        pass
    return out


def build_draft_layer_parity_report(
    reference: dict[str, Any],
    vllm_capture: dict[str, Any],
    cosine_threshold: float = 0.999,
) -> dict[str, Any]:
    """Test G report.

    For each decoder layer i, compares HF's "full residual stream" output at
    the d0 row (stored in reference["draft_layer_outputs_at_d0"][str(i)])
    against vLLM's ``hs_out_row + residual_row`` at the same row (Test G
    captures both components). Also compares the post-norm hidden state that
    feeds the draft lm_head.

    Reports, per layer: cosine, L2 diff, max-abs diff, ref/vllm norms. The
    first layer whose cosine falls below ``cosine_threshold`` is flagged as
    the likely divergence entry point.
    """
    ref_layers = reference.get("draft_layer_outputs_at_d0") or {}
    ref_pre_norm = list(reference.get("draft_pre_norm_residual_at_d0") or [])
    ref_post_norm = list(reference.get("draft_post_norm_at_d0") or [])
    ref_noise = list(reference.get("draft_noise_embed_at_d0") or [])
    ref_d0_row = reference.get("draft_d0_row_index")
    ref_num_layers = reference.get("draft_num_hidden_layers")

    v_per_layer = vllm_capture.get("per_layer") or {}
    v_pre_norm = list(vllm_capture.get("pre_norm_residual_at_d0") or [])
    v_post_norm = list(vllm_capture.get("post_norm_at_d0") or [])
    v_layer_input0 = list(vllm_capture.get("layer_input_at_d0_layer0") or [])
    v_d0_row = vllm_capture.get("d0_row")
    v_num_layers = vllm_capture.get("num_layers")

    def _compare(a: list[float], b: list[float]) -> dict[str, Any]:
        if not a or not b or len(a) != len(b):
            return {
                "status": "shape_mismatch",
                "ref_len": len(a),
                "vllm_len": len(b),
            }
        return {
            "status": "ok",
            "len": len(a),
            "cosine": float(_cosine(a, b)),
            "l2_diff": _l2_diff(a, b),
            "max_abs_diff": _max_abs_diff(a, b),
            "mean_abs_diff": _mean_abs([x - y for x, y in zip(a, b)]),
            "ref_norm": math.sqrt(sum(x * x for x in a)),
            "vllm_norm": math.sqrt(sum(x * x for x in b)),
        }

    per_layer_report: dict[str, Any] = {}
    first_divergence: dict[str, Any] | None = None

    sorted_layer_keys = sorted(
        (int(k) for k in ref_layers.keys()),
    )
    for li in sorted_layer_keys:
        ref_vec = list(ref_layers.get(str(li)) or [])
        v_entry = v_per_layer.get(str(li)) or {}
        v_vec = list(v_entry.get("full_residual_row") or [])
        cmp = _compare(ref_vec, v_vec)
        cmp["vllm_hs_out_norm"] = (
            (v_entry.get("hs_out_stats") or {}).get("norm")
        )
        cmp["vllm_residual_norm"] = (
            (v_entry.get("residual_stats") or {}).get("norm")
        )
        per_layer_report[str(li)] = cmp
        if (
            first_divergence is None
            and cmp.get("status") == "ok"
            and cmp.get("cosine", 1.0) < cosine_threshold
        ):
            first_divergence = {
                "layer_idx": int(li),
                "cosine": cmp.get("cosine"),
                "l2_diff": cmp.get("l2_diff"),
                "max_abs_diff": cmp.get("max_abs_diff"),
                "ref_norm": cmp.get("ref_norm"),
                "vllm_norm": cmp.get("vllm_norm"),
            }

    report: dict[str, Any] = {
        "d0_row_check": {
            "reference_d0_row_index": ref_d0_row,
            "vllm_d0_row": v_d0_row,
            "reference_num_draft_layers": ref_num_layers,
            "vllm_num_draft_layers": v_num_layers,
            "match_num_layers": (
                ref_num_layers is not None
                and v_num_layers is not None
                and int(ref_num_layers) == int(v_num_layers)
            ),
        },
        "noise_embed_parity_at_d0": _compare(ref_noise, v_layer_input0)
        if ref_noise and v_layer_input0
        else {"skipped": True},
        "per_layer_parity": per_layer_report,
        "first_divergence_layer": first_divergence,
        "pre_norm_residual_parity": _compare(ref_pre_norm, v_pre_norm)
        if ref_pre_norm and v_pre_norm
        else {"skipped": True},
        "post_norm_parity": _compare(ref_post_norm, v_post_norm)
        if ref_post_norm and v_post_norm
        else {"skipped": True},
        "cosine_threshold": float(cosine_threshold),
    }
    return report


# ---------------------------------------------------------------------------
# Test H: intra-layer bisect at d0 (default: layer 1).
# ---------------------------------------------------------------------------
# Goal: pinpoint which sub-operation (input_layernorm, self_attn, post_attn_ln,
# or mlp) in the first diverging layer is responsible for the draft's hidden
# state drift between HF and vLLM.
#
# Six tap points at the d0 row inside the chosen layer:
#   (A) layer_input          = residual stream entering the layer
#   (B) post_input_ln        = hidden after input_layernorm (pre-QKV)
#   (C) self_attn_out        = raw self-attention output (pre-residual-add)
#   (D) post_attn_residual   = residual stream right after the attn residual-add
#   (E) post_post_attn_ln    = hidden after post_attention_layernorm (pre-MLP)
#   (F) mlp_out              = raw MLP output (pre-residual-add)
# The layer's full output at d0 = (D) + (F), which is Test G's
# ``full_residual_row`` (and HF's ``draft.layers[idx]`` return).
#
# HF's Qwen3DFlashDecoderLayer uses the classic pre-norm pattern with plain
# RMSNorm and explicit residual adds (see dflash/model/dflash.py), so the six
# taps are easy to capture with forward pre/post hooks on submodules.
#
# vLLM's DFlashQwen3DecoderLayer uses fused RMSNorm-with-residual that returns
# ``(normalized, residual_plus_input)``. We recover the six taps as follows:
#   (A) = output[1] of input_layernorm forward_hook  (residual + incoming hs)
#   (B) = output[0] of input_layernorm forward_hook  (norm of A)
#   (C) = output of self_attn forward_hook (single tensor)
#   (D) = output[1] of post_attention_layernorm forward_hook (= A + C)
#   (E) = output[0] of post_attention_layernorm forward_hook (norm of D)
#   (F) = output of mlp forward_hook (single tensor)


def install_vllm_draft_layer1_bisect_capture(
    worker: Any,
    d0_row: int = 1,
    req_idx: int = 0,
    num_query_per_req: int | None = None,
    bisect_layer_idx: int = 1,
) -> dict[str, Any]:
    """Test H installer.

    Registers forward hooks on the *sub-modules* of
    ``drafter.model.model.layers[bisect_layer_idx]`` to capture six
    intra-layer tap points at the d0 row on the FIRST forward call.
    """
    import torch

    worker_obj = getattr(worker, "worker", worker)
    model_runner = getattr(worker_obj, "model_runner", None)
    if model_runner is None:
        return {"error": "no model_runner"}
    drafter = getattr(model_runner, "drafter", None)
    if drafter is None:
        return {"error": "no drafter"}
    draft_model = getattr(drafter, "model", None)
    if draft_model is None:
        return {"error": "no drafter.model"}
    draft_inner = getattr(draft_model, "model", None)
    if draft_inner is None:
        return {"error": "no drafter.model.model"}
    layers = getattr(draft_inner, "layers", None)
    if layers is None or bisect_layer_idx >= len(layers):
        return {
            "error": f"invalid bisect_layer_idx={bisect_layer_idx}"
            f" (num_layers={len(layers) if layers is not None else None})"
        }
    layer = layers[bisect_layer_idx]
    input_ln = getattr(layer, "input_layernorm", None)
    self_attn = getattr(layer, "self_attn", None)
    post_attn_ln = getattr(layer, "post_attention_layernorm", None)
    mlp = getattr(layer, "mlp", None)
    if input_ln is None or self_attn is None or post_attn_ln is None or mlp is None:
        return {
            "error": "missing sub-module on target layer",
            "have": {
                "input_layernorm": input_ln is not None,
                "self_attn": self_attn is not None,
                "post_attention_layernorm": post_attn_ln is not None,
                "mlp": mlp is not None,
            },
        }

    if num_query_per_req is None:
        try:
            num_query_per_req = int(1 + drafter.num_speculative_tokens)
        except Exception:
            num_query_per_req = None
    effective_d0_row = (
        int(req_idx) * int(num_query_per_req) + int(d0_row)
        if num_query_per_req is not None
        else int(d0_row)
    )

    state: dict[str, Any] = {
        "bisect_layer_idx": int(bisect_layer_idx),
        "d0_row": effective_d0_row,
        "num_query_per_req": num_query_per_req,
        "handles": [],
        "captured": {
            "input_layernorm": False,
            "self_attn": False,
            "post_attention_layernorm": False,
            "mlp": False,
        },
        "taps": {},
        "stats": {},
        "call_index": 0,
    }

    def _row_to_cpu(t: Any, row: int) -> list[float] | None:
        if t is None or not isinstance(t, torch.Tensor):
            return None
        if t.dim() < 2:
            return None
        if row < 0 or row >= t.shape[0]:
            return None
        return t[row].detach().cpu().float().tolist()

    def _tensor_stats(t: Any) -> dict[str, Any] | None:
        if t is None or not isinstance(t, torch.Tensor):
            return None
        tf = t.detach().float()
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "norm": float(tf.norm().item()),
            "abs_mean": float(tf.abs().mean().item()),
        }

    def _store(tap_name: str, tensor: Any) -> None:
        row = state["d0_row"]
        state["taps"][tap_name] = _row_to_cpu(tensor, row)
        state["stats"][tap_name] = _tensor_stats(tensor)

    def input_ln_hook(_module, _args, output):
        if state["captured"]["input_layernorm"]:
            return None
        hs_out: Any = None
        res_out: Any = None
        if isinstance(output, tuple):
            if len(output) >= 1:
                hs_out = output[0]
            if len(output) >= 2:
                res_out = output[1]
        elif isinstance(output, torch.Tensor):
            hs_out = output
        _store("layer_input", res_out)
        _store("post_input_ln", hs_out)
        state["captured"]["input_layernorm"] = True
        return None

    def self_attn_hook(_module, _args, output):
        if state["captured"]["self_attn"]:
            return None
        attn_out: Any = None
        if isinstance(output, tuple):
            if len(output) >= 1:
                attn_out = output[0]
        elif isinstance(output, torch.Tensor):
            attn_out = output
        _store("self_attn_out", attn_out)
        state["captured"]["self_attn"] = True
        return None

    def post_attn_ln_hook(_module, _args, output):
        if state["captured"]["post_attention_layernorm"]:
            return None
        hs_out: Any = None
        res_out: Any = None
        if isinstance(output, tuple):
            if len(output) >= 1:
                hs_out = output[0]
            if len(output) >= 2:
                res_out = output[1]
        elif isinstance(output, torch.Tensor):
            hs_out = output
        _store("post_attn_residual", res_out)
        _store("post_post_attn_ln", hs_out)
        state["captured"]["post_attention_layernorm"] = True
        return None

    def mlp_hook(_module, _args, output):
        if state["captured"]["mlp"]:
            return None
        mlp_out: Any = None
        if isinstance(output, tuple):
            if len(output) >= 1:
                mlp_out = output[0]
        elif isinstance(output, torch.Tensor):
            mlp_out = output
        _store("mlp_out", mlp_out)
        state["captured"]["mlp"] = True
        state["call_index"] += 1
        return None

    state["handles"].append(input_ln.register_forward_hook(input_ln_hook))
    state["handles"].append(self_attn.register_forward_hook(self_attn_hook))
    state["handles"].append(
        post_attn_ln.register_forward_hook(post_attn_ln_hook)
    )
    state["handles"].append(mlp.register_forward_hook(mlp_hook))

    worker_obj._draft_quality_layer1_bisect_capture = state
    return {
        "status": "installed",
        "bisect_layer_idx": int(bisect_layer_idx),
        "d0_row": effective_d0_row,
        "num_query_per_req": num_query_per_req,
    }


def retrieve_vllm_draft_layer1_bisect_capture(worker: Any) -> dict[str, Any]:
    worker_obj = getattr(worker, "worker", worker)
    state = getattr(worker_obj, "_draft_quality_layer1_bisect_capture", None)
    if state is None:
        return {"error": "no capture state"}
    for h in state.get("handles", []):
        try:
            h.remove()
        except Exception:
            pass
    out = {
        "bisect_layer_idx": state.get("bisect_layer_idx"),
        "d0_row": state.get("d0_row"),
        "num_query_per_req": state.get("num_query_per_req"),
        "captured": state.get("captured"),
        "call_index_at_capture": state.get("call_index"),
        "taps": state.get("taps") or {},
        "stats": state.get("stats") or {},
    }
    try:
        delattr(worker_obj, "_draft_quality_layer1_bisect_capture")
    except Exception:
        pass
    return out


def build_draft_layer1_bisect_report(
    reference: dict[str, Any],
    vllm_capture: dict[str, Any],
    cosine_threshold: float = 0.999,
) -> dict[str, Any]:
    """Test H report.

    Compares six intra-layer taps at the d0 row between HF and vLLM for the
    first diverging layer (default: layer 1). The taps are ordered so the
    first entry whose cosine drops below ``cosine_threshold`` pinpoints the
    sub-operation introducing the drift.
    """
    ref_bisect = reference.get("draft_layer1_bisect_at_d0") or {}
    v_taps = vllm_capture.get("taps") or {}
    v_stats = vllm_capture.get("stats") or {}
    ref_layer_idx = ref_bisect.get("layer_idx")
    v_layer_idx = vllm_capture.get("bisect_layer_idx")

    tap_order = [
        "layer_input",
        "post_input_ln",
        "self_attn_out",
        "post_attn_residual",
        "post_post_attn_ln",
        "mlp_out",
    ]

    def _compare(a: list[float], b: list[float]) -> dict[str, Any]:
        if not a or not b or len(a) != len(b):
            return {
                "status": "shape_mismatch",
                "ref_len": len(a) if a else 0,
                "vllm_len": len(b) if b else 0,
            }
        return {
            "status": "ok",
            "len": len(a),
            "cosine": float(_cosine(a, b)),
            "l2_diff": _l2_diff(a, b),
            "max_abs_diff": _max_abs_diff(a, b),
            "mean_abs_diff": _mean_abs([x - y for x, y in zip(a, b)]),
            "ref_norm": math.sqrt(sum(x * x for x in a)),
            "vllm_norm": math.sqrt(sum(x * x for x in b)),
        }

    per_tap_report: dict[str, Any] = {}
    first_divergence: dict[str, Any] | None = None
    for tap in tap_order:
        ref_vec = list(ref_bisect.get(tap) or [])
        v_vec = list(v_taps.get(tap) or [])
        if not ref_vec or not v_vec:
            per_tap_report[tap] = {
                "skipped": True,
                "ref_present": bool(ref_vec),
                "vllm_present": bool(v_vec),
            }
            continue
        cmp = _compare(ref_vec, v_vec)
        v_stat = v_stats.get(tap) or {}
        cmp["vllm_dtype"] = v_stat.get("dtype")
        cmp["vllm_shape"] = v_stat.get("shape")
        per_tap_report[tap] = cmp
        if (
            first_divergence is None
            and cmp.get("status") == "ok"
            and cmp.get("cosine", 1.0) < cosine_threshold
        ):
            first_divergence = {
                "tap": tap,
                "cosine": cmp.get("cosine"),
                "l2_diff": cmp.get("l2_diff"),
                "max_abs_diff": cmp.get("max_abs_diff"),
                "ref_norm": cmp.get("ref_norm"),
                "vllm_norm": cmp.get("vllm_norm"),
            }

    report: dict[str, Any] = {
        "layer_idx_check": {
            "reference_layer_idx": ref_layer_idx,
            "vllm_bisect_layer_idx": v_layer_idx,
            "match": (
                ref_layer_idx is not None
                and v_layer_idx is not None
                and int(ref_layer_idx) == int(v_layer_idx)
            ),
        },
        "d0_row_vllm": vllm_capture.get("d0_row"),
        "tap_order": tap_order,
        "per_tap_parity": per_tap_report,
        "first_divergence_tap": first_divergence,
        "cosine_threshold": float(cosine_threshold),
        "vllm_captured_flags": vllm_capture.get("captured"),
    }
    return report


# ---------------------------------------------------------------------------
# Test I: intra-self_attn bisect at d0 + context K/V probe.
# ---------------------------------------------------------------------------
# Goal: given Test H's finding that self_attn is the first big divergence at
# layer 1, pinpoint WHICH sub-operation inside self_attn is responsible:
#
# I-1 (query-side): four taps inside self_attn.forward at d0 row:
#   i1_q_post_qproj      -- Q after qkv_proj split (pre-q_norm)
#   i2_q_post_qnorm      -- Q after per-head q_norm (pre-RoPE)
#   i3_q_post_rope       -- Q after RoPE
#   i4_attn_out_pre_oproj -- attention output pre o_proj
# (i5 = self_attn_out post-o_proj is already covered by Test H.)
# Also captures k_noise and v_noise at d0 for completeness.
#
# I-2 (context-KV probe): K and V at the last context position for the chosen
# layer, after all pre-processing (hidden_norm, k_proj/v_proj, k_norm, RoPE).
# HF stores these inside past_key_values_draft after the layer's forward;
# vLLM stores these inside paged KV cache after precompute_and_store_context_kv.
# We capture vLLM's ``all_k_final[layer, last_ctx]`` and ``all_v[layer, last_ctx]``
# via a monkey-patched copy of precompute_and_store_context_kv.


def install_vllm_draft_layer1_attn_bisect_capture(
    worker: Any,
    d0_row: int = 1,
    req_idx: int = 0,
    num_query_per_req: int | None = None,
    bisect_layer_idx: int = 1,
) -> dict[str, Any]:
    """Test I installer.

    Monkey-patches two methods on the draft model:

    1. ``drafter.model.model.layers[bisect_layer_idx].self_attn.forward`` --
       mirrors the original DFlashQwen3Attention.forward logic but captures
       intermediate tensors at the d0 row (Test I-1).
    2. ``drafter.model.model.precompute_and_store_context_kv`` -- runs the
       original logic unchanged and additionally records
       ``all_k_final[bisect_layer_idx, last_ctx]`` and
       ``all_v[bisect_layer_idx, last_ctx]`` (Test I-2).

    Both captures are one-shot: they record on the FIRST invocation and then
    refuse further captures. The monkey-patches remain in place until
    :func:`retrieve_vllm_draft_layer1_attn_bisect_capture` is called.
    """
    import torch
    import torch.nn.functional as F

    worker_obj = getattr(worker, "worker", worker)
    model_runner = getattr(worker_obj, "model_runner", None)
    if model_runner is None:
        return {"error": "no model_runner"}
    drafter = getattr(model_runner, "drafter", None)
    if drafter is None:
        return {"error": "no drafter"}
    draft_model = getattr(drafter, "model", None)
    if draft_model is None:
        return {"error": "no drafter.model"}
    draft_inner = getattr(draft_model, "model", None)
    if draft_inner is None:
        return {"error": "no drafter.model.model"}
    layers = getattr(draft_inner, "layers", None)
    if layers is None or bisect_layer_idx >= len(layers):
        return {
            "error": f"invalid bisect_layer_idx={bisect_layer_idx}"
        }
    layer = layers[bisect_layer_idx]
    self_attn = getattr(layer, "self_attn", None)
    if self_attn is None:
        return {"error": "no self_attn on target layer"}

    if num_query_per_req is None:
        try:
            num_query_per_req = int(1 + drafter.num_speculative_tokens)
        except Exception:
            num_query_per_req = None
    effective_d0_row = (
        int(req_idx) * int(num_query_per_req) + int(d0_row)
        if num_query_per_req is not None
        else int(d0_row)
    )

    state: dict[str, Any] = {
        "bisect_layer_idx": int(bisect_layer_idx),
        "d0_row": effective_d0_row,
        "num_query_per_req": num_query_per_req,
        "captured_attn": False,
        "captured_ctx_kv": False,
        "attn_taps": {},
        "attn_stats": {},
        "attn_shapes": {},
        "ctx_kv": {},
        "ctx_kv_stats": {},
        "original_self_attn_forward": self_attn.forward,
        "original_precompute": getattr(
            draft_inner, "precompute_and_store_context_kv", None
        ),
        "self_attn": self_attn,
        "draft_inner": draft_inner,
    }

    def _row_to_cpu(t: Any, row: int) -> list[float] | None:
        if t is None or not isinstance(t, torch.Tensor):
            return None
        if t.dim() < 2:
            return None
        if row < 0 or row >= t.shape[0]:
            return None
        return t[row].detach().cpu().float().tolist()

    def _flat_to_cpu(t: Any) -> list[float] | None:
        if t is None or not isinstance(t, torch.Tensor):
            return None
        return t.detach().reshape(-1).cpu().float().tolist()

    def _tensor_stats(t: Any) -> dict[str, Any] | None:
        if t is None or not isinstance(t, torch.Tensor):
            return None
        tf = t.detach().float()
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "norm": float(tf.norm().item()),
            "abs_mean": float(tf.abs().mean().item()),
        }

    def _store_row(tap_name: str, tensor: Any, row: int) -> None:
        state["attn_taps"][tap_name] = _row_to_cpu(tensor, row)
        state["attn_stats"][tap_name] = _tensor_stats(tensor)
        if isinstance(tensor, torch.Tensor):
            state["attn_shapes"][tap_name] = list(tensor.shape)

    original_self_attn_forward = self_attn.forward

    def patched_self_attn_forward(positions, hidden_states):
        if state["captured_attn"]:
            return original_self_attn_forward(positions, hidden_states)
        qkv = F.linear(
            hidden_states,
            self_attn.qkv_proj.weight,
            self_attn.qkv_proj.bias,
        )
        q, k, v = qkv.split(
            [self_attn.q_size, self_attn.kv_size, self_attn.kv_size],
            dim=-1,
        )
        row = state["d0_row"]
        _store_row("i1_q_post_qproj", q, row)
        _store_row("k_noise_post_kproj", k, row)
        _store_row("v_noise_post_vproj", v, row)
        q_shape, k_shape = q.shape, k.shape
        q_normed = self_attn.q_norm(
            q.view(
                *q_shape[:-1],
                q_shape[-1] // self_attn.head_dim,
                self_attn.head_dim,
            )
        ).view(q_shape)
        k_normed = self_attn.k_norm(
            k.view(
                *k_shape[:-1],
                k_shape[-1] // self_attn.head_dim,
                self_attn.head_dim,
            )
        ).view(k_shape)
        _store_row("i2_q_post_qnorm", q_normed, row)
        _store_row("k_noise_post_knorm", k_normed, row)
        q_rot, k_rot = self_attn.rotary_emb(positions, q_normed, k_normed)
        _store_row("i3_q_post_rope", q_rot, row)
        _store_row("k_noise_post_rope", k_rot, row)

        # ---------- Test J: manual SDPA reference ----------
        # Run the "textbook" SDPA over [context K/V | current-batch K/V] using
        # the exact same tensors the kernel is supposed to consume.  If this
        # differs from the kernel output, the bug lives in the kernel /
        # metadata path; if it matches, the kernel is fine and the issue is
        # elsewhere.
        manual_attn_out: torch.Tensor | None = None
        try:
            ctx_k = state.get("_ctx_k_full")
            ctx_v = state.get("_ctx_v_full")
            if ctx_k is not None and ctx_v is not None:
                num_heads = self_attn.num_heads
                num_kv_heads = self_attn.num_kv_heads
                head_dim = self_attn.head_dim
                scaling = self_attn.scaling
                q_len = q_rot.shape[0]
                ctx_len = ctx_k.shape[0]
                # Query-side K/V at this layer (post-RoPE).
                k_noise_full = k_rot.view(q_len, num_kv_heads, head_dim)
                v_noise_full = v.view(q_len, num_kv_heads, head_dim)
                # Build concatenated K/V: context first, then current-batch.
                k_full = torch.cat([ctx_k, k_noise_full], dim=0)
                v_full = torch.cat([ctx_v, v_noise_full], dim=0)
                # GQA expansion: each kv-head feeds `num_heads/num_kv_heads` q-heads.
                repeats = num_heads // num_kv_heads
                k_exp = k_full.repeat_interleave(repeats, dim=1)
                v_exp = v_full.repeat_interleave(repeats, dim=1)
                q4 = q_rot.view(q_len, num_heads, head_dim)
                # Rearrange to (H, q_len, hd) / (H, kv_len, hd) for einsum.
                q_hqd = q4.transpose(0, 1).float()
                k_hkd = k_exp.transpose(0, 1).float()
                v_hkd = v_exp.transpose(0, 1).float()
                scores = torch.einsum("hqd,hkd->hqk", q_hqd, k_hkd) * scaling
                kv_len = k_full.shape[0]
                key_positions = torch.arange(
                    kv_len, device=q_rot.device
                )
                query_positions = ctx_len + torch.arange(
                    q_len, device=q_rot.device
                )
                mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
                scores = scores.masked_fill(
                    mask.unsqueeze(0), float("-inf")
                )
                attn_w = torch.softmax(scores, dim=-1)
                manual = torch.einsum("hqk,hkd->hqd", attn_w, v_hkd)
                manual = manual.transpose(0, 1).reshape(
                    q_len, num_heads * head_dim
                )
                manual_attn_out = manual.to(q_rot.dtype)
                _store_row(
                    "j_manual_attn_out_pre_oproj", manual_attn_out, row
                )
        except Exception as _e:
            state["_j_manual_error"] = str(_e)

        # ---------- Test J: forward-context metadata snapshot ----------
        try:
            from vllm.forward_context import get_forward_context  # noqa: PLC0415

            fc = get_forward_context()
            attn_meta = getattr(fc, "attn_metadata", None)
            if attn_meta is not None:
                # attn_metadata can be dict keyed by layer_name or a single obj.
                if isinstance(attn_meta, dict):
                    layer_name = getattr(self_attn.attn, "layer_name", None)
                    meta_obj = attn_meta.get(layer_name)
                    if meta_obj is None and attn_meta:
                        meta_obj = next(iter(attn_meta.values()))
                else:
                    meta_obj = attn_meta
                state["attn_metadata"] = _snapshot_attn_metadata(meta_obj)
        except Exception as _e:
            state["attn_metadata_error"] = str(_e)

        attn_output = self_attn.attn(q_rot, k_rot, v)
        _store_row("i4_attn_out_pre_oproj", attn_output, row)
        # Manual-vs-kernel comparison done here (where we now have both).
        if manual_attn_out is not None:
            try:
                diff = (manual_attn_out[row].float()
                        - attn_output[row].float())
                state["manual_vs_kernel"] = {
                    "d0_row": int(row),
                    "cosine": float(
                        torch.nn.functional.cosine_similarity(
                            manual_attn_out[row].float().unsqueeze(0),
                            attn_output[row].float().unsqueeze(0),
                            dim=-1,
                        ).item()
                    ),
                    "l2_diff": float(diff.norm().item()),
                    "max_abs_diff": float(diff.abs().max().item()),
                    "manual_norm": float(
                        manual_attn_out[row].float().norm().item()
                    ),
                    "kernel_norm": float(
                        attn_output[row].float().norm().item()
                    ),
                }
            except Exception as _e:
                state["manual_vs_kernel_error"] = str(_e)

        output, _ = self_attn.o_proj(attn_output)
        _store_row("i5_attn_out_post_oproj", output, row)
        state["captured_attn"] = True
        return output

    self_attn.forward = patched_self_attn_forward

    original_precompute = state["original_precompute"]

    def patched_precompute_and_store_context_kv(
        context_states,
        context_positions,
        context_slot_mapping=None,
    ):
        if state["captured_ctx_kv"] or original_precompute is None:
            return original_precompute(
                context_states, context_positions, context_slot_mapping
            )
        di = draft_inner
        if not hasattr(di, "_num_attn_layers"):
            di._build_fused_kv_buffers()
        num_ctx = context_states.shape[0]
        L = di._num_attn_layers
        kv = di._kv_size
        hd = di._head_dim
        nkv = di._num_kv_heads
        normed_context_states = torch.empty_like(context_states)
        from vllm import _custom_ops as ops  # noqa: PLC0415

        ops.rms_norm(
            normed_context_states,
            context_states,
            di._hidden_norm_weight,
            di._rms_norm_eps,
        )
        all_kv_flat = F.linear(
            normed_context_states, di._fused_kv_weight, di._fused_kv_bias
        )
        all_kv = (
            all_kv_flat.view(num_ctx, L, 2, nkv, hd)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        all_k = all_kv[0]
        all_v = all_kv[1]
        all_k_normed = torch.empty_like(all_k)
        for i in range(L):
            ops.rms_norm(
                all_k_normed[i],
                all_k[i],
                di._k_norm_weights[i],
                di._rms_norm_eps,
            )
        # NOTE: the fused RoPE below is IN-PLACE on all_k_flat, which aliases
        # all_k_normed.  Snapshot the pre-RoPE K for Test I-2 BEFORE that call.
        last_ctx = num_ctx - 1
        if 0 <= bisect_layer_idx < L and last_ctx >= 0:
            pre_rope_k = (
                all_k_normed[bisect_layer_idx, last_ctx].detach().clone()
            )
            v_row = all_v[bisect_layer_idx, last_ctx].detach().clone()
            state["ctx_kv"]["k_last_context_pre_rope"] = _flat_to_cpu(pre_rope_k)
            state["ctx_kv"]["v_last_context"] = _flat_to_cpu(v_row)
            state["ctx_kv_stats"]["k_last_context_pre_rope"] = _tensor_stats(
                pre_rope_k
            )
            state["ctx_kv_stats"]["v_last_context"] = _tensor_stats(v_row)
            state["ctx_kv"]["layer_idx"] = int(bisect_layer_idx)
            state["ctx_kv"]["last_context_index"] = int(last_ctx)
            state["ctx_kv"]["num_context"] = int(num_ctx)
        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        positions_repeated = context_positions.repeat(L)
        cos_sin_cache = di._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(
            positions_repeated,
            all_k_flat,
            None,
            di._rope_head_size,
            cos_sin_cache,
            di._rope_is_neox,
        )
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        if 0 <= bisect_layer_idx < L and last_ctx >= 0:
            k_row = all_k_final[bisect_layer_idx, last_ctx].detach().clone()
            state["ctx_kv"]["k_last_context"] = _flat_to_cpu(k_row)
            state["ctx_kv_stats"]["k_last_context"] = _tensor_stats(k_row)
            # Test J: stash the FULL post-RoPE context K and V for the chosen
            # layer, so the patched self_attn forward can run a manual SDPA
            # with the exact same data the attention kernel is supposed to
            # see (context K/V from paged cache + query K/V from the current
            # forward).
            state["_ctx_k_full"] = (
                all_k_final[bisect_layer_idx].detach().clone()
            )
            state["_ctx_v_full"] = (
                all_v[bisect_layer_idx].detach().clone()
            )
            state["_ctx_positions"] = (
                context_positions.detach().clone()
            )
        state["captured_ctx_kv"] = True

        if context_slot_mapping is None:
            return
        for i in range(L):
            attn = di._attn_layers[i]
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                context_slot_mapping,
            )
        return None

    if original_precompute is not None:
        # Bind the wrapper as a method-like replacement on the instance.
        draft_inner.precompute_and_store_context_kv = (
            patched_precompute_and_store_context_kv
        )

    worker_obj._draft_quality_layer1_attn_bisect_capture = state
    return {
        "status": "installed",
        "bisect_layer_idx": int(bisect_layer_idx),
        "d0_row": effective_d0_row,
        "num_query_per_req": num_query_per_req,
        "patched_precompute": original_precompute is not None,
    }


def retrieve_vllm_draft_layer1_attn_bisect_capture(worker: Any) -> dict[str, Any]:
    worker_obj = getattr(worker, "worker", worker)
    state = getattr(worker_obj, "_draft_quality_layer1_attn_bisect_capture", None)
    if state is None:
        return {"error": "no capture state"}
    # Restore self_attn.forward and precompute_and_store_context_kv.
    try:
        self_attn = state.get("self_attn")
        orig_forward = state.get("original_self_attn_forward")
        if self_attn is not None and orig_forward is not None:
            self_attn.forward = orig_forward
    except Exception:
        pass
    try:
        draft_inner = state.get("draft_inner")
        orig_precompute = state.get("original_precompute")
        if draft_inner is not None and orig_precompute is not None:
            draft_inner.precompute_and_store_context_kv = orig_precompute
    except Exception:
        pass
    out = {
        "bisect_layer_idx": state.get("bisect_layer_idx"),
        "d0_row": state.get("d0_row"),
        "num_query_per_req": state.get("num_query_per_req"),
        "captured_attn": state.get("captured_attn"),
        "captured_ctx_kv": state.get("captured_ctx_kv"),
        "attn_taps": state.get("attn_taps") or {},
        "attn_stats": state.get("attn_stats") or {},
        "attn_shapes": state.get("attn_shapes") or {},
        "ctx_kv": state.get("ctx_kv") or {},
        "ctx_kv_stats": state.get("ctx_kv_stats") or {},
        "manual_vs_kernel": state.get("manual_vs_kernel"),
        "manual_vs_kernel_error": state.get("manual_vs_kernel_error"),
        "j_manual_error": state.get("_j_manual_error"),
        "attn_metadata": state.get("attn_metadata"),
        "attn_metadata_error": state.get("attn_metadata_error"),
    }
    try:
        delattr(worker_obj, "_draft_quality_layer1_attn_bisect_capture")
    except Exception:
        pass
    return out


def build_draft_layer1_attn_bisect_report(
    reference: dict[str, Any],
    vllm_capture: dict[str, Any],
    cosine_threshold: float = 0.999,
) -> dict[str, Any]:
    """Test I report (I-1 + I-2).

    Compares intra-self_attn taps (I-1) and context K/V at last context
    position (I-2) between HF and vLLM at the same d0 row / layer.
    """
    ref_attn = reference.get("draft_layer1_attn_bisect_at_d0") or {}
    ref_ctx = reference.get("draft_layer1_context_kv") or {}

    v_taps = vllm_capture.get("attn_taps") or {}
    v_stats = vllm_capture.get("attn_stats") or {}
    v_shapes = vllm_capture.get("attn_shapes") or {}
    v_ctx = vllm_capture.get("ctx_kv") or {}
    v_ctx_stats = vllm_capture.get("ctx_kv_stats") or {}

    def _compare(a: list[float], b: list[float]) -> dict[str, Any]:
        if not a or not b or len(a) != len(b):
            return {
                "status": "shape_mismatch",
                "ref_len": len(a) if a else 0,
                "vllm_len": len(b) if b else 0,
            }
        return {
            "status": "ok",
            "len": len(a),
            "cosine": float(_cosine(a, b)),
            "l2_diff": _l2_diff(a, b),
            "max_abs_diff": _max_abs_diff(a, b),
            "mean_abs_diff": _mean_abs([x - y for x, y in zip(a, b)]),
            "ref_norm": math.sqrt(sum(x * x for x in a)),
            "vllm_norm": math.sqrt(sum(x * x for x in b)),
        }

    attn_tap_order = [
        "i1_q_post_qproj",
        "k_noise_post_kproj",
        "v_noise_post_vproj",
        "i2_q_post_qnorm",
        "k_noise_post_knorm",
        "i3_q_post_rope",
        "k_noise_post_rope",
        "j_manual_attn_out_pre_oproj",
        "i4_attn_out_pre_oproj",
        "i5_attn_out_post_oproj",
    ]

    per_tap_report: dict[str, Any] = {}
    first_divergence: dict[str, Any] | None = None
    for tap in attn_tap_order:
        ref_vec = list(ref_attn.get(tap) or [])
        v_vec = list(v_taps.get(tap) or [])
        if not ref_vec or not v_vec:
            per_tap_report[tap] = {
                "skipped": True,
                "ref_present": bool(ref_vec),
                "vllm_present": bool(v_vec),
            }
            continue
        cmp = _compare(ref_vec, v_vec)
        cmp["vllm_dtype"] = (v_stats.get(tap) or {}).get("dtype")
        cmp["vllm_shape"] = v_shapes.get(tap)
        per_tap_report[tap] = cmp
        if (
            first_divergence is None
            and cmp.get("status") == "ok"
            and cmp.get("cosine", 1.0) < cosine_threshold
        ):
            first_divergence = {
                "tap": tap,
                "cosine": cmp.get("cosine"),
                "l2_diff": cmp.get("l2_diff"),
                "max_abs_diff": cmp.get("max_abs_diff"),
                "ref_norm": cmp.get("ref_norm"),
                "vllm_norm": cmp.get("vllm_norm"),
            }

    ctx_keys = ["k_last_context_pre_rope", "k_last_context", "v_last_context"]
    ctx_report: dict[str, Any] = {}
    for ck in ctx_keys:
        ref_vec = list(ref_ctx.get(ck) or [])
        v_vec = list(v_ctx.get(ck) or [])
        if not ref_vec or not v_vec:
            ctx_report[ck] = {
                "skipped": True,
                "ref_present": bool(ref_vec),
                "vllm_present": bool(v_vec),
            }
            continue
        cmp = _compare(ref_vec, v_vec)
        cmp["vllm_dtype"] = (v_ctx_stats.get(ck) or {}).get("dtype")
        ctx_report[ck] = cmp

    # -------- Test J: manual SDPA vs kernel (both sides) --------
    hf_manual = list(ref_attn.get("j_manual_attn_out_pre_oproj") or [])
    hf_kernel = list(ref_attn.get("i4_attn_out_pre_oproj") or [])
    vllm_manual = list(v_taps.get("j_manual_attn_out_pre_oproj") or [])
    vllm_kernel = list(v_taps.get("i4_attn_out_pre_oproj") or [])

    def _safe_compare(a: list[float], b: list[float]) -> dict[str, Any]:
        if not a or not b or len(a) != len(b):
            return {
                "status": "shape_mismatch",
                "ref_len": len(a) if a else 0,
                "other_len": len(b) if b else 0,
            }
        return {
            "status": "ok",
            "len": len(a),
            "cosine": float(_cosine(a, b)),
            "l2_diff": _l2_diff(a, b),
            "max_abs_diff": _max_abs_diff(a, b),
            "a_norm": math.sqrt(sum(x * x for x in a)),
            "b_norm": math.sqrt(sum(x * x for x in b)),
        }

    test_j_report = {
        "hf_manual_vs_hf_kernel": _safe_compare(hf_manual, hf_kernel),
        "vllm_manual_vs_vllm_kernel": _safe_compare(vllm_manual, vllm_kernel),
        "hf_manual_vs_vllm_manual": _safe_compare(hf_manual, vllm_manual),
        "hf_kernel_vs_vllm_kernel": _safe_compare(hf_kernel, vllm_kernel),
        "vllm_inline_manual_vs_kernel": vllm_capture.get("manual_vs_kernel"),
    }

    report: dict[str, Any] = {
        "layer_idx_check": {
            "reference_attn_layer_idx": ref_attn.get("layer_idx"),
            "vllm_attn_layer_idx": vllm_capture.get("bisect_layer_idx"),
            "reference_ctx_layer_idx": ref_ctx.get("layer_idx"),
            "vllm_ctx_layer_idx": v_ctx.get("layer_idx"),
            "reference_last_context_index": ref_ctx.get("last_context_index"),
            "vllm_last_context_index": v_ctx.get("last_context_index"),
        },
        "d0_row_vllm": vllm_capture.get("d0_row"),
        "attn_tap_order": attn_tap_order,
        "per_attn_tap_parity": per_tap_report,
        "first_attn_divergence_tap": first_divergence,
        "context_kv_parity": ctx_report,
        "test_j_manual_vs_kernel": test_j_report,
        "vllm_attn_metadata": vllm_capture.get("attn_metadata"),
        "vllm_attn_metadata_error": vllm_capture.get("attn_metadata_error"),
        "cosine_threshold": float(cosine_threshold),
        "vllm_captured_flags": {
            "attn": vllm_capture.get("captured_attn"),
            "ctx_kv": vllm_capture.get("captured_ctx_kv"),
        },
    }
    return report


def build_test_k_hf_sdpa_report(
    reference: dict[str, Any],
    vllm_capture: dict[str, Any],
    cosine_threshold: float = 0.999,
) -> dict[str, Any]:
    """Test K report -- HF-side SDPA backend / dtype isolation.

    The HF-side capture produced multiple attention outputs at the d0 row by
    running the SAME q/k/v/mask through different torch SDPA backends and
    dtype combinations.  This report cross-compares every variant against
    three anchors:

      - ``hf_manual_fp32`` = ``j_manual_attn_out_pre_oproj`` (textbook causal
        SDPA computed in fp32 on HF side).  This is the "gold" numerical
        reference.
      - ``hf_kernel_default`` = ``i4_attn_out_pre_oproj`` on HF side.  This is
        whatever torch picked by default.
      - ``vllm_kernel`` = ``i4_attn_out_pre_oproj`` on vLLM side.  This is
        vLLM's flash-attn output; Test J showed it already matches textbook.

    The goal is to isolate which specific backend / dtype combination
    reproduces HF's default kernel output (the ~6% attenuated 151.9 norm)
    and whether any backend matches vLLM's 161.9 / textbook SDPA.  That tells
    us whether the discrepancy is a *kernel-selection* artifact or an
    inherent *bf16 softmax precision* effect.
    """

    variants = reference.get("test_k_hf_sdpa_variants") or {}
    variant_stats = variants.get("_stats") if isinstance(variants, dict) else None

    ref_attn = reference.get("draft_layer1_attn_bisect_at_d0") or {}
    v_taps = vllm_capture.get("attn_taps") or {}

    hf_manual_fp32 = list(ref_attn.get("j_manual_attn_out_pre_oproj") or [])
    hf_kernel_default = list(ref_attn.get("i4_attn_out_pre_oproj") or [])
    vllm_kernel = list(v_taps.get("i4_attn_out_pre_oproj") or [])

    def _cmp(a: list[float], b: list[float]) -> dict[str, Any]:
        if not a or not b or len(a) != len(b):
            return {
                "status": "shape_mismatch",
                "a_len": len(a) if a else 0,
                "b_len": len(b) if b else 0,
            }
        return {
            "status": "ok",
            "len": len(a),
            "cosine": float(_cosine(a, b)),
            "l2_diff": _l2_diff(a, b),
            "max_abs_diff": _max_abs_diff(a, b),
            "a_norm": math.sqrt(sum(x * x for x in a)),
            "b_norm": math.sqrt(sum(x * x for x in b)),
        }

    per_variant: dict[str, Any] = {}
    matches_default: list[tuple[str, float]] = []
    matches_vllm: list[tuple[str, float]] = []
    matches_manual: list[tuple[str, float]] = []

    for name, vec_any in variants.items():
        if name == "_stats":
            continue
        if not isinstance(vec_any, list):
            continue
        vec = list(vec_any)
        row_report = {
            "vs_hf_manual_fp32": _cmp(vec, hf_manual_fp32),
            "vs_hf_kernel_default": _cmp(vec, hf_kernel_default),
            "vs_vllm_kernel": _cmp(vec, vllm_kernel),
        }
        if variant_stats and isinstance(variant_stats, dict):
            row_report["stats"] = variant_stats.get(name)
        per_variant[name] = row_report

        # Track which variant is close to each anchor.
        for anchor_name, anchor_vec, bucket in (
            ("default", hf_kernel_default, matches_default),
            ("vllm", vllm_kernel, matches_vllm),
            ("manual", hf_manual_fp32, matches_manual),
        ):
            if not anchor_vec or not vec or len(anchor_vec) != len(vec):
                continue
            cos = float(_cosine(vec, anchor_vec))
            if cos >= cosine_threshold:
                bucket.append((name, cos))

    # Anchor-anchor comparisons for context (same numbers as Test J but
    # surfaced here so this report is self-contained).
    anchor_report = {
        "hf_manual_fp32_vs_hf_kernel_default": _cmp(
            hf_manual_fp32, hf_kernel_default
        ),
        "hf_manual_fp32_vs_vllm_kernel": _cmp(hf_manual_fp32, vllm_kernel),
        "hf_kernel_default_vs_vllm_kernel": _cmp(
            hf_kernel_default, vllm_kernel
        ),
    }

    verdict: dict[str, Any] = {
        "variants_matching_hf_kernel_default": [
            {"variant": n, "cosine": c} for n, c in matches_default
        ],
        "variants_matching_vllm_kernel": [
            {"variant": n, "cosine": c} for n, c in matches_vllm
        ],
        "variants_matching_hf_manual_fp32": [
            {"variant": n, "cosine": c} for n, c in matches_manual
        ],
    }
    # Human-readable interpretation hints.
    if matches_default and not matches_vllm:
        verdict["interpretation_hint"] = (
            "At least one variant reproduces HF's default kernel output but"
            " none match the vLLM / textbook SDPA output; this suggests the"
            " HF default kernel is systematically deviating (backend"
            " selection and/or bf16 precision) from textbook SDPA, while"
            " vLLM's flash-attn matches textbook."
        )
    elif matches_vllm and not matches_default:
        verdict["interpretation_hint"] = (
            "Variants match vLLM / textbook SDPA but not HF's default"
            " kernel; this isolates HF's default backend as the outlier."
        )
    elif matches_vllm and matches_default:
        verdict["interpretation_hint"] = (
            "Some variants match HF default, others match vLLM / textbook;"
            " compare which variant belongs to which backend/dtype bucket"
            " to identify the attenuation source."
        )
    else:
        verdict["interpretation_hint"] = (
            "No variant passes the cosine threshold against any anchor;"
            " inspect per_variant numbers for partial matches."
        )

    metadata = reference.get("test_k_hf_sdpa_metadata") or {}

    return {
        "cosine_threshold": float(cosine_threshold),
        "anchors_present": {
            "hf_manual_fp32": bool(hf_manual_fp32),
            "hf_kernel_default": bool(hf_kernel_default),
            "vllm_kernel": bool(vllm_kernel),
        },
        "anchor_pairwise": anchor_report,
        "per_variant": per_variant,
        "verdict": verdict,
        "hf_sdpa_metadata": metadata,
        "variant_stats": variant_stats,
    }


# ---------------------------------------------------------------------------
# Test L: multi-step layer-1 intra-self_attn probe on the vLLM side.
#
# The probe installs *on top of* the legacy single-shot probe
# (``install_vllm_draft_layer1_attn_bisect_capture``) so both coexist: the
# legacy probe captures at step 0 only (its ``captured_attn`` flag
# short-circuits further calls), and the Test-L probe captures at every
# speculative iteration index in ``steps_to_capture``.
#
# Iteration index is driven by ``precompute_and_store_context_kv`` call
# count (that method is invoked exactly once per speculative draft pass,
# before the layer stack runs).  Inside the patched ``self_attn.forward``,
# the current iteration's ``_current_slot`` is read so the capture stays
# synchronized: precompute sets the slot for iteration N, then layer-idx
# ``bisect_layer_idx``'s self_attn.forward fires during iteration N with
# that same slot active.
# ---------------------------------------------------------------------------


def install_vllm_test_l_probe(
    worker: Any,
    steps_to_capture: Any,
    bisect_layer_idx: int = 1,
    d0_row: int = 1,
    req_idx: int = 0,
    num_query_per_req: int | None = None,
) -> dict[str, Any]:
    """Install the Test L multi-step probe.

    Parameters
    ----------
    worker : vLLM worker (kw-less call via ``collective_rpc``).
    steps_to_capture : iterable of ints or tuple, the 0-indexed
        speculative-iteration counts at which to snapshot.  E.g.
        ``(0, 5, 15, 30)``.  Values beyond the actual iteration count are
        silently ignored (no snapshot recorded).
    bisect_layer_idx : which draft decoder layer's self_attn to probe
        (default 1, to match Test I).
    d0_row : row-inside-block to tap (1 in the current block layout).
    req_idx, num_query_per_req : same semantics as the legacy attn-bisect
        installer; used to compute the effective d0 row in a batched
        context.

    Returns a status dict, and stashes the running capture state at
    ``worker._draft_quality_test_l_probe``.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    worker_obj = getattr(worker, "worker", worker)
    model_runner = getattr(worker_obj, "model_runner", None)
    if model_runner is None:
        return {"error": "no model_runner"}
    drafter = getattr(model_runner, "drafter", None)
    if drafter is None:
        return {"error": "no drafter"}
    draft_model = getattr(drafter, "model", None)
    if draft_model is None:
        return {"error": "no drafter.model"}
    draft_inner = getattr(draft_model, "model", None)
    if draft_inner is None:
        return {"error": "no drafter.model.model"}
    layers = getattr(draft_inner, "layers", None)
    if layers is None or bisect_layer_idx >= len(layers):
        return {
            "error": f"invalid bisect_layer_idx={bisect_layer_idx}"
        }
    layer = layers[bisect_layer_idx]
    self_attn = getattr(layer, "self_attn", None)
    if self_attn is None:
        return {"error": "no self_attn on target layer"}

    if num_query_per_req is None:
        try:
            num_query_per_req = int(1 + drafter.num_speculative_tokens)
        except Exception:
            num_query_per_req = None
    effective_d0_row = (
        int(req_idx) * int(num_query_per_req) + int(d0_row)
        if num_query_per_req is not None
        else int(d0_row)
    )

    try:
        _steps_set = {int(s) for s in steps_to_capture}
    except Exception:
        return {"error": f"invalid steps_to_capture={steps_to_capture!r}"}

    state: dict[str, Any] = {
        "bisect_layer_idx": int(bisect_layer_idx),
        "d0_row": int(effective_d0_row),
        "num_query_per_req": num_query_per_req,
        "steps_to_capture": sorted(_steps_set),
        "num_precompute_calls": 0,
        "num_self_attn_calls": 0,
        "per_step": [],
        "_current_slot": None,
        "original_self_attn_forward": self_attn.forward,
        "original_precompute": getattr(
            draft_inner, "precompute_and_store_context_kv", None
        ),
        "self_attn": self_attn,
        "draft_inner": draft_inner,
    }

    def _flat_to_cpu(t: Any) -> list[float] | None:
        if t is None or not isinstance(t, torch.Tensor):
            return None
        return t.detach().reshape(-1).cpu().float().tolist()

    def _row_to_cpu(t: Any, row: int) -> list[float] | None:
        if t is None or not isinstance(t, torch.Tensor):
            return None
        if t.dim() < 2 or row < 0 or row >= t.shape[0]:
            return None
        return t[row].detach().cpu().float().tolist()

    def _tensor_stats(t: Any) -> dict[str, Any] | None:
        if t is None or not isinstance(t, torch.Tensor):
            return None
        tf = t.detach().float()
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "norm": float(tf.norm().item()),
            "abs_mean": float(tf.abs().mean().item()),
        }

    def _per_position_stats(t: Any, k_first: int = 32) -> dict[str, Any] | None:
        """Compact per-position fingerprint for Test M.

        Given a (num_positions, kv_dim) or (num_positions, nkv, hd) tensor,
        flatten to (num_positions, kv_dim) and return dict with parallel
        lists: norm[p], abs_mean[p], first<k>[p].  JSON-size is small enough
        that we can persist it at every TEST_L_STEP.
        """
        if t is None or not isinstance(t, torch.Tensor):
            return None
        tf = t.detach().float()
        if tf.dim() > 2:
            tf = tf.reshape(tf.shape[0], -1)
        elif tf.dim() == 1:
            tf = tf.unsqueeze(0)
        per_pos_norm = tf.norm(dim=-1).cpu().tolist()
        per_pos_abs_mean = tf.abs().mean(dim=-1).cpu().tolist()
        k = min(int(k_first), int(tf.shape[-1]))
        first_k = tf[:, :k].cpu().tolist()
        return {
            "num_positions": int(tf.shape[0]),
            "kv_dim": int(tf.shape[-1]),
            "first_k_width": int(k),
            "norm": per_pos_norm,
            "abs_mean": per_pos_abs_mean,
            "first_k": first_k,
        }

    original_self_attn_forward = state["original_self_attn_forward"]
    original_precompute = state["original_precompute"]

    def patched_precompute(
        context_states,
        context_positions,
        context_slot_mapping=None,
    ):
        current_step = state["num_precompute_calls"]
        state["num_precompute_calls"] += 1
        should_capture = current_step in _steps_set
        if not should_capture or original_precompute is None:
            state["_current_slot"] = None
            if original_precompute is None:
                return None
            return original_precompute(
                context_states, context_positions, context_slot_mapping
            )

        # Replicate the precompute internals so we can snapshot the
        # last-context K/V (pre- and post-RoPE) before the underlying
        # original_precompute mutates the paged KV cache.  This mirrors
        # the legacy attn-bisect installer's patched_precompute; Test L
        # just additionally keys the snapshot by iteration index.
        slot: dict[str, Any] = {
            "step": int(current_step),
            "context_len": int(context_states.shape[0]),
            "attn_taps": {},
            "attn_stats": {},
            "attn_shapes": {},
            "ctx_kv": {},
            "ctx_kv_stats": {},
        }
        state["_current_slot"] = slot
        state["per_step"].append(slot)

        try:
            di = draft_inner
            if not hasattr(di, "_num_attn_layers"):
                di._build_fused_kv_buffers()
            num_ctx = context_states.shape[0]
            L = di._num_attn_layers
            kv = di._kv_size
            hd = di._head_dim
            nkv = di._num_kv_heads
            normed_context_states = torch.empty_like(context_states)
            from vllm import _custom_ops as ops  # noqa: PLC0415

            ops.rms_norm(
                normed_context_states,
                context_states,
                di._hidden_norm_weight,
                di._rms_norm_eps,
            )
            all_kv_flat = F.linear(
                normed_context_states, di._fused_kv_weight, di._fused_kv_bias
            )
            all_kv = (
                all_kv_flat.view(num_ctx, L, 2, nkv, hd)
                .permute(2, 1, 0, 3, 4)
                .contiguous()
            )
            all_k = all_kv[0]
            all_v = all_kv[1]
            all_k_normed = torch.empty_like(all_k)
            for i in range(L):
                ops.rms_norm(
                    all_k_normed[i],
                    all_k[i],
                    di._k_norm_weights[i],
                    di._rms_norm_eps,
                )
            last_ctx = num_ctx - 1
            if 0 <= bisect_layer_idx < L and last_ctx >= 0:
                pre_rope_k = (
                    all_k_normed[bisect_layer_idx, last_ctx]
                    .detach()
                    .clone()
                )
                v_row = all_v[bisect_layer_idx, last_ctx].detach().clone()
                slot["ctx_kv"]["k_last_context_pre_rope"] = _flat_to_cpu(
                    pre_rope_k
                )
                slot["ctx_kv"]["v_last_context"] = _flat_to_cpu(v_row)
                slot["ctx_kv_stats"]["k_last_context_pre_rope"] = (
                    _tensor_stats(pre_rope_k)
                )
                slot["ctx_kv_stats"]["v_last_context"] = _tensor_stats(v_row)
                slot["ctx_kv"]["layer_idx"] = int(bisect_layer_idx)
                slot["ctx_kv"]["last_context_index"] = int(last_ctx)
                slot["ctx_kv"]["num_context"] = int(num_ctx)
                # Test M: per-position fingerprint of pre-RoPE K and V
                # across the FULL context window (all num_ctx positions).
                # Lets the post-hoc alignment report detect (a) padding /
                # stale slots in vLLM tail positions and (b) per-position
                # drift across the HF/vLLM overlap region.
                slot["ctx_per_position"] = {
                    "k_ctx_pre_rope": _per_position_stats(
                        all_k_normed[bisect_layer_idx]
                    ),
                    "v_ctx": _per_position_stats(all_v[bisect_layer_idx]),
                }
            # Test M: record the positional indices and slot mapping that
            # vLLM is feeding into ``precompute_and_store_context_kv``.  These
            # are the "ground truth" for what vLLM believes its context
            # window is at this speculative iteration.
            try:
                if isinstance(context_positions, torch.Tensor):
                    slot["context_positions"] = (
                        context_positions.detach().cpu().tolist()
                    )
            except Exception:
                pass
            try:
                if isinstance(context_slot_mapping, torch.Tensor):
                    slot["context_slot_mapping"] = (
                        context_slot_mapping.detach().cpu().tolist()
                    )
            except Exception:
                pass
            all_k_flat = all_k_normed.view(L * num_ctx, kv)
            positions_repeated = context_positions.repeat(L)
            cos_sin_cache = di._rope_cos_sin_cache
            if cos_sin_cache.dtype != all_k_flat.dtype:
                cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
            ops.rotary_embedding(
                positions_repeated,
                all_k_flat,
                None,
                di._rope_head_size,
                cos_sin_cache,
                di._rope_is_neox,
            )
            all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
            if 0 <= bisect_layer_idx < L and last_ctx >= 0:
                k_row = (
                    all_k_final[bisect_layer_idx, last_ctx].detach().clone()
                )
                slot["ctx_kv"]["k_last_context"] = _flat_to_cpu(k_row)
                slot["ctx_kv_stats"]["k_last_context"] = _tensor_stats(k_row)
                # Test M: post-RoPE K fingerprint across full context.
                cpp = slot.get("ctx_per_position") or {}
                cpp["k_ctx_post_rope"] = _per_position_stats(
                    all_k_final[bisect_layer_idx]
                )
                slot["ctx_per_position"] = cpp

            if context_slot_mapping is not None:
                for i in range(L):
                    attn = di._attn_layers[i]
                    kv_cache = attn.kv_cache
                    attn.impl.do_kv_cache_update(
                        attn,
                        all_k_final[i],
                        all_v[i],
                        kv_cache,
                        context_slot_mapping,
                    )
            return None
        except Exception as _e:
            slot["precompute_error"] = str(_e)
            # Fall back to the untouched original so the run keeps going.
            return original_precompute(
                context_states, context_positions, context_slot_mapping
            )

    def patched_self_attn_forward(positions, hidden_states):
        state["num_self_attn_calls"] += 1
        slot = state.get("_current_slot")
        if slot is None:
            return original_self_attn_forward(positions, hidden_states)

        # Replicate the Q/K/V + RoPE + attention math so we can tap
        # intermediate values.  The actual PAGED ATTENTION call uses
        # ``self_attn.attn(...)`` just like the legacy probe.
        row = state["d0_row"]
        try:
            qkv = F.linear(
                hidden_states,
                self_attn.qkv_proj.weight,
                self_attn.qkv_proj.bias,
            )
            q, k, v = qkv.split(
                [self_attn.q_size, self_attn.kv_size, self_attn.kv_size],
                dim=-1,
            )
            slot["attn_taps"]["i1_q_post_qproj"] = _row_to_cpu(q, row)
            slot["attn_taps"]["k_noise_post_kproj"] = _row_to_cpu(k, row)
            slot["attn_taps"]["v_noise_post_vproj"] = _row_to_cpu(v, row)
            slot["attn_stats"]["i1_q_post_qproj"] = _tensor_stats(q)
            slot["attn_stats"]["k_noise_post_kproj"] = _tensor_stats(k)
            slot["attn_stats"]["v_noise_post_vproj"] = _tensor_stats(v)
            slot["attn_shapes"]["i1_q_post_qproj"] = list(q.shape)
            q_shape, k_shape = q.shape, k.shape
            q_normed = self_attn.q_norm(
                q.view(
                    *q_shape[:-1],
                    q_shape[-1] // self_attn.head_dim,
                    self_attn.head_dim,
                )
            ).view(q_shape)
            k_normed = self_attn.k_norm(
                k.view(
                    *k_shape[:-1],
                    k_shape[-1] // self_attn.head_dim,
                    self_attn.head_dim,
                )
            ).view(k_shape)
            slot["attn_taps"]["i2_q_post_qnorm"] = _row_to_cpu(q_normed, row)
            slot["attn_taps"]["k_noise_post_knorm"] = _row_to_cpu(
                k_normed, row
            )
            slot["attn_stats"]["i2_q_post_qnorm"] = _tensor_stats(q_normed)
            slot["attn_stats"]["k_noise_post_knorm"] = _tensor_stats(k_normed)
            q_rot, k_rot = self_attn.rotary_emb(positions, q_normed, k_normed)
            slot["attn_taps"]["i3_q_post_rope"] = _row_to_cpu(q_rot, row)
            slot["attn_taps"]["k_noise_post_rope"] = _row_to_cpu(k_rot, row)
            slot["attn_stats"]["i3_q_post_rope"] = _tensor_stats(q_rot)
            slot["attn_stats"]["k_noise_post_rope"] = _tensor_stats(k_rot)
            attn_output = self_attn.attn(q_rot, k_rot, v)
            slot["attn_taps"]["i4_attn_out_pre_oproj"] = _row_to_cpu(
                attn_output, row
            )
            slot["attn_stats"]["i4_attn_out_pre_oproj"] = _tensor_stats(
                attn_output
            )
            slot["attn_shapes"]["i4_attn_out_pre_oproj"] = list(
                attn_output.shape
            )
            output, _ = self_attn.o_proj(attn_output)
            slot["attn_taps"]["i5_attn_out_post_oproj"] = _row_to_cpu(
                output, row
            )
            slot["attn_stats"]["i5_attn_out_post_oproj"] = _tensor_stats(
                output
            )
            # Clear _current_slot so a repeated self_attn call in the same
            # iteration (unlikely, but safe) does not double-write.
            state["_current_slot"] = None
            return output
        except Exception as _e:
            slot["self_attn_error"] = str(_e)
            state["_current_slot"] = None
            return original_self_attn_forward(positions, hidden_states)

    self_attn.forward = patched_self_attn_forward
    if original_precompute is not None:
        draft_inner.precompute_and_store_context_kv = patched_precompute

    worker_obj._draft_quality_test_l_probe = state
    return {
        "status": "installed",
        "bisect_layer_idx": int(bisect_layer_idx),
        "d0_row": int(effective_d0_row),
        "steps_to_capture": sorted(_steps_set),
        "patched_precompute": original_precompute is not None,
    }


def retrieve_vllm_test_l_probe(worker: Any) -> dict[str, Any]:
    """Uninstall the Test L probe and return the per-step capture list."""
    worker_obj = getattr(worker, "worker", worker)
    state = getattr(worker_obj, "_draft_quality_test_l_probe", None)
    if state is None:
        return {"error": "no Test L probe state"}
    # Restore the patched methods.
    try:
        self_attn = state.get("self_attn")
        orig_forward = state.get("original_self_attn_forward")
        if self_attn is not None and orig_forward is not None:
            self_attn.forward = orig_forward
    except Exception:
        pass
    try:
        draft_inner = state.get("draft_inner")
        orig_precompute = state.get("original_precompute")
        if draft_inner is not None and orig_precompute is not None:
            draft_inner.precompute_and_store_context_kv = orig_precompute
    except Exception:
        pass
    out = {
        "bisect_layer_idx": state.get("bisect_layer_idx"),
        "d0_row": state.get("d0_row"),
        "num_query_per_req": state.get("num_query_per_req"),
        "steps_to_capture": state.get("steps_to_capture"),
        "num_precompute_calls": state.get("num_precompute_calls"),
        "num_self_attn_calls": state.get("num_self_attn_calls"),
        "per_step": state.get("per_step") or [],
    }
    try:
        delattr(worker_obj, "_draft_quality_test_l_probe")
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Test Q: layer-0 hidden-state probe.
#
# Complement to Test P.  Test P showed that at matched decoded positions
# (identical accepted prefix on both sides) the tree-spec drafter's layer-1
# self_attn output diverges from the chain-spec drafter's, and that the
# divergence is already present at ``i1_q_post_qproj`` -- the very first
# tap after layer-1's input.  Since ``i1_q_post_qproj = F.linear(
# layer0_output[d0_row], W_q)`` depends only on layer-0 output and fixed
# weights, this localizes the root cause at or upstream of layer 0.
#
# Test Q directly taps layer 0 of the draft decoder:
#   * layer0 input (hidden_states + residual at layer_0.forward entry) --
#     the residual-stream state the drafter hands off to layer 0.
#   * layer0 self_attn_out (output of layers[0].self_attn, pre-residual-
#     add, pre-post_attention_layernorm) -- what the paged-attention kernel
#     produces at layer 0, before the MLP.
#   * layer0 output (hidden_states + residual at layer_0.forward exit) --
#     = layer-1 input.
#
# Paired with Test P's reconstruction of decoded positions, we can then
# answer:
#   1. Does layer-0 input already differ (cos < ~1)?  -> plumbing bug
#      upstream of layer 0 (embedding / residual / positions array).
#   2. Does layer-0 self_attn_out differ while input matches?  -> layer-0
#      self_attn is the source (paged-K/V contents, attention mask, or
#      slot indices that the kernel reads).
#   3. Does layer-0 output differ while self_attn_out matches?  -> MLP is
#      the source (very unlikely; MLP is a deterministic function of its
#      pre-MLP norm input and fixed weights).
# ---------------------------------------------------------------------------


def install_vllm_test_q_probe(
    worker: Any,
    steps_to_capture: Any,
    d0_row: int = 1,
    req_idx: int = 0,
    num_query_per_req: int | None = None,
    layer_idx: int = 0,
    capture_rows: Any | None = None,
) -> dict[str, Any]:
    """Install the Test Q layer-0 hidden-state probe.

    Uses ``register_forward_pre_hook`` / ``register_forward_hook`` on the
    draft model's ``layers[layer_idx]`` and ``layers[layer_idx].self_attn``,
    which is non-invasive (no monkey-patching of ``forward``) and therefore
    coexists cleanly with the Test L probe and the chain-spec topk probe.

    Parameters
    ----------
    steps_to_capture : iterable of 0-indexed speculative iteration indices.
        ``num_model_forward_calls`` is used as the step counter, which
        matches Test L's ``num_precompute_calls`` under standard DFlash
        semantics (one precompute + one model forward per iter).
    d0_row : row-inside-block to tap (1 in the current block layout).
    req_idx, num_query_per_req : same batched-row semantics as Test L.
    layer_idx : default 0; change to run the probe on a different draft
        layer.  (Test Q's interpretation hints assume layer 0.)
    """
    import torch  # noqa: PLC0415
    from vllm.forward_context import (  # noqa: PLC0415
        get_forward_context,
        is_forward_context_available,
    )

    worker_obj = getattr(worker, "worker", worker)
    model_runner = getattr(worker_obj, "model_runner", None)
    if model_runner is None:
        return {"error": "no model_runner"}
    drafter = getattr(model_runner, "drafter", None)
    if drafter is None:
        return {"error": "no drafter"}
    draft_model = getattr(drafter, "model", None)
    if draft_model is None:
        return {"error": "no drafter.model"}
    draft_inner = getattr(draft_model, "model", None)
    if draft_inner is None:
        return {"error": "no drafter.model.model"}
    layers = getattr(draft_inner, "layers", None)
    if layers is None or layer_idx >= len(layers):
        return {"error": f"invalid layer_idx={layer_idx}"}
    layer = layers[layer_idx]
    self_attn = getattr(layer, "self_attn", None)
    if self_attn is None:
        return {"error": "no self_attn on target layer"}

    if num_query_per_req is None:
        try:
            num_query_per_req = int(1 + drafter.num_speculative_tokens)
        except Exception:
            num_query_per_req = None
    effective_d0_row = (
        int(req_idx) * int(num_query_per_req) + int(d0_row)
        if num_query_per_req is not None
        else int(d0_row)
    )
    if capture_rows is None:
        effective_capture_rows = [int(effective_d0_row)]
    else:
        try:
            local_rows = [int(r) for r in capture_rows]
        except Exception:
            return {"error": f"invalid capture_rows={capture_rows!r}"}
        effective_capture_rows = [
            (
                int(req_idx) * int(num_query_per_req) + int(r)
                if num_query_per_req is not None
                else int(r)
            )
            for r in local_rows
        ]
        if int(effective_d0_row) not in effective_capture_rows:
            effective_capture_rows = [int(effective_d0_row), *effective_capture_rows]
    effective_capture_rows = list(dict.fromkeys(effective_capture_rows))

    try:
        _steps_set = {int(s) for s in steps_to_capture}
    except Exception:
        return {"error": f"invalid steps_to_capture={steps_to_capture!r}"}

    state: dict[str, Any] = {
        "layer_idx": int(layer_idx),
        "d0_row": int(effective_d0_row),
        "capture_rows": effective_capture_rows,
        "num_query_per_req": num_query_per_req,
        "steps_to_capture": sorted(_steps_set),
        "num_layer_forward_calls": 0,
        "per_step": [],
        "_current_slot": None,
        "_pending_model_inputs": {},
        "hook_handles": [],
        "draft_model": draft_model,
        "draft_inner": draft_inner,
        "layer": layer,
        "self_attn": self_attn,
    }

    def _row_to_cpu(t: Any, row: int) -> list[float] | None:
        if t is None or not isinstance(t, torch.Tensor):
            return None
        if t.dim() < 2:
            # 1-D tensor: treat as single row if row==0.
            if row == 0:
                return t.detach().cpu().float().tolist()
            return None
        if row < 0 or row >= t.shape[0]:
            return None
        return t[row].detach().cpu().float().tolist()

    def _tensor_shape_meta(t: Any) -> dict[str, Any] | None:
        """Record only shape+dtype (no device sync, no full-tensor copy).

        We intentionally avoid ``.norm().item()`` / ``.abs().mean().item()``
        style stats here: ``.float()`` on the whole tensor allocates an
        O(numel) FP32 copy of a potentially shared/paged buffer, and the
        ``.item()`` syncs can race with the tree-spec sampler's in-place
        writes to adjacent buffers, showing up as a CUDA illegal memory
        access at a later boolean kernel (observed empirically in
        run_20260419_18{01,10}* -- symptom at ``gpu_tree_accept``'s
        ``prefix_match & anc_match``).
        The per-row vector captured via ``_row_to_cpu`` is sufficient for
        Test Q's cosine comparisons.
        """
        if t is None or not isinstance(t, torch.Tensor):
            return None
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
        }

    def _row_scalar(t: Any, row: int) -> int | float | None:
        if t is None or not isinstance(t, torch.Tensor):
            return None
        if t.dim() == 0:
            if row == 0:
                val = t.detach().cpu().item()
                if isinstance(val, bool):
                    return int(val)
                if isinstance(val, int):
                    return int(val)
                return float(val)
            return None
        if row < 0 or row >= t.shape[0]:
            return None
        val = t[row].detach().cpu().item()
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, int):
            return int(val)
        if isinstance(val, float):
            if val.is_integer():
                return int(val)
            return float(val)
        try:
            return float(val)
        except Exception:
            return None

    def _capture_rows_into_slot(
        slot: dict[str, Any],
        tap_name: str,
        tensor: Any,
    ) -> None:
        row_stats = _tensor_shape_meta(tensor)
        slot.setdefault("per_row_attn_taps", {})
        slot.setdefault("per_row_attn_stats", {})
        for row in state["capture_rows"]:
            row_key = str(int(row))
            slot["per_row_attn_taps"].setdefault(row_key, {})
            slot["per_row_attn_stats"].setdefault(row_key, {})
            slot["per_row_attn_taps"][row_key][tap_name] = _row_to_cpu(
                tensor, int(row)
            )
            slot["per_row_attn_stats"][row_key][tap_name] = row_stats
        slot["attn_taps"][tap_name] = _row_to_cpu(tensor, int(state["d0_row"]))
        slot["attn_stats"][tap_name] = row_stats

    def _capture_row_input_meta(
        slot: dict[str, Any],
        positions_tensor: Any,
    ) -> None:
        input_ids_gpu = getattr(getattr(model_runner, "input_ids", None), "gpu", None)
        slot.setdefault("per_row_meta", {})
        slot.setdefault("meta", {})
        for row in state["capture_rows"]:
            row_key = str(int(row))
            row_meta = {
                "global_row": int(row),
                "query_input_id": _row_scalar(input_ids_gpu, int(row)),
                "query_position": _row_scalar(positions_tensor, int(row)),
            }
            if num_query_per_req is not None:
                local_row = int(row) - int(req_idx) * int(num_query_per_req)
                row_meta["local_row"] = int(local_row)
            slot["per_row_meta"][row_key] = row_meta
        d0_key = str(int(state["d0_row"]))
        if d0_key in slot["per_row_meta"]:
            slot["meta"] = dict(slot["per_row_meta"][d0_key])

    def _capture_actual_forward_meta(
        slot: dict[str, Any],
        input_ids_tensor: Any,
        positions_tensor: Any,
    ) -> None:
        slot.setdefault("per_row_actual_meta", {})
        slot.setdefault("actual_meta", {})
        for row in state["capture_rows"]:
            row_key = str(int(row))
            row_meta = {
                "global_row": int(row),
                "query_input_id": _row_scalar(input_ids_tensor, int(row)),
                "query_position": _row_scalar(positions_tensor, int(row)),
            }
            if num_query_per_req is not None:
                local_row = int(row) - int(req_idx) * int(num_query_per_req)
                row_meta["local_row"] = int(local_row)
            slot["per_row_actual_meta"][row_key] = row_meta
        d0_key = str(int(state["d0_row"]))
        if d0_key in slot["per_row_actual_meta"]:
            slot["actual_meta"] = dict(slot["per_row_actual_meta"][d0_key])

    def _capture_forward_attn_meta(slot: dict[str, Any]) -> None:
        slot["forward_attn_meta"] = {}
        if not is_forward_context_available():
            slot["forward_attn_meta"] = {"available": False}
            return None
        try:
            fc = get_forward_context()
        except Exception as _e:
            slot["forward_attn_meta"] = {
                "available": False,
                "error": str(_e),
            }
            return None

        def _unwrap_mapping(obj: Any) -> dict[str, Any] | None:
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list) and obj:
                for item in obj:
                    if isinstance(item, dict) and item:
                        return item
            return None

        def _resolve_by_candidates(mapping: dict[str, Any] | None, candidates: list[str]) -> Any:
            if not isinstance(mapping, dict) or not mapping:
                return None
            for cand in candidates:
                if cand in mapping:
                    return mapping[cand]
            for cand in candidates:
                for k, v in mapping.items():
                    if isinstance(k, str) and (
                        k.endswith(cand) or cand.endswith(k)
                    ):
                        return v
            if len(mapping) == 1:
                return next(iter(mapping.values()))
            return None

        all_meta = _unwrap_mapping(getattr(fc, "attn_metadata", None))
        all_slot = _unwrap_mapping(getattr(fc, "slot_mapping", None))
        outer_layer_name = getattr(self_attn, "layer_name", None)
        inner_attn_mod = getattr(self_attn, "attn", None)
        inner_layer_name = getattr(inner_attn_mod, "layer_name", None)
        candidate_layer_names = [
            str(x)
            for x in (
                inner_layer_name,
                outer_layer_name,
                f"{outer_layer_name}.attn" if outer_layer_name is not None else None,
                f"{inner_layer_name}.attn" if inner_layer_name is not None else None,
            )
            if x is not None
        ]
        attn_meta_obj: Any = _resolve_by_candidates(
            all_meta, candidate_layer_names
        )
        slot_map_obj: Any = _resolve_by_candidates(
            all_slot, candidate_layer_names
        )
        slot["forward_attn_meta"] = {
            "available": attn_meta_obj is not None,
            "layer_name": outer_layer_name,
            "inner_attn_layer_name": inner_layer_name,
            "candidate_layer_names": candidate_layer_names,
            "attn_metadata_keys_head": (
                list(all_meta.keys())[:8] if isinstance(all_meta, dict) else None
            ),
            "slot_mapping_keys_head": (
                list(all_slot.keys())[:8] if isinstance(all_slot, dict) else None
            ),
            "metadata_type": (
                type(attn_meta_obj).__name__ if attn_meta_obj is not None else None
            ),
            "causal": getattr(attn_meta_obj, "causal", None),
            "num_actual_tokens": getattr(attn_meta_obj, "num_actual_tokens", None),
            "max_query_len": getattr(attn_meta_obj, "max_query_len", None),
            "max_seq_len": getattr(attn_meta_obj, "max_seq_len", None),
            "seq_lens": (
                attn_meta_obj.seq_lens.detach().cpu().tolist()
                if getattr(attn_meta_obj, "seq_lens", None) is not None
                and isinstance(attn_meta_obj.seq_lens, torch.Tensor)
                else None
            ),
            "query_start_loc": (
                attn_meta_obj.query_start_loc.detach().cpu().tolist()
                if getattr(attn_meta_obj, "query_start_loc", None) is not None
                and isinstance(attn_meta_obj.query_start_loc, torch.Tensor)
                else None
            ),
            "slot_mapping": (
                slot_map_obj.detach().cpu().tolist()
                if slot_map_obj is not None and isinstance(slot_map_obj, torch.Tensor)
                else None
            ),
        }
        return None

    def _extract_layer_args(
        args: tuple, kwargs: dict
    ) -> tuple[Any, Any, Any] | None:
        """Return (positions, hidden_states, residual) from the actual
        call site.  DFlashQwen3Model calls its decoder layers via keyword
        arguments (``layer(positions=..., hidden_states=..., residual=...)``),
        so ``args`` is typically ``()`` and all three values live in
        ``kwargs``.  We also support the positional-only call shape for
        robustness against upstream refactors.
        """
        if kwargs:
            if (
                "hidden_states" in kwargs
                and "residual" in kwargs
            ):
                return (
                    kwargs.get("positions"),
                    kwargs["hidden_states"],
                    kwargs["residual"],
                )
        if len(args) >= 3:
            return (args[0], args[1], args[2])
        if len(args) == 2:
            return (None, args[0], args[1])
        return None

    def _extract_model_args(
        args: tuple, kwargs: dict
    ) -> tuple[Any, Any] | None:
        """Return (input_ids, positions) from draft_inner.forward."""
        if kwargs:
            if "input_ids" in kwargs and "positions" in kwargs:
                return (kwargs["input_ids"], kwargs["positions"])
        if len(args) >= 2:
            return (args[0], args[1])
        return None

    def model_pre_hook(module, args, kwargs):  # noqa: ARG001
        current_step = int(state["num_layer_forward_calls"])
        if current_step not in _steps_set:
            return None
        extracted = _extract_model_args(args, kwargs)
        if extracted is None:
            return None
        input_ids_tensor, positions_tensor = extracted
        state["_pending_model_inputs"][current_step] = {
            "input_ids": input_ids_tensor,
            "positions": positions_tensor,
        }
        return None

    def layer_pre_hook(module, args, kwargs):  # noqa: ARG001
        current_step = int(state["num_layer_forward_calls"])
        state["num_layer_forward_calls"] += 1
        if current_step not in _steps_set:
            state["_current_slot"] = None
            return None
        # layer.forward signature: (positions, hidden_states, residual).
        # DFlashQwen3Model invokes layers with KEYWORD args, so we read
        # from ``kwargs`` first (see ``_extract_layer_args``).
        extracted = _extract_layer_args(args, kwargs)
        if extracted is None:
            state["_current_slot"] = None
            return None
        _positions, hidden_states, residual = extracted
        row = int(state["d0_row"])
        slot: dict[str, Any] = {
            "step": current_step,
            "tree_propose_step": int(
                getattr(drafter, "_tree_propose_step", current_step)
            ),
            "layer_idx": int(layer_idx),
            "attn_taps": {},
            "attn_stats": {},
        }
        state["per_step"].append(slot)
        state["_current_slot"] = slot
        _capture_row_input_meta(slot, _positions)
        _capture_forward_attn_meta(slot)
        pending = (state.get("_pending_model_inputs") or {}).pop(
            current_step, None
        )
        if pending is not None:
            _capture_actual_forward_meta(
                slot,
                pending.get("input_ids"),
                pending.get("positions"),
            )
        _capture_rows_into_slot(slot, "layer0_input_hidden", hidden_states)
        if residual is not None:
            _capture_rows_into_slot(slot, "layer0_input_residual", residual)
        else:
            slot["attn_taps"]["layer0_input_residual"] = None
            slot["attn_stats"]["layer0_input_residual"] = {
                "shape": None,
                "dtype": None,
            }
            slot.setdefault("per_row_attn_taps", {})
            slot.setdefault("per_row_attn_stats", {})
            for extra_row in state["capture_rows"]:
                row_key = str(int(extra_row))
                slot["per_row_attn_taps"].setdefault(row_key, {})
                slot["per_row_attn_stats"].setdefault(row_key, {})
                slot["per_row_attn_taps"][row_key]["layer0_input_residual"] = None
                slot["per_row_attn_stats"][row_key]["layer0_input_residual"] = {
                    "shape": None,
                    "dtype": None,
                }
        return None

    def self_attn_post_hook(module, args, output):  # noqa: ARG001
        slot = state.get("_current_slot")
        if slot is None:
            return None
        _capture_rows_into_slot(slot, "layer0_self_attn_out", output)
        return None

    def layer_post_hook(module, args, output):  # noqa: ARG001
        slot = state.get("_current_slot")
        if slot is None:
            return None
        row = int(state["d0_row"])
        # layer.forward returns (hidden_states, residual).
        try:
            hidden_states, residual = output
        except Exception:
            state["_current_slot"] = None
            return None
        _capture_rows_into_slot(slot, "layer0_output_hidden", hidden_states)
        if residual is not None:
            _capture_rows_into_slot(slot, "layer0_output_residual", residual)
        else:
            slot["attn_taps"]["layer0_output_residual"] = None
            slot["attn_stats"]["layer0_output_residual"] = {
                "shape": None,
                "dtype": None,
            }
            slot.setdefault("per_row_attn_taps", {})
            slot.setdefault("per_row_attn_stats", {})
            for extra_row in state["capture_rows"]:
                row_key = str(int(extra_row))
                slot["per_row_attn_taps"].setdefault(row_key, {})
                slot["per_row_attn_stats"].setdefault(row_key, {})
                slot["per_row_attn_taps"][row_key]["layer0_output_residual"] = None
                slot["per_row_attn_stats"][row_key]["layer0_output_residual"] = {
                    "shape": None,
                    "dtype": None,
                }
        state["_current_slot"] = None
        return None

    # ``with_kwargs=True`` is REQUIRED for the pre-hook because
    # DFlashQwen3Model invokes its decoder layers with keyword arguments
    # (``layer(positions=..., hidden_states=..., residual=...)``); the
    # default ``with_kwargs=False`` would deliver ``args=()`` and the
    # extraction would fail, silently skipping every capture step.
    try:
        h_model_outer_pre = draft_model.register_forward_pre_hook(
            model_pre_hook, with_kwargs=True
        )
        h_model_pre = draft_inner.register_forward_pre_hook(
            model_pre_hook, with_kwargs=True
        )
        h_pre = layer.register_forward_pre_hook(
            layer_pre_hook, with_kwargs=True
        )
        h_post = layer.register_forward_hook(layer_post_hook)
        h_attn = self_attn.register_forward_hook(self_attn_post_hook)
    except Exception as _e:
        return {"error": f"register_forward_*hook failed: {_e}"}
    state["hook_handles"] = [
        h_model_outer_pre,
        h_model_pre,
        h_pre,
        h_post,
        h_attn,
    ]

    worker_obj._draft_quality_test_q_probe = state
    return {
        "status": "installed",
        "layer_idx": int(layer_idx),
        "d0_row": int(effective_d0_row),
        "capture_rows": effective_capture_rows,
        "steps_to_capture": sorted(_steps_set),
    }


def retrieve_vllm_test_q_probe(worker: Any) -> dict[str, Any]:
    """Uninstall Test Q probe and return per-step capture list."""
    worker_obj = getattr(worker, "worker", worker)
    state = getattr(worker_obj, "_draft_quality_test_q_probe", None)
    if state is None:
        return {"error": "no Test Q probe state"}
    for h in state.get("hook_handles") or []:
        try:
            h.remove()
        except Exception:
            pass
    out = {
        "layer_idx": state.get("layer_idx"),
        "d0_row": state.get("d0_row"),
        "capture_rows": state.get("capture_rows"),
        "num_query_per_req": state.get("num_query_per_req"),
        "steps_to_capture": state.get("steps_to_capture"),
        "num_layer_forward_calls": state.get("num_layer_forward_calls"),
        "per_step": state.get("per_step") or [],
    }
    try:
        delattr(worker_obj, "_draft_quality_test_q_probe")
    except Exception:
        pass
    return out


def install_vllm_dflash_runtime_bundle_probe(
    worker: Any,
    steps_to_capture: tuple[int, ...] | list[int] | None = None,
) -> dict[str, Any]:
    """Configure the DFlash proposer to retain runtime bundles at selected steps."""
    worker_obj = getattr(worker, "worker", worker)
    drafter = getattr(getattr(worker_obj, "model_runner", None), "drafter", None)
    if drafter is None:
        return {"error": "missing_model_runner_drafter"}
    clear_fn = getattr(drafter, "clear_runtime_bundles", None)
    if clear_fn is not None:
        try:
            clear_fn()
        except Exception:
            pass
    set_steps_fn = getattr(drafter, "set_runtime_capture_steps", None)
    get_steps_fn = getattr(drafter, "get_runtime_capture_steps", None)
    if set_steps_fn is None:
        return {"error": "drafter_missing_set_runtime_capture_steps"}
    set_steps_fn(steps_to_capture)
    configured = (
        get_steps_fn() if get_steps_fn is not None else list(steps_to_capture or [0])
    )
    return {
        "configured_steps": [int(s) for s in configured],
    }


def retrieve_vllm_dflash_runtime_bundle(worker: Any) -> dict[str, Any]:
    """Return the configured DFlash runtime bundles as JSON-safe data."""
    import torch  # noqa: PLC0415

    def _jsonify(val: Any) -> Any:
        if isinstance(val, torch.Tensor):
            t = val.detach().cpu()
            if t.ndim == 0:
                return t.item()
            return t.tolist()
        if isinstance(val, dict):
            return {str(k): _jsonify(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [_jsonify(v) for v in val]
        if isinstance(val, (str, int, float, bool)) or val is None:
            return val
        return str(val)

    worker_obj = getattr(worker, "worker", worker)
    drafter = getattr(getattr(worker_obj, "model_runner", None), "drafter", None)
    if drafter is None:
        return {"error": "missing_model_runner_drafter"}
    try:
        bundles = list(getattr(drafter, "get_runtime_bundles")() or [])
    except Exception as e:
        return {"error": f"get_runtime_bundles_failed: {e}"}
    configured_steps = []
    try:
        get_steps_fn = getattr(drafter, "get_runtime_capture_steps", None)
        if get_steps_fn is not None:
            configured_steps = [int(s) for s in (get_steps_fn() or [])]
    except Exception:
        configured_steps = []
    try:
        clear_fn = getattr(drafter, "clear_runtime_bundles", None)
        if clear_fn is not None:
            clear_fn()
    except Exception:
        pass
    if not bundles:
        return {"error": "no_runtime_bundle_captured"}
    keep_keys = [
        "step",
        "dflash_is_causal",
        "parallel_drafting_token_id",
        "target_token_ids",
        "target_positions",
        "next_token_ids",
        "context_positions",
        "context_slot_mapping",
        "query_input_ids",
        "query_positions",
        "token_indices_to_sample",
        "seq_lens",
        "num_query_per_req",
        "num_context",
        "num_rejected_tokens",
        "input_cad_query_start_loc",
        "input_cad_seq_lens",
        "input_context_lens_from_query_start_loc",
        "input_seq_lens_minus_rejected",
        "input_compacted_context_lens",
        "output_visible_context_lens",
        "output_query_start_loc",
        "output_seq_lens",
        "output_max_seq_len",
        "output_query_slot_mapping_head",
        "input_cad_max_query_len",
        "prepare_next_sampled_token_ids",
        "prepare_next_valid_sampled_tokens_count",
        "prepare_next_next_token_ids",
        "prepare_next_discard_request_mask",
        "prepare_next_backup_token_ids",
        "prepare_inputs_query_start_loc",
        "prepare_inputs_seq_lens",
        "prepare_inputs_valid_sampled_tokens_count",
        "prepare_inputs_cu_num_draft_tokens",
        "prepare_inputs_token_indices_to_sample",
        "prepare_inputs_num_rejected_tokens",
        "prepare_inputs_output_query_start_loc",
        "prepare_inputs_output_seq_lens",
        "builder_tree_budget",
        "builder_tree_num_nodes",
        "builder_tree_construction",
        "builder_score_mode",
        "builder_hybrid_alpha",
    ]
    out_bundles = []
    for raw_bundle in bundles:
        out_bundles.append(
            {
                key: _jsonify(raw_bundle.get(key))
                for key in keep_keys
                if key in raw_bundle
            }
        )
    return {
        "capture_steps_requested": configured_steps,
        "num_bundles_captured": len(out_bundles),
        "captured_steps": [
            int(b.get("step", -1))
            for b in out_bundles
            if b.get("step") is not None
        ],
        "bundles": out_bundles,
    }


def retrieve_vllm_tree_attn_builder_probe(worker: Any) -> dict[str, Any]:
    """Return TreeAttentionMetadataBuilder DFlash debug records as JSON-safe data."""
    import torch  # noqa: PLC0415

    def _jsonify(val: Any) -> Any:
        if isinstance(val, torch.Tensor):
            t = val.detach().cpu()
            if t.ndim == 0:
                return t.item()
            return t.tolist()
        if isinstance(val, dict):
            return {str(k): _jsonify(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [_jsonify(v) for v in val]
        if isinstance(val, (str, int, float, bool)) or val is None:
            return val
        return str(val)

    worker_obj = getattr(worker, "worker", worker)
    model_runner = getattr(worker_obj, "model_runner", None)
    if model_runner is None:
        return {"error": "missing_model_runner"}

    out_records: list[dict[str, Any]] = []
    sources: list[tuple[str, Any, bool]] = [
        ("target_model_runner", getattr(model_runner, "attn_groups", None), True),
        (
            "drafter",
            getattr(getattr(model_runner, "drafter", None), "draft_attn_groups", None),
            False,
        ),
    ]
    for owner_name, groups_root, indexed_builder in sources:
        if not isinstance(groups_root, list):
            continue
        for outer_idx, groups in enumerate(groups_root):
            iter_groups = groups if indexed_builder and isinstance(groups, list) else [groups]
            if not isinstance(iter_groups, list):
                continue
            for inner_idx, attn_group in enumerate(iter_groups):
                builder = None
                try:
                    if indexed_builder:
                        builder = attn_group.get_metadata_builder(0)
                    else:
                        builder = attn_group.get_metadata_builder()
                except Exception:
                    builder = None
                if builder is None or not hasattr(builder, "get_dflash_tree_debug_records"):
                    continue
                try:
                    raw_records = list(builder.get_dflash_tree_debug_records() or [])
                except Exception as e:
                    out_records.append(
                        {
                            "owner_name": owner_name,
                            "kv_cache_group_id": int(outer_idx) if indexed_builder else None,
                            "attn_group_id": int(inner_idx),
                            "error": f"get_dflash_tree_debug_records_failed: {e}",
                        }
                    )
                    continue
                for record in raw_records:
                    if not isinstance(record, dict):
                        continue
                    out_records.append(
                        {
                            "owner_name": owner_name,
                            "kv_cache_group_id": int(outer_idx) if indexed_builder else None,
                            "attn_group_id": int(inner_idx),
                            **_jsonify(record),
                        }
                    )
                try:
                    clear_fn = getattr(builder, "clear_dflash_tree_debug_records", None)
                    if clear_fn is not None:
                        clear_fn()
                except Exception:
                    pass

    if not out_records:
        return {"error": "no_tree_attn_builder_records_captured"}
    return {
        "num_records_captured": len(out_records),
        "captured_steps": [
            int(r.get("tree_propose_step", -1))
            for r in out_records
            if r.get("tree_propose_step") is not None
        ],
        "records": out_records,
    }


def retrieve_vllm_drafter_first_pass_metadata_probe(worker: Any) -> dict[str, Any]:
    """Return direct drafter first-pass CommonAttentionMetadata snapshots."""
    import torch  # noqa: PLC0415

    def _jsonify(val: Any) -> Any:
        if isinstance(val, torch.Tensor):
            t = val.detach().cpu()
            if t.ndim == 0:
                return t.item()
            return t.tolist()
        if isinstance(val, dict):
            return {str(k): _jsonify(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [_jsonify(v) for v in val]
        if isinstance(val, (str, int, float, bool)) or val is None:
            return val
        return str(val)

    worker_obj = getattr(worker, "worker", worker)
    drafter = getattr(getattr(worker_obj, "model_runner", None), "drafter", None)
    if drafter is None:
        return {"error": "missing_model_runner_drafter"}

    try:
        snapshots = list(getattr(drafter, "get_draft_first_pass_metadata_snapshots")() or [])
    except Exception as e:
        return {"error": f"get_draft_first_pass_metadata_snapshots_failed: {e}"}

    try:
        clear_fn = getattr(drafter, "clear_draft_first_pass_metadata_snapshots", None)
        if clear_fn is not None:
            clear_fn()
    except Exception:
        pass

    if not snapshots:
        return {"error": "no_drafter_first_pass_metadata_captured"}
    out = [_jsonify(s) for s in snapshots if isinstance(s, dict)]
    return {
        "num_snapshots_captured": len(out),
        "captured_steps": [
            int(s.get("tree_propose_step", -1))
            for s in out
            if s.get("tree_propose_step") is not None
        ],
        "snapshots": out,
    }


def build_draft_layer1_multistep_report(
    hf_test_l_payload: dict[str, Any],
    hf_reference_capture: dict[str, Any],
    vllm_capture: dict[str, Any],
    cosine_threshold: float = 0.999,
) -> dict[str, Any]:
    """Test L report -- multi-step HF vs vLLM parity at the bisect layer.

    Parameters
    ----------
    hf_test_l_payload : deserialized contents of
        ``reference_capture_test_l.json`` (HF-side per-step snapshots for
        steps > 0 of the reference sample).
    hf_reference_capture : deserialized contents of
        ``reference_capture.json`` (HF-side step-0 full capture; the
        Test-L report falls back to this for the step-0 comparison).
    vllm_capture : output of ``retrieve_vllm_test_l_probe`` (per-step
        vLLM snapshots).

    Produces:
    - ``per_step``: for each configured step, per-tap HF vs vLLM parity
      (cosine / l2_diff / max_abs_diff / norms) plus context_len on each
      side.
    - ``cross_step_drift_hf``: within-HF consistency check (does HF's
      self_attn_out stay at cos ~1 across all steps, as expected given
      teacher-forced context?).
    - ``cross_step_drift_vllm``: within-vLLM drift (does vLLM's
      self_attn_out cosine-to-step-0 decay with step?  If yes, confirms
      cross-iteration state contamination, i.e. Pγ / Pδ).
    - ``verdict``: one-line interpretation hint.
    """
    hf_per_step_raw: dict[str, Any] = hf_test_l_payload.get("per_step") or {}
    vllm_per_step = vllm_capture.get("per_step") or []

    # HF step-0 anchor comes from the legacy reference capture, not the
    # Test-L file (Test-L doesn't duplicate step-0 captures by default).
    ref_attn = hf_reference_capture.get("draft_layer1_attn_bisect_at_d0") or {}
    ref_ctx = hf_reference_capture.get("draft_layer1_context_kv") or {}

    # Build unified HF-side view indexed by int step.
    hf_by_step: dict[int, dict[str, Any]] = {}
    hf_step0_attn_taps: dict[str, list[float]] = {}
    hf_step0_ctx: dict[str, list[float]] = {}
    for k, v in ref_attn.items():
        if k in {"layer_idx", "_stats"} or not isinstance(v, list):
            continue
        hf_step0_attn_taps[k] = v
    for k, v in ref_ctx.items():
        if k in {"layer_idx", "last_context_index", "num_context", "_stats"}:
            continue
        if isinstance(v, list):
            hf_step0_ctx[k] = v
    if hf_step0_attn_taps or hf_step0_ctx:
        hf_by_step[0] = {
            "step": 0,
            "context_len": int(
                hf_reference_capture.get("num_prompt_tokens") or 0
            ),
            "attn_taps": hf_step0_attn_taps,
            "ctx_kv": hf_step0_ctx,
            "source": "reference_capture.json#draft_layer1_attn_bisect_at_d0",
        }
    for str_step, entry in hf_per_step_raw.items():
        try:
            s_idx = int(entry.get("step", str_step))
        except Exception:
            continue
        hf_by_step[s_idx] = {
            "step": s_idx,
            "context_len": int(entry.get("context_len", 0)),
            "attn_taps": entry.get("attn_taps") or {},
            "ctx_kv": entry.get("ctx_kv") or {},
            "source": "reference_capture_test_l.json",
        }

    # Build unified vLLM-side view indexed by int step.
    vllm_by_step: dict[int, dict[str, Any]] = {}
    for entry in vllm_per_step:
        try:
            s_idx = int(entry.get("step"))
        except Exception:
            continue
        vllm_by_step[s_idx] = {
            "step": s_idx,
            "context_len": int(entry.get("context_len", 0)),
            "attn_taps": entry.get("attn_taps") or {},
            "ctx_kv": entry.get("ctx_kv") or {},
            "attn_stats": entry.get("attn_stats") or {},
        }

    def _cmp(a: Any, b: Any) -> dict[str, Any]:
        if not isinstance(a, list) or not isinstance(b, list):
            return {"status": "missing"}
        if not a or not b or len(a) != len(b):
            return {
                "status": "shape_mismatch",
                "hf_len": len(a) if isinstance(a, list) else 0,
                "vllm_len": len(b) if isinstance(b, list) else 0,
            }
        return {
            "status": "ok",
            "len": len(a),
            "cosine": float(_cosine(a, b)),
            "l2_diff": _l2_diff(a, b),
            "max_abs_diff": _max_abs_diff(a, b),
            "hf_norm": math.sqrt(sum(x * x for x in a)),
            "vllm_norm": math.sqrt(sum(x * x for x in b)),
        }

    attn_tap_order = [
        "i1_q_post_qproj",
        "k_noise_post_kproj",
        "v_noise_post_vproj",
        "i2_q_post_qnorm",
        "k_noise_post_knorm",
        "i3_q_post_rope",
        "k_noise_post_rope",
        "i4_attn_out_pre_oproj",
        "i5_attn_out_post_oproj",
    ]
    ctx_key_order = [
        "k_last_context_pre_rope",
        "k_last_context",
        "v_last_context",
    ]

    per_step_report: dict[str, Any] = {}
    configured_steps = sorted(
        set(vllm_by_step.keys()) | set(hf_by_step.keys())
    )
    for s_idx in configured_steps:
        hf_slot = hf_by_step.get(s_idx)
        vllm_slot = vllm_by_step.get(s_idx)
        if hf_slot is None or vllm_slot is None:
            per_step_report[str(s_idx)] = {
                "skipped": True,
                "hf_present": hf_slot is not None,
                "vllm_present": vllm_slot is not None,
                "hf_context_len": (
                    int(hf_slot.get("context_len", 0)) if hf_slot else None
                ),
                "vllm_context_len": (
                    int(vllm_slot.get("context_len", 0))
                    if vllm_slot
                    else None
                ),
            }
            continue
        hf_attn = hf_slot["attn_taps"]
        vllm_attn = vllm_slot["attn_taps"]
        hf_ctx = hf_slot["ctx_kv"]
        vllm_ctx = vllm_slot["ctx_kv"]
        attn_report: dict[str, Any] = {}
        first_divergence: dict[str, Any] | None = None
        for tap in attn_tap_order:
            cmp = _cmp(hf_attn.get(tap), vllm_attn.get(tap))
            attn_report[tap] = cmp
            if (
                first_divergence is None
                and cmp.get("status") == "ok"
                and cmp.get("cosine", 1.0) < cosine_threshold
            ):
                first_divergence = {
                    "tap": tap,
                    "cosine": cmp.get("cosine"),
                    "l2_diff": cmp.get("l2_diff"),
                    "hf_norm": cmp.get("hf_norm"),
                    "vllm_norm": cmp.get("vllm_norm"),
                }
        ctx_report: dict[str, Any] = {}
        for ck in ctx_key_order:
            ctx_report[ck] = _cmp(hf_ctx.get(ck), vllm_ctx.get(ck))
        per_step_report[str(s_idx)] = {
            "hf_context_len": int(hf_slot.get("context_len", 0)),
            "vllm_context_len": int(vllm_slot.get("context_len", 0)),
            "context_len_matches": (
                int(hf_slot.get("context_len", 0))
                == int(vllm_slot.get("context_len", 0))
            ),
            "per_attn_tap_parity": attn_report,
            "context_kv_parity": ctx_report,
            "first_attn_divergence_tap": first_divergence,
            "hf_source": hf_slot.get("source"),
        }

    # Cross-step drift within each side.  For each tap, compare the
    # step-0 capture to step-N capture on the same side; if the
    # self_attn_out or Q/K/V changes substantially, the context / cache
    # state drifted between iterations.  On HF this should always be
    # "it diverged because context changed" (teacher-forced context
    # grows each step); on vLLM a larger drift than HF's suggests extra
    # cross-iteration state (noise K/V accumulation, stale context).
    def _drift_within_side(by_step: dict[int, dict[str, Any]]) -> dict[str, Any]:
        if not by_step:
            return {"error": "no captures"}
        base_step = min(by_step.keys())
        base_slot = by_step[base_step]
        base_attn = base_slot.get("attn_taps") or {}
        drift: dict[str, Any] = {"base_step": int(base_step), "per_step": {}}
        for s_idx in sorted(by_step.keys()):
            if s_idx == base_step:
                continue
            other_slot = by_step[s_idx]
            other_attn = other_slot.get("attn_taps") or {}
            row: dict[str, Any] = {
                "context_len": int(other_slot.get("context_len", 0))
            }
            for tap in attn_tap_order:
                row[tap] = _cmp(base_attn.get(tap), other_attn.get(tap))
            drift["per_step"][str(s_idx)] = row
        return drift

    hf_drift = _drift_within_side(hf_by_step)
    vllm_drift = _drift_within_side(vllm_by_step)

    # Verdict: for the "i4_attn_out_pre_oproj" tap, compare the HF-to-vLLM
    # cosine at step 0 vs at the latest step.  If cosine degrades, the
    # gap is cross-iteration (Pγ / Pδ).  If flat, the per-iteration
    # audit is not the mechanism.
    latest_step = max(configured_steps) if configured_steps else None
    earliest_step = min(configured_steps) if configured_steps else None
    verdict: dict[str, Any] = {
        "configured_steps": configured_steps,
        "earliest_step": earliest_step,
        "latest_step": latest_step,
    }
    try:
        early = per_step_report.get(str(earliest_step), {})
        late = per_step_report.get(str(latest_step), {})
        early_i4 = (early.get("per_attn_tap_parity") or {}).get(
            "i4_attn_out_pre_oproj", {}
        )
        late_i4 = (late.get("per_attn_tap_parity") or {}).get(
            "i4_attn_out_pre_oproj", {}
        )
        if early_i4.get("status") == "ok" and late_i4.get("status") == "ok":
            c_early = float(early_i4.get("cosine", 1.0))
            c_late = float(late_i4.get("cosine", 1.0))
            verdict["i4_attn_out_cos_early"] = c_early
            verdict["i4_attn_out_cos_late"] = c_late
            verdict["i4_attn_out_cos_delta"] = c_late - c_early
            if c_early - c_late > 0.001:
                verdict["interpretation_hint"] = (
                    "Layer-1 self_attn_out HF-vs-vLLM cosine DROPS from "
                    f"{c_early:.5f} at step {earliest_step} to {c_late:.5f}"
                    f" at step {latest_step}: vLLM's self-attention output"
                    " drifts away from HF as speculative iterations"
                    " accumulate, consistent with Pγ (noise-K/V paged cache"
                    " contamination) or Pδ (context-K/V refresh cadence)."
                )
            elif abs(c_late - c_early) <= 0.001:
                verdict["interpretation_hint"] = (
                    "Layer-1 self_attn_out HF-vs-vLLM cosine is flat across"
                    f" steps ({c_early:.5f} -> {c_late:.5f}): cross-"
                    "iteration draft-side state is NOT the mechanism; the"
                    " gap must come from depth>0 (Pε) or trajectory"
                    " misalignment (Pζ)."
                )
            else:
                verdict["interpretation_hint"] = (
                    "Layer-1 self_attn_out HF-vs-vLLM cosine increases with"
                    " step (unexpected); inspect per_step details."
                )
    except Exception as _ve:
        verdict["verdict_error"] = str(_ve)

    return {
        "cosine_threshold": float(cosine_threshold),
        "configured_steps": configured_steps,
        "bisect_layer_idx": hf_test_l_payload.get(
            "bisect_layer_idx"
        )
        or vllm_capture.get("bisect_layer_idx"),
        "attn_tap_order": attn_tap_order,
        "context_kv_key_order": ctx_key_order,
        "per_step": per_step_report,
        "cross_step_drift_hf": hf_drift,
        "cross_step_drift_vllm": vllm_drift,
        "verdict": verdict,
        "hf_steps_present": sorted(hf_by_step.keys()),
        "vllm_steps_present": sorted(vllm_by_step.keys()),
        "vllm_num_precompute_calls": vllm_capture.get(
            "num_precompute_calls"
        ),
        "vllm_num_self_attn_calls": vllm_capture.get(
            "num_self_attn_calls"
        ),
    }


# ---------------------------------------------------------------------------
# Test O: tree-vs-chain vLLM A/B at the same Test-L taps.
#
# Test N-2 showed that chain-spec vLLM (tree_width=1) operates at ~82% d0
# top-1 while tree-spec vLLM sits at ~41.9% -- the gap is tree-mechanics-
# specific.  Test L already captured tree-spec vLLM's per-step self-attn
# taps at steps (0, 5, 15, 30) for reference against HF.  Test O adds the
# *chain-spec* vLLM per-step taps at the same steps.  Comparing the three
# stacks per step answers:
#
#   - At each step K, how far is tree-vLLM's i4_attn_out_pre_oproj from HF's?
#     How far is chain-vLLM's from HF's?  If chain-vLLM stays near HF while
#     tree-vLLM diverges, the corruption is tree-specific (Pγ/Pδ/Pε).
#   - Which tap is the first to split between tree-vLLM and chain-vLLM at
#     step K?  Walking through i1..i5 (plus k/v/context) narrows the
#     mechanism to a specific sub-layer: a pre-RoPE Q/K/V split points at
#     LayerNorm / noise-K/V state, a post-RoPE split points at positions/
#     rotary, an attn-output split points at context K/V or the attn kernel
#     itself.
#
# Note: trajectories differ between tree and chain, so "step K on tree"
# and "step K on chain" do NOT operate on the same accepted prefix in
# general.  What *is* still comparable at step K is the *shape of the
# internal state* given each side's own trajectory; if tree-vLLM's attn
# output is severely attenuated relative to its own step-0 or relative to
# HF, but chain-vLLM's is not, that is a trajectory-independent indictment
# of the tree-spec code path.
# ---------------------------------------------------------------------------


def build_test_o_tree_vs_chain_layer1_report(
    hf_test_l_payload: dict[str, Any],
    hf_reference_capture: dict[str, Any],
    vllm_tree_capture: dict[str, Any],
    vllm_chain_capture: dict[str, Any],
    cosine_threshold: float = 0.999,
) -> dict[str, Any]:
    """Per-step A/B between tree-spec and chain-spec vLLM Test-L taps.

    Inputs
    ------
    hf_test_l_payload:
        Deserialized ``reference_capture_test_l.json`` (HF steps > 0).
    hf_reference_capture:
        Deserialized ``reference_capture.json`` (HF step-0 legacy capture).
    vllm_tree_capture:
        Contents of ``sample_<i>/vllm_test_l_probe_capture.json`` produced
        by the tree-spec run (includes a synthesized step-0 entry spliced
        from the legacy attn-bisect capture).
    vllm_chain_capture:
        Contents of ``sample_<i>/vllm_test_l_probe_capture_chain.json``
        produced by the chain-spec run (captures step 0 directly via the
        standalone Test-L probe).

    Output
    ------
    Per configured step, three cosine slots (HF-vs-tree, HF-vs-chain,
    tree-vs-chain) for each of the standard Test-L taps, plus a verdict
    highlighting (a) whether chain stays close to HF where tree diverges,
    and (b) which tap first splits tree from chain.
    """
    hf_per_step_raw: dict[str, Any] = hf_test_l_payload.get("per_step") or {}
    ref_attn = hf_reference_capture.get("draft_layer1_attn_bisect_at_d0") or {}
    ref_ctx = hf_reference_capture.get("draft_layer1_context_kv") or {}

    def _unpack_vllm(vllm_capture: dict[str, Any]) -> dict[int, dict[str, Any]]:
        by_step: dict[int, dict[str, Any]] = {}
        for entry in vllm_capture.get("per_step") or []:
            try:
                s_idx = int(entry.get("step"))
            except Exception:
                continue
            by_step[s_idx] = {
                "context_len": int(entry.get("context_len", 0)),
                "attn_taps": entry.get("attn_taps") or {},
                "ctx_kv": entry.get("ctx_kv") or {},
            }
        return by_step

    tree_by_step = _unpack_vllm(vllm_tree_capture)
    chain_by_step = _unpack_vllm(vllm_chain_capture)

    hf_by_step: dict[int, dict[str, Any]] = {}
    hf_step0_attn: dict[str, list[float]] = {
        k: v
        for k, v in ref_attn.items()
        if k not in {"layer_idx", "_stats"} and isinstance(v, list)
    }
    hf_step0_ctx: dict[str, list[float]] = {
        k: v
        for k, v in ref_ctx.items()
        if k
        not in {"layer_idx", "last_context_index", "num_context", "_stats"}
        and isinstance(v, list)
    }
    if hf_step0_attn or hf_step0_ctx:
        hf_by_step[0] = {
            "context_len": int(
                hf_reference_capture.get("num_prompt_tokens") or 0
            ),
            "attn_taps": hf_step0_attn,
            "ctx_kv": hf_step0_ctx,
        }
    for str_step, entry in hf_per_step_raw.items():
        try:
            s_idx = int(entry.get("step", str_step))
        except Exception:
            continue
        hf_by_step[s_idx] = {
            "context_len": int(entry.get("context_len", 0)),
            "attn_taps": entry.get("attn_taps") or {},
            "ctx_kv": entry.get("ctx_kv") or {},
        }

    def _cos(a: Any, b: Any) -> float | None:
        if not isinstance(a, list) or not isinstance(b, list):
            return None
        if not a or not b or len(a) != len(b):
            return None
        return float(_cosine(a, b))

    attn_tap_order = [
        "i1_q_post_qproj",
        "k_noise_post_kproj",
        "v_noise_post_vproj",
        "i2_q_post_qnorm",
        "k_noise_post_knorm",
        "i3_q_post_rope",
        "k_noise_post_rope",
        "i4_attn_out_pre_oproj",
        "i5_attn_out_post_oproj",
    ]
    ctx_key_order = [
        "k_last_context_pre_rope",
        "k_last_context",
        "v_last_context",
    ]

    configured_steps = sorted(
        set(tree_by_step.keys())
        | set(chain_by_step.keys())
        | set(hf_by_step.keys())
    )

    per_step_report: dict[str, Any] = {}
    for s_idx in configured_steps:
        hf_slot = hf_by_step.get(s_idx)
        tree_slot = tree_by_step.get(s_idx)
        chain_slot = chain_by_step.get(s_idx)
        slot_info: dict[str, Any] = {
            "hf_present": hf_slot is not None,
            "tree_present": tree_slot is not None,
            "chain_present": chain_slot is not None,
            "hf_context_len": (
                int(hf_slot.get("context_len", 0)) if hf_slot else None
            ),
            "tree_context_len": (
                int(tree_slot.get("context_len", 0)) if tree_slot else None
            ),
            "chain_context_len": (
                int(chain_slot.get("context_len", 0)) if chain_slot else None
            ),
        }

        attn_report: dict[str, Any] = {}
        first_split_tree_vs_chain: dict[str, Any] | None = None
        for tap in attn_tap_order:
            hf_tap = (hf_slot or {}).get("attn_taps", {}).get(tap)
            tree_tap = (tree_slot or {}).get("attn_taps", {}).get(tap)
            chain_tap = (chain_slot or {}).get("attn_taps", {}).get(tap)
            row = {
                "hf_vs_tree_cosine": _cos(hf_tap, tree_tap),
                "hf_vs_chain_cosine": _cos(hf_tap, chain_tap),
                "tree_vs_chain_cosine": _cos(tree_tap, chain_tap),
            }
            attn_report[tap] = row
            tvc = row["tree_vs_chain_cosine"]
            if (
                first_split_tree_vs_chain is None
                and tvc is not None
                and tvc < cosine_threshold
            ):
                first_split_tree_vs_chain = {"tap": tap, "cosine": tvc}

        ctx_report: dict[str, Any] = {}
        for ck in ctx_key_order:
            hf_ck = (hf_slot or {}).get("ctx_kv", {}).get(ck)
            tree_ck = (tree_slot or {}).get("ctx_kv", {}).get(ck)
            chain_ck = (chain_slot or {}).get("ctx_kv", {}).get(ck)
            ctx_report[ck] = {
                "hf_vs_tree_cosine": _cos(hf_ck, tree_ck),
                "hf_vs_chain_cosine": _cos(hf_ck, chain_ck),
                "tree_vs_chain_cosine": _cos(tree_ck, chain_ck),
            }

        per_step_report[str(s_idx)] = {
            **slot_info,
            "per_attn_tap": attn_report,
            "context_kv": ctx_report,
            "first_tree_vs_chain_split_tap": first_split_tree_vs_chain,
        }

    # Verdict: compare tree and chain to HF at the latest step for the
    # canonical attn-out tap.  If chain stays close to HF while tree
    # diverges, that is prima-facie evidence the tree-spec code path (not
    # the kernel/cache/cross-attn itself) is the corruption source.
    verdict: dict[str, Any] = {
        "configured_steps": configured_steps,
    }
    try:
        latest = max(configured_steps) if configured_steps else None
        late = per_step_report.get(str(latest), {}) if latest is not None else {}
        late_attn = late.get("per_attn_tap") or {}
        late_i4 = late_attn.get("i4_attn_out_pre_oproj") or {}
        c_hf_tree = late_i4.get("hf_vs_tree_cosine")
        c_hf_chain = late_i4.get("hf_vs_chain_cosine")
        c_tree_chain = late_i4.get("tree_vs_chain_cosine")
        verdict["latest_step"] = latest
        verdict["i4_attn_out_hf_vs_tree_cos_latest"] = c_hf_tree
        verdict["i4_attn_out_hf_vs_chain_cos_latest"] = c_hf_chain
        verdict["i4_attn_out_tree_vs_chain_cos_latest"] = c_tree_chain
        if (
            c_hf_tree is not None
            and c_hf_chain is not None
            and c_hf_chain - c_hf_tree > 0.2
        ):
            verdict["interpretation_hint"] = (
                "At step "
                f"{latest} chain-vLLM stays close to HF "
                f"(cos={c_hf_chain:.4f}) while tree-vLLM diverges "
                f"(cos={c_hf_tree:.4f}) on i4_attn_out_pre_oproj: the "
                "tree-spec code path specifically corrupts layer-1 self-"
                "attention output.  Walk the earlier taps in per_step "
                "to find where tree and chain first split (see "
                "``first_tree_vs_chain_split_tap``)."
            )
        elif (
            c_hf_tree is not None
            and c_hf_chain is not None
            and abs(c_hf_chain - c_hf_tree) <= 0.05
        ):
            verdict["interpretation_hint"] = (
                f"At step {latest} tree-vLLM and chain-vLLM are equally "
                f"close (or equally far) from HF on i4_attn_out_pre_oproj "
                f"(cos_tree={c_hf_tree}, cos_chain={c_hf_chain}); the "
                "issue is NOT tree-specific at this tap -- inspect "
                "depth>0 branches or trajectory-divergence effects."
            )
        else:
            verdict["interpretation_hint"] = (
                "Mixed / partial evidence; examine per_step details."
            )
    except Exception as _ve:
        verdict["verdict_error"] = str(_ve)

    return {
        "cosine_threshold": float(cosine_threshold),
        "configured_steps": configured_steps,
        "attn_tap_order": attn_tap_order,
        "context_kv_key_order": ctx_key_order,
        "per_step": per_step_report,
        "verdict": verdict,
        "hf_steps_present": sorted(hf_by_step.keys()),
        "tree_steps_present": sorted(tree_by_step.keys()),
        "chain_steps_present": sorted(chain_by_step.keys()),
    }


# ---------------------------------------------------------------------------
# Test P: position-matched tree-vs-chain internal-state A/B.
#
# Test O compared tree-spec and chain-spec Test-L taps at identical
# speculative *iteration indices*, but tree-spec advances multiple accepted
# tokens per iter while chain-spec advances ~1, so the two were scored on
# completely different decoded positions.  Test N made it clear the aggregate
# gap is tree-specific (chain-spec ~82% top1 vs tree-spec ~44%), so we now
# need a like-for-like internal-state A/B at *matched decoded positions*.
#
# Both runs follow the same target-greedy accepted trajectory, so for each
# tree iter with position P there (almost always) exists a chain iter with
# the same position.  We reconstruct per-iter starting position from
# topk_log's ``accepted_len`` column (per-iter advance = accepted_len + 1
# under standard DFlash rejection semantics) and pair (tree_iter, chain_iter)
# whenever positions AND root tokens agree.  At each matched pair we compute
# tree-vs-chain cosines on every Test-L self_attn tap and ctx K/V key.  A
# low cosine at a matched pair is direct evidence of tree-specific state
# corruption at that layer/tap at that decoded position; flat-high cosines
# across all matched positions instead localize the gap to higher-depth
# behavior (which Test-L's depth-0 probe does not observe).
# ---------------------------------------------------------------------------


def build_test_p_position_matched_report(
    vllm_tree_capture: dict[str, Any],
    vllm_chain_capture: dict[str, Any],
    vllm_tree_summary: dict[str, Any],
    vllm_chain_summary: dict[str, Any],
    hf_test_l_payload: dict[str, Any] | None = None,
    hf_reference_capture: dict[str, Any] | None = None,
    target_forced_summary: dict[str, Any] | None = None,
    sample_index: int = 0,
    cosine_threshold: float = 0.999,
) -> dict[str, Any]:
    """Test P: tree-vs-chain internal-state A/B at matched decoded positions.

    Inputs
    ------
    vllm_tree_capture, vllm_chain_capture:
        Per-step Test-L probe captures (from retrieve_vllm_test_l_probe) for
        the tree-spec and chain-spec vLLM runs respectively.  Each must have
        ``per_step: list[{"step": int, "context_len": int, "attn_taps": {},
        "ctx_kv": {}}]``.
    vllm_tree_summary, vllm_chain_summary:
        Aggregate draft-quality summaries containing per-sample ``steps``
        lists with ``accepted_len`` / ``root_token`` / ``draft_top1_token``
        / ``target_next_token`` fields.  Used to reconstruct per-iter
        decoded position.
    hf_test_l_payload, hf_reference_capture, target_forced_summary:
        Optional HF anchors.  When present, the report also records
        HF-vs-tree and HF-vs-chain cosines at matched positions (HF's iter
        index == decoded-position index for chain-greedy runs, so HF lines
        up with chain one-to-one).
    sample_index:
        Which sample's topk_log to use for position reconstruction.
    cosine_threshold:
        Report the first matched pair at which any tap's tree-vs-chain
        cosine drops below this in the verdict.

    Output (JSON-serializable dict)
    -------------------------------
    ``per_matched_pair``:
        List of matched (tree_iter, chain_iter) pairs at the same decoded
        position, with per-tap tree-vs-chain cosines (and HF-vs-tree,
        HF-vs-chain when HF data is available) plus draft-top1 / target-next
        agreement booleans.
    ``summary``:
        Aggregate per-tap mean/min cosine across matched pairs and
        top-1/target agreement rates.
    ``verdict``:
        Interpretation hint localizing the first position / tap at which
        tree-vs-chain diverges.
    """

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> tuple[int, list[dict[str, Any]]]:
        """Return (num_prompt_tokens, list of per-iter dicts augmented with
        ``__position`` = decoded-position-since-end-of-prompt at iter start).

        We intentionally use the *relative* position (``cumulative``) rather
        than ``num_prompt + cumulative`` so that all three sides (tree,
        chain, HF) share a common origin even when vLLM's summary does not
        carry ``num_prompt_tokens`` (HF's does).
        """
        for s in summary.get("samples") or []:
            if int(s.get("sample_index", -1)) != int(sample_idx):
                continue
            try:
                num_prompt = int(s.get("num_prompt_tokens") or 0)
            except (TypeError, ValueError):
                num_prompt = 0
            steps_sorted = sorted(
                s.get("steps") or [], key=lambda e: int(e.get("step", 0))
            )
            cumulative = 0
            out: list[dict[str, Any]] = []
            for e in steps_sorted:
                enriched = dict(e)
                enriched["__position"] = cumulative
                out.append(enriched)
                try:
                    cumulative += int(e.get("accepted_len", 0)) + 1
                except (TypeError, ValueError):
                    cumulative += 1
            return num_prompt, out
        return 0, []

    num_prompt_tree, tree_iters = _per_iter_positions(
        vllm_tree_summary, sample_index
    )
    num_prompt_chain, chain_iters = _per_iter_positions(
        vllm_chain_summary, sample_index
    )

    def _capture_by_step(cap: dict[str, Any]) -> dict[int, dict[str, Any]]:
        by_step: dict[int, dict[str, Any]] = {}
        for entry in cap.get("per_step") or []:
            try:
                s_idx = int(entry.get("step"))
            except Exception:
                continue
            by_step[s_idx] = {
                "context_len": int(entry.get("context_len", 0)),
                "attn_taps": entry.get("attn_taps") or {},
                "ctx_kv": entry.get("ctx_kv") or {},
            }
        return by_step

    tree_caps = _capture_by_step(vllm_tree_capture)
    chain_caps = _capture_by_step(vllm_chain_capture)

    # Build position -> iter lookups restricted to ITERS THAT WERE CAPTURED.
    tree_captured_by_position: dict[int, dict[str, Any]] = {}
    for it in tree_iters:
        si = int(it.get("step", -1))
        if si in tree_caps:
            tree_captured_by_position[int(it["__position"])] = {
                "iter": si,
                "iter_entry": it,
                "capture": tree_caps[si],
            }
    chain_captured_by_position: dict[int, dict[str, Any]] = {}
    for it in chain_iters:
        si = int(it.get("step", -1))
        if si in chain_caps:
            chain_captured_by_position[int(it["__position"])] = {
                "iter": si,
                "iter_entry": it,
                "capture": chain_caps[si],
            }

    # Optional HF anchors: HF position at iter i == num_prompt_tokens + i
    # (chain-greedy target-forced run advances exactly 1 token per iter).
    hf_captured_by_position: dict[int, dict[str, Any]] = {}
    # HF-side positions use the same "decoded-position-since-end-of-prompt"
    # convention as vLLM's _per_iter_positions above.  For HF's chain-greedy
    # target-forced run, each outer-loop iter accepts exactly 1 token, so
    # HF's normalized position at iter ``i`` is simply ``i``.
    if (
        hf_test_l_payload is not None
        and target_forced_summary is not None
    ):
        hf_per_step = hf_test_l_payload.get("per_step") or {}
        for str_step, entry in hf_per_step.items():
            try:
                s_idx = int(entry.get("step", str_step))
            except Exception:
                continue
            hf_captured_by_position[s_idx] = {
                "iter": s_idx,
                "capture": {
                    "context_len": int(entry.get("context_len", 0)),
                    "attn_taps": entry.get("attn_taps") or {},
                    "ctx_kv": entry.get("ctx_kv") or {},
                },
            }
        # Optional step-0 anchor from reference_capture.json (legacy
        # bisect probe, not the Test-L probe).
        if hf_reference_capture is not None:
            ref_attn = (
                hf_reference_capture.get("draft_layer1_attn_bisect_at_d0")
                or {}
            )
            ref_ctx = (
                hf_reference_capture.get("draft_layer1_context_kv") or {}
            )
            step0_attn = {
                k: v
                for k, v in ref_attn.items()
                if k not in {"layer_idx", "_stats"} and isinstance(v, list)
            }
            step0_ctx = {
                k: v
                for k, v in ref_ctx.items()
                if k
                not in {
                    "layer_idx",
                    "last_context_index",
                    "num_context",
                    "_stats",
                }
                and isinstance(v, list)
            }
            if (step0_attn or step0_ctx) and 0 not in hf_captured_by_position:
                hf_captured_by_position[0] = {
                    "iter": 0,
                    "capture": {
                        "context_len": int(
                            hf_reference_capture.get("num_prompt_tokens") or 0
                        ),
                        "attn_taps": step0_attn,
                        "ctx_kv": step0_ctx,
                    },
                }

    attn_tap_order = [
        "i1_q_post_qproj",
        "k_noise_post_kproj",
        "v_noise_post_vproj",
        "i2_q_post_qnorm",
        "k_noise_post_knorm",
        "i3_q_post_rope",
        "k_noise_post_rope",
        "i4_attn_out_pre_oproj",
        "i5_attn_out_post_oproj",
    ]
    ctx_key_order = [
        "k_last_context_pre_rope",
        "k_last_context",
        "v_last_context",
    ]

    def _cos(a: Any, b: Any) -> float | None:
        if not isinstance(a, list) or not isinstance(b, list):
            return None
        if not a or not b or len(a) != len(b):
            return None
        return float(_cosine(a, b))

    matched_positions = sorted(
        set(tree_captured_by_position) & set(chain_captured_by_position)
    )

    per_pair: list[dict[str, Any]] = []
    first_low_cos_pair: dict[str, Any] | None = None
    for pos in matched_positions:
        t = tree_captured_by_position[pos]
        c = chain_captured_by_position[pos]
        t_entry = t["iter_entry"]
        c_entry = c["iter_entry"]
        t_cap = t["capture"]
        c_cap = c["capture"]

        root_tree = t_entry.get("root_token")
        root_chain = c_entry.get("root_token")
        tgt_tree = t_entry.get("target_next_token")
        tgt_chain = c_entry.get("target_next_token")
        top1_tree = t_entry.get("draft_top1_token")
        top1_chain = c_entry.get("draft_top1_token")

        hf_slot = hf_captured_by_position.get(pos)

        attn_report: dict[str, Any] = {}
        pair_min_tvc: float | None = None
        pair_first_low_tap: dict[str, Any] | None = None
        for tap in attn_tap_order:
            t_tap = (t_cap.get("attn_taps") or {}).get(tap)
            c_tap = (c_cap.get("attn_taps") or {}).get(tap)
            row = {"tree_vs_chain_cosine": _cos(t_tap, c_tap)}
            if hf_slot is not None:
                h_tap = (hf_slot["capture"].get("attn_taps") or {}).get(tap)
                row["hf_vs_tree_cosine"] = _cos(h_tap, t_tap)
                row["hf_vs_chain_cosine"] = _cos(h_tap, c_tap)
            attn_report[tap] = row
            tvc = row["tree_vs_chain_cosine"]
            if tvc is not None:
                pair_min_tvc = (
                    tvc if pair_min_tvc is None else min(pair_min_tvc, tvc)
                )
                if pair_first_low_tap is None and tvc < cosine_threshold:
                    pair_first_low_tap = {"tap": tap, "cosine": tvc}

        ctx_report: dict[str, Any] = {}
        for ck in ctx_key_order:
            t_ck = (t_cap.get("ctx_kv") or {}).get(ck)
            c_ck = (c_cap.get("ctx_kv") or {}).get(ck)
            row = {"tree_vs_chain_cosine": _cos(t_ck, c_ck)}
            if hf_slot is not None:
                h_ck = (hf_slot["capture"].get("ctx_kv") or {}).get(ck)
                row["hf_vs_tree_cosine"] = _cos(h_ck, t_ck)
                row["hf_vs_chain_cosine"] = _cos(h_ck, c_ck)
            ctx_report[ck] = row

        pair_entry: dict[str, Any] = {
            "position": int(pos),
            "tree_iter": int(t["iter"]),
            "chain_iter": int(c["iter"]),
            "tree_context_len": int(t_cap.get("context_len", 0)),
            "chain_context_len": int(c_cap.get("context_len", 0)),
            "hf_iter": int(hf_slot["iter"]) if hf_slot else None,
            "hf_context_len": (
                int(hf_slot["capture"].get("context_len", 0))
                if hf_slot
                else None
            ),
            "root_token_tree": root_tree,
            "root_token_chain": root_chain,
            "root_tokens_match": (
                root_tree == root_chain
                if root_tree is not None and root_chain is not None
                else None
            ),
            "target_next_token_tree": tgt_tree,
            "target_next_token_chain": tgt_chain,
            "target_next_tokens_match": (
                tgt_tree == tgt_chain
                if tgt_tree is not None and tgt_chain is not None
                else None
            ),
            "draft_top1_tree": top1_tree,
            "draft_top1_chain": top1_chain,
            "draft_top1_agree": (
                top1_tree == top1_chain
                if top1_tree is not None and top1_chain is not None
                else None
            ),
            "tree_top1_matches_target": bool(
                t_entry.get("draft_top1_match")
            ),
            "chain_top1_matches_target": bool(
                c_entry.get("draft_top1_match")
            ),
            "per_attn_tap": attn_report,
            "context_kv": ctx_report,
            "pair_min_tree_vs_chain_cosine": pair_min_tvc,
            "pair_first_low_tap": pair_first_low_tap,
        }
        per_pair.append(pair_entry)
        if (
            first_low_cos_pair is None
            and pair_first_low_tap is not None
        ):
            first_low_cos_pair = {
                "position": int(pos),
                "tap": pair_first_low_tap["tap"],
                "cosine": pair_first_low_tap["cosine"],
            }

    # ------------------------------------------------------------------
    # Aggregate summary across matched pairs.
    # ------------------------------------------------------------------
    def _mean(xs: list[float]) -> float | None:
        return (sum(xs) / len(xs)) if xs else None

    def _min(xs: list[float]) -> float | None:
        return min(xs) if xs else None

    per_tap_stats: dict[str, Any] = {}
    for tap in attn_tap_order:
        tvcs = [
            p["per_attn_tap"][tap]["tree_vs_chain_cosine"]
            for p in per_pair
            if p["per_attn_tap"][tap]["tree_vs_chain_cosine"] is not None
        ]
        per_tap_stats[tap] = {
            "num_pairs": len(tvcs),
            "mean_tree_vs_chain_cosine": _mean(tvcs),
            "min_tree_vs_chain_cosine": _min(tvcs),
        }
    per_ctx_stats: dict[str, Any] = {}
    for ck in ctx_key_order:
        tvcs = [
            p["context_kv"][ck]["tree_vs_chain_cosine"]
            for p in per_pair
            if p["context_kv"][ck]["tree_vs_chain_cosine"] is not None
        ]
        per_ctx_stats[ck] = {
            "num_pairs": len(tvcs),
            "mean_tree_vs_chain_cosine": _mean(tvcs),
            "min_tree_vs_chain_cosine": _min(tvcs),
        }

    num_pairs = len(per_pair)
    num_root_match = sum(
        1 for p in per_pair if p["root_tokens_match"] is True
    )
    num_tgt_match = sum(
        1 for p in per_pair if p["target_next_tokens_match"] is True
    )
    num_top1_agree = sum(
        1 for p in per_pair if p["draft_top1_agree"] is True
    )
    num_tree_hits = sum(
        1 for p in per_pair if p["tree_top1_matches_target"]
    )
    num_chain_hits = sum(
        1 for p in per_pair if p["chain_top1_matches_target"]
    )

    summary = {
        "num_matched_pairs": num_pairs,
        "num_captured_tree_positions": len(tree_captured_by_position),
        "num_captured_chain_positions": len(chain_captured_by_position),
        "num_captured_hf_positions": len(hf_captured_by_position),
        "num_pairs_with_root_match": num_root_match,
        "num_pairs_with_target_next_match": num_tgt_match,
        "num_pairs_with_draft_top1_agree": num_top1_agree,
        "tree_top1_hit_rate_on_matched": (
            num_tree_hits / num_pairs if num_pairs else None
        ),
        "chain_top1_hit_rate_on_matched": (
            num_chain_hits / num_pairs if num_pairs else None
        ),
        "draft_top1_agreement_rate_on_matched": (
            num_top1_agree / num_pairs if num_pairs else None
        ),
        "per_attn_tap": per_tap_stats,
        "context_kv": per_ctx_stats,
    }

    # ------------------------------------------------------------------
    # Verdict.
    # ------------------------------------------------------------------
    verdict: dict[str, Any] = {
        "sample_index": int(sample_index),
        "cosine_threshold": float(cosine_threshold),
        "num_matched_pairs": num_pairs,
        "first_low_cos_pair": first_low_cos_pair,
    }
    try:
        i4_stats = per_tap_stats.get("i4_attn_out_pre_oproj") or {}
        i5_stats = per_tap_stats.get("i5_attn_out_post_oproj") or {}
        verdict["mean_i4_attn_out_pre_oproj_tree_vs_chain_cosine"] = (
            i4_stats.get("mean_tree_vs_chain_cosine")
        )
        verdict["min_i4_attn_out_pre_oproj_tree_vs_chain_cosine"] = (
            i4_stats.get("min_tree_vs_chain_cosine")
        )
        verdict["mean_i5_attn_out_post_oproj_tree_vs_chain_cosine"] = (
            i5_stats.get("mean_tree_vs_chain_cosine")
        )
        mean_i5 = i5_stats.get("mean_tree_vs_chain_cosine")
        mean_i4 = i4_stats.get("mean_tree_vs_chain_cosine")
        if num_pairs == 0:
            verdict["interpretation_hint"] = (
                "No matched (same-position, same-root) pairs were produced; "
                "expand TEST_P_CAPTURE_STEPS on one or both sides so at "
                "least one tree iter's starting position coincides with a "
                "chain iter's starting position."
            )
        elif (
            mean_i4 is not None
            and mean_i5 is not None
            and mean_i4 > 0.995
            and mean_i5 > 0.995
        ):
            verdict["interpretation_hint"] = (
                "At matched decoded positions, tree and chain self_attn "
                "output cosines are ~1.0 across all pairs; the tree-vs-"
                "chain gap does NOT come from kernel/state divergence at "
                "layer-1 depth-0.  The corruption is either at later "
                "layers or at depth>0 branches -- add depth>0 taps."
            )
        elif (
            mean_i4 is not None
            and mean_i4 < 0.9
        ):
            verdict["interpretation_hint"] = (
                "At matched decoded positions, mean tree-vs-chain cosine "
                f"on i4_attn_out_pre_oproj is {mean_i4:.4f}; tree-spec "
                "layer-1 self_attn state is materially corrupted relative "
                "to chain-spec at the SAME position.  See "
                "``first_low_cos_pair`` for the earliest matched position "
                "at which this shows up, and walk earlier taps (Q/K "
                "projections, Q/K norm, RoPE) to localize the split."
            )
        else:
            verdict["interpretation_hint"] = (
                "Moderate tree-vs-chain divergence at matched positions; "
                "inspect ``per_matched_pair`` in detail."
            )
    except Exception as _ve:
        verdict["verdict_error"] = str(_ve)

    return {
        "cosine_threshold": float(cosine_threshold),
        "sample_index": int(sample_index),
        "attn_tap_order": attn_tap_order,
        "context_kv_key_order": ctx_key_order,
        "num_prompt_tokens_tree": int(num_prompt_tree),
        "num_prompt_tokens_chain": int(num_prompt_chain),
        "tree_captured_positions": sorted(tree_captured_by_position.keys()),
        "chain_captured_positions": sorted(chain_captured_by_position.keys()),
        "hf_captured_positions": sorted(hf_captured_by_position.keys()),
        "matched_positions": matched_positions,
        "per_matched_pair": per_pair,
        "summary": summary,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Test Q: layer-0 hidden-state A/B at matched decoded positions.
#
# Follow-up to Test P.  Test P showed that at matched decoded positions the
# tree-spec drafter's layer-1 ``i1_q_post_qproj`` diverges from chain
# (cos ~0.65) -- i.e. the residual stream entering layer 1 (= layer-0
# output) already differs.  Test Q taps LAYER 0 directly to identify which
# of {input plumbing, self_attn call, MLP} inside layer 0 is the first to
# split, and to cross-check the paged-KV slot-mapping (H5-slot) and
# accepted-prefix K/V contents (H1 / H2 / H3).
#
# Scoring the live hypotheses:
#   * layer0_input_hidden cos ~1.0 across pairs => input plumbing sound;
#     bug is inside layer 0.
#   * layer0_self_attn_out cos < 1 while input cos ~1 => the split is born
#     in layer-0 self_attn itself (K/V contents read by the kernel OR the
#     attention mask OR slot indices).
#   * layer0_output_hidden cos < 1 while self_attn_out cos ~1 => bug is in
#     MLP (unlikely -- MLP is a deterministic function of its input).
#   * context_positions[0..overlap] / context_slot_mapping[0..overlap]
#     disagree across tree/chain => H5-slot (off-by-depth or shifted KV
#     indices) confirmed.
#   * context_positions / context_slot_mapping agree, self_attn_out splits
#     => narrows to H1 (stale slots leaking via mask) or H2/H3 (wrong K/V
#     content at accepted-prefix slots), to be disambiguated by inspecting
#     accepted-prefix ctx_per_position in Test L captures.
# ---------------------------------------------------------------------------


def build_test_q_layer0_report(
    vllm_tree_q_capture: dict[str, Any],
    vllm_chain_q_capture: dict[str, Any],
    vllm_tree_summary: dict[str, Any],
    vllm_chain_summary: dict[str, Any],
    vllm_tree_l_capture: dict[str, Any] | None = None,
    vllm_chain_l_capture: dict[str, Any] | None = None,
    sample_index: int = 0,
    cosine_threshold: float = 0.999,
    capture_row: int | None = None,
    row_label: str = "depth-0 row",
) -> dict[str, Any]:
    """Test Q report -- position-matched tree-vs-chain layer-0 A/B.

    Parameters
    ----------
    vllm_tree_q_capture, vllm_chain_q_capture : output of
        ``retrieve_vllm_test_q_probe`` for each run.  Each ``per_step``
        slot carries ``attn_taps`` with
        {``layer0_input_hidden``, ``layer0_input_residual``,
         ``layer0_self_attn_out``,
         ``layer0_output_hidden``, ``layer0_output_residual``}.
    vllm_tree_summary, vllm_chain_summary : the per-sample summaries used to
        reconstruct per-iteration decoded positions (same convention as
        Test P: decoded-position-since-end-of-prompt).
    vllm_tree_l_capture, vllm_chain_l_capture : optional Test-L captures,
        from which we additionally compare ``context_positions`` and
        ``context_slot_mapping`` at matched iters (H5-slot direct check).
    sample_index : sample id to use for position reconstruction.
    cosine_threshold : report the first tap/position under this as the
        "first divergence" signal for the verdict.
    """

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> list[dict[str, Any]]:
        """Return the per-iter entries augmented with ``__position``.

        Same relative-position convention as Test P: cumulative decoded
        count since end of prompt.
        """
        for s in summary.get("samples") or []:
            if int(s.get("sample_index", -1)) != int(sample_idx):
                continue
            steps_sorted = sorted(
                s.get("steps") or [], key=lambda e: int(e.get("step", 0))
            )
            cumulative = 0
            out: list[dict[str, Any]] = []
            for e in steps_sorted:
                enriched = dict(e)
                enriched["__position"] = cumulative
                out.append(enriched)
                try:
                    cumulative += int(e.get("accepted_len", 0)) + 1
                except (TypeError, ValueError):
                    cumulative += 1
            return out
        return []

    def _q_by_step(cap: dict[str, Any]) -> dict[int, dict[str, Any]]:
        by_step: dict[int, dict[str, Any]] = {}
        for entry in cap.get("per_step") or []:
            try:
                s_idx = int(entry.get("step"))
            except Exception:
                continue
            selected_taps = entry.get("attn_taps") or {}
            selected_stats = entry.get("attn_stats") or {}
            if capture_row is not None:
                row_key = str(int(capture_row))
                per_row_taps = entry.get("per_row_attn_taps") or {}
                per_row_stats = entry.get("per_row_attn_stats") or {}
                if row_key in per_row_taps:
                    selected_taps = per_row_taps.get(row_key) or {}
                    selected_stats = per_row_stats.get(row_key) or {}
            by_step[s_idx] = {
                "attn_taps": selected_taps,
                "attn_stats": selected_stats,
            }
        return by_step

    def _l_by_step(cap: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
        if cap is None:
            return {}
        by_step: dict[int, dict[str, Any]] = {}
        for entry in cap.get("per_step") or []:
            try:
                s_idx = int(entry.get("step"))
            except Exception:
                continue
            by_step[s_idx] = {
                "context_len": int(entry.get("context_len", 0)),
                "context_positions": entry.get("context_positions"),
                "context_slot_mapping": entry.get("context_slot_mapping"),
            }
        return by_step

    tree_iters = _per_iter_positions(vllm_tree_summary, sample_index)
    chain_iters = _per_iter_positions(vllm_chain_summary, sample_index)
    tree_q = _q_by_step(vllm_tree_q_capture)
    chain_q = _q_by_step(vllm_chain_q_capture)
    tree_l = _l_by_step(vllm_tree_l_capture)
    chain_l = _l_by_step(vllm_chain_l_capture)

    tree_captured_by_position: dict[int, dict[str, Any]] = {}
    for it in tree_iters:
        si = int(it.get("step", -1))
        if si in tree_q:
            tree_captured_by_position[int(it["__position"])] = {
                "iter": si,
                "iter_entry": it,
                "q_capture": tree_q[si],
                "l_capture": tree_l.get(si),
            }
    chain_captured_by_position: dict[int, dict[str, Any]] = {}
    for it in chain_iters:
        si = int(it.get("step", -1))
        if si in chain_q:
            chain_captured_by_position[int(it["__position"])] = {
                "iter": si,
                "iter_entry": it,
                "q_capture": chain_q[si],
                "l_capture": chain_l.get(si),
            }

    matched_positions = sorted(
        set(tree_captured_by_position) & set(chain_captured_by_position)
    )

    def _cos(a: Any, b: Any) -> float | None:
        if not isinstance(a, list) or not isinstance(b, list):
            return None
        if not a or not b or len(a) != len(b):
            return None
        return float(_cosine(a, b))

    tap_order = [
        "layer0_input_hidden",
        "layer0_input_residual",
        "layer0_self_attn_out",
        "layer0_output_hidden",
        "layer0_output_residual",
    ]

    per_pair: list[dict[str, Any]] = []
    first_low_tap_entry: dict[str, Any] | None = None
    for pos in matched_positions:
        t = tree_captured_by_position[pos]
        c = chain_captured_by_position[pos]
        t_taps = t["q_capture"].get("attn_taps") or {}
        c_taps = c["q_capture"].get("attn_taps") or {}
        t_stats = t["q_capture"].get("attn_stats") or {}
        c_stats = c["q_capture"].get("attn_stats") or {}

        per_tap_report: dict[str, Any] = {}
        pair_min_cos: float | None = None
        pair_first_low_tap: dict[str, Any] | None = None
        for tap in tap_order:
            tc = _cos(t_taps.get(tap), c_taps.get(tap))
            row: dict[str, Any] = {
                "tree_vs_chain_cosine": tc,
                "tree_stats": t_stats.get(tap),
                "chain_stats": c_stats.get(tap),
            }
            per_tap_report[tap] = row
            if tc is not None:
                pair_min_cos = (
                    tc if pair_min_cos is None else min(pair_min_cos, tc)
                )
                if pair_first_low_tap is None and tc < cosine_threshold:
                    pair_first_low_tap = {"tap": tap, "cosine": tc}

        # Slot-mapping / context-positions parity (H5-slot direct probe).
        slot_report: dict[str, Any] = {}
        t_l = t.get("l_capture")
        c_l = c.get("l_capture")
        if t_l is not None and c_l is not None:
            t_pos = t_l.get("context_positions")
            c_pos = c_l.get("context_positions")
            t_slot = t_l.get("context_slot_mapping")
            c_slot = c_l.get("context_slot_mapping")
            t_ctx_len = int(t_l.get("context_len", 0))
            c_ctx_len = int(c_l.get("context_len", 0))
            overlap = min(
                len(t_pos) if isinstance(t_pos, list) else 0,
                len(c_pos) if isinstance(c_pos, list) else 0,
            )

            def _first_diff(a: Any, b: Any, up_to: int) -> int | None:
                if not isinstance(a, list) or not isinstance(b, list):
                    return None
                for i in range(up_to):
                    if i >= len(a) or i >= len(b):
                        return i
                    if a[i] != b[i]:
                        return i
                return None

            def _count_mismatch(a: Any, b: Any, up_to: int) -> int | None:
                if not isinstance(a, list) or not isinstance(b, list):
                    return None
                c_ = 0
                for i in range(up_to):
                    if i >= len(a) or i >= len(b):
                        break
                    if a[i] != b[i]:
                        c_ += 1
                return c_

            slot_report = {
                "tree_context_len": t_ctx_len,
                "chain_context_len": c_ctx_len,
                "tree_context_positions_len": (
                    len(t_pos) if isinstance(t_pos, list) else None
                ),
                "chain_context_positions_len": (
                    len(c_pos) if isinstance(c_pos, list) else None
                ),
                "tree_context_slot_mapping_len": (
                    len(t_slot) if isinstance(t_slot, list) else None
                ),
                "chain_context_slot_mapping_len": (
                    len(c_slot) if isinstance(c_slot, list) else None
                ),
                "overlap_len": overlap,
                "context_positions_first_mismatch_idx": _first_diff(
                    t_pos, c_pos, overlap
                ),
                "context_positions_num_mismatch_in_overlap": _count_mismatch(
                    t_pos, c_pos, overlap
                ),
                "context_slot_mapping_first_mismatch_idx": _first_diff(
                    t_slot, c_slot, overlap
                ),
                "context_slot_mapping_num_mismatch_in_overlap": (
                    _count_mismatch(t_slot, c_slot, overlap)
                ),
                # Brief samples of each side for eyeballing (first 8).
                "tree_context_positions_head": (
                    t_pos[:8] if isinstance(t_pos, list) else None
                ),
                "chain_context_positions_head": (
                    c_pos[:8] if isinstance(c_pos, list) else None
                ),
                "tree_context_slot_mapping_head": (
                    t_slot[:8] if isinstance(t_slot, list) else None
                ),
                "chain_context_slot_mapping_head": (
                    c_slot[:8] if isinstance(c_slot, list) else None
                ),
            }

        t_entry = t["iter_entry"]
        c_entry = c["iter_entry"]
        pair_entry: dict[str, Any] = {
            "position": int(pos),
            "tree_iter": int(t["iter"]),
            "chain_iter": int(c["iter"]),
            "root_token_tree": t_entry.get("root_token"),
            "root_token_chain": c_entry.get("root_token"),
            "root_tokens_match": (
                t_entry.get("root_token") == c_entry.get("root_token")
                if t_entry.get("root_token") is not None
                and c_entry.get("root_token") is not None
                else None
            ),
            "per_tap": per_tap_report,
            "slot_mapping": slot_report,
            "pair_min_tree_vs_chain_cosine": pair_min_cos,
            "pair_first_low_tap": pair_first_low_tap,
        }
        per_pair.append(pair_entry)
        if first_low_tap_entry is None and pair_first_low_tap is not None:
            first_low_tap_entry = {
                "position": int(pos),
                "tap": pair_first_low_tap["tap"],
                "cosine": pair_first_low_tap["cosine"],
            }

    def _mean(xs: list[float]) -> float | None:
        return (sum(xs) / len(xs)) if xs else None

    def _min(xs: list[float]) -> float | None:
        return min(xs) if xs else None

    per_tap_stats: dict[str, Any] = {}
    for tap in tap_order:
        vals = [
            p["per_tap"][tap]["tree_vs_chain_cosine"]
            for p in per_pair
            if p["per_tap"][tap]["tree_vs_chain_cosine"] is not None
        ]
        per_tap_stats[tap] = {
            "num_pairs": len(vals),
            "mean_tree_vs_chain_cosine": _mean(vals),
            "min_tree_vs_chain_cosine": _min(vals),
        }

    num_pairs = len(per_pair)
    num_pos_mismatch = sum(
        1
        for p in per_pair
        if (
            (p["slot_mapping"] or {}).get(
                "context_positions_num_mismatch_in_overlap"
            )
            or 0
        )
        > 0
    )
    num_slot_mismatch = sum(
        1
        for p in per_pair
        if (
            (p["slot_mapping"] or {}).get(
                "context_slot_mapping_num_mismatch_in_overlap"
            )
            or 0
        )
        > 0
    )

    summary_out: dict[str, Any] = {
        "capture_row": capture_row,
        "row_label": row_label,
        "num_matched_pairs": num_pairs,
        "num_captured_tree_positions": len(tree_captured_by_position),
        "num_captured_chain_positions": len(chain_captured_by_position),
        "per_tap": per_tap_stats,
        "num_pairs_with_context_positions_mismatch_in_overlap": (
            num_pos_mismatch
        ),
        "num_pairs_with_context_slot_mapping_mismatch_in_overlap": (
            num_slot_mismatch
        ),
    }

    verdict: dict[str, Any] = {
        "sample_index": int(sample_index),
        "cosine_threshold": float(cosine_threshold),
        "capture_row": capture_row,
        "row_label": row_label,
        "num_matched_pairs": num_pairs,
        "first_low_cos": first_low_tap_entry,
    }
    try:
        inp_stats = per_tap_stats.get("layer0_input_hidden") or {}
        sa_stats = per_tap_stats.get("layer0_self_attn_out") or {}
        out_stats = per_tap_stats.get("layer0_output_hidden") or {}
        verdict["mean_layer0_input_hidden_cosine"] = inp_stats.get(
            "mean_tree_vs_chain_cosine"
        )
        verdict["mean_layer0_self_attn_out_cosine"] = sa_stats.get(
            "mean_tree_vs_chain_cosine"
        )
        verdict["mean_layer0_output_hidden_cosine"] = out_stats.get(
            "mean_tree_vs_chain_cosine"
        )
        mean_inp = inp_stats.get("mean_tree_vs_chain_cosine")
        mean_sa = sa_stats.get("mean_tree_vs_chain_cosine")
        mean_out = out_stats.get("mean_tree_vs_chain_cosine")
        if num_pairs == 0:
            verdict["interpretation_hint"] = (
                "No matched (same-position) pairs were produced; expand "
                "capture steps or inspect Test P first to ensure "
                "positions overlap between tree and chain."
            )
        elif mean_inp is not None and mean_inp < 0.995:
            verdict["interpretation_hint"] = (
                f"For the {row_label}, layer-0 INPUT hidden cos is {mean_inp:.4f} (<0.995) at "
                "matched positions.  Something upstream of layer 0 already "
                "differs in tree-spec -- check embedding, positions tensor, "
                "or the incoming residual carry.  (This would REFUTE the "
                "current localization to layer 0's self_attn.)"
            )
        elif mean_sa is not None and mean_sa < 0.995:
            verdict["interpretation_hint"] = (
                f"For the {row_label}, layer-0 INPUT cos is ~{mean_inp:.4f} but "
                f"self_attn_out cos is {mean_sa:.4f}.  Divergence is born "
                "inside layer-0 self_attn -- i.e. the paged-attention "
                "kernel reads different K/V content, different attention "
                "mask, or different slot indices.  Cross-reference slot_"
                "mapping fields in ``per_matched_pair`` to pin H5-slot; "
                "if slot-mapping matches, inspect accepted-prefix "
                "ctx_per_position in the Test-L captures to "
                "disambiguate H1 vs H2/H3."
            )
        elif mean_out is not None and mean_out < 0.995:
            verdict["interpretation_hint"] = (
                f"For the {row_label}, layer-0 INPUT and self_attn_out both ~1.0, but "
                f"layer-0 OUTPUT cos is {mean_out:.4f}.  Bug is inside "
                "layer-0 MLP (post-attention_layernorm + gate/up/down).  "
                "This is unlikely for a deterministic MLP; re-check the "
                "probe's d0 row alignment before trusting this verdict."
            )
        else:
            verdict["interpretation_hint"] = (
                f"All layer-0 taps for the {row_label} match at ~1.0 across matched pairs.  "
                "This CONTRADICTS the Test-P localization and means the "
                "tree-vs-chain gap must arise BETWEEN layer 0 and layer 1 "
                "(e.g. residual normalization, positions-tensor "
                "reshaping) or probe alignment is off.  Double-check "
                "d0_row and num_query_per_req on both runs."
            )
    except Exception as _ve:
        verdict["verdict_error"] = str(_ve)

    return {
        "cosine_threshold": float(cosine_threshold),
        "sample_index": int(sample_index),
        "tap_order": tap_order,
        "tree_captured_positions": sorted(tree_captured_by_position.keys()),
        "chain_captured_positions": sorted(
            chain_captured_by_position.keys()
        ),
        "matched_positions": matched_positions,
        "per_matched_pair": per_pair,
        "summary": summary_out,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Test M: per-position context-K/V alignment across the full context window.
#
# Test L confirmed that layer-1 self_attn_out cosine collapses from 0.9997
# at step 0 to ~0.09 at step 30.  Test L also surfaced two red flags: vLLM's
# reported ``num_context`` pins at 255 while HF's grows linearly (150 -> 180),
# and V at the *last* context slot goes from cos 0.9998 -> -0.04.  Test M
# answers the follow-up questions:
#
#   1. Is ``num_context=255`` in vLLM a probe artifact, or is vLLM actually
#      feeding a 255-wide context window into precompute_and_store_context_kv?
#   2. Do the HF ctx positions [0..HF_ctx-1] match vLLM ctx positions
#      [0..HF_ctx-1] position-by-position (overlap region)?  If not, where
#      does divergence first appear?
#   3. What lives in vLLM's tail positions [HF_ctx..vLLM_ctx-1]?  Are they
#      zero (pre-allocated but unused), stale (leftover from a previous
#      iteration, Pγ), or populated with plausible-looking hidden-state K/V?
#
# The probe on both sides dumps a compact per-position fingerprint (norm,
# abs_mean, first-4 elements of the flattened kv-dim vector).  ``first_k``
# is enough to do a cheap per-position HF-vs-vLLM cosine without storing
# full (ctx_len, kv_dim) tensors.
# ---------------------------------------------------------------------------


def build_draft_layer1_context_kv_alignment_report(
    hf_test_l_payload: dict[str, Any],
    vllm_capture: dict[str, Any],
    tail_mismatch_norm_ratio_threshold: float = 0.2,
    first_k_cosine_threshold: float = 0.95,
) -> dict[str, Any]:
    """Test M report -- per-position context K/V alignment HF vs vLLM.

    Parameters
    ----------
    hf_test_l_payload : deserialized ``reference_capture_test_l.json``.
        Each ``per_step[s]`` slot carries a ``ctx_per_position`` dict with
        per-position ``norm``/``abs_mean``/``first_k`` arrays for
        ``k_ctx_pre_rope``, ``k_ctx_post_rope``, ``v_ctx``.
    vllm_capture : output of ``retrieve_vllm_test_l_probe``; per-step slots
        carry the matching ``ctx_per_position`` dict plus
        ``context_positions`` / ``context_slot_mapping`` lists.
    tail_mismatch_norm_ratio_threshold : any overlap position whose
        ``|vllm_norm - hf_norm| / max(hf_norm, eps)`` exceeds this is
        flagged as a mismatch.
    first_k_cosine_threshold : overlap positions whose ``first_k`` cosine
        falls below this are counted as "low cos" in the summary.

    Returns a JSON-serializable report keyed by speculative step.
    """
    import math  # noqa: PLC0415

    ctx_taps = ("k_ctx_pre_rope", "k_ctx_post_rope", "v_ctx")

    # Bucket HF steps.
    hf_steps: dict[int, dict[str, Any]] = {}
    hf_per_step = (hf_test_l_payload or {}).get("per_step") or {}
    for k, v in hf_per_step.items():
        try:
            hf_steps[int(k)] = v
        except Exception:
            continue

    # Bucket vLLM steps.
    vllm_steps: dict[int, dict[str, Any]] = {}
    for slot in vllm_capture.get("per_step") or []:
        try:
            vllm_steps[int(slot.get("step", -1))] = slot
        except Exception:
            continue

    common_steps = sorted(set(hf_steps.keys()) & set(vllm_steps.keys()))

    def _first_k_cos(a: list[float], b: list[float]) -> float | None:
        if not a or not b:
            return None
            
        n = min(len(a), len(b))
        if n == 0:
            return None
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i in range(n):
            ai = float(a[i])
            bi = float(b[i])
            dot += ai * bi
            na += ai * ai
            nb += bi * bi
        if na <= 0 or nb <= 0:
            return None
        return dot / (math.sqrt(na) * math.sqrt(nb))

    def _compare_one_tap(
        hf_stats: dict[str, Any] | None,
        vllm_stats: dict[str, Any] | None,
        overlap_n: int,
    ) -> dict[str, Any]:
        if hf_stats is None or vllm_stats is None:
            return {"status": "missing"}
        hf_norm = hf_stats.get("norm") or []
        vllm_norm = vllm_stats.get("norm") or []
        hf_first = hf_stats.get("first_k") or []
        vllm_first = vllm_stats.get("first_k") or []
        hf_abs = hf_stats.get("abs_mean") or []
        vllm_abs = vllm_stats.get("abs_mean") or []

        # Overlap-region position-wise parity.
        first_k_cos: list[float | None] = []
        norm_ratios: list[float] = []
        low_cos_positions: list[int] = []
        large_norm_positions: list[int] = []
        for p in range(overlap_n):
            a = hf_first[p] if p < len(hf_first) else []
            b = vllm_first[p] if p < len(vllm_first) else []
            c = _first_k_cos(a, b)
            first_k_cos.append(c)
            if c is not None and c < first_k_cosine_threshold:
                low_cos_positions.append(p)
            hn = float(hf_norm[p]) if p < len(hf_norm) else 0.0
            vn = float(vllm_norm[p]) if p < len(vllm_norm) else 0.0
            denom = max(hn, 1e-6)
            ratio = abs(vn - hn) / denom
            norm_ratios.append(ratio)
            if ratio > tail_mismatch_norm_ratio_threshold:
                large_norm_positions.append(p)

        # First overlap position whose first_k cos drops below the threshold
        # (the "cliff").  If the HF and vLLM per-position K/V are aligned
        # they should all be ~1.0; if vLLM writes stale data at position N
        # onward, this will point at N.
        first_divergent_position: int | None = None
        for p, c in enumerate(first_k_cos):
            if c is not None and c < first_k_cosine_threshold:
                first_divergent_position = p
                break

        # Aggregate stats on overlap region.
        overlap_stats: dict[str, Any] = {
            "overlap_n": int(overlap_n),
            "low_cos_count": int(len(low_cos_positions)),
            "large_norm_count": int(len(large_norm_positions)),
            "first_divergent_position": first_divergent_position,
            "first_k_cos_sample": [
                first_k_cos[i]
                for i in (
                    0,
                    overlap_n // 4,
                    overlap_n // 2,
                    max(0, overlap_n - 1),
                )
                if 0 <= i < overlap_n
            ],
        }

        # vLLM-only tail: positions >= overlap_n.
        tail_n = max(0, len(vllm_norm) - overlap_n)
        tail_stats: dict[str, Any] = {"tail_n": int(tail_n)}
        if tail_n > 0:
            tail_slice = vllm_norm[overlap_n : overlap_n + tail_n]
            tail_abs = (
                vllm_abs[overlap_n : overlap_n + tail_n]
                if vllm_abs
                else []
            )
            tail_stats["norm_min"] = float(min(tail_slice))
            tail_stats["norm_max"] = float(max(tail_slice))
            tail_stats["norm_mean"] = float(
                sum(tail_slice) / len(tail_slice)
            )
            if tail_abs:
                tail_stats["abs_mean_mean"] = float(
                    sum(tail_abs) / len(tail_abs)
                )
            zeros = sum(1 for x in tail_slice if float(x) <= 1e-6)
            tail_stats["zero_norm_count"] = int(zeros)

        return {
            "status": "ok",
            "hf_num_positions": int(len(hf_norm)),
            "vllm_num_positions": int(len(vllm_norm)),
            "overlap": overlap_stats,
            "vllm_tail": tail_stats,
        }

    per_step_report: dict[str, Any] = {}
    for step in common_steps:
        hf_slot = hf_steps[step]
        vllm_slot = vllm_steps[step]
        hf_ctx_len = int(hf_slot.get("context_len", 0))
        vllm_ctx_len = int(
            (vllm_slot.get("ctx_kv") or {}).get("num_context")
            or vllm_slot.get("context_len", 0)
            or 0
        )
        overlap = min(hf_ctx_len, vllm_ctx_len)
        hf_cpp = hf_slot.get("ctx_per_position") or {}
        vllm_cpp = vllm_slot.get("ctx_per_position") or {}
        per_tap: dict[str, Any] = {}
        for tap in ctx_taps:
            per_tap[tap] = _compare_one_tap(
                hf_cpp.get(tap), vllm_cpp.get(tap), overlap
            )
        # Positional sanity: are vLLM's context_positions contiguous 0..N-1
        # as HF would have them implicitly?
        ctx_positions = vllm_slot.get("context_positions") or []
        ctx_positions_contig = (
            len(ctx_positions) > 0
            and all(
                int(ctx_positions[i]) == i for i in range(len(ctx_positions))
            )
        )
        ctx_slot_mapping = vllm_slot.get("context_slot_mapping") or []
        per_step_report[str(step)] = {
            "step": int(step),
            "hf_ctx_len": hf_ctx_len,
            "vllm_ctx_len": vllm_ctx_len,
            "overlap_n": int(overlap),
            "vllm_tail_n": int(max(0, vllm_ctx_len - overlap)),
            "per_ctx_tap": per_tap,
            "vllm_context_positions_len": int(len(ctx_positions)),
            "vllm_context_positions_contiguous_from_zero": bool(
                ctx_positions_contig
            ),
            "vllm_context_positions_head": ctx_positions[:8],
            "vllm_context_positions_tail": ctx_positions[-8:]
            if len(ctx_positions) >= 8
            else ctx_positions,
            "vllm_context_slot_mapping_len": int(len(ctx_slot_mapping)),
            "vllm_context_slot_mapping_head": ctx_slot_mapping[:8],
            "vllm_context_slot_mapping_tail": ctx_slot_mapping[-8:]
            if len(ctx_slot_mapping) >= 8
            else ctx_slot_mapping,
        }

    # Verdict -- focus on ``v_ctx`` since that collapsed to cos ~0 in Test L.
    verdict: dict[str, Any] = {"steps": common_steps}
    cliff_positions: dict[str, int | None] = {}
    for tap in ctx_taps:
        cliffs = []
        for step in common_steps:
            report = per_step_report.get(str(step), {}).get("per_ctx_tap", {})
            slot = report.get(tap) or {}
            if slot.get("status") == "ok":
                fdp = (slot.get("overlap") or {}).get(
                    "first_divergent_position"
                )
                if fdp is not None:
                    cliffs.append((step, fdp))
        cliff_positions[tap] = cliffs[0][1] if cliffs else None

    # Pin-vs-grow check -- is vLLM's ctx_len constant while HF grows?
    hf_ctx_lens = [
        per_step_report.get(str(s), {}).get("hf_ctx_len") for s in common_steps
    ]
    vllm_ctx_lens = [
        per_step_report.get(str(s), {}).get("vllm_ctx_len") for s in common_steps
    ]
    verdict["hf_ctx_lens"] = hf_ctx_lens
    verdict["vllm_ctx_lens"] = vllm_ctx_lens
    try:
        hf_grows = len(set(hf_ctx_lens)) > 1
        vllm_pins = (
            len(set(x for x in vllm_ctx_lens if x is not None)) == 1
            and common_steps
            and common_steps[0] != common_steps[-1]
        )
    except Exception:
        hf_grows = False
        vllm_pins = False
    verdict["hf_context_grows"] = bool(hf_grows)
    verdict["vllm_context_pinned"] = bool(vllm_pins)
    verdict["cliff_positions"] = cliff_positions

    interp_parts: list[str] = []
    if vllm_pins and hf_grows:
        interp_parts.append(
            "vLLM's num_context is pinned (not an artifact: "
            "context_states.shape[0] is the ground truth) while HF's grows"
            " linearly -- vLLM is feeding a wider context window than HF."
        )
    if cliff_positions.get("v_ctx") is not None:
        interp_parts.append(
            "v_ctx first diverges at position"
            f" {cliff_positions['v_ctx']} in the overlap region -- this is"
            " the position at which vLLM's V-cache starts disagreeing with"
            " HF's rolling V, so the slot-mapping / write-offset / refresh"
            " cadence is wrong starting there (Pδ)."
        )
    elif any(
        (per_step_report.get(str(s), {}).get("per_ctx_tap") or {})
        .get("v_ctx", {})
        .get("vllm_tail", {})
        .get("tail_n", 0)
        > 0
        for s in common_steps
    ):
        interp_parts.append(
            "No cliff inside the overlap region but vLLM's V-cache has"
            " a non-empty tail beyond HF's context -- attention averages"
            " over stale/pre-allocated V rows (Pγ or Pδ)."
        )
    if interp_parts:
        verdict["interpretation_hint"] = " ".join(interp_parts)

    return {
        "first_k_cosine_threshold": float(first_k_cosine_threshold),
        "tail_mismatch_norm_ratio_threshold": float(
            tail_mismatch_norm_ratio_threshold
        ),
        "ctx_tap_order": list(ctx_taps),
        "per_step": per_step_report,
        "verdict": verdict,
        "hf_steps_present": sorted(hf_steps.keys()),
        "vllm_steps_present": sorted(vllm_steps.keys()),
    }


def build_test_r_kv_position_audit(
    vllm_tree_summary: dict[str, Any],
    vllm_chain_summary: dict[str, Any],
    vllm_tree_l_capture: dict[str, Any],
    vllm_chain_l_capture: dict[str, Any],
    hf_reference_test_l: dict[str, Any] | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    """Test R -- offline audit of per-iter context_positions / slot_mapping.

    Purely offline analysis: consumes already-captured Test-L artifacts
    (tree + chain) + HF reference Test-L ``context_len`` as oracle.

    For each iter on each side we compute:
      * ``decoded_pos_root`` = min(unique(context_positions)) - num_prompt
      * ``decoded_pos_tail`` = max(unique(context_positions)) - num_prompt
      * ``ctx_window_width`` = decoded_pos_tail - decoded_pos_root + 1
      * ``slot_span`` = max(slot) - min(slot) + 1
      * ``num_unique_positions``, ``num_unique_slots``

    Chain's `decoded_pos_tail` at iter N equals the expected oracle
    (``cum_accepted_len`` after iter N); HF's ``ctx_len`` equals the same
    oracle.  Tree's `decoded_pos_tail` should equal the same value at a
    trajectory-matched iter; any systematic mismatch pins the hypothesis:

    * **H2 (precompute writes wrong positions)** --
      tree's ``decoded_pos_root`` / ``decoded_pos_tail`` deviate from
      chain/HF oracle by a **deterministic function of the current iter's
      topology** (e.g., constant offset or scales with ``tree_width``).

    * **H3 (accept-path propagation leaves KV inconsistent)** --
      tree's offset **correlates with cumulative tree-vs-chain accepted-len
      delta at same decoded position**, i.e., accumulates over iters.

    * **H5-slot (slot-mapping / position-ID off-by-depth)** --
      tree's ``num_unique_positions`` vs ``ctx_window_width`` diverges
      (e.g., gaps or stride-based) while chain's are contiguous.

    * **H1 (stale slots from rejected branches)** --
      positions tree writes are **correct** (match chain / HF oracle), but
      at matched decoded_pos the observed per-position cosines in Test Q
      were already low; content corruption is consistent with H1 only if
      position bookkeeping is clean here.  If tree positions are wrong,
      H1 is subsumed by H2/H5.
    """

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> list[dict[str, Any]]:
        for s in summary.get("samples") or []:
            if int(s.get("sample_index", -1)) != int(sample_idx):
                continue
            steps_sorted = sorted(
                s.get("steps") or [], key=lambda e: int(e.get("step", 0))
            )
            cumulative = 0
            out: list[dict[str, Any]] = []
            for e in steps_sorted:
                enriched = dict(e)
                enriched["__position"] = cumulative
                out.append(enriched)
                try:
                    cumulative += int(e.get("accepted_len", 0)) + 1
                except (TypeError, ValueError):
                    cumulative += 1
            return out
        return []

    def _summary_num_prompt(
        summary: dict[str, Any], sample_idx: int
    ) -> int | None:
        for s in summary.get("samples") or []:
            if int(s.get("sample_index", -1)) != int(sample_idx):
                continue
            for k in (
                "num_prompt_tokens",
                "prompt_length",
                "num_prompt",
            ):
                v = s.get(k)
                if v is not None:
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        pass
        return None

    def _l_by_step(
        cap: dict[str, Any] | None,
    ) -> dict[int, dict[str, Any]]:
        if cap is None:
            return {}
        per_step = cap.get("per_step")
        if isinstance(per_step, dict):
            entries = per_step.values()
        else:
            entries = per_step or []
        by_step: dict[int, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                s_idx = int(entry.get("step"))
            except Exception:
                continue
            by_step[s_idx] = entry
        return by_step

    tree_iters = _per_iter_positions(vllm_tree_summary, sample_index)
    chain_iters = _per_iter_positions(vllm_chain_summary, sample_index)
    tree_l = _l_by_step(vllm_tree_l_capture)
    chain_l = _l_by_step(vllm_chain_l_capture)
    hf_l = _l_by_step(hf_reference_test_l)

    # The prompt length is not currently serialised in vLLM summaries
    # (noted in Test-P fix); we infer it from the prefill-step's
    # ``context_len`` in the Test-L capture, which equals
    # ``num_prompt_tokens`` at step 0.
    def _infer_prompt_len(
        by_step: dict[int, dict[str, Any]],
        summary: dict[str, Any],
        sample_idx: int,
    ) -> int | None:
        v = _summary_num_prompt(summary, sample_idx)
        if v is not None:
            return v
        # step-0 prefill carries context_len == prompt_len.
        step0 = by_step.get(0)
        if step0 is not None:
            try:
                return int(step0.get("context_len"))
            except (TypeError, ValueError):
                pass
        return None

    num_prompt_tree = _infer_prompt_len(
        tree_l, vllm_tree_summary, sample_index
    )
    num_prompt_chain = _infer_prompt_len(
        chain_l, vllm_chain_summary, sample_index
    )
    num_prompt_hf = None
    # HF reference capture's step 0 context_len == num_prompt as well.
    hf_step0 = hf_l.get(0)
    if hf_step0 is not None:
        try:
            num_prompt_hf = int(hf_step0.get("context_len"))
        except (TypeError, ValueError):
            num_prompt_hf = None

    def _per_iter_table(
        iters: list[dict[str, Any]],
        by_step: dict[int, dict[str, Any]],
        num_prompt: int | None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cumulative_accepted = 0
        for it in iters:
            step = int(it.get("step", -1))
            accepted_len = int(it.get("accepted_len", 0) or 0)
            entry = by_step.get(step)
            row: dict[str, Any] = {
                "step": step,
                "decoded_pos_root_from_summary": int(
                    it.get("__position", 0)
                ),
                "accepted_len": accepted_len,
                "cum_accepted_before_iter": cumulative_accepted,
            }
            cum_after = cumulative_accepted + accepted_len + 1
            row["cum_accepted_after_iter"] = int(cum_after)
            # Oracle last context position (absolute): num_prompt +
            # cum_accepted_after_iter - 1 (or cum_before, depending on
            # convention).  We record both and let the comparison decide.
            if num_prompt is not None:
                row["expected_ctx_pos_min_absolute"] = int(
                    num_prompt + cumulative_accepted
                )
                row["expected_ctx_pos_max_absolute_tail"] = int(
                    num_prompt + cum_after - 1
                )
            else:
                row["expected_ctx_pos_min_absolute"] = None
                row["expected_ctx_pos_max_absolute_tail"] = None

            cp = None
            sm = None
            ctx_len = None
            if entry is not None:
                cp = entry.get("context_positions")
                sm = entry.get("context_slot_mapping")
                try:
                    ctx_len = int(entry.get("context_len"))
                except (TypeError, ValueError):
                    ctx_len = None
            row["ctx_len"] = ctx_len
            if isinstance(cp, list) and cp:
                uniq_pos = sorted(set(int(x) for x in cp))
                row["num_unique_positions"] = len(uniq_pos)
                row["unique_pos_min"] = uniq_pos[0]
                row["unique_pos_max"] = uniq_pos[-1]
                row["ctx_window_width"] = (
                    uniq_pos[-1] - uniq_pos[0] + 1
                )
                row["is_contiguous"] = bool(
                    len(uniq_pos) == row["ctx_window_width"]
                )
                if num_prompt is not None:
                    row["decoded_pos_root"] = int(uniq_pos[0] - num_prompt)
                    row["decoded_pos_tail"] = int(uniq_pos[-1] - num_prompt)
                    row["delta_vs_expected_min"] = int(
                        uniq_pos[0] - row["expected_ctx_pos_min_absolute"]
                    )
                    row["delta_vs_expected_tail"] = int(
                        uniq_pos[-1]
                        - row["expected_ctx_pos_max_absolute_tail"]
                    )
                else:
                    row["decoded_pos_root"] = None
                    row["decoded_pos_tail"] = None
                    row["delta_vs_expected_min"] = None
                    row["delta_vs_expected_tail"] = None
            else:
                row["num_unique_positions"] = 0
                row["unique_pos_min"] = None
                row["unique_pos_max"] = None
                row["ctx_window_width"] = None
                row["is_contiguous"] = None
                row["decoded_pos_root"] = None
                row["decoded_pos_tail"] = None
                row["delta_vs_expected_min"] = None
                row["delta_vs_expected_tail"] = None

            if isinstance(sm, list) and sm:
                uniq_slot = sorted(set(int(x) for x in sm))
                row["num_unique_slots"] = len(uniq_slot)
                row["unique_slot_min"] = uniq_slot[0]
                row["unique_slot_max"] = uniq_slot[-1]
                row["slot_span"] = uniq_slot[-1] - uniq_slot[0] + 1
                row["slots_contiguous"] = bool(
                    len(uniq_slot) == row["slot_span"]
                )
            else:
                row["num_unique_slots"] = 0
                row["unique_slot_min"] = None
                row["unique_slot_max"] = None
                row["slot_span"] = None
                row["slots_contiguous"] = None

            out.append(row)
            cumulative_accepted = cum_after
        return out

    tree_rows = _per_iter_table(tree_iters, tree_l, num_prompt_tree)
    chain_rows = _per_iter_table(chain_iters, chain_l, num_prompt_chain)

    # HF rows: HF Test-L capture is keyed by step and carries ``context_len``
    # but no ``context_positions``.  We use ``ctx_len`` as the oracle for
    # absolute position last-index (ctx_len == num_prompt +
    # cum_accepted_after).
    hf_rows: list[dict[str, Any]] = []
    for step_key in sorted(hf_l.keys()):
        entry = hf_l[step_key]
        try:
            cl = int(entry.get("context_len"))
        except (TypeError, ValueError):
            cl = None
        row = {"step": int(step_key), "ctx_len": cl}
        if cl is not None and num_prompt_hf is not None:
            row["decoded_pos_tail"] = int(cl - num_prompt_hf)
        hf_rows.append(row)

    def _rows_by_decoded_pos_root(
        rows: list[dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for r in rows:
            pos = r.get("decoded_pos_root")
            if pos is None:
                continue
            out[int(pos)] = r
        return out

    tree_by_pos = _rows_by_decoded_pos_root(tree_rows)
    chain_by_pos = _rows_by_decoded_pos_root(chain_rows)
    matched_positions = sorted(
        set(tree_by_pos.keys()) & set(chain_by_pos.keys())
    )

    position_matched: list[dict[str, Any]] = []
    for pos in matched_positions:
        t = tree_by_pos[pos]
        c = chain_by_pos[pos]
        entry: dict[str, Any] = {
            "decoded_pos_root": int(pos),
            "tree_step": t.get("step"),
            "chain_step": c.get("step"),
            "tree_unique_pos": (
                t.get("unique_pos_min"),
                t.get("unique_pos_max"),
            ),
            "chain_unique_pos": (
                c.get("unique_pos_min"),
                c.get("unique_pos_max"),
            ),
            "tree_unique_slot": (
                t.get("unique_slot_min"),
                t.get("unique_slot_max"),
            ),
            "chain_unique_slot": (
                c.get("unique_slot_min"),
                c.get("unique_slot_max"),
            ),
            "tree_ctx_width": t.get("ctx_window_width"),
            "chain_ctx_width": c.get("ctx_window_width"),
            "tree_num_unique_positions": t.get("num_unique_positions"),
            "chain_num_unique_positions": c.get("num_unique_positions"),
            "tree_contiguous": t.get("is_contiguous"),
            "chain_contiguous": c.get("is_contiguous"),
            "tree_slots_contiguous": t.get("slots_contiguous"),
            "chain_slots_contiguous": c.get("slots_contiguous"),
            "tree_delta_vs_expected_min": t.get(
                "delta_vs_expected_min"
            ),
            "chain_delta_vs_expected_min": c.get(
                "delta_vs_expected_min"
            ),
            "tree_delta_vs_expected_tail": t.get(
                "delta_vs_expected_tail"
            ),
            "chain_delta_vs_expected_tail": c.get(
                "delta_vs_expected_tail"
            ),
        }
        if t.get("unique_pos_min") is not None and c.get(
            "unique_pos_min"
        ) is not None:
            entry["tree_minus_chain_pos_min"] = int(
                t["unique_pos_min"] - c["unique_pos_min"]
            )
            entry["tree_minus_chain_pos_max"] = int(
                t["unique_pos_max"] - c["unique_pos_max"]
            )
        if t.get("unique_slot_min") is not None and c.get(
            "unique_slot_min"
        ) is not None:
            entry["tree_minus_chain_slot_min"] = int(
                t["unique_slot_min"] - c["unique_slot_min"]
            )
            entry["tree_minus_chain_slot_max"] = int(
                t["unique_slot_max"] - c["unique_slot_max"]
            )
        position_matched.append(entry)

    # Aggregates.
    def _stats(values: list[int]) -> dict[str, Any]:
        vs = [int(v) for v in values if v is not None]
        if not vs:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "all_zero": None,
                "nonzero_count": 0,
            }
        nonzero = sum(1 for v in vs if v != 0)
        return {
            "count": len(vs),
            "min": min(vs),
            "max": max(vs),
            "mean": float(sum(vs) / len(vs)),
            "all_zero": bool(nonzero == 0),
            "nonzero_count": int(nonzero),
        }

    tree_deltas_min = [r.get("delta_vs_expected_min") for r in tree_rows]
    tree_deltas_tail = [r.get("delta_vs_expected_tail") for r in tree_rows]
    chain_deltas_min = [r.get("delta_vs_expected_min") for r in chain_rows]
    chain_deltas_tail = [
        r.get("delta_vs_expected_tail") for r in chain_rows
    ]
    pm_tree_vs_chain_min = [
        e.get("tree_minus_chain_pos_min") for e in position_matched
    ]
    pm_tree_vs_chain_max = [
        e.get("tree_minus_chain_pos_max") for e in position_matched
    ]
    pm_tree_vs_chain_slot_min = [
        e.get("tree_minus_chain_slot_min") for e in position_matched
    ]
    pm_tree_vs_chain_slot_max = [
        e.get("tree_minus_chain_slot_max") for e in position_matched
    ]

    aggregates = {
        "tree_delta_vs_expected_min_stats": _stats(tree_deltas_min),
        "tree_delta_vs_expected_tail_stats": _stats(tree_deltas_tail),
        "chain_delta_vs_expected_min_stats": _stats(chain_deltas_min),
        "chain_delta_vs_expected_tail_stats": _stats(chain_deltas_tail),
        "matched_positions_count": int(len(matched_positions)),
        "pos_matched_tree_minus_chain_min_stats": _stats(
            pm_tree_vs_chain_min
        ),
        "pos_matched_tree_minus_chain_max_stats": _stats(
            pm_tree_vs_chain_max
        ),
        "pos_matched_tree_minus_chain_slot_min_stats": _stats(
            pm_tree_vs_chain_slot_min
        ),
        "pos_matched_tree_minus_chain_slot_max_stats": _stats(
            pm_tree_vs_chain_slot_max
        ),
    }

    # Contiguity / distinctness flags.
    tree_any_noncontig = any(
        r.get("is_contiguous") is False for r in tree_rows
    )
    chain_any_noncontig = any(
        r.get("is_contiguous") is False for r in chain_rows
    )
    tree_any_slot_noncontig = any(
        r.get("slots_contiguous") is False for r in tree_rows
    )
    chain_any_slot_noncontig = any(
        r.get("slots_contiguous") is False for r in chain_rows
    )
    aggregates["tree_any_positions_noncontiguous"] = bool(
        tree_any_noncontig
    )
    aggregates["chain_any_positions_noncontiguous"] = bool(
        chain_any_noncontig
    )
    aggregates["tree_any_slots_noncontiguous"] = bool(
        tree_any_slot_noncontig
    )
    aggregates["chain_any_slots_noncontiguous"] = bool(
        chain_any_slot_noncontig
    )

    # HF oracle cross-check: does chain's unique_pos_max == HF's ctx_len-1?
    hf_ctx_len_by_step = {r["step"]: r.get("ctx_len") for r in hf_rows}
    chain_vs_hf: list[dict[str, Any]] = []
    tree_vs_hf: list[dict[str, Any]] = []
    for rows, out in (
        (chain_rows, chain_vs_hf),
        (tree_rows, tree_vs_hf),
    ):
        for r in rows:
            step = r.get("step")
            if step not in hf_ctx_len_by_step:
                continue
            cl = hf_ctx_len_by_step[step]
            if cl is None or r.get("unique_pos_max") is None:
                continue
            # Same relative tail position?  HF: ctx_len - 1 absolute.
            rel_hf = int(cl) - 1
            rel_side = int(r["unique_pos_max"])
            out.append(
                {
                    "step": step,
                    "hf_abs_tail": rel_hf,
                    "side_abs_tail": rel_side,
                    "side_minus_hf_tail": int(rel_side - rel_hf),
                }
            )

    aggregates["chain_vs_hf_same_iter_tail_stats"] = _stats(
        [e["side_minus_hf_tail"] for e in chain_vs_hf]
    )
    aggregates["tree_vs_hf_same_iter_tail_stats"] = _stats(
        [e["side_minus_hf_tail"] for e in tree_vs_hf]
    )

    # Discriminative verdict.
    #
    # We do NOT anchor the verdict to the summary's ``num_prompt +
    # cum_accepted`` oracle -- empirically the summary's cumulative
    # accepted-count can disagree with the capture's observed
    # ``unique_pos_min`` on the tree side (an independently interesting
    # reporting artefact).  Instead we score the four live hypotheses
    # directly from position-matched tree-vs-chain deltas + contiguity,
    # which is exactly the invariant Test Q was designed to expose.

    verdict: dict[str, Any] = {}
    reasons: list[str] = []

    h_scores: dict[str, str] = {}

    # H4 already refuted by Test Q at position 0.
    h_scores["H4_tree_attn_kernel_bug"] = (
        "refuted_by_test_q_position_0"
    )

    matched_count = int(aggregates["matched_positions_count"])
    # Exclude the prefill (decoded_pos = -num_prompt) from the score,
    # since step 0 is mechanically identical on both sides and tells us
    # nothing about the spec-loop bookkeeping.
    matched_spec = [
        e
        for e in position_matched
        if int(e.get("decoded_pos_root", -num_prompt_chain or 0)) >= 0
    ]
    matched_spec_count = int(len(matched_spec))

    pm_min_all_zero = aggregates[
        "pos_matched_tree_minus_chain_min_stats"
    ].get("all_zero")
    pm_slot_min_all_zero = aggregates[
        "pos_matched_tree_minus_chain_slot_min_stats"
    ].get("all_zero")
    pm_max_stats = aggregates["pos_matched_tree_minus_chain_max_stats"]

    verdict["matched_positions_total"] = int(matched_count)
    verdict["matched_positions_spec_only"] = int(matched_spec_count)
    verdict["tree_minus_chain_min_all_zero"] = bool(pm_min_all_zero)
    verdict["tree_minus_chain_slot_min_all_zero"] = bool(
        pm_slot_min_all_zero
    )
    verdict["tree_minus_chain_max_range"] = (
        pm_max_stats.get("min"),
        pm_max_stats.get("max"),
    )
    verdict["tree_positions_contiguous"] = not bool(
        aggregates["tree_any_positions_noncontiguous"]
    )
    verdict["tree_slots_contiguous"] = not bool(
        aggregates["tree_any_slots_noncontiguous"]
    )

    if matched_spec_count == 0:
        reasons.append(
            "No matched spec-iter decoded-positions found -- cannot"
            " score position-bookkeeping hypotheses.  Run with denser"
            " Test-L step coverage or ensure both sides share early"
            " captured iters."
        )
        for k in (
            "H1_stale_rejected_branch_slots",
            "H2_precompute_wrong_positions",
            "H3_accept_path_propagation_drift",
            "H5_slot_mapping_offbydepth",
        ):
            h_scores[k] = "cannot_score"
    else:
        # H2 (precompute writes wrong positions).
        if (
            pm_min_all_zero
            and not aggregates["tree_any_positions_noncontiguous"]
        ):
            h_scores["H2_precompute_wrong_positions"] = (
                "refuted"
                " (at every matched decoded-position, tree's context"
                " window-min equals chain's; tree's positions are"
                " contiguous; tree only extends further ahead by up to"
                f" {pm_max_stats.get('max')} positions -- exactly the"
                " tree_width - 1 forward-virtual offset that is"
                " expected of tree-spec precompute)"
            )
            reasons.append(
                "Tree writes to the same start position as chain at"
                " matched decoded-positions; any supposed offset in"
                " Test Q (e.g. '-2 at pos=36') was an artefact of"
                " matching by summary.__position rather than by the"
                " captured uniq_min."
            )
        else:
            h_scores["H2_precompute_wrong_positions"] = (
                "supported"
                " (tree's start position disagrees with chain's at"
                " matched decoded-positions, or tree has non-contiguous"
                " writes)"
            )

        # H5 (slot-mapping / position-ID off-by-depth).
        if (
            pm_slot_min_all_zero
            and not aggregates["tree_any_slots_noncontiguous"]
            and pm_min_all_zero
        ):
            h_scores["H5_slot_mapping_offbydepth"] = (
                "refuted"
                " (tree and chain map identical start-positions to"
                " identical start-slots; tree's slots are contiguous;"
                " no off-by-depth stride is observed)"
            )
        else:
            h_scores["H5_slot_mapping_offbydepth"] = (
                "supported"
                " (slot-min disagrees or slots are non-contiguous)"
            )

        # H3 (accept-path propagation drift) -- would manifest as
        # pm-min-offsets that accumulate with iter / matched-pos.
        # With pm-min all zero this is not happening for the WRITE
        # path; but H3 could still corrupt the READ-path content
        # (which is the cross-attn context V on older positions).
        if pm_min_all_zero:
            h_scores["H3_accept_path_propagation_drift"] = (
                "refuted_for_write_path"
                " (accept-path cannot be leaving stale start-positions"
                " in the draft-KV write window -- but it could still"
                " be leaving stale CONTENT at previously-written"
                " positions, which is H1 territory)"
            )
        else:
            h_scores["H3_accept_path_propagation_drift"] = (
                "supported"
                " (start-position drifts across matched pairs)"
            )

        # H1 (stale content at correct positions) -- the only
        # hypothesis that survives when positions + slots are
        # pristine AND Test Q still shows layer-0 self_attn_out
        # divergence at matched decoded-position + matched input
        # hidden state.
        if (
            pm_min_all_zero
            and pm_slot_min_all_zero
            and not aggregates["tree_any_positions_noncontiguous"]
            and not aggregates["tree_any_slots_noncontiguous"]
        ):
            h_scores["H1_stale_rejected_branch_slots"] = (
                "primary_suspect_by_elimination"
                " (all position/slot bookkeeping checks pass, yet"
                " Test Q shows layer-0 self_attn_out diverges at"
                " matched decoded-position 36 even with identical"
                " input-hidden -- the remaining mechanism is stale"
                " content in paged KV slots that are logically"
                " 'live' for the draft but were last written by a"
                " rejected speculative branch)"
            )
            reasons.append(
                "Positions + slots are clean; stale-content in paged"
                " KV (H1) is the only mechanism that can explain"
                " Test Q's position-36 divergence with matching"
                " input-hidden.  Recommend: inspect whether tree-spec"
                " write-back of target hidden-states for ACCEPTED"
                " tokens overwrites the same slots that REJECTED"
                " branches wrote to in the prior iter, vs. chain-spec"
                " where the write-back path is single-token."
            )
        else:
            h_scores["H1_stale_rejected_branch_slots"] = (
                "inconclusive"
                " (position / slot bookkeeping is imperfect; H1 vs"
                " H2/H3/H5 cannot be cleanly disentangled yet)"
            )

    # Summary-vs-capture drift: an independently interesting signal.
    # Chain's summary.__position should equal chain's uniq_min -
    # num_prompt at every iter.  Tree empirically drifts.
    def _summary_vs_capture_drift(
        iters: list[dict[str, Any]],
        by_step: dict[int, dict[str, Any]],
        num_prompt: int | None,
    ) -> dict[str, Any]:
        if num_prompt is None:
            return {"count": 0}
        cumulative = 0
        diffs: list[int] = []
        for it in iters:
            step = int(it.get("step", -1))
            entry = by_step.get(step)
            if entry is not None:
                cp = entry.get("context_positions")
                if isinstance(cp, list) and cp:
                    um = min(int(x) for x in cp)
                    diffs.append(int((um - num_prompt) - cumulative))
            accepted_len = int(it.get("accepted_len", 0) or 0)
            cumulative += accepted_len + 1
        if not diffs:
            return {"count": 0}
        nonzero = sum(1 for d in diffs if d != 0)
        return {
            "count": len(diffs),
            "min": min(diffs),
            "max": max(diffs),
            "mean": float(sum(diffs) / len(diffs)),
            "nonzero_count": int(nonzero),
            "all_zero": bool(nonzero == 0),
        }

    verdict["chain_summary_vs_capture_uniq_min_drift"] = (
        _summary_vs_capture_drift(chain_iters, chain_l, num_prompt_chain)
    )
    verdict["tree_summary_vs_capture_uniq_min_drift"] = (
        _summary_vs_capture_drift(tree_iters, tree_l, num_prompt_tree)
    )

    verdict["hypothesis_scores"] = h_scores
    verdict["reasons"] = reasons

    return {
        "sample_index": int(sample_index),
        "num_prompt_tokens": {
            "tree": num_prompt_tree,
            "chain": num_prompt_chain,
            "hf": num_prompt_hf,
        },
        "tree_per_iter": tree_rows,
        "chain_per_iter": chain_rows,
        "hf_per_iter": hf_rows,
        "position_matched": position_matched,
        "chain_vs_hf_same_iter": chain_vs_hf,
        "tree_vs_hf_same_iter": tree_vs_hf,
        "aggregates": aggregates,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Test S: per-position context K/V content A/B (tree vs chain) at matched
# decoded positions.  Directly scores H1 (stale KV content at accepted-prefix
# paged slots) vs H2 (wrong write-back content at the tail precompute window).
#
# Inputs reused (no new inference):
#   - vllm_tree_capture / vllm_chain_capture (Test-L captures).  Each per-step
#     slot carries:
#       * ``context_positions`` / ``context_slot_mapping`` (len == context_len)
#       * ``ctx_per_position`` : dict of {tap_name: {
#             "num_positions": int,
#             "kv_dim": int,
#             "first_k_width": int,
#             "norm": list[float] (len == num_positions),
#             "abs_mean": list[float] (len == num_positions),
#             "first_k": list[list[float]] (shape (num_positions, first_k_width)),
#         }} for taps ``k_ctx_pre_rope``, ``k_ctx_post_rope``, ``v_ctx``.
#   - vllm_tree_summary / vllm_chain_summary (for decoded-position matching).
#   - Optional hf_test_l_payload (HF oracle at chain-greedy positions).
#
# Per matched (tree_iter, chain_iter) pair at the same decoded position P:
#   * Enumerate unique decoded positions present in both sides'
#     ``context_positions`` for that iter.
#   * For each shared absolute position ``abs_pos``:
#       - index_tree  = tree_positions.index(abs_pos)
#       - index_chain = chain_positions.index(abs_pos)
#       - per-tap cosine on first_k[index_tree] vs first_k[index_chain]
#       - |norm_diff| / max_norm, |abs_mean_diff| / max_abs_mean
#   * Classify each shared position as "accepted-prefix" (abs_pos <
#     num_prompt + P) or "tail" (abs_pos >= num_prompt + P).
#
# Verdict logic (scores H1 / H2 / H3-write):
#   - H1  suspected  if any matched pair has ``any`` accepted-prefix shared
#     position with first_k cos < threshold.
#   - H2  suspected  if matched pairs show divergence only at tail positions
#     (newly-computed tree-width entries that tree writes but chain does
#     not).
#   - Refuted if every shared position across all matched pairs has cos ~
#     1.0 on all three taps.
#
# NOTE: ``first_k_width`` is now 32 on the writer side but existing captures
# from earlier runs carry width 4.  The audit honors whatever width is
# available per capture (cosine on 4 floats is noisier but still
# directionally discriminating).
# ---------------------------------------------------------------------------


def build_test_s_ctx_content_report(
    vllm_tree_capture: dict[str, Any],
    vllm_chain_capture: dict[str, Any],
    vllm_tree_summary: dict[str, Any],
    vllm_chain_summary: dict[str, Any],
    hf_test_l_payload: dict[str, Any] | None = None,
    sample_index: int = 0,
    cosine_threshold: float = 0.99,
    norm_ratio_threshold: float = 0.05,
) -> dict[str, Any]:
    """Test S: tree-vs-chain per-position context K/V content audit.

    Parameters
    ----------
    vllm_tree_capture, vllm_chain_capture:
        Test-L probe captures (tree-spec and chain-spec sides).  Each
        ``per_step[s]`` slot must carry ``context_positions`` and
        ``ctx_per_position`` (with ``first_k`` arrays for the three context
        taps).
    vllm_tree_summary, vllm_chain_summary:
        Draft-quality summaries used to reconstruct decoded-position per
        iter (mirrors ``_per_iter_positions`` from Test P).
    hf_test_l_payload:
        Optional HF-side Test-L payload (chain-greedy target-forced).  When
        present, each shared position also records HF-vs-tree / HF-vs-chain
        cosines as an oracle anchor.
    cosine_threshold:
        Per-position ``first_k`` cosine under which a shared position is
        flagged as "divergent".
    norm_ratio_threshold:
        ``|a - b| / max(|a|, |b|)`` over norm / abs_mean ratios; used as a
        lighter-weight secondary signal when ``first_k_width`` is small.
    """

    TAP_KEYS = ("k_ctx_pre_rope", "k_ctx_post_rope", "v_ctx")

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> tuple[int, list[dict[str, Any]]]:
        for s in summary.get("samples") or []:
            if int(s.get("sample_index", -1)) != int(sample_idx):
                continue
            try:
                num_prompt = int(s.get("num_prompt_tokens") or 0)
            except (TypeError, ValueError):
                num_prompt = 0
            steps_sorted = sorted(
                s.get("steps") or [], key=lambda e: int(e.get("step", 0))
            )
            cumulative = 0
            out: list[dict[str, Any]] = []
            for e in steps_sorted:
                enriched = dict(e)
                enriched["__position"] = cumulative
                out.append(enriched)
                try:
                    cumulative += int(e.get("accepted_len", 0)) + 1
                except (TypeError, ValueError):
                    cumulative += 1
            return num_prompt, out
        return 0, []

    def _capture_by_step(cap: dict[str, Any]) -> dict[int, dict[str, Any]]:
        by_step: dict[int, dict[str, Any]] = {}
        for entry in cap.get("per_step") or []:
            try:
                s_idx = int(entry.get("step"))
            except Exception:
                continue
            by_step[s_idx] = {
                "context_len": int(entry.get("context_len", 0)),
                "context_positions": list(
                    entry.get("context_positions") or []
                ),
                "context_slot_mapping": list(
                    entry.get("context_slot_mapping") or []
                ),
                "ctx_per_position": entry.get("ctx_per_position") or {},
            }
        return by_step

    def _hf_capture_by_iter(
        hf_payload: dict[str, Any] | None,
    ) -> dict[int, dict[str, Any]]:
        if hf_payload is None:
            return {}
        by_iter: dict[int, dict[str, Any]] = {}
        for str_step, entry in (hf_payload.get("per_step") or {}).items():
            try:
                s_idx = int(entry.get("step", str_step))
            except Exception:
                continue
            by_iter[s_idx] = {
                "context_len": int(entry.get("context_len", 0)),
                "ctx_per_position": entry.get("ctx_per_position") or {},
            }
        return by_iter

    def _first_index(positions: list[int], abs_pos: int) -> int | None:
        try:
            return positions.index(int(abs_pos))
        except ValueError:
            return None

    def _get_tap_entry(
        ctx_per_pos: dict[str, Any],
        tap: str,
        idx: int,
    ) -> dict[str, Any] | None:
        stats = ctx_per_pos.get(tap) if isinstance(ctx_per_pos, dict) else None
        if not isinstance(stats, dict):
            return None
        first_k = stats.get("first_k") or []
        norms = stats.get("norm") or []
        abs_means = stats.get("abs_mean") or []
        if idx is None or idx < 0 or idx >= len(first_k):
            return None
        return {
            "first_k": first_k[idx],
            "norm": norms[idx] if idx < len(norms) else None,
            "abs_mean": abs_means[idx] if idx < len(abs_means) else None,
            "first_k_width": int(stats.get("first_k_width", 0)),
            "kv_dim": int(stats.get("kv_dim", 0)),
        }

    def _ratio(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        m = max(abs(float(a)), abs(float(b)))
        if m < 1e-6:
            return 0.0
        return abs(float(a) - float(b)) / m

    num_prompt_tree, tree_iters = _per_iter_positions(
        vllm_tree_summary, sample_index
    )
    num_prompt_chain, chain_iters = _per_iter_positions(
        vllm_chain_summary, sample_index
    )
    # HF oracle (optional): HF's iter index is the decoded position for
    # chain-greedy target-forced runs.  We don't need num_prompt_hf here
    # because we address HF by iter (== decoded position).
    tree_caps = _capture_by_step(vllm_tree_capture)
    chain_caps = _capture_by_step(vllm_chain_capture)
    hf_caps = _hf_capture_by_iter(hf_test_l_payload)

    # Fallback: older summaries omit ``num_prompt_tokens`` (==0).  At step 0
    # the drafter writes the prompt tokens' context K/V verbatim, so
    # ``max(step0.context_positions) + 1`` equals the prompt length.  Use
    # this as the fallback when the summary lacks the field.
    def _infer_num_prompt_from_step0(
        caps: dict[int, dict[str, Any]],
    ) -> int:
        step0 = caps.get(0)
        if not step0:
            return 0
        positions = step0.get("context_positions") or []
        if not positions:
            return 0
        try:
            return int(max(int(p) for p in positions)) + 1
        except (TypeError, ValueError):
            return 0

    if num_prompt_tree <= 0:
        num_prompt_tree = _infer_num_prompt_from_step0(tree_caps)
    if num_prompt_chain <= 0:
        num_prompt_chain = _infer_num_prompt_from_step0(chain_caps)

    tree_captured_by_position: dict[int, dict[str, Any]] = {}
    for it in tree_iters:
        si = int(it.get("step", -1))
        if si in tree_caps:
            tree_captured_by_position[int(it["__position"])] = {
                "iter": si,
                "iter_entry": it,
                "capture": tree_caps[si],
            }
    chain_captured_by_position: dict[int, dict[str, Any]] = {}
    for it in chain_iters:
        si = int(it.get("step", -1))
        if si in chain_caps:
            chain_captured_by_position[int(it["__position"])] = {
                "iter": si,
                "iter_entry": it,
                "capture": chain_caps[si],
            }

    matched_positions = sorted(
        set(tree_captured_by_position) & set(chain_captured_by_position)
    )

    # Aggregates
    all_tvc_acc: list[float] = []  # accepted-prefix tree-vs-chain
    all_tvc_tail: list[float] = []  # tail-side tree-vs-chain
    per_pair: list[dict[str, Any]] = []

    # Iter-that-wrote-position reconstruction (for attribution when a shared
    # accepted-prefix position is divergent, we tell the user which prior
    # iter wrote content at that position).  A position P is "produced" by
    # the iter whose __position is the largest that is <= P - num_prompt.
    # Since we only control tree-side attribution, we build the tree lookup.
    def _build_iter_for_position(
        iters: list[dict[str, Any]], num_prompt: int
    ) -> dict[int, int]:
        out: dict[int, int] = {}
        for it in iters:
            start = num_prompt + int(it["__position"])
            try:
                accepted = int(it.get("accepted_len", 0)) + 1
            except (TypeError, ValueError):
                accepted = 1
            for p in range(start, start + accepted):
                out[p] = int(it.get("step", -1))
        return out

    tree_iter_for_abs_pos = _build_iter_for_position(
        tree_iters, num_prompt_tree
    )

    for pos in matched_positions:
        t = tree_captured_by_position[pos]
        c = chain_captured_by_position[pos]
        t_cap = t["capture"]
        c_cap = c["capture"]

        t_positions = list(t_cap.get("context_positions") or [])
        c_positions = list(c_cap.get("context_positions") or [])
        t_cpp = t_cap.get("ctx_per_position") or {}
        c_cpp = c_cap.get("ctx_per_position") or {}

        t_unique = sorted(set(int(x) for x in t_positions))
        c_unique = sorted(set(int(x) for x in c_positions))
        shared = sorted(set(t_unique) & set(c_unique))

        # HF oracle: HF uses iter index == decoded position for chain-greedy
        # runs.  Its ctx_per_position at iter ``pos`` corresponds to a
        # sliding window ending at (num_prompt + pos - 1) approximately
        # (convention varies by HF probe); we don't rely on the indexing
        # contract here -- we only report HF-vs-vLLM when HF has an entry
        # for this matched iter AND lengths line up.
        hf_entry = hf_caps.get(pos)

        # Absolute accepted-prefix cutoff: positions < num_prompt + pos are
        # "prior to this iter's root" and hence corrupt iff prior-iter
        # writes leaked stale content there.
        acc_prefix_cutoff = num_prompt_tree + int(pos)

        shared_rows: list[dict[str, Any]] = []
        pair_min_cos_acc: float | None = None
        pair_min_cos_tail: float | None = None
        first_divergent_accepted_prefix_pos: int | None = None
        first_divergent_tail_pos: int | None = None

        for abs_pos in shared:
            i_t = _first_index(t_positions, abs_pos)
            i_c = _first_index(c_positions, abs_pos)
            if i_t is None or i_c is None:
                continue
            row: dict[str, Any] = {
                "abs_pos": int(abs_pos),
                "rel_pos": int(abs_pos - num_prompt_tree),
                "tree_idx": int(i_t),
                "chain_idx": int(i_c),
                "is_accepted_prefix": bool(abs_pos < acc_prefix_cutoff),
                "written_by_tree_iter": tree_iter_for_abs_pos.get(
                    int(abs_pos)
                ),
            }
            per_tap: dict[str, Any] = {}
            row_min_cos: float | None = None
            for tap in TAP_KEYS:
                t_tap = _get_tap_entry(t_cpp, tap, i_t)
                c_tap = _get_tap_entry(c_cpp, tap, i_c)
                tap_row: dict[str, Any] = {}
                cos_val: float | None = None
                if t_tap is not None and c_tap is not None:
                    cos_val = _cosine(
                        list(t_tap["first_k"]),
                        list(c_tap["first_k"]),
                    )
                    tap_row.update(
                        {
                            "tree_vs_chain_cosine": float(cos_val),
                            "tree_vs_chain_norm_ratio": _ratio(
                                t_tap.get("norm"), c_tap.get("norm")
                            ),
                            "tree_vs_chain_abs_mean_ratio": _ratio(
                                t_tap.get("abs_mean"),
                                c_tap.get("abs_mean"),
                            ),
                            "first_k_width": int(
                                t_tap.get("first_k_width", 0)
                            ),
                        }
                    )
                if hf_entry is not None:
                    h_cpp = hf_entry.get("ctx_per_position") or {}
                    # HF indexes by the iter's own context window; map by
                    # absolute position when possible via the same first-
                    # occurrence convention.  If HF ``first_k`` length
                    # mismatches the vLLM dim, we skip HF cosine.
                    h_stats = h_cpp.get(tap) or {}
                    h_first = h_stats.get("first_k") or []
                    # HF per-step stores one row per absolute position
                    # in its context window; we pick index = abs_pos - (
                    # start_of_hf_window).  Without the HF window start we
                    # probe the last index that matches the target vector
                    # length instead (HF probe emits the full window
                    # contiguously).
                    if h_first and t_tap is not None:
                        hf_last = h_first[-1] if h_first else None
                        if (
                            isinstance(hf_last, list)
                            and len(hf_last) == len(t_tap["first_k"])
                            and len(h_first) > (abs_pos - 0)
                        ):
                            # Best-effort absolute indexing: HF's list is
                            # 0..context_len-1 aligned with decoded
                            # positions (0 == first prompt token).
                            try:
                                hf_vec = h_first[int(abs_pos)]
                                tap_row["hf_vs_tree_cosine"] = float(
                                    _cosine(
                                        list(hf_vec),
                                        list(t_tap["first_k"]),
                                    )
                                )
                                tap_row["hf_vs_chain_cosine"] = float(
                                    _cosine(
                                        list(hf_vec),
                                        list(c_tap["first_k"]),
                                    )
                                )
                            except Exception:
                                pass
                per_tap[tap] = tap_row
                if cos_val is not None:
                    row_min_cos = (
                        cos_val
                        if row_min_cos is None
                        else min(row_min_cos, cos_val)
                    )
            row["per_tap"] = per_tap
            row["min_tree_vs_chain_cosine"] = row_min_cos

            if row_min_cos is not None:
                if row["is_accepted_prefix"]:
                    all_tvc_acc.append(row_min_cos)
                    pair_min_cos_acc = (
                        row_min_cos
                        if pair_min_cos_acc is None
                        else min(pair_min_cos_acc, row_min_cos)
                    )
                    if (
                        row_min_cos < cosine_threshold
                        and first_divergent_accepted_prefix_pos is None
                    ):
                        first_divergent_accepted_prefix_pos = int(abs_pos)
                else:
                    all_tvc_tail.append(row_min_cos)
                    pair_min_cos_tail = (
                        row_min_cos
                        if pair_min_cos_tail is None
                        else min(pair_min_cos_tail, row_min_cos)
                    )
                    if (
                        row_min_cos < cosine_threshold
                        and first_divergent_tail_pos is None
                    ):
                        first_divergent_tail_pos = int(abs_pos)
            shared_rows.append(row)

        per_pair.append(
            {
                "decoded_position": int(pos),
                "tree_iter": int(t["iter"]),
                "chain_iter": int(c["iter"]),
                "tree_context_len": int(t_cap.get("context_len", 0)),
                "chain_context_len": int(c_cap.get("context_len", 0)),
                "tree_unique_context_positions": t_unique,
                "chain_unique_context_positions": c_unique,
                "shared_unique_positions": shared,
                "num_shared_positions": len(shared),
                "num_accepted_prefix_shared": sum(
                    1
                    for r in shared_rows
                    if r.get("is_accepted_prefix")
                ),
                "num_tail_shared": sum(
                    1
                    for r in shared_rows
                    if not r.get("is_accepted_prefix")
                ),
                "min_cos_accepted_prefix": pair_min_cos_acc,
                "min_cos_tail": pair_min_cos_tail,
                "first_divergent_accepted_prefix_pos": (
                    first_divergent_accepted_prefix_pos
                ),
                "first_divergent_tail_pos": first_divergent_tail_pos,
                "per_shared_position": shared_rows,
            }
        )

    def _stats(xs: list[float]) -> dict[str, Any]:
        if not xs:
            return {"count": 0}
        return {
            "count": int(len(xs)),
            "min": float(min(xs)),
            "max": float(max(xs)),
            "mean": float(sum(xs) / len(xs)),
        }

    num_pairs_with_acc_div = sum(
        1
        for e in per_pair
        if e.get("first_divergent_accepted_prefix_pos") is not None
    )
    num_pairs_with_tail_div = sum(
        1
        for e in per_pair
        if e.get("first_divergent_tail_pos") is not None
    )
    num_pairs_total = len(per_pair)

    aggregates = {
        "num_matched_pairs": num_pairs_total,
        "num_pairs_with_accepted_prefix_divergence": (
            num_pairs_with_acc_div
        ),
        "num_pairs_with_tail_divergence": num_pairs_with_tail_div,
        "accepted_prefix_tree_vs_chain_cosine": _stats(all_tvc_acc),
        "tail_tree_vs_chain_cosine": _stats(all_tvc_tail),
        "cosine_threshold": float(cosine_threshold),
    }

    # Verdict scoring
    #   H1  (stale content at accepted-prefix paged slots):
    #       num_pairs_with_acc_div > 0  =>  suspected
    #       plus accepted_prefix min cos << 1
    #   H2  (wrong tail-window content from precompute):
    #       num_pairs_with_acc_div == 0 AND num_pairs_with_tail_div > 0
    #       => suspected (tree writes correct accepted-prefix but bad tail)
    #   H3  (write-path): identical signature to H1 if the bug lives in
    #       the accepted-path propagation (stale content survives into
    #       the next iter).  Reported alongside H1 (cannot be disentangled
    #       from content alone; H3 specifically would also show slot-
    #       mapping drift, which Test R has already refuted).
    h_scores: dict[str, Any] = {
        "H1_stale_accepted_prefix_content": {},
        "H2_tail_precompute_content": {},
        "H3_write_path_propagation": {},
    }
    reasons: list[str] = []

    if num_pairs_total == 0:
        verdict_label = "inconclusive_no_matched_pairs"
    elif num_pairs_with_acc_div > 0:
        verdict_label = "H1_suspected"
        reasons.append(
            f"{num_pairs_with_acc_div}/{num_pairs_total} matched pairs "
            f"show tree-vs-chain first_k cosine < {cosine_threshold} at "
            f"an accepted-prefix shared position (abs_pos < num_prompt + "
            f"decoded_position).  Tree and chain write to the *same* "
            f"slots for these positions (Test R), so divergent content "
            f"means tree's paged cache holds leftover values from "
            f"rejected speculative branches of prior iters."
        )
        first_ex = next(
            (
                e
                for e in per_pair
                if e.get("first_divergent_accepted_prefix_pos") is not None
            ),
            None,
        )
        if first_ex is not None:
            reasons.append(
                "first divergent accepted-prefix position = "
                f"abs_pos={first_ex['first_divergent_accepted_prefix_pos']} "
                f"(at matched decoded_position={first_ex['decoded_position']}, "
                f"tree_iter={first_ex['tree_iter']})"
            )
    elif num_pairs_with_tail_div > 0:
        verdict_label = "H2_suspected_tail_only"
        reasons.append(
            "All matched pairs agree on accepted-prefix content, but tree "
            "and chain disagree at tail (future-position) entries.  Since "
            "tail entries are newly computed each iter, this points at the "
            "tree-specific precompute write-back window rather than stale "
            "accepted-prefix content."
        )
    else:
        verdict_label = "H1_H2_both_refuted"
        reasons.append(
            "Every shared (tree, chain) position across all matched pairs "
            "has tree-vs-chain first_k cosine >= threshold on all three "
            "taps.  The divergence observed in Test P must come from a "
            "different stage (e.g., self-attn KV lifecycle, kernel, or "
            "target-hidden propagation) and not from paged-context K/V "
            "content mismatch."
        )

    h_scores["H1_stale_accepted_prefix_content"] = {
        "suspected": verdict_label == "H1_suspected",
        "num_divergent_pairs": num_pairs_with_acc_div,
        "accepted_prefix_cosine_stats": _stats(all_tvc_acc),
    }
    h_scores["H2_tail_precompute_content"] = {
        "suspected": verdict_label in ("H2_suspected_tail_only",),
        "num_divergent_pairs": num_pairs_with_tail_div,
        "tail_cosine_stats": _stats(all_tvc_tail),
    }
    h_scores["H3_write_path_propagation"] = {
        "suspected_alongside_H1": verdict_label == "H1_suspected",
        "note": (
            "H3 has the same content signature as H1.  Test R already "
            "refuted the H3 slot-mapping drift hypothesis; any H1-style "
            "signal here is thus attributable to content (paged-slot "
            "leakage from rejected branches), not to write-path "
            "misrouting."
        ),
    }

    verdict = {
        "label": verdict_label,
        "hypothesis_scores": h_scores,
        "reasons": reasons,
        "cosine_threshold": float(cosine_threshold),
    }

    return {
        "sample_index": int(sample_index),
        "num_prompt_tokens": {
            "tree": num_prompt_tree,
            "chain": num_prompt_chain,
        },
        "tap_keys": list(TAP_KEYS),
        "per_matched_pair": per_pair,
        "aggregates": aggregates,
        "verdict": verdict,
    }


def build_test_t_tail_layout_report(
    vllm_tree_capture: dict[str, Any],
    vllm_chain_capture: dict[str, Any],
    vllm_tree_summary: dict[str, Any],
    vllm_chain_summary: dict[str, Any],
    sample_index: int = 0,
    block_size: int = 16,
    cosine_threshold: float = 0.99,
) -> dict[str, Any]:
    """Test T: offline tail-only layout audit at matched decoded positions.

    This reuses the Test-L captures to answer a narrower question than Test R/S:
    when the remaining tree-vs-chain gap is confined to tail/future positions,
    is the tree tail *layout* already correct and the residual problem therefore
    content-only (H2), or do tail positions / slots themselves look malformed
    (H5)?

    The report inspects, for each tree/chain matched decoded position:
      * shared tail positions (present on both sides and >= current root)
      * tree-only tail positions (present only on tree side)
      * per-position slot equality for shared tail rows
      * position / slot modulo consistency (slot % block_size == pos % block_size)
      * whether tree-only positions are exactly the contiguous extension beyond
        the chain tail
      * whether tail content divergence (via Test-S-style first_k cosine) can be
        explained without any layout anomaly.
    """

    TAP_KEYS = ("k_ctx_pre_rope", "k_ctx_post_rope", "v_ctx")

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> tuple[int, list[dict[str, Any]]]:
        for s in summary.get("samples") or []:
            if int(s.get("sample_index", -1)) != int(sample_idx):
                continue
            try:
                num_prompt = int(s.get("num_prompt_tokens") or 0)
            except (TypeError, ValueError):
                num_prompt = 0
            steps_sorted = sorted(
                s.get("steps") or [], key=lambda e: int(e.get("step", 0))
            )
            cumulative = 0
            out: list[dict[str, Any]] = []
            for e in steps_sorted:
                enriched = dict(e)
                enriched["__position"] = cumulative
                out.append(enriched)
                try:
                    cumulative += int(e.get("accepted_len", 0)) + 1
                except (TypeError, ValueError):
                    cumulative += 1
            return num_prompt, out
        return 0, []

    def _capture_by_step(cap: dict[str, Any]) -> dict[int, dict[str, Any]]:
        by_step: dict[int, dict[str, Any]] = {}
        for entry in cap.get("per_step") or []:
            try:
                s_idx = int(entry.get("step"))
            except Exception:
                continue
            by_step[s_idx] = {
                "context_len": int(entry.get("context_len", 0)),
                "context_positions": list(entry.get("context_positions") or []),
                "context_slot_mapping": list(
                    entry.get("context_slot_mapping") or []
                ),
                "ctx_per_position": entry.get("ctx_per_position") or {},
            }
        return by_step

    def _infer_num_prompt_from_step0(caps: dict[int, dict[str, Any]]) -> int:
        step0 = caps.get(0)
        if not step0:
            return 0
        positions = step0.get("context_positions") or []
        if not positions:
            return 0
        try:
            return int(max(int(p) for p in positions)) + 1
        except (TypeError, ValueError):
            return 0

    def _first_index(positions: list[int], abs_pos: int) -> int | None:
        try:
            return positions.index(int(abs_pos))
        except ValueError:
            return None

    def _slot_at(
        positions: list[int], slots: list[int], abs_pos: int
    ) -> tuple[int | None, int | None]:
        idx = _first_index(positions, abs_pos)
        if idx is None or idx >= len(slots):
            return idx, None
        try:
            return idx, int(slots[idx])
        except (TypeError, ValueError):
            return idx, None

    def _get_tap_entry(
        ctx_per_pos: dict[str, Any],
        tap: str,
        idx: int | None,
    ) -> dict[str, Any] | None:
        stats = ctx_per_pos.get(tap) if isinstance(ctx_per_pos, dict) else None
        if not isinstance(stats, dict) or idx is None or idx < 0:
            return None
        first_k = stats.get("first_k") or []
        norms = stats.get("norm") or []
        if idx >= len(first_k):
            return None
        return {
            "first_k": first_k[idx],
            "norm": norms[idx] if idx < len(norms) else None,
            "first_k_width": int(stats.get("first_k_width", 0)),
        }

    def _shared_tail_min_cos(
        t_cpp: dict[str, Any],
        c_cpp: dict[str, Any],
        t_positions: list[int],
        c_positions: list[int],
        tail_positions: list[int],
    ) -> tuple[float | None, int | None]:
        min_cos: float | None = None
        min_pos: int | None = None
        for abs_pos in tail_positions:
            i_t = _first_index(t_positions, abs_pos)
            i_c = _first_index(c_positions, abs_pos)
            if i_t is None or i_c is None:
                continue
            row_min: float | None = None
            for tap in TAP_KEYS:
                t_tap = _get_tap_entry(t_cpp, tap, i_t)
                c_tap = _get_tap_entry(c_cpp, tap, i_c)
                if t_tap is None or c_tap is None:
                    continue
                cos_val = _cosine(
                    list(t_tap["first_k"]),
                    list(c_tap["first_k"]),
                )
                row_min = cos_val if row_min is None else min(row_min, cos_val)
            if row_min is None:
                continue
            if min_cos is None or row_min < min_cos:
                min_cos = float(row_min)
                min_pos = int(abs_pos)
        return min_cos, min_pos

    def _slot_mod_ok(abs_pos: int, slot: int | None) -> bool | None:
        if slot is None or slot < 0 or block_size <= 0:
            return None
        return int(slot) % int(block_size) == int(abs_pos) % int(block_size)

    num_prompt_tree, tree_iters = _per_iter_positions(
        vllm_tree_summary, sample_index
    )
    num_prompt_chain, chain_iters = _per_iter_positions(
        vllm_chain_summary, sample_index
    )
    tree_caps = _capture_by_step(vllm_tree_capture)
    chain_caps = _capture_by_step(vllm_chain_capture)

    if num_prompt_tree <= 0:
        num_prompt_tree = _infer_num_prompt_from_step0(tree_caps)
    if num_prompt_chain <= 0:
        num_prompt_chain = _infer_num_prompt_from_step0(chain_caps)

    tree_captured_by_position: dict[int, dict[str, Any]] = {}
    for it in tree_iters:
        si = int(it.get("step", -1))
        if si in tree_caps:
            tree_captured_by_position[int(it["__position"])] = {
                "iter": si,
                "iter_entry": it,
                "capture": tree_caps[si],
            }

    chain_captured_by_position: dict[int, dict[str, Any]] = {}
    for it in chain_iters:
        si = int(it.get("step", -1))
        if si in chain_caps:
            chain_captured_by_position[int(it["__position"])] = {
                "iter": si,
                "iter_entry": it,
                "capture": chain_caps[si],
            }

    matched_positions = sorted(
        set(tree_captured_by_position) & set(chain_captured_by_position)
    )

    per_pair: list[dict[str, Any]] = []
    num_pairs_with_tail_layout_issue = 0
    num_pairs_with_tail_content_only_issue = 0

    for pos in matched_positions:
        t = tree_captured_by_position[pos]
        c = chain_captured_by_position[pos]
        t_cap = t["capture"]
        c_cap = c["capture"]

        t_positions = list(t_cap.get("context_positions") or [])
        c_positions = list(c_cap.get("context_positions") or [])
        t_slots = list(t_cap.get("context_slot_mapping") or [])
        c_slots = list(c_cap.get("context_slot_mapping") or [])
        t_cpp = t_cap.get("ctx_per_position") or {}
        c_cpp = c_cap.get("ctx_per_position") or {}

        t_unique = sorted(set(int(x) for x in t_positions))
        c_unique = sorted(set(int(x) for x in c_positions))
        shared = sorted(set(t_unique) & set(c_unique))
        if not shared:
            continue

        acc_prefix_cutoff = num_prompt_tree + int(pos)
        shared_tail_positions = [
            int(p) for p in shared if int(p) >= int(acc_prefix_cutoff)
        ]
        tree_only_tail_positions = [
            int(p)
            for p in t_unique
            if int(p) >= int(acc_prefix_cutoff) and int(p) not in set(c_unique)
        ]

        chain_tail_max = max(shared_tail_positions) if shared_tail_positions else max(
            shared
        )
        expected_tree_only_tail_positions = list(
            range(int(chain_tail_max) + 1, int(max(t_unique)) + 1)
        )

        shared_tail_rows: list[dict[str, Any]] = []
        shared_tail_slot_mismatch_count = 0
        shared_tail_slot_mod_mismatch_count = 0
        for abs_pos in shared_tail_positions:
            t_idx, t_slot = _slot_at(t_positions, t_slots, abs_pos)
            c_idx, c_slot = _slot_at(c_positions, c_slots, abs_pos)
            tree_slot_matches_chain = (
                None if t_slot is None or c_slot is None else int(t_slot) == int(c_slot)
            )
            tree_slot_mod_ok = _slot_mod_ok(abs_pos, t_slot)
            chain_slot_mod_ok = _slot_mod_ok(abs_pos, c_slot)
            if tree_slot_matches_chain is False:
                shared_tail_slot_mismatch_count += 1
            if tree_slot_mod_ok is False or chain_slot_mod_ok is False:
                shared_tail_slot_mod_mismatch_count += 1
            shared_tail_rows.append(
                {
                    "abs_pos": int(abs_pos),
                    "tree_idx": t_idx,
                    "chain_idx": c_idx,
                    "tree_slot": t_slot,
                    "chain_slot": c_slot,
                    "tree_slot_matches_chain": tree_slot_matches_chain,
                    "tree_slot_mod_matches_position_mod": tree_slot_mod_ok,
                    "chain_slot_mod_matches_position_mod": chain_slot_mod_ok,
                }
            )

        tree_only_tail_rows: list[dict[str, Any]] = []
        tree_only_tail_slot_mod_mismatch_count = 0
        for abs_pos in tree_only_tail_positions:
            t_idx, t_slot = _slot_at(t_positions, t_slots, abs_pos)
            slot_mod_ok = _slot_mod_ok(abs_pos, t_slot)
            if slot_mod_ok is False:
                tree_only_tail_slot_mod_mismatch_count += 1
            tree_only_tail_rows.append(
                {
                    "abs_pos": int(abs_pos),
                    "tree_idx": t_idx,
                    "tree_slot": t_slot,
                    "tree_slot_mod_matches_position_mod": slot_mod_ok,
                }
            )

        min_shared_tail_cos, first_divergent_shared_tail_pos = _shared_tail_min_cos(
            t_cpp,
            c_cpp,
            t_positions,
            c_positions,
            shared_tail_positions,
        )

        tree_only_tail_positions_match_expected = (
            tree_only_tail_positions == expected_tree_only_tail_positions
        )
        tail_layout_issue = any(
            [
                shared_tail_slot_mismatch_count > 0,
                shared_tail_slot_mod_mismatch_count > 0,
                tree_only_tail_slot_mod_mismatch_count > 0,
                not tree_only_tail_positions_match_expected,
            ]
        )
        tail_content_only_issue = (
            not tail_layout_issue
            and min_shared_tail_cos is not None
            and float(min_shared_tail_cos) < float(cosine_threshold)
        )
        if tail_layout_issue:
            num_pairs_with_tail_layout_issue += 1
        if tail_content_only_issue:
            num_pairs_with_tail_content_only_issue += 1

        per_pair.append(
            {
                "decoded_position": int(pos),
                "tree_iter": int(t["iter"]),
                "chain_iter": int(c["iter"]),
                "accepted_prefix_cutoff_abs_pos": int(acc_prefix_cutoff),
                "shared_tail_positions": shared_tail_positions,
                "tree_only_tail_positions": tree_only_tail_positions,
                "expected_tree_only_tail_positions": (
                    expected_tree_only_tail_positions
                ),
                "tree_only_tail_positions_match_expected": (
                    tree_only_tail_positions_match_expected
                ),
                "shared_tail_slot_mismatch_count": (
                    shared_tail_slot_mismatch_count
                ),
                "shared_tail_slot_mod_mismatch_count": (
                    shared_tail_slot_mod_mismatch_count
                ),
                "tree_only_tail_slot_mod_mismatch_count": (
                    tree_only_tail_slot_mod_mismatch_count
                ),
                "min_shared_tail_cosine": min_shared_tail_cos,
                "first_divergent_shared_tail_pos": (
                    first_divergent_shared_tail_pos
                ),
                "tail_layout_issue": bool(tail_layout_issue),
                "tail_content_only_issue": bool(tail_content_only_issue),
                "per_shared_tail_position": shared_tail_rows,
                "per_tree_only_tail_position": tree_only_tail_rows,
            }
        )

    reasons: list[str] = []
    if not per_pair:
        label = "inconclusive_no_matched_pairs"
    elif num_pairs_with_tail_layout_issue > 0:
        label = "H5_suspected_tail_layout"
        reasons.append(
            f"{num_pairs_with_tail_layout_issue}/{len(per_pair)} matched pairs "
            "show a tail layout anomaly (shared-tail slot mismatch, slot-modulo "
            "mismatch, or non-contiguous tree-only tail positions)."
        )
    elif num_pairs_with_tail_content_only_issue > 0:
        label = "H2_suspected_tail_content_only"
        reasons.append(
            f"{num_pairs_with_tail_content_only_issue}/{len(per_pair)} matched "
            "pairs have tail content cosine below threshold while all tested "
            "tail layout checks pass. This points at a tree-specific tail "
            "precompute content/write-back issue rather than slot layout."
        )
    else:
        label = "tail_layout_and_content_refuted"
        reasons.append(
            "No matched pair shows a tail layout anomaly, and no shared-tail "
            "content cosine falls below threshold."
        )

    verdict = {
        "label": label,
        "hypothesis_scores": {
            "H2_tail_precompute_content": {
                "suspected": label == "H2_suspected_tail_content_only",
                "num_pairs": int(num_pairs_with_tail_content_only_issue),
            },
            "H5_slot_mapping_offbydepth": {
                "suspected": label == "H5_suspected_tail_layout",
                "num_pairs": int(num_pairs_with_tail_layout_issue),
            },
        },
        "reasons": reasons,
        "cosine_threshold": float(cosine_threshold),
        "block_size": int(block_size),
    }

    return {
        "sample_index": int(sample_index),
        "num_prompt_tokens": {
            "tree": num_prompt_tree,
            "chain": num_prompt_chain,
        },
        "aggregates": {
            "num_matched_pairs": int(len(per_pair)),
            "num_pairs_with_tail_layout_issue": int(
                num_pairs_with_tail_layout_issue
            ),
            "num_pairs_with_tail_content_only_issue": int(
                num_pairs_with_tail_content_only_issue
            ),
        },
        "per_matched_pair": per_pair,
        "verdict": verdict,
    }


def build_test_u_valid_tail_content_report(
    vllm_tree_capture: dict[str, Any],
    vllm_chain_capture: dict[str, Any],
    vllm_tree_summary: dict[str, Any],
    vllm_chain_summary: dict[str, Any],
    sample_index: int = 0,
    cosine_threshold: float = 0.99,
) -> dict[str, Any]:
    """Test U: valid-tail-only tree-vs-chain content audit.

    Reuses the Test-L captures, but unlike Test T it only compares tail rows
    that will actually be written into paged KV (slot_mapping != -1 on both
    sides).  This answers whether the residual tree-vs-chain gap can be pinned
    to *written* tail K/V content, or whether the current tail-only mismatch is
    confined to masked rows and therefore not yet causal.
    """

    TAP_KEYS = ("k_ctx_pre_rope", "k_ctx_post_rope", "v_ctx")

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> tuple[int, list[dict[str, Any]]]:
        for s in summary.get("samples") or []:
            if int(s.get("sample_index", -1)) != int(sample_idx):
                continue
            try:
                num_prompt = int(s.get("num_prompt_tokens") or 0)
            except (TypeError, ValueError):
                num_prompt = 0
            steps_sorted = sorted(
                s.get("steps") or [], key=lambda e: int(e.get("step", 0))
            )
            cumulative = 0
            out: list[dict[str, Any]] = []
            for e in steps_sorted:
                enriched = dict(e)
                enriched["__position"] = cumulative
                out.append(enriched)
                try:
                    cumulative += int(e.get("accepted_len", 0)) + 1
                except (TypeError, ValueError):
                    cumulative += 1
            return num_prompt, out
        return 0, []

    def _capture_by_step(cap: dict[str, Any]) -> dict[int, dict[str, Any]]:
        by_step: dict[int, dict[str, Any]] = {}
        for entry in cap.get("per_step") or []:
            try:
                s_idx = int(entry.get("step"))
            except Exception:
                continue
            by_step[s_idx] = {
                "context_len": int(entry.get("context_len", 0)),
                "context_positions": list(entry.get("context_positions") or []),
                "context_slot_mapping": list(
                    entry.get("context_slot_mapping") or []
                ),
                "ctx_per_position": entry.get("ctx_per_position") or {},
            }
        return by_step

    def _infer_num_prompt_from_step0(caps: dict[int, dict[str, Any]]) -> int:
        step0 = caps.get(0)
        if not step0:
            return 0
        positions = step0.get("context_positions") or []
        if not positions:
            return 0
        try:
            return int(max(int(p) for p in positions)) + 1
        except (TypeError, ValueError):
            return 0

    def _first_valid_index(
        positions: list[int], slots: list[int], abs_pos: int
    ) -> tuple[int | None, int | None]:
        for idx, pos in enumerate(positions):
            try:
                if int(pos) != int(abs_pos):
                    continue
            except (TypeError, ValueError):
                continue
            if idx >= len(slots):
                continue
            try:
                slot = int(slots[idx])
            except (TypeError, ValueError):
                continue
            if slot >= 0:
                return idx, slot
        return None, None

    def _get_tap_entry(
        ctx_per_pos: dict[str, Any],
        tap: str,
        idx: int | None,
    ) -> dict[str, Any] | None:
        stats = ctx_per_pos.get(tap) if isinstance(ctx_per_pos, dict) else None
        if not isinstance(stats, dict) or idx is None or idx < 0:
            return None
        first_k = stats.get("first_k") or []
        norms = stats.get("norm") or []
        abs_means = stats.get("abs_mean") or []
        if idx >= len(first_k):
            return None
        return {
            "first_k": first_k[idx],
            "norm": norms[idx] if idx < len(norms) else None,
            "abs_mean": abs_means[idx] if idx < len(abs_means) else None,
            "first_k_width": int(stats.get("first_k_width", 0)),
        }

    num_prompt_tree, tree_iters = _per_iter_positions(
        vllm_tree_summary, sample_index
    )
    num_prompt_chain, chain_iters = _per_iter_positions(
        vllm_chain_summary, sample_index
    )
    tree_caps = _capture_by_step(vllm_tree_capture)
    chain_caps = _capture_by_step(vllm_chain_capture)

    if num_prompt_tree <= 0:
        num_prompt_tree = _infer_num_prompt_from_step0(tree_caps)
    if num_prompt_chain <= 0:
        num_prompt_chain = _infer_num_prompt_from_step0(chain_caps)

    tree_captured_by_position: dict[int, dict[str, Any]] = {}
    for it in tree_iters:
        si = int(it.get("step", -1))
        if si in tree_caps:
            tree_captured_by_position[int(it["__position"])] = {
                "iter": si,
                "iter_entry": it,
                "capture": tree_caps[si],
            }

    chain_captured_by_position: dict[int, dict[str, Any]] = {}
    for it in chain_iters:
        si = int(it.get("step", -1))
        if si in chain_caps:
            chain_captured_by_position[int(it["__position"])] = {
                "iter": si,
                "iter_entry": it,
                "capture": chain_caps[si],
            }

    matched_positions = sorted(
        set(tree_captured_by_position) & set(chain_captured_by_position)
    )

    per_pair: list[dict[str, Any]] = []
    all_valid_tail_cos: list[float] = []
    num_pairs_with_valid_tail_divergence = 0
    num_pairs_with_shared_valid_tail_rows = 0
    num_pairs_without_shared_valid_tail_rows = 0

    for pos in matched_positions:
        t = tree_captured_by_position[pos]
        c = chain_captured_by_position[pos]
        t_cap = t["capture"]
        c_cap = c["capture"]
        t_positions = list(t_cap.get("context_positions") or [])
        c_positions = list(c_cap.get("context_positions") or [])
        t_slots = list(t_cap.get("context_slot_mapping") or [])
        c_slots = list(c_cap.get("context_slot_mapping") or [])
        t_cpp = t_cap.get("ctx_per_position") or {}
        c_cpp = c_cap.get("ctx_per_position") or {}

        acc_prefix_cutoff = num_prompt_tree + int(pos)
        shared_tail_candidates = sorted(
            set(int(x) for x in t_positions if int(x) >= acc_prefix_cutoff)
            & set(int(x) for x in c_positions if int(x) >= acc_prefix_cutoff)
        )

        per_shared_valid_tail: list[dict[str, Any]] = []
        pair_min_cos: float | None = None
        pair_first_divergent_abs_pos: int | None = None
        for abs_pos in shared_tail_candidates:
            t_idx, t_slot = _first_valid_index(t_positions, t_slots, abs_pos)
            c_idx, c_slot = _first_valid_index(c_positions, c_slots, abs_pos)
            if t_idx is None or c_idx is None:
                continue
            row: dict[str, Any] = {
                "abs_pos": int(abs_pos),
                "tree_idx": int(t_idx),
                "chain_idx": int(c_idx),
                "tree_slot": int(t_slot) if t_slot is not None else None,
                "chain_slot": int(c_slot) if c_slot is not None else None,
                "tree_slot_matches_chain": (
                    int(t_slot) == int(c_slot)
                    if t_slot is not None and c_slot is not None
                    else None
                ),
            }
            row_min: float | None = None
            per_tap: dict[str, Any] = {}
            for tap in TAP_KEYS:
                t_tap = _get_tap_entry(t_cpp, tap, t_idx)
                c_tap = _get_tap_entry(c_cpp, tap, c_idx)
                tap_row: dict[str, Any] = {}
                if t_tap is not None and c_tap is not None:
                    cos_val = _cosine(
                        list(t_tap["first_k"]),
                        list(c_tap["first_k"]),
                    )
                    tap_row["tree_vs_chain_cosine"] = float(cos_val)
                    row_min = cos_val if row_min is None else min(row_min, cos_val)
                per_tap[tap] = tap_row
            row["per_tap"] = per_tap
            row["min_tree_vs_chain_cosine"] = row_min
            per_shared_valid_tail.append(row)
            if row_min is not None:
                all_valid_tail_cos.append(float(row_min))
                pair_min_cos = row_min if pair_min_cos is None else min(
                    pair_min_cos, row_min
                )
                if (
                    row_min < cosine_threshold
                    and pair_first_divergent_abs_pos is None
                ):
                    pair_first_divergent_abs_pos = int(abs_pos)

        if per_shared_valid_tail:
            num_pairs_with_shared_valid_tail_rows += 1
        else:
            num_pairs_without_shared_valid_tail_rows += 1

        pair_has_divergence = pair_first_divergent_abs_pos is not None
        if pair_has_divergence:
            num_pairs_with_valid_tail_divergence += 1

        per_pair.append(
            {
                "decoded_position": int(pos),
                "tree_iter": int(t["iter"]),
                "chain_iter": int(c["iter"]),
                "accepted_prefix_cutoff_abs_pos": int(acc_prefix_cutoff),
                "shared_tail_candidate_positions": shared_tail_candidates,
                "num_shared_valid_tail_rows": int(len(per_shared_valid_tail)),
                "min_shared_valid_tail_cosine": pair_min_cos,
                "first_divergent_shared_valid_tail_pos": (
                    pair_first_divergent_abs_pos
                ),
                "has_valid_tail_divergence": bool(pair_has_divergence),
                "per_shared_valid_tail_position": per_shared_valid_tail,
            }
        )

    def _stats(xs: list[float]) -> dict[str, Any]:
        if not xs:
            return {"count": 0}
        return {
            "count": int(len(xs)),
            "min": float(min(xs)),
            "max": float(max(xs)),
            "mean": float(sum(xs) / len(xs)),
        }

    reasons: list[str] = []
    if not per_pair:
        label = "inconclusive_no_matched_pairs"
    elif num_pairs_with_shared_valid_tail_rows == 0:
        label = "inconclusive_no_shared_valid_tail_rows"
        reasons.append(
            "No matched pair contains a shared tail position with slot != -1 on "
            "both tree and chain. The previously observed tail-only mismatch is "
            "confined to masked rows, so this audit cannot yet attribute the "
            "residual gap to written tail K/V."
        )
    elif num_pairs_with_valid_tail_divergence > 0:
        label = "written_tail_content_diverges"
        reasons.append(
            f"{num_pairs_with_valid_tail_divergence}/{len(per_pair)} matched "
            f"pairs show a valid written tail row with cosine < {cosine_threshold}. "
            "This is direct evidence that the residual tree-vs-chain gap reaches "
            "rows that are actually written into paged KV."
        )
    else:
        label = "written_tail_content_matches"
        reasons.append(
            "All shared valid tail rows have cosine above threshold. The "
            "currently observed tail mismatch is restricted to masked rows and "
            "is unlikely to be the direct cause of the residual gap."
        )

    verdict = {
        "label": label,
        "reasons": reasons,
        "cosine_threshold": float(cosine_threshold),
    }

    return {
        "sample_index": int(sample_index),
        "num_prompt_tokens": {
            "tree": num_prompt_tree,
            "chain": num_prompt_chain,
        },
        "aggregates": {
            "num_matched_pairs": int(len(per_pair)),
            "num_pairs_with_shared_valid_tail_rows": int(
                num_pairs_with_shared_valid_tail_rows
            ),
            "num_pairs_without_shared_valid_tail_rows": int(
                num_pairs_without_shared_valid_tail_rows
            ),
            "num_pairs_with_valid_tail_divergence": int(
                num_pairs_with_valid_tail_divergence
            ),
            "shared_valid_tail_cosine": _stats(all_valid_tail_cos),
        },
        "per_matched_pair": per_pair,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Test W: effective layer-0 hidden-stream parity (tree-vs-chain).
#
# Motivation (reconciling Test P vs Test Q).
#   Test Q reports ``layer0_output_hidden`` cos ~0.45 and
#   ``layer0_output_residual`` cos ~0.92 at the d0 row, while Test P reports
#   ``layer1 i1_q_post_qproj`` cos ~0.9999 at the same matched positions.
#   Layer 1 is a deterministic function of layer 0's output, so these two
#   signals cannot both be right -- UNLESS we are comparing only half of
#   layer 0's output stream.
#
#   ``DFlashQwen3DecoderLayer.forward`` returns ``(hidden_states, residual)``
#   where ``hidden_states`` is the post-MLP delta and ``residual`` is the
#   pre-MLP accumulated stream.  Layer K+1 consumes BOTH via the fused
#   add-norm: ``input_layernorm(hidden_states, residual)`` which is
#   effectively ``RMSNorm(hidden_states + residual)``.  So the *actual*
#   stream flowing into layer 1 is ``hidden + residual``, and the two
#   Test-Q cosines only tell us about each summand in isolation.
#
# Test W closes this gap by using the existing per-row Test-Q captures and
# computing ``effective = hidden + residual`` per-row before computing the
# tree-vs-chain cosine.  This:
#
#   * Tells us the TRUE layer-0 output parity (independent of how the
#     residual-stream accounting splits across the two tensors).
#   * If effective cos ~1.0 while hidden/residual cosines are split,
#     confirms the original "layer-0 output divergence" was a probe
#     accounting artifact and narrows the residual gap to somewhere OTHER
#     than the layer-0 output boundary.
#   * If effective cos <1 even after the sum, confirms a real layer-0
#     divergence and the cosine is the correct magnitude to reason with.
#
# Test W is purely offline: it reuses Test Q's per-row captures and the
# per-iter position reconstruction already used by Test P / Test Q.
# ---------------------------------------------------------------------------


def build_test_w_effective_hidden_stream_report(
    vllm_tree_q_capture: dict[str, Any],
    vllm_chain_q_capture: dict[str, Any],
    vllm_tree_summary: dict[str, Any],
    vllm_chain_summary: dict[str, Any],
    sample_index: int = 0,
    capture_row: int | None = None,
    row_label: str = "depth-0 row",
    cosine_threshold: float = 0.999,
) -> dict[str, Any]:
    """Test W -- effective (hidden+residual) layer-0 output parity.

    Consumes Test Q captures (``retrieve_vllm_test_q_probe`` output for tree
    and chain).  For each matched decoded position (same matching convention
    as Test P/Q), at the requested query row we reconstruct the effective
    layer-0 output stream ``hidden + residual`` for both tree and chain and
    compute their cosine.  We also recompute the effective *input* stream
    at the layer-0 boundary for sanity (it should be ~1.0 when Test Q's
    ``layer0_input_hidden`` was ~1.0).

    Parameters
    ----------
    vllm_tree_q_capture, vllm_chain_q_capture :
        ``retrieve_vllm_test_q_probe`` output from each run.
    vllm_tree_summary, vllm_chain_summary :
        Per-sample summaries (reused for decoded-position reconstruction).
    sample_index :
        Sample id used for position reconstruction.
    capture_row :
        Query row to analyze.  ``None`` falls back to the probe's
        ``d0_row`` (typically row 1).  When set to ``capture_row>=2``, the
        report returns the effective-stream cosine at that row if BOTH
        tree and chain captured it (tree always has branch rows; chain
        typically does not, in which case the row's report will show
        ``effective_output_cosine=None``).
    cosine_threshold :
        The first tap/pair with effective cosine below this flags the
        verdict.
    """

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> list[dict[str, Any]]:
        """Per-iter entries augmented with decoded-position (Test P rule)."""
        for s in summary.get("samples") or []:
            if int(s.get("sample_index", -1)) != int(sample_idx):
                continue
            steps_sorted = sorted(
                s.get("steps") or [], key=lambda e: int(e.get("step", 0))
            )
            cumulative = 0
            out: list[dict[str, Any]] = []
            for e in steps_sorted:
                enriched = dict(e)
                enriched["__position"] = cumulative
                out.append(enriched)
                try:
                    cumulative += int(e.get("accepted_len", 0)) + 1
                except (TypeError, ValueError):
                    cumulative += 1
            return out
        return []

    def _taps_for_row(
        entry: dict[str, Any], row: int | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (attn_taps, attn_stats) for the requested row.

        When ``row is None`` (or equals the probe's d0_row), the top-level
        ``attn_taps`` / ``attn_stats`` are returned (these are already the
        d0 row's view by construction).  Otherwise, we read from
        ``per_row_attn_taps[str(row)]``; missing rows return empty dicts.
        """
        if row is None:
            return (
                entry.get("attn_taps") or {},
                entry.get("attn_stats") or {},
            )
        per_row_taps = entry.get("per_row_attn_taps") or {}
        per_row_stats = entry.get("per_row_attn_stats") or {}
        row_key = str(int(row))
        return (
            per_row_taps.get(row_key) or {},
            per_row_stats.get(row_key) or {},
        )

    def _index_by_step(cap: dict[str, Any]) -> dict[int, dict[str, Any]]:
        by_step: dict[int, dict[str, Any]] = {}
        for entry in cap.get("per_step") or []:
            try:
                s_idx = int(entry.get("step"))
            except Exception:
                continue
            by_step[s_idx] = entry
        return by_step

    def _vec_add(a: Any, b: Any) -> list[float] | None:
        if not isinstance(a, list) or not isinstance(b, list):
            return None
        if not a or not b or len(a) != len(b):
            return None
        return [float(x) + float(y) for x, y in zip(a, b)]

    def _cos_of_vec(a: Any, b: Any) -> float | None:
        if not isinstance(a, list) or not isinstance(b, list):
            return None
        if not a or not b or len(a) != len(b):
            return None
        return float(_cosine(a, b))

    tree_iters = _per_iter_positions(vllm_tree_summary, sample_index)
    chain_iters = _per_iter_positions(vllm_chain_summary, sample_index)
    tree_by_step = _index_by_step(vllm_tree_q_capture)
    chain_by_step = _index_by_step(vllm_chain_q_capture)

    # Propagate the requested row.  ``None`` means "use probe's d0_row"
    # (the top-level taps).  The probe's d0_row is recorded on the capture;
    # fall back to 1 if unknown.
    tree_d0_row = int(vllm_tree_q_capture.get("d0_row") or 1)
    chain_d0_row = int(vllm_chain_q_capture.get("d0_row") or 1)
    effective_row: int | None = (
        int(capture_row) if capture_row is not None else None
    )

    tree_pos_to_entry: dict[int, dict[str, Any]] = {}
    for it in tree_iters:
        si = int(it.get("step", -1))
        if si in tree_by_step:
            tree_pos_to_entry[int(it["__position"])] = {
                "iter": si,
                "iter_entry": it,
                "q_entry": tree_by_step[si],
            }
    chain_pos_to_entry: dict[int, dict[str, Any]] = {}
    for it in chain_iters:
        si = int(it.get("step", -1))
        if si in chain_by_step:
            chain_pos_to_entry[int(it["__position"])] = {
                "iter": si,
                "iter_entry": it,
                "q_entry": chain_by_step[si],
            }

    matched_positions = sorted(
        set(tree_pos_to_entry) & set(chain_pos_to_entry)
    )

    per_pair: list[dict[str, Any]] = []
    effective_out_cos_vals: list[float] = []
    effective_in_cos_vals: list[float] = []
    raw_hidden_out_cos_vals: list[float] = []
    raw_residual_out_cos_vals: list[float] = []

    for pos in matched_positions:
        t = tree_pos_to_entry[pos]
        c = chain_pos_to_entry[pos]
        t_taps, t_stats = _taps_for_row(t["q_entry"], effective_row)
        c_taps, c_stats = _taps_for_row(c["q_entry"], effective_row)

        t_out_hidden = t_taps.get("layer0_output_hidden")
        t_out_res = t_taps.get("layer0_output_residual")
        c_out_hidden = c_taps.get("layer0_output_hidden")
        c_out_res = c_taps.get("layer0_output_residual")

        t_in_hidden = t_taps.get("layer0_input_hidden")
        t_in_res = t_taps.get("layer0_input_residual")
        c_in_hidden = c_taps.get("layer0_input_hidden")
        c_in_res = c_taps.get("layer0_input_residual")

        # Effective output stream = hidden + residual (layer 1's input to
        # input_layernorm).  If residual is None (layer 0 has no carry),
        # the effective stream is just ``hidden``.
        t_eff_out = (
            _vec_add(t_out_hidden, t_out_res)
            if isinstance(t_out_res, list)
            else (t_out_hidden if isinstance(t_out_hidden, list) else None)
        )
        c_eff_out = (
            _vec_add(c_out_hidden, c_out_res)
            if isinstance(c_out_res, list)
            else (c_out_hidden if isinstance(c_out_hidden, list) else None)
        )

        # Effective input stream = input_hidden + input_residual (what
        # layer 0 actually normalizes).  When input_residual is None, the
        # effective input equals ``input_hidden`` (first layer convention).
        t_eff_in = (
            _vec_add(t_in_hidden, t_in_res)
            if isinstance(t_in_res, list)
            else (t_in_hidden if isinstance(t_in_hidden, list) else None)
        )
        c_eff_in = (
            _vec_add(c_in_hidden, c_in_res)
            if isinstance(c_in_res, list)
            else (c_in_hidden if isinstance(c_in_hidden, list) else None)
        )

        eff_out_cos = _cos_of_vec(t_eff_out, c_eff_out)
        eff_in_cos = _cos_of_vec(t_eff_in, c_eff_in)
        raw_hidden_out_cos = _cos_of_vec(t_out_hidden, c_out_hidden)
        raw_residual_out_cos = _cos_of_vec(t_out_res, c_out_res)

        if eff_out_cos is not None:
            effective_out_cos_vals.append(eff_out_cos)
        if eff_in_cos is not None:
            effective_in_cos_vals.append(eff_in_cos)
        if raw_hidden_out_cos is not None:
            raw_hidden_out_cos_vals.append(raw_hidden_out_cos)
        if raw_residual_out_cos is not None:
            raw_residual_out_cos_vals.append(raw_residual_out_cos)

        pair_entry = {
            "position": int(pos),
            "tree_iter": int(t["iter"]),
            "chain_iter": int(c["iter"]),
            "effective_input_cosine": eff_in_cos,
            "effective_output_cosine": eff_out_cos,
            "raw_output_hidden_cosine": raw_hidden_out_cos,
            "raw_output_residual_cosine": raw_residual_out_cos,
            "tree_shapes": {
                "layer0_output_hidden": (t_stats.get("layer0_output_hidden")
                                         or {}).get("shape"),
                "layer0_output_residual": (t_stats.get(
                    "layer0_output_residual") or {}).get("shape"),
            },
            "chain_shapes": {
                "layer0_output_hidden": (c_stats.get("layer0_output_hidden")
                                         or {}).get("shape"),
                "layer0_output_residual": (c_stats.get(
                    "layer0_output_residual") or {}).get("shape"),
            },
        }
        per_pair.append(pair_entry)

    def _mean(xs: list[float]) -> float | None:
        return (sum(xs) / len(xs)) if xs else None

    def _min(xs: list[float]) -> float | None:
        return min(xs) if xs else None

    aggregates = {
        "num_matched_pairs": len(per_pair),
        "effective_output_cosine": {
            "num": len(effective_out_cos_vals),
            "mean": _mean(effective_out_cos_vals),
            "min": _min(effective_out_cos_vals),
        },
        "effective_input_cosine": {
            "num": len(effective_in_cos_vals),
            "mean": _mean(effective_in_cos_vals),
            "min": _min(effective_in_cos_vals),
        },
        "raw_output_hidden_cosine": {
            "num": len(raw_hidden_out_cos_vals),
            "mean": _mean(raw_hidden_out_cos_vals),
            "min": _min(raw_hidden_out_cos_vals),
        },
        "raw_output_residual_cosine": {
            "num": len(raw_residual_out_cos_vals),
            "mean": _mean(raw_residual_out_cos_vals),
            "min": _min(raw_residual_out_cos_vals),
        },
    }

    verdict: dict[str, Any] = {
        "sample_index": int(sample_index),
        "capture_row": effective_row,
        "row_label": row_label,
        "tree_probe_d0_row": tree_d0_row,
        "chain_probe_d0_row": chain_d0_row,
        "cosine_threshold": float(cosine_threshold),
        "num_matched_pairs": len(per_pair),
    }

    mean_eff_out = aggregates["effective_output_cosine"]["mean"]
    mean_eff_in = aggregates["effective_input_cosine"]["mean"]
    mean_raw_h = aggregates["raw_output_hidden_cosine"]["mean"]
    mean_raw_r = aggregates["raw_output_residual_cosine"]["mean"]

    if len(per_pair) == 0:
        verdict["interpretation_hint"] = (
            "No matched (same-decoded-position) pairs were produced; "
            "ensure Test Q probes are installed on both tree and chain "
            "runs and that capture_steps overlap across the two."
        )
    elif mean_eff_in is not None and mean_eff_in < cosine_threshold:
        verdict["interpretation_hint"] = (
            f"Effective layer-0 INPUT stream for {row_label} shows "
            f"cos={mean_eff_in:.4f} (<{cosine_threshold}).  Divergence "
            "predates layer 0 (embedding/positions/residual carry).  "
            "This REFUTES any localization inside layer 0."
        )
    elif mean_eff_out is not None and mean_eff_out < cosine_threshold:
        verdict["interpretation_hint"] = (
            f"Effective layer-0 OUTPUT stream for {row_label} shows "
            f"cos={mean_eff_out:.4f} (<{cosine_threshold}) while INPUT "
            f"cos={mean_eff_in}.  Layer 0 genuinely produces diverging "
            "output for tree vs chain -- the raw_output_hidden / "
            "raw_output_residual split was not just a probe accounting "
            "artifact.  Inspect layer-0 self_attn taps for kernel/K-V "
            "content origin."
        )
    else:
        verdict["interpretation_hint"] = (
            f"Effective layer-0 output stream for {row_label} matches at "
            f"mean cos={mean_eff_out:.4f} -- the earlier split between "
            f"raw hidden cos={mean_raw_h} and residual cos={mean_raw_r} "
            "was a probe-accounting artifact (Qwen3 returns "
            "(hidden,residual) separately; the next layer adds them).  "
            "Layer 0's output boundary is clean for this row; any "
            "residual tree-vs-chain gap must live at deeper layers, in "
            "the sampling head, or in depth>0 branch rows.  "
            "Cross-reference Test P (layer 1 taps) and Test X "
            "(branch-row health) to localize."
        )

    return {
        "cosine_threshold": float(cosine_threshold),
        "sample_index": int(sample_index),
        "capture_row": effective_row,
        "row_label": row_label,
        "matched_positions": matched_positions,
        "per_matched_pair": per_pair,
        "aggregates": aggregates,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Test X: tree-internal branch-row health audit.
#
# Test V (row-2-vs-row-2 tree-vs-chain) is structurally invalid: chain has
# only 2 query rows per req while tree has up to 16.  To probe depth>0
# branch behavior we need a tree-internal sanity check.
#
# Test X uses the per-row Test-Q captures on the TREE side only.  For each
# captured iteration and each captured branch row (rows >= 2), it records:
#
#   * per-row norm / abs_mean at each layer-0 tap (shape-meta only; the
#     probe itself does not store per-row norms to avoid CUDA syncs, so
#     we re-derive from the per-row vectors that ARE captured).
#   * row_k-vs-row_1 cosine at each tap (row 1 is the d0 row; branches
#     should be "similar context, different speculated token", so cos
#     should be high but <1).
#   * flags for pathological norms (NaN / >10x deviation from row-1 norm)
#     or near-orthogonal cosines that would indicate an attention-mask
#     bug or a corrupted branch KV lane.
#
# This gives us a tree-only signal we can correlate against the
# per-iteration top1 histogram to see whether branch pathology clusters
# on failing iterations.  It is entirely offline, reusing Test Q's
# existing ``per_row_attn_taps`` payload -- no new probes required.
# ---------------------------------------------------------------------------


def build_test_x_tree_branch_row_health_report(
    vllm_tree_q_capture: dict[str, Any],
    vllm_tree_summary: dict[str, Any],
    sample_index: int = 0,
    d0_row: int | None = None,
    branch_cos_low_threshold: float = 0.5,
    norm_ratio_high_threshold: float = 10.0,
) -> dict[str, Any]:
    """Test X -- tree-internal branch-row health (tree-only).

    Parameters
    ----------
    vllm_tree_q_capture : output of ``retrieve_vllm_test_q_probe`` for the
        tree-spec vLLM run (must have been installed with ``capture_rows``
        covering d0_row AND at least one branch row).
    vllm_tree_summary : per-sample summary, used only to annotate
        per-iter decoded-position and accepted_len.
    sample_index : sample id to inspect.
    d0_row : query row to use as the anchor ("row 1" -- depth-0 mask).
        ``None`` falls back to ``vllm_tree_q_capture['d0_row']`` or 1.
    branch_cos_low_threshold : per-pair row_k-vs-row_1 cosines at
        ``layer0_input_hidden`` below this flag the branch row as
        suspect (input plumbing or position-embedding should keep branch
        rows quite close to row 1 at layer-0 input).
    norm_ratio_high_threshold : any row_k norm that exceeds
        ``threshold x row_1_norm`` at any tap is flagged.

    Returns a JSON-serializable report.  Because the probe intentionally
    avoids per-element norm computation on device, we compute norm /
    abs_mean here from the captured per-row vectors.  This is tolerable
    because the vectors are small (hidden_size == 4096 floats) and we
    only process a handful of captured iterations.
    """
    import math  # noqa: PLC0415

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> list[dict[str, Any]]:
        for s in summary.get("samples") or []:
            if int(s.get("sample_index", -1)) != int(sample_idx):
                continue
            steps_sorted = sorted(
                s.get("steps") or [], key=lambda e: int(e.get("step", 0))
            )
            cumulative = 0
            out: list[dict[str, Any]] = []
            for e in steps_sorted:
                enriched = dict(e)
                enriched["__position"] = cumulative
                out.append(enriched)
                try:
                    cumulative += int(e.get("accepted_len", 0)) + 1
                except (TypeError, ValueError):
                    cumulative += 1
            return out
        return []

    def _vec_norm(v: Any) -> float | None:
        if not isinstance(v, list) or not v:
            return None
        s = 0.0
        for x in v:
            fx = float(x)
            s += fx * fx
        return math.sqrt(s)

    def _vec_abs_mean(v: Any) -> float | None:
        if not isinstance(v, list) or not v:
            return None
        s = 0.0
        for x in v:
            s += abs(float(x))
        return s / len(v)

    def _cos(a: Any, b: Any) -> float | None:
        if not isinstance(a, list) or not isinstance(b, list):
            return None
        if not a or not b or len(a) != len(b):
            return None
        return float(_cosine(a, b))

    anchor_row = (
        int(d0_row)
        if d0_row is not None
        else int(vllm_tree_q_capture.get("d0_row") or 1)
    )
    capture_rows: list[int] = [
        int(r) for r in (vllm_tree_q_capture.get("capture_rows") or [])
    ]
    branch_rows = sorted(
        r for r in capture_rows if int(r) != int(anchor_row)
    )

    tap_order = [
        "layer0_input_hidden",
        "layer0_input_residual",
        "layer0_self_attn_out",
        "layer0_output_hidden",
        "layer0_output_residual",
    ]

    per_iter_annot: dict[int, dict[str, Any]] = {}
    for it in _per_iter_positions(vllm_tree_summary, sample_index):
        per_iter_annot[int(it.get("step", -1))] = {
            "position": int(it.get("__position", 0)),
            "accepted_len": it.get("accepted_len"),
            "root_token": it.get("root_token"),
        }

    per_iter: list[dict[str, Any]] = []
    num_flagged_iters = 0
    num_flagged_branches = 0
    branch_row_cos_sums: dict[int, dict[str, list[float]]] = {}

    for entry in vllm_tree_q_capture.get("per_step") or []:
        try:
            step = int(entry.get("step", -1))
        except Exception:
            continue
        per_row_taps = entry.get("per_row_attn_taps") or {}
        anchor_taps = per_row_taps.get(str(int(anchor_row))) or {}

        row_stats_out: dict[str, dict[str, Any]] = {}
        anchor_norms: dict[str, float | None] = {}
        for tap in tap_order:
            vec = anchor_taps.get(tap)
            anchor_norms[tap] = _vec_norm(vec)
            row_stats_out.setdefault(str(int(anchor_row)), {})[tap] = {
                "norm": _vec_norm(vec),
                "abs_mean": _vec_abs_mean(vec),
                "cos_to_anchor": 1.0 if isinstance(vec, list) else None,
            }

        iter_flagged = False
        per_branch_summary: list[dict[str, Any]] = []
        for br in branch_rows:
            br_key = str(int(br))
            br_taps = per_row_taps.get(br_key) or {}
            branch_flags: list[str] = []
            per_tap: dict[str, Any] = {}
            for tap in tap_order:
                vec = br_taps.get(tap)
                anchor_vec = anchor_taps.get(tap)
                n = _vec_norm(vec)
                a = _vec_abs_mean(vec)
                c = _cos(vec, anchor_vec)
                per_tap[tap] = {
                    "norm": n,
                    "abs_mean": a,
                    "cos_to_anchor": c,
                }
                row_stats_out.setdefault(br_key, {})[tap] = per_tap[tap]
                # Flag pathological norms.
                if n is None or (isinstance(n, float) and math.isnan(n)):
                    if isinstance(vec, list) and vec:
                        branch_flags.append(
                            f"{tap}:nan_or_zero_norm"
                        )
                anchor_n = anchor_norms.get(tap)
                if (
                    n is not None
                    and anchor_n is not None
                    and anchor_n > 1e-6
                    and (n / anchor_n) > norm_ratio_high_threshold
                ):
                    branch_flags.append(
                        f"{tap}:norm_ratio={n / anchor_n:.2f}x"
                    )
                # Flag near-orthogonal input hidden -- branch rows share
                # the exact embedding row (mask token) so layer-0 input
                # should be ~1.0 vs anchor.  Anything lower is a
                # fundamental branch plumbing bug.
                if (
                    tap == "layer0_input_hidden"
                    and c is not None
                    and c < branch_cos_low_threshold
                ):
                    branch_flags.append(
                        f"{tap}:cos_to_anchor={c:.3f}<"
                        f"{branch_cos_low_threshold}"
                    )
                # Accumulate per-tap cosines across iterations for the
                # aggregate distribution.
                sums = branch_row_cos_sums.setdefault(int(br), {})
                sums.setdefault(tap, [])
                if c is not None:
                    sums[tap].append(c)

            per_branch_summary.append(
                {
                    "row": int(br),
                    "per_tap": per_tap,
                    "flags": branch_flags,
                }
            )
            if branch_flags:
                iter_flagged = True
                num_flagged_branches += 1

        annot = per_iter_annot.get(step) or {}
        iter_out = {
            "step": step,
            "position": annot.get("position"),
            "accepted_len": annot.get("accepted_len"),
            "root_token": annot.get("root_token"),
            "anchor_row": int(anchor_row),
            "branch_rows": branch_rows,
            "per_branch": per_branch_summary,
            "anchor_norms": anchor_norms,
            "iter_flagged": iter_flagged,
        }
        per_iter.append(iter_out)
        if iter_flagged:
            num_flagged_iters += 1

    def _stats(xs: list[float]) -> dict[str, Any]:
        if not xs:
            return {"num": 0, "mean": None, "min": None, "max": None}
        return {
            "num": len(xs),
            "mean": sum(xs) / len(xs),
            "min": min(xs),
            "max": max(xs),
        }

    aggregate: dict[str, Any] = {
        "num_captured_iters": len(per_iter),
        "num_flagged_iters": int(num_flagged_iters),
        "num_flagged_branches": int(num_flagged_branches),
        "branch_rows": branch_rows,
        "per_branch_cos_to_anchor": {
            str(int(br)): {
                tap: _stats(vals)
                for tap, vals in (branch_row_cos_sums.get(int(br)) or {}).items()
            }
            for br in branch_rows
        },
    }

    verdict: dict[str, Any] = {
        "sample_index": int(sample_index),
        "anchor_row": int(anchor_row),
        "branch_rows": branch_rows,
        "branch_cos_low_threshold": float(branch_cos_low_threshold),
        "norm_ratio_high_threshold": float(norm_ratio_high_threshold),
        "num_captured_iters": len(per_iter),
        "num_flagged_iters": int(num_flagged_iters),
        "num_flagged_branches": int(num_flagged_branches),
    }

    if not branch_rows:
        verdict["interpretation_hint"] = (
            "No branch rows were captured (Test-Q probe was installed "
            "with only the d0 row).  Re-run with "
            "``capture_rows=(d0_row, 2, ...)`` to enable Test X."
        )
    elif len(per_iter) == 0:
        verdict["interpretation_hint"] = (
            "Probe installed but no iterations were captured on the "
            "tree side.  Check steps_to_capture and that the probe is "
            "installed before generate()."
        )
    elif num_flagged_iters == 0:
        verdict["interpretation_hint"] = (
            "No pathological branch rows detected: branch rows share "
            "the mask-token embedding and show sane norms + high input "
            "cosine to the d0 row.  This REFUTES 'branch plumbing "
            "catastrophe' as a cause of the residual tree gap.  "
            "Residual gap must come from deeper layers or downstream "
            "sampling / KV-cache effects rather than branch-row input "
            "corruption."
        )
    else:
        verdict["interpretation_hint"] = (
            f"{num_flagged_iters}/{len(per_iter)} captured iterations "
            f"have at least one flagged branch row "
            f"({num_flagged_branches} branch-row flags total).  "
            "Inspect ``per_iter[i].per_branch[j].flags`` to see which "
            "tap tripped.  A ``layer0_input_hidden:cos_to_anchor<X`` "
            "flag indicates branch-row input plumbing (position "
            "encoding / depth alignment) is wrong; a ``*:norm_ratio`` "
            "flag indicates an attention-kernel / mask anomaly "
            "amplifying the branch state."
        )

    return {
        "sample_index": int(sample_index),
        "anchor_row": int(anchor_row),
        "branch_rows": branch_rows,
        "branch_cos_low_threshold": float(branch_cos_low_threshold),
        "norm_ratio_high_threshold": float(norm_ratio_high_threshold),
        "per_iter": per_iter,
        "aggregate": aggregate,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Test Y: tree-only branch-row input-origin audit.
#
# Goal: pinpoint WHETHER the branch-row anomaly seen in Test X is caused by
# wrong query token ids / wrong query positions, or whether the row is already
# carrying the wrong hidden state *despite* apparently sane metadata.
#
# DFlash first-pass semantics construct query tokens as:
#   [bonus_token, mask_token, mask_token, ...]
# and the draft model's layer-0 input is exactly ``embed_input_ids(input_ids)``
# (positions are only applied later inside attention via RoPE).  Therefore:
#
#   * If two query rows carry the SAME input_id, their ``layer0_input_hidden``
#     should be effectively identical.
#   * If a branch row has the same input_id as the anchor row but low cosine to
#     the anchor at ``layer0_input_hidden``, the corruption is definitively
#     upstream of self-attention -- i.e. in query-row construction / row
#     ordering / hidden-state plumbing, not in K/V reads.
#   * If the branch row also has the expected position stride
#     (``query_position = anchor_position + (branch_row - anchor_row)``), then
#     positions are not the immediate cause either; the row's hidden vector is
#     simply not the embedding of the token id the metadata says it carries.
#
# This test is tree-only and fully offline.  It reuses the extended Test-Q
# capture payload (``per_row_meta`` + ``per_row_attn_taps``) and annotates the
# captured steps with decoded positions from the tree summary.
# ---------------------------------------------------------------------------


def build_test_y_branch_input_origin_report(
    vllm_tree_q_capture: dict[str, Any],
    vllm_tree_summary: dict[str, Any],
    sample_index: int = 0,
    anchor_row: int | None = None,
    hidden_cos_threshold: float = 0.999,
) -> dict[str, Any]:
    """Test Y -- targeted tree-only audit of branch-row input construction.

    Parameters
    ----------
    vllm_tree_q_capture : output of ``retrieve_vllm_test_q_probe`` for the
        tree run.  Must include ``per_row_meta`` fields added by the updated
        Test-Q probe.
    vllm_tree_summary : per-sample tree summary used to annotate captured
        steps with decoded position / accepted_len / root_token.
    sample_index : sample id to inspect.
    anchor_row : row-inside-block to treat as the reference mask row.
        ``None`` falls back to the probe's recorded ``d0_row``.
    hidden_cos_threshold : cosine below this, with ``same_input_id=True``,
        counts as a pre-attention input-construction failure.
    """

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> list[dict[str, Any]]:
        for s in summary.get("samples") or []:
            if int(s.get("sample_index", -1)) != int(sample_idx):
                continue
            steps_sorted = sorted(
                s.get("steps") or [], key=lambda e: int(e.get("step", 0))
            )
            cumulative = 0
            out: list[dict[str, Any]] = []
            for e in steps_sorted:
                enriched = dict(e)
                enriched["__position"] = cumulative
                out.append(enriched)
                try:
                    cumulative += int(e.get("accepted_len", 0)) + 1
                except (TypeError, ValueError):
                    cumulative += 1
            return out
        return []

    def _cos(a: Any, b: Any) -> float | None:
        if not isinstance(a, list) or not isinstance(b, list):
            return None
        if not a or not b or len(a) != len(b):
            return None
        return float(_cosine(a, b))

    anchor_global_row = (
        int(anchor_row)
        if anchor_row is not None
        else int(vllm_tree_q_capture.get("d0_row") or 1)
    )
    capture_rows = [
        int(r) for r in (vllm_tree_q_capture.get("capture_rows") or [])
    ]
    branch_rows = [r for r in capture_rows if int(r) != int(anchor_global_row)]

    per_iter_annot: dict[int, dict[str, Any]] = {}
    for it in _per_iter_positions(vllm_tree_summary, sample_index):
        per_iter_annot[int(it.get("step", -1))] = {
            "position": int(it.get("__position", 0)),
            "accepted_len": it.get("accepted_len"),
            "root_token": it.get("root_token"),
        }

    per_iter: list[dict[str, Any]] = []
    num_branch_rows = 0
    num_same_input_id = 0
    num_same_input_low_cos = 0
    num_same_input_low_cos_with_expected_pos_stride = 0
    num_unexpected_pos_stride = 0

    for entry in vllm_tree_q_capture.get("per_step") or []:
        try:
            step = int(entry.get("step", -1))
        except Exception:
            continue
        per_row_taps = entry.get("per_row_attn_taps") or {}
        per_row_meta = entry.get("per_row_meta") or {}
        anchor_key = str(int(anchor_global_row))
        anchor_meta = per_row_meta.get(anchor_key) or {}
        anchor_hidden = (
            (per_row_taps.get(anchor_key) or {}).get("layer0_input_hidden")
        )
        branch_reports: list[dict[str, Any]] = []
        for br in branch_rows:
            br_key = str(int(br))
            br_meta = per_row_meta.get(br_key) or {}
            br_hidden = (per_row_taps.get(br_key) or {}).get(
                "layer0_input_hidden"
            )
            cos_to_anchor = _cos(anchor_hidden, br_hidden)
            same_input_id = (
                anchor_meta.get("query_input_id") == br_meta.get("query_input_id")
                if anchor_meta.get("query_input_id") is not None
                and br_meta.get("query_input_id") is not None
                else None
            )
            local_anchor = anchor_meta.get("local_row")
            local_branch = br_meta.get("local_row")
            anchor_pos = anchor_meta.get("query_position")
            branch_pos = br_meta.get("query_position")
            expected_pos_delta: int | None = None
            observed_pos_delta: int | None = None
            has_expected_pos_stride: bool | None = None
            if (
                isinstance(local_anchor, int)
                and isinstance(local_branch, int)
                and anchor_pos is not None
                and branch_pos is not None
            ):
                expected_pos_delta = int(local_branch) - int(local_anchor)
                observed_pos_delta = int(branch_pos) - int(anchor_pos)
                has_expected_pos_stride = (
                    int(observed_pos_delta) == int(expected_pos_delta)
                )
            diagnosis = "insufficient_metadata"
            if same_input_id is True:
                if cos_to_anchor is not None and cos_to_anchor < hidden_cos_threshold:
                    if has_expected_pos_stride is False:
                        diagnosis = (
                            "same_input_id_but_low_hidden_cos_and_wrong_pos_stride"
                        )
                    else:
                        diagnosis = (
                            "same_input_id_but_low_hidden_cos_before_attention"
                        )
                else:
                    diagnosis = "same_input_id_and_input_hidden_matches_anchor"
            elif same_input_id is False:
                if has_expected_pos_stride is False:
                    diagnosis = "different_input_id_and_wrong_pos_stride"
                else:
                    diagnosis = "different_input_id"
            elif has_expected_pos_stride is False:
                diagnosis = "unknown_input_id_but_wrong_pos_stride"

            branch_reports.append(
                {
                    "row": int(br),
                    "anchor_query_input_id": anchor_meta.get("query_input_id"),
                    "branch_query_input_id": br_meta.get("query_input_id"),
                    "same_input_id": same_input_id,
                    "anchor_query_position": anchor_pos,
                    "branch_query_position": branch_pos,
                    "expected_position_delta": expected_pos_delta,
                    "observed_position_delta": observed_pos_delta,
                    "has_expected_position_stride": has_expected_pos_stride,
                    "layer0_input_hidden_cos_to_anchor": cos_to_anchor,
                    "diagnosis": diagnosis,
                }
            )
            num_branch_rows += 1
            if same_input_id is True:
                num_same_input_id += 1
                if cos_to_anchor is not None and cos_to_anchor < hidden_cos_threshold:
                    num_same_input_low_cos += 1
                    if has_expected_pos_stride is True:
                        num_same_input_low_cos_with_expected_pos_stride += 1
            if has_expected_pos_stride is False:
                num_unexpected_pos_stride += 1

        annot = per_iter_annot.get(step) or {}
        per_iter.append(
            {
                "step": step,
                "position": annot.get("position"),
                "accepted_len": annot.get("accepted_len"),
                "root_token": annot.get("root_token"),
                "anchor_row": int(anchor_global_row),
                "anchor_meta": anchor_meta,
                "per_branch": branch_reports,
            }
        )

    aggregate = {
        "num_captured_iters": len(per_iter),
        "num_branch_rows_examined": int(num_branch_rows),
        "num_same_input_id_rows": int(num_same_input_id),
        "num_same_input_id_but_low_hidden_cos_rows": int(num_same_input_low_cos),
        "num_same_input_id_low_hidden_cos_with_expected_pos_stride": int(
            num_same_input_low_cos_with_expected_pos_stride
        ),
        "num_rows_with_unexpected_position_stride": int(
            num_unexpected_pos_stride
        ),
    }

    verdict: dict[str, Any] = {
        "sample_index": int(sample_index),
        "anchor_row": int(anchor_global_row),
        "branch_rows": branch_rows,
        "hidden_cos_threshold": float(hidden_cos_threshold),
        **aggregate,
    }
    if not branch_rows:
        verdict["interpretation_hint"] = (
            "No branch rows were captured in Test-Q. Re-run with "
            "``capture_rows=(d0_row, 2, ...)`` before using Test Y."
        )
    elif num_same_input_low_cos_with_expected_pos_stride > 0:
        verdict["interpretation_hint"] = (
            f"{num_same_input_low_cos_with_expected_pos_stride} captured branch "
            "rows have the SAME input_id as the anchor row, the EXPECTED "
            "position stride, yet already show low cosine at "
            "``layer0_input_hidden`` before attention. This localizes the bug "
            "to tree query-row input construction / row ordering / hidden-state "
            "plumbing upstream of self-attention; it is NOT explainable by KV "
            "content reads or by position metadata alone."
        )
    elif num_same_input_low_cos > 0:
        verdict["interpretation_hint"] = (
            f"{num_same_input_low_cos} captured branch rows have the SAME "
            "input_id as the anchor row but low ``layer0_input_hidden`` cosine. "
            "That already implicates pre-attention row construction. Inspect "
            "the per-branch position-stride fields to see whether position "
            "metadata is also malformed."
        )
    elif num_unexpected_pos_stride > 0:
        verdict["interpretation_hint"] = (
            f"{num_unexpected_pos_stride} captured branch rows show an "
            "unexpected query-position stride relative to the anchor row. "
            "This directly implicates row-to-position alignment in tree mode."
        )
    else:
        verdict["interpretation_hint"] = (
            "All captured branch rows either differ in token id as expected, or "
            "match the anchor row when they share the same token id. This "
            "refutes pre-attention query-row construction as the primary cause "
            "for the captured steps."
        )

    return {
        "sample_index": int(sample_index),
        "anchor_row": int(anchor_global_row),
        "branch_rows": branch_rows,
        "hidden_cos_threshold": float(hidden_cos_threshold),
        "per_iter": per_iter,
        "aggregate": aggregate,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Test Z: branch-row audit using ACTUAL draft-model forward inputs.
#
# Test Y sampled query_input_id from ``model_runner.input_ids.gpu`` at the
# layer hook site. That buffer can drift from the exact ids used by the current
# draft forward (e.g. conditioned-input / cloned model_kwargs paths), so Y is
# not authoritative. Test Z removes that ambiguity by capturing the *actual*
# ``input_ids`` / ``positions`` arguments passed into ``draft_inner.forward``.
#
# This lets us answer, for each captured branch row:
#   1. Did row 2 really receive a different token id than row 1?
#   2. If yes, does the low layer-0 input cosine simply reflect different
#      token embeddings rather than corruption?
#   3. If no, does low cosine persist even with identical actual input ids and
#      expected position stride, which would conclusively implicate pre-attn
#      row construction / hidden-state plumbing?
# ---------------------------------------------------------------------------


def build_test_z_actual_forward_input_report(
    vllm_tree_q_capture: dict[str, Any],
    vllm_tree_summary: dict[str, Any],
    sample_index: int = 0,
    anchor_row: int | None = None,
    hidden_cos_threshold: float = 0.999,
) -> dict[str, Any]:
    """Test Z -- tree-only branch audit using actual model-forward input ids."""

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> list[dict[str, Any]]:
        for s in summary.get("samples") or []:
            if int(s.get("sample_index", -1)) != int(sample_idx):
                continue
            steps_sorted = sorted(
                s.get("steps") or [], key=lambda e: int(e.get("step", 0))
            )
            cumulative = 0
            out: list[dict[str, Any]] = []
            for e in steps_sorted:
                enriched = dict(e)
                enriched["__position"] = cumulative
                out.append(enriched)
                try:
                    cumulative += int(e.get("accepted_len", 0)) + 1
                except (TypeError, ValueError):
                    cumulative += 1
            return out
        return []

    def _cos(a: Any, b: Any) -> float | None:
        if not isinstance(a, list) or not isinstance(b, list):
            return None
        if not a or not b or len(a) != len(b):
            return None
        return float(_cosine(a, b))

    anchor_global_row = (
        int(anchor_row)
        if anchor_row is not None
        else int(vllm_tree_q_capture.get("d0_row") or 1)
    )
    capture_rows = [
        int(r) for r in (vllm_tree_q_capture.get("capture_rows") or [])
    ]
    branch_rows = [r for r in capture_rows if int(r) != int(anchor_global_row)]

    per_iter_annot: dict[int, dict[str, Any]] = {}
    for it in _per_iter_positions(vllm_tree_summary, sample_index):
        per_iter_annot[int(it.get("step", -1))] = {
            "position": int(it.get("__position", 0)),
            "accepted_len": it.get("accepted_len"),
            "root_token": it.get("root_token"),
        }

    per_iter: list[dict[str, Any]] = []
    num_branch_rows = 0
    num_rows_missing_actual_meta = 0
    num_rows_same_actual_input_id = 0
    num_rows_diff_actual_input_id = 0
    num_same_actual_id_but_low_hidden_cos = 0
    num_diff_actual_id_and_low_hidden_cos = 0
    num_rows_with_unexpected_actual_pos_stride = 0

    for entry in vllm_tree_q_capture.get("per_step") or []:
        try:
            step = int(entry.get("step", -1))
        except Exception:
            continue
        per_row_taps = entry.get("per_row_attn_taps") or {}
        per_row_actual_meta = entry.get("per_row_actual_meta") or {}
        per_row_runner_meta = entry.get("per_row_meta") or {}
        anchor_key = str(int(anchor_global_row))
        anchor_actual = per_row_actual_meta.get(anchor_key) or {}
        anchor_runner = per_row_runner_meta.get(anchor_key) or {}
        anchor_hidden = (
            (per_row_taps.get(anchor_key) or {}).get("layer0_input_hidden")
        )

        branch_reports: list[dict[str, Any]] = []
        for br in branch_rows:
            br_key = str(int(br))
            br_actual = per_row_actual_meta.get(br_key) or {}
            br_runner = per_row_runner_meta.get(br_key) or {}
            br_hidden = (per_row_taps.get(br_key) or {}).get(
                "layer0_input_hidden"
            )
            cos_to_anchor = _cos(anchor_hidden, br_hidden)

            actual_same_input_id = (
                anchor_actual.get("query_input_id") == br_actual.get("query_input_id")
                if anchor_actual.get("query_input_id") is not None
                and br_actual.get("query_input_id") is not None
                else None
            )
            actual_anchor_pos = anchor_actual.get("query_position")
            actual_branch_pos = br_actual.get("query_position")
            actual_expected_delta: int | None = None
            actual_observed_delta: int | None = None
            actual_pos_stride_ok: bool | None = None
            if (
                isinstance(anchor_actual.get("local_row"), int)
                and isinstance(br_actual.get("local_row"), int)
                and actual_anchor_pos is not None
                and actual_branch_pos is not None
            ):
                actual_expected_delta = int(br_actual["local_row"]) - int(
                    anchor_actual["local_row"]
                )
                actual_observed_delta = int(actual_branch_pos) - int(
                    actual_anchor_pos
                )
                actual_pos_stride_ok = (
                    int(actual_expected_delta) == int(actual_observed_delta)
                )

            runner_same_input_id = (
                anchor_runner.get("query_input_id") == br_runner.get("query_input_id")
                if anchor_runner.get("query_input_id") is not None
                and br_runner.get("query_input_id") is not None
                else None
            )

            diagnosis = "missing_actual_forward_meta"
            if not anchor_actual or not br_actual:
                num_rows_missing_actual_meta += 1
            elif actual_same_input_id is True:
                num_rows_same_actual_input_id += 1
                if cos_to_anchor is not None and cos_to_anchor < hidden_cos_threshold:
                    num_same_actual_id_but_low_hidden_cos += 1
                    if actual_pos_stride_ok is False:
                        diagnosis = (
                            "same_actual_input_id_but_low_hidden_cos_and_bad_actual_pos"
                        )
                    else:
                        diagnosis = (
                            "same_actual_input_id_but_low_hidden_cos_before_attention"
                        )
                else:
                    diagnosis = "same_actual_input_id_and_hidden_matches_anchor"
            elif actual_same_input_id is False:
                num_rows_diff_actual_input_id += 1
                if cos_to_anchor is not None and cos_to_anchor < hidden_cos_threshold:
                    num_diff_actual_id_and_low_hidden_cos += 1
                    diagnosis = "different_actual_input_id_explains_low_hidden_cos"
                else:
                    diagnosis = "different_actual_input_id"
            else:
                diagnosis = "actual_input_id_unavailable"

            if actual_pos_stride_ok is False:
                num_rows_with_unexpected_actual_pos_stride += 1

            branch_reports.append(
                {
                    "row": int(br),
                    "actual_anchor_query_input_id": anchor_actual.get(
                        "query_input_id"
                    ),
                    "actual_branch_query_input_id": br_actual.get("query_input_id"),
                    "actual_same_input_id": actual_same_input_id,
                    "actual_anchor_query_position": actual_anchor_pos,
                    "actual_branch_query_position": actual_branch_pos,
                    "actual_expected_position_delta": actual_expected_delta,
                    "actual_observed_position_delta": actual_observed_delta,
                    "actual_position_stride_ok": actual_pos_stride_ok,
                    "runner_anchor_query_input_id": anchor_runner.get(
                        "query_input_id"
                    ),
                    "runner_branch_query_input_id": br_runner.get("query_input_id"),
                    "runner_same_input_id": runner_same_input_id,
                    "layer0_input_hidden_cos_to_anchor": cos_to_anchor,
                    "diagnosis": diagnosis,
                }
            )
            num_branch_rows += 1

        annot = per_iter_annot.get(step) or {}
        per_iter.append(
            {
                "step": step,
                "position": annot.get("position"),
                "accepted_len": annot.get("accepted_len"),
                "root_token": annot.get("root_token"),
                "anchor_row": int(anchor_global_row),
                "actual_anchor_meta": anchor_actual,
                "runner_anchor_meta": anchor_runner,
                "per_branch": branch_reports,
            }
        )

    aggregate = {
        "num_captured_iters": len(per_iter),
        "num_branch_rows_examined": int(num_branch_rows),
        "num_rows_missing_actual_forward_meta": int(num_rows_missing_actual_meta),
        "num_rows_same_actual_input_id": int(num_rows_same_actual_input_id),
        "num_rows_diff_actual_input_id": int(num_rows_diff_actual_input_id),
        "num_same_actual_input_id_but_low_hidden_cos": int(
            num_same_actual_id_but_low_hidden_cos
        ),
        "num_diff_actual_input_id_and_low_hidden_cos": int(
            num_diff_actual_id_and_low_hidden_cos
        ),
        "num_rows_with_unexpected_actual_position_stride": int(
            num_rows_with_unexpected_actual_pos_stride
        ),
    }

    verdict: dict[str, Any] = {
        "sample_index": int(sample_index),
        "anchor_row": int(anchor_global_row),
        "branch_rows": branch_rows,
        "hidden_cos_threshold": float(hidden_cos_threshold),
        **aggregate,
    }
    if not branch_rows:
        verdict["interpretation_hint"] = (
            "No branch rows were captured in Test-Q. Re-run with "
            "``capture_rows=(d0_row, 2, ...)`` before using Test Z."
        )
    elif num_rows_missing_actual_meta > 0 and num_branch_rows == num_rows_missing_actual_meta:
        verdict["interpretation_hint"] = (
            "Actual draft-model forward input metadata was missing for every "
            "captured branch row, so Test Z is inconclusive. Inspect the new "
            "draft_inner.forward pre-hook path."
        )
    elif num_same_actual_id_but_low_hidden_cos > 0:
        verdict["interpretation_hint"] = (
            f"{num_same_actual_id_but_low_hidden_cos} branch rows have the SAME "
            "actual forward input_id as the anchor row but already show low "
            "cosine at ``layer0_input_hidden``. This conclusively localizes the "
            "bug upstream of self-attention in query-row construction / hidden "
            "plumbing."
        )
    elif num_diff_actual_id_and_low_hidden_cos > 0:
        verdict["interpretation_hint"] = (
            f"{num_diff_actual_id_and_low_hidden_cos} low-cos branch rows also "
            "have DIFFERENT actual forward input ids from the anchor row. This "
            "means the old Test-X branch anomaly is compatible with intended "
            "different row semantics rather than pre-attention corruption."
        )
    elif num_rows_with_unexpected_actual_pos_stride > 0:
        verdict["interpretation_hint"] = (
            f"{num_rows_with_unexpected_actual_pos_stride} branch rows have an "
            "unexpected actual forward position stride. This implicates "
            "row-to-position alignment directly."
        )
    else:
        verdict["interpretation_hint"] = (
            "Actual forward-boundary metadata shows no evidence of pre-attention "
            "branch-row corruption. The top remaining suspect stays the depth-0 "
            "layer-0 self-attention divergence seen in Tests Q/W."
        )

    return {
        "sample_index": int(sample_index),
        "anchor_row": int(anchor_global_row),
        "branch_rows": branch_rows,
        "hidden_cos_threshold": float(hidden_cos_threshold),
        "per_iter": per_iter,
        "aggregate": aggregate,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Test AA: depth-0 visible-prefix audit for tree-vs-chain.
#
# Purpose: directly test the "read visibility" hypothesis for the depth-0 row.
# We reuse:
#   * Test Q capture: now includes the active forward-context attn metadata
#     (notably ``seq_lens`` and the query-token ``slot_mapping``) for the
#     current layer at each captured step.
#   * Test L capture: contains ``context_positions`` / ``context_slot_mapping``
#     for the tree and chain runs.
#
# For a matched decoded position, the active attention window for request 0 is
# determined by ``attn_metadata.seq_lens[0]``. Since DFlash query length is
# fixed per request, the visible *context* prefix length is:
#
#   visible_context_len = seq_lens[0] - num_query_per_req
#
# If the tree run is still reading a malformed tail, that should show up as one
# of the following inside the first ``visible_context_len`` entries:
#   * repeated / non-monotonic context positions
#   * PAD slots (-1) inside the visible prefix
#   * mismatch against chain's visible prefix at the same decoded position
#
# If the visible prefixes are sane and match, then the residual depth-0 split
# is probably *not* caused by a simple tail-visibility bug and we should look
# deeper at K/V content or kernel semantics.
# ---------------------------------------------------------------------------


def build_test_aa_depth0_visible_prefix_report(
    vllm_tree_q_capture: dict[str, Any],
    vllm_chain_q_capture: dict[str, Any],
    vllm_tree_l_capture: dict[str, Any],
    vllm_chain_l_capture: dict[str, Any],
    vllm_tree_summary: dict[str, Any],
    vllm_chain_summary: dict[str, Any],
    sample_index: int = 0,
) -> dict[str, Any]:
    """Test AA -- depth-0 self-attention visible-prefix audit."""

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> list[dict[str, Any]]:
        for s in summary.get("samples") or []:
            if int(s.get("sample_index", -1)) != int(sample_idx):
                continue
            steps_sorted = sorted(
                s.get("steps") or [], key=lambda e: int(e.get("step", 0))
            )
            cumulative = 0
            out: list[dict[str, Any]] = []
            for e in steps_sorted:
                enriched = dict(e)
                enriched["__position"] = cumulative
                out.append(enriched)
                try:
                    cumulative += int(e.get("accepted_len", 0)) + 1
                except (TypeError, ValueError):
                    cumulative += 1
            return out
        return []

    def _q_by_step(cap: dict[str, Any]) -> dict[int, dict[str, Any]]:
        by_step: dict[int, dict[str, Any]] = {}
        for entry in cap.get("per_step") or []:
            try:
                by_step[int(entry.get("step"))] = entry
            except Exception:
                continue
        return by_step

    def _l_by_step(cap: dict[str, Any]) -> dict[int, dict[str, Any]]:
        by_step: dict[int, dict[str, Any]] = {}
        for entry in cap.get("per_step") or []:
            try:
                by_step[int(entry.get("step"))] = entry
            except Exception:
                continue
        return by_step

    def _visible_context_len(q_entry: dict[str, Any], num_query_per_req: int) -> int | None:
        meta = q_entry.get("forward_attn_meta") or {}
        seq_lens = meta.get("seq_lens")
        if not isinstance(seq_lens, list) or not seq_lens:
            return None
        try:
            return int(seq_lens[0]) - int(num_query_per_req)
        except Exception:
            return None

    def _is_strictly_increasing(xs: Any) -> bool | None:
        if not isinstance(xs, list) or not xs:
            return None
        try:
            return all(int(xs[i]) < int(xs[i + 1]) for i in range(len(xs) - 1))
        except Exception:
            return None

    def _first_mismatch(a: Any, b: Any) -> int | None:
        if not isinstance(a, list) or not isinstance(b, list):
            return None
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return i
        if len(a) != len(b):
            return min(len(a), len(b))
        return None

    def _has_pad_slot(xs: Any) -> bool | None:
        if not isinstance(xs, list):
            return None
        return any(int(x) == -1 for x in xs)

    tree_num_query = int(vllm_tree_q_capture.get("num_query_per_req") or 0)
    chain_num_query = int(vllm_chain_q_capture.get("num_query_per_req") or 0)
    tree_q = _q_by_step(vllm_tree_q_capture)
    chain_q = _q_by_step(vllm_chain_q_capture)
    tree_l = _l_by_step(vllm_tree_l_capture)
    chain_l = _l_by_step(vllm_chain_l_capture)

    tree_pos_to_entry: dict[int, dict[str, Any]] = {}
    for it in _per_iter_positions(vllm_tree_summary, sample_index):
        si = int(it.get("step", -1))
        if si in tree_q and si in tree_l:
            tree_pos_to_entry[int(it["__position"])] = {
                "iter": si,
                "q": tree_q[si],
                "l": tree_l[si],
            }
    chain_pos_to_entry: dict[int, dict[str, Any]] = {}
    for it in _per_iter_positions(vllm_chain_summary, sample_index):
        si = int(it.get("step", -1))
        if si in chain_q and si in chain_l:
            chain_pos_to_entry[int(it["__position"])] = {
                "iter": si,
                "q": chain_q[si],
                "l": chain_l[si],
            }

    matched_positions = sorted(set(tree_pos_to_entry) & set(chain_pos_to_entry))
    per_pair: list[dict[str, Any]] = []
    num_pairs_with_tree_visible_prefix_anomaly = 0
    num_pairs_with_visible_prefix_mismatch = 0

    for pos in matched_positions:
        t = tree_pos_to_entry[pos]
        c = chain_pos_to_entry[pos]
        t_vis_len = _visible_context_len(t["q"], tree_num_query)
        c_vis_len = _visible_context_len(c["q"], chain_num_query)
        t_pos_full = (t["l"].get("context_positions") or [])
        c_pos_full = (c["l"].get("context_positions") or [])
        t_slot_full = (t["l"].get("context_slot_mapping") or [])
        c_slot_full = (c["l"].get("context_slot_mapping") or [])
        t_prefix_pos = (
            t_pos_full[:t_vis_len] if isinstance(t_vis_len, int) and t_vis_len >= 0 else None
        )
        c_prefix_pos = (
            c_pos_full[:c_vis_len] if isinstance(c_vis_len, int) and c_vis_len >= 0 else None
        )
        t_prefix_slot = (
            t_slot_full[:t_vis_len] if isinstance(t_vis_len, int) and t_vis_len >= 0 else None
        )
        c_prefix_slot = (
            c_slot_full[:c_vis_len] if isinstance(c_vis_len, int) and c_vis_len >= 0 else None
        )
        overlap = min(
            len(t_prefix_pos) if isinstance(t_prefix_pos, list) else 0,
            len(c_prefix_pos) if isinstance(c_prefix_pos, list) else 0,
        )
        pos_mismatch_idx = _first_mismatch(
            t_prefix_pos[:overlap] if isinstance(t_prefix_pos, list) else None,
            c_prefix_pos[:overlap] if isinstance(c_prefix_pos, list) else None,
        )
        slot_mismatch_idx = _first_mismatch(
            t_prefix_slot[:overlap] if isinstance(t_prefix_slot, list) else None,
            c_prefix_slot[:overlap] if isinstance(c_prefix_slot, list) else None,
        )
        tree_has_anomaly = any(
            [
                _is_strictly_increasing(t_prefix_pos) is False,
                _has_pad_slot(t_prefix_slot) is True,
            ]
        )
        visible_prefix_mismatch = (
            pos_mismatch_idx is not None or slot_mismatch_idx is not None or t_vis_len != c_vis_len
        )
        if tree_has_anomaly:
            num_pairs_with_tree_visible_prefix_anomaly += 1
        if visible_prefix_mismatch:
            num_pairs_with_visible_prefix_mismatch += 1
        per_pair.append(
            {
                "position": int(pos),
                "tree_iter": int(t["iter"]),
                "chain_iter": int(c["iter"]),
                "tree_visible_context_len": t_vis_len,
                "chain_visible_context_len": c_vis_len,
                "tree_visible_positions_head": (
                    t_prefix_pos[:8] if isinstance(t_prefix_pos, list) else None
                ),
                "chain_visible_positions_head": (
                    c_prefix_pos[:8] if isinstance(c_prefix_pos, list) else None
                ),
                "tree_visible_slots_head": (
                    t_prefix_slot[:8] if isinstance(t_prefix_slot, list) else None
                ),
                "chain_visible_slots_head": (
                    c_prefix_slot[:8] if isinstance(c_prefix_slot, list) else None
                ),
                "tree_positions_strictly_increasing": _is_strictly_increasing(t_prefix_pos),
                "chain_positions_strictly_increasing": _is_strictly_increasing(c_prefix_pos),
                "tree_visible_prefix_has_pad_slot": _has_pad_slot(t_prefix_slot),
                "chain_visible_prefix_has_pad_slot": _has_pad_slot(c_prefix_slot),
                "visible_prefix_overlap_len": overlap,
                "visible_positions_first_mismatch_idx": pos_mismatch_idx,
                "visible_slots_first_mismatch_idx": slot_mismatch_idx,
                "tree_visible_prefix_anomaly": tree_has_anomaly,
                "visible_prefix_mismatch": visible_prefix_mismatch,
                "tree_forward_attn_meta": t["q"].get("forward_attn_meta"),
                "chain_forward_attn_meta": c["q"].get("forward_attn_meta"),
            }
        )

    verdict: dict[str, Any] = {
        "sample_index": int(sample_index),
        "num_matched_pairs": len(per_pair),
        "num_pairs_with_tree_visible_prefix_anomaly": int(
            num_pairs_with_tree_visible_prefix_anomaly
        ),
        "num_pairs_with_visible_prefix_mismatch": int(
            num_pairs_with_visible_prefix_mismatch
        ),
    }
    if not per_pair:
        verdict["interpretation_hint"] = (
            "No matched decoded-position pairs had both Test-Q forward-attn "
            "metadata and Test-L context captures. Expand capture steps first."
        )
    elif num_pairs_with_tree_visible_prefix_anomaly > 0:
        verdict["interpretation_hint"] = (
            f"{num_pairs_with_tree_visible_prefix_anomaly}/{len(per_pair)} matched "
            "pairs show an anomaly inside the tree run's *visible* depth-0 "
            "context prefix (non-monotonic positions or PAD slots inside the "
            "active window). This strongly supports a read-visibility bug."
        )
    elif num_pairs_with_visible_prefix_mismatch > 0:
        verdict["interpretation_hint"] = (
            f"{num_pairs_with_visible_prefix_mismatch}/{len(per_pair)} matched "
            "pairs show tree-vs-chain mismatch in the visible depth-0 context "
            "prefix. This supports a tree-only read-path / visibility mismatch."
        )
    else:
        verdict["interpretation_hint"] = (
            "The visible depth-0 context prefixes are sane and match between tree "
            "and chain across matched pairs. This argues against a simple draft-"
            "tail visibility / logical-prefix bug and points instead to K/V "
            "content or attention-kernel semantics."
        )

    return {
        "sample_index": int(sample_index),
        "matched_positions": matched_positions,
        "per_matched_pair": per_pair,
        "summary": {
            "num_matched_pairs": len(per_pair),
            "num_pairs_with_tree_visible_prefix_anomaly": int(
                num_pairs_with_tree_visible_prefix_anomaly
            ),
            "num_pairs_with_visible_prefix_mismatch": int(
                num_pairs_with_visible_prefix_mismatch
            ),
        },
        "verdict": verdict,
    }


def build_test_ab_first_pass_context_compaction_report(
    runtime_bundle: dict[str, Any],
    vllm_tree_summary: dict[str, Any] | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    """Test AB -- DFlash first-pass accepted-context compaction audit.

    Uses one or more tree proposer runtime bundles captured from the executed
    vLLM path. The goal is to verify that the *visible accepted prefix* of the
    first-pass context buffers is compact and clean: no PAD slots, no repeated
    positions, and matching the expected accepted prefix length. When a tree
    summary is provided, bundles are annotated with decoded positions so later
    failing iterations can be inspected directly.
    """

    def _iter_bundles(payload: dict[str, Any]) -> list[dict[str, Any]]:
        bundles = payload.get("bundles")
        if isinstance(bundles, list):
            return [b for b in bundles if isinstance(b, dict)]
        if isinstance(payload.get("context_positions"), list):
            return [payload]
        return []

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> dict[int, dict[str, Any]]:
        sample_key = f"sample_{int(sample_idx):03d}"
        sample = summary.get(sample_key) or {}
        steps = sample.get("per_step") or []
        if not isinstance(steps, list):
            return {}
        out: dict[int, dict[str, Any]] = {}
        cumulative = 0
        for e in sorted(
            [x for x in steps if isinstance(x, dict)],
            key=lambda x: int(x.get("step", -1)),
        ):
            enriched = dict(e)
            enriched["__position"] = cumulative
            out[int(e.get("step", -1))] = enriched
            try:
                cumulative += int(e.get("accepted_len", 0)) + 1
            except Exception:
                cumulative += 1
        return out

    def _first_idx(xs: list[Any], pred) -> int | None:
        for i, x in enumerate(xs):
            try:
                if pred(x):
                    return i
            except Exception:
                continue
        return None

    def _strictly_increasing(xs: list[Any]) -> bool | None:
        if not xs:
            return None
        try:
            return all(int(xs[i]) < int(xs[i + 1]) for i in range(len(xs) - 1))
        except Exception:
            return None

    bundles = _iter_bundles(runtime_bundle)
    if not bundles:
        return {
            "skipped": True,
            "reason": "missing_context_buffers_in_runtime_bundle",
        }
    per_iter_annot = (
        _per_iter_positions(vllm_tree_summary, sample_index)
        if isinstance(vllm_tree_summary, dict)
        else {}
    )

    per_bundle: list[dict[str, Any]] = []
    num_prefix_pad = 0
    num_prefix_non_monotonic = 0
    num_visible_len_mismatch = 0
    num_target_prefix_mismatch = 0
    for bundle in sorted(bundles, key=lambda b: int(b.get("step", -1))):
        context_positions = bundle.get("context_positions")
        context_slot_mapping = bundle.get("context_slot_mapping")
        seq_lens = bundle.get("seq_lens")
        num_query_per_req = bundle.get("num_query_per_req")
        num_rejected_tokens = bundle.get("num_rejected_tokens")
        target_positions = bundle.get("target_positions")
        if not isinstance(context_positions, list) or not isinstance(
            context_slot_mapping, list
        ):
            continue
        if not isinstance(seq_lens, list) or not seq_lens:
            continue
        if not isinstance(num_query_per_req, int):
            continue
        step = int(bundle.get("step", -1))
        iter_annot = per_iter_annot.get(step) or {}

        visible_context_len: int | None = None
        try:
            visible_context_len = int(seq_lens[0]) - int(num_query_per_req)
        except Exception:
            visible_context_len = None

        expected_accepted_context_len: int | None = None
        try:
            if isinstance(num_rejected_tokens, list) and num_rejected_tokens:
                expected_accepted_context_len = len(context_positions) - int(
                    num_rejected_tokens[0]
                )
        except Exception:
            expected_accepted_context_len = None

        prefix_positions = (
            context_positions[:visible_context_len]
            if isinstance(visible_context_len, int) and visible_context_len >= 0
            else None
        )
        prefix_slots = (
            context_slot_mapping[:visible_context_len]
            if isinstance(visible_context_len, int) and visible_context_len >= 0
            else None
        )
        first_pad_idx = _first_idx(
            context_slot_mapping, lambda x: int(x) == -1
        )
        prefix_has_pad = (
            any(int(x) == -1 for x in prefix_slots)
            if isinstance(prefix_slots, list)
            else None
        )
        prefix_strict = (
            _strictly_increasing(prefix_positions)
            if isinstance(prefix_positions, list)
            else None
        )
        prefix_positions_match_target = None
        if isinstance(prefix_positions, list) and isinstance(target_positions, list):
            prefix_positions_match_target = (
                prefix_positions == target_positions[: len(prefix_positions)]
            )
        if prefix_has_pad is True:
            num_prefix_pad += 1
        if prefix_strict is False:
            num_prefix_non_monotonic += 1
        if (
            expected_accepted_context_len is not None
            and visible_context_len != expected_accepted_context_len
        ):
            num_visible_len_mismatch += 1
        if prefix_positions_match_target is False:
            num_target_prefix_mismatch += 1

        per_bundle.append(
            {
                "step": step,
                "decoded_position_from_summary": iter_annot.get("__position"),
                "accepted_len_from_summary": iter_annot.get("accepted_len"),
                "root_token_from_summary": iter_annot.get("root_token"),
                "visible_context_len": visible_context_len,
                "expected_accepted_context_len": expected_accepted_context_len,
                "num_context_rows": len(context_positions),
                "num_query_per_req": int(num_query_per_req),
                "num_rejected_tokens": num_rejected_tokens,
                "first_pad_idx": first_pad_idx,
                "prefix_has_pad_slot": prefix_has_pad,
                "prefix_positions_strictly_increasing": prefix_strict,
                "prefix_positions_match_target_prefix": (
                    prefix_positions_match_target
                ),
                "prefix_positions_head": (
                    prefix_positions[:12]
                    if isinstance(prefix_positions, list)
                    else None
                ),
                "prefix_slots_head": (
                    prefix_slots[:12] if isinstance(prefix_slots, list) else None
                ),
                "context_positions_head": context_positions[:12],
                "context_slot_mapping_head": context_slot_mapping[:12],
            }
        )

    verdict: dict[str, Any] = {
        "sample_index": int(sample_index),
        "num_bundles_analyzed": len(per_bundle),
        "num_bundles_with_prefix_pad_slot": int(num_prefix_pad),
        "num_bundles_with_non_monotonic_prefix_positions": int(
            num_prefix_non_monotonic
        ),
        "num_bundles_with_visible_len_mismatch": int(num_visible_len_mismatch),
        "num_bundles_with_target_prefix_mismatch": int(num_target_prefix_mismatch),
    }
    if not per_bundle:
        verdict["interpretation_hint"] = (
            "No runtime bundles were analyzable; confirm runtime capture was "
            "installed before the tree run."
        )
    elif (
        num_prefix_pad
        or num_prefix_non_monotonic
        or num_visible_len_mismatch
        or num_target_prefix_mismatch
    ):
        verdict["interpretation_hint"] = (
            "One or more captured tree iterations already show a malformed "
            "visible first-pass context prefix. This supports a later-step "
            "post-rejection prefix-compaction bug."
        )
    else:
        verdict["interpretation_hint"] = (
            "All captured tree iterations show a compact visible first-pass "
            "context prefix. The remaining bug is likely after this stage."
        )

    return {
        "sample_index": int(sample_index),
        "capture_steps_requested": runtime_bundle.get("capture_steps_requested"),
        "captured_steps": runtime_bundle.get("captured_steps"),
        "per_bundle": per_bundle,
        "summary": {
            "num_bundles_analyzed": len(per_bundle),
            "num_bundles_with_prefix_pad_slot": int(num_prefix_pad),
            "num_bundles_with_non_monotonic_prefix_positions": int(
                num_prefix_non_monotonic
            ),
            "num_bundles_with_visible_len_mismatch": int(
                num_visible_len_mismatch
            ),
            "num_bundles_with_target_prefix_mismatch": int(
                num_target_prefix_mismatch
            ),
        },
        "verdict": verdict,
    }


def build_test_ae_seq_len_derivation_report(
    runtime_bundle: dict[str, Any],
    vllm_tree_summary: dict[str, Any] | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    """Test AE -- identify which source produced first-pass visible seq_lens."""

    def _iter_bundles(payload: dict[str, Any]) -> list[dict[str, Any]]:
        bundles = payload.get("bundles")
        if isinstance(bundles, list):
            return [b for b in bundles if isinstance(b, dict)]
        if isinstance(payload.get("context_positions"), list):
            return [payload]
        return []

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> dict[int, dict[str, Any]]:
        sample_key = f"sample_{int(sample_idx):03d}"
        sample = summary.get(sample_key) or {}
        steps = sample.get("per_step") or []
        if not isinstance(steps, list):
            return {}
        out: dict[int, dict[str, Any]] = {}
        cumulative = 0
        for e in sorted(
            [x for x in steps if isinstance(x, dict)],
            key=lambda x: int(x.get("step", -1)),
        ):
            enriched = dict(e)
            enriched["__position"] = cumulative
            out[int(e.get("step", -1))] = enriched
            try:
                cumulative += int(e.get("accepted_len", 0)) + 1
            except Exception:
                cumulative += 1
        return out

    def _front_valid_len(xs: list[Any]) -> int | None:
        if not isinstance(xs, list):
            return None
        n = 0
        for x in xs:
            try:
                if int(x) == -1:
                    break
            except Exception:
                break
            n += 1
        return n

    def _head0(xs: Any) -> int | None:
        if isinstance(xs, list) and xs:
            try:
                return int(xs[0])
            except Exception:
                return None
        return None

    bundles = _iter_bundles(runtime_bundle)
    if not bundles:
        return {"skipped": True, "reason": "missing_runtime_bundles"}
    per_iter_annot = (
        _per_iter_positions(vllm_tree_summary, sample_index)
        if isinstance(vllm_tree_summary, dict)
        else {}
    )

    per_bundle: list[dict[str, Any]] = []
    num_matches_seq_minus_rejected = 0
    num_matches_compacted_context = 0
    num_matches_front_valid = 0
    num_seq_not_compacted = 0
    for bundle in sorted(bundles, key=lambda b: int(b.get("step", -1))):
        step = int(bundle.get("step", -1))
        annot = per_iter_annot.get(step) or {}
        context_slot_mapping = bundle.get("context_slot_mapping")
        output_visible_context_lens = bundle.get("output_visible_context_lens")
        input_seq_lens_minus_rejected = bundle.get("input_seq_lens_minus_rejected")
        input_compacted_context_lens = bundle.get("input_compacted_context_lens")
        input_context_lens = bundle.get("input_context_lens_from_query_start_loc")
        input_cad_seq_lens = bundle.get("input_cad_seq_lens")
        num_rejected_tokens = bundle.get("num_rejected_tokens")
        if not isinstance(context_slot_mapping, list):
            continue

        output_visible = _head0(output_visible_context_lens)
        if output_visible is None:
            seq_lens = bundle.get("seq_lens")
            num_query_per_req = bundle.get("num_query_per_req")
            if (
                isinstance(seq_lens, list)
                and seq_lens
                and isinstance(num_query_per_req, int)
            ):
                try:
                    output_visible = int(seq_lens[0]) - int(num_query_per_req)
                except Exception:
                    output_visible = None
        seq_minus_rejected = _head0(input_seq_lens_minus_rejected)
        compacted_context = _head0(input_compacted_context_lens)
        raw_context_rows = _head0(input_context_lens)
        input_cad_seq_len0 = _head0(input_cad_seq_lens)
        front_valid_context_len = _front_valid_len(context_slot_mapping)

        matches_seq_minus_rejected = (
            output_visible == seq_minus_rejected
            if output_visible is not None and seq_minus_rejected is not None
            else None
        )
        matches_compacted_context = (
            output_visible == compacted_context
            if output_visible is not None and compacted_context is not None
            else None
        )
        matches_front_valid = (
            output_visible == front_valid_context_len
            if output_visible is not None and front_valid_context_len is not None
            else None
        )

        if matches_seq_minus_rejected is True:
            num_matches_seq_minus_rejected += 1
        if matches_compacted_context is True:
            num_matches_compacted_context += 1
        if matches_front_valid is True:
            num_matches_front_valid += 1
        if matches_seq_minus_rejected is True and matches_compacted_context is False:
            num_seq_not_compacted += 1

        per_bundle.append(
            {
                "step": step,
                "decoded_position_from_summary": annot.get("__position"),
                "accepted_len_from_summary": annot.get("accepted_len"),
                "root_token_from_summary": annot.get("root_token"),
                "input_cad_seq_len0": input_cad_seq_len0,
                "num_rejected_tokens0": _head0(num_rejected_tokens),
                "input_context_rows0": raw_context_rows,
                "input_seq_minus_rejected0": seq_minus_rejected,
                "input_compacted_context_len0": compacted_context,
                "front_valid_context_len": front_valid_context_len,
                "output_visible_context_len0": output_visible,
                "output_matches_input_seq_minus_rejected": (
                    matches_seq_minus_rejected
                ),
                "output_matches_input_compacted_context_len": (
                    matches_compacted_context
                ),
                "output_matches_front_valid_context_len": matches_front_valid,
            }
        )

    verdict: dict[str, Any] = {
        "sample_index": int(sample_index),
        "num_bundles_analyzed": len(per_bundle),
        "num_bundles_where_output_matches_input_seq_minus_rejected": int(
            num_matches_seq_minus_rejected
        ),
        "num_bundles_where_output_matches_input_compacted_context_len": int(
            num_matches_compacted_context
        ),
        "num_bundles_where_output_matches_front_valid_context_len": int(
            num_matches_front_valid
        ),
        "num_bundles_where_output_matches_seq_minus_rejected_but_not_compacted": int(
            num_seq_not_compacted
        ),
    }
    if not per_bundle:
        verdict["interpretation_hint"] = (
            "No runtime bundles were analyzable; confirm runtime capture was "
            "installed before the tree run."
        )
    elif num_seq_not_compacted:
        verdict["interpretation_hint"] = (
            "At one or more captured tree steps, the emitted visible-context "
            "length matches ``cad.seq_lens - num_rejected`` but not the compacted "
            "accepted-prefix length from ``query_start_loc`` / front-valid rows. "
            "This strongly points to a seq_lens derivation bug rather than a "
            "forward-context handoff bug."
        )
    else:
        verdict["interpretation_hint"] = (
            "The emitted visible-context length does not preferentially track "
            "the logical ``cad.seq_lens - num_rejected`` path. The remaining "
            "bug may be in compaction itself or in another upstream metadata source."
        )

    return {
        "sample_index": int(sample_index),
        "capture_steps_requested": runtime_bundle.get("capture_steps_requested"),
        "captured_steps": runtime_bundle.get("captured_steps"),
        "per_bundle": per_bundle,
        "summary": {
            "num_bundles_analyzed": len(per_bundle),
            "num_bundles_where_output_matches_input_seq_minus_rejected": int(
                num_matches_seq_minus_rejected
            ),
            "num_bundles_where_output_matches_input_compacted_context_len": int(
                num_matches_compacted_context
            ),
            "num_bundles_where_output_matches_front_valid_context_len": int(
                num_matches_front_valid
            ),
            "num_bundles_where_output_matches_seq_minus_rejected_but_not_compacted": int(
                num_seq_not_compacted
            ),
        },
        "verdict": verdict,
    }


def build_test_af_rejection_count_producer_report(
    runtime_bundle: dict[str, Any],
    vllm_tree_summary: dict[str, Any] | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    """Test AF -- producer-side rejection accounting audit.

    This explains whether the tiny compacted context seen by Test AE is already
    implied by the producer-side values from ``prepare_next_token_ids_padded`` /
    ``prepare_inputs_padded`` (valid sampled-token count, rejected-token count,
    and token-index selection), before ``set_inputs_first_pass`` or the
    copy/expand kernel can distort anything.
    """

    def _iter_bundles(payload: dict[str, Any]) -> list[dict[str, Any]]:
        bundles = payload.get("bundles")
        if isinstance(bundles, list):
            return [b for b in bundles if isinstance(b, dict)]
        if isinstance(payload.get("context_positions"), list):
            return [payload]
        return []

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> dict[int, dict[str, Any]]:
        sample_key = f"sample_{int(sample_idx):03d}"
        sample = summary.get(sample_key) or {}
        steps = sample.get("per_step") or []
        if not isinstance(steps, list):
            return {}
        out: dict[int, dict[str, Any]] = {}
        cumulative = 0
        for e in sorted(
            [x for x in steps if isinstance(x, dict)],
            key=lambda x: int(x.get("step", -1)),
        ):
            enriched = dict(e)
            enriched["__position"] = cumulative
            out[int(e.get("step", -1))] = enriched
            try:
                cumulative += int(e.get("accepted_len", 0)) + 1
            except Exception:
                cumulative += 1
        return out

    def _head0(xs: Any) -> int | None:
        if isinstance(xs, list) and xs:
            try:
                return int(xs[0])
            except Exception:
                return None
        return None

    def _front_valid_len(xs: list[Any]) -> int | None:
        if not isinstance(xs, list):
            return None
        n = 0
        for x in xs:
            try:
                if int(x) == -1:
                    break
            except Exception:
                break
            n += 1
        return n

    bundles = _iter_bundles(runtime_bundle)
    if not bundles:
        return {"skipped": True, "reason": "missing_runtime_bundles"}
    per_iter_annot = (
        _per_iter_positions(vllm_tree_summary, sample_index)
        if isinstance(vllm_tree_summary, dict)
        else {}
    )

    per_bundle: list[dict[str, Any]] = []
    num_formula_match = 0
    num_token_index_match = 0
    num_compacted_matches_front_valid = 0
    num_tiny_valid_counts = 0
    for bundle in sorted(bundles, key=lambda b: int(b.get("step", -1))):
        step = int(bundle.get("step", -1))
        annot = per_iter_annot.get(step) or {}
        prepare_valid = bundle.get("prepare_inputs_valid_sampled_tokens_count")
        prepare_cu = bundle.get("prepare_inputs_cu_num_draft_tokens")
        prepare_qsl = bundle.get("prepare_inputs_query_start_loc")
        prepare_token_indices = bundle.get("prepare_inputs_token_indices_to_sample")
        prepare_num_rejected = bundle.get("prepare_inputs_num_rejected_tokens")
        context_slot_mapping = bundle.get("context_slot_mapping")
        input_compacted_context_lens = bundle.get("input_compacted_context_lens")

        valid_count0 = _head0(prepare_valid)
        cu_num_draft0 = _head0(prepare_cu)
        qsl0 = _head0(prepare_qsl)
        qsl1 = prepare_qsl[1] if isinstance(prepare_qsl, list) and len(prepare_qsl) > 1 else None
        token_index0 = _head0(prepare_token_indices)
        num_rejected0 = _head0(prepare_num_rejected)
        compacted_context_len0 = _head0(input_compacted_context_lens)
        front_valid_context_len = _front_valid_len(context_slot_mapping)

        expected_num_rejected0 = None
        if cu_num_draft0 is not None and valid_count0 is not None:
            expected_num_rejected0 = (
                int(cu_num_draft0) + 1 - int(valid_count0)
                if int(cu_num_draft0) > 0
                else 0
            )
        rejection_formula_matches = (
            int(num_rejected0) == int(expected_num_rejected0)
            if num_rejected0 is not None and expected_num_rejected0 is not None
            else None
        )

        query_rows0 = None
        q_last_tok_idx0 = None
        expected_token_index0 = None
        if qsl0 is not None and qsl1 is not None:
            try:
                query_rows0 = int(qsl1) - int(qsl0)
                q_last_tok_idx0 = int(qsl1) - 1
            except Exception:
                query_rows0 = None
                q_last_tok_idx0 = None
        if q_last_tok_idx0 is not None and num_rejected0 is not None:
            expected_token_index0 = int(q_last_tok_idx0) - int(num_rejected0)
        token_index_matches = (
            int(token_index0) == int(expected_token_index0)
            if token_index0 is not None and expected_token_index0 is not None
            else None
        )

        compacted_matches_front_valid = (
            int(compacted_context_len0) == int(front_valid_context_len)
            if compacted_context_len0 is not None
            and front_valid_context_len is not None
            else None
        )

        if rejection_formula_matches is True:
            num_formula_match += 1
        if token_index_matches is True:
            num_token_index_match += 1
        if compacted_matches_front_valid is True:
            num_compacted_matches_front_valid += 1
        if isinstance(valid_count0, int) and valid_count0 <= 2:
            num_tiny_valid_counts += 1

        per_bundle.append(
            {
                "step": step,
                "decoded_position_from_summary": annot.get("__position"),
                "accepted_len_from_summary": annot.get("accepted_len"),
                "root_token_from_summary": annot.get("root_token"),
                "valid_sampled_tokens_count0": valid_count0,
                "cu_num_draft_tokens0": cu_num_draft0,
                "expected_num_rejected0": expected_num_rejected0,
                "actual_num_rejected0": num_rejected0,
                "rejection_formula_matches": rejection_formula_matches,
                "query_rows0": query_rows0,
                "q_last_tok_idx0": q_last_tok_idx0,
                "token_index_to_sample0": token_index0,
                "expected_token_index0": expected_token_index0,
                "token_index_matches_formula": token_index_matches,
                "input_compacted_context_len0": compacted_context_len0,
                "front_valid_context_len": front_valid_context_len,
                "compacted_matches_front_valid": compacted_matches_front_valid,
            }
        )

    verdict: dict[str, Any] = {
        "sample_index": int(sample_index),
        "num_bundles_analyzed": len(per_bundle),
        "num_bundles_with_rejection_formula_match": int(num_formula_match),
        "num_bundles_with_token_index_formula_match": int(num_token_index_match),
        "num_bundles_with_compacted_matches_front_valid": int(
            num_compacted_matches_front_valid
        ),
        "num_bundles_with_tiny_valid_sampled_count_le_2": int(num_tiny_valid_counts),
    }
    if not per_bundle:
        verdict["interpretation_hint"] = (
            "No runtime bundles were analyzable; confirm runtime capture was "
            "installed before the tree run."
        )
    elif (
        num_formula_match == len(per_bundle)
        and num_token_index_match == len(per_bundle)
        and num_compacted_matches_front_valid == len(per_bundle)
    ):
        verdict["interpretation_hint"] = (
            "The compacted context collapse is already fully implied by the "
            "producer-side values (`valid_sampled_tokens_count`, "
            "`num_rejected_tokens`, and `token_indices_to_sample`) before "
            "`set_inputs_first_pass`. The next inspection target is the "
            "producer-side rejected-token accounting rather than the "
            "copy/expand kernel itself."
        )
    else:
        verdict["interpretation_hint"] = (
            "One or more producer-side formula checks failed, so the collapse may "
            "still be introduced between producer-side preparation and first-pass "
            "materialization."
        )

    return {
        "sample_index": int(sample_index),
        "capture_steps_requested": runtime_bundle.get("capture_steps_requested"),
        "captured_steps": runtime_bundle.get("captured_steps"),
        "per_bundle": per_bundle,
        "summary": {
            "num_bundles_analyzed": len(per_bundle),
            "num_bundles_with_rejection_formula_match": int(num_formula_match),
            "num_bundles_with_token_index_formula_match": int(num_token_index_match),
            "num_bundles_with_compacted_matches_front_valid": int(
                num_compacted_matches_front_valid
            ),
            "num_bundles_with_tiny_valid_sampled_count_le_2": int(
                num_tiny_valid_counts
            ),
        },
        "verdict": verdict,
    }


def build_test_ag_tree_attn_builder_passthrough_report(
    builder_probe: dict[str, Any],
    runtime_bundle: dict[str, Any] | None = None,
    vllm_tree_summary: dict[str, Any] | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    """Test AG -- audit whether TreeAttentionMetadataBuilder mutates DFlash lengths.

    This compares the builder input (`CommonAttentionMetadata`) against the
    emitted `TreeAttentionMetadata` / `decode_metadata` fields for the same
    DFlash tree step. If everything is identical, the builder is a pure
    pass-through and the remaining suspect moves to the caller or backend
    consumption path rather than the builder itself.

    By default this report isolates the *draft first-pass* builder path
    (`builder_owner == "drafter"` and
    `caller_role == "drafter_build_per_group_and_layer_attn_metadata"`),
    because that is the metadata instance relevant to the first-pass draft
    attention bug.
    """

    def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
        xs = payload.get("records")
        if isinstance(xs, list):
            return [x for x in xs if isinstance(x, dict)]
        return []

    def _bundles(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        xs = payload.get("bundles")
        if isinstance(xs, list):
            return [x for x in xs if isinstance(x, dict)]
        if isinstance(payload.get("context_positions"), list):
            return [payload]
        return []

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> dict[int, dict[str, Any]]:
        sample_key = f"sample_{int(sample_idx):03d}"
        sample = summary.get(sample_key) or {}
        steps = sample.get("per_step") or []
        if not isinstance(steps, list):
            return {}
        out: dict[int, dict[str, Any]] = {}
        cumulative = 0
        for e in sorted(
            [x for x in steps if isinstance(x, dict)],
            key=lambda x: int(x.get("step", -1)),
        ):
            enriched = dict(e)
            enriched["__position"] = cumulative
            out[int(e.get("step", -1))] = enriched
            try:
                cumulative += int(e.get("accepted_len", 0)) + 1
            except Exception:
                cumulative += 1
        return out

    def _head0(xs: Any) -> int | None:
        if isinstance(xs, list) and xs:
            try:
                return int(xs[0])
            except Exception:
                return None
        return None

    records = _records(builder_probe)
    if not records:
        return {"skipped": True, "reason": "missing_builder_probe_records"}

    all_records_summary: dict[str, int] = {}
    for record in records:
        owner = str(record.get("builder_owner") or record.get("owner_name") or "unknown")
        role = str(record.get("caller_role") or "unknown")
        method = str(record.get("build_method") or "unknown")
        key = f"{owner}:{role}:{method}"
        all_records_summary[key] = int(all_records_summary.get(key, 0)) + 1

    selection_applied = {
        "builder_owner": "drafter",
        "caller_role": "drafter_build_per_group_and_layer_attn_metadata",
        "build_method": "build_for_drafting",
    }
    selected_records = [
        r
        for r in records
        if str(r.get("builder_owner") or r.get("owner_name") or "") == "drafter"
        and str(r.get("caller_role") or "")
        == "drafter_build_per_group_and_layer_attn_metadata"
        and str(r.get("build_method") or "") == "build_for_drafting"
    ]
    if not selected_records:
        selected_records = [
            r
            for r in records
            if str(r.get("builder_owner") or r.get("owner_name") or "") == "drafter"
            and str(r.get("build_method") or "") == "build_for_drafting"
        ]
        selection_applied = {
            "builder_owner": "drafter",
            "caller_role": None,
            "build_method": "build_for_drafting",
        }
    if not selected_records:
        selected_records = records
        selection_applied = {
            "builder_owner": None,
            "caller_role": None,
            "build_method": None,
        }

    runtime_by_step = {
        int(b.get("step", -1)): b for b in _bundles(runtime_bundle)
    }
    per_iter_annot = (
        _per_iter_positions(vllm_tree_summary, sample_index)
        if isinstance(vllm_tree_summary, dict)
        else {}
    )

    per_record: list[dict[str, Any]] = []
    num_output_seq_equal_input = 0
    num_decode_seq_equal_output = 0
    num_output_qsl_equal_input = 0
    num_output_max_seq_equal_input = 0
    num_runtime_seq_match = 0
    for record in sorted(
        selected_records,
        key=lambda r: (
            int(r.get("tree_propose_step", -1)),
            int(r.get("kv_cache_group_id", -1)),
            int(r.get("attn_group_id", -1)),
        ),
    ):
        step = int(record.get("tree_propose_step", -1))
        annot = per_iter_annot.get(step) or {}
        runtime = runtime_by_step.get(step) or {}

        input_seq0 = _head0(record.get("input_seq_lens"))
        output_seq0 = _head0(record.get("output_seq_lens"))
        decode_seq0 = _head0(record.get("decode_seq_lens"))
        runtime_seq0 = _head0(runtime.get("seq_lens"))
        runtime_visible0 = _head0(runtime.get("output_visible_context_lens"))
        num_query_per_req = runtime.get("num_query_per_req")
        expected_runtime_seq0 = None
        if runtime_visible0 is not None and num_query_per_req is not None:
            try:
                expected_runtime_seq0 = (
                    int(runtime_visible0) + int(num_query_per_req)
                )
            except Exception:
                expected_runtime_seq0 = None

        output_seq_equal_input = (
            int(output_seq0) == int(input_seq0)
            if input_seq0 is not None and output_seq0 is not None
            else None
        )
        decode_seq_equal_output = (
            int(decode_seq0) == int(output_seq0)
            if output_seq0 is not None and decode_seq0 is not None
            else None
        )
        output_qsl_equal_input = (
            list(record.get("output_query_start_loc") or [])
            == list(record.get("input_query_start_loc") or [])
        )
        output_max_seq_equal_input = (
            int(record.get("output_max_seq_len")) == int(record.get("input_max_seq_len"))
            if record.get("output_max_seq_len") is not None
            and record.get("input_max_seq_len") is not None
            else None
        )
        runtime_seq_matches_builder = (
            int(runtime_seq0) == int(output_seq0)
            if runtime_seq0 is not None and output_seq0 is not None
            else None
        )

        if output_seq_equal_input is True:
            num_output_seq_equal_input += 1
        if decode_seq_equal_output is True:
            num_decode_seq_equal_output += 1
        if output_qsl_equal_input is True:
            num_output_qsl_equal_input += 1
        if output_max_seq_equal_input is True:
            num_output_max_seq_equal_input += 1
        if runtime_seq_matches_builder is True:
            num_runtime_seq_match += 1

        per_record.append(
            {
                "step": step,
                "decoded_position_from_summary": annot.get("__position"),
                "accepted_len_from_summary": annot.get("accepted_len"),
                "builder_owner": record.get("builder_owner") or record.get("owner_name"),
                "caller_role": record.get("caller_role"),
                "build_method": record.get("build_method"),
                "draft_index": record.get("draft_index"),
                "kv_cache_group_id": record.get("kv_cache_group_id"),
                "attn_group_id": record.get("attn_group_id"),
                "for_cudagraph_capture": record.get("for_cudagraph_capture"),
                "input_seq_len0": input_seq0,
                "output_seq_len0": output_seq0,
                "decode_seq_len0": decode_seq0,
                "runtime_seq_len0": runtime_seq0,
                "runtime_visible_context_len0": runtime_visible0,
                "runtime_num_query_per_req": num_query_per_req,
                "expected_runtime_seq_len0_from_visible_plus_query": (
                    expected_runtime_seq0
                ),
                "output_seq_len_matches_input": output_seq_equal_input,
                "decode_seq_len_matches_output": decode_seq_equal_output,
                "output_query_start_loc_matches_input": output_qsl_equal_input,
                "output_max_seq_len_matches_input": output_max_seq_equal_input,
                "runtime_seq_len_matches_builder_output": runtime_seq_matches_builder,
            }
        )

    verdict: dict[str, Any] = {
        "sample_index": int(sample_index),
        "num_records_analyzed": len(per_record),
        "num_records_where_output_seq_len_matches_input": int(
            num_output_seq_equal_input
        ),
        "num_records_where_decode_seq_len_matches_output": int(
            num_decode_seq_equal_output
        ),
        "num_records_where_output_query_start_loc_matches_input": int(
            num_output_qsl_equal_input
        ),
        "num_records_where_output_max_seq_len_matches_input": int(
            num_output_max_seq_equal_input
        ),
        "num_records_where_runtime_seq_len_matches_builder_output": int(
            num_runtime_seq_match
        ),
    }
    if (
        per_record
        and num_output_seq_equal_input == len(per_record)
        and num_decode_seq_equal_output == len(per_record)
        and num_output_qsl_equal_input == len(per_record)
        and num_output_max_seq_equal_input == len(per_record)
    ):
        verdict["interpretation_hint"] = (
            "TreeAttentionMetadataBuilder is a pure pass-through for DFlash tree "
            "metadata on the captured steps: it forwards `seq_lens`, "
            "`max_seq_len`, and `query_start_loc` unchanged into decode metadata. "
            "If the lengths are wrong, the source is upstream in the caller's "
            "`CommonAttentionMetadata` or downstream in backend/kernel consumption."
        )
    else:
        verdict["interpretation_hint"] = (
            "The builder mutated one or more fields between input common metadata "
            "and emitted decode metadata, so the builder path itself remains a "
            "live suspect."
        )

    return {
        "sample_index": int(sample_index),
        "captured_steps": builder_probe.get("captured_steps"),
        "num_records_captured": builder_probe.get("num_records_captured"),
        "all_records_summary": all_records_summary,
        "selection": {
            **selection_applied,
            "selected_records": len(selected_records),
        },
        "per_record": per_record,
        "summary": {
            "num_records_analyzed": len(per_record),
            "num_records_where_output_seq_len_matches_input": int(
                num_output_seq_equal_input
            ),
            "num_records_where_decode_seq_len_matches_output": int(
                num_decode_seq_equal_output
            ),
            "num_records_where_output_query_start_loc_matches_input": int(
                num_output_qsl_equal_input
            ),
            "num_records_where_output_max_seq_len_matches_input": int(
                num_output_max_seq_equal_input
            ),
            "num_records_where_runtime_seq_len_matches_builder_output": int(
                num_runtime_seq_match
            ),
        },
        "verdict": verdict,
    }


def build_test_ah_drafter_first_pass_metadata_report(
    drafter_probe: dict[str, Any],
    runtime_bundle: dict[str, Any] | None = None,
    vllm_tree_summary: dict[str, Any] | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    """Test AH -- direct drafter first-pass metadata audit.

    This compares the exact `CommonAttentionMetadata` seen by the drafter's
    first-pass `build_per_group_and_layer_attn_metadata(..., draft_index=0)`
    path against the DFlash runtime bundle captured around first-pass input
    preparation. If they already agree on the long logical `seq_lens`, then the
    bug is upstream of the drafter builder itself.
    """

    def _snapshots(payload: dict[str, Any]) -> list[dict[str, Any]]:
        xs = payload.get("snapshots")
        if isinstance(xs, list):
            return [x for x in xs if isinstance(x, dict)]
        return []

    def _bundles(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        xs = payload.get("bundles")
        if isinstance(xs, list):
            return [x for x in xs if isinstance(x, dict)]
        if isinstance(payload.get("context_positions"), list):
            return [payload]
        return []

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> dict[int, dict[str, Any]]:
        sample_key = f"sample_{int(sample_idx):03d}"
        sample = summary.get(sample_key) or {}
        steps = sample.get("per_step") or []
        if not isinstance(steps, list):
            return {}
        out: dict[int, dict[str, Any]] = {}
        cumulative = 0
        for e in sorted(
            [x for x in steps if isinstance(x, dict)],
            key=lambda x: int(x.get("step", -1)),
        ):
            enriched = dict(e)
            enriched["__position"] = cumulative
            out[int(e.get("step", -1))] = enriched
            try:
                cumulative += int(e.get("accepted_len", 0)) + 1
            except Exception:
                cumulative += 1
        return out

    def _head0(xs: Any) -> int | None:
        if isinstance(xs, list) and xs:
            try:
                return int(xs[0])
            except Exception:
                return None
        return None

    snapshots = _snapshots(drafter_probe)
    if not snapshots:
        return {"skipped": True, "reason": "missing_drafter_first_pass_snapshots"}

    runtime_by_step = {int(b.get("step", -1)): b for b in _bundles(runtime_bundle)}
    per_iter_annot = (
        _per_iter_positions(vllm_tree_summary, sample_index)
        if isinstance(vllm_tree_summary, dict)
        else {}
    )

    per_snapshot: list[dict[str, Any]] = []
    num_seq_match_runtime = 0
    num_max_query_len_match_runtime = 0
    num_seq_matches_logical_path = 0
    num_seq_differs_from_compacted = 0
    for snap in sorted(snapshots, key=lambda s: int(s.get("tree_propose_step", -1))):
        step = int(snap.get("tree_propose_step", -1))
        runtime = runtime_by_step.get(step) or {}
        annot = per_iter_annot.get(step) or {}

        drafter_seq0 = _head0(snap.get("seq_lens"))
        drafter_qsl = snap.get("query_start_loc")
        drafter_max_query_len = snap.get("max_query_len")
        runtime_seq0 = _head0(runtime.get("seq_lens"))
        runtime_logical0 = _head0(runtime.get("input_seq_lens_minus_rejected"))
        runtime_compacted0 = _head0(runtime.get("input_compacted_context_lens"))
        runtime_num_query_per_req = runtime.get("num_query_per_req")

        seq_matches_runtime = (
            int(drafter_seq0) == int(runtime_seq0)
            if drafter_seq0 is not None and runtime_seq0 is not None
            else None
        )
        max_query_len_matches_runtime = (
            int(drafter_max_query_len) == int(runtime_num_query_per_req)
            if drafter_max_query_len is not None and runtime_num_query_per_req is not None
            else None
        )
        seq_matches_logical_path = (
            int(drafter_seq0) == int(runtime_logical0)
            if drafter_seq0 is not None and runtime_logical0 is not None
            else None
        )
        seq_differs_from_compacted = (
            int(drafter_seq0) != int(runtime_compacted0)
            if drafter_seq0 is not None and runtime_compacted0 is not None
            else None
        )

        if seq_matches_runtime is True:
            num_seq_match_runtime += 1
        if max_query_len_matches_runtime is True:
            num_max_query_len_match_runtime += 1
        if seq_matches_logical_path is True:
            num_seq_matches_logical_path += 1
        if seq_differs_from_compacted is True:
            num_seq_differs_from_compacted += 1

        per_snapshot.append(
            {
                "step": step,
                "decoded_position_from_summary": annot.get("__position"),
                "accepted_len_from_summary": annot.get("accepted_len"),
                "root_token_from_summary": annot.get("root_token"),
                "draft_index": snap.get("draft_index"),
                "drafter_max_query_len": drafter_max_query_len,
                "drafter_query_start_loc": drafter_qsl,
                "drafter_seq_len0": drafter_seq0,
                "runtime_num_query_per_req": runtime_num_query_per_req,
                "runtime_query_start_loc": runtime.get("output_query_start_loc"),
                "runtime_seq_len0": runtime_seq0,
                "runtime_input_seq_minus_rejected0": runtime_logical0,
                "runtime_input_compacted_context_len0": runtime_compacted0,
                "drafter_seq_len_matches_runtime_seq_len": seq_matches_runtime,
                "drafter_max_query_len_matches_runtime_num_query_per_req": (
                    max_query_len_matches_runtime
                ),
                "drafter_seq_len_matches_logical_seq_minus_rejected": (
                    seq_matches_logical_path
                ),
                "drafter_seq_len_differs_from_compacted_context_len": (
                    seq_differs_from_compacted
                ),
            }
        )

    verdict: dict[str, Any] = {
        "sample_index": int(sample_index),
        "num_snapshots_analyzed": len(per_snapshot),
        "num_snapshots_where_drafter_seq_len_matches_runtime_seq_len": int(
            num_seq_match_runtime
        ),
        "num_snapshots_where_drafter_max_query_len_matches_runtime_num_query_per_req": int(
            num_max_query_len_match_runtime
        ),
        "num_snapshots_where_drafter_seq_len_matches_logical_seq_minus_rejected": int(
            num_seq_matches_logical_path
        ),
        "num_snapshots_where_drafter_seq_len_differs_from_compacted_context_len": int(
            num_seq_differs_from_compacted
        ),
    }
    if (
        per_snapshot
        and num_seq_match_runtime == len(per_snapshot)
        and num_seq_matches_logical_path == len(per_snapshot)
        and num_seq_differs_from_compacted == len(per_snapshot)
    ):
        verdict["interpretation_hint"] = (
            "The drafter first-pass path already receives the long logical "
            "`seq_lens` (`seq_lens - rejected`) instead of the compact accepted "
            "prefix length. That localizes the bug upstream of the drafter "
            "builder, in the metadata handed to the drafter first pass."
        )
    else:
        verdict["interpretation_hint"] = (
            "The direct drafter first-pass snapshots do not cleanly match the "
            "runtime bundle, so there may still be a substitution or probe-gap "
            "between first-pass preparation and drafter metadata construction."
        )

    return {
        "sample_index": int(sample_index),
        "captured_steps": drafter_probe.get("captured_steps"),
        "num_snapshots_captured": drafter_probe.get("num_snapshots_captured"),
        "per_snapshot": per_snapshot,
        "summary": {
            "num_snapshots_analyzed": len(per_snapshot),
            "num_snapshots_where_drafter_seq_len_matches_runtime_seq_len": int(
                num_seq_match_runtime
            ),
            "num_snapshots_where_drafter_max_query_len_matches_runtime_num_query_per_req": int(
                num_max_query_len_match_runtime
            ),
            "num_snapshots_where_drafter_seq_len_matches_logical_seq_minus_rejected": int(
                num_seq_matches_logical_path
            ),
            "num_snapshots_where_drafter_seq_len_differs_from_compacted_context_len": int(
                num_seq_differs_from_compacted
            ),
        },
        "verdict": verdict,
    }


def build_test_ad_runtime_vs_forward_metadata_report(
    runtime_bundle: dict[str, Any],
    vllm_tree_q_capture: dict[str, Any],
    vllm_tree_summary: dict[str, Any] | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    """Test AD -- compare first-pass runtime metadata with live forward metadata.

    Test AB/AC validate the DFlash first-pass runtime bundle in isolation.
    Test Q captures the forward-context metadata actually consumed by the live
    layer-0 draft forward. If these disagree at the same tree step, the bug is
    likely in the handoff into forward_context / attention-metadata building
    rather than in first-pass buffer materialization itself.
    """

    def _iter_bundles(payload: dict[str, Any]) -> list[dict[str, Any]]:
        bundles = payload.get("bundles")
        if isinstance(bundles, list):
            return [b for b in bundles if isinstance(b, dict)]
        if isinstance(payload.get("context_positions"), list):
            return [payload]
        return []

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for s in summary.get("samples") or []:
            try:
                if int(s.get("sample_index", -1)) != int(sample_idx):
                    continue
            except Exception:
                continue
            steps_sorted = sorted(
                [e for e in (s.get("steps") or []) if isinstance(e, dict)],
                key=lambda e: int(e.get("step", 0)),
            )
            cumulative = 0
            for e in steps_sorted:
                enriched = dict(e)
                enriched["__position"] = cumulative
                out[int(e.get("step", -1))] = enriched
                try:
                    cumulative += int(e.get("accepted_len", 0)) + 1
                except Exception:
                    cumulative += 1
            break
        return out

    bundles = _iter_bundles(runtime_bundle)
    q_steps = [
        e for e in (vllm_tree_q_capture.get("per_step") or []) if isinstance(e, dict)
    ]
    if not bundles:
        return {"skipped": True, "reason": "missing_runtime_bundles"}
    if not q_steps:
        return {"skipped": True, "reason": "missing_test_q_per_step_capture"}

    per_iter_annot = (
        _per_iter_positions(vllm_tree_summary, sample_index)
        if isinstance(vllm_tree_summary, dict)
        else {}
    )
    bundle_by_step = {int(b.get("step", -1)): b for b in bundles}
    q_d0_row = int(vllm_tree_q_capture.get("d0_row") or 1)
    q_by_outer_step: dict[int, list[dict[str, Any]]] = {}
    for e in q_steps:
        try:
            outer_step = int(e.get("tree_propose_step", e.get("step", -1)))
        except Exception:
            outer_step = int(e.get("step", -1))
        q_by_outer_step.setdefault(outer_step, []).append(e)
    matched_steps = sorted(set(bundle_by_step).intersection(q_by_outer_step))

    per_step: list[dict[str, Any]] = []
    num_live_visible_len_mismatch = 0
    num_live_query_len_mismatch = 0
    num_live_query_pos_mismatch = 0
    num_live_seq_lens_missing = 0
    for step in matched_steps:
        bundle = bundle_by_step[step]
        annot = per_iter_annot.get(step) or {}

        runtime_seq_lens = bundle.get("seq_lens")
        runtime_num_query_per_req = bundle.get("num_query_per_req")
        runtime_query_input_ids = bundle.get("query_input_ids")
        runtime_query_positions = bundle.get("query_positions")
        runtime_context_positions = bundle.get("context_positions")
        runtime_context_slot_mapping = bundle.get("context_slot_mapping")
        runtime_num_rejected = bundle.get("num_rejected_tokens")

        runtime_visible_context_len = None
        if (
            isinstance(runtime_seq_lens, list)
            and runtime_seq_lens
            and isinstance(runtime_num_query_per_req, int)
        ):
            try:
                runtime_visible_context_len = (
                    int(runtime_seq_lens[0]) - int(runtime_num_query_per_req)
                )
            except Exception:
                runtime_visible_context_len = None

        runtime_expected_context_len = None
        if (
            isinstance(runtime_context_positions, list)
            and isinstance(runtime_num_rejected, list)
            and runtime_num_rejected
        ):
            try:
                runtime_expected_context_len = len(runtime_context_positions) - int(
                    runtime_num_rejected[0]
                )
            except Exception:
                runtime_expected_context_len = None

        runtime_first_pad_idx = None
        if isinstance(runtime_context_slot_mapping, list):
            for i, x in enumerate(runtime_context_slot_mapping):
                try:
                    if int(x) == -1:
                        runtime_first_pad_idx = i
                        break
                except Exception:
                    continue

        runtime_query_row = None
        runtime_query_position = None
        runtime_query_input_id = None
        try:
            if isinstance(runtime_num_query_per_req, int) and runtime_num_query_per_req > q_d0_row:
                runtime_query_row = int(q_d0_row)
        except Exception:
            runtime_query_row = None
        if (
            runtime_query_row is not None
            and isinstance(runtime_query_positions, list)
            and len(runtime_query_positions) > runtime_query_row
        ):
            try:
                runtime_query_position = int(runtime_query_positions[runtime_query_row])
            except Exception:
                runtime_query_position = None
        if (
            runtime_query_row is not None
            and isinstance(runtime_query_input_ids, list)
            and len(runtime_query_input_ids) > runtime_query_row
        ):
            try:
                runtime_query_input_id = int(runtime_query_input_ids[runtime_query_row])
            except Exception:
                runtime_query_input_id = None

        candidates = q_by_outer_step.get(step) or []

        def _candidate_score(entry: dict[str, Any]) -> tuple[int, int, int]:
            actual_meta = entry.get("actual_meta") or {}
            score = 0
            if runtime_query_input_id is not None:
                try:
                    if int(actual_meta.get("query_input_id")) == int(runtime_query_input_id):
                        score += 4
                except Exception:
                    pass
            if runtime_query_position is not None:
                try:
                    if int(actual_meta.get("query_position")) == int(runtime_query_position):
                        score += 4
                except Exception:
                    pass
            if runtime_num_query_per_req is not None:
                try:
                    fm = entry.get("forward_attn_meta") or {}
                    if int(fm.get("max_query_len")) == int(runtime_num_query_per_req):
                        score += 1
                except Exception:
                    pass
            # Prefer the earliest forward call within the same outer step when
            # scores tie (usually the unconditioned initial draft forward).
            try:
                forward_call_idx = int(entry.get("step", 1 << 30))
            except Exception:
                forward_call_idx = 1 << 30
            return (score, -forward_call_idx, -len(candidates))

        q_entry = max(candidates, key=_candidate_score) if candidates else {}
        forward_meta = q_entry.get("forward_attn_meta") or {}
        actual_meta = q_entry.get("actual_meta") or {}
        live_seq_lens = forward_meta.get("seq_lens")
        live_max_query_len = forward_meta.get("max_query_len")
        live_visible_context_len = None
        if (
            isinstance(live_seq_lens, list)
            and live_seq_lens
            and live_max_query_len is not None
        ):
            try:
                live_visible_context_len = int(live_seq_lens[0]) - int(
                    live_max_query_len
                )
            except Exception:
                live_visible_context_len = None
        else:
            num_live_seq_lens_missing += 1

        live_query_row = actual_meta.get("local_row")
        live_query_position = actual_meta.get("query_position")
        selected_forward_call_kind = "unknown"
        try:
            actual_qid = actual_meta.get("query_input_id")
            actual_qpos = actual_meta.get("query_position")
            if (
                runtime_query_input_id is not None
                and actual_qid is not None
                and int(actual_qid) == int(runtime_query_input_id)
                and runtime_query_position is not None
                and actual_qpos is not None
                and int(actual_qpos) == int(runtime_query_position)
            ):
                selected_forward_call_kind = "runtime_matching_forward"
            elif (
                runtime_query_position is not None
                and actual_qpos is not None
                and int(actual_qpos) == int(runtime_query_position)
            ):
                selected_forward_call_kind = "same_position_nonruntime_input"
            else:
                selected_forward_call_kind = "nonruntime_forward_candidate"
        except Exception:
            selected_forward_call_kind = "unknown"

        live_visible_len_matches_runtime = (
            int(live_visible_context_len) == int(runtime_visible_context_len)
            if live_visible_context_len is not None
            and runtime_visible_context_len is not None
            else None
        )
        live_query_len_matches_runtime = (
            int(live_max_query_len) == int(runtime_num_query_per_req)
            if live_max_query_len is not None and runtime_num_query_per_req is not None
            else None
        )
        live_query_pos_matches_runtime = (
            int(live_query_position) == int(runtime_query_position)
            if live_query_position is not None and runtime_query_position is not None
            else None
        )

        if live_visible_len_matches_runtime is False:
            num_live_visible_len_mismatch += 1
        if live_query_len_matches_runtime is False:
            num_live_query_len_mismatch += 1
        if live_query_pos_matches_runtime is False:
            num_live_query_pos_mismatch += 1

        per_step.append(
            {
                "step": int(step),
                "decoded_position_from_summary": annot.get("__position"),
                "accepted_len_from_summary": annot.get("accepted_len"),
                "root_token_from_summary": annot.get("root_token"),
                "runtime_query_row": runtime_query_row,
                "runtime_query_input_id": runtime_query_input_id,
                "runtime_query_position": runtime_query_position,
                "runtime_visible_context_len": runtime_visible_context_len,
                "runtime_expected_accepted_context_len": runtime_expected_context_len,
                "runtime_num_query_per_req": runtime_num_query_per_req,
                "runtime_first_pad_idx": runtime_first_pad_idx,
                "selected_q_capture_step": q_entry.get("step"),
                "selected_tree_propose_step": q_entry.get("tree_propose_step"),
                "selected_forward_call_kind": selected_forward_call_kind,
                "live_forward_seq_lens": live_seq_lens,
                "live_forward_max_query_len": live_max_query_len,
                "live_forward_visible_context_len": live_visible_context_len,
                "live_query_row": live_query_row,
                "live_query_input_id": actual_meta.get("query_input_id"),
                "live_query_pos0": live_query_position,
                "live_visible_len_matches_runtime": live_visible_len_matches_runtime,
                "live_query_len_matches_runtime": live_query_len_matches_runtime,
                "live_query_pos_matches_runtime": live_query_pos_matches_runtime,
            }
        )

    verdict: dict[str, Any] = {
        "sample_index": int(sample_index),
        "num_matched_steps": len(per_step),
        "num_steps_with_live_visible_len_mismatch": int(
            num_live_visible_len_mismatch
        ),
        "num_steps_with_live_query_len_mismatch": int(
            num_live_query_len_mismatch
        ),
        "num_steps_with_live_query_pos_mismatch": int(
            num_live_query_pos_mismatch
        ),
        "num_steps_missing_live_seq_lens": int(num_live_seq_lens_missing),
    }
    if not per_step:
        verdict["interpretation_hint"] = (
            "No steps were shared between the runtime-bundle capture and Test Q "
            "capture. Ensure both probes use the same step schedule."
        )
    elif (
        num_live_visible_len_mismatch
        or num_live_query_len_mismatch
        or num_live_query_pos_mismatch
    ):
        verdict["interpretation_hint"] = (
            "At one or more shared tree steps, the live forward-attention "
            "metadata disagrees with the first-pass runtime bundle. This points "
            "to the handoff into forward_context / attention-metadata building "
            "rather than to first-pass buffer materialization itself."
        )
    else:
        verdict["interpretation_hint"] = (
            "The live forward-attention metadata matches the first-pass runtime "
            "bundle at all shared tree steps. The remaining bug is likely later "
            "inside the attention read path or probe interpretation."
        )

    return {
        "sample_index": int(sample_index),
        "capture_steps_requested": runtime_bundle.get("capture_steps_requested"),
        "runtime_captured_steps": runtime_bundle.get("captured_steps"),
        "q_capture_steps": [
            int(e.get("step", -1)) for e in q_steps if e.get("step") is not None
        ],
        "matched_steps": [int(s) for s in matched_steps],
        "per_step": per_step,
        "summary": {
            "num_matched_steps": len(per_step),
            "num_steps_with_live_visible_len_mismatch": int(
                num_live_visible_len_mismatch
            ),
            "num_steps_with_live_query_len_mismatch": int(
                num_live_query_len_mismatch
            ),
            "num_steps_with_live_query_pos_mismatch": int(
                num_live_query_pos_mismatch
            ),
            "num_steps_missing_live_seq_lens": int(num_live_seq_lens_missing),
        },
        "verdict": verdict,
    }


def build_test_ac_first_pass_metadata_consistency_report(
    runtime_bundle: dict[str, Any],
    vllm_tree_summary: dict[str, Any] | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    """Test AC -- first-pass metadata/buffer consistency audit.

    Checks whether the post-``set_inputs_first_pass`` metadata (notably
    ``seq_lens``) is consistent with the materialized context/query buffers
    that the DFlash tree path actually uses. When a tree summary is provided,
    bundles are annotated with decoded positions so later failing iterations
    can be inspected directly.
    """

    def _iter_bundles(payload: dict[str, Any]) -> list[dict[str, Any]]:
        bundles = payload.get("bundles")
        if isinstance(bundles, list):
            return [b for b in bundles if isinstance(b, dict)]
        if isinstance(payload.get("context_positions"), list):
            return [payload]
        return []

    def _per_iter_positions(
        summary: dict[str, Any], sample_idx: int
    ) -> dict[int, dict[str, Any]]:
        sample_key = f"sample_{int(sample_idx):03d}"
        sample = summary.get(sample_key) or {}
        steps = sample.get("per_step") or []
        if not isinstance(steps, list):
            return {}
        out: dict[int, dict[str, Any]] = {}
        cumulative = 0
        for e in sorted(
            [x for x in steps if isinstance(x, dict)],
            key=lambda x: int(x.get("step", -1)),
        ):
            enriched = dict(e)
            enriched["__position"] = cumulative
            out[int(e.get("step", -1))] = enriched
            try:
                cumulative += int(e.get("accepted_len", 0)) + 1
            except Exception:
                cumulative += 1
        return out

    def _front_valid_len(xs: list[Any]) -> int:
        n = 0
        for x in xs:
            try:
                if int(x) == -1:
                    break
            except Exception:
                break
            n += 1
        return n

    bundles = _iter_bundles(runtime_bundle)
    if not bundles:
        return {
            "skipped": True,
            "reason": "missing_context_buffers_in_runtime_bundle",
        }
    per_iter_annot = (
        _per_iter_positions(vllm_tree_summary, sample_index)
        if isinstance(vllm_tree_summary, dict)
        else {}
    )

    per_bundle: list[dict[str, Any]] = []
    num_visible_vs_front_valid_mismatch = 0
    num_visible_vs_expected_mismatch = 0
    num_query_pos0_mismatch = 0
    num_query_id0_mismatch = 0
    for bundle in sorted(bundles, key=lambda b: int(b.get("step", -1))):
        context_positions = bundle.get("context_positions")
        context_slot_mapping = bundle.get("context_slot_mapping")
        seq_lens = bundle.get("seq_lens")
        num_query_per_req = bundle.get("num_query_per_req")
        num_rejected_tokens = bundle.get("num_rejected_tokens")
        query_positions = bundle.get("query_positions")
        query_input_ids = bundle.get("query_input_ids")
        next_token_ids = bundle.get("next_token_ids")
        if not isinstance(context_positions, list) or not isinstance(
            context_slot_mapping, list
        ):
            continue
        if not isinstance(seq_lens, list) or not seq_lens:
            continue
        if not isinstance(num_query_per_req, int):
            continue
        step = int(bundle.get("step", -1))
        iter_annot = per_iter_annot.get(step) or {}

        visible_context_len: int | None = None
        try:
            visible_context_len = int(seq_lens[0]) - int(num_query_per_req)
        except Exception:
            visible_context_len = None

        front_valid_context_len = _front_valid_len(context_slot_mapping)
        total_valid_context_rows = 0
        for x in context_slot_mapping:
            try:
                if int(x) != -1:
                    total_valid_context_rows += 1
            except Exception:
                continue

        expected_accepted_context_len: int | None = None
        try:
            if isinstance(num_rejected_tokens, list) and num_rejected_tokens:
                expected_accepted_context_len = len(context_positions) - int(
                    num_rejected_tokens[0]
                )
        except Exception:
            expected_accepted_context_len = None

        query_pos0_matches_context = None
        if (
            isinstance(query_positions, list)
            and query_positions
            and isinstance(visible_context_len, int)
            and visible_context_len > 0
            and len(context_positions) >= visible_context_len
        ):
            try:
                query_pos0_matches_context = (
                    int(query_positions[0])
                    == int(context_positions[visible_context_len - 1]) + 1
                )
            except Exception:
                query_pos0_matches_context = None

        query_id0_matches_bonus = None
        if (
            isinstance(query_input_ids, list)
            and query_input_ids
            and isinstance(next_token_ids, list)
            and next_token_ids
        ):
            try:
                query_id0_matches_bonus = (
                    int(query_input_ids[0]) == int(next_token_ids[0])
                )
            except Exception:
                query_id0_matches_bonus = None

        first_pad_idx = None
        for i, x in enumerate(context_slot_mapping):
            try:
                if int(x) == -1:
                    first_pad_idx = i
                    break
            except Exception:
                continue

        if (
            isinstance(visible_context_len, int)
            and visible_context_len != front_valid_context_len
        ):
            num_visible_vs_front_valid_mismatch += 1
        if (
            expected_accepted_context_len is not None
            and visible_context_len != expected_accepted_context_len
        ):
            num_visible_vs_expected_mismatch += 1
        if query_pos0_matches_context is False:
            num_query_pos0_mismatch += 1
        if query_id0_matches_bonus is False:
            num_query_id0_mismatch += 1

        per_bundle.append(
            {
                "step": step,
                "decoded_position_from_summary": iter_annot.get("__position"),
                "accepted_len_from_summary": iter_annot.get("accepted_len"),
                "root_token_from_summary": iter_annot.get("root_token"),
                "visible_context_len": visible_context_len,
                "front_valid_context_len": front_valid_context_len,
                "total_valid_context_rows": total_valid_context_rows,
                "expected_accepted_context_len": expected_accepted_context_len,
                "first_pad_idx": first_pad_idx,
                "num_query_per_req": int(num_query_per_req),
                "query_pos0_matches_context": query_pos0_matches_context,
                "query_id0_matches_bonus": query_id0_matches_bonus,
                "query_positions_head": (
                    query_positions[:8]
                    if isinstance(query_positions, list)
                    else None
                ),
                "query_input_ids_head": (
                    query_input_ids[:8]
                    if isinstance(query_input_ids, list)
                    else None
                ),
                "next_token_ids_head": (
                    next_token_ids[:8]
                    if isinstance(next_token_ids, list)
                    else None
                ),
            }
        )

    verdict: dict[str, Any] = {
        "sample_index": int(sample_index),
        "num_bundles_analyzed": len(per_bundle),
        "num_bundles_with_visible_vs_front_valid_mismatch": int(
            num_visible_vs_front_valid_mismatch
        ),
        "num_bundles_with_visible_vs_expected_mismatch": int(
            num_visible_vs_expected_mismatch
        ),
        "num_bundles_with_query_pos0_mismatch": int(num_query_pos0_mismatch),
        "num_bundles_with_query_id0_mismatch": int(num_query_id0_mismatch),
    }
    if not per_bundle:
        verdict["interpretation_hint"] = (
            "No runtime bundles were analyzable; confirm runtime capture was "
            "installed before the tree run."
        )
    elif (
        num_visible_vs_front_valid_mismatch
        or num_visible_vs_expected_mismatch
        or num_query_pos0_mismatch
        or num_query_id0_mismatch
    ):
        verdict["interpretation_hint"] = (
            "One or more captured tree iterations already show metadata/buffer "
            "inconsistency immediately after first-pass setup. This supports a "
            "later-step post-rejection visible-prefix bug."
        )
    else:
        verdict["interpretation_hint"] = (
            "All captured tree iterations show metadata/buffer consistency after "
            "first-pass setup. The remaining bug is likely later in execution."
        )

    return {
        "sample_index": int(sample_index),
        "capture_steps_requested": runtime_bundle.get("capture_steps_requested"),
        "captured_steps": runtime_bundle.get("captured_steps"),
        "per_bundle": per_bundle,
        "summary": {
            "num_bundles_analyzed": len(per_bundle),
            "num_bundles_with_visible_vs_front_valid_mismatch": int(
                num_visible_vs_front_valid_mismatch
            ),
            "num_bundles_with_visible_vs_expected_mismatch": int(
                num_visible_vs_expected_mismatch
            ),
            "num_bundles_with_query_pos0_mismatch": int(
                num_query_pos0_mismatch
            ),
            "num_bundles_with_query_id0_mismatch": int(num_query_id0_mismatch),
        },
        "verdict": verdict,
    }
