#!/usr/bin/env python3
"""Compare Step-3.7 target outputs between SpecForge/SGLang and HF.

This is intentionally a target-model probe only.  It answers whether the HF
reference path used by draft-quality diagnostics is numerically aligned with
the SpecForge training anchor, which uses SGLang's DFlash hidden-state capture.

Typical use:

  1. Run the SGLang anchor with torchrun.
  2. Run the HF payload in a single process.
  3. Compare the two saved payloads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


DEFAULT_MODEL = "/mnt/lanxiangh/models/Step-3.7-Flash"
DEFAULT_LAYER_IDS = (1, 9, 17, 25, 33, 41)
DEFAULT_PROMPT = (
    "Implement Python function `truncate_number(number: float) -> float`.\n"
    "Return the decimal part of the number.\n"
    "Example: truncate_number(3.5) == 0.5"
)


def _add_repo_paths(specforge_root: str, vllm_root: str) -> None:
    for path in (vllm_root, specforge_root):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def _install_optional_specforge_dependency_stubs() -> None:
    """Avoid importing optional USP/yunchang dependencies for target-only probes.

    SpecForge's package __init__ eagerly imports draft modules, including USP
    attention code backed by yunchang.  The target parity probe only uses the
    SGLang DFlash target wrapper, so these optional imports are not exercised.
    Stubbing them keeps the probe focused and avoids requiring the full training
    dependency set in the SGLang profiling venv.
    """
    if "yunchang" not in sys.modules:
        yunchang_mod = types.ModuleType("yunchang")
        yunchang_mod.EXTRACT_FUNC_DICT = {}
        sys.modules["yunchang"] = yunchang_mod

    if "yunchang.globals" not in sys.modules:
        globals_mod = types.ModuleType("yunchang.globals")

        class _ProcessGroup:
            ULYSSES_PG = None
            RING_PG = None

        def _set_seq_parallel_pg(*_args: Any, **_kwargs: Any) -> None:
            return None

        globals_mod.PROCESS_GROUP = _ProcessGroup
        globals_mod.set_seq_parallel_pg = _set_seq_parallel_pg
        sys.modules["yunchang.globals"] = globals_mod

    if "yunchang.comm" not in sys.modules:
        comm_mod = types.ModuleType("yunchang.comm")

        class _SeqAllToAll4D:
            @staticmethod
            def apply(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError(
                    "SeqAllToAll4D is unavailable in this target-only probe"
                )

        comm_mod.SeqAllToAll4D = _SeqAllToAll4D
        sys.modules["yunchang.comm"] = comm_mod

    if "yunchang.kernels" not in sys.modules:
        kernels_mod = types.ModuleType("yunchang.kernels")

        class _AttnType:
            FA = "FA"
            SPARSE_SAGE = "SPARSE_SAGE"

        def _select_flash_attn_impl(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(
                "yunchang kernels are unavailable in this target-only probe"
            )

        kernels_mod.AttnType = _AttnType
        kernels_mod.select_flash_attn_impl = _select_flash_attn_impl
        sys.modules["yunchang.kernels"] = kernels_mod


def _is_step3p7_model_path(model_path: str) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("model_type") == "step3p7"


def _tokenizer_kwargs(model_path: str) -> dict[str, Any]:
    if _is_step3p7_model_path(model_path):
        return {"fix_mistral_regex": True}
    return {}


def _apply_step3p7_specforge_template(prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def _encode_prompt(args: argparse.Namespace) -> tuple[str, list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        **_tokenizer_kwargs(args.model),
    )
    prompt = _load_prompt(args)
    if args.prompt_is_templated:
        templated = prompt
    elif _is_step3p7_model_path(args.model):
        templated = _apply_step3p7_specforge_template(prompt)
    else:
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    input_ids = tokenizer.encode(templated, add_special_tokens=False)
    return templated, [int(x) for x in input_ids]


def _tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    x = t.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "dtype": str(t.dtype),
        "mean": float(x.mean().item()) if x.numel() else 0.0,
        "std": float(x.std(unbiased=False).item()) if x.numel() else 0.0,
        "min": float(x.min().item()) if x.numel() else 0.0,
        "max": float(x.max().item()) if x.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(x).item()) if x.numel() else 0.0,
    }


def _topk(logits: torch.Tensor, k: int) -> dict[str, list[Any]]:
    lp = torch.log_softmax(logits.detach().float().cpu(), dim=-1)
    top_lp, top_tok = lp.topk(min(k, lp.shape[-1]), dim=-1)
    return {
        "tokens": [int(x) for x in top_tok.tolist()],
        "logprobs": [float(x) for x in top_lp.tolist()],
    }


def _payload_paths(output_dir: Path, backend: str) -> tuple[Path, Path]:
    return output_dir / f"{backend}_target_payload.pt", output_dir / f"{backend}_target_summary.json"


def _moe_debug_paths(output_dir: Path, backend: str, layer_id: int) -> tuple[Path, Path]:
    stem = f"{backend}_moe_debug_layer{layer_id}"
    return output_dir / f"{stem}.pt", output_dir / f"{stem}_summary.json"


def _prefill_debug_paths(
    output_dir: Path, backend: str, layer_id: int
) -> tuple[Path, Path]:
    stem = f"{backend}_prefill_debug_layer{layer_id}"
    return output_dir / f"{stem}.pt", output_dir / f"{stem}_summary.json"


def _attn_debug_paths(
    output_dir: Path, backend: str, layer_id: int
) -> tuple[Path, Path]:
    stem = f"{backend}_attn_debug_layer{layer_id}"
    return output_dir / f"{stem}.pt", output_dir / f"{stem}_summary.json"


def _phase_debug_path(output_dir: Path) -> Path:
    return output_dir / "vllm_phase_debug.pt"


def _phase_report_path(output_dir: Path) -> Path:
    return output_dir / "vllm_phase_debug_report.json"


def _debug_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().cpu()


def _debug_last_token(t: torch.Tensor, prompt_len: int) -> torch.Tensor:
    x = t.detach().float().cpu()
    if x.ndim >= 3 and x.shape[0] == 1 and x.shape[1] >= prompt_len:
        return x[0, prompt_len - 1].contiguous()
    if x.ndim >= 2 and x.shape[0] > prompt_len:
        return x[-1].contiguous()
    if x.ndim >= 2 and x.shape[0] == prompt_len:
        return x[prompt_len - 1].contiguous()
    return x.contiguous()


def _write_moe_debug_payload(
    *,
    args: argparse.Namespace,
    backend: str,
    input_ids: list[int],
    layer_id: int,
    captures: dict[str, Any],
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path, summary_path = _moe_debug_paths(output_dir, backend, layer_id)
    prompt_len = len(input_ids)
    tensors = {
        name: _debug_tensor(value)
        for name, value in captures.items()
        if isinstance(value, torch.Tensor)
    }
    last_token = {
        name: _debug_last_token(value, prompt_len)
        for name, value in tensors.items()
    }
    payload = {
        "backend": backend,
        "model": args.model,
        "input_ids": input_ids,
        "layer_id": layer_id,
        "tensors": tensors,
        "last_token": last_token,
    }
    torch.save(payload, payload_path)
    summary = {
        "backend": backend,
        "payload_path": str(payload_path),
        "model": args.model,
        "num_prompt_tokens": len(input_ids),
        "layer_id": layer_id,
        "tensors": {
            name: _tensor_stats(value)
            for name, value in tensors.items()
        },
        "last_token": {
            name: _tensor_stats(value)
            for name, value in last_token.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {backend} MoE debug payload to {payload_path}")
    print(f"Wrote {backend} MoE debug summary to {summary_path}")


def _write_prefill_debug_payload(
    *,
    args: argparse.Namespace,
    backend: str,
    input_ids: list[int],
    layer_id: int,
    captures: dict[str, torch.Tensor],
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path, summary_path = _prefill_debug_paths(output_dir, backend, layer_id)
    payload = {
        "backend": backend,
        "model": args.model,
        "layer_id": layer_id,
        "input_ids": input_ids,
        "captures": {k: _debug_tensor(v) for k, v in captures.items()},
    }
    torch.save(payload, payload_path)
    summary = {
        "backend": backend,
        "layer_id": layer_id,
        "num_prompt_tokens": len(input_ids),
        "keys": sorted(payload["captures"].keys()),
        "shapes": {
            k: list(v.shape) for k, v in payload["captures"].items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {backend} prefill debug payload to {payload_path}")
    print(f"Wrote {backend} prefill debug summary to {summary_path}")


def _write_attn_debug_payload(
    *,
    args: argparse.Namespace,
    backend: str,
    input_ids: list[int],
    layer_id: int,
    captures: dict[str, torch.Tensor],
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path, summary_path = _attn_debug_paths(output_dir, backend, layer_id)
    payload = {
        "backend": backend,
        "model": args.model,
        "layer_id": layer_id,
        "input_ids": input_ids,
        "captures": {k: _debug_tensor(v) for k, v in captures.items()},
    }
    torch.save(payload, payload_path)
    summary = {
        "backend": backend,
        "layer_id": layer_id,
        "num_prompt_tokens": len(input_ids),
        "keys": sorted(payload["captures"].keys()),
        "shapes": {k: list(v.shape) for k, v in payload["captures"].items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {backend} attention debug payload to {payload_path}")
    print(f"Wrote {backend} attention debug summary to {summary_path}")


def _install_hf_attn_debug_hook(
    text_model: nn.Module,
    layer_id: int,
    tp_size: int,
) -> dict[str, torch.Tensor]:
    layer = _get_decoder_layer(text_model, layer_id)
    attn = layer.self_attn
    attn_module = sys.modules[attn.__class__.__module__]
    apply_rotary_pos_emb = getattr(attn_module, "apply_rotary_pos_emb")
    eager_attention_forward = getattr(attn_module, "eager_attention_forward")
    all_attention_functions = getattr(attn_module, "ALL_ATTENTION_FUNCTIONS")
    captures: dict[str, torch.Tensor] = {}

    def _capture_once(key: str, value: torch.Tensor) -> None:
        captures.setdefault(key, value.detach().cpu())

    def patched_attn_forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_value: Any = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        if position_ids is not None:
            _capture_once("position_ids", position_ids)

        q_proj = attn.q_proj(hidden_states)
        k_proj = attn.k_proj(hidden_states)
        v_proj = attn.v_proj(hidden_states)
        _capture_once("q_proj", q_proj)
        _capture_once("k_proj", k_proj)
        _capture_once("v_proj", v_proj)

        q_norm_by_head = attn.q_norm(q_proj.view(hidden_shape))
        k_norm_by_head = attn.k_norm(k_proj.view(hidden_shape))
        _capture_once("q_norm", q_norm_by_head.reshape(*input_shape, -1))
        _capture_once("k_norm", k_norm_by_head.reshape(*input_shape, -1))

        query_states = q_norm_by_head.transpose(1, 2)
        key_states = k_norm_by_head.transpose(1, 2)
        value_states = v_proj.view(hidden_shape).transpose(1, 2)
        gate_states = None
        if attn.use_head_wise_attn_gate:
            gate_states = attn.g_proj(hidden_states)
            _capture_once("gate", gate_states)
            _capture_once("gate_sigmoid", gate_states.sigmoid())

        cos, sin = attn.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        _capture_once(
            "q_rope",
            query_states.transpose(1, 2).reshape(*input_shape, -1),
        )
        _capture_once(
            "k_rope",
            key_states.transpose(1, 2).reshape(*input_shape, -1),
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, attn.layer_idx, cache_kwargs
            )

        attention_interface = eager_attention_forward
        if attn.config._attn_implementation != "eager":
            attention_interface = all_attention_functions[
                attn.config._attn_implementation
            ]
        attn_output, attn_weights = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            sliding_window=attn.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1)
        _capture_once("attn_core", attn_output)
        if attn.use_head_wise_attn_gate:
            assert gate_states is not None
            output = (
                attn_output.view(
                    *attn_output.shape[:-1],
                    attn.num_attention_heads,
                    attn.head_dim,
                )
                * gate_states.unsqueeze(-1).sigmoid()
            )
            attn_output = output.view(*attn_output.shape)
            _capture_once("attn_after_gate", attn_output)
        local_q_heads = attn.num_attention_heads // tp_size
        local_q_dim = local_q_heads * attn.head_dim
        _capture_once(
            "attn_o_proj_tp0",
            F.linear(attn_output[..., :local_q_dim], attn.o_proj.weight[:, :local_q_dim]),
        )
        attn_output = attn.o_proj(attn_output)
        _capture_once("attn_o_proj", attn_output)
        return attn_output, attn_weights

    attn.forward = patched_attn_forward
    return captures


def _install_hf_prefill_debug_hook(
    text_model: nn.Module,
    layer_id: int,
) -> dict[str, torch.Tensor]:
    layer = _get_decoder_layer(text_model, layer_id)
    captures: dict[str, torch.Tensor] = {}

    def patched_layer_forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        captures.setdefault("layer_input", residual.detach().cpu())
        hidden_states = layer.input_layernorm(hidden_states)
        captures.setdefault("input_norm", hidden_states.detach().cpu())
        attn_output, _ = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        captures.setdefault("attn_output", attn_output.detach().cpu())
        hidden_states = residual + attn_output
        captures.setdefault("attn_residual", hidden_states.detach().cpu())

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        captures.setdefault("post_attn_norm", hidden_states.detach().cpu())
        captures.setdefault("moe_input", hidden_states.detach().cpu())
        if layer.use_moe:
            share_output = layer.share_expert(hidden_states)
            moe_output = layer.moe(hidden_states)
            ffn_output = moe_output + share_output
        else:
            ffn_output = layer.mlp(hidden_states)
        if isinstance(ffn_output, tuple):
            hidden_states, _ = ffn_output
        else:
            hidden_states = ffn_output

        hidden_states = residual + hidden_states
        captures.setdefault("layer_output", hidden_states.detach().cpu())
        return hidden_states

    layer.forward = patched_layer_forward
    return captures


def _write_phase_debug_payload(
    *,
    args: argparse.Namespace,
    prompt_text: str,
    input_ids: list[int],
    captures: dict[str, Any],
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = _phase_debug_path(output_dir)
    payload = {
        "backend": "vllm",
        "model": args.model,
        "prompt_text": prompt_text,
        "input_ids": input_ids,
        "layer_ids": _parse_ints(args.layer_ids),
        "compute_events": captures.get("compute_events", []),
        "propose_events": captures.get("propose_events", []),
    }
    torch.save(payload, payload_path)

    def _json_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return value

    summary = {
        "payload_path": str(payload_path),
        "num_prompt_tokens": len(input_ids),
        "layer_ids": payload["layer_ids"],
        "compute_events": [
            {
                "call_index": event.get("call_index"),
                "hidden_shape": event.get("hidden_shape"),
                "logits_shape": event.get("logits_shape"),
            }
            for event in payload["compute_events"]
        ],
        "propose_events": [
            {
                "call_index": event.get("call_index"),
                "sampled_token_ids": _json_value(event.get("sampled_token_ids")),
                "aux_hidden_shapes": event.get("aux_hidden_shapes"),
            }
            for event in payload["propose_events"]
        ],
    }
    summary_path = output_dir / "vllm_phase_debug_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote vLLM phase debug payload to {payload_path}")
    print(f"Wrote vLLM phase debug summary to {summary_path}")


def _write_payload(
    *,
    args: argparse.Namespace,
    backend: str,
    prompt_text: str,
    input_ids: list[int],
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path, summary_path = _payload_paths(output_dir, backend)

    hidden_cpu = hidden_states.detach().float().cpu()
    logits_cpu = logits.detach().float().cpu()
    last_hidden = hidden_cpu[0, -1].contiguous()
    last_logits = logits_cpu[0, -1].contiguous()
    payload = {
        "backend": backend,
        "model": args.model,
        "prompt_text": prompt_text,
        "input_ids": input_ids,
        "layer_ids": _parse_ints(args.layer_ids),
        "hidden_states": hidden_cpu,
        "last_hidden": last_hidden,
        "last_logits": last_logits,
    }
    torch.save(payload, payload_path)

    summary = {
        "backend": backend,
        "payload_path": str(payload_path),
        "model": args.model,
        "num_prompt_tokens": len(input_ids),
        "prompt_text": prompt_text,
        "input_ids": input_ids,
        "layer_ids": _parse_ints(args.layer_ids),
        "hidden_states_stats": _tensor_stats(hidden_cpu),
        "last_hidden_stats": _tensor_stats(last_hidden),
        "last_logits_stats": _tensor_stats(last_logits),
        "last_logits_topk": _topk(last_logits, args.topk),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {backend} payload to {payload_path}")
    print(f"Wrote {backend} summary to {summary_path}")


def _run_sglang(args: argparse.Namespace) -> None:
    _add_repo_paths(args.specforge_root, args.vllm_root)
    _install_optional_specforge_dependency_stubs()
    from specforge.distributed import destroy_distributed, init_distributed
    from specforge.modeling.target.dflash_target_model import SGLangDFlashTargetModel

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    rank = dist.get_rank()
    try:
        prompt_text, input_ids = _encode_prompt(args)
        device = torch.device("cuda", torch.cuda.current_device())
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(ids)
        loss_mask = torch.ones_like(ids, dtype=torch.float32)

        target = SGLangDFlashTargetModel.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attention_backend=args.sglang_attention_backend,
            mem_fraction_static=args.sglang_mem_fraction_static,
            context_length=args.max_model_len,
            ep_size=args.sglang_ep_size,
            disable_custom_all_reduce=args.sglang_disable_custom_all_reduce,
        )
        _set_sglang_capture_layers(target, _parse_ints(args.layer_ids))
        out = target.generate_dflash_data(
            input_ids=ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            return_logits=True,
        )
        if rank == 0:
            _write_payload(
                args=args,
                backend="sglang",
                prompt_text=prompt_text,
                input_ids=input_ids,
                hidden_states=out.hidden_states,
                logits=out.logits,
            )
        dist.barrier()
    finally:
        try:
            destroy_distributed()
        except Exception as exc:
            if rank == 0:
                print(f"WARNING: destroy_distributed failed: {exc}")


def _set_sglang_capture_layers(target: Any, layer_ids: list[int]) -> None:
    """Set aux-hidden capture on the actual SGLang Step3.7 text module.

    SpecForge's DFlash wrapper currently checks for the older
    ``set_eagle3_layers_to_capture`` name, while the SGLang Step3.7/Step3p5
    reference exposes ``set_dflash_layers_to_capture`` on the nested text model.
    Try both names across the common wrapper locations.
    """
    target.capture_layer_ids = layer_ids
    root_model = target.model_runner.model
    candidates = [
        root_model,
        getattr(root_model, "language_model", None),
        getattr(getattr(root_model, "language_model", None), "model", None),
        getattr(root_model, "model", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        for setter_name in (
            "set_dflash_layers_to_capture",
            "set_eagle3_layers_to_capture",
        ):
            setter = getattr(candidate, setter_name, None)
            if setter is not None:
                setter(layer_ids)
                inner_model = getattr(candidate, "model", candidate)
                print(
                    "Set SGLang capture layers via "
                    f"{type(candidate).__name__}.{setter_name}: "
                    f"{getattr(inner_model, 'layers_to_capture', layer_ids)}"
                )
                return
    if _install_step3p5_capture_patch(root_model, layer_ids):
        return
    candidate_types = [type(c).__name__ for c in candidates if c is not None]
    raise RuntimeError(
        "Target model does not expose a DFlash/EAGLE capture setter. "
        f"Checked candidates: {candidate_types}"
    )


def _install_step3p5_capture_patch(root_model: nn.Module, layer_ids: list[int]) -> bool:
    """Patch older Step3p5 SGLang commits that lack DFlash capture helpers."""
    language_model = getattr(root_model, "language_model", None)
    text_model = getattr(language_model, "model", None)
    if language_model is None or text_model is None:
        return False
    if not hasattr(text_model, "layers") or not hasattr(language_model, "logits_processor"):
        return False

    import types as _types

    text_model.layers_to_capture = [int(x) for x in layer_ids]

    def _patched_text_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Any = None,
    ) -> Any:
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        capture_set = set(int(x) for x in getattr(self, "layers_to_capture", []))
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
            if int(i) in capture_set:
                layer_out = hidden_states if residual is None else hidden_states + residual
                aux_hidden_states.append(layer_out)

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        hidden_states_before_norm = None
        if hidden_states.shape[0] > 0:
            hidden_states_before_norm = (
                hidden_states if residual is None else hidden_states + residual
            )
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, hidden_states_before_norm, aux_hidden_states

    def _patched_lm_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Any = None,
    ) -> torch.Tensor:
        model_out = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if isinstance(model_out, tuple) and len(model_out) == 3:
            hidden_states, hidden_states_before_norm, aux_hidden_states = model_out
        else:
            hidden_states, hidden_states_before_norm = model_out
            aux_hidden_states = None

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states=aux_hidden_states,
                hidden_states_before_norm=hidden_states_before_norm,
            )
        return hidden_states

    text_model.forward = _types.MethodType(_patched_text_forward, text_model)
    language_model.forward = _types.MethodType(_patched_lm_forward, language_model)
    print(
        "Installed Step3p5 DFlash capture monkey-patch with layers: "
        f"{text_model.layers_to_capture}"
    )
    return True


def _target_text_model(target: nn.Module) -> nn.Module:
    if hasattr(target, "language_model"):
        return target.language_model
    if hasattr(target, "model") and hasattr(target.model, "language_model"):
        return target.model.language_model
    if hasattr(target, "model"):
        return target.model
    return target


def _target_lm_head(target: nn.Module) -> nn.Module:
    if hasattr(target, "lm_head"):
        return target.lm_head
    lm_head = target.get_output_embeddings()
    if lm_head is None:
        raise AttributeError("Unable to locate target lm_head")
    return lm_head


def _module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _get_decoder_layer(text_model: nn.Module, layer_id: int) -> nn.Module:
    layers = getattr(text_model, "layers", None)
    if layers is None:
        raise AttributeError("Unable to locate decoder layers on text model")
    return layers[layer_id]


def _install_hf_moe_debug_hook(
    text_model: nn.Module,
    layer_id: int,
) -> dict[str, torch.Tensor]:
    layer = _get_decoder_layer(text_model, layer_id)
    if not hasattr(layer, "moe") or not hasattr(layer, "share_expert"):
        raise AttributeError(f"Layer {layer_id} is not a Step3p7 MoE layer")

    captures: dict[str, torch.Tensor] = {}
    moe = layer.moe
    share_expert = layer.share_expert
    orig_share_forward = share_expert.forward

    def patched_share_forward(hidden_states, *args, **kwargs):
        captures["mlp_input"] = hidden_states.detach().cpu()
        output = orig_share_forward(hidden_states, *args, **kwargs)
        captures["shared_output"] = output.detach().cpu()
        return output

    def patched_moe_forward(hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)
        if moe.need_fp32_gate:
            router_logits = torch.matmul(
                flat_hidden.to(torch.float32),
                moe.gate.weight.t().to(torch.float32),
            )
        else:
            router_logits = moe.gate(flat_hidden)

        if moe.custom_routing_function:
            routing_weights, selected_experts = moe.custom_routing_function(
                router_logits, moe.top_k, renormalize=True
            )
        else:
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(
                routing_weights, moe.top_k, dim=-1
            )

        routing_weights = routing_weights * moe.routed_scaling_factor
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=flat_hidden.dtype,
            device=flat_hidden.device,
        )
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=moe.num_experts
        ).permute(2, 1, 0)

        for expert_idx in range(moe.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = flat_hidden[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                moe.get_expert_output(current_state, expert_idx)
                * routing_weights[top_x, idx, None]
            )
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(flat_hidden.dtype)
            )

        captures["router_logits"] = router_logits.detach().cpu()
        captures["topk_ids"] = selected_experts.detach().cpu()
        captures["topk_weights"] = routing_weights.detach().cpu()
        captures["routed_output"] = final_hidden_states.detach().cpu()
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    share_expert.forward = patched_share_forward
    moe.forward = patched_moe_forward
    return captures


@torch.inference_mode()
def _run_hf(args: argparse.Namespace) -> None:
    _add_repo_paths(args.specforge_root, args.vllm_root)
    prompt_text, input_ids = _encode_prompt(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": args.hf_attn_implementation,
    }
    if args.hf_device_map_auto:
        load_kwargs["device_map"] = "balanced"
    target = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).eval()
    if not args.hf_device_map_auto:
        target = target.to(device)

    text_model = _target_text_model(target)
    moe_debug_captures: dict[str, torch.Tensor] | None = None
    prefill_debug_captures: dict[str, torch.Tensor] | None = None
    attn_debug_captures: dict[str, torch.Tensor] | None = None
    if args.debug_attn_layer is not None:
        attn_debug_captures = _install_hf_attn_debug_hook(
            text_model, args.debug_attn_layer, args.tp_size
        )
    if args.debug_prefill_layer is not None:
        prefill_debug_captures = _install_hf_prefill_debug_hook(
            text_model, args.debug_prefill_layer
        )
    if args.debug_moe_layer is not None:
        moe_debug_captures = _install_hf_moe_debug_hook(
            text_model, args.debug_moe_layer
        )
    embed = (
        text_model.embed_tokens
        if hasattr(text_model, "embed_tokens")
        else target.get_input_embeddings()
    )
    input_device = _module_device(embed)
    ids = torch.tensor([input_ids], dtype=torch.long, device=input_device)
    position_ids = torch.arange(ids.shape[1], device=input_device).unsqueeze(0)

    outputs = text_model(
        input_ids=ids,
        position_ids=position_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    selected = [
        outputs.hidden_states[int(layer_id) + 1]
        for layer_id in _parse_ints(args.layer_ids)
    ]
    hidden_states = torch.cat([x.to(device) for x in selected], dim=-1)
    lm_head = _target_lm_head(target)
    logits = lm_head(outputs.last_hidden_state.to(_module_device(lm_head))).to(device)
    if moe_debug_captures is not None:
        if "routed_output" in moe_debug_captures and "shared_output" in moe_debug_captures:
            shared = moe_debug_captures["shared_output"]
            routed = moe_debug_captures["routed_output"].reshape(shared.shape)
            moe_debug_captures["moe_output"] = routed + shared
        _write_moe_debug_payload(
            args=args,
            backend="hf",
            input_ids=input_ids,
            layer_id=int(args.debug_moe_layer),
            captures=moe_debug_captures,
        )
    if prefill_debug_captures is not None:
        _write_prefill_debug_payload(
            args=args,
            backend="hf",
            input_ids=input_ids,
            layer_id=int(args.debug_prefill_layer),
            captures=prefill_debug_captures,
        )
    if attn_debug_captures is not None:
        _write_attn_debug_payload(
            args=args,
            backend="hf",
            input_ids=input_ids,
            layer_id=int(args.debug_attn_layer),
            captures=attn_debug_captures,
        )
    _write_payload(
        args=args,
        backend="hf",
        prompt_text=prompt_text,
        input_ids=input_ids,
        hidden_states=hidden_states,
        logits=logits,
    )


def install_vllm_target_parity_probe(worker: Any) -> dict[str, Any]:
    """Install worker-local hooks to capture vLLM target logits/aux states."""
    import types as _types

    worker_obj = getattr(worker, "worker", worker)
    model_runner = getattr(worker_obj, "model_runner", None)
    if model_runner is None:
        return {"installed": False, "reason": "no_model_runner"}
    model = getattr(model_runner, "model", None)
    drafter = getattr(model_runner, "drafter", None)
    if model is None:
        return {"installed": False, "reason": "no_model"}

    state: dict[str, Any] = {
        "orig_compute_logits": getattr(model, "compute_logits", None),
        "orig_propose": getattr(drafter, "propose", None) if drafter is not None else None,
        "orig_moe_forwards": [],
        "orig_layer_forwards": [],
        "orig_attn_forwards": [],
        "compute_call_count": 0,
        "propose_call_count": 0,
        "captures": {},
    }
    if state["orig_compute_logits"] is None:
        return {"installed": False, "reason": "model_has_no_compute_logits"}

    orig_compute_logits = state["orig_compute_logits"]

    def patched_compute_logits(self, hidden_states, *args, **kwargs):
        logits = orig_compute_logits(hidden_states, *args, **kwargs)
        try:
            call_index = int(state.get("compute_call_count", 0))
            state["compute_call_count"] = call_index + 1
            if call_index < 2:
                state["captures"].setdefault("compute_events", []).append(
                    {
                        "call_index": call_index,
                        "hidden_shape": list(hidden_states.shape),
                        "logits_shape": list(logits.shape),
                        "hidden_states": hidden_states.detach().cpu(),
                        "logits": logits.detach().cpu(),
                    }
                )
            if "logits" not in state["captures"]:
                state["captures"]["sample_hidden_states"] = hidden_states.detach().cpu()
                state["captures"]["logits"] = logits.detach().cpu()
        except Exception as exc:
            state["captures"]["logits_capture_error"] = str(exc)
        return logits

    model.compute_logits = _types.MethodType(patched_compute_logits, model)

    debug_layer_env = os.environ.get("STEP3P5_DEBUG_MOE_LAYER")
    if debug_layer_env:
        debug_layer_id = int(debug_layer_env)
        for _module_name, module in model.named_modules():
            if (
                type(module).__name__ == "FusedMoEBlock"
                and getattr(module, "layer_idx", None) == debug_layer_id
            ):
                orig_moe_forward = module.forward
                state["orig_moe_forwards"].append((module, orig_moe_forward))

                def patched_moe_forward(self, hidden_states):
                    num_tokens, hidden_dim = hidden_states.shape
                    flat_hidden = hidden_states.view(-1, hidden_dim)
                    router_logits, _ = self.gate(flat_hidden)
                    topk_weights, topk_ids = self.experts.router.select_experts(
                        hidden_states=flat_hidden,
                        router_logits=router_logits,
                    )
                    fused_moe_out = self.experts(
                        hidden_states=flat_hidden,
                        router_logits=router_logits,
                    )
                    shared_output, routed_output = fused_moe_out
                    if self.share_expert is not None:
                        assert shared_output is not None
                        reduced_shared = (
                            self.experts.maybe_all_reduce_tensor_model_parallel(
                                shared_output.clone()
                            )
                        )
                        reduced_routed = (
                            self.experts.maybe_all_reduce_tensor_model_parallel(
                                routed_output.clone()
                            )
                        )
                        final_hidden_states = routed_output + shared_output
                    else:
                        reduced_shared = None
                        reduced_routed = (
                            self.experts.maybe_all_reduce_tensor_model_parallel(
                                routed_output.clone()
                            )
                        )
                        final_hidden_states = routed_output

                    if self.tp_size > 1:
                        final_hidden_states = (
                            self.experts.maybe_all_reduce_tensor_model_parallel(
                                final_hidden_states
                            )
                        )

                    if "moe_debug" not in state["captures"]:
                        debug = {
                            "mlp_input": flat_hidden.detach().cpu(),
                            "router_logits": router_logits.detach().cpu(),
                            "topk_ids": topk_ids.detach().cpu(),
                            "topk_weights": topk_weights.detach().cpu(),
                            "routed_output": reduced_routed.detach().cpu(),
                            "moe_output": final_hidden_states.detach().cpu(),
                        }
                        if reduced_shared is not None:
                            debug["shared_output"] = reduced_shared.detach().cpu()
                        state["captures"]["moe_debug"] = debug

                    return final_hidden_states.view(num_tokens, hidden_dim)

                module.forward = _types.MethodType(patched_moe_forward, module)

    debug_prefill_layer_env = os.environ.get("STEP3P5_DEBUG_PREFILL_LAYER")
    if debug_prefill_layer_env:
        debug_prefill_layer_id = int(debug_prefill_layer_env)
        for _module_name, module in model.named_modules():
            if (
                type(module).__name__ == "Step3p5DecoderLayer"
                and getattr(module, "layer_idx", None) == debug_prefill_layer_id
            ):
                orig_layer_forward = module.forward
                state["orig_layer_forwards"].append((module, orig_layer_forward))

                def patched_layer_forward(self, positions, hidden_states):
                    residual = hidden_states
                    capture = "prefill_debug" not in state["captures"]
                    debug: dict[str, torch.Tensor] = {}
                    if capture:
                        debug["layer_input"] = residual.detach().cpu()

                    hidden_states = self.input_layernorm(hidden_states)
                    if capture:
                        debug["input_norm"] = hidden_states.detach().cpu()

                    hidden_states = self.self_attn(
                        positions=positions,
                        hidden_states=hidden_states,
                    )
                    if capture:
                        debug["attn_output"] = hidden_states.detach().cpu()

                    hidden_states = self.reduce_attention_output_and_add_residual(
                        hidden_states, residual
                    )
                    if capture:
                        debug["attn_residual"] = hidden_states.detach().cpu()

                    residual = hidden_states
                    hidden_states = self.post_attention_layernorm(hidden_states)
                    if capture:
                        debug["post_attn_norm"] = hidden_states.detach().cpu()
                        debug["moe_input"] = hidden_states.detach().cpu()

                    if self.use_moe:
                        ffn_output = self.moe(hidden_states)
                    else:
                        ffn_output = self.mlp(hidden_states)
                    hidden_states = ffn_output + residual
                    if capture:
                        debug["layer_output"] = hidden_states.detach().cpu()
                        state["captures"]["prefill_debug"] = debug
                    return hidden_states

                module.forward = _types.MethodType(patched_layer_forward, module)

    debug_attn_layer_env = os.environ.get("STEP3P5_DEBUG_ATTN_LAYER")
    if debug_attn_layer_env:
        debug_attn_layer_id = int(debug_attn_layer_env)
        for _module_name, module in model.named_modules():
            if (
                type(module).__name__ == "Step3p5Attention"
                and getattr(module, "layer_idx", None) == debug_attn_layer_id
            ):
                orig_attn_forward = module.forward
                state["orig_attn_forwards"].append((module, orig_attn_forward))

                def patched_attn_forward(self, positions, hidden_states):
                    capture = "attn_debug" not in state["captures"]
                    debug: dict[str, torch.Tensor] = {}
                    if capture:
                        debug["positions"] = positions.detach().cpu()

                    qkv, _ = self.qkv_proj(hidden_states)
                    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
                    if capture:
                        debug["q_proj"] = q.detach().cpu()
                        debug["k_proj"] = k.detach().cpu()
                        debug["v_proj"] = v.detach().cpu()

                    q_by_head = q.view(
                        *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
                    )
                    q_by_head = self.q_norm(q_by_head.contiguous())
                    q = q_by_head.view(q.shape)
                    if capture:
                        debug["q_norm"] = q.detach().cpu()

                    k_by_head = k.view(
                        *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
                    )
                    k_by_head = self.k_norm(k_by_head.contiguous())
                    k = k_by_head.view(k.shape)
                    if capture:
                        debug["k_norm"] = k.detach().cpu()

                    if self.use_rope:
                        q, k = self.rotary_emb(positions, q, k)
                    if capture:
                        debug["q_rope"] = q.detach().cpu()
                        debug["k_rope"] = k.detach().cpu()

                    attn_output = self.attn(q, k, v)
                    if capture:
                        debug["attn_core"] = attn_output.detach().cpu()
                    if self.use_head_wise_attn_gate:
                        extra_dims, _ = self.g_proj(hidden_states)
                        if capture:
                            debug["gate"] = extra_dims.detach().cpu()
                            debug["gate_sigmoid"] = extra_dims.sigmoid().detach().cpu()
                        output = (
                            attn_output.view(
                                *attn_output.shape[:-1], self.num_heads, self.head_dim
                            )
                            * extra_dims.unsqueeze(-1).sigmoid()
                        )
                        attn_output = output.view(*attn_output.shape)
                        if capture:
                            debug["attn_after_gate"] = attn_output.detach().cpu()
                    output, _ = self.o_proj(attn_output)
                    if capture:
                        debug["attn_o_proj"] = output.detach().cpu()
                        state["captures"]["attn_debug"] = debug
                    return output

                module.forward = _types.MethodType(patched_attn_forward, module)

    if drafter is not None and state["orig_propose"] is not None:
        orig_propose = state["orig_propose"]

        def patched_propose(self, *args, **kwargs):
            target_hidden_states = kwargs.get("target_hidden_states")
            sampled_token_ids = kwargs.get("sampled_token_ids")
            if target_hidden_states is None and len(args) >= 2:
                sampled_token_ids = args[0]
                target_hidden_states = args[1]
            try:
                call_index = int(state.get("propose_call_count", 0))
                state["propose_call_count"] = call_index + 1
                if isinstance(target_hidden_states, list) and call_index < 2:
                    sampled_cpu = None
                    if sampled_token_ids is not None:
                        sampled_cpu = (
                            sampled_token_ids.detach().cpu()
                            if hasattr(sampled_token_ids, "detach")
                            else sampled_token_ids
                        )
                    state["captures"].setdefault("propose_events", []).append(
                        {
                            "call_index": call_index,
                            "sampled_token_ids": sampled_cpu,
                            "aux_hidden_shapes": [
                                list(h.shape) for h in target_hidden_states
                            ],
                            "aux_hidden_states": [
                                h.detach().cpu() for h in target_hidden_states
                            ],
                        }
                    )
                if (
                    isinstance(target_hidden_states, list)
                    and "aux_hidden_states" not in state["captures"]
                ):
                    state["captures"]["aux_hidden_states"] = [
                        h.detach().cpu() for h in target_hidden_states
                    ]
                if (
                    sampled_token_ids is not None
                    and "sampled_token_ids" not in state["captures"]
                ):
                    state["captures"]["sampled_token_ids"] = (
                        sampled_token_ids.detach().cpu()
                        if hasattr(sampled_token_ids, "detach")
                        else sampled_token_ids
                    )
            except Exception as exc:
                state["captures"]["aux_capture_error"] = str(exc)
            return orig_propose(*args, **kwargs)

        drafter.propose = _types.MethodType(patched_propose, drafter)

    worker_obj._target_parity_probe_state = state
    return {
        "installed": True,
        "has_drafter": drafter is not None,
        "drafter_type": type(drafter).__name__ if drafter is not None else None,
    }


def retrieve_vllm_target_parity_probe(worker: Any) -> dict[str, Any]:
    worker_obj = getattr(worker, "worker", worker)
    state = getattr(worker_obj, "_target_parity_probe_state", None)
    if state is None:
        return {"error": "probe_not_installed"}
    model_runner = getattr(worker_obj, "model_runner", None)
    model = getattr(model_runner, "model", None) if model_runner is not None else None
    drafter = getattr(model_runner, "drafter", None) if model_runner is not None else None
    if model is not None and state.get("orig_compute_logits") is not None:
        model.compute_logits = state["orig_compute_logits"]
    if drafter is not None and state.get("orig_propose") is not None:
        drafter.propose = state["orig_propose"]
    for module, orig_forward in state.get("orig_moe_forwards") or []:
        module.forward = orig_forward
    for module, orig_forward in state.get("orig_layer_forwards") or []:
        module.forward = orig_forward
    for module, orig_forward in state.get("orig_attn_forwards") or []:
        module.forward = orig_forward
    captures = dict(state.get("captures") or {})
    try:
        delattr(worker_obj, "_target_parity_probe_state")
    except Exception:
        pass
    return captures


def _run_vllm(args: argparse.Namespace) -> None:
    _add_repo_paths(args.specforge_root, args.vllm_root)
    from vllm import LLM, SamplingParams

    prompt_text, input_ids = _encode_prompt(args)
    if args.debug_moe_layer is not None:
        os.environ["STEP3P5_DEBUG_MOE_LAYER"] = str(args.debug_moe_layer)
    if args.debug_prefill_layer is not None:
        os.environ["STEP3P5_DEBUG_PREFILL_LAYER"] = str(args.debug_prefill_layer)
    if args.debug_attn_layer is not None:
        os.environ["STEP3P5_DEBUG_ATTN_LAYER"] = str(args.debug_attn_layer)
    layer_ids = _parse_ints(args.layer_ids)
    # vLLM's EagleModelMixin captures after decoder layer ``i`` when the
    # configured aux id is ``i + 1``. SpecForge/HF layer ids refer to the
    # actual decoder layer output, so shift only the vLLM capture config.
    vllm_capture_layer_ids = [layer_id + 1 for layer_id in layer_ids]
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enforce_eager=args.vllm_enforce_eager,
        enable_expert_parallel=args.enable_expert_parallel,
        disable_cascade_attn=args.disable_cascade_attn,
        language_model_only=True,
        limit_mm_per_prompt={"image": 0},
        skip_mm_profiling=True,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": vllm_capture_layer_ids,
                }
            },
        },
    )
    try:
        install_result = llm.collective_rpc(install_vllm_target_parity_probe)
        print(f"Installed vLLM target parity probe: {install_result}")
        llm.generate(
            [prompt_text],
            SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
        )
        captures_list = llm.collective_rpc(retrieve_vllm_target_parity_probe)
        capture = captures_list[0] if captures_list else {}
        if "error" in capture:
            raise RuntimeError(f"vLLM capture failed: {capture}")
        if args.debug_phase:
            _write_phase_debug_payload(
                args=args,
                prompt_text=prompt_text,
                input_ids=input_ids,
                captures=capture,
            )
        aux_hidden_states = capture.get("aux_hidden_states")
        logits = capture.get("logits")
        moe_debug = capture.get("moe_debug")
        prefill_debug = capture.get("prefill_debug")
        attn_debug = capture.get("attn_debug")
        if not aux_hidden_states or logits is None:
            raise RuntimeError(
                "vLLM capture missing aux hidden states or logits: "
                f"keys={sorted(capture.keys())}"
            )
        prompt_len = len(input_ids)
        # vLLM's prefill aux capture includes the prompt-aligned hidden states
        # at the tail. Keep the final prompt_len rows so the last row matches
        # the last prompt token used by HF/SGLang.
        aux_hidden_states = [h[-prompt_len:] for h in aux_hidden_states]
        hidden_states = torch.cat([h.unsqueeze(0) for h in aux_hidden_states], dim=-1)
        if logits.ndim == 2 and logits.shape[0] > 1:
            logits = logits[prompt_len - 1 : prompt_len]
        if args.debug_moe_layer is not None:
            if not isinstance(moe_debug, dict):
                raise RuntimeError(
                    "vLLM MoE debug capture missing: "
                    f"keys={sorted(capture.keys())}"
                )
            _write_moe_debug_payload(
                args=args,
                backend="vllm",
                input_ids=input_ids,
                layer_id=int(args.debug_moe_layer),
                captures=moe_debug,
            )
        if args.debug_prefill_layer is not None:
            if not isinstance(prefill_debug, dict):
                raise RuntimeError(
                    "vLLM prefill debug capture missing: "
                    f"keys={sorted(capture.keys())}"
                )
            _write_prefill_debug_payload(
                args=args,
                backend="vllm",
                input_ids=input_ids,
                layer_id=int(args.debug_prefill_layer),
                captures=prefill_debug,
            )
        if args.debug_attn_layer is not None:
            if not isinstance(attn_debug, dict):
                raise RuntimeError(
                    "vLLM attention debug capture missing: "
                    f"keys={sorted(capture.keys())}"
                )
            _write_attn_debug_payload(
                args=args,
                backend="vllm",
                input_ids=input_ids,
                layer_id=int(args.debug_attn_layer),
                captures=attn_debug,
            )
        _write_payload(
            args=args,
            backend="vllm",
            prompt_text=prompt_text,
            input_ids=input_ids,
            hidden_states=hidden_states,
            logits=logits.unsqueeze(0),
        )
    finally:
        try:
            llm.shutdown()
        except Exception:
            pass


def _compare_tensor(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if list(a.shape) != list(b.shape):
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "shape_match": False}
    diff = a - b
    return {
        "shape": list(a.shape),
        "shape_match": True,
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.abs().mean().item()) if diff.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()) if diff.numel() else 0.0,
        "cosine": float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())
        if diff.numel()
        else 1.0,
    }


def _run_compare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    sglang_path = _payload_paths(output_dir, "sglang")[0]
    candidate_backend = args.candidate_backend
    candidate_path = _payload_paths(output_dir, candidate_backend)[0]
    missing = [str(path) for path in (sglang_path, candidate_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot compare target payloads because required files are missing: "
            f"{missing}. Run both stages first: `--backend sglang` under the "
            "SGLang/SpecForge torchrun environment and the candidate backend "
            "under the same --output-dir."
        )
    sglang_payload = torch.load(sglang_path, map_location="cpu")
    candidate_payload = torch.load(candidate_path, map_location="cpu")
    if sglang_payload["input_ids"] != candidate_payload["input_ids"]:
        raise ValueError(f"SGLang and {candidate_backend} payload input_ids differ")

    sgl_logits = sglang_payload["last_logits"].float()
    candidate_logits = candidate_payload["last_logits"].float()
    sgl_top = _topk(sgl_logits, args.topk)
    candidate_top = _topk(candidate_logits, args.topk)
    top_overlap = sorted(set(sgl_top["tokens"]) & set(candidate_top["tokens"]))
    report = {
        "output_dir": str(output_dir),
        "candidate_backend": candidate_backend,
        "num_prompt_tokens": len(sglang_payload["input_ids"]),
        "input_ids": sglang_payload["input_ids"],
        "layer_ids": sglang_payload["layer_ids"],
        "hidden_states": _compare_tensor(
            sglang_payload["hidden_states"],
            candidate_payload["hidden_states"],
        ),
        "last_hidden": _compare_tensor(
            sglang_payload["last_hidden"],
            candidate_payload["last_hidden"],
        ),
        "last_logits": _compare_tensor(sgl_logits, candidate_logits),
        "sglang_topk": sgl_top,
        f"{candidate_backend}_topk": candidate_top,
        "topk_overlap_tokens": top_overlap,
        "topk_overlap_ratio": len(top_overlap) / max(1, min(args.topk, len(sgl_top["tokens"]))),
        "top1_match": bool(
            sgl_top["tokens"]
            and candidate_top["tokens"]
            and sgl_top["tokens"][0] == candidate_top["tokens"][0]
        ),
    }

    hidden_size = int(report["last_hidden"]["shape"][0] // len(report["layer_ids"]))
    per_layer = {}
    for idx, layer_id in enumerate(report["layer_ids"]):
        start = idx * hidden_size
        end = start + hidden_size
        per_layer[str(layer_id)] = _compare_tensor(
            sglang_payload["last_hidden"][start:end],
            candidate_payload["last_hidden"][start:end],
        )
    report["last_hidden_per_layer"] = per_layer

    report_path = output_dir / f"sglang_{candidate_backend}_target_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote comparison report to {report_path}")
    print(json.dumps({
        "top1_match": report["top1_match"],
        "hidden_max_abs": report["last_hidden"].get("max_abs"),
        "hidden_cosine": report["last_hidden"].get("cosine"),
        "logits_max_abs": report["last_logits"].get("max_abs"),
        "logits_cosine": report["last_logits"].get("cosine"),
        "topk_overlap_ratio": report["topk_overlap_ratio"],
    }, indent=2))

    if args.debug_moe_layer is not None and candidate_backend == "vllm":
        _run_moe_debug_compare(args)
    if args.debug_prefill_layer is not None and candidate_backend == "vllm":
        _run_prefill_debug_compare(args)
    if args.debug_attn_layer is not None and candidate_backend == "vllm":
        _run_attn_debug_compare(args)


def _run_moe_debug_compare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    layer_id = int(args.debug_moe_layer)
    hf_path = _moe_debug_paths(output_dir, "hf", layer_id)[0]
    vllm_path = _moe_debug_paths(output_dir, "vllm", layer_id)[0]
    missing = [str(path) for path in (hf_path, vllm_path) if not path.exists()]
    if missing:
        print(
            "Skipping MoE debug compare because required files are missing: "
            f"{missing}"
        )
        return

    hf_payload = torch.load(hf_path, map_location="cpu")
    vllm_payload = torch.load(vllm_path, map_location="cpu")
    if hf_payload["input_ids"] != vllm_payload["input_ids"]:
        raise ValueError("HF and vLLM MoE debug payload input_ids differ")

    hf_last = hf_payload["last_token"]
    vllm_last = vllm_payload["last_token"]
    common_keys = sorted(set(hf_last) & set(vllm_last))
    report = {
        "output_dir": str(output_dir),
        "layer_id": layer_id,
        "num_prompt_tokens": len(hf_payload["input_ids"]),
        "last_token": {
            key: _compare_tensor(hf_last[key], vllm_last[key])
            for key in common_keys
        },
    }
    if "topk_ids" in common_keys:
        hf_ids = hf_last["topk_ids"].to(torch.int64).flatten()
        vllm_ids = vllm_last["topk_ids"].to(torch.int64).flatten()
        report["topk_ids_exact_match"] = bool(torch.equal(hf_ids, vllm_ids))
        report["topk_ids_match_ratio"] = float((hf_ids == vllm_ids).float().mean())

    report_path = output_dir / f"hf_vllm_moe_debug_layer{layer_id}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote MoE debug comparison report to {report_path}")
    print(json.dumps(report.get("last_token", {}), indent=2))


def _run_prefill_debug_compare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    layer_id = int(args.debug_prefill_layer)
    hf_path = _prefill_debug_paths(output_dir, "hf", layer_id)[0]
    vllm_path = _prefill_debug_paths(output_dir, "vllm", layer_id)[0]
    missing = [str(path) for path in (hf_path, vllm_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot compare prefill debug payloads because required files are "
            f"missing: {missing}"
        )
    hf_payload = torch.load(hf_path, map_location="cpu")
    vllm_payload = torch.load(vllm_path, map_location="cpu")
    if hf_payload["input_ids"] != vllm_payload["input_ids"]:
        raise ValueError("HF and vLLM prefill debug payload input_ids differ")
    prompt_len = len(hf_payload["input_ids"])
    hf_captures = hf_payload["captures"]
    vllm_captures = vllm_payload["captures"]
    keys = [
        "layer_input",
        "input_norm",
        "attn_output",
        "attn_residual",
        "post_attn_norm",
        "moe_input",
        "layer_output",
    ]
    report: dict[str, Any] = {
        "output_dir": str(output_dir),
        "layer_id": layer_id,
        "num_prompt_tokens": prompt_len,
        "keys": {},
    }
    for key in keys:
        if key not in hf_captures or key not in vllm_captures:
            continue
        hf_tensor = hf_captures[key]
        vllm_tensor = vllm_captures[key]
        hf_tail = hf_tensor[0, -prompt_len:] if hf_tensor.ndim == 3 else hf_tensor[-prompt_len:]
        vllm_tail = vllm_tensor[-prompt_len:] if vllm_tensor.ndim == 2 else vllm_tensor[0, -prompt_len:]
        report["keys"][key] = {
            "hf_shape": list(hf_tensor.shape),
            "vllm_shape": list(vllm_tensor.shape),
            "last_prompt_token": _compare_tensor(
                _debug_last_token(hf_tensor, prompt_len),
                _debug_last_token(vllm_tensor, prompt_len),
            ),
            "prompt_tail": _compare_tensor(hf_tail, vllm_tail),
        }

    report_path = output_dir / f"hf_vllm_prefill_debug_layer{layer_id}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote prefill debug comparison report to {report_path}")
    print(json.dumps({
        key: {
            "last_cosine": value["last_prompt_token"].get("cosine"),
            "last_max_abs": value["last_prompt_token"].get("max_abs"),
            "tail_cosine": value["prompt_tail"].get("cosine"),
        }
        for key, value in report["keys"].items()
    }, indent=2))


def _squeeze_hf_sequence_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


def _hf_local_tail_for_vllm(
    hf_tensor: torch.Tensor,
    vllm_tensor: torch.Tensor,
    prompt_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hf_seq = _squeeze_hf_sequence_tensor(hf_tensor).float()
    vllm_seq = _squeeze_hf_sequence_tensor(vllm_tensor).float()
    hf_tail = hf_seq[-prompt_len:]
    vllm_tail = vllm_seq[-prompt_len:]
    if hf_tail.shape[-1] != vllm_tail.shape[-1]:
        hf_tail = hf_tail[..., : vllm_tail.shape[-1]]
    return hf_tail, vllm_tail


def _run_attn_debug_compare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    layer_id = int(args.debug_attn_layer)
    hf_path = _attn_debug_paths(output_dir, "hf", layer_id)[0]
    vllm_path = _attn_debug_paths(output_dir, "vllm", layer_id)[0]
    missing = [str(path) for path in (hf_path, vllm_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot compare attention debug payloads because required files are "
            f"missing: {missing}"
        )
    hf_payload = torch.load(hf_path, map_location="cpu")
    vllm_payload = torch.load(vllm_path, map_location="cpu")
    if hf_payload["input_ids"] != vllm_payload["input_ids"]:
        raise ValueError("HF and vLLM attention debug payload input_ids differ")
    prompt_len = len(hf_payload["input_ids"])
    hf_captures = hf_payload["captures"]
    vllm_captures = vllm_payload["captures"]
    key_pairs = [
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
        ("q_norm", "q_norm"),
        ("k_norm", "k_norm"),
        ("q_rope", "q_rope"),
        ("k_rope", "k_rope"),
        ("attn_core", "attn_core"),
        ("gate", "gate"),
        ("gate_sigmoid", "gate_sigmoid"),
        ("attn_after_gate", "attn_after_gate"),
        ("attn_o_proj_tp0", "attn_o_proj"),
    ]
    report: dict[str, Any] = {
        "output_dir": str(output_dir),
        "layer_id": layer_id,
        "num_prompt_tokens": prompt_len,
        "keys": {},
    }
    for hf_key, vllm_key in key_pairs:
        if hf_key not in hf_captures or vllm_key not in vllm_captures:
            continue
        hf_tail, vllm_tail = _hf_local_tail_for_vllm(
            hf_captures[hf_key], vllm_captures[vllm_key], prompt_len
        )
        report_key = hf_key if hf_key == vllm_key else f"{hf_key}_vs_{vllm_key}"
        report["keys"][report_key] = {
            "hf_shape": list(hf_captures[hf_key].shape),
            "vllm_shape": list(vllm_captures[vllm_key].shape),
            "compared_shape": list(hf_tail.shape),
            "last_prompt_token": _compare_tensor(hf_tail[-1], vllm_tail[-1]),
            "prompt_tail": _compare_tensor(hf_tail, vllm_tail),
        }

    report_path = output_dir / f"hf_vllm_attn_debug_layer{layer_id}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote attention debug comparison report to {report_path}")
    print(json.dumps({
        key: {
            "last_cosine": value["last_prompt_token"].get("cosine"),
            "last_max_abs": value["last_prompt_token"].get("max_abs"),
            "tail_cosine": value["prompt_tail"].get("cosine"),
        }
        for key, value in report["keys"].items()
    }, indent=2))


def _flatten_sampled_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            out.extend(_flatten_sampled_token_ids(item))
        return out
    return [int(value)]


@torch.inference_mode()
def _run_phase_compare(args: argparse.Namespace) -> None:
    _add_repo_paths(args.specforge_root, args.vllm_root)
    output_dir = Path(args.output_dir)
    phase_path = _phase_debug_path(output_dir)
    if not phase_path.exists():
        raise FileNotFoundError(
            f"Missing vLLM phase debug payload: {phase_path}. "
            "Run `--backend vllm --debug-phase` first."
        )
    phase_payload = torch.load(phase_path, map_location="cpu")
    input_ids = [int(x) for x in phase_payload["input_ids"]]
    layer_ids = [int(x) for x in phase_payload["layer_ids"]]
    propose_events = phase_payload.get("propose_events") or []
    if not propose_events:
        raise RuntimeError("vLLM phase payload has no propose events")

    sampled_ids: list[int] = []
    for event in propose_events:
        sampled_ids = _flatten_sampled_token_ids(event.get("sampled_token_ids"))
        if sampled_ids:
            break
    if not sampled_ids:
        raise RuntimeError("Unable to recover sampled token id from vLLM phase payload")
    sampled_token_id = int(sampled_ids[-1])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": args.hf_attn_implementation,
    }
    if args.hf_device_map_auto:
        load_kwargs["device_map"] = "balanced"
    target = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).eval()
    if not args.hf_device_map_auto:
        target = target.to(device)
    text_model = _target_text_model(target)
    embed = (
        text_model.embed_tokens
        if hasattr(text_model, "embed_tokens")
        else target.get_input_embeddings()
    )
    input_device = _module_device(embed)

    def run_hf(ids_list: list[int]) -> dict[str, Any]:
        ids = torch.tensor([ids_list], dtype=torch.long, device=input_device)
        position_ids = torch.arange(ids.shape[1], device=input_device).unsqueeze(0)
        outputs = text_model(
            input_ids=ids,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        selected = [
            outputs.hidden_states[layer_id + 1][0, -1].detach().float().cpu()
            for layer_id in layer_ids
        ]
        return {
            "num_tokens": len(ids_list),
            "last_hidden_by_layer": selected,
        }

    hf_refs = {
        "prompt": run_hf(input_ids),
        "prompt_plus_sampled": run_hf(input_ids + [sampled_token_id]),
    }

    report: dict[str, Any] = {
        "output_dir": str(output_dir),
        "input_ids": input_ids,
        "layer_ids": layer_ids,
        "sampled_token_id": sampled_token_id,
        "propose_events": [],
    }
    for event in propose_events:
        aux = event.get("aux_hidden_states") or []
        event_report: dict[str, Any] = {
            "call_index": event.get("call_index"),
            "sampled_token_ids": _flatten_sampled_token_ids(
                event.get("sampled_token_ids")
            ),
            "aux_hidden_shapes": event.get("aux_hidden_shapes"),
            "vs_hf": {},
        }
        for ref_name, ref in hf_refs.items():
            per_layer = {}
            cosines = []
            for idx, layer_id in enumerate(layer_ids):
                if idx >= len(aux):
                    continue
                vllm_last = _debug_last_token(aux[idx], ref["num_tokens"])
                cmp = _compare_tensor(ref["last_hidden_by_layer"][idx], vllm_last)
                per_layer[str(layer_id)] = cmp
                if cmp.get("shape_match"):
                    cosines.append(float(cmp["cosine"]))
            event_report["vs_hf"][ref_name] = {
                "num_tokens": ref["num_tokens"],
                "mean_cosine": sum(cosines) / len(cosines) if cosines else None,
                "per_layer": per_layer,
            }
        report["propose_events"].append(event_report)

    report_path = _phase_report_path(output_dir)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote phase debug report to {report_path}")
    print(json.dumps({
        "sampled_token_id": sampled_token_id,
        "events": [
            {
                "call_index": event["call_index"],
                "aux_hidden_shapes": event.get("aux_hidden_shapes"),
                "prompt_mean_cosine": event["vs_hf"]["prompt"]["mean_cosine"],
                "prompt_plus_sampled_mean_cosine": event["vs_hf"][
                    "prompt_plus_sampled"
                ]["mean_cosine"],
            }
            for event in report["propose_events"]
        ],
    }, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=(
            "sglang",
            "hf",
            "vllm",
            "compare",
            "phase_compare",
            "prefill_compare",
            "attn_compare",
        ),
        required=True,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-is-templated", action="store_true")
    parser.add_argument("--layer-ids", default=",".join(str(x) for x in DEFAULT_LAYER_IDS))
    parser.add_argument("--debug-phase", action="store_true")
    parser.add_argument("--debug-prefill-layer", type=int)
    parser.add_argument("--debug-attn-layer", type=int)
    parser.add_argument("--debug-moe-layer", type=int)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--specforge-root", default="/root/workspace/specforge")
    parser.add_argument("--vllm-root", default="/root/workspace/vllm-parallel-drafting")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dist-timeout", type=int, default=180)
    parser.add_argument("--sglang-attention-backend", default="triton")
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.4)
    parser.add_argument("--sglang-ep-size", type=int, default=8)
    parser.add_argument(
        "--sglang-disable-custom-all-reduce",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hf-attn-implementation", default="eager")
    parser.add_argument(
        "--hf-device-map-auto",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--candidate-backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=16384)
    parser.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--disable-cascade-attn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.backend == "sglang":
        _run_sglang(args)
    elif args.backend == "hf":
        _run_hf(args)
    elif args.backend == "vllm":
        _run_vllm(args)
    elif args.backend == "compare":
        _run_compare(args)
    elif args.backend == "phase_compare":
        _run_phase_compare(args)
    elif args.backend == "prefill_compare":
        if args.debug_prefill_layer is None:
            raise ValueError("--debug-prefill-layer is required for prefill_compare")
        _run_prefill_debug_compare(args)
    elif args.backend == "attn_compare":
        if args.debug_attn_layer is None:
            raise ValueError("--debug-attn-layer is required for attn_compare")
        _run_attn_debug_compare(args)
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")


if __name__ == "__main__":
    main()
