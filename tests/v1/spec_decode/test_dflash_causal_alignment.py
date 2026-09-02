# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import numpy as np
import os
from datetime import datetime
from pathlib import Path

import pytest
import torch

import vllm.v1.spec_decode.dflash as dflash_mod
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.v1.spec_decode.dflash import DFlashProposer

_DEBUG_DIR_ENV = "DFLASH_TEST_DEBUG_DIR"
_DEFAULT_DEBUG_DIR = Path("/tmp") / f"debug_{datetime.now():%Y%m%d_%H%M%S}"


class _FakeCopyAndExpandDFlashInputsKernel:
    """CPU stand-in for the Triton kernel used by DFlash first-pass setup."""

    def __getitem__(self, grid):
        del grid

        def _launch(
            *,
            next_token_ids_ptr: torch.Tensor,
            target_positions_ptr: torch.Tensor,
            out_input_ids_ptr: torch.Tensor,
            out_context_positions_ptr: torch.Tensor,
            out_query_positions_ptr: torch.Tensor,
            out_context_slot_mapping_ptr: torch.Tensor,
            out_query_slot_mapping_ptr: torch.Tensor,
            out_token_indices_ptr: torch.Tensor,
            block_table_ptr: torch.Tensor,
            block_table_stride: int,
            query_start_loc_ptr: torch.Tensor,
            num_rejected_tokens_ptr: torch.Tensor | int,
            parallel_drafting_token_id: int,
            block_size: int,
            num_query_per_req: int,
            num_speculative_tokens: int,
            total_input_tokens: int,
            BLOCK_SIZE: int,
            HAS_NUM_REJECTED: bool = False,
        ) -> None:
            del block_table_stride, total_input_tokens, BLOCK_SIZE

            batch_size = query_start_loc_ptr.numel() - 1
            for req_idx in range(batch_size):
                ctx_start = int(query_start_loc_ptr[req_idx].item())
                ctx_end = int(query_start_loc_ptr[req_idx + 1].item())
                num_ctx = ctx_end - ctx_start

                if HAS_NUM_REJECTED:
                    num_rejected = int(num_rejected_tokens_ptr[req_idx].item())
                    valid_ctx_end = ctx_end - num_rejected
                else:
                    valid_ctx_end = ctx_end

                last_pos = int(target_positions_ptr[valid_ctx_end - 1].item())

                for j in range(num_ctx):
                    ctx_pos = int(target_positions_ptr[ctx_start + j].item())
                    block_num = min(ctx_pos // block_size, block_table_ptr.shape[1] - 1)
                    block_id = int(block_table_ptr[req_idx, block_num].item())
                    slot = block_id * block_size + (ctx_pos % block_size)

                    out_context_positions_ptr[ctx_start + j] = ctx_pos
                    out_context_slot_mapping_ptr[ctx_start + j] = slot

                for query_off in range(num_query_per_req):
                    query_out = req_idx * num_query_per_req + query_off
                    query_pos = last_pos + 1 + query_off
                    block_num = min(query_pos // block_size, block_table_ptr.shape[1] - 1)
                    block_id = int(block_table_ptr[req_idx, block_num].item())
                    slot = block_id * block_size + (query_pos % block_size)

                    out_query_positions_ptr[query_out] = query_pos
                    out_query_slot_mapping_ptr[query_out] = slot
                    out_input_ids_ptr[query_out] = (
                        int(next_token_ids_ptr[req_idx].item())
                        if query_off == 0
                        else parallel_drafting_token_id
                    )
                    if query_off > 0:
                        sample_out_idx = req_idx * num_speculative_tokens + (query_off - 1)
                        out_token_indices_ptr[sample_out_idx] = query_out

        return _launch


def _make_stub_proposer(
    *,
    num_speculative_tokens: int,
    dflash_is_causal: bool,
    device: torch.device,
) -> DFlashProposer:
    proposer = object.__new__(DFlashProposer)
    proposer.device = device
    proposer.block_size = 16
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer.parallel_drafting_token_id = 151669
    proposer.dflash_is_causal = dflash_is_causal
    proposer.input_ids = torch.zeros(64, dtype=torch.int32, device=device)
    proposer.positions = torch.zeros(64, dtype=torch.int64, device=device)
    proposer._context_positions_buffer = torch.zeros(64, dtype=torch.int64, device=device)
    proposer._context_slot_mapping_buffer = torch.zeros(64, dtype=torch.int64, device=device)
    proposer._slot_mapping_buffer = torch.zeros(64, dtype=torch.int64, device=device)
    proposer.arange = torch.arange(65, device=device, dtype=torch.int32)
    proposer.token_arange_np = np.arange(65, dtype=np.int32)
    return proposer


def _patch_dflash_cpu_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dflash_mod,
        "copy_and_expand_dflash_inputs_kernel",
        _FakeCopyAndExpandDFlashInputsKernel(),
    )
    monkeypatch.setattr(
        dflash_mod.triton,
        "next_power_of_2",
        lambda x: 1 if x <= 1 else 1 << (x - 1).bit_length(),
        raising=False,
    )
    monkeypatch.setattr(
        dflash_mod.triton,
        "cdiv",
        lambda x, y: (x + y - 1) // y,
        raising=False,
    )


def _get_debug_dir() -> Path:
    debug_dir = Path(os.environ.get(_DEBUG_DIR_ENV, str(_DEFAULT_DEBUG_DIR)))
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _write_debug_artifacts(name: str, payload: dict) -> None:
    debug_dir = _get_debug_dir()
    torch_payload = {}
    json_payload = {}

    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            torch_payload[key] = tensor
            json_payload[key] = tensor.tolist()
        else:
            torch_payload[key] = value
            json_payload[key] = value

    torch.save(torch_payload, debug_dir / f"{name}.pt")
    (debug_dir / f"{name}.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("dflash_is_causal", "expected_causal"),
    [(True, True), (False, False)],
)
def test_set_inputs_first_pass_propagates_resolved_causal_mode(
    monkeypatch: pytest.MonkeyPatch,
    dflash_is_causal: bool,
    expected_causal: bool,
) -> None:
    _patch_dflash_cpu_dependencies(monkeypatch)

    device = torch.device("cpu")
    proposer = _make_stub_proposer(
        num_speculative_tokens=2,
        dflash_is_causal=dflash_is_causal,
        device=device,
    )

    batch_spec = BatchSpec(seq_lens=[4, 3], query_lens=[4, 3])
    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=proposer.block_size,
        device=device,
        arange_block_indices=True,
    )

    target_token_ids = torch.tensor([10, 11, 12, 13, 20, 21, 22], dtype=torch.int32)
    target_positions = torch.tensor([7, 8, 9, 10, 5, 6, 7], dtype=torch.int64)
    target_hidden_states = torch.randn(7, 4, dtype=torch.float32)
    next_token_ids = torch.tensor([100, 200], dtype=torch.int32)

    num_tokens, token_indices_to_sample, output_cad = proposer.set_inputs_first_pass(
        target_token_ids=target_token_ids,
        next_token_ids=next_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        token_indices_to_sample=None,
        cad=common_attn_metadata,
        num_rejected_tokens_gpu=None,
    )

    assert num_tokens == 6
    assert output_cad.causal is expected_causal
    assert torch.equal(
        proposer.input_ids[:num_tokens],
        torch.tensor([100, 151669, 151669, 200, 151669, 151669], dtype=torch.int32),
    )
    assert torch.equal(
        proposer.positions[:num_tokens],
        torch.tensor([11, 12, 13, 8, 9, 10], dtype=torch.int64),
    )
    assert torch.equal(
        token_indices_to_sample,
        torch.tensor([1, 2, 4, 5], dtype=torch.int32),
    )
    _write_debug_artifacts(
        f"vllm_set_inputs_first_pass_causal_{str(expected_causal).lower()}",
        {
            "expected_causal": expected_causal,
            "num_tokens": num_tokens,
            "target_positions": target_positions,
            "target_hidden_states": target_hidden_states,
            "next_token_ids": next_token_ids,
            "query_input_ids": proposer.input_ids[:num_tokens],
            "query_positions": proposer.positions[:num_tokens],
            "context_positions": proposer._context_positions_buffer[: target_token_ids.numel()],
            "token_indices_to_sample": token_indices_to_sample,
            "query_start_loc": output_cad.query_start_loc,
            "seq_lens": output_cad.seq_lens,
        },
    )


def test_set_inputs_first_pass_uses_last_valid_position_when_tokens_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_dflash_cpu_dependencies(monkeypatch)

    device = torch.device("cpu")
    proposer = _make_stub_proposer(
        num_speculative_tokens=2,
        dflash_is_causal=True,
        device=device,
    )

    batch_spec = BatchSpec(seq_lens=[4, 3], query_lens=[4, 3])
    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=proposer.block_size,
        device=device,
        arange_block_indices=True,
    )

    target_token_ids = torch.tensor([10, 11, 12, 13, 20, 21, 22], dtype=torch.int32)
    target_positions = torch.tensor([7, 8, 9, 10, 5, 6, 7], dtype=torch.int64)
    target_hidden_states = torch.randn(7, 4, dtype=torch.float32)
    next_token_ids = torch.tensor([100, 200], dtype=torch.int32)
    num_rejected_tokens_gpu = torch.tensor([1, 2], dtype=torch.int32)

    num_tokens, token_indices_to_sample, output_cad = proposer.set_inputs_first_pass(
        target_token_ids=target_token_ids,
        next_token_ids=next_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        token_indices_to_sample=None,
        cad=common_attn_metadata,
        num_rejected_tokens_gpu=num_rejected_tokens_gpu,
    )

    assert num_tokens == 6
    assert output_cad.causal is True
    assert torch.equal(
        proposer.positions[:num_tokens],
        torch.tensor([10, 11, 12, 6, 7, 8], dtype=torch.int64),
    )
    assert torch.equal(
        token_indices_to_sample,
        torch.tensor([1, 2, 4, 5], dtype=torch.int32),
    )
    assert torch.equal(
        output_cad.seq_lens,
        torch.tensor([6, 4], dtype=torch.int32),
    )
    _write_debug_artifacts(
        "vllm_set_inputs_first_pass_with_rejections",
        {
            "num_tokens": num_tokens,
            "num_rejected_tokens_gpu": num_rejected_tokens_gpu,
            "target_positions": target_positions,
            "next_token_ids": next_token_ids,
            "query_input_ids": proposer.input_ids[:num_tokens],
            "query_positions": proposer.positions[:num_tokens],
            "context_positions": proposer._context_positions_buffer[: target_token_ids.numel()],
            "token_indices_to_sample": token_indices_to_sample,
            "query_start_loc": output_cad.query_start_loc,
            "seq_lens": output_cad.seq_lens,
            "causal": output_cad.causal,
        },
    )
