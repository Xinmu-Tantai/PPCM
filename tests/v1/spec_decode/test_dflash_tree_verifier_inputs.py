# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.spec_decode.dflash_tree import (
    _build_attention_bias_np,
    build_ancestor_matrix_np,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _allowed_cols(bias: np.ndarray, row: int) -> list[int]:
    return np.where(bias[row] == 0.0)[0].tolist()


def test_depth_one_sibling_bias_matches_first_bad_node_shape() -> None:
    # Mirrors the first bad probe row:
    #   node 0 = root "```"
    #   nodes 1..7 = depth-1 alternatives, including node 6 = "c"
    parents = [-1, 0, 0, 0, 0, 0, 0, 0]
    depths = [0, 1, 1, 1, 1, 1, 1, 1]
    context_len = 167
    per_query_abs_pos = [context_len + depth for depth in depths]

    neg_inf = float(torch.finfo(torch.float32).min)
    bias = _build_attention_bias_np(parents, neg_inf)
    ancestors = build_ancestor_matrix_np(parents)

    assert depths[6] == 1
    assert per_query_abs_pos[6] == context_len + 1
    assert _allowed_cols(bias, 6) == [0, 6]
    np.testing.assert_array_equal(
        np.where(ancestors[6] == 1)[0],
        np.array([0, 6]),
    )
    assert all(bias[6, sibling] == neg_inf for sibling in range(1, 6))


def test_batched_block_diagonal_bias_uses_global_query_offsets() -> None:
    neg_inf = float(torch.finfo(torch.float32).min)
    chain_parents = [-1, 0, 1]
    tree_parents = [-1, 0, 0, 0, 0, 0, 0, 0]
    total_query_len = len(chain_parents) + len(tree_parents)
    bias = np.full((total_query_len, total_query_len), neg_inf, dtype=np.float32)
    bias[:3, :3] = _build_attention_bias_np(chain_parents, neg_inf)
    bias[3:, 3:] = _build_attention_bias_np(tree_parents, neg_inf)

    # Request 1 local node 6 lives at global row 9.  It may attend to the
    # request-local root at global col 3 and itself at col 9 only.
    assert _allowed_cols(bias, 9) == [3, 9]
    assert all(bias[9, sibling] == neg_inf for sibling in range(4, 9))
    assert all(bias[9, prior_req_col] == neg_inf for prior_req_col in range(3))


def test_tree_depth_positions_update_xdrope_model_facing_buffer() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 4
    runner.positions = torch.tensor([10, 11, 12, 13, 99], dtype=torch.int64)
    runner.num_computed_tokens = torch.tensor([167], dtype=torch.int64)
    runner.xdrope_positions = SimpleNamespace(
        gpu=torch.tensor(
            [
                [167, 168, 169, 170, 999],
                [167, 168, 169, 170, 999],
                [167, 168, 169, 170, 999],
                [167, 168, 169, 170, 999],
            ],
            dtype=torch.int64,
        )
    )

    spec_decode_metadata = SimpleNamespace(
        logits_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        depths=torch.tensor([0, 1, 1, 2], dtype=torch.int64),
    )
    req_indices_gpu = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int64)

    runner._apply_dflash_tree_depth_positions(
        spec_decode_metadata,
        req_indices_gpu,
    )

    expected = torch.tensor([167, 168, 168, 169], dtype=torch.int64)
    assert torch.equal(runner.positions[:4], expected)
    assert torch.equal(
        runner.xdrope_positions.gpu[:, :4],
        expected.unsqueeze(0).expand(4, -1),
    )
    assert runner.positions[4].item() == 99
    assert runner.xdrope_positions.gpu[0, 4].item() == 999


def test_tree_depth_positions_update_mrope_model_facing_buffer() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.uses_mrope = True
    runner.uses_xdrope_dim = 0
    runner.positions = torch.tensor([10, 11, 12], dtype=torch.int64)
    runner.num_computed_tokens = torch.tensor([42], dtype=torch.int64)
    runner.mrope_positions = SimpleNamespace(
        gpu=torch.tensor(
            [
                [42, 43, 44],
                [42, 43, 44],
                [42, 43, 44],
            ],
            dtype=torch.int64,
        )
    )

    spec_decode_metadata = SimpleNamespace(
        logits_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
        depths=torch.tensor([0, 1, 1], dtype=torch.int64),
    )
    req_indices_gpu = torch.tensor([0, 0, 0], dtype=torch.int64)

    runner._apply_dflash_tree_depth_positions(
        spec_decode_metadata,
        req_indices_gpu,
    )

    expected = torch.tensor([42, 43, 43], dtype=torch.int64)
    assert torch.equal(runner.positions, expected)
    assert torch.equal(
        runner.mrope_positions.gpu,
        expected.unsqueeze(0).expand(3, -1),
    )


def test_verifier_state_probe_attaches_slots_and_kv(monkeypatch) -> None:
    monkeypatch.setenv("DFLASH_VERIFIER_PROBE_NODES", "0,2")
    monkeypatch.setenv("DFLASH_VERIFIER_PROBE_KV_LAYERS", "0")
    monkeypatch.setenv("DFLASH_VERIFIER_PROBE_CONTEXT_TAIL", "2")

    runner = object.__new__(GPUModelRunner)
    runner.positions = torch.tensor([10, 11, 12], dtype=torch.int64)
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    kv_cache = torch.zeros((2, 8, 4, 1, 2), dtype=torch.float32)
    for block in range(8):
        for offset in range(4):
            slot = block * 4 + offset
            kv_cache[0, block, offset, 0] = torch.tensor([slot, slot + 0.5])
            kv_cache[1, block, offset, 0] = torch.tensor([100 + slot, 100.5 + slot])
    runner.kv_caches = [kv_cache]

    bundle: dict[str, object] = {}
    spec_decode_metadata = SimpleNamespace(
        depths=torch.tensor([0, 1, 2], dtype=torch.int64),
        parent_indices=torch.tensor([-1, 0, 1], dtype=torch.int64),
    )
    common_attn_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        max_seq_len=8,
        block_table_tensor=torch.tensor([[0, 1]], dtype=torch.int32),
    )
    req_slots = torch.tensor([5, 6, 7], dtype=torch.int64)

    runner._attach_dflash_verifier_state_probe(
        bundle=bundle,
        req_idx=0,
        req_start=0,
        qlen=3,
        req_slots=req_slots,
        spec_decode_metadata=spec_decode_metadata,
        common_attn_metadata=common_attn_metadata,
    )

    assert bundle["slot_mapping"].tolist() == [5, 6, 7]
    assert bundle["verifier_probe_node_indices"].tolist() == [0, 2]
    assert bundle["verifier_probe_slots"].tolist() == [5, 7]
    assert bundle["verifier_probe_positions"].tolist() == [10, 12]
    assert bundle["verifier_probe_model_position_kind"] == "default"
    assert bundle["verifier_probe_model_positions"].tolist() == [[10, 12]]
    assert bundle["verifier_probe_depths"].tolist() == [0, 2]
    assert bundle["verifier_probe_parent_indices"].tolist() == [-1, 1]
    layer0 = bundle["verifier_probe_kv"]["0"]
    assert layer0["key"].shape == (2, 1, 2)
    assert layer0["value"].shape == (2, 1, 2)
    assert bundle["verifier_probe_context_positions"].tolist() == [3, 4]
    assert bundle["verifier_probe_context_slots"].tolist() == [3, 4]


def test_verifier_forward_probe_attaches_logits_and_hidden(monkeypatch) -> None:
    monkeypatch.setenv("DFLASH_VERIFIER_PROBE_NODES", "0,2")
    monkeypatch.setenv("DFLASH_VERIFIER_PROBE_TOPK", "2")
    monkeypatch.setenv("DFLASH_VERIFIER_PROBE_TOKEN_IDS", "1,5")

    runner = object.__new__(GPUModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.input_ids = SimpleNamespace(
        gpu=torch.tensor([10, 11, 12], dtype=torch.int64)
    )
    runner._dflash_runtime_verify_bundles = [
        {"req_id": "req-0", "query_start": 0, "query_end": 3}
    ]

    logits = torch.tensor(
        [
            [0.0, 0.5, 0.1, 2.0, 0.3, 1.0, 0.2, 0.4],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.4, 0.2, 3.0, 0.1, 2.0, 0.9, 0.0, 0.8],
        ],
        dtype=torch.float32,
    )
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    spec_decode_metadata = SimpleNamespace(
        logits_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
        query_lens=[3],
        is_tree_req=[True],
    )

    runner._attach_dflash_verifier_forward_probe(
        logits=logits,
        sample_hidden_states=hidden,
        spec_decode_metadata=spec_decode_metadata,
    )

    bundle = runner._dflash_runtime_verify_bundles[0]
    assert bundle["verifier_forward_probe_node_indices"].tolist() == [0, 2]
    assert bundle["verifier_forward_probe_token_ids"].tolist() == [10, 12]
    assert bundle["verifier_forward_probe_greedy_token_ids"].tolist() == [3, 2]
    assert bundle["verifier_forward_probe_topk_token_ids"].tolist() == [
        [3, 5],
        [2, 4],
    ]
    assert bundle["verifier_forward_probe_candidate_token_ids"].tolist() == [1, 2, 3, 5]
    assert torch.allclose(
        bundle["verifier_forward_probe_candidate_logits"],
        torch.tensor([[0.5, 0.1, 2.0, 1.0], [0.2, 3.0, 0.1, 0.9]]),
    )
