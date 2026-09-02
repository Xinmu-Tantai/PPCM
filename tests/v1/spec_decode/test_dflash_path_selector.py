# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.models.dflash_path_selector import (
    DFlashPathSelector,
    DFlashPathSelectorConfig,
    dflash_path_selector_loss,
)
from vllm.v1.attention.selector import _uses_non_causal_dflash_attention
from vllm.v1.spec_decode.dflash_path import build_all_top2_paths
from vllm.v1.spec_decode.dflash_tree import build_tree_from_topk


def _make_selector() -> DFlashPathSelector:
    return DFlashPathSelector(
        DFlashPathSelectorConfig(
            input_hidden_size=8,
            hidden_size=16,
            head_dim=8,
            intermediate_size=16,
            shared_query=True,
        )
    )


def _make_pair_selector() -> DFlashPathSelector:
    return DFlashPathSelector(
        DFlashPathSelectorConfig(
            input_hidden_size=8,
            hidden_size=16,
            head_dim=8,
            intermediate_size=16,
            num_layers=0,
            shared_query=True,
            selector_type="pair_scorer",
        )
    )


def _make_gated_pair_selector() -> DFlashPathSelector:
    return DFlashPathSelector(
        DFlashPathSelectorConfig(
            input_hidden_size=8,
            hidden_size=16,
            head_dim=8,
            intermediate_size=16,
            num_layers=0,
            selector_type="gated_pair_scorer",
            switch_margin=0.5,
        )
    )


def _make_inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    node_depths = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]]
    ).expand(batch_size, -1)
    node_ranks = torch.tensor(
        [[-1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]
    ).expand(batch_size, -1)
    return {
        "root_hidden": torch.randn(batch_size, 8),
        "depth_hidden": torch.randn(batch_size, 7, 8),
        "node_embeddings": torch.randn(batch_size, 15, 8),
        "node_depths": node_depths,
        "node_ranks": node_ranks,
        "node_numeric_features": torch.randn(batch_size, 15, 2),
        "node_valid_mask": torch.ones(batch_size, 15, dtype=torch.bool),
    }


def test_fixed_path_masks_cover_all_binary_paths():
    selector = _make_selector()
    assert selector.fixed_path_ranks.shape == (128, 7)
    assert selector.fixed_path_node_indices.shape == (128, 7)
    assert selector.fixed_path_mask.shape == (128, 7, 15)
    assert selector.fixed_path_ranks[0].tolist() == [0] * 7
    assert selector.fixed_path_ranks[-1].tolist() == [1] * 7
    assert selector.fixed_path_node_indices[0].tolist() == [
        1,
        3,
        5,
        7,
        9,
        11,
        13,
    ]
    visible = torch.nonzero(
        selector.fixed_path_mask[0, 3],
        as_tuple=False,
    ).flatten()
    assert visible.tolist() == [0, 1, 3, 5, 7]


def test_build_all_top2_paths_has_stable_head_mapping():
    candidates = torch.arange(14).view(1, 7, 2)
    logprobs = torch.tensor([[[-0.1, -1.0]]]).expand(1, 7, 2)
    paths = build_all_top2_paths(candidates, logprobs)
    selector = _make_selector()
    torch.testing.assert_close(paths.ranks[0], selector.fixed_path_ranks)
    assert paths.token_ids.shape == (1, 128, 7)
    assert paths.token_ids[0, 0].tolist() == [0, 2, 4, 6, 8, 10, 12]
    assert paths.token_ids[0, 1].tolist() == [0, 2, 4, 6, 8, 10, 13]
    assert paths.token_ids[0, -1].tolist() == [1, 3, 5, 7, 9, 11, 13]


def test_complete_top2_tree_covers_all_128_paths():
    topk_token_ids = torch.arange(14).view(7, 2)
    topk_logprobs = torch.tensor([[-0.1, -1.0]]).expand(7, 2)
    tree = build_tree_from_topk(
        99,
        topk_token_ids,
        topk_logprobs,
        budget=255,
        device=torch.device("cpu"),
        depth_first=False,
        path_prune_budget=255,
    )

    assert tree.num_nodes == 255
    leaves = torch.nonzero(tree.depth == 7, as_tuple=False).flatten().tolist()
    assert len(leaves) == 128

    observed_paths = set()
    parents = tree.parent_indices.tolist()
    tokens = tree.token_ids.tolist()
    for leaf in leaves:
        ranks = []
        node = leaf
        while parents[node] >= 0:
            ranks.append(tokens[node] % 2)
            node = parents[node]
        observed_paths.add(tuple(reversed(ranks)))

    expected_paths = {
        tuple(pattern.tolist())
        for pattern in build_all_top2_paths(
            topk_token_ids.unsqueeze(0),
            topk_logprobs.unsqueeze(0),
        ).ranks[0]
    }
    assert observed_paths == expected_paths


def test_path_oracle_splits_target_and_draft_attention_causality():
    draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(dflash_config={}),
    )
    oracle_config = SimpleNamespace(
        method="dflash",
        head_type="auto",
        enable_path_oracle=True,
        tree_width=2,
        draft_model_config=draft_model_config,
    )

    # Target full-tree verification must select causal TREE_ATTN.
    assert not _uses_non_causal_dflash_attention(oracle_config)

    # DFlashProposer copies the same config and sets tree_width=1 before
    # loading the draft. That copy must retain the checkpoint's non-causal
    # attention behavior.
    oracle_config.tree_width = 1
    assert _uses_non_causal_dflash_attention(oracle_config)


def test_selector_computes_shared_kv_once_and_returns_all_paths():
    torch.manual_seed(0)
    selector = _make_selector()
    key_calls = 0
    value_calls = 0

    def count_key(*_args):
        nonlocal key_calls
        key_calls += 1

    def count_value(*_args):
        nonlocal value_calls
        value_calls += 1

    key_hook = selector.blocks[0].key_proj.register_forward_hook(count_key)
    value_hook = selector.blocks[0].value_proj.register_forward_hook(count_value)
    output = selector(**_make_inputs())
    key_hook.remove()
    value_hook.remove()
    assert key_calls == 1
    assert value_calls == 1
    assert output.path_states.shape == (2, 128, 7, 8)
    assert output.conditional_survival.shape == (2, 128, 7)
    assert output.expected_accept_length.shape == (2, 128)
    assert output.best_path_indices.shape == (2,)
    assert torch.isfinite(output.path_states).all()
    assert torch.isfinite(output.expected_accept_length).all()


def test_pair_scorer_encodes_seven_pairs_and_scores_all_paths():
    torch.manual_seed(4)
    selector = _make_pair_selector()
    output = selector(**_make_inputs())
    assert output.choice_logits.shape == (2, 7, 3)
    assert output.path_states.shape == (2, 7, 8)
    assert output.conditional_survival.shape == (2, 128, 7)
    assert output.expected_accept_length.shape == (2, 128)
    assert output.best_path_indices.shape == (2,)
    assert torch.isfinite(output.path_states).all()
    assert torch.isfinite(output.expected_accept_length).all()
    # Candidate probabilities and None form one conditional distribution.
    probabilities = torch.softmax(output.choice_logits, dim=-1)
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0])
    )


def test_gated_pair_scorer_returns_one_switch_logit_per_request():
    selector = _make_gated_pair_selector()
    output = selector(**_make_inputs())
    assert output.switch_logits.shape == (2,)
    assert output.choice_logits.shape == (2, 7, 3)
    assert torch.isfinite(output.switch_logits).all()


def test_v7_gate_returns_calibratable_utility_delta():
    selector = DFlashPathSelector(
        DFlashPathSelectorConfig(
            input_hidden_size=8,
            hidden_size=16,
            head_dim=8,
            intermediate_size=16,
            num_layers=0,
            selector_type="gated_pair_scorer",
            gate_feature_version=2,
            gate_predict_delta=True,
            switch_margin=0.25,
        )
    )
    output = selector(**_make_inputs())
    assert output.switch_logits.shape == (2,)
    assert output.switch_deltas.shape == (2,)
    assert torch.isfinite(output.switch_deltas).all()


def test_switch_margin_falls_back_to_top1_path():
    selector = DFlashPathSelector(
        DFlashPathSelectorConfig(
            input_hidden_size=8,
            hidden_size=16,
            head_dim=8,
            intermediate_size=16,
            shared_query=True,
            switch_margin=8.0,
        )
    ).eval()
    with torch.no_grad():
        output = selector(**_make_inputs())
    assert output.best_path_indices.tolist() == [0, 0]


def test_path_mask_blocks_off_path_candidate_changes():
    torch.manual_seed(1)
    selector = _make_selector().eval()
    inputs = _make_inputs(batch_size=1)
    changed = {name: value.clone() for name, value in inputs.items()}
    # Node 2 is the Top-2 candidate at depth 1. Head 0 permanently follows
    # the all-Top-1 path and must not observe it.
    changed["node_embeddings"][:, 2] += 100
    first = selector(**inputs)
    second = selector(**changed)
    torch.testing.assert_close(
        first.path_states[:, 0],
        second.path_states[:, 0],
    )


def test_shared_query_keeps_shared_prefix_states_identical():
    torch.manual_seed(3)
    selector = _make_selector().eval()
    output = selector(**_make_inputs(batch_size=1))
    # Paths 0 and 1 only diverge at depth 7.
    torch.testing.assert_close(
        output.path_states[:, 0, :6],
        output.path_states[:, 1, :6],
        rtol=0,
        atol=0,
    )


def test_selector_loss_has_finite_gradients():
    torch.manual_seed(2)
    selector = _make_selector()
    output = selector(**_make_inputs())
    accepted_lengths = torch.arange(128)[None].remainder(8).expand(2, -1)
    losses = dflash_path_selector_loss(output, accepted_lengths)
    losses.loss.backward()
    assert torch.isfinite(losses.loss)
    assert torch.isfinite(losses.hazard_loss)
    assert torch.isfinite(losses.ranking_loss)
    for parameter in selector.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
