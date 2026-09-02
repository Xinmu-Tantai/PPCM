# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sanity checks for DFlash tree CPU optimizations.

Run after each optimization step to verify correctness is preserved:
    python -m pytest tests/v1/spec_decode/test_dflash_tree_cpu_opt.py -v
"""

import numpy as np
import torch

from vllm.v1.spec_decode.dflash_tree import (
    build_attention_bias_from_parents,
    build_block_diagonal_attention_bias,
    build_tree_from_topk,
    gpu_tree_accept,
    prune_and_regrow,
    prune_tree_by_path_logprob,
    sample_topk_from_logits,
    top2gap_fanout_caps,
    tree_accept,
)


# ---------------------------------------------------------------------------
# Step 1: tree building
# ---------------------------------------------------------------------------

def test_build_tree_from_topk_budget_and_structure():
    topk_tokens = torch.tensor([[7, 8, 3], [9, 10, 4], [11, 12, 5]])
    topk_logprobs = torch.tensor(
        [[0.0, -1.0, -2.0], [-0.5, -1.5, -2.5], [-1.0, -2.0, -3.0]]
    )
    tree = build_tree_from_topk(
        42, topk_tokens, topk_logprobs, budget=20, device=torch.device("cpu")
    )
    assert tree.num_nodes == 20
    assert tree.token_ids[0].item() == 42
    assert tree.parent_indices[0].item() == -1
    depths = tree.depth.tolist()
    assert depths[0] == 0
    assert max(depths) >= 1


def test_build_tree_deterministic():
    topk_tokens = torch.tensor([[7, 8], [9, 10]], dtype=torch.long)
    topk_logprobs = torch.tensor([[0.0, -1.0], [0.0, -1.0]])
    t1 = build_tree_from_topk(5, topk_tokens, topk_logprobs, budget=5,
                               device=torch.device("cpu"))
    t2 = build_tree_from_topk(5, topk_tokens, topk_logprobs, budget=5,
                               device=torch.device("cpu"))
    assert t1.token_ids.tolist() == t2.token_ids.tolist()
    assert t1.parent_indices.tolist() == t2.parent_indices.tolist()
    assert t1.depth.tolist() == t2.depth.tolist()


def test_depth_first_guarantees_full_spine():
    """depth_first must produce a root-to-max-depth greedy chain."""
    depth_count = 15
    width = 5
    topk_tokens = torch.arange(depth_count * width).view(depth_count, width)
    topk_logprobs = torch.zeros(depth_count, width)
    for d in range(depth_count):
        topk_logprobs[d] = torch.tensor(
            [-0.3 * (i + 1) for i in range(width)]
        )

    budget = 255
    tree = build_tree_from_topk(
        root_token=0,
        topk_tokens=topk_tokens,
        topk_logprobs=topk_logprobs,
        budget=budget,
        device=torch.device("cpu"),
        depth_first=True,
    )
    max_depth = tree.depth.max().item()
    assert max_depth == depth_count, (
        f"depth_first tree should reach depth {depth_count}, got {max_depth}"
    )
    assert tree.num_nodes == budget


def test_breadth_first_may_be_shallow():
    """breadth_first with large width can result in shallow trees."""
    depth_count = 15
    width = 5
    topk_tokens = torch.arange(depth_count * width).view(depth_count, width)
    topk_logprobs = torch.zeros(depth_count, width)
    for d in range(depth_count):
        topk_logprobs[d] = torch.tensor(
            [-0.3 * (i + 1) for i in range(width)]
        )

    budget = 255
    tree = build_tree_from_topk(
        root_token=0,
        topk_tokens=topk_tokens,
        topk_logprobs=topk_logprobs,
        budget=budget,
        device=torch.device("cpu"),
        depth_first=False,
    )
    max_depth = tree.depth.max().item()
    assert max_depth < depth_count, (
        f"breadth_first with width={width}, budget={budget} should not "
        f"reach full depth {depth_count}, got {max_depth}"
    )
    assert tree.num_nodes == budget


def test_depth_first_spine_contains_top1_tokens():
    """The greedy spine in depth_first should use top-1 token at each depth."""
    depth_count = 5
    width = 3
    topk_tokens = torch.tensor([
        [10, 11, 12],
        [20, 21, 22],
        [30, 31, 32],
        [40, 41, 42],
        [50, 51, 52],
    ])
    topk_logprobs = torch.tensor([
        [-0.1, -1.0, -2.0],
        [-0.2, -1.0, -2.0],
        [-0.3, -1.0, -2.0],
        [-0.4, -1.0, -2.0],
        [-0.5, -1.0, -2.0],
    ])
    tree = build_tree_from_topk(
        root_token=99,
        topk_tokens=topk_tokens,
        topk_logprobs=topk_logprobs,
        budget=50,
        device=torch.device("cpu"),
        depth_first=True,
    )
    tokens = tree.token_ids.tolist()
    parents = tree.parent_indices.tolist()
    depths = tree.depth.tolist()

    spine = {0: 99}
    node = 0
    for d in range(1, depth_count + 1):
        for i in range(tree.num_nodes):
            if parents[i] == node and depths[i] == d:
                if tokens[i] == topk_tokens[d - 1, 0].item():
                    spine[d] = tokens[i]
                    node = i
                    break
    assert len(spine) == depth_count + 1, (
        f"Expected spine of length {depth_count + 1}, got {len(spine)}"
    )
    expected = [99, 10, 20, 30, 40, 50]
    assert list(spine.values()) == expected


def test_path_logprob_pruning_selects_prefix_closed_top_nodes():
    """Pruning ranks complete root-to-node paths, not individual edges."""
    tree = build_tree_from_topk(
        root_token=99,
        topk_tokens=torch.tensor([
            [10, 11],
            [20, 21],
            [30, 31],
        ]),
        topk_logprobs=torch.tensor([
            [-0.1, -0.9],
            [-0.1, -2.0],
            [-0.1, -3.0],
        ]),
        budget=6,
        device=torch.device("cpu"),
        depth_first=True,
    )

    pruned = prune_tree_by_path_logprob(
        tree=tree,
        topk_tokens=torch.tensor([
            [10, 11],
            [20, 21],
            [30, 31],
        ]),
        topk_logprobs=torch.tensor([
            [-0.1, -0.9],
            [-0.1, -2.0],
            [-0.1, -3.0],
        ]),
        budget=4,
        device=torch.device("cpu"),
    )

    assert pruned.num_nodes == 4
    assert pruned.token_ids.tolist() == [99, 10, 20, 30]
    assert pruned.parent_indices.tolist() == [-1, 0, 1, 2]
    assert pruned.depth.tolist() == [0, 1, 2, 3]

    direct = build_tree_from_topk(
        root_token=99,
        topk_tokens=torch.tensor([
            [10, 11],
            [20, 21],
            [30, 31],
        ]),
        topk_logprobs=torch.tensor([
            [-0.1, -0.9],
            [-0.1, -2.0],
            [-0.1, -3.0],
        ]),
        budget=6,
        path_prune_budget=4,
        device=torch.device("cpu"),
        depth_first=True,
    )
    assert direct.token_ids.tolist() == pruned.token_ids.tolist()
    assert direct.parent_indices.tolist() == pruned.parent_indices.tolist()
    assert direct.depth.tolist() == pruned.depth.tolist()


def test_path_logprob_pruning_uses_ancestor_first_ties():
    """Zero-logprob ties must keep shallow ancestors before descendants."""
    topk_tokens = torch.tensor([[10, 11], [20, 21]])
    topk_logprobs = torch.zeros(2, 2)
    tree = build_tree_from_topk(
        99,
        topk_tokens,
        topk_logprobs,
        budget=5,
        device=torch.device("cpu"),
        depth_first=False,
    )
    pruned = prune_tree_by_path_logprob(
        tree,
        topk_tokens,
        topk_logprobs,
        budget=3,
        device=torch.device("cpu"),
    )

    assert pruned.num_nodes == 3
    assert pruned.depth.tolist() == [0, 1, 1]
    assert pruned.parent_indices.tolist() == [-1, 0, 0]


def test_top2gap_fanout_caps_narrow_decisive_depths():
    """Large top-2 gaps collapse fanout toward a chain."""
    caps = top2gap_fanout_caps([
        [0.0, -8.0, -9.0, -10.0],
        [-0.1, -6.0, -7.0, -8.0],
    ])
    assert caps == [1, 1]


def test_top2gap_fanout_caps_keep_close_runner_up():
    """Small top-2 gaps keep wider fanout where rank 2 is plausible."""
    caps = top2gap_fanout_caps([
        [0.0, -0.05, -4.0, -5.0],
    ])
    assert caps[0] > 1


def test_top2gap_fanout_builds_chain_for_decisive_logits():
    topk_tokens = torch.tensor([
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ])
    topk_logprobs = torch.tensor([
        [0.0, -8.0, -9.0, -10.0],
        [0.0, -8.0, -9.0, -10.0],
        [0.0, -8.0, -9.0, -10.0],
    ])
    tree = build_tree_from_topk(
        root_token=99,
        topk_tokens=topk_tokens,
        topk_logprobs=topk_logprobs,
        budget=40,
        device=torch.device("cpu"),
        score_mode="top2gap_fanout",
    )
    assert tree.num_nodes == 4
    assert tree.token_ids.tolist() == [99, 10, 20, 30]
    assert tree.parent_indices.tolist() == [-1, 0, 1, 2]
    assert tree.depth.tolist() == [0, 1, 2, 3]


def test_sample_topk_from_logits_shape():
    logits = torch.randn(10, 1000)
    lp, tok = sample_topk_from_logits(logits, 5)
    assert lp.shape == (10, 5)
    assert tok.shape == (10, 5)
    assert (lp[:, 0] >= lp[:, -1]).all()


def test_prune_and_regrow_preserves_root_and_budget():
    topk_tokens = torch.tensor([[7, 8], [9, 10]], dtype=torch.long)
    topk_logprobs = torch.tensor([[0.0, -1.0], [0.0, -1.0]])
    tree = build_tree_from_topk(5, topk_tokens, topk_logprobs, budget=5,
                                 device=torch.device("cpu"))
    cond_logits = torch.randn(2, 20)
    new_tree = prune_and_regrow(tree, cond_logits, block_size=3,
                                 tree_width=2, budget=5)
    assert new_tree.num_nodes <= 5
    assert new_tree.token_ids[0].item() == 5
    assert new_tree.parent_indices[0].item() == -1


# ---------------------------------------------------------------------------
# Step 2: attention bias from parents
# ---------------------------------------------------------------------------

def test_attention_bias_from_parents_known_tree():
    # root(0) -> A(1), B(2);  A -> C(3), D(4)
    parents = torch.tensor([-1, 0, 0, 1, 1], dtype=torch.int64)
    bias = build_attention_bias_from_parents(
        parents, dtype=torch.float32, device=torch.device("cpu")
    )
    assert bias.shape == (5, 5)
    # C(3) can see root(0), A(1), and itself
    assert bias[3, 0].item() == 0.0
    assert bias[3, 1].item() == 0.0
    assert bias[3, 3].item() == 0.0
    # C(3) cannot see B(2) or D(4)
    assert bias[3, 2].item() < -1e30
    assert bias[3, 4].item() < -1e30


def test_attention_bias_root_visible_to_all():
    parents = torch.tensor([-1, 0, 0, 1, 2], dtype=torch.int64)
    bias = build_attention_bias_from_parents(
        parents, dtype=torch.float32, device=torch.device("cpu")
    )
    for row in range(5):
        assert bias[row, 0].item() == 0.0


# ---------------------------------------------------------------------------
# Step 3: tree accept
# ---------------------------------------------------------------------------

def test_tree_accept_all_rejected():
    tree_tokens = torch.tensor([5, 99, 98], dtype=torch.int64)
    parent_indices = torch.tensor([-1, 0, 0], dtype=torch.int64)
    logits = torch.zeros((3, 100), dtype=torch.float32)
    logits[0, 50] = 10.0
    path, acc_len, correction = tree_accept(
        tree_tokens, parent_indices, logits, temperature=0.0
    )
    assert acc_len == 0
    assert path == [0]
    assert correction == 50


def test_tree_accept_full_chain():
    tree_tokens = torch.tensor([5, 7, 9], dtype=torch.int64)
    parent_indices = torch.tensor([-1, 0, 1], dtype=torch.int64)
    logits = torch.full((3, 100), -1000.0, dtype=torch.float32)
    logits[0, 7] = 1.0
    logits[1, 9] = 1.0
    logits[2, 42] = 1.0
    path, acc_len, correction = tree_accept(
        tree_tokens, parent_indices, logits, temperature=0.0
    )
    assert acc_len == 2
    assert path == [0, 1, 2]
    assert correction == 42


def test_tree_accept_picks_longer_branch():
    # root -> A(1), B(2);  A -> C(3);  B -> D(4), E(5)
    tree_tokens = torch.tensor([5, 7, 8, 9, 10, 11], dtype=torch.int64)
    parent_indices = torch.tensor([-1, 0, 0, 1, 2, 2], dtype=torch.int64)
    logits = torch.full((6, 100), -1000.0, dtype=torch.float32)
    logits[0, 8] = 1.0   # root -> B(2) accepted
    logits[2, 10] = 1.0  # B -> D(4) accepted
    logits[4, 50] = 1.0  # correction
    path, acc_len, correction = tree_accept(
        tree_tokens, parent_indices, logits, temperature=0.0
    )
    assert acc_len == 2
    assert path == [0, 2, 4]
    assert correction == 50


# ---------------------------------------------------------------------------
# Step 3b: gpu_tree_accept — parity with CPU tree_accept
# ---------------------------------------------------------------------------


def _make_greedy_targets(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def _depths_from_parents(parent_indices: torch.Tensor) -> torch.Tensor:
    n = parent_indices.shape[0]
    depths = torch.zeros(n, dtype=torch.long, device=parent_indices.device)
    for i in range(1, n):
        depths[i] = depths[parent_indices[i].clamp(min=0)] + 1
    return depths


def test_gpu_tree_accept_all_rejected():
    tree_tokens = torch.tensor([5, 99, 98], dtype=torch.int64)
    parent_indices = torch.tensor([-1, 0, 0], dtype=torch.int64)
    logits = torch.zeros((3, 100), dtype=torch.float32)
    logits[0, 50] = 10.0
    greedy = _make_greedy_targets(logits)
    depths = _depths_from_parents(parent_indices)

    path, acc_len, correction = gpu_tree_accept(
        tree_tokens, greedy, parent_indices, depths, max_depth=3,
    )
    assert acc_len == 0
    assert path.tolist() == [0]
    assert correction.item() == 50


def test_gpu_tree_accept_full_chain():
    tree_tokens = torch.tensor([5, 7, 9], dtype=torch.int64)
    parent_indices = torch.tensor([-1, 0, 1], dtype=torch.int64)
    logits = torch.full((3, 100), -1000.0, dtype=torch.float32)
    logits[0, 7] = 1.0
    logits[1, 9] = 1.0
    logits[2, 42] = 1.0
    greedy = _make_greedy_targets(logits)
    depths = _depths_from_parents(parent_indices)

    path, acc_len, correction = gpu_tree_accept(
        tree_tokens, greedy, parent_indices, depths, max_depth=3,
    )
    assert acc_len == 2
    assert path.tolist() == [0, 1, 2]
    assert correction.item() == 42


def test_gpu_tree_accept_picks_longer_branch():
    tree_tokens = torch.tensor([5, 7, 8, 9, 10, 11], dtype=torch.int64)
    parent_indices = torch.tensor([-1, 0, 0, 1, 2, 2], dtype=torch.int64)
    logits = torch.full((6, 100), -1000.0, dtype=torch.float32)
    logits[0, 8] = 1.0
    logits[2, 10] = 1.0
    logits[4, 50] = 1.0
    greedy = _make_greedy_targets(logits)
    depths = _depths_from_parents(parent_indices)

    path, acc_len, correction = gpu_tree_accept(
        tree_tokens, greedy, parent_indices, depths, max_depth=5,
    )
    assert acc_len == 2
    assert path.tolist() == [0, 2, 4]
    assert correction.item() == 50


def test_gpu_tree_accept_matches_cpu_tree_accept():
    """Reference parity: gpu_tree_accept produces same results as tree_accept."""
    tree_tokens = torch.tensor([5, 7, 8, 9, 10], dtype=torch.int64)
    parent_indices = torch.tensor([-1, 0, 0, 1, 1], dtype=torch.int64)
    logits = torch.full((5, 32), -1000.0, dtype=torch.float32)
    logits[0, 7] = 1.0
    logits[1, 9] = 1.0
    logits[3, 13] = 1.0

    cpu_path, cpu_len, cpu_correction = tree_accept(
        tree_tokens, parent_indices, logits, temperature=0.0,
    )

    greedy = _make_greedy_targets(logits)
    depths = _depths_from_parents(parent_indices)
    gpu_path, gpu_len, gpu_correction = gpu_tree_accept(
        tree_tokens, greedy, parent_indices, depths, max_depth=5,
    )

    assert gpu_path.tolist() == cpu_path
    assert gpu_len == cpu_len
    assert gpu_correction.item() == cpu_correction


def test_gpu_tree_accept_single_node():
    tree_tokens = torch.tensor([42], dtype=torch.int64)
    parent_indices = torch.tensor([-1], dtype=torch.int64)
    greedy = torch.tensor([99], dtype=torch.int64)
    depths = torch.tensor([0], dtype=torch.int64)

    path, acc_len, correction = gpu_tree_accept(
        tree_tokens, greedy, parent_indices, depths, max_depth=1,
    )
    assert acc_len == 0
    assert path.tolist() == [0]
    assert correction.item() == 99


# ---------------------------------------------------------------------------
# Step 4 & 6: block diagonal bias
# ---------------------------------------------------------------------------

def test_block_diagonal_bias_structure():
    b1 = torch.zeros(3, 3, dtype=torch.float32)
    b2 = torch.zeros(4, 4, dtype=torch.float32)
    combined = build_block_diagonal_attention_bias(
        [b1, b2], dtype=torch.float32, device=torch.device("cpu")
    )
    assert combined.shape == (7, 7)
    # Diagonal blocks are 0
    assert combined[0, 0].item() == 0.0
    assert combined[3, 3].item() == 0.0
    # Off-diagonal blocks are -inf
    assert combined[0, 3].item() < -1e30
    assert combined[3, 0].item() < -1e30


# ---------------------------------------------------------------------------
# Regression: all functions accept both CPU and GPU-like tensors
# ---------------------------------------------------------------------------

def test_all_functions_work_on_cpu_tensors():
    """End-to-end: build tree, build bias, accept — all on CPU."""
    topk_tokens = torch.tensor([[7, 8], [9, 10]], dtype=torch.long)
    topk_logprobs = torch.tensor([[0.0, -1.0], [-0.5, -1.5]])
    tree = build_tree_from_topk(5, topk_tokens, topk_logprobs, budget=5,
                                 device=torch.device("cpu"))

    parents_full = tree.parent_indices
    bias = build_attention_bias_from_parents(
        parents_full, dtype=torch.float32, device=torch.device("cpu")
    )
    assert bias.shape[0] == tree.num_nodes

    logits = torch.randn(tree.num_nodes, 20)
    path, acc_len, correction = tree_accept(
        tree.token_ids, tree.parent_indices, logits, temperature=0.0
    )
    assert len(path) >= 1
    assert path[0] == 0
    assert 0 <= correction < 20
