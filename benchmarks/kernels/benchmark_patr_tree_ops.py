# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark reference and optimized PATR tree metadata operations."""

import argparse

import torch

from vllm.v1.spec_decode.dflash_tree import (
    DraftTree,
    pack_seed_tree_batch,
    select_prefix_closed_subtrees,
)


def reference_ancestor_mask(
    parents: torch.Tensor,
    depths: torch.Tensor,
    valid: torch.Tensor,
    max_depth: int,
) -> torch.Tensor:
    batch_size, seed_budget = parents.shape
    identity = torch.eye(seed_budget, dtype=torch.bool, device=parents.device)
    ancestors = identity.unsqueeze(0).expand(batch_size, -1, -1).clone()
    safe_parents = parents.clamp_min(0)
    for depth_idx in range(1, max_depth + 1):
        parent_ancestors = torch.gather(
            ancestors,
            1,
            safe_parents[..., None].expand(-1, -1, seed_budget),
        )
        rows = parent_ancestors | identity.unsqueeze(0)
        ancestors = torch.where(
            (valid & (depths == depth_idx))[..., None],
            rows,
            ancestors,
        )
    ancestors &= valid[..., None]
    ancestors[..., 0] |= ~valid
    return ancestors


def reference_pack(trees: list[DraftTree], seed_budget: int, max_depth: int):
    device = trees[0].token_ids.device
    batch_size = len(trees)
    token_ids = torch.zeros(
        (batch_size, seed_budget), dtype=torch.long, device=device
    )
    parents = torch.zeros_like(token_ids)
    depths = torch.zeros_like(token_ids)
    valid = torch.zeros(
        (batch_size, seed_budget), dtype=torch.bool, device=device
    )
    edge_lp = torch.full(
        (batch_size, seed_budget),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    ranks = torch.full(
        (batch_size, seed_budget), -1, dtype=torch.int16, device=device
    )
    for batch_idx, tree in enumerate(trees):
        node_count = tree.num_nodes
        token_ids[batch_idx, :node_count] = tree.token_ids
        parents[batch_idx, :node_count] = tree.parent_indices
        depths[batch_idx, :node_count] = tree.depth
        valid[batch_idx, :node_count] = True
        edge_lp[batch_idx, :node_count] = tree.seed_edge_logprobs
        ranks[batch_idx, :node_count] = tree.seed_ranks
    return reference_ancestor_mask(parents, depths, valid, max_depth)


def reference_select(seed, scores: torch.Tensor, final_budget: int):
    results = []
    for batch_idx in range(scores.shape[0]):
        row_scores = scores[batch_idx].masked_fill(
            ~seed.valid_mask[batch_idx],
            float("-inf"),
        )
        depths = seed.depths[batch_idx]
        order = torch.arange(scores.shape[1], device=scores.device)
        order = order[torch.argsort(depths[order], stable=True)]
        selection_scores = row_scores.clone()
        selection_scores[0] = float("inf")
        order = order[
            torch.argsort(
                selection_scores[order],
                descending=True,
                stable=True,
            )
        ]
        selected = torch.sort(order[:final_budget]).values
        parents = seed.parent_indices[batch_idx]
        old_to_new = torch.full_like(parents, -1)
        old_to_new.scatter_(
            0,
            selected,
            torch.arange(final_budget, device=scores.device),
        )
        final_parents = old_to_new[parents[selected].clamp_min(0)]
        final_parents = torch.where(
            selected == 0,
            torch.full_like(final_parents, -1),
            final_parents,
        )
        results.append((selected, final_parents))
    return results


def measure(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    device = torch.device("cuda")

    parents = torch.full((255,), -1, dtype=torch.long)
    depths = torch.zeros(255, dtype=torch.long)
    edge_logprobs = torch.zeros(255)
    ranks = torch.full((255,), -1, dtype=torch.int16)
    for node_idx in range(1, 255):
        parent = (node_idx - 1) // 7
        parents[node_idx] = parent
        depths[node_idx] = depths[parent] + 1
        edge_logprobs[node_idx] = -0.01 * (node_idx % 7 + 1)
        ranks[node_idx] = (node_idx - 1) % 7
    tree = DraftTree(
        token_ids=torch.arange(255, device=device),
        parent_indices=parents.to(device),
        depth=depths.to(device),
        num_nodes=255,
        seed_edge_logprobs=edge_logprobs.to(device),
        seed_ranks=ranks.to(device),
    )
    seed = pack_seed_tree_batch(
        [tree] * args.batch_size,
        seed_budget=255,
        max_depth=15,
    )
    scores = -torch.rand(args.batch_size, 255, device=device)
    scores[:, 0] = 0

    reference_mask = reference_ancestor_mask(
        seed.parent_indices,
        seed.depths,
        seed.valid_mask,
        15,
    )
    if not torch.equal(reference_mask, seed.ancestor_mask):
        raise SystemExit("Ancestor mask implementations differ.")
    reference_trees = reference_select(seed, scores, 64)
    optimized_trees = select_prefix_closed_subtrees(seed, scores, 64)
    for (indices, final_parents), tree_output in zip(
        reference_trees,
        optimized_trees,
    ):
        if not torch.equal(indices, tree_output.token_ids):
            raise SystemExit("Tree selection indices differ.")
        if not torch.equal(final_parents, tree_output.parent_indices):
            raise SystemExit("Tree selection parents differ.")

    reference_pack_ms = measure(
        lambda: reference_pack([tree] * args.batch_size, 255, 15),
        args.warmup,
        args.iterations,
    )
    optimized_pack_ms = measure(
        lambda: pack_seed_tree_batch(
            [tree] * args.batch_size,
            seed_budget=255,
            max_depth=15,
        ),
        args.warmup,
        args.iterations,
    )
    reference_select_ms = measure(
        lambda: reference_select(seed, scores, 64),
        args.warmup,
        args.iterations,
    )
    optimized_select_ms = measure(
        lambda: select_prefix_closed_subtrees(seed, scores, 64),
        args.warmup,
        args.iterations,
    )
    print(f"batch_size={args.batch_size}")
    print(f"reference_full_pack_ms={reference_pack_ms:.5f}")
    print(f"optimized_full_pack_ms={optimized_pack_ms:.5f}")
    print(f"reference_select_ms={reference_select_ms:.5f}")
    print(f"optimized_select_ms={optimized_select_ms:.5f}")
    print(f"select_speedup={reference_select_ms / optimized_select_ms:.3f}x")


if __name__ == "__main__":
    main()
