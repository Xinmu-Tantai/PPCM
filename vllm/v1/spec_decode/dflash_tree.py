# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
import torch


def compute_per_depth_entropy(logits: torch.Tensor) -> list[float]:
    """Compute entropy of each depth's distribution.

    *logits* has shape ``(depth, vocab)`` (or ``(batch, depth, vocab)``
    for batched usage — only the last two dims are used here).
    Returns a Python list of length ``depth``.
    """
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    return ent.tolist()


def _node_expand_score(
    cum_lp: float,
    depth: int,
    score_mode: str,
    per_depth_entropy: list[float] | None,
    hybrid_alpha: float,
    parent_expansion_entropy: float | None = None,
) -> float:
    """Score a node for heap-based expansion.

    Higher returned value = higher expansion priority (the heap stores
    negated scores so that ``heapq`` pops the best first).

    For entropy/hybrid modes the relevant entropy is the distribution that
    *produced* this node (its parent's expansion), not the distribution at
    the node's own depth.  When ``parent_expansion_entropy`` is provided it
    is used directly; otherwise we fall back to ``per_depth_entropy[depth]``
    which is correct for root / spine nodes that were not created by heap
    expansion.
    """
    if score_mode == "entropy":
        if parent_expansion_entropy is not None:
            return parent_expansion_entropy
        if per_depth_entropy is not None and depth < len(per_depth_entropy):
            return per_depth_entropy[depth]
        return 0.0
    if score_mode == "hybrid":
        if parent_expansion_entropy is not None:
            ent = parent_expansion_entropy
        elif per_depth_entropy is not None and depth < len(per_depth_entropy):
            ent = per_depth_entropy[depth]
        else:
            ent = 0.0
        return cum_lp + hybrid_alpha * ent
    # "accum_logp" (default)
    return cum_lp


@dataclass
class DraftTree:
    """GPU-resident tree representation (legacy interface).

    Prefer ``DraftTreeCPU`` for the proposer hot-path to avoid
    GPU<->CPU round-trips.  This class is still used by the verifier
    and tests that pass GPU tensors.
    """
    # Root-inclusive BFS layout.
    token_ids: torch.Tensor
    parent_indices: torch.Tensor
    depth: torch.Tensor
    num_nodes: int
    # Optional PATR seed metadata. Root uses edge log-probability 0 and rank -1.
    seed_edge_logprobs: torch.Tensor | None = None
    seed_ranks: torch.Tensor | None = None

    def paths(self) -> list[list[int]]:
        parents = self.parent_indices.tolist()
        children: list[list[int]] = [[] for _ in range(self.num_nodes)]
        for idx in range(1, self.num_nodes):
            children[parents[idx]].append(idx)
        leaves = [idx for idx in range(self.num_nodes) if not children[idx]]
        paths: list[list[int]] = []
        for leaf in leaves:
            path: list[int] = []
            node = leaf
            while node >= 0:
                path.append(node)
                node = parents[node]
            paths.append(path[::-1])
        return paths

    def longest_path(self) -> list[int]:
        if self.num_nodes == 0:
            return []
        if self.num_nodes == 1:
            return [0]

        parents = self.parent_indices.tolist()
        depths = self.depth.tolist()

        child_count = [0] * self.num_nodes
        for idx in range(1, self.num_nodes):
            child_count[parents[idx]] += 1

        best_leaf = -1
        best_depth = -1
        for idx in range(self.num_nodes):
            if child_count[idx] == 0 and depths[idx] > best_depth:
                best_depth = depths[idx]
                best_leaf = idx

        if best_leaf < 0:
            return [0]

        path: list[int] = []
        node = best_leaf
        while node >= 0:
            path.append(node)
            node = parents[node]
        return path[::-1]


@dataclass
class DraftTreeCPU:
    """CPU-only tree used during the proposer hot-path.

    All fields are plain Python lists — no GPU tensors are created until
    the tree is finalised via :meth:`to_gpu`.
    """
    token_ids: list[int]
    parent_indices: list[int]
    depths: list[int]
    num_nodes: int
    seed_edge_logprobs: list[float] | None = None
    seed_ranks: list[int] | None = None

    def longest_path(self) -> list[int]:
        if self.num_nodes == 0:
            return []
        if self.num_nodes == 1:
            return [0]

        parents = self.parent_indices
        depths = self.depths

        child_count = [0] * self.num_nodes
        for idx in range(1, self.num_nodes):
            child_count[parents[idx]] += 1

        best_leaf = -1
        best_depth = -1
        for idx in range(self.num_nodes):
            if child_count[idx] == 0 and depths[idx] > best_depth:
                best_depth = depths[idx]
                best_leaf = idx

        if best_leaf < 0:
            return [0]

        path: list[int] = []
        node = best_leaf
        while node >= 0:
            path.append(node)
            node = parents[node]
        return path[::-1]

    def to_gpu(self, device: torch.device) -> DraftTree:
        return DraftTree(
            token_ids=torch.tensor(self.token_ids, dtype=torch.long,
                                   device=device),
            parent_indices=torch.tensor(self.parent_indices, dtype=torch.long,
                                        device=device),
            depth=torch.tensor(self.depths, dtype=torch.long, device=device),
            num_nodes=self.num_nodes,
            seed_edge_logprobs=(
                torch.tensor(
                    self.seed_edge_logprobs, dtype=torch.float32, device=device,
                )
                if self.seed_edge_logprobs is not None else None
            ),
            seed_ranks=(
                torch.tensor(self.seed_ranks, dtype=torch.int16, device=device)
                if self.seed_ranks is not None else None
            ),
        )


def pad_draft_tree_to_budget(tree: DraftTree, budget: int) -> DraftTree:
    """Pad a semantic tree to a larger physical verifier row count.

    Extra nodes are attached as children of root so they are not ancestors of
    any real node. Root therefore does not attend to them, and they cannot
    enter the accepted path unless the root greedy token is 0.
    """
    pad_count = int(budget) - int(tree.num_nodes)
    if pad_count <= 0:
        return tree

    def _pad(tensor: torch.Tensor, fill: int | float) -> torch.Tensor:
        extra = torch.full(
            (pad_count,),
            fill,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat((tensor, extra))

    return DraftTree(
        token_ids=_pad(tree.token_ids, 0),
        parent_indices=_pad(tree.parent_indices, 0),
        depth=_pad(tree.depth, 1),
        num_nodes=int(budget),
        seed_edge_logprobs=(
            None
            if tree.seed_edge_logprobs is None
            else _pad(tree.seed_edge_logprobs, 0.0)
        ),
        seed_ranks=(
            None if tree.seed_ranks is None else _pad(tree.seed_ranks, -1)
        ),
    )


def select_best_rescored_chain(
    tree: DraftTree,
    root_residuals: torch.Tensor,
    transition_residuals: torch.Tensor,
    *,
    max_depth: int = 7,
) -> DraftTree:
    """Rescore a pruned Top-K tree and return its best root-to-leaf chain.

    Native conditional log-probabilities remain the base score.  The learned
    lattice values are additive residuals indexed by the surviving node's
    original Top-K rank.  A complete ``max_depth`` leaf is preferred; if
    pruning did not retain one, selection falls back to leaves at the deepest
    available depth.
    """
    if tree.seed_edge_logprobs is None or tree.seed_ranks is None:
        raise ValueError("Rescoring requires native edge log-probs and Top-K ranks.")
    if root_residuals.ndim != 1:
        raise ValueError("root_residuals must have shape [K].")
    top_k = int(root_residuals.shape[0])
    expected_transition = (max_depth - 1, top_k, top_k)
    if tuple(transition_residuals.shape) != expected_transition:
        raise ValueError(
            "transition_residuals must have shape "
            f"[{max_depth - 1}, {top_k}, {top_k}]."
        )

    root_residuals = root_residuals.float()
    transition_residuals = transition_residuals.float()
    parents_tensor = tree.parent_indices.to(torch.long)
    safe_parents = parents_tensor.clamp_min(0)
    depths_tensor = tree.depth.to(torch.long)
    ranks = tree.seed_ranks.to(torch.long)
    child_ranks = ranks.clamp(0, top_k - 1)
    parent_ranks = ranks[safe_parents].clamp(0, top_k - 1)
    transition_depths = (depths_tensor - 2).clamp(0, max_depth - 2)
    transition_values = transition_residuals[
        transition_depths, parent_ranks, child_ranks
    ]
    edge_residuals = torch.where(
        depths_tensor == 1,
        root_residuals[child_ranks],
        transition_values,
    )
    is_root = depths_tensor == 0
    edge_residuals = torch.where(
        is_root, torch.zeros_like(edge_residuals), edge_residuals
    )
    edge_scores = tree.seed_edge_logprobs.float() + edge_residuals
    edge_scores = torch.where(is_root, torch.zeros_like(edge_scores), edge_scores)

    # Parent-before-child plus a known maximum depth lets us accumulate the
    # entire tree in max_depth fixed-shape vector steps instead of one GPU
    # sync per node.
    path_scores = torch.zeros_like(edge_scores)
    for depth_idx in range(1, max_depth + 1):
        at_depth = depths_tensor == depth_idx
        path_scores = torch.where(
            at_depth,
            path_scores[safe_parents] + edge_scores,
            path_scores,
        )

    child_count = torch.zeros(
        tree.num_nodes, dtype=torch.int32, device=tree.token_ids.device
    )
    if tree.num_nodes > 1:
        child_count.scatter_add_(
            0,
            parents_tensor[1:],
            torch.ones_like(parents_tensor[1:], dtype=torch.int32),
        )
    leaf_mask = child_count == 0
    deepest = depths_tensor.masked_fill(~leaf_mask, -1).max()
    eligible = leaf_mask & (depths_tensor == deepest)
    best_leaf = int(path_scores.masked_fill(~eligible, float("-inf")).argmax().item())

    selected: list[int] = []
    parents = parents_tensor.tolist()
    node = best_leaf
    while node >= 0:
        selected.append(node)
        node = parents[node]
    selected.reverse()
    selected_tensor = torch.tensor(selected, device=tree.token_ids.device)
    length = len(selected)
    return DraftTree(
        token_ids=tree.token_ids[selected_tensor],
        parent_indices=torch.arange(
            -1, length - 1, dtype=tree.parent_indices.dtype,
            device=tree.parent_indices.device,
        ),
        depth=torch.arange(
            length, dtype=tree.depth.dtype, device=tree.depth.device,
        ),
        num_nodes=length,
        seed_edge_logprobs=tree.seed_edge_logprobs[selected_tensor],
        seed_ranks=tree.seed_ranks[selected_tensor],
    )


@dataclass
class SeedTreeBatch:
    """Fixed-shape, root-inclusive PATR input for a batch of seed trees."""

    token_ids: torch.Tensor
    parent_indices: torch.Tensor
    depths: torch.Tensor
    valid_mask: torch.Tensor
    seed_edge_logprobs: torch.Tensor
    seed_ranks: torch.Tensor
    ancestor_mask: torch.Tensor | None
    ancestor_indices: torch.Tensor | None = None
    ancestor_valid_mask: torch.Tensor | None = None


def compute_tree_budget(
    block_size: int,
    tree_width: int,
    max_budget: int | None = None,
) -> int:
    if tree_width <= 1:
        return block_size
    full_tree = (tree_width**block_size - 1) // (tree_width - 1)
    if max_budget is not None and max_budget > 0:
        return min(full_tree, max_budget)
    return full_tree


def sample_topk_from_logits(
    logits: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.topk(log_probs, k, dim=-1)


def batch_topk_to_cpu(
    logits: torch.Tensor,
    k: int,
) -> tuple[list, list, list]:
    """Batched log_softmax + topk on GPU, then one `.tolist()` each.

    *logits* may be 2-D ``(depth, vocab)`` or 3-D ``(batch, depth, vocab)``.

    Returns ``(full_lp_cpu, topk_lp_cpu, topk_tok_cpu)`` where each is a
    nested Python list (outer dim = batch when 3-D).
    """
    lp = torch.log_softmax(logits, dim=-1)
    topk_lp, topk_tok = torch.topk(lp, k, dim=-1)
    return lp.tolist(), topk_lp.tolist(), topk_tok.tolist()


def _build_tree_breadth_first(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
    score_mode: str = "accum_logp",
    per_depth_entropy: list[float] | None = None,
    hybrid_alpha: float = 1.0,
    fanout_caps: list[int] | None = None,
) -> DraftTreeCPU:
    """Heap expansion (breadth-biased) using the chosen scoring strategy.

    With large width and limited budget the tree may not reach full depth.
    """
    depth_count = len(topk_tokens_cpu)
    width = len(topk_tokens_cpu[0]) if depth_count else 0

    tokens_list = [root_token]
    parents_list = [-1]
    depths_list = [0]
    edge_logprobs = [0.0]
    ranks = [-1]
    num_nodes = 1

    root_score = _node_expand_score(
        0.0, 0, score_mode, per_depth_entropy, hybrid_alpha,
    )
    counter = 0
    heap: list[tuple[float, int, int]] = [(-root_score, counter, 0)]
    cum_lp_at: list[float] = [0.0]
    while heap and num_nodes < budget:
        _, _, node_idx = heapq.heappop(heap)
        depth = depths_list[node_idx]
        if depth >= depth_count:
            continue
        fanout_cap = fanout_caps[depth] if fanout_caps is not None else width
        children_to_add = min(fanout_cap, width, budget - num_nodes)
        row_tokens = topk_tokens_cpu[depth]
        row_logprobs = topk_logprobs_cpu[depth]
        expansion_ent = (
            per_depth_entropy[depth]
            if per_depth_entropy is not None and depth < len(per_depth_entropy)
            else None
        )
        for child_idx in range(children_to_add):
            tokens_list.append(row_tokens[child_idx])
            edge_logprobs.append(float(row_logprobs[child_idx]))
            ranks.append(child_idx)
            child_cum_lp = cum_lp_at[node_idx] + row_logprobs[child_idx]
            parents_list.append(node_idx)
            child_depth = depth + 1
            depths_list.append(child_depth)
            cum_lp_at.append(child_cum_lp)
            score = _node_expand_score(
                child_cum_lp, child_depth, score_mode,
                per_depth_entropy, hybrid_alpha,
                parent_expansion_entropy=expansion_ent,
            )
            counter += 1
            heapq.heappush(heap, (-score, counter, num_nodes))
            num_nodes += 1

    return DraftTreeCPU(
        token_ids=tokens_list,
        parent_indices=parents_list,
        depths=depths_list,
        num_nodes=num_nodes,
        seed_edge_logprobs=edge_logprobs,
        seed_ranks=ranks,
    )


def _top2gap_sigmoid_cap(
    gap: float,
    width: int,
    beta: float = 2.0,
    g_0: float = 1.0,
) -> int:
    """Return fanout cap from rank-1/rank-2 logprob gap.

    Large gap means the drafter is decisive, so keep a narrow fanout. Small
    gap means rank 1 and 2 are close, so spend more tree budget at that depth.
    """
    arg = -beta * (gap - g_0)
    if arg >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-arg))
    else:
        exp_arg = math.exp(arg)
        sigmoid = exp_arg / (1.0 + exp_arg)
    return max(1, int(round(width * sigmoid)))


def top2gap_fanout_caps(
    topk_logprobs_cpu: list[list[float]],
    beta: float = 2.0,
    g_0: float = 1.0,
) -> list[int]:
    """Compute per-depth fanout caps for ``top2gap_fanout`` mode."""
    if not topk_logprobs_cpu:
        return []
    width = len(topk_logprobs_cpu[0])
    if width < 2:
        return [1] * len(topk_logprobs_cpu)
    return [
        _top2gap_sigmoid_cap(row[0] - row[1], width, beta=beta, g_0=g_0)
        for row in topk_logprobs_cpu
    ]


def _build_tree_with_per_depth_cap(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
    fanout_caps: list[int],
) -> DraftTreeCPU:
    """Heap expansion using a per-depth fanout cap."""
    depth_count = len(topk_tokens_cpu)
    width = len(topk_tokens_cpu[0]) if depth_count else 0

    tokens_list: list[int] = [root_token]
    parents_list: list[int] = [-1]
    depths_list: list[int] = [0]
    edge_logprobs: list[float] = [0.0]
    ranks: list[int] = [-1]
    num_nodes = 1

    counter = 0
    heap: list[tuple[float, int, int]] = [(0.0, counter, 0)]
    cum_lp_at: list[float] = [0.0]

    while heap and num_nodes < budget:
        neg_cum_lp, _, node_idx = heapq.heappop(heap)
        depth = depths_list[node_idx]
        if depth >= depth_count:
            continue
        cap = fanout_caps[depth] if depth < len(fanout_caps) else width
        children_to_add = min(cap, width, budget - num_nodes)
        row_tokens = topk_tokens_cpu[depth]
        row_logprobs = topk_logprobs_cpu[depth]
        for child_idx in range(children_to_add):
            child_cum_lp = -neg_cum_lp + row_logprobs[child_idx]
            tokens_list.append(row_tokens[child_idx])
            parents_list.append(node_idx)
            depths_list.append(depth + 1)
            edge_logprobs.append(float(row_logprobs[child_idx]))
            ranks.append(child_idx)
            cum_lp_at.append(child_cum_lp)
            counter += 1
            heapq.heappush(heap, (-child_cum_lp, counter, num_nodes))
            num_nodes += 1

    return DraftTreeCPU(
        token_ids=tokens_list,
        parent_indices=parents_list,
        depths=depths_list,
        num_nodes=num_nodes,
        seed_edge_logprobs=edge_logprobs,
        seed_ranks=ranks,
    )


def _build_tree_depth_first(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
    score_mode: str = "accum_logp",
    per_depth_entropy: list[float] | None = None,
    hybrid_alpha: float = 1.0,
    fanout_caps: list[int] | None = None,
) -> DraftTreeCPU:
    """Depth-first tree construction that guarantees the greedy spine.

    Phase 1: pre-allocate the top-1 (greedy) chain from root to full
    depth, ensuring tree acceptance >= linear-chain acceptance.

    Phase 2: spend remaining budget on side branches via heap expansion
    using the chosen *score_mode*, skipping the top-1 child for spine
    nodes (already present).
    """
    depth_count = len(topk_tokens_cpu)
    width = len(topk_tokens_cpu[0]) if depth_count else 0

    tokens_list: list[int] = [root_token]
    parents_list: list[int] = [-1]
    depths_list: list[int] = [0]
    edge_logprobs: list[float] = [0.0]
    ranks: list[int] = [-1]
    num_nodes = 1

    spine_set: set[int] = {0}
    spine_cum_lp = 0.0
    prev_idx = 0
    for d in range(depth_count):
        if num_nodes >= budget:
            break
        tokens_list.append(topk_tokens_cpu[d][0])
        edge_logprobs.append(float(topk_logprobs_cpu[d][0]))
        ranks.append(0)
        spine_cum_lp += topk_logprobs_cpu[d][0]
        parents_list.append(prev_idx)
        depths_list.append(d + 1)
        spine_set.add(num_nodes)
        prev_idx = num_nodes
        num_nodes += 1

    counter = 0
    heap: list[tuple[float, int, int]] = []
    cum_lp_at: list[float] = [0.0] * num_nodes
    for idx in range(num_nodes):
        d = depths_list[idx]
        if d > 0:
            cum_lp_at[idx] = (
                cum_lp_at[parents_list[idx]] + topk_logprobs_cpu[d - 1][0]
            )
        if d < depth_count:
            score = _node_expand_score(
                cum_lp_at[idx], d, score_mode,
                per_depth_entropy, hybrid_alpha,
            )
            counter += 1
            heapq.heappush(heap, (-score, counter, idx))

    while heap and num_nodes < budget:
        _, _, node_idx = heapq.heappop(heap)
        depth = depths_list[node_idx]
        if depth >= depth_count:
            continue
        start_child = 1 if node_idx in spine_set else 0
        fanout_cap = fanout_caps[depth] if fanout_caps is not None else width
        capped_children = max(fanout_cap - start_child, 0)
        children_to_add = min(
            capped_children, width - start_child, budget - num_nodes,
        )
        if children_to_add <= 0:
            continue
        row_tokens = topk_tokens_cpu[depth]
        row_logprobs = topk_logprobs_cpu[depth]
        expansion_ent = (
            per_depth_entropy[depth]
            if per_depth_entropy is not None and depth < len(per_depth_entropy)
            else None
        )
        for child_idx in range(start_child, start_child + children_to_add):
            tokens_list.append(row_tokens[child_idx])
            edge_logprobs.append(float(row_logprobs[child_idx]))
            ranks.append(child_idx)
            child_cum_lp = cum_lp_at[node_idx] + row_logprobs[child_idx]
            parents_list.append(node_idx)
            child_depth = depth + 1
            depths_list.append(child_depth)
            cum_lp_at.append(child_cum_lp)
            score = _node_expand_score(
                child_cum_lp, child_depth, score_mode,
                per_depth_entropy, hybrid_alpha,
                parent_expansion_entropy=expansion_ent,
            )
            counter += 1
            heapq.heappush(heap, (-score, counter, num_nodes))
            num_nodes += 1

    return DraftTreeCPU(
        token_ids=tokens_list,
        parent_indices=parents_list,
        depths=depths_list,
        num_nodes=num_nodes,
        seed_edge_logprobs=edge_logprobs,
        seed_ranks=ranks,
    )


def _build_tree_opt_prefix(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
) -> DraftTreeCPU:
    """Provably optimal tree under factorized draft marginals (DDTree / OPT-Tree).

    Each heap entry is a *rank tuple* representing a root-to-node path.
    Popping always yields the globally highest prefix-probability node.
    Two successors are pushed per pop:
      - next sibling  (same depth, next-best token)
      - first child   (one depth deeper, best token)
    This adds exactly one node per pop and produces the top-B prefixes
    without enumerating the exponential prefix space.
    """
    depth_count = len(topk_tokens_cpu)
    K = min(budget, len(topk_tokens_cpu[0])) if depth_count else 0
    if K == 0 or budget <= 1:
        return DraftTreeCPU(
            token_ids=[root_token],
            parent_indices=[-1],
            depths=[0],
            num_nodes=1,
        )

    tokens_list: list[int] = [root_token]
    parents_list: list[int] = [-1]
    depths_list: list[int] = [0]
    num_nodes = 1

    # Map rank-tuples to their node index so we can look up parent indices.
    # Key: tuple of ranks (length = depth), Value: node index.
    rank_to_idx: dict[tuple[int, ...], int] = {(): 0}

    counter = 0
    # Heap entries: (-score, counter, rank_tuple)
    # Start with rank (0,) = best token at depth 0
    init_score = topk_logprobs_cpu[0][0]
    heap: list[tuple[float, int, tuple[int, ...]]] = [
        (-init_score, counter, (0,))
    ]

    while heap and num_nodes < budget:
        neg_score, _, ranks = heapq.heappop(heap)
        score = -neg_score
        d = len(ranks)  # 1-based depth of this node

        parent_ranks = ranks[:-1]
        parent_idx = rank_to_idx[parent_ranks]
        rank_at_d = ranks[-1]

        tokens_list.append(topk_tokens_cpu[d - 1][rank_at_d])
        parents_list.append(parent_idx)
        depths_list.append(d)
        node_idx = num_nodes
        rank_to_idx[ranks] = node_idx
        num_nodes += 1

        # Push next sibling: same parent, next-ranked token at this depth
        next_rank = rank_at_d + 1
        if next_rank < K:
            sib_score = (score
                         - topk_logprobs_cpu[d - 1][rank_at_d]
                         + topk_logprobs_cpu[d - 1][next_rank])
            sib_ranks = parent_ranks + (next_rank,)
            counter += 1
            heapq.heappush(heap, (-sib_score, counter, sib_ranks))

        # Push first child: extend path with best token at next depth
        if d < depth_count:
            child_score = score + topk_logprobs_cpu[d][0]
            child_ranks = ranks + (0,)
            counter += 1
            heapq.heappush(heap, (-child_score, counter, child_ranks))

    return DraftTreeCPU(
        token_ids=tokens_list,
        parent_indices=parents_list,
        depths=depths_list,
        num_nodes=num_nodes,
    )


def _build_tree_cpu(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
    depth_first: bool = True,
    score_mode: str = "accum_logp",
    per_depth_entropy: list[float] | None = None,
    hybrid_alpha: float = 1.0,
) -> DraftTreeCPU:
    if score_mode == "opt_prefix":
        return _build_tree_opt_prefix(
            root_token, topk_tokens_cpu, topk_logprobs_cpu, budget,
        )
    if score_mode == "top2gap_fanout":
        fanout_caps = top2gap_fanout_caps(topk_logprobs_cpu)
        builder = (
            _build_tree_depth_first if depth_first else _build_tree_breadth_first
        )
        return builder(
            root_token, topk_tokens_cpu, topk_logprobs_cpu, budget,
            score_mode="accum_logp",
            per_depth_entropy=None,
            hybrid_alpha=hybrid_alpha,
            fanout_caps=fanout_caps,
        )
    builder = _build_tree_depth_first if depth_first else _build_tree_breadth_first
    return builder(
        root_token, topk_tokens_cpu, topk_logprobs_cpu, budget,
        score_mode=score_mode,
        per_depth_entropy=per_depth_entropy,
        hybrid_alpha=hybrid_alpha,
        fanout_caps=None,
    )


def build_tree_from_topk(
    root_token: int,
    topk_tokens: torch.Tensor,
    topk_logprobs: torch.Tensor,
    budget: int,
    device: torch.device,
    depth_first: bool = True,
    score_mode: str = "accum_logp",
    per_depth_entropy: list[float] | None = None,
    hybrid_alpha: float = 1.0,
    path_prune_budget: int | None = None,
) -> DraftTree:
    """GPU-tensor interface — wraps :func:`_build_tree_cpu`."""
    topk_tokens_cpu = topk_tokens.tolist()
    topk_logprobs_cpu = topk_logprobs.tolist()
    tree_cpu = _build_tree_cpu(
        root_token,
        topk_tokens_cpu,
        topk_logprobs_cpu,
        budget,
        depth_first=depth_first,
        score_mode=score_mode,
        per_depth_entropy=per_depth_entropy,
        hybrid_alpha=hybrid_alpha,
    )
    if path_prune_budget is not None and tree_cpu.num_nodes > path_prune_budget:
        tree_cpu = _prune_tree_by_path_logprob_cpu(
            tree_cpu,
            topk_tokens_cpu,
            topk_logprobs_cpu,
            path_prune_budget,
        )
    return tree_cpu.to_gpu(device)


def build_trees_from_topk(
    root_tokens: torch.Tensor,
    topk_tokens: torch.Tensor,
    topk_logprobs: torch.Tensor,
    budget: int,
    device: torch.device,
    *,
    depth_first: bool = True,
    score_mode: str = "accum_logp",
    per_depth_entropies: list[list[float] | None] | None = None,
    hybrid_alpha: float = 1.0,
    path_prune_budget: int | None = None,
) -> list[DraftTree]:
    """Build a batch of reference-equivalent trees after one batched D2H sync.

    The builder remains on CPU to preserve the exact heap semantics, but all
    request inputs are materialized together. This removes the per-request
    ``item``/``tolist`` synchronization pattern from the serving loop while a
    fully fused GPU builder is validated separately.
    """
    batch_size = topk_tokens.shape[0]
    if root_tokens.shape[0] != batch_size:
        raise ValueError("root_tokens and top-k tensors must share batch size.")
    if topk_logprobs.shape != topk_tokens.shape:
        raise ValueError("topk_tokens and topk_logprobs must have identical shape.")
    if per_depth_entropies is None:
        per_depth_entropies = [None] * batch_size
    if len(per_depth_entropies) != batch_size:
        raise ValueError("per_depth_entropies must have one entry per request.")

    # The first conversion synchronizes all preceding draft/top-k work. The
    # remaining tensors are already ready, avoiding 3*B request-local syncs.
    roots_cpu = root_tokens.tolist()
    tokens_cpu = topk_tokens.tolist()
    logprobs_cpu = topk_logprobs.tolist()
    result: list[DraftTree] = []
    for req_idx in range(batch_size):
        tree_cpu = _build_tree_cpu(
            roots_cpu[req_idx],
            tokens_cpu[req_idx],
            logprobs_cpu[req_idx],
            budget,
            depth_first=depth_first,
            score_mode=score_mode,
            per_depth_entropy=per_depth_entropies[req_idx],
            hybrid_alpha=hybrid_alpha,
        )
        if (
            path_prune_budget is not None
            and tree_cpu.num_nodes > path_prune_budget
        ):
            tree_cpu = _prune_tree_by_path_logprob_cpu(
                tree_cpu,
                tokens_cpu[req_idx],
                logprobs_cpu[req_idx],
                path_prune_budget,
            )
        result.append(tree_cpu.to_gpu(device))
    return result


def pack_seed_tree_batch(
    trees: list[DraftTree],
    seed_budget: int,
    *,
    max_depth: int | None = None,
    build_dense_ancestor_mask: bool = True,
    build_sparse_ancestor_indices: bool = False,
) -> SeedTreeBatch:
    """Pad seed trees and construct dense/sparse ancestor metadata."""
    if not trees:
        raise ValueError("PATR requires at least one seed tree.")
    device = trees[0].token_ids.device
    batch_size = len(trees)
    token_ids = torch.zeros(
        (batch_size, seed_budget), dtype=torch.long, device=device,
    )
    parents = torch.zeros_like(token_ids)
    depths = torch.zeros_like(token_ids)
    valid = torch.zeros(
        (batch_size, seed_budget), dtype=torch.bool, device=device,
    )
    edge_lp = torch.full(
        (batch_size, seed_budget), float("-inf"),
        dtype=torch.float32, device=device,
    )
    ranks = torch.full(
        (batch_size, seed_budget), -1, dtype=torch.int16, device=device,
    )
    for batch_idx, tree in enumerate(trees):
        n = tree.num_nodes
        if n > seed_budget:
            raise ValueError(
                f"Seed tree has {n} nodes, exceeding budget {seed_budget}."
            )
        if tree.seed_edge_logprobs is None or tree.seed_ranks is None:
            raise ValueError("PATR seed tree is missing edge probability metadata.")
        token_ids[batch_idx, :n] = tree.token_ids
        parents[batch_idx, :n] = tree.parent_indices
        depths[batch_idx, :n] = tree.depth
        valid[batch_idx, :n] = True
        edge_lp[batch_idx, :n] = tree.seed_edge_logprobs.float()
        ranks[batch_idx, :n] = tree.seed_ranks

    safe_parents = parents.clamp_min(0)
    # Serving passes the fixed speculative depth, avoiding a device sync to
    # discover it.  The fallback is intended for small CPU tests/callers.
    if max_depth is None:
        max_depth = seed_budget - 1
    ancestor_indices: torch.Tensor | None = None
    ancestor_valid: torch.Tensor | None = None
    ancestors: torch.Tensor | None = None
    if build_dense_ancestor_mask:
        # Reference dense path used by the default SDPA backend. It is faster
        # than constructing sparse metadata and scattering it back to dense on
        # A100 batch=1.
        identity = torch.eye(seed_budget, dtype=torch.bool, device=device)
        ancestors = identity.unsqueeze(0).expand(batch_size, -1, -1).clone()
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

    if build_sparse_ancestor_indices:
        node_indices = torch.arange(
            seed_budget,
            dtype=torch.long,
            device=device,
        ).expand(batch_size, -1)
        ancestor_indices = torch.zeros(
            (batch_size, seed_budget, max_depth + 1),
            dtype=torch.long,
            device=device,
        )
        ancestor_valid = torch.zeros(
            (batch_size, seed_budget, max_depth + 1),
            dtype=torch.bool,
            device=device,
        )
        ancestor_valid[..., 0] = valid & (depths == 0)
        for depth_idx in range(1, max_depth + 1):
            parent_indices_row = torch.gather(
                ancestor_indices,
                1,
                safe_parents[..., None].expand(-1, -1, max_depth + 1),
            )
            parent_valid_row = torch.gather(
                ancestor_valid,
                1,
                safe_parents[..., None].expand(-1, -1, max_depth + 1),
            )
            at_depth = valid & (depths == depth_idx)
            ancestor_indices = torch.where(
                at_depth[..., None],
                parent_indices_row,
                ancestor_indices,
            )
            ancestor_valid = torch.where(
                at_depth[..., None],
                parent_valid_row,
                ancestor_valid,
            )
            ancestor_indices[..., depth_idx] = torch.where(
                at_depth,
                node_indices,
                ancestor_indices[..., depth_idx],
            )
            ancestor_valid[..., depth_idx] |= at_depth
        ancestor_valid[..., 0] |= ~valid

    return SeedTreeBatch(
        token_ids=token_ids,
        parent_indices=parents,
        depths=depths,
        valid_mask=valid,
        seed_edge_logprobs=edge_lp,
        seed_ranks=ranks,
        ancestor_mask=ancestors,
        ancestor_indices=ancestor_indices,
        ancestor_valid_mask=ancestor_valid,
    )


def validate_tree_topology(
    parents: list[int],
    depths: list[int],
    expected_nodes: int | None = None,
) -> None:
    """Validate an already materialized root-inclusive tree topology."""
    num_nodes = len(parents)
    if len(depths) != num_nodes:
        raise ValueError("DFlash parent and depth arrays must have equal length.")
    if expected_nodes is not None and num_nodes != expected_nodes:
        raise ValueError(
            f"Expected {expected_nodes} tree nodes, got {num_nodes}."
        )
    if num_nodes < 1:
        raise ValueError("A DFlash tree must contain its root.")
    if parents[0] != -1 or depths[0] != 0:
        raise ValueError("DFlash root must have parent=-1 and depth=0.")
    for node_idx in range(1, num_nodes):
        parent = parents[node_idx]
        if not 0 <= parent < node_idx:
            raise ValueError(
                f"Invalid parent ordering at node {node_idx}: {parent}."
            )
        if depths[node_idx] != depths[parent] + 1:
            raise ValueError(
                "Invalid tree depth at node "
                f"{node_idx}: {depths[node_idx]} vs parent {depths[parent]}."
            )


def validate_tree_contract(tree: DraftTree, expected_nodes: int | None = None) -> None:
    """Validate the root-inclusive topology consumed by Target verification."""
    validate_tree_topology(
        tree.parent_indices.tolist(),
        tree.depth.tolist(),
        expected_nodes,
    )


def select_prefix_closed_subtrees(
    seed_batch: SeedTreeBatch,
    path_logprobs: torch.Tensor,
    final_budget: int,
    *,
    protect_spine: bool = False,
    edge_logprobs: torch.Tensor | None = None,
) -> list[DraftTree]:
    """Select exact-size PATR subtrees and remap request-local parents."""
    if path_logprobs.shape != seed_batch.valid_mask.shape:
        raise ValueError(
            "path_logprobs and seed tree batch must have identical [B, N] shape."
        )
    batch_size, seed_budget = path_logprobs.shape
    if batch_size == 1:
        return _select_prefix_closed_subtrees_single(
            seed_batch,
            path_logprobs,
            final_budget,
            protect_spine=protect_spine,
            edge_logprobs=edge_logprobs,
        )
    scores = path_logprobs.masked_fill(
        ~seed_batch.valid_mask,
        float("-inf"),
    )
    if scores.device.type == "cpu":
        valid_counts = seed_batch.valid_mask.sum(dim=1)
        if bool(torch.any(valid_counts < final_budget)):
            min_valid = int(valid_counts.min().item())
            raise ValueError(
                f"PATR seed has {min_valid} nodes but final budget is "
                f"{final_budget}."
            )

    # The default serving path is fully batched. Stable sorts applied from
    # least-significant to most-significant key implement exactly:
    # path score descending, depth ascending, old index ascending.
    depth_order = torch.argsort(
        seed_batch.depths,
        dim=1,
        stable=True,
    )
    selection_scores = scores.clone()
    if not protect_spine:
        selection_scores[:, 0] = float("inf")
    depth_order_scores = torch.gather(selection_scores, 1, depth_order)
    score_order = torch.argsort(
        depth_order_scores,
        dim=1,
        descending=True,
        stable=True,
    )
    order = torch.gather(depth_order, 1, score_order)

    if protect_spine:
        # Preserve the compatibility behavior for the non-default protected
        # spine path. It contains data-dependent synchronization and therefore
        # intentionally remains outside the serving fast path.
        selected_rows: list[torch.Tensor] = []
        for batch_idx in range(batch_size):
            valid_count = int(seed_batch.valid_mask[batch_idx].sum().item())
            if valid_count < final_budget:
                raise ValueError(
                    f"PATR seed has {valid_count} nodes but final budget is "
                    f"{final_budget}."
                )
            # This compatibility path is not used by the default PATR config.
            # Keep its exact protected-spine semantics; the common path below
            # is fixed-shape and has no device synchronization.
            protected: list[int] = [0]
            current = 0
            while True:
                children = torch.nonzero(
                    (seed_batch.parent_indices[batch_idx, :valid_count] == current)
                    & (seed_batch.seed_ranks[batch_idx, :valid_count] == 0),
                    as_tuple=False,
                ).flatten()
                if children.numel() == 0:
                    break
                current = int(children[0].item())
                protected.append(current)
            protected_tensor = torch.tensor(
                protected,
                dtype=torch.long,
                device=scores.device,
            )
            is_protected = torch.zeros_like(scores, dtype=torch.bool)
            is_protected[batch_idx, protected_tensor] = True
            selected_rows.append(torch.cat(
                (
                    protected_tensor,
                    order[batch_idx][~is_protected[batch_idx, order[batch_idx]]][
                        : final_budget - len(protected)
                    ],
                )
            ))
        selected = torch.stack(selected_rows)
    else:
        # Root has infinite selection priority, so the fixed slice retains it
        # without boolean filtering or nonzero synchronization.
        selected = order[:, :final_budget]

    # Restore parent-before-child order, then remap all request-local parents
    # with one batched scatter/gather sequence.
    selected = torch.sort(selected, dim=1).values
    old_to_new = torch.full_like(seed_batch.parent_indices, -1)
    new_indices = torch.arange(
        final_budget,
        device=scores.device,
    ).expand(batch_size, -1)
    old_to_new.scatter_(1, selected, new_indices)
    selected_parents = torch.gather(
        seed_batch.parent_indices,
        1,
        selected,
    ).clamp_min(0)
    final_parents = torch.gather(old_to_new, 1, selected_parents)
    final_parents = torch.where(
        selected == 0,
        torch.full_like(final_parents, -1),
        final_parents,
    )
    selected_tokens = torch.gather(seed_batch.token_ids, 1, selected)
    selected_depths = torch.gather(seed_batch.depths, 1, selected)

    result = [
        DraftTree(
            token_ids=selected_tokens[batch_idx],
            parent_indices=final_parents[batch_idx],
            depth=selected_depths[batch_idx],
            num_nodes=final_budget,
            seed_edge_logprobs=torch.gather(
                seed_batch.seed_edge_logprobs
                if edge_logprobs is None else edge_logprobs,
                1,
                selected,
            )[batch_idx],
            seed_ranks=torch.gather(seed_batch.seed_ranks, 1, selected)[batch_idx],
        )
        for batch_idx in range(batch_size)
    ]
    if scores.device.type == "cpu":
        # CPU callers are primarily tests, where retaining eager contract
        # validation is useful. The serving GPU path is prefix-closed by the
        # non-positive path-score invariant and avoids topology D2H syncs.
        for tree in result:
            validate_tree_contract(tree, expected_nodes=final_budget)
    return result


def _select_prefix_closed_subtrees_single(
    seed_batch: SeedTreeBatch,
    path_logprobs: torch.Tensor,
    final_budget: int,
    *,
    protect_spine: bool,
    edge_logprobs: torch.Tensor | None,
) -> list[DraftTree]:
    """Original request-local selector retained for the batch=1 fast path."""
    scores = path_logprobs[0].masked_fill(
        ~seed_batch.valid_mask[0],
        float("-inf"),
    )
    if scores.device.type == "cpu":
        valid_count = int(seed_batch.valid_mask[0].sum().item())
        if valid_count < final_budget:
            raise ValueError(
                f"PATR seed has {valid_count} nodes but final budget is "
                f"{final_budget}."
            )
    depths = seed_batch.depths[0]
    order = torch.arange(scores.shape[0], device=scores.device)
    order = order[torch.argsort(depths[order], stable=True)]
    selection_scores = scores
    if not protect_spine:
        selection_scores = scores.clone()
        selection_scores[0] = float("inf")
    order = order[
        torch.argsort(
            selection_scores[order],
            descending=True,
            stable=True,
        )
    ]
    if protect_spine:
        valid_count = int(seed_batch.valid_mask[0].sum().item())
        protected = [0]
        current = 0
        while True:
            children = torch.nonzero(
                (seed_batch.parent_indices[0, :valid_count] == current)
                & (seed_batch.seed_ranks[0, :valid_count] == 0),
                as_tuple=False,
            ).flatten()
            if children.numel() == 0:
                break
            current = int(children[0].item())
            protected.append(current)
        protected_tensor = torch.tensor(
            protected,
            dtype=torch.long,
            device=scores.device,
        )
        is_protected = torch.zeros_like(scores, dtype=torch.bool)
        is_protected[protected_tensor] = True
        selected = torch.cat(
            (
                protected_tensor,
                order[~is_protected[order]][: final_budget - len(protected)],
            )
        )
    else:
        selected = order[:final_budget]
    selected = torch.sort(selected).values
    parents = seed_batch.parent_indices[0]
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
    tree = DraftTree(
        token_ids=seed_batch.token_ids[0, selected],
        parent_indices=final_parents,
        depth=seed_batch.depths[0, selected],
        num_nodes=final_budget,
        seed_edge_logprobs=(
            seed_batch.seed_edge_logprobs if edge_logprobs is None
            else edge_logprobs
        )[0, selected],
        seed_ranks=seed_batch.seed_ranks[0, selected],
    )
    if scores.device.type == "cpu":
        validate_tree_contract(tree, expected_nodes=final_budget)
    return [tree]


def _prune_and_regrow_cpu(
    tree: DraftTreeCPU,
    node_logprobs: list[float],
    cond_topk_lp_cpu: list[list[float]],
    cond_topk_tok_cpu: list[list[int]],
    block_size: int,
    tree_width: int,
    budget: int,
    prune_ratio: float = 0.25,
) -> DraftTreeCPU:
    """Prune lowest-scoring leaves and regrow — pure CPU, zero GPU ops.

    *node_logprobs* contains the per-node conditional log-prob
    (``lp[depth-1][token_id]``) already gathered on GPU.
    ``node_logprobs[0]`` must be 0.0 (root).
    *prune_ratio* controls the fraction of leaves to prune (0.0–1.0).
    """
    max_depth = block_size - 1

    tokens = tree.token_ids
    parents = tree.parent_indices
    depths = tree.depths
    num_nodes = tree.num_nodes

    cum_lp: list[float] = [0.0] * num_nodes
    for i in range(1, num_nodes):
        cum_lp[i] = cum_lp[parents[i]] + node_logprobs[i]

    children_of: list[list[int]] = [[] for _ in range(num_nodes)]
    for i in range(1, num_nodes):
        children_of[parents[i]].append(i)

    leaves_with_score: list[tuple[float, int]] = []
    for i in range(num_nodes):
        if not children_of[i]:
            leaves_with_score.append((cum_lp[i], i))
    if not leaves_with_score:
        return tree
    leaves_with_score.sort()

    n_prune = max(1, int(len(leaves_with_score) * prune_ratio))
    pruned_set: set[int] = set()
    for _, leaf_idx in leaves_with_score[:n_prune]:
        node = leaf_idx
        if node <= 0:
            continue
        parent = parents[node]
        siblings_alive = sum(
            1 for c in children_of[parent] if c not in pruned_set
        )
        if siblings_alive > 1:
            pruned_set.add(node)

    kept = [i for i in range(num_nodes) if i not in pruned_set]
    if not kept:
        kept = [0]
    old_to_new = [-1] * num_nodes
    for new_i, old_i in enumerate(kept):
        old_to_new[old_i] = new_i

    new_tokens = [tokens[i] for i in kept]
    new_parents = [old_to_new[parents[i]] if i > 0 else -1 for i in kept]
    new_depths = [depths[i] for i in kept]
    new_cum_lp = [cum_lp[i] for i in kept]
    num_nodes = len(kept)
    freed = budget - num_nodes

    if freed > 0:
        new_children_of: list[list[int]] = [[] for _ in range(num_nodes)]
        for i in range(1, num_nodes):
            new_children_of[new_parents[i]].append(i)

        counter = 0
        regrow_heap: list[tuple[float, int, int]] = []
        for i in range(num_nodes):
            if not new_children_of[i] and new_depths[i] < max_depth:
                counter += 1
                heapq.heappush(regrow_heap, (-new_cum_lp[i], counter, i))

        while regrow_heap and freed > 0:
            _, _, node_idx = heapq.heappop(regrow_heap)
            depth_idx = new_depths[node_idx]
            if depth_idx >= max_depth:
                continue
            children_to_add = min(tree_width, freed)
            row_tokens = cond_topk_tok_cpu[depth_idx]
            row_logprobs = cond_topk_lp_cpu[depth_idx]
            for child_idx in range(children_to_add):
                child_cum = new_cum_lp[node_idx] + row_logprobs[child_idx]
                new_tokens.append(row_tokens[child_idx])
                new_parents.append(node_idx)
                new_depths.append(depth_idx + 1)
                new_cum_lp.append(child_cum)
                counter += 1
                heapq.heappush(regrow_heap, (-child_cum, counter, num_nodes))
                num_nodes += 1
                freed -= 1

    return DraftTreeCPU(
        token_ids=new_tokens,
        parent_indices=new_parents,
        depths=new_depths,
        num_nodes=num_nodes,
    )


def _gather_node_logprobs(
    lp: torch.Tensor,
    tree: DraftTree,
) -> list[float]:
    """Gather per-node conditional log-probs on GPU, return as CPU list.

    ``lp`` has shape ``(depth, vocab)``.  For each non-root node *i* in
    *tree*, the result contains ``lp[depth[i]-1, token_id[i]]``.
    Index 0 (root) is always 0.0.
    """
    n = tree.num_nodes
    if n <= 1:
        return [0.0] * n
    depths = tree.depth[1:]
    tokens = tree.token_ids[1:]
    gathered = lp[depths - 1, tokens]
    return [0.0] + gathered.tolist()


def prune_and_regrow(
    tree: DraftTree,
    cond_logits: torch.Tensor,
    block_size: int,
    tree_width: int,
    budget: int,
    device: torch.device | None = None,
    prune_ratio: float = 0.25,
) -> DraftTree:
    """Prune lowest-scoring leaves and regrow using conditioned logits."""
    if device is None:
        device = tree.token_ids.device
    lp = torch.log_softmax(cond_logits, dim=-1)
    topk_lp, topk_tok = torch.topk(lp, tree_width, dim=-1)

    node_logprobs = _gather_node_logprobs(lp, tree)
    topk_lp_cpu = topk_lp.tolist()
    topk_tok_cpu = topk_tok.tolist()
    tree_cpu = DraftTreeCPU(
        token_ids=tree.token_ids.tolist(),
        parent_indices=tree.parent_indices.tolist(),
        depths=tree.depth.tolist(),
        num_nodes=tree.num_nodes,
    )
    result = _prune_and_regrow_cpu(
        tree_cpu, node_logprobs, topk_lp_cpu, topk_tok_cpu,
        block_size, tree_width, budget, prune_ratio=prune_ratio,
    )
    return result.to_gpu(device)


def _prune_tree_by_path_logprob_cpu(
    tree: DraftTreeCPU,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
) -> DraftTreeCPU:
    """Pure-CPU cumulative-path pruning used before the final GPU upload."""
    if budget < 1:
        raise ValueError(f"Tree pruning budget must be positive, got {budget}.")
    if tree.num_nodes <= budget:
        return tree

    tokens = tree.token_ids
    parents = tree.parent_indices
    depths = tree.depths

    edge_logprob_by_depth: list[dict[int, float]] = []
    for row_tokens, row_logprobs in zip(
        topk_tokens_cpu, topk_logprobs_cpu, strict=True,
    ):
        edge_logprob_by_depth.append({
            int(token): min(float(logprob), 0.0)
            for token, logprob in zip(row_tokens, row_logprobs, strict=True)
        })

    path_logprobs = [0.0] * tree.num_nodes
    for node_idx in range(1, tree.num_nodes):
        parent_idx = int(parents[node_idx])
        node_depth = int(depths[node_idx])
        if not 0 <= parent_idx < node_idx:
            raise ValueError(
                "DFlash tree must be parent-before-child before path pruning: "
                f"node={node_idx}, parent={parent_idx}."
            )
        if not 1 <= node_depth <= len(edge_logprob_by_depth):
            raise ValueError(
                f"Invalid DFlash node depth {node_depth} at node {node_idx}."
            )
        token = int(tokens[node_idx])
        row = edge_logprob_by_depth[node_depth - 1]
        if token not in row:
            raise ValueError(
                "Tree node token is absent from its depth's seed Top-K row: "
                f"node={node_idx}, depth={node_depth}, token={token}."
            )
        path_logprobs[node_idx] = path_logprobs[parent_idx] + row[token]

    ranked_non_root = sorted(
        range(1, tree.num_nodes),
        key=lambda idx: (-path_logprobs[idx], depths[idx], idx),
    )
    selected_set = {0, *ranked_non_root[:budget - 1]}
    for node_idx in selected_set:
        if node_idx != 0 and parents[node_idx] not in selected_set:
            raise RuntimeError(
                "Path-logprob pruning violated prefix closure: "
                f"node={node_idx}, parent={parents[node_idx]}."
            )

    kept = sorted(selected_set)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept)}
    kept_tokens = [tokens[old_idx] for old_idx in kept]
    kept_parents = [
        -1 if old_idx == 0 else old_to_new[parents[old_idx]]
        for old_idx in kept
    ]
    kept_depths = [depths[old_idx] for old_idx in kept]
    kept_edge_logprobs = (
        [tree.seed_edge_logprobs[old_idx] for old_idx in kept]
        if tree.seed_edge_logprobs is not None else None
    )
    kept_ranks = (
        [tree.seed_ranks[old_idx] for old_idx in kept]
        if tree.seed_ranks is not None else None
    )

    return DraftTreeCPU(
        token_ids=kept_tokens,
        parent_indices=kept_parents,
        depths=kept_depths,
        num_nodes=len(kept),
        seed_edge_logprobs=kept_edge_logprobs,
        seed_ranks=kept_ranks,
    )


def prune_tree_by_path_logprob(
    tree: DraftTree,
    topk_tokens: torch.Tensor,
    topk_logprobs: torch.Tensor,
    budget: int,
    device: torch.device | None = None,
) -> DraftTree:
    """Select a prefix-closed subtree by cumulative seed path log-probability.

    The root is always retained. Remaining nodes are ordered by the stable
    lexicographic key ``(path_logprob desc, depth asc, old_index asc)``. Since
    every conditional log-probability is non-positive, an ancestor cannot
    score below its descendants; the depth tie-break makes equal-score paths
    ancestor-first. The returned layout keeps old-index order and remaps every
    parent, preserving the verifier's parent-before-child contract.

    The proposer hot path invokes the same CPU implementation from
    :func:`build_tree_from_topk`, before uploading the tree to GPU. This public
    wrapper remains useful for tests and callers that already hold a DraftTree.
    """
    if device is None:
        device = tree.token_ids.device
    tree_cpu = DraftTreeCPU(
        token_ids=tree.token_ids.tolist(),
        parent_indices=tree.parent_indices.tolist(),
        depths=tree.depth.tolist(),
        num_nodes=tree.num_nodes,
    )
    pruned = _prune_tree_by_path_logprob_cpu(
        tree_cpu,
        topk_tokens.tolist(),
        topk_logprobs.tolist(),
        budget,
    )
    return pruned.to_gpu(device)


def _adjust_tree_to_size_cpu(
    tree: DraftTreeCPU,
    target_size: int,
    node_logprobs: list[float] | None,
    cond_topk_lp_cpu: list[list[float]] | None,
    cond_topk_tok_cpu: list[list[int]] | None,
    block_size: int,
    tree_width: int,
) -> DraftTreeCPU:
    """Grow or shrink *tree* to ``target_size`` — pure CPU, zero GPU ops.

    *node_logprobs* contains the per-node conditional log-prob already
    gathered on GPU.  ``node_logprobs[0]`` must be 0.0 (root).
    """
    if tree.num_nodes == target_size or target_size <= 0:
        return tree

    max_depth = block_size - 1
    tokens = list(tree.token_ids)
    parents = list(tree.parent_indices)
    depths = list(tree.depths)
    num_nodes = tree.num_nodes

    cum_lp: list[float] = [0.0] * num_nodes
    if node_logprobs is not None:
        for i in range(1, num_nodes):
            cum_lp[i] = cum_lp[parents[i]] + node_logprobs[i]

    if num_nodes > target_size:
        children_of: list[list[int]] = [[] for _ in range(num_nodes)]
        for i in range(1, num_nodes):
            children_of[parents[i]].append(i)

        alive = [True] * num_nodes
        leaf_heap: list[tuple[float, int]] = []
        for i in range(num_nodes):
            if not children_of[i]:
                heapq.heappush(leaf_heap, (cum_lp[i], i))

        while num_nodes > target_size and leaf_heap:
            _, idx = heapq.heappop(leaf_heap)
            if not alive[idx]:
                continue
            if idx == 0:
                break
            alive[idx] = False
            num_nodes -= 1
            p = parents[idx]
            children_of[p] = [c for c in children_of[p] if alive[c]]
            if not children_of[p]:
                heapq.heappush(leaf_heap, (cum_lp[p], p))

        kept = [i for i in range(len(alive)) if alive[i]]
        old_to_new = [-1] * len(alive)
        for new_i, old_i in enumerate(kept):
            old_to_new[old_i] = new_i
        tokens = [tokens[i] for i in kept]
        parents = [old_to_new[parents[i]] if i > 0 else -1 for i in kept]
        depths = [depths[i] for i in kept]
        cum_lp = [cum_lp[i] for i in kept]
        num_nodes = len(kept)

    if num_nodes < target_size and cond_topk_tok_cpu is not None:
        children_of_g: list[list[int]] = [[] for _ in range(num_nodes)]
        for i in range(1, num_nodes):
            children_of_g[parents[i]].append(i)

        counter = 0
        grow_heap: list[tuple[float, int, int]] = []
        for i in range(num_nodes):
            if not children_of_g[i] and depths[i] < max_depth:
                counter += 1
                heapq.heappush(grow_heap, (-cum_lp[i], counter, i))

        freed = target_size - num_nodes
        while grow_heap and freed > 0:
            _, _, node_idx = heapq.heappop(grow_heap)
            depth_idx = depths[node_idx]
            if depth_idx >= max_depth:
                continue
            children_to_add = min(tree_width, freed)
            assert cond_topk_lp_cpu is not None
            row_tokens = cond_topk_tok_cpu[depth_idx]
            row_logprobs = cond_topk_lp_cpu[depth_idx]
            for child_idx in range(children_to_add):
                child_cum = cum_lp[node_idx] + row_logprobs[child_idx]
                tokens.append(row_tokens[child_idx])
                parents.append(node_idx)
                depths.append(depth_idx + 1)
                cum_lp.append(child_cum)
                counter += 1
                heapq.heappush(grow_heap, (-child_cum, counter, num_nodes))
                num_nodes += 1
                freed -= 1

    return DraftTreeCPU(
        token_ids=tokens,
        parent_indices=parents,
        depths=depths,
        num_nodes=num_nodes,
    )


def adjust_tree_to_size(
    tree: DraftTree,
    target_size: int,
    cond_logits: torch.Tensor | None,
    block_size: int,
    tree_width: int,
    device: torch.device | None = None,
) -> DraftTree:
    """Grow or shrink *tree* to *target_size* using conditioned logits."""
    if tree.num_nodes == target_size or target_size <= 0:
        return tree
    if device is None:
        device = tree.token_ids.device

    if cond_logits is not None:
        lp = torch.log_softmax(cond_logits, dim=-1)
        topk_lp, topk_tok = torch.topk(lp, tree_width, dim=-1)
        node_logprobs: list[float] | None = _gather_node_logprobs(lp, tree)
        topk_lp_cpu: list[list[float]] | None = topk_lp.tolist()
        topk_tok_cpu: list[list[int]] | None = topk_tok.tolist()
    else:
        node_logprobs = topk_lp_cpu = topk_tok_cpu = None

    tree_cpu = DraftTreeCPU(
        token_ids=tree.token_ids.tolist(),
        parent_indices=tree.parent_indices.tolist(),
        depths=tree.depth.tolist(),
        num_nodes=tree.num_nodes,
    )
    result = _adjust_tree_to_size_cpu(
        tree_cpu, target_size, node_logprobs, topk_lp_cpu, topk_tok_cpu,
        block_size, tree_width,
    )
    return result.to_gpu(device)


def find_closest_capture_size(
    num_nodes: int,
    capture_sizes: list[int],
) -> int:
    """Return the smallest capture size >= *num_nodes*.

    Falls back to the largest capture size if *num_nodes* exceeds all of them.
    *capture_sizes* must be sorted ascending.
    """
    import bisect
    idx = bisect.bisect_left(capture_sizes, num_nodes)
    if idx < len(capture_sizes):
        return capture_sizes[idx]
    return capture_sizes[-1]


def tree_signature(tree: DraftTreeCPU | DraftTree) -> str:
    """One-line summary of tree shape: ``N=150 D=5 L=30 depth=[1:5,2:20,...]``."""
    if isinstance(tree, DraftTree):
        depths = tree.depth.tolist()
        parents = tree.parent_indices.tolist()
    else:
        depths = tree.depths
        parents = tree.parent_indices
    n = tree.num_nodes
    if n == 0:
        return "N=0"
    max_d = max(depths) if depths else 0
    depth_counts: dict[int, int] = {}
    num_leaves = 0
    child_count = [0] * n
    for i in range(1, n):
        child_count[parents[i]] += 1
    for i in range(n):
        d = depths[i]
        depth_counts[d] = depth_counts.get(d, 0) + 1
        if child_count[i] == 0:
            num_leaves += 1
    dist_str = ",".join(f"{d}:{c}" for d, c in sorted(depth_counts.items()))
    return f"N={n} D={max_d} L={num_leaves} depth=[{dist_str}]"


def build_tree_entropy_guided(
    root_token: int,
    draft_logits: torch.Tensor,
    block_size: int,
    tree_width: int,
    budget: int,
    cond_logits: torch.Tensor | None = None,
    device: torch.device | None = None,
    prune_ratio: float = 0.25,
    depth_first: bool = True,
) -> DraftTree:
    if device is None:
        device = draft_logits.device
    topk_logprobs, topk_tokens = sample_topk_from_logits(draft_logits, tree_width)
    tree = build_tree_from_topk(
        root_token=root_token,
        topk_tokens=topk_tokens,
        topk_logprobs=topk_logprobs,
        budget=budget,
        device=device,
        depth_first=depth_first,
    )
    if cond_logits is not None:
        tree = prune_and_regrow(
            tree,
            cond_logits=cond_logits,
            block_size=block_size,
            tree_width=tree_width,
            budget=budget,
            device=device,
            prune_ratio=prune_ratio,
        )
    return tree


def _build_attention_bias_np(
    parents_list: list[int],
    neg_inf: float,
) -> np.ndarray:
    """Build tree attention bias as a numpy array on CPU."""
    query_len = len(parents_list)
    mask = np.zeros((query_len, query_len), dtype=np.bool_)
    for i in range(query_len):
        mask[i, i] = True
    mask[:, 0] = True
    for node_idx in range(1, query_len):
        parent = parents_list[node_idx]
        while parent > 0:
            mask[node_idx, parent] = True
            parent = parents_list[parent]

    bias_np = np.full((query_len, query_len), neg_inf, dtype=np.float32)
    bias_np[mask] = 0.0
    return bias_np


def _build_causal_bias_np(
    query_len: int,
    neg_inf: float,
) -> np.ndarray:
    """Build causal attention bias as a numpy array on CPU."""
    bias_np = np.full((query_len, query_len), neg_inf, dtype=np.float32)
    il = np.tril_indices(query_len)
    bias_np[il] = 0.0
    return bias_np


def build_ancestor_matrix_np(parents_list: list[int]) -> np.ndarray:
    """Build an (N, N) int32 ancestor matrix from parent indices on CPU.

    ancestor[i, j] == 1 iff j is on the root-to-i path (inclusive of i itself).
    This is the tree attention mask consumed by the optimus SM90 kernel.
    """
    N = len(parents_list)
    ancestor = np.eye(N, dtype=np.int32)
    for i in range(1, N):
        ancestor[i] |= ancestor[parents_list[i]]
    return ancestor


def build_causal_ancestor_matrix_np(query_len: int) -> np.ndarray:
    """Build a causal (lower-triangular) ancestor matrix for chain topology."""
    return np.tril(np.ones((query_len, query_len), dtype=np.int32))


def build_attention_bias_from_parents(
    parent_indices: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    parents_cpu = parent_indices.tolist()
    neg_inf = float(torch.finfo(dtype).min)
    bias_np = _build_attention_bias_np(parents_cpu, neg_inf)
    return torch.from_numpy(bias_np).to(dtype=dtype, device=device)


def build_causal_attention_bias(
    query_len: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    neg_inf = float(torch.finfo(dtype).min)
    bias_np = np.full((query_len, query_len), neg_inf, dtype=np.float32)
    il = np.tril_indices(query_len)
    bias_np[il] = 0.0
    return torch.from_numpy(bias_np).to(dtype=dtype, device=device)


def build_block_diagonal_attention_bias(
    biases: list[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    total_query_len = sum(bias.shape[0] for bias in biases)
    if total_query_len == 0:
        return torch.empty((0, 0), dtype=dtype, device=device)
    neg_inf = float(torch.finfo(dtype).min)
    output_np = np.full((total_query_len, total_query_len), neg_inf,
                         dtype=np.float32)
    cursor = 0
    for bias in biases:
        qlen = bias.shape[0]
        if isinstance(bias, np.ndarray):
            output_np[cursor:cursor + qlen, cursor:cursor + qlen] = bias
        else:
            output_np[cursor:cursor + qlen, cursor:cursor + qlen] = (
                bias.cpu().numpy()
            )
        cursor += qlen
    return torch.from_numpy(output_np).to(dtype=dtype, device=device)


def gpu_tree_accept(
    tree_tokens: torch.Tensor,
    greedy_targets: torch.Tensor,
    parent_indices: torch.Tensor,
    depths: torch.Tensor,
    max_depth: int = 15,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """GPU-native greedy tree acceptance — no CPU round-trips.

    All matching and path extraction runs on GPU.  Only a single
    ``accepted_depth.item()`` sync is needed to slice the output path.

    Args:
        tree_tokens: [N] draft token ids (root + tree nodes).
        greedy_targets: [N] argmax of the *target* model logits (pre-computed).
        parent_indices: [N] parent node index per node (-1 for root).
        depths: [N] depth of each node (root = 0).
        max_depth: upper bound on tree depth (used for loop unrolling).

    Returns:
        accepted_path: 1-D tensor of node indices from root to deepest
            accepted node (length = accepted_len + 1).
        accepted_len: Python int — number of accepted *draft* tokens
            (excludes root).
        correction_token: 0-D tensor with the target-model greedy token
            at the last accepted position.
    """
    device = tree_tokens.device
    N = tree_tokens.shape[0]

    if N <= 1:
        return (
            torch.zeros(1, dtype=torch.long, device=device),
            0,
            greedy_targets[0],
        )

    safe_parents = parent_indices.clamp(min=0)

    # --- vectorised match: does each node agree with its parent's target? ---
    match = torch.ones(N, dtype=torch.bool, device=device)
    match[1:] = tree_tokens[1:] == greedy_targets[safe_parents[1:]]

    # --- prefix-match via parallel doubling (ceil-log2 iterations) ---
    prefix_match = match.clone()
    jump = safe_parents.clone()
    for _ in range(max(1, max_depth.bit_length())):
        anc_match = prefix_match[jump]
        prefix_match = prefix_match & anc_match
        jump = jump[jump]

    # --- deepest fully-accepted node ---
    score = torch.where(
        prefix_match,
        depths,
        torch.tensor(-1, device=device, dtype=depths.dtype),
    )
    best_node = torch.argmax(score)
    accepted_depth_t = depths[best_node]
    correction = greedy_targets[best_node]

    # --- extract root-to-best path (walk parent pointers on GPU) ---
    path_buf = torch.zeros(max_depth + 1, dtype=torch.long, device=device)
    current = best_node.unsqueeze(0)
    for d in range(max_depth, -1, -1):
        path_buf[d : d + 1] = current
        current = safe_parents[current]

    accepted_len = int(accepted_depth_t.item())  # single GPU→CPU sync
    valid_start = max_depth - accepted_len
    accepted_path = path_buf[valid_start : max_depth + 1].contiguous()

    return accepted_path, accepted_len, correction


def tree_accept(
    tree_tokens: torch.Tensor,
    parent_indices: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float = 0.0,
) -> tuple[list[int], int, int]:
    """Reference-style tree acceptance for flattened vLLM tree tensors.

    Returns:
        accepted_path: root-inclusive accepted path indices.
        acceptance_length: accepted draft-token count (excludes root).
        correction_token: posterior token sampled at the last accepted node.
    """
    if temperature != 0.0:
        raise NotImplementedError(
            "Native DFlash tree acceptance currently supports greedy decoding "
            "only."
        )

    posterior_cpu = torch.argmax(target_logits, dim=-1).tolist()
    tokens_cpu = tree_tokens.tolist()
    parents_cpu = parent_indices.tolist()
    num_nodes = len(tokens_cpu)

    children: list[list[int]] = [[] for _ in range(num_nodes)]
    for node_idx in range(1, num_nodes):
        children[parents_cpu[node_idx]].append(node_idx)

    def _paths() -> list[list[int]]:
        leaves = [idx for idx in range(num_nodes) if not children[idx]]
        paths: list[list[int]] = []
        for leaf in leaves:
            path: list[int] = []
            node = leaf
            while node >= 0:
                path.append(node)
                node = parents_cpu[node]
            paths.append(path[::-1])
        return paths

    best_path = [0]
    best_len = 0
    for path in _paths():
        accepted = 0
        for depth_idx in range(1, len(path)):
            parent_node = path[depth_idx - 1]
            child_node = path[depth_idx]
            if tokens_cpu[child_node] == posterior_cpu[parent_node]:
                accepted += 1
            else:
                break
        if accepted > best_len:
            best_len = accepted
            best_path = path[: accepted + 1]

    correction_token = posterior_cpu[best_path[-1]]
    return best_path, best_len, correction_token


def tree_accept_greedy(
    tree_tokens: torch.Tensor,
    parent_indices: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> tuple[list[int], int]:
    tokens_cpu = tree_tokens.tolist()
    parents_cpu = parent_indices.tolist()
    targets_cpu = target_token_ids.tolist()
    num_nodes = len(tokens_cpu)

    children: list[list[int]] = [[] for _ in range(num_nodes)]
    for node_idx in range(1, num_nodes):
        children[parents_cpu[node_idx]].append(node_idx)

    best_path = [0]
    best_len = 0
    stack: list[list[int]] = [[0]]
    while stack:
        path = stack.pop()
        leaf = path[-1]
        child_nodes = children[leaf]
        if not child_nodes:
            accepted = 0
            for depth_idx in range(1, len(path)):
                parent = path[depth_idx - 1]
                child = path[depth_idx]
                if tokens_cpu[child] == targets_cpu[parent]:
                    accepted += 1
                else:
                    break
            if accepted > best_len:
                best_len = accepted
                best_path = path[: accepted + 1]
            continue
        for child in reversed(child_nodes):
            stack.append([*path, child])
    return best_path, best_len
