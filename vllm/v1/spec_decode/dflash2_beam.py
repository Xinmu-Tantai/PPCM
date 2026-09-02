# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DFlash2 Top-K lattice construction and fixed-width beam traversal.

The score definition and single-path traversal mirror SGLang's production
DFlash2 selector.  ``select_lattice_beams`` extends that traversal from one
locally greedy path to a fixed number of globally scored paths without
materialising the exponential Cartesian product.
"""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

DFLASH2_TOP_K = 16
DFLASH2_BEAM_SIZE = 16


@dataclass
class DFlash2BeamOutput:
    token_ids: torch.Tensor
    candidate_ranks: torch.Tensor
    step_scores: torch.Tensor
    path_scores: torch.Tensor


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    uniforms_ptr,
    temperatures_ptr,
    greedy_ptr,
    tokens_ptr,
    q_ptr,
    slots: tl.constexpr,
    top_k: tl.constexpr,
):
    """SGLang DFlash2 register-resident selector walk, ported verbatim."""
    row = tl.program_id(0)
    offsets = tl.arange(0, top_k)
    temperature = tl.load(temperatures_ptr + row)
    greedy = tl.load(greedy_ptr + row) != 0
    previous = 0
    for slot in range(slots):
        base = (row * slots + slot) * top_k
        scores = tl.load(scores_ptr + (base + previous) * top_k + offsets).to(
            tl.float32
        )
        if greedy:
            best = tl.max(scores, axis=0)
            index = tl.min(tl.where(scores == best, offsets, top_k), axis=0)
            probabilities = tl.where(offsets == index, 1.0, 0.0)
        else:
            scaled = scores / temperature
            exponentials = tl.exp(scaled - tl.max(scaled, axis=0))
            probabilities = exponentials / tl.sum(exponentials, axis=0)
            uniform = tl.load(uniforms_ptr + row * slots + slot)
            index = tl.sum(
                tl.where(uniform >= tl.cumsum(probabilities, axis=0), 1, 0), axis=0
            )
            index = tl.minimum(index, top_k - 1)
        tl.store(q_ptr + base + offsets, probabilities)
        tl.store(tokens_ptr + row * slots + slot, tl.load(candidate_ptr + base + index))
        previous = index


def selector_walk_triton(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the production single-path walk; beam traversal uses PyTorch below."""
    batch, slots, top_k = candidate_ids.shape
    tokens = torch.empty((batch, slots), dtype=torch.int64, device=scores.device)
    q_rows = torch.empty(
        (batch, slots, top_k), dtype=torch.float32, device=scores.device
    )
    _selector_walk_kernel[(batch,)](
        scores.contiguous(),
        candidate_ids.contiguous(),
        uniforms.contiguous(),
        temperatures.contiguous(),
        greedy_mask.contiguous(),
        tokens,
        q_rows,
        slots=slots,
        top_k=top_k,
        num_warps=1,
    )
    return tokens, q_rows


def score_dflash2_lattice(
    *,
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    hidden_projection_weight: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Build DFlash2 transition scores.

    Returns ``[B, D, K, K]``.  At depth zero only predecessor row zero is
    meaningful; later depths use candidate ``p`` from the preceding slot.
    """
    if candidate_ids.ndim != 3:
        raise ValueError("candidate_ids must have shape [B, D, K].")
    if unary_logits.shape != candidate_ids.shape:
        raise ValueError("unary_logits must match candidate_ids.")
    if hidden_states.shape[:2] != candidate_ids.shape[:2]:
        raise ValueError("hidden_states must match candidate batch and depth.")
    batch_size, depth, top_k = candidate_ids.shape
    if anchor_token_ids.shape != (batch_size,):
        raise ValueError("anchor_token_ids must have shape [B].")

    projected = torch.nn.functional.linear(hidden_states, hidden_projection_weight)
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        [
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ],
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    edges = torch.einsum(
        "bdpr,bdcr,bdr->bdpc", predecessors, successors, projected
    )
    return unary_logits[:, :, None, :].float() + edges.float()


def walk_dflash2_lattice(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
) -> DFlash2BeamOutput:
    """Reference greedy walk equivalent to the production selector walk."""
    batch_size, depth, top_k = candidate_ids.shape
    if scores.shape != (batch_size, depth, top_k, top_k):
        raise ValueError("scores must have shape [B, D, K, K].")
    predecessor = torch.zeros(batch_size, dtype=torch.long, device=scores.device)
    ranks: list[torch.Tensor] = []
    chosen_scores: list[torch.Tensor] = []
    for position in range(depth):
        rows = scores[:, position].gather(
            1,
            predecessor[:, None, None].expand(-1, 1, top_k),
        )[:, 0]
        predecessor = rows.argmax(dim=-1)
        ranks.append(predecessor)
        chosen_scores.append(rows.gather(1, predecessor[:, None])[:, 0])
    rank_tensor = torch.stack(ranks, dim=1)
    token_ids = candidate_ids.gather(-1, rank_tensor[..., None])[..., 0]
    step_scores = torch.stack(chosen_scores, dim=1)
    return DFlash2BeamOutput(
        token_ids=token_ids[:, None],
        candidate_ranks=rank_tensor[:, None],
        step_scores=step_scores[:, None],
        path_scores=step_scores.sum(dim=-1, keepdim=True),
    )


def select_lattice_beams(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    *,
    beam_size: int = DFLASH2_BEAM_SIZE,
    normalize_edges: bool = True,
    include_greedy: bool = True,
) -> DFlash2BeamOutput:
    """Select the best fixed-width paths through a DFlash2 lattice.

    ``normalize_edges`` accumulates conditional log-probabilities instead of
    raw logits, making scores at different draft positions comparable.
    """
    batch_size, depth, top_k = candidate_ids.shape
    if scores.shape != (batch_size, depth, top_k, top_k):
        raise ValueError("scores must have shape [B, D, K, K].")
    if not 1 <= beam_size <= top_k**depth:
        raise ValueError("beam_size is outside the number of possible paths.")
    edge_scores = (
        torch.log_softmax(scores.float(), dim=-1)
        if normalize_edges
        else scores.float()
    )

    first = edge_scores[:, 0, 0]
    width = min(beam_size, top_k)
    path_scores, first_ranks = first.topk(width, dim=-1)
    ranks = first_ranks[..., None]
    selected_steps = first.gather(1, first_ranks)[..., None]

    for position in range(1, depth):
        predecessor = ranks[..., -1]
        rows = edge_scores[:, position].gather(
            1, predecessor[..., None].expand(-1, -1, top_k)
        )
        expanded = path_scores[..., None] + rows
        width = min(beam_size, expanded.shape[1] * top_k)
        path_scores, flat_indices = expanded.flatten(1).topk(width, dim=-1)
        parent = torch.div(flat_indices, top_k, rounding_mode="floor")
        successor = flat_indices.remainder(top_k)
        ranks = torch.cat(
            [
                ranks.gather(1, parent[..., None].expand(-1, -1, ranks.shape[-1])),
                successor[..., None],
            ],
            dim=-1,
        )
        selected_steps = torch.cat(
            [
                selected_steps.gather(
                    1, parent[..., None].expand(-1, -1, selected_steps.shape[-1])
                ),
                rows.gather(1, parent[..., None].expand(-1, -1, top_k))
                .gather(2, successor[..., None]),
            ],
            dim=-1,
        )

    token_ids = candidate_ids[:, None].expand(-1, ranks.shape[1], -1, -1)
    token_ids = token_ids.gather(-1, ranks[..., None])[..., 0]
    output = DFlash2BeamOutput(token_ids, ranks, selected_steps, path_scores)
    if not include_greedy or beam_size == 1:
        return output

    greedy = walk_dflash2_lattice(candidate_ids, edge_scores)
    matches = (output.candidate_ranks == greedy.candidate_ranks).all(dim=-1)
    missing = ~matches.any(dim=-1)
    if missing.any():
        # Protect the production DFlash2 path by replacing only the last beam.
        output.token_ids[missing, -1] = greedy.token_ids[missing, 0]
        output.candidate_ranks[missing, -1] = greedy.candidate_ranks[missing, 0]
        output.step_scores[missing, -1] = greedy.step_scores[missing, 0]
        output.path_scores[missing, -1] = greedy.path_scores[missing, 0]
    return output


__all__ = [
    "DFLASH2_BEAM_SIZE",
    "DFLASH2_TOP_K",
    "DFlash2BeamOutput",
    "score_dflash2_lattice",
    "select_lattice_beams",
    "selector_walk_triton",
    "walk_dflash2_lattice",
]
