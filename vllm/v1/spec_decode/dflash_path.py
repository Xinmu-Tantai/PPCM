# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixed 7-step, Top-2 path construction for DFlash."""

from dataclasses import dataclass

import torch

DFLASH_PATH_DEPTH = 7
DFLASH_PATH_TOP_K = 2
DFLASH_PATH_COUNT = 1 << DFLASH_PATH_DEPTH
DFLASH_PATH_MEMORY_NODES = 1 + DFLASH_PATH_DEPTH * DFLASH_PATH_TOP_K


@dataclass
class DFlashCandidatePaths:
    token_ids: torch.Tensor
    ranks: torch.Tensor
    node_indices: torch.Tensor
    draft_scores: torch.Tensor


def binary_path_rank_patterns(device: torch.device) -> torch.Tensor:
    """Return all 128 fixed Top-1/Top-2 rank patterns in lexical order."""
    path_ids = torch.arange(DFLASH_PATH_COUNT, device=device)
    shifts = torch.arange(
        DFLASH_PATH_DEPTH - 1,
        -1,
        -1,
        device=device,
    )
    return torch.bitwise_and(path_ids[:, None] >> shifts, 1)


def build_all_top2_paths(
    topk_token_ids: torch.Tensor,
    topk_logprobs: torch.Tensor,
) -> DFlashCandidatePaths:
    """Materialize all 2^7 paths without a Cartesian-product Python loop."""
    expected_tail = (DFLASH_PATH_DEPTH, DFLASH_PATH_TOP_K)
    if topk_token_ids.ndim != 3 or topk_token_ids.shape[1:] != expected_tail:
        raise ValueError(
            "topk_token_ids must have shape [B, 7, 2], got "
            f"{tuple(topk_token_ids.shape)}."
        )
    if topk_logprobs.shape != topk_token_ids.shape:
        raise ValueError("topk_logprobs must match topk_token_ids.")

    batch_size = topk_token_ids.shape[0]
    ranks = binary_path_rank_patterns(topk_token_ids.device)
    ranks = ranks[None].expand(batch_size, -1, -1)
    candidates = topk_token_ids[:, None].expand(
        -1,
        DFLASH_PATH_COUNT,
        -1,
        -1,
    )
    token_ids = torch.gather(candidates, 3, ranks[..., None]).squeeze(-1)
    candidate_logprobs = topk_logprobs[:, None].expand_as(candidates)
    selected_logprobs = torch.gather(
        candidate_logprobs,
        3,
        ranks[..., None],
    ).squeeze(-1)
    depth_offsets = (
        torch.arange(DFLASH_PATH_DEPTH, device=topk_token_ids.device)
        * DFLASH_PATH_TOP_K
        + 1
    )
    return DFlashCandidatePaths(
        token_ids=token_ids,
        ranks=ranks,
        node_indices=ranks + depth_offsets,
        draft_scores=selected_logprobs.sum(dim=-1),
    )
