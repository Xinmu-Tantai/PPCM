# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class SpecDecodeMetadata:
    # [num_tokens]
    draft_token_ids: torch.Tensor
    # [batch_size]
    num_draft_tokens: list[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]
    target_logits_indices: torch.Tensor
    # [batch_size]
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]
    logits_indices: torch.Tensor

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flattened_draft_token_ids = sum(draft_token_ids, [])
        num_tokens = len(flattened_draft_token_ids)

        draft_token_ids_tensor = torch.tensor(
            flattened_draft_token_ids, dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(
            device
        )

        target_logits_indices = torch.zeros(
            num_tokens, dtype=torch.int32, device=device
        )
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(
            num_tokens + batch_size, dtype=torch.int32, device=device
        )
        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )


@dataclass
class DFlashRequestTreeSpec:
    # Parents for speculative tree nodes only (root excluded). Parent indices
    # are expressed in the target-query space where 0 is the already-sampled
    # root token and the first speculative node starts at index 1.
    parent_indices: list[int]
    # Depth for speculative tree nodes only (root excluded). Root depth is 0.
    depths: list[int]

    @property
    def num_draft_nodes(self) -> int:
        return len(self.parent_indices)

    def full_parent_indices(self) -> list[int]:
        return [-1, *self.parent_indices]

    def full_depths(self) -> list[int]:
        return [0, *self.depths]

    def nodes_per_depth(self, num_depths: int) -> list[int]:
        """Count draft nodes at each depth (root excluded)."""
        counts = [0] * num_depths
        for d in self.depths:
            if 0 < d <= num_depths:
                counts[d - 1] += 1
        return counts


@dataclass
class DFlashTreeSpecDecodeMetadata(SpecDecodeMetadata):
    # Per-request query lengths including the implicit root token.
    query_lens: list[int] = field(default_factory=list)
    # Flattened parent indices including roots. Length = sum(query_lens).
    parent_indices: torch.Tensor | None = None
    # Flattened depths including roots. Length = sum(query_lens).
    depths: torch.Tensor | None = None
    # Block-diagonal attention bias over all scheduled query tokens.
    tree_attn_bias: torch.Tensor | None = None
    # Per-request query ranges in the flattened target logits tensor.
    cu_query_lens: torch.Tensor | None = None
    # Requests with actual DFlash tree topology. Requests without spec tokens use
    # a causal-chain fallback block and are marked False here.
    is_tree_req: list[bool] = field(default_factory=list)
    # Per-request ancestor matrices for optimus SM90 kernel path.
    # Shape: (B, N_padded, N_padded) int32.  None when using the triton bias path.
    ancestor_masks: torch.Tensor | None = None
    # Experimental DFlash logical KV layout. One entry per request; each tensor
    # maps a contiguous logical context suffix to physical KV slots.
    logical_kv_slots: torch.Tensor | list[torch.Tensor | None] | None = None
    logical_kv_slot_lens: torch.Tensor | None = None
    logical_kv_starts: torch.Tensor | list[int] | None = None
    # Root-inclusive preferred tree path used by logical KV layout to place
    # likely accepted nodes directly into canonical future sequence slots.
    canonical_lane_indices: list[list[int]] | None = None

    def __post_init__(self):
        super().__post_init__()
        if not self.query_lens:
            self.query_lens = [n + 1 for n in self.num_draft_tokens]
        if not self.is_tree_req:
            self.is_tree_req = [n > 0 for n in self.num_draft_tokens]
