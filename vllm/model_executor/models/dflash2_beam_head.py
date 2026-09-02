# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sixteen-head path reranker for DFlash2 lattice beams."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DFlash2BeamHeadOutput:
    conditional_survival: torch.Tensor
    expected_accept_length: torch.Tensor
    path_delta: torch.Tensor
    override_logit: torch.Tensor


class DFlash2BeamPathHead(nn.Module):
    """Shared path encoder followed by 16-head self-attention.

    The path axis is dynamic data, not a parameter identity.  All beam paths
    therefore share weights; this differs deliberately from the old fixed
    Top-2/128-private-query selector.
    """

    def __init__(
        self,
        *,
        input_hidden_size: int,
        hidden_size: int = 256,
        num_heads: int = 16,
        num_layers: int = 2,
        max_depth: int = 7,
        top_k: int = 16,
        num_numeric_features: int = 5,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.max_depth = int(max_depth)
        self.top_k = int(top_k)
        self.hidden_projection = nn.Linear(input_hidden_size, hidden_size, bias=False)
        self.token_projection = nn.Linear(input_hidden_size, hidden_size, bias=False)
        self.numeric_projection = nn.Linear(
            num_numeric_features, hidden_size, bias=False
        )
        self.position_embedding = nn.Embedding(max_depth + 1, hidden_size)
        self.rank_embedding = nn.Embedding(top_k + 1, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.path_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.survival_head = nn.Linear(hidden_size, 1)
        self.delta_head = nn.Linear(hidden_size, 1)
        self.override_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        *,
        depth_hidden: torch.Tensor,
        token_embeddings: torch.Tensor,
        candidate_ranks: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> DFlash2BeamHeadOutput:
        """Score beam paths.

        Shapes are ``depth_hidden=[B,D,H]``, and all other inputs begin with
        ``[B,beam,D]``.  The numeric tail must match ``num_numeric_features``.
        """
        if candidate_ranks.ndim != 3:
            raise ValueError("candidate_ranks must have shape [B, beam, D].")
        batch_size, beam_size, depth = candidate_ranks.shape
        if depth != self.max_depth:
            raise ValueError(f"expected depth {self.max_depth}, got {depth}.")
        if token_embeddings.shape[:3] != (batch_size, beam_size, depth):
            raise ValueError("token_embeddings must have shape [B, beam, D, H].")
        if numeric_features.shape[:3] != (batch_size, beam_size, depth):
            raise ValueError("numeric_features must have shape [B, beam, D, F].")

        positions = torch.arange(depth, device=candidate_ranks.device) + 1
        dtype = self.hidden_projection.weight.dtype
        states = (
            self.hidden_projection(depth_hidden.to(dtype=dtype))[:, None]
            + self.token_projection(token_embeddings.to(dtype=dtype))
            + self.numeric_projection(numeric_features.to(dtype=dtype))
            + self.position_embedding(positions)[None, None]
            + self.rank_embedding(candidate_ranks + 1)
        )
        states = states.flatten(0, 1)
        causal_mask = torch.ones(
            depth, depth, dtype=torch.bool, device=states.device
        ).triu(1)
        states = self.path_encoder(states, mask=causal_mask).view(
            batch_size, beam_size, depth, -1
        )
        conditional_survival = torch.sigmoid(self.survival_head(states).squeeze(-1))
        expected_accept_length = torch.cumprod(
            conditional_survival.float(), dim=-1
        ).sum(dim=-1)
        pooled = states[:, :, -1]
        return DFlash2BeamHeadOutput(
            conditional_survival=conditional_survival,
            expected_accept_length=expected_accept_length,
            path_delta=self.delta_head(pooled).squeeze(-1),
            override_logit=self.override_head(pooled).squeeze(-1),
        )


@dataclass
class DFlash2BeamSelectorConfig:
    input_hidden_size: int
    vocab_size: int
    selector_rank: int = 16
    hidden_size: int = 256
    num_heads: int = 16
    num_layers: int = 2
    max_depth: int = 7
    top_k: int = 16
    beam_size: int = 16
    num_numeric_features: int = 5


class DFlash2BeamSelectorWeights(nn.Module):
    """Learned DFlash2 pairwise lattice scorer and beam path reranker."""

    def __init__(self, config: DFlash2BeamSelectorConfig) -> None:
        super().__init__()
        self.config = config
        self.unary_bias = nn.Embedding(config.vocab_size, 1)
        self.predecessor = nn.Embedding(config.vocab_size, config.selector_rank)
        self.successor = nn.Embedding(config.vocab_size, config.selector_rank)
        self.context_gate = nn.Linear(
            config.input_hidden_size, config.selector_rank, bias=False
        )
        self.path_head = DFlash2BeamPathHead(
            input_hidden_size=config.input_hidden_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_depth=config.max_depth,
            top_k=config.top_k,
            num_numeric_features=config.num_numeric_features,
        )


__all__ = [
    "DFlash2BeamHeadOutput",
    "DFlash2BeamPathHead",
    "DFlash2BeamSelectorConfig",
    "DFlash2BeamSelectorWeights",
]
