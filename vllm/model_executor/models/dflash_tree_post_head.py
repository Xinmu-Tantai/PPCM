# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Parallel Path Causal Modeling (PPCM) head for DFlash/PPCM seed trees.

PPCM takes the Top-7 candidates and jointly models adjacent positions
through a three-layer encoder:

* Causal Context Encoding Layer (CCEL)
* Candidate Token Interaction Layer (CTIL)
* Causal Path Refinement Layer (CPRL)

A separate path score readout sits after the encoder.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TreePostHeadConfig:
    input_hidden_size: int
    max_depth: int
    tree_width: int
    hidden_size: int = 512
    num_layers: int = 2
    num_heads: int = 8
    num_kv_heads: int = 2
    intermediate_size: int = 1024
    scorer_size: int = 128
    rms_norm_eps: float = 1e-6
    use_sparse_ancestor_attention: bool = False
    use_compact_sibling_layout: bool = False


@dataclass
class TreePostHeadOutput:
    refined_edge_logprobs: torch.Tensor
    refined_path_logprobs: torch.Tensor
    candidate_mass: torch.Tensor
    raw_edge_residuals: torch.Tensor
    node_states: torch.Tensor


class PATRRMSNorm(nn.Module):
    """Small standalone RMSNorm for the independently executed PATR head."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps).to(
            dtype=hidden_states.dtype,
        )
        return normalized * self.weight


class CausalContextEncodingLayer(nn.Module):
    """CCEL: encode each tree node from token, root context, depth, and rank."""

    def __init__(self, config: TreePostHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.token_norm = PATRRMSNorm(
            config.input_hidden_size,
            config.rms_norm_eps,
        )
        self.context_norm = PATRRMSNorm(
            config.input_hidden_size,
            config.rms_norm_eps,
        )
        self.token_projection = nn.Linear(
            config.input_hidden_size,
            config.hidden_size,
            bias=False,
        )
        # Root and all depth states deliberately share this projection.
        self.context_projection = nn.Linear(
            config.input_hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.depth_embedding = nn.Embedding(config.max_depth + 1, config.hidden_size)
        self.rank_embedding = nn.Embedding(config.tree_width + 1, config.hidden_size)
        self.input_norm = PATRRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        *,
        root_hidden: torch.Tensor,
        depth_hidden: torch.Tensor,
        node_embeddings: torch.Tensor,
        depths: torch.Tensor,
        seed_ranks: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_states = self.token_projection(self.token_norm(node_embeddings))
        root_context = self.context_projection(self.context_norm(root_hidden))
        depth_context = self.context_projection(self.context_norm(depth_hidden))
        rank_index = torch.where(
            seed_ranks >= 0,
            seed_ranks.long().clamp(max=self.config.tree_width - 1),
            torch.full_like(seed_ranks.long(), self.config.tree_width),
        )
        hidden_states = self.input_norm(
            token_states
            + root_context[:, None, :]
            + self.depth_embedding(depths.clamp(0, self.config.max_depth))
            + self.rank_embedding(rank_index)
        ).masked_fill(~valid_mask[..., None], 0)
        return hidden_states, token_states, depth_context


class CandidateTokenInteractionAttention(nn.Module):
    """Ancestor-only GQA used by CTIL and CPRL."""

    def __init__(self, config: TreePostHeadConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_mask: torch.Tensor,
        ancestor_mask: torch.Tensor | None = None,
        ancestor_indices: torch.Tensor | None = None,
        ancestor_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .view(
                batch_size,
                num_nodes,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(
                batch_size,
                num_nodes,
                self.num_kv_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(
                batch_size,
                num_nodes,
                self.num_kv_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        if (
            ancestor_indices is not None
            and ancestor_valid_mask is not None
        ):
            # Gather only self-plus-ancestor K/V rows. This preserves the
            # exact visibility contract while reducing attention work from
            # O(N^2) to O(N*D). Keep scores/softmax in FP32, matching the
            # probability precision used by the rest of PATR.
            num_ancestors = ancestor_indices.shape[-1]
            repeat = self.num_heads // self.num_kv_heads
            kv_head_indices = torch.arange(
                self.num_heads,
                device=hidden_states.device,
            ) // repeat
            k = k[:, kv_head_indices]
            v = v[:, kv_head_indices]
            gather_indices = ancestor_indices[:, None, :, :, None].expand(
                batch_size,
                self.num_heads,
                num_nodes,
                num_ancestors,
                self.head_dim,
            )
            expanded_k = k[:, :, None, :, :].expand(
                -1, -1, num_nodes, -1, -1,
            )
            expanded_v = v[:, :, None, :, :].expand(
                -1, -1, num_nodes, -1, -1,
            )
            ancestor_k = torch.gather(expanded_k, 3, gather_indices)
            ancestor_v = torch.gather(expanded_v, 3, gather_indices)
            scores = (
                q[:, :, :, None, :].float() * ancestor_k.float()
            ).sum(dim=-1) * self.head_dim**-0.5
            scores = scores.masked_fill(
                ~ancestor_valid_mask[:, None, :, :],
                float("-inf"),
            )
            weights = torch.softmax(scores, dim=-1).to(ancestor_v.dtype)
            attn = (weights[..., None] * ancestor_v).sum(dim=-2)
        else:
            if ancestor_mask is None:
                raise ValueError(
                    "PATR requires dense ancestor_mask or sparse ancestor indices."
                )
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=ancestor_mask[:, None, :, :],
                dropout_p=0.0,
            )
        attn = attn.transpose(1, 2).reshape(batch_size, num_nodes, -1)
        return self.o_proj(attn).masked_fill(~valid_mask[..., None], 0)


class CandidateTokenInteractionBlock(nn.Module):
    """One ancestor-GQA + SwiGLU encoder layer."""

    def __init__(self, config: TreePostHeadConfig) -> None:
        super().__init__()
        self.attn_norm = PATRRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = CandidateTokenInteractionAttention(config)
        self.mlp_norm = PATRRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_mask: torch.Tensor,
        ancestor_mask: torch.Tensor | None = None,
        ancestor_indices: torch.Tensor | None = None,
        ancestor_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states),
            valid_mask,
            ancestor_mask=ancestor_mask,
            ancestor_indices=ancestor_indices,
            ancestor_valid_mask=ancestor_valid_mask,
        )
        gate, up = self.gate_up_proj(
            self.mlp_norm(hidden_states),
        ).chunk(2, dim=-1)
        hidden_states = hidden_states + self.down_proj(F.silu(gate) * up)
        return hidden_states.masked_fill(~valid_mask[..., None], 0)


class CandidateTokenInteractionLayer(CandidateTokenInteractionBlock):
    """CTIL: second encoder layer over Top-7 candidates at adjacent positions."""


class CausalPathRefinementLayer(CandidateTokenInteractionBlock):
    """CPRL: third encoder layer; refines representations along the causal path."""


def _log1mexp(log_x: torch.Tensor) -> torch.Tensor:
    """Stable log(1-exp(log_x)) for log_x < 0."""
    split = -0.6931471805599453
    return torch.where(
        log_x < split,
        torch.log1p(-torch.exp(log_x)),
        torch.log(-torch.expm1(log_x)),
    )


class PathScoreHead(nn.Module):
    """Readout after the three-layer encoder: edge residual, mass, and path scores."""

    def __init__(self, config: TreePostHeadConfig) -> None:
        super().__init__()
        self.config = config
        # Keep the scalar on-device so CUDA Graph capture does not allocate a
        # new tensor inside forward. ``persistent=False`` preserves the exact
        # checkpoint/state_dict contract of existing PATR weights.
        self.register_buffer(
            "_negative_float32_eps",
            torch.tensor(-torch.finfo(torch.float32).eps),
            persistent=False,
        )
        self.candidate_norm = PATRRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.edge_query = nn.Linear(
            config.hidden_size,
            config.scorer_size,
            bias=False,
        )
        self.edge_key = nn.Linear(
            config.hidden_size,
            config.scorer_size,
            bias=False,
        )
        self.edge_scale = nn.Parameter(torch.zeros(config.max_depth))
        self.scalar_mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.mass_head = nn.Linear(config.hidden_size + 3, 1)
        self.reset_residual_parameters()

    def reset_residual_parameters(self) -> None:
        """Make a newly initialized head recover every seed edge exactly."""
        nn.init.zeros_(self.edge_scale)
        nn.init.zeros_(self.scalar_mlp[-1].weight)
        nn.init.zeros_(self.scalar_mlp[-1].bias)
        nn.init.zeros_(self.mass_head.weight)
        nn.init.zeros_(self.mass_head.bias)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        token_states: torch.Tensor,
        depth_context: torch.Tensor,
        parent_indices: torch.Tensor,
        depths: torch.Tensor,
        valid_mask: torch.Tensor,
        seed_edge_logprobs: torch.Tensor,
        seed_ranks: torch.Tensor,
    ) -> TreePostHeadOutput:
        batch_size, num_nodes, _ = hidden_states.shape
        parent_depth_index = depths.clamp(max=self.config.max_depth - 1)
        depth_prior = torch.gather(
            depth_context,
            1,
            parent_depth_index[..., None].expand(-1, -1, depth_context.shape[-1]),
        )
        candidate_states = self.candidate_norm(hidden_states + depth_prior)
        safe_parents = parent_indices.clamp_min(0)
        parent_states = torch.gather(
            candidate_states,
            1,
            safe_parents[..., None].expand(-1, -1, candidate_states.shape[-1]),
        )
        q = self.edge_query(parent_states)
        k = self.edge_key(token_states)
        path_delta = (q * k).sum(dim=-1) * self.config.scorer_size**-0.5
        parent_depths = torch.gather(depths, 1, safe_parents)
        scaled_path_delta = (
            path_delta
            * self.edge_scale[parent_depths.clamp(max=self.config.max_depth - 1)]
        )

        child_valid = valid_mask.clone()
        child_valid[:, 0] = False
        seed_lp = seed_edge_logprobs.float()
        child_indices: torch.Tensor | None = None
        child_slot_valid: torch.Tensor | None = None
        dense_parent_mask: torch.Tensor | None = None
        if self.config.use_compact_sibling_layout:
            # The tree drafter emits at most tree_width children per parent and records
            # their unique rank. Pack them as [B,parent,rank], reducing sibling
            # workspace from O(N^2) to O(N*K).
            child_ranks = seed_ranks.long().clamp(
                0, self.config.tree_width - 1,
            )
            child_slots = safe_parents * self.config.tree_width + child_ranks
            sentinel_slot = num_nodes * self.config.tree_width
            child_slots = torch.where(
                child_valid,
                child_slots,
                torch.full_like(child_slots, sentinel_slot),
            )
            flat_slot_count = sentinel_slot + 1
            node_indices = torch.arange(
                num_nodes,
                device=parent_indices.device,
            ).expand(batch_size, -1)
            packed_child_indices = torch.zeros(
                (batch_size, flat_slot_count),
                dtype=torch.long,
                device=parent_indices.device,
            )
            packed_child_indices.scatter_(1, child_slots, node_indices)
            packed_child_valid = torch.zeros(
                (batch_size, flat_slot_count),
                dtype=torch.int16,
                device=parent_indices.device,
            )
            packed_child_valid.scatter_add_(
                1,
                child_slots,
                child_valid.to(torch.int16),
            )
            child_indices = packed_child_indices[:, :sentinel_slot].view(
                batch_size,
                num_nodes,
                self.config.tree_width,
            )
            child_slot_valid = packed_child_valid[:, :sentinel_slot].view(
                batch_size,
                num_nodes,
                self.config.tree_width,
            ).bool()
            sibling_count = child_slot_valid.sum(dim=-1)
            grouped_seed_lp = torch.gather(
                seed_lp,
                1,
                child_indices.flatten(1),
            ).view_as(child_indices).masked_fill(
                ~child_slot_valid,
                float("-inf"),
            )
        else:
            dense_parent_mask = F.one_hot(
                safe_parents,
                num_classes=num_nodes,
            ).transpose(1, 2).bool()
            dense_parent_mask &= child_valid[:, None, :]
            sibling_count = dense_parent_mask.sum(dim=-1)
            grouped_seed_lp = seed_lp[:, None, :].masked_fill(
                ~dense_parent_mask,
                float("-inf"),
            )
        top_values = torch.topk(
            grouped_seed_lp,
            k=min(2, grouped_seed_lp.shape[-1]),
            dim=-1,
        ).values
        if self.config.tree_width > 1:
            sibling_gap = torch.where(
                sibling_count > 1,
                top_values[..., 0] - top_values[..., 1],
                torch.zeros_like(top_values[..., 0]),
            )
        else:
            sibling_gap = torch.zeros_like(top_values[..., 0])
        child_gap = torch.gather(sibling_gap, 1, safe_parents)
        child_count = torch.gather(sibling_count, 1, safe_parents)

        finite_seed_lp = torch.where(
            child_valid,
            seed_lp,
            torch.zeros_like(seed_lp),
        )
        scalar_features = torch.stack(
            (
                finite_seed_lp,
                torch.where(
                    child_valid,
                    seed_ranks.float(),
                    torch.zeros_like(seed_lp),
                ) / self.config.tree_width,
                parent_depths.float() / self.config.max_depth,
                child_gap,
                child_count.float() / self.config.tree_width,
            ),
            dim=-1,
        ).to(dtype=token_states.dtype)
        delta = (
            scaled_path_delta + self.scalar_mlp(scalar_features).squeeze(-1)
        ).float()
        raw_residual = torch.where(child_valid, delta, torch.zeros_like(delta))

        log_rho0 = torch.logsumexp(grouped_seed_lp, dim=-1)
        # A strict subset has mass below one. Protect only FP rounding.
        log_rho0 = torch.minimum(
            log_rho0,
            self._negative_float32_eps,
        )
        mass_features = torch.stack(
            (
                sibling_gap,
                depths.float() / self.config.max_depth,
                sibling_count.float() / self.config.tree_width,
            ),
            dim=-1,
        ).to(dtype=candidate_states.dtype)
        mass_delta = self.mass_head(
            torch.cat((candidate_states, mass_features), dim=-1),
        ).squeeze(-1).float()
        log_mass_numerator = log_rho0 + mass_delta
        log_mass = log_mass_numerator - torch.logaddexp(
            _log1mexp(log_rho0),
            log_mass_numerator,
        )

        log_within_unnormalized = finite_seed_lp + delta
        if self.config.use_compact_sibling_layout:
            assert child_indices is not None
            assert child_slot_valid is not None
            grouped_log_within = torch.gather(
                log_within_unnormalized,
                1,
                child_indices.flatten(1),
            ).view_as(child_indices).masked_fill(
                ~child_slot_valid,
                float("-inf"),
            )
        else:
            assert dense_parent_mask is not None
            grouped_log_within = log_within_unnormalized[:, None, :].masked_fill(
                ~dense_parent_mask,
                float("-inf"),
            )
        group_log_normalizer = torch.logsumexp(grouped_log_within, dim=-1)
        child_log_mass = torch.gather(log_mass, 1, safe_parents)
        child_log_normalizer = torch.gather(
            group_log_normalizer,
            1,
            safe_parents,
        )
        refined_candidates = (
            child_log_mass + log_within_unnormalized - child_log_normalizer
        )
        refined_edge = torch.where(
            child_valid,
            refined_candidates,
            torch.full_like(refined_candidates, float("-inf")),
        )
        refined_edge[:, 0] = 0.0
        candidate_mass = torch.where(
            sibling_count > 0,
            torch.exp(log_mass),
            torch.zeros_like(log_mass),
        )
        candidate_mass[:, 0] = torch.where(
            sibling_count[:, 0] > 0,
            candidate_mass[:, 0],
            torch.ones_like(candidate_mass[:, 0]),
        )

        # Parent-before-child ordering means all nodes at one depth can be
        # accumulated together.  This reduces 254 sequential iterations to
        # max_depth (15 in production), while preserving FP32 path scores.
        path_logprobs = torch.full_like(refined_edge, float("-inf"))
        path_logprobs[:, 0] = 0.0
        for depth_idx in range(1, self.config.max_depth + 1):
            parent_paths = torch.gather(path_logprobs, 1, safe_parents)
            at_depth = valid_mask & (depths == depth_idx)
            path_logprobs = torch.where(
                at_depth,
                parent_paths + refined_edge,
                path_logprobs,
            )

        return TreePostHeadOutput(
            refined_edge_logprobs=refined_edge,
            refined_path_logprobs=path_logprobs,
            candidate_mass=candidate_mass,
            raw_edge_residuals=raw_residual,
            node_states=hidden_states,
        )


# Paper-facing aliases.
CCEL = CausalContextEncodingLayer
CTIL = CandidateTokenInteractionLayer
CPRL = CausalPathRefinementLayer
# Previous internal names.
TreeAdapterAttention = CandidateTokenInteractionAttention
TreeAdapterBlock = CandidateTokenInteractionBlock


class DFlashTreePostHead(nn.Module):
    """PPCM: three-layer encoder CCEL → CTIL → CPRL, then path-score readout."""

    def __init__(self, config: TreePostHeadConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("PATR hidden_size must be divisible by num_heads.")
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError("PATR num_heads must be divisible by num_kv_heads.")
        self.config = config
        self.ccel = CausalContextEncodingLayer(config)
        self.ctil = CandidateTokenInteractionLayer(config)
        self.cprl = CausalPathRefinementLayer(config)
        self.score = PathScoreHead(config)

    def reset_residual_parameters(self) -> None:
        self.score.reset_residual_parameters()

    @property
    def edge_scale(self) -> nn.Parameter:
        return self.score.edge_scale

    @property
    def scalar_mlp(self) -> nn.Sequential:
        return self.score.scalar_mlp

    @property
    def mass_head(self) -> nn.Linear:
        return self.score.mass_head

    @property
    def candidate_norm(self) -> PATRRMSNorm:
        return self.score.candidate_norm

    def _run_encoder_layer(
        self,
        layer: CandidateTokenInteractionBlock,
        hidden_states: torch.Tensor,
        valid_mask: torch.Tensor,
        ancestor_mask: torch.Tensor | None,
        ancestor_indices: torch.Tensor | None,
        ancestor_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        use_sparse = (
            self.config.use_sparse_ancestor_attention
            and ancestor_indices is not None
            and ancestor_valid_mask is not None
        )
        return layer(
            hidden_states,
            valid_mask,
            ancestor_mask=None if use_sparse else ancestor_mask,
            ancestor_indices=ancestor_indices if use_sparse else None,
            ancestor_valid_mask=ancestor_valid_mask if use_sparse else None,
        )

    def forward(
        self,
        *,
        root_hidden: torch.Tensor,
        depth_hidden: torch.Tensor,
        node_embeddings: torch.Tensor,
        parent_indices: torch.Tensor,
        depths: torch.Tensor,
        valid_mask: torch.Tensor,
        seed_edge_logprobs: torch.Tensor,
        seed_ranks: torch.Tensor,
        ancestor_mask: torch.Tensor | None = None,
        ancestor_indices: torch.Tensor | None = None,
        ancestor_valid_mask: torch.Tensor | None = None,
    ) -> TreePostHeadOutput:
        if depth_hidden.shape[1] != self.config.max_depth:
            raise ValueError(
                f"PATR expected depth {self.config.max_depth}, got "
                f"{depth_hidden.shape[1]}."
            )
        hidden_states, token_states, depth_context = self.ccel(
            root_hidden=root_hidden,
            depth_hidden=depth_hidden,
            node_embeddings=node_embeddings,
            depths=depths,
            seed_ranks=seed_ranks,
            valid_mask=valid_mask,
        )
        hidden_states = self._run_encoder_layer(
            self.ctil,
            hidden_states,
            valid_mask,
            ancestor_mask,
            ancestor_indices,
            ancestor_valid_mask,
        )
        hidden_states = self._run_encoder_layer(
            self.cprl,
            hidden_states,
            valid_mask,
            ancestor_mask,
            ancestor_indices,
            ancestor_valid_mask,
        )
        return self.score(
            hidden_states=hidden_states,
            token_states=token_states,
            depth_context=depth_context,
            parent_indices=parent_indices,
            depths=depths,
            valid_mask=valid_mask,
            seed_edge_logprobs=seed_edge_logprobs,
            seed_ranks=seed_ranks,
        )
