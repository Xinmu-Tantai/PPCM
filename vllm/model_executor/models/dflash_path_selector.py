# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Post-selector for fixed Top-2 DFlash block drafting.

V3 provides path-aligned attention with shared candidate K/V. V4 provides a
lower-latency pair scorer that encodes the two candidates at each depth once
and expands their probabilities into all 128 path scores. V5 adds a
conservative request-level switch gate. V8 directly scores and argmaxes all
128 paths without a gate or threshold. None of these modes produces
Target-model KV; outputs only rank draft paths before exact verification.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DFlashPathSelectorConfig:
    input_hidden_size: int
    max_depth: int = 7
    max_rank: int = 2
    num_paths: int = 128
    hidden_size: int = 512
    head_dim: int = 64
    num_layers: int = 1
    intermediate_size: int = 256
    num_numeric_features: int = 2
    # False preserves checkpoints produced by the original per-path-query
    # implementation. New checkpoints explicitly set this to true.
    shared_query: bool = False
    selector_type: str = "path_attention"
    gate_feature_version: int = 1
    gate_predict_delta: bool = False
    switch_margin: float = 0.0
    teacher_hidden_size: int = 128
    rms_norm_eps: float = 1e-6


@dataclass
class DFlashPathSelectorOutput:
    conditional_logits: torch.Tensor
    conditional_survival: torch.Tensor
    cumulative_survival: torch.Tensor
    expected_accept_length: torch.Tensor
    best_path_indices: torch.Tensor
    path_states: torch.Tensor
    choice_logits: torch.Tensor | None = None
    switch_logits: torch.Tensor | None = None
    switch_deltas: torch.Tensor | None = None
    aligned_hidden: torch.Tensor | None = None
    teacher_logprob_predictions: torch.Tensor | None = None


@dataclass
class DFlashPathSelectorLoss:
    loss: torch.Tensor
    hazard_loss: torch.Tensor
    ranking_loss: torch.Tensor


class PathSelectorRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps).to(
            hidden_states.dtype
        )
        return normalized * self.weight


def build_path_attention_mask(
    path_node_indices: torch.Tensor,
    num_memory_nodes: int,
    *,
    root_index: int = 0,
    path_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build [B, P, D, S] path-prefix visibility masks."""
    if path_node_indices.ndim != 3:
        raise ValueError("path_node_indices must have shape [B, P, D].")
    if num_memory_nodes <= 0:
        raise ValueError("num_memory_nodes must be positive.")
    if path_node_indices.numel() > 0:
        min_index = int(path_node_indices.min().item())
        max_index = int(path_node_indices.max().item())
        if min_index < 0 or max_index >= num_memory_nodes:
            raise ValueError(
                "path_node_indices contains an index outside the shared "
                f"memory: min={min_index}, max={max_index}, "
                f"memory_nodes={num_memory_nodes}."
            )

    memory_indices = torch.arange(
        num_memory_nodes,
        device=path_node_indices.device,
    )
    step_nodes = path_node_indices[..., None] == memory_indices
    allowed = step_nodes.to(torch.int32).cumsum(dim=2).bool()
    allowed[..., root_index] = True
    if path_valid_mask is not None:
        if path_valid_mask.shape != path_node_indices.shape:
            raise ValueError(
                "path_valid_mask must have the same [B, P, D] shape as "
                "path_node_indices."
            )
        allowed &= path_valid_mask[..., None]
        # Keep one legal key for padded queries so softmax never receives an
        # all-masked row. Their outputs are zeroed after attention.
        allowed[..., root_index] = True
    return allowed


def gather_path_nodes(
    node_states: torch.Tensor,
    path_node_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather shared candidate nodes into [B, P, D, hidden] paths."""
    if node_states.ndim != 3 or path_node_indices.ndim != 3:
        raise ValueError(
            "node_states and path_node_indices must be [B, S, H] and "
            "[B, P, D]."
        )
    batch_size, _, hidden_size = node_states.shape
    if path_node_indices.shape[0] != batch_size:
        raise ValueError("node_states and paths must have the same batch size.")
    expanded = node_states[:, None].expand(
        -1,
        path_node_indices.shape[1],
        -1,
        -1,
    )
    indices = path_node_indices[..., None].expand(-1, -1, -1, hidden_size)
    return torch.gather(expanded, 2, indices)


class PathAlignedMQABlock(nn.Module):
    """Path-masked query lanes over one shared candidate K/V memory."""

    def __init__(self, config: DFlashPathSelectorConfig) -> None:
        super().__init__()
        self.num_paths = config.num_paths
        self.head_dim = config.head_dim
        self.query_norm = PathSelectorRMSNorm(
            config.head_dim,
            config.rms_norm_eps,
        )
        self.memory_norm = PathSelectorRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.shared_query = config.shared_query
        query_shape = (
            (config.head_dim, config.head_dim)
            if config.shared_query
            else (config.num_paths, config.head_dim, config.head_dim)
        )
        self.query_weight = nn.Parameter(torch.empty(query_shape))
        self.key_proj = nn.Linear(
            config.hidden_size,
            config.head_dim,
            bias=False,
        )
        self.value_proj = nn.Linear(
            config.hidden_size,
            config.head_dim,
            bias=False,
        )
        self.output_proj = nn.Linear(
            config.head_dim,
            config.head_dim,
            bias=False,
        )
        self.attn_output_norm = PathSelectorRMSNorm(
            config.head_dim,
            config.rms_norm_eps,
        )
        self.ffn_norm = PathSelectorRMSNorm(
            config.head_dim,
            config.rms_norm_eps,
        )
        self.gate_up_proj = nn.Linear(
            config.head_dim,
            2 * config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.head_dim,
            bias=False,
        )
        if config.shared_query:
            nn.init.xavier_uniform_(self.query_weight)
        else:
            for weight in self.query_weight:
                nn.init.xavier_uniform_(weight)

    def forward(
        self,
        query_states: torch.Tensor,
        memory_states: torch.Tensor,
        path_mask: torch.Tensor,
        path_valid_mask: torch.Tensor,
        node_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query_states.shape[1] != self.num_paths:
            raise ValueError(
                f"Expected {self.num_paths} paths, got {query_states.shape[1]}."
            )
        normalized_queries = self.query_norm(query_states)
        if self.shared_query:
            queries = torch.einsum(
                "bptc,ch->bpth",
                normalized_queries,
                self.query_weight,
            )
        else:
            queries = torch.einsum(
                "bptc,pch->bpth",
                normalized_queries,
                self.query_weight,
            )
        normalized_memory = self.memory_norm(memory_states)
        keys = self.key_proj(normalized_memory)
        values = self.value_proj(normalized_memory)
        scores = torch.einsum(
            "bptc,bsc->bpts",
            queries.float(),
            keys.float(),
        )
        scores.mul_(self.head_dim**-0.5)
        visible = path_mask & node_valid_mask[:, None, None, :]
        scores.masked_fill_(~visible, float("-inf"))
        weights = torch.softmax(scores, dim=-1).to(values.dtype)
        attention_output = torch.einsum("bpts,bsc->bptc", weights, values)
        hidden_states = self.attn_output_norm(
            query_states + self.output_proj(attention_output)
        )
        gate, up = self.gate_up_proj(self.ffn_norm(hidden_states)).chunk(
            2,
            dim=-1,
        )
        hidden_states = hidden_states + self.down_proj(F.silu(gate) * up)
        return hidden_states.masked_fill(~path_valid_mask[..., None], 0)


class PathCausalSelfAttentionBlock(nn.Module):
    """Private path queries with shared K/V/O over each causal path lane."""

    def __init__(self, config: DFlashPathSelectorConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.input_norm = PathSelectorRMSNorm(config.head_dim, config.rms_norm_eps)
        self.query_weight = nn.Parameter(
            torch.empty(config.num_paths, config.head_dim, config.head_dim)
        )
        self.key_proj = nn.Linear(config.head_dim, config.head_dim, bias=False)
        self.value_proj = nn.Linear(config.head_dim, config.head_dim, bias=False)
        self.output_proj = nn.Linear(config.head_dim, config.head_dim, bias=False)
        self.attn_output_norm = PathSelectorRMSNorm(
            config.head_dim, config.rms_norm_eps
        )
        self.ffn_norm = PathSelectorRMSNorm(config.head_dim, config.rms_norm_eps)
        self.gate_up_proj = nn.Linear(
            config.head_dim, 2 * config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.head_dim, bias=False
        )
        for weight in self.query_weight:
            nn.init.xavier_uniform_(weight)

    def forward(
        self, hidden_states: torch.Tensor, causal_mask: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.input_norm(hidden_states)
        queries = torch.einsum(
            "bplc,pch->bplh", normalized, self.query_weight
        )
        keys = self.key_proj(normalized)
        values = self.value_proj(normalized)
        scores = torch.einsum("bplc,bptc->bplt", queries.float(), keys.float())
        scores.mul_(self.head_dim**-0.5)
        scores.masked_fill_(~causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).to(values.dtype)
        attention_output = torch.einsum("bplt,bptc->bplc", weights, values)
        hidden_states = self.attn_output_norm(
            hidden_states + self.output_proj(attention_output)
        )
        gate, up = self.gate_up_proj(self.ffn_norm(hidden_states)).chunk(2, dim=-1)
        return hidden_states + self.down_proj(F.silu(gate) * up)


class DFlashPathSelector(nn.Module):
    """DFlash post-head supporting V3 path attention and V4 pair scoring."""

    def __init__(self, config: DFlashPathSelectorConfig) -> None:
        super().__init__()
        if config.max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        if config.max_rank <= 0:
            raise ValueError("max_rank must be positive.")
        if config.num_paths <= 0:
            raise ValueError("num_paths must be positive.")
        if config.selector_type not in {
            "path_attention",
            "pair_scorer",
            "gated_pair_scorer",
            "direct_path_scorer",
            "path_distilled_attention",
        }:
            raise ValueError(
                "selector_type must be 'path_attention', 'pair_scorer', "
                "'gated_pair_scorer', 'direct_path_scorer', or "
                "'path_distilled_attention'."
            )
        if config.selector_type in {
            "path_attention", "path_distilled_attention"
        } and config.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if config.selector_type != "gated_pair_scorer" and config.switch_margin < 0:
            raise ValueError("switch_margin must be non-negative.")
        if config.gate_feature_version not in {1, 2}:
            raise ValueError("gate_feature_version must be 1 or 2.")
        if config.gate_predict_delta and config.selector_type != "gated_pair_scorer":
            raise ValueError("gate_predict_delta requires gated_pair_scorer.")
        if (
            config.max_depth != 7
            or config.max_rank != 2
            or config.num_paths != 128
        ):
            raise ValueError(
                "DFlashPathSelector is fixed to depth=7, Top-2, and 128 paths."
            )
        self.config = config
        path_ids = torch.arange(config.num_paths)
        shifts = torch.arange(config.max_depth - 1, -1, -1)
        fixed_path_ranks = torch.bitwise_and(path_ids[:, None] >> shifts, 1)
        depth_offsets = torch.arange(config.max_depth) * config.max_rank + 1
        fixed_path_node_indices = fixed_path_ranks + depth_offsets
        fixed_path_mask = build_path_attention_mask(
            fixed_path_node_indices[None],
            1 + config.max_depth * config.max_rank,
        )[0]
        self.register_buffer(
            "fixed_path_ranks",
            fixed_path_ranks,
            persistent=False,
        )
        self.register_buffer(
            "fixed_path_node_indices",
            fixed_path_node_indices,
            persistent=False,
        )
        self.register_buffer(
            "fixed_path_mask",
            fixed_path_mask,
            persistent=False,
        )
        self.token_norm = PathSelectorRMSNorm(
            config.input_hidden_size,
            config.rms_norm_eps,
        )
        self.context_norm = PathSelectorRMSNorm(
            config.input_hidden_size,
            config.rms_norm_eps,
        )
        self.token_proj = nn.Linear(
            config.input_hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.context_proj = nn.Linear(
            config.input_hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.depth_embedding = nn.Embedding(
            config.max_depth + 1,
            config.hidden_size,
        )
        # The final row is reserved for root/padding.
        self.rank_embedding = nn.Embedding(
            config.max_rank + 1,
            config.hidden_size,
        )
        self.numeric_proj = nn.Linear(
            config.num_numeric_features,
            config.hidden_size,
            bias=False,
        )
        self.input_norm = PathSelectorRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        if config.selector_type == "path_attention":
            self.query_input_proj = nn.Linear(
                config.hidden_size,
                config.head_dim,
                bias=False,
            )
            self.blocks = nn.ModuleList(
                PathAlignedMQABlock(config) for _ in range(config.num_layers)
            )
            self.output_norm = PathSelectorRMSNorm(
                config.head_dim,
                config.rms_norm_eps,
            )
            self.survival_head = nn.Linear(config.head_dim, 1)
        elif config.selector_type == "path_distilled_attention":
            self.path_input_proj = nn.Linear(
                config.hidden_size, config.head_dim, bias=False
            )
            self.path_blocks = nn.ModuleList(
                PathCausalSelfAttentionBlock(config)
                for _ in range(config.num_layers)
            )
            self.path_output_norm = PathSelectorRMSNorm(
                config.head_dim, config.rms_norm_eps
            )
            self.survival_head = nn.Linear(config.head_dim, 1)
            self.teacher_logprob_head = nn.Linear(config.head_dim, 1)
            self.hidden_alignment_head = nn.Linear(
                config.head_dim, config.teacher_hidden_size, bias=False
            )
            path_length = config.max_depth + 1
            self.register_buffer(
                "path_causal_mask",
                torch.ones(path_length, path_length, dtype=torch.bool)[
                    None, None
                ]
                .tril()
                .expand(1, config.num_paths, -1, -1)
                .clone(),
                persistent=False,
            )
        else:
            pair_width = 4 * config.hidden_size
            self.pair_input_norm = PathSelectorRMSNorm(
                pair_width,
                config.rms_norm_eps,
            )
            self.pair_gate_up_proj = nn.Linear(
                pair_width,
                2 * config.intermediate_size,
                bias=False,
            )
            self.pair_down_proj = nn.Linear(
                config.intermediate_size,
                config.head_dim,
                bias=False,
            )
            self.pair_output_norm = PathSelectorRMSNorm(
                config.head_dim,
                config.rms_norm_eps,
            )
            self.choice_head = nn.Linear(config.head_dim, 3)
            if config.selector_type == "direct_path_scorer":
                path_width = config.max_depth * (config.head_dim + 3)
                self.path_score_input_norm = PathSelectorRMSNorm(
                    path_width,
                    config.rms_norm_eps,
                )
                self.path_score_hidden = nn.Linear(
                    path_width,
                    config.head_dim,
                    bias=False,
                )
                self.path_score_output = nn.Linear(config.head_dim, 1)
                nn.init.zeros_(self.path_score_output.weight)
                nn.init.zeros_(self.path_score_output.bias)
            if config.selector_type == "gated_pair_scorer":
                gate_width = config.max_depth * (config.head_dim + 3)
                if config.gate_feature_version == 2:
                    gate_width += config.max_depth * 6 + 5
                self.switch_gate_input_norm = PathSelectorRMSNorm(
                    gate_width,
                    config.rms_norm_eps,
                )
                self.switch_gate_hidden = nn.Linear(
                    gate_width,
                    config.head_dim,
                    bias=False,
                )
                self.switch_gate_output = nn.Linear(config.head_dim, 1)
                if config.gate_predict_delta:
                    self.switch_delta_output = nn.Linear(config.head_dim, 1)

    def _build_gate_features(
        self,
        pair_states: torch.Tensor,
        choice_logits: torch.Tensor,
        choice_probabilities: torch.Tensor,
        expected_accept_length: torch.Tensor,
        alternative_index: torch.Tensor,
    ) -> torch.Tensor:
        base = [
            pair_states.flatten(1),
            choice_logits.to(pair_states.dtype).flatten(1),
        ]
        if self.config.gate_feature_version == 1:
            return torch.cat(base, dim=-1)

        batch_size = pair_states.shape[0]
        row = torch.arange(batch_size, device=pair_states.device)
        selected_ranks = self.fixed_path_ranks[alternative_index]
        depth = torch.arange(self.config.max_depth, device=pair_states.device)
        selected_probability = choice_probabilities[
            row[:, None], depth[None], selected_ranks
        ]
        top1_probability = choice_probabilities[..., 0]
        selected_cumulative = torch.cumprod(selected_probability, dim=-1)
        top1_cumulative = torch.cumprod(top1_probability, dim=-1)
        alternative_score = expected_accept_length[row, alternative_index]
        top1_score = expected_accept_length[:, 0]
        differs = selected_ranks.bool()
        first_divergence = torch.where(
            differs.any(dim=-1),
            differs.float().argmax(dim=-1),
            torch.full(
                (batch_size,),
                self.config.max_depth,
                device=pair_states.device,
            ),
        ).to(pair_states.dtype)
        scale = float(self.config.max_depth)
        per_depth = torch.stack(
            [
                selected_ranks.to(pair_states.dtype),
                selected_probability.to(pair_states.dtype),
                top1_probability.to(pair_states.dtype),
                (selected_probability - top1_probability).to(pair_states.dtype),
                selected_cumulative.to(pair_states.dtype),
                top1_cumulative.to(pair_states.dtype),
            ],
            dim=-1,
        ).flatten(1)
        decision = torch.stack(
            [
                alternative_score,
                top1_score,
                alternative_score - top1_score,
                first_divergence / scale,
                selected_ranks.float().mean(dim=-1),
            ],
            dim=-1,
        ).to(pair_states.dtype)
        return torch.cat([*base, per_depth, decision], dim=-1)

    def _build_memory(
        self,
        *,
        root_hidden: torch.Tensor,
        depth_hidden: torch.Tensor,
        node_embeddings: torch.Tensor,
        node_depths: torch.Tensor,
        node_ranks: torch.Tensor,
        node_numeric_features: torch.Tensor,
        node_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = node_embeddings.shape
        if depth_hidden.shape[1] != self.config.max_depth:
            raise ValueError(
                f"Expected depth {self.config.max_depth}, got "
                f"{depth_hidden.shape[1]}."
            )
        if node_depths.shape != (batch_size, num_nodes):
            raise ValueError("node_depths must have shape [B, S].")
        if node_ranks.shape != (batch_size, num_nodes):
            raise ValueError("node_ranks must have shape [B, S].")
        if node_numeric_features.shape != (
            batch_size,
            num_nodes,
            self.config.num_numeric_features,
        ):
            raise ValueError(
                "node_numeric_features must have shape "
                f"[B, S, {self.config.num_numeric_features}]."
            )

        safe_depths = node_depths.long().clamp(0, self.config.max_depth)
        context_bank = torch.cat(
            [root_hidden[:, None, :], depth_hidden],
            dim=1,
        )
        context = torch.gather(
            context_bank,
            1,
            safe_depths[..., None].expand(-1, -1, context_bank.shape[-1]),
        )
        safe_ranks = torch.where(
            node_ranks >= 0,
            node_ranks.long().clamp(max=self.config.max_rank - 1),
            torch.full_like(node_ranks.long(), self.config.max_rank),
        )
        memory = (
            self.token_proj(self.token_norm(node_embeddings))
            + self.context_proj(self.context_norm(context))
            + self.depth_embedding(safe_depths)
            + self.rank_embedding(safe_ranks)
            + self.numeric_proj(node_numeric_features.to(node_embeddings.dtype))
        )
        return self.input_norm(memory).masked_fill(
            ~node_valid_mask[..., None],
            0,
        )

    def forward(
        self,
        *,
        root_hidden: torch.Tensor,
        depth_hidden: torch.Tensor,
        node_embeddings: torch.Tensor,
        node_depths: torch.Tensor,
        node_ranks: torch.Tensor,
        node_numeric_features: torch.Tensor,
        node_valid_mask: torch.Tensor,
    ) -> DFlashPathSelectorOutput:
        expected_nodes = 1 + self.config.max_depth * self.config.max_rank
        if node_embeddings.shape[1] != expected_nodes:
            raise ValueError(
                f"Expected root plus 14 candidate nodes ({expected_nodes}), "
                f"got {node_embeddings.shape[1]}."
            )
        batch_size = node_embeddings.shape[0]
        path_node_indices = self.fixed_path_node_indices[None].expand(
            batch_size,
            -1,
            -1,
        )
        path_valid_mask = torch.ones_like(
            path_node_indices,
            dtype=torch.bool,
        )
        memory_states = self._build_memory(
            root_hidden=root_hidden,
            depth_hidden=depth_hidden,
            node_embeddings=node_embeddings,
            node_depths=node_depths,
            node_ranks=node_ranks,
            node_numeric_features=node_numeric_features,
            node_valid_mask=node_valid_mask,
        )
        choice_logits = None
        switch_logits = None
        switch_deltas = None
        aligned_hidden = None
        teacher_logprob_predictions = None
        if self.config.selector_type == "path_attention":
            gathered_memory = gather_path_nodes(memory_states, path_node_indices)
            query_states = self.query_input_proj(gathered_memory)
            path_mask = self.fixed_path_mask[None].expand(
                batch_size,
                -1,
                -1,
                -1,
            )
            for block in self.blocks:
                query_states = block(
                    query_states,
                    memory_states,
                    path_mask,
                    path_valid_mask,
                    node_valid_mask,
                )
            path_states = self.output_norm(query_states).masked_fill(
                ~path_valid_mask[..., None],
                0,
            )
            conditional_logits = self.survival_head(path_states).squeeze(-1)
            conditional_logits = conditional_logits.masked_fill(
                ~path_valid_mask,
                -torch.inf,
            )
            conditional_survival = torch.sigmoid(conditional_logits)
            log_conditional = F.logsigmoid(conditional_logits.float())
            cumulative_survival = torch.exp(log_conditional.cumsum(dim=-1))
            cumulative_survival = cumulative_survival.masked_fill(
                ~path_valid_mask,
                0,
            )
        elif self.config.selector_type == "path_distilled_attention":
            candidate_states = gather_path_nodes(memory_states, path_node_indices)
            root_states = memory_states[:, None, :1].expand(
                -1, self.config.num_paths, -1, -1
            )
            path_states_with_root = self.path_input_proj(
                torch.cat([root_states, candidate_states], dim=2)
            )
            for block in self.path_blocks:
                path_states_with_root = block(
                    path_states_with_root, self.path_causal_mask
                )
            path_states = self.path_output_norm(path_states_with_root[:, :, 1:])
            conditional_logits = self.survival_head(path_states).squeeze(-1)
            conditional_survival = torch.sigmoid(conditional_logits)
            cumulative_survival = torch.exp(
                F.logsigmoid(conditional_logits.float()).cumsum(dim=-1)
            )
            # Distillation-only heads remain in the state dict for strict
            # checkpoint parity but are intentionally skipped during serving.
        else:
            candidates = memory_states[:, 1:].reshape(
                batch_size,
                self.config.max_depth,
                self.config.max_rank,
                self.config.hidden_size,
            )
            first, second = candidates.unbind(dim=2)
            root = memory_states[:, :1].expand(-1, self.config.max_depth, -1)
            pair_inputs = torch.cat(
                [first, second, first - second, root], dim=-1
            )
            gate, up = self.pair_gate_up_proj(
                self.pair_input_norm(pair_inputs)
            ).chunk(2, dim=-1)
            pair_states = self.pair_output_norm(
                self.pair_down_proj(F.silu(gate) * up)
            )
            choice_logits = self.choice_head(pair_states)
            choice_probabilities = torch.softmax(
                choice_logits.float(), dim=-1
            )[..., : self.config.max_rank]
            gather_indices = self.fixed_path_ranks[None].expand(
                batch_size, -1, -1
            )
            conditional_survival = torch.gather(
                choice_probabilities[:, None].expand(
                    -1, self.config.num_paths, -1, -1
                ),
                3,
                gather_indices[..., None],
            ).squeeze(-1)
            conditional_logits = torch.logit(
                conditional_survival.clamp(1e-6, 1.0 - 1e-6)
            )
            cumulative_survival = torch.cumprod(
                conditional_survival, dim=-1
            )
            path_states = pair_states
        expected_accept_length = cumulative_survival.sum(dim=-1)
        if self.config.selector_type == "direct_path_scorer":
            expanded_pairs = pair_states[:, None].expand(
                -1,
                self.config.num_paths,
                -1,
                -1,
            )
            rank_features = self.fixed_path_ranks[None, :, :, None].expand(
                batch_size,
                -1,
                -1,
                -1,
            ).to(pair_states.dtype)
            path_features = torch.cat(
                [
                    expanded_pairs,
                    rank_features,
                    conditional_survival[..., None].to(pair_states.dtype),
                    cumulative_survival[..., None].to(pair_states.dtype),
                ],
                dim=-1,
            ).flatten(2)
            path_hidden = F.silu(
                self.path_score_hidden(
                    self.path_score_input_norm(path_features)
                )
            )
            path_residual = self.path_score_output(path_hidden).squeeze(-1)
            expected_accept_length = expected_accept_length + path_residual.float()
        alternative_score, alternative_index = expected_accept_length[:, 1:].max(
            dim=-1
        )
        alternative_index = alternative_index + 1
        if self.config.selector_type == "gated_pair_scorer":
            gate_features = self._build_gate_features(
                pair_states,
                choice_logits,
                choice_probabilities,
                expected_accept_length,
                alternative_index,
            )
            gate_hidden = F.silu(
                self.switch_gate_hidden(
                    self.switch_gate_input_norm(gate_features)
                )
            )
            switch_logits = self.switch_gate_output(gate_hidden).squeeze(-1)
            if self.config.gate_predict_delta:
                switch_deltas = self.switch_delta_output(gate_hidden).squeeze(-1)
        if self.config.selector_type in {
            "direct_path_scorer",
            "path_distilled_attention",
        }:
            best_path_indices = expected_accept_length.argmax(dim=-1)
        elif switch_logits is None:
            should_switch = (
                alternative_score
                > expected_accept_length[:, 0] + self.config.switch_margin
            )
        else:
            decision_score = (
                switch_deltas if switch_deltas is not None else switch_logits
            )
            should_switch = (
                decision_score > self.config.switch_margin
            ) & (alternative_score > expected_accept_length[:, 0])
        if self.config.selector_type not in {
            "direct_path_scorer", "path_distilled_attention"
        }:
            best_path_indices = torch.where(
                should_switch,
                alternative_index,
                torch.zeros_like(alternative_index),
            )
        return DFlashPathSelectorOutput(
            conditional_logits=conditional_logits,
            conditional_survival=conditional_survival,
            cumulative_survival=cumulative_survival,
            expected_accept_length=expected_accept_length,
            best_path_indices=best_path_indices,
            path_states=path_states,
            choice_logits=choice_logits,
            switch_logits=switch_logits,
            switch_deltas=switch_deltas,
            aligned_hidden=aligned_hidden,
            teacher_logprob_predictions=teacher_logprob_predictions,
        )


def dflash_path_selector_loss(
    output: DFlashPathSelectorOutput,
    accepted_lengths: torch.Tensor,
    *,
    ranking_weight: float = 1.0,
) -> DFlashPathSelectorLoss:
    """Hazard NLL plus pairwise ranking over paths from the same prefix."""
    if accepted_lengths.shape != output.expected_accept_length.shape:
        raise ValueError(
            "accepted_lengths must have shape [B, P], matching selector scores."
        )
    max_depth = output.conditional_logits.shape[-1]
    accepted_lengths = accepted_lengths.long().clamp(0, max_depth)
    positions = torch.arange(
        1,
        max_depth + 1,
        device=accepted_lengths.device,
    )
    labels = positions <= accepted_lengths[..., None]
    active = positions <= torch.minimum(
        accepted_lengths + 1,
        torch.full_like(accepted_lengths, max_depth),
    )[..., None]
    hazard_terms = F.binary_cross_entropy_with_logits(
        output.conditional_logits.float(),
        labels.float(),
        reduction="none",
    )
    hazard_loss = hazard_terms[active].mean()

    length_diff = accepted_lengths[:, :, None] - accepted_lengths[:, None, :]
    ordered_pairs = length_diff > 0
    score_diff = (
        output.expected_accept_length[:, :, None]
        - output.expected_accept_length[:, None, :]
    )
    if ordered_pairs.any():
        ranking_loss = F.softplus(-score_diff[ordered_pairs]).mean()
    else:
        ranking_loss = hazard_loss.new_zeros(())
    loss = hazard_loss + ranking_weight * ranking_loss
    return DFlashPathSelectorLoss(
        loss=loss,
        hazard_loss=hazard_loss,
        ranking_loss=ranking_loss,
    )
