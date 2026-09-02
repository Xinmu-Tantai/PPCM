# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with TreeAttention."""

import ast
import logging
import os
from dataclasses import dataclass, field
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


def _parse_dflash_debug_int_list_env(name: str) -> list[int] | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            result.append(int(part))
        except ValueError:
            logger.warning("Ignoring invalid integer %r in %s=%r", part, name, value)
    return result


def _extract_dflash_layer_index(layer_name: str) -> int | None:
    marker = ".layers."
    if marker not in layer_name:
        return None
    suffix = layer_name.split(marker, 1)[1]
    layer_idx = suffix.split(".", 1)[0]
    try:
        return int(layer_idx)
    except ValueError:
        return None


def _should_probe_dflash_attention_layer(layer_name: str) -> bool:
    layer_indices = _parse_dflash_debug_int_list_env(
        "DFLASH_VERIFIER_PROBE_ATTN_LAYERS"
    )
    if layer_indices is None:
        layer_indices = _parse_dflash_debug_int_list_env(
            "DFLASH_VERIFIER_PROBE_KV_LAYERS"
        )
    if not layer_indices:
        return False
    layer_idx = _extract_dflash_layer_index(layer_name)
    return layer_idx in set(layer_indices)


class TreeAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "TREE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TreeAttentionImpl"]:
        return TreeAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["TreeAttentionMetadataBuilder"]:
        return TreeAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class TreeAttentionMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    tree_attn_bias: torch.Tensor | None = None
    ancestor_masks: torch.Tensor | None = None
    logical_kv_slots: torch.Tensor | None = None
    logical_kv_slot_lens: torch.Tensor | None = None
    logical_kv_starts: torch.Tensor | None = None
    # Depth-based absolute positions for each tree query token.
    # Shape: [num_decode_tokens].  When set, the attention kernel uses these
    # positions (instead of sequential query positions) for causal masking and
    # sliding-window bounds.  This is required for exact AR-parity: tree nodes
    # at depth d appear at sequential index S >= d in BFS order, so using S
    # for the sliding window start shifts the window by (S - d) relative to
    # what the equivalent AR pass would see.
    per_query_abs_pos: torch.Tensor | None = None
    # Debug-only attention-boundary records collected during DFlash verifier
    # forwards.  Parent and cached decode metadata share the same list.
    debug_attn_records: list[dict[str, object]] = field(default_factory=list)

    # Cached Prefill/decode metadata.
    _cached_prefill_metadata: "TreeAttentionMetadata | None" = None
    _cached_decode_metadata: "TreeAttentionMetadata | None" = None

    @property
    def prefill_metadata(self) -> "TreeAttentionMetadata | None":
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        q_start_loc = self.query_start_loc[self.num_decodes :]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[self.num_decodes :]
        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = TreeAttentionMetadata(
            num_actual_tokens=self.num_prefill_tokens,
            # Avoid GPU scalar sync during CUDA graph capture. The parent
            # metadata max is conservative and already available on CPU.
            max_query_len=self.max_query_len,
            query_start_loc=q_start_loc - q_start_loc[0],
            max_seq_len=self.max_seq_len,
            seq_lens=kv_seqlens,
            block_table=self.block_table[self.num_decodes :],
            slot_mapping=self.slot_mapping[self.num_decode_tokens :],
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> "TreeAttentionMetadata | None":
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata

        q_start_loc = self.query_start_loc[: self.num_decodes + 1]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[: self.num_decodes]
        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = TreeAttentionMetadata(
            num_actual_tokens=self.num_decode_tokens,
            # Avoid GPU scalar sync during CUDA graph capture. The parent
            # metadata max is conservative and already available on CPU.
            max_query_len=self.max_query_len,
            query_start_loc=q_start_loc,
            max_seq_len=self.max_seq_len,
            seq_lens=kv_seqlens,
            block_table=self.block_table[: self.num_decodes],
            slot_mapping=self.slot_mapping[: self.num_decode_tokens],
            tree_attn_bias=self.tree_attn_bias,
            ancestor_masks=self.ancestor_masks,
            logical_kv_slots=self.logical_kv_slots,
            logical_kv_slot_lens=self.logical_kv_slot_lens,
            logical_kv_starts=self.logical_kv_starts,
            per_query_abs_pos=self.per_query_abs_pos,
            debug_attn_records=self.debug_attn_records,
        )
        return self._cached_decode_metadata


class TreeAttentionMetadataBuilder(AttentionMetadataBuilder[TreeAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        spec_config = vllm_config.speculative_config
        spec_token_tree: str | None = None
        if spec := spec_config:
            spec_token_tree = spec.speculative_token_tree
        tree_choices: list[tuple[int, ...]] = (
            ast.literal_eval(spec_token_tree) if spec_token_tree is not None else [(0,)]
        )
        # Construct the tree attention bias.
        depth_counts = _get_depth_counts(tree_choices)
        self.tree_attn_bias = _prepare_tree_attn_bias(
            tree_choices,
            depth_counts,
            dtype=torch.float32,
            device=device,
        )

        self.reorder_batch_threshold = self.tree_attn_bias.shape[0]
        self._cudagraph_tree_attn_bias: torch.Tensor | None = None
        self._cudagraph_ancestor_masks: torch.Tensor | None = None
        self._cudagraph_logical_kv_slots: torch.Tensor | None = None
        self._cudagraph_logical_kv_slot_lens: torch.Tensor | None = None
        self._cudagraph_logical_kv_starts: torch.Tensor | None = None
        self._max_cudagraph_tree_query_len = self._init_max_cudagraph_tree_query_len()
        self._max_cudagraph_tree_bias_len = self._init_max_cudagraph_tree_bias_len()
        self._max_cudagraph_batch_size = vllm_config.scheduler_config.max_num_seqs
        self._max_cudagraph_logical_kv_slots = int(
            getattr(vllm_config.model_config, "max_model_len", 0)
            or self._max_cudagraph_tree_query_len
        )
        self._dflash_tree_debug_records: list[dict[str, object]] = []

    def get_dflash_tree_debug_records(self) -> list[dict[str, object]]:
        return list(self._dflash_tree_debug_records)

    def clear_dflash_tree_debug_records(self) -> None:
        self._dflash_tree_debug_records.clear()

    def _dflash_tree_debug_enabled(self) -> bool:
        return logger.isEnabledFor(logging.DEBUG)

    def _clear_dflash_tree_debug_context(self) -> None:
        if hasattr(self, "_dflash_tree_debug_context"):
            delattr(self, "_dflash_tree_debug_context")

    def _append_dflash_tree_debug_record(
        self,
        *,
        build_method: str,
        common_attn_metadata: CommonAttentionMetadata,
        output_metadata: TreeAttentionMetadata,
        tree_attn_bias: torch.Tensor | None,
        ancestor_masks: torch.Tensor | None,
        logical_kv_slots: list[torch.Tensor | None] | torch.Tensor | None = None,
        debug_context: dict[str, object],
    ) -> None:
        if not self._dflash_tree_debug_enabled():
            return

        decode_meta = output_metadata.decode_metadata
        logical_kv_indirection_shape = (
            list(output_metadata.logical_kv_slots.shape)
            if output_metadata.logical_kv_slots is not None
            else None
        )
        logical_kv_indirection_lens = (
            output_metadata.logical_kv_slot_lens.detach().cpu()
            if output_metadata.logical_kv_slot_lens is not None
            else None
        )
        logical_kv_starts = (
            output_metadata.logical_kv_starts.detach().cpu()
            if output_metadata.logical_kv_starts is not None
            else None
        )
        if torch.is_tensor(output_metadata.logical_kv_slot_lens):
            logical_kv_slot_lens = output_metadata.logical_kv_slot_lens.detach().cpu().tolist()
        elif isinstance(logical_kv_slots, list):
            logical_kv_slot_lens = [
                int(slots.numel()) if slots is not None else 0
                for slots in logical_kv_slots
            ]
        else:
            logical_kv_slot_lens = None
        # TODO(dflash-logical-kv-cleanup): remove these verbose diagnostics once
        # the logical KV layout is stable.  They compare the logical slot
        # indirection against the canonical block-table slots that attention
        # would read without logical remapping.
        logical_kv_slot_debug: list[dict[str, object]] | None = None
        if (
            output_metadata.logical_kv_slots is not None
            and output_metadata.logical_kv_slot_lens is not None
            and output_metadata.logical_kv_starts is not None
        ):
            slots_cpu = output_metadata.logical_kv_slots.detach().cpu()
            lens_cpu = output_metadata.logical_kv_slot_lens.detach().cpu()
            starts_cpu = output_metadata.logical_kv_starts.detach().cpu()
            block_table_cpu = output_metadata.block_table.detach().cpu()
            seq_lens_cpu = output_metadata.seq_lens.detach().cpu()
            q_lens_cpu = torch.diff(output_metadata.query_start_loc.detach().cpu())
            logical_kv_slot_debug = []
            for req_idx in range(slots_cpu.shape[0]):
                slot_len = int(lens_cpu[req_idx].item())
                logical_start = int(starts_cpu[req_idx].item())
                query_len = (
                    int(q_lens_cpu[req_idx].item())
                    if req_idx < q_lens_cpu.numel()
                    else 0
                )
                seq_len = int(seq_lens_cpu[req_idx].item())
                context_len = seq_len - query_len
                if slot_len <= 0:
                    logical_kv_slot_debug.append(
                        {
                            "req_idx": req_idx,
                            "logical_start": logical_start,
                            "logical_len": 0,
                            "context_len": context_len,
                            "query_len": query_len,
                        }
                    )
                    continue

                logical_slots = slots_cpu[req_idx, :slot_len].to(torch.int64)
                logical_positions = torch.arange(
                    logical_start,
                    logical_start + slot_len,
                    dtype=torch.int64,
                )
                block_indices = torch.div(
                    logical_positions, self.block_size, rounding_mode="floor"
                )
                in_block_table = block_indices < block_table_cpu.shape[1]
                canonical_slots = torch.full_like(logical_slots, -1)
                if bool(in_block_table.any().item()):
                    valid_positions = logical_positions[in_block_table]
                    valid_block_indices = block_indices[in_block_table]
                    valid_blocks = block_table_cpu[
                        req_idx, valid_block_indices
                    ].to(torch.int64)
                    canonical_slots[in_block_table] = (
                        valid_blocks * self.block_size
                        + valid_positions % self.block_size
                    )

                mismatch = logical_slots != canonical_slots
                sample_count = min(8, slot_len)
                mismatch_indices = torch.nonzero(mismatch, as_tuple=False).flatten()
                mismatch_sample_count = min(8, int(mismatch_indices.numel()))
                mismatch_sample = [
                    {
                        "logical_pos": int(logical_positions[idx].item()),
                        "logical_slot": int(logical_slots[idx].item()),
                        "canonical_slot": int(canonical_slots[idx].item()),
                    }
                    for idx in mismatch_indices[:mismatch_sample_count]
                ]

                logical_kv_slot_debug.append(
                    {
                        "req_idx": req_idx,
                        "logical_start": logical_start,
                        "logical_end": logical_start + slot_len,
                        "logical_len": slot_len,
                        "context_len": context_len,
                        "query_len": query_len,
                        "mapped_past_context": logical_start + slot_len > context_len,
                        "logical_slot_min": int(logical_slots.min().item()),
                        "logical_slot_max": int(logical_slots.max().item()),
                        "logical_slot_head": logical_slots[:sample_count].tolist(),
                        "logical_slot_tail": logical_slots[-sample_count:].tolist(),
                        "canonical_slot_head": canonical_slots[
                            :sample_count
                        ].tolist(),
                        "canonical_slot_tail": canonical_slots[
                            -sample_count:
                        ].tolist(),
                        "canonical_mismatch_count": int(mismatch.sum().item()),
                        "duplicate_logical_slot_count": (
                            slot_len - int(torch.unique(logical_slots).numel())
                        ),
                        "invalid_canonical_count": int(
                            (~in_block_table).sum().item()
                        ),
                        "mismatch_sample": mismatch_sample,
                    }
                )
        self._dflash_tree_debug_records.append(
            {
                "build_method": build_method,
                "caller_role": str(debug_context.get("caller_role", "unknown")),
                "builder_owner": str(debug_context.get("builder_owner", "unknown")),
                "tree_propose_step": int(debug_context.get("tree_propose_step", -1)),
                "draft_index": (
                    int(debug_context.get("draft_index", -1))
                    if debug_context.get("draft_index") is not None
                    else None
                ),
                "for_cudagraph_capture": bool(
                    debug_context.get("for_cudagraph_capture", False)
                ),
                "input_num_actual_tokens": int(common_attn_metadata.num_actual_tokens),
                "input_num_reqs": int(common_attn_metadata.num_reqs),
                "input_max_query_len": int(common_attn_metadata.max_query_len),
                "input_max_seq_len": int(common_attn_metadata.max_seq_len),
                "input_query_start_loc": common_attn_metadata.query_start_loc.detach()
                .cpu(),
                "input_seq_lens": common_attn_metadata.seq_lens.detach().cpu(),
                "input_slot_mapping": common_attn_metadata.slot_mapping.detach().cpu(),
                "tree_attn_bias_shape": (
                    list(tree_attn_bias.shape) if tree_attn_bias is not None else None
                ),
                "ancestor_masks_shape": (
                    list(ancestor_masks.shape)
                    if ancestor_masks is not None
                    else None
                ),
                "logical_kv_layout": debug_context.get("logical_kv_layout"),
                "logical_kv_slot_lens": logical_kv_slot_lens,
                "logical_kv_indirection_shape": logical_kv_indirection_shape,
                "logical_kv_indirection_lens": logical_kv_indirection_lens,
                "logical_kv_starts": logical_kv_starts,
                "logical_kv_slot_debug": logical_kv_slot_debug,
                "logical_kv_num_mapped_reqs": (
                    sum(1 for n in logical_kv_slot_lens if n > 0)
                    if logical_kv_slot_lens is not None
                    else None
                ),
                "output_max_query_len": int(output_metadata.max_query_len),
                "output_max_seq_len": int(output_metadata.max_seq_len),
                "output_query_start_loc": output_metadata.query_start_loc.detach().cpu(),
                "output_seq_lens": output_metadata.seq_lens.detach().cpu(),
                "decode_max_query_len": (
                    int(decode_meta.max_query_len) if decode_meta is not None else None
                ),
                "decode_max_seq_len": (
                    int(decode_meta.max_seq_len) if decode_meta is not None else None
                ),
                "decode_query_start_loc": (
                    decode_meta.query_start_loc.detach().cpu()
                    if decode_meta is not None
                    else None
                ),
                "decode_seq_lens": (
                    decode_meta.seq_lens.detach().cpu()
                    if decode_meta is not None
                    else None
                ),
            }
        )

    def _init_max_cudagraph_tree_query_len(self) -> int:
        capture_hints = self.vllm_config.compilation_config.cudagraph_capture_sizes
        spec_config = self.vllm_config.speculative_config
        dflash_tree_budget = 0
        if spec_config is not None and getattr(spec_config, "method", None) == "dflash":
            dflash_tree_budget = int(getattr(spec_config, "dflash_tree_budget", 0))
            capture_hints = getattr(spec_config, "cudagraph_tree_capture_sizes", None)
        static_tree_len = max(self.tree_attn_bias.shape[0], dflash_tree_budget)
        if capture_hints:
            return max(max(capture_hints), static_tree_len)
        return static_tree_len

    def _init_max_cudagraph_tree_bias_len(self) -> int:
        """Capacity for Triton's block-diagonal batched tree bias.

        ``tree_attn_bias`` is shaped by total query tokens across the batch,
        unlike Optimus ancestor masks which use per-request tree size.
        """
        capture_hints = self.vllm_config.compilation_config.cudagraph_capture_sizes
        spec_config = self.vllm_config.speculative_config
        max_tree_budget = 0
        if spec_config is not None and getattr(spec_config, "method", None) == "dflash":
            max_tree_budget = int(getattr(spec_config, "dflash_tree_budget", 0))
        max_batch_tree_len = (
            max_tree_budget * self.vllm_config.scheduler_config.max_num_seqs
            if max_tree_budget > 0
            else 0
        )
        static_tree_len = max(self.tree_attn_bias.shape[0], max_tree_budget)
        hinted_tree_len = max(capture_hints) if capture_hints else 0
        return max(static_tree_len, hinted_tree_len, max_batch_tree_len)

    def _copy_tree_attn_bias_for_cudagraph(
        self, tree_attn_bias: torch.Tensor
    ) -> torch.Tensor:
        query_len = tree_attn_bias.shape[0]
        if query_len > self._max_cudagraph_tree_bias_len:
            raise ValueError(
                "Tree attention bias exceeds cudagraph capture capacity: "
                f"{query_len} > {self._max_cudagraph_tree_bias_len}"
            )
        if self._cudagraph_tree_attn_bias is None:
            self._cudagraph_tree_attn_bias = torch.empty(
                (
                    self._max_cudagraph_tree_bias_len,
                    self._max_cudagraph_tree_bias_len,
                ),
                dtype=tree_attn_bias.dtype,
                device=self.device,
            )
        self._cudagraph_tree_attn_bias[:query_len, :query_len].copy_(tree_attn_bias)
        return self._cudagraph_tree_attn_bias[:query_len, :query_len]

    def _copy_ancestor_masks_for_cudagraph(
        self, ancestor_masks: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = ancestor_masks.shape
        max_B = self._max_cudagraph_batch_size
        max_N = self._max_cudagraph_tree_query_len
        if self._cudagraph_ancestor_masks is None:
            self._cudagraph_ancestor_masks = torch.zeros(
                (max_B, max_N, max_N), dtype=torch.int32, device=self.device
            )
        self._cudagraph_ancestor_masks[:B, :N, :N].copy_(ancestor_masks)
        return self._cudagraph_ancestor_masks[:B, :N, :N]

    def _copy_logical_kv_for_cudagraph(
        self,
        logical_kv_slots: list[torch.Tensor | None] | torch.Tensor,
        logical_kv_slot_lens: torch.Tensor | None,
        logical_kv_starts: list[int] | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(logical_kv_slots):
            assert logical_kv_slot_lens is not None
            assert torch.is_tensor(logical_kv_starts)
            num_reqs = logical_kv_slots.shape[0]
            max_slots = logical_kv_slots.shape[1]
        else:
            num_reqs = len(logical_kv_slots)
            lens_list = [
                int(slots.numel()) if slots is not None else 0
                for slots in logical_kv_slots
            ]
            max_slots = max(lens_list, default=0)

        max_B = self._max_cudagraph_batch_size
        capacity = self._max_cudagraph_logical_kv_slots
        if num_reqs > max_B:
            raise ValueError(
                "Logical KV cudagraph batch exceeds capacity: "
                f"{num_reqs} > {max_B}"
            )
        if max_slots > capacity:
            raise ValueError(
                "Logical KV slots exceed cudagraph capture capacity: "
                f"{max_slots} > {capacity}"
            )
        if (
            torch.is_tensor(logical_kv_slots)
            and max_slots == capacity
            and logical_kv_slots.dtype == torch.int64
            and logical_kv_slots.device == self.device
            and logical_kv_slots.is_contiguous()
            and logical_kv_slot_lens is not None
            and logical_kv_slot_lens.dtype == torch.int32
            and logical_kv_slot_lens.device == self.device
            and torch.is_tensor(logical_kv_starts)
            and logical_kv_starts.dtype == torch.int32
            and logical_kv_starts.device == self.device
        ):
            return (
                logical_kv_slots[:num_reqs],
                logical_kv_slot_lens[:num_reqs],
                logical_kv_starts[:num_reqs],
            )

        if self._cudagraph_logical_kv_slots is None:
            self._cudagraph_logical_kv_slots = torch.empty(
                (max_B, capacity), dtype=torch.int64, device=self.device
            )
            self._cudagraph_logical_kv_slot_lens = torch.empty(
                (max_B,), dtype=torch.int32, device=self.device
            )
            self._cudagraph_logical_kv_starts = torch.empty(
                (max_B,), dtype=torch.int32, device=self.device
            )

        assert self._cudagraph_logical_kv_slots is not None
        assert self._cudagraph_logical_kv_slot_lens is not None
        assert self._cudagraph_logical_kv_starts is not None
        self._cudagraph_logical_kv_slot_lens[:num_reqs].zero_()
        self._cudagraph_logical_kv_starts[:num_reqs].zero_()

        if torch.is_tensor(logical_kv_slots):
            self._cudagraph_logical_kv_slots[
                :num_reqs, :max_slots
            ].copy_(logical_kv_slots[:, :max_slots])
            self._cudagraph_logical_kv_slot_lens[
                :num_reqs
            ].copy_(logical_kv_slot_lens[:num_reqs].to(torch.int32))
            assert torch.is_tensor(logical_kv_starts)
            self._cudagraph_logical_kv_starts[:num_reqs].copy_(
                logical_kv_starts[:num_reqs].to(torch.int32)
            )
        else:
            starts_list = (
                logical_kv_starts
                if isinstance(logical_kv_starts, list)
                else [0] * num_reqs
            )
            for req_idx, slots in enumerate(logical_kv_slots):
                slot_len = int(slots.numel()) if slots is not None else 0
                self._cudagraph_logical_kv_slot_lens[req_idx] = slot_len
                self._cudagraph_logical_kv_starts[req_idx] = int(
                    starts_list[req_idx]
                )
                if slot_len > 0:
                    assert slots is not None
                    self._cudagraph_logical_kv_slots[
                        req_idx, :slot_len
                    ].copy_(
                        slots.to(
                            device=self.device,
                            dtype=torch.int64,
                            non_blocking=True,
                        )
                    )

        return (
            self._cudagraph_logical_kv_slots[:num_reqs],
            self._cudagraph_logical_kv_slot_lens[:num_reqs],
            self._cudagraph_logical_kv_starts[:num_reqs],
        )

    def build_for_dflash_tree(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        tree_attn_bias: torch.Tensor | None,
        *,
        for_cudagraph_capture: bool = False,
        ancestor_masks: torch.Tensor | None = None,
        logical_kv_slots: list[torch.Tensor | None] | torch.Tensor | None = None,
        logical_kv_slot_lens: torch.Tensor | None = None,
        logical_kv_starts: list[int] | torch.Tensor | None = None,
        per_query_abs_pos: torch.Tensor | None = None,
    ) -> TreeAttentionMetadata:
        debug_context = getattr(self, "_dflash_tree_debug_context", {}) or {}
        if tree_attn_bias is not None and for_cudagraph_capture:
            tree_attn_bias = self._copy_tree_attn_bias_for_cudagraph(tree_attn_bias)
        if ancestor_masks is not None and for_cudagraph_capture:
            ancestor_masks = self._copy_ancestor_masks_for_cudagraph(
                ancestor_masks
            )

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_reqs = common_attn_metadata.num_reqs
        logical_kv_slots_t: torch.Tensor | None = None
        logical_kv_slot_lens_t: torch.Tensor | None = None
        logical_kv_starts_t: torch.Tensor | None = None
        if logical_kv_slots is not None:
            if for_cudagraph_capture:
                (
                    logical_kv_slots_t,
                    logical_kv_slot_lens_t,
                    logical_kv_starts_t,
                ) = self._copy_logical_kv_for_cudagraph(
                    logical_kv_slots,
                    logical_kv_slot_lens,
                    logical_kv_starts,
                )
            elif torch.is_tensor(logical_kv_slots):
                assert logical_kv_slot_lens is not None
                assert torch.is_tensor(logical_kv_starts)
                logical_kv_slots_t = logical_kv_slots
                logical_kv_slot_lens_t = logical_kv_slot_lens
                logical_kv_starts_t = logical_kv_starts
            else:
                logical_kv_slot_lens_list = [
                    int(slots.numel()) if slots is not None else 0
                    for slots in logical_kv_slots
                ]
                max_logical_kv_slots = max(logical_kv_slot_lens_list, default=0)
                if max_logical_kv_slots > 0:
                    logical_kv_slots_t = torch.full(
                        (len(logical_kv_slots), max_logical_kv_slots),
                        -1,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    for req_idx, slots in enumerate(logical_kv_slots):
                        if slots is None or slots.numel() == 0:
                            continue
                        logical_kv_slots_t[req_idx, : slots.numel()] = slots.to(
                            device=self.device, dtype=torch.int64, non_blocking=True
                        )
                    logical_kv_slot_lens_t = torch.tensor(
                        logical_kv_slot_lens_list,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    logical_kv_starts_t = torch.tensor(
                        logical_kv_starts
                        if logical_kv_starts is not None
                        else [0] * len(logical_kv_slots),
                        dtype=torch.int32,
                        device=self.device,
                    )

        debug_attn_records: list[dict[str, object]] = []
        decode_meta = TreeAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            tree_attn_bias=tree_attn_bias,
            ancestor_masks=ancestor_masks,
            logical_kv_slots=logical_kv_slots_t,
            logical_kv_slot_lens=logical_kv_slot_lens_t,
            logical_kv_starts=logical_kv_starts_t,
            per_query_abs_pos=per_query_abs_pos,
            debug_attn_records=debug_attn_records,
        )
        meta = TreeAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_prefill_tokens=0,
            num_decode_tokens=num_actual_tokens,
            num_prefills=0,
            num_decodes=num_reqs,
            tree_attn_bias=tree_attn_bias,
            ancestor_masks=ancestor_masks,
            logical_kv_slots=logical_kv_slots_t,
            logical_kv_slot_lens=logical_kv_slot_lens_t,
            logical_kv_starts=logical_kv_starts_t,
            per_query_abs_pos=per_query_abs_pos,
            debug_attn_records=debug_attn_records,
            _cached_decode_metadata=decode_meta,
        )
        self._append_dflash_tree_debug_record(
            build_method="build_for_dflash_tree",
            common_attn_metadata=common_attn_metadata,
            output_metadata=meta,
            tree_attn_bias=tree_attn_bias,
            ancestor_masks=ancestor_masks,
            logical_kv_slots=logical_kv_slots,
            debug_context={
                **debug_context,
                "for_cudagraph_capture": bool(
                    debug_context.get("for_cudagraph_capture", for_cudagraph_capture)
                ),
            },
        )
        self._clear_dflash_tree_debug_context()
        return meta

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TreeAttentionMetadata:
        decode_threshold = self.tree_attn_bias.shape[0]
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=decode_threshold
            )
        )

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        q_start_loc = common_attn_metadata.query_start_loc
        max_query_len = common_attn_metadata.max_query_len
        kv_seqlens = common_attn_metadata.seq_lens
        max_seq_len = common_attn_metadata.max_seq_len
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        return TreeAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            max_query_len=max_query_len,
            query_start_loc=q_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=kv_seqlens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            tree_attn_bias=self.tree_attn_bias,
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> TreeAttentionMetadata:
        debug_context = getattr(self, "_dflash_tree_debug_context", {}) or {}
        # Cache the original tree attention bias.
        orig_tree_attn_bias = self.tree_attn_bias

        if draft_index == 0:
            # Use prefill for drafting at the root level.
            self.tree_attn_bias = torch.empty(0)
        else:
            # Slice the tree attention bias for drafting. Exclude
            # the root level.
            start, end = 1, 1 + common_attn_metadata.max_query_len
            self.tree_attn_bias = self.tree_attn_bias[start:end, start:end].contiguous()

        # Build attention bias.
        attn_metadata = self.build(0, common_attn_metadata, fast_build=True)

        # Reset the tree attention bias to the original value.
        self.tree_attn_bias = orig_tree_attn_bias
        self._append_dflash_tree_debug_record(
            build_method="build_for_drafting",
            common_attn_metadata=common_attn_metadata,
            output_metadata=attn_metadata,
            tree_attn_bias=self.tree_attn_bias,
            ancestor_masks=None,
            debug_context={
                **debug_context,
                "draft_index": int(draft_index),
            },
        )
        self._clear_dflash_tree_debug_context()
        return attn_metadata


def _get_depth_counts(sorted_tree_choices: list[tuple[int, ...]]) -> list[int]:
    # Count the number of choices at each depth of the tree.
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    return depth_counts


def _prepare_tree_attn_bias(
    sorted_tree_choices: list[tuple[int, ...]],
    depth_counts: list[int],
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> torch.Tensor:
    # +1 comes from the additional root node.
    tree_len = len(sorted_tree_choices) + 1
    tree_attn_mask = torch.full(
        (tree_len, tree_len), -torch.inf, device=device, dtype=dtype
    )

    # Set diagonal to all zeros. Each token should
    # attend to itself.
    mask_val = 0
    for i in range(tree_len):
        tree_attn_mask[i, i] = mask_val

    # Set root to all zeros. All tokens attend to it.
    tree_attn_mask[:, 0] = mask_val

    # Set all ancestors to zeros.
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            # Retrieve ancestor position.
            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(
                    sorted_tree_choices.index(cur_tree_choice[: c + 1]) + 1
                )
            tree_attn_mask[j + start + 1, ancestor_idx] = mask_val
        start += depth_counts[i]
    return tree_attn_mask


class TreeAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if logits_soft_cap is None:
            # Setting logits_soft_cap to 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TreeAttentionImpl."
            )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        key_cache, value_cache = kv_cache.unbind(0)

        # Reshape the input keys and values and store them in the cache.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def _maybe_record_dflash_verifier_attention_probe(
        self,
        *,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        decode_meta: TreeAttentionMetadata,
        output: torch.Tensor,
        kernel: str,
    ) -> None:
        layer_name = str(getattr(layer, "layer_name", ""))
        if not _should_probe_dflash_attention_layer(layer_name):
            return
        node_indices = _parse_dflash_debug_int_list_env(
            "DFLASH_VERIFIER_PROBE_NODES"
        )
        if node_indices is None:
            node_indices = [0, 6]
        node_indices = sorted({idx for idx in node_indices if idx >= 0})
        if not node_indices:
            return
        if output.is_cuda and torch.cuda.is_current_stream_capturing():
            return

        q_start_loc = decode_meta.query_start_loc.detach().to("cpu")
        flat_indices: list[int] = []
        node_records: list[dict[str, object]] = []
        for req_idx in range(max(0, int(q_start_loc.numel()) - 1)):
            req_start = int(q_start_loc[req_idx].item())
            req_end = int(q_start_loc[req_idx + 1].item())
            qlen = req_end - req_start
            for node_idx in node_indices:
                if node_idx >= qlen:
                    continue
                flat_idx = req_start + node_idx
                flat_indices.append(flat_idx)
                node_records.append(
                    {
                        "req_idx": req_idx,
                        "node_index": node_idx,
                        "flat_index": flat_idx,
                    }
                )
        if not flat_indices:
            return

        flat_idx_t = torch.tensor(
            flat_indices, dtype=torch.long, device=output.device
        )
        selected = output.index_select(0, flat_idx_t).detach().float()
        flat_selected = selected.flatten(start_dim=1)
        norms = flat_selected.norm(dim=1).cpu().tolist()
        mean_abs = flat_selected.abs().mean(dim=1).cpu().tolist()
        max_abs = flat_selected.abs().amax(dim=1).cpu().tolist()

        reference_records = self._compute_dflash_attention_reference_probe(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            decode_meta=decode_meta,
            output=output,
            node_records=node_records,
        )
        for node_record, norm, mean_abs_i, max_abs_i, ref_record in zip(
            node_records, norms, mean_abs, max_abs, reference_records, strict=True
        ):
            node_record["output_norm"] = float(norm)
            node_record["output_mean_abs"] = float(mean_abs_i)
            node_record["output_max_abs"] = float(max_abs_i)
            node_record.update(ref_record)

        decode_meta.debug_attn_records.append(
            {
                "layer_name": layer_name,
                "layer_index": _extract_dflash_layer_index(layer_name),
                "kernel": kernel,
                "output_shape": [int(dim) for dim in output.shape],
                "nodes": node_records,
            }
        )

    def _compute_dflash_attention_reference_probe(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        decode_meta: TreeAttentionMetadata,
        output: torch.Tensor,
        node_records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if (
            decode_meta.tree_attn_bias is None
            or decode_meta.block_table is None
            or self.alibi_slopes is not None
        ):
            return [{"reference_skipped": True} for _ in node_records]

        block_size = int(key_cache.shape[1])
        kv_head_indices = torch.div(
            torch.arange(self.num_heads, device=query.device),
            max(1, self.num_heads // self.num_kv_heads),
            rounding_mode="floor",
        )
        q_start_loc = decode_meta.query_start_loc.detach().to("cpu")
        seq_lens = decode_meta.seq_lens.detach().to("cpu")
        per_query_abs_pos = decode_meta.per_query_abs_pos
        tree_attn_bias = decode_meta.tree_attn_bias
        results: list[dict[str, object]] = []

        def _slot_sample(tensor: torch.Tensor, limit: int = 8) -> list[int]:
            if tensor.numel() == 0:
                return []
            return [int(x) for x in tensor[: min(limit, tensor.numel())].detach().cpu()]

        def _float_sample(tensor: torch.Tensor, limit: int = 8) -> list[float]:
            if tensor.numel() == 0:
                return []
            return [
                float(x)
                for x in tensor.flatten()[: min(limit, tensor.numel())].detach().cpu()
            ]

        for node_record in node_records:
            req_idx = int(node_record["req_idx"])
            node_idx = int(node_record["node_index"])
            flat_idx = int(node_record["flat_index"])
            req_start = int(q_start_loc[req_idx].item())
            req_end = int(q_start_loc[req_idx + 1].item())
            qlen = req_end - req_start
            seq_len = int(seq_lens[req_idx].item())
            context_len = max(0, seq_len - qlen)

            if per_query_abs_pos is not None:
                abs_pos = int(per_query_abs_pos[flat_idx].detach().cpu().item())
            else:
                abs_pos = context_len + node_idx
            context_start = 0
            if self.sliding_window is not None:
                window_left = (
                    self.sliding_window[0]
                    if isinstance(self.sliding_window, (tuple, list))
                    else self.sliding_window
                )
                if window_left is not None and int(window_left) >= 0:
                    context_start = max(0, abs_pos - int(window_left))
            context_end = min(context_len, abs_pos + 1)
            context_positions = torch.arange(
                context_start,
                context_end,
                dtype=torch.int64,
                device=query.device,
            )
            block_table_req = decode_meta.block_table[req_idx].to(
                device=query.device,
                dtype=torch.int64,
            )
            if context_positions.numel() > 0:
                context_blocks = block_table_req[
                    torch.div(context_positions, block_size, rounding_mode="floor")
                ]
                context_slots = (
                    context_blocks * block_size + (context_positions % block_size)
                )
            else:
                context_slots = torch.empty(
                    0, dtype=torch.int64, device=query.device
                )

            bias_row = tree_attn_bias[node_idx, :qlen]
            allowed_query_cols = torch.nonzero(
                bias_row == 0, as_tuple=False
            ).flatten()
            sequential_query_cols = torch.arange(
                0, min(node_idx + 1, qlen), dtype=torch.long, device=query.device
            )
            all_query_cols = torch.arange(0, qlen, dtype=torch.long, device=query.device)
            tree_slots = decode_meta.slot_mapping[
                req_start + allowed_query_cols.to(decode_meta.slot_mapping.device)
            ].to(device=query.device, dtype=torch.int64)
            sequential_tree_slots = decode_meta.slot_mapping[
                req_start + sequential_query_cols.to(decode_meta.slot_mapping.device)
            ].to(device=query.device, dtype=torch.int64)
            all_tree_slots = decode_meta.slot_mapping[
                req_start + all_query_cols.to(decode_meta.slot_mapping.device)
            ].to(device=query.device, dtype=torch.int64)
            canonical_tree_positions = context_len + allowed_query_cols.to(query.device)
            canonical_tree_blocks = block_table_req[
                torch.div(canonical_tree_positions, block_size, rounding_mode="floor")
            ]
            canonical_tree_slots = (
                canonical_tree_blocks * block_size
                + (canonical_tree_positions % block_size)
            )
            if decode_meta.logical_kv_slots is not None:
                logical_start = int(
                    decode_meta.logical_kv_starts[req_idx].detach().cpu().item()
                )
                logical_len = int(
                    decode_meta.logical_kv_slot_lens[req_idx].detach().cpu().item()
                )
                logical_end = logical_start + logical_len
                logical_tree_slots = canonical_tree_slots.clone()
                use_logical = (
                    (canonical_tree_positions >= logical_start)
                    & (canonical_tree_positions < logical_end)
                )
                if bool(use_logical.any().item()):
                    logical_tree_slots[use_logical] = decode_meta.logical_kv_slots[
                        req_idx,
                        (canonical_tree_positions[use_logical] - logical_start).to(
                            decode_meta.logical_kv_slots.device
                        ),
                    ].to(device=query.device, dtype=torch.int64)
            else:
                logical_start = None
                logical_len = None
                logical_tree_slots = canonical_tree_slots

            slots = torch.cat([context_slots, tree_slots])
            if slots.numel() == 0:
                results.append({"reference_skipped": True})
                continue

            q = query[flat_idx].detach().float().view(self.num_heads, self.head_size)
            kernel_out = output[flat_idx].detach().float().view(
                self.num_heads, self.head_size
            )
            blocks = torch.div(slots, block_size, rounding_mode="floor")
            offsets = slots % block_size
            main_key = key_cache[blocks, offsets].detach().float()
            main_value = value_cache[blocks, offsets].detach().float()
            main_k_for_heads = main_key[:, kv_head_indices, :]
            main_v_for_heads = main_value[:, kv_head_indices, :]
            unscaled_scores = torch.einsum("hd,khd->hk", q, main_k_for_heads)
            tree_score_indices = torch.arange(
                context_slots.numel(), slots.numel(), device=query.device
            )
            block_m = 16 if self.num_heads // self.num_kv_heads <= 16 else 2 ** (
                self.num_heads // self.num_kv_heads - 1
            ).bit_length()
            block_q = max(1, block_m // max(1, self.num_heads // self.num_kv_heads))
            q_block_local_idx = node_idx // block_q
            q_block_lo = q_block_local_idx * block_q
            q_block_hi = min(
                q_block_lo + (block_m - 1) // max(1, self.num_heads // self.num_kv_heads),
                qlen - 1,
            )

            def _compute_ref_for_slots(
                ref_slots: torch.Tensor,
                ref_query: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                q_for_ref = q if ref_query is None else ref_query
                blocks = torch.div(ref_slots, block_size, rounding_mode="floor")
                offsets = ref_slots % block_size
                key = key_cache[blocks, offsets].detach().float()
                value = value_cache[blocks, offsets].detach().float()
                k_for_heads = key[:, kv_head_indices, :]
                v_for_heads = value[:, kv_head_indices, :]
                scores = torch.einsum("hd,khd->hk", q_for_ref, k_for_heads) * self.scale
                if self.logits_soft_cap > 0:
                    scores = torch.tanh(scores / self.logits_soft_cap)
                    scores = scores * self.logits_soft_cap
                probs = torch.softmax(scores, dim=-1)
                ref_out = torch.einsum("hk,khd->hd", probs, v_for_heads)
                return ref_out, probs, scores

            ref, probs, scores = _compute_ref_for_slots(slots)
            diff = (kernel_out - ref).abs()
            sequential_ref, _, _ = _compute_ref_for_slots(
                torch.cat([context_slots, sequential_tree_slots])
            )
            all_tree_ref, _, _ = _compute_ref_for_slots(
                torch.cat([context_slots, all_tree_slots])
            )
            canonical_ref, _, _ = _compute_ref_for_slots(
                torch.cat([context_slots, canonical_tree_slots])
            )
            logical_ref, _, _ = _compute_ref_for_slots(
                torch.cat([context_slots, logical_tree_slots])
            )
            next_query_diff = None
            if node_idx + 1 < qlen:
                next_q = query[flat_idx + 1].detach().float().view(
                    self.num_heads, self.head_size
                )
                next_query_ref, _, _ = _compute_ref_for_slots(slots, next_q)
                next_query_diff = float(
                    (kernel_out - next_query_ref).abs().max().detach().cpu().item()
                )
            head_diffs = diff.flatten(start_dim=1).amax(dim=1)
            score_ranges = torch.stack(
                [scores.amin(dim=1), scores.amax(dim=1)], dim=1
            )
            unscaled_score_ranges = torch.stack(
                [unscaled_scores.amin(dim=1), unscaled_scores.amax(dim=1)], dim=1
            )
            prob_ranges = torch.stack(
                [probs.amin(dim=1), probs.amax(dim=1)], dim=1
            )
            top_prob_values, top_prob_indices = torch.topk(
                probs[0], k=min(8, int(probs.shape[1]))
            )
            results.append(
                {
                    "reference_skipped": False,
                    "reference_key_count": int(slots.numel()),
                    "reference_context_key_count": int(context_slots.numel()),
                    "reference_tree_key_count": int(tree_slots.numel()),
                    "reference_allowed_query_cols": _slot_sample(allowed_query_cols),
                    "reference_sequential_query_cols": _slot_sample(
                        sequential_query_cols
                    ),
                    "reference_context_slot_head": _slot_sample(context_slots),
                    "reference_context_slot_tail": _slot_sample(
                        context_slots[-min(8, context_slots.numel()) :]
                    ),
                    "reference_tree_slots": _slot_sample(tree_slots),
                    "reference_canonical_tree_slots": _slot_sample(
                        canonical_tree_slots
                    ),
                    "reference_logical_tree_slots": _slot_sample(logical_tree_slots),
                    "reference_logical_start": logical_start,
                    "reference_logical_len": logical_len,
                    "reference_attention_scale": float(self.scale),
                    "reference_key_cache_dtype": str(key_cache.dtype),
                    "reference_value_cache_dtype": str(value_cache.dtype),
                    "reference_key_cache_shape": [
                        int(dim) for dim in key_cache.shape
                    ],
                    "reference_value_cache_shape": [
                        int(dim) for dim in value_cache.shape
                    ],
                    "reference_key_cache_stride": [
                        int(stride) for stride in key_cache.stride()
                    ],
                    "reference_value_cache_stride": [
                        int(stride) for stride in value_cache.stride()
                    ],
                    "reference_slot_blocks_head": _slot_sample(blocks),
                    "reference_slot_offsets_head": _slot_sample(offsets),
                    "reference_selected_key_abs_max": float(
                        main_key.abs().max().detach().cpu().item()
                    ),
                    "reference_selected_value_abs_max": float(
                        main_value.abs().max().detach().cpu().item()
                    ),
                    "reference_selected_key_norm_sample": _float_sample(
                        main_key.flatten(start_dim=1).norm(dim=1)
                    ),
                    "reference_selected_value_norm_sample": _float_sample(
                        main_value.flatten(start_dim=1).norm(dim=1)
                    ),
                    "reference_repeated_key_abs_max": float(
                        main_k_for_heads.abs().max().detach().cpu().item()
                    ),
                    "reference_repeated_value_abs_max": float(
                        main_v_for_heads.abs().max().detach().cpu().item()
                    ),
                    "reference_tree_key_head0_scores": _float_sample(
                        unscaled_scores[0, tree_score_indices]
                    ),
                    "reference_context_tail_head0_scores": _float_sample(
                        unscaled_scores[0, -min(8, unscaled_scores.shape[1]) :]
                    ),
                    "reference_kernel_block_m": int(block_m),
                    "reference_kernel_block_q": int(block_q),
                    "reference_kernel_q_block_local_idx": int(q_block_local_idx),
                    "reference_kernel_q_block_lo": int(q_block_lo),
                    "reference_kernel_q_block_hi": int(q_block_hi),
                    "reference_query_norms": [
                        float(x) for x in q.norm(dim=1).detach().cpu()
                    ],
                    "reference_unscaled_score_min_max_by_head": [
                        [float(v) for v in row]
                        for row in unscaled_score_ranges.detach().cpu()
                    ],
                    "reference_score_min_max_by_head": [
                        [float(v) for v in row] for row in score_ranges.detach().cpu()
                    ],
                    "reference_prob_min_max_by_head": [
                        [float(v) for v in row] for row in prob_ranges.detach().cpu()
                    ],
                    "reference_output_norm": float(ref.norm().detach().cpu().item()),
                    "kernel_reference_max_abs_diff": float(
                        diff.max().detach().cpu().item()
                    ),
                    "kernel_reference_mean_abs_diff": float(
                        diff.mean().detach().cpu().item()
                    ),
                    "kernel_reference_head_max_abs_diff": [
                        float(x) for x in head_diffs.detach().cpu()
                    ],
                    "kernel_sequential_tree_max_abs_diff": float(
                        (kernel_out - sequential_ref).abs().max().detach().cpu().item()
                    ),
                    "kernel_all_tree_max_abs_diff": float(
                        (kernel_out - all_tree_ref).abs().max().detach().cpu().item()
                    ),
                    "kernel_canonical_tree_max_abs_diff": float(
                        (kernel_out - canonical_ref).abs().max().detach().cpu().item()
                    ),
                    "kernel_logical_tree_max_abs_diff": float(
                        (kernel_out - logical_ref).abs().max().detach().cpu().item()
                    ),
                    "kernel_next_query_same_keys_max_abs_diff": next_query_diff,
                    "reference_head0_top_prob_indices": [
                        int(x) for x in top_prob_indices.detach().cpu()
                    ],
                    "reference_head0_top_prob_values": [
                        float(x) for x in top_prob_values.detach().cpu()
                    ],
                }
            )
        return results

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TreeAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with TreeAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for TreeAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        key_cache, value_cache = kv_cache.unbind(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, key.shape[1])
        if prefill_meta := attn_metadata.prefill_metadata:
            unified_attention(
                q=query[num_decode_tokens:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[num_decode_tokens:num_actual_tokens],
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                seqused_k=prefill_meta.seq_lens,
                max_seqlen_k=prefill_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=prefill_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # Not supported
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )

        if decode_meta := attn_metadata.decode_metadata:
            dflash_debug_kernel = "triton_unified"
            if decode_meta.ancestor_masks is not None:
                dflash_debug_kernel = "optimus_tree"
                try:
                    from optimus_cutedsl.flash_attn import (
                        flash_attn_varlen_tree_paged_sm90,
                        flash_attn_varlen_tree_paged_sm100,
                    )
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        "tree_attn_kernel='optimus' requires the optimus_cutedsl "
                        "package. Set PYTHONPATH to include the optimus src dir, "
                        "e.g. PYTHONPATH=/path/to/optimus_jit_local/src:$PYTHONPATH"
                    ) from exc
                requested_optimus_arch = os.getenv(
                    "OPTIMUS_TREE_KERNEL_ARCH", ""
                ).lower()
                compute_capability = torch.cuda.get_device_capability(query.device)[0]
                device_name = torch.cuda.get_device_name(query.device).lower()
                use_sm100 = (
                    requested_optimus_arch in ("sm100", "blackwell", "b300")
                    or compute_capability >= 10
                    or "b300" in device_name
                    or "blackwell" in device_name
                )
                if use_sm100:
                    optimus_tree_kernel = flash_attn_varlen_tree_paged_sm100
                else:
                    optimus_tree_kernel = flash_attn_varlen_tree_paged_sm90
                optimus_tree_kernel(
                    q=query[:num_decode_tokens],
                    k=key_cache,
                    v=value_cache,
                    tree_mask=decode_meta.ancestor_masks,
                    cu_seqlens_q=decode_meta.query_start_loc.to(torch.int32),
                    seqused_k=decode_meta.seq_lens.to(torch.int32),
                    page_table=decode_meta.block_table.to(torch.int32),
                    logical_kv_slots=decode_meta.logical_kv_slots,
                    logical_kv_slot_lens=decode_meta.logical_kv_slot_lens,
                    logical_kv_starts=decode_meta.logical_kv_starts,
                    softmax_scale=self.scale,
                    out=output[:num_decode_tokens],
                )
            else:
                # When per_query_abs_pos is provided (tree verification), each
                # tree node carries a depth-based absolute position used for
                # causal masking and sliding-window bounds.  This ensures exact
                # AR parity: a tree node at depth d sees window [L+d-W, L+d],
                # regardless of its sequential index S within the batch.
                # The old heuristic of expanding the window by (max_query_len-1)
                # is no longer needed because the kernel now uses the correct
                # per-node position for every node, including the root.
                unified_attention(
                    q=query[:num_decode_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_decode_tokens],
                    cu_seqlens_q=decode_meta.query_start_loc,
                    max_seqlen_q=decode_meta.max_query_len,
                    seqused_k=decode_meta.seq_lens,
                    max_seqlen_k=decode_meta.max_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    alibi_slopes=self.alibi_slopes,
                    qq_bias=decode_meta.tree_attn_bias,
                    logical_kv_slots=decode_meta.logical_kv_slots,
                    logical_kv_slot_lens=decode_meta.logical_kv_slot_lens,
                    logical_kv_starts=decode_meta.logical_kv_starts,
                    window_size=self.sliding_window,
                    block_table=decode_meta.block_table,
                    softcap=self.logits_soft_cap,
                    q_descale=None,  # Not supported
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                    per_query_abs_pos=decode_meta.per_query_abs_pos,
                )
            self._maybe_record_dflash_verifier_attention_probe(
                layer=layer,
                query=query[:num_decode_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                decode_meta=decode_meta,
                output=output[:num_decode_tokens],
                kernel=dflash_debug_kernel,
            )
        return output
