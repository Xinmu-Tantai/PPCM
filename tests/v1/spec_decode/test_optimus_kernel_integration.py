# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for optimus SM90 tree-attention kernel integration.

Covers:
  1. build_ancestor_matrix_np / build_causal_ancestor_matrix_np correctness
  2. Ancestor matrix vs attention bias semantic equivalence
  3. DFlashTreeSpecDecodeMetadata conditional field population
  4. TreeAttentionMetadata field propagation
  5. Config plumbing (tree_attn_kernel in SpeculativeConfig)
  6. dflash_profiling.py CLI argument parsing

Run:
    python -m pytest tests/v1/spec_decode/test_optimus_kernel_integration.py -v
"""

import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm.v1.spec_decode.dflash_tree import (
    _build_attention_bias_np,
    build_ancestor_matrix_np,
    build_attention_bias_from_parents,
    build_causal_ancestor_matrix_np,
)
from vllm.v1.spec_decode.metadata import (
    DFlashTreeSpecDecodeMetadata,
)

try:
    from optimus_cutedsl.flash_attn import flash_attn_varlen_tree_paged_sm90
except Exception:  # pragma: no cover - optional dependency
    flash_attn_varlen_tree_paged_sm90 = None


_DEBUG_DIR_ENV = "DFLASH_KERNEL_DEBUG_DIR"
_DEFAULT_DEBUG_DIR = Path("/tmp/debug_kernel")


# ── helpers ──────────────────────────────────────────────────────────────

def _make_dummy_metadata(
    draft_token_ids_per_req: list[list[int]],
    *,
    use_optimus: bool = False,
    parent_lists: list[list[int]] | None = None,
    depth_lists: list[list[int]] | None = None,
) -> DFlashTreeSpecDecodeMetadata:
    """Build a DFlashTreeSpecDecodeMetadata on CPU for testing."""
    device = torch.device("cpu")
    batch_size = len(draft_token_ids_per_req)
    num_draft_tokens = [len(ids) for ids in draft_token_ids_per_req]
    num_sampled_tokens = [n + 1 for n in num_draft_tokens]
    flat_draft = sum(draft_token_ids_per_req, [])
    num_tokens = len(flat_draft)
    query_lens = [n + 1 for n in num_draft_tokens]

    draft_token_ids_t = torch.tensor(flat_draft, dtype=torch.int32, device=device)
    cu_num_draft_np = np.cumsum(num_draft_tokens, dtype=np.int32)
    cu_num_sampled_np = np.cumsum(num_sampled_tokens, dtype=np.int32)
    cu_query_lens_np = np.cumsum(query_lens, dtype=np.int32)

    logits_indices_np = np.repeat(
        cu_num_sampled_np - np.array(num_sampled_tokens, dtype=np.int32),
        num_sampled_tokens,
    )
    if logits_indices_np.size:
        logits_indices_np += np.arange(logits_indices_np.size, dtype=np.int32)

    target_logits_indices_np = np.repeat(
        cu_num_sampled_np - np.array(num_sampled_tokens, dtype=np.int32),
        num_draft_tokens,
    )
    if target_logits_indices_np.size:
        target_logits_indices_np += np.arange(
            target_logits_indices_np.size, dtype=np.int32
        )
    bonus_logits_indices_np = cu_num_sampled_np - 1

    full_parent_lists: list[list[int]] = []
    full_depth_lists: list[list[int]] = []
    for req_idx, query_len in enumerate(query_lens):
        draft_len = num_draft_tokens[req_idx]
        if parent_lists is None:
            parents = [-1, *range(query_len - 1)]
        else:
            parents = parent_lists[req_idx]
            if len(parents) == draft_len:
                parents = [-1, *parents]
        if depth_lists is None:
            depths = [0] * len(parents)
            for idx in range(1, len(parents)):
                depths[idx] = depths[parents[idx]] + 1
        else:
            depths = depth_lists[req_idx]
            if len(depths) == draft_len:
                depths = [0, *depths]
        assert len(parents) == query_len
        assert len(depths) == query_len
        full_parent_lists.append(list(parents))
        full_depth_lists.append(list(depths))

    flat_parents = [parent for parents in full_parent_lists for parent in parents]
    flat_depths = [depth for depths in full_depth_lists for depth in depths]

    tree_attn_bias = None
    ancestor_masks = None
    if use_optimus:
        max_qlen = max(query_lens, default=0)
        ancestor_masks = torch.zeros(
            (batch_size, max_qlen, max_qlen), dtype=torch.int32, device=device
        )
        for req_idx, parents in enumerate(full_parent_lists):
            ancestor = torch.from_numpy(build_ancestor_matrix_np(parents)).to(
                device=device, dtype=torch.int32
            )
            qlen = len(parents)
            ancestor_masks[req_idx, :qlen, :qlen] = ancestor
    else:
        total_query_len = sum(query_lens)
        neg_inf = float(torch.finfo(torch.float32).min)
        tree_attn_bias = torch.full(
            (total_query_len, total_query_len),
            neg_inf,
            dtype=torch.float32,
            device=device,
        )
        cursor = 0
        for parents in full_parent_lists:
            qlen = len(parents)
            bias = torch.from_numpy(_build_attention_bias_np(parents, neg_inf)).to(
                device=device, dtype=torch.float32
            )
            tree_attn_bias[cursor:cursor + qlen, cursor:cursor + qlen] = bias
            cursor += qlen

    return DFlashTreeSpecDecodeMetadata(
        draft_token_ids=draft_token_ids_t,
        num_draft_tokens=num_draft_tokens,
        cu_num_draft_tokens=torch.tensor(
            cu_num_draft_np, dtype=torch.int32, device=device
        ),
        cu_num_sampled_tokens=torch.tensor(
            cu_num_sampled_np, dtype=torch.int32, device=device
        ),
        target_logits_indices=torch.tensor(
            target_logits_indices_np, dtype=torch.int32, device=device
        ),
        bonus_logits_indices=torch.tensor(
            bonus_logits_indices_np, dtype=torch.int32, device=device
        ),
        logits_indices=torch.tensor(
            logits_indices_np, dtype=torch.int32, device=device
        ),
        query_lens=query_lens,
        parent_indices=torch.tensor(flat_parents, dtype=torch.int64, device=device),
        depths=torch.tensor(flat_depths, dtype=torch.int64, device=device),
        tree_attn_bias=tree_attn_bias,
        cu_query_lens=torch.tensor(cu_query_lens_np, dtype=torch.int32, device=device),
        is_tree_req=[n > 0 for n in num_draft_tokens],
        ancestor_masks=ancestor_masks,
    )


def _get_debug_dir() -> Path:
    debug_dir = Path(os.environ.get(_DEBUG_DIR_ENV, str(_DEFAULT_DEBUG_DIR)))
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _write_debug_artifacts(name: str, payload: dict) -> None:
    debug_dir = _get_debug_dir()
    torch_payload = {}
    json_payload = {}

    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            torch_payload[key] = tensor
            json_payload[key] = tensor.tolist()
        else:
            torch_payload[key] = value
            json_payload[key] = value

    torch.save(torch_payload, debug_dir / f"{name}.pt")
    (debug_dir / f"{name}.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _pack_kv_to_paged(
    key: torch.Tensor,
    value: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_kv_heads, kv_len, head_dim = key.shape
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    padded_kv_len = num_pages_per_seq * page_size
    if padded_kv_len != kv_len:
        pad = padded_kv_len - kv_len
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
    key_pages = key.permute(0, 2, 1, 3).contiguous().view(
        batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
    )
    value_pages = value.permute(0, 2, 1, 3).contiguous().view(
        batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
    )
    page_table = torch.arange(
        batch_size * num_pages_per_seq,
        device=key.device,
        dtype=torch.int32,
    ).view(batch_size, num_pages_per_seq)
    return (
        key_pages.view(batch_size * num_pages_per_seq, page_size, num_kv_heads, head_dim),
        value_pages.view(batch_size * num_pages_per_seq, page_size, num_kv_heads, head_dim),
        page_table,
    )


def _reference_tree_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ancestor: torch.Tensor,
    prefix_len: int,
    sm_scale: float,
) -> torch.Tensor:
    batch_size, num_heads, num_queries, head_dim = query.shape
    _, num_kv_heads, kv_len, _ = key.shape
    num_kv_groups = num_heads // num_kv_heads
    if num_kv_groups > 1:
        key = key.repeat_interleave(num_kv_groups, dim=1)
        value = value.repeat_interleave(num_kv_groups, dim=1)

    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * sm_scale
    q_idx = torch.arange(num_queries, device=query.device)
    kv_idx = torch.arange(kv_len, device=query.device)
    is_prefix = kv_idx[None, :] < prefix_len
    in_tree = kv_idx[None, :] >= prefix_len
    tree_kv = (kv_idx[None, :] - prefix_len).clamp(min=0)
    is_ancestor = ancestor[q_idx[:, None], tree_kv].bool()
    attend = is_prefix | (in_tree & is_ancestor)
    scores.masked_fill_(~attend.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value.float())
    return output.to(query.dtype)


def _make_tree_tensors(
    parents: list[int],
    *,
    root_token: int,
    tree_tokens: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    depths = [0] * len(parents)
    for idx in range(1, len(parents)):
        depths[idx] = depths[parents[idx]] + 1
    token_ids = torch.tensor([root_token, *tree_tokens], dtype=torch.long, device=device)
    parent_indices = torch.tensor(parents, dtype=torch.long, device=device)
    depth = torch.tensor(depths, dtype=torch.long, device=device)
    return token_ids, parent_indices, depth


def _run_tree_attention_equivalence_case(
    *,
    name: str,
    parents: list[int],
    root_token: int,
    tree_tokens: list[int],
    prefix_len: int = 64,
    batch_size: int = 1,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
) -> dict[str, object]:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    token_ids, parent_indices, depths = _make_tree_tensors(
        parents,
        root_token=root_token,
        tree_tokens=tree_tokens,
        device=device,
    )
    num_queries = token_ids.numel()
    kv_len = prefix_len + num_queries
    sm_scale = head_dim ** -0.5

    ancestor = torch.from_numpy(
        build_ancestor_matrix_np(parents)
    ).to(device=device, dtype=torch.int32)
    query = torch.randn(
        batch_size, num_heads, num_queries, head_dim,
        dtype=dtype, device=device,
    )
    key = torch.randn(
        batch_size, num_kv_heads, kv_len, head_dim,
        dtype=dtype, device=device,
    )
    value = torch.randn(
        batch_size, num_kv_heads, kv_len, head_dim,
        dtype=dtype, device=device,
    )

    ref_out = _reference_tree_attention(
        query, key, value, ancestor, prefix_len, sm_scale
    )
    k_exp = key.repeat_interleave(num_heads // num_kv_heads, dim=1)
    v_exp = value.repeat_interleave(num_heads // num_kv_heads, dim=1)
    tree_bias = build_attention_bias_from_parents(
        parent_indices,
        dtype=dtype,
        device=device,
    )
    dense_mask = torch.zeros(
        1, 1, num_queries, kv_len, dtype=dtype, device=device
    )
    dense_mask[:, :, :, prefix_len:] = tree_bias.unsqueeze(0).unsqueeze(0)
    sdpa_out = F.scaled_dot_product_attention(
        query, k_exp, v_exp, attn_mask=dense_mask, scale=sm_scale
    )

    q_varlen = query.permute(0, 2, 1, 3).contiguous().view(
        batch_size * num_queries, num_heads, head_dim
    )
    k_paged, v_paged, page_table = _pack_kv_to_paged(key, value, 128)
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * num_queries, num_queries,
        device=device, dtype=torch.int32,
    )
    context_lens = torch.full(
        (batch_size,), kv_len, device=device, dtype=torch.int32
    )
    optimus_out = flash_attn_varlen_tree_paged_sm90(
        q_varlen,
        k_paged,
        v_paged,
        ancestor.to(torch.uint8),
        cu_seqlens_q,
        context_lens,
        page_table,
        softmax_scale=sm_scale,
        pack_gqa=num_kv_heads != num_heads,
        m_block_size=128,
        n_block_size=128,
    ).view(batch_size, num_queries, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    ref_vs_sdpa = (ref_out.float() - sdpa_out.float()).abs()
    ref_vs_optimus = (ref_out.float() - optimus_out.float()).abs()
    sdpa_vs_optimus = (sdpa_out.float() - optimus_out.float()).abs()

    ref_vs_sdpa_per_query = ref_vs_sdpa.amax(dim=(0, 1, 3))
    ref_vs_optimus_per_query = ref_vs_optimus.amax(dim=(0, 1, 3))
    sdpa_vs_optimus_per_query = sdpa_vs_optimus.amax(dim=(0, 1, 3))
    unique_depths = sorted(set(depths.tolist()))
    ref_vs_sdpa_per_depth_max = []
    ref_vs_optimus_per_depth_max = []
    sdpa_vs_optimus_per_depth_max = []
    for depth in unique_depths:
        depth_mask = depths == depth
        ref_vs_sdpa_per_depth_max.append(
            float(ref_vs_sdpa_per_query[depth_mask].max().item())
        )
        ref_vs_optimus_per_depth_max.append(
            float(ref_vs_optimus_per_query[depth_mask].max().item())
        )
        sdpa_vs_optimus_per_depth_max.append(
            float(sdpa_vs_optimus_per_query[depth_mask].max().item())
        )

    payload = {
        "token_ids": token_ids,
        "parent_indices": parent_indices,
        "depths": depths,
        "ancestor": ancestor,
        "prefix_len": prefix_len,
        "sm_scale": sm_scale,
        "ref_out": ref_out,
        "sdpa_out": sdpa_out,
        "optimus_out": optimus_out,
        "ref_vs_sdpa_max_abs_diff": float(ref_vs_sdpa.max().item()),
        "ref_vs_sdpa_mean_abs_diff": float(ref_vs_sdpa.mean().item()),
        "ref_vs_optimus_max_abs_diff": float(ref_vs_optimus.max().item()),
        "ref_vs_optimus_mean_abs_diff": float(ref_vs_optimus.mean().item()),
        "sdpa_vs_optimus_max_abs_diff": float(sdpa_vs_optimus.max().item()),
        "sdpa_vs_optimus_mean_abs_diff": float(sdpa_vs_optimus.mean().item()),
        "ref_vs_sdpa_per_query_max_abs_diff": ref_vs_sdpa_per_query,
        "ref_vs_optimus_per_query_max_abs_diff": ref_vs_optimus_per_query,
        "sdpa_vs_optimus_per_query_max_abs_diff": sdpa_vs_optimus_per_query,
        "unique_depths": unique_depths,
        "ref_vs_sdpa_per_depth_max_abs_diff": ref_vs_sdpa_per_depth_max,
        "ref_vs_optimus_per_depth_max_abs_diff": ref_vs_optimus_per_depth_max,
        "sdpa_vs_optimus_per_depth_max_abs_diff": sdpa_vs_optimus_per_depth_max,
    }
    _write_debug_artifacts(name, payload)
    return payload


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_optimus_root_posterior_matches_dense_sdpa_debug_dump():
    if flash_attn_varlen_tree_paged_sm90 is None:
        pytest.skip("optimus_cutedsl is not importable")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 required for Optimus tree kernel")

    payload = _run_tree_attention_equivalence_case(
        name="optimus_root_posterior_equivalence",
        parents=[-1, 0, 0, 1, 1, 2, 2],
        root_token=42,
        tree_tokens=[101, 102, 201, 202, 301, 302],
    )

    ref_out = payload["ref_out"]
    sdpa_out = payload["sdpa_out"]
    optimus_out = payload["optimus_out"]
    root_ref = ref_out[:, :, 0, :]
    root_sdpa = sdpa_out[:, :, 0, :]
    root_optimus = optimus_out[:, :, 0, :]

    torch.testing.assert_close(root_ref, root_sdpa, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(root_ref, root_optimus, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_optimus_full_tree_matches_dense_sdpa_debug_dump():
    if flash_attn_varlen_tree_paged_sm90 is None:
        pytest.skip("optimus_cutedsl is not importable")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 required for Optimus tree kernel")

    payload = _run_tree_attention_equivalence_case(
        name="optimus_full_tree_equivalence",
        parents=[-1, 0, 0, 1, 1, 2, 2, 3, 3, 6, 6],
        root_token=17,
        tree_tokens=[31, 32, 41, 42, 51, 52, 61, 62, 71, 72],
        prefix_len=96,
    )

    torch.testing.assert_close(
        payload["ref_out"], payload["sdpa_out"], atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        payload["ref_out"], payload["optimus_out"], atol=1e-2, rtol=1e-2
    )


# ── 1. build_ancestor_matrix_np ─────────────────────────────────────────

class TestBuildAncestorMatrixNp:

    def test_chain_topology(self):
        """Linear chain: 0 -> 1 -> 2 -> 3."""
        parents = [-1, 0, 1, 2]
        anc = build_ancestor_matrix_np(parents)
        assert anc.shape == (4, 4)
        assert anc.dtype == np.int32
        np.testing.assert_array_equal(anc, np.tril(np.ones((4, 4), dtype=np.int32)))

    def test_binary_tree(self):
        """root(0) -> A(1), B(2); A -> C(3), D(4)."""
        parents = [-1, 0, 0, 1, 1]
        anc = build_ancestor_matrix_np(parents)
        assert anc.shape == (5, 5)
        # C(3) sees root(0), A(1), and self(3)
        assert anc[3, 0] == 1
        assert anc[3, 1] == 1
        assert anc[3, 3] == 1
        # C(3) does NOT see B(2) or D(4)
        assert anc[3, 2] == 0
        assert anc[3, 4] == 0
        # B(2) sees root(0) and self(2)
        assert anc[2, 0] == 1
        assert anc[2, 2] == 1
        assert anc[2, 1] == 0

    def test_single_node(self):
        parents = [-1]
        anc = build_ancestor_matrix_np(parents)
        np.testing.assert_array_equal(anc, np.array([[1]], dtype=np.int32))

    def test_root_row_is_identity(self):
        parents = [-1, 0, 0, 1, 2]
        anc = build_ancestor_matrix_np(parents)
        expected_root_row = np.zeros(5, dtype=np.int32)
        expected_root_row[0] = 1
        np.testing.assert_array_equal(anc[0], expected_root_row)

    def test_diagonal_always_one(self):
        parents = [-1, 0, 1, 0, 3, 3]
        anc = build_ancestor_matrix_np(parents)
        for i in range(6):
            assert anc[i, i] == 1

    def test_column_zero_all_ones(self):
        """Every node must see the root."""
        parents = [-1, 0, 0, 1, 2]
        anc = build_ancestor_matrix_np(parents)
        np.testing.assert_array_equal(anc[:, 0], np.ones(5, dtype=np.int32))


class TestBuildCausalAncestorMatrixNp:

    def test_is_lower_triangular(self):
        anc = build_causal_ancestor_matrix_np(5)
        expected = np.tril(np.ones((5, 5), dtype=np.int32))
        np.testing.assert_array_equal(anc, expected)

    def test_size_one(self):
        anc = build_causal_ancestor_matrix_np(1)
        np.testing.assert_array_equal(anc, np.array([[1]], dtype=np.int32))


# ── 2. Semantic equivalence: ancestor matrix vs attention bias ───────────

class TestAncestorVsBiasEquivalence:

    @pytest.mark.parametrize(
        "parents",
        [
            [-1, 0, 1, 2],
            [-1, 0, 0, 1, 1],
            [-1, 0, 0, 1, 2, 2, 3],
        ],
    )
    def test_ancestor_matches_bias(self, parents):
        """ancestor[i,j]==1 iff bias[i,j]==0.0 (not -inf)."""
        anc = build_ancestor_matrix_np(parents)
        neg_inf = float(torch.finfo(torch.float32).min)
        bias_np = _build_attention_bias_np(parents, neg_inf)
        allowed = (bias_np == 0.0).astype(np.int32)
        np.testing.assert_array_equal(anc, allowed)

    def test_causal_equivalence(self):
        n = 6
        anc = build_causal_ancestor_matrix_np(n)
        parents = [-1] + list(range(n - 1))
        neg_inf = float(torch.finfo(torch.float32).min)
        bias_np = _build_attention_bias_np(parents, neg_inf)
        allowed = (bias_np == 0.0).astype(np.int32)
        np.testing.assert_array_equal(anc, allowed)


# ── 3. DFlashTreeSpecDecodeMetadata conditional fields ───────────────────

class TestMetadataConditionalFields:

    def test_triton_path_has_bias_no_ancestors(self):
        meta = _make_dummy_metadata([[10, 20, 30]], use_optimus=False)
        assert meta.tree_attn_bias is not None
        assert meta.ancestor_masks is None

    def test_optimus_path_has_ancestors_no_bias(self):
        meta = _make_dummy_metadata([[10, 20, 30]], use_optimus=True)
        assert meta.tree_attn_bias is None
        assert meta.ancestor_masks is not None

    def test_optimus_ancestor_shape(self):
        meta = _make_dummy_metadata(
            [[10, 20], [30, 40, 50, 60]], use_optimus=True
        )
        B = 2
        max_qlen = max(meta.query_lens)
        assert meta.ancestor_masks.shape == (B, max_qlen, max_qlen)

    def test_optimus_ancestor_padding_is_zero(self):
        meta = _make_dummy_metadata(
            [[10], [30, 40, 50]], use_optimus=True
        )
        short_qlen = meta.query_lens[0]
        max_qlen = max(meta.query_lens)
        padded_region = meta.ancestor_masks[0, short_qlen:, :]
        assert (padded_region == 0).all()


# ── 4. TreeAttentionMetadata field propagation ───────────────────────────

class TestTreeAttentionMetadataFields:

    def test_has_ancestor_masks_field(self):
        from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata
        field_names = {f.name for f in fields(TreeAttentionMetadata)}
        assert "ancestor_masks" in field_names

    def test_decode_metadata_propagates_ancestor_masks(self):
        from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata

        anc = torch.eye(3, dtype=torch.int32).unsqueeze(0)
        meta = TreeAttentionMetadata(
            num_actual_tokens=3,
            max_query_len=3,
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            max_seq_len=8,
            seq_lens=torch.tensor([8], dtype=torch.int32),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            slot_mapping=torch.zeros(3, dtype=torch.int64),
            num_prefill_tokens=0,
            num_decode_tokens=3,
            num_prefills=0,
            num_decodes=1,
            ancestor_masks=anc,
        )
        dm = meta.decode_metadata
        assert dm is not None
        assert dm.ancestor_masks is not None
        torch.testing.assert_close(dm.ancestor_masks, anc)


# ── 5. SpeculativeConfig tree_attn_kernel field ─────────────────────────

class TestSpeculativeConfigKernel:

    def test_field_exists_with_default(self):
        from dataclasses import fields as dc_fields
        from vllm.config.speculative import SpeculativeConfig
        field_map = {f.name: f for f in dc_fields(SpeculativeConfig)}
        assert "tree_attn_kernel" in field_map
        assert field_map["tree_attn_kernel"].default == "triton"

    def test_valid_choices(self):
        """Verify type annotation accepts 'triton' and 'optimus'."""
        import typing
        from vllm.config.speculative import SpeculativeConfig
        hints = typing.get_type_hints(SpeculativeConfig)
        args = typing.get_args(hints["tree_attn_kernel"])
        assert "triton" in args
        assert "optimus" in args

    def test_tree_draft_includes_top2gap_fanout(self):
        """Verify DFlash tree_draft accepts top2gap_fanout."""
        import typing
        from vllm.config.speculative import SpeculativeConfig

        hints = typing.get_type_hints(SpeculativeConfig)
        args = typing.get_args(hints["tree_draft"])
        assert "top2gap_fanout" in args

    def test_tree_kv_layout_field(self):
        """Verify DFlash tree KV layout defaults to physical compaction."""
        import typing
        from dataclasses import fields as dc_fields
        from vllm.config.speculative import SpeculativeConfig

        field_map = {f.name: f for f in dc_fields(SpeculativeConfig)}
        assert "tree_kv_layout" in field_map
        assert field_map["tree_kv_layout"].default == "physical"

        hints = typing.get_type_hints(SpeculativeConfig)
        args = typing.get_args(hints["tree_kv_layout"])
        assert "physical" in args
        assert "logical" in args

    def test_num_cudagraph_tree_captures_uses_budget_only(self):
        from vllm.config.speculative import SpeculativeConfig

        config = object.__new__(SpeculativeConfig)
        config.method = "dflash"
        config.num_speculative_tokens = 15
        config.tree_width = 7
        config.max_tree_budget = 255
        config.num_cudagraph_tree_captures = 4

        assert config.cudagraph_tree_capture_sizes == [255]

    def test_single_cudagraph_tree_capture_uses_budget(self):
        from vllm.config.speculative import SpeculativeConfig

        config = object.__new__(SpeculativeConfig)
        config.method = "dflash"
        config.num_speculative_tokens = 15
        config.tree_width = 7
        config.max_tree_budget = 255
        config.num_cudagraph_tree_captures = 1

        assert config.cudagraph_tree_capture_sizes == [255]


# ── 6. dflash_profiling.py CLI argument parsing ─────────────────────────

class TestDflashProfilingCLI:

    def test_tree_attn_kernel_help(self):
        result = subprocess.run(
            [
                sys.executable,
                "examples/offline_inference/dflash_profiling.py",
                "--help",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[3],
        )
        assert "--tree-attn-kernel" in result.stdout
        assert "triton" in result.stdout
        assert "optimus" in result.stdout
        assert "--tree-kv-layout" in result.stdout
        assert "physical" in result.stdout
        assert "logical" in result.stdout


# ── 7. Ancestor matrix properties (fuzz-like) ───────────────────────────

class TestAncestorMatrixProperties:

    @pytest.mark.parametrize("seed", range(5))
    def test_random_tree_properties(self, seed):
        rng = np.random.RandomState(seed)
        n = rng.randint(2, 30)
        parents = [-1]
        for i in range(1, n):
            parents.append(rng.randint(0, i))
        anc = build_ancestor_matrix_np(parents)

        assert anc.shape == (n, n)
        for i in range(n):
            assert anc[i, i] == 1, "diagonal must be 1"
        np.testing.assert_array_equal(
            anc[:, 0], np.ones(n, dtype=np.int32),
            err_msg="all nodes must see root"
        )
        for i in range(1, n):
            p = parents[i]
            for j in range(n):
                if anc[p, j] == 1:
                    assert anc[i, j] == 1, (
                        f"node {i} must inherit ancestors of parent {p}"
                    )

    @pytest.mark.parametrize("seed", range(5))
    def test_symmetry_with_bias(self, seed):
        rng = np.random.RandomState(seed)
        n = rng.randint(2, 20)
        parents = [-1]
        for i in range(1, n):
            parents.append(rng.randint(0, i))
        anc = build_ancestor_matrix_np(parents)
        neg_inf = float(torch.finfo(torch.float32).min)
        bias = _build_attention_bias_np(parents, neg_inf)
        allowed = (bias == 0.0).astype(np.int32)
        np.testing.assert_array_equal(anc, allowed)


# ── 8. build_for_dflash_tree builder accepts None bias ───────────────────

class TestBuilderAcceptsNoneBias:

    def test_metadata_accepts_none_bias_with_ancestors(self):
        from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata

        anc = torch.eye(3, dtype=torch.int32).unsqueeze(0)
        meta = TreeAttentionMetadata(
            num_actual_tokens=3,
            max_query_len=3,
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            max_seq_len=8,
            seq_lens=torch.tensor([8], dtype=torch.int32),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            slot_mapping=torch.zeros(3, dtype=torch.int64),
            num_prefill_tokens=0,
            num_decode_tokens=3,
            num_prefills=0,
            num_decodes=1,
            tree_attn_bias=None,
            ancestor_masks=anc,
        )
        assert meta.tree_attn_bias is None
        assert meta.ancestor_masks is not None
        dm = meta.decode_metadata
        assert dm is not None
        assert dm.tree_attn_bias is None
        assert dm.ancestor_masks is not None
