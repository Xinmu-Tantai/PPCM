# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import torch
import torch.nn.functional as F

from vllm.model_executor.models.dflash_tree_post_head import (
    DFlashTreePostHead,
    TreePostHeadConfig,
)
from vllm.v1.spec_decode.dflash_tree import (
    DraftTree,
    build_tree_from_topk,
    build_trees_from_topk,
    pack_seed_tree_batch,
    select_prefix_closed_subtrees,
)


def _make_seed_tree() -> DraftTree:
    return DraftTree(
        token_ids=torch.tensor([99, 10, 11, 20, 21]),
        parent_indices=torch.tensor([-1, 0, 0, 1, 1]),
        depth=torch.tensor([0, 1, 1, 2, 2]),
        num_nodes=5,
        seed_edge_logprobs=torch.tensor([0.0, -0.4, -1.2, -0.3, -1.7]),
        seed_ranks=torch.tensor([-1, 0, 1, 0, 1], dtype=torch.int16),
    )


def _make_head() -> DFlashTreePostHead:
    return DFlashTreePostHead(
        TreePostHeadConfig(
            input_hidden_size=8,
            max_depth=2,
            tree_width=2,
            hidden_size=8,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            scorer_size=4,
        )
    )


def _run_head(head: DFlashTreePostHead, node_embeddings: torch.Tensor):
    seed = pack_seed_tree_batch([_make_seed_tree()], seed_budget=5)
    return head(
        root_hidden=torch.randn(1, 8),
        depth_hidden=torch.randn(1, 2, 8),
        node_embeddings=node_embeddings,
        parent_indices=seed.parent_indices,
        depths=seed.depths,
        valid_mask=seed.valid_mask,
        seed_edge_logprobs=seed.seed_edge_logprobs,
        seed_ranks=seed.seed_ranks,
        ancestor_mask=seed.ancestor_mask,
    )


def test_patr_zero_residual_recovers_seed_edges():
    torch.manual_seed(0)
    head = _make_head()
    seed = _make_seed_tree()
    output = _run_head(head, torch.randn(1, 5, 8))
    torch.testing.assert_close(
        output.refined_edge_logprobs[0, 1:],
        seed.seed_edge_logprobs[1:],
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        output.refined_edge_logprobs[0, 1:3].exp().sum(),
        output.candidate_mass[0, 0],
    )
    torch.testing.assert_close(
        output.refined_edge_logprobs[0, 3:5].exp().sum(),
        output.candidate_mass[0, 1],
    )
    assert torch.isfinite(output.refined_path_logprobs).all()


def test_patr_ancestor_mask_isolates_siblings():
    torch.manual_seed(1)
    head = _make_head().eval()
    embeddings = torch.randn(1, 5, 8)
    changed = embeddings.clone()
    changed[:, 2] += 10
    # Reuse identical context inputs to isolate the tree mask itself.
    seed = pack_seed_tree_batch([_make_seed_tree()], seed_budget=5)
    root_hidden = torch.randn(1, 8)
    depth_hidden = torch.randn(1, 2, 8)
    kwargs = dict(
        root_hidden=root_hidden,
        depth_hidden=depth_hidden,
        parent_indices=seed.parent_indices,
        depths=seed.depths,
        valid_mask=seed.valid_mask,
        seed_edge_logprobs=seed.seed_edge_logprobs,
        seed_ranks=seed.seed_ranks,
        ancestor_mask=seed.ancestor_mask,
    )
    first = head(node_embeddings=embeddings, **kwargs)
    second = head(node_embeddings=changed, **kwargs)
    torch.testing.assert_close(first.node_states[:, 1], second.node_states[:, 1])


def test_patr_sparse_ancestor_attention_matches_dense_forward_and_gradient():
    torch.manual_seed(11)
    base_head = _make_head()
    sparse_head = DFlashTreePostHead(
        replace(
            base_head.config,
            use_sparse_ancestor_attention=True,
        )
    )
    sparse_head.load_state_dict(base_head.state_dict(), strict=True)
    dense_head = DFlashTreePostHead(
        replace(
            sparse_head.config,
            use_sparse_ancestor_attention=False,
        )
    )
    dense_head.load_state_dict(sparse_head.state_dict(), strict=True)
    seed = pack_seed_tree_batch(
        [_make_seed_tree(), _make_seed_tree()],
        seed_budget=7,
        max_depth=2,
        build_sparse_ancestor_indices=True,
    )
    root_hidden = torch.randn(2, 8)
    depth_hidden = torch.randn(2, 2, 8)
    dense_embeddings = torch.randn(2, 7, 8, requires_grad=True)
    sparse_embeddings = dense_embeddings.detach().clone().requires_grad_(True)
    common = dict(
        root_hidden=root_hidden,
        depth_hidden=depth_hidden,
        parent_indices=seed.parent_indices,
        depths=seed.depths,
        valid_mask=seed.valid_mask,
        seed_edge_logprobs=seed.seed_edge_logprobs,
        seed_ranks=seed.seed_ranks,
    )
    dense = dense_head(
        node_embeddings=dense_embeddings,
        ancestor_mask=seed.ancestor_mask,
        **common,
    )
    sparse = sparse_head(
        node_embeddings=sparse_embeddings,
        ancestor_indices=seed.ancestor_indices,
        ancestor_valid_mask=seed.ancestor_valid_mask,
        **common,
    )
    torch.testing.assert_close(
        sparse.node_states,
        dense.node_states,
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        sparse.refined_edge_logprobs,
        dense.refined_edge_logprobs,
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        sparse.refined_path_logprobs,
        dense.refined_path_logprobs,
        atol=2e-6,
        rtol=2e-6,
    )
    dense.refined_path_logprobs[seed.valid_mask].sum().backward()
    sparse.refined_path_logprobs[seed.valid_mask].sum().backward()
    torch.testing.assert_close(
        sparse_embeddings.grad,
        dense_embeddings.grad,
        atol=3e-6,
        rtol=3e-6,
    )


def test_patr_selection_remaps_prefix_closed_tree():
    seed = pack_seed_tree_batch([_make_seed_tree()], seed_budget=5)
    path_scores = torch.tensor([[0.0, -0.4, -1.2, -0.7, -2.1]])
    final = select_prefix_closed_subtrees(seed, path_scores, final_budget=3)[0]
    assert final.token_ids.tolist() == [99, 10, 20]
    assert final.parent_indices.tolist() == [-1, 0, 1]
    assert final.depth.tolist() == [0, 1, 2]


def test_patr_batched_selection_preserves_stable_tie_break():
    first = _make_seed_tree()
    second = _make_seed_tree()
    second.token_ids = second.token_ids + 100
    seed = pack_seed_tree_batch([first, second], seed_budget=5)
    # Nodes 1 and 2 tie on score. The stable secondary keys keep the old
    # index order; node 3 then wins over node 4.
    path_scores = torch.tensor([
        [0.0, -0.4, -0.4, -0.7, -2.1],
        [0.0, -0.4, -0.4, -0.7, -2.1],
    ])
    final = select_prefix_closed_subtrees(seed, path_scores, final_budget=4)
    assert len(final) == 2
    assert final[0].token_ids.tolist() == [99, 10, 11, 20]
    assert final[1].token_ids.tolist() == [199, 110, 111, 120]
    for tree in final:
        assert tree.parent_indices.tolist() == [-1, 0, 0, 1]
        assert tree.depth.tolist() == [0, 1, 1, 2]


def test_patr_padded_batch_has_finite_gradients():
    torch.manual_seed(2)
    head = _make_head()
    seed = pack_seed_tree_batch(
        [_make_seed_tree()],
        seed_budget=7,
        max_depth=2,
    )
    node_embeddings = torch.randn(1, 7, 8, requires_grad=True)
    output = head(
        root_hidden=torch.randn(1, 8),
        depth_hidden=torch.randn(1, 2, 8),
        node_embeddings=node_embeddings,
        parent_indices=seed.parent_indices,
        depths=seed.depths,
        valid_mask=seed.valid_mask,
        seed_edge_logprobs=seed.seed_edge_logprobs,
        seed_ranks=seed.seed_ranks,
        ancestor_mask=seed.ancestor_mask,
    )
    valid_paths = output.refined_path_logprobs[seed.valid_mask]
    assert torch.isfinite(valid_paths).all()
    assert torch.isneginf(output.refined_edge_logprobs[~seed.valid_mask]).all()
    valid_paths.sum().backward()
    assert node_embeddings.grad is not None
    assert torch.isfinite(node_embeddings.grad).all()
    for parameter in head.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_patr_vectorized_sibling_math_matches_group_definition():
    torch.manual_seed(3)
    head = _make_head()
    with torch.no_grad():
        head.edge_scale.normal_(std=0.1)
        head.scalar_mlp[-1].weight.normal_(std=0.1)
        head.mass_head.weight.normal_(std=0.1)
    seed = pack_seed_tree_batch([_make_seed_tree()], seed_budget=5)
    output = head(
        root_hidden=torch.randn(1, 8),
        depth_hidden=torch.randn(1, 2, 8),
        node_embeddings=torch.randn(1, 5, 8),
        parent_indices=seed.parent_indices,
        depths=seed.depths,
        valid_mask=seed.valid_mask,
        seed_edge_logprobs=seed.seed_edge_logprobs,
        seed_ranks=seed.seed_ranks,
        ancestor_mask=seed.ancestor_mask,
    )
    expected_edges = torch.full((5,), float("-inf"))
    expected_edges[0] = 0
    for parent_idx in (0, 1):
        children = torch.nonzero(
            seed.parent_indices[0] == parent_idx,
            as_tuple=False,
        ).flatten()
        children = children[children != 0]
        logits = (
            seed.seed_edge_logprobs[0, children]
            + output.raw_edge_residuals[0, children]
        )
        expected_edges[children] = (
            output.candidate_mass[0, parent_idx].log()
            + logits
            - torch.logsumexp(logits, dim=0)
        )
    torch.testing.assert_close(
        output.refined_edge_logprobs[0],
        expected_edges,
        atol=2e-6,
        rtol=2e-6,
    )
    expected_paths = expected_edges.clone()
    for node_idx in range(1, 5):
        expected_paths[node_idx] += expected_paths[
            seed.parent_indices[0, node_idx]
        ]
    torch.testing.assert_close(
        output.refined_path_logprobs[0],
        expected_paths,
        atol=2e-6,
        rtol=2e-6,
    )


def test_patr_compact_sibling_layout_matches_dense_reference():
    torch.manual_seed(4)
    base_head = _make_head()
    head = DFlashTreePostHead(
        replace(
            base_head.config,
            use_compact_sibling_layout=True,
        )
    )
    head.load_state_dict(base_head.state_dict(), strict=True)
    with torch.no_grad():
        head.edge_scale.normal_(std=0.1)
        head.scalar_mlp[-1].weight.normal_(std=0.1)
        head.mass_head.weight.normal_(std=0.1)
    seed = pack_seed_tree_batch(
        [_make_seed_tree(), _make_seed_tree()],
        seed_budget=7,
        max_depth=2,
    )
    captured: dict[str, torch.Tensor] = {}
    handle = head.candidate_norm.register_forward_hook(
        lambda _module, _inputs, result: captured.setdefault(
            "candidate_states",
            result,
        )
    )
    output = head(
        root_hidden=torch.randn(2, 8),
        depth_hidden=torch.randn(2, 2, 8),
        node_embeddings=torch.randn(2, 7, 8),
        parent_indices=seed.parent_indices,
        depths=seed.depths,
        valid_mask=seed.valid_mask,
        seed_edge_logprobs=seed.seed_edge_logprobs,
        seed_ranks=seed.seed_ranks,
        ancestor_mask=seed.ancestor_mask,
    )
    handle.remove()

    safe_parents = seed.parent_indices.clamp_min(0)
    child_valid = seed.valid_mask.clone()
    child_valid[:, 0] = False
    dense_parent_mask = F.one_hot(
        safe_parents,
        num_classes=seed.token_ids.shape[1],
    ).transpose(1, 2).bool()
    dense_parent_mask &= child_valid[:, None, :]
    sibling_count = dense_parent_mask.sum(dim=-1)
    grouped_seed_lp = seed.seed_edge_logprobs[:, None, :].masked_fill(
        ~dense_parent_mask,
        float("-inf"),
    )
    top_values = torch.topk(grouped_seed_lp, k=2, dim=-1).values
    sibling_gap = torch.where(
        sibling_count > 1,
        top_values[..., 0] - top_values[..., 1],
        torch.zeros_like(top_values[..., 0]),
    )
    log_rho0 = torch.logsumexp(grouped_seed_lp, dim=-1)
    log_rho0 = torch.minimum(
        log_rho0,
        log_rho0.new_tensor(-torch.finfo(torch.float32).eps),
    )
    mass_features = torch.stack(
        (
            sibling_gap,
            seed.depths.float() / head.config.max_depth,
            sibling_count.float() / head.config.tree_width,
        ),
        dim=-1,
    ).to(dtype=captured["candidate_states"].dtype)
    mass_delta = head.mass_head(
        torch.cat((captured["candidate_states"], mass_features), dim=-1)
    ).squeeze(-1).float()
    log_mass_numerator = log_rho0 + mass_delta
    split = -0.6931471805599453
    log_complement = torch.where(
        log_rho0 < split,
        torch.log1p(-torch.exp(log_rho0)),
        torch.log(-torch.expm1(log_rho0)),
    )
    log_mass = log_mass_numerator - torch.logaddexp(
        log_complement,
        log_mass_numerator,
    )
    expected_mass = torch.where(
        sibling_count > 0,
        torch.exp(log_mass),
        torch.zeros_like(log_mass),
    )
    expected_mass[:, 0] = torch.where(
        sibling_count[:, 0] > 0,
        expected_mass[:, 0],
        torch.ones_like(expected_mass[:, 0]),
    )
    torch.testing.assert_close(output.candidate_mass, expected_mass)

    log_within = torch.where(
        child_valid,
        seed.seed_edge_logprobs,
        torch.zeros_like(seed.seed_edge_logprobs),
    ) + output.raw_edge_residuals
    grouped_log_within = log_within[:, None, :].masked_fill(
        ~dense_parent_mask,
        float("-inf"),
    )
    normalizer = torch.logsumexp(grouped_log_within, dim=-1)
    expected_edges = (
        torch.gather(log_mass, 1, safe_parents)
        + log_within
        - torch.gather(normalizer, 1, safe_parents)
    )
    expected_edges = torch.where(
        child_valid,
        expected_edges,
        torch.full_like(expected_edges, float("-inf")),
    )
    expected_edges[:, 0] = 0
    torch.testing.assert_close(output.refined_edge_logprobs, expected_edges)


def test_patr_compact_sibling_backend_matches_original_dense_backend():
    torch.manual_seed(12)
    dense_head = _make_head()
    compact_head = DFlashTreePostHead(
        replace(
            dense_head.config,
            use_compact_sibling_layout=True,
        )
    )
    compact_head.load_state_dict(dense_head.state_dict(), strict=True)
    seed = pack_seed_tree_batch(
        [_make_seed_tree(), _make_seed_tree()],
        seed_budget=7,
        max_depth=2,
    )
    kwargs = dict(
        root_hidden=torch.randn(2, 8),
        depth_hidden=torch.randn(2, 2, 8),
        node_embeddings=torch.randn(2, 7, 8),
        parent_indices=seed.parent_indices,
        depths=seed.depths,
        valid_mask=seed.valid_mask,
        seed_edge_logprobs=seed.seed_edge_logprobs,
        seed_ranks=seed.seed_ranks,
        ancestor_mask=seed.ancestor_mask,
    )
    compact = compact_head(**kwargs)
    dense = dense_head(**kwargs)
    torch.testing.assert_close(compact.candidate_mass, dense.candidate_mass)
    torch.testing.assert_close(
        compact.refined_edge_logprobs,
        dense.refined_edge_logprobs,
    )
    torch.testing.assert_close(
        compact.refined_path_logprobs,
        dense.refined_path_logprobs,
    )


def test_patr_runtime_buffer_preserves_checkpoint_contract():
    head = _make_head()
    assert "_negative_float32_eps" not in head.state_dict()
    clone = _make_head()
    clone.load_state_dict(head.state_dict(), strict=True)


def test_batched_tree_builder_matches_request_local_reference():
    torch.manual_seed(5)
    logits = torch.randn(2, 3, 11)
    topk_logprobs, topk_tokens = torch.topk(
        torch.log_softmax(logits, dim=-1),
        k=2,
        dim=-1,
    )
    roots = torch.tensor([50, 60])
    expected = [
        build_tree_from_topk(
            int(roots[req_idx]),
            topk_tokens[req_idx],
            topk_logprobs[req_idx],
            budget=7,
            device=torch.device("cpu"),
            depth_first=False,
        )
        for req_idx in range(2)
    ]
    actual = build_trees_from_topk(
        roots,
        topk_tokens,
        topk_logprobs,
        budget=7,
        device=torch.device("cpu"),
        depth_first=False,
    )
    for reference, optimized in zip(expected, actual):
        torch.testing.assert_close(optimized.token_ids, reference.token_ids)
        torch.testing.assert_close(
            optimized.parent_indices,
            reference.parent_indices,
        )
        torch.testing.assert_close(optimized.depth, reference.depth)
        torch.testing.assert_close(
            optimized.seed_edge_logprobs,
            reference.seed_edge_logprobs,
        )
        torch.testing.assert_close(optimized.seed_ranks, reference.seed_ranks)
