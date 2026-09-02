# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch

from vllm.model_executor.models.dflash2_beam_head import DFlash2BeamPathHead
from vllm.v1.spec_decode.dflash2_beam import (
    score_dflash2_lattice,
    select_lattice_beams,
    walk_dflash2_lattice,
)


def test_lattice_score_matches_pairwise_formula():
    predecessor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    successor = torch.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]])
    candidates = torch.tensor([[[1, 2], [0, 2]]])
    unary = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
    hidden = torch.tensor([[[1.0, 1.0], [2.0, 1.0]]])
    weight = torch.eye(2)
    scores = score_dflash2_lattice(
        predecessor_table=predecessor,
        successor_table=successor,
        hidden_projection_weight=weight,
        candidate_ids=candidates,
        unary_logits=unary,
        hidden_states=hidden,
        anchor_token_ids=torch.tensor([0]),
    )
    expected = unary[0, 1, 1] + torch.dot(
        predecessor[0] * hidden[0, 1], successor[2]
    )
    torch.testing.assert_close(scores[0, 1, 0, 1], expected)


def test_beam_matches_exhaustive_top_paths():
    torch.manual_seed(3)
    batch, depth, top_k, beam = 1, 4, 3, 5
    candidates = torch.arange(depth * top_k).view(batch, depth, top_k)
    scores = torch.randn(batch, depth, top_k, top_k)
    output = select_lattice_beams(
        candidates, scores, beam_size=beam, normalize_edges=False, include_greedy=False
    )
    exhaustive = []
    for path in itertools.product(range(top_k), repeat=depth):
        total = scores[0, 0, 0, path[0]]
        for position in range(1, depth):
            total = total + scores[0, position, path[position - 1], path[position]]
        exhaustive.append((float(total), path))
    expected = sorted(exhaustive, reverse=True)[:beam]
    assert output.candidate_ranks[0].tolist() == [list(path) for _, path in expected]
    torch.testing.assert_close(
        output.path_scores[0], torch.tensor([score for score, _ in expected])
    )


def test_beam_protects_production_greedy_walk():
    torch.manual_seed(7)
    candidates = torch.arange(4 * 3).view(1, 4, 3)
    scores = torch.randn(1, 4, 3, 3)
    greedy = walk_dflash2_lattice(candidates, scores)
    beams = select_lattice_beams(candidates, scores, beam_size=2, include_greedy=True)
    assert ((beams.candidate_ranks == greedy.candidate_ranks).all(dim=-1)).any()


def test_sixteen_head_path_reranker_shapes():
    head = DFlash2BeamPathHead(
        input_hidden_size=32,
        hidden_size=64,
        num_heads=16,
        num_layers=1,
    )
    output = head(
        depth_hidden=torch.randn(2, 7, 32),
        token_embeddings=torch.randn(2, 16, 7, 32),
        candidate_ranks=torch.randint(0, 16, (2, 16, 7)),
        numeric_features=torch.randn(2, 16, 7, 5),
    )
    assert output.conditional_survival.shape == (2, 16, 7)
    assert output.expected_accept_length.shape == (2, 16)
    assert output.path_delta.shape == (2, 16)
    assert output.override_logit.shape == (2, 16)
