# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark dense and sparse PATR ancestor attention backends."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import load_file

from vllm.model_executor.models.dflash_tree_post_head import (
    DFlashTreePostHead,
    TreePostHeadConfig,
)
from vllm.v1.spec_decode.dflash_tree import (
    build_tree_from_topk,
    pack_seed_tree_batch,
    select_prefix_closed_subtrees,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--compile",
        choices=("none", "blocks", "full", "graph"),
        default="none",
    )
    return parser.parse_args()


def load_heads(checkpoint: Path, device: torch.device):
    config_data = json.loads((checkpoint / "config.json").read_text())
    reference_config = TreePostHeadConfig(
        input_hidden_size=config_data["input_hidden_size"],
        max_depth=config_data["max_depth"],
        tree_width=config_data["tree_width"],
        hidden_size=config_data["hidden_size"],
        num_layers=config_data["num_layers"],
        num_heads=config_data["num_heads"],
        num_kv_heads=config_data["num_kv_heads"],
        intermediate_size=config_data["intermediate_size"],
        scorer_size=config_data["scorer_size"],
        rms_norm_eps=config_data["rms_norm_eps"],
        use_sparse_ancestor_attention=False,
        use_compact_sibling_layout=False,
    )
    weights = load_file(checkpoint / "model.safetensors")
    reference = DFlashTreePostHead(reference_config)
    reference.load_state_dict(weights, strict=True)
    optimized = DFlashTreePostHead(
        replace(reference_config, use_compact_sibling_layout=True)
    )
    optimized.load_state_dict(weights, strict=True)
    sparse = DFlashTreePostHead(
        replace(
            reference_config,
            use_sparse_ancestor_attention=True,
            use_compact_sibling_layout=True,
        )
    )
    sparse.load_state_dict(weights, strict=True)
    return (
        reference.to(device=device, dtype=torch.bfloat16).eval(),
        optimized.to(device=device, dtype=torch.bfloat16).eval(),
        sparse.to(device=device, dtype=torch.bfloat16).eval(),
        reference_config,
    )


def make_inputs(config, batch_size: int, device: torch.device):
    torch.manual_seed(17)
    topk_logits = torch.randn(config.max_depth, config.tree_width)
    topk_logprobs, order = torch.sort(
        torch.log_softmax(topk_logits, dim=-1),
        dim=-1,
        descending=True,
    )
    token_ids = torch.arange(
        config.max_depth * config.tree_width,
    ).view(config.max_depth, config.tree_width)
    topk_tokens = torch.gather(token_ids, 1, order) + 100
    tree = build_tree_from_topk(
        99,
        topk_tokens,
        topk_logprobs,
        budget=255,
        device=device,
        depth_first=False,
    )
    seed = pack_seed_tree_batch(
        [tree] * batch_size,
        seed_budget=255,
        max_depth=config.max_depth,
        build_dense_ancestor_mask=True,
        build_sparse_ancestor_indices=True,
    )
    dtype = torch.bfloat16
    return seed, {
        "root_hidden": torch.randn(
            batch_size,
            config.input_hidden_size,
            device=device,
            dtype=dtype,
        ),
        "depth_hidden": torch.randn(
            batch_size,
            config.max_depth,
            config.input_hidden_size,
            device=device,
            dtype=dtype,
        ),
        "node_embeddings": torch.randn(
            batch_size,
            255,
            config.input_hidden_size,
            device=device,
            dtype=dtype,
        ),
        "parent_indices": seed.parent_indices,
        "depths": seed.depths,
        "valid_mask": seed.valid_mask,
        "seed_edge_logprobs": seed.seed_edge_logprobs,
        "seed_ranks": seed.seed_ranks,
    }


@torch.inference_mode()
def benchmark(head, kwargs, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        head(**kwargs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        head(**kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


class GraphedPATR:
    """CUDA Graph replay wrapper used to validate exact eager-kernel reuse."""

    def __init__(self, head, kwargs):
        self.head = head
        self.static_inputs = {
            name: value.clone() for name, value in kwargs.items()
        }
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.static_output = self.head(**self.static_inputs)
        stream.synchronize()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.head(**self.static_inputs)

    def __call__(self, **kwargs):
        for name, value in kwargs.items():
            self.static_inputs[name].copy_(value)
        self.graph.replay()
        return self.static_output


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    reference, optimized, sparse, config = load_heads(args.checkpoint, device)
    if args.compile == "full":
        optimized = torch.compile(reference, fullgraph=True)
    elif args.compile == "blocks":
        for block in optimized.blocks:
            block.forward = torch.compile(block.forward, fullgraph=True)
    seed, common = make_inputs(config, args.batch_size, device)
    dense_kwargs = dict(common, ancestor_mask=seed.ancestor_mask)
    sparse_kwargs = dict(
        common,
        ancestor_indices=seed.ancestor_indices,
        ancestor_valid_mask=seed.ancestor_valid_mask,
    )
    if args.compile == "graph":
        optimized = GraphedPATR(reference, dense_kwargs)
    with torch.inference_mode():
        reference_output = reference(**dense_kwargs)
        optimized_output = optimized(**dense_kwargs)
        sparse_output = sparse(**sparse_kwargs)
    compact_path_error = (
        reference_output.refined_path_logprobs
        - optimized_output.refined_path_logprobs
    ).abs()[seed.valid_mask].max().item()
    sparse_path_error = (
        reference_output.refined_path_logprobs
        - sparse_output.refined_path_logprobs
    ).abs()[seed.valid_mask].max().item()
    reference_trees = select_prefix_closed_subtrees(
        seed,
        reference_output.refined_path_logprobs,
        final_budget=64,
    )
    optimized_trees = select_prefix_closed_subtrees(
        seed,
        optimized_output.refined_path_logprobs,
        final_budget=64,
    )
    sparse_trees = select_prefix_closed_subtrees(
        seed,
        sparse_output.refined_path_logprobs,
        final_budget=64,
    )
    compact_trees_equal = all(
        torch.equal(left.token_ids, right.token_ids)
        and torch.equal(left.parent_indices, right.parent_indices)
        and torch.equal(left.depth, right.depth)
        for left, right in zip(reference_trees, optimized_trees)
    )
    sparse_trees_equal = all(
        torch.equal(left.token_ids, right.token_ids)
        and torch.equal(left.parent_indices, right.parent_indices)
        and torch.equal(left.depth, right.depth)
        for left, right in zip(reference_trees, sparse_trees)
    )
    reference_ms = benchmark(
        reference, dense_kwargs, args.warmup, args.iterations,
    )
    optimized_ms = benchmark(
        optimized, dense_kwargs, args.warmup, args.iterations,
    )
    sparse_ms = benchmark(sparse, sparse_kwargs, args.warmup, args.iterations)
    print(f"batch_size={args.batch_size}")
    print(f"reference_ms={reference_ms:.4f}")
    print(f"compact_sibling_ms={optimized_ms:.4f}")
    print(f"sparse_ms={sparse_ms:.4f}")
    print(f"compact_speedup={reference_ms / optimized_ms:.3f}x")
    print(f"sparse_speedup={reference_ms / sparse_ms:.3f}x")
    print(f"compact_path_error={compact_path_error:.8f}")
    print(f"sparse_path_error={sparse_path_error:.8f}")
    print(f"compact_trees_equal={compact_trees_equal}")
    print(f"sparse_trees_equal={sparse_trees_equal}")
    if not compact_trees_equal:
        raise SystemExit("Compact sibling layout changed the verify tree.")
    if not sparse_trees_equal:
        raise SystemExit("Sparse attention changed the selected verify tree.")


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"wall_seconds={time.perf_counter() - started:.3f}")
