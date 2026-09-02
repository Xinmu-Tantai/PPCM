#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

DEFAULT_TARGET_MODEL = "/data/models/Qwen3-8B"
DEFAULT_DRAFT_MODEL = "/data/models/Qwen3-8B-DFlash-b16"

DEFAULT_PROMPTS = [
    "Write a short explanation of speculative decoding.",
    "Summarize how KV cache is used in transformer inference.",
    "Give three tips for profiling CUDA workloads.",
    "Explain why acceptance rate matters for DFlash.",
]

CODING_PROMPTS = [
    (
        "Implement a Python function `binary_search(arr, target)` for a sorted "
        "list of integers that returns the index of target or -1."
    ),
    (
        "Implement a Python class `LRUCache` with methods `get(key)` and "
        "`put(key, value)` using O(1) average-time operations."
    ),
    (
        "Implement Python function `merge_intervals(intervals)` that merges "
        "overlapping intervals and returns a sorted merged list."
    ),
    (
        "Implement Python function `dijkstra(n, edges, src)` that returns "
        "shortest distances from src in a weighted graph with nonnegative weights."
    ),
]


def load_dataset_prompt_bank(prompt_set: str) -> list[str]:
    from datasets import load_dataset
    if prompt_set == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = (
            "{question}\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}."
        )
        return [prompt_fmt.format(**row) for row in dataset]
    if prompt_set == "math-500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = (
            "{problem}\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}."
        )
        return [prompt_fmt.format(**row) for row in dataset]
    if prompt_set == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = (
            "Write a solution to the following problem and make sure that it "
            "passes the tests:\n```python\n{prompt}\n```"
        )
        return [prompt_fmt.format(**row) for row in dataset]
    raise ValueError(f"Unknown dataset-backed prompt set: {prompt_set}")


def get_prompt_bank(prompt_set: str) -> list[str]:
    if prompt_set == "example-mix":
        return DEFAULT_PROMPTS
    if prompt_set == "example-coding":
        return CODING_PROMPTS
    if prompt_set in {"gsm8k", "humaneval", "math-500"}:
        return load_dataset_prompt_bank(prompt_set)
    raise ValueError(f"Unknown prompt set: {prompt_set}")


def build_prompt_batches(prompts: list[str], batch_size: int) -> list[list[str]]:
    if batch_size < 1:
        raise ValueError(f"Invalid batch size: {batch_size}. Expected batch size >= 1")
    return [prompts[start : start + batch_size] for start in range(0, len(prompts), batch_size)]


def _is_step3p7_model_path(model_path: str) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False
    try:
        with config_path.open() as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("model_type") == "step3p7"


def tokenizer_load_kwargs(model_path: str) -> dict[str, Any]:
    if _is_step3p7_model_path(model_path):
        return {"fix_mistral_regex": True}
    return {}


def _apply_step3p7_regen_template(tokenizer, prompt: str) -> str:
    """Render the Step-3.7 reasoning-low prompt with no-thinking prefill.

    Use the model's native chat template for BOS, role tokens, and
    ``Reasoning: low`` injection, then close the empty thinking bucket so raw
    offline generation starts on answer tokens.
    """
    templated_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    think_prefix = "<think>\n"
    if not templated_prompt.endswith(think_prefix):
        raise ValueError(
            "Unexpected Step-3.7 chat template: generation prompt does not end "
            f"with {think_prefix!r}"
        )
    return templated_prompt + "\n</think>\n\n"


def apply_chat_template(
    tokenizer,
    prompts: list[str],
    model_path: str,
) -> list[str]:
    is_step3p7 = _is_step3p7_model_path(model_path)
    templated_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        if is_step3p7:
            templated_prompt = _apply_step3p7_regen_template(tokenizer, prompt)
            templated_prompts.append(templated_prompt)
            continue
        try:
            templated_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            templated_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        templated_prompts.append(templated_prompt)
    return templated_prompts


def get_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["ar", "dflash"]
    return [mode]


def collect_spec_decode_counters(metrics) -> dict[str, object]:
    counters: dict[str, object] = {
        "num_drafts": 0.0,
        "num_draft_tokens": 0.0,
        "num_accepted_tokens": 0.0,
        "num_tree_drafts": 0.0,
        "num_tree_nodes": 0.0,
        "accepted_per_pos": [],
        "tree_nodes_per_depth": [],
    }
    for metric in metrics:
        name = getattr(metric, "name", "")
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            values = getattr(metric, "values", None)
            if values is not None:
                counters["accepted_per_pos"] = [float(v) for v in values]
            continue
        if name == "vllm:spec_decode_num_tree_nodes_per_depth":
            values = getattr(metric, "values", None)
            if values is not None:
                counters["tree_nodes_per_depth"] = [float(v) for v in values]
            continue
        value = getattr(metric, "value", None)
        if value is None:
            continue
        if name == "vllm:spec_decode_num_drafts":
            counters["num_drafts"] = float(counters["num_drafts"]) + float(value)
        elif name == "vllm:spec_decode_num_draft_tokens":
            counters["num_draft_tokens"] = float(counters["num_draft_tokens"]) + float(value)
        elif name == "vllm:spec_decode_num_accepted_tokens":
            counters["num_accepted_tokens"] = float(counters["num_accepted_tokens"]) + float(value)
        elif name == "vllm:spec_decode_num_tree_drafts":
            counters["num_tree_drafts"] = float(counters["num_tree_drafts"]) + float(value)
        elif name == "vllm:spec_decode_num_tree_nodes":
            counters["num_tree_nodes"] = float(counters["num_tree_nodes"]) + float(value)
    return counters


def diff_counters(after: dict[str, object], before: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for k in after:
        a, b = after.get(k, 0.0), before.get(k, 0.0)
        if isinstance(a, list) and isinstance(b, list):
            maxlen = max(len(a), len(b))
            result[k] = [
                (a[i] if i < len(a) else 0.0) - (b[i] if i < len(b) else 0.0)
                for i in range(maxlen)
            ]
        else:
            result[k] = float(a) - float(b)  # type: ignore[arg-type]
    return result


def _duration_to_seconds(value: float, unit: str) -> float:
    if unit == "s":
        return value
    if unit == "ms":
        return value / 1000.0
    if unit == "us":
        return value / 1_000_000.0
    raise ValueError(f"Unsupported duration unit: {unit}")


def collect_execute_context_cuda_seconds(run_output_dir: Path) -> dict[str, float]:
    """Aggregate execute_context self-CUDA time by phase from profiler text files."""
    totals = {"prefill_cuda_s": 0.0, "decode_cuda_s": 0.0, "mixed_cuda_s": 0.0}
    files = sorted(run_output_dir.glob("profiler_out_*.txt"))
    if not files:
        return totals

    name_pattern = re.compile(
        r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)"
    )
    # Last columns are: self_cuda, self_cuda%, cuda_total, cuda_avg, calls.
    # Capture self_cuda right before the percentage column.
    self_cuda_pattern = re.compile(
        r"\s([0-9.]+)(us|ms|s)\s+[0-9.]+%\s+[0-9.]+(?:us|ms|s)\s+[0-9.]+(?:us|ms|s)\s+\d+\s*$"
    )

    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            name_match = name_pattern.search(line)
            if not name_match:
                continue
            cuda_match = self_cuda_pattern.search(line)
            if not cuda_match:
                continue

            ctx_reqs, ctx_tokens, gen_reqs, gen_tokens = map(int, name_match.groups())
            self_cuda_s = _duration_to_seconds(
                float(cuda_match.group(1)),
                cuda_match.group(2),
            )

            if gen_reqs == 0 and gen_tokens == 0 and (ctx_reqs > 0 or ctx_tokens > 0):
                totals["prefill_cuda_s"] += self_cuda_s
            elif ctx_reqs == 0 and ctx_tokens == 0 and (gen_reqs > 0 or gen_tokens > 0):
                totals["decode_cuda_s"] += self_cuda_s
            else:
                totals["mixed_cuda_s"] += self_cuda_s

    return totals


def _parse_profiler_table_duration(value: str, unit: str) -> float:
    return _duration_to_seconds(float(value), unit)


def collect_profiler_table_rows(
    run_output_dir: Path,
    names: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Collect CPU/CUDA totals from torch profiler table rows.

    If names is provided, only those rows are returned. Otherwise every parsed
    row is returned. Self totals are useful for grouping without double-counting
    nested ranges; total columns are useful for named high-level ranges.
    """
    rows: dict[str, dict[str, float]] = {}
    if names is not None:
        rows = {
            name: {
                "self_cpu_s": 0.0,
                "cpu_total_s": 0.0,
                "self_cuda_s": 0.0,
                "cuda_total_s": 0.0,
                "calls": 0.0,
            }
            for name in names
        }
    files = sorted(run_output_dir.glob("profiler_out_*.txt"))
    if not files:
        return rows

    row_pattern = re.compile(
        r"^\s*(?P<name>.*?)\s+"
        r"(?P<self_cpu_pct>[0-9.]+)%\s+"
        r"(?P<self_cpu>[0-9.]+)(?P<self_cpu_unit>us|ms|s)\s+"
        r"(?P<cpu_total_pct>[0-9.]+)%\s+"
        r"(?P<cpu_total>[0-9.]+)(?P<cpu_total_unit>us|ms|s)\s+"
        r"(?P<cpu_avg>[0-9.]+)(?P<cpu_avg_unit>us|ms|s)\s+"
        r"(?P<self_cuda>[0-9.]+)(?P<self_cuda_unit>us|ms|s)\s+"
        r"(?P<self_cuda_pct>[0-9.]+)%\s+"
        r"(?P<cuda_total>[0-9.]+)(?P<cuda_total_unit>us|ms|s)\s+"
        r"(?P<cuda_avg>[0-9.]+)(?P<cuda_avg_unit>us|ms|s)\s+"
        r"(?P<calls>\d+)\s*$"
    )

    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = row_pattern.match(line)
            if not match:
                continue
            name = match.group("name").strip()
            if names is not None and name not in rows:
                continue
            if name not in rows:
                rows[name] = {
                    "self_cpu_s": 0.0,
                    "cpu_total_s": 0.0,
                    "self_cuda_s": 0.0,
                    "cuda_total_s": 0.0,
                    "calls": 0.0,
                }
            row = rows[name]
            row["self_cpu_s"] += _parse_profiler_table_duration(
                match.group("self_cpu"), match.group("self_cpu_unit")
            )
            row["cpu_total_s"] += _parse_profiler_table_duration(
                match.group("cpu_total"), match.group("cpu_total_unit")
            )
            row["self_cuda_s"] += _parse_profiler_table_duration(
                match.group("self_cuda"), match.group("self_cuda_unit")
            )
            row["cuda_total_s"] += _parse_profiler_table_duration(
                match.group("cuda_total"), match.group("cuda_total_unit")
            )
            row["calls"] += float(match.group("calls"))

    return rows


def collect_profiler_named_ranges(
    run_output_dir: Path,
    names: set[str],
) -> dict[str, dict[str, float]]:
    """Collect CPU/CUDA totals for selected torch profiler table rows."""
    return collect_profiler_table_rows(run_output_dir, names)


def write_dflash_residual_grouped_report(
    run_output_dir: Path,
    *,
    elapsed_s: float,
    num_drafts: float,
    residual_s: float,
) -> dict[str, dict[str, float]]:
    """Group non-DFlash profiler rows to explain residual wall time.

    This report uses profiler self time so categories are less prone to
    double-counting than high-level total ranges. The percentages are relative
    to wall time for readability, but CUDA self time and CPU self time can
    overlap asynchronously.
    """
    rows = collect_profiler_table_rows(run_output_dir)
    denom = num_drafts if num_drafts > 0 else 1.0

    groups: dict[str, dict[str, object]] = {
        "cuda_graph_replay": {
            "patterns": ("cudaGraph", "cudaStreamIsCapturing"),
            "rows": [],
        },
        "host_device_transfers": {
            "patterns": ("Memcpy", "memcpy", "copy_"),
            "rows": [],
        },
        "small_tensor_metadata_ops": {
            "patterns": (
                "aten::index",
                "aten::_index_put_impl_",
                "aten::cat",
                "aten::_unique",
                "aten::nonzero",
                "aten::sort",
                "aten::fill_",
                "aten::add",
                "aten::sub",
                "aten::div",
                "aten::remainder",
                "aten::bitwise_and",
                "_compute_slot_mapping_kernel",
            ),
            "rows": [],
        },
        "host_sync_scalar_ops": {
            "patterns": ("aten::_local_scalar_dense", "DtoH"),
            "rows": [],
        },
        "sampling_topk_softmax": {
            "patterns": (
                "aten::topk",
                "aten::_log_softmax",
                "SoftMax",
                "mbtopk",
                "radixSort",
                "bitonicSort",
            ),
            "rows": [],
        },
        "attention_and_model_kernels": {
            "patterns": (
                "kernel_unified_attention",
                "unified_attention",
                "_vllm_fa3_C::fwd",
                "FlashAttn",
                "nvjet_tst",
                "aten::mm",
                "cublas",
                "rms_norm",
                "rotary_embedding",
                "triton_",
            ),
            "rows": [],
        },
        "kv_cache_ops": {
            "patterns": ("reshape_and_cache", "_C_cache_ops"),
            "rows": [],
        },
        "dflash_named_ranges": {
            "patterns": ("dflash_", "gpu_model_runner:"),
            "rows": [],
        },
        "uncategorized": {
            "patterns": (),
            "rows": [],
        },
    }

    def add_to_group(group_name: str, row_name: str, row: dict[str, float]) -> None:
        group_rows = groups[group_name]["rows"]
        assert isinstance(group_rows, list)
        group_rows.append((row_name, row))

    for row_name, row in rows.items():
        assigned = False
        for group_name, config in groups.items():
            if group_name == "uncategorized":
                continue
            patterns = config["patterns"]
            assert isinstance(patterns, tuple)
            if any(pattern in row_name for pattern in patterns):
                add_to_group(group_name, row_name, row)
                assigned = True
                break
        if not assigned:
            add_to_group("uncategorized", row_name, row)

    grouped_metrics: dict[str, dict[str, float]] = {}
    lines = [
        "# DFlash grouped residual report",
        "# Uses torch profiler self time for grouping; CPU and CUDA work may overlap.",
        f"elapsed_s={elapsed_s:.6f}",
        f"num_drafts={num_drafts:.0f}",
        f"residual_wall_estimate_s={residual_s:.6f}",
        "",
        "# Groups",
    ]

    for group_name, config in groups.items():
        group_rows = config["rows"]
        assert isinstance(group_rows, list)
        self_cpu_s = sum(row["self_cpu_s"] for _, row in group_rows)
        self_cuda_s = sum(row["self_cuda_s"] for _, row in group_rows)
        calls = sum(row["calls"] for _, row in group_rows)
        grouped_metrics[group_name] = {
            "self_cpu_s": self_cpu_s,
            "self_cuda_s": self_cuda_s,
            "calls": calls,
        }
        cpu_pct = 100.0 * self_cpu_s / elapsed_s if elapsed_s > 0 else 0.0
        cuda_pct = 100.0 * self_cuda_s / elapsed_s if elapsed_s > 0 else 0.0
        lines.append(
            f"{group_name}: self_cpu_s={self_cpu_s:.6f} "
            f"cpu_pct_wall={cpu_pct:.2f} "
            f"cpu_per_step_ms={1000.0 * self_cpu_s / denom:.3f} "
            f"self_cuda_s={self_cuda_s:.6f} "
            f"cuda_pct_wall={cuda_pct:.2f} "
            f"cuda_per_step_ms={1000.0 * self_cuda_s / denom:.3f} "
            f"calls={calls:.0f}"
        )
        top_rows = sorted(
            group_rows,
            key=lambda item: item[1]["self_cpu_s"] + item[1]["self_cuda_s"],
            reverse=True,
        )[:8]
        for row_name, row in top_rows:
            lines.append(
                f"  - {row_name}: calls={row['calls']:.0f} "
                f"self_cpu_s={row['self_cpu_s']:.6f} "
                f"self_cuda_s={row['self_cuda_s']:.6f} "
                f"cpu_total_s={row['cpu_total_s']:.6f} "
                f"cuda_total_s={row['cuda_total_s']:.6f}"
            )
    report_path = run_output_dir / "dflash_residual_grouped_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote DFlash grouped residual report to: {report_path}")
    return grouped_metrics


def build_dflash_pipeline_buckets(
    rows: dict[str, dict[str, float]],
    *,
    elapsed_s: float,
    phase_cuda: dict[str, float],
) -> list[dict[str, float | str]]:
    """Build wall-like DFlash pipeline buckets from named profiler ranges.

    Torch profiler CPU and CUDA totals are not a strict wall-time decomposition,
    so each named range contributes the larger of its CPU/CUDA totals. The
    buckets below avoid obvious parent/child double-counting and leave any
    remaining elapsed time as an explicit unassigned gap.
    """

    def row_wall_like(name: str) -> float:
        row = rows[name]
        return max(row["cpu_total_s"], row["cuda_total_s"])

    def row_calls(*names: str) -> float:
        return max((rows[name]["calls"] for name in names), default=0.0)

    def total(*names: str) -> float:
        return sum(row_wall_like(name) for name in names)

    target_verify_s = phase_cuda["decode_cuda_s"]
    if target_verify_s == 0.0:
        target_verify_s = row_wall_like("gpu_model_runner: forward")

    bucket_specs = [
        (
            "prefill",
            phase_cuda["prefill_cuda_s"],
            row_calls("gpu_model_runner: forward"),
        ),
        (
            "target_preprocess",
            row_wall_like("gpu_model_runner: preprocess"),
            row_calls("gpu_model_runner: preprocess"),
        ),
        (
            "target_verification_forward",
            target_verify_s,
            row_calls("gpu_model_runner: forward"),
        ),
        (
            "target_postprocess_sample",
            total("gpu_model_runner: postprocess", "gpu_model_runner: sample"),
            row_calls("gpu_model_runner: postprocess", "gpu_model_runner: sample"),
        ),
        (
            "draft_model_forward_logits",
            total("dflash_draft_forward", "dflash_draft_logits"),
            row_calls("dflash_draft_forward", "dflash_draft_logits"),
        ),
        (
            "draft_input_setup",
            total(
                "dflash_combine_hidden_states",
                "dflash_set_inputs_first_pass",
                "dflash_build_attn_metadata",
                "dflash_draft_cg_dispatch",
                "dflash_zero_padded_positions",
                "dflash_build_model_inputs",
                "dflash_context_kv_pair_metadata_sync",
            ),
            row_calls(
                "dflash_combine_hidden_states",
                "dflash_set_inputs_first_pass",
                "dflash_build_attn_metadata",
                "dflash_draft_cg_dispatch",
                "dflash_zero_padded_positions",
                "dflash_build_model_inputs",
                "dflash_context_kv_pair_metadata_sync",
            ),
        ),
        (
            "draft_context_kv_precompute",
            total(
                "dflash_context_kv_precompute",
                "dflash_context_kv_precompute_full",
                "dflash_context_kv_precompute_suffix",
            ),
            row_calls(
                "dflash_context_kv_precompute",
                "dflash_context_kv_precompute_full",
                "dflash_context_kv_precompute_suffix",
            ),
        ),
        (
            "tree_build_pack",
            total(
                "dflash_tree_prebuild_setup",
                "dflash_tree_entropy_setup",
                "dflash_tree_build",
                "dflash_tree_topk",
                "dflash_tree_root_token_sync",
                "dflash_tree_cpu_build",
                "dflash_patr_refine",
                "dflash_patr_select",
                "dflash_tree_refine",
                "dflash_tree_cg_adjust",
                "dflash_tree_cg_log_detail",
                "dflash_tree_spec_pack",
            ),
            row_calls(
                "dflash_tree_prebuild_setup",
                "dflash_tree_entropy_setup",
                "dflash_tree_build",
                "dflash_tree_topk",
                "dflash_tree_root_token_sync",
                "dflash_tree_cpu_build",
                "dflash_patr_refine",
                "dflash_patr_select",
                "dflash_tree_refine",
                "dflash_tree_cg_adjust",
                "dflash_tree_cg_log_detail",
                "dflash_tree_spec_pack",
            ),
        ),
        (
            "tree_accept_sampling",
            total(
                "dflash_tree_sample_prepare",
                "dflash_tree_accept",
                "dflash_tree_sample_pack",
            ),
            row_calls(
                "dflash_tree_sample_prepare",
                "dflash_tree_accept",
                "dflash_tree_sample_pack",
            ),
        ),
        (
            "commit_compaction",
            total(
                "dflash_tree_hidden_state_compaction",
                "dflash_tree_kv_commit",
                "dflash_tree_kv_commit_filter_identity",
                "dflash_tree_kv_commit_copy",
            ),
            row_calls(
                "dflash_tree_hidden_state_compaction",
                "dflash_tree_kv_commit",
                "dflash_tree_kv_commit_filter_identity",
                "dflash_tree_kv_commit_copy",
            ),
        ),
        (
            "debug_capture",
            total("dflash_runtime_debug_capture", "dflash_tree_builder_debug_capture"),
            row_calls("dflash_runtime_debug_capture", "dflash_tree_builder_debug_capture"),
        ),
        (
            "draft_runner_orchestration",
            row_wall_like("gpu_model_runner: draft"),
            row_calls("gpu_model_runner: draft"),
        ),
    ]

    buckets: list[dict[str, float | str]] = []
    visible_total_s = 0.0
    for name, seconds, calls in bucket_specs:
        visible_total_s += seconds
        buckets.append({"name": name, "seconds": seconds, "calls": calls})

    unassigned_s = elapsed_s - visible_total_s
    buckets.append(
        {
            "name": "unassigned_wall_gap",
            "seconds": max(0.0, unassigned_s),
            "calls": 0.0,
        }
    )
    if unassigned_s < 0.0:
        buckets.append(
            {
                "name": "overlap_overage",
                "seconds": -unassigned_s,
                "calls": 0.0,
            }
        )
    return buckets


def write_dflash_pipeline_bucket_report(
    run_output_dir: Path,
    *,
    elapsed_s: float,
    num_drafts: float,
    buckets: list[dict[str, float | str]],
) -> None:
    denom = num_drafts if num_drafts > 0 else 1.0
    lines = [
        "# DFlash pipeline bucket report",
        "# Bucket seconds use max(cpu_total_s, cuda_total_s) per named range.",
        "# Buckets are wall-like estimates; CPU/CUDA work may overlap.",
        f"elapsed_s={elapsed_s:.6f}",
        f"num_drafts={num_drafts:.0f}",
        "",
        "# Pipeline buckets",
    ]
    for bucket in buckets:
        name = str(bucket["name"])
        seconds = float(bucket["seconds"])
        calls = float(bucket["calls"])
        pct = 100.0 * seconds / elapsed_s if elapsed_s > 0 else 0.0
        per_step_ms = 1000.0 * seconds / denom
        lines.append(
            f"{name}: seconds={seconds:.6f} pct_wall={pct:.2f} "
            f"per_tree_step_ms={per_step_ms:.3f} calls={calls:.0f}"
        )
    report_path = run_output_dir / "dflash_pipeline_bucket_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote DFlash pipeline bucket report to: {report_path}")


def write_dflash_breakdown_report(
    run_output_dir: Path,
    *,
    elapsed_s: float,
    phase_cuda: dict[str, float],
    num_drafts: float,
) -> dict[str, float]:
    """Write a fine-grained DFlash timing report from torch profiler rows."""
    names = {
        "gpu_model_runner: draft",
        "gpu_model_runner: sample",
        "gpu_model_runner: forward",
        "gpu_model_runner: preprocess",
        "gpu_model_runner: postprocess",
        "dflash_combine_hidden_states",
        "dflash_propose_setup",
        "dflash_set_inputs_first_pass",
        "dflash_build_attn_metadata",
        "dflash_draft_cg_dispatch",
        "dflash_zero_padded_positions",
        "dflash_build_model_inputs",
        "dflash_context_kv_precompute",
        "dflash_context_kv_precompute_full",
        "dflash_context_kv_precompute_suffix",
        "dflash_context_kv_pair_metadata_sync",
        "dflash_draft_forward",
        "dflash_draft_logits",
        "dflash_runtime_debug_capture",
        "dflash_tree_prebuild_setup",
        "dflash_tree_entropy_setup",
        "dflash_tree_build",
        "dflash_tree_topk",
        "dflash_tree_root_token_sync",
        "dflash_tree_cpu_build",
        "dflash_tree_builder_debug_capture",
        "dflash_patr_refine",
        "dflash_patr_select",
        "dflash_tree_refine",
        "dflash_tree_cg_adjust",
        "dflash_tree_cg_log_detail",
        "dflash_tree_spec_pack",
        "dflash_tree_sample_prepare",
        "dflash_tree_accept",
        "dflash_tree_sample_pack",
        "dflash_tree_hidden_state_compaction",
        "dflash_tree_kv_commit",
        "dflash_tree_kv_commit_filter_identity",
        "dflash_tree_kv_commit_copy",
    }
    rows = collect_profiler_named_ranges(run_output_dir, names)

    def cpu_total(*row_names: str) -> float:
        return sum(rows[name]["cpu_total_s"] for name in row_names)

    def cuda_total(*row_names: str) -> float:
        return sum(rows[name]["cuda_total_s"] for name in row_names)

    target_verify_cuda_s = phase_cuda["decode_cuda_s"]
    if target_verify_cuda_s == 0.0:
        target_verify_cuda_s = rows["gpu_model_runner: forward"]["cuda_total_s"]
    draft_model_cuda_s = cuda_total("dflash_draft_forward", "dflash_draft_logits")
    draft_model_cpu_s = cpu_total("dflash_draft_forward", "dflash_draft_logits")
    tree_build_cpu_s = cpu_total("dflash_tree_build")
    tree_build_cuda_s = cuda_total("dflash_tree_build")
    cg_adjust_cpu_s = cpu_total("dflash_tree_cg_adjust")
    cg_adjust_cuda_s = cuda_total("dflash_tree_cg_adjust")
    sample_cpu_s = cpu_total(
        "dflash_tree_sample_prepare",
        "dflash_tree_accept",
        "dflash_tree_sample_pack",
    )
    sample_cuda_s = cuda_total(
        "dflash_tree_sample_prepare",
        "dflash_tree_accept",
        "dflash_tree_sample_pack",
    )
    kv_commit_cpu_s = cpu_total(
        "dflash_tree_hidden_state_compaction",
        "dflash_tree_kv_commit",
        "dflash_tree_kv_commit_copy",
    )
    kv_commit_cuda_s = cuda_total(
        "dflash_tree_hidden_state_compaction",
        "dflash_tree_kv_commit",
        "dflash_tree_kv_commit_copy",
    )

    known_wall_like_s = (
        phase_cuda["prefill_cuda_s"]
        + target_verify_cuda_s
        + draft_model_cuda_s
        + tree_build_cpu_s
        + cg_adjust_cpu_s
        + sample_cpu_s
        + kv_commit_cpu_s
    )
    residual_s = max(0.0, elapsed_s - known_wall_like_s)
    pipeline_buckets = build_dflash_pipeline_buckets(
        rows,
        elapsed_s=elapsed_s,
        phase_cuda=phase_cuda,
    )
    pipeline_unassigned_s = next(
        float(bucket["seconds"])
        for bucket in pipeline_buckets
        if bucket["name"] == "unassigned_wall_gap"
    )
    denom = num_drafts if num_drafts > 0 else 1.0

    metrics = {
        "elapsed_s": elapsed_s,
        "num_drafts": num_drafts,
        "prefill_cuda_s": phase_cuda["prefill_cuda_s"],
        "target_verification_cuda_s": target_verify_cuda_s,
        "draft_model_cpu_total_s": draft_model_cpu_s,
        "draft_model_cuda_total_s": draft_model_cuda_s,
        "tree_build_cpu_total_s": tree_build_cpu_s,
        "tree_build_cuda_total_s": tree_build_cuda_s,
        "cudagraph_adjust_cpu_total_s": cg_adjust_cpu_s,
        "cudagraph_adjust_cuda_total_s": cg_adjust_cuda_s,
        "sampling_cpu_total_s": sample_cpu_s,
        "sampling_cuda_total_s": sample_cuda_s,
        "kv_commit_cpu_total_s": kv_commit_cpu_s,
        "kv_commit_cuda_total_s": kv_commit_cuda_s,
        "residual_wall_estimate_s": residual_s,
        "pipeline_unassigned_wall_gap_s": pipeline_unassigned_s,
    }

    def fmt_seconds(key: str) -> str:
        val = metrics[key]
        pct = (100.0 * val / elapsed_s) if elapsed_s > 0 else 0.0
        per_step_ms = 1000.0 * val / denom
        return f"{key}={val:.6f} pct_wall={pct:.2f} per_tree_step_ms={per_step_ms:.3f}"

    lines = [
        "# DFlash fine-grained breakdown",
        "# CPU totals can overlap asynchronous CUDA work; residual is a wall-time estimate.",
        f"elapsed_s={elapsed_s:.6f}",
        f"num_drafts={num_drafts:.0f}",
        fmt_seconds("prefill_cuda_s"),
        fmt_seconds("target_verification_cuda_s"),
        fmt_seconds("draft_model_cpu_total_s"),
        fmt_seconds("draft_model_cuda_total_s"),
        fmt_seconds("tree_build_cpu_total_s"),
        fmt_seconds("tree_build_cuda_total_s"),
        fmt_seconds("cudagraph_adjust_cpu_total_s"),
        fmt_seconds("cudagraph_adjust_cuda_total_s"),
        fmt_seconds("sampling_cpu_total_s"),
        fmt_seconds("sampling_cuda_total_s"),
        fmt_seconds("kv_commit_cpu_total_s"),
        fmt_seconds("kv_commit_cuda_total_s"),
        fmt_seconds("residual_wall_estimate_s"),
        fmt_seconds("pipeline_unassigned_wall_gap_s"),
        "",
        "# Pipeline buckets",
    ]
    for bucket in pipeline_buckets:
        name = str(bucket["name"])
        seconds = float(bucket["seconds"])
        calls = float(bucket["calls"])
        pct = (100.0 * seconds / elapsed_s) if elapsed_s > 0 else 0.0
        per_step_ms = 1000.0 * seconds / denom
        lines.append(
            f"{name}: seconds={seconds:.6f} pct_wall={pct:.2f} "
            f"per_tree_step_ms={per_step_ms:.3f} calls={calls:.0f}"
        )
    lines.extend([
        "",
        "# Raw profiler rows",
    ])
    for name in sorted(rows):
        row = rows[name]
        lines.append(
            f"{name}: calls={row['calls']:.0f} "
            f"cpu_total_s={row['cpu_total_s']:.6f} "
            f"cuda_total_s={row['cuda_total_s']:.6f} "
            f"self_cpu_s={row['self_cpu_s']:.6f} "
            f"self_cuda_s={row['self_cuda_s']:.6f}"
        )
    report_path = run_output_dir / "dflash_breakdown_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote DFlash breakdown report to: {report_path}")
    write_dflash_residual_grouped_report(
        run_output_dir,
        elapsed_s=elapsed_s,
        num_drafts=num_drafts,
        residual_s=residual_s,
    )
    write_dflash_pipeline_bucket_report(
        run_output_dir,
        elapsed_s=elapsed_s,
        num_drafts=num_drafts,
        buckets=pipeline_buckets,
    )
    return metrics


def collect_dflash_tree_debug_records(llm: LLM) -> list[dict[str, Any]]:
    """Collect JSON-safe DFlash tree attention debug records from workers."""

    def _extract_tree_debug_records(worker):
        import torch  # noqa: PLC0415

        def _jsonify(val):
            if isinstance(val, torch.Tensor):
                t = val.detach().cpu()
                if t.ndim == 0:
                    return t.item()
                return t.tolist()
            if isinstance(val, dict):
                return {str(k): _jsonify(v) for k, v in val.items()}
            if isinstance(val, (list, tuple)):
                return [_jsonify(v) for v in val]
            if isinstance(val, (str, int, float, bool)) or val is None:
                return val
            return str(val)

        worker_obj = getattr(worker, "worker", worker)
        model_runner = getattr(worker_obj, "model_runner", None)
        if model_runner is None:
            return [{"error": "missing_model_runner"}]

        out_records: list[dict[str, Any]] = []
        sources = [
            ("target_model_runner", getattr(model_runner, "attn_groups", None), True),
            (
                "drafter",
                getattr(getattr(model_runner, "drafter", None), "draft_attn_groups", None),
                False,
            ),
        ]
        for owner_name, groups_root, indexed_builder in sources:
            if not isinstance(groups_root, list):
                continue
            for outer_idx, groups in enumerate(groups_root):
                iter_groups = (
                    groups if indexed_builder and isinstance(groups, list) else [groups]
                )
                if not isinstance(iter_groups, list):
                    continue
                for inner_idx, attn_group in enumerate(iter_groups):
                    try:
                        builder = (
                            attn_group.get_metadata_builder(0)
                            if indexed_builder
                            else attn_group.get_metadata_builder()
                        )
                    except Exception:
                        builder = None
                    if builder is None or not hasattr(
                        builder, "get_dflash_tree_debug_records"
                    ):
                        continue
                    try:
                        raw_records = list(
                            builder.get_dflash_tree_debug_records() or []
                        )
                    except Exception as e:
                        out_records.append(
                            {
                                "owner_name": owner_name,
                                "kv_cache_group_id": (
                                    int(outer_idx) if indexed_builder else None
                                ),
                                "attn_group_id": int(inner_idx),
                                "error": f"get_dflash_tree_debug_records_failed: {e}",
                            }
                        )
                        continue
                    for record in raw_records:
                        if not isinstance(record, dict):
                            continue
                        out_records.append(
                            {
                                "owner_name": owner_name,
                                "kv_cache_group_id": (
                                    int(outer_idx) if indexed_builder else None
                                ),
                                "attn_group_id": int(inner_idx),
                                **_jsonify(record),
                            }
                        )
                    clear_fn = getattr(builder, "clear_dflash_tree_debug_records", None)
                    if clear_fn is not None:
                        try:
                            clear_fn()
                        except Exception:
                            pass
        return out_records

    records: list[dict[str, Any]] = []
    try:
        worker_records = llm.collective_rpc(_extract_tree_debug_records)
    except Exception as e:
        print(f"WARNING: collective tree debug record extraction failed: {e}")
        worker_records = []
    for worker_record_set in worker_records or []:
        if isinstance(worker_record_set, list):
            records.extend(worker_record_set)

    if records:
        return records

    try:
        worker = llm.llm_engine.model_executor.driver_worker.worker
        direct_records = _extract_tree_debug_records(worker)
        if isinstance(direct_records, list):
            records.extend(direct_records)
    except Exception as e:
        print(f"WARNING: direct tree debug record extraction failed: {e}")
    return records


def write_dflash_tree_debug_records(llm: LLM, run_output_dir: Path) -> None:
    records = collect_dflash_tree_debug_records(llm)
    if not records:
        print("WARNING: DFlash tree debug records are empty")
        return

    debug_path = run_output_dir / "dflash_tree_debug_records.json"
    debug_path.write_text(json.dumps(records, indent=2))

    mapped_records = [
        record
        for record in records
        if record.get("logical_kv_num_mapped_reqs") not in (None, 0)
    ]
    layouts = sorted(
        {
            str(record.get("logical_kv_layout"))
            for record in records
            if record.get("logical_kv_layout") is not None
        }
    )
    sample_lens = [
        record.get("logical_kv_slot_lens")
        for record in mapped_records[:5]
    ]
    print(
        "[TREE_DEBUG] "
        f"records={len(records)} mapped_records={len(mapped_records)} "
        f"layouts={layouts} sample_logical_kv_slot_lens={sample_lens}"
    )
    print(f"Wrote DFlash tree debug records to {debug_path}")


def _jsonify_dflash_value(val: Any) -> Any:
    if isinstance(val, torch.Tensor):
        t = val.detach().cpu()
        if t.ndim == 0:
            return t.item()
        return t.tolist()
    if isinstance(val, dict):
        return {str(k): _jsonify_dflash_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_jsonify_dflash_value(v) for v in val]
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    return str(val)


def _raw_dflash_verify_bundles(llm: LLM) -> list[Any]:
    """Return in-memory verify bundles without converting the whole list first."""

    def _extract_raw(worker):
        worker_obj = getattr(worker, "worker", worker)
        model_runner = getattr(worker_obj, "model_runner", None)
        if model_runner is None:
            return [{"error": "missing_model_runner"}]
        bundles = getattr(model_runner, "_dflash_runtime_verify_bundles", None)
        return bundles if isinstance(bundles, list) else []

    try:
        worker = llm.llm_engine.model_executor.driver_worker.worker
        direct = _extract_raw(worker)
        if direct:
            return direct
    except Exception as e:
        print(f"WARNING: direct verify bundle access failed: {e}")
    try:
        worker_records = llm.collective_rpc(_extract_raw)
    except Exception as e:
        print(f"WARNING: collective verify bundle extraction failed: {e}")
        return []
    records: list[Any] = []
    for worker_record_set in worker_records or []:
        if isinstance(worker_record_set, list):
            records.extend(worker_record_set)
    return records


def collect_dflash_runtime_verify_bundles(llm: LLM) -> list[dict[str, Any]]:
    """Collect JSON-safe target verification bundles from model runners."""
    return [_jsonify_dflash_value(bundle) for bundle in _raw_dflash_verify_bundles(llm)]


def write_dflash_runtime_verify_bundles(llm: LLM, run_output_dir: Path) -> None:
    bundles = _raw_dflash_verify_bundles(llm)
    if not bundles:
        print("WARNING: DFlash runtime verify bundles are empty")
        return
    debug_path = run_output_dir / "dflash_runtime_verify_bundles.json"
    tmp_path = debug_path.with_suffix(".json.partial")
    indent = None if os.environ.get("DFLASH_OBS1_DIAG", "").strip() else 2
    total = len(bundles)
    print(f"Streaming {total} DFlash runtime verify bundles to {debug_path}", flush=True)
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write("[")
        for index, bundle in enumerate(bundles):
            if index:
                handle.write(",")
            json.dump(
                _jsonify_dflash_value(bundle),
                handle,
                ensure_ascii=False,
                indent=indent,
                separators=(",", ":"),
            )
            bundles[index] = None
            if (index + 1) % 20 == 0 or index + 1 == total:
                handle.flush()
                print(f"dump {index + 1}/{total} bundles", flush=True)
        handle.write("]")
        handle.flush()
    tmp_path.replace(debug_path)
    print(f"Wrote {total} DFlash runtime verify bundles to {debug_path}", flush=True)


def collect_dflash_tree_commit_debug_records(llm: LLM) -> list[dict[str, Any]]:
    """Collect JSON-safe DFlash KV commit counters from workers."""

    def _extract_commit_debug_records(worker):
        import torch  # noqa: PLC0415

        def _jsonify(val):
            if isinstance(val, torch.Tensor):
                t = val.detach().cpu()
                if t.ndim == 0:
                    return t.item()
                return t.tolist()
            if isinstance(val, dict):
                return {str(k): _jsonify(v) for k, v in val.items()}
            if isinstance(val, (list, tuple)):
                return [_jsonify(v) for v in val]
            if isinstance(val, (str, int, float, bool)) or val is None:
                return val
            return str(val)

        worker_obj = getattr(worker, "worker", worker)
        model_runner = getattr(worker_obj, "model_runner", None)
        if model_runner is None:
            return [{"error": "missing_model_runner"}]

        get_records = getattr(
            model_runner, "get_dflash_tree_commit_debug_records", None
        )
        if get_records is None:
            return [{"error": "missing_commit_debug_records_accessor"}]

        try:
            records = [_jsonify(record) for record in get_records() or []]
        except Exception as e:
            return [{"error": f"get_commit_debug_records_failed: {e}"}]

        clear_records = getattr(
            model_runner, "clear_dflash_tree_commit_debug_records", None
        )
        if clear_records is not None:
            try:
                clear_records()
            except Exception:
                pass
        return records

    records: list[dict[str, Any]] = []
    try:
        worker_records = llm.collective_rpc(_extract_commit_debug_records)
    except Exception as e:
        print(f"WARNING: collective commit debug extraction failed: {e}")
        worker_records = []
    for worker_record_set in worker_records or []:
        if isinstance(worker_record_set, list):
            records.extend(worker_record_set)

    if records:
        return records

    try:
        worker = llm.llm_engine.model_executor.driver_worker.worker
        direct_records = _extract_commit_debug_records(worker)
        if isinstance(direct_records, list):
            records.extend(direct_records)
    except Exception as e:
        print(f"WARNING: direct commit debug extraction failed: {e}")
    return records


def write_dflash_tree_commit_debug_records(llm: LLM, run_output_dir: Path) -> None:
    records = collect_dflash_tree_commit_debug_records(llm)
    if not records:
        print("WARNING: DFlash tree commit debug records are empty")
        return

    debug_path = run_output_dir / "dflash_tree_commit_debug_records.json"
    debug_path.write_text(json.dumps(records, indent=2))

    total_entries = sum(int(r.get("accepted_entries_total", 0)) for r in records)
    copied_entries = sum(int(r.get("accepted_entries_copied", 0)) for r in records)
    skipped_entries = sum(
        int(r.get("accepted_entries_skipped_identity", 0)) for r in records
    )
    copied_across_caches = sum(
        int(r.get("kv_entries_copied_across_caches", 0)) for r in records
    )
    skipped_across_caches = sum(
        int(r.get("kv_entries_skipped_across_caches", 0)) for r in records
    )
    avoidance_rate = skipped_entries / total_entries if total_entries else 0.0
    print(
        "[TREE_COMMIT_DEBUG] "
        f"records={len(records)} accepted_entries_total={total_entries} "
        f"copied={copied_entries} skipped_identity={skipped_entries} "
        f"copy_avoidance_rate={avoidance_rate:.6f} "
        f"kv_entries_copied_across_caches={copied_across_caches} "
        f"kv_entries_skipped_across_caches={skipped_across_caches}"
    )
    print(f"Wrote DFlash tree commit debug records to {debug_path}")


def enable_dflash_debug_artifacts(llm: LLM, max_records: int = 512) -> None:
    """Enable runner-side DFlash debug buffers before measured generation."""
    env_max = os.environ.get("DFLASH_DEBUG_MAX_RECORDS", "").strip()
    if env_max:
        max_records = int(env_max)

    def _enable(worker):
        worker_obj = getattr(worker, "worker", worker)
        model_runner = getattr(worker_obj, "model_runner", None)
        if model_runner is None:
            return "missing_model_runner"
        enable_fn = getattr(model_runner, "enable_dflash_debug_artifacts", None)
        if enable_fn is None:
            return "missing_enable_dflash_debug_artifacts"
        enable_fn(max_records=max_records)
        return "ok"

    results: list[Any] = []
    try:
        rpc_results = llm.collective_rpc(_enable)
        if isinstance(rpc_results, list):
            results.extend(rpc_results)
    except Exception as e:
        print(f"WARNING: collective debug artifact enable failed: {e}")

    if not results:
        try:
            worker = llm.llm_engine.model_executor.driver_worker.worker
            results.append(_enable(worker))
        except Exception as e:
            print(f"WARNING: direct debug artifact enable failed: {e}")

    print(f"[DFLASH_DEBUG] enabled runner debug artifacts: {results}")


def resolve_effective_block_size(args) -> int | None:
    if args.block_size is None:
        return None
    if args.block_size <= 1:
        raise ValueError("--block-size must be greater than 1.")
    return args.block_size


def resolve_effective_num_speculative_tokens(args) -> int:
    block_size = resolve_effective_block_size(args)
    if block_size is not None:
        return block_size - 1
    return args.num_speculative_tokens


def mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def validate_native_only_settings(args) -> None:
    return None


def get_compilation_config_for_cudagraph_mode(
    cudagraph_mode: str,
) -> dict[str, str] | None:
    if cudagraph_mode == "default":
        return None
    mode_map = {
        "none": "NONE",
        "full": "FULL",
        "full_decode_only": "FULL_DECODE_ONLY",
        "full_and_piecewise": "FULL_AND_PIECEWISE",
        "piecewise": "PIECEWISE",
    }
    return {"cudagraph_mode": mode_map[cudagraph_mode]}


def log_cuda_memory(prefix: str) -> None:
    if not torch.cuda.is_available():
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(
        f"[CLEANUP] {prefix}: "
        f"free={free_bytes / 1024**3:.2f} GiB "
        f"total={total_bytes / 1024**3:.2f} GiB"
    )


def cleanup_llm(llm: LLM | None, shutdown_timeout: float | None = None) -> None:
    """Release vLLM engine resources before constructing the next engine."""
    log_cuda_memory("before")

    if llm is not None:
        # Level 2 discards weights and KV cache from GPU before tearing down
        # the in-process engine.
        llm.sleep(level=2, mode="abort")
        llm.llm_engine.engine_core.shutdown(timeout=shutdown_timeout)
        llm.llm_engine = None  # type: ignore[assignment]

    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    log_cuda_memory("after")


def run_native_profile(
    *,
    args,
    mode: str,
    prompt_batches: list[list[str]],
    sampling_params: SamplingParams,
    tp_size: int,
    batch_size: int,
    effective_num_speculative_tokens: int,
    effective_max_num_seqs: int,
    profiler_config: dict[str, str],
    run_output_dir: Path,
    post_generation_hook: Callable[[Any], None] | None = None,
    pre_generation_hook: Callable[[Any], None] | None = None,
) -> dict[str, object]:
    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=effective_max_num_seqs,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        profiler_config=profiler_config,
        disable_log_stats=False,
    )
    if args.enable_expert_parallel:
        llm_kwargs["enable_expert_parallel"] = True
    if args.disable_cascade_attn:
        llm_kwargs["disable_cascade_attn"] = True
    if _is_step3p7_model_path(args.model):
        llm_kwargs["language_model_only"] = True
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0}
        llm_kwargs["skip_mm_profiling"] = True
        llm_kwargs["mm_processor_cache_gb"] = 0
    compilation_config = get_compilation_config_for_cudagraph_mode(
        args.cudagraph_mode
    )
    if compilation_config is not None:
        llm_kwargs["compilation_config"] = compilation_config
    if args.attention_backend is not None:
        llm_kwargs["attention_backend"] = args.attention_backend
    if mode == "dflash":
        llm_kwargs["speculative_config"] = {
            "method": "dflash",
            "model": args.draft_model,
            "num_speculative_tokens": effective_num_speculative_tokens,
            "max_model_len": args.max_model_len,
            "head_type": args.head_type,
            "tree_width": args.tree_width,
            "max_tree_budget": args.max_tree_budget,
            "tree_draft": args.tree_draft,
            "tree_hybrid_alpha": args.tree_hybrid_alpha,
            "max_draft_passes": args.max_draft_passes,
            "tree_prune_ratio": args.tree_prune_ratio,
            "tree_construction": args.tree_construction,
            "tree_attn_kernel": args.tree_attn_kernel,
            "tree_kv_layout": args.tree_kv_layout,
            "num_cudagraph_tree_captures": args.num_cudagraph_tree_captures,
        }
        if args.tree_seed_budget is not None:
            llm_kwargs["speculative_config"]["tree_seed_budget"] = (
                args.tree_seed_budget
            )
        if args.tree_attn_kernel == "optimus":
            llm_kwargs["block_size"] = 128

    llm = LLM(**llm_kwargs)
    if pre_generation_hook is not None:
        try:
            pre_generation_hook(llm)
        except Exception as hook_exc:
            import traceback

            print(f"WARNING: pre_generation_hook failed: {hook_exc}")
            traceback.print_exc()
    warmup_batches = prompt_batches[: min(args.num_warmup_runs, len(prompt_batches))]
    for batch_prompts in warmup_batches:
        llm.generate(batch_prompts, sampling_params=sampling_params)

    if mode == "dflash" and args.write_dflash_debug_artifacts:
        enable_dflash_debug_artifacts(llm)

    metrics_before = collect_spec_decode_counters(llm.get_metrics())
    if args.profiler != "none":
        llm.start_profile()
    t0 = time.perf_counter()
    total_output_tokens = 0
    total_prompt_tokens = 0
    time_per_output_token_samples: list[float] = []
    sampled_outputs: list[tuple[str, str]] = []
    num_batches = 0
    num_samples = 0
    prompts_per_run = sum(len(batch_prompts) for batch_prompts in prompt_batches)
    total_progress_prompts = args.num_runs * prompts_per_run
    completed_progress_prompts = 0
    for run_idx in range(args.num_runs):
        for batch_idx, batch_prompts in enumerate(prompt_batches):
            batch_t0 = time.perf_counter()
            outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
            batch_elapsed = time.perf_counter() - batch_t0
            batch_output_tokens = sum(
                len(output.outputs[0].token_ids) for output in outputs
            )
            batch_prompt_tokens = sum(
                len(output.prompt_token_ids) for output in outputs
                if output.prompt_token_ids is not None
            )
            total_output_tokens += batch_output_tokens
            total_prompt_tokens += batch_prompt_tokens
            num_batches += 1
            num_samples += len(outputs)
            completed_progress_prompts += len(batch_prompts)
            print(
                "[PROGRESS] "
                f"prompt_set={args.prompt_set} mode={mode} tp={tp_size} "
                f"bs={batch_size} run={run_idx + 1}/{args.num_runs} "
                f"batch={batch_idx + 1}/{len(prompt_batches)} "
                f"prompts={completed_progress_prompts}/{total_progress_prompts}",
                flush=True,
            )
            if batch_output_tokens > 0:
                time_per_output_token_samples.append(
                    batch_elapsed / batch_output_tokens
                )
            if run_idx == args.num_runs - 1 and batch_idx < 3:
                sampled_outputs.extend(
                    (output.prompt, output.outputs[0].text) for output in outputs
                )
    elapsed = time.perf_counter() - t0
    if args.profiler != "none":
        llm.stop_profile()
    metrics_after = collect_spec_decode_counters(llm.get_metrics())
    metrics_delta = diff_counters(metrics_after, metrics_before)

    if mode == "dflash" and args.write_dflash_debug_artifacts:
        def _extract_topk_log(worker):
            worker_obj = getattr(worker, "worker", worker)
            model_runner = getattr(worker_obj, "model_runner", None)
            drafter = getattr(model_runner, "drafter", None)
            if drafter is not None and hasattr(drafter, "get_topk_log"):
                return drafter.get_topk_log()
            return []

        try:
            topk_logs = llm.collective_rpc(_extract_topk_log)
            topk_entries = topk_logs[0] if topk_logs else []
        except Exception as e:
            import traceback
            print(f"\n{'='*60}")
            print(f"ERROR extracting topk_log via collective_rpc: {e}")
            traceback.print_exc()
            print(f"{'='*60}\n")
            topk_entries = []
        if not topk_entries:
            try:
                worker = llm.llm_engine.model_executor.driver_worker.worker
                drafter = getattr(worker.model_runner, "drafter", None)
                if drafter is not None and hasattr(drafter, "get_topk_log"):
                    topk_entries = drafter.get_topk_log()
            except Exception as e:
                print(f"WARNING: direct topk_log extraction failed: {e}")
        if topk_entries:
            topk_path = run_output_dir / "topk_log.json"
            topk_path.write_text(json.dumps(topk_entries, indent=2))
            print(f"Wrote {len(topk_entries)} topk log entries to {topk_path}")
        else:
            print("WARNING: topk_log is empty — no entries collected from drafter")

        write_dflash_tree_debug_records(llm, run_output_dir)
        write_dflash_runtime_verify_bundles(llm, run_output_dir)
        write_dflash_tree_commit_debug_records(llm, run_output_dir)

    if post_generation_hook is not None:
        try:
            post_generation_hook(llm)
        except Exception as hook_exc:
            import traceback

            print(f"WARNING: post_generation_hook failed: {hook_exc}")
            traceback.print_exc()

    if args.skip_engine_cleanup:
        print(
            "Skipping explicit engine cleanup; process teardown will release "
            "resources."
        )
    else:
        cleanup_llm(
            llm,
            shutdown_timeout=getattr(args, "engine_shutdown_timeout", 10.0),
        )
        del llm
        time.sleep(args.sleep_after_stop)
    phase_cuda = collect_execute_context_cuda_seconds(run_output_dir)

    num_drafts = float(metrics_delta["num_drafts"])
    num_draft_tokens = float(metrics_delta["num_draft_tokens"])
    num_accepted_tokens = float(metrics_delta["num_accepted_tokens"])
    num_tree_drafts = float(metrics_delta["num_tree_drafts"])
    num_tree_nodes = float(metrics_delta["num_tree_nodes"])
    accepted_per_pos = metrics_delta.get("accepted_per_pos", [])
    if not isinstance(accepted_per_pos, list):
        accepted_per_pos = []
    tree_nodes_per_depth_raw = metrics_delta.get("tree_nodes_per_depth", [])
    if not isinstance(tree_nodes_per_depth_raw, list):
        tree_nodes_per_depth_raw = []

    draft_tokens = (
        num_tree_nodes - num_tree_drafts
        if num_tree_drafts > 0
        else num_draft_tokens
    )

    per_pos_acceptance_rate: list[float] = []
    if num_drafts > 0 and accepted_per_pos:
        per_pos_acceptance_rate = [v / num_drafts for v in accepted_per_pos]

    acceptance_length_histogram: list[float] = []
    if num_drafts > 0 and per_pos_acceptance_rate:
        n = len(per_pos_acceptance_rate)
        cumulative = list(per_pos_acceptance_rate)
        hist = [0.0] * (n + 1)
        for d in range(n):
            rate_at_d = cumulative[d]
            rate_at_next = cumulative[d + 1] if d + 1 < n else 0.0
            hist[d + 1] = rate_at_d - rate_at_next
        hist[0] = 1.0 - cumulative[0] if cumulative else 1.0
        acceptance_length_histogram = hist

    prefill_cuda_s = phase_cuda["prefill_cuda_s"]
    decode_cuda_s = phase_cuda["decode_cuda_s"]
    prefill_throughput = (
        total_prompt_tokens / prefill_cuda_s if prefill_cuda_s > 0 else 0.0
    )
    decode_throughput = (
        total_output_tokens / decode_cuda_s if decode_cuda_s > 0 else 0.0
    )
    gpu_active_s = prefill_cuda_s + decode_cuda_s + phase_cuda["mixed_cuda_s"]
    gpu_utilization = gpu_active_s / elapsed if elapsed > 0 else 0.0

    dflash_breakdown: dict[str, float] = {}
    if mode == "dflash" and args.profiler == "torch":
        dflash_breakdown = write_dflash_breakdown_report(
            run_output_dir,
            elapsed_s=elapsed,
            phase_cuda=phase_cuda,
            num_drafts=num_drafts,
        )

    total_tokens = total_prompt_tokens + total_output_tokens
    return {
        "elapsed": elapsed,
        "total_output_tokens": total_output_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_tokens": total_tokens,
        "throughput": total_output_tokens / elapsed if elapsed > 0 else 0.0,
        "total_throughput": total_tokens / elapsed if elapsed > 0 else 0.0,
        "prefill_throughput": prefill_throughput,
        "decode_throughput": decode_throughput,
        "gpu_utilization": gpu_utilization,
        "draft_tokens": draft_tokens,
        "accepted_tokens": num_accepted_tokens,
        "drafts": num_drafts,
        "acceptance_rate": (
            num_accepted_tokens / draft_tokens if draft_tokens > 0 else 0.0
        ),
        "acceptance_length": 1.0 + (
            num_accepted_tokens / num_drafts
            if num_drafts > 0
            else 0.0
        ),
        "per_pos_acceptance_rate": per_pos_acceptance_rate,
        "acceptance_length_histogram": acceptance_length_histogram,
        "avg_tree_nodes_per_depth": (
            [v / num_tree_drafts for v in tree_nodes_per_depth_raw]
            if num_tree_drafts > 0 and tree_nodes_per_depth_raw
            else []
        ),
        "phase_cuda": phase_cuda,
        "dflash_breakdown": dflash_breakdown,
        "outputs": sampled_outputs,
        "engine_label": "vllm_native",
        "mean_time_per_output_token_s": mean_or_zero(time_per_output_token_samples),
        "num_batches": num_batches,
        "num_samples": num_samples,
    }


def parse_args():
    parser = FlexibleArgumentParser(
        description=(
            "Profile vLLM DFlash offline generation.\n\n"
            "Positional `model` can be a local directory or HF model id.\n"
            f"If omitted, defaults to {DEFAULT_TARGET_MODEL}."
        )
    )
    parser.add_argument(
        "--prompt-set",
        type=str,
        default="example-mix",
        choices=["example-mix", "example-coding", "gsm8k", "humaneval", "math-500"],
        help=(
            "Prompt set to use. "
            "'example-mix' uses general profiling prompts; 'example-coding' "
            "uses 4 Python algorithm/data-structure tasks; 'gsm8k', 'humaneval', and "
            "'math-500' match dataset prompt formatting used in the dflash repo."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=16,
        help=(
            "Maximum number of prompts or dataset entries to benchmark. "
            "Use a non-positive value to keep the full prompt set."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TARGET_MODEL,
        help=(
            "Target model path or HF model id. "
            "Can be a local directory; defaults to Qwen/Qwen3-8B."
        ),
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=DEFAULT_DRAFT_MODEL,
        help=(
            "DFlash draft/speculator model path or HF model id. "
            "Can be a local directory; defaults to z-lab/Qwen3-8B-DFlash-b16."
        ),
    )
    parser.add_argument(
        "--profiler",
        type=str,
        default="torch",
        choices=["none", "torch", "cuda"],
        help="Profiler backend. Use 'none' for throughput-only runs.",
    )
    parser.add_argument(
        "--torch-profiler-dir",
        type=str,
        default="./vllm_profile_dflash",
        help="Output directory for torch profiler traces.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=16,
        help="Number of speculative tokens for DFlash.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help=(
            "Reference-style DFlash block size. When set, vLLM uses "
            "num_speculative_tokens=block_size-1 so native runs match the "
            "reference benchmark's block-size semantics."
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Max model length.",
    )
    parser.add_argument(
        "--tp-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Tensor parallel sizes to profile. Native parity mode currently supports only 1.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes to profile. Native parity mode currently supports only 1.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["dflash", "ar", "both"],
        help="Profile dflash only, ar only, or both for throughput gains.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization target.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=51200,
        help=(
            "Scheduler max_num_batched_tokens passed to vLLM. Defaults to 51200 "
            "for tree profiling so full-tree slot reservation does not collapse "
            "the scheduled-token budget."
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help=(
            "Scheduler max_num_seqs passed to vLLM. Defaults to the largest "
            "profiled batch size when unset."
        ),
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs and use eager mode.",
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="Forward enable_expert_parallel=True to vLLM LLM construction.",
    )
    parser.add_argument(
        "--disable-cascade-attn",
        action="store_true",
        help="Forward disable_cascade_attn=True to vLLM LLM construction.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Generation max tokens.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=2,
        help="Number of timed and profiled generate calls.",
    )
    parser.add_argument(
        "--num-warmup-runs",
        type=int,
        default=1,
        help="Number of warmup generate calls before profiling.",
    )
    parser.add_argument(
        "--sleep-after-stop",
        type=int,
        default=10,
        help="Seconds to wait after stop_profile to allow trace flush.",
    )
    parser.add_argument(
        "--skip-engine-cleanup",
        action="store_true",
        help=(
            "Skip explicit llm.sleep()/engine shutdown at the end of a run. "
            "Useful when each mode runs in its own subprocess and TP worker "
            "shutdown would otherwise block metrics reporting."
        ),
    )
    parser.add_argument(
        "--engine-shutdown-timeout",
        type=float,
        default=10.0,
        help=(
            "Maximum seconds to wait for vLLM engine process shutdown during "
            "explicit cleanup."
        ),
    )
    parser.add_argument(
        "--write-dflash-debug-artifacts",
        action="store_true",
        help=(
            "Collect and write DFlash tree/debug/commit JSON artifacts after "
            "generation. Disabled by default to keep profiling runs focused on "
            "throughput and profiler reports."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable trust_remote_code for model loading.",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="FLASH_ATTN",
        help=(
            "Optional attention backend override, e.g. FLASH_ATTN, TRITON_ATTN, TREE_ATTN, "
            "or FLEX_ATTENTION."
        ),
    )
    parser.add_argument(
        "--head-type",
        type=str,
        default="auto",
        choices=["auto", "bidirectional", "causal"],
        help=(
            "DFlash draft attention mode. 'auto' follows the draft checkpoint, "
            "'bidirectional' forces non-causal draft attention, and 'causal' "
            "forces causal draft attention."
        ),
    )
    parser.add_argument(
        "--tree-width",
        type=int,
        default=1,
        help=(
            "Requested DFlash draft tree width. Width 1 keeps the current linear "
            "parallel drafting path."
        ),
    )
    parser.add_argument(
        "--max-tree-budget",
        type=int,
        default=None,
        help="Optional cap on total tree nodes for DFlash tree experiments.",
    )
    parser.add_argument(
        "--tree-seed-budget",
        type=int,
        default=None,
        help=(
            "Optional root-inclusive seed-tree budget. When omitted, the "
            "seed budget equals max-tree-budget."
        ),
    )
    parser.add_argument(
        "--tree-draft",
        type=str,
        default="accum_logp",
        choices=[
            "accum_logp",
            "entropy",
            "hybrid",
            "opt_prefix",
            "top2gap_fanout",
        ],
        help=(
            "Scoring strategy for tree node expansion. "
            "'accum_logp' prioritises high-probability prefixes. "
            "'entropy' prioritises uncertain positions. "
            "'hybrid' combines cumulative log-prob with entropy. "
            "'opt_prefix' builds the provably optimal tree under factorized "
            "draft marginals (ignores --tree-construction). "
            "'top2gap_fanout' caps per-depth fanout from the rank-1/rank-2 "
            "logprob gap."
        ),
    )
    parser.add_argument(
        "--tree-hybrid-alpha",
        type=float,
        default=1.0,
        help="Weight for per-depth entropy in 'hybrid' scoring mode.",
    )
    parser.add_argument(
        "--max-draft-passes",
        type=int,
        default=5,
        help="Number of prune/regrow refinement passes after initial tree build.",
    )
    parser.add_argument(
        "--tree-prune-ratio",
        type=float,
        default=0.25,
        help="Fraction of leaves to prune per refinement pass (0.0-1.0).",
    )
    parser.add_argument(
        "--tree-construction",
        type=str,
        default="breadth_first",
        choices=["depth_first", "breadth_first"],
        help=(
            "Tree node allocation strategy. 'depth_first' pre-allocates "
            "the greedy spine to full depth before side branches. "
            "'breadth_first' uses best-cumulative-logprob heap in a "
            "breadth-first manner (legacy)."
        ),
    )
    parser.add_argument(
        "--tree-attn-kernel",
        type=str,
        default="triton",
        choices=["triton", "optimus"],
        help=(
            "Attention kernel for DFlash tree verification. "
            "'triton' uses Triton bias-based path; "
            "'optimus' uses fused SM90 paged tree-mask kernel."
        ),
    )
    parser.add_argument(
        "--tree-kv-layout",
        type=str,
        default="physical",
        choices=["physical", "logical"],
        help=(
            "DFlash tree KV-cache layout. 'physical' uses the current "
            "compact-after-accept path; 'logical' enables the experimental "
            "accepted-slot indirection path."
        ),
    )
    parser.add_argument(
        "--num-cudagraph-tree-captures",
        type=int,
        default=0,
        help=(
            "Enable DFlash tree CUDAGraph capture when > 0. "
            "Tree verification captures max-tree-budget * num_reqs shapes, "
            "and trees are adjusted to max-tree-budget for guaranteed "
            "CUDAGraph hits. 0 disables."
        ),
    )
    parser.add_argument(
        "--cudagraph-mode",
        type=str,
        default="default",
        choices=[
            "default",
            "none",
            "full",
            "full_decode_only",
            "full_and_piecewise",
            "piecewise",
        ],
        help=(
            "Override global vLLM CUDA graph mode. Use 'full_decode_only' "
            "to test DFlash target-model full CUDA graph replay, or 'none' "
            "to test non-eager execution without CUDA graph capture."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.profiler == "torch":
        os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"
    else:
        os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "0"
        os.environ["VLLM_NVTX_SCOPES_FOR_PROFILING"] = "0"
    if args.num_runs < 1:
        raise ValueError("--num-runs must be >= 1")
    if args.num_warmup_runs < 0:
        raise ValueError("--num-warmup-runs must be >= 0")
    if args.tree_width < 1:
        raise ValueError("--tree-width must be >= 1")
    validate_native_only_settings(args)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        **tokenizer_load_kwargs(args.model),
    )
    prompt_bank = apply_chat_template(
        tokenizer,
        get_prompt_bank(args.prompt_set),
        args.model,
    )
    if args.max_samples > 0:
        prompt_bank = prompt_bank[: args.max_samples]
    modes = get_modes(args.mode)
    effective_num_speculative_tokens = resolve_effective_num_speculative_tokens(args)
    effective_block_size = (
        resolve_effective_block_size(args) or (effective_num_speculative_tokens + 1)
    )
    effective_max_num_seqs = args.max_num_seqs or max(args.batch_sizes)
    # key: (tp_size, batch_size, mode) -> throughput tokens/s
    throughputs: dict[tuple[int, int, str], float] = {}
    benchmark_tok_s: dict[tuple[int, int, str], float] = {}
    execute_context_cuda: dict[tuple[int, int, str], dict[str, float]] = {}
    summary_lines = []

    for tp_size in args.tp_sizes:
        if tp_size < 1:
            raise ValueError(f"Invalid tp size: {tp_size}. Expected tp >= 1")
        for batch_size in args.batch_sizes:
            if batch_size < 1:
                raise ValueError(
                    f"Invalid batch size: {batch_size}. Expected batch size >= 1"
                )
            prompt_batches = build_prompt_batches(prompt_bank, batch_size)

            for mode in modes:
                print("=" * 80)
                print(
                    f"Profiling mode={mode}, tensor_parallel_size={tp_size}, "
                    f"batch_size={batch_size}"
                )
                cleanup_llm(None)

                if args.profiler == "none":
                    run_output_dir = Path(
                        f"{args.torch_profiler_dir}/{mode}/tp{tp_size}/bs{batch_size}"
                    )
                    profiler_config = {"profiler": None}
                elif args.profiler == "torch":
                    run_output_dir = Path(
                        f"{args.torch_profiler_dir}/{mode}/tp{tp_size}/bs{batch_size}"
                    )
                    profiler_config = {
                        "profiler": "torch",
                        "torch_profiler_dir": str(run_output_dir),
                    }
                else:
                    run_output_dir = Path(
                        f"{args.torch_profiler_dir}/{mode}/tp{tp_size}/bs{batch_size}"
                    )
                    profiler_config = {"profiler": "cuda"}
                run_output_dir.mkdir(parents=True, exist_ok=True)
                result = run_native_profile(
                    args=args,
                    mode=mode,
                    prompt_batches=prompt_batches,
                    sampling_params=sampling_params,
                    tp_size=tp_size,
                    batch_size=batch_size,
                    effective_num_speculative_tokens=effective_num_speculative_tokens,
                    effective_max_num_seqs=effective_max_num_seqs,
                    profiler_config=profiler_config,
                    run_output_dir=run_output_dir,
                )

                elapsed = float(result["elapsed"])
                total_output_tokens = int(result["total_output_tokens"])
                total_prompt_tokens = int(result["total_prompt_tokens"])
                total_tokens = int(result["total_tokens"])
                throughput = float(result["throughput"])
                total_throughput = float(result["total_throughput"])
                prefill_throughput = float(result["prefill_throughput"])
                decode_throughput = float(result["decode_throughput"])
                gpu_utilization = float(result["gpu_utilization"])
                throughputs[(tp_size, batch_size, mode)] = throughput
                draft_tokens = float(result["draft_tokens"])
                accepted_tokens = float(result["accepted_tokens"])
                drafts = float(result["drafts"])
                acceptance_rate = float(result["acceptance_rate"])
                acceptance_length = float(result["acceptance_length"])
                phase_cuda = result["phase_cuda"]
                mean_time_per_output_token_s = float(result["mean_time_per_output_token_s"])
                benchmark_speed_tok_s = (
                    1.0 / mean_time_per_output_token_s
                    if mean_time_per_output_token_s > 0
                    else 0.0
                )
                benchmark_tok_s[(tp_size, batch_size, mode)] = benchmark_speed_tok_s
                print(
                    f"[RESULT] mode={mode} tp={tp_size} bs={batch_size} "
                    f"prompt_tokens={total_prompt_tokens} "
                    f"output_tokens={total_output_tokens} total_tokens={total_tokens} "
                    f"elapsed_s={elapsed:.3f} "
                    f"throughput_tok_s={throughput:.2f}"
                )
                print(
                    f"[THROUGHPUT] mode={mode} tp={tp_size} bs={batch_size} "
                    f"prefill_tok_s={prefill_throughput:.2f} "
                    f"decode_tok_s={decode_throughput:.2f} "
                    f"e2e_output_tok_s={throughput:.2f} "
                    f"e2e_total_tok_s={total_throughput:.2f} "
                    f"gpu_utilization={gpu_utilization:.2%}"
                )
                print(
                    f"[BENCHMARK] mode={mode} tp={tp_size} bs={batch_size} "
                    f"num_samples={result['num_samples']} num_batches={result['num_batches']} "
                    f"mean_time_per_output_token_s={mean_time_per_output_token_s:.6f} "
                    f"benchmark_tok_s={benchmark_speed_tok_s:.2f}"
                )
                print(
                    f"[SPEC_METRICS] mode={mode} tp={tp_size} bs={batch_size} "
                    f"num_drafts={drafts:.0f} num_draft_tokens={draft_tokens:.0f} "
                    f"num_accepted_tokens={accepted_tokens:.0f} "
                    f"acceptance_rate={acceptance_rate:.4f} "
                    f"acceptance_length={acceptance_length:.4f}"
                )
                per_pos_rate = result.get("per_pos_acceptance_rate", [])
                acc_len_hist = result.get("acceptance_length_histogram", [])
                if per_pos_rate:
                    pos_str = " ".join(
                        f"d{i}={r:.4f}" for i, r in enumerate(per_pos_rate)
                    )
                    print(
                        f"[PER_DEPTH_ACCEPTANCE] mode={mode} tp={tp_size} "
                        f"bs={batch_size} {pos_str}"
                    )
                if acc_len_hist:
                    hist_str = " ".join(
                        f"len{i}={p:.4f}" for i, p in enumerate(acc_len_hist)
                    )
                    print(
                        f"[ACCEPTANCE_LEN_HIST] mode={mode} tp={tp_size} "
                        f"bs={batch_size} {hist_str}"
                    )
                avg_tree_shape = result.get("avg_tree_nodes_per_depth", [])
                if avg_tree_shape:
                    shape_str = " ".join(
                        f"d{i}={v:.1f}" for i, v in enumerate(avg_tree_shape)
                    )
                    print(
                        f"[TREE_SHAPE] mode={mode} tp={tp_size} "
                        f"bs={batch_size} {shape_str}"
                    )
                for prompt, generated_text in result["outputs"]:
                    print("-" * 80)
                    print(f"Prompt: {prompt}")
                    print(f"Generated: {generated_text}")

                execute_context_cuda[(tp_size, batch_size, mode)] = phase_cuda

                report_lines = [
                    f"mode={mode}",
                    f"engine={result['engine_label']}",
                    f"prompt_set={args.prompt_set}",
                    "prompt_format=chat_template",
                    f"attention_backend={args.attention_backend}",
                    f"head_type={args.head_type}",
                    f"block_size={effective_block_size}",
                    f"tree_width={args.tree_width}",
                    f"max_tree_budget={args.max_tree_budget}",
                    f"tree_draft={args.tree_draft}",
                    f"max_draft_passes={args.max_draft_passes}",
                    f"max_num_batched_tokens={args.max_num_batched_tokens}",
                    f"max_num_seqs={effective_max_num_seqs}",
                    f"num_samples={result['num_samples']}",
                    f"num_batches={result['num_batches']}",
                    f"tp_size={tp_size}",
                    f"batch_size={batch_size}",
                    f"prompt_tokens={total_prompt_tokens}",
                    f"output_tokens={total_output_tokens}",
                    f"total_tokens={total_tokens}",
                    f"elapsed_s={elapsed:.6f}",
                    f"e2e_throughput_tok_s={throughput:.6f}",
                    f"e2e_total_throughput_tok_s={total_throughput:.6f}",
                    f"prefill_throughput_tok_s={prefill_throughput:.6f}",
                    f"decode_throughput_tok_s={decode_throughput:.6f}",
                    f"gpu_utilization={gpu_utilization:.6f}",
                    f"mean_time_per_output_token_s={mean_time_per_output_token_s:.6f}",
                    f"benchmark_tok_s={benchmark_speed_tok_s:.6f}",
                    f"num_drafts={drafts:.0f}",
                    f"num_draft_tokens={draft_tokens:.0f}",
                    f"num_accepted_tokens={accepted_tokens:.0f}",
                    f"acceptance_rate={acceptance_rate:.6f}",
                    f"acceptance_length={acceptance_length:.6f}",
                    f"prefill_execute_context_cuda_s={phase_cuda['prefill_cuda_s']:.6f}",
                    f"decode_execute_context_cuda_s={phase_cuda['decode_cuda_s']:.6f}",
                    f"mixed_execute_context_cuda_s={phase_cuda['mixed_cuda_s']:.6f}",
                ]
                if per_pos_rate:
                    report_lines.append(
                        "per_depth_acceptance_rate="
                        + ",".join(f"{r:.6f}" for r in per_pos_rate)
                    )
                if acc_len_hist:
                    report_lines.append(
                        "acceptance_length_histogram="
                        + ",".join(f"{p:.6f}" for p in acc_len_hist)
                    )
                if avg_tree_shape:
                    report_lines.append(
                        "avg_tree_nodes_per_depth="
                        + ",".join(f"{v:.2f}" for v in avg_tree_shape)
                    )
                (run_output_dir / "metrics_report.txt").write_text(
                    "\n".join(report_lines) + "\n", encoding="utf-8"
                )
                summary_lines.append(
                    " ".join(
                        [
                            f"mode={mode}",
                            f"engine={result['engine_label']}",
                            f"prompt_set={args.prompt_set}",
                            "prompt_format=chat_template",
                            f"attention_backend={args.attention_backend}",
                            f"head_type={args.head_type}",
                            f"block_size={effective_block_size}",
                            f"tree_width={args.tree_width}",
                            f"max_tree_budget={args.max_tree_budget}",
                            f"tree_draft={args.tree_draft}",
                            f"max_draft_passes={args.max_draft_passes}",
                            f"max_num_batched_tokens={args.max_num_batched_tokens}",
                            f"max_num_seqs={effective_max_num_seqs}",
                            f"num_samples={result['num_samples']}",
                            f"num_batches={result['num_batches']}",
                            f"tp={tp_size}",
                            f"bs={batch_size}",
                            f"prompt_tokens={total_prompt_tokens}",
                            f"output_tokens={total_output_tokens}",
                            f"total_tokens={total_tokens}",
                            f"e2e_throughput_tok_s={throughput:.6f}",
                            f"e2e_total_throughput_tok_s={total_throughput:.6f}",
                            f"prefill_throughput_tok_s={prefill_throughput:.6f}",
                            f"decode_throughput_tok_s={decode_throughput:.6f}",
                            f"gpu_utilization={gpu_utilization:.6f}",
                            f"mean_time_per_output_token_s={mean_time_per_output_token_s:.6f}",
                            f"benchmark_tok_s={benchmark_speed_tok_s:.6f}",
                            f"num_drafts={drafts:.0f}",
                            f"num_draft_tokens={draft_tokens:.0f}",
                            f"num_accepted_tokens={accepted_tokens:.0f}",
                            f"acceptance_rate={acceptance_rate:.6f}",
                            f"acceptance_length={acceptance_length:.6f}",
                            f"prefill_execute_context_cuda_s={phase_cuda['prefill_cuda_s']:.6f}",
                            f"decode_execute_context_cuda_s={phase_cuda['decode_cuda_s']:.6f}",
                            f"mixed_execute_context_cuda_s={phase_cuda['mixed_cuda_s']:.6f}",
                        ] + (
                            [
                                "per_depth_acceptance_rate="
                                + ",".join(f"{r:.6f}" for r in per_pos_rate)
                            ] if per_pos_rate else []
                        ) + (
                            [
                                "acceptance_length_histogram="
                                + ",".join(f"{p:.6f}" for p in acc_len_hist)
                            ] if acc_len_hist else []
                        ) + (
                            [
                                "avg_tree_nodes_per_depth="
                                + ",".join(f"{v:.2f}" for v in avg_tree_shape)
                            ] if avg_tree_shape else []
                        )
                    )
                )

    if "dflash" in modes and "ar" in modes:
        print("=" * 80)
        print("Throughput gains report (DFlash vs AR)")
        gains = []
        gain_lines = []
        for tp_size in args.tp_sizes:
            for batch_size in args.batch_sizes:
                ar_tps = throughputs.get((tp_size, batch_size, "ar"), 0.0)
                dflash_tps = throughputs.get((tp_size, batch_size, "dflash"), 0.0)
                ar_bench_tps = benchmark_tok_s.get((tp_size, batch_size, "ar"), 0.0)
                dflash_bench_tps = benchmark_tok_s.get((tp_size, batch_size, "dflash"), 0.0)
                if ar_tps <= 0:
                    print(f"tp={tp_size} bs={batch_size}: AR throughput unavailable")
                    continue
                gain = dflash_tps / ar_tps
                gains.append(gain)
                gain_pct = (gain - 1.0) * 100.0
                benchmark_gain = (
                    dflash_bench_tps / ar_bench_tps if ar_bench_tps > 0 else 0.0
                )
                ar_cuda = execute_context_cuda.get(
                    (tp_size, batch_size, "ar"),
                    {"prefill_cuda_s": 0.0, "decode_cuda_s": 0.0, "mixed_cuda_s": 0.0},
                )
                dflash_cuda = execute_context_cuda.get(
                    (tp_size, batch_size, "dflash"),
                    {"prefill_cuda_s": 0.0, "decode_cuda_s": 0.0, "mixed_cuda_s": 0.0},
                )
                decode_speedup = (
                    ar_cuda["decode_cuda_s"] / dflash_cuda["decode_cuda_s"]
                    if dflash_cuda["decode_cuda_s"] > 0
                    else 0.0
                )
                prefill_speedup = (
                    ar_cuda["prefill_cuda_s"] / dflash_cuda["prefill_cuda_s"]
                    if dflash_cuda["prefill_cuda_s"] > 0
                    else 0.0
                )
                print(
                    f"tp={tp_size} bs={batch_size}: "
                    f"AR={ar_tps:.2f} tok/s, DFlash={dflash_tps:.2f} tok/s, "
                    f"gain={gain:.3f}x ({gain_pct:+.2f}%), "
                    f"benchmark_gain={benchmark_gain:.3f}x, "
                    f"decode_cuda_speedup={decode_speedup:.3f}x, "
                    f"prefill_cuda_speedup={prefill_speedup:.3f}x"
                )
                gain_lines.append(
                    " ".join(
                        [
                            f"tp={tp_size}",
                            f"bs={batch_size}",
                            f"throughput_gain={gain:.6f}",
                            f"throughput_gain_pct={gain_pct:+.2f}",
                            f"benchmark_gain={benchmark_gain:.6f}",
                            f"ar_benchmark_tok_s={ar_bench_tps:.6f}",
                            f"dflash_benchmark_tok_s={dflash_bench_tps:.6f}",
                            f"decode_cuda_speedup={decode_speedup:.6f}",
                            f"prefill_cuda_speedup={prefill_speedup:.6f}",
                            f"ar_decode_cuda_s={ar_cuda['decode_cuda_s']:.6f}",
                            f"dflash_decode_cuda_s={dflash_cuda['decode_cuda_s']:.6f}",
                            f"ar_prefill_cuda_s={ar_cuda['prefill_cuda_s']:.6f}",
                            f"dflash_prefill_cuda_s={dflash_cuda['prefill_cuda_s']:.6f}",
                        ]
                    )
                )
        if gains:
            avg_gain = sum(gains) / len(gains)
            print(f"Average gain across settings: {avg_gain:.3f}x")
        gains_path = Path(args.torch_profiler_dir) / "gains_report.txt"
        gains_path.parent.mkdir(parents=True, exist_ok=True)
        gains_path.write_text("\n".join(gain_lines) + "\n", encoding="utf-8")
        print(f"Wrote gains report to: {gains_path}")
    summary_path = Path(args.torch_profiler_dir) / "metrics_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote metrics summary to: {summary_path}")


if __name__ == "__main__":
    main()
