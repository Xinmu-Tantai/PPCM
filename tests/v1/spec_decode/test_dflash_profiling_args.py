# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

from examples.offline_inference.dflash_profiling import (
    get_compilation_config_for_cudagraph_mode,
    validate_native_only_settings,
)


def _args(
    *,
    tp_sizes: list[int] | None = None,
    batch_sizes: list[int] | None = None,
    cudagraph_mode: str = "default",
):
    return SimpleNamespace(
        tp_sizes=tp_sizes or [1],
        batch_sizes=batch_sizes or [1],
        cudagraph_mode=cudagraph_mode,
    )


def test_dflash_profiling_allows_bsz_gt_one_without_cuda_graph() -> None:
    validate_native_only_settings(
        _args(batch_sizes=[2], cudagraph_mode="none")
    )


def test_dflash_profiling_allows_bsz_gt_one_with_cuda_graph() -> None:
    validate_native_only_settings(
        _args(batch_sizes=[2], cudagraph_mode="default")
    )


def test_dflash_profiling_allows_tp_gt_one() -> None:
    validate_native_only_settings(
        _args(tp_sizes=[8], batch_sizes=[2], cudagraph_mode="default")
    )


def test_cudagraph_mode_default_leaves_compilation_config_unset() -> None:
    assert get_compilation_config_for_cudagraph_mode("default") is None


def test_cudagraph_mode_full_decode_only_sets_vllm_mode() -> None:
    assert get_compilation_config_for_cudagraph_mode("full_decode_only") == {
        "cudagraph_mode": "FULL_DECODE_ONLY",
    }


def test_cudagraph_mode_none_sets_vllm_mode() -> None:
    assert get_compilation_config_for_cudagraph_mode("none") == {
        "cudagraph_mode": "NONE",
    }
