# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce the split custom-op hidden KV dependency pattern.

The attention wrapper uses one custom op to update a KV cache reached through
global forward context, then passes a small dummy tensor into a second custom op
that reads that hidden cache.  The cache tensor itself is not in either op's
schema.  This test checks whether torch.compile preserves that hidden mutation
when the only visible dependency is the dummy tensor.
"""

from __future__ import annotations

import pytest
import torch

from vllm.utils.torch_utils import direct_register_custom_op

_HIDDEN_KV: torch.Tensor | None = None


def _set_hidden_kv(cache: torch.Tensor) -> None:
    global _HIDDEN_KV
    _HIDDEN_KV = cache


def _hidden_kv_update(src: torch.Tensor) -> torch.Tensor:
    assert _HIDDEN_KV is not None
    _HIDDEN_KV.copy_(src)
    return torch.empty(0, device=src.device, dtype=src.dtype)


def _hidden_kv_update_fake(src: torch.Tensor) -> torch.Tensor:
    return torch.empty(0, device=src.device, dtype=src.dtype)


def _hidden_kv_read(dep: torch.Tensor, out: torch.Tensor) -> None:
    del dep
    assert _HIDDEN_KV is not None
    out.copy_(_HIDDEN_KV)


def _hidden_kv_read_fake(dep: torch.Tensor, out: torch.Tensor) -> None:
    return


direct_register_custom_op(
    op_name="dflash_test_hidden_kv_update",
    op_func=_hidden_kv_update,
    fake_impl=_hidden_kv_update_fake,
    mutates_args=[],
)
direct_register_custom_op(
    op_name="dflash_test_hidden_kv_read",
    op_func=_hidden_kv_read,
    fake_impl=_hidden_kv_read_fake,
    mutates_args=["out"],
)


def _split_hidden_kv_ops(src: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    dep = torch.ops.vllm.dflash_test_hidden_kv_update(src)
    torch.ops.vllm.dflash_test_hidden_kv_read(dep, out)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
@torch.inference_mode()
def test_split_custom_ops_preserve_hidden_kv_update_under_compile() -> None:
    cache = torch.zeros((4, 8), device="cuda", dtype=torch.float32)
    src = torch.arange(cache.numel(), device="cuda", dtype=torch.float32).view_as(cache)
    out = torch.empty_like(cache)
    _set_hidden_kv(cache)

    eager_out = _split_hidden_kv_ops(src, out)
    torch.testing.assert_close(eager_out, src)

    cache.zero_()
    out.zero_()
    compiled = torch.compile(_split_hidden_kv_ops, fullgraph=True)
    compiled_out = compiled(src, out)

    torch.testing.assert_close(compiled_out, src)
    torch.testing.assert_close(cache, src)
