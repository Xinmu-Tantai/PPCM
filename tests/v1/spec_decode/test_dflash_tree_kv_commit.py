# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_dflash_tree_kv_commit_records_root_only_path() -> None:
    runner = object.__new__(GPUModelRunner)
    runner._dflash_tree_accept_paths_gpu = [
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0, 2], dtype=torch.int64),
    ]
    runner._dflash_tree_accept_paths = []
    runner._dflash_tree_commit_debug_records = []
    runner._dflash_debug_artifact_max_records = 8
    runner._dflash_runtime_verify_bundles = []
    runner.input_batch = SimpleNamespace(req_ids=["req0", "req1"])
    runner._commit_dflash_logical_kv_slots = lambda *_args, **_kwargs: False

    kv_cache = torch.zeros((2, 1, 16, 1, 1), dtype=torch.float32)
    for slot in range(16):
        kv_cache[0, 0, slot, 0, 0] = 1000 + slot
        kv_cache[1, 0, slot, 0, 0] = 2000 + slot
    runner.kv_caches = [kv_cache]

    spec_decode_metadata = SimpleNamespace(
        query_lens=[3, 3],
        is_tree_req=[True, True],
    )
    common_attn_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([5, 6, 7, 10, 11, 12], dtype=torch.int64),
    )

    runner._compact_dflash_tree_kv_cache(
        spec_decode_metadata,
        common_attn_metadata,
    )

    records = runner.get_dflash_tree_commit_debug_records()
    assert [r["accepted_path"].tolist() for r in records] == [[0], [0, 2]]
    assert records[0]["compact_src_slots"].tolist() == [5]
    assert records[0]["compact_dst_slots"].tolist() == [5]

    assert kv_cache[0, 0, 11, 0, 0].item() == 1012
    assert kv_cache[1, 0, 11, 0, 0].item() == 2012
    assert runner._dflash_tree_accept_paths_gpu is None
    assert runner._dflash_tree_accept_paths is None
