# DFlash Tree CPU Overhead Plan

## Goal

Reduce CPU-side overhead in native DFlash causal-tree decoding, especially for future `tp > 1` and `bsz > 1`.

## Main Hotspots

### 1. Per-step tree metadata rebuild

Current path:
- `vllm/v1/worker/gpu_model_runner.py::_calc_dflash_tree_spec_decode_metadata`

Current work each step:
- build Python lists for `query_lens`, `parent_indices`, `depths`
- create new tensors from those lists
- build per-request dense tree biases
- build one dense block-diagonal bias

### 2. Dense bias construction

Current helpers:
- `vllm/v1/spec_decode/dflash_tree.py::build_attention_bias_from_parents`
- `vllm/v1/spec_decode/dflash_tree.py::build_block_diagonal_attention_bias`

Problems:
- quadratic in query length
- repeated allocation
- repeated rebuild even when topology may repeat

### 3. CPU sync and Python conversion

Examples:
- `.cpu().tolist()`
- `torch.tensor(list, device=...)`
- Python-side assembly before GPU execution

### 4. Tree acceptance path

Current path:
- `vllm/v1/worker/gpu_model_runner.py::_sample_dflash_tree`

Problems:
- per-request Python loop
- accepted-path extraction on Python side
- extra small tensor creation

### 5. KV compaction

Current path:
- `vllm/v1/worker/gpu_model_runner.py::_compact_dflash_tree_kv_cache`

Problems:
- per-request loops
- index tensor creation
- `.clone()` copies

## Low-risk wins

### A. Reuse buffers

Preallocate and reuse:
- `parent_indices`
- `depths`
- `cu_query_lens`
- `target_logits_indices`
- block-diagonal bias output

Expected benefit:
- less allocator churn
- less Python list -> tensor overhead

### B. Cache causal fallback bias by `query_len`

Current non-tree fallback in the tree path rebuilds the same causal bias repeatedly.

### C. Remove small tensor churn

Replace repeated:
- `torch.tensor(list, ...)`
- tiny temporary index tensors

With:
- reusable scratch buffers
- in-place writes

### D. Reduce `.cpu().tolist()`

Keep more metadata in tensor form and avoid host syncs where possible.

## Medium-risk wins

### E. Cache per-request tree metadata by topology

Possible key:
- `tuple(parent_indices)`
- optionally `query_len` or `depths`

Reusable objects:
- per-request bias
- parent/depth tensors

### F. Rebuild only changed requests

For batched tree decode:
- reuse metadata for requests whose topology did not change
- rebuild only changed blocks

### G. Reuse block-diagonal output storage

Even if the contents change, the final dense bias tensor does not need to be reallocated each step.

## High-impact redesign

### H. Stop materializing dense tree bias tensors

Longer-term goal:
- represent tree structure compactly
- pass parent/ancestor structure directly to `TREE_ATTN`
- avoid dense `Q x Q` bias construction in Python

This is the biggest likely win for `tp > 1` and `bsz > 1`.

## Acceptance and KV follow-ups

### I. Move more tree acceptance logic off Python

Targets:
- less per-request looping
- fewer list conversions
- keep accepted-path processing in tensor form longer

### J. Optimize KV compaction

Possible directions:
- reusable index buffers
- batched gather/scatter
- fused compaction kernel
- fewer clones

## Recommended order

1. Reuse metadata buffers
2. Cache causal fallback bias
3. Remove obvious `.cpu().tolist()` and small tensor churn
4. Reuse block-diagonal storage
5. Cache per-request tree metadata by topology
6. Rebuild only changed requests
7. Redesign `TREE_ATTN` to avoid dense bias construction
8. Optimize acceptance and KV compaction

## Metrics to track

- end-to-end throughput
- decode latency
- CPU time around `_calc_dflash_tree_spec_decode_metadata`
- CPU -> GPU metadata copy volume
- average tree size
- acceptance rate
- acceptance length

## Success criteria

Short term:
- lower CPU overhead for `tp=1`, `bs=1`, tree mode
- no change in acceptance behavior

Mid term:
- stable tree mode with `tp > 1`
- acceptable overhead for `bsz > 1`

Long term:
- tree mode scales closer to linear DFlash
- metadata is no longer the main bottleneck
