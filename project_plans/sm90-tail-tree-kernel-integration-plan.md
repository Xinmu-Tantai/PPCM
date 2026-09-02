# SM90 Tail-Tree Kernel Integration Plan (vLLM)

## Goal

Integrate `optimus_cutedsl.flash_attn_varlen_tree_paged_sm90` as an optimized tree-verification attention path for causal parallel drafting, while preserving existing vLLM behavior via safe fallback.

## Why this kernel

- Matches production decode layout: paged KV cache + varlen metadata.
- Uses compact tree mask `(N,N)` / `(B,N,N)` rather than dense full mask.
- Keeps dense prefix path fast; specializes only tail tree blocks.
- Designed for one-kernel decode execution on SM90.

## Current vLLM mapping

Existing tree path (`vllm/v1/attention/backends/tree_attn.py`) already has:
- paged KV cache (`kv_cache.unbind(0)`),
- `query_start_loc` (maps to `cu_seqlens_q`),
- `seq_lens` (maps to `seqused_k`),
- `block_table` (maps to `page_table`),
- tree metadata (`tree_attn_bias`) from `TreeAttentionMetadataBuilder`.

Main gap: current backend consumes dense `qq_bias`; SM90 kernel wants compact tree mask + strict runtime constraints.

## Proposed architecture

1. **Dispatch strategy**
   - Add SM90 fast path inside `TreeAttentionImpl.forward` decode branch.
   - Keep current `unified_attention` path as default fallback.
   - Gate by capability/shape/dtype checks.

2. **Metadata extension**
   - Extend `TreeAttentionMetadata` with compact tree mask tensor:
     - `tree_mask_int` (int32, shape `(N,N)` or `(B,N,N)`).
   - Keep `tree_attn_bias` for fallback and compatibility.

3. **Backend invocation**
   - Convert vLLM tensors to kernel signature:
     - `q` -> decode query tensor
     - `k`, `v` -> paged KV cache tensors
     - `tree_mask` -> compact tree mask
     - `cu_seqlens_q` -> `query_start_loc`
     - `seqused_k` -> `seq_lens`
     - `page_table` -> `block_table`
   - Preserve GQA behavior (`pack_gqa` when needed).

## File touchpoints

- `vllm/v1/attention/backends/tree_attn.py`
  - Add fast-path dispatch + fallback logic in `TreeAttentionImpl.forward`.
  - Add guarded import and runtime check helpers.
- `vllm/v1/attention/backends/tree_attn.py` (metadata classes in same file)
  - Extend `TreeAttentionMetadata` / builder to carry compact mask.
- `vllm/v1/attention/backends/registry.py` (optional)
  - Optional explicit backend enum for SM90 tree path; otherwise keep as internal fast path of `TREE_ATTN`.
- `vllm/v1/spec_decode/eagle.py` (only if needed)
  - Ensure drafting metadata path carries new mask field correctly when building per-level tree metadata.

## Dispatch conditions (must pass)

- GPU compute capability major == 9 (SM90).
- dtype in `{fp16, bf16}` for q/k/v.
- paged KV shape is valid and contiguous enough for kernel.
- page size alignment matches kernel expectations (`page_size == n_block_size`).
- attention mode is causal tree verify path (not generic non-causal DFlash path).

If any check fails -> fallback to existing `unified_attention` + `qq_bias`.

## Validation plan

1. **Numerical parity**
   - Compare outputs against current vLLM tree path for fixed seeds:
     - multiple tree widths/depths,
     - GQA and non-GQA settings,
     - mixed sequence lengths.

2. **Behavior parity**
   - Verify acceptance length/rate and generated tokens match baseline tolerance.

3. **Performance**
   - Measure per-step latency split:
     - attention kernel time,
     - metadata build time,
     - rejection sampler time.
   - Benchmark with workload close to:
     - causal head,
     - entropy-guided tree draft,
     - multi-dataset script settings.

## Rollout plan

- Phase 1: hidden feature flag (off by default).
- Phase 2: enable on SM90 for selected configs.
- Phase 3: broader enablement after parity + perf thresholds hold.

## Risks and mitigations

- **Kernel constraint mismatch (page/tile size):**
  - Add startup validation and explicit warning/fallback.
- **Metadata shape drift in tree drafting levels:**
  - Add assertions on mask/query shapes per draft level.
- **Silent behavior regressions:**
  - Add deterministic A/B tests in CI for tree verify path.

