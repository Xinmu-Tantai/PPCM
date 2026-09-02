# Causal Head + Tree Drafting Execution Plan

## Goal

Add support to `vllm-parallel-drafting` for:

1. DFlash with a causal draft head.
2. DFlash with causal/bidirectional draft head plus tree inference.
3. Optional SM90 tree-verification acceleration using `optimus_cutedsl.flash_attn_varlen_tree_paged_sm90`.

This plan assumes the current DFlash path already supports:

- bidirectional draft head
- linear parallel drafting
- target-model verification with existing vLLM tree-attention infrastructure

## Non-Goals For First Pass

- Do not implement entropy-guided tree drafting in the first milestone.
- Do not replace all tree attention paths with the Optimus kernel.
- Do not change non-DFlash speculative methods unless required for shared backend support.

## Design Principles

1. Separate draft-side changes from verifier-side changes.
2. Land correctness first, then land kernel acceleration.
3. Reuse vLLM tree metadata/verifier infrastructure where possible.
4. Keep a clean fallback path for non-SM90 and unsupported cases.

## High-Level Milestones

### Milestone 1: Causal DFlash, no tree

Deliverables:

- DFlash supports `head_type=auto|bidirectional|causal`.
- `auto` resolves from checkpoint `dflash_config.causal_head`.
- Linear DFlash works for both bidirectional and causal heads.

### Milestone 2: DFlash crossproduct tree drafting

Deliverables:

- DFlash supports tree inference with the same crossproduct-style tree construction used in the `dflash` repo.
- Tree proposal uses DFlash parallel block logits, not EAGLE autoregressive tree drafting.
- Verification reuses existing vLLM tree verifier first.

### Milestone 3: Optimus SM90 tree verifier integration

Deliverables:

- Tree decode verification can optionally use the SM90 paged-tree kernel.
- Existing vLLM tree verification remains the fallback.
- Integration is generic at the tree verifier layer, not DFlash-only.

### Milestone 4: Optional entropy-guided tree drafting

Deliverables:

- Add `tree_draft=entropy_guided`.
- Add iterative prune/regrow draft refinement.
- Keep this behind a separate config path and test suite.

## Step-By-Step Execution Plan

### Step 0: Baseline inventory and freeze

Objective:

- Record the current DFlash behavior before making architectural changes.

Tasks:

1. Confirm the current DFlash path in:
   - `vllm/v1/spec_decode/dflash.py`
   - `vllm/model_executor/models/qwen3_dflash.py`
   - `vllm/v1/attention/backends/tree_attn.py`
2. Record current profiler behavior for:
   - bidirectional linear DFlash
   - AR baseline
3. Save one known-good run configuration and outputs for regression checking.

Exit criteria:

- A known-good bidirectional linear DFlash run exists for comparison after later changes.

### Step 1: Extend DFlash runtime config surface

Objective:

- Add explicit runtime controls for causal head and tree mode.

Files to update:

- `vllm/config/speculative.py`
- `vllm/engine/arg_utils.py`

Tasks:

1. Extend `speculative_config` to carry DFlash-specific knobs:
   - `head_type`: `auto|bidirectional|causal`
   - `tree_width`: integer, default `1`
   - `max_tree_budget`: optional integer
   - `tree_draft`: `crossproduct|entropy_guided`
   - optional verifier backend toggle if needed later
2. Keep defaults backward-compatible:
   - `head_type=auto`
   - `tree_width=1`
   - `tree_draft=crossproduct`
3. Validate config combinations:
   - `tree_width=1` means linear path
   - `entropy_guided` requires `tree_width > 1`
4. Ensure JSON `--speculative-config` can carry these fields without special-case parsing hacks.

Exit criteria:

- Users can request causal head and tree settings through standard vLLM config.

### Step 2: Resolve causal head in the DFlash stack

Objective:

- Support causal draft attention without breaking existing bidirectional DFlash.

Files to update:

- `vllm/v1/spec_decode/dflash.py`
- `vllm/model_executor/models/qwen3_dflash.py`

Tasks:

1. Add a helper equivalent to the reference `resolve_causal_head()` behavior.
2. Resolve `head_type` once during DFlash initialization:
   - `auto` uses checkpoint `dflash_config.causal_head`
   - `bidirectional` forces `False`
   - `causal` forces `True`
3. Remove the current hard-coded `causal=False` assumption in `DFlashProposer`.
4. Pass the resolved mode into the DFlash draft attention path.
5. Check whether metadata-based causal masking is sufficient.
6. If metadata-based masking is insufficient for some backend:
   - add explicit DFlash causal masking in `qwen3_dflash.py`
   - restrict unsupported backends explicitly

Validation:

1. Bidirectional linear DFlash still matches current behavior.
2. Causal linear DFlash runs without correctness regressions or crashes.
3. Backends fail clearly if unsupported.

Exit criteria:

- DFlash linear path supports both causal and bidirectional heads.

### Step 3: Define DFlash-specific tree proposal semantics

Objective:

- Add a DFlash tree proposal path that matches the `dflash` repo algorithm.

Key requirement:

- Do not reuse `EagleProposer.propose_tree()` as-is for DFlash proposal generation.

Reason:

- EAGLE tree drafting is autoregressive by level.
- DFlash crossproduct tree drafting is built from one parallel block pass plus per-position top-k marginals.

Files to update:

- `vllm/v1/spec_decode/dflash.py`
- possibly shared helper module under `vllm/v1/spec_decode/`

Tasks:

1. After the first parallel DFlash block pass, collect logits for speculative positions.
2. Add DFlash tree construction helpers for:
   - top-k extraction per position
   - budget computation
   - BFS tree construction
   - ancestor/path bookkeeping
3. Keep node ordering deterministic and consistent with verifier metadata.
4. Build the tree from DFlash logits using the crossproduct semantics used in the reference `dflash` repo.
5. Route DFlash into:
   - linear path when `tree_width == 1`
   - DFlash tree path when `tree_width > 1`

Validation:

1. On toy logits, tree construction matches the reference implementation.
2. BFS ordering is stable and documented.

Exit criteria:

- DFlash can produce a tree-shaped proposal from one parallel draft pass.

### Step 4: Bridge DFlash tree proposal to vLLM verifier metadata

Objective:

- Make DFlash tree proposals consumable by the existing vLLM tree verification machinery.

Files to update:

- `vllm/config/speculative.py`
- `vllm/v1/attention/backends/tree_attn.py`
- `vllm/v1/spec_decode/dflash.py`
- possibly `vllm/v1/spec_decode/metadata.py`

Tasks:

1. Decide the source of truth for verifier topology:
   - preferred: derive verifier topology from a vLLM-native tree representation
   - avoid duplicating tree semantics in multiple places
2. Convert DFlash tree proposal into the same tree topology expected by vLLM verifier metadata.
3. Populate `speculative_token_tree` or an equivalent internal tree representation consistently.
4. Ensure accepted-path indexing in verification matches the DFlash proposal node ordering.
5. Verify that correction-token insertion and KV-cache selection are aligned with accepted node indices.

Validation:

1. Compare accepted path and final output against the reference `dflash` tree verifier on small deterministic cases.
2. Verify no off-by-one mismatch between root, speculative positions, and correction token slots.

Exit criteria:

- DFlash tree proposals can be verified correctly with existing vLLM tree attention.

### Step 5: Land crossproduct tree inference using current verifier

Objective:

- Get correctness with the existing vLLM verifier before introducing a new kernel.

Tasks:

1. Use current `TreeAttentionImpl` for target verification.
2. Keep current `qq_bias`-based tree attention path as the initial implementation.
3. Add tests for:
   - bidirectional + tree
   - causal + tree
4. Benchmark throughput and acceptance behavior against:
   - bidirectional linear DFlash
   - causal linear DFlash
   - AR baseline

Exit criteria:

- Tree inference works correctly using existing vLLM tree verification.

### Step 6: Integrate the Optimus SM90 tree verification kernel

Objective:

- Accelerate tree decode verification using the Hopper paged-tree kernel.

Kernel to integrate:

- `optimus_cutedsl.flash_attn_varlen_tree_paged_sm90`

Relevant kernel assumptions:

1. Prefix is dense.
2. Tree tail is sparse by ancestor relation.
3. Query uses varlen packed layout.
4. KV uses paged cache.
5. Device must be SM90.
6. Verification is causal on the target side.

Files to update:

- `vllm/v1/attention/backends/tree_attn.py`
- likely a small wrapper/helper module under `vllm/v1/attention/ops/` or backend utils

Tasks:

1. Add optional import/wrapper around the Optimus kernel.
2. Extend `TreeAttentionMetadata` to carry a bool/int tree mask in addition to `tree_attn_bias`.
3. Add a helper to build a kernel-friendly tree mask from the same tree topology used for `qq_bias`.
4. In `TreeAttentionImpl.forward()`:
   - keep prefill path unchanged
   - in decode path, if supported, call the Optimus kernel
   - otherwise fall back to `unified_attention(... qq_bias=...)`
5. Add compatibility checks:
   - SM90 only
   - dtype supported
   - KV cache/page layout compatible
   - no unsupported output scaling path
6. Add runtime logging/debug switch so it is obvious which verifier path is active.

Validation:

1. Compare outputs of:
   - current vLLM tree verifier
   - Optimus tree kernel verifier
2. Require close numerical agreement on deterministic tests.
3. Benchmark decode verification latency on SM90.

Exit criteria:

- SM90 tree verification can use the Optimus kernel with a safe fallback path.

### Step 7: Add end-to-end tests

Objective:

- Cover the new functionality with focused regression tests.

Suggested test areas:

1. Config parsing:
   - `head_type`
   - `tree_width`
   - `tree_draft`
2. DFlash proposer correctness:
   - linear bidirectional
   - linear causal
   - tree crossproduct
3. Tree verifier correctness:
   - fallback verifier
   - Optimus verifier when available
4. Acceptance/correction alignment:
   - accepted path
   - correction token
   - KV-cache selection

Files likely involved:

- `tests/v1/spec_decode/test_eagle.py`
- new DFlash-specific spec decode tests
- `tests/v1/spec_decode/test_tree_attention.py`

Exit criteria:

- New tests cover causal head, crossproduct tree construction, and verifier equivalence.

### Step 8: Add profiling and benchmarking support

Objective:

- Make it easy to compare new modes.

Files to update:

- `examples/offline_inference/dflash_profiling.py`
- wrapper scripts under `examples/offline_inference/`

Tasks:

1. Add `head_type` to profiling CLI.
2. Add tree settings to profiling CLI:
   - `tree_width`
   - `max_tree_budget`
   - `tree_draft`
3. Add mode labels to reports:
   - bidirectional-linear
   - causal-linear
   - bidirectional-tree
   - causal-tree
4. If the Optimus verifier is integrated, add the verifier backend label to output reports.

Exit criteria:

- Performance comparisons can be run with consistent reports across all modes.

### Step 9: Optional entropy-guided tree drafting

Objective:

- Add parity with the reference repo’s iterative tree refinement.

Tasks:

1. Implement conditional re-draft passes for the best path.
2. Add prune-and-regrow logic.
3. Make per-request dynamic trees compatible with verifier metadata.
4. Re-check whether shared-tree assumptions still hold.

Risk:

- This is the highest-complexity part and should not block causal head or crossproduct tree support.

Exit criteria:

- Entropy-guided drafting is isolated, tested, and optional.

## File-Level Work Breakdown

### Core DFlash files

- `vllm/v1/spec_decode/dflash.py`
  - causal mode resolution
  - DFlash tree proposal path
  - routing between linear/tree modes

- `vllm/model_executor/models/qwen3_dflash.py`
  - causal draft attention support
  - explicit masking fallback if metadata-only control is insufficient

### Shared speculative config

- `vllm/config/speculative.py`
  - new DFlash runtime knobs
  - validation rules

- `vllm/engine/arg_utils.py`
  - expose config surface cleanly

### Tree verifier backend

- `vllm/v1/attention/backends/tree_attn.py`
  - metadata extension for kernel-friendly tree mask
  - optional Optimus kernel dispatch
  - fallback path retention

- `vllm/v1/attention/ops/...`
  - small integration wrapper if needed

### Tests

- `tests/v1/spec_decode/test_eagle.py`
- `tests/v1/spec_decode/test_tree_attention.py`
- new DFlash-specific tests as needed

## Validation Matrix

Minimum matrix to run before calling the feature complete:

1. Bidirectional linear DFlash
2. Causal linear DFlash
3. Bidirectional tree DFlash, fallback verifier
4. Causal tree DFlash, fallback verifier
5. Bidirectional tree DFlash, Optimus verifier on SM90
6. Causal tree DFlash, Optimus verifier on SM90

For each case, check:

1. No crashes
2. Output stability on deterministic decoding
3. Acceptance behavior is sensible
4. Throughput regression is understood
5. Fallback path still works when the kernel is unavailable

## Main Risks

### Risk 1: Reusing the wrong tree proposal algorithm

Mitigation:

- Implement a DFlash-specific tree proposal path instead of forcing EAGLE tree drafting semantics onto DFlash.

### Risk 2: Mismatch between tree proposal node order and verifier metadata

Mitigation:

- Define one canonical BFS ordering and test it explicitly.

### Risk 3: Backend-specific causal behavior differences

Mitigation:

- Gate unsupported paths clearly and add backend-specific tests.

### Risk 4: Kernel integration becomes correctness-critical too early

Mitigation:

- First land correctness using the existing verifier, then layer the Optimus kernel as an optimization.

## Recommended Order Of Execution

1. Add config knobs.
2. Land causal linear DFlash.
3. Land DFlash crossproduct tree proposal.
4. Verify tree mode using current vLLM verifier.
5. Integrate the Optimus SM90 verifier kernel as an optional fast path.
6. Extend profiling scripts.
7. Optionally add entropy-guided drafting.

## Definition Of Done

The work is complete when:

1. DFlash supports `head_type=auto|bidirectional|causal`.
2. DFlash supports crossproduct tree inference for both bidirectional and causal heads.
3. Tree verification works with existing vLLM verifier everywhere.
4. SM90 can optionally use the Optimus paged-tree verifier kernel.
5. Tests cover the new modes.
6. Profiling scripts can benchmark the new modes cleanly.
