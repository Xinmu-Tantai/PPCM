# DFlash Causal Tree — CPU Optimization Cookbook

## Current State

- Async scheduling is **disabled** for tree mode (`tree_width > 1`) because the `AsyncScheduler` uses fixed-length placeholders (`num_spec_tokens`), not the variable-size tree budget.
- Sync scheduling works: `take_draft_token_ids` → `update_draft_token_ids` propagates the full tree (254 tokens) to the scheduler each step.
- The main bottleneck is CPU overhead between decode steps, not scheduling itself.

---

## 1. Async Scheduling for Tree Mode

### Why it's hard

Async scheduling pre-schedules step N+1 while step N runs on GPU. It assumes a fixed draft length (`num_spec_tokens`) so it can insert placeholders without waiting for the drafter. Tree mode produces variable-length drafts (up to `tree_budget`) with tree topology metadata that the scheduler needs for `_is_full_dflash_tree_request` checks.

### How to implement

**Option A — Fixed-budget async (recommended first step)**

For `bs=1` with a fixed `max_tree_budget`, the tree always fills to the budget. Specialize `AsyncScheduler._update_after_schedule` to use `tree_budget - 1` placeholders when the request has `spec_tree_metadata`. Patch `_is_full_dflash_tree_request` to accept placeholder metadata during async pre-scheduling. After model execution, replace placeholders with real tokens/metadata via `_copy_draft_token_ids_to_cpu`.

Complexity: moderate. The key invariant is that tree budget is always fully used when `bs=1`.

**Option B — Deferred-schedule async**

Don't pre-schedule spec tokens. Run `take_draft_token_ids` → `update_draft_token_ids` between GPU completion and the next schedule call, but overlap GPU compute with non-scheduling CPU work (metadata construction, acceptance, KV compaction). This is a lighter form of pipelining.

Complexity: low, but smaller latency benefit.

**Option C — Full variable-length async**

Generalize `AsyncScheduler` to handle variable-length spec tokens. Requires the scheduler to retroactively adjust `num_scheduled_tokens` after draft tokens arrive. Significant refactor of `SchedulerOutput` and downstream consumers.

Complexity: high. Defer until `bs > 1` is needed.

### Recommendation

Start with **Option B** (overlap CPU work with GPU), then move to **Option A** once stable.

---

## 2. CPU Overhead Optimizations (by effort)

### Tier 1 — Drop-in fixes (< 1 day each)

| Optimization | Where | Impact |
|---|---|---|
| **Preallocate metadata tensors** — reuse `parent_indices`, `depths`, `cu_query_lens` buffers across steps instead of rebuilding from Python lists | `_calc_dflash_tree_spec_decode_metadata` | Eliminates per-step tensor alloc |
| **Cache causal fallback bias** — for non-tree requests in a mixed batch, the causal bias is the same every step; cache by `query_len` | `build_causal_attention_bias` | Avoids redundant O(n²) bias construction |
| **Reuse block-diagonal bias storage** — preallocate the max-size dense bias and write in-place | `build_block_diagonal_attention_bias` | Eliminates largest allocation |
| **Use pinned CPU tensors for tree metadata** — pre-pin `parent_indices`, `token_ids` on CPU for faster H2D transfer | metadata construction | Reduces H2D latency |

### Tier 2 — Moderate refactors (1–3 days each)

| Optimization | Where | Impact |
|---|---|---|
| **Move tree acceptance to C++/Triton** — `tree_accept` is pure Python with per-node loops; a fused kernel or C++ extension eliminates Python overhead | `_sample_dflash_tree` | Major win for large trees |
| **Batched KV compaction** — replace per-request loops + `.clone()` with a single batched gather/scatter | `_compact_dflash_tree_kv_cache` | Proportional to `bs` |
| **Cache tree bias by topology hash** — `tuple(parent_indices)` as key; skip bias rebuild when the tree shape repeats | `build_attention_bias_from_parents` | Effective when `max_draft_passes=1` (same topology often repeats) |
| **Keep accepted-path in tensor form** — avoid `tolist()` round-trips; use `torch.index_select` for token gathering | `_sample_dflash_tree` | Reduces Python↔CUDA sync |

### Tier 3 — Architectural changes (1+ week)

| Optimization | Where | Impact |
|---|---|---|
| **Sparse tree attention** — pass parent/depth structure directly to the kernel instead of materializing dense Q×Q bias | `TreeAttentionBackend` | Eliminates O(budget²) CPU work; essential for larger budgets |
| **Persistent tree metadata object** — construct once at tree creation, update in-place on prune/regrow; never rebuild from scratch | `DFlashTreeSpecDecodeMetadata` | Removes the entire metadata-rebuild hotspot |
| **GPU-side tree acceptance** — run verification + acceptance as a single fused kernel; return only the accepted path length to CPU | new kernel | Removes the biggest CPU bottleneck entirely |

---

## 3. Recommended Execution Order

1. Preallocate metadata tensors + reuse block-diagonal bias (Tier 1)
2. Implement Option B async overlap (Section 1)
3. Cache tree bias by topology hash (Tier 2)
4. Move tree acceptance to C++/Triton (Tier 2)
5. Batched KV compaction (Tier 2)
6. Option A fixed-budget async scheduling (Section 1)
7. Sparse tree attention (Tier 3)

---

## 4. Measurement

Profile with `py-spy` or `torch.profiler` focusing on:
- Wall time between consecutive `execute_model` calls (the "CPU gap")
- Time in `_calc_dflash_tree_spec_decode_metadata`
- Time in `_sample_dflash_tree`
- Time in `_compact_dflash_tree_kv_cache`
- Tensor allocation count per step (via `torch.cuda.memory_stats`)
