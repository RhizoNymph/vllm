# Verified prefix-cache hits for EAGLE / MTP drafter KV

`--draft-prefix-hit verify_next_tokens` (`CacheConfig.draft_prefix_hit`)

## Scope

- How a prefix-cache hit treats the KV cache of an EAGLE / MTP drafter's
  full-attention layers, with the unitary and the hybrid (attention + Mamba /
  GatedDeltaNet) coordinators.
- Recording, per cached drafter block, the tokens the drafter consumed past the
  block, and accepting a cached block only when the request has the same
  tokens there.

Not in scope: sliding-window drafter groups (they keep the legacy drop), the
fine-grained partial-hash hits of `--prefix-match-unit` for drafter groups
(not taken in this mode), KV connectors' per-group lookups
(`find_longest_cache_hit_per_group` keeps the legacy drop), the drafter's
proposals themselves.

## Why

A drafter's KV at position `p` is computed from the target's hidden state at
`p` and the token at `p + 1` (multi-module MTP: up to `p + k`). The block hash
of the block holding `p` covers tokens up to the block end only, so the last
slot(s) of a cached block depend on how the *writer* continued. The legacy
policy (`drop_last_block`) requires one more block to match and then drops it.
On a hybrid model whose recurrent state can only resume at block boundaries,
that loses a whole block of every shared prefix: on Qwen3.6-27B (block 416) a
2,419-token shared system prompt hit 1,664 tokens instead of 2,080. It is also
only a heuristic: the extra block may have been cached by a request other than
the one that wrote the reused block.

With `verify_next_tokens` nothing is dropped and every reused drafter slot was
computed from exactly the request's own following tokens. Target outputs are
unaffected either way (the drafter only proposes); the mode changes how much
prefix is reused and makes the reused drafter KV exact.

## Data and control flow

1. `Scheduler.__init__` reads `cache_config.draft_prefix_hit` and passes
   `verify_draft_tails` to `KVCacheManager` -> `get_kv_cache_coordinator`.
2. `KVCacheCoordinator.__init__` sets `draft_lookahead = max(1,
   num_prefill_lookahead)` when the mode is on, caching is enabled and a
   drafter is present. After the managers' `use_eagle` bits are set
   (`UnitaryKVCacheCoordinator.__init__`,
   `HybridKVCacheCoordinator.verify_and_split_kv_cache_groups`),
   `_enable_draft_tail_verification` sets `manager.draft_lookahead` on every
   drafter `FullAttentionManager`.
3. Caching: `SingleTypeKVCacheManager.cache_blocks` caps `num_tokens` at
   `request.num_tokens - draft_lookahead` for such managers, so a block is
   cached only once the tokens past it are known, and passes `draft_lookahead`
   to `BlockPool.cache_full_blocks`, which stores
   `KVCacheBlock.draft_tail = DraftTail(end, all_token_ids[end:end+lookahead])`
   on each block it gives a primary hash. `KVCacheBlock.reset_hash` clears it
   (eviction, reset, partial -> full promotion).
4. Lookup: `KVCacheManager.get_computed_blocks` passes
   `token_ids=request.all_token_ids` to `coordinator.find_longest_cache_hit`,
   which builds a `DraftTailCheck(token_ids, draft_lookahead)`.
   - Unitary: passed to `FullAttentionManager.find_longest_cache_hit`.
   - Hybrid: passed for full-attention spec groups that would drop; their
     one-block "eagle margin" is not added. Mamba groups are unchanged and
     still bound the hit to a cached state boundary.
5. `FullAttentionManager.find_longest_cache_hit` (phase 1) asks
   `BlockPool.get_cached_block(hash, group_ids, accept)` with
   `accept = DraftTailCheck.accepts(end=(i + 1) * block_size)`;
   `BlockHashToBlockMap.get_one_block` returns the first duplicate whose tail
   matches. The run stops at the first block with no accepted copy. Phase 2
   (partial tail) is skipped and no block is dropped.

## Files

| file | role |
|---|---|
| `vllm/config/cache.py` | `DraftPrefixHit`, `CacheConfig.draft_prefix_hit` |
| `vllm/engine/arg_utils.py` | `--draft-prefix-hit` |
| `vllm/v1/core/kv_cache_utils.py` | `DraftTail`, `DraftTailCheck.accepts`, `draft_tail_of`, `KVCacheBlock.draft_tail` |
| `vllm/v1/core/block_pool.py` | `get_one_block(accept)`, `get_cached_block(accept)`, `cache_full_blocks(draft_lookahead)` |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `draft_lookahead`, capped caching, verified full-attention lookup |
| `vllm/v1/core/kv_cache_coordinator.py` | `verify_draft_tails`, `_enable_draft_tail_verification`, `_draft_tail_check`, margin removal |
| `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/sched/scheduler.py` | plumbing |
| `tests/v1/core/test_draft_prefix_hit.py` | behaviour tests |

## Invariants

- A drafter block is reused by a request only if
  `block.draft_tail == DraftTail(end, request.all_token_ids[end:end+lookahead])`.
- A drafter block is cached only when the tokens past it are known; until then
  it has no hash (it is cached on a later step).
- `draft_tail` lives and dies with the block's primary hash.
- The default (`drop_last_block`) behaviour is unchanged.
- Target-model KV and Mamba / GDN states are never affected: they are keyed by
  the block hash alone as before.
