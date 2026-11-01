# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-cache hits for EAGLE / MTP drafter KV groups with
``draft_prefix_hit="verify_next_tokens"``: a cached drafter block is reused only
if the request has the same tokens past it as the request that wrote it, and
no block is dropped."""

import pytest
import torch

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    DraftTail,
    DraftTailCheck,
    KVCacheBlock,
    draft_tail_of,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

from .test_prefix_caching import make_kv_cache_manager, make_request

pytestmark = pytest.mark.cpu_test

BLOCK = 16


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def _config(spec_types: list[str], num_blocks: int = 64) -> KVCacheConfig:
    groups = []
    for i, kind in enumerate(spec_types):
        if kind == "full":
            spec = FullAttentionSpec(
                block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
            )
            groups.append(KVCacheGroupSpec([f"attn{i}"], spec, is_eagle_group=True))
        else:
            spec = MambaSpec(
                block_size=BLOCK,
                shapes=(1, 1),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            )
            groups.append(KVCacheGroupSpec([f"mamba{i}"], spec))
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )


def _manager(spec_types: list[str], mode: str, lookahead: int = 1):
    return make_kv_cache_manager(
        _config(spec_types),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=BLOCK,
        use_eagle=True,
        num_prefill_lookahead=lookahead,
        verify_draft_tails=mode == "verify_next_tokens",
    )


def _prefill(manager, req, chunks: list[int]) -> None:
    """Schedule the request's prompt in the given chunk sizes (each chunk end
    is where an align-mode Mamba state is materialized and cached)."""
    computed_blocks, num_hit, _ = manager.get_computed_blocks(req)
    for i, n in enumerate(chunks):
        if i == 0:
            blocks = manager.allocate_slots(req, n, num_hit, computed_blocks)
            req.num_computed_tokens = num_hit
        else:
            blocks = manager.allocate_slots(req, n)
        assert blocks is not None
        req.num_computed_tokens += n
        manager.new_step_starts()


def _hit(manager, req) -> int:
    _, num_computed, _ = manager.get_computed_blocks(req)
    return num_computed


PREFIX = [i // BLOCK for i in range(3 * BLOCK)]  # 3 full blocks
SHARED = PREFIX + [77] * 5  # 5 more shared tokens past the third block


def test_draft_tail_of_needs_the_following_tokens():
    tokens = list(range(40))
    assert draft_tail_of(tokens, 32, 1) == DraftTail(32, (32,))
    assert draft_tail_of(tokens, 32, 3) == DraftTail(32, (32, 33, 34))
    assert draft_tail_of(tokens, 40, 1) is None
    assert draft_tail_of(tokens, 38, 3) is None


def test_draft_tail_check_accepts_only_the_same_following_tokens():
    block = KVCacheBlock(block_id=1)
    block.draft_tail = DraftTail(16, (5, 6))
    assert DraftTailCheck([0] * 16 + [5, 6, 7], 2).accepts(block, 16)
    assert not DraftTailCheck([0] * 16 + [5, 9, 7], 2).accepts(block, 16)
    # Recorded for a different block end.
    assert not DraftTailCheck([0] * 32 + [5, 6], 2).accepts(block, 32)
    # Too few following tokens in the request.
    assert not DraftTailCheck([0] * 16 + [5], 2).accepts(block, 16)
    block.draft_tail = None
    assert not DraftTailCheck([0] * 16 + [5, 6], 2).accepts(block, 16)


@pytest.mark.parametrize("mode", ["drop_last_block", "verify_next_tokens"])
def test_hybrid_shared_prefix_hits_its_block_boundary(mode):
    """The shared prefix runs 5 tokens past a block boundary whose Mamba state
    is cached. Verification hits that boundary; the legacy drop needs the next
    block too and so loses the boundary."""
    manager = _manager(["full", "mamba"], mode)
    writer = make_request("w", SHARED + [100] * 20, BLOCK, sha256)
    _prefill(manager, writer, [3 * BLOCK, 25])

    reader = make_request("r", SHARED + [200] * 9, BLOCK, sha256)
    hit = _hit(manager, reader)
    if mode == "verify_next_tokens":
        assert hit == 3 * BLOCK
    else:
        assert hit < 3 * BLOCK


def test_hybrid_rejects_a_block_whose_following_token_differs():
    """The reader diverges exactly at the block boundary: the writer's drafter
    slot at position 47 was computed from a different token 48, so the block
    must be recomputed."""
    manager = _manager(["full", "mamba"], "verify_next_tokens")
    writer = make_request("w", PREFIX + [100] * 20, BLOCK, sha256)
    _prefill(manager, writer, [2 * BLOCK, BLOCK, 20])

    reader = make_request("r", PREFIX + [200] * 20, BLOCK, sha256)
    assert _hit(manager, reader) == 2 * BLOCK

    # A reader with the writer's continuation gets the whole prefix.
    same = make_request("s", PREFIX + [100] * 3, BLOCK, sha256)
    assert _hit(manager, same) == 3 * BLOCK


def test_pure_attention_drafter_hit_is_not_dropped():
    manager = _manager(["full"], "verify_next_tokens")
    writer = make_request("w", SHARED + [100] * 20, BLOCK, sha256)
    _prefill(manager, writer, [len(SHARED) + 20])
    reader = make_request("r", SHARED + [200] * 9, BLOCK, sha256)
    assert _hit(manager, reader) == 3 * BLOCK

    legacy = _manager(["full"], "drop_last_block")
    writer = make_request("w", SHARED + [100] * 20, BLOCK, sha256)
    _prefill(legacy, writer, [len(SHARED) + 20])
    reader = make_request("r", SHARED + [200] * 9, BLOCK, sha256)
    assert _hit(legacy, reader) == 2 * BLOCK


def test_block_is_cached_only_once_its_following_tokens_are_known():
    manager = _manager(["full"], "verify_next_tokens")
    req = make_request("w", PREFIX[: 2 * BLOCK], BLOCK, sha256)
    _prefill(manager, req, [2 * BLOCK])
    attn = manager.coordinator.single_type_managers[0]
    blocks = attn.req_to_blocks["w"]
    # Block 1 ends at the last known token: its drafter slot will be computed
    # from the token sampled in this step, so it is not cached yet.
    assert attn.num_cached_block["w"] == 1
    assert blocks[0].draft_tail == DraftTail(BLOCK, (PREFIX[BLOCK],))
    assert blocks[1].block_hash is None

    req.append_output_token_ids(123)
    assert manager.allocate_slots(req, 1) is not None
    assert attn.num_cached_block["w"] == 2
    assert blocks[1].draft_tail == DraftTail(2 * BLOCK, (123,))


def test_duplicate_blocks_pick_the_one_with_the_matching_tail():
    """Two writers computed the same prefix block (no hit between them) but
    continued differently; each reader gets the copy its drafter slot fits."""
    manager = _manager(["full"], "verify_next_tokens")
    w1 = make_request("w1", PREFIX + [100] * 4, BLOCK, sha256)
    w2 = make_request("w2", PREFIX + [200] * 4, BLOCK, sha256)
    for w in (w1, w2):
        computed_blocks, _, _ = manager.get_computed_blocks(w)
        assert manager.allocate_slots(w, len(w.all_token_ids), 0, None) is not None
    attn = manager.coordinator.single_type_managers[0]
    b1 = attn.req_to_blocks["w1"][2]
    b2 = attn.req_to_blocks["w2"][2]
    assert b1.block_hash == b2.block_hash and b1 is not b2

    r1 = make_request("r1", PREFIX + [100] * 9, BLOCK, sha256)
    blocks, hit, _ = manager.get_computed_blocks(r1)
    assert hit == 3 * BLOCK and blocks.blocks[0][2] is b1
    r2 = make_request("r2", PREFIX + [200] * 9, BLOCK, sha256)
    blocks, hit, _ = manager.get_computed_blocks(r2)
    assert hit == 3 * BLOCK and blocks.blocks[0][2] is b2


def test_multi_token_lookahead_compares_every_following_token():
    manager = _manager(["full"], "verify_next_tokens", lookahead=2)
    writer = make_request("w", PREFIX + [100, 101] + [0] * 6, BLOCK, sha256)
    _prefill(manager, writer, [len(writer.all_token_ids)])
    same = make_request("a", PREFIX + [100, 101, 5, 5], BLOCK, sha256)
    assert _hit(manager, same) == 3 * BLOCK
    second_differs = make_request("b", PREFIX + [100, 102, 5, 5], BLOCK, sha256)
    assert _hit(manager, second_differs) == 2 * BLOCK


def test_eviction_clears_the_draft_tail():
    manager = _manager(["full"], "verify_next_tokens")
    req = make_request("w", SHARED, BLOCK, sha256)
    _prefill(manager, req, [len(SHARED)])
    block = manager.coordinator.single_type_managers[0].req_to_blocks["w"][0]
    assert block.draft_tail is not None
    manager.free(req)
    assert manager.reset_prefix_cache()
    assert block.draft_tail is None
