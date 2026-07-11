# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for activation-patching-aware prefix cache key generation.

A patched request's KV differs from the unpatched run at every position >=
its lowest patched position, so those blocks must hash differently from the
vanilla token chain (else a later unpatched request could be served patched
KV — silent poisoning). Blocks strictly below the floor stay shareable.
"""

from types import SimpleNamespace

import pytest

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256_cbor
from vllm.v1.core.kv_cache_utils import (
    _gen_patch_extra_hash_keys,
    hash_block_tokens,
    init_none_hash,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256_cbor)


def _entry(layer=3, hook="post_block", dest=20, run="clean", src=20, alpha=1.0):
    return {
        "layer": layer,
        "hook": hook,
        "dest_position": dest,
        "source_run": run,
        "source_position": src,
        "alpha": alpha,
    }


def _req(patch):
    return SimpleNamespace(sampling_params=SamplingParams(patch=patch))


class TestPatchKvTaint:
    def test_none_without_patch(self):
        assert SamplingParams().patch_kv_taint is None

    def test_min_dest_position(self):
        sp = SamplingParams(patch=[_entry(dest=20), _entry(layer=5, dest=7)])
        assert sp.patch_kv_taint[0] == 7

    def test_deterministic_and_order_insensitive(self):
        a = SamplingParams(patch=[_entry(dest=20), _entry(layer=5, dest=7)])
        b = SamplingParams(patch=[_entry(layer=5, dest=7), _entry(dest=20)])
        assert a.patch_kv_taint == b.patch_kv_taint

    def test_spec_sensitivity(self):
        base = SamplingParams(patch=[_entry()]).patch_kv_taint[1]
        assert SamplingParams(patch=[_entry(alpha=0.5)]).patch_kv_taint[1] != base
        assert SamplingParams(patch=[_entry(layer=4)]).patch_kv_taint[1] != base
        assert SamplingParams(patch=[_entry(run="other")]).patch_kv_taint[1] != base


class TestGenPatchExtraHashKeys:
    def test_no_patch_no_keys(self):
        req = SimpleNamespace(sampling_params=SamplingParams())
        assert _gen_patch_extra_hash_keys(req, 0, 16) == []

    def test_no_sampling_params_no_keys(self):
        req = SimpleNamespace(sampling_params=None)
        assert _gen_patch_extra_hash_keys(req, 0, 16) == []

    def test_block_below_floor_untainted(self):
        # patch at position 20 -> block [0, 16) is byte-identical to unpatched
        req = _req([_entry(dest=20)])
        assert _gen_patch_extra_hash_keys(req, 0, 16) == []

    def test_block_containing_floor_tainted(self):
        req = _req([_entry(dest=20)])
        keys = _gen_patch_extra_hash_keys(req, 16, 32)
        assert keys == [req.sampling_params.patch_kv_taint[1]]

    def test_block_after_floor_tainted(self):
        # attention propagates the patch forward: all later blocks tainted
        req = _req([_entry(dest=20)])
        assert _gen_patch_extra_hash_keys(req, 32, 48) != []

    def test_floor_at_block_boundary(self):
        # floor == end -> block [0, 16) with patch at 16 is untainted;
        # block [16, 32) is tainted.
        req = _req([_entry(dest=16)])
        assert _gen_patch_extra_hash_keys(req, 0, 16) == []
        assert _gen_patch_extra_hash_keys(req, 16, 32) != []

    def test_distinct_specs_distinct_block_hashes(self):
        req_a = _req([_entry(layer=3, dest=4)])
        req_b = _req([_entry(layer=7, dest=4)])
        tokens = list(range(16))
        ha = hash_block_tokens(
            sha256_cbor,
            None,
            tokens,
            tuple(_gen_patch_extra_hash_keys(req_a, 0, 16)),
        )
        hb = hash_block_tokens(
            sha256_cbor,
            None,
            tokens,
            tuple(_gen_patch_extra_hash_keys(req_b, 0, 16)),
        )
        h_clean = hash_block_tokens(sha256_cbor, None, tokens, None)
        assert ha != hb
        assert ha != h_clean and hb != h_clean

    def test_identical_specs_share_block_hashes(self):
        req_a = _req([_entry()])
        req_b = _req([_entry()])
        tokens = list(range(16))
        ha = hash_block_tokens(
            sha256_cbor,
            None,
            tokens,
            tuple(_gen_patch_extra_hash_keys(req_a, 16, 32)),
        )
        hb = hash_block_tokens(
            sha256_cbor,
            None,
            tokens,
            tuple(_gen_patch_extra_hash_keys(req_b, 16, 32)),
        )
        assert ha == hb
