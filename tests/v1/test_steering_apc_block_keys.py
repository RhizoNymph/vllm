# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-hash keying for the dynamic-steering APC notification.

Validates the load-bearing invariant
(docs/design/dynamic_steering_apc_notification.md): a decode KV block is
keyed by the effective decode steering signature in force when it was
produced, applied *forward-only* — a mid-stream signature change keys only
new blocks, never retroactively rekeying the clean prefix. Consequences
tested:
  - admitted-only and dynamically-steered requests with identical tokens
    get DIFFERENT decode block keys (no false reuse);
  - two requests under the IDENTICAL signature + tokens get the SAME keys
    (reuse preserved);
  - ``update_decode_steering_signature`` does not disturb already-hashed
    blocks (forward-only).
"""

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256_cbor
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

BLOCK = 4
SIG = 0x1234ABCD  # a stand-in dynamic effective-decode signature

init_none_hash(sha256_cbor)


def _req(rid: str) -> Request:
    return Request(
        request_id=rid,
        prompt_token_ids=[1, 2, 3, 4],  # exactly one prompt block
        sampling_params=SamplingParams(max_tokens=64),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256_cbor),
    )


def _seq_append(req: Request, n: int) -> None:
    # Deterministic decode tokens, identical content across requests.
    base = len(req._output_token_ids)
    for i in range(n):
        req.append_output_token_ids(1000 + base + i)


def _decode_blocks(req: Request) -> list:
    # block 0 is the prompt block; 1+ are decode blocks.
    return req.block_hashes[1:]


def test_admitted_vs_dynamic_keys_differ():
    a = _req("a")  # admitted-only (decode sig 0)
    b = _req("b")
    b.update_decode_steering_signature(SIG)  # dynamic from first decode block
    for r in (a, b):
        _seq_append(r, 8)  # two decode blocks
    assert _decode_blocks(a) != _decode_blocks(b)
    # Prompt block (prefill key) is unaffected by decode steering.
    assert a.block_hashes[0] == b.block_hashes[0]


def test_identical_signature_same_keys():
    b = _req("b")
    c = _req("c")
    b.update_decode_steering_signature(SIG)
    c.update_decode_steering_signature(SIG)
    _seq_append(b, 8)
    _seq_append(c, 8)
    assert _decode_blocks(b) == _decode_blocks(c)


def test_forward_only_does_not_rekey_clean_prefix():
    # reqD: first decode block under admitted (0), then switch to SIG.
    a = _req("a")  # all-admitted reference
    d = _req("d")
    _seq_append(a, 8)
    _seq_append(d, 4)  # first decode block under admitted
    first_decode_admitted = d.block_hashes[1]
    assert first_decode_admitted == a.block_hashes[1]  # same so far
    d.update_decode_steering_signature(SIG)  # forward-only
    _seq_append(d, 4)  # second decode block under SIG
    # The already-hashed first decode block is untouched (no retroactive rekey).
    assert d.block_hashes[1] == first_decode_admitted
    # The new block diverges from the all-admitted reference.
    assert d.block_hashes[2] != a.block_hashes[2]
    # Chain sensitivity: it also differs from a request steered for BOTH
    # decode blocks — a different steering *history* is a different prefix,
    # hence a different key (block hashes chain through the parent).
    b = _req("b")
    b.update_decode_steering_signature(SIG)
    _seq_append(b, 8)
    assert d.block_hashes[2] != b.block_hashes[2]
    # A request with the IDENTICAL history (admitted block 1, then SIG for
    # block 2) reproduces d's keys exactly — correct reuse.
    f = _req("f")
    _seq_append(f, 4)
    f.update_decode_steering_signature(SIG)
    _seq_append(f, 4)
    assert f.block_hashes == d.block_hashes


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
