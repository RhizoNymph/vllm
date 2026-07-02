# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the declarative steering gate schema (host-side).

Covers ``build_steering_gates`` (raw JSON validation + name resolution),
``resolve_gates`` (packed → numpy), the msgpack round-trip through
``RequestMetadata``, and the validation error paths. CPU-only, no engine.
"""

import base64

import msgspec
import numpy as np
import pytest

from vllm.v1.request_metadata import RequestMetadata
from vllm.v1.steering_schema import (
    InlineVec,
    SteeringGate,
    build_steering_gates,
    resolve_gates,
)

HIDDEN = 6


def _pack(vec, layer, hook="post_block"):
    a = np.asarray(vec, dtype=np.float32)
    return {
        hook: {
            "dtype": "float32",
            "shape": [1, a.shape[0]],
            "layer_indices": [layer],
            "data": base64.b64encode(a.tobytes()).decode(),
        }
    }


def _inline(vec, layer, hook="post_block"):
    return {"kind": "inline", "packed": _pack(vec, layer, hook)}


def _add_gate(scope, *, probe=False, strength=1.0, layer=5):
    when = (
        {"kind": "probe", "probe": _inline(np.ones(HIDDEN), layer), "threshold": 0.0}
        if probe
        else {"kind": "always"}
    )
    return {
        "when": when,
        "scope": scope,
        "apply": {
            "kind": "add",
            "steer": _inline(np.ones(HIDDEN) * 2, layer),
            "strength": strength,
        },
    }


def test_build_and_resolve_inline():
    gates = build_steering_gates([_add_gate("this_token", probe=True, strength=3.0)], None)
    assert len(gates) == 1
    res = resolve_gates(gates)
    g = res[0]
    assert g.scope == "this_token"
    assert g.when_kind == "probe"
    assert g.probe_site == (5, "post_block")
    assert g.apply_kind == "add"
    # steer × strength = 2 * 3 = 6
    np.testing.assert_allclose(g.steer_vectors["post_block"][5], np.full(HIDDEN, 6.0))
    np.testing.assert_allclose(g.probe_vec, np.ones(HIDDEN))


def test_attenuate_gate():
    gate = {
        "when": {"kind": "always"},
        "scope": "rest_of_request",
        "apply": {"kind": "attenuate", "strength": 0.25},
    }
    res = resolve_gates(build_steering_gates([gate], None))
    assert res[0].apply_kind == "attenuate"
    assert res[0].strength == 0.25
    assert res[0].steer_vectors is None


def test_request_metadata_msgpack_roundtrip():
    gates = build_steering_gates(
        [_add_gate("rest_of_conversation", probe=True)], None
    )
    rm = RequestMetadata(conversation_id="c1", steering=gates)
    assert rm.is_empty() is False
    dec = msgspec.msgpack.decode(msgspec.msgpack.encode(rm), type=RequestMetadata)
    assert dec.conversation_id == "c1"
    assert len(dec.steering) == 1
    # after decode, sources are still InlineVec and resolvable
    g = resolve_gates(dec.steering)[0]
    assert g.scope == "rest_of_conversation"
    assert isinstance(dec.steering[0].when.probe, InlineVec)


def test_empty_metadata():
    assert RequestMetadata().is_empty() is True
    assert RequestMetadata(conversation_id="x").is_empty() is False
    assert RequestMetadata(steering=[]).is_empty() is False


@pytest.mark.parametrize(
    "raw",
    [
        [{"when": {"kind": "nope"}, "scope": "this_token",
          "apply": {"kind": "attenuate", "strength": 0.0}}],
        [{"when": {"kind": "always"}, "scope": "bogus_scope",
          "apply": {"kind": "attenuate", "strength": 0.0}}],
        [{"when": {"kind": "always"}, "scope": "this_token",
          "apply": {"kind": "weird"}}],
    ],
)
def test_malformed_gates_rejected(raw):
    with pytest.raises(ValueError):
        build_steering_gates(raw, None)


def test_probe_must_be_single_site():
    packed = _pack(np.ones(HIDDEN), 5)
    packed["pre_attn"] = _pack(np.ones(HIDDEN), 6, "pre_attn")["pre_attn"]
    raw = [{
        "when": {"kind": "probe", "probe": {"kind": "inline", "packed": packed},
                 "threshold": 0.0},
        "scope": "this_token",
        "apply": {"kind": "add", "steer": _inline(np.ones(HIDDEN), 5)},
    }]
    with pytest.raises(ValueError, match="exactly one"):
        build_steering_gates(raw, None)


def test_invalid_hook_rejected():
    raw = [{
        "when": {"kind": "always"},
        "scope": "this_token",
        "apply": {"kind": "add", "steer": _inline(np.ones(HIDDEN), 5, "bogus_hook")},
    }]
    with pytest.raises(ValueError, match="hook"):
        build_steering_gates(raw, None)


def test_named_source_without_registry_rejected():
    raw = [{
        "when": {"kind": "always"},
        "scope": "this_token",
        "apply": {"kind": "add", "steer": {"kind": "name", "name": "x"}},
    }]
    with pytest.raises(ValueError, match="registry"):
        build_steering_gates(raw, None)


def test_none_and_empty_passthrough():
    assert build_steering_gates(None, None) is None
    assert build_steering_gates([], None) is None
    assert resolve_gates(None) is None
    assert resolve_gates([]) is None


def test_gate_type_is_msgspec_struct():
    g = build_steering_gates([_add_gate("next_step")], None)[0]
    assert isinstance(g, SteeringGate)


def test_attenuate_this_token_probe_rejected():
    # The substrate cannot damp a row only when the probe fires this token
    # (the per-row monitor gates toward zero when the probe is LOW).
    raw = [{
        "when": {"kind": "probe", "probe": _inline(np.ones(HIDDEN), 5),
                 "threshold": 0.0},
        "scope": "this_token",
        "apply": {"kind": "attenuate", "strength": 0.5},
    }]
    with pytest.raises(ValueError, match="attenuate"):
        build_steering_gates(raw, None)


def test_attenuate_this_token_always_still_allowed():
    # Only the probe variant is unsupported; always+this_token+attenuate is fine.
    raw = [{
        "when": {"kind": "always"},
        "scope": "this_token",
        "apply": {"kind": "attenuate", "strength": 0.5},
    }]
    gates = build_steering_gates(raw, None)
    assert len(gates) == 1


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
def test_probe_bad_sharpness_rejected(bad):
    raw = [{
        "when": {"kind": "probe", "probe": _inline(np.ones(HIDDEN), 5),
                 "threshold": 0.0, "sharpness": bad},
        "scope": "this_token",
        "apply": {"kind": "add", "steer": _inline(np.ones(HIDDEN), 5)},
    }]
    with pytest.raises(ValueError, match="sharpness"):
        build_steering_gates(raw, None)


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_probe_non_finite_threshold_rejected(bad):
    raw = [{
        "when": {"kind": "probe", "probe": _inline(np.ones(HIDDEN), 5),
                 "threshold": bad},
        "scope": "rest_of_request",
        "apply": {"kind": "add", "steer": _inline(np.ones(HIDDEN), 5)},
    }]
    with pytest.raises(ValueError, match="threshold"):
        build_steering_gates(raw, None)
