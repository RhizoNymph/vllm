# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the declarative steering gate schema (host-side).

Covers ``build_steering_gates`` (raw JSON validation + name resolution),
``resolve_gates`` (packed → numpy), the msgpack round-trip through
``RequestMetadata``, and the validation error paths. CPU-only, no engine.
"""

import base64
import contextlib
import logging

import msgspec
import numpy as np
import pytest

import vllm.v1.steering_schema as steering_schema
from vllm.v1.request_metadata import RequestMetadata
from vllm.v1.steering_schema import (
    InlineVec,
    SteeringGate,
    build_steering_gates,
    resolve_gates,
    resolve_gates_safe,
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


# --------------------------------------------------------------------------
# Fail-safe admission (``resolve_gates_safe``) — a malformed payload that
# bypassed the frontend dry-run must be skipped, not crash the engine core.
# --------------------------------------------------------------------------


def _bad_gate(bad_packed):
    """A structurally valid ``SteeringGate`` whose inline steer packed blob is
    malformed. Struct constructors do not deep-validate the packed dict, so
    the failure surfaces only when ``resolve_gates`` unpacks it."""
    from vllm.v1.steering_schema import AddApply, AlwaysWhen, GateScope

    return SteeringGate(
        when=AlwaysWhen(),
        scope=GateScope.THIS_TOKEN,
        apply=AddApply(steer=InlineVec(packed=bad_packed), strength=1.0),
    )


def _packed_with(**overrides):
    blob = {
        "dtype": "float32",
        "shape": [1, HIDDEN],
        "layer_indices": [5],
        "data": base64.b64encode(
            np.ones(HIDDEN, dtype=np.float32).tobytes()
        ).decode(),
    }
    blob.update(overrides)
    for k in overrides:
        if overrides[k] is _MISSING:
            del blob[k]
    return {"post_block": blob}


_MISSING = object()


@pytest.fixture(autouse=True)
def _reset_resolve_warned():
    steering_schema._resolve_failure_warned = False
    yield
    steering_schema._resolve_failure_warned = False


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _capture_warnings():
    # vLLM's logger does not propagate to root, so caplog can't see it; attach
    # a handler directly to the module logger.
    handler = _CaptureHandler()
    steering_schema.logger.addHandler(handler)
    try:
        yield handler
    finally:
        steering_schema.logger.removeHandler(handler)


@pytest.mark.parametrize(
    "bad_packed",
    [
        _packed_with(dtype="garbage"),  # np.dtype("garbage") -> TypeError
        _packed_with(data="!!!not-base64!!!"),  # decode/length -> ValueError
        _packed_with(dtype=_MISSING),  # blob["dtype"] -> KeyError
        _packed_with(shape="not-a-list"),  # wrong type
        _packed_with(shape=[HIDDEN]),  # not 2-D -> ValueError
    ],
)
def test_resolve_gates_safe_skips_malformed(bad_packed):
    gates = [_bad_gate(bad_packed)]
    # Sanity: the raw resolver raises on this payload (would kill the runner).
    with pytest.raises(Exception):
        resolve_gates(gates)
    with _capture_warnings() as cap:
        out = resolve_gates_safe(gates, req_id="req-abc")
    assert out is None
    assert any("req-abc" in m for m in cap.messages)
    assert steering_schema._resolve_failure_warned is True


def test_resolve_gates_safe_warns_once():
    gates = [_bad_gate(_packed_with(dtype="garbage"))]
    with _capture_warnings() as cap:
        resolve_gates_safe(gates, req_id="r1")
        resolve_gates_safe(gates, req_id="r2")
    warnings = [m for m in cap.messages if "gates skipped" in m]
    assert len(warnings) == 1


def test_resolve_gates_safe_passes_valid_through():
    gates = build_steering_gates(
        [_add_gate("this_token", probe=True, strength=3.0)], None
    )
    with _capture_warnings() as cap:
        out = resolve_gates_safe(gates, req_id="ok")
    ref = resolve_gates(gates)
    assert out is not None and len(out) == len(ref) == 1
    assert out[0].scope == ref[0].scope == "this_token"
    np.testing.assert_allclose(
        out[0].steer_vectors["post_block"][5], np.full(HIDDEN, 6.0)
    )
    assert steering_schema._resolve_failure_warned is False
    assert not any("gates skipped" in m for m in cap.messages)


def test_resolve_gates_safe_empty_passthrough():
    assert resolve_gates_safe(None) is None
    assert resolve_gates_safe([]) is None


# --------------------------------------------------------------------------
# Frontend: non-ValueError unpack failures normalize to ValueError (HTTP 400,
# not 500). The serving handlers catch only ValueError.
# --------------------------------------------------------------------------


def test_build_gates_bad_dtype_raises_valueerror():
    raw = [{
        "when": {"kind": "always"},
        "scope": "this_token",
        "apply": {
            "kind": "add",
            "steer": {"kind": "inline", "packed": _packed_with(dtype="garbage")},
        },
    }]
    # Without the fix this leaks TypeError (np.dtype) -> HTTP 500.
    with pytest.raises(ValueError, match="steering gate spec"):
        build_steering_gates(raw, None)
