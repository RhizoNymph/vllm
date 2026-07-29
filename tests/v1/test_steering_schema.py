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
    ClampApply,
    InlineVec,
    SteeringGate,
    build_steering_gates,
    resolve_gates,
    resolve_gates_safe,
    validate_clamp_gate_support,
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
    # Inline sources are only allowed on ephemeral scopes now; use
    # rest_of_request (rest_of_conversation requires a NamedVec, tested below).
    gates = build_steering_gates([_add_gate("rest_of_request", probe=True)], None)
    rm = RequestMetadata(conversation_id="c1", steering=gates)
    assert rm.is_empty() is False
    dec = msgspec.msgpack.decode(msgspec.msgpack.encode(rm), type=RequestMetadata)
    assert dec.conversation_id == "c1"
    assert len(dec.steering) == 1
    # after decode, sources are still InlineVec and resolvable
    g = resolve_gates(dec.steering)[0]
    assert g.scope == "rest_of_request"
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


# --------------------------------------------------------------------------
# Named vectors: the frontend leaves NamedVec un-inflated (validates
# existence only); the worker registry resolves it. rest_of_conversation add
# requires a name; ephemeral scopes keep inline.
# --------------------------------------------------------------------------


def _registry(**named):
    """A worker registry populated with ``{name: (kind, vec, layer)}`` entries."""
    from vllm.v1.worker.steering_vector_registry import (
        WorkerSteeringVectorRegistry,
    )

    reg = WorkerSteeringVectorRegistry()
    for name, (kind, vec, layer) in named.items():
        reg.register(name, kind, _pack(vec, layer))
    return reg


def _name_gate(scope, *, probe=False, strength=1.0):
    when = (
        {"kind": "probe", "probe": {"kind": "name", "name": "p"}, "threshold": 0.0}
        if probe
        else {"kind": "always"}
    )
    return {
        "when": when,
        "scope": scope,
        "apply": {
            "kind": "add",
            "steer": {"kind": "name", "name": "s"},
            "strength": strength,
        },
    }


def test_named_vec_passes_through_uninflated():
    from vllm.v1.steering_schema import NamedVec

    reg = _registry(
        s=("steer", np.ones(HIDDEN) * 2, 5), p=("probe", np.ones(HIDDEN), 5)
    )
    gates = build_steering_gates([_name_gate("this_token", probe=True)], reg)
    assert len(gates) == 1
    # The wire form still carries the names — no inline inflation.
    assert isinstance(gates[0].apply.steer, NamedVec)
    assert gates[0].apply.steer.name == "s"
    assert isinstance(gates[0].when.probe, NamedVec)
    assert gates[0].when.probe.name == "p"


def test_named_vec_resolves_against_registry_with_name_and_digest():
    from vllm.config.steering_types import steering_vector_content_digest

    reg = _registry(s=("steer", np.ones(HIDDEN) * 2, 5))
    gates = build_steering_gates([_name_gate("rest_of_request", strength=3.0)], reg)
    res = resolve_gates(gates, reg)[0]
    # steer × strength = 2 * 3 = 6
    np.testing.assert_allclose(res.steer_vectors["post_block"][5], np.full(HIDDEN, 6.0))
    assert res.steer_name == "s"
    assert res.steer_digest == steering_vector_content_digest(
        _pack(np.ones(HIDDEN) * 2, 5)
    )


def test_rest_of_conversation_requires_named_steer():
    reg = _registry(s=("steer", np.ones(HIDDEN), 5))
    # inline steer + rest_of_conversation + add -> rejected
    with pytest.raises(ValueError, match="registered vector name"):
        build_steering_gates([_add_gate("rest_of_conversation")], reg)
    # named steer + rest_of_conversation -> accepted
    gates = build_steering_gates([_name_gate("rest_of_conversation")], reg)
    assert len(gates) == 1


def test_ephemeral_scopes_keep_inline():
    for scope in ("this_token", "next_step", "rest_of_request"):
        gates = build_steering_gates([_add_gate(scope)], None)
        assert len(gates) == 1


def test_unknown_named_vector_rejected_at_frontend():
    reg = _registry(s=("steer", np.ones(HIDDEN), 5))
    raw = [{
        "when": {"kind": "always"},
        "scope": "this_token",
        "apply": {"kind": "add", "steer": {"kind": "name", "name": "missing"}},
    }]
    with pytest.raises(ValueError, match="unknown steer vector name"):
        build_steering_gates(raw, reg)


def test_resolve_gates_safe_skips_unknown_name_gracefully():
    # A NamedVec that is not in the worker registry (race: unregistered
    # between the frontend check and worker admission) must be skipped, not
    # crash the engine core.
    reg = _registry(s=("steer", np.ones(HIDDEN), 5))
    gates = build_steering_gates([_name_gate("rest_of_request")], reg)
    empty = _registry()  # worker no longer has the name
    with _capture_warnings() as cap:
        out = resolve_gates_safe(gates, req_id="racy", registry=empty)
    assert out is None
    assert any("racy" in m for m in cap.messages)


# --------------------------------------------------------------------------
# Clamp-target gates (apply.kind == "clamp"): a FORWARD-COMPAT schema member.
# Per-request clamp gates are unsupported in every engine mode today (the
# clamp op reads the shared row-gate buffer, written only by the GLOBAL
# cross-layer monitor), so the frontend rejects them outright. The wire
# member is kept so the tagged union stays stable when support lands.
# --------------------------------------------------------------------------


def _clamp_gate(scope, *, probe=False, strength=1.0, layer=5):
    when = (
        {"kind": "probe", "probe": _inline(np.ones(HIDDEN), layer), "threshold": 0.0}
        if probe
        else {"kind": "always"}
    )
    return {
        "when": when,
        "scope": scope,
        "apply": {"kind": "clamp", "strength": strength},
    }


def _convert_clamp_gates(raw):
    """Bypass build-time validation: pin the WIRE schema (forward-compat)."""
    return msgspec.convert(raw, type=list[SteeringGate])


def test_clamp_gate_rejected_at_build_regardless_of_mode():
    # Unconditional: no engine mode honors per-request clamp gates today.
    for mode in (None, False, True):
        with pytest.raises(ValueError, match="clamp"):
            build_steering_gates(
                [_clamp_gate("rest_of_request")], None, monitor_writes_gates=mode
            )


def test_clamp_gate_rejection_messages_are_actionable():
    with pytest.raises(ValueError, match="enable_cross_layer_monitor"):
        build_steering_gates(
            [_clamp_gate("rest_of_request")], None, monitor_writes_gates=False
        )
    # Materialized mode: the error explains the global-monitor reality.
    with pytest.raises(ValueError, match="GLOBAL cross-layer monitor"):
        build_steering_gates(
            [_clamp_gate("rest_of_request")], None, monitor_writes_gates=True
        )


def test_validate_clamp_gate_support_rejects_both_modes():
    gates = _convert_clamp_gates([_clamp_gate("rest_of_request")])
    with pytest.raises(ValueError, match="monitor_writes_gates"):
        validate_clamp_gate_support(gates, monitor_writes_gates=False)
    with pytest.raises(ValueError, match="gate_rows"):
        validate_clamp_gate_support(gates, monitor_writes_gates=True)


def test_validate_clamp_gate_support_ignores_non_clamp_gates():
    gates = build_steering_gates([_add_gate("this_token", probe=True)], None)
    # add/attenuate gates are unaffected by the clamp-gate rejection.
    validate_clamp_gate_support(gates, monitor_writes_gates=False)
    validate_clamp_gate_support(gates, monitor_writes_gates=True)


def test_clamp_gate_wire_schema_forward_compat():
    """The tagged-union member decodes/resolves (wire stability), even though
    build-time validation rejects it."""
    gates = _convert_clamp_gates(
        [_clamp_gate("rest_of_request", probe=True, strength=0.5)]
    )
    assert isinstance(gates[0].apply, ClampApply)
    res = resolve_gates(gates)
    g = res[0]
    assert g.apply_kind == "clamp"
    assert g.strength == 0.5
    assert g.steer_vectors is None  # clamps are declared statically, not here


def test_clamp_gate_strength_defaults_to_one():
    res = resolve_gates(_convert_clamp_gates([_clamp_gate("rest_of_request")]))
    assert res[0].apply_kind == "clamp"
    assert res[0].strength == 1.0


def test_clamp_gate_msgpack_roundtrip():
    gates = _convert_clamp_gates([_clamp_gate("rest_of_request", probe=True)])
    meta = RequestMetadata(conversation_id="c1", steering=gates)
    buf = msgspec.msgpack.encode(meta)
    back = msgspec.msgpack.decode(buf, type=RequestMetadata)
    assert isinstance(back.steering[0].apply, ClampApply)
    res = resolve_gates(back.steering)
    assert res[0].apply_kind == "clamp"
