# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the named-module pre-materialization fast path.

The API layer calls ``pre_materialize_steering_module`` immediately
after broadcasting a named module so the first request resolving to
that name finds its ``(hash, phase)`` row already populated in the
manager's refcount table — turning a ~15 ms cold-path
``register_config.materialize`` (synchronous bf16 H2D upload of every
layer) into a ~5 µs refcount bump on its TTFT.

Refcount invariants exercised here:

- Pre-materialize bumps refcount by ``+1`` per ``(hash, phase)`` and
  pins the row.
- A request resolving to the same name bumps refcount by another
  ``+1`` (refcount-hit, no re-materialize).
- Request completion drops by ``+1`` but the pin keeps the row alive.
- :meth:`release_pre_materialized_steering_module` drops the pinned
  ``+1``; once the last in-flight request finishes the row is GC'd.
- Concurrent register + first-request never double-materialize: the
  refcount is the safety net.
"""

from __future__ import annotations

from vllm.config.steering_types import hash_steering_config
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.steering_manager import SteeringManager
from vllm.v1.worker.steering_model_runner_mixin import (
    SteeringModelRunnerMixin,
)

HIDDEN = 4


class _PrematerializeStub(SteeringModelRunnerMixin):
    """Mixin subclass with a real ``SteeringManager`` but no model.

    Skips ``_init_steering_state``'s model-walk so we can drive the
    register / pre-materialize / release paths in isolation.  The
    manager is allocated on CPU so ``_stack_vectors_to_device`` doesn't
    require CUDA.
    """

    def __init__(self, max_configs: int = 8):
        self._steering_manager = SteeringManager(
            max_steering_configs=max_configs,
            device=None,
        )
        self._steering_module_registry: dict = {}
        self._steering_module_resolved_cache: dict = {}
        self._steering_module_pinned_rows: dict = {}
        self._locally_owned_layers = frozenset({0, 1})


def _spec(layer_to_vec: dict[int, list[float]]) -> dict:
    """Build a single-hook SteeringVectorSpec on hook ``post_block``."""
    return {"post_block": dict(layer_to_vec)}


def _module_payload(
    base: dict | None = None,
    prefill: dict | None = None,
    decode: dict | None = None,
) -> dict:
    return {
        "vectors": base,
        "prefill_vectors": prefill,
        "decode_vectors": decode,
    }


# ---------------------------------------------------------------------------
# Pre-materialize: rows arrive in refcount table before any request
# ---------------------------------------------------------------------------


class TestPreMaterializeMaterializesRows:
    def test_pin_creates_refcount_entry_before_request(self):
        """After register + pre-materialize, the (hash, phase) row exists.

        This is the core property the optimization buys: a request
        carrying ``steering_module_ref=(name, 1.0)`` arrives to find
        ``config_to_row`` already populated, so its
        ``register_config`` call is a refcount-hit (~5 µs) instead of
        the ~15 ms materialize.
        """
        stub = _PrematerializeStub()
        base = _spec({0: [1.0, 2.0, 3.0, 4.0]})
        stub.register_steering_modules({"m": _module_payload(base=base)}, replace=True)

        # No request has arrived yet.
        assert stub._steering_manager.num_active_configs == 0
        pinned = stub.pre_materialize_steering_module("m")

        # Pre-materialize installed pins for both phases (base vectors
        # apply to both).
        phases = {phase for _h, phase in pinned}
        assert phases == {"prefill", "decode"}

        # Each pinned row has refcount=1 and exists in the manager's
        # table — this is what a downstream request will look up.
        for h, phase in pinned:
            assert (h, phase) in stub._steering_manager.config_to_row
            assert stub._steering_manager.config_refcounts[(h, phase)] == 1

    def test_pin_hashes_match_request_hash_format(self):
        """The hash a request computes for ``(name, 1.0)`` must equal
        the pinned hash.  Otherwise the request would still cold-path
        ``register_config.materialize`` against a different row.
        """
        stub = _PrematerializeStub()
        base = _spec({0: [1.0, 2.0, 3.0, 4.0]})
        stub.register_steering_modules({"m": _module_payload(base=base)}, replace=True)
        stub.pre_materialize_steering_module("m")

        sp = SamplingParams(steering_module_ref=("m", 1.0))
        request_prefill_hash = sp.prefill_steering_config_hash
        request_decode_hash = sp.decode_steering_config_hash

        assert (request_prefill_hash, "prefill") in (
            stub._steering_manager.config_to_row
        )
        assert (request_decode_hash, "decode") in (stub._steering_manager.config_to_row)

    def test_decode_only_module_pins_decode_only(self):
        stub = _PrematerializeStub()
        decode = _spec({0: [1.0, 2.0, 3.0, 4.0]})
        stub.register_steering_modules(
            {"m": _module_payload(decode=decode)}, replace=True
        )
        pinned = stub.pre_materialize_steering_module("m")
        assert {phase for _h, phase in pinned} == {"decode"}

    def test_pin_returns_empty_for_unknown_name(self):
        """Pre-materialize on a name with no resolved cache is a no-op."""
        stub = _PrematerializeStub()
        assert stub.pre_materialize_steering_module("never_registered") == []
        assert stub._steering_manager.num_active_configs == 0


# ---------------------------------------------------------------------------
# Refcount semantics: request after pre-materialize is a refcount-hit
# ---------------------------------------------------------------------------


class TestRefcountAfterPreMaterialize:
    def test_request_register_after_pin_is_refcount_hit(self):
        """A request reaching ``register_config`` after the pin is
        applied bumps refcount from 1 → 2 (no materialize, no new row).
        """
        stub = _PrematerializeStub()
        base = _spec({0: [1.0, 2.0, 3.0, 4.0]})
        stub.register_steering_modules({"m": _module_payload(base=base)}, replace=True)
        stub.pre_materialize_steering_module("m")

        # Snapshot the pre-request state.
        snapshot_rows = dict(stub._steering_manager.config_to_row)
        snapshot_refcounts = dict(stub._steering_manager.config_refcounts)

        # The request hot path goes through ``register_config`` directly
        # with the resolved vectors (mirrors what ``_steering_add_request``
        # does after ``_resolve_request_steering`` returns the cached array).
        sp = SamplingParams(steering_module_ref=("m", 1.0))
        for phase in ("prefill", "decode"):
            resolved = stub._resolve_request_steering(sp, phase)
            assert resolved is not None
            request_hash = (
                sp.prefill_steering_config_hash
                if phase == "prefill"
                else sp.decode_steering_config_hash
            )
            stub._steering_manager.register_config(
                request_hash,
                resolved,
                phase=phase,
                locally_owned_layers=stub._locally_owned_layers,
            )

        # Same rows assigned (no new allocations) and refcount went
        # from 1 (pin) to 2 (pin + request).
        assert dict(stub._steering_manager.config_to_row) == snapshot_rows
        for key, prev in snapshot_refcounts.items():
            assert stub._steering_manager.config_refcounts[key] == prev + 1

    def test_request_completion_drops_to_pin_only(self):
        """When the last in-flight request finishes its
        ``release_config`` bumps refcount back down to 1 (the pin).
        The row stays alive.
        """
        stub = _PrematerializeStub()
        base = _spec({0: [1.0, 2.0, 3.0, 4.0]})
        stub.register_steering_modules({"m": _module_payload(base=base)}, replace=True)
        stub.pre_materialize_steering_module("m")

        sp = SamplingParams(steering_module_ref=("m", 1.0))
        resolved_prefill = stub._resolve_request_steering(sp, "prefill")
        h_prefill = sp.prefill_steering_config_hash
        stub._steering_manager.register_config(
            h_prefill,
            resolved_prefill,
            phase="prefill",
            locally_owned_layers=stub._locally_owned_layers,
        )

        # Request completes.
        stub._steering_manager.release_config(h_prefill, "prefill")

        # Row still active because of the pin.
        assert (h_prefill, "prefill") in stub._steering_manager.config_to_row
        assert stub._steering_manager.config_refcounts[(h_prefill, "prefill")] == 1


# ---------------------------------------------------------------------------
# Unregister releases the pin, allowing GC
# ---------------------------------------------------------------------------


class TestUnregisterReleasesPin:
    def test_unregister_drops_pin_and_gcs_row(self):
        """``unregister_steering_modules`` releases the pinned ``+1``;
        with no in-flight requests the row is GC'd immediately.
        """
        stub = _PrematerializeStub()
        base = _spec({0: [1.0, 2.0, 3.0, 4.0]})
        stub.register_steering_modules({"m": _module_payload(base=base)}, replace=True)
        stub.pre_materialize_steering_module("m")
        assert stub._steering_manager.num_active_configs == 2

        stub.unregister_steering_modules(["m"])
        assert stub._steering_manager.num_active_configs == 0
        assert "m" not in stub._steering_module_pinned_rows

    def test_unregister_keeps_row_while_request_in_flight(self):
        """If a request still references the row when unregister fires
        the per-request refcount keeps the row alive; it GC's only on
        the matching ``release_config`` from request completion.
        """
        stub = _PrematerializeStub()
        base = _spec({0: [1.0, 2.0, 3.0, 4.0]})
        stub.register_steering_modules({"m": _module_payload(base=base)}, replace=True)
        stub.pre_materialize_steering_module("m")

        sp = SamplingParams(steering_module_ref=("m", 1.0))
        resolved = stub._resolve_request_steering(sp, "prefill")
        h = sp.prefill_steering_config_hash
        stub._steering_manager.register_config(
            h,
            resolved,
            phase="prefill",
            locally_owned_layers=stub._locally_owned_layers,
        )
        # Refcount=2 (pin + request)

        stub.unregister_steering_modules(["m"])
        # Pin released, but request still holds the row.
        assert (h, "prefill") in stub._steering_manager.config_to_row
        assert stub._steering_manager.config_refcounts[(h, "prefill")] == 1

        # Request completes; row finally GC'd.
        stub._steering_manager.release_config(h, "prefill")
        assert (h, "prefill") not in stub._steering_manager.config_to_row

    def test_release_with_no_pin_is_noop(self):
        """Calling release on a name that was never pre-materialized
        must not raise.  Used by ``unregister_steering_modules`` so
        the unregister path is uniform regardless of whether
        pre-materialization ran.
        """
        stub = _PrematerializeStub()
        # No prior register or pre-materialize.
        stub.release_pre_materialized_steering_module("nope")  # must not raise
        assert stub._steering_manager.num_active_configs == 0


# ---------------------------------------------------------------------------
# Concurrent register + first-request safety net
# ---------------------------------------------------------------------------


class TestConcurrentRegisterAndRequest:
    def test_request_first_then_pre_materialize_does_not_double_allocate(self):
        """Race scenario: a request's ``register_config`` lands before
        the pre-materialize RPC.  Pre-materialize must not allocate a
        second row for the same ``(hash, phase)`` — the refcount is
        the safety net.
        """
        stub = _PrematerializeStub()
        base = _spec({0: [1.0, 2.0, 3.0, 4.0]})
        stub.register_steering_modules({"m": _module_payload(base=base)}, replace=True)

        # Request lands first (raced past the pre-materialize RPC).
        sp = SamplingParams(steering_module_ref=("m", 1.0))
        resolved = stub._resolve_request_steering(sp, "prefill")
        h = sp.prefill_steering_config_hash
        request_row = stub._steering_manager.register_config(
            h,
            resolved,
            phase="prefill",
            locally_owned_layers=stub._locally_owned_layers,
        )
        # Refcount = 1 (request only).
        assert stub._steering_manager.config_refcounts[(h, "prefill")] == 1

        # Pre-materialize RPC arrives.  Refcount-hit path inside
        # ``register_config`` bumps to 2 instead of allocating.
        pinned = stub.pre_materialize_steering_module("m")

        # Same row, refcount = 2 (pin + request).
        same_row = stub._steering_manager.config_to_row[(h, "prefill")]
        assert same_row == request_row
        assert stub._steering_manager.config_refcounts[(h, "prefill")] == 2
        # The pin is recorded so a later unregister still releases the
        # right number of refs.
        assert (h, "prefill") in pinned

    def test_pre_materialize_idempotent(self):
        """Issuing pre-materialize twice for the same name must not
        double-pin.  The second call short-circuits via the existing-
        pin check; the first pin remains and is the only one to be
        released on unregister.
        """
        stub = _PrematerializeStub()
        base = _spec({0: [1.0, 2.0, 3.0, 4.0]})
        stub.register_steering_modules({"m": _module_payload(base=base)}, replace=True)
        stub.pre_materialize_steering_module("m")
        # Snapshot refcounts.
        before = dict(stub._steering_manager.config_refcounts)

        result = stub.pre_materialize_steering_module("m")
        assert result == []  # no-op

        after = dict(stub._steering_manager.config_refcounts)
        assert before == after

        # Unregister releases exactly the original pin.
        stub.unregister_steering_modules(["m"])
        assert stub._steering_manager.num_active_configs == 0


# ---------------------------------------------------------------------------
# Re-register replaces stale pin
# ---------------------------------------------------------------------------


class TestReRegisterRefreshesPin:
    def test_re_register_drops_stale_pin_then_pin_again(self):
        """Re-registering a name must drop the prior pin so the
        bookkeeping refcount accounting stays balanced.

        The pinned hash for a named-only request is purely a function
        of ``(name, scale)`` — it does NOT change when the underlying
        vectors change (see
        :meth:`SamplingParams.prefill_steering_config_hash`).  But the
        row's *contents* do change, so the manager re-uploads the new
        vectors when the pre-materialize fires.  The pin tracking
        still has to be re-established because re-register is a
        logical "clean slate" for the name.
        """
        stub = _PrematerializeStub()
        original = _spec({0: [1.0, 2.0, 3.0, 4.0]})
        stub.register_steering_modules(
            {"m": _module_payload(base=original)}, replace=True
        )
        stub.pre_materialize_steering_module("m")

        # The pinned hash is the named-only hash, identical across
        # re-registrations of the same name+scale.
        named_only_h = hash_steering_config(None, module_ref=("m", 1.0))
        assert (named_only_h, "prefill") in stub._steering_manager.config_to_row
        # Refcount=1 (just the pin).
        assert stub._steering_manager.config_refcounts[(named_only_h, "prefill")] == 1

        # Re-register with new vectors.  The stale pin is released
        # before the new spec is stored — leaves refcount at 0 so the
        # row is GC'd, then the new pre-materialize allocates fresh.
        replacement = _spec({0: [9.0, 9.0, 9.0, 9.0]})
        stub.register_steering_modules(
            {"m": _module_payload(base=replacement)}, replace=False
        )
        assert "m" not in stub._steering_module_pinned_rows
        assert (named_only_h, "prefill") not in stub._steering_manager.config_to_row

        # New pre-materialize installs a fresh pin against the same
        # name+scale hash but with new vector contents.
        stub.pre_materialize_steering_module("m")
        assert (named_only_h, "prefill") in stub._steering_manager.config_to_row
        # Verify contents match the *new* spec by checking the manager
        # stored the new vector in its config_vectors map.
        stored = stub._steering_manager.config_vectors[(named_only_h, "prefill")]
        layer0_t = stored["post_block"][0].squeeze(0)
        assert layer0_t.tolist() == [9.0, 9.0, 9.0, 9.0]

    def test_replace_true_drops_all_prior_pins(self):
        """``replace=True`` clears every prior name; every pin against
        a dropped name must be released so no rows leak.
        """
        stub = _PrematerializeStub()
        a = _spec({0: [1.0, 2.0, 3.0, 4.0]})
        b = _spec({0: [5.0, 6.0, 7.0, 8.0]})
        stub.register_steering_modules(
            {
                "a": _module_payload(base=a),
                "b": _module_payload(base=b),
            },
            replace=True,
        )
        stub.pre_materialize_steering_module("a")
        stub.pre_materialize_steering_module("b")
        assert stub._steering_manager.num_active_configs == 4  # 2 names × 2 phases

        # Push a brand-new registry (replace=True).
        c = _spec({0: [10.0, 10.0, 10.0, 10.0]})
        stub.register_steering_modules(
            {"c": _module_payload(base=c)},
            replace=True,
        )
        # Old pins dropped; 'c' is unpinned (router would call
        # pre_materialize separately).
        assert stub._steering_manager.num_active_configs == 0
        assert stub._steering_module_pinned_rows == {}
