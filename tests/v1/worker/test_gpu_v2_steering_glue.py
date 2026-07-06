# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the v2 runner's steering control-plane glue.

Covers the v2-specific lifecycle (register on add, release on finish,
prefill->decode transition) and the per-step index build, using a fake
``SteeringManager`` and CPU tensors. The fused kernel / real manager are
exercised separately in ``tests/v1/worker/test_steering_manager*.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    HOOK_POINT_MONITOR_ACTIVE_ATTR,
    SteeringHookPoint,
)
from vllm.v1.worker.gpu.steering_runner_mixin import SteeringRunnerMixin
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin


class _FakeManager:
    def __init__(self, has_globals=False, has_dynamic_tier=False, dyn_pool=0):
        self.config_to_row: dict = {}
        self.global_base_vectors: dict = {"pre_attn": {0: [1.0]}} if has_globals else {}
        self.global_prefill_vectors: dict = {}
        self.global_decode_vectors: dict = {}
        self._tables_dirty = False
        self._scales_dirty = False
        self.has_dynamic = False
        self.has_dynamic_tier = has_dynamic_tier
        self.has_monitor = False
        self.has_row_monitor = False
        self.dynamic_tier_gain = 2.0
        self.max_dynamic_steering_configs = dyn_pool
        self.registered: list[tuple[int, str]] = []
        self.released: list[tuple[int, str]] = []
        self.populated = 0
        self.scales_populated = 0
        # Dynamic override pool: dyn_id -> row.
        self._dynamic_to_row: dict[int, int] = {}
        self._next_dyn_id = 100
        self.dynamic_registered: list = []
        self.dynamic_released: list[int] = []

    def register_config(self, h, effective, phase, locally_owned_layers):
        self.registered.append((h, phase))
        self.config_to_row[(h, phase)] = len(self.config_to_row) + 3

    def release_config(self, h, phase):
        self.released.append((h, phase))

    def get_row_for_config(self, h, is_prefill):
        # Mirror SteeringManager: a per-request hash maps to its own row; hash 0
        # maps to the global prefill (1) / decode (2) row when globals are set,
        # else to the row-0 no-steer sentinel.
        if h != 0:
            return int(h)
        if self.global_base_vectors:
            return 1 if is_prefill else 2
        return 0

    def populate_steering_tables(self, layers):
        self.populated += 1
        self._tables_dirty = False

    def populate_steering_scales(self, layers):
        self.scales_populated += 1
        self._scales_dirty = False

    # ---- dynamic override pool ----
    def register_dynamic_config(self, vectors, locally_owned_layers):
        dyn_id = self._next_dyn_id
        self._next_dyn_id += 1
        row = 200 + len(self._dynamic_to_row)
        self._dynamic_to_row[dyn_id] = row
        self.has_dynamic = True
        self.dynamic_registered.append(dyn_id)
        return dyn_id, row

    def update_dynamic_config(self, dyn_id, vectors, locally_owned_layers):
        pass

    def release_dynamic_config(self, dyn_id):
        self._dynamic_to_row.pop(dyn_id, None)
        self.dynamic_released.append(dyn_id)
        if not self._dynamic_to_row:
            self.has_dynamic = False

    def get_dynamic_row(self, dyn_id):
        return self._dynamic_to_row[dyn_id]

    def effective_decode_signature(self, dyn_id, base):
        # No override/tier/monitor -> admitted signature (None sentinel);
        # an override folds the dyn row into the signature.
        if dyn_id is not None:
            return base ^ (self._dynamic_to_row.get(dyn_id, 0) << 8)
        return None


def _layer(num_tokens=16):
    layer = SimpleNamespace(
        steering_index=torch.zeros(num_tokens, dtype=torch.long),
        steering_token_scales=torch.zeros(num_tokens, dtype=torch.float32),
        steering_row_gate=torch.zeros(num_tokens, dtype=torch.float32),
        steering_decode_mask=torch.zeros(num_tokens, dtype=torch.float32),
    )
    for hp in SteeringHookPoint:
        setattr(layer, HOOK_POINT_ANY_ACTIVE_ATTR[hp], torch.ones(1, dtype=torch.bool))
        setattr(
            layer, HOOK_POINT_MONITOR_ACTIVE_ATTR[hp], torch.ones(1, dtype=torch.bool)
        )
    return layer


def _make_glue(
    num_computed,
    prompt_len=None,
    has_globals=False,
    has_dynamic_tier=False,
    dyn_pool=0,
    max_tokens=16,
    max_seqs=8,
):
    glue = SteeringRunnerMixin.__new__(SteeringRunnerMixin)
    glue._steering_manager = _FakeManager(
        has_globals=has_globals, has_dynamic_tier=has_dynamic_tier, dyn_pool=dyn_pool
    )
    glue._steerable_layers_cache = {0: _layer(max_tokens)}
    glue._steering_reqs = {}
    glue._steering_index_dirty = False
    glue._locally_owned_layers = frozenset({0})
    glue._req_dynamic_decode = {}
    glue._req_override_source = {}
    glue._req_decode_sig_reported = {}
    glue._pending_decode_sigs = {}
    glue._dynamic_steering_stats = {}
    glue._steering_rows_scratch = np.zeros(max_seqs, dtype=np.int64)
    glue._steering_n_tokens_scratch = np.zeros(max_seqs, dtype=np.int64)
    glue._steering_tier_gain_scratch = np.zeros(max_seqs, dtype=np.float32)
    glue._steering_decode_mask_scratch = np.zeros(max_seqs, dtype=np.float32)
    glue._steering_index_pinned = torch.zeros(max_tokens, dtype=torch.long)
    glue._steering_token_scales_pinned = torch.zeros(max_tokens, dtype=torch.float32)
    glue._steering_decode_mask_pinned = torch.zeros(max_tokens, dtype=torch.float32)
    if prompt_len is None:
        prompt_len = [0] * len(num_computed)
    glue.req_states = SimpleNamespace(
        req_id_to_index={},
        num_computed_tokens_np=np.asarray(num_computed, dtype=np.int32),
        prompt_len=SimpleNamespace(np=np.asarray(prompt_len, dtype=np.int32)),
    )
    # Avoid building real SamplingParams: the resolve step just needs to be truthy.
    glue._resolve_request_steering = lambda sp, phase: {"pre_attn": {0: [1.0]}}
    return glue


def _new_req(req_id, prefill_hash, decode_hash, prompt_len, num_computed=0):
    return SimpleNamespace(
        req_id=req_id,
        sampling_params=object(),
        prefill_steering_config_hash=prefill_hash,
        decode_steering_config_hash=decode_hash,
        prompt_token_ids=list(range(prompt_len)),
        prompt_embeds=None,
        num_computed_tokens=num_computed,
    )


def test_add_request_registers_prefill():
    glue = _make_glue(num_computed=[0])
    glue._steering_add_request(
        _new_req("a", prefill_hash=7, decode_hash=9, prompt_len=10)
    )

    rs = glue._steering_reqs["a"]
    assert rs.phase == "prefill"
    assert rs.num_prompt_tokens == 10
    assert glue._steering_manager.registered == [(7, "prefill")]


def test_add_request_direct_to_decode_on_full_prefix_hit():
    glue = _make_glue(num_computed=[10])
    glue._steering_add_request(
        _new_req("a", prefill_hash=0, decode_hash=5, prompt_len=10, num_computed=10)
    )

    rs = glue._steering_reqs["a"]
    assert rs.phase == "decode"
    assert glue._steering_manager.registered == [(5, "decode")]


def test_add_request_no_hashes_is_untracked():
    glue = _make_glue(num_computed=[0])
    glue._steering_add_request(
        _new_req("a", prefill_hash=0, decode_hash=0, prompt_len=4)
    )

    assert "a" not in glue._steering_reqs
    assert glue._steering_manager.registered == []


def test_finish_request_releases_current_phase():
    glue = _make_glue(num_computed=[0])
    glue._steering_add_request(
        _new_req("a", prefill_hash=7, decode_hash=9, prompt_len=10)
    )

    glue._steering_finish_requests(["a"])

    assert "a" not in glue._steering_reqs
    assert glue._steering_manager.released == [(7, "prefill")]


def test_streaming_readd_releases_old_then_registers_new():
    glue = _make_glue(num_computed=[0])
    glue._steering_add_request(
        _new_req("a", prefill_hash=7, decode_hash=9, prompt_len=10)
    )
    # Re-add same id with a different config (streaming update).
    glue._steering_add_request(
        _new_req("a", prefill_hash=11, decode_hash=9, prompt_len=10)
    )

    assert glue._steering_manager.released == [(7, "prefill")]
    assert glue._steering_manager.registered == [(7, "prefill"), (11, "prefill")]


def test_update_buffers_builds_per_token_index_and_transition():
    # Two requests, batch order [decode "d", prefill "p"].
    glue = _make_glue(num_computed=[10, 8])
    # d: direct-to-decode (computed 10 >= prompt 10), decode_hash 5.
    glue._steering_add_request(
        _new_req("d", prefill_hash=0, decode_hash=5, prompt_len=10, num_computed=10)
    )
    # p: prefilling, computed 8 of 10, prefill_hash 7 / decode_hash 9; this
    # step schedules 3 tokens -> crosses the boundary -> transition fires.
    glue._steering_add_request(
        _new_req("p", prefill_hash=7, decode_hash=9, prompt_len=10)
    )
    glue._steering_reqs["p"].num_prompt_tokens = 10  # ensure boundary at 10

    input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["d", "p"],
        idx_mapping_np=np.asarray([0, 1], dtype=np.int32),
    )
    sched = SimpleNamespace(num_scheduled_tokens={"d": 1, "p": 3})

    glue.input_batch = input_batch
    glue._update_steering_buffers(sched)

    steering_index = glue._steerable_layers_cache[0].steering_index
    # d -> row 5 (1 token); p -> row 7 (3 tokens); tail zeroed.
    assert steering_index[:4].tolist() == [5, 7, 7, 7]
    assert steering_index[4:].sum().item() == 0
    # Boundary crossed for p (8 + 3 >= 10): prefill 7 released, decode 9 added.
    assert (7, "prefill") in glue._steering_manager.released
    assert (9, "decode") in glue._steering_manager.registered
    assert glue._steering_reqs["p"].phase == "decode"


def test_global_steering_applies_to_untracked_request():
    # Globals are set but the request carries no per-request config (untracked);
    # it must still pick up the global row, not the no-steer sentinel.
    glue = _make_glue(num_computed=[0], prompt_len=[5], has_globals=True)
    input_batch = SimpleNamespace(
        num_reqs=1, req_ids=["g"], idx_mapping_np=np.asarray([0], dtype=np.int32)
    )
    sched = SimpleNamespace(num_scheduled_tokens={"g": 5})

    glue.input_batch = input_batch
    glue._update_steering_buffers(sched)

    steering_index = glue._steerable_layers_cache[0].steering_index
    # Prefilling (computed 0 < prompt 5) with globals -> global prefill row 1.
    assert steering_index[:5].tolist() == [1, 1, 1, 1, 1]
    assert steering_index[5:].sum().item() == 0


def test_update_buffers_short_circuit_zeroes_dirty_index():
    glue = _make_glue(num_computed=[0])
    # No tracked requests and no globals -> nothing active.
    layer = glue._steerable_layers_cache[0]
    layer.steering_index[:3] = torch.tensor([1, 2, 3])
    glue._steering_index_dirty = True

    input_batch = SimpleNamespace(
        num_reqs=0, req_ids=[], idx_mapping_np=np.asarray([], dtype=np.int32)
    )
    sched = SimpleNamespace(num_scheduled_tokens={})

    glue.input_batch = input_batch
    glue._update_steering_buffers(sched)

    assert layer.steering_index.sum().item() == 0
    assert glue._steering_index_dirty is False
    # any_active + monitor-active flags cleared so apply_steering short-circuits.
    for hp in SteeringHookPoint:
        assert getattr(layer, HOOK_POINT_ANY_ACTIVE_ATTR[hp]).item() is False
        assert getattr(layer, HOOK_POINT_MONITOR_ACTIVE_ATTR[hp]).item() is False
    # The dynamic-tier gate and decode mask are cleared; the row gate is reset
    # to full strength so no stale monitor reduction survives.
    assert layer.steering_token_scales.sum().item() == 0.0
    assert layer.steering_decode_mask.sum().item() == 0.0
    assert torch.all(layer.steering_row_gate == 1.0)


# ---- §5.4 dynamic tier + Phase 2 row gating ----------------------------------


def test_dynamic_tier_gates_decode_tokens_only():
    # Tier active; one decode request + one prefilling request. Decode tokens
    # carry the tier gain in token_scales; prefill tokens get 0 (cache safety).
    glue = _make_glue(num_computed=[10, 8], has_dynamic_tier=True)
    glue._steering_add_request(
        _new_req("d", prefill_hash=0, decode_hash=5, prompt_len=10, num_computed=10)
    )
    glue._steering_add_request(
        _new_req("p", prefill_hash=7, decode_hash=9, prompt_len=10)
    )
    glue._steering_reqs["p"].num_prompt_tokens = 10

    input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["d", "p"],
        idx_mapping_np=np.asarray([0, 1], dtype=np.int32),
    )
    sched = SimpleNamespace(num_scheduled_tokens={"d": 1, "p": 3})

    glue.input_batch = input_batch
    glue._update_steering_buffers(sched)

    layer = glue._steerable_layers_cache[0]
    # d is a decode token -> gain 2.0; p's 3 prefill tokens -> 0.
    assert layer.steering_token_scales[:4].tolist() == [2.0, 0.0, 0.0, 0.0]
    assert layer.steering_token_scales[4:].sum().item() == 0.0
    # Decode mask: 1 for the decode token, 0 for the prefill tokens.
    assert layer.steering_decode_mask[:4].tolist() == [1.0, 0.0, 0.0, 0.0]
    # Row gate reset to full strength every step.
    assert torch.all(layer.steering_row_gate == 1.0)


def test_tier_only_state_defeats_short_circuit():
    # The latent bug: a tier-only manager (no per-request config, no globals)
    # must NOT be short-circuited, else the tier never applies.
    glue = _make_glue(num_computed=[10], prompt_len=[10], has_dynamic_tier=True)
    input_batch = SimpleNamespace(
        num_reqs=1, req_ids=["d"], idx_mapping_np=np.asarray([0], dtype=np.int32)
    )
    sched = SimpleNamespace(num_scheduled_tokens={"d": 1})

    glue.input_batch = input_batch
    glue._update_steering_buffers(sched)

    layer = glue._steerable_layers_cache[0]
    # Untracked decode request with no per-request/global config -> row 0, but
    # the tier gate still applies on the decode token.
    assert layer.steering_token_scales[0].item() == 2.0
    assert glue._steering_index_dirty is True


# ---- dynamic decode override pool --------------------------------------------


def test_dynamic_override_routes_decode_to_pool_row():
    glue = _make_glue(num_computed=[10], dyn_pool=4)
    glue._steering_add_request(
        _new_req("d", prefill_hash=0, decode_hash=5, prompt_len=10, num_computed=10)
    )
    # Engage an override: register a pool row and record the routing.
    mgr = glue._steering_manager
    dyn_id, row = mgr.register_dynamic_config({}, frozenset({0}))
    glue._req_dynamic_decode["d"] = dyn_id

    input_batch = SimpleNamespace(
        num_reqs=1, req_ids=["d"], idx_mapping_np=np.asarray([0], dtype=np.int32)
    )
    sched = SimpleNamespace(num_scheduled_tokens={"d": 1})

    glue.input_batch = input_batch
    glue._update_steering_buffers(sched)

    steering_index = glue._steerable_layers_cache[0].steering_index
    # Decode routed to the override pool row, NOT the admitted decode row 5.
    assert steering_index[0].item() == row
    assert row != 5


def test_apply_request_override_rejects_prefill():
    glue = _make_glue(num_computed=[3], prompt_len=[10], dyn_pool=4)
    glue.req_states.req_id_to_index = {"p": 0}
    from vllm.v1.worker.steering_action_queue import RequestSteeringOverride

    action = RequestSteeringOverride(
        req_id="p", vectors={"pre_attn": {0: np.zeros(1, dtype=np.float32)}}
    )
    assert glue._apply_request_override(action, source="t") is False
    assert "p" not in glue._req_dynamic_decode


def test_apply_request_override_clear_is_idempotent():
    glue = _make_glue(num_computed=[10], dyn_pool=4)
    from vllm.v1.worker.steering_action_queue import RequestSteeringOverride

    # Clearing with no live override is a no-op success.
    action = RequestSteeringOverride(req_id="d", vectors=None)
    assert glue._apply_request_override(action, source="t") is True
    # Now register one and clear it.
    dyn_id, _row = glue._steering_manager.register_dynamic_config({}, frozenset({0}))
    glue._req_dynamic_decode["d"] = dyn_id
    assert glue._apply_request_override(action, source="t") is True
    assert "d" not in glue._req_dynamic_decode
    assert dyn_id in glue._steering_manager.dynamic_released


def test_apply_request_override_rejects_when_pool_disabled():
    glue = _make_glue(num_computed=[10], dyn_pool=0)
    from vllm.v1.worker.steering_action_queue import RequestSteeringOverride

    action = RequestSteeringOverride(
        req_id="d", vectors={"pre_attn": {0: np.zeros(1, dtype=np.float32)}}
    )
    assert glue._apply_request_override(action, source="t") is False


def test_finish_drops_dynamic_override():
    glue = _make_glue(num_computed=[10], dyn_pool=4)
    glue._steering_add_request(
        _new_req("d", prefill_hash=0, decode_hash=5, prompt_len=10, num_computed=10)
    )
    dyn_id, _row = glue._steering_manager.register_dynamic_config({}, frozenset({0}))
    glue._req_dynamic_decode["d"] = dyn_id

    glue._steering_finish_requests(["d"])

    assert "d" not in glue._req_dynamic_decode
    assert dyn_id in glue._steering_manager.dynamic_released
    assert "d" not in glue._steering_reqs


# ---- APC effective-decode-signature deltas -----------------------------------


def test_decode_signature_delta_reports_override_then_reverts():
    glue = _make_glue(num_computed=[10], dyn_pool=4)
    glue._steering_add_request(
        _new_req("d", prefill_hash=0, decode_hash=5, prompt_len=10, num_computed=10)
    )
    dyn_id, _row = glue._steering_manager.register_dynamic_config({}, frozenset({0}))
    glue._req_dynamic_decode["d"] = dyn_id

    input_batch = SimpleNamespace(
        num_reqs=1, req_ids=["d"], idx_mapping_np=np.asarray([0], dtype=np.int32)
    )
    sched = SimpleNamespace(num_scheduled_tokens={"d": 1})

    glue.input_batch = input_batch
    glue._update_steering_buffers(sched)
    # First step with the override -> a non-admitted signature is reported.
    assert "d" in glue._pending_decode_sigs
    folded = glue._pending_decode_sigs["d"]
    assert folded != 5

    # Same override next step -> no change, nothing reported.
    glue.input_batch = input_batch
    glue._update_steering_buffers(sched)
    assert "d" not in glue._pending_decode_sigs

    # Drop the override -> revert to the admitted decode hash 5.
    glue._req_dynamic_decode.pop("d")
    glue._steering_manager.release_dynamic_config(dyn_id)
    glue.input_batch = input_batch
    glue._update_steering_buffers(sched)
    assert glue._pending_decode_sigs.get("d") == 5


# ---- §5.3 cheap scales-only populate path ------------------------------------


def test_scales_dirty_takes_cheap_populate_path():
    glue = _make_glue(num_computed=[10])
    glue._steering_add_request(
        _new_req("d", prefill_hash=0, decode_hash=5, prompt_len=10, num_computed=10)
    )
    mgr = glue._steering_manager
    mgr._tables_dirty = False
    mgr._scales_dirty = True

    input_batch = SimpleNamespace(
        num_reqs=1, req_ids=["d"], idx_mapping_np=np.asarray([0], dtype=np.int32)
    )
    sched = SimpleNamespace(num_scheduled_tokens={"d": 1})

    glue.input_batch = input_batch
    glue._update_steering_buffers(sched)

    # Cheap path taken: scales repopulated, no full table recompose.
    assert mgr.scales_populated == 1
    assert mgr.populated == 0


# ---- declarative override parity: compose + operator-wins precedence ----------
#
# Mirrors the v1 semantics in
# ``SteeringModelRunnerMixin._apply_request_override`` on the v2 runner: the
# ``compose_admitted`` fold, the ``_req_override_source`` bookkeeping, and the
# operator-wins precedence check the inherited scale/monitor paths rely on.


def _pre_attn(value: float) -> dict:
    return {"pre_attn": {0: np.array([value], dtype=np.float32)}}


def _decode_override_glue(req_id: str = "d", hidden: int = 1):
    """A decode-phase glue whose layer carries a real steering table so the
    override / scale / monitor validators pass and the apply path runs end to
    end (the bare ``_layer`` fake has no ``steering_table_*`` buffer)."""
    glue = _make_glue(num_computed=[10], prompt_len=[8], dyn_pool=4)
    glue.req_states.req_id_to_index = {req_id: 0}
    glue._steerable_layers_cache[0].steering_table_pre_attn = torch.zeros(1, hidden)
    return glue


def test_apply_request_override_compose_folds_admitted():
    from vllm.v1.worker.gpu.steering_runner_mixin import _SteeringReqState
    from vllm.v1.worker.steering_action_queue import RequestSteeringOverride

    glue = _decode_override_glue()
    # Admitted decode steering resolves to pre_attn 1.0 (the _make_glue stub).
    glue._steering_reqs["d"] = _SteeringReqState(
        sampling_params=object(),
        prefill_hash=0,
        decode_hash=5,
        num_prompt_tokens=8,
        phase="decode",
    )
    captured: dict = {}
    orig = glue._steering_manager.register_dynamic_config

    def _capture(vectors, locally_owned_layers):
        captured["vectors"] = vectors
        return orig(vectors, locally_owned_layers)

    glue._steering_manager.register_dynamic_config = _capture

    action = RequestSteeringOverride(
        req_id="d", vectors=_pre_attn(2.0), compose_admitted=True, source="declarative"
    )
    assert glue._apply_request_override(action, source="declarative") is True
    # Admitted decode (1.0) folded with the gate delta (2.0) => 3.0, not raw 2.0.
    folded = np.asarray(captured["vectors"]["pre_attn"][0])
    assert float(folded[0]) == 3.0
    assert glue._req_override_source["d"] == "declarative"


def test_apply_request_override_no_compose_registers_raw():
    from vllm.v1.worker.gpu.steering_runner_mixin import _SteeringReqState
    from vllm.v1.worker.steering_action_queue import RequestSteeringOverride

    glue = _decode_override_glue()
    glue._steering_reqs["d"] = _SteeringReqState(
        sampling_params=object(),
        prefill_hash=0,
        decode_hash=5,
        num_prompt_tokens=8,
        phase="decode",
    )
    captured: dict = {}
    orig = glue._steering_manager.register_dynamic_config

    def _capture(vectors, locally_owned_layers):
        captured["vectors"] = vectors
        return orig(vectors, locally_owned_layers)

    glue._steering_manager.register_dynamic_config = _capture

    action = RequestSteeringOverride(
        req_id="d", vectors=_pre_attn(2.0), compose_admitted=False, source="declarative"
    )
    assert glue._apply_request_override(action, source="declarative") is True
    # No fold: the admitted 1.0 is ignored, the gate delta 2.0 registered raw.
    assert float(np.asarray(captured["vectors"]["pre_attn"][0])[0]) == 2.0


def test_declarative_override_yields_to_operator_owner():
    from vllm.v1.worker.steering_action_queue import RequestSteeringOverride

    glue = _decode_override_glue()
    op = RequestSteeringOverride(req_id="d", vectors=_pre_attn(5.0), source="operator")
    assert glue._apply_request_override(op, source="operator") is True
    assert glue._req_override_source["d"] == "operator"
    op_dyn = glue._req_dynamic_decode["d"]

    # A client declarative gate for the same request is rejected; state kept.
    decl = RequestSteeringOverride(
        req_id="d", vectors=_pre_attn(9.0), compose_admitted=True, source="declarative"
    )
    assert glue._apply_request_override(decl, source="declarative") is False
    assert glue._req_override_source["d"] == "operator"
    assert glue._req_dynamic_decode["d"] == op_dyn


def test_operator_override_takes_over_declarative_owner():
    from vllm.v1.worker.steering_action_queue import RequestSteeringOverride

    glue = _decode_override_glue()
    decl = RequestSteeringOverride(
        req_id="d", vectors=_pre_attn(3.0), source="declarative"
    )
    assert glue._apply_request_override(decl, source="declarative") is True
    assert glue._req_override_source["d"] == "declarative"
    decl_dyn = glue._req_dynamic_decode["d"]

    # An operator source is not declarative -> it takes over (update-in-place,
    # same dyn row) and claims ownership.
    op = RequestSteeringOverride(req_id="d", vectors=_pre_attn(7.0), source="operator")
    assert glue._apply_request_override(op, source="operator") is True
    assert glue._req_override_source["d"] == "operator"
    assert glue._req_dynamic_decode["d"] == decl_dyn


def test_override_clear_purges_source():
    from vllm.v1.worker.steering_action_queue import RequestSteeringOverride

    glue = _decode_override_glue()
    op = RequestSteeringOverride(req_id="d", vectors=_pre_attn(5.0), source="operator")
    assert glue._apply_request_override(op, source="operator") is True
    assert "d" in glue._req_override_source

    clear = RequestSteeringOverride(req_id="d", vectors=None, source="operator")
    assert glue._apply_request_override(clear, source="operator") is True
    assert "d" not in glue._req_override_source
    assert "d" not in glue._req_dynamic_decode


def test_finish_purges_override_source():
    from vllm.v1.worker.steering_action_queue import RequestSteeringOverride

    glue = _decode_override_glue()
    glue._steering_add_request(
        _new_req("d", prefill_hash=0, decode_hash=5, prompt_len=8, num_computed=10)
    )
    op = RequestSteeringOverride(req_id="d", vectors=_pre_attn(5.0), source="operator")
    assert glue._apply_request_override(op, source="operator") is True
    assert "d" in glue._req_override_source

    glue._steering_finish_requests(["d"])
    assert "d" not in glue._req_override_source
    assert "d" not in glue._req_dynamic_decode


def test_scale_update_by_req_id_yields_to_operator_owner_v2():
    from vllm.v1.worker.steering_action_queue import (
        RequestSteeringOverride,
        SteeringScaleUpdate,
    )

    glue = _decode_override_glue()
    op = RequestSteeringOverride(req_id="d", vectors=_pre_attn(5.0), source="operator")
    assert glue._apply_request_override(op, source="operator") is True

    # The inherited scale path now sees _req_override_source on v2: a
    # declarative scale targeting the operator-owned request is rejected.
    scale = SteeringScaleUpdate(scale=0.5, req_id="d", source="declarative")
    assert glue._apply_scale_update(scale, source="declarative") is False


def test_monitor_update_by_req_id_yields_to_operator_owner_v2():
    from vllm.v1.worker.steering_action_queue import (
        RequestSteeringOverride,
        SteeringMonitorUpdate,
    )

    glue = _decode_override_glue()
    glue._row_monitor_enabled = True
    op = RequestSteeringOverride(req_id="d", vectors=_pre_attn(5.0), source="operator")
    assert glue._apply_request_override(op, source="operator") is True

    # The inherited per-row monitor path yields too (probe=None clear still
    # reaches the precedence check).
    mon = SteeringMonitorUpdate(
        hook="pre_attn", layer=0, probe=None, req_id="d", source="declarative"
    )
    assert glue._apply_monitor_update(mon, source="declarative") is False


# ---- de-fork step F: shared override reads batch position via one accessor ----


def _v1_position_host(order, computed, prompt):
    """Bare v1 mixin host exposing only the ``input_batch`` fields the
    ``_steering_req_position`` accessor reads."""
    host = SteeringModelRunnerMixin.__new__(SteeringModelRunnerMixin)
    host.input_batch = SimpleNamespace(
        req_id_to_index={r: i for i, r in enumerate(order)},
        num_computed_tokens_cpu=np.asarray(computed, dtype=np.int32),
        num_prompt_tokens=np.asarray(prompt, dtype=np.int32),
    )
    return host


def test_steering_req_position_v1_v2_parity():
    # Same logical batch state: two requests, one decode (computed >= prompt),
    # one prefill (computed < prompt), plus one not in the batch.
    order = ["d", "p"]
    computed = [10, 3]
    prompt = [10, 8]

    v1 = _v1_position_host(order, computed, prompt)
    v2 = _make_glue(num_computed=computed, prompt_len=prompt)
    v2.req_states.req_id_to_index = {r: i for i, r in enumerate(order)}

    for req_id in order:
        assert v1._steering_req_position(req_id) == v2._steering_req_position(req_id), (
            req_id
        )
    # Decode request: computed 10 >= prompt 10; prefill: 3 < 8.
    assert v1._steering_req_position("d") == (10, 10)
    assert v1._steering_req_position("p") == (3, 8)
    # Not in the batch -> both return None.
    assert v1._steering_req_position("ghost") is None
    assert v2._steering_req_position("ghost") is None
