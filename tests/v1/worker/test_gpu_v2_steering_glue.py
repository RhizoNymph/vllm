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

    glue._update_steering_buffers_v2(sched, input_batch)

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

    glue._update_steering_buffers_v2(sched, input_batch)

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

    glue._update_steering_buffers_v2(sched, input_batch)

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

    glue._update_steering_buffers_v2(sched, input_batch)

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

    glue._update_steering_buffers_v2(sched, input_batch)

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

    glue._update_steering_buffers_v2(sched, input_batch)

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

    glue._update_steering_buffers_v2(sched, input_batch)
    # First step with the override -> a non-admitted signature is reported.
    assert "d" in glue._pending_decode_sigs
    folded = glue._pending_decode_sigs["d"]
    assert folded != 5

    # Same override next step -> no change, nothing reported.
    glue._update_steering_buffers_v2(sched, input_batch)
    assert "d" not in glue._pending_decode_sigs

    # Drop the override -> revert to the admitted decode hash 5.
    glue._req_dynamic_decode.pop("d")
    glue._steering_manager.release_dynamic_config(dyn_id)
    glue._update_steering_buffers_v2(sched, input_batch)
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

    glue._update_steering_buffers_v2(sched, input_batch)

    # Cheap path taken: scales repopulated, no full table recompose.
    assert mgr.scales_populated == 1
    assert mgr.populated == 0
