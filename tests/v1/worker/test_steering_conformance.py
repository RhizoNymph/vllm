# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-runner conformance harness for the steering control plane.

Step A of the planned v1/v2 steering de-fork. The dynamic-steering control
plane is currently forked across two GPU model runners:

* v1 -- ``vllm/v1/worker/gpu_model_runner.py`` +
  ``vllm/v1/worker/steering_model_runner_mixin.py``
* v2 -- ``vllm/v1/worker/gpu/model_runner.py`` +
  ``vllm/v1/worker/gpu/steering_runner_mixin.py`` (inherits parts of the v1
  mixin)

That fork has already produced three drift bugs (an override apply losing
``compose_admitted`` + precedence on v2, fixed in PR #224; capture
``client_request_id`` present on v2 only; streaming re-add metadata refresh on
v2 only). A de-fork is planned; this harness lands FIRST as the safety net.

Design
------
The suite drives BOTH runners' REAL mixin methods through a common driver
interface (:class:`V1Host` / :class:`V2Host`) against a shared *recording*
fake ``SteeringManager`` (:class:`RecordingManager`). Every state-mutating
manager call (register/release config, dynamic-override alloc/update/release,
scale + monitor set/clear, table/scale populate) is appended to an ordered
event list. That event log IS the conformance artifact.

For a given logical request timeline, both runners must drive the manager
IDENTICALLY. Where behavior is identical (everything except preemption) the
suite asserts the two hosts' event logs are EQUAL -- the strongest form. Most
scenarios run through :func:`_run` + the equality assertion in
:func:`test_conformance`.

Preemption -- unified on release-at-preemption
----------------------------------------------
De-fork step D landed: both runners RELEASE a preempted request's config rows
+ any per-request override at preemption time (``_steering_finish_requests``
called with finished + preempted) and re-register a fresh prefill config on
resume. :func:`test_preemption_unified` asserts the two runners drive the
manager identically across the whole preempt+resume region -- the same
elementwise-equal artifact as every other scenario. (Previously v1 HELD its
rows across preemption and released only at resume; that divergence, pinned by
the former ``_V1_PREEMPT_EVENTS`` constant, is now gone.)

CPU-only, no GPU, no network. No production code is imported for
re-implementation: the value is exercising the real mixin code paths.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    HOOK_POINT_MONITOR_ACTIVE_ATTR,
    HOOK_POINT_ROW_ACTIVE_ATTR,
    SteeringHookPoint,
)
from vllm.v1.worker.gpu.steering_runner_mixin import SteeringRunnerMixin
from vllm.v1.worker.steering_action_queue import (
    RequestSteeringOverride,
    SteeringMonitorUpdate,
    SteeringScaleUpdate,
    install_steering_action_queue,
)
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin
from vllm.v1.worker.steering_owner import RowOwner

HIDDEN = 8
HOOK = "post_block"
TABLE_ROWS = 16
MAX_TOKENS = 64
MAX_SEQS = 16


def _vec(value: float = 1.0) -> dict:
    return {HOOK: {0: np.full(HIDDEN, value, dtype=np.float32)}}


def _probe() -> np.ndarray:
    return np.arange(HIDDEN, dtype=np.float32)


# ---------------------------------------------------------------------------
# Recording fake manager -- the conformance artifact is its ``events`` list.
# ---------------------------------------------------------------------------


class RecordingManager:
    """A behavior-faithful fake ``SteeringManager`` that records every
    state-mutating call as an ordered event.

    Adapted from the ``_FakeManager`` patterns in
    ``test_gpu_v2_steering_glue.py``. Only the surface the mixins actually
    touch is modeled; row *values* are deterministic so the two runners --
    driving the same call sequence -- derive identical rows, ids, and
    signatures. Both hosts share this class, so their event logs are directly
    comparable.
    """

    def __init__(self, max_steering_configs: int = 4, max_dynamic: int = 2):
        self.max_steering_configs = max_steering_configs
        self.max_dynamic_steering_configs = max_dynamic
        self.events: list[tuple] = []

        # Static pool (rows 3 .. max_steering_configs + 2), lowest first.
        self.config_to_row: dict[tuple[int, str], int] = {}
        self.config_refcounts: dict[tuple[int, str], int] = defaultdict(int)
        self.free_rows: list[int] = list(range(max_steering_configs + 2, 2, -1))

        # Dynamic-override pool (rows above the static pool). Monotonic ids.
        self._dynamic_free_rows: list[int] = list(
            range(
                max_steering_configs + 2 + max_dynamic,
                max_steering_configs + 2,
                -1,
            )
        )
        self._dynamic_to_row: dict[int, int] = {}
        self._dynamic_sig: dict[int, int] = {}
        self._next_dynamic_id = 1

        # Global state the short-circuit reads (never engaged here: the
        # scenarios steer per-request only, matching the fork's hot paths).
        self.global_base_vectors: dict = {}
        self.global_prefill_vectors: dict = {}
        self.global_decode_vectors: dict = {}
        self.dynamic_tier_vectors: dict = {}
        self.dynamic_tier_gain = 1.0

        # Scales + monitors, keyed by logical owner (survive row reassignment).
        self._global_scales: dict[str, float] = {}
        self._config_scales: dict[tuple[int, str], float] = {}
        self._dynamic_scales: dict[int, float] = {}
        self._monitor: dict[tuple[str, int], dict] = {}
        self._row_monitor_owners: dict[tuple, dict] = {}

        self._tables_dirty = True
        self._scales_dirty = True

    def _rec(self, event: tuple) -> None:
        self.events.append(event)

    # ---- static config pool ----
    def register_config(
        self, config_hash, vectors, phase="prefill", *, locally_owned_layers=None
    ):
        self._rec(("register", config_hash, phase))
        key = (config_hash, phase)
        if key in self.config_to_row:
            self.config_refcounts[key] += 1
            return self.config_to_row[key]
        if not self.free_rows:
            raise RuntimeError("No free steering table rows")
        row = self.free_rows.pop()
        self.config_to_row[key] = row
        self.config_refcounts[key] = 1
        self._tables_dirty = True
        return row

    def release_config(self, config_hash, phase):
        self._rec(("release", config_hash, phase))
        key = (config_hash, phase)
        if key not in self.config_to_row:
            return
        self.config_refcounts[key] -= 1
        if self.config_refcounts[key] <= 0:
            row = self.config_to_row.pop(key)
            del self.config_refcounts[key]
            self.free_rows.append(row)
            self._tables_dirty = True

    def get_row_for_config(self, config_hash, is_prefill=False):
        if config_hash == 0:
            return 1 if is_prefill else 2
        phase = "prefill" if is_prefill else "decode"
        row = self.config_to_row.get((config_hash, phase))
        if row is not None:
            return row
        raise RuntimeError(
            f"Steering config (hash={config_hash}, phase={phase}) not registered"
        )

    # ---- dynamic-override pool ----
    @property
    def has_dynamic(self) -> bool:
        return bool(self._dynamic_to_row)

    @property
    def num_active_dynamic_configs(self) -> int:
        return len(self._dynamic_to_row)

    def register_dynamic_config(self, vectors, *, locally_owned_layers=None):
        if not self._dynamic_free_rows:
            raise RuntimeError("No free dynamic steering rows")
        dyn_id = self._next_dynamic_id
        self._next_dynamic_id += 1
        row = self._dynamic_free_rows.pop()
        self._dynamic_to_row[dyn_id] = row
        self._dynamic_sig[dyn_id] = 1
        self._tables_dirty = True
        self._rec(("register_dyn", dyn_id))
        return dyn_id, row

    def update_dynamic_config(self, dyn_id, vectors, *, locally_owned_layers=None):
        if dyn_id not in self._dynamic_to_row:
            raise KeyError(dyn_id)
        self._tables_dirty = True
        self._rec(("update_dyn", dyn_id))

    def release_dynamic_config(self, dyn_id):
        row = self._dynamic_to_row.pop(dyn_id, None)
        if row is None:
            return
        self._dynamic_free_rows.append(row)
        self._dynamic_sig.pop(dyn_id, None)
        if self._dynamic_scales.pop(dyn_id, None) is not None:
            self._scales_dirty = True
        owner = RowOwner.dyn(dyn_id)
        for key in [k for k in self._row_monitor_owners if k[2] == owner]:
            del self._row_monitor_owners[key]
        self._tables_dirty = True
        self._rec(("release_dyn", dyn_id))

    def get_dynamic_row(self, dyn_id):
        return self._dynamic_to_row[dyn_id]

    # ---- scales (§5.3) ----
    def set_global_scale(self, phase, scale):
        self._global_scales[phase] = float(scale)
        self._scales_dirty = True
        self._rec(("set_global_scale", phase, float(scale)))

    def set_row_scale(self, config_hash, phase, scale):
        self._config_scales[(config_hash, phase)] = float(scale)
        self._scales_dirty = True
        self._rec(("set_row_scale", config_hash, phase, float(scale)))

    def set_dynamic_scale(self, dyn_id, scale):
        self._dynamic_scales[dyn_id] = float(scale)
        self._scales_dirty = True
        self._rec(("set_dyn_scale", dyn_id, float(scale)))

    def set_dynamic_tier_gain(self, gain):
        self.dynamic_tier_gain = float(gain)
        self._rec(("set_tier_gain", float(gain)))

    # ---- global / per-row monitor (§8) ----
    @property
    def has_dynamic_tier(self) -> bool:
        return bool(self.dynamic_tier_vectors)

    @property
    def has_monitor(self) -> bool:
        return bool(self._monitor)

    @property
    def has_row_monitor(self) -> bool:
        return bool(self._row_monitor_owners)

    def set_monitor(
        self,
        hook,
        layer,
        probe,
        threshold,
        sharpness,
        gate_rows=False,
        locally_owned_layers=None,
    ):
        self._monitor[(hook, layer)] = {
            "threshold": float(threshold),
            "sharpness": float(sharpness),
        }
        self._tables_dirty = True
        self._rec(("set_monitor", hook, layer))

    def clear_monitor(self, hook=None, layer=None):
        self._monitor.pop((hook, layer), None)
        self._tables_dirty = True
        self._rec(("clear_monitor", hook, layer))

    def set_row_monitor(
        self,
        hook,
        layer,
        owner_key,
        probe,
        threshold,
        sharpness,
        locally_owned_layers=None,
    ):
        self._row_monitor_owners[(hook, layer, owner_key)] = {
            "threshold": float(threshold),
            "sharpness": float(sharpness),
        }
        self._tables_dirty = True
        self._rec(("set_row_monitor", hook, layer, owner_key))

    def clear_row_monitor(self, hook=None, layer=None, owner_key=None):
        self._row_monitor_owners.pop((hook, layer, owner_key), None)
        self._tables_dirty = True
        self._rec(("clear_row_monitor", hook, layer, owner_key))

    # ---- populate ----
    def populate_steering_tables(self, layers):
        self._rec(("populate_tables",))
        self._tables_dirty = False
        self._scales_dirty = False

    def populate_steering_scales(self, layers):
        self._rec(("populate_scales",))
        self._scales_dirty = False

    # ---- APC decode signature ----
    def effective_decode_signature(self, dyn_id, base):
        if dyn_id is not None and dyn_id in self._dynamic_to_row:
            return base ^ (self._dynamic_to_row[dyn_id] << 8)
        return None


def _new_layer() -> SimpleNamespace:
    """A CPU layer carrying just the buffers the buffer-build + validators
    touch: a real ``steering_table_<hook>`` (for probe/vector validation), the
    per-token index / tier gate / row gate / decode mask, and the per-hook
    active flags the nothing-active short-circuit clears."""
    layer = SimpleNamespace(
        steering_table_post_block=torch.zeros(TABLE_ROWS, HIDDEN),
        steering_index=torch.zeros(MAX_TOKENS, dtype=torch.long),
        steering_token_scales=torch.zeros(MAX_TOKENS, dtype=torch.float32),
        steering_row_gate=torch.zeros(MAX_TOKENS, dtype=torch.float32),
        steering_decode_mask=torch.zeros(MAX_TOKENS, dtype=torch.float32),
    )
    for hp in SteeringHookPoint:
        setattr(layer, HOOK_POINT_ANY_ACTIVE_ATTR[hp], torch.ones(1, dtype=torch.bool))
        setattr(
            layer, HOOK_POINT_MONITOR_ACTIVE_ATTR[hp], torch.ones(1, dtype=torch.bool)
        )
        setattr(layer, HOOK_POINT_ROW_ACTIVE_ATTR[hp], torch.ones(1, dtype=torch.bool))
    return layer


def _install_scratch(host) -> None:
    host._steering_rows_scratch = np.zeros(MAX_SEQS, dtype=np.int64)
    host._steering_n_tokens_scratch = np.zeros(MAX_SEQS, dtype=np.int64)
    host._steering_tier_gain_scratch = np.zeros(MAX_SEQS, dtype=np.float32)
    host._steering_decode_mask_scratch = np.zeros(MAX_SEQS, dtype=np.float32)
    host._steering_index_pinned = torch.zeros(MAX_TOKENS, dtype=torch.long)
    host._steering_token_scales_pinned = torch.zeros(MAX_TOKENS, dtype=torch.float32)
    host._steering_decode_mask_pinned = torch.zeros(MAX_TOKENS, dtype=torch.float32)


# A truthy resolve stub -> register_config fires whenever the hash is nonzero,
# exactly mirroring both runners' admission/transition guards. Identical on
# both hosts so any divergence is in the runner glue, not the resolver.
def _resolve_stub(sp, phase):
    return {HOOK: {0: [1.0] * HIDDEN}}


# ---------------------------------------------------------------------------
# Common driver interface
# ---------------------------------------------------------------------------
#
#   admit(req, prefill_hash, decode_hash, prompt_len, num_computed=0)
#   step(scheduled: dict[req_id, n_tokens])
#   apply_action(action) / apply_batch(actions)
#   preempt(req_ids) / resume(req, from_prefill) / stream_readd(req, ...)
#   finish(req_ids)
#
# Each host wires the runner-appropriate fake batch state and calls the REAL
# mixin entry points at the same call sites the production runners use.


class V1Host(SteeringModelRunnerMixin):
    runner = "v1"

    def __init__(self, max_steering_configs=4, max_dynamic=2, row_monitor=False):
        install_steering_action_queue(None)
        self._steering_manager = RecordingManager(max_steering_configs, max_dynamic)
        self._steerable_layers_cache = {0: _new_layer()}
        self._locally_owned_layers = frozenset({0})
        self._dynamic_steering_stats = {}
        self._req_dynamic_decode = {}
        self._req_override_source = {}
        self._steering_reqs = {}
        self._steering_index_dirty = False
        self._row_monitor_enabled = row_monitor
        self.requests = {}
        self._req_decode_sig_reported = {}
        self._pending_decode_sigs = {}
        _install_scratch(self)
        self._resolve_request_steering = _resolve_stub
        # Per-host batch model: req_id -> attrs, plus batch order.
        self._reqs: dict[str, dict] = {}
        self._order: list[str] = []
        self.apply_results: list[tuple[int, int]] = []
        self.input_batch = self._build_batch()

    def _build_batch(self) -> SimpleNamespace:
        order = self._order
        return SimpleNamespace(
            num_reqs=len(order),
            req_ids=list(order),
            req_id_to_index={r: i for i, r in enumerate(order)},
            num_computed_tokens_cpu=np.array(
                [self._reqs[r]["num_computed"] for r in order], dtype=np.int32
            ),
            num_prompt_tokens=np.array(
                [self._reqs[r]["num_prompt"] for r in order], dtype=np.int32
            ),
            request_prefill_steering_hash=np.array(
                [self._reqs[r]["prefill_hash"] for r in order], dtype=np.int64
            ),
            request_decode_steering_hash=np.array(
                [self._reqs[r]["decode_hash"] for r in order], dtype=np.int64
            ),
        )

    def admit(self, req_id, prefill_hash, decode_hash, prompt_len, num_computed=0):
        self._reqs[req_id] = dict(
            prefill_hash=prefill_hash,
            decode_hash=decode_hash,
            num_prompt=prompt_len,
            num_computed=num_computed,
        )
        if req_id not in self._order:
            self._order.append(req_id)
        self.input_batch = self._build_batch()
        sp = object()
        self.requests[req_id] = SimpleNamespace(
            num_prompt_tokens=prompt_len,
            sampling_params=sp,
            prefill_steering_config_hash=prefill_hash,
            decode_steering_config_hash=decode_hash,
        )
        new_req_data = SimpleNamespace(
            req_id=req_id,
            sampling_params=sp,
            prefill_steering_config_hash=prefill_hash,
            decode_steering_config_hash=decode_hash,
            prompt_token_ids=list(range(prompt_len)),
            prompt_embeds=None,
            num_computed_tokens=num_computed,
        )
        self._steering_add_request(new_req_data)

    def step(self, scheduled):
        self._update_steering_buffers(
            SimpleNamespace(num_scheduled_tokens=dict(scheduled))
        )
        for r, n in scheduled.items():
            self._reqs[r]["num_computed"] += n
        self.input_batch = self._build_batch()

    def apply_action(self, action, source="test"):
        res = self._apply_steering_actions([action], source=source)
        self.apply_results.append(res)
        return res

    def apply_batch(self, actions, source="test"):
        res = self._apply_steering_actions(list(actions), source=source)
        self.apply_results.append(res)
        return res

    def preempt(self, req_ids):
        # Unified release-at-preemption (de-fork step D): v1 releases the
        # config rows + any dynamic override at preemption, mirroring v2. The
        # request re-registers a fresh prefill config on resume.
        self._steering_finish_requests(list(req_ids))
        for r in req_ids:
            if r in self._order:
                self._order.remove(r)
        self.input_batch = self._build_batch()

    def resume(self, req_id, from_prefill):
        new_computed = 0 if from_prefill else self._reqs[req_id]["num_computed"]
        self._reqs[req_id]["num_computed"] = new_computed
        if req_id not in self._order:
            self._order.append(req_id)
        self.input_batch = self._build_batch()
        self._reset_steering_for_resumption(req_id, self.requests[req_id], new_computed)

    def stream_readd(self, req_id, new_prefill_hash, new_decode_hash, new_computed=0):
        old = self._reqs[req_id]
        prompt_len = old["num_prompt"]
        old.update(
            prefill_hash=new_prefill_hash,
            decode_hash=new_decode_hash,
            num_computed=new_computed,
        )
        self.input_batch = self._build_batch()
        sp = object()
        self.requests[req_id] = SimpleNamespace(
            num_prompt_tokens=prompt_len,
            sampling_params=sp,
            prefill_steering_config_hash=new_prefill_hash,
            decode_steering_config_hash=new_decode_hash,
        )
        # Streaming re-adds route back through the canonical admission path,
        # which releases the prior instance before registering the new config.
        self._steering_add_request(
            SimpleNamespace(
                req_id=req_id,
                sampling_params=sp,
                prefill_steering_config_hash=new_prefill_hash,
                decode_steering_config_hash=new_decode_hash,
                prompt_token_ids=list(range(prompt_len)),
                prompt_embeds=None,
                num_computed_tokens=new_computed,
            )
        )

    def finish(self, req_ids):
        self._steering_finish_requests(set(req_ids))
        for r in req_ids:
            if r in self._order:
                self._order.remove(r)
            self._reqs.pop(r, None)
            self.requests.pop(r, None)
        self.input_batch = self._build_batch()

    def events(self):
        return list(self._steering_manager.events)


class V2Host(SteeringRunnerMixin):
    runner = "v2"

    def __init__(self, max_steering_configs=4, max_dynamic=2, row_monitor=False):
        install_steering_action_queue(None)
        self._steering_manager = RecordingManager(max_steering_configs, max_dynamic)
        self._steerable_layers_cache = {0: _new_layer()}
        self._locally_owned_layers = frozenset({0})
        self._dynamic_steering_stats = {}
        self._req_dynamic_decode = {}
        self._req_override_source = {}
        self._steering_reqs = {}
        self._steering_index_dirty = False
        self._row_monitor_enabled = row_monitor
        self._req_decode_sig_reported = {}
        self._pending_decode_sigs = {}
        _install_scratch(self)
        self._resolve_request_steering = _resolve_stub
        # v2 batch model: stable per-request slot into the req_states arrays.
        self._reqs: dict[str, dict] = {}
        self._order: list[str] = []
        self._slot: dict[str, int] = {}
        self._next_slot = 0
        self._computed = np.zeros(MAX_SEQS, dtype=np.int32)
        self._prompt = np.zeros(MAX_SEQS, dtype=np.int32)
        self.apply_results: list[tuple[int, int]] = []
        self._sync_req_states()

    def _sync_req_states(self) -> None:
        self.req_states = SimpleNamespace(
            req_id_to_index=dict(self._slot),
            num_computed_tokens_np=self._computed,
            prompt_len=SimpleNamespace(np=self._prompt),
        )

    def _build_input_batch(self) -> SimpleNamespace:
        return SimpleNamespace(
            num_reqs=len(self._order),
            req_ids=list(self._order),
            idx_mapping_np=np.array(
                [self._slot[r] for r in self._order], dtype=np.int32
            ),
        )

    def admit(self, req_id, prefill_hash, decode_hash, prompt_len, num_computed=0):
        if req_id not in self._slot:
            self._slot[req_id] = self._next_slot
            self._next_slot += 1
        slot = self._slot[req_id]
        self._computed[slot] = num_computed
        self._prompt[slot] = prompt_len
        self._reqs[req_id] = dict(
            prefill_hash=prefill_hash, decode_hash=decode_hash, num_prompt=prompt_len
        )
        if req_id not in self._order:
            self._order.append(req_id)
        self._sync_req_states()
        sp = object()
        new_req_data = SimpleNamespace(
            req_id=req_id,
            sampling_params=sp,
            prefill_steering_config_hash=prefill_hash,
            decode_steering_config_hash=decode_hash,
            prompt_token_ids=list(range(prompt_len)),
            prompt_embeds=None,
            num_computed_tokens=num_computed,
        )
        self._steering_add_request(new_req_data)

    def step(self, scheduled):
        # v2 sets ``self.input_batch`` before the shared hot path; the v2
        # ``_steering_batch_view`` override reads it + ``req_states``.
        self.input_batch = self._build_input_batch()
        self._update_steering_buffers(
            SimpleNamespace(num_scheduled_tokens=dict(scheduled))
        )
        for r, n in scheduled.items():
            self._computed[self._slot[r]] += n
        self._sync_req_states()

    def apply_action(self, action, source="test"):
        res = self._apply_steering_actions([action], source=source)
        self.apply_results.append(res)
        return res

    def apply_batch(self, actions, source="test"):
        res = self._apply_steering_actions(list(actions), source=source)
        self.apply_results.append(res)
        return res

    def preempt(self, req_ids):
        # v2 RELEASES at preemption: preempted requests re-enter through
        # add_requests on resume, which re-registers a fresh prefill config.
        self._steering_finish_requests(list(req_ids))
        for r in req_ids:
            if r in self._order:
                self._order.remove(r)
        self._sync_req_states()

    def resume(self, req_id, from_prefill):
        slot = self._slot[req_id]
        new_computed = 0 if from_prefill else int(self._computed[slot])
        r = self._reqs[req_id]
        self.admit(
            req_id,
            r["prefill_hash"],
            r["decode_hash"],
            r["num_prompt"],
            num_computed=new_computed,
        )

    def stream_readd(self, req_id, new_prefill_hash, new_decode_hash, new_computed=0):
        # v2 routes streaming re-adds back through ``_steering_add_request``,
        # which releases the prior instance before registering the new one.
        r = self._reqs[req_id]
        self.admit(
            req_id, new_prefill_hash, new_decode_hash, r["num_prompt"], new_computed
        )

    def finish(self, req_ids):
        self._steering_finish_requests(list(req_ids))
        for r in req_ids:
            if r in self._order:
                self._order.remove(r)
            self._reqs.pop(r, None)
        self._sync_req_states()

    def events(self):
        return list(self._steering_manager.events)


def make_v1(**kw) -> V1Host:
    return V1Host(**kw)


def make_v2(**kw) -> V2Host:
    return V2Host(**kw)


def _run(make: Callable, script: Callable):
    host = make()
    script(host)
    return host


# ---------------------------------------------------------------------------
# Scenario scripts (identical-behavior set)
# ---------------------------------------------------------------------------


def _s_admit_prefill_transition_decode(h):
    h.admit("r1", 7, 9, 10, 0)
    h.step({"r1": 10})  # 0 + 10 >= 10 -> prefill->decode transition
    h.step({"r1": 1})  # decode step


def _s_admit_direct_decode(h):
    # Full prefix-cache hit: admitted straight into decode.
    h.admit("r1", 0, 9, 10, 10)
    h.step({"r1": 1})


def _s_override_apply_update_clear(h):
    h.admit("r1", 0, 9, 8, 8)
    h.apply_action(RequestSteeringOverride(req_id="r1", vectors=_vec(5.0)))
    h.apply_action(RequestSteeringOverride(req_id="r1", vectors=_vec(2.0)))
    h.apply_action(RequestSteeringOverride(req_id="r1", vectors=None))


def _s_override_rejected_prefill(h):
    h.admit("r1", 7, 9, 10, 0)  # still prefilling
    h.apply_action(RequestSteeringOverride(req_id="r1", vectors=_vec(5.0)))


def _s_scale_and_row_monitor_by_req(h):
    h.admit("r1", 0, 9, 8, 8)
    h.apply_action(RequestSteeringOverride(req_id="r1", vectors=_vec(5.0)))
    h.apply_action(SteeringScaleUpdate(scale=0.25, req_id="r1"))
    h.apply_action(
        SteeringMonitorUpdate(
            hook=HOOK,
            layer=0,
            probe=_probe(),
            threshold=0.0,
            sharpness=1.0,
            req_id="r1",
        )
    )


def _s_finish_releases_and_purges(h):
    h.admit("r1", 0, 9, 8, 8)
    h.apply_action(RequestSteeringOverride(req_id="r1", vectors=_vec(5.0)))
    h.apply_action(SteeringScaleUpdate(scale=0.5, req_id="r1"))
    h.apply_action(
        SteeringMonitorUpdate(
            hook=HOOK,
            layer=0,
            probe=_probe(),
            threshold=0.0,
            sharpness=1.0,
            req_id="r1",
        )
    )
    h.finish(["r1"])


def _s_streaming_readd_drops_override(h):
    h.admit("r1", 7, 9, 10, 0)
    h.step({"r1": 10})  # -> decode
    h.step({"r1": 1})
    h.apply_action(RequestSteeringOverride(req_id="r1", vectors=_vec(5.0)))
    # Re-add re-enters prefill (continuation): both runners drop the override
    # + release the old decode config + register the new prefill config.
    h.stream_readd("r1", new_prefill_hash=11, new_decode_hash=9, new_computed=0)


def _s_pool_exhaustion_keeps_state(h):
    h.admit("r1", 0, 9, 8, 8)
    h.admit("r2", 0, 9, 8, 8)
    h.admit("r3", 0, 9, 8, 8)
    h.apply_action(RequestSteeringOverride(req_id="r1", vectors=_vec(5.0)))
    h.apply_action(RequestSteeringOverride(req_id="r2", vectors=_vec(5.0)))
    # Pool (size 2) exhausted -> r3 rejected, r1/r2 untouched.
    h.apply_action(RequestSteeringOverride(req_id="r3", vectors=_vec(5.0)))


def _s_mixed_action_batch(h):
    h.admit("r1", 0, 9, 8, 8)
    # Override (installs dyn row) then a scale targeting the same request in
    # ONE apply call -- the scale resolves the row registered earlier in the
    # same batch.
    h.apply_batch(
        [
            RequestSteeringOverride(req_id="r1", vectors=_vec(5.0)),
            SteeringScaleUpdate(scale=0.5, req_id="r1"),
        ]
    )


def _s_decode_signature_deltas(h):
    h.admit("r1", 0, 9, 8, 8)
    h.apply_action(RequestSteeringOverride(req_id="r1", vectors=_vec(5.0)))
    h.step({"r1": 1})
    assert "r1" in h._pending_decode_sigs
    assert h._pending_decode_sigs["r1"] != 9  # folded, not the admitted hash
    h.step({"r1": 1})
    assert "r1" not in h._pending_decode_sigs  # unchanged -> nothing reported
    h.apply_action(RequestSteeringOverride(req_id="r1", vectors=None))
    h.step({"r1": 1})
    assert h._pending_decode_sigs.get("r1") == 9  # reverts to admitted


# ---------------------------------------------------------------------------
# Scenario table: name -> (script, host kwargs, expected_events|None, check|None)
# ``expected_events`` locks the ABSOLUTE sequence (so a bug that shifts BOTH
# runners identically is still caught); ``check`` asserts per-runner invariants
# on each host after the script.
# ---------------------------------------------------------------------------

_DYN = dict(max_dynamic=2)
_DYN_MON = dict(max_dynamic=2, row_monitor=True)


def _chk_override_cleared(h):
    assert h._req_dynamic_decode == {}
    assert not h._steering_manager.has_dynamic


def _chk_override_rejected(h):
    assert h.apply_results == [(0, 1)]
    assert h._req_dynamic_decode == {}


def _chk_scale_and_monitor(h):
    assert h._steering_manager.has_row_monitor is True
    dyn_id = h._req_dynamic_decode["r1"]
    assert h._steering_manager._dynamic_scales[dyn_id] == 0.25


def _chk_finish_purged(h):
    assert h._req_dynamic_decode == {}
    assert h._steering_manager.has_row_monitor is False
    assert h._steering_manager._dynamic_scales == {}


def _chk_streaming(h):
    assert h._req_dynamic_decode == {}
    assert not h._steering_manager.has_dynamic


def _chk_pool_exhaustion(h):
    assert set(h._req_dynamic_decode) == {"r1", "r2"}
    assert h.apply_results == [(1, 0), (1, 0), (0, 1)]


def _chk_mixed(h):
    assert h.apply_results == [(2, 0)]


SCENARIOS: dict[str, tuple] = {
    "admit_prefill_transition_decode": (
        _s_admit_prefill_transition_decode,
        {},
        [
            ("register", 7, "prefill"),
            ("populate_tables",),
            ("release", 7, "prefill"),
            ("register", 9, "decode"),
            ("populate_tables",),
        ],
        None,
    ),
    "admit_direct_decode": (
        _s_admit_direct_decode,
        {},
        [("register", 9, "decode"), ("populate_tables",)],
        None,
    ),
    "override_apply_update_clear": (
        _s_override_apply_update_clear,
        _DYN,
        [
            ("register", 9, "decode"),
            ("register_dyn", 1),
            ("update_dyn", 1),
            ("release_dyn", 1),
        ],
        _chk_override_cleared,
    ),
    "override_rejected_prefill": (
        _s_override_rejected_prefill,
        _DYN,
        [("register", 7, "prefill")],
        _chk_override_rejected,
    ),
    "scale_and_row_monitor_by_req": (
        _s_scale_and_row_monitor_by_req,
        _DYN_MON,
        [
            ("register", 9, "decode"),
            ("register_dyn", 1),
            ("set_dyn_scale", 1, 0.25),
            ("set_row_monitor", HOOK, 0, RowOwner.dyn(1)),
        ],
        _chk_scale_and_monitor,
    ),
    "finish_releases_and_purges": (
        _s_finish_releases_and_purges,
        _DYN_MON,
        [
            ("register", 9, "decode"),
            ("register_dyn", 1),
            ("set_dyn_scale", 1, 0.5),
            ("set_row_monitor", HOOK, 0, RowOwner.dyn(1)),
            ("release_dyn", 1),
            ("release", 9, "decode"),
        ],
        _chk_finish_purged,
    ),
    "streaming_readd_drops_override": (
        _s_streaming_readd_drops_override,
        _DYN,
        [
            ("register", 7, "prefill"),
            ("populate_tables",),
            ("release", 7, "prefill"),
            ("register", 9, "decode"),
            ("populate_tables",),
            ("register_dyn", 1),
            ("release_dyn", 1),
            ("release", 9, "decode"),
            ("register", 11, "prefill"),
        ],
        _chk_streaming,
    ),
    "pool_exhaustion_keeps_state": (
        _s_pool_exhaustion_keeps_state,
        _DYN,
        [
            ("register", 9, "decode"),
            ("register", 9, "decode"),
            ("register", 9, "decode"),
            ("register_dyn", 1),
            ("register_dyn", 2),
        ],
        _chk_pool_exhaustion,
    ),
    "mixed_action_batch": (
        _s_mixed_action_batch,
        _DYN,
        [
            ("register", 9, "decode"),
            ("register_dyn", 1),
            ("set_dyn_scale", 1, 0.5),
        ],
        _chk_mixed,
    ),
    "decode_signature_deltas": (
        _s_decode_signature_deltas,
        _DYN,
        [
            ("register", 9, "decode"),
            ("register_dyn", 1),
            ("populate_tables",),
            ("release_dyn", 1),
            ("populate_tables",),
        ],
        None,
    ),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_conformance(name):
    """v1 and v2 drive the SteeringManager IDENTICALLY for the same script."""
    script, kwargs, expected, check = SCENARIOS[name]
    h1 = _run(lambda: make_v1(**kwargs), script)
    h2 = _run(lambda: make_v2(**kwargs), script)

    e1, e2 = h1.events(), h2.events()
    assert e1 == e2, f"{name}: v1/v2 manager event logs diverge:\n{e1}\n{e2}"
    assert h1.apply_results == h2.apply_results, f"{name}: apply results diverge"
    if expected is not None:
        assert e1 == expected, f"{name}: absolute event log changed:\n{e1}"
    if check is not None:
        check(h1)
        check(h2)


# ---------------------------------------------------------------------------
# Preemption -- UNIFIED on release-at-preemption (de-fork step D).
# ---------------------------------------------------------------------------
#
# Both runners release a preempted request's config rows + dynamic override at
# preemption and re-register a fresh prefill config on resume.
_PREEMPT_EVENTS: list[tuple] = [
    ("release_dyn", 1),
    ("release", 9, "decode"),
]


def _run_preemption(make):
    h = make(max_dynamic=2)
    h.admit("r1", 7, 9, 10, 0)
    h.step({"r1": 10})  # -> decode
    h.step({"r1": 1})  # decode step
    h.apply_action(RequestSteeringOverride(req_id="r1", vectors=_vec(5.0)))
    i0 = len(h.events())
    h.preempt(["r1"])
    at_preempt = h.events()[i0:]
    mgr = h._steering_manager
    mid_state = ((9, "decode") in mgr.config_to_row, mgr.has_dynamic)
    h.resume("r1", from_prefill=True)
    region = h.events()[i0:]  # preempt + resume
    return at_preempt, mid_state, region


def test_preemption_unified():
    v1_preempt, v1_mid, v1_region = _run_preemption(make_v1)
    v2_preempt, v2_mid, v2_region = _run_preemption(make_v2)

    # Both runners release the config row + override AT preemption.
    assert v1_preempt == _PREEMPT_EVENTS
    assert v2_preempt == _PREEMPT_EVENTS

    # State DURING preemption: both freed the config row + dynamic override
    # (release-at-preemption). A preempted request no longer pins pool rows.
    assert v1_mid == (False, False)
    assert v2_mid == (False, False)

    # The whole preempt+resume region is now elementwise identical between the
    # runners -- the same strongest-form artifact as every other scenario.
    assert v1_region == v2_region
    assert v1_region == [
        ("release_dyn", 1),
        ("release", 9, "decode"),
        ("register", 7, "prefill"),
    ]


def test_preempt_and_finish_same_step_no_double_release():
    """A request both preempted and finished in one step must release its
    steering exactly once. The finish site unions finished + preempted ids;
    ``_steering_finish_requests`` pops the canonical state, so the second id in
    the union is a no-op (no double release_config / release_dyn)."""
    for make in (make_v1, make_v2):
        h = make(max_dynamic=2)
        h.admit("r1", 7, 9, 10, 0)
        h.step({"r1": 10})  # -> decode
        h.step({"r1": 1})
        h.apply_action(RequestSteeringOverride(req_id="r1", vectors=_vec(5.0)))
        i0 = len(h.events())
        # Same step: the runner passes finished UNION preempted. Model that as
        # a single finish call over the deduplicated union (a set), then a
        # second finish over the same id to prove idempotence.
        h._steering_finish_requests({"r1"})
        h._steering_finish_requests({"r1"})
        released = h.events()[i0:]
        assert released == [
            ("release_dyn", 1),
            ("release", 9, "decode"),
        ], f"{make.__name__}: expected exactly one release, got {released}"
        assert h._req_dynamic_decode == {}
        assert not h._steering_manager.has_dynamic
        assert (9, "decode") not in h._steering_manager.config_to_row


# ---------------------------------------------------------------------------
# Device-buffer byte-equality (de-fork step E)
# ---------------------------------------------------------------------------
#
# The scenario harness above asserts the two runners drive the manager with
# equal EVENT logs. This asserts the complementary, stronger property for the
# unified per-step hot path: the SAME scenario produces byte-identical
# device-bound steering buffers — the per-token tensors a CUDA-graph replay
# actually reads. A mixed prefill+decode batch with a live dynamic override
# exercises steering_index (prefill row, dynamic-override row), decode_mask
# (per-token decode vs prefill), and row_gate (reset to 1.0), so the
# comparison is non-trivial across positions.


def _s_buffer_content_mixed(h):
    # r1 keeps prefilling this step (2 + 3 = 5 < 6): admitted prefill row.
    h.admit("r1", 7, 9, 6, 2)
    # r2 is a full prefix-cache hit -> admitted straight into decode.
    h.admit("r2", 0, 5, 4, 4)
    # A live override routes r2's decode tokens to its dynamic-pool row.
    h.apply_action(RequestSteeringOverride(req_id="r2", vectors=_vec(3.0)))
    # One mixed step: r1 gets 3 prefill tokens, r2 gets 1 decode token.
    h.step({"r1": 3, "r2": 1})


def test_buffer_contents_elementwise_equal():
    """The unified hot path writes byte-identical device buffers on both runners.

    The conformance harness checks manager events; this checks the per-token
    buffers the forward (and its CUDA-graph replay) reads — the strongest CPU
    proof the unification is byte-faithful.
    """
    h1 = _run(make_v1, _s_buffer_content_mixed)
    h2 = _run(make_v2, _s_buffer_content_mixed)
    layer1 = h1._steerable_layers_cache[0]
    layer2 = h2._steerable_layers_cache[0]
    for name in (
        "steering_index",
        "steering_token_scales",
        "steering_row_gate",
        "steering_decode_mask",
    ):
        b1 = getattr(layer1, name)
        b2 = getattr(layer2, name)
        assert torch.equal(b1, b2), (
            f"{name} diverged v1 vs v2:\n  v1={b1[:8].tolist()}\n  v2={b2[:8].tolist()}"
        )

    # Sanity: the batch actually populated the buffers (not an all-zero compare).
    # Batch order [r1 (3 prefill tokens), r2 (1 decode token)].
    idx = layer1.steering_index
    assert idx[:3].tolist() != [0, 0, 0]  # r1's prefill row
    assert int(idx[3]) != 0  # r2's dynamic-override row
    assert layer1.steering_decode_mask[:4].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert torch.all(layer1.steering_row_gate == 1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
