# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner-side fail-closed rollback for declarative this_token+probe+add.

A declarative ``this_token+probe+add`` gate emits an (unconditional)
``RequestSteeringOverride`` first and a ``req_id``-keyed ``SteeringMonitorUpdate``
second (the monitor resolves to the override's freshly-registered dyn row). If
the monitor is rejected on the step thread, the override would otherwise stick
and steer EVERY token unconditionally — the opposite of the client's
probe-gated intent. ``_apply_steering_actions`` must roll the override back so
the request fails closed. Scoped narrowly: same batch, same req_id, declarative
source. CPU-only, no engine.
"""

import numpy as np
import torch
import torch.nn as nn

from vllm.v1.worker.steering_action_queue import (
    DECLARATIVE_SOURCE,
    RequestSteeringOverride,
    SteeringMonitorUpdate,
)
from vllm.v1.worker.steering_manager import SteeringManager
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin

HIDDEN = 8
MAX_STATIC = 4
MAX_DYNAMIC = 2
NUM_ROWS = MAX_STATIC + MAX_DYNAMIC + 3
_HP = "post_block"


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = 0
        self.register_buffer(
            "steering_table_post_block", torch.zeros(NUM_ROWS, HIDDEN)
        )


class _FakeInputBatch:
    def __init__(self, reqs: list[dict]):
        self.num_reqs = len(reqs)
        self.req_ids = [r["req_id"] for r in reqs]
        self.req_id_to_index = {r["req_id"]: i for i, r in enumerate(reqs)}
        self.num_computed_tokens_cpu = np.array(
            [r["num_computed"] for r in reqs], dtype=np.int32
        )
        self.num_prompt_tokens = np.array(
            [r["num_prompt"] for r in reqs], dtype=np.int32
        )


class _Host(SteeringModelRunnerMixin):
    def __init__(self, reqs: list[dict], *, row_monitor: bool):
        self._steering_manager = SteeringManager(
            max_steering_configs=MAX_STATIC,
            device=None,
            max_dynamic_steering_configs=MAX_DYNAMIC,
        )
        self._steerable_layers_cache = {0: _Layer()}
        self._locally_owned_layers = frozenset({0})
        self._dynamic_steering_stats = {}
        self._req_dynamic_decode = {}
        self._req_override_source = {}
        self._row_monitor_enabled = row_monitor
        self.input_batch = _FakeInputBatch(reqs)
        self.requests = {}


def _decode(req_id: str = "r1") -> dict:
    return {"req_id": req_id, "num_computed": 12, "num_prompt": 8}


def _add_override(req_id: str = "r1", source: str = DECLARATIVE_SOURCE):
    return RequestSteeringOverride(
        req_id=req_id,
        vectors={_HP: {0: np.ones(HIDDEN, dtype=np.float32)}},
        compose_admitted=True,
        source=source,
    )


def _monitor(req_id: str = "r1", source: str = DECLARATIVE_SOURCE):
    return SteeringMonitorUpdate(
        hook=_HP,
        layer=0,
        probe=np.ones(HIDDEN, dtype=np.float32),
        threshold=0.0,
        sharpness=1.0,
        req_id=req_id,
        source=source,
    )


def test_rejected_monitor_rolls_back_paired_override():
    # Row monitor disabled -> the paired req_id monitor is rejected; the
    # override installed earlier in the same batch must be rolled back.
    host = _Host([_decode("r1")], row_monitor=False)
    applied, rejected = host._apply_steering_actions(
        [_add_override("r1"), _monitor("r1")], source=DECLARATIVE_SOURCE
    )
    assert (applied, rejected) == (0, 2)
    assert "r1" not in host._req_dynamic_decode
    assert host._steering_manager.has_dynamic is False


def test_rollback_scoped_to_same_req_id():
    # A rejected monitor for a DIFFERENT request must not disturb r1's override.
    host = _Host([_decode("r1")], row_monitor=False)
    applied, rejected = host._apply_steering_actions(
        [_add_override("r1"), _monitor("r2")], source=DECLARATIVE_SOURCE
    )
    assert (applied, rejected) == (1, 1)
    assert "r1" in host._req_dynamic_decode


def test_operator_flow_not_rolled_back():
    # A non-declarative (operator) override is never rolled back, even when its
    # monitor is rejected — fail-closed is a declarative-only guarantee.
    host = _Host([_decode("r1")], row_monitor=False)
    applied, rejected = host._apply_steering_actions(
        [_add_override("r1", source="operator"), _monitor("r1", source="operator")],
        source="operator",
    )
    assert (applied, rejected) == (1, 1)
    assert "r1" in host._req_dynamic_decode
