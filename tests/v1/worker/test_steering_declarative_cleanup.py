# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``release_dynamic_config`` must purge the row's per-row monitor + scale.

A per-request declarative gate installs a dynamic override (a dyn row) and may
attach a per-row monitor (``SteeringMonitorUpdate(req_id=...)``) and/or a
strength scale to that row. When the request finishes the runner releases the
dyn row; without cleanup the ``RowOwner.dyn(dyn_id)`` monitor/scale entries would
leak for the process lifetime (dyn_ids are monotonic, never reused). CPU-only.
"""

import numpy as np
import torch

from vllm.v1.worker.steering_manager import SteeringManager
from vllm.v1.worker.steering_owner import RowOwner

HIDDEN = 8
HOOK = "post_block"


def _mgr() -> SteeringManager:
    return SteeringManager(
        max_steering_configs=4,
        device=None,
        max_dynamic_steering_configs=2,
    )


def test_release_dynamic_config_purges_row_monitor_and_scale():
    mgr = _mgr()
    dyn_id, _row = mgr.register_dynamic_config({HOOK: {0: np.ones(HIDDEN)}})
    owner = RowOwner.dyn(dyn_id)

    mgr.set_row_monitor(HOOK, 0, owner, torch.ones(HIDDEN), 0.0, 1.0)
    mgr.set_dynamic_scale(dyn_id, 0.5)
    assert mgr.has_row_monitor is True
    assert dyn_id in mgr._dynamic_scales

    mgr.release_dynamic_config(dyn_id)

    # the ("dyn", dyn_id) monitor entry and the scale are both gone
    assert mgr.has_row_monitor is False
    assert dyn_id not in mgr._dynamic_scales


def test_release_leaves_other_owners_intact():
    mgr = _mgr()
    d1, _ = mgr.register_dynamic_config({HOOK: {0: np.ones(HIDDEN)}})
    d2, _ = mgr.register_dynamic_config({HOOK: {0: np.ones(HIDDEN)}})
    mgr.set_row_monitor(HOOK, 0, RowOwner.dyn(d1), torch.ones(HIDDEN), 0.0, 1.0)
    mgr.set_row_monitor(HOOK, 0, RowOwner.dyn(d2), torch.ones(HIDDEN), 0.0, 1.0)
    mgr.set_row_monitor(
        HOOK, 0, RowOwner.config(7, "decode"), torch.ones(HIDDEN), 0.0, 1.0
    )

    mgr.release_dynamic_config(d1)

    # d1 gone; d2 and the static config owner remain
    assert mgr.has_row_monitor is True
    layers = mgr._row_monitor[HOOK][0]
    assert RowOwner.dyn(d1) not in layers
    assert RowOwner.dyn(d2) in layers
    assert RowOwner.config(7, "decode") in layers
