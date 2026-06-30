# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manager-side tests for the PER-ROW (per-request) in-graph monitor.

Unlike the global monitor (one probe per site), the per-row monitor stores a
probe + ``[threshold, sharpness]`` keyed by the row's LOGICAL owner
(``("config", hash, "decode")`` / ``("dyn", dyn_id)`` / ``("global",
"decode")``), and at populate time scatters them into the per-(layer, hook)
``*_monitor_probe_table`` / ``*_monitor_row_params`` buffers in row-position
order. Opt-in: the buffers are dummy ``(1, 1)`` until
``resize_steering_row_monitor_buffers`` is called. CPU-only, no engine.
"""

import numpy as np
import torch
import torch.nn as nn

from vllm.model_executor.layers.steering import (
    _ROW_MONITOR_DEFAULT_PARAMS,
    HOOK_POINT_ROW_ACTIVE_ATTR,
    HOOK_POINT_ROW_PARAMS_ATTR,
    HOOK_POINT_ROW_PROBE_ATTR,
    SteeringHookPoint,
    register_steering_buffers,
    resize_steering_row_monitor_buffers,
)
from vllm.v1.worker.steering_manager import SteeringManager

HIDDEN = 8
MAX_STATIC = 4
MAX_DYNAMIC = 2
_HP = SteeringHookPoint.POST_BLOCK
_HP_S = _HP.value


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = 0


def _layer(*, enable_row_monitor: bool) -> _Layer:
    layer = _Layer()
    register_steering_buffers(
        layer,
        HIDDEN,
        max_steering_tokens=16,
        max_steering_configs=MAX_STATIC + MAX_DYNAMIC,
    )
    resize_steering_row_monitor_buffers([layer], enable=enable_row_monitor)
    return layer


def _mgr() -> SteeringManager:
    return SteeringManager(
        max_steering_configs=MAX_STATIC,
        device=None,
        max_dynamic_steering_configs=MAX_DYNAMIC,
    )


def _probe_tbl(layer):
    return getattr(layer, HOOK_POINT_ROW_PROBE_ATTR[_HP])


def _row_params(layer):
    return getattr(layer, HOOK_POINT_ROW_PARAMS_ATTR[_HP])


def _row_active(layer) -> bool:
    return bool(getattr(layer, HOOK_POINT_ROW_ACTIVE_ATTR[_HP]).item())


def _reg_decode_config(mgr, config_hash) -> int:
    return mgr.register_config(
        config_hash, {_HP_S: {0: np.zeros(HIDDEN)}}, phase="decode"
    )


# ---------------------------------------------------------------------------
# Store: set / clear / has
# ---------------------------------------------------------------------------


def test_set_clear_has_row_monitor():
    mgr = _mgr()
    assert mgr.has_row_monitor is False
    probe = torch.ones(HIDDEN)
    mgr.set_row_monitor(_HP_S, 0, ("config", 7, "decode"), probe, 1.0, 2.0)
    assert mgr.has_row_monitor is True
    mgr.clear_row_monitor(_HP_S, 0, ("config", 7, "decode"))
    assert mgr.has_row_monitor is False


def test_clear_row_monitor_granularity():
    mgr = _mgr()
    mgr.set_row_monitor(_HP_S, 0, ("config", 1, "decode"), torch.ones(HIDDEN), 0.0, 1.0)
    mgr.set_row_monitor(_HP_S, 0, ("dyn", 5), torch.ones(HIDDEN), 0.0, 1.0)
    mgr.clear_row_monitor(_HP_S, 0, ("dyn", 5))
    assert mgr.has_row_monitor is True  # config entry remains
    mgr.clear_row_monitor()  # clear all
    assert mgr.has_row_monitor is False


# ---------------------------------------------------------------------------
# Populate: row-position scatter + default pass-through
# ---------------------------------------------------------------------------


def test_populate_writes_probe_to_owner_row():
    mgr = _mgr()
    layer = _layer(enable_row_monitor=True)
    ch = 4242
    row = _reg_decode_config(mgr, ch)
    probe = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mgr.set_row_monitor(_HP_S, 0, ("config", ch, "decode"), probe, 3.0, 5.0)
    mgr.populate_steering_tables({0: layer})

    assert _row_active(layer) is True
    torch.testing.assert_close(_probe_tbl(layer)[row], probe)
    assert _row_params(layer)[row, 0].item() == 3.0
    assert _row_params(layer)[row, 1].item() == 5.0


def test_populate_default_rows_pass_through():
    mgr = _mgr()
    layer = _layer(enable_row_monitor=True)
    ch = 11
    _reg_decode_config(mgr, ch)
    mgr.set_row_monitor(
        _HP_S, 0, ("config", ch, "decode"), torch.ones(HIDDEN), 0.0, 1.0
    )
    mgr.populate_steering_tables({0: layer})

    _, sharp0 = _ROW_MONITOR_DEFAULT_PARAMS
    # Row 0 (sentinel) and row 1 (prefill) stay at the ungated default:
    # zero probe + a very-negative threshold ⇒ sigmoid → 1.0. (The exact
    # -1e30 sentinel is rounded by fp32 storage, so assert "very negative".)
    for sentinel in (0, 1):
        torch.testing.assert_close(_probe_tbl(layer)[sentinel], torch.zeros(HIDDEN))
        assert _row_params(layer)[sentinel, 0].item() <= -1.0e29
        assert _row_params(layer)[sentinel, 1].item() == sharp0


def test_populate_no_config_deactivates():
    mgr = _mgr()
    layer = _layer(enable_row_monitor=True)
    _reg_decode_config(mgr, 99)  # a row, but no monitor on it
    mgr.populate_steering_tables({0: layer})
    assert _row_active(layer) is False


def test_row_reassignment_relocates_probe():
    mgr = _mgr()
    layer = _layer(enable_row_monitor=True)
    ch_a, ch_b = 100, 200
    _reg_decode_config(mgr, ch_a)
    row_b = _reg_decode_config(mgr, ch_b)
    probe_b = torch.arange(HIDDEN, dtype=torch.float32)
    mgr.set_row_monitor(_HP_S, 0, ("config", ch_b, "decode"), probe_b, 0.0, 1.0)
    mgr.populate_steering_tables({0: layer})
    torch.testing.assert_close(_probe_tbl(layer)[row_b], probe_b)

    # Release ch_a; ch_b may move to a different row. The probe must follow
    # the OWNER, not the old row index.
    mgr.release_config(ch_a, "decode")
    mgr.populate_steering_tables({0: layer})
    new_row_b = mgr.get_row_for_config(ch_b, is_prefill=False)
    torch.testing.assert_close(_probe_tbl(layer)[new_row_b], probe_b)


# ---------------------------------------------------------------------------
# Opt-in: disabled ⇒ dummy buffers, never activated
# ---------------------------------------------------------------------------


def test_disabled_keeps_dummy_buffers_and_never_activates():
    mgr = _mgr()
    layer = _layer(enable_row_monitor=False)
    assert tuple(_probe_tbl(layer).shape) == (1, 1)
    ch = 5
    _reg_decode_config(mgr, ch)
    # Even with a configured row monitor, a disabled layer must not activate.
    mgr.set_row_monitor(
        _HP_S, 0, ("config", ch, "decode"), torch.ones(HIDDEN), 0.0, 1.0
    )
    mgr.populate_steering_tables({0: layer})
    assert _row_active(layer) is False


# ---------------------------------------------------------------------------
# APC: per-row monitor folds into the effective decode signature
# ---------------------------------------------------------------------------


def test_apc_signature_folds_row_monitor():
    mgr = _mgr()
    ch = 777
    _reg_decode_config(mgr, ch)
    # No monitor anywhere ⇒ no dynamic decode signature.
    assert mgr.effective_decode_signature(None, ch) is None

    mgr.set_row_monitor(
        _HP_S, 0, ("config", ch, "decode"), torch.ones(HIDDEN), 0.0, 1.0
    )
    sig = mgr.effective_decode_signature(None, ch)
    assert sig is not None
    # A different (unmonitored) config is unaffected.
    assert mgr.effective_decode_signature(None, 123) is None


def test_apc_signature_changes_with_probe():
    mgr = _mgr()
    ch = 888
    _reg_decode_config(mgr, ch)
    mgr.set_row_monitor(
        _HP_S, 0, ("config", ch, "decode"), torch.ones(HIDDEN), 0.0, 1.0
    )
    sig1 = mgr.effective_decode_signature(None, ch)
    # Reconfigure the probe ⇒ the steered decode KV differs ⇒ new key.
    mgr.set_row_monitor(
        _HP_S, 0, ("config", ch, "decode"), torch.full((HIDDEN,), 2.0), 0.0, 1.0
    )
    sig2 = mgr.effective_decode_signature(None, ch)
    assert sig1 != sig2


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-v"]))
