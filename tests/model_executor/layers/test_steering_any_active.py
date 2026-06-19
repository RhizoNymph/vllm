# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the per-hook ``any_active`` short-circuit.

The steering op accepts a fourth tensor argument — a single-element
bool buffer that the kernel reads at launch and uses to skip the
gather + add when no row is currently active for the hook point.
The flag is a tensor (not a Python branch) so the ``torch.compile``
graph topology stays stable across batches whose active-hook set
differs.

These tests cover:

* The CPU eager path returns a clone of ``hidden_states`` when the
  flag is ``False``, regardless of table contents (i.e. a deliberately
  garbage table must not leak into the result).
* The CPU eager path matches the active-path math when the flag is
  ``True``.
* The CUDA Triton path mirrors both behaviours when CUDA is available
  (skipped otherwise so the tests run on CPU-only environments).
* ``register_steering_buffers`` registers a properly-sized bool flag
  next to each table buffer.
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
    register_steering_buffers,
)
from vllm.model_executor.layers.steering import apply_steering as _real_apply_steering


def apply_steering(*args):
    """Back-compat shim: the op gained 4 fused-monitor args
    (probe, params, active, decode_mask). These any_active tests predate
    the monitor and call with the original 8 args; default the monitor to
    inactive so their assertions (no monitor gating) still hold."""
    if len(args) == 12:
        return _real_apply_steering(*args)
    hidden, table = args[0], args[1]
    n, h = hidden.shape[0], table.shape[1]
    dev = table.device
    return _real_apply_steering(
        *args,
        torch.zeros(h, device=dev),  # monitor probe
        torch.tensor([0.0, 1.0, 0.0], device=dev),  # [threshold, sharp, gate_rows]
        torch.zeros(1, dtype=torch.bool, device=dev),  # monitor active (off)
        torch.zeros(n, device=dev),  # decode mask
    )

# ---------------------------------------------------------------------------
# CPU eager path
# ---------------------------------------------------------------------------


class TestApplySteeringCPU:
    """Eager-mode behaviour of the registered ``apply_steering`` op."""

    def test_inactive_returns_clone_ignores_table(self):
        """Flag=False must short-circuit the gather entirely.

        We deliberately fill the table with garbage and the index with
        an arbitrary row; if the short-circuit isn't honoured, the
        garbage will leak into the result.
        """
        hidden = torch.randn(4, 8, dtype=torch.float32)
        table = torch.full((6, 8), float("nan"))  # garbage that would NaN
        index = torch.full((4,), 5, dtype=torch.long)  # row 5 = NaN row
        any_active = torch.zeros(1, dtype=torch.bool)

        result = apply_steering(
            hidden,
            table,
            index,
            any_active,
            torch.ones(table.shape[0], device=table.device),
            torch.zeros(table.shape[1], device=table.device),
            torch.zeros(hidden.shape[0], device=table.device),
            torch.ones(hidden.shape[0], device=table.device),
        )

        torch.testing.assert_close(result, hidden)
        # Output must be a fresh tensor, not an alias of ``hidden``.
        assert result.data_ptr() != hidden.data_ptr()

    def test_active_matches_indexed_gather(self):
        """Flag=True must match ``hidden + table[index]``."""
        torch.manual_seed(0)
        hidden = torch.randn(5, 16, dtype=torch.float32)
        table = torch.randn(8, 16, dtype=torch.float32)
        index = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
        any_active = torch.ones(1, dtype=torch.bool)

        result = apply_steering(
            hidden,
            table,
            index,
            any_active,
            torch.ones(table.shape[0], device=table.device),
            torch.zeros(table.shape[1], device=table.device),
            torch.zeros(hidden.shape[0], device=table.device),
            torch.ones(hidden.shape[0], device=table.device),
        )
        expected = hidden + table[index]

        torch.testing.assert_close(result, expected)

    def test_active_flag_true_with_zero_table_is_identity(self):
        """Flag=True + zero table must still equal hidden_states."""
        hidden = torch.randn(3, 4, dtype=torch.float32)
        table = torch.zeros(6, 4, dtype=torch.float32)
        index = torch.zeros(3, dtype=torch.long)
        any_active = torch.ones(1, dtype=torch.bool)

        result = apply_steering(
            hidden,
            table,
            index,
            any_active,
            torch.ones(table.shape[0], device=table.device),
            torch.zeros(table.shape[1], device=table.device),
            torch.zeros(hidden.shape[0], device=table.device),
            torch.ones(hidden.shape[0], device=table.device),
        )

        torch.testing.assert_close(result, hidden)

    def test_inactive_with_index_buffer_larger_than_batch(self):
        """Excess index entries must be ignored on the inactive path.

        Mirrors the ``_apply_steering`` invariant that only ``index[:N]``
        is read; on the inactive path it shouldn't matter at all.
        """
        hidden = torch.ones(3, 4, dtype=torch.float32)
        table = torch.full((6, 4), float("nan"))
        index = torch.zeros(100, dtype=torch.long)
        index[3:] = 999  # would be out-of-bounds on the active path
        any_active = torch.zeros(1, dtype=torch.bool)

        result = apply_steering(
            hidden,
            table,
            index,
            any_active,
            torch.ones(table.shape[0], device=table.device),
            torch.zeros(table.shape[1], device=table.device),
            torch.zeros(hidden.shape[0], device=table.device),
            torch.ones(hidden.shape[0], device=table.device),
        )
        torch.testing.assert_close(result, hidden)


# ---------------------------------------------------------------------------
# Buffer registration
# ---------------------------------------------------------------------------


class TestRegisterSteeringBuffersFlag:
    """``register_steering_buffers`` must allocate per-hook flag tensors."""

    def test_each_hook_gets_a_bool_flag(self):
        mod = nn.Module()
        register_steering_buffers(
            mod,
            hidden_size=8,
            max_steering_tokens=16,
            max_steering_configs=4,
        )
        for hp in SteeringHookPoint:
            flag_attr = HOOK_POINT_ANY_ACTIVE_ATTR[hp]
            flag = getattr(mod, flag_attr)
            assert flag.dtype == torch.bool
            assert flag.numel() == 1
            assert not bool(flag.item()), (
                f"Flag for {hp.value} must initialise to False so the "
                f"kernel short-circuits when no state has been populated."
            )

    def test_flag_attr_co_located_with_table(self):
        """The flag attribute must live on the same module as the table."""
        mod = nn.Module()
        register_steering_buffers(
            mod,
            hidden_size=8,
            max_steering_tokens=16,
            max_steering_configs=4,
        )
        for hp in SteeringHookPoint:
            assert hasattr(mod, HOOK_POINT_TABLE_ATTR[hp])
            assert hasattr(mod, HOOK_POINT_ANY_ACTIVE_ATTR[hp])


# ---------------------------------------------------------------------------
# CUDA Triton path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestApplySteeringCUDA:
    """Triton-kernel behaviour mirrors the eager path for both flag states."""

    def test_inactive_kernel_skips_gather(self):
        """Flag=False must produce ``hidden_states`` even with a NaN table."""
        device = torch.device("cuda")
        hidden = torch.randn(4, 64, dtype=torch.float16, device=device)
        table = torch.full((6, 64), float("nan"), dtype=torch.float16, device=device)
        index = torch.full((4,), 5, dtype=torch.long, device=device)
        any_active = torch.zeros(1, dtype=torch.bool, device=device)

        result = apply_steering(
            hidden,
            table,
            index,
            any_active,
            torch.ones(table.shape[0], device=table.device),
            torch.zeros(table.shape[1], device=table.device),
            torch.zeros(hidden.shape[0], device=table.device),
            torch.ones(hidden.shape[0], device=table.device),
        )

        torch.testing.assert_close(result, hidden)
        assert result.data_ptr() != hidden.data_ptr()

    def test_active_kernel_matches_eager(self):
        """Flag=True must match ``hidden + table[index]`` (cast included)."""
        device = torch.device("cuda")
        torch.manual_seed(0)
        hidden = torch.randn(5, 128, dtype=torch.float16, device=device)
        table = torch.randn(8, 128, dtype=torch.float16, device=device)
        index = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long, device=device)
        any_active = torch.ones(1, dtype=torch.bool, device=device)

        result = apply_steering(
            hidden,
            table,
            index,
            any_active,
            torch.ones(table.shape[0], device=table.device),
            torch.zeros(table.shape[1], device=table.device),
            torch.zeros(hidden.shape[0], device=table.device),
            torch.ones(hidden.shape[0], device=table.device),
        )
        expected = hidden + table[index]

        torch.testing.assert_close(result, expected)

    def test_kernel_handles_non_power_of_two_hidden(self):
        """The masked walk handles unusual hidden sizes on both paths."""
        device = torch.device("cuda")
        hidden = torch.randn(3, 17, dtype=torch.float16, device=device)
        table = torch.full((6, 17), float("nan"), dtype=torch.float16, device=device)
        index = torch.full((3,), 4, dtype=torch.long, device=device)
        any_active = torch.zeros(1, dtype=torch.bool, device=device)

        result = apply_steering(
            hidden,
            table,
            index,
            any_active,
            torch.ones(table.shape[0], device=table.device),
            torch.zeros(table.shape[1], device=table.device),
            torch.zeros(hidden.shape[0], device=table.device),
            torch.ones(hidden.shape[0], device=table.device),
        )
        torch.testing.assert_close(result, hidden)

        # And re-run with the flag on, replacing the table with sane values
        # so the sum is well-defined.
        table = torch.randn(6, 17, dtype=torch.float16, device=device)
        any_active.fill_(True)
        result = apply_steering(
            hidden,
            table,
            index,
            any_active,
            torch.ones(table.shape[0], device=table.device),
            torch.zeros(table.shape[1], device=table.device),
            torch.zeros(hidden.shape[0], device=table.device),
            torch.ones(hidden.shape[0], device=table.device),
        )
        expected = hidden + table[index]
        torch.testing.assert_close(result, expected)


class TestFusedMonitorCPU:
    """The in-graph monitor gate is fused into ``apply_steering`` (non-
    mutating). These exercise the CPU eager path of that fused gate."""

    @staticmethod
    def _ref(hidden, table, index, scales, dvec, tscale, rgate,
             probe, params, dmask, monitor_on):
        ts, rg = tscale.clone(), rgate.clone()
        if monitor_on:
            g = torch.sigmoid(params[1] * (hidden @ probe - params[0]))
            ts = ts * g
            if params[2].item() != 0.0:
                rg = rg * (dmask * g + (1.0 - dmask))
        row = table[index] * scales[index].unsqueeze(-1) * rg.unsqueeze(-1)
        tier = dvec.unsqueeze(0) * ts.unsqueeze(-1)
        return hidden + row + tier

    def _run(self, params, monitor_active, dmask, *, gate_rows_expect):
        torch.manual_seed(0)
        n, h = 4, 16
        hidden = torch.randn(n, h, dtype=torch.float32)
        table = torch.randn(6, h, dtype=torch.float32)
        table[0] = 0.0
        index = torch.tensor([1, 2, 3, 1], dtype=torch.long)
        any_active = torch.ones(1, dtype=torch.bool)
        scales = torch.ones(6)
        dvec = torch.randn(h)
        tscale = torch.full((n,), 2.0)  # nonzero tier so gating is visible
        rgate = torch.ones(n)
        probe = torch.randn(h)
        out = _real_apply_steering(
            hidden, table, index, any_active, scales, dvec, tscale, rgate,
            probe, params, monitor_active, dmask,
        )
        exp = self._ref(hidden, table, index, scales, dvec, tscale, rgate,
                        probe, params, dmask, bool(monitor_active.item()))
        torch.testing.assert_close(out, exp)

    def test_monitor_inactive_is_plain_steering(self):
        params = torch.tensor([0.0, 4.0, 1.0])
        self._run(params, torch.zeros(1, dtype=torch.bool),
                  torch.ones(4), gate_rows_expect=False)

    def test_monitor_active_gates_tier_only(self):
        # gate_rows = 0 ⇒ only the tier (token_scales) is gated, not rows.
        params = torch.tensor([0.0, 4.0, 0.0])
        self._run(params, torch.ones(1, dtype=torch.bool),
                  torch.ones(4), gate_rows_expect=False)

    def test_monitor_active_gates_rows_decode_only(self):
        # gate_rows = 1 ⇒ rows gated for decode tokens (mask 1), prefill stays.
        params = torch.tensor([0.0, 4.0, 1.0])
        dmask = torch.tensor([1.0, 1.0, 0.0, 0.0])  # 2 decode, 2 prefill
        self._run(params, torch.ones(1, dtype=torch.bool),
                  dmask, gate_rows_expect=True)


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-v"]))
