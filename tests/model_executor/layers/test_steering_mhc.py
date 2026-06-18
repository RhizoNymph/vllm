# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for mHC (multi-stream) activation steering.

Covers the two framework additions that let DeepSeek-V4-style manifold
hyper-connection models steer at their mHC hook points:

* model-selective, per-width buffer registration
  (:func:`register_steering_buffers` with ``hook_widths``), and
* the multi-stream apply helper
  (:func:`apply_layer_steering_streams`), which flattens a
  ``(tokens, streams, hidden)`` residual, gathers/adds a per-stream
  steering row, and reshapes back.

All paths run on CPU via the eager ``apply_steering`` implementation.
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    HOOK_POINT_TABLE_ATTR,
    STANDARD_STEERING_HOOKS,
    SteeringHookPoint,
    apply_layer_steering_streams,
    apply_steering,
    register_steering_buffers,
)


def _op_cpu_dispatchable() -> bool:
    """Whether ``torch.ops.vllm.apply_steering`` runs on CPU in this build.

    The op is registered under ``current_platform.dispatch_key``, so on a
    CUDA build it has no CPU kernel. Tests that drive the op directly skip
    when it is unavailable; the eager ``apply_steering`` function (which the
    op wraps) is always CPU-runnable and carries the math coverage.
    """
    try:
        torch.ops.vllm.apply_steering(
            torch.zeros(1, 1),
            torch.zeros(3, 1),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.bool),
        )
    except (NotImplementedError, RuntimeError):
        return False
    return True

HIDDEN = 4
HC_MULT = 3
HC_DIM = HC_MULT * HIDDEN
MAX_CONFIGS = 2
NUM_ROWS = MAX_CONFIGS + 3


def _bare_module() -> nn.Module:
    mod = nn.Module()
    mod.layer_idx = 7
    return mod


class TestRegisterSteeringBuffers:
    """Model-selective, per-width buffer registration."""

    def test_default_registers_only_standard_hooks(self):
        mod = _bare_module()
        register_steering_buffers(
            mod,
            HIDDEN,
            max_steering_tokens=8,
            max_steering_configs=MAX_CONFIGS,
            dtype=torch.float32,
        )
        for hp in STANDARD_STEERING_HOOKS:
            table = getattr(mod, HOOK_POINT_TABLE_ATTR[hp])
            assert table.shape == (NUM_ROWS, HIDDEN)
        # mHC hooks must NOT be registered on a standard model.
        for hp in (
            SteeringHookPoint.MLP_IN,
            SteeringHookPoint.MHC_STREAMS_PRE_ATTN,
            SteeringHookPoint.MHC_STREAMS_FINAL,
        ):
            assert not hasattr(mod, HOOK_POINT_TABLE_ATTR[hp])

    def test_hook_widths_selects_and_sizes_tables(self):
        mod = _bare_module()
        register_steering_buffers(
            mod,
            HIDDEN,
            max_steering_tokens=8,
            max_steering_configs=MAX_CONFIGS,
            dtype=torch.float32,
            hook_widths={
                SteeringHookPoint.PRE_ATTN: HIDDEN,
                SteeringHookPoint.MHC_STREAMS_PRE_ATTN: HC_DIM,
            },
        )
        single = getattr(mod, HOOK_POINT_TABLE_ATTR[SteeringHookPoint.PRE_ATTN])
        multi = getattr(
            mod, HOOK_POINT_TABLE_ATTR[SteeringHookPoint.MHC_STREAMS_PRE_ATTN]
        )
        assert single.shape == (NUM_ROWS, HIDDEN)
        assert multi.shape == (NUM_ROWS, HC_DIM)
        # A hook absent from the map gets no table.
        assert not hasattr(mod, HOOK_POINT_TABLE_ATTR[SteeringHookPoint.POST_MLP])
        # The any-active flags and shared index come along too.
        assert hasattr(
            mod, HOOK_POINT_ANY_ACTIVE_ATTR[SteeringHookPoint.MHC_STREAMS_PRE_ATTN]
        )
        assert hasattr(mod, "steering_index")

    def test_disabled_engine_registers_nothing(self):
        mod = _bare_module()
        register_steering_buffers(
            mod,
            HIDDEN,
            max_steering_tokens=8,
            max_steering_configs=0,
            dtype=torch.float32,
            hook_widths={SteeringHookPoint.MHC_STREAMS_PRE_ATTN: HC_DIM},
        )
        assert not hasattr(
            mod, HOOK_POINT_TABLE_ATTR[SteeringHookPoint.MHC_STREAMS_PRE_ATTN]
        )
        assert not hasattr(mod, "steering_index")


class TestApplyLayerSteeringStreams:
    """Flatten / gather-add / reshape over a multi-stream residual."""

    def _module_with_streams_table(self) -> nn.Module:
        mod = _bare_module()
        register_steering_buffers(
            mod,
            HIDDEN,
            max_steering_tokens=8,
            max_steering_configs=MAX_CONFIGS,
            dtype=torch.float32,
            hook_widths={SteeringHookPoint.MHC_STREAMS_PRE_ATTN: HC_DIM},
        )
        return mod

    def test_per_stream_layout_contract(self):
        """A flattened per-stream row lands on the matching stream.

        Drives the eager ``apply_steering`` function (CPU-safe) composed with
        the same flatten/reshape ``apply_layer_steering_streams`` performs, so
        the per-stream layout is verified regardless of whether the op has a
        CPU dispatch in this build. A distinct value per stream makes a
        broadcast bug visible.
        """
        per_stream = torch.arange(HC_DIM, dtype=torch.float32) + 1.0
        table = torch.zeros(NUM_ROWS, HC_DIM)
        table[3] = per_stream
        flag = torch.ones(1, dtype=torch.bool)
        n = 5
        index = torch.full((n,), 3, dtype=torch.long)
        streams = torch.randn(n, HC_MULT, HIDDEN)

        steered = apply_steering(streams.flatten(1), table, index, flag).view_as(
            streams
        )

        assert steered.shape == streams.shape
        assert torch.allclose(steered, streams + per_stream.view(HC_MULT, HIDDEN))

    def test_missing_table_returns_input_unchanged(self):
        """Steering disabled for this hook -> static short-circuit (no op)."""
        mod = _bare_module()  # no buffers registered at all
        streams = torch.randn(3, HC_MULT, HIDDEN)
        out = apply_layer_steering_streams(
            mod, streams, SteeringHookPoint.MHC_STREAMS_PRE_ATTN
        )
        assert out is streams

    @pytest.mark.skipif(
        not _op_cpu_dispatchable(),
        reason="apply_steering op has no CPU dispatch in this build",
    )
    def test_helper_active_path_matches_reference(self):
        """The full helper (flatten -> op -> reshape) on the active path."""
        mod = self._module_with_streams_table()
        hp = SteeringHookPoint.MHC_STREAMS_PRE_ATTN
        per_stream = torch.arange(HC_DIM, dtype=torch.float32) + 1.0
        getattr(mod, HOOK_POINT_TABLE_ATTR[hp])[3] = per_stream
        getattr(mod, HOOK_POINT_ANY_ACTIVE_ATTR[hp]).fill_(True)
        n = 5
        mod.steering_index[:n] = 3
        streams = torch.randn(n, HC_MULT, HIDDEN)

        out = apply_layer_steering_streams(mod, streams, hp)

        assert out.shape == streams.shape
        assert torch.allclose(out, streams + per_stream.view(HC_MULT, HIDDEN))

    @pytest.mark.skipif(
        not _op_cpu_dispatchable(),
        reason="apply_steering op has no CPU dispatch in this build",
    )
    def test_helper_inactive_flag_is_noop(self):
        mod = self._module_with_streams_table()
        hp = SteeringHookPoint.MHC_STREAMS_PRE_ATTN
        getattr(mod, HOOK_POINT_TABLE_ATTR[hp])[3] = 99.0
        # any_active stays False -> emits hidden_states unchanged.
        n = 4
        mod.steering_index[:n] = 3
        streams = torch.randn(n, HC_MULT, HIDDEN)
        out = apply_layer_steering_streams(mod, streams, hp)
        assert torch.allclose(out, streams)
