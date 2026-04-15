# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for :mod:`vllm.model_executor.layers.activation_capture`'s
custom op and cold-path constant folding.

All tests run on CPU. They exercise ``torch.ops.vllm.capture_residual``
and the :func:`maybe_capture_residual` gate directly, plus the
pristine-residual invariant from
:func:`vllm.model_executor.layers.steering.apply_layer_steering`.

Phase 3 done-when items:

1. op registered and callable
2. fake impl shape agreement
3. real impl is a no-op when no manager is installed
4. real impl forwards to ``on_hook`` when a manager is installed
5. ``maybe_capture_residual`` gate does not call the op when disabled
6. ``maybe_capture_residual`` gate invokes the op when enabled
7. ``torch._dynamo.export`` produces a graph without ``capture_residual``
   when no manager is installed (invariant 3)
8. ``torch._dynamo.export`` produces a graph with ``capture_residual``
   when a manager IS installed
9. ``apply_layer_steering`` calls capture with the *pristine* residual
   (invariant 2)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers import activation_capture as ac_module
from vllm.model_executor.layers.activation_capture import (
    _HOOK_NAME_TO_ID,
    ActivationCaptureManager,
    get_active_capture_manager,
    maybe_capture_residual,
    set_active_capture_manager,
)
from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
    apply_layer_steering,
    register_steering_buffers,
)

HIDDEN_SIZE = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def disabled_manager():
    """Assert the manager is None on entry and leave it None on exit."""
    # This fixture is defensive: a previous test leaving the global set
    # would taint every downstream constant-folding test. Fail loud.
    assert get_active_capture_manager() is None
    yield
    set_active_capture_manager(None)
    assert get_active_capture_manager() is None


@pytest.fixture
def fake_manager():
    """Install a MagicMock manager spec'd to ActivationCaptureManager.

    Yields the mock. Restores ``None`` on teardown.
    """
    mgr = MagicMock(spec=ActivationCaptureManager)
    set_active_capture_manager(mgr)
    try:
        yield mgr
    finally:
        set_active_capture_manager(None)


# ---------------------------------------------------------------------------
# 1. Op registered
# ---------------------------------------------------------------------------


def test_op_registered():
    assert hasattr(torch.ops.vllm, "capture_residual")
    # Must be callable with (Tensor, int, int).
    hidden = torch.zeros((2, HIDDEN_SIZE), dtype=torch.float32)
    # No manager -> returns input unchanged. Just making sure the call
    # signature binds.
    out = torch.ops.vllm.capture_residual(hidden, 0, _HOOK_NAME_TO_ID["pre_attn"])
    assert out.shape == hidden.shape


def test_hook_name_id_mapping_covers_all_hook_points():
    # Pin the full set of hook names the capture op understands. This
    # must stay in lockstep with ``SteeringHookPoint`` and
    # ``VALID_ACTIVATION_HOOK_NAMES``; the custom op encodes the name
    # as an int, so any drift would break the op dispatch silently.
    assert set(_HOOK_NAME_TO_ID.keys()) == {
        "pre_attn",
        "post_attn",
        "post_mlp",
        "mlp_in",
        "mlp_out",
    }
    # IDs must be dense starting at 0 and unique.
    assert sorted(_HOOK_NAME_TO_ID.values()) == list(range(len(_HOOK_NAME_TO_ID)))


def test_op_dispatches_all_hook_ids_round_trip_when_active(fake_manager):
    # Every known hook id must round-trip through the op, reach the
    # manager's ``on_hook``, and land there with the correct
    # string name. Regression guard against ``_HOOK_ID_TO_NAME``
    # drifting from ``_HOOK_NAME_TO_ID``.
    hidden = torch.randn((2, HIDDEN_SIZE), dtype=torch.float32)
    for hook_name, hook_id in _HOOK_NAME_TO_ID.items():
        fake_manager.on_hook.reset_mock()
        ac_module._capture_residual_impl(hidden, 3, hook_id)
        fake_manager.on_hook.assert_called_once()
        call_args = fake_manager.on_hook.call_args
        assert call_args.args[0] == 3
        assert call_args.args[1] == hook_name


# ---------------------------------------------------------------------------
# 2. Fake impl returns empty_like
# ---------------------------------------------------------------------------


def test_fake_impl_returns_empty_like():
    hidden = torch.zeros((4, HIDDEN_SIZE), dtype=torch.float32)
    result = ac_module._capture_residual_fake(hidden, 0, 0)
    assert result.shape == hidden.shape
    assert result.dtype == hidden.dtype


# ---------------------------------------------------------------------------
# 3-4. Real impl behavior under disabled / enabled manager
# ---------------------------------------------------------------------------


def test_real_impl_no_op_when_manager_none(disabled_manager):
    hidden = torch.randn((3, HIDDEN_SIZE), dtype=torch.float32)
    out = ac_module._capture_residual_impl(hidden, 7, _HOOK_NAME_TO_ID["post_mlp"])
    assert out is hidden


def test_real_impl_calls_manager_on_hook_when_active(fake_manager):
    hidden = torch.randn((3, HIDDEN_SIZE), dtype=torch.float32)

    ac_module._capture_residual_impl(hidden, 12, _HOOK_NAME_TO_ID["post_mlp"])

    fake_manager.on_hook.assert_called_once()
    call_args = fake_manager.on_hook.call_args
    # Positional args: (layer_idx, hook_name, hidden_states).
    assert call_args.args[0] == 12
    assert call_args.args[1] == "post_mlp"
    assert torch.equal(call_args.args[2], hidden)


# ---------------------------------------------------------------------------
# 5-6. maybe_capture_residual gate behavior
# ---------------------------------------------------------------------------


def test_maybe_capture_residual_skips_call_when_disabled(disabled_manager, monkeypatch):
    """With no manager installed, the gate must not dispatch the op."""
    call_count = {"n": 0}

    original = torch.ops.vllm.capture_residual

    def trap(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    # Patch the bound call on torch.ops.vllm by replacing the attr on
    # the Python-visible OpOverloadPacket. We wrap the existing binding
    # to observe invocations.
    monkeypatch.setattr(torch.ops.vllm, "capture_residual", trap)

    hidden = torch.randn((2, HIDDEN_SIZE), dtype=torch.float32)
    maybe_capture_residual(hidden, 0, "pre_attn")

    assert call_count["n"] == 0


def test_maybe_capture_residual_invokes_op_when_active(fake_manager):
    hidden = torch.randn((2, HIDDEN_SIZE), dtype=torch.float32)
    maybe_capture_residual(hidden, 4, "post_mlp")

    fake_manager.on_hook.assert_called_once()
    call_args = fake_manager.on_hook.call_args
    assert call_args.args[0] == 4
    assert call_args.args[1] == "post_mlp"


# ---------------------------------------------------------------------------
# 7-8. torch.compile / dynamo export constant folding
# ---------------------------------------------------------------------------


def _graph_has_capture_op(graph_module: torch.fx.GraphModule) -> bool:
    """Return True iff any node's target references ``capture_residual``."""
    for node in graph_module.graph.nodes:
        target = getattr(node, "target", None)
        if target is None:
            continue
        target_repr = repr(target)
        if "capture_residual" in target_repr:
            return True
    return False


def test_torch_compile_constant_folds_when_disabled(disabled_manager):
    """With no manager, ``torch._dynamo.export`` traces the gate to a
    no-op and the exported graph contains no ``capture_residual``
    nodes (spec invariant 3)."""

    def f(x):
        maybe_capture_residual(x, 0, "pre_attn")
        return x * 2

    x = torch.randn((4, HIDDEN_SIZE), dtype=torch.float32)
    try:
        exported = torch._dynamo.export(f)(x)
    except Exception as exc:
        pytest.skip(f"torch._dynamo.export unavailable on this build: {exc}")

    # Exported result is a tuple (graph_module, guards) on older APIs
    # or an ExportResult on newer ones. Handle both shapes.
    gm = exported.graph_module if hasattr(exported, "graph_module") else exported[0]

    assert not _graph_has_capture_op(gm), (
        "capture_residual leaked into compiled graph with no active manager"
    )


def test_torch_compile_keeps_op_when_active(fake_manager):
    """With a manager installed, the exported graph must preserve the
    ``capture_residual`` op. ``mutates_args=['hidden_states']`` on the
    registered op is what ensures it survives DCE even though its
    return value is discarded by the caller."""

    def f(x):
        maybe_capture_residual(x, 0, "pre_attn")
        return x * 2

    x = torch.randn((4, HIDDEN_SIZE), dtype=torch.float32)
    try:
        exported = torch._dynamo.export(f)(x)
    except Exception as exc:
        pytest.skip(f"torch._dynamo.export unavailable on this build: {exc}")

    gm = exported.graph_module if hasattr(exported, "graph_module") else exported[0]

    assert _graph_has_capture_op(gm), (
        "capture_residual was DCE'd out of the compiled graph even though a "
        "manager was installed; verify mutates_args=['hidden_states'] on the "
        "registered custom op"
    )


# ---------------------------------------------------------------------------
# 9. apply_layer_steering calls capture first (invariant 2)
# ---------------------------------------------------------------------------


class _FakeSteerableLayer(torch.nn.Module):
    """Minimal decoder-layer stub with steering buffers and layer_idx."""

    def __init__(self, layer_idx: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.layer_idx = layer_idx
        register_steering_buffers(
            self,
            hidden_size,
            max_steering_tokens=16,
            max_steering_configs=4,
        )
        # Mutate the POST_MLP table so the steering op makes a visible
        # difference between the pristine residual and the post-steering
        # residual.
        table = getattr(self, HOOK_POINT_TABLE_ATTR[SteeringHookPoint.POST_MLP])
        table[0] = 0.0  # no-steering sentinel row
        table[1] = torch.ones(hidden_size) * 7.0  # global row
        # Steering index default is zeros -> no steering applied.
        # Override it so the steering op adds 7.0 to every row.
        self.steering_index.zero_()
        self.steering_index[:] = 1  # all tokens use the global row


def test_apply_layer_steering_calls_capture_first(fake_manager):
    """Capture must observe the pristine residual — i.e., the
    ``hidden_states`` tensor before the steering vector is added."""

    layer = _FakeSteerableLayer(layer_idx=13)
    hidden = torch.randn((3, HIDDEN_SIZE), dtype=torch.float32)
    pristine = hidden.clone()

    out = apply_layer_steering(layer, hidden, SteeringHookPoint.POST_MLP)

    fake_manager.on_hook.assert_called_once()
    call_args = fake_manager.on_hook.call_args
    assert call_args.args[0] == 13
    assert call_args.args[1] == "post_mlp"
    captured_tensor = call_args.args[2]
    # The tensor handed to capture must equal the pristine residual,
    # NOT the post-steering value.
    assert torch.equal(captured_tensor, pristine), (
        "capture observed a post-steering residual; invariant 2 violated"
    )
    # And the steering op did run (out != pristine).
    assert not torch.equal(out, pristine), (
        "steering op did not apply; this test cannot prove pristine ordering"
    )


def test_apply_layer_steering_no_capture_when_disabled(disabled_manager):
    """Sanity check: with no manager installed, apply_layer_steering
    still works and produces the same output as plain steering."""
    layer = _FakeSteerableLayer(layer_idx=0)
    hidden = torch.randn((3, HIDDEN_SIZE), dtype=torch.float32)
    out = apply_layer_steering(layer, hidden, SteeringHookPoint.POST_MLP)
    # With steering_index=1 and table[1]=7.0, expected = hidden + 7.
    expected = hidden + 7.0
    assert torch.allclose(out, expected)
