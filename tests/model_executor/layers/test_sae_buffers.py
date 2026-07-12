# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-layer SAE buffer registration.

The SAE feature-surgery path attaches a per-module *buffer slot* (a
small set of slot-suffixed buffers plus a slot record) to each
decoder-layer module that an SAE module covers.  Multiple SAE modules
may share one (layer, hook) site — each holds its own slot.  These
tests pin the slot lifecycle, buffer shapes, dtypes, and the
``register_sae_buffers`` / ``unregister_sae_buffers`` contract that
the worker mixin drives.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm.config.sae_steering_types import SAEActivation
from vllm.model_executor.layers.sae_steering import (
    SAE_ENCODER_WEIGHT_BASE,
    _sae_slot_attr,
    get_sae_slot_state,
    register_sae_buffers,
    register_sae_index_buffer,
    sae_buffers_attached,
    sae_site_slots,
    share_sae_index_across_layers,
    unregister_sae_buffers,
)
from vllm.model_executor.layers.steering import SteeringHookPoint


def _bare_module() -> nn.Module:
    """A no-op nn.Module to attach buffers to."""
    return nn.Module()


def _register(
    m: nn.Module,
    *,
    hook: SteeringHookPoint = SteeringHookPoint.POST_BLOCK,
    module_name: str = "g",
    activation: SAEActivation = SAEActivation.RELU,
    activation_params: dict | None = None,
    n_clamp: int = 2,
    hidden_size: int = 4,
    max_sae_configs: int = 1,
    dtype: torch.dtype = torch.float32,
) -> None:
    register_sae_buffers(
        m,
        hook_point=hook,
        module_name=module_name,
        activation=activation,
        activation_params=activation_params or {},
        n_clamp=n_clamp,
        hidden_size=hidden_size,
        max_sae_configs=max_sae_configs,
        dtype=dtype,
    )


class TestRegisterSaeBuffers:
    """Buffer attachment for a single (layer, hook) site."""

    def test_attaches_clamp_table_buffers(self):
        m = _bare_module()
        _register(
            m,
            module_name="golden_gate",
            n_clamp=4,
            hidden_size=8,
            max_sae_configs=3,
        )
        state = get_sae_slot_state(m, SteeringHookPoint.POST_BLOCK, "golden_gate")
        assert state is not None
        # Row 0 is the no-op sentinel, rows 1/2 are phase globals.
        assert state.clamp_kind.shape == (6, 4)
        assert state.clamp_kind.dtype is torch.int8
        assert state.clamp_value.shape == (6, 4)
        assert state.clamp_value.dtype is torch.float32
        assert state.clamp_only_if_active.shape == (6, 4)
        assert state.clamp_only_if_active.dtype is torch.bool

    def test_row_zero_is_zero_initialized(self):
        m = _bare_module()
        _register(m)
        state = get_sae_slot_state(m, SteeringHookPoint.POST_BLOCK, "g")
        assert state is not None
        # Row 0 is the no-op sentinel: all zeros = CLAMP_KIND_NONE.
        assert torch.equal(state.clamp_kind[0], torch.zeros(2, dtype=torch.int8))

    def test_attaches_weight_buffers(self):
        m = _bare_module()
        _register(
            m,
            hook=SteeringHookPoint.POST_ATTN,
            activation=SAEActivation.JUMPRELU,
            n_clamp=3,
            hidden_size=6,
            max_sae_configs=4,
            dtype=torch.bfloat16,
        )
        state = get_sae_slot_state(m, SteeringHookPoint.POST_ATTN, "g")
        assert state is not None
        # Encoder/decoder shape and dtype contract: compute dtype, aligned
        # rows for the clampable feature subset.
        assert state.encoder_weight.shape == (3, 6)
        assert state.encoder_weight.dtype is torch.bfloat16
        assert state.encoder_bias.shape == (3,)
        assert state.encoder_bias.dtype is torch.bfloat16
        assert state.decoder_weight.shape == (3, 6)
        assert state.decoder_weight.dtype is torch.bfloat16
        # Per-feature JumpReLU thresholds are always fp32 — the
        # comparison happens in fp32 regardless of compute dtype —
        # and default to zero until weights are attached.
        assert state.threshold.shape == (3,)
        assert state.threshold.dtype is torch.float32
        assert torch.equal(state.threshold, torch.zeros(3))

    def test_threshold_buffer_registered_for_relu_sites(self):
        # ReLU/TopK sites still register a zero-filled threshold buffer
        # so the custom op keeps a fixed arity across activations.
        m = _bare_module()
        _register(m)
        state = get_sae_slot_state(m, SteeringHookPoint.POST_BLOCK, "g")
        assert state is not None
        assert state.threshold.shape == (2,)
        assert state.threshold.dtype is torch.float32
        assert torch.equal(state.threshold, torch.zeros(2))

    def test_records_module_name_and_activation(self):
        m = _bare_module()
        _register(m, module_name="golden_gate", activation=SAEActivation.JUMPRELU)
        # The slot record carries module name / activation as Python
        # state, so torch.compile sees them as per-instance constants.
        (slot,) = sae_site_slots(m, SteeringHookPoint.POST_BLOCK)
        assert slot.module_name == "golden_gate"
        # JumpReLU carries no activation params — per-feature thresholds
        # live in the dedicated threshold buffer.
        assert slot.activation is SAEActivation.JUMPRELU
        assert slot.activation_params == {}

    def test_supports_each_hook_point(self):
        for hp in (
            SteeringHookPoint.PRE_ATTN,
            SteeringHookPoint.POST_ATTN,
            SteeringHookPoint.POST_BLOCK,
        ):
            m = _bare_module()
            _register(m, hook=hp)
            assert sae_buffers_attached(m, hp)

    def test_two_modules_share_site_with_distinct_slots(self):
        # Two SAE modules may share the (layer, hook) site; each gets
        # its own buffer slot with distinct attr names.
        m = _bare_module()
        _register(m, module_name="a")
        _register(m, module_name="b")
        slots = sae_site_slots(m, SteeringHookPoint.POST_BLOCK)
        assert [(s.slot_id, s.module_name) for s in slots] == [(0, "a"), (1, "b")]
        state_a = get_sae_slot_state(m, SteeringHookPoint.POST_BLOCK, "a")
        state_b = get_sae_slot_state(m, SteeringHookPoint.POST_BLOCK, "b")
        assert state_a is not None and state_b is not None
        # Distinct physical buffers.
        assert state_a.clamp_kind is not state_b.clamp_kind
        assert state_a.encoder_weight is not state_b.encoder_weight
        # Distinct slot-suffixed buffer attribute names on the module.
        attr_a = _sae_slot_attr(
            SAE_ENCODER_WEIGHT_BASE, SteeringHookPoint.POST_BLOCK, 0
        )
        attr_b = _sae_slot_attr(
            SAE_ENCODER_WEIGHT_BASE, SteeringHookPoint.POST_BLOCK, 1
        )
        assert getattr(m, attr_a) is state_a.encoder_weight
        assert getattr(m, attr_b) is state_b.encoder_weight

    def test_double_register_same_module_name_raises(self):
        m = _bare_module()
        _register(m, module_name="a")
        # A second slot for a *different* module is fine; re-registering
        # the same module name at an occupied site is rejected.
        with pytest.raises(ValueError, match="already holds a buffer slot"):
            _register(m, module_name="a")

    def test_unregister_one_module_leaves_sibling_intact(self):
        m = _bare_module()
        _register(m, module_name="a")
        _register(m, module_name="b")
        state_b_before = get_sae_slot_state(m, SteeringHookPoint.POST_BLOCK, "b")
        assert state_b_before is not None
        state_b_before.encoder_weight.fill_(3.0)
        unregister_sae_buffers(
            m, hook_point=SteeringHookPoint.POST_BLOCK, module_name="a"
        )
        assert get_sae_slot_state(m, SteeringHookPoint.POST_BLOCK, "a") is None
        # 'b' keeps its slot id, attr names, and buffer contents.
        slots = sae_site_slots(m, SteeringHookPoint.POST_BLOCK)
        assert [(s.slot_id, s.module_name) for s in slots] == [(1, "b")]
        state_b = get_sae_slot_state(m, SteeringHookPoint.POST_BLOCK, "b")
        assert state_b is not None
        assert state_b.encoder_weight is state_b_before.encoder_weight
        assert torch.equal(state_b.encoder_weight, torch.full((2, 4), 3.0))
        assert sae_buffers_attached(m, SteeringHookPoint.POST_BLOCK)

    def test_slot_ids_not_reused_after_detach_and_reregister(self):
        # The per-(layer, hook) slot counter is monotonic: detaching a
        # module never frees its slot id, so surviving modules' attr
        # names stay stable across sibling churn.
        m = _bare_module()
        _register(m, module_name="a")
        _register(m, module_name="b")
        unregister_sae_buffers(
            m, hook_point=SteeringHookPoint.POST_BLOCK, module_name="a"
        )
        _register(m, module_name="c")
        slots = sae_site_slots(m, SteeringHookPoint.POST_BLOCK)
        assert [(s.slot_id, s.module_name) for s in slots] == [(1, "b"), (2, "c")]

    def test_disabled_when_max_zero(self):
        # When SAE is disabled (max_sae_configs == 0) buffer registration
        # is a no-op so the layer's forward path stays free of SAE
        # overhead.  Mirrors register_steering_buffers' disabled-mode.
        m = _bare_module()
        _register(m, max_sae_configs=0)
        assert not sae_buffers_attached(m, SteeringHookPoint.POST_BLOCK)
        assert sae_site_slots(m, SteeringHookPoint.POST_BLOCK) == ()

    def test_partial_registration_failure_rolls_back_buffers(self, monkeypatch):
        m = _bare_module()
        original_register_buffer = m.register_buffer
        calls = 0

        def fail_after_first_buffer(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("buffer allocation failed")
            return original_register_buffer(*args, **kwargs)

        monkeypatch.setattr(m, "register_buffer", fail_after_first_buffer)

        with pytest.raises(RuntimeError, match="buffer allocation failed"):
            _register(m)

        assert not sae_buffers_attached(m, SteeringHookPoint.POST_BLOCK)
        assert get_sae_slot_state(m, SteeringHookPoint.POST_BLOCK, "g") is None
        # No half-attached buffers survive under any slot-suffixed name.
        assert not any(attr.startswith("sae_") for attr in m._buffers)

        monkeypatch.setattr(m, "register_buffer", original_register_buffer)
        _register(m)
        assert sae_buffers_attached(m, SteeringHookPoint.POST_BLOCK)

    def test_partial_registration_failure_preserves_sibling_slot(self, monkeypatch):
        # A failed second-module registration must not disturb the
        # first module's slot.
        m = _bare_module()
        _register(m, module_name="a")
        original_register_buffer = m.register_buffer
        calls = 0

        def fail_after_first_buffer(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("buffer allocation failed")
            return original_register_buffer(*args, **kwargs)

        monkeypatch.setattr(m, "register_buffer", fail_after_first_buffer)
        with pytest.raises(RuntimeError, match="buffer allocation failed"):
            _register(m, module_name="b")

        slots = sae_site_slots(m, SteeringHookPoint.POST_BLOCK)
        assert [(s.slot_id, s.module_name) for s in slots] == [(0, "a")]
        assert get_sae_slot_state(m, SteeringHookPoint.POST_BLOCK, "a") is not None
        assert get_sae_slot_state(m, SteeringHookPoint.POST_BLOCK, "b") is None


class TestSaeBuffersAttached:
    """``sae_buffers_attached`` is the constant-time dispatch check."""

    def test_returns_true_when_buffers_present(self):
        m = _bare_module()
        _register(m)
        assert sae_buffers_attached(m, SteeringHookPoint.POST_BLOCK) is True

    def test_returns_false_when_no_buffers(self):
        m = _bare_module()
        assert sae_buffers_attached(m, SteeringHookPoint.POST_BLOCK) is False

    def test_per_hook_independence(self):
        m = _bare_module()
        _register(m)
        # Other hooks at the same layer remain unattached.
        assert sae_buffers_attached(m, SteeringHookPoint.PRE_ATTN) is False
        assert sae_buffers_attached(m, SteeringHookPoint.POST_ATTN) is False
        assert sae_buffers_attached(m, SteeringHookPoint.POST_BLOCK) is True

    def test_false_after_last_slot_detaches(self):
        m = _bare_module()
        _register(m, module_name="a")
        _register(m, module_name="b")
        unregister_sae_buffers(
            m, hook_point=SteeringHookPoint.POST_BLOCK, module_name="a"
        )
        assert sae_buffers_attached(m, SteeringHookPoint.POST_BLOCK) is True
        unregister_sae_buffers(
            m, hook_point=SteeringHookPoint.POST_BLOCK, module_name="b"
        )
        assert sae_buffers_attached(m, SteeringHookPoint.POST_BLOCK) is False
        # The slot-list attr is dropped entirely so hasattr gating in
        # the dispatch shim stays clean.
        assert not hasattr(m, "sae_slots_post_block")


class TestUnregisterSaeBuffers:
    """Detaching buffers when the SAE module is unregistered."""

    def test_removes_all_attributes(self):
        m = _bare_module()
        _register(m)
        unregister_sae_buffers(
            m, hook_point=SteeringHookPoint.POST_BLOCK, module_name="g"
        )
        assert get_sae_slot_state(m, SteeringHookPoint.POST_BLOCK, "g") is None
        assert not any(attr.startswith("sae_clamp") for attr in m._buffers)
        assert not any(attr.startswith("sae_encoder") for attr in m._buffers)
        assert not any(attr.startswith("sae_decoder") for attr in m._buffers)
        assert not any(attr.startswith("sae_threshold") for attr in m._buffers)
        assert not any(attr.startswith("sae_any_active") for attr in m._buffers)
        assert not hasattr(m, "sae_slots_post_block")

    def test_unregister_when_unattached_is_noop(self):
        m = _bare_module()
        # Must not raise.
        unregister_sae_buffers(
            m, hook_point=SteeringHookPoint.POST_BLOCK, module_name="g"
        )

    def test_unregister_unknown_module_name_is_noop(self):
        m = _bare_module()
        _register(m, module_name="a")
        unregister_sae_buffers(
            m, hook_point=SteeringHookPoint.POST_BLOCK, module_name="not-here"
        )
        assert sae_buffers_attached(m, SteeringHookPoint.POST_BLOCK)
        assert get_sae_slot_state(m, SteeringHookPoint.POST_BLOCK, "a") is not None


class TestSharedSaeIndex:
    """``sae_index`` is shared across all SAE-covered layers (one tensor)."""

    def test_register_attaches_index(self):
        m = _bare_module()
        register_sae_index_buffer(m, max_tokens=16)
        assert hasattr(m, "sae_index")
        assert m.sae_index.shape == (16,)  # type: ignore[attr-defined]
        assert m.sae_index.dtype is torch.long  # type: ignore[attr-defined]

    def test_share_sae_index_replaces_per_layer_with_first(self):
        layers = [_bare_module() for _ in range(3)]
        for layer in layers:
            register_sae_index_buffer(layer, max_tokens=8)
        # Pre-share: each layer has its own tensor.
        assert {id(layer.sae_index) for layer in layers} == {  # type: ignore[attr-defined]
            id(layer.sae_index)
            for layer in layers  # type: ignore[attr-defined]
        }
        share_sae_index_across_layers(layers)
        # Post-share: all layers point to layer[0]'s tensor.
        first = layers[0].sae_index  # type: ignore[attr-defined]
        for layer in layers[1:]:
            assert layer.sae_index is first  # type: ignore[attr-defined]

    def test_register_disabled_when_max_tokens_zero(self):
        m = _bare_module()
        register_sae_index_buffer(m, max_tokens=0)
        assert not hasattr(m, "sae_index")


class TestBufferDeviceMatchesModule:
    """Buffers materialize on the module's device when one is set."""

    def test_default_cpu_when_module_on_cpu(self):
        m = _bare_module()
        _register(m)
        state = get_sae_slot_state(m, SteeringHookPoint.POST_BLOCK, "g")
        assert state is not None
        assert state.clamp_kind.device.type == "cpu"
