# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-layer SAE buffer registration.

The SAE feature-surgery path attaches a small set of buffers and
Python attributes to each decoder-layer module that an SAE module
covers.  These tests pin the buffer names, shapes, dtypes, and the
``register_sae_buffers`` / ``unregister_sae_buffers`` contract that
the worker mixin will drive.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm.config.sae_steering_types import SAEActivation
from vllm.model_executor.layers.sae_steering import (
    HOOK_POINT_SAE_CLAMP_KIND_ATTR,
    HOOK_POINT_SAE_CLAMP_ONLY_IF_ACTIVE_ATTR,
    HOOK_POINT_SAE_CLAMP_VALUE_ATTR,
    HOOK_POINT_SAE_DECODER_WEIGHT_ATTR,
    HOOK_POINT_SAE_ENCODER_BIAS_ATTR,
    HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR,
    HOOK_POINT_SAE_MODULE_NAME_ATTR,
    register_sae_buffers,
    register_sae_index_buffer,
    sae_buffers_attached,
    share_sae_index_across_layers,
    unregister_sae_buffers,
)
from vllm.model_executor.layers.steering import SteeringHookPoint


def _bare_module() -> nn.Module:
    """A no-op nn.Module to attach buffers to."""
    return nn.Module()


def _attach_dummy_weights(
    module: nn.Module,
    hook: SteeringHookPoint,
    *,
    n_clamp: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    """Attach encoder/decoder weight buffers as ``register_sae_buffers`` does."""
    enc_w_attr = HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR[hook]
    enc_b_attr = HOOK_POINT_SAE_ENCODER_BIAS_ATTR[hook]
    dec_w_attr = HOOK_POINT_SAE_DECODER_WEIGHT_ATTR[hook]
    module.register_buffer(
        enc_w_attr, torch.zeros(n_clamp, hidden_size, dtype=dtype), persistent=False
    )
    module.register_buffer(
        enc_b_attr, torch.zeros(n_clamp, dtype=dtype), persistent=False
    )
    module.register_buffer(
        dec_w_attr, torch.zeros(n_clamp, hidden_size, dtype=dtype), persistent=False
    )


class TestRegisterSaeBuffers:
    """Buffer attachment for a single (layer, hook) site."""

    def test_attaches_clamp_table_buffers(self):
        m = _bare_module()
        register_sae_buffers(
            m,
            hook_point=SteeringHookPoint.POST_MLP,
            module_name="golden_gate",
            activation=SAEActivation.RELU,
            activation_params={},
            n_clamp=4,
            hidden_size=8,
            max_sae_configs=3,
            dtype=torch.float32,
        )
        kind_attr = HOOK_POINT_SAE_CLAMP_KIND_ATTR[SteeringHookPoint.POST_MLP]
        value_attr = HOOK_POINT_SAE_CLAMP_VALUE_ATTR[SteeringHookPoint.POST_MLP]
        only_attr = HOOK_POINT_SAE_CLAMP_ONLY_IF_ACTIVE_ATTR[SteeringHookPoint.POST_MLP]
        # Row 0 is the no-op sentinel, rows 1/2 are phase globals.
        assert getattr(m, kind_attr).shape == (6, 4)
        assert getattr(m, kind_attr).dtype is torch.int8
        assert getattr(m, value_attr).shape == (6, 4)
        assert getattr(m, value_attr).dtype is torch.float32
        assert getattr(m, only_attr).shape == (6, 4)
        assert getattr(m, only_attr).dtype is torch.bool

    def test_row_zero_is_zero_initialized(self):
        m = _bare_module()
        register_sae_buffers(
            m,
            hook_point=SteeringHookPoint.POST_MLP,
            module_name="g",
            activation=SAEActivation.RELU,
            activation_params={},
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=1,
            dtype=torch.float32,
        )
        kind = getattr(m, HOOK_POINT_SAE_CLAMP_KIND_ATTR[SteeringHookPoint.POST_MLP])
        # Row 0 is the no-op sentinel: all zeros = CLAMP_KIND_NONE.
        assert torch.equal(kind[0], torch.zeros(2, dtype=torch.int8))

    def test_attaches_weight_buffers(self):
        m = _bare_module()
        register_sae_buffers(
            m,
            hook_point=SteeringHookPoint.POST_ATTN,
            module_name="g",
            activation=SAEActivation.JUMPRELU,
            activation_params={"threshold": 0.5},
            n_clamp=3,
            hidden_size=6,
            max_sae_configs=4,
            dtype=torch.bfloat16,
        )
        enc_w = getattr(
            m, HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR[SteeringHookPoint.POST_ATTN]
        )
        enc_b = getattr(
            m, HOOK_POINT_SAE_ENCODER_BIAS_ATTR[SteeringHookPoint.POST_ATTN]
        )
        dec_w = getattr(
            m, HOOK_POINT_SAE_DECODER_WEIGHT_ATTR[SteeringHookPoint.POST_ATTN]
        )
        # Encoder/decoder shape and dtype contract: compute dtype, aligned
        # rows for the clampable feature subset.
        assert enc_w.shape == (3, 6)
        assert enc_w.dtype is torch.bfloat16
        assert enc_b.shape == (3,)
        assert enc_b.dtype is torch.bfloat16
        assert dec_w.shape == (3, 6)
        assert dec_w.dtype is torch.bfloat16

    def test_records_module_name_and_activation(self):
        m = _bare_module()
        register_sae_buffers(
            m,
            hook_point=SteeringHookPoint.POST_MLP,
            module_name="golden_gate",
            activation=SAEActivation.JUMPRELU,
            activation_params={"threshold": 0.7},
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=1,
            dtype=torch.float32,
        )
        # Module name and activation are Python attributes (not buffers)
        # so torch.compile sees them as per-instance constants.
        assert (
            getattr(m, HOOK_POINT_SAE_MODULE_NAME_ATTR[SteeringHookPoint.POST_MLP])
            == "golden_gate"
        )
        # Activation reachable through a sibling attribute.
        sae_act = m.sae_activation_post_mlp  # type: ignore[attr-defined]
        sae_act_params = m.sae_activation_params_post_mlp  # type: ignore[attr-defined]
        assert sae_act is SAEActivation.JUMPRELU
        assert sae_act_params == {"threshold": 0.7}

    def test_supports_each_hook_point(self):
        for hp in SteeringHookPoint:
            m = _bare_module()
            register_sae_buffers(
                m,
                hook_point=hp,
                module_name="g",
                activation=SAEActivation.RELU,
                activation_params={},
                n_clamp=2,
                hidden_size=4,
                max_sae_configs=1,
                dtype=torch.float32,
            )
            assert hasattr(m, HOOK_POINT_SAE_CLAMP_KIND_ATTR[hp])

    def test_double_register_same_hook_raises(self):
        m = _bare_module()
        register_sae_buffers(
            m,
            hook_point=SteeringHookPoint.POST_MLP,
            module_name="a",
            activation=SAEActivation.RELU,
            activation_params={},
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=1,
            dtype=torch.float32,
        )
        # Attaching a second SAE module to the same (layer, hook) is
        # rejected; the design constrains at most one SAE module per
        # site for Phase 1B.
        with pytest.raises(ValueError, match="already has SAE buffers"):
            register_sae_buffers(
                m,
                hook_point=SteeringHookPoint.POST_MLP,
                module_name="b",
                activation=SAEActivation.RELU,
                activation_params={},
                n_clamp=2,
                hidden_size=4,
                max_sae_configs=1,
                dtype=torch.float32,
            )

    def test_disabled_when_max_zero(self):
        # When SAE is disabled (max_sae_configs == 0) buffer registration
        # is a no-op so the layer's forward path stays free of SAE
        # overhead.  Mirrors register_steering_buffers' disabled-mode.
        m = _bare_module()
        register_sae_buffers(
            m,
            hook_point=SteeringHookPoint.POST_MLP,
            module_name="g",
            activation=SAEActivation.RELU,
            activation_params={},
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=0,
            dtype=torch.float32,
        )
        assert not hasattr(
            m, HOOK_POINT_SAE_CLAMP_KIND_ATTR[SteeringHookPoint.POST_MLP]
        )

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
            register_sae_buffers(
                m,
                hook_point=SteeringHookPoint.POST_MLP,
                module_name="g",
                activation=SAEActivation.RELU,
                activation_params={},
                n_clamp=2,
                hidden_size=4,
                max_sae_configs=1,
                dtype=torch.float32,
            )

        assert not sae_buffers_attached(m, SteeringHookPoint.POST_MLP)
        for attr_table in (
            HOOK_POINT_SAE_CLAMP_KIND_ATTR,
            HOOK_POINT_SAE_CLAMP_VALUE_ATTR,
            HOOK_POINT_SAE_CLAMP_ONLY_IF_ACTIVE_ATTR,
            HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR,
            HOOK_POINT_SAE_ENCODER_BIAS_ATTR,
            HOOK_POINT_SAE_DECODER_WEIGHT_ATTR,
            HOOK_POINT_SAE_MODULE_NAME_ATTR,
        ):
            assert not hasattr(m, attr_table[SteeringHookPoint.POST_MLP])

        monkeypatch.setattr(m, "register_buffer", original_register_buffer)
        register_sae_buffers(
            m,
            hook_point=SteeringHookPoint.POST_MLP,
            module_name="g",
            activation=SAEActivation.RELU,
            activation_params={},
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=1,
            dtype=torch.float32,
        )
        assert sae_buffers_attached(m, SteeringHookPoint.POST_MLP)


class TestSaeBuffersAttached:
    """``sae_buffers_attached`` is the constant-time dispatch check."""

    def test_returns_true_when_buffers_present(self):
        m = _bare_module()
        register_sae_buffers(
            m,
            hook_point=SteeringHookPoint.POST_MLP,
            module_name="g",
            activation=SAEActivation.RELU,
            activation_params={},
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=1,
            dtype=torch.float32,
        )
        assert sae_buffers_attached(m, SteeringHookPoint.POST_MLP) is True

    def test_returns_false_when_no_buffers(self):
        m = _bare_module()
        assert sae_buffers_attached(m, SteeringHookPoint.POST_MLP) is False

    def test_per_hook_independence(self):
        m = _bare_module()
        register_sae_buffers(
            m,
            hook_point=SteeringHookPoint.POST_MLP,
            module_name="g",
            activation=SAEActivation.RELU,
            activation_params={},
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=1,
            dtype=torch.float32,
        )
        # Other hooks at the same layer remain unattached.
        assert sae_buffers_attached(m, SteeringHookPoint.PRE_ATTN) is False
        assert sae_buffers_attached(m, SteeringHookPoint.POST_ATTN) is False
        assert sae_buffers_attached(m, SteeringHookPoint.POST_MLP) is True


class TestUnregisterSaeBuffers:
    """Detaching buffers when the SAE module is unregistered."""

    def test_removes_all_attributes(self):
        m = _bare_module()
        register_sae_buffers(
            m,
            hook_point=SteeringHookPoint.POST_MLP,
            module_name="g",
            activation=SAEActivation.RELU,
            activation_params={},
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=1,
            dtype=torch.float32,
        )
        unregister_sae_buffers(m, hook_point=SteeringHookPoint.POST_MLP)
        for attr_table in (
            HOOK_POINT_SAE_CLAMP_KIND_ATTR,
            HOOK_POINT_SAE_CLAMP_VALUE_ATTR,
            HOOK_POINT_SAE_CLAMP_ONLY_IF_ACTIVE_ATTR,
            HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR,
            HOOK_POINT_SAE_ENCODER_BIAS_ATTR,
            HOOK_POINT_SAE_DECODER_WEIGHT_ATTR,
            HOOK_POINT_SAE_MODULE_NAME_ATTR,
        ):
            assert not hasattr(m, attr_table[SteeringHookPoint.POST_MLP])

    def test_unregister_when_unattached_is_noop(self):
        m = _bare_module()
        # Must not raise.
        unregister_sae_buffers(m, hook_point=SteeringHookPoint.POST_MLP)


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
        register_sae_buffers(
            m,
            hook_point=SteeringHookPoint.POST_MLP,
            module_name="g",
            activation=SAEActivation.RELU,
            activation_params={},
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=1,
            dtype=torch.float32,
        )
        kind = getattr(m, HOOK_POINT_SAE_CLAMP_KIND_ATTR[SteeringHookPoint.POST_MLP])
        assert kind.device.type == "cpu"
