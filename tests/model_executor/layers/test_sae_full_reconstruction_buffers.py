# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for full-reconstruction SAE per-(layer, hook) buffer registration.

Mirrors :mod:`tests.model_executor.layers.test_sae_buffers` for the
delta path: register / unregister / re-register / shape validation.
The buffers themselves are zero-initialised and only the *shapes /
dtypes / per-hook attribute names* matter at this stage; the worker
mixin (Stage 3) is responsible for populating values.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm.config.sae_steering_types import SAEActivation
from vllm.model_executor.layers.sae_full_reconstruction import (
    HOOK_POINT_FR_ACTIVATION_ATTR,
    HOOK_POINT_FR_ACTIVATION_PARAMS_ATTR,
    HOOK_POINT_FR_CLAMP_KIND_ATTR,
    HOOK_POINT_FR_CLAMP_ONLY_IF_ACTIVE_ATTR,
    HOOK_POINT_FR_CLAMP_VALUE_ATTR,
    HOOK_POINT_FR_CLAMPABLE_FEATURES_ATTR,
    HOOK_POINT_FR_DECODER_BIAS_ATTR,
    HOOK_POINT_FR_DECODER_WEIGHT_ATTR,
    HOOK_POINT_FR_ENCODER_BIAS_ATTR,
    HOOK_POINT_FR_ENCODER_WEIGHT_ATTR,
    HOOK_POINT_FR_MODULE_NAME_ATTR,
    register_sae_full_recon_buffers,
    register_sae_recon_index_buffer,
    sae_full_recon_buffers_attached,
    share_sae_recon_index_across_layers,
    unregister_sae_full_recon_buffers,
)
from vllm.model_executor.layers.steering import SteeringHookPoint


def _make_layer(layer_idx: int = 0) -> nn.Module:
    layer = nn.Module()
    layer.layer_idx = layer_idx  # type: ignore[attr-defined]
    return layer


class TestRegisterFullReconBuffers:
    """Buffer attachment shape, dtype, and lookup."""

    def test_basic_registration_shapes(self):
        layer = _make_layer()
        feats = torch.tensor([0, 3, 7], dtype=torch.int64)
        register_sae_full_recon_buffers(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            activation=SAEActivation.RELU,
            activation_params={},
            d_sae=16,
            n_clamp=3,
            hidden_size=8,
            max_recon_configs=4,
            clampable_features=feats,
            dtype=torch.float32,
        )
        # Encoder / decoder buffers carry the *full* d_sae rows.
        enc_w = getattr(
            layer, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        enc_b = getattr(
            layer, HOOK_POINT_FR_ENCODER_BIAS_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        dec_w = getattr(
            layer, HOOK_POINT_FR_DECODER_WEIGHT_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        dec_b = getattr(
            layer, HOOK_POINT_FR_DECODER_BIAS_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert enc_w.shape == (16, 8)
        assert enc_b.shape == (16,)
        assert dec_w.shape == (16, 8)
        assert dec_b.shape == (8,)
        for buf in (enc_w, enc_b, dec_w, dec_b):
            assert buf.dtype is torch.float32
            assert torch.equal(buf, torch.zeros_like(buf))
        # Clamp tables: (max_recon_configs + 1, n_clamp), with row 0
        # reserved as the no-reconstruction sentinel.
        kind_table = getattr(
            layer, HOOK_POINT_FR_CLAMP_KIND_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        value_table = getattr(
            layer, HOOK_POINT_FR_CLAMP_VALUE_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        only_table = getattr(
            layer, HOOK_POINT_FR_CLAMP_ONLY_IF_ACTIVE_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert kind_table.shape == (5, 3)
        assert kind_table.dtype is torch.int8
        assert value_table.shape == (5, 3)
        assert value_table.dtype is torch.float32
        assert only_table.shape == (5, 3)
        assert only_table.dtype is torch.bool
        # Clampable feature index buffer is what the dispatch shim
        # passes to the op so the decoder pass sees the right rows.
        feat_buf = getattr(
            layer, HOOK_POINT_FR_CLAMPABLE_FEATURES_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert feat_buf.dtype is torch.int64
        assert torch.equal(feat_buf, feats)

    def test_python_attributes_are_set(self):
        layer = _make_layer()
        register_sae_full_recon_buffers(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="golden_gate",
            activation=SAEActivation.JUMPRELU,
            activation_params={"threshold": 0.42},
            d_sae=8,
            n_clamp=2,
            hidden_size=4,
            max_recon_configs=4,
            clampable_features=torch.tensor([0, 1], dtype=torch.int64),
            dtype=torch.float32,
        )
        assert (
            getattr(layer, HOOK_POINT_FR_MODULE_NAME_ATTR[SteeringHookPoint.POST_BLOCK])
            == "golden_gate"
        )
        assert (
            getattr(layer, HOOK_POINT_FR_ACTIVATION_ATTR[SteeringHookPoint.POST_BLOCK])
            is SAEActivation.JUMPRELU
        )
        params = getattr(
            layer, HOOK_POINT_FR_ACTIVATION_PARAMS_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert params == {"threshold": 0.42}
        # Stored as a fresh dict (not a reference to the caller-supplied
        # mapping) so caller mutations don't leak into the module.
        params["threshold"] = 9.0
        params2 = getattr(
            layer, HOOK_POINT_FR_ACTIVATION_PARAMS_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert params2["threshold"] == 9.0  # same dict
        # But the original caller-supplied dict is independent.
        independent = {"threshold": 0.5}
        register_sae_full_recon_buffers(
            _make_layer(layer_idx=1),
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="x",
            activation=SAEActivation.JUMPRELU,
            activation_params=independent,
            d_sae=4,
            n_clamp=1,
            hidden_size=4,
            max_recon_configs=4,
            clampable_features=torch.tensor([0], dtype=torch.int64),
            dtype=torch.float32,
        )
        assert independent == {"threshold": 0.5}

    def test_disabled_engine_is_no_op(self):
        # max_recon_configs == 0 → registration is a no-op so disabled
        # engines pay zero attribute cost on the layer.
        layer = _make_layer()
        register_sae_full_recon_buffers(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            activation=SAEActivation.RELU,
            activation_params={},
            d_sae=16,
            n_clamp=2,
            hidden_size=8,
            max_recon_configs=0,
            clampable_features=torch.tensor([0, 1], dtype=torch.int64),
            dtype=torch.float32,
        )
        assert not sae_full_recon_buffers_attached(layer, SteeringHookPoint.POST_BLOCK)

    def test_double_registration_at_same_site_raises(self):
        layer = _make_layer()
        feats = torch.tensor([0, 1], dtype=torch.int64)
        register_sae_full_recon_buffers(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="a",
            activation=SAEActivation.RELU,
            activation_params={},
            d_sae=8,
            n_clamp=2,
            hidden_size=4,
            max_recon_configs=4,
            clampable_features=feats,
            dtype=torch.float32,
        )
        with pytest.raises(ValueError, match="already has full-reconstruction"):
            register_sae_full_recon_buffers(
                layer,
                hook_point=SteeringHookPoint.POST_BLOCK,
                module_name="b",
                activation=SAEActivation.RELU,
                activation_params={},
                d_sae=8,
                n_clamp=2,
                hidden_size=4,
                max_recon_configs=4,
                clampable_features=feats,
                dtype=torch.float32,
            )

    def test_distinct_hook_points_coexist(self):
        layer = _make_layer()
        for hp, name in (
            (SteeringHookPoint.POST_BLOCK, "m1"),
            (SteeringHookPoint.POST_ATTN, "m2"),
        ):
            register_sae_full_recon_buffers(
                layer,
                hook_point=hp,
                module_name=name,
                activation=SAEActivation.RELU,
                activation_params={},
                d_sae=8,
                n_clamp=2,
                hidden_size=4,
                max_recon_configs=4,
                clampable_features=torch.tensor([0, 1], dtype=torch.int64),
                dtype=torch.float32,
            )
        # Both buffers exist independently.
        for hp in (SteeringHookPoint.POST_BLOCK, SteeringHookPoint.POST_ATTN):
            assert sae_full_recon_buffers_attached(layer, hp)

    @pytest.mark.parametrize(
        "bad_kwargs,match",
        [
            ({"d_sae": 0}, "d_sae"),
            ({"n_clamp": -1}, "n_clamp"),
        ],
    )
    def test_invalid_dimensions_rejected(self, bad_kwargs, match):
        layer = _make_layer()
        kwargs = {
            "hook_point": SteeringHookPoint.POST_BLOCK,
            "module_name": "m",
            "activation": SAEActivation.RELU,
            "activation_params": {},
            "d_sae": 8,
            "n_clamp": 2,
            "hidden_size": 4,
            "max_recon_configs": 4,
            "clampable_features": torch.tensor([0, 1], dtype=torch.int64),
            "dtype": torch.float32,
        }
        kwargs.update(bad_kwargs)
        # Adjust clampable_features when n_clamp changes so we get the
        # expected error rather than an n_clamp-vs-features mismatch.
        if "n_clamp" in bad_kwargs and bad_kwargs["n_clamp"] >= 0:
            kwargs["clampable_features"] = torch.zeros(
                bad_kwargs["n_clamp"], dtype=torch.int64
            )
        with pytest.raises(ValueError, match=match):
            register_sae_full_recon_buffers(layer, **kwargs)  # type: ignore[arg-type]

    def test_clampable_features_wrong_dtype_rejected(self):
        layer = _make_layer()
        with pytest.raises(ValueError, match="int64"):
            register_sae_full_recon_buffers(
                layer,
                hook_point=SteeringHookPoint.POST_BLOCK,
                module_name="m",
                activation=SAEActivation.RELU,
                activation_params={},
                d_sae=8,
                n_clamp=2,
                hidden_size=4,
                max_recon_configs=4,
                clampable_features=torch.tensor([0, 1], dtype=torch.int32),
                dtype=torch.float32,
            )

    def test_clampable_features_wrong_shape_rejected(self):
        layer = _make_layer()
        with pytest.raises(ValueError, match="shape"):
            register_sae_full_recon_buffers(
                layer,
                hook_point=SteeringHookPoint.POST_BLOCK,
                module_name="m",
                activation=SAEActivation.RELU,
                activation_params={},
                d_sae=8,
                n_clamp=3,
                hidden_size=4,
                max_recon_configs=4,
                clampable_features=torch.tensor([0, 1], dtype=torch.int64),
                dtype=torch.float32,
            )


class TestUnregisterFullReconBuffers:
    """Unregistration must remove every buffer + python attribute."""

    def test_unregister_clears_all_buffers(self):
        layer = _make_layer()
        register_sae_full_recon_buffers(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            activation=SAEActivation.RELU,
            activation_params={},
            d_sae=8,
            n_clamp=2,
            hidden_size=4,
            max_recon_configs=4,
            clampable_features=torch.tensor([0, 1], dtype=torch.int64),
            dtype=torch.float32,
        )
        assert sae_full_recon_buffers_attached(layer, SteeringHookPoint.POST_BLOCK)
        unregister_sae_full_recon_buffers(layer, hook_point=SteeringHookPoint.POST_BLOCK)
        assert not sae_full_recon_buffers_attached(layer, SteeringHookPoint.POST_BLOCK)
        # Every attribute must be gone — buffers and python attrs alike.
        for attr_table in (
            HOOK_POINT_FR_ENCODER_WEIGHT_ATTR,
            HOOK_POINT_FR_ENCODER_BIAS_ATTR,
            HOOK_POINT_FR_DECODER_WEIGHT_ATTR,
            HOOK_POINT_FR_DECODER_BIAS_ATTR,
            HOOK_POINT_FR_CLAMP_KIND_ATTR,
            HOOK_POINT_FR_CLAMP_VALUE_ATTR,
            HOOK_POINT_FR_CLAMP_ONLY_IF_ACTIVE_ATTR,
            HOOK_POINT_FR_CLAMPABLE_FEATURES_ATTR,
            HOOK_POINT_FR_MODULE_NAME_ATTR,
            HOOK_POINT_FR_ACTIVATION_ATTR,
            HOOK_POINT_FR_ACTIVATION_PARAMS_ATTR,
        ):
            assert not hasattr(layer, attr_table[SteeringHookPoint.POST_BLOCK])

    def test_unregister_when_absent_is_no_op(self):
        # Idempotent: calling on a layer with no buffers is fine.
        layer = _make_layer()
        unregister_sae_full_recon_buffers(layer, hook_point=SteeringHookPoint.POST_BLOCK)

    def test_unregister_then_reregister(self):
        # Round trip: re-registration after detach must succeed.
        layer = _make_layer()
        feats = torch.tensor([0, 1], dtype=torch.int64)
        register_sae_full_recon_buffers(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="m1",
            activation=SAEActivation.RELU,
            activation_params={},
            d_sae=8,
            n_clamp=2,
            hidden_size=4,
            max_recon_configs=4,
            clampable_features=feats,
            dtype=torch.float32,
        )
        unregister_sae_full_recon_buffers(layer, hook_point=SteeringHookPoint.POST_BLOCK)
        register_sae_full_recon_buffers(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="m2",
            activation=SAEActivation.RELU,
            activation_params={},
            d_sae=8,
            n_clamp=2,
            hidden_size=4,
            max_recon_configs=4,
            clampable_features=feats,
            dtype=torch.float32,
        )
        assert (
            getattr(layer, HOOK_POINT_FR_MODULE_NAME_ATTR[SteeringHookPoint.POST_BLOCK])
            == "m2"
        )


class TestReconIndexBuffer:
    """Per-token ``sae_recon_index`` buffer + sharing helper."""

    def test_register_creates_int64_buffer(self):
        layer = _make_layer()
        register_sae_recon_index_buffer(layer, max_tokens=128)
        assert hasattr(layer, "sae_recon_index")
        idx = layer.sae_recon_index
        assert idx.shape == (128,)
        assert idx.dtype is torch.long
        assert torch.equal(idx, torch.zeros(128, dtype=torch.long))

    def test_register_zero_max_tokens_is_no_op(self):
        layer = _make_layer()
        register_sae_recon_index_buffer(layer, max_tokens=0)
        assert not hasattr(layer, "sae_recon_index")

    def test_share_across_layers(self):
        layers = [_make_layer(layer_idx=i) for i in range(3)]
        for layer in layers:
            register_sae_recon_index_buffer(layer, max_tokens=16)
        # Independent buffers before sharing.
        assert layers[0].sae_recon_index is not layers[1].sae_recon_index
        share_sae_recon_index_across_layers(layers)
        assert layers[0].sae_recon_index is layers[1].sae_recon_index
        assert layers[1].sae_recon_index is layers[2].sae_recon_index

    def test_share_across_layers_skips_layers_without_buffer(self):
        # Mix — some layers don't carry the buffer (e.g. PP-non-owned).
        layers = [_make_layer(layer_idx=i) for i in range(3)]
        register_sae_recon_index_buffer(layers[0], max_tokens=16)
        register_sae_recon_index_buffer(layers[2], max_tokens=16)
        share_sae_recon_index_across_layers(layers)
        # The shared buffer is whatever the first carrier was.
        assert layers[0].sae_recon_index is layers[2].sae_recon_index
        assert not hasattr(layers[1], "sae_recon_index")
