# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SAE-kind discriminator on SteeringModuleRegistry."""

from __future__ import annotations

import pytest

from vllm.config.sae_steering_types import SAEActivation, SteeringModuleKind
from vllm.entrypoints.openai.steering.registry import (
    SAEModuleManifest,
    SteeringModuleRegistry,
    sae_manifest_from_dict,
)


def _manifest(
    *,
    d_model: int = 4096,
    d_sae: int = 65536,
    activation: SAEActivation = SAEActivation.JUMPRELU,
    layers: tuple[tuple[int, str], ...] = ((20, "post_mlp"),),
    clampable_features: tuple[int, ...] = (0, 1, 2, 34),
    activation_params: dict[str, float] | None = None,
    weights_uri: str | None = "/tmp/sae",
) -> SAEModuleManifest:
    return SAEModuleManifest(
        d_model=d_model,
        d_sae=d_sae,
        activation=activation,
        layers=layers,
        clampable_features=clampable_features,
        activation_params=activation_params or {},
        weights_uri=weights_uri,
    )


class TestRegisterAdditive:
    """Existing additive registrations must still work bit-for-bit."""

    @pytest.mark.asyncio
    async def test_additive_default_kind(self):
        reg = SteeringModuleRegistry()
        await reg.register(name="m", vectors={"post_mlp": {0: [0.1, 0.2]}})
        mod = reg.get("m")
        assert mod is not None
        assert mod.kind is SteeringModuleKind.ADDITIVE
        assert mod.vectors == {"post_mlp": {0: [0.1, 0.2]}}
        assert mod.sae_manifest is None

    @pytest.mark.asyncio
    async def test_additive_rejects_sae_manifest(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="sae_manifest is only valid"):
            await reg.register(
                name="m",
                vectors={"post_mlp": {0: [0.1]}},
                sae_manifest=_manifest(),
            )


class TestRegisterSAE:
    @pytest.mark.asyncio
    async def test_sae_basic(self):
        reg = SteeringModuleRegistry()
        await reg.register(
            name="g",
            kind=SteeringModuleKind.SAE_DELTA,
            sae_manifest=_manifest(),
        )
        mod = reg.get("g")
        assert mod is not None
        assert mod.kind is SteeringModuleKind.SAE_DELTA
        assert mod.sae_manifest is not None
        assert mod.vectors is None

    @pytest.mark.asyncio
    async def test_sae_requires_manifest(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="sae_manifest is required"):
            await reg.register(name="g", kind=SteeringModuleKind.SAE_DELTA)

    @pytest.mark.asyncio
    async def test_sae_rejects_additive_vectors(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(
            ValueError, match="additive vector fields are not valid"
        ):
            await reg.register(
                name="g",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(),
                vectors={"post_mlp": {0: [0.1]}},
            )

    @pytest.mark.asyncio
    async def test_sae_rejects_clampable_out_of_range(self):
        reg = SteeringModuleRegistry()
        # d_sae=10 but clampable_features includes 100 — should fail.
        with pytest.raises(ValueError, match="out of range"):
            await reg.register(
                name="g",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(d_sae=10, clampable_features=(0, 100)),
            )

    @pytest.mark.asyncio
    async def test_sae_validates_layer_indices(self):
        reg = SteeringModuleRegistry(valid_layer_indices={0, 1, 2})
        with pytest.raises(ValueError, match="unknown layer index"):
            await reg.register(
                name="g",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(layers=((42, "post_mlp"),)),
            )

    @pytest.mark.asyncio
    async def test_sae_validates_hook_point(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="unknown hook point"):
            await reg.register(
                name="g",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(layers=((0, "bogus_hook"),)),
            )


class TestValidateAdditiveLookup:
    """The kind-aware ``steering_name`` validator that both OpenAI
    serving entrypoints share."""

    @pytest.mark.asyncio
    async def test_unknown_returns_error(self):
        reg = SteeringModuleRegistry()
        err = reg.validate_additive_lookup("missing")
        assert err is not None
        assert "Unknown steering module" in err
        assert "'missing'" in err

    @pytest.mark.asyncio
    async def test_additive_returns_none(self):
        reg = SteeringModuleRegistry()
        await reg.register(name="m", vectors={"post_mlp": {0: [0.1]}})
        assert reg.validate_additive_lookup("m") is None

    @pytest.mark.asyncio
    async def test_sae_returns_kind_error(self):
        """An SAE-kind module under the same name must reject as a
        client error here, not bubble up as a worker-side missing-name
        runtime crash."""
        reg = SteeringModuleRegistry()
        await reg.register(
            name="g",
            kind=SteeringModuleKind.SAE_DELTA,
            sae_manifest=_manifest(),
        )
        err = reg.validate_additive_lookup("g")
        assert err is not None
        assert "kind='sae_delta'" in err
        assert "sae_clamp_specs" in err


class TestMixedKindRegistration:
    """Phase-0 must reject SAE registrations that also carry additive
    fields — the API server's router forwards every field, so the
    registry rejection here protects against silent payload loss."""

    @pytest.mark.asyncio
    async def test_sae_with_vectors_rejected(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(
            ValueError, match="additive vector fields are not valid"
        ):
            await reg.register(
                name="x",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(),
                vectors={"post_mlp": {0: [0.1]}},
            )

    @pytest.mark.asyncio
    async def test_sae_with_prefill_vectors_rejected(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(
            ValueError, match="additive vector fields are not valid"
        ):
            await reg.register(
                name="x",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(),
                prefill_vectors={"pre_attn": {0: [0.1]}},
            )

    @pytest.mark.asyncio
    async def test_sae_with_decode_vectors_rejected(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(
            ValueError, match="additive vector fields are not valid"
        ):
            await reg.register(
                name="x",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(),
                decode_vectors={"post_mlp": {0: [0.1]}},
            )


class TestBroadcastRoundTrip:
    """The dump/restore path that crosses the multiprocessing boundary."""

    @pytest.mark.asyncio
    async def test_additive_payload_carries_kind(self):
        reg = SteeringModuleRegistry()
        await reg.register(name="m", vectors={"post_mlp": {0: [0.1]}})
        dump = reg.dump_for_broadcast()
        assert dump["m"]["kind"] == "additive"
        assert dump["m"]["vectors"] == {"post_mlp": {0: [0.1]}}

    @pytest.mark.asyncio
    async def test_sae_payload_carries_manifest(self):
        reg = SteeringModuleRegistry()
        manifest = _manifest()
        await reg.register(
            name="g",
            kind=SteeringModuleKind.SAE_DELTA,
            sae_manifest=manifest,
        )
        dump = reg.dump_for_broadcast()
        assert dump["g"]["kind"] == "sae_delta"
        assert "sae_manifest" in dump["g"]
        # Round-trip through the dict form yields a structurally
        # equivalent manifest, with the activation enum re-wrapped.
        restored = sae_manifest_from_dict(dump["g"]["sae_manifest"])
        assert restored.d_model == manifest.d_model
        assert restored.d_sae == manifest.d_sae
        assert restored.activation == manifest.activation
        assert restored.layers == manifest.layers
        assert restored.clampable_features == manifest.clampable_features
