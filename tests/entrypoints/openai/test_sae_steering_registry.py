# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SAE-kind discriminator on SteeringModuleRegistry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vllm.config.sae_steering_types import SAEActivation, SteeringModuleKind
from vllm.entrypoints.openai.steering.registry import (
    SAEModuleManifest,
    SteeringModuleRegistry,
    sae_manifest_from_dict,
)
from vllm.entrypoints.openai.steering.sae_loader import _site_filename


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
    if activation_params is None and activation is SAEActivation.JUMPRELU:
        activation_params = {"threshold": 0.0}
    elif activation_params is None and activation is SAEActivation.TOPK:
        activation_params = {"k": 1.0}
    return SAEModuleManifest(
        d_model=d_model,
        d_sae=d_sae,
        activation=activation,
        layers=layers,
        clampable_features=clampable_features,
        activation_params=activation_params or {},
        weights_uri=weights_uri,
    )


def _raw_sae_clamp_spec(module_name: str = "g") -> dict:
    return {
        "module_name": module_name,
        "clamps": {
            "post_mlp": {
                "20": [
                    {
                        "feature_idx": 34,
                        "kind": "absolute",
                        "value": 5.0,
                    }
                ]
            }
        },
    }


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

    @pytest.mark.asyncio
    async def test_additive_rejects_bool_layer_index(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="non-negative integer"):
            await reg.register(name="m", vectors={"post_mlp": {True: [0.1]}})

    @pytest.mark.asyncio
    async def test_additive_rejects_oversized_layer_index(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="2147483647"):
            await reg.register(name="m", vectors={"post_mlp": {2**31: [0.1]}})


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
        with pytest.raises(ValueError, match="additive vector fields are not valid"):
            await reg.register(
                name="g",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(),
                vectors={"post_mlp": {0: [0.1]}},
            )

    @pytest.mark.asyncio
    async def test_sae_rejects_empty_additive_vectors(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="additive vector fields are not valid"):
            await reg.register(
                name="g",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(),
                vectors={},
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
    async def test_sae_rejects_non_string_weights_uri(self):
        reg = SteeringModuleRegistry()
        manifest = _manifest()
        manifest.weights_uri = True  # type: ignore[assignment]
        with pytest.raises(ValueError, match="weights_uri"):
            await reg.register(
                name="g",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=manifest,
            )

    @pytest.mark.asyncio
    async def test_sae_rejects_bool_dimensions_layers_and_features(self):
        reg = SteeringModuleRegistry()
        invalid_manifests = [
            _manifest(d_model=True),  # type: ignore[arg-type]
            _manifest(d_sae=True),  # type: ignore[arg-type]
            _manifest(layers=((True, "post_mlp"),)),  # type: ignore[arg-type]
            _manifest(clampable_features=(True,)),  # type: ignore[arg-type]
        ]
        for manifest in invalid_manifests:
            with pytest.raises(ValueError):
                await reg.register(
                    name="g",
                    kind=SteeringModuleKind.SAE_DELTA,
                    sae_manifest=manifest,
                )

    @pytest.mark.asyncio
    async def test_sae_rejects_duplicate_clampable_features(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="clampable_features.*duplicates"):
            await reg.register(
                name="g",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(clampable_features=(0, 1, 0)),
            )

    @pytest.mark.asyncio
    async def test_sae_rejects_empty_clampable_features(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="clampable_features must not be empty"):
            await reg.register(
                name="g",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(clampable_features=()),
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "manifest",
        [
            _manifest(activation=SAEActivation.RELU, activation_params={"k": 1.0}),
            _manifest(activation_params={}),
            _manifest(activation_params={"threshold": True}),  # type: ignore[dict-item]
            _manifest(activation=SAEActivation.TOPK, activation_params={}),
            _manifest(activation=SAEActivation.TOPK, activation_params={"k": 0.0}),
            _manifest(activation=SAEActivation.TOPK, activation_params={"k": 1.5}),
            _manifest(activation=SAEActivation.TOPK, activation_params={"k": True}),  # type: ignore[dict-item]
        ],
    )
    async def test_sae_rejects_invalid_activation_params(
        self,
        manifest: SAEModuleManifest,
    ):
        reg = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="activation_params"):
            await reg.register(
                name="g",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=manifest,
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
    async def test_sae_rejects_negative_layer_indices(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="non-negative"):
            await reg.register(
                name="g",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(layers=((-1, "post_mlp"),)),
            )

    @pytest.mark.asyncio
    async def test_sae_rejects_oversized_indices(self):
        reg = SteeringModuleRegistry()
        invalid_manifests = [
            _manifest(layers=((2**31, "post_mlp"),)),
            _manifest(d_sae=2**31 + 1, clampable_features=(2**31,)),
        ]
        for manifest in invalid_manifests:
            with pytest.raises(ValueError, match="2147483647"):
                await reg.register(
                    name="g",
                    kind=SteeringModuleKind.SAE_DELTA,
                    sae_manifest=manifest,
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

    @pytest.mark.asyncio
    async def test_sae_rejects_duplicate_layer_hook_sites(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="duplicate.*sites"):
            await reg.register(
                name="g",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(
                    layers=((0, "post_mlp"), (0, "post_mlp")),
                ),
            )

    @pytest.mark.asyncio
    async def test_sae_rejects_site_overlap_with_existing_module(self):
        reg = SteeringModuleRegistry()
        await reg.register(
            name="g",
            kind=SteeringModuleKind.SAE_DELTA,
            sae_manifest=_manifest(layers=((0, "post_mlp"),)),
        )

        with pytest.raises(ValueError, match="overlap existing SAE module"):
            await reg.register(
                name="h",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(layers=((0, "post_mlp"),)),
            )

    @pytest.mark.asyncio
    async def test_sae_replacement_may_reuse_its_existing_sites(self):
        reg = SteeringModuleRegistry()
        await reg.register(
            name="g",
            kind=SteeringModuleKind.SAE_DELTA,
            sae_manifest=_manifest(layers=((0, "post_mlp"),)),
        )

        await reg.register(
            name="g",
            kind=SteeringModuleKind.SAE_DELTA,
            sae_manifest=_manifest(layers=((0, "post_mlp"),)),
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


class TestResolveForRequest:
    @pytest.mark.asyncio
    async def test_sae_module_rejected_as_additive_resolution(self):
        reg = SteeringModuleRegistry()
        await reg.register(
            name="g",
            kind=SteeringModuleKind.SAE_DELTA,
            sae_manifest=_manifest(),
        )

        vectors, prefill, decode, err = reg.resolve_for_request(
            "g", None, None, None
        )

        assert vectors is None
        assert prefill is None
        assert decode is None
        assert err is not None
        assert "kind='sae_delta'" in err
        assert "sae_clamp_specs" in err


class TestValidateSAEClampSpecs:
    """API-side validation of sae_clamp_specs against the registry."""

    @pytest.mark.asyncio
    async def test_none_returns_none(self):
        reg = SteeringModuleRegistry()
        assert reg.validate_sae_clamp_specs(None) is None
        assert reg.validate_sae_clamp_specs([]) is None
        assert reg.validate_sae_clamp_specs(()) is None

    @pytest.mark.asyncio
    async def test_unknown_module_dict_form(self):
        reg = SteeringModuleRegistry()
        err = reg.validate_sae_clamp_specs([_raw_sae_clamp_spec("missing")])
        assert err is not None
        assert "unknown module 'missing'" in err

    @pytest.mark.asyncio
    async def test_additive_module_rejected(self):
        """An additive-kind module under the same name must surface as
        an API-side 400 instead of crashing the worker."""
        reg = SteeringModuleRegistry()
        await reg.register(name="m", vectors={"post_mlp": {0: [0.1]}})
        err = reg.validate_sae_clamp_specs([_raw_sae_clamp_spec("m")])
        assert err is not None
        assert "kind 'additive'" in err
        assert "steering_name" in err

    @pytest.mark.asyncio
    async def test_sae_module_passes(self):
        reg = SteeringModuleRegistry()
        await reg.register(
            name="g",
            kind=SteeringModuleKind.SAE_DELTA,
            sae_manifest=_manifest(),
        )
        assert reg.validate_sae_clamp_specs([_raw_sae_clamp_spec("g")]) is None

    @pytest.mark.asyncio
    async def test_undeclared_site_rejected(self):
        reg = SteeringModuleRegistry()
        await reg.register(
            name="g",
            kind=SteeringModuleKind.SAE_DELTA,
            sae_manifest=_manifest(layers=((20, "post_mlp"),)),
        )
        raw = _raw_sae_clamp_spec("g")
        raw["clamps"] = {
            "post_mlp": {
                "21": [
                    {
                        "feature_idx": 34,
                        "kind": "absolute",
                        "value": 5.0,
                    }
                ]
            }
        }

        err = reg.validate_sae_clamp_specs([raw])

        assert err is not None
        assert "not declared" in err
        assert "sae_manifest.layers" in err

    @pytest.mark.asyncio
    async def test_unclampable_feature_rejected(self):
        reg = SteeringModuleRegistry()
        await reg.register(
            name="g",
            kind=SteeringModuleKind.SAE_DELTA,
            sae_manifest=_manifest(clampable_features=(0, 1)),
        )

        err = reg.validate_sae_clamp_specs([_raw_sae_clamp_spec("g")])

        assert err is not None
        assert "feature_idx=34" in err
        assert "clampable_features" in err

    @pytest.mark.asyncio
    async def test_malformed_clamp_shape_rejected(self):
        reg = SteeringModuleRegistry()
        await reg.register(
            name="g",
            kind=SteeringModuleKind.SAE_DELTA,
            sae_manifest=_manifest(),
        )
        err = reg.validate_sae_clamp_specs(
            [
                {
                    "module_name": "g",
                    "clamps": {
                        "post_mlp": {
                            "20": [
                                {
                                    "feature_idx": 34,
                                    "kind": "absolute",
                                    "value": "5.0",
                                }
                            ]
                        }
                    },
                }
            ]
        )
        assert err is not None
        assert "value must be a finite float" in err

    @pytest.mark.asyncio
    async def test_missing_module_name_field(self):
        reg = SteeringModuleRegistry()
        err = reg.validate_sae_clamp_specs([{"clamps": {}}])
        assert err is not None
        assert "module_name" in err


class TestMixedKindRegistration:
    """Phase-0 must reject SAE registrations that also carry additive
    fields — the API server's router forwards every field, so the
    registry rejection here protects against silent payload loss."""

    @pytest.mark.asyncio
    async def test_sae_with_vectors_rejected(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="additive vector fields are not valid"):
            await reg.register(
                name="x",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(),
                vectors={"post_mlp": {0: [0.1]}},
            )

    @pytest.mark.asyncio
    async def test_sae_with_prefill_vectors_rejected(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="additive vector fields are not valid"):
            await reg.register(
                name="x",
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=_manifest(),
                prefill_vectors={"pre_attn": {0: [0.1]}},
            )

    @pytest.mark.asyncio
    async def test_sae_with_decode_vectors_rejected(self):
        reg = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="additive vector fields are not valid"):
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

    @pytest.mark.asyncio
    async def test_sae_payload_can_include_weights(self, monkeypatch):
        reg = SteeringModuleRegistry()
        manifest = _manifest(weights_uri="/tmp/weights")
        await reg.register(
            name="g",
            kind=SteeringModuleKind.SAE_DELTA,
            sae_manifest=manifest,
        )
        weights = {
            (20, "post_mlp"): {
                "encoder_weight": torch.zeros(4, 4096),
                "encoder_bias": torch.zeros(4),
                "decoder_weight": torch.zeros(4, 4096),
            }
        }

        def fake_load(loaded_manifest, path):
            assert loaded_manifest is manifest
            assert str(path) == "/tmp/weights"
            return weights

        monkeypatch.setattr(
            "vllm.entrypoints.openai.steering.sae_loader._load_weights_for_manifest",
            fake_load,
        )

        dump = reg.dump_for_broadcast(include_sae_weights=True)

        assert dump["g"]["kind"] == "sae_delta"
        assert dump["g"]["sae_manifest"]["weights_uri"] == "/tmp/weights"
        assert dump["g"]["sae_weights"] is weights

    @pytest.mark.asyncio
    async def test_load_from_file_accepts_sae_directory(self, tmp_path: Path):
        manifest_payload = {
            "d_model": 8,
            "d_sae": 16,
            "activation": "relu",
            "layers": [[0, "post_mlp"]],
            "clampable_features": [0, 1],
            "activation_params": {},
            "weights_uri": None,
        }
        (tmp_path / "manifest.json").write_text(
            json.dumps(manifest_payload), encoding="utf-8"
        )
        from safetensors.torch import save_file

        weights = {
            "encoder_weight": torch.ones(2, 8),
            "encoder_bias": torch.zeros(2),
            "decoder_weight": torch.full((2, 8), 2.0),
        }
        save_file(weights, str(tmp_path / _site_filename(0, "post_mlp")))

        reg = SteeringModuleRegistry()
        await reg.load_from_file("g", str(tmp_path))

        module = reg.get("g")
        assert module is not None
        assert module.kind is SteeringModuleKind.SAE_DELTA
        assert module.sae_manifest is not None
        assert module.sae_manifest.weights_uri == str(tmp_path)

        dump = reg.dump_for_broadcast(include_sae_weights=True)
        assert dump["g"]["kind"] == "sae_delta"
        assert torch.allclose(
            dump["g"]["sae_weights"][(0, "post_mlp")]["decoder_weight"],
            weights["decoder_weight"],
        )
