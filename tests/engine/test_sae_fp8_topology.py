# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``storage_dtype`` manifest / topology plumbing tests.

Covers manifest parsing + validation of the opt-in fp8 storage field,
its distillation into ``SAEModuleTopology`` (both module-dir and
vllm-rs JSON forms), ``compute_hash`` sensitivity, and
``sae_topology_mismatch`` rejection of storage-dtype drift.
"""

import json

import pytest

from vllm.config.steering import (
    SAEModuleTopology,
    SteeringConfig,
    sae_topology_mismatch,
)
from vllm.engine.arg_utils import _build_sae_module_topology
from vllm.entrypoints.openai.steering.registry import (
    SteeringModuleRegistry,
    _sae_manifest_to_dict,
    sae_manifest_from_dict,
)


class _StubModelConfig:
    def get_hidden_size(self) -> int:
        return 64


def _manifest_payload(**overrides) -> dict:
    payload = {
        "d_model": 64,
        "d_sae": 32,
        "activation": "relu",
        "layers": [[3, "post_block"]],
        "clampable_features": [0, 5],
        "activation_params": {},
    }
    payload.update(overrides)
    return payload


class TestManifestParsing:
    def test_default_is_auto(self):
        manifest = sae_manifest_from_dict(_manifest_payload())
        assert manifest.storage_dtype == "auto"

    def test_fp8_value_round_trips(self):
        manifest = sae_manifest_from_dict(_manifest_payload(storage_dtype="fp8_e4m3"))
        assert manifest.storage_dtype == "fp8_e4m3"
        assert _sae_manifest_to_dict(manifest)["storage_dtype"] == "fp8_e4m3"

    def test_auto_value_round_trips(self):
        manifest = sae_manifest_from_dict(_manifest_payload(storage_dtype="auto"))
        assert _sae_manifest_to_dict(manifest)["storage_dtype"] == "auto"

    @pytest.mark.parametrize("bad", ["fp8", "int8", "bf16", 8, None])
    def test_unknown_value_fails_loudly(self, bad):
        with pytest.raises((ValueError, TypeError), match="storage_dtype"):
            sae_manifest_from_dict(_manifest_payload(storage_dtype=bad))

    def test_registry_validator_rejects_bad_storage_dtype(self):
        manifest = sae_manifest_from_dict(_manifest_payload())
        manifest.storage_dtype = "nope"
        with pytest.raises(ValueError, match="storage_dtype"):
            SteeringModuleRegistry()._validate_sae_manifest(name="m", manifest=manifest)


def _write_manifest_dir(tmp_path, **overrides):
    d = tmp_path / "mod"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps(_manifest_payload(**overrides)))
    return d


class TestTopologyDistillation:
    def test_dir_manifest_storage_dtype_distilled(self, tmp_path):
        d = _write_manifest_dir(tmp_path, storage_dtype="fp8_e4m3")
        (topo,) = _build_sae_module_topology([("m", str(d))], _StubModelConfig())
        assert topo.storage_dtype == "fp8_e4m3"

    def test_dir_manifest_defaults_auto(self, tmp_path):
        d = _write_manifest_dir(tmp_path)
        (topo,) = _build_sae_module_topology([("m", str(d))], _StubModelConfig())
        assert topo.storage_dtype == "auto"

    def test_dir_manifest_bad_storage_dtype_fails_fast(self, tmp_path):
        d = _write_manifest_dir(tmp_path, storage_dtype="fp4")
        with pytest.raises(ValueError, match="storage_dtype"):
            _build_sae_module_topology([("m", str(d))], _StubModelConfig())

    def test_rust_style_json_storage_dtype_distilled(self, tmp_path):
        f = tmp_path / "rust_delta.json"
        f.write_text(
            json.dumps(
                {
                    "kind": "sae_delta",
                    "sae_manifest": _manifest_payload(storage_dtype="fp8_e4m3"),
                    "sae_weights": {"3:post_block": {}},
                }
            )
        )
        (topo,) = _build_sae_module_topology([("m", str(f))], _StubModelConfig())
        assert topo.storage_dtype == "fp8_e4m3"


def _topo(**overrides) -> SAEModuleTopology:
    base = dict(
        name="m",
        kind="sae_delta",
        layers=((1, "post_block"),),
        d_model=64,
        d_sae=32,
        n_clamp=3,
        activation="relu",
        activation_params={},
    )
    base.update(overrides)
    return SAEModuleTopology(**base)


class TestHashAndMismatch:
    def test_storage_dtype_is_a_hash_factor(self):
        a = SteeringConfig(sae_module_topology=[_topo()]).compute_hash()
        b = SteeringConfig(
            sae_module_topology=[_topo(storage_dtype="fp8_e4m3")]
        ).compute_hash()
        assert a != b

    def test_mismatch_detects_storage_dtype_drift(self):
        t = _topo(storage_dtype="fp8_e4m3")
        msg = sae_topology_mismatch(
            t,
            kind="sae_delta",
            layers=((1, "post_block"),),
            d_model=64,
            d_sae=32,
            n_clamp=3,
            activation="relu",
            activation_params={},
            storage_dtype="auto",
        )
        assert msg is not None and "storage_dtype" in msg

    def test_mismatch_default_matches_auto_topology(self):
        t = _topo()
        assert (
            sae_topology_mismatch(
                t,
                kind="sae_delta",
                layers=((1, "post_block"),),
                d_model=64,
                d_sae=32,
                n_clamp=3,
                activation="relu",
                activation_params={},
            )
            is None
        )

    def test_matching_fp8_is_accepted(self):
        t = _topo(storage_dtype="fp8_e4m3")
        assert (
            sae_topology_mismatch(
                t,
                kind="sae_delta",
                layers=((1, "post_block"),),
                d_model=64,
                d_sae=32,
                n_clamp=3,
                activation="relu",
                activation_params={},
                storage_dtype="fp8_e4m3",
            )
            is None
        )
