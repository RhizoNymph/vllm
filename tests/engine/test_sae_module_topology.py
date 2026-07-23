# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup SAE-module topology distillation and config-hash tests.

The engine reads each ``--steering-modules`` directory's
``manifest.json`` (no tensor I/O) at config-build time and carries the
distilled :class:`SAEModuleTopology` on ``SteeringConfig`` so workers
can pre-allocate SAE buffers before compile/capture.  These tests pin
the distillation (``_build_sae_module_topology``), its fail-fast
contract, the ``from_cli_args`` namespace pickup, and the
``compute_hash`` sensitivity of the new topology / spare-slot fields.
"""

import argparse
import json
from pathlib import Path

import pytest

from vllm.config.steering import (
    SAEModuleTopology,
    SteeringConfig,
    sae_topology_mismatch,
)
from vllm.engine.arg_utils import EngineArgs, _build_sae_module_topology


class _StubModelConfig:
    def __init__(self, hidden_size: int = 64) -> None:
        self._hidden = hidden_size

    def get_hidden_size(self) -> int:
        return self._hidden


def _write_manifest(tmp_path: Path, name: str = "mod", **overrides) -> Path:
    manifest = {
        "d_model": 64,
        "d_sae": 32,
        "activation": "jumprelu",
        "layers": [[3, "post_block"], [1, "post_block"]],
        "clampable_features": [0, 5, 7],
        "activation_params": {},
    }
    manifest.update(overrides)
    d = tmp_path / name
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


class TestBuildTopology:
    def test_distills_delta_dir(self, tmp_path):
        d = _write_manifest(tmp_path)
        (topo,) = _build_sae_module_topology([("m", str(d))], _StubModelConfig())
        assert topo.name == "m"
        assert topo.kind == "sae_delta"
        # Layers canonically sorted; feature count only, not the ids.
        assert topo.layers == ((1, "post_block"), (3, "post_block"))
        assert topo.n_clamp == 3
        assert topo.d_sae == 32
        assert topo.activation == "jumprelu"

    def test_distills_fr_dir_via_kind_key(self, tmp_path):
        d = _write_manifest(tmp_path, kind="sae_full_reconstruction")
        (topo,) = _build_sae_module_topology([("m", str(d))], _StubModelConfig())
        assert topo.kind == "sae_full_reconstruction"

    def test_additive_json_file_is_skipped(self, tmp_path):
        f = tmp_path / "additive.json"
        f.write_text("{}")
        assert _build_sae_module_topology([("a", str(f))], _StubModelConfig()) == []

    def test_missing_path_fails_fast(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            _build_sae_module_topology(
                [("a", str(tmp_path / "nope"))], _StubModelConfig()
            )

    def test_missing_manifest_fails_fast(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(ValueError, match="manifest.json"):
            _build_sae_module_topology([("a", str(d))], _StubModelConfig())

    def test_corrupt_manifest_fails_fast(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "manifest.json").write_text("{not json")
        with pytest.raises(ValueError, match="cannot read"):
            _build_sae_module_topology([("a", str(d))], _StubModelConfig())

    def test_unknown_kind_fails_fast(self, tmp_path):
        d = _write_manifest(tmp_path, kind="wat")
        with pytest.raises(ValueError, match="unsupported kind"):
            _build_sae_module_topology([("a", str(d))], _StubModelConfig())

    def test_d_model_mismatch_fails_fast(self, tmp_path):
        d = _write_manifest(tmp_path)
        with pytest.raises(ValueError, match="hidden size"):
            _build_sae_module_topology(
                [("a", str(d))], _StubModelConfig(hidden_size=128)
            )

    def test_n_clamp_over_d_sae_fails_fast(self, tmp_path):
        d = _write_manifest(tmp_path, d_sae=2, clampable_features=[0, 1, 2])
        with pytest.raises(ValueError, match="exceed d_sae"):
            _build_sae_module_topology([("a", str(d))], _StubModelConfig())

    def test_duplicate_names_fail_fast(self, tmp_path):
        d1 = _write_manifest(tmp_path, name="one")
        d2 = _write_manifest(tmp_path, name="two")
        with pytest.raises(ValueError, match="duplicate"):
            _build_sae_module_topology(
                [("a", str(d1)), ("a", str(d2))], _StubModelConfig()
            )

    def test_accepts_dict_and_dataclass_entries(self, tmp_path):
        d = _write_manifest(tmp_path)
        from vllm.entrypoints.openai.models.protocol import SteeringModulePath

        for entry in (
            {"name": "m", "path": str(d)},
            SteeringModulePath(name="m", path=str(d)),
            ("m", str(d)),
        ):
            (topo,) = _build_sae_module_topology([entry], _StubModelConfig())
            assert topo.name == "m"


class TestFromCliArgs:
    def test_steering_modules_picked_off_shared_namespace(self):
        # The CLI flag is frontend-registered; EngineArgs mirrors the
        # parsed value from the shared serve-parser namespace.
        ns = argparse.Namespace(model="m", steering_modules=[("a", "/p")])
        assert EngineArgs.from_cli_args(ns).steering_modules == [("a", "/p")]

    def test_absent_namespace_attr_defaults_none(self):
        assert (
            EngineArgs.from_cli_args(argparse.Namespace(model="m")).steering_modules
            is None
        )


def _topo(**overrides) -> SAEModuleTopology:
    base = dict(
        name="m",
        kind="sae_delta",
        layers=((1, "post_block"),),
        d_model=64,
        d_sae=32,
        n_clamp=3,
        activation="jumprelu",
        activation_params={},
    )
    base.update(overrides)
    return SAEModuleTopology(**base)


class TestComputeHash:
    def test_topology_changes_hash(self):
        base = SteeringConfig().compute_hash()
        with_topo = SteeringConfig(sae_module_topology=[_topo()]).compute_hash()
        assert base != with_topo

    @pytest.mark.parametrize(
        "change",
        [
            dict(layers=((2, "post_block"),)),
            dict(d_sae=64),
            dict(n_clamp=4),
            dict(activation="relu"),
            dict(activation_params={"k": 2.0}),
            dict(kind="sae_full_reconstruction"),
        ],
    )
    def test_every_shape_field_is_a_hash_factor(self, change):
        a = SteeringConfig(sae_module_topology=[_topo()]).compute_hash()
        b = SteeringConfig(sae_module_topology=[_topo(**change)]).compute_hash()
        assert a != b

    def test_module_order_does_not_change_hash(self):
        t1, t2 = _topo(name="a"), _topo(name="b", layers=((5, "post_block"),))
        h1 = SteeringConfig(sae_module_topology=[t1, t2]).compute_hash()
        h2 = SteeringConfig(sae_module_topology=[t2, t1]).compute_hash()
        assert h1 == h2

    def test_spare_slot_knobs_are_hash_factors(self):
        base = SteeringConfig().compute_hash()
        assert (
            base != SteeringConfig(sae_spare_slot_sites=["1:post_block"]).compute_hash()
        )
        assert base != SteeringConfig(sae_spare_slot_features=8).compute_hash()
        assert SteeringConfig(sae_spare_slots_per_site=2).compute_hash() != base


class TestTopologyMismatch:
    def test_exact_match_returns_none(self):
        t = _topo()
        assert (
            sae_topology_mismatch(
                t,
                kind="sae_delta",
                layers=((1, "post_block"),),
                d_model=64,
                d_sae=32,
                n_clamp=3,
                activation="jumprelu",
                activation_params={},
            )
            is None
        )

    def test_site_order_is_canonicalized(self):
        t = _topo(layers=((1, "post_block"), (3, "post_block")))
        assert (
            sae_topology_mismatch(
                t,
                kind="sae_delta",
                layers=((3, "post_block"), (1, "post_block")),
                d_model=64,
                d_sae=32,
                n_clamp=3,
                activation="jumprelu",
                activation_params={},
            )
            is None
        )

    def test_mismatch_names_the_field(self):
        t = _topo()
        msg = sae_topology_mismatch(
            t,
            kind="sae_delta",
            layers=((1, "post_block"),),
            d_model=64,
            d_sae=99,
            n_clamp=3,
            activation="jumprelu",
            activation_params={},
        )
        assert msg is not None and "d_sae" in msg
