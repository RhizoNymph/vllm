# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the on-disk SAE checkpoint loader.

The loader has two readers — one for the generic ``manifest.json`` +
safetensors layout and one for the Gemma Scope ``params.npz``.  The
two share an output type (:class:`LoadedSAEModule`) and a feature-
subsetting contract; the tests synthesise both layouts on disk so
the round-trip can be verified without any network access.

A real-weights end-to-end test exercising the loader against a
downloaded Gemma Scope checkpoint lives in
``tests/models/language/generation/test_sae_steering_real_weights.py``
and is gated behind CUDA availability.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.config.sae_steering_types import SAEActivation
from vllm.entrypoints.openai.steering.sae_loader import (
    LoadedSAEModule,
    _site_filename,
    load_gemma_scope_sae,
    load_gemma_scope_sae_full_recon,
    load_sae_module_from_dir,
    merge_loaded_sae_modules,
)

# ---------------------------------------------------------------------------
# Fixtures: synthesise the two on-disk layouts the loader supports.
# ---------------------------------------------------------------------------


def _write_manifest(
    base: Path,
    *,
    d_model: int,
    d_sae: int,
    activation: str,
    layers: list[tuple[int, str]],
    clampable_features: list[int],
    activation_params: dict[str, float] | None = None,
) -> None:
    payload = {
        "d_model": d_model,
        "d_sae": d_sae,
        "activation": activation,
        "layers": [list(p) for p in layers],
        "clampable_features": list(clampable_features),
        "activation_params": activation_params or {},
        "weights_uri": None,
    }
    with (base / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _write_site(
    base: Path,
    *,
    layer_idx: int,
    hook_str: str,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    threshold: torch.Tensor | None = None,
) -> None:
    from safetensors.torch import save_file

    tensors = {
        "encoder_weight": encoder_weight.contiguous(),
        "encoder_bias": encoder_bias.contiguous(),
        "decoder_weight": decoder_weight.contiguous(),
    }
    if threshold is not None:
        tensors["threshold"] = threshold.contiguous()
    save_file(
        tensors,
        str(base / _site_filename(layer_idx, hook_str)),
    )


def _make_synthetic_dir(
    base: Path,
    *,
    d_model: int = 8,
    d_sae: int = 32,
    n_clamp: int = 4,
    layers: list[tuple[int, str]] | None = None,
    activation: str = "relu",
) -> tuple[list[int], dict[tuple[int, str], dict[str, torch.Tensor]]]:
    layers = layers or [(0, "post_block")]
    rng = torch.Generator(device="cpu").manual_seed(0)
    feats = list(range(n_clamp))
    weights: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for layer_idx, hook_str in layers:
        enc_w = torch.randn(n_clamp, d_model, generator=rng)
        enc_b = torch.randn(n_clamp, generator=rng)
        dec_w = torch.randn(n_clamp, d_model, generator=rng)
        _write_site(
            base,
            layer_idx=layer_idx,
            hook_str=hook_str,
            encoder_weight=enc_w,
            encoder_bias=enc_b,
            decoder_weight=dec_w,
        )
        weights[(layer_idx, hook_str)] = {
            "encoder_weight": enc_w,
            "encoder_bias": enc_b,
            "decoder_weight": dec_w,
        }
    _write_manifest(
        base,
        d_model=d_model,
        d_sae=d_sae,
        activation=activation,
        layers=layers,
        clampable_features=feats,
    )
    return feats, weights


def _make_gemma_scope_npz(
    path: Path,
    *,
    d_model: int = 8,
    d_sae: int = 16,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    arrs = {
        "W_enc": rng.standard_normal((d_model, d_sae)).astype(np.float32),
        "W_dec": rng.standard_normal((d_sae, d_model)).astype(np.float32),
        "b_enc": rng.standard_normal((d_sae,)).astype(np.float32),
        "b_dec": rng.standard_normal((d_model,)).astype(np.float32),
        "threshold": rng.uniform(0.0, 1.0, (d_sae,)).astype(np.float32),
    }
    np.savez(str(path), **arrs)
    return arrs


# ---------------------------------------------------------------------------
# Generic ``manifest.json`` + safetensors layout
# ---------------------------------------------------------------------------


class TestLoadSAEModuleFromDir:
    """Round-trip a synthetic on-disk module through the loader."""

    def test_round_trip_single_site(self, tmp_path: Path):
        feats, weights = _make_synthetic_dir(tmp_path)

        loaded = load_sae_module_from_dir(tmp_path)

        assert isinstance(loaded, LoadedSAEModule)
        assert loaded.manifest.d_model == 8
        assert loaded.manifest.d_sae == 32
        assert loaded.manifest.activation is SAEActivation.RELU
        assert loaded.manifest.clampable_features == tuple(feats)
        assert loaded.manifest.layers == ((0, "post_block"),)

        for site, expected in weights.items():
            for key in ("encoder_weight", "encoder_bias", "decoder_weight"):
                assert torch.allclose(loaded.weights[site][key], expected[key])

    def test_round_trip_multi_site(self, tmp_path: Path):
        layers = [(0, "post_block"), (1, "post_attn"), (2, "pre_attn")]
        _, weights = _make_synthetic_dir(tmp_path, layers=layers, n_clamp=2, d_sae=8)

        loaded = load_sae_module_from_dir(tmp_path)

        assert loaded.manifest.layers == tuple((li, hp) for li, hp in layers)
        for site, expected in weights.items():
            assert torch.allclose(
                loaded.weights[site]["encoder_weight"], expected["encoder_weight"]
            )
            assert torch.allclose(
                loaded.weights[site]["decoder_weight"], expected["decoder_weight"]
            )

    def test_missing_directory_raises_filenotfound(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_sae_module_from_dir(tmp_path / "does-not-exist")

    def test_missing_manifest_raises_filenotfound(self, tmp_path: Path):
        # Directory exists but has no manifest.json.
        (tmp_path / "stray.bin").touch()
        with pytest.raises(FileNotFoundError, match="manifest"):
            load_sae_module_from_dir(tmp_path)

    def test_missing_site_file_raises_filenotfound(self, tmp_path: Path):
        # Write a manifest declaring two sites but only one file.
        _write_manifest(
            tmp_path,
            d_model=4,
            d_sae=8,
            activation="relu",
            layers=[(0, "post_block"), (1, "post_attn")],
            clampable_features=[0, 1],
        )
        # Only first site's file is written.
        _write_site(
            tmp_path,
            layer_idx=0,
            hook_str="post_block",
            encoder_weight=torch.zeros(2, 4),
            encoder_bias=torch.zeros(2),
            decoder_weight=torch.zeros(2, 4),
        )
        with pytest.raises(FileNotFoundError, match="weight file"):
            load_sae_module_from_dir(tmp_path)

    def test_invalid_hook_in_manifest_raises_value_error(self, tmp_path: Path):
        _write_manifest(
            tmp_path,
            d_model=4,
            d_sae=8,
            activation="relu",
            layers=[(0, "bogus_hook")],
            clampable_features=[0],
        )
        # The manifest validator (sae_manifest_from_dict) rejects unknown
        # hook strings via SAEActivation/hook validation downstream;
        # surface it as a clear error from our loader.
        with pytest.raises(ValueError):
            load_sae_module_from_dir(tmp_path)

    def test_bool_integer_fields_in_manifest_raise_value_error(self, tmp_path: Path):
        _write_manifest(
            tmp_path,
            d_model=True,  # type: ignore[arg-type]
            d_sae=8,
            activation="relu",
            layers=[(0, "post_block")],
            clampable_features=[0],
        )
        with pytest.raises(ValueError, match="d_model"):
            load_sae_module_from_dir(tmp_path)

    def test_invalid_activation_params_in_manifest_raise_value_error(
        self,
        tmp_path: Path,
    ):
        _write_manifest(
            tmp_path,
            d_model=4,
            d_sae=8,
            activation="topk",
            layers=[(0, "post_block")],
            clampable_features=[0],
            activation_params={"k": 0.0},
        )
        with pytest.raises(ValueError, match="activation_params"):
            load_sae_module_from_dir(tmp_path)

    def test_shape_mismatch_raises_value_error(self, tmp_path: Path):
        # Manifest claims n_clamp=2, but the safetensors file has 3 rows.
        _write_manifest(
            tmp_path,
            d_model=4,
            d_sae=8,
            activation="relu",
            layers=[(0, "post_block")],
            clampable_features=[0, 1],
        )
        _write_site(
            tmp_path,
            layer_idx=0,
            hook_str="post_block",
            encoder_weight=torch.zeros(3, 4),  # wrong: should be (2, 4)
            encoder_bias=torch.zeros(3),
            decoder_weight=torch.zeros(3, 4),
        )
        with pytest.raises(ValueError, match="encoder_weight"):
            load_sae_module_from_dir(tmp_path)

    def test_missing_tensor_in_site_raises_value_error(self, tmp_path: Path):
        _write_manifest(
            tmp_path,
            d_model=4,
            d_sae=8,
            activation="relu",
            layers=[(0, "post_block")],
            clampable_features=[0],
        )
        from safetensors.torch import save_file

        save_file(
            {
                "encoder_weight": torch.zeros(1, 4),
                # encoder_bias deliberately omitted
                "decoder_weight": torch.zeros(1, 4),
            },
            str(tmp_path / _site_filename(0, "post_block")),
        )
        with pytest.raises(ValueError, match="encoder_bias"):
            load_sae_module_from_dir(tmp_path)

    def test_non_floating_site_tensor_raises_value_error(self, tmp_path: Path):
        _write_manifest(
            tmp_path,
            d_model=4,
            d_sae=8,
            activation="relu",
            layers=[(0, "post_block")],
            clampable_features=[0],
        )
        _write_site(
            tmp_path,
            layer_idx=0,
            hook_str="post_block",
            encoder_weight=torch.zeros(1, 4, dtype=torch.int64),
            encoder_bias=torch.zeros(1),
            decoder_weight=torch.zeros(1, 4),
        )
        with pytest.raises(ValueError, match="floating dtype"):
            load_sae_module_from_dir(tmp_path)

    def test_non_finite_site_tensor_raises_value_error(self, tmp_path: Path):
        _write_manifest(
            tmp_path,
            d_model=4,
            d_sae=8,
            activation="relu",
            layers=[(0, "post_block")],
            clampable_features=[0],
        )
        _write_site(
            tmp_path,
            layer_idx=0,
            hook_str="post_block",
            encoder_weight=torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]),
            encoder_bias=torch.zeros(1),
            decoder_weight=torch.zeros(1, 4),
        )
        with pytest.raises(ValueError, match="finite values"):
            load_sae_module_from_dir(tmp_path)

    def test_jumprelu_manifest_round_trips_threshold_tensor(self, tmp_path: Path):
        _write_manifest(
            tmp_path,
            d_model=4,
            d_sae=8,
            activation="jumprelu",
            layers=[(0, "post_block")],
            clampable_features=[0, 3],
        )
        threshold = torch.tensor([0.25, 0.75])
        _write_site(
            tmp_path,
            layer_idx=0,
            hook_str="post_block",
            encoder_weight=torch.zeros(2, 4),
            encoder_bias=torch.zeros(2),
            decoder_weight=torch.zeros(2, 4),
            threshold=threshold,
        )
        loaded = load_sae_module_from_dir(tmp_path)
        assert loaded.manifest.activation is SAEActivation.JUMPRELU
        assert loaded.manifest.activation_params == {}
        assert torch.equal(loaded.weights[(0, "post_block")]["threshold"], threshold)

    def test_jumprelu_manifest_missing_threshold_tensor_raises(self, tmp_path: Path):
        _write_manifest(
            tmp_path,
            d_model=4,
            d_sae=8,
            activation="jumprelu",
            layers=[(0, "post_block")],
            clampable_features=[0, 3],
        )
        _write_site(
            tmp_path,
            layer_idx=0,
            hook_str="post_block",
            encoder_weight=torch.zeros(2, 4),
            encoder_bias=torch.zeros(2),
            decoder_weight=torch.zeros(2, 4),
            # threshold deliberately omitted — required for JumpReLU.
        )
        with pytest.raises(ValueError, match="threshold"):
            load_sae_module_from_dir(tmp_path)

    def test_jumprelu_manifest_threshold_shape_mismatch_raises(self, tmp_path: Path):
        _write_manifest(
            tmp_path,
            d_model=4,
            d_sae=8,
            activation="jumprelu",
            layers=[(0, "post_block")],
            clampable_features=[0, 3],
        )
        _write_site(
            tmp_path,
            layer_idx=0,
            hook_str="post_block",
            encoder_weight=torch.zeros(2, 4),
            encoder_bias=torch.zeros(2),
            decoder_weight=torch.zeros(2, 4),
            threshold=torch.zeros(3),  # wrong: should be (2,)
        )
        with pytest.raises(ValueError, match="threshold"):
            load_sae_module_from_dir(tmp_path)


# ---------------------------------------------------------------------------
# Gemma Scope NPZ layout
# ---------------------------------------------------------------------------


class TestLoadGemmaScopeSAE:
    """Subset a synthetic NPZ and verify shapes and content."""

    def test_subsets_encoder_and_decoder_rows(self, tmp_path: Path):
        d_model, d_sae = 8, 16
        path = tmp_path / "params.npz"
        arrs = _make_gemma_scope_npz(path, d_model=d_model, d_sae=d_sae)

        feats = [3, 7, 11]
        loaded = load_gemma_scope_sae(
            path,
            layer_idx=20,
            hook_str="post_block",
            clampable_features=feats,
        )

        assert isinstance(loaded, LoadedSAEModule)
        assert loaded.manifest.d_model == d_model
        assert loaded.manifest.d_sae == d_sae
        assert loaded.manifest.layers == ((20, "post_block"),)
        assert loaded.manifest.clampable_features == tuple(feats)

        site = loaded.weights[(20, "post_block")]
        # encoder_weight has rows from W_enc.T at the feature indices.
        expected_enc = arrs["W_enc"].T[feats]
        expected_dec = arrs["W_dec"][feats]
        expected_b = arrs["b_enc"][feats]
        assert torch.allclose(
            site["encoder_weight"], torch.from_numpy(expected_enc).float()
        )
        assert torch.allclose(
            site["decoder_weight"], torch.from_numpy(expected_dec).float()
        )
        assert torch.allclose(
            site["encoder_bias"], torch.from_numpy(expected_b).float()
        )

    def test_npz_file_handle_is_closed(self, tmp_path: Path, monkeypatch):
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path)

        npz = np.load(str(path))

        def tracking_load(*args, **kwargs):
            return npz

        monkeypatch.setattr(np, "load", tracking_load)
        load_gemma_scope_sae(
            path,
            layer_idx=20,
            hook_str="post_block",
            clampable_features=[0, 1],
        )
        assert npz.zip is None

    def test_default_threshold_is_per_feature_subset(self, tmp_path: Path):
        d_model, d_sae = 4, 8
        path = tmp_path / "params.npz"
        arrs = _make_gemma_scope_npz(path, d_model=d_model, d_sae=d_sae)

        feats = [0, 2, 4]
        loaded = load_gemma_scope_sae(
            path, layer_idx=0, hook_str="post_block", clampable_features=feats
        )
        assert loaded.manifest.activation is SAEActivation.JUMPRELU
        # Per-feature thresholds ride the weights dict, aligned with the
        # clampable-features order; the manifest stays param-free.
        assert loaded.manifest.activation_params == {}
        threshold = loaded.weights[(0, "post_block")]["threshold"]
        assert threshold.dtype is torch.float32
        assert torch.equal(
            threshold, torch.from_numpy(arrs["threshold"][feats]).float()
        )

    def test_explicit_activation_params_override(self, tmp_path: Path):
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path)

        loaded = load_gemma_scope_sae(
            path,
            layer_idx=0,
            hook_str="post_block",
            clampable_features=[0, 1],
            activation_params={"threshold": 0.123},
        )
        # The override emits a constant per-feature vector; the manifest
        # still carries empty activation_params for JumpReLU.
        assert loaded.manifest.activation_params == {}
        threshold = loaded.weights[(0, "post_block")]["threshold"]
        assert threshold.dtype is torch.float32
        assert torch.equal(threshold, torch.full((2,), 0.123))

    def test_invalid_explicit_activation_params_raise(self, tmp_path: Path):
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path)

        with pytest.raises(ValueError, match="activation_params"):
            load_gemma_scope_sae(
                path,
                layer_idx=0,
                hook_str="post_block",
                clampable_features=[0, 1],
                activation_params={"threshold": True},
            )

    def test_non_floating_npz_array_raises_value_error(self, tmp_path: Path):
        path = tmp_path / "params.npz"
        arrs = _make_gemma_scope_npz(path)
        arrs["W_dec"] = arrs["W_dec"].astype(np.int64)
        np.savez(str(path), **arrs)

        with pytest.raises(ValueError, match="W_dec.*floating dtype"):
            load_gemma_scope_sae(
                path,
                layer_idx=20,
                hook_str="post_block",
                clampable_features=[0],
            )

    def test_non_finite_npz_array_raises_value_error(self, tmp_path: Path):
        path = tmp_path / "params.npz"
        arrs = _make_gemma_scope_npz(path)
        arrs["W_enc"][0, 0] = np.inf
        np.savez(str(path), **arrs)

        with pytest.raises(ValueError, match="finite values"):
            load_gemma_scope_sae(
                path,
                layer_idx=20,
                hook_str="post_block",
                clampable_features=[0],
            )

    def test_relu_activation_drops_threshold(self, tmp_path: Path):
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path)

        loaded = load_gemma_scope_sae(
            path,
            layer_idx=0,
            hook_str="post_block",
            clampable_features=[0, 1],
            activation=SAEActivation.RELU,
        )
        assert loaded.manifest.activation is SAEActivation.RELU
        # ReLU has no params; default behaviour returns an empty dict.
        assert loaded.manifest.activation_params == {}
        # And the weights dict carries no threshold tensor — the
        # zero-filled runtime buffer suffices for non-JumpReLU sites.
        assert "threshold" not in loaded.weights[(0, "post_block")]

    def test_unexpected_jumprelu_override_keys_raise(self, tmp_path: Path):
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path)

        with pytest.raises(ValueError, match="activation_params"):
            load_gemma_scope_sae(
                path,
                layer_idx=0,
                hook_str="post_block",
                clampable_features=[0, 1],
                activation_params={"threshold": 0.1, "bogus": 1.0},
            )

    def test_merge_sites_with_differing_thresholds_succeeds(self, tmp_path: Path):
        # Two Gemma Scope sites whose per-feature thresholds differ —
        # under the old scalar-median fold their activation_params
        # disagreed and the merge failed; per-site threshold tensors
        # merge cleanly.
        d_model, d_sae = 4, 8
        path_a = tmp_path / "a.npz"
        path_b = tmp_path / "b.npz"
        arrs_a = _make_gemma_scope_npz(path_a, d_model=d_model, d_sae=d_sae, seed=1)
        arrs_b = _make_gemma_scope_npz(path_b, d_model=d_model, d_sae=d_sae, seed=2)
        feats = [1, 3]
        assert not np.allclose(arrs_a["threshold"][feats], arrs_b["threshold"][feats])

        a = load_gemma_scope_sae(
            path_a, layer_idx=0, hook_str="post_block", clampable_features=feats
        )
        b = load_gemma_scope_sae(
            path_b, layer_idx=1, hook_str="post_attn", clampable_features=feats
        )
        merged = merge_loaded_sae_modules([a, b])

        assert merged.manifest.layers == ((0, "post_block"), (1, "post_attn"))
        assert merged.manifest.activation_params == {}
        assert torch.equal(
            merged.weights[(0, "post_block")]["threshold"],
            torch.from_numpy(arrs_a["threshold"][feats]).float(),
        )
        assert torch.equal(
            merged.weights[(1, "post_attn")]["threshold"],
            torch.from_numpy(arrs_b["threshold"][feats]).float(),
        )

    def test_dtype_argument_is_honoured(self, tmp_path: Path):
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path)

        loaded = load_gemma_scope_sae(
            path,
            layer_idx=0,
            hook_str="post_block",
            clampable_features=[0, 1],
            weights_dtype=torch.bfloat16,
        )
        site = loaded.weights[(0, "post_block")]
        assert site["encoder_weight"].dtype is torch.bfloat16
        assert site["encoder_bias"].dtype is torch.bfloat16
        assert site["decoder_weight"].dtype is torch.bfloat16

    def test_invalid_hook_raises_value_error(self, tmp_path: Path):
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path)

        with pytest.raises(ValueError, match="not a valid hook point"):
            load_gemma_scope_sae(
                path, layer_idx=0, hook_str="nonsense", clampable_features=[0]
            )

    def test_negative_layer_idx_raises(self, tmp_path: Path):
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path)

        with pytest.raises(ValueError, match="non-negative"):
            load_gemma_scope_sae(
                path,
                layer_idx=-1,
                hook_str="post_block",
                clampable_features=[0],
            )

    def test_non_floating_weights_dtype_raises(self, tmp_path: Path):
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path)

        with pytest.raises(ValueError, match="weights_dtype"):
            load_gemma_scope_sae(
                path,
                layer_idx=0,
                hook_str="post_block",
                clampable_features=[0],
                weights_dtype=torch.int64,
            )

    def test_empty_clampable_features_raises(self, tmp_path: Path):
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path)
        with pytest.raises(ValueError, match="non-empty"):
            load_gemma_scope_sae(
                path, layer_idx=0, hook_str="post_block", clampable_features=[]
            )

    def test_duplicate_clampable_features_raises(self, tmp_path: Path):
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path)
        with pytest.raises(ValueError, match="unique"):
            load_gemma_scope_sae(
                path,
                layer_idx=0,
                hook_str="post_block",
                clampable_features=[0, 1, 0],
            )

    @pytest.mark.parametrize(
        ("layer_idx", "clampable_features", "match"),
        [
            (True, [0], "layer_idx"),
            (0, [True], "clampable_features"),
            (0, ["1"], "clampable_features"),
        ],
    )
    def test_non_integer_indices_raise(
        self,
        tmp_path: Path,
        layer_idx,
        clampable_features,
        match: str,
    ):
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path)
        with pytest.raises(ValueError, match=match):
            load_gemma_scope_sae(
                path,
                layer_idx=layer_idx,
                hook_str="post_block",
                clampable_features=clampable_features,
            )

    def test_out_of_range_clampable_features_raises(self, tmp_path: Path):
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path, d_sae=4)
        with pytest.raises(ValueError, match="outside"):
            load_gemma_scope_sae(
                path,
                layer_idx=0,
                hook_str="post_block",
                clampable_features=[0, 99],
            )

    def test_missing_keys_raises(self, tmp_path: Path):
        # Synthesise an NPZ that lacks the threshold array.
        d_model, d_sae = 4, 8
        path = tmp_path / "broken.npz"
        np.savez(
            str(path),
            W_enc=np.zeros((d_model, d_sae), dtype=np.float32),
            W_dec=np.zeros((d_sae, d_model), dtype=np.float32),
            b_enc=np.zeros((d_sae,), dtype=np.float32),
        )
        with pytest.raises(ValueError, match="missing required keys"):
            load_gemma_scope_sae(
                path, layer_idx=0, hook_str="post_block", clampable_features=[0]
            )

    def test_inconsistent_shapes_raise(self, tmp_path: Path):
        # W_enc claims d_sae=8, W_dec claims d_sae=16.
        path = tmp_path / "broken.npz"
        np.savez(
            str(path),
            W_enc=np.zeros((4, 8), dtype=np.float32),
            W_dec=np.zeros((16, 4), dtype=np.float32),
            b_enc=np.zeros((8,), dtype=np.float32),
            threshold=np.zeros((8,), dtype=np.float32),
        )
        with pytest.raises(ValueError, match="d_sae"):
            load_gemma_scope_sae(
                path, layer_idx=0, hook_str="post_block", clampable_features=[0]
            )


class TestLoadGemmaScopeSAEFullRecon:
    """Full-reconstruction loader: full (d_sae,) threshold vector."""

    def test_emits_full_d_sae_threshold_vector(self, tmp_path: Path):
        d_model, d_sae = 8, 16
        path = tmp_path / "params.npz"
        arrs = _make_gemma_scope_npz(path, d_model=d_model, d_sae=d_sae)

        loaded = load_gemma_scope_sae_full_recon(
            path,
            layer_idx=20,
            hook_str="post_block",
            clampable_features=[3, 7],
        )
        assert loaded.manifest.activation is SAEActivation.JUMPRELU
        assert loaded.manifest.activation_params == {}
        site = loaded.weights[(20, "post_block")]
        threshold = site["threshold"]
        assert threshold.shape == (d_sae,)
        assert threshold.dtype is torch.float32
        assert torch.equal(threshold, torch.from_numpy(arrs["threshold"]).float())

    def test_override_emits_constant_full_vector(self, tmp_path: Path):
        d_model, d_sae = 8, 16
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path, d_model=d_model, d_sae=d_sae)

        loaded = load_gemma_scope_sae_full_recon(
            path,
            layer_idx=20,
            hook_str="post_block",
            clampable_features=[3, 7],
            activation_params={"threshold": 0.5},
        )
        assert loaded.manifest.activation_params == {}
        threshold = loaded.weights[(20, "post_block")]["threshold"]
        assert torch.equal(threshold, torch.full((d_sae,), 0.5))

    def test_relu_activation_has_no_threshold(self, tmp_path: Path):
        path = tmp_path / "params.npz"
        _make_gemma_scope_npz(path)

        loaded = load_gemma_scope_sae_full_recon(
            path,
            layer_idx=20,
            hook_str="post_block",
            clampable_features=[0, 1],
            activation=SAEActivation.RELU,
        )
        assert loaded.manifest.activation_params == {}
        assert "threshold" not in loaded.weights[(20, "post_block")]


# ---------------------------------------------------------------------------
# Multi-site composition
# ---------------------------------------------------------------------------


class TestMergeLoadedSAEModules:
    """Combining per-(layer, hook) parts into a multi-site module."""

    def _part(
        self,
        *,
        layer_idx: int,
        hook_str: str,
        d_model: int = 4,
        d_sae: int = 8,
        clampable_features: tuple[int, ...] = (0, 1),
        activation: SAEActivation = SAEActivation.RELU,
        activation_params: dict[str, float] | None = None,
    ) -> LoadedSAEModule:
        from vllm.entrypoints.openai.steering.registry import SAEModuleManifest

        n_clamp = len(clampable_features)
        rng = torch.Generator(device="cpu").manual_seed(layer_idx * 100 + 1)
        manifest = SAEModuleManifest(
            d_model=d_model,
            d_sae=d_sae,
            activation=activation,
            layers=((layer_idx, hook_str),),
            clampable_features=clampable_features,
            activation_params=activation_params or {},
        )
        weights = {
            (layer_idx, hook_str): {
                "encoder_weight": torch.randn(n_clamp, d_model, generator=rng),
                "encoder_bias": torch.randn(n_clamp, generator=rng),
                "decoder_weight": torch.randn(n_clamp, d_model, generator=rng),
            },
        }
        return LoadedSAEModule(manifest=manifest, weights=weights)

    def test_merges_two_sites(self):
        a = self._part(layer_idx=0, hook_str="post_block")
        b = self._part(layer_idx=1, hook_str="post_attn")

        merged = merge_loaded_sae_modules([a, b])

        assert merged.manifest.layers == ((0, "post_block"), (1, "post_attn"))
        assert (0, "post_block") in merged.weights
        assert (1, "post_attn") in merged.weights
        # Weights pass through unchanged.
        assert torch.equal(
            merged.weights[(0, "post_block")]["encoder_weight"],
            a.weights[(0, "post_block")]["encoder_weight"],
        )

    def test_rejects_disagreeing_d_model(self):
        a = self._part(layer_idx=0, hook_str="post_block", d_model=4)
        b = self._part(layer_idx=1, hook_str="post_attn", d_model=8)
        with pytest.raises(ValueError, match="d_model"):
            merge_loaded_sae_modules([a, b])

    def test_rejects_duplicate_sites(self):
        a = self._part(layer_idx=0, hook_str="post_block")
        b = self._part(layer_idx=0, hook_str="post_block")
        with pytest.raises(ValueError, match="declared"):
            merge_loaded_sae_modules([a, b])

    def test_rejects_disagreeing_clampable_features(self):
        a = self._part(layer_idx=0, hook_str="post_block", clampable_features=(0, 1))
        b = self._part(layer_idx=1, hook_str="post_attn", clampable_features=(0, 2))
        with pytest.raises(ValueError, match="clampable_features"):
            merge_loaded_sae_modules([a, b])

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="non-empty"):
            merge_loaded_sae_modules([])
