# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-disk SAE checkpoint readers.

Two layouts are supported:

* **Generic ``manifest.json`` + per-(layer, hook) safetensors.**  The
  intended on-disk format for vLLM-side SAE registration.  The
  directory contains a JSON manifest (the same shape that
  :func:`vllm.entrypoints.openai.steering.registry.sae_manifest_from_dict`
  consumes) plus one safetensors file per ``(layer, hook)`` site whose
  rows are *already* aligned to the manifest's ``clampable_features``
  ordering.

* **Gemma Scope NPZ (``params.npz``).**  Per-(layer, hook) Gemma Scope
  releases publish a single NPZ that holds the *full* SAE for that
  site (``W_enc``, ``W_dec``, ``b_enc``, ``threshold``, ``b_dec``).
  This loader reads that file, validates shape, subsets the encoder
  / decoder rows by a caller-supplied feature index list, and produces
  the same ``(manifest, weights)`` pair the safetensors path produces.

Both paths return a :class:`LoadedSAEModule` whose ``manifest`` and
``weights`` fields can be passed verbatim to
``register_steering_modules`` (after marshalling via
:func:`_sae_manifest_to_dict`) and ``attach_sae_weights`` on the
worker mixin.

Out of scope:

* **Network downloads.**  Callers (tests, CLI, runtime API) fetch the
  artifact and hand the loader a local path.  Keeping this module
  network-free preserves the unit-test contract — the readers are
  deterministic functions of on-disk bytes.
* **Per-feature JumpReLU thresholds.**  The current kernel takes a
  scalar ``threshold`` parameter; Gemma Scope ships per-feature
  thresholds.  When the loader reads a per-feature threshold array,
  the median of the *clampable-subset* thresholds is used by default,
  and an explicit ``activation_params={"threshold": value}`` override
  is honoured if supplied.  Tracked as a follow-up; documented inline
  on :func:`load_gemma_scope_sae`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.config.sae_steering_types import SAEActivation
from vllm.entrypoints.openai.steering.registry import (
    SAEModuleManifest,
    sae_manifest_from_dict,
)
from vllm.model_executor.layers.steering import VALID_HOOK_POINT_NAMES


@dataclass
class LoadedSAEModule:
    """In-memory result of loading an SAE checkpoint from disk.

    ``manifest`` is the strongly-typed manifest the registry expects;
    ``weights`` maps each ``(layer_idx, hook_str)`` site declared by
    the manifest to a dict of ``{"encoder_weight", "encoder_bias",
    "decoder_weight"}`` tensors whose rows are already aligned to
    ``manifest.clampable_features``.

    ``weights`` is sized to *all* (layer, hook) sites declared in
    ``manifest.layers``; callers (e.g. :meth:`attach_sae_weights`)
    silently skip sites that the local rank doesn't own.
    """

    manifest: SAEModuleManifest
    weights: dict[tuple[int, str], dict[str, torch.Tensor]]


# ---------------------------------------------------------------------------
# Generic ``manifest.json`` + safetensors layout
# ---------------------------------------------------------------------------


def _site_filename(layer_idx: int, hook_str: str) -> str:
    """Filename for the per-(layer, hook) safetensors weight file.

    The format ``layer_<idx>_<hook>.safetensors`` is the convention
    this loader writes against; alternative layouts can be supported
    by writing a sibling reader.
    """
    return f"layer_{layer_idx}_{hook_str}.safetensors"


def load_sae_module_from_dir(path: str | Path) -> LoadedSAEModule:
    """Load an SAE module from a ``manifest.json`` + safetensors layout.

    Expected directory layout::

        <path>/
            manifest.json                       — SAEModuleManifest dump
            layer_<idx>_<hook>.safetensors      — one per (layer, hook)

    Each safetensors file must contain three tensors keyed
    ``encoder_weight`` (``n_clamp, d_model``), ``encoder_bias``
    (``n_clamp,``), and ``decoder_weight`` (``n_clamp, d_model``),
    where the row index ``i`` corresponds to
    ``manifest.clampable_features[i]``.

    Raises:
        FileNotFoundError: when the directory or a required file is
            missing.
        ValueError: when the manifest fails validation, a tensor is
            missing, or shapes disagree with the manifest.
    """
    base = Path(path)
    if not base.is_dir():
        raise FileNotFoundError(f"SAE module directory not found: {base!r}.")
    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"SAE manifest not found at {manifest_path!r}.  Expected "
            "'manifest.json' inside the SAE module directory."
        )
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest_payload = json.load(fh)
    if not isinstance(manifest_payload, dict):
        raise ValueError(f"SAE manifest at {manifest_path!r} must be a JSON object.")
    manifest = sae_manifest_from_dict(manifest_payload)
    weights = _load_weights_for_manifest(manifest, base)
    return LoadedSAEModule(manifest=manifest, weights=weights)


def _load_weights_for_manifest(
    manifest: SAEModuleManifest, base: Path
) -> dict[tuple[int, str], dict[str, torch.Tensor]]:
    """Read one safetensors file per ``(layer, hook)`` declared in *manifest*."""
    # Local import: ``safetensors`` may not be present in every test env;
    # only deferred-imported here so the loader module stays importable.
    from safetensors.torch import load_file

    n_clamp = len(manifest.clampable_features)
    out: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for layer_idx, hook_str in manifest.layers:
        if hook_str not in VALID_HOOK_POINT_NAMES:
            raise ValueError(
                f"Manifest declares unsupported hook point {hook_str!r} for "
                f"layer {layer_idx}.  Valid hook points: "
                f"{sorted(VALID_HOOK_POINT_NAMES)}."
            )
        weight_path = base / _site_filename(layer_idx, hook_str)
        if not weight_path.exists():
            raise FileNotFoundError(
                f"SAE weight file not found at {weight_path!r}.  "
                f"Manifest declares site (layer={layer_idx}, "
                f"hook={hook_str!r})."
            )
        tensors = load_file(str(weight_path))
        site = _validate_site_tensors(
            tensors,
            d_model=manifest.d_model,
            n_clamp=n_clamp,
            site=(layer_idx, hook_str),
        )
        out[(layer_idx, hook_str)] = site
    return out


def _validate_site_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    d_model: int,
    n_clamp: int,
    site: tuple[int, str],
) -> dict[str, torch.Tensor]:
    """Validate the three required tensors for one ``(layer, hook)`` site."""
    expected_shapes: dict[str, tuple[int, ...]] = {
        "encoder_weight": (n_clamp, d_model),
        "encoder_bias": (n_clamp,),
        "decoder_weight": (n_clamp, d_model),
    }
    out: dict[str, torch.Tensor] = {}
    for key, expected in expected_shapes.items():
        if key not in tensors:
            raise ValueError(
                f"SAE site {site!r}: weight file is missing tensor {key!r}.  "
                f"Required: {sorted(expected_shapes)}."
            )
        tensor = tensors[key]
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"SAE site {site!r}: tensor {key!r} has shape "
                f"{tuple(tensor.shape)}; expected {expected}."
            )
        out[key] = tensor
    return out


# ---------------------------------------------------------------------------
# Gemma Scope NPZ layout
# ---------------------------------------------------------------------------

# Required keys in a Gemma Scope ``params.npz``.  Other keys (e.g.
# ``b_dec``) are ignored: the delta op only needs the encoder rows for
# the clampable subset and the per-feature decoder directions.
_GEMMA_SCOPE_KEYS = ("W_enc", "W_dec", "b_enc", "threshold")


def load_gemma_scope_sae(
    npz_path: str | Path,
    *,
    layer_idx: int,
    hook_str: str,
    clampable_features: Sequence[int],
    activation: SAEActivation = SAEActivation.JUMPRELU,
    activation_params: Mapping[str, float] | None = None,
    weights_dtype: torch.dtype = torch.float32,
) -> LoadedSAEModule:
    """Load one ``(layer, hook)`` SAE from a Gemma Scope ``params.npz``.

    Gemma Scope distributes SAEs as per-(layer, hook) NPZ archives.
    Each archive holds the full SAE for that site:

        W_enc:     (d_model, d_sae)        encoder weights
        W_dec:     (d_sae, d_model)        decoder rows (per feature)
        b_enc:     (d_sae,)                encoder biases
        threshold: (d_sae,)                per-feature JumpReLU thresholds
        b_dec:     (d_model,)              decoder bias (unused here)

    All released Gemma Scope SAEs use JumpReLU activation; the
    encoder activation function for one feature is
    ``f_i = pre_act_i if pre_act_i > threshold[i] else 0``.

    The current SAE kernel takes a *scalar* ``threshold`` parameter
    (kept simple so the kernel JIT specialises on a single constexpr
    branch).  When ``activation_params`` is omitted, the median of
    the *clampable-subset* thresholds is used.  Pass
    ``activation_params={"threshold": value}`` to override.  This
    simplification is acceptable for the absolute-clamp path
    (``delta = target - f``) where the steering effect is dominated
    by ``target`` and the choice of threshold only perturbs ``f``;
    it is **not** acceptable for additive clamps with
    ``only_if_active=True``, which depend on whether ``f > 0`` matches
    the SAE's training distribution.  Tracked as a follow-up.

    Returns a :class:`LoadedSAEModule` covering only the supplied
    ``(layer_idx, hook_str)`` site.  Multi-site composition (one SAE
    spanning multiple layers) is built by combining several
    :class:`LoadedSAEModule` results; the test harness in
    ``tests/models/language/generation/test_sae_steering_real_weights.py``
    shows how.

    Args:
        npz_path: path to the Gemma Scope ``params.npz`` for this site.
        layer_idx: layer index this SAE is bound to.
        hook_str: hook point string (e.g. ``"post_mlp"``); validated
            against :data:`VALID_HOOK_POINT_NAMES`.
        clampable_features: feature indices to extract from the full
            SAE.  Must be non-empty and unique.  Out-of-range indices
            raise :class:`ValueError`.
        activation: encoder activation function.  Defaults to
            :attr:`SAEActivation.JUMPRELU`, matching Gemma Scope.
        activation_params: explicit activation params override.
            JumpReLU expects ``{"threshold": float}``.  When omitted,
            the median per-feature threshold over the clampable
            subset is used.
        weights_dtype: dtype the encoder/decoder tensors are
            materialised in.  Defaults to fp32; callers that intend
            to pass weights to the worker can supply the model's
            compute dtype to avoid an extra cast in
            :meth:`attach_sae_weights`.
    """
    if hook_str not in VALID_HOOK_POINT_NAMES:
        raise ValueError(
            f"hook_str {hook_str!r} is not a valid hook point.  "
            f"Valid: {sorted(VALID_HOOK_POINT_NAMES)}."
        )
    feature_list = list(clampable_features)
    if not feature_list:
        raise ValueError(
            "clampable_features must be a non-empty sequence of feature indices."
        )
    if len(set(feature_list)) != len(feature_list):
        raise ValueError(
            f"clampable_features must be unique; got duplicates in {feature_list}."
        )

    npz = np.load(str(npz_path))
    missing = [k for k in _GEMMA_SCOPE_KEYS if k not in npz.files]
    if missing:
        raise ValueError(
            f"Gemma Scope NPZ at {npz_path!r} is missing required keys: "
            f"{missing}.  Found: {sorted(npz.files)}."
        )
    W_enc = npz["W_enc"]  # (d_model, d_sae)
    W_dec = npz["W_dec"]  # (d_sae, d_model)
    b_enc = npz["b_enc"]  # (d_sae,)
    thresholds = npz["threshold"]  # (d_sae,)

    if W_enc.ndim != 2 or W_dec.ndim != 2:
        raise ValueError(
            f"Gemma Scope NPZ at {npz_path!r}: W_enc/W_dec must be 2D; "
            f"got W_enc.shape={W_enc.shape}, W_dec.shape={W_dec.shape}."
        )
    d_model_enc, d_sae_enc = W_enc.shape
    d_sae_dec, d_model_dec = W_dec.shape
    if d_model_enc != d_model_dec:
        raise ValueError(
            f"Gemma Scope NPZ at {npz_path!r}: W_enc and W_dec disagree on "
            f"d_model ({d_model_enc} vs {d_model_dec})."
        )
    if d_sae_enc != d_sae_dec:
        raise ValueError(
            f"Gemma Scope NPZ at {npz_path!r}: W_enc and W_dec disagree on "
            f"d_sae ({d_sae_enc} vs {d_sae_dec})."
        )
    if b_enc.shape != (d_sae_enc,):
        raise ValueError(
            f"Gemma Scope NPZ at {npz_path!r}: b_enc.shape={b_enc.shape}, "
            f"expected ({d_sae_enc},)."
        )
    if thresholds.shape != (d_sae_enc,):
        raise ValueError(
            f"Gemma Scope NPZ at {npz_path!r}: threshold.shape="
            f"{thresholds.shape}, expected ({d_sae_enc},)."
        )

    d_model = d_model_enc
    d_sae = d_sae_enc
    out_of_range = [f for f in feature_list if f < 0 or f >= d_sae]
    if out_of_range:
        raise ValueError(
            f"clampable_features contains indices outside [0, {d_sae}): {out_of_range}."
        )

    feat_idx = np.asarray(feature_list, dtype=np.int64)
    encoder_subset = W_enc.T[feat_idx]  # (n_clamp, d_model)
    encoder_bias_subset = b_enc[feat_idx]  # (n_clamp,)
    decoder_subset = W_dec[feat_idx]  # (n_clamp, d_model)
    threshold_subset = thresholds[feat_idx]  # (n_clamp,)

    enc_w_t = torch.from_numpy(np.ascontiguousarray(encoder_subset)).to(weights_dtype)
    enc_b_t = torch.from_numpy(np.ascontiguousarray(encoder_bias_subset)).to(
        weights_dtype
    )
    dec_w_t = torch.from_numpy(np.ascontiguousarray(decoder_subset)).to(weights_dtype)

    resolved_params = _resolve_jumprelu_params(
        activation=activation,
        activation_params=activation_params,
        threshold_subset=threshold_subset,
    )
    manifest = SAEModuleManifest(
        d_model=int(d_model),
        d_sae=int(d_sae),
        activation=activation,
        layers=((int(layer_idx), str(hook_str)),),
        clampable_features=tuple(int(f) for f in feature_list),
        activation_params=resolved_params,
    )
    return LoadedSAEModule(
        manifest=manifest,
        weights={
            (int(layer_idx), str(hook_str)): {
                "encoder_weight": enc_w_t,
                "encoder_bias": enc_b_t,
                "decoder_weight": dec_w_t,
            },
        },
    )


def _resolve_jumprelu_params(
    *,
    activation: SAEActivation,
    activation_params: Mapping[str, float] | None,
    threshold_subset: np.ndarray,
) -> dict[str, float]:
    """Pick the scalar threshold the kernel will use for this site.

    Honours an explicit ``activation_params['threshold']`` override
    when supplied; otherwise falls back to the median of the
    clampable-subset thresholds for JumpReLU activations.  Other
    activations carry their caller-supplied params unchanged.
    """
    if activation_params is not None:
        return dict(activation_params)
    if activation is SAEActivation.JUMPRELU:
        if threshold_subset.size == 0:
            return {"threshold": 0.0}
        return {"threshold": float(np.median(threshold_subset))}
    return {}


def merge_loaded_sae_modules(
    parts: Sequence[LoadedSAEModule],
    *,
    name: str | None = None,
) -> LoadedSAEModule:
    """Combine per-(layer, hook) :class:`LoadedSAEModule`s into one module.

    All parts must share ``d_model``, ``d_sae``, ``activation``,
    ``activation_params``, and ``clampable_features`` (the kernel
    needs the same feature ordering across every site of one module).
    Sites must be unique — declaring the same ``(layer_idx, hook_str)``
    in two parts raises :class:`ValueError`.

    Args:
        parts: per-(layer, hook) loaded modules to merge.
        name: optional label included in error messages; the manifest
            doesn't carry a name field, so this is purely diagnostic.

    Returns:
        A single :class:`LoadedSAEModule` whose ``manifest.layers``
        is the union of all parts' sites and whose ``weights`` dict
        is the union of all parts' weight dicts.
    """
    if not parts:
        raise ValueError("merge_loaded_sae_modules: parts must be non-empty.")
    head = parts[0]
    layers_seen: set[tuple[int, str]] = set()
    merged_layers: list[tuple[int, str]] = []
    merged_weights: dict[tuple[int, str], dict[str, torch.Tensor]] = {}

    def _check(field: str, expected: Any, got: Any, idx: int) -> None:
        if expected != got:
            tag = f" [{name!r}]" if name else ""
            raise ValueError(
                f"merge_loaded_sae_modules{tag}: parts disagree on {field!r} "
                f"(part 0: {expected!r}; part {idx}: {got!r})."
            )

    for idx, part in enumerate(parts):
        m = part.manifest
        _check("d_model", head.manifest.d_model, m.d_model, idx)
        _check("d_sae", head.manifest.d_sae, m.d_sae, idx)
        _check("activation", head.manifest.activation, m.activation, idx)
        _check(
            "activation_params",
            head.manifest.activation_params,
            m.activation_params,
            idx,
        )
        _check(
            "clampable_features",
            head.manifest.clampable_features,
            m.clampable_features,
            idx,
        )
        for layer_idx, hook_str in m.layers:
            key = (int(layer_idx), str(hook_str))
            if key in layers_seen:
                tag = f" [{name!r}]" if name else ""
                raise ValueError(
                    f"merge_loaded_sae_modules{tag}: site {key!r} declared "
                    "more than once across input parts."
                )
            layers_seen.add(key)
            merged_layers.append(key)
            if key not in part.weights:
                tag = f" [{name!r}]" if name else ""
                raise ValueError(
                    f"merge_loaded_sae_modules{tag}: part {idx} declares "
                    f"site {key!r} in its manifest but has no weights for it."
                )
            merged_weights[key] = part.weights[key]
    merged_manifest = SAEModuleManifest(
        d_model=head.manifest.d_model,
        d_sae=head.manifest.d_sae,
        activation=head.manifest.activation,
        layers=tuple(merged_layers),
        clampable_features=head.manifest.clampable_features,
        activation_params=dict(head.manifest.activation_params),
    )
    return LoadedSAEModule(manifest=merged_manifest, weights=merged_weights)


# ---------------------------------------------------------------------------
# Gemma Scope NPZ → full-reconstruction weights
# ---------------------------------------------------------------------------


def load_gemma_scope_sae_full_recon(
    npz_path: str | Path,
    *,
    layer_idx: int,
    hook_str: str,
    clampable_features: Sequence[int],
    activation: SAEActivation = SAEActivation.JUMPRELU,
    activation_params: Mapping[str, float] | None = None,
    weights_dtype: torch.dtype = torch.float32,
) -> LoadedSAEModule:
    """Load one ``(layer, hook)`` Gemma Scope SAE for the *full-reconstruction* path.

    Sibling to :func:`load_gemma_scope_sae` for the delta path.  The
    delta loader subsets the encoder / decoder rows to
    ``clampable_features``; the full-reconstruction path needs the
    *complete* SAE forward (every feature participates in the
    reconstruction), so this loader returns the **full** ``d_sae`` ×
    ``d_model`` encoder and decoder, plus the decoder bias.

    Output shapes match what :meth:`attach_sae_full_recon_weights` on
    the worker mixin expects:

    * ``encoder_weight`` : ``(d_sae, d_model)``       — ``W_enc.T``
    * ``encoder_bias``   : ``(d_sae,)``               — ``b_enc``
    * ``decoder_weight`` : ``(d_sae, d_model)``       — ``W_dec``
    * ``decoder_bias``   : ``(d_model,)``             — ``b_dec``

    The returned :class:`SAEModuleManifest` carries the full ``d_sae``
    in ``d_sae`` and the caller-supplied ``clampable_features`` —
    those are the indices where the request-side spec may apply
    clamps.  The encoder / decoder buffers themselves cover every
    feature; the kernel reads the clampable subset by gathering at
    those indices.

    JumpReLU thresholds: same simplification as the delta-path
    loader.  Gemma Scope ships per-feature thresholds; the kernel
    takes a scalar.  The median over the *clampable subset* is used
    by default; pass ``activation_params={"threshold": value}`` to
    override.  This matters more for the full-reconstruction path
    than for the delta path — the encoder runs over every feature,
    so a uniform threshold over the full SAE is a noticeable
    simplification.  Tracked as a follow-up; for the qualitative
    output-shift assertion in the e2e test the simplification is
    acceptable, but quantitative reconstruction-fidelity work
    needs per-feature thresholds first.
    """
    if hook_str not in VALID_HOOK_POINT_NAMES:
        raise ValueError(
            f"hook_str {hook_str!r} is not a valid hook point.  "
            f"Valid: {sorted(VALID_HOOK_POINT_NAMES)}."
        )
    feature_list = list(clampable_features)
    if not feature_list:
        raise ValueError(
            "clampable_features must be a non-empty sequence of feature indices."
        )
    if len(set(feature_list)) != len(feature_list):
        raise ValueError(
            f"clampable_features must be unique; got duplicates in {feature_list}."
        )

    npz = np.load(str(npz_path))
    missing = [k for k in (*_GEMMA_SCOPE_KEYS, "b_dec") if k not in npz.files]
    if missing:
        raise ValueError(
            f"Gemma Scope NPZ at {npz_path!r} is missing required keys for "
            f"full-reconstruction: {missing}.  Found: {sorted(npz.files)}."
        )
    W_enc = npz["W_enc"]  # (d_model, d_sae)
    W_dec = npz["W_dec"]  # (d_sae, d_model)
    b_enc = npz["b_enc"]  # (d_sae,)
    b_dec = npz["b_dec"]  # (d_model,)
    thresholds = npz["threshold"]  # (d_sae,)

    if W_enc.ndim != 2 or W_dec.ndim != 2:
        raise ValueError(
            f"Gemma Scope NPZ at {npz_path!r}: W_enc/W_dec must be 2D; "
            f"got W_enc.shape={W_enc.shape}, W_dec.shape={W_dec.shape}."
        )
    d_model_enc, d_sae_enc = W_enc.shape
    d_sae_dec, d_model_dec = W_dec.shape
    if d_model_enc != d_model_dec:
        raise ValueError(
            f"Gemma Scope NPZ at {npz_path!r}: W_enc and W_dec disagree on "
            f"d_model ({d_model_enc} vs {d_model_dec})."
        )
    if d_sae_enc != d_sae_dec:
        raise ValueError(
            f"Gemma Scope NPZ at {npz_path!r}: W_enc and W_dec disagree on "
            f"d_sae ({d_sae_enc} vs {d_sae_dec})."
        )
    if b_enc.shape != (d_sae_enc,):
        raise ValueError(
            f"Gemma Scope NPZ at {npz_path!r}: b_enc.shape={b_enc.shape}, "
            f"expected ({d_sae_enc},)."
        )
    if b_dec.shape != (d_model_enc,):
        raise ValueError(
            f"Gemma Scope NPZ at {npz_path!r}: b_dec.shape={b_dec.shape}, "
            f"expected ({d_model_enc},)."
        )
    if thresholds.shape != (d_sae_enc,):
        raise ValueError(
            f"Gemma Scope NPZ at {npz_path!r}: threshold.shape={thresholds.shape}, "
            f"expected ({d_sae_enc},)."
        )

    d_model = d_model_enc
    d_sae = d_sae_enc
    out_of_range = [f for f in feature_list if f < 0 or f >= d_sae]
    if out_of_range:
        raise ValueError(
            f"clampable_features contains indices outside [0, {d_sae}): {out_of_range}."
        )

    # Full encoder / decoder + biases — no subsetting on the rows
    # themselves; we pass the entire SAE through.  ``W_enc.T`` is
    # what the buffer expects: row ``i`` is the encoder direction
    # for feature ``i``.
    enc_w_t = torch.from_numpy(np.ascontiguousarray(W_enc.T)).to(weights_dtype)
    enc_b_t = torch.from_numpy(np.ascontiguousarray(b_enc)).to(weights_dtype)
    dec_w_t = torch.from_numpy(np.ascontiguousarray(W_dec)).to(weights_dtype)
    dec_b_t = torch.from_numpy(np.ascontiguousarray(b_dec)).to(weights_dtype)

    feat_idx = np.asarray(feature_list, dtype=np.int64)
    threshold_subset = thresholds[feat_idx]
    resolved_params = _resolve_jumprelu_params(
        activation=activation,
        activation_params=activation_params,
        threshold_subset=threshold_subset,
    )
    manifest = SAEModuleManifest(
        d_model=int(d_model),
        d_sae=int(d_sae),
        activation=activation,
        layers=((int(layer_idx), str(hook_str)),),
        clampable_features=tuple(int(f) for f in feature_list),
        activation_params=resolved_params,
    )
    return LoadedSAEModule(
        manifest=manifest,
        weights={
            (int(layer_idx), str(hook_str)): {
                "encoder_weight": enc_w_t,
                "encoder_bias": enc_b_t,
                "decoder_weight": dec_w_t,
                "decoder_bias": dec_b_t,
            },
        },
    )
