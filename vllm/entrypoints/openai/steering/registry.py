# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named steering vector registry.

Provides :class:`SteeringModuleRegistry` for loading, storing, and
resolving named steering vector configurations that can be referenced
by name in API requests.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm.config.sae_steering_types import (
    SAEActivation,
    SAEClampSpec,
    SteeringModuleKind,
    coerce_sae_clamp_specs,
)
from vllm.config.steering_types import (
    ResolvedSteeringVectorSpec,
    SteeringVectorSpec,
    hash_steering_config,
    merge_steering_specs,
    normalize_layer_entry,
    resolve_effective_vectors,
    scale_steering_spec,
    validate_steering_index,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import VALID_HOOK_POINT_NAMES

logger = init_logger(__name__)


@dataclass
class SAEModuleManifest:
    """Shape description for an SAE-kind named steering module.

    The manifest is the authoritative description of the SAE that
    registration and worker execution need: model and dictionary
    dimensions, activation function, the set of (layer, hook) sites
    it covers, and the ordered set of feature indices that may be
    clamped at runtime.
    """

    d_model: int
    d_sae: int
    activation: SAEActivation
    layers: tuple[tuple[int, str], ...]
    clampable_features: tuple[int, ...]
    activation_params: dict[str, float] = field(default_factory=dict)
    weights_uri: str | None = None


@dataclass
class SteeringModule:
    """A named steering module.

    Distinguishes two kinds via :pyattr:`kind`:

    * ``ADDITIVE`` — existing precomputed-vector path; ``vectors`` /
      ``prefill_vectors`` / ``decode_vectors`` carry per-(hook, layer)
      directions and ``sae_manifest`` is ``None``.
    * ``SAE_DELTA`` — SAE feature-surgery path; ``sae_manifest``
      describes the SAE's shape and the additive vector fields are
      ``None``.

    A request references the module by name via either
    ``SamplingParams.steering_module_ref`` (additive) or
    ``SamplingParams.sae_clamp_specs`` (SAE delta).  The worker
    dispatches based on :pyattr:`kind`.
    """

    name: str
    kind: SteeringModuleKind = SteeringModuleKind.ADDITIVE
    vectors: SteeringVectorSpec | None = None
    prefill_vectors: SteeringVectorSpec | None = None
    decode_vectors: SteeringVectorSpec | None = None
    sae_manifest: SAEModuleManifest | None = None
    prefill_additive_hash: int = 0
    decode_additive_hash: int = 0


class SteeringModuleRegistry:
    """Registry for named steering vector configurations."""

    def __init__(self, valid_layer_indices: set[int] | None = None) -> None:
        self._modules: dict[str, SteeringModule] = {}
        self._lock = asyncio.Lock()
        self._valid_layer_indices = valid_layer_indices

    async def register(
        self,
        name: str,
        vectors: SteeringVectorSpec | None = None,
        prefill_vectors: SteeringVectorSpec | None = None,
        decode_vectors: SteeringVectorSpec | None = None,
        *,
        kind: SteeringModuleKind = SteeringModuleKind.ADDITIVE,
        sae_manifest: SAEModuleManifest | None = None,
    ) -> None:
        """Register a named steering module. Overwrites if name exists.

        For ``kind=ADDITIVE`` (default) the additive vector tiers are
        validated as before; passing a non-empty ``sae_manifest`` is
        an error.  For ``kind=SAE_DELTA`` the manifest is required and
        all additive vector fields must be empty.
        """
        if kind is SteeringModuleKind.ADDITIVE:
            if sae_manifest is not None:
                raise ValueError(
                    f"Steering module '{name}': sae_manifest is only valid "
                    "for kind=SAE_DELTA."
                )
            if not vectors and not prefill_vectors and not decode_vectors:
                raise ValueError(f"Steering module '{name}' has no vectors in any tier")

            # Validate hook point names and entry format
            for spec in [vectors, prefill_vectors, decode_vectors]:
                if spec:
                    invalid = set(spec.keys()) - VALID_HOOK_POINT_NAMES
                    if invalid:
                        raise ValueError(
                            f"Invalid hook point name(s) in module '{name}': "
                            f"{sorted(invalid)}. "
                            f"Valid: {sorted(VALID_HOOK_POINT_NAMES)}"
                        )
                    # Validate entries are well-formed
                    for hook_name, layers in spec.items():
                        for layer_idx, entry in layers.items():
                            self._validate_layer_index(name=name, layer_idx=layer_idx)
                            self._validate_layer_entry(
                                name=name,
                                hook_name=hook_name,
                                layer_idx=layer_idx,
                                entry=entry,
                            )

            module = SteeringModule(
                name=name,
                kind=kind,
                vectors=vectors,
                prefill_vectors=prefill_vectors,
                decode_vectors=decode_vectors,
            )
            module.prefill_additive_hash = hash_steering_config(
                resolve_effective_vectors(vectors, prefill_vectors)
            )
            module.decode_additive_hash = hash_steering_config(
                resolve_effective_vectors(vectors, decode_vectors)
            )
        elif kind is SteeringModuleKind.SAE_DELTA:
            if (
                vectors is not None
                or prefill_vectors is not None
                or decode_vectors is not None
            ):
                raise ValueError(
                    f"Steering module '{name}': additive vector fields are "
                    "not valid for kind=SAE_DELTA."
                )
            if sae_manifest is None:
                raise ValueError(
                    f"Steering module '{name}': sae_manifest is required "
                    "for kind=SAE_DELTA."
                )
            self._validate_sae_manifest(name=name, manifest=sae_manifest)
            module = SteeringModule(
                name=name,
                kind=kind,
                sae_manifest=sae_manifest,
            )
        else:
            raise ValueError(f"Steering module '{name}': unsupported kind {kind!r}")

        async with self._lock:
            self._modules[name] = module
        logger.info("Registered steering module '%s' (kind=%s)", name, kind.value)

    async def unregister(self, name: str) -> bool:
        """Remove a named module. Returns True if it existed."""
        async with self._lock:
            removed = self._modules.pop(name, None)
        if removed:
            logger.info("Unregistered steering module '%s'", name)
        return removed is not None

    async def restore_or_remove(self, name: str, prev: SteeringModule | None) -> None:
        """Rollback helper used after a failed ``register`` + broadcast.

        When *prev* is ``None`` the name was newly created by the
        failed call, so remove it; otherwise restore the prior module
        in place so a re-registration that fails mid-broadcast does
        not destroy a previously-working entry.
        """
        async with self._lock:
            if prev is None:
                self._modules.pop(name, None)
            else:
                self._modules[name] = prev

    def get(self, name: str) -> SteeringModule | None:
        """Look up a module by name. Thread-safe for reads."""
        return self._modules.get(name)

    def validate_sae_clamp_specs(
        self, specs: list[Any] | tuple[Any, ...] | None
    ) -> str | None:
        """Validate ``sae_clamp_specs`` against the registry.

        Symmetric with :meth:`validate_additive_lookup`: returns an
        English error message when any spec references a name that
        either isn't registered or isn't an SAE-kind module, otherwise
        returns ``None``.

        Doing this at the API server (instead of only at the worker)
        means a request that names a missing or wrong-kind SAE module
        becomes a 400 immediately rather than either crashing the
        worker (steering enabled) or being silently dropped (steering
        disabled, the worker mixin's admission guard never fires).
        ``specs`` may be a tuple of :class:`SAEClampSpec` (typed
        ``SamplingParams`` form) or a list of dicts (raw OpenAI
        payload before coercion).  Raw payloads are coerced here, so
        malformed clamp shapes fail at API admission instead of later
        during ``to_sampling_params`` construction.
        """
        if not specs:
            return None
        try:
            coerced_specs = coerce_sae_clamp_specs(specs)
        except ValueError as exc:
            return str(exc)
        if not coerced_specs:
            return None
        for i, spec in enumerate(coerced_specs):
            assert isinstance(spec, SAEClampSpec)
            module_name = spec.module_name
            module = self._modules.get(module_name)
            if module is None:
                return (
                    f"sae_clamp_specs[{i}] references unknown module "
                    f"{module_name!r}.  Available: "
                    f"{self.list_modules() or 'none'}"
                )
            if module.kind is not SteeringModuleKind.SAE_DELTA:
                return (
                    f"sae_clamp_specs[{i}] references module "
                    f"{module_name!r} of kind {module.kind.value!r}; "
                    "only kind='sae_delta' modules can be referenced "
                    "from sae_clamp_specs.  Use 'steering_name' for "
                    "additive modules."
                )
            manifest = module.sae_manifest
            if manifest is None:
                return (
                    f"sae_clamp_specs[{i}] references SAE module "
                    f"{module_name!r}, but the registry entry has no "
                    "sae_manifest."
                )
            covered = set(manifest.layers)
            clampable = set(manifest.clampable_features)
            for hook_name, layer_map in spec.clamps.items():
                for layer_idx, entries in layer_map.items():
                    if (layer_idx, hook_name) not in covered:
                        return (
                            f"sae_clamp_specs[{i}] for module {module_name!r} "
                            f"targets site (layer={layer_idx}, "
                            f"hook={hook_name!r}) which is not declared in "
                            "the module's sae_manifest.layers."
                        )
                    for entry in entries:
                        if entry.feature_idx not in clampable:
                            return (
                                f"sae_clamp_specs[{i}] for module "
                                f"{module_name!r} clamps "
                                f"feature_idx={entry.feature_idx}, which is "
                                "not in the module's sae_manifest."
                                "clampable_features."
                            )
        return None

    def validate_additive_lookup(self, name: str) -> str | None:
        """Validate ``name`` for use as the OpenAI ``steering_name`` field.

        ``steering_name`` references an additive-tier module only.  An
        SAE-kind module under the same name must produce a 400-style
        client error here rather than slipping through to the worker
        (where additive resolution would crash with a missing-name
        error).  Returns ``None`` when the module exists and is
        additive; otherwise returns an English error message suitable
        for a ``HTTPStatus.BAD_REQUEST`` response.
        """
        module = self._modules.get(name)
        if module is None:
            return (
                f"Unknown steering module {name!r}. "
                f"Available: {self.list_modules() or 'none'}"
            )
        if module.kind is not SteeringModuleKind.ADDITIVE:
            return (
                f"Steering module {name!r} is "
                f"kind={module.kind.value!r}; the 'steering_name' field "
                "references additive modules only.  Use 'sae_clamp_specs' "
                "to drive SAE feature surgery."
            )
        return None

    def apply_sampling_params_hash_overrides(
        self,
        sampling_params: Any,
        steering_name: str,
        *,
        scale: float = 1.0,
    ) -> str | None:
        """Align named-module request hashes with phase-effective vectors.

        ``SamplingParams`` only carries a compact ``(name, scale)`` reference
        to the worker, but the API server registry knows whether that module is
        active in prefill, decode, both, or neither after inline overrides are
        merged.  Populate the cached hash properties from those resolved
        per-phase vectors so scheduler capacity accounting and prefix-cache
        keys do not reserve rows for a phase that resolves to no additive
        steering.
        """
        module = self.get(steering_name)
        if module is None:
            return (
                f"Unknown steering module '{steering_name}'. "
                f"Available: {self.list_modules() or 'none'}"
            )
        if module.kind is not SteeringModuleKind.ADDITIVE:
            return (
                f"Steering module '{steering_name}' is "
                f"kind={module.kind.value!r}; hash overrides only support "
                "additive modules."
            )

        def _resolve_phase(phase: str) -> dict[str, dict[int, Any]] | None:
            inline_phase = (
                sampling_params.prefill_steering_vectors
                if phase == "prefill"
                else sampling_params.decode_steering_vectors
            )
            phase_module = (
                scale_steering_spec(module.prefill_vectors, scale)
                if phase == "prefill"
                else scale_steering_spec(module.decode_vectors, scale)
            )
            merged_base = merge_steering_specs(
                scale_steering_spec(module.vectors, scale),
                sampling_params.steering_vectors,
            )
            merged_phase = merge_steering_specs(phase_module, inline_phase)
            return resolve_effective_vectors(merged_base, merged_phase)

        has_inline_overrides = (
            sampling_params.steering_vectors is not None
            or sampling_params.prefill_steering_vectors is not None
            or sampling_params.decode_steering_vectors is not None
        )
        if (
            not has_inline_overrides
            and scale == 1.0
            and not sampling_params.sae_clamp_specs
        ):
            sampling_params.__dict__["prefill_additive_steering_config_hash"] = (
                module.prefill_additive_hash
            )
            sampling_params.__dict__["decode_additive_steering_config_hash"] = (
                module.decode_additive_hash
            )
            sampling_params.__dict__["prefill_steering_config_hash"] = (
                module.prefill_additive_hash
            )
            sampling_params.__dict__["decode_steering_config_hash"] = (
                module.decode_additive_hash
            )
            return None

        try:
            prefill = _resolve_phase("prefill")
            decode = _resolve_phase("decode")
        except ValueError as exc:
            return (
                f"Invalid steering composition for module '{steering_name}': {exc}"
            )

        prefill_additive_hash = hash_steering_config(prefill)
        decode_additive_hash = hash_steering_config(decode)
        _ = sampling_params.prefill_sae_clamp_config_hash
        _ = sampling_params.decode_sae_clamp_config_hash
        sampling_params.__dict__["prefill_additive_steering_config_hash"] = (
            prefill_additive_hash
        )
        sampling_params.__dict__["decode_additive_steering_config_hash"] = (
            decode_additive_hash
        )
        sampling_params.__dict__["prefill_steering_config_hash"] = (
            hash_steering_config(
                prefill,
                sae_clamp_specs=sampling_params._phase_filtered_sae_specs("prefill"),
            )
        )
        sampling_params.__dict__["decode_steering_config_hash"] = (
            hash_steering_config(
                decode,
                sae_clamp_specs=sampling_params._phase_filtered_sae_specs("decode"),
            )
        )
        return None

    def list_modules(self) -> list[str]:
        """Return sorted list of registered module names."""
        return sorted(self._modules.keys())

    def dump_for_broadcast(
        self,
        *,
        include_sae_weights: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Return a broadcastable view of every registered module.

        Used by the API server to broadcast the full registry to workers
        via ``collective_rpc`` so every worker holds an identical
        ``_steering_module_registry``.  The returned mapping is keyed by
        module name; each value is a ``dict`` carrying a ``kind``
        discriminator plus the kind-specific payload:

        * Additive: ``vectors``, ``prefill_vectors``, ``decode_vectors``
        * SAE delta: ``sae_manifest`` (a JSON-safe dict), plus
          ``sae_weights`` when *include_sae_weights* is true.

        The default form is JSON-safe.  When *include_sae_weights* is
        true, SAE tensors are loaded from each manifest's
        ``weights_uri`` and included in the payload so workers do not
        register zero-filled SAE buffers during startup/full-sync
        broadcasts.

        Note: the additive form returns *unresolved*
        :class:`SteeringVectorSpec` entries — per-layer values may
        still be in ``{"vector": [...], "scale": float}`` form.  The
        worker-side merge step normalizes scales just like the
        original ``resolve_for_request`` did on the server.
        """
        out: dict[str, dict[str, Any]] = {}
        for name, module in self._modules.items():
            payload: dict[str, Any] = {"kind": module.kind.value}
            if module.kind is SteeringModuleKind.ADDITIVE:
                payload["vectors"] = module.vectors
                payload["prefill_vectors"] = module.prefill_vectors
                payload["decode_vectors"] = module.decode_vectors
            else:
                assert module.sae_manifest is not None
                manifest = module.sae_manifest
                payload["sae_manifest"] = _sae_manifest_to_dict(manifest)
                if include_sae_weights:
                    if not manifest.weights_uri:
                        raise ValueError(
                            f"Cannot broadcast SAE module {name!r}: "
                            "manifest has no 'weights_uri' to load weights from."
                        )
                    from vllm.entrypoints.openai.steering.sae_loader import (
                        _load_weights_for_manifest,
                    )

                    payload["sae_weights"] = _load_weights_for_manifest(
                        manifest,
                        Path(manifest.weights_uri),
                    )
            out[name] = payload
        return out

    async def load_from_file(self, name: str, path: str) -> None:
        """Load a steering module from disk and register it.

        Additive modules are loaded from a JSON file with this format::

            {
                "vectors": {"post_mlp": {"14": [0.1, ...]}},
                "prefill_vectors": {"pre_attn": {"14": [0.3, ...]}},
                "decode_vectors": null,
            }

        JSON uses string keys for layer indices; they are converted to int.
        SAE delta modules are loaded from a directory containing the
        generic ``manifest.json`` + per-site safetensors layout accepted by
        :func:`vllm.entrypoints.openai.steering.sae_loader.load_sae_module_from_dir`.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Steering module file not found: {path}")

        if file_path.is_dir():
            from vllm.entrypoints.openai.steering.sae_loader import (
                load_sae_module_from_dir,
            )

            loaded = load_sae_module_from_dir(file_path)
            # Startup/full-sync broadcasts can include SAE weights only if the
            # manifest has a stable local path to reload from.
            loaded.manifest.weights_uri = str(file_path)
            await self.register(
                name=name,
                kind=SteeringModuleKind.SAE_DELTA,
                sae_manifest=loaded.manifest,
            )
            return

        with open(file_path) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Steering module file must contain a JSON object, "
                f"got {type(data).__name__}"
            )

        # Extract three tiers, converting string layer keys to int
        vectors = _convert_layer_keys(data.get("vectors"), field_name="vectors")
        prefill_vectors = _convert_layer_keys(
            data.get("prefill_vectors"), field_name="prefill_vectors"
        )
        decode_vectors = _convert_layer_keys(
            data.get("decode_vectors"), field_name="decode_vectors"
        )

        await self.register(
            name=name,
            vectors=vectors,
            prefill_vectors=prefill_vectors,
            decode_vectors=decode_vectors,
        )

    def resolve_for_request(
        self,
        steering_name: str,
        inline_vectors: SteeringVectorSpec | None,
        inline_prefill: SteeringVectorSpec | None,
        inline_decode: SteeringVectorSpec | None,
    ) -> tuple[
        ResolvedSteeringVectorSpec | None,
        ResolvedSteeringVectorSpec | None,
        ResolvedSteeringVectorSpec | None,
        str | None,
    ]:
        """Resolve a named module and merge with inline vectors.

        Returns ``(merged_vectors, merged_prefill, merged_decode, error)``.
        On success *error* is ``None``.  On failure the first three are
        ``None``.
        """
        module = self.get(steering_name)
        if module is None:
            return (
                None,
                None,
                None,
                (
                    f"Unknown steering module '{steering_name}'. "
                    f"Available: {self.list_modules() or 'none'}"
                ),
            )
        if module.kind is not SteeringModuleKind.ADDITIVE:
            return (
                None,
                None,
                None,
                (
                    f"Steering module '{steering_name}' is "
                    f"kind={module.kind.value!r}; resolve_for_request only "
                    "supports additive modules. Use sae_clamp_specs to "
                    "drive SAE feature surgery."
                ),
            )

        try:
            merged_vectors = merge_steering_specs(module.vectors, inline_vectors)
            merged_prefill = merge_steering_specs(
                module.prefill_vectors, inline_prefill
            )
            merged_decode = merge_steering_specs(module.decode_vectors, inline_decode)
        except ValueError as exc:
            return (
                None,
                None,
                None,
                (f"Invalid steering composition for module '{steering_name}': {exc}"),
            )

        return merged_vectors, merged_prefill, merged_decode, None

    @staticmethod
    def _validate_layer_entry(
        name: str,
        hook_name: str,
        layer_idx: int,
        entry: Any,
    ) -> None:
        """Validate a stored layer entry matches request-time constraints."""
        prefix = f"module {name!r}[{hook_name!r}][{layer_idx}]"
        vector: Any
        if isinstance(entry, dict):
            vector, scale = normalize_layer_entry(entry)
            if isinstance(entry["scale"], bool) or not isinstance(
                entry["scale"], (int, float)
            ):
                raise ValueError(
                    f"{prefix}['scale'] must be a finite float, "
                    f"got {type(entry['scale']).__name__}."
                )
            if not math.isfinite(scale):
                raise ValueError(f"{prefix}['scale'] must be finite, got {scale}.")
        else:
            vector, _ = normalize_layer_entry(entry)

        if not isinstance(vector, list):
            target = prefix if isinstance(entry, list) else f"{prefix}['vector']"
            raise ValueError(
                f"{target} must be a list of floats, got {type(vector).__name__}."
            )

        value_prefix = prefix if isinstance(entry, list) else f"{prefix}['vector']"
        for i, value in enumerate(vector):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{value_prefix}[{i}] must be a finite float, "
                    f"got {type(value).__name__}."
                )
            if not math.isfinite(value):
                raise ValueError(f"{value_prefix}[{i}] must be finite, got {value}.")

    def _validate_layer_index(self, name: str, layer_idx: int) -> None:
        """Validate layer indices against the loaded model when available."""
        validate_steering_index(
            layer_idx, f"Steering module {name!r} layer_idx"
        )
        if self._valid_layer_indices is None:
            return
        if layer_idx not in self._valid_layer_indices:
            raise ValueError(
                f"Steering module '{name}' references unknown layer index "
                f"{layer_idx}. Valid steerable layers: "
                f"{sorted(self._valid_layer_indices) or 'none'}"
            )

    def _validate_sae_manifest(self, *, name: str, manifest: SAEModuleManifest) -> None:
        """Structural validation for an SAE manifest.

        Bad shapes should fail loudly at registration time rather than
        at request time or during worker buffer attachment.
        """
        prefix = f"SAE module {name!r}"
        if type(manifest.d_model) is not int or manifest.d_model <= 0:
            raise ValueError(
                f"{prefix}: d_model must be a positive int, got {manifest.d_model!r}."
            )
        if type(manifest.d_sae) is not int or manifest.d_sae <= 0:
            raise ValueError(
                f"{prefix}: d_sae must be a positive int, got {manifest.d_sae!r}."
            )
        if not isinstance(manifest.activation, SAEActivation):
            raise ValueError(
                f"{prefix}: activation must be an SAEActivation enum, "
                f"got {type(manifest.activation).__name__}."
            )
        if manifest.weights_uri is not None and not isinstance(
            manifest.weights_uri, str
        ):
            raise ValueError(
                f"{prefix}: weights_uri must be a string or None, "
                f"got {manifest.weights_uri!r}."
            )
        self._validate_sae_activation_params(
            prefix=prefix,
            activation=manifest.activation,
            activation_params=manifest.activation_params,
        )
        if not isinstance(manifest.layers, tuple) or not manifest.layers:
            raise ValueError(
                f"{prefix}: layers must be a non-empty tuple of "
                "(layer_idx, hook_point) pairs."
            )
        seen_sites: set[tuple[int, str]] = set()
        for entry in manifest.layers:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[1], str)
            ):
                raise ValueError(
                    f"{prefix}: layer entries must be (int, str) tuples, got {entry!r}."
                )
            layer_idx, hook_point = entry
            validate_steering_index(layer_idx, f"{prefix}: layer index")
            self._validate_layer_index(name=name, layer_idx=layer_idx)
            if hook_point not in VALID_HOOK_POINT_NAMES:
                raise ValueError(
                    f"{prefix}: unknown hook point {hook_point!r}.  "
                    f"Valid: {sorted(VALID_HOOK_POINT_NAMES)}."
                )
            site = (layer_idx, hook_point)
            if site in seen_sites:
                raise ValueError(
                    f"{prefix}: layers must not contain duplicate "
                    f"(layer_idx, hook_point) sites; got {site!r} more than once."
                )
            seen_sites.add(site)
        for other_name, other_module in self._modules.items():
            if (
                other_name == name
                or other_module.kind is not SteeringModuleKind.SAE_DELTA
            ):
                continue
            other_manifest = other_module.sae_manifest
            if other_manifest is None:
                continue
            overlap = seen_sites & set(other_manifest.layers)
            if overlap:
                raise ValueError(
                    f"{prefix}: layers overlap existing SAE module "
                    f"{other_name!r} at site(s) {sorted(overlap)!r}.  "
                    "At most one SAE module may own a (layer_idx, "
                    "hook_point) site; unregister the existing module first."
                )
        if not isinstance(manifest.clampable_features, tuple):
            raise ValueError(
                f"{prefix}: clampable_features must be a tuple of "
                "non-negative integers."
            )
        if not manifest.clampable_features:
            raise ValueError(f"{prefix}: clampable_features must not be empty.")
        seen_features: set[int] = set()
        for feat in manifest.clampable_features:
            validate_steering_index(feat, f"{prefix}: clampable_features entry")
            if feat >= manifest.d_sae:
                raise ValueError(
                    f"{prefix}: clampable_features entry {feat!r} is out "
                    f"of range [0, d_sae={manifest.d_sae})."
                )
            if feat in seen_features:
                raise ValueError(
                    f"{prefix}: clampable_features must not contain "
                    f"duplicates; got feature {feat!r} more than once."
                )
            seen_features.add(feat)

    @staticmethod
    def _validate_sae_activation_params(
        *,
        prefix: str,
        activation: SAEActivation,
        activation_params: dict[str, float],
    ) -> None:
        if not isinstance(activation_params, dict):
            raise ValueError(f"{prefix}: activation_params must be a dict.")
        if activation is SAEActivation.RELU:
            if activation_params:
                raise ValueError(
                    f"{prefix}: activation_params must be empty for relu, "
                    f"got {activation_params!r}."
                )
            return
        if activation is SAEActivation.JUMPRELU:
            threshold = activation_params.get("threshold")
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold))
            ):
                raise ValueError(
                    f"{prefix}: jumprelu activation_params requires a finite "
                    f"'threshold', got {threshold!r}."
                )
            return
        if activation is SAEActivation.TOPK:
            k = activation_params.get("k")
            if (
                isinstance(k, bool)
                or not isinstance(k, (int, float))
                or not math.isfinite(float(k))
                or float(k) < 1.0
                or float(k) != float(int(k))
            ):
                raise ValueError(
                    f"{prefix}: topk activation_params requires a positive "
                    f"integer-valued 'k', got {k!r}."
                )
            return
        raise ValueError(f"{prefix}: unsupported activation {activation!r}.")


def _sae_manifest_to_dict(manifest: SAEModuleManifest | None) -> dict[str, Any]:
    """JSON-safe representation of an :class:`SAEModuleManifest`.

    Used by :meth:`SteeringModuleRegistry.dump_for_broadcast` so the
    manifest crosses the multiprocessing boundary as a plain dict.
    The worker reconstructs the dataclass via
    :func:`sae_manifest_from_dict`.
    """
    assert manifest is not None
    return {
        "d_model": manifest.d_model,
        "d_sae": manifest.d_sae,
        "activation": manifest.activation.value,
        "layers": [list(p) for p in manifest.layers],
        "clampable_features": list(manifest.clampable_features),
        "activation_params": dict(manifest.activation_params),
        "weights_uri": manifest.weights_uri,
    }


def _require_int_field(value: Any, *, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int, got {value!r}.")
    return value


def sae_manifest_from_dict(payload: dict[str, Any]) -> SAEModuleManifest:
    """Reconstruct an :class:`SAEModuleManifest` from a broadcast dict.

    Mirrors :func:`_sae_manifest_to_dict`.  Validates only the fields
    that aren't already covered by
    :meth:`SteeringModuleRegistry._validate_sae_manifest` — the worker
    calls the registry validator after reconstruction.
    """
    layers: list[tuple[int, str]] = []
    for i, entry in enumerate(payload["layers"]):
        try:
            layer_idx, hook_point = entry
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"layers[{i}] must be a (layer_idx, hook_point) pair."
            ) from exc
        layers.append(
            (
                _require_int_field(layer_idx, field_name=f"layers[{i}][0]"),
                str(hook_point),
            )
        )
    return SAEModuleManifest(
        d_model=_require_int_field(payload["d_model"], field_name="d_model"),
        d_sae=_require_int_field(payload["d_sae"], field_name="d_sae"),
        activation=SAEActivation(payload["activation"]),
        layers=tuple(layers),
        clampable_features=tuple(
            _require_int_field(f, field_name=f"clampable_features[{i}]")
            for i, f in enumerate(payload["clampable_features"])
        ),
        activation_params=dict(payload.get("activation_params") or {}),
        weights_uri=payload.get("weights_uri"),
    )


def _convert_layer_keys(
    spec: dict[str, Any] | None,
    *,
    field_name: str,
) -> SteeringVectorSpec | None:
    """Convert JSON string layer keys to int."""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(
            f"Steering module field '{field_name}' must be a JSON object, "
            f"got {type(spec).__name__}"
        )
    if not spec:
        return None
    result: SteeringVectorSpec = {}
    for hook_name, layers in spec.items():
        if not isinstance(layers, dict):
            raise ValueError(
                f"Steering module field '{field_name}'[{hook_name!r}] must be "
                f"a JSON object mapping layer indices to entries, got "
                f"{type(layers).__name__}"
            )
        converted: dict[int, Any] = {}
        for layer_key, entry in layers.items():
            if type(layer_key) is int:
                layer_idx = layer_key
            elif isinstance(layer_key, str):
                try:
                    layer_idx = int(layer_key)
                except ValueError as exc:
                    raise ValueError(
                        f"Steering module field '{field_name}'[{hook_name!r}] has "
                        f"invalid layer index {layer_key!r}; expected an integer"
                    ) from exc
            else:
                raise ValueError(
                    f"Steering module field '{field_name}'[{hook_name!r}] has "
                    f"invalid layer index {layer_key!r}; expected an integer"
                )
            layer_idx = validate_steering_index(
                layer_idx,
                f"Steering module field '{field_name}'[{hook_name!r}] layer key",
            )
            if layer_idx in converted:
                raise ValueError(
                    f"Steering module field '{field_name}'[{hook_name!r}] "
                    f"contains duplicate layer key {layer_idx!r} after "
                    "integer normalization."
                )
            converted[layer_idx] = entry
        if converted:
            result[hook_name] = converted
    return result if result else None
