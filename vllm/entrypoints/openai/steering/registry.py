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

from vllm.config.sae_steering_types import SAEActivation, SteeringModuleKind
from vllm.config.steering_types import (
    SteeringVectorSpec,
    merge_steering_specs,
    normalize_layer_entry,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import VALID_HOOK_POINT_NAMES

logger = init_logger(__name__)


@dataclass
class SAEModuleManifest:
    """Shape description for an SAE-kind named steering module.

    Phase-0 stores the manifest only — the runtime accepts requests
    that reference SAE modules but the worker raises
    ``NotImplementedError`` if asked to apply them.  The full loader
    (weights, encoder/decoder GEMMs, kernel) lands in Phase-1+.

    The manifest is the authoritative description of the SAE that
    the kernel will need: model and dictionary dimensions, activation
    function, the set of (layer, hook) sites it covers, and the set
    of feature indices that may be clamped at runtime (so encoder
    rows can be restricted to that set).
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
        elif kind is SteeringModuleKind.SAE_DELTA:
            if vectors or prefill_vectors or decode_vectors:
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
        elif kind is SteeringModuleKind.SAE_FULL_RECONSTRUCTION:
            if vectors or prefill_vectors or decode_vectors:
                raise ValueError(
                    f"Steering module '{name}': additive vector fields are "
                    "not valid for kind=SAE_FULL_RECONSTRUCTION."
                )
            if sae_manifest is None:
                raise ValueError(
                    f"Steering module '{name}': sae_manifest is required "
                    "for kind=SAE_FULL_RECONSTRUCTION."
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
        payload before coercion); only the ``module_name`` field is
        read so both shapes work.
        """
        if not specs:
            return None
        for i, spec in enumerate(specs):
            if isinstance(spec, dict):
                module_name = spec.get("module_name")
            else:
                module_name = getattr(spec, "module_name", None)
            if not isinstance(module_name, str) or not module_name:
                return (
                    f"sae_clamp_specs[{i}] is missing a non-empty 'module_name' field."
                )
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

    def list_modules(self) -> list[str]:
        """Return sorted list of registered module names."""
        return sorted(self._modules.keys())

    def dump_for_broadcast(self) -> dict[str, dict[str, Any]]:
        """Return a JSON-safe view of every registered module.

        Used by the API server to broadcast the full registry to workers
        via ``collective_rpc`` so every worker holds an identical
        ``_steering_module_registry``.  The returned mapping is keyed by
        module name; each value is a ``dict`` carrying a ``kind``
        discriminator plus the kind-specific payload:

        * Additive: ``vectors``, ``prefill_vectors``, ``decode_vectors``
        * SAE delta: ``sae_manifest`` (a JSON-safe dict)

        All structures are plain Python collections suitable for
        cloudpickle serialization across the engine multiprocess
        boundary.

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
                payload["sae_manifest"] = _sae_manifest_to_dict(module.sae_manifest)
            out[name] = payload
        return out

    async def load_from_file(self, name: str, path: str) -> None:
        """Load a steering module from a JSON file and register it.

        Expected JSON format::

            {
                "vectors": {"post_mlp": {"14": [0.1, ...]}},
                "prefill_vectors": {"pre_attn": {"14": [0.3, ...]}},
                "decode_vectors": null,
            }

        JSON uses string keys for layer indices; they are converted to int.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Steering module file not found: {path}")

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
        SteeringVectorSpec | None,
        SteeringVectorSpec | None,
        SteeringVectorSpec | None,
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
            if not isinstance(entry["scale"], (int, float)):
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
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{value_prefix}[{i}] must be a finite float, "
                    f"got {type(value).__name__}."
                )
            if not math.isfinite(value):
                raise ValueError(f"{value_prefix}[{i}] must be finite, got {value}.")

    def _validate_layer_index(self, name: str, layer_idx: int) -> None:
        """Validate layer indices against the loaded model when available."""
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

        Phase-0 only stores the manifest, but we still want bad shapes to
        fail loudly at registration time rather than at request time.
        """
        prefix = f"SAE module {name!r}"
        if not isinstance(manifest.d_model, int) or manifest.d_model <= 0:
            raise ValueError(
                f"{prefix}: d_model must be a positive int, got {manifest.d_model!r}."
            )
        if not isinstance(manifest.d_sae, int) or manifest.d_sae <= 0:
            raise ValueError(
                f"{prefix}: d_sae must be a positive int, got {manifest.d_sae!r}."
            )
        if not isinstance(manifest.activation, SAEActivation):
            raise ValueError(
                f"{prefix}: activation must be an SAEActivation enum, "
                f"got {type(manifest.activation).__name__}."
            )
        if not isinstance(manifest.layers, tuple) or not manifest.layers:
            raise ValueError(
                f"{prefix}: layers must be a non-empty tuple of "
                "(layer_idx, hook_point) pairs."
            )
        for entry in manifest.layers:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], int)
                or not isinstance(entry[1], str)
            ):
                raise ValueError(
                    f"{prefix}: layer entries must be (int, str) tuples, got {entry!r}."
                )
            layer_idx, hook_point = entry
            self._validate_layer_index(name=name, layer_idx=layer_idx)
            if hook_point not in VALID_HOOK_POINT_NAMES:
                raise ValueError(
                    f"{prefix}: unknown hook point {hook_point!r}.  "
                    f"Valid: {sorted(VALID_HOOK_POINT_NAMES)}."
                )
        if not isinstance(manifest.clampable_features, tuple):
            raise ValueError(
                f"{prefix}: clampable_features must be a tuple of "
                "non-negative integers."
            )
        for feat in manifest.clampable_features:
            if not isinstance(feat, int) or feat < 0 or feat >= manifest.d_sae:
                raise ValueError(
                    f"{prefix}: clampable_features entry {feat!r} is out "
                    f"of range [0, d_sae={manifest.d_sae})."
                )


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


def sae_manifest_from_dict(payload: dict[str, Any]) -> SAEModuleManifest:
    """Reconstruct an :class:`SAEModuleManifest` from a broadcast dict.

    Mirrors :func:`_sae_manifest_to_dict`.  Validates only the fields
    that aren't already covered by
    :meth:`SteeringModuleRegistry._validate_sae_manifest` — the worker
    calls the registry validator after reconstruction.
    """
    return SAEModuleManifest(
        d_model=int(payload["d_model"]),
        d_sae=int(payload["d_sae"]),
        activation=SAEActivation(payload["activation"]),
        layers=tuple((int(li), str(hp)) for li, hp in payload["layers"]),
        clampable_features=tuple(int(f) for f in payload["clampable_features"]),
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
            try:
                converted[int(layer_key)] = entry
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Steering module field '{field_name}'[{hook_name!r}] has "
                    f"invalid layer index {layer_key!r}; expected an integer"
                ) from exc
        if converted:
            result[hook_name] = converted
    return result if result else None
