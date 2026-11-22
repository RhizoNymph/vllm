# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-aware shape of the steering sites: per-hook row width + layer count.

Steering specs are validated model-blind at ingestion
(``SamplingParams.__post_init__``, the registry coercers): hook names are
checked against every :class:`SteeringHookPoint`, rows for finiteness.
What a spec must additionally satisfy depends on the served model:

- the hook must be one the model steers (``mhc_*`` stream hooks exist only
  on hyper-connection models; DeepSeek-V4 has no ``post_block``);
- each row must be as wide as *its hook's* table — ``hidden_size`` for the
  single-stream hooks, ``hc * hidden_size`` for the multi-stream residual
  hooks of hyper-connection models;
- the layer index must be below ``num_hidden_layers`` (a row for a layer
  that does not exist is otherwise dropped silently by the workers).

:class:`SteeringSiteGeometry` carries exactly that, resolved once per
process from the model config by :func:`resolve_steering_site_geometry`
(which reuses the capture framework's hook schema, so steering and capture
agree on every hook's width), and every seam that admits client specs —
inline packing in ``AsyncLLM`` / ``LLM``, the engine input processor, the
named-module registry — validates against it. Failures raise
:class:`vllm.exceptions.SteeringSiteValidationError` (HTTP 400).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.exceptions import SteeringSiteValidationError

if TYPE_CHECKING:
    from vllm.config.steering_types import SteeringClamps
    from vllm.v1.capture.types import HookSchema

# Separator of the ``"layer@pass"`` key form looped models accept.
_PASS_SEPARATOR = "@"


def _steerable_hook_names() -> frozenset[str]:
    # Lazy: the hook-point enum lives with the torch-side steering layers.
    from vllm.model_executor.layers.steering import VALID_HOOK_POINT_NAMES

    return VALID_HOOK_POINT_NAMES


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


@dataclass(frozen=True)
class SteeringSiteGeometry:
    """The served model's steering sites.

    Attributes:
        hook_widths: Read-only ``{hook_name: row_width}`` for every hook the
            model steers. A hook absent here is not steerable on the model.
        num_hidden_layers: Decoder layer count; valid layer indices are
            ``[0, num_hidden_layers)``.

    """

    hook_widths: Mapping[str, int]
    num_hidden_layers: int

    def __post_init__(self) -> None:
        widths = dict(self.hook_widths)
        if not widths:
            raise ValueError("SteeringSiteGeometry needs at least one hook")
        unknown = set(widths) - _steerable_hook_names()
        if unknown:
            raise ValueError(
                f"SteeringSiteGeometry: {sorted(unknown)} are not steering hooks"
            )
        for hook, width in widths.items():
            if type(width) is not int or width <= 0:
                raise ValueError(
                    f"SteeringSiteGeometry: width of {hook!r} must be a "
                    f"positive int, got {width!r}"
                )
        if type(self.num_hidden_layers) is not int or self.num_hidden_layers <= 0:
            raise ValueError(
                "SteeringSiteGeometry: num_hidden_layers must be a positive "
                f"int, got {self.num_hidden_layers!r}"
            )
        object.__setattr__(self, "hook_widths", MappingProxyType(widths))

    @classmethod
    def standard(
        cls, hidden_size: int, *, num_hidden_layers: int
    ) -> SteeringSiteGeometry:
        """A standard decoder: the five residual hooks, all ``hidden_size``."""
        from vllm.v1.capture.types import STANDARD_HOOKS

        return cls(
            hook_widths={hook: hidden_size for hook in STANDARD_HOOKS},
            num_hidden_layers=num_hidden_layers,
        )

    @classmethod
    def from_hook_schema(
        cls, schema: Mapping[str, HookSchema], *, num_hidden_layers: int
    ) -> SteeringSiteGeometry:
        """The steerable subset of a capture hook schema.

        Capture-only hooks (the mHC mixing coefficients) are dropped.
        """
        steerable = _steerable_hook_names()
        return cls(
            hook_widths={
                hook: s.width for hook, s in schema.items() if hook in steerable
            },
            num_hidden_layers=num_hidden_layers,
        )

    # ------------------------------------------------------------------
    # Per-site checks
    # ------------------------------------------------------------------

    def width_of(self, hook: str, *, field_name: str) -> int:
        """Row width of *hook*; reject a hook this model does not steer."""
        width = self.hook_widths.get(hook)
        if width is None:
            raise SteeringSiteValidationError(
                f"{field_name}[{hook!r}]: {hook!r} is not a steering hook of "
                f"this model. Steering hooks: {sorted(self.hook_widths)}",
                parameter=field_name,
            )
        return width

    def layer_of(self, layer_key: object, *, hook: str, field_name: str) -> int:
        """Range-check the layer part of *layer_key* and return it.

        Keys are ints, their string form, or ``"layer@pass"`` (looped
        models). Only the layer part is checked here; pass validity belongs
        to the loop-layer expansion.
        """
        layer: int | None = None
        if _is_int(layer_key):
            layer = int(layer_key)  # type: ignore[arg-type]
        elif isinstance(layer_key, str):
            layer_text = layer_key.strip().partition(_PASS_SEPARATOR)[0]
            if layer_text.isdigit():
                layer = int(layer_text)
        if layer is None or layer < 0:
            raise SteeringSiteValidationError(
                f"{field_name}[{hook!r}]: layer key {layer_key!r} must be a "
                "non-negative layer index (or 'layer@pass' on a looped model)",
                parameter=field_name,
            )
        if layer >= self.num_hidden_layers:
            raise SteeringSiteValidationError(
                f"{field_name}[{hook!r}][{layer_key}]: layer index {layer} out "
                f"of range for num_hidden_layers={self.num_hidden_layers} "
                f"(valid: 0..{self.num_hidden_layers - 1})",
                parameter=field_name,
            )
        return layer

    # ------------------------------------------------------------------
    # Whole-spec checks
    # ------------------------------------------------------------------

    def validate_vectors(
        self, spec: Mapping[str, Any] | None, *, field_name: str
    ) -> None:
        """Validate a ``{hook: {layer: entry}}`` additive spec.

        Entries may be bare lists, ``{"vector", "scale"}`` dicts or
        ndarrays (the packed-form decode).
        """
        if not spec:
            return
        from vllm.config.steering_types import normalize_layer_entry

        for hook, layers in spec.items():
            width = self.width_of(hook, field_name=field_name)
            for layer_key, entry in layers.items():
                self.layer_of(layer_key, hook=hook, field_name=field_name)
                vector, _ = normalize_layer_entry(entry)
                if len(vector) != width:
                    raise SteeringSiteValidationError(
                        f"{field_name}[{hook!r}][{layer_key}]: vector width "
                        f"{len(vector)} != expected steering row width {width} "
                        f"for hook {hook!r}",
                        parameter=field_name,
                    )

    def validate_clamps(
        self, clamps: SteeringClamps | None, *, field_name: str
    ) -> None:
        """Validate a canonical clamp spec: hooks, layers, direction widths."""
        if clamps is None:
            return
        for hook, table in clamps.hooks.items():
            width = self.width_of(hook, field_name=field_name)
            for layer in table.site_counts():
                self.layer_of(layer, hook=hook, field_name=field_name)
            if table.width != width:
                raise SteeringSiteValidationError(
                    f"{field_name}[{hook!r}]: clamp direction width "
                    f"{table.width} != expected steering row width {width} "
                    f"for hook {hook!r}",
                    parameter=field_name,
                )


def resolve_steering_site_geometry(model_config: Any) -> SteeringSiteGeometry:
    """Resolve the steering sites of the model *model_config* describes.

    Per-hook widths come from the capture hook schema
    (:func:`vllm.v1.capture.types.build_model_hook_schema`), the single
    source of hook geometry per model family; the layer bound from
    :func:`vllm.config.loop_layers.num_layer_keys` (all pipeline stages;
    on a looped model, every pass's layer keys).
    """
    from vllm.config.loop_layers import num_layer_keys
    from vllm.v1.capture.types import build_model_hook_schema

    schema = build_model_hook_schema(
        model_config.get_hidden_size(),
        model_config.dtype,
        getattr(model_config, "hf_config", None),
        getattr(model_config, "hf_text_config", None),
    )
    return SteeringSiteGeometry.from_hook_schema(
        schema, num_hidden_layers=num_layer_keys(model_config)
    )
