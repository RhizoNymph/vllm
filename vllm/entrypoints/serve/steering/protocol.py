# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SetSteeringRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Each tier accepts either:
    #   - the legacy ``SteeringVectorSpec`` shape
    #     ``{hook: {layer_idx: list[float] | {"vector": [...], "scale": float}}}``
    #   - the binary-wire ``SteeringVectorSpecPacked`` shape
    #     (see ``vllm.config.steering_types.SteeringHookPacked``)
    # Discriminated at the inner-dict level by the presence of ``data`` and
    # ``dtype`` marker keys.  Pydantic v2 cannot auto-disambiguate the union
    # of two TypedDicts whose discriminator is nested, so the field type is
    # widened to ``dict[str, Any]`` and the handler calls
    # ``coerce_steering_spec`` to normalize.
    vectors: dict[str, Any] | None = Field(
        default=None,
        description="Base steering vectors applied to both prefill and "
        "decode phases. Keyed by hook point name (pre_attn, post_attn, "
        "post_block). Each hook's value is either a legacy layer map "
        '({layer_idx: list[float] | {"vector": [...], "scale": float}}) '
        "or a binary-wire SteeringHookPacked blob (base64-encoded "
        "(num_layers, hidden_size) buffer + layer_indices + dtype/shape, "
        "optional per-row scales).",
    )
    prefill_vectors: dict[str, Any] | None = Field(
        default=None,
        description="Phase-specific steering vectors added to base during "
        "prefill only. Same accepted shapes as vectors.",
    )
    decode_vectors: dict[str, Any] | None = Field(
        default=None,
        description="Phase-specific steering vectors added to base during "
        "decode only. Same accepted shapes as vectors.",
    )
    # Clamp tiers: {hook: {layer_idx: [ClampEntry, ...]}} where each entry
    # is {"vector": [...], "min": float|None, "max": float|None,
    # "strength": float = 1.0} or the sugar {"vector": [...], "value": c}.
    # Directions are unit-normalized server-side; layer keys arrive as
    # strings over JSON and are int-coerced by the handler.
    clamps: dict[str, Any] | None = Field(
        default=None,
        description="Base directional clamps applied to both prefill and "
        "decode phases. Keyed by hook point name, then layer index; each "
        "value is a LIST of clamp entries ({'vector': [...], 'min': float?, "
        "'max': float?, 'strength': float=1.0} or {'vector': [...], "
        "'value': c} to pin). Each entry constrains the hidden state's "
        "projection along its unit-normalized direction to [min, max].",
    )
    prefill_clamps: dict[str, Any] | None = Field(
        default=None,
        description="Phase-specific clamps concatenated after base during "
        "prefill only. Same shape as clamps.",
    )
    decode_clamps: dict[str, Any] | None = Field(
        default=None,
        description="Phase-specific clamps concatenated after base during "
        "decode only. Same shape as clamps.",
    )
    replace: bool = Field(
        default=False,
        description="When True, clears all existing steering vectors "
        "and clamps before applying the new ones, making the operation an "
        "atomic replacement. When False (default), only the specified "
        "layers are updated and other layers keep their current state.",
    )


class SetSAEGlobalClampsRequest(BaseModel):
    """Request body for ``POST /v1/steering/sae/set``.

    Each specs field carries JSON-shape SAE clamp specs (the same
    shape accepted by the per-request ``sae_clamp_specs`` sampling
    field); coercion into ``SAEClampSpec`` happens on the workers via
    ``coerce_sae_clamp_specs``, so the fields stay wide here.
    """

    model_config = ConfigDict(extra="forbid")

    prefill_specs: list[Any] | None = Field(
        default=None,
        description="JSON-shape SAE clamp specs installed into the "
        "global prefill tier (applied to every prefill token). Each "
        "entry is {module_name, phase?, gated?, clamps: {hook: {layer: "
        "[{feature_idx, kind, value, only_if_active?}]}}}. None "
        "leaves the existing prefill globals untouched.",
    )
    decode_specs: list[Any] | None = Field(
        default=None,
        description="JSON-shape SAE clamp specs installed into the "
        "global decode tier (applied to every decode token). Same "
        "shape as prefill_specs. None leaves the existing decode "
        "globals untouched.",
    )
    replace: bool = Field(
        default=False,
        description="When True, clears all existing global SAE clamps "
        "before applying the new ones, making the operation an atomic "
        "replacement of the global configuration. When False "
        "(default), the new specs are appended to the existing tiers.",
    )
