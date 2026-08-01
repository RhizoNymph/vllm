# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class SAEModuleManifestRequest(BaseModel):
    """Wire-format manifest for a kind=``sae_delta`` module."""

    model_config = ConfigDict(extra="forbid")

    d_model: StrictInt = Field(
        description="Residual-stream dimension the SAE was trained against.",
        gt=0,
    )
    d_sae: StrictInt = Field(
        description="Number of SAE features (encoder/decoder rows).", gt=0
    )
    activation: Literal["relu", "jumprelu", "topk"] = Field(
        description="Encoder activation function the SAE was trained with."
    )
    layers: list[tuple[StrictInt, str]] = Field(
        description="(layer_idx, hook_point) pairs the SAE applies to.",
        min_length=1,
    )
    clampable_features: list[StrictInt] = Field(
        description=(
            "Feature indices that may be clamped at runtime.  "
            "Encoder/decoder rows are loaded only for this subset."
        ),
        min_length=1,
    )
    activation_params: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Activation-specific parameters (e.g. JumpReLU threshold, TopK k)."
        ),
    )
    weights_uri: str | None = Field(
        default=None,
        description=(
            "Local path or URI for SAE weight artifacts.  The register "
            "endpoint loads these weights and broadcasts them with the "
            "manifest."
        ),
    )
    storage_dtype: Literal["auto", "fp8_e4m3"] = Field(
        default="auto",
        description=(
            "Storage dtype for the SAE weight tables: 'auto' keeps the "
            "engine compute dtype; 'fp8_e4m3' stores the large "
            "encoder/decoder matrices as float8_e4m3fn with per-row "
            "fp32 scales (quantized worker-side at attach time)."
        ),
    )

    @field_validator("activation_params", mode="before")
    @classmethod
    def validate_activation_params(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("activation_params must be a dict.")
        for key, param in value.items():
            if not isinstance(key, str):
                raise ValueError("activation_params keys must be strings.")
            if (
                isinstance(param, bool)
                or not isinstance(param, (int, float))
                or not math.isfinite(float(param))
            ):
                raise ValueError(
                    "activation_params values must be finite numbers, "
                    f"got {param!r} for key {key!r}."
                )
        return value


class RegisterSteeringModuleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Unique name for the steering module.",
    )
    kind: Literal["additive", "sae_delta"] = Field(
        default="additive",
        description=(
            "Module kind discriminator.  ``additive`` (default) accepts "
            "the precomputed-vector tier fields; ``sae_delta`` requires "
            "``sae_manifest`` and rejects the additive fields."
        ),
    )
    # Each additive tier accepts either the legacy ``SteeringVectorSpec``
    # shape or the binary-wire ``SteeringVectorSpecPacked`` shape; see
    # ``SetSteeringRequest`` in ``protocol.py`` for the discrimination
    # rationale.  The handler calls ``coerce_steering_spec`` to normalize.
    vectors: dict[str, Any] | None = Field(
        default=None,
        description="Base steering vectors (both phases). Same accepted "
        "shapes as the /v1/steering/set endpoint (legacy SteeringVectorSpec "
        "or binary-wire SteeringHookPacked per hook).  Additive-kind only.",
    )
    prefill_vectors: dict[str, Any] | None = Field(
        default=None,
        description="Prefill-phase steering vectors. Same accepted shapes "
        "as vectors.  Additive-kind only.",
    )
    decode_vectors: dict[str, Any] | None = Field(
        default=None,
        description="Decode-phase steering vectors. Same accepted shapes "
        "as vectors.  Additive-kind only.",
    )
    sae_manifest: SAEModuleManifestRequest | None = Field(
        default=None,
        description=(
            "SAE shape manifest.  Required when ``kind=sae_delta``, "
            "rejected for ``additive``."
        ),
    )
    clamps: dict[str, Any] | None = Field(
        default=None,
        description="Base directional clamps (both phases): {hook: {layer: "
        "[{'vector': [...], 'min': float?, 'max': float?, 'strength': "
        "float=1.0} | {'vector': [...], 'value': c}]}}. Same shape as the "
        "/v1/steering/set clamps field.",
    )
    prefill_clamps: dict[str, Any] | None = Field(
        default=None,
        description="Prefill-phase clamps, concatenated after base.",
    )
    decode_clamps: dict[str, Any] | None = Field(
        default=None,
        description="Decode-phase clamps, concatenated after base.",
    )


class UnregisterSteeringModuleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Name of the steering module to remove.",
    )
