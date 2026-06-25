# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from vllm.config.steering_types import SteeringVectorSpec


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
    vectors: SteeringVectorSpec | None = Field(
        default=None,
        description="Base steering vectors (both phases). Same format as "
        "the /v1/steering/set endpoint.  Additive-kind only.",
    )
    prefill_vectors: SteeringVectorSpec | None = Field(
        default=None,
        description="Prefill-phase steering vectors.  Additive-kind only.",
    )
    decode_vectors: SteeringVectorSpec | None = Field(
        default=None,
        description="Decode-phase steering vectors.  Additive-kind only.",
    )
    sae_manifest: SAEModuleManifestRequest | None = Field(
        default=None,
        description=(
            "SAE shape manifest.  Required when ``kind=sae_delta``, "
            "rejected for ``additive``."
        ),
    )


class UnregisterSteeringModuleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Name of the steering module to remove.",
    )
