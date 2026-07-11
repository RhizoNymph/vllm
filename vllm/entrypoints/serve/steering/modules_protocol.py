# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RegisterSteeringModuleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Unique name for the steering module.",
    )
    # Each tier accepts either the legacy ``SteeringVectorSpec`` shape or
    # the binary-wire ``SteeringVectorSpecPacked`` shape; see
    # ``SetSteeringRequest`` in ``protocol.py`` for the discrimination
    # rationale.  The handler calls ``coerce_steering_spec`` to normalize.
    vectors: dict[str, Any] | None = Field(
        default=None,
        description="Base steering vectors (both phases). Same accepted "
        "shapes as the /v1/steering/set endpoint (legacy SteeringVectorSpec "
        "or binary-wire SteeringHookPacked per hook).",
    )
    prefill_vectors: dict[str, Any] | None = Field(
        default=None,
        description="Prefill-phase steering vectors. Same accepted shapes as vectors.",
    )
    decode_vectors: dict[str, Any] | None = Field(
        default=None,
        description="Decode-phase steering vectors. Same accepted shapes as vectors.",
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
