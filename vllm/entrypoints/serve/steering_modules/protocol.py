# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import BaseModel, ConfigDict, Field

from vllm.config.steering_types import SteeringVectorSpec


class RegisterSteeringModuleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Unique name for the steering module.",
    )
    vectors: SteeringVectorSpec | None = Field(
        default=None,
        description="Base steering vectors (both phases). Same format as "
        "the /v1/steering/set endpoint.",
    )
    prefill_vectors: SteeringVectorSpec | None = Field(
        default=None,
        description="Prefill-phase steering vectors.",
    )
    decode_vectors: SteeringVectorSpec | None = Field(
        default=None,
        description="Decode-phase steering vectors.",
    )


class UnregisterSteeringModuleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Name of the steering module to remove.",
    )
