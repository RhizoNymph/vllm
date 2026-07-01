# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RegisterVectorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique name for the probe/steer vector.")
    kind: Literal["probe", "steer"] = Field(
        description="Namespace: 'probe' (referenced by a gate's when.probe) or "
        "'steer' (referenced by a gate's apply.steer)."
    )
    packed: dict[str, Any] = Field(
        description="The vector(s) as a {hook: SteeringHookPacked} packed spec "
        "(base64). A 'probe' must resolve to exactly one (hook, layer)."
    )


class UnregisterVectorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Name of the vector to remove.")
    kind: Literal["probe", "steer"] = Field(description="Namespace to remove from.")
