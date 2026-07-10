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
        "(base64). A 'probe' must resolve to exactly one (hook, layer). "
        "Registration is broadcast to every worker so a {kind:'name'} gate "
        "resolves worker-side; a name is also the only way to express a "
        "rest_of_conversation add gate (which persists server-side by "
        "reference to this name, not by inlining the client's bytes)."
    )


class UnregisterVectorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Name of the vector to remove.")
    kind: Literal["probe", "steer"] = Field(description="Namespace to remove from.")
