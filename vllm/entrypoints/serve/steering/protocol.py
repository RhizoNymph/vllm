# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vllm.config.steering_types import SteeringVectorSpec


class SetSteeringRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vectors: SteeringVectorSpec | None = Field(
        default=None,
        description="Base steering vectors applied to both prefill and "
        "decode phases. Keyed by hook point name (pre_attn, post_attn, "
        "post_mlp), then layer index. Values "
        "are either bare lists (scale=1.0) or "
        '{"vector": [...], "scale": float}.',
    )
    prefill_vectors: SteeringVectorSpec | None = Field(
        default=None,
        description="Phase-specific steering vectors added to base during "
        "prefill only. Same format as vectors.",
    )
    decode_vectors: SteeringVectorSpec | None = Field(
        default=None,
        description="Phase-specific steering vectors added to base during "
        "decode only. Same format as vectors.",
    )
    replace: bool = Field(
        default=False,
        description="When True, clears all existing steering vectors "
        "before applying the new ones, making the operation an atomic "
        "replacement. When False (default), only the specified layers "
        "are updated and other layers keep their current state.",
    )
    target: Literal["main", "draft"] | None = Field(
        default=None,
        description="Which model the vectors apply to. 'main' targets the "
        "primary model only; 'draft' targets the speculative-decoding "
        "draft model only; omitted means apply to the main model (and — "
        "in a future release — to the draft model as well via "
        "tags-along). 'draft' is currently returned as HTTP 501; see "
        "the Speculative decoding section of docs/features/steering.md.",
    )


class ClearSteeringRequest(BaseModel):
    """Optional body for ``POST /v1/steering/clear``.

    Passing no body (or an empty JSON object) clears the main model's
    steering state, matching legacy behavior. Pass ``{"target": "main"}``
    or ``{"target": "draft"}`` to scope the clear. ``"draft"`` returns
    HTTP 501 in this release.
    """

    model_config = ConfigDict(extra="forbid")

    target: Literal["main", "draft"] | None = Field(
        default=None,
        description="See `SetSteeringRequest.target`.",
    )
