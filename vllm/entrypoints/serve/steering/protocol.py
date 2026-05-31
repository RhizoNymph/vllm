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
        "post_mlp). Each hook's value is either a legacy layer map "
        "({layer_idx: list[float] | {\"vector\": [...], \"scale\": float}}) "
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
    replace: bool = Field(
        default=False,
        description="When True, clears all existing steering vectors "
        "before applying the new ones, making the operation an atomic "
        "replacement. When False (default), only the specified layers "
        "are updated and other layers keep their current state.",
    )
