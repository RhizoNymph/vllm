# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-body schema for per-request MLP neuron clamps.

Shared by the chat-completion and completion request models. Malformed
entries fail pydantic validation (HTTP 400) before a ``SamplingParams`` is
built; cross-entry checks (overlapping duplicates) and model range checks
raise :class:`vllm.config.neuron_clamp_types.NeuronClampValidationError`
(also HTTP 400) downstream.
"""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from vllm.config.neuron_clamp_types import NeuronClampMode, NeuronClampPhase


class NeuronClampSpec(BaseModel):
    """One clamp on the SiLU-gated MLP activation (``down_proj`` input)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    layer: Annotated[int, Field(ge=0, description="Decoder layer index.")]
    neuron: Annotated[
        int, Field(ge=0, description="Global MLP intermediate (neuron) index.")
    ]
    mode: NeuronClampMode = Field(
        description="'set' replaces the activation with value, 'add' adds "
        "value, 'scale' multiplies by value."
    )
    value: Annotated[float, Field(allow_inf_nan=False)]
    phase: NeuronClampPhase = Field(
        default="both",
        description="Apply during 'prefill', 'decode', or 'both'.",
    )

    def to_wire(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "neuron": self.neuron,
            "mode": self.mode,
            "value": float(self.value),
            "phase": self.phase,
        }


def neuron_clamps_to_wire(
    specs: list[NeuronClampSpec] | None,
) -> list[dict[str, Any]] | None:
    """Convert request-body clamps to the ``SamplingParams`` wire form."""
    if not specs:
        return None
    return [s.to_wire() for s in specs]
