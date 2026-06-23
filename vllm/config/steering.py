# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import ConfigDict, Field

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash


@config(config=ConfigDict(arbitrary_types_allowed=True))
class SteeringConfig:
    """Configuration for per-request activation steering."""

    max_steering_configs: int = Field(default=4, ge=1)
    """Max number of distinct per-request steering configs in a single batch."""

    max_dynamic_steering_configs: int = Field(default=4, ge=0)
    """Size of the dynamic steering row pool: extra steering-table rows
    reserved for runtime per-request overrides driven by dynamic
    steering (sync capture consumers / the steering action queue).
    Separate from ``max_steering_configs`` so dynamic registrations can
    never exhaust rows the scheduler reserved for admitted requests.
    ``0`` disables per-request dynamic overrides. See
    docs/design/dynamic_steering.md §5.2."""

    enable_cross_layer_monitor: bool = Field(default=False)
    """Opt in to the cross-layer in-graph monitor (Phase 2, §8): a probe at
    layer L writes a per-token gate that steering at layers > L reads, same
    forward ("detect at L, gate at layers ≥ L"). When ``True``,
    ``apply_layer_steering`` emits the mutating ``steering_monitor`` op at every
    steered hook (no-op unless the manager activated a probe there) and the
    same-hook fused gate is bypassed. When ``False`` (default) the monitor is
    the same-hook non-mutating fused gate only. This flips the compiled graph
    topology, so it is part of ``compute_hash``. See
    docs/design/dynamic_steering.md §8."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list = []
        factors.append(self.max_steering_configs)
        # Dynamic pool size changes the steering-table buffer shape,
        # which is baked into compiled graphs.
        factors.append(self.max_dynamic_steering_configs)
        # Cross-layer monitor changes the compiled graph topology (emits the
        # mutating steering_monitor op at every steered hook).
        factors.append(self.enable_cross_layer_monitor)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str
