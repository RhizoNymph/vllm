# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import ConfigDict, Field

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash


@config(config=ConfigDict(arbitrary_types_allowed=True))
class SteeringConfig:
    """Configuration for per-request activation steering."""

    max_steering_configs: int = Field(default=32, ge=1)
    """Max number of distinct per-request steering configs in a single batch.

    This bounds the worker's steering table rows; the scheduler reserves a row
    per distinct ``(config_hash, phase)`` and applies backpressure (holds
    requests in the waiting queue) once the pool is full, so a request that
    asked for steering is never silently run unsteered. Configs are cheap
    (one table row each), so this defaults generously."""

    max_dynamic_steering_configs: int = Field(default=4, ge=0)
    """Size of the dynamic steering row pool: extra steering-table rows
    reserved for runtime per-request overrides driven by dynamic
    steering (sync capture consumers / the steering action queue).
    Separate from ``max_steering_configs`` so dynamic registrations can
    never exhaust rows the scheduler reserved for admitted requests.
    ``0`` disables per-request dynamic overrides. See
    docs/design/dynamic_steering.md §5.2."""

    max_clamp_directions: int = Field(default=4, ge=0)
    """Max directional-clamp directions per (steering row, hook site) — the K
    dimension of the per-layer clamp buffers. Each steering config (global or
    per-request) may carry up to K clamp entries per (hook, layer); requests
    exceeding K are rejected at resolution. ``0`` disables clamping entirely
    (no clamp buffers are attached and the apply path constant-folds out).

    Memory: the dirs buffer is ``rows x K x hidden`` in the model dtype per
    (hook, layer) — roughly 78 MB total at defaults on a gemma-3-4b-class
    model (39 rows x K=4 x 2560 x 2 B x 3 hooks x 34 layers). Changes
    captured buffer shapes, so it is part of ``compute_hash``."""

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

    enable_row_monitor: bool = Field(default=False)
    """Opt in to the PER-ROW (per-request) in-graph monitor: each steering
    table row carries its own probe + ``[threshold, sharpness]``, so concurrent
    requests at a site can be gated by different probe conditions (true
    per-request same-step steering). When ``True``, the per-(layer, hook) probe
    table is sized to ``(max rows, hidden)`` (instead of a ``(1, 1)`` dummy),
    which is baked into the captured graph buffers, so it is part of
    ``compute_hash``. When ``False`` (default) only the global monitor exists.
    See docs/design/dynamic_steering.md."""

    enable_declarative_gates: bool = Field(default=True)
    """Enable the built-in declarative per-request steering consumer: a client
    can attach its own conditional steering to a request (a nested list of
    ``when × scope × apply`` gates in ``RequestMetadata.steering``) without a
    server-registered consumer. When enabled (and steering is on, pipeline
    parallelism is 1) the consumer is auto-registered and ``enable_row_monitor``
    is turned on so ``this_token`` probe gates run in-graph. Does not itself
    change the compiled graph (the implied row-monitor flag does), so it is not a
    ``compute_hash`` factor. See docs/design/dynamic_steering.md."""

    declarative_probe_sites: list[str] = Field(default_factory=list)
    """``layer:hook`` sites whose residual the declarative consumer captures so
    host-evaluated probes (non-``this_token`` scopes with ``when=probe``) can
    read them. ``this_token`` probes are computed in-graph and need no capture.
    Empty ⇒ a single default site is captured (see the consumer)."""

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
        # Per-row monitor changes the per-row probe-table buffer shape, which
        # is baked into captured graphs.
        factors.append(self.enable_row_monitor)
        # Clamp direction count changes the clamp buffer shapes (and 0 vs >0
        # changes whether the clamp ops are emitted at all).
        factors.append(self.max_clamp_directions)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str
