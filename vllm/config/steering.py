# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from pydantic import ConfigDict, Field

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

if TYPE_CHECKING:
    from vllm.config.vllm import VllmConfig


@dataclass(frozen=True)
class SAEModuleTopology:
    """Graph-shape summary of one startup-declared SAE steering module.

    Distilled from the module dir's ``manifest.json`` at engine-config
    build time (no tensor I/O) and carried on
    ``SteeringConfig.sae_module_topology`` so workers can pre-allocate
    the module's buffers in ``_init_steering_state`` — before
    torch.compile tracing / CUDA-graph capture. Holds only what
    determines buffer shapes and per-slot trace-time constants; feature
    *ids* and weights are data, delivered later by the frontend
    broadcast into the pre-allocated buffers.
    """

    name: str
    kind: str
    """``"sae_delta"`` or ``"sae_full_reconstruction"``."""
    layers: tuple[tuple[int, str], ...]
    """Canonically sorted ``(layer_idx, hook_point)`` sites."""
    d_model: int
    d_sae: int
    n_clamp: int
    """``len(clampable_features)`` — buffer width only, not the ids."""
    activation: str
    """``SAEActivation`` value; baked into the graph per slot."""
    activation_params: dict[str, float] = field(default_factory=dict)


def is_steering_topology_frozen(vllm_config: "VllmConfig | None") -> bool:
    """Whether SAE buffer topology is fixed after model load.

    True when a compiled artifact or captured CUDA graph will hold
    references to the steering buffers, so slot/site creation and
    deletion after ``load_model`` is unsafe — only in-place data
    updates (weight refresh, deactivation) are allowed. Eager engines
    return False and keep fully dynamic registration.
    """
    if vllm_config is None:
        return False
    # Lazy imports: this module is imported during config-package init,
    # before vllm.config.compilation is guaranteed importable.
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        return False
    if compilation_config.mode == CompilationMode.VLLM_COMPILE:
        return True
    model_config = getattr(vllm_config, "model_config", None)
    enforce_eager = bool(getattr(model_config, "enforce_eager", False))
    return not enforce_eager and compilation_config.cudagraph_mode != CUDAGraphMode.NONE


def sae_topology_mismatch(
    topo: SAEModuleTopology,
    *,
    kind: str,
    layers: tuple[tuple[int, str], ...],
    d_model: int,
    d_sae: int,
    n_clamp: int,
    activation: str,
    activation_params: dict[str, float],
) -> str | None:
    """Compare a declared topology against an incoming registration.

    Returns a human-readable description of the first mismatch, or
    ``None`` when the registration matches the declared graph shape.
    Single source of truth for both the worker frozen-topology check
    and the frontend register-endpoint precheck, so the two can never
    diverge.
    """
    if kind != topo.kind:
        return f"kind {kind!r} != declared {topo.kind!r}"
    if tuple(sorted(layers)) != topo.layers:
        return f"sites {sorted(layers)} != declared {list(topo.layers)}"
    if d_model != topo.d_model:
        return f"d_model {d_model} != declared {topo.d_model}"
    if d_sae != topo.d_sae:
        return f"d_sae {d_sae} != declared {topo.d_sae}"
    if n_clamp != topo.n_clamp:
        return f"n_clamp {n_clamp} != declared {topo.n_clamp}"
    if activation != topo.activation:
        return f"activation {activation!r} != declared {topo.activation!r}"
    if dict(activation_params) != topo.activation_params:
        return (
            f"activation_params {dict(activation_params)} != declared "
            f"{topo.activation_params}"
        )
    return None


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

    sae_module_topology: list[SAEModuleTopology] = Field(default_factory=list)
    """Graph-shape summaries of startup-declared SAE modules, distilled
    from ``--steering-modules`` manifests at engine-config build time.
    Workers pre-allocate each module's zero-filled buffers in
    ``_init_steering_state`` — before compile/capture — so SAE steering
    survives compiled serving; the frontend broadcast then only fills
    weights in place. On a compiled engine this set (plus spare slots)
    IS the SAE topology: registrations that don't fit are rejected."""

    sae_spare_slot_sites: list[str] = Field(default_factory=list)
    """``layer:hook`` sites that reserve spare SAE *delta* buffer slots
    for modules not known at startup, so they can hot-register on a
    compiled engine. Spare slots are baked as JumpReLU (per-feature
    threshold 0 ⇒ exact ReLU, so both activations land via data alone;
    TopK modules cannot claim spares). Each spare costs
    ``~2 × sae_spare_slot_features × d_model`` in compute dtype per
    site per layer — explicit opt-in."""

    sae_spare_slots_per_site: int = Field(default=1, ge=1)
    """Spare delta slots reserved at each ``sae_spare_slot_sites``
    entry."""

    sae_spare_slot_features: int = Field(default=0, ge=0)
    """Clampable-feature capacity (``n_clamp``) reserved per spare
    slot. A claiming module needs ``n_clamp <= sae_spare_slot_features``
    (smaller modules are zero-padded). ``0`` disables spare slots even
    when sites are listed."""

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
        # SAE topology: every field determines pre-allocated buffer shapes
        # or per-slot trace-time constants baked into the compiled graph.
        # Canonical sorted-JSON so dict ordering can't shift the hash;
        # feature ids / weights are data and deliberately not factors.
        factors.append(
            json.dumps(
                [
                    asdict(t)
                    for t in sorted(self.sae_module_topology, key=lambda t: t.name)
                ],
                sort_keys=True,
            )
        )
        # Spare slots add zero-filled delta slots (ops) at their sites.
        factors.append(sorted(self.sae_spare_slot_sites))
        factors.append(self.sae_spare_slots_per_site)
        factors.append(self.sae_spare_slot_features)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str
