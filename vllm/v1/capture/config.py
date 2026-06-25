# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration dataclasses for the capture-consumer framework.

``CaptureConsumersConfig`` is wired into ``VllmConfig`` and read by
the runner at engine init.  It enumerates the ordered set of consumer
instances to construct, each identified by a registry ``name`` plus an
optional ``instance_name`` disambiguator and opaque ``params`` dict.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CaptureConsumerSpec:
    """One consumer entry from config."""

    name: str
    """Entry-point name (e.g., ``"filesystem"``)."""

    instance_name: str | None = None
    """Optional unique alias for this consumer instance."""

    params: dict[str, Any] = field(default_factory=dict)
    """Arbitrary key-value parameters forwarded to the consumer factory."""


@dataclass
class CaptureConsumersConfig:
    """Top-level capture-consumers configuration.

    ``consumers`` lists config-driven entries (entry-point name + params).
    ``instances`` carries pre-constructed ``CaptureConsumer`` instances
    passed directly to ``LLM(capture_consumers=[...])``.  Instances ride
    on this config so they survive the ``EngineArgs → VllmConfig`` plumbing
    and reach the runner alongside the dict-form entries.  Only
    ``location='driver'`` instances are permitted (the runner's registry
    enforces this).  Instances don't contribute to ``compute_hash``
    because they're per-run driver-side state, not compile-cache inputs.

    ``activation_cache_bytes`` is the byte budget for the activation store
    (the prefix-cache/capture reuse layer): ``0`` disables it. Like
    ``instances`` it is runtime-only and excluded from ``compute_hash``.
    """

    consumers: list[CaptureConsumerSpec]
    instances: list[Any] = field(default_factory=list)
    activation_cache_bytes: int = 0

    # ---- Capture-aware piecewise cudagraph fallback ----
    # When True, a per-request (client-spec) capture on a decode step replays
    # the piecewise cudagraph (breaking only at the tapped ``capture_residual``
    # split op) instead of forcing the whole step eager. This requires the
    # capture op to be registered as a graph split point, which adds a break at
    # *every* steering/capture hook in *every* layer (per-request taps are not
    # known at compile time, so all hooks must be splittable). That increases
    # the piecewise segment count on every decode step — a steady-state
    # host-side cost — so the fallback is opt-in: enable it only for workloads
    # with high client-capture density, where the per-captured-step win
    # (~4x -> ~1.x) outweighs the extra graph launches on non-capturing steps.
    # Part of ``compute_hash`` because it changes ``splitting_ops`` and thus
    # the compiled graph.
    piecewise_capture_fallback: bool = False
    # ---- Graph-safe per-request capture allowlist ----
    # Startup-configured ``(layer, hook)`` keys for which the manager
    # pre-allocates persistent capture buffers, so a *per-request* (client)
    # spec that taps only these keys is served by the CUDA-graph-safe
    # persistent-buffer path (a fixed-shape full-residual ``copy_`` baked into
    # the graph at warmup + a post-forward host slice) instead of the dynamic
    # in-hook ``index_select`` that forces the whole step eager. A client spec
    # that taps any key outside this allowlist still forces eager (graceful
    # fallback).
    #
    # Memory trade-off: one persistent buffer per covered key, sized
    # ``max_num_tokens × hidden × dtype``, plus a fixed full-residual copy on
    # every forward step at each covered layer regardless of whether any
    # in-flight request currently taps it (so the graph stays static).
    #
    # Empty (default) leaves per-request capture on the eager path.
    graphsafe_keys: list[tuple[int, str]] = field(default_factory=list)

    # ---- Backpressure / overload control (capture-manager level) ----
    # The dispatch queue is the single GPU-facing backpressure point.
    # ``dispatch_queue_size <= 0`` keeps the legacy unbounded behaviour
    # (no backpressure — overload grows memory without bound).
    dispatch_queue_size: int = 256
    # What happens when a bounded dispatch queue is full:
    #   "block" — stall the forward pass (no loss, bounded memory)
    #   "drop"  — discard the step's captures (counted; serving never stalls)
    #   "spill" — park overflow on local disk, replay when the queue drains
    overload_policy: str = "spill"
    # Local scratch directory for the ``spill`` policy. ``None`` defaults to
    # ``$TMPDIR/vllm-capture-spill``. Should be FAST local storage (NVMe/
    # tmpfs) acting as an elastic buffer in front of slower primary storage.
    spill_dir: str | None = None
    # Cap on bytes buffered in the spill area; once exceeded, ``spill``
    # degrades to ``block`` (no loss).
    spill_max_bytes: int = 4 << 30  # 4 GiB

    def compute_hash(self) -> str:
        """Deterministic hash for ``VllmConfig.compute_hash()``."""
        h = hashlib.md5(usedforsecurity=False)
        for spec in self.consumers:
            h.update(spec.name.encode())
            if spec.instance_name:
                h.update(spec.instance_name.encode())
            for k in sorted(spec.params):
                h.update(f"{k}={spec.params[k]}".encode())
        # The piecewise fallback changes splitting_ops (the compiled graph),
        # so it must participate in the compile-cache key.
        h.update(
            f"piecewise_capture_fallback={self.piecewise_capture_fallback}".encode()
        )
        # The graph-safe allowlist changes which persistent buffers and
        # full-residual copies get baked into the CUDA graph, so it is a
        # compile-cache input.
        for layer, hook in sorted(self.graphsafe_keys):
            h.update(f"gs={layer}:{hook}".encode())
        # Backpressure settings are runtime behaviour, not compile-cache
        # inputs, so they are intentionally excluded from the hash.
        return h.hexdigest()[:16]


def parse_consumer_spec(shorthand: str) -> CaptureConsumerSpec:
    """Parse CLI shorthand: ``'name:key=val,key=val'`` into a spec.

    Simple values only — no commas or equals in values.  Use YAML for
    complex params.

    Raises:
        ValueError: If *shorthand* is empty or a key=value pair is malformed.
    """
    if not shorthand or not shorthand.strip():
        raise ValueError("Consumer spec must not be empty")

    shorthand = shorthand.strip()

    # Split on first ':' — left is name, right is key=val pairs
    if ":" in shorthand:
        name, params_str = shorthand.split(":", 1)
    else:
        name = shorthand
        params_str = ""

    name = name.strip()
    if not name:
        raise ValueError("Consumer spec name must not be empty")

    params: dict[str, Any] = {}
    if params_str.strip():
        for pair in params_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"Malformed key=value pair {pair!r} in consumer spec "
                    f"{shorthand!r}; expected 'key=value'"
                )
            key, value = pair.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Empty key in consumer spec {shorthand!r}")
            params[key] = value.strip()

    return CaptureConsumerSpec(name=name, params=params)


_VALID_HOOK_NAMES = frozenset(
    {"pre_attn", "post_attn", "post_block", "mlp_in", "mlp_out"}
)


def parse_graphsafe_key(shorthand: str) -> tuple[int, str]:
    """Parse a graph-safe key shorthand ``'layer:hook'`` into ``(layer, hook)``.

    Raises:
        ValueError: If *shorthand* is malformed or names an unknown hook.
    """
    text = shorthand.strip()
    if ":" not in text:
        raise ValueError(
            f"graph-safe capture key {shorthand!r} must be 'layer:hook' "
            f"(e.g. '12:post_block')"
        )
    layer_str, hook = text.split(":", 1)
    hook = hook.strip()
    try:
        layer = int(layer_str.strip())
    except ValueError as exc:
        raise ValueError(
            f"graph-safe capture key {shorthand!r} has a non-integer layer"
        ) from exc
    if layer < 0:
        raise ValueError(
            f"graph-safe capture key {shorthand!r} has a negative layer"
        )
    if hook not in _VALID_HOOK_NAMES:
        raise ValueError(
            f"graph-safe capture key {shorthand!r} names unknown hook "
            f"{hook!r}; valid hooks: {sorted(_VALID_HOOK_NAMES)}"
        )
    return (layer, hook)


# Hooks that ``:all`` fans out to. Restricted to the standard residual-stream
# taps that are actually wired; ``mlp_in``/``mlp_out`` are valid to name
# explicitly but excluded here so ``:all`` never reserves buffers for taps
# that never fire.
_GRAPHSAFE_ALL_HOOKS: tuple[str, ...] = ("pre_attn", "post_attn", "post_block")


def expand_graphsafe_keys(
    shorthands: list[str], num_layers: int
) -> list[tuple[int, str]]:
    """Expand graph-safe key shorthands into concrete ``(layer, hook)`` keys.

    Supported forms (``L`` is a layer index, ``hook`` a wired hook name):

    - ``"L:hook"``   — a single ``(L, hook)`` key.
    - ``"L:all"``    — every standard hook (``pre_attn``/``post_attn``/
      ``post_block``) at layer ``L``.
    - ``"all:hook"`` — ``hook`` on every layer in ``[0, num_layers)``.
    - ``"all:all"``  — every standard hook on every layer.

    The result is sorted and de-duplicated, so overlapping shorthands (e.g.
    ``"5:post_block"`` and ``"5:all"``) collapse cleanly.

    Args:
        shorthands: Raw ``layer:hook`` strings (``layer``/``hook`` may be
            ``all``).
        num_layers: Total decoder layers, used to expand ``all`` layers.

    Raises:
        ValueError: If a shorthand is malformed, names an unknown hook, or
            references a layer outside ``[0, num_layers)``.
    """
    keys: set[tuple[int, str]] = set()
    for shorthand in shorthands:
        text = shorthand.strip()
        if ":" not in text:
            raise ValueError(
                f"graph-safe capture key {shorthand!r} must be 'layer:hook' "
                f"(e.g. '12:post_block', '12:all', 'all:post_block', 'all:all')"
            )
        layer_str, hook_str = (part.strip() for part in text.split(":", 1))

        if hook_str == "all":
            hooks: tuple[str, ...] = _GRAPHSAFE_ALL_HOOKS
        elif hook_str in _VALID_HOOK_NAMES:
            hooks = (hook_str,)
        else:
            raise ValueError(
                f"graph-safe capture key {shorthand!r} names unknown hook "
                f"{hook_str!r}; valid hooks: {sorted(_VALID_HOOK_NAMES)} or 'all'"
            )

        if layer_str == "all":
            layers: range = range(num_layers)
        else:
            try:
                layer = int(layer_str)
            except ValueError as exc:
                raise ValueError(
                    f"graph-safe capture key {shorthand!r} has a non-integer "
                    f"layer (use an int or 'all')"
                ) from exc
            if not 0 <= layer < num_layers:
                raise ValueError(
                    f"graph-safe capture key {shorthand!r} references layer "
                    f"{layer} outside the model's [0, {num_layers}) layers"
                )
            layers = range(layer, layer + 1)

        for layer in layers:
            for hook in hooks:
                keys.add((layer, hook))

    return sorted(keys)


def resolve_graphsafe_shorthands(
    consumer_specs: list[CaptureConsumerSpec],
    cli_keys: list[str] | None,
) -> tuple[list[str], str]:
    """Resolve the graph-safe key shorthands (pre-expansion) for a run.

    An explicit ``cli_keys`` (``--capture-graphsafe-key`` /
    ``capture_graphsafe_keys``) OVERRIDES; otherwise the default is the union
    of what each registered consumer declares via
    :meth:`CaptureConsumer.declared_graphsafe_keys`.

    Returns ``(shorthands, source)`` where ``source`` is a short human label
    for logging. A consumer whose class fails to load is skipped here (the
    later ``build_consumers`` reports it with a clearer error).
    """
    if cli_keys:
        return list(cli_keys), "--capture-graphsafe-key"

    # Imported lazily to avoid a config<->registry import cycle.
    from vllm.v1.capture.registry import load_consumer_class

    shorthands: list[str] = []
    for spec in consumer_specs:
        try:
            consumer_cls = load_consumer_class(spec.name)
        except Exception:
            continue
        shorthands.extend(consumer_cls.declared_graphsafe_keys(spec.params))
    return shorthands, "registered consumer(s)"


def graphsafe_buffer_bytes(
    num_keys: int, max_num_tokens: int, hidden_size: int, dtype_bytes: int
) -> int:
    """Estimate persistent VRAM (bytes) for ``num_keys`` graph-safe buffers.

    Each key holds one ``[max_num_tokens, hidden_size]`` buffer of the model
    dtype, reserved for the lifetime of the engine.
    """
    return num_keys * max_num_tokens * hidden_size * dtype_bytes


def validate_consumer_specs(specs: list[CaptureConsumerSpec]) -> None:
    """Validate consumer specs: non-empty names, unique instance names.

    Raises:
        ValueError: On empty names or duplicate instance names.
    """
    for spec in specs:
        if not spec.name or not spec.name.strip():
            raise ValueError("Consumer spec name must not be empty")

    # Check for duplicate instance names (only among specs that have one)
    instance_names: list[str] = []
    for spec in specs:
        effective_name = spec.instance_name if spec.instance_name else spec.name
        instance_names.append(effective_name)

    seen: set[str] = set()
    for iname in instance_names:
        if iname in seen:
            raise ValueError(
                f"Duplicate consumer instance name {iname!r}; use "
                f"'instance_name' to disambiguate multiple consumers "
                f"with the same entry-point"
            )
        seen.add(iname)
