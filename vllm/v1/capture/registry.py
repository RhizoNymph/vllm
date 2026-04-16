# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Entry-point discovery for capture consumers.

Third-party plugins and vLLM's own built-in consumers register by
advertising a class under the ``vllm.capture_consumers`` entry-point
group in their ``pyproject.toml``::

    [project.entry-points."vllm.capture_consumers"]
    my_trainer = "my_plugin:RewardTrainer"

This module enumerates that group on first access and caches the
resolved ``name -> class`` map for the process lifetime. Cache
invalidation is not supported — plugins are discovered once per
engine startup.

Phase A ships the singular helpers used by ad-hoc instantiation.
Phase G adds the plural ``build_consumers`` helper that walks config
and installs driver bridges for ``location = "driver"`` consumers.
"""

from __future__ import annotations

import importlib.metadata
import logging
import threading
from typing import TYPE_CHECKING, Any

from vllm.v1.capture.consumer import CaptureConsumer, _BatchedAdapter
from vllm.v1.capture.driver_bridge import install_driver_consumer
from vllm.v1.capture.errors import UnknownCaptureConsumerError
from vllm.v1.capture.sink import CaptureSink

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


ENTRY_POINT_GROUP = "vllm.capture_consumers"

_cache_lock = threading.Lock()
_class_cache: dict[str, type[CaptureConsumer]] | None = None


def _load_entry_points() -> dict[str, type[CaptureConsumer]]:
    """Enumerate the ``vllm.capture_consumers`` entry-point group and
    resolve every entry point to its class.

    Caches the resulting map; subsequent calls return the cached dict.
    """
    global _class_cache
    with _cache_lock:
        if _class_cache is not None:
            return _class_cache

        resolved: dict[str, type[CaptureConsumer]] = {}
        entry_points = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        for ep in entry_points:
            cls = ep.load()
            if not isinstance(cls, type) or not issubclass(cls, CaptureConsumer):
                raise TypeError(
                    f"Entry point {ep.name!r} in group "
                    f"{ENTRY_POINT_GROUP!r} resolved to {cls!r}, which is "
                    f"not a CaptureConsumer subclass."
                )
            resolved[ep.name] = cls

        _class_cache = resolved
        return _class_cache


def load_consumer_class(name: str) -> type[CaptureConsumer]:
    """Return the ``CaptureConsumer`` subclass registered under
    ``name`` in the ``vllm.capture_consumers`` entry-point group.

    Raises ``UnknownCaptureConsumerError`` if no entry point matches.
    """
    resolved = _load_entry_points()
    try:
        return resolved[name]
    except KeyError:
        available = sorted(resolved.keys())
        raise UnknownCaptureConsumerError(
            f"No capture consumer named {name!r} is registered. "
            f"Available consumers: {available}. "
            f"Plugins must advertise a class under the "
            f"{ENTRY_POINT_GROUP!r} entry-point group."
        ) from None


def build_consumer(
    name: str,
    vllm_config: VllmConfig,
    params: dict[str, Any],
) -> CaptureConsumer:
    """Resolve ``name`` via the registry and construct an instance.

    Equivalent to ``load_consumer_class(name)(vllm_config, params)``.
    """
    cls = load_consumer_class(name)
    return cls(vllm_config, params)


def build_consumers(
    vllm_config: VllmConfig,
    consumer_instances: list[CaptureConsumer] | None = None,
) -> tuple[CaptureSink, ...]:
    """Build the complete consumer/sink tuple from config + instances.

    Two sources of consumers are supported:

    1. **Config entries** — dicts of the form
       ``{"name": "...", "params": {...}}`` stored in
       ``vllm_config``. These are resolved through the entry-point
       registry via ``build_consumer`` and can have any ``location``.

    2. **Pre-constructed instances** — passed directly (e.g. from
       ``LLM(..., capture_consumers=[instance])``).  These *must*
       declare ``location = "driver"`` since the ``LLM`` constructor
       runs in the driver process.

    For each consumer:

    - ``location = "worker"`` → wrap in a ``_BatchedAdapter`` (runs
      in-process on the worker).
    - ``location = "driver"`` → install a driver bridge via
      ``install_driver_consumer``, which returns a worker-side shim.

    Returns:
        A tuple of ``CaptureSink`` objects, one per consumer, in the
        order they were provided (config entries first, then instances).
    """
    sinks: list[CaptureSink] = []

    # --- config-driven consumers (entry-point registry) ---
    config_entries: list[dict[str, Any]] = (
        getattr(vllm_config, "capture_consumers_config", None) or []
    )
    for entry in config_entries:
        name = entry["name"]
        params = entry.get("params", {})
        consumer = build_consumer(name, vllm_config, params)
        sinks.append(_wrap_consumer(consumer))

    # --- pre-constructed instances ---
    for instance in consumer_instances or []:
        if instance.location != "driver":
            raise ValueError(
                f"Pre-constructed CaptureConsumer instances passed to "
                f"LLM() must have location='driver', but "
                f"{type(instance).__name__} has "
                f"location={instance.location!r}."
            )
        sinks.append(_wrap_consumer(instance))

    return tuple(sinks)


def _wrap_consumer(consumer: CaptureConsumer) -> CaptureSink:
    """Wrap a consumer with the appropriate sink for its location."""
    if consumer.location == "driver":
        logger.info(
            "Installing driver bridge for consumer %s",
            type(consumer).__name__,
        )
        return install_driver_consumer(consumer)
    # Worker consumers use the in-process batched adapter.
    return _BatchedAdapter(consumer)


def _reset_cache_for_testing() -> None:
    """Drop the cached entry-point map.

    Tests that patch ``importlib.metadata.entry_points`` must call this
    before and after patching so the cache does not leak across tests.
    """
    global _class_cache
    with _cache_lock:
        _class_cache = None
