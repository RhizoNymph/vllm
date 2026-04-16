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

Phase G will extend this module with a plural ``build_consumers``
helper that walks ``vllm_config.capture_consumers_config``. Phase A
ships only the singular helpers used by ad-hoc instantiation.
"""

from __future__ import annotations

import importlib.metadata
import threading
from typing import TYPE_CHECKING, Any

from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.errors import UnknownCaptureConsumerError

if TYPE_CHECKING:
    from vllm.config import VllmConfig


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


def _reset_cache_for_testing() -> None:
    """Drop the cached entry-point map.

    Tests that patch ``importlib.metadata.entry_points`` must call this
    before and after patching so the cache does not leak across tests.
    """
    global _class_cache
    with _cache_lock:
        _class_cache = None
