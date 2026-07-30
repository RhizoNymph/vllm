# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Entry-point loading for capture consumers, without monkeypatching.

The plugin-backed tests need ``tests/plugins/vllm_add_dummy_capture_consumer``
installed (``uv pip install -e tests/plugins/vllm_add_dummy_capture_consumer``)
and skip when its entry point is absent, mirroring the other
``tests/plugins`` packages. The built-in-consumer tests run everywhere.
"""

import importlib.metadata

import pytest

from vllm.config import VllmConfig
from vllm.v1.capture import registry
from vllm.v1.capture.errors import UnknownCaptureConsumerError

PLUGIN_NAME = "dummy_capture_consumer"


def _plugin_entry_point_present() -> bool:
    eps = importlib.metadata.entry_points(group=registry.ENTRY_POINT_GROUP)
    return PLUGIN_NAME in eps.names


requires_plugin = pytest.mark.skipif(
    not _plugin_entry_point_present(),
    reason=(
        f"{PLUGIN_NAME!r} entry point not installed; run "
        "`uv pip install -e tests/plugins/vllm_add_dummy_capture_consumer`"
    ),
)


@pytest.fixture(autouse=True)
def _fresh_registry_cache():
    """Isolate the process-lifetime entry-point cache per test."""
    registry._reset_cache_for_testing()
    yield
    registry._reset_cache_for_testing()


def test_builtin_consumers_resolve_without_entry_points():
    """patch_source/_declarative_steering survive stale dist-info."""
    for name in ("patch_source", "_declarative_steering"):
        assert isinstance(registry.load_consumer_class(name), type)


def test_unknown_consumer_error_lists_available():
    with pytest.raises(UnknownCaptureConsumerError, match="patch_source"):
        registry.load_consumer_class("no_such_consumer")


@requires_plugin
def test_plugin_class_resolves_via_entry_point():
    """A real installed entry point resolves through importlib.metadata."""
    from dummy_capture_consumer.dummy_capture_consumer import DummyCaptureConsumer

    assert registry.load_consumer_class(PLUGIN_NAME) is DummyCaptureConsumer


@requires_plugin
def test_plugin_instance_built_from_config_params():
    """build_consumer constructs the plugin with (vllm_config, params)."""
    from dummy_capture_consumer.dummy_capture_consumer import DummyCaptureConsumer

    instance = registry.build_consumer(
        PLUGIN_NAME, VllmConfig(), {"hooks": {"post_block": [3]}}
    )
    assert isinstance(instance, DummyCaptureConsumer)
    spec = instance.global_capture_spec()
    assert spec.hooks == {"post_block": [3]}
    assert spec.positions == "last_prompt"
