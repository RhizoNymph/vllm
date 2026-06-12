# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the sync consumer execution axis (Phase 1a M1).

Covers: registry discovery/validation of ``execution="sync"`` classes,
the four-tuple ``build_consumers`` split, ``build_sync_consumers`` for
non-zero TP ranks, the slim ``CaptureManager`` mode, and
``extra_global_specs`` buffer allocation. CPU-only.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.v1.capture import registry as _registry
from vllm.v1.capture.manager import CaptureManager
from vllm.v1.capture.registry import _reset_cache_for_testing
from vllm.v1.capture.step_view import StepCaptureView
from vllm.v1.capture.types import CaptureSpec

HIDDEN = 8
NUM_LAYERS = 4
MAX_TOKENS = 16


class _SyncProbe:
    """Minimal sync-execution consumer."""

    location = "worker"
    execution = "sync"
    reads_client_spec = False

    def __init__(self, vllm_config, params):
        self.params = params
        self.seen_views: list[StepCaptureView] = []

    def global_capture_spec(self):
        return CaptureSpec(hooks={"post_mlp": [1]}, positions="all_generated")

    def on_step(self, view):
        self.seen_views.append(view)
        return None


def _vllm_config(consumers, pp_size: int = 1) -> MagicMock:
    cfg = MagicMock()
    cfg.capture_consumers_config = consumers
    cfg.parallel_config.pipeline_parallel_size = pp_size
    return cfg


def _patch_registry(classes: dict[str, type]):
    entries = []
    for ep_name, cls in classes.items():
        entry = MagicMock()
        entry.name = ep_name
        entry.load.return_value = cls
        entries.append(entry)

    return patch(
        "vllm.v1.capture.registry.importlib.metadata.entry_points",
        side_effect=lambda *, group: entries,
    )


@pytest.fixture(autouse=True)
def _reset_registry_cache():
    _reset_cache_for_testing()
    yield
    _reset_cache_for_testing()


# ---------------------------------------------------------------------------
# Registry: discovery + validation
# ---------------------------------------------------------------------------


def test_sync_class_accepted_by_entry_points():
    with _patch_registry({"probe": _SyncProbe}):
        cls = _registry.load_consumer_class("probe")
    assert cls is _SyncProbe


def test_non_sync_non_sink_class_still_rejected():
    class _Bogus:
        pass

    with _patch_registry({"bogus": _Bogus}), pytest.raises(TypeError):
        _registry.load_consumer_class("bogus")


def test_build_consumers_splits_sync_out():
    with _patch_registry({"probe": _SyncProbe}):
        config = [{"name": "probe", "params": {"x": 1}}]
        sinks, validators, name_to_index, sync_consumers = _registry.build_consumers(
            _vllm_config(config)
        )
    assert sinks == ()
    assert validators == ()
    assert name_to_index == {}
    assert len(sync_consumers) == 1
    key, instance = sync_consumers[0]
    assert key == "probe"
    assert isinstance(instance, _SyncProbe)
    assert instance.params == {"x": 1}


def test_build_sync_consumers_skips_async():
    from vllm.v1.capture.consumers.logging import LoggingConsumer

    with _patch_registry({"probe": _SyncProbe, "logging": LoggingConsumer}):
        config = [
            {"name": "logging", "params": {"hooks": {"post_mlp": [0]}}},
            {"name": "probe", "params": {}},
        ]
        sync_consumers = _registry.build_sync_consumers(_vllm_config(config))
    assert [name for name, _ in sync_consumers] == ["probe"]


def test_sync_requires_worker_location():
    class _DriverSync(_SyncProbe):
        location = "driver"

    with (
        _patch_registry({"probe": _DriverSync}),
        pytest.raises(ValueError, match="location='worker'"),
    ):
        _registry.build_consumers(_vllm_config([{"name": "probe", "params": {}}]))


def test_sync_rejects_client_spec():
    class _ClientSync(_SyncProbe):
        reads_client_spec = True

    with (
        _patch_registry({"probe": _ClientSync}),
        pytest.raises(ValueError, match="reads_client_spec"),
    ):
        _registry.build_consumers(_vllm_config([{"name": "probe", "params": {}}]))


def test_sync_requires_global_spec():
    class _NoSpecSync(_SyncProbe):
        def global_capture_spec(self):
            return None

    with (
        _patch_registry({"probe": _NoSpecSync}),
        pytest.raises(ValueError, match="global_capture_spec"),
    ):
        _registry.build_consumers(_vllm_config([{"name": "probe", "params": {}}]))


def test_sync_rejects_pipeline_parallel():
    with (
        _patch_registry({"probe": _SyncProbe}),
        pytest.raises(ValueError, match="pipeline_parallel_size=1"),
    ):
        _registry.build_consumers(
            _vllm_config([{"name": "probe", "params": {}}], pp_size=2)
        )


def test_preconstructed_sync_instance_rejected():
    instance = _SyncProbe(MagicMock(), {})
    with _patch_registry({}), pytest.raises(ValueError, match="config-driven"):
        _registry.build_consumers(_vllm_config(None), consumer_instances=[instance])


# ---------------------------------------------------------------------------
# Slim manager
# ---------------------------------------------------------------------------


def _slim_manager(**overrides) -> CaptureManager:
    kwargs = dict(
        consumers=(),
        consumer_specs=(),
        extra_global_specs=(
            CaptureSpec(hooks={"post_mlp": [1]}, positions="all_generated"),
        ),
        num_hidden_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        model_dtype=torch.float32,
        device="cpu",
        max_num_tokens=MAX_TOKENS,
        slim=True,
    )
    kwargs.update(overrides)
    return CaptureManager(**kwargs)


def test_slim_manager_spawns_no_threads():
    mgr = _slim_manager()
    assert mgr._dispatch_thread is None
    assert mgr._finalize_thread is None
    # Shutdown is a no-op, not a crash.
    mgr.shutdown()


def test_slim_manager_allocates_extra_global_buffers():
    mgr = _slim_manager()
    buf = mgr.global_buffer((1, "post_mlp"))
    assert buf is not None
    assert buf.shape == (MAX_TOKENS, HIDDEN)
    assert mgr.global_buffer((0, "post_mlp")) is None


def test_slim_manager_on_hook_fills_buffer():
    mgr = _slim_manager()
    hidden = torch.arange(3 * HIDDEN, dtype=torch.float32).reshape(3, HIDDEN)
    mgr.on_hook(1, "post_mlp", hidden)
    buf = mgr.global_buffer((1, "post_mlp"))
    torch.testing.assert_close(buf[:3], hidden)
    # Non-monitored key: no-op, no crash.
    mgr.on_hook(0, "pre_attn", hidden)


def test_slim_manager_pipeline_entry_points_raise():
    mgr = _slim_manager()
    with pytest.raises(RuntimeError, match="slim"):
        mgr.register_request("r1", None, num_prompt_tokens=4)
    with pytest.raises(RuntimeError, match="slim"):
        mgr.dispatch_step_captures(MagicMock())
    with pytest.raises(RuntimeError, match="slim"):
        mgr.finalize_request("r1")
    with pytest.raises(RuntimeError, match="slim"):
        mgr.finalize_request_async("r1", lambda _r: None)


def test_full_manager_accepts_extra_global_specs():
    """A full (non-slim) manager allocates buffers for sync monitor keys
    on top of its sink consumers' specs."""
    mgr = CaptureManager(
        consumers=(),
        consumer_specs=(),
        extra_global_specs=(CaptureSpec(hooks={"pre_attn": [0, 2]}, positions="all"),),
        num_hidden_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        model_dtype=torch.float32,
        device="cpu",
        max_num_tokens=MAX_TOKENS,
    )
    try:
        assert mgr.global_buffer((0, "pre_attn")) is not None
        assert mgr.global_buffer((2, "pre_attn")) is not None
        assert not mgr._slim
        assert mgr._dispatch_thread.is_alive()
    finally:
        mgr.shutdown()


def test_extra_global_specs_respect_local_layer_range():
    mgr = _slim_manager(
        extra_global_specs=(CaptureSpec(hooks={"post_mlp": [0, 3]}, positions="all"),),
        local_layer_range=(0, 2),
    )
    assert mgr.global_buffer((0, "post_mlp")) is not None
    assert mgr.global_buffer((3, "post_mlp")) is None
