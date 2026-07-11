# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ``SyncCaptureConsumer`` contract ABC.

The contract used to live only in scattered base classes and tribal
knowledge; a sync consumer missing ``declared_graphsafe_keys`` crashed with a
cryptic ``AttributeError`` deep in the capture registry's config build (never
in a unit test that instantiated the class directly). The ABC turns those gaps
into clear failures at construction time and ships a ``[]`` default for
``declared_graphsafe_keys``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm.v1.capture.consumer import SyncCaptureConsumer
from vllm.v1.capture.types import CaptureSpec


class _GoodConsumer(SyncCaptureConsumer):
    def global_capture_spec(self) -> CaptureSpec:
        return CaptureSpec(hooks={"post_block": [0]}, positions="all_generated")

    def on_step(self, view):
        return None


def test_complete_subclass_constructs_and_has_fixed_metadata():
    consumer = _GoodConsumer(MagicMock(), {})
    assert consumer.location == "worker"
    assert consumer.execution == "sync"
    assert consumer.reads_client_spec is False


def test_missing_on_step_fails_clearly_at_construction():
    class _MissingOnStep(SyncCaptureConsumer):
        def global_capture_spec(self) -> CaptureSpec:
            return CaptureSpec(hooks={"post_block": [0]}, positions="all_generated")

    with pytest.raises(TypeError, match="on_step"):
        _MissingOnStep(MagicMock(), {})  # type: ignore[abstract]


def test_missing_global_capture_spec_fails_clearly_at_construction():
    class _MissingSpec(SyncCaptureConsumer):
        def on_step(self, view):
            return None

    with pytest.raises(TypeError, match="global_capture_spec"):
        _MissingSpec(MagicMock(), {})  # type: ignore[abstract]


def test_declared_graphsafe_keys_default_is_empty():
    # Callable on the *class* (the registry resolves it at config-build time
    # without constructing an instance) and on an instance.
    assert _GoodConsumer.declared_graphsafe_keys({"anything": 1}) == []
    assert _GoodConsumer(MagicMock(), {}).declared_graphsafe_keys({}) == []


def test_declared_graphsafe_keys_is_overridable():
    class _WithKeys(_GoodConsumer):
        @classmethod
        def declared_graphsafe_keys(cls, params):
            return [f"{params['layer']}:post_block"]

    assert _WithKeys.declared_graphsafe_keys({"layer": 3}) == ["3:post_block"]


def test_shutdown_default_is_noop():
    # Must not raise.
    _GoodConsumer(MagicMock(), {}).shutdown(timeout=1.0)
