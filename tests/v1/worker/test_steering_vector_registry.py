# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the worker-side named vector registry and its RPC methods.

Covers the rank-replicated store the declarative gate resolution and the
latch-by-reference bridge read from: register / resolve / unregister, digest
storage and re-registration, and the ``SteeringModelRunnerMixin`` RPC wrappers
that mutate the process-global registry.
"""

import base64

import numpy as np
import pytest

from vllm.config.steering_types import steering_vector_content_digest
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin
from vllm.v1.worker.steering_vector_registry import (
    WorkerSteeringVectorRegistry,
    get_worker_steering_vector_registry,
    install_worker_steering_vector_registry,
)

HIDDEN = 6


def _pack(vec, layer=5, hook="post_block"):
    a = np.asarray(vec, dtype=np.float32)
    return {
        hook: {
            "dtype": "float32",
            "shape": [1, a.shape[0]],
            "layer_indices": [layer],
            "data": base64.b64encode(a.tobytes()).decode(),
        }
    }


def test_register_resolve_unregister():
    reg = WorkerSteeringVectorRegistry()
    packed = _pack(np.arange(HIDDEN))
    reg.register("s", "steer", packed)
    assert reg.count() == 1
    resolved = reg.resolve_vectors("s", "steer")
    assert resolved is not None
    vectors, digest = resolved
    np.testing.assert_allclose(vectors["post_block"][5], np.arange(HIDDEN))
    assert digest == steering_vector_content_digest(packed)
    # namespaces are independent
    assert reg.resolve_vectors("s", "probe") is None
    assert reg.unregister("s", "steer") is True
    assert reg.unregister("s", "steer") is False
    assert reg.resolve_vectors("s", "steer") is None


def test_digest_passthrough_and_recompute_match():
    reg = WorkerSteeringVectorRegistry()
    packed = _pack(np.ones(HIDDEN))
    digest = steering_vector_content_digest(packed)
    # Explicit digest (frontend-broadcast) and recomputed digest agree.
    reg.register("a", "steer", packed, digest)
    reg.register("b", "steer", packed)  # recomputed
    assert reg.resolve_vectors("a", "steer")[1] == reg.resolve_vectors("b", "steer")[1]


def test_reregister_replaces_content_and_digest():
    reg = WorkerSteeringVectorRegistry()
    reg.register("s", "steer", _pack(np.ones(HIDDEN)))
    d1 = reg.resolve_vectors("s", "steer")[1]
    reg.register("s", "steer", _pack(np.ones(HIDDEN) * 5))
    vectors, d2 = reg.resolve_vectors("s", "steer")
    np.testing.assert_allclose(vectors["post_block"][5], np.full(HIDDEN, 5.0))
    assert d1 != d2


def test_invalid_kind_rejected():
    reg = WorkerSteeringVectorRegistry()
    with pytest.raises(ValueError):
        reg.register("s", "bogus", _pack(np.ones(HIDDEN)))
    with pytest.raises(ValueError):
        reg.resolve_vectors("s", "bogus")


@pytest.fixture
def _fresh_global():
    prev = get_worker_steering_vector_registry()
    install_worker_steering_vector_registry(WorkerSteeringVectorRegistry())
    yield
    install_worker_steering_vector_registry(prev)


def test_mixin_rpc_methods_mutate_global(_fresh_global):
    # The RPC handlers only touch the process-global registry, so a bare
    # (uninitialized) mixin instance suffices to exercise them.
    mixin = object.__new__(SteeringModelRunnerMixin)
    packed = _pack(np.arange(HIDDEN))
    digest = steering_vector_content_digest(packed)
    mixin.register_steering_vector_name("s", "steer", packed, digest)
    resolved = get_worker_steering_vector_registry().resolve_vectors("s", "steer")
    assert resolved is not None and resolved[1] == digest
    assert mixin.unregister_steering_vector_name("s", "steer") is True
    assert mixin.unregister_steering_vector_name("s", "steer") is False


def test_mixin_rpc_installs_registry_when_absent():
    # If no registry is installed yet, the register RPC installs one so the
    # replicated state cannot diverge across ranks.
    install_worker_steering_vector_registry(None)
    try:
        mixin = object.__new__(SteeringModelRunnerMixin)
        mixin.register_steering_vector_name("s", "steer", _pack(np.ones(HIDDEN)))
        assert get_worker_steering_vector_registry() is not None
        assert (
            get_worker_steering_vector_registry().resolve_vectors("s", "steer")
            is not None
        )
    finally:
        install_worker_steering_vector_registry(None)
