# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the frontend named probe/steer vector registry."""

import base64

import numpy as np
import pytest

from vllm.entrypoints.openai.steering.vector_registry import SteeringVectorRegistry

HIDDEN = 6


def _pack(vec, layer, hook="post_block"):
    a = np.asarray(vec, dtype=np.float32)
    return {
        hook: {
            "dtype": "float32",
            "shape": [1, a.shape[0]],
            "layer_indices": [layer],
            "data": base64.b64encode(a.tobytes()).decode(),
        }
    }


@pytest.mark.asyncio
async def test_register_get_list_unregister():
    reg = SteeringVectorRegistry()
    assert reg.count() == 0
    await reg.register("p", "probe", _pack(np.arange(HIDDEN), 5))
    await reg.register("s", "steer", _pack(np.ones(HIDDEN), 5))
    assert reg.list_vectors() == {"probe": ["p"], "steer": ["s"]}
    assert reg.count() == 2
    # probe and steer namespaces are independent
    assert reg.get_packed("p", "probe") is not None
    assert reg.get_packed("p", "steer") is None
    assert await reg.unregister("p", "probe") is True
    assert await reg.unregister("p", "probe") is False
    assert reg.list_vectors()["probe"] == []


@pytest.mark.asyncio
async def test_probe_must_be_single_site():
    reg = SteeringVectorRegistry()
    packed = _pack(np.ones(HIDDEN), 5)
    packed["pre_attn"] = _pack(np.ones(HIDDEN), 6, "pre_attn")["pre_attn"]
    with pytest.raises(ValueError, match="exactly one"):
        await reg.register("bad", "probe", packed)
    # a steer may be multi-site, though
    await reg.register("ok", "steer", packed)
    assert "ok" in reg.list_vectors()["steer"]


@pytest.mark.asyncio
async def test_invalid_hook_and_kind_rejected():
    reg = SteeringVectorRegistry()
    with pytest.raises(ValueError):
        await reg.register("x", "bogus", _pack(np.ones(HIDDEN), 5))
    with pytest.raises(ValueError, match="hook"):
        await reg.register("x", "steer", _pack(np.ones(HIDDEN), 5, "nope"))


@pytest.mark.asyncio
async def test_layer_index_validation():
    reg = SteeringVectorRegistry(valid_layer_indices={0, 1, 2})
    with pytest.raises(ValueError, match="layer"):
        await reg.register("x", "steer", _pack(np.ones(HIDDEN), 9))
    await reg.register("ok", "steer", _pack(np.ones(HIDDEN), 1))
