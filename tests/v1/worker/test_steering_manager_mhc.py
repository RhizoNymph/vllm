# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SteeringManager table population with mixed per-hook widths.

mHC models register single-stream hooks (``hidden`` wide) and multi-stream
residual hooks (``hc_mult * hidden`` wide) on the same layer. These tests
exercise :meth:`SteeringManager.populate_steering_tables` across that mixed
set: the populate path must group tables by width, write each table's own
width, and keep the row layout (zeros / global-prefill / global-decode /
per-request) identical to the uniform-width case.
"""

import torch
import torch.nn as nn

from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
    register_steering_buffers,
)
from vllm.v1.worker.steering_manager import SteeringManager

HIDDEN = 4
HC_MULT = 3
HC_DIM = HC_MULT * HIDDEN
MAX_CONFIGS = 2

_SINGLE = SteeringHookPoint.PRE_ATTN
_MULTI = SteeringHookPoint.MHC_STREAMS_PRE_ATTN


def _make_layer(layer_idx: int) -> nn.Module:
    mod = nn.Module()
    mod.layer_idx = layer_idx
    register_steering_buffers(
        mod,
        HIDDEN,
        max_steering_tokens=8,
        max_steering_configs=MAX_CONFIGS,
        dtype=torch.float32,
        hook_widths={_SINGLE: HIDDEN, _MULTI: HC_DIM},
    )
    return mod


def _table(mod: nn.Module, hp: SteeringHookPoint) -> torch.Tensor:
    return getattr(mod, HOOK_POINT_TABLE_ATTR[hp])


def test_mixed_width_global_and_per_request_rows():
    mgr = SteeringManager(max_steering_configs=MAX_CONFIGS, device=None)
    mod = _make_layer(layer_idx=0)
    steerable = {0: mod}

    single_base = torch.full((HIDDEN,), 2.0)
    multi_base = torch.arange(HC_DIM, dtype=torch.float32) + 1.0
    mgr.update_global_vectors(_SINGLE.value, 0, single_base, phase="base")
    mgr.update_global_vectors(_MULTI.value, 0, multi_base, phase="base")

    single_req = [1.0] * HIDDEN
    multi_req = [0.5] * HC_DIM
    row = mgr.register_config(
        config_hash=42,
        vectors={_SINGLE.value: {0: single_req}, _MULTI.value: {0: multi_req}},
        phase="prefill",
    )

    mgr.populate_steering_tables(steerable)

    single = _table(mod, _SINGLE)
    multi = _table(mod, _MULTI)
    # Each table keeps its own width.
    assert single.shape[1] == HIDDEN
    assert multi.shape[1] == HC_DIM

    # Row 0 is always the zero sentinel.
    assert torch.allclose(single[0], torch.zeros(HIDDEN))
    assert torch.allclose(multi[0], torch.zeros(HC_DIM))

    # Rows 1/2 are the phase-global effective vectors (base only here).
    assert torch.allclose(single[1], single_base)
    assert torch.allclose(single[2], single_base)
    assert torch.allclose(multi[1], multi_base)
    assert torch.allclose(multi[2], multi_base)

    # The per-request prefill row folds global + per-request, at each width.
    assert torch.allclose(single[row], single_base + torch.tensor(single_req))
    assert torch.allclose(multi[row], multi_base + torch.tensor(multi_req))

    # Both hooks carry non-zero content -> any-active flags set.
    assert bool(getattr(mod, HOOK_POINT_ANY_ACTIVE_ATTR[_SINGLE]).item())
    assert bool(getattr(mod, HOOK_POINT_ANY_ACTIVE_ATTR[_MULTI]).item())


def test_only_multistream_hook_populates():
    """A layer steered only at the wide hook still populates correctly."""
    mgr = SteeringManager(max_steering_configs=MAX_CONFIGS, device=None)
    mod = nn.Module()
    mod.layer_idx = 0
    register_steering_buffers(
        mod,
        HIDDEN,
        max_steering_tokens=8,
        max_steering_configs=MAX_CONFIGS,
        dtype=torch.float32,
        hook_widths={_MULTI: HC_DIM},
    )

    multi_base = torch.arange(HC_DIM, dtype=torch.float32)
    mgr.update_global_vectors(_MULTI.value, 0, multi_base, phase="base")
    mgr.populate_steering_tables({0: mod})

    multi = _table(mod, _MULTI)
    assert multi.shape[1] == HC_DIM
    assert torch.allclose(multi[1], multi_base)
    assert torch.allclose(multi[0], torch.zeros(HC_DIM))
