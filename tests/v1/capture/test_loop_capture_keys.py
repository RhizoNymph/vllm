# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture on looped models: a requested layer captures every visit, one
row per (layer, pass), each under its layer key (`vllm.config.loop_layers`).
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import loop_layers
from vllm.config.loop_layers import LoopLayerSpace
from vllm.v1.capture.manager import CaptureManager
from vllm.v1.capture.types import CaptureSpec, expand_capture_spec

pytestmark = pytest.mark.skip_global_cleanup

# Layer 0 prelude, layers 1-2 run 3 passes, layer 3 coda (as the tiny
# Inkling2 loop test model).
SPACE = LoopLayerSpace(num_layers=4, core_start=1, core_end=3, num_passes=3)


@pytest.fixture
def looped(monkeypatch):
    monkeypatch.setattr(loop_layers, "_ACTIVE", SPACE)
    monkeypatch.setattr(loop_layers, "_INSTALLED", True)


def _sink():
    sink = MagicMock()
    sink.location = "worker"
    sink.submit_chunk_batch = None
    return sink


def test_expand_capture_spec():
    spec = CaptureSpec(hooks={"post_block": [0, 1], "pre_attn": [3]}, positions="all")
    out = expand_capture_spec(spec, SPACE)
    assert out.hooks == {"post_block": [0, 5, 9, 13], "pre_attn": [3]}
    assert out.positions == "all"
    assert expand_capture_spec(out, SPACE).hooks == out.hooks
    assert expand_capture_spec(spec, None) is spec


def test_manager_registers_one_row_per_visit(looped):
    mgr = CaptureManager(
        consumers=(_sink(),),
        consumer_specs=(
            CaptureSpec(hooks={"post_attn": [2]}, positions="last_prompt"),
        ),
        num_hidden_layers=4,
        hidden_size=8,
        model_dtype=torch.float32,
    )
    # Global specs are expanded at construction.
    assert mgr._consumer_specs[0].hooks == {"post_attn": [6, 10, 14]}
    mgr.register_request(
        "r",
        {0: CaptureSpec(hooks={"post_block": [1, 3]}, positions="all")},
        num_prompt_tokens=4,
    )
    spec = mgr._requests["r"].consumer_specs[0]
    assert spec.hooks == {"post_block": [5, 9, 13, 3]}
    mgr.shutdown()


def test_manager_still_rejects_out_of_range_layers(looped):
    mgr = CaptureManager(
        consumers=(_sink(),),
        consumer_specs=(None,),
        num_hidden_layers=4,
        hidden_size=8,
        model_dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="out of range"):
        mgr.register_request(
            "r",
            {0: CaptureSpec(hooks={"post_block": [4]}, positions="all")},
            num_prompt_tokens=4,
        )
    mgr.shutdown()
