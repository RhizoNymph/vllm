# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for PatchSourceConsumer (validate + on_capture -> source store)."""

import pytest
import torch

from vllm.v1.capture.consumers.patch_source import PatchSourceConsumer
from vllm.v1.capture.errors import CaptureValidationError
from vllm.v1.capture.source_store import (
    PatchSourceStore,
    set_active_patch_source_store,
)
from vllm.v1.capture.types import CaptureContext


def _ctx(req_id: str, num_prompt: int, num_layers: int = 4) -> CaptureContext:
    return CaptureContext(
        vllm_internal_request_id=req_id,
        num_prompt_tokens=num_prompt,
        num_computed_tokens=0,
        num_hidden_layers=num_layers,
        hidden_size=8,
        element_size_bytes=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


def _consumer() -> PatchSourceConsumer:
    return PatchSourceConsumer(vllm_config=None, params={})


class TestValidate:
    def test_all_prompt_records_positions(self):
        c = _consumer()
        spec = c.validate_client_spec(
            {"run": "R1", "hooks": {"post_block": [1, 2]}, "positions": "all_prompt"},
            _ctx("req0", num_prompt=3),
        )
        assert spec.hooks == {"post_block": [1, 2]}
        assert spec.positions == "all_prompt"
        assert c._req_state["req0"][0] == "R1"
        assert c._req_state["req0"][1] == [0, 1, 2]

    def test_layers_all_expands(self):
        c = _consumer()
        spec = c.validate_client_spec(
            {"hooks": {"post_block": "all"}, "positions": "last_prompt"},
            _ctx("req0", num_prompt=5, num_layers=4),
        )
        assert spec.hooks == {"post_block": [0, 1, 2, 3]}

    def test_explicit_positions(self):
        c = _consumer()
        c.validate_client_spec(
            {"hooks": {"pre_attn": [0]}, "positions": [0, 2]},
            _ctx("req0", num_prompt=5),
        )
        assert c._req_state["req0"][1] == [0, 2]

    def test_rejects_non_dict(self):
        c = _consumer()
        with pytest.raises(CaptureValidationError):
            c.validate_client_spec([1, 2], _ctx("r", 3))

    def test_rejects_all_positions(self):
        c = _consumer()
        with pytest.raises(CaptureValidationError):
            c.validate_client_spec(
                {"hooks": {"post_block": [0]}, "positions": "all"}, _ctx("r", 3)
            )

    def test_rejects_generated_positions(self):
        c = _consumer()
        with pytest.raises(CaptureValidationError):
            c.validate_client_spec(
                {"hooks": {"post_block": [0]}, "positions": "all_generated"},
                _ctx("r", 3),
            )

    def test_rejects_non_injectable_hook(self):
        c = _consumer()
        with pytest.raises(CaptureValidationError):
            c.validate_client_spec(
                {"hooks": {"attn_scores": [0]}, "positions": "all_prompt"},
                _ctx("r", 3),
            )

    def test_accepts_mlp_hooks(self):
        c = _consumer()
        spec = c.validate_client_spec(
            {"hooks": {"mlp_in": [0], "mlp_out": [1]}, "positions": "all_prompt"},
            _ctx("r", 3),
        )
        assert spec.hooks == {"mlp_in": [0], "mlp_out": [1]}

    def test_rejects_missing_hooks(self):
        c = _consumer()
        with pytest.raises(CaptureValidationError):
            c.validate_client_spec({"positions": "all_prompt"}, _ctx("r", 3))

    def test_rejects_layer_out_of_range(self):
        c = _consumer()
        with pytest.raises(CaptureValidationError):
            c.validate_client_spec(
                {"hooks": {"post_block": [9]}, "positions": "all_prompt"},
                _ctx("r", 3, num_layers=4),
            )


class TestOnCapture:
    def test_rows_mapped_to_positions(self):
        store = PatchSourceStore(max_bytes=0)
        set_active_patch_source_store(store)
        try:
            c = _consumer()
            c.validate_client_spec(
                {"run": "R1", "hooks": {"post_block": [2]}, "positions": "all_prompt"},
                _ctx("req0", num_prompt=3),
            )
            # 3 rows, one per prompt position, distinct values.
            tensor = torch.stack([torch.full((8,), float(p)) for p in range(3)])
            c.on_capture(("req0", 2, "post_block"), tensor, {"client_request_id": "X"})
            for p in range(3):
                row = store.get_row("R1", 2, "post_block", p)
                assert row is not None
                assert torch.allclose(row, torch.full((8,), float(p)))
        finally:
            set_active_patch_source_store(None)

    def test_run_defaults_to_client_request_id(self):
        store = PatchSourceStore(max_bytes=0)
        set_active_patch_source_store(store)
        try:
            c = _consumer()
            c.validate_client_spec(
                {"hooks": {"post_block": [0]}, "positions": "last_prompt"},
                _ctx("req0", num_prompt=4),
            )
            tensor = torch.full((1, 8), 7.0)  # one row -> last_prompt position 3
            c.on_capture(
                ("req0", 0, "post_block"), tensor, {"client_request_id": "client-42"}
            )
            assert store.has_run("client-42")
            row = store.get_row("client-42", 0, "post_block", 3)
            assert row is not None and torch.allclose(row, torch.full((8,), 7.0))
        finally:
            set_active_patch_source_store(None)

    def test_no_store_is_safe(self):
        set_active_patch_source_store(None)
        c = _consumer()
        c.validate_client_spec(
            {"hooks": {"post_block": [0]}, "positions": "all_prompt"},
            _ctx("req0", num_prompt=2),
        )
        # Should not raise even with no active store.
        c.on_capture(("req0", 0, "post_block"), torch.zeros(2, 8), {})

    def test_untracked_request_ignored(self):
        store = PatchSourceStore(max_bytes=0)
        set_active_patch_source_store(store)
        try:
            c = _consumer()
            c.on_capture(("ghost", 0, "post_block"), torch.zeros(2, 8), {})
            assert store.stats().rows_put == 0
        finally:
            set_active_patch_source_store(None)
