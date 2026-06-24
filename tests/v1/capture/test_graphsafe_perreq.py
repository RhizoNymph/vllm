# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for graph-safe per-request capture (Tier 2).

A startup-configured ``(layer, hook)`` allowlist pre-allocates persistent
capture buffers so a *per-request* (client) spec tapping only allowlisted
keys is served by the CUDA-graph-safe persistent-buffer path instead of the
dynamic in-hook ``index_select`` that forces the whole step eager.

These tests prove, on CPU:

1. The :class:`CaptureStepGate` does **not** force eager for a client spec
   tapping only covered keys, but **does** for a spec tapping an uncovered
   key (graceful fallback).
2. The manager routes a covered client key through the persistent buffer
   (``global_gather_indices``) and the post-forward slice yields the same
   rows as the eager dynamic gather.
3. Config / CLI parsing of the allowlist.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.capture.config import (
    CaptureConsumersConfig,
    CaptureConsumerSpec,
    expand_graphsafe_keys,
    graphsafe_buffer_bytes,
    parse_graphsafe_key,
    resolve_graphsafe_shorthands,
)
from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.consumers.filesystem import FilesystemConsumer
from vllm.v1.capture.manager import CaptureManager
from vllm.v1.capture.plan import CaptureBatchView
from vllm.v1.capture.step_gate import CaptureStepGate
from vllm.v1.capture.types import CaptureSpec

NUM_LAYERS = 4
HIDDEN_SIZE = 8
MODEL_DTYPE = torch.float32


def _make_sink() -> MagicMock:
    sink = MagicMock()
    sink.location = "worker"
    sink.submit_chunk = MagicMock()
    sink.submit_chunk_batch = None
    sink.submit_finalize = MagicMock()
    sink.get_result = MagicMock(return_value=None)
    sink.wait_for_result = MagicMock(return_value=None)
    sink.shutdown = MagicMock()
    return sink


def _view(rows):
    """Build a CaptureBatchView from (req_id, num_prompt, num_computed,
    num_scheduled) tuples."""
    req_ids, npt, ncomp, nsched, offsets = [], [], [], [], []
    off = 0
    for req_id, p, c, s in rows:
        req_ids.append(req_id)
        npt.append(p)
        ncomp.append(c)
        nsched.append(s)
        offsets.append(off)
        off += s
    return CaptureBatchView(
        req_ids=req_ids,
        num_prompt_tokens=npt,
        num_computed_tokens=ncomp,
        num_scheduled_tokens=nsched,
        token_offsets=offsets,
    )


# ---------------------------------------------------------------------------
# Config / CLI parsing
# ---------------------------------------------------------------------------


class TestConfig:
    def test_parse_graphsafe_key(self):
        assert parse_graphsafe_key("12:post_block") == (12, "post_block")
        assert parse_graphsafe_key("  0:pre_attn ") == (0, "pre_attn")

    def test_parse_graphsafe_key_rejects_bad_forms(self):
        with pytest.raises(ValueError):
            parse_graphsafe_key("12")  # no hook
        with pytest.raises(ValueError):
            parse_graphsafe_key("x:post_block")  # non-integer layer
        with pytest.raises(ValueError):
            parse_graphsafe_key("-1:post_block")  # negative
        with pytest.raises(ValueError):
            parse_graphsafe_key("3:not_a_hook")  # unknown hook

    def test_graphsafe_keys_in_compute_hash(self):
        base = CaptureConsumersConfig(consumers=[CaptureConsumerSpec(name="fs")])
        with_keys = CaptureConsumersConfig(
            consumers=[CaptureConsumerSpec(name="fs")],
            graphsafe_keys=[(1, "post_block")],
        )
        assert base.compute_hash() != with_keys.compute_hash()


class TestExpandShorthands:
    def test_concrete_key(self):
        assert expand_graphsafe_keys(["12:post_block"], num_layers=32) == [
            (12, "post_block")
        ]

    def test_layer_all_fans_out_standard_hooks(self):
        assert expand_graphsafe_keys(["5:all"], num_layers=32) == [
            (5, "post_attn"),
            (5, "post_block"),
            (5, "pre_attn"),
        ]

    def test_all_layers_one_hook(self):
        assert expand_graphsafe_keys(["all:post_block"], num_layers=4) == [
            (0, "post_block"),
            (1, "post_block"),
            (2, "post_block"),
            (3, "post_block"),
        ]

    def test_all_all(self):
        keys = expand_graphsafe_keys(["all:all"], num_layers=2)
        assert len(keys) == 2 * 3
        assert (0, "pre_attn") in keys and (1, "post_block") in keys

    def test_dedup_overlapping_shorthands(self):
        # "5:all" already covers "5:post_block" -> no duplicate.
        keys = expand_graphsafe_keys(["5:post_block", "5:all"], num_layers=32)
        assert keys == [(5, "post_attn"), (5, "post_block"), (5, "pre_attn")]

    def test_all_forms_does_not_include_unwired_hooks(self):
        # mlp_in/mlp_out are valid to name explicitly but excluded from :all.
        keys = expand_graphsafe_keys(["3:all"], num_layers=8)
        assert all(h not in ("mlp_in", "mlp_out") for _, h in keys)
        # ...but explicit naming still works.
        assert expand_graphsafe_keys(["3:mlp_in"], num_layers=8) == [(3, "mlp_in")]

    def test_layer_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="outside the model's"):
            expand_graphsafe_keys(["40:post_block"], num_layers=32)

    def test_unknown_hook_rejected(self):
        with pytest.raises(ValueError, match="unknown hook"):
            expand_graphsafe_keys(["3:nope"], num_layers=32)

    def test_missing_colon_rejected(self):
        with pytest.raises(ValueError, match="must be 'layer:hook'"):
            expand_graphsafe_keys(["12"], num_layers=32)

    def test_buffer_bytes_estimate(self):
        # 2 keys x 8192 tokens x 1024 hidden x 2 bytes (bf16) = 32 MiB.
        assert graphsafe_buffer_bytes(2, 8192, 1024, 2) == 2 * 8192 * 1024 * 2


class TestConsumerDeclaredKeys:
    def test_base_consumer_declares_nothing(self):
        assert CaptureConsumer.declared_graphsafe_keys({}) == []

    def test_filesystem_declares_from_list_param(self):
        assert FilesystemConsumer.declared_graphsafe_keys(
            {"root": "/x", "graphsafe_keys": ["12:post_block", "all:post_attn"]}
        ) == ["12:post_block", "all:post_attn"]

    def test_filesystem_declares_from_cli_style_string(self):
        # ',' separates CLI params, so the list uses ';'.
        assert FilesystemConsumer.declared_graphsafe_keys(
            {"root": "/x", "graphsafe_keys": "12:post_block;20:post_block"}
        ) == ["12:post_block", "20:post_block"]

    def test_filesystem_declares_nothing_without_param(self):
        assert FilesystemConsumer.declared_graphsafe_keys({"root": "/x"}) == []


class TestResolveShorthands:
    def test_cli_overrides_consumer_declarations(self):
        specs = [
            CaptureConsumerSpec(
                name="filesystem",
                params={"root": "/x", "graphsafe_keys": ["12:post_block"]},
            )
        ]
        keys, source = resolve_graphsafe_shorthands(specs, cli_keys=["5:post_block"])
        assert keys == ["5:post_block"]
        assert source == "--capture-graphsafe-key"

    def test_default_is_union_of_consumer_declarations(self):
        specs = [
            CaptureConsumerSpec(
                name="filesystem",
                params={"root": "/a", "graphsafe_keys": ["12:post_block"]},
            ),
            CaptureConsumerSpec(
                name="filesystem",
                params={"root": "/b", "graphsafe_keys": ["20:post_block"]},
            ),
        ]
        keys, source = resolve_graphsafe_shorthands(specs, cli_keys=None)
        assert keys == ["12:post_block", "20:post_block"]
        assert source == "registered consumer(s)"

    def test_empty_when_nothing_declared(self):
        specs = [CaptureConsumerSpec(name="filesystem", params={"root": "/x"})]
        keys, _ = resolve_graphsafe_shorthands(specs, cli_keys=None)
        assert keys == []

    def test_union_then_expand_dedups(self):
        # End-to-end: overlapping consumer declarations expand+dedup cleanly.
        specs = [
            CaptureConsumerSpec(
                name="filesystem",
                params={"root": "/a", "graphsafe_keys": ["5:post_block"]},
            ),
            CaptureConsumerSpec(
                name="filesystem",
                params={"root": "/b", "graphsafe_keys": ["5:all"]},
            ),
        ]
        raw, _ = resolve_graphsafe_shorthands(specs, cli_keys=None)
        assert expand_graphsafe_keys(raw, num_layers=32) == [
            (5, "post_attn"),
            (5, "post_block"),
            (5, "pre_attn"),
        ]


# ---------------------------------------------------------------------------
# Step gate: covered vs uncovered keys
# ---------------------------------------------------------------------------


class TestGateCoverage:
    def test_covered_client_key_does_not_force_eager(self):
        gate = CaptureStepGate(graphsafe_keys=frozenset({(1, "post_block")}))
        gate.register(
            "a",
            {"fs": {"hooks": {"post_block": [1]}, "positions": "all_generated"}},
        )
        assert gate.tracked_requests() == 1
        # A decode step that captures (all_generated hits the gen token) must
        # NOT force eager because the only tapped key is covered.
        assert gate.step_captures(_view([("a", 10, 10, 1)])) is False

    def test_uncovered_layer_forces_eager(self):
        # Allowlist covers layer 1; the spec taps layer 2 → still eager.
        gate = CaptureStepGate(graphsafe_keys=frozenset({(1, "post_block")}))
        gate.register(
            "a",
            {"fs": {"hooks": {"post_block": [2]}, "positions": "all_generated"}},
        )
        assert gate.step_captures(_view([("a", 10, 10, 1)])) is True

    def test_uncovered_hook_forces_eager(self):
        # Same layer, different hook → not covered.
        gate = CaptureStepGate(graphsafe_keys=frozenset({(1, "post_block")}))
        gate.register(
            "a",
            {"fs": {"hooks": {"pre_attn": [1]}, "positions": "all_generated"}},
        )
        assert gate.step_captures(_view([("a", 10, 10, 1)])) is True

    def test_partial_coverage_forces_eager(self):
        # One key covered, one not → the uncovered one forces eager.
        gate = CaptureStepGate(graphsafe_keys=frozenset({(1, "post_block")}))
        gate.register(
            "a",
            {"fs": {"hooks": {"post_block": [1, 2]}, "positions": "all_generated"}},
        )
        assert gate.step_captures(_view([("a", 10, 10, 1)])) is True

    def test_covered_but_out_of_window_does_not_force_eager(self):
        # Covered key, but last_prompt position is not in this decode window.
        gate = CaptureStepGate(graphsafe_keys=frozenset({(1, "post_block")}))
        gate.register(
            "a",
            {"fs": {"hooks": {"post_block": [1]}, "positions": "last_prompt"}},
        )
        assert gate.step_captures(_view([("a", 10, 10, 1)])) is False
        # Prefill chunk covering the last prompt token is still graph-safe.
        assert gate.step_captures(_view([("a", 10, 0, 10)])) is False

    def test_no_allowlist_forces_eager_like_before(self):
        # Without an allowlist every client capture forces eager (legacy).
        gate = CaptureStepGate()
        gate.register(
            "a",
            {"fs": {"hooks": {"post_block": [1]}, "positions": "all_generated"}},
        )
        assert gate.step_captures(_view([("a", 10, 10, 1)])) is True

    def test_unparseable_hooks_forces_eager(self):
        # A spec whose hooks aren't the expected mapping is treated as
        # uncovered (conservative) even if an allowlist is configured.
        gate = CaptureStepGate(graphsafe_keys=frozenset({(1, "post_block")}))
        gate.register("a", {"fs": {"positions": "all_generated"}})  # no hooks
        assert gate.step_captures(_view([("a", 10, 10, 1)])) is True

    def test_mixed_batch_one_covered_one_uncovered(self):
        gate = CaptureStepGate(graphsafe_keys=frozenset({(1, "post_block")}))
        gate.register(
            "safe",
            {"fs": {"hooks": {"post_block": [1]}, "positions": "all_generated"}},
        )
        gate.register(
            "unsafe",
            {"fs": {"hooks": {"post_block": [3]}, "positions": "all_generated"}},
        )
        # The uncovered request forces eager for the whole batch.
        assert (
            gate.step_captures(_view([("safe", 10, 10, 1), ("unsafe", 8, 8, 1)]))
            is True
        )


# ---------------------------------------------------------------------------
# Manager: per-request key served from the persistent buffer
# ---------------------------------------------------------------------------


def _make_graphsafe_manager(
    graphsafe_keys,
    specs=(None,),
    max_num_tokens: int = 16,
):
    """Manager with a graph-safe allowlist and no global spec by default."""
    sink = _make_sink()
    mgr = CaptureManager(
        consumers=(sink,),
        consumer_specs=specs,
        num_hidden_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        model_dtype=MODEL_DTYPE,
        max_num_tokens=max_num_tokens,
        graphsafe_keys=graphsafe_keys,
    )
    return mgr, sink


class TestManagerBufferPath:
    def test_allowlist_allocates_buffers_without_global_spec(self):
        mgr, _ = _make_graphsafe_manager(graphsafe_keys=[(1, "post_block")])
        # The allowlisted key gets a persistent buffer even with no global
        # spec (consumer_specs is (None,)).
        assert (1, "post_block") in mgr._global_buffers
        assert mgr.graphsafe_keys == frozenset({(1, "post_block")})
        buf = mgr._global_buffers[(1, "post_block")]
        assert buf.shape == (16, HIDDEN_SIZE)

    def test_no_buffers_without_max_num_tokens(self):
        mgr, _ = _make_graphsafe_manager(
            graphsafe_keys=[(1, "post_block")], max_num_tokens=0
        )
        assert mgr._global_buffers == {}
        assert mgr.graphsafe_keys == frozenset()

    def test_out_of_range_allowlist_key_dropped(self):
        # Layer 99 is out of the model's layer range → no buffer.
        mgr, _ = _make_graphsafe_manager(graphsafe_keys=[(99, "post_block")])
        assert mgr._global_buffers == {}
        assert mgr.graphsafe_keys == frozenset()

    def test_client_spec_on_covered_key_routes_to_buffer(self):
        mgr, _ = _make_graphsafe_manager(graphsafe_keys=[(1, "post_block")])
        client_spec = CaptureSpec(
            hooks={"post_block": [1]}, positions="last_prompt"
        )
        mgr.register_request(
            "r1", client_specs={0: client_spec}, num_prompt_tokens=10
        )
        plan = mgr.build_step_plan(
            _view([("r1", 10, 0, 10)])
        )
        # Covered client key takes the persistent-buffer path, not the
        # dynamic in-hook gather (which would force eager).
        assert (1, "post_block") in plan.global_gather_indices
        assert (1, "post_block") not in plan.gather_indices
        assert plan.gather_indices == {}
        # last_prompt of a 10-token prompt is absolute row 9.
        assert plan.global_gather_indices[(1, "post_block")].tolist() == [9]

    def test_buffer_slice_matches_eager_gather(self):
        """The post-forward buffer slice yields the same rows as the eager
        dynamic gather would have."""
        mgr, sink = _make_graphsafe_manager(graphsafe_keys=[(1, "post_block")])
        client_spec = CaptureSpec(
            hooks={"post_block": [1]}, positions="last_prompt"
        )
        mgr.register_request(
            "r1", client_specs={0: client_spec}, num_prompt_tokens=10
        )
        plan = mgr.build_step_plan(_view([("r1", 10, 0, 10)]))

        hidden = torch.arange(10 * HIDDEN_SIZE, dtype=MODEL_DTYPE).reshape(
            10, HIDDEN_SIZE
        )
        # on_hook performs the fixed full-residual copy into the buffer
        # (the recorded copy at graph replay).
        mgr.on_hook(1, "post_block", hidden)
        # The buffer holds the full residual; on_hook must NOT populate
        # scratch for a buffered key (host slices it post-forward).
        assert (1, "post_block") not in plan.scratch_gpu

        mgr.dispatch_step_captures(plan)
        mgr._drain_dispatch_queue()

        assert sink.submit_chunk.call_count == 1
        chunk = sink.submit_chunk.call_args_list[0].args[0]
        # The dispatched row equals what an eager index_select([9]) would give.
        eager = hidden.index_select(0, torch.tensor([9]))
        torch.testing.assert_close(chunk.tensor, eager)

        results = mgr.finalize_request("r1")
        assert 0 in results

    def test_uncovered_client_key_stays_on_dynamic_path(self):
        # An allowlist that covers layer 1 only; the client taps layer 2,
        # which must keep the dynamic in-hook gather (forces eager at runtime).
        mgr, _ = _make_graphsafe_manager(graphsafe_keys=[(1, "post_block")])
        client_spec = CaptureSpec(
            hooks={"post_block": [2]}, positions="last_prompt"
        )
        mgr.register_request(
            "r1", client_specs={0: client_spec}, num_prompt_tokens=10
        )
        plan = mgr.build_step_plan(_view([("r1", 10, 0, 10)]))
        assert (2, "post_block") in plan.gather_indices
        assert (2, "post_block") not in plan.global_gather_indices
