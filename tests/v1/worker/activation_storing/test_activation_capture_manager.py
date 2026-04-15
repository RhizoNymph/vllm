# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for :mod:`vllm.model_executor.layers.activation_capture`.

These tests exercise the capture manager as a pure CPU component: they
never install the manager as the process-global
``_ACTIVE_CAPTURE_MANAGER`` and they drive ``build_step_plan`` /
``on_hook`` directly. That keeps the tests fast and decoupled from
``vllm.v1.worker.gpu_input_batch``.

Coverage matches the Phase 3 done-when list:

1. ``last_prompt`` register + single-step plan correctness
2. multi-layer, single-hook plan
3. single-layer, multi-hook plan
4. request whose only position is in a future step
5. multi-step accumulation with growing ``num_computed_tokens``
6. batch-view req_ids the manager doesn't know about are skipped
7. ``unregister_request`` drops state
8. ``on_hook`` index-select correctness
9. ``on_hook`` skips unknown (layer, hook) pairs
10. ``on_hook`` casts scratch to the plan dtype
11. request errors surface via ``plan.request_errors`` + ``consume_step_plan``
"""

from __future__ import annotations

import pytest
import torch

from vllm.config.activation_storing_types import ActivationStoringSpec
from vllm.entrypoints.openai.activation_storing_validation import (
    ResolvedActivationStoringSpec,
)
from vllm.model_executor.layers.activation_capture import (
    ActivationCaptureManager,
    CaptureBatchView,
    CapturePositionEntry,
    StepCapturePlan,
)

HIDDEN_SIZE = 8
NUM_HIDDEN_LAYERS = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    num_hidden_layers: int = NUM_HIDDEN_LAYERS,
    hidden_size: int = HIDDEN_SIZE,
) -> ActivationCaptureManager:
    return ActivationCaptureManager(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        model_dtype=dtype,
        device=device,
    )


def _make_resolved_spec(
    request_id: str,
    tag: str,
    hooks: dict[str, list[int]],
    positions: list[int] | str,
    position_kind: str,
) -> ResolvedActivationStoringSpec:
    """Build a :class:`ResolvedActivationStoringSpec` for tests.

    We intentionally construct the resolved spec directly without
    running the validator so we can feed the manager plans that exercise
    specific code paths (e.g., symbolic all_generated) in isolation.
    """
    # A raw spec so ``raw`` is not ``None`` — it's echoed through to
    # Phase 4 for reproducibility but the manager itself never looks at
    # it.
    raw_positions: list[int] | str = (
        list(positions) if isinstance(positions, list) else positions
    )
    raw = ActivationStoringSpec(
        request_id=request_id,
        tag=tag,
        hooks={hook: list(layers) for hook, layers in hooks.items()},
        positions=raw_positions,
    )
    return ResolvedActivationStoringSpec(
        request_id_slug=request_id,
        tag_slug=tag,
        hooks={hook: list(layers) for hook, layers in hooks.items()},
        positions=positions,
        position_kind=position_kind,
        estimated_bytes=0,
        raw=raw,
    )


def _single_request_batch_view(
    req_id: str,
    num_prompt_tokens: int,
    num_computed_tokens: int,
    num_scheduled_tokens: int,
    token_offset: int = 0,
) -> CaptureBatchView:
    return CaptureBatchView(
        req_ids=[req_id],
        num_prompt_tokens=[num_prompt_tokens],
        num_computed_tokens=[num_computed_tokens],
        num_scheduled_tokens=[num_scheduled_tokens],
        token_offsets=[token_offset],
    )


# ---------------------------------------------------------------------------
# 1. Single-step plan with last_prompt positions
# ---------------------------------------------------------------------------


def test_register_then_build_plan_last_prompt():
    mgr = _make_manager()
    num_prompt = 5
    # last_prompt resolves to [4]
    spec = _make_resolved_spec(
        request_id="r0",
        tag="t",
        hooks={"post_mlp": [12]},
        positions=[num_prompt - 1],
        position_kind="last_prompt",
    )
    mgr.register_request("r0", spec, num_prompt_tokens=num_prompt)

    batch_view = _single_request_batch_view(
        req_id="r0",
        num_prompt_tokens=num_prompt,
        num_computed_tokens=0,
        num_scheduled_tokens=num_prompt,
        token_offset=0,
    )
    plan = mgr.build_step_plan(batch_view)

    assert list(plan.gather_indices.keys()) == [(12, "post_mlp")]
    # Absolute row for position 4 with token_offset=0 is row 4.
    idx = plan.gather_indices[(12, "post_mlp")]
    assert idx.dtype == torch.int64
    assert idx.tolist() == [4]
    assert plan.scratch_gpu[(12, "post_mlp")].shape == (1, HIDDEN_SIZE)
    assert plan.scratch_dtype[(12, "post_mlp")] == torch.float32
    assert len(plan.entries) == 1
    entry = plan.entries[0]
    assert entry.request_id == "r0"
    assert entry.layer == 12
    assert entry.hook == "post_mlp"
    assert entry.logical_pos == num_prompt - 1
    assert entry.scratch_row == 0
    assert entry.step_index == 0


def test_register_then_build_plan_last_prompt_respects_token_offset():
    """Absolute row = token_offset + (logical_pos - step_start)."""
    mgr = _make_manager()
    num_prompt = 3
    spec = _make_resolved_spec(
        request_id="r0",
        tag="t",
        hooks={"post_mlp": [0]},
        positions=[num_prompt - 1],
        position_kind="last_prompt",
    )
    mgr.register_request("r0", spec, num_prompt_tokens=num_prompt)

    batch_view = _single_request_batch_view(
        req_id="r0",
        num_prompt_tokens=num_prompt,
        num_computed_tokens=0,
        num_scheduled_tokens=num_prompt,
        token_offset=17,
    )
    plan = mgr.build_step_plan(batch_view)
    # Last prompt logical pos = 2, step_start = 0, token_offset = 17
    # => absolute row = 17 + (2 - 0) = 19
    assert plan.gather_indices[(0, "post_mlp")].tolist() == [19]


# ---------------------------------------------------------------------------
# 2. Multiple layers under one hook
# ---------------------------------------------------------------------------


def test_build_plan_multiple_layers_one_hook():
    mgr = _make_manager()
    num_prompt = 4
    spec = _make_resolved_spec(
        request_id="r0",
        tag="t",
        hooks={"post_mlp": [12, 24]},
        positions=[num_prompt - 1],
        position_kind="last_prompt",
    )
    mgr.register_request("r0", spec, num_prompt_tokens=num_prompt)

    batch_view = _single_request_batch_view(
        req_id="r0",
        num_prompt_tokens=num_prompt,
        num_computed_tokens=0,
        num_scheduled_tokens=num_prompt,
    )
    plan = mgr.build_step_plan(batch_view)

    assert set(plan.gather_indices.keys()) == {(12, "post_mlp"), (24, "post_mlp")}
    for key in [(12, "post_mlp"), (24, "post_mlp")]:
        assert plan.gather_indices[key].tolist() == [num_prompt - 1]
        assert plan.scratch_gpu[key].shape == (1, HIDDEN_SIZE)

    # One entry per (layer, hook).
    layers_in_entries = sorted({e.layer for e in plan.entries})
    assert layers_in_entries == [12, 24]
    assert len(plan.entries) == 2


# ---------------------------------------------------------------------------
# 3. Multiple hooks under one layer
# ---------------------------------------------------------------------------


def test_build_plan_multiple_hooks_one_layer():
    mgr = _make_manager()
    num_prompt = 4
    spec = _make_resolved_spec(
        request_id="r0",
        tag="t",
        hooks={"pre_attn": [12], "post_mlp": [12]},
        positions=[num_prompt - 1],
        position_kind="last_prompt",
    )
    mgr.register_request("r0", spec, num_prompt_tokens=num_prompt)

    batch_view = _single_request_batch_view(
        req_id="r0",
        num_prompt_tokens=num_prompt,
        num_computed_tokens=0,
        num_scheduled_tokens=num_prompt,
    )
    plan = mgr.build_step_plan(batch_view)

    assert set(plan.gather_indices.keys()) == {(12, "pre_attn"), (12, "post_mlp")}
    hooks_in_entries = sorted({e.hook for e in plan.entries})
    assert hooks_in_entries == ["post_mlp", "pre_attn"]


# ---------------------------------------------------------------------------
# 4. Position falls outside the step's [num_computed, num_computed+scheduled)
# ---------------------------------------------------------------------------


def test_build_plan_one_request_no_position_in_step():
    """A request targeting logical pos 10 with num_scheduled=5 and
    num_computed=0 produces no entries and no scratch."""
    mgr = _make_manager()
    num_prompt = 20
    spec = _make_resolved_spec(
        request_id="r0",
        tag="t",
        hooks={"post_mlp": [12]},
        positions=[10],  # explicit
        position_kind="explicit",
    )
    mgr.register_request("r0", spec, num_prompt_tokens=num_prompt)

    batch_view = _single_request_batch_view(
        req_id="r0",
        num_prompt_tokens=num_prompt,
        num_computed_tokens=0,
        num_scheduled_tokens=5,
    )
    plan = mgr.build_step_plan(batch_view)
    assert plan.gather_indices == {}
    assert plan.scratch_gpu == {}
    assert plan.entries == []
    # steps_seen should NOT have been bumped for a step with no entries.
    # Walking another step that DOES capture should see step_index=0.
    batch_view2 = _single_request_batch_view(
        req_id="r0",
        num_prompt_tokens=num_prompt,
        num_computed_tokens=10,
        num_scheduled_tokens=5,
    )
    plan2 = mgr.build_step_plan(batch_view2)
    assert len(plan2.entries) == 1
    assert plan2.entries[0].step_index == 0


# ---------------------------------------------------------------------------
# 5. Multi-step accumulation with growing num_computed_tokens
# ---------------------------------------------------------------------------


def test_build_plan_multistep_all_generated():
    """Two consecutive build_step_plan calls grow step_index monotonically
    for a request capturing all_generated."""
    mgr = _make_manager()
    num_prompt = 4
    spec = _make_resolved_spec(
        request_id="r0",
        tag="t",
        hooks={"post_mlp": [12]},
        positions="all_generated",
        position_kind="all_generated",
    )
    mgr.register_request("r0", spec, num_prompt_tokens=num_prompt)

    # Step 1: prefill (4 prompt tokens). all_generated is still empty.
    plan1 = mgr.build_step_plan(
        _single_request_batch_view(
            req_id="r0",
            num_prompt_tokens=num_prompt,
            num_computed_tokens=0,
            num_scheduled_tokens=num_prompt,
        )
    )
    assert plan1.entries == []

    # Step 2: decode token 4 (first generated).
    plan2 = mgr.build_step_plan(
        _single_request_batch_view(
            req_id="r0",
            num_prompt_tokens=num_prompt,
            num_computed_tokens=num_prompt,
            num_scheduled_tokens=1,
        )
    )
    assert len(plan2.entries) == 1
    assert plan2.entries[0].logical_pos == num_prompt
    assert plan2.entries[0].step_index == 0

    # Step 3: decode token 5.
    plan3 = mgr.build_step_plan(
        _single_request_batch_view(
            req_id="r0",
            num_prompt_tokens=num_prompt,
            num_computed_tokens=num_prompt + 1,
            num_scheduled_tokens=1,
        )
    )
    assert len(plan3.entries) == 1
    assert plan3.entries[0].logical_pos == num_prompt + 1
    assert plan3.entries[0].step_index == 1


# ---------------------------------------------------------------------------
# 6. Unregistered request ids in the batch view are skipped silently
# ---------------------------------------------------------------------------


def test_build_plan_skips_unregistered_request():
    mgr = _make_manager()
    num_prompt = 4
    spec = _make_resolved_spec(
        request_id="r0",
        tag="t",
        hooks={"post_mlp": [12]},
        positions=[num_prompt - 1],
        position_kind="last_prompt",
    )
    mgr.register_request("r0", spec, num_prompt_tokens=num_prompt)

    batch_view = CaptureBatchView(
        req_ids=["other", "r0", "another"],
        num_prompt_tokens=[num_prompt, num_prompt, num_prompt],
        num_computed_tokens=[0, 0, 0],
        num_scheduled_tokens=[num_prompt, num_prompt, num_prompt],
        token_offsets=[0, num_prompt, 2 * num_prompt],
    )
    plan = mgr.build_step_plan(batch_view)

    # Only r0 is a capture request, and its scheduled rows start at
    # token_offset=num_prompt, so absolute row = num_prompt + 3 = 7.
    assert plan.gather_indices[(12, "post_mlp")].tolist() == [2 * num_prompt - 1]
    assert len(plan.entries) == 1
    assert plan.entries[0].request_id == "r0"


# ---------------------------------------------------------------------------
# 7. unregister_request clears state
# ---------------------------------------------------------------------------


def test_unregister_clears_request_state():
    mgr = _make_manager()
    num_prompt = 4
    spec = _make_resolved_spec(
        request_id="r0",
        tag="t",
        hooks={"post_mlp": [12]},
        positions=[num_prompt - 1],
        position_kind="last_prompt",
    )
    mgr.register_request("r0", spec, num_prompt_tokens=num_prompt)
    assert mgr.is_active()

    mgr.unregister_request("r0")
    assert not mgr.is_active()

    batch_view = _single_request_batch_view(
        req_id="r0",
        num_prompt_tokens=num_prompt,
        num_computed_tokens=0,
        num_scheduled_tokens=num_prompt,
    )
    plan = mgr.build_step_plan(batch_view)
    assert plan.gather_indices == {}
    assert plan.entries == []


# ---------------------------------------------------------------------------
# 8. on_hook index-select correctness
# ---------------------------------------------------------------------------


def test_on_hook_index_select_correctness():
    mgr = _make_manager()
    # Install a known plan directly, bypassing register + build.
    num_rows = 5
    hidden_states = torch.arange(num_rows * HIDDEN_SIZE, dtype=torch.float32).view(
        num_rows, HIDDEN_SIZE
    )
    rows_to_gather = [1, 3, 4]
    key = (7, "post_mlp")
    plan = StepCapturePlan(
        gather_indices={key: torch.tensor(rows_to_gather, dtype=torch.int64)},
        scratch_gpu={
            key: torch.empty((len(rows_to_gather), HIDDEN_SIZE), dtype=torch.float32)
        },
        scratch_dtype={key: torch.float32},
        entries=[
            CapturePositionEntry(
                request_id="r0",
                layer=7,
                hook="post_mlp",
                logical_pos=i,
                scratch_row=j,
                step_index=0,
            )
            for j, i in enumerate(rows_to_gather)
        ],
    )
    mgr.set_step_plan(plan)

    mgr.on_hook(7, "post_mlp", hidden_states)

    expected = hidden_states.index_select(
        0, torch.tensor(rows_to_gather, dtype=torch.int64)
    )
    assert torch.equal(plan.scratch_gpu[(7, "post_mlp")], expected)


def test_on_hook_skips_layer_hook_pair_with_no_plan():
    mgr = _make_manager()
    hidden_states = torch.ones((3, HIDDEN_SIZE), dtype=torch.float32)
    # Plan only covers (7, post_mlp).
    plan = StepCapturePlan(
        gather_indices={(7, "post_mlp"): torch.tensor([0, 1, 2], dtype=torch.int64)},
        scratch_gpu={
            (7, "post_mlp"): torch.empty((3, HIDDEN_SIZE), dtype=torch.float32)
        },
        scratch_dtype={(7, "post_mlp"): torch.float32},
        entries=[],
    )
    mgr.set_step_plan(plan)

    # Call on_hook with a (layer, hook) that is NOT in the plan.
    mgr.on_hook(99, "pre_attn", hidden_states)

    # Plan is unchanged, no new keys added.
    assert set(plan.gather_indices.keys()) == {(7, "post_mlp")}
    assert set(plan.scratch_gpu.keys()) == {(7, "post_mlp")}


def test_on_hook_casts_to_scratch_dtype():
    mgr = _make_manager(dtype=torch.bfloat16)
    hidden_states = torch.randn((4, HIDDEN_SIZE), dtype=torch.float16)
    plan = StepCapturePlan(
        gather_indices={(0, "post_mlp"): torch.tensor([0, 2], dtype=torch.int64)},
        scratch_gpu={
            (0, "post_mlp"): torch.empty((2, HIDDEN_SIZE), dtype=torch.bfloat16)
        },
        scratch_dtype={(0, "post_mlp"): torch.bfloat16},
        entries=[],
    )
    mgr.set_step_plan(plan)

    mgr.on_hook(0, "post_mlp", hidden_states)

    scratch = plan.scratch_gpu[(0, "post_mlp")]
    assert scratch.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# 11. Request errors surface through plan
# ---------------------------------------------------------------------------


def test_request_errors_surface_through_plan():
    mgr = _make_manager()
    num_prompt = 4
    # Explicit position that is valid so admission passes but step-time
    # resolution for symbolic positions could fail on future steps.
    spec = _make_resolved_spec(
        request_id="r0",
        tag="t",
        hooks={"post_mlp": [12]},
        positions="all_generated",
        position_kind="all_generated",
    )
    mgr.register_request("r0", spec, num_prompt_tokens=num_prompt)

    # Inject an internal error to exercise the plumbing without depending
    # on a real resolver failure.
    mgr._requests["r0"].error = "synthetic test error"

    plan = mgr.build_step_plan(
        _single_request_batch_view(
            req_id="r0",
            num_prompt_tokens=num_prompt,
            num_computed_tokens=0,
            num_scheduled_tokens=num_prompt,
        )
    )
    assert plan.request_errors == {"r0": "synthetic test error"}
    assert plan.entries == []
    assert plan.gather_indices == {}

    # consume_step_plan returns the full plan including the errors.
    consumed = mgr.consume_step_plan()
    assert consumed is plan
    assert consumed.request_errors == {"r0": "synthetic test error"}
    # And clears the manager's view of the plan.
    assert mgr.consume_step_plan() is None


# ---------------------------------------------------------------------------
# Extra: manager validation failures
# ---------------------------------------------------------------------------


def test_register_request_rejects_out_of_range_layer():
    mgr = _make_manager(num_hidden_layers=16)
    spec = _make_resolved_spec(
        request_id="r0",
        tag="t",
        hooks={"post_mlp": [999]},
        positions=[0],
        position_kind="explicit",
    )
    with pytest.raises(ValueError, match="out of range"):
        mgr.register_request("r0", spec, num_prompt_tokens=4)


def test_register_request_rejects_duplicate():
    mgr = _make_manager()
    spec = _make_resolved_spec(
        request_id="r0",
        tag="t",
        hooks={"post_mlp": [0]},
        positions=[0],
        position_kind="explicit",
    )
    mgr.register_request("r0", spec, num_prompt_tokens=4)
    with pytest.raises(ValueError, match="already registered"):
        mgr.register_request("r0", spec, num_prompt_tokens=4)
