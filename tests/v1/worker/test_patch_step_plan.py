# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the activation-patching per-step planner and buffer writer.

These exercise the runner-agnostic control plane without a real model runner:

* :func:`build_patch_step_plan` — the pure address map
  ``abs_row = token_offset + (dest_pos - num_computed)`` under chunked prefill,
  decode positions, mixed batches, multi-site requests, PP layer ownership, and
  strict pool overflow.
* ``PatchModelRunnerMixin._write_patch_step`` — writing the plan into real
  per-(layer, hook) buffers (via ``register_patch_buffers``), the passthrough
  default, and the dirty-cleanup of sites no longer patched.
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.patch import (
    PATCH_ANY_ACTIVE_ATTR,
    PATCH_INDEX_ATTR,
    PATCH_TABLE_ATTR,
    apply_patch,
    register_patch_buffers,
)
from vllm.model_executor.layers.steering import SteeringHookPoint
from vllm.v1.worker.patch_runner_mixin import (
    PatchEntry,
    PatchModelRunnerMixin,
    PatchPoolOverflow,
    _PatchBatchView,
    build_patch_step_plan,
)

POST_BLOCK = SteeringHookPoint.POST_BLOCK
PRE_ATTN = SteeringHookPoint.PRE_ATTN


def _vec(h: int, fill: float) -> torch.Tensor:
    return torch.full((h,), fill)


def _ones(h: int) -> torch.Tensor:
    """Full-replace per-dim alpha row (alpha == 1 on every dim)."""
    return torch.ones(h, dtype=torch.float32)


class TestBuildPatchStepPlan:
    def test_prefill_single_site(self):
        h = 4
        specs = {
            "r0": [
                PatchEntry(
                    layer=3,
                    hook=POST_BLOCK,
                    dest_pos=2,
                    source=_vec(h, 1.0),
                    alpha_row=_ones(h),
                )
            ]
        }
        plan = build_patch_step_plan(
            req_ids=["r0"],
            num_computed=[0],
            num_scheduled=[5],
            token_offsets=[0],
            specs=specs,
            local_layers=frozenset({3}),
            max_patch_slots=8,
        )
        assert set(plan) == {(3, POST_BLOCK)}
        site = plan[(3, POST_BLOCK)]
        assert site.abs_rows == [2]  # offset 0 + (2 - 0)
        assert len(site.alpha_rows) == 1
        assert torch.allclose(site.alpha_rows[0], _ones(h))

    def test_chunked_prefill_only_fires_in_window(self):
        h = 4
        entry = PatchEntry(
            layer=1,
            hook=POST_BLOCK,
            dest_pos=10,
            source=_vec(h, 1.0),
            alpha_row=_ones(h),
        )
        # Chunk 1: positions [0, 8) -> not yet.
        plan1 = build_patch_step_plan(
            req_ids=["r0"],
            num_computed=[0],
            num_scheduled=[8],
            token_offsets=[0],
            specs={"r0": [entry]},
            local_layers=frozenset({1}),
            max_patch_slots=8,
        )
        assert plan1 == {}
        # Chunk 2: positions [8, 16) -> fires at abs_row = 0 + (10 - 8) = 2.
        plan2 = build_patch_step_plan(
            req_ids=["r0"],
            num_computed=[8],
            num_scheduled=[8],
            token_offsets=[0],
            specs={"r0": [entry]},
            local_layers=frozenset({1}),
            max_patch_slots=8,
        )
        assert plan2[(1, POST_BLOCK)].abs_rows == [2]

    def test_decode_position(self):
        h = 4
        # Decode step: num_computed == dest_pos, num_scheduled == 1.
        entry = PatchEntry(
            layer=2,
            hook=POST_BLOCK,
            dest_pos=20,
            source=_vec(h, 1.0),
            alpha_row=_ones(h),
        )
        plan = build_patch_step_plan(
            req_ids=["r0"],
            num_computed=[20],
            num_scheduled=[1],
            token_offsets=[5],  # decode row sits at flat offset 5
            specs={"r0": [entry]},
            local_layers=frozenset({2}),
            max_patch_slots=8,
        )
        assert plan[(2, POST_BLOCK)].abs_rows == [5]

    def test_mixed_batch_only_patched_requests(self):
        h = 4
        specs = {
            "r1": [
                PatchEntry(
                    layer=0,
                    hook=POST_BLOCK,
                    dest_pos=1,
                    source=_vec(h, 1.0),
                    alpha_row=_ones(h),
                )
            ]
        }
        # r0 and r2 have no spec; r1 is patched at its position 1.
        plan = build_patch_step_plan(
            req_ids=["r0", "r1", "r2"],
            num_computed=[0, 0, 0],
            num_scheduled=[3, 3, 3],
            token_offsets=[0, 3, 6],
            specs=specs,
            local_layers=frozenset({0}),
            max_patch_slots=8,
        )
        assert plan[(0, POST_BLOCK)].abs_rows == [3 + 1]  # r1 offset 3 + pos 1

    def test_multi_site_same_request(self):
        h = 4
        specs = {
            "r0": [
                PatchEntry(
                    layer=0,
                    hook=POST_BLOCK,
                    dest_pos=1,
                    source=_vec(h, 1.0),
                    alpha_row=_ones(h),
                ),
                PatchEntry(
                    layer=0,
                    hook=POST_BLOCK,
                    dest_pos=2,
                    source=_vec(h, 2.0),
                    alpha_row=_ones(h),
                ),
                PatchEntry(
                    layer=1,
                    hook=PRE_ATTN,
                    dest_pos=1,
                    source=_vec(h, 3.0),
                    alpha_row=_ones(h),
                ),
            ]
        }
        plan = build_patch_step_plan(
            req_ids=["r0"],
            num_computed=[0],
            num_scheduled=[4],
            token_offsets=[0],
            specs=specs,
            local_layers=frozenset({0, 1}),
            max_patch_slots=8,
        )
        assert plan[(0, POST_BLOCK)].abs_rows == [1, 2]  # two slots, same site
        assert len(plan[(0, POST_BLOCK)].alpha_rows) == 2
        assert all(
            torch.allclose(a, _ones(h)) for a in plan[(0, POST_BLOCK)].alpha_rows
        )
        assert plan[(1, PRE_ATTN)].abs_rows == [1]

    def test_non_local_layer_skipped(self):
        h = 4
        specs = {
            "r0": [
                PatchEntry(
                    layer=5,
                    hook=POST_BLOCK,
                    dest_pos=0,
                    source=_vec(h, 1.0),
                    alpha_row=_ones(h),
                )
            ]
        }
        # Layer 5 is owned by another PP rank -> skipped here.
        plan = build_patch_step_plan(
            req_ids=["r0"],
            num_computed=[0],
            num_scheduled=[2],
            token_offsets=[0],
            specs=specs,
            local_layers=frozenset({0, 1, 2}),
            max_patch_slots=8,
        )
        assert plan == {}

    def test_strict_overflow_raises(self):
        h = 4
        # 3 positions patched at one site, but pool holds only 2 usable slots
        # (max_patch_slots=3 -> slots 1,2 usable).
        specs = {
            "r0": [
                PatchEntry(
                    layer=0,
                    hook=POST_BLOCK,
                    dest_pos=p,
                    source=_vec(h, 1.0),
                    alpha_row=_ones(h),
                )
                for p in range(3)
            ]
        }
        with pytest.raises(PatchPoolOverflow):
            build_patch_step_plan(
                req_ids=["r0"],
                num_computed=[0],
                num_scheduled=[4],
                token_offsets=[0],
                specs=specs,
                local_layers=frozenset({0}),
                max_patch_slots=3,
            )

    def test_scheduler_reservation_matches_worker_capacity(self):
        # The scheduler backpressure reserves against
        # PatchConfig.usable_slots; the worker's step plan must accept exactly
        # that many same-site patches and raise at one more. An off-by-one
        # between the two admits a step the worker then kills the engine over.
        from vllm.config.patch import PatchConfig

        h = 4
        for max_slots in (2, 3, 8, 64):
            usable = PatchConfig(max_patch_slots=max_slots).usable_slots
            assert usable == max_slots - 1

            def _plan(n: int, max_slots: int = max_slots):
                specs = {
                    "r0": [
                        PatchEntry(
                            layer=0,
                            hook=POST_BLOCK,
                            dest_pos=p,
                            source=_vec(h, 1.0),
                            alpha_row=_ones(h),
                        )
                        for p in range(n)
                    ]
                }
                return build_patch_step_plan(
                    req_ids=["r0"],
                    num_computed=[0],
                    num_scheduled=[n + 1],
                    token_offsets=[0],
                    specs=specs,
                    local_layers=frozenset({0}),
                    max_patch_slots=max_slots,
                )

            plan = _plan(usable)
            assert len(plan[(0, POST_BLOCK)].abs_rows) == usable
            with pytest.raises(PatchPoolOverflow):
                _plan(usable + 1)


class _FakeRunner(PatchModelRunnerMixin):
    """Minimal harness: register patch buffers on a few fake layers."""

    def __init__(self, n_layers: int, hidden: int, max_slots: int, max_tokens: int):
        self._patch_specs: dict[str, list[PatchEntry]] = {}
        self._patch_touched_sites: set[tuple[int, SteeringHookPoint]] = set()
        self._patch_index_dirty = False
        self._patchable_layers = {}
        for li in range(n_layers):
            mod = nn.Module()
            mod.layer_idx = li
            register_patch_buffers(
                mod,
                hidden,
                max_patch_tokens=max_tokens,
                max_patch_slots=max_slots,
                dtype=torch.float32,
            )
            self._patchable_layers[li] = mod
        self._locally_owned_patch_layers = frozenset(self._patchable_layers)
        self._patch_max_slots = max_slots


def _apply_at(runner, layer, hook, hidden):
    mod = runner._patchable_layers[layer]
    return apply_patch(
        hidden,
        getattr(mod, PATCH_TABLE_ATTR[hook]),
        getattr(mod, PATCH_INDEX_ATTR[hook]),
        getattr(mod, "patch_alpha_" + hook.value),
        getattr(mod, PATCH_ANY_ACTIVE_ATTR[hook]),
    )


class TestWritePatchStep:
    def test_write_then_apply_replaces_only_target_row(self):
        h, n_tok = 4, 6
        runner = _FakeRunner(n_layers=2, hidden=h, max_slots=8, max_tokens=64)
        src = _vec(h, 9.0)
        runner._patch_specs = {
            "r0": [
                PatchEntry(
                    layer=1,
                    hook=POST_BLOCK,
                    dest_pos=2,
                    source=src,
                    alpha_row=_ones(h),
                )
            ]
        }
        view = _PatchBatchView(
            req_ids=["r0"],
            num_computed=[0],
            num_scheduled=[n_tok],
            token_offsets=[0],
        )
        runner._write_patch_step(view)

        # any_active set on the touched site only.
        mod1 = runner._patchable_layers[1]
        assert bool(getattr(mod1, PATCH_ANY_ACTIVE_ATTR[POST_BLOCK])[0])
        mod0 = runner._patchable_layers[0]
        assert not bool(getattr(mod0, PATCH_ANY_ACTIVE_ATTR[POST_BLOCK])[0])

        hidden = torch.arange(n_tok * h, dtype=torch.float32).reshape(n_tok, h)
        out = _apply_at(runner, 1, POST_BLOCK, hidden)
        # Row 2 replaced with source; all others unchanged.
        for r in range(n_tok):
            if r == 2:
                assert torch.allclose(out[r], src)
            else:
                assert torch.allclose(out[r], hidden[r])

    def test_dirty_cleanup_when_request_finishes(self):
        h, n_tok = 4, 4
        runner = _FakeRunner(n_layers=1, hidden=h, max_slots=8, max_tokens=64)
        runner._patch_specs = {
            "r0": [
                PatchEntry(
                    layer=0,
                    hook=POST_BLOCK,
                    dest_pos=1,
                    source=_vec(h, 5.0),
                    alpha_row=_ones(h),
                )
            ]
        }
        view = _PatchBatchView(
            req_ids=["r0"], num_computed=[0], num_scheduled=[n_tok], token_offsets=[0]
        )
        runner._write_patch_step(view)
        mod = runner._patchable_layers[0]
        assert bool(getattr(mod, PATCH_ANY_ACTIVE_ATTR[POST_BLOCK])[0])

        # Request finishes; next step has no patch -> site cleared.
        runner._patch_finish_requests(["r0"])
        empty_view = _PatchBatchView(
            req_ids=["r0"], num_computed=[1], num_scheduled=[1], token_offsets=[0]
        )
        runner._write_patch_step(empty_view)
        assert not bool(getattr(mod, PATCH_ANY_ACTIVE_ATTR[POST_BLOCK])[0])
        hidden = torch.randn(n_tok, h)
        out = _apply_at(runner, 0, POST_BLOCK, hidden)
        assert torch.allclose(out, hidden)

    def test_site_moves_between_steps_clears_old(self):
        """A request patching layer 0 then (after a re-add) layer 1 must not
        leave layer 0 active."""
        h, n_tok = 4, 4
        runner = _FakeRunner(n_layers=2, hidden=h, max_slots=8, max_tokens=64)
        runner._patch_specs = {
            "r0": [
                PatchEntry(
                    layer=0,
                    hook=POST_BLOCK,
                    dest_pos=1,
                    source=_vec(h, 5.0),
                    alpha_row=_ones(h),
                )
            ]
        }
        view = _PatchBatchView(
            req_ids=["r0"], num_computed=[0], num_scheduled=[n_tok], token_offsets=[0]
        )
        runner._write_patch_step(view)
        assert (0, POST_BLOCK) in runner._patch_touched_sites

        runner._patch_specs = {
            "r0": [
                PatchEntry(
                    layer=1,
                    hook=POST_BLOCK,
                    dest_pos=1,
                    source=_vec(h, 5.0),
                    alpha_row=_ones(h),
                )
            ]
        }
        runner._write_patch_step(view)
        mod0 = runner._patchable_layers[0]
        mod1 = runner._patchable_layers[1]
        assert not bool(getattr(mod0, PATCH_ANY_ACTIVE_ATTR[POST_BLOCK])[0])
        assert bool(getattr(mod1, PATCH_ANY_ACTIVE_ATTR[POST_BLOCK])[0])
