# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixin-level tests for per-role steering.

PR 5a wires up a second ``SteeringManager`` for the speculative-
decoding draft model and changes ``set_steering_vectors`` /
``clear_steering_vectors`` / ``get_steering_status`` /
``list_steerable_layers`` to accept an optional ``target`` kwarg
(``None`` = tags-along, ``"main"``, ``"draft"``). These tests
exercise that dispatch without a real spec-decode setup — the mixin
is instrumented with a mock drafter that has a ``.model`` with fake
steerable layers.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vllm.exceptions import SteeringVectorError
from vllm.model_executor.layers.steering import DEFAULT_HOOK_POINT
from vllm.v1.worker.steering_manager import SteeringManager
from vllm.v1.worker.steering_model_runner_mixin import (
    SteeringModelRunnerMixin,
    _RoleState,
)

_HP = DEFAULT_HOOK_POINT.value


class _FakeLayer(nn.Module):
    def __init__(self, layer_idx: int, hidden_size: int, max_configs: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.register_buffer(
            "steering_table_post_mlp",
            torch.zeros(max_configs + 2, hidden_size),
            persistent=False,
        )
        self.register_buffer(
            "steering_index",
            torch.zeros(16, dtype=torch.long),
            persistent=False,
        )


class _FakeModel(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        max_configs: int = 2,
        layer_offset: int = 0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _FakeLayer(layer_offset + i, hidden_size, max_configs)
                for i in range(num_layers)
            ]
        )


class _FakeDrafter:
    """Minimal drafter stub exposing ``.model`` like SpecDecodeBaseProposer."""

    def __init__(self, model: nn.Module | None):
        self.model = model


class _FakeRunner(SteeringModelRunnerMixin):
    """Minimal runner that exposes ``get_model()`` and ``self.drafter``."""

    def __init__(
        self,
        main_model: nn.Module,
        draft_model: nn.Module | None = None,
    ):
        self._main_model = main_model
        self.drafter = _FakeDrafter(draft_model) if draft_model is not None else None

    def get_model(self) -> nn.Module:
        return self._main_model


@pytest.fixture
def runner_main_only():
    main = _FakeModel(num_layers=2, hidden_size=8)
    r = _FakeRunner(main_model=main)
    # Pretend both managers have been manually set up (no lazy init).
    r._steering_manager = SteeringManager(max_steering_configs=2, device=None)
    r._locally_owned_layers = frozenset({0, 1})
    return r


@pytest.fixture
def runner_with_draft():
    main = _FakeModel(num_layers=2, hidden_size=8)
    draft = _FakeModel(num_layers=2, hidden_size=8)
    r = _FakeRunner(main_model=main, draft_model=draft)
    r._steering_manager = SteeringManager(max_steering_configs=2, device=None)
    r._locally_owned_layers = frozenset({0, 1})
    r._draft_steering_manager = SteeringManager(max_steering_configs=2, device=None)
    r._draft_locally_owned_layers = frozenset({0, 1})
    return r


@pytest.fixture
def runner_mismatched_shapes():
    """Draft with different hidden_size — exercises shape-mismatch path."""
    main = _FakeModel(num_layers=2, hidden_size=8)
    draft = _FakeModel(num_layers=2, hidden_size=4)
    r = _FakeRunner(main_model=main, draft_model=draft)
    r._steering_manager = SteeringManager(max_steering_configs=2, device=None)
    r._locally_owned_layers = frozenset({0, 1})
    r._draft_steering_manager = SteeringManager(max_steering_configs=2, device=None)
    r._draft_locally_owned_layers = frozenset({0, 1})
    return r


class TestSelectRoleState:
    def test_main_target(self, runner_with_draft):
        state = runner_with_draft._select_role_state("main")
        assert isinstance(state, _RoleState)
        assert state.role == "main"
        assert state.manager_attr == "_steering_manager"
        assert state.pending_attr == "_pending_steering_globals"
        assert 0 in state.steerable and 1 in state.steerable

    def test_draft_target(self, runner_with_draft):
        state = runner_with_draft._select_role_state("draft")
        assert state.role == "draft"
        assert state.manager_attr == "_draft_steering_manager"
        assert state.pending_attr == "_draft_pending_steering_globals"
        assert 0 in state.steerable and 1 in state.steerable

    def test_draft_target_with_no_draft_model_returns_empty(self, runner_main_only):
        state = runner_main_only._select_role_state("draft")
        assert state.steerable == {}


class TestResolveTargetRoles:
    def test_main_explicit(self, runner_with_draft):
        assert runner_with_draft._resolve_target_roles("main") == ("main",)

    def test_draft_explicit(self, runner_with_draft):
        assert runner_with_draft._resolve_target_roles("draft") == ("draft",)

    def test_tags_along_both_present(self, runner_with_draft):
        assert runner_with_draft._resolve_target_roles(None) == (
            "main",
            "draft",
        )

    def test_tags_along_main_only(self, runner_main_only):
        assert runner_main_only._resolve_target_roles(None) == ("main",)


class TestListSteerableLayers:
    def test_flat_form_main(self, runner_with_draft):
        result = runner_with_draft.list_steerable_layers(target="main")
        assert set(result.keys()) == {0, 1}
        assert result[0] == [_HP]

    def test_flat_form_draft(self, runner_with_draft):
        result = runner_with_draft.list_steerable_layers(target="draft")
        assert set(result.keys()) == {0, 1}

    def test_nested_form_both_roles(self, runner_with_draft):
        result = runner_with_draft.list_steerable_layers()  # target=None
        assert set(result.keys()) == {"main", "draft"}
        assert 0 in result["main"] and 0 in result["draft"]

    def test_nested_form_main_only(self, runner_main_only):
        result = runner_main_only.list_steerable_layers()
        assert set(result.keys()) == {"main"}


class TestSetSteeringVectors:
    def test_main_only_hits_main_manager(self, runner_with_draft):
        vec = [1.0] * 8
        result = runner_with_draft.set_steering_vectors(
            vectors={_HP: {0: vec}}, target="main"
        )
        assert result[2] == [0]
        main_mgr = runner_with_draft._steering_manager
        draft_mgr = runner_with_draft._draft_steering_manager
        assert 0 in main_mgr.global_base_vectors.get(_HP, {})
        assert 0 not in draft_mgr.global_base_vectors.get(_HP, {})

    def test_draft_only_hits_draft_manager(self, runner_with_draft):
        vec = [2.0] * 8
        result = runner_with_draft.set_steering_vectors(
            vectors={_HP: {0: vec}}, target="draft"
        )
        assert result[2] == [0]
        main_mgr = runner_with_draft._steering_manager
        draft_mgr = runner_with_draft._draft_steering_manager
        assert 0 not in main_mgr.global_base_vectors.get(_HP, {})
        assert 0 in draft_mgr.global_base_vectors.get(_HP, {})

    def test_tags_along_hits_both_managers(self, runner_with_draft):
        vec = [3.0] * 8
        result = runner_with_draft.set_steering_vectors(
            vectors={_HP: {0: vec}}
        )  # target=None
        assert result[2] == [0]
        main_mgr = runner_with_draft._steering_manager
        draft_mgr = runner_with_draft._draft_steering_manager
        assert 0 in main_mgr.global_base_vectors.get(_HP, {})
        assert 0 in draft_mgr.global_base_vectors.get(_HP, {})

    def test_tags_along_main_only_when_no_draft(self, runner_main_only):
        vec = [1.0] * 8
        result = runner_main_only.set_steering_vectors(vectors={_HP: {0: vec}})
        assert result[2] == [0]
        # No draft manager → tags-along dropped the draft role silently.

    def test_draft_target_without_draft_raises(self, runner_main_only):
        vec = [1.0] * 8
        with pytest.raises(SteeringVectorError, match="no draft model"):
            runner_main_only.set_steering_vectors(
                vectors={_HP: {0: vec}}, target="draft"
            )

    def test_invalid_target(self, runner_main_only):
        with pytest.raises(SteeringVectorError, match="Invalid target"):
            runner_main_only.set_steering_vectors(
                vectors={_HP: {0: [1.0] * 8}}, target="nonsense"
            )

    def test_tags_along_shape_mismatch_raises(self, runner_mismatched_shapes):
        """Main has hidden_size=8, draft has hidden_size=4.

        Under tags-along, the same vector would be rejected by one
        role's ``_validate_vectors_spec`` (size mismatch). The mixin
        validates per role and surfaces the error.
        """
        vec = [1.0] * 8
        with pytest.raises(SteeringVectorError, match="expected vector of size"):
            runner_mismatched_shapes.set_steering_vectors(vectors={_HP: {0: vec}})

    def test_tags_along_explicit_main_avoids_shape_issue(
        self, runner_mismatched_shapes
    ):
        """Caller can opt into main-only to bypass the mismatch."""
        vec = [1.0] * 8
        result = runner_mismatched_shapes.set_steering_vectors(
            vectors={_HP: {0: vec}}, target="main"
        )
        assert result[2] == [0]


class TestClearSteeringVectors:
    def test_clear_main_only_leaves_draft_untouched(self, runner_with_draft):
        runner_with_draft.set_steering_vectors(
            vectors={_HP: {0: [1.0] * 8, 1: [2.0] * 8}}
        )
        runner_with_draft.clear_steering_vectors(target="main")
        main_mgr = runner_with_draft._steering_manager
        draft_mgr = runner_with_draft._draft_steering_manager
        assert not main_mgr.global_base_vectors
        assert draft_mgr.global_base_vectors.get(_HP, {})

    def test_clear_draft_only_leaves_main_untouched(self, runner_with_draft):
        runner_with_draft.set_steering_vectors(
            vectors={_HP: {0: [1.0] * 8, 1: [2.0] * 8}}
        )
        runner_with_draft.clear_steering_vectors(target="draft")
        main_mgr = runner_with_draft._steering_manager
        draft_mgr = runner_with_draft._draft_steering_manager
        assert main_mgr.global_base_vectors.get(_HP, {})
        assert not draft_mgr.global_base_vectors

    def test_clear_tags_along_hits_both(self, runner_with_draft):
        runner_with_draft.set_steering_vectors(vectors={_HP: {0: [1.0] * 8}})
        runner_with_draft.clear_steering_vectors()
        main_mgr = runner_with_draft._steering_manager
        draft_mgr = runner_with_draft._draft_steering_manager
        assert not main_mgr.global_base_vectors
        assert not draft_mgr.global_base_vectors

    def test_clear_draft_without_draft_raises(self, runner_main_only):
        with pytest.raises(SteeringVectorError, match="no draft model"):
            runner_main_only.clear_steering_vectors(target="draft")


class TestGetSteeringStatus:
    def test_main_only_reports_main(self, runner_with_draft):
        runner_with_draft.set_steering_vectors(
            vectors={_HP: {0: [1.0] * 8}}, target="main"
        )
        status = runner_with_draft.get_steering_status(target="main")
        assert 0 in status

    def test_draft_only_reports_draft(self, runner_with_draft):
        runner_with_draft.set_steering_vectors(
            vectors={_HP: {0: [2.0] * 8}}, target="draft"
        )
        status = runner_with_draft.get_steering_status(target="draft")
        assert 0 in status

    def test_tags_along_merges(self, runner_with_draft):
        runner_with_draft.set_steering_vectors(
            vectors={_HP: {0: [1.0] * 8}}, target="main"
        )
        runner_with_draft.set_steering_vectors(
            vectors={_HP: {1: [2.0] * 8}}, target="draft"
        )
        status = runner_with_draft.get_steering_status()  # target=None
        assert set(status.keys()) == {0, 1}

    def test_draft_without_draft_returns_empty(self, runner_main_only):
        assert runner_main_only.get_steering_status(target="draft") == {}
