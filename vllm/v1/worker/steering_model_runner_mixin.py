# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define activation steering functionality mixin for model runners.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.config.steering_types import MODEL_ROLES, ModelRole
from vllm.exceptions import SteeringVectorError
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
)
from vllm.v1.worker.steering_manager import SteeringManager


@dataclass(frozen=True)
class _RoleState:
    """Per-role state descriptor returned by ``_select_role_state``.

    Groups the four pieces the public steering methods need so each
    method's role-dispatch loop stays readable. ``manager_attr`` and
    ``pending_attr`` are attribute names on the mixin; callers read
    them with ``getattr`` to preserve the lazy-init sentinel (the
    mixin uses ``hasattr(self, "_steering_manager")`` as the trigger,
    so class-level defaults would defeat initialisation).
    """

    role: ModelRole
    steerable: dict
    manager_attr: str
    pending_attr: str
    locally_owned: frozenset[int] | None


def _get_steering_ranks() -> tuple[int, int]:
    """Return ``(tp_rank, pp_rank)`` for the current worker.

    Used to tag steering RPC results so the router can detect TP
    divergence (a server-side invariant violation). Guarded so that
    tests / single-rank setups that haven't initialized the
    distributed groups still work.
    """
    try:
        from vllm.distributed.parallel_state import (
            get_pp_group,
            get_tp_group,
        )

        return (get_tp_group().rank_in_group, get_pp_group().rank_in_group)
    except Exception:
        return (0, 0)


if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)


# Defined as a mixin for GPUModelRunner
class SteeringModelRunnerMixin:
    """Consolidates all activation-steering state and logic on the model runner.

    Mirrors the ``LoRAModelRunnerMixin`` pattern: the mixin owns every
    piece of steering-related state and exposes the public API
    (``set_steering_vectors``, ``clear_steering_vectors``,
    ``list_steerable_layers``, ``get_steering_status``) that
    ``WorkerBase`` and its concrete subclasses delegate to via thin
    passthroughs.
    """

    # --- class-level attribute declarations --------------------------------
    # These back the mixin's stateful methods.  Attributes that are
    # unconditionally read (possibly before lazy init) have
    # class-level defaults so the mixin can use plain attribute
    # access without ``hasattr``/``getattr`` guards.
    #
    # ``_steering_manager`` intentionally does NOT have a default:
    # ``_update_steering_buffers`` uses ``hasattr(self, "_steering_manager")``
    # as the lazy-init trigger, so assigning a class-level default would
    # make initialisation skip permanently.
    _steerable_layers_cache: dict[int, nn.Module] | None = None
    _pending_steering_globals: (
        list[tuple[dict[str, dict[int, torch.Tensor]], str]] | None
    ) = None
    # The attributes below are populated by the lazy init in
    # ``_update_steering_buffers`` and are only read after that path
    # has run.  Test fixtures that exercise the mixin in isolation
    # must set them explicitly.
    _steering_manager: SteeringManager | None
    _pending_steering_transitions: list[
        tuple[str, int, dict[str, dict[int, list[float]]], str]
    ]
    _pending_steering_registrations: list[
        tuple[str, int, dict[str, dict[int, list[float]]], str]
    ]
    _req_steering_phase: dict[str, str]
    _steering_index_dirty: bool
    # Set of layer indices physically owned by this worker.  Under PP,
    # this is a contiguous subset of ``[0, num_layers)``; under single-
    # worker and under TP (which replicates all layers per rank), it
    # equals the full model's layer set.  Populated during lazy init
    # from ``_steerable_layers_cache`` and threaded into
    # ``SteeringManager`` calls so non-local tensors are never
    # materialized on this rank.
    _locally_owned_layers: frozenset[int]

    # --- Draft-model ("draft" role) state --------------------------------
    # A second SteeringManager and layer cache for the speculative-decoding
    # draft model. ``None`` when spec decoding is disabled, when the
    # drafter is an n-gram proposer (no nn.Module), or when the draft
    # model's architecture does not register steering buffers. Under the
    # same determinism contract as the main manager: every rank derives
    # identical state from identical ``collective_rpc`` calls.
    _draft_steering_manager: SteeringManager | None = None
    _draft_steerable_layers_cache: dict[int, nn.Module] | None = None
    _draft_locally_owned_layers: frozenset[int] = frozenset()
    _draft_pending_steering_globals: (
        list[tuple[dict[str, dict[int, torch.Tensor]], str]] | None
    ) = None

    # Attributes provided by the concrete model runner that mixes this
    # class in.  Declared here purely so static type checking can see
    # them — there is no runtime assignment.
    if TYPE_CHECKING:
        vllm_config: VllmConfig
        input_batch: InputBatch
        requests: dict[str, CachedRequestState]

        def get_model(self) -> nn.Module: ...

    # -----------------------------------------------------------------------
    # Steerable-layer discovery and vector-spec validation
    # -----------------------------------------------------------------------

    def _steerable_layers(self) -> dict:
        """Return ``{layer_idx: module}`` for layers with steering buffers.

        Works with any model runner that exposes ``get_model()``,
        including the V2 runner.  Result is cached after first
        successful discovery.

        A layer is considered steerable if it has ``layer_idx`` and at
        least one ``steering_table_*`` buffer for any hook point.
        """
        cache = self._steerable_layers_cache
        if cache is not None:
            return cache

        if not hasattr(self, "get_model"):
            return {}
        layers: dict = {}
        for mod in self.get_model().modules():
            if not hasattr(mod, "layer_idx"):
                continue
            has_any_table = any(
                hasattr(mod, attr) for attr in HOOK_POINT_TABLE_ATTR.values()
            )
            if has_any_table:
                layers[mod.layer_idx] = mod

        if layers:
            self._steerable_layers_cache = layers

        return layers

    def _get_drafter_model(self) -> "nn.Module | None":
        """Return the speculative-decoding draft ``nn.Module`` if one exists.

        The draft model lives on the concrete model runner as
        ``self.drafter`` (a :class:`SpecDecodeBaseProposer` subclass —
        ``DraftModelProposer``, ``EagleProposer``, ``MedusaProposer``,
        etc.) which exposes the loaded draft as ``self.drafter.model``.
        ``NgramProposer`` / suffix-decoding proposers do not carry a
        neural model and return ``None`` here.
        """
        drafter = getattr(self, "drafter", None)
        if drafter is None:
            return None
        return getattr(drafter, "model", None)

    def _draft_steerable_layers(self) -> dict:
        """Like :meth:`_steerable_layers` but over the draft model.

        Returns ``{}`` when no draft model is present or when the
        draft's architecture does not register steering buffers.
        """
        cache = self._draft_steerable_layers_cache
        if cache is not None:
            return cache

        drafter_model = self._get_drafter_model()
        if drafter_model is None:
            return {}
        layers: dict = {}
        for mod in drafter_model.modules():
            if not hasattr(mod, "layer_idx"):
                continue
            has_any_table = any(
                hasattr(mod, attr) for attr in HOOK_POINT_TABLE_ATTR.values()
            )
            if has_any_table:
                layers[mod.layer_idx] = mod

        if layers:
            self._draft_steerable_layers_cache = layers

        return layers

    def _select_role_state(self, target: ModelRole) -> _RoleState:
        """Return the per-role state descriptor for *target*.

        The manager itself is not returned — the caller reads it via
        ``getattr(self, state.manager_attr, None)`` to preserve the
        lazy-init sentinel. Callers that mutate pending globals read
        / write through ``state.pending_attr``.
        """
        if target == "draft":
            return _RoleState(
                role="draft",
                steerable=self._draft_steerable_layers(),
                manager_attr="_draft_steering_manager",
                pending_attr="_draft_pending_steering_globals",
                locally_owned=getattr(self, "_draft_locally_owned_layers", frozenset()),
            )
        return _RoleState(
            role="main",
            steerable=self._steerable_layers(),
            manager_attr="_steering_manager",
            pending_attr="_pending_steering_globals",
            locally_owned=getattr(self, "_locally_owned_layers", None),
        )

    def _resolve_target_roles(self, target: ModelRole | None) -> tuple[ModelRole, ...]:
        """Expand ``target`` into the sequence of roles to dispatch to.

        - ``"main"`` / ``"draft"`` → that role only.
        - ``None`` (tags-along) → every role that actually has
          steerable layers on this worker. Roles with no steerable
          layers are dropped silently so a flat-spec request never
          fails on a worker that happens to not host a draft model.
        """
        if target == "draft":
            return ("draft",)
        if target == "main":
            return ("main",)
        # Tags-along: drop roles with no steerable layers.
        out: list[ModelRole] = []
        for role in MODEL_ROLES:
            if self._select_role_state(role).steerable:
                out.append(role)
        return tuple(out) if out else ("main",)

    def _validate_vectors_spec(
        self,
        vectors_data: dict[str, dict[int, list[float]]],
        steerable: dict,
    ) -> set[int]:
        """Validate hook-point / layer / vector combinations.

        Returns the set of valid layer indices on this worker.
        Raises ``SteeringVectorError`` on invalid hook points,
        mismatched sizes, or non-finite values.
        """
        valid_indices: set[int] = set()
        for hook_point_str, layer_vecs in vectors_data.items():
            try:
                hp_enum = SteeringHookPoint(hook_point_str)
            except ValueError as exc:
                raise SteeringVectorError(
                    f"Invalid hook point: {hook_point_str!r}"
                ) from exc
            table_attr = HOOK_POINT_TABLE_ATTR[hp_enum]

            for idx, vec_values in layer_vecs.items():
                if idx not in steerable:
                    continue
                mod = steerable[idx]
                if not hasattr(mod, table_attr):
                    raise SteeringVectorError(
                        f"Hook point {hook_point_str!r} not active on layer {idx}"
                    )
                buf = getattr(mod, table_attr)
                expected_size = buf.shape[1]
                if len(vec_values) != expected_size:
                    raise SteeringVectorError(
                        f"Layer {idx} ({hook_point_str}): expected "
                        f"vector of size {expected_size}, "
                        f"got {len(vec_values)}"
                    )
                if not all(math.isfinite(v) for v in vec_values):
                    raise SteeringVectorError(
                        f"Layer {idx} ({hook_point_str}): steering "
                        f"vector contains non-finite values "
                        f"(NaN or Infinity)"
                    )
                valid_indices.add(idx)
        return valid_indices

    def list_steerable_layers(
        self,
        target: ModelRole | None = None,
    ) -> dict[int, list[str]] | dict[ModelRole, dict[int, list[str]]]:
        """Return steerable layers on this worker with their hook points.

        *target* selects the return shape:

        - ``"main"`` or ``"draft"`` → flat ``{layer_idx: [hook_name, ...]}``
          for that role. Empty dict if the role has no steerable layers.
        - ``None`` → nested ``{"main": {...}, "draft": {...}}``.
          Roles without steerable layers are omitted.

        Hook-point names are sorted for determinism. The router uses
        the flat form for per-role queries and the nested form to
        validate tags-along requests (shape / hook-point mismatch) in
        a single RPC.
        """
        if target is not None:
            layers = self._select_role_state(target).steerable
            return self._describe_layers(layers)
        nested: dict[ModelRole, dict[int, list[str]]] = {}
        for role in MODEL_ROLES:
            described = self._describe_layers(self._select_role_state(role).steerable)
            if described:
                nested[role] = described
        return nested

    @staticmethod
    def _describe_layers(layers: dict) -> dict[int, list[str]]:
        """Map ``{layer_idx: Module}`` → ``{layer_idx: sorted hook names}``."""
        result: dict[int, list[str]] = {}
        for idx, mod in layers.items():
            result[idx] = sorted(
                hp.value
                for hp, attr in HOOK_POINT_TABLE_ATTR.items()
                if hasattr(mod, attr)
            )
        return result

    def _notify_manager_vectors(
        self,
        state: _RoleState,
        vectors_data: dict[str, dict[int, list[float]]],
        valid_indices: set[int],
        phase: str,
    ) -> None:
        """Notify a role's SteeringManager of global vector changes.

        Converts the raw ``list[float]`` values from *vectors_data*
        into tensors matching the layer buffer dtype/device, then passes
        them to the manager for *state.role*. When the manager has not
        been lazily initialized yet, the converted tensors are stored in
        ``state.pending_attr`` for replay during lazy init in
        ``_update_steering_buffers``.
        """
        steerable = state.steerable
        # Use getattr so the lazy-init sentinel on the main manager
        # (``hasattr(self, "_steering_manager")``) is preserved.
        mgr = getattr(self, state.manager_attr, None)
        if mgr is None:
            # Manager not yet initialized -- capture current vectors
            # for replay during lazy init.
            captured: dict[str, dict[int, torch.Tensor]] = {}
            for hook_point_str, layer_vecs in vectors_data.items():
                table_attr = HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook_point_str)]
                captured_layers: dict[int, torch.Tensor] = {}
                for idx, vec_values in layer_vecs.items():
                    if idx not in valid_indices or idx not in steerable:
                        continue
                    mod = steerable[idx]
                    if hasattr(mod, table_attr):
                        buf = getattr(mod, table_attr)
                        captured_layers[idx] = torch.tensor(
                            vec_values, dtype=buf.dtype, device=buf.device
                        )
                if captured_layers:
                    captured[hook_point_str] = captured_layers
            if captured:
                pending = getattr(self, state.pending_attr, None)
                if pending is None:
                    pending = []
                    setattr(self, state.pending_attr, pending)
                pending.append((captured, phase))
            return
        locally_owned = state.locally_owned
        for hook_point_str, layer_vecs in vectors_data.items():
            table_attr = HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook_point_str)]
            for idx, vec_values in layer_vecs.items():
                if idx not in valid_indices or idx not in steerable:
                    continue
                mod = steerable[idx]
                if hasattr(mod, table_attr):
                    buf = getattr(mod, table_attr)
                    t = torch.tensor(vec_values, dtype=buf.dtype, device=buf.device)
                    mgr.update_global_vectors(
                        hook_point_str,
                        idx,
                        t,
                        phase=phase,
                        locally_owned_layers=locally_owned,
                    )

    # -----------------------------------------------------------------------
    # Public steering API (mirrored by thin passthroughs on the worker)
    # -----------------------------------------------------------------------

    def set_steering_vectors(
        self,
        vectors: dict[str, dict[int, list[float]]] | None = None,
        prefill_vectors: dict[str, dict[int, list[float]]] | None = None,
        decode_vectors: dict[str, dict[int, list[float]]] | None = None,
        replace: bool = False,
        validate_only: bool = False,
        target: ModelRole | None = None,
    ) -> tuple[int, int, list[int]]:
        """Set activation steering vectors from plain Python data.

        Supports three-tier steering:

        - *vectors*: base vectors applied to both prefill and decode.
          Notified to SteeringManager with ``phase="base"``.
        - *prefill_vectors*: phase-specific vectors for prefill only.
          Notified to SteeringManager with ``phase="prefill"``.
        - *decode_vectors*: phase-specific vectors for decode only.
          Notified to SteeringManager with ``phase="decode"``.

        All vectors should already be in pre-scaled flat-list form
        (the API router normalizes co-located scales before calling
        this method).

        When *replace* is ``True``, all existing vectors across all
        tiers are cleared before applying.

        When *validate_only* is ``True``, vectors are validated
        without being applied.

        *target* selects which model the vectors apply to:

        - ``"main"`` — apply only to the target (primary) model.
        - ``"draft"`` — apply only to the speculative-decoding draft.
          Raises :class:`SteeringVectorError` when no draft model is
          steerable on this worker.
        - ``None`` (tags-along) — apply to every role that has
          steerable layers. Roles without steerable layers are silently
          skipped so a flat request never fails on a worker that
          happens not to host a draft.

        Returns:
            ``(tp_rank, pp_rank, sorted_valid_layers)``. The rank info
            lets the router detect TP-divergence (a server-side
            invariant violation — TP ranks within the same PP stage
            must own identical layer sets). The sorted layer list is
            the union across every dispatched role of layer indices
            actually updated (or *would* be updated when
            *validate_only*) on this worker. The router unions these
            across workers.
        """
        self._validate_target(target)
        tp_rank, pp_rank = _get_steering_ranks()

        # Tags-along drops roles with no steerable layers; explicit
        # "draft" with no draft model is an error.
        if target == "draft" and not self._draft_steerable_layers():
            raise SteeringVectorError(
                "target='draft' requested but no draft model is "
                "steerable on this worker (no speculative-decoding "
                "draft present, or the draft's architecture does not "
                "register steering buffers)."
            )
        roles = self._resolve_target_roles(target)

        # Collect all tiers with data.
        all_tiers: list[tuple[str, dict[str, dict[int, list[float]]]]] = []
        if vectors:
            all_tiers.append(("base", vectors))
        if prefill_vectors:
            all_tiers.append(("prefill", prefill_vectors))
        if decode_vectors:
            all_tiers.append(("decode", decode_vectors))

        if not all_tiers:
            if replace:
                self.clear_steering_vectors(target=target)
            return (tp_rank, pp_rank, [])

        # Validate per role and collect the union of valid indices.
        role_valid: dict[ModelRole, set[int]] = {}
        all_valid: set[int] = set()
        for role in roles:
            state = self._select_role_state(role)
            if not state.steerable:
                continue
            valid: set[int] = set()
            for _phase, tier_data in all_tiers:
                valid.update(self._validate_vectors_spec(tier_data, state.steerable))
            role_valid[role] = valid
            all_valid.update(valid)

        if not all_valid:
            return (tp_rank, pp_rank, [])

        if validate_only:
            return (tp_rank, pp_rank, sorted(all_valid))

        # Clear per role if replacing. Phase-specific vectors go only
        # to the manager, not the shared buffers — writing them would
        # overwrite base values and cause get_steering_status() to
        # report the wrong tier.
        for role in roles:
            state = self._select_role_state(role)
            if not state.steerable:
                continue
            valid = role_valid.get(role, set())
            if replace:
                self._clear_role_vectors(state)
            if vectors:
                self._notify_manager_vectors(state, vectors, valid, "base")
            if prefill_vectors:
                self._notify_manager_vectors(state, prefill_vectors, valid, "prefill")
            if decode_vectors:
                self._notify_manager_vectors(state, decode_vectors, valid, "decode")

        return (tp_rank, pp_rank, sorted(all_valid))

    def clear_steering_vectors(self, target: ModelRole | None = None) -> None:
        """Clear all tiers (base, prefill, decode) in the SteeringManager.

        *target* selects which model's steering state to clear. See
        :meth:`set_steering_vectors` for target semantics.
        """
        self._validate_target(target)
        if target == "draft" and not self._draft_steerable_layers():
            raise SteeringVectorError(
                "target='draft' requested but no draft model is "
                "steerable on this worker."
            )
        for role in self._resolve_target_roles(target):
            state = self._select_role_state(role)
            self._clear_role_vectors(state)

    def _clear_role_vectors(self, state: _RoleState) -> None:
        """Clear one role's manager state and pending queue."""
        mgr = getattr(self, state.manager_attr, None)
        if mgr is not None:
            mgr.clear_global_vectors()
        # Also clear any pending globals queued before manager init,
        # so they are not replayed on lazy initialization.
        setattr(self, state.pending_attr, None)

    def get_steering_status(self, target: ModelRole | None = None) -> dict:
        """Return per-hook-point status for active layers.

        Returns ``{layer_idx: {hook_point: {"norm": float,
        "prefill_norm"?: float, "decode_norm"?: float}}}`` for
        layers/hook-points that have a non-zero steering vector.

        All norms (base, prefill, decode) are read from the
        SteeringManager when it exists, or from the role's pending
        globals queue before manager initialization.

        *target* selects which model's state to inspect. ``None``
        (tags-along) merges main and draft role status — roles
        typically report disjoint layer indices so the merge is lossless.
        """
        self._validate_target(target)
        if target == "draft" and not self._draft_steerable_layers():
            return {}
        result: dict = {}
        for role in self._resolve_target_roles(target):
            state = self._select_role_state(role)
            self._read_role_status_into(state, result)
        return result

    def _read_role_status_into(self, state: _RoleState, result: dict) -> None:
        """Merge one role's live-or-pending global-vector norms into *result*."""
        mgr = getattr(self, state.manager_attr, None)
        if mgr is not None:
            for phase_name, phase_dict in (
                ("base", mgr.global_base_vectors),
                ("prefill", mgr.global_prefill_vectors),
                ("decode", mgr.global_decode_vectors),
            ):
                norm_key = "norm" if phase_name == "base" else f"{phase_name}_norm"
                for hp_str, layer_vecs in phase_dict.items():
                    for layer_idx, vec in layer_vecs.items():
                        norm = vec.norm().item()
                        if norm > 0.0:
                            result.setdefault(layer_idx, {}).setdefault(hp_str, {})[
                                norm_key
                            ] = round(norm, 6)
            return
        pending = getattr(self, state.pending_attr, None)
        if not pending:
            return
        for captured_vectors, phase in pending:
            norm_key = "norm" if phase == "base" else f"{phase}_norm"
            for hp_str, layer_vecs in captured_vectors.items():
                for layer_idx, vec in layer_vecs.items():
                    norm = vec.norm().item()
                    if norm > 0.0:
                        result.setdefault(layer_idx, {}).setdefault(hp_str, {})[
                            norm_key
                        ] = round(norm, 6)

    def _validate_target(self, target: ModelRole | None) -> None:
        if target is not None and target not in MODEL_ROLES:
            raise SteeringVectorError(
                f"Invalid target {target!r}; expected one of "
                f"{list(MODEL_ROLES)!r} or None."
            )

    # -----------------------------------------------------------------------
    # Per-step buffer / index maintenance
    # -----------------------------------------------------------------------

    def _update_steering_buffers(self, scheduler_output: "SchedulerOutput") -> None:
        """Update per-layer steering tables and the shared steering index.

        Lazily initializes the SteeringManager on first call.  Each step:
        1. Populate each layer's per-hook steering_table from the manager
        2. Build the steering_index mapping tokens to table rows
        3. Detect prefill->decode phase transitions and swap configs
        """
        # Stash the scheduler output on the runner so spec-decode
        # proposers (``EagleProposer.propose()`` etc.) can replicate
        # main's token walk on the draft model during their forward
        # pass. See ``_populate_draft_steering_index``.
        self._last_scheduler_output = scheduler_output
        # Short-circuit when steering is disabled.  Steerable models
        # (e.g. Gemma3) unconditionally register per-layer steering_table
        # buffers so the forward path can stay branch-free, but when
        # --enable-steering is off there is no SteeringConfig and no work
        # to do — populating tables and building the index every step is
        # pure overhead.
        if getattr(self.vllm_config, "steering_config", None) is None:
            if not hasattr(self, "_steering_manager"):
                self._steering_manager = None
                self._steerable_layers_cache = {}
            return

        # Lazy init
        if not hasattr(self, "_steering_manager"):
            steerable: dict = {}
            model = self.get_model()
            for mod in model.modules():
                if not hasattr(mod, "layer_idx"):
                    continue
                has_any_table = any(
                    hasattr(mod, attr) for attr in HOOK_POINT_TABLE_ATTR.values()
                )
                if has_any_table:
                    steerable[mod.layer_idx] = mod
            self._steerable_layers_cache = steerable
            # Snapshot the set of layer indices this worker physically
            # owns.  Used to skip tensor materialization for non-local
            # layers when passing vectors into the SteeringManager.
            # Under PP this is a contiguous subset; under TP/single-
            # worker it's the full set.  Row allocation in the manager
            # stays rank-oblivious so row IDs remain identical across
            # ranks — only the stored tensors are filtered.
            self._locally_owned_layers = frozenset(steerable.keys())

            if steerable:
                steering_config = getattr(self.vllm_config, "steering_config", None)
                max_configs = (
                    steering_config.max_steering_configs if steering_config else 0
                )

                # Resolve device from the first steerable layer's table
                # buffer so per-request vectors are allocated on the same
                # device, avoiding CPU->GPU copies each step.
                table_device: torch.device | None = None
                for mod in steerable.values():
                    for attr in HOOK_POINT_TABLE_ATTR.values():
                        if hasattr(mod, attr):
                            table_device = getattr(mod, attr).device
                            break
                    if table_device is not None:
                        break

                self._steering_manager = SteeringManager(
                    max_configs, device=table_device
                )
                # Each entry: (req_id, config_hash, vectors, phase).
                # Transitions (prefill→decode) are retried before new
                # admissions; the transitions queue must be fully
                # drained before any registration entry is attempted.
                self._pending_steering_transitions: list[
                    tuple[str, int, dict[str, dict[int, list[float]]], str]
                ] = []
                self._pending_steering_registrations: list[
                    tuple[str, int, dict[str, dict[int, list[float]]], str]
                ] = []
                self._req_steering_phase: dict[str, str] = {}
                # Tracks whether steering_index has been written with non-zero
                # row references. Used by the no-active-state short-circuit
                # to know if it needs to zero the index on transition.
                self._steering_index_dirty: bool = False

                # Replay any pending phase-specific global vectors that
                # were set via set_steering_vectors() before the manager
                # existed.
                pending = getattr(self, "_pending_steering_globals", None)
                if pending:
                    for captured_vectors, phase in pending:
                        for hook_point_str, layer_vecs in captured_vectors.items():
                            for layer_idx, vec in layer_vecs.items():
                                self._steering_manager.update_global_vectors(
                                    hook_point_str,
                                    layer_idx,
                                    vec,
                                    phase=phase,
                                    locally_owned_layers=(self._locally_owned_layers),
                                )
                    self._pending_steering_globals = None
                # Register any configs that were added to the batch
                # before the manager existed (first-step race).
                for i in range(self.input_batch.num_reqs):
                    rid = self.input_batch.req_ids[i]
                    rs = self.requests.get(rid)
                    if rs is None or rs.sampling_params is None:
                        continue
                    ri = self.input_batch.req_id_to_index.get(rid)
                    if ri is None:
                        continue

                    num_computed = int(self.input_batch.num_computed_tokens_cpu[ri])
                    num_prompt = int(self.input_batch.num_prompt_tokens[ri])

                    if num_computed < num_prompt:
                        # In prefill — register prefill config
                        ph = int(
                            self.input_batch.request_prefill_steering_hash_main[ri]
                        )
                        if ph != 0:
                            eff = rs.sampling_params.effective_prefill_steering
                            if eff:
                                try:
                                    self._steering_manager.register_config(
                                        ph,
                                        eff,
                                        phase="prefill",
                                        locally_owned_layers=(
                                            self._locally_owned_layers
                                        ),
                                    )
                                except RuntimeError:
                                    self._pending_steering_registrations.append(
                                        (rid, ph, eff, "prefill")
                                    )
                                    logger.warning(
                                        "Deferred prefill steering config "
                                        "(hash=%d) during init -- capacity "
                                        "full, will retry next step",
                                        ph,
                                    )
                        self._req_steering_phase[rid] = "prefill"
                    else:
                        # In decode (full prefix-cache hit) — register
                        # decode config
                        dh = int(self.input_batch.request_decode_steering_hash_main[ri])
                        if dh != 0:
                            eff = rs.sampling_params.effective_decode_steering
                            if eff:
                                try:
                                    self._steering_manager.register_config(
                                        dh,
                                        eff,
                                        phase="decode",
                                        locally_owned_layers=(
                                            self._locally_owned_layers
                                        ),
                                    )
                                except RuntimeError:
                                    self._pending_steering_registrations.append(
                                        (rid, dh, eff, "decode")
                                    )
                                    logger.warning(
                                        "Deferred decode steering config "
                                        "(hash=%d) during init -- capacity "
                                        "full, will retry next step",
                                        dh,
                                    )
                        self._req_steering_phase[rid] = "decode"
            else:
                self._steering_manager = None
                self._steerable_layers_cache = {}

            # --- Draft manager lazy init ---------------------------------
            # When speculative decoding is active AND the draft model's
            # architecture registers steering buffers, bring up a parallel
            # SteeringManager for it. Row allocation on the draft manager
            # is independent of main's: each worker's draft manager sees
            # the same collective_rpc calls and SchedulerOutput stream
            # (determinism contract), so rows stay in lock-step per role
            # without needing to agree with main's rows for the same hash.
            draft_steerable = self._draft_steerable_layers()
            if draft_steerable:
                steering_config = getattr(self.vllm_config, "steering_config", None)
                draft_max_configs = (
                    steering_config.max_steering_configs if steering_config else 0
                )
                draft_table_device: torch.device | None = None
                for mod in draft_steerable.values():
                    for attr in HOOK_POINT_TABLE_ATTR.values():
                        if hasattr(mod, attr):
                            draft_table_device = getattr(mod, attr).device
                            break
                    if draft_table_device is not None:
                        break
                self._draft_steering_manager = SteeringManager(
                    draft_max_configs, device=draft_table_device
                )
                self._draft_locally_owned_layers = frozenset(draft_steerable.keys())
                # Replay any pending draft global vectors queued before
                # the manager existed.
                draft_pending = getattr(self, "_draft_pending_steering_globals", None)
                if draft_pending:
                    for captured_vectors, phase in draft_pending:
                        for hook_point_str, layer_vecs in captured_vectors.items():
                            for layer_idx, vec in layer_vecs.items():
                                self._draft_steering_manager.update_global_vectors(
                                    hook_point_str,
                                    layer_idx,
                                    vec,
                                    phase=phase,
                                    locally_owned_layers=(
                                        self._draft_locally_owned_layers
                                    ),
                                )
                    self._draft_pending_steering_globals = None

        if self._steering_manager is None or not self._steerable_layers_cache:
            # Main disabled → draft steering cannot meaningfully run
            # alone either (the main model's forward is the outer loop
            # driving the draft), so short-circuit.
            return

        # Process deferred steering entries with a two-queue priority
        # model.  Transitions (prefill→decode) are drained first because
        # they represent in-flight requests that already consumed KV
        # cache.  New-request registrations are only attempted once the
        # transitions queue is empty.  Entries are dropped when the
        # originating request has finished or changed phase, preventing
        # row leaks.
        if self._pending_steering_transitions:
            still_transitions: list[
                tuple[str, int, dict[str, dict[int, list[float]]], str]
            ] = []
            for (
                d_req_id,
                d_hash,
                d_vecs,
                d_phase,
            ) in self._pending_steering_transitions:
                if d_req_id not in self.requests:
                    continue
                if self._req_steering_phase.get(d_req_id) != d_phase:
                    continue
                try:
                    self._steering_manager.register_config(
                        d_hash,
                        d_vecs,
                        phase=d_phase,
                        locally_owned_layers=self._locally_owned_layers,
                    )
                except RuntimeError:
                    still_transitions.append((d_req_id, d_hash, d_vecs, d_phase))
            self._pending_steering_transitions = still_transitions

        if (
            not self._pending_steering_transitions
            and self._pending_steering_registrations
        ):
            still_pending: list[
                tuple[str, int, dict[str, dict[int, list[float]]], str]
            ] = []
            for (
                d_req_id,
                d_hash,
                d_vecs,
                d_phase,
            ) in self._pending_steering_registrations:
                if d_req_id not in self.requests:
                    continue
                if self._req_steering_phase.get(d_req_id) != d_phase:
                    continue
                try:
                    self._steering_manager.register_config(
                        d_hash,
                        d_vecs,
                        phase=d_phase,
                        locally_owned_layers=self._locally_owned_layers,
                    )
                except RuntimeError:
                    still_pending.append((d_req_id, d_hash, d_vecs, d_phase))
            self._pending_steering_registrations = still_pending

        # Short-circuit when no steering state is actually active. The model
        # runner allocates per-layer steering buffers (zero-initialized) and
        # the forward path always calls apply_steering, but if no per-request
        # configs are registered and no global vectors have been set, every
        # gather hits the zero sentinel and adds nothing. There is nothing
        # to populate.
        #
        # Correctness: when we previously had active steering and now don't
        # (e.g., the last steered request just finished), the steering_index
        # may still contain non-zero row references from the previous step.
        # We must zero it before returning to ensure all gathers point to
        # row 0. We only do this on the transition; in the steady "nothing
        # ever active" case the index is already zero from initialization.
        if (
            not self._steering_manager.config_to_row
            and not self._steering_manager.global_base_vectors
            and not self._steering_manager.global_prefill_vectors
            and not self._steering_manager.global_decode_vectors
        ):
            if self._steering_index_dirty:
                any_layer = next(iter(self._steerable_layers_cache.values()))
                any_layer.steering_index.zero_()
                self._steering_index_dirty = False
            return

        # 1. Populate steering tables — but only if state has changed since
        # the last populate. populate_steering_tables() clears the flag at
        # the end, and every state mutator (register_config new-row,
        # release_config refcount->0, update_global_vectors,
        # clear_global_vectors) sets it. In steady-state decode steps
        # where no config churn happens, this skips ~102 kernel launches
        # per step.
        if self._steering_manager._tables_dirty:
            self._steering_manager.populate_steering_tables(
                self._steerable_layers_cache
            )

        # Populate draft tables when the draft manager exists. Draft's
        # ``_tables_dirty`` flag is independent of main's, so this may
        # or may not trigger a write on any given step. The draft
        # ``steering_index`` is NOT populated here — that lives in
        # the per-proposer ``propose()`` hook and arrives in a later PR.
        draft_mgr = self._draft_steering_manager
        if (
            draft_mgr is not None
            and self._draft_steerable_layers_cache
            and draft_mgr._tables_dirty
        ):
            draft_mgr.populate_steering_tables(self._draft_steerable_layers_cache)

        # 2. Build steering index
        # Get the shared steering_index buffer (all layers share one tensor)
        any_layer = next(iter(self._steerable_layers_cache.values()))
        steering_index = any_layer.steering_index

        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids

        # Walk requests in batch order, assigning each token's table row
        token_offset = 0
        for i in range(num_reqs):
            req_id = req_ids[i]
            n_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if n_tokens == 0:
                continue

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # Request not in batch yet (shouldn't happen but guard)
                steering_index[token_offset : token_offset + n_tokens] = 0
                token_offset += n_tokens
                continue

            # Determine phase from num_computed vs num_prompt
            num_computed = int(self.input_batch.num_computed_tokens_cpu[req_index])
            num_prompt = int(self.input_batch.num_prompt_tokens[req_index])
            is_prefilling = num_computed < num_prompt

            if is_prefilling:
                # Prefill: use prefill steering hash
                prefill_hash = int(
                    self.input_batch.request_prefill_steering_hash_main[req_index]
                )
                row = self._steering_manager.get_row_for_config(
                    prefill_hash, is_prefill=True
                )
                steering_index[token_offset : token_offset + n_tokens] = row

                # Check if this request will transition to decode after
                # this step's tokens are processed.
                num_computed_after = num_computed + n_tokens
                if num_computed_after >= num_prompt:
                    self._handle_steering_transition(req_id, req_index, prefill_hash)
            else:
                # Decode: use decode steering hash
                decode_hash = int(
                    self.input_batch.request_decode_steering_hash_main[req_index]
                )
                row = self._steering_manager.get_row_for_config(
                    decode_hash, is_prefill=False
                )
                steering_index[token_offset : token_offset + n_tokens] = row

            token_offset += n_tokens

        # Zero out remaining positions
        if token_offset < steering_index.shape[0]:
            steering_index[token_offset:].zero_()

        # Mark the index as having non-zero row references this step. The
        # no-active-state short-circuit on a future step will zero the index
        # if needed when transitioning back to "nothing active".
        self._steering_index_dirty = True

    def _populate_draft_steering_index(
        self,
        mode: str,
        n_tokens: int,
    ) -> None:
        """Write the draft model's per-forward ``steering_index`` buffer.

        Called from inside a speculative-decoding proposer's
        ``propose()`` method, once per draft forward pass.

        ``mode="first"`` mirrors the main runner's token walk: for each
        request in batch order, assign the request's ``_draft`` row
        (prefill or decode depending on the request's phase) to the
        ``num_scheduled_tokens[req]`` contiguous index slots. Used
        when the draft's first forward consumes the full set of
        accepted tokens (Eagle / Draft-model first pass).

        ``mode="loop"`` writes ``n_tokens`` entries in batch order,
        one row per request, always using the decode-phase ``_draft``
        hash. Used for Eagle's per-speculative-token loop and for
        Medusa's single-shot proposer — both consume one draft token
        per active request per forward.

        No-ops when the draft manager is absent or the draft has no
        steerable layers. Bails quickly when steering is disabled.
        """
        draft_mgr = getattr(self, "_draft_steering_manager", None)
        draft_layers = self._draft_steerable_layers_cache
        if draft_mgr is None or not draft_layers:
            return
        scheduler_output = getattr(self, "_last_scheduler_output", None)
        if scheduler_output is None:
            return
        any_layer = next(iter(draft_layers.values()))
        steering_index = any_layer.steering_index

        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids

        if mode == "first":
            token_offset = 0
            for i in range(num_reqs):
                req_id = req_ids[i]
                n = scheduler_output.num_scheduled_tokens.get(req_id, 0)
                if n == 0:
                    continue
                ri = self.input_batch.req_id_to_index.get(req_id)
                if ri is None:
                    steering_index[token_offset : token_offset + n] = 0
                    token_offset += n
                    continue
                num_computed = int(self.input_batch.num_computed_tokens_cpu[ri])
                num_prompt = int(self.input_batch.num_prompt_tokens[ri])
                is_prefilling = num_computed < num_prompt
                if is_prefilling:
                    h = int(self.input_batch.request_prefill_steering_hash_draft[ri])
                    row = self._get_draft_row_for_config(
                        h, is_prefill=True, req_id=req_id
                    )
                else:
                    h = int(self.input_batch.request_decode_steering_hash_draft[ri])
                    row = self._get_draft_row_for_config(
                        h, is_prefill=False, req_id=req_id
                    )
                steering_index[token_offset : token_offset + n] = row
                token_offset += n
            if token_offset < steering_index.shape[0]:
                steering_index[token_offset:].zero_()
            return

        if mode == "loop":
            # One row per active request in batch order; always decode.
            write_count = min(n_tokens, num_reqs, steering_index.shape[0])
            for i in range(write_count):
                req_id = req_ids[i]
                ri = self.input_batch.req_id_to_index.get(req_id)
                if ri is None:
                    steering_index[i] = 0
                    continue
                h = int(self.input_batch.request_decode_steering_hash_draft[ri])
                steering_index[i] = self._get_draft_row_for_config(
                    h, is_prefill=False, req_id=req_id
                )
            if write_count < steering_index.shape[0]:
                steering_index[write_count:].zero_()
            return

        raise ValueError(
            f"_populate_draft_steering_index: unknown mode {mode!r}; "
            f"expected 'first' or 'loop'."
        )

    def _get_draft_row_for_config(
        self,
        config_hash: int,
        is_prefill: bool,
        req_id: str,
    ) -> int:
        """Look up (and lazily register) a per-request row on the draft
        manager, falling back to the phase-global row on capacity /
        missing-config.

        Lazy registration is a concession: unlike main's scheduler-
        level capacity tracking, the draft manager has no upstream
        admission gate — so we register here on first sight and fall
        back on ``RuntimeError`` (capacity full). Accept the
        capacity-loss edge case as a PR-B MVP limitation: per-request
        draft steering silently degrades to the global-vector path.
        """
        mgr = self._draft_steering_manager
        if mgr is None or config_hash == 0:
            return 1 if is_prefill else 2
        phase = "prefill" if is_prefill else "decode"
        row = mgr.config_to_row.get((config_hash, phase))
        if row is not None:
            return row
        # Lazy register from the owning request's SamplingParams.
        rs = self.requests.get(req_id)
        if rs is None or rs.sampling_params is None:
            return 1 if is_prefill else 2
        eff = (
            rs.sampling_params.effective_prefill_steering_draft
            if is_prefill
            else rs.sampling_params.effective_decode_steering_draft
        )
        if not eff:
            return 1 if is_prefill else 2
        try:
            return mgr.register_config(
                config_hash,
                eff,
                phase=phase,
                locally_owned_layers=self._draft_locally_owned_layers,
            )
        except RuntimeError:
            logger.warning(
                "Draft steering manager capacity exhausted — falling "
                "back to phase-global row for config_hash=%d (%s). "
                "Draft request receives global steering only.",
                config_hash,
                phase,
            )
            return 1 if is_prefill else 2

    def _handle_steering_transition(
        self,
        req_id: str,
        req_index: int,
        prefill_hash: int,
    ) -> None:
        """Handle prefill->decode steering config transition.

        Called when a request will complete prefill after this step.
        Releases the prefill config and registers the decode config
        so it is ready for the next step's table population.

        If the steering table is at capacity, the decode registration
        is deferred to ``_pending_steering_registrations`` and retried
        on the next scheduler step.  The existing ``get_row_for_config``
        fallback (returns row 2 for unregistered decode hashes) provides
        graceful degradation during the deferral period.
        """
        mgr = self._steering_manager
        assert mgr is not None, (
            "_handle_steering_transition called without an initialised manager"
        )
        if prefill_hash != 0:
            mgr.release_config(prefill_hash, "prefill")

        decode_hash = int(self.input_batch.request_decode_steering_hash_main[req_index])
        if decode_hash != 0:
            req_state = self.requests.get(req_id)
            if req_state is not None and req_state.sampling_params is not None:
                sp = req_state.sampling_params
                if sp.effective_decode_steering:
                    try:
                        mgr.register_config(
                            decode_hash,
                            sp.effective_decode_steering,
                            phase="decode",
                            locally_owned_layers=self._locally_owned_layers,
                        )
                    except RuntimeError:
                        self._pending_steering_transitions.append(
                            (
                                req_id,
                                decode_hash,
                                sp.effective_decode_steering,
                                "decode",
                            )
                        )
                        logger.warning(
                            "Deferred decode steering config (hash=%d) "
                            "-- capacity full, will retry next step",
                            decode_hash,
                        )

        # Update phase tracking regardless of whether decode
        # registration succeeded or was deferred.
        self._req_steering_phase[req_id] = "decode"

    def _reset_steering_for_resumption(
        self,
        req_id: str,
        req_state: "CachedRequestState",
        new_num_computed_tokens: int,
    ) -> None:
        """Reset steering config registration when a request re-enters prefill.

        Called when a preempted request is resumed with num_computed_tokens
        reset. If the request had transitioned to decode before preemption,
        its decode config is still registered and its phase is stale.
        This helper releases the stale decode config and re-registers the
        prefill config (or defers it on capacity exhaustion).
        """
        mgr = getattr(self, "_steering_manager", None)
        if mgr is None:
            return
        prev_phase = self._req_steering_phase.get(req_id)
        if prev_phase != "decode":
            return
        if new_num_computed_tokens >= req_state.num_prompt_tokens:
            return  # still in decode, nothing to reset

        # Release the stale decode config.
        if req_state.decode_steering_config_hash != 0:
            mgr.release_config(req_state.decode_steering_config_hash, "decode")

        # Drop any stale deferred entries for this request.
        if self._pending_steering_transitions:
            self._pending_steering_transitions = [
                e for e in self._pending_steering_transitions if e[0] != req_id
            ]
        if self._pending_steering_registrations:
            self._pending_steering_registrations = [
                e for e in self._pending_steering_registrations if e[0] != req_id
            ]

        self._req_steering_phase[req_id] = "prefill"

        sp = req_state.sampling_params
        prefill_hash = req_state.prefill_steering_config_hash
        if prefill_hash == 0 or sp is None or not sp.effective_prefill_steering:
            return
        try:
            mgr.register_config(
                prefill_hash,
                sp.effective_prefill_steering,
                phase="prefill",
                locally_owned_layers=self._locally_owned_layers,
            )
        except RuntimeError:
            self._pending_steering_registrations.append(
                (req_id, prefill_hash, sp.effective_prefill_steering, "prefill")
            )
            logger.warning(
                "Deferred prefill steering config (hash=%d) on resumption "
                "-- capacity full, will retry next step",
                prefill_hash,
            )

    # -----------------------------------------------------------------------
    # Hooks called from _update_states() / _update_streaming_request()
    # -----------------------------------------------------------------------

    def _release_finished_steering_configs(
        self, finished_req_ids: "set[str] | list[str]"
    ) -> None:
        """Release the currently-active steering config for finished requests.

        Also drops deferred entries for those requests before
        ``self.requests`` is pruned, preventing row leaks.  Called
        before finished request state is popped so
        ``prefill_steering_config_hash`` /
        ``decode_steering_config_hash`` are still accessible.
        """
        mgr = getattr(self, "_steering_manager", None)
        if mgr is None:
            return

        for req_id in finished_req_ids:
            phase = self._req_steering_phase.pop(req_id, None)
            if phase is not None:
                req_state = self.requests.get(req_id)
                if req_state is not None:
                    if phase == "prefill":
                        h = req_state.prefill_steering_config_hash
                    else:
                        h = req_state.decode_steering_config_hash
                    if h != 0:
                        mgr.release_config(h, phase)

        # Also remove any deferred steering entries for finished
        # requests to prevent registering rows for dead requests.
        # (The retry loop also checks, but this eagerly drops
        # entries before self.requests is pruned below.)
        finished = set(finished_req_ids)
        if self._pending_steering_transitions:
            self._pending_steering_transitions = [
                entry
                for entry in self._pending_steering_transitions
                if entry[0] not in finished
            ]
        if self._pending_steering_registrations:
            self._pending_steering_registrations = [
                entry
                for entry in self._pending_steering_registrations
                if entry[0] not in finished
            ]

    def _register_initial_steering_config(
        self,
        req_id: str,
        new_req_data: "NewRequestData",
        req_state: "CachedRequestState",
    ) -> None:
        """Register the initial-phase steering config for a new request.

        Normally requests start in prefill, but a full prefix-cache hit
        (``num_computed >= num_prompt``) puts a request directly into
        decode.  Handles capacity-exhaustion deferral.
        """
        mgr = getattr(self, "_steering_manager", None)
        if mgr is None or new_req_data.sampling_params is None:
            return

        sp = new_req_data.sampling_params
        if new_req_data.num_computed_tokens >= req_state.num_prompt_tokens:
            # Already past prefill — register decode config.
            if (
                new_req_data.decode_steering_config_hash != 0
                and sp.effective_decode_steering
            ):
                try:
                    mgr.register_config(
                        new_req_data.decode_steering_config_hash,
                        sp.effective_decode_steering,
                        phase="decode",
                        locally_owned_layers=self._locally_owned_layers,
                    )
                except RuntimeError:
                    self._pending_steering_registrations.append(
                        (
                            req_id,
                            new_req_data.decode_steering_config_hash,
                            sp.effective_decode_steering,
                            "decode",
                        )
                    )
                    logger.warning(
                        "Deferred decode steering config "
                        "(hash=%d) -- capacity full, "
                        "will retry next step",
                        new_req_data.decode_steering_config_hash,
                    )
            self._req_steering_phase[req_id] = "decode"
        else:
            # Normal: start in prefill; decode registered
            # on transition in _update_steering_buffers.
            if (
                new_req_data.prefill_steering_config_hash != 0
                and sp.effective_prefill_steering
            ):
                try:
                    mgr.register_config(
                        new_req_data.prefill_steering_config_hash,
                        sp.effective_prefill_steering,
                        phase="prefill",
                        locally_owned_layers=self._locally_owned_layers,
                    )
                except RuntimeError:
                    self._pending_steering_registrations.append(
                        (
                            req_id,
                            new_req_data.prefill_steering_config_hash,
                            sp.effective_prefill_steering,
                            "prefill",
                        )
                    )
                    logger.warning(
                        "Deferred prefill steering config "
                        "(hash=%d) -- capacity full, "
                        "will retry next step",
                        new_req_data.prefill_steering_config_hash,
                    )
            self._req_steering_phase[req_id] = "prefill"

    def _refresh_streaming_steering(
        self,
        req_id: str,
        new_req_data: "NewRequestData",
        old_prefill_hash: int,
        old_decode_hash: int,
        new_prefill_hash: int,
        new_decode_hash: int,
    ) -> None:
        """Refresh steering state for a streaming re-added request.

        Streaming re-adds go back through prefill, so we must:
        1. Release the old config (whatever phase we were tracking)
        2. Purge stale deferred entries for this request
        3. Register the new prefill config
        4. Update phase tracking
        """
        mgr = getattr(self, "_steering_manager", None)
        if mgr is None:
            return

        # Release the old phase config.
        old_phase = self._req_steering_phase.get(req_id)
        if old_phase is not None:
            if old_phase == "prefill" and old_prefill_hash != 0:
                mgr.release_config(old_prefill_hash, "prefill")
            elif old_phase == "decode" and old_decode_hash != 0:
                mgr.release_config(old_decode_hash, "decode")

        # Purge stale deferred entries for this request.
        if self._pending_steering_transitions:
            self._pending_steering_transitions = [
                entry
                for entry in self._pending_steering_transitions
                if entry[0] != req_id
            ]
        if self._pending_steering_registrations:
            self._pending_steering_registrations = [
                entry
                for entry in self._pending_steering_registrations
                if entry[0] != req_id
            ]

        # Register new prefill config (streaming re-adds start
        # in prefill).
        sp = new_req_data.sampling_params
        if new_prefill_hash != 0 and sp is not None and sp.effective_prefill_steering:
            try:
                mgr.register_config(
                    new_prefill_hash,
                    sp.effective_prefill_steering,
                    phase="prefill",
                    locally_owned_layers=self._locally_owned_layers,
                )
            except RuntimeError:
                self._pending_steering_registrations.append(
                    (
                        req_id,
                        new_prefill_hash,
                        sp.effective_prefill_steering,
                        "prefill",
                    )
                )
                logger.warning(
                    "Deferred prefill steering config "
                    "(hash=%d) for streaming re-add -- "
                    "capacity full, will retry next step",
                    new_prefill_hash,
                )
            self._req_steering_phase[req_id] = "prefill"
        elif new_prefill_hash == 0 and new_decode_hash == 0:
            # No steering for this request anymore.
            self._req_steering_phase.pop(req_id, None)
        else:
            # Has hashes but no effective prefill vectors (e.g.,
            # decode-only steering).  Mark as prefill since the
            # request re-enters prefill; transition to decode
            # will handle decode registration.
            self._req_steering_phase[req_id] = "prefill"
