# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define activation steering functionality mixin for model runners.
"""

import math
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn

from vllm.config.steering_types import (
    SteeringClampSpec,
    SteeringVectorSpec,
    hash_steering_config,
    merge_steering_specs,
    normalize_clamp_entry,
    resolve_effective_clamps,
    resolve_effective_vectors,
    scale_steering_spec,
)
from vllm.exceptions import SteeringVectorError
from vllm.logger import init_logger
from vllm.model_executor.layers.clamp import CLAMP_ANY_ACTIVE_ATTR
from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    HOOK_POINT_MONITOR_ACTIVE_ATTR,
    HOOK_POINT_ROW_ACTIVE_ATTR,
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
    resize_steering_row_monitor_buffers,
    share_steering_decode_mask_across_layers,
    share_steering_row_gate_across_layers,
    share_steering_token_scales_across_layers,
)
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.worker.steering_action_queue import (
    DECLARATIVE_SOURCE,
    RequestSteeringOverride,
    SteeringActionQueue,
    SteeringMonitorUpdate,
    SteeringScaleUpdate,
    SteeringVectorUpdate,
    apply_steering_updates,
    get_steering_action_queue,
    install_steering_action_queue,
    steering_update_accepted,
    validate_steering_monitor,
    validate_steering_scale,
    validate_steering_vectors,
)
from vllm.v1.worker.steering_batch_view import SteeringBatchView
from vllm.v1.worker.steering_manager import SteeringManager
from vllm.v1.worker.steering_owner import RowOwner
from vllm.v1.worker.steering_vector_registry import (
    WorkerSteeringVectorRegistry,
    get_worker_steering_vector_registry,
    install_worker_steering_vector_registry,
)


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


_U64_MASK = (1 << 64) - 1


def _mix64(state: int, value: int) -> int:
    """Fold ``value`` into ``state`` with a splitmix64-style mix.

    Non-commutative (order-sensitive) and free of any process-seeded
    randomness (``PYTHONHASHSEED`` never enters), so two workers folding
    the same sequence reach the same u64 state. Used to accumulate the
    rolling checksum of applied steering actions.
    """
    state = (state + (value & _U64_MASK) + 0x9E3779B97F4A7C15) & _U64_MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return z ^ (z >> 31)


def _array_digest(arr: "np.ndarray | None") -> bytes:
    """Compact, bit-exact digest of a numpy array (shape + CRC of bytes).

    Steering actions are host-side numpy built from rank-identical
    inputs, so a bit-exact content digest is strictly stronger than a
    norm and never diverges across ranks. Cheap: one CRC over the
    contiguous float32 bytes.
    """
    if arr is None:
        return b"none"
    contig = np.ascontiguousarray(arr, dtype=np.float32)
    return b"%b:%d" % (
        repr(contig.shape).encode(),
        zlib.crc32(contig.tobytes()) & 0xFFFFFFFF,
    )


def _vectors_digest(vectors: "dict | None") -> bytes:
    """Deterministic digest of a ``{hook: {layer: ndarray}}`` vector dict."""
    if vectors is None:
        return b"none"
    parts: list[bytes] = []
    for hook in sorted(vectors):
        for layer in sorted(vectors[hook]):
            arr_digest = _array_digest(vectors[hook][layer])
            parts.append(b"%b|%d|%b" % (hook.encode(), layer, arr_digest))
    return b"@".join(parts)


def _steering_action_digest(action) -> bytes:
    """Order-independent, PYTHONHASHSEED-free digest of one action's content.

    Pure function of the action's identifying fields (class name, target
    req_id / config_hash / dyn_id, hook / layer, source) plus a bit-exact
    digest of any vector / probe payload. Folded into the running
    checksum only for actions that were actually applied.
    """
    name = type(action).__name__.encode()
    if isinstance(action, SteeringVectorUpdate):
        return b";".join(
            (
                name,
                action.phase.encode(),
                action.source.encode(),
                _vectors_digest(action.vectors),
            )
        )
    if isinstance(action, RequestSteeringOverride):
        return b";".join(
            (
                name,
                action.req_id.encode(),
                b"1" if action.compose_admitted else b"0",
                action.source.encode(),
                _vectors_digest(action.vectors),
            )
        )
    if isinstance(action, SteeringScaleUpdate):
        return b";".join(
            (
                name,
                struct.pack("<d", action.scale),
                repr(action.config_hash).encode(),
                repr(action.dyn_id).encode(),
                repr(action.req_id).encode(),
                b"1" if action.tier_gain else b"0",
                action.source.encode(),
            )
        )
    if isinstance(action, SteeringMonitorUpdate):
        return b";".join(
            (
                name,
                action.hook.encode(),
                b"%d" % action.layer,
                struct.pack("<d", action.threshold),
                struct.pack("<d", action.sharpness),
                b"1" if action.gate_rows else b"0",
                repr(action.req_id).encode(),
                repr(action.config_hash).encode(),
                repr(action.dyn_id).encode(),
                action.source.encode(),
                _array_digest(action.probe),
            )
        )
    return name


if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)


@dataclass
class _SteeringReqState:
    """Canonical per-request steering identity, runner-agnostic.

    Populated identically on both runners from the broadcast
    ``NewRequestData`` at admission (and on streaming re-add / preemption
    resume). Captures everything the transition / release / resolve paths
    need without reaching into a runner-specific request map: the params
    (for re-resolving the decode tier lazily), both config hashes, the
    prompt length (for the prefill->decode boundary), and the
    currently-registered phase.
    """

    sampling_params: SamplingParams
    prefill_hash: int
    decode_hash: int
    num_prompt_tokens: int
    phase: str  # "prefill" | "decode"


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
    # All steering state is initialised eagerly by ``_init_steering_state``
    # at the end of ``GPUModelRunner.load_model``.  The class-level
    # defaults below cover the pre-init window (e.g. unit tests that
    # construct the mixin without going through load_model) so plain
    # attribute access is safe without ``hasattr`` guards.
    _steering_manager: SteeringManager | None = None
    _steerable_layers_cache: dict[int, nn.Module] | None = None
    # Canonical per-request steering store, populated identically on both
    # runners from the broadcast ``NewRequestData``. Replaces the former
    # v1-only ``_req_steering_phase`` dict + v2-only ``_steering_reqs`` copy;
    # ``_steering_reqs[rid].phase`` is the single source of truth for a
    # request's registered phase.
    _steering_reqs: dict[str, _SteeringReqState]
    _steering_index_dirty: bool
    # Rolling determinism checksum of *applied* dynamic steering actions
    # (u64) and the count of folds, plus a per-drain-batch ordinal. Class
    # defaults cover the pre-init window (duck-typed test hosts that skip
    # ``_init_steering_state``); the router compares these across ranks to
    # detect a silent lock-step desync. See §6 of the design doc.
    _steering_action_checksum: int = 0
    _steering_action_count: int = 0
    _steering_apply_batches: int = 0
    # Worker-side mirror of the API server's named steering module
    # registry.  Populated via ``register_steering_modules`` RPC during
    # API server bootstrap and on every /v1/steering/modules/{register,
    # unregister} call.  Per-process, per-worker — collective_rpc
    # guarantees identical state across TP × PP ranks.
    _steering_module_registry: dict[
        str,
        tuple[
            SteeringVectorSpec | None,
            SteeringVectorSpec | None,
            SteeringVectorSpec | None,
        ],
    ]
    # Pre-resolved spec cache for the named-module fast path.  Each entry
    # stores ``(resolved_prefill, resolved_decode)``: the output of
    # :func:`resolve_effective_vectors` applied to the module's
    # ``(base, phase)`` specs at registration time.  The hot path in
    # :meth:`_resolve_request_steering` skips the per-request merge +
    # resolve work (~37 ms/generate at 3 hooks on gemma-3-4b-it) when a
    # request references a name with no inline overrides; ``scale!=1.0``
    # is handled by an in-place ``arr * scale`` over the cached arrays.
    # Populated alongside ``_steering_module_registry`` and invalidated
    # together.
    _steering_module_resolved_cache: dict[
        str,
        tuple[
            dict[str, dict[int, "np.ndarray"]] | None,
            dict[str, dict[int, "np.ndarray"]] | None,
        ],
    ]
    # Tracks (config_hash, phase) pairs the worker pinned at named-module
    # register time via :meth:`pre_materialize_steering_module`.  Each entry
    # is a hold of ``+1`` on the manager's refcount table for the matching
    # row, on top of any per-request refcounts the hot path adds.  The
    # pin keeps the row materialized between registration and the first
    # request that resolves to the module, eliminating the cold-path
    # ``register_config.materialize`` cost (~15 ms on gemma-3-4b-it/3090
    # in named_shared mode) from the request's TTFT.  Released on
    # :meth:`release_pre_materialized_steering_module` (called by
    # ``unregister_steering_modules`` from the API layer).
    _steering_module_pinned_rows: dict[str, list[tuple[int, str]]]
    # Set of layer indices physically owned by this worker.  Under PP,
    # this is a contiguous subset of ``[0, num_layers)``; under single-
    # worker and under TP (which replicates all layers per rank), it
    # equals the full model's layer set.  Threaded into
    # ``SteeringManager`` calls so non-local tensors are never
    # materialized on this rank.
    _locally_owned_layers: frozenset[int]
    # CPU scratch arrays used by ``_update_steering_buffers`` to build
    # the per-token row mapping in a single ``np.repeat`` + non-blocking
    # H2D copy, replacing the per-request slice-assign loop.  The
    # per-request scratches are sized to ``max_num_seqs``; the
    # row-per-token scratch is a pinned-memory torch tensor sized to
    # ``max_num_batched_tokens`` so the H2D copy can actually overlap
    # compute (``non_blocking=True`` on a non-pinned source silently
    # falls back to a synchronous copy).  ``None`` when steering is
    # inactive.
    _steering_rows_scratch: np.ndarray | None = None
    _steering_n_tokens_scratch: np.ndarray | None = None
    _steering_index_pinned: torch.Tensor | None = None
    # Per-request dynamic-tier gain scratch + pinned per-token gate staging
    # (§5.4); mirror the index scratches above.
    _steering_tier_gain_scratch: np.ndarray | None = None
    _steering_token_scales_pinned: torch.Tensor | None = None
    # Per-request decode mask scratch + pinned per-token staging for the
    # Phase 2 row gate; mirror the gate scratches above.
    _steering_decode_mask_scratch: np.ndarray | None = None
    _steering_decode_mask_pinned: torch.Tensor | None = None
    # Reusable per-step batch view (the de-fork seam between v1's batch-ordered
    # ``input_batch`` columns and v2's ``idx_mapping`` + ``RequestState``). One
    # instance mutated in place each step -> no per-step allocation. The v1
    # identity slot->row map is grown lazily to the batch size and cached.
    _steering_bview: SteeringBatchView | None = None
    _steering_idx_identity: np.ndarray | None = None

    # Attributes provided by the concrete model runner that mixes this
    # class in.  Declared here purely so static type checking can see
    # them — there is no runtime assignment.
    if TYPE_CHECKING:
        vllm_config: VllmConfig
        input_batch: InputBatch
        requests: dict[str, CachedRequestState]

        def get_model(self) -> nn.Module: ...

    # -----------------------------------------------------------------------
    # Eager initialisation
    # -----------------------------------------------------------------------

    def _init_steering_state(self) -> None:
        """Initialise steering state at the end of model load.

        Walks the loaded model for layers that registered steering
        buffers, captures the buffer device, and constructs the
        ``SteeringManager``.  Must be called exactly once — typically
        from ``GPUModelRunner.load_model`` after the model is fully
        loaded.

        When steering is disabled (no ``SteeringConfig``) or the model
        has no steerable layers, ``_steering_manager`` stays ``None``
        so per-step ``_update_steering_buffers`` and the public API
        methods can short-circuit cheaply.
        """
        steerable: dict = {}
        if hasattr(self, "get_model"):
            for mod in self.get_model().modules():
                if not hasattr(mod, "layer_idx"):
                    continue
                has_any_table = any(
                    hasattr(mod, attr) for attr in HOOK_POINT_TABLE_ATTR.values()
                )
                if has_any_table:
                    steerable[mod.layer_idx] = mod
        self._steerable_layers_cache = steerable
        # Share the per-token dynamic-tier gate across layers (one per-step
        # H2D, like steering_index) — done here, the single site with all
        # steerable layers, to avoid per-model-file share calls.
        share_steering_token_scales_across_layers(steerable.values())
        share_steering_row_gate_across_layers(steerable.values())
        share_steering_decode_mask_across_layers(steerable.values())
        self._locally_owned_layers = frozenset(steerable.keys())
        self._steering_reqs = {}
        self._steering_index_dirty = False
        self._steering_module_registry = {}
        self._steering_module_resolved_cache = {}
        self._steering_module_clamps = {}
        self._steering_module_clamps_effective = {}
        self._steering_module_pinned_rows = {}
        # Worker-resident named probe/steer vector registry (rank-replicated
        # via ``collective_rpc``). Installed as a process-global so the sync
        # steering consumer — no runner handle — can re-resolve a latch's
        # referenced name at bridge time. Always present so ``NamedVec`` gate
        # resolution and the register RPC have a target even when this worker
        # has no steerable layers.
        if get_worker_steering_vector_registry() is None:
            install_worker_steering_vector_registry(WorkerSteeringVectorRegistry())
        # Per-source applied/rejected counters for dynamic steering
        # actions (both transports), keyed by submitting source name.
        self._dynamic_steering_stats: dict[str, dict[str, int]] = {}
        # Rolling checksum of every *applied* action, folded in application
        # order with the drain-batch ordinal (see ``_fold_steering_action``).
        # Compared across ranks at status time to catch a silent desync.
        self._steering_action_checksum = 0
        self._steering_action_count = 0
        self._steering_apply_batches = 0
        # Live per-request dynamic decode overrides: req_id -> dyn_id in
        # the manager's dynamic pool. Pure routing state on top of the
        # admission machinery — admitted config hashes, refcounts, and
        # scheduler accounting are never touched. Cleaned up on request
        # finish, preemption resumption, and streaming re-add.
        self._req_dynamic_decode: dict[str, int] = {}
        # req_id -> source tag that owns the request's dynamic override.
        # Enforces precedence: an operator/server consumer (any non-declarative
        # source) WINS over a client declarative gate. Cleaned up alongside
        # ``_req_dynamic_decode``.
        self._req_override_source: dict[str, str] = {}
        # APC steering-signature reporting (see
        # docs/design/dynamic_steering_apc_notification.md). Last effective
        # decode signature reported to the scheduler per request, so each
        # step only the *changed* signatures ride ``ModelRunnerOutput``.
        # ``_pending_decode_sigs`` holds the current step's delta for the
        # model runner to attach; recomputed every ``_update_steering_buffers``.
        self._req_decode_sig_reported: dict[str, int] = {}
        self._pending_decode_sigs: dict[str, int] = {}

        steering_config = getattr(self.vllm_config, "steering_config", None)
        if steering_config is None or not steerable:
            self._steering_manager = None
            return

        # Stamp the cross-layer-monitor opt-in onto every steerable layer as a
        # plain Python attribute, so ``apply_layer_steering`` branches on it as
        # a torch.compile-time constant. Set once here (before any forward /
        # graph trace) and constant for the model's lifetime. When False
        # (default) the monitor is the same-hook fused gate; when True the
        # mutating cross-layer ``steering_monitor`` op is emitted at every
        # steered hook. See docs/design/dynamic_steering.md §8.
        cross_layer = bool(
            getattr(steering_config, "enable_cross_layer_monitor", False)
        )
        from vllm.model_executor.layers.clamp import CLAMP_GATE_ACTIVE_ATTR

        for mod in steerable.values():
            mod._cross_layer_monitor = cross_layer
            # Directional clamps honor the shared ``steering_row_gate`` only in
            # the materialized (cross-layer) monitor mode: there the standalone
            # ``steering_monitor`` op WRITES the per-token gate into the shared
            # buffer that layers >= L read, so a clamp at those layers reads the
            # same gate the additive row term reads ("detect at L, clamp at
            # layers >= L"). In the default fused mode the gate is recomputed in
            # the steering kernel and never materialized, so clamps stay ungated
            # (and a declarative gate targeting clamps is rejected at admission).
            # Set once here, constant for the model's lifetime.
            for hp in SteeringHookPoint:
                gate_flag = getattr(mod, CLAMP_GATE_ACTIVE_ATTR[hp], None)
                if gate_flag is not None:
                    gate_flag.fill_(cross_layer)

        # Per-row (per-request) monitor: opt-in. When enabled, resize the
        # dummy ``(1, 1)`` probe/params buffers to full per-row tables across
        # every steerable layer (the single site with all layers in hand,
        # like the share_* helpers). The flag also gates the per-row branch of
        # ``_apply_monitor_update`` so a targeted monitor on a disabled engine
        # is rejected rather than silently dropped.
        self._row_monitor_enabled = bool(
            getattr(steering_config, "enable_row_monitor", False)
        )
        resize_steering_row_monitor_buffers(
            steerable.values(), enable=self._row_monitor_enabled
        )

        # Resolve device from the first steerable layer's table buffer
        # so per-request vectors are allocated on the same device,
        # avoiding CPU->GPU copies each step.
        table_device: torch.device | None = None
        table_dtype: torch.dtype | None = None
        hidden_size: int | None = None
        for mod in steerable.values():
            for attr in HOOK_POINT_TABLE_ATTR.values():
                if hasattr(mod, attr):
                    table_buf = getattr(mod, attr)
                    table_device = table_buf.device
                    table_dtype = table_buf.dtype
                    hidden_size = table_buf.shape[1]
                    break
            if table_device is not None:
                break

        self._max_clamp_directions = int(
            getattr(steering_config, "max_clamp_directions", 0)
        )
        self._steering_manager = SteeringManager(
            steering_config.max_steering_configs,
            device=table_device,
            max_dynamic_steering_configs=getattr(
                steering_config, "max_dynamic_steering_configs", 0
            ),
            max_clamp_directions=self._max_clamp_directions,
        )

        # Dynamic steering action queue (Phase 0). Installed only in
        # single-rank topologies: capture consumers — the expected
        # submitters — are constructed on TP rank 0 only, so updates
        # originating from one would silently diverge the steering
        # tables across TP/PP ranks. See steering_action_queue.py and
        # docs/design/dynamic_steering.md for the determinism contract.
        parallel_config = getattr(self.vllm_config, "parallel_config", None)
        tp_size = getattr(parallel_config, "tensor_parallel_size", 1)
        pp_size = getattr(parallel_config, "pipeline_parallel_size", 1)
        if tp_size == 1 and pp_size == 1:
            install_steering_action_queue(SteeringActionQueue())
            logger.info("dynamic steering action queue installed (tp=1, pp=1)")
        else:
            install_steering_action_queue(None)
            logger.info(
                "dynamic steering action queue unavailable: requires "
                "tp=1 and pp=1, got tp=%d pp=%d",
                tp_size,
                pp_size,
            )

        # Pre-allocate CPU scratch buffers for the vectorized
        # steering_index build in ``_update_steering_buffers``.  The
        # per-request numpy buffers hold one entry per request in the
        # batch (bounded by ``max_num_seqs``); the pinned torch tensor
        # holds the expanded per-token row array (bounded by
        # ``max_num_batched_tokens``) and is the source of the single
        # H2D copy each step.  Pinning lets ``non_blocking=True`` on
        # the copy actually overlap with model compute.
        scheduler_config = getattr(self.vllm_config, "scheduler_config", None)
        if scheduler_config is not None:
            max_tokens = int(scheduler_config.max_num_batched_tokens)
            max_seqs = int(getattr(scheduler_config, "max_num_seqs", max_tokens))
            self._steering_rows_scratch = np.zeros(max_seqs, dtype=np.int64)
            self._steering_n_tokens_scratch = np.zeros(max_seqs, dtype=np.int64)
            self._steering_tier_gain_scratch = np.zeros(max_seqs, dtype=np.float32)
            # Per-request decode mask (1.0 decode / 0.0 prefill) for Phase 2
            # row gating; expanded per-token and H2D'd into steering_decode_mask.
            self._steering_decode_mask_scratch = np.zeros(max_seqs, dtype=np.float32)
            try:
                self._steering_index_pinned = torch.zeros(
                    max_tokens, dtype=torch.long, pin_memory=True
                )
                self._steering_token_scales_pinned = torch.zeros(
                    max_tokens, dtype=torch.float32, pin_memory=True
                )
                self._steering_decode_mask_pinned = torch.zeros(
                    max_tokens, dtype=torch.float32, pin_memory=True
                )
            except RuntimeError:
                # Pinned memory unavailable (e.g. CPU-only test
                # environment); fall back to a regular CPU tensor.
                self._steering_index_pinned = torch.zeros(max_tokens, dtype=torch.long)
                self._steering_token_scales_pinned = torch.zeros(
                    max_tokens, dtype=torch.float32
                )
                self._steering_decode_mask_pinned = torch.zeros(
                    max_tokens, dtype=torch.float32
                )

        # Warm the fused-apply Triton kernel so first-call JIT cost
        # happens before any captured forward pass. Without this, the
        # initial CUDA-graph capture step could trigger a Triton compile
        # and fail capture, and — as observed on a 3090 with
        # gemma-3-4b-it — every served-window also pays ~18-25 ms of
        # ``cuLibraryLoadData`` events for shape variants that only show
        # up at runtime. Driving the warmup over every captured batch
        # size eliminates those compiles from the served window.
        if (
            table_device is not None
            and table_device.type == "cuda"
            and table_dtype is not None
            and hidden_size is not None
        ):
            from vllm.model_executor.layers.steering_kernel import (
                warmup_apply_steering_kernel,
            )

            compute_dtype = getattr(self.vllm_config.model_config, "dtype", table_dtype)
            compilation_config = getattr(self.vllm_config, "compilation_config", None)
            capture_sizes = (
                getattr(compilation_config, "cudagraph_capture_sizes", None)
                if compilation_config is not None
                else None
            )
            warmup_apply_steering_kernel(
                hidden_size=hidden_size,
                table_rows=(
                    steering_config.max_steering_configs
                    + getattr(steering_config, "max_dynamic_steering_configs", 0)
                    + 3
                ),
                table_dtype=table_dtype,
                compute_dtype=compute_dtype,
                device=table_device,
                capture_sizes=list(capture_sizes) if capture_sizes else None,
                row_monitor_enabled=self._row_monitor_enabled,
            )

            # Warm the in-graph monitor kernel (Phase 2, §8) too — the
            # monitor op is emitted at every steered hook, so its Triton
            # JIT must retire before CUDA-graph capture even when no probe
            # is configured yet (the inactive branch shares the artifact).
            from vllm.model_executor.layers.steering_monitor_kernel import (
                warmup_steering_monitor_kernel,
            )

            warmup_steering_monitor_kernel(
                hidden_size=hidden_size,
                compute_dtype=compute_dtype,
                device=table_device,
                capture_sizes=list(capture_sizes) if capture_sizes else None,
            )

            # Warm the directional-clamp kernels when clamping is enabled —
            # the clamp ops are emitted at every steered hook once buffers
            # exist, so their Triton JIT must retire before CUDA-graph
            # capture even when no clamp is configured yet.
            if self._max_clamp_directions > 0:
                from vllm.model_executor.layers.clamp_kernel import (
                    warmup_apply_clamp_kernel,
                )

                warmup_apply_clamp_kernel(
                    hidden_size=hidden_size,
                    table_rows=(
                        steering_config.max_steering_configs
                        + getattr(steering_config, "max_dynamic_steering_configs", 0)
                        + 3
                    ),
                    max_directions=self._max_clamp_directions,
                    table_dtype=table_dtype,
                    compute_dtype=compute_dtype,
                    device=table_device,
                    capture_sizes=list(capture_sizes) if capture_sizes else None,
                )

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

    def _validate_clamps_spec(
        self,
        clamps_data: dict[str, dict[int, list[dict]]],
        steerable: dict,
    ) -> set[int]:
        """Validate a global clamp spec against this worker's layers.

        Clamp sibling of :meth:`_validate_vectors_spec`: checks hook
        points, per-entry structure (via :func:`normalize_clamp_entry`),
        direction width against the layer's table hidden size, and the
        per-site K cap. Returns the set of valid layer indices on this
        worker; raises ``SteeringVectorError`` on any invalid input.
        """
        max_dirs = getattr(self, "_max_clamp_directions", 0)
        valid_indices: set[int] = set()
        for hook_point_str, layer_entries in clamps_data.items():
            try:
                hp_enum = SteeringHookPoint(hook_point_str)
            except ValueError as exc:
                raise SteeringVectorError(
                    f"Invalid hook point: {hook_point_str!r}"
                ) from exc
            table_attr = HOOK_POINT_TABLE_ATTR[hp_enum]

            for idx, entries in layer_entries.items():
                if idx not in steerable:
                    continue
                mod = steerable[idx]
                if not hasattr(mod, table_attr):
                    raise SteeringVectorError(
                        f"Hook point {hook_point_str!r} not active on layer {idx}"
                    )
                if not isinstance(entries, list):
                    raise SteeringVectorError(
                        f"Layer {idx} ({hook_point_str}): clamps must be a "
                        f"list of clamp entries, got {type(entries).__name__}"
                    )
                if max_dirs <= 0:
                    raise SteeringVectorError(
                        "Clamping is disabled on this engine "
                        "(steering_config.max_clamp_directions=0)"
                    )
                if len(entries) > max_dirs:
                    raise SteeringVectorError(
                        f"Layer {idx} ({hook_point_str}): {len(entries)} "
                        f"clamp directions exceed max_clamp_directions="
                        f"{max_dirs}"
                    )
                expected_size = getattr(mod, table_attr).shape[1]
                for i, entry in enumerate(entries):
                    try:
                        vec, _lo, _hi, _strength = normalize_clamp_entry(entry)
                    except (TypeError, ValueError) as exc:
                        raise SteeringVectorError(
                            f"Layer {idx} ({hook_point_str}) clamp[{i}]: {exc}"
                        ) from exc
                    if len(vec) != expected_size:
                        raise SteeringVectorError(
                            f"Layer {idx} ({hook_point_str}) clamp[{i}]: "
                            f"expected direction of size {expected_size}, "
                            f"got {len(vec)}"
                        )
                valid_indices.add(idx)
        return valid_indices

    def list_steerable_layers(self) -> dict[int, list[str]]:
        """Return steerable layers on this worker with their hook points.

        Returns ``{layer_idx: [hook_point_name, ...]}`` for every
        layer owned by this worker that has at least one steering
        table buffer registered. Hook-point names are sorted for
        determinism.
        """
        result: dict[int, list[str]] = {}
        for idx, mod in self._steerable_layers().items():
            hooks = sorted(
                hp.value
                for hp, attr in HOOK_POINT_TABLE_ATTR.items()
                if hasattr(mod, attr)
            )
            result[idx] = hooks
        return result

    def _notify_manager_vectors(
        self,
        vectors_data: dict[str, dict[int, list[float]]],
        steerable: dict,
        valid_indices: set[int],
        phase: str,
    ) -> None:
        """Notify SteeringManager of global vector changes for a given
        phase (``"base"``, ``"prefill"``, or ``"decode"``).

        Converts the raw ``list[float]`` values from *vectors_data*
        into tensors matching the layer buffer dtype/device, then passes
        them to the manager.  This avoids reading from shared buffers,
        which would silently use stale or overwritten data for
        phase-specific tiers.
        """
        mgr = self._steering_manager
        if mgr is None:
            return
        locally_owned = getattr(self, "_locally_owned_layers", None)
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

    def _notify_manager_clamps(
        self,
        clamps_data: dict[str, dict[int, list[dict]]],
        steerable: dict,
        valid_indices: set[int],
        phase: str,
    ) -> None:
        """Notify SteeringManager of global clamp changes for a phase."""
        mgr = self._steering_manager
        if mgr is None:
            return
        locally_owned = getattr(self, "_locally_owned_layers", None)
        for hook_point_str, layer_entries in clamps_data.items():
            for idx, entries in layer_entries.items():
                if idx not in valid_indices or idx not in steerable:
                    continue
                mgr.update_global_clamps(
                    hook_point_str,
                    idx,
                    entries,
                    phase=phase,
                    locally_owned_layers=locally_owned,
                )

    def set_steering_vectors(
        self,
        vectors: dict[str, dict[int, list[float]]] | None = None,
        prefill_vectors: dict[str, dict[int, list[float]]] | None = None,
        decode_vectors: dict[str, dict[int, list[float]]] | None = None,
        replace: bool = False,
        validate_only: bool = False,
        clamps: dict[str, dict[int, list[dict]]] | None = None,
        prefill_clamps: dict[str, dict[int, list[dict]]] | None = None,
        decode_clamps: dict[str, dict[int, list[dict]]] | None = None,
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

        Returns:
            ``(tp_rank, pp_rank, sorted_valid_layers)``. The rank info
            lets the router detect TP-divergence (a server-side
            invariant violation — TP ranks within the same PP stage
            must own identical layer sets). The sorted layer list is
            the set of layer indices actually updated (or *would* be
            updated when *validate_only*) on this worker. The router
            unions these across workers.
        """
        tp_rank, pp_rank = _get_steering_ranks()
        steerable = self._steerable_layers()
        if not steerable:
            return (tp_rank, pp_rank, [])

        # Collect all tiers with data.
        all_tiers: list[tuple[str, dict[str, dict[int, list[float]]]]] = []
        if vectors:
            all_tiers.append(("base", vectors))
        if prefill_vectors:
            all_tiers.append(("prefill", prefill_vectors))
        if decode_vectors:
            all_tiers.append(("decode", decode_vectors))
        clamp_tiers: list[tuple[str, dict[str, dict[int, list[dict]]]]] = []
        if clamps:
            clamp_tiers.append(("base", clamps))
        if prefill_clamps:
            clamp_tiers.append(("prefill", prefill_clamps))
        if decode_clamps:
            clamp_tiers.append(("decode", decode_clamps))

        if not all_tiers and not clamp_tiers:
            if replace:
                self.clear_steering_vectors()
            return (tp_rank, pp_rank, [])

        # Validate all tiers.
        valid_indices: set[int] = set()
        for _phase, tier_data in all_tiers:
            valid_indices.update(self._validate_vectors_spec(tier_data, steerable))
        for _phase, tier_clamps in clamp_tiers:
            valid_indices.update(self._validate_clamps_spec(tier_clamps, steerable))

        if not valid_indices:
            return (tp_rank, pp_rank, [])

        if validate_only:
            return (tp_rank, pp_rank, sorted(valid_indices))

        # Clear if replacing.
        if replace:
            self.clear_steering_vectors()

        # Notify manager with base vectors.
        if vectors:
            self._notify_manager_vectors(vectors, steerable, valid_indices, "base")

        # Phase-specific vectors go only to the manager, not the shared
        # buffers — writing them would overwrite base values and cause
        # get_steering_status() to report the wrong tier.
        if prefill_vectors:
            self._notify_manager_vectors(
                prefill_vectors, steerable, valid_indices, "prefill"
            )

        if decode_vectors:
            self._notify_manager_vectors(
                decode_vectors, steerable, valid_indices, "decode"
            )

        # Clamp tiers ride the same call; the manager stores them in the
        # parallel global clamp dicts and populate concatenates them into
        # rows 1/2 and every config row.
        for phase, tier_clamps in clamp_tiers:
            self._notify_manager_clamps(tier_clamps, steerable, valid_indices, phase)

        return (tp_rank, pp_rank, sorted(valid_indices))

    def clear_steering_vectors(self) -> None:
        """Clear all tiers (base, prefill, decode) in the SteeringManager,
        including the global clamp tiers."""
        mgr = self._steering_manager
        if mgr is not None:
            mgr.clear_global_vectors()
            mgr.clear_global_clamps()

    def get_steering_status(self) -> dict:
        """Return per-hook-point status for active layers.

        Returns ``{layer_idx: {hook_point: {"norm": float,
        "prefill_norm"?: float, "decode_norm"?: float}}}`` for
        layers/hook-points that have a non-zero steering vector.
        """
        result: dict = {}
        mgr = self._steering_manager
        if mgr is None:
            return result
        for phase_name, phase_dict in [
            ("base", mgr.global_base_vectors),
            ("prefill", mgr.global_prefill_vectors),
            ("decode", mgr.global_decode_vectors),
        ]:
            norm_key = "norm" if phase_name == "base" else f"{phase_name}_norm"
            for hp_str, layer_vecs in phase_dict.items():
                for layer_idx, vec in layer_vecs.items():
                    norm = vec.norm().item()
                    if norm > 0.0:
                        if layer_idx not in result:
                            result[layer_idx] = {}
                        if hp_str not in result[layer_idx]:
                            result[layer_idx][hp_str] = {}
                        result[layer_idx][hp_str][norm_key] = round(norm, 6)
        return result

    def get_dynamic_steering_status(self) -> dict:
        """Worker-side status for ``GET /v1/steering/dynamic``.

        Everything returned is plain primitives (picklable across the
        ``collective_rpc`` boundary). Per-worker dicts are surfaced
        unaggregated by the router so TP-rank divergence in sync
        consumer decisions is visible by inspection.
        """
        tp_rank, pp_rank = _get_steering_ranks()
        mgr = self._steering_manager
        status: dict = {
            "tp_rank": tp_rank,
            "pp_rank": pp_rank,
            "steering_initialized": mgr is not None,
            # Rolling determinism checksum of applied actions (hex u64) and
            # the fold count. The router compares these across ranks to
            # detect a silent lock-step desync (§6). Plain primitives keep
            # the status dict picklable across the collective_rpc boundary.
            "action_checksum": f"{self._steering_action_checksum & _U64_MASK:016x}",
            "action_count": self._steering_action_count,
        }

        # Async transport (Phase 0 queue).
        queue = get_steering_action_queue()
        if queue is not None:
            qstats = queue.stats()
            status["action_queue"] = {
                "submitted": qstats.submitted,
                "dropped": qstats.dropped,
                "applied": qstats.applied,
                "rejected": qstats.rejected,
            }
        else:
            status["action_queue"] = None

        # Apply-path counters by source (both transports).
        status["apply_stats"] = {
            source: dict(counts)
            for source, counts in getattr(self, "_dynamic_steering_stats", {}).items()
        }

        # Dynamic-override pool occupancy.
        if mgr is not None:
            status["dynamic_pool"] = {
                "capacity": mgr.max_dynamic_steering_configs,
                "in_use": mgr.num_active_dynamic_configs,
                "overrides": dict(getattr(self, "_req_dynamic_decode", {})),
            }
        else:
            status["dynamic_pool"] = None

        # Dynamic additive tier (global decode steering that composes with
        # operator-set steering, §5.4): which hooks/layers carry a tier
        # vector right now.
        if mgr is not None:
            status["dynamic_tier"] = {
                "active": mgr.has_dynamic_tier,
                "gain": mgr.dynamic_tier_gain,
                "hooks": {
                    hook: sorted(layers)
                    for hook, layers in mgr.dynamic_tier_vectors.items()
                },
            }
        else:
            status["dynamic_tier"] = None

        # In-graph monitor (Phase 2, §8): configured probe sites and their
        # policy params. The probe vectors themselves are omitted (large);
        # only the site + threshold/sharpness are reported.
        if mgr is not None:
            status["monitor"] = {
                "active": mgr.has_monitor,
                "sites": {
                    hook: {
                        layer: {
                            "threshold": cfg["threshold"],
                            "sharpness": cfg["sharpness"],
                        }
                        for layer, cfg in layers.items()
                    }
                    for hook, layers in mgr.monitor_configs.items()
                    if layers
                },
            }
        else:
            status["monitor"] = None

        # Per-row (per-request) monitor: configured per-owner sites + params
        # (probe vectors omitted). ``enabled`` reflects the engine opt-in.
        if mgr is not None:
            status["row_monitor"] = {
                "enabled": bool(getattr(self, "_row_monitor_enabled", False)),
                "active": mgr.has_row_monitor,
                "sites": {
                    hook: {
                        layer: {
                            str(owner.legacy_key): {
                                "threshold": cfg["threshold"],
                                "sharpness": cfg["sharpness"],
                            }
                            for owner, cfg in owners.items()
                        }
                        for layer, owners in layers.items()
                        if owners
                    }
                    for hook, layers in mgr._row_monitor.items()
                    if layers
                },
            }
        else:
            status["row_monitor"] = None

        # Per-row strength scales (§5.3): non-default scales by owner.
        if mgr is not None:
            status["dynamic_scales"] = {
                "global": dict(mgr._global_scales),
                "config": {f"{h}:{p}": s for (h, p), s in mgr._config_scales.items()},
                "dynamic": {str(d): s for d, s in mgr._dynamic_scales.items()},
            }
        else:
            status["dynamic_scales"] = None

        # Sync consumers: timing, ring of recent steps, optional
        # consumer-provided policy status.
        consumers = {name: c for name, c in getattr(self, "_sync_consumers", [])}
        sync_status: dict = {}
        for name, stats in getattr(self, "_sync_consumer_stats", {}).items():
            gpu_steps = stats.get("gpu_steps", 0)
            entry = {
                "steps": stats["steps"],
                # Wall-clock span; absorbs the forward-pass GPU drain
                # (the consumer's D2H is the step's first CUDA sync) and
                # is a misleading proxy for added cost — diagnostic only.
                "total_ms": round(stats["total_ms"], 3),
                "max_ms": round(stats["max_ms"], 3),
                # CUDA-event GPU time of only the consumer's own work —
                # the honest added cost. Lags wall time by one step.
                "gpu_steps": gpu_steps,
                "gpu_total_ms": round(stats.get("gpu_total_ms", 0.0), 3),
                "gpu_max_ms": round(stats.get("gpu_max_ms", 0.0), 3),
                "gpu_avg_ms": (
                    round(stats["gpu_total_ms"] / gpu_steps, 3) if gpu_steps else None
                ),
                "gpu_last_ms": stats.get("gpu_last_ms"),
                "over_budget_steps": stats.get("over_budget_steps", 0),
                "ring": [list(item) for item in stats.get("ring", ())],
            }
            consumer = consumers.get(name)
            if consumer is not None and hasattr(consumer, "status"):
                try:
                    entry["status"] = consumer.status()
                except Exception as exc:  # noqa: BLE001 — consumer isolation
                    entry["status"] = {"error": f"{type(exc).__name__}: {exc}"}
            sync_status[name] = entry
        status["sync_consumers"] = sync_status
        return status

    # -----------------------------------------------------------------------
    # Worker-side named steering module registry
    # -----------------------------------------------------------------------

    @staticmethod
    def _module_payload_to_specs(
        payload: dict,
    ) -> tuple[
        SteeringVectorSpec | None,
        SteeringVectorSpec | None,
        SteeringVectorSpec | None,
    ]:
        """Normalize a broadcast payload entry into three tier specs.

        Layer keys may arrive as strings (when the payload round-tripped
        through JSON) or ints (when it was constructed in-process).  We
        coerce to int here so subsequent comparisons against the worker's
        layer-owned set are consistent.
        """

        def _coerce(spec):
            if spec is None:
                return None
            coerced: SteeringVectorSpec = {}
            for hook, layer_dict in spec.items():
                converted: dict[int, object] = {}
                for layer_key, entry in layer_dict.items():
                    converted[int(layer_key)] = entry
                if converted:
                    coerced[hook] = converted  # type: ignore[assignment]
            return coerced or None

        return (
            _coerce(payload.get("vectors")),
            _coerce(payload.get("prefill_vectors")),
            _coerce(payload.get("decode_vectors")),
        )

    @staticmethod
    def _module_payload_to_clamps(
        payload: dict,
    ) -> tuple[
        SteeringClampSpec | None,
        SteeringClampSpec | None,
        SteeringClampSpec | None,
    ]:
        """Normalize a broadcast payload's optional clamp tiers.

        Same int-key coercion as :meth:`_module_payload_to_specs`, applied
        to the ``clamps`` / ``prefill_clamps`` / ``decode_clamps`` keys.
        """

        def _coerce(spec):
            if spec is None:
                return None
            coerced: SteeringClampSpec = {}
            for hook, layer_dict in spec.items():
                converted: dict[int, list] = {}
                for layer_key, entries in layer_dict.items():
                    converted[int(layer_key)] = entries
                if converted:
                    coerced[hook] = converted
            return coerced or None

        return (
            _coerce(payload.get("clamps")),
            _coerce(payload.get("prefill_clamps")),
            _coerce(payload.get("decode_clamps")),
        )

    def register_steering_modules(
        self,
        modules: dict[str, dict],
        replace: bool = False,
    ) -> None:
        """Worker-side handler for the named-module broadcast.

        *modules* maps module name to a dict with optional ``vectors``,
        ``prefill_vectors`` and ``decode_vectors`` (the same shape that
        :class:`SteeringModuleRegistry.dump_for_broadcast` emits).  When
        *replace* is ``True`` the worker's registry is cleared before the
        new entries are stored — used during API-server startup to push
        the initial registry state.

        Mirrors the strict-capacity contract of the rest of the steering
        runtime: requests referencing a name that has not yet been
        broadcast raise loudly in :meth:`_resolve_request_steering`
        rather than silently falling back to inline-only behaviour.
        """
        if replace:
            # Releasing pre-materialized pins before clearing the
            # registry preserves the refcount invariant: every pin taken
            # by a previous register has a matching release.  Without
            # this, a startup ``replace=True`` push that drops a name
            # would leak the row until process exit.
            for prior_name in list(self._steering_module_pinned_rows.keys()):
                self.release_pre_materialized_steering_module(prior_name)
            self._steering_module_registry.clear()
            self._steering_module_resolved_cache.clear()
            self._module_clamps_store().clear()
            self._module_clamps_effective_store().clear()
        for name, payload in modules.items():
            if not isinstance(payload, dict):
                raise SteeringVectorError(
                    f"Steering module '{name}' broadcast payload is not a dict"
                )
            # Re-registering an existing name replaces its vectors, which
            # changes the (hash, phase) the named module resolves to.  Drop
            # the stale pin first so the next pre-materialize call can
            # install a fresh pin against the new hashes; without this
            # the old row leaks until ``unregister_steering_modules``.
            if name in self._steering_module_pinned_rows:
                self.release_pre_materialized_steering_module(name)
            specs = self._module_payload_to_specs(payload)
            self._steering_module_registry[name] = specs
            # Pre-resolve once at registration so the per-request hot path
            # in ``_resolve_request_steering`` can skip the merge + resolve
            # numpy work entirely when there are no inline overrides.
            base_spec, prefill_spec, decode_spec = specs
            self._steering_module_resolved_cache[name] = (
                resolve_effective_vectors(base_spec, prefill_spec),
                resolve_effective_vectors(base_spec, decode_spec),
            )
            # Optional clamps tier: stored in a parallel registry (the
            # 3-tuple registry shape is consumed by the vector resolve
            # path) with a pre-concatenated per-phase effective cache.
            base_c, prefill_c, decode_c = self._module_payload_to_clamps(payload)
            if base_c or prefill_c or decode_c:
                self._module_clamps_store()[name] = (base_c, prefill_c, decode_c)
                self._module_clamps_effective_store()[name] = (
                    resolve_effective_clamps(base_c, prefill_c),
                    resolve_effective_clamps(base_c, decode_c),
                )
            else:
                self._module_clamps_store().pop(name, None)
                self._module_clamps_effective_store().pop(name, None)
        if modules:
            logger.debug(
                "Worker received %d steering module(s) (replace=%s)",
                len(modules),
                replace,
            )

    def unregister_steering_modules(self, names: list[str]) -> None:
        """Drop the listed names from the worker-side registry.

        The pinned refcount the worker held via
        :meth:`pre_materialize_steering_module` is released first so the
        manager's row table can GC the row once the last in-flight
        request that referenced it finishes.  Doing the release before
        dropping the registry entry preserves the invariant that every
        ``(hash, phase)`` pair the worker pinned has a matching release.
        """
        for name in names:
            self.release_pre_materialized_steering_module(name)
            self._steering_module_registry.pop(name, None)
            self._steering_module_resolved_cache.pop(name, None)
            self._module_clamps_store().pop(name, None)
            self._module_clamps_effective_store().pop(name, None)
        if names:
            logger.debug(
                "Worker unregistered %d steering module(s)",
                len(names),
            )

    def pre_materialize_steering_module(self, name: str) -> list[tuple[int, str]]:
        """Eagerly materialize a named module's rows on the manager.

        Called by the API layer immediately after
        ``register_steering_modules`` succeeds, so the ``(hash, phase)``
        rows that requests carrying ``steering_module_ref=(name, 1.0)``
        will resolve to are already populated by the time the first
        such request arrives.  This eliminates the cold-path
        ``register_config.materialize`` cost from the request's TTFT —
        on gemma-3-4b-it/3090 in named_shared mode the first
        request previously paid ~15 ms (almost all of it the
        synchronous bf16 H2D upload of 34 layers × hidden_size in
        :meth:`SteeringManager._stack_vectors_to_device`).

        Refcount semantics:

        * Each pre-materialize call adds ``+1`` to the manager's
          refcount for every ``(hash, phase)`` it materializes — the
          "pinned" reference.
        * Each request that resolves to the same ``(hash, phase)`` adds
          another ``+1`` via the existing per-request register path.
        * Request completion drops by ``+1`` (existing release path).
        * :meth:`release_pre_materialized_steering_module` (called by
          ``unregister_steering_modules``) drops the pinned ``+1``,
          allowing GC once the last in-flight request finishes.

        Idempotent: a second call on the same name is a no-op (the
        first pin remains, no extra refcount is added) so concurrent
        register + first-request paths never double-pin.

        Returns the list of ``(hash, phase)`` pairs pinned by this
        call (empty if the manager is uninitialised, the module has no
        resolved vectors, or the pin was already in place).  The return
        value is consumed by the test surface — production callers
        only care that the row is materialized.
        """
        mgr = self._steering_manager
        if mgr is None:
            return []
        if self._steering_module_pinned_rows.get(name):
            # Already pinned — preserve idempotency. The pin guarantees
            # the row is still there even if every request that
            # transiently bumped the refcount has finished.
            return []
        cached = self._steering_module_resolved_cache.get(name)
        if cached is None:
            return []
        prefill_resolved, decode_resolved = cached
        # The hash format MUST match the one a request carrying
        # ``steering_module_ref=(name, 1.0)`` will compute (see
        # :meth:`SamplingParams.prefill_steering_config_hash` and
        # :func:`hash_steering_config`).  For a named-only request
        # (no inline vectors), the request's
        # ``effective_*_steering`` cached property returns ``None``
        # — only ``module_ref`` enters the digest.  Pre-materialization
        # mirrors that exactly: ``hash_steering_config(None,
        # module_ref=(name, 1.0))``.  Requests carrying inline overrides
        # produce a different (content-derived) hash and fall through to
        # the existing lazy register path.  ``scale=1.0`` is the natural
        # default and the only scale auto-promote ever issues.
        module_ref = (name, 1.0)
        pinned: list[tuple[int, str]] = []
        locally_owned = self._locally_owned_layers
        # Compute the named-only hash once; both phases share the same
        # request hash because ``effective_prefill_steering`` and
        # ``effective_decode_steering`` are both ``None`` for a named-
        # only request.  The (hash, phase) tuple still distinguishes
        # them because ``register_config`` keys on (hash, phase).
        named_only_hash = hash_steering_config(None, module_ref=module_ref)
        clamps_cached = self._module_clamps_effective_store().get(name)
        prefill_clamps = clamps_cached[0] if clamps_cached else None
        decode_clamps = clamps_cached[1] if clamps_cached else None
        for phase, resolved, phase_clamps in (
            ("prefill", prefill_resolved, prefill_clamps),
            ("decode", decode_resolved, decode_clamps),
        ):
            if not resolved and not phase_clamps:
                continue
            mgr.register_config(
                named_only_hash,
                resolved or {},
                phase=phase,
                locally_owned_layers=locally_owned,
                **({"clamps": phase_clamps} if phase_clamps else {}),
            )
            pinned.append((named_only_hash, phase))
        self._steering_module_pinned_rows[name] = pinned
        if pinned:
            logger.debug(
                "Pre-materialized steering module '%s' (%d phase(s))",
                name,
                len(pinned),
            )
        return pinned

    def release_pre_materialized_steering_module(self, name: str) -> None:
        """Release the pinned refcount taken by pre-materialization.

        Drops one ``release_config`` per ``(hash, phase)`` pair that
        :meth:`pre_materialize_steering_module` registered.  If
        in-flight requests still reference the row their per-request
        refcounts keep it alive; the row is GC'd by the existing
        release path when the last request finishes.

        Safe to call when no pin exists — used by
        :meth:`unregister_steering_modules` so the unregister path
        is a single uniform step regardless of whether
        pre-materialization ran.
        """
        mgr = self._steering_manager
        pinned = self._steering_module_pinned_rows.pop(name, None)
        if pinned is None or mgr is None:
            return
        for config_hash, phase in pinned:
            mgr.release_config(config_hash, phase)

    # -----------------------------------------------------------------------
    # Worker-side named probe/steer vector registry
    # -----------------------------------------------------------------------

    @staticmethod
    def _worker_vector_registry() -> WorkerSteeringVectorRegistry:
        """Return the process-global registry, installing one if absent.

        Registration RPCs may (in principle) arrive before
        ``_init_steering_state`` has installed the registry — e.g. a worker
        with no steerable layers that took the early return. Installing on
        demand keeps the register/unregister RPCs total across ranks so their
        replicated state cannot diverge.
        """
        registry = get_worker_steering_vector_registry()
        if registry is None:
            registry = WorkerSteeringVectorRegistry()
            install_worker_steering_vector_registry(registry)
        return registry

    def register_steering_vector_name(
        self,
        name: str,
        kind: str,
        packed: dict,
        digest: str | None = None,
    ) -> None:
        """Worker-side handler for a named-vector register broadcast.

        Stores the unpacked vectors and *digest* under ``name`` in the
        ``kind`` namespace (``"probe"`` / ``"steer"``). Re-registering a name
        replaces its content and digest. Mirrors ``/v1/steering/set``'s
        rank-replicated collective_rpc mutation flow; the engine serializes
        RPCs so every rank applies the same ordered sequence.
        """
        self._worker_vector_registry().register(name, kind, packed, digest)

    def unregister_steering_vector_name(self, name: str, kind: str) -> bool:
        """Worker-side handler for a named-vector unregister broadcast.

        Returns ``True`` if the name existed in the ``kind`` namespace.
        """
        return self._worker_vector_registry().unregister(name, kind)

    def _resolve_request_steering(
        self,
        sp: SamplingParams,
        phase: str,
    ) -> dict[str, dict[int, list[float]]] | None:
        """Resolve the effective steering for a request in the given *phase*.

        Encapsulates the two cases:

        - **Inline-only** (``sp.steering_module_ref`` is ``None``):
          returns the existing ``effective_prefill_steering`` /
          ``effective_decode_steering`` cached property — bit-for-bit
          identical to today.
        - **Named module (+ optional inline overrides)**: looks up the
          named module in ``self._steering_module_registry``, applies
          the request's module-level scale uniformly via
          :func:`scale_steering_spec`, merges the result with any inline
          tier specs via :func:`merge_steering_specs`, then collapses to
          pre-scaled flat vectors via
          :func:`resolve_effective_vectors`.  The merge order matches the
          original server-side ``resolve_for_request`` so semantics are
          preserved.

        Raises :class:`RuntimeError` when the request references a name
        that is missing from the worker's registry.  This matches the
        strict-capacity contract elsewhere in the steering runtime —
        silent fall-through to inline-only would change the request
        payload after the scheduler has already committed to a hash.
        """
        if phase not in ("prefill", "decode"):
            raise ValueError(f"phase must be 'prefill' or 'decode', got {phase!r}")

        ref = sp.steering_module_ref
        if ref is None:
            return (
                sp.effective_prefill_steering
                if phase == "prefill"
                else sp.effective_decode_steering
            )

        name, scale = ref
        module_specs = self._steering_module_registry.get(name)
        if module_specs is None:
            available = sorted(self._steering_module_registry.keys())
            raise RuntimeError(
                f"Steering module '{name}' is not registered on this worker. "
                f"Available: {available or 'none'}.  This indicates the "
                "module-registry RPC has not been broadcast yet, or the "
                "module was unregistered after the request was scheduled."
            )

        inline_phase_spec = (
            sp.prefill_steering_vectors
            if phase == "prefill"
            else sp.decode_steering_vectors
        )

        # Fast path: no inline overrides on either tier.  Use the
        # pre-resolved cache populated at registration time and skip the
        # per-request merge + resolve numpy work.  Profiling on
        # gemma-3-4b-it (3 active hooks) showed this path eliminates
        # ~37 ms/generate of host-side stalls — see
        # docs/features/sae_steering.md "Named-module fast path" for
        # the decomposition.
        if sp.steering_vectors is None and inline_phase_spec is None:
            cached = self._steering_module_resolved_cache.get(name)
            if cached is not None:
                resolved = cached[0] if phase == "prefill" else cached[1]
                if resolved is None:
                    return None
                if scale == 1.0:
                    return resolved
                # Scaled fast path: one numpy multiply per (hook, layer)
                # array vs. the full merge_steering_specs +
                # resolve_effective_vectors machinery.
                return {
                    hook: {layer: arr * scale for layer, arr in layer_dict.items()}
                    for hook, layer_dict in resolved.items()
                }

        # Slow path: inline overrides force a per-request merge + resolve.
        base_spec, prefill_spec, decode_spec = module_specs
        scaled_base = scale_steering_spec(base_spec, scale)
        phase_module_spec = (
            scale_steering_spec(prefill_spec, scale)
            if phase == "prefill"
            else scale_steering_spec(decode_spec, scale)
        )

        merged_base = merge_steering_specs(scaled_base, sp.steering_vectors)
        merged_phase = merge_steering_specs(phase_module_spec, inline_phase_spec)
        return resolve_effective_vectors(merged_base, merged_phase)

    def _resolve_request_clamps(
        self,
        sp: SamplingParams,
        phase: str,
    ) -> SteeringClampSpec | None:
        """Resolve the effective clamps for a request in the given *phase*.

        Clamp sibling of :meth:`_resolve_request_steering`: inline tiers
        come from the ``effective_*_clamps`` cached properties; a named
        module's clamps tier (when the module carries one) is concatenated
        BEFORE the inline entries, mirroring the tier-merge order.  The
        module-level scale does NOT apply to clamps (bounds are absolute
        constraints in unit-projection space, not scalable magnitudes).

        The per-site K cap (``max_clamp_directions``) is enforced here so
        an over-budget request rejects before any manager row is touched.
        Missing-module errors are raised by :meth:`_resolve_request_steering`,
        which every caller invokes alongside this helper.
        """
        if phase not in ("prefill", "decode"):
            raise ValueError(f"phase must be 'prefill' or 'decode', got {phase!r}")

        # getattr-tolerant reads: several worker tests drive this mixin with
        # duck-typed SamplingParams stand-ins that predate the clamp fields.
        inline = getattr(
            sp,
            "effective_prefill_clamps"
            if phase == "prefill"
            else "effective_decode_clamps",
            None,
        )
        module_clamps = None
        ref = getattr(sp, "steering_module_ref", None)
        if ref is not None:
            cached = self._module_clamps_effective_store().get(ref[0])
            if cached is not None:
                module_clamps = cached[0] if phase == "prefill" else cached[1]
        if inline is None and module_clamps is None:
            return None
        max_dirs = getattr(self, "_max_clamp_directions", 0)
        return resolve_effective_clamps(
            module_clamps,
            inline,
            max_directions=max_dirs if max_dirs > 0 else None,
        )

    def _module_clamps_store(self) -> dict:
        """Named-module clamp registry, lazily created.

        ``__dict__.setdefault`` (rather than an ``_init_steering_state``
        attribute alone) keeps duck-typed test hosts that skip init
        working, and avoids a shared class-level mutable default.
        """
        return self.__dict__.setdefault("_steering_module_clamps", {})

    def _module_clamps_effective_store(self) -> dict:
        """Per-phase pre-concatenated module clamp cache, lazily created."""
        return self.__dict__.setdefault("_steering_module_clamps_effective", {})

    # -----------------------------------------------------------------------
    # Per-step buffer / index maintenance
    # -----------------------------------------------------------------------

    def _apply_steering_actions(
        self,
        actions: list,
        *,
        source: str,
        queue: "SteeringActionQueue | None" = None,
    ) -> tuple[int, int]:
        """Validate and apply dynamic steering actions on the step thread.

        The single apply path shared by both transports: the async
        action queue (drained at the top of
        ``_update_steering_buffers``) and sync consumers' ``on_step``
        returns (applied inline post-forward by the runner). Returns
        ``(applied, rejected)`` and records per-``source`` counters.

        Unknown action types are rejected (counted, warned) rather than
        raised — observer isolation extends to malformed action lists.
        """
        applied = 0
        rejected = 0
        # Per-drain-batch ordinal folded into the checksum so the same
        # action applied in different steps yields a different digest.
        batch_ordinal = self._steering_apply_batches + 1
        self._steering_apply_batches = batch_ordinal
        updates: list[SteeringVectorUpdate] = []
        # Declarative per-request overrides *installed* earlier in THIS batch,
        # keyed by req_id. Used to fail closed when a paired per-row monitor is
        # rejected: a declarative this_token+probe+add emits the (unconditional)
        # override first and the per-row monitor second (the monitor's req_id
        # resolves to the override's freshly-registered dyn row). If the monitor
        # is rejected the override would otherwise stick and steer EVERY token
        # unconditionally — the opposite of the client's probe-gated intent, and
        # only a log line. Roll the override back instead. Scoped narrowly: same
        # batch, same req_id, declarative source; operator flows are untouched.
        declarative_installs: dict[str, RequestSteeringOverride] = {}
        for action in actions:
            if isinstance(action, SteeringVectorUpdate):
                updates.append(action)
            elif isinstance(action, RequestSteeringOverride):
                if self._apply_request_override(action, source=source):
                    applied += 1
                    self._fold_steering_action(action, batch_ordinal)
                    if source == DECLARATIVE_SOURCE and action.vectors is not None:
                        declarative_installs[action.req_id] = action
                    else:
                        # A clear (or non-declarative source) supersedes any
                        # tracked install for this request in this batch.
                        declarative_installs.pop(action.req_id, None)
                else:
                    rejected += 1
            elif isinstance(action, SteeringScaleUpdate):
                if self._apply_scale_update(action, source=source):
                    applied += 1
                    self._fold_steering_action(action, batch_ordinal)
                else:
                    rejected += 1
            elif isinstance(action, SteeringMonitorUpdate):
                if self._apply_monitor_update(action, source=source):
                    applied += 1
                    self._fold_steering_action(action, batch_ordinal)
                else:
                    rejected += 1
                    install = (
                        declarative_installs.pop(action.req_id, None)
                        if source == DECLARATIVE_SOURCE and action.req_id is not None
                        else None
                    )
                    if install is not None:
                        # Fail closed: undo the override this batch installed for
                        # the request so its probe-gated steering is not applied
                        # every token unconditionally.
                        rollback = RequestSteeringOverride(
                            req_id=action.req_id,
                            vectors=None,
                            source=DECLARATIVE_SOURCE,
                        )
                        self._apply_request_override(rollback, source=source)
                        # The rollback clear is itself an applied mutation; fold
                        # it so the checksum records the full mutation sequence.
                        self._fold_steering_action(rollback, batch_ordinal)
                        applied -= 1
                        rejected += 1
                        logger.warning(
                            "declarative steering: rolled back per-request "
                            "override for %s because its paired per-row monitor "
                            "was rejected (failing closed rather than steering "
                            "every token unconditionally)",
                            action.req_id,
                        )
            else:
                rejected += 1
                logger.warning(
                    "rejected dynamic steering action from %s: "
                    "unsupported action type %s",
                    source,
                    type(action).__name__,
                )

        if updates:
            if self._steering_manager is None or not self._steerable_layers_cache:
                rejected += len(updates)
                logger.warning(
                    "rejected %d dynamic steering update(s) from %s: "
                    "steering is not initialized on this worker",
                    len(updates),
                    source,
                )
            else:
                ok, bad = apply_steering_updates(
                    updates,
                    self._steering_manager,
                    self._steerable_layers_cache,
                    queue=queue,
                )
                applied += ok
                rejected += bad
                # Fold each update that was actually applied (same accept
                # predicate ``apply_steering_updates`` uses) in list order.
                if ok:
                    for update in updates:
                        if steering_update_accepted(
                            update, self._steerable_layers_cache
                        ):
                            self._fold_steering_action(update, batch_ordinal)

        stats = self._dynamic_steering_stats.setdefault(
            source, {"applied": 0, "rejected": 0}
        )
        stats["applied"] += applied
        stats["rejected"] += rejected
        return applied, rejected

    def _fold_steering_action(self, action, batch_ordinal: int) -> None:
        """Fold one *applied* action into the rolling determinism checksum.

        Cheap (one CRC over a compact digest plus two integer mixes) and
        called only when an action is actually applied, so idle steps pay
        nothing. The digest is a pure, ``PYTHONHASHSEED``-free function of
        the action content; the drain-batch ordinal makes the same action
        applied in different steps fold to a different value. Every TP rank
        in a stage applies the identical stream, so their checksums stay in
        lock-step — a mismatch surfaces a silent per-rank fault.
        """
        digest = _steering_action_digest(action)
        crc = zlib.crc32(digest) & 0xFFFFFFFF
        chk = _mix64(self._steering_action_checksum, crc)
        chk = _mix64(chk, batch_ordinal)
        self._steering_action_checksum = chk
        self._steering_action_count += 1

    def _steering_req_position(self, req_id: str) -> tuple[int, int] | None:
        """Return ``(num_computed_tokens, num_prompt_tokens)`` for ``req_id``.

        The de-fork seam for the per-request decode-only phase guard: the
        shared override apply reads batch position through this hook instead
        of touching a runner-specific batch structure. Returns ``None`` when
        the request is not in the current batch. This default is the v1
        implementation (``self.input_batch``); the v2 runner overrides it to
        read its ``req_states``.
        """
        req_index = self.input_batch.req_id_to_index.get(req_id)
        if req_index is None:
            return None
        num_computed = int(self.input_batch.num_computed_tokens_cpu[req_index])
        num_prompt = int(self.input_batch.num_prompt_tokens[req_index])
        return num_computed, num_prompt

    def _apply_request_override(
        self,
        action: "RequestSteeringOverride",
        *,
        source: str,
    ) -> bool:
        """Apply one per-request dynamic decode override. Returns success.

        Routing-only mutation: allocates/updates/releases a dynamic-pool
        row and records it in ``_req_dynamic_decode``; the request's
        admitted config lifecycle (hashes, refcounts, transition,
        release, scheduler accounting) is untouched. A rejected action
        keeps the previous state — old override if any, else admitted
        routing.

        Known limitation (see docs/design/dynamic_steering.md §5.2):
        scheduler-side steering-aware APC block hashes never see
        overrides, so streaming-continuation cache keys reflect admitted
        steering only.
        """
        mgr = self._steering_manager
        req_id = action.req_id

        def _reject(reason: str) -> bool:
            logger.warning(
                "rejected dynamic steering override (source=%s, req=%s): %s",
                source,
                req_id,
                reason,
            )
            return False

        if mgr is None or not self._steerable_layers_cache:
            return _reject("steering is not initialized on this worker")
        if mgr.max_dynamic_steering_configs <= 0:
            return _reject(
                "dynamic override pool is disabled (max_dynamic_steering_configs=0)"
            )

        # Precedence: a client declarative gate must yield to an
        # operator/server consumer that already owns this request's override.
        owner = self._req_override_source.get(req_id)
        if source == DECLARATIVE_SOURCE and owner is not None and owner != source:
            return _reject(
                f"declarative gate yields: request already steered by '{owner}'"
            )

        existing_dyn_id = self._req_dynamic_decode.get(req_id)

        # Clear: revert to admitted routing. Clearing a request with no
        # live override is a no-op success (idempotent disengage).
        if action.vectors is None:
            if existing_dyn_id is not None:
                self._req_dynamic_decode.pop(req_id, None)
                self._req_override_source.pop(req_id, None)
                mgr.release_dynamic_config(existing_dyn_id)
            return True

        position = self._steering_req_position(req_id)
        if position is None:
            return _reject("request is not in the batch")
        num_computed, num_prompt = position
        if num_computed < num_prompt:
            return _reject(
                "request is still prefilling (overrides are decode-only; "
                "prefill steering feeds prefix-cache keys)"
            )

        # Compose-on-top: fold the request's admitted decode steering delta
        # into the override so ``action.vectors`` adds to (rather than
        # replaces) the client's static decode steering. Resolving the admitted
        # spec can raise if the request's steering module is not registered on
        # this worker; keep prior state rather than escaping the boundary.
        vectors = action.vectors
        if action.compose_admitted:
            rs = self._steering_reqs.get(req_id)
            sp = rs.sampling_params if rs is not None else None
            if sp is not None:
                try:
                    admitted = self._resolve_request_steering(sp, "decode")
                except RuntimeError as exc:
                    return _reject(str(exc))
                if admitted:
                    vectors = merge_steering_specs(admitted, action.vectors)

        try:
            validate_steering_vectors(vectors, self._steerable_layers_cache)
        except SteeringVectorError as exc:
            return _reject(str(exc))

        if existing_dyn_id is not None:
            mgr.update_dynamic_config(
                existing_dyn_id,
                vectors,
                locally_owned_layers=self._locally_owned_layers,
            )
            self._req_override_source[req_id] = source
            return True
        try:
            dyn_id, _row = mgr.register_dynamic_config(
                vectors,
                locally_owned_layers=self._locally_owned_layers,
            )
        except RuntimeError as exc:
            # Pool exhausted: previous state (admitted routing) kept.
            return _reject(str(exc))
        self._req_dynamic_decode[req_id] = dyn_id
        self._req_override_source[req_id] = source
        return True

    def _apply_scale_update(
        self,
        action: "SteeringScaleUpdate",
        *,
        source: str,
    ) -> bool:
        """Apply one per-row strength-scale change (§5.3). Returns success.

        Decode-tier only and routing-light: it sets a manager scale (a
        cheap scales-buffer write on the next step), never touches
        vectors or admission state. A rejected action keeps the previous
        scale.
        """
        mgr = self._steering_manager

        def _reject(reason: str) -> bool:
            logger.warning(
                "rejected steering scale update (source=%s): %s", source, reason
            )
            return False

        if mgr is None or not self._steerable_layers_cache:
            return _reject("steering is not initialized on this worker")
        try:
            validate_steering_scale(action)
        except SteeringVectorError as exc:
            return _reject(str(exc))

        if action.tier_gain:
            mgr.set_dynamic_tier_gain(action.scale)
        elif action.dyn_id is not None:
            if action.dyn_id not in mgr._dynamic_to_row:
                return _reject(f"unknown dynamic row dyn_id={action.dyn_id}")
            mgr.set_dynamic_scale(action.dyn_id, action.scale)
        elif action.req_id is not None:
            # Resolve req_id -> the request's live dynamic-override row.
            # Lets a sync consumer modulate a per-request override's
            # strength cheaply without ever seeing the internal dyn_id.
            owner = self._req_override_source.get(action.req_id)
            if source == DECLARATIVE_SOURCE and owner is not None and owner != source:
                return _reject(
                    f"declarative gate yields: request {action.req_id} already "
                    f"steered by '{owner}'"
                )
            dyn_id = self._req_dynamic_decode.get(action.req_id)
            if dyn_id is None:
                return _reject(
                    f"request {action.req_id} has no live dynamic override to scale"
                )
            mgr.set_dynamic_scale(dyn_id, action.scale)
        elif action.config_hash is not None:
            mgr.set_row_scale(action.config_hash, "decode", action.scale)
        else:
            mgr.set_global_scale("decode", action.scale)
        return True

    def _apply_monitor_update(
        self,
        action: "SteeringMonitorUpdate",
        *,
        source: str,
    ) -> bool:
        """Configure/clear the in-graph monitor at a probe site (§8).

        Two modes by targeting. Untargeted ⇒ the GLOBAL monitor (one probe
        per site, gates the dynamic tier and, when ``gate_rows``, all rows).
        Targeted (``req_id``/``config_hash``/``dyn_id``) ⇒ a PER-ROW monitor
        on that owner's decode row only — true per-request gating, requires
        ``enable_row_monitor``. Decode-tier/decode-row only and cache-safe by
        construction. A rejected action keeps the previous monitor state;
        ``probe=None`` clears the target.
        """
        mgr = self._steering_manager

        def _reject(reason: str) -> bool:
            logger.warning(
                "rejected steering monitor update (source=%s): %s", source, reason
            )
            return False

        if mgr is None or not self._steerable_layers_cache:
            return _reject("steering is not initialized on this worker")
        try:
            validate_steering_monitor(action, self._steerable_layers_cache)
        except SteeringVectorError as exc:
            return _reject(str(exc))

        targeted = (
            action.req_id is not None
            or action.config_hash is not None
            or action.dyn_id is not None
        )
        if not targeted:
            # GLOBAL monitor (unchanged behavior).
            if action.probe is None:
                mgr.clear_monitor(action.hook, action.layer)
                return True
            probe = torch.from_numpy(
                np.ascontiguousarray(action.probe, dtype=np.float32)
            )
            mgr.set_monitor(
                action.hook,
                action.layer,
                probe,
                action.threshold,
                action.sharpness,
                gate_rows=action.gate_rows,
                locally_owned_layers=self._locally_owned_layers,
            )
            return True

        # PER-ROW (per-request) monitor.
        if not getattr(self, "_row_monitor_enabled", False):
            return _reject(
                "per-row monitor requires the engine flag enable_row_monitor"
            )
        if action.req_id is not None:
            owner = self._req_override_source.get(action.req_id)
            if source == DECLARATIVE_SOURCE and owner is not None and owner != source:
                return _reject(
                    f"declarative gate yields: request {action.req_id} already "
                    f"steered by '{owner}'"
                )
            dyn_id = self._req_dynamic_decode.get(action.req_id)
            if dyn_id is None:
                return _reject(
                    f"request {action.req_id} has no live dynamic override to "
                    "attach a per-row monitor to"
                )
            owner_key: RowOwner = RowOwner.dyn(dyn_id)
        elif action.dyn_id is not None:
            if action.dyn_id not in mgr._dynamic_to_row:
                return _reject(f"unknown dynamic row dyn_id={action.dyn_id}")
            owner_key = RowOwner.dyn(action.dyn_id)
        else:
            owner_key = RowOwner.config(int(action.config_hash), "decode")

        if action.probe is None:
            mgr.clear_row_monitor(action.hook, action.layer, owner_key)
            return True
        probe = torch.from_numpy(np.ascontiguousarray(action.probe, dtype=np.float32))
        mgr.set_row_monitor(
            action.hook,
            action.layer,
            owner_key,
            probe,
            action.threshold,
            action.sharpness,
            locally_owned_layers=self._locally_owned_layers,
        )
        return True

    def _drop_request_dynamic_override(self, req_id: str) -> None:
        """Release ``req_id``'s dynamic override, if any. Idempotent.

        ``release_dynamic_config`` also purges the row's per-row monitor and
        strength scale, so a per-request declarative gate leaves no residue.
        """
        self._req_override_source.pop(req_id, None)
        dyn_id = self._req_dynamic_decode.pop(req_id, None)
        if dyn_id is not None and self._steering_manager is not None:
            self._steering_manager.release_dynamic_config(dyn_id)

    def _steering_batch_view(self) -> SteeringBatchView:
        """Build the per-step batch view for the unified hot path (v1 default).

        v1's per-request arrays are batch-ordered, so the slot->state-row map
        is the identity; ``num_computed_tokens_cpu`` / ``num_prompt_tokens``
        index directly by batch position. The v2 runner overrides this to
        route through ``idx_mapping`` + ``RequestState``. Returns one reusable
        instance mutated in place — the identity map is grown lazily to the
        batch size, so steady state allocates nothing.
        """
        ib = self.input_batch
        num_reqs = ib.num_reqs
        ident = self._steering_idx_identity
        if ident is None or ident.shape[0] < num_reqs:
            ident = np.arange(max(num_reqs, 1), dtype=np.int64)
            self._steering_idx_identity = ident
        bv = self._steering_bview
        if bv is None:
            bv = SteeringBatchView(
                num_reqs=num_reqs,
                req_ids=ib.req_ids,
                idx_np=ident,
                num_computed_np=ib.num_computed_tokens_cpu,
                num_prompt_np=ib.num_prompt_tokens,
            )
            self._steering_bview = bv
        else:
            bv.num_reqs = num_reqs
            bv.req_ids = ib.req_ids
            bv.idx_np = ident
            bv.num_computed_np = ib.num_computed_tokens_cpu
            bv.num_prompt_np = ib.num_prompt_tokens
        return bv

    def _compute_decode_signature_deltas(
        self, scheduler_output: "SchedulerOutput", bview: SteeringBatchView
    ) -> dict[str, int]:
        """Per-request effective-decode-signature *deltas* for the scheduler.

        For each request in decode this step, compute its effective decode
        steering signature (admitted config folded with any override / tier
        / monitor; see the manager). Report only requests whose signature
        changed since the last report — a request reverting to admitted
        steering reports its plain admitted hash so the scheduler keys its
        future decode blocks back to the admitted config. The result rides
        ``ModelRunnerOutput.steering_decode_signatures`` to
        ``Scheduler.update_from_output`` (rank 0's output is canonical; the
        signature is rank-identical). See
        docs/design/dynamic_steering_apc_notification.md.

        Reads only the broadcast ``scheduler_output`` and rank-replicated
        state (``bview`` + ``_steering_reqs``), so the result is
        TP-deterministic. The base decode hash comes from ``_steering_reqs``
        (the canonical per-request steering identity), not any batch column.
        """
        mgr = self._steering_manager
        if mgr is None:
            return {}
        reqs = self._steering_reqs
        num_reqs = bview.num_reqs
        req_ids = bview.req_ids
        idx_np = bview.idx_np
        num_computed_np = bview.num_computed_np
        prompt_len_np = bview.num_prompt_np
        deltas: dict[str, int] = {}
        seen: set[str] = set()
        for i in range(num_reqs):
            req_id = req_ids[i]
            if req_id is None:
                continue
            if scheduler_output.num_scheduled_tokens.get(req_id, 0) == 0:
                continue
            req_idx = int(idx_np[i])
            num_computed = int(num_computed_np[req_idx])
            rs = reqs.get(req_id)
            num_prompt = (
                rs.num_prompt_tokens if rs is not None else int(prompt_len_np[req_idx])
            )
            if num_computed < num_prompt:
                # Prefill: decode steering (and its signature) does not apply;
                # prefill cache keys are admission-fixed and already correct.
                continue
            seen.add(req_id)
            base = rs.decode_hash if rs is not None else 0
            dyn_id = self._req_dynamic_decode.get(req_id)
            sig = mgr.effective_decode_signature(dyn_id, base)
            report_val = base if sig is None else sig
            if self._req_decode_sig_reported.get(req_id) != report_val:
                deltas[req_id] = report_val
                self._req_decode_sig_reported[req_id] = report_val
        # Drop reported state for requests no longer in the decode batch
        # (bounded memory; a re-appearing request simply re-reports — the
        # scheduler applies it idempotently).
        if self._req_decode_sig_reported:
            for rid in [r for r in self._req_decode_sig_reported if r not in seen]:
                self._req_decode_sig_reported.pop(rid, None)
        return deltas

    def _update_steering_buffers(self, scheduler_output: "SchedulerOutput") -> None:
        """Update per-layer steering tables and the shared steering index.

        Each step:
        1. Populate each layer's per-hook steering_table from the manager
        2. Build the steering_index mapping tokens to table rows

        The ``SteeringManager`` is constructed eagerly during model
        load by ``_init_steering_state``.  When steering is disabled
        or no steerable layers exist, the manager is ``None`` and this
        function short-circuits — model code (e.g. Gemma3) registers
        per-layer steering_table buffers unconditionally so the forward
        path stays branch-free.

        This is the shared per-step hot path for both runners. The only
        runner-specific state — batch order + per-request token counts — is
        read through :meth:`_steering_batch_view`; per-request steering
        identity (hashes + phase) comes from the canonical ``_steering_reqs``
        store. The body reads only the broadcast ``scheduler_output`` and
        rank-replicated state, so every buffer it writes is TP-deterministic.
        """
        if self._steering_manager is None or not self._steerable_layers_cache:
            self._pending_decode_sigs = {}
            return

        # Fresh each step; populated at the exits below so the model runner
        # attaches this step's effective-decode-signature deltas (APC).
        self._pending_decode_sigs = {}

        # Dynamic steering, async transport: drain the in-process action
        # queue before anything else so updates submitted during step N
        # (by a capture consumer on the dispatch thread) are visible to
        # the tables built for step N+1. Must run before the
        # nothing-active short-circuit below — a drained update may be
        # exactly what activates steering. Application sets
        # ``_tables_dirty``, so the existing populate path uploads the
        # new state; no new buffer code path. Empty-queue steady state
        # costs one global read and one truthiness check per step.
        # (The sync transport — sync consumers' ``on_step`` returns —
        # applies through the same ``_apply_steering_actions`` path,
        # inline at the end of the previous step.)
        action_queue = get_steering_action_queue()
        if action_queue is not None and action_queue:
            self._apply_steering_actions(
                action_queue.drain(),
                source="action_queue",
                queue=action_queue,
            )

        bview = self._steering_batch_view()
        reqs = self._steering_reqs
        num_reqs = bview.num_reqs
        req_ids = bview.req_ids
        idx_np = bview.idx_np
        num_computed_np = bview.num_computed_np
        prompt_len_np = bview.num_prompt_np

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
        # A request carrying a per-request steering hash whose config is not
        # yet registered must defeat the short-circuit. The decode-only case
        # (prefill_hash == 0, decode_hash != 0) registers its decode config
        # lazily at the prefill->decode transition in the loop below; if the
        # short-circuit returns first, that transition never runs, the config
        # is never registered, and the steering is silently dropped forever
        # (config_to_row stays empty, so the short-circuit keeps firing). The
        # admitted prefill config is registered at admission and shows up in
        # config_to_row, but a decode-only request has none — hence the
        # explicit batch scan here. A missed predicate below silently serves
        # stale buffers, so the list must stay exhaustive.
        batch_has_per_request_steering = any(
            (rs := reqs.get(req_ids[i])) is not None
            and (rs.prefill_hash != 0 or rs.decode_hash != 0)
            for i in range(num_reqs)
        )
        if (
            not batch_has_per_request_steering
            and not self._steering_manager.config_to_row
            and not self._steering_manager.has_dynamic
            and not self._steering_manager.has_dynamic_tier
            and not self._steering_manager.has_monitor
            and not self._steering_manager.has_row_monitor
            and not self._steering_manager.global_base_vectors
            and not self._steering_manager.global_prefill_vectors
            and not self._steering_manager.global_decode_vectors
            # getattr-tolerant: duck-typed test managers predate clamps.
            and not getattr(self._steering_manager, "has_global_clamps", False)
        ):
            if self._steering_index_dirty:
                any_layer = next(iter(self._steerable_layers_cache.values()))
                steering_index = cast(torch.Tensor, any_layer.steering_index)
                steering_index.zero_()
                # Also clear the per-token dynamic-tier gate so a stale gate
                # doesn't apply a now-removed tier.
                tscales = getattr(any_layer, "steering_token_scales", None)
                if tscales is not None:
                    tscales.zero_()
                # Reset the Phase 2 row gate to 1.0 and clear the decode
                # mask so a stale monitor reduction doesn't gate now-removed
                # steering rows.
                rgate = getattr(any_layer, "steering_row_gate", None)
                if rgate is not None:
                    rgate.fill_(1.0)
                dmask = getattr(any_layer, "steering_decode_mask", None)
                if dmask is not None:
                    dmask.zero_()
                # Nothing-active transition: clear every per-layer
                # ``_any_active`` flag so apply_steering short-circuits on
                # this and subsequent steps.  Mirrors the index zero-out
                # above — only paid on the active->inactive transition;
                # steady-state inactive runs skip this branch entirely
                # (``_steering_index_dirty`` stays False).
                for mod in self._steerable_layers_cache.values():
                    for hp in SteeringHookPoint:
                        flag_buf = getattr(mod, HOOK_POINT_ANY_ACTIVE_ATTR[hp], None)
                        if flag_buf is not None:
                            flag_buf.zero_()
                        # Also deactivate any in-graph monitor (Phase 2) so a
                        # stale probe doesn't gate a now-removed tier. Mirrors
                        # the any-active zero-out; only paid on the transition.
                        mon_buf = getattr(mod, HOOK_POINT_MONITOR_ACTIVE_ATTR[hp], None)
                        if mon_buf is not None:
                            mon_buf.zero_()
                        # Same for the per-row monitor active flag.
                        row_buf = getattr(mod, HOOK_POINT_ROW_ACTIVE_ATTR[hp], None)
                        if row_buf is not None:
                            row_buf.zero_()
                        # And the directional-clamp active flag, so stale
                        # clamp rows never fire after the last clamp state
                        # is removed.
                        clamp_buf = getattr(mod, CLAMP_ANY_ACTIVE_ATTR[hp], None)
                        if clamp_buf is not None:
                            clamp_buf.zero_()
                self._steering_index_dirty = False
            # Nothing dynamic is active; revert any request still reported as
            # dynamically steered back to its admitted decode key.
            self._pending_decode_sigs = self._compute_decode_signature_deltas(
                scheduler_output, bview
            )
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
        elif self._steering_manager._scales_dirty:
            # Cheap path (§5.3): a strength-scale change needs only the
            # per-row scale buffers rewritten — no table recompose, no
            # vector H2D.
            self._steering_manager.populate_steering_scales(
                self._steerable_layers_cache
            )

        # 2. Build steering index
        # Get the shared steering_index buffer (all layers share one tensor)
        any_layer = next(iter(self._steerable_layers_cache.values()))
        steering_index = cast(torch.Tensor, any_layer.steering_index)

        # Vectorized build: walk requests once to record each request's
        # table row + token count into pre-allocated CPU int64 scratch
        # buffers, then expand the row-per-request array into a
        # row-per-token array via ``np.repeat`` and copy the whole
        # thing to the GPU in a single non-blocking H2D.  Replaces
        # ``num_reqs`` independent ``_set_item`` kernel launches per
        # step with one ``copy_``.
        rows_scratch = self._steering_rows_scratch
        n_tokens_scratch = self._steering_n_tokens_scratch
        index_pinned = self._steering_index_pinned
        tier_gain_scratch = self._steering_tier_gain_scratch
        token_scales_pinned = self._steering_token_scales_pinned
        decode_mask_scratch = self._steering_decode_mask_scratch
        decode_mask_pinned = self._steering_decode_mask_pinned
        assert rows_scratch is not None
        assert n_tokens_scratch is not None
        assert index_pinned is not None
        assert tier_gain_scratch is not None
        assert token_scales_pinned is not None
        assert decode_mask_scratch is not None
        assert decode_mask_pinned is not None

        # Grow per-request scratches if the batch ever exceeds the
        # initial sizing.  This is defensive — ``max_num_seqs`` should
        # bound ``num_reqs`` — but cheap to handle.
        if rows_scratch.shape[0] < num_reqs:
            rows_scratch = np.zeros(num_reqs, dtype=np.int64)
            n_tokens_scratch = np.zeros(num_reqs, dtype=np.int64)
            tier_gain_scratch = np.zeros(num_reqs, dtype=np.float32)
            decode_mask_scratch = np.zeros(num_reqs, dtype=np.float32)
            self._steering_rows_scratch = rows_scratch
            self._steering_n_tokens_scratch = n_tokens_scratch
            self._steering_tier_gain_scratch = tier_gain_scratch
            self._steering_decode_mask_scratch = decode_mask_scratch

        # Per-token dynamic-tier gate (§5.4): the gain for decode tokens of
        # a tier-active state, 0 otherwise (so the tier stays decode-only).
        tier_gain = (
            self._steering_manager.dynamic_tier_gain
            if self._steering_manager.has_dynamic_tier
            else 0.0
        )

        active_count = 0
        for i in range(num_reqs):
            req_id = req_ids[i]
            n_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if n_tokens == 0:
                continue

            req_idx = int(idx_np[i])
            # Defensive (v1 parity): a request scheduled this step whose batch
            # state row can't be resolved (a scheduler/runner race) routes to
            # row 0 — the no-steer sentinel — and shields the array reads below
            # from an out-of-range index. Never fires under the normal
            # invariant (every scheduled request has a valid state row); cheap
            # to keep.
            if req_idx < 0 or req_idx >= num_computed_np.shape[0]:
                rows_scratch[active_count] = 0
                n_tokens_scratch[active_count] = n_tokens
                tier_gain_scratch[active_count] = 0.0
                decode_mask_scratch[active_count] = 0.0
                active_count += 1
                continue

            # Steering identity (hashes + prompt length + phase) comes from the
            # canonical per-request store. An untracked request (no per-request
            # config) routes with hash 0 so any global vectors apply — the
            # manager maps hash 0 to the global prefill/decode row (or the row-0
            # no-steer sentinel when no globals are set).
            num_computed = int(num_computed_np[req_idx])
            rs = reqs.get(req_id)
            if rs is not None:
                num_prompt = rs.num_prompt_tokens
                prefill_hash = rs.prefill_hash
                decode_hash = rs.decode_hash
            else:
                num_prompt = int(prompt_len_np[req_idx])
                prefill_hash = 0
                decode_hash = 0

            if num_computed < num_prompt:
                # Prefill: use prefill steering hash.
                row = self._steering_manager.get_row_for_config(
                    prefill_hash, is_prefill=True
                )
                rows_scratch[active_count] = row
                n_tokens_scratch[active_count] = n_tokens
                # Prefill tokens never get the dynamic tier (cache safety).
                tier_gain_scratch[active_count] = 0.0
                # Prefill rows are never row-gated (cache safety).
                decode_mask_scratch[active_count] = 0.0

                # Check if this request will transition to decode after
                # this step's tokens are processed. Must happen in this
                # same pass — the registration / refcount semantics are
                # externally observable. Only tracked requests transition
                # (an untracked request has no config to release/register).
                if rs is not None and num_computed + n_tokens >= num_prompt:
                    self._steering_transition(rs)
            else:
                # Decode: a live dynamic override routes this request's
                # tokens to its dynamic-pool row INSTEAD of the admitted
                # config's row. Pure routing — the admitted config stays
                # registered (refcounts, transition, release all proceed
                # as if the override didn't exist).
                dyn_id = self._req_dynamic_decode.get(req_id)
                if dyn_id is not None:
                    row = self._steering_manager.get_dynamic_row(dyn_id)
                else:
                    row = self._steering_manager.get_row_for_config(
                        decode_hash, is_prefill=False
                    )
                rows_scratch[active_count] = row
                n_tokens_scratch[active_count] = n_tokens
                # Decode tokens carry the dynamic-tier gate.
                tier_gain_scratch[active_count] = tier_gain
                # Decode tokens are eligible for in-graph row gating.
                decode_mask_scratch[active_count] = 1.0

            active_count += 1

        # Single non-blocking H2D copy: expand per-request rows into
        # the per-token row array (written into the pre-allocated
        # pinned-memory scratch), then copy that prefix to the GPU
        # in one shot.
        if active_count > 0:
            expanded = np.repeat(
                rows_scratch[:active_count],
                n_tokens_scratch[:active_count],
            )
            n_expanded = int(expanded.shape[0])
            # Cap to the device buffer size; the scheduler enforces
            # this bound but cap defensively to avoid out-of-range
            # writes if upstream invariants ever drift.
            n_expanded = min(n_expanded, index_pinned.shape[0], steering_index.shape[0])
            # Stage in the pinned scratch so the copy is genuinely
            # asynchronous on CUDA devices.
            index_pinned[:n_expanded].copy_(torch.from_numpy(expanded[:n_expanded]))
            steering_index[:n_expanded].copy_(
                index_pinned[:n_expanded], non_blocking=True
            )
        else:
            n_expanded = 0

        # Zero out remaining positions so old tokens past the active
        # prefix read row 0 (the no-steering sentinel).
        if n_expanded < steering_index.shape[0]:
            steering_index[n_expanded:].zero_()

        # Per-token dynamic-tier gate (§5.4): same expand + H2D as the
        # index. tier_gain_scratch is 0 for prefill / non-tier tokens and
        # ``dynamic_tier_gain`` for decode tokens of a tier-active state.
        token_scales = cast(torch.Tensor, any_layer.steering_token_scales)
        if active_count > 0:
            gate_expanded = np.repeat(
                tier_gain_scratch[:active_count],
                n_tokens_scratch[:active_count],
            )
            n_gate = min(
                int(gate_expanded.shape[0]),
                token_scales_pinned.shape[0],
                token_scales.shape[0],
            )
            token_scales_pinned[:n_gate].copy_(torch.from_numpy(gate_expanded[:n_gate]))
            token_scales[:n_gate].copy_(token_scales_pinned[:n_gate], non_blocking=True)
        else:
            n_gate = 0
        if n_gate < token_scales.shape[0]:
            token_scales[n_gate:].zero_()

        # Phase 2 row gating: reset the per-token row gate to 1.0 (rows at
        # full strength) so any monitor reduction from the previous step is
        # cleared — the in-graph monitor reduces it again this forward. And
        # write the decode mask (1.0 decode / 0.0 prefill) so the monitor
        # only gates decode rows. Both are no-ops downstream unless a
        # row-gating monitor is active. Mirrors the token_scales expand+H2D.
        row_gate = cast(torch.Tensor, any_layer.steering_row_gate)
        decode_mask = cast(torch.Tensor, any_layer.steering_decode_mask)
        row_gate.fill_(1.0)
        if active_count > 0:
            mask_expanded = np.repeat(
                decode_mask_scratch[:active_count],
                n_tokens_scratch[:active_count],
            )
            n_mask = min(
                int(mask_expanded.shape[0]),
                decode_mask_pinned.shape[0],
                decode_mask.shape[0],
            )
            decode_mask_pinned[:n_mask].copy_(torch.from_numpy(mask_expanded[:n_mask]))
            decode_mask[:n_mask].copy_(decode_mask_pinned[:n_mask], non_blocking=True)
        else:
            n_mask = 0
        if n_mask < decode_mask.shape[0]:
            decode_mask[n_mask:].zero_()

        # Mark the index as having non-zero row references this step. The
        # no-active-state short-circuit on a future step will zero the index
        # if needed when transitioning back to "nothing active".
        self._steering_index_dirty = True

        # Effective-decode-signature deltas for APC (computed from the
        # steering state as applied THIS step — before any sync consumer
        # mutates it for the next step).
        self._pending_decode_sigs = self._compute_decode_signature_deltas(
            scheduler_output, bview
        )

    # -----------------------------------------------------------------------
    # Canonical per-request steering lifecycle
    #
    # These methods own ``self._steering_reqs`` — the single source of truth
    # for a request's registered steering identity + phase — and are driven
    # identically by both runners. v1 calls them from ``_update_states`` /
    # ``_update_streaming_request``; v2 from ``add_requests`` /
    # ``finish_requests``. See docs/design/v2_runner_steering_capture.md.
    # -----------------------------------------------------------------------

    def _steering_add_request(self, new_req_data: "NewRequestData") -> None:
        """Track a newly admitted request and register its initial config.

        Also covers streaming re-adds (the caller removes the prior instance
        first): any state we already held for this id is released before the
        fresh prefill config is registered.
        """
        if self._steering_manager is None:
            return
        num_prompt = length_from_prompt_token_ids_or_embeds(
            new_req_data.prompt_token_ids,
            new_req_data.prompt_embeds,
        )
        self._steering_register_request(
            new_req_data.req_id,
            sampling_params=new_req_data.sampling_params,
            prefill_hash=new_req_data.prefill_steering_config_hash,
            decode_hash=new_req_data.decode_steering_config_hash,
            num_prompt_tokens=num_prompt,
            num_computed_tokens=new_req_data.num_computed_tokens,
        )

    def _steering_register_request(
        self,
        req_id: str,
        *,
        sampling_params: "SamplingParams | None",
        prefill_hash: int,
        decode_hash: int,
        num_prompt_tokens: int,
        num_computed_tokens: int,
    ) -> None:
        """Register fresh steering state for ``req_id``.

        The core shared by admission, streaming re-add, and preemption resume.
        Any state we still hold for this id is released first (idempotent when
        nothing is tracked). A (re-)registration re-enters prefill semantics,
        so any live dynamic decode override belongs to the prior run and is
        dropped — the driving policy re-engages on the continuation's decode.

        A full prefix-cache hit (``num_computed >= num_prompt``) admits the
        request directly into decode. The scheduler reserves the matching row
        at admission/resume, so ``register_config`` is expected to succeed; a
        ``RuntimeError`` indicates a scheduler accounting bug and propagates.
        """
        mgr = self._steering_manager
        if mgr is None:
            return

        self._drop_request_dynamic_override(req_id)
        old = self._steering_reqs.pop(req_id, None)
        if old is not None:
            self._steering_release_state(old)

        sp = sampling_params
        if sp is None or (prefill_hash == 0 and decode_hash == 0):
            return

        rs = _SteeringReqState(
            sampling_params=sp,
            prefill_hash=prefill_hash,
            decode_hash=decode_hash,
            num_prompt_tokens=num_prompt_tokens,
            phase="prefill",
        )
        self._steering_reqs[req_id] = rs

        if num_computed_tokens >= num_prompt_tokens:
            # Already past prefill — register the decode config now.
            # Clamp-only configs (nonzero hash, empty effective vectors)
            # must still register: the hash reserved a scheduler row, and
            # skipping here would crash get_row_for_config downstream.
            effective_decode = self._resolve_request_steering(sp, "decode")
            decode_clamps = self._resolve_request_clamps(sp, "decode")
            if decode_hash != 0 and (effective_decode or decode_clamps):
                # ``clamps=`` only when present: clamp-free requests keep the
                # original call shape (duck-typed test managers rely on it).
                mgr.register_config(
                    decode_hash,
                    effective_decode or {},
                    phase="decode",
                    locally_owned_layers=self._locally_owned_layers,
                    **({"clamps": decode_clamps} if decode_clamps else {}),
                )
            rs.phase = "decode"
        else:
            # Normal: start in prefill; the decode config is registered lazily
            # at the prefill->decode boundary in _update_steering_buffers.
            effective_prefill = self._resolve_request_steering(sp, "prefill")
            prefill_clamps = self._resolve_request_clamps(sp, "prefill")
            if prefill_hash != 0 and (effective_prefill or prefill_clamps):
                mgr.register_config(
                    prefill_hash,
                    effective_prefill or {},
                    phase="prefill",
                    locally_owned_layers=self._locally_owned_layers,
                    **({"clamps": prefill_clamps} if prefill_clamps else {}),
                )
            rs.phase = "prefill"

    def _steering_finish_requests(self, req_ids: "set[str] | list[str]") -> None:
        """Release configs for finished (or preempted) requests.

        Preempted requests are released too: they re-enter through the
        admission / resume path, which re-registers a fresh prefill config.
        Reads only ``self._steering_reqs``, so the ordering relative to the
        runner popping its own request state is not load-bearing.
        """
        if self._steering_manager is None:
            return
        for req_id in req_ids:
            # Drop any live dynamic decode override (routing state local to the
            # finished/preempted decode run).
            self._drop_request_dynamic_override(req_id)
            rs = self._steering_reqs.pop(req_id, None)
            if rs is not None:
                self._steering_release_state(rs)

    def _steering_release_state(self, rs: _SteeringReqState) -> None:
        """Release the config for whichever phase ``rs`` is currently in."""
        mgr = self._steering_manager
        if mgr is None:
            return
        if rs.phase == "prefill" and rs.prefill_hash != 0:
            mgr.release_config(rs.prefill_hash, "prefill")
        elif rs.phase == "decode" and rs.decode_hash != 0:
            mgr.release_config(rs.decode_hash, "decode")

    def _steering_transition(self, rs: _SteeringReqState) -> None:
        """Handle a request crossing the prefill->decode boundary this step.

        Releases the prefill config and registers the decode config so it is
        ready for the next step's table population. The scheduler reserves the
        decode row at the step prefill completes, so ``register_config`` is
        expected to succeed; a ``RuntimeError`` indicates a scheduler
        accounting bug and propagates.
        """
        mgr = self._steering_manager
        assert mgr is not None, (
            "_steering_transition called without an initialised manager"
        )
        if rs.prefill_hash != 0:
            mgr.release_config(rs.prefill_hash, "prefill")
        if rs.decode_hash != 0:
            effective_decode = self._resolve_request_steering(
                rs.sampling_params, "decode"
            )
            decode_clamps = self._resolve_request_clamps(rs.sampling_params, "decode")
            if effective_decode or decode_clamps:
                mgr.register_config(
                    rs.decode_hash,
                    effective_decode or {},
                    phase="decode",
                    locally_owned_layers=self._locally_owned_layers,
                    **({"clamps": decode_clamps} if decode_clamps else {}),
                )
        rs.phase = "decode"

    def _reset_steering_for_resumption(
        self,
        req_id: str,
        req_state: "CachedRequestState",
        new_num_computed_tokens: int,
    ) -> None:
        """Re-register steering for a preempted request resumed by v1.

        Under release-at-preemption (de-fork step D), a preempted request's
        configs + dynamic override were freed at preemption time by
        ``_steering_finish_requests``. On resume the request re-enters with
        ``new_num_computed_tokens`` and must re-register fresh, exactly as
        admission does. The resume arrives on v1 through
        ``scheduled_cached_reqs`` (no ``NewRequestData``), so the fields come
        from the ``CachedRequestState``; ``_steering_register_request``
        defensively releases any state that somehow survived, keeping the path
        idempotent.
        """
        self._steering_register_request(
            req_id,
            sampling_params=req_state.sampling_params,
            prefill_hash=req_state.prefill_steering_config_hash,
            decode_hash=req_state.decode_steering_config_hash,
            num_prompt_tokens=req_state.num_prompt_tokens,
            num_computed_tokens=new_num_computed_tokens,
        )
