# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define activation steering functionality mixin for model runners.
"""

import base64
import binascii
import math
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn

from vllm.config.sae_steering_types import SAEActivation
from vllm.config.steering_types import (
    SteeringVectorSpec,
    hash_steering_config,
    merge_steering_specs,
    resolve_effective_vectors,
    scale_steering_spec,
    validate_steering_index,
)
from vllm.entrypoints.openai.steering.registry import (
    SAEModuleManifest,
    sae_manifest_from_dict,
)
from vllm.exceptions import SteeringVectorError
from vllm.logger import init_logger
from vllm.model_executor.layers.sae_full_reconstruction import (
    HOOK_POINT_FR_DECODER_BIAS_ATTR,
    HOOK_POINT_FR_DECODER_WEIGHT_ATTR,
    HOOK_POINT_FR_ENCODER_BIAS_ATTR,
    HOOK_POINT_FR_ENCODER_WEIGHT_ATTR,
    HOOK_POINT_FR_THRESHOLD_ATTR,
    populate_sae_full_recon_clamp_table,
    register_sae_full_recon_buffers,
    register_sae_recon_index_buffer,
    sae_full_recon_buffers_attached,
    share_sae_recon_index_across_layers,
    unregister_sae_full_recon_buffers,
)
from vllm.model_executor.layers.sae_steering import (
    get_sae_slot_state,
    populate_sae_clamp_table,
    register_sae_buffers,
    register_sae_index_buffer,
    sae_buffers_attached,
    share_sae_index_across_layers,
    unregister_sae_buffers,
)
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
from vllm.v1.worker.sae_clamp_manager import SAEClampManager
from vllm.v1.worker.sae_full_reconstruction_manager import (
    SAEFullReconstructionManager,
)
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


def _coerce_sae_wire_site_key(key: object) -> tuple[int, str]:
    """Normalize one ``sae_weights`` site key to ``(layer_idx, hook_str)``.

    Accepts the in-process ``(int, str)`` tuple, the msgpack-degraded
    ``[layer, hook]`` list, and the Rust frontend's ``"layer:hook"``
    string. Anything else raises a clear :class:`ValueError`.
    """
    if isinstance(key, (tuple, list)) and len(key) == 2:
        layer, hook = key
        if (
            isinstance(layer, int)
            and not isinstance(layer, bool)
            and isinstance(hook, str)
        ):
            return (layer, hook)
    elif isinstance(key, str):
        layer_str, sep, hook = key.partition(":")
        if sep and hook:
            try:
                return (int(layer_str), hook)
            except ValueError:
                pass
    raise ValueError(
        "SAE weights site key must be an (int layer, str hook) pair or a "
        f"'layer:hook' string, got {key!r}."
    )


def _coerce_sae_wire_tensor(
    value: object, *, site: tuple[int, str], tensor_name: str
) -> torch.Tensor:
    """Rebuild one weight tensor from its wire form.

    Torch tensors do not survive the ``collective_rpc`` msgpack hop: the
    utility-call decoder yields ``[dtype_str, shape_list, buffer]``
    triples (or, for the Rust frontend, ``{dtype, shape, data}`` dicts
    with base64-encoded ``data``). This helper accepts:

    * ``torch.Tensor`` — passed through unchanged (in-process callers);
    * ``{"dtype": str, "shape": [int, ...], "data": bytes | memoryview
      | base64 str}`` packed dicts;
    * ``[dtype_str, shape_list, buffer]`` triples (the degraded-tensor
      wire form).

    Raises a clear :class:`ValueError` for anything else — including the
    dangling aux-buffer indices large tensors degrade to, which cannot
    be recovered worker-side.
    """
    if isinstance(value, torch.Tensor):
        return value
    where = f"tensor {tensor_name!r} at site {site!r}"
    if isinstance(value, dict):
        missing = [k for k in ("dtype", "shape", "data") if k not in value]
        if missing:
            raise ValueError(
                f"SAE weights {where}: packed dict is missing key(s) {missing}."
            )
        dtype_str, shape, data = value["dtype"], value["shape"], value["data"]
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        dtype_str, shape, data = value
    else:
        raise ValueError(
            f"SAE weights {where}: expected a torch.Tensor, a packed "
            "{dtype, shape, data} dict, or a [dtype, shape, data] triple, "
            f"got {type(value).__name__}."
        )

    if not isinstance(dtype_str, str):
        raise ValueError(f"SAE weights {where}: dtype must be a string.")
    torch_dtype = getattr(torch, dtype_str.removeprefix("torch."), None)
    if not isinstance(torch_dtype, torch.dtype):
        raise ValueError(
            f"SAE weights {where}: unknown dtype {dtype_str!r}."
        )
    if not isinstance(shape, (list, tuple)) or not all(
        isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0
        for dim in shape
    ):
        raise ValueError(
            f"SAE weights {where}: shape must be a list of non-negative "
            f"ints, got {shape!r}."
        )
    shape = tuple(shape)
    if isinstance(data, str):
        try:
            buffer = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                f"SAE weights {where}: data is not valid base64: {exc}"
            ) from exc
    elif isinstance(data, (bytes, bytearray, memoryview)):
        buffer = bytes(data)
    else:
        raise ValueError(
            f"SAE weights {where}: data must be bytes, a memoryview, or a "
            f"base64 string, got {type(data).__name__} (large tensors "
            "must be sent packed — raw tensors do not survive the "
            "collective_rpc hop)."
        )
    expected = math.prod(shape) * torch_dtype.itemsize
    if len(buffer) != expected:
        raise ValueError(
            f"SAE weights {where}: byte length {len(buffer)} does not "
            f"match expected {expected} for shape {shape} dtype "
            f"{dtype_str!r}."
        )
    if expected == 0:
        return torch.empty(shape, dtype=torch_dtype)
    # bytearray copy: frombuffer requires a writable buffer that the
    # tensor aliases; copying also detaches the tensor's storage from the
    # transient RPC frame. The uint8 view trick sidesteps dtypes without
    # a Python buffer-protocol format (bfloat16), mirroring
    # vllm/v1/serial_utils.py's tensor decode.
    raw = torch.frombuffer(bytearray(buffer), dtype=torch.uint8)
    return raw.view(torch_dtype).view(shape)


def _coerce_sae_weights_wire(
    raw: object,
) -> dict[tuple[int, str], dict[str, torch.Tensor]]:
    """Normalize an ``sae_weights`` broadcast payload into attach form.

    Applied to the payload in both SAE register branches before
    :meth:`SteeringModelRunnerMixin.attach_sae_weights` /
    :meth:`~SteeringModelRunnerMixin.attach_sae_full_recon_weights`.
    Site keys become ``(layer_idx, hook_str)`` tuples and tensor values
    become :class:`torch.Tensor` (see :func:`_coerce_sae_wire_tensor`
    for the accepted wire forms). Tensor names pass through opaquely so
    additional keys (e.g. a per-feature ``threshold``) survive.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            "SAE weights payload must be a dict keyed by (layer, hook) "
            f"site, got {type(raw).__name__}."
        )
    out: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for key, site_tensors in raw.items():
        site = _coerce_sae_wire_site_key(key)
        if not isinstance(site_tensors, dict):
            raise ValueError(
                f"SAE weights for site {site!r} must be a dict of tensor "
                f"name to tensor, got {type(site_tensors).__name__}."
            )
        out[site] = {
            str(name): _coerce_sae_wire_tensor(
                tensor, site=site, tensor_name=str(name)
            )
            for name, tensor in site_tensors.items()
        }
    return out


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

    # Worker-side mirror of registered SAE-kind modules.  The manifest
    # drives buffer attachment, request admission, and per-layer clamp
    # table population for requests carrying
    # ``SamplingParams.sae_clamp_specs``.  Disjoint from
    # ``_steering_module_registry``; the broadcast payload's ``kind``
    # field discriminates which dict an incoming module lands in.
    _sae_module_registry: dict[str, "SAEModuleManifest"]
    # Parallel structures for the SAE feature-surgery path.  The SAE
    # manager owns row allocation for ``sae_clamp_specs`` admissions;
    # ``_req_sae_phase`` / ``_req_sae_hash`` track the worker phase and
    # SAE-only row hash a request was last admitted under so
    # completion/transition can release the right row.
    # ``_sae_steerable_sites`` is populated as SAE modules register and
    # is the iteration target for both buffer attachment / detachment
    # and per-step clamp-table population.  ``None`` when SAE is disabled
    # or no SAE module has registered yet.
    _sae_clamp_manager: SAEClampManager | None = None
    _req_sae_phase: dict[str, str]
    _req_sae_hash: dict[str, int]
    # (module_name, layer_idx, hook_str) -> module reference of the
    # attached layer.  Populated by ``register_steering_modules`` for
    # SAE-kind modules and consumed by the per-step populator + the
    # weight-injection path.
    _sae_steerable_sites: dict[tuple[str, int, str], nn.Module]
    _sae_rows_scratch: np.ndarray | None = None
    _sae_index_pinned: torch.Tensor | None = None
    _sae_index_dirty: bool = False
    _req_transition_scan_candidates: set[str]
    # Phase-4 full-reconstruction state — parallel to the delta path
    # above.  Disjoint from ``_sae_module_registry`` so the kind
    # discriminator routes a registered name to exactly one path.
    _sae_fr_module_registry: dict[str, "SAEModuleManifest"]
    _sae_fr_clamp_manager: "SAEFullReconstructionManager | None" = None
    _req_sae_fr_phase: dict[str, str]
    _req_sae_fr_hash: dict[str, int]
    _sae_fr_steerable_sites: dict[tuple[str, int, str], nn.Module]
    _sae_fr_rows_scratch: np.ndarray | None = None
    _sae_fr_index_pinned: torch.Tensor | None = None
    _sae_fr_index_dirty: bool = False

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
        # SAE feature-surgery tiers (delta + full-reconstruction), parallel
        # to the canonical additive lifecycle above.
        self._sae_module_registry = {}
        self._req_sae_phase = {}
        self._req_sae_hash = {}
        self._sae_steerable_sites = {}
        self._sae_index_dirty = False
        self._req_transition_scan_candidates = set()
        self._sae_fr_module_registry = {}
        self._req_sae_fr_phase = {}
        self._req_sae_fr_hash = {}
        self._sae_fr_steerable_sites = {}
        self._sae_fr_index_dirty = False

        steering_config = getattr(self.vllm_config, "steering_config", None)
        if steering_config is None or not steerable:
            self._steering_manager = None
            self._sae_clamp_manager = None
            self._sae_fr_clamp_manager = None
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
        for mod in steerable.values():
            mod._cross_layer_monitor = cross_layer

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

        self._steering_manager = SteeringManager(
            steering_config.max_steering_configs,
            device=table_device,
            max_dynamic_steering_configs=getattr(
                steering_config, "max_dynamic_steering_configs", 0
            ),
        )
        # SAE manager shares the additive ``max_steering_configs``
        # admission budget per the design doc: a request that uses both
        # an additive config and an SAE clamp consumes one row from each
        # manager and the scheduler reserves both.
        self._sae_clamp_manager = SAEClampManager(
            steering_config.max_steering_configs,
        )
        # Phase-4: full-reconstruction manager shares the
        # ``max_steering_configs`` admission budget with the additive and
        # delta paths.  Per the design doc, a request that uses delta +
        # full-reconstruction simultaneously consumes one row from each
        # manager and the scheduler reserves all of them.
        self._sae_fr_clamp_manager = SAEFullReconstructionManager(
            steering_config.max_steering_configs,
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
            # SAE-tier per-request row scratches, parallel to the additive
            # scratches above.
            self._sae_rows_scratch = np.zeros(max_seqs, dtype=np.int64)
            self._sae_fr_rows_scratch = np.zeros(max_seqs, dtype=np.int64)
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
                self._sae_index_pinned = torch.zeros(
                    max_tokens, dtype=torch.long, pin_memory=True
                )
                self._sae_fr_index_pinned = torch.zeros(
                    max_tokens, dtype=torch.long, pin_memory=True
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
                self._sae_index_pinned = torch.zeros(max_tokens, dtype=torch.long)
                self._sae_fr_index_pinned = torch.zeros(max_tokens, dtype=torch.long)

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

    def set_steering_vectors(
        self,
        vectors: dict[str, dict[int, list[float]]] | None = None,
        prefill_vectors: dict[str, dict[int, list[float]]] | None = None,
        decode_vectors: dict[str, dict[int, list[float]]] | None = None,
        replace: bool = False,
        validate_only: bool = False,
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

        if not all_tiers:
            if replace:
                self.clear_steering_vectors()
            return (tp_rank, pp_rank, [])

        # Validate all tiers.
        valid_indices: set[int] = set()
        for _phase, tier_data in all_tiers:
            valid_indices.update(self._validate_vectors_spec(tier_data, steerable))

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

        return (tp_rank, pp_rank, sorted(valid_indices))

    def clear_steering_vectors(self) -> None:
        """Clear all tiers (base, prefill, decode) in the SteeringManager."""
        mgr = self._steering_manager
        if mgr is not None:
            mgr.clear_global_vectors()

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
        # Lazy-init the SAE-kind state for duck-typed hosts (test stubs,
        # runners that skip ``_init_steering_state``) so the kind routing
        # below never trips on a missing attribute.
        if not hasattr(self, "_sae_module_registry"):
            self._sae_module_registry = {}
        if not hasattr(self, "_sae_steerable_sites"):
            self._sae_steerable_sites = {}
        if replace:
            self._replace_steering_modules_atomically(modules)
            return
        for name, payload in modules.items():
            if not isinstance(payload, dict):
                raise SteeringVectorError(
                    f"Steering module '{name}' broadcast payload is not a dict"
                )
            # Each payload carries a ``kind`` discriminator: ``"additive"``
            # (the default for legacy payloads) routes into the additive
            # registry; ``"sae_delta"`` / ``"sae_full_reconstruction"`` route
            # into the SAE registries and attach the manifest's buffers.
            kind = payload.get("kind", "additive")
            if kind == "additive":
                if payload.get("sae_manifest") is not None:
                    raise SteeringVectorError(
                        f"Steering module '{name}': sae_manifest is not valid "
                        "for kind='additive'."
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
                # If a name is being re-registered as additive, drop any
                # stale SAE entry (and its buffers) so the registries
                # stay disjoint.
                if name in self._sae_module_registry:
                    self._detach_sae_buffers(name)
                    self._sae_module_registry.pop(name, None)
            elif kind == "sae_delta":
                if (
                    payload.get("vectors") is not None
                    or payload.get("prefill_vectors") is not None
                    or payload.get("decode_vectors") is not None
                ):
                    raise SteeringVectorError(
                        f"Steering module '{name}': additive vector fields "
                        "are not valid for kind='sae_delta'."
                    )
                manifest_payload = payload.get("sae_manifest")
                if not isinstance(manifest_payload, dict):
                    raise SteeringVectorError(
                        f"Steering module '{name}': kind='sae_delta' "
                        "requires 'sae_manifest' dict in broadcast payload."
                    )
                try:
                    manifest = sae_manifest_from_dict(manifest_payload)
                    self._validate_worker_sae_manifest(name, manifest)
                except (KeyError, TypeError, ValueError) as exc:
                    raise SteeringVectorError(
                        f"Steering module '{name}': invalid sae_manifest "
                        f"in broadcast payload: {exc}"
                    ) from exc
                # Replacement snapshot: capture both the prior SAE state
                # *and* any prior additive entry under this name so a failed
                # replacement can restore whichever one existed.
                prev_manifest = self._sae_module_registry.get(name)
                prev_additive = self._steering_module_registry.get(name)
                prev_additive_cache = self._steering_module_resolved_cache.get(name)
                prev_weights: dict[tuple[int, str], dict[str, torch.Tensor]] | None = (
                    None
                )
                if prev_manifest is not None:
                    prev_weights = self._snapshot_sae_weights(name)
                    self._detach_sae_buffers(name)
                # Re-registering a name as a different kind drops the FR
                # entry (and buffers) so the registries stay disjoint.
                fr_registry = getattr(self, "_sae_fr_module_registry", None)
                if fr_registry is not None and name in fr_registry:
                    self._detach_sae_full_recon_buffers(name)
                    fr_registry.pop(name, None)
                # Converting an additive name to SAE: release its pin so
                # theirs' pinned-row refcount invariant is preserved.
                if name in self._steering_module_pinned_rows:
                    self.release_pre_materialized_steering_module(name)
                self._sae_module_registry[name] = manifest
                self._steering_module_registry.pop(name, None)
                self._steering_module_resolved_cache.pop(name, None)
                # Atomic register-and-attach: when the payload carries
                # ``sae_weights`` the buffers are attached *and* the weights
                # copied in one indivisible step.  On failure the entry and
                # any half-attached buffers roll back; a prior module (either
                # kind) is reattached so a failed replacement does not destroy
                # the previously-working module.
                sae_weights = payload.get("sae_weights")
                try:
                    # Wire coercion inside the atomic step: broadcast
                    # payloads carry weights in the msgpack-safe wire form
                    # (Rust frontend packed dicts, or tensors degraded by
                    # the collective_rpc hop), and a malformed payload must
                    # roll back like any other attach failure.
                    self._attach_sae_buffers(name, manifest)
                    if sae_weights is not None:
                        self.attach_sae_weights(
                            name, _coerce_sae_weights_wire(sae_weights)
                        )
                except Exception:
                    self._detach_sae_buffers(name)
                    self._sae_module_registry.pop(name, None)
                    if prev_manifest is not None:
                        self._sae_module_registry[name] = prev_manifest
                        try:
                            self._attach_sae_buffers(name, prev_manifest)
                            if prev_weights is not None:
                                self.attach_sae_weights(name, prev_weights)
                        except Exception:
                            self._detach_sae_buffers(name)
                            self._sae_module_registry.pop(name, None)
                            raise
                    elif prev_additive is not None:
                        self._steering_module_registry[name] = prev_additive
                        if prev_additive_cache is not None:
                            self._steering_module_resolved_cache[name] = (
                                prev_additive_cache
                            )
                    raise
            elif kind == "sae_full_reconstruction":
                if (
                    payload.get("vectors") is not None
                    or payload.get("prefill_vectors") is not None
                    or payload.get("decode_vectors") is not None
                ):
                    raise SteeringVectorError(
                        f"Steering module '{name}': additive vector fields "
                        "are not valid for kind='sae_full_reconstruction'."
                    )
                manifest_payload = payload.get("sae_manifest")
                if not isinstance(manifest_payload, dict):
                    raise SteeringVectorError(
                        f"Steering module '{name}': kind='sae_full_"
                        "reconstruction' requires 'sae_manifest' dict in "
                        "broadcast payload."
                    )
                try:
                    manifest = sae_manifest_from_dict(manifest_payload)
                    self._validate_worker_sae_manifest(name, manifest)
                except (KeyError, TypeError, ValueError) as exc:
                    raise SteeringVectorError(
                        f"Steering module '{name}': invalid sae_manifest "
                        f"in broadcast payload: {exc}"
                    ) from exc
                fr_registry = getattr(self, "_sae_fr_module_registry", None)
                if fr_registry is None:
                    raise SteeringVectorError(
                        f"Steering module '{name}': kind='sae_full_"
                        "reconstruction' requires the worker mixin to be "
                        "initialised via _init_steering_state before "
                        "registration."
                    )
                if name in fr_registry:
                    self._detach_sae_full_recon_buffers(name)
                # Re-registering as a different kind drops the stale entry.
                if name in self._sae_module_registry:
                    self._detach_sae_buffers(name)
                    self._sae_module_registry.pop(name, None)
                if name in self._steering_module_pinned_rows:
                    self.release_pre_materialized_steering_module(name)
                fr_registry[name] = manifest
                self._steering_module_registry.pop(name, None)
                self._steering_module_resolved_cache.pop(name, None)
                sae_weights = payload.get("sae_weights")
                try:
                    self._attach_sae_full_recon_buffers(name, manifest)
                    if sae_weights is not None:
                        self.attach_sae_full_recon_weights(
                            name, _coerce_sae_weights_wire(sae_weights)
                        )
                except Exception:
                    self._detach_sae_full_recon_buffers(name)
                    fr_registry.pop(name, None)
                    raise
            else:
                raise SteeringVectorError(
                    f"Steering module '{name}': unknown kind {kind!r} in "
                    "broadcast payload."
                )
        if modules:
            logger.debug(
                "Worker received %d steering module(s) (replace=%s)",
                len(modules),
                replace,
            )

    def _replace_steering_modules_atomically(self, modules: dict[str, dict]) -> None:
        """Replace worker registries while preserving prior state on failure.

        Used by ``register_steering_modules(replace=True)`` (the API-server
        startup push).  Snapshots every registry (additive + both SAE kinds,
        including attached SAE weights), releases theirs' pre-materialized
        pins (refcount invariant: every pin taken by a previous register has
        a matching release), clears, and re-adds through the normal
        ``replace=False`` path.  On failure the prior registries and SAE
        buffers/weights are restored so a bad push cannot destroy a working
        registry.  Pins are not re-established on rollback — they are a
        performance optimization re-installed by the next pre-materialize
        call.
        """
        prev_additive = dict(self._steering_module_registry)
        prev_cache = dict(self._steering_module_resolved_cache)
        prev_sae = dict(self._sae_module_registry)
        prev_weights = {name: self._snapshot_sae_weights(name) for name in prev_sae}
        prev_fr_registry = getattr(self, "_sae_fr_module_registry", None)
        prev_fr = dict(prev_fr_registry) if prev_fr_registry is not None else {}
        prev_fr_weights = {
            name: self._snapshot_sae_full_recon_weights(name) for name in prev_fr
        }

        # Releasing pre-materialized pins before clearing the registry
        # preserves the refcount invariant.  Without this, a startup
        # ``replace=True`` push that drops a name would leak the row until
        # process exit.
        pinned_rows = getattr(self, "_steering_module_pinned_rows", None)
        if pinned_rows:
            for prior_name in list(pinned_rows.keys()):
                self.release_pre_materialized_steering_module(prior_name)

        try:
            for sae_name in list(self._sae_module_registry):
                self._detach_sae_buffers(sae_name)
            if prev_fr_registry is not None:
                for fr_name in list(prev_fr_registry):
                    self._detach_sae_full_recon_buffers(fr_name)
                prev_fr_registry.clear()
            self._steering_module_registry.clear()
            self._steering_module_resolved_cache.clear()
            self._sae_module_registry.clear()
            self.register_steering_modules(modules, replace=False)
        except Exception:
            for sae_name in list(self._sae_module_registry):
                self._detach_sae_buffers(sae_name)
            if prev_fr_registry is not None:
                for fr_name in list(prev_fr_registry):
                    self._detach_sae_full_recon_buffers(fr_name)
                prev_fr_registry.clear()
            self._steering_module_registry.clear()
            self._steering_module_resolved_cache.clear()
            self._sae_module_registry.clear()

            self._steering_module_registry.update(prev_additive)
            self._steering_module_resolved_cache.update(prev_cache)
            for name, manifest in prev_sae.items():
                self._sae_module_registry[name] = manifest
                try:
                    self._attach_sae_buffers(name, manifest)
                    weights = prev_weights.get(name)
                    if weights is not None:
                        self.attach_sae_weights(name, weights)
                except Exception:
                    self._detach_sae_buffers(name)
                    self._sae_module_registry.pop(name, None)
                    raise
            if prev_fr_registry is not None:
                for name, manifest in prev_fr.items():
                    prev_fr_registry[name] = manifest
                    try:
                        self._attach_sae_full_recon_buffers(name, manifest)
                        weights = prev_fr_weights.get(name)
                        if weights is not None:
                            self.attach_sae_full_recon_weights(name, weights)
                    except Exception:
                        self._detach_sae_full_recon_buffers(name)
                        prev_fr_registry.pop(name, None)
                        raise
            raise

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
            # A name might exist in either the additive registry or an SAE
            # registry; remove from both (detaching SAE buffers) so a
            # re-registration with a different kind lands in a clean slot.
            # ``getattr`` guards: duck-typed hosts may lack the SAE state.
            sae_registry = getattr(self, "_sae_module_registry", None)
            if sae_registry is not None and name in sae_registry:
                self._detach_sae_buffers(name)
                sae_registry.pop(name, None)
            fr_registry = getattr(self, "_sae_fr_module_registry", None)
            if fr_registry is not None and name in fr_registry:
                self._detach_sae_full_recon_buffers(name)
                fr_registry.pop(name, None)
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
        for phase, resolved in (
            ("prefill", prefill_resolved),
            ("decode", decode_resolved),
        ):
            if not resolved:
                continue
            mgr.register_config(
                named_only_hash,
                resolved,
                phase=phase,
                locally_owned_layers=locally_owned,
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
                self._steering_index_dirty = False
            # Nothing dynamic is active; revert any request still reported as
            # dynamically steered back to its admitted decode key.
            self._pending_decode_sigs = self._compute_decode_signature_deltas(
                scheduler_output, bview
            )
            # SAE tiers are independent of the additive fast path: SAE-only /
            # decode-only requests still need their per-token index rebuilt and
            # their prefill->decode transitions handled this step.
            self._update_sae_all(scheduler_output)
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
                # A request that carries only SAE state has a nonzero
                # combined request hash (the hash folds SAE clamp /
                # reconstruction identity) but no additive entry.  Gate the
                # additive lookup on this manager's row map and fall back to
                # the global no-op row (hash==0 semantics) instead of
                # tripping the unregistered-hash crash below.
                if (
                    prefill_hash != 0
                    and (prefill_hash, "prefill")
                    not in self._steering_manager.config_to_row
                ):
                    additive_hash = 0
                else:
                    additive_hash = prefill_hash
                row = self._steering_manager.get_row_for_config(
                    additive_hash, is_prefill=True
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
                    # SAE-only fallback — see the prefill branch above.
                    if (
                        decode_hash != 0
                        and (decode_hash, "decode")
                        not in self._steering_manager.config_to_row
                    ):
                        additive_hash = 0
                    else:
                        additive_hash = decode_hash
                    row = self._steering_manager.get_row_for_config(
                        additive_hash, is_prefill=False
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

        # SAE tiers (delta + full-reconstruction): build their per-token row
        # index and apply prefill->decode SAE transitions independently of the
        # additive path handled above.
        self._update_sae_all(scheduler_output)

    # -----------------------------------------------------------------------
    # Canonical per-request steering lifecycle
    #
    # These methods own ``self._steering_reqs`` — the single source of truth
    # for a request's registered steering identity + phase — and are driven
    # identically by both runners. v1 calls them from ``_update_states`` /
    # ``_update_streaming_request``; v2 from ``add_requests`` /
    # ``finish_requests``. See docs/design/v2_runner_steering_capture.md.
    # -----------------------------------------------------------------------

    @staticmethod
    def _content_hash_kwargs(sp, phase: str) -> dict:
        """Optional ``content_hash`` kwarg for ``register_config``.

        Real ``SamplingParams`` always carry the additive-only per-phase
        hashes (the physical-row identity the scheduler reserves against).
        Duck-typed test hosts may pass bare objects — and their fake
        managers may not accept the kwarg — so it is omitted when absent.
        """
        h = getattr(sp, f"{phase}_additive_steering_config_hash", None)
        return {} if h is None else {"content_hash": h}

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

        sp = sampling_params
        # Validate SAE clamp / full-reconstruction specs against the
        # registered SAE modules BEFORE any mutation (kernel feasibility
        # check).  A rejected spec must leave every prior row — including
        # the old instance's rows on a streaming re-add — untouched.
        if sp is not None:
            self._assert_sae_clamps_can_be_applied(sp)
            if self._sae_fr_clamp_manager is not None:
                self._assert_sae_full_recon_specs_can_be_applied(sp)

        self._drop_request_dynamic_override(req_id)
        # Release any SAE-tier state still held for this id (streaming re-add /
        # preemption resume re-enter here); idempotent when nothing is tracked.
        self._release_sae_for_request(req_id, 0, 0)
        self._release_sae_full_recon_for_request(req_id, 0, 0)
        self._transition_scan_candidates().discard(req_id)
        old = self._steering_reqs.pop(req_id, None)
        if old is not None:
            self._steering_release_state(old)

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

        additive_registered: tuple[int, str] | None = None
        if num_computed_tokens >= num_prompt_tokens:
            # Already past prefill — register the decode config now.
            effective_decode = self._resolve_request_steering(sp, "decode")
            if decode_hash != 0 and effective_decode:
                mgr.register_config(
                    decode_hash,
                    effective_decode,
                    phase="decode",
                    locally_owned_layers=self._locally_owned_layers,
                    **self._content_hash_kwargs(sp, "decode"),
                )
                additive_registered = (decode_hash, "decode")
            rs.phase = "decode"
        else:
            # Normal: start in prefill; the decode config is registered lazily
            # at the prefill->decode boundary in _update_steering_buffers.
            effective_prefill = self._resolve_request_steering(sp, "prefill")
            if prefill_hash != 0 and effective_prefill:
                mgr.register_config(
                    prefill_hash,
                    effective_prefill,
                    phase="prefill",
                    locally_owned_layers=self._locally_owned_layers,
                    **self._content_hash_kwargs(sp, "prefill"),
                )
                additive_registered = (prefill_hash, "prefill")
            rs.phase = "prefill"

        # SAE-tier admission runs in parallel using SAE-only row hashes; the
        # additive and SAE managers deduplicate independently. Runs even when
        # the additive hashes are 0 (SAE-only request) — the combined config
        # hash folds SAE clamp / reconstruction identity so those requests
        # still passed the guard above.  A failed SAE admission rolls the
        # just-registered additive row (and the tracking entry) back so a
        # rejected request leaves no leaked state behind.
        is_prefilling = num_computed_tokens < num_prompt_tokens
        try:
            self._register_initial_sae_clamps(
                req_id, sp, prefill_hash, decode_hash, is_prefilling
            )
            if self._sae_fr_clamp_manager is not None:
                self._register_initial_sae_full_recon(
                    req_id, sp, prefill_hash, decode_hash, is_prefilling
                )
        except Exception:
            if additive_registered is not None:
                mgr.release_config(*additive_registered)
            self._steering_reqs.pop(req_id, None)
            self._pop_req_sae_row(req_id)
            self._pop_req_sae_fr_row(req_id)
            self._transition_scan_candidates().discard(req_id)
            raise

        # A prefilling request whose SAE state is decode-only has no SAE
        # prefill row to keep the transition machinery aware of it; track it
        # as a scan candidate so the prefill->decode boundary still registers
        # its decode-phase SAE rows.
        if is_prefilling and decode_hash != 0:
            has_sae_prefill_row = (
                req_id in self._req_sae_phase_map()
                or req_id in self._req_sae_fr_phase_map()
            )
            has_sae_decode = bool(
                (
                    getattr(sp, "sae_clamp_specs", None)
                    and sp._phase_filtered_sae_specs("decode")
                )
                or (
                    getattr(sp, "sae_full_reconstruction_specs", None)
                    and sp._phase_filtered_sae_full_recon_specs("decode")
                )
            )
            if has_sae_decode and not has_sae_prefill_row:
                self._transition_scan_candidates().add(req_id)

    def _steering_finish_requests(self, req_ids: "set[str] | list[str]") -> None:
        """Release configs for finished (or preempted) requests.

        Preempted requests are released too: they re-enter through the
        admission / resume path, which re-registers a fresh prefill config.
        Reads only ``self._steering_reqs``, so the ordering relative to the
        runner popping its own request state is not load-bearing.
        """
        # SAE-tier release must not depend on the additive manager: it pops
        # the SAE tracker and releases the right hash for whichever phase the
        # request was last admitted under, even when the additive phase was
        # never tracked (e.g. an SAE-only request).  No-op when the SAE
        # managers are uninitialised.
        for req_id in req_ids:
            self._transition_scan_candidates().discard(req_id)
            self._release_sae_for_request(req_id, 0, 0)
            self._release_sae_full_recon_for_request(req_id, 0, 0)
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
            if effective_decode:
                mgr.register_config(
                    rs.decode_hash,
                    effective_decode,
                    phase="decode",
                    locally_owned_layers=self._locally_owned_layers,
                    **self._content_hash_kwargs(rs.sampling_params, "decode"),
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

    def set_sae_global_clamps(
        self,
        prefill_specs_raw: object = None,
        decode_specs_raw: object = None,
        *,
        replace: bool = False,
        validate_only: bool = False,
    ) -> tuple[int, int]:
        """Install global SAE delta clamps applied to every token in a phase.

        Mirrors :meth:`set_steering_vectors` for the SAE delta path:
        the supplied specs are validated against the worker SAE
        registry, then committed to :class:`SAEClampManager`'s global
        tier.  Tokens whose request does not carry per-request SAE
        clamps gather the phase-specific global row on the next
        forward pass (row 1 for prefill, row 2 for decode; row 0 stays
        the all-zero no-op sentinel), so the per-token dispatch picks
        globals up without per-request bookkeeping.

        Args:
            prefill_specs_raw: JSON-shape clamp specs (as accepted by
                :func:`coerce_sae_clamp_specs`) for the prefill global
                tier, or ``None`` to leave existing prefill globals
                untouched.
            decode_specs_raw: analogous for the decode global tier.
            replace: when True, clear existing globals before applying
                — used for atomic swap of the global configuration.
            validate_only: when True, coerce and validate the specs
                but skip the commit to the manager's global tier.
                Used by the router's two-phase validate-then-apply
                flow so a rank-local validation failure surfaces
                before any rank mutates state.

        Returns:
            ``(tp_rank, pp_rank)`` for router-side TP divergence checks.
        """
        tp_rank, pp_rank = _get_steering_ranks()
        mgr = self._sae_clamp_manager
        if mgr is None:
            return (tp_rank, pp_rank)
        from vllm.config.sae_steering_types import coerce_sae_clamp_specs

        prefill_specs = coerce_sae_clamp_specs(prefill_specs_raw)
        decode_specs = coerce_sae_clamp_specs(decode_specs_raw)
        # Validate every spec's module name is registered and every
        # referenced (layer, hook) site / feature is covered.  Reuses
        # the per-request admission validator so the worker fails loud
        # for the same problems regardless of whether clamps arrive
        # per-request or globally.
        for tier_specs in (prefill_specs, decode_specs):
            if tier_specs is None:
                continue
            # Build a stand-in SamplingParams-like object so we can
            # reuse :meth:`_assert_sae_clamps_can_be_applied` without
            # duplicating its logic here.
            class _Stub:
                pass

            stub = _Stub()
            stub.sae_clamp_specs = tier_specs
            self._assert_sae_clamps_can_be_applied(stub)  # type: ignore[arg-type]
        if validate_only:
            return (tp_rank, pp_rank)
        mgr.set_global_clamps(
            prefill_specs=prefill_specs,
            decode_specs=decode_specs,
            replace=replace,
        )
        return (tp_rank, pp_rank)

    def clear_sae_global_clamps(self) -> None:
        """Drop all configured global SAE delta clamps.

        Symmetric to :meth:`clear_steering_vectors` for the SAE delta
        path.  Row 0 will be re-zeroed on the next per-step populate,
        restoring no-op semantics for tokens whose request does not
        carry per-request SAE clamps.
        """
        mgr = self._sae_clamp_manager
        if mgr is not None:
            mgr.clear_global_clamps()

    def get_sae_global_clamps_status(self) -> dict:
        """Return a JSON-safe summary of the currently-configured globals.

        Returns ``{"prefill": [...], "decode": [...]}`` where each
        list entry is a JSON-encodable view of a
        :class:`SAEClampSpec`.  Empty lists mean "no globals
        configured for this phase".
        """
        mgr = self._sae_clamp_manager
        if mgr is None:
            return {"prefill": [], "decode": []}

        def _spec_to_dict(spec) -> dict:
            return {
                "module_name": spec.module_name,
                "phase": spec.phase,
                "clamps": {
                    hook_name: {
                        str(layer_idx): [
                            {
                                "feature_idx": e.feature_idx,
                                "kind": e.kind,
                                "value": e.value,
                                "only_if_active": e.only_if_active,
                            }
                            for e in entries
                        ]
                        for layer_idx, entries in layer_map.items()
                    }
                    for hook_name, layer_map in spec.clamps.items()
                },
            }

        return {
            "prefill": [_spec_to_dict(s) for s in mgr.global_prefill_specs],
            "decode": [_spec_to_dict(s) for s in mgr.global_decode_specs],
        }


    @staticmethod
    def _validate_worker_sae_manifest(
        name: str,
        manifest: SAEModuleManifest,
    ) -> None:
        """Validate SAE manifests received over worker RPC.

        API-side registration validates the same invariants before
        broadcast, but worker RPC handlers are also reachable from
        startup state sync and direct tests.  Validate here so malformed
        payloads fail before mutating registries or attaching buffers.
        """
        prefix = f"SAE module {name!r}"
        if type(manifest.d_model) is not int or manifest.d_model <= 0:
            raise ValueError(f"{prefix}: d_model must be positive.")
        if type(manifest.d_sae) is not int or manifest.d_sae <= 0:
            raise ValueError(f"{prefix}: d_sae must be positive.")
        if manifest.activation is SAEActivation.RELU:
            if manifest.activation_params:
                raise ValueError(
                    f"{prefix}: activation_params must be empty for relu."
                )
        elif manifest.activation is SAEActivation.JUMPRELU:
            if manifest.activation_params:
                raise ValueError(
                    f"{prefix}: activation_params must be empty for jumprelu "
                    "— per-feature thresholds ride the weights payload as a "
                    "'threshold' tensor."
                )
        elif manifest.activation is SAEActivation.TOPK:
            k = manifest.activation_params.get("k")
            if (
                isinstance(k, bool)
                or not isinstance(k, (int, float))
                or not math.isfinite(float(k))
                or float(k) < 1.0
                or float(k) != float(int(k))
            ):
                raise ValueError(
                    f"{prefix}: topk activation_params requires a positive "
                    "integer-valued 'k'."
                )
        else:
            raise ValueError(
                f"{prefix}: unsupported activation {manifest.activation!r}."
            )
        if not manifest.layers:
            raise ValueError(f"{prefix}: layers must not be empty.")
        seen_sites: set[tuple[int, str]] = set()
        for layer_idx, hook_str in manifest.layers:
            validate_steering_index(layer_idx, f"{prefix}: layer index")
            if hook_str not in {hp.value for hp in SteeringHookPoint}:
                raise ValueError(
                    f"{prefix}: unknown hook point {hook_str!r} in layers."
                )
            site = (layer_idx, hook_str)
            if site in seen_sites:
                raise ValueError(
                    f"{prefix}: layers must not contain duplicate "
                    f"(layer_idx, hook_point) sites; got {site!r} more than once."
                )
            seen_sites.add(site)
        if not manifest.clampable_features:
            raise ValueError(f"{prefix}: clampable_features must not be empty.")
        seen_features: set[int] = set()
        for feat in manifest.clampable_features:
            validate_steering_index(feat, f"{prefix}: clampable_features entry")
            if feat >= manifest.d_sae:
                raise ValueError(
                    f"{prefix}: clampable_features entry {feat!r} is out "
                    f"of range [0, d_sae={manifest.d_sae})."
                )
            if feat in seen_features:
                raise ValueError(
                    f"{prefix}: clampable_features must not contain "
                    f"duplicates; got feature {feat!r} more than once."
                )
            seen_features.add(feat)

    def _attach_sae_buffers(
        self,
        module_name: str,
        manifest: SAEModuleManifest,
    ) -> None:
        """Attach per-(layer, hook) SAE buffers for a registered SAE module.

        Walks ``manifest.layers``, filters to layers this rank owns,
        and attaches the SAE clamp tables / encoder / decoder buffers
        plus the shared ``sae_index`` to each owning layer module.
        Buffers default to zero; weights are populated separately via
        :meth:`attach_sae_weights` once a loader (or test fixture)
        provides them.
        """
        steerable = self._steerable_layers_cache or {}
        vllm_config = getattr(self, "vllm_config", None)
        steering_config = (
            getattr(vllm_config, "steering_config", None)
            if vllm_config is not None
            else None
        )
        if steering_config is None or not steerable:
            return
        max_sae_configs = int(steering_config.max_steering_configs)
        # Discover compute dtype + max_tokens from the additive table
        # buffers — the additive path is the existing source of truth
        # for the engine's allocated CPU/GPU resources.
        any_layer = next(iter(steerable.values()))
        ref_dtype: torch.dtype | None = None
        for attr in HOOK_POINT_TABLE_ATTR.values():
            if hasattr(any_layer, attr):
                ref_buffer = getattr(any_layer, attr)
                ref_dtype = ref_buffer.dtype
                table_device = ref_buffer.device
                break
        if ref_dtype is None:
            ref_dtype = getattr(self.vllm_config.model_config, "dtype", torch.float32)
            table_device = torch.device("cpu")
        scheduler_config = getattr(self.vllm_config, "scheduler_config", None)
        max_tokens = (
            int(scheduler_config.max_num_batched_tokens)
            if scheduler_config is not None
            else 0
        )
        n_clamp = len(manifest.clampable_features)
        attached_layers: list[nn.Module] = []
        for layer_idx, hook_str in manifest.layers:
            if layer_idx not in self._locally_owned_layers:
                continue
            layer = steerable.get(layer_idx)
            if layer is None:
                continue
            try:
                hook_point = SteeringHookPoint(hook_str)
            except ValueError as exc:
                raise SteeringVectorError(
                    f"SAE module {module_name!r} declares unsupported hook "
                    f"point {hook_str!r}."
                ) from exc
            register_sae_buffers(
                layer,
                hook_point=hook_point,
                module_name=module_name,
                activation=manifest.activation,
                activation_params=manifest.activation_params,
                n_clamp=n_clamp,
                hidden_size=manifest.d_model,
                max_sae_configs=max_sae_configs,
                dtype=ref_dtype,
                device=table_device,
            )
            self._sae_steerable_sites[(module_name, layer_idx, hook_str)] = layer
            register_sae_index_buffer(
                layer, max_tokens=max_tokens, device=table_device
            )
            attached_layers.append(layer)
        if attached_layers:
            # Share ``sae_index`` across every SAE-covered layer on this
            # worker — not just the layers attached for *this* module.
            # ``_update_sae_buffers`` writes through a single tensor
            # picked from ``_sae_steerable_sites``; layers from a
            # previously-registered SAE module whose ``sae_index`` was
            # not rebound would gather row 0 and silently no-op.
            unique_layers: dict[int, nn.Module] = {}
            for layer in self._sae_steerable_sites.values():
                unique_layers.setdefault(id(layer), layer)
            share_sae_index_across_layers(list(unique_layers.values()))
            self._warmup_sae_kernel_for_module(
                manifest=manifest,
                attached_layers=attached_layers,
                ref_dtype=ref_dtype,
            )

    def _warmup_sae_kernel_for_module(
        self,
        *,
        manifest: SAEModuleManifest,
        attached_layers: list[nn.Module],
        ref_dtype: torch.dtype,
    ) -> None:
        """JIT-warm the SAE Triton kernel for the activation this module uses.

        Mirrors the additive ``warmup_apply_steering_kernel`` call in
        :meth:`_init_steering_state`: pays the Triton first-call JIT
        cost outside any captured forward pass so subsequent CUDA-graph
        capture won't trigger a compile mid-capture.

        Each (activation_code, BLOCK_H, BLOCK_C) triple specialises to
        its own JIT artifact, so we warm once per attached SAE module.
        Multiple modules with identical specialisation share the
        cached binary on subsequent attach calls.
        """
        if not attached_layers:
            return
        device = attached_layers[0].sae_index.device  # type: ignore[union-attr]
        if device.type != "cuda":
            return
        from vllm.model_executor.layers.sae_steering import (
            _ACTIVATION_TO_CODE,
            _activation_to_scalar,
        )
        from vllm.model_executor.layers.sae_steering_kernel import (
            warmup_apply_sae_delta_kernel,
        )

        compute_dtype = getattr(self.vllm_config.model_config, "dtype", ref_dtype)
        warmup_apply_sae_delta_kernel(
            hidden_size=manifest.d_model,
            n_clamp=len(manifest.clampable_features),
            table_dtype=ref_dtype,
            compute_dtype=compute_dtype,
            device=device,
            activation_code=_ACTIVATION_TO_CODE[manifest.activation],
            activation_param=_activation_to_scalar(
                manifest.activation, manifest.activation_params
            ),
        )

    def _detach_sae_buffers(self, module_name: str) -> None:
        """Detach all per-(layer, hook) buffers attached for ``module_name``.

        Only ``module_name``'s buffer slots are removed; sibling
        modules sharing a (layer, hook) site keep their slots (and
        their stable attr names — slot ids are never reused).
        """
        keys = [k for k in self._sae_steerable_sites if k[0] == module_name]
        for key in keys:
            _, _layer_idx, hook_str = key
            layer = self._sae_steerable_sites.pop(key)
            try:
                hook_point = SteeringHookPoint(hook_str)
            except ValueError:
                continue
            unregister_sae_buffers(
                layer, hook_point=hook_point, module_name=module_name
            )

    def _snapshot_sae_weights(
        self, module_name: str
    ) -> dict[tuple[int, str], dict[str, torch.Tensor]]:
        """Clone the currently-attached SAE weights for ``module_name``.

        Used by the rollback path: the encoder/decoder/encoder_bias
        buffers are about to be destroyed by ``_detach_sae_buffers``
        for a replacement registration, and we need a copy so the
        prior weights can be restored if the new attach fails.
        """
        snapshot: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
        for (name, layer_idx, hook_str), site in self._sae_steerable_sites.items():
            if name != module_name:
                continue
            try:
                hook_point = SteeringHookPoint(hook_str)
            except ValueError:
                continue
            state = get_sae_slot_state(site, hook_point, module_name)
            if state is None:
                continue
            snapshot[(layer_idx, hook_str)] = {
                "encoder_weight": state.encoder_weight.detach().clone(),
                "encoder_bias": state.encoder_bias.detach().clone(),
                "decoder_weight": state.decoder_weight.detach().clone(),
                "threshold": state.threshold.detach().clone(),
            }
        return snapshot

    def attach_sae_weights(
        self,
        module_name: str,
        weights: dict[tuple[int, str], dict[str, torch.Tensor]],
    ) -> None:
        """Inject encoder / decoder weight tensors into the SAE buffers.

        ``weights`` maps ``(layer_idx, hook_str)`` to a dict with keys
        ``"encoder_weight"``, ``"encoder_bias"``, and
        ``"decoder_weight"``, plus a per-feature ``(n_clamp,)``
        ``"threshold"`` tensor — required when the registered
        manifest's activation is JumpReLU, optional otherwise.  Each
        tensor is copied into the corresponding zero-initialised
        buffer in place; shape and dtype must match what
        ``_attach_sae_buffers`` allocated.

        This is the injection point used by tests, runtime registration,
        and startup/full-registry broadcasts after the on-disk loader has
        materialised tensors per (layer, hook) site.
        """
        if module_name not in self._sae_module_registry:
            raise SteeringVectorError(
                f"attach_sae_weights: SAE module {module_name!r} is not registered."
            )
        manifest = self._sae_module_registry[module_name]
        threshold_required = manifest.activation is SAEActivation.JUMPRELU
        expected_sites = {
            (layer_idx, hook_str)
            for name, layer_idx, hook_str in self._sae_steerable_sites
            if name == module_name
        }
        provided_owned_sites = set(weights) & expected_sites
        missing_sites = sorted(expected_sites - provided_owned_sites)
        if missing_sites:
            raise SteeringVectorError(
                f"attach_sae_weights({module_name!r}): missing weights for "
                f"owned SAE site(s): {missing_sites}."
            )

        copy_plan: list[tuple[torch.Tensor, torch.Tensor]] = []
        for (layer_idx, hook_str), tensors in weights.items():
            site = self._sae_steerable_sites.get((module_name, layer_idx, hook_str))
            if site is None:
                # Layer not owned by this rank or not in the manifest.
                continue
            try:
                hook_point = SteeringHookPoint(hook_str)
            except ValueError as exc:
                raise SteeringVectorError(
                    f"attach_sae_weights: unsupported hook point {hook_str!r}."
                ) from exc
            state = get_sae_slot_state(site, hook_point, module_name)
            if state is None:
                # Site tracked but slot missing — buffers were torn down
                # between attach and this call; treat as unowned.
                continue
            for tensor_key, buf, required in (
                ("encoder_weight", state.encoder_weight, True),
                ("encoder_bias", state.encoder_bias, True),
                ("decoder_weight", state.decoder_weight, True),
                ("threshold", state.threshold, threshold_required),
            ):
                if tensor_key not in tensors:
                    if not required:
                        continue
                    raise SteeringVectorError(
                        f"attach_sae_weights({module_name!r}): missing "
                        f"{tensor_key!r} for site (layer={layer_idx}, "
                        f"hook={hook_str!r})."
                )
                raw = tensors[tensor_key]
                if not isinstance(raw, torch.Tensor):
                    raise SteeringVectorError(
                        f"attach_sae_weights({module_name!r}): {tensor_key} "
                        f"must be a torch.Tensor, got {type(raw).__name__} at "
                        f"site (layer={layer_idx}, hook={hook_str!r})."
                    )
                if not torch.is_floating_point(raw):
                    raise SteeringVectorError(
                        f"attach_sae_weights({module_name!r}): {tensor_key} "
                        f"must have a floating dtype, got {raw.dtype} at "
                        f"site (layer={layer_idx}, hook={hook_str!r})."
                    )
                if not bool(torch.isfinite(raw).all().item()):
                    raise SteeringVectorError(
                        f"attach_sae_weights({module_name!r}): {tensor_key} "
                        "must contain only finite values at "
                        f"site (layer={layer_idx}, hook={hook_str!r})."
                    )
                src = raw.to(dtype=buf.dtype, device=buf.device)
                if src.shape != buf.shape:
                    raise SteeringVectorError(
                        f"attach_sae_weights({module_name!r}): {tensor_key} "
                        f"shape {tuple(src.shape)} does not match buffer "
                        f"shape {tuple(buf.shape)} at site "
                        f"(layer={layer_idx}, hook={hook_str!r})."
                    )
                copy_plan.append((buf, src))
        for buf, src in copy_plan:
            buf.copy_(src)

    def _assert_sae_clamps_can_be_applied(self, sp: SamplingParams) -> None:
        """Validate ``sp.sae_clamp_specs`` against the worker SAE registry.

        Mirrors the strict-capacity contract of the additive path: a
        request that names a missing SAE module, references an
        unsupported ``(layer, hook)`` site, or clamps a feature outside
        the module's ``clampable_features`` set fails admission
        immediately rather than silently running with stale buffers.

        Admission row allocation (registering the spec with the SAE
        manager) is done separately by
        :meth:`_register_initial_sae_clamps`; this method only does
        the validation that the kernel can apply the spec.
        """
        specs = getattr(sp, "sae_clamp_specs", None)
        if not specs:
            return
        for spec in specs:
            manifest = self._sae_module_registry.get(spec.module_name)
            if manifest is None:
                available = sorted(self._sae_module_registry.keys())
                raise SteeringVectorError(
                    f"SAE clamp spec references unknown module "
                    f"{spec.module_name!r}.  Available SAE modules on "
                    f"this worker: {available or 'none'}.  "
                    "Register the module via "
                    "POST /v1/steering/modules/register with "
                    "kind='sae_delta' before submitting a clamp spec."
                )
            covered = set(manifest.layers)
            clampable = set(manifest.clampable_features)
            for hook_name, layer_map in spec.clamps.items():
                for layer_idx, entries in layer_map.items():
                    if (layer_idx, hook_name) not in covered:
                        raise SteeringVectorError(
                            f"SAE clamp spec for module "
                            f"{spec.module_name!r} targets site "
                            f"(layer={layer_idx}, hook={hook_name!r}) "
                            "which is not declared in the module's "
                            "manifest.layers."
                        )
                    for entry in entries:
                        if entry.feature_idx not in clampable:
                            raise SteeringVectorError(
                                f"SAE clamp spec for module "
                                f"{spec.module_name!r} clamps "
                                f"feature_idx={entry.feature_idx}, "
                                "which is not in the module's "
                                "clampable_features set."
                            )

    def _req_sae_phase_map(self) -> dict[str, str]:
        """Return request -> SAE phase state, creating it for duck-typed hosts."""
        if not hasattr(self, "_req_sae_phase"):
            self._req_sae_phase = {}
        return self._req_sae_phase

    def _req_sae_hash_map(self) -> dict[str, int]:
        """Return request -> SAE row hash state, creating it for old tests."""
        if not hasattr(self, "_req_sae_hash"):
            self._req_sae_hash = {}
        return self._req_sae_hash

    def _set_req_sae_row(self, req_id: str, sae_hash: int, phase: str) -> None:
        self._req_sae_phase[req_id] = phase
        self._req_sae_hash_map()[req_id] = sae_hash

    def _pop_req_sae_row(self, req_id: str) -> tuple[int, str] | None:
        phase = self._req_sae_phase.pop(req_id, None)
        sae_hash = self._req_sae_hash_map().pop(req_id, 0)
        if phase is None:
            return None
        return sae_hash, phase

    def _get_req_sae_row(self, req_id: str) -> tuple[int, str] | None:
        phase = self._req_sae_phase.get(req_id)
        sae_hash = self._req_sae_hash_map().get(req_id, 0)
        if phase is None:
            return None
        return sae_hash, phase

    def _req_sae_fr_phase_map(self) -> dict[str, str]:
        """Return request -> SAE full-recon phase state."""
        if not hasattr(self, "_req_sae_fr_phase"):
            self._req_sae_fr_phase = {}
        return self._req_sae_fr_phase

    def _req_sae_fr_hash_map(self) -> dict[str, int]:
        """Return request -> SAE full-recon row hash state."""
        if not hasattr(self, "_req_sae_fr_hash"):
            self._req_sae_fr_hash = {}
        return self._req_sae_fr_hash

    def _set_req_sae_fr_row(self, req_id: str, recon_hash: int, phase: str) -> None:
        self._req_sae_fr_phase_map()[req_id] = phase
        self._req_sae_fr_hash_map()[req_id] = recon_hash

    def _pop_req_sae_fr_row(self, req_id: str) -> tuple[int, str] | None:
        phase = self._req_sae_fr_phase_map().pop(req_id, None)
        recon_hash = self._req_sae_fr_hash_map().pop(req_id, 0)
        if phase is None:
            return None
        return recon_hash, phase

    def _get_req_sae_fr_row(self, req_id: str) -> tuple[int, str] | None:
        phase = self._req_sae_fr_phase_map().get(req_id)
        recon_hash = self._req_sae_fr_hash_map().get(req_id, 0)
        if phase is None:
            return None
        return recon_hash, phase

    # -----------------------------------------------------------------------
    # Phase-4: SAE full-reconstruction buffer / weight / admission lifecycle
    # -----------------------------------------------------------------------

    def _attach_sae_full_recon_buffers(
        self,
        module_name: str,
        manifest: SAEModuleManifest,
    ) -> None:
        """Attach per-(layer, hook) full-reconstruction buffers.

        Walks ``manifest.layers``, filters to layers this rank owns,
        and attaches the full encoder / decoder weight buffers + per-
        row clamp tables + the shared ``sae_recon_index`` to each
        owning layer module.  Buffers default to zero; weights are
        populated separately via :meth:`attach_sae_full_recon_weights`.
        """
        steerable = self._steerable_layers_cache or {}
        vllm_config = getattr(self, "vllm_config", None)
        steering_config = (
            getattr(vllm_config, "steering_config", None)
            if vllm_config is not None
            else None
        )
        if steering_config is None or not steerable:
            return
        max_recon_configs = int(steering_config.max_steering_configs)
        any_layer = next(iter(steerable.values()))
        ref_dtype: torch.dtype | None = None
        table_device: torch.device | None = None
        for attr in HOOK_POINT_TABLE_ATTR.values():
            if hasattr(any_layer, attr):
                ref_buffer = getattr(any_layer, attr)
                ref_dtype = ref_buffer.dtype
                table_device = ref_buffer.device
                break
        if ref_dtype is None:
            ref_dtype = getattr(self.vllm_config.model_config, "dtype", torch.float32)
            table_device = torch.device("cpu")
        scheduler_config = getattr(self.vllm_config, "scheduler_config", None)
        max_tokens = (
            int(scheduler_config.max_num_batched_tokens)
            if scheduler_config is not None
            else 0
        )
        n_clamp = len(manifest.clampable_features)
        clampable_features = torch.tensor(
            list(manifest.clampable_features), dtype=torch.int64, device=table_device
        )
        attached_layers: list[nn.Module] = []
        for layer_idx, hook_str in manifest.layers:
            if layer_idx not in self._locally_owned_layers:
                continue
            layer = steerable.get(layer_idx)
            if layer is None:
                continue
            try:
                hook_point = SteeringHookPoint(hook_str)
            except ValueError as exc:
                raise SteeringVectorError(
                    f"SAE full-reconstruction module {module_name!r} declares "
                    f"unsupported hook point {hook_str!r}."
                ) from exc
            register_sae_full_recon_buffers(
                layer,
                hook_point=hook_point,
                module_name=module_name,
                activation=manifest.activation,
                activation_params=manifest.activation_params,
                d_sae=manifest.d_sae,
                n_clamp=n_clamp,
                hidden_size=manifest.d_model,
                max_recon_configs=max_recon_configs,
                clampable_features=clampable_features,
                dtype=ref_dtype,
                device=table_device,
            )
            register_sae_recon_index_buffer(
                layer, max_tokens=max_tokens, device=table_device
            )
            self._sae_fr_steerable_sites[(module_name, layer_idx, hook_str)] = layer
            attached_layers.append(layer)
        if attached_layers:
            unique_layers: dict[int, nn.Module] = {}
            for layer in self._sae_fr_steerable_sites.values():
                unique_layers.setdefault(id(layer), layer)
            share_sae_recon_index_across_layers(list(unique_layers.values()))
            self._warmup_sae_full_recon_for_module(
                manifest=manifest,
                attached_layers=attached_layers,
                ref_dtype=ref_dtype,
            )

    def _warmup_sae_full_recon_for_module(
        self,
        *,
        manifest: SAEModuleManifest,
        attached_layers: list[nn.Module],
        ref_dtype: torch.dtype,
    ) -> None:
        """Pre-warm the full-reconstruction CUDA path for ``manifest``."""
        if not attached_layers:
            return
        device = attached_layers[0].sae_recon_index.device  # type: ignore[union-attr]
        if device.type != "cuda":
            return
        from vllm.model_executor.layers.sae_full_reconstruction_kernel import (
            warmup_apply_sae_full_recon_kernel,
        )
        from vllm.model_executor.layers.sae_steering import (
            _ACTIVATION_TO_CODE,
            _activation_to_scalar,
        )

        compute_dtype = getattr(self.vllm_config.model_config, "dtype", ref_dtype)
        warmup_apply_sae_full_recon_kernel(
            hidden_size=manifest.d_model,
            d_sae=manifest.d_sae,
            n_clamp=len(manifest.clampable_features),
            table_dtype=ref_dtype,
            compute_dtype=compute_dtype,
            device=device,
            activation_code=_ACTIVATION_TO_CODE[manifest.activation],
            activation_param=_activation_to_scalar(
                manifest.activation, manifest.activation_params
            ),
        )

    def _detach_sae_full_recon_buffers(self, module_name: str) -> None:
        """Detach per-(layer, hook) full-reconstruction buffers for the module."""
        sites = getattr(self, "_sae_fr_steerable_sites", None)
        if sites is None:
            return
        keys = [k for k in sites if k[0] == module_name]
        for key in keys:
            _, _layer_idx, hook_str = key
            layer = sites.pop(key)
            try:
                hook_point = SteeringHookPoint(hook_str)
            except ValueError:
                continue
            unregister_sae_full_recon_buffers(layer, hook_point=hook_point)

    def _snapshot_sae_full_recon_weights(
        self, module_name: str
    ) -> dict[tuple[int, str], dict[str, torch.Tensor]]:
        """Clone the currently-attached full-recon weights for rollback."""
        snapshot: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
        sites = getattr(self, "_sae_fr_steerable_sites", None)
        if sites is None:
            return snapshot
        for (name, layer_idx, hook_str), site in sites.items():
            if name != module_name:
                continue
            try:
                hook_point = SteeringHookPoint(hook_str)
            except ValueError:
                continue
            tensors: dict[str, torch.Tensor] = {}
            for tensor_key, attr_table in (
                ("encoder_weight", HOOK_POINT_FR_ENCODER_WEIGHT_ATTR),
                ("encoder_bias", HOOK_POINT_FR_ENCODER_BIAS_ATTR),
                ("decoder_weight", HOOK_POINT_FR_DECODER_WEIGHT_ATTR),
                ("decoder_bias", HOOK_POINT_FR_DECODER_BIAS_ATTR),
                ("threshold", HOOK_POINT_FR_THRESHOLD_ATTR),
            ):
                buf = getattr(site, attr_table[hook_point], None)
                if buf is None:
                    continue
                tensors[tensor_key] = buf.detach().clone()
            if tensors:
                snapshot[(layer_idx, hook_str)] = tensors
        return snapshot

    def attach_sae_full_recon_weights(
        self,
        module_name: str,
        weights: dict[tuple[int, str], dict[str, torch.Tensor]],
    ) -> None:
        """Inject encoder / decoder weight + bias tensors into full-recon buffers.

        ``weights`` maps ``(layer_idx, hook_str)`` to a dict with keys
        ``"encoder_weight"``, ``"encoder_bias"``,
        ``"decoder_weight"``, and ``"decoder_bias"``, plus a
        per-feature ``(d_sae,)`` ``"threshold"`` tensor — required
        when the registered manifest's activation is JumpReLU,
        optional otherwise.  Each tensor is copied into the
        corresponding zero-initialised buffer in place; shape and
        dtype must match what
        :meth:`_attach_sae_full_recon_buffers` allocated.
        """
        if module_name not in self._sae_fr_module_registry:
            raise SteeringVectorError(
                f"attach_sae_full_recon_weights: SAE full-reconstruction "
                f"module {module_name!r} is not registered."
            )
        manifest = self._sae_fr_module_registry[module_name]
        threshold_required = manifest.activation is SAEActivation.JUMPRELU
        for (layer_idx, hook_str), tensors in weights.items():
            site = self._sae_fr_steerable_sites.get((module_name, layer_idx, hook_str))
            if site is None:
                continue
            try:
                hook_point = SteeringHookPoint(hook_str)
            except ValueError as exc:
                raise SteeringVectorError(
                    f"attach_sae_full_recon_weights: unsupported hook point "
                    f"{hook_str!r}."
                ) from exc
            for tensor_key, attr_table, required in (
                ("encoder_weight", HOOK_POINT_FR_ENCODER_WEIGHT_ATTR, True),
                ("encoder_bias", HOOK_POINT_FR_ENCODER_BIAS_ATTR, True),
                ("decoder_weight", HOOK_POINT_FR_DECODER_WEIGHT_ATTR, True),
                ("decoder_bias", HOOK_POINT_FR_DECODER_BIAS_ATTR, True),
                ("threshold", HOOK_POINT_FR_THRESHOLD_ATTR, threshold_required),
            ):
                if tensor_key not in tensors:
                    if not required:
                        continue
                    raise SteeringVectorError(
                        f"attach_sae_full_recon_weights({module_name!r}): "
                        f"missing {tensor_key!r} for site (layer="
                        f"{layer_idx}, hook={hook_str!r})."
                    )
                buf = getattr(site, attr_table[hook_point])
                src = tensors[tensor_key].to(dtype=buf.dtype, device=buf.device)
                if src.shape != buf.shape:
                    raise SteeringVectorError(
                        f"attach_sae_full_recon_weights({module_name!r}): "
                        f"{tensor_key} shape {tuple(src.shape)} does not "
                        f"match buffer shape {tuple(buf.shape)} at site "
                        f"(layer={layer_idx}, hook={hook_str!r})."
                    )
                buf.copy_(src)

    def _assert_sae_full_recon_specs_can_be_applied(self, sp: SamplingParams) -> None:
        """Validate ``sp.sae_full_reconstruction_specs`` against the registry."""
        specs = getattr(sp, "sae_full_reconstruction_specs", None)
        if not specs:
            return
        for spec in specs:
            manifest = self._sae_fr_module_registry.get(spec.module_name)
            if manifest is None:
                available = sorted(self._sae_fr_module_registry.keys())
                raise SteeringVectorError(
                    f"SAE full-reconstruction spec references unknown module "
                    f"{spec.module_name!r}.  Available full-reconstruction "
                    f"modules on this worker: {available or 'none'}.  "
                    "Register the module via "
                    "POST /v1/steering/modules/register with "
                    "kind='sae_full_reconstruction' before submitting a spec."
                )
            covered = set(manifest.layers)
            clampable = set(manifest.clampable_features)
            for hook_name, layer_map in spec.clamps.items():
                for layer_idx, entries in layer_map.items():
                    if (layer_idx, hook_name) not in covered:
                        raise SteeringVectorError(
                            f"SAE full-reconstruction spec for module "
                            f"{spec.module_name!r} targets site "
                            f"(layer={layer_idx}, hook={hook_name!r}) "
                            "which is not declared in the module's "
                            "manifest.layers."
                        )
                    for entry in entries:
                        if entry.feature_idx not in clampable:
                            raise SteeringVectorError(
                                f"SAE full-reconstruction spec for module "
                                f"{spec.module_name!r} clamps feature_idx="
                                f"{entry.feature_idx}, which is not in the "
                                "module's clampable_features set."
                            )

    def _register_initial_sae_full_recon(
        self,
        req_id: str,
        sp: SamplingParams,
        prefill_hash: int,
        decode_hash: int,
        is_prefilling: bool,
    ) -> None:
        """Admit ``sp.sae_full_reconstruction_specs`` against the manager.

        Uses the scheduler's combined hashes only as phase-presence gates.
        Rows are keyed by the full-reconstruction-only phase hashes so
        additive and decode-only state cannot accidentally route tokens to
        a reconstruction row in the wrong phase.
        """
        mgr = self._sae_fr_clamp_manager
        specs = getattr(sp, "sae_full_reconstruction_specs", None)
        if mgr is None or not specs:
            return
        prefill_specs = sp._phase_filtered_sae_full_recon_specs("prefill")
        decode_specs = sp._phase_filtered_sae_full_recon_specs("decode")
        prefill_recon_hash = sp.prefill_sae_full_recon_config_hash
        decode_recon_hash = sp.decode_sae_full_recon_config_hash
        if is_prefilling:
            if prefill_hash != 0 and prefill_recon_hash != 0 and prefill_specs:
                mgr.register_recon_spec(
                    prefill_recon_hash, prefill_specs, "prefill"
                )
                self._set_req_sae_fr_row(req_id, prefill_recon_hash, "prefill")
        else:
            if decode_hash != 0 and decode_recon_hash != 0 and decode_specs:
                mgr.register_recon_spec(decode_recon_hash, decode_specs, "decode")
                self._set_req_sae_fr_row(req_id, decode_recon_hash, "decode")

    def _release_sae_full_recon_for_request(
        self,
        req_id: str,
        prefill_hash: int,
        decode_hash: int,
    ) -> None:
        """Release any rows allocated for ``req_id``.  Idempotent."""
        mgr = self._sae_fr_clamp_manager
        if mgr is None:
            return
        row = self._pop_req_sae_fr_row(req_id)
        if row is None:
            return
        recon_hash, phase = row
        if recon_hash != 0:
            mgr.release_recon_spec(recon_hash, phase)
    def _register_initial_sae_clamps(
        self,
        req_id: str,
        sp: SamplingParams,
        prefill_hash: int,
        decode_hash: int,
        is_prefilling: bool,
    ) -> None:
        """Admit ``sp.sae_clamp_specs`` against the SAE manager.

        Mirrors the additive ``_steering_register_request``
        admission flow: the scheduler reserves capacity at admission
        time, so registration is expected to succeed.  When the
        request is being admitted directly into decode (full prefix-
        cache hit), admits the decode-active SAE-only hash; otherwise
        admits the prefill-active SAE-only hash and the prefill→decode
        transition path registers the decode row.
        """
        mgr = self._sae_clamp_manager
        if mgr is None or not getattr(sp, "sae_clamp_specs", None):
            return
        prefill_specs = sp._phase_filtered_sae_specs("prefill")
        decode_specs = sp._phase_filtered_sae_specs("decode")
        prefill_sae_hash = sp.prefill_sae_clamp_config_hash
        decode_sae_hash = sp.decode_sae_clamp_config_hash
        if is_prefilling:
            if prefill_hash != 0 and prefill_sae_hash != 0 and prefill_specs:
                mgr.register_clamp_spec(prefill_sae_hash, prefill_specs, "prefill")
                self._set_req_sae_row(req_id, prefill_sae_hash, "prefill")
        else:
            if decode_hash != 0 and decode_sae_hash != 0 and decode_specs:
                mgr.register_clamp_spec(decode_sae_hash, decode_specs, "decode")
                self._set_req_sae_row(req_id, decode_sae_hash, "decode")

    def _release_sae_for_request(
        self,
        req_id: str,
        prefill_hash: int,
        decode_hash: int,
    ) -> None:
        """Release the active SAE row for a finished or refreshed request."""
        mgr = self._sae_clamp_manager
        if mgr is None:
            return
        row = self._pop_req_sae_row(req_id)
        if row is None:
            return
        sae_hash, phase = row
        if sae_hash != 0:
            mgr.release_clamp_spec(sae_hash, phase)


    def _transition_scan_candidates(self) -> set[str]:
        candidates = self.__dict__.get("_req_transition_scan_candidates")
        if not isinstance(candidates, set):
            candidates = set()
            self.__dict__["_req_transition_scan_candidates"] = candidates
        return candidates


    def _apply_batched_sae_transitions(
        self,
        transitions: list[tuple[str, int, int, SamplingParams | None]],
    ) -> None:
        """Apply prefill->decode SAE transitions for one scheduler step."""
        if not transitions:
            return
        sae_mgr = self._sae_clamp_manager
        if sae_mgr is None:
            return

        for req_id, _prefill_hash, _decode_hash, _sp in transitions:
            old_row = self._get_req_sae_row(req_id)
            if old_row is not None and old_row[1] == "prefill" and old_row[0] != 0:
                sae_mgr.release_clamp_spec(old_row[0], "prefill")
                self._pop_req_sae_row(req_id)

        for req_id, _prefill_hash, decode_hash, sp in transitions:
            decode_specs = (
                sp._phase_filtered_sae_specs("decode")
                if sp is not None and getattr(sp, "sae_clamp_specs", None)
                else None
            )
            decode_sae_hash = (
                sp.decode_sae_clamp_config_hash
                if sp is not None and getattr(sp, "sae_clamp_specs", None)
                else 0
            )
            if decode_hash != 0 and decode_sae_hash != 0 and decode_specs:
                sae_mgr.register_clamp_spec(decode_sae_hash, decode_specs, "decode")
                self._set_req_sae_row(req_id, decode_sae_hash, "decode")
            else:
                self._pop_req_sae_row(req_id)
            self._transition_scan_candidates().discard(req_id)

    def _handle_sae_transitions_for_scheduled_prefill_completions(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """Run prefill->decode transitions when additive work is inactive.

        ``_update_steering_buffers`` has a fast path for the common case
        where additive steering has no active rows or globals.  SAE-only
        and decode-only-additive requests can still need a transition on
        the final prefill token, so this mirrors the transition detection
        from the main loop without rebuilding the additive index.
        """
        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids
        transitions: list[tuple[str, int, int, SamplingParams | None]] = []
        for i in range(num_reqs):
            req_id = req_ids[i]
            n_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if n_tokens == 0:
                continue
            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                continue
            num_computed = int(self.input_batch.num_computed_tokens_cpu[req_index])
            num_prompt = int(self.input_batch.num_prompt_tokens[req_index])
            if num_computed >= num_prompt or num_computed + n_tokens < num_prompt:
                continue
            # Combined per-phase steering hashes from the canonical store
            # (theirs moved these off the InputBatch columns); used here only
            # as phase-presence gates for the SAE transition scan.
            _rs = self._steering_reqs.get(req_id)
            prefill_hash = _rs.prefill_hash if _rs is not None else 0
            decode_hash = _rs.decode_hash if _rs is not None else 0
            if (
                prefill_hash == 0
                and decode_hash == 0
                and req_id not in self._req_sae_phase
                and req_id not in self._req_sae_fr_phase_map()
                and req_id not in self._transition_scan_candidates()
            ):
                continue
            req_state = self.requests.get(req_id)
            sp = req_state.sampling_params if req_state is not None else None
            transitions.append((req_id, prefill_hash, decode_hash, sp))
        # Additive prefill->decode transitions are handled inline by
        # ``_steering_transition`` in the canonical ``_update_steering_buffers``
        # loop; here we only drive the SAE tiers.
        self._apply_batched_sae_transitions(transitions)
        self._apply_batched_sae_full_recon_transitions(transitions)

    def _may_need_prefill_completion_transition_scan(self) -> bool:
        """Return whether no-active shortcut must scan for phase transitions."""
        # The additive no-active shortcut is only reached when the additive
        # manager has no active rows or globals.  Decode-only additive rows
        # would keep the additive manager active, so the remaining transition
        # work here is for SAE-only requests whose prefill row must stay live
        # through the final prefill token, plus decode-only requests that have
        # no prefill row to keep the additive manager active.
        return (
            any(phase == "prefill" for phase in self._req_sae_phase.values())
            or any(
                phase == "prefill" for phase in self._req_sae_fr_phase_map().values()
            )
            or bool(self._transition_scan_candidates())
        )

    def _handle_sae_transition(
        self,
        req_id: str,
        prefill_hash: int,
        decode_hash: int,
        sp: SamplingParams | None,
    ) -> None:
        """Release prefill SAE row and register decode SAE row."""
        sae_mgr = self._sae_clamp_manager
        if sae_mgr is None or sp is None or not getattr(sp, "sae_clamp_specs", None):
            return
        old_row = self._get_req_sae_row(req_id)
        if old_row is not None and old_row[1] == "prefill" and old_row[0] != 0:
            sae_mgr.release_clamp_spec(old_row[0], "prefill")
            self._pop_req_sae_row(req_id)
        decode_specs = sp._phase_filtered_sae_specs("decode")
        decode_sae_hash = sp.decode_sae_clamp_config_hash
        if decode_hash != 0 and decode_sae_hash != 0 and decode_specs:
            sae_mgr.register_clamp_spec(decode_sae_hash, decode_specs, "decode")
            self._set_req_sae_row(req_id, decode_sae_hash, "decode")
        else:
            # Decode hash is 0 → no SAE state in decode (e.g.
            # spec.phase == "prefill").  Drop the tracking entry.
            self._pop_req_sae_row(req_id)

    def _apply_batched_sae_full_recon_transitions(
        self,
        transitions: list[tuple[str, int, int, SamplingParams | None]],
    ) -> None:
        """Apply prefill->decode full-reconstruction transitions for one step.

        Releases all completed prefill references first, then registers
        any decode rows so requests sharing the same prefill row can free
        it before the first decode allocation is attempted.  Uses the
        combined per-request hash for row addressing (same key the
        scheduler reserves capacity against).
        """
        if not transitions:
            return
        fr_mgr = self._sae_fr_clamp_manager
        if fr_mgr is None:
            return

        for req_id, _prefill_hash, _decode_hash, _sp in transitions:
            old_row = self._get_req_sae_fr_row(req_id)
            if old_row is not None and old_row[1] == "prefill" and old_row[0] != 0:
                fr_mgr.release_recon_spec(old_row[0], "prefill")
                self._pop_req_sae_fr_row(req_id)

        for req_id, _prefill_hash, decode_hash, sp in transitions:
            decode_specs = (
                sp._phase_filtered_sae_full_recon_specs("decode")
                if sp is not None
                and getattr(sp, "sae_full_reconstruction_specs", None)
                else None
            )
            decode_recon_hash = (
                sp.decode_sae_full_recon_config_hash
                if sp is not None
                and getattr(sp, "sae_full_reconstruction_specs", None)
                else 0
            )
            if decode_hash != 0 and decode_recon_hash != 0 and decode_specs:
                fr_mgr.register_recon_spec(
                    decode_recon_hash, decode_specs, "decode"
                )
                self._set_req_sae_fr_row(req_id, decode_recon_hash, "decode")
            else:
                self._pop_req_sae_fr_row(req_id)
            self._transition_scan_candidates().discard(req_id)


    def _update_sae_buffers(self, scheduler_output: "SchedulerOutput") -> None:
        """Populate SAE per-layer clamp tables and the shared ``sae_index``.

        Mirrors the additive ``_update_steering_buffers`` core loop:
        per-request lookup against the SAE manager, accumulate into a
        scratch row-per-request array, then a single non-blocking
        H2D copy into the shared ``sae_index`` tensor.

        Short-circuits when the SAE manager is uninitialized or no
        SAE module has registered (so no layer holds SAE buffers).
        """
        sae_mgr = self._sae_clamp_manager
        if sae_mgr is None or not self._sae_steerable_sites:
            return

        def clear_sae_any_active_flags() -> None:
            for (module_name, _layer_idx, hook_str), site in (
                self._sae_steerable_sites.items()
            ):
                try:
                    hook_point = SteeringHookPoint(hook_str)
                except ValueError:
                    continue
                # Per-slot flag: each module sharing the site has its
                # own ``any_active`` buffer.
                state = get_sae_slot_state(site, hook_point, module_name)
                if state is not None:
                    state.any_active.zero_()

        # Fast no-active path: if every SAE row has been released *and*
        # no global SAE clamps are configured, the only required work
        # is clearing a previously nonzero shared index.  Stale nonzero
        # table rows are harmless once no token points at them.  Clear
        # each site's ``any_active`` flag on the active->inactive
        # transition so layer hooks skip the SAE op entirely until a
        # later row reuse repopulates tables.  Globals are checked
        # alongside per-request rows because the global tier still
        # writes content into row 0 that the dispatch shim has to see,
        # even when no per-request rows are live.
        if not sae_mgr.config_to_row and not sae_mgr.has_global_clamps():
            if sae_mgr._tables_dirty:
                clear_sae_any_active_flags()
                sae_mgr.mark_tables_clean()
            if self._sae_index_dirty:
                any_layer = next(iter(self._sae_steerable_sites.values()))
                if hasattr(any_layer, "sae_index"):
                    sae_index = cast(torch.Tensor, any_layer.sae_index)
                    sae_index.zero_()
                self._sae_index_dirty = False
            return

        # 1. Flush manager rows into per-(layer, hook) clamp tables.
        # Only when manager state has changed since the last populate.
        if sae_mgr._tables_dirty:
            for (
                module_name,
                layer_idx,
                hook_str,
            ), site in self._sae_steerable_sites.items():
                manifest = self._sae_module_registry.get(module_name)
                if manifest is None:
                    # Module was unregistered between attach and this
                    # populate — buffers will be torn down on the next
                    # step; skip this site.
                    continue
                try:
                    hook_point = SteeringHookPoint(hook_str)
                except ValueError:
                    continue
                # Each row in the manager is registered under its own
                # worker phase (``row_phase``); the populator writes
                # every active row under its own phase in a single
                # pass.  No per-call ``worker_phase`` argument needed
                # in production — that argument is reserved for tests
                # that want to assert phase-gating behaviour.
                populate_sae_clamp_table(
                    manager=sae_mgr,
                    module=site,
                    hook_point=hook_point,
                    module_name=module_name,
                    clampable_features=manifest.clampable_features,
                    layer_idx=layer_idx,
                )
            sae_mgr.mark_tables_clean()

        # 2. Build sae_index per token from the manager's row map.
        any_layer = next(iter(self._sae_steerable_sites.values()))
        if (
            not sae_buffers_attached(any_layer, SteeringHookPoint.PRE_ATTN)
            and not sae_buffers_attached(any_layer, SteeringHookPoint.POST_ATTN)
            and not sae_buffers_attached(any_layer, SteeringHookPoint.POST_BLOCK)
        ):
            return
        sae_index = cast(torch.Tensor, any_layer.sae_index)

        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids
        rows_scratch = self._sae_rows_scratch
        n_tokens_scratch = self._steering_n_tokens_scratch
        index_pinned = self._sae_index_pinned
        if rows_scratch is None or n_tokens_scratch is None or index_pinned is None:
            return
        if rows_scratch.shape[0] < num_reqs or n_tokens_scratch.shape[0] < num_reqs:
            rows_scratch = np.zeros(num_reqs, dtype=np.int64)
            n_tokens_scratch = np.zeros(num_reqs, dtype=np.int64)
            self._sae_rows_scratch = rows_scratch
            self._steering_n_tokens_scratch = n_tokens_scratch

        active_count = 0
        for i in range(num_reqs):
            req_id = req_ids[i]
            n_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if n_tokens == 0:
                continue

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                rows_scratch[active_count] = 0
                n_tokens_scratch[active_count] = n_tokens
                active_count += 1
                continue

            num_computed = int(self.input_batch.num_computed_tokens_cpu[req_index])
            num_prompt = int(self.input_batch.num_prompt_tokens[req_index])
            is_prefilling = num_computed < num_prompt
            tracked_sae_row = self._get_req_sae_row(req_id)
            # The combined per-phase steering hash lives on the canonical
            # ``_steering_reqs`` store (theirs moved it off the InputBatch
            # columns); it is used here only as a phase-presence gate.
            _rs = self._steering_reqs.get(req_id)

            if is_prefilling:
                combined_hash = _rs.prefill_hash if _rs is not None else 0
                sae_hash = tracked_sae_row[0] if tracked_sae_row is not None else 0
                if (
                    combined_hash == 0
                    or sae_hash == 0
                    or tracked_sae_row is None
                    or tracked_sae_row[1] != "prefill"
                    or (sae_hash, "prefill") not in sae_mgr.config_to_row
                ):
                    sae_hash = 0
                row = sae_mgr.get_row_for_config(sae_hash, is_prefill=True)
            else:
                combined_hash = _rs.decode_hash if _rs is not None else 0
                sae_hash = tracked_sae_row[0] if tracked_sae_row is not None else 0
                if (
                    combined_hash == 0
                    or sae_hash == 0
                    or tracked_sae_row is None
                    or tracked_sae_row[1] != "decode"
                    or (sae_hash, "decode") not in sae_mgr.config_to_row
                ):
                    sae_hash = 0
                row = sae_mgr.get_row_for_config(sae_hash, is_prefill=False)
            rows_scratch[active_count] = row
            n_tokens_scratch[active_count] = n_tokens
            active_count += 1

        if active_count > 0:
            expanded = np.repeat(
                rows_scratch[:active_count],
                n_tokens_scratch[:active_count],
            )
            n_expanded = int(expanded.shape[0])
            n_expanded = min(n_expanded, index_pinned.shape[0], sae_index.shape[0])
            index_pinned[:n_expanded].copy_(torch.from_numpy(expanded[:n_expanded]))
            sae_index[:n_expanded].copy_(index_pinned[:n_expanded], non_blocking=True)
        else:
            n_expanded = 0

        if n_expanded < sae_index.shape[0]:
            sae_index[n_expanded:].zero_()
        self._sae_index_dirty = True

    def _update_sae_full_recon_buffers(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Populate full-reconstruction clamp tables + ``sae_recon_index``.

        Mirrors :meth:`_update_sae_buffers` for the full-reconstruction
        path: per-request lookup against the
        :class:`SAEFullReconstructionManager`, accumulate into the
        full-recon scratch row-per-request array, then a single
        non-blocking H2D copy into the shared ``sae_recon_index``
        tensor.  Tokens whose request has no active full-
        reconstruction row are routed to row 0 (the no-reconstruction
        sentinel) so the dispatch shim's per-site active-row table
        short-circuits them.
        """
        fr_mgr = self._sae_fr_clamp_manager
        if fr_mgr is None or not self._sae_fr_steerable_sites:
            return

        if fr_mgr._tables_dirty:
            for (
                module_name,
                layer_idx,
                hook_str,
            ), site in self._sae_fr_steerable_sites.items():
                manifest = self._sae_fr_module_registry.get(module_name)
                if manifest is None:
                    continue
                try:
                    hook_point = SteeringHookPoint(hook_str)
                except ValueError:
                    continue
                populate_sae_full_recon_clamp_table(
                    manager=fr_mgr,
                    module=site,
                    hook_point=hook_point,
                    module_name=module_name,
                    clampable_features=manifest.clampable_features,
                    layer_idx=layer_idx,
                )
            fr_mgr.mark_tables_clean()

        any_layer = next(iter(self._sae_fr_steerable_sites.values()))
        if (
            not sae_full_recon_buffers_attached(any_layer, SteeringHookPoint.PRE_ATTN)
            and not sae_full_recon_buffers_attached(
                any_layer, SteeringHookPoint.POST_ATTN
            )
            and not sae_full_recon_buffers_attached(
                any_layer, SteeringHookPoint.POST_BLOCK
            )
        ):
            return
        recon_index = cast(torch.Tensor, any_layer.sae_recon_index)

        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids
        rows_scratch = self._sae_fr_rows_scratch
        n_tokens_scratch = self._steering_n_tokens_scratch
        index_pinned = self._sae_fr_index_pinned
        if rows_scratch is None or n_tokens_scratch is None or index_pinned is None:
            return
        if rows_scratch.shape[0] < num_reqs:
            rows_scratch = np.zeros(num_reqs, dtype=np.int64)
            self._sae_fr_rows_scratch = rows_scratch

        active_count = 0
        for i in range(num_reqs):
            req_id = req_ids[i]
            n_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if n_tokens == 0:
                continue

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                rows_scratch[active_count] = 0
                n_tokens_scratch[active_count] = n_tokens
                active_count += 1
                continue

            num_computed = int(self.input_batch.num_computed_tokens_cpu[req_index])
            num_prompt = int(self.input_batch.num_prompt_tokens[req_index])
            is_prefilling = num_computed < num_prompt
            tracked_recon_row = self._get_req_sae_fr_row(req_id)
            # Combined per-phase steering hash from the canonical store
            # (phase-presence gate only).
            _rs = self._steering_reqs.get(req_id)

            if is_prefilling:
                combined_hash = _rs.prefill_hash if _rs is not None else 0
                recon_hash = (
                    tracked_recon_row[0] if tracked_recon_row is not None else 0
                )
                if (
                    combined_hash == 0
                    or recon_hash == 0
                    or tracked_recon_row is None
                    or tracked_recon_row[1] != "prefill"
                    or (recon_hash, "prefill") not in fr_mgr.config_to_row
                ):
                    recon_hash = 0
                row = fr_mgr.get_row_for_config(recon_hash, is_prefill=True)
            else:
                combined_hash = _rs.decode_hash if _rs is not None else 0
                recon_hash = (
                    tracked_recon_row[0] if tracked_recon_row is not None else 0
                )
                if (
                    combined_hash == 0
                    or recon_hash == 0
                    or tracked_recon_row is None
                    or tracked_recon_row[1] != "decode"
                    or (recon_hash, "decode") not in fr_mgr.config_to_row
                ):
                    recon_hash = 0
                row = fr_mgr.get_row_for_config(recon_hash, is_prefill=False)
            rows_scratch[active_count] = row
            n_tokens_scratch[active_count] = n_tokens
            active_count += 1

        if active_count > 0:
            expanded = np.repeat(
                rows_scratch[:active_count],
                n_tokens_scratch[:active_count],
            )
            n_expanded = int(expanded.shape[0])
            n_expanded = min(n_expanded, index_pinned.shape[0], recon_index.shape[0])
            index_pinned[:n_expanded].copy_(torch.from_numpy(expanded[:n_expanded]))
            recon_index[:n_expanded].copy_(index_pinned[:n_expanded], non_blocking=True)
        else:
            n_expanded = 0

        if n_expanded < recon_index.shape[0]:
            recon_index[n_expanded:].zero_()
        self._sae_fr_index_dirty = True


    def _update_sae_all(self, scheduler_output: "SchedulerOutput") -> None:
        """Drive the SAE clamp + full-reconstruction tiers for one step.

        Independent of the additive index/transition path: builds the SAE
        (and FR) per-token row index, then applies any prefill->decode SAE
        transitions for requests completing prefill this step.  Called from
        both exits of ``_update_steering_buffers`` so SAE-only and
        decode-only requests are handled even when the additive fast path
        short-circuits.
        """
        if self._sae_clamp_manager is None and self._sae_fr_clamp_manager is None:
            return
        self._update_sae_buffers(scheduler_output)
        if self._sae_fr_clamp_manager is not None:
            self._update_sae_full_recon_buffers(scheduler_output)
        # The transition scan is only needed while some request holds an SAE
        # prefill row (or is a decode-only-SAE scan candidate); skipping it
        # otherwise keeps the steady state free of per-request scans.
        if self._may_need_prefill_completion_transition_scan():
            self._handle_sae_transitions_for_scheduled_prefill_completions(
                scheduler_output
            )
