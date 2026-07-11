# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define activation steering functionality mixin for model runners.
"""

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn

from vllm.config.sae_steering_types import SAEActivation
from vllm.config.steering_types import (
    ResolvedSteeringVectorSpec,
    SteeringVectorSpec,
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
    populate_sae_full_recon_clamp_table,
    register_sae_full_recon_buffers,
    register_sae_recon_index_buffer,
    sae_full_recon_buffers_attached,
    share_sae_recon_index_across_layers,
    unregister_sae_full_recon_buffers,
)
from vllm.model_executor.layers.sae_steering import (
    HOOK_POINT_SAE_ANY_ACTIVE_ATTR,
    HOOK_POINT_SAE_DECODER_WEIGHT_ATTR,
    HOOK_POINT_SAE_ENCODER_BIAS_ATTR,
    HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR,
    populate_sae_clamp_table,
    register_sae_buffers,
    register_sae_index_buffer,
    sae_buffers_attached,
    share_sae_index_across_layers,
    unregister_sae_buffers,
)
from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.sae_clamp_manager import SAEClampManager
from vllm.v1.worker.sae_full_reconstruction_manager import (
    SAEFullReconstructionManager,
)
from vllm.v1.worker.steering_manager import SteeringManager


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
    # All steering state is initialised eagerly by ``_init_steering_state``
    # at the end of ``GPUModelRunner.load_model``.  The class-level
    # defaults below cover the pre-init window (e.g. unit tests that
    # construct the mixin without going through load_model) so plain
    # attribute access is safe without ``hasattr`` guards.
    _steering_manager: SteeringManager | None = None
    _steerable_layers_cache: dict[int, nn.Module] | None = None
    _req_steering_phase: dict[str, str]
    _steering_index_dirty: bool
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
    # resolve work when a request references a name with no inline
    # overrides; ``scale!=1.0`` is handled by multiplying the cached arrays.
    # Populated alongside ``_steering_module_registry`` and invalidated
    # together.
    _steering_module_resolved_cache: dict[
        str,
        tuple[
            ResolvedSteeringVectorSpec | None,
            ResolvedSteeringVectorSpec | None,
        ],
    ]
    # Worker-side mirror of registered SAE-kind modules.  The manifest
    # drives buffer attachment, request admission, and per-layer clamp
    # table population for requests carrying
    # ``SamplingParams.sae_clamp_specs``.  Disjoint from
    # ``_steering_module_registry``; the broadcast payload's ``kind``
    # field discriminates which dict an incoming module lands in.
    _sae_module_registry: dict[str, "SAEModuleManifest"]
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
    # Parallel structures for the SAE feature-surgery path.  The SAE
    # manager owns row allocation for ``sae_clamp_specs`` admissions;
    # ``_req_sae_phase`` / ``_req_sae_hash`` track the worker phase and
    # SAE-only row hash a request was last admitted under so
    # completion/transition can release the right row.
    # ``_sae_steerable_sites`` is populated as SAE modules
    # register and is the iteration target for both buffer attachment
    # / detachment and per-step clamp-table population.  ``None``
    # when SAE is disabled or no SAE module has registered yet.
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
        self._locally_owned_layers = frozenset(steerable.keys())
        self._req_steering_phase = {}
        self._steering_index_dirty = False
        self._steering_module_registry = {}
        self._steering_module_resolved_cache = {}
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
        )
        # SAE manager shares the additive ``max_steering_configs``
        # admission budget per the design doc: a request that uses both
        # an additive config and an SAE clamp consumes one row from
        # each manager and the scheduler reserves both.
        self._sae_clamp_manager = SAEClampManager(
            steering_config.max_steering_configs,
        )
        # Phase-4: full-reconstruction manager shares the
        # ``max_steering_configs`` admission budget with the additive
        # and delta paths.  Per the design doc, a request that uses
        # delta + full-reconstruction simultaneously consumes one row
        # from each manager and the scheduler reserves all of them.
        self._sae_fr_clamp_manager = SAEFullReconstructionManager(
            steering_config.max_steering_configs,
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
            self._sae_rows_scratch = np.zeros(max_seqs, dtype=np.int64)
            self._sae_fr_rows_scratch = np.zeros(max_seqs, dtype=np.int64)
            try:
                self._steering_index_pinned = torch.zeros(
                    max_tokens, dtype=torch.long, pin_memory=True
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
                self._sae_index_pinned = torch.zeros(max_tokens, dtype=torch.long)
                self._sae_fr_index_pinned = torch.zeros(max_tokens, dtype=torch.long)

        # Warm the fused-apply Triton kernel so first-call JIT cost
        # happens before any captured forward pass. Without this, the
        # initial CUDA-graph capture step could trigger a Triton compile
        # and fail capture.
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
            warmup_apply_steering_kernel(
                hidden_size=hidden_size,
                table_rows=steering_config.max_steering_configs + 3,
                table_dtype=table_dtype,
                compute_dtype=compute_dtype,
                device=table_device,
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
                    if type(layer_key) is int:
                        layer_idx = layer_key
                    elif isinstance(layer_key, str):
                        try:
                            layer_idx = int(layer_key)
                        except ValueError as exc:
                            raise SteeringVectorError(
                                f"Steering module payload hook {hook!r} has "
                                f"invalid layer key {layer_key!r}; expected "
                                "an integer."
                            ) from exc
                    else:
                        raise SteeringVectorError(
                            f"Steering module payload hook {hook!r} has "
                            f"invalid layer key {layer_key!r}; expected "
                            "an integer."
                        )
                    layer_idx = validate_steering_index(
                        layer_idx,
                        f"Steering module payload hook {hook!r} layer key",
                    )
                    if layer_idx in converted:
                        raise SteeringVectorError(
                            f"Steering module payload hook {hook!r} contains "
                            f"duplicate layer key {layer_idx!r} after "
                            "integer normalization."
                        )
                    converted[layer_idx] = entry
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

        *modules* maps module name to a JSON-safe dict produced by
        :meth:`SteeringModuleRegistry.dump_for_broadcast`.  Each payload
        carries a ``kind`` discriminator: ``"additive"`` (the default
        for legacy payloads without the field) routes the module into
        the additive ``_steering_module_registry``;  ``"sae_delta"``
        routes it into ``_sae_module_registry`` and attaches the
        manifest's SAE buffers.

        When *replace* is ``True`` both worker-side registries are
        cleared before the new entries are stored — used during
        API-server startup to push the initial registry state.

        Mirrors the strict-capacity contract of the rest of the
        steering runtime: requests referencing a name that has not yet
        been broadcast raise loudly in
        :meth:`_resolve_request_steering` rather than silently falling
        back to inline-only behaviour.
        """
        if not hasattr(self, "_steering_module_registry"):
            self._steering_module_registry = {}
        if not hasattr(self, "_steering_module_resolved_cache"):
            self._steering_module_resolved_cache = {}
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
            kind = payload.get("kind", "additive")
            if kind == "additive":
                if payload.get("sae_manifest") is not None:
                    raise SteeringVectorError(
                        f"Steering module '{name}': sae_manifest is not valid "
                        "for kind='additive'."
                    )
                # Legacy paths and additive registrations both flow here.
                specs = self._module_payload_to_specs(payload)
                self._steering_module_registry[name] = specs
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
                    self._validate_worker_sae_site_ownership(name, manifest)
                except (KeyError, TypeError, ValueError) as exc:
                    raise SteeringVectorError(
                        f"Steering module '{name}': invalid sae_manifest "
                        f"in broadcast payload: {exc}"
                    ) from exc
                # Replacement snapshot: capture both the prior SAE
                # state *and* any prior additive entry under this name
                # so a failed replacement can restore whichever one
                # existed.  Without the additive snapshot, an
                # additive-to-SAE replacement that fails at attach
                # would leave the worker with no entry at all —
                # diverging from the server registry's rollback that
                # restored the additive module.
                prev_manifest = self._sae_module_registry.get(name)
                prev_additive = self._steering_module_registry.get(name)
                prev_additive_cache = self._steering_module_resolved_cache.get(name)
                prev_weights: dict[tuple[int, str], dict[str, torch.Tensor]] | None = (
                    None
                )
                if prev_manifest is not None:
                    prev_weights = self._snapshot_sae_weights(name)
                    self._detach_sae_buffers(name)
                # If a name is being re-registered with a different
                # kind, drop the full-reconstruction entry (and its
                # buffers) so the registries stay disjoint.  Use
                # ``getattr`` so harnesses that don't initialise the
                # full-reconstruction state still work.
                fr_registry = getattr(self, "_sae_fr_module_registry", None)
                if fr_registry is not None and name in fr_registry:
                    self._detach_sae_full_recon_buffers(name)
                    fr_registry.pop(name, None)
                self._sae_module_registry[name] = manifest
                self._steering_module_registry.pop(name, None)
                self._steering_module_resolved_cache.pop(name, None)
                # Atomic register-and-attach: when the payload carries
                # ``sae_weights``, the buffers are attached *and* the
                # weights copied in one indivisible step on the worker.
                # If either half raises, the registry entry and any
                # half-attached buffers are rolled back; when a prior
                # module existed at this name (either kind), its state
                # is reattached so a failed replacement does not
                # destroy the previously-working module.
                sae_weights = payload.get("sae_weights")
                try:
                    self._attach_sae_buffers(name, manifest)
                    if sae_weights is not None:
                        self.attach_sae_weights(name, sae_weights)
                except Exception:
                    self._detach_sae_buffers(name)
                    self._sae_module_registry.pop(name, None)
                    if prev_manifest is not None:
                        # Best-effort restore.  Reattach uses the same
                        # code path as a fresh registration; the
                        # cloned tensors are the source of truth for
                        # the rolled-back weights.
                        self._sae_module_registry[name] = prev_manifest
                        try:
                            self._attach_sae_buffers(name, prev_manifest)
                            if prev_weights is not None:
                                self.attach_sae_weights(name, prev_weights)
                        except Exception:
                            # Restoration itself failed (e.g. OOM at
                            # reattach).  Drop the entry rather than
                            # leaving a registered name with no
                            # buffers.
                            self._detach_sae_buffers(name)
                            self._sae_module_registry.pop(name, None)
                            raise
                    elif prev_additive is not None:
                        # additive-to-SAE replacement: the additive
                        # entry was popped before attach so the worker
                        # state wouldn't hold two kinds under one name.
                        # Restore it now so the failed replacement
                        # doesn't silently delete a working module.
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
                # Re-registering as a different kind drops the stale
                # entry so the registries stay disjoint.
                if name in self._sae_module_registry:
                    self._detach_sae_buffers(name)
                    self._sae_module_registry.pop(name, None)
                fr_registry[name] = manifest
                self._steering_module_registry.pop(name, None)
                self._steering_module_resolved_cache.pop(name, None)
                sae_weights = payload.get("sae_weights")
                try:
                    self._attach_sae_full_recon_buffers(name, manifest)
                    if sae_weights is not None:
                        self.attach_sae_full_recon_weights(name, sae_weights)
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
        """Replace worker registries while preserving prior state on failure."""
        prev_additive = dict(self._steering_module_registry)
        prev_cache = dict(self._steering_module_resolved_cache)
        prev_sae = dict(self._sae_module_registry)
        prev_weights = {name: self._snapshot_sae_weights(name) for name in prev_sae}
        prev_fr_registry = getattr(self, "_sae_fr_module_registry", None)
        prev_fr = dict(prev_fr_registry) if prev_fr_registry is not None else {}
        prev_fr_weights = {
            name: self._snapshot_sae_full_recon_weights(name) for name in prev_fr
        }

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
            threshold = manifest.activation_params.get("threshold")
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold))
            ):
                raise ValueError(
                    f"{prefix}: jumprelu activation_params requires a finite "
                    "'threshold'."
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

    def _validate_worker_sae_site_ownership(
        self,
        name: str,
        manifest: SAEModuleManifest,
    ) -> None:
        """Reject SAE site overlap before attaching per-layer buffers."""
        new_sites = set(manifest.layers)
        for other_name, other_manifest in self._sae_module_registry.items():
            if other_name == name:
                continue
            overlap = new_sites & set(other_manifest.layers)
            if overlap:
                raise ValueError(
                    f"SAE module {name!r}: layers overlap existing SAE "
                    f"module {other_name!r} at site(s) {sorted(overlap)!r}.  "
                    "At most one SAE module may own a (layer_idx, "
                    "hook_point) site on a worker."
                )

    def unregister_steering_modules(self, names: list[str]) -> None:
        """Drop the listed names from the worker-side registries.

        A name might exist in either the additive registry or the SAE
        registry; remove from both so re-registration with a different
        kind always lands in a clean slot.  Detaches per-layer SAE
        buffers for any SAE-kind modules being removed.
        """
        for name in names:
            self._steering_module_registry.pop(name, None)
            self._steering_module_resolved_cache.pop(name, None)
            if name in self._sae_module_registry:
                self._detach_sae_buffers(name)
                self._sae_module_registry.pop(name, None)
            fr_registry = getattr(self, "_sae_fr_module_registry", None)
            if fr_registry is not None and name in fr_registry:
                self._detach_sae_full_recon_buffers(name)
                fr_registry.pop(name, None)
        if names:
            logger.debug(
                "Worker unregistered %d steering module(s)",
                len(names),
            )

    # -----------------------------------------------------------------------
    # SAE buffer attachment lifecycle
    # -----------------------------------------------------------------------

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
        """Detach all per-(layer, hook) buffers attached for ``module_name``."""
        keys = [k for k in self._sae_steerable_sites if k[0] == module_name]
        for key in keys:
            _, _layer_idx, hook_str = key
            layer = self._sae_steerable_sites.pop(key)
            try:
                hook_point = SteeringHookPoint(hook_str)
            except ValueError:
                continue
            unregister_sae_buffers(layer, hook_point=hook_point)

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
            tensors: dict[str, torch.Tensor] = {}
            for tensor_key, attr_table in (
                ("encoder_weight", HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR),
                ("encoder_bias", HOOK_POINT_SAE_ENCODER_BIAS_ATTR),
                ("decoder_weight", HOOK_POINT_SAE_DECODER_WEIGHT_ATTR),
            ):
                buf = getattr(site, attr_table[hook_point], None)
                if buf is None:
                    continue
                tensors[tensor_key] = buf.detach().clone()
            if tensors:
                snapshot[(layer_idx, hook_str)] = tensors
        return snapshot

    def attach_sae_weights(
        self,
        module_name: str,
        weights: dict[tuple[int, str], dict[str, torch.Tensor]],
    ) -> None:
        """Inject encoder / decoder weight tensors into the SAE buffers.

        ``weights`` maps ``(layer_idx, hook_str)`` to a dict with keys
        ``"encoder_weight"``, ``"encoder_bias"``, and
        ``"decoder_weight"``.  Each tensor is copied into the
        corresponding zero-initialised buffer in place; shape and
        dtype must match what ``_attach_sae_buffers`` allocated.

        This is the injection point used by tests, runtime registration,
        and startup/full-registry broadcasts after the on-disk loader has
        materialised tensors per (layer, hook) site.
        """
        if module_name not in self._sae_module_registry:
            raise SteeringVectorError(
                f"attach_sae_weights: SAE module {module_name!r} is not registered."
            )
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
            for tensor_key, attr_table in (
                ("encoder_weight", HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR),
                ("encoder_bias", HOOK_POINT_SAE_ENCODER_BIAS_ATTR),
                ("decoder_weight", HOOK_POINT_SAE_DECODER_WEIGHT_ATTR),
            ):
                if tensor_key not in tensors:
                    raise SteeringVectorError(
                        f"attach_sae_weights({module_name!r}): missing "
                        f"{tensor_key!r} for site (layer={layer_idx}, "
                        f"hook={hook_str!r})."
                )
                buf = getattr(site, attr_table[hook_point])
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
        specs = sp.sae_clamp_specs
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
        ``"decoder_weight"``, and ``"decoder_bias"``.  Each tensor is
        copied into the corresponding zero-initialised buffer in
        place; shape and dtype must match what
        :meth:`_attach_sae_full_recon_buffers` allocated.
        """
        if module_name not in self._sae_fr_module_registry:
            raise SteeringVectorError(
                f"attach_sae_full_recon_weights: SAE full-reconstruction "
                f"module {module_name!r} is not registered."
            )
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
            for tensor_key, attr_table in (
                ("encoder_weight", HOOK_POINT_FR_ENCODER_WEIGHT_ATTR),
                ("encoder_bias", HOOK_POINT_FR_ENCODER_BIAS_ATTR),
                ("decoder_weight", HOOK_POINT_FR_DECODER_WEIGHT_ATTR),
                ("decoder_bias", HOOK_POINT_FR_DECODER_BIAS_ATTR),
            ):
                if tensor_key not in tensors:
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

        Mirrors the additive ``_register_initial_steering_config``
        admission flow: the scheduler reserves capacity at admission
        time, so registration is expected to succeed.  When the
        request is being admitted directly into decode (full prefix-
        cache hit), admits the decode-active SAE-only hash; otherwise
        admits the prefill-active SAE-only hash and the prefill→decode
        transition path registers the decode row.
        """
        mgr = self._sae_clamp_manager
        if mgr is None or not sp.sae_clamp_specs:
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

    def _resolve_request_steering(
        self,
        sp: SamplingParams,
        phase: str,
    ) -> ResolvedSteeringVectorSpec | None:
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
        # per-request merge + resolve numpy work inherited from
        # ``feat/steering``.
        if sp.steering_vectors is None and inline_phase_spec is None:
            cached = getattr(self, "_steering_module_resolved_cache", {}).get(name)
            if cached is not None:
                resolved = cached[0] if phase == "prefill" else cached[1]
                if resolved is None:
                    return None
                if scale == 1.0:
                    return resolved
                return {
                    hook: {layer: arr * scale for layer, arr in layer_dict.items()}
                    for hook, layer_dict in resolved.items()
                }

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

    def _transition_scan_candidates(self) -> set[str]:
        candidates = self.__dict__.get("_req_transition_scan_candidates")
        if not isinstance(candidates, set):
            candidates = set()
            self.__dict__["_req_transition_scan_candidates"] = candidates
        return candidates

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
        """
        if self._steering_manager is None or not self._steerable_layers_cache:
            return

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
                steering_index = cast(torch.Tensor, any_layer.steering_index)
                steering_index.zero_()
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
                self._steering_index_dirty = False
            # SAE buffers can remain attached after the last SAE row is
            # released.  Keep their shared index in sync even when the
            # additive path has no active work; otherwise stale nonzero
            # row IDs from the previous steered step would keep applying
            # released clamps.  Run SAE transitions after building the
            # current step's SAE index so the final prefill token still
            # sees the prefill row before it is released.
            self._update_sae_buffers(scheduler_output)
            if getattr(self, "_sae_fr_clamp_manager", None) is not None:
                self._update_sae_full_recon_buffers(scheduler_output)
            if self._may_need_prefill_completion_transition_scan():
                self._handle_sae_transitions_for_scheduled_prefill_completions(
                    scheduler_output
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

        # 2. Build steering index
        # Get the shared steering_index buffer (all layers share one tensor)
        any_layer = next(iter(self._steerable_layers_cache.values()))
        steering_index = cast(torch.Tensor, any_layer.steering_index)

        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids

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
        assert rows_scratch is not None
        assert n_tokens_scratch is not None
        assert index_pinned is not None

        # Grow per-request scratches if the batch ever exceeds the
        # initial sizing.  This is defensive — ``max_num_seqs`` should
        # bound ``num_reqs`` — but cheap to handle.
        if rows_scratch.shape[0] < num_reqs or n_tokens_scratch.shape[0] < num_reqs:
            rows_scratch = np.zeros(num_reqs, dtype=np.int64)
            n_tokens_scratch = np.zeros(num_reqs, dtype=np.int64)
            self._steering_rows_scratch = rows_scratch
            self._steering_n_tokens_scratch = n_tokens_scratch

        active_count = 0
        steering_transitions: list[tuple[str, int, int, SamplingParams | None]] = []
        sae_transitions: list[tuple[str, int, int, SamplingParams | None]] = []
        for i in range(num_reqs):
            req_id = req_ids[i]
            n_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if n_tokens == 0:
                continue

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # Request not in batch yet (shouldn't happen but guard).
                # Row 0 is the no-steering sentinel.
                rows_scratch[active_count] = 0
                n_tokens_scratch[active_count] = n_tokens
                active_count += 1
                continue

            # Determine phase from num_computed vs num_prompt
            num_computed = int(self.input_batch.num_computed_tokens_cpu[req_index])
            num_prompt = int(self.input_batch.num_prompt_tokens[req_index])
            is_prefilling = num_computed < num_prompt

            if is_prefilling:
                # Prefill: use prefill steering hash
                prefill_hash = int(
                    self.input_batch.request_prefill_steering_hash[req_index]
                )
                # A request that carries only SAE clamps has a nonzero
                # combined request hash but no additive entry.  Gate the
                # additive lookup on this manager's row map and fall back
                # to the global no-op row (hash==0 semantics).
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

                # Check if this request will transition to decode after
                # this step's tokens are processed. Must happen in this
                # same pass, after the current step's prefill index has
                # been built.  Collect transitions and apply them in a
                # release-all/register-all order after the loop: if two
                # requests share the same prefill row and finish together,
                # the first decode registration must not see the shared
                # prefill row as still occupied by the second request.
                num_computed_after = num_computed + n_tokens
                if num_computed_after >= num_prompt:
                    decode_hash = int(
                        self.input_batch.request_decode_steering_hash[req_index]
                    )
                    req_state = self.requests.get(req_id)
                    sp = req_state.sampling_params if req_state is not None else None
                    steering_transitions.append(
                        (req_id, prefill_hash, decode_hash, sp)
                    )
                    sae_transitions.append((req_id, prefill_hash, decode_hash, sp))
            else:
                # Decode: use decode steering hash
                decode_hash = int(
                    self.input_batch.request_decode_steering_hash[req_index]
                )
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

        # Mark the index as having non-zero row references this step. The
        # no-active-state short-circuit on a future step will zero the index
        # if needed when transitioning back to "nothing active".
        self._steering_index_dirty = True

        self._apply_batched_steering_transitions(steering_transitions)

        # Parallel SAE pass: populate per-(layer, hook) SAE clamp
        # tables and build the shared sae_index.  Independent of the
        # additive flow above — a request may carry only additive,
        # only SAE, or both.
        self._update_sae_buffers(scheduler_output)
        self._apply_batched_sae_transitions(sae_transitions)
        # Phase-4 full-reconstruction pass: structurally identical to
        # the delta pass above, with its own manager, registry, sites,
        # and shared per-token routing tensor (``sae_recon_index``).
        # Guarded so harnesses that don't initialise the new state
        # (e.g. delta-only test runners) skip cleanly.
        if getattr(self, "_sae_fr_clamp_manager", None) is not None:
            self._update_sae_full_recon_buffers(scheduler_output)
            self._apply_batched_sae_full_recon_transitions(sae_transitions)

    def _apply_batched_steering_transitions(
        self,
        transitions: list[tuple[str, int, int, SamplingParams | None]],
    ) -> None:
        """Apply prefill->decode additive transitions for one scheduler step.

        The current step's ``steering_index`` is already built against
        prefill rows.  Release all completed prefill references before
        registering any decode rows so requests sharing the same prefill row
        can free it before the first decode allocation is attempted.
        """
        if not transitions:
            return
        mgr = self._steering_manager
        assert mgr is not None, (
            "_apply_batched_steering_transitions called without an "
            "initialised manager"
        )
        for _req_id, prefill_hash, _decode_hash, _sp in transitions:
            if prefill_hash != 0:
                mgr.release_config(prefill_hash, "prefill")

        for req_id, _prefill_hash, decode_hash, sp in transitions:
            if decode_hash != 0 and sp is not None:
                effective_decode = self._resolve_request_steering(sp, "decode")
                if effective_decode:
                    mgr.register_config(
                        decode_hash,
                        effective_decode,
                        phase="decode",
                        content_hash=sp.decode_additive_steering_config_hash,
                        locally_owned_layers=self._locally_owned_layers,
                    )
            self._req_steering_phase[req_id] = "decode"
            self._transition_scan_candidates().discard(req_id)

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
                if sp is not None and sp.sae_clamp_specs
                else None
            )
            decode_sae_hash = (
                sp.decode_sae_clamp_config_hash
                if sp is not None and sp.sae_clamp_specs
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
            prefill_hash = int(
                self.input_batch.request_prefill_steering_hash[req_index]
            )
            decode_hash = int(self.input_batch.request_decode_steering_hash[req_index])
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
        self._apply_batched_steering_transitions(transitions)
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
        if sae_mgr is None or sp is None or not sp.sae_clamp_specs:
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
            for (_module_name, _layer_idx, hook_str), site in (
                self._sae_steerable_sites.items()
            ):
                try:
                    hook_point = SteeringHookPoint(hook_str)
                except ValueError:
                    continue
                flag_buf = getattr(
                    site,
                    HOOK_POINT_SAE_ANY_ACTIVE_ATTR[hook_point],
                    None,
                )
                if flag_buf is not None:
                    flag_buf.zero_()

        # Fast no-active path: if every SAE row has been released, the
        # only required work is clearing a previously nonzero shared
        # index.  Stale nonzero table rows are harmless once no token
        # points at them.  Clear each site's ``any_active`` flag on the
        # active->inactive transition so layer hooks skip the SAE op
        # entirely until a later row reuse repopulates tables.
        if not sae_mgr.config_to_row:
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
            and not sae_buffers_attached(any_layer, SteeringHookPoint.POST_MLP)
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

            if is_prefilling:
                combined_hash = int(
                    self.input_batch.request_prefill_steering_hash[req_index]
                )
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
                combined_hash = int(
                    self.input_batch.request_decode_steering_hash[req_index]
                )
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
                any_layer, SteeringHookPoint.POST_MLP
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

            if is_prefilling:
                combined_hash = int(
                    self.input_batch.request_prefill_steering_hash[req_index]
                )
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
                combined_hash = int(
                    self.input_batch.request_decode_steering_hash[req_index]
                )
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

    def _handle_steering_transition(
        self,
        req_id: str,
        req_index: int,
        prefill_hash: int,
        *,
        handle_sae: bool = True,
    ) -> None:
        """Handle prefill->decode steering config transition.

        Called when a request will complete prefill after this step.
        Releases the prefill config and registers the decode config
        so it is ready for the next step's table population.

        Capacity for the decode row is reserved by the scheduler at
        the same step the prefill is scheduled to complete (see the
        ``will_complete`` branch in ``Scheduler._schedule_running``),
        so ``register_config`` is expected to succeed.  If it raises,
        that's a scheduler accounting bug and the exception
        propagates.
        """
        mgr = self._steering_manager
        assert mgr is not None, (
            "_handle_steering_transition called without an initialised manager"
        )
        if prefill_hash != 0:
            mgr.release_config(prefill_hash, "prefill")

        decode_hash = int(self.input_batch.request_decode_steering_hash[req_index])
        req_state = self.requests.get(req_id)
        sp = req_state.sampling_params if req_state is not None else None
        if decode_hash != 0 and sp is not None:
            effective_decode = self._resolve_request_steering(sp, "decode")
            if effective_decode:
                mgr.register_config(
                    decode_hash,
                    effective_decode,
                    phase="decode",
                    content_hash=sp.decode_additive_steering_config_hash,
                    locally_owned_layers=self._locally_owned_layers,
                )

        self._req_steering_phase[req_id] = "decode"

        # SAE-side transition must happen after the current step's
        # ``sae_index`` is populated.  The main update loop passes
        # ``handle_sae=False`` and calls this separately after
        # ``_update_sae_buffers``; direct callers keep the historical
        # one-shot behavior.
        if handle_sae:
            self._handle_sae_transition(req_id, prefill_hash, decode_hash, sp)

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
        prefill config.  The scheduler reserves the prefill row when it
        re-admits the resumed request, so ``register_config`` is expected
        to succeed; a ``RuntimeError`` here indicates a scheduler bug and
        propagates.
        """
        mgr = self._steering_manager
        if mgr is None:
            return
        prev_phase = self._req_steering_phase.get(req_id)
        if prev_phase != "decode":
            return
        if new_num_computed_tokens >= req_state.num_prompt_tokens:
            return  # still in decode, nothing to reset

        self._transition_scan_candidates().discard(req_id)

        # Release the stale decode config.
        if req_state.decode_steering_config_hash != 0:
            mgr.release_config(req_state.decode_steering_config_hash, "decode")

        self._req_steering_phase[req_id] = "prefill"

        sp = req_state.sampling_params
        prefill_hash = req_state.prefill_steering_config_hash
        decode_hash = req_state.decode_steering_config_hash
        additive_registered: tuple[int, str] | None = None
        if sp is not None and prefill_hash != 0:
            effective_prefill = self._resolve_request_steering(sp, "prefill")
            if effective_prefill:
                mgr.register_config(
                    prefill_hash,
                    effective_prefill,
                    phase="prefill",
                    content_hash=sp.prefill_additive_steering_config_hash,
                    locally_owned_layers=self._locally_owned_layers,
                )
                additive_registered = (prefill_hash, "prefill")
            elif decode_hash != 0:
                self._transition_scan_candidates().add(req_id)
        elif decode_hash != 0:
            self._transition_scan_candidates().add(req_id)

        # SAE-side reset: a request being resumed back into prefill
        # also needs its SAE row reset.  Release any decode-phase row
        # we admitted, then register the prefill row.
        sae_mgr = self._sae_clamp_manager
        sae_registered: tuple[int, str] | None = None
        fr_registered: tuple[int, str] | None = None
        if sae_mgr is not None and sp is not None and sp.sae_clamp_specs:
            try:
                old_row = self._get_req_sae_row(req_id)
                if (
                    old_row is not None
                    and old_row[1] == "decode"
                    and old_row[0] != 0
                ):
                    sae_mgr.release_clamp_spec(old_row[0], "decode")
                    self._pop_req_sae_row(req_id)
                prefill_specs = sp._phase_filtered_sae_specs("prefill")
                prefill_sae_hash = sp.prefill_sae_clamp_config_hash
                if prefill_hash != 0 and prefill_sae_hash != 0 and prefill_specs:
                    sae_mgr.register_clamp_spec(
                        prefill_sae_hash, prefill_specs, "prefill"
                    )
                    self._set_req_sae_row(req_id, prefill_sae_hash, "prefill")
                    sae_registered = (prefill_sae_hash, "prefill")
                else:
                    self._pop_req_sae_row(req_id)
            except Exception:
                if sae_registered is not None:
                    sae_mgr.release_clamp_spec(*sae_registered)
                if additive_registered is not None:
                    mgr.release_config(*additive_registered)
                self._req_steering_phase.pop(req_id, None)
                self._pop_req_sae_row(req_id)
                self._pop_req_sae_fr_row(req_id)
                self._transition_scan_candidates().discard(req_id)
                raise

        fr_mgr = getattr(self, "_sae_fr_clamp_manager", None)
        if (
            fr_mgr is not None
            and sp is not None
            and getattr(sp, "sae_full_reconstruction_specs", None)
        ):
            try:
                old_row = self._get_req_sae_fr_row(req_id)
                if (
                    old_row is not None
                    and old_row[1] == "decode"
                    and old_row[0] != 0
                ):
                    fr_mgr.release_recon_spec(old_row[0], "decode")
                    self._pop_req_sae_fr_row(req_id)
                prefill_specs = sp._phase_filtered_sae_full_recon_specs("prefill")
                prefill_recon_hash = sp.prefill_sae_full_recon_config_hash
                if (
                    prefill_hash != 0
                    and prefill_recon_hash != 0
                    and prefill_specs
                ):
                    fr_mgr.register_recon_spec(
                        prefill_recon_hash, prefill_specs, "prefill"
                    )
                    self._set_req_sae_fr_row(
                        req_id, prefill_recon_hash, "prefill"
                    )
                    fr_registered = (prefill_recon_hash, "prefill")
                else:
                    self._pop_req_sae_fr_row(req_id)
            except Exception:
                if fr_registered is not None:
                    fr_mgr.release_recon_spec(*fr_registered)
                if sae_registered is not None and sae_mgr is not None:
                    sae_mgr.release_clamp_spec(*sae_registered)
                if additive_registered is not None:
                    mgr.release_config(*additive_registered)
                self._req_steering_phase.pop(req_id, None)
                self._pop_req_sae_row(req_id)
                self._pop_req_sae_fr_row(req_id)
                self._transition_scan_candidates().discard(req_id)
                raise

    # -----------------------------------------------------------------------
    # Hooks called from _update_states() / _update_streaming_request()
    # -----------------------------------------------------------------------

    def _release_finished_steering_configs(
        self, finished_req_ids: "set[str] | list[str]"
    ) -> None:
        """Release the currently-active steering config for finished requests.

        Called before finished request state is popped so
        ``prefill_steering_config_hash`` /
        ``decode_steering_config_hash`` are still accessible.
        """
        mgr = self._steering_manager

        for req_id in finished_req_ids:
            self._transition_scan_candidates().discard(req_id)
            phase = self._req_steering_phase.pop(req_id, None)
            req_state = self.requests.get(req_id)
            if mgr is not None and phase is not None and req_state is not None:
                if phase == "prefill":
                    h = req_state.prefill_steering_config_hash
                else:
                    h = req_state.decode_steering_config_hash
                if h != 0:
                    mgr.release_config(h, phase)
            # SAE-side release runs even when the additive phase was
            # never tracked (e.g. an SAE-only request): it pops the
            # SAE tracker and releases the right hash for whichever
            # phase the request was last admitted under.
            if req_state is not None:
                self._release_sae_for_request(
                    req_id,
                    req_state.prefill_steering_config_hash,
                    req_state.decode_steering_config_hash,
                )
                if getattr(self, "_sae_fr_clamp_manager", None) is not None:
                    self._release_sae_full_recon_for_request(
                        req_id,
                        req_state.prefill_steering_config_hash,
                        req_state.decode_steering_config_hash,
                    )

    def _register_initial_steering_config(
        self,
        req_id: str,
        new_req_data: "NewRequestData",
        req_state: "CachedRequestState",
    ) -> None:
        """Register the initial-phase steering config for a new request.

        Normally requests start in prefill, but a full prefix-cache hit
        (``num_computed >= num_prompt``) puts a request directly into
        decode.  The scheduler reserves the appropriate row at admission
        time, so ``register_config`` is expected to succeed; a
        ``RuntimeError`` here indicates a scheduler bug and propagates.
        """
        mgr = self._steering_manager
        if mgr is None or new_req_data.sampling_params is None:
            return

        sp = new_req_data.sampling_params
        # Validate SAE clamp spec against the registered SAE modules
        # before admitting (kernel feasibility check).  Stage 2 also
        # admits the spec into the SAE manager below.
        self._assert_sae_clamps_can_be_applied(sp)
        if getattr(self, "_sae_fr_module_registry", None) is not None:
            self._assert_sae_full_recon_specs_can_be_applied(sp)
        prefill_hash = new_req_data.prefill_steering_config_hash
        decode_hash = new_req_data.decode_steering_config_hash
        is_prefilling = new_req_data.num_computed_tokens < req_state.num_prompt_tokens
        prefill_registered = False
        additive_registered: tuple[int, str] | None = None
        if not is_prefilling:
            # Already past prefill — register decode config.
            effective_decode = self._resolve_request_steering(sp, "decode")
            if decode_hash != 0 and effective_decode:
                mgr.register_config(
                    decode_hash,
                    effective_decode,
                    phase="decode",
                    content_hash=sp.decode_additive_steering_config_hash,
                    locally_owned_layers=self._locally_owned_layers,
                )
                additive_registered = (decode_hash, "decode")
            self._req_steering_phase[req_id] = "decode"
        else:
            # Normal: start in prefill; decode registered
            # on transition in _update_steering_buffers.
            effective_prefill = self._resolve_request_steering(sp, "prefill")
            if prefill_hash != 0 and effective_prefill:
                mgr.register_config(
                    prefill_hash,
                    effective_prefill,
                    phase="prefill",
                    content_hash=sp.prefill_additive_steering_config_hash,
                    locally_owned_layers=self._locally_owned_layers,
                )
                prefill_registered = True
                additive_registered = (prefill_hash, "prefill")
            self._req_steering_phase[req_id] = "prefill"
        # SAE-side admission runs in parallel using SAE-only row hashes;
        # additive and SAE managers deduplicate independently.
        try:
            self._register_initial_sae_clamps(
                req_id, sp, prefill_hash, decode_hash, is_prefilling
            )
            if getattr(self, "_sae_fr_clamp_manager", None) is not None:
                self._register_initial_sae_full_recon(
                    req_id, sp, prefill_hash, decode_hash, is_prefilling
                )
        except Exception:
            if additive_registered is not None:
                mgr.release_config(*additive_registered)
            self._req_steering_phase.pop(req_id, None)
            self._pop_req_sae_row(req_id)
            self._pop_req_sae_fr_row(req_id)
            self._transition_scan_candidates().discard(req_id)
            raise
        if is_prefilling:
            prefill_sae_specs = (
                sp._phase_filtered_sae_specs("prefill")
                if sp.sae_clamp_specs
                else None
            )
            prefill_fr_specs = (
                sp._phase_filtered_sae_full_recon_specs("prefill")
                if getattr(sp, "sae_full_reconstruction_specs", None)
                else None
            )
            if (
                decode_hash != 0
                and not prefill_registered
                and not prefill_sae_specs
                and not prefill_fr_specs
            ):
                self._transition_scan_candidates().add(req_id)

    def _refresh_streaming_steering(
        self,
        req_id: str,
        new_req_data: "NewRequestData",
        old_prefill_hash: int,
        old_decode_hash: int,
        new_prefill_hash: int,
        new_decode_hash: int,
        old_sampling_params: "SamplingParams | None" = None,
    ) -> None:
        """Refresh steering state for a streaming re-added request.

        Streaming re-adds go back through prefill, so we must:
        1. Release the old config (whatever phase we were tracking)
        2. Register the new prefill config
        3. Update phase tracking

        The scheduler reserves the prefill row when re-admitting the
        streaming request, so ``register_config`` is expected to
        succeed; a ``RuntimeError`` here indicates a scheduler bug and
        propagates.
        """
        mgr = self._steering_manager
        if mgr is None:
            return

        sp = new_req_data.sampling_params
        if sp is not None:
            self._assert_sae_clamps_can_be_applied(sp)
            if getattr(self, "_sae_fr_module_registry", None) is not None:
                self._assert_sae_full_recon_specs_can_be_applied(sp)

        old_phase = self._req_steering_phase.get(req_id)
        old_sp: SamplingParams | None
        if old_sampling_params is not None:
            old_sp = old_sampling_params
        else:
            old_req_state = self.requests.get(req_id)
            old_sp = (
                old_req_state.sampling_params
                if old_req_state is not None
                and getattr(old_req_state, "sampling_params", None) is not None
                else None
            )
        old_additive_restore: tuple[int, str, dict, int] | None = None
        if old_phase is not None and old_sp is not None:
            old_hash = old_prefill_hash if old_phase == "prefill" else old_decode_hash
            if old_hash != 0:
                old_effective = self._resolve_request_steering(old_sp, old_phase)
                if old_effective:
                    old_content_hash = (
                        old_sp.prefill_additive_steering_config_hash
                        if old_phase == "prefill"
                        else old_sp.decode_additive_steering_config_hash
                    )
                    old_additive_restore = (
                        old_hash,
                        old_phase,
                        old_effective,
                        old_content_hash,
                    )

        sae_mgr = self._sae_clamp_manager
        old_sae_restore: tuple[int, str, tuple] | None = None
        if sae_mgr is not None:
            old_sae_row = self._get_req_sae_row(req_id)
            if old_sae_row is not None and old_sae_row[0] != 0:
                old_sae_specs = sae_mgr.config_specs.get(old_sae_row)
                if old_sae_specs is not None:
                    old_sae_restore = (old_sae_row[0], old_sae_row[1], old_sae_specs)

        fr_mgr = getattr(self, "_sae_fr_clamp_manager", None)
        old_fr_restore: tuple[int, str, tuple] | None = None
        if fr_mgr is not None:
            old_fr_row = self._get_req_sae_fr_row(req_id)
            if old_fr_row is not None and old_fr_row[0] != 0:
                old_fr_specs = fr_mgr.config_specs.get(old_fr_row)
                if old_fr_specs is not None:
                    old_fr_restore = (old_fr_row[0], old_fr_row[1], old_fr_specs)

        transition_candidates = self.__dict__.get("_req_transition_scan_candidates")
        if not isinstance(transition_candidates, set):
            transition_candidates = set()
            self.__dict__["_req_transition_scan_candidates"] = transition_candidates
        was_transition_candidate = req_id in transition_candidates

        def restore_old_state() -> None:
            if old_additive_restore is not None:
                old_hash, phase, old_effective, old_content_hash = (
                    old_additive_restore
                )
                mgr.register_config(
                    old_hash,
                    old_effective,
                    phase=phase,
                    content_hash=old_content_hash,
                    locally_owned_layers=self._locally_owned_layers,
                )
            if old_phase is not None:
                self._req_steering_phase[req_id] = old_phase
            else:
                self._req_steering_phase.pop(req_id, None)
            if sae_mgr is not None and old_sae_restore is not None:
                old_sae_hash, old_sae_phase, old_sae_specs = old_sae_restore
                sae_mgr.register_clamp_spec(
                    old_sae_hash, old_sae_specs, old_sae_phase
                )
                self._set_req_sae_row(req_id, old_sae_hash, old_sae_phase)
            elif sae_mgr is not None:
                self._pop_req_sae_row(req_id)
            if fr_mgr is not None and old_fr_restore is not None:
                old_fr_hash, old_fr_phase, old_fr_specs = old_fr_restore
                fr_mgr.register_recon_spec(old_fr_hash, old_fr_specs, old_fr_phase)
                self._set_req_sae_fr_row(req_id, old_fr_hash, old_fr_phase)
            elif fr_mgr is not None:
                self._pop_req_sae_fr_row(req_id)
            if was_transition_candidate:
                transition_candidates.add(req_id)
            else:
                transition_candidates.discard(req_id)

        # Release the old phase config.
        transition_candidates.discard(req_id)
        additive_registered: tuple[int, str] | None = None
        try:
            if old_phase is not None:
                if old_phase == "prefill" and old_prefill_hash != 0:
                    mgr.release_config(old_prefill_hash, "prefill")
                elif old_phase == "decode" and old_decode_hash != 0:
                    mgr.release_config(old_decode_hash, "decode")

            # Register new prefill config (streaming re-adds start
            # in prefill).
            effective_prefill = (
                self._resolve_request_steering(sp, "prefill")
                if sp is not None
                else None
            )
            if new_prefill_hash != 0 and sp is not None and effective_prefill:
                mgr.register_config(
                    new_prefill_hash,
                    effective_prefill,
                    phase="prefill",
                    content_hash=sp.prefill_additive_steering_config_hash,
                    locally_owned_layers=self._locally_owned_layers,
                )
                additive_registered = (new_prefill_hash, "prefill")
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
                if new_decode_hash != 0:
                    transition_candidates.add(req_id)

            # SAE-side refresh: same release/register dance against the
            # SAE manager.  A streaming re-add always re-enters prefill,
            # so register the new prefill SAE clamps if the request
            # carries any.
            if sae_mgr is not None:
                old_row = self._get_req_sae_row(req_id)
                if old_row is not None and old_row[0] != 0:
                    sae_mgr.release_clamp_spec(old_row[0], old_row[1])
                    self._pop_req_sae_row(req_id)
                prefill_specs = (
                    sp._phase_filtered_sae_specs("prefill")
                    if sp is not None and sp.sae_clamp_specs
                    else None
                )
                prefill_sae_hash = (
                    sp.prefill_sae_clamp_config_hash
                    if sp is not None and sp.sae_clamp_specs
                    else 0
                )
                if new_prefill_hash != 0 and prefill_sae_hash != 0 and prefill_specs:
                    sae_mgr.register_clamp_spec(
                        prefill_sae_hash, prefill_specs, "prefill"
                    )
                    self._set_req_sae_row(req_id, prefill_sae_hash, "prefill")
                else:
                    self._pop_req_sae_row(req_id)
            if fr_mgr is not None:
                old_row = self._get_req_sae_fr_row(req_id)
                if old_row is not None and old_row[0] != 0:
                    fr_mgr.release_recon_spec(old_row[0], old_row[1])
                    self._pop_req_sae_fr_row(req_id)
                prefill_specs = (
                    sp._phase_filtered_sae_full_recon_specs("prefill")
                    if sp is not None
                    and getattr(sp, "sae_full_reconstruction_specs", None)
                    else None
                )
                prefill_recon_hash = (
                    sp.prefill_sae_full_recon_config_hash
                    if sp is not None
                    and getattr(sp, "sae_full_reconstruction_specs", None)
                    else 0
                )
                if (
                    new_prefill_hash != 0
                    and prefill_recon_hash != 0
                    and prefill_specs
                ):
                    fr_mgr.register_recon_spec(
                        prefill_recon_hash, prefill_specs, "prefill"
                    )
                    self._set_req_sae_fr_row(
                        req_id, prefill_recon_hash, "prefill"
                    )
                else:
                    self._pop_req_sae_fr_row(req_id)
        except Exception:
            if additive_registered is not None:
                mgr.release_config(*additive_registered)
            self._req_steering_phase.pop(req_id, None)
            self._pop_req_sae_row(req_id)
            self._pop_req_sae_fr_row(req_id)
            restore_old_state()
            raise
