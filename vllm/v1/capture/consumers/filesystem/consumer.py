# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filesystem capture consumer — streams captured activations to disk
via ``ActivationWriter``.

Implements ``CaptureSink`` directly (bypasses ``CaptureConsumer`` /
``_BatchedAdapter``) because long captures must not be buffered in
memory. Each ``CaptureChunk`` is converted to bytes and submitted as a
``WriteTask``; ``CaptureFinalize`` produces a ``FinalizeTask`` with
sidecar JSON.

See ``docs/design/capture_consumers.md`` for the rationale.
"""

from __future__ import annotations

import logging
import pathlib
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from vllm.v1.capture.consumers.filesystem.types import (
    VALID_LAYOUTS,
    FilesystemCaptureRequest,
    FilesystemConsumerParams,
    packed_bin_name,
    packed_index_name,
    shard_bin_name,
    shard_index_name,
)
from vllm.v1.capture.consumers.filesystem.writer import (
    ActivationWriter,
    FinalizeTask,
    WriteTask,
)
from vllm.v1.capture.errors import CaptureValidationError
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureFinalize,
    CaptureKey,
    CaptureResult,
    CaptureSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.capture.types import CaptureContext

logger = logging.getLogger(__name__)


def _parse_params(params: dict[str, Any]) -> FilesystemConsumerParams:
    """Parse a raw ``params`` dict into ``FilesystemConsumerParams``.

    Only extracts known fields; unknown keys are silently ignored so
    forward-compatible config additions don't break old consumers.
    """
    if "root" not in params:
        raise ValueError(
            "FilesystemConsumer requires a 'root' parameter pointing "
            "to the output directory."
        )
    default_layout = str(params.get("default_layout", "per_file"))
    if default_layout not in VALID_LAYOUTS:
        raise ValueError(
            f"default_layout must be one of {sorted(VALID_LAYOUTS)}, "
            f"got {default_layout!r}"
        )
    return FilesystemConsumerParams(
        root=str(params["root"]),
        writer_threads=int(params.get("writer_threads", 4)),
        queue_size=int(params.get("queue_size", 1024)),
        timeout_seconds=float(params.get("timeout_seconds", 30.0)),
        on_collision=str(params.get("on_collision", "overwrite")),
        fd_cache_size=int(params.get("fd_cache_size", 256)),
        fsync=bool(params.get("fsync", True)),
        atomic_publish=bool(params.get("atomic_publish", True)),
        default_layout=str(params.get("default_layout", "per_file")),
        coalesce_max_bytes=int(params.get("coalesce_max_bytes", 1 << 20)),
        num_shards=int(params.get("num_shards", 8)),
        shard_max_bytes=int(params.get("shard_max_bytes", 256 << 20)),
    )


def _pp_geometry(
    vllm_config: Any,
) -> tuple[int, int, tuple[int, int] | None]:
    """Resolve ``(pp_size, pp_rank, local_layer_range)`` for this worker.

    Under pipeline parallelism each stage's consumer runs in its own
    process, owns a disjoint slice of the model's layers, and must (a)
    write packed/sharded files under per-stage names so stages don't race
    for the same path on the shared mount, and (b) track completion against
    only the layers this stage actually captures. Both need the stage's
    pp rank and its ``[start, end)`` global layer slice — derived here
    exactly as :meth:`ModelConfig.get_layers_start_end_indices` /
    ``CaptureManager`` derive them, so the consumer and manager agree on
    the slice.

    Returns ``(1, 0, None)`` when pipeline parallelism is disabled or the
    config does not expose usable parallel/model info (e.g. a test
    ``MagicMock``, whose attributes don't coerce to ``int``). In that case
    packed/sharded use the legacy pp-agnostic filenames and track every
    requested layer.
    """
    try:
        parallel_config = vllm_config.parallel_config
        pp_size = int(parallel_config.pipeline_parallel_size)
    except (AttributeError, TypeError, ValueError):
        return 1, 0, None
    if pp_size <= 1:
        return 1, 0, None
    # PP is genuinely enabled: a failure to resolve the layer slice here is
    # a real misconfiguration, so let it surface rather than silently
    # falling back to colliding pp-agnostic filenames.
    tp_size = int(parallel_config.tensor_parallel_size)
    pp_rank = (int(parallel_config.rank) // tp_size) % pp_size
    start, end = vllm_config.model_config.get_layers_start_end_indices(parallel_config)
    return pp_size, pp_rank, (int(start), int(end))


# Sentinel writer-key components for a request's single packed file.
# layer=-1 / hook="__packed__" never collide with real (layer>=0, hook)
# keys, so all of a request's captures funnel to one writer fd + result.
_PACKED_LAYER = -1
_PACKED_HOOK = "__packed__"


@dataclass
class _PackedState:
    """Per-request accumulation state for the ``packed`` layout.

    Mutated on the dispatch thread (``submit_chunk``) and the finalize
    thread (``submit_finalize``) for *different* requests concurrently,
    so all access is guarded by ``FilesystemConsumer._lock``. Within a
    single request, chunks (dispatch thread) all precede finalizes
    (finalize thread) thanks to the manager's dispatch-drain barrier, so
    byte offsets recorded here match submission/append order exactly.
    """

    tag_slug: str
    request_id_slug: str
    expected_keys: set[tuple[int, str]]
    dtype: str | None = None
    running_offset: int = 0
    entries: list[dict[str, Any]] = field(default_factory=list)
    finalized: set[tuple[int, str]] = field(default_factory=set)
    sidecar_fields: dict[str, Any] = field(default_factory=dict)
    # Cached on first chunk so the (slow) pathlib construction runs once
    # per request instead of once per submitted batch.
    bin_path: pathlib.Path | None = None


# Sentinel hook for shard writer keys (see _ShardState). The shard
# writer key is ``(f"{tag_slug}#{shard_idx}", seq, _SHARD_HOOK)``: a
# stable key[0] keeps a shard pinned to one writer thread (ordered
# appends), and ``seq`` in the layer slot rotates the fd on each seal.
_SHARD_HOOK = "__shard__"


@dataclass
class _ShardState:
    """Accumulation state for one shard file (``sharded`` layout).

    Keyed by ``(tag_slug, shard_idx)`` in the consumer. Holds the current
    rotation ``seq``, byte offset, and the per-chunk index entries (each
    carrying ``request_id``, since a shard interleaves many requests).
    Sealed when ``running_offset`` crosses ``shard_max_bytes`` or at
    shutdown; sealing publishes the index + renames the .bin and bumps
    ``seq`` to start a fresh shard. Guarded by ``FilesystemConsumer._lock``.
    """

    tag_slug: str
    shard_idx: int
    seq: int = 0
    running_offset: int = 0
    entries: list[dict[str, Any]] = field(default_factory=list)
    dtype: str | None = None


@dataclass
class _ShardedRequestState:
    """Per-request completion tracking for the ``sharded`` layout.

    A request's chunks all go to one shard (``hash(request_id) %
    num_shards`` within its tag). Unlike per_file/packed, the request's
    result is *not* a writer ``WriteResult`` (the shard file is shared and
    seals asynchronously): the request is reported ``ok`` once all its
    expected keys finalize, with payload = the shard file(s) it landed in.
    A capture becomes *readable* only after its shard seals (size or
    shutdown) — an end-of-run/bulk model.
    """

    tag_slug: str
    request_id_slug: str
    shard_idx: int
    expected_keys: set[tuple[int, str]]
    finalized: set[tuple[int, str]] = field(default_factory=set)
    shard_bins: set[str] = field(default_factory=set)
    done: bool = False


class FilesystemConsumer:
    """Capture consumer that streams activations to the filesystem.

    Implements the ``CaptureSink`` protocol directly. Wraps the
    in-package ``ActivationWriter`` thread pool.

    Path layout: ``{root}/{tag_slug}/{request_id_slug}/{layer}_{hook}.bin``
    """

    location: Literal["worker", "driver"] = "worker"
    reads_client_spec: ClassVar[bool] = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        params: dict[str, Any],
    ) -> None:
        self._vllm_config = vllm_config
        self._params = _parse_params(params)
        # ``expanduser`` so ``root=~/path`` works: a shell does not expand the
        # ``~`` in ``--capture-consumers filesystem:root=~/path`` (it is not at
        # a word boundary), so the literal ``~`` would otherwise become a
        # directory under the server's cwd.
        self._root = pathlib.Path(self._params.root).expanduser()
        # Pipeline-parallel geometry. Under PP each stage owns a disjoint
        # global layer slice and writes its own packed/sharded files keyed
        # by stage rank (``_file_pp_rank``); completion is tracked against
        # only this stage's layers (``_local_layer_range``). With PP off
        # both are inert (legacy filenames, all layers expected).
        self._pp_size, self._pp_rank, self._local_layer_range = _pp_geometry(
            vllm_config
        )
        # ``None`` selects the legacy pp-agnostic filenames; an int embeds
        # the stage rank so per-stage files never collide on the mount.
        self._file_pp_rank: int | None = None if self._pp_size <= 1 else self._pp_rank
        self._packed_bin_name = packed_bin_name(self._file_pp_rank)
        self._packed_index_name = packed_index_name(self._file_pp_rank)
        self._writer = ActivationWriter(
            root=self._root,
            num_threads=self._params.writer_threads,
            queue_size=self._params.queue_size,
            timeout_seconds=self._params.timeout_seconds,
            on_collision=self._params.on_collision,  # type: ignore[arg-type]
            fd_cache_size=self._params.fd_cache_size,
            fsync=self._params.fsync,
            atomic_publish=self._params.atomic_publish,
            coalesce_max_bytes=self._params.coalesce_max_bytes,
        )
        # Lock protecting _key_paths which tracks the slug-based path
        # components per CaptureKey for use at finalize time.
        self._lock = threading.Lock()
        self._key_paths: dict[CaptureKey, tuple[str, str]] = {}
        # Slug components recorded at admission time, keyed by
        # ``vllm_internal_request_id``. The framework's ``CaptureSpec``
        # only carries ``(hooks, positions)``, so the per-request
        # ``tag`` and ``request_id`` from the client's
        # ``FilesystemCaptureRequest`` would otherwise be lost between
        # ``validate_client_spec`` and ``submit_chunk``/``submit_finalize``.
        # Looked up by ``str(_request_id)`` because ``CaptureKey``'s
        # request-id component carries that string form.
        self._request_slugs: dict[str, tuple[str, str]] = {}
        # Per-request resolved layout ("per_file" | "packed"), recorded
        # at admission. Requests not seen at admission default to the
        # consumer-level ``default_layout``.
        self._request_layout: dict[str, str] = {}
        # Per-request packed accumulation state, created lazily on first
        # chunk/finalize for a packed request. Guarded by self._lock.
        self._packed_states: dict[str, _PackedState] = {}
        # "sharded" layout state. Open shards keyed by (tag_slug,
        # shard_idx); per-request completion tracking keyed by req_str;
        # per-request completion events for wait_for_result. All guarded
        # by self._lock / self._wait_lock.
        self._shard_states: dict[tuple[str, int], _ShardState] = {}
        self._sharded_requests: dict[str, _ShardedRequestState] = {}
        self._sharded_events: dict[str, threading.Event] = {}
        # Logical dtype string per key (per_file), captured from the
        # first chunk so the finalize sidecar is self-describing.
        self._key_dtype: dict[CaptureKey, str] = {}
        # Total captured rows per key, accumulated across chunks. Used
        # to write ``shape`` into the sidecar JSON at finalize time so
        # readers can ``np.frombuffer`` the .bin without recomputing
        # from filesize / hidden_size externally.
        self._key_shapes: dict[CaptureKey, list[int]] = {}
        # Per writer-key events set by _on_write_result when a result
        # goes terminal. Keyed by writer-side (str, int, str) tuples.
        self._wait_lock = threading.Lock()
        self._wait_events: dict[tuple[str, int, str], threading.Event] = {}
        self._writer.add_status_callback(self._on_write_result)

    # ------------------------------------------------------------------
    # CaptureSink protocol
    # ------------------------------------------------------------------

    def global_capture_spec(self) -> None:
        """Filesystem consumer has no global spec — capture is per-request."""
        return None

    def validate_client_spec(
        self,
        raw_spec: Any,
        ctx: CaptureContext,
    ) -> CaptureSpec:
        """Validate a per-request ``FilesystemCaptureRequest``.

        Accepts either a ``FilesystemCaptureRequest`` dataclass or a
        plain dict with the same fields.
        """
        if isinstance(raw_spec, dict):
            raw_spec = FilesystemCaptureRequest(**raw_spec)
        if not isinstance(raw_spec, FilesystemCaptureRequest):
            raise TypeError(
                f"Expected FilesystemCaptureRequest or dict, "
                f"got {type(raw_spec).__name__}"
            )
        # Lazy import to avoid pulling in pydantic (via vllm.config) at
        # module load time — keeps the consumer importable in lightweight
        # test environments.
        from vllm.v1.capture.consumers.filesystem.validation import (
            slug_path_components,
            validate_filesystem_request,
        )

        spec = validate_filesystem_request(raw_spec, self._vllm_config, ctx)
        # Record the slugs keyed by vllm_internal_request_id. The
        # ``CaptureSpec`` we return only carries hooks+positions; we
        # need the tag/request_id again at submit time to build the
        # correct on-disk path. Slugging happens here (admission) so
        # invalid slugs surface as a HTTP 400 rather than silently
        # falling through to ``"default"``.
        tag_slug, request_id_slug = slug_path_components(raw_spec)

        # Resolve the on-disk layout for this request: explicit
        # per-request ``layout`` wins, else the consumer default.
        layout = raw_spec.layout or self._params.default_layout
        if layout not in VALID_LAYOUTS:
            raise CaptureValidationError(
                f"capture.layout must be one of {sorted(VALID_LAYOUTS)}, got {layout!r}"
            )
        # ``packed``/``sharded`` write one file per request (per tag). Under
        # pipeline parallelism each stage owns a disjoint global layer slice
        # and writes its own ``packed-pp{rank}``/``shard-pp{rank}`` files
        # (see ``_file_pp_rank``), so stages never race for the same path on
        # the shared mount. The pp-agnostic ``per_file`` layout is keyed by
        # global layer index and is collision-free regardless.

        req_str = str(ctx.vllm_internal_request_id)
        # The (layer, hook) set this stage must see finalized before its
        # share of the request is complete (used by packed + sharded). The
        # client spec carries *global* layers (validated against the global
        # layer space); under PP the manager only ever captures and
        # finalizes the layers this stage owns, so completion must be
        # tracked against the local slice — otherwise the packed index
        # (sharded ``done`` flag) would wait forever on layers another
        # stage holds. With PP off ``_local_layer_range`` is ``None`` and
        # every requested layer is expected.
        expected = {
            (layer_idx, hook_name)
            for hook_name, layers in spec.hooks.items()
            for layer_idx in layers
        }
        if self._local_layer_range is not None:
            start, end = self._local_layer_range
            expected = {
                (layer_idx, hook_name)
                for (layer_idx, hook_name) in expected
                if start <= layer_idx < end
            }

        with self._lock:
            self._request_slugs[req_str] = (tag_slug, request_id_slug)
            self._request_layout[req_str] = layout
            # When this stage owns none of the request's layers, the manager
            # drops the consumer for the request — no chunks/finalizes will
            # arrive — so don't create accumulation state that would never
            # complete (and would leak).
            if expected:
                if layout == "packed":
                    self._packed_states[req_str] = _PackedState(
                        tag_slug=tag_slug,
                        request_id_slug=request_id_slug,
                        expected_keys=expected,
                    )
                elif layout == "sharded":
                    shard_idx = hash(req_str) % max(1, self._params.num_shards)
                    self._sharded_requests[req_str] = _ShardedRequestState(
                        tag_slug=tag_slug,
                        request_id_slug=request_id_slug,
                        shard_idx=shard_idx,
                        expected_keys=expected,
                    )
        return spec

    # ------------------------------------------------------------------
    # Layout / key helpers
    # ------------------------------------------------------------------

    def _layout_for(self, req_str: str) -> str:
        """Resolved layout for a request ("per_file" | "packed")."""
        with self._lock:
            return self._request_layout.get(req_str, self._params.default_layout)

    def _writer_key_for(self, key: CaptureKey) -> tuple[str, int, str]:
        """Underlying ActivationWriter key for a capture key.

        ``per_file`` → one writer key per ``(layer, hook)``. ``packed`` →
        a single sentinel key per request, so all of a request's captures
        share one file, one fd, and one ``WriteResult``.
        """
        request_id, layer_idx, hook_name = key
        req_str = str(request_id)
        if self._layout_for(req_str) == "packed":
            return (req_str, _PACKED_LAYER, _PACKED_HOOK)
        return (req_str, layer_idx, hook_name)

    def _resolve_chunk_slugs(
        self, req_str: str, chunk: CaptureChunk
    ) -> tuple[str, str]:
        """Slug priority: chunk.metadata override → admission record →
        ``("default", req)`` fallback."""
        with self._lock:
            recorded = self._request_slugs.get(req_str)
        fallback_tag, fallback_req = recorded or ("default", req_str)
        return (
            chunk.metadata.get("tag_slug", fallback_tag),
            chunk.metadata.get("request_id_slug", fallback_req),
        )

    @staticmethod
    def _tensor_to_bytes(tensor: Any) -> tuple[bytes, int, int, str]:
        """Return ``(raw_bytes, rows, hidden, logical_dtype_str)``.

        bf16 is viewed as uint16 before numpy (numpy has no native bf16),
        matching the on-disk spec; the *logical* dtype string ("bfloat16")
        is still reported so the sidecar stays self-describing.
        """
        import torch as _torch

        dtype_str = str(tensor.dtype).removeprefix("torch.")
        view = tensor.view(_torch.uint16) if tensor.dtype == _torch.bfloat16 else tensor
        rows, hidden = int(tensor.shape[0]), int(tensor.shape[1])
        return view.numpy().tobytes(), rows, hidden, dtype_str

    # ------------------------------------------------------------------
    # submit_chunk
    # ------------------------------------------------------------------

    def submit_chunk(self, chunk: CaptureChunk) -> None:
        """Convert a ``CaptureChunk`` to bytes and submit a ``WriteTask``."""
        req_str = str(chunk.key[0])
        layout = self._layout_for(req_str)
        if layout == "packed":
            self._submit_chunk_packed(chunk, req_str)
        elif layout == "sharded":
            self._submit_chunk_sharded(chunk, req_str)
        else:
            self._submit_chunk_per_file(chunk, req_str)

    def submit_chunk_batch(self, chunks: list[CaptureChunk]) -> None:
        """Submit one dispatch step's chunks, amortizing per-chunk overhead.

        For the ``packed`` layout every chunk of a request in this step
        targets the same file (one writer key), so they are concatenated
        into a single ``WriteTask`` and their index entries recorded under a
        single lock acquisition — collapsing ~num_layers tasks/locks per
        request per step into one. ``per_file``/``sharded`` chunks fall
        through to the per-chunk path (each targets its own key/shard, so
        there is no same-key batching win there).
        """
        # Group by request in one pass, then resolve the layout once per
        # request (not once per chunk — the layout is fixed per request).
        by_req: dict[str, list[CaptureChunk]] = {}
        for chunk in chunks:
            by_req.setdefault(str(chunk.key[0]), []).append(chunk)

        for req_str, group in by_req.items():
            layout = self._layout_for(req_str)
            if layout == "packed":
                self._submit_chunk_packed_batch(group, req_str)
            elif layout == "sharded":
                for chunk in group:
                    self._submit_chunk_sharded(chunk, req_str)
            else:
                for chunk in group:
                    self._submit_chunk_per_file(chunk, req_str)

    def _submit_chunk_per_file(self, chunk: CaptureChunk, req_str: str) -> None:
        key = chunk.key
        _request_id, layer_idx, hook_name = key
        tag_slug, request_id_slug = self._resolve_chunk_slugs(req_str, chunk)
        tensor_bytes, rows, hidden, dtype_str = self._tensor_to_bytes(chunk.tensor)

        # Cache slug paths and accumulate the row count / dtype so the
        # finalize sidecar carries an accurate, self-describing layout.
        with self._lock:
            if key not in self._key_paths:
                self._key_paths[key] = (tag_slug, request_id_slug)
            self._key_dtype.setdefault(key, dtype_str)
            existing = self._key_shapes.get(key)
            if existing is None:
                self._key_shapes[key] = [rows, hidden]
            else:
                existing[0] += rows
                if hidden != existing[1]:
                    logger.warning(
                        "hidden-size mismatch across chunks for key=%s: "
                        "expected %d, got %d",
                        key,
                        existing[1],
                        hidden,
                    )

        bin_path = (
            self._root / tag_slug / request_id_slug / f"{layer_idx}_{hook_name}.bin"
        )
        self._writer.submit(
            WriteTask(
                path=bin_path,
                payload=tensor_bytes,
                append=True,
                key=(req_str, layer_idx, hook_name),
            )
        )

    def _submit_chunk_packed(self, chunk: CaptureChunk, req_str: str) -> None:
        _request_id, layer_idx, hook_name = chunk.key
        tag_slug, request_id_slug = self._resolve_chunk_slugs(req_str, chunk)
        tensor_bytes, rows, hidden, dtype_str = self._tensor_to_bytes(chunk.tensor)
        nbytes = len(tensor_bytes)

        # Reserve a byte range and record the index entry under the lock,
        # then submit outside it. submit_chunk for a single request runs
        # on one (dispatch) thread, so offset-assignment order == append
        # order even with the lock released before submit; cross-request
        # writes target different files, so interleaving is harmless.
        with self._lock:
            state = self._packed_states.get(req_str)
            if state is None:
                # No admission record (e.g. a non-validated path). Create
                # lazily; expected_keys empty → publishes on first
                # finalize. The normal per-request path always records
                # expected_keys in validate_client_spec.
                state = _PackedState(
                    tag_slug=tag_slug,
                    request_id_slug=request_id_slug,
                    expected_keys=set(),
                )
                self._packed_states[req_str] = state
            if state.dtype is None:
                state.dtype = dtype_str
            elif state.dtype != dtype_str:
                logger.warning(
                    "dtype mismatch in packed request %s: %s vs %s",
                    req_str,
                    state.dtype,
                    dtype_str,
                )
            state.entries.append(
                {
                    "layer": layer_idx,
                    "hook": hook_name,
                    "offset": state.running_offset,
                    "nbytes": nbytes,
                    "shape": [rows, hidden],
                }
            )
            state.running_offset += nbytes
            # Use the state's slugs (recorded at admission) so chunk
            # writes and the finalize index land in the same directory.
            bin_tag, bin_req = state.tag_slug, state.request_id_slug

        bin_path = self._root / bin_tag / bin_req / self._packed_bin_name
        self._writer.submit(
            WriteTask(
                path=bin_path,
                payload=tensor_bytes,
                append=True,
                key=(req_str, _PACKED_LAYER, _PACKED_HOOK),
            )
        )

    def _submit_chunk_packed_batch(
        self, chunks: list[CaptureChunk], req_str: str
    ) -> None:
        """Batched packed submit: one WriteTask + one lock for the group.

        All chunks belong to ``req_str`` (same packed file / writer key).
        Serialize outside the lock, reserve all byte ranges and record all
        index entries in one lock acquisition (preserving append order, so
        offsets match the concatenated payload), then submit a single
        ``WriteTask`` with the concatenated bytes.
        """
        if not chunks:
            return
        # Serialize every chunk outside the lock (cheap; not the bottleneck).
        serialized: list[tuple[int, str, bytes, int, int, str]] = []
        for chunk in chunks:
            _request_id, layer_idx, hook_name = chunk.key
            payload, rows, hidden, dtype_str = self._tensor_to_bytes(chunk.tensor)
            serialized.append((layer_idx, hook_name, payload, rows, hidden, dtype_str))

        payloads: list[bytes] = []
        with self._lock:
            state = self._packed_states.get(req_str)
            if state is None:
                # Resolve slugs only on first sight of the request (under the
                # lock we already hold); they're fixed for the request, so
                # this avoids a per-batch _resolve_chunk_slugs lock round-trip.
                recorded = self._request_slugs.get(req_str)
                fallback_tag, fallback_req = recorded or ("default", req_str)
                meta = chunks[0].metadata
                state = _PackedState(
                    tag_slug=meta.get("tag_slug", fallback_tag),
                    request_id_slug=meta.get("request_id_slug", fallback_req),
                    expected_keys=set(),
                )
                self._packed_states[req_str] = state
            for layer_idx, hook_name, payload, rows, hidden, dtype_str in serialized:
                if state.dtype is None:
                    state.dtype = dtype_str
                elif state.dtype != dtype_str:
                    logger.warning(
                        "dtype mismatch in packed request %s: %s vs %s",
                        req_str,
                        state.dtype,
                        dtype_str,
                    )
                state.entries.append(
                    {
                        "layer": layer_idx,
                        "hook": hook_name,
                        "offset": state.running_offset,
                        "nbytes": len(payload),
                        "shape": [rows, hidden],
                    }
                )
                state.running_offset += len(payload)
                payloads.append(payload)
            if state.bin_path is None:
                state.bin_path = (
                    self._root
                    / state.tag_slug
                    / state.request_id_slug
                    / self._packed_bin_name
                )
            bin_path = state.bin_path

        combined = payloads[0] if len(payloads) == 1 else b"".join(payloads)
        self._writer.submit(
            WriteTask(
                path=bin_path,
                payload=combined,
                append=True,
                key=(req_str, _PACKED_LAYER, _PACKED_HOOK),
            )
        )

    def _submit_chunk_sharded(self, chunk: CaptureChunk, req_str: str) -> None:
        _request_id, layer_idx, hook_name = chunk.key
        tensor_bytes, rows, hidden, dtype_str = self._tensor_to_bytes(chunk.tensor)
        nbytes = len(tensor_bytes)

        # Route to the request's shard (assigned at admission). Append the
        # bytes to the shard's open .bin and record a per-chunk index
        # entry (carrying request_id, since shards interleave requests).
        # Seal + rotate the shard if it crosses the size threshold. All
        # state mutation is under the lock; the WriteTask submit (and any
        # seal task) is issued after releasing it.
        seal_task: FinalizeTask | None = None
        with self._lock:
            req = self._sharded_requests.get(req_str)
            if req is None:
                # No admission record (non-validated path) — create lazily
                # with the default shard assignment and empty expected set.
                shard_idx = hash(req_str) % max(1, self._params.num_shards)
                tag_slug, request_id_slug = self._resolve_chunk_slugs(req_str, chunk)
                req = _ShardedRequestState(
                    tag_slug=tag_slug,
                    request_id_slug=request_id_slug,
                    shard_idx=shard_idx,
                    expected_keys=set(),
                )
                self._sharded_requests[req_str] = req

            shard_key = (req.tag_slug, req.shard_idx)
            shard = self._shard_states.get(shard_key)
            if shard is None:
                shard = _ShardState(tag_slug=req.tag_slug, shard_idx=req.shard_idx)
                self._shard_states[shard_key] = shard
            if shard.dtype is None:
                shard.dtype = dtype_str

            shard.entries.append(
                {
                    "request_id": req_str,
                    "layer": layer_idx,
                    "hook": hook_name,
                    "offset": shard.running_offset,
                    "nbytes": nbytes,
                    "shape": [rows, hidden],
                }
            )
            shard.running_offset += nbytes
            bin_name = shard_bin_name(shard.shard_idx, shard.seq, self._file_pp_rank)
            bin_path = self._root / req.tag_slug / bin_name
            req.shard_bins.add(str(bin_path))
            writer_key = (f"{req.tag_slug}#{shard.shard_idx}", shard.seq, _SHARD_HOOK)
            # Seal when the shard crosses the size threshold.
            if shard.running_offset >= self._params.shard_max_bytes:
                seal_task = self._build_shard_seal_locked(shard)

        self._writer.submit(
            WriteTask(path=bin_path, payload=tensor_bytes, append=True, key=writer_key)
        )
        if seal_task is not None:
            self._writer.submit(seal_task)

    def _build_shard_seal_locked(self, shard: _ShardState) -> FinalizeTask:
        """Build the seal ``FinalizeTask`` for ``shard`` and rotate it.

        Must be called with ``self._lock`` held. Snapshots the current
        entries into an index payload, returns the FinalizeTask (publishes
        ``shard-k-seq.bin`` + ``.json``), then bumps ``seq`` and resets the
        shard's running state so subsequent chunks open a fresh file.
        """
        tag_dir = self._root / shard.tag_slug
        bin_path = tag_dir / shard_bin_name(
            shard.shard_idx, shard.seq, self._file_pp_rank
        )
        sidecar_path = tag_dir / shard_index_name(
            shard.shard_idx, shard.seq, self._file_pp_rank
        )
        index_payload: dict[str, Any] = {
            "layout": "sharded",
            "shard_idx": shard.shard_idx,
            "seq": shard.seq,
            "dtype": shard.dtype or "float32",
            "entries": shard.entries,
        }
        # Under PP each stage seals its own shard files; record the stage.
        if self._file_pp_rank is not None:
            index_payload["pp_rank"] = self._file_pp_rank
            index_payload["pp_size"] = self._pp_size
        writer_key = (f"{shard.tag_slug}#{shard.shard_idx}", shard.seq, _SHARD_HOOK)
        task = FinalizeTask(
            bin_path=bin_path,
            sidecar_path=sidecar_path,
            sidecar_payload=index_payload,
            key=writer_key,
        )
        # Rotate: next chunks for this shard start a fresh file.
        shard.seq += 1
        shard.running_offset = 0
        shard.entries = []
        return task

    # ------------------------------------------------------------------
    # submit_finalize
    # ------------------------------------------------------------------

    def submit_finalize(self, finalize: CaptureFinalize) -> None:
        """Finalize one capture key (per_file) or accumulate toward the
        single packed-file finalize (packed)."""
        req_str = str(finalize.key[0])
        layout = self._layout_for(req_str)
        if layout == "packed":
            self._submit_finalize_packed(finalize, req_str)
        elif layout == "sharded":
            self._submit_finalize_sharded(finalize, req_str)
        else:
            self._submit_finalize_per_file(finalize, req_str)

    def _submit_finalize_per_file(
        self, finalize: CaptureFinalize, req_str: str
    ) -> None:
        key = finalize.key
        _request_id, layer_idx, hook_name = key

        with self._lock:
            path_info = self._key_paths.pop(key, None)
            shape_info = self._key_shapes.pop(key, None)
            dtype_str = self._key_dtype.pop(key, None)
            recorded = self._request_slugs.get(req_str)

        if path_info is not None:
            tag_slug, request_id_slug = path_info
        elif recorded is not None:
            tag_slug, request_id_slug = recorded
        else:
            tag_slug, request_id_slug = "default", req_str

        tag_slug = finalize.sidecar.get("tag_slug", tag_slug)
        request_id_slug = finalize.sidecar.get("request_id_slug", request_id_slug)

        bin_path = (
            self._root / tag_slug / request_id_slug / f"{layer_idx}_{hook_name}.bin"
        )
        sidecar_path = bin_path.with_suffix(".json")

        sidecar_payload: dict[str, Any] = {
            "request_id": req_str,
            "layer": layer_idx,
            "hook": hook_name,
        }
        sidecar_payload.update(finalize.sidecar)
        if shape_info is not None:
            sidecar_payload["shape"] = list(shape_info)
        if dtype_str is not None:
            sidecar_payload["dtype"] = dtype_str

        self._writer.submit(
            FinalizeTask(
                bin_path=bin_path,
                sidecar_path=sidecar_path,
                sidecar_payload=sidecar_payload,
                key=(req_str, layer_idx, hook_name),
            )
        )

    def _submit_finalize_packed(self, finalize: CaptureFinalize, req_str: str) -> None:
        _request_id, layer_idx, hook_name = finalize.key

        # Per-key finalizes arrive in one synchronous burst (post drain
        # barrier). Publish the single packed file only once every
        # expected (layer, hook) has been finalized.
        task: FinalizeTask | None = None
        with self._lock:
            state = self._packed_states.get(req_str)
            if state is None:
                return
            state.finalized.add((layer_idx, hook_name))
            state.sidecar_fields.update(finalize.sidecar)
            state.tag_slug = finalize.sidecar.get("tag_slug", state.tag_slug)
            state.request_id_slug = finalize.sidecar.get(
                "request_id_slug", state.request_id_slug
            )
            if not state.expected_keys.issubset(state.finalized):
                return
            # Last expected finalize — build the index and publish.
            self._packed_states.pop(req_str, None)
            index_payload: dict[str, Any] = dict(state.sidecar_fields)
            index_payload.update(
                {
                    "request_id": req_str,
                    "layout": "packed",
                    "dtype": state.dtype or "float32",
                    "entries": state.entries,
                }
            )
            # Under PP this index covers only this stage's layers; record
            # the stage so readers/operators can attribute the file.
            if self._file_pp_rank is not None:
                index_payload["pp_rank"] = self._file_pp_rank
                index_payload["pp_size"] = self._pp_size
            req_dir = self._root / state.tag_slug / state.request_id_slug
            task = FinalizeTask(
                bin_path=req_dir / self._packed_bin_name,
                sidecar_path=req_dir / self._packed_index_name,
                sidecar_payload=index_payload,
                key=(req_str, _PACKED_LAYER, _PACKED_HOOK),
            )

        if task is not None:
            self._writer.submit(task)

    def _submit_finalize_sharded(self, finalize: CaptureFinalize, req_str: str) -> None:
        _request_id, layer_idx, hook_name = finalize.key
        # The request's bytes are already in its shard (written during
        # submit_chunk). Finalize just tracks completion: once every
        # expected (layer, hook) has finalized, the request is "ok" — its
        # data lives in shard_bins and becomes readable when those shards
        # seal (size threshold or shutdown). No per-request file to write.
        completed = False
        with self._lock:
            req = self._sharded_requests.get(req_str)
            if req is None:
                return
            req.finalized.add((layer_idx, hook_name))
            if req.expected_keys.issubset(req.finalized) and not req.done:
                req.done = True
                completed = True
        if completed:
            with self._wait_lock:
                event = self._sharded_events.get(req_str)
            if event is not None:
                event.set()

    def get_result(self, key: CaptureKey) -> CaptureResult | None:
        """Map the writer's ``WriteResult`` to a ``CaptureResult``.

        For ``packed`` requests every ``(layer, hook)`` key maps to the
        request's single packed ``WriteResult``, so each key reports the
        packed file's status/paths. For ``sharded`` requests the result is
        consumer-tracked (the shard file is shared and seals async): every
        key reports ``ok`` once the request's keys have all finalized, with
        payload = the shard file(s) it landed in (readable after seal).
        """
        req_str = str(key[0])
        if self._layout_for(req_str) == "sharded":
            with self._lock:
                req = self._sharded_requests.get(req_str)
                if req is None or not req.done:
                    return None
                payload = sorted(req.shard_bins)
            return CaptureResult(key=key, status="ok", error=None, payload=payload)

        writer_key = self._writer_key_for(key)
        write_result = self._writer.get_result(writer_key)
        if write_result is None:
            return None

        error_str: str | None = None
        if write_result.error is not None:
            error_str = str(write_result.error)

        payload: list[str] | None = None
        if write_result.bin_path is not None:
            paths = [str(write_result.bin_path)]
            if write_result.sidecar_path is not None:
                paths.append(str(write_result.sidecar_path))
            payload = paths

        return CaptureResult(
            key=key,
            status=write_result.status,
            error=error_str,
            payload=payload,
        )

    def wait_for_result(
        self,
        key: CaptureKey,
        timeout: float,
    ) -> CaptureResult | None:
        """Block up to ``timeout`` seconds for the terminal result for ``key``.

        Uses a per-key :class:`threading.Event` that
        :meth:`_on_write_result` sets when the underlying
        ``ActivationWriter`` worker transitions the result to a terminal
        status.  If the result is already terminal on entry the method
        returns immediately without waiting.
        """
        req_str = str(key[0])
        if self._layout_for(req_str) == "sharded":
            # Sharded completion is consumer-tracked (set when the
            # request's last expected key finalizes), not a WriteResult.
            with self._wait_lock:
                with self._lock:
                    req = self._sharded_requests.get(req_str)
                    already_done = req is not None and req.done
                if already_done:
                    return self.get_result(key)
                event = self._sharded_events.setdefault(req_str, threading.Event())
            event.wait(timeout=timeout)
            return self.get_result(key)

        writer_key = self._writer_key_for(key)

        with self._wait_lock:
            wr = self._writer.get_result(writer_key)
            if wr is not None and wr.status in ("ok", "error"):
                return self.get_result(key)
            event = self._wait_events.setdefault(writer_key, threading.Event())

        event.wait(timeout=timeout)
        return self.get_result(key)

    def _on_write_result(self, write_result: Any) -> None:
        """Status callback fired by ``ActivationWriter`` on terminal transitions.

        Called outside ``ActivationWriter._results_lock`` (see the
        ``fired`` list pattern in ``_record_ok`` / ``_record_error``).
        Sets the per-key event so any thread blocked in
        :meth:`wait_for_result` can wake up.
        """
        if write_result.status in ("ok", "error"):
            with self._wait_lock:
                event = self._wait_events.pop(write_result.key, None)
            if event is not None:
                event.set()

    def shutdown(self, timeout: float = 30.0) -> None:
        """Seal any open shards, then drain the underlying writer.

        Open ``sharded`` shards holding unsealed data are published now so
        their captures become readable; otherwise a partial final shard
        would never get an index. per_file/packed have nothing to seal.
        """
        seal_tasks: list[FinalizeTask] = []
        with self._lock:
            for shard in self._shard_states.values():
                if shard.entries:
                    seal_tasks.append(self._build_shard_seal_locked(shard))
        for task in seal_tasks:
            try:
                self._writer.submit(task)
            except Exception:
                logger.exception("failed to submit shard seal at shutdown")
        self._writer.shutdown(timeout=timeout)
