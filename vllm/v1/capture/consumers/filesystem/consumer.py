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
    PACKED_BIN_NAME,
    PACKED_INDEX_NAME,
    VALID_LAYOUTS,
    FilesystemCaptureRequest,
    FilesystemConsumerParams,
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
        timeout_seconds=float(params.get("timeout_seconds", 180.0)),
        on_collision=str(params.get("on_collision", "overwrite")),
        fd_cache_size=int(params.get("fd_cache_size", 256)),
        fsync=bool(params.get("fsync", True)),
        atomic_publish=bool(params.get("atomic_publish", True)),
        default_layout=str(params.get("default_layout", "per_file")),
    )


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
        self._root = pathlib.Path(self._params.root)
        self._writer = ActivationWriter(
            root=self._root,
            num_threads=self._params.writer_threads,
            queue_size=self._params.queue_size,
            timeout_seconds=self._params.timeout_seconds,
            on_collision=self._params.on_collision,  # type: ignore[arg-type]
            fd_cache_size=self._params.fd_cache_size,
            fsync=self._params.fsync,
            atomic_publish=self._params.atomic_publish,
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

        req_str = str(ctx.vllm_internal_request_id)
        with self._lock:
            self._request_slugs[req_str] = (tag_slug, request_id_slug)
            self._request_layout[req_str] = layout
            if layout == "packed":
                # The full (layer, hook) set we must see finalized
                # before the single packed file can be published.
                expected = {
                    (layer_idx, hook_name)
                    for hook_name, layers in spec.hooks.items()
                    for layer_idx in layers
                }
                self._packed_states[req_str] = _PackedState(
                    tag_slug=tag_slug,
                    request_id_slug=request_id_slug,
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
        if self._layout_for(req_str) == "packed":
            self._submit_chunk_packed(chunk, req_str)
        else:
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

        bin_path = self._root / bin_tag / bin_req / PACKED_BIN_NAME
        self._writer.submit(
            WriteTask(
                path=bin_path,
                payload=tensor_bytes,
                append=True,
                key=(req_str, _PACKED_LAYER, _PACKED_HOOK),
            )
        )

    # ------------------------------------------------------------------
    # submit_finalize
    # ------------------------------------------------------------------

    def submit_finalize(self, finalize: CaptureFinalize) -> None:
        """Finalize one capture key (per_file) or accumulate toward the
        single packed-file finalize (packed)."""
        req_str = str(finalize.key[0])
        if self._layout_for(req_str) == "packed":
            self._submit_finalize_packed(finalize, req_str)
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
            req_dir = self._root / state.tag_slug / state.request_id_slug
            task = FinalizeTask(
                bin_path=req_dir / PACKED_BIN_NAME,
                sidecar_path=req_dir / PACKED_INDEX_NAME,
                sidecar_payload=index_payload,
                key=(req_str, _PACKED_LAYER, _PACKED_HOOK),
            )

        if task is not None:
            self._writer.submit(task)

    def get_result(self, key: CaptureKey) -> CaptureResult | None:
        """Map the writer's ``WriteResult`` to a ``CaptureResult``.

        For ``packed`` requests every ``(layer, hook)`` key maps to the
        request's single packed ``WriteResult``, so each key reports the
        packed file's status/paths.
        """
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
        """Forward shutdown to the underlying ``ActivationWriter``."""
        self._writer.shutdown(timeout=timeout)
