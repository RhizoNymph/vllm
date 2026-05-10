# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-memory IPC region for inline steering vectors.

The packed inline-steering path (see
:func:`vllm.config.steering_types.maybe_pack_inline_steering_for_request`)
serializes ``np.ndarray`` blobs through msgpack to ship the request from
the API client / ``LLM`` driver to the engine-core worker process.  Even
the bf16-truncated form copies the full vector bytes through ZMQ for
every request submission, which costs measurable CPU time once the
vectors get large (e.g. 34 layers × 2560 d_model ≈ 174 KB / request at
fp32).

This module replaces the wire-payload memcpy with a shared-memory
hand-off:

- The engine instantiates :class:`SteeringShmRegion` at startup.  The
  region creates a file in ``/dev/shm`` keyed by the engine UUID, mmaps
  it read-write, and registers an :mod:`atexit` handler to unlink the
  file on engine teardown.
- Before submitting a request the client calls
  :meth:`SteeringShmRegion.write_packed`, which copies the packed bytes
  into the bump-pointer arena and returns a
  ``(offset, length, dtype, shape)`` tuple.  Only that ~32-byte tuple
  ships over ZMQ.
- The worker mmaps the same path read-only on first use and decodes
  the ndarray on demand via :meth:`SteeringShmRegion.read_packed`.

The bump-pointer allocator resets to offset 0 only when no in-flight
request still references the current generation; an in-flight refcount
guards against torn reads.

The class is Linux-only (``/dev/shm`` is a Linux-specific tmpfs mount).
On non-Linux platforms construction fails soft via :meth:`maybe_create`,
which returns ``None`` and lets callers fall back to the existing
inline-packed wire path.
"""

from __future__ import annotations

import atexit
import mmap
import os
import sys
import threading
import time
import uuid
import weakref

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)

# Default 512 MiB region.  Sized to comfortably hold a few thousand
# in-flight packed vectors at typical d_model — the bump cursor resets
# back to 0 between request batches once nothing references the current
# generation.
_DEFAULT_REGION_BYTES = 512 * 1024 * 1024


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """Spin-wait until *fd*'s file reaches *expected_size* bytes."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for steering shm file to reach "
                f"{expected_size} bytes"
            )
        time.sleep(0.005)


class SteeringShmRegion:
    """Bump-pointer mmap region for packed steering vectors.

    Construction creates a ``/dev/shm/vllm_steering_<pid>_<engine_uuid>.mmap``
    file (default 512 MiB) and mmaps it read-write.  Calls to
    :meth:`write_packed` copy a packed ``np.ndarray`` into the next free
    slot and return a descriptor tuple ``(offset, length, dtype_str,
    shape)`` that fully describes how to read it back via
    :meth:`read_packed`.

    Reset semantics: the bump cursor resets to 0 only when the in-flight
    request refcount drops to zero — see :meth:`request_started` /
    :meth:`request_finished`.  Callers that don't bother to track
    in-flight requests still get correctness, just no reuse of earlier
    bytes within the same allocator generation.

    Read-side workers should construct via :meth:`open_readonly` to get
    a ``PROT_READ`` mapping of the same path.
    """

    def __init__(
        self,
        engine_uuid: str | None = None,
        total_size_bytes: int = _DEFAULT_REGION_BYTES,
        path: str | None = None,
    ) -> None:
        self.engine_uuid = engine_uuid or uuid.uuid4().hex
        self.total_size_bytes = int(total_size_bytes)
        self.mmap_path = path or (
            f"/dev/shm/vllm_steering_{os.getpid()}_{self.engine_uuid}.mmap"
        )
        self._creator = False
        self.fd: int | None = None
        self.mmap_obj: mmap.mmap | None = None

        # Bump-pointer cursor protected by a lock so concurrent
        # ``write_packed`` calls from multiple submission threads (HTTP
        # server, LLM batch driver) don't corrupt each other's
        # allocations.  Cursor advances monotonically until the in-flight
        # refcount drops to 0, at which point :meth:`_maybe_reset` resets
        # it.
        self._cursor = 0
        self._inflight = 0
        self._lock = threading.Lock()
        self._closed = False

        try:
            self.fd = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
            os.ftruncate(self.fd, self.total_size_bytes)
            self._creator = True
            logger.debug(
                "Created steering shm file %s (%.1f MiB)",
                self.mmap_path,
                self.total_size_bytes / (1024 * 1024),
            )
        except FileExistsError:
            self.fd = os.open(self.mmap_path, os.O_RDWR)
            _wait_for_file_size(self.fd, self.total_size_bytes)
            logger.debug("Opened existing steering shm file %s", self.mmap_path)

        self.mmap_obj = mmap.mmap(
            self.fd,
            self.total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        # Register cleanup via a weakref-bound atexit hook so the file
        # gets unlinked on interpreter shutdown even if the caller
        # forgets to call :meth:`close`.  The weakref avoids extending
        # the region's lifetime past the caller's last reference.
        weak_self = weakref.ref(self)

        def _atexit_cleanup() -> None:
            inst = weak_self()
            if inst is not None:
                inst.close()

        atexit.register(_atexit_cleanup)
        self._atexit_cleanup = _atexit_cleanup

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def maybe_create(
        cls,
        engine_uuid: str | None = None,
        total_size_bytes: int = _DEFAULT_REGION_BYTES,
    ) -> SteeringShmRegion | None:
        """Create a region, returning ``None`` on platforms that don't
        support ``/dev/shm`` or when shm is otherwise unavailable.

        Used by engine init paths that want fail-soft fallback to the
        existing inline-packed wire path on non-Linux deployments.
        """
        if not _is_linux():
            logger.debug(
                "SteeringShmRegion: /dev/shm is Linux-only; skipping shm path"
            )
            return None
        try:
            return cls(
                engine_uuid=engine_uuid,
                total_size_bytes=total_size_bytes,
            )
        except OSError as exc:
            logger.warning(
                "SteeringShmRegion: failed to create shm region (%s); "
                "falling back to inline-packed wire path",
                exc,
            )
            return None

    @classmethod
    def open_readonly(cls, path: str) -> SteeringShmRegion:
        """Open an existing shm file in PROT_READ mode.

        Used by worker processes that received the path via
        ``vllm_config`` (or environment) and need to materialize
        ndarrays from offset descriptors.
        """
        # Bypass the writable constructor entirely.
        inst = cls.__new__(cls)
        inst.engine_uuid = ""
        inst.mmap_path = path
        inst._creator = False
        inst._cursor = 0
        inst._inflight = 0
        inst._lock = threading.Lock()
        inst._closed = False

        inst.fd = os.open(path, os.O_RDONLY)
        size = os.fstat(inst.fd).st_size
        inst.total_size_bytes = size
        inst.mmap_obj = mmap.mmap(
            inst.fd,
            size,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ,
        )
        inst._atexit_cleanup = None
        return inst

    # ------------------------------------------------------------------
    # Allocator API
    # ------------------------------------------------------------------

    def request_started(self) -> int:
        """Bump the in-flight refcount.

        Callers that produce descriptors via :meth:`write_packed` should
        call this once at request submission and pair it with
        :meth:`request_finished` once the worker is done with the
        descriptors (typically when the request completes).  Returns the
        current generation id; callers may stash it to detect
        post-reset descriptor reuse, although the bump-pointer
        invariants make that unnecessary in practice.
        """
        with self._lock:
            self._inflight += 1
            return self._cursor  # opaque; not actually consumed today

    def request_finished(self) -> None:
        """Drop the in-flight refcount; reset cursor when it hits 0."""
        with self._lock:
            if self._inflight > 0:
                self._inflight -= 1
            if self._inflight == 0:
                self._cursor = 0

    def write_packed(
        self, arr: np.ndarray
    ) -> tuple[int, int, str, tuple[int, ...]]:
        """Copy *arr* into the region.  Returns its descriptor.

        Returns ``(offset, length, dtype_str, shape)`` where:
        - ``offset`` is the byte offset within the region.
        - ``length`` is ``arr.nbytes``.
        - ``dtype_str`` is ``arr.dtype.str`` (e.g. ``"<f4"``).
        - ``shape`` is ``arr.shape`` as a tuple of ints.

        The descriptor uniquely identifies the bytes — the worker can
        reconstruct the ndarray via
        ``np.frombuffer(...).reshape(shape)``.
        """
        if self._closed or self.mmap_obj is None:
            raise RuntimeError("SteeringShmRegion is closed")

        # ``np.ascontiguousarray`` is a no-op when *arr* is already
        # C-contiguous; otherwise it allocates a contiguous copy so the
        # raw byte layout matches ``arr.shape`` order.
        contig = np.ascontiguousarray(arr)
        nbytes = contig.nbytes
        dtype_str = contig.dtype.str
        shape = tuple(int(d) for d in contig.shape)

        with self._lock:
            offset = self._cursor
            new_cursor = offset + nbytes
            if new_cursor > self.total_size_bytes:
                raise RuntimeError(
                    f"SteeringShmRegion overflow: trying to write "
                    f"{nbytes} bytes at offset {offset} exceeds "
                    f"region size {self.total_size_bytes}"
                )
            # Write under the lock so the cursor advance is atomic with
            # the byte copy — concurrent writers can't tear each other's
            # payloads.
            self.mmap_obj[offset:new_cursor] = contig.tobytes()
            self._cursor = new_cursor

        return (offset, nbytes, dtype_str, shape)

    def read_packed(
        self,
        offset: int,
        length: int,
        dtype_str: str,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        """Materialize an ndarray from the region.

        The returned ndarray aliases the mmap (no extra copy).  Callers
        that need to mutate it or hold it past the next reset should
        ``.copy()`` first; in practice the worker reads and then immediately
        consumes via ``register_config``, so aliasing is safe.
        """
        if self.mmap_obj is None:
            raise RuntimeError("SteeringShmRegion is closed")
        end = offset + length
        if offset < 0 or end > self.total_size_bytes:
            raise ValueError(
                f"SteeringShmRegion read out of bounds: "
                f"offset={offset} length={length} region={self.total_size_bytes}"
            )
        dtype = np.dtype(dtype_str)
        # ``np.frombuffer`` zero-copies over the underlying buffer; use a
        # ``memoryview`` slice so the resulting ndarray sees only the
        # requested window.  ``copy()`` is intentionally avoided — the
        # worker consumes the bytes immediately during request resolve
        # and the array doesn't outlive a single resolver call.
        view = memoryview(self.mmap_obj)[offset:end]
        arr = np.frombuffer(view, dtype=dtype)
        return arr.reshape(shape)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the mmap and (if creator) unlink the backing file."""
        if self._closed:
            return
        self._closed = True
        if self.mmap_obj is not None:
            try:
                self.mmap_obj.close()
            except (BufferError, ValueError):
                # Outstanding views (typically from ``read_packed`` on the
                # writer side during teardown) prevent close.  Drop the
                # reference and let GC release the mapping.
                pass
            self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        if self._creator and self.mmap_path:
            try:
                os.unlink(self.mmap_path)
                logger.debug("Removed steering shm file %s", self.mmap_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning(
                    "Failed to unlink steering shm file %s",
                    self.mmap_path,
                    exc_info=True,
                )
            self._creator = False

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Process-local handles
# ---------------------------------------------------------------------------
#
# The cached_properties on :class:`vllm.sampling_params.SamplingParams`
# need to read packed bytes back out of shm when a request reaches the
# worker.  We can't ship the :class:`SteeringShmRegion` instance through
# msgspec (it's a stateful resource owning fds), so instead each side
# of the IPC boundary registers a *process-local* handle that
# ``effective_*_steering`` can look up by side-effect.  Two slots are
# kept independent so the same process can act as both the client (for
# in-process testing) and the worker (for engine-core spawned procs).

_CLIENT_REGION: SteeringShmRegion | None = None
_WORKER_REGION: SteeringShmRegion | None = None


def set_client_region(region: SteeringShmRegion | None) -> None:
    """Register the writer-side region for this process.

    Called by :class:`vllm.entrypoints.LLM` and
    :class:`vllm.v1.engine.async_llm.AsyncLLM` at engine init.  Setting
    to ``None`` clears the slot — used by tests to isolate state across
    cases.
    """
    global _CLIENT_REGION
    _CLIENT_REGION = region


def get_client_region() -> SteeringShmRegion | None:
    return _CLIENT_REGION


def set_worker_region(region: SteeringShmRegion | None) -> None:
    """Register the reader-side region for this process.

    Called lazily from :class:`SteeringModelRunnerMixin` the first time
    it sees a SamplingParams whose ``_effective_*_steering_shm`` field
    is populated.
    """
    global _WORKER_REGION
    _WORKER_REGION = region


def get_worker_region() -> SteeringShmRegion | None:
    return _WORKER_REGION


def get_any_region() -> SteeringShmRegion | None:
    """Return whichever region is registered for this process.

    The cached_property on ``SamplingParams`` calls this to materialize
    descriptor tuples into ndarrays.  Worker-side wins when both are set
    (i.e. in the process that owns the worker-side mmap).
    """
    return _WORKER_REGION or _CLIENT_REGION


def materialize_shm_dict(
    shm_dict: dict[str, dict[int, tuple[int, int, str, tuple[int, ...]]]],
    region: SteeringShmRegion,
) -> dict[str, dict[int, np.ndarray]]:
    """Resolve every descriptor tuple in *shm_dict* into an ndarray."""
    out: dict[str, dict[int, np.ndarray]] = {}
    for hook, layer_dict in shm_dict.items():
        layer_arrs: dict[int, np.ndarray] = {}
        for layer_idx, descriptor in layer_dict.items():
            offset, length, dtype_str, shape = descriptor
            layer_arrs[layer_idx] = region.read_packed(
                offset, length, dtype_str, shape
            )
        if layer_arrs:
            out[hook] = layer_arrs
    return out
