# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``CaptureSink`` protocol — the low-level streaming interface
between the capture manager and a consumer.

Most consumer authors should subclass ``CaptureConsumer`` (which
fulfills ``CaptureSink`` via an internal batched adapter) instead of
implementing this protocol directly. Implement ``CaptureSink`` directly
only when you need one of:

- True streaming semantics — writing rows incrementally without
  buffering the full tensor (this is what the filesystem consumer
  does for long captures).
- Chunk-level visibility into ``row_offset`` / ``step_index`` beyond
  what the batched consumer exposes.
- A custom threading / concurrency model.

See ``docs/design/capture_consumers.md`` § "Sinks and Consumers".
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureFinalize,
    CaptureKey,
    CaptureResult,
)


@runtime_checkable
class CaptureSink(Protocol):
    """Low-level streaming interface between the capture manager and a
    consumer.

    Ordering guarantees from the manager side:

    - For a given ``CaptureKey``, chunks arrive in ``row_offset`` order.
    - ``CaptureFinalize`` for a key arrives after all chunks for that
      key.
    - Different keys have no ordering relationship.

    All methods must be thread-safe: the manager may call the ``submit_*``
    methods from any worker thread, and the engine output processor may
    call ``get_result`` concurrently from the engine-core thread.
    """

    location: Literal["worker", "driver"]

    def submit_chunk(self, chunk: CaptureChunk) -> None:
        """Non-blocking enqueue of a chunk of captured rows."""
        ...

    def submit_chunk_batch(self, chunks: list[CaptureChunk]) -> None:
        """Optional: enqueue one dispatch step's worth of chunks at once.

        The manager hands every chunk it produced for this sink in a single
        step to this method in one call. Sinks that can amortize per-chunk
        overhead across the batch — locking, write-task creation, payload
        concatenation — should override this; the default just forwards each
        chunk to ``submit_chunk`` so the manager can always call it.

        The manager invokes this opportunistically (via ``getattr``); sinks
        that do not define it keep working through the per-chunk path.
        """
        for chunk in chunks:
            self.submit_chunk(chunk)

    def submit_finalize(self, finalize: CaptureFinalize) -> None:
        """Non-blocking request-completion signal.

        On receipt the sink should flush any buffered state for the
        ``CaptureKey`` and transition ``get_result(key)`` to a terminal
        ``CaptureResult``.
        """
        ...

    def get_result(self, key: CaptureKey) -> CaptureResult | None:
        """Return the terminal result for ``key``, or ``None`` if the
        sink has not yet finalized that key."""
        ...

    def wait_for_result(
        self,
        key: CaptureKey,
        timeout: float,
    ) -> CaptureResult | None:
        """Block up to ``timeout`` seconds for the terminal result for
        ``key``.

        Returns ``None`` if the result is still unavailable when the
        timeout expires.
        """
        ...

    def shutdown(self, timeout: float = 30.0) -> None:
        """Drain in-flight work with a bounded grace period."""
        ...
