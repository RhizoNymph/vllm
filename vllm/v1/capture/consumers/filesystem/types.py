# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Types for the filesystem capture consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FilesystemCaptureRequest:
    """Per-request client spec for the filesystem consumer.

    This is the raw shape that arrives via
    ``SamplingParams.capture["filesystem"]``. The consumer's
    ``validate_client_spec`` converts it into a ``CaptureSpec`` after
    delegating to the existing activation-storing admission validator.
    """

    request_id: str
    tag: str
    hooks: dict[str, Any]
    positions: str | list[int]
    # On-disk layout for this request's captures:
    #   "per_file" — one .bin + .json per (layer, hook); low latency,
    #                supports mid-request streaming. (current behavior)
    #   "packed"   — one .bin + one .json index per request, with all
    #                (layer, hook) tensors concatenated; far fewer files
    #                (the throughput win on network mounts), available
    #                once the request finalizes.
    # ``None`` defers to the consumer-level ``default_layout``.
    layout: str | None = None


@dataclass
class FilesystemConsumerParams:
    """Consumer-level configuration parsed from ``params`` dict.

    Mirrors the ``ActivationWriter`` constructor arguments so the
    consumer can forward them without loss.
    """

    root: str
    writer_threads: int = 4
    queue_size: int = 1024
    timeout_seconds: float = 180.0
    on_collision: str = "overwrite"
    fd_cache_size: int = 256
    # fsync each file before its atomic rename. Durable but, on network
    # filesystems, the dominant finalize cost (a synchronous server
    # round-trip per file). Set False to trade crash-durability for
    # throughput; the atomic rename still gives readers complete files.
    fsync: bool = True
    # Publish via .tmp + atomic rename (True) or write straight to the
    # final path (False). False drops two rename metadata RPCs per
    # capture — the main small-capture throughput lever on NFS — at the
    # cost of atomic visibility. Requires on_collision='overwrite'.
    atomic_publish: bool = True
    # Default on-disk layout for requests that don't set their own
    # ``layout`` ("per_file" or "packed"). See FilesystemCaptureRequest.
    default_layout: str = "per_file"
    # Merge consecutive same-key queued writes into one vectored write,
    # up to this many bytes. Amortizes per-write syscall/RTT overhead —
    # most effective for the ``packed`` layout (many small per-step
    # appends share one file). 0 disables.
    coalesce_max_bytes: int = 1 << 20


# Valid values for FilesystemCaptureRequest.layout / default_layout.
VALID_LAYOUTS: frozenset[str] = frozenset(("per_file", "packed"))

# Filenames used by the "packed" layout (one set per request directory).
PACKED_BIN_NAME = "packed.bin"
PACKED_INDEX_NAME = "packed.json"
