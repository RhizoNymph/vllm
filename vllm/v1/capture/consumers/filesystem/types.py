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
    #   "sharded"  — many requests' captures share a small set of large
    #                shard files (per tag), sealed by size/at shutdown.
    #                Fewest files for the many-tiny-requests case; a
    #                capture is readable only after its shard seals
    #                (end-of-run / bulk reader model).
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
    # How long ``submit`` blocks on a full writer queue before raising. Short
    # so a wedged writer surfaces quickly rather than stalling for minutes;
    # the dispatch-queue overload policy is the primary backpressure path.
    timeout_seconds: float = 30.0
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
    # Optional GLOBAL capture spec: when set, EVERY request is captured
    # without per-request opt-in (dedicated batch-capture servers). String
    # format is CLI-shorthand-safe (no commas):
    # ``"<hook>:<layers>[;<hook>:<layers>]"`` where ``<layers>`` is
    # ``all`` | ``<a>-<b>`` (inclusive) | ``<i>.<j>.<k>`` (dot-separated).
    # e.g. ``global_hooks=post_block:all`` or ``pre_attn:0-17;post_block:20``.
    global_hooks: str | None = None
    # Position selector for the global spec (default "all_prompt").
    global_positions: str = "all_prompt"
    # Merge consecutive same-key queued writes into one vectored write,
    # up to this many bytes. Amortizes per-write syscall/RTT overhead —
    # most effective for the ``packed`` layout (many small per-step
    # appends share one file). 0 disables.
    coalesce_max_bytes: int = 1 << 20
    # "sharded" layout: number of shard files per tag (requests assigned
    # by hash(request_id) % num_shards) and the size threshold at which
    # an open shard is sealed (published) and a new one started.
    num_shards: int = 8
    shard_max_bytes: int = 256 << 20  # 256 MiB


# Valid values for FilesystemCaptureRequest.layout / default_layout.
VALID_LAYOUTS: frozenset[str] = frozenset(("per_file", "packed", "sharded"))

# Filenames used by the "packed" layout (one set per request directory).
# These are the pipeline-parallel-agnostic names used when pp_size == 1.
# Under pipeline parallelism each stage owns a disjoint slice of the
# request's layers and writes its own file (``packed-pp{rank}.*``), so the
# stages never race for the same path on the shared mount.
PACKED_BIN_NAME = "packed.bin"
PACKED_INDEX_NAME = "packed.json"


def packed_bin_name(pp_rank: int | None = None) -> str:
    """``.bin`` filename for a request's packed file.

    ``pp_rank is None`` (pipeline parallelism disabled) keeps the legacy
    ``packed.bin``; under PP each stage writes ``packed-pp{rank}.bin`` so
    per-stage files don't collide in the shared request directory.
    """
    if pp_rank is None:
        return PACKED_BIN_NAME
    return f"packed-pp{pp_rank:02d}.bin"


def packed_index_name(pp_rank: int | None = None) -> str:
    """``.json`` index filename for a request's packed file (see
    :func:`packed_bin_name`)."""
    if pp_rank is None:
        return PACKED_INDEX_NAME
    return f"packed-pp{pp_rank:02d}.json"


def shard_bin_name(shard_idx: int, seq: int, pp_rank: int | None = None) -> str:
    """``.bin`` filename for shard ``shard_idx`` rotation ``seq``.

    Under pipeline parallelism (``pp_rank is not None``) the stage rank is
    embedded so each stage's shards land in distinct files within the tag
    directory; the ``shard-*`` prefix is preserved so the reader's glob
    still discovers them.
    """
    if pp_rank is None:
        return f"shard-{shard_idx:03d}-{seq:06d}.bin"
    return f"shard-pp{pp_rank:02d}-{shard_idx:03d}-{seq:06d}.bin"


def shard_index_name(shard_idx: int, seq: int, pp_rank: int | None = None) -> str:
    """``.json`` index filename for shard ``shard_idx`` rotation ``seq``
    (see :func:`shard_bin_name`)."""
    if pp_rank is None:
        return f"shard-{shard_idx:03d}-{seq:06d}.json"
    return f"shard-pp{pp_rank:02d}-{shard_idx:03d}-{seq:06d}.json"


# Glob to find sealed shard index files in a tag directory. Matches both
# the pp-agnostic ``shard-NNN-NNNNNN.json`` and the per-stage
# ``shard-ppNN-NNN-NNNNNN.json`` produced under pipeline parallelism.
SHARD_INDEX_GLOB = "shard-*.json"

# Glob to find a request's packed index files. Matches the pp-agnostic
# ``packed.json`` and the per-stage ``packed-ppNN.json``.
PACKED_INDEX_GLOB = "packed*.json"
