# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference reader for filesystem-captured activations.

Reads both on-disk layouts produced by :class:`FilesystemConsumer` and
returns the captured tensors as NumPy arrays. NumPy-only (no torch) so
offline analysis scripts can import it in a lightweight environment.

Layouts
-------
``per_file`` (one file per ``(layer, hook)``)::

    {root}/{tag}/{request}/{layer}_{hook}.bin    raw bytes, residual dtype
    {root}/{tag}/{request}/{layer}_{hook}.json   {request_id, layer, hook,
                                                  shape, dtype, ...}

``packed`` (one file per request, all tensors concatenated)::

    {root}/{tag}/{request}/packed.bin            raw bytes, concatenated
    {root}/{tag}/{request}/packed.json           {request_id, layout:"packed",
                                                  dtype, entries:[
                                                    {layer, hook, offset,
                                                     nbytes, shape}, ...]}

Both ``.bin`` files use the same byte encoding as the original format:
raw little-endian bytes in the model's residual dtype, with ``bfloat16``
stored as raw ``uint16`` (NumPy has no native bf16). Such captures are
returned here as ``uint16`` arrays; recover bf16 with
``torch.from_numpy(arr).view(torch.bfloat16)``.

Hooks may differ in width and dtype within one request (DeepSeek-V4 mHC:
bf16 multi-stream residual + fp32 mixing coefficients). Two optional,
self-describing fields carry that: a per-row ``row_shape`` (the logical
shape each flat ``(rows, width)`` row reshapes back to, e.g.
``[hc_mult, hidden]``) and, in ``packed`` / ``sharded`` entries, a
per-entry ``dtype`` overriding the index-level default. Both are absent
for standard residual captures, which decode exactly as before.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

from vllm.v1.capture.consumers.filesystem.types import (
    PACKED_INDEX_GLOB,
    PACKED_INDEX_NAME,
    SHARD_INDEX_GLOB,
)

# Logical dtype string (as recorded in the sidecar) -> NumPy dtype used to
# interpret the on-disk bytes. bfloat16 has no NumPy equivalent, so its
# bytes are read as uint16 (their on-disk representation).
_DTYPE_TO_NUMPY: dict[str, str] = {
    "float64": "float64",
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "uint16",
    "uint16": "uint16",
    "int8": "int8",
    "uint8": "uint8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
}

# Used when a per_file sidecar predates the self-describing ``dtype`` field.
_DEFAULT_DTYPE = "float32"


@dataclass
class CaptureEntry:
    """One captured ``(layer, hook)`` tensor, decoded to NumPy."""

    layer: int
    hook: str
    array: np.ndarray  # shape (rows, hidden); bf16 captures come back uint16
    dtype: str  # logical dtype string from the sidecar (e.g. "bfloat16")
    # Absolute logical token position of each row, when the sidecar records
    # it. ``None`` for older captures. Under speculative decoding a generated
    # position can appear in several rows (a rejected draft and the accepted
    # re-forward); rows are in write order, so the last row for a position is
    # canonical. Use :func:`latest_per_position` to collapse to one row each.
    positions: list[int] | None = None


def _np_dtype(logical: str) -> np.dtype:
    try:
        return np.dtype(_DTYPE_TO_NUMPY[logical])
    except KeyError as exc:
        raise ValueError(
            f"unknown capture dtype {logical!r}; known: {sorted(_DTYPE_TO_NUMPY)}"
        ) from exc


def _decode(buf: bytes, shape: list[int], logical_dtype: str) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=_np_dtype(logical_dtype))
    return arr.reshape(shape)


def _decode_shape(shape: list[int], row_shape: list[int] | None) -> list[int]:
    """Target reshape for a captured tensor.

    ``shape`` is ``[total_rows, width]``. When the sidecar records a
    per-row logical ``row_shape`` (e.g. ``[hc_mult, hidden]`` for an mHC
    stream hook), reshape to ``[total_rows, *row_shape]`` since
    ``prod(row_shape) == width``. Without it, the flat ``[rows, width]``
    shape is preserved (standard residual hooks)."""
    if row_shape:
        return [shape[0], *(int(x) for x in row_shape)]
    return shape


def _merge_positions(parts: list[list[int] | None]) -> list[int] | None:
    """Concatenate per-chunk position lists in (already-sorted) part order.

    Returns ``None`` if any part lacks positions, so a capture is reported
    as position-labelled only when every row is labelled."""
    if any(p is None for p in parts):
        return None
    merged: list[int] = []
    for p in parts:
        merged.extend(int(x) for x in p)  # type: ignore[union-attr]
    return merged


def read_per_file(bin_path: str | pathlib.Path) -> CaptureEntry:
    """Read a single ``per_file`` capture (``{layer}_{hook}.bin`` + sidecar)."""
    bin_path = pathlib.Path(bin_path)
    sidecar_path = bin_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    dtype = sidecar.get("dtype", _DEFAULT_DTYPE)
    shape = _decode_shape(sidecar["shape"], sidecar.get("row_shape"))
    array = _decode(bin_path.read_bytes(), shape, dtype)
    positions = sidecar.get("positions")
    return CaptureEntry(
        layer=int(sidecar["layer"]),
        hook=str(sidecar["hook"]),
        array=array,
        dtype=dtype,
        positions=[int(p) for p in positions] if positions is not None else None,
    )


def _read_one_packed_index(
    index_path: pathlib.Path,
) -> dict[tuple[int, str], CaptureEntry]:
    """Decode a single packed index file (``packed*.json``) + its ``.bin``."""
    index = json.loads(index_path.read_text())
    index_dtype = index["dtype"]
    bin_path = index_path.with_suffix(".bin")
    raw = bin_path.read_bytes()

    # A (layer, hook) capture may span several entries — one per
    # submitted chunk (decode step) — because chunks for different keys
    # interleave in submission order. Group by key and concatenate the
    # chunk arrays in byte-offset order to recover the full tensor.
    # Per-entry ``dtype`` is authoritative (a request may pack hooks of
    # different dtypes); the index-level dtype is the fallback default.
    grouped: dict[
        tuple[int, str], tuple[str, list[tuple[int, np.ndarray, list[int] | None]]]
    ] = {}
    for entry in index["entries"]:
        offset = int(entry["offset"])
        nbytes = int(entry["nbytes"])
        chunk = raw[offset : offset + nbytes]
        if len(chunk) != nbytes:
            raise ValueError(
                f"packed bin {bin_path} truncated: entry "
                f"({entry['layer']}, {entry['hook']}) wants bytes "
                f"[{offset}:{offset + nbytes}] but file is {len(raw)} bytes"
            )
        layer, hook = int(entry["layer"]), str(entry["hook"])
        entry_dtype = entry.get("dtype", index_dtype)
        shape = _decode_shape(entry["shape"], entry.get("row_shape"))
        grouped.setdefault((layer, hook), (entry_dtype, []))[1].append(
            (offset, _decode(chunk, shape, entry_dtype), entry.get("positions"))
        )

    out: dict[tuple[int, str], CaptureEntry] = {}
    for (layer, hook), (entry_dtype, parts) in grouped.items():
        parts.sort(key=lambda p: p[0])
        arrays = [a for _, a, _ in parts]
        array = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
        out[(layer, hook)] = CaptureEntry(
            layer=layer,
            hook=hook,
            array=array,
            dtype=entry_dtype,
            positions=_merge_positions([p for _, _, p in parts]),
        )
    return out


def read_packed(
    path: str | pathlib.Path,
) -> dict[tuple[int, str], CaptureEntry]:
    """Read a ``packed`` capture, keyed by ``(layer, hook)``.

    ``path`` may be a packed index file, a packed bin file, or the request
    directory. Under pipeline parallelism a request directory holds one
    ``packed-pp{rank}.json``/``.bin`` pair per stage (each over the layers
    that stage owns); pointing ``read_packed`` at the directory merges all
    of them into the request's full layer set. Pointing it at a single
    index/bin reads just that file. With PP off the directory holds the
    single legacy ``packed.json``/``packed.bin`` pair.
    """
    path = pathlib.Path(path)
    if path.is_dir():
        index_paths = sorted(path.glob(PACKED_INDEX_GLOB))
        if not index_paths:
            # Preserve the legacy "missing file" error surface.
            index_paths = [path / PACKED_INDEX_NAME]
    elif path.suffix == ".bin":
        index_paths = [path.with_suffix(".json")]
    else:
        index_paths = [path]

    out: dict[tuple[int, str], CaptureEntry] = {}
    for index_path in index_paths:
        out.update(_read_one_packed_index(index_path))
    return out


def read_request(
    request_dir: str | pathlib.Path,
) -> dict[tuple[int, str], CaptureEntry]:
    """Read every capture for a request, auto-detecting the layout.

    ``packed`` if any ``packed*.json`` index is present (one per pipeline
    stage under PP, else the single ``packed.json``), in which case the
    per-stage files are merged. Otherwise ``per_file`` (one entry per
    ``{layer}_{hook}.bin``).
    """
    request_dir = pathlib.Path(request_dir)
    if any(request_dir.glob(PACKED_INDEX_GLOB)):
        return read_packed(request_dir)
    out: dict[tuple[int, str], CaptureEntry] = {}
    for bin_path in sorted(request_dir.glob("*.bin")):
        # Skip packed bins (``packed.bin`` / ``packed-pp{rank}.bin``); a
        # per_file capture is always ``{layer}_{hook}.bin``.
        if bin_path.name.startswith("packed"):
            continue
        entry = read_per_file(bin_path)
        out[(entry.layer, entry.hook)] = entry
    return out


def read_sharded(
    tag_dir: str | pathlib.Path,
) -> dict[str, dict[tuple[int, str], CaptureEntry]]:
    """Read every capture in a tag's shard files, grouped by request.

    Scans all sealed ``shard-*.json`` indexes under ``tag_dir`` (the
    "sharded" layout writes many requests' captures into shared shard
    files). Returns ``{request_id: {(layer, hook): CaptureEntry}}``. A
    capture that spans multiple shards (because a shard sealed mid-request)
    is reassembled across them in (shard seq, byte-offset) order.

    Only **sealed** shards are visible — unsealed shards (still being
    written, or never flushed) have no ``.json`` and are skipped.
    """
    tag_dir = pathlib.Path(tag_dir)
    # (request_id, layer, hook) -> (dtype, list[(sort_key, array, positions)])
    grouped: dict[
        tuple[str, int, str],
        tuple[str, list[tuple[tuple[int, int], np.ndarray, list[int] | None]]],
    ] = {}
    for index_path in sorted(tag_dir.glob(SHARD_INDEX_GLOB)):
        index = json.loads(index_path.read_text())
        index_dtype = index["dtype"]
        seq = int(index.get("seq", 0))
        bin_path = index_path.with_suffix(".bin")
        raw = bin_path.read_bytes()
        for entry in index["entries"]:
            offset = int(entry["offset"])
            nbytes = int(entry["nbytes"])
            chunk = raw[offset : offset + nbytes]
            if len(chunk) != nbytes:
                raise ValueError(
                    f"shard bin {bin_path} truncated: entry "
                    f"({entry['request_id']}, {entry['layer']}, {entry['hook']}) "
                    f"wants [{offset}:{offset + nbytes}] but file is {len(raw)} bytes"
                )
            rid = str(entry["request_id"])
            layer, hook = int(entry["layer"]), str(entry["hook"])
            # Per-entry dtype is authoritative; index-level is the fallback.
            entry_dtype = entry.get("dtype", index_dtype)
            shape = _decode_shape(entry["shape"], entry.get("row_shape"))
            key = (rid, layer, hook)
            if key not in grouped:
                grouped[key] = (entry_dtype, [])
            grouped[key][1].append(
                (
                    (seq, offset),
                    _decode(chunk, shape, entry_dtype),
                    entry.get("positions"),
                )
            )

    out: dict[str, dict[tuple[int, str], CaptureEntry]] = {}
    for (rid, layer, hook), (dtype, parts) in grouped.items():
        parts.sort(key=lambda p: p[0])
        arrays = [a for _, a, _ in parts]
        array = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
        out.setdefault(rid, {})[(layer, hook)] = CaptureEntry(
            layer=layer,
            hook=hook,
            array=array,
            dtype=dtype,
            positions=_merge_positions([p for _, _, p in parts]),
        )
    return out


def latest_per_position(entry: CaptureEntry) -> CaptureEntry:
    """Collapse a capture to one row per logical position, keeping the last.

    Under speculative decoding a generated position can appear in several
    rows — a rejected draft captured in a verify step, then the accepted
    token re-forwarded in a later step. Rows are stored in write (step)
    order, so the last row for each position is the canonical (accepted)
    one. This returns a new :class:`CaptureEntry` with the rows reduced to
    the last occurrence of each position, ordered by ascending position.

    Raises ``ValueError`` if ``entry.positions`` is ``None`` (no per-row
    position labels — e.g. an older capture), since dedup is impossible
    without them.
    """
    if entry.positions is None:
        raise ValueError(
            "entry has no per-row positions; cannot dedup "
            "(capture predates position recording)"
        )
    last_row: dict[int, int] = {}
    for row, pos in enumerate(entry.positions):
        last_row[pos] = row
    ordered = sorted(last_row)
    row_idx = [last_row[p] for p in ordered]
    return CaptureEntry(
        layer=entry.layer,
        hook=entry.hook,
        array=entry.array[row_idx],
        dtype=entry.dtype,
        positions=ordered,
    )


__all__ = [
    "CaptureEntry",
    "latest_per_position",
    "read_per_file",
    "read_packed",
    "read_request",
    "read_sharded",
]
