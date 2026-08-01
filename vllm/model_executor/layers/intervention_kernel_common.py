# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared warmup scaffolding for intervention-tier Triton kernels.

The residual-stream intervention tiers (steering, monitor, patching — and any
future tier) each ship Triton kernels that must be JIT-compiled ahead of CUDA
graph capture, at every batch dim vLLM will capture, through the registered
custom op so the compiled stride specialization matches the dispatched runtime
call. The warmup skeleton around that drive loop — size normalization,
compiled-variant accounting, wall-clock logging, optional JIT-cache dump — is
tier-independent and lives here; each tier keeps only its buffer allocation
and per-size op launches (the payload semantics).

Cache introspection handles both Triton layouts: ``JITFunction.cache``
(``{device: {key: kernel}}``, Triton < 3.6) and ``device_caches`` (per-device
tuple values whose FIRST element is the ``{key: kernel}`` dict, Triton >= 3.6).
A probe that reads only the old attribute silently reports 0 forever, making
every warmup cache check vacuous.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def default_warmup_sizes() -> list[int]:
    """Fallback warmup batch sizes when no capture-size list is supplied.

    Mirrors the powers-of-two and small-batch shapes that vLLM commonly
    captures when ``cudagraph_capture_sizes`` is left to its default. Used
    only when the caller cannot pass an explicit list (e.g. standalone
    tests). Shared by every intervention tier so all kernels warm up the
    same batch dims.
    """
    return [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def normalize_warmup_sizes(capture_sizes: list[int] | None) -> list[int]:
    """Dedupe, drop non-positive entries, and sort ascending.

    Smaller shapes are driven first (smaller compiles tend to be slightly
    cheaper). Falls back to :func:`default_warmup_sizes` when no list is
    supplied.
    """
    sizes = capture_sizes if capture_sizes else default_warmup_sizes()
    return sorted({int(s) for s in sizes if int(s) > 0})


def _jit_device_caches(kernel) -> dict | None:
    """Normalize one kernel's per-device JIT cache across Triton versions.

    Returns a ``{device: {key: kernel}}`` view, or None when the kernel has
    no cache attribute (Triton disabled in the importing process).
    """
    cache = getattr(kernel, "cache", None)
    if cache is None:
        cache = getattr(kernel, "device_caches", None)
    if cache is None:
        return None
    return {
        dev: (dc[0] if isinstance(dc, tuple) and dc else dc)
        for dev, dc in cache.items()
    }


def kernel_cache_size(kernels: Sequence) -> int:
    """Total compiled variants across all devices for the given kernels.

    Returns 0 for kernels not yet built (no cache attribute, e.g. when
    Triton is disabled in the importing process).
    """
    total = 0
    for kernel in kernels:
        cache = _jit_device_caches(kernel)
        if cache is None:
            continue
        for device_cache in cache.values():
            try:
                total += len(device_cache)
            except TypeError:
                continue
    return total


def dump_jit_cache_keys(kernels: Sequence, label: str) -> None:
    """One-shot diagnostic — log every variant key in the kernels' caches."""
    total = 0
    for kernel in kernels:
        cache = _jit_device_caches(kernel)
        if cache is None:
            logger.info(
                "%s JIT cache dump requested but kernel %s has no cache "
                "attribute (Triton may be disabled)",
                label,
                getattr(kernel, "__name__", repr(kernel)),
            )
            continue
        for device_id, device_cache in cache.items():
            try:
                keys = list(device_cache.keys())
            except AttributeError:
                keys = []
            total += len(keys)
            logger.info(
                "%s JIT cache: kernel=%s device=%s variants=%d",
                label,
                getattr(kernel, "__name__", repr(kernel)),
                device_id,
                len(keys),
            )
            for i, key in enumerate(keys):
                logger.info("  variant[%d]: %r", i, key)
    logger.info("%s JIT cache: total_variants=%d", label, total)


def run_kernel_warmup(
    *,
    label: str,
    kernels: Sequence,
    sizes: list[int],
    drive: Callable[[int], None],
    dump_env_var: str | None = None,
) -> None:
    """Drive a tier's registered ops at every warmup size, with accounting.

    ``sizes`` must already be normalized (:func:`normalize_warmup_sizes`) —
    callers need the final list before this call to allocate their drive
    buffers at ``max(sizes)``. ``drive(n)`` launches the tier's ops at batch
    dim ``n``; buffer allocation stays outside the timed window, matching
    the historical per-tier warmup functions.
    """
    cache_before = kernel_cache_size(kernels)
    t0 = time.perf_counter()
    for n in sizes:
        drive(n)
    # Block until every JIT compile (and cuLibraryLoadData) has retired so
    # the wall-clock measurement and cache-size readback reflect reality.
    torch.accelerator.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    cache_after = kernel_cache_size(kernels)
    logger.info(
        "%s kernel warmup: shapes=%d variants_compiled=%d "
        "cache_total=%d elapsed_ms=%.1f",
        label,
        len(sizes),
        cache_after - cache_before,
        cache_after,
        elapsed_ms,
    )

    if dump_env_var and os.environ.get(dump_env_var, "0") == "1":
        dump_jit_cache_keys(kernels, label)
