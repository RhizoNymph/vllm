# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``warmup_apply_steering_kernel``.

The Triton ``_apply_steering_kernel`` JIT-compiles a fresh variant for
each new specialization (batch dim ``N``, stride alignment class, etc.)
that it sees at runtime.  Profiling on a 3090 with gemma-3-4b-it shows
that even a single un-warmed shape produces an ~10 ms
``cuLibraryLoadData`` event inside the served-request window.

These tests verify that:

* The warmup is a no-op on CPU (the kernel never executes there).
* The CUDA warmup populates Triton's per-device JIT cache with at
  least one variant for the supplied shape list.
* Re-driving the warmup at the same shape list does **not** add new
  variants — i.e. the cache is reused, so warmup is idempotent and
  cannot regress into a per-step recompile loop.
"""

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.layers.steering_kernel import (
    _kernel_cache_size,
    warmup_apply_steering_kernel,
)


class TestWarmupCPU:
    """Warmup must short-circuit cleanly when the device is CPU."""

    def test_cpu_device_is_noop(self):
        """No exception, no kernel launches."""
        warmup_apply_steering_kernel(
            hidden_size=64,
            table_rows=8,
            table_dtype=torch.float32,
            compute_dtype=torch.float32,
            device=torch.device("cpu"),
            capture_sizes=[1, 2, 4],
        )

    def test_cpu_device_with_no_capture_sizes(self):
        """Falls back to default sizes but still short-circuits on CPU."""
        warmup_apply_steering_kernel(
            hidden_size=64,
            table_rows=8,
            table_dtype=torch.float32,
            compute_dtype=torch.float32,
            device=torch.device("cpu"),
            capture_sizes=None,
        )

    def test_kernel_cache_size_handles_missing_attr(self):
        """Helper must not crash when the kernel hasn't been built yet."""
        # On a CPU-only host Triton is disabled and the kernel is a
        # placeholder without a ``cache`` attribute.  The helper should
        # treat that as zero variants.
        size = _kernel_cache_size()
        assert size >= 0


# ---------------------------------------------------------------------------
# CUDA path — actually exercises the JIT cache
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestWarmupCUDA:
    """Warmup populates the Triton cache and is idempotent on re-run."""

    def test_warmup_populates_cache(self):
        """The cache must hold at least one variant after warmup."""
        device = torch.device("cuda")
        # Representative shape list — mirrors a small subset of vLLM's
        # default ``cudagraph_capture_sizes`` so the test exercises the
        # iteration loop without taking forever.
        shapes = [1, 2, 4, 8, 16, 32]
        warmup_apply_steering_kernel(
            hidden_size=128,
            table_rows=8,
            table_dtype=torch.float16,
            compute_dtype=torch.float16,
            device=device,
            capture_sizes=shapes,
        )
        assert _kernel_cache_size() >= 1, (
            "Warmup must compile at least one Triton variant; cache is empty."
        )

    def test_rewarmup_does_not_grow_cache(self):
        """Re-running warmup at the same shapes must not add variants."""
        device = torch.device("cuda")
        shapes = [1, 2, 4, 8, 16, 32]
        warmup_apply_steering_kernel(
            hidden_size=128,
            table_rows=8,
            table_dtype=torch.float16,
            compute_dtype=torch.float16,
            device=device,
            capture_sizes=shapes,
        )
        size_after_first = _kernel_cache_size()
        warmup_apply_steering_kernel(
            hidden_size=128,
            table_rows=8,
            table_dtype=torch.float16,
            compute_dtype=torch.float16,
            device=device,
            capture_sizes=shapes,
        )
        size_after_second = _kernel_cache_size()
        assert size_after_second == size_after_first, (
            f"Warmup must be idempotent at fixed shapes; cache grew from "
            f"{size_after_first} to {size_after_second}."
        )

    @pytest.mark.parametrize("row_monitor_enabled", [False, True])
    def test_subsequent_invocations_at_warmed_shape_no_new_variants(
        self, row_monitor_enabled
    ):
        """Calling the registered op at a warmed shape must not recompile.

        This is the property that actually matters for the served-window
        ``cuLibraryLoadData`` budget: once warmup has touched a shape,
        the runtime call at that shape must hit the cache. The runtime-mimic
        must use the SAME per-row monitor buffer shapes the runner keeps for
        this config — the ``(1, 1)`` / ``(1, 2)`` registered dummies when the
        row monitor is disabled (the default), the resized ``(8, 128)`` /
        ``(8, 2)`` tables when it is enabled — because Triton specializes the
        probe table's leading stride into a distinct variant. (The previous
        version always used full-size buffers, matching the buggy warmup rather
        than the true default runtime, so it never caught the mismatch.)
        """
        device = torch.device("cuda")
        shapes = [4, 16]
        table_rows = 8
        warmup_apply_steering_kernel(
            hidden_size=128,
            table_rows=table_rows,
            table_dtype=torch.float16,
            compute_dtype=torch.float16,
            device=device,
            capture_sizes=shapes,
            row_monitor_enabled=row_monitor_enabled,
        )
        baseline = _kernel_cache_size()

        # Per-row monitor buffer shapes as the runner keeps them for this
        # config (matching steering.py's registered dummies /
        # resize_steering_row_monitor_buffers output).
        if row_monitor_enabled:
            rprobe = torch.zeros(table_rows, 128, dtype=torch.float32, device=device)
            rparams = (
                torch.tensor([-1.0e30, 1.0], dtype=torch.float32, device=device)
                .expand(table_rows, 2)
                .clone()
            )
        else:
            rprobe = torch.zeros(1, 1, dtype=torch.float32, device=device)
            rparams = torch.tensor(
                [[-1.0e30, 1.0]], dtype=torch.float32, device=device
            )

        # Mimic a runtime invocation for each warmed shape.
        for n in shapes:
            hidden = torch.zeros(n, 128, dtype=torch.float16, device=device)
            table = torch.zeros(table_rows, 128, dtype=torch.float16, device=device)
            index = torch.zeros(n, dtype=torch.long, device=device)
            active = torch.ones(1, dtype=torch.bool, device=device)
            scales = torch.ones(table_rows, dtype=torch.float32, device=device)
            dvec = torch.zeros(128, dtype=torch.float32, device=device)
            tscale = torch.zeros(n, dtype=torch.float32, device=device)
            rgate = torch.ones(n, dtype=torch.float32, device=device)
            probe = torch.zeros(128, dtype=torch.float32, device=device)
            mparams = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
            mactive = torch.zeros(1, dtype=torch.bool, device=device)
            dmask = torch.zeros(n, dtype=torch.float32, device=device)
            ractive = torch.zeros(1, dtype=torch.bool, device=device)
            torch.ops.vllm.apply_steering(
                hidden,
                table,
                index,
                active,
                scales,
                dvec,
                tscale,
                rgate,
                probe,
                mparams,
                mactive,
                dmask,
                rprobe,
                rparams,
                ractive,
            )

        torch.accelerator.synchronize()
        assert _kernel_cache_size() == baseline, (
            "Runtime calls at warmed shapes must reuse cached variants; "
            f"cache grew from {baseline} to {_kernel_cache_size()}."
        )
