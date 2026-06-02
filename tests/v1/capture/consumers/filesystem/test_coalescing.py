# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for ActivationWriter same-key write coalescing.

Coalescing merges consecutive same-key ``WriteTask``s already queued on a
worker into a single ``os.writev``. The invariant: the bytes on disk for
each key must be byte-identical to (and in the same order as) writing each
chunk separately. Exercises interleaved keys (so batches stop at the
interleaving boundary) and a tiny ``coalesce_max_bytes`` (so a single key
spans several writev batches, hitting the multi-batch + short-write paths).
"""

from __future__ import annotations

import pathlib
import random
import threading

from vllm.v1.capture.consumers.filesystem.writer import (
    ActivationWriter,
    FinalizeTask,
    WriteTask,
)


def _drive(
    root: pathlib.Path,
    *,
    coalesce_max_bytes: int,
    num_threads: int,
    seed: int,
    num_requests: int = 40,
    max_steps: int = 25,
) -> int:
    """Submit interleaved multi-step captures; verify byte-exact output.

    Returns the number of mismatched files (0 == pass).
    """
    rng = random.Random(seed)
    writer = ActivationWriter(
        root,
        num_threads=num_threads,
        queue_size=100_000,
        coalesce_max_bytes=coalesce_max_bytes,
    )
    done = threading.Event()
    finalized = {"n": 0}
    lock = threading.Lock()

    expected: dict[pathlib.Path, bytes] = {}
    plans: list[tuple[tuple[str, int, str], pathlib.Path, pathlib.Path, list[bytes]]] = []
    for r in range(num_requests):
        req = f"req_{r:04d}"
        steps = rng.randint(1, max_steps)
        layer, hook = rng.randint(0, 5), "post_mlp"
        d = root / req
        d.mkdir(parents=True, exist_ok=True)
        bp = d / f"{layer}_{hook}.bin"
        sp = d / f"{layer}_{hook}.json"
        key = (req, layer, hook)
        payloads = [
            bytes([rng.randint(0, 255)]) * rng.choice([8, 64, 4096, 8192, 200_000])
            for _ in range(steps)
        ]
        expected[bp] = b"".join(payloads)
        plans.append((key, bp, sp, payloads))

    total = len(plans)

    def on_status(result):
        if result.status in ("ok", "error"):
            with lock:
                finalized["n"] += 1
                if finalized["n"] >= total:
                    done.set()

    writer.add_status_callback(on_status)

    # Round-robin one write at a time across requests so the per-worker
    # queues interleave keys (coalescing must stop at each boundary).
    pending = [[k, bp, sp, pl, 0] for (k, bp, sp, pl) in plans]
    while pending:
        for item in list(pending):
            key, bp, sp, pl, i = item
            if i < len(pl):
                writer.submit(WriteTask(path=bp, payload=pl[i], append=(i > 0), key=key))
                item[4] += 1
            else:
                writer.submit(
                    FinalizeTask(
                        bin_path=bp, sidecar_path=sp,
                        sidecar_payload={"k": list(key)}, key=key,
                    )
                )
                pending.remove(item)

    assert done.wait(timeout=60.0), f"timeout: {finalized['n']}/{total}"
    writer.shutdown(timeout=30.0)

    bad = 0
    for bp, exp in expected.items():
        if not bp.exists() or bp.read_bytes() != exp:
            bad += 1
    return bad


class TestCoalescing:
    def test_byte_exact_off(self, tmp_path: pathlib.Path) -> None:
        assert _drive(tmp_path, coalesce_max_bytes=0, num_threads=1, seed=1) == 0

    def test_byte_exact_default_cap(self, tmp_path: pathlib.Path) -> None:
        assert _drive(tmp_path, coalesce_max_bytes=1 << 20, num_threads=4, seed=2) == 0

    def test_byte_exact_tiny_cap_multibatch(self, tmp_path: pathlib.Path) -> None:
        # 4 KiB cap forces a single key to span many writev batches and
        # exercises the writev short-write drain loop with big payloads.
        assert _drive(tmp_path, coalesce_max_bytes=4096, num_threads=4, seed=3) == 0

    def test_byte_exact_single_thread_tiny(self, tmp_path: pathlib.Path) -> None:
        assert _drive(tmp_path, coalesce_max_bytes=4096, num_threads=1, seed=4) == 0
