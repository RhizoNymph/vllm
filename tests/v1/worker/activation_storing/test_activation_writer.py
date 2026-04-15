# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for :mod:`vllm.v1.worker.activation_writer`.

Covers the Phase 2 "Done when" bullets from
``docs/features/activation_storing_roadmap.md``:

1. happy-path WriteTask + FinalizeTask
2. multi-step append for a single key
3. append ordering preserved across partitions for multiple req_ids
4. atomic visibility of the destination .bin/.json
5. collision policies: overwrite / error / suffix
6. permission-denied on tag dir
7. disk-full (ENOSPC) simulation
8. graceful shutdown drain
9. shutdown with backlog past timeout
10. FD cache eviction and reopen

Every test uses ``tmp_path`` for isolation.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import pathlib
import threading
import time

import pytest

from vllm.v1.worker.activation_writer import (
    ActivationWriter,
    FinalizeTask,
    WriteError,
    WriteResult,
    WriteTask,
)

# ---------------------------------------------------------------------------
# Helpers


def _bin_path(root: pathlib.Path, key: tuple[str, int, str]) -> pathlib.Path:
    request_id, layer, hook = key
    return root / "tag" / str(layer) / hook / f"{request_id}.bin"


def _sidecar_path(root: pathlib.Path, key: tuple[str, int, str]) -> pathlib.Path:
    request_id, layer, hook = key
    return root / "tag" / str(layer) / hook / f"{request_id}.json"


def _sidecar_payload(key: tuple[str, int, str]) -> dict:
    return {
        "request_id": key[0],
        "layer": key[1],
        "hook": key[2],
        "shape": [1, 4],
        "dtype": "float32",
    }


def _wait_for(
    writer: ActivationWriter,
    key: tuple[str, int, str],
    *,
    timeout: float = 5.0,
) -> WriteResult:
    """Spin until the writer reports a terminal result for ``key``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = writer.get_result(key)
        if result is not None and result.status in ("ok", "error"):
            return result
        time.sleep(0.005)
    pytest.fail(f"timeout waiting for terminal result on {key}")


def _wait_until(cond, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.005)
    return False


# ---------------------------------------------------------------------------
# 1. Happy path


def test_write_and_finalize_happy_path(tmp_path: pathlib.Path) -> None:
    writer = ActivationWriter(tmp_path, num_threads=2)
    try:
        key = ("req-1", 12, "post_mlp")
        bin_path = _bin_path(tmp_path, key)
        sidecar_path = _sidecar_path(tmp_path, key)
        payload = b"\x00\x01\x02\x03"

        writer.submit(WriteTask(path=bin_path, payload=payload, append=False, key=key))
        writer.submit(
            FinalizeTask(
                bin_path=bin_path,
                sidecar_path=sidecar_path,
                sidecar_payload=_sidecar_payload(key),
                key=key,
            )
        )

        result = _wait_for(writer, key)
        assert result.status == "ok", result.error
        assert result.bin_path == bin_path
        assert result.sidecar_path == sidecar_path

        assert bin_path.read_bytes() == payload
        manifest = json.loads(sidecar_path.read_text())
        assert manifest == _sidecar_payload(key)
        assert not bin_path.with_name(bin_path.name + ".tmp").exists()
        assert not sidecar_path.with_name(sidecar_path.name + ".tmp").exists()
    finally:
        writer.shutdown()


# ---------------------------------------------------------------------------
# 2. Multi-step append for a single key


def test_multi_step_append_concatenates_in_order(
    tmp_path: pathlib.Path,
) -> None:
    writer = ActivationWriter(tmp_path, num_threads=2)
    try:
        key_a = ("req-a", 3, "pre_attn")
        key_b = ("req-b", 3, "pre_attn")
        bin_a = _bin_path(tmp_path, key_a)
        bin_b = _bin_path(tmp_path, key_b)
        sidecar_a = _sidecar_path(tmp_path, key_a)
        sidecar_b = _sidecar_path(tmp_path, key_b)

        # Interleave several append writes for two keys.
        sequence_a = [b"aa", b"bb", b"cc", b"dd"]
        sequence_b = [b"11", b"22", b"33"]

        writer.submit(
            WriteTask(path=bin_a, payload=sequence_a[0], append=False, key=key_a)
        )
        writer.submit(
            WriteTask(path=bin_b, payload=sequence_b[0], append=False, key=key_b)
        )
        writer.submit(
            WriteTask(path=bin_a, payload=sequence_a[1], append=True, key=key_a)
        )
        writer.submit(
            WriteTask(path=bin_b, payload=sequence_b[1], append=True, key=key_b)
        )
        writer.submit(
            WriteTask(path=bin_a, payload=sequence_a[2], append=True, key=key_a)
        )
        writer.submit(
            WriteTask(path=bin_a, payload=sequence_a[3], append=True, key=key_a)
        )
        writer.submit(
            WriteTask(path=bin_b, payload=sequence_b[2], append=True, key=key_b)
        )

        writer.submit(
            FinalizeTask(
                bin_path=bin_a,
                sidecar_path=sidecar_a,
                sidecar_payload=_sidecar_payload(key_a),
                key=key_a,
            )
        )
        writer.submit(
            FinalizeTask(
                bin_path=bin_b,
                sidecar_path=sidecar_b,
                sidecar_payload=_sidecar_payload(key_b),
                key=key_b,
            )
        )

        assert _wait_for(writer, key_a).status == "ok"
        assert _wait_for(writer, key_b).status == "ok"

        assert bin_a.read_bytes() == b"".join(sequence_a)
        assert bin_b.read_bytes() == b"".join(sequence_b)
    finally:
        writer.shutdown()


# ---------------------------------------------------------------------------
# 3. Append ordering preserved across partitions


def test_append_ordering_across_partitions(tmp_path: pathlib.Path) -> None:
    writer = ActivationWriter(tmp_path, num_threads=4)
    try:
        # Use many request_ids so we have a reasonable chance of
        # landing on more than one partition.
        req_ids = [f"req-{i}" for i in range(8)]
        per_req_payloads: dict[str, list[bytes]] = {}
        keys: dict[str, tuple[str, int, str]] = {}

        for req in req_ids:
            key = (req, 0, "post_mlp")
            keys[req] = key
            # Eight chunks each. Unique bytes per chunk so
            # concatenation order is checkable.
            per_req_payloads[req] = [f"{req}#{idx:02d}".encode() for idx in range(8)]

        # Interleave tasks: round 0 for every req, then round 1, etc.
        for round_idx in range(8):
            for req in req_ids:
                key = keys[req]
                bin_path = _bin_path(tmp_path, key)
                writer.submit(
                    WriteTask(
                        path=bin_path,
                        payload=per_req_payloads[req][round_idx],
                        append=(round_idx > 0),
                        key=key,
                    )
                )

        for req in req_ids:
            key = keys[req]
            writer.submit(
                FinalizeTask(
                    bin_path=_bin_path(tmp_path, key),
                    sidecar_path=_sidecar_path(tmp_path, key),
                    sidecar_payload=_sidecar_payload(key),
                    key=key,
                )
            )

        for req in req_ids:
            key = keys[req]
            assert _wait_for(writer, key).status == "ok"
            expected = b"".join(per_req_payloads[req])
            assert _bin_path(tmp_path, key).read_bytes() == expected
    finally:
        writer.shutdown()


# ---------------------------------------------------------------------------
# 4. Atomic visibility


def test_destination_invisible_until_finalize(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = ActivationWriter(tmp_path, num_threads=1)
    try:
        key = ("req-visible", 1, "pre_attn")
        bin_path = _bin_path(tmp_path, key)
        sidecar_path = _sidecar_path(tmp_path, key)

        # Gate completion of the first write so we can observe the
        # in-flight state with certainty.
        release = threading.Event()
        real_write = os.write

        def slow_write(fd: int, data: bytes) -> int:
            release.wait(timeout=5.0)
            return real_write(fd, data)

        monkeypatch.setattr("vllm.v1.worker.activation_writer.os.write", slow_write)

        writer.submit(
            WriteTask(
                path=bin_path,
                payload=b"bytes",
                append=False,
                key=key,
            )
        )

        # The destination .bin must not exist while the write is
        # still pending.
        assert not bin_path.exists()
        tmp_bin = bin_path.with_name(bin_path.name + ".tmp")
        # Give the worker a moment to open the tmp file.
        _wait_until(lambda: tmp_bin.exists(), timeout=1.0)
        assert tmp_bin.exists()
        assert not bin_path.exists()

        release.set()

        # Finalize uses the real os.write via the same monkeypatch;
        # release is already set so it returns immediately.
        writer.submit(
            FinalizeTask(
                bin_path=bin_path,
                sidecar_path=sidecar_path,
                sidecar_payload=_sidecar_payload(key),
                key=key,
            )
        )
        assert _wait_for(writer, key).status == "ok"
        assert bin_path.exists()
        assert sidecar_path.exists()
        assert not tmp_bin.exists()
    finally:
        writer.shutdown()


# ---------------------------------------------------------------------------
# 5. Collision policies


def _write_once(
    root: pathlib.Path,
    key: tuple[str, int, str],
    *,
    policy: str,
    payload: bytes,
) -> WriteResult:
    writer = ActivationWriter(root, num_threads=1, on_collision=policy)  # type: ignore[arg-type]
    try:
        bin_path = _bin_path(root, key)
        sidecar_path = _sidecar_path(root, key)
        writer.submit(WriteTask(path=bin_path, payload=payload, append=False, key=key))
        writer.submit(
            FinalizeTask(
                bin_path=bin_path,
                sidecar_path=sidecar_path,
                sidecar_payload=_sidecar_payload(key),
                key=key,
            )
        )
        return _wait_for(writer, key)
    finally:
        writer.shutdown()


def test_collision_overwrite_replaces_existing(
    tmp_path: pathlib.Path,
) -> None:
    key = ("req-overwrite", 5, "post_mlp")
    first = _write_once(tmp_path, key, policy="overwrite", payload=b"first")
    assert first.status == "ok"
    assert _bin_path(tmp_path, key).read_bytes() == b"first"

    second = _write_once(tmp_path, key, policy="overwrite", payload=b"second!!")
    assert second.status == "ok"
    assert _bin_path(tmp_path, key).read_bytes() == b"second!!"


def test_collision_error_preserves_existing(tmp_path: pathlib.Path) -> None:
    key = ("req-error", 5, "post_mlp")
    first = _write_once(tmp_path, key, policy="error", payload=b"orig")
    assert first.status == "ok"

    second = _write_once(tmp_path, key, policy="error", payload=b"nope!")
    assert second.status == "error"
    assert second.error is not None
    assert second.error.errno_code == errno.EEXIST
    # Original file untouched.
    assert _bin_path(tmp_path, key).read_bytes() == b"orig"


def test_collision_suffix_writes_to_new_path(
    tmp_path: pathlib.Path,
) -> None:
    key = ("req-suffix", 5, "post_mlp")
    first = _write_once(tmp_path, key, policy="suffix", payload=b"orig")
    assert first.status == "ok"
    assert first.bin_path == _bin_path(tmp_path, key)

    second = _write_once(tmp_path, key, policy="suffix", payload=b"new!!")
    assert second.status == "ok"
    assert second.bin_path is not None
    assert second.bin_path != _bin_path(tmp_path, key)
    # Original untouched.
    assert _bin_path(tmp_path, key).read_bytes() == b"orig"
    # New payload landed at the reported path.
    assert second.bin_path.read_bytes() == b"new!!"
    # Path shape: <stem>.{ms}<suffix>
    assert second.bin_path.suffix == ".bin"
    assert "." in second.bin_path.stem
    stem_parts = second.bin_path.stem.split(".")
    assert stem_parts[0] == "req-suffix"
    assert stem_parts[-1].isdigit()


# ---------------------------------------------------------------------------
# 6. Permission denied


def test_permission_denied_surfaces_structured_error(
    tmp_path: pathlib.Path,
) -> None:
    writer = ActivationWriter(tmp_path, num_threads=1)
    try:
        bad_key = ("req-denied", 7, "post_mlp")
        good_key = ("req-ok", 8, "post_mlp")
        bad_bin = _bin_path(tmp_path, bad_key)
        good_bin = _bin_path(tmp_path, good_key)

        # Create the tag dir with read-only permissions so the
        # writer's mkdir-then-open fails with EACCES. Both requests
        # target the same tag dir; the bad one triggers the failure,
        # the good one goes to a different layer under a fresh path.
        bad_dir = bad_bin.parent
        bad_dir.mkdir(parents=True, exist_ok=True)
        try:
            bad_dir.chmod(0o555)

            writer.submit(
                WriteTask(
                    path=bad_bin,
                    payload=b"doomed",
                    append=False,
                    key=bad_key,
                )
            )
            writer.submit(
                FinalizeTask(
                    bin_path=bad_bin,
                    sidecar_path=_sidecar_path(tmp_path, bad_key),
                    sidecar_payload=_sidecar_payload(bad_key),
                    key=bad_key,
                )
            )

            # A completely unrelated request still succeeds even after
            # the permission failure; the pool keeps draining.
            writer.submit(
                WriteTask(
                    path=good_bin,
                    payload=b"survives",
                    append=False,
                    key=good_key,
                )
            )
            writer.submit(
                FinalizeTask(
                    bin_path=good_bin,
                    sidecar_path=_sidecar_path(tmp_path, good_key),
                    sidecar_payload=_sidecar_payload(good_key),
                    key=good_key,
                )
            )

            bad_result = _wait_for(writer, bad_key)
            assert bad_result.status == "error"
            assert bad_result.error is not None
            assert bad_result.error.errno_code == errno.EACCES

            good_result = _wait_for(writer, good_key)
            assert good_result.status == "ok"
            assert good_bin.read_bytes() == b"survives"
        finally:
            # Always restore write permission so tmp_path teardown
            # can clean up.
            bad_dir.chmod(0o755)
    finally:
        writer.shutdown()


# ---------------------------------------------------------------------------
# 7. Disk full simulation


def test_enospc_marks_task_failed_other_tasks_succeed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = ActivationWriter(tmp_path, num_threads=1)
    try:
        failing_key = ("req-enospc", 1, "post_mlp")
        failing_bin = _bin_path(tmp_path, failing_key)
        success_key = ("req-fine", 1, "post_mlp")
        success_bin = _bin_path(tmp_path, success_key)

        real_write = os.write
        real_open = os.open
        real_close = os.close
        # Map live fds -> opened path so we can check the file
        # identity at write time rather than chasing fd reuse.
        fd_to_path: dict[int, str] = {}
        state: dict[str, int] = {"calls": 0}

        def tracking_open(path, flags, mode=0o777, *a, **kw):  # type: ignore[override]
            fd = real_open(path, flags, mode, *a, **kw)
            fd_to_path[fd] = str(path)
            return fd

        def tracking_close(fd: int) -> None:
            fd_to_path.pop(fd, None)
            real_close(fd)

        def flaky_write(fd: int, data: bytes) -> int:
            path = fd_to_path.get(fd)
            # Only fail the single failing_bin.tmp, not the sidecar
            # or any subsequent key's file.
            if path is not None and path == str(failing_bin) + ".tmp":
                state["calls"] += 1
                raise OSError(errno.ENOSPC, "no space left on device")
            return real_write(fd, data)

        monkeypatch.setattr("vllm.v1.worker.activation_writer.os.open", tracking_open)
        monkeypatch.setattr("vllm.v1.worker.activation_writer.os.close", tracking_close)
        monkeypatch.setattr("vllm.v1.worker.activation_writer.os.write", flaky_write)

        writer.submit(
            WriteTask(
                path=failing_bin,
                payload=b"this will fail",
                append=False,
                key=failing_key,
            )
        )
        writer.submit(
            FinalizeTask(
                bin_path=failing_bin,
                sidecar_path=_sidecar_path(tmp_path, failing_key),
                sidecar_payload=_sidecar_payload(failing_key),
                key=failing_key,
            )
        )

        # A second, unrelated capture writes to a different file and
        # must continue to succeed even after the ENOSPC on the first.
        writer.submit(
            WriteTask(
                path=success_bin,
                payload=b"ok",
                append=False,
                key=success_key,
            )
        )
        writer.submit(
            FinalizeTask(
                bin_path=success_bin,
                sidecar_path=_sidecar_path(tmp_path, success_key),
                sidecar_payload=_sidecar_payload(success_key),
                key=success_key,
            )
        )

        bad = _wait_for(writer, failing_key)
        assert bad.status == "error"
        assert bad.error is not None
        assert bad.error.errno_code == errno.ENOSPC

        good = _wait_for(writer, success_key)
        assert good.status == "ok"
        assert success_bin.read_bytes() == b"ok"
        assert state["calls"] >= 1
    finally:
        writer.shutdown()


# ---------------------------------------------------------------------------
# 8. Graceful shutdown drain


def test_graceful_shutdown_drains_inflight_tasks(
    tmp_path: pathlib.Path,
) -> None:
    writer = ActivationWriter(tmp_path, num_threads=2)
    keys = [(f"req-drain-{i}", 2, "post_mlp") for i in range(6)]

    for key in keys:
        bin_path = _bin_path(tmp_path, key)
        writer.submit(
            WriteTask(
                path=bin_path,
                payload=f"payload-{key[0]}".encode(),
                append=False,
                key=key,
            )
        )
        writer.submit(
            FinalizeTask(
                bin_path=bin_path,
                sidecar_path=_sidecar_path(tmp_path, key),
                sidecar_payload=_sidecar_payload(key),
                key=key,
            )
        )

    writer.shutdown(timeout=5.0)

    for key in keys:
        result = writer.get_result(key)
        assert result is not None
        assert result.status == "ok", result.error
        assert _bin_path(tmp_path, key).exists()
        assert (
            not _bin_path(tmp_path, key)
            .with_name(_bin_path(tmp_path, key).name + ".tmp")
            .exists()
        )


# ---------------------------------------------------------------------------
# 9. Shutdown with backlog past timeout


def test_shutdown_with_backlog_marks_remaining_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = ActivationWriter(tmp_path, num_threads=1, queue_size=32)
    try:
        block = threading.Event()
        unblock = threading.Event()
        real_write = os.write

        def blocking_write(fd: int, data: bytes) -> int:
            block.set()
            unblock.wait(timeout=5.0)
            return real_write(fd, data)

        monkeypatch.setattr("vllm.v1.worker.activation_writer.os.write", blocking_write)

        # First task pins the single worker thread inside os.write.
        pinned_key = ("req-pinned", 0, "post_mlp")
        writer.submit(
            WriteTask(
                path=_bin_path(tmp_path, pinned_key),
                payload=b"blocking",
                append=False,
                key=pinned_key,
            )
        )
        assert block.wait(timeout=5.0)

        # Pile on several queued tasks that will never run inside
        # the grace period.
        queued_keys = [(f"req-queued-{i}", 0, "post_mlp") for i in range(4)]
        for key in queued_keys:
            writer.submit(
                WriteTask(
                    path=_bin_path(tmp_path, key),
                    payload=b"never",
                    append=False,
                    key=key,
                )
            )

        start = time.monotonic()
        shutdown_thread = threading.Thread(target=lambda: writer.shutdown(timeout=0.1))
        shutdown_thread.start()

        # Allow the shutdown grace period to expire while the pinned
        # task is still blocked on os.write.
        shutdown_thread.join(timeout=3.0)
        elapsed = time.monotonic() - start
        assert not shutdown_thread.is_alive(), (
            "shutdown did not return within the bounded grace period"
        )
        assert elapsed < 2.5

        # Release the pinned task so the hung worker can exit after
        # the test (keeps the process clean).
        unblock.set()

        for key in queued_keys:
            result = writer.get_result(key)
            assert result is not None
            assert result.status == "error"
            assert result.error is not None
            assert "shutdown" in result.error.message.lower()

        pinned = writer.get_result(pinned_key)
        assert pinned is not None
        # Pinned task was mid-write at shutdown; it should have been
        # marked errored as part of the "pending past grace" sweep.
        assert pinned.status == "error"
        assert pinned.error is not None
    finally:
        unblock.set()
        with contextlib.suppress(Exception):
            writer.shutdown(timeout=1.0)


# ---------------------------------------------------------------------------
# 10. FD cache eviction and reopen


def test_fd_cache_eviction_and_reopen(tmp_path: pathlib.Path) -> None:
    writer = ActivationWriter(tmp_path, num_threads=1, fd_cache_size=2)
    try:
        key_a = ("req-evict-a", 1, "post_mlp")
        key_b = ("req-evict-b", 1, "post_mlp")
        key_c = ("req-evict-c", 1, "post_mlp")
        keys = [key_a, key_b, key_c]
        bin_paths = {k: _bin_path(tmp_path, k) for k in keys}

        # Step 1: open key_a then key_b. FD cache is at capacity.
        writer.submit(
            WriteTask(
                path=bin_paths[key_a],
                payload=b"aa",
                append=False,
                key=key_a,
            )
        )
        writer.submit(
            WriteTask(
                path=bin_paths[key_b],
                payload=b"bb",
                append=False,
                key=key_b,
            )
        )
        # Step 2: open key_c, which must evict key_a (least-recently
        # used). The evicted fd gets fsync+close. The .tmp file for
        # key_a still exists on disk with "aa" inside.
        writer.submit(
            WriteTask(
                path=bin_paths[key_c],
                payload=b"cc",
                append=False,
                key=key_c,
            )
        )
        # Step 3: a new write for key_a must reopen the .tmp file and
        # append the next chunk without losing the evicted bytes. The
        # task uses ``append=True`` to preserve prior data.
        writer.submit(
            WriteTask(
                path=bin_paths[key_a],
                payload=b"AA",
                append=True,
                key=key_a,
            )
        )

        for key in keys:
            writer.submit(
                FinalizeTask(
                    bin_path=bin_paths[key],
                    sidecar_path=_sidecar_path(tmp_path, key),
                    sidecar_payload=_sidecar_payload(key),
                    key=key,
                )
            )

        for key in keys:
            result = _wait_for(writer, key)
            assert result.status == "ok", result.error

        assert bin_paths[key_a].read_bytes() == b"aaAA"
        assert bin_paths[key_b].read_bytes() == b"bb"
        assert bin_paths[key_c].read_bytes() == b"cc"
    finally:
        writer.shutdown()


# ---------------------------------------------------------------------------
# Extra: sidecar payload validation happens synchronously.


def test_non_serializable_sidecar_rejected_synchronously(
    tmp_path: pathlib.Path,
) -> None:
    writer = ActivationWriter(tmp_path, num_threads=1)
    try:
        key = ("req-bad-sidecar", 0, "post_mlp")
        bad_payload: dict = {"set": {1, 2, 3}}  # sets aren't JSON
        with pytest.raises(WriteError):
            writer.submit(
                FinalizeTask(
                    bin_path=_bin_path(tmp_path, key),
                    sidecar_path=_sidecar_path(tmp_path, key),
                    sidecar_payload=bad_payload,
                    key=key,
                )
            )
    finally:
        writer.shutdown()
