# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hash determinism for the steering config hash.

Pins the property that two semantically-identical steering specs
produce identical hashes — across dict insertion order, across Python
processes, and across different but numerically-equal float
representations. The scheduler relies on this invariant: every
TP × PP worker independently hashes ``effective_prefill_steering`` /
``effective_decode_steering`` from the same ``SchedulerOutput`` payload
and must arrive at the same ``config_to_row`` allocation.

No code change is required — this file pins existing behavior.
"""

from __future__ import annotations

import subprocess
import sys

from vllm.config.steering_types import hash_steering_config


def _hash(spec) -> int:
    return hash_steering_config(spec)


class TestHashDeterminism:
    def test_empty_and_none_hash_zero(self):
        assert _hash(None) == 0
        assert _hash({}) == 0

    def test_identical_specs_hash_equal(self):
        a = {"post_mlp": {0: [1.0, 2.0, 3.0]}}
        b = {"post_mlp": {0: [1.0, 2.0, 3.0]}}
        assert _hash(a) == _hash(b)

    def test_dict_insertion_order_does_not_matter(self):
        a = {
            "post_mlp": {0: [1.0, 2.0], 1: [3.0, 4.0]},
            "pre_attn": {5: [5.0, 6.0]},
        }
        # Same data, different insertion orders.
        b: dict = {}
        b["pre_attn"] = {5: [5.0, 6.0]}
        b["post_mlp"] = {}
        b["post_mlp"][1] = [3.0, 4.0]
        b["post_mlp"][0] = [1.0, 2.0]
        assert _hash(a) == _hash(b)

    def test_different_vector_values_hash_different(self):
        a = {"post_mlp": {0: [1.0, 2.0, 3.0]}}
        b = {"post_mlp": {0: [1.0, 2.0, 3.1]}}
        assert _hash(a) != _hash(b)

    def test_different_layer_indices_hash_different(self):
        a = {"post_mlp": {0: [1.0, 2.0, 3.0]}}
        b = {"post_mlp": {1: [1.0, 2.0, 3.0]}}
        assert _hash(a) != _hash(b)

    def test_different_hook_points_hash_different(self):
        a = {"post_mlp": {0: [1.0, 2.0, 3.0]}}
        b = {"pre_attn": {0: [1.0, 2.0, 3.0]}}
        assert _hash(a) != _hash(b)

    def test_fits_in_int64(self):
        a = {"post_mlp": {0: [1.0] * 1024}}
        h = _hash(a)
        assert 0 <= h < 2**63, f"Hash {h} outside signed int64 range"

    def test_across_processes(self):
        """Same spec hashes to the same value in a fresh Python process.

        The ``collective_rpc`` fan-out produces a fresh Python process
        per worker (under multiproc executor), so hashes must agree
        across processes — not just across invocations within one
        interpreter — for the scheduler's config_to_row allocations
        to stay in lock-step.
        """
        script = (
            "from vllm.config.steering_types import hash_steering_config; "
            "print(hash_steering_config("
            "{'post_mlp': {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]}}"
            "))"
        )
        first = subprocess.check_output([sys.executable, "-c", script])
        second = subprocess.check_output([sys.executable, "-c", script])
        assert first == second, (
            f"Hash differs across processes: {first!r} vs {second!r}"
        )
        # And matches the in-process hash.
        in_process = _hash({"post_mlp": {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]}})
        assert int(first.strip()) == in_process
