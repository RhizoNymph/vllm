# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression: decode-only per-request steering vs the nothing-active
short-circuit in ``_update_steering_buffers``.

A request that carries *decode* steering but no *prefill* steering
(``prefill_hash == 0``, ``decode_hash != 0``), with no global vectors
active, registers its decode config lazily at the prefill->decode
transition inside ``_update_steering_buffers``. If the nothing-active
short-circuit returns first (empty manager state), that transition never
runs, the decode config is never registered, and the steering is silently
dropped forever (``config_to_row`` stays empty, so the short-circuit keeps
firing). The fix makes a pending per-request steering hash in the batch
defeat the short-circuit. CPU-only, no engine.
"""

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.steering_manager import SteeringManager
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin

HIDDEN = 8
MAX_CONFIGS = 4
NUM_ROWS = MAX_CONFIGS + 3
_HP = "post_mlp"


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("steering_table_post_mlp", torch.zeros(NUM_ROWS, HIDDEN))
        self.register_buffer(
            "steering_table_post_mlp_any_active", torch.zeros(1, dtype=torch.bool)
        )
        self.register_buffer("steering_index", torch.zeros(16, dtype=torch.long))


class _FakeInputBatch:
    def __init__(self, reqs: list[dict]):
        self.num_reqs = len(reqs)
        self.req_ids = [r["req_id"] for r in reqs]
        self.req_id_to_index = {r["req_id"]: i for i, r in enumerate(reqs)}
        self.num_computed_tokens_cpu = np.array(
            [r["num_computed"] for r in reqs], dtype=np.int32
        )
        self.num_prompt_tokens = np.array(
            [r["num_prompt"] for r in reqs], dtype=np.int32
        )
        self.request_prefill_steering_hash = np.array(
            [r.get("prefill_hash", 0) for r in reqs], dtype=np.int64
        )
        self.request_decode_steering_hash = np.array(
            [r.get("decode_hash", 0) for r in reqs], dtype=np.int64
        )


class _FakeSchedulerOutput:
    def __init__(self, scheduled: dict[str, int]):
        self.num_scheduled_tokens = dict(scheduled)


class _Host(SteeringModelRunnerMixin):
    def __init__(self, reqs: list[dict]):
        self._steering_manager = SteeringManager(
            max_steering_configs=MAX_CONFIGS, device=None
        )
        self._steerable_layers_cache = {0: _Layer()}
        self._locally_owned_layers = frozenset({0})
        self._req_steering_phase = {}
        self._steering_index_dirty = False
        self.input_batch = _FakeInputBatch(reqs)
        self.requests = {}
        self._steering_rows_scratch = np.zeros(8, dtype=np.int64)
        self._steering_n_tokens_scratch = np.zeros(8, dtype=np.int64)
        self._steering_index_pinned = torch.zeros(16, dtype=torch.long)


def test_decode_only_request_defeats_nothing_active_short_circuit():
    decode_hash = 4242
    host = _Host(
        [
            {
                "req_id": "r1",
                "num_computed": 6,
                "num_prompt": 8,
                "prefill_hash": 0,
                "decode_hash": decode_hash,
            }
        ]
    )
    sp = SamplingParams(
        max_tokens=4, decode_steering_vectors={_HP: {0: [1.0] * HIDDEN}}
    )
    host.requests = {"r1": SimpleNamespace(sampling_params=sp)}
    # Nothing registered yet — pre-fix this short-circuits and never registers.
    assert not host._steering_manager.config_to_row
    # 6 + 2 >= 8 ⇒ prefill completes this step ⇒ prefill->decode transition.
    host._update_steering_buffers(_FakeSchedulerOutput({"r1": 2}))
    assert (decode_hash, "decode") in host._steering_manager.config_to_row, (
        "decode-only request's config was never registered — the short-circuit "
        "swallowed the prefill->decode transition"
    )
    assert host._req_steering_phase.get("r1") == "decode"


def test_no_steering_still_short_circuits():
    """A batch with no per-request steering hashes still short-circuits
    (the guard must not defeat the optimization in the common case)."""
    host = _Host(
        [{"req_id": "r1", "num_computed": 6, "num_prompt": 8}]
    )
    host._update_steering_buffers(_FakeSchedulerOutput({"r1": 2}))
    assert not host._steering_manager.config_to_row
    assert "r1" not in host._req_steering_phase


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
