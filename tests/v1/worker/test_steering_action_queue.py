# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the dynamic-steering action queue (Phase 0).

Covers the queue (bounded submit, FIFO drain, stats, install slot) and
the drain-time validation/application path against a fake
``SteeringManager`` — no GPU or engine required.
"""

import threading

import numpy as np
import pytest
import torch

from vllm.v1.worker.steering_action_queue import (
    SteeringActionQueue,
    SteeringVectorUpdate,
    apply_steering_updates,
    get_steering_action_queue,
    install_steering_action_queue,
)

HIDDEN = 16


class _FakeLayer:
    """Stands in for a decoder layer with steering buffers attached."""

    def __init__(self, hooks: tuple[str, ...] = ("post_mlp",)) -> None:
        for hook in hooks:
            setattr(
                self,
                f"steering_table_{hook}",
                torch.zeros(4, HIDDEN),
            )


class _FakeManager:
    """Records ``SteeringManager`` apply calls.

    ``calls`` tracks ``update_global_vectors`` (the base/prefill
    cache-unsafe escape hatch); ``tier_calls`` tracks
    ``update_dynamic_tier`` (the decode path, where dynamic updates now
    land so they compose additively with operator-set decode steering).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, torch.Tensor, str]] = []
        self.tier_calls: list[tuple[str, int, torch.Tensor]] = []
        self._tables_dirty = False

    def update_global_vectors(self, hook_point, layer_idx, vector, phase="base"):
        self.calls.append((hook_point, layer_idx, vector.clone(), phase))
        self._tables_dirty = True

    def update_dynamic_tier(self, hook_point, layer_idx, vector):
        self.tier_calls.append((hook_point, layer_idx, vector.clone()))
        self._tables_dirty = True


def _update(
    layer: int = 0,
    hook: str = "post_mlp",
    phase: str = "decode",
    vec: np.ndarray | None = None,
) -> SteeringVectorUpdate:
    if vec is None:
        vec = np.ones(HIDDEN, dtype=np.float32)
    return SteeringVectorUpdate(
        vectors={hook: {layer: vec}}, phase=phase, source="test"
    )


# ---------------------------------------------------------------------------
# Queue mechanics
# ---------------------------------------------------------------------------


def test_submit_drain_fifo_order():
    q = SteeringActionQueue()
    updates = [_update(layer=i % 2) for i in range(5)]
    for u in updates:
        assert q.submit(u)
    assert len(q) == 5
    drained = q.drain()
    assert drained == updates
    assert len(q) == 0
    assert q.drain() == []


def test_bounded_submit_drops_newest():
    q = SteeringActionQueue(maxsize=2)
    assert q.submit(_update())
    assert q.submit(_update())
    overflow = _update(layer=1)
    assert not q.submit(overflow)
    drained = q.drain()
    assert len(drained) == 2
    assert overflow not in drained
    stats = q.stats()
    assert stats.submitted == 3
    assert stats.dropped == 1


def test_invalid_maxsize_rejected():
    with pytest.raises(ValueError):
        SteeringActionQueue(maxsize=0)


def test_bool_and_len_reflect_contents():
    q = SteeringActionQueue()
    assert not q
    q.submit(_update())
    assert q
    assert len(q) == 1


def test_concurrent_submit_keeps_all_updates():
    q = SteeringActionQueue(maxsize=10_000)
    n_threads, per_thread = 8, 100

    def worker():
        for _ in range(per_thread):
            q.submit(_update())

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(q.drain()) == n_threads * per_thread


# ---------------------------------------------------------------------------
# Process-global install slot
# ---------------------------------------------------------------------------


def test_install_and_get_roundtrip():
    try:
        q = SteeringActionQueue()
        install_steering_action_queue(q)
        assert get_steering_action_queue() is q
        install_steering_action_queue(None)
        assert get_steering_action_queue() is None
    finally:
        install_steering_action_queue(None)


# ---------------------------------------------------------------------------
# Drain-time validation + application
# ---------------------------------------------------------------------------


def test_apply_valid_decode_update_routes_to_tier():
    mgr = _FakeManager()
    layers = {0: _FakeLayer()}
    vec = np.arange(HIDDEN, dtype=np.float32)
    applied, rejected = apply_steering_updates([_update(vec=vec)], mgr, layers)
    assert (applied, rejected) == (1, 0)
    # Decode updates land on the dynamic tier (additive), not the global
    # decode tier (overwrite).
    assert mgr.calls == []
    assert len(mgr.tier_calls) == 1
    hook, layer, tensor = mgr.tier_calls[0]
    assert (hook, layer) == ("post_mlp", 0)
    assert torch.equal(tensor, torch.from_numpy(vec))
    assert mgr._tables_dirty


def test_apply_multi_layer_update():
    mgr = _FakeManager()
    layers = {0: _FakeLayer(), 1: _FakeLayer()}
    vec = np.ones(HIDDEN, dtype=np.float32)
    update = SteeringVectorUpdate(
        vectors={"post_mlp": {0: vec, 1: vec * 2}}, phase="decode"
    )
    applied, rejected = apply_steering_updates([update], mgr, layers)
    assert (applied, rejected) == (1, 0)
    assert {(c[0], c[1]) for c in mgr.tier_calls} == {("post_mlp", 0), ("post_mlp", 1)}


@pytest.mark.parametrize("phase", ["base", "prefill"])
def test_cache_unsafe_phases_rejected_by_default(phase):
    mgr = _FakeManager()
    layers = {0: _FakeLayer()}
    applied, rejected = apply_steering_updates([_update(phase=phase)], mgr, layers)
    assert (applied, rejected) == (0, 1)
    assert mgr.calls == []


@pytest.mark.parametrize("phase", ["base", "prefill"])
def test_cache_unsafe_phases_applied_with_override(phase):
    mgr = _FakeManager()
    layers = {0: _FakeLayer()}
    applied, rejected = apply_steering_updates(
        [_update(phase=phase)], mgr, layers, allow_cache_unsafe_phases=True
    )
    assert (applied, rejected) == (1, 0)
    assert mgr.calls[0][3] == phase


def test_invalid_phase_rejected():
    mgr = _FakeManager()
    applied, rejected = apply_steering_updates(
        [_update(phase="nonsense")], mgr, {0: _FakeLayer()}
    )
    assert (applied, rejected) == (0, 1)


def test_invalid_hook_rejected():
    mgr = _FakeManager()
    applied, rejected = apply_steering_updates(
        [_update(hook="not_a_hook")], mgr, {0: _FakeLayer()}
    )
    assert (applied, rejected) == (0, 1)
    assert mgr.calls == []


def test_unknown_layer_rejected():
    mgr = _FakeManager()
    applied, rejected = apply_steering_updates(
        [_update(layer=7)], mgr, {0: _FakeLayer()}
    )
    assert (applied, rejected) == (0, 1)


def test_hook_missing_on_layer_rejected():
    mgr = _FakeManager()
    layers = {0: _FakeLayer(hooks=("pre_attn",))}
    applied, rejected = apply_steering_updates([_update()], mgr, layers)
    assert (applied, rejected) == (0, 1)


def test_wrong_size_vector_rejected():
    mgr = _FakeManager()
    applied, rejected = apply_steering_updates(
        [_update(vec=np.ones(HIDDEN + 1, dtype=np.float32))],
        mgr,
        {0: _FakeLayer()},
    )
    assert (applied, rejected) == (0, 1)


def test_non_finite_vector_rejected():
    vec = np.ones(HIDDEN, dtype=np.float32)
    vec[3] = np.nan
    mgr = _FakeManager()
    applied, rejected = apply_steering_updates(
        [_update(vec=vec)], mgr, {0: _FakeLayer()}
    )
    assert (applied, rejected) == (0, 1)


def test_empty_vectors_rejected():
    mgr = _FakeManager()
    update = SteeringVectorUpdate(vectors={}, phase="decode")
    applied, rejected = apply_steering_updates([update], mgr, {0: _FakeLayer()})
    assert (applied, rejected) == (0, 1)


def test_bad_update_does_not_block_good_one():
    """Observer-isolation: one malformed update must not affect others."""
    mgr = _FakeManager()
    layers = {0: _FakeLayer()}
    bad = _update(layer=9)
    good = _update()
    applied, rejected = apply_steering_updates([bad, good], mgr, layers)
    assert (applied, rejected) == (1, 1)
    assert len(mgr.tier_calls) == 1  # decode update -> tier


def test_drain_stats_recorded_on_queue():
    q = SteeringActionQueue()
    q.submit(_update())
    q.submit(_update(layer=9))  # will be rejected at apply time
    mgr = _FakeManager()
    apply_steering_updates(q.drain(), mgr, {0: _FakeLayer()}, queue=q)
    stats = q.stats()
    assert stats.applied == 1
    assert stats.rejected == 1


def test_float64_input_cast_to_float32():
    mgr = _FakeManager()
    vec = np.ones(HIDDEN, dtype=np.float64)
    applied, _ = apply_steering_updates([_update(vec=vec)], mgr, {0: _FakeLayer()})
    assert applied == 1
    assert mgr.tier_calls[0][2].dtype == torch.float32  # decode update -> tier


# ---------------------------------------------------------------------------
# End-to-end against the real SteeringManager
# ---------------------------------------------------------------------------


class _RealSteerableLayer(torch.nn.Module):
    """Layer with a real registered steering-table buffer."""

    def __init__(self, num_rows: int, hidden_size: int):
        super().__init__()
        self.register_buffer(
            "steering_table_post_mlp", torch.zeros(num_rows, hidden_size)
        )


def _real_setup():
    from vllm.v1.worker.steering_manager import SteeringManager

    mgr = SteeringManager(max_steering_configs=4, device=None)
    layers = {0: _RealSteerableLayer(4 + 3, HIDDEN)}
    return mgr, layers


def test_real_manager_decode_update_lands_in_row_2():
    """Drained decode update must materialize in the global-decode row
    (row 2) of the layer's table after the populate path runs — the
    same sequence ``_update_steering_buffers`` performs each step."""
    mgr, layers = _real_setup()
    q = SteeringActionQueue()
    vec = np.linspace(0.0, 1.5, HIDDEN, dtype=np.float32)
    q.submit(_update(vec=vec))

    applied, rejected = apply_steering_updates(q.drain(), mgr, layers, queue=q)
    assert (applied, rejected) == (1, 0)
    assert mgr._tables_dirty

    mgr.populate_steering_tables(layers)
    table = layers[0].steering_table_post_mlp
    torch.testing.assert_close(table[2], torch.from_numpy(vec))
    # Prefill row and sentinel untouched.
    assert torch.all(table[0] == 0)
    assert torch.all(table[1] == 0)


def test_real_manager_zero_vector_disengages():
    """A follow-up zero-vector update must zero the decode row (the
    plugin's disengage emission)."""
    mgr, layers = _real_setup()
    apply_steering_updates(
        [_update(vec=np.ones(HIDDEN, dtype=np.float32))], mgr, layers
    )
    mgr.populate_steering_tables(layers)
    assert torch.all(layers[0].steering_table_post_mlp[2] == 1.0)

    apply_steering_updates(
        [_update(vec=np.zeros(HIDDEN, dtype=np.float32))], mgr, layers
    )
    mgr.populate_steering_tables(layers)
    assert torch.all(layers[0].steering_table_post_mlp[2] == 0.0)


def test_real_manager_decode_update_composes_with_per_request_config():
    """Rows 3+ combine global decode + per-request vectors; a dynamic
    global-decode update must flow into a decode-phase request row."""
    mgr, layers = _real_setup()
    req_vec = [2.0] * HIDDEN
    row = mgr.register_config(
        config_hash=7, vectors={"post_mlp": {0: req_vec}}, phase="decode"
    )
    apply_steering_updates(
        [_update(vec=np.ones(HIDDEN, dtype=np.float32))], mgr, layers
    )
    mgr.populate_steering_tables(layers)
    table = layers[0].steering_table_post_mlp
    # Request row = global decode (1.0) + per-request (2.0).
    assert torch.all(table[row] == 3.0)
