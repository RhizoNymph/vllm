# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the capture-aware piecewise cudagraph fallback.

When a per-request (client-spec) capture fires on a decode step, the runner
must keep the dynamic ``index_select`` gather off any cudagraph. Previously
this forced the *entire* step eager (``CUDAGraphMode.NONE``); the fallback
instead selects a PIECEWISE descriptor so every segment except the tapped
``vllm::capture_residual`` split region keeps replaying.

These tests cover the descriptor-selection logic only (pure CPU): that the
gate picks PIECEWISE rather than NONE when capture is active and a piecewise
graph exists, and that it correctly falls back to eager when none does.
"""

from __future__ import annotations

from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.config.vllm import (
    CAPTURE_RESIDUAL_SPLIT_OP,
    maybe_add_capture_split_op,
)
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp


def _make_manager(
    candidates: dict[tuple[int, int], list[BatchExecutionDescriptor]],
    captured: bool = True,
) -> CudaGraphManager:
    """Build a bare manager with just the fields ``dispatch``/
    ``dispatch_piecewise`` read, skipping the GPU-touching ``__init__``."""
    mgr = object.__new__(CudaGraphManager)
    mgr._candidates = candidates
    mgr._graphs_captured = captured
    mgr._lora_dispatch_map = {}
    mgr._max_lora_case = 0
    mgr.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    return mgr


def _full(num_tokens: int, num_reqs: int) -> BatchExecutionDescriptor:
    return BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=num_tokens, num_reqs=num_reqs
    )


def _piecewise(num_tokens: int) -> BatchExecutionDescriptor:
    # PIECEWISE descriptors carry num_reqs=None (no request padding).
    return BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=num_tokens, num_reqs=None
    )


# --------------------------------------------------------------------------
# dispatch_piecewise: candidate selection
# --------------------------------------------------------------------------


def test_dispatch_piecewise_returns_piecewise_candidate():
    # FULL_AND_PIECEWISE: both descriptors exist for this token count.
    mgr = _make_manager({(8, 0): [_full(8, 8), _piecewise(8)]})
    desc = mgr.dispatch_piecewise(
        num_reqs=4, num_tokens=8, uniform_token_count=2, num_active_loras=0
    )
    assert desc is not None
    assert desc.cg_mode == CUDAGraphMode.PIECEWISE


def test_dispatch_piecewise_none_when_only_full():
    # FULL-only mode has no piecewise candidate -> caller falls back to eager.
    mgr = _make_manager({(8, 0): [_full(8, 8)]})
    desc = mgr.dispatch_piecewise(
        num_reqs=4, num_tokens=8, uniform_token_count=2, num_active_loras=0
    )
    assert desc is None


def test_dispatch_piecewise_none_when_uncaptured_shape():
    mgr = _make_manager({(8, 0): [_piecewise(8)]})
    # No candidate registered for num_tokens=16.
    desc = mgr.dispatch_piecewise(
        num_reqs=4, num_tokens=16, uniform_token_count=4, num_active_loras=0
    )
    assert desc is None


def test_dispatch_piecewise_none_before_capture():
    mgr = _make_manager({(8, 0): [_piecewise(8)]}, captured=False)
    desc = mgr.dispatch_piecewise(
        num_reqs=4, num_tokens=8, uniform_token_count=2, num_active_loras=0
    )
    assert desc is None


# --------------------------------------------------------------------------
# dispatch_cg_and_sync_dp: capture-aware fallback (dp_size == 1)
# --------------------------------------------------------------------------


def test_fallback_selects_piecewise_not_none():
    mgr = _make_manager({(8, 0): [_full(8, 8), _piecewise(8)]})
    desc, across = dispatch_cg_and_sync_dp(
        mgr,
        num_reqs=4,
        num_tokens=8,
        uniform_token_count=2,
        dp_size=1,
        dp_rank=0,
        need_eager=True,
        capture_piecewise=True,
    )
    assert across is None
    assert desc.cg_mode == CUDAGraphMode.PIECEWISE


def test_fallback_to_eager_when_no_piecewise():
    # need_eager + capture_piecewise but only FULL captured -> eager (NONE).
    mgr = _make_manager({(8, 0): [_full(8, 8)]})
    desc, _ = dispatch_cg_and_sync_dp(
        mgr,
        num_reqs=4,
        num_tokens=8,
        uniform_token_count=2,
        dp_size=1,
        dp_rank=0,
        need_eager=True,
        capture_piecewise=True,
    )
    assert desc.cg_mode == CUDAGraphMode.NONE


def test_no_capture_piecewise_keeps_eager():
    # need_eager without the capture flag must still force NONE (e.g. profile).
    mgr = _make_manager({(8, 0): [_full(8, 8), _piecewise(8)]})
    desc, _ = dispatch_cg_and_sync_dp(
        mgr,
        num_reqs=4,
        num_tokens=8,
        uniform_token_count=2,
        dp_size=1,
        dp_rank=0,
        need_eager=True,
        capture_piecewise=False,
    )
    assert desc.cg_mode == CUDAGraphMode.NONE


def test_no_force_eager_uses_normal_dispatch():
    # Without need_eager the normal dispatch path runs (FULL for this shape).
    mgr = _make_manager({(8, 0): [_full(8, 8), _piecewise(8)]})
    desc, _ = dispatch_cg_and_sync_dp(
        mgr,
        num_reqs=4,
        num_tokens=8,
        uniform_token_count=2,
        dp_size=1,
        dp_rank=0,
        need_eager=False,
        capture_piecewise=False,
    )
    # dispatch() returns the first compatible candidate (FULL, listed first).
    assert desc.cg_mode == CUDAGraphMode.FULL


# --------------------------------------------------------------------------
# maybe_add_capture_split_op: splitting_ops registration
# --------------------------------------------------------------------------


def _compilation_config(
    cudagraph_mode: CUDAGraphMode,
    splitting_ops: list[str] | None,
    use_inductor_graph_partition: bool = False,
) -> CompilationConfig:
    cfg = CompilationConfig()
    cfg.cudagraph_mode = cudagraph_mode
    cfg.splitting_ops = splitting_ops
    cfg.use_inductor_graph_partition = use_inductor_graph_partition
    return cfg


def test_split_op_added_when_capture_and_piecewise():
    cfg = _compilation_config(
        CUDAGraphMode.FULL_AND_PIECEWISE,
        splitting_ops=["vllm::unified_attention_with_output"],
    )
    maybe_add_capture_split_op(cfg, capture_enabled=True)
    assert CAPTURE_RESIDUAL_SPLIT_OP in cfg.splitting_ops


def test_split_op_added_for_piecewise_only_mode():
    cfg = _compilation_config(
        CUDAGraphMode.PIECEWISE,
        splitting_ops=["vllm::unified_attention_with_output"],
    )
    maybe_add_capture_split_op(cfg, capture_enabled=True)
    assert CAPTURE_RESIDUAL_SPLIT_OP in cfg.splitting_ops


def test_split_op_not_added_without_capture():
    cfg = _compilation_config(
        CUDAGraphMode.FULL_AND_PIECEWISE,
        splitting_ops=["vllm::unified_attention_with_output"],
    )
    maybe_add_capture_split_op(cfg, capture_enabled=False)
    assert CAPTURE_RESIDUAL_SPLIT_OP not in cfg.splitting_ops


def test_split_op_not_added_for_full_only_mode():
    # FULL (no piecewise cudagraphs): the fallback can't help, don't split.
    cfg = _compilation_config(
        CUDAGraphMode.FULL,
        splitting_ops=[],
    )
    maybe_add_capture_split_op(cfg, capture_enabled=True)
    assert CAPTURE_RESIDUAL_SPLIT_OP not in cfg.splitting_ops


def test_split_op_not_added_with_inductor_partition():
    cfg = _compilation_config(
        CUDAGraphMode.FULL_AND_PIECEWISE,
        splitting_ops=["vllm::unified_attention_with_output"],
        use_inductor_graph_partition=True,
    )
    maybe_add_capture_split_op(cfg, capture_enabled=True)
    assert CAPTURE_RESIDUAL_SPLIT_OP not in cfg.splitting_ops


def test_split_op_idempotent():
    cfg = _compilation_config(
        CUDAGraphMode.FULL_AND_PIECEWISE,
        splitting_ops=["vllm::unified_attention_with_output"],
    )
    maybe_add_capture_split_op(cfg, capture_enabled=True)
    maybe_add_capture_split_op(cfg, capture_enabled=True)
    assert cfg.splitting_ops.count(CAPTURE_RESIDUAL_SPLIT_OP) == 1
