# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level e2e for the opt-in **cross-layer** in-graph monitor (§8): a
probe at layer L gates steering at layers > L in the same forward ("detect at
L, gate at layers ≥ L"). The unit tests prove the branch wiring; this proves
the cross-layer reach AND the graph ordering end to end on a real ``LLM``.

Setup: ``enable_cross_layer_monitor=True`` + a strong dynamic **tier** vector at
``TIER_LAYER`` (post_mlp). A saturating gate-OFF monitor (probe=0,
threshold=+1e6 ⇒ gate≈0) multiplies the shared ``token_scales`` the tier reads.

- monitor BEFORE the tier ⇒ the gate (written at L < tier) reaches the tier ⇒
  tier SUPPRESSED ⇒ output == unsteered baseline.
- monitor AFTER the tier ⇒ the gate (written at L > tier) cannot reach the tier
  (graph order) ⇒ tier NOT suppressed ⇒ output == steered.

Greedy single request ⇒ deterministic given steering state, so token-id
sequences are compared directly (no FP-noise floor needed). All configs run in
one process; the manager is reconfigured between generates (in-proc engine).

Requires CUDA + a local tapped gemma4 with enough layers (tier@40 needs ≥~50).
Skipped unless run manually:

    DYNSTEER_E2E_MODEL=/path/to/gemma-4-31B-it-Q4_K_S.gguf DYNSTEER_E2E_LAYER=40 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    .venv/bin/python -m pytest tests/v1/worker/test_cross_layer_monitor_e2e.py -v -s
"""

from __future__ import annotations

import os

# Direct-manager access (below) needs the in-process engine.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import numpy as np
import pytest
import torch

MODEL = os.environ.get("DYNSTEER_E2E_MODEL", "google/gemma-4-E2B-it")
TIER_LAYER = int(os.environ.get("DYNSTEER_E2E_LAYER", "40"))
IS_LOCAL = MODEL.endswith(".gguf") or os.path.exists(MODEL)
HOOK = "post_mlp"
PROMPT = "The capital of France is"
MAX_TOKENS = 32

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not IS_LOCAL,
    reason="needs CUDA + a local tapped gemma4 with ≥~50 layers "
    "(set DYNSTEER_E2E_MODEL to a local gemma-4 GGUF)",
)


def _runner(llm):
    return llm.llm_engine.model_executor.driver_worker.worker.model_runner


def test_cross_layer_monitor_gates_later_layers_only():
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        enable_steering=True,
        enable_cross_layer_monitor=True,
        max_model_len=3072,
        max_num_seqs=32,
        gpu_memory_utilization=0.92,
        seed=0,
    )
    mr = _runner(llm)
    mgr = mr._steering_manager
    assert mgr is not None, "steering manager not initialised"
    owned = mr._locally_owned_layers
    layers = sorted(mr._steerable_layers_cache)
    # opt-in stamped onto the layers as a trace-time constant
    assert mr._steerable_layers_cache[layers[0]]._cross_layer_monitor

    mon_before = max(layers[0], TIER_LAYER - (TIER_LAYER - layers[0]) // 2)
    mon_after = min(layers[-1], TIER_LAYER + (layers[-1] - TIER_LAYER) // 2)
    assert mon_before < TIER_LAYER < mon_after, (
        f"need layers straddling {TIER_LAYER}; got {layers[0]}..{layers[-1]}"
    )

    hidden = getattr(
        mr._steerable_layers_cache[TIER_LAYER], f"steering_table_{HOOK}"
    ).shape[1]
    v = np.random.default_rng(0).standard_normal(hidden).astype(np.float32)
    v = v / float(np.linalg.norm(v)) * 12.0
    tier_vec = torch.from_numpy(v)

    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=0)

    def gen():
        return list(llm.generate([PROMPT], sp)[0].outputs[0].token_ids)

    def set_tier(on: bool):
        if on:
            mgr.update_dynamic_tier(
                HOOK, TIER_LAYER, tier_vec, locally_owned_layers=owned
            )
            mgr.set_dynamic_tier_gain(1.0)
        else:
            mgr.clear_dynamic_tier()

    def set_monitor(layer: int | None, thr: float):
        mgr.clear_monitor()
        if layer is not None:
            mgr.set_monitor(
                HOOK, layer, torch.zeros(hidden), threshold=thr,
                sharpness=1.0, gate_rows=False, locally_owned_layers=owned,
            )

    set_tier(False)
    set_monitor(None, 0.0)
    base = gen()
    set_tier(True)
    set_monitor(None, 0.0)
    steered = gen()
    set_tier(True)
    set_monitor(mon_before, 1e6)
    before = gen()
    set_tier(True)
    set_monitor(mon_after, 1e6)
    after = gen()

    # tier actually steers
    assert steered != base, "tier vector had no effect on output"
    # cross-layer reach: gate written before the tier suppresses it
    assert before == base, (
        f"gate@{mon_before} did not suppress tier@{TIER_LAYER} "
        f"(cross-layer reach broken): {before[:6]} vs base {base[:6]}"
    )
    # graph ordering: gate written after the tier cannot suppress it
    assert after == steered, (
        f"gate@{mon_after} wrongly affected tier@{TIER_LAYER} "
        f"(ordering broken): {after[:6]} vs steered {steered[:6]}"
    )
