# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level e2e for two per-request dynamic-steering knobs that the
unit/op tests prove only in isolation: the in-graph **row gate**
(``gate_rows``) and the **req_id-keyed scale**. Each is validated end to
end through a real ``LLM`` + a sync capture consumer, using the
within-run target-vs-control technique (so it is robust to the batched-FP
nondeterminism that defeats cross-run output comparison — see
``test_dynamic_steering_e2e.py``).

The :class:`ConfigurableOverrideStub` steers the first request it sees in
decode (the other identical request is the in-batch control) and emits a
companion action in the same step:

- **rowgate**: an override row + a ``gate_rows`` monitor whose threshold is
  saturated to force the per-token gate fully ON or OFF for any residual.
  Gate ON ⇒ the target's row is applied (target diverges from the control
  early); gate OFF ⇒ the row is suppressed (target tracks the control to
  the FP-noise floor). The contrast isolates the row gate.
- **reqscale**: an override row + a ``SteeringScaleUpdate(req_id=...)``.
  ``scale=0`` suppresses the target's row (target ≈ control); the no-scale
  override run diverges. The contrast isolates the req_id→dyn_id scale path.

Requires CUDA + a tapped gemma4 (only gemma4 carries the steering hooks).
Skipped unless run manually against such a model:

    DYNSTEER_E2E_MODEL=/path/to/gemma-4-31B-it-Q4_K_S.gguf \
    DYNSTEER_E2E_LAYER=30 VLLM_USE_FLASHINFER_SAMPLER=0 \
    .venv/bin/python -m pytest tests/v1/worker/test_steering_gating_e2e.py -v -s
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pytest
import torch

MODEL = os.environ.get("DYNSTEER_E2E_MODEL", "google/gemma-4-E2B-it")
LAYER = int(os.environ.get("DYNSTEER_E2E_LAYER", "8"))
IS_LOCAL = MODEL.endswith(".gguf") or os.path.exists(MODEL)

PROMPT = "The capital of France is"
MAX_TOKENS = 24

# Real per-request steering forces an EARLY divergence between the target
# and the in-batch control; two identical prompts left unsteered only
# diverge much later from batched-FP noise. NOISE_FLOOR separates the two
# regimes (see test_dynamic_steering_e2e.py).
NOISE_FLOOR = 10


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _build_llm(params: dict):
    from vllm import LLM

    kwargs: dict = dict(
        model=MODEL,
        enable_steering=True,
        max_dynamic_steering_configs=4,
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.92,
        seed=0,
        capture_consumers=[{"name": "dynamic_steering_e2e_cfg", "params": params}],
    )
    if not IS_LOCAL:
        kwargs["load_format"] = "dummy"
    return LLM(**kwargs)


def _two_outputs(params: dict) -> tuple[list[int], list[int]]:
    from vllm import SamplingParams

    llm = _build_llm(params)
    try:
        sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=0)
        outs = llm.generate([PROMPT, PROMPT], sp)
        return list(outs[0].outputs[0].token_ids), list(outs[1].outputs[0].token_ids)
    finally:
        del llm


_BASE = {"steer_layer": LAYER, "steer_hook": "post_block", "steer_norm": 24.0,
         "emit_after_steps": 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(
    IS_LOCAL and not os.path.exists(MODEL),
    reason=f"DYNSTEER_E2E_MODEL path not found: {MODEL}",
)
def test_row_gate_gates_per_request_row():
    """``gate_rows`` ON applies the target's row (early divergence); OFF
    suppresses it (target tracks the control to the noise floor)."""
    on_a, on_b = _two_outputs({**_BASE, "mode": "rowgate", "gate_on": True})
    on_diff = _common_prefix_len(on_a, on_b)
    print(f"[gate ON]  first_diff={on_diff} a={on_a}\n           b={on_b}")
    assert on_a != on_b, "gate ON steered neither/both — row never applied"
    assert 1 <= on_diff <= NOISE_FLOOR, (
        f"gate ON: expected early steered divergence in [1,{NOISE_FLOOR}], "
        f"got {on_diff}"
    )

    off_a, off_b = _two_outputs({**_BASE, "mode": "rowgate", "gate_on": False})
    off_diff = _common_prefix_len(off_a, off_b)
    print(f"[gate OFF] first_diff={off_diff} a={off_a}\n           b={off_b}")
    assert off_diff > NOISE_FLOOR, (
        f"gate OFF: row not suppressed — target diverged at {off_diff} "
        f"(expected > {NOISE_FLOOR}, i.e. unsteered-like)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(
    IS_LOCAL and not os.path.exists(MODEL),
    reason=f"DYNSTEER_E2E_MODEL path not found: {MODEL}",
)
def test_req_id_scale_modulates_override_row():
    """``SteeringScaleUpdate(req_id=, scale=0)`` suppresses exactly the
    target's override row; the unscaled override run diverges early."""
    z_a, z_b = _two_outputs({**_BASE, "mode": "reqscale", "scale": 0.0})
    z_diff = _common_prefix_len(z_a, z_b)
    print(f"[scale 0] first_diff={z_diff} a={z_a}\n          b={z_b}")
    assert z_diff > NOISE_FLOOR, (
        f"scale=0 did not suppress the row — target diverged at {z_diff} "
        f"(expected > {NOISE_FLOOR})"
    )

    o_a, o_b = _two_outputs({**_BASE, "mode": "override"})
    o_diff = _common_prefix_len(o_a, o_b)
    print(f"[no scale] first_diff={o_diff} a={o_a}\n           b={o_b}")
    assert o_a != o_b and 1 <= o_diff <= NOISE_FLOOR, (
        f"unscaled override did not steer early: first_diff={o_diff}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
