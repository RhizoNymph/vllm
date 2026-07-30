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

The row-gate case also runs in CUDA-graph mode: the in-graph gate write
must be visible to the replayed graph.

Requires CUDA + a tapped gemma4 (only gemma4 carries the steering hooks).
Skipped unless run manually against such a model:

    DYNSTEER_E2E_MODEL=/path/to/gemma-4-31B-it-Q4_K_S.gguf \
    DYNSTEER_E2E_LAYER=30 VLLM_USE_FLASHINFER_SAMPLER=0 \
    .venv/bin/python -m pytest tests/v1/worker/test_steering_gating_e2e.py -v -s
"""

from __future__ import annotations

from tests.v1.worker.steering_e2e_utils import (  # isort: skip
    MAX_TOKENS,
    NOISE_FLOOR,
    PROMPT,
    build_llm,
    common_prefix_len,
    env_layer,
    requires_consumer_plugin,
    requires_cuda,
    requires_model_path,
)

import pytest

LAYER = env_layer(8)


def _two_outputs(
    params: dict,
    *,
    enable_row_monitor: bool = False,
    enforce_eager: bool = True,
) -> tuple[list[int], list[int]]:
    from vllm import SamplingParams

    extra: dict = {}
    if enable_row_monitor:
        extra["enable_row_monitor"] = True
    llm = build_llm(
        [{"name": "dynamic_steering_e2e_cfg", "params": params}],
        extra=extra,
        enforce_eager=enforce_eager,
    )
    try:
        sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=0)
        outs = llm.generate([PROMPT, PROMPT], sp)
        return list(outs[0].outputs[0].token_ids), list(outs[1].outputs[0].token_ids)
    finally:
        del llm


_BASE = {
    "steer_layer": LAYER,
    "steer_hook": "post_block",
    "steer_norm": 24.0,
    "emit_after_steps": 1,
}


@requires_cuda
@requires_model_path
@requires_consumer_plugin("dynamic_steering_e2e_cfg")
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "cudagraph"])
def test_row_gate_gates_per_request_row(enforce_eager):
    """``gate_rows`` ON applies the target's row (early divergence); OFF
    suppresses it (target tracks the control to the noise floor)."""
    on_a, on_b = _two_outputs(
        {**_BASE, "mode": "rowgate", "gate_on": True}, enforce_eager=enforce_eager
    )
    on_diff = common_prefix_len(on_a, on_b)
    print(f"[gate ON]  first_diff={on_diff} a={on_a}\n           b={on_b}")
    assert on_a != on_b, "gate ON steered neither/both — row never applied"
    assert 1 <= on_diff <= NOISE_FLOOR, (
        f"gate ON: expected early steered divergence in [1,{NOISE_FLOOR}], "
        f"got {on_diff}"
    )

    off_a, off_b = _two_outputs(
        {**_BASE, "mode": "rowgate", "gate_on": False}, enforce_eager=enforce_eager
    )
    off_diff = common_prefix_len(off_a, off_b)
    print(f"[gate OFF] first_diff={off_diff} a={off_a}\n           b={off_b}")
    assert off_diff > NOISE_FLOOR, (
        f"gate OFF: row not suppressed — target diverged at {off_diff} "
        f"(expected > {NOISE_FLOOR}, i.e. unsteered-like)"
    )


@requires_cuda
@requires_model_path
@requires_consumer_plugin("dynamic_steering_e2e_cfg")
def test_req_id_scale_modulates_override_row():
    """``SteeringScaleUpdate(req_id=, scale=0)`` suppresses exactly the
    target's override row; the unscaled override run diverges early."""
    z_a, z_b = _two_outputs({**_BASE, "mode": "reqscale", "scale": 0.0})
    z_diff = common_prefix_len(z_a, z_b)
    print(f"[scale 0] first_diff={z_diff} a={z_a}\n          b={z_b}")
    assert z_diff > NOISE_FLOOR, (
        f"scale=0 did not suppress the row — target diverged at {z_diff} "
        f"(expected > {NOISE_FLOOR})"
    )

    o_a, o_b = _two_outputs({**_BASE, "mode": "override"})
    o_diff = common_prefix_len(o_a, o_b)
    print(f"[no scale] first_diff={o_diff} a={o_a}\n           b={o_b}")
    assert o_a != o_b and 1 <= o_diff <= NOISE_FLOOR, (
        f"unscaled override did not steer early: first_diff={o_diff}"
    )


@requires_cuda
@requires_model_path
@requires_consumer_plugin("dynamic_steering_e2e_cfg")
def test_per_row_monitor_gates_per_request_row():
    """The PER-ROW monitor (``SteeringMonitorUpdate(req_id=...)``,
    ``enable_row_monitor``) gates ONLY the target's override row by its own
    probe. Gate ON ⇒ the target's add is applied (early divergence from the
    in-batch control); gate OFF ⇒ the add is suppressed (target tracks the
    control to the noise floor). The control request, having no per-row
    monitor, is unaffected either way — isolating the per-request probe."""
    on_a, on_b = _two_outputs(
        {**_BASE, "mode": "perrow", "gate_on": True}, enable_row_monitor=True
    )
    on_diff = common_prefix_len(on_a, on_b)
    print(f"[perrow ON]  first_diff={on_diff} a={on_a}\n             b={on_b}")
    assert on_a != on_b, "per-row gate ON steered neither/both — row never applied"
    assert 1 <= on_diff <= NOISE_FLOOR, (
        f"per-row gate ON: expected early steered divergence in "
        f"[1,{NOISE_FLOOR}], got {on_diff}"
    )

    off_a, off_b = _two_outputs(
        {**_BASE, "mode": "perrow", "gate_on": False}, enable_row_monitor=True
    )
    off_diff = common_prefix_len(off_a, off_b)
    print(f"[perrow OFF] first_diff={off_diff} a={off_a}\n             b={off_b}")
    assert off_diff > NOISE_FLOOR, (
        f"per-row gate OFF: row not suppressed — target diverged at "
        f"{off_diff} (expected > {NOISE_FLOOR}, i.e. unsteered-like)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
