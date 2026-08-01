# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level e2e for **declarative per-request steering gates** — the
client-facing feature (``when × scope × apply`` gates carried on
``RequestMetadata.steering``; see :mod:`vllm.v1.steering_schema`). The
unit/op tests prove gate resolution and the substrate wiring in isolation;
this drives a real engine end to end so the built-in declarative consumer,
its in-graph per-request row monitor, and request-finish cleanup are all
exercised on GPU.

Reachability note: ``RequestMetadata.steering`` is only threaded to the
worker by :meth:`AsyncLLM.generate` (and thus the OpenAI HTTP frontend);
the offline synchronous ``LLM``/``LLMEngine.add_request`` path has no
``request_metadata`` parameter, so declarative gates cannot be attached
through ``LLM.generate``. This test therefore drives ``AsyncLLM``
in-process (the minimal offline surface that can carry the gates), attaching
gates with **inline** vector sources so no server-side vector registry is
required.

Each property is asserted with the within-run target-vs-control technique
used by ``test_dynamic_steering_e2e.py``: two identical prompts run in one
batch, only one carrying a gate. Real steering forces an EARLY divergence
between the two, well separated from the late batched-FP-noise floor that
identical prompts hit much later.

Covers:

1. ``always × rest_of_request × add`` steers ONLY the tagged request (the
   in-batch control, carrying no gate, is unaffected).
2. ``probe × this_token × add`` with a saturated-ON threshold steers
   (proving the per-request in-graph row monitor engages), and with a
   saturated-OFF threshold does not (the gate suppresses the add).
3. Request finish cleans up: a later untagged pair on the same engine is
   unsteered (no override leaks past the tagged request's lifetime).

Requires CUDA + a tapped gemma4 (only gemma4 carries the steering hooks).
Skipped unless run manually against such a model:

    DYNSTEER_E2E_MODEL=/path/to/gemma-4-31B-it-Q4_K_S.gguf \
    DYNSTEER_E2E_LAYER=30 VLLM_USE_FLASHINFER_SAMPLER=0 \
    .venv/bin/python -m pytest tests/v1/worker/test_declarative_gates_e2e.py -v -s
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import pytest
import torch

MODEL = os.environ.get("DYNSTEER_E2E_MODEL", "google/gemma-4-E2B-it")
LAYER = int(os.environ.get("DYNSTEER_E2E_LAYER", "8"))
IS_LOCAL = MODEL.endswith(".gguf") or os.path.exists(MODEL)

HOOK = "post_block"
PROMPT = "The capital of France is"
MAX_TOKENS = 24
STEER_NORM = 24.0

# Real per-request steering forces an EARLY divergence between the tagged
# request and its in-batch control; two identical prompts left unsteered
# only diverge much later from batched-FP noise. NOISE_FLOOR separates the
# two regimes (see test_dynamic_steering_e2e.py).
NOISE_FLOOR = 10

_rid = itertools.count()


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _packed(vec: np.ndarray, layer: int) -> dict:
    """Pack a single-row vector into the ``SteeringHookPacked`` wire shape."""
    v = np.ascontiguousarray(vec, dtype=np.float32)
    return {
        HOOK: {
            "dtype": "float32",
            "shape": [1, int(v.shape[0])],
            "layer_indices": [layer],
            "data": base64.b64encode(v.tobytes()).decode("ascii"),
        }
    }


def _unit(seed: int, hidden: int, norm: float) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(hidden).astype(np.float32)
    v /= float(np.linalg.norm(v))
    return (v * norm).astype(np.float32)


def _inline(vec: np.ndarray, layer: int) -> dict:
    return {"kind": "inline", "packed": _packed(vec, layer)}


def _always_add_gate(vec: np.ndarray, layer: int) -> list:
    from vllm.v1.steering_schema import build_steering_gates

    return build_steering_gates(
        [
            {
                "when": {"kind": "always"},
                "scope": "rest_of_request",
                "apply": {"kind": "add", "steer": _inline(vec, layer), "strength": 1.0},
            }
        ],
        None,
    )


def _probe_this_token_gate(
    vec: np.ndarray, probe: np.ndarray, layer: int, *, gate_on: bool
) -> list:
    from vllm.v1.steering_schema import build_steering_gates

    # Saturate the sigmoid so the gate is a deterministic 1 (gate_on) or 0.
    threshold = -1.0e6 if gate_on else 1.0e6
    return build_steering_gates(
        [
            {
                "when": {
                    "kind": "probe",
                    "probe": _inline(probe, layer),
                    "threshold": threshold,
                    "sharpness": 1.0,
                },
                "scope": "this_token",
                "apply": {"kind": "add", "steer": _inline(vec, layer), "strength": 1.0},
            }
        ],
        None,
    )


def _engine():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    kwargs: dict = dict(
        model=MODEL,
        enable_steering=True,
        max_dynamic_steering_configs=4,
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.92,
        seed=0,
    )
    if not IS_LOCAL:
        kwargs["load_format"] = "dummy"
    return AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))


async def _gen(engine, prompt: str, gates) -> list[int]:
    from vllm import SamplingParams
    from vllm.v1.request_metadata import RequestMetadata

    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=0)
    rm = RequestMetadata(steering=gates) if gates else None
    last = None
    async for out in engine.generate(
        request_id=f"req-{next(_rid)}",
        prompt=prompt,
        sampling_params=sp,
        request_metadata=rm,
    ):
        last = out
    assert last is not None
    return list(last.outputs[0].token_ids)


async def _pair(engine, target_gates) -> tuple[list[int], list[int]]:
    """Run the tagged (target) request and an untagged control in one batch."""
    target, control = await asyncio.gather(
        _gen(engine, PROMPT, target_gates),
        _gen(engine, PROMPT, None),
    )
    return target, control


async def _run_all() -> dict:
    engine = _engine()
    try:
        hidden = engine.vllm_config.model_config.get_hidden_size()
        steer = _unit(0, hidden, STEER_NORM)
        probe = _unit(1, hidden, 1.0)

        out: dict = {}
        out["always"] = await _pair(engine, _always_add_gate(steer, LAYER))
        out["probe_on"] = await _pair(
            engine, _probe_this_token_gate(steer, probe, LAYER, gate_on=True)
        )
        out["probe_off"] = await _pair(
            engine, _probe_this_token_gate(steer, probe, LAYER, gate_on=False)
        )
        # Cleanup: after the gated requests have finished, an untagged pair
        # must be fully unsteered (no override leaked past request finish).
        out["cleanup"] = await asyncio.gather(
            _gen(engine, PROMPT, None), _gen(engine, PROMPT, None)
        )
        return out
    finally:
        engine.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(
    IS_LOCAL and not os.path.exists(MODEL),
    reason=f"DYNSTEER_E2E_MODEL path not found: {MODEL}",
)
def test_declarative_gates_steer_target_only_and_clean_up():
    results = asyncio.run(_run_all())

    # 1) always x rest_of_request x add: only the tagged request is steered.
    t, c = results["always"]
    diff = _common_prefix_len(t, c)
    print(f"[always]    first_diff={diff}\n  target ={t}\n  control={c}")
    assert t != c, "always-add gate steered neither/both request"
    assert diff <= NOISE_FLOOR, (
        f"always-add divergence at token {diff} looks like FP noise, not "
        f"steering (expected <= {NOISE_FLOOR})"
    )

    # 2a) probe x this_token x add, gate saturated ON: steers (row monitor
    # engages in-graph).
    on_t, on_c = results["probe_on"]
    on_diff = _common_prefix_len(on_t, on_c)
    print(f"[probe ON]  first_diff={on_diff}\n  target ={on_t}\n  control={on_c}")
    assert on_t != on_c, "probe this_token gate ON steered neither/both"
    assert on_diff <= NOISE_FLOOR, (
        f"probe ON divergence at token {on_diff} looks like FP noise, not "
        f"steering (expected <= {NOISE_FLOOR})"
    )

    # 2b) same gate saturated OFF: the gate suppresses the add, so the target
    # tracks the control to the noise floor.
    off_t, off_c = results["probe_off"]
    off_diff = _common_prefix_len(off_t, off_c)
    print(f"[probe OFF] first_diff={off_diff}\n  target ={off_t}\n  control={off_c}")
    assert off_diff > NOISE_FLOOR, (
        f"probe OFF: gate did not suppress the add — target diverged at "
        f"{off_diff} (expected > {NOISE_FLOOR}, i.e. unsteered-like)"
    )

    # 3) cleanup: a later untagged pair is fully unsteered (no leak).
    cl_a, cl_b = results["cleanup"]
    cl_diff = _common_prefix_len(cl_a, cl_b)
    print(f"[cleanup]   first_diff={cl_diff}\n  a={cl_a}\n  b={cl_b}")
    assert cl_diff > NOISE_FLOOR, (
        f"untagged requests diverged early (first_diff={cl_diff}) — a prior "
        f"gate's override leaked past request finish"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
