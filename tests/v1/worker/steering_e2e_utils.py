# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared scaffolding for the engine-level dynamic-steering e2e tests.

The ``test_*_steering_e2e.py`` files are standalone by design (no conftest
dependencies); each imports this module *before* torch so the spawn method
is set first.  Centralised here:

* model / layer resolution from ``DYNSTEER_E2E_MODEL`` / ``DYNSTEER_E2E_LAYER``;
* skip guards — CUDA, a missing local model path, an unavailable/gated HF
  repo (converted to a skip instead of an error), and uninstalled example
  capture-consumer plugins (``examples/capture_consumers/
  dynamic_steering_controller``);
* the shared ``LLM`` / ``AsyncLLM`` builder with the common e2e engine
  kwargs;
* the within-run target-vs-control helpers (``common_prefix_len``,
  ``NOISE_FLOOR``) — see ``test_dynamic_steering_e2e.py`` for why the
  technique is used.
"""

from __future__ import annotations

import importlib.metadata
import os

# The engine core spawns workers; the e2e tests touch CUDA in the parent
# (skip guards), so force spawn to avoid "Cannot re-initialize CUDA in
# forked subprocess". vLLM's conftest normally sets this, but these tests
# are run standalone.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pytest
import requests
import torch

MODEL = os.environ.get("DYNSTEER_E2E_MODEL", "google/gemma-4-E2B-it")
GPU_UTIL = float(os.environ.get("DYNSTEER_E2E_GPU_UTIL", "0.92"))
IS_LOCAL = MODEL.endswith(".gguf") or os.path.exists(MODEL)

PROMPT = "The capital of France is"
MAX_TOKENS = 24

# Greedy decoding of two identical prompts in one batch is NOT bitwise
# identical deep into generation: batched reductions use position-
# dependent orders, so the two diverge from pure FP noise after many
# tokens (~20+ observed on gemma4-31B). The e2e tests therefore compare
# the two outputs *within the same steered run* (target vs. in-batch
# control): real steering forces an EARLY divergence (~token 2), well
# separated from the late FP-noise floor. ``NOISE_FLOOR`` is the margin
# between the two regimes.
NOISE_FLOOR = 10

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="dynamic-steering e2e requires CUDA"
)

requires_model_path = pytest.mark.skipif(
    IS_LOCAL and not os.path.exists(MODEL),
    reason=f"DYNSTEER_E2E_MODEL path not found: {MODEL}",
)


def env_layer(default: int = 8) -> int:
    return int(os.environ.get("DYNSTEER_E2E_LAYER", str(default)))


def installed_capture_consumers() -> set[str]:
    try:
        eps = importlib.metadata.entry_points(group="vllm.capture_consumers")
    except Exception:
        return set()
    return {ep.name for ep in eps}


def requires_consumer_plugin(*names: str):
    """Skip when the named ``vllm.capture_consumers`` entry points are absent.

    The e2e stubs live in an out-of-tree example package; without it the
    engine raises ``UnknownCaptureConsumerError`` at ``LLM(...)``, which
    should read as a skip, not a failure.
    """
    missing = sorted(set(names) - installed_capture_consumers())
    return pytest.mark.skipif(
        bool(missing),
        reason=(
            f"capture-consumer entry point(s) not installed: {missing} — "
            "pip install examples/capture_consumers/dynamic_steering_controller"
        ),
    )


def skip_on_model_access_failure(exc: Exception, model: str) -> None:
    """Convert HF gated-repo / network failures into skips (else re-raise).

    Mirrors ``tests/models/language/generation/test_steering.py``: without
    this, a CUDA box with no HF access errors out of the default (gated)
    gemma-4 repo instead of skipping.
    """
    if isinstance(exc, requests.exceptions.RequestException):
        pytest.skip(f"{model} skipped due to model download timeout/error: {exc}")
    if isinstance(exc, OSError):
        msg = str(exc).lower()
        if (
            "gated repo" in msg
            or "connection error" in msg
            or "couldn't connect" in msg
            or "read timeout" in msg
            or "repository not found" in msg
            or "permission" in msg  # unwritable HF cache on shared boxes
        ):
            pytest.skip(f"{model} skipped due to model access error: {exc}")
    if isinstance(exc, TypeError):
        if "expected str, bytes or os.PathLike object, not NoneType" in str(exc):
            pytest.skip(f"{model} skipped due to model setup issue: {exc}")


def _engine_kwargs(
    capture_consumers: list | None,
    extra: dict | None,
    enforce_eager: bool,
) -> dict:
    kwargs: dict = dict(
        model=MODEL,
        enable_steering=True,
        max_dynamic_steering_configs=4,
        max_model_len=256,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=GPU_UTIL,
        seed=0,
    )
    if not IS_LOCAL:
        # Tiny dummy-weight gemma4 for CI / no-GGUF environments.
        kwargs["load_format"] = "dummy"
    if capture_consumers is not None:
        kwargs["capture_consumers"] = capture_consumers
    if extra:
        kwargs.update(extra)
    return kwargs


def build_llm(
    capture_consumers: list | None = None,
    *,
    extra: dict | None = None,
    enforce_eager: bool = True,
):
    """Build the shared e2e ``LLM``; model-access failures become skips."""
    from vllm import LLM

    kwargs = _engine_kwargs(capture_consumers, extra, enforce_eager)
    try:
        return LLM(**kwargs)
    except Exception as exc:
        skip_on_model_access_failure(exc, kwargs["model"])
        raise


def build_async_llm(*, extra: dict | None = None, enforce_eager: bool = True):
    """AsyncLLM variant of :func:`build_llm` (declarative-gates e2e)."""
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    kwargs = _engine_kwargs(None, extra, enforce_eager)
    try:
        return AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
    except Exception as exc:
        skip_on_model_access_failure(exc, kwargs["model"])
        raise


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n
