# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the live-server interpretability e2e suite.

These tests spawn one real ``vllm serve`` process (see ``conftest.py``)
and drive the interpretability surfaces over actual HTTP, closing the
gap between the mocked-engine router tests and the manual
``tests/gpu_*_validate.py`` scripts. Grading follows the scripts'
convention: greedy ``max_tokens=1`` completions of a fixed prompt,
scored by the answer token's logprob in the top-20 window.
"""

from __future__ import annotations

import base64
import os
from typing import Any

import numpy as np
import pytest
import torch

CLEAN = "The capital of France is"
ANSWER = " Paris"

MODEL = os.environ.get("INTERP_E2E_MODEL", "Qwen/Qwen3-0.6B")
LAYER = int(os.environ.get("INTERP_E2E_LAYER", "14"))
HIDDEN = int(os.environ.get("INTERP_E2E_HIDDEN", "1024"))
HOOK = "post_block"

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="live-server e2e needs a GPU to serve the model",
)


def unit(seed: int) -> np.ndarray:
    """A deterministic random unit vector of width ``HIDDEN``."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(HIDDEN)
    return v / np.linalg.norm(v)


def pack_hook_vectors(layer: int, row: np.ndarray, hook: str = HOOK) -> dict:
    """Pack one steering row into the binary-wire per-request shape."""
    row = np.ascontiguousarray(row[None, :], dtype=np.float32)
    return {
        hook: {
            "dtype": "float32",
            "shape": list(row.shape),
            "layer_indices": [layer],
            "data": base64.b64encode(row.tobytes()).decode(),
        }
    }


def clamp_entries(entries: list[dict], layer: int = LAYER) -> dict:
    """Wrap clamp entries into the ``{hook: {layer: entries}}`` wire shape."""
    return {HOOK: {str(layer): entries}}


def complete(http, prompt: str = CLEAN, max_tokens: int = 1, **extra: Any) -> dict:
    """Greedy completion returning the first choice; raises on non-2xx."""
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": 20,
        **extra,
    }
    r = http.post("/v1/completions", json=body, timeout=300.0)
    r.raise_for_status()
    return r.json()["choices"][0]


def answer_lp(choice: dict) -> float:
    """The ANSWER token's logprob at the first sampled position.

    Falls back to -50.0 when the answer left the top-20 window (the
    scripts' sentinel for "devastated").
    """
    return choice["logprobs"]["top_logprobs"][0].get(ANSWER, -50.0)
