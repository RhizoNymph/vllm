# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend helpers for declarative per-request steering."""

from __future__ import annotations

from typing import Any


def monitor_writes_gates_from_request(raw_request: Any | None) -> bool | None:
    """Return the engine's ``enable_cross_layer_monitor`` for gate validation.

    Per-request clamp-target gates are rejected in every engine mode today
    (see :class:`vllm.v1.steering_schema.ClampApply`); the serving handlers
    read this flag off ``app.state`` and thread it to
    :func:`vllm.v1.steering_schema.build_steering_gates` so the HTTP 400
    error is mode-tailored (and so materialized-mode acceptance can later be
    enabled in :func:`~vllm.v1.steering_schema.validate_clamp_gate_support`
    without touching the handlers).

    Args:
        raw_request: The FastAPI request, or ``None`` (offline / batch).

    Returns:
        ``True``/``False`` when the steering config is reachable, else
        ``None`` (mode unknown; the unconditional rejection still applies).
    """
    if raw_request is None:
        return None
    vllm_config = getattr(raw_request.app.state, "vllm_config", None)
    steering_config = getattr(vllm_config, "steering_config", None)
    if steering_config is None:
        return None
    return bool(getattr(steering_config, "enable_cross_layer_monitor", False))
