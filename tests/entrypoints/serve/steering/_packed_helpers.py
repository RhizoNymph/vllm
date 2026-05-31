# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helper for building ``SteeringHookPacked`` payloads in tests.

Mirrors the per-request packing recipe at
``examples/online_serving/openai_steering_client.py`` so the global-set,
module-register, and JSON-loader test suites all build payloads the same
way they appear in the public client example.
"""

from __future__ import annotations

import numpy as np
import pybase64 as base64

_DEFAULT_PACK_DTYPE = np.dtype(np.float32)


def pack_hook(
    layer_vectors: dict[int, list[float]],
    *,
    dtype: np.dtype = _DEFAULT_PACK_DTYPE,
    scales: list[float] | None = None,
) -> dict:
    """Build a single ``SteeringHookPacked`` blob from per-layer Python lists.

    Layer order in the resulting blob matches sorted layer indices so
    tests can assert against deterministic ``layer_indices`` lists.
    """
    layer_indices = sorted(layer_vectors.keys())
    stacked = np.stack(
        [np.asarray(layer_vectors[i], dtype=dtype) for i in layer_indices],
        axis=0,
    )
    blob: dict = {
        "dtype": str(stacked.dtype),
        "shape": list(stacked.shape),
        "layer_indices": layer_indices,
        "data": base64.b64encode(stacked.tobytes()).decode("ascii"),
    }
    if scales is not None:
        blob["scales"] = scales
    return blob
