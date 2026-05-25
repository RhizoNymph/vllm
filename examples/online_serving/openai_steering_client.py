# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request steering through the OpenAI-compatible server.

The HTTP steering fields use a binary wire format
(see ``vllm.config.steering_types.SteeringHookPacked``): each hook carries
one base64-encoded ``(num_layers, hidden_size)`` blob plus a sibling
``layer_indices`` list.  The server unpacks via ``np.frombuffer``
(microseconds vs. the ~10–15 ms per request a JSON ``list[float]`` payload
would cost on the API-server event loop).

Start the server with per-request steering enabled:

    vllm serve google/gemma-3-4b-it \\
        --enable-steering \\
        --max-steering-configs 4

Then run this script:

    python examples/online_serving/openai_steering_client.py
"""

import numpy as np
import pybase64 as base64
from openai import OpenAI

MODEL = "google/gemma-3-4b-it"
HIDDEN_SIZE = 2560  # gemma-3-4b-it residual width

# Match the model's compute dtype.  np.float16 / np.float32 / np.float64 are
# supported; bfloat16 weights fall back to float32 on the server side (numpy
# lacks a native bf16), still a ~2.25x IPC reduction vs JSON floats.
PACK_DTYPE = np.float16


_DEFAULT_PACK_DTYPE = np.dtype(PACK_DTYPE)


def pack_hook(
    layer_vectors: dict[int, np.ndarray],
    *,
    dtype: np.dtype = _DEFAULT_PACK_DTYPE,
    scales: dict[int, float] | None = None,
) -> dict:
    """Pack one hook's per-layer vectors into a ``SteeringHookPacked`` dict.

    ``layer_vectors`` maps layer index to a 1-D vector of length
    ``hidden_size``.  Rows are stacked in layer-index order; ``layer_indices``
    records the original mapping so the server can scatter back per-layer.

    When ``scales`` is given, each row carries a per-layer scalar that the
    server multiplies in at decode time — mirrors the in-process
    ``{"vector": [...], "scale": float}`` form available to
    ``SamplingParams`` without baking the multiplier into the bytes.
    """
    if not layer_vectors:
        raise ValueError("layer_vectors must be non-empty")
    layer_indices = sorted(layer_vectors.keys())
    stacked = np.stack(
        [np.asarray(layer_vectors[i], dtype=dtype) for i in layer_indices],
        axis=0,
    )
    if stacked.ndim != 2:
        raise ValueError(f"expected 2-D stacked array, got shape {stacked.shape}")
    blob: dict = {
        "dtype": str(stacked.dtype),
        "shape": list(stacked.shape),
        "layer_indices": layer_indices,
        "data": base64.b64encode(stacked.tobytes()).decode("ascii"),
    }
    if scales is not None:
        blob["scales"] = [float(scales.get(i, 1.0)) for i in layer_indices]
    return blob


def main() -> None:
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

    rng = np.random.default_rng(seed=0)

    # Base steering applied to both prefill and decode.
    base = {
        "post_mlp": pack_hook(
            {15: rng.standard_normal(HIDDEN_SIZE).astype(PACK_DTYPE)},
            # Per-layer scales: the server multiplies row-by-row without
            # re-encoding the bytes, so the same vector can be reused at
            # different strengths across requests.
            scales={15: 2.0},
        ),
    }

    # Prefill-only addition (applied during prefill in addition to base).
    prefill_only = {
        "pre_attn": pack_hook(
            {15: rng.standard_normal(HIDDEN_SIZE).astype(PACK_DTYPE)},
        ),
    }

    # Decode-only addition (applied during decode in addition to base).
    decode_only = {
        "pre_attn": pack_hook(
            {15: rng.standard_normal(HIDDEN_SIZE).astype(PACK_DTYPE)},
        ),
    }

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64,
        temperature=0.0,
        extra_body={
            "steering_vectors": base,
            "prefill_steering_vectors": prefill_only,
            "decode_steering_vectors": decode_only,
        },
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
