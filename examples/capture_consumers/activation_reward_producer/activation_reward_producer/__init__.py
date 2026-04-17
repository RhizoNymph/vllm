# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-alignment reward producer — a capture consumer that turns
residual-stream activations into a scalar reward by cosine-aligning them
to a pre-derived reference direction.

Implements ``CaptureSink`` directly (not ``CaptureConsumer``) so the
consumer can attach a diagnostic dict to ``CaptureResult.payload``;
``_BatchedAdapter`` hard-codes ``payload=None`` on success. See the
companion ``README.md`` for drift caveats and the frozen-scorer
deployment pattern.
"""

from __future__ import annotations

import math
import pathlib
import threading
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
import torch.nn.functional as F

from vllm.v1.capture.errors import CaptureValidationError
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureFinalize,
    CaptureKey,
    CaptureResult,
    CaptureSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.capture.types import CaptureContext


_HOOK_NAMES = frozenset(
    {"pre_attn", "post_attn", "post_mlp", "mlp_in", "mlp_out"}
)
_NONLIN = {
    "tanh": math.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-x)),
    "identity": lambda x: float(x),
}
_DTYPES = {"float32": torch.float32, "float64": torch.float64}


class ActivationRewardProducer:
    """Scalar-reward producer over residual-stream activations.

    Per-request opt-in only — the client lists this consumer's name (or
    ``instance_name``) in ``sampling_params.capture`` with an empty dict
    value. Layer, hook, reference vector, scale, and nonlinearity are
    pinned in engine config so clients cannot alter the training signal
    mid-run.

    Payload shape (dict on ``CaptureResult.payload``):

    - ``reward`` (float) — ``nonlin(scale * cos)``.
    - ``cos`` (float) — raw cosine similarity, pre-scale/pre-nonlinearity.
    - ``act_norm`` (float) — L2 norm of the mean activation.
    - ``num_positions`` (int) — how many generated rows made it through
      the configured slice.
    - ``status`` (str) — ``"ok"`` or ``"empty_window"``.
    """

    location: ClassVar[Literal["worker"]] = "worker"
    reads_client_spec: ClassVar[bool] = True

    def __init__(
        self,
        vllm_config: "VllmConfig",
        params: dict[str, Any],
    ) -> None:
        model_config = getattr(vllm_config, "model_config", None)
        hidden_size = (
            model_config.get_hidden_size()
            if model_config is not None
            else None
        )
        num_hidden_layers = (
            int(model_config.hf_config.num_hidden_layers)
            if model_config is not None
            else None
        )

        layer = int(params["layer"])
        if num_hidden_layers is not None and not (
            0 <= layer < num_hidden_layers
        ):
            raise ValueError(
                f"ActivationRewardProducer: layer={layer} out of range "
                f"[0, {num_hidden_layers})."
            )

        hook = str(params["hook"])
        if hook not in _HOOK_NAMES:
            raise ValueError(
                f"ActivationRewardProducer: hook={hook!r} not one of "
                f"{sorted(_HOOK_NAMES)}."
            )

        nonlin = str(params.get("nonlinearity", "tanh"))
        if nonlin not in _NONLIN:
            raise ValueError(
                f"ActivationRewardProducer: nonlinearity={nonlin!r} "
                f"not one of {sorted(_NONLIN)}."
            )

        dtype_name = str(params.get("dtype", "float32"))
        if dtype_name not in _DTYPES:
            raise ValueError(
                f"ActivationRewardProducer: dtype={dtype_name!r} "
                f"not one of {sorted(_DTYPES)}."
            )

        slice_cfg = dict(params.get("position_slice") or {})
        start = int(slice_cfg.get("start", 10))
        end = slice_cfg.get("end")
        stride = int(slice_cfg.get("stride", 1))
        if start < 0 or stride < 1 or (end is not None and int(end) < start):
            raise ValueError(
                "ActivationRewardProducer: position_slice must satisfy "
                "start >= 0, stride >= 1, end >= start."
            )

        vector_path = pathlib.Path(str(params["vector_path"]))
        vector = torch.load(
            vector_path, map_location="cpu", weights_only=True
        )
        if not isinstance(vector, torch.Tensor):
            raise TypeError(
                f"ActivationRewardProducer: vector_path={vector_path} "
                f"did not load a torch.Tensor (got {type(vector).__name__})."
            )
        if vector.dim() != 1:
            raise ValueError(
                f"ActivationRewardProducer: reference vector must be 1-D, "
                f"got shape {tuple(vector.shape)}."
            )
        if hidden_size is not None and vector.shape[0] != hidden_size:
            raise ValueError(
                f"ActivationRewardProducer: reference vector size "
                f"{vector.shape[0]} does not match model hidden_size "
                f"{hidden_size}."
            )

        compute_dtype = _DTYPES[dtype_name]
        vector = vector.to(dtype=compute_dtype)
        norm = vector.norm()
        if float(norm) == 0.0:
            raise ValueError(
                "ActivationRewardProducer: reference vector has zero norm."
            )
        vector = vector / norm

        self._layer = layer
        self._hook = hook
        self._v = vector
        self._scale = float(params.get("scale", 1.0))
        self._nonlin_fn = _NONLIN[nonlin]
        self._slice = slice(start, int(end) if end is not None else None, stride)
        self._compute_dtype = compute_dtype

        self._lock = threading.Lock()
        self._pending: dict[CaptureKey, list[tuple[int, torch.Tensor]]] = {}
        self._results: dict[CaptureKey, CaptureResult] = {}

    # ------------------------------------------------------------------
    # Consumer-API surface (for the registry and admission path)
    # ------------------------------------------------------------------

    def global_capture_spec(self) -> CaptureSpec | None:
        return None

    def validate_client_spec(
        self,
        raw_spec: Any,
        ctx: "CaptureContext",
    ) -> CaptureSpec:
        if ctx.tensor_parallel_size != 1 or ctx.pipeline_parallel_size != 1:
            raise CaptureValidationError(
                "ActivationRewardProducer requires "
                "tensor_parallel_size=1 and pipeline_parallel_size=1 "
                f"(got tp={ctx.tensor_parallel_size}, "
                f"pp={ctx.pipeline_parallel_size})."
            )
        if raw_spec not in (None, {}):
            raise CaptureValidationError(
                "ActivationRewardProducer accepts only an empty per-request "
                "spec (opt-in only). Layer, hook, vector, scale, and "
                "nonlinearity are pinned in engine config. "
                f"Got: {raw_spec!r}"
            )
        return CaptureSpec(
            hooks={self._hook: [self._layer]},
            positions="all_generated",
        )

    # ------------------------------------------------------------------
    # CaptureSink protocol
    # ------------------------------------------------------------------

    def submit_chunk(self, chunk: CaptureChunk) -> None:
        with self._lock:
            self._pending.setdefault(chunk.key, []).append(
                (chunk.row_offset, chunk.tensor)
            )

    def submit_finalize(self, finalize: CaptureFinalize) -> None:
        key = finalize.key
        with self._lock:
            buffered = self._pending.pop(key, None)

        if not buffered:
            with self._lock:
                self._results[key] = CaptureResult(
                    key=key,
                    status="partial_error",
                    error="no chunks received before finalize",
                )
            return

        ordered = sorted(buffered, key=lambda pair: pair[0])
        tensors = [t for _, t in ordered]
        tensor = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)

        try:
            payload = self._compute_reward(tensor)
        except Exception as exc:  # noqa: BLE001 — consumer isolation
            with self._lock:
                self._results[key] = CaptureResult(
                    key=key,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            return

        with self._lock:
            self._results[key] = CaptureResult(
                key=key, status="ok", payload=payload
            )

    def get_result(self, key: CaptureKey) -> CaptureResult | None:
        with self._lock:
            return self._results.get(key)

    def shutdown(self, timeout: float = 30.0) -> None:
        with self._lock:
            self._pending.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_reward(self, tensor: torch.Tensor) -> dict[str, Any]:
        sliced = tensor[self._slice]
        num_positions = int(sliced.shape[0])
        if num_positions == 0:
            return {
                "reward": float("nan"),
                "cos": float("nan"),
                "act_norm": float("nan"),
                "num_positions": 0,
                "status": "empty_window",
            }

        act = sliced.to(dtype=self._compute_dtype)
        mean_act = act.mean(dim=0)
        cos = F.cosine_similarity(mean_act, self._v, dim=0).item()
        reward = self._nonlin_fn(self._scale * cos)
        return {
            "reward": float(reward),
            "cos": float(cos),
            "act_norm": float(mean_act.norm().item()),
            "num_positions": num_positions,
            "status": "ok",
        }
