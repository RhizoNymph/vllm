# SPDX-License-Identifier: Apache-2.0
"""Live Jacobian-lens readout consumer.

Streams per-token lens readouts ``topk(unembed(J_l @ h))`` to JSONL while the
model generates: a live "what is the model disposed to say at layer l" ticker
next to the completion. Direct ``CaptureSink`` implementation (like the
filesystem consumer) so rows are processed per forward step, not buffered
until request finish.

The lens transport ``J_l`` and the unembedding come from the jacobian-lens
fitting pipeline (goodfire-ai/MegaFire ``examples/jlens/``):

- lens file: ``{"J": {layer: [d, d]}, "n_prompts", "d_model"}``
- unembed bundle: ``{"norm_weight", "lm_head_weight", "embed_weight",
  "rms_norm_eps"}``

Enable (server):
    --capture-consumers "jlens:lens=/path/lens.pt,unembed=/path/unembed.pt,\
layers=30;40;50,topk=5,out=/path/readout,device=cuda:0"

Request (client, OpenAI extra fields):
    "capture": {"jlens": {"hooks": {"post_block": [30, 40, 50]},
                          "positions": "all_generated"}}

Output: ``{out}/{internal_request_id}.jsonl`` — one line per (step, layer):
``{"pos": [...], "layer": L, "top": [[[tok, ...k] per row]]}``; a final line
``{"event": "done", "client_request_id": ...}`` on request completion. Tail
the newest file for a live ticker.

Params:
    lens, unembed  (required paths)
    layers         ';'- or '|'-separated band, default "30;40;50" (',' splits
                   CLI params). Only these layers' J matrices are loaded.
    topk           default 5
    out            default /tmp/jlens_readout
    device         torch device for the readout matmuls, default cpu. Weights
                   are loaded lazily on the first chunk, so only the TP-rank-0
                   worker (the one that receives captures) pays the memory.
    tokenizer      HF dir for decoding top-k ids, default: the served model.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, ClassVar, Literal

import torch

from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureContext,
    CaptureFinalize,
    CaptureKey,
    CaptureResult,
    CaptureSpec,
)

_VALID_HOOKS = {"pre_attn", "post_attn", "post_block"}


class JLensReadoutConsumer:
    """Direct-sink consumer: per-step lens readouts to JSONL."""

    location: ClassVar[Literal["worker", "driver"]] = "worker"
    reads_client_spec: ClassVar[bool] = True
    execution: ClassVar[Literal["async", "sync"]] = "async"

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        self._lens_path = str(params["lens"])
        self._unembed_path = str(params["unembed"])
        raw_layers = str(params.get("layers", "30;40;50")).replace("|", ";")
        self._layers = sorted({int(x) for x in raw_layers.split(";") if x.strip()})
        self._topk = int(params.get("topk", 8))
        self._out = str(params.get("out", "/tmp/jlens_readout"))
        self._device = str(params.get("device", "cpu"))
        self._tokenizer_path = str(
            params.get("tokenizer") or vllm_config.model_config.tokenizer
        )
        os.makedirs(self._out, exist_ok=True)

        self._lock = threading.Lock()
        self._results: dict[CaptureKey, CaptureResult] = {}
        self._files: dict[str, Any] = {}  # req_str -> open file handle
        # Lazy weights: only the rank that actually receives chunks loads.
        self._loaded = False
        self._J: dict[int, torch.Tensor] = {}
        self._norm_w: torch.Tensor | None = None
        self._lm_head: torch.Tensor | None = None
        self._eps = 1e-5
        self._tok = None

    # ------------------------------------------------------------------
    # admission
    # ------------------------------------------------------------------

    def validate_client_spec(self, raw_spec: Any, ctx: CaptureContext) -> CaptureSpec:
        if not isinstance(raw_spec, dict):
            raise TypeError(f"jlens capture spec must be a dict, got {type(raw_spec)}")
        hooks = raw_spec.get("hooks") or {"post_block": self._layers}
        for hook, layers in hooks.items():
            if hook not in _VALID_HOOKS:
                raise ValueError(f"unknown hook {hook!r}")
            unknown = set(layers) - set(self._layers)
            if unknown:
                raise ValueError(
                    f"layers {sorted(unknown)} not in this consumer's loaded band "
                    f"{self._layers} (server --capture-consumers jlens:layers=...)"
                )
        positions = raw_spec.get("positions", "all")
        return CaptureSpec(hooks=hooks, positions=positions)

    # ------------------------------------------------------------------
    # lazy weight loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        bundle = torch.load(self._lens_path, map_location="cpu", weights_only=True)
        self._J = {
            l: bundle["J"][l].to(self._device, dtype=torch.float16)
            for l in self._layers
            if l in bundle["J"]
        }
        u = torch.load(self._unembed_path, map_location="cpu", weights_only=True)
        self._norm_w = u["norm_weight"].to(self._device, dtype=torch.float32)
        self._lm_head = u["lm_head_weight"].to(self._device, dtype=torch.float16)
        self._eps = float(u.get("rms_norm_eps", 1e-5))
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(
            self._tokenizer_path, trust_remote_code=True
        )
        self._loaded = True

    def _unembed_top(self, x: torch.Tensor) -> tuple[list[list[str]], list[list[float]]]:
        """x: [rows, d] residual-basis vectors -> per-row top-k (tokens, probs)."""
        # fp8 residuals can carry non-finite values; scrub before softmax so a
        # single inf/nan position can't poison the row into all-NaN probs (and
        # emit invalid `NaN` JSON tokens downstream).
        xf = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self._eps)
        logits = ((xf * self._norm_w).to(torch.float16) @ self._lm_head.T).float()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        probs = logits.softmax(dim=-1)
        top_p, top_i = probs.topk(self._topk, dim=-1)
        top_i, top_p = top_i.cpu(), top_p.cpu()
        toks = [[self._tok.decode([int(t)]) for t in row] for row in top_i]
        def _f(p):
            p = float(p)
            return round(p, 5) if p == p else 0.0  # NaN != NaN
        return toks, [[_f(p) for p in row] for row in top_p]

    def _readout(self, h: torch.Tensor, layer: int) -> dict:
        """Both lens readouts for one layer's rows.

        JACOBIAN_LENS transports by J_l first; layers without a fitted J
        (e.g. the final layer) fall back to identity, matching the reference
        convention that the final row is the model's own output. LOGIT_LENS
        is the bare unembed of the raw residual.
        """
        x = h.to(self._device, dtype=torch.float16)
        ll_t, ll_p = self._unembed_top(x)
        if layer in self._J:
            jl_t, jl_p = self._unembed_top(x @ self._J[layer].T)
        else:
            jl_t, jl_p = ll_t, ll_p
        return {"top": jl_t, "jl": {"t": jl_t, "p": jl_p}, "ll": {"t": ll_t, "p": ll_p}}

    # ------------------------------------------------------------------
    # CaptureSink protocol
    # ------------------------------------------------------------------

    def submit_chunk(self, chunk: CaptureChunk) -> None:
        try:
            self._ensure_loaded()
            req_str, layer_idx, hook = (
                str(chunk.key[0]),
                int(chunk.key[1]),
                str(chunk.key[2]),
            )
            readout = self._readout(chunk.tensor, layer_idx)
            positions = list(chunk.metadata.get("positions", []))
            line = json.dumps(
                {"pos": positions, "layer": layer_idx, "hook": hook, **readout},
                ensure_ascii=False,
            )
            with self._lock:
                fh = self._files.get(req_str)
                if fh is None:
                    fh = open(
                        os.path.join(self._out, f"{req_str}.jsonl"),
                        "a",
                        encoding="utf-8",
                    )
                    self._files[req_str] = fh
                fh.write(line + "\n")
                fh.flush()
        except Exception as exc:  # noqa: BLE001 — never take down the engine
            with self._lock:
                self._results[chunk.key] = CaptureResult(
                    key=chunk.key, status="error", error=str(exc)
                )

    def submit_finalize(self, finalize: CaptureFinalize) -> None:
        key = finalize.key
        req_str = str(key[0])
        with self._lock:
            fh = self._files.pop(req_str, None)
            if fh is not None:
                fh.write(
                    json.dumps(
                        {
                            "event": "done",
                            "client_request_id": finalize.sidecar.get(
                                "client_request_id"
                            ),
                        }
                    )
                    + "\n"
                )
                fh.close()
            if key not in self._results:
                self._results[key] = CaptureResult(key=key, status="ok")

    def get_result(self, key: CaptureKey) -> CaptureResult | None:
        with self._lock:
            return self._results.get(key)

    def shutdown(self, timeout: float = 30.0) -> None:
        with self._lock:
            for fh in self._files.values():
                fh.close()
            self._files.clear()
