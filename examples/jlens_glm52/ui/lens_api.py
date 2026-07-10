"""Neuronpedia lens-API adapter: POST /api/lens/prompt -> NDJSON stream.

Implements the contract the Neuronpedia jlens UI speaks (camelCase
LensPromptRequest in; ``meta`` -> ``prompt`` -> ``token``* -> ``done``
NDJSON out) on top of this fork's serving stack: generation via
/v1/completions with token-id prompts, per-position readouts via the jlens
capture consumer's JSONL stream, steering via the fork's packed
per-request steering vectors.

v1 simplifications (documented, all degrade gracefully in the UI):
  - ``reuse_len`` is always 0 (no prefix cache; the client re-receives
    every position).
  - ``steerAblate`` / ``swapToken`` are rejected with an in-stream error
    (the backend supports additive steering only today).
  - ``filterNonWordTokens`` is accepted and ignored (readouts are raw
    top-k).
  - steering strength scales the per-layer MEAN residual norm, not the
    per-position norm.
"""

from __future__ import annotations

import asyncio
import base64
import glob
import json
import os
import time
import uuid
from typing import Any

import httpx
import numpy as np

VOCAB_SIZE = 154880
FINAL_LAYER = 77


class LensAPI:
    """Bound to the sidecar's Lens (weights + tokenizer) at startup."""

    def __init__(self, lens, server_base_fn, readout_dir: str, layers: list[int]):
        self.lens = lens
        self._server_base = server_base_fn
        self._readout_dir = readout_dir
        # capture layers, ascending, final layer always present
        self.layers = sorted({*layers, FINAL_LAYER})

    # -- request pieces ----------------------------------------------------

    def _build_ids(self, body: dict) -> tuple[list[int], int]:
        """Token ids + completion-token budget from the request."""
        n_gen = int(body.get("numCompletionTokens", 128))
        if body.get("inputTokenIds"):
            return [int(t) for t in body["inputTokenIds"]], 0
        tok = self.lens.tok
        if body.get("chat"):
            chat = [
                {"role": m["role"], "content": m["content"]} for m in body["chat"]
            ]
            prefill = chat[-1]["role"] == "assistant"
            ids = tok.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=not prefill,
                continue_final_message=prefill,
                enable_thinking=bool(body.get("enableThinking", False)),
            )
            return list(ids), n_gen
        text = body.get("prompt") or ""
        return list(tok(text).input_ids), n_gen

    def _steering_payload(self, body: dict) -> dict:
        """Map Neuronpedia steer fields onto packed steering vectors."""
        tokens = body.get("steerTokens") or []
        if not tokens:
            return {}
        if body.get("steerAblate") or body.get("swapToken"):
            raise ValueError(
                "ablate/swap interventions are not supported on this backend "
                "yet - additive steering only"
            )
        layers = [int(l) for l in (body.get("steerLayers") or [])] or [40]
        strength = float(body.get("steerStrength", -0.1))
        tok = self.lens.tok
        rows, idxs = [], []
        for layer in layers:
            if layer not in self.lens.J and layer != FINAL_LAYER:
                continue
            acc = None
            for st in tokens:
                tid = tok(st["token"], add_special_tokens=False).input_ids[0]
                w = self.lens.norm_w * self.lens.lm_head[tid]
                if st.get("type") == "LOGIT_LENS" or layer not in self.lens.J:
                    d = w
                else:
                    d = w @ self.lens.J[layer]
                d = d / d.norm()
                acc = d if acc is None else acc + d
            acc = acc / acc.norm() * self.lens.norms.get(layer, 1.0)
            rows.append(acc.numpy().astype("<f4"))
            idxs.append(layer)
        if not rows:
            return {}
        mat = np.ascontiguousarray(np.stack(rows))
        spec = {
            "post_block": {
                "dtype": "float32",
                "shape": list(mat.shape),
                "layer_indices": idxs,
                "data": base64.b64encode(mat.tobytes()).decode(),
                "scales": [strength] * len(idxs),
            }
        }
        field = (
            "steering_vectors"
            if body.get("steerGeneratedTokens")
            else "prefill_steering_vectors"
        )
        return {field: spec}

    # -- readout accumulation ------------------------------------------------

    async def _tail(self, req_prefix: str, started: float, sink, stop: asyncio.Event):
        path = None
        while path is None:
            found = glob.glob(os.path.join(self._readout_dir, f"*{req_prefix}*.jsonl"))
            if found:
                path = max(found, key=os.path.getmtime)
            elif stop.is_set() and time.time() - started > 20:
                return
            else:
                await asyncio.sleep(0.1)
        pos = 0
        idle = 0.0
        while idle < 12.0:
            with open(path, encoding="utf-8") as f:
                f.seek(pos)
                new = f.read()
                pos = f.tell()
            if new:
                idle = 0.0
                for line in new.splitlines():
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("event") == "done":
                        return
                    await sink(rec)
            else:
                idle = idle + 0.2 if stop.is_set() else 0.0
                await asyncio.sleep(0.2)

    # -- the endpoint --------------------------------------------------------

    async def stream(self, body: dict):
        """Async generator yielding NDJSON lines (str, newline-terminated)."""
        types = body.get("type") or ["JACOBIAN_LENS", "LOGIT_LENS"]
        top_n = min(int(body.get("topN", 8)), 8)
        temperature = float(body.get("temperature", 0))
        rid = f"npjl-{uuid.uuid4().hex[:12]}"

        try:
            ids, n_gen = self._build_ids(body)
            steer_fields = self._steering_payload(body)
        except ValueError as exc:
            yield json.dumps({"kind": "error", "error": str(exc)}) + "\n"
            return
        prompt_len = len(ids)
        tok = self.lens.tok

        yield json.dumps({
            "kind": "meta", "model": body.get("modelId", "glm-5.2"),
            "types": types,
            "layers_by_type": {t: self.layers for t in types},
            "top_n": top_n, "prompt_len": prompt_len,
            "num_completion_tokens": n_gen, "temperature": temperature,
            "prepend_bos": bool(body.get("prependBos", True)), "reuse_len": 0,
        }) + "\n"
        yield json.dumps({
            "kind": "prompt",
            "tokens": [
                {"position": i, "token": tok.decode([t]), "id": int(t),
                 "is_generated": False}
                for i, t in enumerate(ids)
            ],
        }) + "\n"

        # per-position readout accumulation; a position is emittable once
        # every capture layer has reported
        acc: dict[int, dict[int, dict]] = {}
        gen_ids: dict[int, int] = {}       # position -> generated token id
        ready = asyncio.Event()

        async def sink(rec: dict) -> None:
            layer = rec["layer"]
            for i, p in enumerate(rec["pos"]):
                slot = acc.setdefault(p, {})
                slot[layer] = {
                    "JACOBIAN_LENS": (rec["jl"]["t"][i][:top_n], rec["jl"]["p"][i][:top_n]),
                    "LOGIT_LENS": (rec["ll"]["t"][i][:top_n], rec["ll"]["p"][i][:top_n]),
                }
            ready.set()

        started = time.time()
        stop = asyncio.Event()
        tail_task = asyncio.create_task(self._tail(rid, started, sink, stop))

        payload = {
            "model": "glm-5.2",
            "prompt": ids,
            "max_tokens": max(n_gen, 1),
            "temperature": temperature,
            "seed": 0,
            "stream": True,
            "logprobs": 0,
            "return_tokens_as_token_ids": True,
            "request_id": rid,
            "capture": {
                "jlens": {
                    "hooks": {"post_block": [l for l in self.layers]},
                    "positions": "all",
                }
            },
            **steer_fields,
        }

        gen_done = asyncio.Event()
        gen_error: list[str] = []

        async def generate() -> None:
            try:
                async with httpx.AsyncClient(timeout=1800) as client:
                    async with client.stream(
                        "POST", f"{self._server_base()}/v1/completions", json=payload
                    ) as resp:
                        if resp.status_code != 200:
                            raw = (await resp.aread()).decode(errors="replace")
                            try:
                                raw = json.loads(raw)["error"]["message"]
                            except Exception:
                                pass
                            gen_error.append(str(raw))
                            return
                        pos = prompt_len
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            if line[6:].strip() == "[DONE]":
                                break
                            chunk = json.loads(line[6:])
                            lp = chunk["choices"][0].get("logprobs") or {}
                            for t in lp.get("tokens", []):
                                if t.startswith("token_id:"):
                                    gen_ids[pos] = int(t.split(":", 1)[1])
                                    pos += 1
                                    ready.set()
            finally:
                gen_done.set()
                stop.set()
                ready.set()

        gen_task = asyncio.create_task(generate())

        # emit token messages strictly in position order
        drop_last_gen = n_gen == 0  # served max_tokens=1 for lens-only requests
        total = prompt_len + (0 if drop_last_gen else n_gen)
        next_pos = 0
        try:
            while next_pos < total:
                emittable = (
                    next_pos in acc
                    and len(acc[next_pos]) >= len(self.layers)
                    and (next_pos < prompt_len or next_pos in gen_ids)
                )
                if not emittable:
                    if gen_done.is_set() and tail_task.done():
                        break  # stream ended (stop/abort/short generation)
                    ready.clear()
                    await ready.wait()
                    continue
                is_gen = next_pos >= prompt_len
                tid = gen_ids[next_pos] if is_gen else int(ids[next_pos])
                results = []
                for t in types:
                    slot = acc[next_pos]
                    results.append({
                        "type": t,
                        "top_tokens": [slot[l][t][0] for l in self.layers],
                        "top_probs": [slot[l][t][1] for l in self.layers],
                    })
                yield json.dumps({
                    "kind": "token", "position": next_pos,
                    "token": tok.decode([tid]), "id": tid,
                    "is_generated": is_gen, "results": results,
                }, ensure_ascii=False) + "\n"
                del acc[next_pos]
                next_pos += 1

            if gen_error:
                yield json.dumps({"kind": "error", "error": gen_error[0]}) + "\n"
                return
            completion = "".join(
                tok.decode([gen_ids[p]]) for p in sorted(gen_ids)
            ) if not drop_last_gen else ""
            yield json.dumps({
                "kind": "done",
                "seq_len": max(next_pos, prompt_len + len(gen_ids) * (not drop_last_gen)),
                "prompt_len": prompt_len, "vocab_size": VOCAB_SIZE,
                "completion": completion,
            }) + "\n"
        finally:
            stop.set()
            gen_task.cancel()
            tail_task.cancel()
