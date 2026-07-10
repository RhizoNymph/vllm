"""Web UI sidecar for the live Jacobian-lens demo.

Serves a single-page interface (static/index.html) and one SSE endpoint that,
per generation request:

  1. optionally derives a steering direction for an arbitrary word from the
     lens (J^T readout direction, residual-norm scaled) and packs it into the
     fork's wire format;
  2. proxies the chat request to the vllm server with the jlens capture
     consumer armed (and the steering vectors attached);
  3. tails the consumer's JSONL readout stream and relays token + readout
     events over one SSE stream. With steering set, a baseline pass runs
     first for side-by-side comparison.

Run (in the serving venv, anywhere that sees /mnt/data):
    python sidecar.py --port 7860
Then expose:  tunnel-url 7860 -t 24

Single-demo-user assumptions (documented, not enforced): readout files are
matched by creation time, so concurrent generations may cross streams.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import hashlib
import base64
import glob
import json
import os
import time
import uuid

import httpx
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

RUN_INFO = os.environ.get("JLENS_RUN_INFO", "/mnt/data/artifacts/jlens/serve/current.json")
READOUT_DIR = os.environ.get("JLENS_READOUT_DIR", "/mnt/data/artifacts/jlens/readout")
LENS_PATH = os.environ.get("JLENS_LENS", "/mnt/data/artifacts/jlens/glm52_fit_1k/lens_glm52_1k.pt")
UNEMBED_PATH = os.environ.get("JLENS_UNEMBED", "/mnt/data/artifacts/jlens/glm52_unembed.pt")
NORMS_PATH = os.environ.get("JLENS_NORMS", "/mnt/data/artifacts/jlens/glm52_norms/residual_norms.pt")
BAND = [30, 40, 50]

app = FastAPI(title="jlens live demo")
_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
# capture layers configured on the serving side (must match the jlens
# consumer's layers= param); stride-4 sweep + final by default
CAPTURE_LAYERS = [int(x) for x in os.environ.get(
    "JLENS_CAPTURE_LAYERS", ";".join(str(l) for l in range(0, 77, 4))
).replace("|", ";").split(";")]


class Lens:
    """Band J + unembed + norms; steering-direction derivation."""

    def __init__(self) -> None:
        bundle = torch.load(LENS_PATH, map_location="cpu", weights_only=True)
        wanted = set(BAND) | set(CAPTURE_LAYERS)
        self.J = {l: bundle["J"][l].float() for l in wanted if l in bundle["J"]}
        del bundle
        u = torch.load(UNEMBED_PATH, map_location="cpu", weights_only=True)
        self.norm_w = u["norm_weight"].float()
        self.lm_head = u["lm_head_weight"].float()
        del u
        self.norms = torch.load(NORMS_PATH, map_location="cpu", weights_only=True)[
            "mean_residual_norm"
        ]
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(
            "/mnt/data/artifacts/GLM-5.2", trust_remote_code=True
        )

    def steering_spec(self, word: str, strength: float, layers: list[int]) -> dict:
        token_id = self.tok(word, add_special_tokens=False).input_ids[0]
        rows, scales = [], []
        for layer in layers:
            w = self.norm_w * self.lm_head[token_id]
            d = w @ self.J[layer]
            d = d / d.norm() * self.norms[layer]
            rows.append(d.numpy().astype("<f4"))
            scales.append(strength)
        mat = np.ascontiguousarray(np.stack(rows))
        return {
            "post_block": {
                "dtype": "float32",
                "shape": list(mat.shape),
                "layer_indices": layers,
                "data": base64.b64encode(mat.tobytes()).decode(),
                "scales": scales,
            }
        }


LENS = Lens()

from lens_api import LensAPI  # noqa: E402  (needs Lens defined)

LENS_API = LensAPI(LENS, lambda: _server_base(), READOUT_DIR, CAPTURE_LAYERS)


@app.post("/api/lens/prompt")
async def lens_prompt(body: dict) -> StreamingResponse:
    """Neuronpedia jlens UI contract (NDJSON). See lens_api.py."""
    return StreamingResponse(
        LENS_API.stream(body), media_type="application/x-ndjson"
    )


_np_dist = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui-np-dist")
if os.path.isdir(_np_dist):
    from fastapi.staticfiles import StaticFiles

    app.mount("/np", StaticFiles(directory=_np_dist, html=True), name="np")


def _server_base() -> str:
    info = json.load(open(RUN_INFO))
    return f"http://{info['host']}:{info['port']}"


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Completion cache: per PANE, keyed on everything that affects that pane's
# output (the baseline pane's key excludes steering entirely, so iterating on
# steering strength replays the baseline instantly). Values are the pane's
# full ordered event list (tokens + readouts), so a hit reproduces completion
# AND ticker. Memory LRU + disk (survives sidecar restarts); CACHE_V salts
# away stale formats when the readout schema or server config changes.
# ---------------------------------------------------------------------------
CACHE_V = 2
CACHE_DIR = os.environ.get(
    "JLENS_CACHE_DIR", os.path.join(os.path.dirname(RUN_INFO), "cache")
)
os.makedirs(CACHE_DIR, exist_ok=True)
_cache_mem: collections.OrderedDict[str, list] = collections.OrderedDict()
_CACHE_MEM_MAX = 32


def _cache_key(desc: dict) -> str:
    blob = json.dumps({"v": CACHE_V, **desc}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _cache_get(key: str) -> list | None:
    if key in _cache_mem:
        _cache_mem.move_to_end(key)
        return _cache_mem[key]
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            events = json.load(open(path, encoding="utf-8"))
        except Exception:
            return None
        _cache_mem[key] = events
        while len(_cache_mem) > _CACHE_MEM_MAX:
            _cache_mem.popitem(last=False)
        return events
    return None


def _cache_put(key: str, events: list) -> None:
    _cache_mem[key] = events
    while len(_cache_mem) > _CACHE_MEM_MAX:
        _cache_mem.popitem(last=False)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False)
    os.replace(tmp, path)


async def _tail_readout(req_prefix: str, started: float, queue: asyncio.Queue,
                        stop: asyncio.Event) -> None:
    """Find this request's readout JSONL (filename embeds the client request
    id — exact handshake, no timing race) and stream its lines."""
    path = None
    while path is None:
        candidates = glob.glob(os.path.join(READOUT_DIR, f"*{req_prefix}*.jsonl"))
        if candidates:
            path = max(candidates, key=os.path.getmtime)
        elif stop.is_set() and time.time() - started > 15:
            return
        else:
            await asyncio.sleep(0.15)
    pos = 0
    idle = 0.0
    while idle < 10.0:
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
                await queue.put({"type": "readout", **rec})
        else:
            if stop.is_set():
                idle += 0.25
            await asyncio.sleep(0.25)


async def _generate_pane(
    pane: str, payload: dict, queue: asyncio.Queue, with_readout: bool,
    cache_key: str | None = None,
) -> None:
    if cache_key:
        cached = _cache_get(cache_key)
        if cached is not None:
            await queue.put({"type": "cached", "pane": pane})
            for event in cached:
                await queue.put(event)
            await queue.put({"type": "pane_done", "pane": pane})
            return
    recorded: list = []

    async def put(event: dict) -> None:
        recorded.append(event)
        await queue.put(event)

    started = time.time()
    stop = asyncio.Event()
    import types as _types
    _rec_queue = _types.SimpleNamespace(put=put)
    tail = (
        asyncio.create_task(
            _tail_readout(payload["request_id"], started, _rec_queue, stop)
        )
        if with_readout
        else None
    )
    # GLM-5.2's chat template pre-opens a <think> block: streamed content is
    # reasoning until the closing tag. Label channels so the UI can render
    # thinking distinctly; strip the tags themselves. Tags may split across
    # deltas, so hold back a partial-tag suffix.
    thinking_enabled = "chat_template_kwargs" not in payload
    channel = "think" if thinking_enabled else "answer"
    carry = ""
    try:
        async with httpx.AsyncClient(timeout=900) as client:
            async with client.stream(
                "POST", f"{_server_base()}/v1/chat/completions", json=payload
            ) as resp:
                if resp.status_code != 200:
                    body_text = (await resp.aread()).decode(errors="replace")
                    try:
                        body_text = json.loads(body_text)["error"]["message"]
                    except Exception:
                        pass
                    await queue.put({"type": "token", "pane": pane,
                                     "channel": "answer",
                                     "text": f"[server error] {body_text}"})
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    delta = (
                        json.loads(data)["choices"][0].get("delta", {}).get("content")
                    )
                    if not delta:
                        continue
                    buf = carry + delta
                    carry = ""
                    while buf:
                        if channel == "think" and "</think>" in buf:
                            pre, buf = buf.split("</think>", 1)
                            pre = pre.replace("<think>", "")
                            if pre:
                                await put({"type": "token", "pane": pane,
                                           "channel": "think", "text": pre})
                            channel = "answer"
                            continue
                        # hold back a suffix that could be a split tag
                        hold = 0
                        for k in range(min(8, len(buf)), 0, -1):
                            if "</think>"[:k] == buf[-k:] or "<think>"[:k] == buf[-k:]:
                                hold = k
                                break
                        emit, carry = (buf[:-hold], buf[-hold:]) if hold else (buf, "")
                        emit = emit.replace("<think>", "")
                        if emit:
                            await put({"type": "token", "pane": pane,
                                       "channel": channel, "text": emit})
                        buf = ""
                if carry:
                    await put({"type": "token", "pane": pane,
                               "channel": channel, "text": carry})
    except asyncio.CancelledError:
        # client hit Stop / disconnected: closing the httpx stream makes
        # vllm abort the request server-side; tear the tailer down fast.
        if tail:
            tail.cancel()
        cache_key = None  # never cache a partial pane
        raise
    finally:
        stop.set()
        if tail:
            with contextlib.suppress(asyncio.CancelledError):
                await tail
        if cache_key and any(
            e["type"] == "token" and not e["text"].startswith("[server error]")
            for e in recorded
        ):
            _cache_put(cache_key, recorded)
        with contextlib.suppress(Exception):
            await queue.put({"type": "pane_done", "pane": pane})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(os.path.join(_static, "index.html"))


@app.get("/api/info")
async def info() -> dict:
    return {"band": BAND, "server": _server_base(), "model": "GLM-5.2 (fp8, TP8)",
            "lens": os.path.basename(LENS_PATH)}


@app.post("/api/generate")
async def generate(body: dict) -> StreamingResponse:
    prompt = body.get("prompt", "").strip() or "Tell me about your weekend plans."
    # No sidecar-side cap: the server enforces its own context budget
    # (--max-model-len minus prompt) and errors informatively past it.
    max_tokens = max(1, int(body.get("max_tokens", 96)))
    thinking = bool(body.get("thinking", True))
    steer = body.get("steer")  # {word, strength, layers} | None
    rid = f"jlens-{uuid.uuid4().hex[:12]}"

    def payload(steered: bool) -> dict:
        p = {
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 0,
            "stream": True,
            # handshake: the consumer names its readout file by the internal
            # request id, which embeds this client-supplied id — the tailer
            # matches by prefix instead of racing on mtimes (multi-user safe)
            "request_id": f"{rid}-{'s' if steered else 'b'}",
            "capture": {
                "jlens": {"hooks": {"post_block": BAND}, "positions": "all_generated"}
            },
        }
        if not thinking:
            p["chat_template_kwargs"] = {"enable_thinking": False}
        if steered and steer:
            spec = LENS.steering_spec(
                str(steer["word"]),
                float(steer["strength"]),
                [int(l) for l in steer.get("layers", [40])],
            )
            p["decode_steering_vectors"] = spec
        return p

    async def stream():
        queue: asyncio.Queue = asyncio.Queue()

        base_desc = {"prompt": prompt, "max_tokens": max_tokens,
                     "thinking": thinking}
        steer_desc = None
        if steer:
            steer_desc = {
                "word": str(steer["word"]),
                "strength": float(steer["strength"]),
                "layers": sorted(int(l) for l in steer.get("layers", [40])),
            }

        async def run() -> None:
            if steer:
                # readout ticker follows the steered pane
                await _generate_pane(
                    "baseline", payload(False), queue, with_readout=False,
                    cache_key=_cache_key({**base_desc, "readout": False}),
                )
                await _generate_pane(
                    "steered", payload(True), queue, with_readout=True,
                    cache_key=_cache_key(
                        {**base_desc, "readout": True, "steer": steer_desc}
                    ),
                )
            else:
                await _generate_pane(
                    "baseline", payload(False), queue, with_readout=True,
                    cache_key=_cache_key({**base_desc, "readout": True}),
                )
            await queue.put({"type": "done"})

        task = asyncio.create_task(run())
        try:
            while True:
                event = await queue.get()
                yield _sse(event)
                if event["type"] == "done":
                    break
        finally:
            task.cancel()

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
