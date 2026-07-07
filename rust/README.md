# vllm-frontend-rs

This is a Rust drop-in alternative frontend for vLLM. The current goal is to rebuild the northbound serving layer in Rust while still talking to the core Python vLLM engine process(es) via ZMQ over the existing engine boundary.

It should still be considered experimental, and is not feature-complete. We are working to add more functionality from the python front-end.

See <https://github.com/Inferact/vllm-frontend-rs> for the original commit history before it was moved into the main vllm repo.

## Architecture

The component is organized as a Cargo workspace with several crates, layered bottom-up:

```text
┌─────────────────────────────────┐
│  vllm-cmd / vllm-rs             │  CLI entrypoint:
│                                 │  Python vLLM frontend subprocess
│                                 │  Rust managed-engine serve mode
├─────────────────────────────────┤
│  vllm-server                    │  OpenAI-compatible HTTP API (axum)
├─────────────────────────────────┤
│  vllm-chat                      │  Chat completions: template rendering,
│                                 │  structured assistant events,
│                                 │  reasoning & tool parsing
├─────────────────────────────────┤
│  vllm-text                      │  Tokenizer & incremental detokenizer
├─────────────────────────────────┤
│  vllm-llm                       │  Thin token-in/token-out facade over
│                                 │  the engine client
├─────────────────────────────────┤
│  vllm-engine-core-client        │  ZMQ transport + MessagePack protocol
│                                 │  for the headless vLLM engine
└─────────────────────────────────┘
```

`vllm-rs` integrates into Python `vllm` as a Rust frontend subprocess.
Python owns process startup and launches the Rust API server as a Python-supervised worker, while
passing the inherited listening socket and transport addresses into `vllm-rs`.

For example:

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B
```

### External Engine

`vllm-rs serve` can be run standalone with `--data-parallel-size-local 0` when the Python engines
are started elsewhere and this node should run only the Rust frontend. The frontend still uses
the global `--data-parallel-size` to determine how many engines it expects to join the shared handshake.

```bash
vllm serve Qwen/Qwen3-0.6B \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1
```

Then start the Rust frontend-only server:

```bash
vllm-rs serve Qwen/Qwen3-0.6B \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size 1 \
  --data-parallel-size-local 0
```

To build the `vllm-rs` in isolation:

```bash
# from the local checkout
./build_rust.sh
```

### Example Request

After either startup path, you can use any OpenAI-compatible client:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": true
  }'
```

## Activation patching support

A per-request `patch` spec (a list of `(layer, hook, dest_position, source_run,
source_position, alpha)` site entries — see
[`docs/features/activation_patching.md`](../docs/features/activation_patching.md))
works through the Rust frontend on **both** `POST /v1/completions` and `POST
/v1/chat/completions`. It is forwarded verbatim southbound to the Python
engine-core, which resolves it with **full engine-side admission** — the same
treatment served Python requests get:

- precise prefix-cache floors → APC position-windowing (only the patched prompt
  positions and after are re-forwarded; everything below still serves from
  cache),
- cache-taint protection, scheduler per-`(layer, hook)` backpressure, and strict
  pool-overflow rejection.

Admission runs in the engine's input processor
(`vllm/v1/engine/input_processor.py`), so it is idempotent: a spec already
stamped by the Python OpenAI serving layer is not re-admitted, and a Rust /
offline request is admitted here instead. An invalid spec (bad hook,
out-of-range layer, per-site overflow, or patching not enabled) is rejected by
the engine.

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "The Eiffel Tower is in",
    "patch": [
      {"layer": 14, "hook": "post_block", "dest_position": 6,
       "source_run": "clean", "source_position": 6, "alpha": 1.0}
    ]
  }'
```

### What is not available through the Rust frontend

Only the **per-request** patch path is ported. The server-side sweep
orchestration is **deliberately not** ported to Rust — it is a large,
research-workflow-specific surface (`POST /v1/patch_sweep` plus one-call
auto-capture, substring→position resolution, multi-hook grids, SSE streaming,
the `DELETE /v1/patch_source/{run_id}` source-run lifecycle route, and the
`PatchStudy` client). None of these exist through Rust. Also absent:

- **Admission-time `400`s for a missing source run.** The Python serving layer
  pre-checks source existence via an engine RPC; the engine-side admission the
  Rust path uses does not. A missing source instead surfaces (loudly) at worker
  resolution via the resolution-failure registry backstop, not as an up-front
  request rejection.
- **The multimodal guard.** Rejecting patch specs on multimodal prompts is a
  frontend concern and is not enforced on the Rust path.

**Decision.** Sweep orchestration stays Python-only: the maintenance cost of a
parity port outweighs current research-workflow demand, and the primitive that
matters for programmatic patching (per-request `patch=`) is fully available. For
grid sweeps / causal-tracing studies, run the Python `vllm.entrypoints.openai.api_server`.
If Rust-served sweeps ever become a real need, the escalation ladder is: (1) run
the Python api_server alongside the Rust frontend, (2) reverse-proxy just the
`/v1/patch_sweep` route from Rust to Python, (3) lift sweep orchestration into a
shared, frontend-agnostic layer both call.

### gRPC

Patch is wired on the **HTTP** surface only. The served gRPC `Generate` path
does not carry a `patch` field yet: the proto's `capture` uses a
`google.protobuf.Struct` (a JSON object), but a patch spec is a JSON *list*, so
gRPC support needs a new proto field plus list-shaped conversion — deferred.

### Known limitation: logprobs

Requesting `logprobs` through the Rust frontend is a known pre-existing issue
(returns HTTP 500 in live serving). This matters for patching because grading a
sweep is logprob-based. Investigation found the Rust decode / wire-resolution /
response-assembly paths structurally sound and CPU-test-covered (they return
typed errors, not panics, on any shape mismatch), so there is no contained
CPU-reproducible fix; the failure needs a live engine emitting logprobs frames
to capture the exact shape mismatch. Tracked as a separate limitation, not
addressed by patch support.
