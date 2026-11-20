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
│                                 │  Engine-free render mode
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

To also expose the Rust frontend's gRPC inference, control, and render services,
add `--grpc-port`. The HTTP listener remains on `--port`:

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 8000 --grpc-port 50051
```

The gRPC listener binds on `--host` like HTTP, or on `127.0.0.1` when HTTP uses
`--uds`. When HTTP uses TCP, its port and the gRPC port must differ unless
requesting an automatically assigned port with `0`.

This also works with Python-supervised hybrid data parallelism. It requires a
frontend process and cannot be combined with `--headless` or
`--data-parallel-multi-port-external-lb`.

The separate `vllm serve MODEL --grpc --port 50051` command launches the Python
SMG `VllmEngine` gRPC server in place of HTTP. The Rust listener exposes the
`vllm.Inference`, `vllm.Control`, and `vllm.Render` services alongside HTTP. These are different
gRPC APIs; `--grpc` and `--grpc-port` are mutually exclusive.

### gRPC health checks

The gRPC listener also serves `grpc.health.v1.Health`. Kubernetes probes should
use the `liveness` service for liveness and `readiness` for readiness; see
[Rust frontend gRPC health checks](../docs/deployment/k8s.md#rust-frontend-grpc-health-checks).

### RL weight synchronization

With `VLLM_SERVER_DEV_MODE=1`, the Rust frontend supports the HTTP weight-transfer
lifecycle used by `HTTPVLLMWeightSyncClient`: initialization, starting an update,
transferring weights, and finishing the update. It also supports draft-model
updates and `/update_weight_version` and `/weight_info` for version tracking.

For trainer-side usage, see the [IPC](../examples/rl/rlhf_http_ipc.py) and
[NCCL](../examples/rl/rlhf_http_nccl.py) examples. Configure the weight-transfer
backend on the server and pause generation during updates, as shown there.

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
./tools/build_rust.sh
```

### Hybrid and External DP Load Balancing

Like `vllm serve` (see [Data Parallel Deployment](../docs/serving/data_parallel_deployment.md)), `vllm-rs serve` can run one Rust frontend per node that serves only the node's data parallel ranks, with an upstream load balancer spreading requests across nodes. Every node passes the global `--data-parallel-size`, the address of the node that owns rank 0 as `--data-parallel-address`, and the same explicit `--data-parallel-rpc-port`, because all engines join the handshake of the rank-0 frontend. For MoE models, the rank-0 frontend also runs the DP coordinator.

```bash
# Node 0: ranks 0-1
vllm-rs serve <moe-model> \
  --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-hybrid-lb \
  --data-parallel-address 10.0.0.1 --data-parallel-rpc-port 13345
# Node 1: ranks 2-3
vllm-rs serve <moe-model> \
  --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 2 \
  --data-parallel-address 10.0.0.1 --data-parallel-rpc-port 13345
```

For external load balancing, start one `vllm-rs serve` per rank with `--data-parallel-rank <rank>`. A request can be pinned to one of a frontend's ranks with the `X-data-parallel-rank` header, which takes the global rank. Unlike `vllm serve`, `--data-parallel-rpc-port` has no default here and ranks are not inferred from `--nnodes`/`--node-rank`.

### Engine-free renderer

`vllm-rs render` serves request preprocessing without starting or connecting
to a Python inference engine:

```bash
cargo run --manifest-path rust/Cargo.toml -p vllm-cmd --release -- \
  render Qwen/Qwen3-32B \
  --host 127.0.0.1 --max-model-len 32768
```

It exposes `/v1/chat/completions/render` and `/v1/completions/render`. Only
tokenizer, model and processor configuration files are loaded; model weights,
PyTorch, and vLLM kernels are not required.

For multimodal models, chat render fetches and preprocesses images, video and
audio, and returns them as the `features` object that `/inference/v1/generate`
accepts, in the format the Python render API uses. The tensors are produced in
`--dtype`, which must match the engine's: `auto` (the default) follows the
model config, running float32 checkpoints in bfloat16 like vLLM, but unlike
vLLM it does not look at the weights or the GPU, so pass the engine's dtype
when they decide it. Set `--max-model-len` to the engine's too: some image
processors size images to fit the context. Large images make `features` big;
the generate server may need `VLLM_HTTP_MAX_REQUEST_BODY_SIZE` raised above
its 32 MiB default to accept them. Set
`"mm_payload": "none"` on a render request to get only item hashes and
placeholder ranges, for example for cache-aware routing; such a response cannot
be sent to generate. Pass `--language-model-only` to skip multimodal
processing, and set `--limit-mm-per-prompt` to the engine's limits so that
render rejects the same requests.

Pass `--lora-modules name=path ...` to accept LoRA adapter names as the request
`model`; rendering does not load the adapter weights, and the adapter name is
passed through to generate.

To serve HTTPS, pass `--ssl-certfile`. If the certificate and private key are
in separate files, pass `--ssl-keyfile` for the key; otherwise the key is read from
the certificate file. For mTLS, pass `--ssl-ca-certs` with a CA
bundle and set `--ssl-cert-reqs` to `1` (optional) or `2` (required) to
verify client certificates. Use `--ssl-ciphers` to override the default
OpenSSL cipher list.

```bash
# Combined PEM (cert + key in one file):
vllm-rs \
  render Qwen/Qwen3-32B \
  --host 127.0.0.1 --max-model-len 32768 \
  --ssl-certfile /path/to/combined.pem

# Separate cert and key files:
vllm-rs \
  render Qwen/Qwen3-32B \
  --host 127.0.0.1 --max-model-len 32768 \
  --ssl-certfile /path/to/cert.pem --ssl-keyfile /path/to/key.pem
```

The render endpoints return the public token-in `GenerateRequest` consumed by
the Rust `/inference/v1/generate` endpoint. A chat render response, or one item
from a completion render response, can be submitted to that endpoint without
changing its fields.

The render and inference paths use the same `vllm-chat` and `vllm-text`
request-preparation logic; render mode stops before engine submission.
Tool-call and reasoning parsers use model-based auto-detection by default. Use
`--tool-call-parser` and `--reasoning-parser` to override either selection;
unified parsers require the same selection for both options.

```bash
curl http://127.0.0.1:8000/v1/chat/completions/render \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_completion_tokens": 16
  }'
```

Pass `--grpc-port` to also serve the `vllm.Render` gRPC service on `--host`.
`Render.RenderChat` renders a chat request into prompt token IDs plus the
settings rendering decides (structured-output constraint, reasoning gate,
preprocessed media), which the client copies into its `vllm.GenerateRequest`
for `Inference.Generate`. See [`render.proto`](proto/render.proto).

### Example Request

After either full-frontend startup path, you can use any OpenAI-compatible
client against the inference endpoints:

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

**Client-provided patch values** ride through Rust for free. In addition to
`source_run` (a capture run), a patch entry may source its value from a named
steering module (`"source_module": "<name>"`), the reserved zero row
(`"source_module": "zeros"`), or a row of a request-level packed
`patch_vectors` table (`"source_inline": <row>`); an optional per-entry `mask`
(`{"indices": [...]}` or `{"inline": <row>}`) restricts the patch to a subset
of dims. The `patch_vectors` table is a sibling request field forwarded
verbatim southbound exactly like `patch` (omitted from the wire when unset).
Named-module *name* existence is only pre-validated on the Python OpenAI
frontend (its steering registry); over Rust an unknown name surfaces via the
worker resolution-failure backstop.

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

# Zero-ablate residual dims 12 and 40 at layer 14 (no capture run needed):
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "The Eiffel Tower is in",
    "patch": [
      {"layer": 14, "hook": "post_block", "dest_position": 6,
       "source_module": "zeros", "mask": {"indices": [12, 40]}}
    ]
  }'
```

The sweep-endpoint additions (vector-sourced ablation sweeps) ride the existing
Python patch sidecar below — no Rust changes are needed for them.

### Server-side sweeps via the patch sidecar

The full server-side sweep surface (`POST /v1/patch_sweep` plus one-call
auto-capture, substring→position resolution, multi-hook grids, SSE streaming,
the `DELETE /v1/patch_source/{run_id}` source-run lifecycle route, and the
`PatchStudy` client) **is** available through the Rust frontend, served by an
auto-spawned Python **patch sidecar**:

- When vLLM launches the Rust frontend **and** patching is enabled
  (`--enable-patching`), the driver additionally spawns **one** ordinary Python
  `api_server` bound to loopback (`127.0.0.1:<port>`, `--patch-sidecar-port`,
  default auto-picked) attached to the **same engines** as a second engine
  client. There is no second engine set — one set of weights, one KV cache, one
  worker-side `PatchSourceStore` shared by both frontends.
- The Rust server **reverse-proxies** `POST /v1/patch_sweep` and
  `DELETE /v1/patch_source/{run_id}` to the sidecar: it forwards the method,
  path, body, and relevant headers and streams the response back
  **incrementally** (the sweep endpoint's `text/event-stream` SSE chunks pass
  through as they land, never buffered). A client disconnect drops the upstream
  connection so the Python side cancels the in-flight sweep's GPU work.
- **Opt out** with `VLLM_RUST_PATCH_SIDECAR=0`: the sidecar is not spawned and
  the Rust frontend returns **HTTP 501** for the sweep routes (pointing back at
  this section). The sidecar is also absent whenever patching is disabled.

The per-request `patch=` path above is served natively by Rust and does **not**
go through the sidecar. Still absent on the Rust-native path (per-request only):

- **Admission-time `400`s for a missing source run.** The Python serving layer
  pre-checks source existence via an engine RPC; the engine-side admission the
  Rust path uses does not. A missing source instead surfaces (loudly) at worker
  resolution via the resolution-failure registry backstop, not as an up-front
  request rejection. (Sweeps proxied to the sidecar keep the Python `400`s.)
- **The multimodal guard.** Rejecting patch specs on multimodal prompts is a
  frontend concern and is not enforced on the Rust-native per-request path.

**Decision record.** Sweep *orchestration* itself stays Python-only — a parity
port to Rust is not worth the maintenance cost. The sidecar realizes escalation
tier 2 (reverse-proxy `/v1/patch_sweep` from Rust to Python) of the original
ladder and supersedes the earlier "run the Python `api_server` separately"
tier-1 workaround: one launch now gives the Rust frontend the whole sweep
surface with a single engine set. Tier 3 (lifting orchestration into a shared,
frontend-agnostic layer both call) remains a future option if a Rust-native
implementation is ever justified.

### gRPC

Patch is wired on the **HTTP** surface only. The served gRPC `Generate` path
does not carry a `patch` field yet: the proto's `capture` uses a
`google.protobuf.Struct` (a JSON object), but a patch spec is a JSON *list*, so
gRPC support needs a new proto field plus list-shaped conversion — deferred.

### Output wire-format re-sync (logprobs now decode)

The previously documented "logprobs return HTTP 500" symptom was not a
logprobs-specific bug: the Rust `EngineCoreOutputs` protocol structs had drifted
from the Python `msgspec` wire format after a large upstream merge, so the
frontend failed to decode *every* `EngineCoreOutputs` frame (even a plain
1-token completion). The concrete drift: upstream appended a
`late_capture_results` field to the `array_like` `EngineCoreOutputs` tuple,
between `utility_output` and `finished_requests`. It arrives as a (usually
empty) map at tuple index 5; the Rust struct decoded that map against
`finished_requests` (a set) and failed with "invalid type: map, expected a
sequence". Because a single failed decode used to tear down the dispatcher, one
bad frame wedged the client permanently.

Fixed by adding `late_capture_results` to `EngineCoreOutputs` (decoded
permissively into `HashMap<String, HashMap<String, CaptureResult>>`, ignored by
the frontend for now) and re-syncing `SchedulerStats` with the upstream
`waiting_lora_adapters` / `running_lora_adapters` maps. With the outer tuple
aligned, logprobs decode through the existing wire path: `LogprobsLists` /
`LogprobsTensors` are `NamedTuple`s of `(dtype, shape, data)` ndarray/tensor
triples, where `data` is either an inline `CUSTOM_TYPE_RAW_VIEW` (ext code 3)
byte blob for small tensors or an aux-frame index for large ones sent zero-copy
over multipart ZMQ. Both forms, `<i8`/`<f4` dtypes, and little/big/native
endianness are handled by the array decoder. Live GPU validation of the full
serving path is performed separately.

The output dispatcher no longer wedges on a bad frame: a per-frame decode
failure is logged and skipped so outputs for other requests keep flowing; only
fatal transport / engine-dead conditions tear down the registries.
