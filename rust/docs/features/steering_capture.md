# Per-request steering & capture (Rust frontend)

Per-request **activation steering** and **activation capture** support in the
Rust frontend. Clients pass steering vectors and/or capture specs on a single
request; the frontend forwards them southbound to the Python engine-core, which
performs all resolution, admission, and execution.

## Scope

- Accept steering vectors and capture specs on **both** inbound surfaces:
  - OpenAI HTTP: `POST /v1/completions` and `POST /v1/chat/completions`
  - the served gRPC `Generate` service (`rust/proto/vllm_grpc.proto`)
- Decode the **packed** steering wire format (base64 tensor + dtype/shape
  metadata) into the inline form engine-core resolves.
- Forward `steering_name` as a named-module reference, and the raw `capture`
  dict, verbatim southbound.
- **Named steering modules**: load module JSON files at startup
  (`--steering-modules name=path`), broadcast them to the engine workers, and
  validate per-request `steering_name` against the registry. Also manage the
  registry at runtime via `GET`/`POST`/`DELETE /v1/steering/modules`. See
  *Named steering modules* below.

### Non-scope

- **No frontend-side per-request admission/resolution.** Steering unpacking to
  model-dtype arrays and capture prefix-cache admission live in the Python
  engine-core. The frontend only decodes the wire format and forwards. (Capture
  admission is the engine-core's idempotent "offline" path in
  `vllm/v1/engine/input_processor.py`.)
- **Capture results are fire-and-forget.** The frontend does not surface
  `capture_results` in responses; consumers (e.g. filesystem) write out of band.
  See *Follow-ups* below.

## Data / control flow

Inbound (two entry points), converging on `vllm_text::SamplingParams`:

```
HTTP CompletionRequest / ChatCompletionRequest        gRPC pb::GenerateRequest
  steering_vectors / prefill_ / decode_ (packed)         Steering{...} (packed)
  steering_name, capture (JSON)                           capture (Struct)
        │                                                        │
        │ prepare_*_request (convert.rs)                         │ to_text_request (grpc/convert.rs)
        │  unpack_steering_field  (base64 → inline)              │  convert_packed_steering (bytes → inline)
        │                                                        │  proto_struct_to_json_prefer_int (capture)
        ▼                                                        ▼
                       vllm_text::SamplingParams
       { steering_vectors, prefill_, decode_ : SteeringVectorSpec,
         steering_name: Option<String>, capture: Option<Value> }
                                  │
                                  │ lower_sampling_params (text/src/lower.rs)
                                  │   steering_name → steering_module_ref = (name, 1.0)
                                  ▼
                    EngineCoreSamplingParams
       { steering_vectors, prefill_, decode_, steering_module_ref, capture }
                                  │
                                  │ rmp_serde::to_vec_named (field-name map)
                                  ▼
                  Python engine-core SamplingParams (ZMQ msgpack)
```

The packed → inline decode is the only transformation; everything else is
pass-through field threading.

### Steering wire format (packed)

Per hook point, one `SteeringHookPacked`:

| field           | meaning                                                       |
| --------------- | ------------------------------------------------------------- |
| `dtype`         | `float32` \| `float16` \| `bfloat16` \| `float64`             |
| `shape`         | `[num_rows, hidden_size]`                                     |
| `layer_indices` | layer index per row (length == `num_rows`)                    |
| `data`          | base64 (HTTP) / raw bytes (gRPC), contiguous little-endian    |
| `scales`        | optional per-row multiplier (length == `num_rows`); else 1.0  |

Decoded into `SteeringVectorSpec = HashMap<String, HashMap<u32, SteeringLayerEntry>>`
where `SteeringLayerEntry { vector: Vec<f32>, scale: f32 }`. The layer map uses
an integer (`u32`) key so it serializes to **msgpack integer keys**, which Python
decodes into `dict[int, ...]`.

## Named steering modules

At startup the server loads each `--steering-modules name=path` JSON file, then
broadcasts the whole registry to every engine worker so per-request references
avoid re-sending vector blobs:

```
build_state (lib.rs)
  EngineCoreClient::connect
  load_and_broadcast_steering_modules(client, config.steering_modules)
    load_steering_module(path)            # JSON {vectors, prefill_vectors, decode_vectors}
                                          #   each tier inline or packed (per-tier detect)
    collective_rpc("register_steering_modules", kwargs={modules, replace:true})
    collective_rpc("pre_materialize_steering_module", kwargs={name})  # per module
  → returns the registered name set → AppState.steering_module_names
```

A module file is JSON with optional `vectors` / `prefill_vectors` /
`decode_vectors` tiers; each tier is either inline
(`{hook: {layer: [floats] | {vector, scale}}}`, layer keys string or int) or the
packed shape (per-hook `{dtype, shape, layer_indices, data, scales}`). The
broadcast payload is the same inline `SteeringVectorSpec` (integer layer keys)
used for per-request steering — no ndarray encoding.

Per-request `steering_name` is validated against `AppState.steering_module_names`
before forwarding: an unknown name is rejected up front (HTTP
`invalid_request` on `steering_name`; gRPC `NotFound`) rather than failing later
in the worker.

### Runtime registry endpoints

The registry can also be managed at runtime (the counterpart of the startup
load), re-broadcasting on every change:

- `GET /v1/steering/modules` — list registered names → `{ "modules": [...] }`.
- `POST /v1/steering/modules` — register/replace. Body:
  `{ "modules": { "<name>": { "vectors": ..., "prefill_vectors": ..., "decode_vectors": ... } }, "replace": false }`
  (tiers inline or packed, same shapes as the module file). `replace: true`
  makes the provided set the entire registry; `false` adds/overrides. Calls
  `register_modules` (broadcast + pre-materialize) then updates the name set.
- `DELETE /v1/steering/modules/{name}` — unregister one module
  (`unregister_steering_modules` worker RPC, releasing its pinned rows). Unknown
  name → `400`.

Mutations are serialized by `AppState.lock_steering_mutations()` so concurrent
register/unregister requests cannot interleave their broadcasts. The name set is
an `RwLock<HashSet<String>>` read on every request for `steering_name`
validation.

## Related files

| file | role / key exports |
| ---- | ------------------ |
| `rust/proto/vllm_grpc.proto` | `SteeringHookPacked`, `Steering`, and `GenerateRequest.steering` / `.capture` (`google.protobuf.Struct`) |
| `src/server/src/steering_modules.rs` | `load_steering_module`, `parse_module`, `load_and_broadcast_steering_modules`, `register_modules`, `unregister_modules` |
| `src/server/src/routes/steering.rs` | runtime endpoints: list / register / unregister |
| `src/server/src/config.rs` | `SteeringModulePath`, `Config.steering_modules` |
| `src/cmd/src/cli.rs` | `--steering-modules name=path` flag + `parse_steering_module` |
| `src/server/src/state.rs` | `AppState.steering_module_names` (RwLock), `steering_module_error`, registry mutators, `lock_steering_mutations` |
| `src/engine-core-client/src/protocol/steering.rs` | inline types `SteeringLayerEntry`, `SteeringVectorSpec` |
| `src/engine-core-client/src/protocol/mod.rs` | `EngineCoreSamplingParams` southbound fields (`steering_vectors`, `prefill_steering_vectors`, `decode_steering_vectors`, `steering_module_ref`, `capture`) |
| `src/text/src/request.rs` | `SamplingParams` user-facing fields (incl. `steering_name`, `capture`) |
| `src/text/src/lower.rs` | `lower_sampling_params` — threads fields; `steering_name → steering_module_ref` |
| `src/server/src/routes/openai/utils/steering.rs` | `SteeringHookPacked` HTTP DTO, `SteeringSpecPacked`, `SteeringDecodeError`, `unpack_steering_hook`, `unpack_steering_spec` |
| `src/server/src/utils.rs` | `unpack_steering_field` — decode one packed field → `ApiError` on failure |
| `src/server/src/routes/openai/{completions,chat_completions}/types.rs` | request fields |
| `src/server/src/routes/openai/{completions,chat_completions}/convert.rs` | decode + set on `SamplingParams` |
| `src/server/src/grpc/convert.rs` | `convert_packed_steering`, `proto_struct_to_json_prefer_int` |

## Invariants & constraints

- **Integer layer keys**: the southbound layer map MUST serialize with integer
  keys (enforced by the `u32` key type; covered by a wire-format test).
- **Field names** on `EngineCoreSamplingParams` MUST match Python
  `SamplingParams` attributes exactly (`steering_vectors`, …, `capture`).
- **Decode validation** (`unpack_steering_hook`): `shape` is 2-D; byte length ==
  `rows × hidden × element_size`; `layer_indices` / `scales` lengths == `rows`;
  dtype is a supported float. Failures → HTTP `invalid_request` (tagged with the
  field) / gRPC `InvalidArgument`.
- **gRPC capture int coercion**: protobuf `Struct` encodes numbers as doubles;
  whole-valued numbers are coerced back to JSON integers so the forwarded
  capture dict matches the HTTP/JSON path (layer/position indices stay ints).
- **`steering_name` → `(name, 1.0)`**: matches the Python OpenAI entrypoint; the
  worker applies the request scale.

## Follow-ups (TODO)

- Surface `capture_results` in responses. The Python `EngineCoreOutput` carries
  `capture_results` at tuple index 7 (between `stop_reason` and `events`); the
  Rust `EngineCoreOutput` tuple in `protocol/mod.rs` predates that field and is
  therefore misaligned for capture-enabled engines. Returning results requires
  inserting the field at the correct position and threading it to the responses.
