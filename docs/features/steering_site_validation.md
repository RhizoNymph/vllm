# Steering Site Validation

Model-aware admission of client steering specs: every per-request or
named-module steering vector and directional clamp is checked against the
served model's steering sites before it can reach a worker.

## Scope

- Inline per-request vectors (`steering_vectors`, `prefill_steering_vectors`,
  `decode_steering_vectors`) on the OpenAI frontends (`AsyncLLM`), the
  offline `LLM`, and the Rust/msgpack path (engine input processor).
- Inline per-request clamps (`steering_clamps` and its phase tiers).
- Named modules registered at startup (`--steering-modules`) or at runtime
  (`POST /v1/steering/modules/register`): vector and clamp tiers.
- Three checks per `(hook, layer, row)`:
  1. the hook is one the model steers;
  2. the row is exactly as wide as that hook's steering table;
  3. the layer index is in `[0, num_hidden_layers)`.
- The error type and its HTTP mapping (400 with the offending field, hook,
  layer and widths in the message).

## Non-scope

- Model-blind structural checks (hook name is a `SteeringHookPoint`, finite
  values, packed-blob shape/dtype, non-negative layer indices) — these stay
  in `SamplingParams.__post_init__` and the registry coercers.
- Global steering (`POST /v1/steering/set`): validated worker-side against
  the live layer modules (each table's own width, layer presence); the
  router already rejects layers no worker owns with
  `400 "Layer(s) [...] not found in model"`.
- Loop-pass validity of `"layer@pass"` keys (looped models): only the layer
  part is range-checked here.
- The per-site direction budget (`max_clamp_directions`) and SAE specs —
  checked by their own validators at the same seams.
- `mhc_streams_final` being registered on the last layer only: a row for it
  at another in-range layer passes this gate and is not applied.

## Types

`vllm/config/steering_geometry.py`

```python
@dataclass(frozen=True)
class SteeringSiteGeometry:
    hook_widths: Mapping[str, int]   # read-only MappingProxyType
    num_hidden_layers: int
```

- `hook_widths` lists exactly the hooks the model steers, each with its row
  width. A hook absent from it is rejected.
- Construction validates: non-empty, every key a `SteeringHookPoint` value,
  every width a positive `int`, `num_hidden_layers` a positive `int`. The
  caller's mapping is copied, so the geometry cannot be mutated afterwards.
- `SteeringSiteGeometry.standard(hidden_size, num_hidden_layers=...)`: the
  five standard residual hooks at `hidden_size`.
- `SteeringSiteGeometry.from_hook_schema(schema, num_hidden_layers=...)`: the
  steerable subset of a capture hook schema (capture-only hooks such as the
  mHC mixing coefficients are dropped).
- `resolve_steering_site_geometry(model_config)`: the one resolver. Widths
  come from `vllm.v1.capture.types.build_model_hook_schema(hidden_size,
  dtype, hf_config, hf_text_config)` — the same per-family hook geometry the
  capture framework uses — and the layer count from
  `model_config.get_total_num_hidden_layers()` (all PP stages).

Resolved widths per family:

| Model family | Hooks | Width |
| --- | --- | --- |
| standard decoders | `pre_attn`, `post_attn`, `post_block`, `mlp_in`, `mlp_out` | `hidden` |
| DeepSeek-V4 (`hf_config.hc_mult`) | `pre_attn`, `post_attn`, `mlp_in`, `mlp_out` | `hidden` |
| | `mhc_streams_pre_attn`, `mhc_streams_pre_mlp`, `mhc_streams_final` | `hc_mult * hidden` |

A model family with its own hook geometry plugs in by extending
`build_model_hook_schema`; steering validation and capture pick it up
together.

Error: `vllm.exceptions.SteeringSiteValidationError(SteeringVectorError,
ValueError)`. It is a `VLLMValidationError` (so a 4xx client error) carrying
`parameter=<field name>`, and a `ValueError` so pydantic validators and the
registry routes' existing `ValueError` handlers still catch it.

## Data and control flow

```
once per process
  InputProcessor.steering_site_geometry (cached_property, first use)
      ──resolve_steering_site_geometry(model_config)──▶ geometry
      (None iff steering disabled)
  init_app_state (--enable-steering) ──resolve──▶ SteeringModuleRegistry(site_geometry=...)

per request (OpenAI server)
  CompletionRequest / ChatCompletionRequest
    └─ to_sampling_params ─▶ SamplingParams.__post_init__   (model-blind checks)
  serving.create_completion ─▶ AsyncLLM.generate
    └─ AsyncLLM.add_request
         └─ maybe_pack_inline_steering_for_request(params, dtype,
                site_geometry=input_processor.steering_site_geometry)
              └─ geometry.validate_vectors(spec, field_name)   ◀── site gate
              └─ pack + clear inline fields
         └─ input_processor.process_inputs ─▶ _validate_steering
              └─ geometry.validate_vectors / validate_clamps     ◀── site gate
    SteeringSiteValidationError raised anywhere above:
      AsyncLLM.generate `except VLLMClientError: raise` (not wrapped in
      EngineGenerateError) ─▶ vllm_error_handler ─▶ create_error_response
      ─▶ 400 BadRequestError, message names field[hook][layer] and widths,
         param = field name

offline LLM
  OfflineInferenceMixin._maybe_pack_inline_steering
    └─ maybe_pack_inline_steering_for_request(...,
           site_geometry=llm_engine.input_processor.steering_site_geometry)

named modules
  SteeringModuleRegistry.register
    └─ coerce (packed → lists), hook-name + entry checks
    └─ site_geometry.validate_vectors(tier) / validate_clamps(tier)
```

The inline packer is the seam every inline-vector request crosses before
the multiprocessing boundary: it clears the raw fields after packing, so
the input processor's vector check only sees raw specs on paths that did
not pack (Rust frontend). Clamps are not packed and are checked by the
input processor on every path.

Before this gate a wrong-width row shape-crashed the worker's steering
table population, a stream-hook row was rejected for not being `hidden`
wide, and a row for a nonexistent layer was dropped silently by the
workers (unsteered generation).

### Layer keys

`layer_of` accepts an int (including numpy integers), its decimal string,
or `"layer@pass"` (looped models) and range-checks only the layer part.
Anything else — negative, non-numeric, float, bool, `None` — is rejected.

## Files

| File | Role | Key exports |
| --- | --- | --- |
| `vllm/config/steering_geometry.py` | geometry type, checks, resolver | `SteeringSiteGeometry`, `resolve_steering_site_geometry` |
| `vllm/exceptions.py` | typed error | `SteeringSiteValidationError` |
| `vllm/v1/capture/types.py` | per-family hook geometry | `build_model_hook_schema`, `build_hook_schema`, `default_hook_schema`, `HookSchema` |
| `vllm/config/steering_types.py` | inline packer + gate | `maybe_pack_inline_steering_for_request(sp, torch_dtype, site_geometry=None)` |
| `vllm/v1/engine/input_processor.py` | resolves the geometry once; engine-side gate | `InputProcessor.steering_site_geometry`, `_validate_steering` |
| `vllm/v1/engine/async_llm.py` | HTTP-path packing call | `AsyncLLM.add_request` |
| `vllm/entrypoints/common/offline.py` | offline packing call | `OfflineInferenceMixin._maybe_pack_inline_steering` |
| `vllm/entrypoints/openai/steering/registry.py` | named-module gate | `SteeringModuleRegistry(valid_layer_indices=None, site_geometry=None)` |
| `vllm/entrypoints/launchers/api_server/app_state.py` | builds the registry with the resolved geometry | `init_app_state` |
| `tests/v1/test_steering_site_geometry.py` | CPU unit tests for all of the above | — |

## Invariants and constraints

- One geometry per process, resolved from the model config on first use
  (a cached property, so constructing an `InputProcessor` never touches the
  HF configs); never recomputed per request.
- `InputProcessor.steering_site_geometry` is non-`None` iff
  `steering_config` is set. With steering disabled, requests with steering are rejected before
  the geometry is consulted.
- `site_geometry=None` on the packer and the registry means "no model
  known" and skips only the model-aware checks (unit tests, tooling).
- Every row width check is against the row's own hook, never a single
  model-wide width.
- Steering and capture share hook geometry through `build_model_hook_schema`;
  only hooks that are `SteeringHookPoint` values are steerable.
- All three checks raise `SteeringSiteValidationError` with the field name as
  `parameter`; the frontends answer it with 400 and the message.
