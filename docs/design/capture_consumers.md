# Capture Consumers Design

This document describes the runtime design of the capture-consumer
framework. It is intended for contributors working on the capture
manager, the consumer plugin API, the runner integration, or adding
new built-in consumers.

For user-facing setup and examples, see
[Capture Consumers](../features/capture_consumers.md). For authoring
third-party plugins, see
[Plugin Authoring Guide](../capture_consumers/plugin_authoring.md).

## Responsibilities

The framework has three responsibilities:

1. **Produce** captured activations at well-defined hook points inside
   each decoder layer, respecting per-consumer and per-request
   configuration, without breaking `torch.compile` or CUDA graphs.
2. **Route** captured rows to the right consumer(s) at each forward
   step, preserving batch order and per-key append semantics.
3. **Extend** via an entry-point registry so third parties can add
   new consumer types without modifying vLLM core.

## Architecture

Three layers, cleanly separated:

```text
┌─────────────────────────────────────────────────────────────┐
│                Decoder-layer forward code                   │
│  (model files call apply_layer_steering, which already      │
│   calls maybe_capture_residual)                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ torch.ops.vllm.capture_residual
                          │ (compile-graph-opaque custom op)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     CaptureManager                          │
│  (vllm/v1/capture/manager.py)                               │
│                                                             │
│  - Holds per-consumer global specs                          │
│  - Holds per-request client specs                           │
│  - Builds per-step plans (gather_indices, scratch tensors)  │
│  - Dispatches captured rows to each consumer's sink         │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ CaptureSink protocol
                          │ (submit_chunk, submit_finalize, ...)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     Consumer sinks                          │
│                                                             │
│  - _BatchedAdapter (for CaptureConsumer subclasses)         │
│  - Direct CaptureSink implementations (e.g. filesystem)     │
│  - _DriverQueueShim (worker-side proxy for driver-side      │
│    consumers; events cross to the driver via                │
│    torch.multiprocessing.Queue)                             │
└─────────────────────────────────────────────────────────────┘
```

The manager is the single producer. Sinks are the consumer-facing
interface. `CaptureConsumer` is the ergonomic base class most plugin
authors subclass; the framework wraps each instance in a
`_BatchedAdapter` that buffers chunks and delivers a single
concatenated tensor to `on_capture` per key. Consumers that need true
streaming implement `CaptureSink` directly — the built-in filesystem
consumer does this.

## Module Layout

```text
vllm/v1/capture/
├── __init__.py                 # Public re-exports
├── types.py                    # Core types (torch-aware)
├── sink.py                     # CaptureSink protocol
├── consumer.py                 # CaptureConsumer + _BatchedAdapter
├── config.py                   # CaptureConsumersConfig + CaptureConsumerSpec
│                               #   + parse_consumer_spec, validate_consumer_specs
├── errors.py                   # CaptureValidationError,
│                               #   UnknownCaptureConsumerError
├── registry.py                 # Entry-point discovery, build_consumer(s)
├── plan.py                     # StepCapturePlan, CapturePositionEntry,
│                               #   CaptureBatchView
├── manager.py                  # CaptureManager
├── driver_bridge.py            # _DriverQueueShim, _DriverReceiver,
│                               #   install_driver_consumer
└── consumers/
    ├── __init__.py
    ├── logging.py              # LoggingConsumer (CaptureConsumer subclass)
    └── filesystem/
        ├── __init__.py         # Re-exports
        ├── consumer.py         # FilesystemConsumer (direct CaptureSink)
        ├── types.py            # FilesystemCaptureRequest, params
        ├── validation.py       # validate_filesystem_request
        └── writer.py           # ActivationWriter thread pool
```

Model-facing helpers stay at
`vllm/model_executor/layers/activation_capture.py` (custom op, hook-ID
table, `maybe_capture_residual`, `set_active_capture_manager`). The
capture manager imports from it; it does not import from the manager.

Re-export shims:

- `vllm/config/capture_consumers.py` — re-exports
  `CaptureConsumersConfig` / `CaptureConsumerSpec` so `vllm/config/`
  can follow its own relative-import pattern.

## Core Types

All of these live in `vllm/v1/capture/types.py`.

```python
VllmInternalRequestId = NewType("VllmInternalRequestId", str)
CaptureKey = tuple[VllmInternalRequestId, int, str]
# (request id, layer index, hook name)

HookName = Literal[
    # Standard residual hooks (every model): full residual, model dtype.
    "pre_attn", "post_attn", "post_mlp", "mlp_in", "mlp_out",
    # DeepSeek-V4 mHC: multi-stream residual (hc_mult * hidden, bf16).
    "mhc_streams_pre_attn", "mhc_streams_pre_mlp", "mhc_streams_final",
    # DeepSeek-V4 mHC: stream-mixing coefficients (fp32).
    "mhc_attn_post_mix", "mhc_ffn_post_mix",
    "mhc_attn_res_mix", "mhc_ffn_res_mix",
]
PositionSelector = (
    Literal["last_prompt", "all_prompt", "all_generated", "all"]
    | list[int]
)

# Per-hook row geometry. Standard hooks are (hidden_size, model_dtype,
# (hidden_size,)); mHC hooks vary in width and dtype (e.g. an attn res_mix
# row is (hc_mult * hc_mult,) fp32, reshaped back to (hc_mult, hc_mult)).
@dataclass(frozen=True)
class HookSchema:
    width: int                  # flattened per-row element count
    dtype: torch.dtype
    logical_shape: tuple[int, ...]   # per-row shape; prod == width

@dataclass(frozen=True)
class CaptureSpec:
    hooks: dict[HookName, list[int]]
    positions: PositionSelector

@dataclass
class CaptureChunk:
    key: CaptureKey
    tensor: torch.Tensor        # CPU, shape (num_rows, width)
    dtype: torch.dtype
    row_offset: int
    step_index: int
    metadata: dict[str, Any]    # incl. "row_shape" and per-row "positions"

@dataclass
class CaptureFinalize:
    key: CaptureKey
    sidecar: dict[str, Any]

CaptureStatus = Literal["pending", "ok", "partial_error", "error", "not_requested"]

@dataclass
class CaptureResult:
    key: CaptureKey
    status: CaptureStatus
    error: str | None = None
    payload: Any = None

@dataclass
class CaptureContext:
    vllm_internal_request_id: VllmInternalRequestId
    num_prompt_tokens: int
    num_computed_tokens: int
    num_hidden_layers: int
    hidden_size: int
    element_size_bytes: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    hook_schema: dict[str, HookSchema]   # which hooks the model taps + geometry
```

`HookName` must stay in lockstep with `_HOOK_NAME_TO_ID` in
`vllm/model_executor/layers/activation_capture.py`.

**Per-hook schema.** `build_hook_schema(hidden_size, dtype, hc_mult)` (in
`types.py`) returns the hooks a model taps, keyed by name, each with its
`HookSchema`. Without `hc_mult` it is the standard wired residual hooks
(`pre_attn` / `post_attn` / `post_mlp`); a model exposing `hf_config.hc_mult`
(DeepSeek-V4) gets the mHC hooks instead, sized from `hc_mult`. The runner
and the OpenAI entrypoints build it from `model_config` and pass it on
`CaptureContext` (admission) and to the `CaptureManager` (buffer sizing /
scratch dtype). The schema is the source of truth for *which hooks are
tapped*: admission rejects any hook not in it, so `mlp_in` / `mlp_out` /
`mhc_*` are accepted only on models that wire them.

**Model-level hooks.** `MODEL_LEVEL_HOOKS` (currently `{mhc_streams_final}`)
fire once per request at the model tail rather than per decoder layer. They
are keyed to the last layer (`num_hidden_layers - 1`); admission ignores
their layer selector and normalizes to that index, so a caller writes
`{"mhc_streams_final": "all"}` without knowing it.

## Sinks and Consumers

### `CaptureSink` (protocol)

The low-level streaming interface:

```python
class CaptureSink(Protocol):
    location: ClassVar[Literal["worker", "driver"]]

    def submit_chunk(self, chunk: CaptureChunk) -> None: ...
    def submit_chunk_batch(self, chunks: list[CaptureChunk]) -> None: ...
    def submit_finalize(self, finalize: CaptureFinalize) -> None: ...
    def get_result(self, key: CaptureKey) -> CaptureResult | None: ...
    def shutdown(self, timeout: float = 30.0) -> None: ...
```

Ordering guarantees:

- For a given key, chunks arrive in `row_offset` order.
- `CaptureFinalize` for a key arrives after all chunks for that key.
- Different keys have no ordering relationship.
- `submit_chunk_batch` delivers all of one dispatch step's chunks for a
  sink in a single call (chunks within the batch retain submission order).
  It is **optional**: the default forwards each chunk to `submit_chunk`, and
  the manager calls it opportunistically (via `getattr`). Sinks that can
  amortize per-chunk overhead across the step — locking, write-task
  creation, payload concatenation — override it; the filesystem consumer
  does so for the `packed` layout (one WriteTask + one lock per request per
  step instead of one per `(layer, hook)`).
- All methods are called from the manager's dispatch thread; direct
  `CaptureSink` implementations are responsible for their own thread
  safety (the manager does not serialize calls across sinks).

### `CaptureConsumer` (base class)

The ergonomic user-facing base. Class-level metadata:

- `location: "worker" | "driver"` (default `"worker"`).
- `required_sidecar_fields: frozenset[str]` (default empty).
- `reads_client_spec: bool` (default `False`).

Override points:

- `__init__(self, vllm_config, params)`.
- `global_capture_spec(self) -> CaptureSpec | None`.
- `validate_client_spec(self, raw_spec, ctx) -> CaptureSpec`.
- `on_capture(self, key, tensor, sidecar) -> None` — the main one.
- `on_error(self, key, error) -> None`.
- `shutdown(self, timeout) -> None`.

### `_BatchedAdapter`

`CaptureConsumer` subclasses are wrapped at registration in
`_BatchedAdapter`, which implements `CaptureSink`:

- `submit_chunk` appends `(row_offset, tensor)` to an in-memory list
  keyed by `CaptureKey` under a mutex.
- `submit_finalize` pops the list, sorts by `row_offset` (defensive —
  the manager already preserves order), concatenates, and calls
  `on_capture`. Exceptions are caught; the result becomes
  `CaptureResult(status="error")` and `on_error` is invoked
  best-effort.
- `get_result` returns the cached terminal result or `None`.

Direct `CaptureSink` implementations (e.g. `FilesystemConsumer`) are
not wrapped — the registry installs them as-is.

## Config and Registry

### `CaptureConsumersConfig`

```python
@dataclass
class CaptureConsumerSpec:
    name: str                             # entry-point name
    instance_name: str | None = None      # disambiguates duplicates
    params: dict[str, Any]                # opaque per-consumer

@dataclass
class CaptureConsumersConfig:
    consumers: list[CaptureConsumerSpec]

    def compute_hash(self) -> str: ...    # 16-char md5
```

`VllmConfig.capture_consumers_config` holds an optional
`CaptureConsumersConfig`; when set, its hash contributes to the
compile-cache key.

Entry surfaces:

- `--capture-consumers NAME:k=v,k=v` on the CLI — repeatable,
  parsed via `parse_consumer_spec`.
- Programmatic `EngineArgs.capture_consumers_config_override` for
  code paths (like `LLM(...)`) that build the config directly.
- `LLM(capture_consumers=[...])` accepts both dict entries (become
  `CaptureConsumerSpec`s) and pre-constructed `CaptureConsumer`
  instances; both ride on `CaptureConsumersConfig` (`.consumers` and
  `.instances`) so they survive the `EngineArgs → VllmConfig` plumbing
  and reach the worker.

`validate_consumer_specs` enforces:

- non-empty `name`;
- uniqueness of `instance_name or name` across the list.

### Registry

`vllm/v1/capture/registry.py` enumerates
`importlib.metadata.entry_points(group="vllm.capture_consumers")`
once per process and caches the resolved `name -> class` map. Entries
may be `CaptureConsumer` subclasses **or** classes that implement
`CaptureSink` directly (detected via `submit_chunk` /
`submit_finalize` / `get_result` attributes).

`build_consumers(vllm_config, consumer_instances=None)` is the runner
entry point. It returns `(sinks, validators, name_to_index)`:

- `sinks` — tuple of `CaptureSink`s (in config order, then
  pre-constructed instances).
- `validators` — parallel tuple. For `_BatchedAdapter`-wrapped
  consumers this is the underlying `CaptureConsumer`; for direct
  sinks it is the sink itself; for driver shims it is the original
  driver-side consumer reference. This is the object the manager
  calls `global_capture_spec()` / `validate_client_spec()` on.
- `name_to_index` — `instance_name or name` mapped onto the sink
  index. Duplicates get a `#2`, `#3`, ... suffix so the map stays
  injective.

`_wrap_consumer` decides:

- `location == "driver"` → `install_driver_consumer` (builds the shim
    - receiver).
- `location == "worker"` and instance is a `CaptureConsumer` → wrap
  in `_BatchedAdapter`.
- `location == "worker"` and already a `CaptureSink` → install as-is.

Pre-constructed instances (from `LLM(capture_consumers=[instance])`)
must have `location = "driver"` — they were built in the driver
process and cannot be shipped to the worker.

## Manager Runtime

`CaptureManager` is one-per-runner. Constructed with the sink tuple,
the parallel tuple of global specs (one `CaptureSpec | None` per
consumer), plus model shape info and the device scratch tensors
should live on.

### Registration

`register_request(req_id, client_specs, num_prompt_tokens,
sidecar_fields)` merges:

1. Every consumer's `global_capture_spec()` (already resolved at
   runner init).
2. Per-consumer client specs from `client_specs: dict[int,
   CaptureSpec]`.

Merge rule: a client spec for consumer `i` **replaces** (does not
union with) the global spec for that consumer. Consumers with no
spec are inactive for this request. If no consumer has a spec, the
call silently returns.

Hook layers are validated against `num_hidden_layers`. Position
selectors are classified into *static* (`last_prompt`, `all_prompt`,
explicit list — resolved once at registration) and *dynamic*
(`all_generated`, `all` — re-expanded each step).

### Per-Step Plan

`build_step_plan(batch_view)` walks the batch in order:

1. For each request with a registered spec, resolve each consumer's
   position selector against `[num_computed, num_computed +
   num_scheduled)`.
2. For each `(layer, hook)` that any consumer wants, record the
   absolute batch row index plus a per-position `consumer_mask`
   bitset (bit *i* set ⇒ consumer *i* wants this row).
3. Allocate an `int64` row-index tensor for each `(layer, hook)`,
   routed by capture path (see below), plus `scratch_dtype` (model
   residual dtype). `scratch_gpu` starts empty and is populated during
   (client keys) or after (global keys) the forward.

Gather happens **once** per `(layer, hook)` regardless of how many
consumers want it. Fan-out happens at dispatch time via the
`consumer_mask`.

**Two gather paths.** Each `(layer, hook)` is routed by whether it
belongs to a **global** spec (one whose `(layer, hook)` set is fixed at
startup, served by the CUDA-graph-safe persistent buffer) or only a
**client** spec (per-request, dynamic):

- **Client keys → `gather_indices`.** The in-hook `index_select`
  (variable output size, fresh allocation) — only valid eager, so the
  step gate forces eager whenever a client key captures.
- **Global keys → `global_gather_indices`.** Served from a persistent
  per-key buffer that `on_hook` fills with a fixed-shape full-residual
  `copy_` (graph-safe — recorded into the cudagraph at warmup). The host
  slices these rows out of the buffer *after* the forward. A key wanted
  by both a global and a client consumer this step routes to the global
  path; the buffer holds the full residual, so any consumer's rows can be
  sliced from it.

`StepCapturePlan` holds:

- `gather_indices: dict[(layer, hook), Tensor]` — client keys (in-hook).
- `global_gather_indices: dict[(layer, hook), Tensor]` — global keys
  (sliced from the persistent buffer post-forward).
- `scratch_gpu: dict[(layer, hook), Tensor]`
- `scratch_dtype: dict[(layer, hook), torch.dtype]`
- `entries: list[CapturePositionEntry]` — one per captured row, with
  `(request_id, layer, hook, logical_pos, scratch_row, step_index,
  consumer_mask)`. `scratch_row` ordering is identical for both paths,
  so dispatch treats global and client keys uniformly.
- `request_errors: dict[req_id, str]` — registration-time or
  step-time failures.

### Forward and Dispatch

During the forward:

- `apply_layer_steering` calls `maybe_capture_residual` **before**
  applying the steering add — captures see the pristine residual
  (invariant 1).
- `maybe_capture_residual` returns immediately if no manager is
  installed (invariant 2 — cold path is free). When a manager is
  active, it dispatches
  `torch.ops.vllm.capture_residual(hidden_states, layer_idx,
  hook_id)`. The op is registered with
  `mutates_args=["hidden_states"]` — a deliberate white lie so
  `torch.compile` does not DCE it.
- The op looks up the active manager and calls `on_hook(layer_idx,
  hook_name, hidden_states)`. `hidden_states` is never mutated. `on_hook`
  has two independent branches:
    - **Global key** (`(layer, hook)` in `_global_buffers`): a
      fixed-shape `buf[:n].copy_(hidden_states)` of the full residual
      into the persistent buffer. This runs *independently of the
      per-step plan*, so it fires during the warmup capture pass (where
      the `copy_` kernel is **recorded into every cudagraph descriptor**)
      and on eager steps alike; at replay no Python runs but the recorded
      copy reproduces against the persistent address. Collective-free, so
      recording it on only the capturer rank's graphs is safe.
    - **Client key** (`plan.gather_indices`): the dynamic
      `index_select` into `plan.scratch_gpu[(layer, hook)]`. Allocates a
      fresh, variable-size output, so it only runs eager.

After the forward step, the runner calls `consume_step_plan()` to
take ownership of the plan and `dispatch_step_captures(plan)`, which
first calls `_materialize_global_keys(plan)` — for each global key it
slices `buf.index_select(0, global_gather_indices[key])` into
`scratch_gpu` (eager, off-graph, on the post-forward compute stream),
after which global and client keys are handled identically:

1. For each consumer index `i`, group entries where bit `i` is set
   in `consumer_mask` by `(request_id, layer, hook)`.
2. For each group, `index_select` the consumer's rows out of the
   scratch tensor, `.cpu()` them, and build a `CaptureChunk` (`metadata`
   carries `consumer_index` and the logical positions). The step's chunks
   for a consumer are collected and handed over in one `submit_chunk_batch`
   call (falling back to per-chunk `submit_chunk` if the sink lacks it), so
   sinks can amortize per-chunk overhead across the whole step.
3. Wrap per-consumer dispatch in `try/except` so a failing sink
   never prevents delivery to the others (invariant 9). A failing
   dispatch records an error on every request that consumer was
   handling; the error is propagated through the per-request
   `CaptureResult`.

### Finalize

`finalize_request(req_id)` is called when the request finishes (any
finish reason):

1. Pop the `_RequestCaptureState`.
2. For every consumer that had a spec, for every `(layer, hook)` in
   that spec, build a `CaptureFinalize` with the cached
   `sidecar_fields + {"consumer_index": i}` and call
   `sink.submit_finalize`. Exceptions are logged per consumer and
   do not fail the request.
3. For each consumer, call `sink.get_result(first_key)` (current
   implementation returns a single representative result per
   consumer — aggregating across keys is a TODO) and synthesize
   `CaptureResult(status="ok")` if the sink hasn't produced a
   terminal result yet.

Returns `dict[consumer_index, CaptureResult]`; the runner maps
indices back to consumer names via `_capture_index_to_name`.

## Runner Wiring

`GPUModelRunner.__init__` constructs the manager when
`vllm_config.capture_consumers_config is not None`:

1. Call `registry.build_consumers(vllm_config,
   consumer_instances=vllm_config.capture_consumers_config.instances)`.
2. For each validator, call `global_capture_spec()` defensively (any
   exception becomes `None`).
3. Instantiate `CaptureManager(sinks, global_specs, num_hidden_layers,
   hidden_size, model_dtype, device)`.
4. Call `set_active_capture_manager(manager)` so
   `maybe_capture_residual` finds it from inside the compiled graph.

Per-step hooks:

- Before the cudagraph dispatch decision, the runner builds the batch
  view (matching `SteeringModelRunnerMixin._update_steering_buffers`
  offset walk exactly so gather indices line up). The view feeds two
  things: the rank-replicated `CaptureStepGate` (which decides whether
  this step must run eager — see [Force-eager step gate](#force-eager-step-gate)),
  and, on the capturer rank, `manager.build_step_plan`.
- `_finalize_capture_step()` — calls `manager.consume_step_plan()`
  and `manager.dispatch_step_captures(plan)` after the forward pass.

Per-request finalize: when a request completes, the runner calls
`_finalize_capture_for_request_async(req_id)`, which has the manager pop
the request's capture state on the step thread (so `_requests` stays
single-threaded) and run the blocking finalize — `submit_finalize` plus
the per-key `wait_for_result` round-trips — on the manager's dedicated
finalize thread via `manager.finalize_request_async(req_id, on_complete)`.
The `on_complete` callback (on the finalize thread) translates indices
back to names and stashes the dict on the lock-guarded
`_pending_capture_results`, which a later step drains onto
`ModelRunnerOutput.capture_results`. Result delivery is therefore
**best-effort**: if finalize has not completed by the time the request's
output is emitted, the result is absent from the response even though the
captured data was written. The synchronous `manager.finalize_request` is
retained for tests and inline callers.

### Force-eager step gate

`CaptureManager.on_hook`'s **client**-key gather (`index_select`) allocates
a fresh, variable-size output each step and cannot run inside a replayed
CUDA graph, so any step that gathers for a client spec must run eager.
(Global keys take the graph-safe persistent-buffer copy instead — see
[Forward and Dispatch](#forward-and-dispatch) — and never force eager.)
Under TP/PP every rank must reach the *same* eager-vs-cudagraph decision or
`num_tokens_padded` diverges and the TP all-reduce / PP send-recv deadlock;
a per-step collective to agree would itself deadlock inside PP's
asynchronous pipeline.

`CaptureStepGate` (`vllm/v1/capture/step_gate.py`) resolves this with no
collective. It is built on *every* rank, populated from the broadcast
new-request stream (`register`) and finished stream (`drop`), and each step
evaluates `step_captures(view)` purely from the rank-identical
`scheduler_output` plus each request's client capture spec (which rides in
`SamplingParams` on every rank). Identical inputs + a pure predicate ⇒
identical decision by construction. The gate ignores pipeline-stage layer
filtering and admission validation, so it is a strict **superset** of the
capturer's actual client gather: it never leaves a client-capturing step in
cudagraph mode (which would silently no-op the gather), only occasionally
over-forces a harmless extra eager step. Position logic is shared with
`build_step_plan` via `selector_hits_window`. Global specs no longer force
eager — they are served by the persistent buffer — so a `logging`-style
global consumer keeps full cudagraph speed; `force_all` remains only as a
manual always-eager escape hatch (off by default).

## Driver Bridge

For `location = "driver"` consumers, `install_driver_consumer`:

1. Creates an event `torch.multiprocessing.Queue` and a result
   queue (both bounded; default `queue_size=1024`, `timeout=30s`).
2. Starts `_DriverReceiver` as a daemon thread in the driver process
   that pops events, runs them through a `_BatchedAdapter` around
   the user's consumer, and posts `CaptureResult`s back on the
   result queue.
3. Returns `_DriverQueueShim`, which the capture manager installs as
   the worker-side sink.

The shim serializes `("chunk", CaptureChunk)` and `("finalize",
CaptureFinalize)` events. `put` respects the bounded queue — on
`queue.Full`, the shim records `CaptureResult(status="partial_error")`
for the affected key and logs a warning. On `get_result`, the shim
drains available results from the result queue before returning.

Tensors cross via `torch.multiprocessing` shared memory; metadata
dicts pickle normally. Receiver exceptions land on the affected key
as `status="error"` via `_BatchedAdapter`'s isolation.

## OpenAI Entrypoint

Both chat completion and completion endpoints run
`_admit_capture(sampling_params, request_id)` before forwarding to
the engine:

1. If `sampling_params.capture is None`, skip.
2. Build a `CaptureContext` from `engine_client.vllm_config` (model
   shape, TP/PP sizes, `num_computed_tokens=0` — admission runs before
   the prefix cache is consulted).
3. For each `(name, raw_spec)` in `sampling_params.capture`:
   - Look up the consumer by `instance_name or name` in the
     per-serving-instance `self._capture_consumers` dict (built
     during serving init via `registry.build_consumer`).
   - Unknown name → HTTP 400.
   - Call `consumer.validate_client_spec(raw_spec, ctx)`.
   - `CaptureValidationError` → HTTP 400 with the message.
4. Mutate `sampling_params.capture` in place, replacing each raw
   value with the resolved `CaptureSpec`. The runner tolerates
   both shapes.

Serving-instance consumers are separate from the runner's consumers
— they exist only for admission-time validation. The actual capture
dispatch still goes through the runner-side manager.

## Built-in Consumers

### `LoggingConsumer`

`vllm/v1/capture/consumers/logging.py`.

- `CaptureConsumer` subclass → wrapped in `_BatchedAdapter`.
- `location = "worker"`, `reads_client_spec = False`.
- `global_capture_spec` returns a `CaptureSpec` built from
  `params["hooks"]` and `params.get("positions", "last_prompt")`.
- `on_capture` emits one `logger.log(level, "capture key=... rows=N
  dtype=D", ...)` line and discards the tensor.

Primary role: reference for plugin authors, smoke test for the
framework.

### `FilesystemConsumer`

`vllm/v1/capture/consumers/filesystem/`.

Implements `CaptureSink` directly so long captures stream to disk
without buffering the whole tensor. Owns a private `ActivationWriter`
thread pool (`writer.py`).

- `location = "worker"`, `reads_client_spec = True`.
- `global_capture_spec()` returns `None` — captures are always
  per-request via `SamplingParams.capture["filesystem"]`.
- `validate_client_spec` accepts `FilesystemCaptureRequest` or a
  matching dict, then lazily delegates to
  `validation.validate_filesystem_request` (lazy to avoid pulling
  pydantic in at module import).

Per-chunk flow:

1. `submit_chunk` extracts `tag_slug` / `request_id_slug` from
   `chunk.metadata` (defaults `"default"` / `str(request_id)` if
   absent — in practice the runner currently doesn't populate
   these, so the defaults are what's on disk).
2. Build `{root}/{tag_slug}/{request_id_slug}/{layer}_{hook}.bin`.
3. `tensor.numpy().tobytes()`; submit a `WriteTask(path, payload,
   append=True, key)` to the writer.
4. Writer thread appends to `{path}.bin.tmp`, holding an fd in a
   per-thread LRU cache keyed by `CaptureKey`.

On finalize:

1. Look up the cached `(tag_slug, request_id_slug)` for the key;
   allow `finalize.sidecar` to override either.
2. Build a `FinalizeTask` with `bin_path`, `sidecar_path` (same stem
   with `.json`), and a sidecar dict `{request_id, layer, hook,
   **finalize.sidecar}`.
3. Writer thread `fsync`s the `.bin.tmp`, `os.replace`s it to the
   final `.bin`, writes + `fsync`s + renames the sidecar JSON.

**Layout modes** (`FilesystemCaptureRequest.layout`, else the consumer's
`default_layout`; `per_file` is the default and unchanged behavior):

- **`per_file`** — the flow above: one `.bin` + `.json` per
  `(request, layer, hook)`. Lowest latency; supports mid-request
  streaming (a reader can tail a `.bin` as steps append). File count
  scales with `requests × layers × hooks`.
- **`packed`** — one `packed.bin` + one `packed.json` index per
  *request*, all `(layer, hook)` tensors concatenated. The consumer
  routes every chunk of a request to a single writer key (one fd, one
  file), records a per-chunk index entry `{layer, hook, offset, nbytes,
  shape, row_shape, dtype, positions}` (`row_shape` is the per-row
  logical shape for reshaping e.g. mHC `(hc_mult, hidden)` rows;
  per-entry `dtype` lets one request mix dtypes such as bf16 streams and
  fp32 mHC coefficients; `positions` is the per-row absolute token
  position), and — because per-key finalizes arrive in one synchronous
  burst after the dispatch-drain barrier — publishes the file only once
  **all** expected `(layer, hook)` keys have finalized. Every key's
  `CaptureResult` then maps to the single packed `WriteResult`. Cuts
  file count by `layers × hooks` — the throughput lever on network
  mounts (~4.5× on NFS in `bench_capture_packed.py`); a capture is
  readable once its request finalizes. Packing is entirely
  consumer-side; the writer is unchanged.

Both layouts write raw residual-dtype bytes (bf16 as `uint16`) and a
self-describing sidecar carrying `dtype` + `shape`. `reader.py`
(`read_per_file` / `read_packed` / `read_request`, NumPy-only) decodes
either layout; `read_request` auto-detects by the presence of
`packed.json`. Packed entries are per-chunk, so a `(layer, hook)`
spanning multiple steps appears as several entries the reader
concatenates in offset order.

Writer details (`writer.py`):

- One `queue.Queue` per thread, partitioned by `hash(request_id) %
  num_threads` — preserves per-key append order without cross-thread
  locks.
- Per-thread LRU fd cache (default 256 entries); eviction `fsync`s +
  closes the fd.
- `fsync` (default True): when False, skip every `os.fsync`. On NFS the
  per-file fsync is near-redundant with the close-time COMMIT, so it is
  mostly a no-op there but a real durability/throughput knob on other
  backends.
- `atomic_publish` (default True): when False, write straight to the
  final path (no `.tmp` + rename), dropping two rename RPCs per file at
  the cost of atomic visibility. Requires `on_collision="overwrite"`.
- Collision policy (`overwrite` / `error` / `suffix`) applied at
  finalize.
- Structured `WriteError` with errno, path, key; surfaces back on
  `WriteResult.error`.
- `get_result(key)` maps `WriteResult` → `CaptureResult`: status
  pass-through; payload is `[bin_path, sidecar_path]` on success,
  `None` on error.

**Validation constraints** (`validation.py`):

- TP / PP / EP / DP are all accepted for the replicated residual hooks
  (no parallel-size rejection). See
  [Capture Consumers under Parallelism](capture_parallelism.md).
- Every hook name is one the model taps, i.e. present in
  `ctx.hook_schema` (falls back to `{pre_attn, post_attn, post_mlp}` when
  no schema is supplied). On a DeepSeek-V4 model this also accepts the
  `mhc_*` hooks and `mlp_in` / `mlp_out`; on a standard model those are
  rejected.
- Every resolved layer is in `[0, num_hidden_layers)`, the **global**
  layer count (admission validates the full layer space; the runner then
  filters each pipeline stage's spec to its owned slice). Exception:
  model-level hooks (`MODEL_LEVEL_HOOKS`, e.g. `mhc_streams_final`) ignore
  their layer selector and normalize to `num_hidden_layers - 1`.
- Tag / request_id: non-empty, ≤256 chars, no `..`, no leading `/`;
  characters outside `[a-zA-Z0-9._-]` become `_`.
- Explicit positions ≥ `num_computed_tokens` (reject prefix-cache
  hits that were never forwarded).
- `all_generated` / `all` position kinds are deferred to the
  runner — the validator returns them symbolically so the manager
  can re-resolve each step.

## Invariants

1. **Capture reads the pristine residual.** `maybe_capture_residual`
   fires before the steering add at the same hook point. Inherited
   from the activation-storing design.
2. **Cold path is free.** When no consumer is configured,
   `maybe_capture_residual` returns on the first `None` check;
   `torch.compile` constant-folds the call away and the compiled
   graph contains no `capture_residual` ops.
3. **Per-step plan is batch-order consistent.** `_build_capture_batch_view`
   walks `input_batch.req_ids` and cumulates `token_offset` by
   `num_scheduled_map[req_id]` in exactly the same order the
   steering buffer walk uses. Any deviation corrupts `gather_indices`.
4. **Multi-step captures preserve append order.** The writer
   partitions work by `hash(request_id) % num_threads`, so chunks
   for a given key all land on the same thread and append in
   submission order. `_BatchedAdapter` also sorts by `row_offset`
   defensively.
5. **Finalize is atomic per consumer.** Filesystem writes land on
   `.tmp` and `os.replace` to the final name only after `fsync`. A
   `CaptureResult.status == "ok"` means the bytes are durable.
6. **Partial failures never abort generation.** Sink errors, queue
   overflows, shutdown timeouts — all surface as
   `partial_error` / `error` on `CaptureResult`. Token streaming
   continues.
7. **Prefix-cache positions rejected at admission.** Filesystem
   validator raises `CaptureValidationError` on any explicit
   position below `num_computed_tokens`.
8. **Parallelism is supported for replicated residual hooks.** The
   residual stream is replicated across the TP/EP plane within each
   pipeline stage, so TP rank 0 of each stage captures its
   global-indexed layers and the engine unions the per-stage results
   (`merge_capture_results`). Worker-location consumers (incl.
   `filesystem`) work under TP/PP/EP/DP; sharded-activation capture and
   single-process driver-side gather across PP stages remain future
   work. See [Capture Consumers under Parallelism](capture_parallelism.md).
9. **Consumer isolation.** `dispatch_step_captures` wraps each
   consumer's slice-and-submit in `try/except`; `_BatchedAdapter.
   submit_finalize` catches `on_capture` exceptions and records
   them as `status="error"`; the driver bridge catches receiver
   exceptions the same way.
10. **`vllm_internal_request_id` is the only identity the framework
    guarantees.** Consumers opt into richer sidecar fields via
    `required_sidecar_fields`; the current runner wiring propagates
    only `consumer_index` plus whatever the consumer put in its
    own `client_specs`/admission path.

## Prefix-Cache Interaction

A capture tap reads the residual stream at a `(layer, hook)` point for a
token position. That residual exists only if the position is forwarded
through the model. Prefix caching reuses the **KV** of a matched prefix
and skips the forward pass for those positions, so a tap on a cached
prompt position never fires — the activation simply does not exist. The
conflict is therefore **per-position**: it arises only when a capture
taps a position that a prefix-cache hit would serve from cache.

The fix is staged in three layers, B → C → A:

- **B — admission class gate (implemented).** A capture only conflicts
  when it taps a *prompt-range* position. `spec_touches_prompt`
  (`vllm/v1/capture/types.py`) classifies a resolved `CaptureSpec`
  against `num_prompt_tokens`: `"all_generated"` and explicit lists
  entirely `>= num_prompt_tokens` are generated-only and never conflict;
  `"all"`, unresolved `"all_prompt"`/`"last_prompt"`, and any
  prompt-range index are prompt-touching. The OpenAI entrypoint's
  `_admit_capture` (chat + completion serving) computes this once the
  consumer specs are resolved and records it on
  `SamplingParams.capture_touches_prompt`
  (`True`/`False`/`None`-unclassified) — see also C, which records the
  re-forward floor alongside it. `Request.get_skip_reading_prefix_cache`
  skips prefix-cache reuse entirely only when the flag is `None`
  (unclassified). Generated-only captures keep full prefix caching;
  prompt-touching captures fall through to C's clamp rather than skipping
  wholesale. Both entry points resolve specs through the shared
  `resolve_capture_prefix_flags` (`vllm/v1/capture/admission.py`): the
  OpenAI `_admit_capture` calls it (mapping errors to HTTP 400), and the
  offline `LLM` path calls it from `InputProcessor.process_inputs`
  (`vllm/v1/engine/input_processor.py`) before the request reaches the
  scheduler. The offline call is idempotent (it only fires when the flag
  is still `None`, so served requests already stamped by `_admit_capture`
  skip it). Both entry points build their `name -> validator` map via the
  shared `build_admission_validators` (`vllm/v1/capture/registry.py`),
  which keys consumers exactly as the worker's `name_to_index` — dict/spec
  form rebuilt from `vllm_config.capture_consumers_config.consumers`,
  pre-built instances (`LLM(capture_consumers=[obj])`) used directly from
  `.instances` and keyed by class name — so the same client name resolves
  to the same consumer at admission and at the worker. Resolution is
  best-effort: an invalid spec leaves the flag `None` (conservatively
  disabled, correct) and the worker re-validates the raw spec at
  registration.

- **C — schedule-time hit clamp (implemented).** Rather than skipping
  reuse for a prompt-touching capture, clamp it. `_admit_capture` records
  the request-wide re-forward floor — the lowest captured prompt position
  across consumers, via `min_captured_prompt_position` — on
  `SamplingParams.capture_min_prompt_position`, which travels to the
  engine core on the sampling params. `Request.get_capture_prefix_cache_limit`
  surfaces it, and `KVCacheManager.get_computed_blocks` caps
  `max_cache_hit_length` at that floor before calling
  `find_longest_cache_hit` (which block-aligns the cap down, so the
  request stays block-size aligned). The prefix below the floor is reused
  from cache; the floor and everything after are re-forwarded so their
  residuals can be captured. A floor of `0` (`all_prompt`/`all`) clamps to
  no reuse; `last_prompt` clamps to almost the whole prompt. The floor is
  computed by the shared `resolve_capture_prefix_flags` on both paths (see
  B), so the offline `LLM` path gets the same clamp as the served path for
  both dict/spec-form and instance-form consumers.

- **A — activation store (implemented).**
  A pristine residual is a pure function of the prefix token ids and the
  weights, so it is content-addressable by the same block-hash chain the
  KV cache uses. A CPU-RAM, bounded, LRU, drop-on-eviction store
  (`vllm/v1/capture/activation_store.py`, `ActivationStore`) keyed by
  `(block_hash, offset_in_block, layer, hook)` lets repeated `all_prompt`
  captures over a fixed corpus serve from the store instead of
  re-forwarding. A miss or eviction falls through to C (re-forward) — the
  store is a pure cache, never a source of truth; durability is the
  consumer's output, not the store's. Dtype/weights are not in the key:
  one model runs at one dtype, and `invalidate_all` is called wholesale on
  weight update (same path as `reset_prefix_cache`).

  Two findings shaped the design. (1) Capture is TP1/PP1 only, and
  `world_size==1` uses `UniProcExecutor`, so the scheduler
  (`KVCacheManager`) and the capture manager run in **one process** — a
  single shared `ActivationStore` instance, no cross-process coherence;
  access is mutex-guarded because write-through runs on the capture
  dispatch thread while the scheduler reads on the main thread. (2)
  Activation steering is present in this tree, and the store stays correct
  with steering **on** without any extra gate: the store keys off the
  request's `block_hashes`, which already fold the request's steering
  config hash into the block hash (`_gen_steering_extra_hash_keys` in
  `vllm/v1/core/kv_cache_utils.py`, the same mechanism that makes KV prefix
  caching steering-aware). A captured residual at a layer is pre-steering
  at that hook but still reflects *upstream* steering, so two requests
  reusing a prefix must share the same steering config to share residuals —
  which is exactly what the steering-aware block hash enforces. A request
  therefore only ever retrieves store rows written by an
  identically-steered prior request, so the served value matches what its
  own forward would produce.

  Remaining wiring sub-steps (block hashes live only in the engine-core
  `Request`, not in the `CaptureManager`):
  - **A.1 store core (done):** the data structure + global accessor +
    tests.
  - **A.2 write-through (done):** the worker populates the store with
    freshly-captured prompt residuals. `NewRequestData.from_request`
    carries the request's prompt `block_hashes` and the scheduler's
    `hash_block_size` to the worker (only for capture requests); the
    `CaptureManager` keys each captured prompt row via `activation_key`
    (`(block_hash, offset_in_block, layer, hook)`) and writes a clone into
    the store from the dispatch thread, once per `(request, layer, hook,
    position)`. The hash block size travels with the hashes rather than
    being re-derived worker-side, because it can diverge from the KV block
    size; using the wrong granularity would silently misalign keys. No
    behavior change yet — this only fills the store.
  - **A.3 read floor + serve-inject (done):** when the whole captured
    prefix is store-resident, reuse the full KV prefix (skip the
    re-forward) and inject the stored rows. The race-safe primitives:
    `ActivationStore.extract_all` (an atomic all-or-nothing snapshot under
    the store lock — taking the rows at decision time frees the read path
    from a check-then-serve eviction race against dispatch-thread
    write-through/eviction), the same-process serve bridge
    (`stash_pending_serve` / `pop_pending_serve`, sound because capture is
    TP1/PP1), and `try_reserve_store_serve` (the scheduler-side reserve:
    compose keys, extract, stash). Read floor:
    `KVCacheManager.get_computed_blocks` calls `try_reserve_store_serve`
    before the C clamp — on success it skips the clamp (full KV reuse), on
    any miss (all-or-nothing) it falls through to C, so a captured position
    is never served from KV without its residual available. Admission
    threads the union `capture_store_hook_layers` / `capture_store_positions`.

    The spec-split (resolution 1) is **implicit**: serving advances
    `num_computed_tokens` over the prompt, so the worker validates the spec
    with `num_computed=0` (it pops the pending serve at registration) to
    avoid the validator's `< num_computed` rejection, and `build_step_plan`
    then naturally forward-captures only the step-window (generated) tail —
    no manual spec surgery. The served prompt rows are injected by
    `CaptureManager.serve_from_store`, which assembles per-consumer chunks
    from the payload and submits them like forward-path chunks (sinks lock
    `submit_chunk`; a request's serve precedes its own forward dispatch).
    Because admission also resolves with `num_computed=0`, the worker's
    resolution matches the threaded union exactly, so the serve payload is
    always complete.
  - **A.4 config + invalidation (done):** `--capture-activation-cache-gb`
    (a `CaptureConsumersConfig.activation_cache_bytes` budget, runtime-only
    so excluded from `compute_hash`); the runner instantiates the
    `ActivationStore` and calls `set_active_activation_store` when capture
    is on and the budget > 0; and `KVCacheManager.reset_prefix_cache`
    calls `invalidate_all` (the single chokepoint every reset / RLHF
    weight-update path flows through). With the budget at its default `0`
    the store is never installed, so A.2/A.3 are inert — the flag is the
    on-switch for the whole reuse layer.

## Known Limitations

These are behaviors the current implementation exhibits that may be
worth tightening:

- **`CaptureManager.finalize_request` returns a single
  representative result per consumer**, not an aggregated result
  across all `(layer, hook)` keys for that request. The runner
  surfaces one payload per consumer; multi-key aggregation is a
  TODO.
- **Runner does not populate `tag_slug` / `request_id_slug`
  metadata** on `CaptureChunk`. The filesystem consumer falls back
  to `"default"` and `str(vllm_internal_request_id)`, so on-disk
  files currently live at
  `{root}/default/{vllm_internal_request_id}/{layer}_{hook}.bin`.
  Wiring the admission-time slugs through the runner is tracked
  work.
- **`CaptureChunk.row_offset` is always `0`** today — the dispatch
  path does not cumulate offsets across steps. Order is still
  correct because the writer's partition-by-request-id invariant
  carries multi-step appends; `_BatchedAdapter`'s sort is a no-op.
- **Sidecar schema is minimal.** The framework propagates what the
  manager puts in (`consumer_index`) plus whatever the consumer
  inserts into `finalize.sidecar`. Optional fields like
  `client_request_id`, `tag`, `prompt_token_ids`,
  `generated_token_ids`, `model_name`, `created_at`,
  `finalized_at`, `finish_reason` are not yet populated by the
  runner — consumers that want them will need the runner to plumb
  them through.
- **Speculative decoding over-captures generated positions.** The
  capture path is spec-decode-unaware and purely position-based, and
  rejection sampling runs *after* the target forward. So in a verify
  step the taps gather hidden states for *all* candidate positions —
  including draft tokens that are rejected and re-forwarded in a later
  step. A generated position therefore appears in multiple rows (the
  stale rejected draft, then the accepted re-forward), so an
  `all_generated` / `all` capture has more rows than the request
  generated. The accepted token's hidden state is always present (the
  last row written for its position); to recover one row per token, the
  consumer records each row's absolute logical position in the sidecar
  (`positions`) and the reader exposes `latest_per_position(entry)`,
  which keeps the last row per position. Prompt selectors
  (`last_prompt` / `all_prompt`) are unaffected — prefill is not
  speculative. A fuller fix (emit rows only for accepted positions,
  which requires deferring dispatch until the accept mask is known) is
  not implemented. The behavior is hook-agnostic — it affects standard
  residual hooks and mHC hooks identically.
- **Shutdown sequencing.** Consumers are shut down when the runner
  tears down, but there is no explicit LIFO ordering or per-consumer
  budget propagation — each consumer's `shutdown(timeout)` default
  is 30s.

## Testing

- `tests/v1/capture/` — unit tests for types, config, manager,
  plan, registry, driver bridge, `_BatchedAdapter`, per-consumer
  tests.
- `tests/v1/capture/test_runner_integration.py` +
  `test_multi_consumer_runner.py` — runner-level integration.
- `tests/v1/capture/test_driver_consumer_e2e.py` — worker→driver
  end-to-end.
- `tests/v1/capture/test_sampling_params.py` — structural
  validation on `SamplingParams.capture`.
- `tests/v1/capture/test_prefix_cache_gating.py` — the B/C
  `spec_touches_prompt` / `min_captured_prompt_position` classifiers,
  relaxed `SamplingParams` construction, and the
  `Request.get_skip_reading_prefix_cache` / `get_capture_prefix_cache_limit`
  accessors. `tests/entrypoints/openai/test_capture_protocol.py` covers
  `_admit_capture` setting `capture_touches_prompt` and
  `capture_min_prompt_position`.
- `tests/v1/core/test_prefix_caching.py::test_capture_clamps_prefix_cache_hit`
  — the C clamp end-to-end in `KVCacheManager.get_computed_blocks`.
- `tests/v1/capture/test_activation_store.py` — the A-layer
  `ActivationStore` core (LRU eviction, byte budget, oversize skip,
  replacement accounting, invalidation, global accessor), the
  `activation_key` composition, `CaptureManager` write-through
  (content keying, prompt-only, clone-off-buffer, no-op guards), and the
  A.3 read primitives (`extract_all` all-or-nothing, the pending-serve
  bridge, `try_reserve_store_serve`), and `serve_from_store` chunk
  injection.
- `tests/v1/core/test_prefix_caching.py::test_capture_serves_from_store_instead_of_reforwarding`
  — the A.3 read floor end-to-end (full reuse on a resident prefix;
  all-or-nothing fallback to the C clamp on a miss).
- `tests/v1/capture/test_admission.py` — the shared
  `resolve_capture_prefix_flags` / `build_capture_context`
  (`vllm/v1/capture/admission.py`): generated-only vs prompt-touching
  classification, the cross-consumer floor, the store serve-set, and the
  unknown-consumer / invalid-spec error contract (`capture_param`).
- `tests/v1/engine/test_input_processor_capture.py` — the offline
  `InputProcessor` admission path: spec-form and instance-form flag
  stamping, and the conservative skip on an invalid spec (no rejection).
- `tests/v1/capture/test_registry.py::TestBuildAdmissionValidators` —
  `build_admission_validators` keys validators exactly as the worker's
  `name_to_index`: dict/spec form by `instance_name or name`, instances by
  class name with `#2` collision suffixing.
- `tests/engine/test_arg_utils.py::TestCaptureConsumersFlag` —
  CLI-flag parsing.
