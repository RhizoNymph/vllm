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

HookName = Literal["pre_attn", "post_attn", "post_mlp", "mlp_in", "mlp_out"]
PositionSelector = (
    Literal["last_prompt", "all_prompt", "all_generated", "all"]
    | list[int]
)

@dataclass(frozen=True)
class CaptureSpec:
    hooks: dict[HookName, list[int]]
    positions: PositionSelector

@dataclass
class CaptureChunk:
    key: CaptureKey
    tensor: torch.Tensor        # CPU, shape (num_rows, hidden_size)
    dtype: torch.dtype
    row_offset: int
    step_index: int
    metadata: dict[str, Any]

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
```

`HookName` must stay in lockstep with `_HOOK_NAME_TO_ID` in
`vllm/model_executor/layers/activation_capture.py`.

## Sinks and Consumers

### `CaptureSink` (protocol)

The low-level streaming interface:

```python
class CaptureSink(Protocol):
    location: ClassVar[Literal["worker", "driver"]]

    def submit_chunk(self, chunk: CaptureChunk) -> None: ...
    def submit_finalize(self, finalize: CaptureFinalize) -> None: ...
    def get_result(self, key: CaptureKey) -> CaptureResult | None: ...
    def shutdown(self, timeout: float = 30.0) -> None: ...
```

Ordering guarantees:

- For a given key, chunks arrive in `row_offset` order.
- `CaptureFinalize` for a key arrives after all chunks for that key.
- Different keys have no ordering relationship.
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
  instances (passed to the worker via
  `VllmConfig._capture_consumer_instances`).

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
3. Allocate `gather_indices[(layer, hook)]` (device `int64`) and
   `scratch_gpu[(layer, hook)]` (device, model residual dtype) sized
   for the union.

Gather happens **once** per `(layer, hook)` regardless of how many
consumers want it. Fan-out happens at dispatch time via the
`consumer_mask`.

`StepCapturePlan` holds:

- `gather_indices: dict[(layer, hook), Tensor]`
- `scratch_gpu: dict[(layer, hook), Tensor]`
- `scratch_dtype: dict[(layer, hook), torch.dtype]`
- `entries: list[CapturePositionEntry]` — one per captured row, with
  `(request_id, layer, hook, logical_pos, scratch_row, step_index,
  consumer_mask)`.
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
  hook_name, hidden_states)`, which `index_select`s into
  `plan.scratch_gpu[(layer, hook)]`. `hidden_states` is never mutated.

After the forward step, the runner calls `consume_step_plan()` to
take ownership of the plan and `dispatch_step_captures(plan)`:

1. For each consumer index `i`, group entries where bit `i` is set
   in `consumer_mask` by `(request_id, layer, hook)`.
2. For each group, `index_select` the consumer's rows out of the
   scratch tensor, `.cpu()` them, build a `CaptureChunk`, and call
   `sink.submit_chunk`. `metadata` carries `consumer_index` and the
   logical positions.
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
   consumer_instances=vllm_config._capture_consumer_instances)`.
2. For each validator, call `global_capture_spec()` defensively (any
   exception becomes `None`).
3. Instantiate `CaptureManager(sinks, global_specs, num_hidden_layers,
   hidden_size, model_dtype, device)`.
4. Call `set_active_capture_manager(manager)` so
   `maybe_capture_residual` finds it from inside the compiled graph.

Per-step hooks:

- `_prepare_capture_step(scheduler_output)` — builds the batch view
  (matching `SteeringModelRunnerMixin._update_steering_buffers`
  offset walk exactly so gather indices line up) and calls
  `manager.build_step_plan`.
- `_finalize_capture_step()` — calls `manager.consume_step_plan()`
  and `manager.dispatch_step_captures(plan)`.

Per-request finalize: when a request completes, the runner calls
`_finalize_capture_for_request(req_id)` which invokes
`manager.finalize_request(req_id)` and translates indices back to
names, surfacing the dict on `ModelRunnerOutput.capture_results`.

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

Writer details (`writer.py`):

- One `queue.Queue` per thread, partitioned by `hash(request_id) %
  num_threads` — preserves per-key append order without cross-thread
  locks.
- Per-thread LRU fd cache (default 256 entries); eviction `fsync`s +
  closes the fd.
- Collision policy (`overwrite` / `error` / `suffix`) applied at
  finalize.
- Structured `WriteError` with errno, path, key; surfaces back on
  `WriteResult.error`.
- `get_result(key)` maps `WriteResult` → `CaptureResult`: status
  pass-through; payload is `[bin_path, sidecar_path]` on success,
  `None` on error.

**Validation constraints** (`validation.py`):

- `tensor_parallel_size == 1 && pipeline_parallel_size == 1`.
- Every hook name is in `{pre_attn, post_attn, post_mlp, mlp_in,
  mlp_out}`.
- Every resolved layer is in `[0, num_hidden_layers)`.
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
8. **TP > 1 / PP > 1 rejected with a clear error.** The filesystem
   validator checks `CaptureContext.tensor_parallel_size` and
   `pipeline_parallel_size` before any other work. Other
   residual-collecting consumers should do the same.
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
  wholesale. The offline `LLM` path does not resolve specs at admission,
  so it leaves the flag `None` and stays conservatively disabled
  (correct, not optimized).

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
  no reuse; `last_prompt` clamps to almost the whole prompt. This is still
  API-path for its *data*: the floor is computed at admission, so the
  offline `LLM` path (no resolved specs at admission) keeps B's
  conservative skip. Offline-path clamping would require resolving specs
  in the offline path — a later extension.

- **A — activation store (store core implemented; wiring in progress).**
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
  Activation steering is not yet implemented in this tree, so the
  steering-off premise holds unconditionally for now; when steering lands,
  the read/serve path must gate on steering being off (taps are
  pre-steering, but upstream steering poisons the residual).

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
  - **A.3 read floor + serve-inject (primitives done; serve wiring
    pending a decision):** raise C's floor past store-resident positions
    (so they are not re-forwarded) and have the worker fetch the stored
    rows and dispatch them to the consumer as if captured. The race-safe
    store primitives are landed: `ActivationStore.extract_all` (an atomic
    all-or-nothing snapshot under the store lock — taking the rows at
    decision time is what frees the read path from a check-then-serve
    eviction race against dispatch-thread write-through/eviction), the
    same-process serve bridge (`stash_pending_serve` / `pop_pending_serve`,
    sound because capture is TP1/PP1), and `try_reserve_store_serve` (the
    scheduler-side reserve: compose keys, extract, stash for the worker).

    Wiring these into `get_computed_blocks` (raise the floor) and the
    worker (serve-inject) is blocked on one design decision: serving a
    prompt position means `num_computed_tokens` advances over it, but the
    capture validator *rejects* captured positions `< num_computed_tokens`
    (the "captured ⇒ forwarded" invariant, `validation.py`). The two clean
    resolutions are (1) **spec-split** — register only the
    generated/uncached positions for forward-capture and serve the prompt
    positions from the store separately (pure `all_prompt` registers
    nothing, serves everything); or (2) **serve-aware validator** — thread
    the served-position set into `CaptureContext` so the validator skips
    its rejection for them. (1) is preferred — it keeps the validator
    invariant intact and localizes the change to registration.
  - **A.4 config + invalidation:** a `--capture-activation-cache-gb` budget
    flag, store instantiation, and the `invalidate_all` call on weight
    update.

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
- **Shutdown sequencing.** Consumers are shut down when the runner
  tears down, but there is no explicit LIFO ordering or per-consumer
  budget propagation — each consumer's `shutdown(timeout)` default
  is 30s.
- **`LLM(capture_consumers=[instance])` is not fully wired.** The
  `LLM` constructor stores instances on
  `self._capture_consumer_instances` but does not attach them to
  `VllmConfig`. The runner reads
  `vllm_config._capture_consumer_instances` (see
  `GPUModelRunner.__init__`), so end-to-end instance handoff
  requires closing the gap between the `LLM` field and the
  `VllmConfig` attribute. Dict-form entries flow through the config
  and work today.

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
  bridge, `try_reserve_store_serve`).
- `tests/engine/test_arg_utils.py::TestCaptureConsumersFlag` —
  CLI-flag parsing.
