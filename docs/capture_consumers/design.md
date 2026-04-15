# Capture Consumers

## Overview

The **capture consumer framework** is a pluggable system for observing
and routing hidden-state activations produced inside vLLM's compiled
forward pass. It generalizes the existing per-request activation
storing feature into a shared plumbing layer that many different
consumers — disk writers, in-process training loops, dynamic steering
probes, live dashboards, SAE trainers, observability hooks — can hang
off of.

The feature has three responsibilities:

1. **Produce** captured activations at well-defined hook points inside
   each decoder layer, respecting per-consumer and per-request
   configuration, without breaking `torch.compile` or CUDA graphs.
2. **Route** captured activations to the right consumer(s) at request
   finalization, preserving batch order and multi-step append
   semantics.
3. **Extend** through an entry-point-based plugin registry so that
   third parties can add new consumer types without modifying vLLM
   core.

This spec defines the protocol, the config surface, the plugin
contract, and the runtime mechanics. The **filesystem consumer** (what
used to be the `activation_storing` feature's `ActivationWriter`) is
shipped as the first built-in consumer, registered via exactly the
same entry-point mechanism a third-party plugin would use, to
guarantee the framework itself is not special-cased for the one
consumer that ships with it.

### What this replaces

Everything previously under the `activation_storing` name. The
following are removed, renamed, or repurposed:

| Old (removed) | New (replacement) |
|---|---|
| `--activation-storing /path` CLI flag | `--capture-consumers filesystem:root=/path` (or YAML `capture_consumers:`) |
| `ActivationStoringConfig` | No top-level config; filesystem consumer reads its own params dict |
| `SamplingParams.activation_storing: ActivationStoringSpec \| None` | `SamplingParams.capture: dict[str, Any] \| None` (dict keyed by consumer name) |
| `RequestOutput.activation_storage: CaptureResult \| None` | `RequestOutput.capture_results: dict[str, CaptureResult]` (dict keyed by consumer name) |
| `vllm/config/activation_storing.py` | Removed |
| `vllm/config/activation_storing_types.py` | Moved to `vllm/v1/capture/types.py`, pruned to framework types |
| `vllm/entrypoints/openai/activation_storing_validation.py` | Moved to `vllm/v1/capture/consumers/filesystem/validation.py` — it's filesystem-consumer-specific validation |
| `vllm/model_executor/layers/activation_capture.py` | Split: manager/sink/types move to `vllm/v1/capture/`, the custom op stays at the old path (it's model-facing, not framework-facing) |
| `vllm/v1/worker/activation_writer.py` | Moved to `vllm/v1/capture/consumers/filesystem/writer.py` as a private implementation detail of the filesystem consumer |

**No backward compatibility.** The existing activation storing API is
removed wholesale. Users migrating from the old API will need to
change their CLI flags, their `SamplingParams` usage, and their
`RequestOutput` handling. The migration is mechanical — see
§ "Migration from activation storing".

### What it does NOT replace

- The **capture custom op** (`torch.ops.vllm.capture_residual`) stays
  exactly as it is today at
  `vllm/model_executor/layers/activation_capture.py`, along with
  `_HOOK_NAME_TO_ID`, `maybe_capture_residual`, and
  `set_active_capture_manager`. These are hook-point-level machinery
  used by model files and by the capture manager; they do not care
  about consumers.
- The **steering feature** (`apply_layer_steering`, `apply_steering`,
  the `SteeringHookPoint` enum, the 65 decoder-layer model-file call
  sites). Capture consumers tap the same hook points via
  `maybe_capture_residual` but do not alter steering. Steering at
  MLP-internal hook points (`mlp_in`, `mlp_out`) continues to work
  exactly as before.

## Non-goals

- **Same-forward-pass feedback.** A consumer wanting to observe layer
  12's activations and inject a steering vector at layer 24 *in the
  same forward pass* is out of scope for v1. This would require a
  different integration point (synchronous hook inside the compiled
  graph, operating on GPU tensors, with bounded latency between
  layers) and a different consumer interface (GPU tensors, not CPU
  tensors; sub-millisecond latency budget). The framework leaves room
  for it — the capture manager's hook fires can be extended with a
  sync-tap variant without rewriting the consumer protocol — but the
  consumer types defined here run on CPU tensors at request
  finalization, not mid-forward. Cross-request feedback and
  prefill→decode feedback are supported via the driver-side consumer
  path (see § "Location model").

- **Multi-tenant isolation / access control.** Consumers observe
  activations across every request that matches their declared spec.
  Operators who enable a consumer accept that it sees all traffic.
  Session-level scoping (preventing user A's capture from leaking to
  user B's feedback) is the consumer's responsibility, using
  whatever identity information the operator chooses to expose via
  optional sidecar fields. The framework exposes
  `vllm_internal_request_id` by default and nothing else; consumers
  that need external identity declare optional fields explicitly.

- **Compressed / columnar formats.** Captured tensors are raw
  row-major bytes at whatever dtype the manager's scratch buffer
  used. Consumers that want Parquet / Zarr / safetensors can
  implement their own serialization; the filesystem consumer writes
  raw bytes + sidecar JSON, the same format the old activation
  storing feature used.

- **Managed catalog / query engine.** No built-in catalog, index, or
  query language over captured activations. The filesystem consumer's
  directory layout is the "catalog" for that consumer; other
  consumers manage their own addressing.

## Architecture overview

Three layers, clean separation:

```
  ┌─────────────────────────────────────────────────────────────┐
  │                  Decoder-layer forward code                  │
  │  (65 model files, unchanged — they call apply_layer_steering │
  │   which already calls maybe_capture_residual)                │
  └─────────────────────────────────────────────────────────────┘
                                │
                                │ torch.ops.vllm.capture_residual
                                │ (compile-graph-opaque custom op)
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                     Capture Manager                          │
  │  (vllm/v1/capture/manager.py)                                │
  │                                                              │
  │  - Holds per-consumer global specs                           │
  │  - Holds per-request client specs                            │
  │  - Builds per-step plans (gather_indices, scratch_gpu)       │
  │  - Dispatches captures to the right consumer(s) on finalize  │
  │  - Owns the pinned-CPU scratch + D2H copy + concat           │
  └─────────────────────────────────────────────────────────────┘
                                │
                                │ CaptureSink protocol
                                │ (submit_chunk, submit_finalize, ...)
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                     Capture Sinks                            │
  │  (framework — vllm/v1/capture/sink.py)                       │
  │                                                              │
  │  Low-level streaming interface. Most users never see this.   │
  │  CaptureConsumer implements it via batched accumulation.     │
  │                                                              │
  │  Two flavors:                                                │
  │  - Worker-side sinks: run in the engine-core subprocess      │
  │  - Driver-side shim sinks: push to a queue, a driver-side    │
  │    receiver thread unpacks and calls the user's consumer     │
  └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                     Capture Consumers                        │
  │  (vllm/v1/capture/consumer.py — user-facing base class)      │
  │                                                              │
  │  Subclasses override on_capture(key, tensor, sidecar).       │
  │  - location = "worker" or "driver" (class attribute)         │
  │  - required_sidecar_fields declares needed optional fields   │
  │  - global_capture_spec() declares always-on spec             │
  │  - reads_client_spec = True if the consumer accepts          │
  │    per-request client opt-in via SamplingParams.capture      │
  └─────────────────────────────────────────────────────────────┘

  ┌──────────────────────────┐    ┌──────────────────────────────┐
  │  Built-in consumers      │    │  Third-party consumers       │
  │  (vllm/v1/capture/       │    │  (ship as separate packages, │
  │   consumers/)            │    │   register via entry points) │
  │                          │    │                              │
  │  - filesystem            │    │  - reward_trainer            │
  │  - logging               │    │  - dynamic_steering          │
  │                          │    │  - live_dashboard            │
  │                          │    │  - sae_trainer               │
  └──────────────────────────┘    └──────────────────────────────┘
```

The capture manager is the single producer. Sinks are the low-level
consumer-side interface. Consumers are the ergonomic user-facing base
class that 90% of plugin authors will subclass. Built-in consumers
ship with vLLM but are registered via the same entry-point mechanism
third parties use, so the plugin path is not special-cased.

### Data flow — one request through the system

1. **Admission.** Client submits a request. If the request has
   `SamplingParams.capture`, the manager routes each
   `(consumer_name, client_spec)` entry to the consumer's validator
   (if `reads_client_spec = True`). Invalid specs become HTTP 400.

2. **Registration.** Once admitted, the manager registers the
   request against the union of:
   - Every active consumer's `global_capture_spec()`.
   - The request's per-consumer `SamplingParams.capture` entries.
   Each registered spec is tagged with the consumer that owns it.

3. **Per-step plan.** Before each forward step, the manager walks
   the `InputBatch` in order and, for each active request, computes
   the gather indices for every `(layer, hook)` that at least one
   consumer cares about. The step plan is a dict keyed by
   `(layer, hook)` pointing at a GPU int64 tensor of batch positions
   plus a GPU scratch tensor for the captured rows.

4. **Forward.** Each model's `apply_layer_steering` call fires
   `maybe_capture_residual` which dispatches the custom op. The op
   reads the step plan, index_selects into the scratch buffer, and
   returns `hidden_states` unchanged. Zero compiled-graph impact.

5. **Finalize step.** After sampling, the runner's
   `_finalize_capture_step` does a pinned-CPU D2H copy of every
   populated scratch tensor, syncs once, and walks the plan's entry
   list. For each captured row, it determines which consumers care
   about that `(request, layer, hook)` and dispatches a
   `CaptureChunk` to each. Dispatch goes through the consumer's
   `CaptureSink` (which for user-facing `CaptureConsumer` subclasses
   is a batched adapter that accumulates chunks in CPU memory).

6. **Finalize request.** When a request finishes (any finish reason),
   the manager walks the consumers that had registered specs against
   this request and calls `submit_finalize(...)` on each. For a
   user-facing `CaptureConsumer`, this triggers the batched adapter
   to concatenate buffered chunks and invoke `on_capture(key, tensor,
   sidecar)`. For a streaming-aware sink (like the filesystem
   consumer), this triggers an atomic rename + sidecar write.

7. **Response.** The manager polls every involved consumer's
   `get_result(key)` and populates
   `RequestOutput.capture_results[consumer_name]` with a
   `CaptureResult`. The client sees per-consumer status and
   consumer-specific payload (filesystem gets file paths; reward
   trainer gets None; dashboard gets a URL; etc.).

## Core types

All types live under `vllm/v1/capture/`. This module is **torch-aware**
(unlike the old `activation_storing_types.py` which was torch-free); it
has to be, because `CaptureChunk` carries a `torch.Tensor`. Unit tests
for the types can still run without torch if they only exercise the
protocol-level logic (e.g., the registry, the spec parser).

### `CaptureKey`

```python
from typing import NewType

# The unique identifier vLLM assigns internally to a request. Always
# available; never client-controlled; opaque string. Consumers that
# want to correlate with external identity should declare the
# appropriate optional sidecar field (e.g., client_request_id, tag).
VllmInternalRequestId = NewType("VllmInternalRequestId", str)

CaptureKey = tuple[VllmInternalRequestId, int, str]
# (vllm_internal_request_id, layer_idx, hook_name)
```

### `CaptureSpec`

```python
from typing import Literal
from dataclasses import dataclass

HookName = Literal["pre_attn", "post_attn", "post_mlp", "mlp_in", "mlp_out"]

PositionSelector = (
    Literal["last_prompt", "all_prompt", "all_generated", "all"]
    | list[int]
)

@dataclass(frozen=True)
class CaptureSpec:
    """Describes which activations to capture for a request.

    - ``hooks`` maps each hook point to the layer indices at which
      the hook fires. Empty list ⇒ hook disabled.
    - ``positions`` selects which token positions are captured at
      every hook/layer.

    ``CaptureSpec`` is the in-framework representation. It is
    produced by:
    - Consumer's ``global_capture_spec()`` (applies to every request)
    - Consumer's per-request validator, parsing ``SamplingParams.capture``

    It is NOT directly serializable across the process boundary;
    consumers that ship CaptureSpecs via IPC should go through the
    consumer's own validator.
    """

    hooks: dict[HookName, list[int]]
    positions: PositionSelector
```

### `CaptureChunk`

```python
@dataclass
class CaptureChunk:
    """One batch of captured rows for a ``CaptureKey``.

    Emitted by the manager after every forward step that produces rows
    for this key. For a single key, chunks arrive in ``row_offset``
    order. Different keys have no ordering relationship.
    """

    key: CaptureKey
    tensor: torch.Tensor       # CPU tensor, shape (num_rows, hidden_size)
    dtype: torch.dtype         # Explicit; avoids tensor.dtype dispatch in consumers
    row_offset: int            # Cumulative row index within this key's sequence
    step_index: int            # Which forward step produced this chunk
    metadata: dict[str, Any]   # Per-chunk context (see § "Sidecar schema")
```

### `CaptureFinalize`

```python
@dataclass
class CaptureFinalize:
    """Request-completion signal for a ``CaptureKey``.

    Emitted by the manager when the owning request finishes (any
    finish reason). Arrives after all ``CaptureChunk``s for the key.
    On receipt the sink should flush any buffered state for this key
    and produce a terminal ``CaptureResult`` accessible via
    ``get_result(key)``.
    """

    key: CaptureKey
    sidecar: dict[str, Any]    # See § "Sidecar schema" for the full schema
```

### `CaptureResult`

```python
CaptureStatus = Literal[
    "pending",
    "ok",
    "partial_error",
    "error",
    "not_requested",
]

@dataclass
class CaptureResult:
    """Terminal per-key result from a consumer.

    Attached to ``RequestOutput.capture_results[consumer_name]`` on
    request completion. The ``payload`` field is consumer-specific
    and opaque to the framework — filesystem returns ``list[Path]``,
    a dashboard returns ``dict[str, str]``, a silent consumer returns
    ``None``.
    """

    key: CaptureKey
    status: CaptureStatus
    error: str | None = None
    payload: Any = None
```

### `CaptureSink` protocol

```python
from typing import Protocol, ClassVar

class CaptureSink(Protocol):
    """Low-level streaming interface between the capture manager and a
    consumer. Most consumer authors should subclass ``CaptureConsumer``
    (which implements ``CaptureSink`` via a batched adapter) instead of
    implementing this protocol directly.

    Implement this protocol directly when you need:
    - True streaming semantics (write incrementally, don't buffer
      the full tensor — this is what the filesystem consumer does).
    - Chunk-level visibility into `row_offset` / `step_index` beyond
      what the batched consumer exposes.
    - Custom threading and concurrency models.

    Ordering guarantees from the manager side:
    - For a given ``CaptureKey``, chunks arrive in ``row_offset`` order.
    - ``CaptureFinalize`` for a key arrives after all chunks for that key.
    - Different keys have no ordering relationship.

    All methods must be thread-safe: the manager may call submit_* from
    any worker thread, and the output processor may call get_result
    from the engine-core thread.
    """

    location: ClassVar[Literal["worker", "driver"]]

    def submit_chunk(self, chunk: CaptureChunk) -> None:
        """Non-blocking enqueue of a chunk of captured rows."""

    def submit_finalize(self, finalize: CaptureFinalize) -> None:
        """Non-blocking request-completion signal. Flushes buffered state."""

    def get_result(self, key: CaptureKey) -> CaptureResult | None:
        """Return terminal result for a key, or None if still pending."""

    def shutdown(self, timeout: float = 30.0) -> None:
        """Drain in-flight work with a bounded grace period."""
```

### `CaptureConsumer` base class

```python
from abc import ABC, abstractmethod

class CaptureConsumer(ABC):
    """User-facing base class for capture consumers.

    The default implementation accumulates ``CaptureChunk``s per-key in
    CPU memory until ``CaptureFinalize`` arrives, then invokes
    ``on_capture(key, tensor, sidecar)`` with the concatenated tensor.
    Subclasses override ``on_capture`` and (usually) nothing else.

    Subclasses set class-level metadata:

    - ``location``: where the consumer runs. ``"worker"`` (default)
      runs in the engine-core subprocess alongside the model runner,
      with direct in-process access to the capture manager and no
      IPC overhead. ``"driver"`` runs in the main Python process where
      the ``LLM`` lives, with full access to driver-process state
      (user models, optimizers, data loaders, the steering registry's
      driver-side API). vLLM transparently handles worker→driver
      IPC via ``torch.multiprocessing.Queue`` with shared-memory
      tensor handoff.

    - ``required_sidecar_fields``: the set of optional sidecar field
      names the consumer needs the framework to populate.
      ``vllm_internal_request_id`` is always present and never needs
      to be declared. Any other field (``client_request_id``,
      ``tag``, ``prompt_token_ids``, ``generated_token_ids``,
      ``model_name``, etc.) must be listed here or it may not be in
      the sidecar dict passed to ``on_capture``. Declaring fields a
      consumer doesn't use costs memory and latency (the manager has
      to keep the data); declaring too few means the consumer can't
      do its job. Default: empty set (consumer sees only
      ``vllm_internal_request_id``).

    - ``reads_client_spec``: whether the consumer accepts per-request
      client opt-in via ``SamplingParams.capture[consumer_name]``.
      Most consumers don't — they have a global spec set at
      registration time. The filesystem consumer does, so clients
      can trigger per-request captures with specific tags and
      positions. Default: ``False``.

    Override points (in order of necessity):

    - ``__init__(self, vllm_config, params)`` — called once at engine
      startup with the operator-provided params dict. Does whatever
      setup the consumer needs (load a probe model, open a log file,
      spawn a background thread, whatever).

    - ``global_capture_spec(self) -> CaptureSpec | None`` — returns
      the consumer's global capture spec, applied to every request.
      Default: ``None`` (no global spec; consumer only observes
      per-request client opt-ins). Most plugin consumers return a
      concrete spec here (e.g., "observe post_mlp at layers 12 and 24
      with positions='last_prompt'"). The filesystem consumer
      returns ``None``.

    - ``validate_client_spec(self, raw_spec: Any, ctx: CaptureContext) -> CaptureSpec``
      — if ``reads_client_spec = True``, called at admission time with
      the raw ``SamplingParams.capture[consumer_name]`` value. Must
      validate and convert it to a ``CaptureSpec``, or raise
      ``CaptureValidationError`` (surfaces as HTTP 400). Default:
      raise ``NotImplementedError``.

    - ``on_capture(self, key, tensor, sidecar)`` — called once per
      finalized ``CaptureKey`` with the fully assembled CPU tensor.
      This is the main override. ``tensor`` is shape ``(num_rows,
      hidden_size)`` in ``dtype`` as captured. ``sidecar`` is the
      dict described in § "Sidecar schema", filtered to the
      consumer's ``required_sidecar_fields`` plus
      ``vllm_internal_request_id``.

    - ``on_error(self, key, error)`` — called on capture failure for
      this key. Default: pass.

    - ``shutdown(self, timeout)`` — called on engine teardown.
      Default: pass.
    """

    location: ClassVar[Literal["worker", "driver"]] = "worker"
    required_sidecar_fields: ClassVar[frozenset[str]] = frozenset()
    reads_client_spec: ClassVar[bool] = False

    def __init__(self, vllm_config: "VllmConfig", params: dict[str, Any]) -> None:
        pass

    def global_capture_spec(self) -> "CaptureSpec | None":
        return None

    def validate_client_spec(
        self,
        raw_spec: Any,
        ctx: "CaptureContext",
    ) -> "CaptureSpec":
        raise NotImplementedError(
            f"{type(self).__name__} has reads_client_spec=True but "
            f"did not override validate_client_spec()."
        )

    @abstractmethod
    def on_capture(
        self,
        key: CaptureKey,
        tensor: "torch.Tensor",
        sidecar: dict[str, Any],
    ) -> None:
        """Called once per finalized capture key."""

    def on_error(self, key: CaptureKey, error: str) -> None:
        pass

    def shutdown(self, timeout: float = 30.0) -> None:
        pass
```

### `CaptureContext`

```python
@dataclass
class CaptureContext:
    """Per-request context passed to ``validate_client_spec``.

    Contains everything a validator needs to check a client spec
    against the request's actual shape. Fields are deliberately
    narrow — validators should not poke at ``vllm_config`` beyond
    these. If a validator needs more, add it here explicitly.
    """

    vllm_internal_request_id: VllmInternalRequestId
    num_prompt_tokens: int
    num_computed_tokens: int  # Prefix cache hits, for rejecting cached positions
    num_hidden_layers: int
    hidden_size: int
    element_size_bytes: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
```

## Location model

Capture consumers run in one of two process locations, declared via
the class-level `location` attribute:

### `location = "worker"`

The consumer instance lives in the engine-core subprocess — the same
process as the `GPUModelRunner` and the capture manager. The
manager calls the consumer's sink methods directly, no IPC.

**Use for:**
- High-throughput streaming consumers (filesystem, local queue,
  shared-memory ring buffer).
- Consumers that need direct access to worker-local state (the
  steering registry's worker-side buffers, the runner's scratch
  tensors, any shared memory the worker holds).
- Consumers with sub-millisecond per-chunk latency requirements.

**Constraints:**
- Must be constructable from `(vllm_config, params)`. The class
  reference + params dict ship from the driver to the worker at
  startup; the worker imports the class and constructs the instance
  locally. You cannot pass a pre-constructed instance with live
  driver-process state to a worker consumer.
- Everything the consumer holds must be pickleable OR constructable
  from the params dict alone. A filesystem path is fine; a reference
  to a live PyTorch model in the driver process is not.
- Runs on whatever thread the sink dispatches on. If the consumer
  spawns background threads, it must shut them down cleanly.

### `location = "driver"`

The consumer instance lives in the main Python process where the
`LLM` is instantiated. vLLM transparently handles the worker→driver
plumbing:

1. At engine startup, the driver process instantiates the consumer:
   `consumer = MyDriverConsumer(vllm_config, params)`.
2. The driver creates a `torch.multiprocessing.Queue` dedicated to
   this consumer (one queue per driver consumer, for isolation).
3. A **worker-side shim** — an internal `_DriverQueueSink`
   implementing `CaptureSink` with `location = "worker"` — is
   installed on the capture manager. It serializes every incoming
   chunk/finalize and pushes it to the queue. Tensor payloads cross
   the boundary via shared memory (no pickle serialization for the
   bytes).
4. A **driver-side receiver** — an internal `_DriverReceiver` — runs
   as a background thread in the driver process. It pops events from
   the queue and invokes the user's consumer's `on_capture` /
   `on_error` on the receiver thread.

The user sees exactly the same API as for worker consumers: subclass
`CaptureConsumer`, set `location = "driver"`, implement `on_capture`.
The plumbing is invisible.

**Use for:**
- Training loops that hold the optimizer state in the driver process.
- Dynamic steering probes that update the driver-side steering
  registry based on captured activations.
- Consumers that call back into `llm.generate()` or other
  driver-process APIs.
- Interactive dashboards, logging to user-visible file paths,
  anything that wants to be in the user's main process.

**Constraints:**
- Latency: cross-process queue + shared-memory tensor handoff. The
  tensor bytes don't copy, but the queue round-trip is ~50-200μs per
  event. Fine for per-request feedback; too slow for per-chunk tight
  loops.
- Back-pressure: the queue is bounded. If the driver-side consumer
  falls behind, `submit_chunk`/`submit_finalize` on the worker
  side blocks up to a configurable timeout (default 30s) and then
  marks the affected requests `partial_error`. Same semantic as the
  existing filesystem consumer's slow-disk handling.
- Process boundary: the user's `on_capture` callback cannot
  directly modify worker-side state (scratch tensors, GPU buffers,
  the runner's step plan). It can call back into the driver process's
  steering registry, optimizer, model, etc.

### Process model summary

| Consumer location | Runs in | Can access | Cannot access |
|---|---|---|---|
| `worker` | Engine-core subprocess | Worker-local state (steering buffers, runner, scratch tensors); CPU tensors directly | Driver-process state; user's live Python objects |
| `driver` | Main Python (LLM) process | Driver-process state (optimizers, user models, data loaders, driver-side steering registry); `llm.generate()` | Worker-local state; mid-forward hooks |

## Spec model

Two kinds of capture specs exist, with different registration lifecycles:

### Global specs (consumer-declared)

Declared by the consumer at engine startup via
`global_capture_spec()`. Applies to every request that passes through
the engine. Used by consumers that observe traffic without requiring
client cooperation — dynamic steering probes, reward trainers, live
dashboards, most observation plugins.

Example:

```python
class RewardTrainer(CaptureConsumer):
    location = "driver"

    def __init__(self, vllm_config, params):
        self._trainer = build_trainer(params["checkpoint_path"])
        self._layers = params["layers"]

    def global_capture_spec(self):
        return CaptureSpec(
            hooks={"post_mlp": self._layers},
            positions="last_prompt",
        )

    def on_capture(self, key, tensor, sidecar):
        self._trainer.step(tensor)
```

Consumers with a non-`None` global spec receive captures from every
request automatically. Clients sending requests don't need to know
the consumer exists.

### Per-request client specs

Declared by clients via `SamplingParams.capture[consumer_name]`. Only
applies to the specific request. Used by consumers where the client
decides what gets captured — the filesystem consumer is the main
example: clients choose a tag, layers, and positions per-request.

Example:

```python
from vllm.v1.capture.consumers.filesystem import FilesystemCaptureRequest

sampling_params = SamplingParams(
    max_tokens=16,
    capture={
        "filesystem": FilesystemCaptureRequest(
            request_id="probe_0001",
            tag="mnist-probe-v1",
            hooks={"post_mlp": [12]},
            positions="last_prompt",
        ),
    },
)
```

Only consumers with `reads_client_spec = True` accept per-request
specs. At admission time, the manager walks every key in
`SamplingParams.capture`, looks up the consumer by name, calls the
consumer's `validate_client_spec(raw_value, ctx)` to get a
`CaptureSpec`, and registers it for the request. Invalid specs raise
`CaptureValidationError` which surfaces as HTTP 400.

A single request can hit multiple consumers — both global specs from
all active consumers AND per-request specs from any consumers the
client explicitly named. The manager dedupes the union when building
the per-step plan.

### Plan building — union and dispatch

At each forward step, the manager:

1. Walks the `InputBatch` in order, matching each request to its
   registered consumer specs (global + per-request).
2. For each `(layer, hook)` that at least one consumer wants,
   allocates a GPU scratch tensor and builds the gather_indices for
   the requesting rows.
3. During the forward, `capture_residual` populates the scratch
   tensors.
4. In `_finalize_capture_step`, the manager:
   - Does a single pinned-CPU D2H copy of all scratch tensors.
   - Synchronizes once.
   - For each captured row, looks up which consumers asked for it
     (could be one, could be several) and dispatches a
     `CaptureChunk` to each.

Dispatching the same row to multiple consumers is the common case
when multiple consumers have overlapping specs. The manager does not
duplicate the capture — it reads once, fans out on dispatch.

### Finalize routing

When a request finishes, the manager walks its registered consumers,
for each one calls `submit_finalize(...)` with the per-consumer
`CaptureFinalize` containing the consumer-requested sidecar fields
(plus the always-present `vllm_internal_request_id`). Each consumer's
sink produces a `CaptureResult` accessible via `get_result(key)`.

Eventually the manager aggregates:

```python
request_output.capture_results = {
    consumer_name: consumer_sink.get_result((req_id, layer, hook))
    for (consumer_name, consumer_sink) in active_consumers_for_request
    for (layer, hook) in consumer_specs[consumer_name]
}
```

Clients see per-consumer results on the final `RequestOutput`. A
filesystem-capture request returns paths; a reward-trainer request
returns nothing but contributes to the trainer's state.

## Sidecar schema

Sidecar fields are the metadata each consumer receives alongside the
captured tensor. The framework defines a minimal always-present set
and a larger optional set that consumers opt into.

### Always present (framework guarantees)

These fields are in every `CaptureChunk.metadata` and every
`CaptureFinalize.sidecar`, regardless of consumer configuration:

| Field | Type | Meaning |
|---|---|---|
| `vllm_internal_request_id` | `str` | Unique vLLM-internal request identifier. Stable within a process lifetime. Opaque; not client-controlled. |
| `layer` | `int` | Zero-based layer index this capture came from. |
| `hook` | `str` | One of the `HookName` literals. |
| `shape` | `list[int]` | Shape of the captured tensor. Usually `[num_rows, hidden_size]`. |
| `dtype` | `str` | Tensor dtype as a string (e.g., `"bfloat16"`). |
| `element_size` | `int` | Bytes per element. |
| `captured_at` | `str` | ISO 8601 UTC timestamp of finalize. |

These are enough for a minimal observation consumer to log
`"request X captured N rows at layer L hook H"` without knowing
anything about the client or the prompt.

### Optional (consumer opts in)

Consumers declare which optional fields they want via
`required_sidecar_fields`. The framework populates only the declared
fields — requesting a field you don't use costs memory (for token
lists) and time (for the manager to copy them into the dispatch dict).

| Field | Type | Meaning | Available when |
|---|---|---|---|
| `client_request_id` | `str` | The client-provided request identifier (slugged). | The client explicitly set one (e.g., via OpenAI `request_id` header). |
| `tag` | `str` | Client-provided grouping label (slugged). | The client set it via a per-request client spec. Empty string for global-only captures. |
| `prompt_token_ids` | `list[int]` | Full prompt token ids. | Always capturable but expensive to copy. |
| `generated_token_ids` | `list[int]` | Generated token ids (only finalized ones). | Always capturable. |
| `last_prompt_token_index` | `int` | The token index of the final prompt token. | Always capturable. |
| `positions` | `list[int]` | The resolved token indices captured in this tensor. | Always. |
| `position_kind` | `str` | One of `last_prompt`/`all_prompt`/`all_generated`/`all`/`explicit`. | When the spec was a symbolic position selector. |
| `model_name` | `str` | The model name as configured on the engine. | Always. |
| `model_dtype` | `str` | Model's residual dtype. | Always. |
| `created_at` | `str` | ISO 8601 UTC timestamp of request admission. | Always. |
| `finalized_at` | `str` | ISO 8601 UTC timestamp of finalize. | Always. |
| `finish_reason` | `str` | vLLM's finish reason for the request. | At finalize time. |

The schema is **intentionally freeform**: consumers iterate over the
dict, and new fields can be added in minor releases without breaking
existing consumers (they just ignore unknown keys). Losing mypy
coverage on the sidecar shape is an explicit tradeoff for plugin
extensibility.

**Privacy note.** A consumer that declares no optional sidecar fields
sees exactly `vllm_internal_request_id` plus the capture metadata
(layer, hook, shape, dtype). It cannot correlate captures with client
identity, prompts, or generations. This is the default, and it's the
right default for plugins that want to do their work without
touching user data. Consumers that need richer context declare the
specific fields they want.

## Registry and discovery

Capture consumers are discovered via the standard Python entry-point
mechanism. vLLM enumerates the `vllm.capture_consumers` group at
engine startup:

```toml
# Third-party plugin's pyproject.toml
[project.entry-points."vllm.capture_consumers"]
my_trainer = "my_plugin:RewardTrainer"
live_dashboard = "my_plugin:DashboardConsumer"
```

```toml
# vLLM's own pyproject.toml — built-in consumers register the same way
[project.entry-points."vllm.capture_consumers"]
filesystem = "vllm.v1.capture.consumers.filesystem:FilesystemConsumer"
logging = "vllm.v1.capture.consumers.logging:LoggingConsumer"
```

Each entry point is a **class**, not a factory function. The class
has the class-level metadata (`location`, `required_sidecar_fields`,
`reads_client_spec`) and an `__init__(self, vllm_config, params)`.

### Startup sequence

1. Driver process reads `capture_consumers` from config (YAML + CLI +
   Python LLM kwargs).
2. Driver enumerates `importlib.metadata.entry_points(group="vllm.capture_consumers")`.
3. For each configured consumer name, resolves the entry point to a
   class. Unknown names raise `UnknownCaptureConsumerError` at
   engine init time.
4. For each class, reads `cls.location`:
   - `location == "driver"`: driver instantiates the class directly,
     stores the instance, spawns a dedicated queue and receiver
     thread.
   - `location == "worker"`: driver ships `(class_qualname, params)`
     to the worker process (via the existing engine-core IPC).
     Worker imports the class and instantiates it there.
5. Engine-core subprocess installs every worker-side consumer instance
   (either native worker or driver-shim) on the capture manager.
6. Engine starts serving.

### Instance passing (Python LLM only)

For driver consumers that need live driver-process state at
construction time — a pre-loaded model, a running optimizer, a data
loader with open file handles — the Python API also accepts
pre-constructed instances:

```python
trainer = RewardTrainer.from_live(my_model, my_optimizer)

llm = LLM(
    model="meta-llama/Llama-3-8B",
    capture_consumers=[
        {"name": "filesystem", "params": {"root": "/tmp/captures"}},
        trainer,  # Already an instance; goes directly to the driver consumer list
    ],
)
```

The list is heterogeneous: dict entries go through the registry lookup
path; `CaptureConsumer` instances are installed directly. Instances
must have `location = "driver"` — you cannot pass a worker-side
consumer instance (the instance lives in the wrong process). Worker
consumers only support the registry path.

`vllm serve` does not have an instance-passing path. Operators who
need live state must either (a) use a worker-side consumer that
constructs its state from the params dict, or (b) use a driver-side
consumer whose `__init__` loads state from disk (checkpoint path,
config file, etc.).

## Config schema

One config key at the top level of vLLM's config tree:
`capture_consumers`. It's a list of `{name, params}` dicts.

### YAML

```yaml
model: meta-llama/Llama-3-8B
capture_consumers:
  - name: filesystem
    params:
      root: /mnt/nas/activations
      writer_threads: 4
      on_collision: overwrite
  - name: reward_trainer
    params:
      checkpoint_path: /ckpt/reward_v1.pt
      lr: 0.0001
      layers: [12, 24]
  - name: live_dashboard
    params:
      websocket_url: ws://localhost:9000/captures
```

### CLI

```bash
vllm serve meta-llama/Llama-3-8B \
    --capture-consumers filesystem:root=/mnt/nas/activations \
    --capture-consumers reward_trainer:checkpoint_path=/ckpt/reward_v1.pt,lr=0.0001
```

The CLI shorthand is `name:key1=value1,key2=value2,...`. Complex
params (nested dicts, lists) require YAML. For flat scalar configs,
the CLI is ergonomic; for anything richer, use `--config-file
my_config.yaml`.

### Python `LLM(...)`

```python
from vllm import LLM
from my_plugin import RewardTrainer

llm = LLM(
    model="meta-llama/Llama-3-8B",
    capture_consumers=[
        # Declarative — same form as YAML. Goes through the registry.
        {"name": "filesystem", "params": {"root": "/tmp/captures"}},

        # Instance — only valid for driver-side consumers. Goes
        # directly into the driver consumer list.
        RewardTrainer.from_live(model=my_live_model, optimizer=my_opt),
    ],
)
```

Notes:
- `capture_consumers` is the only config key for this feature. There
  is no top-level `activation_storing`, no
  `ActivationStoringConfig`, no `--activation-storing` flag.
- The order in the list doesn't matter for correctness — consumers
  don't depend on each other. It does affect shutdown order (LIFO,
  so consumers shut down in reverse registration order).
- Passing a dict entry is equivalent to the YAML form. Mixing dicts
  and instances is allowed.

### `SamplingParams.capture`

Per-request opt-in for consumers that accept client specs:

```python
sampling_params = SamplingParams(
    max_tokens=16,
    capture={
        "filesystem": FilesystemCaptureRequest(
            request_id="probe_0001",
            tag="mnist-probe-v1",
            hooks={"post_mlp": [12]},
            positions="last_prompt",
        ),
    },
)
```

`capture` is `dict[str, Any] | None`. Keys are consumer names (must
match a registered consumer with `reads_client_spec = True`). Values
are whatever the named consumer accepts — the consumer's
`validate_client_spec` method parses and validates them. Typically the
consumer ships a dataclass or TypedDict for this so client code has
type hints.

Requests that don't include `capture` are not rejected — they just
only receive captures from consumers with global specs.

### `RequestOutput.capture_results`

Per-consumer results on the final `RequestOutput`:

```python
result = next(llm.generate([prompt], sampling_params))
fs_result = result.capture_results.get("filesystem")
if fs_result and fs_result.status == "ok":
    for path in fs_result.payload:
        print("wrote", path)

trainer_result = result.capture_results.get("reward_trainer")
if trainer_result and trainer_result.status == "partial_error":
    logger.warning("reward trainer error: %s", trainer_result.error)
```

The dict is keyed by consumer name. Only consumers that were active
for this request appear. A request that triggered no captures (no
global specs, no per-request client specs) gets an empty dict.

### OpenAI-compatible HTTP request

```python
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3-8B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "extra_body": {
            "capture": {
                "filesystem": {
                    "request_id": "probe_train_0001",
                    "tag": "capital-probe",
                    "hooks": {"post_mlp": [12, 16, 20, 24]},
                    "positions": "last_prompt",
                },
            },
        },
    },
    timeout=60,
).json()

print(response["capture_results"]["filesystem"])
# {
#   "status": "ok",
#   "error": null,
#   "payload": [
#     "/mnt/nas/activations/meta-llama/Llama-3-8B/bfloat16/capital-probe/12/post_mlp/probe_train_0001.bin",
#     ...
#   ]
# }
```

Server-side, the entrypoint pulls `extra_body.capture` off the
request, copies it onto `SamplingParams.capture`, and threads the
eventual `RequestOutput.capture_results` into the response body as
`capture_results`.

The streaming path surfaces `capture_results` in the final SSE frame
alongside `usage` (as a sibling field, not nested inside `usage`),
same as the existing activation storing design.

## Built-in consumers

vLLM ships two built-in consumers in
`vllm/v1/capture/consumers/`, both registered via entry points in
vLLM's own `pyproject.toml` under `vllm.capture_consumers`.

### `filesystem`

Module: `vllm/v1/capture/consumers/filesystem/`

The filesystem consumer is the direct replacement for the old
activation storing feature's `ActivationWriter`. It writes captured
activations to a `.bin` file per `(request, layer, hook)` with an
atomic rename from `.bin.tmp` on finalize, and a sidecar `.json` file
with the full capture metadata.

**Class attributes:**
- `location = "worker"` (runs in the engine-core subprocess for
  direct access to CPU tensors and the writer thread pool).
- `required_sidecar_fields` = everything the sidecar format needs
  (tag, client_request_id, prompt_token_ids, generated_token_ids,
  model_name, model_dtype, positions, position_kind,
  last_prompt_token_index, created_at, finalized_at).
- `reads_client_spec = True`. Clients opt in via
  `SamplingParams.capture["filesystem"]`.
- `global_capture_spec()` returns `None`. Filesystem captures are
  always per-request.

**Params (from YAML / CLI / `LLM(...)`):**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `root` | `str` | required | Root directory for all captures. |
| `writer_queue_size` | `int` | `1024` | Bounded queue between the manager and writer threads. |
| `writer_timeout_seconds` | `int` | `180` | Per-write timeout; failures become `partial_error`. |
| `writer_threads` | `int` | `4` | Writer thread pool size. |
| `on_collision` | `"overwrite" \| "error" \| "suffix"` | `"overwrite"` | What to do when a file already exists. |
| `max_bytes_per_request` | `int` | `0` | Per-request byte cap; `0` disables. |

**Per-request client spec (`FilesystemCaptureRequest`):**

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class FilesystemCaptureRequest:
    request_id: str              # Required; filename stem, slugged server-side.
    tag: str                     # Required; grouping label, slugged.
    hooks: dict[HookName, list[int] | str | dict]  # Same shape as the current
                                                    # ActivationStoringSpec.hooks.
    positions: PositionSelector  # Same.
```

The filesystem consumer's `validate_client_spec` walks `hooks` and
`positions` with the same rules as the current
`validate_activation_storing` in Phase 1, producing a `CaptureSpec`
or raising `CaptureValidationError`.

**On-disk layout** (unchanged from the current filesystem writer):

```
{root}/
  {model_slug}/
    {model_dtype}/
      {tag_slug}/
        {layer_idx}/
          {hook_name}/
            {request_id_slug}.bin
            {request_id_slug}.json
```

**Payload format:** raw bytes at the model's residual dtype. bf16 is
stored as raw bytes; readers round-trip through `torch.uint16` +
`view(torch.bfloat16)`. Same as today.

**Sidecar format:** JSON with the same fields the current filesystem
writer produces. See the old `docs/features/activation_storing.md`
"File format" section for the canonical schema — the filesystem
consumer's sidecar is identical byte-for-byte to what the old
`ActivationWriter` wrote, which is the only piece of backward
compatibility the new design preserves.

**Why it's a worker consumer:** streaming semantics. Long captures
(e.g., `positions="all"` over a 1k-token prompt × 80 layers) can't
be buffered in memory — the consumer has to write incrementally to
`.bin.tmp` as chunks arrive. The filesystem consumer implements
`CaptureSink` directly (not via the `CaptureConsumer` batched
adapter) to avoid buffering.

### `logging`

Module: `vllm/v1/capture/consumers/logging.py`

Minimal observation consumer. Logs one line per finalized capture:
"request X produced Y rows at layer Z hook H in dtype D". Discards
the actual tensor. Useful as a reference implementation for plugin
authors and as a smoke test for the framework.

**Class attributes:**
- `location = "worker"` (cheap; no reason to cross the process
  boundary).
- `required_sidecar_fields` = empty. Logs only what's always
  present.
- `reads_client_spec = False`.
- `global_capture_spec()` returns a spec configured from params —
  operator-configurable.

**Params:**
- `hooks`: same shape as `FilesystemCaptureRequest.hooks`. Required.
- `positions`: same as `FilesystemCaptureRequest.positions`. Required.
- `level`: logging level. Default `INFO`.

Mostly exists as a worked example for § "Consumer authoring guide".

## Consumer authoring guide

A third-party capture consumer plugin is a single Python package with
a subclass of `CaptureConsumer`, an entry point in `pyproject.toml`,
and optionally a client-spec dataclass.

### Minimal observation consumer

```python
# my_plugin/__init__.py
from vllm.v1.capture import CaptureConsumer, CaptureSpec

class FeatureCounter(CaptureConsumer):
    """Counts how many times each (layer, hook) pair was captured."""

    location = "worker"
    required_sidecar_fields = frozenset()  # Only needs vllm_internal_request_id.
    reads_client_spec = False

    def __init__(self, vllm_config, params):
        super().__init__(vllm_config, params)
        self._counts = {}
        self._hooks = params["hooks"]

    def global_capture_spec(self):
        return CaptureSpec(
            hooks=self._hooks,
            positions="last_prompt",
        )

    def on_capture(self, key, tensor, sidecar):
        req_id, layer, hook = key
        self._counts[(layer, hook)] = self._counts.get((layer, hook), 0) + 1

    def shutdown(self, timeout=30.0):
        for (layer, hook), count in sorted(self._counts.items()):
            print(f"{layer}/{hook}: {count}")
```

```toml
# my_plugin/pyproject.toml
[project.entry-points."vllm.capture_consumers"]
feature_counter = "my_plugin:FeatureCounter"
```

```yaml
# operator's vLLM config
capture_consumers:
  - name: feature_counter
    params:
      hooks:
        post_mlp: [12, 24]
```

Install the plugin with `pip install my_plugin`, boot `vllm serve`
with the config, and captures flow to the counter. No vLLM source
changes.

### Driver-side training loop

```python
# reward_trainer/__init__.py
import torch
from vllm.v1.capture import CaptureConsumer, CaptureSpec

class RewardTrainer(CaptureConsumer):
    location = "driver"
    required_sidecar_fields = frozenset({"generated_token_ids"})
    reads_client_spec = False

    def __init__(self, vllm_config, params):
        super().__init__(vllm_config, params)
        self._model = torch.load(params["checkpoint_path"])
        self._opt = torch.optim.AdamW(self._model.parameters(), lr=params["lr"])
        self._target_layer = params.get("target_layer", 12)

    def global_capture_spec(self):
        return CaptureSpec(
            hooks={"post_mlp": [self._target_layer]},
            positions="last_prompt",
        )

    def on_capture(self, key, tensor, sidecar):
        # tensor: (1, hidden_size) — last_prompt
        # sidecar: includes vllm_internal_request_id, generated_token_ids, plus
        #   the framework-always fields (layer, hook, shape, dtype, ...)
        reward = self._model(tensor.unsqueeze(0)).squeeze()
        loss = -reward.mean()
        loss.backward()
        self._opt.step()
        self._opt.zero_grad()

    def shutdown(self, timeout=30.0):
        torch.save(self._model.state_dict(), "reward_checkpoint.pt")
```

The class has `location = "driver"`, so vLLM handles all the
worker→driver plumbing. The user's code is 20 lines.

### Client-spec-reading consumer

For consumers where clients drive the capture selection (like the
filesystem consumer), set `reads_client_spec = True` and implement
`validate_client_spec`:

```python
@dataclass
class SnapshotRequest:
    tag: str
    layer: int
    hook: str
    position: int

class SnapshotConsumer(CaptureConsumer):
    location = "worker"
    reads_client_spec = True

    def __init__(self, vllm_config, params):
        super().__init__(vllm_config, params)
        self._out_dir = Path(params["out_dir"])
        self._num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )

    def validate_client_spec(self, raw, ctx):
        # raw is whatever the client put in SamplingParams.capture["snapshot"].
        # We accept a dict matching SnapshotRequest.
        req = SnapshotRequest(**raw)
        if req.layer < 0 or req.layer >= self._num_layers:
            raise CaptureValidationError(
                f"layer {req.layer} out of range [0, {self._num_layers})"
            )
        if req.position < 0 or req.position >= ctx.num_prompt_tokens:
            raise CaptureValidationError(
                f"position {req.position} out of range [0, {ctx.num_prompt_tokens})"
            )
        return CaptureSpec(
            hooks={req.hook: [req.layer]},
            positions=[req.position],
        )

    def on_capture(self, key, tensor, sidecar):
        # per-request snapshot to disk
        ...
```

Clients then:

```python
sampling_params = SamplingParams(
    capture={
        "snapshot": {"tag": "demo", "layer": 12, "hook": "post_mlp", "position": 42}
    }
)
```

The dict value can be any JSON-shaped structure — the consumer's
validator decides the schema. The `ctx: CaptureContext` argument
gives the validator what it needs to sanity-check against the
request (prompt length, num_hidden_layers, etc.).

## Error handling

### Per-key status lifecycle

Every capture key passes through a small state machine:

```
(admission) --> pending --> ok
                      \-->  partial_error
                      \-->  error
```

- **`pending`**: the request is registered, captures may still be
  flowing.
- **`ok`**: finalize succeeded; consumer produced a terminal payload
  (file paths, callback return, whatever). The captured data is
  durable / has been handed off.
- **`partial_error`**: some of the captures for this key landed
  successfully, but at least one chunk or the finalize failed.
  The consumer should leave any partial state on disk / in memory
  for inspection. Text generation is never aborted; the request
  still returns tokens.
- **`error`**: no data for this key was produced (admission rejected,
  validator raised, etc.). The client sees the error string in
  `RequestOutput.capture_results[consumer_name].error`.

A consumer's `get_result(key)` returns `None` until the key reaches a
terminal state, then returns the `CaptureResult`.

### Back-pressure

The framework provides one back-pressure mechanism: bounded queues
with timeouts.

- **Worker consumer internal queues.** The consumer chooses its own
  bound. The filesystem consumer uses the existing `writer_queue_size`
  and `writer_timeout_seconds`.
- **Driver consumer cross-process queues.** The shim's `submit_chunk`
  blocks on `queue.put` up to a configurable timeout (default 30s).
  On timeout, the shim records `partial_error` for the affected
  request and drops the chunk.
- **Runner-level admission.** At request-admission time, if a
  consumer is currently backed up (e.g., queue > 75% full), the
  framework can reject new captures for that consumer with
  `partial_error`. This is opt-in per consumer via a
  `max_pending_keys` class attribute — default is unbounded
  (accept everything).

Back-pressure failure never halts the engine. The worst case is a
request returning `capture_results[consumer_name].status ==
"partial_error"` while still producing tokens.

### Graceful shutdown

Engine teardown calls `shutdown(timeout)` on every active consumer in
LIFO registration order. Each consumer has its own timeout budget
(default 30s). During shutdown:

- Consumers drain their in-flight queues up to the timeout.
- Any work still in flight at timeout is marked `error` with a
  `"shutdown"` reason.
- Worker-side consumer threads join; driver-side receiver threads
  join.
- Driver consumer instances get their `shutdown()` called on the
  driver-process main thread (where they were instantiated).

## Threading and process model

- The **capture manager** runs on whatever thread the runner's
  `_finalize_capture_step` runs on (normally the worker's main
  execution thread).
- **Worker consumers** may spawn their own background threads. The
  manager's `submit_chunk` / `submit_finalize` must be non-blocking,
  so consumers that do work (disk writes, network sends) must
  offload to their own threads.
- **Driver consumers** run on a dedicated receiver thread in the
  driver process — one thread per driver consumer. The receiver
  thread is spawned when the consumer is instantiated and joined at
  shutdown.
- **Cross-process tensor handoff** uses `torch.multiprocessing.Queue`
  with shared-memory tensor storage. Tensors cross the boundary in
  O(1); metadata dicts cross via pickle (negligible for small dicts).
- **The manager's scratch tensors** live in the worker process and
  are freed when the step finalizes. Consumers never see GPU
  tensors; they only see CPU tensors produced by the D2H copy.

## Invariants

The implementation must preserve these invariants. Violations are
bugs.

1. **Capture reads the pristine residual.** `maybe_capture_residual`
   runs *before* the steering vector is added at the same hook
   point. Captures reflect the un-steered residual, independent of
   any active steering state. Inherited from the activation storing
   invariants; preserved unchanged.

2. **Cold path is free.** When `capture_consumers` is empty, no
   capture code runs during the forward pass. The custom op is
   constant-folded by `torch.compile`. Inherited; preserved unchanged.

3. **Per-step plan is batch-order consistent.** Gather indices for
   a `(layer, hook)` key are in the same order the runner fed tokens
   into the model. The finalize walk preserves that order so the
   dispatch to consumers sees rows in a stable sequence.

4. **Multi-step captures preserve append order.** For a given
   `CaptureKey`, `CaptureChunk`s are delivered to every interested
   consumer in `row_offset` order. Consumers are guaranteed a
   monotonic stream per key.

5. **Finalize is atomic per consumer.** When
   `RequestOutput.capture_results[consumer_name].status` is `"ok"`,
   the consumer's state for that key is fully written. The client
   can rely on the status field; partial visibility (some rows
   present, not others) never corresponds to `"ok"`.

6. **Partial failures never abort text generation.** Writer errors,
   consumer callback exceptions, back-pressure timeouts, shutdown —
   none of these cause the request to fail. The request always
   returns tokens. Capture failures surface on
   `capture_results[consumer_name]` for clients that care.

7. **Prefix-cache positions rejected at admission.** Clients
   requesting logical positions that fall below
   `initial_num_computed_tokens` receive HTTP 400 immediately, not
   a silent empty file after the request finishes. Enforced by
   consumer validators (via `CaptureContext.num_computed_tokens`).

8. **TP > 1 / PP > 1 rejected with a clear error.** If any
   `capture_consumers` entry would be active under TP/PP > 1, engine
   init fails with an error referencing the consumer name. Inherited
   from activation storing; preserved.

9. **Consumer isolation.** A consumer crashing or raising an
   exception inside `on_capture` / `on_chunk` does not affect other
   consumers. The framework catches exceptions, records them as
   `error` for the affected key, and continues dispatching. Other
   consumers continue receiving captures normally.

10. **`vllm_internal_request_id` is the only identity the framework
    guarantees.** All other sidecar fields are consumer-opt-in. A
    consumer that only uses `vllm_internal_request_id` can be
    written without any knowledge of the client-side identity model.

## File layout

```
vllm/v1/capture/
├── __init__.py               # Re-exports the public API
├── types.py                  # CaptureKey, CaptureChunk, CaptureFinalize,
│                             #   CaptureResult, CaptureStatus, CaptureSpec,
│                             #   CaptureContext, HookName, PositionSelector
├── sink.py                   # CaptureSink protocol
├── consumer.py               # CaptureConsumer base class + BatchedAdapter
├── manager.py                # CaptureManager (renamed from
│                             #   ActivationCaptureManager; merged with the
│                             #   runner-side prepare/finalize helpers)
├── plan.py                   # StepCapturePlan, CapturePositionEntry,
│                             #   CaptureBatchView (moved from
│                             #   activation_capture.py)
├── registry.py               # Entry-point discovery, name resolution,
│                             #   consumer instantiation
├── config.py                 # Top-level CaptureConsumersConfig + per-consumer
│                             #   params parser
├── driver_bridge.py          # _DriverQueueShim, _DriverReceiver, all the
│                             #   cross-process plumbing for driver consumers
├── errors.py                 # CaptureValidationError, UnknownCaptureConsumerError
│
└── consumers/                # Built-in consumers — ship as plugins
    ├── __init__.py
    ├── filesystem/
    │   ├── __init__.py       # FilesystemConsumer (CaptureSink implementation)
    │   ├── writer.py         # Internal writer thread pool (was the old
    │   │                     #   vllm/v1/worker/activation_writer.py)
    │   ├── validation.py     # FilesystemCaptureRequest validation (was the
    │   │                     #   old activation_storing_validation.py)
    │   └── types.py          # FilesystemCaptureRequest dataclass
    └── logging.py            # LoggingConsumer (CaptureConsumer subclass)
```

**Stays where it is:**
```
vllm/model_executor/layers/activation_capture.py
  - torch.ops.vllm.capture_residual op registration
  - maybe_capture_residual gate
  - _HOOK_NAME_TO_ID / _HOOK_ID_TO_NAME
  - set_active_capture_manager / get_active_capture_manager
```

This module is model-facing (called by `apply_layer_steering`) and
should not move. The capture manager imports from it; it does not
import from the capture manager. The name can optionally be renamed
(`activation_capture.py` → `capture_op.py`) but the module location
stays. Leave the name alone unless there's a compelling reason.

**Removed / consolidated:**
```
vllm/config/activation_storing.py           → gone
vllm/config/activation_storing_types.py     → partially moved to vllm/v1/capture/types.py;
                                               filesystem-specific parts move to
                                               vllm/v1/capture/consumers/filesystem/types.py
vllm/entrypoints/openai/activation_storing_validation.py
                                            → moved to vllm/v1/capture/consumers/filesystem/validation.py
vllm/v1/worker/activation_writer.py         → moved to vllm/v1/capture/consumers/filesystem/writer.py
tests/v1/worker/activation_storing/         → moved to tests/v1/capture/
tests/entrypoints/openai/test_activation_storing_protocol.py
                                            → moved to tests/v1/capture/consumers/filesystem/test_validation.py
```

**Updated:**
```
vllm/v1/worker/gpu_model_runner.py
  - Remove all `activation_storing`-named state
  - Wire in `CaptureManager` from `vllm/v1/capture/manager.py`
  - New helpers: `_prepare_capture_step`, `_finalize_capture_step`
    (renamed from `_prepare_activation_storing_step` etc.)

vllm/v1/engine/__init__.py
  - `EngineCoreOutput.capture_result` becomes
    `EngineCoreOutput.capture_results: dict[str, CaptureResult]`

vllm/v1/engine/output_processor.py
  - Thread the dict through to `RequestOutput.capture_results`

vllm/v1/outputs.py
  - `ModelRunnerOutput.capture_results: dict[req_id, dict[consumer_name, CaptureResult]]`

vllm/outputs.py
  - Remove `RequestOutput.activation_storage`
  - Add `RequestOutput.capture_results: dict[str, CaptureResult]`

vllm/sampling_params.py
  - Remove `activation_storing`
  - Add `capture: dict[str, Any] | None = None`

vllm/entrypoints/openai/chat_completion/protocol.py
vllm/entrypoints/openai/completion/protocol.py
  - Remove `activation_storing` request field
  - Remove `ActivationStorageResponse`
  - Add `capture: dict[str, Any] | None` request field
  - Add `CaptureResultResponse` response model (keyed by consumer name)

vllm/entrypoints/openai/chat_completion/serving.py
vllm/entrypoints/openai/completion/serving.py
  - Remove `_admit_activation_storing`
  - Add `_admit_capture` that walks `request.capture`, calls each
    targeted consumer's `validate_client_spec` via the manager, and
    mutates `sampling_params.capture` in place
  - Thread `final_res.capture_results` onto the response body

vllm/engine/arg_utils.py
  - Remove all `--activation-storing*` flags
  - Add `--capture-consumers` list flag (`name:key=value,key=value`)
  - Add top-level `capture_consumers` config field on `VllmConfig`

vllm/config/vllm.py
  - Remove `activation_storing_config`
  - Add `capture_consumers_config: CaptureConsumersConfig | None`
```

## Migration from activation storing

The old `activation_storing` feature goes away wholesale. The
migration path for clients:

| Old | New |
|---|---|
| `vllm serve --activation-storing /mnt/nas/activations` | `vllm serve --capture-consumers 'filesystem:root=/mnt/nas/activations'` (or YAML) |
| `LLM(model=..., activation_storing="/tmp")` | `LLM(model=..., capture_consumers=[{"name": "filesystem", "params": {"root": "/tmp"}}])` |
| `ActivationStoringConfig(root_path=..., writer_threads=4)` | `{"name": "filesystem", "params": {"root": ..., "writer_threads": 4}}` |
| `SamplingParams(activation_storing=ActivationStoringSpec(request_id="x", tag="y", hooks={...}, positions=...))` | `SamplingParams(capture={"filesystem": FilesystemCaptureRequest(request_id="x", tag="y", hooks={...}, positions=...)})` |
| `extra_body={"activation_storing": {...}}` | `extra_body={"capture": {"filesystem": {...}}}` |
| `result.activation_storage.status` | `result.capture_results["filesystem"].status` |
| `result.activation_storage.paths` | `result.capture_results["filesystem"].payload` |

The sidecar JSON format on disk is **unchanged** byte-for-byte —
existing scripts that read `.bin` + `.json` pairs from a capture root
keep working without modification. The directory layout is also
unchanged (`{root}/{model_slug}/{dtype}/{tag}/{layer}/{hook}/{req}.bin`).
All the breaking changes are on the API surface, not the on-disk
format.

## Future work

Out of scope for the initial implementation but architecturally
compatible:

1. **Same-forward-pass feedback.** A new sync-hook variant of the
   custom op that fires during the compiled forward graph and can
   modify the steering buffer for a later layer in the same pass.
   Requires a separate `SyncCaptureSink` protocol operating on GPU
   tensors. The `CaptureConsumer`/`CaptureSink` hierarchy designed
   here does not need to change — it's a parallel integration
   point.

2. **Explicit `session_id` on `SamplingParams`.** The `tag` field
   today is overloaded: it's a path segment for the filesystem
   consumer AND it's the recommended scope identifier for
   cross-request feedback consumers. A future release can add a
   dedicated `session_id` or `scope_id` field to keep the concerns
   separate. The framework doesn't need this now — consumers that
   need scoping use `tag` or client-custom fields declared via
   `required_sidecar_fields`.

3. **Driver → worker RPC.** Some driver-side consumers (like a
   dynamic steering probe) want to send state updates back to the
   worker — "update this steering vector before the next forward
   pass". Today this happens via the existing steering registry
   which has its own driver-side API. A future extension could
   add a generic `driver_to_worker` RPC channel as part of the
   framework for plugins that need their own state channel back to
   the worker.

4. **Consumer-level filtering on `(request, layer, hook)`.** The
   manager currently dispatches captures based on which consumers'
   specs match. A future extension could let consumers filter
   dynamically at dispatch time — e.g., "I want post_mlp at layer 12,
   but only for requests whose tag matches a regex". This would
   allow finer-grained scoping without adding new config fields.

5. **Live capture streaming.** The protocol finalizes on request
   completion. A future addition could expose per-step dispatch
   publicly so consumers can react to captures mid-request. Useful
   for live interpretability dashboards and long-generation
   observability.

6. **Composable consumer pipelines.** A consumer that produces a
   transformed tensor (e.g., "apply this PCA projection") that
   downstream consumers receive. Would require an explicit DAG of
   consumer dependencies. Not needed for the initial set of use
   cases.

## FAQ

**Why a new subpackage instead of extending the existing activation
storing module?**
Because the existing module's public API assumes a single consumer
(the filesystem writer) and bakes that assumption into `SamplingParams`,
`RequestOutput`, and the config tree. Pluggability requires breaking
those assumptions, and breaking them in place would leave deprecated
APIs for no benefit. A clean subpackage + wholesale rename is simpler
than a migration shim.

**Why not just make `ActivationWriter` abstract and have people
subclass it?**
The writer's streaming semantics aren't the right abstraction for
most use cases. Training loops, reward probes, observation plugins,
dashboards — none of them want "append bytes to a file". They want
"here's a finalized tensor per request". `CaptureConsumer` is the
right abstraction for 90% of consumers; `CaptureSink` is the escape
hatch for the remaining 10% that need chunk-level streaming.

**Why entry points instead of a `register_capture_consumer()`
function?**
Entry points are the standard Python plugin mechanism and they work
with `vllm serve`, `pip install`, and containerized deployments
without requiring operators to write Python code or modify a config
registry. A user who pip-installs `my-vllm-plugin` can immediately
reference `name: my_trainer` in their vLLM YAML config. The
registration function is always there as an escape hatch for advanced
users (e.g., dynamically-registered consumers in a notebook), but
entry points are the primary discovery path.

**Why does the filesystem consumer stay shipped with vLLM instead of
becoming an external package?**
Because it's the reference implementation everyone will compare
against, and because it's the only consumer with non-trivial
streaming semantics (bf16 byte layout, multi-step append, atomic
rename). Keeping it in-tree ensures it stays compile-graph-safe and
invariant-respecting. It's structured as a plugin (registered via
entry points, not special-cased) so the framework itself doesn't
depend on filesystem-specific knowledge.

**Can I have multiple instances of the same consumer type with
different configs?**
Yes — consumer names in `capture_consumers` are instance names, not
class names. You can have:

```yaml
capture_consumers:
  - name: filesystem  # ...but wait, 'name' is the entry-point name
```

OK actually this is a design subtlety. The `name` field in the config
resolves to an entry-point class, and each entry gets its own
instance. If you want two filesystem consumers with different roots:

```yaml
capture_consumers:
  - name: filesystem
    instance_name: primary
    params: {root: /mnt/nas/primary}
  - name: filesystem
    instance_name: mirror
    params: {root: /mnt/nas/mirror}
```

The `instance_name` field (optional) disambiguates. Defaults to
`name` if omitted. `RequestOutput.capture_results` is keyed by
`instance_name`. Duplicate instance names in the config raise at
engine init.

**What happens if a driver-side consumer raises inside `on_capture`?**
The receiver thread catches the exception, records it as
`CaptureResult(status="error", error=str(exc))` for the affected
key, logs a warning, and continues. Other consumers and other keys
are unaffected. The request still returns tokens. Invariants 6 and 9.

**Can a consumer write to the steering registry to update a steering
vector?**
Yes, if it has access to it. Driver-side consumers have
driver-process access, so they can call `steering_registry.update_global_vector(...)`
directly. Worker-side consumers run in the engine-core subprocess
alongside the steering manager's worker-side buffers, so they can
poke at those directly (carefully — thread safety applies). The
framework doesn't provide a unified "update steering from a
consumer" API; consumers use whatever the steering feature exposes.

**What's the minimum amount of code to ship a no-op capture consumer
to test that my plugin infrastructure works?**

```python
# minimal_plugin/__init__.py
from vllm.v1.capture import CaptureConsumer, CaptureSpec

class NoOpConsumer(CaptureConsumer):
    location = "worker"

    def global_capture_spec(self):
        return CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")

    def on_capture(self, key, tensor, sidecar):
        pass
```

```toml
[project.entry-points."vllm.capture_consumers"]
noop = "minimal_plugin:NoOpConsumer"
```

```yaml
capture_consumers:
  - name: noop
    params: {}
```

That's 11 lines of Python. Installing the plugin and adding the YAML
entry makes every request produce a captured-and-discarded row at
layer 0 `post_mlp`.
