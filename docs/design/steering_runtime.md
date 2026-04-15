# Steering Runtime Design

This document describes the runtime design of activation steering in vLLM.
It is intended for contributors working on scheduler, worker, model wiring,
prefix caching, or graph execution.

For user-facing setup and examples, see [Activation Steering](../features/steering.md).

## Design Goals

The runtime has to satisfy all of the following at once:

- support global and per-request steering
- distinguish prefill and decode behavior
- preserve prefix-cache correctness
- work under continuous batching
- work with `torch.compile` and CUDA graph replay
- avoid per-step graph recompilation or graph invalidation

That combination is what makes steering more subtle than a simple residual add.

## Core Model

Steering is represented as effective phase-specific configs:

```text
effective_prefill = base + prefill_specific
effective_decode  = base + decode_specific
```

This exists independently for:

- global steering state
- per-request steering state

At runtime, the effective per-token steering seen by the model is:

```text
prefill token -> global_prefill_effective + request_prefill_effective
decode token  -> global_decode_effective  + request_decode_effective
```

## Main Components

### `SamplingParams`

`SamplingParams` owns the user-facing fields:

- `steering_vectors`
- `prefill_steering_vectors`
- `decode_steering_vectors`

It resolves them into phase-specific effective vectors and hashes.

### `Request`

Each request carries:

- `prefill_steering_config_hash`
- `decode_steering_config_hash`

These are phase-specific identities used by:

- scheduler admission
- worker registration
- prefix-cache key generation for prefill

### `Scheduler`

The scheduler admits requests subject to steering-capacity limits and must
reason about phase transitions:

- prefill requests consume prefill steering capacity
- decode requests consume decode steering capacity
- requests near the prefill/decode boundary may require transition-aware
  reservation so decode admission does not fail mid-step

### `SteeringManager`

The worker-side `SteeringManager` owns:

- per-request config registration
- refcounting
- table row assignment
- global vector caches
- population of per-layer steering tables

Rows are phase-aware. A config hash is not enough on its own; the manager
must also know whether that hash is registered as prefill or decode.

### Model Runner

The model runner assembles:

- request-to-row mappings
- token-to-row steering index buffers
- per-layer steering tables

It also handles deferred registration when decode registration cannot happen
immediately because the batch is at steering capacity.

## Table and Index Layout

Each steerable layer owns a steering table per hook point. Conceptually:

| Row | Meaning |
| --- | --- |
| 0 | no steering |
| 1 | global effective prefill |
| 2 | global effective decode |
| 3+ | per-request rows |

All layers share the same token-to-row index for a step. Different hook
points reuse the same row mapping but look up different per-hook tables.

That is why steering supports multiple hook points without multiplying the
per-token bookkeeping cost.

## Phase Semantics

The critical invariant is:

- prefill semantics are tied to prompt-token KV creation
- decode semantics are tied to generated-token continuation

This affects both admission and APC.

Phase detection cannot rely on trivial heuristics like "one token means
decode". It has to use the actual request state, because chunked prefill,
full cache hits, and resumed streaming requests all complicate that boundary.

## Prefix Cache Semantics

Prefix caching must separate requests whose prompt KV differs.

That means:

- prefill steering is part of the cache identity
- decode-only steering is not part of prompt KV identity
- global base/prefill steering changes invalidate cache reuse

The cache key integration happens through extra block-hash components attached
to prefill hashing.

### Why Streaming Continuation Is Tricky

In resumable streaming, prior output tokens can be folded back into the prompt
for the next turn. That means a block that was previously decode-only can
become a prompt block in the continued request.

When that happens, the request must refresh all APC-related state:

- phase-specific steering hashes
- block-hash override fields used by cache hashing
- any block-hash chain whose old phase interpretation is now stale
- prefix-cache read policy derived from current sampling params

If those are not refreshed together, cache hits and misses become incorrect.

## Deferred and Pending Registration

Decode registration can be deferred when a request transitions phases but the
worker has no free steering rows at that moment.  New-request registrations
can also be deferred on capacity exhaustion during admission.

The runtime uses a two-queue priority model:

1. **Transitions queue** (`_pending_steering_transitions`): prefill→decode
   transitions for in-flight requests that have already consumed KV cache.
   This queue is drained first (FIFO).
2. **Registrations queue** (`_pending_steering_registrations`): new-request
   deferrals.  Only processed once the transitions queue is empty.

A third deferral mechanism handles the lazy-init case: when the HTTP API sets
global vectors before the `SteeringManager` has been constructed (the manager
is lazily created on the first steering-relevant step), those vectors are
queued on `_pending_steering_globals` at the model runner and replayed during
manager init. This is distinct from the two queues above: those handle
capacity exhaustion after the manager exists; `_pending_steering_globals`
handles the case where the manager doesn't exist yet.

This priority ordering ensures that requests already consuming resources get
steering table rows before newly admitted requests that haven't started yet.

The runtime has to preserve these invariants:

- active requests keep running even if decode registration is deferred
- transitions are retried before new-request registrations
- new-request registrations are only attempted when no transitions are pending
- stale pending entries are dropped if the request finishes or changes phase
- fallback behavior must remain correct if a request temporarily uses a global row

This is one of the places where scheduler capacity logic and worker state must
match exactly.

## Continuous Batching

Steering has to work with mixed batches containing:

- unsteered requests
- globally steered requests
- distinct per-request steered requests
- prefill and decode tokens in the same step

The key runtime requirement is that every token in the flattened batch maps
to the correct steering row for its request and phase.

That is why the system uses request-aware row assignment plus a per-step token
index buffer rather than trying to mutate model weights directly.

## `torch.compile` and CUDA Graphs

Steering correctness under compiled execution depends on one core rule:

- graph replay must read live steering buffers, not constants specialized at trace time

The current design achieves that by using persistent GPU buffers and an opaque
custom steering op. Steering data is updated in-place between steps, and graph
replay observes the updated buffer contents.

Important consequences:

- steering changes do not require recompiling the model
- graph replay can serve requests with different steering configs across steps
- correctness depends on buffer updates and row/index population happening
  before the forward pass

## Extending Steering to New Models

To add steering to another model family, contributors need to wire:

- layer indices
- per-hook steering tables
- the shared steering index
- `apply_steering` calls at the intended residual-stream hook points

The extension work is model-specific, but the runtime invariants above do not
change.

## Current Boundaries

This design document reflects the v1 steering runtime. Known boundaries:

- no v2 model runner integration yet (v2 is dev-flag-gated in vllm; steering
  integration is pending)
- see [Activation Steering](../features/steering.md#supported-scope) for the
  current list of wired decoder architectures

## Named Steering Modules (runtime)

Named steering modules are pre-registered vector configurations that requests
reference by name instead of sending vectors inline. The runtime shape is:

- The registry lives on FastAPI app state
  (`app.state.steering_module_registry`), populated either at startup via
  `--steering-modules` or at runtime via
  `POST /v1/steering/modules/register`. The registry implementation is in
  `vllm/entrypoints/openai/steering/registry.py`.
- Resolution happens in the OpenAI serving handlers
  (`chat_completion/serving.py`, `completion/serving.py`) when a request
  specifies `steering_name` in `extra_body`. The resolver looks up the named
  module, merges it with any inline `steering_vectors` fields via
  `merge_steering_specs`, and writes the merged spec back onto
  `SamplingParams` before the request enters the scheduler.
- The scheduler and worker do not distinguish named from inline vectors once
  the spec is on `SamplingParams` — the rest of the runtime sees only the
  final resolved vectors.
