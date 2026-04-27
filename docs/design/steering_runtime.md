# Steering Runtime Design

This document describes the runtime design of activation steering in vLLM.
It is intended for contributors working on scheduler, worker, model wiring,
prefix caching, or graph execution.

For user-facing setup and examples, see [Activation Steering](../features/steering.md).

## Design Goals

The runtime has to satisfy all of the following at once:

- support per-request steering
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

At runtime, the effective per-token steering seen by the model is:

```text
prefill token -> request_prefill_effective
decode token  -> request_decode_effective
```

The table layout below reserves rows for global prefill/decode vectors that a
later PR exposes through an HTTP API; until then those rows are always zero
and have no effect.

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
guarantee that capacity is available for every admitted request across all
phases:

- prefill requests consume prefill steering capacity
- decode requests consume decode steering capacity
- decode-only requests (e.g., full cache hits that skip prefill) are now
  capacity-checked at admission, closing a previous gap where they could
  bypass steering accounting
- the scheduler guarantees that by the time a request reaches the worker,
  its steering row can be allocated; there is no deferred-registration
  fallback path

### `SteeringManager`

The worker-side `SteeringManager` owns:

- per-request config registration
- refcounting
- table row assignment
- population of per-layer steering tables

Rows are phase-aware. A config hash is not enough on its own; the manager
must also know whether that hash is registered as prefill or decode.

### Model Runner

The model runner assembles:

- request-to-row mappings
- token-to-row steering index buffers
- per-layer steering tables

Registration is expected to succeed for every request in the batch. If
`get_row_for_config` is called with an unregistered nonzero config hash,
the manager raises `RuntimeError` instead of silently falling back to a
global row. This fail-fast behavior is safe because the scheduler has
already guaranteed capacity before admission.

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

## Guaranteed Capacity and Hard Registration Errors

The scheduler guarantees that by the time a request reaches the worker, its
steering row can be allocated. There are no pending queues or deferred
registration paths at the worker.

This guarantee is enforced end-to-end:

- The scheduler tracks both prefill and decode steering capacity, including
  decode-only requests that skip prefill entirely (e.g., full cache hits).
- Admission is refused if there is no free steering row for the request's
  phase-specific config hash.
- At the worker, `SteeringManager.get_row_for_config` raises `RuntimeError`
  if called with an unregistered nonzero config hash, rather than silently
  falling back to a global (zero-steering) row.

The pending queues (`_pending_steering_transitions` and
`_pending_steering_registrations`) and their associated two-queue priority
drain have been removed. The rationale: tokens generated under wrong steering
poison the KV cache permanently. A request that runs with zero steering while
"waiting" for a row produces KV entries that cannot be corrected later. The
only safe behavior is to never admit a request unless its steering row is
guaranteed.

Invariants the runtime preserves:

- every admitted request has a steering row available at registration time
- `get_row_for_config` never silently degrades to a global row for
  unregistered configs
- decode-only requests are capacity-checked at the scheduler, closing the
  gap where they previously bypassed steering accounting
- the scheduler and worker capacity models are kept in strict agreement

## Continuous Batching

Steering has to work with mixed batches containing:

- unsteered requests
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
- only the Gemma 3 decoder is wired; additional model families, a global HTTP
  API, named modules, and distributed-execution support land in follow-up PRs
  per the staged plan in the RFC
