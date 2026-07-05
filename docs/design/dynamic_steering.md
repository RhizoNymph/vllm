# Dynamic Steering — Design

Status: Phases 0 and 1a implemented (this branch); 1b–2 proposed
Branch: `feat/dynamic-steering`
Audience: contributors familiar with the steering runtime
([steering_runtime.md](steering_runtime.md)) and the capture-consumer
framework ([capture_consumers.md](capture_consumers.md)).

Supersedes and folds in the earlier draft sketch "Dynamic Steering via
Capture Consumers" (previously uncommitted in the
`feat+steering-consumer-controller` worktree); differences from that
sketch are called out in [§10](#10-relationship-to-the-earlier-sketch).

## 1. Problem

Activation capture observes the residual stream; steering mutates it.
Today the only bridge between them is a human: look at captured
activations offline, derive vectors, post them to `/v1/steering/set`.

**Dynamic steering** closes the loop in-process: use the activations the
model is producing *right now* to decide **when** to steer (probe-gated
triggering), **how much** (scale modulation), and eventually **with
what** (vector selection/synthesis) — at latencies between one decode
step and zero (same-token, in-graph).

Concrete uses:

- Conditional interventions: apply a refusal/jailbreak/persona vector
  only when a trained probe fires, instead of steering every token.
- Closed-loop magnitude control: hold a concept's projection at a
  target level (proportional control) instead of open-loop scaling.
- Adaptive safety probes: monitor for drift, engage countermeasures.

## 2. Where the systems already meet (verified)

The two systems intersect in a single line. Every steerable layer runs,
per hook point (`vllm/model_executor/layers/steering.py:166`):

```python
def apply_layer_steering(module, hidden_states, hook_point):
    maybe_capture_residual(hidden_states, module.layer_idx, hook_point.value)
    ...
    return torch.ops.vllm.apply_steering(hidden_states, table, index, any_active)
```

Capture fires **before** steering at the same site, so a monitor always
sees the pristine (un-steered) residual — the loop never measures its
own intervention at the monitored hook.

The per-step runtime anatomy both phases of this design build on:

| Mechanism | Location | Property we exploit |
| --- | --- | --- |
| Steering tables | per-layer persistent GPU buffers `(max_configs+3, hidden)`; row 0 = zeros, rows 1/2 = global prefill/decode, rows 3+ = per-request (`steering.py:65-115`) | mutations between steps are visible to CUDA-graph replay |
| `steering_index` | shared `(max_tokens,)` int64, token→row, rebuilt each step (`steering_model_runner_mixin.py:_update_steering_buffers`, called at `gpu_model_runner.py:4700` right before forward) | a per-step, host-controlled routing decision |
| `SteeringManager` | `vllm/v1/worker/steering_manager.py`; `update_global_vectors()` / `register_config()` / `release_config()`, `_tables_dirty` → `populate_steering_tables()` | a complete in-process mutation API; no HTTP needed |
| Global-spec capture | `CaptureManager.on_hook` (`vllm/v1/capture/manager.py:817-820`): fixed-shape `copy_` of the full `[num_tokens, hidden]` residual into a persistent buffer, **baked into CUDA graphs at warmup** | graph-safe, zero-eager-forcing observation of any fixed `(layer, hook)` set |
| Client-spec capture | dynamic `index_select` (`manager.py:826-836`), forces eager via `CaptureStepGate` | why dynamic-steering monitors must *not* ride the client-spec path |
| Dispatch pipeline | side-stream D2H → pinned CPU → dispatch thread → consumers (`manager.py:870+`, `1137+`) | per-step chunks reach a worker consumer ~ms after the forward |
| Triton apply kernel | `steering_kernel.py:38-100`: one program per token, `row = index[pid]`, fused gather+cast+add | trivially extensible with scale multiplies |

## 3. Architecture: monitor → policy → actuate

A dynamic steering system decomposes into three stages, each with a
cheap/fast axis and an expressive/slow axis:

- **Monitor** — which activations to observe. Always via the
  *global-spec persistent-buffer* mechanism (graph-safe, no eager
  forcing). A probe bank `(k, hidden)` per monitored `(layer, hook)`
  turns residuals into `k` scalar scores per token.
- **Policy** — scores → decisions. Threshold + hysteresis (gating),
  proportional/PID (modulation), argmax over a bank (selection).
  Policies should be pure functions of rank-replicated inputs (see §6).
- **Actuate** — decisions → steering state, in increasing cost:
  1. **Scale a row / the dynamic tier** (Phase 1b primitives, no H2D
     vector traffic),
  2. **Gate a token** (Phase 2 in-graph per-token scale),
  3. **Rewrite vectors** (existing `update_global_vectors` /
     `register_config` machinery; one table repopulate).

Three loop latencies fall out:

| Loop | Path | Latency | Phase |
| --- | --- | --- | --- |
| Async consumer | capture dispatch thread → action queue → next `_update_steering_buffers` | 1–3 decode steps | **0 (implemented)** |
| Sync consumer | `on_step` on the model-runner thread post-forward, reading persistent capture buffers directly; actions applied inline before the next step | exactly 1 step | **1a (implemented)** |
| In-graph conditional | monitor op at layer L writes per-token scales consumed by steering at layers > L, same forward; parameters tuned between steps by a consumer | 0 (same token) | 2 |

All three are one mechanism at different depths: async consumers are
Python policy off the critical path, sync consumers are Python policy
*on* the critical path (with a budget), and the in-graph op is compiled
policy whose parameters a consumer tunes. Phases 1 and 2 extend the
existing capture-consumer framework rather than adding a parallel
controller system — see §5.1.

### 3.1 Policy expressiveness contract (locked)

"Policy" is three separable components, and what each tier admits is a
fixed contract — do not blur them:

- **Probe** (detector: activations → score). A *pretrained* probe is
  fixed weights at inference — a direction, linear classifier, MLP, or
  SAE feature. **Usable on every tier**, in-graph included, as long as
  its evaluation is per-token graph-safe tensor math. (The example
  controller already loads one via `probe_path` / `probe_packed_path`.)
- **Decision function** (score → gate/gain). In-graph: a fixed
  elementwise map (sigmoid/step/affine) with parameters in persistent
  buffers. Sync/async: arbitrary Python.
- **Controller state** (anything aggregating across tokens, steps, or
  requests: EMA, hysteresis, "engage N tokens", budget throttle, RL /
  online learning). **Sync or async tier only** — *not* in-graph,
  except simple state expressible as an in-graph persistent-buffer
  read-modify-write (§8 reset-discipline territory).

The in-graph tier (Phase 2) is bounded to ops that are **fixed-shape,
collective-free, allocation-free, per-token** (no cross-token/cross-step
reductions), with **no host sync and no data-dependent control flow**;
tunable parameters live in small persistent buffers so a consumer can
retune them between steps without recapture. A *learned controller is
therefore not lost at the in-graph tier* — it runs one tier up (sync,
1-step latency) and tunes the cheap per-token in-graph gate beneath it.

Determinism (all worker-side tiers): every probe/policy must be a pure,
deterministic function of the **rank-identical post-all-reduce
residual**, so each TP rank computes the same decision with no
communication (§6). Standard pretrained probes satisfy this trivially.

## 4. Phase 0 — consumer + action queue (implemented on this branch)

Smallest core surface that lets a capture consumer drive steering, for
validating probes/policies on real traffic.

**New:** `vllm/v1/worker/steering_action_queue.py`

- `SteeringVectorUpdate(vectors: {hook: {layer: np.float32[hidden]}}, phase, source)` —
  overwrite semantics per `(hook, layer)`, matching
  `SteeringManager.update_global_vectors` (set, not add; a zero vector
  disengages).
- `SteeringActionQueue` — bounded, thread-safe, non-throwing `submit()`
  (drops newest on overflow, rate-limited warning), `drain()` on the
  step thread only.
- Process-global `install_steering_action_queue()` /
  `get_steering_action_queue()`, mirroring
  `set_active_capture_manager()`. Consumers look the queue up lazily at
  submit time because they are constructed before steering init.
- `apply_steering_updates()` — drain-side validation (hook validity,
  layer steerable, hidden-size match, finite values — mirroring
  `set_steering_vectors`) and application via
  `update_global_vectors()`. Per-update isolation: one malformed update
  is rejected with a structured warning; the rest apply.

**Mixin changes** (`steering_model_runner_mixin.py`):

- `_init_steering_state()` installs the queue **only when
  `tp == 1 and pp == 1`** (see §6 for why), else installs `None`.
- `_update_steering_buffers()` drains the queue first thing — before
  the nothing-active short-circuit, since a drained update may be what
  activates steering. Application sets `_tables_dirty`, so the existing
  populate path uploads the state; no new buffer code. Empty-queue
  steady state costs one global read + truthiness check per step.

**Plugin:** `examples/capture_consumers/dynamic_steering_controller/`
(entry point `dynamic_steering`) — a direct `CaptureSink` (the
`CaptureConsumer` batched adapter only delivers at request finalize,
far too late). Global spec on one `(layer, hook)`, `all_generated`
positions; per-token probe scores (cosine or dot), per-request EMA,
max/mean aggregation, threshold+hysteresis engagement, binary or
proportional gain, min-delta emission gating; actuates the **global
decode tier** only. Diagnostics ride `CaptureResult.payload`.

**Hard scope limits (enforced, not advisory):**

- `tp=1, pp=1` — double-enforced (queue not installed; plugin refuses
  to construct).
- Decode tier only — drain rejects `base`/`prefill` updates because
  this path performs no prefix-cache invalidation (§7). The
  `allow_cache_unsafe_phases` escape hatch exists for callers that own
  invalidation, but nothing sets it today.
- Global actuation only — the policy aggregates across requests and
  steers everyone. Per-request actuation is Phase 1.

**Timing** (one full loop):

```
step N    forward: graph-baked copy_ fills the monitor's persistent buffer
          post-forward: dispatch → (H2D, dispatch thread) → controller chunks
          controller: scores → policy → queue.submit(update)        [~ms after fwd]
step N+1  _update_steering_buffers: drain → update_global_vectors → populate
          forward: decode tokens steered by vec * gain
```

If the controller's decision lands after step N+1's drain it applies at
N+2 — hence "1–3 steps". Acceptable for validation; Phase 1 makes it
deterministic.

## 5. Phase 1 — sync consumers + scale primitives

Phase 1 ships in two sub-phases (decided 2026-06-11, see §11):

- **Phase 1a — sync consumers + per-request actuation**
  (**implemented on this branch**). No kernel changes: the `execution`
  axis (§5.1), per-request actuation via a dynamic-override row pool
  (§5.2), the budget metric, the observability ring buffer +
  `GET /v1/steering/dynamic` (§5.5), `SteeringHookPacked` probe banks,
  and the example plugin migrated to sync with per-request actuation
  as its default.
- **Phase 1b — gain primitives.** The kernel work: per-row scale
  tensor (§5.3) and the dynamic additive tier (§5.4), which is also
  the substrate Phase 2's per-token gating extends.

Implementation note: the sync `on_step` hook runs inside
`sample_tokens` (immediately after `_finalize_capture_step`, i.e.
post-sampling), not in `execute_model` — `scheduler_output` is in
scope there, all TP ranks execute it, and same-thread ordering with
the next step's `_update_steering_buffers` preserves the
single-mutator contract.

### 5.1 The `execution` axis: sync vs async consumers

The consumer framework today has two axes that are partially tangled:
`location` (`"worker"` / `"driver"`) says which *process* a consumer
runs in, while the class shape (`CaptureConsumer` vs direct
`CaptureSink`) accidentally determines *when* it sees data (request
finalize vs per-step chunks on the dispatch thread). Both are **async**
relative to the step loop — the model runner never waits on a consumer.

Phase 1 makes execution mode an explicit, declared axis:

```python
class CaptureConsumer:
    location: Literal["worker", "driver"] = "worker"
    execution: ClassVar[Literal["async", "sync"]] = "async"   # new
```

|  | data delivery | thread | actuation latency | data form |
| --- | --- | --- | --- | --- |
| `async` (today) | per-step chunks or finalize | dispatch/finalize thread | 1–3 steps via the action queue | CPU tensors, post-D2H |
| `sync` (new) | per-step, immediately post-forward | model-runner step thread | exactly 1 step | GPU views of persistent capture buffers |

Sync consumers implement one callback instead of the chunk/finalize
surface:

```python
def on_step(self, view: StepCaptureView) -> list[SteeringVectorUpdate] | None:
    # view.tensors:  {(layer, hook): GPU tensor [n_tokens, hidden]} —
    #                zero-copy views into CaptureManager._global_buffers
    # view.requests: rank-identical per-request metadata (derived from
    #                the broadcast scheduler_output + sampled ids):
    #                token spans into the buffer, prefill/decode phase,
    #                request id, and the step window's token ids
    ...
```

`StepCaptureView` v1 contents (decided): token spans + phase +
request id + the window's token ids — everything trivially
rank-identical. Token ids enable policies that react to emitted tokens
(trigger phrases) alongside activation probes. Sampling params and
richer metadata are deliberately excluded until a concrete policy needs
them; the one host-side field a policy has since needed — the client
`conversation_id` — rides the dedicated request-metadata channel (§5.6),
not `SamplingParams`.

Returned actions are validated and applied **inline** through the same
`apply_steering_updates` path Phase 0 built (we are already on the
single-mutator thread, so no queue hop) — one validation path, two
transports. The pure-return style is deliberate: it keeps sync
consumers testable and makes the rank-replication contract (§6)
auditable — state in, actions out.

Registration-time constraints (validated, not advisory):

- `sync` ⇒ `location="worker"`. A cross-process round-trip on the step
  thread is a non-starter.
- `sync` ⇒ global capture spec only. Client specs force eager and have
  variable per-request keys; sync rides the fixed-key persistent-buffer
  path exclusively, so it never affects the `CaptureStepGate` or
  cudagraph eligibility.
- `sync` consumers are constructed on **every** TP rank (today all
  consumers exist on rank 0 only). Rank 0 builds everything; other
  ranks build only sync consumers, with `CaptureManager` in a slim mode
  — persistent global buffers for the sync keys, no dispatch/finalize
  pipeline. This is the §6 rank-replication requirement expressed as
  framework wiring.

The critical-path contract: async consumers keep the framework's
never-on-the-critical-path guarantee; sync consumers are explicitly on
it, opt-in, with a per-consumer step-time metric and a documented soft
budget. The realistic pattern — probe GEMM on GPU, one tiny D2H of `k`
score floats, Python policy — adds approximately nothing, because the
step thread synchronizes for sampling immediately afterward anyway.
Heavy D2H or blocking I/O in `on_step` stalls serving; the metric makes
that visible and attributable.

Score computation happens inside `on_step` on the compute stream:
`scores = view.tensors[key] @ probes.T` (fixed-shape GEMM,
`n × k × hidden`; microseconds for k ≤ 32). Request-level reductions
(segment means over each request's span) are fine here — this is
off-graph, so variable shapes cost nothing.

The runner's hook point: sync consumers' `on_step` runs in
`execute_model` right after the forward (next to
`_finalize_capture_step`), so returned actions are in place before the
next step's `_update_steering_buffers` builds tables and index. The
monitor keys join the global-spec key set at `CaptureManager`
construction, so the persistent-buffer `copy_` is baked into graphs
exactly as today — sync consumers never touch the dispatch/chunk
pipeline (no D2H of activations, no thread hop).

### 5.2 Per-request actuation (Phase 1a — implemented)

The first actuation target (decided — it needs **no kernel changes**
and is the mode that stays meaningful under data parallelism and
multi-tenant serving).

**Implementation correction (supersedes the earlier sketch's §4.3
hash-swap sequence).** Mid-flight swapping of a request's *admitted*
config hash is unsafe, verified against the code: the scheduler builds
`scheduled_steering_configs` fresh from its own `Request` objects each
cycle with no worker feedback, so worker-side registrations of
unreserved hashes can exhaust rows the scheduler believes are free —
making `register_config` fail for a newly *admitted* request, which is
a contract violation. Hash swaps also desynchronize
`_req_steering_phase`, `steering_hash_to_request_ids`, and the
scheduler-side `Request.block_hash_decode_steering_config_hash`.

What is implemented instead — **dynamic-override rows**
(`vllm/v1/worker/steering_action_queue.py::RequestSteeringOverride`,
`steering_manager.py` dynamic pool, mixin routing):

- A dedicated row pool above the static pool: `SteeringConfig.
  max_dynamic_steering_configs` (default 4, `0` disables) extra table
  rows, sized centrally via `get_steering_buffer_config` (zero
  model-file edits). Dynamic registrations can never steal
  scheduler-reserved rows — the pools share nothing.
- `RequestSteeringOverride(req_id, vectors | None)` routes the
  request's decode tokens to a dynamic row populated as
  `global_decode_effective + override_vectors`. **Pure routing**: the
  admitted config's registration, refcounts, prefill→decode
  transition, and release-on-finish proceed exactly as if the override
  didn't exist. `vectors=None` clears (idempotent).
- Monotonic, never-reused `dyn_id`s keep ranks in lock-step (the
  register/release sequence is identical on every rank).
- Lifecycle: overrides are dropped automatically on request finish,
  preemption-resumption into prefill, and streaming re-add. Rejected
  actions (pool exhausted, request unknown/prefilling, bad vectors)
  keep previous state and are counted per source.
- Semantics: while active, the override *replaces* the request's
  admitted per-request decode delta for routing purposes.

Known limitation: scheduler-side steering-aware APC block hashes
(`Request.block_hash_decode_steering_config_hash`) never see overrides
— streaming-continuation cache keys reflect admitted steering only.
Fixing this needs a worker→scheduler notification; deferred.

In Phase 1a a gain change means re-registering the override's vectors
(an H2D + repopulate per change — fine at engagement-flip frequency,
wasteful for continuous modulation). The §5.3 scale tensor then makes
the common special case free: modulating an *existing* override's
strength becomes `set_row_scale` on its dynamic row.

### 5.3 Per-row scale tensor (Phase 1b) — IMPLEMENTED

Changing steering *strength* used to require re-uploading vectors
(`register_config` H2D + table repopulate). Now a per-row scale the
kernel multiplies the gathered vector by makes a strength change a cheap
scales-buffer write.

As built (differs slightly from the original sketch below): a **single
shared** `steering_scales` buffer per layer (shape `(max_configs + 3,)`,
fp32, default 1.0) — registered alongside `steering_index` in
`register_steering_buffers`, used by every hook's kernel call (the row
index is hook-independent). The kernel
(`steering_kernel.py::_apply_steering_kernel`) and the eager path load
`scale = scales[row]` and compute `out = hidden + table[row] * scale`;
`apply_steering`/`apply_steering_fake`/`apply_layer_steering` and the
warmup carry the new arg. The `SteeringManager` keys scales by a typed
*logical owner* — a frozen, totally-ordered `RowOwner`
(`vllm/v1/worker/steering_owner.py`): `RowOwner.global_(phase)`,
`RowOwner.config(hash, phase)`, `RowOwner.dyn(dyn_id)` — held in one
`_row_scales: dict[RowOwner, float]` so they survive row reuse. (The
legacy `_global_scales` / `_config_scales` / `_dynamic_scales` names are
now thin read-only views over `_row_scales` for the status payload.) All
owner-keyed runtime state (per-row scales **and** the per-row monitors of
§8.1) is dropped through a single `_purge_owner(owner)` at release:
`release_dynamic_config` and the **refcount-0 branch** of `release_config`
(a live→0 transition) both call it, so a scale/monitor set for a content
hash cannot silently re-apply to a future request that re-registers that
hash. A scale *pre-armed* for a not-yet-registered hash is untouched
(purge fires only on live→0). Scales are written in
`populate_steering_tables` (alongside the tables) plus a cheap
scales-only path `populate_steering_scales`. Populate scheduling is a
small `_DirtyState` (`content` / `membership` / `scales`) with the
implications encoded in code (membership ⇒ content; a full populate clears
scales too; the cheap path needs scales dirty and neither content nor
membership) — the whole point: a strength change costs no table recompose,
no vector H2D. The mixin calls the cheap path from
`_update_steering_buffers` when only scales are dirty. **Row 0 and all
prefill rows are pinned to 1.0** at populate (scaling them is meaningless
or cache-unsafe per §7). API: `SteeringManager.set_global_scale` /
`set_row_scale` / `set_dynamic_scale` / `clear_scales`; driven at runtime
by the `SteeringScaleUpdate` action (decode-only; `target` = global /
config_hash / dyn_id) through the same `_apply_steering_actions` path,
surfaced in `GET /v1/steering/dynamic` as `dynamic_scales`. Scale is
runtime state — never in `compute_hash` / `hash_steering_config`. Tests:
`tests/v1/worker/test_steering_scale.py`,
`tests/model_executor/layers/test_steering_op.py::TestPerRowScale`, plus
scale cases in the action-queue/override suites. Caveat (unchanged): a
per-row scale multiplies the row's *entire combined* content (global +
per-request), so it is cleanest on rows owned outright.

---

Original sketch (superseded by the shared-buffer design above): add, per
hook point, alongside each layer's table:

```
steering_scales_{hook}: float32[(max_configs + 3,)]   (persistent buffer)
```

and one extra load+multiply in the Triton kernel
(`steering_kernel.py`): `out = hidden + table[row] * scales[row]`
(plus the same in the CPU eager path). The kernel already loads
`row` per token; the scale load is one float per token. Persistent
buffer ⇒ graph-replay-visible, like the tables themselves.

`SteeringManager` grows `set_row_scale(config_hash, phase, scale)` and
`set_global_scale(phase, scale)` (rows 1/2), surfaced to consumers as
new action types flowing through the same apply path. Scale writes are
a single-element `copy_` — no vector H2D, no repopulate. This gives a
sync consumer a near-free "how much" knob over any row a dynamic
config owns.

Composition rule: scales compose multiplicatively *on top of* any
scale baked into the row at registration (`{"vector": ..., "scale":
...}` is pre-multiplied at `register_config`). Rows initialize to
`1.0`; the dynamic scale is a separate, multiplicative, runtime-owned
factor — never persisted into config hashes (it is not part of
steering *identity*, see §7).

Caveat: a per-row scale multiplies the row's **entire combined**
vector. Rows 3+ are populated as global + per-request sums, so scaling
a shared row scales both components. The scale knob is therefore only
semantically clean on rows whose content the dynamic config owns
outright — which per-request dynamic configs (§5.2) give us, and which
the dynamic tier (§5.4) gives the global case.

### 5.4 Dynamic additive tier (Phase 1b)

Decided: dynamic steering must compose with, not clobber,
operator-set steering (§11 Q7). Phase 0 overwrites the global decode
tier — wrong whenever an operator also uses `/v1/steering/set`.

A correction to an earlier note in this doc: this **cannot** be a
"reserved row" in the existing tables. `steering_index` maps each
token to exactly one row, so rows are exclusive — additivity across
rows does not exist at the kernel level (rows 3+ are *pre-combined* at
populate time instead). Two implementations, by phase:

- **Dedicated gather — IMPLEMENTED (replaces populate-folding).** The
  tier is a separate additive kernel term, not folded into rows:
  `out = hidden + table[row]*scale[row] + dynamic_vec * token_scales[t]`.
  `steering_dynamic_vec_{hook}` is a per-(layer, hook) `(hidden,)` buffer
  (the manager writes it from `dynamic_tier_vectors` in populate);
  `steering_token_scales` is a shared `(max_tokens,)` per-token gate
  (shared across layers in `_init_steering_state`, like `steering_index`)
  the runner writes each step — `dynamic_tier_gain` for decode tokens of
  a tier-active state, `0` for prefill / inactive (so the tier is
  strictly decode-only by construction, §7, with no row-routing needed).
  `update_dynamic_tier`/`clear_dynamic_tier`/`has_dynamic_tier` +
  `dynamic_tier_gain`/`set_dynamic_tier_gain` on the manager; the tier
  marks a (layer, hook) `any_active` so the kernel runs even with no row
  steering. Composition is unchanged vs populate-folding —
  `base + operator_decode + override + tier` — but decomposed: rows hold
  `base+operator+override`, the tier sits in `dynamic_vec`, and the
  kernel sums them per decode token. Wins over populate-folding: (a) the
  tier can be modulated **per token** (the Phase 2 substrate — Phase 2
  has an in-graph monitor write `token_scales` instead of the runner's
  flat gate), and (b) tier strength changes are **free** (next step's
  `token_scales` rebuild folds in `dynamic_tier_gain`; no buffer rewrite).
  Driven by `SteeringScaleUpdate(tier_gain=True)`; surfaced in
  `GET /v1/steering/dynamic` as `dynamic_tier: {active, gain, hooks}`.
  **Latent bug fixed here:** the runner's nothing-active short-circuit
  didn't check `has_dynamic_tier`, so a tier-only state (global-mode
  consumer, no operator/override steering) was wrongly skipped; now
  fixed. Tests: `tests/v1/worker/test_steering_dynamic_tier.py`,
  `test_steering_op.py::TestDynamicTierTerm`, token-gate + short-circuit
  tests in the override suite.

### 5.5 Configuration surface

- **No new plugin system.** Sync consumers register under the existing
  `vllm.capture_consumers` entry-point group and are configured via the
  existing `--capture-consumers name:key=value,...` / YAML / Python
  surfaces — `execution` is a class attribute, not config. The Phase 0
  plugin migrates by flipping `execution="sync"` and swapping its chunk
  plumbing for `on_step`; its `ProbePolicy` is already pure and carries
  over unchanged.
- **Probe/vector banks** use the `SteeringHookPacked` packed-JSON
  shape (decided) — the same wire format as `--steering-modules`,
  `/v1/steering/set`, and per-request steering, so the loaders already
  exist. The Phase 0 plugin's `torch.save` single-vector convenience
  path stays for one-probe prototyping.
- **Observability (decided):** read-only `GET /v1/steering/dynamic`
  reporting monitor sites, policy state (engaged/gain per site),
  queue/apply/reject counters, **and a per-consumer ring buffer of
  recent `(step, score, gain, action)` tuples** — closed-loop behavior
  is time-dependent, and post-hoc debugging without a trace is painful.
- Prometheus: applied/rejected/dropped update counters by source, plus
  per-sync-consumer `on_step` wall-time (the §5.1 budget metric —
  decided: metric + rate-limited warning only in v1, no automatic
  disable; a hard kill is itself a rank-divergence hazard unless its
  trigger is rank-replicated).

### 5.6 Request-level metadata channel (`RequestMetadata`)

Some policies need per-request host-side context that is *not* a sampling
parameter — e.g. the conversation-latch consumer correlates successive
requests of one conversation by an opaque client `conversation_id`. Such
fields live on `vllm.v1.request_metadata.RequestMetadata`, a small typed
`msgspec.Struct` carried on `EngineCoreRequest.request_metadata` alongside
`external_req_id` (the client request id), *not* on `SamplingParams`. It is
the extensible home for this class of field: `conversation_id` today,
declarative steering specs as siblings later. New fields keep a default so
older callers and serialized payloads stay valid.

Flow (mirrors the `client_request_id` precedent): the OpenAI entrypoint
builds it from the request (`ChatCompletionRequest.to_request_metadata()` /
`CompletionRequest.to_request_metadata()`) and passes it to
`generate(...)`; it threads through `AsyncLLM.add_request` →
`InputProcessor.process_inputs` → `EngineCoreRequest` →
`Request.request_metadata` → `NewRequestData.request_metadata`. Both runners
read `conversation_id` off it at admission and stash it for the per-step
view. Because it is pure host-side string metadata (no GPU work / D2H), it
is surfaced identically on the v1 and v2 runners via
`StepRequestView.conversation_id`.
### 5.7 Consumer-contract ABC and the controller base

The sync-consumer contract — `on_step`, `global_capture_spec`, the fixed
`location="worker"` / `execution="sync"` / `reads_client_spec=False`
metadata, and the `declared_graphsafe_keys(cls, params) -> list` classmethod
the capture registry calls at config-build time — is encoded as an `abc.ABC`,
`SyncCaptureConsumer` (`vllm/v1/capture/consumer.py`). It is a sibling of
`CaptureConsumer`, not a subclass: a sync consumer never implements
`on_capture`. `declared_graphsafe_keys` ships a `[]` default so a sync
consumer is never forced to override it (a missing one used to crash deep in
`resolve_graphsafe_shorthands` with a cryptic `AttributeError`, never in a
unit test); `on_step` and `global_capture_spec` are `@abstractmethod`, so an
incomplete consumer fails with a clear `TypeError` at construction instead.

`SteeringController` (`vllm/v1/capture/controller.py`) is a higher-level ABC
**on top of** `SyncCaptureConsumer` that owns the bookkeeping every dynamic
consumer otherwise re-writes by hand, leaving the subclass a single policy
method:

```python
def decide(self, request_view, residual) -> SteeringAction | None: ...
```

The base implements `on_step` in terms of `decide` and owns:

- **per-request lifecycle** — tracks the live request ids each step and
  prunes per-request `armed` state to the live set (state for finished /
  preempted requests is dropped automatically);
- **conversation scoping** — keys decisions on
  `StepRequestView.conversation_id`, skipping untagged and prefill rows, with
  a bounded (FIFO-evicted) `conversation_id -> sticky-override` map;
- **the latch pattern** — a `RequestSteeringOverride` returned by `decide`
  (the *trigger*) is latched onto the conversation and applied to the firing
  request; every later request of that conversation is *bridged* (re-issued
  the same override rebound to the new request, no re-trigger).

The base resolves the single monitored `(layer, hook)` from the subclass's
`global_capture_spec()` and hands `decide` the firing request's residual
window. `decide` reuses the existing action vocabulary
(`RequestSteeringOverride`, `SteeringVectorUpdate`, …) — no new actions.
`ConversationLatchExample` (the example plugin) is the proof: it subclasses
`SteeringController` and collapses to just the probe projection + threshold
decision in `decide`, with all latch/bridge/prune/scope plumbing inherited.

## 6. Distributed execution (the determinism problem)

This is the design's sharpest constraint, and where the earlier sketch
was unsound. Verified facts:

- Steering correctness across ranks relies on **lock-step state**: every
  rank processes identical `register_config` sequences so row IDs agree,
  with no hot-path collectives (`steering_runtime.md`).
- Capture consumers are constructed on **TP rank 0 only**
  (`gpu_model_runner.py:571-572`); other ranks run the capture cold
  path. So *any* consumer-originated steering mutation diverges TP
  ranks — there is no rank-replicated submitter to mirror it.

Resolution by phase:

- **Phase 0**: refuse multi-rank topologies outright (queue not
  installed; plugin won't construct). Honest and simple.
- **Phase 1**: make sync consumers **rank-replicated**, like
  `CaptureStepGate`: the residual at the steering hooks is replicated
  across TP ranks within a stage (read post-all-reduce), so every TP
  rank can run the same monitor GEMM + pure policy on identical inputs
  and reach identical decisions with **zero communication**. This
  requires (a) constructing sync consumers on every TP rank, with the
  slim `CaptureManager` mode of §5.1 allocating the monitor's
  persistent buffer and recording the `copy_` on each (today only
  rank 0 does — the buffer copy is collective-free, so extending it to
  all ranks is safe and costs one D2D copy per monitored key per step
  per rank); (b) the sync-consumer determinism contract: `on_step` and
  the consumer's internal state evolution are bit-deterministic pure
  functions of rank-identical inputs (no RNG, no wall clock, no
  cross-request iteration-order dependence). The pure
  state-in/actions-out shape of `on_step` makes this auditable; a debug
  mode that periodically cross-checks a hash of emitted actions across
  ranks is cheap insurance.
- **PP**: a stage can only monitor layers it owns. v1 restricts each
  sync consumer's monitor *and* steer sites to one stage (validated at
  registration). Cross-stage next-step decisions would need a sideband
  broadcast — deferred. Cross-stage *same-pass* (monitor stage k, steer
  stage k+1) is naturally forward-flowing and becomes available with
  Phase 2's per-token scales carried in `IntermediateTensors`, also
  deferred.
- **DP**: replicas are independent engines over disjoint requests; each
  runs its own consumers. Per-request actuation partitions naturally.
  *Global* actuation diverges replicas by design — document, and prefer
  per-request actuation under DP.
- **Spec decode**: draft model steering is already separate; dynamic
  updates between target steps can land between draft and verify.
  Phase 1 should pin update application to the target-model step
  boundary only (drain once per scheduler step, which the current drain
  point already guarantees).

### 6.1 Determinism-divergence detector (implemented)

The "cheap insurance" hash above is implemented as an always-on rolling
checksum. In `_apply_steering_actions`, each action that is *actually*
applied (rejected actions are excluded) is folded into a per-worker u64
checksum: a `zlib.crc32` over a compact, `PYTHONHASHSEED`-free digest of
the action's content (class, target `req_id`/`config_hash`/`dyn_id`,
`hook`/`layer`, `source`, and a bit-exact shape+CRC of any vector/probe
payload), mixed in application order with a per-drain-batch ordinal so
"same actions, different step" differs. Because actions are host-side
numpy built from rank-identical inputs, the digest is bit-exact rather
than a norm — strictly stronger and never legitimately divergent across
ranks. Cost is O(applied actions) and zero on idle steps.

`get_dynamic_steering_status` exposes `action_checksum` (hex) and
`action_count`; `GET /v1/steering/dynamic` compares them across workers
via `check_action_determinism`. Comparison is scoped **within each PP
stage** (grouped by `pp_rank`): TP ranks in a stage own identical layers
and must match, while PP stages own disjoint layers and may legitimately
differ. Sync-consumer-originated actions only exist at `pp == 1` anyway,
where this reduces to an all-workers comparison. A mismatch does not 500;
the response carries `determinism: {consistent: false, checksums: {...}}`
and a rate-limited server-side ERROR fires. Granularity is **poll-time**:
a desync is detected on the next status poll, not the step it occurs, so
the checksum bounds (does not prevent) corrupted output — pair it with
periodic polling for timely detection.

## 7. Steering identity, prefix caching, and phases

The steering runtime's correctness rules (see
[steering_runtime.md](steering_runtime.md)) that dynamic steering must
not violate:

- **Prefill steering is part of KV-cache identity.** Block hashes
  incorporate the prefill steering config; the HTTP set path calls
  `reset_prefix_cache(reset_running_requests=True)` after mutating
  global base/prefill vectors. The Phase 0 queue path performs no
  invalidation, hence: **decode tier only** (decode steering is
  explicitly excluded from cache keys). Any future phase that wants
  dynamic prefill steering must trigger the same invalidation — likely
  a non-starter for a high-frequency loop; treat prefill as out of
  scope for dynamic steering generally.
- **Decode-tier mutation and APC are compatible** by construction; no
  cache interaction.
- **Dynamic scale is runtime state, not identity.** The §5.3 scale
  tensor deliberately lives *outside* config hashes: two requests with
  the same vectors but different dynamic gains share a row whose scale
  is global... which is wrong for per-request gains. Resolution: row
  scales apply at row granularity — a per-request gain requires the
  request to own a distinct row (which per-request dynamic configs,
  §5.2, give it), and the *global* dynamic gain lives in the §5.4
  dynamic tier rather than on shared rows. Sharing a row across
  requests with *different* dynamic gains is impossible by
  construction, matching the existing row-per-config-hash model. For
  decode-only scale changes this is cache-safe; we must simply ensure
  scale is excluded from block-hash computation (it is, trivially — it
  never enters `hash_steering_config`).
- **Determinism / batch invariance**: a dynamic gain changes logits for
  *all* requests when applied to global rows. That is the feature's
  point, but it breaks per-request reproducibility. Mitigations:
  per-request actuation (Phase 1), and recording emitted updates
  (timestamped) so a run can be replayed.

## 8. Phase 2 — in-graph same-token conditional steering — IMPLEMENTED

The differentiated end state: detect at layer L, intervene at layers
> L, *same token*, full cudagraph speed (CAST-style conditional
steering).

As built (replaces the open design notes that follow):

- The §5.4 substrate already carries `steering_token_scales:
  float32[(max_tokens,)]`, shared across layers like `steering_index`,
  and the kernel adds `dynamic_vec * token_scales[pid]`. Phase 2 keeps
  exactly that gate buffer and kernel term — it only changes *who writes
  the gate*. v1 gates **only the dynamic tier** (the substrate term);
  gating the row gather (per-request configs) is the same mechanics and
  is left as future work (resolves the §11 open item).
- A **monitor custom op** (`torch.ops.vllm.steering_monitor`) registered
  at the probe site, called inside `apply_layer_steering` right after
  `maybe_capture_residual` and before the steer op. It computes per
  token `score = hidden @ probe`,
  `gate = sigmoid(sharpness * (score - threshold))`, and **multiplies**
  it into `steering_token_scales[:n]` in place (read-modify-write). `g`
  is a fixed elementwise policy with constant params; fixed shapes,
  collective-free, no allocation ⇒ recordable into the graph. Every TP
  rank records it — the residual is rank-identical post-all-reduce and
  the probe is replicated, so every rank computes the same gate (§3.1).
- **Reset discipline (resolved):** the runner overwrites
  `token_scales[:n]` fresh every step — `gain` for decode tokens, `0`
  for prefill, then zeros the tail. *That per-step overwrite is the
  reset.* The monitor read-modify-writes (multiplies) within the step,
  so there is no cross-step accumulation and no separate in-graph reset
  is needed. Decode-only cache-safety (§7) falls out of `0 * gate == 0`
  regardless of what the probe outputs, so a probe can never engage a
  prefill token.
- The monitor is the cheap per-token gate beneath the sync/async tier:
  the operator/consumer sets the tier vector (`dynamic_vec`) and the max
  strength (`dynamic_tier_gain`), and the in-graph monitor scales it per
  token in [0, 1] — completing the §3 hierarchy (async → sync →
  in-graph, each configuring the layer below). Policy parameters
  (threshold, sharpness, probe weights) live in small persistent buffers
  (`*_monitor_probe`, `*_monitor_params`, `*_monitor_active` per
  `(layer, hook)`), host-tunable between steps without recapture; the
  natural tuner is a sync consumer. The op is emitted at *every* steered
  hook (stable graph topology) but is a no-op unless its `active` flag is
  set, so only the configured site does work and the site can move at
  runtime without recapture.
- Constraints: per-token decisions only (no cross-token reductions
  in-graph); no data-dependent control flow beyond the tensor `active`
  flag; layers/hooks before the monitor site read the runner's flat gate
  (so dynamic-tier vectors are expected at sites ≥ the monitor — detect
  at L, steer at layers > L).
- Ordering caveat: hook execution order within a layer is pre_attn →
  post_attn → post_mlp; "later sites" includes later hooks of the same
  layer.

**Update (2026-06-21) — same-hook fusion default + opt-in cross-layer.**
The monitor now has two modes, selected by
`SteeringConfig.enable_cross_layer_monitor` (default `False`):

- **Same-hook (default):** the gate is **fused into `apply_steering`** —
  computed in-kernel from the pre-steering residual and folded into the
  tier/row terms in registers, never writing a shared buffer. The op stays
  **non-mutating** and the standalone `steering_monitor` op is *not* emitted.
  This gates only the probe's own `(layer, hook)`.
- **Cross-layer (opt-in):** `apply_layer_steering` emits the standalone
  *mutating* `steering_monitor` op at every steered hook (writing the shared
  `token_scales`/`row_gate` that later sites read — the "detect at L, gate at
  layers ≥ L" behaviour described above) and passes the always-False
  `steering_monitor_off` flag to `apply_steering` so the fused gate is bypassed
  (the per-token gate is applied exactly once, not twice at L). The opt-in is
  stamped onto each steerable layer as `module._cross_layer_monitor` in
  `_init_steering_state`, a torch.compile-time constant ⇒ stable graph topology
  per process; the flag is part of `SteeringConfig.compute_hash`.

The mutating op was originally suspected of a bs>16 cudagraph regression and
was fused away on that basis; nsys on the 60-layer gemma-4-31B (the reproducing
model) later showed the mutating monitor adds **zero** launches/syncs — 1 op or
60 ops are byte-identical to plain steering (the host-side cost is the 8-arg
`apply_steering` op itself, present with no monitor). So the cross-layer mutating
path is cudagraph-safe; it ships opt-in to keep the default path byte-identical.
See `steering-bench/docs/dynamic_steering_cudagraph_finding.md`.

### 8.1 Per-row (per-request) monitor — IMPLEMENTED

The global monitor above carries **one** probe per `(layer, hook)`: every
request at a site shares its probe/threshold, and `gate_rows` gates all
per-request rows uniformly. That blocks *per-request* same-step gating —
concurrent requests cannot be conditioned on different probes, and a request's
add vector cannot be gated independently of its neighbours'.

The **per-row monitor** (opt-in via `SteeringConfig.enable_row_monitor`) lifts
this: each steering table row carries its OWN probe + `[threshold, sharpness]`,
so the gate scaling a token's row term is computed from that token's row's
probe (`gate = sigmoid(sharpness*(residual@probe[row] - threshold[row]))`).
The steering ROW is already per-request (the override pool / static config
rows), so a per-request same-step `add` becomes in-graph: install the add
vector in the request's row, attach a per-row monitor to that row's owner, and
the kernel gates it per decode token — the added vector is per-request, only
the kernel machinery is shared. Decode-only (folded into `row_gate` via the
decode mask exactly like the global `gate_rows` path), so prefill rows stay
ungated and prefix-cache keys are untouched.

Design mirrors the §5.3 per-row scale and the §5.2 override pool:

- **Buffers** (per `(layer, hook)`, like the table): `*_monitor_probe_table`
  `(rows, hidden)` and `*_monitor_row_params` `(rows, 2)` = `[threshold,
  sharpness]`, plus a `*_monitor_row_active` flag. Default params
  `[-1e30, 1.0]` + zero probe ⇒ `sigmoid → 1.0` ⇒ unconfigured rows pass
  through with no branch. Per-`(layer, hook)` (not shared across layers) so a
  row configured at one site never imposes its threshold at another where the
  probe is zero. **Opt-in:** registered at a `(1, 1)` dummy size and resized
  to full only when `enable_row_monitor` (so the op signature is stable
  without touching model files; `resize_steering_row_monitor_buffers` runs once
  from `_init_steering_state`). The flag is in `compute_hash` (buffer shape is
  baked into captured graphs).
- **Manager** keys configs by the typed logical owner
  (`RowOwner.global_("decode")` / `RowOwner.config(hash, "decode")` /
  `RowOwner.dyn(dyn_id)`, `vllm/v1/worker/steering_owner.py`) so they survive
  row reassignment, and scatters them in row-position order with the same
  `indices` as the table write (`set_row_monitor`/`clear_row_monitor`/
  `has_row_monitor`/`_build_row_probe_and_params`). Per-row monitors are
  purged with the owner's other runtime state via the single `_purge_owner`
  at release (refcount-0 for configs; see §5.3), so a monitor never survives
  a hash's release to re-apply to a re-registration.
- **Action**: `SteeringMonitorUpdate` gains optional `req_id`/`config_hash`/
  `dyn_id` (at most one). None ⇒ the global monitor (unchanged); one set ⇒ a
  per-row monitor on that owner (`req_id` resolved to its live `dyn_id`).
- **APC**: a request's effective decode signature folds in its row's probe +
  params, so a temporal probe change re-keys the steered decode KV.
- **Kernel**: a second per-token reduction gathers `probe_table[row]` and
  multiplies its gate into `row_gate` (decode-only), guarded by a tensor
  `row_active` flag (graph topology stable). Orthogonal to the global monitor;
  both compose. Remaining shared resource: nothing — each row is independent;
  only the kernel reduction is per-site.

**Per-row wiring (additions):** `steering_kernel.py` (per-row reduction +
strides + warmup), `steering.py` (3 attr maps + dummy buffers +
`resize_steering_row_monitor_buffers` + 15-arg op + eager per-row block),
`steering_manager.py` (`_row_monitor` state + set/clear/has + populate +
signature fold), `steering_action_queue.py` (targeting + validation),
`steering_model_runner_mixin.py` + `gpu/steering_runner_mixin.py` (per-row
apply branch + short-circuit + transition deactivation + status),
`config/steering.py` (`enable_row_monitor`).

**Wiring:** `vllm/model_executor/layers/steering_monitor_kernel.py`
(Triton), `steering.py` (op + per-hook buffers + `apply_layer_steering`
call + warmup), `steering_manager.py` (`set_monitor`/`clear_monitor`/
`has_monitor` + populate writes the monitor buffers),
`steering_model_runner_mixin.py` (short-circuit + transition deactivation
+ `SteeringMonitorUpdate` dispatch + status + warmup),
`steering_action_queue.py` (`SteeringMonitorUpdate` +
`validate_steering_monitor`). The gemma4 taps are unchanged — the monitor
rides the existing `apply_layer_steering` call at every hook.

### 8.2 Declarative per-request gates — IMPLEMENTED

Everything above requires an operator to author and deploy a capture consumer.
Declarative gates let a **client** attach its own conditional steering to a
request — no server-registered consumer. A request carries a nested list of
gates in `RequestMetadata.steering` (§5.6), each a **`when × scope × apply`**:

- **when**: `always` | `probe` (`sigmoid(sharpness·(residual@probe −
  threshold))`).
- **scope**: `this_token` | `next_step` | `rest_of_request` |
  `rest_of_conversation`.
- **apply**: `add` (vector × strength, composed **on top of** the request's
  static decode steering) | `attenuate` (damp existing steering by a factor).

**Schema** (`vllm/v1/steering_schema.py`): msgspec tagged unions (`kind`
discriminator) so gates ride the `EngineCoreRequest` msgpack channel like
`conversation_id`. A vector source is `{"kind":"name","name":...}` (a
server-registered probe/steer vector) or `{"kind":"inline","packed":{hook:
SteeringHookPacked}}` (the base64 escape hatch, §5.6). **Name resolution is
worker-side** (reversing the earlier "the worker only ever sees packed bytes —
no worker-side registry" decision): `build_steering_gates` at the **frontend**
(`to_request_metadata`) only *validates* a `NamedVec`'s existence and passes it
through un-inflated, so the short name — not the full base64 blob — rides the
wire. `resolve_gates` then resolves every source to numpy once at admission —
`NamedVec` against the rank-replicated worker registry
(`vllm/v1/worker/steering_vector_registry.py`), `InlineVec` by unpacking — and
surfaces `ResolvedGate`s (carrying the resolved steer source's name + content
digest) on `StepRequestView.steering` (both runners). The reversal is
**required** by the persistence semantics below: a `rest_of_conversation` latch
must re-resolve its vectors *at bridge time* from server-resident state, which
means the worker has to own the name→vectors mapping. Admission wraps
`resolve_gates` in `resolve_gates_safe`: a malformed payload, **or a `NamedVec`
whose name is unknown to the worker** (a benign register/admission race), is
**gracefully skipped and logged once** — the request proceeds without
declarative steering rather than crashing the engine core (all TP ranks see the
same bytes and the same replicated registry, so the graceful path can't desync
them).

**Scope rule: `rest_of_conversation` + `add` requires a `NamedVec`.** Such a
gate is latched and bridged across later turns, so persisting the client's
inline bytes would pin them in server memory indefinitely. `build_steering_gates`
rejects an inline steer on that combination with an HTTP 400 (register the
vector and reference it by name, or use `rest_of_request`); ephemeral scopes
(`this_token`/`next_step`/`rest_of_request`) keep inline support unchanged, and
`attenuate` gates carry no vectors and are unaffected. A non-frontend producer
that still emits an inline `rest_of_conversation` add is skipped-and-warned by
the consumer (it cannot be latched by reference). Auto-registration of inline
payloads into a content-addressed store was considered and **deferred**.

**Built-in consumer** (`vllm/v1/capture/declarative.py`,
`DeclarativeSteeringConsumer`): subclasses `SteeringController` to reuse the
bounded conversation latch/bridge and `_armed` lifecycle but overrides
`on_step` (multi-gate, multi-scope). Auto-registered under the reserved name
`_declarative_steering` when steering is on, `enable_declarative_gates` is set
(default), and `pp==1` (sync-consumer constraint); enabling it also turns on
`enable_row_monitor` (§8.1). Gate → substrate:

- `add`: one `RequestSteeringOverride(compose_admitted=True)` per request (all
  `add` gates merge into one override row). `this_token+probe` also emits a
  per-request `SteeringMonitorUpdate(req_id=...)` so the gate re-evaluates
  **in-graph every decode token** (§8.1, free). Host-evaluated scopes
  (`next_step`/`rest_of_request`/`rest_of_conversation` + `probe`) evaluate the
  probe once on the CPU against the captured residual; `rest_of_conversation`
  latches **by reference** (a `ByRefLatch` of the gate's named steer vectors,
  §8.3) and bridges later turns by re-resolving those names at bridge time.
- `attenuate`: a per-request `SteeringScaleUpdate` (installing an admitted-only
  override first so the damp is per-request, not shared across a config row).
  **`attenuate` with `when=probe` and `scope=this_token` is unsupported** and
  rejected: same-token conditional damping would need a per-row gate of the form
  `scale = 1 − (1 − strength)·gate`, but the per-row monitor (§8.1) can only
  multiply a row's contribution *toward zero* when the probe is LOW — the wrong
  shape for "damp when the probe fires this token". Clients get an HTTP 400 from
  `build_steering_gates` (`_validate_gate_semantics`) telling them to use
  `next_step` / `rest_of_request` (host-evaluated) instead; if such a gate still
  reaches the worker from a non-frontend producer (offline, the Rust frontend)
  the consumer skips-and-warns it once rather than attenuating unconditionally.

**Probe sites & capture (`--declarative-probe-sites`).** The consumer's
`global_capture_spec()` is a configured allow-list of `layer:hook` sites (config
field `declarative_probe_sites`; CLI accepts both comma- and space-separated,
default a single site). It matters ONLY for **host-evaluated** probes
(`next_step`/`rest_of_request`/`rest_of_conversation` with `when=probe`), which
read the residual from `view.tensors[(layer, hook)]`; `this_token` probes are
computed in-kernel (per-row monitor, §8.1) and need **no** capture at any layer.
A host-probe gate naming a site outside the allow-list is **gracefully skipped
and logged once** — never a crash. Malformed sites (bad layer / unknown hook /
missing `:`) fail fast at startup with a clear error.

**Capture footprint.** Each captured site is a persistent buffer sized to the
full forward width, so the VRAM cost is
`num_sites × max_num_tokens × hidden_size × dtype_bytes`
(vLLM's `graphsafe_buffer_bytes` helper; the buffer covers the whole step's
residual, prefill included, not just decode rows). `max_num_tokens`
(`≈ max_num_batched_tokens`) is the dominant lever — it multiplies every site
equally, and the set is frozen at graph-capture time (a new site can't be added
under CUDA graphs without re-capture). Both runners log the total at startup:
`persistent capture buffers: N sites x T tokens x H hidden = X MiB VRAM`. Rule
of thumb (bf16): trivial for ≤1B models, single-digit GB for capturing *all*
layers of a 4–8B at large batch widths, prohibitive for 70B — so capture a
curated handful of layers (or lean on the zero-capture `this_token` path), not
everything.

**Precedence** (operator wins, `steering_model_runner_mixin.py` for v1 and
`gpu/steering_runner_mixin.py` for v2 — both `_apply_request_override`
implementations are kept in lock-step): every declarative action is stamped
`source="declarative"`; the runner records the owning source per request
(`_req_override_source`) and rejects a declarative action for a request already
owned by another (operator) source, and vice-versa the operator source takes
over a declarative-owned request. Compose-on-top is a runner-side fold
(`RequestSteeringOverride.compose_admitted` → `_resolve_request_steering(...,
"decode")` + the gate delta; guarded so a missing-module `RuntimeError` becomes
a structured rejection rather than crashing the engine). On request finish
`release_dynamic_config` purges the row's per-row monitor + scale so nothing
leaks.

**Fail-closed for `this_token+probe+add`.** The override (unconditional) is
emitted first and the per-row monitor (`req_id`-keyed) second — the ordering is
forced because the monitor's `req_id` resolves to the override's
freshly-registered `dyn_id`. `_apply_monitor_update` can still reject the monitor
after the override applied (row monitor disabled, probe hidden-size mismatch,
probe layer not steerable, non-finite/negative params), which would strand the
override applying **every** token unconditionally — the opposite of the client's
probe-gated intent. Two layers prevent this: the frontend validates
`sharpness`/`threshold` (finite, `sharpness ≥ 0`) so a well-formed request can
never hit rejection; and `_apply_steering_actions` **rolls the override back**
(applies an equivalent clear) when a declarative-source `SteeringMonitorUpdate`
targeting a `req_id` is rejected and an override for that same `req_id` was
applied earlier in the same action batch — scoped to same batch / same req_id /
declarative source, so operator flows are untouched. The request reverts to its
admitted (static) decode steering; the probe-gated add is dropped, not applied
blind.

**Named vector registry** (frontend mirror `vllm/entrypoints/openai/steering/
vector_registry.py` + admin routes `vllm/entrypoints/serve/steering/
vectors_router.py`, worker mirror `vllm/v1/worker/steering_vector_registry.py`):
`POST /v1/steering/vectors/register|unregister`, `GET /v1/steering/vectors`,
gated by dev mode (`VLLM_SERVER_DEV_MODE`). Each register/unregister is
**broadcast to every worker** via `engine.collective_rpc`
(`register_steering_vector_name` / `unregister_steering_vector_name`, mirroring
`/v1/steering/set`'s rank-replicated flow), so a `NamedVec` gate resolves
worker-side; the frontend copy stays as the validating mirror (existence checks
+ listing). Both sides store a sha256 content digest
(`steering_vector_content_digest`) over the canonical packed serialization, so
latch-by-reference digests match across the worker boundary. Registration stays
dev-mode gated and, unlike the module registry / `/v1/steering/set`, is **not**
behind the steering API key — but note the registry is now *load-bearing* (the
only path to a `rest_of_conversation` latch), so the auth deferral is a
deliberate, revisitable choice (§8.3). Distinct from the module registry (§5.7)
— single named probe/steer vectors.

### 8.3 Trust model and multi-tenancy

The dynamic-steering stack assumes a **single-tenant / trusted-client**
deployment: every client that can reach the engine is trusted not to interfere
with another client's steering state. It ships **no authentication or
per-client isolation** of its own. The consequences and the operator's
responsibilities:

- **`conversation_id` is a global, client-chosen, unauthenticated namespace.**
  It rides the request-metadata channel (§5.6) and keys the controller's latch
  map (§5.7). There is no ownership check: any client that presents a given
  `conversation_id` bridges (inherits) whatever steering is latched on it, and
  any client can pre-latch steering onto an id another client will later use.
  A guessed or reused id therefore lets one client steer — or read the steered
  behavior of — another client's turns.

  **Requirement for shared deployments.** In any multi-client deployment the
  **operator (or a gateway sitting in front of vLLM) must namespace
  `conversation_id`s per client** — e.g. prefix each id with an authenticated
  client/tenant identifier — so ids from different clients can never collide.
  vLLM does not and cannot do this itself: the id is opaque to it, and it has
  no notion of client identity. Without per-client namespacing, treat the
  latch as shared mutable state visible to every client.

- **Latch bounds (memory + churn).** The latch map is bounded on two axes,
  both enforced by the controller base (§5.7, `SteeringController`): an **entry
  count** (`max_conversations`, default 1024) and an **aggregate payload-byte**
  bound (`max_latched_bytes`, default 256 MiB). A latched entry is a tagged
  union: an operator-authored `RequestSteeringOverride` (raw vectors, **byte
  accounted**) or a declarative **`ByRefLatch`** (name references + digests,
  ~0 bytes — the vectors live in the worker registry, so the byte cap does not
  apply to it). The byte bound exists because a raw-vector latch pins full
  steering vectors (all hooks x layers, float32) on **every TP rank**; by
  latching *by reference*, the declarative path (the one clients drive) retires
  that host-memory pressure entirely — a client can no longer inflate the
  server's latch footprint with its own bytes, because a `rest_of_conversation`
  add must name a server-resident vector. Eviction is now **LRU** (a bridge
  refreshes a conversation's recency via `move_to_end`; the least-recently-used
  entry is dropped until both caps fit); a single raw-vector override exceeding
  the byte cap alone is still refused (the triggering request steers that turn;
  only cross-turn persistence drops). Eviction is a pure function of the
  latch/bridge sequence (no time-based logic), so it stays rank-deterministic.
  LRU narrows — but does not eliminate — the churn caveat: a client that floods
  fresh `conversation_id`s (while *bridging* none) can still evict idle
  latches, though actively-bridged conversations now survive unrelated churn.
  This is a denial-of-persistence surface, not a correctness bug — another
  reason shared deployments should gate/namespace ids at the edge.

- **Digest-guarded bridging.** A `ByRefLatch` stores, per referenced name, the
  content digest observed when the latch was installed. Bridging a later turn
  re-resolves the name from the worker registry and **verifies the digest**: a
  name that was unregistered, or re-registered with *different* content
  mid-conversation, fails the check and the latch **disengages** (drops, warns
  once) rather than silently steering the conversation with changed content. So
  the shared-registry integrity caveat below cannot corrupt a live
  conversation's steering — the worst case is a clean disengage, never a
  substituted vector.

- **Named-vector registry is a shared, load-bearing mutable namespace.** The
  registry (§8.2, frontend + worker mirrors) keys vectors by a global name that
  any request may resolve against, and `register`/`unregister` **silently
  overwrite/delete** an existing name. A client can thus repoint or drop a name
  other clients depend on; digest-guarded bridging turns that into a disengage
  (above), and admission of a fresh request naming a dropped vector is a
  graceful skip (§8.2). The registry is now **load-bearing** — it is the *only*
  way to express a `rest_of_conversation` latch (inline is refused) — yet it is
  still deliberately **not behind the steering API key**: for the ephemeral
  scopes a named vector remains capability-equivalent to the already-open inline
  path, and adding auth is a follow-up (the latch is by reference, so an
  unauthenticated register no longer *also* buys unbounded server memory). The
  integrity caveat (shared, last-writer-wins names) stands; the same per-client
  namespacing / edge-gating guidance applies if names must be isolated.

None of the above is behind authentication by design; hardening for a genuine
multi-tenant deployment belongs at the operator/gateway layer in front of
vLLM, not in this stack. Auth on the registry, a client-visible
latched/bridged/evicted signal, and auto-registration of inline payloads
(content-addressed store) are all deferred follow-ups.

## 9. Test plan

- **Phase 0 (on this branch)**: unit tests for queue mechanics, drain
  validation/application isolation, and the full plugin policy state
  machine + sink lifecycle, plus real-`SteeringManager` end-to-end
  (drain → populate → table rows, zero-vector disengage, composition
  into per-request rows)
  (`tests/v1/worker/test_steering_action_queue.py`,
  `examples/capture_consumers/dynamic_steering_controller/test.py`;
  46 tests, CPU-only). Existing steering/capture suites unaffected
  (339 passed).
- **Phase 0 still missing**: an engine-level integration test — tiny
  fixture decoder, stub consumer submits an update after step 0, assert
  step ≥ 1 logits shift in the steering vector's direction (pattern:
  `tests/models/language/generation/test_steering.py` fixture tests).
  And a GPU smoke run (gemma-3-4b on a 3090 node) — see the validation
  recipe in the plugin README.
- **Phase 1a (on this branch, CPU)**: registration-time validation for
  the `execution` axis (sync ⇒ worker / no client spec / global spec /
  pp=1); slim-manager behavior; step-view construction
  (spans/phase/token ids/zero-copy); the shared apply path; dynamic
  pool mechanics + populate composition + indices-cache cycles;
  override apply/validate matrix incl. pool-exhaustion-keeps-prior;
  steering-index routing with admitted state untouched; cleanup hooks
  + leak test; budget metric + bounded ring; status-RPC picklability;
  plugin policy/actuation/packed banks
  (`tests/v1/capture/test_sync_consumers.py`,
  `tests/v1/worker/test_sync_steering_integration.py`,
  `tests/v1/worker/test_steering_dynamic_override.py`, plugin
  `test.py`).
- **Phase 1a GPU-validated (tp=1, 2026-06-11)**: gemma4-31B Q4_K_S
  GGUF on an RTX 3090, real serving config (4096 ctx, 32 seqs, 0.95
  util, chunked prefill). Verified: cudagraphs capture/replay normally
  with the sync consumer active (no eager forcing); shadow mode
  (threshold=-1, gain=0) shows per-request engagement with zero
  emissions and baseline outputs; active mode (gain=6) emits exactly
  one override per request, visibly changes greedy outputs starting
  one step after engagement (the expected 1-step latency: "…is
  Paris." then steered tokens), `/v1/steering/dynamic` shows
  `dynamic_pool.in_use=1` + the request→dyn_id mapping mid-decode and
  a clean drain to 0 on finish, `applied`/`rejected` counters correct.
  **Finding (resolved)**: the original metric reported `on_step` wall
  time (~30 ms ≈ the model's decode step time), because the consumer's
  scores D2H is the step's *first* CUDA sync point and the
  `perf_counter` span absorbs the forward pass's GPU drain rather than
  measuring added cost. Fixed by adding CUDA-event timing: a per-consumer
  event pair brackets `on_step`, and because both events sit behind the
  forward in the stream queue, `start.elapsed_time(end)` measures only
  the consumer's own enqueued GPU work (GEMV + D2H), excluding the drain.
  The pair is read one step late (the prior step's events are guaranteed
  complete), so the read never blocks and never forces a sync onto the
  critical path; the GPU reading lags wall time by exactly one step. The
  endpoint now reports both: `total_ms`/`max_ms` (wall, diagnostic) and
  `gpu_total_ms`/`gpu_avg_ms`/`gpu_max_ms` (the honest added cost). The
  budget check (`sync_budget_ms`) charges the GPU reading, not the
  drain-inflated wall time, removing the spurious over-budget warnings.
- **Event metric GPU-validated (tp=1, 2026-06-13, node2)**: gemma4-31B
  Q4_K_S on an RTX 3090, per-request actuation, threshold=-1/gain=6 so
  every request engages (maximal consumer load), pool=32. Over a 120-step
  run the endpoint reported wall ~31 ms/step (= the model's decode step
  time, the old misleading figure) but `gpu_avg_ms=1.04`,
  `gpu_last_ms=0.054`, `gpu_steps=119` (one fewer than 120 — the
  one-step deferral). The 1.04 ms average is dominated by a single
  ~117 ms first-step outlier (lazy cuBLAS / probe H2D init); steady-state
  decode added cost is ~0.05 ms/step (~0.16% of a 31 ms step), and
  `over_budget_steps` was 1 (the warmup outlier) rather than ~120 as the
  wall-based check would have flagged. A throughput A/B corroborated:
  16 concurrent × 256 tokens gave 96.67 tok/s baseline (no consumer) vs.
  95.92 tok/s with the consumer active — a 0.78% difference, within
  run-to-run noise and consistent with the event metric (the old 31 ms
  wall figure would have implied a ~2× slowdown that plainly does not
  occur). cudagraphs captured normally (PIECEWISE + FULL decode) with the
  consumer active. Note: this node needs `VLLM_USE_FLASHINFER_SAMPLER=0`
  (its CUDA/CUB toolchain fails the flashinfer sampling-kernel JIT —
  unrelated to steering).
- **Sync-consumer warmup hook (GPU-validated, node2)**: the ~117 ms
  first-`on_step` outlier above is a one-time cuBLAS/probe-H2D init paid
  on the critical path. `GPUModelRunner._warmup_sync_consumers()` (called
  at the end of `capture_model`, full CUDA context) gives each sync
  consumer an optional `warmup(device, dtype)` to pre-pay it; the example
  controller runs one GEMV matching `on_step`. After the hook the same
  120-step run reported `gpu_max_ms=0.098` (was 117.22), `gpu_avg_ms=0.059`
  (was 1.04 — now the true steady-state, no longer skewed by the outlier),
  and `over_budget_steps=0` (was 1). Skipped under `enforce_eager` (no
  `capture_model`); the one-time cost is then paid on the first step.
- **Engine-level e2e test (GPU-validated, node2)**:
  `tests/v1/worker/test_dynamic_steering_e2e.py` drives a real `LLM` with
  a config-driven sync stub (`DeterministicOverrideStub`, a second entry
  point in the example package) that steers exactly one of two identical
  concurrent requests on its first decode step. Asserted *within the
  steered run* (target vs. in-batch control) to dodge batched-FP
  nondeterminism — two identical greedy prompts are NOT bitwise identical
  deep in generation (they diverge from FP noise at ~token 22 on
  gemma4-31B), so a cross-run "identical baseline" assumption is invalid.
  Result: `first_diff=2` — token 0 (prefill) and token 1 (the emit step,
  override not yet applied) match; divergence begins at token 2. This
  proves, end to end, exactly-one-step actuation latency *and* per-request
  targeting, cleanly separated from the FP-noise floor. Skip-marked
  (needs CUDA + a tapped gemma4); model via `DYNSTEER_E2E_MODEL`
  (defaults to the gated tiny HF gemma4 with dummy weights; point at a
  local GGUF to run offline). Forces `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- **Phase 1a still missing**: tp=2 rank-replication smoke (identical
  tables and emitted actions across ranks — needs a 2-GPU node).
- **Phase 1b**: kernel-level tests for the scale tensor and dynamic
  tier (row-scale/baked-scale composition, dynamic-tier additivity
  with operator-set vectors, any_active interaction, zero-gain
  equivalence with steering disabled).
- **Phase 2 (CPU)**: the monitor op math + CPU eager (gate =
  `sigmoid(sharpness*(h@probe - thr))`, in-place multiply on `[:n]`,
  inactive no-op, prefill-zero preserved, composition into the tier);
  manager `set/clear/has_monitor` + populate into the per-layer
  monitor buffers + unconfigured-site deactivation; the runner
  short-circuit/transition (a monitor-only state defeats the
  nothing-active short circuit; clearing deactivates on the transition);
  `SteeringMonitorUpdate` validation + dispatch through the shared apply
  path (`tests/model_executor/layers/test_steering_monitor_op.py`,
  `tests/v1/worker/test_steering_monitor.py`, additions to
  `test_steering_dynamic_override.py` and `test_steering_action_queue.py`).
- **Phase 2 GPU-validated (tp=1, 2026-06-16, node2)**: gemma4-31B
  Q4_K_S on an RTX 3090. (a) **Standalone**: the real Triton monitor
  kernel matches the fp32 eager reference across batch sizes 1–256 in
  bf16 (max|Δ| ≤ 4e-5) and fp32, with prefill-zero preserved; a
  **hand-built CUDA graph** capturing `steering_monitor` →
  `apply_steering` and replayed across three steps with different
  runner gains (6/2/9) reproduces eager (rel ≤ 5e-3) — proving the
  monitor's in-place `token_scales` mutation is visible to the later
  steer op *within the same graph*, that the runner's per-step
  overwrite is the reset (no cross-step accumulation), and that a
  0-gate (prefill) token stays tier-free; inactive monitor is a no-op.
  (b) **Engine**: booting with `enforce_eager=False` captured CUDA
  graphs normally with the monitor op emitted at every steered hook
  (PIECEWISE 11/11 + FULL 7/7, 7 s, 0.28 GiB) and produced coherent
  greedy output (no probe configured ⇒ the always-emitted monitor is a
  true no-op). The monitor kernel warmup retired before capture
  (`shapes=11`, ~4 ms). Needs `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **Engine-level e2e for row-gate / req_id-scale / async transport
  (GPU-validated, tp=1, 2026-06-17, node2)**: gemma4-31B Q4_K_S, layer
  30, `enforce_eager=True`. Three skip-marked tests, each driving a real
  `LLM` + a config-driven consumer; the two per-request paths use the
  within-run target-vs-control technique (robust to the batched-FP noise
  floor `NOISE_FLOOR=10`).
  - **Row gating** (`test_steering_gating_e2e.py::test_row_gate_*`,
    `ConfigurableOverrideStub` mode `rowgate`): an override row plus a
    `gate_rows=True` monitor whose threshold is saturated (±1e6) to force
    the per-token gate fully on/off. Gate ON ⇒ the target's per-request
    row is applied (early divergence in `[1,10]`); gate OFF ⇒ the row is
    suppressed (target tracks the control past the noise floor). Proves
    the in-graph monitor gates the **per-request row term**, not just the
    §5.4 tier, end to end.
  - **req_id scale** (`..::test_req_id_scale_*`, mode `reqscale`): an
    override row plus `SteeringScaleUpdate(req_id=, scale=0)` emitted in
    the same step (override first so the runner resolves the fresh
    `req_id → dyn_id`). `scale=0` suppresses exactly the target's row
    (≈control past the floor); the unscaled override diverges early.
    Proves the cheap per-request strength knob routes correctly.
  - **Async transport** (`test_async_steering_e2e.py`, `AsyncTierExample`):
    a global-tier `SteeringVectorUpdate` submitted through the action
    queue from `on_capture`. **Finding**: `on_capture` runs at request
    *finalize*, so the update never steers its own request — it steers a
    *subsequent* one. Modelled by repeating the prompt in one engine:
    `gen[0]` is the baseline; `gen[1..]` are steered (the strong tier
    collapses them to a repeat), proving the queue → drain →
    `_apply_steering_actions` path delivers across requests.

## 10. Relationship to the earlier sketch

Kept: the queue + step-thread drain point (its §4.2 reasoning is
correct and is exactly what Phase 0 implements); the per-request
mutation sequence via config hashes (§4.3 → our §5.2); non-throwing
observer isolation (§4.5); HTTP-path coexistence strategy (§4.6 option
2 — the drain point and the HTTP RPC both mutate manager state on the
step thread / under the engine's existing serialization, so no new
locking).

Changed:

- **Determinism**: the sketch assumed updates originate driver-side and
  broadcast via RPC; worker-location consumers (the low-latency case)
  exist only on TP rank 0, so we gate Phase 0 to single-rank and make
  Phase 1 rank-replicated instead (§6).
- **Injection**: the sketch threaded a controller handle through
  `registry.build_consumer` (`wants_steering_controller`); Phase 0 uses
  the process-global slot pattern the codebase already uses for the
  capture manager — no registry signature change, and it composes with
  direct `CaptureSink` consumers, which the batched-adapter-based
  injection did not. Phase 1 goes further: instead of handing async
  consumers a steering handle, it adds the `execution` axis so fast
  policies become *sync consumers* — the sketch's controller object
  disappears into the consumer framework entirely.
- **New**: the sync/async execution axis (§5.1), scale-tensor and
  dynamic-tier primitives (§5.3–5.4), in-graph Phase 2 (§8), and the
  decode-only/prefix-cache analysis (§7).

## 11. Decisions and open items

All headline questions were settled on 2026-06-11:

1. **Sync vs separate controller** → fast policies are **sync
   consumers** (`execution` axis, §5.1); the async consumer loop
   survives as the home for slow policies (driver-side, learned,
   I/O-bound), sharing the action queue and validation path.
2. **Actuation order** → **per-request first** (Phase 1a, §5.2): no
   kernel changes, stays meaningful under DP and multi-tenant serving.
   Gain primitives (scale tensor + dynamic tier) follow in Phase 1b.
3. **Composition with operator-set steering** → **dynamic additive
   tier** (§5.4), not overwriting the global decode tier. Note the
   correction recorded there: this is a separate per-layer
   vector+gain, not a "reserved row" — rows are exclusive via
   `steering_index`, so cross-row additivity does not exist.
4. **Sync budget** → **metric + rate-limited warning only** in v1; no
   automatic disable (a hard kill is a rank-divergence hazard unless
   its trigger is rank-replicated, e.g. step-counted — revisit after
   real `on_step` timings exist).
5. **`StepCaptureView` contents** → **minimal + token ids**: spans,
   phase, request id, and the step window's token ids (all
   rank-identical); enables token-trigger policies alongside probes.
6. **Probe/vector bank format** → **`SteeringHookPacked`** packed JSON
   everywhere; `torch.save` 1-D tensors remain as a single-probe
   prototyping convenience.
7. **Observability** → **ring buffer + counters** behind
   `GET /v1/steering/dynamic` (§5.5).
8. **Policy expressiveness across tiers** (settled 2026-06-13, §3.1):
   probe / decision-function / controller-state are separate concerns.
   A pretrained probe is usable on every tier (in-graph too, if its
   per-token evaluation is graph-safe). Learned/stateful controllers
   live on the sync or async tier — never in-graph — and tune a cheap
   fixed per-token gate beneath them; the in-graph tier is bounded to
   fixed-shape, collective-free, allocation-free, per-token ops with no
   host sync or data-dependent control flow. All worker-side tiers must
   be pure functions of the rank-identical post-all-reduce residual.

Still open (non-blocking):

- **`model_runner_v2`**: steering integration there is pending
  (dev-flag-gated upstream); dynamic steering inherits whatever lands.
  Keep the sync `on_step` hook behind the same mixin seam so it ports.
- ~~**Phase 2 in-graph details** (§8): token-scale reset discipline,
  and whether per-request rows also need token gating.~~ **Resolved
  (implemented, §8):** the runner's per-step overwrite of `token_scales`
  is the reset; the monitor op read-modify-writes (multiplies) within the
  step. v1 gates only the dynamic tier; per-request-row token gating is
  deferred (same mechanics, additive when needed).

## 12. References

- `vllm/v1/worker/steering_action_queue.py` — Phase 0 bridge (this branch)
- `examples/capture_consumers/dynamic_steering_controller/` — Phase 0 plugin (this branch)
- `vllm/v1/worker/steering_manager.py`, `steering_model_runner_mixin.py` — steering runtime
- `vllm/v1/capture/manager.py` (`on_hook`, `_global_buffers`), `step_gate.py` — capture runtime
- `vllm/model_executor/layers/steering.py`, `steering_kernel.py` — hook + kernel
- [steering_runtime.md](steering_runtime.md), [capture_consumers.md](capture_consumers.md), [capture_parallelism.md](capture_parallelism.md)
- docs/features/steering.md §Prefix Caching — cache-identity rules
