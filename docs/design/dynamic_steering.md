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
them.

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
warmup carry the new arg. The `SteeringManager` keys scales by *logical
owner* (`_global_scales[phase]`, `_config_scales[(hash, phase)]`,
`_dynamic_scales[dyn_id]`) so they survive row reuse, and writes them in
`populate_steering_tables` (alongside the tables) plus a cheap
scales-only path `populate_steering_scales` gated by `_scales_dirty`
(separate from `_tables_dirty` — the whole point: a strength change costs
no table recompose, no vector H2D). The mixin calls the cheap path from
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

**Wiring:** `vllm/model_executor/layers/steering_monitor_kernel.py`
(Triton), `steering.py` (op + per-hook buffers + `apply_layer_steering`
call + warmup), `steering_manager.py` (`set_monitor`/`clear_monitor`/
`has_monitor` + populate writes the monitor buffers),
`steering_model_runner_mixin.py` (short-circuit + transition deactivation
+ `SteeringMonitorUpdate` dispatch + status + warmup),
`steering_action_queue.py` (`SteeringMonitorUpdate` +
`validate_steering_monitor`). The gemma4 taps are unchanged — the monitor
rides the existing `apply_layer_steering` call at every hook.

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
