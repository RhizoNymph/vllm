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

### 5.3 Per-row scale tensor (Phase 1b)

Today changing steering *strength* requires re-uploading vectors
(`register_config` H2D + table repopulate). Add, per hook point,
alongside each layer's table:

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

- **Populate-folding (no kernel change, available any time):** treat
  the dynamic tier as a fourth vector source folded into
  `populate_steering_tables` composition (rows 1/2 and 3+ all gain
  `+ dynamic_vec * gain`). Gain changes mark `_tables_dirty` and cost
  one repopulate — same price as today's global updates.
- **Dedicated gather (Phase 1b, preferred):** per layer/hook, a single
  dynamic vector buffer + scalar gain read unconditionally by the
  kernel: `out += dynamic_vec * dynamic_gain`. Gain changes are a
  single-element `copy_`; vector changes are one row's H2D. This is
  exactly the substrate Phase 2 extends — replace the scalar
  `dynamic_gain` with the per-token `steering_token_scales` and the
  in-graph monitor writes it. Q7 and Phase 2 share machinery.

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

## 8. Phase 2 — in-graph same-token conditional steering

The differentiated end state: detect at layer L, intervene at layers
> L, *same token*, full cudagraph speed (CAST-style conditional
steering).

- Add `steering_token_scales: float32[(max_tokens,)]`, shared across
  layers like `steering_index`. The natural CAST shape gates the §5.4
  dynamic tier per token — `out += dynamic_vec * token_scales[pid]` —
  i.e. Phase 2 replaces that tier's scalar gain with a per-token one;
  gating the row gather (`table[row] * scales_row[row] *
  token_scales[pid]`) is the same mechanics if per-request configs
  should also be token-gated. All loads are fixed-shape against
  persistent buffers — graph-safe.
- A **monitor custom op** registered at the probe site (inside
  `apply_layer_steering`, after `maybe_capture_residual`):
  `token_scales[:n] = g(hidden @ probe)` with `g` a fixed elementwise
  policy (sigmoid/step with constant parameters). Fixed shapes,
  collective-free, no allocation ⇒ recordable into the graph. Every TP
  rank records it (inputs replicated ⇒ outputs identical).
- Constraints: per-token decisions only (no cross-token reductions
  in-graph); policy parameters (threshold, gain, probe weights) live in
  small persistent buffers so they remain host-tunable between steps
  without recapture — the natural tuner is a sync consumer, completing
  the hierarchy of §3 (async → sync → in-graph, each configuring the
  layer below); layers ≤ L are unaffected (scales default 1.0 and are
  reset off-graph each step... note: reset must also be in-graph or the
  buffer must be written *by* the monitor op for all n each step —
  design detail to settle in implementation).
- Ordering caveat: hook execution order within a layer is pre_attn →
  post_attn → post_mlp; "later sites" includes later hooks of the same
  layer.

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
- **Phase 1a still missing**: engine-level fixture test (sync stub
  consumer emits an override after step 0, assert step ≥1 logits shift
  for the targeted request only); tp=2 rank-replication smoke
  (identical tables and emitted actions across ranks — needs a
  2-GPU node).
- **Phase 1b**: kernel-level tests for the scale tensor and dynamic
  tier (row-scale/baked-scale composition, dynamic-tier additivity
  with operator-set vectors, any_active interaction, zero-gain
  equivalence with steering disabled).
- **Phase 2**: graph-capture test asserting monitor op + scale reads
  replay correctly across batch sizes; eager-vs-graph equivalence on
  sums (pattern: the global-spec capture validation in
  `capture-global-spec-cudagraph`).

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

Still open (non-blocking):

- **`model_runner_v2`**: steering integration there is pending
  (dev-flag-gated upstream); dynamic steering inherits whatever lands.
  Keep the sync `on_step` hook behind the same mixin seam so it ports.
- **Phase 2 in-graph details** (§8): token-scale reset discipline
  (in-graph reset vs full overwrite by the monitor op), and whether
  per-request rows also need token gating or only the dynamic tier.

## 12. References

- `vllm/v1/worker/steering_action_queue.py` — Phase 0 bridge (this branch)
- `examples/capture_consumers/dynamic_steering_controller/` — Phase 0 plugin (this branch)
- `vllm/v1/worker/steering_manager.py`, `steering_model_runner_mixin.py` — steering runtime
- `vllm/v1/capture/manager.py` (`on_hook`, `_global_buffers`), `step_gate.py` — capture runtime
- `vllm/model_executor/layers/steering.py`, `steering_kernel.py` — hook + kernel
- [steering_runtime.md](steering_runtime.md), [capture_consumers.md](capture_consumers.md), [capture_parallelism.md](capture_parallelism.md)
- docs/features/steering.md §Prefix Caching — cache-identity rules
