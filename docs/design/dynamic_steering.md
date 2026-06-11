# Dynamic Steering — Design

Status: Phase 0 implemented (this branch); Phases 1–2 proposed
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
| Client-spec capture | dynamic `index_select` (`manager.py:826-836`), forces eager via `CaptureStepGate` | why the controller must *not* ride the client-spec path |
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
  1. **Scale a row** (Phase 1 primitive, no H2D vector traffic),
  2. **Gate a token** (Phase 2 in-graph per-token scale),
  3. **Rewrite vectors** (existing `update_global_vectors` /
     `register_config` machinery; one table repopulate).

Three loop latencies fall out:

| Loop | Path | Latency | Phase |
| --- | --- | --- | --- |
| Consumer loop | capture dispatch thread → action queue → next `_update_steering_buffers` | 1–3 decode steps | **0 (implemented)** |
| Step-thread controller | read persistent capture buffers post-forward on the step thread, GPU GEMM scores, actuate before next step | exactly 1 step | 1 |
| In-graph conditional | monitor op at layer L writes per-token scales consumed by steering at layers > L, same forward | 0 (same token) | 2 |

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

## 5. Phase 1 — step-thread controller + scale primitives

### 5.1 Per-row scale tensor (the key new primitive)

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
`set_global_scale(phase, scale)` (rows 1/2). Scale writes are a single
element `copy_` — no vector H2D, no repopulate. This gives the
controller a near-free "how much" knob over both global tiers and any
registered per-request config.

Interaction to settle: scales compose multiplicatively *on top of* any
scale baked into the row at registration (`{"vector": ..., "scale":
...}` is pre-multiplied at `register_config`). Proposal: rows
initialize to `1.0`; the dynamic scale is a separate, multiplicative,
runtime-owned factor — never persisted into config hashes (it is not
part of steering *identity*, see §7).

### 5.2 `SteeringController` on the step thread

A controller object owned by the model runner (sibling of
`SteeringManager` / `CaptureManager`):

- **Monitor registration**: controller `(layer, hook)` keys join the
  global-spec key set at `CaptureManager` construction, so the
  persistent-buffer `copy_` is baked into graphs the same way. The
  controller reads `_global_buffers[key]` directly post-forward — it
  does *not* ride the dispatch/chunk pipeline (no D2H, no thread hop).
- **Score computation**: `scores = buf[:n] @ probes.T` on the compute
  stream right after forward (fixed-shape GEMM, `n × k × hidden`;
  microseconds for k ≤ 32). Request-level reductions (segment means
  over each request's token span) happen here too — off-graph, so
  variable shapes are fine.
- **Policy + actuation**: runs in `execute_model` after the forward (or
  equivalently at the top of the next step before
  `_update_steering_buffers`); writes row scales (5.1) and/or queues
  vector updates. Exactly-one-step latency, deterministic.

### 5.3 Per-request actuation

Per-request dynamic steering needs the config/row machinery, reusing
the earlier sketch's mutation sequence at the drain point:

1. merge the controller's per-request delta into the request's
   effective decode spec;
2. recompute `hash_steering_config(...)` (same function as admission);
3. no-op if unchanged, else `register_config(new_hash, "decode", ...)`
   (row reuse via refcounts), update the request's tracked decode hash
   so `_build_steering_index` maps its tokens to the new row,
   `release_config(old_hash, "decode")`.

Guardrails: `max_steering_configs` capacity is checked first — a
rejected update keeps the request's previous config (no silent fallback
to unsteered); structured error surfaces via the controller's stats and
the owning consumer's `on_error`. Note the scheduler also tracks
`scheduled_steering_configs` for admission; worker-side dynamic
registration must stay within the already-reserved capacity, i.e.
dynamic per-request updates should only ever *swap* a request's row,
never grow the set beyond what admission allowed. With the 5.1 scale
tensor, the cheap special case — modulating an *existing* per-request
config's strength — needs no hash churn at all: `set_row_scale` on the
request's current row.

### 5.4 Configuration surface

- `--steering-controller name:key=value,...` (mirroring
  `--capture-consumers`), entry-point group `vllm.steering_controllers`;
  controller classes get `(vllm_config, params)` plus the manager and
  capture-buffer handles at a defined init point.
- Read-only `GET /v1/steering/dynamic` reporting monitor sites, policy
  state (engaged/gain per site), and queue/apply/reject counters.
- Prometheus counters for applied/rejected/dropped updates, by source.

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
- **Phase 1**: make the controller **rank-replicated**, like
  `CaptureStepGate`: the residual at the steering hooks is replicated
  across TP ranks within a stage (read post-all-reduce), so every TP
  rank can run the same monitor GEMM + pure policy on identical inputs
  and reach identical decisions with **zero communication**. This
  requires (a) every TP rank allocates the monitor's persistent buffer
  and records the `copy_` (today only rank 0 does — the buffer copy is
  collective-free, so extending it to all ranks is safe and costs one
  D2D copy per monitored key per step per rank); (b) policies are
  bit-deterministic pure functions (no RNG, no wall clock, no
  cross-request iteration-order dependence).
- **PP**: a stage can only monitor layers it owns. v1 restricts each
  controller's monitor *and* steer sites to one stage (validated at
  startup). Cross-stage next-step decisions would need a sideband
  broadcast — deferred. Cross-stage *same-pass* (monitor stage k, steer
  stage k+1) is naturally forward-flowing and becomes available with
  Phase 2's per-token scales carried in `IntermediateTensors`, also
  deferred.
- **DP**: replicas are independent engines over disjoint requests; each
  runs its own controller. Per-request actuation partitions naturally.
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
- **Dynamic scale is runtime state, not identity.** The 5.1 scale
  tensor deliberately lives *outside* config hashes: two requests with
  the same vectors but different dynamic gains share a row whose scale
  is global... which is wrong for per-request gains. Resolution: row
  scales apply at row granularity — global rows (1/2) carry the global
  dynamic gain; a per-request gain requires the request to own a
  distinct row (which per-request steering already gives it). Sharing a
  row across requests with *different* dynamic gains is impossible by
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
  layers like `steering_index`; kernel multiplies
  `table[row] * scales_row[row] * token_scales[pid]`. All loads are
  fixed-shape against persistent buffers — graph-safe.
- A **monitor custom op** registered at the probe site (inside
  `apply_layer_steering`, after `maybe_capture_residual`):
  `token_scales[:n] = g(hidden @ probe)` with `g` a fixed elementwise
  policy (sigmoid/step with constant parameters). Fixed shapes,
  collective-free, no allocation ⇒ recordable into the graph. Every TP
  rank records it (inputs replicated ⇒ outputs identical).
- Constraints: per-token decisions only (no cross-token reductions
  in-graph); policy parameters (threshold, gain) live in small
  persistent buffers so they remain host-tunable between steps without
  recapture; layers ≤ L are unaffected (scales default 1.0 and are
  reset off-graph each step... note: reset must also be in-graph or the
  buffer must be written *by* the monitor op for all n each step —
  design detail to settle in implementation).
- Ordering caveat: hook execution order within a layer is pre_attn →
  post_attn → post_mlp; "later sites" includes later hooks of the same
  layer.

## 9. Test plan

- **Phase 0 (on this branch)**: unit tests for queue mechanics, drain
  validation/application isolation, and the full plugin policy state
  machine + sink lifecycle (`tests/v1/worker/test_steering_action_queue.py`,
  `examples/capture_consumers/dynamic_steering_controller/test.py`;
  43 tests, CPU-only). Existing steering/capture suites unaffected
  (316 passed).
- **Phase 0 still missing**: an engine-level integration test — tiny
  fixture decoder, stub consumer submits an update after step 0, assert
  step ≥ 1 logits shift in the steering vector's direction (pattern:
  `tests/models/language/generation/test_steering.py` fixture tests).
  And a GPU smoke run (gemma-3-4b on a 3090 node) — see the validation
  recipe in the plugin README.
- **Phase 1**: kernel-level tests for scale tensors (row/token scale
  composition, any_active interaction); rank-replication test in the
  TP/PP steering matrix (drive updates through the controller on all
  ranks, assert identical tables — analogous to the existing
  TP-divergence test for `set_steering_vectors`); per-request swap
  leak test (refcounts return to baseline after churn).
- **Phase 2**: graph-capture test asserting monitor op + scale reads
  replay correctly across batch sizes; eager-vs-graph equivalence on
  sums (pattern: the global-spec capture validation in
  `capture-global-spec-cudagraph`).

## 10. Relationship to the earlier sketch

Kept: the queue + step-thread drain point (its §4.2 reasoning is
correct and is exactly what Phase 0 implements); the per-request
mutation sequence via config hashes (§4.3 → our §5.3); non-throwing
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
  `registry.build_consumer` (`wants_steering_controller`); we use the
  process-global slot pattern the codebase already uses for the capture
  manager — no registry signature change, and it composes with direct
  `CaptureSink` consumers, which the batched-adapter-based injection
  did not.
- **New**: scale-tensor primitives (§5.1), step-thread monitor reading
  capture's persistent buffers directly (§5.2), in-graph Phase 2 (§8),
  and the decode-only/prefix-cache analysis (§7).

## 11. Open questions (with recommendations)

1. **Phase 1 actuation default: global rows vs per-request rows?**
   Recommend shipping global-row scale modulation + the kernel scale
   plumbing first; per-request row swap (§5.3) second. Rationale: no
   scheduler-capacity interaction, and the kernel/manager work is a
   prerequisite for both.
2. **Probe/vector bank format.** Phase 0 uses `torch.save` 1-D tensors.
   For banks, recommend reusing the steering-module packed-JSON shape
   (`SteeringHookPacked`) so one format serves `--steering-modules`,
   `/v1/steering/set`, and probe banks.
3. **Should the consumer loop survive into Phase 1, or be replaced by
   the step-thread controller?** Recommend keeping it: it is the
   extension point for *slow* policies (driver-side, learned, or
   I/O-bound) while the step-thread controller covers fast policies.
   They share the action queue.
4. **Policy plugin interface for Phase 1** — entry-point group +
   `(vllm_config, params)` like consumers, or in-tree-only policies
   first? Recommend in-tree first (threshold/hysteresis/proportional
   cover the validation needs); plugin group once the controller API
   stabilizes.
5. **Observability**: counters only, or a ring buffer of recent
   (step, score, gain) tuples queryable via the debug endpoint?
   Recommend the ring buffer — closing the loop makes behavior
   time-dependent, and post-hoc debugging without a trace is painful.
6. **`model_runner_v2`**: steering integration there is still pending
   (dev-flag-gated upstream); dynamic steering inherits whatever lands.
   No action now, but Phase 1's controller should live behind the same
   mixin seam so it ports.
7. **Eviction semantics on engagement flap**: should disengage *clear*
   the global decode vector (current Phase 0: writes zeros) or restore
   a pre-engagement baseline? Current behavior assumes the controller
   owns the decode tier exclusively; if operators also set decode
   vectors via HTTP, the controller will clobber them. Recommend
   Phase 1 give the controller its own additive row instead of writing
   the global tier (needs one reserved row: trivial with §5.1).

## 12. References

- `vllm/v1/worker/steering_action_queue.py` — Phase 0 bridge (this branch)
- `examples/capture_consumers/dynamic_steering_controller/` — Phase 0 plugin (this branch)
- `vllm/v1/worker/steering_manager.py`, `steering_model_runner_mixin.py` — steering runtime
- `vllm/v1/capture/manager.py` (`on_hook`, `_global_buffers`), `step_gate.py` — capture runtime
- `vllm/model_executor/layers/steering.py`, `steering_kernel.py` — hook + kernel
- [steering_runtime.md](steering_runtime.md), [capture_consumers.md](capture_consumers.md), [capture_parallelism.md](capture_parallelism.md)
- docs/features/steering.md §Prefix Caching — cache-identity rules
