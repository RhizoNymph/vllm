# Plan: worker→scheduler steering notification for APC correctness

Status: **IMPLEMENTED + GPU-validated (tp=1, 2026-06-16, localhost RTX
3090, gemma4-31B Q4_K_S).** A continuation of a dynamically-steered
request (override via the e2e stub) reused **0** of its prior sequence's
KV, while a continuation of an unsteered request reused 64/75 tokens
(prompt + generated) — i.e. steered decode blocks are not falsely reused,
unsteered prefix reuse is preserved (`/tmp` test `apc_correctness.py`,
mirrored as a skip-marked repo test). Companion to
`docs/design/dynamic_steering.md` (this resolves the §5.2 "known
limitation" and the analogous holes for the §5.4 tier and §8 monitor).
See "As built" below for how the final mechanism differs from (and
simplifies) the strategy survey that follows.

## 0. As built

The implementation is simpler than Strategy A (§6) feared, because the
block hasher is already **incremental and per-block**: `update_block_hashes`
only appends hashes for newly-full blocks
(`request_block_hasher`, kv_cache_utils.py), and each block's steering
key is read from `request.block_hash_decode_steering_config_hash` at
append time (`_gen_steering_extra_hash_keys`). So per-block keying needs
**no hasher-core change** — only a *forward-only* update of that field
(never the retroactive `clear()` in `set_block_hash_steering_overrides`).

Final mechanism:
- **Signature (worker, `SteeringManager.effective_decode_signature`)**: a
  per-request `int` folding the admitted decode hash with the override-vec
  hash, the tier-vec hash + gain (no quantization), and the monitor-param
  hash. Component hashes are cached and invalidated on mutation
  (`_dynamic_sig[dyn_id]`, `_tier_sig_cache`, `_monitor_sig_cache`).
  Returns `None` when nothing dynamic applies. Rank-identical
  (`hash_steering_config` over rank-replicated state).
- **Worker reporting (mixin `_compute_decode_signature_deltas`)**: computed
  at the end of `_update_steering_buffers` — i.e. from the steering state
  *as applied this step*, before the sync consumer mutates state for the
  next step. Delta-only (`_req_decode_sig_reported`): a request reverting
  to admitted reports its plain admitted hash. Stashed on
  `_pending_decode_sigs`, attached to `ModelRunnerOutput.steering_decode_signatures`
  by the model runner (rank 0 canonical).
- **Scheduler (`update_from_output` → `_apply_steering_decode_signatures`)**:
  applies the deltas forward-only via `Request.update_decode_steering_signature`
  *before* the per-request output loop appends this step's token (so the
  block hashed in `_update_request_with_output` — sync and async schedulers
  alike — gets the right key, then `cache_blocks` commits it correctly).
  `_set_request_block_hash_steering_overrides` early-returns for requests
  in `_req_decode_signature` (their key is forward-only owned; admission
  capacity-fallback doesn't apply to dynamic-pool routing). Tracking is
  cleared on finish (`_free_request`) and preemption (`_preempt_request`).
- **Chain sensitivity (correctness bonus)**: block hashes chain through
  the parent, so a block's key already reflects the *entire steering
  history* of the prefix, not just the current block — a request that was
  steered earlier in its decode and a request that wasn't get distinct
  keys for every subsequent block automatically.

Tier/monitor coverage shipped together with overrides (one signature
function), so the M0/M1/M2 split below collapsed into a single correct
implementation. The conservative "non-shareable poison" fallback (M0) was
not needed.

Tests: `tests/v1/worker/test_steering_apc_signature.py` (manager
signature), `tests/v1/test_steering_apc_block_keys.py` (forward-only
per-block keying + chain sensitivity), `TestSchedulerAPCDecodeSignatures`
in `tests/v1/test_request_steering.py` (scheduler glue), plus the
`_MixinHost` reporting path in `test_steering_dynamic_override.py`.

GPU validation (2026-06-16): APC correctness at tp=1 (a continuation of a
steered request reused 0 tokens vs 64 for an unsteered one), plus tp=2 and
pp=2 cross-node smokes. **A pre-existing, unrelated bug was found and fixed
during this validation** (commit f7afef9531): per-request *static*
`decode_steering_vectors` for a **decode-only** request (prefill_hash == 0)
was silently dropped — the nothing-active short-circuit in
`_update_steering_buffers` returned before the prefill→decode transition
that registers the decode config, and nothing primed `config_to_row`. The
fix makes a pending per-request steering hash in the batch defeat the
short-circuit. The APC notification paths are independent of this fix
(dynamic override/tier/monitor set their own active flags). Method note:
verify steering via logprobs or direct state inspection, never
output-token equality (ambiguous on confident prompts) or capture (records
the pre-steering residual).

## 1. Problem

Steering-aware prefix caching keys decode KV blocks by
`decode_steering_config_hash` (`kv_cache_utils.py::maybe_get_block_steering_keys`,
driven by `Request.decode_steering_config_hash`). Reuse is correct **only
if that hash fully identifies the steering under which the block's KV was
produced.**

Three dynamic mechanisms violate that, all in the same way — they change
the decode-token residuals (hence KV) without changing the hash:

1. **Per-request override rows** (§5.2) — worker-side routing to
   `global_decode + override_vectors`; the scheduler still reads the
   *admitted* `decode_steering_config_hash`.
2. **Dynamic additive tier** (§5.4) — a global, *time-varying* decode
   vector + gain, never reflected in any hash.
3. **In-graph monitor** (§8) — a per-token gate on the tier; the effective
   steering varies *within* a request, even within a block.

Consequence: a later request whose prompt equals an earlier request's
`[prompt + streamed output]` reuses the earlier request's decode blocks
keyed by the admitted hash, silently inheriting KV produced under
different (dynamic) steering. This is a **correctness** bug — wrong KV,
not just a cache miss — and it bites exactly the important cases
(streaming continuations, long multi-turn conversations after a steered
turn).

**Scope of this plan:** restore the reuse-correctness invariant for all
three mechanisms via a worker→scheduler notification plus a per-block
steering signature. Per-request-row *token* gating (deferred separately)
does not change this plan — it is just another contributor to the
effective decode signature.

## 2. The invariant to restore

> A decode KV block may be reused by another request iff both blocks were
> (or would be) produced under a **byte-identical effective decode
> steering signature** over the block's token span.

Because KV at a position is a deterministic function of (token prefix,
model, steering applied so far), identical tokens + identical effective
steering ⇒ identical KV ⇒ safe reuse. So a *content* signature of the
effective decode steering is sufficient (not merely a poison flag), and
the monitor's per-token gate is covered for free: given the same tokens
and the same monitor params, the gate is deterministic.

## 3. The structural finding (why this is more than a notification)

APC steering keys are **per-request-constant**: `decode_steering_config_hash`
is a `cached_property` on `SamplingParams`, fixed at admission, and the
block-hash path applies *one* decode hash uniformly to *all* of a
request's decode blocks. `Request.set_block_hash_steering_overrides`
(request.py:237) even `clear()`s and recomputes **all** block hashes when
the override changes.

Dynamic steering changes **mid-stream**. If an override/tier/monitor
engages at decode block 6, you cannot express "blocks 1–5 = admitted,
blocks 6+ = dynamic" with a single per-request decode hash. Worse,
retroactively rekeying blocks 1–5 to the dynamic hash is *incorrect*: a
later request that actually ran the dynamic config would then reuse
blocks 1–5, which were produced under the *admitted* config.

**Therefore the decode block hash must vary per block, by the steering
signature in force when that block was filled — not per request.** This
is the core change; the notification is necessary but not sufficient.

## 4. Existing substrate (anchors)

- `Request.set_block_hash_steering_overrides(prefill, decode)` — overrides
  the block-hash steering config and recomputes (request.py:237). Today
  used only for the scheduler capacity-fallback case.
- `Scheduler._set_request_block_hash_steering_overrides` — called per
  running request each `schedule()` (scheduler.py:330, invoked at 660).
  The natural place to inject a dynamic decode signature.
- `ModelRunnerOutput.capture_results` — precedent for worker→scheduler
  structured per-step data, consumed in `update_from_output`
  (scheduler.py:1473). The notification rides the same channel.
- `hash_steering_config(...)` (sampling_params.py:907/930 via the helper)
  — the deterministic effective-vector hasher to reuse for the signature.
- Worker-side override state: `_req_dynamic_decode` (req_id→dyn_id),
  `_apply_request_override` / `_drop_request_dynamic_override`,
  `dynamic_tier_vectors` / `dynamic_tier_gain`, `monitor_configs`
  (all in `steering_model_runner_mixin.py` / `steering_manager.py`).

## 5. Mechanism

### 5.1 Notification channel (worker → scheduler)

Add `steering_decode_signatures: dict[str, int] | None` to
`ModelRunnerOutput` (outputs.py). Each step, the **rank-0** model runner
(the one whose output reaches the scheduler — rank-replicated state means
all ranks agree, so rank 0 is canonical) emits, for each request whose
effective decode steering differs from its admitted hash this step, the
current **effective decode signature** (§5.2). Absent/0 ⇒ "admitted hash
applies" (no override/tier/monitor touching this request).

`Scheduler.update_from_output` records these into a new
`self._req_decode_signature: dict[str, int]` (cleared on finish). This is
pure scheduler-local bookkeeping; it never mutates admission state (the
single-mutator contract holds — overrides remain worker-side routing).

### 5.2 The effective decode signature (what the worker hashes)

For a request in decode this step, the worker computes a deterministic
`int` folding everything that shaped its decode KV:

```
sig = hash_steering_config(admitted_decode_effective_vectors)         # base
      ⊕ tier_signature        # hash(tier vectors) ⊕ quantized gain, or 0
      ⊕ override_signature     # hash(override vectors) for this req, or 0
      ⊕ monitor_signature      # hash(layer,hook,probe,threshold,sharpness), or 0
```

- **Tier**: hash the per-(layer,hook) tier vectors and fold the gain.
  **Decision (locked): no quantization.** Any gain change ⇒ a new
  signature, because KV at gain G1 is not byte-identical to KV at G2 and
  bucketing would hand back wrong KV. This is correctness-exact. In
  practice a steering configuration sits at a *fixed* gain for a given
  vector, so the keyspace stays small; the only case that forgoes decode
  reuse is a *continuously-modulating* (proportional) gain — see §8.1,
  documented as intended behavior, not a regression.
- **Override**: hash the override delta vectors (the consumer-supplied
  arrays), composed identically to how the row is populated
  (`global_decode_effective + override`).
- **Monitor**: a monitor active at a site that gates this request's tier
  ⇒ fold `(layer, hook, probe_hash, threshold, sharpness)`. Correct
  because the per-token gate is deterministic given tokens + these params.

The signature must be **stable across the worker/scheduler boundary** and
**rank-identical** (it is: all inputs are rank-replicated; use the same
`hash_steering_config` the scheduler already trusts).

### 5.3 Per-block application (the structural change)

`_set_request_block_hash_steering_overrides` gains a branch: when
`self._req_decode_signature.get(req_id)` is set, the **decode** hash for
that request's *currently-being-filled and future* decode blocks becomes
that signature. To make this per-block rather than per-request, extend the
block-hash decode keying so a decode block records the signature in force
when it was filled (see §6 for the two strategies). Prefill blocks are
untouched (prefill steering is admission-fixed and already correct).

## 6. Two implementation strategies for per-block keying

**Strategy A — per-block steering epoch (precise, more work).**
Tag each decode block, at fill time, with the effective decode signature
then in force. Extend `maybe_get_block_steering_keys` + `update_block_hashes`
to read a per-block signature (carried on the request as a small
`list[(block_index_or_token_range, sig)]`) instead of one
`block_hash_decode_steering_config_hash`. Blocks before the first dynamic
change keep the admitted hash (and stay shareable with admitted-only
requests); blocks after keep the dynamic signature. Maximal correctness
**and** maximal reuse. Cost: touches the block-hash generation core and
its tests.

**Strategy B — boundary-flush (simpler key model, coarser reuse).**
When a request's effective decode signature changes, force the current
partial decode block to flush at the change point and switch the
per-request decode hash going forward. Each block is then wholly under one
regime, so the existing single-decode-hash machinery stays valid; the
scheduler just needs to (a) track the change point and (b) avoid the
retroactive `clear()`-all-block-hashes behavior (only future blocks get
the new hash). Cost: block-boundary control mid-stream is awkward in the
current scheduler and may waste a partial block per change.

**Recommendation: Strategy A.** The retroactive `clear()` in
`set_block_hash_steering_overrides` is already the wrong primitive for
mid-stream change; replacing the per-request decode hash with a per-block
signature list fixes the root cause and is the only option that preserves
reuse of the pre-dynamic prefix. Keep `set_block_hash_steering_overrides`
for the existing capacity-fallback (admission-time) use.

## 7. Milestones

- **M0 — conservative-correct (ship first, unblocks correctness).**
  No keying subtlety: when the worker reports *any* dynamic decode
  steering for a request, the scheduler marks that request's decode blocks
  produced from that point **non-cacheable / non-shareable** (a per-request
  unique decode signature, applied per-block from the change point so it
  does not retroactively rekey the clean prefix). Definitely correct;
  forgoes reuse of dynamically-steered decode blocks. Requires §5.1
  notification + the per-block-from-change-point mechanism (a reduced
  Strategy A where the "signature" is just a per-request sentinel).
- **M1 — content signatures (restore reuse).** Replace the M0 sentinel
  with the real §5.2 signature so two requests under identical effective
  steering + identical tokens reuse each other's dynamically-steered
  decode blocks.
- **M2 — tier/monitor coverage + gain quantization policy.** Fold tier and
  monitor into the signature (M0/M1 can start with overrides only, since
  that is the user's headline case, but the tier hole is real and should
  not ship half-fixed — call this out in the M0 PR).

### 8.1 Gain modulation and reuse (intended behavior)

With no quantization, the reusability of a steered decode block is decided
by how the gain moves:

- **Fixed gain** (a configuration pinned to one strength — the common
  case): the signature is stable, so blocks under that config are
  reusable across requests with the identical config + tokens.
- **Binary engage** (gain ∈ {0, G}): two signatures; reuse works within
  engaged and within disengaged spans.
- **Continuously-modulating (proportional) gain**: the signature changes
  most steps, so decode blocks produced while it is active are effectively
  non-reusable — and because the *tier* is global, a proportional global
  tier forgoes decode reuse for every request while active. This is
  **intended**: returning KV from a different gain would be incorrect.
  Document it so it reads as a policy consequence, not a cache regression.

## 8. Timing / latency analysis

Override engages at decode step N (worker). `update_from_output(N)` runs
after the step → scheduler learns at the boundary between step N and the
`schedule()` for N+1. `_set_request_block_hash_steering_overrides` then
applies the signature for blocks filled from N+1 onward. The tokens
generated **at step N** land in the block that was being filled under the
old signature. Two cases:

- The step-N token starts a fresh block ⇒ clean: that block is keyed with
  the new signature once the scheduler learns (it is not yet full/cached).
- The step-N token lands mid-block ⇒ that block straddles the change. Under
  Strategy A, the block's signature is the one in force when it *completes*
  (or we conservatively assign the dynamic signature to any block touched
  after the change). Document: a block straddling a steering change is
  keyed dynamic (safe; at worst forgoes reuse of that one block).

Because blocks are only *shared* once full and cached, and the scheduler
always learns within one step, **no block is ever cached under a stale
signature** as long as the signature is resolved before the block is
committed to the cache. The plan's correctness rests on: *signature
resolved at-or-before block-commit*. Verify the commit point
(`cache_blocks` / coordinator) sits after `update_from_output` in the step
loop — it does (caching happens during the next `schedule()`).

## 9. TP / PP and rank considerations

- The signature is a pure function of rank-replicated state, so all ranks
  agree; only rank 0's `ModelRunnerOutput` reaches the scheduler, so rank 0
  is the canonical emitter (no cross-rank reconciliation needed).
- The async action queue is tp=1-gated, but overrides at tp>1 arrive via
  sync-consumer returns (every rank, identical). The notification is
  emitted from the same per-step path, so it composes with the pending
  tp=2 work — and the tp=2 smoke should additionally assert the emitted
  signatures match across ranks.
- PP: a request's monitor/steer sites live on one stage (v1 restriction);
  the signature is computed where that state lives and surfaced on the
  stage that produces `ModelRunnerOutput`. Confirm during implementation.

## 10. Risks & open questions

- **Block-commit ordering** (§8) — the linchpin, and **non-trivial**:
  there are multiple `cache_blocks` paths (`kv_cache_manager`/coordinator,
  the KV-connector remote paths at scheduler.py:2263/2273, and a separate
  `async_scheduler.py` path). **Async scheduling is enabled by default** in
  the current build (observed in the engine boot log:
  "Asynchronous scheduling is enabled"), which decouples scheduling from
  output handling by a step — so the worker→scheduler signal may arrive a
  step later relative to caching than the sync analysis in §8 assumes.
  Implementation must confirm, **for both the sync and async schedulers**,
  that a decode block is not committed to the shared cache before the
  scheduler has applied that block's signature; if the async path can
  commit a block one step early, M0 must additionally hold-back / invalidate
  the in-flight block on the step a dynamic change is reported.
- **Gain quantization** (§5.2): continuous tier-gain changes could thrash
  the keyspace. Decide a grid or "any change ⇒ new signature."
- **`set_block_hash_steering_overrides` retroactive `clear()`**: must NOT
  be reused for the dynamic path (it rekeys the clean prefix). Strategy A
  introduces a separate per-block path and leaves the capacity-fallback
  use intact.
- **Hash collision domain**: dynamic signatures must not collide with
  admitted config hashes in a way that causes false reuse. Reuse
  `hash_steering_config` and fold a domain tag ("dynamic") to keep spaces
  disjoint.
- **Streaming re-add / preemption**: overrides drop on these (existing
  lifecycle); the signature must follow — clear `_req_decode_signature` on
  the same hooks that call `_drop_request_dynamic_override`.

## 11. Test plan

- **Unit (scheduler, CPU):** given a worker-reported signature for a req,
  `_set_request_block_hash_steering_overrides` keys that req's decode
  blocks-from-change-point with it and leaves the clean prefix + other
  requests untouched; clears on finish/preempt/stream-readd.
- **Unit (block hashing, CPU):** per-block signature list produces correct
  keys for a request that changes signature mid-stream; admitted-only and
  dynamic requests with the same tokens do **not** collide; two dynamic
  requests with identical signature + tokens **do** match.
- **Unit (worker, CPU):** the emitted signature is correct + rank-stable
  for override / tier / monitor; 0/absent when nothing dynamic applies.
- **Engine e2e (GPU):** the headline scenario — request A runs with a
  dynamic override and streams output; request B submits A's
  `[prompt+output]` as its prompt. Assert B does **not** reuse A's
  dynamically-steered decode blocks (cache-miss / recompute), and that an
  identical-steering B *does* reuse them (M1). Pattern mirrors the
  steering-aware APC tests in `tests/.../test_steering*` + the prefix-cache
  validation recipe.
- **Regression:** non-steered and admission-only-steered requests are
  byte-for-byte unchanged in cache behavior (zero impact when no dynamic
  steering is active).

## 12. Surface-area summary (files to touch)

- `vllm/v1/outputs.py` — `ModelRunnerOutput.steering_decode_signatures`.
- `vllm/v1/worker/steering_model_runner_mixin.py` — compute + attach the
  per-request signature each step (override/tier/monitor); clear hooks.
- `vllm/v1/worker/steering_manager.py` — helpers to hash tier/override/
  monitor state deterministically.
- `vllm/v1/core/sched/scheduler.py` — consume in `update_from_output`;
  inject into `_set_request_block_hash_steering_overrides`; lifecycle clears.
- `vllm/v1/core/kv_cache_utils.py` + `vllm/v1/request.py` — per-block
  decode steering signature (Strategy A); keep the admission-time path.
- Tests as in §11.
```
