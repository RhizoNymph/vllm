# Steering + Capture on the V2 Model Runner

Design/implementation contract for porting the **activation steering** and
**activation capture** control planes from the v1 GPU model runner
(`vllm/v1/worker/gpu_model_runner.py`) to the experimental v2 runner
(`vllm/v1/worker/gpu/model_runner.py`).

## Scope

In scope:

- Wiring the runner-agnostic steering/capture subsystems into the v2 runner's
  lifecycle so both features behave identically to v1.
- A v2-native control plane (new modules under `vllm/v1/worker/gpu/`) that keeps
  its own per-request state, since v2 does not retain a `CachedRequestState`
  dict the way v1 does.

Out of scope:

- The data plane (model-side custom ops, layer buffers, Triton kernels). These
  live in `vllm/model_executor/` and are **already shared** by both runners.
- Refactoring the v1 mixins. The v1 path is validated/production; we leave it
  untouched and write v2-native modules instead.
- Dynamic-steering (steer-from-capture feedback) and routed-experts capture
  (tracked separately).

## Key architectural fact

The activation read/write mechanism is **not** in either model runner. Decoder
layers in `vllm/model_executor/models/*.py` call `apply_layer_steering()` and
`maybe_capture_residual()`; `register_steering_buffers()` runs in the model's
`__init__`. Both runners load the *same* model object via
`model_loader.load_model()`, so in v2 the buffers already exist, the custom ops
(`torch.ops.vllm.apply_steering`, `torch.ops.vllm.capture_residual`) already
fire, and they safely no-op when no control plane drives them (steering tables
stay zero → `any_active=False`; `get_active_capture_manager()` is `None` →
constant-folds under `torch.compile`).

The split is therefore:

| Plane | Location | Status in v2 |
| --- | --- | --- |
| Data plane (ops, buffers, kernels, store, managers, gate, types) | `model_executor/`, `v1/capture/`, `v1/worker/steering_manager.py` | shared, reused unchanged |
| Scheduler handoff (`NewRequestData.{prefill,decode}_steering_config_hash`, `capture_block_hashes`, `sampling_params.capture`; `ModelRunnerOutput.capture_results`) | `v1/core/sched/output.py`, `v1/outputs.py` | shared, already present |
| Control plane (init, per-step buffer fill / plan build, force-eager, request lifecycle, output drain) | runner | **absent — this port** |

## V2 runner seams

The v2 runner splits the monolithic v1 `execute_model` into discrete methods.
The port attaches to these (all in `gpu/model_runner.py`):

- `load_model` (266): construct managers/gate/store; init steerable-layer
  discovery. Buffers are already registered model-side.
- `add_requests` (691): per `new_req_data` in `scheduled_new_reqs` — register
  steering config + track phase; `gate.register` (all ranks) + capture
  `register_request` (TP0). Note `add_requests` calls `_remove_request` first
  for streaming re-adds, so refresh state accordingly.
- `update_requests` (736): prefill→decode transition / resumption bookkeeping.
- `finish_requests` (678): use `scheduler_output.finished_req_ids` for steering
  release + capture finalize + `gate.drop`; `preempted_req_ids` → steering
  reset, **not** capture finalize.
- `execute_model` (1009):
  - Force-eager seam at the `dispatch_cg_and_sync_dp(..., need_eager=...)` call
    (1042–1050): OR in `capture_pending` (client-spec captures only; global
    specs ride the cudagraph-safe persistent-buffer path). **Steering needs no
    force-eager** — its tables/index are persistent buffers written before the
    forward, so graph replay reads them correctly.
  - After `prepare_inputs` (1060) and before the model forward (1167): build the
    per-step view, `_update_steering_buffers(view)`, and
    `capture_manager.build_step_plan(view)` (TP0).
  - After the forward (after 1210, before non-last-PP return at 1220):
    `_finalize_capture_step()` (consume plan, async dispatch).
- `sample_tokens` (1229): attach drained `_pending_capture_results` to the
  `ModelRunnerOutput` (1276); `_finalize_capture_for_request_async` results land
  here, same as v1's `get_output`.

### Per-request state ownership

v2's `RequestState` (`gpu/states.py`) holds only tokens/lengths — not
`sampling_params` or steering hashes. The control plane therefore keeps its own
dicts (`req_id → (prefill_hash, decode_hash, phase)` for steering;
gate selectors + manager registration for capture), populated from
`NewRequestData` in `add_requests`. The per-step view is built from v2's
`InputBatch` (`req_ids` ordering + `idx_mapping_np` + `num_scheduled_tokens`)
plus `req_states` (`num_computed_tokens_np`, `prefill_len`, `prompt_len`).

## Rank-replication invariant

Preserved exactly as in v1: the force-eager decision (`CaptureStepGate`) and
steering-manager row allocation are rank-local and deterministic, fed by the
broadcast `scheduler_output`. No hot-path collectives. Every new seam must read
only rank-identical inputs.

## CUDA-graph interaction

- Steering: persistent buffers (`steering_table_*`, `steering_index`) written
  in-place before the forward → FULL-graph replay reads them. Safe.
- Capture global specs: fixed-shape full-residual copy into persistent
  `_global_buffers`, baked at warmup. Safe (no force-eager).
- Capture client specs: dynamic `index_select` → not graph-capturable → gate
  forces eager for that step only.

## Workstreams

1. **Capture control plane** — DONE (CPU-tested, GPU pending).
   The runner-agnostic control plane is shared with the v1 runner in
   `vllm/v1/worker/capture_runner_mixin.py` (`CaptureRunnerMixin`):
   `_init_capture_state`, `_register_capture_request`, `_capture_add_request`,
   `_capture_finish_request`, `_finalize_capture_step`,
   `_finalize_capture_for_request_async`, the sync-consumer step loop
   (`_build_step_capture_view` / `_run_sync_consumers` / `_warmup_sync_consumers`),
   and the result drains. Both runners implement two hooks the shared step-view
   builder calls: `_iter_step_capture_rows` (v1 walks `input_batch` accumulating
   offsets; v2 reads `query_start_loc_np` + `req_states`) and
   `_step_view_token_ids` (v1 copies the CPU token window; v2 returns empty).
   `gpu/capture_runner_mixin.py` (`CaptureRunnerMixin`) subclasses the shared
   mixin and keeps only the genuinely-v2 pieces: the force-eager gate view and
   gather-plan view (`_build_capture_{gate,batch}_view`, `_capture_gate_decision`,
   `_capture_build_plan`), which must be built before v2's `InputBatch` exists,
   plus the two hooks above. Tests: `tests/v1/worker/test_gpu_v2_capture_glue.py`,
   `tests/v1/worker/test_sync_steering_integration.py`.
2. **Steering control plane** — DONE (CPU-tested, GPU pending).
   Fully shared on `SteeringModelRunnerMixin` (de-fork complete, step H): the
   v2 runner mixes it in directly
   (`GPUModelRunner(..., CaptureRunnerMixin, SteeringModelRunnerMixin)`) and the
   per-runner steering surface is just two batch-state accessor overrides —
   `_steering_batch_view` (the per-step hot path's view) and
   `_steering_req_position` (the override-apply decode-only phase guard) — which
   read v2's `req_states` + `input_batch` instead of v1's batch-ordered columns.
   They live on `gpu/capture_runner_mixin.py`, colocated with the capture glue
   that already owns those same v2 arrays; the former
   `gpu/steering_runner_mixin.py` (`SteeringRunnerMixin`) is deleted. No
   force-eager seam (persistent buffers). `gpu_worker.py` already forwards the
   RPCs to `self.model_runner.*`.
   Tests: `tests/v1/worker/test_gpu_v2_steering_glue.py`.

   **Canonical per-request steering state (de-fork step C).** The per-request
   steering identity + phase now lives in one shared store,
   `SteeringModelRunnerMixin._steering_reqs: dict[str, _SteeringReqState]`,
   populated identically on both runners from the broadcast `NewRequestData`.
   The whole lifecycle is shared and runner-agnostic:
   `_steering_add_request` (admission + streaming re-add),
   `_steering_register_request` (the register-fresh core, also used by resume),
   `_steering_finish_requests` (release on finish/preempt),
   `_steering_release_state`, `_steering_transition` (prefill→decode), and
   `_reset_steering_for_resumption` (v1 resume re-register). v1 drives them from
   `_update_states` / `_update_streaming_request`; v2 from `add_requests` /
   `finish_requests`. v1's former `_register_initial_steering_config` /
   `_refresh_streaming_steering` / `_release_finished_steering_configs` /
   `_req_steering_phase` and v2's private `_SteeringReqState` copy are gone.
   The per-step hot path is unified too (de-fork step E, below): both runners
   run one shared `_update_steering_buffers` that reads per-request steering
   identity from `_steering_reqs` only. v1's former `_handle_steering_transition`
   shim (a bridge from the batch hash columns to `_steering_transition`) and the
   v1 input-batch hash columns themselves are gone. The cross-runner conformance
   harness (`tests/v1/worker/test_steering_conformance.py`) asserts both runners
   drive the manager identically across the whole lifecycle *and* write
   byte-identical device buffers.

   **Preemption unified on release-at-preemption (de-fork step D).** Both
   runners now RELEASE a preempted request's config rows *and* any per-request
   dynamic override at preemption time (the finish site unions
   `finished_req_ids` with `scheduler_output.preempted_req_ids`), then
   re-register a fresh prefill config on resume — v1 via
   `_reset_steering_for_resumption`, v2 via the `add_requests` new-request path.
   Previously v1 HELD its rows across preemption (releasing only at resume);
   that pinned pool rows for the duration of the preemption. The scheduler
   already agrees: `_preempt_request` (`scheduler.py:1256`) resets
   `num_computed_tokens = 0` and drops the decode-signature tracking, and the
   waiting-queue admission loop (`scheduler.py:705–718`, `:762–790`) reserves the
   resumed request's prefill + decode steering rows before re-admitting it, so
   the re-registration's `register_config` cannot overflow (the same guarantee
   v2 already relied on). A request both preempted and finished in one step
   releases exactly once (the union is a set; `_steering_finish_requests` pops
   the canonical state idempotently).

   **Unified per-step hot path (de-fork step E).** Both runners run one shared
   `SteeringModelRunnerMixin._update_steering_buffers`. The only runner-specific
   input — batch order + per-request token counts — is read through a
   `SteeringBatchView` (`vllm/v1/worker/steering_batch_view.py`), a reusable
   holder each runner mutates in place once per step (zero per-step allocation).
   `_steering_batch_view` is the seam: v1 builds it from its batch-ordered
   `input_batch` columns (identity slot→row map); the v2 runner overrides it
   (on `gpu/capture_runner_mixin.py`) to read `input_batch.idx_mapping_np` +
   `req_states` (`num_computed_tokens_np` / `prompt_len.np`). Steering identity (hashes + phase) comes from `_steering_reqs`
   only, which retired v1's input-batch hash columns
   (`request_prefill_steering_hash` / `request_decode_steering_hash` and the
   `steering_hash_to_request_ids` index — the hot path was their only reader).
   The shared body, beyond the per-token `steering_index`, every step:
   - **§5.4 dynamic tier** — `steering_token_scales` (gain for decode tokens of a
     tier-active state, `0` for prefill = §7 cache safety) via the same
     per-request → `np.repeat` → pinned-H2D pattern as the index.
   - **Phase 2 row gating** — resets `steering_row_gate` to `1.0` each step and
     writes `steering_decode_mask` (`1.0` decode / `0.0` prefill) so an in-graph
     monitor only gates decode rows.
   - **Dynamic override pool** — a live `_req_dynamic_decode[req_id]` routes that
     request's decode tokens to its pool row (`get_dynamic_row`) instead of the
     admitted decode row; overrides drop on finish / preempt / streaming re-add.
     The shared `_apply_request_override` reads the decode-only phase guard
     through `_steering_req_position`, which the v2 runner overrides (on
     `gpu/capture_runner_mixin.py`) to read v2's `req_states` (`req_id_to_index`
     + `num_computed_tokens_np` / `prompt_len.np`).
   - **Async transport** — drains the in-process `SteeringActionQueue` at the top
     (before the nothing-active short-circuit, so a drained update can activate
     steering) via the shared `_apply_steering_actions`.
   - **APC notification** — the shared `_compute_decode_signature_deltas` folds
     admitted decode config + override / tier / monitor into an effective
     signature and reports only changed signatures; the runner attaches them on
     `ModelRunnerOutput.steering_decode_signatures`.
   - **Short-circuit predicates** — extended to `has_dynamic` / `has_dynamic_tier`
     / `has_monitor` (the tier-only latent bug fix) and the active→inactive
     transition resets `token_scales` / `row_gate` / `decode_mask` / monitor-active
     flags. The cheap `_scales_dirty` populate path (§5.3) is wired too.

### Validation

GPU-validated on Qwen3-0.6B (RTX 3090, TP1/PP1), forcing
`VLLM_USE_V2_MODEL_RUNNER=1`:

- Steering (eager **and** cudagraph): global `set_steering_vectors` shifts the
  output and `clear_steering_vectors` restores the exact baseline — confirming
  the persistent-buffer path is cudagraph-safe (no force-eager).
- Capture (eager): a client-spec request (`post_attn`, layer 5, `last_prompt`)
  delivers one `(1, hidden)` bf16 row to a driver consumer's `on_capture`.

Once validated, the interim Phase-1 fallback guard was removed so v2 actually
runs these features (auto-selected for Qwen3, or via the env override).

Expanded GPU matrix (Qwen3-0.6B unless noted):

- Steering: global (eager + cudagraph); per-request inline; **mixed batch** (a
  steered and an unsteered request together — the unsteered output is
  byte-identical to baseline, so per-request rows don't cross-contaminate);
  decode-only (lazy decode-config registration at the prefill→decode boundary);
  per-request under cudagraph; chunked prefill (multi-step prefill).
- Capture: client-spec eager; client-spec **under cudagraph** (the force-eager
  gate fires for that step); `all_generated` positions (multi-step decode);
  global-spec under cudagraph (persistent-buffer path, no force-eager).
- Cross-node (2×3090, Ray): steering under TP=2 and PP=2 (rank-replication and
  `locally_owned_layers` confirmed).
- Model coverage: gemma-3-4b-it runs on v2 with steering (hidden 2560 / 34
  layers) — the port is not Qwen3-specific.

Additional GPU coverage:

- Steering: named-module (`register_steering_modules` + `steering_module_ref`)
  and per-request scale (scale 0 → baseline, scale 1 → steered); async
  scheduling.
- Capture: filesystem consumer (worker-location, files read back); multiple
  consumers (filesystem + global logging); activation-store **write** path
  (64 prompt rows written with prefix caching on — block-hash wiring works);
  async scheduling.
- Capture under **TP=2** (exactly one rank — TP rank 0 — writes; the other
  writes nothing) and **PP=2** (stage 0 captures its layer 5, stage 1 captures
  its layer 20 — per-stage `local_layer_range` filtering correct), via the
  worker-location filesystem consumer.
- Capture under **TP=2 and PP=2 *with cudagraph*** (cross-node, 2×3090 Ray;
  `enforce_eager=False`, FULL_AND_PIECEWISE graphs compiled): mixing plain
  (cudagraph) and client-spec capturing (force-eager gate) requests in the same
  run does not hang — the force-eager decision stays rank-replicated across
  ranks/stages so every rank toggles eager↔graph in lockstep (including the PP
  P2P send/recv between stages). TP rank 0 / each PP stage wrote exactly its own
  30720-byte (`15 gen tokens × 1024 × bf16`) capture and no other rank did.
- Steering **under prefix caching**: a steered request whose prompt is largely
  served from the KV cache (partial hit — the scheduler always reserves the last
  block to recompute logits) still steers correctly (degenerate `οοο` output vs
  the unsteered baseline of the same tokens). KV block hashes are steering-aware,
  so a steered request does not reuse an unsteered cache. The narrower
  *admit-straight-to-decode* branch (`_steering_add_request`, `num_computed >=
  num_prompt`) is **not reachable under single-engine APC** — the scheduler caps
  `num_computed` at `num_prompt − block_size`, never `>= num_prompt`; that branch
  is only reachable when an external mechanism (KV connector / disaggregated
  prefill) sets `num_computed_tokens` to the full prompt at admission. It mirrors
  the v1 mixin's identical defensive branch and stays formally unexercised
  without a connector. (Reminder: inline `SamplingParams.steering_vectors`
  requires `enable_steering=True` at engine init — otherwise the worker steering
  manager is `None` and steering silently no-ops; the per-request hash is still
  packed host-side.)

- Capture + prefix caching via the **OpenAI server** (v2, activation store on):
  a repeated prefix reuses under `all_generated` (32-token prefix-cache hit on
  the 2nd request) and recaptures under `all_prompt` (0 hits, full re-forward) —
  identical to the documented v1 behavior. Store write validated separately
  (64 rows). The Step-A store *serve* path (`pop_pending_serve` /
  `serve_from_store`) was not observed to trigger, consistent with v1's
  "all_prompt → full recapture", so those two lines stay formally unexercised.

- **Preemption resume**: under a tiny KV cache, the worker observed 248
  preemption events; all 16 steered requests still produced the correct steered
  output (no config leak). Capture under preemption: 72 preemption events, all
  24/24 capturing requests still delivered — preempted capturing requests resume
  and capture cleanly (no lost/double captures).
- **Steering hook points**: pre_attn and post_block both shift/clear correctly
  (post_attn was already covered); the prefill-only tier
  (`prefill_steering_vectors`) steers.
- **Capture positions**: `all` (prompt+generated rows) and an explicit index
  list (`[0, 2]` → exactly 2 rows), in addition to `last_prompt`/`all_generated`.

- **Streaming re-add**: an async streaming-input session (prompt fed in chunks
  via `AsyncLLM.generate(prompt=<async generator of StreamingInput>)`) with
  steering produced steered output, and the port's re-add branch fired
  (`_steering_add_request` saw an already-tracked req_id → released the old
  config + registered the new one). No crash.

- **Capture re-add / preemption resume** (the asymmetry steering already
  handled): `add_requests` calls `_remove_request` first, which does *not* touch
  capture state, so a re-admitted request would re-register an already-registered
  id. The capture manager raises on duplicate ids (`already registered`), caught
  as a request error. Two paths reach this:
    - *Streaming re-add* — the request is still live (`_remove_request` returns
      `True`); the grown prompt makes the prior chunk's registration stale, so
      `_capture_add_request` now discards it (gate `drop` + `unregister_request`,
      no finalize) and re-registers against the new prompt.
    - *Preemption resume* — on v2 the scheduler folds `scheduled_resumed_reqs`
      into `scheduled_new_reqs`, so a resumed request flows through
      `_capture_add_request` with `was_present=False` while its registration
      survived (capture is intentionally not finalized on preempt). It is kept
      as-is (skip re-registration), preserving rows captured before preemption.
  GPU-validated on Qwen3-0.6B with a clean before/after: pre-fix, a streaming
  session logged `capture request '...' is already registered` on each re-add,
  and the preemption scenario (24 capturing requests, 64-block KV cache) logged
  20 such rejections; post-fix, both paths log zero rejections and deliver all
  captures (24/24 under preemption). CPU glue tests cover the three branches
  (fresh / streaming re-add / preemption resume) plus the non-capturer rank.

Still unverified: spec-decode; DP; combined 2-D parallelism (TP *and* PP on the
same request, and TP/PP > 2 — each validated independently, intersection needs
≥4 GPUs); the async-dispatch overload policies (`spill`/`drop`/`block` —
runner-agnostic transport code shared with v1, not touched by the port); the
store *serve* path (doesn't trigger for `all_prompt` even on v1 — full recapture
by design); and the steering *admit-straight-to-decode* branch (needs a KV
connector / disaggregated prefill — unreachable under single-engine APC, see
above). (The capture-consumer entry points were missing from one prebuilt
install's metadata — a stale dist-info issue fixed by reinstalling; pyproject
already declares them.)

## Validation

- CPU unit tests for the v2 control-plane glue (state bookkeeping, view
  construction, gate decisions) where the data plane can be exercised via the
  python fns (the CUDA ops are dispatch-only on GPU).
- GPU end-to-end on node2 (gemma + qwen3): steering eager-vs-cudagraph parity,
  capture client-spec (`all_prompt` → recapture) and `all_generated` → reuse,
  TP/PP rank agreement. Mirrors the v1 validation matrix.
