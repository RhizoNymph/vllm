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
   `gpu/capture_runner_mixin.py` (`CaptureRunnerMixin`): init, gate,
   force-eager seam, `_build_capture_{gate,batch}_view`,
   `_register_capture_request`, `_finalize_capture_step`,
   `_finalize_capture_for_request_async`, output drain, activation store.
   Tests: `tests/v1/worker/test_gpu_v2_capture_glue.py`.
2. **Steering control plane** — DONE (CPU-tested, GPU pending).
   `gpu/steering_runner_mixin.py` (`SteeringRunnerMixin`, a subclass of
   `SteeringModelRunnerMixin` that reuses init / discovery / validation / the
   public RPC API / `_resolve_request_steering` and overrides only the three
   v1-state-coupled paths). Keeps its own `_steering_reqs` per-request state;
   `_steering_add_request` (register + streaming re-add), `_steering_finish_requests`
   (release on finish/preempt), `_update_steering_buffers_v2` (transition +
   per-token index). No force-eager seam (persistent buffers). `gpu_worker.py`
   already forwards the RPCs to `self.model_runner.*`.
   Tests: `tests/v1/worker/test_gpu_v2_steering_glue.py`.

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

Not yet exercised on GPU (mirrors v1, but unverified here): TP>1 / PP>1,
per-request inline steering and named modules, capture prefix-cache reuse,
preemption resume, and spec-decode token layout.

## Validation

- CPU unit tests for the v2 control-plane glue (state bookkeeping, view
  construction, gate decisions) where the data plane can be exercised via the
  python fns (the CUDA ops are dispatch-only on GPU).
- GPU end-to-end on node2 (gemma + qwen3): steering eager-vs-cudagraph parity,
  capture client-spec (`all_prompt` → recapture) and `all_generated` → reuse,
  TP/PP rank agreement. Mirrors the v1 validation matrix.
