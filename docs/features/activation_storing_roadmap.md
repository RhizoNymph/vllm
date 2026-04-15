# Activation Storing — Implementation Roadmap

Companion to [`activation_storing.md`](activation_storing.md). That doc
is the design spec; this doc is the execution plan. It slices the
feature into phases so work can run in parallel worktrees and each PR
stays reviewable on its own.

## Guiding constraints

- Each phase ships as its own PR off `feat/activation-storing`.
- Phases marked **parallelizable** can run concurrently in sibling
  worktrees; expected merge conflicts are called out per-phase.
- No phase may break existing tests, even if its own feature is not yet
  reachable end-to-end. Dead-but-correct code is fine during the ramp;
  wiring happens in phase 4.
- The invariants listed in the spec (§ Invariants) are load-bearing from
  day one — especially invariant 2 (capture reads pristine residual)
  and invariant 3 (cold path is free). Reviewers should reject any
  phase that violates them even if the feature "works".

## Phase map

```
        ┌────────────────────────┐
        │ P1: types + validation │ ─┐
        └────────────────────────┘  │
                                    ├──▶ P3: capture manager + op ──┐
        ┌────────────────────────┐  │                               │
        │ P2: writer pool        │ ─┘                               ├──▶ P4: runner integration ──▶ P5: protocol + response
        └────────────────────────┘                                  │
                                                                    │
                                        (P5 can start in parallel with P4 once response shape is frozen)
```

Phases 1 and 2 run in parallel. Phase 3 depends on phase 1. Phase 4
depends on phases 2 and 3. Phase 5 starts once phase 4 has a stable
`EngineCoreOutput.capture_*` contract and can land independently.

---

## Phase 1 — Types, config, admission validation

**Branch:** `feat/activation-storing-types`

**Goal.** Land all pure-data scaffolding: config object, per-request
spec, selector expansion, CLI flags, and admission-time validators.
Nothing this phase adds runs on the forward pass.

**Files.**

| File | Change |
|---|---|
| `vllm/config/activation_storing.py` | New. `ActivationStoringConfig` dataclass, defaults, post-init validation. |
| `vllm/config/activation_storing_types.py` | New. `ActivationStoringSpec`, `HookLayerSelector`, `PositionSelector`, expansion + slugging helpers. |
| `vllm/config/vllm.py` | Wire `activation_storing_config: ActivationStoringConfig \| None` into `VllmConfig`. |
| `vllm/engine/arg_utils.py` | Add `--activation-storing*` CLI flags; populate `ActivationStoringConfig`. |
| `vllm/sampling_params.py` | Add `activation_storing: ActivationStoringSpec \| None` field (no runtime use yet). |
| `vllm/entrypoints/openai/activation_storing_validation.py` | New. Pure function that takes a spec + `VllmConfig` + request context, returns resolved spec or typed error. |
| `tests/entrypoints/openai/test_activation_storing_protocol.py` | Unit tests for selector expansion, slugging, TP/PP/layer-range/prefix-cache rejection, byte budget. |

**Done when.**
- `uv run pytest tests/entrypoints/openai/test_activation_storing_protocol.py` green.
- Selector expansion covered: `"all"`, explicit list, `{layers, ranges}` mixed form, inclusive-range semantics, out-of-range rejection.
- Slugging matches spec (`re.sub(r'[^a-zA-Z0-9._-]', '_', ...)`, `..`/leading-`/`/length-256 rejected).
- CLI flags parse and round-trip through `VllmConfig`; unset root leaves `activation_storing_config = None`.
- **No model-runner changes.** If a reviewer sees a touch in `vllm/v1/worker/*`, kick it back to phase 3 or 4.

**Parallelizable with:** P2.

**Expected conflicts:** none with P2. Minor conflict with P5 on
`sampling_params.py` — P1 reserves the field slot, P5 populates it
from the entrypoint.

---

## Phase 2 — Writer pool

**Branch:** `feat/activation-storing-writer`

**Goal.** Land the filesystem writer as a standalone, testable module.
No dependency on the runner, scheduler, or any vLLM config other than
the root path.

**Files.**

| File | Change |
|---|---|
| `vllm/v1/worker/activation_writer.py` | New. `ActivationWriter`, `WriteTask`, `FinalizeTask`, partitioned thread pool, FD cache with LRU eviction, atomic `.tmp` → final rename, fsync-before-rename for `.bin` and `.json`. |
| `tests/v1/capture/test_activation_writer.py` | New. Unit tests against `tmp_path`: append ordering per `(req_id, layer, hook)`, atomic rename, partial_error on disk-full simulation, collision policies (`overwrite` / `error` / `suffix`), graceful shutdown drain. |

**Done when.**
- Writer accepts tasks, writes `.bin.tmp`, appends on subsequent tasks
  for the same key, fsyncs, renames atomically on finalize.
- Partitioning by `hash(request_id) % num_threads` proven: interleaved
  append tasks for two req_ids never cross threads.
- Disk-full / permission-denied paths return structured `WriteError`;
  the pool keeps draining the queue rather than deadlocking.
- Graceful shutdown drains with a bounded grace period; leftover tasks
  surface as `"error"` with a shutdown message.
- `uv run pytest tests/v1/capture/test_activation_writer.py` green.

**Parallelizable with:** P1.

**Expected conflicts:** none. This module is net-new.

---

## Phase 3 — Capture manager + custom op

**Branch:** `feat/activation-storing-capture`

**Depends on:** P1 (uses `ActivationStoringSpec` and the expansion
helpers).

**Goal.** Land the in-process plumbing that translates a spec into a
per-step plan, exposes the custom op, and piggybacks on
`apply_layer_steering`. Still no runner wiring — tests drive the
manager directly.

**Files.**

| File | Change |
|---|---|
| `vllm/model_executor/layers/activation_capture.py` | New. `ActivationCaptureManager`, `StepCapturePlan`, `CapturePositionEntry`, module-global `_ACTIVE_CAPTURE_MANAGER`, `maybe_capture_residual`, `direct_register_custom_op("capture_residual", ...)` with fake impl. |
| `vllm/model_executor/layers/steering.py` | One-line edit: `apply_layer_steering` calls `maybe_capture_residual(hidden_states, layer_idx, hook_name)` **before** adding the steering vector. Invariant 2. |
| `tests/v1/capture/test_activation_capture_manager.py` | New. Plan building against a fake `input_batch`, index_select correctness, `(layer, hook)` skipping when no request wants it, cold-path `None` check. |
| `tests/v1/capture/test_capture_custom_op.py` | New. Custom op registration, fake impl shape agreement, `torch.compile` constant-folding smoke test when manager is `None`. |

**Done when.**
- Fake input_batch + fake `ActivationStoringSpec` roundtrips through
  plan builder and produces correct `gather_indices` and
  `scratch_gpu` layout.
- `torch.compile` trace under `_ACTIVE_CAPTURE_MANAGER = None` contains
  no `capture_residual` ops (invariant 3).
- `apply_layer_steering` still passes existing steering tests
  unchanged — the edit is a guarded no-op when the manager is `None`.
- Does **not** touch `gpu_model_runner.py`. That's phase 4.

**Parallelizable with:** P2 (after P1 lands).

**Expected conflicts:** `steering.py` has one touch point; if P3 and
an unrelated steering PR both edit `apply_layer_steering`, resolve in
favor of keeping `maybe_capture_residual` as the first call in the
function body.

---

## Phase 4 — Runner integration

**Branch:** `feat/activation-storing-runner`

**Depends on:** P2 and P3.

**Goal.** End-to-end capture path working: manager gets plans built
per step, scratch copies to pinned CPU, writer pool receives tasks,
finalization fires atomic rename, status propagates through the
engine core output. This is the first phase where the feature is
reachable (via `SamplingParams.activation_storing` directly in
offline tests) even though the OpenAI entrypoint is still phase 5.

**Files.**

| File | Change |
|---|---|
| `vllm/v1/worker/gpu_model_runner.py` | New `_prepare_activation_storing_step` (registers new requests, builds plan, assigns plan to manager), `_finalize_activation_storing_step` (pinned-CPU copy, single accelerator sync, enqueue writer tasks). Call sites next to existing steering-buffer update. |
| `vllm/v1/outputs.py` | `ModelRunnerOutput.capture_status` / `capture_paths` / `capture_error` fields. |
| `vllm/v1/engine/__init__.py` | `EngineCoreOutput.capture_status` / `capture_paths` / `capture_error`. |
| `vllm/v1/engine/output_processor.py` | Thread capture status off `EngineCoreOutput` onto downstream output objects (stubbed destination until P5 wires `RequestOutput.activation_storage`). |
| `tests/v1/capture/test_runner_integration.py` | New. Offline `LLM(...)` with `activation_storing=...` + `SamplingParams(activation_storing=...)`, single-step and multi-step capture, last_prompt/all_prompt/all_generated/all/explicit, partial_error on forced writer failure, atomic file visibility. |
| `tests/v1/capture/test_finalization.py` | New. Sidecar payload assembly, `.tmp` → final rename ordering, capture_status propagation. |

**Done when.**
- Offline capture run on a tiny model produces the exact filesystem
  layout the spec describes, including dtype segment and sidecar JSON
  with all required fields.
- Multi-step captures (`"all_generated"`, `"all"`) append correctly
  and finalize into a single coherent `.bin`.
- Forced writer failure (e.g., unwritable tag dir) yields
  `capture_status = "partial_error"` on the `EngineCoreOutput` without
  aborting text generation (invariant 7).
- TP/PP > 1 rejected at engine init if `activation_storing_config` is
  set (invariant 9); verified by an assert with a clear message.
- Existing non-capture tests unaffected.

**Parallelizable with:** none until P2+P3 land. Once `EngineCoreOutput`
capture fields are stable, P5 can start in parallel.

**Expected conflicts:** `gpu_model_runner.py` is heavily touched by
other in-flight work. Keep the phase-4 diff surgical — one prepare
call, one finalize call, both adjacent to the steering update block.
If a conflicting PR lands first, rebase rather than splitting further.

---

## Phase 5 — Protocol and response threading

**Branch:** `feat/activation-storing-protocol`

**Depends on:** P1 (for `ActivationStoringSpec`) and P4 (for
`EngineCoreOutput.capture_*`). Can start as soon as the P4
`EngineCoreOutput` contract is merged or stable on a branch.

**Goal.** Surface the feature through the OpenAI-compatible entrypoint
and the offline `RequestOutput`. Admission validation (from P1) runs
here at the entrypoint boundary.

**Files.**

| File | Change |
|---|---|
| `vllm/entrypoints/openai/chat_completion/protocol.py` | Add `activation_storing: ActivationStoringSpec \| None` to the chat request model; add `ActivationStorageResponse` and the `activation_storage` field. |
| `vllm/entrypoints/openai/completion/protocol.py` | Same for legacy completions. |
| `vllm/entrypoints/openai/chat_completion/serving.py` | Call admission validator, attach spec to `SamplingParams`, thread `activation_storage` onto the response (final SSE frame for streaming). |
| `vllm/entrypoints/openai/completion/serving.py` | Same for legacy completions. |
| `vllm/outputs.py` | `RequestOutput.activation_storage: ActivationStorageResponse \| None`. |
| `vllm/v1/engine/output_processor.py` | Finish the wiring stubbed in P4: populate `RequestOutput.activation_storage` from the engine core output's capture fields. |
| `tests/entrypoints/openai/test_activation_storing_e2e.py` | New. `extra_body` happy path, admission rejection surfaces HTTP 400 with clear error, streaming surfaces `activation_storage` in the final frame's usage block, `status = "not_requested"` when the field is absent. |

**Done when.**
- A curl/httpx request with `extra_body.activation_storing = {...}`
  returns `activation_storage` pointing at the on-disk files.
- Streaming flow: tokens arrive normally over SSE; final frame carries
  the `activation_storage` pointer.
- Admission failures (TP>1, unknown layer, prefix-cache position,
  etc.) return HTTP 400 with a descriptive error — no silent drops.
- Requests with no `activation_storing` field return
  `"status": "not_requested"` and incur zero disk I/O.

**Parallelizable with:** P4 once the `EngineCoreOutput` contract is
stable. Otherwise must come after.

**Expected conflicts:** `serving.py` in both chat and completion sees
frequent edits; keep diffs narrow and rebase aggressively.

---

## Post-phase checklist

These are required before declaring the feature "done" on the
`feat/activation-storing` branch, but they are not blocking for any
individual phase PR:

- [ ] `docs/features/activation_storing.md` — spec already written.
      Update if any phase's implementation drifts from it.
- [ ] `docs/OVERVIEW.md` — add a Features Index entry pointing at the
      spec and listing entry points / depends_on.
- [ ] `CHANGELOG` entry (if the project keeps one).
- [ ] End-to-end smoke test against a real model + NFS mount, not just
      `tmp_path`, before the branch merges to `main`.
- [ ] Confirm invariant 3 on a production-sized model: when
      `--activation-storing` is unset, `torch.compile`'d graphs have
      no `capture_residual` ops.

## Things to confirm with the user before starting P1

- **Model coverage.** The spec piggybacks on `apply_layer_steering`,
  so only steering-instrumented models are capture-capable. Is that
  the final plan or do we want a fallback path for non-steering
  models in v1?
- **Collision policy default.** Spec says `overwrite`. Is that still
  what we want given probe-training workflows that often rerun the
  same `request_id`?
- **Byte cap default.** Spec says `0` (unbounded). Do we want a safer
  default (e.g., 1 GiB) to avoid first-use foot-guns?
- **Tests location.** `tests/v1/capture/` vs. `tests/v1/worker/` vs.
  `tests/activation_storing/` — project convention call.
