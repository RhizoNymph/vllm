# Capture Consumers — Implementation Roadmap

Companion to [`design.md`](design.md). That doc is the design spec;
this doc is the execution plan.

## Context and current state

The current state of `feat/capture-consumers` is:

- **Phase 1 of activation storing** shipped `vllm/config/activation_storing.py`,
  `vllm/config/activation_storing_types.py`,
  `vllm/entrypoints/openai/activation_storing_validation.py`, and the
  `SamplingParams.activation_storing` field.
- **Phase 2** shipped `vllm/v1/worker/activation_writer.py` — the
  thread-pool-backed filesystem writer.
- **Phase 3** shipped `vllm/model_executor/layers/activation_capture.py`
  (the capture manager + custom op) and added `mlp_in` / `mlp_out`
  hook points in a follow-up.
- **Phase 4** wired the capture manager into `GPUModelRunner` and
  added `CaptureResult` + `RequestOutput.activation_storage`.
- **Phase 5** wired the OpenAI entrypoint protocol + serving.

This roadmap covers a **refactor + extension** that replaces the
activation storing feature with a pluggable capture consumer
framework. The filesystem writer becomes the first built-in consumer;
third-party consumers can register via entry points; driver-side
consumers are first-class citizens. **No backward compatibility is
preserved on the API surface.** The on-disk format from the
filesystem consumer is preserved byte-for-byte.

Read `design.md` before picking up any phase — it defines the types,
the protocol, the config surface, and the invariants that every phase
must respect. The roadmap assumes you know the design.

## Guiding constraints

- **Each phase ships as its own PR** off `feat/capture-consumers`.
- **Phases marked "parallel with X" can run concurrently** in sibling
  worktrees; file-level overlap is minimal by construction.
- **No phase breaks test suites for phases that haven't landed yet.**
  During the refactor, legacy tests may be rewritten in the same PR
  they're invalidated by.
- **The on-disk format produced by the filesystem consumer must
  match the old `ActivationWriter` output byte-for-byte.** This is
  the one backward-compat guarantee — scripts reading existing
  captures keep working.
- **Invariants 1–10 from `design.md`** are binding on every phase.
  Reviewers should reject any phase that violates them.

## Phase map

```
                         ┌────────────────┐
                         │ A: core types  │
                         └────────┬───────┘
                                  │
      ┌──────────┬────────────────┼────────────────┬──────────┐
      │          │                │                │          │
      ▼          ▼                ▼                ▼          ▼
  ┌───────┐ ┌────────┐    ┌────────────┐     ┌────────┐ ┌──────────┐
  │ B:mgr │ │ C: fs  │    │ E: config  │     │ G:     │ │ H:       │
  │ (add) │ │ (add)  │    │ (add)      │     │ driver │ │ logging  │
  │       │ │        │    │            │     │ (add)  │ │ + docs   │
  └───┬───┘ └────┬───┘    └─────┬──────┘     └───┬────┘ └─────┬────┘
      │          │              │                │            │
      │          │              │                │            │
      └─────┬────┴──────────────┘                │            │
            │                                    │            │
            ▼                                    │            │
    ┌────────────────┐                           │            │
    │ D: runner      │                           │            │
    │ integration    │◄──────────────────────────┘            │
    │ (cutover)      │                                        │
    └────────┬───────┘                                        │
             │                                                │
             │        ┌─────────────────┐                     │
             │        │ F: protocol     │                     │
             │        │ + entrypoint    │                     │
             │        │ (cutover)       │                     │
             │        └────────┬────────┘                     │
             │                 │                              │
             └────────┬────────┴──────────────────────────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │ I: cleanup +        │
            │ legacy removal      │
            │ (all deletions)     │
            └─────────────────────┘
```

## Parallelism strategy

The roadmap is structured so most phases are **strictly additive**:
they create new files and new code paths alongside the existing
activation-storing code without modifying or deleting anything. The
old `ActivationWriter`, `ActivationCaptureManager`, `SamplingParams.activation_storing`
field, and `--activation-storing*` CLI flags all stay alive during
the refactor. Deletions are deferred to a single cleanup phase at the
end (Phase I).

This unlocks substantially more parallelism than a "move-and-delete
in each phase" structure would. The trade-off is that the tree has
two managers, two writers, two configs, and two `SamplingParams`
fields coexisting for the duration of the refactor — visibly ugly
but temporary.

**Three bands of phases:**

1. **Foundation (sequential):** Phase A must land first. It adds the
   new subpackage `vllm/v1/capture/` with types, protocols, and the
   registry. Zero production-code impact; nothing imports it yet.

2. **Additive parallel (5-way fan-out after A):** Phases B, C, E, G,
   and H all run concurrently in separate worktrees. Each creates
   new files under `vllm/v1/capture/` (plus entry-point registration
   in `pyproject.toml`). None of them touches the existing
   activation-storing production code paths. File-level conflicts
   between them are limited to additions in `pyproject.toml` (C, G,
   H each add entry points) and are trivially resolvable on rebase.

3. **Cutover (after B + C + E):** Phases D and F perform the actual
   cutover — D rewires `GPUModelRunner` to use the new
   `CaptureManager` + consumer list from the new config; F swaps
   `SamplingParams.activation_storing` → `SamplingParams.capture`
   and rewires the OpenAI entrypoint. D and F can run in parallel
   with each other because they touch different files (D: runner,
   scheduler, engine-core output plumbing; F: sampling params,
   outputs, entrypoint protocol). D carries a small compat shim
   that reads `sampling_params.capture` via `getattr` so it can
   land before F if needed; if F lands first, D picks up the new
   field cleanly on rebase.

4. **Cleanup (sequential, last):** Phase I deletes every file and
   field the additive phases left alive. Pure-deletion PR; big
   diff, mechanical review (confirm every deletion has no remaining
   callers via grep).

**Peak concurrency after A lands: 5 simultaneous worktrees.**

**Merge order constraints** (authoring can happen in parallel
regardless of these):

- A must merge before anything else.
- B, C, E can merge in any order after A.
- G, H can merge in any order after A. (G and H don't block D or F.)
- D requires B + C + E to be merged (runner needs the new manager,
  new consumers, and new config).
- F requires A + E (protocol needs the new config field on
  `VllmConfig` so the serving layer can look up registered
  consumers via the registry).
- D and F can merge in either order; whichever lands second rebases
  cleanly because they touch disjoint files.
- I requires every other phase to be merged.

## Worktree layout

Same in-tree convention as the existing activation-storing phases.
Worktrees live at `worktrees/<branch-name>` inside the project tree
(so the sandbox's `.` write area covers them) and each branch is cut
from the appropriate base — most phases cut from `feat/capture-consumers`
directly, cutover phases cut from a post-additive base that has B +
C + E merged.

| Phase | Branch | Cut from | Fork point |
|---|---|---|---|
| A | `feat/capture-consumers-types` | `feat/capture-consumers` | First — blocks everything |
| B | `feat/capture-consumers-manager` | `feat/capture-consumers` + A | Parallel band 1 |
| C | `feat/capture-consumers-filesystem` | `feat/capture-consumers` + A | Parallel band 1 |
| E | `feat/capture-consumers-config` | `feat/capture-consumers` + A | Parallel band 1 |
| G | `feat/capture-consumers-driver` | `feat/capture-consumers` + A | Parallel band 1 |
| H | `feat/capture-consumers-docs` | `feat/capture-consumers` + A | Parallel band 1 |
| D | `feat/capture-consumers-runner` | `feat/capture-consumers` + B + C + E | Cutover band |
| F | `feat/capture-consumers-protocol` | `feat/capture-consumers` + A + E | Cutover band (parallel with D) |
| I | `feat/capture-consumers-cleanup` | `feat/capture-consumers` + all others | Last |

**Practical setup:** when A merges, spin up 5 sibling worktrees for B,
C, E, G, H. Each branches from `feat/capture-consumers` (which now
has A merged). Work them in parallel. When B + C + E have merged,
spin up a worktree for D, branching from `feat/capture-consumers`
(which now has A, B, C, E merged). F can spin up as soon as E has
merged, branching from that same updated tip. When D and F have both
merged, spin up I.

---

## Phase A — Core types and framework skeleton

**Branch:** `feat/capture-consumers-types`
**Depends on:** nothing
**Parallel with:** nothing (must land first)
**Reviewer burden:** low — pure additive, no runtime effect

### Goal

Land the framework types + protocol + registry stubs in a new
subpackage `vllm/v1/capture/`. Zero runtime effect on the existing
activation storing feature. Everything in this phase is importable
but not yet used by any production code path.

### Files to create

| File | Status | Contents |
|---|---|---|
| `vllm/v1/capture/__init__.py` | NEW | Re-exports the public API: `CaptureKey`, `CaptureChunk`, `CaptureFinalize`, `CaptureResult`, `CaptureStatus`, `CaptureSpec`, `CaptureContext`, `CaptureSink`, `CaptureConsumer`, `HookName`, `PositionSelector`, `CaptureValidationError`, `UnknownCaptureConsumerError`. |
| `vllm/v1/capture/types.py` | NEW | Dataclasses and type aliases for `CaptureKey`, `CaptureChunk`, `CaptureFinalize`, `CaptureResult`, `CaptureStatus`, `CaptureSpec`, `CaptureContext`, `HookName`, `PositionSelector`. Exactly as specified in `design.md` § "Core types". |
| `vllm/v1/capture/sink.py` | NEW | `CaptureSink` `Protocol`. |
| `vllm/v1/capture/consumer.py` | NEW | `CaptureConsumer` ABC with `location`, `required_sidecar_fields`, `reads_client_spec` class attributes, `__init__`, `global_capture_spec()`, `validate_client_spec()`, `on_capture()`, `on_error()`, `shutdown()`. Plus an internal `_BatchedAdapter` class that implements `CaptureSink` on top of a `CaptureConsumer` instance (per-key chunk accumulation + finalize-triggered `on_capture` call). |
| `vllm/v1/capture/errors.py` | NEW | `CaptureValidationError(ValueError)`, `UnknownCaptureConsumerError(ValueError)`. |
| `vllm/v1/capture/registry.py` | NEW | Entry-point discovery via `importlib.metadata.entry_points(group="vllm.capture_consumers")`. `load_consumer_class(name) -> type[CaptureConsumer]`. `build_consumer(name, vllm_config, params) -> CaptureConsumer`. Caches entry-point resolution for the engine lifetime. |
| `tests/v1/capture/__init__.py` | NEW | Empty package marker. |
| `tests/v1/capture/test_types.py` | NEW | Unit tests for the dataclasses — construction, defaults, equality. No torch dependency. |
| `tests/v1/capture/test_consumer_base.py` | NEW | Tests for `_BatchedAdapter` — per-key accumulation, finalize triggers `on_capture`, out-of-order chunks are reassembled in `row_offset` order, error in `on_capture` produces `CaptureResult(status="error")`. Uses a fake `CaptureConsumer` subclass. |
| `tests/v1/capture/test_registry.py` | NEW | Tests for entry-point loading with a fake entry-point, `UnknownCaptureConsumerError` on bad name, instance construction via `build_consumer`. |

### Done when

- `.venv/bin/python -m pytest tests/v1/capture/ -v` is fully green.
- `pre-commit run --all-files` clean on changed files.
- `git diff feat/capture-consumers --name-status` shows **only new
  files under `vllm/v1/capture/` and `tests/v1/capture/`**. No
  changes to any existing file.
- Nothing under `vllm/v1/capture/` is imported by production code
  yet. It's all dead but correct.
- The types + protocol + registry are sufficient to implement a
  hello-world consumer in a test file — prove it with one test.

### Design risks

1. **`CaptureConsumer` ABC vs `Protocol`**. The spec uses `ABC` for
   user-facing ergonomics (subclasses don't have to redeclare
   methods that have default implementations). Keep it as ABC.
   `CaptureSink` is a `Protocol` because consumers don't subclass
   it — they implement it structurally (or via the batched adapter).

2. **`_BatchedAdapter` ownership**. Does it live in `consumer.py` as
   a private helper, or in `sink.py`? Put it in `consumer.py` —
   it's an implementation detail of how `CaptureConsumer` fulfills
   the `CaptureSink` contract. `sink.py` stays protocol-only.

3. **Type imports and torch**. `CaptureChunk.tensor: torch.Tensor`
   means `types.py` imports torch. That's fine — this module is
   torch-aware by design (the old
   `activation_storing_types.py` was torch-free because it
   pre-dated the capture manager, which is now moot). Unit tests
   for the protocol-level logic still run fast because they use
   small CPU tensors.

---

## Phase B — Capture manager (additive)

**Branch:** `feat/capture-consumers-manager`
**Depends on:** A
**Parallel with:** C, E, G, H (all in parallel band 1)
**Reviewer burden:** medium — creates a new manager alongside the old one

### Goal

Create the new `CaptureManager` in `vllm/v1/capture/manager.py` —
supports multiple consumers with global specs and per-request client
specs, dispatches captures to the right consumer(s) on finalize. The
new manager is a **new file** added alongside the existing
`ActivationCaptureManager`; Phase B does **not** touch
`vllm/model_executor/layers/activation_capture.py`. The old manager
stays alive, still used by the runner, until Phase D cuts over. Phase
I deletes the old manager.

### Files to create / edit

| File | Status | Change |
|---|---|---|
| `vllm/v1/capture/manager.py` | NEW | `CaptureManager` class. Holds `tuple[CaptureSink, ...]` of active consumers; tracks `(consumer_index, global_spec)` for each; tracks per-request `dict[consumer_name, CaptureSpec]` for client-opt-in specs; plan builder computes the union of global + per-request specs; dispatch at finalize routes captures to every consumer whose spec matches. |
| `vllm/v1/capture/plan.py` | NEW | `StepCapturePlan`, `CapturePositionEntry`, `CaptureBatchView`. Structurally similar to the existing types in `activation_capture.py`, but extended with a `consumer_mask: int` (uint64 bitset) on each `CapturePositionEntry` — records which consumers want this row. This is a fresh implementation; Phase B does **not** move code out of `activation_capture.py`. Duplication is deliberate and temporary. |
| `tests/v1/capture/test_manager.py` | NEW | Tests for the new manager: register multiple consumers with overlapping global specs, per-request client specs, build a plan against a fake batch view, verify the plan's gather_indices match the union, verify finalize dispatches to the correct consumers. Covers spec union, dispatch routing, consumer isolation on exceptions. Uses fake sinks (`MagicMock`-based) to avoid dependency on concrete consumers. |
| `tests/v1/capture/test_plan.py` | NEW | Unit tests for `StepCapturePlan`, `CaptureBatchView`, `CapturePositionEntry`, `consumer_mask` bit manipulation. |

### What B deliberately does NOT do

- **Does not touch `vllm/model_executor/layers/activation_capture.py`.**
  The old manager stays there, still used by the runner.
- **Does not touch `vllm/v1/worker/gpu_model_runner.py`.** The runner
  keeps calling the old manager.
- **Does not introduce a legacy adapter shim.** There's nothing to
  bridge — the new manager is a standalone type that nothing imports
  yet until Phase D.
- **Does not touch any existing tests.** The existing
  `tests/v1/worker/activation_storing/test_activation_capture_manager.py`
  keeps running unchanged against the still-alive old manager.

### Done when

- `.venv/bin/python -m pytest tests/v1/capture/test_manager.py tests/v1/capture/test_plan.py -v` is green.
- `.venv/bin/python -m pytest tests/v1/worker/activation_storing/ -v` is **unchanged green** — no existing tests were touched.
- `.venv/bin/python -m pytest tests/v1/worker/test_steering_manager.py tests/model_executor/layers/test_steering_op.py -v` unchanged green.
- `pre-commit run --all-files` clean.
- `git diff feat/capture-consumers --name-status` shows **only new
  files under `vllm/v1/capture/` and `tests/v1/capture/`**. Zero
  edits to existing files.

### Design risks

1. **Consumer mask representation.** The plan's entries need to
   know which consumers care about each row. Options: (a) list of
   consumer names per entry; (b) bitset where each bit is a
   consumer index; (c) separate dispatch tables keyed by
   `(layer, hook)`. Pick (b) — it's compact and fast to test for
   membership. Max 64 consumers per instance (one uint64); if you
   need more, extend to bytes.

2. **Per-request-spec lifetime.** The manager stores per-request
   client specs alongside global specs. When the request finalizes,
   both are cleared. Make sure the per-request entries don't leak.
   Add a test that creates 1000 requests with per-request specs and
   verifies the internal dict is empty after they all finalize.

3. **Thread safety of the consumer list.** The manager's consumer
   list is set once at engine init and read on every forward step.
   Reads don't need locking if the list is never mutated. Enforce
   immutability: `_consumers: tuple[CaptureSink, ...]` instead of
   list. Document in the manager's docstring that adding consumers
   mid-engine is not supported.

---

## Phase C — Filesystem consumer (additive)

**Branch:** `feat/capture-consumers-filesystem`
**Depends on:** A (for `CaptureSink` + types)
**Parallel with:** B, E, G, H (all in parallel band 1)
**Reviewer burden:** medium — new consumer class wrapping existing writer

### Goal

Create the filesystem consumer as a new class in `vllm/v1/capture/consumers/filesystem/`
that implements `CaptureSink` directly (streaming, not batched). The
consumer wraps the **existing** `ActivationWriter` by importing it from
its current location (`vllm.v1.worker.activation_writer`) — no file
moves. Register the consumer via entry point in vLLM's own
`pyproject.toml`. The old `ActivationWriter` keeps serving the old
runner path during B–H. Phase I moves the writer file into the
`filesystem/` subdirectory and deletes the old path.

### Files to create / edit

| File | Status | Change |
|---|---|---|
| `vllm/v1/capture/consumers/__init__.py` | NEW | Empty package marker. |
| `vllm/v1/capture/consumers/filesystem/__init__.py` | NEW | Re-exports `FilesystemConsumer`, `FilesystemCaptureRequest`, `FilesystemValidationError`. |
| `vllm/v1/capture/consumers/filesystem/consumer.py` | NEW | `FilesystemConsumer` class. `location = "worker"`. `reads_client_spec = True`. `global_capture_spec()` returns `None`. `validate_client_spec()` delegates to `validation.validate_filesystem_request`. Implements `CaptureSink` directly by instantiating a `vllm.v1.worker.activation_writer.ActivationWriter` internally and forwarding `submit_chunk` → `WriteTask`, `submit_finalize` → `FinalizeTask`. **Imports the existing writer by its current path**; Phase I is where the writer's file actually moves under `filesystem/`. |
| `vllm/v1/capture/consumers/filesystem/validation.py` | NEW | Re-export + thin wrapper over `vllm.entrypoints.openai.activation_storing_validation.validate_activation_storing`. Exposes `validate_filesystem_request(raw, ctx) -> CaptureSpec` on top of the existing validator. Phase I moves the underlying validator's implementation here. |
| `vllm/v1/capture/consumers/filesystem/types.py` | NEW | `FilesystemCaptureRequest` dataclass (the per-request client spec shape). Plus `FilesystemConsumerParams` for consumer-level config (`root`, `writer_threads`, etc.). |
| `pyproject.toml` | EDIT | Add entry-point registration: `[project.entry-points."vllm.capture_consumers"] filesystem = "vllm.v1.capture.consumers.filesystem:FilesystemConsumer"`. |
| `tests/v1/capture/consumers/__init__.py` | NEW | Empty package marker. |
| `tests/v1/capture/consumers/filesystem/__init__.py` | NEW | Empty package marker. |
| `tests/v1/capture/consumers/filesystem/test_consumer.py` | NEW | Tests for `FilesystemConsumer`: submit chunks, finalize, verify files + sidecar, verify atomic visibility. Uses `tmp_path`. Exercises `validate_client_spec` through the consumer. Golden test: produces byte-identical `.bin` + `.json` output to what the old `ActivationWriter` produces directly. |

### What C deliberately does NOT do

- **Does not delete `vllm/v1/worker/activation_writer.py`.** The
  existing writer stays alive, imported by the still-unchanged
  runner path. Phase I moves it.
- **Does not delete `vllm/entrypoints/openai/activation_storing_validation.py`.**
  Phase I moves it.
- **Does not touch `vllm/v1/worker/gpu_model_runner.py`** or any
  engine code. The runner still uses the old writer directly.
- **Does not touch any existing tests.** The existing
  `tests/v1/worker/activation_storing/test_activation_writer.py`
  keeps running against the untouched writer.

### Done when

- `.venv/bin/python -m pytest tests/v1/capture/consumers/filesystem/ -v` fully green.
- `.venv/bin/python -m pytest tests/v1/worker/activation_storing/test_activation_writer.py -v` **unchanged green** — no existing tests were touched.
- `pre-commit run --all-files` clean.
- `FilesystemConsumer` produces byte-identical on-disk output to
  what `ActivationWriter` produces directly, verified by a golden
  test.
- After reinstalling the vLLM package,
  `python -c "import importlib.metadata; print([ep.name for ep in importlib.metadata.entry_points(group='vllm.capture_consumers')])"`
  lists `filesystem`.
- `git diff feat/capture-consumers --name-status` shows **only new
  files plus a one-line addition to `pyproject.toml`**. Zero edits
  to existing `vllm/` source files.

### Design risks

1. **Streaming sink implementation, not batched.** Do **not**
   subclass `CaptureConsumer`. The filesystem consumer implements
   `CaptureSink` directly because long captures
   (`positions="all"`) can't be buffered in memory. This means
   `FilesystemConsumer` has its own `submit_chunk`, `submit_finalize`,
   `get_result`, and `shutdown` — the batched adapter is skipped
   entirely.

2. **Byte-for-byte compatibility.** The on-disk format must not
   change. Golden test that captures one request through
   `FilesystemConsumer` and compares the `.bin` + `.json` output
   byte-for-byte against the Phase 2 `ActivationWriter` output.

3. **Import path coupling.** `FilesystemConsumer` imports
   `ActivationWriter` from `vllm.v1.worker.activation_writer` during
   Phases C–H. Phase I's move is what changes the import path.
   Document this in the `consumer.py` docstring so anyone reading
   it during the refactor window understands why the import looks
   wrong.

---

## Phase D — Runner integration (cutover)

**Branch:** `feat/capture-consumers-runner`
**Depends on:** B (for `CaptureManager`), C (for `FilesystemConsumer`), E (for `capture_consumers_config` on `VllmConfig`), G (for `build_consumers` registry helper — optional, see design risks)
**Parallel with:** F (touches disjoint files)
**Reviewer burden:** high — touches `gpu_model_runner.py` and the engine output pipeline

### Goal

Rewire `GPUModelRunner` to use `CaptureManager` (from B) with a
consumer list sourced from `capture_consumers_config` (added by E)
via the registry (from A). This is the first phase that actually
cuts the runner over — after D lands, the old
`ActivationCaptureManager`, `ActivationWriter`, and
`_register_activation_storing_request` are dead code paths (still
imported somewhere, but no longer exercised). Phase I deletes them.

The `SamplingParams` field rename is Phase F's responsibility;
Phase D reads from `sampling_params.capture` via `getattr` so it
works regardless of whether F has landed yet.

### Files to edit

| File | Status | Change |
|---|---|---|
| `vllm/v1/worker/gpu_model_runner.py` | EDIT | Add `self._capture_manager: CaptureManager \| None` alongside the old `self._activation_capture_manager` (which is no longer populated). In `__init__`, if `vllm_config.capture_consumers_config is not None`, construct the new manager via `registry.build_consumers(vllm_config)` (from G, or a minimal local equivalent if G hasn't merged yet). Remove the old `self._activation_capture_manager` + `self._activation_writer` construction. Replace `_register_activation_storing_request` with `_register_capture_request` that walks `getattr(sampling_params, "capture", None)` and calls each consumer's `validate_client_spec`. Replace `_prepare_activation_storing_step` / `_finalize_activation_storing_step` with `_prepare_capture_step` / `_finalize_capture_step` that dispatch to the new manager. |
| `vllm/v1/outputs.py` | EDIT | `ModelRunnerOutput.capture_results: dict[req_id, CaptureResult]` becomes `dict[req_id, dict[consumer_name, CaptureResult]]`. Nested shape. |
| `vllm/v1/core/sched/scheduler.py` | EDIT | Update the 2-line fan-out that threads `capture_results` from `ModelRunnerOutput` into `EngineCoreOutput` to handle the new nested dict shape. |
| `vllm/v1/engine/__init__.py` | EDIT | `EngineCoreOutput.capture_result: CaptureResult \| None` → `EngineCoreOutput.capture_results: dict[str, CaptureResult]`. |
| `vllm/v1/engine/output_processor.py` | EDIT | Thread `capture_results` dict through `RequestState.make_request_output`. Set it on `RequestOutput` via `setattr` or a new kwarg — the final field name is added to `RequestOutput` in Phase F; until F lands, use `setattr` to attach the dict as an attribute. |
| `vllm/v1/engine/core.py` | EDIT | Add the TP/PP assertion for `capture_consumers_config`: raise `ValueError` if `capture_consumers_config is not None` and TP>1 or PP>1. Leave the existing `activation_storing_config` assertion in place — it still fires for operators using the old config. Phase I removes it. |
| `tests/v1/capture/test_runner_integration.py` | NEW | Offline `LLM(..., capture_consumers=[...])` tests: filesystem consumer via registry, verify the expected files appear, matching the Phase 4 activation-storing behavior byte-for-byte. |
| `tests/v1/capture/test_multi_consumer_runner.py` | NEW | Multi-consumer test: filesystem + logging consumers active simultaneously, verify both produce results on `RequestOutput.capture_results`. |

### What D deliberately does NOT do

- **Does not delete `vllm/model_executor/layers/activation_capture.py`'s
  old manager** — the classes still exist (unused), removed in Phase I.
- **Does not delete `vllm/v1/worker/activation_writer.py`** — it's still
  referenced by the import path in Phase C's `FilesystemConsumer`.
  Phase I moves it.
- **Does not touch anything under `vllm/entrypoints/openai/`** — that's
  Phase F.
- **Does not touch `vllm/sampling_params.py` or `vllm/outputs.py`** —
  that's Phase F.

### Done when

- `.venv/bin/python -m pytest tests/v1/capture/test_runner_integration.py tests/v1/capture/test_multi_consumer_runner.py -v` fully green.
- `.venv/bin/python -m pytest tests/v1/worker/test_steering_manager.py tests/model_executor/layers/test_steering_op.py -v` unchanged green.
- Offline `LLM(..., capture_consumers=[...])` with a filesystem
  consumer produces the expected files, matching the Phase 4
  activation-storing behavior byte-for-byte.
- Multi-consumer test: construct an `LLM` with filesystem + logging
  consumers, send one request, verify both consumers report a
  terminal result on `RequestOutput.capture_results`.
- `git diff feat/capture-consumers --name-status` does not touch
  anything under `vllm/entrypoints/openai/` or
  `vllm/sampling_params.py` or `vllm/outputs.py` (those are Phase F).
- `pre-commit run --all-files` clean.

### Design risks

1. **`sampling_params.capture` access via `getattr`.** Phase F is
   what adds the `capture` field to `SamplingParams`. If D lands
   before F, `getattr(sampling_params, "capture", None)` returns
   `None` (because the attribute doesn't exist), and per-request
   client specs are effectively disabled for the old API. Global
   specs still work. When F lands, the field appears and
   per-request client specs start flowing. This asymmetry is OK
   because users transitioning to the new API will land F
   eventually.

2. **Registry helper `build_consumers` sourced from G.** Phase D's
   runner calls `registry.build_consumers(vllm_config)` which is
   created by G. If G hasn't merged when D is built, inline a
   minimal local version of `build_consumers` in `registry.py` that
   only handles worker-side consumers. When G lands, it supersedes
   the local version. Both can coexist — G's `build_consumers` is
   a drop-in replacement of D's placeholder.

3. **TP/PP assertion across multiple consumers.** The new check is
   "if `capture_consumers_config` is non-empty AND TP>1". Error
   message references the configured consumer names. Leave the
   old `activation_storing_config` assertion in place — Phase I
   removes it.

4. **`RequestOutput` attribute via `setattr` during the D→F gap.**
   D needs to attach the `capture_results` dict to `RequestOutput`
   somewhere, but the field is added in F. During the gap, use
   `setattr(request_output, "capture_results", ...)`. When F lands,
   the field exists as a declared class attribute and the `setattr`
   call becomes redundant but harmless. Phase I cleans it up.

---

## Phase E — Config and CLI surface (additive)

**Branch:** `feat/capture-consumers-config`
**Depends on:** A
**Parallel with:** B, C, G, H (all in parallel band 1)
**Reviewer burden:** medium — arg_utils is finicky

### Goal

Add the new `capture_consumers` config surface alongside the
existing `activation_storing` config. Both sets of fields and flags
coexist after Phase E lands. The new fields are wired into
`VllmConfig` but are **not yet read** by the runner — Phase D is
what switches the runner over. Phase I removes the old fields.

### Files to edit

| File | Status | Change |
|---|---|---|
| `vllm/v1/capture/config.py` | NEW | `CaptureConsumersConfig` dataclass: holds an ordered `list[CaptureConsumerSpec]` where each spec is `{name: str, instance_name: str \| None, params: dict[str, Any]}`. Plus a parser that accepts the CLI shorthand `name:key=value,key=value` form. Plus a YAML loader. Plus validation (unique instance names, non-empty `name` field, known consumer names per the registry). |
| `vllm/engine/arg_utils.py` | EDIT | Add `--capture-consumers` (repeatable flag, each occurrence adds one consumer spec) and `--capture-consumer-params` for complex params. Add a corresponding `EngineArgs` field. Construct `CaptureConsumersConfig` from the flags + YAML config at `VllmConfig` construction time. **Do not remove the existing `--activation-storing*` flags** — they stay alive until Phase I. |
| `vllm/config/vllm.py` | EDIT | Add `capture_consumers_config: CaptureConsumersConfig \| None = None` (None when no consumers are configured). Include in `compute_hash()` alongside the existing `activation_storing_config`. **Do not remove `activation_storing_config`.** Both fields coexist. |
| `vllm/config/__init__.py` | EDIT | Add `CaptureConsumersConfig` export. Leave the existing `ActivationStoringConfig` export in place. |
| `tests/v1/capture/test_config.py` | NEW | Tests for `CaptureConsumersConfig` parsing: YAML form, CLI shorthand form, error on duplicate instance names, error on missing `name`, happy path round-trip. |
| `tests/engine/test_arg_utils.py` | EDIT (additive) | Add tests for `--capture-consumers` flag parsing. **Do not remove** existing tests for `--activation-storing*` flags. |

### What E deliberately does NOT do

- **Does not remove any existing `--activation-storing*` flag.** They
  stay in `arg_utils.py` and keep working. Phase I removes them.
- **Does not remove `VllmConfig.activation_storing_config`.** Both
  config fields exist side-by-side. Phase I removes the old field.
- **Does not delete `vllm/config/activation_storing.py`.** Phase I
  deletes it.
- **Does not touch `vllm/config/activation_storing_types.py`.** It
  stays in place with its current contents (filesystem-consumer-
  specific types that will move in Phase I, and per-request spec
  types that are still read by the runner until Phase D).

### Done when

- `.venv/bin/python -m pytest tests/v1/capture/test_config.py -v` green.
- `.venv/bin/python -m pytest tests/engine/test_arg_utils.py -v` green (both old and new flag tests pass).
- `vllm serve --help` shows **both** `--capture-consumers` and the existing `--activation-storing*` flags.
- CLI round-trip: `vllm serve ... --capture-consumers 'filesystem:root=/tmp/foo'` parses into `CaptureConsumersConfig([CaptureConsumerSpec(name="filesystem", params={"root": "/tmp/foo"})])`.
- YAML round-trip: loading the example YAML from `design.md` produces the expected config.
- `git diff feat/capture-consumers --name-status` shows mostly new files, plus additive edits to `arg_utils.py`, `vllm.py`, `__init__.py`. Zero deletions.
- `pre-commit run --all-files` clean.

### Design risks

1. **CLI shorthand escaping.** Params with commas or equals signs
   in values need escaping. Document the escape rules or punt to
   YAML for complex params. Start with "no commas or equals in
   values; use YAML for anything complex" and extend only if users
   complain.

2. **`compute_hash()` includes both fields.** During the additive
   period, `VllmConfig.compute_hash()` hashes both
   `activation_storing_config` and `capture_consumers_config`.
   Engines started with only one configured produce the same hash
   they did before the refactor (the other field is `None`). Engines
   started with both configured get a different hash — that's fine
   because the runner behavior is actually different.

---

## Phase F — Protocol and entrypoint (cutover)

**Branch:** `feat/capture-consumers-protocol`
**Depends on:** A (for types), E (for `VllmConfig.capture_consumers_config` to look up registered consumers at admission time)
**Parallel with:** D (touches disjoint files — D: runner/engine, F: sampling params/outputs/entrypoint)
**Reviewer burden:** medium — touches `SamplingParams`, `RequestOutput`, and entrypoint serving/protocol files

### Goal

Add `SamplingParams.capture` and `RequestOutput.capture_results`
alongside the existing `activation_storing` / `activation_storage`
fields. Both coexist after F lands — old clients keep working via
the old fields (which D no longer honors on the runner side, but the
types stay alive until I), new clients use the new fields. The
OpenAI entrypoint accepts the new `capture` request field and returns
`capture_results` in the response body.

### Files to edit

| File | Status | Change |
|---|---|---|
| `vllm/sampling_params.py` | EDIT | Add `capture: dict[str, Any] \| None = None`. Add a `_validate_capture` helper that checks structural shape only (dict, keys are strings) — per-consumer validation happens at the entrypoint, not in `SamplingParams.__post_init__`. **Do not remove** the existing `activation_storing` field or `_validate_activation_storing` helper. |
| `vllm/outputs.py` | EDIT | Add `RequestOutput.capture_results: dict[str, CaptureResult]` (default factory `dict`). Update `RequestOutput.__init__` to accept the new kwarg. **Do not remove** the existing `activation_storage` field. |
| `vllm/entrypoints/openai/chat_completion/protocol.py` | EDIT | Add `capture: dict[str, Any] \| None` request field on `ChatCompletionRequest`. Add `CaptureResultResponse` pydantic model. Add `capture_results: dict[str, CaptureResultResponse] \| None` to `ChatCompletionResponse` and `ChatCompletionStreamResponse`. **Do not remove** the existing `activation_storing` request field or `ActivationStorageResponse` model. |
| `vllm/entrypoints/openai/completion/protocol.py` | EDIT | Same additive pattern for legacy completions. Imports `CaptureResultResponse` from `chat_completion/protocol.py`. |
| `vllm/entrypoints/openai/chat_completion/serving.py` | EDIT | Add `_admit_capture` that walks `request.capture` (a dict), looks up each consumer by name via the registry (from `vllm_config.capture_consumers_config`), calls the consumer's `validate_client_spec(raw, ctx)` to get a `CaptureSpec`, and mutates `sampling_params.capture[name]` to the validated spec. Any `CaptureValidationError` becomes HTTP 400. Thread `final_res.capture_results` (attached by D's `setattr`) onto the response body's new `capture_results` field. **Do not remove** `_admit_activation_storing` or the old response threading. |
| `vllm/entrypoints/openai/completion/serving.py` | EDIT | Same additive pattern. |
| `tests/v1/capture/test_sampling_params.py` | NEW | Tests for the new `SamplingParams.capture` field shape and structural validation. |
| `tests/entrypoints/openai/test_capture_protocol.py` | NEW | Tests for the new protocol fields: request round-trip, response round-trip, admission failure → HTTP 400. |

### What F deliberately does NOT do

- **Does not remove `SamplingParams.activation_storing`.** Both
  fields coexist. After D has landed, the runner only reads from
  `.capture`, so any value set on `.activation_storing` is silently
  ignored. Phase I removes the field.
- **Does not remove `RequestOutput.activation_storage`.** The old
  field stays as an always-`None` attribute until Phase I removes
  it.
- **Does not remove the old OpenAI request/response fields.** They
  stay in the pydantic models until Phase I.
- **Does not remove `_admit_activation_storing` from the serving
  layer.** It stays as a no-op (since `activation_storing_config`
  is no longer honored by the runner after D) until Phase I.

### Done when

- `.venv/bin/python -m pytest tests/v1/capture/test_sampling_params.py tests/entrypoints/openai/test_capture_protocol.py -v` green.
- `.venv/bin/python -m pytest tests/entrypoints/openai/test_activation_storing_protocol.py -v` **unchanged green** — the existing tests keep running against the still-alive old fields (they may produce no-op behavior since the runner no longer honors the old fields, but pydantic validation and serving-layer code paths still work).
- HTTP round-trip: request with `extra_body.capture = {"filesystem": {...}}` returns `capture_results["filesystem"]` in the response body.
- Admission error (unknown consumer name, invalid client spec) returns HTTP 400 with a descriptive error message.
- `git diff feat/capture-consumers --name-status` shows edits to the files above plus new test files. Zero deletions.
- `pre-commit run --all-files` clean.

### Design risks

1. **Both fields coexist, only one works.** After F lands, a client
   can set both `sampling_params.activation_storing` and
   `sampling_params.capture`. The runner (post-D) only honors
   `.capture`. Document this clearly in the PR body: the old field
   becomes a no-op as of D, and F is what adds the replacement.
   Users should treat D+F as a coordinated breaking change.

2. **Pydantic model additions.** The `capture` field is additive to
   `ChatCompletionRequest`. Existing clients that only set
   `activation_storing` continue to parse successfully — they just
   don't get captures anymore. Document this in the PR body and in
   the migration section of `docs/capture_consumers/design.md`.

3. **`CaptureResultResponse` shape.** The new response field is a
   dict keyed by consumer name. Use `dict[str, CaptureResultResponse]`
   where the inner model is strict. Default to omitting the field
   entirely when it's empty (not `{}`) to minimize payload size for
   requests that didn't capture.

---

## Phase G — Driver bridge (additive)

**Branch:** `feat/capture-consumers-driver`
**Depends on:** A
**Parallel with:** B, C, E, H (all in parallel band 1)
**Reviewer burden:** medium-high — new cross-process code

### Goal

Implement the worker→driver plumbing for `location = "driver"`
consumers. Includes the worker-side queue shim, the driver-side
receiver thread, the instance-passing path in `LLM(...)`, and the
back-pressure / shutdown semantics.

### Files to create / edit

| File | Status | Change |
|---|---|---|
| `vllm/v1/capture/driver_bridge.py` | NEW | `_DriverQueueShim(CaptureSink)` with `location = "worker"`. Holds a `torch.multiprocessing.Queue`. `submit_chunk` / `submit_finalize` serialize the event and put on the queue. `get_result` tracks per-key terminal state based on ack messages from the receiver. `shutdown` joins the queue writer. Plus `_DriverReceiver`: runs a thread in the driver process. Pops events from the queue and invokes the user's consumer's `on_capture` / `on_error`. One receiver thread per driver consumer instance. Plus `install_driver_consumer(consumer, vllm_config) -> _DriverQueueShim`: called during engine startup on the driver side for each `location = "driver"` consumer. Creates the queue, spawns the receiver thread, returns the worker-side shim. |
| `vllm/v1/capture/registry.py` | EDIT (additive) | Add `build_consumers(vllm_config) -> tuple[CaptureSink, ...]`. Walks `vllm_config.capture_consumers_config`, resolves each consumer class via the entry-point registry, and for each class: if `location == "worker"`, constructs the instance directly (in the worker process, via the engine-core IPC that ships class refs); if `location == "driver"`, constructs the instance in the driver process and installs a `_DriverQueueShim` on the worker side via `install_driver_consumer`. Returns the ordered tuple of worker-side sinks for the manager. Phase A already created `registry.py` with entry-point discovery; Phase G adds `build_consumers`. |
| `vllm/entrypoints/llm.py` (or wherever the `LLM(...)` constructor lives) | EDIT | Add `capture_consumers: list[dict \| CaptureConsumer] \| None = None` kwarg. Dict entries go through the registry. `CaptureConsumer` instances must have `location = "driver"`; raise `ValueError` if a worker-side instance is passed. |
| `tests/v1/capture/test_driver_bridge.py` | NEW | Tests: round-trip a chunk through a fake queue + receiver, back-pressure timeout produces `partial_error`, shutdown drains + joins, exception in `on_capture` produces `error` status and continues. Does not require multi-process — use a real `torch.multiprocessing.Queue` but a single-process test with two threads. |
| `tests/v1/capture/test_driver_consumer_e2e.py` | NEW | End-to-end: instantiate an `LLM` with a driver-side callback consumer, send one request, verify the callback fires with the expected tensor. Requires torch + a tiny model (skip on CPU-only machines). |

### What G deliberately does NOT do

- **Does not touch `vllm/v1/engine/core.py`.** The call site that
  invokes `build_consumers` is in the runner (touched by Phase D).
  G just creates the helper; D wires it in.
- **Does not touch `vllm/v1/capture/manager.py`.** The manager takes
  a `tuple[CaptureSink, ...]` at init and doesn't care whether any
  of the sinks are driver-side shims. The driver/worker distinction
  is invisible below the sink interface.
- **Does not touch `vllm/v1/capture/consumer.py`.** `CaptureConsumer`
  already has the `location` class attribute from Phase A; G just
  reads it.

### Done when

- `.venv/bin/python -m pytest tests/v1/capture/test_driver_bridge.py -v` green (single-process).
- `.venv/bin/python -m pytest tests/v1/capture/test_driver_consumer_e2e.py -v` green (on a CUDA box).
- Instance-passing works: `LLM(capture_consumers=[MyDriverConsumer(...)])` installs the consumer correctly.
- Back-pressure works: a slow consumer causes `partial_error` status after the timeout, doesn't block text generation.
- Shutdown works: a dirty shutdown with pending events marks them `error` with a "shutdown" reason, joins threads, returns.
- `pre-commit run --all-files` clean.

### Design risks

1. **`torch.multiprocessing` quirks.** Passing `torch.Tensor` via
   `torch.multiprocessing.Queue` uses shared memory under the hood
   but has platform quirks (fork vs spawn, CUDA IPC,
   macOS `spawn`-only semantics). Test on Linux first; Windows/mac
   are nice-to-have.

2. **Receiver thread lifecycle.** The receiver thread runs for the
   entire `LLM` lifetime. When the engine shuts down, the worker
   pushes a sentinel onto the queue; the receiver sees the sentinel
   and exits. Make sure the receiver is a daemon thread so a crash
   in the driver process doesn't hang on join.

3. **Tensor ownership and GC.** After `submit_chunk(chunk)`, the
   worker may release its reference to `chunk.tensor`. If the
   driver side hasn't read it yet, the shared-memory segment needs
   to stay alive. `torch.multiprocessing.Queue` handles refcounting
   correctly if you use it idiomatically; verify with a stress test
   that doesn't OOM after 10k chunks.

4. **Per-request-result aggregation.** Driver-side consumers produce
   results in the driver process. The per-request
   `capture_results[consumer_name]` needs to come from the driver,
   not the worker. Option A: driver process sends an ack back
   through a result queue, worker stores it, manager reads it on
   finalize. Option B: manager doesn't try to collect driver
   results at finalize — `capture_results[consumer_name].status` is
   `"pending"` until the driver receiver processes the finalize,
   and the client polls the final status separately. Start with
   option A for consistency with worker consumers; option B is
   simpler but creates a weird polling gap.

---

## Phase H — Logging consumer, examples, and docs (additive)

**Branch:** `feat/capture-consumers-docs`
**Depends on:** A
**Parallel with:** B, C, E, G (all in parallel band 1). Can optionally wait for G if the example plugin wants to demo a driver-side consumer, but the minimal plugin is worker-side and needs only A.
**Reviewer burden:** low — mostly docs and examples

### Goal

Ship a reference `LoggingConsumer`, a minimal example plugin package,
and user-facing documentation.

### Files to create / edit

| File | Status | Change |
|---|---|---|
| `vllm/v1/capture/consumers/logging.py` | NEW | `LoggingConsumer(CaptureConsumer)`. `location = "worker"`. Params: `{hooks, positions, level}`. `global_capture_spec()` returns a `CaptureSpec` from params. `on_capture` logs one line per finalized capture. |
| `pyproject.toml` | EDIT | Add `logging` entry point: `logging = "vllm.v1.capture.consumers.logging:LoggingConsumer"`. Note potential rebase conflict with Phase C (which also adds an entry point); both are one-line additions, trivial to resolve. |
| `tests/v1/capture/consumers/test_logging.py` | NEW | Tests for `LoggingConsumer`: construct, `global_capture_spec` correct, `on_capture` logs expected format. |
| `docs/capture_consumers/plugin_authoring.md` | NEW | "How to write a capture consumer plugin" guide. Copies the examples from `design.md` § "Consumer authoring guide", expands with more detail. Structured as a tutorial. |
| `docs/capture_consumers/examples/minimal_plugin/` | NEW | Full working plugin package with `pyproject.toml`, `my_plugin/__init__.py`, and a README showing how to install and use it. Acts as a reference for plugin authors. |
| `docs/OVERVIEW.md` | EDIT | Add a `capture_consumers` entry to the features index. Do **not** mark `activation_storing` as removed yet — that's Phase I's job after the old code is actually gone. |

### What H deliberately does NOT do

- **Does not touch `docs/features/activation_storing.md` or
  `docs/features/activation_storing_roadmap.md`.** Those stay alive
  until Phase I deletes them.
- **Does not modify `design.md`.** Any design-risk resolutions
  discovered during implementation get folded in when Phase I's
  cleanup touches docs.

### Done when

- `.venv/bin/python -m pytest tests/v1/capture/consumers/test_logging.py -v` green.
- The example plugin package in `docs/capture_consumers/examples/minimal_plugin/` is installable with `pip install -e .` and works when referenced in a vLLM config.
- The plugin authoring guide is complete and self-contained.
- `docs/OVERVIEW.md` reflects the removal of `activation_storing` and the addition of `capture_consumers`.

### Design risks

1. **Example plugin package as a doc artifact.** It's not a full
   Python package that ships to PyPI — it's a reference. Make sure
   the structure is self-explanatory and the README walks through
   every file. Could be rewritten as a `docs/capture_consumers/examples/`
   tutorial if the "working package" framing is confusing.

---

## Phase I — Cleanup and legacy removal

**Branch:** `feat/capture-consumers-cleanup`
**Depends on:** B, C, D, E, F, G, H (every phase that produced
additive changes)
**Parallel with:** nothing — last phase
**Reviewer burden:** medium — big diff but mechanical. Every deletion
has to be verified by grep to have no remaining callers.

### Goal

Absorb all the deletions that every additive phase deferred. After
this phase, the tree has a single capture implementation
(`vllm/v1/capture/`), and every trace of `activation_storing` is
gone from production code and tests. This is the phase that actually
moves `ActivationWriter` and its validation file into the
`filesystem/` consumer package.

### Files to delete

**Config / types layer (deferred from E):**
- `vllm/config/activation_storing.py`
- `vllm/config/activation_storing_types.py`

**Manager / writer / validation layer (deferred from B + C):**
- `vllm/model_executor/layers/activation_capture.py` — strip to just
  the custom op + `maybe_capture_residual` + `set_active_capture_manager`
  + `_HOOK_NAME_TO_ID` / `_HOOK_ID_TO_NAME`. Delete the old
  `ActivationCaptureManager`, `StepCapturePlan`, `CapturePositionEntry`,
  `CaptureBatchView`. (This is an EDIT, not a full DELETE — the file
  stays with the op machinery.)

**SamplingParams / outputs layer (deferred from F):**
- Remove `SamplingParams.activation_storing` field and
  `_validate_activation_storing` helper.
- Remove `RequestOutput.activation_storage` field from `vllm/outputs.py`.
- Remove `setattr(request_output, "capture_results", ...)` hack from
  `vllm/v1/engine/output_processor.py`; replace with direct kwarg.

**Entrypoint layer (deferred from F):**
- Remove `activation_storing` request field from
  `vllm/entrypoints/openai/chat_completion/protocol.py` and
  `completion/protocol.py`.
- Remove `ActivationStorageResponse` pydantic model.
- Remove `activation_storage` response field from
  `ChatCompletionResponse` / `ChatCompletionStreamResponse` /
  `CompletionResponse` / `CompletionStreamResponse`.
- Remove `_admit_activation_storing` from `chat_completion/serving.py`
  and `completion/serving.py`.

**CLI / arg_utils layer (deferred from E):**
- Remove `--activation-storing*` flags from `vllm/engine/arg_utils.py`.
- Remove corresponding `EngineArgs` fields.
- Remove `VllmConfig.activation_storing_config` field from
  `vllm/config/vllm.py`.
- Remove `ActivationStoringConfig` export from `vllm/config/__init__.py`.

**TP/PP assertion (deferred from D):**
- Remove the old `activation_storing_config` TP/PP assertion from
  `vllm/v1/engine/core.py`. Keep only the `capture_consumers_config`
  assertion.

**File moves (deferred from C):**
- Move `vllm/v1/worker/activation_writer.py` →
  `vllm/v1/capture/consumers/filesystem/writer.py`. Update the
  import in `vllm/v1/capture/consumers/filesystem/consumer.py` to
  the new path.
- Move `vllm/entrypoints/openai/activation_storing_validation.py` →
  `vllm/v1/capture/consumers/filesystem/validation.py` (absorbing
  its contents into the thin wrapper C created). Update any
  remaining imports.

**Old tests:**
- `tests/v1/worker/activation_storing/` — entire subtree. Every
  test file in here has been superseded by equivalents in
  `tests/v1/capture/`.
- `tests/entrypoints/openai/test_activation_storing_protocol.py`
- `tests/entrypoints/openai/test_activation_storing_e2e.py`

**Old docs:**
- `docs/features/activation_storing.md` — delete, or keep as a
  redirect stub with a single line pointing at
  `docs/capture_consumers/design.md`.
- `docs/features/activation_storing_roadmap.md` — delete.

### Files to verify clean

- Grep the tree for `activation_storing`, `ActivationStoring`,
  `activation_storage`, `ActivationStorageResponse`,
  `ActivationWriter`, `ActivationCaptureManager`. Every hit should
  be in an intentional location (e.g., `capture_consumers/design.md`'s
  "What this replaces" migration section). No production code hits.

### Files to update (docs cleanup)

- `docs/capture_consumers/design.md` — fold in any design-risk
  resolutions that came up during implementation. Remove any "will
  exist temporarily during refactor" notes that are now obsolete.
- `docs/OVERVIEW.md` — mark `activation_storing` as removed, ensure
  the `capture_consumers` entry is complete.

### Done when

- `grep -rn "activation_storing\|activation_storage\|ActivationStoring\|ActivationWriter\|ActivationCaptureManager" vllm/ tests/` returns no hits in production code (may have intentional hits in docs).
- `.venv/bin/python -m pytest tests/v1/capture/ tests/entrypoints/openai/test_capture_protocol.py -v` fully green.
- `.venv/bin/python -m pytest tests/v1/worker/test_steering_manager.py tests/model_executor/layers/test_steering_op.py -v` unchanged green.
- `pre-commit run --all-files` clean.
- The full test suite (`.venv/bin/python -m pytest tests/ -x -q`) passes at least the subset that was green before Phase A, minus the activation-storing tests that have been removed (those were moved or replaced, not regressed).
- `vllm serve --help` shows only `--capture-consumers`, not `--activation-storing*`.
- One manual smoke test: boot `vllm serve` with a filesystem consumer via `--capture-consumers 'filesystem:root=/tmp/caps'`, send one capture request, verify files appear with the same on-disk layout as Phase 2 originally produced.

### Design risks

1. **Breadth of the diff.** Phase I will be the largest diff of the
   refactor. Most of it is deletions and moves. The review strategy
   is to diff-stat first (to scope the size), then walk file-by-file
   with a checklist, confirming each deletion has zero remaining
   callers via grep.

2. **Moving `activation_writer.py`.** The file moves from
   `vllm/v1/worker/activation_writer.py` to
   `vllm/v1/capture/consumers/filesystem/writer.py`. Use
   `git mv` so git recognizes it as a rename in the history, not a
   delete + add. Update the import in
   `vllm/v1/capture/consumers/filesystem/consumer.py` in the same
   commit.

3. **Stripping `activation_capture.py`.** Be careful with the diff
   here — the file retains the custom op + gate, but loses the
   manager + plan types. Mark the kept sections explicitly with
   comments so a reviewer can tell at a glance which parts are
   intentional survivors.

---

## Post-phase checklist

Required before declaring the refactor complete:

- [ ] Every invariant in `design.md` § "Invariants" has at least one
      test covering it.
- [ ] The end-to-end smoke test from the old activation storing spec
      (boot `vllm serve` + `--capture-consumers filesystem:root=/tmp/caps`
      + one capture request + verify files) passes.
- [ ] A driver-side consumer (e.g., a minimal callback that records
      captures into a dict) demonstrably works via both
      `LLM(capture_consumers=[instance])` and the entry-point path
      via `vllm serve`.
- [ ] `docs/OVERVIEW.md` is up to date.
- [ ] `docs/capture_consumers/design.md` FAQ is updated with answers
      to any questions that came up during implementation.
- [ ] The deprecated `docs/features/activation_storing.md` and
      `docs/features/activation_storing_roadmap.md` files are either
      deleted or left as redirect stubs.

## Things to confirm with the user before starting

Unlike the activation storing roadmap, this refactor does not have
load-bearing ambiguities — the design doc is complete enough that
phases should flow without stopping for clarification. The one
exception:

- **Phase G driver-side result aggregation (option A vs option B
  from the design risks).** The choice of whether the driver
  receiver sends an ack back to the worker or whether
  `capture_results` is async-polled is worth confirming before Phase
  G starts. Default: option A (ack back to worker for consistency).
  Flag this in the Phase G PR body if you go a different way.

## Estimated PR shapes

For context on "how big is each phase". Under the additive
restructuring, phases B–H are dominated by new files with very
small edits to existing code; Phase I absorbs the bulk of the
deletions and file moves.

| Phase | Rough size | Churn | Notes |
|---|---|---|---|
| A | ~800 LOC new, 0 changed | Pure additive | New subpackage skeleton |
| B | ~800 LOC new, 0 changed | Pure additive | New manager + plan alongside old ones |
| C | ~400 LOC new, +1 line in `pyproject.toml` | Additive | New consumer wraps existing `ActivationWriter` by import path |
| D | ~250 LOC edited in `gpu_model_runner.py` + output plumbing files | Cutover | Runner reads from new manager, old manager becomes unused |
| E | ~500 LOC new, ~40 LOC added to `arg_utils.py` + `vllm.py` | Additive | New config alongside old |
| F | ~300 LOC added across sampling params / outputs / protocol / serving | Additive | New fields alongside old |
| G | ~700 LOC new | Additive | Cross-process bridge, not yet wired into engine/core |
| H | ~200 LOC new (logging consumer + tests) + ~300 LOC docs + example plugin | Additive | Mostly docs |
| I | ~300 LOC deleted + several file moves (via `git mv`) + ~50 LOC edited | Cleanup | Largest diff of the refactor, all mechanical |

Total across all phases: ~4000 LOC new, ~500 LOC edited in place,
~1200 LOC deleted (in Phase I alone). Big refactor but bounded and
mechanically reducible phase-by-phase. The additive phases are all
small enough to land in isolation; the cutover phases (D, F) are
the highest-review-burden pieces because they're where production
behavior actually changes.
