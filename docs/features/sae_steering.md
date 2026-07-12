# SAE-Based Steering (Delta + Full Reconstruction)

> **Status:** both variants shipped — delta (fused Triton kernel) and
> full reconstruction (compaction-based CUDA path) — integrated with
> the rebuilt steering framework (config-pool backpressure scheduler,
> `_steering_register_request` runner lifecycle, `post_block` hook).
> Companion to [`steering.md`](steering.md) and
> [`../design/steering_runtime.md`](../design/steering_runtime.md).

## In Scope

- Per-(layer, hook_point) **SAE feature surgery (delta)**: encode the
  live residual into SAE feature space, replace a small set of named
  feature activations with caller-supplied clamp values (or relative
  offsets), and add the resulting decoder-direction delta back into the
  residual.
- **SAE full reconstruction**: replace the residual entirely with the
  SAE's `decode(activate(encode(h))) + b_dec`, optionally with
  per-(layer, hook) clamps applied to the activation before decode
  (`SteeringModuleKind.SAE_FULL_RECONSTRUCTION`, wire value
  `sae_full_reconstruction`).
- Loading SAE weights as kinds of named steering module
  (`SteeringModuleKind.SAE_DELTA` / `SAE_FULL_RECONSTRUCTION`),
  distinct from the additive `vectors`-based modules.
- Per-request clamp specs: a request says "for SAE module `golden_gate`,
  clamp feature 34 to value 5.0 in the `post_block` hook on layer 20."
- A **global SAE clamp tier** applying operator-installed clamps to
  every token, phase-keyed (prefill/decode), with zero per-request
  bookkeeping.
- Composition with the existing additive tier (additive vector, then
  SAE delta, then SAE full reconstruction on the same hook, in that
  order).
- TP / PP correctness, prefix-cache correctness, CUDA-graph
  compatibility.

## Out of Scope

- **Training SAEs.** Weights are loaded from disk (Gemma Scope, Llama
  Scope, user-provided checkpoints, etc.).
- **Cross-layer crosscoders / transcoders.** Each SAE site is bound to
  one `(layer, hook_point)` pair, but one registered SAE module may
  cover multiple independent sites when its manifest lists multiple
  pairs. The runtime does not model cross-layer feature interactions.
- **Probe-based / classifier-based steering.** That is the
  dynamic-steering control plane
  ([`../design/dynamic_steering.md`](../design/dynamic_steering.md));
  this document is specifically about decoder-direction interventions
  on SAE feature activations.
- **Rust frontend internals.** The Rust frontend forwards the SAE
  request fields and registers SAE modules, but its implementation is
  documented separately in `rust/docs/features/steering_capture.md` —
  see [Current Limitations](#current-limitations) for the surface it
  covers.

## Background and Choice of Variant

Two variants of SAE steering are visible in the literature, and both
are implemented:

1. **Reconstruction replacement** (Anthropic 2024, Templeton et al.):
   `x_new = W_dec · clamp(encode(x), i, v) + b_dec`; the original
   residual is discarded along with the SAE's reconstruction error.
   Cost: one full encoder GEMM and one full decoder GEMM per *opted-in*
   token at every hooked layer. Implemented as the full-reconstruction
   kind; the compaction path restricts the GEMMs to the tokens whose
   requests opted in, so uninvolved tokens pay nothing and keep zero
   reconstruction error.

2. **Delta / feature surgery** (most follow-up steering work,
   Neuronpedia / Gemma Scope tooling, refusal-steering literature):
   `x_new = x + Σ_i (v_i − f_i) · W_dec[:, i]` for the small set of
   clamped features `i ∈ I`, where `f_i` is the live encoder-projection
   activation. Non-targeted features and the SAE's reconstruction error
   are untouched. Cost: `|I| · d_model` per token plus `|I|` encoder
   rows. Implemented as the delta kind; this is the cheap default.

## Type System

Request-side types live in
[`vllm/config/sae_steering_types.py`](../../vllm/config/sae_steering_types.py);
the module manifest lives with the registry in
[`vllm/entrypoints/openai/steering/registry.py`](../../vllm/entrypoints/openai/steering/registry.py).

```python
# vllm/config/sae_steering_types.py
class SteeringModuleKind(str, Enum):
    ADDITIVE = "additive"
    SAE_DELTA = "sae_delta"
    SAE_FULL_RECONSTRUCTION = "sae_full_reconstruction"

class SAEActivation(str, Enum): ...   # RELU, JUMP_RELU, TOPK

@dataclass(frozen=True)
class SAEClampEntry:
    feature_idx: int
    kind: Literal["absolute", "additive"]
    value: float
    only_if_active: bool = False
    # Derived property: whether this entry needs the live encoder
    # activation (absolute clamps and only_if_active gating do).
    requires_encoder_pass: bool

@dataclass(frozen=True)
class SAEClampSpec:
    """Per-request delta-clamp spec for one SAE module."""
    module_name: str
    clamps: dict[str, dict[int, tuple[SAEClampEntry, ...]]]  # hook -> layer -> entries
    phase: Literal["both", "prefill", "decode"] = "both"

@dataclass(frozen=True)
class SAEFullReconstructionSpec:
    """Per-request full-reconstruction spec; empty clamps = pure recon."""
    module_name: str
    clamps: dict[str, dict[int, tuple[SAEClampEntry, ...]]] = ...
    phase: Literal["both", "prefill", "decode"] = "both"

# vllm/entrypoints/openai/steering/registry.py
@dataclass
class SAEModuleManifest:
    d_model: int
    d_sae: int
    activation: SAEActivation
    layers: tuple[tuple[int, str], ...]       # (layer_idx, hook_point_str)
    clampable_features: tuple[int, ...]       # row order for loaded weights
    activation_params: dict[str, float] = field(default_factory=dict)
    weights_uri: str | None = None
```

`kind="absolute"` means `f_i := value`; `kind="additive"` means
`f_i := f_i + value`. `only_if_active` gates the clamp on the feature
being live in the encoder pass: `f_i > 0` for ReLU/JumpReLU, and
`f_i != 0` under TopK (a selected feature may legitimately be
negative). Hook names in `clamps` validate against
`VALID_HOOK_POINT_NAMES` (derived from `SteeringHookPoint`:
`pre_attn`, `post_attn`, `post_block`).

## Hook Points and Block-Output Semantics

SAE surgery attaches at any of the three hook points. `pre_attn` and
`post_attn` are single-tensor sites inside `apply_layer_steering`.

`post_block` (the historical `post_mlp`, renamed and semantically
corrected) is special. vLLM defers each layer's MLP branch-add into the
*next* fused add+norm, so at the end of a decoder layer `residual` does
NOT yet include this layer's MLP output — the true block output is
`residual + hidden_states`. Additive steering is indifferent to this
(adding a vector to either summand propagates identically), but SAE
encode/decode is not: encoding the bare residual would miss the layer's
MLP contribution, and full reconstruction would drop it entirely.

`apply_block_steering` therefore forms the true block output for SAE
and writes back through the residual summand as a delta:

```python
# vllm/model_executor/layers/steering.py :: apply_block_steering
if <SAE delta or FR buffers attached at POST_BLOCK>:
    block_out = residual + hidden_states
    steered = _maybe_apply_layer_sae(module, block_out, POST_BLOCK)
    residual = residual + (steered - block_out)
```

Tokens the SAE ops leave untouched stay bit-identical (`steered ==
block_out` exactly for pass-through rows, so the delta is exactly
zero), and the extra adds only exist when SAE buffers are attached at
the site — a static per-process property, so `torch.compile` traces the
no-SAE branch away.

Per-hook composition order (each stage independently gated by a static
buffer-presence check): capture → patch → additive steering → SAE
delta → SAE full reconstruction.

## Data Flow

### Startup / Module Registration

1. An operator registers an SAE module via either:
   - CLI: `--steering-modules golden_gate=/path/to/sae_dir/` where
     `sae_dir/` contains `manifest.json` plus per-site safetensors
     (see [`sae_loader.py`](../../vllm/entrypoints/openai/steering/sae_loader.py):
     `load_sae_module_from_dir`, `load_gemma_scope_sae`,
     `load_gemma_scope_sae_full_recon`, `merge_loaded_sae_modules`).
   - Runtime API: `POST /v1/steering/modules/register` with
     `kind: "sae_delta"` or `"sae_full_reconstruction"` and a
     `sae_manifest` payload
     ([`modules_router.py`](../../vllm/entrypoints/serve/steering/modules_router.py),
     key-gated like all steering mutations).
2. `NamedSteeringModuleRegistry.register` branches on `kind`; SAE kinds
   require a manifest and reject additive vector tiers. Manifest
   validation checks `d_model` against the model's hidden size,
   dtype/shape of every site tensor, and hook-point names.
3. The registry broadcasts to every worker via
   `collective_rpc("register_steering_modules", ...)`. Payloads carry a
   `kind` discriminator on every entry;
   `dump_for_broadcast(include_sae_weights=True)` inlines the SAE
   weights for startup/full-registry pushes. Each worker's
   `register_steering_modules` (in
   [`steering_model_runner_mixin.py`](../../vllm/v1/worker/steering_model_runner_mixin.py))
   routes SAE kinds to `_attach_sae_buffers` /
   `_attach_sae_full_recon_buffers`, which filter `manifest.layers` by
   `_locally_owned_layers` (PP sharding), attach per-(layer, hook)
   buffers on the owning layer modules, and pre-warm the kernels
   (`warmup_apply_sae_delta_kernel` /
   `warmup_apply_sae_full_recon_kernel`) so first-call JIT/cuBLAS costs
   land outside any captured forward. `attach_sae_weights` /
   `attach_sae_full_recon_weights` copy encoder/decoder tensors into
   the buffers.
4. Removal broadcasts drop the additive pre-materialize pin first
   (`release_pre_materialized_steering_module`, a no-op for SAE
   modules, which are never pre-materialized) and then unregister; a
   partial broadcast failure triggers a compensating broadcast
   (re-register the prior payload, or unregister when there was none).
   Unregistering a prefill-affecting module resets the prefix cache.

Cross-rank invariant: the *set of clampable feature indices* and the
*module name → (layer, hook) → row layout* must be byte-identical
across ranks, because per-request clamp configs are hashed and turned
into shared row indices via deterministic replay. Weight tensors are
local; only the schema is replicated.

### Request Hashing

`SamplingParams` carries `sae_clamp_specs` and
`sae_full_reconstruction_specs`. Six cached hash properties feed the
runtime:

- `prefill_steering_config_hash` / `decode_steering_config_hash` — the
  *combined* per-phase identity: `hash_steering_config(...)` folds the
  additive vectors, `steering_module_ref`, the phase-filtered SAE clamp
  specs, and the phase-filtered FR specs, each under a distinct domain
  separator (a delta spec and an FR spec with identical content never
  collide). These are the prefix-cache keys and the scheduler's
  presence gates.
- `prefill_sae_clamp_config_hash` / `decode_sae_clamp_config_hash` —
  SAE-delta-only phase hashes (`hash_sae_clamp_specs_for_phase`); key
  `SAEClampManager` rows so identical clamp content shares worker
  capacity even when additive content differs.
- `prefill_sae_full_recon_config_hash` /
  `decode_sae_full_recon_config_hash` — FR-only phase hashes; key
  `SAEFullReconstructionManager` rows so additive/delta state can never
  route a token into a reconstruction row in the wrong phase.

`InputProcessor._validate_steering` treats a request with any SAE spec
as a steering request (engine must be started with steering enabled)
and re-raises spec validation errors at admission.

### Scheduler Capacity (three independent pools)

The scheduler runs config-pool backpressure: rather than falling back
to unsteered execution, a request whose steering rows would overflow
worker capacity is held in the waiting queue. Three pools are tracked
independently:

- additive rows (`_request_steering_config_pairs(...)[0]`,
  gated by `_request_uses_additive_steering`),
- SAE delta rows (`_request_steering_config_pairs(...)[1]`, keyed by
  the SAE-only phase hashes),
- FR rows (`_request_sae_full_recon_config_pairs`, keyed by the
  FR-only phase hashes).

`_steering_pool_would_overflow` is checked for each pool at every
admission and reservation point (running-request pre-pass, waiting
admission, post-admit decode reservation), over-reserving both phases.
A request using all three tiers needs a row in each pool; each pool is
sized by `SteeringConfig.max_steering_configs`
(`--max-steering-configs`, default 32).

### Worker Runtime Lifecycle

All SAE runtime state lives in `SteeringModelRunnerMixin`, shared by
the v1 (`gpu_model_runner.py`) and v2 (`worker/gpu/model_runner.py`)
runners:

- **Admission** — `_steering_register_request` validates SAE specs
  (`_assert_sae_clamps_can_be_applied`,
  `_assert_sae_full_recon_specs_can_be_applied`: unknown modules,
  uncovered sites, unclampable features fail loud) *before any
  mutation*, then registers rows with `_register_initial_sae_clamps` /
  `_register_initial_sae_full_recon` under the phase the request
  starts in. A failure after the additive row registered rolls the
  additive row back — admission is all-or-nothing across tiers.
- **Per-step** — `_update_steering_buffers` drives `_update_sae_all`
  at both of its exits (so SAE-only and decode-only requests are
  handled even when the additive fast path short-circuits), which runs
  `_update_sae_buffers` and `_update_sae_full_recon_buffers`: populate
  the per-layer clamp tables from manager state when dirty
  (`populate_sae_clamp_table` /
  `populate_sae_full_recon_clamp_table`), build the per-token
  `sae_index` / `sae_recon_index` routing buffers (`np.repeat` +
  non-blocking H2D, same pipeline as the additive `steering_index`),
  and then apply prefill→decode SAE transitions for requests that
  complete prefill this step
  (`_apply_batched_sae_transitions` /
  `_apply_batched_sae_full_recon_transitions`, batched
  release-then-register so shared rows free before re-registration).
  The transition scan is skipped in steady state
  (`_may_need_prefill_completion_transition_scan`).
- **Release** — `_steering_finish_requests` releases SAE and FR rows
  independently of the additive manager (SAE-only deployments work
  with `_steering_manager is None`).
- **Preemption / resumption** — `_reset_steering_for_resumption`
  releases decode-phase SAE/FR rows and re-registers prefill rows,
  mirroring the additive tier.

Row-capacity note: the mixin passes the SAE-only content hashes to the
managers, and passes the additive-only content hash as `content_hash`
to `SteeringManager.register_config` — identical additive content
aliases to one physical row across requests whose *combined* hashes
differ (because their SAE specs differ). Explicit `content_hash`
aliasing applies to both phases; *implicit* content aliasing (no
`content_hash` supplied) applies to prefill rows only, keeping decode
rows 1:1 with logical hashes for the per-row scale / monitor / dynamic
machinery.

### Per-Step Forward

`apply_layer_sae_delta` dispatches to
`torch.ops.vllm.apply_sae_delta_indexed` (tensor-only schema, integer
activation code + one float scalar), so `torch.compile` treats the SAE
op as an opaque splitting point — same shape as the additive
`apply_steering`. The indexed op receives the persistent clamp tables
plus the shared `sae_index` and `any_active` buffers and performs the
per-token row gather *inside* the op, which is what keeps the call
CUDA-graph-safe: replays read whatever the populator wrote into the
persistent tables between steps, and the `any_active` flag
short-circuits the whole op when no SAE row is live. On CUDA it runs a
fused Triton kernel
([`sae_steering_kernel.py`](../../vllm/model_executor/layers/sae_steering_kernel.py)):
encoder rows, activation, clamp logic, decoder rows, and add-back in a
single launch, one program per token, the clamp axis register-resident.
The direct-tensor API (`apply_sae_delta` /
`torch.ops.vllm.apply_sae_delta`) takes pre-gathered per-token clamp
tensors and is the eager reference and test surface; its Python body
is also the CPU fallback.

`apply_layer_sae_full_reconstruction` derives its per-token gate from
the site's **active-row table** (`sae_fr_row_active_*`):
`recon_mask = active_table[recon_index]`. Row 0 is never active (the
no-reconstruction sentinel) and rows owned by *other* modules' sites
are inactive here, so a shared `sae_recon_index` can never cause
cross-module reconstruction. The CUDA path
([`sae_full_reconstruction_kernel.py`](../../vllm/model_executor/layers/sae_full_reconstruction_kernel.py),
`apply_sae_full_recon_triton`) compacts active tokens into a dense
subset, runs the full encoder/clamp/decoder math on that subset via
cuBLAS-backed matmuls, and scatters the reconstructed rows back;
inactive tokens keep their original residual bit-for-bit.

## Row Layout and the Global Clamp Tier

`SAEClampManager`
([`sae_clamp_manager.py`](../../vllm/v1/worker/sae_clamp_manager.py))
mirrors the additive manager's table layout:

| Row | Contents |
|---|---|
| 0 | no-op sentinel (always zero) |
| 1 | global **prefill** clamps |
| 2 | global **decode** clamps |
| 3 … max+2 | per-(spec-content-hash, phase) request rows |

`get_row_for_config(0, is_prefill)` routes hash-0 tokens to row 1 or 2
by phase, so a request with no per-request SAE state picks up whatever
globals are installed with zero per-request bookkeeping; when no
globals are configured those rows stay zero (true no-op, bit-for-bit
parity with SAE-disabled). Per-layer buffers are sized
`max_sae_configs + 3` rows accordingly (`register_sae_buffers`).

Global-tier contract (`set_global_clamps` / `clear_global_clamps` /
`has_global_clamps` / `global_specs_for_phase`):

- **Atomic install.** `set_global_clamps(prefill_specs, decode_specs,
  replace=...)` validates the complete new state (including overlap
  against every active per-request row, per phase) before mutating; a
  validation failure leaves the previous global state fully intact.
- **Overlap rejected both directions.** Registering a per-request spec
  that clamps a feature covered by a same-phase global is rejected
  (`register_clamp_spec` validates against the installed globals), and
  installing a global that collides with an active per-request row is
  rejected. Cross-phase combinations are allowed. The populator
  additionally refuses (rather than silently overwrites) any collision
  that reaches write time.
- **Stacks with per-request clamps.** The populator merges the
  phase-appropriate globals into every per-request row, so a request
  that opts into its own clamps still gets the globals.

Worker-side RPC surface and the HTTP endpoints
([`api_router.py`](../../vllm/entrypoints/serve/steering/api_router.py))
that wrap it:

| Endpoint | Worker call | Effect |
|---|---|---|
| `POST /v1/steering/sae/set` | `SteeringModelRunnerMixin.set_sae_global_clamps` | Validate against the registry + active rows, install |
| `POST /v1/steering/sae/clear` | `SteeringModelRunnerMixin.clear_sae_global_clamps` | Drop both phase tiers |
| `GET /v1/steering/sae` | `SteeringModelRunnerMixin.get_sae_global_clamps_status` | JSON-safe view of both tiers |

`POST /v1/steering/sae/set` takes
`{prefill_specs, decode_specs, replace}` (JSON-shape clamp specs, same
shape as the per-request `sae_clamp_specs` sampling field — see
`SetSAEGlobalClampsRequest` in
[`protocol.py`](../../vllm/entrypoints/serve/steering/protocol.py)).
The mutating endpoints share the additive router's discipline: gated
by `--steering-api-key`, serialized under the steering lock, two-phase
validate-then-apply (`validate_only=True` fans out first, so a
rank-local validation failure surfaces before any rank mutates), and
a mandatory prefix-cache reset after a successful set or clear (503 if
the reset fails — global clamps affect every token's prefill/decode
KV). `GET /v1/steering/sae` is unauthenticated and verifies all
workers report identical global state (500 on divergence).

`SAEFullReconstructionManager` has **no global tier**: row 0 is the
no-reconstruction sentinel and rows `1..max` are per-request. Its
per-site active-row table is what distinguishes "row allocated" from
"row applies at this site".

## Files

Core implementation:

- [`vllm/model_executor/layers/sae_steering.py`](../../vllm/model_executor/layers/sae_steering.py)
  — delta path: `register_sae_buffers` / `unregister_sae_buffers` /
  `sae_buffers_attached`, shared `sae_index` buffer
  (`register_sae_index_buffer`, `share_sae_index_across_layers`),
  dispatch shim `apply_layer_sae_delta`, public eager API
  `apply_sae_delta`, custom ops `apply_sae_delta_op` /
  `apply_sae_delta_indexed_op`, populator `populate_sae_clamp_table`,
  clamp-kind constants shared with the FR path.
- [`vllm/model_executor/layers/sae_steering_kernel.py`](../../vllm/model_executor/layers/sae_steering_kernel.py)
  — fused Triton delta kernel + `warmup_apply_sae_delta_kernel`.
- [`vllm/model_executor/layers/sae_full_reconstruction.py`](../../vllm/model_executor/layers/sae_full_reconstruction.py)
  — FR buffers (including the `sae_fr_row_active_*` active-row tables
  and shared `sae_recon_index`), dispatch shim
  `apply_layer_sae_full_reconstruction`, eager API + custom op,
  populator `populate_sae_full_recon_clamp_table`, full-encoder
  helper `sae_encode_full`.
- [`vllm/model_executor/layers/sae_full_reconstruction_kernel.py`](../../vllm/model_executor/layers/sae_full_reconstruction_kernel.py)
  — compaction-based CUDA path `apply_sae_full_recon_triton` +
  `warmup_apply_sae_full_recon_kernel` (cuBLAS-backed GEMMs on the
  active-token subset; deliberately not a bespoke Triton GEMM).
- [`vllm/model_executor/layers/steering.py`](../../vllm/model_executor/layers/steering.py)
  — `SteeringHookPoint`, `VALID_HOOK_POINT_NAMES`, the SAE
  marker-attr dicts (`HOOK_POINT_SAE_CLAMP_KIND_ATTR`,
  `HOOK_POINT_SAE_FR_CLAMP_KIND_ATTR` — plain strings so the no-SAE
  hot path never imports the SAE modules), `_maybe_apply_layer_sae`,
  and the block-output SAE application in `apply_block_steering`.
- [`vllm/config/sae_steering_types.py`](../../vllm/config/sae_steering_types.py)
  — `SteeringModuleKind`, `SAEActivation`, `SAEClampEntry`,
  `SAEClampSpec`, `SAEFullReconstructionSpec`, coercers, overlap
  validation, and the phase-aware hash helpers.
- [`vllm/config/steering.py`](../../vllm/config/steering.py) —
  `SteeringConfig` (`max_steering_configs` sizes every pool).
- [`vllm/v1/worker/sae_clamp_manager.py`](../../vllm/v1/worker/sae_clamp_manager.py)
  — `SAEClampManager` (row allocation / refcount / strict capacity /
  deterministic replay / global tier / overlap validation).
- [`vllm/v1/worker/sae_full_reconstruction_manager.py`](../../vllm/v1/worker/sae_full_reconstruction_manager.py)
  — `SAEFullReconstructionManager` (same contract, no global tier).
- [`vllm/v1/worker/steering_model_runner_mixin.py`](../../vllm/v1/worker/steering_model_runner_mixin.py)
  — all lifecycle wiring listed above; SAE state access in shared
  framework paths is `getattr`-guarded so partial harnesses and
  SAE-free deployments never touch it.
- [`vllm/v1/core/sched/scheduler.py`](../../vllm/v1/core/sched/scheduler.py)
  — the three-pool capacity accounting.
- [`vllm/sampling_params.py`](../../vllm/sampling_params.py) — SAE
  spec fields, phase filters, the six cached hash properties.
- [`vllm/entrypoints/openai/steering/registry.py`](../../vllm/entrypoints/openai/steering/registry.py)
  — `SAEModuleManifest`, kind-branched `register`,
  `validate_sae_clamp_specs`, `validate_additive_lookup`,
  `apply_sampling_params_hash_overrides`, `restore_or_remove`,
  `dump_for_broadcast(include_sae_weights=...)`.
- [`vllm/entrypoints/openai/steering/sae_loader.py`](../../vllm/entrypoints/openai/steering/sae_loader.py)
  — manifest + safetensors reader (`load_sae_module_from_dir`), Gemma
  Scope NPZ readers (`load_gemma_scope_sae`,
  `load_gemma_scope_sae_full_recon`), `merge_loaded_sae_modules`.
- [`vllm/entrypoints/serve/steering/modules_router.py`](../../vllm/entrypoints/serve/steering/modules_router.py)
  / [`modules_protocol.py`](../../vllm/entrypoints/serve/steering/modules_protocol.py)
  — HTTP registration surface (`SAEModuleManifestRequest`, `kind`,
  compensating broadcasts, prefix-cache reset on unregister;
  pre-materialization is additive-only).

Test anchors: `tests/model_executor/layers/test_sae_*` (op / kernel /
buffer / dispatch / populator contracts),
`tests/v1/worker/test_sae_*` (managers, mixin lifecycle, module
admission, end-to-end per-step pipeline),
`tests/v1/test_sae_*` (types and SamplingParams),
`tests/v1/core/test_steering_scheduler.py` (pool separation),
`tests/entrypoints/**/test_sae_*` + `test_modules_router_sae.py`
(loader, registry, router), and the CUDA + HF-gated real-weights
generation tests in `tests/models/language/generation/`.

Decoder `*.py` model files: **no changes** — SAE dispatch is
centralized inside `apply_layer_steering` / `apply_block_steering`, so
every existing hook site picks it up without per-model edits.

## Invariants and Constraints

- **Determinism contract preserved.** Every worker sees identical
  registry broadcasts and scheduler output, so every worker derives
  identical `config_to_row` mappings in all three managers, even if it
  owns zero layers of a module under PP. Row allocation is rank-local
  but symmetric.
- **Capacity is strict, per pool, backpressure not fallback.** A
  request needing a row in a full pool waits in the queue; it is never
  silently downgraded to unsteered execution. Rows are keyed by
  *content* hashes, so identical specs across requests share a row via
  refcounting.
- **Prefix-cache keys.** SAE and FR specs that affect prefill fold
  into `prefill_steering_config_hash` (domain-separated from the
  additive and from each other); decode-only specs do not. Registering
  or unregistering a prefill-affecting module resets the prefix cache.
- **Admission is all-or-nothing across tiers**, and validation runs
  before any mutation, so a rejected streaming continuation leaves the
  request's previous rows intact.
- **Disabled-mode is free.** With no SAE module registered, no SAE
  buffers exist, the marker-dict `hasattr` gates are statically false,
  and the forward is bit-identical to the additive-only path — the
  no-SAE hot path never imports the SAE modules.
- **Cross-module isolation (FR).** The per-site active-row table
  guarantees a token routed to another module's FR row passes through
  this site unchanged.
- **TP rank divergence** surfaces at module-registration time, not at
  request time (TP ranks within a PP stage must own identical layer
  sets).
- **Numerical contract.** Encoder/decoder GEMMs run in the model's
  compute dtype (bf16/fp16); the per-feature activation vector and the
  activation-function evaluation (JumpReLU threshold, TopK selection,
  `delta = clamp(f, target) − f`) promote to fp32 and cast back before
  the decoder add. TopK tie-breaks keep the lowest feature indices
  (deterministic), and `only_if_active` under TopK treats selected
  negative features as active (`f != 0`).

## Encoder Footprint

The delta path loads only the encoder/decoder **rows for the declared
`clampable_features`** (typical Golden-Gate-style work: `|clampable| ≤
64` → ~512 KB per site instead of ~512 MB for a full 64k × 4k encoder).
Operators who want "any feature clampable at runtime" declare the full
feature set and pay the full cost. Consequence: the delta path's TopK
is "TopK within the clampable subset"; true full-`d_sae` TopK semantics
require the full-reconstruction kind, which always loads the full
encoder/decoder (that is its cost model — one extra MLP-sized pair of
GEMMs per opted-in token per hooked layer).

## Current Limitations

- **Rust frontend: HTTP only.** `rust/` accepts the per-request
  `sae_clamp_specs` / `sae_full_reconstruction_specs` fields and its
  steering-modules surface (startup `--steering-modules` files and
  `POST /v1/steering/modules`) registers both SAE kinds, sending
  weights inline as base64-packed `{dtype, shape, data}` tensors
  keyed by `"layer:hook"` (torch tensors do not survive the
  `collective_rpc` hop; the worker's `_coerce_sae_weights_wire`
  rebuilds them). gRPC remains request-passthrough only — SAE module
  registration over gRPC is deferred. See
  `rust/docs/features/steering_capture.md`.
- **`replace=True` module pushes are per-worker atomic but not
  cross-rank transactional.** On each worker,
  `register_steering_modules(replace=True)`
  (`_replace_steering_modules_atomically`) snapshots the additive and
  both SAE registries — including attached SAE/FR weights — before
  clearing and re-adding, and restores them on failure, so a poisoned
  push cannot destroy a working registry. Pre-materialize pins are
  released before the clear and *not* re-established on rollback (the
  next `pre_materialize_steering_module` call re-installs them). What
  remains best-effort is cross-rank consistency: `collective_rpc` is
  not transactional, so a failure on some ranks after others committed
  is not compensated — the same constraint as the additive per-name
  path (see `_compensating_broadcast_after_failure` in
  `entrypoints/serve/steering/modules_router.py`).
- **Batch chat API requires packed steering vectors**
  (`SteeringVectorSpecPacked`); legacy dict-of-lists vectors are not
  accepted on the batch surface.
- **At most one full-reconstruction SAE module per (layer, hook)
  site**; double-registration raises by design — two residual
  replacements on one site are semantically ill-defined. Delta
  (`sae_delta`) modules are not so limited: any number may share a
  site (each gets its own buffer slot) and their deltas compose
  sequentially in registration order, which is identical across ranks.
  An FR module may also share a site with delta modules — deltas run
  first, the reconstruction replaces last. Re-registering the *same*
  module name at a site it already occupies still raises.

## Resolved Design Decisions

- **Composition order.** SAE delta runs after the additive op; full
  reconstruction runs last, so upstream additive/delta perturbations
  remain observable on tokens that don't opt into replacement. Chosen
  so the additive path's behavior is unchanged when SAE is layered on
  top.
- **Block-output semantics.** SAE at `post_block` operates on the true
  block output `residual + hidden_states` with delta write-back
  through the residual summand (see
  [Hook Points](#hook-points-and-block-output-semantics)). The
  historical `post_mlp` name is gone; manifests and clamp specs must
  key `"post_block"`.
- **Phase-split global rows.** Globals live in dedicated rows 1/2
  rather than sharing row 0 with the sentinel, so prefill and decode
  globals can differ and the sentinel stays a true no-op; hash-0
  lookups route by phase.
- **FR hash separation.** FR rows are keyed by FR-only phase hashes,
  never the combined steering hash, so additive/delta identity can't
  alias a token into a reconstruction row.
- **Encoder pass is spec-driven, per clamp entry**
  (`requires_encoder_pass`, derived): absolute clamps and
  `only_if_active` gating need the live activation; plain additive
  clamps don't. The kernel still runs the encoder for active rows;
  skipping it for pure-additive rows remains an optimization
  opportunity.
- **Manifest format: both.** Runtime registration accepts the
  vLLM-native manifest; the loader also reads Gemma Scope `params.npz`
  and synthesizes the native form, isolating upstream layout churn
  from worker buffer attachment.
- **Activations at launch: ReLU, JumpReLU, TopK** (carried by the
  manifest; adding more is a switch-table extension).
- **Numeric dtype: hybrid** — compute-dtype GEMMs, fp32 activation
  math on the tiny per-feature tensors (see Invariants).

## References

- [Steering (user-facing guide)](steering.md)
- [Steering Runtime Design](../design/steering_runtime.md)
- [Dynamic Steering Design](../design/dynamic_steering.md)
- Templeton et al., *Scaling Monosemanticity*, Anthropic, 2024.
- Lieberum et al., *Gemma Scope*, DeepMind, 2024.
- Kantamneni et al., *Steering Language Model Refusal with Sparse
  Autoencoders*, 2024 (delta intervention reference).
