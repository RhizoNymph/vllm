# SAE-Based Steering (Delta / Feature-Surgery)

> **Status:** design doc, not yet implemented. Companion to
> [`steering.md`](steering.md) and
> [`../design/steering_runtime.md`](../design/steering_runtime.md).
> This document defines the contract for the next-generation steering
> path that lifts the per-token operation from "additive vector" to
> "feature-level surgery on a sparse autoencoder."

## In Scope

- Per-(layer, hook_point) **SAE feature surgery**: encode the live
  residual into SAE feature space, replace a small set of named feature
  activations with caller-supplied clamp values (or relative offsets),
  and add the resulting decoder-direction delta back into the residual.
- Loading SAE weights as a new kind of named steering module
  (`SteeringModuleKind.SAE`), distinct from the existing additive
  `vectors`-based modules.
- Per-request clamp specs: a request says "for SAE module `golden_gate`,
  clamp feature 34 to value 5.0 in the post-MLP hook on layer 20."
- Composition with the existing additive tier (additive vector + SAE
  delta on the same hook).
- TP / PP correctness, prefix-cache correctness, CUDA-graph compatibility.

## Out of Scope

- **Full SAE forward pass with reconstruction-replacement** (Anthropic
  Scaling Monosemanticity / Golden Gate Claude semantics: replace `x`
  with `decode(modify(encode(x)))` and discard `x`'s reconstruction
  error). That is a different op with materially different cost,
  numerical, and prefix-cache properties. It can be added later as a
  parallel `SteeringModuleKind.SAE_FULL_RECONSTRUCTION` without
  reshaping the surface defined here.
- **Training SAEs.** Weights are loaded from disk (Gemma Scope, Llama
  Scope, user-provided checkpoints, etc.).
- **Cross-layer crosscoders / transcoders.** Each SAE module here is
  scoped to a single (layer, hook_point) pair; composition across layers
  happens by registering multiple modules.
- **Probe-based / classifier-based steering.** This document is
  specifically about decoder-direction interventions on SAE feature
  activations.

## Background and Choice of Variant

Two variants of SAE steering are visible in the literature:

1. **Reconstruction replacement** (Anthropic 2024, Templeton et al.):
   `x_new = W_dec · clamp(encode(x), i, v) + b_dec`, the original
   residual is discarded along with the SAE's reconstruction error.
   Cost: one full encoder GEMM and one full decoder GEMM per token at
   every hooked layer (i.e., an extra MLP per layer).

2. **Delta / feature surgery** (most follow-up steering work,
   Neuronpedia / Gemma Scope tooling, refusal-steering literature):
   `x_new = x + Σ_i (v_i − f_i) · W_dec[:, i]` for the small set of
   clamped features `i ∈ I`, where `f_i` is the live encoder-projection
   activation `relu(W_enc[i, :] · x + b_enc[i])` (or the appropriate
   activation function for that SAE). Non-targeted features and the
   SAE's reconstruction error are untouched. Cost: `|I| · d_model` per
   token plus `|I|` encoder rows.

We adopt **variant 2**. The vLLM steering runtime is built around the
admission-and-row-allocation contract in
[`steering_manager.py`](../../vllm/v1/worker/steering_manager.py); a
fixed-cost-per-feature-clamp op composes naturally with that contract,
preserves the existing additive kernel as a hot path, and avoids
inflicting SAE reconstruction error on tokens that are not being
steered. The full-reconstruction variant is left as a future module
kind; it is not on the critical path.

## Type System

The goal is to make invalid states unrepresentable at the API and
worker boundaries.

```python
class SteeringModuleKind(str, Enum):
    ADDITIVE = "additive"   # the existing path: precomputed vectors per (hook, layer)
    SAE_DELTA = "sae_delta" # this document: encode -> clamp -> decoder-direction delta

@dataclass
class SAELayerWeights:
    """Single (layer, hook_point) SAE bound to one position in the model."""
    layer_idx: int
    hook_point: SteeringHookPoint
    # Encoder rows used at inference time. Shape (d_sae, d_model).
    # We only need rows for features that may be clamped; in the common
    # case d_sae is large (32k-128k) but |clampable_features| is small.
    # See "Encoder Footprint" below.
    encoder_weight: torch.Tensor      # (d_sae, d_model) or (k_clamp, d_model)
    encoder_bias: torch.Tensor        # (d_sae,) or (k_clamp,)
    decoder_weight: torch.Tensor      # (d_sae, d_model) — full, for decoder rows
    activation: SAEActivation         # ReLU / TopK / JumpReLU
    # JumpReLU threshold or TopK k, optional depending on activation.
    activation_params: dict[str, float] | None = None
    feature_index_map: dict[int, int] | None = None  # public_id -> internal_row

@dataclass
class SAESteeringModule:
    name: str
    kind: Literal[SteeringModuleKind.SAE_DELTA]
    d_model: int
    d_sae: int
    layers: dict[tuple[int, str], SAELayerWeights]  # (layer_idx, hook_point_str)

@dataclass(frozen=True)
class SAEClampEntry:
    feature_idx: int
    # Exactly one of these is set:
    target_value: float | None = None     # absolute clamp: f_i := target_value
    additive: float | None = None         # relative: f_i := f_i + additive
    # If True, only apply when f_i > 0 in the live encoder pass; otherwise
    # always apply. Lets users say "amplify when present" vs "force-set".
    only_if_active: bool = False
    # Derived in __post_init__: whether this entry needs the encoder
    # GEMM at runtime. True iff target_value is set OR only_if_active
    # is True. False for plain `additive` entries that don't gate on
    # the live activation. The kernel skips the encoder pass entirely
    # when no active clamp entry needs it.
    requires_encoder_pass: bool = field(init=False)

@dataclass(frozen=True)
class SAEClampSpec:
    """Per-request clamp spec for one SAE module."""
    module_name: str
    # (layer_idx, hook_point) -> list of clamp entries
    clamps: dict[tuple[int, SteeringHookPoint], tuple[SAEClampEntry, ...]]
```

`target_value` and `additive` are mutually exclusive at the type level
via a discriminated-union helper (validated at construction). The
`only_if_active` flag distinguishes the "Golden-Gate-style amplify"
semantics from the "force a feature on" semantics; both are common in
the literature.

## Data Flow

### Startup

1. Operator registers an SAE module via either:
   - CLI: `--steering-modules sae:golden_gate=/path/to/sae_dir/` where
     `sae_dir/` contains a `manifest.json` plus per-layer weight
     tensors in safetensors.
   - Runtime API: `POST /v1/steering/modules/register` with
     `kind: "sae_delta"` and a pointer to a local checkpoint path
     (network-loaded weights are not in scope; admins point the server
     at on-disk artifacts).
2. The registry validates that every `(layer_idx, hook_point)`
   declared in the manifest is steerable on the loaded model
   (intersected with `_steerable_layers()` and the existing
   `valid_layer_indices` check).
3. The registry broadcasts the loaded module to every worker via
   `collective_rpc`. Each worker:
   - Filters `module.layers` by `_locally_owned_layers` (PP sharding).
   - For each owned `(layer_idx, hook_point)`, materializes the weight
     tensors on the worker's device, in the model's compute dtype.
   - Stashes the module under `worker.sae_modules[name]`.

Cross-rank invariant: the *set of clampable feature indices* and the
*module name → (layer, hook) → row layout* must be byte-identical
across ranks, because per-request clamp configs are hashed and turned
into shared row indices via the existing `SteeringManager` machinery.
Weight tensors themselves are local; only the schema is replicated.

### Per-Request Admission

A request that uses SAE steering carries an `SAEClampSpec` (one or more,
since multiple SAE modules can compose). The admission pipeline:

1. The request hash is `hash(additive_spec, sae_clamp_specs)` —
   computed on the engine side, identical on every worker. Prefix-cache
   keys reuse this hash for prefill-affecting tiers, exactly like the
   additive path.
2. The scheduler reserves capacity in
   `SteeringManager.max_steering_configs` *before* dispatch, identical
   to the existing flow.
3. On admission, the worker resolves each `SAEClampSpec` against the
   broadcast SAE registry, allocates a per-`(name, phase)` row in a new
   parallel structure (`SAEClampManager`, see below), and writes the
   row's contents into per-layer SAE clamp tables.

### Per-Step Forward

For each steerable decoder layer with at least one clamp active, the
new op `apply_layer_sae_delta(hidden_states, sae_clamp_table_*)` runs
*after* the existing `apply_layer_steering` additive call. The
existing additive path is unchanged.

```python
# Existing, unchanged:
hidden = apply_layer_steering(hidden, hook_point=POST_MLP)

# New, only enters the graph if at least one SAE module is registered
# AND at least one row in this batch needs SAE delta on this hook:
hidden = apply_layer_sae_delta(hidden, hook_point=POST_MLP)
```

`apply_layer_sae_delta` is registered as `torch.ops.vllm.apply_sae_delta`
so torch.compile treats it as an opaque splitting point, mirroring
`apply_steering`. Inside, it performs:

```text
# Per token t in the batch, with row r = sae_index[t]:
# (1) Project residual onto the encoder rows for the clamp targets.
#     pre_acts = h @ W_enc_clamped.T + b_enc_clamped
# (2) Apply activation (ReLU / JumpReLU / TopK-mask) -> live f.
# (3) Compute delta = clamp(f, target) - f, where the clamp comes from
#     the row of sae_clamp_table for r (target_value, additive,
#     only_if_active flags packed into a small struct per feature).
# (4) Add delta @ W_dec_clamped to h.
```

The kernel is a single Triton kernel that holds the per-row clamp spec
in shared memory and walks the small set of clamp targets. For the
common case `|I| ≤ 8`, this reduces to a handful of dot products per
token.

### Phase-Aware and Tiered Composition

The existing three-tier additive model (base / prefill / decode) is
preserved verbatim. SAE clamps live in a *fourth, parallel* tier,
also phase-aware:

```text
effective_prefill = additive_prefill_effective (existing)
                  + sae_delta(prefill_clamps)
effective_decode  = additive_decode_effective (existing)
                  + sae_delta(decode_clamps)
```

The two paths share the same `(config_hash, phase)` keying so prefill
and decode can carry different clamp specs in flight, and so a request
that prefills under one config and decodes under another (the existing
streaming-continuation case) is handled by the same admission machinery.

## Files

New files:

- `vllm/model_executor/layers/sae_steering.py` — public op
  `apply_layer_sae_delta`, custom-op registration, per-layer buffer
  allocation analogous to `register_steering_buffers`.
- `vllm/model_executor/layers/sae_steering_kernel.py` — Triton kernel.
- `vllm/v1/worker/sae_clamp_manager.py` — clamp-table allocator,
  parallel to `SteeringManager`. Shares the determinism-by-replay
  contract; no cross-rank collectives in the hot path.
- `vllm/v1/worker/sae_module_registry.py` — worker-side SAE weight
  registry, keyed by name; populated by `collective_rpc` broadcast.
- `vllm/entrypoints/openai/steering/sae_loader.py` — manifest +
  safetensors reader. Validates dtype, shape, hook-point names,
  `d_model` against the loaded model's hidden size.
- `tests/models/language/generation/test_sae_steering.py` — small
  end-to-end fixture test plus a numeric equivalence test against an
  eager Python reference.

Touched files:

- `vllm/sampling_params.py` — add `sae_clamp_specs` field; validation
  delegates to `SAEClampSpec.__post_init__`.
- `vllm/config/steering_types.py` — extend `hash_steering_config` to
  fold SAE clamp specs into the hash; add `merge_sae_clamp_specs`.
- `vllm/v1/worker/steering_model_runner_mixin.py` — wire SAE clamp
  manager alongside the additive `_steering_manager`. Phase transition
  and resumption paths register/release SAE rows the same way the
  additive ones do.
- `vllm/entrypoints/openai/steering/registry.py` — add
  `SteeringModuleKind` discriminator; existing additive modules keep
  their current behavior under `kind="additive"`.
- `vllm/entrypoints/serve/steering/modules_router.py` — accept
  `kind: "sae_delta"` and an SAE manifest payload.
- Decoder `*.py` files: **no changes**. The new op is wired only at the
  layer-hook call sites that already exist for `apply_layer_steering`,
  via a sibling call. (Phased plan below makes this concrete.)
- `docs/features/steering.md` — cross-link to this doc.
- `docs/design/steering_runtime.md` — add an "SAE delta" subsection
  describing the parallel admission flow.

## Invariants and Constraints

- **Determinism contract is preserved.** Every worker sees identical
  registry-broadcast and identical scheduler output, so every worker
  derives identical `sae_config_to_row` mappings, even if it owns
  zero layers of a given module under PP. Row-allocation is rank-local
  but symmetric, exactly as for the additive manager.
- **Capacity is strict.** `--max-steering-configs` continues to gate
  *combined* admission: a request that uses both an additive config
  and an SAE clamp spec consumes one row from each manager. The
  scheduler reserves both before dispatch; if either fills, the
  request is deferred via the existing two-queue priority model.
- **Prefix-cache keys.** SAE clamps that affect prefill must be folded
  into the prefill cache key. Decode-only clamps must not. This mirrors
  the additive tier and is implemented via the same
  `prefill_steering_config_hash` / `decode_steering_config_hash`
  pattern.
- **Disabled-mode is free.** When no SAE module is registered, no per-
  layer SAE buffers are attached and `apply_layer_sae_delta` short-
  circuits the same way `apply_layer_steering` does today. The
  torch.compile branch is decided at module init and stays static for
  the layer's lifetime.
- **TP rank divergence.** TP ranks within the same PP stage must own
  identical layer sets, same as today. The SAE registry surfaces a
  divergence error at module-registration time, not at request time.
- **No cross-layer state.** Each `(layer, hook_point)` SAE op is
  independent. There is no implicit ordering between layers beyond what
  the model's own forward pass already imposes.
- **Numerical contract.** The encoder/decoder GEMMs and the W_dec
  gather run in the model's compute dtype (bf16 in most deployments),
  matching Llama Scope's published inference recipe and the existing
  additive op's buffer-dtype contract (see
  `get_steering_buffer_dtype`). The per-feature activation vector
  `(k_clamp,)` is promoted to fp32 for the activation-function
  evaluation (JumpReLU threshold comparison, ReLU, TopK selection)
  and for the `delta = clamp(f, target) − f` subtraction; the
  resulting `(k_clamp,)` delta is cast back to compute dtype before
  the `delta @ W_dec_clamped` add into the residual. This is one
  promotion on a tiny tensor — no cost on the d_model GEMMs — and it
  isolates dtype sensitivity to exactly the scalar comparisons that
  are sensitive to it.

## Encoder Footprint

The encoder is the memory-cost question. Two options:

1. **Full encoder loaded.** `(d_sae, d_model)` per (layer, hook). For
   d_sae=64k, d_model=4k, fp16: 512 MB per layer per hook. For 28
   layers with one hook each: 14 GB. Prohibitive on a single GPU
   alongside the model.
2. **Clamp-set encoder rows only.** At registration time the operator
   declares which features are clampable for a given module (or the
   union is computed across all named clamp recipes). Only those rows
   of `W_enc` and `b_enc` are loaded. For typical Golden-Gate-style
   work, `|clampable| ≤ 64`, dropping per-(layer, hook) cost to
   `64 · 4096 · 2 bytes = 512 KB`.

We adopt option 2 for the initial implementation. Operators who want
"any feature may be clamped at runtime" can declare the full feature
set explicitly and pay option 1's cost. The decoder is always loaded
in full because feature-deltas can index any decoder row at runtime;
in practice operators who load only a partial encoder will also load
only the matching decoder rows, and the registry enforces that the
encoder/decoder row sets are equal.

## Phased Rollout

1. **Phase 0 — Plumbing only, no kernel.**
   - Add `SteeringModuleKind`, the registry kind discriminator, and
     the broadcast machinery for SAE modules. Existing additive flow
     unchanged.
   - Wire `SAEClampSpec` through `SamplingParams` and the OpenAI
     server. Validation is real; the worker accepts the spec but
     reports "SAE_DELTA not yet implemented" if asked to apply it.
   - Tests: registration, broadcast determinism across mock TP ranks,
     hash-folding for prefix-cache keys, SamplingParams round-trip.

2. **Phase 1A — Eager reference op (math primitive). _Shipped._**
   - `vllm/model_executor/layers/sae_steering.py` exposes
     `apply_sae_delta` and `sae_encode`, a vectorized PyTorch eager
     path that takes per-token clamp tensors directly.  No Triton
     kernel; no layer-hook wiring yet.  Same input shape and dtype
     contract that the Phase-2 Triton kernel will adopt.
   - Activations: ReLU, JumpReLU (`activation_params['threshold']`),
     TopK (`activation_params['k']`).  TopK is "TopK among encoder
     rows passed in" — partial encoders give "TopK in the clampable
     subset"; full-d_sae TopK semantics require loading the full
     encoder.
   - Numeric dtype: encoder/decoder GEMMs in compute dtype; the
     `(n_tokens, n_clamp)` activation tensor promoted to fp32 for
     activation + `delta = clamp(f, target) − f`, cast back before
     the decoder GEMM.
   - Tests: `tests/model_executor/layers/test_sae_steering_op.py` —
     hand-rolled per-(token, feature) reference, all activations
     and clamp variants, dtype contract, shape validation.

3. **Phase 1B — Layer-hook integration.** Split into stages so each
   piece is reviewable on its own; integration touches the additive
   path's hot loops, so each stage gets independent testing.
   - **Stage 1 (shipped).** Worker-side machinery, no decoder-model
     wiring and no worker-mixin integration.  `SAEClampManager` in
     `vllm/v1/worker/sae_clamp_manager.py` — row allocation /
     refcount / strict capacity / deterministic-replay parallel to
     `SteeringManager`, but with no global tier (row 0 is no-op,
     rows 1+ are per-request).  `register_sae_buffers`,
     `apply_layer_sae_delta`, and `populate_sae_clamp_table` in
     `vllm/model_executor/layers/sae_steering.py` — buffer attach
     / detach, the layer-hook shim that gathers per-token clamps
     from the row table and dispatches to `apply_sae_delta`, and
     the projector that writes manager rows back into the per-
     `(layer, hook)` buffers.  Phase 1B Stage 1 enforces "at most
     one SAE module per (layer, hook) site"; multi-module overlap
     is a follow-up.
   - **Stage 2 (shipped).** Worker mixin integration: the Phase-0
     admission `NotImplementedError` is replaced by real validation
     in `_assert_sae_clamps_can_be_applied` (rejects unknown
     modules, uncovered (layer, hook) sites, and unclampable
     features) plus admission via the SAE manager.  SAE-aware
     handlers added to `_register_initial_steering_config`,
     `_handle_steering_transition`, `_refresh_streaming_steering`,
     `_reset_steering_for_resumption`, and
     `_release_finished_steering_configs`; per-step buffer state
     is materialised by `_update_sae_buffers`.  SAE buffers attach
     to owned layers when an SAE module registers (with PP
     filtering on `_locally_owned_layers`) and detach on
     unregister or kind-swap.  New worker-side method
     `attach_sae_weights(name, weights)` injects encoder/decoder
     tensors into the per-(layer, hook) buffers — the test-fixture
     and future-loader entry point.
   - **Stage 3.** Decoder-model wiring: call `apply_layer_sae_delta`
     from the same hook-point sites that already invoke
     `apply_layer_steering`.  End-to-end tiny-model fixture tests
     for prefill and decode.

4. **Phase 2 — Triton kernel + CUDA-graph integration.**
   - Replace the eager body with a Triton kernel under the same
     custom-op shim, mirroring `steering_kernel.py`'s shape.
   - CUDA graph replay test (steering already exercises this; extend
     the suite to cover SAE clamps).

5. **Phase 3 — Real SAE checkpoint integration test.**
   - Add a Gemma-Scope-driven end-to-end test alongside the existing
     `*_real_weights` Gemma 3 tests. Verify "Golden-Gate-style"
     behavior reproduces qualitatively (i.e., a feature known to fire
     on a topic actually drives outputs toward that topic when
     clamped high).

6. **Phase 4 (deferred) — Full-reconstruction variant.**
   - If/when needed, add `SteeringModuleKind.SAE_FULL_RECONSTRUCTION`
     as a separate op. Does not require touching the delta path. Most
     likely needs admission-tier changes because the operation is no
     longer a small additive perturbation and prefix-cache invariants
     differ.

## Resolved Design Decisions

The five questions raised during planning have been answered. Recording
the decisions here, since the *why* needs to outlive the planning
conversation.

- **Composition order.** SAE delta runs **after** the additive op on
  the same hook. SAE sees the additively-steered residual when
  computing live `f`. (This is also what the data-flow section above
  describes.) No strong principle backs it over "SAE first"; chosen
  so the additive path's behavior is unchanged when SAE is added on
  top, rather than the other way around.

- **Encoder pass is spec-driven, per clamp entry.** Each
  `SAEClampEntry` carries a `requires_encoder_pass: bool` flag,
  derived automatically by `__post_init__`:
    - `target_value` set, `only_if_active=False` → encoder pass
      **required** (need live `f_i` to compute the subtraction
      `target_value − f_i`).
    - `target_value` set, `only_if_active=True` → encoder pass
      **required** (need to gate on `f_i > 0`).
    - `additive` set → encoder pass **optional**; defaults to off
      since the delta `additive · W_dec[i]` is independent of `f_i`.
      Users who want clamp-with-floor-on-existing-activation
      semantics can request it explicitly.
  At dispatch, the kernel checks the union of `requires_encoder_pass`
  flags across the active clamp set. If none require it, the encoder
  GEMM is skipped entirely and the path collapses to a sparse
  decoder-direction add — basically the existing additive op's hot
  path with a different row layout.

- **Manifest format: both.** Phase-0 plumbing accepts a vLLM-native
  JSON manifest. Phase-1 ships an SAE Lens / Gemma Scope adapter that
  reads the upstream on-disk layout and synthesizes the native
  manifest internally. The runtime only ever sees the native form.
  This isolates upstream layout churn from the worker's loader.

- **Activations supported at launch: ReLU, JumpReLU, TopK.** No
  batch-TopK, no Gated SAE, no others in the initial scope. The
  activation is a discriminated-union field on
  `SAELayerWeights.activation` so adding more later is a switch-table
  extension, not a rewrite.

- **Numeric dtype: hybrid.** Encoder/decoder GEMMs + W_dec gather in
  the model's compute dtype (bf16/fp16 in most deployments), matching
  Llama Scope's published inference recipe and the existing additive
  op's `get_steering_buffer_dtype` contract. The `(k_clamp,)`
  activation vector and the JumpReLU threshold comparison promote to
  fp32 for the activation function and the `delta = clamp(f, target)
  − f` subtraction, then cast back to compute dtype before the final
  decoder add. One cast on a tiny tensor; no cost on the d_model
  matmuls.

## References

- [Steering (user-facing guide)](steering.md)
- [Steering Runtime Design](../design/steering_runtime.md)
- Templeton et al., *Scaling Monosemanticity*, Anthropic, 2024.
- Lieberum et al., *Gemma Scope*, DeepMind, 2024.
- Kantamneni et al., *Steering Language Model Refusal with Sparse
  Autoencoders*, 2024 (delta intervention reference).
