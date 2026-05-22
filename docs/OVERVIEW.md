# Repository Overview

This repository is a fork of [vllm-project/vllm](https://github.com/vllm-project/vllm)
that adds **activation steering** as a first-class runtime feature.
This overview is scoped to fork-specific work; for general vLLM
internals, refer to the upstream documentation under `docs/design/`,
`docs/features/`, and `docs/serving/`.

## Description

Goal: serve large language models with the standard vLLM stack
(continuous batching, prefix caching, `torch.compile`, CUDA graphs,
TP/PP) **and** allow injection of activation steering — both
precomputed additive vectors (today) and SAE-based feature surgery
(see [`features/sae_steering.md`](features/sae_steering.md))
— into the residual stream of decoder layers, with phase-aware
per-request and global tiers.

## Subsystems

The fork-specific runtime spans four cooperating subsystems:

1. **Hook-point library**
   `vllm/model_executor/layers/steering.py` plus
   `vllm/model_executor/layers/steering_kernel.py`. Defines the
   per-(hook, layer) buffer layout, the `apply_steering` custom op,
   and the Triton fast-path. Decoder-layer modules call
   `apply_layer_steering(...)` at three positions in the residual
   stream: `pre_attn`, `post_attn`, `post_mlp`.

2. **Per-worker steering manager**
   `vllm/v1/worker/steering_manager.py`. Owns the row allocator and
   ref-counter for per-request configs, plus the global-tier vector
   cache. Materializes the per-layer table buffers each step from
   the (small) cached state. Shared-nothing across ranks; row IDs
   stay synchronized via deterministic replay rather than any
   cross-rank collective.

3. **Model-runner integration**
   `vllm/v1/worker/steering_model_runner_mixin.py`. Threads steering
   through admission, phase transitions (prefill→decode), capture
   consumers, and resumption. Drives `populate_steering_tables`
   exactly when the manager is dirty.

4. **API surface**
   - Global steering: `vllm/entrypoints/serve/steering/api_router.py`
     (`POST /v1/steering/{set,clear}`, `GET /v1/steering`,
     `GET /v1/steering/layers`).
   - Named modules: `vllm/entrypoints/openai/steering/registry.py`
     and `vllm/entrypoints/serve/steering/modules_router.py`.
   - Per-request steering: fields on `SamplingParams`, surfaced
     through the OpenAI-compatible server's `extra_body`.
   - CLI: `--enable-steering`, `--max-steering-configs`,
     `--steering-modules`, `--steering-api-key`.

## Data Flow

Across one inference step:

```
Engine receives request
   ├── SamplingParams.steering_* validated and hashed
   │     -> prefill_steering_config_hash, decode_steering_config_hash
   └── Scheduler reserves additive and/or SAE steering row capacity
       before dispatch

Worker receives SchedulerOutput
   ├── Mixin: register/release rows for new+finished requests
   ├── Mixin: phase-transition handler (prefill -> decode) re-binds row
   ├── Mixin: populate_steering_tables(...) if manager dirty
   │     -> writes per-layer steering_table_{pre_attn,post_attn,post_mlp}
   └── Mixin: writes steering_index[t] for each token slot

Model forward (under torch.compile / CUDA graph):
   for each decoder layer:
       residual = apply_layer_steering(residual, PRE_ATTN)
       ... attention ...
       residual = apply_layer_steering(residual, POST_ATTN)
       ... mlp ...
       residual = apply_layer_steering(residual, POST_MLP)
   # apply_layer_steering = additive gather/add by steering_index,
   # followed by SAE delta when SAE buffers are attached.

Engine collects outputs
   └── On request finish: SteeringManager.release_config drops refcounts
```

Global API mutations are broadcast to every worker via
`collective_rpc`. Prefix-cache keying for prefill-affecting tiers is
folded into the standard cache hash.

## Features Index

### Activation Steering (additive vectors)

- description: Precomputed direction vectors added into the residual
  stream at one of three hook points per layer. Three-tier additive
  composition (base / prefill / decode), phase-aware admission,
  named-module registry, distributed under TP/PP.
- entry_points:
    - `vllm.model_executor.layers.steering.apply_layer_steering`
    - `vllm.v1.worker.steering_manager.SteeringManager`
    - `POST /v1/steering/set`, `POST /v1/steering/modules/register`
    - `SamplingParams.steering_vectors` (and `prefill_*`, `decode_*`,
      `steering_module_ref`)
- depends_on: vLLM scheduler, prefix cache, custom-op machinery, TP/PP
  comms, `torch.compile`/CUDA graph integration.
- doc: [`features/steering.md`](features/steering.md)
- design: [`design/steering_runtime.md`](design/steering_runtime.md)

### SAE-Based Steering (delta / feature surgery)

- description: Per-(layer, hook) SAE feature surgery — encode the
  live residual, replace a small set of feature activations with
  caller-supplied clamp values, add the resulting decoder-direction
  delta back into the residual. Composes additively with the
  existing additive tier; uses the same admission, TP/PP, and
  prefix-cache machinery. Adopts the "delta intervention" variant
  (most follow-up SAE-steering work) rather than the
  reconstruction-replacement variant from Anthropic's Scaling
  Monosemanticity, whose runtime integration remains follow-up work.
- entry_points:
    - `vllm.model_executor.layers.sae_steering.apply_layer_sae_delta`
    - `vllm.v1.worker.sae_clamp_manager.SAEClampManager`
    - `POST /v1/steering/modules/register` with
      `kind: "sae_delta"`
    - `SamplingParams.sae_clamp_specs`
- depends_on: Activation Steering runtime patterns (runner mixin,
  named-module registry, custom-op shim, scheduler admission, and
  prefix-cache key folding). SAE rows are owned by a parallel
  `SAEClampManager`.
- doc: [`features/sae_steering.md`](features/sae_steering.md)

## Where to Read Next

- New to the steering runtime? Start with
  [`features/steering.md`](features/steering.md) for the user-facing
  surface, then [`design/steering_runtime.md`](design/steering_runtime.md)
  for invariants and the distributed contract.
- Working on the SAE path? Read
  [`features/sae_steering.md`](features/sae_steering.md) and verify
  the open design questions there before writing code.
- Working on a non-fork part of vLLM? Use the upstream docs index at
  [`README.md`](README.md).
