# Neuron Clamp Design

Per-request MLP neuron clamps override coordinates of the SiLU-gated MLP
activation (the `down_proj` input) at chosen decoder layers. They reuse the
steering runtime end to end: a clamp is part of a request's phase-effective
steering config. User-facing docs are in
[Activation Steering](../features/steering.md#neuron-clamps); the runtime
they build on is described in [Steering Runtime Design](steering_runtime.md).

## Scope

In scope:

- `set` / `add` / `scale` of single neurons, per request, per phase
  (`both` / `prefill` / `decode`), alone or together with steering vectors
- exact zero/mean ablation (`set` is written exactly)
- TP and PP under the existing determinism contract
- `torch.compile` + CUDA graph replay (live tables, no recapture)
- prefix-cache correctness (prefill clamps fork prompt KV)

Not in scope:

- global (server-wide) clamps: rows 1/2 never carry clamps
- clamps on dynamic-override rows: a request with decode clamps rejects
  dynamic per-request overrides instead of silently losing its clamps
- models other than native `qwen3.py` decoders (`Qwen3ForCausalLM`,
  through `Qwen3MLP`); other architectures, including `Qwen2ForCausalLM`
  (now served by the Transformers backend), are rejected at the input
  processor
- capturing the MLP activation (the reserved `mlp_in` capture hook is still
  unwired)
- the V2 model runner (steering itself is V1-only)

## Data and control flow

```text
request.neuron_clamps (pydantic NeuronClampSpec, HTTP 400 on schema errors)
  -> SamplingParams.neuron_clamps (wire form: list of dicts)
       __post_init__: coerce_neuron_clamps -> NeuronClamp tuple
                      (overlapping duplicates -> NeuronClampValidationError)
       effective_{prefill,decode}_neuron_clamps (phase filter, sorted)
       {prefill,decode}_steering_config_hash
           = hash_steering_config(vectors, module_ref, ..., neuron_clamps=phase clamps)
       {prefill,decode}_additive_steering_config_hash (the physical-row
           identity) folds the phase clamps in too
  -> InputProcessor._validate_neuron_clamps
       steering enabled, supported architecture, layer/neuron range,
       per-layer per-phase count <= max_neuron_clamps_per_layer
  -> Request.{prefill,decode}_steering_config_hash
       scheduler admission against max_steering_configs (unchanged)
       prefix-cache extra keys use the prefill hash (unchanged)
  -> worker: SteeringModelRunnerMixin._register_request_phase_config
       registers (hash, phase) when it has vectors, directional clamps OR
       neuron clamps (clamp-only configs pass empty vectors); every
       admission / transition site goes through it
  -> SteeringManager.register_config(..., neuron_clamps=...)
       config_neuron_clamps[(hash, phase)] = clamps, on a fresh row and on a
       row aliased by content (same additive hash => same clamps)
  -> SteeringManager.populate_steering_tables (on content/membership dirty)
       _populate_neuron_clamp_tables: per layer, per physical row (aliased
       configs written once),
       global neuron -> TP-local slot, (mul, add) affine form,
       any_active flag per layer
  -> _update_steering_buffers writes the shared steering_index (unchanged)
  -> Qwen3MLP.forward:
       act = SiluAndMul(gate_up_proj(x))
       torch.ops.vllm.apply_neuron_clamp(act, index, mul, add,
                                         steering_index, any_active)
       down_proj(act)
```

The clamp is an affine map `x * mul + add` per slot: `set` is
`(0, value)` (kernels write `add` directly when `mul == 0`, so a non-finite
activation cannot leak through `0 * x`), `add` is `(1, value)`, `scale` is
`(value, 0)`. Arithmetic is fp32 with a single rounding to the activation
dtype.

## Buffers

Registered on each clamp-capable MLP module by
`register_decoder_mlp_neuron_clamp` (only when steering is enabled and
`max_neuron_clamps_per_layer > 0`):

| Buffer | Shape / dtype | Meaning |
| --- | --- | --- |
| `neuron_clamp_index` | `(rows, K)` int32 | TP-local neuron, `-1` = empty slot |
| `neuron_clamp_mul` | `(rows, K)` fp32 | affine multiplier |
| `neuron_clamp_add` | `(rows, K)` fp32 | affine addend |
| `neuron_clamp_any_active` | `(1,)` bool | any row has a slot on this layer |
| `steering_index` | shared | same tensor as the decoder layers' index |

`rows` equals the steering table row count (`max_steering_configs +
max_dynamic_steering_configs + 3`), so the shared token-to-row index
addresses both. `K` is `--max-neuron-clamps-per-layer`. Plain attributes
`neuron_clamp_tp_rank` / `neuron_clamp_local_size` come from the
row-parallel `down_proj` (`tp_rank`, `input_size_per_partition`).

## The op

`apply_neuron_clamp` mutates the activation in place (`mutates_args=["act"]`;
the activation is a fresh intermediate) and returns nothing, so there is no
copy of the `(tokens, intermediate)` tensor. On CUDA a Triton kernel runs
one program per token, exits immediately when `any_active` is clear, and
otherwise rewrites at most `K` scalars of the row. On CPU an eager
reference implementation runs. The op is warmed at every captured batch
size in `_init_steering_state` so no Triton JIT happens during graph
capture.

## Files

| File | Role |
| --- | --- |
| `vllm/config/neuron_clamp_types.py` | `NeuronClamp`, `NeuronClampValidationError`, coercion/duplicate checks, phase filter, hash encoding, model-range validation, `global_to_local_neuron` |
| `vllm/config/steering_types.py` | `hash_steering_config(..., neuron_clamps=)` |
| `vllm/config/steering.py` | `SteeringConfig.max_neuron_clamps_per_layer` (in the compile hash) |
| `vllm/engine/arg_utils.py` | `--max-neuron-clamps-per-layer` |
| `vllm/sampling_params.py` | `neuron_clamps` field, parsed/effective clamp properties, hashes |
| `vllm/entrypoints/openai/steering/neuron_clamp_protocol.py` | `NeuronClampSpec` request schema |
| `vllm/entrypoints/openai/{chat_completion,completion}/protocol.py` | `neuron_clamps` request field |
| `vllm/v1/engine/input_processor.py` | model/engine-dependent validation |
| `vllm/model_executor/layers/neuron_clamp.py` | buffers, `apply_mlp_neuron_clamp`, custom op, eager reference |
| `vllm/model_executor/layers/neuron_clamp_kernel.py` | Triton kernel + warmup |
| `vllm/model_executor/layers/steering.py` | `share_steering_index_across_layers` re-points the MLP index |
| `vllm/model_executor/models/qwen3.py` | wiring in `Qwen3MLP.forward`, `Qwen3DecoderLayer` and `Qwen3Model` |
| `vllm/v1/worker/steering_manager.py` | `config_neuron_clamps`, clamp table population, flag clearing |
| `vllm/v1/worker/steering_model_runner_mixin.py` | `_register_request_phase_config`, warmup, nothing-active flag clear, override rejection |

## Invariants

- A phase with clamps has a nonzero hash; a phase without clamps hashes
  exactly as before clamps existed (prefix-cache keys of vector-only
  requests are unchanged).
- At most one clamp per `(layer, neuron)` per phase (validated), so table
  slots never alias within a row, and at most `K` per layer per phase, so a
  row never overflows (the manager raises if it ever would).
- Rows 0, 1, 2 and dynamic-override rows hold no clamp slots.
- Configs share a physical row only when their additive hash, which
  includes the neuron clamps, matches; populate writes each physical row's
  clamps once, so `add` / `scale` never stack across aliases, and a row
  keeps its clamps while any alias is live.
- `NeuronClampValidationError` is both a `VLLMValidationError` (HTTP 400)
  and a `ValueError` (pydantic field errors).
- Every rank allocates the same rows for the same configs (determinism
  contract); each rank fills only the slots for neurons it owns, so exactly
  one TP rank clamps each neuron and PP stages skip layers they do not own.
- `any_active` is set iff the layer has at least one slot on this rank; it
  is cleared on the runner's nothing-active transition, where populate does
  not run.
- Tables are only rewritten on config churn (the manager's content /
  membership dirty flags), and only for layers that have or had entries.
