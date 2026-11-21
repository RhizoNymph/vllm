# Transformers Backend Interventions

Steering, activation capture, activation patching, directional clamps, dynamic
steering, and SAE steering on decoder layers built by the Transformers modeling
backend (`vllm/model_executor/models/transformers/`), with the same hook points
and the same semantics as native vLLM decoder layers.

## Scope

In scope:

- Every decoder layer of a model built through `Base` (causal LM, MoE,
  multimodal language models, pooling/embedding models), on the pipeline
  ranks that own it.
- The five standard hook points: `pre_attn`, `post_attn`, `mlp_in`,
  `mlp_out`, `post_block`.
- Two residual layouts: *sequential* (an attention residual add followed by an
  MLP residual add, with any norms, dropouts or multipliers on the branches)
  and *parallel* (one residual add over an attention and an MLP branch).

Not in scope:

- MTP draft models (`MTPMixin.supports_interventions = False`).
- Vision/audio towers of multimodal models.
- Multi-stream (mHC) hook points.
- Layers whose residual adds live inside a submodule (e.g. BERT's
  `BertSelfOutput`), use in-place updates, loops, or helper methods over the
  residual, or have a third residual add (cross-attention). These raise
  `UnsupportedInterventionLayout` when an intervention is configured.

## Data and control flow

1. `Base.__init__` builds the HF model, runs `recursive_replace` (vLLM
   linears, fusers, MoE) and `_create_attention_instances` (attaches a vLLM
   `Attention` as `attn` on each attention module), then calls
   `Base._install_interventions()`.
2. `_install_interventions` returns immediately unless
   `interventions_requested(vllm_config)` (any of `steering_config`,
   `patch_config`, `capture_consumers_config`), so servers without
   interventions run the upstream forward bit-for-bit. Otherwise it requires
   `_decoder_layers` and calls `install_interventions` for
   `range(*self._pp_indices())`.
3. `install_interventions` gets a hooked forward per layer from
   `hooked_forward` (cached per layer class and child structure), checks
   `layer_idx`, then for each layer sets `layer_idx`, calls
   `register_steering_buffers` (steering, patch, clamp buffers), binds the
   forward on the instance, and finally `share_steering_index_across_layers`.
4. `analysis.analyze_layer` parses the class's forward source
   (`fx_utils.recover_forward`) and abstractly interprets it, path by path
   through every `if`. Each local name is tracked as the residual stream (with
   its add depth) or as a value derived from layer children. Children are
   classified as attention (subtree holds an `AttentionLayerBase`), heavy
   (projections or matrix parameters; heavy non-attention = MLP), or neither
   (norms, dropouts). An add with exactly one stream operand and only
   attention/MLP-derived other operands is a residual add. Calls to
   `self.<attr>` that the instance lacks or holds as `None` make a path
   infeasible (dropped). Each feasible path must end returning the final
   stream and match one of the two layouts; the result is a `LayerPlan`.
5. `rewrite.rewrite_forward` edits the plan's AST: the MLP call's input is
   wrapped in `mlp_input`, and each residual add is replaced by
   `attn_residual_add`, `mlp_residual_add`, or `parallel_residual_add`. The
   function is compiled in the HF module's namespace, which gains one global,
   `_vllm_intervention_hooks`.
6. At run time `hooks` flattens each `[1, tokens, hidden]` tensor to
   `[tokens, hidden]` and calls `apply_layer_steering` /
   `apply_block_steering`, which run capture → patch → steer → SAE → clamp
   exactly as on native layers.

### Hook-point semantics

Sequential `h = x + attn(n1(x)); out = h + mlp(n2(h))`:

| Hook | Site | Effect of an additive steer |
| --- | --- | --- |
| `pre_attn` | skip operand `x` of the attention add | flows into `h` and the MLP input; not this layer's attention input |
| `post_attn` | skip operand `h` of the MLP add | added to the output; not the MLP input |
| `mlp_in` | argument of the MLP call | passes through the MLP |
| `mlp_out` | MLP branch operand (after post-norms, dropout, multipliers) | added to the output |
| `post_block` | `h + mlp_branch` via `apply_block_steering` | added to the output |

This matches native fused-residual layers (Llama, Qwen3), where the norm that
feeds a sublayer is computed before the residual is steered.

Parallel `out = x + attn(n(x)) + mlp(n(x))` is evaluated as
`(x + attn) + mlp`: `pre_attn` on `x`, `post_attn` on `x + attn` (as the native
Phi layer did), `mlp_out` on the MLP branch, `post_block` on the sum. The
re-association can change outputs by rounding relative to the unhooked model.

## Files

| File | Role | Key exports |
| --- | --- | --- |
| `vllm/model_executor/models/transformers/interventions/__init__.py` | Public API | re-exports below |
| `.../interventions/plan.py` | Layout types | `LayerPlan`, `SequentialResidual`, `ParallelResidual`, `MlpInputSite`, `ResidualLayout` |
| `.../interventions/analysis.py` | Source analysis | `analyze_layer`, `classify_children`, `is_vllm_attention`, `MAX_PATHS` |
| `.../interventions/rewrite.py` | AST rewrite + compile | `rewrite_forward`, `HOOKS_NAME` |
| `.../interventions/hooks.py` | Runtime sites | `attn_residual_add`, `mlp_residual_add`, `parallel_residual_add`, `mlp_input` |
| `.../interventions/install.py` | Buffers, binding, gating | `install_interventions`, `hooked_forward`, `interventions_requested`, `BufferSizes` |
| `.../interventions/errors.py` | Typed failure | `UnsupportedInterventionLayout(layer_cls, reason)` |
| `vllm/model_executor/models/transformers/base.py` | Calls the installer | `Base._install_interventions`, `Base.supports_interventions` |
| `vllm/model_executor/models/transformers/mtp.py` | Opts drafts out | `MTPMixin.supports_interventions = False` |
| `tests/models/transformers/interventions/` | CPU tests | synthetic layers, real HF architectures, `Base` wiring |

## Invariants and constraints

- No intervention configured ⇒ no buffers, no rewrite, upstream forward.
- With nothing active, a sequential layer's output is bitwise identical to the
  unhooked forward; a parallel layer's differs only by re-association.
- Hook sites only call the native custom ops on pre-registered buffers; no
  data-dependent Python branches, so `torch.compile` (fullgraph) and CUDA
  graphs behave as for native hooks.
- Every tensor at a hook site is fully reduced under TP: the backend's
  row-parallel linears and fused MoE reduce their own outputs, and no fusion
  defers a reduction past a residual add. A future fusion that does would
  have to be disabled when interventions are configured, as for Qwen3.5.
- Only PP-local layers are hooked; `layer_idx` is the global layer index.
- Analysis either proves the layout or raises; it never guesses. Every layer
  is analysed before any is modified.
- The rewritten forward is cached per (class, child names, attention
  children, heavy children), since optional children select feasible paths.
