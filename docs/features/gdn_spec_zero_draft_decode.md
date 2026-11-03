# Linear-attention state for decodes without a draft

Speculative decoding (n-gram, or any proposer that can return fewer drafts
than `num_speculative_tokens`) on hybrid models with GatedDeltaNet / Mamba
layers (Qwen3.5 / 3.6, Qwen3-Next, Bailing linear attention).

## Scope

- Which rows of a step the linear-attention metadata builders run on their
  spec-decode path, and therefore where each row reads its initial recurrent
  (SSM) and conv state.
- The V1 runner (`GPUModelRunner`) and the V2 runner
  (`MambaHybridModelState`) inputs to those builders.

Not in scope: how drafts are proposed or verified, the mamba "align"
prefix-cache copies (`mamba_utils.preprocess_mamba` /
`postprocess_mamba_align_gpu`, unchanged), FULL-cudagraph dispatch (see
`spec_uniform_decode_cudagraph.md`).

## Why

During a verification step the GDN kernels write the state after each scheduled
token `t` of a row into slot `t` of the row's `1 + num_speculative_tokens` state
slots, and the conv kernel leaves a rolling window whose valid part starts at
offset `t`. After `a` tokens are accepted, the row's state is in slot `a - 1`.
The next step must read it there: the spec-decode path does, through
`num_accepted_tokens`; the non-spec decode/prefill path always reads slot 0
and offset 0.

Before this change a row was a spec-decode row only when the scheduler gave it
drafts this step, and the GDN builder fell back to the non-spec path for the
whole batch when no row had a draft. With n-gram speculation a row often gets
no draft (no n-gram match) right after a step that accepted several tokens; it
then resumed from slot 0, i.e. from the state after the first accepted token,
and silently dropped the others from its recurrent state. Greedy n-gram
decoding diverged from non-speculative decoding after a few accepted drafts
(Qwen3.5-0.8B, 24 real requests: 0/24 replies identical, with or without
prefix caching). MTP / EAGLE always propose the full draft count, so they never
hit this.

## Data and control flow

1. `GPUModelRunner._prepare_inputs` calls `_stage_num_decode_draft_tokens` on
   every step when speculative decoding is configured (previously only when
   some row had drafts).
2. `_stage_num_decode_draft_tokens` fills `self.num_decode_draft_tokens`
   (CPU + GPU buffer) per row:
   - the row's draft count when it is in `scheduled_spec_decode_tokens` and
     schedules exactly `drafts + 1` tokens (unchanged);
   - `0` when it schedules one token, has no drafts and its prompt is fully
     computed (`num_computed_tokens >= num_prompt_tokens`): a decode without a
     draft;
   - `-1` otherwise (prompt chunks).
3. `execute_model` passes `use_spec_decode = speculative_config is not None` to
   `_build_attention_metadata`, so GDN / Mamba2 / Bailing builders get
   `num_accepted_tokens` and `num_decode_draft_tokens_cpu` on every step.
4. `GDNAttentionMetadataBuilder.build` takes the spec path whenever any row has
   an entry `>= 0` (previously: whenever the sum of drafts was positive). A
   zero-draft row is a spec row with a one-token query; its state slots are
   `block_table[:, :1 + num_spec]` and it starts from slot
   `num_accepted_tokens - 1`, writing its new state to slot 0, which the next
   step reads with `num_accepted_tokens == 1`.
5. V2: `MambaHybridModelState` builds the same per-row array: rows that are
   not prefilling and schedule `drafts + 1` tokens (drafts may be 0) are spec
   rows; `num_draft_tokens_per_req is None` (no drafts this step) no longer
   turns every row into a non-spec row.

## Files

| File | Role |
|---|---|
| `vllm/v1/worker/gpu_model_runner.py` | `_stage_num_decode_draft_tokens` (new), `_prepare_inputs` calls it every spec step; `execute_model` passes the spec-decode builder inputs whenever speculation is on |
| `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | V2 equivalent in `prepare_attn` |
| `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionMetadataBuilder.build`: spec path iff any row `>= 0` |
| `tests/v1/attention/test_gdn_metadata_builder.py` | zero-draft classification cases, accepted-slot propagation |
| `tests/v1/worker/test_stage_num_decode_draft_tokens.py` | row marking of `_stage_num_decode_draft_tokens` |

## Invariants

- A decode row of a speculative hybrid model always runs the spec-decode path
  of the linear-attention kernels, draft or not; only prompt chunks run the
  non-spec path. `num_accepted_tokens` is therefore always consumed after a
  verification step.
- `num_decode_draft_tokens[i] >= 0` iff row `i` is a decode row; the value is
  its draft count. `-1` marks prompt chunks (not `0`: guided decoding may roll
  back speculative tokens of a chunk).
- A one-token prompt chunk (`num_computed_tokens < num_prompt_tokens`) is not a
  decode row: it may have no initial state, which only the prefill path zeroes.
- Steps where no row has a draft now build the spec metadata (one-token spec
  queries); such batches have no uniform `1 + k` decode shape, so they run
  piecewise rather than as a FULL decode cudagraph.
