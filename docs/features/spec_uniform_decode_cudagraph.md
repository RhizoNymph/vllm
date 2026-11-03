# Uniform-decode cudagraphs with speculative decoding

Which steps of a speculative V1 run (`GPUModelRunner`) may replay the FULL
cudagraph captured for uniform decode batches, and why a prompt chunk of the
decode length must not.

## Scope

- `GPUModelRunner._is_uniform_decode` and its caller
  `_determine_batch_execution_and_padding`, on steps with speculative decoding
  (`uniform_decode_query_len = 1 + num_speculative_tokens > 1`).

Not in scope: piecewise / mixed prefill-decode graphs, the V2 runner (its
`get_uniform_decode_token_count` already requires `has_prefill == False`),
cudagraph capture (which forces the decision through `force_uniform_decode`),
runs without speculation (unchanged shape rule).

## Why

The uniform-decode test was shape only: every row schedules
`1 + num_speculative_tokens` tokens. A prompt chunk can have that length too,
e.g. the last chunk of a prompt that `--prefix-match-unit` split at the
prompt's last hash boundary (2 tokens left with MTP k=1), or a chunk cut by the
token budget. The batch was then dispatched to the FULL decode graph.

That graph was captured on the GDN builder's pure spec-decode path, which reads
its state indices, query start locations, token indices and accepted counts
from persistent buffers. The builder refreshes those buffers only for
decode-only batches (`num_prefills == 0`); a batch with a prompt chunk has a
prefill row, so the replay ran on the previous step's buffers. Every request in
the step read and wrote the wrong recurrent-state slots. On Qwen3.5-0.8B with
MTP k=1 and `--prefix-match-unit 16`, 7-8 of 24 concurrent replies diverged
within their first 7 tokens, as a contiguous group (all requests in flight at
the bad step); a request run alone, MTP alone and `--prefix-match-unit` alone
were all fine.

## Data and control flow

1. `execute_model` calls `_determine_batch_execution_and_padding(...,
   num_spec_decode_reqs=self._count_full_spec_decode_reqs(scheduler_output))`.
2. `_count_full_spec_decode_reqs` returns `None` without speculation; otherwise
   the number of rows in `scheduled_spec_decode_tokens` that verify exactly
   `num_speculative_tokens` drafts and schedule `1 + num_speculative_tokens`
   tokens. Rows the scheduler pads with `-1` drafts (a new request starting in
   decode, padded on purpose to stay on this graph) count.
3. `_is_uniform_decode` additionally requires
   `num_spec_decode_reqs in (None, num_reqs)`. A batch with any other row
   falls back to the mixed / piecewise path, where the GDN builder's fresh
   metadata is used directly.
4. `force_uniform_decode` (capture, dummy runs) still overrides the decision.

## Files

| File | Role |
|---|---|
| `vllm/v1/worker/gpu_model_runner.py` | `_is_uniform_decode` (`num_spec_decode_reqs`), `_count_full_spec_decode_reqs` (new), `_determine_batch_execution_and_padding` (`num_spec_decode_reqs` argument), `execute_model` call site |
| `vllm/v1/attention/backends/gdn_attn.py` | (unchanged) the persistent-buffer refresh that is only valid for decode-only batches |
| `tests/v1/worker/test_uniform_decode_spec_prefill.py` | uniform / non-uniform cases |

## Invariants

- With speculative decoding, a step replays the uniform-decode FULL graph only
  if every row verifies a full set of drafts (real or `-1` padding).
- Without speculative decoding the shape rule is unchanged: a one-token prompt
  chunk still runs in the decode graph (the non-spec GDN decode path handles
  it, and the builder refreshes its buffers because it counts it as a decode).
- Correctness never depends on the dispatch: the check only removes FULL
  replays the builder's buffers cannot serve.
