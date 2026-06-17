# Plan: per-request-row token gating (Phase 2 follow-up)

Status: **PLAN.** Extends the Phase 2 in-graph monitor (§8 of
`dynamic_steering.md`) to gate **per-request rows** per token, not just the
§5.4 dynamic tier.

## 1. What exists vs. what this adds

The steering kernel computes, per token `t` with row `r = index[t]`:

```
out = hidden + table[r]·scale[r]            +  dvec·token_scales[t]
              └─── per-request row gather ──┘    └──── dynamic tier ────┘
```

The Phase 2 monitor writes `token_scales[t]` (a per-token gate ∈ [0, gain])
so it conditions **only the tier**. This adds a per-token gate on the
**row gather** too:

```
out = hidden + table[r]·scale[r]·row_gate[t] + dvec·token_scales[t]
```

Use case: a request admitted with per-request config X is steered by X
only on tokens where a probe trips — per-request *and* token-conditional,
same forward.

## 2. Why rows can't reuse the tier's gate mechanism

The tier gate is prefill-safe because the runner writes `token_scales=0`
for prefill and the monitor *multiplies* (`0·gate=0`). Rows are different:
a prefill row must apply at **full strength** (1.0) — prefill steering
feeds prefix-cache keys (§7), and the keys do **not** include monitor
params, so a data-dependent prefill-row gate would corrupt APC. There is
no base `b` with `b·gate = 1.0` for arbitrary `gate`, so the
multiply-preserves-prefill trick cannot keep prefill rows at 1.0.

**Therefore in-graph row gating must skip prefill positions**, which
requires the monitor to know which positions are decode. Hence a decode
mask.

## 3. Design (decided)

Two new shared per-token buffers (sized `max_tokens`, like
`steering_index`/`token_scales`):

- **`steering_row_gate`** — fp32, default **1.0**. The kernel multiplies
  the row term by it. The runner resets it to 1.0 each step (rows apply at
  full strength by default); the monitor reduces it for decode tokens.
- **`steering_decode_mask`** — fp32, default **0.0**. The runner writes
  1.0 for decode tokens, 0.0 for prefill. The monitor reads it to protect
  prefill.

**Kernel** (`_apply_steering_kernel`): load `rgate = row_gate[t]`; row term
becomes `t_vals·scale·rgate`. Default 1.0 ⇒ identical to today.

**Monitor op** (`steering_monitor`): after computing the per-token gate
`g[t] = sigmoid(sharpness·(score − threshold))`, in addition to
`token_scales[t] *= g[t]` (tier, unchanged), also
`row_gate[t] *= mask[t]·g[t] + (1 − mask[t])` — i.e. decode→`·g`,
prefill→unchanged (×1). Gated behind a per-(layer,hook) `gate_rows` flag
in the monitor params so existing tier-only monitors are unaffected. The
write lands at the monitor site; rows at layers > L read the gated value,
layers < L read the runner's 1.0 (detect at L, gate rows at layers > L) —
same ordering contract as the tier gate.

**Runner** (`_update_steering_buffers`): write `steering_row_gate` = 1.0
for the active prefix (+1.0 tail), and `steering_decode_mask` = 1.0 for
decode tokens / 0.0 for prefill — reusing the same per-request →
`np.repeat` → pinned-H2D pattern as the index/token_scales. The decode
mask falls out of the prefill/decode branch already computed in the loop.

**Manager**: monitor config gains a `gate_rows: bool`; populate writes it
into the monitor params buffer. `set_monitor(..., gate_rows=False)`
default preserves current behaviour.

**Cache safety** (§7): row gating is **decode-only** by construction
(`decode_mask=0` ⇒ `row_gate` stays 1.0 for prefill). Decode rows are
already keyed by the effective-decode-steering signature (the APC
notification); the monitor gate is deterministic given tokens + monitor
params (already folded into that signature), so reuse stays correct — no
new APC work.

## 4. Op-signature impact

- `apply_steering` / `apply_steering_triton` / fake / eager / warmup gain
  `steering_row_gate` (7 → 8 args).
- `steering_monitor` gains `steering_decode_mask` + `steering_row_gate`
  (5 → 7 args; mutates `token_scales` **and** `row_gate`).
- `apply_layer_steering` threads both shared buffers (shared in
  `_init_steering_state`, like `token_scales`).

## 5. Milestones

- **M1 — kernel + buffers.** `row_gate` term + buffers + share helper +
  warmup; CPU op tests (row gate 1.0 == today; <1 scales the row;
  prefill row untouched).
- **M2 — monitor row gating.** `gate_rows` param; monitor writes
  `row_gate` from `g`+`mask`; manager `set_monitor(gate_rows=)`; runner
  writes `row_gate`/`decode_mask`. CPU tests (decode row gated, prefill
  row 1.0, tier still gated; monitor without `gate_rows` leaves row_gate
  1.0).
- **M3 — GPU validation.** Engage/disengage on a per-request config row
  via the monitor (logprobs), prefill unaffected, cudagraph capture with
  the 8-arg op; eager-vs-graph parity.

## 6. Out of scope

Sync-tier (off-graph, 1-step) per-request row-gate scalars — the runner
could also drive `row_gate` per request from a consumer without a monitor.
Cheap to add later on the same buffer; not needed for the in-graph story.
