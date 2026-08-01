# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager-CPU tests for the per-row monitor term in ``apply_steering``.

The per-row monitor gates each token's ROW term by that token's own row's
probe + ``[threshold, sharpness]`` (``probe_table[row]``,
``row_params[row]``), decode-only via ``decode_mask``. It is orthogonal to the
global monitor (both multiply into the row gate). These tests exercise the
registered op's CPU branch directly (15-arg signature).
"""

import torch

H = 4
# Off states for the global monitor / tier (so only the per-row path matters).
_GPROBE = torch.zeros(H)
_GPARAMS = torch.tensor([0.0, 1.0, 0.0])  # [thr, sharp, gate_rows]
_OFF = torch.zeros(1, dtype=torch.bool)
_ROW_DEFAULT = (-1.0e30, 1.0)


def _call(
    hidden,
    table,
    index,
    *,
    rprobe,
    rparams,
    ractive,
    decode_mask,
    scales=None,
):
    n, rows = hidden.shape[0], table.shape[0]
    scales = torch.ones(rows) if scales is None else scales
    return torch.ops.vllm.apply_steering(
        hidden,
        table,
        index,
        torch.ones(1, dtype=torch.bool),  # any_active
        scales,
        torch.zeros(H),  # dynamic vec
        torch.zeros(n),  # token scales (tier off)
        torch.ones(n),  # row gate
        _GPROBE,
        _GPARAMS,
        _OFF,  # global monitor off
        decode_mask,
        rprobe,
        rparams,
        ractive,
    )


def _rparams(rows, entries):
    """Build a ``(rows, 2)`` params table, default ungated, then override."""
    thr0, sharp0 = _ROW_DEFAULT
    p = torch.tensor([thr0, sharp0]).expand(rows, 2).clone()
    for row, (thr, sharp) in entries.items():
        p[row, 0] = thr
        p[row, 1] = sharp
    return p


def test_per_row_gate_only_affects_own_row():
    # Two requests routed to rows 3 and 4, same probe direction (+x0), but
    # row 4's threshold is huge ⇒ its gate ≈ 0 (add suppressed). Row 3 fires.
    rows = 5
    table = torch.zeros(rows, H)
    table[3] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # request A add vector
    table[4] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # request B add vector
    index = torch.tensor([3, 4])
    hidden = torch.zeros(2, H)
    hidden[:, 0] = 1.0  # x0 = 1 for both
    rprobe = torch.zeros(rows, H)
    rprobe[3] = torch.tensor([10.0, 0.0, 0.0, 0.0])  # A: score 10
    rprobe[4] = torch.tensor([10.0, 0.0, 0.0, 0.0])  # B: score 10
    rparams = _rparams(rows, {3: (0.0, 5.0), 4: (1.0e30, 5.0)})  # B thr huge

    out = _call(
        hidden,
        table,
        index,
        rprobe=rprobe,
        rparams=rparams,
        ractive=torch.ones(1, dtype=torch.bool),
        decode_mask=torch.ones(2),  # both decode
    )
    # A (row 3): gate≈1 ⇒ +1 on x0.
    assert abs(out[0, 0].item() - 2.0) < 1e-4
    # B (row 4): gate≈0 ⇒ add suppressed.
    assert abs(out[1, 1].item()) < 1e-4


def test_per_row_gate_decode_only():
    # A prefill token (decode_mask=0) must never be gated, even when its
    # probe would fire — prefill rows feed prefix-cache keys.
    rows = 4
    table = torch.zeros(rows, H)
    table[3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    index = torch.tensor([3, 3])
    hidden = torch.zeros(2, H)
    hidden[:, 0] = -5.0  # score very negative ⇒ gate ≈ 0 for decode
    rprobe = torch.zeros(rows, H)
    rprobe[3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    rparams = _rparams(rows, {3: (0.0, 5.0)})
    out = _call(
        hidden,
        table,
        index,
        rprobe=rprobe,
        rparams=rparams,
        ractive=torch.ones(1, dtype=torch.bool),
        decode_mask=torch.tensor([1.0, 0.0]),  # tok0 decode, tok1 prefill
    )
    # tok0 (decode): gate≈0 ⇒ add suppressed (~hidden).
    assert abs(out[0, 0].item() - hidden[0, 0].item()) < 1e-3
    # tok1 (prefill): ungated ⇒ full add (hidden + 1).
    assert abs(out[1, 0].item() - (hidden[1, 0].item() + 1.0)) < 1e-4


def test_inactive_flag_skips_per_row_gate():
    # With ractive False, the per-row buffers must be ignored entirely (even
    # a firing probe leaves the row at full strength).
    rows = 4
    table = torch.zeros(rows, H)
    table[3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    index = torch.tensor([3])
    hidden = torch.zeros(1, H)
    hidden[0, 0] = -5.0
    rprobe = torch.zeros(rows, H)
    rprobe[3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    rparams = _rparams(rows, {3: (0.0, 5.0)})
    out = _call(
        hidden,
        table,
        index,
        rprobe=rprobe,
        rparams=rparams,
        ractive=torch.zeros(1, dtype=torch.bool),  # OFF
        decode_mask=torch.ones(1),
    )
    assert abs(out[0, 0].item() - (hidden[0, 0].item() + 1.0)) < 1e-4


def test_default_params_pass_through():
    # A configured-active site whose row has the DEFAULT params (thr -1e30)
    # gates to 1.0 ⇒ full add, regardless of activation.
    rows = 4
    table = torch.zeros(rows, H)
    table[3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    index = torch.tensor([3])
    hidden = torch.zeros(1, H)
    rprobe = torch.zeros(rows, H)  # row 3 probe is zero (unconfigured)
    rparams = _rparams(rows, {})  # all default
    out = _call(
        hidden,
        table,
        index,
        rprobe=rprobe,
        rparams=rparams,
        ractive=torch.ones(1, dtype=torch.bool),
        decode_mask=torch.ones(1),
    )
    assert abs(out[0, 0].item() - 1.0) < 1e-4


def test_composes_with_row_scale():
    # The per-row gate multiplies the scaled row: out = hidden + table*scale*gate.
    rows = 4
    table = torch.zeros(rows, H)
    table[3] = torch.tensor([2.0, 0.0, 0.0, 0.0])
    index = torch.tensor([3])
    hidden = torch.zeros(1, H)
    hidden[0, 0] = 10.0  # strong probe fire
    rprobe = torch.zeros(rows, H)
    rprobe[3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    rparams = _rparams(rows, {3: (0.0, 50.0)})  # sharp ⇒ gate≈1
    scales = torch.ones(rows)
    scales[3] = 0.5
    out = _call(
        hidden,
        table,
        index,
        rprobe=rprobe,
        rparams=rparams,
        ractive=torch.ones(1, dtype=torch.bool),
        decode_mask=torch.ones(1),
        scales=scales,
    )
    # hidden 10 + table 2 * scale 0.5 * gate 1 = 11.
    assert abs(out[0, 0].item() - 11.0) < 1e-3


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-v"]))
