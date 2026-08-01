# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the in-graph steering monitor op (Phase 2, §8).

The monitor runs at a probe site (one ``(layer, hook)``), reads the
rank-identical pre-steering residual, and writes a per-token gate into
the shared ``steering_token_scales`` buffer that the §5.4 dynamic tier
multiplies by. Its job is to turn the sync/async-set flat decode gain
into a *per-token* gate computed from a probe, same forward
(detect at L, steer at layers > L):

    score[i] = hidden[i] @ probe                       # (n,)
    gate[i]  = sigmoid(sharpness * (score[i] - threshold))
    token_scales[i] *= gate[i]                          # read-modify-write

Multiply (not overwrite) is deliberate: the runner writes
``token_scales`` fresh each step (``gain`` for decode tokens, ``0`` for
prefill), which is the per-step reset; the monitor modulates within the
step. Prefill cache-safety (§7) falls out of ``0 * gate == 0`` regardless
of what the probe says.

These tests exercise the tensor math directly and the op's CPU eager
implementation (importable without the full CUDA build); the Triton path
is covered by the GPU smoke in the design doc §9.
"""

import torch

from vllm.model_executor.layers.steering import steering_monitor


def _monitor_gate(
    hidden: torch.Tensor,
    probe: torch.Tensor,
    threshold: float,
    sharpness: float,
) -> torch.Tensor:
    """Reference per-token gate: ``sigmoid(sharpness*(h@probe - thr))``."""
    score = hidden.to(torch.float32) @ probe.to(torch.float32)
    return torch.sigmoid(sharpness * (score - threshold))


def _params(threshold: float, sharpness: float, gate_rows: float = 0.0) -> torch.Tensor:
    return torch.tensor([threshold, sharpness, gate_rows], dtype=torch.float32)


def _run(hidden, probe, params, active, token_scales, *, decode_mask=None,
         row_gate=None):
    """Call the op with default (no-op) row-gating buffers unless given."""
    dm = torch.zeros(token_scales.shape[0]) if decode_mask is None else decode_mask
    rg = torch.ones(token_scales.shape[0]) if row_gate is None else row_gate
    steering_monitor(hidden, probe, params, active, token_scales, dm, rg)
    return rg


class TestMonitorGateMath:
    """The fixed elementwise policy ``g`` (sigmoid threshold)."""

    def test_gate_matches_sigmoid_reference(self):
        torch.manual_seed(0)
        hidden = torch.randn(5, 8)
        probe = torch.randn(8)
        gate = _monitor_gate(hidden, probe, threshold=0.5, sharpness=2.0)
        score = hidden.to(torch.float32) @ probe.to(torch.float32)
        torch.testing.assert_close(gate, torch.sigmoid(2.0 * (score - 0.5)))

    def test_gate_is_bounded_unit_interval(self):
        torch.manual_seed(1)
        hidden = torch.randn(16, 4) * 100.0  # extreme scores both signs
        probe = torch.randn(4)
        gate = _monitor_gate(hidden, probe, threshold=0.0, sharpness=5.0)
        assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)

    def test_high_score_engages_low_score_disengages(self):
        # probe = ones; token a has large positive projection, token b large
        # negative. Sharp threshold at 0 ⇒ a~1, b~0.
        hidden = torch.stack([torch.full((4,), 10.0), torch.full((4,), -10.0)])
        probe = torch.ones(4)
        gate = _monitor_gate(hidden, probe, threshold=0.0, sharpness=10.0)
        assert gate[0] > 0.99  # engaged
        assert gate[1] < 0.01  # disengaged


class TestMonitorOpCpuEager:
    """The registered op's CPU eager path (in-place multiply on [:n])."""

    def test_active_multiplies_gate_into_scales(self):
        torch.manual_seed(2)
        hidden = torch.randn(3, 8)
        probe = torch.randn(8)
        params = _params(0.0, 1.5)
        token_scales = torch.full((6,), 4.0)  # runner-written gain on [:3]
        expected = token_scales.clone()
        expected[:3] = expected[:3] * _monitor_gate(hidden, probe, 0.0, 1.5)

        _run(
            hidden, probe, params, torch.tensor([True]), token_scales
        )
        torch.testing.assert_close(token_scales, expected)

    def test_inactive_is_noop(self):
        hidden = torch.randn(3, 8)
        probe = torch.randn(8)
        token_scales = torch.full((6,), 4.0)
        before = token_scales.clone()
        _run(
            hidden, probe, _params(0.0, 1.0), torch.tensor([False]), token_scales
        )
        torch.testing.assert_close(token_scales, before)

    def test_prefill_zero_is_preserved(self):
        # token_scales=0 marks prefill / non-decode tokens (cache safety).
        # Any gate keeps them 0 (0 * gate == 0), regardless of the probe.
        hidden = torch.randn(4, 8) * 50.0
        probe = torch.randn(8)
        token_scales = torch.tensor([0.0, 6.0, 0.0, 6.0])
        _run(
            hidden, probe, _params(0.0, 3.0), torch.tensor([True]), token_scales
        )
        assert token_scales[0].item() == 0.0
        assert token_scales[2].item() == 0.0

    def test_only_leading_n_touched(self):
        # The gate is written only on the first n = hidden.shape[0] entries;
        # the tail (old positions) is left for the runner's zero-out.
        hidden = torch.randn(2, 8)
        probe = torch.randn(8)
        token_scales = torch.tensor([5.0, 5.0, 9.0, 9.0])
        _run(
            hidden, probe, _params(0.0, 1.0), torch.tensor([True]), token_scales
        )
        torch.testing.assert_close(token_scales[2:], torch.tensor([9.0, 9.0]))

    def test_composes_into_full_tier(self):
        # End-to-end shape of the composed decode tier:
        #   tier[i] = dvec * (runner_gain[i] * monitor_gate[i])
        # The monitor produces the gate factor; this checks the product the
        # steering kernel then applies as ``dvec * token_scales``.
        torch.manual_seed(3)
        hidden = torch.randn(3, 8)
        probe = torch.randn(8)
        dvec = torch.full((8,), 2.0)
        gain = 6.0
        token_scales = torch.full((3,), gain)
        gate = _monitor_gate(hidden, probe, 0.1, 2.0)

        _run(
            hidden, probe, _params(0.1, 2.0), torch.tensor([True]), token_scales
        )
        tier = dvec.unsqueeze(0) * token_scales.unsqueeze(-1)
        expected = dvec.unsqueeze(0) * (gain * gate).unsqueeze(-1)
        torch.testing.assert_close(tier, expected)


class TestMonitorRowGating:
    """gate_rows: the monitor also gates the per-request row term, decode-only.

    Row gate update: ``row_gate[t] *= mask[t]*gate[t] + (1-mask[t])`` —
    decode (mask=1) → ``*gate``, prefill (mask=0) → unchanged.
    """

    def test_row_gate_untouched_when_gate_rows_off(self):
        hidden = torch.randn(3, 8)
        probe = torch.randn(8)
        token_scales = torch.zeros(3)
        rgate = torch.ones(3)
        # params gate_rows=0 ⇒ row gate must stay 1.0.
        _run(
            hidden, probe, _params(0.0, 1.0, gate_rows=0.0),
            torch.tensor([True]), token_scales,
            decode_mask=torch.ones(3), row_gate=rgate,
        )
        torch.testing.assert_close(rgate, torch.ones(3))

    def test_row_gate_gates_decode_preserves_prefill(self):
        probe = torch.ones(8)
        # token 0 decode + high score (gate~1), token 1 prefill (mask 0),
        # token 2 decode + very negative score (gate~0).
        hidden = torch.stack([
            torch.full((8,), 10.0),
            torch.full((8,), 10.0),   # prefill — must stay 1.0 regardless
            torch.full((8,), -10.0),
        ])
        token_scales = torch.zeros(3)
        rgate = torch.ones(3)
        decode_mask = torch.tensor([1.0, 0.0, 1.0])  # token 1 is prefill
        _run(
            hidden, probe, _params(0.0, 10.0, gate_rows=1.0),
            torch.tensor([True]), token_scales,
            decode_mask=decode_mask, row_gate=rgate,
        )
        assert rgate[0] > 0.99            # decode, engaged ⇒ ~1
        assert rgate[1].item() == 1.0     # prefill ⇒ exactly 1 (never gated)
        assert rgate[2] < 0.01            # decode, disengaged ⇒ ~0

    def test_row_gate_composes_with_prior_value(self):
        # row_gate starts <1 (e.g. a runner-set per-request gate); the
        # monitor multiplies, so decode positions compound.
        probe = torch.ones(8)
        rgate = torch.full((1,), 0.5)
        _run(
            torch.full((1, 8), -10.0),  # gate ~0
            probe, _params(0.0, 10.0, gate_rows=1.0),
            torch.tensor([True]), torch.zeros(1),
            decode_mask=torch.ones(1), row_gate=rgate,
        )
        assert rgate[0] < 0.01  # 0.5 * ~0 ⇒ ~0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
