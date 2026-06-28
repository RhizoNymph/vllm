# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the patch-sweep endpoint's pure helpers (no engine needed)."""

from types import SimpleNamespace

from vllm.entrypoints.serve.patch.api_router import (
    answer_logprob,
    argmax_cell,
    cell_metric,
    resolve_layers,
    resolve_positions,
)
from vllm.entrypoints.serve.patch.protocol import LayerRange, PatchSweepRequest


def _lp(d: dict[int, tuple[float, str]]) -> dict[int, SimpleNamespace]:
    return {
        tid: SimpleNamespace(logprob=val, decoded_token=tok)
        for tid, (val, tok) in d.items()
    }


def _req(**kw) -> PatchSweepRequest:
    base = dict(
        prompt="x", source_run="R1", layers=[0], metric="logprob"
    )
    base.update(kw)
    return PatchSweepRequest(**base)


class TestResolve:
    def test_layer_range(self):
        assert resolve_layers(LayerRange(start=0, stop=6, step=2)) == [0, 2, 4]

    def test_layer_list(self):
        assert resolve_layers([1, 3, 5]) == [1, 3, 5]

    def test_all_prompt(self):
        assert resolve_positions("all_prompt", 4) == [0, 1, 2, 3]

    def test_explicit_positions(self):
        assert resolve_positions([0, 2], 9) == [0, 2]


class TestAnswerLogprob:
    def test_by_id_exact(self):
        lp = _lp({5: (-0.5, " Paris"), 7: (-2.0, " Rome")})
        assert answer_logprob(lp, 5, None) == -0.5

    def test_by_id_missing(self):
        lp = _lp({7: (-2.0, " Rome")})
        assert answer_logprob(lp, 5, None) is None

    def test_by_string_whitespace_tolerant(self):
        lp = _lp({5: (-0.7, "Paris")})
        assert answer_logprob(lp, None, " Paris") == -0.7

    def test_by_string_missing(self):
        lp = _lp({7: (-2.0, " Rome")})
        assert answer_logprob(lp, None, " Paris") is None

    def test_id_preferred_over_string(self):
        lp = _lp({5: (-0.5, " Paris")})
        assert answer_logprob(lp, 5, " Rome") == -0.5


class TestCellMetric:
    def test_logprob(self):
        lp = _lp({5: (-0.5, " Paris")})
        assert cell_metric(lp, _req(answer_token_id=5)) == -0.5

    def test_logit_diff(self):
        lp = _lp({5: (-0.5, " Paris"), 7: (-2.0, " Rome")})
        req = _req(metric="logit_diff", answer_token_id=5, foil_token_id=7)
        assert cell_metric(lp, req) == 1.5

    def test_logit_diff_missing_foil(self):
        lp = _lp({5: (-0.5, " Paris")})
        req = _req(metric="logit_diff", answer_token_id=5, foil_token_id=7)
        assert cell_metric(lp, req) is None


class TestArgmax:
    def test_argmax_cell(self):
        grid = [[-3.0, -2.0], [-1.0, -0.5]]
        out = argmax_cell(grid, [0, 1], [0, 1])
        assert out == {"layer": 1, "position": 1, "value": -0.5}

    def test_argmax_skips_none(self):
        grid = [[None, -2.0], [None, None]]
        out = argmax_cell(grid, [0, 1], [0, 1])
        assert out == {"layer": 0, "position": 1, "value": -2.0}

    def test_argmax_all_none(self):
        assert argmax_cell([[None]], [0], [0]) is None
