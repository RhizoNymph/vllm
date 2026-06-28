# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the PatchStudy client's pure logic (no server needed)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "online_serving"
    / "openai_patch_client.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("openai_patch_client", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass decorators can resolve __module__.
    sys.modules["openai_patch_client"] = mod
    spec.loader.exec_module(mod)
    return mod


pc = _load_module()


def _choice(top: dict[str, float]):
    return SimpleNamespace(
        logprobs=SimpleNamespace(top_logprobs=[top], tokens=None, token_logprobs=None)
    )


class TestAnswerLogprob:
    def test_reads_top_logprob(self):
        ch = _choice({" Paris": -0.5, " Rome": -2.0})
        assert pc._answer_logprob(ch, " Paris") == -0.5

    def test_missing_token_returns_none(self):
        ch = _choice({" Rome": -2.0})
        assert pc._answer_logprob(ch, " Paris") is None

    def test_whitespace_tolerant_match(self):
        ch = _choice({"Paris": -0.7})
        assert pc._answer_logprob(ch, " Paris") == -0.7

    def test_falls_back_to_generated_token(self):
        ch = SimpleNamespace(
            logprobs=SimpleNamespace(
                top_logprobs=None, tokens=[" Paris"], token_logprobs=[-0.3]
            )
        )
        assert pc._answer_logprob(ch, " Paris") == -0.3


class TestMetric:
    def test_logprob(self):
        ch = _choice({" Paris": -0.5})
        assert pc.PatchStudy._metric(ch, " Paris", None, "logprob") == -0.5

    def test_logit_diff(self):
        ch = _choice({" Paris": -0.5, " Rome": -2.0})
        assert pc.PatchStudy._metric(ch, " Paris", " Rome", "logit_diff") == 1.5

    def test_logit_diff_requires_foil(self):
        ch = _choice({" Paris": -0.5})
        with pytest.raises(ValueError):
            pc.PatchStudy._metric(ch, " Paris", None, "logit_diff")

    def test_metric_none_when_missing(self):
        ch = _choice({" Rome": -2.0})
        assert pc.PatchStudy._metric(ch, " Paris", None, "logprob") is None


class TestSweepResult:
    def _result(self):
        return pc.SweepResult(
            layers=[0, 1, 2],
            positions=[0, 1],
            hook="post_block",
            metric_name="logprob",
            grid=[[-3.0, -2.0], [-1.0, -0.5], [-4.0, -2.5]],
        )

    def test_argmax_cell(self):
        assert self._result().argmax_cell() == (1, 1)  # layer 1, position 1

    def test_top(self):
        top = self._result().top(2)
        assert top[0] == (1, 1, -0.5)
        assert top[1] == (1, 0, -1.0)

    def test_to_numpy_shape(self):
        arr = self._result().to_numpy()
        assert arr.shape == (3, 2)

    def test_argmax_skips_none(self):
        r = pc.SweepResult(
            layers=[0, 1],
            positions=[0],
            hook="post_block",
            metric_name="logprob",
            grid=[[None], [-5.0]],
        )
        assert r.argmax_cell() == (1, 0)
