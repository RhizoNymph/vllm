# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the patch-sweep endpoint's helpers and auto-capture flow."""

import asyncio
import json
from types import SimpleNamespace

from fastapi.responses import JSONResponse

import vllm.v1.capture.patch_admission as patch_admission
from vllm.entrypoints.serve.patch.api_router import (
    answer_logprob,
    argmax_cell,
    cell_metric,
    patch_sweep,
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


# ---- endpoint auto-capture flow (mocked engine) ---------------------------

_CLEAN_LP = -0.5   # answer logprob for the internal clean (capture) generation
_CORRUPT_LP = -2.0  # answer logprob for the unpatched baseline
_CELL_LP = -1.0    # answer logprob for a patched cell


class _Tok:
    """Deterministic tokenizer: one id per character."""

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


class _RequestOutput:
    def __init__(self, logprobs_dict, prompt_token_ids, capture_results=None):
        self.outputs = [SimpleNamespace(logprobs=[logprobs_dict])]
        self.prompt_token_ids = prompt_token_ids
        self.capture_results = capture_results
        self.request_id = "rid"


class _MockEngine:
    """Engine stub for the patch-sweep endpoint.

    Serves manifests from a mutable ``runs`` list; a ``capture`` generation
    appends the captured run to it (simulating the worker source store) so a
    later existence refresh finds it. Records every ``SamplingParams`` seen.
    """

    def __init__(self, runs, num_layers=12):
        self.runs = runs
        self.vllm_config = SimpleNamespace(
            patch_config=SimpleNamespace(max_patch_slots=64),
            model_config=SimpleNamespace(
                get_total_num_hidden_layers=lambda: num_layers
            ),
        )
        self._tok = _Tok()
        self.sampling_params = []
        self.waited = []

    def get_tokenizer(self):
        return self._tok

    async def generate(self, prompt, sp, request_id):
        self.sampling_params.append(sp)
        pids = self._tok.encode(prompt["prompt"])
        if sp.capture is not None:
            spec = sp.capture["patch_source"]
            hook_layers = [
                [hook, layer]
                for hook, layers in spec["hooks"].items()
                for layer in layers
            ]
            self.runs.append(
                {
                    "run_id": spec["run"],
                    "hook_layers": hook_layers,
                    "positions": list(range(len(pids))),
                    "num_prompt_tokens": len(pids),
                }
            )
            lp = _CLEAN_LP
        elif sp.patch:
            lp = _CELL_LP
        else:
            lp = _CORRUPT_LP
        yield _RequestOutput(
            {5: SimpleNamespace(logprob=lp, decoded_token=" Paris")}, pids
        )

    async def collective_rpc(self, method, args=None):
        if method == "get_patch_source_manifests":
            return [list(self.runs)]
        if method == "pop_patch_resolution_failures":
            return [{}]
        return [None]

    async def wait_for_capture_results(self, request_id):
        self.waited.append(request_id)
        return {"patch_source": "ok"}


def _raw_request(eng):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=eng))
    )


def _run(eng, **kw):
    # Fresh source-run cache per call: it is a process global.
    patch_admission._PATCH_SOURCE_CACHE = patch_admission._PatchSourceCache()
    base = dict(
        prompt="ab",
        source_run="R1",
        layers=[0, 1],
        hook="post_block",
        answer_token_id=5,
    )
    base.update(kw)
    body = PatchSweepRequest(**base)
    return asyncio.run(patch_sweep(body, _raw_request(eng)))


def _existing_run():
    return [
        {
            "run_id": "R1",
            "hook_layers": [["post_block", 0], ["post_block", 1]],
            "positions": [0, 1],
            "num_prompt_tokens": 2,
        }
    ]


class TestAutoCapture:
    def test_triggers_on_missing_run_with_clean_prompt(self):
        eng = _MockEngine(runs=[])
        resp = _run(eng, clean_prompt="ab")
        assert not isinstance(resp, JSONResponse)
        assert resp.auto_captured is True
        assert resp.captured_source_run == "R1"
        # A capture generation happened and was waited on for durability.
        assert any(sp.capture is not None for sp in eng.sampling_params)
        assert eng.waited  # capture_wait invoked
        assert resp.grid[0][0] == _CELL_LP  # grid actually ran

    def test_missing_run_without_clean_prompt_400s(self):
        eng = _MockEngine(runs=[])
        resp = _run(eng)  # no clean_prompt
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        assert "patch source not found" in json.loads(resp.body)["error"]
        assert not any(sp.capture is not None for sp in eng.sampling_params)

    def test_existing_run_skips_capture(self):
        eng = _MockEngine(runs=_existing_run())
        resp = _run(eng, clean_prompt="ab")
        assert not isinstance(resp, JSONResponse)
        assert resp.auto_captured is False
        assert resp.captured_source_run is None
        assert not any(sp.capture is not None for sp in eng.sampling_params)
        assert not eng.waited

    def test_clean_baseline_computed_from_autocapture(self):
        eng = _MockEngine(runs=[])
        resp = _run(eng, clean_prompt="ab", metric="recovered")
        assert not isinstance(resp, JSONResponse)
        assert resp.auto_captured is True
        # clean graded from the internal clean generation (caller gave none).
        assert resp.clean == _CLEAN_LP
        assert resp.corrupt == _CORRUPT_LP
        # recovered = (cell - corrupt) / (clean - corrupt)
        expected = (_CELL_LP - _CORRUPT_LP) / (_CLEAN_LP - _CORRUPT_LP)
        assert abs(resp.grid[0][0] - expected) < 1e-9
