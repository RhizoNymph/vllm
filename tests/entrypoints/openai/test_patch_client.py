# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the PatchStudy client's pure logic (no server needed)."""

import asyncio
import json
from types import SimpleNamespace

import pytest

import vllm.entrypoints.serve.patch.client as pc

pytestmark = pytest.mark.cpu_test


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


def _offsets(spans: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Build (text, per-token char offsets) from consecutive token strings."""
    text = ""
    offsets = []
    for s in spans:
        offsets.append((len(text), len(text) + len(s)))
        text += s
    return text, offsets


class TestResolveSpanPositions:
    # "The Colosseum is" tokenized as leading-space subwords.
    TOKENS = ["The", " Col", "os", "seum", " is"]

    def test_single_token_span(self):
        text, offsets = _offsets(self.TOKENS)
        # "The" covers only position 0.
        assert pc._resolve_span_positions(offsets, text, "The") == [0]

    def test_multi_token_span(self):
        text, offsets = _offsets(self.TOKENS)
        # "Colosseum" spans the leading-space token plus its continuations.
        assert pc._resolve_span_positions(offsets, text, "Colosseum") == [1, 2, 3]

    def test_span_with_leading_space_matches_same_tokens(self):
        text, offsets = _offsets(self.TOKENS)
        assert pc._resolve_span_positions(offsets, text, " Colosseum") == [1, 2, 3]

    def test_partial_token_overlap_included(self):
        text, offsets = _offsets(self.TOKENS)
        # "seu" lies inside token 3 only.
        assert pc._resolve_span_positions(offsets, text, "seu") == [3]

    def test_span_crossing_token_boundary(self):
        text, offsets = _offsets(self.TOKENS)
        # "osseum is" overlaps tokens 2, 3, 4.
        assert pc._resolve_span_positions(offsets, text, "osseum is") == [2, 3, 4]

    def test_not_found_raises(self):
        text, offsets = _offsets(self.TOKENS)
        with pytest.raises(ValueError, match="not found"):
            pc._resolve_span_positions(offsets, text, "Rome")

    def test_empty_span_raises(self):
        text, offsets = _offsets(self.TOKENS)
        with pytest.raises(ValueError, match="non-empty"):
            pc._resolve_span_positions(offsets, text, "")

    def test_multiple_occurrences_select_by_index(self):
        text, offsets = _offsets(["a", "b", "a", "b"])  # "abab"
        assert pc._resolve_span_positions(offsets, text, "a", 0) == [0]
        assert pc._resolve_span_positions(offsets, text, "a", 1) == [2]

    def test_occurrence_out_of_range_raises(self):
        text, offsets = _offsets(["a", "b", "a", "b"])
        with pytest.raises(ValueError, match="out of range"):
            pc._resolve_span_positions(offsets, text, "a", 2)

    def test_special_token_empty_span_never_selected(self):
        # A BOS-like token detokenizes to "" (empty span) at position 0.
        text, offsets = _offsets(["", "The", " cat"])
        assert pc._resolve_span_positions(offsets, text, "The") == [1]


class TestTokenCharOffsets:
    def test_incremental_detokenize_builds_offsets(self, monkeypatch):
        study = pc.PatchStudy.__new__(pc.PatchStudy)
        study.model = "m"
        # decode(ids[:k]) returns the concatenation of the first k pieces.
        pieces = ["", "The", " cat"]

        def fake_detok(ids):
            return "".join(pieces[: len(ids)])

        monkeypatch.setattr(study, "_detokenize", fake_detok)
        text, offsets = study._token_char_offsets([1, 2, 3])
        assert text == "The cat"
        # BOS -> empty span; then "The" [0,3); then " cat" [3,7).
        assert offsets == [(0, 0), (0, 3), (3, 7)]

    def test_detokenize_failure_raises(self, monkeypatch):
        study = pc.PatchStudy.__new__(pc.PatchStudy)
        study.model = "m"
        monkeypatch.setattr(study, "_detokenize", lambda ids: None)
        with pytest.raises(RuntimeError, match="detokenize"):
            study._token_char_offsets([1, 2])


class TestPositionsFor:
    def _study(self, monkeypatch, ids, pieces):
        study = pc.PatchStudy.__new__(pc.PatchStudy)
        study.model = "m"
        monkeypatch.setattr(study, "_tokenize", lambda text: ids)
        monkeypatch.setattr(
            study, "_detokenize", lambda i: "".join(pieces[: len(i)])
        )
        return study

    def test_positions_for(self, monkeypatch):
        study = self._study(
            monkeypatch, [1, 2, 3, 4], ["", "The", " Colos", "seum"]
        )
        got = asyncio.run(study.positions_for("The Colosseum", "Colosseum"))
        assert got == [2, 3]

    def test_positions_for_tokenize_failure(self, monkeypatch):
        study = self._study(monkeypatch, None, [])
        with pytest.raises(RuntimeError, match="tokenize"):
            asyncio.run(study.positions_for("x", "y"))


_SWEEP_RESP = {
    "layers": [0, 1],
    "positions": [5, 6, 7],
    "hook": "post_block",
    "metric": "logprob",
    "grid": [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
    "clean": -0.5,
    "corrupt": -2.0,
    "skipped": [],
    "alignment": None,
}


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FakeAsyncClient:
    """Records every /patch_sweep payload; echoes an auto-capture response."""

    payloads: list[dict] = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        _FakeAsyncClient.payloads.append(json)
        data = dict(
            _SWEEP_RESP,
            auto_captured=json["clean_prompt"] is not None,
            captured_source_run=json["source_run"],
        )
        return _FakeResp(data)


def _server_study():
    study = pc.PatchStudy.__new__(pc.PatchStudy)
    study.model = "m"
    study.base_url = "http://localhost:8000/v1"
    study.api_key = "unused"
    study.concurrency = 4
    study.hook = "post_block"
    study.logprobs = 20
    return study


class TestServerSideOneCall:
    def _patch_httpx(self, monkeypatch):
        import httpx

        _FakeAsyncClient.payloads = []
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    def test_one_call_generates_run_and_forwards_spans(self, monkeypatch):
        self._patch_httpx(monkeypatch)
        study = _server_study()
        res = asyncio.run(
            study.sweep_layers_positions(
                "The cat",
                layers=[0, 1],
                positions=[pc.Span("cat"), 4],
                answer_token=" Paris",
                clean_prompt="The clean cat",
                server_side=True,
            )
        )
        p = _FakeAsyncClient.payloads[-1]
        # clean_prompt forwarded; spans forwarded as objects (not resolved).
        assert p["clean_prompt"] == "The clean cat"
        assert p["positions"] == [{"span": "cat", "occurrence": 0}, 4]
        # A fresh uuid4-hex run was generated for auto-capture.
        assert len(p["source_run"]) == 32
        assert p["clean_baseline"] is None  # no clean handle => server grades it
        # auto_captured / captured_source_run surfaced on the result.
        assert res.auto_captured is True
        assert res.captured_source_run == p["source_run"]

    def test_run_is_fresh_per_call(self, monkeypatch):
        self._patch_httpx(monkeypatch)
        study = _server_study()
        for _ in range(2):
            asyncio.run(
                study.sweep_layers_positions(
                    "The cat",
                    layers=[0],
                    positions=[0],
                    answer_token=" P",
                    clean_prompt="clean",
                    server_side=True,
                )
            )
        runs = [p["source_run"] for p in _FakeAsyncClient.payloads]
        assert runs[0] != runs[1]  # fresh per call

    def test_existing_run_wins_over_autocapture(self, monkeypatch):
        self._patch_httpx(monkeypatch)
        study = _server_study()
        clean = pc.CleanRun(
            run_id="R1", num_prompt_tokens=3, hook="post_block", prompt="clean"
        )
        asyncio.run(
            study.sweep_layers_positions(
                "corrupt",
                layers=[0],
                positions=[0],
                answer_token=" P",
                clean=clean,
                clean_prompt="explicit clean",
                server_side=True,
            )
        )
        p = _FakeAsyncClient.payloads[-1]
        assert p["source_run"] == "R1"  # reuse, not a fresh uuid
        assert p["clean_prompt"] == "explicit clean"  # drives alignment

    def test_clean_prompt_per_cell_path_raises(self):
        study = _server_study()
        with pytest.raises(ValueError, match="server_side"):
            asyncio.run(
                study.sweep_layers_positions(
                    "corrupt",
                    layers=[0],
                    positions=[0],
                    answer_token=" P",
                    clean_prompt="x",  # per-cell path cannot auto-capture
                )
            )

    def test_server_side_without_run_or_clean_prompt_raises(self):
        study = _server_study()
        with pytest.raises(ValueError, match="run="):
            asyncio.run(
                study.sweep_layers_positions(
                    "corrupt",
                    layers=[0],
                    positions=[0],
                    answer_token=" P",
                    server_side=True,
                )
            )


def _summary_data(payload: dict) -> dict:
    """Server response payload for a given request (auto-capture echoed)."""
    return dict(
        _SWEEP_RESP,
        auto_captured=payload["clean_prompt"] is not None,
        captured_source_run=payload["source_run"],
    )


def _sse_line(event: dict) -> str:
    return "data: " + json.dumps(event)


class _FakeStreamResp:
    def __init__(self, lines: list[str]):
        self._lines = lines
        self.headers = {"content-type": "text/event-stream"}

    def raise_for_status(self):
        pass

    async def aread(self):
        return b""

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _StreamCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *a):
        return False


class _FakeStreamClient:
    """Serves /patch_sweep as both a plain POST and an SSE stream."""

    streamed: list[dict] = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        return _FakeResp(_summary_data(json))

    def stream(self, method, url, json=None, headers=None):
        payload = json
        _FakeStreamClient.streamed.append(payload)
        data = _summary_data(payload)
        lines = [_sse_line({"type": "start"})]
        for i, layer in enumerate(data["layers"]):
            for j, pos in enumerate(data["positions"]):
                lines.append(_sse_line({
                    "type": "cell", "hook": data["hook"], "layer": layer,
                    "position": pos, "value": data["grid"][i][j],
                }))
        lines.append(_sse_line({"type": "summary", **data}))
        lines.append("data: [DONE]")
        return _StreamCtx(_FakeStreamResp(lines))


class TestServerSideStreaming:
    def _patch(self, monkeypatch):
        import httpx

        _FakeStreamClient.streamed = []
        monkeypatch.setattr(httpx, "AsyncClient", _FakeStreamClient)

    def test_on_cell_invoked_per_cell_and_result_matches(self, monkeypatch):
        self._patch(monkeypatch)
        study = _server_study()
        seen: list[dict] = []
        streamed = asyncio.run(
            study.sweep_layers_positions(
                "The cat",
                run="R1",
                layers=[0, 1],
                positions=[5, 6, 7],
                answer_token=" Paris",
                server_side=True,
                on_cell=seen.append,
            )
        )
        # 2 layers x 3 positions = 6 cell events, each carrying hook.
        assert len(seen) == 6
        assert all(e["type"] == "cell" for e in seen)
        assert all(e["hook"] == "post_block" for e in seen)
        # stream=true was sent.
        assert _FakeStreamClient.streamed[-1]["stream"] is True

        plain = asyncio.run(
            study.sweep_layers_positions(
                "The cat",
                run="R1",
                layers=[0, 1],
                positions=[5, 6, 7],
                answer_token=" Paris",
                server_side=True,
            )
        )
        # Streamed result == non-streaming result (field by field).
        assert streamed == plain

    def test_on_cell_requires_server_side(self):
        study = _server_study()
        with pytest.raises(ValueError, match="server_side"):
            asyncio.run(
                study.sweep_layers_positions(
                    "corrupt",
                    run="R1",
                    layers=[0],
                    positions=[0],
                    answer_token=" P",
                    on_cell=lambda e: None,
                )
            )


class TestResolvePositions:
    def test_mixes_ints_and_spans_dedup_in_order(self, monkeypatch):
        study = pc.PatchStudy.__new__(pc.PatchStudy)

        async def fake_positions_for(prompt, span, *, occurrence=0):
            return {"cat": [2, 3], "dog": [3, 5]}[span]

        monkeypatch.setattr(study, "positions_for", fake_positions_for)
        out = asyncio.run(
            study._resolve_positions([0, pc.Span("cat"), pc.Span("dog")], "prompt")
        )
        # 0, then cat->2,3, then dog->3(dupe dropped),5.
        assert out == [0, 2, 3, 5]

    def test_single_span_marker(self, monkeypatch):
        study = pc.PatchStudy.__new__(pc.PatchStudy)

        async def fake_positions_for(prompt, span, *, occurrence=0):
            assert occurrence == 1
            return [4, 5]

        monkeypatch.setattr(study, "positions_for", fake_positions_for)
        out = asyncio.run(study._resolve_positions(pc.Span("cat", occurrence=1), "p"))
        assert out == [4, 5]
