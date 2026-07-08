# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the patch-sweep endpoint's helpers and auto-capture flow."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse

import vllm.v1.capture.patch_admission as patch_admission
from vllm.entrypoints.serve.patch.api_router import (
    answer_logprob,
    argmax_cell,
    cell_metric,
    drop_patch_source,
    patch_sweep,
    resolve_layers,
    resolve_span_body_positions,
)
from vllm.entrypoints.serve.patch.protocol import (
    LayerRange,
    PatchSweepRequest,
    SpanPosition,
)
from vllm.entrypoints.serve.patch.spans import (
    dedup_positions,
    prompt_char_offsets,
)


def _lp(d: dict[int, tuple[float, str]]) -> dict[int, SimpleNamespace]:
    return {
        tid: SimpleNamespace(logprob=val, decoded_token=tok)
        for tid, (val, tok) in d.items()
    }


def _req(**kw) -> PatchSweepRequest:
    base = dict(prompt="x", source_run="R1", layers=[0], metric="logprob")
    base.update(kw)
    return PatchSweepRequest(**base)


class TestResolve:
    def test_layer_range(self):
        assert resolve_layers(LayerRange(start=0, stop=6, step=2)) == [0, 2, 4]

    def test_layer_list(self):
        assert resolve_layers([1, 3, 5]) == [1, 3, 5]


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

_CLEAN_LP = -0.5  # answer logprob for the internal clean (capture) generation
_CORRUPT_LP = -2.0  # answer logprob for the unpatched baseline
_CELL_LP = -1.0  # answer logprob for a patched cell


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
        self._tok: Any = _Tok()
        self.sampling_params = []
        self.waited = []
        self.dropped = []

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
        if method == "drop_patch_source_run":
            run_id = args[0]
            self.dropped.append(run_id)
            before = len(self.runs)
            self.runs[:] = [r for r in self.runs if r["run_id"] != run_id]
            return [len(self.runs) < before]
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


def _gen_buckets(eng) -> tuple[int, int, int]:
    """(capture, patched, unpatched-baseline) generation counts."""
    capture = sum(1 for sp in eng.sampling_params if sp.capture is not None)
    patched = sum(1 for sp in eng.sampling_params if sp.patch)
    baseline = sum(
        1 for sp in eng.sampling_params if sp.capture is None and not sp.patch
    )
    return capture, patched, baseline


class TestMultiHook:
    def test_empty_hooks_400(self):
        eng = _MockEngine(runs=_existing_run())
        resp = _run(eng, hooks=[])
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        assert "non-empty" in json.loads(resp.body)["error"]

    def test_bad_hook_in_hooks_400(self):
        eng = _MockEngine(runs=_existing_run())
        resp = _run(eng, hooks=["mlp_out"])
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        assert "injectable" in json.loads(resp.body)["error"]

    def test_single_hook_list_parity(self):
        eng1 = _MockEngine(runs=_existing_run())
        r1 = _run(eng1, hook="post_block")
        eng2 = _MockEngine(runs=_existing_run())
        r2 = _run(eng2, hooks=["post_block"])
        # Same grid values; hook="post_block" has no hook_grids, the list form
        # carries exactly one.
        assert r2.grid == r1.grid
        assert r2.hook == "post_block"
        assert r1.hook_grids is None
        assert [hg.hook for hg in r2.hook_grids] == ["post_block"]
        assert r2.hook_grids[0].grid == r1.grid

    def test_multi_hook_response_shape(self):
        eng = _MockEngine(runs=[])
        resp = _run(eng, clean_prompt="ab", hooks=["pre_attn", "post_block"])
        assert not isinstance(resp, JSONResponse)
        assert [hg.hook for hg in resp.hook_grids] == ["pre_attn", "post_block"]
        # Top-level grid/hook/argmax mirror the first hook.
        assert resp.hook == "pre_attn"
        assert resp.grid == resp.hook_grids[0].grid
        for hg in resp.hook_grids:
            assert len(hg.grid) == 2 and len(hg.grid[0]) == 2

    def test_shared_baseline_and_argmax_computed_once(self):
        eng = _MockEngine(runs=[])
        _run(eng, clean_prompt="ab", hooks=["pre_attn", "post_attn", "post_block"])
        capture, patched, baseline = _gen_buckets(eng)
        # Baseline + noise-floor rerun are computed ONCE, not per hook.
        assert baseline == 2
        assert capture == 1
        # Cells fan out across every (hook, layer, position).
        assert patched == 3 * 2 * 2


class TestHookCompleteAutoCapture:
    def test_autocapture_taps_all_three_hooks(self):
        eng = _MockEngine(runs=[])
        _run(eng, clean_prompt="ab")  # single hook sweep still taps all hooks
        cap = next(sp for sp in eng.sampling_params if sp.capture is not None)
        hooks = cap.capture["patch_source"]["hooks"]
        assert set(hooks) == {"pre_attn", "post_attn", "post_block"}
        # Every hook covers the swept layer set.
        for layers in hooks.values():
            assert list(layers) == [0, 1]


class TestAutoDrop:
    def test_auto_drop_fires_after_autocapture(self):
        eng = _MockEngine(runs=[])
        resp = _run(eng, clean_prompt="ab")
        assert resp.auto_captured is True
        assert "R1" in eng.dropped
        assert not any(r["run_id"] == "R1" for r in eng.runs)

    def test_keep_source_retains_run(self):
        eng = _MockEngine(runs=[])
        resp = _run(eng, clean_prompt="ab", keep_source=True)
        assert resp.auto_captured is True
        assert resp.captured_source_run == "R1"
        assert eng.dropped == []
        assert any(r["run_id"] == "R1" for r in eng.runs)

    def test_preexisting_run_never_dropped(self):
        eng = _MockEngine(runs=_existing_run())
        resp = _run(eng, clean_prompt="ab")  # reuses existing R1, no capture
        assert resp.auto_captured is False
        assert eng.dropped == []
        assert any(r["run_id"] == "R1" for r in eng.runs)

    def test_drop_then_resweep_reauto_captures(self):
        # Cache coherence on the auto-drop path: a second identical sweep must
        # auto-capture again (a stale positive cache would skip it and 400).
        patch_admission._PATCH_SOURCE_CACHE = patch_admission._PatchSourceCache()
        eng = _MockEngine(runs=[])

        def _sweep():
            body = PatchSweepRequest(
                prompt="ab",
                source_run="R1",
                layers=[0, 1],
                hook="post_block",
                answer_token_id=5,
                clean_prompt="ab",
            )
            return asyncio.run(patch_sweep(body, _raw_request(eng)))

        r1 = _sweep()
        assert r1.auto_captured is True
        assert not any(r["run_id"] == "R1" for r in eng.runs)  # auto-dropped
        r2 = _sweep()
        assert r2.auto_captured is True


class TestDropRoute:
    def test_delete_drops_existing_run(self):
        eng = _MockEngine(runs=_existing_run())
        resp = asyncio.run(drop_patch_source("R1", _raw_request(eng)))
        assert resp.status_code == 200
        assert json.loads(resp.body)["dropped"] is True
        assert not any(r["run_id"] == "R1" for r in eng.runs)

    def test_delete_missing_run_404(self):
        eng = _MockEngine(runs=[])
        resp = asyncio.run(drop_patch_source("ghost", _raw_request(eng)))
        assert resp.status_code == 404
        assert "not found" in json.loads(resp.body)["error"]

    def test_delete_invalidates_manifest_cache(self):
        patch_admission._PATCH_SOURCE_CACHE = patch_admission._PatchSourceCache()
        eng = _MockEngine(runs=_existing_run())
        exists = patch_admission.patch_source_run_exists
        assert asyncio.run(exists(eng, "R1")) is True  # populates cache
        resp = asyncio.run(drop_patch_source("R1", _raw_request(eng)))
        assert resp.status_code == 200
        # Cache no longer reports the dropped run present.
        assert asyncio.run(exists(eng, "R1")) is False


# ---- streaming (SSE) sweep ------------------------------------------------


def _drain_sse(resp: StreamingResponse) -> list[str]:
    """Collect a StreamingResponse body into its raw ``data:`` payload lines."""

    async def drain() -> str:
        chunks = []
        async for chunk in resp.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
        return "".join(chunks)

    text = asyncio.run(drain())
    return [
        block[len("data: ") :]
        for block in text.split("\n\n")
        if block.startswith("data: ")
    ]


def _sse_events(resp: StreamingResponse) -> tuple[list[dict], str]:
    """Parse an SSE sweep stream into (json events, terminator)."""
    payloads = _drain_sse(resp)
    assert payloads[-1] == "[DONE]"
    events = [json.loads(p) for p in payloads[:-1]]
    return events, payloads[-1]


class _VoidEngine(_MockEngine):
    """Engine that reports the first patched cell as a resolution failure."""

    def __init__(self, runs):
        super().__init__(runs=runs)
        self.patched_ids: list[str] = []

    async def generate(self, prompt, sp, request_id):
        if sp.patch:
            self.patched_ids.append(request_id)
        async for out in super().generate(prompt, sp, request_id):
            yield out

    async def collective_rpc(self, method, args=None):
        if method == "pop_patch_resolution_failures" and self.patched_ids:
            return [{self.patched_ids[0]: ["source evicted mid-sweep"]}]
        return await super().collective_rpc(method, args)


class TestStreamingSweep:
    def test_stream_returns_sse_response(self):
        eng = _MockEngine(runs=_existing_run())
        resp = _run(eng, stream=True)
        assert isinstance(resp, StreamingResponse)
        assert resp.media_type == "text/event-stream"

    def test_cell_events_equal_grid_size(self):
        eng = _MockEngine(runs=_existing_run())
        resp = _run(eng, stream=True)
        events, done = _sse_events(resp)
        assert done == "[DONE]"
        assert events[0]["type"] == "start"
        cells = [e for e in events if e["type"] == "cell"]
        # grid is layers[0,1] x positions[0,1] = 4 cells.
        assert len(cells) == 4
        for c in cells:
            assert c["hook"] == "post_block"
            assert c["value"] == _CELL_LP
            assert {"layer", "position"} <= c.keys()
        assert events[-1]["type"] == "summary"

    def test_summary_equals_nonstreaming(self):
        # Same mock config run both ways -> byte-identical response payload.
        eng_plain = _MockEngine(runs=_existing_run())
        plain = _run(eng_plain, metric="logprob")
        assert not isinstance(plain, JSONResponse)

        eng_stream = _MockEngine(runs=_existing_run())
        resp = _run(eng_stream, metric="logprob", stream=True)
        events, _ = _sse_events(resp)
        summary = next(e for e in events if e["type"] == "summary")
        summary = {k: v for k, v in summary.items() if k != "type"}
        assert summary == plain.model_dump()

    def test_summary_equals_nonstreaming_recovered(self):
        eng_plain = _MockEngine(runs=[])
        plain = _run(eng_plain, clean_prompt="ab", metric="recovered")
        eng_stream = _MockEngine(runs=[])
        resp = _run(eng_stream, clean_prompt="ab", metric="recovered", stream=True)
        events, _ = _sse_events(resp)
        summary = next(e for e in events if e["type"] == "summary")
        summary = {k: v for k, v in summary.items() if k != "type"}
        assert summary == plain.model_dump()

    def test_pre_fanout_error_is_plain_json_400(self):
        # A missing run with no clean_prompt 400s before the stream starts.
        eng = _MockEngine(runs=[])
        resp = _run(eng, stream=True)
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        assert "patch source not found" in json.loads(resp.body)["error"]

    def test_bad_hook_is_plain_json_400(self):
        eng = _MockEngine(runs=_existing_run())
        resp = _run(eng, hook="not_a_hook", stream=True)
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400

    def test_voided_cell_emits_null_error_event(self):
        eng = _VoidEngine(runs=_existing_run())
        resp = _run(eng, stream=True)
        events, _ = _sse_events(resp)
        voided = [e for e in events if e["type"] == "cell" and e["value"] is None]
        assert len(voided) == 1
        assert voided[0]["error"] == "source evicted mid-sweep"
        assert voided[0]["hook"] == "post_block"
        assert {"layer", "position"} <= voided[0].keys()
        # The summary reflects the void: a skipped entry + a null grid cell.
        summary = next(e for e in events if e["type"] == "summary")
        assert summary["skipped"]
        assert any(None in row for row in summary["grid"])


# ---- streaming x multi-hook x lifecycle (combined) ------------------------


class TestStreamingMultiHook:
    def test_cell_events_cover_every_hook_layer_position(self):
        eng = _MockEngine(runs=[])
        resp = _run(
            eng,
            clean_prompt="ab",
            hooks=["pre_attn", "post_block"],
            stream=True,
        )
        assert isinstance(resp, StreamingResponse)
        events, done = _sse_events(resp)
        assert done == "[DONE]"
        # start event carries the requested hooks list (additive, multi-hook).
        assert events[0]["type"] == "start"
        assert events[0]["hooks"] == ["pre_attn", "post_block"]
        cells = [e for e in events if e["type"] == "cell"]
        # 2 hooks x layers[0,1] x positions[0,1] = 8 cells, correctly labelled.
        seen = {(c["hook"], c["layer"], c["position"]) for c in cells}
        assert seen == {
            (hook, layer, pos)
            for hook in ("pre_attn", "post_block")
            for layer in (0, 1)
            for pos in (0, 1)
        }
        for c in cells:
            assert c["hook"] in ("pre_attn", "post_block")

    def test_streamed_summary_hook_grids_equal_nonstreaming(self):
        # Same mock config both ways -> identical multi-hook summary.
        eng_plain = _MockEngine(runs=[])
        plain = _run(eng_plain, clean_prompt="ab", hooks=["pre_attn", "post_block"])
        assert not isinstance(plain, JSONResponse)

        eng_stream = _MockEngine(runs=[])
        resp = _run(
            eng_stream, clean_prompt="ab", hooks=["pre_attn", "post_block"], stream=True
        )
        events, _ = _sse_events(resp)
        summary = next(e for e in events if e["type"] == "summary")
        summary = {k: v for k, v in summary.items() if k != "type"}
        assert summary == plain.model_dump()
        assert [hg["hook"] for hg in summary["hook_grids"]] == [
            "pre_attn",
            "post_block",
        ]


class TestAutoDropBothPaths:
    def test_auto_drop_fires_nonstreaming(self):
        eng = _MockEngine(runs=[])
        resp = _run(eng, clean_prompt="ab")
        assert not isinstance(resp, JSONResponse)
        assert resp.auto_captured is True
        assert "R1" in eng.dropped  # drop RPC hit the engine

    def test_auto_drop_fires_streaming(self):
        eng = _MockEngine(runs=[])
        resp = _run(eng, clean_prompt="ab", stream=True)
        assert isinstance(resp, StreamingResponse)
        # The drop lives inside the generator (before the summary yields), so it
        # only fires once the stream is actually consumed.
        events, _ = _sse_events(resp)
        assert any(e["type"] == "summary" for e in events)
        assert "R1" in eng.dropped

    def test_keep_source_streaming_retains_run(self):
        eng = _MockEngine(runs=[])
        resp = _run(eng, clean_prompt="ab", keep_source=True, stream=True)
        _sse_events(resp)
        assert eng.dropped == []


# ---- shared span-resolution math ------------------------------------------


class TestDedupPositions:
    def test_order_preserving_dedup(self):
        # A span -> [5,6,7] mixed with an explicit 6 drops the duplicate.
        assert dedup_positions([[5, 6, 7], [6]]) == [5, 6, 7]

    def test_int_first_then_span(self):
        assert dedup_positions([[6], [5, 6, 7]]) == [6, 5, 7]

    def test_empty(self):
        assert dedup_positions([]) == []


class _FastTok:
    """Fake fast tokenizer: BOS + one token per character.

    Supports the offset-mapping fast path plus encode/decode, all consistent
    (char per token, offsets index the original prompt, BOS -> empty span).
    """

    is_fast = True
    BOS = 2  # distinct from any printable ord(c)

    def encode(self, text, add_special_tokens=True):
        ids = [self.BOS] if add_special_tokens else []
        return ids + [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i != self.BOS)

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        ids = self.encode(text, add_special_tokens)
        offsets, pos = [], 0
        for i in ids:
            if i == self.BOS:
                offsets.append((0, 0))
            else:
                offsets.append((pos, pos + 1))
                pos += 1
        return {"input_ids": ids, "offset_mapping": offsets}


class _SlowTok:
    """Tokenizer with only encode/decode (no offset mapping): fallback path."""

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class TestPromptCharOffsets:
    def test_fast_path_offsets_and_bos(self):
        text, offsets = prompt_char_offsets(_FastTok(), "hi")
        assert text == "hi"
        # BOS at (0,0); then 'h' [0,1); 'i' [1,2).
        assert offsets == [(0, 0), (0, 1), (1, 2)]

    def test_fallback_incremental_decode(self):
        text, offsets = prompt_char_offsets(_SlowTok(), "hi")
        assert text == "hi"
        assert offsets == [(0, 1), (1, 2)]


class TestResolveSpanBodyPositions:
    def test_no_spans_passes_ints_through(self):
        # No tokenization when there are no spans (tokenizer would raise here).
        assert resolve_span_body_positions(None, "hi", [0, 2]) == [0, 2]

    def test_single_token_span(self):
        got = resolve_span_body_positions(
            _FastTok(), "The cat", [SpanPosition(span="h")]
        )
        assert got == [2]  # BOS=0, T=1, h=2

    def test_multi_token_span(self):
        got = resolve_span_body_positions(
            _FastTok(), "The cat", [SpanPosition(span="cat")]
        )
        assert got == [5, 6, 7]

    def test_mixed_span_and_int_dedup_order(self):
        got = resolve_span_body_positions(
            _FastTok(), "The cat", [SpanPosition(span="cat"), 6]
        )
        assert got == [5, 6, 7]

    def test_special_token_never_selected(self):
        got = resolve_span_body_positions(
            _FastTok(), "The cat", [SpanPosition(span="The")]
        )
        assert got == [1, 2, 3]
        assert 0 not in got  # BOS excluded


# ---- endpoint span resolution (mock engine + fast tokenizer) --------------


def _span_engine(prompt, layers, hook="post_block"):
    """Engine whose fast tokenizer resolves spans, with a matching run R1."""
    tok = _FastTok()
    n = len(tok.encode(prompt))
    runs = [
        {
            "run_id": "R1",
            "hook_layers": [[hook, layer] for layer in layers],
            "positions": list(range(n)),
            "num_prompt_tokens": n,
        }
    ]
    eng = _MockEngine(runs=runs)
    eng._tok = tok
    return eng


def _run_span(eng, prompt, positions, **kw):
    patch_admission._PATCH_SOURCE_CACHE = patch_admission._PatchSourceCache()
    base = dict(
        prompt=prompt,
        source_run="R1",
        layers=[0, 1],
        hook="post_block",
        answer_token_id=5,
        positions=positions,
    )
    base.update(kw)
    body = PatchSweepRequest(**base)
    return asyncio.run(patch_sweep(body, _raw_request(eng)))


# ---- vector-sourced sweeps (no capture run) -------------------------------


class _FakeRegistry:
    def __init__(self, names):
        self._names = dict.fromkeys(names, object())

    def get(self, name):
        return self._names.get(name)

    def list_modules(self):
        return list(self._names)


def _raw_request_reg(eng, registry=None):
    state = SimpleNamespace(engine_client=eng, steering_module_registry=registry)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _vec_engine(num_layers=12, hidden_size=8):
    eng = _MockEngine(runs=[], num_layers=num_layers)
    eng.vllm_config.model_config.get_hidden_size = lambda: hidden_size
    return eng


def _pack_rows(rows, width):
    import numpy as np
    import pybase64 as base64

    arr = np.zeros((rows, width), dtype=np.float32)
    return {
        "dtype": "float32",
        "shape": [rows, width],
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _run_vec(eng, registry=None, **kw):
    patch_admission._PATCH_SOURCE_CACHE = patch_admission._PatchSourceCache()
    base = dict(prompt="ab", layers=[0, 1], hook="post_block", answer_token_id=5)
    base.update(kw)
    body = PatchSweepRequest(**base)
    return asyncio.run(patch_sweep(body, _raw_request_reg(eng, registry)))


class TestVectorSourcedSweep:
    def test_zeros_happy_path_no_capture(self):
        eng = _vec_engine()
        resp = _run_vec(eng, source_module="zeros")
        assert not isinstance(resp, JSONResponse)
        assert resp.auto_captured is False
        assert resp.captured_source_run is None
        assert not any(sp.capture is not None for sp in eng.sampling_params)
        assert eng.dropped == []
        # grid populated (2 layers x 2 positions), each patched cell.
        assert len(resp.grid) == 2 and len(resp.grid[0]) == 2
        assert resp.grid[0][0] == _CELL_LP
        # every cell request carried a source_module=zeros patch (no source_run).
        patched = [sp for sp in eng.sampling_params if sp.patch]
        assert patched
        for sp in patched:
            e = sp.patch[0]
            assert e["source_module"] == "zeros"
            assert "source_run" not in e

    def test_recovered_metric_400(self):
        eng = _vec_engine()
        resp = _run_vec(eng, source_module="zeros", metric="recovered")
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        assert "recovered" in json.loads(resp.body)["error"]

    def test_unknown_source_module_400(self):
        eng = _vec_engine()
        resp = _run_vec(eng, registry=_FakeRegistry(["good"]), source_module="bad")
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        assert "unknown source_module" in json.loads(resp.body)["error"]

    def test_known_source_module_ok(self):
        eng = _vec_engine()
        resp = _run_vec(eng, registry=_FakeRegistry(["good"]), source_module="good")
        assert not isinstance(resp, JSONResponse)
        assert resp.grid[0][0] == _CELL_LP

    def test_source_inline_forwards_patch_vectors(self):
        eng = _vec_engine(hidden_size=8)
        pv = _pack_rows(2, 8)
        resp = _run_vec(eng, source_inline=1, patch_vectors=pv)
        assert not isinstance(resp, JSONResponse)
        patched = [sp for sp in eng.sampling_params if sp.patch]
        assert patched
        for sp in patched:
            assert sp.patch[0]["source_inline"] == 1
            assert sp.patch_vectors == pv

    def test_source_inline_width_mismatch_400(self):
        eng = _vec_engine(hidden_size=8)
        resp = _run_vec(eng, source_inline=0, patch_vectors=_pack_rows(1, 4))
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        assert "width" in json.loads(resp.body)["error"]

    def test_source_inline_without_table_400(self):
        eng = _vec_engine()
        resp = _run_vec(eng, source_inline=0)
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400

    def test_streaming_parity_with_nonstreaming(self):
        eng_plain = _vec_engine()
        plain = _run_vec(eng_plain, source_module="zeros")
        assert not isinstance(plain, JSONResponse)
        eng_stream = _vec_engine()
        resp = _run_vec(eng_stream, source_module="zeros", stream=True)
        assert isinstance(resp, StreamingResponse)
        events, _ = _sse_events(resp)
        summary = next(e for e in events if e["type"] == "summary")
        summary = {k: v for k, v in summary.items() if k != "type"}
        assert summary == plain.model_dump()

    def test_source_run_and_vector_conflict_400(self):
        eng = _vec_engine()
        resp = _run_vec(eng, source_module="zeros", source_run="R1")
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400

    def test_mask_zeros_ablation(self):
        eng = _vec_engine()
        resp = _run_vec(eng, source_module="zeros", mask={"indices": [0, 2]})
        assert not isinstance(resp, JSONResponse)
        patched = [sp for sp in eng.sampling_params if sp.patch]
        for sp in patched:
            assert sp.patch[0]["mask"] == {"indices": [0, 2]}


class TestEndpointSpans:
    def test_all_prompt_expands_to_full_axis(self):
        eng = _span_engine("The cat", [0, 1])
        resp = _run_span(eng, "The cat", "all_prompt")
        assert not isinstance(resp, JSONResponse)
        assert resp.positions == list(range(8))  # BOS + 7 chars

    def test_resolved_axis_is_span_positions(self):
        eng = _span_engine("The cat", [0, 1])
        resp = _run_span(eng, "The cat", [{"span": "cat"}])
        assert not isinstance(resp, JSONResponse)
        assert resp.positions == [5, 6, 7]

    def test_mixed_span_and_int(self):
        eng = _span_engine("The cat", [0, 1])
        resp = _run_span(eng, "The cat", [{"span": "cat"}, 6])
        assert not isinstance(resp, JSONResponse)
        assert resp.positions == [5, 6, 7]  # explicit 6 deduped

    def test_bos_never_selected(self):
        eng = _span_engine("The cat", [0, 1])
        resp = _run_span(eng, "The cat", [{"span": "The"}])
        assert not isinstance(resp, JSONResponse)
        assert resp.positions == [1, 2, 3]
        assert 0 not in resp.positions

    def test_not_found_400(self):
        eng = _span_engine("The cat", [0, 1])
        resp = _run_span(eng, "The cat", [{"span": "dog"}])
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        assert "not found" in json.loads(resp.body)["error"]

    def test_empty_span_400(self):
        eng = _span_engine("The cat", [0, 1])
        resp = _run_span(eng, "The cat", [{"span": ""}])
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        assert "non-empty" in json.loads(resp.body)["error"]

    def test_occurrence_out_of_range_400(self):
        eng = _span_engine("cat cat", [0, 1])
        resp = _run_span(eng, "cat cat", [{"span": "cat", "occurrence": 2}])
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        assert "out of range" in json.loads(resp.body)["error"]
