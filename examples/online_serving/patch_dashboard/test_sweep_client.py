# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the dashboard's sweep client (no server needed)."""

import json
import threading

import pytest
import sweep_client as sc

# ---------------------------------------------------------------- helpers


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}".encode()


START = {
    "type": "start",
    "layers": [0, 4, 8],
    "positions": [1, 2],
    "hook": "post_block",
    "hooks": ["post_block", "mlp_out"],
    "metric": "recovered",
    "auto_captured": True,
    "captured_source_run": "dash-abc",
}


def _cell(hook="post_block", layer=4, position=2, value=0.5, **extra):
    return {
        "type": "cell",
        "hook": hook,
        "layer": layer,
        "position": position,
        "value": value,
        **extra,
    }


SUMMARY = {
    "type": "summary",
    "layers": [0, 4, 8],
    "positions": [1, 2],
    "hook": "post_block",
    "metric": "recovered",
    "grid": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    "clean": -0.1,
    "corrupt": -9.0,
    "noise_floor": 0.003,
    "skipped": [],
    "auto_captured": True,
    "captured_source_run": "dash-abc",
    "hook_grids": [
        {"hook": "post_block", "grid": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]},
        {"hook": "mlp_out", "grid": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.1]]},
    ],
}


class FakeResponse:
    def __init__(
        self, status_code=200, content_type="text/event-stream", lines=(), body=b"{}"
    ):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self._lines = list(lines)
        self._body = body
        self.closed = False

    def iter_lines(self):
        yield from self._lines

    def json(self):
        return json.loads(self._body)

    @property
    def text(self):
        return self._body.decode()

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------- visible_token


def test_visible_token_escapes_html():
    # Plotly parses labels as pseudo-HTML; raw special tokens must not
    # survive as tags.
    assert sc.visible_token("<s>") == "&lt;s&gt;"
    assert sc.visible_token("<|im_start|>") == "&lt;|im_start|&gt;"
    assert sc.visible_token("A&B") == "A&amp;B"


def test_visible_token_marks_whitespace():
    assert sc.visible_token(" Paris") == "␣Paris"
    assert sc.visible_token("a\nb") == "a⏎b"


# ---------------------------------------------------------------- parse_int_list


def test_parse_int_list():
    assert sc.parse_int_list("1, 2,5", what="x") == [1, 2, 5]
    assert sc.parse_int_list("", what="x") == []
    assert sc.parse_int_list(None, what="x") == []


def test_parse_int_list_rejects_junk():
    with pytest.raises(sc.SweepRequestError, match="mask dims"):
        sc.parse_int_list("1, two", what="mask dims")


# ---------------------------------------------------------------- build_payload


def _payload(**overrides):
    kwargs = dict(
        corrupt_prompt="The Colosseum is in the city of",
        clean_prompt="The Eiffel Tower is in the city of",
        source="clean",
        mask_indices=None,
        hooks=["post_block"],
        layer_start=0,
        layer_stop=8,
        layer_step=2,
        positions_mode="all_prompt",
        span_text=None,
        span_occurrence=0,
        position_indices=None,
        metric="recovered",
        answer_token=" Paris",
        foil_token=None,
        alpha=1.0,
        keep_source=False,
        model=None,
    )
    kwargs.update(overrides)
    return sc.build_payload(**kwargs)


def test_build_payload_capture_sourced():
    p = _payload()
    assert p["prompt"].startswith("The Colosseum")
    assert p["clean_prompt"].startswith("The Eiffel")
    assert p["source_run"].startswith("dash-")
    assert p["hooks"] == ["post_block"]
    assert p["layers"] == {"start": 0, "stop": 8, "step": 2}
    assert p["positions"] == "all_prompt"
    assert p["metric"] == "recovered"
    assert p["answer_token"] == " Paris"
    assert p["stream"] is True
    assert "source_module" not in p
    assert "model" not in p


def test_build_payload_span_positions():
    p = _payload(positions_mode="span", span_text="Colosseum", span_occurrence=1)
    assert p["positions"] == [{"span": "Colosseum", "occurrence": 1}]


def test_build_payload_index_positions():
    p = _payload(positions_mode="indices", position_indices=[4, 7])
    assert p["positions"] == [4, 7]


def test_build_payload_zeros_ablation():
    p = _payload(
        source="zeros", clean_prompt=None, metric="logprob", mask_indices=[12, 40]
    )
    assert p["source_module"] == "zeros"
    assert p["mask"] == {"indices": [12, 40]}
    assert "source_run" not in p
    assert "clean_prompt" not in p


def test_build_payload_keeps_source_and_model():
    p = _payload(keep_source=True, model="Qwen/Qwen3-0.6B", alpha=0.5)
    assert p["keep_source"] is True
    assert p["model"] == "Qwen/Qwen3-0.6B"
    assert p["alpha"] == 0.5


def test_build_payload_rejects_empty_prompt():
    with pytest.raises(sc.SweepRequestError, match="corrupt prompt"):
        _payload(corrupt_prompt="  ")


def test_build_payload_rejects_bad_hooks():
    with pytest.raises(sc.SweepRequestError, match="hook"):
        _payload(hooks=[])
    with pytest.raises(sc.SweepRequestError, match="hook"):
        _payload(hooks=["post_block", "resid_mid"])


def test_build_payload_rejects_bad_layers():
    with pytest.raises(sc.SweepRequestError, match="layer"):
        _payload(layer_start=8, layer_stop=8)
    with pytest.raises(sc.SweepRequestError, match="layer"):
        _payload(layer_step=0)
    with pytest.raises(sc.SweepRequestError, match="layer"):
        _payload(layer_start=None)


def test_build_payload_rejects_missing_answer_token():
    with pytest.raises(sc.SweepRequestError, match="answer token"):
        _payload(answer_token="")


def test_build_payload_recovered_needs_clean_capture():
    with pytest.raises(sc.SweepRequestError, match="recovered"):
        _payload(source="zeros", clean_prompt=None)
    with pytest.raises(sc.SweepRequestError, match="clean prompt"):
        _payload(clean_prompt=" ")


def test_build_payload_logit_diff_needs_foil():
    with pytest.raises(sc.SweepRequestError, match="foil"):
        _payload(metric="logit_diff", foil_token=None)
    p = _payload(metric="logit_diff", foil_token=" Rome")
    assert p["foil_token"] == " Rome"


def test_build_payload_rejects_span_without_text():
    with pytest.raises(sc.SweepRequestError, match="span"):
        _payload(positions_mode="span", span_text=" ")


def test_build_payload_rejects_indices_without_values():
    with pytest.raises(sc.SweepRequestError, match="position"):
        _payload(positions_mode="indices", position_indices=[])


# ---------------------------------------------------------------- iter_sse_events


def test_iter_sse_events_parses_data_lines():
    lines = [
        b"",
        b": keepalive",
        _sse(START),
        b"event: noise",
        _sse(_cell()),
        b"data: [DONE]",
        _sse({"type": "late"}),
    ]
    events = list(sc.iter_sse_events(lines))
    assert events[0]["type"] == "start"
    assert events[1]["type"] == "cell"
    assert events[2] == sc.DONE
    assert len(events) == 3  # nothing after [DONE]


def test_iter_sse_events_accepts_str_lines():
    events = list(sc.iter_sse_events(['data: {"type": "cell"}']))
    assert events == [{"type": "cell"}]


# ---------------------------------------------------------------- SweepState


def test_state_start_sizes_grids():
    st = sc.SweepState()
    st.apply_event(START)
    snap = st.snapshot()
    assert snap["status"] == "streaming"
    assert snap["hooks"] == ["post_block", "mlp_out"]
    assert snap["layers"] == [0, 4, 8]
    assert snap["positions"] == [1, 2]
    assert snap["total_cells"] == 3 * 2 * 2
    assert snap["grids"]["mlp_out"] == [[None, None]] * 3
    assert snap["auto_captured"] is True


def test_state_start_without_hooks_list_uses_hook():
    st = sc.SweepState()
    st.apply_event({k: v for k, v in START.items() if k != "hooks"})
    assert st.snapshot()["hooks"] == ["post_block"]


def test_state_cell_writes_grid_and_counts():
    st = sc.SweepState()
    st.apply_event(START)
    st.apply_event(_cell(hook="mlp_out", layer=8, position=1, value=0.25))
    snap = st.snapshot()
    assert snap["grids"]["mlp_out"][2][0] == 0.25
    assert snap["done_cells"] == 1


def test_state_voided_cell_records_error():
    st = sc.SweepState()
    st.apply_event(START)
    st.apply_event(_cell(value=0.5))
    st.apply_event(_cell(value=None, error="source evicted"))
    snap = st.snapshot()
    assert snap["grids"]["post_block"][1][1] is None
    assert snap["done_cells"] == 1  # a void re-emit is not new progress
    assert snap["cell_errors"] == [
        {"hook": "post_block", "layer": 4, "position": 2, "error": "source evicted"}
    ]


def test_state_ignores_unknown_cells():
    st = sc.SweepState()
    st.apply_event(START)
    st.apply_event(_cell(hook="pre_attn"))
    st.apply_event(_cell(layer=99))
    st.apply_event(_cell(position=99))
    assert st.snapshot()["done_cells"] == 0


def test_state_summary_finalizes():
    st = sc.SweepState()
    st.apply_event(START)
    st.apply_event(SUMMARY)
    snap = st.snapshot()
    assert snap["status"] == "done"
    assert snap["grids"]["mlp_out"][2][1] == 0.1
    assert snap["summary"]["noise_floor"] == 0.003


def test_state_snapshot_is_a_copy():
    st = sc.SweepState()
    st.apply_event(START)
    snap = st.snapshot()
    snap["grids"]["post_block"][0][0] = 123.0
    snap["layers"].append(99)
    fresh = st.snapshot()
    assert fresh["grids"]["post_block"][0][0] is None
    assert fresh["layers"] == [0, 4, 8]


def test_state_fail_and_cancel():
    st = sc.SweepState()
    st.fail("boom")
    assert st.snapshot()["status"] == "error"
    assert st.snapshot()["error"] == "boom"
    st2 = sc.SweepState()
    st2.cancel()
    assert st2.cancelled


# ---------------------------------------------------------------- run_sweep


def _no_labels(server_url, model, prompt, post):
    return {}


def test_run_sweep_happy_path():
    lines = [_sse(START), _sse(_cell()), _sse(SUMMARY), b"data: [DONE]"]
    resp = FakeResponse(lines=lines)
    st = sc.SweepState()
    sc.run_sweep(
        st,
        "http://x:8000",
        {"stream": True},
        post=lambda url, **kw: resp,
        label_fn=_no_labels,
    )
    snap = st.snapshot()
    assert snap["status"] == "done"
    assert snap["done_cells"] == 1
    assert resp.closed


def test_run_sweep_pre_fanout_json_error():
    resp = FakeResponse(
        status_code=400,
        content_type="application/json",
        body=json.dumps({"error": {"message": "unknown hook 'resid'"}}).encode(),
    )
    st = sc.SweepState()
    sc.run_sweep(
        st, "http://x:8000", {}, post=lambda url, **kw: resp, label_fn=_no_labels
    )
    snap = st.snapshot()
    assert snap["status"] == "error"
    assert "unknown hook" in snap["error"]


def test_run_sweep_connection_error():
    def post(url, **kw):
        raise sc.requests.ConnectionError("refused")

    st = sc.SweepState()
    sc.run_sweep(st, "http://x:8000", {}, post=post, label_fn=_no_labels)
    assert st.snapshot()["status"] == "error"
    assert "refused" in st.snapshot()["error"]


def test_run_sweep_cancel_stops_stream():
    st = sc.SweepState()

    def lines():
        yield _sse(START)
        st.cancel()
        yield _sse(_cell())
        yield _sse(SUMMARY)

    resp = FakeResponse(lines=lines())
    sc.run_sweep(
        st, "http://x:8000", {}, post=lambda url, **kw: resp, label_fn=_no_labels
    )
    snap = st.snapshot()
    assert snap["status"] == "cancelled"
    assert snap["done_cells"] == 0
    assert resp.closed


def test_run_sweep_truncated_stream_is_an_error():
    resp = FakeResponse(lines=[_sse(START), _sse(_cell())])
    st = sc.SweepState()
    sc.run_sweep(
        st, "http://x:8000", {}, post=lambda url, **kw: resp, label_fn=_no_labels
    )
    snap = st.snapshot()
    assert snap["status"] == "error"
    assert "ended" in snap["error"]


def test_run_sweep_posts_to_patch_sweep_url():
    seen = {}

    def post(url, **kw):
        seen["url"] = url
        seen["json"] = kw.get("json")
        return FakeResponse(lines=[_sse(START), _sse(SUMMARY), b"data: [DONE]"])

    st = sc.SweepState()
    sc.run_sweep(st, "http://x:8000/", {"prompt": "p"}, post=post, label_fn=_no_labels)
    assert seen["url"] == "http://x:8000/v1/patch_sweep"
    assert seen["json"] == {"prompt": "p"}


def test_run_sweep_label_failure_is_nonfatal():
    def label_fn(server_url, model, prompt, post):
        raise sc.requests.ConnectionError("no tokenize route")

    resp = FakeResponse(lines=[_sse(START), _sse(SUMMARY), b"data: [DONE]"])
    st = sc.SweepState()
    sc.run_sweep(
        st,
        "http://x:8000",
        {"prompt": "p"},
        post=lambda url, **kw: resp,
        label_fn=label_fn,
    )
    snap = st.snapshot()
    assert snap["status"] == "done"
    assert snap["token_labels"] == {}


# ---------------------------------------------------------------- token labels


def test_fetch_token_labels_incremental_detokenize():
    calls = []

    def post(url, **kw):
        body = kw["json"]
        calls.append(url)
        if url.endswith("/tokenize"):
            return FakeResponse(
                content_type="application/json",
                body=json.dumps({"tokens": [1, 2, 3]}).encode(),
            )
        prefix = {1: "<s>", 2: "<s>The", 3: "<s>The city"}
        return FakeResponse(
            content_type="application/json",
            body=json.dumps({"prompt": prefix[len(body["tokens"])]}).encode(),
        )

    labels = sc.fetch_token_labels("http://x:8000", None, "The city", post=post)
    assert labels == {0: "<s>", 1: "The", 2: " city"}
    assert calls[0] == "http://x:8000/tokenize"


def test_fetch_token_labels_http_error_raises():
    def post(url, **kw):
        return FakeResponse(
            status_code=500, content_type="application/json", body=b'{"error": "boom"}'
        )

    with pytest.raises(sc.requests.HTTPError):
        sc.fetch_token_labels("http://x:8000", "m", "p", post=post)


# ---------------------------------------------------------------- concurrency


def test_state_is_thread_safe_under_concurrent_cells():
    st = sc.SweepState()
    layers = list(range(8))
    positions = list(range(8))
    st.apply_event(
        {**START, "layers": layers, "positions": positions, "hooks": ["post_block"]}
    )

    def write(layer):
        for pos in positions:
            st.apply_event(_cell(layer=layer, position=pos, value=0.1))

    threads = [threading.Thread(target=write, args=(la,)) for la in layers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert st.snapshot()["done_cells"] == 64
