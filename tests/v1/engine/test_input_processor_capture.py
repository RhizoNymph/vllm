# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline capture admission in the v1 ``InputProcessor``.

The OpenAI serving layer resolves a request's ``capture`` dict into
prefix-cache reuse flags in ``_admit_capture``. Offline / direct ``LLM``
requests never pass through that layer, so the ``InputProcessor`` runs the
same resolution (``_resolve_capture_prefix_flags``) before the request
reaches the scheduler. These tests cover that offline path directly,
bypassing the heavy ``InputProcessor.__init__``.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm.sampling_params import SamplingParams
from vllm.v1.capture import CaptureConsumer, CaptureSpec
from vllm.v1.capture.errors import CaptureValidationError
from vllm.v1.engine.input_processor import InputProcessor


class _PromptConsumer(CaptureConsumer):
    reads_client_spec = True

    def __init__(self) -> None:  # noqa: D107
        pass

    def validate_client_spec(self, raw_spec, ctx):  # type: ignore[override]
        return CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")

    def on_capture(self, key, tensor, sidecar):  # pragma: no cover - unused
        pass


def _stub_vllm_config(capture_config):
    """Minimal ``vllm_config`` exposing what admission reads."""
    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )
    model_config = SimpleNamespace(
        get_total_num_hidden_layers=lambda: 32,
        get_hidden_size=lambda: 4096,
        dtype=SimpleNamespace(itemsize=2),
    )
    return SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        capture_consumers_config=capture_config,
    )


def _processor(capture_config, consumers) -> InputProcessor:
    proc = InputProcessor.__new__(InputProcessor)
    proc.vllm_config = _stub_vllm_config(capture_config)
    proc._capture_consumers = consumers
    return proc


class TestOfflinePrefixFlagResolution:
    def test_stamps_flags_for_prompt_touching_spec(self) -> None:
        proc = _processor(object(), {"fs": _PromptConsumer()})
        sp = SamplingParams(capture={"fs": {"hooks": {"post_mlp": [0]}}})

        proc._resolve_capture_prefix_flags("r", sp, [1, 2, 3, 4, 5, 6, 7, 8], None)

        # ``last_prompt`` over an 8-token prompt taps position 7.
        assert sp.capture_touches_prompt is True
        assert sp.capture_min_prompt_position == 7
        assert sp.capture_store_positions == [7]

    def test_noop_when_no_capture_config(self) -> None:
        # No ``capture_consumers_config`` → resolution is skipped entirely
        # and the flags stay unset (conservative: prefix caching disabled
        # for the capture by the request accessor).
        proc = _processor(None, None)
        sp = SamplingParams(capture={"fs": {}})

        proc._resolve_capture_prefix_flags("r", sp, [1, 2, 3], None)

        assert sp.capture_touches_prompt is None

    def test_unknown_consumer_is_conservative_skip(self) -> None:
        # An unresolvable consumer name (e.g. instance-form) must NOT reject
        # the request offline; it leaves the flags unset so the worker
        # re-validates and the request stays correct (no reuse).
        proc = _processor(object(), {"fs": _PromptConsumer()})
        sp = SamplingParams(capture={"not_configured": {}})

        proc._resolve_capture_prefix_flags("r", sp, [1, 2, 3], None)

        assert sp.capture_touches_prompt is None
        assert sp.capture_min_prompt_position is None

    def test_invalid_spec_is_conservative_skip(self) -> None:
        class _Rejects(CaptureConsumer):
            reads_client_spec = True

            def __init__(self) -> None:
                pass

            def validate_client_spec(self, raw_spec, ctx):  # type: ignore[override]
                raise CaptureValidationError("nope")

            def on_capture(self, key, tensor, sidecar):  # pragma: no cover
                pass

        proc = _processor(object(), {"fs": _Rejects()})
        sp = SamplingParams(capture={"fs": {}})

        proc._resolve_capture_prefix_flags("r", sp, [1, 2, 3], None)

        assert sp.capture_touches_prompt is None


class TestBuildCaptureConsumers:
    def test_builds_spec_form_and_skips_instances(self, monkeypatch) -> None:
        built: list[str] = []

        def _fake_build_consumer(name, vllm_config, params):
            built.append(name)
            return _PromptConsumer()

        import vllm.v1.capture.registry as registry

        monkeypatch.setattr(registry, "build_consumer", _fake_build_consumer)

        config = SimpleNamespace(
            consumers=[
                SimpleNamespace(name="filesystem", instance_name=None, params={}),
                SimpleNamespace(
                    name="filesystem", instance_name="fs2", params={"root": "/x"}
                ),
            ],
            instances=[object()],  # instance-form: intentionally ignored
        )
        proc = _processor(config, None)

        consumers = proc._build_capture_consumers(config)

        # Keyed by ``instance_name or name``; instances are not included.
        assert set(consumers.keys()) == {"filesystem", "fs2"}
        assert built == ["filesystem", "filesystem"]

    def test_empty_config_builds_nothing(self) -> None:
        config = SimpleNamespace(consumers=[], instances=[])
        proc = _processor(config, None)
        assert proc._build_capture_consumers(config) == {}
