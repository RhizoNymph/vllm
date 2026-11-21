# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-aware validation of per-request and named-module steering specs.

Covers :class:`vllm.config.steering_geometry.SteeringSiteGeometry`: the
per-hook row widths and layer count resolved once from the model config,
and every seam that gates client steering specs with it (inline packing on
the AsyncLLM / offline paths, the engine input processor, and the named
module registry), plus the HTTP status the frontends give its errors.
"""

import base64
from http import HTTPStatus
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.steering import SteeringConfig
from vllm.config.steering_geometry import (
    SteeringSiteGeometry,
    resolve_steering_site_geometry,
)
from vllm.config.steering_types import (
    SteeringClamps,
    coerce_steering_spec,
    maybe_pack_inline_steering_for_request,
)
from vllm.exceptions import (
    SteeringSiteValidationError,
    SteeringVectorError,
    VLLMClientError,
    VLLMValidationError,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.capture.types import (
    build_hook_schema,
    build_model_hook_schema,
    default_hook_schema,
)

HIDDEN = 8
HC = 4
LAYERS = 48
STREAM = HC * HIDDEN

STANDARD = ("pre_attn", "post_attn", "post_block", "mlp_in", "mlp_out")
STREAM_HOOKS = ("mhc_streams_pre_attn", "mhc_streams_pre_mlp", "mhc_streams_final")


def _model_config(*, hc_mult=None, text_config=None, layers=LAYERS):
    return SimpleNamespace(
        get_hidden_size=lambda: HIDDEN,
        get_total_num_hidden_layers=lambda: layers,
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(hc_mult=hc_mult)
        if hc_mult is not None
        else SimpleNamespace(),
        hf_text_config=text_config or SimpleNamespace(model_type="llama"),
    )


def _mhc_geometry() -> SteeringSiteGeometry:
    return resolve_steering_site_geometry(_model_config(hc_mult=HC))


def _standard_geometry() -> SteeringSiteGeometry:
    return resolve_steering_site_geometry(_model_config())


def _packed(rows: np.ndarray, layers: list[int]) -> dict:
    rows = rows.astype(np.float32)
    return {
        "dtype": "float32",
        "shape": list(rows.shape),
        "layer_indices": layers,
        "data": base64.b64encode(rows.tobytes()).decode("ascii"),
    }


# ---------------------------------------------------------------------------
# Hook schema resolution from the HF configs
# ---------------------------------------------------------------------------


class TestBuildModelHookSchema:
    def test_standard_model_matches_default_schema(self):
        schema = build_model_hook_schema(
            HIDDEN, torch.bfloat16, SimpleNamespace(), SimpleNamespace()
        )
        assert schema == default_hook_schema(HIDDEN, torch.bfloat16)

    def test_hc_mult_model_matches_mhc_schema(self):
        schema = build_model_hook_schema(
            HIDDEN, torch.bfloat16, SimpleNamespace(hc_mult=HC), SimpleNamespace()
        )
        assert schema == build_hook_schema(HIDDEN, torch.bfloat16, HC)

    def test_missing_configs_fall_back_to_default(self):
        schema = build_model_hook_schema(HIDDEN, torch.float32, None, None)
        assert schema == default_hook_schema(HIDDEN, torch.float32)


# ---------------------------------------------------------------------------
# SteeringSiteGeometry construction and resolution
# ---------------------------------------------------------------------------


class TestResolveGeometry:
    def test_standard_model_every_hook_is_hidden_wide(self):
        geom = _standard_geometry()
        assert dict(geom.hook_widths) == {hook: HIDDEN for hook in STANDARD}
        assert geom.num_hidden_layers == LAYERS

    def test_mhc_model_stream_hooks_are_hc_times_hidden(self):
        geom = _mhc_geometry()
        for hook in STREAM_HOOKS:
            assert geom.hook_widths[hook] == STREAM
        for hook in ("pre_attn", "post_attn", "mlp_in", "mlp_out"):
            assert geom.hook_widths[hook] == HIDDEN

    def test_mhc_model_capture_only_hooks_are_not_steerable(self):
        # The mixing-coefficient hooks are capture targets only; V4 has no
        # single-stream post_block.
        geom = _mhc_geometry()
        assert set(geom.hook_widths) == {
            "pre_attn",
            "post_attn",
            "mlp_in",
            "mlp_out",
            *STREAM_HOOKS,
        }

    def test_hook_widths_are_read_only(self):
        geom = _standard_geometry()
        assert isinstance(geom.hook_widths, MappingProxyType)
        with pytest.raises(TypeError):
            geom.hook_widths["pre_attn"] = 3  # type: ignore[index]

    def test_geometry_is_frozen(self):
        geom = _standard_geometry()
        with pytest.raises(AttributeError):
            geom.num_hidden_layers = 3  # type: ignore[misc]

    def test_standard_constructor(self):
        geom = SteeringSiteGeometry.standard(HIDDEN, num_hidden_layers=LAYERS)
        assert geom == _standard_geometry()

    def test_caller_mapping_is_copied(self):
        widths = {"pre_attn": HIDDEN}
        geom = SteeringSiteGeometry(hook_widths=widths, num_hidden_layers=2)
        widths["pre_attn"] = 1
        assert geom.hook_widths["pre_attn"] == HIDDEN

    @pytest.mark.parametrize(
        "widths,layers",
        [
            ({}, 4),
            ({"pre_attn": 0}, 4),
            ({"pre_attn": -1}, 4),
            ({"pre_attn": 2.0}, 4),
            ({"pre_attn": True}, 4),
            ({"not_a_hook": 4}, 4),
            ({"pre_attn": 4}, 0),
            ({"pre_attn": 4}, -2),
        ],
    )
    def test_invalid_geometry_rejected(self, widths, layers):
        with pytest.raises(ValueError):
            SteeringSiteGeometry(hook_widths=widths, num_hidden_layers=layers)


# ---------------------------------------------------------------------------
# Spec validation against the geometry
# ---------------------------------------------------------------------------


class TestValidateVectors:
    def test_stream_hook_at_stream_width_accepted(self):
        _mhc_geometry().validate_vectors(
            {"mhc_streams_pre_attn": {3: [0.0] * STREAM}},
            field_name="steering_vectors",
        )

    def test_stream_hook_at_hidden_width_rejected(self):
        with pytest.raises(
            SteeringSiteValidationError,
            match=(
                r"steering_vectors\['mhc_streams_pre_attn'\]\[3\]: vector "
                rf"width {HIDDEN} != expected steering row width {STREAM}"
            ),
        ):
            _mhc_geometry().validate_vectors(
                {"mhc_streams_pre_attn": {3: [0.0] * HIDDEN}},
                field_name="steering_vectors",
            )

    def test_standard_hook_on_mhc_model_stays_hidden_wide(self):
        geom = _mhc_geometry()
        geom.validate_vectors(
            {"pre_attn": {0: [0.0] * HIDDEN}}, field_name="steering_vectors"
        )
        with pytest.raises(SteeringSiteValidationError, match="width"):
            geom.validate_vectors(
                {"pre_attn": {0: [0.0] * STREAM}}, field_name="steering_vectors"
            )

    def test_wrong_width_on_standard_hook_names_field_and_widths(self):
        with pytest.raises(SteeringSiteValidationError) as exc_info:
            _standard_geometry().validate_vectors(
                {"pre_attn": {0: [0.0] * 100}}, field_name="steering_vectors"
            )
        msg = str(exc_info.value)
        assert "steering_vectors['pre_attn'][0]" in msg
        assert "100" in msg and str(HIDDEN) in msg
        assert exc_info.value.parameter == "steering_vectors"

    def test_scaled_and_ndarray_entries_width_checked(self):
        geom = _standard_geometry()
        with pytest.raises(SteeringSiteValidationError, match="width 3"):
            geom.validate_vectors(
                {"post_block": {1: {"vector": [1.0, 2.0, 3.0], "scale": 2.0}}},
                field_name="prefill_steering_vectors",
            )
        with pytest.raises(SteeringSiteValidationError, match="width 5"):
            geom.validate_vectors(
                {"post_block": {1: np.zeros(5, dtype=np.float32)}},
                field_name="decode_steering_vectors",
            )

    def test_hook_not_on_this_model_rejected(self):
        with pytest.raises(
            SteeringSiteValidationError,
            match=r"'mhc_streams_pre_attn' is not a steering hook of this model",
        ) as exc_info:
            _standard_geometry().validate_vectors(
                {"mhc_streams_pre_attn": {0: [0.0] * STREAM}},
                field_name="steering_vectors",
            )
        # The message lists what the model does offer.
        assert "post_block" in str(exc_info.value)

    def test_post_block_not_on_mhc_model_rejected(self):
        with pytest.raises(SteeringSiteValidationError, match="post_block"):
            _mhc_geometry().validate_vectors(
                {"post_block": {0: [0.0] * HIDDEN}}, field_name="steering_vectors"
            )

    @pytest.mark.parametrize("layer", [LAYERS, 99, 10**6])
    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_layer_out_of_range_rejected(self, layer, value):
        with pytest.raises(
            SteeringSiteValidationError,
            match=(
                rf"steering_vectors\['pre_attn'\]\[{layer}\]: layer index "
                rf"{layer} out of range for num_hidden_layers={LAYERS}"
            ),
        ):
            _standard_geometry().validate_vectors(
                {"pre_attn": {layer: [value] * HIDDEN}},
                field_name="steering_vectors",
            )

    def test_last_layer_accepted(self):
        _standard_geometry().validate_vectors(
            {"pre_attn": {0: [0.0] * HIDDEN, LAYERS - 1: [0.0] * HIDDEN}},
            field_name="steering_vectors",
        )

    def test_numpy_integer_layer_key_range_checked(self):
        geom = _standard_geometry()
        geom.validate_vectors(
            {"pre_attn": {np.int64(3): [0.0] * HIDDEN}}, field_name="f"
        )
        with pytest.raises(SteeringSiteValidationError, match="out of range"):
            geom.validate_vectors(
                {"pre_attn": {np.int64(LAYERS): [0.0] * HIDDEN}}, field_name="f"
            )

    def test_layer_at_pass_key_checks_only_the_layer_part(self):
        # Looped models address one pass of a layer as "layer@pass"; pass
        # validity is checked by the loop-layer expansion, so only the
        # layer part is range-checked here.
        geom = _standard_geometry()
        geom.validate_vectors(
            {"pre_attn": {"11@1": [0.0] * HIDDEN, "7": [0.0] * HIDDEN}},
            field_name="f",
        )
        with pytest.raises(SteeringSiteValidationError, match="layer index 50"):
            geom.validate_vectors(
                {"pre_attn": {"50@1": [0.0] * HIDDEN}}, field_name="f"
            )

    @pytest.mark.parametrize("key", ["abc", "@1", "-1", 1.5, None, True])
    def test_malformed_layer_key_rejected(self, key):
        with pytest.raises(SteeringSiteValidationError, match="layer"):
            _standard_geometry().validate_vectors(
                {"pre_attn": {key: [0.0] * HIDDEN}}, field_name="f"
            )

    def test_empty_spec_is_noop(self):
        geom = _standard_geometry()
        geom.validate_vectors(None, field_name="f")
        geom.validate_vectors({}, field_name="f")


class TestValidateClamps:
    def _clamps(self, hook: str, layer: int, width: int) -> SteeringClamps:
        spec = SteeringClamps.from_obj(
            {hook: {layer: [{"vector": [1.0] + [0.0] * (width - 1), "max": 1.0}]}}
        )
        assert spec is not None
        return spec

    def test_stream_hook_clamp_at_stream_width_accepted(self):
        _mhc_geometry().validate_clamps(
            self._clamps("mhc_streams_pre_mlp", 2, STREAM), field_name="clamps"
        )

    def test_stream_hook_clamp_at_hidden_width_rejected(self):
        with pytest.raises(
            SteeringSiteValidationError,
            match=rf"clamps\['mhc_streams_pre_mlp'\].*width {HIDDEN}.*{STREAM}",
        ):
            _mhc_geometry().validate_clamps(
                self._clamps("mhc_streams_pre_mlp", 2, HIDDEN), field_name="clamps"
            )

    def test_clamp_layer_out_of_range_rejected(self):
        with pytest.raises(SteeringSiteValidationError, match="out of range"):
            _standard_geometry().validate_clamps(
                self._clamps("post_block", LAYERS, HIDDEN), field_name="clamps"
            )

    def test_clamp_hook_not_on_model_rejected(self):
        with pytest.raises(SteeringSiteValidationError, match="not a steering hook"):
            _standard_geometry().validate_clamps(
                self._clamps("mhc_streams_final", 0, STREAM), field_name="clamps"
            )

    def test_none_is_noop(self):
        _standard_geometry().validate_clamps(None, field_name="clamps")


# ---------------------------------------------------------------------------
# Error typing and HTTP mapping
# ---------------------------------------------------------------------------


class TestErrorTyping:
    def test_error_is_a_client_validation_error_and_value_error(self):
        err = SteeringSiteValidationError("boom", parameter="steering_vectors")
        assert isinstance(err, SteeringVectorError)
        assert isinstance(err, VLLMValidationError)
        assert isinstance(err, VLLMClientError)
        # Kept a ValueError so pydantic validators and existing ValueError
        # handlers keep catching it.
        assert isinstance(err, ValueError)

    def test_create_error_response_maps_to_400_with_message(self):
        from vllm.entrypoints.serve.exception_handling.error_response import (
            create_error_response,
        )

        with pytest.raises(SteeringSiteValidationError) as exc_info:
            _standard_geometry().validate_vectors(
                {"pre_attn": {0: [0.0] * 100}}, field_name="steering_vectors"
            )
        resp = create_error_response(exc_info.value)
        assert resp.error.code == HTTPStatus.BAD_REQUEST
        assert resp.error.type == "BadRequestError"
        assert "steering_vectors['pre_attn'][0]" in resp.error.message
        assert "100" in resp.error.message
        assert resp.error.param == "steering_vectors"


# ---------------------------------------------------------------------------
# Inline packing gate (the AsyncLLM / offline LLM seam)
# ---------------------------------------------------------------------------


class TestInlinePackingGate:
    def test_mhc_stream_vector_packed_form_accepted(self):
        rows = np.full((1, STREAM), 0.5, dtype=np.float32)
        spec = coerce_steering_spec({"mhc_streams_pre_attn": _packed(rows, [9])})
        sp = SamplingParams(max_tokens=1, steering_vectors=spec)
        maybe_pack_inline_steering_for_request(
            sp, torch.float32, site_geometry=_mhc_geometry()
        )
        assert sp.steering_vectors is None
        packed = sp._effective_prefill_steering_packed
        assert packed is not None
        assert packed["mhc_streams_pre_attn"][9].shape == (STREAM,)

    def test_wrong_width_rejected_before_packing(self):
        sp = SamplingParams(
            max_tokens=1, steering_vectors={"pre_attn": {0: [0.0] * 100}}
        )
        with pytest.raises(SteeringSiteValidationError, match="width 100"):
            maybe_pack_inline_steering_for_request(
                sp, torch.float32, site_geometry=_standard_geometry()
            )
        assert sp.steering_vectors is not None
        assert sp._effective_prefill_steering_packed is None

    @pytest.mark.parametrize("layer", [LAYERS, 99])
    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_out_of_range_layer_rejected(self, layer, value):
        sp = SamplingParams(
            max_tokens=1,
            decode_steering_vectors={"post_block": {layer: [value] * HIDDEN}},
        )
        with pytest.raises(
            SteeringSiteValidationError,
            match=r"decode_steering_vectors\['post_block'\].*out of range",
        ):
            maybe_pack_inline_steering_for_request(
                sp, torch.float32, site_geometry=_standard_geometry()
            )


# ---------------------------------------------------------------------------
# AsyncLLM: the gate's error reaches the frontend typed, not as a bare 500
# ---------------------------------------------------------------------------


def _async_llm_stub(geometry: SteeringSiteGeometry):
    from vllm.v1.engine.async_llm import AsyncLLM

    stub = SimpleNamespace(
        errored=False,
        log_requests=False,
        observability_config=SimpleNamespace(otlp_traces_endpoint=None),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.float32),
        ),
        input_processor=SimpleNamespace(steering_site_geometry=geometry),
    )
    stub.add_request = AsyncLLM.add_request.__get__(stub)
    stub.generate = AsyncLLM.generate.__get__(stub)
    return stub


class TestAsyncLLMSurfacesTypedError:
    def test_add_request_raises_typed_width_error(self):
        import asyncio

        stub = _async_llm_stub(_standard_geometry())
        sp = SamplingParams(
            max_tokens=1, steering_vectors={"pre_attn": {0: [0.0] * 100}}
        )
        with pytest.raises(SteeringSiteValidationError, match="width 100"):
            asyncio.run(stub.add_request("r0", "hi", sp))

    def test_generate_does_not_wrap_in_engine_generate_error(self):
        import asyncio

        stub = _async_llm_stub(_mhc_geometry())
        sp = SamplingParams(
            max_tokens=1, steering_vectors={"pre_attn": {LAYERS: [1.0] * HIDDEN}}
        )

        async def _drive():
            async for _ in stub.generate("hi", sp, "r0"):
                pass

        with pytest.raises(SteeringSiteValidationError, match="out of range"):
            asyncio.run(_drive())

    def test_add_request_accepts_stream_hook_vector(self):
        import asyncio

        stub = _async_llm_stub(_mhc_geometry())
        sp = SamplingParams(
            max_tokens=1,
            steering_vectors={"mhc_streams_pre_attn": {0: [0.0] * STREAM}},
        )

        # Past the steering gate the stub has no further engine plumbing;
        # reaching it (AttributeError on the missing cache config) proves
        # the gate accepted and packed the stream-hook vector.
        with pytest.raises(AttributeError, match="cache_config"):
            asyncio.run(stub.add_request("r0", "hi", sp))
        assert sp._effective_prefill_steering_packed is not None


class TestOfflineLLMGate:
    def test_offline_pack_uses_engine_geometry(self):
        from vllm.entrypoints.common.offline import OfflineInferenceMixin as LLM

        stub = SimpleNamespace(
            llm_engine=SimpleNamespace(
                model_config=SimpleNamespace(dtype=torch.float32),
                input_processor=SimpleNamespace(steering_site_geometry=_mhc_geometry()),
            )
        )
        ok = SamplingParams(
            max_tokens=1,
            steering_vectors={"mhc_streams_final": {LAYERS - 1: [0.0] * STREAM}},
        )
        LLM._maybe_pack_inline_steering(stub, ok)
        assert ok._effective_prefill_steering_packed is not None

        bad = SamplingParams(
            max_tokens=1, steering_vectors={"pre_attn": {99: [0.0] * HIDDEN}}
        )
        with pytest.raises(SteeringSiteValidationError, match="out of range"):
            LLM._maybe_pack_inline_steering(stub, bad)


# ---------------------------------------------------------------------------
# Engine input processor (every frontend, including Rust)
# ---------------------------------------------------------------------------


def _processor(geometry: SteeringSiteGeometry, max_clamp_directions: int = 4):
    from vllm.v1.engine.input_processor import InputProcessor

    proc = SimpleNamespace(
        steering_config=SteeringConfig(max_clamp_directions=max_clamp_directions),
        steering_site_geometry=geometry,
    )
    proc._validate_steering = InputProcessor._validate_steering.__get__(proc)
    return proc


class TestInputProcessorGate:
    def test_stream_hook_vector_accepted(self):
        _processor(_mhc_geometry())._validate_steering(
            SamplingParams(
                steering_vectors={"mhc_streams_pre_mlp": {5: [0.0] * STREAM}}
            )
        )

    def test_stream_hook_clamp_accepted(self):
        _processor(_mhc_geometry())._validate_steering(
            SamplingParams(
                steering_clamps={
                    "mhc_streams_pre_attn": {
                        5: [{"vector": [1.0] + [0.0] * (STREAM - 1), "max": 1.0}]
                    }
                }
            )
        )

    def test_out_of_range_layer_rejected_typed(self):
        with pytest.raises(SteeringSiteValidationError, match="out of range") as e:
            _processor(_standard_geometry())._validate_steering(
                SamplingParams(steering_vectors={"pre_attn": {99: [0.0] * HIDDEN}})
            )
        assert e.value.parameter == "steering_vectors"

    def test_wrong_width_clamp_rejected_typed(self):
        with pytest.raises(SteeringSiteValidationError, match="width"):
            _processor(_mhc_geometry())._validate_steering(
                SamplingParams(
                    steering_clamps={
                        "mhc_streams_pre_attn": {
                            5: [{"vector": [1.0] + [0.0] * (HIDDEN - 1), "max": 1.0}]
                        }
                    }
                )
            )


# ---------------------------------------------------------------------------
# Named module registry
# ---------------------------------------------------------------------------


class TestRegistryGate:
    def _registry(self, geometry: SteeringSiteGeometry):
        from vllm.entrypoints.openai.steering.registry import SteeringModuleRegistry

        return SteeringModuleRegistry(site_geometry=geometry)

    def test_stream_hook_module_accepted(self):
        import asyncio

        rows = np.zeros((2, STREAM), dtype=np.float32)
        registry = self._registry(_mhc_geometry())
        asyncio.run(
            registry.register(
                name="streams",
                vectors={"mhc_streams_pre_attn": _packed(rows, [3, 4])},
            )
        )
        assert "streams" in registry.list_modules()

    def test_stream_hook_module_wrong_width_rejected(self):
        import asyncio

        registry = self._registry(_mhc_geometry())
        with pytest.raises(SteeringSiteValidationError, match=rf"{STREAM}"):
            asyncio.run(
                registry.register(
                    name="narrow",
                    vectors={"mhc_streams_pre_attn": {3: [0.0] * HIDDEN}},
                )
            )

    def test_out_of_range_layer_rejected(self):
        import asyncio

        registry = self._registry(_standard_geometry())
        with pytest.raises(SteeringSiteValidationError, match="out of range"):
            asyncio.run(
                registry.register(
                    name="deep", decode_vectors={"post_block": {LAYERS: [0.0] * HIDDEN}}
                )
            )

    def test_stream_hook_clamp_module_accepted(self):
        import asyncio

        registry = self._registry(_mhc_geometry())
        asyncio.run(
            registry.register(
                name="clamped",
                clamps={
                    "mhc_streams_final": {
                        str(LAYERS - 1): [
                            {"vector": [1.0] + [0.0] * (STREAM - 1), "max": 1.0}
                        ]
                    }
                },
            )
        )
        assert "clamped" in registry.list_modules()
