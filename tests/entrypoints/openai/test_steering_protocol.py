# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for steering vector plumbing through OpenAI-compatible protocol models.

The HTTP fields ``steering_vectors`` / ``prefill_steering_vectors`` /
``decode_steering_vectors`` accept only the binary wire format
(``SteeringHookPacked``).  The legacy ``list[float]`` form is rejected at
pydantic validation — clients must pack vectors before sending.

Covers:

- both ``ChatCompletionRequest`` and ``CompletionRequest`` accept all three
  tiers in packed form
- legacy ``list[float]`` and ``{"vector": [...], "scale": float}`` request
  bodies fail validation
- ``to_sampling_params`` unpacks the wire format to per-layer ``ndarray``
  dicts on ``SamplingParams``
- optional per-row ``scales`` are applied at unpack time
- ``steering_name`` field is unaffected
"""

import numpy as np
import pybase64 as base64
import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_HIDDEN = 8


_DEFAULT_PACK_DTYPE = np.dtype(np.float32)


def _pack(
    layer_vectors: dict[int, list[float]],
    *,
    dtype: np.dtype = _DEFAULT_PACK_DTYPE,
    scales: list[float] | None = None,
) -> dict:
    """Build one ``SteeringHookPacked`` blob from per-layer Python lists."""
    layer_indices = sorted(layer_vectors.keys())
    stacked = np.stack(
        [np.asarray(layer_vectors[i], dtype=dtype) for i in layer_indices],
        axis=0,
    )
    blob: dict = {
        "dtype": str(stacked.dtype),
        "shape": list(stacked.shape),
        "layer_indices": layer_indices,
        "data": base64.b64encode(stacked.tobytes()).decode("ascii"),
    }
    if scales is not None:
        blob["scales"] = scales
    return blob


_BASE_PACKED = {"pre_attn": _pack({15: [0.1] * _HIDDEN})}
_PREFILL_PACKED = {"pre_attn": _pack({15: [0.7] * _HIDDEN})}
_DECODE_PACKED = {"post_attn": _pack({20: [1.0] * _HIDDEN})}

_CHAT_BASE = {
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "test-model",
}

_COMPLETION_BASE = {
    "prompt": "Hello",
    "model": "test-model",
}


def _make_chat(**extra):
    return ChatCompletionRequest.model_validate({**_CHAT_BASE, **extra})


def _make_completion(**extra):
    return CompletionRequest.model_validate({**_COMPLETION_BASE, **extra})


# ---------------------------------------------------------------------------
# ChatCompletionRequest tests
# ---------------------------------------------------------------------------


class TestChatCompletionSteering:
    """ChatCompletionRequest steering field acceptance."""

    def test_no_steering_fields(self):
        req = _make_chat()
        assert req.steering_vectors is None
        assert req.prefill_steering_vectors is None
        assert req.decode_steering_vectors is None

    def test_packed_steering_vectors(self):
        req = _make_chat(steering_vectors=_BASE_PACKED)
        assert req.steering_vectors == _BASE_PACKED

    def test_packed_prefill_steering_vectors(self):
        req = _make_chat(prefill_steering_vectors=_PREFILL_PACKED)
        assert req.prefill_steering_vectors == _PREFILL_PACKED

    def test_packed_decode_steering_vectors(self):
        req = _make_chat(decode_steering_vectors=_DECODE_PACKED)
        assert req.decode_steering_vectors == _DECODE_PACKED

    def test_all_three_tiers(self):
        req = _make_chat(
            steering_vectors=_BASE_PACKED,
            prefill_steering_vectors=_PREFILL_PACKED,
            decode_steering_vectors=_DECODE_PACKED,
        )
        assert req.steering_vectors == _BASE_PACKED
        assert req.prefill_steering_vectors == _PREFILL_PACKED
        assert req.decode_steering_vectors == _DECODE_PACKED

    def test_legacy_list_of_floats_rejected(self):
        with pytest.raises(ValidationError):
            _make_chat(steering_vectors={"pre_attn": {15: [0.1, 0.2, 0.3]}})

    def test_legacy_scaled_dict_rejected(self):
        with pytest.raises(ValidationError):
            _make_chat(
                steering_vectors={
                    "post_block": {10: {"vector": [0.4, 0.5, 0.6], "scale": 2.0}}
                }
            )

    def test_to_sampling_params_unpacks_all_fields(self):
        req = _make_chat(
            steering_vectors=_BASE_PACKED,
            prefill_steering_vectors=_PREFILL_PACKED,
            decode_steering_vectors=_DECODE_PACKED,
        )
        sp = req.to_sampling_params(max_tokens=100, default_sampling_params={})
        assert sp.steering_vectors is not None
        assert sp.steering_vectors["pre_attn"][15].tolist() == pytest.approx(
            [0.1] * _HIDDEN
        )
        assert sp.prefill_steering_vectors["pre_attn"][15].tolist() == pytest.approx(
            [0.7] * _HIDDEN
        )
        assert sp.decode_steering_vectors["post_attn"][20].tolist() == pytest.approx(
            [1.0] * _HIDDEN
        )

    def test_to_sampling_params_none_when_absent(self):
        req = _make_chat()
        sp = req.to_sampling_params(max_tokens=100, default_sampling_params={})
        assert sp.steering_vectors is None
        assert sp.prefill_steering_vectors is None
        assert sp.decode_steering_vectors is None

    def test_per_row_scales_applied_at_unpack(self):
        packed = {
            "post_block": _pack({10: [1.0] * _HIDDEN}, scales=[2.0]),
        }
        req = _make_chat(steering_vectors=packed)
        sp = req.to_sampling_params(max_tokens=100, default_sampling_params={})
        assert sp.steering_vectors["post_block"][10].tolist() == pytest.approx(
            [2.0] * _HIDDEN
        )


# ---------------------------------------------------------------------------
# CompletionRequest tests
# ---------------------------------------------------------------------------


class TestCompletionSteering:
    """CompletionRequest steering field acceptance."""

    def test_no_steering_fields(self):
        req = _make_completion()
        assert req.steering_vectors is None
        assert req.prefill_steering_vectors is None
        assert req.decode_steering_vectors is None

    def test_packed_steering_vectors(self):
        req = _make_completion(steering_vectors=_BASE_PACKED)
        assert req.steering_vectors == _BASE_PACKED

    def test_packed_prefill_steering_vectors(self):
        req = _make_completion(prefill_steering_vectors=_PREFILL_PACKED)
        assert req.prefill_steering_vectors == _PREFILL_PACKED

    def test_packed_decode_steering_vectors(self):
        req = _make_completion(decode_steering_vectors=_DECODE_PACKED)
        assert req.decode_steering_vectors == _DECODE_PACKED

    def test_all_three_tiers(self):
        req = _make_completion(
            steering_vectors=_BASE_PACKED,
            prefill_steering_vectors=_PREFILL_PACKED,
            decode_steering_vectors=_DECODE_PACKED,
        )
        assert req.steering_vectors == _BASE_PACKED
        assert req.prefill_steering_vectors == _PREFILL_PACKED
        assert req.decode_steering_vectors == _DECODE_PACKED

    def test_legacy_list_of_floats_rejected(self):
        with pytest.raises(ValidationError):
            _make_completion(steering_vectors={"pre_attn": {15: [0.1, 0.2, 0.3]}})

    def test_legacy_scaled_dict_rejected(self):
        with pytest.raises(ValidationError):
            _make_completion(
                steering_vectors={
                    "post_block": {10: {"vector": [0.4, 0.5, 0.6], "scale": 2.0}}
                }
            )

    def test_to_sampling_params_unpacks_all_fields(self):
        req = _make_completion(
            steering_vectors=_BASE_PACKED,
            prefill_steering_vectors=_PREFILL_PACKED,
            decode_steering_vectors=_DECODE_PACKED,
        )
        sp = req.to_sampling_params(max_tokens=100)
        assert sp.steering_vectors["pre_attn"][15].tolist() == pytest.approx(
            [0.1] * _HIDDEN
        )
        assert sp.prefill_steering_vectors["pre_attn"][15].tolist() == pytest.approx(
            [0.7] * _HIDDEN
        )
        assert sp.decode_steering_vectors["post_attn"][20].tolist() == pytest.approx(
            [1.0] * _HIDDEN
        )

    def test_to_sampling_params_none_when_absent(self):
        req = _make_completion()
        sp = req.to_sampling_params(max_tokens=100)
        assert sp.steering_vectors is None
        assert sp.prefill_steering_vectors is None
        assert sp.decode_steering_vectors is None

    def test_per_row_scales_applied_at_unpack(self):
        packed = {
            "post_block": _pack({10: [1.0] * _HIDDEN}, scales=[2.0]),
        }
        req = _make_completion(steering_vectors=packed)
        sp = req.to_sampling_params(max_tokens=100)
        assert sp.steering_vectors["post_block"][10].tolist() == pytest.approx(
            [2.0] * _HIDDEN
        )


# ---------------------------------------------------------------------------
# steering_name field tests
# ---------------------------------------------------------------------------


class TestSteeringNameField:
    """Tests for the steering_name protocol field."""

    def test_chat_completion_steering_name_field(self):
        """steering_name field is accepted and defaults to None."""
        chat = _make_chat()
        assert chat.steering_name is None

        chat_with_name = _make_chat(steering_name="creativity")
        assert chat_with_name.steering_name == "creativity"

    def test_completion_steering_name_field(self):
        """steering_name field is accepted and defaults to None."""
        comp = _make_completion()
        assert comp.steering_name is None

        comp_with_name = _make_completion(steering_name="safety")
        assert comp_with_name.steering_name == "safety"

    def test_steering_name_coexists_with_inline_vectors(self):
        """Both steering_name and inline vectors can be set simultaneously."""
        chat = _make_chat(
            steering_name="base_personality",
            steering_vectors=_BASE_PACKED,
        )
        assert chat.steering_name == "base_personality"
        assert chat.steering_vectors is not None

    def test_conversation_id_defaults_none(self):
        assert _make_chat().conversation_id is None
        assert _make_completion().conversation_id is None

    def test_conversation_id_threads_to_sampling_params(self):
        """conversation_id on the request reaches SamplingParams (and thus the
        worker / StepRequestView), for both chat and completion."""
        chat = _make_chat(conversation_id="conv-7")
        assert chat.conversation_id == "conv-7"
        sp = chat.to_sampling_params(max_tokens=8, default_sampling_params={})
        assert sp.conversation_id == "conv-7"

        comp = _make_completion(conversation_id="conv-9")
        assert comp.conversation_id == "conv-9"
        sp_c = comp.to_sampling_params(max_tokens=8, default_sampling_params={})
        assert sp_c.conversation_id == "conv-9"

    def test_conversation_id_none_when_absent(self):
        sp = _make_chat().to_sampling_params(max_tokens=8, default_sampling_params={})
        assert sp.conversation_id is None
