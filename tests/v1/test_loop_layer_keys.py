# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer keys of looped models (`vllm.config.loop_layers`): one intervention
site per (layer, pass), user `"L@p"` / plain-`L` (all passes) addressing,
and idempotent expansion."""

from types import SimpleNamespace

import pytest

from vllm.config import loop_layers
from vllm.config.loop_layers import (
    LoopLayerKeyError,
    LoopLayerSpace,
    expand_layer_key,
    expand_layer_list,
    expand_layer_map,
    num_layer_keys,
    parse_layer_key,
)

pytestmark = pytest.mark.skip_global_cleanup

# Inkling 2 Mini: prelude 0-5, core 6-17 x 3 passes, coda 18-23.
MINI = LoopLayerSpace(num_layers=24, core_start=6, core_end=18, num_passes=3)


@pytest.fixture(autouse=True)
def _no_global_space(monkeypatch):
    monkeypatch.setattr(loop_layers, "_ACTIVE", None)
    monkeypatch.setattr(loop_layers, "_INSTALLED", True)


def test_mini_keys_prelude_core_coda():
    assert MINI.key(0, 0) == 0 and MINI.key(5, 0) == 5
    assert MINI.key(6, 0) == 30 and MINI.key(6, 1) == 54 and MINI.key(6, 2) == 78
    assert MINI.key(17, 2) == 89
    assert MINI.key(18, 0) == 18 and MINI.key(23, 0) == 23
    assert MINI.num_keys == 96
    assert MINI.keys(7) == [31, 55, 79]
    assert MINI.keys(7, 1) == [55]
    assert MINI.keys(20) == [20]


def test_visit_keys_follow_execution_order():
    keys = MINI.visit_keys()
    assert len(keys) == 48 == len(set(keys))
    assert keys[:6] == list(range(6))
    assert keys[6:18] == [MINI.key(i, 0) for i in range(6, 18)]
    assert keys[18:30] == [MINI.key(i, 1) for i in range(6, 18)]
    assert keys[30:42] == [MINI.key(i, 2) for i in range(6, 18)]
    assert keys[42:] == list(range(18, 24))
    for key in keys:
        assert MINI.is_key(key)
        layer, pass_idx = MINI.decode(key)
        assert MINI.key(layer, pass_idx) == key


def test_plain_core_indices_are_not_keys():
    # A plain core index means "all passes"; it never names one site.
    assert not MINI.is_key(7)
    assert not MINI.is_key(24) and not MINI.is_key(29) and not MINI.is_key(96)
    with pytest.raises(LoopLayerKeyError):
        MINI.decode(7)
    assert MINI.label(55) == "7@1" and MINI.label(20) == "20"


@pytest.mark.parametrize(
    "key,expected",
    [(7, (7, None)), ("7", (7, None)), ("7@2", (7, 2)), (" 7@0 ", (7, 0))],
)
def test_parse_layer_key(key, expected):
    assert parse_layer_key(key) == expected


@pytest.mark.parametrize("key", [-1, "x", "7@", "@1", "7@-1", 1.5, True, None, "7@1@2"])
def test_parse_layer_key_rejects(key):
    with pytest.raises(LoopLayerKeyError):
        parse_layer_key(key)


def test_expand_on_looped_model():
    assert expand_layer_key(7, space=MINI) == [31, 55, 79]
    assert expand_layer_key("7@1", space=MINI) == [55]
    assert expand_layer_key(7, 2, space=MINI) == [79]
    assert expand_layer_key(3, space=MINI) == [3]
    assert expand_layer_key("3@0", space=MINI) == [3]
    assert expand_layer_key(21, 0, space=MINI) == [21]


@pytest.mark.parametrize(
    "key,pass_idx,match",
    [
        ("7@3", None, "pass 3 out of range"),
        ("3@1", None, "outside the loop core"),
        (24, None, "out of range"),
        ("24@0", None, "out of range"),
        ("7@1", 1, "already names a pass"),
        (7, -1, "non-negative"),
    ],
)
def test_expand_rejects_bad_references(key, pass_idx, match):
    with pytest.raises(LoopLayerKeyError, match=match):
        expand_layer_key(key, pass_idx, space=MINI)


def test_expansion_is_idempotent():
    """Specs are re-validated in every process they cross: expanded keys
    expand to themselves."""
    once = expand_layer_list([7, "8@1", 2, 20], space=MINI)
    assert once == [31, 55, 79, 56, 2, 20]
    assert expand_layer_list(once, space=MINI) == once


def test_non_looped_model_rejects_a_pass():
    assert expand_layer_key(7) == [7]
    assert expand_layer_key("7") == [7]
    with pytest.raises(LoopLayerKeyError, match="needs a looped model"):
        expand_layer_key("7@0")
    with pytest.raises(LoopLayerKeyError, match="needs a looped model"):
        expand_layer_key(7, 1)


def test_expand_layer_map_merges_all_pass_and_pass_entries():
    merged = expand_layer_map(
        {7: [1.0], "7@1": [10.0], 2: [5.0]},
        combine=lambda a, b: [x + y for x, y in zip(a, b)],
        space=MINI,
    )
    assert merged == {31: [1.0], 55: [11.0], 79: [1.0], 2: [5.0]}


def test_installed_space_is_used_by_default(monkeypatch):
    monkeypatch.setattr(loop_layers, "_ACTIVE", MINI)
    assert expand_layer_key("7@2") == [79]
    loop_layers.install_loop_layer_space(None)
    with pytest.raises(LoopLayerKeyError):
        expand_layer_key("7@2")


def test_space_from_text_config_and_key_bound():
    loop = {"core_start": 6, "core_end": 18, "num_passes": 3}
    text = SimpleNamespace(num_hidden_layers=24, loop_config=loop, num_loop_passes=3)
    assert LoopLayerSpace.from_text_config(text) == MINI
    # A variable-loop checkpoint served at fewer passes.
    text.num_loop_passes = 2
    assert LoopLayerSpace.from_text_config(text).num_passes == 2
    assert (
        LoopLayerSpace.from_text_config(SimpleNamespace(num_hidden_layers=24)) is None
    )
    looped = SimpleNamespace(
        hf_text_config=SimpleNamespace(**{**vars(text), "num_loop_passes": 3})
    )
    assert num_layer_keys(looped) == 96
    flat = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_hidden_layers=24),
        get_total_num_hidden_layers=lambda: 24,
    )
    assert num_layer_keys(flat) == 24


# --------------------------------------------------------------------------
# Ingestion seams: vectors, packed vectors, clamps, patches, SAE specs
# --------------------------------------------------------------------------


@pytest.fixture
def mini(monkeypatch):
    monkeypatch.setattr(loop_layers, "_ACTIVE", MINI)
    return MINI


def _vec(value, width=4):
    return [float(value)] * width


def test_sampling_params_steering_targets_passes(mini):
    from vllm.sampling_params import SamplingParams

    params = SamplingParams(
        steering_vectors={
            "post_block": {"7": _vec(1), "7@1": _vec(10), 20: _vec(5)},
        },
        decode_steering_vectors={
            "pre_attn": {"8@2": {"vector": _vec(2), "scale": 3.0}}
        },
    )
    base = params.steering_vectors["post_block"]
    assert set(base) == {31, 55, 79, 20}
    assert list(base[31]) == _vec(1) and list(base[79]) == _vec(1)
    # "7" (every pass) and "7@1" add at pass 1.
    assert list(base[55]) == _vec(11)
    assert list(params.decode_steering_vectors["pre_attn"]) == [80]
    # Re-validation (another process's decode) keeps the keys.
    params._validate_steering_vectors()
    assert set(params.steering_vectors["post_block"]) == {31, 55, 79, 20}


def test_sampling_params_rejects_bad_passes(mini):
    from vllm.sampling_params import SamplingParams

    with pytest.raises(ValueError, match="outside the loop core"):
        SamplingParams(steering_vectors={"post_block": {"3@1": _vec(1)}})
    with pytest.raises(ValueError, match="out of range"):
        SamplingParams(steering_vectors={"post_block": {"7@3": _vec(1)}})


def test_non_looped_model_keeps_indices_and_rejects_passes():
    from vllm.sampling_params import SamplingParams

    params = SamplingParams(steering_vectors={"post_block": {7: _vec(1)}})
    assert list(params.steering_vectors["post_block"]) == [7]
    with pytest.raises(ValueError, match="needs a looped model"):
        SamplingParams(steering_vectors={"post_block": {"7@0": _vec(1)}})


def test_packed_vectors_accept_pass_indices(mini):
    import base64

    import numpy as np

    from vllm.config.steering_types import unpack_steering_vectors

    rows = np.array([[1.0] * 4, [10.0] * 4], dtype=np.float32)
    packed = {
        "post_attn": {
            "dtype": "float32",
            "shape": [2, 4],
            "layer_indices": [7, "7@2"],
            "data": base64.b64encode(rows.tobytes()).decode(),
        }
    }
    out = unpack_steering_vectors(packed)["post_attn"]
    assert set(out) == {31, 55, 79}
    assert out[79].tolist() == [11.0] * 4 and out[31].tolist() == [1.0] * 4


def test_clamps_target_passes(mini):
    import base64

    import numpy as np

    from vllm.config.steering_types import SteeringClamps

    entry = {"vector": _vec(1), "max": 1.0}
    clamps = SteeringClamps.from_obj({"post_attn": {"7@2": [entry], "20": [entry]}})
    assert clamps.hooks["post_attn"].layer_indices == [20, 79]
    every = SteeringClamps.from_obj({"post_attn": {"7": [entry]}})
    assert every.hooks["post_attn"].layer_indices == [31, 55, 79]
    rows = np.ones((1, 4), dtype=np.float64)
    packed = SteeringClamps.from_obj(
        {
            "post_attn": {
                "dtype": "float64",
                "shape": [1, 4],
                "layer_indices": [7],
                "data": base64.b64encode(rows.tobytes()).decode(),
                "bounds": [[None, 1.0]],
                "strengths": [1.0],
            }
        }
    )
    table = packed.hooks["post_attn"]
    assert table.layer_indices == [31, 55, 79] and table.shape == [3, 4]
    assert table.hi == [1.0] * 3
    with pytest.raises(ValueError, match="invalid layer index"):
        SteeringClamps.from_obj({"post_attn": {"3@2": [entry]}})


def test_patch_entries_target_passes(mini):
    from vllm.sampling_params import SamplingParams

    entry = {"hook": "post_block", "dest_position": 0, "source_module": "zeros"}
    params = SamplingParams(
        patch=[
            {**entry, "layer": 7, "pass": 1},
            {**entry, "layer": "8@0"},
            {**entry, "layer": 9},
            {**entry, "layer": 2},
        ]
    )
    assert [e["layer"] for e in params.patch] == [55, 32, 33, 57, 81, 2]
    assert all("pass" not in e for e in params.patch)
    params._validate_patch()
    assert [e["layer"] for e in params.patch] == [55, 32, 33, 57, 81, 2]
    with pytest.raises(ValueError, match="already names a pass"):
        SamplingParams(patch=[{**entry, "layer": "7@1", "pass": 1}])


def test_sae_clamp_specs_target_passes(mini):
    from vllm.config.sae_steering_types import coerce_sae_clamp_specs

    feature = {"feature_idx": 3, "kind": "absolute", "value": 1.0}
    other = {"feature_idx": 4, "kind": "absolute", "value": 1.0}
    (spec,) = coerce_sae_clamp_specs(
        [
            {
                "module_name": "m",
                "clamps": {"post_block": {"7@1": [feature], "7": [other]}},
            }
        ]
    )
    layers = spec.clamps["post_block"]
    assert set(layers) == {31, 55, 79}
    assert [e.feature_idx for e in layers[55]] == [3, 4]
    with pytest.raises(ValueError, match="duplicate layer key"):
        coerce_sae_clamp_specs(
            [
                {
                    "module_name": "m",
                    "clamps": {"post_block": {"7@1": [feature], "7@01": [other]}},
                }
            ]
        )
