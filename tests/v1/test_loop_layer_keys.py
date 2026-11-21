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
