# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for per-request MLP neuron clamp types, validation and hashing."""

import copy
import math
import pickle

import numpy as np
import pytest

from vllm.config.neuron_clamp_types import (
    NeuronClamp,
    NeuronClampValidationError,
    coerce_neuron_clamps,
    effective_clamps_for_phase,
    encode_clamps_for_hash,
    global_to_local_neuron,
    make_neuron_clamp,
    validate_neuron_clamps_for_model,
)
from vllm.config.steering_types import hash_steering_config
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.cpu_test


def _c(layer=9, neuron=3757, mode="set", value=0.0, phase="both"):
    return {
        "layer": layer,
        "neuron": neuron,
        "mode": mode,
        "value": value,
        "phase": phase,
    }


# ---------------------------------------------------------------------------
# NeuronClamp construction
# ---------------------------------------------------------------------------


class TestNeuronClamp:
    def test_valid_defaults(self):
        c = NeuronClamp(1, 2, "add", 3)
        assert (c.layer, c.neuron, c.mode, c.value, c.phase) == (
            1,
            2,
            "add",
            3.0,
            "both",
        )
        assert isinstance(c.value, float)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(layer=-1, neuron=0, mode="set", value=0.0),
            dict(layer=True, neuron=0, mode="set", value=0.0),
            dict(layer=1.0, neuron=0, mode="set", value=0.0),
            dict(layer=0, neuron=-3, mode="set", value=0.0),
            dict(layer=0, neuron="4", mode="set", value=0.0),
            dict(layer=0, neuron=0, mode="zero", value=0.0),
            dict(layer=0, neuron=0, mode="set", value=math.nan),
            dict(layer=0, neuron=0, mode="set", value=math.inf),
            dict(layer=0, neuron=0, mode="set", value="1"),
            dict(layer=0, neuron=0, mode="set", value=False),
            dict(layer=0, neuron=0, mode="set", value=0.0, phase="all"),
        ],
    )
    def test_invalid_rejected(self, kwargs):
        with pytest.raises(NeuronClampValidationError):
            NeuronClamp(**kwargs)

    def test_error_is_typed_validation_error(self):
        with pytest.raises(VLLMValidationError) as ei:
            NeuronClamp(0, 0, "nope", 0.0)
        assert ei.value.parameter == "neuron_clamps"
        assert isinstance(ei.value, ValueError)

    def test_immutable(self):
        c = NeuronClamp(0, 0, "set", 0.0)
        with pytest.raises(AttributeError):
            c.value = 1.0  # type: ignore[misc]

    def test_eq_hash_pickle_copy(self):
        a = NeuronClamp(1, 2, "scale", 0.5, "decode")
        b = NeuronClamp(1, 2, "scale", 0.5, "decode")
        assert a == b and hash(a) == hash(b)
        assert pickle.loads(pickle.dumps(a)) == a
        assert copy.deepcopy(a) == a
        assert a != NeuronClamp(1, 2, "scale", 0.5, "prefill")

    @pytest.mark.parametrize(
        "mode,value,expected",
        [
            ("set", 4.0, (0.0, 4.0)),
            ("add", 4.0, (1.0, 4.0)),
            ("scale", 4.0, (4.0, 0.0)),
        ],
    )
    def test_affine(self, mode, value, expected):
        assert NeuronClamp(0, 0, mode, value).affine() == expected

    def test_applies_to(self):
        assert NeuronClamp(0, 0, "set", 0, "both").applies_to("prefill")
        assert NeuronClamp(0, 0, "set", 0, "both").applies_to("decode")
        assert NeuronClamp(0, 0, "set", 0, "prefill").applies_to("prefill")
        assert not NeuronClamp(0, 0, "set", 0, "prefill").applies_to("decode")
        assert not NeuronClamp(0, 0, "set", 0, "decode").applies_to("prefill")

    def test_dict_roundtrip(self):
        c = NeuronClamp(3, 7, "add", -1.5, "prefill")
        assert make_neuron_clamp(c.to_dict()) == c


class TestMakeNeuronClamp:
    def test_phase_defaults_to_both(self):
        d = _c()
        del d["phase"]
        assert make_neuron_clamp(d).phase == "both"

    def test_extra_key_rejected(self):
        with pytest.raises(NeuronClampValidationError, match="unexpected"):
            make_neuron_clamp({**_c(), "foo": 1})

    @pytest.mark.parametrize("missing", ["layer", "neuron", "mode", "value"])
    def test_missing_key_rejected(self, missing):
        d = _c()
        del d[missing]
        with pytest.raises(NeuronClampValidationError, match="missing"):
            make_neuron_clamp(d)

    def test_non_dict_rejected(self):
        with pytest.raises(NeuronClampValidationError):
            make_neuron_clamp([9, 3757, "set", 0.0])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# coerce / duplicates / phases
# ---------------------------------------------------------------------------


class TestCoerce:
    def test_none_and_empty(self):
        assert coerce_neuron_clamps(None) is None
        assert coerce_neuron_clamps([]) is None

    def test_not_a_list(self):
        with pytest.raises(NeuronClampValidationError):
            coerce_neuron_clamps(_c())  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "p1,p2",
        [
            ("both", "both"),
            ("both", "prefill"),
            ("decode", "both"),
            ("prefill", "prefill"),
            ("decode", "decode"),
        ],
    )
    def test_overlapping_duplicates_rejected(self, p1, p2):
        with pytest.raises(NeuronClampValidationError, match="duplicate"):
            coerce_neuron_clamps([_c(phase=p1), _c(phase=p2, value=1.0)])

    def test_disjoint_phases_allowed(self):
        out = coerce_neuron_clamps(
            [_c(phase="prefill", value=1.0), _c(phase="decode", value=2.0)]
        )
        assert out is not None and len(out) == 2

    def test_same_neuron_different_layer_allowed(self):
        out = coerce_neuron_clamps([_c(layer=8), _c(layer=9)])
        assert out is not None and len(out) == 2


class TestEffectivePerPhase:
    def test_split(self):
        clamps = coerce_neuron_clamps(
            [
                _c(layer=9, neuron=5, phase="both"),
                _c(layer=8, neuron=1, phase="prefill", value=1.0),
                _c(layer=18, neuron=2, phase="decode", value=2.0),
            ]
        )
        pre = effective_clamps_for_phase(clamps, "prefill")
        dec = effective_clamps_for_phase(clamps, "decode")
        assert [(c.layer, c.neuron) for c in pre] == [(8, 1), (9, 5)]
        assert [(c.layer, c.neuron) for c in dec] == [(9, 5), (18, 2)]

    def test_none_when_no_phase_match(self):
        clamps = coerce_neuron_clamps([_c(phase="prefill")])
        assert effective_clamps_for_phase(clamps, "decode") is None
        assert effective_clamps_for_phase(None, "prefill") is None

    def test_canonical_order(self):
        a = coerce_neuron_clamps([_c(layer=9, neuron=2), _c(layer=8, neuron=7)])
        b = coerce_neuron_clamps([_c(layer=8, neuron=7), _c(layer=9, neuron=2)])
        assert effective_clamps_for_phase(a, "decode") == effective_clamps_for_phase(
            b, "decode"
        )


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------

_VEC = {"post_block": {9: np.arange(4, dtype=np.float64)}}


class TestHash:
    def test_no_clamps_unchanged(self):
        assert hash_steering_config(None) == 0
        assert hash_steering_config(None, neuron_clamps=None) == 0
        assert hash_steering_config(None, neuron_clamps=()) == 0
        assert hash_steering_config(_VEC) == hash_steering_config(
            _VEC, neuron_clamps=None
        )

    def test_clamp_only_nonzero(self):
        clamps = (NeuronClamp(9, 3757, "set", 0.0),)
        assert hash_steering_config(None, neuron_clamps=clamps) != 0

    def test_clamps_change_hash(self):
        clamps = (NeuronClamp(9, 3757, "set", 0.0),)
        assert hash_steering_config(_VEC, neuron_clamps=clamps) != (
            hash_steering_config(_VEC)
        )

    @pytest.mark.parametrize(
        "other",
        [
            NeuronClamp(9, 3757, "set", 1.0),
            NeuronClamp(9, 3756, "set", 0.0),
            NeuronClamp(8, 3757, "set", 0.0),
            NeuronClamp(9, 3757, "scale", 0.0),
            NeuronClamp(9, 3757, "add", 0.0),
        ],
    )
    def test_every_field_matters(self, other):
        base = (NeuronClamp(9, 3757, "set", 0.0),)
        assert hash_steering_config(None, neuron_clamps=base) != (
            hash_steering_config(None, neuron_clamps=(other,))
        )

    def test_order_independent(self):
        a = NeuronClamp(9, 1, "set", 0.0)
        b = NeuronClamp(8, 2, "add", 1.0)
        assert encode_clamps_for_hash([a, b]) == encode_clamps_for_hash([b, a])

    def test_module_ref_and_clamps_distinct(self):
        clamps = (NeuronClamp(9, 3757, "set", 0.0),)
        h1 = hash_steering_config(None, module_ref=("m", 1.0))
        h2 = hash_steering_config(None, module_ref=("m", 1.0), neuron_clamps=clamps)
        assert h1 != h2 and h2 != 0


# ---------------------------------------------------------------------------
# SamplingParams integration
# ---------------------------------------------------------------------------


class TestSamplingParams:
    def test_default_none(self):
        sp = SamplingParams()
        assert sp.neuron_clamps is None
        assert sp.parsed_neuron_clamps is None
        assert sp.prefill_steering_config_hash == 0
        assert sp.decode_steering_config_hash == 0

    def test_empty_list_normalized(self):
        sp = SamplingParams(neuron_clamps=[])
        assert sp.neuron_clamps is None
        assert sp.prefill_steering_config_hash == 0

    def test_clamp_only_hash_nonzero_both_phases(self):
        sp = SamplingParams(neuron_clamps=[_c()])
        assert sp.prefill_steering_config_hash != 0
        assert sp.decode_steering_config_hash != 0

    def test_normalizes_to_wire_dicts(self):
        sp = SamplingParams(
            neuron_clamps=[
                NeuronClamp(9, 3757, "set", 0),
                {"layer": 8, "neuron": 1931, "mode": "scale", "value": 2},
            ]
        )
        assert sp.neuron_clamps == [
            {"layer": 9, "neuron": 3757, "mode": "set", "value": 0.0, "phase": "both"},
            {
                "layer": 8,
                "neuron": 1931,
                "mode": "scale",
                "value": 2.0,
                "phase": "both",
            },
        ]

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError):
            SamplingParams(neuron_clamps=[_c(mode="zero")])
        with pytest.raises(NeuronClampValidationError):
            SamplingParams(neuron_clamps=[_c(), _c(value=3.0)])

    def test_same_clamps_same_hash(self):
        a = SamplingParams(neuron_clamps=[_c(layer=9), _c(layer=8, neuron=1)])
        b = SamplingParams(neuron_clamps=[_c(layer=8, neuron=1), _c(layer=9)])
        assert a.prefill_steering_config_hash == b.prefill_steering_config_hash
        assert a.decode_steering_config_hash == b.decode_steering_config_hash

    def test_different_clamps_different_hash(self):
        a = SamplingParams(neuron_clamps=[_c(value=0.0)])
        b = SamplingParams(neuron_clamps=[_c(value=-30.0)])
        assert a.prefill_steering_config_hash != b.prefill_steering_config_hash

    def test_vectors_plus_clamps_combine(self):
        vec = {"post_block": {9: [1.0, 2.0, 3.0]}}
        v = SamplingParams(steering_vectors=vec)
        c = SamplingParams(neuron_clamps=[_c()])
        vc = SamplingParams(steering_vectors=vec, neuron_clamps=[_c()])
        hashes = {
            v.prefill_steering_config_hash,
            c.prefill_steering_config_hash,
            vc.prefill_steering_config_hash,
        }
        assert len(hashes) == 3 and 0 not in hashes
        assert vc.prefill_steering_config_hash == hash_steering_config(
            vc.effective_prefill_steering,
            neuron_clamps=(NeuronClamp(9, 3757, "set", 0.0),),
        )

    def test_vector_only_hash_unchanged(self):
        vec = {"post_block": {9: [1.0, 2.0, 3.0]}}
        sp = SamplingParams(steering_vectors=vec)
        assert sp.prefill_steering_config_hash == hash_steering_config(
            sp.effective_prefill_steering
        )

    def test_phase_specific_hashes(self):
        sp = SamplingParams(neuron_clamps=[_c(phase="prefill")])
        assert sp.prefill_steering_config_hash != 0
        assert sp.decode_steering_config_hash == 0
        sp = SamplingParams(neuron_clamps=[_c(phase="decode")])
        assert sp.prefill_steering_config_hash == 0
        assert sp.decode_steering_config_hash != 0

    def test_prefill_clamp_with_decode_vectors(self):
        vec = {"post_block": {9: [1.0, 2.0, 3.0]}}
        sp = SamplingParams(
            decode_steering_vectors=vec, neuron_clamps=[_c(phase="prefill")]
        )
        ref = SamplingParams(decode_steering_vectors=vec)
        assert sp.decode_steering_config_hash == ref.decode_steering_config_hash
        assert sp.prefill_steering_config_hash != 0

    def test_msgpack_roundtrip(self):
        sp = SamplingParams(
            neuron_clamps=[
                _c(),
                _c(layer=18, neuron=7807, mode="add", value=1.25, phase="decode"),
            ]
        )
        out = MsgpackDecoder(SamplingParams).decode(MsgpackEncoder().encode(sp))
        assert out.neuron_clamps == sp.neuron_clamps
        assert out.parsed_neuron_clamps == sp.parsed_neuron_clamps
        assert out.prefill_steering_config_hash == sp.prefill_steering_config_hash
        assert out.decode_steering_config_hash == sp.decode_steering_config_hash

    def test_clone_preserves(self):
        sp = SamplingParams(neuron_clamps=[_c()])
        h = sp.prefill_steering_config_hash
        cl = sp.clone()
        assert cl.neuron_clamps == sp.neuron_clamps
        assert cl.prefill_steering_config_hash == h
        assert cl.effective_decode_neuron_clamps == sp.effective_decode_neuron_clamps

    def test_from_optional(self):
        sp = SamplingParams.from_optional(neuron_clamps=[_c()])
        assert sp.parsed_neuron_clamps == (NeuronClamp(9, 3757, "set", 0.0),)


# ---------------------------------------------------------------------------
# model-range validation
# ---------------------------------------------------------------------------


def _validate(clamps, *, layers=36, inter=12288, k=16):
    validate_neuron_clamps_for_model(
        coerce_neuron_clamps(clamps),
        num_hidden_layers=layers,
        intermediate_size=inter,
        max_clamps_per_layer=k,
    )


class TestModelValidation:
    def test_ok(self):
        _validate([_c(layer=35, neuron=12287)])

    def test_layer_out_of_range(self):
        with pytest.raises(NeuronClampValidationError, match="layer 36"):
            _validate([_c(layer=36)])

    def test_neuron_out_of_range(self):
        with pytest.raises(NeuronClampValidationError, match="neuron 12288"):
            _validate([_c(neuron=12288)])

    def test_disabled(self):
        with pytest.raises(NeuronClampValidationError, match="disabled"):
            _validate([_c()], k=0)

    def test_per_layer_limit_per_phase(self):
        clamps = [_c(neuron=i) for i in range(3)]
        _validate(clamps, k=3)
        with pytest.raises(NeuronClampValidationError, match="exceed"):
            _validate(clamps, k=2)

    def test_disjoint_phases_counted_separately(self):
        clamps = [_c(neuron=0, phase="prefill"), _c(neuron=1, phase="decode")]
        _validate(clamps, k=1)
        with pytest.raises(NeuronClampValidationError):
            _validate(clamps + [_c(neuron=2, phase="both")], k=1)


# ---------------------------------------------------------------------------
# TP index mapping
# ---------------------------------------------------------------------------


class TestGlobalToLocal:
    def test_tp1_identity(self):
        assert global_to_local_neuron(3757, tp_rank=0, local_size=12288) == 3757
        assert global_to_local_neuron(12288, tp_rank=0, local_size=12288) is None

    def test_tp2_partition(self):
        local = 12288 // 2
        for g in (0, 1931, 3757, 6143, 6144, 7807, 12287):
            owners = [
                (r, global_to_local_neuron(g, tp_rank=r, local_size=local))
                for r in range(2)
            ]
            owned = [(r, li) for r, li in owners if li is not None]
            # Exactly one rank owns each neuron, at the expected offset.
            assert len(owned) == 1
            r, li = owned[0]
            assert r == g // local and li == g - r * local
        assert global_to_local_neuron(3757, tp_rank=0, local_size=local) == 3757
        assert global_to_local_neuron(3757, tp_rank=1, local_size=local) is None
        assert global_to_local_neuron(7807, tp_rank=0, local_size=local) is None
        assert global_to_local_neuron(7807, tp_rank=1, local_size=local) == 1663

    def test_tp2_matches_row_parallel_shard(self):
        # down_proj (RowParallel) rank r holds weight[:, r*L:(r+1)*L];
        # the local activation column j must multiply global column r*L+j.
        full = np.arange(12)
        local = 6
        for g in range(12):
            for r in range(2):
                li = global_to_local_neuron(g, tp_rank=r, local_size=local)
                if li is not None:
                    assert full[r * local : (r + 1) * local][li] == g
