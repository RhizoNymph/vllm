# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.config.sae_steering_types import SAEClampEntry, SAEClampSpec
from vllm.config.steering_types import (
    maybe_pack_inline_steering_for_request,
    pack_effective_steering,
    pack_steering_for_dtype,
)


def test_pack_steering_for_dtype_prescales_entries():
    out = pack_steering_for_dtype(
        {"post_mlp": {0: {"vector": [1.0, 2.0], "scale": 2.0}}},
        np.float32,
    )

    assert out is not None
    assert out["post_mlp"][0].dtype == np.float32
    assert out["post_mlp"][0].tolist() == pytest.approx([2.0, 4.0])


def test_pack_effective_steering_resolves_phase_sum():
    out = pack_effective_steering(
        {"post_mlp": {0: [1.0, 2.0]}},
        {"post_mlp": {0: [3.0, 4.0]}},
        np.float16,
    )

    assert out is not None
    assert out["post_mlp"][0].dtype == np.float16
    assert out["post_mlp"][0].tolist() == pytest.approx([4.0, 6.0])


def test_maybe_pack_inline_steering_clears_inline_fields_and_preserves_hash():
    sp = SamplingParams(
        steering_vectors={"post_mlp": {0: [1.0, 2.0]}},
        prefill_steering_vectors={"post_mlp": {0: [3.0, 4.0]}},
    )
    prefill_hash = sp.prefill_steering_config_hash
    decode_hash = sp.decode_steering_config_hash
    prefill_additive_hash = sp.prefill_additive_steering_config_hash
    decode_additive_hash = sp.decode_additive_steering_config_hash

    maybe_pack_inline_steering_for_request(sp, torch.float32)

    assert sp.steering_vectors is None
    assert sp.prefill_steering_vectors is None
    assert sp.decode_steering_vectors is None
    assert sp._effective_prefill_steering_packed is not None
    assert sp._effective_decode_steering_packed is not None
    assert sp.effective_prefill_steering is sp._effective_prefill_steering_packed
    assert sp.prefill_steering_config_hash == prefill_hash
    assert sp.decode_steering_config_hash == decode_hash
    assert sp.prefill_additive_steering_config_hash == prefill_additive_hash
    assert sp.decode_additive_steering_config_hash == decode_additive_hash


def test_maybe_pack_inline_steering_preserves_additive_hash_before_fp16_cast():
    sp = SamplingParams(steering_vectors={"post_mlp": {0: [0.1, 0.2, 0.3]}})
    prefill_additive_hash = sp.prefill_additive_steering_config_hash
    decode_additive_hash = sp.decode_additive_steering_config_hash

    maybe_pack_inline_steering_for_request(sp, torch.float16)

    assert sp._effective_prefill_steering_packed is not None
    assert sp._effective_prefill_steering_packed["post_mlp"][0].dtype == np.float16
    assert sp.prefill_additive_steering_config_hash == prefill_additive_hash
    assert sp.decode_additive_steering_config_hash == decode_additive_hash


def test_maybe_pack_inline_steering_preserves_sae_hash_contribution():
    sae_spec = SAEClampSpec(
        module_name="g",
        clamps={
            "post_mlp": {
                0: (
                    SAEClampEntry(
                        feature_idx=1,
                        kind="absolute",
                        value=2.0,
                    ),
                )
            }
        },
    )
    sp = SamplingParams(
        steering_vectors={"post_mlp": {0: [1.0, 2.0]}},
        sae_clamp_specs=(sae_spec,),
    )
    prefill_hash = sp.prefill_steering_config_hash
    decode_hash = sp.decode_steering_config_hash
    prefill_additive_hash = sp.prefill_additive_steering_config_hash
    decode_additive_hash = sp.decode_additive_steering_config_hash
    prefill_sae_hash = sp.prefill_sae_clamp_config_hash
    decode_sae_hash = sp.decode_sae_clamp_config_hash

    maybe_pack_inline_steering_for_request(sp, torch.float32)

    assert sp.prefill_steering_config_hash == prefill_hash
    assert sp.decode_steering_config_hash == decode_hash
    assert sp.prefill_additive_steering_config_hash == prefill_additive_hash
    assert sp.decode_additive_steering_config_hash == decode_additive_hash
    assert sp.prefill_sae_clamp_config_hash == prefill_sae_hash
    assert sp.decode_sae_clamp_config_hash == decode_sae_hash
    assert sp.sae_clamp_specs == (sae_spec,)


def test_maybe_pack_inline_steering_skips_named_module_overrides():
    sp = SamplingParams(
        steering_module_ref=("named", 1.0),
        steering_vectors={"post_mlp": {0: [1.0, 2.0]}},
        prefill_steering_vectors={"post_mlp": {0: [3.0, 4.0]}},
    )

    maybe_pack_inline_steering_for_request(sp, torch.float32)

    assert sp.steering_module_ref == ("named", 1.0)
    assert sp.steering_vectors == {"post_mlp": {0: [1.0, 2.0]}}
    assert sp.prefill_steering_vectors == {"post_mlp": {0: [3.0, 4.0]}}
    assert sp._effective_prefill_steering_packed is None
    assert sp._effective_decode_steering_packed is None
