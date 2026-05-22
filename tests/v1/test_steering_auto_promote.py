# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle

import pytest

from vllm import SamplingParams
from vllm.config.sae_steering_types import SAEClampEntry, SAEClampSpec
from vllm.config.steering_types import (
    SteeringAutoPromoteLRU,
    maybe_auto_promote_steering_modules,
    maybe_auto_promote_steering_modules_async,
)


def _recording_rpc(calls):
    def rpc(method, **kwargs):
        calls.append((method, kwargs))

    return rpc


def test_lru_two_strikes_then_registered_hit():
    lru = SteeringAutoPromoteLRU(capacity=4)
    key = (1, 2)

    assert lru.observe(key) == ("first", None, None)
    assert lru.observe(key) == ("second", None, None)
    lru.mark_registered(key, "_auto")
    assert lru.observe(key) == ("registered", "_auto", None)
    assert lru.get(key) == "_auto"


def test_auto_promote_first_sighting_does_not_rpc():
    calls = []
    lru = SteeringAutoPromoteLRU(capacity=4)
    sp = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})

    maybe_auto_promote_steering_modules(sp, _recording_rpc(calls), lru)

    assert calls == []
    assert sp.steering_module_ref is None


def test_auto_promote_second_sighting_registers_and_rewrites_request():
    calls = []
    lru = SteeringAutoPromoteLRU(capacity=4)
    sp_a = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})
    sp_b = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})
    original_hashes = (
        sp_b.prefill_steering_config_hash,
        sp_b.decode_steering_config_hash,
    )

    maybe_auto_promote_steering_modules(sp_a, _recording_rpc(calls), lru)
    maybe_auto_promote_steering_modules(sp_b, _recording_rpc(calls), lru)

    assert len(calls) == 1
    method, kwargs = calls[0]
    assert method == "register_steering_modules"
    assert kwargs["kwargs"]["replace"] is False
    name = next(iter(kwargs["kwargs"]["modules"]))
    assert sp_b.steering_module_ref == (name, 1.0)
    assert sp_b.steering_vectors is None
    assert sp_b.prefill_steering_config_hash == original_hashes[0]
    assert sp_b.decode_steering_config_hash == original_hashes[1]


def test_auto_promote_register_failure_does_not_poison_lru():
    calls = []
    lru = SteeringAutoPromoteLRU(capacity=4)
    sp_a = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})
    sp_b = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})
    sp_c = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})

    maybe_auto_promote_steering_modules(sp_a, _recording_rpc(calls), lru)

    def failing_rpc(method, **kwargs):
        calls.append((method, kwargs))
        raise RuntimeError("broadcast failed")

    with pytest.raises(RuntimeError, match="broadcast failed"):
        maybe_auto_promote_steering_modules(sp_b, failing_rpc, lru)
    assert sp_b.steering_module_ref is None

    calls.clear()
    maybe_auto_promote_steering_modules(sp_c, _recording_rpc(calls), lru)
    assert len(calls) == 1
    assert calls[0][0] == "register_steering_modules"
    assert sp_c.steering_module_ref is not None


@pytest.mark.asyncio
async def test_auto_promote_async_register_failure_does_not_poison_lru():
    calls = []
    lru = SteeringAutoPromoteLRU(capacity=4)
    sp_a = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})
    sp_b = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})
    sp_c = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})

    async def recording_rpc(method, **kwargs):
        calls.append((method, kwargs))

    async def failing_rpc(method, **kwargs):
        calls.append((method, kwargs))
        raise RuntimeError("broadcast failed")

    await maybe_auto_promote_steering_modules_async(sp_a, recording_rpc, lru)
    with pytest.raises(RuntimeError, match="broadcast failed"):
        await maybe_auto_promote_steering_modules_async(sp_b, failing_rpc, lru)
    assert sp_b.steering_module_ref is None

    calls.clear()
    await maybe_auto_promote_steering_modules_async(sp_c, recording_rpc, lru)
    assert len(calls) == 1
    assert calls[0][0] == "register_steering_modules"
    assert sp_c.steering_module_ref is not None


def test_auto_promote_registered_hit_rewrites_without_rpc():
    calls = []
    lru = SteeringAutoPromoteLRU(capacity=4)
    sp_a = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})
    sp_b = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})
    sp_c = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})

    maybe_auto_promote_steering_modules(sp_a, _recording_rpc(calls), lru)
    maybe_auto_promote_steering_modules(sp_b, _recording_rpc(calls), lru)
    calls.clear()
    maybe_auto_promote_steering_modules(sp_c, _recording_rpc(calls), lru)

    assert calls == []
    assert sp_c.steering_module_ref == sp_b.steering_module_ref


def test_auto_promote_first_sighting_does_not_evict_registered_module():
    calls = []
    lru = SteeringAutoPromoteLRU(capacity=1)
    sp_a = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})
    sp_b = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})

    maybe_auto_promote_steering_modules(sp_a, _recording_rpc(calls), lru)
    maybe_auto_promote_steering_modules(sp_b, _recording_rpc(calls), lru)
    registered_ref = sp_b.steering_module_ref
    assert registered_ref is not None
    calls.clear()

    sp_one_off = SamplingParams(steering_vectors={"post_mlp": {0: [9.0, 10.0]}})
    maybe_auto_promote_steering_modules(sp_one_off, _recording_rpc(calls), lru)

    assert calls == []
    assert sp_one_off.steering_module_ref is None
    registered_key = (
        sp_a.prefill_additive_steering_config_hash,
        sp_a.decode_additive_steering_config_hash,
    )
    one_off_key = (
        sp_one_off.prefill_additive_steering_config_hash,
        sp_one_off.decode_additive_steering_config_hash,
    )
    assert lru.get(registered_key) == registered_ref[0]
    assert one_off_key not in lru


def test_lru_first_sighting_evicts_only_pending_entries():
    lru = SteeringAutoPromoteLRU(capacity=1)

    assert lru.observe((1, 1)) == ("first", None, None)
    assert (1, 1) in lru
    assert lru.observe((2, 2)) == ("first", None, ((1, 1), None))
    assert (1, 1) not in lru
    assert (2, 2) in lru


def test_auto_promote_reused_params_after_fp16_pack_hits_same_lru_key():
    """Offline LLM can reuse and mutate one SamplingParams across prompts."""
    from vllm.config.steering_types import maybe_pack_inline_steering_for_request

    calls = []
    lru = SteeringAutoPromoteLRU(capacity=4)
    sp = SamplingParams(steering_vectors={"post_mlp": {0: [0.1, 0.2]}})
    original_hashes = (
        sp.prefill_steering_config_hash,
        sp.decode_steering_config_hash,
    )

    maybe_auto_promote_steering_modules(sp, _recording_rpc(calls), lru)
    maybe_pack_inline_steering_for_request(sp, "float16")
    maybe_auto_promote_steering_modules(sp, _recording_rpc(calls), lru)

    assert len(calls) == 1
    assert calls[0][0] == "register_steering_modules"
    assert sp.steering_module_ref is not None
    assert sp.prefill_steering_config_hash == original_hashes[0]
    assert sp.decode_steering_config_hash == original_hashes[1]


def test_auto_promote_preserves_hash_after_pickle_roundtrip():
    calls = []
    lru = SteeringAutoPromoteLRU(capacity=4)
    sp_a = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})
    sp_b = SamplingParams(steering_vectors={"post_mlp": {0: [1.0, 2.0]}})
    original_hashes = (
        sp_b.prefill_steering_config_hash,
        sp_b.decode_steering_config_hash,
    )

    maybe_auto_promote_steering_modules(sp_a, _recording_rpc(calls), lru)
    maybe_auto_promote_steering_modules(sp_b, _recording_rpc(calls), lru)
    restored = pickle.loads(pickle.dumps(sp_b))

    assert restored.steering_module_ref == sp_b.steering_module_ref
    assert restored.prefill_steering_config_hash == original_hashes[0]
    assert restored.decode_steering_config_hash == original_hashes[1]


def test_auto_promote_key_deduplicates_additive_payload_with_sae_clamps():
    calls = []
    lru = SteeringAutoPromoteLRU(capacity=4)
    sae_spec_a = SAEClampSpec(
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
    sae_spec_b = SAEClampSpec(
        module_name="g",
        clamps={
            "post_mlp": {
                0: (
                    SAEClampEntry(
                        feature_idx=2,
                        kind="absolute",
                        value=3.0,
                    ),
                )
            }
        },
    )
    inline = {"post_mlp": {0: [1.0, 2.0]}}
    sp_sae_a = SamplingParams(steering_vectors=inline, sae_clamp_specs=(sae_spec_a,))
    sp_sae_b = SamplingParams(steering_vectors=inline, sae_clamp_specs=(sae_spec_b,))
    sae_b_original_hash = sp_sae_b.prefill_steering_config_hash

    rpc = _recording_rpc(calls)
    maybe_auto_promote_steering_modules(sp_sae_a, rpc, lru)
    maybe_auto_promote_steering_modules(sp_sae_b, rpc, lru)
    sae_b_ref = sp_sae_b.steering_module_ref
    sae_b_prefill_hash = sp_sae_b.prefill_steering_config_hash

    sp_sae_c = SamplingParams(steering_vectors=inline, sae_clamp_specs=(sae_spec_a,))
    maybe_auto_promote_steering_modules(sp_sae_c, rpc, lru)

    registered = [
        call for call in calls if call[0] == "register_steering_modules"
    ]
    assert len(registered) == 1
    assert sp_sae_b.sae_clamp_specs == (sae_spec_b,)
    assert sae_b_prefill_hash == sae_b_original_hash
    assert sp_sae_c.sae_clamp_specs == (sae_spec_a,)
    assert sp_sae_c.steering_module_ref == sae_b_ref
    assert sp_sae_c.prefill_steering_config_hash != sae_b_prefill_hash


def test_auto_promote_preserves_sae_hash_after_pickle_roundtrip():
    calls = []
    lru = SteeringAutoPromoteLRU(capacity=4)
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
    inline = {"post_mlp": {0: [1.0, 2.0]}}
    sp_a = SamplingParams(steering_vectors=inline, sae_clamp_specs=(sae_spec,))
    sp_b = SamplingParams(steering_vectors=inline, sae_clamp_specs=(sae_spec,))
    original_hashes = (
        sp_b.prefill_steering_config_hash,
        sp_b.decode_steering_config_hash,
    )
    original_sae_hashes = (
        sp_b.prefill_sae_clamp_config_hash,
        sp_b.decode_sae_clamp_config_hash,
    )

    maybe_auto_promote_steering_modules(sp_a, _recording_rpc(calls), lru)
    maybe_auto_promote_steering_modules(sp_b, _recording_rpc(calls), lru)
    restored = pickle.loads(pickle.dumps(sp_b))

    assert restored.steering_module_ref == sp_b.steering_module_ref
    assert restored.sae_clamp_specs == (sae_spec,)
    assert restored.prefill_steering_config_hash == original_hashes[0]
    assert restored.decode_steering_config_hash == original_hashes[1]
    assert restored.prefill_sae_clamp_config_hash == original_sae_hashes[0]
    assert restored.decode_sae_clamp_config_hash == original_sae_hashes[1]


def test_lru_rejects_zero_capacity():
    with pytest.raises(ValueError, match="capacity"):
        SteeringAutoPromoteLRU(capacity=0)
