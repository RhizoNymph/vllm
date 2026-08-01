# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manager-side tests for the APC effective-decode-steering signature.

The worker reports a per-request signature so steered decode KV blocks are
keyed by the steering that produced them, not the admitted config. The
signature must be: None when nothing dynamic applies; deterministic and
content-addressed (same effective steering ⇒ same int) so two requests
under identical steering + tokens reuse; and distinct whenever the
override / tier / gain / monitor differs. See
docs/design/dynamic_steering_apc_notification.md. CPU-only.
"""

import numpy as np
import torch

from vllm.v1.worker.steering_manager import SteeringManager

HIDDEN = 8
_HP = "post_block"


def _mgr() -> SteeringManager:
    return SteeringManager(
        max_steering_configs=4, device=None, max_dynamic_steering_configs=4
    )


def _vec(v: float) -> dict[str, dict[int, np.ndarray]]:
    return {_HP: {0: np.full(HIDDEN, v, dtype=np.float32)}}


def test_no_dynamic_returns_none():
    mgr = _mgr()
    assert mgr.effective_decode_signature(None, base_decode_hash=123) is None


def test_override_signature_is_deterministic_and_content_addressed():
    mgr = _mgr()
    dyn_a, _ = mgr.register_dynamic_config(_vec(5.0))
    sig_a = mgr.effective_decode_signature(dyn_a, base_decode_hash=100)
    assert sig_a is not None
    # Same base + same override vectors on a second request ⇒ same sig
    # (so they can reuse each other's steered decode blocks).
    dyn_b, _ = mgr.register_dynamic_config(_vec(5.0))
    sig_b = mgr.effective_decode_signature(dyn_b, base_decode_hash=100)
    assert sig_a == sig_b
    # Different override vectors ⇒ different sig.
    dyn_c, _ = mgr.register_dynamic_config(_vec(9.0))
    assert mgr.effective_decode_signature(dyn_c, base_decode_hash=100) != sig_a


def test_signature_folds_base_decode_hash():
    mgr = _mgr()
    dyn, _ = mgr.register_dynamic_config(_vec(5.0))
    s1 = mgr.effective_decode_signature(dyn, base_decode_hash=100)
    s2 = mgr.effective_decode_signature(dyn, base_decode_hash=200)
    assert s1 != s2  # same override, different admitted config ⇒ distinct


def test_signature_disjoint_from_plain_admitted_hash():
    # An override on top of admitted config X must NOT collide with the
    # plain admitted hash X (else an override request would falsely reuse
    # an admitted-only request's blocks).
    mgr = _mgr()
    dyn, _ = mgr.register_dynamic_config(_vec(5.0))
    base = 7777
    assert mgr.effective_decode_signature(dyn, base) != base


def test_tier_changes_signature():
    mgr = _mgr()
    base = 100
    assert mgr.effective_decode_signature(None, base) is None
    mgr.update_dynamic_tier(_HP, 0, torch.full((HIDDEN,), 2.0))
    sig_tier = mgr.effective_decode_signature(None, base)
    assert sig_tier is not None
    # Different tier vector ⇒ different sig.
    mgr.update_dynamic_tier(_HP, 0, torch.full((HIDDEN,), 3.0))
    assert mgr.effective_decode_signature(None, base) != sig_tier


def test_tier_gain_changes_signature_no_quantization():
    mgr = _mgr()
    base = 100
    mgr.update_dynamic_tier(_HP, 0, torch.full((HIDDEN,), 2.0))
    mgr.set_dynamic_tier_gain(1.0)
    s1 = mgr.effective_decode_signature(None, base)
    mgr.set_dynamic_tier_gain(1.0001)  # tiny change ⇒ still a new key
    s2 = mgr.effective_decode_signature(None, base)
    assert s1 != s2


def test_monitor_changes_signature():
    mgr = _mgr()
    base = 100
    mgr.set_monitor(_HP, 0, torch.ones(HIDDEN), threshold=0.5, sharpness=2.0)
    s1 = mgr.effective_decode_signature(None, base)
    assert s1 is not None
    mgr.set_monitor(_HP, 0, torch.ones(HIDDEN), threshold=0.9, sharpness=2.0)
    assert mgr.effective_decode_signature(None, base) != s1


def test_components_compose():
    # override + tier + monitor together differ from any subset.
    mgr = _mgr()
    base = 100
    dyn, _ = mgr.register_dynamic_config(_vec(5.0))
    only_ovr = mgr.effective_decode_signature(dyn, base)
    mgr.update_dynamic_tier(_HP, 0, torch.full((HIDDEN,), 2.0))
    ovr_tier = mgr.effective_decode_signature(dyn, base)
    mgr.set_monitor(_HP, 0, torch.ones(HIDDEN), 0.5, 2.0)
    ovr_tier_mon = mgr.effective_decode_signature(dyn, base)
    assert len({only_ovr, ovr_tier, ovr_tier_mon}) == 3


def test_release_drops_override_signature_component():
    mgr = _mgr()
    base = 100
    dyn, _ = mgr.register_dynamic_config(_vec(5.0))
    assert mgr.effective_decode_signature(dyn, base) is not None
    mgr.release_dynamic_config(dyn)
    assert dyn not in mgr._dynamic_sig
    # With the override gone and nothing else dynamic ⇒ None again.
    assert mgr.effective_decode_signature(dyn, base) is None


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
