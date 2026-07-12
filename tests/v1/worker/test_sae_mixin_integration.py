# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-2 worker-mixin integration tests for the SAE feature-surgery path.

These tests exercise the worker-side state machinery — buffer
attachment lifecycle, weight injection, and the
admission/release/transition helpers — without spinning up a real
model runner.  We construct a richer harness that mirrors the parts
of the mixin Stage 2 actually touches: a fake ``vllm_config``, a
synthetic ``_steerable_layers_cache`` with bare ``nn.Module`` layers,
and the empty SAE state Stage 2 expects ``_init_steering_state`` to
have prepared.

Coverage:

* Buffer attach on ``register_steering_modules`` / detach on
  ``unregister_steering_modules``, including the replace-clears-both
  path and the kind-swap path.
* ``attach_sae_weights`` validates shape and copies tensors
  in-place into the right per-(layer, hook) buffer.
* ``_register_initial_sae_clamps`` honours ``is_prefilling`` and the
  hash==0 short-circuit.
* ``_release_sae_for_request`` releases under whichever phase the
  request was last admitted, idempotent on already-released requests.
* ``_handle_steering_transition`` releases prefill SAE row and
  registers decode SAE row in lockstep with the additive path.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm.v1.worker.steering_model_runner_mixin as steering_mixin_mod
from vllm import SamplingParams
from vllm.config.sae_steering_types import (
    SAEClampEntry,
    SAEClampSpec,
)
from vllm.exceptions import SteeringVectorError
from vllm.model_executor.layers.sae_steering import (
    SAESlotState,
    get_sae_slot_state,
    sae_buffers_attached,
    sae_site_slots,
)
from vllm.model_executor.layers.steering import (
    SteeringHookPoint,
    register_steering_buffers,
)
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.sae_clamp_manager import SAEClampManager
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin


@dataclass
class _StubModelConfig:
    dtype: torch.dtype = torch.float32


@dataclass
class _StubSchedulerConfig:
    max_num_batched_tokens: int = 16
    max_num_seqs: int = 8


@dataclass
class _StubSteeringConfig:
    max_steering_configs: int = 4


@dataclass
class _StubVllmConfig:
    model_config: _StubModelConfig = field(default_factory=_StubModelConfig)
    scheduler_config: _StubSchedulerConfig = field(default_factory=_StubSchedulerConfig)
    steering_config: _StubSteeringConfig = field(default_factory=_StubSteeringConfig)


def _slot(
    site: nn.Module,
    module_name: str = "g",
    hook: SteeringHookPoint = SteeringHookPoint.POST_BLOCK,
) -> SAESlotState:
    state = get_sae_slot_state(site, hook, module_name)
    assert state is not None, f"no slot for {module_name!r} at {hook.value!r}"
    return state


def _make_decoder_layer(layer_idx: int, hidden_size: int = 4) -> nn.Module:
    """Bare decoder-layer stand-in with the additive table buffers attached."""
    m = nn.Module()
    m.layer_idx = layer_idx  # type: ignore[attr-defined]
    register_steering_buffers(
        m,
        hidden_size,
        max_steering_tokens=16,
        max_steering_configs=4,
        dtype=torch.float32,
    )
    return m


class _RichHarness(SteeringModelRunnerMixin):
    """Mixin instance with the Stage-2 state surface populated."""

    def __init__(
        self,
        *,
        layer_indices: tuple[int, ...] = (20, 21),
        hidden_size: int = 4,
        max_sae_configs: int = 4,
    ) -> None:
        self.vllm_config = _StubVllmConfig(
            steering_config=_StubSteeringConfig(max_steering_configs=max_sae_configs),
        )
        self._steerable_layers_cache = {
            idx: _make_decoder_layer(idx, hidden_size=hidden_size)
            for idx in layer_indices
        }
        self._locally_owned_layers = frozenset(layer_indices)
        self._steering_module_registry: dict = {}
        self._steering_module_resolved_cache: dict = {}
        self._steering_module_pinned_rows: dict = {}
        self._sae_module_registry: dict = {}
        self._sae_steerable_sites: dict = {}
        self._req_sae_phase: dict = {}
        self._steering_reqs: dict = {}
        self._req_transition_scan_candidates: set[str] = set()
        self._steering_index_dirty = False
        self._sae_clamp_manager = SAEClampManager(max_sae_configs)
        self._steering_manager = None  # not exercised in these tests
        self._steering_rows_scratch = None
        self._steering_n_tokens_scratch = None
        self._steering_index_pinned = None
        self._sae_rows_scratch = None
        self._sae_index_pinned = None
        self.requests: dict = {}


def _sae_payload(
    *,
    layers: tuple[tuple[int, str], ...] = ((20, "post_block"),),
    clampable: tuple[int, ...] = (0, 1, 2, 34),
    activation: str = "relu",
    activation_params: dict[str, float] | None = None,
    d_model: int = 4,
    d_sae: int = 64,
) -> dict:
    return {
        "kind": "sae_delta",
        "sae_manifest": {
            "d_model": d_model,
            "d_sae": d_sae,
            "activation": activation,
            "layers": [list(p) for p in layers],
            "clampable_features": list(clampable),
            "activation_params": activation_params or {},
            "weights_uri": None,
        },
    }


def _make_clamp_spec(
    *,
    module_name: str = "g",
    feature_idx: int = 0,
    layer: int = 20,
    hook: str = "post_block",
    value: float = 5.0,
    kind: str = "absolute",
    phase: str = "both",
) -> SAEClampSpec:
    return SAEClampSpec(
        module_name=module_name,
        phase=phase,  # type: ignore[arg-type]
        clamps={
            hook: {
                layer: (
                    SAEClampEntry(
                        feature_idx=feature_idx,
                        kind=kind,  # type: ignore[arg-type]
                        value=value,
                    ),
                )
            }
        },
    )


class TestRegisterAttachesSaeBuffers:
    """``register_steering_modules`` for a SAE module attaches per-layer buffers."""

    def test_attaches_to_every_owned_layer_in_manifest(self):
        h = _RichHarness(layer_indices=(20, 21))
        h.register_steering_modules(
            {
                "g": _sae_payload(
                    layers=((20, "post_block"), (21, "post_block")),
                ),
            }
        )
        site20 = h._steerable_layers_cache[20]
        site21 = h._steerable_layers_cache[21]
        assert sae_buffers_attached(site20, SteeringHookPoint.POST_BLOCK)
        assert sae_buffers_attached(site21, SteeringHookPoint.POST_BLOCK)
        # POST_ATTN has no SAE buffers because the manifest doesn't
        # declare it.
        assert not sae_buffers_attached(site20, SteeringHookPoint.POST_ATTN)

    def test_skips_layers_not_owned_by_this_rank(self):
        # PP scenario: rank owns layers {20}, but manifest declares 20+21.
        h = _RichHarness(layer_indices=(20,))
        h.register_steering_modules(
            {
                "g": _sae_payload(
                    layers=((20, "post_block"), (21, "post_block")),
                ),
            }
        )
        # Registry entry must still exist (broadcast determinism).
        assert "g" in h._sae_module_registry
        # Site for layer 20 attached; layer 21 was never on this rank.
        assert sae_buffers_attached(
            h._steerable_layers_cache[20], SteeringHookPoint.POST_BLOCK
        )
        assert ("g", 21, "post_block") not in h._sae_steerable_sites

    def test_records_module_name_in_slot_record(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        site = h._steerable_layers_cache[20]
        slots = sae_site_slots(site, SteeringHookPoint.POST_BLOCK)
        assert [s.module_name for s in slots] == ["g"]

    def test_buffer_shape_matches_manifest(self):
        h = _RichHarness(hidden_size=8)
        h.register_steering_modules(
            {"g": _sae_payload(d_model=8, clampable=(0, 1, 2))},
        )
        site = h._steerable_layers_cache[20]
        assert _slot(site).encoder_weight.shape == (3, 8)

    def test_runtime_buffers_follow_existing_layer_device(self):
        h = _RichHarness(layer_indices=(20,))
        layer = h._steerable_layers_cache[20].to("meta")
        h._steerable_layers_cache[20] = layer

        h.register_steering_modules({"g": _sae_payload()})

        assert layer.sae_index.device.type == "meta"  # type: ignore[attr-defined]
        state = _slot(layer)
        assert state.encoder_weight.device.type == "meta"
        assert state.clamp_kind.device.type == "meta"

    def test_index_buffer_attached_and_shared(self):
        h = _RichHarness(layer_indices=(20, 21))
        h.register_steering_modules(
            {
                "g": _sae_payload(
                    layers=((20, "post_block"), (21, "post_block")),
                ),
            }
        )
        site20 = h._steerable_layers_cache[20]
        site21 = h._steerable_layers_cache[21]
        assert site20.sae_index is site21.sae_index  # type: ignore[attr-defined]

    def test_index_shared_across_distinct_modules(self):
        """Two SAE modules on disjoint layers must share one ``sae_index``.

        The per-step populator writes through a single tensor picked
        from ``_sae_steerable_sites``; if the second module's layer
        kept its own ``sae_index``, clamp requests against it would
        gather row 0 and silently no-op.
        """
        h = _RichHarness(layer_indices=(20, 21))
        h.register_steering_modules(
            {"a": _sae_payload(layers=((20, "post_block"),))},
        )
        h.register_steering_modules(
            {"b": _sae_payload(layers=((21, "post_block"),))},
        )
        site20 = h._steerable_layers_cache[20]
        site21 = h._steerable_layers_cache[21]
        assert site20.sae_index is site21.sae_index  # type: ignore[attr-defined]

    def test_register_warms_attached_sae_module(self, monkeypatch):
        h = _RichHarness(layer_indices=(20, 21))
        calls: list[tuple[object, tuple[int, ...], torch.dtype]] = []

        def record_warmup(self, *, manifest, attached_layers, ref_dtype):
            calls.append(
                (
                    manifest,
                    tuple(layer.layer_idx for layer in attached_layers),
                    ref_dtype,
                )
            )

        monkeypatch.setattr(
            SteeringModelRunnerMixin,
            "_warmup_sae_kernel_for_module",
            record_warmup,
        )

        h.register_steering_modules(
            {
                "g": _sae_payload(
                    layers=((20, "post_block"), (21, "post_block")),
                    clampable=(0, 1),
                    d_model=4,
                ),
            }
        )

        assert len(calls) == 1
        manifest, layer_indices, ref_dtype = calls[0]
        assert manifest.d_model == 4
        assert manifest.clampable_features == (0, 1)
        assert layer_indices == (20, 21)
        assert ref_dtype is torch.float32

    def test_negative_layer_indices_are_rejected(self):
        h = _RichHarness()
        with pytest.raises(SteeringVectorError, match="non-negative"):
            h.register_steering_modules(
                {"g": _sae_payload(layers=((-1, "post_block"),))},
            )
        assert h._sae_module_registry == {}
        assert h._sae_steerable_sites == {}

    def test_index_registration_failure_rolls_back_half_attached_buffers(
        self,
        monkeypatch,
    ):
        h = _RichHarness(layer_indices=(20,))
        original_register_index = steering_mixin_mod.register_sae_index_buffer

        def fail_register_index(*args, **kwargs):
            raise RuntimeError("index allocation failed")

        monkeypatch.setattr(
            steering_mixin_mod,
            "register_sae_index_buffer",
            fail_register_index,
        )

        with pytest.raises(RuntimeError, match="index allocation failed"):
            h.register_steering_modules({"g": _sae_payload()})

        site = h._steerable_layers_cache[20]
        assert not sae_buffers_attached(site, SteeringHookPoint.POST_BLOCK)
        assert h._sae_module_registry == {}
        assert h._sae_steerable_sites == {}

        monkeypatch.setattr(
            steering_mixin_mod,
            "register_sae_index_buffer",
            original_register_index,
        )
        h.register_steering_modules({"g": _sae_payload()})
        assert sae_buffers_attached(site, SteeringHookPoint.POST_BLOCK)


class TestUnregisterDetachesSaeBuffers:
    def test_unregister_strips_buffers(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        h.unregister_steering_modules(["g"])
        site = h._steerable_layers_cache[20]
        assert not sae_buffers_attached(site, SteeringHookPoint.POST_BLOCK)
        assert h._sae_steerable_sites == {}

    def test_replace_clears_existing_sae_buffers(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        # replace=True drops the old SAE module entirely.
        h.register_steering_modules({"new": _sae_payload()}, replace=True)
        site = h._steerable_layers_cache[20]
        # 'g' should be gone.
        assert "g" not in h._sae_module_registry
        assert get_sae_slot_state(site, SteeringHookPoint.POST_BLOCK, "g") is None
        # 'new' should be attached.
        slots = sae_site_slots(site, SteeringHookPoint.POST_BLOCK)
        assert [s.module_name for s in slots] == ["new"]

    def test_kind_swap_detaches_old_sae_buffers(self):
        # Re-registering the same name as additive must drop the SAE
        # buffers so the layer state is consistent with the registry.
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        site = h._steerable_layers_cache[20]
        assert sae_buffers_attached(site, SteeringHookPoint.POST_BLOCK)
        h.register_steering_modules(
            {
                "g": {
                    "kind": "additive",
                    "vectors": {"post_block": {0: [0.1, 0.2, 0.3, 0.4]}},
                    "prefill_vectors": None,
                    "decode_vectors": None,
                },
            }
        )
        assert not sae_buffers_attached(site, SteeringHookPoint.POST_BLOCK)


class TestAttachSaeWeights:
    def test_copies_into_buffers_in_place(self):
        h = _RichHarness(hidden_size=4)
        h.register_steering_modules({"g": _sae_payload(d_model=4, clampable=(0, 1))})
        site = h._steerable_layers_cache[20]
        enc_w_before = _slot(site).encoder_weight.clone()
        new_w = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        new_b = torch.tensor([0.1, 0.2])
        new_dec = torch.tensor([[9.0, 8.0, 7.0, 6.0], [5.0, 4.0, 3.0, 2.0]])
        h.attach_sae_weights(
            "g",
            {
                (20, "post_block"): {
                    "encoder_weight": new_w,
                    "encoder_bias": new_b,
                    "decoder_weight": new_dec,
                }
            },
        )
        state = _slot(site)
        assert torch.equal(state.encoder_weight, new_w)
        assert torch.equal(state.encoder_bias, new_b)
        assert torch.equal(state.decoder_weight, new_dec)
        assert not torch.equal(state.encoder_weight, enc_w_before)

    def test_unknown_module_raises(self):
        h = _RichHarness()
        with pytest.raises(SteeringVectorError, match="not registered"):
            h.attach_sae_weights("g", {})

    def test_jumprelu_requires_threshold_tensor(self):
        h = _RichHarness()
        h.register_steering_modules(
            {"g": _sae_payload(clampable=(0, 1), activation="jumprelu")}
        )
        with pytest.raises(SteeringVectorError, match="threshold"):
            h.attach_sae_weights(
                "g",
                {
                    (20, "post_block"): {
                        "encoder_weight": torch.zeros(2, 4),
                        "encoder_bias": torch.zeros(2),
                        "decoder_weight": torch.zeros(2, 4),
                        # threshold deliberately absent
                    }
                },
            )

    def test_jumprelu_threshold_copied_into_buffer(self):
        h = _RichHarness()
        h.register_steering_modules(
            {"g": _sae_payload(clampable=(0, 1), activation="jumprelu")}
        )
        threshold = torch.tensor([0.25, 0.75])
        h.attach_sae_weights(
            "g",
            {
                (20, "post_block"): {
                    "encoder_weight": torch.zeros(2, 4),
                    "encoder_bias": torch.zeros(2),
                    "decoder_weight": torch.zeros(2, 4),
                    "threshold": threshold,
                }
            },
        )
        site = h._steerable_layers_cache[20]
        buf = _slot(site).threshold
        assert buf.dtype is torch.float32
        assert torch.equal(buf, threshold)

    def test_relu_threshold_is_optional(self):
        # Non-JumpReLU modules may omit the threshold tensor; the
        # zero-filled buffer stands in for the fixed op arity.
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload(clampable=(0, 1))})
        h.attach_sae_weights(
            "g",
            {
                (20, "post_block"): {
                    "encoder_weight": torch.zeros(2, 4),
                    "encoder_bias": torch.zeros(2),
                    "decoder_weight": torch.zeros(2, 4),
                }
            },
        )
        site = h._steerable_layers_cache[20]
        buf = _slot(site).threshold
        assert torch.equal(buf, torch.zeros(2))

    def test_threshold_shape_mismatch_raises(self):
        h = _RichHarness()
        h.register_steering_modules(
            {"g": _sae_payload(clampable=(0, 1), activation="jumprelu")}
        )
        with pytest.raises(SteeringVectorError, match="threshold.*shape"):
            h.attach_sae_weights(
                "g",
                {
                    (20, "post_block"): {
                        "encoder_weight": torch.zeros(2, 4),
                        "encoder_bias": torch.zeros(2),
                        "decoder_weight": torch.zeros(2, 4),
                        "threshold": torch.zeros(3),  # wrong n_clamp
                    }
                },
            )

    def test_missing_tensor_key_raises(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload(clampable=(0, 1))})
        with pytest.raises(SteeringVectorError, match="missing"):
            h.attach_sae_weights(
                "g",
                {(20, "post_block"): {"encoder_weight": torch.zeros(2, 4)}},
            )

    def test_shape_mismatch_raises(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload(clampable=(0, 1))})
        with pytest.raises(SteeringVectorError, match="shape"):
            h.attach_sae_weights(
                "g",
                {
                    (20, "post_block"): {
                        "encoder_weight": torch.zeros(99, 4),  # wrong n_clamp
                        "encoder_bias": torch.zeros(2),
                        "decoder_weight": torch.zeros(2, 4),
                    }
                },
            )

    def test_shape_mismatch_does_not_partially_copy(self):
        h = _RichHarness(layer_indices=(20, 21))
        h.register_steering_modules(
            {
                "g": _sae_payload(
                    layers=((20, "post_block"), (21, "post_block")),
                    clampable=(0, 1),
                )
            }
        )
        site20 = h._steerable_layers_cache[20]
        enc_w20 = _slot(site20).encoder_weight
        assert torch.count_nonzero(enc_w20) == 0

        with pytest.raises(SteeringVectorError, match="shape"):
            h.attach_sae_weights(
                "g",
                {
                    (20, "post_block"): {
                        "encoder_weight": torch.ones(2, 4),
                        "encoder_bias": torch.ones(2),
                        "decoder_weight": torch.ones(2, 4),
                    },
                    (21, "post_block"): {
                        "encoder_weight": torch.zeros(99, 4),
                        "encoder_bias": torch.zeros(2),
                        "decoder_weight": torch.zeros(2, 4),
                    },
                },
            )
        assert torch.count_nonzero(enc_w20) == 0

    def test_non_floating_tensor_raises(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload(clampable=(0, 1))})
        with pytest.raises(SteeringVectorError, match="floating dtype"):
            h.attach_sae_weights(
                "g",
                {
                    (20, "post_block"): {
                        "encoder_weight": torch.zeros(2, 4, dtype=torch.int64),
                        "encoder_bias": torch.zeros(2),
                        "decoder_weight": torch.zeros(2, 4),
                    }
                },
            )

    def test_non_tensor_weight_raises_steering_error(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload(clampable=(0, 1))})
        with pytest.raises(SteeringVectorError, match="torch.Tensor"):
            h.attach_sae_weights(
                "g",
                {
                    (20, "post_block"): {
                        "encoder_weight": [[0.0, 0.0, 0.0, 0.0]],
                        "encoder_bias": torch.zeros(2),
                        "decoder_weight": torch.zeros(2, 4),
                    }
                },
            )

    def test_non_finite_tensor_raises(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload(clampable=(0, 1))})
        bad = torch.zeros(2, 4)
        bad[0, 0] = float("nan")
        with pytest.raises(SteeringVectorError, match="finite values"):
            h.attach_sae_weights(
                "g",
                {
                    (20, "post_block"): {
                        "encoder_weight": bad,
                        "encoder_bias": torch.zeros(2),
                        "decoder_weight": torch.zeros(2, 4),
                    }
                },
            )

    def test_layer_not_owned_by_rank_silently_skipped(self):
        # PP scenario: rank doesn't own layer 99, so the weights for it
        # are dropped without error.  The on-disk loader can't know
        # which rank owns which layer; pruning is the rank's job.
        h = _RichHarness(layer_indices=(20,))
        h.register_steering_modules(
            {"g": _sae_payload(layers=((99, "post_block"),), clampable=(0, 1))}
        )
        h.attach_sae_weights(
            "g",
            {
                (99, "post_block"): {
                    "encoder_weight": torch.zeros(2, 4),
                    "encoder_bias": torch.zeros(2),
                    "decoder_weight": torch.zeros(2, 4),
                }
            },
        )

    def test_missing_owned_site_raises(self):
        h = _RichHarness(layer_indices=(20, 21))
        h.register_steering_modules(
            {
                "g": _sae_payload(
                    layers=((20, "post_block"), (21, "post_block")),
                    clampable=(0, 1),
                )
            }
        )
        with pytest.raises(SteeringVectorError, match="missing weights"):
            h.attach_sae_weights(
                "g",
                {
                    (20, "post_block"): {
                        "encoder_weight": torch.zeros(2, 4),
                        "encoder_bias": torch.zeros(2),
                        "decoder_weight": torch.zeros(2, 4),
                    }
                },
            )


class TestRegisterWithInlineWeights:
    """``register_steering_modules`` accepts inline weights for SAE
    payloads.  The worker registers the manifest and attaches the
    tensors atomically — there is no observable window where the
    module is registered but the buffers are still zero-filled."""

    def _payload_with_weights(self, *, d_model: int = 4, clampable=(0, 1)) -> dict:
        n = len(clampable)
        body = _sae_payload(d_model=d_model, clampable=clampable)
        body["sae_weights"] = {
            (20, "post_block"): {
                "encoder_weight": torch.ones(n, d_model),
                "encoder_bias": torch.full((n,), 0.5),
                "decoder_weight": torch.full((n, d_model), 2.0),
            }
        }
        return body

    def test_inline_weights_populate_buffers_in_one_call(self):
        h = _RichHarness(hidden_size=4)
        h.register_steering_modules({"g": self._payload_with_weights()})
        site = h._steerable_layers_cache[20]
        assert torch.equal(_slot(site).encoder_weight, torch.ones(2, 4))

    def test_attach_failure_rolls_back_registry_entry(self):
        """A bad weight shape inside the inline payload must leave the
        worker with neither a registered module nor any half-attached
        buffers — otherwise a future request could resolve the name
        and gather rows from zero-filled tables."""
        h = _RichHarness(hidden_size=4)
        bad = _sae_payload(d_model=4, clampable=(0, 1))
        bad["sae_weights"] = {
            (20, "post_block"): {
                "encoder_weight": torch.zeros(99, 4),  # wrong n_clamp
                "encoder_bias": torch.zeros(2),
                "decoder_weight": torch.zeros(2, 4),
            }
        }
        with pytest.raises(SteeringVectorError):
            h.register_steering_modules({"g": bad})
        assert "g" not in h._sae_module_registry
        site = h._steerable_layers_cache[20]
        assert not sae_buffers_attached(site, SteeringHookPoint.POST_BLOCK)

    def test_additive_to_sae_replacement_failure_restores_additive(self):
        """A failed additive-to-SAE replacement must leave the additive
        module intact on the worker.  Otherwise the server-side rollback
        would restore the additive entry on the registry while the
        worker has lost it, diverging the two."""
        h = _RichHarness(hidden_size=4)
        # Seed an additive module at name 'g'.
        additive_payload = {
            "kind": "additive",
            "vectors": {"post_block": {20: [0.1, 0.2, 0.3, 0.4]}},
            "prefill_vectors": None,
            "decode_vectors": None,
        }
        h.register_steering_modules({"g": additive_payload})
        assert "g" in h._steering_module_registry
        prior_additive = h._steering_module_registry["g"]

        # Re-register the same name as SAE with a bad-shape weight
        # payload; attach must fail.
        bad = _sae_payload(d_model=4, clampable=(0, 1))
        bad["sae_weights"] = {
            (20, "post_block"): {
                "encoder_weight": torch.zeros(99, 4),  # wrong n_clamp
                "encoder_bias": torch.zeros(2),
                "decoder_weight": torch.zeros(2, 4),
            }
        }
        with pytest.raises(SteeringVectorError):
            h.register_steering_modules({"g": bad})

        # SAE entry never sticks; additive entry restored bit-for-bit.
        assert "g" not in h._sae_module_registry
        assert "g" in h._steering_module_registry
        assert h._steering_module_registry["g"] is prior_additive
        site = h._steerable_layers_cache[20]
        assert not sae_buffers_attached(site, SteeringHookPoint.POST_BLOCK)

    def test_replacement_failure_restores_previous_module(self):
        """A failed replacement must put the previously-loaded SAE
        weights back, not destroy them along with the bad request."""
        h = _RichHarness(hidden_size=4)
        # Seed a working SAE with recognisable weight values.
        good = _sae_payload(d_model=4, clampable=(0, 1))
        good["sae_weights"] = {
            (20, "post_block"): {
                "encoder_weight": torch.full((2, 4), 7.0),
                "encoder_bias": torch.full((2,), 0.25),
                "decoder_weight": torch.full((2, 4), 3.0),
            }
        }
        h.register_steering_modules({"g": good})
        site = h._steerable_layers_cache[20]
        assert torch.equal(_slot(site).encoder_weight, torch.full((2, 4), 7.0))

        # Re-register the same name with a bad-shape weight payload.
        bad = _sae_payload(d_model=4, clampable=(0, 1))
        bad["sae_weights"] = {
            (20, "post_block"): {
                "encoder_weight": torch.zeros(99, 4),  # wrong n_clamp
                "encoder_bias": torch.zeros(2),
                "decoder_weight": torch.zeros(2, 4),
            }
        }
        with pytest.raises(SteeringVectorError):
            h.register_steering_modules({"g": bad})

        # Registry entry preserved, buffers re-attached with the
        # original tensors.
        assert "g" in h._sae_module_registry
        assert sae_buffers_attached(site, SteeringHookPoint.POST_BLOCK)
        state = _slot(site)
        assert torch.equal(state.encoder_weight, torch.full((2, 4), 7.0))
        assert torch.equal(state.encoder_bias, torch.full((2,), 0.25))

    def test_replacement_failure_restores_jumprelu_threshold(self):
        """Snapshot/rollback must round-trip the per-feature threshold
        buffer, not just the encoder/decoder weights."""
        h = _RichHarness(hidden_size=4)
        threshold = torch.tensor([0.25, 0.75])
        good = _sae_payload(d_model=4, clampable=(0, 1), activation="jumprelu")
        good["sae_weights"] = {
            (20, "post_block"): {
                "encoder_weight": torch.full((2, 4), 7.0),
                "encoder_bias": torch.full((2,), 0.25),
                "decoder_weight": torch.full((2, 4), 3.0),
                "threshold": threshold,
            }
        }
        h.register_steering_modules({"g": good})
        site = h._steerable_layers_cache[20]
        assert torch.equal(_slot(site).threshold, threshold)

        bad = _sae_payload(d_model=4, clampable=(0, 1), activation="jumprelu")
        bad["sae_weights"] = {
            (20, "post_block"): {
                "encoder_weight": torch.zeros(99, 4),  # wrong n_clamp
                "encoder_bias": torch.zeros(2),
                "decoder_weight": torch.zeros(2, 4),
                "threshold": torch.zeros(2),
            }
        }
        with pytest.raises(SteeringVectorError):
            h.register_steering_modules({"g": bad})

        # Registry entry preserved and the original per-feature
        # thresholds restored after the failed replacement.
        assert "g" in h._sae_module_registry
        assert torch.equal(_slot(site).threshold, threshold)

    def test_replace_true_failure_restores_additive_registry(self):
        h = _RichHarness(hidden_size=4)
        h.register_steering_modules(
            {
                "m": {
                    "kind": "additive",
                    "vectors": {"post_block": {20: [0.1, 0.2, 0.3, 0.4]}},
                }
            }
        )
        prior = h._steering_module_registry["m"]
        prior_cache = h._steering_module_resolved_cache["m"]

        with pytest.raises(SteeringVectorError):
            h.register_steering_modules(
                {"bad": {"kind": "sae_delta", "sae_manifest": {"d_model": 4}}},
                replace=True,
            )

        assert h._steering_module_registry["m"] is prior
        assert h._steering_module_resolved_cache["m"] is prior_cache
        assert h._sae_module_registry == {}

    def test_replace_true_failure_restores_sae_buffers_and_weights(self):
        h = _RichHarness(hidden_size=4)
        good = _sae_payload(d_model=4, clampable=(0, 1))
        good["sae_weights"] = {
            (20, "post_block"): {
                "encoder_weight": torch.full((2, 4), 5.0),
                "encoder_bias": torch.full((2,), 0.75),
                "decoder_weight": torch.full((2, 4), 6.0),
            }
        }
        h.register_steering_modules({"g": good})
        site = h._steerable_layers_cache[20]

        bad = _sae_payload(d_model=4, clampable=(0, 1))
        bad["sae_weights"] = {
            (20, "post_block"): {
                "encoder_weight": torch.zeros(99, 4),
                "encoder_bias": torch.zeros(2),
                "decoder_weight": torch.zeros(2, 4),
            }
        }
        with pytest.raises(SteeringVectorError):
            h.register_steering_modules({"bad": bad}, replace=True)

        assert "g" in h._sae_module_registry
        assert "bad" not in h._sae_module_registry
        assert sae_buffers_attached(site, SteeringHookPoint.POST_BLOCK)
        state = _slot(site)
        assert torch.equal(state.encoder_weight, torch.full((2, 4), 5.0))
        assert torch.equal(state.encoder_bias, torch.full((2,), 0.75))


class TestSharedSiteModules:
    """Multiple SAE delta modules sharing one (layer, hook) site."""

    def _payload_with_fill(self, fill: float) -> dict:
        body = _sae_payload(d_model=4, clampable=(0, 1))
        body["sae_weights"] = {
            (20, "post_block"): {
                "encoder_weight": torch.full((2, 4), fill),
                "encoder_bias": torch.full((2,), fill / 10),
                "decoder_weight": torch.full((2, 4), fill * 2),
            }
        }
        return body

    def test_two_modules_share_site_over_rpc_surface(self):
        h = _RichHarness(layer_indices=(20,))
        h.register_steering_modules({"a": _sae_payload(clampable=(0, 1))})
        h.register_steering_modules({"b": _sae_payload(clampable=(0, 1))})
        assert "a" in h._sae_module_registry
        assert "b" in h._sae_module_registry
        assert ("a", 20, "post_block") in h._sae_steerable_sites
        assert ("b", 20, "post_block") in h._sae_steerable_sites
        site = h._steerable_layers_cache[20]
        slots = sae_site_slots(site, SteeringHookPoint.POST_BLOCK)
        assert [s.module_name for s in slots] == ["a", "b"]

    def test_attach_weights_lands_in_own_slot(self):
        h = _RichHarness(layer_indices=(20,))
        h.register_steering_modules({"a": self._payload_with_fill(1.0)})
        h.register_steering_modules({"b": self._payload_with_fill(9.0)})
        site = h._steerable_layers_cache[20]
        assert torch.equal(
            _slot(site, "a").encoder_weight, torch.full((2, 4), 1.0)
        )
        assert torch.equal(
            _slot(site, "b").encoder_weight, torch.full((2, 4), 9.0)
        )

    def test_unregister_one_module_keeps_sibling_slot(self):
        h = _RichHarness(layer_indices=(20,))
        h.register_steering_modules({"a": self._payload_with_fill(1.0)})
        h.register_steering_modules({"b": self._payload_with_fill(9.0)})
        site = h._steerable_layers_cache[20]
        h.unregister_steering_modules(["a"])
        assert "a" not in h._sae_module_registry
        assert ("a", 20, "post_block") not in h._sae_steerable_sites
        assert ("b", 20, "post_block") in h._sae_steerable_sites
        assert get_sae_slot_state(site, SteeringHookPoint.POST_BLOCK, "a") is None
        # Sibling's slot and weights are untouched.
        assert torch.equal(
            _slot(site, "b").encoder_weight, torch.full((2, 4), 9.0)
        )

    def test_replacement_of_one_module_preserves_sibling(self):
        h = _RichHarness(layer_indices=(20,))
        h.register_steering_modules({"a": self._payload_with_fill(1.0)})
        h.register_steering_modules({"b": self._payload_with_fill(9.0)})
        site = h._steerable_layers_cache[20]
        # Failed replacement of 'a' restores 'a' and leaves 'b' alone.
        bad = _sae_payload(d_model=4, clampable=(0, 1))
        bad["sae_weights"] = {
            (20, "post_block"): {
                "encoder_weight": torch.zeros(99, 4),  # wrong n_clamp
                "encoder_bias": torch.zeros(2),
                "decoder_weight": torch.zeros(2, 4),
            }
        }
        with pytest.raises(SteeringVectorError):
            h.register_steering_modules({"a": bad})
        assert torch.equal(
            _slot(site, "a").encoder_weight, torch.full((2, 4), 1.0)
        )
        assert torch.equal(
            _slot(site, "b").encoder_weight, torch.full((2, 4), 9.0)
        )

    def test_replace_true_failure_restores_both_shared_site_modules(self):
        # Atomic replace with a poisoned re-add: both prior modules on
        # the shared site come back, buffers and weights intact.
        h = _RichHarness(layer_indices=(20,))
        h.register_steering_modules({"a": self._payload_with_fill(1.0)})
        h.register_steering_modules({"b": self._payload_with_fill(9.0)})
        site = h._steerable_layers_cache[20]

        bad = _sae_payload(d_model=4, clampable=(0, 1))
        bad["sae_weights"] = {
            (20, "post_block"): {
                "encoder_weight": torch.zeros(99, 4),  # wrong n_clamp
                "encoder_bias": torch.zeros(2),
                "decoder_weight": torch.zeros(2, 4),
            }
        }
        with pytest.raises(SteeringVectorError):
            h.register_steering_modules({"bad": bad}, replace=True)

        assert set(h._sae_module_registry) == {"a", "b"}
        assert "bad" not in h._sae_module_registry
        slots = sae_site_slots(site, SteeringHookPoint.POST_BLOCK)
        assert sorted(s.module_name for s in slots) == ["a", "b"]
        assert torch.equal(
            _slot(site, "a").encoder_weight, torch.full((2, 4), 1.0)
        )
        assert torch.equal(
            _slot(site, "b").encoder_weight, torch.full((2, 4), 9.0)
        )


class TestRegisterInitialSaeClamps:
    def test_admit_under_prefill_when_prefilling(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(),))
        h._register_initial_sae_clamps(
            "req-1",
            sp,
            prefill_hash=111,
            decode_hash=222,
            is_prefilling=True,
        )
        assert h._req_sae_phase["req-1"] == "prefill"
        sae_hash = sp.prefill_sae_clamp_config_hash
        assert h._sae_clamp_manager.config_to_row[(sae_hash, "prefill")] == 3

    def test_admit_under_decode_when_full_prefix_cache_hit(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(),))
        h._register_initial_sae_clamps(
            "req-1",
            sp,
            prefill_hash=111,
            decode_hash=222,
            is_prefilling=False,
        )
        assert h._req_sae_phase["req-1"] == "decode"
        sae_hash = sp.decode_sae_clamp_config_hash
        assert h._sae_clamp_manager.config_to_row[(sae_hash, "decode")] == 3

    def test_same_sae_spec_deduplicates_across_different_combined_hashes(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(),))

        h._register_initial_sae_clamps(
            "req-1",
            sp,
            prefill_hash=111,
            decode_hash=333,
            is_prefilling=True,
        )
        h._register_initial_sae_clamps(
            "req-2",
            sp,
            prefill_hash=222,
            decode_hash=444,
            is_prefilling=True,
        )

        sae_hash = sp.prefill_sae_clamp_config_hash
        assert h._sae_clamp_manager.config_to_row == {(sae_hash, "prefill"): 3}
        assert h._sae_clamp_manager.config_refcounts[(sae_hash, "prefill")] == 2

    def test_no_clamps_no_op(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams()
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=111, decode_hash=222, is_prefilling=True
        )
        assert "req-1" not in h._req_sae_phase
        assert h._sae_clamp_manager.config_to_row == {}

    def test_hash_zero_skips_admission(self):
        # A request might carry sae_clamp_specs but have hash 0 if all
        # clamps are phase-filtered out for the current worker phase.
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(phase="decode"),))
        # Even with specs present, hash 0 means "no SAE state in this
        # phase" — the manager must not register.
        h._register_initial_sae_clamps(
            "req-1",
            sp,
            prefill_hash=0,
            decode_hash=222,
            is_prefilling=True,
        )
        assert "req-1" not in h._req_sae_phase

    def test_decode_only_spec_does_not_consume_prefill_row_from_additive_hash(self):
        # The prefill hash can be nonzero because of additive steering
        # while the SAE spec itself is decode-only.  SAE admission must
        # look at the phase-filtered SAE specs, not just the combined
        # additive+SAE hash, or it burns a no-op SAE row in prefill.
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(phase="decode"),))

        h._register_initial_sae_clamps(
            "req-1",
            sp,
            prefill_hash=111,
            decode_hash=222,
            is_prefilling=True,
        )

        assert "req-1" not in h._req_sae_phase
        assert h._sae_clamp_manager.config_to_row == {}

    def test_prefill_only_spec_does_not_consume_decode_row_from_additive_hash(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(phase="prefill"),))

        h._register_initial_sae_clamps(
            "req-1",
            sp,
            prefill_hash=111,
            decode_hash=222,
            is_prefilling=False,
        )

        assert "req-1" not in h._req_sae_phase
        assert h._sae_clamp_manager.config_to_row == {}


class TestReleaseSaeForRequest:
    def test_releases_under_recorded_phase(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(),))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=111, decode_hash=222, is_prefilling=True
        )
        h._release_sae_for_request("req-1", prefill_hash=111, decode_hash=222)
        assert "req-1" not in h._req_sae_phase
        assert h._sae_clamp_manager.config_to_row == {}

    def test_release_unknown_request_no_op(self):
        h = _RichHarness()
        # Must not raise.
        h._release_sae_for_request("never-admitted", prefill_hash=1, decode_hash=2)

    def test_release_decode_uses_decode_hash(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(),))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=111, decode_hash=222, is_prefilling=False
        )
        h._release_sae_for_request("req-1", prefill_hash=111, decode_hash=222)
        assert h._sae_clamp_manager.config_to_row == {}

    def test_finished_release_does_not_require_additive_manager(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(),))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=111, decode_hash=222, is_prefilling=True
        )
        h.requests["req-1"] = SimpleNamespace(
            prefill_steering_config_hash=111,
            decode_steering_config_hash=222,
        )

        h._steering_finish_requests({"req-1"})

        assert "req-1" not in h._req_sae_phase
        assert h._sae_clamp_manager.config_to_row == {}


class TestUpdateSaeBuffersPopulator:
    """End-to-end populate sweep against a small SAE-attached harness.

    Bypasses the per-step ``_update_sae_buffers`` (which depends on
    ``input_batch`` / ``scheduler_output``) and exercises the
    populator directly.  The per-step pass is covered by the
    integration test in Stage 3.
    """

    def test_populates_active_row_with_clamp(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload(clampable=(34,))})
        sp = SamplingParams(
            sae_clamp_specs=(
                _make_clamp_spec(feature_idx=34, value=7.0, kind="absolute"),
            )
        )
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=111, decode_hash=222, is_prefilling=True
        )

        # Drive the populator via the worker flow.
        site = h._steerable_layers_cache[20]
        from vllm.model_executor.layers.sae_steering import populate_sae_clamp_table

        populate_sae_clamp_table(
            manager=h._sae_clamp_manager,
            module=site,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            clampable_features=(34,),
            layer_idx=20,
        )
        kind = _slot(site).clamp_kind
        row = h._sae_clamp_manager.config_to_row[
            (sp.prefill_sae_clamp_config_hash, "prefill")
        ]
        # The prefill registration row gets the absolute clamp.
        from vllm.model_executor.layers.sae_steering import CLAMP_KIND_ABSOLUTE

        assert kind[row, 0].item() == CLAMP_KIND_ABSOLUTE


class TestKindSwapClearsTracking:
    """Re-registering ``g`` as additive must drop ``_req_sae_phase``-style state."""

    def test_unregister_does_not_release_admitted_specs(self):
        # Concrete: SAE module 'g' admits a clamp; then 'g' is
        # unregistered.  The admitted row in the manager outlives
        # the registry entry — the manager refcount is what frees it.
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(),))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=111, decode_hash=222, is_prefilling=True
        )
        h.unregister_steering_modules(["g"])
        # The manager still has the row; release must come via
        # _release_sae_for_request.  This documents Stage-2 behaviour:
        # the buffer-attach lifecycle and the request-row lifecycle
        # are independent.  (Production-side, the request will fail
        # validation on its next admission step anyway.)
        sae_hash = sp.prefill_sae_clamp_config_hash
        assert h._sae_clamp_manager.config_to_row[(sae_hash, "prefill")] == 3


class TestWorkerRpcSurface:
    def test_gpu_worker_exposes_attach_sae_weights_rpc(self):
        class _Runner:
            def __init__(self):
                self.calls = []

            def attach_sae_weights(self, module_name, weights):
                self.calls.append((module_name, weights))

        worker = Worker.__new__(Worker)
        runner = _Runner()
        worker.model_runner = runner
        weights = {
            (20, "post_block"): {
                "encoder_weight": torch.zeros(1, 2),
                "encoder_bias": torch.zeros(1),
                "decoder_weight": torch.zeros(1, 2),
            }
        }

        worker.attach_sae_weights("g", weights)

        assert runner.calls == [("g", weights)]


def _raw_global_specs(
    *,
    module_name: str = "g",
    layer: int = 20,
    hook: str = "post_block",
    feature_idx: int = 34,
    value: float = 5.0,
) -> list[dict]:
    """JSON-shape clamp specs, as the HTTP router forwards them."""
    return [
        {
            "module_name": module_name,
            "clamps": {
                hook: {
                    str(layer): [
                        {
                            "feature_idx": feature_idx,
                            "kind": "absolute",
                            "value": value,
                        }
                    ]
                }
            },
        }
    ]


class TestSetSaeGlobalClamps:
    """``set_sae_global_clamps`` — the worker half of the
    ``/v1/steering/sae`` HTTP surface, including the two-phase
    ``validate_only`` contract."""

    def test_validate_only_does_not_commit(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})

        ranks = h.set_sae_global_clamps(
            prefill_specs_raw=_raw_global_specs(),
            validate_only=True,
        )

        assert ranks == (0, 0)
        assert h._sae_clamp_manager.global_prefill_specs == ()
        assert h._sae_clamp_manager.global_decode_specs == ()

    def test_validate_only_still_rejects_unknown_module(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})

        with pytest.raises(SteeringVectorError, match="unknown module"):
            h.set_sae_global_clamps(
                prefill_specs_raw=_raw_global_specs(module_name="nope"),
                validate_only=True,
            )
        assert h._sae_clamp_manager.global_prefill_specs == ()

    def test_apply_commits_to_global_tier(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})

        ranks = h.set_sae_global_clamps(
            prefill_specs_raw=_raw_global_specs(),
        )

        assert ranks == (0, 0)
        assert len(h._sae_clamp_manager.global_prefill_specs) == 1
        assert h._sae_clamp_manager.global_prefill_specs[0].module_name == "g"
        assert h._sae_clamp_manager.global_decode_specs == ()

    def test_two_phase_validate_then_apply(self):
        """Mirrors the router's RPC sequence: validate_only then commit."""
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})

        h.set_sae_global_clamps(
            decode_specs_raw=_raw_global_specs(),
            validate_only=True,
        )
        assert h._sae_clamp_manager.global_decode_specs == ()

        h.set_sae_global_clamps(decode_specs_raw=_raw_global_specs())
        assert len(h._sae_clamp_manager.global_decode_specs) == 1

    def test_clear_drops_both_tiers(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        h.set_sae_global_clamps(
            prefill_specs_raw=_raw_global_specs(),
            decode_specs_raw=_raw_global_specs(feature_idx=1),
        )

        h.clear_sae_global_clamps()

        assert h._sae_clamp_manager.global_prefill_specs == ()
        assert h._sae_clamp_manager.global_decode_specs == ()

    def test_status_reports_json_safe_view(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        h.set_sae_global_clamps(prefill_specs_raw=_raw_global_specs())

        status = h.get_sae_global_clamps_status()

        assert status["decode"] == []
        (spec,) = status["prefill"]
        assert spec["module_name"] == "g"
        entry = spec["clamps"]["post_block"]["20"][0]
        assert entry["feature_idx"] == 34
        assert entry["kind"] == "absolute"
        assert entry["value"] == 5.0


class TestCoerceSaeWeightsWire:
    """Wire-form ``sae_weights`` payloads attach correctly.

    Torch tensors do not survive the ``collective_rpc`` msgpack hop, so
    broadcast payloads may carry weights as ``"layer:hook"``-keyed maps
    of packed ``{dtype, shape, data}`` dicts (the Rust frontend's form,
    ``data`` base64 or raw bytes) or as ``[dtype, shape, buffer]``
    triples (a degraded in-process tensor).  The register branches
    normalize them via ``_coerce_sae_weights_wire`` before attach.
    """

    @staticmethod
    def _packed(t: torch.Tensor, *, data_as_bytes: bool = False) -> dict:
        raw = t.contiguous().view(torch.uint8).numpy().tobytes()
        return {
            "dtype": str(t.dtype).removeprefix("torch."),
            "shape": list(t.shape),
            "data": raw if data_as_bytes else base64.b64encode(raw).decode(),
        }

    def _wire_delta_weights(self, *, data_as_bytes: bool = False) -> dict:
        return {
            "20:post_block": {
                "encoder_weight": self._packed(
                    torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
                    data_as_bytes=data_as_bytes,
                ),
                "encoder_bias": self._packed(
                    torch.tensor([0.5, 0.25]), data_as_bytes=data_as_bytes
                ),
                "decoder_weight": self._packed(
                    torch.tensor([[9.0, 8.0, 7.0, 6.0], [5.0, 4.0, 3.0, 2.0]]),
                    data_as_bytes=data_as_bytes,
                ),
            }
        }

    def test_delta_string_keys_and_base64_packed_tensors_attach(self):
        h = _RichHarness(hidden_size=4)
        payload = _sae_payload(d_model=4, clampable=(0, 1))
        payload["sae_weights"] = self._wire_delta_weights()
        h.register_steering_modules({"g": payload})
        site = h._steerable_layers_cache[20]
        enc_w = getattr(
            site, HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        enc_b = getattr(
            site, HOOK_POINT_SAE_ENCODER_BIAS_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        dec_w = getattr(
            site, HOOK_POINT_SAE_DECODER_WEIGHT_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert torch.equal(
            enc_w, torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        )
        assert torch.equal(enc_b, torch.tensor([0.5, 0.25]))
        assert torch.equal(
            dec_w, torch.tensor([[9.0, 8.0, 7.0, 6.0], [5.0, 4.0, 3.0, 2.0]])
        )

    def test_delta_tuple_keys_and_bytes_data_attach(self):
        # Tuple site keys with raw-bytes ``data`` (the packed form after a
        # binary-preserving hop) coerce and attach.
        h = _RichHarness(hidden_size=4)
        payload = _sae_payload(d_model=4, clampable=(0, 1))
        payload["sae_weights"] = {
            (20, "post_block"): self._wire_delta_weights(data_as_bytes=True)[
                "20:post_block"
            ]
        }
        h.register_steering_modules({"g": payload})
        site = h._steerable_layers_cache[20]
        enc_b = getattr(
            site, HOOK_POINT_SAE_ENCODER_BIAS_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert torch.equal(enc_b, torch.tensor([0.5, 0.25]))

    def test_degraded_tensor_triples_attach(self):
        # Torch tensors sent through the Python API server's own
        # collective_rpc broadcast degrade to [dtype, shape, memoryview]
        # triples on arrival; the coercion rebuilds them (the latent
        # online-register bug this helper fixes).
        h = _RichHarness(hidden_size=4)
        payload = _sae_payload(d_model=4, clampable=(0, 1))

        def triple(t: torch.Tensor) -> list:
            return [
                str(t.dtype).removeprefix("torch."),
                list(t.shape),
                memoryview(t.contiguous().view(torch.uint8).numpy().tobytes()),
            ]

        payload["sae_weights"] = {
            (20, "post_block"): {
                "encoder_weight": triple(torch.full((2, 4), 3.0)),
                "encoder_bias": triple(torch.tensor([1.0, 2.0])),
                "decoder_weight": triple(torch.full((2, 4), 4.0)),
            }
        }
        h.register_steering_modules({"g": payload})
        site = h._steerable_layers_cache[20]
        enc_w = getattr(
            site, HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert torch.equal(enc_w, torch.full((2, 4), 3.0))

    def test_bfloat16_packed_tensor_attaches(self):
        h = _RichHarness(hidden_size=4)
        payload = _sae_payload(d_model=4, clampable=(0, 1))
        wire = self._wire_delta_weights()
        wire["20:post_block"]["encoder_weight"] = self._packed(
            torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.bfloat16
            )
        )
        payload["sae_weights"] = wire
        h.register_steering_modules({"g": payload})
        site = h._steerable_layers_cache[20]
        enc_w = getattr(
            site, HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert torch.equal(
            enc_w, torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        )

    def test_full_reconstruction_wire_weights_attach(self):
        from vllm.model_executor.layers.sae_full_reconstruction import (
            HOOK_POINT_FR_DECODER_BIAS_ATTR,
            HOOK_POINT_FR_ENCODER_WEIGHT_ATTR,
        )
        from vllm.v1.worker.sae_full_reconstruction_manager import (
            SAEFullReconstructionManager,
        )

        h = _RichHarness(hidden_size=4)
        # Graft the full-reconstruction surface the FR register branch
        # requires (normally set up by ``_init_steering_state``).
        h._sae_fr_module_registry = {}
        h._sae_fr_steerable_sites = {}
        h._req_sae_fr_phase = {}
        h._sae_fr_clamp_manager = SAEFullReconstructionManager(4)
        h._sae_fr_rows_scratch = None
        h._sae_fr_index_pinned = None

        d_sae, d_model = 8, 4
        payload = _sae_payload(d_model=d_model, d_sae=d_sae, clampable=(0, 1))
        payload["kind"] = "sae_full_reconstruction"
        payload["sae_weights"] = {
            "20:post_block": {
                "encoder_weight": self._packed(torch.full((d_sae, d_model), 2.0)),
                "encoder_bias": self._packed(torch.zeros(d_sae)),
                "decoder_weight": self._packed(torch.full((d_sae, d_model), 3.0)),
                "decoder_bias": self._packed(torch.tensor([0.1, 0.2, 0.3, 0.4])),
            }
        }
        h.register_steering_modules({"g": payload})
        layer = h._steerable_layers_cache[20]
        enc_w = getattr(
            layer, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        dec_b = getattr(
            layer, HOOK_POINT_FR_DECODER_BIAS_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert torch.equal(enc_w, torch.full((d_sae, d_model), 2.0))
        assert torch.equal(dec_b, torch.tensor([0.1, 0.2, 0.3, 0.4]))

    def test_malformed_packed_dict_rejected_and_rolled_back(self):
        h = _RichHarness(hidden_size=4)
        payload = _sae_payload(d_model=4, clampable=(0, 1))
        wire = self._wire_delta_weights()
        del wire["20:post_block"]["encoder_weight"]["data"]
        payload["sae_weights"] = wire
        with pytest.raises(ValueError, match="missing key"):
            h.register_steering_modules({"g": payload})
        # Atomic register-and-attach: the entry rolled back.
        assert "g" not in h._sae_module_registry
        site = h._steerable_layers_cache[20]
        assert not sae_buffers_attached(site, SteeringHookPoint.POST_BLOCK)

    def test_wrong_byte_length_rejected(self):
        h = _RichHarness(hidden_size=4)
        payload = _sae_payload(d_model=4, clampable=(0, 1))
        wire = self._wire_delta_weights()
        wire["20:post_block"]["encoder_bias"]["shape"] = [3]  # 2 encoded
        payload["sae_weights"] = wire
        with pytest.raises(ValueError, match="byte length"):
            h.register_steering_modules({"g": payload})
        assert "g" not in h._sae_module_registry

    def test_bad_site_key_rejected(self):
        h = _RichHarness(hidden_size=4)
        payload = _sae_payload(d_model=4, clampable=(0, 1))
        wire = self._wire_delta_weights()
        payload["sae_weights"] = {"post_block": wire["20:post_block"]}
        with pytest.raises(ValueError, match="site key"):
            h.register_steering_modules({"g": payload})
        assert "g" not in h._sae_module_registry

    def test_unrecoverable_value_rejected_with_clear_error(self):
        # A dangling aux-buffer index (large tensors degrade to a bare
        # int across the hop) cannot be recovered worker-side.
        h = _RichHarness(hidden_size=4)
        payload = _sae_payload(d_model=4, clampable=(0, 1))
        wire = self._wire_delta_weights()
        wire["20:post_block"]["encoder_weight"] = 7
        payload["sae_weights"] = wire
        with pytest.raises(ValueError, match="packed"):
            h.register_steering_modules({"g": payload})
        assert "g" not in h._sae_module_registry
