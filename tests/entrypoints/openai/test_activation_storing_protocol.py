# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for activation storing phase 1: types, config, admission validation.

These tests exercise pure-data scaffolding only — no runner, no writer
pool, no custom op. They cover:

- Selector expansion: ``"all"``, explicit list, mixed ``{layers, ranges}``
  form, out-of-range rejection, inclusive-range semantics.
- Slugging: regex correctness, ``..`` / leading-``/`` / length-256
  rejection.
- Admission validation (:func:`validate_activation_storing`):
  feature-disabled rejection, TP/PP > 1 rejection, unknown-layer
  rejection, prefix-cache position rejection, byte-budget rejection.
- ``ActivationStoringSpec`` structural validation (unknown hook point,
  empty hooks, bad positions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from vllm.config.activation_storing import ActivationStoringConfig
from vllm.config.activation_storing_types import (
    ActivationStoringSlugError,
    ActivationStoringSpec,
    VALID_ACTIVATION_HOOK_NAMES,
    VALID_POSITION_KINDS,
    expand_hook_layers,
    resolve_positions,
    slug,
)
from vllm.entrypoints.openai.activation_storing_validation import (
    ActivationStoringContext,
    ActivationStoringValidationError,
    validate_activation_storing,
)


# ---------------------------------------------------------------------------
# Fake vllm config (avoids importing the real one so these tests stay fast)
# ---------------------------------------------------------------------------


@dataclass
class _FakeVllmConfig:
    """Minimal stand-in for ``VllmConfig`` that the validator actually uses.

    The validator only reads ``vllm_config.activation_storing_config``;
    it does not look at anything else. Using a throwaway dataclass keeps
    the tests from paying the full ``VllmConfig`` import cost.
    """

    activation_storing_config: ActivationStoringConfig | None


def _ctx(
    *,
    num_prompt_tokens: int = 16,
    num_computed_tokens: int = 0,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    num_hidden_layers: int = 32,
    hidden_size: int = 4096,
    element_size_bytes: int = 2,
) -> ActivationStoringContext:
    """Convenience constructor with sensible small-model defaults."""
    return ActivationStoringContext(
        num_prompt_tokens=num_prompt_tokens,
        num_computed_tokens=num_computed_tokens,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        element_size_bytes=element_size_bytes,
    )


def _enabled_config(**overrides: Any) -> _FakeVllmConfig:
    cfg = ActivationStoringConfig(
        root_path="/tmp/activations",
        **overrides,
    )
    return _FakeVllmConfig(activation_storing_config=cfg)


def _spec(**overrides: Any) -> ActivationStoringSpec:
    defaults: dict[str, Any] = dict(
        request_id="probe_0001",
        tag="capital-probe",
        hooks={"post_mlp": [12]},
        positions="last_prompt",
    )
    defaults.update(overrides)
    return ActivationStoringSpec(**defaults)


# ---------------------------------------------------------------------------
# Layer selector expansion
# ---------------------------------------------------------------------------


class TestExpandHookLayers:
    def test_all_shorthand(self):
        assert expand_hook_layers("all", 4) == [0, 1, 2, 3]

    def test_explicit_list(self):
        assert expand_hook_layers([3, 1, 2, 2], 4) == [1, 2, 3]

    def test_explicit_list_sorted_and_deduped(self):
        assert expand_hook_layers([5, 5, 1, 3, 1], 8) == [1, 3, 5]

    def test_mixed_form_layers_only(self):
        assert expand_hook_layers({"layers": [0, 5, 3]}, 8) == [0, 3, 5]

    def test_mixed_form_ranges_only_inclusive(self):
        # Inclusive on both ends: [[10, 20]] → 11 elements
        result = expand_hook_layers({"ranges": [[10, 20]]}, 32)
        assert result == list(range(10, 21))
        assert len(result) == 11

    def test_mixed_form_union(self):
        result = expand_hook_layers(
            {"layers": [1, 31], "ranges": [[10, 20], [25, 28]]},
            32,
        )
        assert result == [
            1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 31
        ]

    def test_mixed_form_dedupes_across_layers_and_ranges(self):
        result = expand_hook_layers(
            {"layers": [5, 10], "ranges": [[5, 7]]},
            16,
        )
        assert result == [5, 6, 7, 10]

    def test_out_of_range_explicit_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            expand_hook_layers([0, 32], 32)

    def test_out_of_range_negative_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            expand_hook_layers([-1, 2], 32)

    def test_out_of_range_in_ranges_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            expand_hook_layers({"ranges": [[30, 40]]}, 32)

    def test_range_reversed_rejected(self):
        with pytest.raises(ValueError, match="start > end"):
            expand_hook_layers({"ranges": [[20, 10]]}, 32)

    def test_bad_string_rejected(self):
        with pytest.raises(ValueError, match="must be 'all'"):
            expand_hook_layers("every", 4)

    def test_bad_dict_key_rejected(self):
        with pytest.raises(ValueError, match="unexpected keys"):
            expand_hook_layers(
                {"layers": [1], "nope": []},  # type: ignore[dict-item]
                4,
            )

    def test_empty_dict_form_rejected(self):
        with pytest.raises(ValueError, match="must set 'layers'"):
            expand_hook_layers({}, 4)

    def test_bool_rejected_as_layer_index(self):
        with pytest.raises(ValueError, match="must be an int"):
            expand_hook_layers([True, 1], 4)  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Position resolution
# ---------------------------------------------------------------------------


class TestResolvePositions:
    def test_last_prompt(self):
        assert resolve_positions("last_prompt", 10, 0) == [9]

    def test_all_prompt(self):
        assert resolve_positions("all_prompt", 4, 0) == [0, 1, 2, 3]

    def test_all_generated(self):
        assert resolve_positions("all_generated", 4, 3) == [4, 5, 6]

    def test_all(self):
        assert resolve_positions("all", 2, 2) == [0, 1, 2, 3]

    def test_explicit_list_bounds_checked(self):
        assert resolve_positions([3, 1, 1], 4, 0) == [1, 3]

    def test_explicit_list_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            resolve_positions([4], 4, 0)

    def test_bad_string(self):
        with pytest.raises(ValueError, match="must be one of"):
            resolve_positions("middle", 4, 0)


# ---------------------------------------------------------------------------
# Slugging
# ---------------------------------------------------------------------------


class TestSlug:
    def test_simple_identity(self):
        assert slug("probe_0001") == "probe_0001"

    def test_regex_replaces_unsafe_chars(self):
        assert slug("hello world") == "hello_world"
        assert slug("a/b/c") == "a_b_c"
        assert slug("foo:bar@baz") == "foo_bar_baz"

    def test_preserves_dot_dash_underscore(self):
        assert slug("v1.2.3-rc_4") == "v1.2.3-rc_4"

    def test_alphanumerics_preserved(self):
        assert slug("Llama3-8B.v1") == "Llama3-8B.v1"

    def test_dotdot_rejected(self):
        with pytest.raises(ActivationStoringSlugError, match=r"'\.\.'"):
            slug("../etc/passwd")

    def test_leading_slash_rejected(self):
        with pytest.raises(ActivationStoringSlugError, match="'/'"):
            slug("/absolute/path")

    def test_empty_rejected(self):
        with pytest.raises(ActivationStoringSlugError, match="non-empty"):
            slug("")

    def test_length_256_allowed(self):
        name = "a" * 256
        assert slug(name) == name

    def test_length_257_rejected(self):
        name = "a" * 257
        with pytest.raises(ActivationStoringSlugError, match="at most 256"):
            slug(name)


# ---------------------------------------------------------------------------
# ActivationStoringSpec structural validation
# ---------------------------------------------------------------------------


class TestActivationStoringSpecPostInit:
    def test_valid_minimal(self):
        spec = ActivationStoringSpec(
            request_id="r1",
            tag="t1",
            hooks={"post_mlp": [0]},
            positions="last_prompt",
        )
        assert spec.request_id == "r1"

    def test_empty_hooks_rejected(self):
        with pytest.raises(ValueError, match="non-empty dict"):
            ActivationStoringSpec(
                request_id="r",
                tag="t",
                hooks={},
                positions="last_prompt",
            )

    def test_unknown_hook_name_rejected(self):
        with pytest.raises(ValueError, match="not a.*valid hook"):
            ActivationStoringSpec(
                request_id="r",
                tag="t",
                hooks={"bogus": [0]},
                positions="last_prompt",
            )

    def test_unknown_position_kind_rejected(self):
        with pytest.raises(ValueError, match="string form must be one"):
            ActivationStoringSpec(
                request_id="r",
                tag="t",
                hooks={"post_mlp": [0]},
                positions="middle",
            )

    def test_empty_request_id_rejected(self):
        with pytest.raises(ValueError, match="request_id"):
            ActivationStoringSpec(
                request_id="",
                tag="t",
                hooks={"post_mlp": [0]},
                positions="last_prompt",
            )

    def test_hook_point_set_mirrors_steering(self):
        # The feature piggybacks on the steering hook-point enum. If
        # this set drifts out of sync with the real enum, we'll silently
        # accept captures at hooks steering doesn't know about and the
        # runner will error out at forward time. Pin it here.
        assert VALID_ACTIVATION_HOOK_NAMES == frozenset(
            {"pre_attn", "post_attn", "post_mlp"}
        )

    def test_position_kind_set(self):
        assert VALID_POSITION_KINDS == frozenset(
            {"last_prompt", "all_prompt", "all_generated", "all"}
        )


# ---------------------------------------------------------------------------
# Admission validation
# ---------------------------------------------------------------------------


class TestValidateActivationStoringHappy:
    def test_resolves_last_prompt(self):
        resolved = validate_activation_storing(
            _spec(),
            _enabled_config(),
            _ctx(num_prompt_tokens=16),
        )
        assert resolved.position_kind == "last_prompt"
        assert resolved.positions == [15]
        assert resolved.hooks == {"post_mlp": [12]}
        assert resolved.request_id_slug == "probe_0001"
        assert resolved.tag_slug == "capital-probe"

    def test_resolves_all_prompt(self):
        resolved = validate_activation_storing(
            _spec(positions="all_prompt"),
            _enabled_config(),
            _ctx(num_prompt_tokens=4),
        )
        assert resolved.positions == [0, 1, 2, 3]
        assert resolved.position_kind == "all_prompt"

    def test_all_generated_stays_symbolic(self):
        resolved = validate_activation_storing(
            _spec(positions="all_generated"),
            _enabled_config(),
            _ctx(),
        )
        assert resolved.positions == "all_generated"
        assert resolved.position_kind == "all_generated"
        # No byte estimate for symbolic positions.
        assert resolved.estimated_bytes == 0

    def test_all_stays_symbolic(self):
        resolved = validate_activation_storing(
            _spec(positions="all"),
            _enabled_config(),
            _ctx(),
        )
        assert resolved.positions == "all"
        assert resolved.position_kind == "all"

    def test_explicit_positions(self):
        resolved = validate_activation_storing(
            _spec(positions=[1, 3, 5]),
            _enabled_config(),
            _ctx(num_prompt_tokens=8),
        )
        assert resolved.positions == [1, 3, 5]
        assert resolved.position_kind == "explicit"

    def test_mixed_layer_form_resolved(self):
        resolved = validate_activation_storing(
            _spec(hooks={"post_mlp": {"layers": [1], "ranges": [[10, 12]]}}),
            _enabled_config(),
            _ctx(num_hidden_layers=32),
        )
        assert resolved.hooks == {"post_mlp": [1, 10, 11, 12]}

    def test_all_hook_layers_resolved(self):
        resolved = validate_activation_storing(
            _spec(hooks={"post_mlp": "all"}),
            _enabled_config(),
            _ctx(num_hidden_layers=4),
        )
        assert resolved.hooks == {"post_mlp": [0, 1, 2, 3]}

    def test_byte_estimate_math(self):
        # 1 position × 1 hook × 2 layers × 4096 hidden × 2 bytes = 16384
        resolved = validate_activation_storing(
            _spec(
                hooks={"post_mlp": [10, 20]},
                positions="last_prompt",
            ),
            _enabled_config(),
            _ctx(num_prompt_tokens=8, hidden_size=4096, element_size_bytes=2),
        )
        assert resolved.estimated_bytes == 1 * 2 * 4096 * 2  # 16384


class TestValidateActivationStoringRejections:
    def test_config_disabled(self):
        with pytest.raises(
            ActivationStoringValidationError,
            match="does not have activation storing enabled",
        ):
            validate_activation_storing(
                _spec(),
                _FakeVllmConfig(activation_storing_config=None),
                _ctx(),
            )

    def test_config_root_path_none(self):
        # ``ActivationStoringConfig()`` defaults to root_path=None and
        # should still be treated as disabled.
        with pytest.raises(
            ActivationStoringValidationError,
            match="does not have activation storing enabled",
        ):
            cfg = ActivationStoringConfig()
            validate_activation_storing(
                _spec(),
                _FakeVllmConfig(activation_storing_config=cfg),
                _ctx(),
            )

    def test_tp_greater_than_one_rejected(self):
        with pytest.raises(
            ActivationStoringValidationError,
            match="tensor_parallel_size=1",
        ):
            validate_activation_storing(
                _spec(),
                _enabled_config(),
                _ctx(tensor_parallel_size=2),
            )

    def test_pp_greater_than_one_rejected(self):
        with pytest.raises(
            ActivationStoringValidationError,
            match="pipeline_parallel_size=1",
        ):
            validate_activation_storing(
                _spec(),
                _enabled_config(),
                _ctx(pipeline_parallel_size=2),
            )

    def test_unknown_layer_rejected(self):
        with pytest.raises(
            ActivationStoringValidationError, match="out of range"
        ):
            validate_activation_storing(
                _spec(hooks={"post_mlp": [100]}),
                _enabled_config(),
                _ctx(num_hidden_layers=32),
            )

    def test_negative_layer_rejected(self):
        with pytest.raises(
            ActivationStoringValidationError, match="out of range"
        ):
            validate_activation_storing(
                _spec(hooks={"post_mlp": [-1]}),
                _enabled_config(),
                _ctx(num_hidden_layers=32),
            )

    def test_prefix_cache_position_rejected(self):
        # positions below num_computed_tokens were served from cache and
        # never forwarded — reject rather than silently drop.
        with pytest.raises(
            ActivationStoringValidationError,
            match="served from the prefix cache",
        ):
            validate_activation_storing(
                _spec(positions=[0, 1, 2]),
                _enabled_config(),
                _ctx(num_prompt_tokens=16, num_computed_tokens=4),
            )

    def test_last_prompt_not_in_prefix_cache(self):
        # last_prompt = num_prompt_tokens - 1 is always above num_computed
        # when we're actually going to forward anything — sanity-check.
        validate_activation_storing(
            _spec(positions="last_prompt"),
            _enabled_config(),
            _ctx(num_prompt_tokens=16, num_computed_tokens=4),
        )

    def test_byte_budget_cap_exceeded(self):
        # 1 position × 1 hook × 32 layers × 4096 hidden × 2 bytes = 262144
        cfg = _enabled_config(max_bytes_per_request=1024)
        with pytest.raises(
            ActivationStoringValidationError,
            match="exceeds --activation-storing-max-bytes-per-request",
        ):
            validate_activation_storing(
                _spec(hooks={"post_mlp": "all"}),
                cfg,
                _ctx(
                    num_prompt_tokens=8,
                    num_hidden_layers=32,
                    hidden_size=4096,
                    element_size_bytes=2,
                ),
            )

    def test_byte_budget_zero_means_unbounded(self):
        cfg = _enabled_config(max_bytes_per_request=0)
        validate_activation_storing(
            _spec(hooks={"post_mlp": "all"}),
            cfg,
            _ctx(
                num_prompt_tokens=8,
                num_hidden_layers=32,
                hidden_size=4096,
                element_size_bytes=2,
            ),
        )

    def test_bad_tag_rejected(self):
        # `..` in tag
        with pytest.raises(
            ActivationStoringValidationError, match="tag is invalid"
        ):
            validate_activation_storing(
                _spec(tag="../etc"),
                _enabled_config(),
                _ctx(),
            )

    def test_bad_request_id_rejected(self):
        # leading `/` in request_id
        with pytest.raises(
            ActivationStoringValidationError, match="request_id is invalid"
        ):
            validate_activation_storing(
                _spec(request_id="/absolute"),
                _enabled_config(),
                _ctx(),
            )


# ---------------------------------------------------------------------------
# ActivationStoringConfig defaults
# ---------------------------------------------------------------------------


class TestActivationStoringConfigDefaults:
    def test_defaults(self):
        cfg = ActivationStoringConfig()
        assert cfg.root_path is None
        assert cfg.writer_queue_size == 1024
        assert cfg.writer_timeout_seconds == 180
        assert cfg.writer_threads == 4
        assert cfg.on_collision == "overwrite"
        assert cfg.max_bytes_per_request == 0

    def test_compute_hash_changes_when_enabled(self):
        disabled = ActivationStoringConfig()
        enabled = ActivationStoringConfig(root_path="/tmp/a")
        assert disabled.compute_hash() != enabled.compute_hash()

    def test_compute_hash_ignores_root_path_specifics(self):
        # The hash should capture "enabled vs disabled" but not the root
        # path itself (two different roots on the same server don't
        # change the compile graph, only enable/disable does).
        a = ActivationStoringConfig(root_path="/mnt/a")
        b = ActivationStoringConfig(root_path="/mnt/b")
        assert a.compute_hash() == b.compute_hash()
