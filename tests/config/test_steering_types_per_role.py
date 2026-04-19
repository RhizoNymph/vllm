# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the per-role normalization helper added in PR 4.

``normalize_to_per_role`` detects whether a steering spec is in the
flat form (hook-point keys) or the nested ``{"main": ..., "draft": ...}``
form and expands/passes it through. Flat specs tag along to both roles;
nested specs are returned verbatim (missing roles stay missing).
"""

from vllm.config.steering_types import (
    MODEL_ROLES,
    normalize_to_per_role,
)

_FLAT_SPEC = {"post_mlp": {0: [1.0, 2.0, 3.0]}}
_NESTED_MAIN_ONLY = {"main": {"post_mlp": {0: [1.0]}}}
_NESTED_BOTH = {
    "main": {"post_mlp": {0: [1.0]}},
    "draft": {"post_mlp": {0: [2.0]}},
}
_NESTED_DRAFT_ONLY = {"draft": {"post_mlp": {0: [2.0]}}}


class TestNormalizeToPerRole:
    def test_none_returns_none(self):
        assert normalize_to_per_role(None) is None

    def test_empty_dict_returns_none(self):
        assert normalize_to_per_role({}) is None

    def test_flat_spec_tags_along_to_both(self):
        result = normalize_to_per_role(_FLAT_SPEC)
        assert result is not None
        assert set(result.keys()) == set(MODEL_ROLES)
        # The very same spec object is installed under both keys
        # (tags-along is an identity mapping, not a copy).
        assert result["main"] == _FLAT_SPEC
        assert result["draft"] == _FLAT_SPEC

    def test_nested_main_only(self):
        result = normalize_to_per_role(_NESTED_MAIN_ONLY)
        assert result is not None
        assert set(result.keys()) == {"main"}
        assert result["main"] == _NESTED_MAIN_ONLY["main"]

    def test_nested_draft_only(self):
        result = normalize_to_per_role(_NESTED_DRAFT_ONLY)
        assert result is not None
        assert set(result.keys()) == {"draft"}
        assert result["draft"] == _NESTED_DRAFT_ONLY["draft"]

    def test_nested_both_roles(self):
        result = normalize_to_per_role(_NESTED_BOTH)
        assert result is not None
        assert set(result.keys()) == {"main", "draft"}
        assert result["main"] == _NESTED_BOTH["main"]
        assert result["draft"] == _NESTED_BOTH["draft"]

    def test_nested_with_empty_role_spec_drops_the_role(self):
        spec = {"main": {"post_mlp": {0: [1.0]}}, "draft": {}}
        result = normalize_to_per_role(spec)
        assert result is not None
        # Empty draft spec is truthy-but-empty; helper should drop it.
        assert set(result.keys()) == {"main"}

    def test_flat_form_detection_is_exact_for_role_keys_only(self):
        """A top-level key set that is a strict subset of role names uses
        nested form; any other key triggers flat form."""
        # ``{"main": ...}`` → nested (keys ⊆ {"main", "draft"}).
        assert normalize_to_per_role({"main": {}}) is None  # empty contents drop
        # ``{"post_mlp": ..., "main": ...}`` would be ambiguous but is
        # not a valid steering payload — hook-point names cannot be
        # "main"/"draft". We verify that a hook-point-only spec is
        # treated as flat form.
        assert normalize_to_per_role({"post_mlp": {0: [1.0]}}) is not None
