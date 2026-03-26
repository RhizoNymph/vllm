# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for get_num_decode_tokens utility."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

import vllm.forward_context as fc_module
from vllm.forward_context import get_num_decode_tokens


class TestGetNumDecodeTokens:
    """Test the get_num_decode_tokens forward context utility."""

    def test_returns_default_when_context_unavailable(self):
        """No forward context set -> return default."""
        with patch.object(fc_module, "_forward_context", None):
            assert get_num_decode_tokens(42) == 42

    def test_returns_default_when_attn_metadata_is_none(self):
        ctx = SimpleNamespace(attn_metadata=None)
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(99) == 99

    def test_dict_layout_with_num_decode_tokens(self):
        """Standard v1 layout: dict[str, AttentionMetadata]."""
        layer_meta = SimpleNamespace(num_decode_tokens=7)
        ctx = SimpleNamespace(attn_metadata={"layer0": layer_meta})
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(100) == 7

    def test_dict_layout_without_num_decode_tokens_attr(self):
        """Backend metadata lacks the attribute -> return default."""
        layer_meta = SimpleNamespace()  # no num_decode_tokens
        ctx = SimpleNamespace(attn_metadata={"layer0": layer_meta})
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(50) == 50

    def test_list_layout_dbo(self):
        """DBO layout: list[dict[str, AttentionMetadata]]."""
        layer_meta = SimpleNamespace(num_decode_tokens=3)
        ctx = SimpleNamespace(attn_metadata=[{"layer0": layer_meta}])
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(100) == 3

    def test_empty_list_layout(self):
        """DBO layout with empty list -> return default."""
        ctx = SimpleNamespace(attn_metadata=[])
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(25) == 25

    def test_empty_dict_layout(self):
        """Empty dict metadata -> return default."""
        ctx = SimpleNamespace(attn_metadata={})
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(30) == 30

    def test_list_with_empty_first_dict(self):
        """DBO list where first microbatch dict is empty -> default."""
        ctx = SimpleNamespace(attn_metadata=[{}])
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(10) == 10

    def test_num_decode_tokens_zero(self):
        """Zero decode tokens is a valid value (all-prefill batch)."""
        layer_meta = SimpleNamespace(num_decode_tokens=0)
        ctx = SimpleNamespace(attn_metadata={"layer0": layer_meta})
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(100) == 0

    def test_multiple_layers_returns_first(self):
        """With multiple layers, returns the first one encountered."""
        meta_a = SimpleNamespace(num_decode_tokens=5)
        meta_b = SimpleNamespace(num_decode_tokens=5)
        ctx = SimpleNamespace(attn_metadata={"layer_a": meta_a, "layer_b": meta_b})
        with patch.object(fc_module, "_forward_context", ctx):
            # All layers in a batch have the same num_decode_tokens
            assert get_num_decode_tokens(100) == 5

    # -- FlashAttention fallback path tests --

    def test_flash_attn_pure_decode_batch(self):
        """max_query_len=1 means all tokens are decode -> num_actual_tokens."""
        layer_meta = SimpleNamespace(
            max_query_len=1,
            num_actual_tokens=10,
        )
        ctx = SimpleNamespace(attn_metadata={"layer0": layer_meta})
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(0) == 10

    def test_flash_attn_mixed_batch(self):
        """Mixed batch: first 2 requests are decodes (qlen=1), third is prefill."""
        # query_start_loc = [0, 1, 2, 7] means:
        #   request 0: tokens [0,1) -> qlen=1  (decode)
        #   request 1: tokens [1,2) -> qlen=1  (decode)
        #   request 2: tokens [2,7) -> qlen=5  (prefill)
        layer_meta = SimpleNamespace(
            max_query_len=5,
            num_actual_tokens=7,
            query_start_loc=torch.tensor([0, 1, 2, 7]),
        )
        ctx = SimpleNamespace(attn_metadata={"layer0": layer_meta})
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(0) == 2

    def test_flash_attn_all_prefill(self):
        """All-prefill batch: first request has qlen>1 -> returns 0."""
        layer_meta = SimpleNamespace(
            max_query_len=5,
            num_actual_tokens=5,
            query_start_loc=torch.tensor([0, 5]),
        )
        ctx = SimpleNamespace(attn_metadata={"layer0": layer_meta})
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(0) == 0

    def test_flash_attn_no_fallback_fields(self):
        """Metadata without num_decode_tokens or fallback fields -> default."""
        layer_meta = SimpleNamespace()
        ctx = SimpleNamespace(attn_metadata={"layer0": layer_meta})
        with patch.object(fc_module, "_forward_context", ctx):
            assert get_num_decode_tokens(77) == 77
