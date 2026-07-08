# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for clean/corrupt token-position alignment."""

import pytest

from vllm.entrypoints.serve.patch.alignment import align_token_positions

pytestmark = pytest.mark.cpu_test


class TestEqualLength:
    def test_identity_everywhere_even_where_tokens_differ(self):
        # France vs Japan at position 3: corresponding positions ARE the
        # causal-tracing pairing.
        a = align_token_positions([1, 2, 3, 40, 5], [1, 2, 3, 99, 5])
        assert a.is_identity
        assert a.mapping == {i: i for i in range(5)}
        assert a.unaligned == []


class TestUnequalLength:
    # clean: [1, 2, 30, 31, 4, 5]  (6 tokens; middle = 30, 31)
    # corrupt: [1, 2, 90, 4, 5]    (5 tokens; middle = 90)
    CLEAN = [1, 2, 30, 31, 4, 5]
    CORRUPT = [1, 2, 90, 4, 5]

    def test_prefix_identity(self):
        a = align_token_positions(self.CLEAN, self.CORRUPT)
        assert a.prefix_len == 2
        assert a.mapping[0] == 0 and a.mapping[1] == 1

    def test_suffix_shifted_by_length_delta(self):
        a = align_token_positions(self.CLEAN, self.CORRUPT)
        assert a.suffix_len == 2
        # corrupt pos 3 (token 4) -> clean pos 4; corrupt pos 4 -> clean pos 5
        assert a.mapping[3] == 4
        assert a.mapping[4] == 5

    def test_middle_unaligned(self):
        a = align_token_positions(self.CLEAN, self.CORRUPT)
        assert a.unaligned == [2]
        assert a.source_for(2) is None

    def test_longer_corrupt_negative_shift(self):
        a = align_token_positions(self.CORRUPT, self.CLEAN)  # swap roles
        assert a.mapping[0] == 0
        assert a.mapping[4] == 3 and a.mapping[5] == 4  # shift -1
        assert a.unaligned == [2, 3]

    def test_no_common_affixes_all_unaligned(self):
        a = align_token_positions([1, 2], [3, 4, 5])
        assert a.mapping == {}
        assert a.unaligned == [0, 1, 2]

    def test_summary_shape(self):
        s = align_token_positions(self.CLEAN, self.CORRUPT).summary()
        assert s == {
            "n_clean": 6,
            "n_corrupt": 5,
            "prefix_len": 2,
            "suffix_len": 2,
            "unaligned_positions": [2],
        }


class TestClientUsesSharedAlignment:
    """The client now imports the shared module rather than mirroring it; this
    guards that the client references the same implementation and that it
    behaves correctly across the alignment cases."""

    def test_client_imports_shared_symbol(self):
        from vllm.entrypoints.serve.patch import client

        assert client.align_token_positions is align_token_positions

    @pytest.mark.parametrize(
        "clean,corrupt",
        [
            ([1, 2, 3], [1, 2, 3]),
            ([1, 2, 3, 40, 5], [1, 2, 3, 99, 5]),
            ([1, 2, 30, 31, 4, 5], [1, 2, 90, 4, 5]),
            ([1, 2, 90, 4, 5], [1, 2, 30, 31, 4, 5]),
            ([1, 2], [3, 4, 5]),
            ([], []),
        ],
    )
    def test_alignment_maps_and_unaligned(self, clean, corrupt):
        a = align_token_positions(clean, corrupt)
        # Every dest position is either mapped or explicitly unaligned.
        assert set(a.mapping) | set(a.unaligned) == set(range(len(corrupt)))
        assert not (set(a.mapping) & set(a.unaligned))
        # Mapped positions never point past the clean run.
        assert all(0 <= v < len(clean) for v in a.mapping.values())
