# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the request-level metadata channel.

``RequestMetadata`` carries per-request host-side fields (currently the
conversation id) distinct from sampling parameters. These tests pin the
defaults, the ``is_empty`` helper, and the msgspec round-trip across the
engine IPC boundary.
"""

from __future__ import annotations

from vllm.v1.request_metadata import RequestMetadata
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def test_defaults_are_empty():
    meta = RequestMetadata()
    assert meta.conversation_id is None
    assert meta.is_empty()


def test_conversation_id_round_trip():
    meta = RequestMetadata(conversation_id="conv-42")
    assert meta.conversation_id == "conv-42"
    assert not meta.is_empty()


def test_is_frozen():
    meta = RequestMetadata(conversation_id="conv-1")
    try:
        meta.conversation_id = "conv-2"  # type: ignore[misc]
    except (AttributeError, TypeError):
        return
    raise AssertionError("RequestMetadata should be immutable")


def test_msgspec_round_trip_with_value():
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(RequestMetadata)
    meta = RequestMetadata(conversation_id="conv-7")
    decoded = decoder.decode(encoder.encode(meta))
    assert decoded == meta
    assert decoded.conversation_id == "conv-7"


def test_msgspec_round_trip_empty():
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(RequestMetadata)
    meta = RequestMetadata()
    decoded = decoder.decode(encoder.encode(meta))
    assert decoded == meta
    assert decoded.is_empty()
