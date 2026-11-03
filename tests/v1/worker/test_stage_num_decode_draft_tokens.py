# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GPUModelRunner._stage_num_decode_draft_tokens: which rows the
linear-attention builders treat as spec-decode rows."""

from types import SimpleNamespace

import numpy as np

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _Buffer:
    def __init__(self, size: int):
        self.np = np.full(size, 7, dtype=np.int32)
        self.copied = 0

    def copy_to_gpu(self) -> None:
        self.copied += 1


def _stage(
    req_ids: list[str],
    num_computed: list[int],
    num_prompt: list[int],
    num_scheduled: list[int],
    drafts: dict[str, list[int]],
    buffer_size: int = 8,
) -> _Buffer:
    buf = _Buffer(buffer_size)
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            num_computed_tokens_cpu=np.array(num_computed, dtype=np.int32),
            num_prompt_tokens=np.array(num_prompt, dtype=np.int32),
        ),
        num_decode_draft_tokens=buf,
    )
    scheduler_output = SimpleNamespace(scheduled_spec_decode_tokens=drafts)
    GPUModelRunner._stage_num_decode_draft_tokens(
        runner,  # type: ignore[arg-type]
        scheduler_output,  # type: ignore[arg-type]
        np.array(num_scheduled, dtype=np.int32),
        len(req_ids),
    )
    return buf


def test_decode_without_draft_is_a_spec_row():
    # "a" decodes with no draft, "b" verifies 3 drafts.
    buf = _stage(
        ["a", "b"],
        num_computed=[120, 90],
        num_prompt=[100, 80],
        num_scheduled=[1, 4],
        drafts={"b": [5, 6, 7]},
    )
    assert buf.np[:2].tolist() == [0, 3]
    assert (buf.np[2:] == -1).all()
    assert buf.copied == 1


def test_no_drafts_anywhere_still_marks_decodes():
    buf = _stage(
        ["a", "b"],
        num_computed=[120, 90],
        num_prompt=[100, 80],
        num_scheduled=[1, 1],
        drafts={},
    )
    assert buf.np[:2].tolist() == [0, 0]


def test_prefill_rows_are_not_spec_rows():
    # "a" is a one-token final prompt chunk, "b" a longer prompt chunk, "c" a
    # first decode right after its prompt.
    buf = _stage(
        ["a", "b", "c"],
        num_computed=[99, 0, 64],
        num_prompt=[100, 80, 64],
        num_scheduled=[1, 80, 1],
        drafts={},
    )
    assert buf.np[:3].tolist() == [-1, -1, 0]


def test_padded_spec_row_of_a_new_request():
    # A new request whose prompt is a full prefix hit but its last token is
    # padded to the uniform decode size with -1 drafts: a spec row with
    # the padded draft count, as before.
    buf = _stage(
        ["a"],
        num_computed=[99],
        num_prompt=[100],
        num_scheduled=[4],
        drafts={"a": [-1, -1, -1]},
    )
    assert buf.np[:1].tolist() == [3]


def test_truncated_drafts_on_a_prefill_chunk_are_not_spec_rows():
    buf = _stage(
        ["a"],
        num_computed=[50],
        num_prompt=[100],
        num_scheduled=[20],
        drafts={"a": [-1, -1, -1]},
    )
    assert buf.np[:1].tolist() == [-1]
