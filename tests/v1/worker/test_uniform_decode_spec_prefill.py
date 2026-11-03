# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A prompt chunk with the uniform-decode length must not make a speculative
batch "uniform decode": the FULL cudagraph for that shape replays the GDN
spec-decode path off persistent metadata the builder does not refresh for a
batch with a prompt chunk, which corrupts the recurrent state of every request
in the step."""

from types import SimpleNamespace

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _count(
    num_spec_tokens: int,
    num_scheduled: dict[str, int],
    drafts: dict[str, list[int]],
) -> int | None:
    runner = SimpleNamespace(uniform_decode_query_len=1 + num_spec_tokens)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens=num_scheduled,
        scheduled_spec_decode_tokens=drafts,
    )
    return GPUModelRunner._count_full_spec_decode_reqs(
        runner,  # type: ignore[arg-type]
        scheduler_output,  # type: ignore[arg-type]
    )


def _uniform(num_scheduled: dict[str, int], drafts: dict[str, list[int]], k: int):
    tokens = list(num_scheduled.values())
    return GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=max(tokens),
        uniform_decode_query_len=1 + k,
        num_tokens=sum(tokens),
        num_reqs=len(tokens),
        num_spec_decode_reqs=_count(k, num_scheduled, drafts),
    )


def test_all_full_spec_decodes_are_uniform():
    assert _uniform({"a": 2, "b": 2}, {"a": [5], "b": [6]}, k=1)


def test_prompt_chunk_of_decode_length_is_not_uniform():
    # "b" is the 2-token tail of its prompt (MTP k=1): decode-shaped, no drafts.
    assert not _uniform({"a": 2, "b": 2}, {"a": [5]}, k=1)


def test_prompt_chunk_of_decode_length_k3_is_not_uniform():
    assert not _uniform({"a": 4, "b": 4, "c": 4}, {"a": [1, 2, 3], "c": [4, 5, 6]}, k=3)


def test_short_draft_is_not_uniform():
    # Same token count as a full decode would need, but one draft short.
    assert not _uniform({"a": 4, "b": 4}, {"a": [1, 2, 3], "b": [4, 5]}, k=3)


def test_padded_new_decode_is_uniform():
    # A new request that starts in decode is padded with -1 drafts on purpose
    # to keep the batch on the uniform-decode cudagraph.
    assert _uniform({"a": 2, "b": 2}, {"a": [5], "b": [-1]}, k=1)


def test_without_spec_decode_shape_rule_is_unchanged():
    assert _count(0, {"a": 1, "b": 1}, {}) is None
    assert GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=1,
        uniform_decode_query_len=1,
        num_tokens=2,
        num_reqs=2,
        num_spec_decode_reqs=None,
    )


def test_force_uniform_decode_wins():
    assert GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=2,
        uniform_decode_query_len=2,
        num_tokens=4,
        num_reqs=2,
        force_uniform_decode=True,
        num_spec_decode_reqs=0,
    )
