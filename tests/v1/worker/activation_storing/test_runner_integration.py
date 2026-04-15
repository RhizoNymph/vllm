# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end runner integration tests for activation storing.

Phase 4 ships the runner-side plumbing that turns a ``SamplingParams.activation_storing``
spec into real files on disk plus a populated
``RequestOutput.activation_storage``. These tests exercise the whole
path through the offline ``LLM`` entrypoint against a tiny
steering-instrumented model (see
``tests/v1/worker/test_steering_manager.py`` for the standard
``LLM`` setup used throughout the Phase 3 tests).

CUDA-gated tests are skipped on CPU-only boxes. They are CI-verified
on the GPU runners.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.config.activation_storing_types import (
    ActivationStoringSpec,
    CaptureResult,
)


_CUDA_REASON = "requires a CUDA device to run a real forward pass"
# Use a small steerable model so the tests run in a few seconds on a
# GPU runner. ``google/gemma-3-1b-it`` is the model the sliding-window
# e2e tests use (see tests/v1/e2e/general/test_correctness_sliding_window.py).
_STEERABLE_MODEL = "google/gemma-3-1b-it"


def _make_spec(
    request_id: str,
    tag: str,
    hooks: dict[str, list[int] | str],
    positions: str | list[int],
) -> ActivationStoringSpec:
    return ActivationStoringSpec(
        request_id=request_id,
        tag=tag,
        hooks=hooks,
        positions=positions,
    )


def _bin_json_pairs(root: pathlib.Path) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Return every ``(bin, json)`` pair discovered under ``root``.

    The captures land under ``{root}/{model}/{dtype}/{tag}/{layer}/{hook}/``
    so a simple recursive glob is enough.
    """
    bins = sorted(root.glob("**/*.bin"))
    pairs: list[tuple[pathlib.Path, pathlib.Path]] = []
    for bin_path in bins:
        sidecar = bin_path.with_suffix(".json")
        assert sidecar.exists(), f"missing sidecar for {bin_path}"
        pairs.append((bin_path, sidecar))
    return pairs


def _load_sidecar(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _assert_no_captures(root: pathlib.Path) -> None:
    bins = list(root.glob("**/*.bin"))
    jsons = list(root.glob("**/*.json"))
    assert not bins, f"unexpected .bin files under {root}: {bins}"
    assert not jsons, f"unexpected .json files under {root}: {jsons}"


def _run_with_capture(
    llm: "LLM",
    prompt: str,
    spec: ActivationStoringSpec | None,
    max_tokens: int = 1,
):
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        activation_storing=spec,
    )
    outputs = llm.generate([prompt], sp)
    assert len(outputs) == 1
    return outputs[0]


@pytest.fixture(scope="module")
def enabled_llm(tmp_path_factory):
    """An ``LLM`` with activation storing enabled.

    Scoped module-wide to amortize model-load latency across the test
    cases that need a warm engine. The capture root is test-local.
    """
    if not torch.cuda.is_available():
        pytest.skip(_CUDA_REASON)
    root = tmp_path_factory.mktemp("act_store_enabled")
    llm = LLM(
        model=_STEERABLE_MODEL,
        activation_storing=str(root),
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=512,
    )
    # Stash the root on the llm for tests to read.
    llm.__dict__["_capture_root"] = root  # type: ignore[attr-defined]
    return llm


# ---------------------------------------------------------------------------
# 1. last_prompt, single layer, single hook
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_REASON)
def test_last_prompt_single_layer_single_hook(enabled_llm):
    root: pathlib.Path = enabled_llm._capture_root
    spec = _make_spec(
        request_id="last_prompt_0",
        tag="phase4_test_1",
        hooks={"post_mlp": [6]},
        positions="last_prompt",
    )
    output = _run_with_capture(enabled_llm, "The capital of France is", spec)
    assert output.activation_storage is not None
    assert output.activation_storage.status == "ok"
    pairs = [
        (b, j)
        for (b, j) in _bin_json_pairs(root)
        if "phase4_test_1" in str(b)
    ]
    assert len(pairs) == 1
    bin_path, json_path = pairs[0]
    meta = _load_sidecar(json_path)
    assert meta["request_id"] == "last_prompt_0"
    assert meta["tag"] == "phase4_test_1"
    assert meta["layer"] == 6
    assert meta["hook"] == "post_mlp"
    assert meta["position_kind"] == "last_prompt"
    assert len(meta["positions"]) == 1
    assert meta["shape"] == [1, meta["shape"][1]]


# ---------------------------------------------------------------------------
# 2. multiple layers, one hook
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_REASON)
def test_multiple_layers_one_hook(enabled_llm):
    root: pathlib.Path = enabled_llm._capture_root
    spec = _make_spec(
        request_id="multi_layer_0",
        tag="phase4_test_2",
        hooks={"post_mlp": [3, 6]},
        positions="last_prompt",
    )
    output = _run_with_capture(enabled_llm, "Hello world", spec)
    assert output.activation_storage is not None
    assert output.activation_storage.status == "ok"
    pairs = [
        (b, j)
        for (b, j) in _bin_json_pairs(root)
        if "phase4_test_2" in str(b)
    ]
    assert len(pairs) == 2
    layers = sorted(_load_sidecar(j)["layer"] for _, j in pairs)
    assert layers == [3, 6]


# ---------------------------------------------------------------------------
# 3. multiple hooks, one layer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_REASON)
def test_multiple_hooks_one_layer(enabled_llm):
    root: pathlib.Path = enabled_llm._capture_root
    spec = _make_spec(
        request_id="multi_hook_0",
        tag="phase4_test_3",
        hooks={"pre_attn": [6], "post_mlp": [6]},
        positions="last_prompt",
    )
    output = _run_with_capture(enabled_llm, "Four score and seven", spec)
    assert output.activation_storage is not None
    assert output.activation_storage.status == "ok"
    pairs = [
        (b, j)
        for (b, j) in _bin_json_pairs(root)
        if "phase4_test_3" in str(b)
    ]
    assert len(pairs) == 2
    hooks = sorted(_load_sidecar(j)["hook"] for _, j in pairs)
    assert hooks == ["post_mlp", "pre_attn"]


# ---------------------------------------------------------------------------
# 4. all_prompt positions
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_REASON)
def test_all_prompt_positions(enabled_llm):
    root: pathlib.Path = enabled_llm._capture_root
    spec = _make_spec(
        request_id="all_prompt_0",
        tag="phase4_test_4",
        hooks={"post_mlp": [6]},
        positions="all_prompt",
    )
    output = _run_with_capture(
        enabled_llm, "One two three four five", spec, max_tokens=1
    )
    assert output.activation_storage is not None
    assert output.activation_storage.status == "ok"
    pairs = [
        (b, j)
        for (b, j) in _bin_json_pairs(root)
        if "phase4_test_4" in str(b)
    ]
    assert len(pairs) == 1
    _, json_path = pairs[0]
    meta = _load_sidecar(json_path)
    num_prompt_tokens = len(meta["prompt_token_ids"])
    assert meta["shape"][0] == num_prompt_tokens
    assert meta["position_kind"] == "all_prompt"


# ---------------------------------------------------------------------------
# 5. all_generated positions, multi-step
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_REASON)
def test_all_generated_positions_multistep(enabled_llm):
    root: pathlib.Path = enabled_llm._capture_root
    spec = _make_spec(
        request_id="all_gen_0",
        tag="phase4_test_5",
        hooks={"post_mlp": [6]},
        positions="all_generated",
    )
    output = _run_with_capture(
        enabled_llm, "Count to ten:", spec, max_tokens=4
    )
    assert output.activation_storage is not None
    assert output.activation_storage.status == "ok"
    pairs = [
        (b, j)
        for (b, j) in _bin_json_pairs(root)
        if "phase4_test_5" in str(b)
    ]
    assert len(pairs) == 1
    _, json_path = pairs[0]
    meta = _load_sidecar(json_path)
    # 4 generated tokens (max_tokens=4) → 4 captured rows
    assert meta["shape"][0] == 4
    assert meta["position_kind"] == "all_generated"


# ---------------------------------------------------------------------------
# 6. all positions (prompt + generated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_REASON)
def test_all_positions(enabled_llm):
    root: pathlib.Path = enabled_llm._capture_root
    spec = _make_spec(
        request_id="all_0",
        tag="phase4_test_6",
        hooks={"post_mlp": [6]},
        positions="all",
    )
    output = _run_with_capture(enabled_llm, "Hello there", spec, max_tokens=2)
    assert output.activation_storage is not None
    assert output.activation_storage.status == "ok"
    pairs = [
        (b, j)
        for (b, j) in _bin_json_pairs(root)
        if "phase4_test_6" in str(b)
    ]
    assert len(pairs) == 1
    _, json_path = pairs[0]
    meta = _load_sidecar(json_path)
    num_prompt = len(meta["prompt_token_ids"])
    num_gen = len(meta["generated_token_ids"])
    assert meta["shape"][0] == num_prompt + num_gen
    assert meta["position_kind"] == "all"


# ---------------------------------------------------------------------------
# 7. explicit position list
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_REASON)
def test_explicit_position_list(enabled_llm):
    root: pathlib.Path = enabled_llm._capture_root
    # We can't know the exact prompt_token count without tokenizing so
    # pick a small explicit list that will always fit our test prompts.
    spec = _make_spec(
        request_id="explicit_0",
        tag="phase4_test_7",
        hooks={"post_mlp": [6]},
        positions=[0, 1, 2],
    )
    output = _run_with_capture(
        enabled_llm, "The quick brown fox jumps over the lazy dog", spec
    )
    assert output.activation_storage is not None
    assert output.activation_storage.status == "ok"
    pairs = [
        (b, j)
        for (b, j) in _bin_json_pairs(root)
        if "phase4_test_7" in str(b)
    ]
    assert len(pairs) == 1
    _, json_path = pairs[0]
    meta = _load_sidecar(json_path)
    assert meta["position_kind"] == "explicit"
    assert meta["shape"][0] == 3


# ---------------------------------------------------------------------------
# 8. two requests in the same batch
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_REASON)
def test_multiple_requests_same_batch(enabled_llm):
    root: pathlib.Path = enabled_llm._capture_root
    spec_a = _make_spec(
        request_id="batch_a",
        tag="phase4_test_8",
        hooks={"post_mlp": [6]},
        positions="last_prompt",
    )
    spec_b = _make_spec(
        request_id="batch_b",
        tag="phase4_test_8",
        hooks={"post_mlp": [6]},
        positions="last_prompt",
    )
    sp_a = SamplingParams(
        temperature=0.0, max_tokens=1, activation_storing=spec_a
    )
    sp_b = SamplingParams(
        temperature=0.0, max_tokens=1, activation_storing=spec_b
    )
    outputs = enabled_llm.generate(
        ["Alpha prompt", "Beta prompt"],
        [sp_a, sp_b],
    )
    assert len(outputs) == 2
    for o in outputs:
        assert o.activation_storage is not None
        assert o.activation_storage.status == "ok"
    pairs = [
        (b, j)
        for (b, j) in _bin_json_pairs(root)
        if "phase4_test_8" in str(b)
    ]
    assert len(pairs) == 2
    rids = sorted(_load_sidecar(j)["request_id"] for _, j in pairs)
    assert rids == ["batch_a", "batch_b"]


# ---------------------------------------------------------------------------
# 9. request without activation storing spec
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_REASON)
def test_request_without_activation_storing_spec(enabled_llm):
    spec = _make_spec(
        request_id="with_spec",
        tag="phase4_test_9",
        hooks={"post_mlp": [6]},
        positions="last_prompt",
    )
    sp_with = SamplingParams(
        temperature=0.0, max_tokens=1, activation_storing=spec
    )
    sp_without = SamplingParams(temperature=0.0, max_tokens=1)
    outputs = enabled_llm.generate(
        ["Prompt with capture", "Prompt without capture"],
        [sp_with, sp_without],
    )
    assert outputs[0].activation_storage is not None
    assert outputs[0].activation_storage.status == "ok"
    # Non-spec request gets a None activation_storage.
    assert outputs[1].activation_storage is None
    # Text generation unaffected on both.
    assert len(outputs[0].outputs[0].token_ids) >= 1
    assert len(outputs[1].outputs[0].token_ids) >= 1


# ---------------------------------------------------------------------------
# 10. cold path: server started without activation storing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_REASON)
def test_cold_path_server_start(tmp_path: pathlib.Path):
    # Server without --activation-storing: RequestOutput.activation_storage
    # is None and no files hit disk.
    llm = LLM(
        model=_STEERABLE_MODEL,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=256,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    outputs = llm.generate(["Hello"], sp)
    assert outputs[0].activation_storage is None
    _assert_no_captures(tmp_path)


# ---------------------------------------------------------------------------
# 11. TP > 1 rejected at engine init
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least 2 CUDA devices for TP=2 rejection test",
)
def test_tp_gt_1_rejected_at_init(tmp_path: pathlib.Path):
    with pytest.raises(ValueError, match=r"(?i)tensor_parallel|activation"):
        LLM(
            model=_STEERABLE_MODEL,
            tensor_parallel_size=2,
            activation_storing=str(tmp_path),
            enforce_eager=True,
            gpu_memory_utilization=0.3,
            max_model_len=256,
        )
