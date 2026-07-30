# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-engine e2e tests for activation capture.

Everything here drives an offline ``LLM`` with real weights on a real
GPU — no mocked engine, no synthetic tensors. Covers the behaviors the
unit tier proves only against fakes: the filesystem consumer writing
correct files from a live forward pass, per-request ``SamplingParams
.capture`` round-tripping to ``RequestOutput.capture_results``, hook
placement against a HuggingFace reference, the prefix-cache capture
floor, a global-spec consumer, and the ``block`` overload policy.

Model is ``CAPTURE_E2E_MODEL`` (default ``Qwen/Qwen3-0.6B`` — small,
ungated, and hook-wired incl. the mlp taps). Skips cleanly without
CUDA or when the checkpoint can't be fetched.

Run: ``pytest tests/v1/capture/test_capture_engine_e2e.py -v``
"""

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import gc
import json
import logging
import pathlib
import time

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.v1.capture.consumers.filesystem import FilesystemCaptureRequest
from vllm.v1.capture.consumers.filesystem.reader import (
    read_per_file,
    read_request,
)

MODEL = os.environ.get("CAPTURE_E2E_MODEL", "Qwen/Qwen3-0.6B")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="capture e2e requires CUDA"
)

_FILE_WAIT_SECONDS = 120.0


def _skip_on_model_access_failure(exc: Exception) -> None:
    """Convert checkpoint fetch failures into skips, mirroring
    tests/models/language/generation/test_steering.py."""
    msg = str(exc).lower()
    if isinstance(exc, OSError) and (
        "gated repo" in msg or "connection error" in msg or "read timeout" in msg
    ):
        pytest.skip(f"{MODEL} unavailable: {exc}")
    if type(exc).__module__.startswith("requests"):
        pytest.skip(f"{MODEL} download failure: {exc}")


def _hf_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(MODEL)


def _prompt_ids(min_tokens: int) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = tok("The quick brown fox jumps over the lazy dog. ").input_ids
    ids: list[int] = []
    while len(ids) < min_tokens:
        ids.extend(base)
    return ids[:min_tokens]


def _wait_for_files(paths: list[pathlib.Path]) -> None:
    """Capture files land asynchronously; wait for the full expected set
    (a fixed sleep races the flush — see the feature doc)."""
    deadline = time.monotonic() + _FILE_WAIT_SECONDS
    while time.monotonic() < deadline:
        if all(p.exists() for p in paths):
            return
        time.sleep(0.2)
    missing = [str(p) for p in paths if not p.exists()]
    pytest.fail(f"capture files not published within timeout: {missing}")


def _fs_request(request_id, tag, hooks, positions, layout=None):
    return FilesystemCaptureRequest(
        request_id=request_id,
        tag=tag,
        hooks=hooks,
        positions=positions,
        layout=layout,
    )


def _generate(llm, prompt_ids, capture, max_tokens=4):
    sampling = SamplingParams(
        max_tokens=max_tokens, temperature=0.0, capture=capture
    )
    [output] = llm.generate(
        [{"prompt_token_ids": prompt_ids}], sampling, use_tqdm=False
    )
    return output


@pytest.fixture(scope="module")
def fs_root(tmp_path_factory) -> pathlib.Path:
    return tmp_path_factory.mktemp("captures")


@pytest.fixture(scope="module")
def fs_llm(fs_root):
    """One real engine with the filesystem consumer for the module."""
    try:
        llm = LLM(
            model=MODEL,
            max_model_len=256,
            enforce_eager=True,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.4,
            seed=0,
            capture_consumers=[
                {"name": "filesystem", "params": {"root": str(fs_root)}}
            ],
        )
    except Exception as exc:
        _skip_on_model_access_failure(exc)
        raise
    yield llm
    del llm
    cleanup_dist_env_and_memory()


def test_per_file_layout_shapes_and_sidecars(fs_llm, fs_root):
    """A live forward writes one .bin/.json per (layer, hook) with the
    prompt row count, hidden size, and residual dtype."""
    cfg = _hf_config()
    layers = sorted({1, cfg.num_hidden_layers // 2})
    ids = _prompt_ids(24)
    _generate(
        fs_llm,
        ids,
        {"filesystem": _fs_request("perfile-req", "e2e", {"post_block": layers},
                                   "all_prompt")},
    )

    req_dir = fs_root / "e2e" / "perfile-req"
    bins = [req_dir / f"{layer}_post_block.bin" for layer in layers]
    sidecars = [p.with_suffix(".json") for p in bins]
    _wait_for_files(bins + sidecars)

    for layer, bin_path in zip(layers, bins):
        entry = read_per_file(bin_path)
        assert entry.layer == layer
        assert entry.hook == "post_block"
        assert entry.array.shape == (len(ids), cfg.hidden_size)
        assert entry.dtype == "bfloat16"
        sidecar = json.loads(bin_path.with_suffix(".json").read_text())
        assert sidecar["shape"] == [len(ids), cfg.hidden_size]


def test_all_generated_rows_span_decode_steps(fs_llm, fs_root):
    """all_generated captures append one row per forwarded generated token.

    A generated token's residual exists only on the step that feeds it
    back through the model, so the final sampled token is never captured:
    n generated tokens yield n-1 rows.
    """
    cfg = _hf_config()
    num_tokens = 8
    output = _generate(
        fs_llm,
        _prompt_ids(16),
        {"filesystem": _fs_request("gen-req", "e2e", {"post_block": [1]},
                                   "all_generated")},
        max_tokens=num_tokens,
    )
    assert len(output.outputs[0].token_ids) == num_tokens

    bin_path = fs_root / "e2e" / "gen-req" / "1_post_block.bin"
    _wait_for_files([bin_path, bin_path.with_suffix(".json")])
    entry = read_per_file(bin_path)
    assert entry.array.shape == (num_tokens - 1, cfg.hidden_size)


def test_packed_layout_roundtrip(fs_llm, fs_root):
    """packed layout: one packed.bin/.json per request, reader recovers
    every (layer, hook) tensor."""
    cfg = _hf_config()
    layers = [1, cfg.num_hidden_layers // 2]
    ids = _prompt_ids(16)
    _generate(
        fs_llm,
        ids,
        {"filesystem": _fs_request("packed-req", "e2e",
                                   {"post_block": layers, "pre_attn": [1]},
                                   "all_prompt", layout="packed")},
    )

    req_dir = fs_root / "e2e" / "packed-req"
    _wait_for_files([req_dir / "packed.bin", req_dir / "packed.json"])
    entries = read_request(req_dir)
    expected_keys = {(layer, "post_block") for layer in layers} | {(1, "pre_attn")}
    assert set(entries.keys()) == expected_keys
    for entry in entries.values():
        assert entry.array.shape == (len(ids), cfg.hidden_size)


def test_capture_results_best_effort(fs_llm, fs_root):
    """capture_results, when attached, reports ok and the written paths.

    Delivery is documented as best-effort (finalize runs off-thread), so
    absence is tolerated — but a present result must be coherent and the
    files are asserted unconditionally.
    """
    output = _generate(
        fs_llm,
        _prompt_ids(16),
        {"filesystem": _fs_request("result-req", "e2e", {"post_block": [1]},
                                   "last_prompt")},
    )

    bin_path = fs_root / "e2e" / "result-req" / "1_post_block.bin"
    _wait_for_files([bin_path])
    assert read_per_file(bin_path).array.shape[0] == 1

    results = getattr(output, "capture_results", None) or {}
    if "filesystem" in results:
        result = results["filesystem"]
        assert result.status in ("ok", "pending"), result.error
        if result.status == "ok" and isinstance(result.payload, list):
            assert any(str(bin_path) in str(p) for p in result.payload)


def test_post_block_matches_hf_hidden_states(fs_llm, fs_root):
    """Hook placement: captured post_block at layer L equals HF's
    hidden_states[L+1] for the same token ids (bf16 tolerance)."""
    cfg = _hf_config()
    layer = cfg.num_hidden_layers // 2
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("The capital of France is Paris, and the capital of "
              "Germany is").input_ids
    _generate(
        fs_llm,
        ids,
        {"filesystem": _fs_request("hfref-req", "e2e", {"post_block": [layer]},
                                   "all_prompt")},
        max_tokens=1,
    )
    bin_path = fs_root / "e2e" / "hfref-req" / f"{layer}_post_block.bin"
    _wait_for_files([bin_path, bin_path.with_suffix(".json")])
    entry = read_per_file(bin_path)
    captured = (
        torch.from_numpy(entry.array.copy()).view(torch.bfloat16).float()
    )

    from transformers import AutoModelForCausalLM

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float32
        )
    except Exception as exc:
        _skip_on_model_access_failure(exc)
        raise
    try:
        with torch.no_grad():
            hf_out = hf_model(
                torch.tensor([ids]), output_hidden_states=True
            )
        reference = hf_out.hidden_states[layer + 1][0]
    finally:
        del hf_model
        gc.collect()

    assert captured.shape == reference.shape
    cos = torch.nn.functional.cosine_similarity(captured, reference, dim=-1)
    assert cos.min().item() > 0.98, (
        f"hook misplacement suspected: min cosine {cos.min().item():.4f}"
    )
    rel_err = (captured - reference).norm() / reference.norm()
    assert rel_err.item() < 0.05, f"relative L2 error {rel_err.item():.4f}"


def test_prefix_cache_floor_all_prompt_reforwards(fs_llm, fs_root):
    """A prompt-range capture re-forwards past the cached prefix: the
    second identical all_prompt request still yields every prompt row,
    and a last_prompt request yields exactly one."""
    cfg = _hf_config()
    ids = _prompt_ids(48)  # >= 2 full KV blocks so APC genuinely engages

    for req_id, positions, expected_rows in (
        ("floor-a", "all_prompt", 48),
        ("floor-b", "all_prompt", 48),
        ("floor-c", "last_prompt", 1),
    ):
        _generate(
            fs_llm,
            ids,
            {"filesystem": _fs_request(req_id, "floor", {"post_block": [1]},
                                       positions)},
        )
        bin_path = fs_root / "floor" / req_id / "1_post_block.bin"
        _wait_for_files([bin_path, bin_path.with_suffix(".json")])
        entry = read_per_file(bin_path)
        assert entry.array.shape == (expected_rows, cfg.hidden_size), (
            f"{req_id} ({positions}): got {entry.array.shape}"
        )


def test_logging_global_consumer_emits(monkeypatch):
    """A global-spec consumer captures every request with no client opt-in.

    Runs the engine in-process and attaches a handler directly to the
    consumer's logger (vLLM loggers don't propagate to root, so caplog
    never sees them).
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    records: list[logging.LogRecord] = []

    class _Recorder(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    capture_logger = logging.getLogger("vllm.capture.logging")
    handler = _Recorder(level=logging.INFO)
    capture_logger.addHandler(handler)
    capture_logger.setLevel(logging.INFO)
    try:
        llm = LLM(
            model=MODEL,
            max_model_len=256,
            enforce_eager=True,
            gpu_memory_utilization=0.25,
            seed=0,
            capture_consumers=[
                {"name": "logging", "params": {"hooks": {"post_block": [1]}}}
            ],
        )
    except Exception as exc:
        _skip_on_model_access_failure(exc)
        raise
    try:
        [output] = llm.generate(
            [{"prompt_token_ids": _prompt_ids(16)}],
            SamplingParams(max_tokens=4, temperature=0.0),
            use_tqdm=False,
        )
        assert len(output.outputs[0].token_ids) == 4

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if any("capture key=" in r.getMessage() for r in records):
                break
            time.sleep(0.2)
        assert any("capture key=" in r.getMessage() for r in records), (
            "logging consumer never observed a finalized capture"
        )
    finally:
        capture_logger.removeHandler(handler)
        del llm
        cleanup_dist_env_and_memory()


def test_overload_block_policy_no_loss(tmp_path):
    """With a tiny dispatch queue and policy=block, heavy capture stalls
    the forward instead of dropping: every expected file is published."""
    cfg = _hf_config()
    root = tmp_path / "overload"
    try:
        llm = LLM(
            model=MODEL,
            max_model_len=256,
            enforce_eager=True,
            gpu_memory_utilization=0.25,
            seed=0,
            capture_dispatch_queue_size=2,
            capture_overload_policy="block",
            capture_consumers=[
                {"name": "filesystem", "params": {"root": str(root)}}
            ],
        )
    except Exception as exc:
        _skip_on_model_access_failure(exc)
        raise
    try:
        ids = _prompt_ids(48)
        num_requests = 4
        sampling = [
            SamplingParams(
                max_tokens=2,
                temperature=0.0,
                capture={
                    "filesystem": _fs_request(
                        f"flood-{i}", "flood", {"post_block": "all"},
                        "all_prompt",
                    )
                },
            )
            for i in range(num_requests)
        ]
        outputs = llm.generate(
            [{"prompt_token_ids": ids}] * num_requests, sampling, use_tqdm=False
        )
        assert len(outputs) == num_requests

        expected = [
            root / "flood" / f"flood-{i}" / f"{layer}_post_block.bin"
            for i in range(num_requests)
            for layer in range(cfg.num_hidden_layers)
        ]
        _wait_for_files(expected)
        entry = read_per_file(expected[0])
        assert entry.array.shape == (len(ids), cfg.hidden_size)
    finally:
        del llm
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
