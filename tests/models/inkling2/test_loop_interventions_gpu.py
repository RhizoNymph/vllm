# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU: loop-aware steering, clamping, patching and capture on tiny random
Inkling2 models (Small and a Mini-shaped loop: prelude 0, core 1-2 x 3
passes, coda 3, so layer keys 0, 5/9/13, 6/10/14, 3).

Runs on any CUDA device the Inkling2 kernels support (SM8x via the
quant-method backend, SM90+ native).
"""

import json
import pathlib
import time

import pytest
import torch

from tests.models.inkling2.test_sm8x_model import TINY_CONFIG, TINY_LOOP_CONFIG
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

PROMPT = list(range(2, 66))
HIDDEN = {False: 512, True: 2304}


def _llm(tmp_path, looped: bool, **kwargs) -> LLM:
    (tmp_path / "config.json").write_text(
        json.dumps(TINY_LOOP_CONFIG if looped else TINY_CONFIG)
    )
    return LLM(
        model=str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        tokenizer_mode="hf",
        max_model_len=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.25 if looped else 0.10,
        limit_mm_per_prompt={"image": 0},
        enable_prefix_caching=False,
        **kwargs,
    )


def _run(llm: LLM, **sampling) -> tuple[list[int], list[float]]:
    out = llm.generate(
        [{"prompt_token_ids": PROMPT}],
        SamplingParams(
            max_tokens=8,
            temperature=0,
            logprobs=0,
            ignore_eos=True,
            detokenize=False,
            **sampling,
        ),
    )[0].outputs[0]
    lps = [lp[t].logprob for lp, t in zip(out.logprobs, out.token_ids)]
    return list(out.token_ids), lps


def _redraw_weights(model: torch.nn.Module) -> None:
    """Dummy weights route every token alike; redraw gates and controllers
    so passes and tokens differ."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    for name, param in model.named_parameters():
        if name.endswith("mlp.gate.weight"):
            param.data.normal_(generator=generator)
    for name, buf in model.named_buffers():
        if name.endswith(("controller_down", "controller_gain_forget")):
            buf.normal_(std=buf.shape[-1] ** -0.5, generator=generator)


@pytest.fixture
def steer_llm(tmp_path, enable_pickle):
    llm = _llm(tmp_path, True, enable_steering=True, max_steering_configs=4)
    llm.apply_model(_redraw_weights)
    yield llm
    del llm
    cleanup_dist_env_and_memory()


def test_sites_are_listed_by_layer_key(steer_llm):
    layers = steer_llm.collective_rpc("list_steerable_layers")[0]
    keys = sorted(int(k) for k in layers)
    assert keys == [0, 3, 5, 6, 9, 10, 13, 14]


def test_zero_vector_is_an_exact_no_op(steer_llm):
    base = _run(steer_llm)
    zero = [0.0] * HIDDEN[True]
    for key in ("1", "1@1", 3):
        steered = _run(steer_llm, steering_vectors={"post_block": {key: zero}})
        assert steered == base


@pytest.mark.parametrize("hook", ["pre_attn", "post_attn", "post_block", "mlp_in"])
def test_pass_targeting_changes_output_and_differs_from_all_passes(steer_llm, hook):
    base = _run(steer_llm)
    vec = [4.0] * HIDDEN[True]
    one_pass = _run(steer_llm, steering_vectors={hook: {"2@2": vec}})
    all_passes = _run(steer_llm, steering_vectors={hook: {"2": vec}})
    assert one_pass != base and all_passes != base
    assert one_pass != all_passes
    # Rows are freed per request: the next unsteered request is baseline.
    assert _run(steer_llm) == base


def test_pass_index_is_rejected_on_non_looped_models(tmp_path):
    llm = _llm(tmp_path, False, enable_steering=True, max_steering_configs=4)
    try:
        with pytest.raises(ValueError, match="looped model"):
            _run(llm, steering_vectors={"post_block": {"1@0": [1.0] * 512}})
        with pytest.raises(ValueError, match="out of range|outside the loop core"):
            SamplingParams(steering_vectors={"post_block": {"1@5": [1.0] * 512}})
    finally:
        del llm
        cleanup_dist_env_and_memory()


def test_clamp_and_patch_target_a_pass(steer_llm):
    base = _run(steer_llm)
    direction = [1.0] * HIDDEN[True]
    clamped = _run(
        steer_llm,
        steering_clamps={"post_block": {"1@1": [{"vector": direction, "value": 0.0}]}},
    )
    assert clamped != base
    patched = _run(
        steer_llm,
        patch=[
            {
                "layer": 1,
                "pass": 2,
                "hook": "post_block",
                "dest_position": len(PROMPT) - 1,
                "source_module": "zeros",
            }
        ],
    )
    assert patched != base


def _wait(paths: list[pathlib.Path], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(p.exists() for p in paths):
            return
        time.sleep(0.2)
    raise AssertionError(f"missing capture files: {[str(p) for p in paths]}")


def test_capture_records_one_row_per_pass_and_replays_as_steering(
    tmp_path, enable_pickle
):
    from vllm.v1.capture.consumers.filesystem import FilesystemCaptureRequest
    from vllm.v1.capture.consumers.filesystem.reader import read_per_file

    root = tmp_path / "cap"
    llm = _llm(
        tmp_path,
        True,
        enable_steering=True,
        max_steering_configs=4,
        enforce_eager=True,
        capture_consumers=[{"name": "filesystem", "params": {"root": str(root)}}],
    )
    try:
        llm.apply_model(_redraw_weights)
        spec = FilesystemCaptureRequest(
            request_id="r",
            tag="t",
            hooks={"post_block": [1, 3]},
            positions="all_prompt",
        )
        _run(llm, capture={"filesystem": spec})
        files = {k: root / "t" / "r" / f"{k}_post_block.bin" for k in (5, 9, 13, 3)}
        _wait(list(files.values()))
        rows = {k: read_per_file(p).array for k, p in files.items()}
        assert all(r.shape == (len(PROMPT), HIDDEN[True]) for r in rows.values())
        # Each pass of layer 1 is its own visit.
        assert not torch.equal(torch.as_tensor(rows[5]), torch.as_tensor(rows[9]))
        # A captured block state is a valid steering vector in the same
        # space: adding it back at the same site changes the output.
        vec = torch.as_tensor(rows[9][-1]).float().tolist()
        assert _run(llm, steering_vectors={"post_block": {"1@1": vec}}) != _run(llm)
    finally:
        del llm
        cleanup_dist_env_and_memory()
