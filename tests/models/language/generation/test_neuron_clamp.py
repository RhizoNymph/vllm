# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for per-request MLP neuron clamps on Qwen3.

The reference is HF transformers with a ``down_proj`` forward-pre-hook that
performs the same clamp on the SiLU-gated activation. vLLM runs with
torch.compile + CUDA graphs (the production path), so these also check that
graph replay reads the live clamp tables.
"""

import pytest
import torch

from vllm import SamplingParams

from ...utils import check_logprobs_close

MODEL = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 16
NUM_LOGPROBS = 5

PROMPTS = [
    "The capital of France is",
    "Write a short poem about the ocean:",
    "def fibonacci(n):",
    "In 1969, the first humans",
]

# Strong clamps (so outputs really change) spread over several layers and
# all three modes.
CLAMPS_BOTH = [
    {"layer": 6, "neuron": 100, "mode": "set", "value": 40.0},
    {"layer": 6, "neuron": 2047, "mode": "scale", "value": -8.0},
    {"layer": 14, "neuron": 7, "mode": "add", "value": 25.0},
]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _with_phase(clamps, phase):
    return [{**c, "phase": phase} for c in clamps]


def _vllm_kwargs(**over):
    kw = dict(
        max_model_len=1024,
        enable_steering=True,
        max_steering_configs=8,
        gpu_memory_utilization=0.45,
        dtype="bfloat16",
        enable_prefix_caching=False,
        max_num_seqs=8,
    )
    kw.update(over)
    return kw


def _to_tlp(req_out):
    o = req_out.outputs[0]
    lps = [{tid: lp.logprob for tid, lp in d.items()} for d in o.logprobs]
    return (list(o.token_ids), o.text, lps)


def _gen(llm, prompts, sps):
    outs = llm.llm.generate(prompts, sps, use_tqdm=False)
    return [_to_tlp(o) for o in outs]


def _sp(clamps=None, steering=None, max_tokens=MAX_TOKENS):
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=NUM_LOGPROBS,
        neuron_clamps=clamps,
        steering_vectors=steering,
    )


def _install_hf_clamps(hf_model, clamps):
    """Register down_proj pre-hooks reproducing ``clamps`` on HF Qwen3."""
    by_layer: dict[int, list[dict]] = {}
    for c in clamps:
        by_layer.setdefault(c["layer"], []).append(c)
    handles = []
    for layer, cs in by_layer.items():
        down = hf_model.model.layers[layer].mlp.down_proj

        def hook(_mod, args, cs=cs):
            (act,) = args
            # generate(): the first forward is the whole prompt (prefill),
            # every later one is a single decode token.
            phase_now = "prefill" if act.shape[1] > 1 else "decode"
            act = act.clone()
            for c in cs:
                phase = c.get("phase", "both")
                if phase not in ("both", phase_now):
                    continue
                n = c["neuron"]
                x = act[..., n].float()
                if c["mode"] == "set":
                    new = torch.full_like(x, c["value"])
                elif c["mode"] == "add":
                    new = x + c["value"]
                else:
                    new = x * c["value"]
                act[..., n] = new.to(act.dtype)
            return (act,)

        handles.append(down.register_forward_pre_hook(hook))
    return handles


def _hf_outputs(hf_runner, clamps):
    with hf_runner(MODEL, dtype="bfloat16") as hf:
        handles = _install_hf_clamps(hf.model, clamps) if clamps else []
        try:
            return hf.generate_greedy_logprobs_limit(PROMPTS, MAX_TOKENS, NUM_LOGPROBS)
        finally:
            for h in handles:
                h.remove()


@pytest.mark.parametrize("phase", ["both", "prefill", "decode"])
def test_clamp_matches_hf_reference(hf_runner, vllm_runner, phase):
    clamps = _with_phase(CLAMPS_BOTH, phase)
    hf_clamped = _hf_outputs(hf_runner, clamps)
    hf_plain = _hf_outputs(hf_runner, None)
    with vllm_runner(MODEL, **_vllm_kwargs()) as llm:
        v_clamped = _gen(llm, PROMPTS, [_sp(clamps)] * len(PROMPTS))
        v_plain = _gen(llm, PROMPTS, [_sp()] * len(PROMPTS))

    # The clamps must matter, otherwise the comparison proves nothing.
    assert [o[0] for o in hf_clamped] != [o[0] for o in hf_plain]
    assert [o[0] for o in v_clamped] != [o[0] for o in v_plain]
    check_logprobs_close(
        outputs_0_lst=hf_clamped,
        outputs_1_lst=v_clamped,
        name_0="hf_clamped",
        name_1="vllm_clamped",
    )
    check_logprobs_close(
        outputs_0_lst=hf_plain, outputs_1_lst=v_plain, name_0="hf", name_1="vllm"
    )
    if phase == "decode":
        # First token comes from prefill logits: unaffected by decode clamps.
        for c, p in zip(v_clamped, v_plain):
            assert c[0][0] == p[0][0]


def test_noop_clamps_and_natural_value(hf_runner, vllm_runner):
    """``scale 1`` / ``add 0`` are bit-exact no-ops; ``set`` to the neuron's
    own natural value (single-position prompt) is a no-op up to bf16
    rounding of the value, while a perturbed set value is not."""
    layer, neuron = 6, 100
    with hf_runner(MODEL, dtype="bfloat16") as hf:
        tok_ids = hf.tokenizer("Hello", add_special_tokens=False).input_ids[:1]
        captured = {}

        def grab(_m, args):
            captured["act"] = args[0][0, -1, neuron].float().item()

        h = hf.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(grab)
        with torch.no_grad():
            hf.model(torch.tensor([tok_ids], device=hf.model.device))
        h.remove()
    natural = captured["act"]

    prompt = [{"prompt_token_ids": tok_ids}]
    one = dict(max_tokens=1)
    with vllm_runner(MODEL, **_vllm_kwargs()) as llm:
        plain = _gen(llm, prompt, [_sp(**one)])[0]
        noop = _gen(
            llm,
            prompt,
            [
                _sp(
                    clamps=[
                        {
                            "layer": layer,
                            "neuron": neuron,
                            "mode": "scale",
                            "value": 1.0,
                        },
                        {"layer": 14, "neuron": 7, "mode": "add", "value": 0.0},
                    ],
                    **one,
                )
            ],
        )[0]
        nat = _gen(
            llm,
            prompt,
            [
                _sp(
                    clamps=[
                        {
                            "layer": layer,
                            "neuron": neuron,
                            "mode": "set",
                            "value": natural,
                        }
                    ],
                    **one,
                )
            ],
        )[0]
        far = _gen(
            llm,
            prompt,
            [
                _sp(
                    clamps=[
                        {
                            "layer": layer,
                            "neuron": neuron,
                            "mode": "set",
                            "value": natural + 60.0,
                        }
                    ],
                    **one,
                )
            ],
        )[0]

    assert noop == plain
    assert nat[0] == plain[0]

    def lp_diff(a, b):
        common = set(a[2][0]) & set(b[2][0])
        return max(abs(a[2][0][t] - b[2][0][t]) for t in common)

    d_nat = lp_diff(nat, plain)
    d_far = lp_diff(far, plain) if set(far[2][0]) & set(plain[2][0]) else 1e9
    assert d_nat < 0.1, d_nat
    assert d_far > 10 * max(d_nat, 1e-3), (d_nat, d_far)


def test_batch_isolation(vllm_runner):
    """Clamp-only and steer+clamp requests in one batch must not leak into
    an unclamped request (or into each other)."""
    hidden = 1024  # Qwen3-0.6B hidden size
    steer = {"post_block": {10: [0.5] * hidden}}
    clamp_a = [{"layer": 6, "neuron": 100, "mode": "set", "value": 40.0}]
    clamp_b = [{"layer": 14, "neuron": 7, "mode": "add", "value": 25.0}]
    with vllm_runner(MODEL, **_vllm_kwargs()) as llm:
        base = _gen(llm, PROMPTS, [_sp()] * 4)
        mixed = _gen(
            llm,
            PROMPTS,
            [
                _sp(),
                _sp(clamps=clamp_a),
                _sp(clamps=clamp_b, steering=steer),
                _sp(steering=steer),
            ],
        )
        # Same batch shape with only request 1 clamped / only 2 steered+clamped.
        only1 = _gen(llm, PROMPTS, [_sp(), _sp(clamps=clamp_a), _sp(), _sp()])
        only2 = _gen(
            llm, PROMPTS, [_sp(), _sp(), _sp(clamps=clamp_b, steering=steer), _sp()]
        )
        only3 = _gen(llm, PROMPTS, [_sp(), _sp(), _sp(), _sp(steering=steer)])

    # Unclamped request unaffected by its neighbours.
    assert mixed[0][0] == base[0][0]
    # Each modified request is changed, and matches its isolated run.
    assert mixed[1][0] != base[1][0]
    assert mixed[1][0] == only1[1][0]
    assert mixed[2][0] != base[2][0]
    assert mixed[2][0] == only2[2][0]
    assert mixed[3][0] == only3[3][0]
    # Neighbours of the isolated runs are untouched.
    assert only1[0][0] == base[0][0] and only1[2][0] == base[2][0]


def test_prefix_cache_respects_prefill_clamps(vllm_runner):
    """Prefill clamps change KV: a clamped request must not reuse KV computed
    without the clamp (and vice versa), identical clamps may share it, and
    decode-only clamps leave the prompt KV shareable."""
    prompt = (
        "Once upon a time in a small village by the sea there lived "
        "an old fisherman who "
    ) * 6
    clamp = [
        {"layer": 6, "neuron": 100, "mode": "set", "value": 40.0, "phase": "prefill"}
    ]
    decode_clamp = [{**clamp[0], "phase": "decode"}]
    with vllm_runner(MODEL, **_vllm_kwargs(enable_prefix_caching=True)) as llm:

        def run(clamps):
            out = llm.llm.generate([prompt], [_sp(clamps=clamps)], use_tqdm=False)[0]
            return _to_tlp(out), out.num_cached_tokens

        plain1, cached = run(None)
        assert cached == 0
        plain2, cached = run(None)
        assert cached > 0 and plain2[0] == plain1[0]

        clamped1, cached = run(clamp)
        assert cached == 0, "prefill-clamped request reused unclamped KV"
        clamped2, cached = run(clamp)
        assert cached > 0 and clamped2[0] == clamped1[0]
        assert clamped1[0] != plain1[0]

        _, cached = run(decode_clamp)
        assert cached > 0, "decode-only clamp should reuse prompt KV"

        # Unclamped request after clamped ones still matches its first run.
        plain3, cached = run(None)
        assert plain3[0] == plain1[0]

    # Reference without any cache: the clamped KV is the real clamped KV.
    with vllm_runner(MODEL, **_vllm_kwargs(enable_prefix_caching=False)) as llm:
        fresh = llm.llm.generate([prompt], [_sp(clamps=clamp)], use_tqdm=False)[0]
    assert _to_tlp(fresh)[0] == clamped1[0]
