<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM with activation steering and activation capture
</h3>

---

## About This Fork

This is a fork of [vLLM](https://github.com/vllm-project/vllm) that adds two
interpretability subsystems wired directly into the model forward pass, built to
the same production bar as the rest of the engine:

- **Activation steering** — add precomputed vectors into the residual stream at
  inference time to shift model behavior without fine-tuning (tone/style,
  behavioral interventions, SAE-derived steering).
- **Activation capture** — a pluggable consumer system that routes hidden-state
  activations out of the forward pass to disk, a training loop, a dashboard, or
  any third-party plugin. Ships with a built-in filesystem consumer.

Both hook the residual stream at three points — `pre_attn`, `post_attn`,
`post_block` — across 100+ decoder architectures, and both run under continuous
batching, `torch.compile`, CUDA graphs, and tensor/pipeline parallelism.

📖 **Full guides:** [Activation Steering](docs/features/steering.md) ·
[Activation Capture](docs/features/capture_consumers.md)

## Design highlights

The hard part isn't adding a vector to the residual stream — it's doing so
without giving up vLLM's performance and correctness guarantees. The notable
engineering:

- **Stays on the CUDA-graph fast path.** Naively, capturing per-request
  activations forces every step eager. Instead, a rank-replicated per-step gate
  runs eager *only* on steps that actually gather, global probes use a
  persistent-buffer `copy_` baked into the graph at warmup, and a startup
  allowlist (`--capture-graphsafe-key`) lets chosen per-request keys ride that
  same graph-safe path. Plain traffic on a capture-enabled server keeps full
  cudagraph speed.
- **Prefix-cache correct.** Steering forks the APC cache key on prefill steering
  (but not decode-only steering, which must not fork prompt KV); capture
  re-forwards only the prompt suffix it needs when a tapped position was served
  from cache. Steering correctness under APC is treated as a correctness
  requirement, not a perf nicety.
- **Distributed.** Global steering fans out to every worker via `collective_rpc`
  with lock-step row allocation and no hot-path coordination; capture merges
  per-stage results under pipeline parallelism. Both validated across TP, PP, and
  cross-node.
- **Pluggable + tuned for real storage.** Capture consumers are entry-point
  plugins; the filesystem consumer offers `per_file` / `packed` / `sharded`
  layouts because on a network mount throughput is governed by file count
  (metadata RPCs), not bytes.

## Quickstart

**Steering** — global state over HTTP, or per-request via `SamplingParams`:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="google/gemma-3-4b-it", enable_steering=True)
params = SamplingParams(
    max_tokens=64,
    steering_vectors={"post_block": {15: {"vector": [0.1, 0.2], "scale": 2.0}}},
    decode_steering_vectors={"pre_attn": {15: [0.5, 0.6]}},
)
outputs = llm.generate(["Hello"], params)
```

Steering composes three additive tiers (base / prefill / decode) at both the
global and per-request level, supports named pre-registered modules, and exposes
`/v1/steering/*` endpoints (gated by `VLLM_SERVER_DEV_MODE=1`). See the
[steering guide](docs/features/steering.md) and the runnable
[`examples/online_serving/openai_steering_client.py`](examples/online_serving/openai_steering_client.py).

**Capture** — enable a consumer, then opt requests in:

```bash
vllm serve meta-llama/Llama-3-8B \
    --capture-consumers filesystem:root=/mnt/nas/activations
```

```python
from vllm import SamplingParams
from vllm.v1.capture.consumers.filesystem import FilesystemCaptureRequest

params = SamplingParams(
    max_tokens=16,
    capture={"filesystem": FilesystemCaptureRequest(
        request_id="probe_0001", tag="mnist-probe-v1",
        hooks={"post_block": [12]}, positions="last_prompt",
    )},
)
```

Consumers can be global (every request, e.g. a `logging` probe) or per-request
(client-driven). Results come back on `RequestOutput.capture_results`. See the
[capture guide](docs/features/capture_consumers.md) for layouts, throughput
tuning, backpressure policies, and the plugin-authoring path, plus example
plugins under [`examples/capture_consumers/`](examples/capture_consumers/).

## Supported models

Hooks are wired into the Llama, Qwen, Gemma, Mixtral/MoE, GLM, InternLM, Olmo,
Exaone, Phi, Plamo, Step, Molmo, Falcon/Baichuan/Command/StableLM families and
more — 100+ decoder architectures. Gemma 3 is the primary end-to-end test
target; several MoE models are validated against real weights. See the full
list in the [steering guide](docs/features/steering.md#supported-scope).

## Roadmap

> In progress / planned; details **subject to change**.

- **Dynamic steering** *(next; open draft [PR #180](https://github.com/RhizoNymph/vllm/pull/180))* —
  activation-conditioned steering that ties capture to steering so the model's
  own activations decide *when* and *how* to steer. A stack of three controller
  tiers — async (steers a later request), sync (per-step, every TP rank), and an
  in-graph monitor that gates a dynamic steering tier *within the same forward
  pass* via `sigmoid(sharpness · (residual · probe − threshold))` — plus the
  APC-correctness notification so dynamically-steered decode KV isn't falsely
  reused. GPU-validated on gemma4-31B across TP=1, TP=2 (cross-node), and PP=2.
- **Activation patching at scale** *(after that)* — transplanting/overwriting
  captured activations across runs as a first-class, high-throughput operation.
  Direction marker; details not yet pinned down.

## Upstream vLLM

This README covers only the steering/capture additions. For installation,
supported-model details, the OpenAI-compatible server, quantization, distributed
inference, and all other vLLM functionality, see the upstream project —
installation is unchanged from upstream:

- Repository: <https://github.com/vllm-project/vllm>
- Documentation: <https://docs.vllm.ai>

## Citation

If you use vLLM for your research, please cite the
[paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```
