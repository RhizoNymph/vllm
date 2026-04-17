<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vllm website to help you get started with vllm. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## Branch: `feat/capture-consumers` — Activation Capture Consumer Framework

This branch adds a pluggable **capture consumer** framework that lets you
observe and route hidden-state activations produced inside vLLM's forward
pass — without breaking `torch.compile` or CUDA graphs.

### What it does

At each decoder layer, up to five hook points fire:
`pre_attn`, `post_attn`, `post_mlp`, `mlp_in`, `mlp_out`.
Consumers subscribe to any combination of `(layer, hook)` pairs and receive
CPU tensors shaped `(num_rows, hidden_size)` for whatever positions they
requested. Capture can be triggered:

- **Globally** — the consumer declares a `CaptureSpec` that applies to
  every request automatically (reward producers, dashboards, probes).
- **Per-request** — the client opts in via `SamplingParams.capture` or the
  `extra_body.capture` field on the OpenAI-compatible API.

Failures in consumers surface as `partial_error` / `error` on
`RequestOutput.capture_results`; text generation always completes.

### Architecture in brief

```
Decoder-layer forward (apply_layer_steering / maybe_capture_residual)
    │
    │  torch.ops.vllm.capture_residual  (compile-graph-opaque custom op)
    ▼
CaptureManager  (vllm/v1/capture/manager.py)
  – resolves per-step gather indices from registered specs
  – index_selects into per-(layer,hook) scratch tensors
  – fans out CPU tensors to each consumer
    │
    ├── _BatchedAdapter  (wraps CaptureConsumer subclasses)
    ├── Direct CaptureSink  (e.g. FilesystemConsumer)
    └── _DriverQueueShim  (proxies driver-side consumers via mp.Queue)
```

Full design: [`docs/design/capture_consumers.md`](docs/design/capture_consumers.md)

### Enabling consumers

**CLI**

```bash
vllm serve meta-llama/Llama-3-8B \
    --capture-consumers filesystem:root=/mnt/activations \
    --capture-consumers logging
```

**YAML**

```yaml
model: meta-llama/Llama-3-8B
capture_consumers:
  - filesystem:root=/mnt/activations,writer_threads=4
  - logging
```

**Python `LLM(...)`**

```python
from vllm import LLM, SamplingParams
from vllm.v1.capture.consumers.filesystem import FilesystemCaptureRequest

llm = LLM(
    model="meta-llama/Llama-3-8B",
    capture_consumers=[
        {"name": "filesystem", "params": {"root": "/tmp/captures"}},
        {"name": "logging",    "params": {"hooks": {"post_mlp": [0]}}},
    ],
)

sampling_params = SamplingParams(
    max_tokens=16,
    capture={
        "filesystem": FilesystemCaptureRequest(
            request_id="req1", tag="demo",
            hooks={"post_mlp": [0]}, positions="last_prompt",
        ),
    },
)

[output] = llm.generate(["Hello"], sampling_params)
result = output.capture_results.get("filesystem")
if result and result.status == "ok":
    for path in result.payload:
        print("wrote", path)
```

**OpenAI-compatible API**

```python
import httpx
response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3-8B",
        "messages": [{"role": "user", "content": "Paris?"}],
        "extra_body": {
            "capture": {
                "filesystem": {
                    "request_id": "probe_0001", "tag": "capital-probe",
                    "hooks": {"post_mlp": [12, 16, 20, 24]},
                    "positions": "last_prompt",
                },
            },
        },
    },
    timeout=60,
).json()
```

### Built-in consumers

| Name | Class | Mode | Description |
|---|---|---|---|
| `filesystem` | `FilesystemConsumer` | per-request | Streams activations to `.bin` files + atomic sidecar JSON. Layout: `{root}/{tag}/{request_id}/{layer}_{hook}.bin`. |
| `logging` | `LoggingConsumer` | global | Logs one line per finalized capture. Discards the tensor. Useful as a smoke test. |

Full user guide: [`docs/features/capture_consumers.md`](docs/features/capture_consumers.md)

### Example plugins (`examples/capture_consumers/`)

#### `minimal_plugin/`

The simplest possible `CaptureConsumer` subclass. Records the running sum
of every captured tensor. Shows the entry-point registration pattern and
the `on_capture` / `global_capture_spec` contract.

Install and use:

```bash
pip install -e examples/capture_consumers/minimal_plugin/
vllm serve my-model --capture-consumers sum:layers=[0,15,31]
```

#### `activation_reward_producer/`

A direct `CaptureSink` that computes a scalar reward for online RL loops:

```
reward = nonlinearity(scale × cos(mean_pool(activations[slice]), reference_vector))
```

The reward is returned on `CaptureResult.payload` alongside diagnostics
(`cos`, `act_norm`, `num_positions`, `status`) so the RL trainer can
monitor for vector drift without a second capture path.

Install:

```bash
pip install -e examples/capture_consumers/activation_reward_producer/
```

Configure at engine init:

```python
llm = LLM(
    model="meta-llama/Llama-3-8B",
    capture_consumers=[{
        "name": "activation_reward",
        "params": {
            "layer": 12, "hook": "post_mlp",
            "vector_path": "/models/reference.pt",
            "scale": 5.0, "nonlinearity": "tanh",
        },
    }],
)
```

Clients opt in per-request with an empty spec:

```python
sp = SamplingParams(max_tokens=128, capture={"activation_reward": {}})
outs = llm.generate(prompts, sp)
rewards = [o.capture_results["activation_reward"].payload["reward"] for o in outs]
```

Multiple reference directions are supported via `instance_name`:

```python
capture_consumers=[
    {"name": "activation_reward", "instance_name": "sadness_reward",
     "params": {"layer": 12, "hook": "post_mlp", "vector_path": "/vec/sad.pt"}},
    {"name": "activation_reward", "instance_name": "anger_reward",
     "params": {"layer": 18, "hook": "post_attn", "vector_path": "/vec/anger.pt"}},
]
```

**RL-loop sketch**:

```python
# Rollout — rewards land on capture_results
outs = llm.generate(prompts, sp)
rewards = [o.capture_results["activation_reward"].payload["reward"] for o in outs]

# External RL update (TRL / verl / custom)
trainer.step(prompts, [o.outputs[0].text for o in outs], rewards)

# Push weights back to vLLM
llm.sleep(level=2)
llm.wake_up(tags=["weights"])
llm.collective_rpc("reload_weights")
llm.wake_up(tags=["kv_cache"])
```

See the plugin README and [`docs/capture_consumers/plugin_authoring.md`](docs/capture_consumers/plugin_authoring.md) for the full plugin authoring guide.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on HuggingFace, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
