# Activation Steering

Activation steering adds precomputed vectors into the residual stream of
decoder layers during inference. It can be used to shift model behavior
without fine-tuning, for example for tone/style changes, behavioral
interventions, or SAE-derived steering vectors.

This page is the user-facing guide to steering in vLLM. For internal
runtime details, see [Steering Runtime Design](../design/steering_runtime.md).

## Supported Scope

Steering in this initial PR is wired into a single decoder family:

- Gemma 3 (`gemma3`)

Additional model families, a global HTTP API, named modules, and
distributed-execution support land in follow-up PRs per the staged plan
in the RFC.

Also supported in PR 1:

- Per-request steering through `SamplingParams`
- Three additive entries per request (base / prefill-specific / decode-specific)
- Three hook points: `pre_attn`, `post_attn`, `post_mlp`
- Phase-aware scheduler admission for per-request steering
- Prefix-cache separation for different prefill steering configs
- Continuous batching
- `torch.compile` and CUDA graph execution

Not currently supported:

- Global steering via HTTP (comes in PR 2)
- Named steering modules (comes in PR 3)
- Tensor / pipeline parallel (comes in PR 4)
- Decoder families other than Gemma 3 (comes in PR 5)
- v2 model runner integration (dev-flag-gated in vllm main; pending)
- Speculative-decoding draft models

## Steering Model

Per-request steering uses a three-tier additive composition model:

```text
effective_prefill = request_base + request_prefill
effective_decode  = request_base + request_decode
```

Each vector entry can be written either as:

- a bare vector: `list[float]`
- a scaled entry: `{"vector": [...], "scale": float}`

Scaled entries are multiplied before addition.

## Enabling Steering

Per-request steering must be enabled explicitly.

```bash
vllm serve google/gemma-3-4b-it \
  --enable-steering \
  --max-steering-configs 4
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--enable-steering` | `False` | Enables per-request steering tables and scheduler admission |
| `--max-steering-configs` | `4` | Maximum distinct per-request steering configs in one batch |

## Hook Points

Steering is applied to the carried residual stream, not to a post-norm
activation that is discarded immediately afterward.

| Hook Point | Meaning |
| --- | --- |
| `pre_attn` | Residual stream before attention |
| `post_attn` | Residual stream after attention |
| `post_mlp` | Residual stream after MLP |

For supported models, these hooks are wired directly into each decoder
layer's forward path. Unused hook points are zero-valued no-ops.

## Per-Request Steering

Per-request steering is configured via `SamplingParams`:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="google/gemma-3-4b-it",
    enable_steering=True,
    max_steering_configs=4,
)

params = SamplingParams(
    max_tokens=64,
    temperature=0.0,
    steering_vectors={
        "post_mlp": {
            15: {"vector": [0.1, 0.2], "scale": 2.0},
        },
    },
    prefill_steering_vectors={
        "pre_attn": {
            15: [0.3, 0.4],
        },
    },
    decode_steering_vectors={
        "pre_attn": {
            15: [0.5, 0.6],
        },
    },
)

outputs = llm.generate(["Hello"], params)
```

Per-request steering is also available through the OpenAI-compatible server
using `extra_body`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "steering_vectors": {
            "post_mlp": {15: [0.1, 0.2]},
        },
        "prefill_steering_vectors": {
            "pre_attn": {15: [0.3, 0.4]},
        },
        "decode_steering_vectors": {
            "pre_attn": {15: [0.5, 0.6]},
        },
    },
)
```

## Runtime Semantics

The main operational rules are:

- Prefill and decode use separate effective steering configs
- Per-request admission is phase-aware
- A request may prefill under one hash and later decode under another
- Prefix-cache keys include prefill steering, not decode-only steering
- Streaming continuation folds prior outputs back into the prompt, so
  prompt/decode boundaries can move between turns
- The scheduler guarantees steering-table capacity before admitting a
  request: by the time a request reaches the worker, its steering row
  can always be allocated. Registration failure at the worker is a hard
  error, not a recoverable condition, because tokens generated under
  wrong steering poison the KV cache permanently

These rules matter for cache correctness and batch admission. See
[Steering Runtime Design](../design/steering_runtime.md) for details.

## Prefix Caching

Steering interacts with automatic prefix caching as follows:

- Different prefill steering must produce different cache keys
- Decode-only steering must not fork prompt KV cache entries
- Streaming continuation must rebuild any block-hash chain whose
  prompt/decode phase boundaries changed

If you use steering with APC, those semantics are required for correctness,
not just performance. See also [Automatic Prefix Caching](automatic_prefix_caching.md).

## Continuous Batching, `torch.compile`, and CUDA Graphs

Steering is designed to work under continuous batching:

- different requests in one batch can use different steering configs
- prefill and decode can coexist in the same batch
- transition from prefill to decode is tracked per request

Steering is also compatible with compiled and graphed execution:

- steering uses persistent GPU buffers updated between steps
- CUDA graph replay reads the current steering table contents
- steering does not require graph partitioning to remain correct

In practice this means steering changes are visible across CUDA graph replays
as long as the model has been wired correctly.

## Operational Limits

- Only the Gemma 3 decoder is wired in this PR
- Per-request steering requires `--enable-steering`
- Distinct steering configs in flight are capped by `--max-steering-configs`

## References

- [Steering Runtime Design](../design/steering_runtime.md)
- [Automatic Prefix Caching](automatic_prefix_caching.md)
