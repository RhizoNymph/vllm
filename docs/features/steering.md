# Activation Steering

Activation steering adds precomputed vectors into the residual stream of
decoder layers during inference. It can be used to shift model behavior
without fine-tuning, for example for tone/style changes, behavioral
interventions, or SAE-derived steering vectors.

This page is the user-facing guide to steering in vLLM. For internal
runtime details, see [Steering Runtime Design](../design/steering_runtime.md).

## Supported Scope

Steering currently supports:

- Global steering through HTTP endpoints
- Per-request steering through `SamplingParams`
- Three additive tiers:
  - base: applied to both phases
  - prefill-specific
  - decode-specific
- Three hook points: `pre_attn`, `post_attn`, `post_mlp`
- Phase-aware scheduler admission for per-request steering
- Prefix-cache separation for different prefill steering configs
- Continuous batching
- `torch.compile` and CUDA graph execution
- Gemma 3

Not currently supported:

- v2 model runner integration

## Steering Model

Steering uses a three-tier additive composition model:

```text
effective_prefill = global_base + global_prefill + request_base + request_prefill
effective_decode  = global_base + global_decode  + request_base + request_decode
```

Each vector entry can be written either as:

- a bare vector: `list[float]`
- a scaled entry: `{"vector": [...], "scale": float}`

Scaled entries are multiplied before addition.

## Enabling Steering

Global steering is always available for steerable models. Per-request
steering must be enabled explicitly.

```bash
# Global steering only
vllm serve google/gemma-3-4b-it

# Per-request steering
vllm serve google/gemma-3-4b-it \
  --enable-steering \
  --max-steering-configs 4
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--enable-steering` | `False` | Enables per-request steering tables and scheduler admission |
| `--max-steering-configs` | `4` | Maximum distinct per-request steering configs in one batch |

Without `--enable-steering`, global steering still works, but per-request
`SamplingParams.steering_vectors` are not admitted as distinct configs.

## Hook Points

Steering is applied to the carried residual stream, not to a post-norm
activation that is discarded immediately afterward.

| Hook Point | Meaning |
| --- | --- |
| `pre_attn` | Residual stream before attention |
| `post_attn` | Residual stream after attention |
| `post_mlp` | Residual stream after MLP |

For Gemma 3, these hooks are wired directly into the decoder layer forward
path. Unused hook points are zero-valued no-ops.

## Global Steering API

Global steering endpoints require `VLLM_SERVER_DEV_MODE=1`.

### Set Steering

```bash
curl -X POST http://localhost:8000/v1/steering/set \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "post_mlp": {
        "15": {"vector": [0.1, 0.2], "scale": 2.0}
      }
    },
    "prefill_vectors": {
      "pre_attn": {
        "15": [0.3, 0.4]
      }
    },
    "decode_vectors": {
      "pre_attn": {
        "15": [0.5, 0.6]
      }
    },
    "replace": false
  }'
```

Important behavior:

- `vectors` affects both prefill and decode
- `prefill_vectors` is additive during prefill only
- `decode_vectors` is additive during decode only
- `replace=true` clears all current steering before applying the new state
- changing global base or prefill steering invalidates prefix-cache reuse

### Clear Steering

```bash
curl -X POST http://localhost:8000/v1/steering/clear
```

### Inspect Steering Status

```bash
curl http://localhost:8000/v1/steering
```

The response reports active hook points and norms for base/prefill/decode
global vectors when present.

## Per-Request Steering

Per-request steering uses the same three-tier model via `SamplingParams`.

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

## Named Steering Modules

Named steering modules let you pre-register steering vector configurations
and reference them by name in requests, avoiding the need to send large
inline vector arrays with every request.

### Registering at Startup (CLI)

```bash
vllm serve google/gemma-3-4b-it \
  --enable-steering \
  --max-steering-configs 4 \
  --steering-modules creativity=/path/to/creativity.json \
                      safety=/path/to/safety.json
```

The JSON file uses the same three-tier format as the global steering API:

```json
{
  "vectors": {
    "post_mlp": {
      "15": [0.1, 0.2, 0.3],
      "20": {"vector": [0.4, 0.5, 0.6], "scale": 2.0}
    }
  },
  "prefill_vectors": {
    "pre_attn": {
      "15": [0.3, 0.4, 0.5]
    }
  },
  "decode_vectors": null
}
```

### Registering at Runtime (API)

Runtime management endpoints require `VLLM_SERVER_DEV_MODE=1`.

```bash
# Register a module
curl -X POST http://localhost:8000/v1/steering/modules/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "creativity",
    "vectors": {
      "post_mlp": {"15": [0.1, 0.2, 0.3]}
    }
  }'

# List modules
curl http://localhost:8000/v1/steering/modules

# Unregister a module
curl -X POST http://localhost:8000/v1/steering/modules/unregister \
  -H "Content-Type: application/json" \
  -d '{"name": "creativity"}'
```

### Using Named Modules in Requests

Reference a named module via `steering_name` in the OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[{"role": "user", "content": "Write a poem"}],
    extra_body={
        "steering_name": "creativity",
    },
)
```

### Composing Named and Inline Vectors

Named modules and inline vectors compose additively. When both are
provided, the named module's vectors are merged with inline vectors
per-tier before being processed:

```python
response = client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[{"role": "user", "content": "Write a poem"}],
    extra_body={
        "steering_name": "creativity",
        "steering_vectors": {
            "post_mlp": {15: [0.05, 0.1, 0.15]},
        },
    },
)
```

In this example, the effective base vectors are `creativity.vectors + inline steering_vectors` (additive per hook/layer).

## Runtime Semantics

The main operational rules are:

- Prefill and decode use separate effective steering configs
- Per-request admission is phase-aware
- A request may prefill under one hash and later decode under another
- Prefix-cache keys include prefill steering, not decode-only steering
- Streaming continuation folds prior outputs back into the prompt, so
  prompt/decode boundaries can move between turns

These rules matter for cache correctness and batch admission. See
[Steering Runtime Design](../design/steering_runtime.md) for details.

## Prefix Caching

Steering interacts with automatic prefix caching as follows:

- Different prefill steering must produce different cache keys
- Decode-only steering must not fork prompt KV cache entries
- Global base/prefill changes require cache invalidation
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

- Gemma 3 is the only model family currently wired for steering
- Global HTTP endpoints are gated behind `VLLM_SERVER_DEV_MODE=1`
- Per-request steering requires `--enable-steering`
- Distinct steering configs in flight are capped by `--max-steering-configs`

## References

- [Steering Runtime Design](../design/steering_runtime.md)
- [Automatic Prefix Caching](automatic_prefix_caching.md)
- [OpenAI-Compatible Server](../serving/openai_compatible_server.md)
