# Activation Steering

Activation steering adds precomputed vectors into the residual stream of
decoder layers during inference. It can be used to shift model behavior
without fine-tuning, for example for tone/style changes, behavioral
interventions, or SAE-derived steering vectors.

This page is the user-facing guide to steering in vLLM. For internal
runtime details, see [Steering Runtime Design](../design/steering_runtime.md).

## Supported Scope

Steering is wired into the following decoder architectures:

- Llama family: `llama`, `llama4`, `arcee`, `nemotron`, `nemotron_nas`,
  `granite`, `solar`, `minicpm`, `minicpm3`, `mistral`, `apertus`
- Qwen family: `qwen2`, `qwen3`, `qwen2_moe`, `qwen3_moe`, `qwen3_next`,
  `qwen3_5`
- Gemma family: `gemma`, `gemma2`, `gemma3`, `gemma3n`, `gemma4`
- Mixtral / MoE: `mixtral`, `phimoe`, `deepseek_v2`, `glm4_moe`,
  `glm4_moe_lite`, `exaone_moe`, `granitemoe`, `granitemoeshared`, `dots1`,
  `ernie45_moe`, `olmoe`, `openpangu`, `grok1`, `jais2`, `minimax_m2`,
  `minimax_text_01`, `arctic`, `param2moe`, `flex_olmo`
- GLM / ChatGLM: `glm4`
- InternLM family: `internlm2`, `internlm2_ve`, `interns1_pro`,
  `iquest_loopcoder`
- Olmo family: `olmo`, `olmo2`, `olmo_hybrid`
- Exaone family: `exaone`, `exaone4`
- Phi family: `phi`
- Plamo family: `plamo2`, `plamo3`
- Step family: `step1`, `step3_text`, `step3p5`
- Molmo family: `molmo`, `molmo2`
- Falcon / Baichuan / Command / StableLM: `falcon`, `baichuan`, `commandr`,
  `stablelm`
- Other: `AXK1`, `gpt_neox`, `hyperclovax`, `opt`, `orion`, `ouro`,
  `persimmon`, `seed_oss`, `starcoder2`, `hunyuan_v1`, `mimo_v2_flash`

End-to-end tested with real weights:

- Gemma 3 (primary test target)
- StableLM, step3p5, Mixtral, DeepSeek V2, PhiMoE, GLM4 MoE, Exaone MoE
  (via `*_real_weights` tests in
  `tests/models/language/generation/test_steering.py`)

Other listed architectures have hook wiring and pass small-decoder fixture
tests but have not been validated against released checkpoints.

Manifold-constrained hyper-connection (mHC) steering is wired for
DeepSeek-V4 (`deepseek_v4`), which carries a multi-stream residual rather
than a single residual. See [mHC Steering](#mhc-steering) for the extra
hook points and the per-stream vector format.

Also supported:

- Global steering through HTTP endpoints
- Per-request steering through `SamplingParams`
- Three additive tiers (base / prefill-specific / decode-specific)
- Three hook points on standard models: `pre_attn`, `post_attn`, `post_mlp`
  (mHC models expose a different hook set — see [mHC Steering](#mhc-steering))
- Phase-aware scheduler admission for per-request steering
- Prefix-cache separation for different prefill steering configs
- Continuous batching
- `torch.compile` and CUDA graph execution

Not currently supported:

- v2 model runner integration (dev-flag-gated in vllm main; steering
  integration pending)

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

For supported models, these hooks are wired directly into each decoder
layer's forward path. Unused hook points are zero-valued no-ops.

## mHC Steering

DeepSeek-V4 uses manifold-constrained hyper-connections (mHC): instead of
one residual stream it carries `hc_mult` parallel hidden-size streams that
are mixed per token. Steering is wired at both the single-stream sublayer
boundaries and the multi-stream residual. The hook names are shared with
the activation-capture framework (on branches that include it), so a
tensor can be both captured and steered under one identifier.

| Hook Point | Tensor | Vector shape | Width |
| --- | --- | --- | --- |
| `pre_attn` | single-stream pre-mixed attention input | `(hidden,)` | `hidden` |
| `post_attn` | single-stream attention output | `(hidden,)` | `hidden` |
| `mlp_in` | single-stream pre-mixed FFN input | `(hidden,)` | `hidden` |
| `mlp_out` | single-stream FFN output | `(hidden,)` | `hidden` |
| `mhc_streams_pre_attn` | multi-stream residual entering attention | `(hc_mult, hidden)` | `hc_mult * hidden` |
| `mhc_streams_pre_mlp` | multi-stream residual entering the FFN | `(hc_mult, hidden)` | `hc_mult * hidden` |
| `mhc_streams_final` | final multi-stream residual before the head fold | `(hc_mult, hidden)` | `hc_mult * hidden` |

DeepSeek-V4 has no single-stream `post_mlp` hook — its end-of-layer
residual is the multi-stream tensor, so steer `mhc_streams_pre_mlp` of the
next layer (or `mhc_streams_final` at the tail) instead.

Multi-stream hooks take an **independent vector per stream**. The vector is
supplied flattened to `hc_mult * hidden` values in stream-major order
(stream 0's full hidden vector, then stream 1's, …); to steer every stream
identically, repeat the same `hidden`-length block `hc_mult` times. The
wire format is unchanged from single-stream steering — the packed
`(num_layers, width)` blob and the `SamplingParams` list-of-floats both
just carry the wider row:

```python
import numpy as np

# Steer stream 1 only, on mhc_streams_pre_attn of layer 20.
hc_mult, hidden = 4, 4096
row = np.zeros((hc_mult, hidden), dtype=np.float32)
row[1] = my_direction  # (hidden,)
params = SamplingParams(
    steering_vectors={"mhc_streams_pre_attn": {20: row.reshape(-1).tolist()}},
)
```

`mhc_streams_final` is a model-level hook: it is keyed to the last decoder
layer index, so request it on that layer.

## Global Steering API

Global steering endpoints require `VLLM_SERVER_DEV_MODE=1`.

### Gating Mutation Endpoints Behind a Steering API Key

`POST /v1/steering/set` and `POST /v1/steering/clear` can optionally be
gated behind a dedicated steering API key, separate from the server-wide
`--api-key`.  This lets operators issue a narrower credential for
mutating global steering state without handing out the main server key.

Configure it with either the CLI flag or the env var:

```bash
vllm serve google/gemma-3-4b-it --steering-api-key my-steering-secret

# or
VLLM_STEERING_API_KEY=my-steering-secret vllm serve google/gemma-3-4b-it
```

The CLI flag accepts multiple values to support key rotation:

```bash
vllm serve google/gemma-3-4b-it \
  --steering-api-key primary-key --steering-api-key rotation-key
```

When configured, requests to `/v1/steering/set` and `/v1/steering/clear`
must include an `Authorization: Bearer <key>` header; anything else
returns `401 Unauthorized` without dispatching the mutation.  This check
is additive with the server-wide `--api-key` middleware: if both are
set, requests must satisfy both.

Read-only `GET /v1/steering` and the `/v1/steering/modules/*` endpoints
are **not** gated by this key.

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

Each tier in `/v1/steering/set` also accepts the binary wire shape used by
per-request steering (`SteeringHookPacked`: base64-encoded
`(num_layers, hidden_size)` blob + `layer_indices` + `dtype`/`shape`,
optional per-row `scales`). The server detects the shape per hook and
decodes via `np.frombuffer`, so a single client-side packing helper works
across global and per-request paths:

```python
import base64

import numpy as np
import requests

vec = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
stacked = np.stack([vec], axis=0)  # (num_layers, hidden_size)

packed_hook = {
    "dtype": str(stacked.dtype),
    "shape": list(stacked.shape),
    "layer_indices": [15],
    "data": base64.b64encode(stacked.tobytes()).decode("ascii"),
    "scales": [2.0],  # optional per-row scales
}

requests.post(
    "http://localhost:8000/v1/steering/set",
    json={"vectors": {"post_mlp": packed_hook}},
)
```

Legacy and packed shapes can be mixed across tiers in the same request —
e.g. `vectors` packed, `prefill_vectors` legacy.

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

### OpenAI-Compatible Server

Per-request steering is also available through the OpenAI-compatible server
using `extra_body`. The HTTP fields use a binary wire format: each hook
carries one base64-encoded `(num_layers, hidden_size)` blob plus a sibling
`layer_indices` list. The server decodes via `np.frombuffer` (zero-copy
view) — microseconds vs. the ~10–15 ms per request a JSON `list[float]`
payload would cost on the API-server event loop.

```python
import base64

import numpy as np
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

vec = np.random.standard_normal(2560).astype(np.float16)
stacked = np.stack([vec], axis=0)  # (num_layers, hidden_size)

base = {
    "post_mlp": {
        "dtype": str(stacked.dtype),  # "float16" | "float32" | "float64"
        "shape": list(stacked.shape),
        "layer_indices": [15],
        "data": base64.b64encode(stacked.tobytes()).decode("ascii"),
        # Optional: per-row scales, matched 1:1 with layer_indices.
        # Mirrors the {"vector": [...], "scale": float} form available
        # to the in-process SamplingParams API without baking the
        # multiplier into the bytes.
        "scales": [2.0],
    }
}

response = client.chat.completions.create(
    model="google/gemma-3-4b-it",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"steering_vectors": base},
)
```

`prefill_steering_vectors` and `decode_steering_vectors` accept the same
packed shape for phase-specific additions.

See [`examples/online_serving/openai_steering_client.py`](../../examples/online_serving/openai_steering_client.py)
for a runnable end-to-end client.

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

Any tier in the file may instead use the binary-wire `SteeringHookPacked`
shape — the loader detects the shape per hook. Useful when steering banks
are large enough that the JSON list-of-floats parse becomes the dominant
startup cost:

```json
{
  "vectors": {
    "post_mlp": {
      "dtype": "float32",
      "shape": [2, 2560],
      "layer_indices": [15, 20],
      "data": "<base64 of the (2, 2560) buffer>",
      "scales": [1.0, 2.0]
    }
  }
}
```

### Registering at Runtime (API)

Runtime management endpoints require `VLLM_SERVER_DEV_MODE=1`.

```bash
# Register a module (legacy JSON form)
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

`/v1/steering/modules/register` accepts the same binary-wire shape as
`/v1/steering/set` and the per-request paths. The example below
registers a module whose `vectors` tier is packed; the registry
normalizes back to the legacy shape before storing so worker broadcast
and per-request resolution stay unchanged:

```python
import base64

import numpy as np
import requests

vec = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
stacked = np.stack([vec], axis=0)

requests.post(
    "http://localhost:8000/v1/steering/modules/register",
    json={
        "name": "creativity",
        "vectors": {
            "post_mlp": {
                "dtype": str(stacked.dtype),
                "shape": list(stacked.shape),
                "layer_indices": [15],
                "data": base64.b64encode(stacked.tobytes()).decode("ascii"),
            }
        },
    },
)
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
- Deferred steering registrations use a two-queue priority model:
  prefill→decode transitions are retried before new-request deferrals,
  ensuring in-flight requests get table rows first

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

- See [Supported Scope](#supported-scope) for the list of wired decoder
  architectures
- Global HTTP endpoints are gated behind `VLLM_SERVER_DEV_MODE=1`
- `POST /v1/steering/set` and `POST /v1/steering/clear` can additionally
  be gated behind `--steering-api-key` / `VLLM_STEERING_API_KEY`
- Per-request steering requires `--enable-steering`
- Distinct steering configs in flight are capped by `--max-steering-configs`

## Distributed Execution

Steering is compatible with tensor and pipeline parallelism.

| Configuration        | Supported | Notes                                                   |
|----------------------|-----------|---------------------------------------------------------|
| `TP=1, PP=1`         | yes       | baseline                                                |
| `TP>1, PP=1`         | yes       | vectors replicated on every TP rank                     |
| `TP=1, PP>1`         | yes       | vectors sharded by layer ownership                      |
| `TP>1, PP>1`         | yes       | both behaviors compose                                  |
| Expert parallelism   | untested  |                                                         |
| Speculative decoding | partial   | target model only; draft-model support is separate      |

Global-vector API calls (`/v1/steering/set`, `/v1/steering/clear`) fan out
to every worker with identical arguments via `collective_rpc`. Each worker
stores vectors only for layers it physically owns but allocates table rows
independently for every config it sees — that keeps row IDs in lock-step
across ranks without any cross-rank coordination in the hot path. See
[Distributed Execution](../design/steering_runtime.md#distributed-execution)
in the design doc for the full contract.

### Debug endpoint: `GET /v1/steering/layers`

Returns per-layer hook-point availability aggregated across TP × PP ranks:

```bash
curl http://localhost:8000/v1/steering/layers
# {"layers": {"0": {"hook_points": ["post_mlp"]}, "1": {"hook_points": ["post_mlp", "pre_attn"]}, ...}}
```

Useful to confirm which layers of the loaded model are steerable before
sending a `/v1/steering/set` request.

### TP-rank divergence

TP ranks within the same PP stage must own identical layer sets. If they
do not, `/v1/steering/set` returns HTTP 500 labelled "Server-side
invariant violation" naming the diverging ranks. This indicates a
model-loading asymmetry (e.g., different weights loaded on different
ranks) and not a user error.

## References

- [Steering Runtime Design](../design/steering_runtime.md)
- [Automatic Prefix Caching](automatic_prefix_caching.md)
- [OpenAI-Compatible Server](../serving/openai_compatible_server.md)
