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

Also supported:

- Global steering through HTTP endpoints
- Per-request steering through `SamplingParams`
- Three additive tiers (base / prefill-specific / decode-specific)
- Three hook points: `pre_attn`, `post_attn`, `post_block`
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

## Directional Clamps

In addition to additive vectors, steering supports **directional
projection clamps**: constrain the hidden state's scalar coordinate along
a direction to an interval, per token, at any hook point:

```text
p  = h · v̂                      # current expression of the feature
h' = h + strength · (clip(p, min, max) − p) · v̂
```

A token whose projection is already inside `[min, max]` is untouched, and
everything orthogonal to the direction is always preserved — unlike an
additive vector, which shifts every token by the same amount.

Clamp entries live in `steering_clamps` / `prefill_steering_clamps` /
`decode_steering_clamps` (per-request), the same tier names on
`/v1/steering/set` (`clamps` / `prefill_clamps` / `decode_clamps`,
global), and an optional clamps tier on named modules:

```json
"steering_clamps": {
  "post_block": {
    "20": [
      {"vector": [/* hidden_size floats */], "max": 4.0},
      {"vector": [/* ... */], "value": 8.0, "strength": 0.5}
    ]
  }
}
```

Entry semantics:

- `{"vector": v, "min": lo, "max": hi}` — clamp the projection to
  `[lo, hi]`; either bound may be omitted (one-sided).
- `{"vector": v, "value": c}` — sugar for `min = max = c` (pin the
  feature to a constant expression; `c = 0` is directional ablation).
- `strength` in `[0, 1]` applies a partial correction (default 1.0).
- Directions are **unit-normalized server-side**, so bounds are in
  unit-projection space and portable across vectors. Zero vectors are
  rejected.
- Unlike vectors, tiers merge by **concatenation** (base entries first,
  then phase entries) — each direction is an independent constraint. Up
  to `--steering-config.max_clamp_directions` (default 4) directions per
  (hook, layer) site after composing global + per-request tiers.

Clamps run **after** additive steering at each hook, so the bound holds on
whatever leaves the site. They participate in the steering config hash,
so prefix caching stays correct, and clamp-only requests are admitted
exactly like vector requests.

Picking bounds: capture activations at the target site (the capture
feature), compute `h · v̂` over representative traffic to see the
projection's natural range, then set `min`/`max` relative to it.

### Packed clamp submission format

Each of the three per-request clamp fields (and the `clamps` /
`prefill_clamps` / `decode_clamps` tiers of `/v1/steering/set` and named
modules) also accepts a **binary packed** form that avoids re-sending clamp
directions as JSON float lists and re-parsing/normalizing them per request.
Per hook point, mirroring `SteeringHookPacked`:

```json
"steering_clamps": {
  "post_block": {
    "dtype": "float64",
    "shape": [3, 4096],
    "layer_indices": [20, 20, 21],
    "data": "<base64 contiguous [n, hidden] tensor>",
    "bounds": [[-2.0, 2.0], [null, 4.0], [0.0, 0.0]],
    "strengths": [1.0, 0.5, 1.0]
  }
}
```

- Row `i` is the direction for `layer_indices[i]`; a layer may appear in
  several rows (one per clamp direction). **Row order within a layer is
  preserved** — it is the tier-concat order that the per-site `K` budget
  applies to.
- `bounds` is one `[lo, hi]` pair per row; `strengths` is one value per row
  (both stay as small JSON lists — only the direction vectors are bulk). An
  **infinite** bound is written as JSON `null` (`lo` null → `-inf`, `hi` null
  → `+inf`); all present bounds must be finite.
- Pack directions at **`float64`** for a bit-identical prefix-cache hash
  versus the equivalent JSON submission. Narrower dtypes are accepted but may
  cost a one-time cache miss when the same config is also sent as JSON
  (same trade-off as the packed steering-vector path).
- The packed form travels verbatim to the engine; the direction is
  unit-normalized worker-side, so a packed and a JSON submission of the same
  logical config are interchangeable.

Legacy JSON and packed clamp tiers can be mixed across fields in the same
request (e.g. `steering_clamps` packed, `prefill_steering_clamps` JSON). The
gRPC API carries the same layout as a `ClampHookPacked` message (with
`bounds` flattened to `[lo0, hi0, lo1, hi1, ...]` and infinities as native
`±inf` doubles).

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
| `post_block` | Residual stream after MLP |

For supported models, these hooks are wired directly into each decoder
layer's forward path. Unused hook points are zero-valued no-ops.

## Global Steering API

Global steering endpoints are mounted whenever the server runs (they answer
with a clear error when `--enable-steering` is not set). Mutation endpoints
should be protected with a steering API key on any shared deployment.

### Gating Mutation Endpoints Behind a Steering API Key

Mutating steering endpoints (`POST /v1/steering/set`, `POST
/v1/steering/clear`, `POST /v1/steering/modules/register`, `POST
/v1/steering/modules/unregister`) can optionally be gated behind a
dedicated steering API key, separate from the server-wide `--api-key`.
This lets operators issue a narrower credential for mutating steering
state without handing out the main server key.

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

When configured, requests to the mutation endpoints must include an
`Authorization: Bearer <key>` header; anything else returns
`401 Unauthorized` without dispatching the mutation.  This check
is additive with the server-wide `--api-key` middleware: if both are
set, requests must satisfy both.

The Rust frontend honors the same key for its steering-module routes
(`POST /v1/steering/modules`, `DELETE /v1/steering/modules/{name}`) —
`--steering-api-key` is forwarded to the Rust process, and
`VLLM_STEERING_API_KEY` works there too.

Read-only endpoints (`GET /v1/steering`, `GET /v1/steering/layers`,
`GET /v1/steering/modules`) are **not** gated by this key.

### Set Steering

```bash
curl -X POST http://localhost:8000/v1/steering/set \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "post_block": {
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
    json={"vectors": {"post_block": packed_hook}},
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
        "post_block": {
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
    "post_block": {
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
    "post_block": {
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
    "post_block": {
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

Runtime registration is a steering *mutation*: when `--steering-api-key` /
`VLLM_STEERING_API_KEY` is configured, these requests need the
`Authorization: Bearer <key>` header.

```bash
# Register a module (legacy JSON form)
curl -X POST http://localhost:8000/v1/steering/modules/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "creativity",
    "vectors": {
      "post_block": {"15": [0.1, 0.2, 0.3]}
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
            "post_block": {
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
            "post_block": {15: [0.05, 0.1, 0.15]},
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
- Mutating HTTP endpoints (`/v1/steering/set`, `/v1/steering/clear`,
  `/v1/steering/modules/register`, `/v1/steering/modules/unregister`) can
  be gated behind `--steering-api-key` / `VLLM_STEERING_API_KEY`; protect
  them on any shared deployment
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
# {"layers": {"0": {"hook_points": ["post_block"]}, "1": {"hook_points": ["post_block", "pre_attn"]}, ...}}
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
