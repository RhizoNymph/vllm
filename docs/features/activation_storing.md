# Per-Request Activation Storing

## Overview

Per-request activation storing lets vLLM clients capture residual-stream
activations from the model's forward pass and write them directly to a
shared filesystem (typically an NFS-mounted NAS, but any POSIX
filesystem works). Clients opt in per request by passing an
`activation_storing` field alongside their sampling parameters; the
server writes raw activation bytes and a sidecar JSON manifest under a
columnar directory layout designed for downstream interpretability
workloads (SAE training, linear probe training, steering-vector
construction).

The bytes never ride the API response — only a small status pointer
(paths written, error state) flows back. This makes the feature
tractable at realistic scales where returning activations inline would
be intractable (thousands of requests × many layers × full hidden
states can easily produce gigabytes per request).

## Motivation

### Target workloads

**Sparse autoencoder (SAE) training.** An SAE is trained on one layer's
activations across a large corpus. The canonical workflow is: run
thousands or millions of prompts through the model, collect the
residual stream at one `(layer, hook)` point, and feed the flat vector
stream to an autoencoder trainer. Today, users either patch
HuggingFace hooks into a stand-alone `transformers` copy of the model
or they write custom vLLM forks. Neither scales well or plays nicely
with vLLM's batching and PagedAttention.

**Linear probe training.** A probe is a small classifier on top of
residual activations at a fixed `(layer, hook, position)`. Training
data is a corpus of (prompt, label) pairs. The user runs all prompts
through vLLM, collects one row per prompt (typically at the last
prompt token), and trains a logistic regression or MLP on the
collected rows. This workflow benefits from deterministic per-request
filenames so the labeling job can line up captures with labels by
path.

**Steering vector construction.** The standard recipe diffs the mean
activation of "positive" prompts against the mean of "negative"
prompts. The same dataset structure as probe training; the downstream
consumer just computes means and differences instead of training a
classifier.

**Interpretability debugging.** One-off experiments where a researcher
wants to poke at a specific prompt's internals. The filesystem-backed
design means the activations are still there after the request
finishes — the researcher can reload them in a notebook, run
ad-hoc analyses, and iterate without re-running inference.

### Why not return bytes via the API

Early designs considered inlining base64-encoded bytes in the
OpenAI-compatible response. Math killed it:

- A 70B model with 80 hidden layers, `bfloat16` residuals
  (`hidden_size = 8192`, 2 bytes/element), and a 1024-token prompt
  producing activations at every token for every layer yields
  `80 × 1024 × 8192 × 2 = 1.28 GB` per request.
- Base64 inflates that to ~1.7 GB.
- JSON parsing, HTTP buffering, and client-side memory pressure all
  scale poorly above ~100 MB.
- Even the "sparse" case (a few layers, a few token positions) is
  awkward to stream back when the same data could be one POSIX
  `write()` call on the worker.

Writing to a shared filesystem sidesteps every one of those problems.
Clients get predictable paths, operators can mount the NAS from
training machines, and the bytes are already in a format that
`numpy.memmap` / `torch.frombuffer` can load directly.

## Non-Goals

- **Live consumption during generation.** Captures are finalized on
  request completion. Readers that want to see activations as tokens
  are generated (e.g., online interpretability dashboards) are out of
  scope. Files are incomplete until the request finishes.
- **Distributed capture across tensor or pipeline parallel ranks.**
  TP > 1 and PP > 1 are rejected in v1. Handling them cleanly
  requires cross-rank residual collection and a rendezvous protocol
  that is worth a separate project.
- **Compression, deduplication, or columnar formats.** The `.bin`
  files are raw row-major bytes. No Parquet, no Zarr, no Arrow, no
  compression. Downstream tools can repack after the fact if they
  need to.
- **A managed catalog or query engine.** The filesystem is the
  catalog. `ls`, `find`, `glob`, and sidecar JSON are the query
  interface. Integrating with W&B/MLflow/etc. is left to the user.
- **Access control.** Writes inherit whatever POSIX permissions the
  vLLM process has on the target directory. Multi-tenant isolation
  is the operator's problem.
- **Schema evolution beyond the v1 fields.** Sidecar JSON gains
  optional fields over time; removed fields require a major bump.

## Glossary

- **Residual stream**: the additive stream that passes through each
  decoder layer, joined by attention output and MLP output. The
  "pristine" residual at a hook point is the value before any
  steering vector has been added.
- **Hook point**: one of three well-defined positions inside a
  decoder layer where the residual is observable — `pre_attn`,
  `post_attn`, `post_mlp`. These are the same three points the
  existing `steering` feature writes to.
- **Logical position**: an absolute token index into the concatenated
  prompt + generated token sequence for a single request. Position
  0 is the first prompt token; position `len(prompt)` is the first
  generated token.
- **Absolute batch position**: the row index into the flat batched
  token dimension the runner feeds into the model. Not the same as
  logical position — the runner sorts multiple requests'
  tokens back-to-back into one contiguous buffer.
- **Tag**: a client-provided grouping label, slugged server-side.
  Purely organizational — the server doesn't interpret tag contents.
- **`request_id` (client-provided)**: a client-chosen deterministic
  identifier used as the filename stem. Distinct from vLLM's
  internal request id (recorded separately in the sidecar).

## User-facing API

### Server startup

```bash
vllm serve meta-llama/Llama-3-8B \
    --activation-storing /mnt/nas/activations \
    --activation-storing-writer-queue-size 1024 \
    --activation-storing-writer-timeout-seconds 180 \
    --activation-storing-writer-threads 4 \
    --activation-storing-on-collision overwrite \
    --activation-storing-max-bytes-per-request 0
```

Python equivalent:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-8B",
    activation_storing="/mnt/nas/activations",
    activation_storing_writer_queue_size=1024,
    activation_storing_writer_timeout_seconds=180,
    activation_storing_writer_threads=4,
    activation_storing_on_collision="overwrite",
)
```

| Flag | Default | Purpose |
|---|---|---|
| `--activation-storing` | unset | Root directory for all captures. Feature is disabled when unset. |
| `--activation-storing-writer-queue-size` | `1024` | Bounded queue between the capture manager and the writer thread pool. |
| `--activation-storing-writer-timeout-seconds` | `180` | Per-write timeout; failures surface as `partial_error`. |
| `--activation-storing-writer-threads` | `4` | Writer thread pool size. Raise if you have many concurrent slow-disk writes. |
| `--activation-storing-on-collision` | `overwrite` | `overwrite` / `error` / `suffix`. |
| `--activation-storing-max-bytes-per-request` | `0` | Per-request byte cap (0 = unbounded). Estimated at admission time; requests over cap are rejected with HTTP 400. |

When `--activation-storing` is unset, any per-request
`activation_storing` field is rejected with HTTP 400. The capture
custom op is constant-folded out of `torch.compile` traces so there's
zero runtime cost on the cold path.

### Per-request schema (OpenAI extra_body)

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3-8B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "extra_body": {
            "activation_storing": {
                "request_id": "probe_train_0001",
                "tag": "capital-probe",
                "hooks": {
                    "post_mlp": [12, 16, 20, 24]
                },
                "positions": "last_prompt"
            }
        }
    },
    timeout=60,
).json()

print(response["activation_storage"])
# {
#   "status": "ok",
#   "error": null,
#   "paths": [
#     "/mnt/nas/activations/meta-llama/Llama-3-8B/bfloat16/capital-probe/12/post_mlp/probe_train_0001.bin",
#     "/mnt/nas/activations/meta-llama/Llama-3-8B/bfloat16/capital-probe/12/post_mlp/probe_train_0001.json",
#     "/mnt/nas/activations/meta-llama/Llama-3-8B/bfloat16/capital-probe/16/post_mlp/probe_train_0001.bin",
#     ... (one pair per (layer, hook))
#   ]
# }
```

### Per-request schema (offline Python)

```python
from vllm import LLM, SamplingParams
from vllm.config.activation_storing_types import ActivationStoringSpec

llm = LLM(
    model="meta-llama/Llama-3-8B",
    activation_storing="/mnt/nas/activations",
)

sampling_params = SamplingParams(
    max_tokens=16,
    activation_storing=ActivationStoringSpec(
        request_id="probe_train_0001",
        tag="capital-probe",
        hooks={"post_mlp": [12, 16, 20, 24]},
        positions="last_prompt",
    ),
)

result = llm.generate(["What is the capital of France?"], sampling_params)
print(result[0].activation_storage.paths)
```

### Request schema reference

```python
ActivationStoringSpec = {
    "request_id": str,           # required. Client-chosen filename stem. Slugged server-side.
    "tag": str,                  # required. Groups batch runs. Slugged server-side.
    "hooks": dict[str, HookLayerSelector],   # required, non-empty.
    "positions": PositionSelector,           # required, no server-side default.
}

HookLayerSelector = (
    "all"                                   # shorthand for every layer
    | list[int]                             # explicit list of layer indices
    | {                                     # mixed form: union of both
        "layers": list[int],                # optional
        "ranges": list[list[int]],          # optional; inclusive both ends
      }
)

PositionSelector = (
    "last_prompt"                           # final prompt token (one row)
    | "all_prompt"                          # every prompt token
    | "all_generated"                       # every generated token
    | "all"                                 # every prompt + generated token
    | list[int]                             # explicit absolute indices
)
```

#### Layer selector examples

```python
# Every layer under one hook
"hooks": {"post_mlp": "all"}

# Two hooks, different layer subsets
"hooks": {
    "post_mlp": [12, 24],
    "pre_attn": [24],
}

# Mixed: explicit layers plus ranges
"hooks": {
    "post_mlp": {
        "layers": [1, 31],
        "ranges": [[10, 20], [25, 28]],
    }
}
# Resolves to: [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 31]
```

Ranges are **inclusive on both ends** (matching human intuition of
"layers 10 through 20") rather than Python's half-open `range(10, 20)`
semantics. The server flattens `layers` and `ranges` into a single
sorted, deduped list and validates that every entry is within
`[0, num_hidden_layers)`.

#### Position selector semantics

| Value | Resolved to | Typical use case |
|---|---|---|
| `"last_prompt"` | `[num_prompt_tokens - 1]` | Linear probe training, most mechinterp defaults |
| `"all_prompt"` | `range(0, num_prompt_tokens)` | Whole-sequence probing, attention analysis |
| `"all_generated"` | `range(num_prompt_tokens, num_prompt_tokens + num_generated)` | Output-conditioned analysis |
| `"all"` | `range(0, num_prompt_tokens + num_generated)` | SAE training corpora, full-sequence interp |
| `list[int]` | The literal list | Targeted probing at specific indices |

Resolution timing:

- `"last_prompt"` and `"all_prompt"` are resolved at registration time
  when the worker first sees the request and knows
  `num_prompt_tokens`.
- `"all_generated"` and `"all"` grow across decode steps — the
  writer appends new rows each step and the final shape is known
  only at request completion.
- Explicit `list[int]` positions are validated at admission time
  against `[0, max_model_len)` and filtered across steps as each
  position's owning token passes through the forward.

Positions that fall below the request's `initial_num_computed_tokens`
(prefix-cache hits that were never forwarded) are rejected at
admission time, not silently dropped.

### Response schema

Responses carry a small pointer, never bytes:

```json
{
  "id": "chatcmpl-...",
  "choices": [...],
  "activation_storage": {
    "status": "ok",
    "error": null,
    "paths": [
      "/mnt/nas/activations/meta-llama/Llama-3-8B/bfloat16/capital-probe/12/post_mlp/probe_train_0001.bin",
      "/mnt/nas/activations/meta-llama/Llama-3-8B/bfloat16/capital-probe/12/post_mlp/probe_train_0001.json",
      ...
    ]
  }
}
```

| `status` | Meaning |
|---|---|
| `"ok"` | All writes completed successfully. |
| `"partial_error"` | Some writes succeeded, some failed. `error` describes the first failure. Partial `.tmp` files remain on disk for inspection. |
| `"error"` | Capture failed before any bytes were written. Text generation still succeeded. |
| `"not_requested"` | Request did not include `activation_storing`. |

Streaming requests receive `activation_storage` in the final SSE
frame's `usage` block, alongside the final token counts. Text
streaming and activation writes run concurrently — the API
client sees tokens arriving normally while the worker writes bytes to
disk in parallel via the background writer thread pool.

## Filesystem layout

### Path structure

```
{root}/
  {model_slug}/
    {model_dtype}/
      {tag_slug}/
        {layer_idx}/
          {hook_name}/
            {request_id_slug}.bin
            {request_id_slug}.json
```

### Worked example

Server: `--activation-storing /mnt/nas/activations`
Model: `meta-llama/Llama-3-8B` loaded in `bfloat16`
Request:

```python
{
    "request_id": "prompt_0042",
    "tag": "mnist-probe",
    "hooks": {"post_mlp": [12, 24]},
    "positions": "last_prompt",
}
```

Writes to:

```
/mnt/nas/activations/
  meta-llama/Llama-3-8B/
    bfloat16/
      mnist-probe/
        12/
          post_mlp/
            prompt_0042.bin     ← shape (1, 4096), bfloat16
            prompt_0042.json
        24/
          post_mlp/
            prompt_0042.bin
            prompt_0042.json
```

### Model slug resolution

The `{model_slug}` segment is resolved in this priority order:

1. **`model_config.served_model_name`**, if the operator set it
   (e.g., `--served-model-name my-llama`). Slugged via
   `re.sub(r'[^a-zA-Z0-9._-]', '_', name)`.
2. **HF-style repo string** (`org/name` with exactly two segments,
   no leading `/`, no `..`). Preserved as-is, so the slash becomes a
   real directory separator. This naturally groups captures by
   organization.
3. **Anything else** (local paths, single-segment names) slugged
   whole via the same regex.

Examples:

| `model_config.model` | `served_model_name` | Resulting slug |
|---|---|---|
| `meta-llama/Llama-3-8B` | (unset) | `meta-llama/Llama-3-8B` |
| `/mnt/models/my-llama` | (unset) | `_mnt_models_my-llama` |
| `meta-llama/Llama-3-8B` | `llama3` | `llama3` |
| `my-custom-model` | (unset) | `my-custom-model` |

### Dtype segment

`{model_dtype}` is always the model's native residual dtype (e.g.,
`bfloat16`, `float16`, `float32`). It is baked into the path so that
different quantizations of the same model name never collide. Clients
cannot override it.

### Tag slugging

`tag` is slugged via `re.sub(r'[^a-zA-Z0-9._-]', '_', tag)` and
rejected at admission time if it contains `..`, starts with `/`, or
exceeds 256 characters. Nested tags with `/` are **not** supported;
use flat tags with hyphens or underscores (`experiment-01-retry` not
`experiment/01/retry`).

### Request id slugging

`request_id` is slugged with the same rule as `tag` and capped at 256
characters. Collisions on `(tag, layer, hook, request_id)` are
handled by the configured policy:

- **`overwrite`** (default): truncate the existing `.bin` and `.json`
  at write time.
- **`error`**: fail the request with HTTP 400 at admission time.
- **`suffix`**: append a timestamp suffix `.{unix_ms}` to
  `request_id` before writing. Breaks determinism — clients must
  parse the response to find the actual path.

## File format

### `.bin`

Raw bytes, row-major `(num_rows, hidden_size)` in `model_dtype`. No
header, no framing, no padding. Readers `np.frombuffer` (or
`np.memmap` for large files) and reshape using the sidecar manifest:

```python
import json
import numpy as np

manifest = json.load(open("probe_train_0001.json"))
# numpy has no bfloat16; interpret bytes as uint16 when dtype is bf16
np_dtype = {
    "float16": np.float16,
    "bfloat16": np.uint16,
    "float32": np.float32,
}[manifest["dtype"]]

arr = np.frombuffer(
    open("probe_train_0001.bin", "rb").read(),
    dtype=np_dtype,
).reshape(manifest["shape"])

# For bf16, reinterpret bits via torch if you need math:
import torch
t = torch.from_numpy(arr).view(torch.bfloat16)
```

Multi-step captures (`"all_generated"`, `"all"`) append rows to the
`.bin.tmp` file as each decode step produces new residuals. The file
is in its final shape only after the request completes and the
rename from `.tmp` fires.

### `.json` sidecar manifest

```json
{
  "request_id": "prompt_0042",
  "tag": "mnist-probe",
  "model": "meta-llama/Llama-3-8B",
  "model_dtype": "bfloat16",
  "layer": 12,
  "hook": "post_mlp",
  "shape": [1, 4096],
  "dtype": "bfloat16",
  "element_size": 2,
  "positions": [42],
  "position_kind": "last_prompt",
  "last_prompt_token_index": 42,
  "prompt_token_ids": [1, 2, 3, 42],
  "generated_token_ids": [1854, 11, 502],
  "created_at": "2026-04-14T15:30:00.123Z",
  "finalized_at": "2026-04-14T15:30:00.456Z",
  "vllm_internal_request_id": "cmpl-abc123",
  "capture_status": "ok",
  "capture_error": null
}
```

| Field | Required | Notes |
|---|---|---|
| `request_id`, `tag`, `model`, `model_dtype` | yes | Echoes the slugged path components. |
| `layer`, `hook` | yes | This sidecar's `(layer, hook)` pair. |
| `shape` | yes | `[num_rows, hidden_size]` of the final `.bin`. |
| `dtype`, `element_size` | yes | Matches `model_dtype`. |
| `positions` | yes | The resolved absolute token indices captured in this file. |
| `position_kind` | yes | One of `last_prompt` / `all_prompt` / `all_generated` / `all` / `explicit`. |
| `last_prompt_token_index` | yes | Always present. Lets readers find the prompt/generation boundary without recounting. |
| `prompt_token_ids`, `generated_token_ids` | yes | Full token lists for reproducibility. |
| `created_at`, `finalized_at` | yes | ISO 8601 UTC timestamps. |
| `vllm_internal_request_id` | yes | vLLM's own request id (distinct from client-provided `request_id`). |
| `capture_status` | yes | `ok` / `partial_error` / `error`. |
| `capture_error` | yes | Human-readable error message or `null`. |

## Architecture

### High-level data flow

```
  Client                Frontend            Scheduler          Worker          Writer pool          Filesystem
  ------                --------            ---------          ------          -----------          ----------
    │                      │                    │                 │                 │                   │
    │  POST /v1/chat/...   │                    │                 │                 │                   │
    │   + activation_      │                    │                 │                 │                   │
    │     storing spec     │                    │                 │                 │                   │
    ├─────────────────────▶│                    │                 │                 │                   │
    │                      │ validate spec      │                 │                 │                   │
    │                      │ (admission)        │                 │                 │                   │
    │                      │ to_sampling_params │                 │                 │                   │
    │                      │────────────────────▶ enqueue request │                 │                   │
    │                      │                    │─────────────────▶ _prepare_        │                   │
    │                      │                    │                 │  activation_     │                   │
    │                      │                    │                 │  storing_step    │                   │
    │                      │                    │                 │                 │                   │
    │                      │                    │                 │ forward pass    │                   │
    │                      │                    │                 │  (custom op     │                   │
    │                      │                    │                 │   writes GPU    │                   │
    │                      │                    │                 │   scratch)      │                   │
    │                      │                    │                 │                 │                   │
    │                      │                    │                 │ _finalize_      │                   │
    │                      │                    │                 │  activation_    │                   │
    │                      │                    │                 │  storing_step:  │                   │
    │                      │                    │                 │  GPU→pinned CPU │                   │
    │                      │                    │                 │  enqueue writes │                   │
    │                      │                    │                 │─────────────────▶ append to .bin.tmp│
    │                      │                    │                 │                 │───────────────────▶
    │                      │                    │                 │                 │                   │
    │                      │                    │                 │ (repeat per     │                   │
    │                      │                    │                 │  decode step)   │                   │
    │                      │                    │                 │                 │                   │
    │                      │                    │                 │ request         │                   │
    │                      │                    │                 │ finished        │                   │
    │                      │                    │                 │ ───────────────▶ write sidecar.tmp  │
    │                      │                    │                 │                 │ rename .tmp→final │
    │                      │                    │                 │                 │───────────────────▶
    │                      │                    │                 │                 │                   │
    │                      │◀─ RequestOutput + activation_storage ◀─ EngineCoreOutput +                  │
    │                      │                                        capture_status                      │
    │◀ response with pointer                                                                              │
```

### Server-side components

The feature adds or extends the following components. File paths are
the intended locations on the `feat/activation-storing` branch; see
the plan file for the exact file list.

**`ActivationStoringConfig` (`vllm/config/activation_storing.py`)**
— server-global configuration object mirroring `SteeringConfig`.
Holds the root path, writer pool parameters, collision policy, and
byte cap. Wired into `VllmConfig.activation_storing_config`.

**`ActivationStoringSpec` (`vllm/config/activation_storing_types.py`)**
— per-request pydantic dataclass with `request_id`, `tag`, `hooks`,
`positions`. Includes the layer selector and position selector helper
types, expansion logic for `{layers, ranges}` and `"all"`, and
validation against `num_hidden_layers`.

**`ActivationCaptureManager` (`vllm/model_executor/layers/activation_capture.py`)**
— lives on the model runner. Holds the active-request dict, builds
per-step `StepCapturePlan` objects, exposes `on_hook` which the
registered `torch.ops.vllm.capture_residual` custom op calls from
inside `apply_layer_steering` (one-line piggyback on the steering
hook helper; zero model-file churn). Mirrors the design of the
existing `SteeringManager`.

**`ActivationWriter` (`vllm/v1/worker/activation_writer.py`)** —
thread pool that drains a bounded `queue.Queue` of `WriteTask`
objects. Each task holds a target path, bytes payload, an `append`
flag, and an optional sidecar payload to write + rename. Writer
threads handle atomic rename on finalization (`mkdir -p`, write
`.tmp`, `fsync`, rename). Failures surface as `capture_status =
partial_error` on the owning request without aborting text
generation.

**`activation_storing_validation.py`
(`vllm/entrypoints/openai/`)** — admission-time validation: config
enabled, TP/PP rejection, layer-in-range, position-not-in-prefix-
cache, byte-budget. Called before `to_sampling_params` so bad
requests get fast feedback.

**`gpu_model_runner.py` additions** — `_prepare_activation_storing_step`
registers new requests against the manager and builds the step plan.
`_finalize_activation_storing_step` drains scratch into pinned CPU,
enqueues writer tasks. Both are called from `execute_model` right
next to the existing steering-buffer update path.

### The custom op

`torch.ops.vllm.capture_residual(hidden_states, layer_idx, hook_id)`
is registered via `direct_register_custom_op` with a fake impl. The
real impl reads the process-global active
`ActivationCaptureManager` and calls `on_hook(layer_idx, hook_name,
hidden_states)`. When the manager has no plan for this
`(layer, hook)`, the op is a dict lookup + return unchanged —
fast-path cost is a handful of Python instructions per hook call.

When `--activation-storing` is unset at server start, the module
global `_ACTIVE_CAPTURE_MANAGER` is `None`. The
`maybe_capture_residual` helper that `apply_layer_steering` calls
checks the global once and skips the custom op entirely. Under
`torch.compile`, this `None`-check is constant-folded at trace time
and the compiled graph contains no capture-related ops whatsoever.

### Per-step plan

The manager builds a `StepCapturePlan` per forward pass:

```python
@dataclass
class StepCapturePlan:
    # GPU int64 tensor of absolute batch positions, per (layer, hook)
    gather_indices: dict[tuple[int, str], torch.Tensor]
    # GPU scratch receiving index_select output
    scratch_gpu: dict[tuple[int, str], torch.Tensor]
    # Row-level metadata tying each scratch row back to (req_id, logical_pos)
    entries: list[CapturePositionEntry]
    # Request ids that had admission-time errors surfaced this step
    request_errors: dict[str, str]
```

Plan building walks `input_batch.req_ids`, computes each request's
`(token_offset, num_computed, num_prompt)` from the same fields the
steering-index builder uses, and intersects the spec's absolute
positions with each step's token slice. For each matching position
and each `(layer, hook)` in the spec, the plan adds an entry
mapping the scratch row back to the source `(req_id, logical_pos)`.

### Forward pass

Inside the compiled forward, each hook call in each decoder layer
invokes `apply_layer_steering` which now calls
`maybe_capture_residual` before adding the steering vector. The
capture custom op does:

```python
def on_hook(self, layer_idx, hook_name, hidden_states):
    plan = self._step_plan
    if plan is None:
        return
    key = (layer_idx, hook_name)
    idx = plan.gather_indices.get(key)
    if idx is None:
        return
    gathered = hidden_states.index_select(0, idx)
    if gathered.dtype != plan.scratch_dtype[key]:
        gathered = gathered.to(plan.scratch_dtype[key])
    plan.scratch_gpu[key] = gathered
```

No GPU scratch allocation happens for `(layer, hook)` pairs that no
request wants this step. The custom op is compiled-graph friendly
because it's opaque.

### Finalize and enqueue

After `execute_model` returns,
`_finalize_activation_storing_step` runs:

1. For each `(layer, hook)` scratch tensor, allocate a pinned CPU
   destination and issue a `non_blocking=True` copy.
2. Synchronize the accelerator **once** for the whole step.
3. Walk `plan.entries` in batch order, chunking rows per
   `(req_id, layer, hook)`. For each chunk, compute the destination
   path, serialize the bytes, and enqueue a `WriteTask` on the
   writer pool.

Multi-step captures reuse the same `.bin.tmp` file — each step's
bytes are appended. The writer keeps a per-`(req_id, layer, hook)`
open file descriptor cache with LRU eviction so high-concurrency
batches don't exhaust FD limits.

### Finalization

When the scheduler signals that a request has finished (whatever the
finish reason), the manager:

1. Flushes any remaining scratch for that request to the writer.
2. Builds the sidecar JSON payload from resolved positions, prompt
   and generation token ids, and capture status accumulated from
   writer task results.
3. Enqueues a `FinalizeTask` per `(layer, hook)`: write sidecar
   `.json.tmp`, rename `.bin.tmp` → `.bin`, rename `.json.tmp` →
   `.json`, close and evict the FD cache entry.
4. Emits the `activation_storage` status on the next
   `EngineCoreOutput` for the request, which the frontend threads
   into `RequestOutput.activation_storage`.

If a writer task fails or times out, the manager records the error
against the request and sets `capture_status` to `partial_error` on
the finalized sidecar (written anyway, to explain what went wrong).
`.tmp` files from failed writes are left in place — operators clean
them up out of band.

### Writer pool semantics

**Bounded queue.** `queue.Queue(maxsize=queue_size)`. When full,
`put` blocks the caller (the model runner's finalize step) until a
slot frees. This creates natural backpressure — a slow disk will
stall the engine before it runs out of memory. The block is bounded
by `writer_timeout_seconds`; on timeout, the write is marked failed
and the request gets `partial_error`.

**Thread count.** Defaults to 4. Each thread services its own
sub-queue partition keyed by `hash(request_id) % num_threads` so
multi-step captures for a single request always go through the same
thread and preserve append ordering without explicit locking.

**Fsync.** Every `.bin.tmp` is `fsync`'d before rename. Sidecar
`.json.tmp` is `fsync`'d before rename. The parent directory is not
`fsync`'d — crash recovery is not promised, and NFS parent-directory
`fsync` is flaky on many implementations.

**Graceful shutdown.** On engine shutdown the writer drains the
queue with a bounded grace period (default 30 seconds). Remaining
tasks are dropped and their owning requests get
`capture_status = "error"` with a shutdown explanation.

## Failure modes and recovery

| Failure | Behavior |
|---|---|
| Target root missing / unwritable at startup | Engine init fails loudly with a clear error. |
| Disk full mid-write | Writer marks that request's capture as `partial_error`. `.tmp` files remain on disk. Text generation continues. Subsequent requests continue to succeed until they hit the same limit. |
| NFS partition during generation | Same as disk full. Writer timeout triggers `partial_error`. No engine-wide failure. |
| Writer queue full for longer than timeout | That request's capture fails with `partial_error`. Queue depth is logged at `WARNING`. |
| Permission denied on a specific `(tag, layer, hook)` subdir | `partial_error` for that request, with a path in the error message. |
| vLLM process crash mid-write | `.tmp` files are left orphaned. Operators sweep manually. No automatic recovery. |
| Concurrent requests writing to the same `(tag, layer, hook, request_id)` | Governed by `--activation-storing-on-collision`. Default `overwrite` semantics; opt into `error` or `suffix` if reruns shouldn't clobber. |

Partial captures are intentionally preserved rather than deleted.
For interpretability workflows, partial data often still has value
(debugging a batch that died 80% through). The sidecar `capture_status
= "partial_error"` plus `capture_error` description documents
exactly what went wrong, and a tool like `jq '.capture_status' */*/*/*/*.json`
can locate all affected captures.

## Reading captures downstream

### numpy + torch

```python
import json
from pathlib import Path

import numpy as np
import torch

def load_capture(bin_path: str | Path) -> tuple[torch.Tensor, dict]:
    bin_path = Path(bin_path)
    manifest = json.loads(bin_path.with_suffix(".json").read_text())

    # numpy lacks bf16, so we route bf16 through torch's bits view
    if manifest["dtype"] == "bfloat16":
        raw = np.frombuffer(bin_path.read_bytes(), dtype=np.uint16)
        tensor = torch.from_numpy(raw.copy()).view(torch.bfloat16)
    else:
        np_dtype = {"float16": np.float16, "float32": np.float32}[manifest["dtype"]]
        tensor = torch.from_numpy(
            np.frombuffer(bin_path.read_bytes(), dtype=np_dtype).copy()
        )

    return tensor.reshape(manifest["shape"]), manifest


activations, meta = load_capture(
    "/mnt/nas/activations/meta-llama/Llama-3-8B/bfloat16/mnist-probe/12/post_mlp/prompt_0042.bin"
)
print(activations.shape, meta["last_prompt_token_index"])
```

### Glob-driven batch loading for probe training

```python
import glob
import torch

bin_files = sorted(glob.glob(
    "/mnt/nas/activations/meta-llama/Llama-3-8B/*/mnist-probe/12/post_mlp/*.bin"
))

X = []
y = []
labels = json.load(open("/home/me/datasets/mnist-probe-labels.json"))

for bin_path in bin_files:
    activations, meta = load_capture(bin_path)
    X.append(activations.squeeze(0))               # (hidden_size,)
    y.append(labels[meta["request_id"]])

X = torch.stack(X)                                 # (N, hidden_size)
y = torch.tensor(y)

# train a linear probe on (X, y) here
```

### mmap for SAE-scale corpora

```python
import numpy as np

# For huge captures, skip the .read_bytes() copy and use memmap
raw = np.memmap(
    "/mnt/nas/activations/meta-llama/Llama-3-8B/bfloat16/sae-train/12/post_mlp/prompt_0042.bin",
    dtype=np.uint16,   # bf16 bytes
    mode="r",
)
# reshape per sidecar manifest; each row is (hidden_size,)
```

## Worked examples

### Example 1 — Linear probe training on last-token activations

**Goal.** Train a classifier to predict a label from layer-12
`post_mlp` activations at the last prompt token.

**Client loop.**

```python
import httpx

for i, (prompt, label) in enumerate(dataset):
    httpx.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "meta-llama/Llama-3-8B",
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 1,    # we only care about the activation, not generation
            "extra_body": {
                "activation_storing": {
                    "request_id": f"train_{i:06d}",
                    "tag": "mnist-probe-v1",
                    "hooks": {"post_mlp": [12]},
                    "positions": "last_prompt",
                }
            },
        },
        timeout=60,
    )
```

**Resulting layout.**

```
/mnt/nas/activations/meta-llama/Llama-3-8B/bfloat16/mnist-probe-v1/12/post_mlp/
    train_000000.bin   train_000000.json
    train_000001.bin   train_000001.json
    ...
```

Every file is `(1, hidden_size)` — one row per prompt. Glob the
directory, `load_capture` each file, stack into a matrix, train the
probe.

### Example 2 — Full SAE training corpus on one layer

**Goal.** Collect every token's post-MLP residual at layer 12 across
a training corpus.

**Client loop.** Same as above, but with `"positions": "all_prompt"`
so every prompt token contributes a row. For a 10k-prompt corpus
averaging 200 tokens each, this produces 2M rows — each a
`hidden_size`-dim vector. At `bfloat16` and `hidden_size = 4096`,
that's `2M × 4096 × 2 bytes = 16 GB` per layer per hook. On NFS,
that's manageable; on the API response it would not have been.

### Example 3 — Contrastive steering vector construction

**Goal.** Compute the mean residual at layer 12 for "positive"
prompts and subtract the mean for "negative" prompts, producing a
steering vector that can be fed back into the steering feature.

**Setup.** Two tags: `pos-stim` and `neg-stim`. Same layer, same
hook, same position kind (`last_prompt`). The downstream script:

```python
pos = torch.stack([load_capture(f)[0].squeeze(0) for f in glob.glob(".../pos-stim/12/post_mlp/*.bin")])
neg = torch.stack([load_capture(f)[0].squeeze(0) for f in glob.glob(".../neg-stim/12/post_mlp/*.bin")])
steering_vector = pos.mean(0) - neg.mean(0)
```

Feed the resulting vector back as a request-scoped steering override
via the existing steering feature.

## Invariants

The implementation must preserve these invariants; violations are
bugs.

1. **Bytes never ride the API response.** `RequestOutput.activation_storage`
   is a pure pointer. Any future change that inlines bytes is a
   scope violation.
2. **Capture reads the pristine residual.** `maybe_capture_residual`
   runs *before* the steering vector is added at the same hook
   point. Captures reflect the un-steered residual, independent of
   any active steering state.
3. **Cold path is free.** When `--activation-storing` is unset at
   server start, no capture code runs during the forward pass. The
   custom op is constant-folded by `torch.compile`.
4. **Per-step plan is batch-order consistent.** Absolute batch
   positions in `gather_indices` are in the same order the runner
   fed tokens into the model. The finalize walk preserves that
   order so the writer sees rows in a stable sequence.
5. **Multi-step captures preserve append order.** Writer task
   partitioning ensures all writes for a single `(request_id,
   layer, hook)` go through the same thread, preserving order
   across decode steps.
6. **Finalization is atomic.** A reader observing
   `{request_id}.bin` AND `{request_id}.json` is guaranteed to see
   consistent shape, dtype, and position metadata. The rename from
   `.tmp` happens only after both files are fully written.
7. **Partial failures never abort text generation.** A broken disk
   or NFS hiccup degrades to `capture_status = "partial_error"` and
   the request still returns its tokens to the client.
8. **Prefix-cache positions are rejected at admission.** Clients
   requesting logical positions that fall below
   `initial_num_computed_tokens` receive HTTP 400 immediately, not a
   silent empty file after the request finishes.
9. **TP/PP > 1 is rejected with a clear error.** No best-effort
   single-rank capture behind the operator's back.
10. **Model dtype is server-decided.** Clients cannot override the
    capture dtype in v1. Baking it into the path means the layout is
    unambiguous without reading sidecars.

## Performance notes

- **Cold path cost**: one `None`-check per hook call when the
  server is started without `--activation-storing`. Constant-folded
  under `torch.compile`, so zero runtime cost after the first
  forward pass.
- **Warm idle cost** (server enabled, no requests capturing): one
  dict lookup per hook call inside the custom op. Handful of
  nanoseconds. Doesn't move the forward-pass latency needle.
- **Capture cost per row**: one `index_select` into GPU scratch + a
  `non_blocking=True` D2H copy. For `hidden_size = 4096` at
  `bfloat16`, the transfer is 8 KB; PCIe 4.0 is 32 GB/s, so the
  copy itself is ~250 ns per row ignoring overhead. In practice
  batch sizes are enough that pool overhead dominates.
- **Disk bandwidth**: dominated by NFS write throughput. A typical
  10 GbE NAS caps around 1 GB/s aggregate. At that ceiling, a
  70B-model SAE training batch of 1k prompts × 80 layers × 4 KB
  per-token (last-token only) writes at ~320 MB in ~320 ms —
  well under the ~1–2 s per decode step the GPU takes. Full-sequence
  captures (`"all_prompt"` or `"all"`) can saturate the NFS; the
  writer queue will backpressure the engine rather than OOMing the
  worker.
- **Writer thread count**: 4 threads is enough for most NFS mounts;
  raise it if the per-write latency is stall-limited rather than
  bandwidth-limited (e.g., a slow-metadata NFS server).

## Files related to this feature

| File | Role |
|---|---|
| `vllm/config/activation_storing.py` | `ActivationStoringConfig` (server-global). |
| `vllm/config/activation_storing_types.py` | `ActivationStoringSpec`, selector types, expansion helpers. |
| `vllm/model_executor/layers/activation_capture.py` | `ActivationCaptureManager`, `StepCapturePlan`, custom op + fake, `maybe_capture_residual`. |
| `vllm/model_executor/layers/steering.py` | One-line edit to `apply_layer_steering` to call `maybe_capture_residual` before the steering op. |
| `vllm/v1/worker/activation_writer.py` | Thread pool, `WriteTask`, atomic rename, partitioned queues. |
| `vllm/v1/worker/gpu_model_runner.py` | `_prepare_activation_storing_step`, `_finalize_activation_storing_step`, finalize hook. |
| `vllm/entrypoints/openai/activation_storing_validation.py` | Admission-time validation (TP/PP, config enabled, layer range, prefix cache, byte cap). |
| `vllm/entrypoints/openai/chat_completion/protocol.py` | `activation_storing` request field, `ActivationStorageResponse`. |
| `vllm/entrypoints/openai/completion/protocol.py` | Same for legacy completions. |
| `vllm/entrypoints/openai/chat_completion/serving.py` | Thread `activation_storage` into the response. |
| `vllm/entrypoints/openai/completion/serving.py` | Same for legacy completions. |
| `vllm/sampling_params.py` | `SamplingParams.activation_storing`. |
| `vllm/v1/engine/__init__.py` | `EngineCoreOutput.capture_status`, `capture_paths`, `capture_error`. |
| `vllm/v1/outputs.py` | `ModelRunnerOutput.capture_status` propagation. |
| `vllm/v1/engine/output_processor.py` | Surface capture status on `RequestOutput`. |
| `vllm/outputs.py` | `RequestOutput.activation_storage`. |
| `vllm/engine/arg_utils.py` | CLI flags. |
| `vllm/config/vllm.py` | Wire `ActivationStoringConfig` into `VllmConfig`. |
| `docs/features/activation_storing.md` | This document. |
| `docs/OVERVIEW.md` | Feature index entry. |
| `tests/v1/capture/` | Unit + integration tests (plan builder, writer, full pipeline). |
| `tests/entrypoints/openai/test_activation_storing_protocol.py` | Request-model validation tests. |

## Scope limits (v1)

Rejected at request admission time with HTTP 400:

- Server started without `--activation-storing`.
- `tensor_parallel_size > 1`.
- `pipeline_parallel_size > 1`.
- `n > 1` or beam search.
- Disaggregated prefill (`kv_transfer_params` present).
- Layer indices outside `[0, num_hidden_layers)`.
- Positions below `initial_num_computed_tokens` (prefix-cache hits).
- Byte estimate exceeding `--activation-storing-max-bytes-per-request`
  when the flag is non-zero.

Streaming requests are **supported** — text streams over SSE while
bytes land on disk via the writer pool in parallel.

## Future work

Out of scope for v1 but worth noting for the design record:

- **TP/PP > 1 support.** Requires a cross-rank residual rendezvous
  protocol and per-hook shape reconciliation. Achievable but
  non-trivial.
- **Decode-position captures with speculative decoding.** Current
  v1 rejects decode positions when spec decode is active. A real
  fix requires filtering on sampler accept/reject output and
  dropping speculated-but-rejected rows before they hit disk.
- **Compression / columnar repacking.** A post-processing tool
  that consolidates `{tag}/{layer}/{hook}/*.bin` into a single
  safetensors or Zarr shard would save inodes and improve
  downstream load times. Deliberately left as a separate tool so
  the hot path stays simple.
- **Tag-level index file.** Appending one line per finalized request
  to `{tag}/_index.jsonl` would turn "list all requests in this
  tag" into an O(1) read. We skipped this in v1 to avoid
  cross-request write contention, but it's a clean additive
  feature.
- **Authenticated / encrypted writes.** If the target is a
  multi-tenant NAS, the vLLM process writes under its own POSIX
  identity. Per-tenant isolation would need a different transport
  entirely.
- **Live capture streaming during generation.** Out of scope
  because the design commits to finalization-on-request-complete.
  A separate feature could write complete sidecar entries per
  decode step to enable live consumers.

## FAQ

**Why a separate flag instead of reusing steering config?**
Steering writes, capture reads. Different lifetimes (capture persists
to disk; steering is ephemeral), different ownership (capture owns
the writer thread pool), different failure modes. Sharing config
would couple two features that have nothing in common except the
three hook-point positions.

**Can I capture without steering enabled?**
Yes. `--activation-storing` works independently of
`--enable-steering`. The only shared code is the `apply_layer_steering`
wrapper function, which exists on every steering-instrumented model
regardless of whether steering is active.

**What's the difference between `request_id` and vLLM's internal
request id?**
`request_id` in the `activation_storing` spec is client-chosen and
becomes the filename stem. vLLM's internal request id
(`cmpl-abc123...`) is process-unique and recorded in the sidecar as
`vllm_internal_request_id` for traceability. Clients that want their
captures to be findable by external identifier should use the
`activation_storing.request_id` field.

**What happens if I set the same `request_id` twice under the same
`tag`?**
Depends on `--activation-storing-on-collision`. Default `overwrite`
truncates and rewrites. `error` fails the second request with HTTP
400. `suffix` appends `.{unix_ms}` — but note this breaks
deterministic paths, so clients must parse the response to find the
actual filename.

**Do I need to run with `--enable-steering` to get activation
storing?**
No, but note that the capture hook is implemented by piggybacking on
the steering hook helper `apply_layer_steering`. As long as the
model you're serving is on the list of steering-instrumented models
(~65 decoder models at time of writing), activation storing works
without steering being enabled.

**Can I tee captures to multiple tags?**
Not in a single request. Run the prompt twice with different tags,
or run a post-processing tool that copies files between tag
directories. v1 does not duplicate captures inside a single request.

**Can I capture intermediate tensors other than the residual
stream?**
No. v1 captures exactly the residual at `pre_attn`, `post_attn`, and
`post_mlp`. Attention patterns, MLP activations, and pre-norm
tensors are out of scope because they'd require new model-file
instrumentation on top of the steering hook points.

**Is the feature safe on a shared NAS with other writers?**
Yes, under these assumptions: (1) no other process writes files with
names matching `{request_id}.bin` or `{request_id}.json` in the
directories vLLM is writing to, and (2) the POSIX identity the vLLM
process runs under has write permission. vLLM never reads or
modifies files outside its own writes.
