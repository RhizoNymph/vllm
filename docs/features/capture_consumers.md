# Capture Consumers

Capture consumers are a pluggable system for observing and routing
hidden-state activations produced inside vLLM's forward pass. Consumers
receive captured tensors at request finalization and can do anything
they want with them — stream them to disk, feed them into a training
loop, ship them to a dashboard, or simply log that a capture occurred.

This page is the user-facing guide. For the internal design and
runtime mechanics, see
[Capture Consumers Design](../design/capture_consumers.md).

## What Capture Consumers Do

Each capture consumer is a plugin registered under the
`vllm.capture_consumers` Python entry-point group. Once enabled, the
engine routes activations from specific `(layer, hook)` points to the
consumer as requests are processed. Consumers can be triggered in two
ways:

- **Global capture**: the consumer declares a `CaptureSpec` that
  applies to every request. Used by observability probes, reward
  trainers, dashboards — anything that wants to see every request
  without clients opting in.
- **Per-request capture**: the client opts in by setting
  `SamplingParams.capture[consumer_name]`. Used by the built-in
  filesystem consumer so callers choose a tag, layers, and positions
  per request.

The two modes compose: a single request can trigger a global consumer
*and* a per-request consumer, and `RequestOutput.capture_results`
returns a per-consumer result dict.

## Built-in Consumers

vLLM ships two consumers, registered in its own `pyproject.toml` via
the same entry-point group third-party plugins use.

### `filesystem`

Streams captured activations to raw `.bin` files with sidecar JSON.
Implemented at `vllm.v1.capture.consumers.filesystem.FilesystemConsumer`.

- `reads_client_spec = True` — captures are always per-request. The
  filesystem consumer has no global spec.
- `location = "worker"` — runs in the engine-core subprocess, so it
  can stream bytes to disk without crossing a process boundary.
- Writes incrementally to `{path}.bin.tmp` as chunks arrive and does
  an atomic `os.replace` on finalize, so readers never see a partial
  file.

**Engine-side parameters** (set via `--capture-consumers` / YAML / the
Python API):

| Field | Type | Default | Purpose |
| --- | --- | --- | --- |
| `root` | `str` | required | Root directory for all captures. |
| `writer_threads` | `int` | `4` | Writer thread pool size. |
| `queue_size` | `int` | `1024` | Per-thread bounded queue capacity. |
| `timeout_seconds` | `float` | `180.0` | Per-write timeout; failures become `partial_error`. |
| `on_collision` | `"overwrite" \| "error" \| "suffix"` | `"overwrite"` | What to do when the target `.bin` already exists. |
| `fd_cache_size` | `int` | `256` | Per-thread LRU file-descriptor cache. |
| `fsync` | `bool` | `True` | `fsync` each file before publish. `False` trades crash-durability for throughput (near-no-op on NFS, where `close` already COMMITs). |
| `atomic_publish` | `bool` | `True` | Publish via `.tmp` + atomic rename. `False` writes straight to the final path (drops two rename RPCs/file, loses atomic visibility; requires `on_collision="overwrite"`). |
| `default_layout` | `"per_file" \| "packed" \| "sharded"` | `"per_file"` | Layout for requests that don't set their own `layout`. |
| `coalesce_max_bytes` | `int` | `1<<20` | Merge consecutive same-key queued writes into one `writev` up to this size (`0` disables). Most effective for `packed`/`sharded`. |
| `num_shards` | `int` | `8` | `sharded` layout: shard files per tag (request → `hash(id) % num_shards`). |
| `shard_max_bytes` | `int` | `256<<20` | `sharded` layout: size at which an open shard is sealed and a new one started. |

**Per-request client spec** (`FilesystemCaptureRequest`):

```python
@dataclass
class FilesystemCaptureRequest:
    request_id: str                      # filename stem, slugged
    tag: str                             # grouping label, slugged
    hooks: dict[str, Any]                # hook name -> layer selector
    positions: str | list[int]           # position selector
    layout: str | None = None     # "per_file" | "packed" | "sharded" (else default)
```

Client `hooks` values may be a list of ints, the literal string
`"all"`, or a dict `{"layers": [...], "ranges": [[a, b], ...]}`.
`positions` accepts `"last_prompt"`, `"all_prompt"`,
`"all_generated"`, `"all"`, or an explicit `list[int]`.

**Layout** (`layout`, else the consumer's `default_layout`):

- **`per_file`** (default) — one `.bin` + `.json` per
  `(layer, hook)`. Lowest latency; a reader can tail a `.bin` as
  decode steps append (mid-request streaming). File count scales with
  `layers × hooks` per request.
- **`packed`** — one `packed.bin` + one `packed.json` index per
  *request*, with all `(layer, hook)` tensors concatenated. Cuts the
  file count by `layers × hooks` — a large throughput win on network
  mounts (one set of metadata RPCs per request instead of per
  `(layer, hook)`; ~4.5× on NFS in benchmarks). A capture is readable
  once its request finalizes (no mid-request streaming). Choose this
  for offline/bulk analysis of multi-layer captures.
- **`sharded`** — many requests' captures share a small set of large
  shard files **per tag** (a request is assigned to shard
  `hash(request_id) % num_shards`). Shards are sealed (published) when
  they cross `shard_max_bytes` or at shutdown, then a new one starts.
  Fewest files for the **many-tiny-requests** case (e.g. thousands of
  `last_prompt` single-row captures), where even `packed` makes one
  tiny file per request. Tradeoff: a capture is readable only **after
  its shard seals** (end-of-run/bulk model), and a request's result is
  `ok` ("captured into shard, durable after seal") rather than a
  ready-to-read file.

**On-disk layout**:

```text
# per_file
{root}/{tag_slug}/{request_id_slug}/{layer_idx}_{hook_name}.bin
{root}/{tag_slug}/{request_id_slug}/{layer_idx}_{hook_name}.json

# packed
{root}/{tag_slug}/{request_id_slug}/packed.bin
{root}/{tag_slug}/{request_id_slug}/packed.json   # index over the .bin

# sharded (per tag; many requests share each shard)
{root}/{tag_slug}/shard-{NNN}-{SEQ}.bin
{root}/{tag_slug}/shard-{NNN}-{SEQ}.json   # index: per-chunk entries with request_id
```

`tag_slug` and `request_id_slug` are produced by the admission
validator — characters outside `[a-zA-Z0-9._-]` are replaced with
`_`, and `..` / leading `/` are rejected outright.

**Payload**: raw tensor bytes in the model's residual dtype. `bf16`
is stored as raw uint16 bytes; readers should round-trip through
`torch.uint16.view(torch.bfloat16)`.

**Sidecar JSON**: written atomically alongside the `.bin` on finalize.
`per_file` sidecars carry `request_id`, `layer`, `hook`, `shape`,
`dtype`, plus framework-propagated fields. `packed` sidecars carry
`request_id`, `layout: "packed"`, `dtype`, and an `entries` list of
`{layer, hook, offset, nbytes, shape}` indexing the `packed.bin`.
`sharded` shard indexes carry `layout: "sharded"`, `shard_idx`, `seq`,
`dtype`, and per-chunk `entries` that additionally include `request_id`
(shards interleave many requests).

**Reading**: `vllm.v1.capture.consumers.filesystem.reader` (NumPy-only)
provides `read_per_file`, `read_packed`, `read_request` (auto-detects
per_file/packed), and `read_sharded(tag_dir)` → `{request_id:
{(layer, hook): array}}` (scans a tag's sealed shard indexes).

#### Throughput tuning

**First, find your bottleneck** — `dd if=/dev/zero of=<target>/probe bs=1M
count=1024 oflag=direct conv=fsync` measures your capture target's raw write
ceiling. The right lever depends on whether you're disk-bound or code-bound:

- **Disk-bound** (slow/network storage — e.g. NFS over a single connection,
  spinning disk, a SATA SSD): the capture *code* can't push past the disk.
  Reach for storage-side levers: faster storage; `nconnect=N` on the NFS
  mount to use parallel bandwidth; the **`packed`** layout (one file per
  request, far fewer metadata round-trips than `per_file`); or **`sharded`**
  for the many-small-requests case (collapses commit count to ~`num_shards`).
  `coalesce_max_bytes` (default 1 MiB) merges per-step appends; larger rarely
  helps. `fsync`/`atomic_publish` toggles are near-no-ops on a sync NFS export.
- **Code-bound** (fast local NVMe / tmpfs, where the disk isn't the wall):
  the single dispatch/submit thread is the limit. Use `packed` (one
  `WriteTask` + one lock per request per step via the batched submit path)
  and a handful of `writer_threads`. On fast storage this reaches multiple
  hundreds of MB/s to ~1 GB/s; on a network mount it makes no difference,
  because the disk caps you first.

For online capture during serving, none of this is on the critical path —
residual-stream volume at token-generation rate is far below these ceilings.

#### Backpressure & overload

When capture volume *does* outrun consumer throughput (heavy capture — many
layers × hooks × all positions, or a prefill burst), the **dispatch queue**
is the single backpressure point. It is bounded by
`--capture-dispatch-queue-size` (default 256; `<=0` is unbounded/legacy,
where overload grows memory without limit). When it fills,
`--capture-overload-policy` decides what happens:

| policy | behaviour | trade-off |
|---|---|---|
| `block` | stall the forward pass until the queue drains | no loss, bounded memory, serving slows |
| `drop` | discard the step's captures (counted via `dropped_packets`) | serving never stalls; lossy |
| `spill` *(default)* | serialize overflow to a local scratch dir and replay it, in order, when the queue drains | no loss, no stall, bounded RAM; uses local disk |

`spill` parks overflow under `--capture-spill-dir` (default
`$TMPDIR/vllm-capture-spill`; use fast local storage). It preserves strict
per-key ordering — once spilling starts, every packet routes through the
spill FIFO until it drains, so replayed data never races live data. The
dispatch thread replays spilled packets when the in-memory queue is idle, and
`finalize` waits for spilled data to reach consumers (no loss). If the spill
area hits `--capture-spill-max-bytes` (default 4 GiB), `spill` degrades to
`block` rather than dropping. The filesystem writer's own submit timeout is
`timeout_seconds` (default 30 s), after which a wedged writer surfaces a
`WriteError` instead of stalling indefinitely.

### `logging`

Minimal observation consumer. Logs one line per finalized capture:
`"capture key=... rows=N dtype=..."`. Discards the actual tensor.
Implemented at `vllm.v1.capture.consumers.logging.LoggingConsumer`.

- `reads_client_spec = False` — activated by its global spec.
- `location = "worker"`.

**Parameters**:

| Field | Type | Default | Purpose |
| --- | --- | --- | --- |
| `hooks` | `dict[str, list[int]]` | required | Hook name to layer indices. |
| `positions` | position selector | `"last_prompt"` | Which positions to capture. |
| `level` | `str` | `"INFO"` | Python logging level. |

## Enabling Consumers

A consumer is enabled when its name is referenced in any of the
config surfaces below. Names are the entry-point names from
`pyproject.toml` (e.g. `filesystem`, `logging`, or whatever a
third-party plugin registers).

### CLI (`vllm serve`)

`--capture-consumers` takes the shorthand `name:key=value,key=value`
and can be repeated to register multiple consumers:

```bash
vllm serve meta-llama/Llama-3-8B \
    --capture-consumers filesystem:root=/mnt/nas/activations \
    --capture-consumers logging
```

The shorthand only accepts flat scalar values — no nested dicts or
lists in values. For richer configuration, use a YAML config file.

### YAML config

`--config path/to.yaml` maps keys onto the same `EngineArgs` fields
as the CLI, so `capture_consumers` in YAML is a list of shorthand
strings — one per consumer:

```yaml
model: meta-llama/Llama-3-8B
capture_consumers:
  - filesystem:root=/mnt/nas/activations,writer_threads=4
  - logging
```

The shorthand is the same `name:key=value,key=value` form the CLI
accepts, with the same flat-scalar limitation. For richer parameters
(nested dicts, multi-layer hook maps) use the Python API below or
build a `CaptureConsumersConfig` and pass it via
`EngineArgs.capture_consumers_config_override`.

To run multiple instances of the same consumer type, disambiguate
them with `instance_name` via the Python API:

```python
llm = LLM(
    model="meta-llama/Llama-3-8B",
    capture_consumers=[
        {"name": "filesystem", "instance_name": "primary",
         "params": {"root": "/mnt/nas/primary"}},
        {"name": "filesystem", "instance_name": "mirror",
         "params": {"root": "/mnt/nas/mirror"}},
    ],
)
```

`RequestOutput.capture_results` is keyed by `instance_name` when
present, otherwise by the entry-point `name`.

### Python `LLM(...)`

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-8B",
    capture_consumers=[
        {"name": "filesystem", "params": {"root": "/tmp/captures"}},
        {"name": "logging", "params": {"hooks": {"post_mlp": [0]}}},
    ],
)
```

Dict entries become `CaptureConsumerSpec`s on `VllmConfig` and flow
through the engine end-to-end.

The list also accepts pre-constructed `CaptureConsumer` instances
(e.g. a driver-side consumer that needs a live Python model or
optimizer); these must have `location = "driver"`. The LLM
constructor validates and stashes such instances, but the plumbing
between `LLM` and `VllmConfig` for pre-constructed instances is
currently incomplete — see
[Capture Consumers Design — Known Limitations](../design/capture_consumers.md#known-limitations).
Use the dict form unless you are working on that plumbing.

### Per-Request Capture

For consumers that set `reads_client_spec = True` (the filesystem
consumer, and any third-party consumer that opts in), clients drive
the capture by attaching a dict to `SamplingParams.capture`:

```python
from vllm import SamplingParams
from vllm.v1.capture.consumers.filesystem import FilesystemCaptureRequest

sampling_params = SamplingParams(
    max_tokens=16,
    capture={
        "filesystem": FilesystemCaptureRequest(
            request_id="probe_0001",
            tag="mnist-probe-v1",
            hooks={"post_mlp": [12]},
            positions="last_prompt",
        ),
    },
)
```

The key is the consumer's entry-point name (or its `instance_name`
if configured). The value is whatever the consumer accepts — the
consumer's own `validate_client_spec` parses it. For the filesystem
consumer, passing a dict with the same fields also works.

Requests that omit `capture` only receive captures from consumers
with global specs.

### OpenAI-compatible API

Send the per-request spec in the `extra_body.capture` field:

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3-8B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "extra_body": {
            "capture": {
                "filesystem": {
                    "request_id": "probe_train_0001",
                    "tag": "capital-probe",
                    "hooks": {"post_mlp": [12, 16, 20, 24]},
                    "positions": "last_prompt",
                    "layout": "packed",
                },
            },
        },
    },
    timeout=60,
).json()
```

`layout` is optional (`"per_file"` default, `"packed"` for one indexed
file per request — see the `filesystem` consumer section). Validation
happens at admission time; an invalid spec (unknown consumer, bad
layout, out-of-range layer, …) returns HTTP 400 with a descriptive
error.

## Reading Results

On request completion, `RequestOutput.capture_results` is a
`dict[str, CaptureResult]` keyed by consumer instance name:

```python
from vllm import LLM, SamplingParams
from vllm.v1.capture.consumers.filesystem import FilesystemCaptureRequest

llm = LLM(
    model="meta-llama/Llama-3-8B",
    capture_consumers=[{"name": "filesystem", "params": {"root": "/tmp"}}],
)

sampling_params = SamplingParams(
    max_tokens=16,
    capture={
        "filesystem": FilesystemCaptureRequest(
            request_id="req1",
            tag="demo",
            hooks={"post_mlp": [0]},
            positions="last_prompt",
        ),
    },
)

[output] = llm.generate(["Hello"], sampling_params)
result = output.capture_results.get("filesystem")
if result is not None and result.status == "ok":
    for path in result.payload:
        print("wrote", path)
```

`CaptureResult` fields:

- `status`: `"pending"`, `"ok"`, `"partial_error"`, `"error"`, or
  `"not_requested"`.
- `error`: a human-readable message when `status != "ok"`.
- `payload`: consumer-specific. Filesystem returns a `list[str]` of
  written paths; other consumers return whatever they like.

On the OpenAI-compatible HTTP path, results are attached to the
response body as `capture_results`, mirroring the structure above.

## Parallelism

Capturing the residual-stream hooks (`pre_attn`, `post_attn`,
`post_mlp`) is supported under **tensor, pipeline, expert, and data
parallelism** for worker-location consumers — including the built-in
`filesystem` consumer. How it works:

- The residual stream these hooks read is **replicated** across the
  tensor- and expert-parallel ranks within each pipeline stage (it is
  read after the TP all-reduce / MoE combine), so exactly one rank — TP
  rank 0 of each stage — captures it; the other ranks add no overhead.
- Under **pipeline parallelism**, each stage's TP rank 0 captures the
  (global-indexed) layers that stage owns and writes them to the capture
  target; the engine merges the per-stage results into one
  `RequestOutput.capture_results`. Layer indices in a client spec are
  always **global** (`0..num_hidden_layers-1`).
- Under **data parallelism**, each replica is an independent engine core
  over disjoint requests; captures partition naturally with no merge.

**Requirement — shared storage:** under pipeline parallelism the capture
target (`filesystem` `root`) must be a **shared mount** (e.g. an NFS
volume) reachable by every pipeline node, because different stages write
different layers of the same request. Files are keyed by global layer
index, so stages never collide.

**Not yet supported:** (1) `location="driver"` consumers that need a
request's *full* layer stack assembled in one process across pipeline
stages (worker-location consumers are unaffected); (2) capturing
genuinely *sharded* activations (MLP intermediate / per-expert outputs).
See [Capture Consumers under Parallelism](../design/capture_parallelism.md).

## Limits

- **Prefix-cache hits**: positions below
  `CaptureContext.num_computed_tokens` were served from the prefix
  cache and never forwarded through the model. Consumers reject such
  positions at admission; enforcement is the consumer's
  responsibility.
- **Hook coverage**: only the decoder architectures that wire the
  `apply_layer_steering` / `maybe_capture_residual` pair fire hooks.
  See [Activation Steering](steering.md) for the list of covered
  architectures — capture coverage matches the steering list.
- **Capture failures don't abort generation**: consumer errors
  surface as `partial_error` / `error` on the corresponding
  `CaptureResult`; text generation always completes.

## Writing a Consumer Plugin

Third-party consumers ship as separate Python packages. See
[Plugin Authoring Guide](../capture_consumers/plugin_authoring.md)
for the worked examples (quick-start consumer, driver-side training
loop, streaming consumer, tests).

Example plugins live under `examples/capture_consumers/`:

- `minimal_plugin/` — the simplest `CaptureConsumer` subclass; records
  the sum of every captured tensor.
- `activation_reward_producer/` — a direct `CaptureSink` that returns
  a cosine-alignment reward plus diagnostic fields on
  `CaptureResult.payload`. Designed for RL loops; the README covers
  vector drift, detection via the diagnostic payload, and the
  frozen-scorer deployment pattern.
