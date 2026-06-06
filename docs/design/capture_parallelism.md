# Capture Consumers under Tensor / Pipeline / Expert / Data Parallelism

Status: **Phases 0–2 implemented.** Worker-location capture consumers
(including the built-in `filesystem` consumer) support TP / PP / EP / DP
for the replicated residual hooks: the filesystem validator no longer
rejects multi-rank requests, TP rank 0 of each pipeline stage captures
that stage's global-indexed layers to a shared mount, and the engine
unions the per-stage results. Phase 3 (a single driver-side consumer
seeing the full layer stack across PP stages) and Phase 4 (sharded
activations) remain future work — see the [Phased Plan](#phased-plan).
This document establishes the technical ground truth, the design space,
and that plan.

For the single-rank design see
[Capture Consumers Design](capture_consumers.md); for the user-facing
guide see [Capture Consumers](../features/capture_consumers.md).

## TL;DR

- **The capturable hooks are replicated.** The three hooks that fire
  today — `pre_attn`, `post_attn`, `post_mlp` — read the residual
  stream *after* the TP all-reduce and the MoE combine, so the tensor
  is full `hidden_size`, **byte-identical on every TP and every EP
  rank**. For these hooks, TP/EP support is a *rank gate*, not a
  gather: exactly one rank per replication group captures; the rest
  no-op.
- **Pipeline parallelism is the substantive work.** Layers are
  partitioned across PP stages by *global* index; each stage's manager
  is built with the *local* layer count and currently rejects its own
  layers. PP needs a global layer space, per-stage spec filtering, and
  cross-rank result aggregation.
- **Data parallelism is essentially free.** Each DP rank is an
  independent engine core handling disjoint requests; captures
  partition by request with no aggregation.
- **Sharded activations (intermediate MLP / per-expert outputs) are a
  separate, larger lift** requiring either an on-device gather in the
  forward or a capture-slices-reassemble-at-read layout. Not needed for
  the replicated residual hooks.
- **vLLM already has the result-aggregation template**:
  `KVOutputAggregator.aggregate()` collects *all* workers'
  `ModelRunnerOutput`s and merges auxiliary fields into the single
  `output_rank` output. Capture results ride the same path.

## Ground Truth

### Residual stream is replicated under TP and EP

The fired hooks tap `hidden_states` after the residual add, which is
downstream of the reducing collectives:

- `RowParallelLinear` all-reduces when `reduce_results=True`
  (`vllm/model_executor/layers/linear.py:1558-1559`), so attention- and
  MLP-output projections produce full `hidden_size` on every TP rank.
- MoE paths all-gather/all-reduce before the residual add
  (`vllm/model_executor/models/deepseek_v2.py:384`), so `post_mlp` on an
  EP rank also sees the full residual.

Hence `pre_attn` / `post_attn` / `post_mlp` are `[num_rows,
hidden_size]` and identical across the TP×EP plane of a PP stage.

What is **genuinely sharded** (and not captured today):

- The MLP **intermediate** (`gate_up_proj` output) is sharded along
  `intermediate_size / tp` (`ColumnParallelLinear`). Note `mlp_in` (the
  input to `gate_up_proj`) is the *replicated* residual; only the
  intermediate between `gate_up` and `down` is sharded.
- Per-expert MoE outputs are sharded across EP ranks before the
  combine.

Capturing either requires a gather (see [Sharded
activations](#sharded-activations)).

### Pipeline parallelism partitions layers by global index

- `get_pp_indices(total_layers, pp_rank, pp_size)` returns `[start,
  end)` for each stage (`vllm/distributed/utils.py`). Each stage
  instantiates only its own layers (others are `PPMissingLayer`); the
  layers keep their **global** indices and fire hooks with them.
- But the manager is constructed with
  `model_config.get_num_layers(parallel_config)`, which returns
  `end - start` — the **local** count (`vllm/config/model.py:1329`,
  used at `vllm/v1/worker/gpu_model_runner.py:560`).
- Result: on stage 1 (global layers 32–63), `_num_hidden_layers == 32`,
  and `register_request` rejects every global index `>= 32`
  (`vllm/v1/capture/manager.py:360`). The registration path assumes one
  global `[0, N)` layer space in a single manager.

### One runner/manager per worker; one authoritative output rank

- There is one `GPUModelRunner` and one `CaptureManager` per worker
  process (per TP×PP rank), `set_active_capture_manager` called once
  (`gpu_model_runner.py:557-582`).
- The executor returns only `output_rank`'s `ModelRunnerOutput`:
  `output_rank = world_size - tp_size * pcp_size`, i.e. **TP rank 0 of
  the last PP stage** (`vllm/v1/executor/multiproc_executor.py:480-494`,
  filtered at `:948-970`). Other ranks' outputs are discarded by
  default.
- **Exception — the aggregation template:** when a `KVOutputAggregator`
  is set, *all* workers' outputs are passed to `aggregate()`, which
  merges per-rank auxiliary fields (`finished_sending`,
  `kv_connector_stats`, …) into `outputs[output_rank]` before returning
  (`vllm/distributed/kv_transfer/kv_connector/utils.py:65-173`).
  `ModelRunnerOutput.capture_results` (`vllm/v1/outputs.py:288`) can be
  merged the same way.

### Data parallelism is independent per rank

- Non-MoE DP ranks are "completely independent" — each runs with
  `data_parallel_size = 1` locally and processes a disjoint request
  queue (`vllm/v1/engine/core.py:1133-1138`). MoE DP ranks synchronize
  *finished* metadata but still own disjoint request batches.
- So captures partition by request across DP; no cross-DP aggregation
  is required. The only hazard is two DP ranks writing the same
  on-disk `request_id`, which cannot happen for distinct requests.

### Collective primitives available

On `GroupCoordinator` (`vllm/distributed/parallel_state.py`):

- Tensors: `all_reduce`, `all_gather(dim)`, `all_gatherv(sizes)`,
  `gather(dst, dim)`, `broadcast(src)`.
- Objects (pickled): `broadcast_object`, `send_object(dst)`,
  `recv_object(src)`, `*_object_list`.
- Mixed tensor+metadata: `send_tensor_dict` / `recv_tensor_dict` /
  `isend_tensor_dict` / `irecv_tensor_dict` — the same primitives PP
  uses to pass `IntermediateTensors`
  (`vllm/v1/worker/gpu_worker.py:826-867`).
- Wrappers: `tensor_model_parallel_all_gather` /
  `_all_reduce` / `_reduce_scatter` (`communication_op.py:12-24`).

Group accessors: `get_tp_group`, `get_pp_group`, `get_ep_group`,
`get_dp_group`, with `.rank_in_group`, `.world_size`, `.is_first_rank`,
`.is_last_rank`.

## Design Space

### Axis-by-axis

| Axis | Capturable (residual) hooks | Sharded hooks |
| --- | --- | --- |
| **DP** | independent per request; nothing to do | same |
| **TP** | rank gate: capture on `tp_rank == 0`, no-op others | all ranks `all_gather`, rank 0 keeps full, or write slices |
| **EP** | rank gate: capture on `ep_rank == 0`, no-op others | gather per-expert outputs across EP, or write slices |
| **PP** | global layer space + per-stage spec filtering + result merge | same plumbing; data still per-stage |

The **capturer rank** within a PP stage is the rank that is rank 0
across every intra-stage replication dim — in practice `tp_rank == 0
and ep_rank == 0`. On non-capturer ranks the manager is simply not
installed (`set_active_capture_manager(None)`), so the cold path holds
and the compiled graph contains no capture op (invariant 2). This is
safe under TP because the capture op is graph-local (an `index_select`)
and introduces **no collective**, so ranks may legitimately differ on
whether the op is present. (This stops being true for the on-device
sharded-gather path — see below.)

### The aggregation fork

Where a request's captures across PP stages get combined splits into
two architectures. They are **not mutually exclusive** — the consumer's
declared `location` can select between them.

#### Option A — independent per-rank writes to shared storage (no gather)

- Each capturer rank registers only the spec layers in its **local**
  `[start, end)` range, captures them, and the (worker-location)
  filesystem consumer writes to a path keyed by **global** layer index
  + request_id on a **shared** mount.
- The on-disk layout merges naturally: stage 0 writes
  `…/req/12_post_mlp.bin`, stage 1 writes `…/req/40_post_mlp.bin`, no
  collision. The `packed`/`sharded` layouts (one file per request / per
  tag) cannot merge by global layer index alone, so under PP each stage
  writes its **own** file keyed by stage rank
  (`packed-pp{RR}.{bin,json}`, `shard-pp{RR}-{NNN}-{SEQ}.{bin,json}`) and
  the reference reader merges the per-stage files on read. **Implemented.**
- The only thing the engine merges is the per-rank `CaptureResult`
  *status/payload* dicts, via the `KVOutputAggregator` pattern (union
  the written-path lists, status = worst-of).
- **Pros:** no tensor crosses a rank boundary; fits the existing
  worker-side streaming consumer and its NFS-tuned `packed`/`sharded`
  layouts; scales to multi-node PP. **Cons:** requires shared storage
  reachable by every PP node; cannot feed a single in-process consumer
  the whole residual stack live.

#### Option B — gather to one rank, single consumer instance

- Within each stage, gather residuals to stage-local rank 0; then ship
  each stage's layers to the global `output_rank` (TP0 of last PP) over
  the PP object channel (`send_object` / piggyback on the PP
  `tensor_dict`), which assembles the full per-request capture and
  hands it to **one** consumer.
- **Required** for `location="driver"` consumers (e.g. a live training
  loop that wants the full layer stack in one process) and for
  consumers that cannot share storage. **Cons:** cross-PP transport of
  potentially large tensors, ordering, memory pressure, and added
  synchronization on the critical path.

**Recommended shape:** let the consumer's `location` drive the choice.
`location="worker"` → Option A (default, cheap). `location="driver"` →
the framework performs the Option-B gather to the driver-bridge rank.
This generalizes the existing `_DriverQueueShim` to a *PP-aware*
gather instead of a single-rank worker→driver hop.

### Sharded activations

For genuinely sharded hooks (MLP intermediate, per-expert outputs),
two layouts mirror the aggregation fork:

- **S-A — gather-on-device:** call `tensor_model_parallel_all_gather`
  (or an EP all-gatherv) inside the hook before `index_select`, so the
  capturer rank sees the full tensor. **All** ranks in the shard group
  must participate in the collective — so this path cannot be
  rank-0-only, and the collective enters the forward. Captures already
  force eager on capturing steps (see memory `capture-firing-cudagraph`),
  so a collective there is tolerable but is a real sync and extra
  bandwidth. Simple to read (one full tensor).
- **S-B — capture slices, reassemble at read:** every shard rank writes
  its own slice plus metadata (`shard_dim`, `shard_index`,
  `shard_count`, and for MoE the expert ids). No forward-time
  collective; the reader concatenates slices across ranks. Cheap at
  capture, more complex on-disk schema and reader, and needs all shard
  ranks (not just rank 0) to write.

Per-expert capture additionally needs **new hook points inside
`FusedMoE`** (before the combine) — the current hook set has none, so
sharded-expert capture is the largest single piece of new model-side
wiring.

## Phased Plan

Phases are ordered by value/effort and separated by area of concern.
**Status:** Phases 0–2 are **implemented** (worker-location consumers,
incl. the built-in `filesystem` consumer, now support TP/PP/EP/DP for
the replicated residual hooks). Phase 3 is **deferred** (see below).
Phase 4 remains future work.

### Phase 0 — global layer space + capturer-rank gate (foundation) ✅

- The manager takes the **global** layer count plus the local
  `[start, end)` range (`CaptureManager.local_layer_range`); registration
  validates hook layers against the global count and **filters** each
  consumer's hooks to the local range
  (`manager._filter_specs_to_layer_range`).
- **Capturer-rank gate**: the runner installs a manager only on
  `get_tp_group().rank_in_group == 0` and calls
  `set_active_capture_manager(None)` on every other rank. The gate is TP
  rank 0 (not additionally `ep_rank == 0`): the residual is replicated
  across the TP group *within each (data-parallel, pipeline) cell*, and
  the EP plane can span TP×DP — gating on `ep_rank == 0` would wrongly
  silence whole DP replicas, which each handle distinct requests. TP
  rank 0 per cell is exactly one capturer per stage per replica.
- Admission (`CaptureContext`) carries the **global** layer count and EP
  / DP sizes (`capture_expert_parallel_size`); the worker-side context
  now uses `get_total_num_hidden_layers()`.
- Unblocks **DP** outright (independent engine cores) and is the
  substrate for TP/EP/PP.

### Phase 1 — TP + EP residual capture (Option A) ✅

- The filesystem validator no longer rejects TP/PP/EP/DP > 1 for the
  replicated residual hooks (`validation.py`).
- Rank-gating means exactly one copy of each `(layer, hook)` reaches
  disk under TP>1 / EP>1.

### Phase 2 — PP residual capture (Option A) + result aggregation ✅

- Per-stage spec filtering (Phase 0) means each pipeline stage's TP
  rank 0 writes its own global-layer files to the shared mount.
- Cross-rank result merge: `_CaptureAwareAggregator`
  (`executor/abstract.py`) wraps the optional `KVOutputAggregator` and
  unions every rank's `ModelRunnerOutput.capture_results` per
  `(request, consumer)` into `output_rank`'s output
  (`manager.merge_capture_results`), reusing the all-outputs reply path.
  It lives on the base `Executor` (via `_output_aggregator()`), so it
  works identically on **single-node multiproc and multi-node Ray**
  (`MultiprocExecutor`, `RayExecutorV2`, and `RayDistributedExecutor`).
  The SPMD `ExecutorWithExternalLauncher` (torchrun) is the one backend
  not wired: it returns a single worker's output with no central
  collection point, so under it the files still land but the
  `capture_results` dict reflects only `output_rank` — use Ray for
  multi-node capture.
- **Shared-storage requirement:** capture targets must be a shared mount
  reachable by every PP node (resolved assumption below).

### Phase 3 — Option B gather for driver-side consumers (DEFERRED)

Deferred after a scoping discovery: `install_driver_consumer` runs
inside `build_consumers` → `GPUModelRunner.__init__`, i.e. **in each
worker process**, and (after Phase 0) on each PP stage's TP-rank-0
worker. The `_DriverReceiver` thread therefore runs in the worker, not a
single driver process — so under PP a `location="driver"` consumer would
get one independent receiver + pickled consumer copy per stage, each
seeing only its stage's keys. Delivering the full per-request layer
stack to **one** consumer requires routing every stage's shim to a
single driver-process receiver: a cross-process redesign that also
depends on closing the pre-existing "`LLM(capture_consumers=[instance])`
is not fully wired" limitation (see
[capture_consumers.md](capture_consumers.md#known-limitations)). Tracked
as future work; worker-location consumers (the common case) are
unaffected.

### Phase 4 — sharded activations (S-A and/or S-B) — future

- New `FusedMoE` / MLP-intermediate hook points.
- Sharded-capture layout + reader extensions (slice metadata) and/or
  on-device gather path. The filesystem validator has a marked insertion
  point where sharded-hook rejection will go.

## Open Questions

- **Sharded layout choice (S-A vs S-B):** trades forward-time
  collective bandwidth against on-disk/reader complexity. (Phase 4.)
- **Option A vs B default for driver consumers:** revisited when Phase 3
  is taken up — `location`-driven selection vs an explicit per-request
  flag.
- **Shared-storage assumption — RESOLVED:** capture targets are always
  a shared mount (e.g. an NFS volume) reachable by every PP node. No
  per-node→driver copy fallback is needed, so Option A is the default
  path for all replicated-hook phases (0–2).
- **`CaptureContext` plumbing — RESOLVED:** the context now carries the
  *global* layer count plus `expert_parallel_size` / `data_parallel_size`
  alongside the existing TP/PP sizes.
