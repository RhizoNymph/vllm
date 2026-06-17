# Dynamic Steering Controller (sync execution)

A sync-execution capture consumer that closes the activation→steering
feedback loop with **exactly-one-step latency**: probe the residual
stream at one `(layer, hook)` site, and steer — per request or globally
— when the probe fires.

This is the Phase 1a controller from
[`docs/design/dynamic_steering.md`](../../../docs/design/dynamic_steering.md).

## How it works

```
forward step N
  └─ residual at (monitor_layer, monitor_hook) copied into a persistent
     buffer inside the CUDA graph (global capture spec — never forces eager)
step N, post-forward (model-runner step thread, every TP rank)
  └─ on_step(view): probe GEMV on the GPU buffer → one tiny D2H of
     per-token scores → per-request EMA + hysteresis policy
       └─ returns RequestSteeringOverride / SteeringVectorUpdate actions,
          applied inline through the engine's validation path
step N+1
  └─ _update_steering_buffers routes the firing request's decode tokens
     to a dynamic-pool steering row (global + override vectors)
```

The capture hook reads the residual *before* steering is applied at
that site, so the monitor never measures its own intervention.

## Actuation modes

- **`per_request`** (default): each request runs its own engagement
  state machine; only firing requests are steered, via
  `RequestSteeringOverride` → a dynamic-pool table row. Admission
  state, scheduler accounting, and prefix caching are untouched
  (requires `--enable-steering` and a non-zero
  `--max-dynamic-steering-configs`, default 4).
- **`global`**: scores aggregate (max/mean) into one engagement state
  driving the global decode tier — steers every request.

## Scope

- **TP > 1 supported**: sync consumers are constructed on every TP rank
  and compute identical decisions from the replicated residual (no
  communication). `on_step` must stay a pure deterministic function of
  the view + policy state.
- **PP must be 1** (enforced at registration).
- Decode-only by construction: per-request overrides are rejected for
  prefilling requests; prefill scores still prime the EMA so engagement
  can fire on the first decode step.
- Known limitation: steering-aware APC block hashes for streaming
  continuation reflect *admitted* steering only, never dynamic
  overrides (`docs/design/dynamic_steering.md` §5.2).

## Usage

```bash
vllm serve google/gemma-3-4b-it \
  --enable-steering --max-dynamic-steering-configs 4 \
  --capture-consumers dynamic_steering:monitor_layer=15,probe_path=/data/probe.pt,steering_vector_path=/data/steer.pt,threshold=0.35,hysteresis=0.1,gain=4.0
```

(Install this package first — `uv pip install -e .` from this directory.
Richer params need the Python API's dict form; the CLI shorthand only
takes flat scalars.)

```python
from vllm import LLM

llm = LLM(
    model="google/gemma-3-4b-it",
    enable_steering=True,
    max_dynamic_steering_configs=4,
    capture_consumers=[{
        "name": "dynamic_steering",
        "params": {
            "monitor_layer": 15,
            "probe_path": "/data/probe.pt",
            "steering_packed_path": "/data/steer_bank.json",
            "actuation": "per_request",
            "threshold": 0.35,
            "hysteresis": 0.10,
            "gain": 4.0,
            "gain_mode": "proportional",
        },
    }],
)
```

## Parameters

| Param | Type | Default | Meaning |
| --- | --- | --- | --- |
| `monitor_layer` | int | required | Layer whose residual is probed. |
| `monitor_hook` | str | `post_mlp` | Hook point to probe (any capture hook). |
| `probe_path` | str | — | `torch.save`'d 1-D tensor; unit-normalized at load. |
| `probe_packed_path` | str | — | Packed-JSON (`SteeringHookPacked`) file; the `monitor_hook` entry must carry a row for `monitor_layer`. One of the two probe params is required. |
| `steering_vector_path` | str | — | `torch.save`'d 1-D tensor applied (scaled by gain) at every `steer_layers` entry. |
| `steering_packed_path` | str | — | Packed-JSON file; its `steer_hook` entry defines per-layer steering vectors (`layer_indices` = steer layers). One of the two steering params is required. |
| `steer_layers` | list[int] | `[monitor_layer]` | Target layers (`.pt` path only). |
| `steer_hook` | str | `monitor_hook` | Must be `pre_attn`/`post_attn`/`post_mlp`. |
| `actuation` | str | `per_request` | `per_request` or `global`. |
| `score` | str | `cosine` | `cosine` (scale-invariant) or `dot` (raw projection onto the unit probe). |
| `threshold` | float | required | Engage level on the EMA-smoothed score. |
| `hysteresis` | float | `0.0` | Disengage at `threshold - hysteresis`. |
| `ema_alpha` | float | `0.25` | Per-request EMA smoothing. |
| `gain` | float | `1.0` | Maximum steering scale. |
| `gain_mode` | str | `binary` | `binary` or `proportional`. |
| `aggregate` | str | `max` | Cross-request aggregation (global mode only). |
| `min_emit_delta` | float | `0.05` | Suppress emissions with smaller gain moves (flips always emit). |
| `forget_after_steps` | int | `16` | Prune policy state for requests absent this many steps. |
| `sync_budget_ms` | float | `5.0` | Soft per-step budget for the engine's over-budget warning. |

## Observability

`GET /v1/steering/dynamic` (dev mode) returns, per worker: the action
queue counters, per-source apply/reject stats, dynamic-pool occupancy,
and — per sync consumer — step timing, a ring of recent
`(step, on_step_ms, n_actions)` tuples, and this controller's
`status()` snapshot (per-request EMA / engaged / last emitted gain).
Per-worker dicts are returned unaggregated, so comparing TP ranks'
rings doubles as a rank-divergence audit.

## Validating a probe (shadow mode)

Run with `gain=0.0`: decisions are computed and fully visible in
`GET /v1/steering/dynamic` (engagement flips still emit, but the
emitted vectors are zero), so you can watch the engagement trace on
real traffic before letting it steer. Raise the gain when the trace
looks right.

## Minimal examples — one per configuration

`DynamicSteeringController` is the kitchen-sink, probe-gated policy. For
clear copy-paste templates of each individual way the runtime can be
driven, see `minimal_examples.py` — small, deterministic consumers that
each emit exactly one kind of action (or use one transport):

| entry point | transport | action |
|---|---|---|
| `steering_ex_override` | sync | `RequestSteeringOverride` (per-request) |
| `steering_ex_global_tier` | sync | `SteeringVectorUpdate` (global decode tier) |
| `steering_ex_tier_scale` | sync | `SteeringScaleUpdate(tier_gain=)` (cheap strength knob) |
| `steering_ex_per_request_scale` | sync | `SteeringScaleUpdate(req_id=)` (per-request strength, resolved req_id→dyn_id) |
| `steering_ex_monitor_tier` | sync | `SteeringMonitorUpdate` (in-graph per-token gate on the tier) |
| `steering_ex_monitor_rowgate` | sync | `SteeringMonitorUpdate(gate_rows=True)` (in-graph per-token gate on per-request rows) |
| `steering_ex_async_tier` | async | `SteeringVectorUpdate` via the action queue (`on_capture`, 1–3 step latency) |

Other reachable targets via the same actions: `SteeringScaleUpdate()` with
no field set scales the global decode row; `config_hash=` scales a static
admitted config's decode row; the base/prefill escape hatch
(`SteeringVectorUpdate(phase="base"/"prefill")`) is cache-unsafe and
requires the caller to own invalidation.

## Tests

`python test.py` (or `pytest test.py`) — pure CPU, no engine needed.
`pytest test_minimal_examples.py` covers the minimal examples.
