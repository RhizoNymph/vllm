# Dynamic Steering Controller (Phase 0 prototype)

A capture consumer that closes the activation→steering feedback loop in
a single vLLM worker: observe the residual stream at one `(layer, hook)`
site, project each generated token onto a probe direction, and modulate
the **global decode** steering vector when the probe fires.

This is the Phase 0 prototype from
[`docs/design/dynamic_steering.md`](../../../docs/design/dynamic_steering.md) —
built to validate probes and policies on real traffic with minimal core
surface, not to be the final architecture. Its feedback latency is one
to a few decode steps (chunk delivery on the capture dispatch thread →
action queue → next step's table rebuild).

## How it works

```
forward step N
  └─ residual at (monitor_layer, monitor_hook) captured via a *global*
     capture spec — graph-safe persistent-buffer path, no eager forcing
       └─ dispatch thread: controller receives per-step chunks
            scores = activations · probe   (cosine or dot)
            per-request EMA → aggregate (max/mean) → hysteresis gate
            gain changed?  →  SteeringActionQueue.submit(...)
step N+1 (model-runner step thread, _update_steering_buffers)
  └─ queue drained → SteeringManager.update_global_vectors(phase="decode")
       └─ tables repopulated; decode tokens now steered by vec * gain
```

The capture hook reads the residual *before* steering is added at that
hook point, so the monitor always sees the model's un-steered state at
the monitored site — the loop does not measure its own intervention
(self-feedback through downstream layers on later tokens is inherent
and intended).

## Scope and limitations (deliberate, Phase 0)

- **`tp=1, pp=1` only.** Capture consumers are constructed on TP rank 0
  only; pushing steering updates from one rank would diverge the
  others' steering tables. The controller refuses to construct
  otherwise, and the engine only installs the action queue for
  single-rank topologies.
- **Global decode tier only.** The policy aggregates across all active
  requests and steers *everyone*. Per-request actuation needs table-row
  machinery that is Phase 1. Decode-only keeps the prototype
  prefix-cache-safe (base/prefill updates are rejected at drain time).
- **Data parallelism**: each DP replica runs an independent controller
  over its own traffic; replicas will not agree on steering state.

## Usage

```bash
vllm serve google/gemma-3-4b-it \
  --capture-consumers dynamic_steering:monitor_layer=15,probe_path=/data/probe.pt,steering_vector_path=/data/steer.pt,threshold=0.35,hysteresis=0.1,gain=4.0
```

(Install this package first — `uv pip install -e .` from this directory —
so the `dynamic_steering` entry point resolves. Richer params, e.g.
multi-layer `steer_layers`, need the Python API's dict form; the CLI
shorthand only takes flat scalars.)

```python
from vllm import LLM

llm = LLM(
    model="google/gemma-3-4b-it",
    capture_consumers=[{
        "name": "dynamic_steering",
        "params": {
            "monitor_layer": 15,
            "monitor_hook": "post_mlp",
            "probe_path": "/data/probe.pt",
            "steering_vector_path": "/data/steer.pt",
            "steer_layers": [15, 20],
            "threshold": 0.35,
            "hysteresis": 0.10,
            "ema_alpha": 0.25,
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
| `monitor_hook` | str | `post_mlp` | Hook point to probe. |
| `probe_path` | str | required | `torch.save`'d 1-D tensor; unit-normalized at load. |
| `steering_vector_path` | str | required | `torch.save`'d 1-D tensor; applied scaled by gain. |
| `steer_layers` | list[int] | `[monitor_layer]` | Layers receiving the steering vector. |
| `steer_hook` | str | `monitor_hook` | Hook point steered. |
| `score` | str | `cosine` | `cosine` (scale-invariant, in [-1,1]) or `dot` (raw projection onto the unit probe). |
| `threshold` | float | required | Engage level on the aggregated score. |
| `hysteresis` | float | `0.0` | Disengage at `threshold - hysteresis` (anti-flap). |
| `ema_alpha` | float | `0.25` | Per-request EMA smoothing of per-token scores. |
| `gain` | float | `1.0` | Maximum steering scale. |
| `gain_mode` | str | `binary` | `binary` (on/off) or `proportional` (ramps from 0 at disengage level to `gain` at threshold). |
| `aggregate` | str | `max` | Cross-request aggregation: `max` (any request firing engages) or `mean`. |
| `min_emit_delta` | float | `0.05` | Suppress updates whose gain moved less than this (flips always emit). |

## Diagnostics

Each finished request's `capture_results["dynamic_steering"]` carries a
payload: `last_score`, `engaged`, `gain` (last emitted), and cumulative
`updates_emitted` / `updates_dropped`. Drain-side accept/reject counts
live on the action queue (`stats()`), logged as warnings on rejection.

## Validating a probe offline

Run traffic with the [`filesystem` consumer](../../../docs/features/capture_consumers.md)
on the same `(layer, hook)`, fit the probe, then replay traffic with
this controller at `gain=0.0` — decisions are computed and visible in
payloads but no steering is applied. Raise the gain once the engagement
trace looks right.

## Tests

`python test.py` (or `pytest test.py`) — pure CPU, no engine needed.
