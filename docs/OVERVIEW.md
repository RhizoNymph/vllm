# Overview

> Scoped to the **steering / capture-consumer** subsystem this fork develops
> on top of upstream vLLM (the rest of vLLM is documented upstream).

```yaml
Overview:
  description: >
    Adds activation capture and activation steering to vLLM's v1/v2 model
    runners, plus a dynamic-steering control plane that closes the
    activation -> steering feedback loop (monitor a residual, decide a
    policy, actuate steering) at the engine step granularity.
  subsystems:
    capture: >
      Taps model residuals at hook points and delivers them to pluggable
      consumers. Async consumers receive batched chunks off the critical
      path (dispatch/finalize). Sync consumers run on the model-runner step
      thread immediately post-forward and read zero-copy GPU views.
    steering: >
      Per-layer persistent GPU steering tables (global prefill/decode rows,
      per-request rows, a dynamic additive tier, per-row scales, in-graph
      monitor gates) mutated between steps and visible to CUDA-graph replay.
    dynamic_steering: >
      The control plane joining the two: an action vocabulary
      (SteeringVectorUpdate / RequestSteeringOverride / SteeringScaleUpdate /
      SteeringMonitorUpdate) carried from consumers to the runner via a sync
      on_step return or an async action queue, applied before the next step.
  data_flow: >
    forward pass -> capture hook writes residual to persistent buffers ->
    (sync) StepCaptureView handed to each sync consumer's on_step on the step
    thread -> consumer returns steering actions -> runner applies them to the
    steering tables before the next step builds its steering_index; OR
    (async) chunks dispatched -> consumer.on_capture -> action queue ->
    drained at the top of the next step.
```

## Features Index

```yaml
capture_consumers:
  description: Pluggable capture-consumer framework (async + sync execution).
  entry_points:
    - vllm/v1/capture/consumer.py (CaptureConsumer, SyncCaptureConsumer)
    - vllm/v1/capture/registry.py (entry-point load, build, sync validation)
    - vllm/v1/capture/config.py (graphsafe-key resolution at config build)
  doc: docs/features/capture_consumers.md

dynamic_steering:
  description: >
    Monitor -> policy -> actuate control plane; sync/async consumers emit
    steering actions applied with 1 (sync) or 1-3 (async) step latency.
  entry_points:
    - vllm/v1/capture/step_view.py (StepCaptureView / StepRequestView)
    - vllm/v1/worker/steering_action_queue.py (action vocabulary + queue)
  depends_on: [capture_consumers, steering]
  doc: docs/design/dynamic_steering.md

consumer_controller_base:
  description: >
    The SyncCaptureConsumer contract ABC and the SteeringController base.
    SyncCaptureConsumer encodes the sync contract (on_step +
    global_capture_spec abstract; declared_graphsafe_keys defaulted to [])
    so an incomplete consumer fails clearly at construction instead of deep
    in config-build. SteeringController sits on top and owns per-request
    lifecycle, conversation scoping (bounded FIFO), and the
    trigger->latch->bridge pattern; subclasses implement only decide().
  entry_points:
    - vllm/v1/capture/consumer.py (SyncCaptureConsumer)
    - vllm/v1/capture/controller.py (SteeringController)
    - examples/.../minimal_examples.py (_SyncBase, ConversationLatchExample)
  depends_on: [capture_consumers, dynamic_steering]
  doc: docs/design/dynamic_steering.md  # §5.6

declarative_per_request_steering:
  description: >
    A client attaches its own conditional steering to a request (a nested list
    of when x scope x apply gates in RequestMetadata.steering) with NO
    server-registered consumer. A built-in auto-registered consumer maps gates
    to the steering substrate: probe x this_token gates run in-graph via the
    per-row monitor; other scopes are host-latched (reusing SteeringController).
    Vector sources are name-first (a probe/steer registry mirrored to every
    worker) with an inline base64 packed escape hatch; a NamedVec rides the wire
    un-inflated and resolves worker-side at admission. rest_of_conversation add
    persists server-side by reference to a registered name (inline refused) and
    bridges later turns by re-resolving the name with a digest guard; operator
    consumers win over client gates.
  entry_points:
    - vllm/v1/steering_schema.py (gate schema + resolve/build)
    - vllm/v1/capture/declarative.py (DeclarativeSteeringConsumer)
    - vllm/v1/capture/controller.py (ByRefLatch + latch/bridge)
    - vllm/v1/worker/steering_vector_registry.py (worker named-vector registry)
    - vllm/entrypoints/openai/steering/vector_registry.py (frontend mirror)
    - vllm/entrypoints/serve/steering/vectors_router.py (admin endpoints)
  depends_on: [dynamic_steering, consumer_controller_base]
  doc: docs/design/dynamic_steering.md  # §8.2
```
