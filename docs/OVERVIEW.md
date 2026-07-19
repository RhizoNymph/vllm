# Overview

This document indexes the interpretability-infrastructure features built on this
vLLM fork. They share one data plane — the residual-stream hook points
(`pre_attn`, `post_attn`, `post_block`) and the persistent-buffer + opaque-op
discipline that keeps interventions CUDA-graph-safe — and layer on each other:
capture taps the residual, steering adds to it, patching overwrites it, the
patch source store reuses the capture pipeline, and the dynamic-steering
control plane closes the capture → steering feedback loop in-process.

```yaml
Overview:
  description: >
    Residual-stream interpretability primitives for vLLM — capture, steering,
    and activation patching — sharing one graph-safe data plane and reusing
    each other's machinery, plus a dynamic-steering control plane that closes
    the activation -> steering feedback loop (monitor a residual, decide a
    policy, actuate steering) at the engine step granularity.
  subsystems:
    data_plane: >
      Per-(layer, hook) residual-stream hook points folded into
      apply_layer_steering / apply_block_steering. Persistent buffers + opaque
      ops (mutates_args=[]) so a FULL cudagraph replay reads each step's values.
      Buffers attach at model build via register_steering_buffers; disabled mode
      constant-folds out of the forward. Tier-agnostic scaffolding (hook-attr
      dicts, buffer-sizing knob, kernel warmup harness, phase-tier storage,
      vector-spec validation core) is shared via intervention_common.py /
      intervention_kernel_common.py / phase_tiers.py — see "Intervention Tier
      Template" in docs/features/steering.md.
    control_plane: >
      Per-request specs on SamplingParams / RequestMetadata, resolved
      rank-locally in the model runner (v1 and v2), with scheduler
      admission/backpressure and prefix-cache floors. Runner-agnostic mixins
      hold the shared logic; thin per-runner accessors project the batch.
    stores: >
      Capture consumers (entry-point plugins) sink tapped activations; the
      run-id-keyed PatchSourceStore holds clean-run activations for patching.
    dynamic_steering: >
      The loop joining capture and steering: an action vocabulary
      (SteeringVectorUpdate / RequestSteeringOverride / SteeringScaleUpdate /
      SteeringMonitorUpdate) carried from consumers to the runner via a sync
      on_step return or an async action queue, applied before the next step.
      Steering state lives in per-layer persistent GPU tables (global
      prefill/decode rows, per-request rows, a dynamic additive tier, per-row
      scales, in-graph monitor gates) mutated between steps and visible to
      CUDA-graph replay.
  data_flow: >
    Steering loop: forward pass -> capture hook writes residual to persistent
    buffers -> (sync) StepCaptureView handed to each sync consumer's on_step on
    the step thread -> consumer returns steering actions -> runner applies them
    to the steering tables before the next step builds its steering_index; OR
    (async) chunks dispatched -> consumer.on_capture -> action queue -> drained
    at the top of the next step.
    Patching: a clean prompt is captured once (patch_source consumer ->
    PatchSourceStore); destination requests carry patch specs; the runner
    resolves source vectors and writes per-(layer, hook) buffers before the
    forward; the apply path overwrites/interpolates the residual, then steering
    adds on top. The /v1/patch_sweep endpoint fans a (hooks x layers x
    positions) grid through continuous batching for one-call causal-tracing
    sweeps; when the referenced source run is missing and clean_prompt is given
    it auto-captures the clean run first. Large grids can stream per-cell
    results over SSE (stream: true / client on_cell).

Features Index:
  activation_capture:
    description: >
      Tap residual-stream activations during inference and content-address them
      into a CPU store via pluggable capture consumers (async chunks off the
      critical path, or sync consumers reading zero-copy GPU views on the step
      thread immediately post-forward).
    entry_points:
      - vllm.capture_consumers (entry-point group)
      - "SamplingParams.capture"
      - vllm/v1/capture/consumer.py (CaptureConsumer, SyncCaptureConsumer)
      - vllm/v1/capture/registry.py (entry-point load, build, sync validation)
      - vllm/v1/capture/config.py (graphsafe-key resolution at config build)
    depends_on: []
    doc: docs/features/capture_consumers.md
  activation_steering:
    description: >
      Per-request, per-token, CUDA-graph-safe additive intervention on the
      residual stream (three tiers, five hook points).
    entry_points: ["SamplingParams.steering_vectors", "--enable-steering"]
    depends_on: [activation_capture]
    doc: docs/features/steering.md
  activation_patching:
    description: >
      Overwrite/interpolate residual activations at (layer, hook, position)
      sites with a source vector — a prior clean run's captured activations
      (causal tracing), a named steering module / reserved "zeros", or a
      client-provided packed patch_vectors row — optionally through a per-dim
      mask (alpha·mask folded into a per-dim alpha table). Includes a
      server-side (hooks × layers × positions) sweep endpoint (capture- or
      vector-sourced) with SSE streaming and source-run lifecycle.
    entry_points:
      ["SamplingParams.patch", "SamplingParams.patch_vectors", "--enable-patching", "POST /v1/patch_sweep"]
    depends_on: [activation_capture, activation_steering]
    doc: docs/features/activation_patching.md
  dynamic_steering:
    description: >
      Monitor -> policy -> actuate control plane; sync/async consumers emit
      steering actions applied with 1 (sync) or 1-3 (async) step latency.
    entry_points:
      - vllm/v1/capture/step_view.py (StepCaptureView / StepRequestView)
      - vllm/v1/worker/steering_action_queue.py (action vocabulary + queue)
    depends_on: [activation_capture, activation_steering]
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
    depends_on: [activation_capture, dynamic_steering]
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
