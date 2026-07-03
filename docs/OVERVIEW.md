# Overview

This document indexes the interpretability-infrastructure features built on this
vLLM fork. They share one data plane — the residual-stream hook points
(`pre_attn`, `post_attn`, `post_block`) and the persistent-buffer + opaque-op
discipline that keeps interventions CUDA-graph-safe — and layer on each other:
capture taps the residual, steering adds to it, patching overwrites it, and the
patch source store reuses the capture pipeline.

```yaml
Overview:
  description: >
    Residual-stream interpretability primitives for vLLM — capture, steering,
    and activation patching — sharing one graph-safe data plane and reusing each
    other's machinery.
  subsystems:
    data_plane: >
      Per-(layer, hook) residual-stream hook points folded into
      apply_layer_steering / apply_block_steering. Persistent buffers + opaque
      ops (mutates_args=[]) so a FULL cudagraph replay reads each step's values.
      Buffers attach at model build via register_steering_buffers; disabled mode
      constant-folds out of the forward.
    control_plane: >
      Per-request specs on SamplingParams, resolved rank-locally in the model
      runner (v1 and v2), with scheduler admission/backpressure and prefix-cache
      floors. Runner-agnostic mixins hold the shared logic; thin per-runner
      subclasses project the batch.
    stores: >
      Capture consumers (entry-point plugins) sink tapped activations; the
      run-id-keyed PatchSourceStore holds clean-run activations for patching.
  data_flow: >
    A clean prompt is captured once (patch_source consumer → PatchSourceStore).
    Destination requests carry patch specs; the runner resolves source vectors
    and writes per-(layer, hook) buffers before the forward; the apply path
    overwrites/interpolates the residual, then steering adds on top. The
    /v1/patch_sweep endpoint fans a (layers × positions) grid through continuous
    batching for one-call causal-tracing sweeps; when the referenced source run
    is missing and clean_prompt is given it auto-captures the clean run first.

Features Index:
  activation_capture:
    description: >
      Tap residual-stream activations during inference and content-address them
      into a CPU store via pluggable capture consumers.
    entry_points: [vllm.capture_consumers, "SamplingParams.capture"]
    depends_on: []
    doc: docs/features/capture_consumers.md
  activation_steering:
    description: >
      Per-request, per-token, CUDA-graph-safe additive intervention on the
      residual stream (three tiers, three hook points).
    entry_points: ["SamplingParams.steering_vectors", "--enable-steering"]
    depends_on: [activation_capture]
    doc: docs/features/steering.md
  activation_patching:
    description: >
      Overwrite/interpolate residual activations at (layer, hook, position)
      sites with a prior clean run's captured activations — the causal-tracing
      primitive. Includes a server-side (layers × positions) sweep endpoint.
    entry_points:
      ["SamplingParams.patch", "--enable-patching", "POST /v1/patch_sweep"]
    depends_on: [activation_capture, activation_steering]
    doc: docs/features/activation_patching.md
```
