Overview:
    description:
        vLLM is a high-throughput, memory-efficient inference and serving engine
        for large language models.  It provides an OpenAI-compatible API server
        with continuous batching, PagedAttention, CUDA graph support, and
        torch.compile integration.

    subsystems:
        engine:
            The core scheduling and request lifecycle management layer.
            Accepts requests, tokenizes, schedules batches, and returns outputs.
        model_executor:
            Loads and runs model forward passes.  Contains model definitions,
            quantization, LoRA, and custom ops (including steering).
        worker:
            Manages GPU resources, the persistent input batch, and per-step
            buffer updates (KV cache, steering tables, LoRA adapters).
        entrypoints:
            HTTP/gRPC server (OpenAI-compatible) and offline LLM API.

    data_flow:
        Request arrives at entrypoints → engine validates and enqueues →
        scheduler picks requests respecting capacity constraints (tokens,
        KV blocks, LoRA slots, steering config slots) → model runner
        prepares inputs and updates GPU buffers → model forward pass →
        sampler selects tokens → engine streams output back to client.

Features Index:
    steering:
        description: >
            Activation steering — inject additive vectors into the residual
            stream to steer model behaviour.  Supports global (server-wide)
            and per-request steering with a three-tier additive composition
            model: base vectors (both phases) + prefill-specific +
            decode-specific.  Co-located scale factors and three hook points
            (pre_attn, post_attn, post_mlp).
            Scheduler predicts mid-step prefill-to-decode transitions and
            reserves capacity for both phases; model runner gracefully defers
            decode registration when capacity is temporarily exhausted.
            Deferred entries use a two-queue priority model: prefill→decode
            transitions are retried before new-request deferrals.
            Status endpoint reports base, prefill, and decode vector norms.
        entry_points:
            - POST /v1/steering/set (global)
            - POST /v1/steering/clear (global)
            - GET /v1/steering (status with phase-specific norms)
            - POST /v1/steering/modules/register (named module registration)
            - POST /v1/steering/modules/unregister (named module removal)
            - GET /v1/steering/modules (list named modules)
            - --steering-modules name=path (CLI pre-registration)
            - SamplingParams.steering_vectors (per-request base)
            - SamplingParams.prefill_steering_vectors (per-request prefill)
            - SamplingParams.decode_steering_vectors (per-request decode)
            - extra_body.steering_name (per-request named module reference)
        depends_on: []
        doc: docs/features/steering.md
    capture_consumers:
        description: >
            Pluggable capture-consumer framework — observe, record, or
            react to activations captured during the forward pass.
            Replaces the legacy activation_storing feature with a general-
            purpose plugin system.  Built-in consumers: filesystem
            (streaming writer), logging (debug observer).  Third-party
            consumers register via Python entry points in the
            ``vllm.capture_consumers`` group, implementing either the
            ``CaptureSink`` streaming protocol or the ``CaptureConsumer``
            batched base class.
        entry_points:
            - --capture-consumers SPEC (CLI, repeatable)
            - SamplingParams.capture (per-request, dict keyed by consumer name)
            - LLM(capture_consumers=[...]) (Python API)
            - VllmConfig.capture_consumers_config (programmatic)
            - pyproject.toml [project.entry-points."vllm.capture_consumers"]
        depends_on: [steering]
        doc: docs/capture_consumers/design.md
    capture_consumers_filesystem:
        description: >
            Built-in filesystem capture consumer.  Implements ``CaptureSink``
            directly to stream captured activations to disk without
            buffering full tensors in memory.  Registered as the
            ``filesystem`` entry point in the ``vllm.capture_consumers``
            group.  Per-request validation is in-package.
        entry_points:
            - vllm.v1.capture.consumers.filesystem.FilesystemConsumer
        depends_on: [capture_consumers]
        doc: docs/features/capture_consumers_filesystem.md
