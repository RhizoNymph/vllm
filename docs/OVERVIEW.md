# vLLM Documentation Overview

## Description

vLLM is a high-throughput, memory-efficient inference and serving engine for
large language models. It provides an OpenAI-compatible API server, continuous
batching, PagedAttention for efficient KV cache management, tensor/pipeline
parallelism, and a pluggable model architecture system.

## Subsystems

- **Scheduler**: Manages request admission, batching, and resource accounting.
  Controls which requests are prefilled and which decode in a given step.
- **Worker / Model Runner**: Executes forward passes on GPU. The model runner
  assembles input tensors, manages CUDA graphs, and dispatches to model code.
- **KV Cache / Block Manager**: Allocates and tracks GPU memory for key-value
  caches using paged allocation.
- **Model Architectures**: Per-family model implementations that wire
  attention, MLP, and any optional hooks (e.g., steering) into a forward pass.
- **API Server**: OpenAI-compatible HTTP server for chat, completions, and
  embeddings endpoints.
- **Tokenizer / Processor**: Handles prompt tokenization, multimodal input
  processing, and detokenization of outputs.

## Data Flow

1. Client sends a request to the API server.
2. The server tokenizes the prompt and creates an internal `Request`.
3. The scheduler admits the request subject to memory and capacity limits.
4. The model runner assembles a batch of tokens and runs the forward pass.
5. Sampling selects next tokens; finished requests are returned to the client.

Steering, LoRA, speculative decoding, and other features hook into this
pipeline at specific points (scheduler admission, model runner preparation,
or model forward pass).

## Features Index

### steering
- **description**: Activation steering adds precomputed vectors into the
  residual stream of decoder layers during inference, enabling per-request
  behavioral shifts without fine-tuning. The scheduler guarantees
  steering-table capacity before admission; registration failure at the
  worker is a hard error because tokens generated under wrong steering
  poison the KV cache permanently.
- **entry_points**: `SamplingParams.steering_vectors`,
  `--enable-steering`, `--max-steering-configs`
- **depends_on**: [scheduler, kv_cache, prefix_caching]
- **doc**: [docs/features/steering.md](features/steering.md)

### automatic_prefix_caching
- **description**: Reuses KV cache blocks across requests that share prompt
  prefixes, reducing redundant computation.
- **entry_points**: `--enable-prefix-caching`
- **depends_on**: [kv_cache]
- **doc**: [docs/features/automatic_prefix_caching.md](features/automatic_prefix_caching.md)

### lora
- **description**: Serves multiple LoRA adapters concurrently on a single
  base model, with per-request adapter selection.
- **entry_points**: `--enable-lora`, `SamplingParams.lora_request`
- **depends_on**: [model_runner]
- **doc**: [docs/features/lora.md](features/lora.md)

### speculative_decoding
- **description**: Uses a smaller draft model to propose tokens verified by
  the target model, improving generation throughput.
- **entry_points**: `--speculative-model`
- **depends_on**: [scheduler, model_runner]
- **doc**: [docs/features/speculative_decoding/](features/speculative_decoding)

### structured_outputs
- **description**: Constrains generation to follow JSON schemas, regex
  patterns, or grammar specifications.
- **entry_points**: `SamplingParams.guided_decoding`
- **depends_on**: []
- **doc**: [docs/features/structured_outputs.md](features/structured_outputs.md)

### tool_calling
- **description**: Enables models to produce structured tool/function calls
  in chat completions.
- **entry_points**: `tools` parameter in chat API
- **depends_on**: [structured_outputs]
- **doc**: [docs/features/tool_calling.md](features/tool_calling.md)

### multimodal_inputs
- **description**: Supports image, audio, and video inputs alongside text
  for multimodal models.
- **entry_points**: `SamplingParams.multi_modal_data`
- **depends_on**: [model_runner]
- **doc**: [docs/features/multimodal_inputs.md](features/multimodal_inputs.md)

### disaggregated_prefill
- **description**: Separates prefill and decode across different worker
  groups for improved resource utilization.
- **entry_points**: `--disagg-prefill`
- **depends_on**: [scheduler, kv_cache]
- **doc**: [docs/features/disagg_prefill.md](features/disagg_prefill.md)
