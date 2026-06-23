# Overview

description: |
  Alternative Rust frontend to the vLLM engine. Serves OpenAI-compatible HTTP
  and a gRPC `Generate` service, lowers requests through a tokenize/chat-template
  pipeline, and talks to the Python engine-core southbound over ZMQ + msgpack.

subsystems: |
  - server (`src/server`)            HTTP + gRPC entry points, request/response DTOs.
  - chat (`src/chat`)                Chat-template rendering, tool/reasoning parsing → TextRequest.
  - text (`src/text`)                Tokenization + lowering: TextRequest → GenerateRequest.
  - llm (`src/llm`)                  GenerateRequest → EngineCoreRequest.
  - engine-core-client (`src/...`)   ZMQ msgpack transport + southbound DTOs (EngineCore{Request,SamplingParams,Output}).
  - tokenizer / tool-parser / reasoning-parser / metrics  supporting crates.

data_flow: |
  Inbound HTTP (CompletionRequest / ChatCompletionRequest) or gRPC
  (pb::GenerateRequest) → vllm_text::SamplingParams + Prompt → lower_sampling_params
  → EngineCoreSamplingParams → EngineCoreRequest → ZMQ (rmp_serde to_vec_named) →
  Python engine-core. Outputs return as EngineCoreOutputs over ZMQ and are decoded
  back up to the response DTOs.

  Backend file resolution: the frontend loads the tokenizer + config + chat
  template from `--model` by default. `--tokenizer <path>` overrides only the
  frontend's resolution source (LoadModelBackendsOptions.tokenizer →
  ResolvedModelFiles::new), so the public model id / engine model can differ from
  where the frontend reads its tokenizer. This decouples the two for formats
  without a loadable `tokenizer.json` (e.g. a GGUF model whose tokenizer the Rust
  `tokenizers` crate cannot extract, unlike the Python engine which reads it from
  the GGUF metadata).

# Features Index

features:
  steering_capture:
    description: Per-request activation steering and capture forwarded to engine-core.
    entry_points:
      - src/server/src/routes/openai/completions/convert.rs
      - src/server/src/routes/openai/chat_completions/convert.rs
      - src/server/src/grpc/convert.rs
    depends_on: [text lowering, engine-core-client southbound DTOs]
    doc: docs/features/steering_capture.md
