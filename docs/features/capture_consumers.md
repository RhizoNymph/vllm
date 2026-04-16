# Capture Consumers

## Scope

**In scope:**
- Configuration dataclasses (`CaptureConsumerSpec`, `CaptureConsumersConfig`)
- CLI shorthand parser (`parse_consumer_spec`) and validator (`validate_consumer_specs`)
- Wiring into `VllmConfig` and `EngineArgs` alongside the existing `ActivationStoringConfig`
- Core types, protocol (`CaptureConsumer`), registry, and sink (added in earlier phases)

**Not in scope (yet):**
- Runtime integration in the model runner (Phase D)
- Built-in consumer implementations (filesystem, S3, etc.)
- Per-request capture spec on `SamplingParams`
- OpenAI-compatible API endpoint for capture results

## Data / Control Flow

```
CLI flags (--capture-consumers)
    |
    v
EngineArgs.capture_consumers: list[str] | None
    |  (parse_consumer_spec for each shorthand string)
    v
list[CaptureConsumerSpec]
    |  (validate_consumer_specs — uniqueness, non-empty names)
    v
CaptureConsumersConfig(consumers=[...])
    |
    v
VllmConfig.capture_consumers_config
    |
    v  (Phase D — not yet wired)
Runner reads config → registry.build_consumer(spec) → CaptureConsumer instances
```

## File Inventory

| File | Role | Key Exports |
|------|------|-------------|
| `vllm/v1/capture/config.py` | Config dataclasses + parsing | `CaptureConsumerSpec`, `CaptureConsumersConfig`, `parse_consumer_spec`, `validate_consumer_specs` |
| `vllm/config/capture_consumers.py` | Re-export shim for `vllm.config` pattern | `CaptureConsumersConfig`, `CaptureConsumerSpec` |
| `vllm/config/vllm.py` | `VllmConfig` field + `compute_hash()` integration | `VllmConfig.capture_consumers_config` |
| `vllm/config/__init__.py` | Public export | `CaptureConsumersConfig` |
| `vllm/engine/arg_utils.py` | CLI flag + `create_engine_config()` wiring | `EngineArgs.capture_consumers` field, `--capture-consumers` flag |
| `vllm/v1/capture/types.py` | Core value types (earlier phase) | `CaptureSpec`, `CaptureChunk`, `CaptureKey`, etc. |
| `vllm/v1/capture/consumer.py` | Abstract consumer protocol (earlier phase) | `CaptureConsumer` |
| `vllm/v1/capture/registry.py` | Entry-point based consumer registry (earlier phase) | `load_consumer_class`, `build_consumer` |
| `vllm/v1/capture/sink.py` | Per-consumer batched adapter (earlier phase) | `CaptureSink` |
| `tests/v1/capture/test_config.py` | Config unit tests | 21 tests |
| `tests/engine/test_arg_utils.py` | CLI flag tests (additive) | `TestCaptureConsumersFlag` (5 tests) |

## Invariants and Constraints

1. **Coexistence**: `--activation-storing*` flags and `--capture-consumers` flags coexist. Neither removes or shadows the other. Both `activation_storing_config` and `capture_consumers_config` may be set simultaneously on `VllmConfig`.

2. **Instance name uniqueness**: When no `instance_name` is provided, the consumer's entry-point `name` is used as its effective instance name. All effective instance names must be unique across the consumer list.

3. **Non-empty names**: Every `CaptureConsumerSpec.name` must be a non-empty, non-whitespace string.

4. **Deterministic hashing**: `CaptureConsumersConfig.compute_hash()` produces the same 16-character hex string for the same input on every call. Changing any field (name, instance_name, params keys/values) changes the hash.

5. **No runtime side effects**: The config module (`vllm/v1/capture/config.py`) is pure data — no I/O, no GPU ops, no process spawning. Consumer instantiation happens downstream in the runner (Phase D).

6. **CLI shorthand format**: `name:key=val,key=val`. No commas or equals in values; use YAML config for complex parameters. First colon splits name from params; first equals in each pair splits key from value.
