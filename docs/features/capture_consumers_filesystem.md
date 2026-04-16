# Filesystem Capture Consumer (Phase C)

## Scope

**In scope:**
- `FilesystemConsumer` class implementing `CaptureSink` directly (streaming, not batched)
- Wrapping `ActivationWriter` from `vllm.v1.worker.activation_writer`
- Per-request validation via delegation to the existing admission validator
- Entry-point registration as `filesystem` in `vllm.capture_consumers`
- Types: `FilesystemCaptureRequest`, `FilesystemConsumerParams`

**Not in scope:**
- Modifying `ActivationWriter` internals
- Modifying the existing admission validator
- Manager-side wiring (Phase D+)
- Multi-rank (TP/PP > 1) capture support

## Data/Control Flow

```
Client request (SamplingParams.capture["filesystem"])
    |
    v
FilesystemConsumer.validate_client_spec()
    |-- Converts FilesystemCaptureRequest -> ActivationStoringSpec
    |-- Delegates to validate_activation_storing()
    |-- Converts ResolvedActivationStoringSpec -> CaptureSpec
    v
CaptureSpec returned to framework
    ...
(During forward pass, manager calls submit_chunk per captured activation)
    |
    v
FilesystemConsumer.submit_chunk(CaptureChunk)
    |-- Extracts tag_slug, request_id_slug from chunk.metadata
    |-- Computes path: {root}/{tag_slug}/{request_id_slug}/{layer}_{hook}.bin
    |-- Converts tensor to bytes via tensor.numpy().tobytes()
    |-- Submits WriteTask(path, payload, append=True, key) to ActivationWriter
    v
(On request completion, manager calls submit_finalize)
    |
    v
FilesystemConsumer.submit_finalize(CaptureFinalize)
    |-- Builds FinalizeTask with bin_path, sidecar_path, sidecar JSON
    |-- Submits to ActivationWriter
    v
ActivationWriter background threads
    |-- fsync + atomic rename .bin.tmp -> .bin
    |-- Write + fsync + rename .json.tmp -> .json
    v
FilesystemConsumer.get_result(key)
    |-- Maps WriteResult -> CaptureResult
    v
CaptureResult returned to framework -> attached to RequestOutput
```

## File Manifest

| File | Role | Key exports |
|------|------|-------------|
| `vllm/v1/capture/consumers/__init__.py` | Package marker | - |
| `vllm/v1/capture/consumers/filesystem/__init__.py` | Re-exports | `FilesystemConsumer`, `FilesystemCaptureRequest` |
| `vllm/v1/capture/consumers/filesystem/types.py` | Data types | `FilesystemCaptureRequest`, `FilesystemConsumerParams` |
| `vllm/v1/capture/consumers/filesystem/validation.py` | Validation bridge | `validate_filesystem_request()` |
| `vllm/v1/capture/consumers/filesystem/consumer.py` | Consumer impl | `FilesystemConsumer` |
| `tests/v1/capture/consumers/filesystem/test_consumer.py` | Tests | 10 test cases |
| `pyproject.toml` | Entry-point registration | `filesystem` entry point |

## Invariants and Constraints

1. **No memory buffering**: `FilesystemConsumer` implements `CaptureSink` directly (not `CaptureConsumer` via `_BatchedAdapter`) because long captures must stream to disk incrementally.

2. **Path layout**: `{root}/{tag_slug}/{request_id_slug}/{layer}_{hook}.bin` must match the existing activation-storing layout for compatibility.

3. **Key compatibility**: Phase A's `CaptureKey = tuple[VllmInternalRequestId, int, str]` vs the writer's `CaptureKey = tuple[str, int, str]`. `VllmInternalRequestId` is a `NewType(str)` so they are compatible at runtime; the consumer does `str(_request_id)` when passing to the writer.

4. **Slug propagation**: Tag and request_id slugs travel via `chunk.metadata["tag_slug"]` / `chunk.metadata["request_id_slug"]` and are cached in `_key_paths` for use at finalize time. If no metadata is present, defaults are used.

5. **Validation isolation**: The validation module imports `vllm.config` (which requires pydantic). The import is lazy (inside `validate_client_spec`) so the consumer module can be imported in lightweight environments. Tests that exercise validation are skipped when pydantic is unavailable.

6. **Thread safety**: `_key_paths` is protected by `_lock`. The underlying `ActivationWriter` handles its own thread safety.

7. **No existing file modifications**: This phase creates only new files (except the one-line pyproject.toml entry-point addition). It does not touch `activation_writer.py`, `activation_storing_validation.py`, or any existing test.
