# How to Write a Capture Consumer Plugin

## 1. Overview

Capture consumers are pluggable observers that receive activations
captured during vLLM's forward pass. They let you record, analyze, or
react to hidden-state tensors without modifying vLLM core.

Write a capture consumer when you need to:

- Stream activations to a custom storage backend (S3, database,
  shared memory).
- Compute online statistics (norms, cosine similarity, anomaly
  scores) during inference.
- Feed activations into a co-located training loop (SAE, linear
  probe, reward model).
- Integrate with an observability pipeline (dashboards, alerting).

vLLM discovers consumers at engine startup via Python
[entry points](https://packaging.python.org/en/latest/specifications/entry-points/).
No vLLM source edits are needed — install your package and configure
the engine.

## 2. Quick Start

Subclass `CaptureConsumer`, implement `on_capture`, and register via
an entry point.

```python
from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.types import CaptureKey, CaptureSpec
import torch
from typing import Any

class MyConsumer(CaptureConsumer):
    location = "worker"

    def __init__(self, vllm_config, params: dict[str, Any]) -> None:
        self._layers = params.get("layers", [0])

    def global_capture_spec(self) -> CaptureSpec:
        return CaptureSpec(
            hooks={"post_mlp": self._layers},
            positions="last_prompt",
        )

    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        # tensor shape: (num_rows, hidden_size)
        print(f"Captured {key}: {tensor.shape}")
```

## 3. Entry-Point Registration

In your package's `pyproject.toml`, advertise your class under the
`vllm.capture_consumers` group:

```toml
[project.entry-points."vllm.capture_consumers"]
my_consumer = "my_package.module:MyConsumer"
```

The key (`my_consumer`) is the name users pass on the CLI or in
config to activate your consumer:

```bash
vllm serve model --capture-consumers my_consumer:layers=[0,1,2]
```

vLLM resolves entry points once at engine startup and caches the
result for the process lifetime. See `vllm/v1/capture/registry.py`
for implementation details.

## 4. Global vs Per-Request Specs

### Global spec — `global_capture_spec()`

Return a `CaptureSpec` from `global_capture_spec()` to capture the
same hooks and positions for every request. This is the most common
pattern — the consumer always needs the same data.

```python
def global_capture_spec(self) -> CaptureSpec:
    return CaptureSpec(
        hooks={"post_mlp": [0, 15, 31]},
        positions="last_prompt",
    )
```

### Per-request spec — `validate_client_spec()`

Set the class variable `reads_client_spec = True` and override
`validate_client_spec()` to accept per-request configuration via
`SamplingParams.capture[consumer_name]`. The framework calls your
validator at admission time with the raw client spec and a
`CaptureContext` describing the request shape.

```python
class FlexConsumer(CaptureConsumer):
    reads_client_spec = True

    def validate_client_spec(self, raw_spec, ctx):
        hooks = raw_spec.get("hooks", {"post_mlp": list(range(ctx.num_hidden_layers))})
        positions = raw_spec.get("positions", "all_prompt")
        return CaptureSpec(hooks=hooks, positions=positions)
```

Raise `CaptureValidationError` if the client spec is invalid — the
serving layer converts it to an HTTP 400.

## 5. Worker vs Driver Location

The `location` class variable controls where your consumer runs:

### `"worker"` (default)

The consumer runs in the engine-core subprocess alongside the model
runner. It has direct in-process access to captured tensors with zero
IPC overhead. Use this when:

- You need maximum throughput.
- You are writing to local disk or shared memory.
- You are computing online statistics that do not need the main
  process.

### `"driver"`

The consumer runs in the main Python process where the `LLM` object
lives. Tensors are shipped from worker to driver via
`torch.multiprocessing.Queue` with shared-memory handoff. Use this
when:

- You need access to the main process (e.g., to update a shared data
  structure the API layer reads).
- You are integrating with a framework that is not fork-safe.

The driver path adds IPC latency per capture event. Prefer `"worker"`
unless you have a specific reason to be in the driver process.

## 6. Streaming Consumers

`CaptureConsumer` buffers all chunks for a key in memory and delivers
them as a single concatenated tensor on finalization. For very long
sequences this can use significant memory.

If you need true streaming semantics — writing rows incrementally as
they arrive — implement the `CaptureSink` protocol directly instead
of subclassing `CaptureConsumer`:

```python
from vllm.v1.capture.sink import CaptureSink
from vllm.v1.capture.types import (
    CaptureChunk, CaptureFinalize, CaptureKey, CaptureResult,
)

class StreamingWriter:
    """Writes chunks to disk as they arrive."""
    location = "worker"

    def submit_chunk(self, chunk: CaptureChunk) -> None:
        # Write chunk.tensor immediately — no buffering.
        ...

    def submit_finalize(self, finalize: CaptureFinalize) -> None:
        # Flush and record the terminal result.
        ...

    def get_result(self, key: CaptureKey) -> CaptureResult | None:
        ...

    def shutdown(self, timeout: float = 30.0) -> None:
        ...
```

All `CaptureSink` methods must be thread-safe.

## 7. Testing Your Consumer

### Unit testing with `_BatchedAdapter`

The internal `_BatchedAdapter` wraps any `CaptureConsumer` as a
`CaptureSink`. Use it in tests to exercise the full chunk-accumulate-
finalize lifecycle without an engine:

```python
from unittest.mock import MagicMock
from vllm.v1.capture.consumer import _BatchedAdapter
from vllm.v1.capture.types import (
    CaptureChunk, CaptureFinalize, VllmInternalRequestId,
)
import torch

consumer = MyConsumer(MagicMock(), {"layers": [0]})
adapter = _BatchedAdapter(consumer)

key = (VllmInternalRequestId("test-req"), 0, "post_mlp")

adapter.submit_chunk(CaptureChunk(
    key=key,
    tensor=torch.randn(4, 128),
    dtype=torch.float32,
    row_offset=0,
    step_index=0,
))
adapter.submit_finalize(CaptureFinalize(key=key))

result = adapter.get_result(key)
assert result is not None
assert result.status == "ok"
```

### Fake configs

In tests, pass `unittest.mock.MagicMock()` as `vllm_config`. The
base `CaptureConsumer.__init__` is a no-op, so it does not inspect
the config object. Your consumer should only read from `params`.

### Running the test suite

```bash
.venv/bin/python -m pytest tests/v1/capture/ -v
```

## 8. Example: SumConsumer

A complete worked example that records the sum of every captured
tensor. See `docs/capture_consumers/examples/minimal_plugin/` for a
pip-installable package version.

```python
from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.types import CaptureKey, CaptureSpec
import torch
from typing import Any, ClassVar, Literal

class SumConsumer(CaptureConsumer):
    """Records the sum of every captured tensor."""
    location: ClassVar[Literal["worker", "driver"]] = "worker"

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        self.sums: dict[CaptureKey, float] = {}
        self._layers: list[int] = params.get("layers", [0])

    def global_capture_spec(self) -> CaptureSpec:
        return CaptureSpec(
            hooks={"post_mlp": self._layers},
            positions="last_prompt",
        )

    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        self.sums[key] = float(tensor.sum().item())
```

Register it in your `pyproject.toml`:

```toml
[project.entry-points."vllm.capture_consumers"]
sum = "my_plugin:SumConsumer"
```

Then activate it:

```bash
vllm serve my-model --capture-consumers sum:layers=[0,15,31]
```
