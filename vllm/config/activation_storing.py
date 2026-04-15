# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server-global configuration for per-request activation storing.

Mirrors the shape of :class:`SteeringConfig` but owns the parameters the
activation storing feature needs at server startup: the on-disk root, the
writer pool knobs, collision policy, and a per-request byte budget used at
admission time.

This module is intentionally small and free of runtime behavior. A server
that does not opt into the feature (``--activation-storing`` unset) keeps
``VllmConfig.activation_storing_config = None`` and every subsystem treats
that as "feature disabled, skip everything".
"""

from typing import Literal

from pydantic import ConfigDict, Field

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

# Collision policy literal surfaced by CLI (``--activation-storing-on-collision``)
# and the OpenAI entrypoint.
ActivationStoringCollisionPolicy = Literal["overwrite", "error", "suffix"]


@config(config=ConfigDict(arbitrary_types_allowed=True))
class ActivationStoringConfig:
    """Configuration for per-request activation storing.

    The feature is disabled when :attr:`root_path` is ``None``. All other
    fields are only consulted when a request provides an
    ``activation_storing`` spec; they have no effect otherwise.
    """

    root_path: str | None = None
    """Filesystem root that all activation captures are written under.

    When ``None`` the feature is disabled: any per-request
    ``activation_storing`` field is rejected by the admission validator and
    no writer pool is spun up on the worker.
    """

    writer_queue_size: int = Field(default=1024, ge=1)
    """Bounded queue size between the capture manager and the writer pool.

    When full, the model runner's finalize step blocks until a slot frees,
    up to ``writer_timeout_seconds``. This creates natural backpressure on
    slow disks without running the engine out of memory.
    """

    writer_timeout_seconds: int = Field(default=180, ge=1)
    """Per-write timeout in seconds. Exceeding the timeout surfaces as
    ``capture_status = "partial_error"`` on the owning request without
    aborting text generation."""

    writer_threads: int = Field(default=4, ge=1)
    """Size of the writer thread pool. Raise if you have many concurrent
    slow-disk writes (e.g., high-latency NFS)."""

    on_collision: ActivationStoringCollisionPolicy = "overwrite"
    """How to resolve collisions on ``(tag, layer, hook, request_id)``:

    - ``"overwrite"``: truncate the existing files at write time.
    - ``"error"``: reject the request at admission time with HTTP 400.
    - ``"suffix"``: append a ``.{unix_ms}`` suffix to ``request_id``. Breaks
      determinism — clients must parse the response to find the final path.
    """

    max_bytes_per_request: int = Field(default=0, ge=0)
    """Per-request byte cap for estimated capture size, enforced at
    admission time. ``0`` means unbounded; any positive value rejects
    requests whose estimated
    ``num_positions × total_layers × hidden_size × element_size``
    exceeds the cap before the request is scheduled."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # None of the activation-storing fields change the computation graph
        # — the capture custom op is constant-folded out of the compiled
        # graph when the feature is disabled. We still hash the enable bit so
        # a switch from disabled → enabled invalidates the compile cache.
        factors: list = []
        factors.append(self.root_path is not None)
        factors.append(self.writer_queue_size)
        factors.append(self.writer_timeout_seconds)
        factors.append(self.writer_threads)
        factors.append(self.on_collision)
        factors.append(self.max_bytes_per_request)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str
