# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import ConfigDict, Field

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash


@config(config=ConfigDict(arbitrary_types_allowed=True))
class PatchConfig:
    """Configuration for activation patching (overwrite/interpolate residual
    activations with vectors captured from a prior clean run)."""

    max_patch_slots: int = Field(default=64, ge=1)
    """Max distinct patched positions at a single (layer, hook) site in one
    forward step. Bounds the per-(layer, hook) patch-table rows. Strict policy:
    the scheduler reserves capacity and a breach is a loud error, never a silent
    drop. Slot 0 is the passthrough sentinel, so the usable count is one less."""

    patch_source_cache_bytes: int = Field(default=0, ge=0)
    """Byte budget for the clean-run source store (CPU). ``0`` disables it.
    Sources are referenced by run handle across a sweep, so size this to hold at
    least one full clean run's captured activations. Evicted whole-run (LRU)."""

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
        factors: list = []
        # max_patch_slots sizes the per-layer patch buffers baked into the
        # model (and the cudagraph), so it affects the computation graph.
        factors.append(self.max_patch_slots)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str
