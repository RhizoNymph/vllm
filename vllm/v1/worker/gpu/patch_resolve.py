# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve a request's ``patch`` spec into source-vector :class:`PatchEntry`s.

A patch spec entry references a clean run's stored activation by
``(source_run, layer, hook, source_position)``; resolution looks each up in the
per-worker patch source store and pairs it with the destination
``(layer, hook, dest_position, alpha)``. The store and the spec field are wired
in later phases; until then this returns an empty list so the runner lifecycle
is import-safe and a no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.worker.patch_runner_mixin import PatchEntry

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData

logger = init_logger(__name__)


def resolve_patch_entries(
    new_req_data: NewRequestData,
    *,
    local_layers: frozenset[int],
) -> list[PatchEntry]:
    """Resolve ``new_req_data``'s patch spec into entries for local layers.

    Returns ``[]`` until the source store + spec field are wired (later phase).
    """
    sampling_params = getattr(new_req_data, "sampling_params", None)
    if sampling_params is None:
        return []
    spec = getattr(sampling_params, "patch", None)
    if not spec:
        return []
    # Source-store resolution is implemented in the source-store phase.
    return []
