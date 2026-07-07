# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture consumer that parks clean-run activations as patch sources.

A clean run captures its residuals through the normal capture pipeline with
``capture={"patch_source": {...}}``; this consumer writes each captured row into
the process-local :class:`PatchSourceStore` under a client-chosen run handle,
keyed by ``(run_id, layer, hook, position)``. Patch requests later reference
those vectors by ``(source_run, layer, hook, source_position)``.

Reuses the entire capture pipeline unchanged (graph-safe taps, CPU offload,
chunked-prefill aggregation, prefix-cache interplay). Only the sink is new.

Row -> position recovery: ``on_capture`` receives the concatenated rows for a
``(req, layer, hook)`` key in captured-position order, so the per-request
position list recorded at ``validate_client_spec`` time maps row ``i`` to its
prompt position. Source runs capture prompt positions only (``all_prompt`` /
``last_prompt`` / explicit list); generated-only / ``all`` selectors are
rejected.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.steering import HOOK_POINT_TABLE_ATTR
from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.errors import CaptureValidationError
from vllm.v1.capture.source_store import get_active_patch_source_store
from vllm.v1.capture.types import (
    CaptureContext,
    CaptureKey,
    CaptureSpec,
    captured_prompt_positions,
)

logger = init_logger(__name__)

# Injection reuses the steering apply sites, which only exist for these three
# hook points — so a patch source may only capture these (capturing others
# would store vectors that can never be injected).
_INJECTABLE_HOOKS: frozenset[str] = frozenset(h.value for h in HOOK_POINT_TABLE_ATTR)

# Bound the per-request validate-time bookkeeping so a long-lived server doesn't
# accumulate state unboundedly (each entry is tiny: run handle + position list).
_MAX_TRACKED_REQUESTS = 4096


class PatchSourceConsumer(CaptureConsumer):
    """Writes captured clean-run rows into the per-worker patch source store."""

    location = "worker"
    reads_client_spec = True

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        # req_id -> (run_override, prompt_positions, num_prompt_tokens)
        self._req_state: OrderedDict[str, tuple[str | None, list[int], int]] = (
            OrderedDict()
        )

    def validate_client_spec(self, raw_spec: Any, ctx: CaptureContext) -> CaptureSpec:
        if not isinstance(raw_spec, dict):
            raise CaptureValidationError(
                f"patch_source spec must be a dict, got {type(raw_spec).__name__}"
            )
        run = raw_spec.get("run")
        if run is not None and not isinstance(run, str):
            raise CaptureValidationError("patch_source 'run' must be a string")

        positions = raw_spec.get("positions", "all_prompt")
        if positions in ("all", "all_generated"):
            raise CaptureValidationError(
                "patch_source captures prompt positions only; use "
                "'all_prompt', 'last_prompt', or an explicit position list"
            )

        hooks_raw = raw_spec.get("hooks")
        if not hooks_raw or not isinstance(hooks_raw, dict):
            raise CaptureValidationError(
                "patch_source spec needs a non-empty 'hooks' dict {hook: layers}"
            )
        hooks: dict[str, list[int]] = {}
        for hook, layers in hooks_raw.items():
            if hook not in _INJECTABLE_HOOKS:
                raise CaptureValidationError(
                    f"hook {hook!r} is not injectable; valid: "
                    f"{sorted(_INJECTABLE_HOOKS)}"
                )
            if layers == "all":
                resolved = list(range(ctx.num_hidden_layers))
            elif isinstance(layers, (list, tuple)):
                resolved = [int(layer) for layer in layers]
            else:
                raise CaptureValidationError(
                    f"hook {hook!r} layers must be 'all' or a list, got "
                    f"{type(layers).__name__}"
                )
            for layer in resolved:
                if not (0 <= layer < ctx.num_hidden_layers):
                    raise CaptureValidationError(
                        f"layer {layer} out of range [0, {ctx.num_hidden_layers})"
                    )
            hooks[hook] = resolved

        spec = CaptureSpec(hooks=hooks, positions=positions)  # type: ignore[arg-type]

        prompt_positions = captured_prompt_positions(spec, ctx.num_prompt_tokens)
        if not prompt_positions:
            raise CaptureValidationError(
                "patch_source spec captures no prompt positions"
            )
        self._req_state[ctx.vllm_internal_request_id] = (
            run,
            prompt_positions,
            ctx.num_prompt_tokens,
        )
        self._req_state.move_to_end(ctx.vllm_internal_request_id)
        while len(self._req_state) > _MAX_TRACKED_REQUESTS:
            self._req_state.popitem(last=False)
        return spec

    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        store = get_active_patch_source_store()
        if store is None:
            logger.warning(
                "patch_source on_capture with no active source store "
                "(patch_source_cache_bytes not set?); dropping rows"
            )
            return
        req_id, layer, hook = key
        state = self._req_state.get(req_id)
        if state is None:
            return
        run_override, positions, num_prompt = state
        run_id = run_override or sidecar.get("client_request_id") or req_id

        n = min(int(tensor.shape[0]), len(positions))
        for i in range(n):
            store.put_row(
                run_id,
                int(layer),
                str(hook),
                positions[i],
                tensor[i],
                num_prompt_tokens=num_prompt,
            )
