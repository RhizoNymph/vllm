# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-level metadata channel.

:class:`RequestMetadata` carries per-request fields that are neither
sampling parameters nor token data — host-side metadata threaded from the
API entrypoint to the worker alongside the (vLLM-internal) request id. It
rides on :class:`vllm.v1.engine.EngineCoreRequest` next to
``external_req_id`` (the client request id) rather than on
:class:`vllm.sampling_params.SamplingParams`, because it does not influence
sampling.

The struct is the extensible home for this class of field: today it holds
the client conversation id; declarative steering specs will be added as
sibling fields. New fields must keep a default so older callers and
serialized payloads remain valid.
"""

from __future__ import annotations

import msgspec


class RequestMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    frozen=True,
):
    """Per-request host-side metadata, distinct from sampling parameters.

    Attributes:
        conversation_id: Optional client-supplied conversation grouping id.
            Surfaced on
            :class:`vllm.v1.capture.step_view.StepRequestView.conversation_id`
            so a sync capture/steering consumer can correlate successive
            requests of the same conversation (e.g. latch a steering decision
            after a trigger fires and re-apply it to later turns). Pure
            host-side string metadata: it costs no GPU work or D2H, so it is
            populated identically on the v1 and v2 runners. ``None`` when the
            request does not opt in.
    """

    conversation_id: str | None = None

    def is_empty(self) -> bool:
        """Return ``True`` when no metadata field is set."""
        return self.conversation_id is None
