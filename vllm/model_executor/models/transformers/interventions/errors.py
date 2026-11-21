# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed errors for Transformers-backend interventions."""


class UnsupportedInterventionLayout(ValueError):
    """A Transformers decoder layer whose residual structure cannot be hooked.

    Raised at model build when steering, patching, or activation capture is
    configured and a decoder layer's forward does not match a residual layout
    the backend can instrument with native hook-point semantics. Failing here,
    rather than leaving the layer unhooked, keeps an intervention from being
    silently ignored.
    """

    def __init__(self, layer_cls: str, reason: str):
        self.layer_cls = layer_cls
        self.reason = reason
        super().__init__(
            f"Steering, patching, and activation capture are not supported for "
            f"{layer_cls} on the Transformers modeling backend: {reason}. Start "
            "the server without --enable-steering, --enable-patching, and "
            "--capture-consumers to serve this model."
        )
