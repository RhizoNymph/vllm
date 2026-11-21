# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steering, patching, capture, clamps, and SAE steering for Transformers-backed
decoder layers.

See docs/features/transformers_backend_interventions.md.
"""

from vllm.model_executor.models.transformers.interventions.analysis import (
    analyze_layer,
    is_vllm_attention,
)
from vllm.model_executor.models.transformers.interventions.errors import (
    UnsupportedInterventionLayout,
)
from vllm.model_executor.models.transformers.interventions.install import (
    BufferSizes,
    hooked_forward,
    install_interventions,
    interventions_requested,
)
from vllm.model_executor.models.transformers.interventions.plan import (
    LayerPlan,
    MlpInputSite,
    ParallelResidual,
    ResidualLayout,
    SequentialResidual,
)
from vllm.model_executor.models.transformers.interventions.rewrite import (
    HOOKS_NAME,
    rewrite_forward,
)

__all__ = [
    "HOOKS_NAME",
    "BufferSizes",
    "LayerPlan",
    "MlpInputSite",
    "ParallelResidual",
    "ResidualLayout",
    "SequentialResidual",
    "UnsupportedInterventionLayout",
    "analyze_layer",
    "hooked_forward",
    "install_interventions",
    "interventions_requested",
    "is_vllm_attention",
    "rewrite_forward",
]
