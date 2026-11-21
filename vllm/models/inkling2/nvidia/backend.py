# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which implementation the Inkling2 decoder layers use.

- `NATIVE`: the upstream kernels (NVFP4 / MXFP8 experts, fused `qkvr`,
  unquantized embeddings), for SM90 / SM100 / SM12x with the checkpoint's
  `Inkling2QuantConfig` or no quantization config. Unchanged from upstream.
- `QUANT_METHOD`: the routed experts are a standard `RoutedExperts` and the
  attention / embedding linears take the quantization config, so any vLLM
  quantization method (e.g. the GGUF plugin's) supplies the kernels. Used for
  every non-Inkling2 quantization config, and on SM8x (no FP4 / FP8 tensor
  cores) also without one (the BF16 reference).

See `notes/inkling2-expert-interface.md` for the contract a quantization
method implements.
"""

import enum
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from ..quant_config import Inkling2QuantConfig
from .ops import is_sm8x


class Inkling2Backend(enum.Enum):
    NATIVE = "native"
    QUANT_METHOD = "quant_method"


class Inkling2UnsupportedError(ValueError):
    """A configuration the Inkling2 kernels cannot run (raised at model
    construction, before any weight is loaded)."""


@dataclass(frozen=True)
class Inkling2LayerQuant:
    """The backend of the decoder layers and the configs it hands out.

    `native` is the checkpoint's precision metadata (NATIVE only) and
    `linear` the config passed to the linears, embeddings and routed experts
    (QUANT_METHOD only; None there means unquantized BF16).
    """

    backend: Inkling2Backend
    native: Inkling2QuantConfig | None = None
    linear: QuantizationConfig | None = None

    @property
    def uses_quant_method(self) -> bool:
        return self.backend is Inkling2Backend.QUANT_METHOD

    @property
    def split_attention_linears(self) -> bool:
        """Whether attention projects q-side / k / v with three quantizable
        linears instead of slicing one fused BF16 `qkvr` weight."""
        return self.uses_quant_method and self.linear is not None


def select_inkling2_backend(
    quant_config: QuantizationConfig | None,
    tp_size: int = 1,
    sm8x: bool | None = None,
) -> Inkling2LayerQuant:
    """Pick the backend for `quant_config` on this device (`sm8x` overrides
    the device check, for tests)."""
    if sm8x is None:
        sm8x = is_sm8x()
    if isinstance(quant_config, Inkling2QuantConfig):
        if sm8x:
            raise Inkling2UnsupportedError(
                "The Inkling2 NVFP4 / MXFP8 checkpoints need SM90 or newer "
                "(FP4 / FP8 tensor cores); on SM8x serve a GGUF or BF16 "
                "export instead"
            )
        return Inkling2LayerQuant(Inkling2Backend.NATIVE, native=quant_config)
    if quant_config is None and not sm8x:
        return Inkling2LayerQuant(Inkling2Backend.NATIVE)
    if tp_size > 1:
        name = "unquantized" if quant_config is None else quant_config.get_name()
        raise Inkling2UnsupportedError(
            f"Inkling2 with {name} weights runs at tensor_parallel_size=1 only "
            "(the fused all-reduce kernels need SM90+ and NVFP4 / MXFP8 "
            f"experts); got tensor_parallel_size={tp_size}"
        )
    return Inkling2LayerQuant(Inkling2Backend.QUANT_METHOD, linear=quant_config)


def cast_dense_weight(param: nn.Parameter, weight: torch.Tensor) -> torch.Tensor:
    """`weight` in `param`'s dtype if both are dense floating tensors (a
    `QUANT_METHOD` checkpoint may store norms or the router in F32);
    quantized tensors and lazily materialized parameters are returned
    unchanged for their quant method's loader."""
    if (
        isinstance(param, UninitializedParameter)
        or not param.dtype.is_floating_point
        or not weight.dtype.is_floating_point
        or param.dtype == weight.dtype
    ):
        return weight
    return weight.to(param.dtype)
