# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-family predicates of the Inkling2 kernels.

Each kernel family keys off one predicate here, so the SM90 / SM100 / SM12x
paths stay as upstream and SM8x (Ampere / Ada) only differs where a
predicate says so:

- `use_sm100_kernels`: tml_fa4, tcgen05 attention, TRT-LLM-gen experts.
- `use_native_conv_mix`: the CUDA conv/mix kernels of `csrc/inkling2`, built
  for 9.0a / 10.x only (SM12x and SM8x run the Triton short conv + mix).
- `use_tma`: TMA tensor-descriptor Triton kernels (SM90 / SM100).
- `use_pdl`: programmatic dependent launch (`griddepcontrol`, SM90+).
- `is_sm8x`: Ampere / Ada, which run the quant-method experts
  (`backend.py`) and never the NVFP4 / MXFP8 kernels.
"""

import functools


def _family(capability: int) -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_device_capability_family(capability)


@functools.cache
def use_sm100_kernels() -> bool:
    """Whether to run the Blackwell kernels (tml_fa4, tcgen05 attention,
    TRT-LLM-gen expert GEMMs, e2m1 conversions). Hopper uses Humming experts;
    SM12x uses Triton experts and does not support tcgen05."""
    return _family(100)


@functools.cache
def use_native_conv_mix() -> bool:
    """Whether `torch.ops._C.inkling2_conv_mix` (and the fused all-reduce
    conv/mix) exist: `csrc/inkling2` is built for 9.0a and 10.x only."""
    return _family(90) or _family(100)


@functools.cache
def use_tma() -> bool:
    """Whether the TMA (tensor descriptor) Triton kernels may run."""
    return _family(90) or _family(100)


@functools.cache
def use_pdl() -> bool:
    """Programmatic dependent launch (`griddepcontrol`) needs SM90+."""
    from vllm.platforms import current_platform

    return current_platform.has_device_capability(90)


@functools.cache
def is_sm8x() -> bool:
    """Ampere / Ada (SM80, SM86, SM87, SM89)."""
    return _family(80)
