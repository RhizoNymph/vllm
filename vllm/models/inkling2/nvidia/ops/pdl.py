# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Programmatic dependent launch (PDL) guards of the Inkling2 Triton kernels.

`griddepcontrol.wait` / `griddepcontrol.launch_dependents` are SM90+ PTX:
ptxas rejects them for SM8x targets. The kernels call `gdc_wait` /
`gdc_launch_dependents` from here, which compile to the PDL instructions
where `use_pdl()` holds (SM90 / SM100 / SM12x, unchanged) and to nothing on
SM8x, and launch with `launch_pdl=LAUNCH_PDL`. Without PDL every kernel
launch is fully stream-ordered, so dropping the waits is exact.
"""

from vllm.triton_utils import tl, triton

from . import use_pdl

# Evaluated once per process (NVML, no CUDA context); a Triton constexpr
# global, so it is part of each kernel's cache key.
PDL = tl.constexpr(use_pdl())
LAUNCH_PDL: bool = bool(use_pdl())


@triton.jit
def gdc_wait():
    if PDL:
        tl.extra.cuda.gdc_wait()


@triton.jit
def gdc_launch_dependents():
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
