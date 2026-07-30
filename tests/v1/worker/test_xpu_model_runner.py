# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``vllm.v1.worker.xpu_model_runner`` (XPU worker / CUDA shims)."""

import subprocess
import sys
import textwrap

import pytest
import torch

# XPU-only: needs distinct torch.cuda vs torch.xpu current_stream symbols.
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not hasattr(torch.xpu, "current_stream"),
    reason="torch.xpu.current_stream is required",
)

# The body runs in a child interpreter because ``_torch_cuda_wrapper`` patches
# ``torch.cuda`` permanently and must not leak into the rest of the session.
#
# This used to use ``@pytest.mark.forked``, but ``pytest_forked``'s
# ``pytest_runtest_protocol`` hook is ``tryfirst`` and returns ``True``, so
# pytest's own protocol — and with it the parent's ``SetupState`` bookkeeping —
# never runs for the item. The parent is left holding stale ``Package``
# collectors, and the first test of the *next* package then dies on
# ``assert col in needed_collectors`` inside ``_pytest/runner.py`` with
# "previous item was not torn down properly". An explicit subprocess gives the
# same isolation without hijacking the protocol.
_CHILD_SCRIPT = textwrap.dedent(
    """
    from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

    from vllm.v1.worker.xpu_model_runner import _torch_cuda_wrapper

    # Same entry point as XPUModelRunner.__init__ (patches persist after exit).
    with _torch_cuda_wrapper():
        pass

    # Fresh handler table build, as on first torch.compile / AOT in the worker.
    # Registers torch.cuda.current_stream and torch.xpu.current_stream
    # separately; if they are the same object (pre-fix alias), this raises
    # "Handler already registered".
    TorchInGraphFunctionVariable._get_handlers.cache_clear()
    TorchInGraphFunctionVariable._get_handlers()
    """
)


def test_torch_cuda_wrapper_allows_dynamo_handler_registration() -> None:
    """Guard against XPU CUDA shim breaking Torch Dynamo during AOT compile.

    Before the fix, ``_torch_cuda_wrapper`` assigned
    ``torch.cuda.current_stream = torch.xpu.current_stream`` (same function object).
    On the first AOT/profile run, Dynamo builds its in-graph handler table and
    registers ``torch.cuda.current_stream`` and ``torch.xpu.current_stream``
    separately; duplicate identity triggers::

        AssertionError: Handler already registered for <function current_stream ...>

    That surfaced as EngineCore failing in ``profile_run`` / ``_get_handlers()``.
    The fix uses distinct shim callables so both can be registered.

    This test replays the post-init state (wrapper applied, patches left on
    ``torch.cuda``) and checks that Dynamo's real ``_get_handlers()`` succeeds.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "XPU CUDA shim broke Dynamo handler registration:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
