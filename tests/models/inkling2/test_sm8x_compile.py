# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM8x-path Triton kernels compile for SM86 / SM89 within their
shared-memory limit (no GPU needed; see `sm8x_compile_harness.py`).

Upstream's kernels issue `griddepcontrol` (SM90+ PTX) unconditionally and
take TMA paths for large batches, so on SM86 most of them fail in ptxas;
this pins the PDL / TMA guards of `vllm/models/inkling2/nvidia/ops`.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).with_name("sm8x_compile_harness.py")
REPO = HARNESS.parents[3]

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("triton") is None, reason="needs Triton (ptxas)"
    ),
    pytest.mark.skip_global_cleanup,
]


@pytest.mark.parametrize("arch", [86, 89])
@pytest.mark.parametrize(
    "selection",
    [
        "k_group",
        "select_experts",
        "rms_norm",
        "attn_res",
        "short_conv",
        "attention_prologue",
        "rel_attention",
        "decode_attention",
        "gate_sink",
        "log_scaling",
        "dspark",
        "vision",
    ],
)
def test_sm8x_kernels_compile(arch, selection):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO), *filter(None, [env.get("PYTHONPATH")])]
    )
    env.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
    result = subprocess.run(
        [sys.executable, str(HARNESS), "--arch", str(arch), "-k", selection],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = [line for line in result.stdout.splitlines() if line]
    assert result.returncode == 0, "\n".join(
        [*(line for line in lines if line.startswith("FAIL")), result.stderr[-4000:]]
    )
    assert lines and lines[-1].endswith(f"cases compiled for sm_{arch}")
    assert not lines[-1].startswith("0/")
