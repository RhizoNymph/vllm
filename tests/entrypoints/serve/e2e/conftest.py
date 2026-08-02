# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One shared live server for the interpretability e2e suite.

Server startup dominates suite runtime, so every file shares a single
``vllm serve`` process carrying all the interpretability flags at once:
steering (+ SAE registration), patching (implies the ``patch_source``
consumer), and a filesystem capture consumer rooted in a session
tmpdir. The server runs in the default compiled/cudagraph mode, so SAE
topology is frozen at load — spare delta slots are reserved at the test
site so the SAE file exercises hot registration on a compiled engine.
Tests must restore any global state they mutate (global steering/SAE
tiers, registered modules) so files stay order-independent.
"""

from __future__ import annotations

import httpx
import pytest

from tests.entrypoints.serve.e2e.helpers import HOOK, LAYER, MODEL
from tests.utils import RemoteOpenAIServer


@pytest.fixture(scope="session")
def capture_root(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("capture"))


@pytest.fixture(scope="session")
def server(capture_root: str):
    args = [
        "--max-model-len",
        "512",
        "--gpu-memory-utilization",
        "0.75",
        "--enable-steering",
        "--max-steering-configs",
        "8",
        "--enable-patching",
        "--capture-consumers",
        f"filesystem:root={capture_root}",
        "--sae-spare-slot-sites",
        f"{LAYER}:{HOOK}",
        "--sae-spare-slots-per-site",
        "2",
        "--sae-spare-slot-features",
        "8",
    ]
    with RemoteOpenAIServer(MODEL, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="session")
def http(server):
    with httpx.Client(base_url=server.url_root, timeout=300.0) as client:
        yield client
