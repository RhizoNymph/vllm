# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the pure activation-patching sidecar launcher helpers."""

import argparse

import pytest

from vllm.entrypoints.cli.serve import (
    build_patch_sidecar_args,
    patch_sidecar_url,
    should_spawn_patch_sidecar,
)


@pytest.mark.parametrize(
    ("rust_path", "patch_enabled", "env_enabled", "expected"),
    [
        ("/usr/bin/vllm-rs", True, True, True),
        (None, True, True, False),
        ("", True, True, False),
        ("/usr/bin/vllm-rs", False, True, False),
        ("/usr/bin/vllm-rs", True, False, False),
    ],
)
def test_should_spawn_patch_sidecar(rust_path, patch_enabled, env_enabled, expected):
    assert should_spawn_patch_sidecar(rust_path, patch_enabled, env_enabled) is expected


def test_patch_sidecar_url():
    assert patch_sidecar_url("127.0.0.1", 9123) == "http://127.0.0.1:9123"


def test_build_patch_sidecar_args_overrides_listener_and_clears_tls():
    args = argparse.Namespace(
        host="0.0.0.0",
        port=8000,
        uds="/tmp/public.sock",
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem",
        ssl_ca_certs="ca.pem",
        model="Qwen/Qwen3-0.6B",
    )
    sidecar = build_patch_sidecar_args(args, "127.0.0.1", 9123)

    assert sidecar.host == "127.0.0.1"
    assert sidecar.port == 9123
    assert sidecar.uds is None
    assert sidecar.ssl_keyfile is None
    assert sidecar.ssl_certfile is None
    assert sidecar.ssl_ca_certs is None
    # Other args are carried through unchanged.
    assert sidecar.model == "Qwen/Qwen3-0.6B"
    # The original namespace is not mutated.
    assert args.host == "0.0.0.0"
    assert args.port == 8000
    assert args.uds == "/tmp/public.sock"
