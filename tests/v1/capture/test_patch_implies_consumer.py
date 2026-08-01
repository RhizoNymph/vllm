# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--enable-patching`` implies the ``patch_source`` capture consumer.

The implication lives in ``VllmConfig.__post_init__`` — the config-finalization
point shared by the online server and the offline ``LLM`` construction paths —
so both get it and neither has to pass ``--capture-consumers patch_source``
explicitly. These tests drive that shared path directly.
"""

from __future__ import annotations

from vllm.config.device import DeviceConfig
from vllm.config.patch import PatchConfig
from vllm.config.vllm import VllmConfig
from vllm.v1.capture.config import CaptureConsumersConfig, CaptureConsumerSpec

PATCH_SOURCE = "patch_source"


def _cpu_config(**kwargs) -> VllmConfig:
    """Build a minimal ``VllmConfig`` that runs ``__post_init__`` off-GPU."""
    return VllmConfig(device_config=DeviceConfig(device="cpu"), **kwargs)


def _consumer_names(cfg: VllmConfig) -> list[str] | None:
    ccc = cfg.capture_consumers_config
    return None if ccc is None else [spec.name for spec in ccc.consumers]


def test_enable_patching_implies_patch_source() -> None:
    """Patching on with no capture config materializes ``patch_source``."""
    cfg = _cpu_config(patch_config=PatchConfig())
    assert _consumer_names(cfg) == [PATCH_SOURCE]


def test_implied_consumer_appended_alongside_others() -> None:
    """The implied consumer joins an existing consumer list."""
    cfg = _cpu_config(
        patch_config=PatchConfig(),
        capture_consumers_config=CaptureConsumersConfig(
            consumers=[CaptureConsumerSpec(name="filesystem")]
        ),
    )
    assert _consumer_names(cfg) == ["filesystem", PATCH_SOURCE]


def test_explicit_patch_source_not_duplicated() -> None:
    """An explicit ``patch_source`` is preserved without duplication."""
    cfg = _cpu_config(
        patch_config=PatchConfig(),
        capture_consumers_config=CaptureConsumersConfig(
            consumers=[CaptureConsumerSpec(name=PATCH_SOURCE)]
        ),
    )
    assert _consumer_names(cfg) == [PATCH_SOURCE]


def test_disabled_patching_never_adds_consumer() -> None:
    """With patching off, ``patch_source`` is never implied."""
    cfg = _cpu_config(
        patch_config=None,
        capture_consumers_config=CaptureConsumersConfig(
            consumers=[CaptureConsumerSpec(name="filesystem")]
        ),
    )
    assert _consumer_names(cfg) == ["filesystem"]


def test_disabled_patching_leaves_capture_config_absent() -> None:
    """Patching off with no capture config keeps it ``None`` (no store)."""
    cfg = _cpu_config(patch_config=None)
    assert cfg.capture_consumers_config is None


def test_implied_spec_matches_explicit_spec() -> None:
    """The implied spec is identical to the one parsed from the CLI flag.

    The offline ``LLM`` path historically dropped newly-added capture knobs,
    so this pins that the implied registration flows through the same spec
    shape as ``--capture-consumers patch_source`` — no bespoke path.
    """
    from vllm.v1.capture.config import parse_consumer_spec

    cfg = _cpu_config(patch_config=PatchConfig())
    implied = cfg.capture_consumers_config.consumers[0]
    explicit = parse_consumer_spec(PATCH_SOURCE)
    assert implied.name == explicit.name
    assert implied.instance_name == explicit.instance_name
    assert implied.params == explicit.params
