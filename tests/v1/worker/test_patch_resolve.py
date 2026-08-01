# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side resolution of client-provided patch sources.

Covers the three new source kinds (``source_module`` incl. scaled entries and
the reserved ``zeros``, ``source_inline`` over all packed dtypes) and per-dim
masks (sparse ``indices`` and graded ``inline``), including the loud-not-fatal
resolution-failure backstop for unresolved sources. Resolution runs the TP1
path (``_tp_group`` unavailable on CPU -> world_size 1, purely local).
"""

from types import SimpleNamespace

import numpy as np
import pybase64 as base64
import torch

import vllm.v1.worker.gpu.patch_resolve as pr
from vllm.model_executor.layers.steering import SteeringHookPoint
from vllm.v1.worker.gpu.patch_resolve import (
    pop_resolution_failures,
    resolve_patch_entries,
)

POST_BLOCK = SteeringHookPoint.POST_BLOCK
LOCAL = frozenset({0, 1, 2, 3})


def _pack(array: np.ndarray, dtype: str) -> dict:
    if dtype == "bfloat16":
        t = torch.as_tensor(np.asarray(array, dtype=np.float32)).to(torch.bfloat16)
        raw = t.contiguous().view(torch.uint8).numpy().tobytes()
    else:
        np_dtype = {"float32": np.float32, "float16": np.float16}[dtype]
        raw = np.ascontiguousarray(array, dtype=np_dtype).tobytes()
    return {
        "dtype": dtype,
        "shape": [int(array.shape[0]), int(array.shape[1])],
        "data": base64.b64encode(raw).decode("ascii"),
    }


def _req(patch, patch_vectors=None, req_id="r0"):
    sp = SimpleNamespace(patch=patch, patch_vectors=patch_vectors)
    return SimpleNamespace(req_id=req_id, sampling_params=sp)


def _resolve(patch, *, module_registry=None, hidden_size=8, patch_vectors=None):
    pop_resolution_failures()  # clear the process-global registry
    return resolve_patch_entries(
        _req(patch, patch_vectors),
        local_layers=LOCAL,
        module_registry=module_registry,
        hidden_size=hidden_size,
    )


class TestModuleSource:
    def test_named_module_hit(self):
        reg = {"m": ({"post_block": {2: [1.0, 2.0, 3.0]}}, None, None)}
        entries = _resolve(
            [
                {
                    "layer": 2,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_module": "m",
                }
            ],
            module_registry=reg,
            hidden_size=3,
        )
        assert len(entries) == 1
        assert torch.allclose(entries[0].source, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(entries[0].alpha_row, torch.ones(3))

    def test_named_module_scaled_entry(self):
        reg = {
            "m": ({"post_block": {2: {"vector": [1.0, 2.0], "scale": 3.0}}}, None, None)
        }
        entries = _resolve(
            [
                {
                    "layer": 2,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_module": "m",
                }
            ],
            module_registry=reg,
            hidden_size=2,
        )
        assert torch.allclose(entries[0].source, torch.tensor([3.0, 6.0]))

    def test_wrong_width_module_row_records_failure_and_skips(self):
        # Registration validates finiteness but not length, so a wrong-width
        # row reaches resolution — it must loud-skip, not shape-crash staging.
        reg = {"m": ({"post_block": {2: [1.0, 2.0, 3.0]}}, None, None)}
        entries = _resolve(
            [
                {
                    "layer": 2,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_module": "m",
                }
            ],
            module_registry=reg,
            hidden_size=8,
        )
        assert entries == []
        failures = pop_resolution_failures()
        assert failures and "width 3 != hook width 8" in failures["r0"][0]

    def test_wrong_width_inline_table_records_failure_and_skips(self):
        table = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        entries = _resolve(
            [
                {
                    "layer": 2,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_inline": 0,
                }
            ],
            hidden_size=8,
            patch_vectors=_pack(table, "float32"),
        )
        assert entries == []
        failures = pop_resolution_failures()
        assert failures and "width 3 != hook width 8" in failures["r0"][0]

    def test_unknown_module_records_failure_and_skips(self):
        entries = _resolve(
            [
                {
                    "layer": 2,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_module": "ghost",
                }
            ],
            module_registry={},
        )
        assert entries == []
        failures = pop_resolution_failures()
        assert failures and "ghost" in failures["r0"][0]

    def test_zeros_builtin_no_registry(self):
        entries = _resolve(
            [
                {
                    "layer": 1,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_module": "zeros",
                }
            ],
            module_registry=None,
            hidden_size=6,
        )
        assert len(entries) == 1
        assert entries[0].source.shape == (6,)
        assert torch.all(entries[0].source == 0.0)


class TestInlineSource:
    def test_inline_all_dtypes(self):
        table = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        for dtype in ("float32", "float16", "bfloat16"):
            entries = _resolve(
                [
                    {
                        "layer": 0,
                        "hook": "post_block",
                        "dest_position": 0,
                        "source_inline": 1,
                    }
                ],
                patch_vectors=_pack(table, dtype),
                hidden_size=3,
            )
            assert len(entries) == 1, dtype
            assert torch.allclose(
                entries[0].source, torch.tensor([4.0, 5.0, 6.0]), atol=0.05
            ), dtype

    def test_inline_index_out_of_range(self):
        table = np.array([[1.0, 2.0]], dtype=np.float32)
        entries = _resolve(
            [
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_inline": 5,
                }
            ],
            patch_vectors=_pack(table, "float32"),
            hidden_size=2,
        )
        assert entries == []
        failures = pop_resolution_failures()
        assert "out of range" in failures["r0"][0]


class TestMaskFolding:
    def test_mask_indices_folds_into_alpha(self):
        reg = {"m": ({"post_block": {0: [1.0, 1.0, 1.0, 1.0]}}, None, None)}
        entries = _resolve(
            [
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_module": "m",
                    "alpha": 1.0,
                    "mask": {"indices": [0, 2]},
                }
            ],
            module_registry=reg,
            hidden_size=4,
        )
        assert torch.allclose(entries[0].alpha_row, torch.tensor([1.0, 0.0, 1.0, 0.0]))

    def test_alpha_times_mask(self):
        reg = {"m": ({"post_block": {0: [1.0, 1.0, 1.0]}}, None, None)}
        entries = _resolve(
            [
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_module": "m",
                    "alpha": 0.5,
                    "mask": {"indices": [1]},
                }
            ],
            module_registry=reg,
            hidden_size=3,
        )
        assert torch.allclose(entries[0].alpha_row, torch.tensor([0.0, 0.5, 0.0]))

    def test_mask_inline_graded_row(self):
        # Row 0 is the source, row 1 is a graded mask in [0, 1].
        table = np.array([[9.0, 9.0, 9.0], [0.0, 0.5, 1.0]], dtype=np.float32)
        entries = _resolve(
            [
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_inline": 0,
                    "alpha": 1.0,
                    "mask": {"inline": 1},
                }
            ],
            patch_vectors=_pack(table, "float32"),
            hidden_size=3,
        )
        assert torch.allclose(entries[0].alpha_row, torch.tensor([0.0, 0.5, 1.0]))


class TestBadPatchVectors:
    def test_bad_base64_records_failure(self):
        entries = _resolve(
            [
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_inline": 0,
                }
            ],
            patch_vectors={"dtype": "float32", "shape": [1, 2], "data": "!!!notb64"},
            hidden_size=2,
        )
        assert entries == []
        failures = pop_resolution_failures()
        assert failures  # decode failure + per-entry inline-unavailable


class TestTpGroupUnavailable:
    def test_cpu_has_no_tp_group(self):
        # Sanity: resolution takes the local (world_size 1) path on CPU.
        assert pr._tp_group() is None
