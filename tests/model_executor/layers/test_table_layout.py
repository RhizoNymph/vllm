# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TableLayout — the single definition of the steering row space.

Congruence tests pin the layout to the arithmetic historically inlined at
the registration / row-allocator / warmup sites, so a layout change that
would silently shift any consumer fails here first.
"""

from types import SimpleNamespace

from vllm.model_executor.layers.steering_table_layout import (
    GLOBAL_DECODE_ROW,
    GLOBAL_PREFILL_ROW,
    NUM_RESERVED_ROWS,
    SENTINEL_ROW,
    TableLayout,
    global_row_for_phase,
)


def test_reserved_row_constants():
    assert SENTINEL_ROW == 0
    assert GLOBAL_PREFILL_ROW == 1
    assert GLOBAL_DECODE_ROW == 2
    assert NUM_RESERVED_ROWS == 3


def test_global_row_for_phase():
    assert global_row_for_phase(is_prefill=True) == GLOBAL_PREFILL_ROW
    assert global_row_for_phase(is_prefill=False) == GLOBAL_DECODE_ROW


def test_row_partition_is_contiguous_and_disjoint():
    layout = TableLayout(num_configs=8, num_dynamic=4)
    reserved = range(NUM_RESERVED_ROWS)
    all_rows = [*reserved, *layout.config_rows, *layout.dynamic_rows]
    assert all_rows == list(range(layout.num_rows))


def test_num_rows_matches_legacy_arithmetic():
    # register_steering_buffers sized tables as ``combined_pool + 3``;
    # the runner warmup as ``static + dynamic + 3``.
    layout = TableLayout(num_configs=8, num_dynamic=4)
    assert layout.pool_rows == 8 + 4
    assert layout.num_rows == 8 + 4 + 3


def test_free_row_pools_match_legacy_manager_init():
    # SteeringManager seeded its allocators as descending lists so pop()
    # hands out the lowest row first:
    #   static:  range(C + 2, 2, -1)
    #   dynamic: range(C + 2 + D, C + 2, -1)
    c, d = 8, 4
    layout = TableLayout(num_configs=c, num_dynamic=d)
    assert list(reversed(layout.config_rows)) == list(range(c + 2, 2, -1))
    assert list(reversed(layout.dynamic_rows)) == list(range(c + 2 + d, c + 2, -1))


def test_from_steering_config():
    layout = TableLayout.from_steering_config(
        SimpleNamespace(max_steering_configs=8, max_dynamic_steering_configs=4)
    )
    assert layout == TableLayout(num_configs=8, num_dynamic=4)


def test_from_steering_config_without_dynamic_pool_attr():
    layout = TableLayout.from_steering_config(SimpleNamespace(max_steering_configs=8))
    assert layout == TableLayout(num_configs=8, num_dynamic=0)
    assert layout.dynamic_rows == range(11, 11)


def test_empty_pools():
    layout = TableLayout(num_configs=0, num_dynamic=0)
    assert layout.num_rows == NUM_RESERVED_ROWS
    assert list(layout.config_rows) == []
    assert list(layout.dynamic_rows) == []
    assert layout.pool_rows == 0
