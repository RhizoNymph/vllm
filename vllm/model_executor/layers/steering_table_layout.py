# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-space layout of the shared steering tables.

Every buffer family that rides the steering rows — the vector tables and
scales today; clamps and row monitors on adopting tiers — shares one row
space per layer::

    Row 0                    no-steering sentinel (always zeros)
    Row 1                    global prefill effective (base + prefill)
    Row 2                    global decode effective (base + decode)
    Rows 3 .. 3+C-1          scheduler-admitted per-request config pool
    Rows 3+C .. 3+C+D-1      dynamic-override pool (runtime-allocated)

Kernels gather rows through the shared ``steering_index`` token->row
buffer, so every family sized against this space must be congruent — a
mismatch fails as silent garbage gathers, not as an error. This module is
the single definition of the arithmetic. The composed tables the manager
uploads keep the same three reserved rows and then pack only the ACTIVE
config/dynamic rows, so the reserved-row constants apply to both the full
buffers and the packed tables; the capacity ranges apply to the full
buffers and the row allocators.

Import-light by design: sits below ``steering.py`` (layer side) and the
worker control plane, like the other intervention scaffolding modules.
"""

from __future__ import annotations

from dataclasses import dataclass

SENTINEL_ROW = 0
GLOBAL_PREFILL_ROW = 1
GLOBAL_DECODE_ROW = 2
NUM_RESERVED_ROWS = 3


def global_row_for_phase(is_prefill: bool) -> int:
    """The reserved global-effective row a no-config request routes to."""
    return GLOBAL_PREFILL_ROW if is_prefill else GLOBAL_DECODE_ROW


@dataclass(frozen=True)
class TableLayout:
    """Row-space capacity split of the full steering table buffers.

    ``num_configs`` is the scheduler-admitted per-request pool size
    (``max_steering_configs``); ``num_dynamic`` the dynamic-override pool
    (``max_dynamic_steering_configs``). Note the layer-side registration
    entry points take the COMBINED pool (:attr:`pool_rows`) as their
    ``max_steering_configs`` argument — models size buffers through
    ``get_steering_buffer_config`` and never see the split.
    """

    num_configs: int
    num_dynamic: int = 0

    @classmethod
    def from_steering_config(cls, steering_config) -> TableLayout:
        """Build from a ``SteeringConfig``-shaped object.

        ``max_dynamic_steering_configs`` is read tolerantly: configs
        predating the dynamic pool simply get an empty one.
        """
        return cls(
            num_configs=steering_config.max_steering_configs,
            num_dynamic=getattr(steering_config, "max_dynamic_steering_configs", 0),
        )

    @property
    def pool_rows(self) -> int:
        """Combined row budget above the reserved rows (static + dynamic)."""
        return self.num_configs + self.num_dynamic

    @property
    def num_rows(self) -> int:
        """Total rows in every full-capacity buffer of this row space."""
        return NUM_RESERVED_ROWS + self.num_configs + self.num_dynamic

    @property
    def config_rows(self) -> range:
        """Rows of the static per-request pool."""
        return range(NUM_RESERVED_ROWS, NUM_RESERVED_ROWS + self.num_configs)

    @property
    def dynamic_rows(self) -> range:
        """Rows of the dynamic-override pool (directly above the static)."""
        first = NUM_RESERVED_ROWS + self.num_configs
        return range(first, first + self.num_dynamic)
