# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed logical row-owner keys for steering runtime state.

Steering table rows have three logical owner kinds:

* ``Global`` — the global prefill/decode tier rows (rows 1/2), keyed by phase.
* ``Config`` — an admitted per-request config row, keyed by ``(config_hash,
  phase)`` and reference counted.
* ``Dyn`` — a dynamic-override row, keyed by a monotonic ``dyn_id``.

Runtime state that must survive row reassignment (per-row strength scales and
per-row monitors) is keyed by :class:`RowOwner` rather than by ad-hoc tuples,
so a single :meth:`SteeringManager._purge_owner` can drop every store for an
owner in one place (closing the "forgot one namespace on release" leak).

``RowOwner`` is frozen (hashable, usable as a dict key) and totally ordered so
signature folds and iteration are independent of dict insertion order — a free
determinism guard on top of the rank-replicated action sequence.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class RowOwner:
    """A frozen, totally-ordered logical owner of a steering table row.

    Construct via :meth:`global_`, :meth:`config`, and :meth:`dyn` rather than
    the raw constructor. The field layout is chosen so the dataclass-generated
    ``order=True`` comparison (``kind`` first) yields a stable total order
    across all three variants regardless of the unused fields.
    """

    kind: str
    phase: str = ""
    config_hash: int = 0
    dyn_id: int = 0

    @classmethod
    def global_(cls, phase: str) -> "RowOwner":
        """Owner of a global tier row (row 1 prefill / row 2 decode)."""
        return cls(kind="global", phase=phase)

    @classmethod
    def config(cls, config_hash: int, phase: str) -> "RowOwner":
        """Owner of an admitted per-request config row."""
        return cls(kind="config", phase=phase, config_hash=int(config_hash))

    @classmethod
    def dyn(cls, dyn_id: int) -> "RowOwner":
        """Owner of a dynamic-override row."""
        return cls(kind="dyn", dyn_id=int(dyn_id))

    @property
    def legacy_key(self) -> tuple:
        """The historical raw-tuple key, for wire / observability parity.

        ``("global", phase)`` / ``("config", config_hash, phase)`` /
        ``("dyn", dyn_id)`` — the shapes the ``/v1/steering/dynamic`` status
        payload has always exposed.
        """
        if self.kind == "global":
            return ("global", self.phase)
        if self.kind == "config":
            return ("config", self.config_hash, self.phase)
        return ("dyn", self.dyn_id)


@dataclass(frozen=True)
class OwnerStore:
    """One owner-keyed runtime store, described uniformly for purge + tests.

    Listed together in :meth:`SteeringManager._owner_stores` so
    :meth:`SteeringManager._purge_owner` drops every store in one place and a
    parametrized test can assert each is purged — a future owner-keyed store
    added to the registry without a working ``purge`` fails that test.
    """

    name: str
    contains: Callable[[RowOwner], bool]
    purge: Callable[[RowOwner], bool]
    install_dummy: Callable[[RowOwner], None]
