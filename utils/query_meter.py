"""Measure black-box cost on one unit: marginal calls on rows not already tabulated.

Why a new unit
--------------
The previous counters were not comparable across methods. Every RL figure was an
exact closed form in the split sizes -- MADA `n_training_queries` = |train|+|val|,
RLDA = K x that, MADA extraction = (12K+3) x |val| with zero residual -- constant
across seeds and independent of the frame budget, so they measured no work that
actually happened. Meanwhile CART was billed a single pass for scoring its leaves
on validation while the RL arms were billed 12-51 re-reads of a cached table for
the identical operation, and `anchor-exp` was billed its real perturbation calls.

The only unit that is fair to all four is:

    a call is MARGINAL if it evaluates rows that are not already in the
    prediction table; otherwise it is a (re)read of the table.

The table itself -- f_hat on the fixed reference splits -- is a one-time shared
setup cost, identical for every method, and is reported separately rather than
folded into per-method totals.

How the classification is measured, not assumed
-----------------------------------------------
`SimpleClassifier.forward` is wrapped at class level, so every instance and every
call path is seen. Reference matrices (the splits) are registered up front by
content hash. A call whose input matches a registered matrix is a table build the
first time it is seen and a table re-read afterwards; anything else -- a
perturbation batch, a synthetic sample -- is marginal. Sub-batches of a split
are recognised **row by row** (`row_level=True`, the default): a call on the rows
inside a candidate box is a table read, not a new query, because every one of
those rows is already tabulated. Only rows whose exact values appear in no
registered split are charged as marginal.

    from utils.query_meter import QueryMeter
    with QueryMeter(reference=[X_train_std, X_val_std, X_test_std]) as m:
        ...
    m.report()
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _fingerprint(x: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    return f"{a.shape}:{hashlib.sha1(a).hexdigest()}"


def _row_hashes(x: np.ndarray) -> List[bytes]:
    """Per-row identity, so a filtered sub-batch of a split is still recognised.

    Without this a call on the rows inside a candidate box counts as marginal
    even though every one of those rows is already tabulated -- which inflated
    MADA from ~0 to 467 'marginal' rows per explanation on housing.
    """
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if a.ndim != 2:
        return []
    return [r.tobytes() for r in a]


class QueryMeter:
    """Split classifier calls into table builds, table re-reads, and marginal calls."""

    def __init__(self, reference: Optional[Iterable[np.ndarray]] = None,
                 row_level: bool = True) -> None:
        self._ref: Dict[str, int] = {}
        self._row_level = bool(row_level)
        self._ref_rows: set = set()
        for arr in reference or []:
            if arr is None:
                continue
            a = np.asarray(arr)
            self._ref[_fingerprint(a)] = int(a.shape[0])
            if self._row_level:
                self._ref_rows.update(_row_hashes(a))
        self._seen_tables: Dict[str, int] = {}
        self.table_rows_unique = 0     # one-time shared setup
        self.table_reads = 0           # repeat reads of an already-built table
        self.table_read_rows = 0
        self.marginal_rows = 0         # the number that matters per method
        self.marginal_calls = 0
        self._orig = None

    def register(self, *arrays: np.ndarray) -> "QueryMeter":
        for arr in arrays:
            if arr is None:
                continue
            a = np.asarray(arr)
            self._ref[_fingerprint(a)] = int(a.shape[0])
        return self

    def _note(self, x) -> None:
        try:
            arr = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        except Exception:
            return
        if arr.ndim != 2:
            self.marginal_rows += int(arr.shape[0]) if arr.ndim else 0
            self.marginal_calls += 1
            return
        fp = _fingerprint(arr)
        if fp in self._ref:
            if fp in self._seen_tables:
                self.table_reads += 1
                self.table_read_rows += int(arr.shape[0])
            else:
                self._seen_tables[fp] = int(arr.shape[0])
                self.table_rows_unique += int(arr.shape[0])
            return
        if self._row_level and self._ref_rows:
            hashes = _row_hashes(arr)
            tabulated = sum(1 for h in hashes if h in self._ref_rows)
            fresh = len(hashes) - tabulated
            if tabulated:
                self.table_reads += 1
                self.table_read_rows += tabulated
            if fresh:
                self.marginal_rows += fresh
                self.marginal_calls += 1
            return
        self.marginal_rows += int(arr.shape[0])
        self.marginal_calls += 1

    def __enter__(self) -> "QueryMeter":
        from utils.networks import SimpleClassifier

        self._orig = SimpleClassifier.forward
        meter = self

        def metered_forward(self, x, *a, **kw):  # noqa: ANN001
            meter._note(x)
            return meter._orig(self, x, *a, **kw)

        SimpleClassifier.forward = metered_forward
        global _ACTIVE
        if _ACTIVE is None:
            _ACTIVE = self
        return self

    def __exit__(self, *exc) -> bool:
        from utils.networks import SimpleClassifier

        if self._orig is not None:
            SimpleClassifier.forward = self._orig
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit": (
                "marginal classifier calls on rows not already in the prediction "
                "table; the table build is a shared one-time setup cost and is "
                "reported separately"
            ),
            "table_rows_unique": int(self.table_rows_unique),
            "table_reads": int(self.table_reads),
            "table_read_rows": int(self.table_read_rows),
            "marginal_rows": int(self.marginal_rows),
            "marginal_calls": int(self.marginal_calls),
        }

    def report(self) -> str:
        d = self.to_dict()
        return (
            f"table build {d['table_rows_unique']:,} rows (one-time, shared) | "
            f"table re-reads {d['table_reads']:,} calls / "
            f"{d['table_read_rows']:,} rows (free) | "
            f"MARGINAL {d['marginal_rows']:,} rows in {d['marginal_calls']:,} calls"
        )


# ---------------------------------------------------------------------------
# Process-level hook so a pipeline stage can record the unit without threading
# the meter through every call site.
# ---------------------------------------------------------------------------

_ACTIVE: Optional["QueryMeter"] = None


def active_meter() -> Optional["QueryMeter"]:
    """The meter currently recording, if any."""
    return _ACTIVE


def start_meter_for(loader) -> Optional["QueryMeter"]:
    """Begin metering against a loader's splits, unless one is already running.

    Returns None when a meter is already active, so an outer harness (for example
    `revision/measure_marginal_cost.py`) keeps ownership and the inner call does
    not double-count.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        return None
    refs = []
    for name in ("X_train_scaled", "X_val_scaled", "X_test_scaled"):
        arr = getattr(loader, name, None)
        if arr is not None:
            refs.append(np.asarray(arr, dtype=np.float32))
    meter = QueryMeter(refs)
    meter.__enter__()
    return meter


def stop_meter(meter: Optional["QueryMeter"]) -> None:
    if meter is not None:
        meter.__exit__(None, None, None)
