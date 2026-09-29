"""Print a box as the rule a reader sees, showing exactly the conditions Len counts.

A condition is a feature with a face that excludes at least one D_train row
(`utils.metrics.LEN_CRITERION`); only those faces are printed, so the printed
length equals Len for every method. The decision is taken in the space the box is
scored in (unit for the RL arms and random search, original units for CART and
the Anchors family), and the bounds are shown in original units:

- integer-valued feature: the face rounded to the integer it admits
  (`MAR <= 4`, not `MAR in [1.000000, 4.371220]`); if that would select other
  D_train rows than the box (a strict `age > 37` stored one float32 step above
  37), the smallest / largest D_train value inside the box instead (`age >= 38`);
- nominal feature: the set of codes inside the box, by name when the loader
  kept its label encoder (`education = 'Masters'`, `workclass in {'Private', ...}`);
- continuous feature: the face itself, to 4 significant digits. A face one
  float32 step beyond a short decimal is a strict bound (Anchors' `x > q`, a CART
  split) and prints as `x > q` / `x < q`.

One-sided conditions print as `x >= a` or `x <= b`.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.metrics import excluding_faces, train_span


def fmt_value(v: float) -> str:
    """4 significant digits, never fewer than the integer part, no trailing zeros."""
    v = float(v)
    if not math.isfinite(v):
        return str(v)
    if v == round(v) and abs(v) < 1e15:
        return str(int(round(v)))
    mag = math.floor(math.log10(abs(v))) if v else 0
    s = f"{v:.{max(0, 3 - mag)}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _strict_face(face: float, side: int) -> Optional[float]:
    """The short decimal a float32 face sits one step beyond (side -1: lower face,
    +1: upper face), or None when the face is not such a strict bound."""
    f = np.float32(face)
    if not np.isfinite(f):
        return None
    v = np.nextafter(f, np.float32(side * np.inf))
    short = lambda x: len(np.format_float_positional(x, unique=True, trim="-").replace("-", "").replace(".", "").strip("0"))
    return float(v) if short(v) + 2 < short(f) else None


def categorical_value_names(loader) -> Dict[int, List[str]]:
    """Label-encoder names per categorical feature (as `get_anchor_env_data` exports them)."""
    names = list(getattr(loader, "feature_names", []) or [])
    encs = getattr(loader, "label_encoders", None) or {}
    return {int(names.index(n)): [str(v) for v in enc.classes_]
            for n, enc in encs.items() if n in names}


class RulePrinter:
    """Counts and prints the conditions of boxes over one dataset split's D_train.

    X_space: D_train in the space the boxes are scored in.
    X_orig:  the same rows in original units (categoricals as integer codes).
    to_orig: maps a bound vector from the box's space to original units; None when
             the boxes are already in original units.
    """

    def __init__(
        self,
        X_space: np.ndarray,
        X_orig: np.ndarray,
        feature_names: Sequence[str],
        *,
        categorical_indices: Sequence[int] = (),
        categorical_values: Optional[Dict[int, List[str]]] = None,
        to_orig: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.X_space = np.asarray(X_space, dtype=np.float32)
        self.X_orig = np.asarray(X_orig, dtype=np.float64)
        self.span: Tuple[np.ndarray, np.ndarray] = train_span(self.X_space)
        self.names = [str(n) for n in feature_names]
        self.categorical = {int(i) for i in categorical_indices}
        self.cat_values = {int(k): list(v) for k, v in (categorical_values or {}).items()}
        self.to_orig = to_orig
        self.integer = np.all(np.isfinite(self.X_orig) & (self.X_orig == np.round(self.X_orig)), axis=0)

    @classmethod
    def for_loader(cls, loader, space: str) -> "RulePrinter":
        """Printer for boxes scored in `space` ('unit' or 'original') on this loader's D_train."""
        from utils.eval_harness import unit_to_original

        cats = list(getattr(loader, "categorical_indices", []) or [])
        common = dict(categorical_indices=cats, categorical_values=categorical_value_names(loader))
        if space == "original":
            return cls(loader.X_train, loader.X_train, loader.feature_names, **common)
        if space != "unit":
            raise ValueError(f"space must be 'unit' or 'original', got {space!r}")
        mean = np.asarray(loader.scaler.mean_, dtype=np.float64)
        scale = np.asarray(loader.scaler.scale_, dtype=np.float64)
        return cls(loader.X_train_unit, loader.X_train, loader.feature_names,
                   to_orig=lambda b: unit_to_original(b, loader.X_min, loader.X_range, mean, scale),
                   **common)

    def mask(self, lower, upper) -> np.ndarray:
        lo_ex, up_ex = excluding_faces(lower, upper, self.span)
        return lo_ex | up_ex

    def count(self, lower, upper) -> int:
        return int(self.mask(lower, upper).sum())

    def conditions(self, lower, upper) -> List[str]:
        lo = np.asarray(lower, dtype=np.float32).reshape(-1)
        up = np.asarray(upper, dtype=np.float32).reshape(-1)
        lo_ex, up_ex = excluding_faces(lo, up, self.span)
        if not (lo_ex | up_ex).any():
            return []
        lo_o = up_o = None
        if self.to_orig is not None:
            lo_o = np.asarray(self.to_orig(lo.astype(np.float64)), dtype=np.float64)
            up_o = np.asarray(self.to_orig(up.astype(np.float64)), dtype=np.float64)
        out = []
        for j in np.flatnonzero(lo_ex | up_ex):
            name = self.names[j] if j < len(self.names) else f"x{j}"
            xs = self.X_space[:, j]
            inside = (xs >= lo[j]) & (xs <= up[j])
            vals = self.X_orig[inside, j]
            if j in self.categorical and vals.size:
                codes = sorted({int(round(v)) for v in vals})
                labels = self.cat_values.get(j)
                shown = [repr(labels[c]) if labels and 0 <= c < len(labels) else str(c) for c in codes]
                out.append(f"{name} = {shown[0]}" if len(shown) == 1
                           else f"{name} in {{{', '.join(shown)}}}")
                continue
            a = float(lo_o[j]) if lo_o is not None else float(lo[j])
            b = float(up_o[j]) if up_o is not None else float(up[j])
            ge, le = ">=", "<="
            if self.integer[j]:
                if vals.size:
                    a_i = math.ceil(a - 4e-7 * max(1.0, abs(a))) if lo_ex[j] else -math.inf
                    b_i = math.floor(b + 4e-7 * max(1.0, abs(b))) if up_ex[j] else math.inf
                    xo = self.X_orig[:, j]
                    if not np.array_equal((xo >= a_i) & (xo <= b_i), inside):
                        a_i, b_i = float(vals.min()), float(vals.max())
                    a, b = float(a_i), float(b_i)
            elif lo_o is None:
                # boxes built in original units: keep a strict bound strict
                sa = _strict_face(a, -1) if lo_ex[j] else None
                sb = _strict_face(b, +1) if up_ex[j] else None
                if sa is not None:
                    a, ge = sa, ">"
                if sb is not None:
                    b, le = sb, "<"
            if lo_ex[j] and up_ex[j]:
                out.append(f"{name} = {fmt_value(a)}" if a == b and ge == ">=" and le == "<="
                           else f"{fmt_value(a)} {'<' if ge == '>' else '<='} {name} {le} {fmt_value(b)}")
            elif lo_ex[j]:
                out.append(f"{name} {ge} {fmt_value(a)}")
            else:
                out.append(f"{name} {le} {fmt_value(b)}")
        return out

    def __call__(self, lower, upper) -> str:
        conds = self.conditions(lower, upper)
        return " and ".join(conds) if conds else "any values"
