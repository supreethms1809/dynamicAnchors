"""Quantile-position MDP helpers: CDFs, leave-corner, CRN D(z|A)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def fit_train_cdfs(
    X_unit: np.ndarray,
    y: np.ndarray,
    target_class: int,
) -> Dict[str, np.ndarray]:
    """Fit per-column empirical CDFs on D_train only.

    v_class[j] is sorted target-class values (action/obs quantiles).
    v_all[j] is sorted all-train values (D(z|A) pool).
    """
    X_unit = np.asarray(X_unit, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    n, d = X_unit.shape
    class_mask = y == int(target_class)
    X_class = X_unit[class_mask] if class_mask.any() else X_unit
    v_class = np.stack(
        [np.sort(X_class[:, j]) for j in range(d)], axis=0
    )
    v_all = np.stack([np.sort(X_unit[:, j]) for j in range(d)], axis=0)
    return {
        "v_class": v_class.astype(np.float64),
        "v_all": v_all.astype(np.float64),
        "n_train": int(n),
        "n_class": int(class_mask.sum()),
    }


def empirical_quantile(v_sorted: np.ndarray, q: float) -> float:
    """Left-continuous inverse CDF. Q(0)=min, Q(1)=max."""
    v = np.asarray(v_sorted, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return 0.0
    if v.size == 1:
        return float(v[0])
    q = float(np.clip(q, 0.0, 1.0))
    if q <= 0.0:
        return float(v[0])
    if q >= 1.0:
        return float(v[-1])
    # np.quantile type=7 (linear) is fine for continuous cols; atoms pin separately.
    try:
        return float(np.quantile(v, q, method="linear"))
    except TypeError:
        return float(np.quantile(v, q, interpolation="linear"))


def value_to_quantile(v_sorted: np.ndarray, x: float) -> float:
    """Empirical CDF F(x) = (# of v <= x) / n, clipped to [0, 1]."""
    v = np.asarray(v_sorted, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return 0.5
    x = float(x)
    # Inclusive rank so ties at the atom map inside the atom's mass.
    rank = float(np.searchsorted(v, x, side="right"))
    return float(np.clip(rank / float(v.size), 0.0, 1.0))


def constrained_mask(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return (a > float(eps)) | (b < (1.0 - float(eps)))


def quantile_to_unit_bounds(
    a: np.ndarray,
    b: np.ndarray,
    v_class: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map (a,b) to unit-space [ℓ,u]. Unconstrained corners stay {0,1}."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    d = a.shape[0]
    lower = np.zeros(d, dtype=np.float32)
    upper = np.ones(d, dtype=np.float32)
    for j in range(d):
        if a[j] > eps:
            lower[j] = np.float32(empirical_quantile(v_class[j], a[j]))
        if b[j] < (1.0 - eps):
            upper[j] = np.float32(empirical_quantile(v_class[j], b[j]))
        if lower[j] > upper[j]:
            lower[j], upper[j] = upper[j], lower[j]
    return lower, upper


def categorical_atom_quantiles(
    v_class_j: np.ndarray,
    value: float,
) -> Tuple[float, float]:
    """Quantile interval covering one categorical atom in the class CDF."""
    v = np.asarray(v_class_j, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return 0.0, 1.0
    rounded = np.round(v, 6)
    target = float(np.round(value, 6))
    lo = int(np.searchsorted(rounded, target, side="left"))
    hi = int(np.searchsorted(rounded, target, side="right"))
    if hi <= lo:
        # Value absent from class column: pin to nearest atom.
        nearest = int(np.argmin(np.abs(v - value)))
        target = float(rounded[nearest])
        lo = int(np.searchsorted(rounded, target, side="left"))
        hi = int(np.searchsorted(rounded, target, side="right"))
    n = float(v.size)
    a = float(lo) / n
    b = float(hi) / n
    if b <= a:
        b = min(1.0, a + 1.0 / n)
    return a, b


def apply_leave_corner_action(
    a: np.ndarray,
    b: np.ndarray,
    action: np.ndarray,
    *,
    eta: float,
    leave_threshold: float,
    max_new_constraints: int,
    min_quantile_width: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Δa/Δb with leave-corner threshold and at most one new constraint."""
    a = np.asarray(a, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    d = a.shape[0]
    da = action[:d]
    db = action[d:]
    currently = constrained_mask(a, b, eps)

    leave_scores: List[Tuple[float, int, str]] = []
    for j in range(d):
        if currently[j]:
            continue
        if a[j] <= eps and da[j] > leave_threshold:
            leave_scores.append((float(da[j]), j, "a"))
        if b[j] >= (1.0 - eps) and db[j] < -leave_threshold:
            leave_scores.append((float(-db[j]), j, "b"))
    leave_scores.sort(key=lambda t: t[0], reverse=True)
    allowed = set()
    allowed_kind: Dict[int, str] = {}
    for score, j, kind in leave_scores:
        if j in allowed:
            continue
        if len(allowed) >= int(max(0, max_new_constraints)):
            break
        allowed.add(j)
        allowed_kind[j] = kind

    new_a = a.copy()
    new_b = b.copy()
    for j in range(d):
        if currently[j]:
            new_a[j] = a[j] + eta * da[j]
            new_b[j] = b[j] + eta * db[j]
        elif j in allowed:
            kind = allowed_kind[j]
            if kind == "a":
                new_a[j] = a[j] + eta * da[j]
            else:
                new_b[j] = b[j] + eta * db[j]
        # else stay at the corner
        new_a[j] = float(np.clip(new_a[j], 0.0, 1.0))
        new_b[j] = float(np.clip(new_b[j], 0.0, 1.0))
        if new_b[j] - new_a[j] < min_quantile_width:
            mid = 0.5 * (new_a[j] + new_b[j])
            half = 0.5 * min_quantile_width
            new_a[j] = float(np.clip(mid - half, 0.0, 1.0 - min_quantile_width))
            new_b[j] = float(np.clip(new_a[j] + min_quantile_width, 0.0, 1.0))
    return new_a.astype(np.float64), new_b.astype(np.float64)


def clip_quantiles_around_qstar(
    a: np.ndarray,
    b: np.ndarray,
    q_star: np.ndarray,
    active: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep q* inside constrained (a,b). Unconstrained dims are untouched."""
    a = np.asarray(a, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()
    q_star = np.asarray(q_star, dtype=np.float64).reshape(-1)
    active = np.asarray(active, dtype=bool).reshape(-1)
    for j in np.flatnonzero(active):
        qj = float(np.clip(q_star[j], 0.0, 1.0))
        if qj < a[j]:
            a[j] = qj
        if qj > b[j]:
            b[j] = qj
        if b[j] - a[j] < eps:
            a[j] = max(0.0, qj - 0.5 * eps)
            b[j] = min(1.0, a[j] + eps)
    return a, b


def crn_perturb(
    X_unit: np.ndarray,
    idx: np.ndarray,
    U: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    active: np.ndarray,
    v_all: np.ndarray,
    *,
    x_star_unit: Optional[np.ndarray] = None,
    class_mode_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Conditional D(z|A) with frozen (idx, U). Inclusive ranks matching box_mask."""
    X_unit = np.asarray(X_unit, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    U = np.asarray(U, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    active = np.asarray(active, dtype=bool).reshape(-1)
    n = idx.shape[0]
    d = X_unit.shape[1]
    z = X_unit[np.clip(idx, 0, X_unit.shape[0] - 1)].copy()
    for j in range(d):
        if not active[j]:
            continue
        v = np.asarray(v_all[j], dtype=np.float64).reshape(-1)
        lo = int(np.searchsorted(v, float(lower[j]), side="left"))
        hi = int(np.searchsorted(v, float(upper[j]), side="right"))
        span = hi - lo
        if span <= 0:
            if x_star_unit is not None:
                z[:, j] = float(np.asarray(x_star_unit).reshape(-1)[j])
            elif class_mode_values is not None:
                z[:, j] = float(np.asarray(class_mode_values).reshape(-1)[j])
            continue
        picks = lo + np.floor(np.clip(U[:, j], 0.0, 1.0 - 1e-12) * span).astype(np.int64)
        picks = np.clip(picks, lo, hi - 1)
        z[:, j] = v[picks]
    return z.astype(np.float32)


def export_knots(cdfs: Dict[str, np.ndarray]) -> Dict[str, list]:
    return {
        "v_class": np.asarray(cdfs["v_class"]).tolist(),
        "v_all": np.asarray(cdfs["v_all"]).tolist(),
        "n_train": int(cdfs["n_train"]),
        "n_class": int(cdfs["n_class"]),
    }


def decode_obs_layout(obs_len: int, n_features: Optional[int] = None) -> Optional[str]:
    """Return "hull" (2n+3 / 2n+2), "quantile" (3n+3), or None if undecidable.

    The two layouts are NOT distinguishable from obs_len alone (2n+3 and 3m+3
    collide), so callers that do not know n_features must treat None as "do not
    slice" rather than guessing -- decoding a quantile observation as bounds
    yields quantile positions silently mislabelled as unit bounds.
    """
    if n_features is None or int(n_features) <= 0:
        return None
    n = int(n_features)
    if obs_len in (3 * n + 3,):
        return "quantile"
    if obs_len in (2 * n + 3, 2 * n + 2):
        return "hull"
    return None
