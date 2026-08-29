"""
Shared evaluation metrics for the RLDA / MADA revision (C-01, C-02, C-04, C-08, C-09, C-11, C-52).

All reported numbers for the rebuttal must go through this module so that
fidelity vs purity, empty-box handling, union-vs-best, and ranking stay consistent
across RLDA, MADA, and the class-level baselines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


MIN_SUPPORT_DEFAULT = 10
RANKING_SCORE_PRECISION_COVERAGE = "precision_coverage"  # score = fid * (1 + cov)
RANKING_SCORE_F1 = "f1"
RANKING_SCORE_FIDELITY = "fidelity"
# A1: support-aware default. Uses the Wilson LOWER bound of fidelity instead of the
# point estimate, so a rule that is "perfect" on one covered sample cannot outrank a
# well-supported rule. fid=1.0 on n=1 has a lower bound of ~0.21; fid=0.92 on n=25 is
# ~0.75. The point-estimate formulas above are kept for ablation / backwards checks.
RANKING_SCORE_LCB_COVERAGE = "lcb_coverage"  # score = wilson_low(fid) * (1 + cov)


# ---------------------------------------------------------------------------
# C-04 — Eq. 11 dynamic precision weight
# ---------------------------------------------------------------------------

def paper_eq11_w_p(p_t: float, tau_p: float) -> float:
    """The formula as written in the reviewed draft.

    For P_t >= tau_P: max(2.0, 1 + (P_t - tau_P)/(1 - tau_P)).
    The second argument lies in [1, 2] on [tau_P, 1], so the max is always 2.0 —
    identical to the else-branch. This is a documented no-op; do not use it.
    """
    p_t = float(p_t)
    tau_p = float(tau_p)
    if p_t >= tau_p and tau_p < 1.0:
        return max(2.0, 1.0 + (p_t - tau_p) / (1.0 - tau_p))
    return 2.0


def intended_w_p(p_t: float, tau_p: float) -> float:
    """Intended non-constant weight: clip 1 + (P - tau)/(1 - tau) to [1, 2].

    Increases from 1 at P = tau_P to 2 at P = 1. Unused in training: both
    environments use potential-based shaping with fixed alpha/beta (C-15 already
    landed). Kept so the unit test can demonstrate the paper formula is constant
    and this one is not. Do not wire this into the reward without a rerun.
    """
    p_t = float(np.clip(p_t, 0.0, 1.0))
    tau_p = float(np.clip(tau_p, 0.0, 1.0 - 1e-9))
    raw = 1.0 + (p_t - tau_p) / (1.0 - tau_p)
    return float(np.clip(raw, 1.0, 2.0))


# ---------------------------------------------------------------------------
# C-02 — Wilson interval, empty-box, min-support
# ---------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (low, high). Empty n -> (nan, nan).
    """
    if n <= 0:
        return float("nan"), float("nan")
    k = int(k)
    n = int(n)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))


def _safe_mean(k: int, n: int) -> float:
    """Proportion, or NaN when the denominator is empty (C-02)."""
    if n <= 0:
        return float("nan")
    return float(k) / float(n)


# ---------------------------------------------------------------------------
# C-09 — Fidelity vs purity
# ---------------------------------------------------------------------------

@dataclass
class BoxMetrics:
    """Metrics of one box (or a union of boxes) on a fixed evaluation set."""

    n_covered: int
    n_eval: int
    n_class: int
    n_covered_class: int
    n_fid_agree: int
    n_pur_agree: int
    fidelity: float          # P(f_hat(x) = target | x in B)   PRIMARY
    purity: float            # P(y = target | x in B)          SECONDARY
    coverage: float          # P(x in B | y = target) if class-conditional else P(x in B)
    coverage_marginal: float # P(x in B)
    fid_ci: Tuple[float, float]
    pur_ci: Tuple[float, float]
    below_min_support: bool
    target_class: int
    class_conditional: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_covered": int(self.n_covered),
            "n_eval": int(self.n_eval),
            "n_class": int(self.n_class),
            "n_covered_class": int(self.n_covered_class),
            "n_fid_agree": int(self.n_fid_agree),
            "n_pur_agree": int(self.n_pur_agree),
            "fidelity": _json_float(self.fidelity),
            "purity": _json_float(self.purity),
            "coverage": _json_float(self.coverage),
            "coverage_marginal": _json_float(self.coverage_marginal),
            "fidelity_ci_low": _json_float(self.fid_ci[0]),
            "fidelity_ci_high": _json_float(self.fid_ci[1]),
            "purity_ci_low": _json_float(self.pur_ci[0]),
            "purity_ci_high": _json_float(self.pur_ci[1]),
            "below_min_support": bool(self.below_min_support),
            "target_class": int(self.target_class),
            "class_conditional": bool(self.class_conditional),
        }


def _json_float(x: float) -> Optional[float]:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return float(x)


def box_mask(X: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    lower = np.asarray(lower, dtype=np.float32).reshape(-1)
    upper = np.asarray(upper, dtype=np.float32).reshape(-1)
    return np.all((X >= lower) & (X <= upper), axis=1)


def evaluate_box(
    X: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_class: int,
    class_conditional: bool = True,
    min_support: int = MIN_SUPPORT_DEFAULT,
    extra_mask: Optional[np.ndarray] = None,
) -> BoxMetrics:
    """Score one axis-aligned box on a held-out set.

    extra_mask: optional additional constraint (e.g. categorical equalities)
    already combined with the interval mask.
    """
    mask = box_mask(X, lower, upper)
    if extra_mask is not None:
        mask = mask & extra_mask
    return evaluate_mask(
        y=y,
        y_hat=y_hat,
        mask=mask,
        target_class=target_class,
        class_conditional=class_conditional,
        min_support=min_support,
    )


def evaluate_mask(
    y: np.ndarray,
    y_hat: np.ndarray,
    mask: np.ndarray,
    target_class: int,
    class_conditional: bool = True,
    min_support: int = MIN_SUPPORT_DEFAULT,
) -> BoxMetrics:
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    mask = np.asarray(mask, dtype=bool)
    n_eval = int(mask.shape[0])
    n_covered = int(mask.sum())
    class_mask = (y == target_class)
    n_class = int(class_mask.sum())
    n_covered_class = int((mask & class_mask).sum())

    if n_covered == 0:
        n_fid = 0
        n_pur = 0
        fid = float("nan")
        pur = float("nan")
    else:
        n_fid = int((y_hat[mask] == target_class).sum())
        n_pur = int((y[mask] == target_class).sum())
        fid = _safe_mean(n_fid, n_covered)
        pur = _safe_mean(n_pur, n_covered)

    if class_conditional:
        cov = _safe_mean(n_covered_class, n_class) if n_class > 0 else float("nan")
    else:
        cov = _safe_mean(n_covered, n_eval)

    return BoxMetrics(
        n_covered=n_covered,
        n_eval=n_eval,
        n_class=n_class,
        n_covered_class=n_covered_class,
        n_fid_agree=n_fid if n_covered else 0,
        n_pur_agree=n_pur if n_covered else 0,
        fidelity=fid,
        purity=pur,
        coverage=cov,
        coverage_marginal=_safe_mean(n_covered, n_eval),
        fid_ci=wilson_interval(n_fid if n_covered else 0, n_covered),
        pur_ci=wilson_interval(n_pur if n_covered else 0, n_covered),
        below_min_support=n_covered < int(min_support),
        target_class=int(target_class),
        class_conditional=bool(class_conditional),
    )


# ---------------------------------------------------------------------------
# C-52 — ranking score (the formula that determines every `best` and every union)
# ---------------------------------------------------------------------------

def ranking_score(
    fidelity: float,
    coverage: float,
    formula: str = RANKING_SCORE_LCB_COVERAGE,
    n_covered: Optional[int] = None,
    min_support: Optional[int] = None,
) -> float:
    """Configurable rule-ranking score.

    Default (A1) is support-aware:

        score = wilson_lower_bound(fidelity, n_covered) * (1 + coverage)

    The previous default, `fidelity * (1 + coverage)`, used the point estimate and
    ignored support entirely, so a box enclosing a single covered sample scored
    fid=1.0 -> 1.0 * (1 + cov) and beat well-supported rules. Those rules then
    covered nothing at test time. The Wilson lower bound penalises exactly that:
    fid=1.0 on n=1 scores ~0.21, while fid=0.92 on n=25 scores ~0.75.

    `n_covered` is required for the LCB formulas; without it they fall back to the
    point estimate (and a rule with unknown support cannot be preferred on the
    strength of an unmeasurable interval).

    NaN fidelity (empty box) scores as -inf so it cannot become `best`.
    """
    if fidelity is None or (isinstance(fidelity, float) and np.isnan(fidelity)):
        return float("-inf")
    fid = float(fidelity)
    cov = 0.0 if coverage is None or (isinstance(coverage, float) and np.isnan(coverage)) else float(coverage)
    if formula == RANKING_SCORE_LCB_COVERAGE:
        if n_covered is None:
            return fid * (1.0 + cov)
        n = int(n_covered)
        floor = 1 if min_support is None else int(min_support)
        if n < floor:
            return float("-inf")
        lo, _ = wilson_interval(int(round(fid * n)), n)
        return float(lo) * (1.0 + cov)
    if formula == RANKING_SCORE_PRECISION_COVERAGE:
        return fid * (1.0 + cov)
    if formula == RANKING_SCORE_F1:
        s = fid + cov
        return 0.0 if s <= 0 else 2.0 * fid * cov / s
    if formula == RANKING_SCORE_FIDELITY:
        return fid
    raise ValueError(f"Unknown ranking formula {formula!r}")


# ---------------------------------------------------------------------------
# C-01 — union over the same pool as `best`, with hard assertions
# ---------------------------------------------------------------------------

@dataclass
class RankedRule:
    rule_id: str
    lower: Optional[np.ndarray]
    upper: Optional[np.ndarray]
    mask: np.ndarray
    metrics: BoxMetrics
    score: float
    display_rule: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnionResult:
    k: int
    n_selected: int
    selected_ids: List[str]
    best: RankedRule
    union_metrics: BoxMetrics
    individual: List[RankedRule]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": int(self.k),
            "n_selected": int(self.n_selected),
            "selected_ids": list(self.selected_ids),
            "best_id": self.best.rule_id,
            "best_fidelity": _json_float(self.best.metrics.fidelity),
            "best_purity": _json_float(self.best.metrics.purity),
            "best_coverage": _json_float(self.best.metrics.coverage),
            "best_n_covered": int(self.best.metrics.n_covered),
            "best_score": float(self.best.score) if np.isfinite(self.best.score) else None,
            "union": self.union_metrics.to_dict(),
            "individual": [
                {
                    "rule_id": r.rule_id,
                    "display_rule": r.display_rule,
                    "score": float(r.score) if np.isfinite(r.score) else None,
                    **r.metrics.to_dict(),
                }
                for r in self.individual
            ],
        }


def _assert_union_vs_best(union_cov: float, individual_covs: Sequence[float], atol: float = 1e-9) -> None:
    """C-01: union coverage must be >= every member's coverage on the same denominator."""
    finite = [c for c in individual_covs if c is not None and np.isfinite(c)]
    if not finite:
        return
    best = max(finite)
    if union_cov is None or not np.isfinite(union_cov):
        if best > atol:
            raise AssertionError(
                f"Union coverage is undefined/NaN but a member has coverage {best:.6f}"
            )
        return
    if union_cov + atol < best:
        raise AssertionError(
            f"Union coverage {union_cov:.10f} < best member coverage {best:.10f} "
            f"(C-01 invariant). The union set and the `best` rule are not the same pool, "
            f"or they were scored on different denominators."
        )


def select_topk_union(
    rules: Sequence[RankedRule],
    y: np.ndarray,
    y_hat: np.ndarray,
    target_class: int,
    k: int,
    class_conditional: bool = True,
    min_support: int = MIN_SUPPORT_DEFAULT,
    enforce_min_support: bool = True,
) -> Optional[UnionResult]:
    """Rank rules, take top-k (same pool), report `best` = rank-1 of that set, union the k.

    `enforce_min_support` must be True only when this call is *selecting* rules
    (i.e. on validation). On the reporting split the rule set is already fixed;
    filtering there by test support would be selection on test data.

    Recommended reporting convention (C-01 action 4).
    """
    scored = [r for r in rules if np.isfinite(r.score)]
    if not scored:
        return None
    # A1: a rule whose selection-split support is below min_support is not a
    # measurement — its fidelity CI spans most of [0,1]. Such rules previously
    # won selection outright (fid=1.0 on n=1) and then covered nothing at test.
    # Drop them, but never return an empty set: if every candidate is
    # under-supported, keep the best-supported ones so the class still gets a
    # rule set and the shortfall is visible in `below_min_support` downstream.
    supported = ([r for r in scored if int(getattr(r.metrics, "n_covered", 0)) >= int(min_support)]
                 if enforce_min_support else list(scored))
    if supported:
        scored = supported
    else:
        scored.sort(key=lambda r: int(getattr(r.metrics, "n_covered", 0)), reverse=True)
        best_n = int(getattr(scored[0].metrics, "n_covered", 0))
        scored = [r for r in scored if int(getattr(r.metrics, "n_covered", 0)) == best_n]
    scored.sort(key=lambda r: (r.score, r.metrics.coverage if np.isfinite(r.metrics.coverage) else -1.0), reverse=True)
    if k is None or k < 0:
        selected = scored
    else:
        selected = scored[: max(1, int(k))]
    best = selected[0]
    union_mask = np.zeros_like(selected[0].mask, dtype=bool)
    for r in selected:
        union_mask |= r.mask
    union_metrics = evaluate_mask(
        y=y,
        y_hat=y_hat,
        mask=union_mask,
        target_class=target_class,
        class_conditional=class_conditional,
        min_support=min_support,
    )
    _assert_union_vs_best(
        union_metrics.coverage,
        [r.metrics.coverage for r in selected],
    )
    if union_metrics.n_class != best.metrics.n_class:
        raise AssertionError(
            f"Union and best-rule class-conditional denominators differ: "
            f"union n_class={union_metrics.n_class}, best n_class={best.metrics.n_class}"
        )
    return UnionResult(
        k=int(k) if k is not None and k >= 0 else len(selected),
        n_selected=len(selected),
        selected_ids=[r.rule_id for r in selected],
        best=best,
        union_metrics=union_metrics,
        individual=list(selected),
    )


# ---------------------------------------------------------------------------
# C-08 — printed-rule vs evaluated-box coverage
# ---------------------------------------------------------------------------

def assert_print_matches_box(
    box_mask_arr: np.ndarray,
    printed_mask: np.ndarray,
    rule_id: str = "",
    rtol: float = 0.0,
    atol: int = 0,
) -> None:
    """Fail if dropping near-full-range features changed the covered set.

    Dropping a constraint can only grow the region, so printed_mask should equal
    box_mask_arr. Any extra samples in the printed rule mean the displayed rule
    is not the rule whose metrics you reported.
    """
    box_mask_arr = np.asarray(box_mask_arr, dtype=bool)
    printed_mask = np.asarray(printed_mask, dtype=bool)
    extra = int(np.sum(printed_mask & ~box_mask_arr))
    missing = int(np.sum(box_mask_arr & ~printed_mask))
    if extra > atol or missing > atol:
        raise AssertionError(
            f"Printed rule {rule_id!r} covers a different set than the evaluated box "
            f"(+{extra} extra, -{missing} missing samples). Metrics must be taken on "
            f"the box, or the printer must not drop features that change coverage."
        )


def active_feature_mask(
    lower: np.ndarray,
    upper: np.ndarray,
    sparsity_width_ratio: float = 0.95,
    feature_min: Optional[np.ndarray] = None,
    feature_max: Optional[np.ndarray] = None,
    quantile_active: Optional[np.ndarray] = None,
) -> np.ndarray:
    """True on constrained dimensions.

    If `quantile_active` is provided it is the source of truth (quantile MDP).
    Otherwise fall back to unit-width < sparsity_width_ratio of full range.
    """
    if quantile_active is not None:
        return np.asarray(quantile_active, dtype=bool).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    if feature_min is None:
        feature_min = np.zeros_like(lower)
    if feature_max is None:
        feature_max = np.ones_like(upper)
    feature_min = np.asarray(feature_min, dtype=np.float64).reshape(-1)
    feature_max = np.asarray(feature_max, dtype=np.float64).reshape(-1)
    full = np.maximum(feature_max - feature_min, 1e-12)
    width = np.maximum(upper - lower, 0.0)
    return width < (float(sparsity_width_ratio) * full)


def sparsify_box(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    sparsity_width_ratio: float,
    max_features: Optional[int] = None,
    active_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop inactive dimensions by restoring their full [0,1] range."""
    lower = np.asarray(lower, dtype=np.float32).copy()
    upper = np.asarray(upper, dtype=np.float32).copy()
    active = active_feature_mask(
        lower, upper, sparsity_width_ratio=sparsity_width_ratio,
        quantile_active=active_mask,
    )
    active_idx = np.flatnonzero(active)
    if max_features not in (None, -1, 0) and len(active_idx) > int(max_features):
        widths = upper[active_idx] - lower[active_idx]
        if active_mask is not None:
            keep = active_idx[np.argsort(widths)[::-1][:int(max_features)]]
        else:
            keep = active_idx[np.argsort(widths)[:int(max_features)]]
        active[:] = False
        active[keep] = True
    lower[~active] = 0.0
    upper[~active] = 1.0
    return lower, upper, active


# ---------------------------------------------------------------------------
# C-11 — seed aggregation
# ---------------------------------------------------------------------------

def mean_ci(values: Iterable[float], alpha: float = 0.05, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Bootstrap mean ± 95% CI over seeds. NaNs are dropped."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0, "std": float("nan")}
    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": 1, "std": 0.0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": int(arr.size), "std": float(arr.std(ddof=1))}


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Paired Wilcoxon signed-rank test of a vs b (e.g. method vs Anchors)."""
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:
        raise ImportError("scipy is required for Wilcoxon tests") from exc
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        md = float((x - y).mean()) if x.size else None
        return {"pvalue": None, "statistic": None, "n": int(x.size), "note": "too few pairs", "mean_diff": md}
    d = x - y
    if np.allclose(d, 0.0):
        return {"pvalue": 1.0, "statistic": 0.0, "n": int(x.size), "note": "all differences zero", "mean_diff": 0.0, "effect_size_rank_biserial": 0.0}
    try:
        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        # Rank-biserial correlation: r = 1 - 2W / (n(n+1)), W = Wilcoxon statistic.
        n = float(x.size)
        r_rb = 1.0 - (2.0 * float(stat)) / (n * (n + 1.0))
        return {
            "pvalue": float(p),
            "statistic": float(stat),
            "n": int(x.size),
            "effect_size_rank_biserial": float(r_rb),
            "mean_diff": float((x - y).mean()),
        }
    except ValueError as e:
        return {"pvalue": None, "statistic": None, "n": int(x.size), "note": str(e)}


# ---------------------------------------------------------------------------
# C-11 — success rate; C-36 — compactness
# ---------------------------------------------------------------------------

def _collect_episode_anchors(obj: Any) -> List[Dict[str, Any]]:
    """Walk nested inference JSON (MADA per-agent slots) for episode records."""
    out: List[Dict[str, Any]] = []
    if not isinstance(obj, dict):
        return out
    direct = obj.get("all_anchors") or obj.get("anchors")
    if isinstance(direct, list) and direct and isinstance(direct[0], dict):
        out.extend(direct)
        return out
    for key in ("per_agent_results", "class_based_results", "agents"):
        nested = obj.get(key)
        if isinstance(nested, dict):
            for v in nested.values():
                out.extend(_collect_episode_anchors(v))
        elif isinstance(nested, list):
            for v in nested:
                out.extend(_collect_episode_anchors(v))
    return out


def _episode_coverage(anc: Dict[str, Any]) -> Optional[float]:
    """Coverage scalar compared to τ_C. Prefer class-conditional (training target)."""
    for key in (
        "coverage_class_conditional_rollout_estimated",
        "coverage_class_conditional_recomputed",
        "coverage_class_conditional",
        "instance_coverage_class_conditional",
        "anchor_coverage_class_conditional",
        "coverage_rollout_estimated",
        "coverage",
    ):
        if anc.get(key) is not None:
            return float(anc[key])
    return None


def episode_success_rate(
    anchors: Sequence[Dict[str, Any]],
    tau_p: float,
    tau_c: float,
    t_max: Optional[int] = None,
) -> Dict[str, Any]:
    """Fraction of episodes that reach both τ_P and τ_C (R1, C-11).

    Precision uses the rollout estimate the environment used for termination.
    Coverage prefers the class-conditional rollout estimate (the training τ_C).
    """
    n = 0
    n_ok = 0
    for anc in anchors or []:
        prec = None
        for key in (
            "precision_rollout_estimated",
            "precision_recomputed",
            "anchor_precision",
            "instance_precision",
            "precision",
        ):
            if anc.get(key) is not None:
                prec = float(anc[key])
                break
        cov = _episode_coverage(anc)
        if prec is None or cov is None:
            continue
        if anc.get("rollout_type") == "class_based":
            continue
        n += 1
        hit = float(prec) + 1e-12 >= float(tau_p) and float(cov) + 1e-12 >= float(tau_c)
        if t_max is not None:
            steps = anc.get("n_steps")
            if steps is not None and int(steps) >= int(t_max) and not hit:
                hit = False
        n_ok += int(hit)
    return {
        "n_episodes": int(n),
        "n_success": int(n_ok),
        "success_rate": (float(n_ok) / float(n)) if n else None,
        "tau_p": float(tau_p),
        "tau_c": float(tau_c),
        "t_max": None if t_max is None else int(t_max),
    }


def collect_success_rate(
    rules: Dict[str, Any],
    tau_p: float,
    tau_c: float,
) -> Dict[str, Any]:
    """Aggregate C-11 success rate from an extracted_rules JSON."""
    metadata = rules.get("metadata") or {}
    t_max = metadata.get("steps_per_episode") or metadata.get("max_cycles")
    per_class: Dict[str, Any] = {}
    n_ep = n_ok = 0
    pcr = rules.get("per_class_results") or {}
    for key, cd in pcr.items():
        if not isinstance(cd, dict):
            continue
        anchors = _collect_episode_anchors(cd)
        stats = episode_success_rate(anchors, tau_p, tau_c, t_max)
        per_class[key] = stats
        if str(key).endswith("_class_based"):
            continue
        n_ep += int(stats["n_episodes"])
        n_ok += int(stats["n_success"])
    return {
        "n_episodes": int(n_ep),
        "n_success": int(n_ok),
        "success_rate": (float(n_ok) / float(n_ep)) if n_ep else None,
        "tau_p": float(tau_p),
        "tau_c": float(tau_c),
        "t_max": None if t_max is None else int(t_max),
        "per_class": per_class,
    }


def compactness_of_box(
    lower: np.ndarray,
    upper: np.ndarray,
    sparsity_width_ratio: float = 0.95,
) -> Dict[str, Any]:
    """Active features / description length of one unit-space box (C-36)."""
    active = active_feature_mask(lower, upper, sparsity_width_ratio=sparsity_width_ratio)
    n_feat = int(active.size)
    n_active = int(active.sum())
    return {
        "n_active_features": n_active,
        "n_features": n_feat,
        "sparsity": (1.0 - n_active / n_feat) if n_feat else None,
        "sparsity_width_ratio": float(sparsity_width_ratio),
    }


def compactness_of_ruleset(
    rules: Sequence[RankedRule],
    sparsity_width_ratio: float = 0.95,
) -> Dict[str, Any]:
    rows = []
    for r in rules:
        if r.lower is None or r.upper is None:
            continue
        rows.append(compactness_of_box(r.lower, r.upper, sparsity_width_ratio))
    if not rows:
        return {
            "n_rules": 0,
            "mean_active_features": None,
            "total_active_features": 0,
            "sparsity_width_ratio": float(sparsity_width_ratio),
        }
    n_active = [r["n_active_features"] for r in rows]
    return {
        "n_rules": len(rows),
        "mean_active_features": float(np.mean(n_active)),
        "total_active_features": int(np.sum(n_active)),
        "n_features": rows[0]["n_features"],
        "sparsity_width_ratio": float(sparsity_width_ratio),
        "per_rule": rows,
    }


# ---------------------------------------------------------------------------
# C-12 — break-even (amortized vs per-instance query cost)
# ---------------------------------------------------------------------------

def break_even_n(
    fixed_cost: float,
    per_instance_cost: float,
    baseline_per_instance: float,
) -> Optional[float]:
    """Smallest n where amortized method is cheaper than a per-instance baseline."""
    gap = float(baseline_per_instance) - float(per_instance_cost)
    if gap <= 0:
        return None
    n = float(fixed_cost) / gap
    return n if n > 0 else 0.0
