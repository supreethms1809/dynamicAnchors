"""Unit tests for the revision metrics / harness (C-01, C-02, C-04, C-08, C-16)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.metrics import (
    active_feature_mask,
    assert_print_matches_box,
    evaluate_mask,
    intended_w_p,
    paper_eq11_w_p,
    ranking_score,
    select_topk_union,
    sparsify_box,
    wilson_interval,
    RankedRule,
)
from utils.eval_harness import (
    evaluate_ruleset_as_classifier,
    reevaluate_ranked_rules,
    rules_from_anchors,
    unit_to_original,
)


def test_c02_empty_box_precision_is_nan():
    y = np.array([0, 0, 1, 1])
    y_hat = np.array([0, 1, 1, 1])
    mask = np.zeros(4, dtype=bool)
    m = evaluate_mask(y, y_hat, mask, target_class=1)
    assert np.isnan(m.fidelity)
    assert np.isnan(m.purity)
    assert m.n_covered == 0
    lo, hi = m.fid_ci
    assert np.isnan(lo) and np.isnan(hi)


def test_c02_wilson_on_singleton_is_wide():
    lo, hi = wilson_interval(k=1, n=1)
    assert lo < 0.5
    assert hi > 0.8
    lo0, hi0 = wilson_interval(k=0, n=0)
    assert np.isnan(lo0) and np.isnan(hi0)


def test_c02_min_support_flag():
    y = np.array([1, 0, 0, 0, 0])
    y_hat = np.array([1, 1, 0, 0, 0])
    mask = np.array([True, False, False, False, False])
    m = evaluate_mask(y, y_hat, mask, target_class=1, min_support=10)
    assert m.fidelity == 1.0
    assert m.below_min_support is True
    assert m.n_covered == 1


def test_c04_paper_eq11_is_constant():
    tau = 0.9
    vals = [paper_eq11_w_p(p, tau) for p in np.linspace(0.0, 1.0, 21)]
    assert np.allclose(vals, 2.0), vals


def test_c04_intended_w_p_is_non_constant():
    tau = 0.9
    vals = [intended_w_p(p, tau) for p in np.linspace(0.0, 1.0, 21)]
    assert len(set(np.round(vals, 6))) > 1
    assert intended_w_p(tau, tau) == pytest.approx(1.0)
    assert intended_w_p(1.0, tau) == pytest.approx(2.0)


def test_c01_union_coverage_ge_best():
    rng = np.random.default_rng(0)
    n = 40
    y = rng.integers(0, 2, size=n)
    y_hat = y.copy()
    m1 = np.zeros(n, dtype=bool)
    m1[:15] = True
    m2 = np.zeros(n, dtype=bool)
    m2[10:25] = True
    rules = []
    for i, mask in enumerate([m1, m2]):
        metrics = evaluate_mask(y, y_hat, mask, target_class=1, class_conditional=True)
        rules.append(RankedRule(
            rule_id=str(i), lower=None, upper=None, mask=mask, metrics=metrics,
            score=ranking_score(metrics.fidelity, metrics.coverage),
        ))
    union = select_topk_union(rules, y, y_hat, target_class=1, k=2)
    assert union is not None
    best_cov = max(r.metrics.coverage for r in union.individual)
    assert union.union_metrics.coverage + 1e-12 >= best_cov
    assert union.best.rule_id in union.selected_ids


def test_c01_assertion_fires_on_inconsistent_pool():
    y = np.array([1, 1, 1, 0, 0, 0])
    y_hat = y.copy()
    wide = np.array([True, True, True, False, False, False])
    tiny = np.array([True, False, False, False, False, False])
    m_wide = evaluate_mask(y, y_hat, wide, 1)
    m_tiny = evaluate_mask(y, y_hat, tiny, 1)
    fake_best = RankedRule("best", None, None, tiny, m_wide, score=99.0)
    other = RankedRule("other", None, None, tiny, m_tiny, score=1.0)
    with pytest.raises(AssertionError, match="Union coverage"):
        select_topk_union([fake_best, other], y, y_hat, target_class=1, k=2)


def test_c08_print_vs_box_assertion():
    box = np.array([True, True, False, False])
    printed_relaxed = np.array([True, True, True, False])
    with pytest.raises(AssertionError, match="Printed rule"):
        assert_print_matches_box(box, printed_relaxed, rule_id="demo")
    assert_print_matches_box(box, box, rule_id="ok")


def test_c08_sparsity_threshold_is_a_hyperparameter():
    lower = np.array([0.0, 0.2])
    upper = np.array([1.0, 0.3])
    active_strict = active_feature_mask(lower, upper, sparsity_width_ratio=0.95)
    assert bool(active_strict[0]) is False
    assert bool(active_strict[1]) is True
    active_loose = active_feature_mask(lower, upper, sparsity_width_ratio=0.05)
    assert bool(active_loose[1]) is False


def test_c08_sparsified_box_drops_unprinted_dimensions():
    lower = np.array([0.0, 0.2, 0.4])
    upper = np.array([1.0, 0.3, 0.6])
    sparse_lo, sparse_up, active = sparsify_box(
        lower, upper, sparsity_width_ratio=0.95, max_features=1
    )
    assert active.tolist() == [False, True, False]
    assert sparse_lo.tolist() == pytest.approx([0.0, 0.2, 0.0])
    assert sparse_up.tolist() == pytest.approx([1.0, 0.3, 1.0])


def test_c14_global_ruleset_conflict_and_abstain():
    y = np.array([0, 0, 1, 1, 0, 1])
    y_hat = y.copy()
    mask0 = np.array([True, True, True, False, False, False])
    mask1 = np.array([False, False, True, True, False, False])
    res = evaluate_ruleset_as_classifier(
        {0: mask0, 1: mask1},
        {0: 0.9, 1: 0.8},
        y, y_hat,
    )
    assert res.n_abstain == 2
    assert res.n_conflict == 1
    assert res.coverage == pytest.approx(4 / 6)


def test_c16_jaccard_overlap_is_symmetric():
    def jaccard(lo_a, up_a, lo_b, up_b):
        inter_lo = np.maximum(lo_a, lo_b)
        inter_up = np.minimum(up_a, up_b)
        inter = float(np.prod(np.maximum(inter_up - inter_lo, 0.0)))
        vol_a = float(np.prod(np.maximum(up_a - lo_a, 1e-9)))
        vol_b = float(np.prod(np.maximum(up_b - lo_b, 1e-9)))
        union = vol_a + vol_b - inter
        return inter / union if union > 0 else 0.0

    rng = np.random.default_rng(1)
    for _ in range(20):
        lo_a = rng.random(4)
        up_a = lo_a + rng.random(4) * 0.5
        lo_b = rng.random(4)
        up_b = lo_b + rng.random(4) * 0.5
        w = 0.1
        p_ab = w * jaccard(lo_a, up_a, lo_b, up_b)
        p_ba = w * jaccard(lo_b, up_b, lo_a, up_a)
        assert p_ab == pytest.approx(p_ba)


def test_c13_unit_to_original_roundtrip():
    x = np.array([0.0, 0.5, 1.0])
    x_min, x_range = np.array([-1.0, -1.0, -1.0]), np.array([2.0, 2.0, 2.0])
    mean, scale = np.array([3.0, 3.0, 3.0]), np.array([2.0, 2.0, 2.0])
    orig = unit_to_original(x, x_min, x_range, mean, scale)
    assert orig[1] == pytest.approx(3.0)
    assert orig[0] == pytest.approx(1.0)


def test_rules_from_anchors_uses_normalized_bounds_for_unit_space():
    anchors = [{
        "rule": "f0 in original units",
        "lower_bounds": [10.0],
        "upper_bounds": [20.0],
        "lower_bounds_normalized": [0.25],
        "upper_bounds_normalized": [0.75],
    }]
    X_unit = np.array([[0.1], [0.5], [0.9]], dtype=np.float32)
    y = np.array([0, 1, 1])
    y_hat = np.array([0, 1, 1])
    rules = rules_from_anchors(
        anchors, X_unit, y, y_hat, target_class=1,
        class_conditional=True, min_support=1,
        ranking_formula="precision_coverage", space="unit",
    )
    assert len(rules) == 1
    assert rules[0].mask.tolist() == [False, True, False]
    assert rules[0].lower.tolist() == pytest.approx([0.25])


def test_validation_selection_is_preserved_on_test():
    y_val = np.array([1, 1, 0, 0])
    y_hat_val = np.array([1, 1, 0, 0])
    X_val = np.array([[0.1], [0.2], [0.8], [0.9]], dtype=np.float32)
    anchors = [
        {"rule_id": "val-best", "lower_bounds_normalized": [0.0], "upper_bounds_normalized": [0.3]},
        {"rule_id": "test-best", "lower_bounds_normalized": [0.7], "upper_bounds_normalized": [1.0]},
    ]
    ranked_val = rules_from_anchors(
        anchors, X_val, y_val, y_hat_val, target_class=1,
        class_conditional=True, min_support=1,
        ranking_formula="precision_coverage", space="unit",
    )
    selected_val = select_topk_union(
        ranked_val, y_val, y_hat_val, target_class=1, k=1,
        class_conditional=True, min_support=1,
    )
    assert selected_val.best.rule_id == "val-best"

    X_test = np.array([[0.15], [0.85], [0.95]], dtype=np.float32)
    y_test = np.array([0, 1, 1])
    y_hat_test = np.array([0, 1, 1])
    report_rules = reevaluate_ranked_rules(
        selected_val.individual, X_test, y_test, y_hat_test, target_class=1,
        class_conditional=True, min_support=1,
    )
    report = select_topk_union(
        report_rules, y_test, y_hat_test, target_class=1, k=1,
        class_conditional=True, min_support=1,
    )
    assert report.best.rule_id == "val-best"


# ---------------------------------------------------------------------------
# A1 — rule ranking must account for statistical support
# ---------------------------------------------------------------------------

def _rule(rule_id, mask, y, y_hat, target_class=1):
    m = evaluate_mask(y, y_hat, mask, target_class=target_class, class_conditional=True)
    return RankedRule(
        rule_id=rule_id, lower=None, upper=None, mask=mask, metrics=m,
        score=ranking_score(m.fidelity, m.coverage, n_covered=m.n_covered),
    )


def _one_vs_many():
    """One perfect-on-a-single-sample rule vs one well-supported imperfect rule."""
    n = 100
    y = np.zeros(n, dtype=int)
    y[:50] = 1
    y_hat = y.copy()
    tiny = np.zeros(n, dtype=bool)
    tiny[0] = True                    # 1 covered sample, fidelity 1.0
    broad = np.zeros(n, dtype=bool)
    broad[:30] = True                 # 30 covered samples, fidelity 1.0 but far more coverage
    noisy = np.zeros(n, dtype=bool)
    noisy[:25] = True
    noisy[50:52] = True               # 27 covered, fidelity ~0.93
    return y, y_hat, tiny, broad, noisy


def test_a1_single_sample_rule_does_not_outrank_supported_rule():
    y, y_hat, tiny, _, noisy = _one_vs_many()
    r_tiny = _rule("tiny", tiny, y, y_hat)
    r_noisy = _rule("noisy", noisy, y, y_hat)
    assert r_tiny.metrics.n_covered == 1
    assert r_tiny.metrics.fidelity == 1.0          # perfect point estimate...
    assert r_noisy.metrics.fidelity < 1.0          # ...vs an imperfect but supported rule
    # Old behaviour ranked the single-sample rule first; the LCB score must not.
    assert r_noisy.score > r_tiny.score


def test_a1_min_support_filters_under_supported_rules():
    y, y_hat, tiny, broad, _ = _one_vs_many()
    rules = [_rule("tiny", tiny, y, y_hat), _rule("broad", broad, y, y_hat)]
    union = select_topk_union(rules, y, y_hat, target_class=1, k=1, min_support=10)
    assert union is not None
    assert union.selected_ids == ["broad"], "under-supported rule must be filtered out"


def test_a1_min_support_never_returns_empty_selection():
    """If every candidate is under-supported we still return the best-supported ones."""
    n = 40
    y = np.zeros(n, dtype=int); y[:20] = 1
    y_hat = y.copy()
    m1 = np.zeros(n, dtype=bool); m1[0] = True
    m2 = np.zeros(n, dtype=bool); m2[1] = True
    rules = [_rule("a", m1, y, y_hat), _rule("b", m2, y, y_hat)]
    union = select_topk_union(rules, y, y_hat, target_class=1, k=2, min_support=10)
    assert union is not None and union.n_selected >= 1


def test_a1_reporting_side_does_not_refilter_on_support():
    """enforce_min_support=False keeps the validation-selected set intact."""
    y, y_hat, tiny, broad, _ = _one_vs_many()
    rules = [_rule("tiny", tiny, y, y_hat), _rule("broad", broad, y, y_hat)]
    union = select_topk_union(
        rules, y, y_hat, target_class=1, k=2, min_support=10, enforce_min_support=False,
    )
    assert union is not None
    assert set(union.selected_ids) == {"tiny", "broad"}
