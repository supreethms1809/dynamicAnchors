"""Anchors predicates must round-trip into box bounds without losing a side.

`anchor-exp` discretizes continuous features, so ~13% of the predicates it emits
on the paper datasets are two-sided bins (`a < feat <= b`) with the lower value
on the *left* of the feature name. The original parser only matched
`feat <op> value`, so every such lower bound was dropped and the baseline box
stayed at the training minimum on that feature — strictly wider than the anchor
it was supposed to represent.
"""
import numpy as np
import pytest

from revision.baselines import _apply_anchor_predicate

NAMES = ["age", "hours-per-week", "temp", "road_temp_set_1", "feature_6"]
LO0, UP0 = -100.0, 100.0


def _apply(pred):
    lo = np.full(len(NAMES), LO0)
    up = np.full(len(NAMES), UP0)
    _apply_anchor_predicate(pred, NAMES, lo, up)
    return lo, up


@pytest.mark.parametrize("pred,idx,expect_lo,expect_up", [
    ("feature_6 <= -0.57", 4, LO0, -0.57),
    ("age > 31.00", 0, np.nextafter(31.0, np.inf), UP0),
    ("age >= 31.00", 0, 31.0, UP0),
    # the regression: value on the left of the name
    ("-1.59 < feature_6 <= -0.57", 4, np.nextafter(-1.59, np.inf), -0.57),
    ("-1.59 <= feature_6 <= -0.57", 4, -1.59, -0.57),
    ("1.2e1 < age <= 4.5e1", 0, np.nextafter(12.0, np.inf), 45.0),
    ("feature_6 = 3", 4, 3.0, 3.0),
])
def test_predicate_binds_both_sides(pred, idx, expect_lo, expect_up):
    lo, up = _apply(pred)
    assert lo[idx] == pytest.approx(expect_lo)
    assert up[idx] == pytest.approx(expect_up)


@pytest.mark.parametrize("pred,idx", [
    ("feature_6 <= -0.57", 4),
    ("-1.59 < feature_6 <= -0.57", 4),
    ("age > 31.00", 0),
])
def test_predicate_touches_only_its_own_feature(pred, idx):
    lo, up = _apply(pred)
    for j in range(len(NAMES)):
        if j == idx:
            continue
        assert lo[j] == LO0 and up[j] == UP0, f"leaked onto {NAMES[j]}"


def test_longer_feature_name_wins_over_substring():
    """`road_temp_set_1` must not also bind the feature named `temp`."""
    lo, up = _apply("road_temp_set_1 <= 5.0")
    assert up[NAMES.index("road_temp_set_1")] == pytest.approx(5.0)
    assert up[NAMES.index("temp")] == UP0


def test_two_sided_bin_is_narrower_than_one_sided():
    """The whole point: the two-sided form must not degrade to the one-sided box."""
    _, up_one = _apply("feature_6 <= -0.57")
    lo_two, up_two = _apply("-1.59 < feature_6 <= -0.57")
    i = NAMES.index("feature_6")
    assert up_two[i] == pytest.approx(up_one[i])
    assert lo_two[i] > LO0


def test_strict_lower_bound_survives_float32():
    """`x > 0` must not become `x >= 0` when the box is float32 (binary features)."""
    lo = np.full(len(NAMES), -100.0, dtype=np.float32)
    up = np.full(len(NAMES), 100.0, dtype=np.float32)
    _apply_anchor_predicate("age > 0.00", NAMES, lo, up)
    assert lo[0] > np.float32(0.0)
    _apply_anchor_predicate("temp > 31.00", NAMES, lo, up)
    assert not (np.float32(31.0) >= lo[2])


# ---------------------------------------------------------------------------
# Exact bins: the box must hold exactly the rows Anchors' discretizer puts in
# the anchor, including edges that print alike at '%.2f' and tied values.
# ---------------------------------------------------------------------------

anchor_tabular = pytest.importorskip("anchor.anchor_tabular")

from revision.baselines import (  # noqa: E402
    _anchor_conditions_box, _anchor_rule_box, _capture_anchor_conditions, _f32_gt, _f32_le,
)
from utils.metrics import box_mask  # noqa: E402


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.normal(0.06, 0.004, n),              # quartiles collide at '%.2f'
        rng.integers(0, 2, n),                   # binary: `> 0.00` is the whole point
        rng.integers(0, 5, n),                   # ties on the cut points
        rng.normal(50, 10, n),
    ]).astype(np.float32)
    return X, ["small", "flag", "count", "wide"]


def _in_anchor(ex, conditions, X):
    D = ex.disc.discretize(X)
    ok = np.ones(len(X), dtype=bool)
    for f, op, v in conditions:
        ok &= (D[:, f] <= v) if op == "leq" else (D[:, f] > v)
    return ok


def test_f32_faces_bracket_the_cut_point():
    for q in (0.1, 6.4, 0.0617, -0.37787):
        assert float(_f32_le(q)) <= q < float(_f32_gt(q))
        assert float(np.nextafter(_f32_le(q), np.float32(np.inf))) > q


def test_conditions_box_matches_discretizer():
    X, names = _data()
    ex = anchor_tabular.AnchorTabularExplainer(["0", "1"], names, X)
    for conds in ([(0, "geq", 0), (0, "leq", 1)], [(1, "geq", 0)], [(2, "leq", 1), (3, "geq", 2)]):
        lo = np.full(4, -np.inf, dtype=np.float32)
        up = np.full(4, np.inf, dtype=np.float32)
        _anchor_conditions_box(ex, conds, lo, up)
        np.testing.assert_array_equal(box_mask(X, lo, up), _in_anchor(ex, conds, X))


def test_real_anchor_box_matches_discretizer():
    X, names = _data()
    ex = anchor_tabular.AnchorTabularExplainer(["0", "1"], names, X)
    _capture_anchor_conditions(ex)
    clf = lambda Z: ((Z[:, 0] > 0.061) & (Z[:, 1] > 0)).astype(int)  # noqa: E731
    np.random.seed(0)
    x = X[np.flatnonzero(clf(X) == 1)[0]]
    exp = ex.explain_instance(x, clf, threshold=0.95)
    conds = exp.exp_map["conditions"]
    assert conds, "expected a non-empty anchor"
    lo = np.full(4, -np.inf, dtype=np.float32)
    up = np.full(4, np.inf, dtype=np.float32)
    _anchor_conditions_box(ex, conds, lo, up)
    np.testing.assert_array_equal(box_mask(X, lo, up), _in_anchor(ex, conds, X))
    # The printed rule, matched back to the cut points, gives the same box.
    lo2 = np.full(4, -np.inf, dtype=np.float32)
    up2 = np.full(4, np.inf, dtype=np.float32)
    _anchor_rule_box(" and ".join(exp.names()), names, ex, x, lo2, up2)
    np.testing.assert_array_equal(box_mask(X, lo2, up2), box_mask(X, lo, up))
