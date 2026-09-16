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
