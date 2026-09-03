"""Compactness must be measured against the space the box actually lives in.

The baselines (CART, greedy/SP Anchors) build boxes in ORIGINAL feature units
and seed the root box at X_train's min/max. `active_feature_mask` defaults to a
unit-space range of [0, 1], so without an explicit range a 3.6 cm-wide iris
dimension reads as "narrow, therefore constrained" and a genuinely one-sided
split reported 0 active features -- while breast_cancer, whose features have
small numeric magnitudes, reported every feature as active.
"""
import numpy as np
import pytest

from utils.metrics import active_feature_mask, compactness_of_box


# iris train-split ranges, original units (sepal len, sepal wid, petal len, petal wid)
IRIS_MIN = np.array([4.3, 2.0, 1.1, 0.1])
IRIS_MAX = np.array([7.9, 4.4, 6.9, 2.5])


def _cart_style_box():
    """`petal length (cm) <= 2.45` -- one constrained dimension, three free."""
    lower = IRIS_MIN.copy()
    upper = IRIS_MAX.copy()
    upper[2] = 2.45
    return lower, upper


def test_one_sided_iris_box_reports_one_active_feature():
    lower, upper = _cart_style_box()
    out = compactness_of_box(
        lower, upper, feature_min=IRIS_MIN, feature_max=IRIS_MAX
    )
    assert out["n_active_features"] == 1, (
        "the petal-length split is the only constrained dimension"
    )
    assert out["n_features"] == 4


def test_unit_space_default_is_wrong_for_original_units():
    """Guards the regression itself: without a range the same box reports 0."""
    lower, upper = _cart_style_box()
    assert compactness_of_box(lower, upper)["n_active_features"] == 0


def test_unconstrained_box_has_no_active_features():
    out = compactness_of_box(
        IRIS_MIN, IRIS_MAX, feature_min=IRIS_MIN, feature_max=IRIS_MAX
    )
    assert out["n_active_features"] == 0


def test_small_magnitude_features_are_not_all_active():
    """breast_cancer-style: tiny numeric ranges read as active under the default."""
    fmin = np.array([0.0, 0.0, 0.0])
    fmax = np.array([0.29, 0.20, 0.10])
    lower, upper = fmin.copy(), fmax.copy()
    upper[0] = 0.145  # one real split
    assert compactness_of_box(lower, upper)["n_active_features"] == 3
    out = compactness_of_box(lower, upper, feature_min=fmin, feature_max=fmax)
    assert out["n_active_features"] == 1


def test_unit_space_boxes_are_unaffected():
    """The RL arms pass no range and must keep their existing behaviour."""
    lower = np.array([0.0, 0.3, 0.0])
    upper = np.array([1.0, 0.6, 1.0])
    assert compactness_of_box(lower, upper)["n_active_features"] == 1
    explicit = compactness_of_box(
        lower, upper, feature_min=np.zeros(3), feature_max=np.ones(3)
    )
    assert explicit["n_active_features"] == 1


@pytest.mark.parametrize("ratio", [0.5, 0.95, 1.0])
def test_ratio_still_applies_with_an_explicit_range(ratio):
    lower, upper = _cart_style_box()
    mask = active_feature_mask(
        lower, upper, sparsity_width_ratio=ratio,
        feature_min=IRIS_MIN, feature_max=IRIS_MAX,
    )
    # petal length keeps 1.35/5.8 = 23% of its range: constrained at every ratio
    assert bool(mask[2]) is True
    # the untouched dimensions sit at exactly 100% of range
    assert not mask[0] and not mask[1] and not mask[3]
