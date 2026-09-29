"""Len counts a condition when it excludes at least one D_train row, and the printer
shows exactly those conditions.

The old count kept a feature when its interval was narrower than 95% of the
feature's range. That dropped real conditions (a lower face at the 3rd percentile
of a skewed feature removes rows but leaves 97% of the range), and the RL printer,
which used the quantile-space mask instead, printed vacuous ones such as
`MAR in [1, 5]` on a feature whose range is 1-5.
"""
import numpy as np

from utils.eval_harness import per_class_block
from utils.metrics import (
    RankedRule, box_mask, compactness_of_ruleset, condition_mask, evaluate_mask,
    select_topk_union, train_span,
)
from utils.rule_print import RulePrinter, fmt_value

rng = np.random.default_rng(0)
N = 500
# 0: skewed continuous; 1: integer 1..5 (MAR-like); 2: nominal codes 0..3; 3: binary 0/1
X = np.column_stack([
    rng.exponential(1.0, N),
    rng.integers(1, 6, N),
    rng.integers(0, 4, N),
    rng.integers(0, 2, N),
]).astype(np.float32)
NAMES = ["income", "MAR", "workclass", "flag"]
CATS = {2: ["Gov", "Private", "Self", "Never"]}
PR = RulePrinter(X, X, NAMES, categorical_indices=[2], categorical_values=CATS)
LO, HI = X.min(0), X.max(0)


def box(**faces):
    lo, hi = LO.copy(), HI.copy()
    for k, (a, b) in faces.items():
        j = NAMES.index(k)
        lo[j], hi[j] = a, b
    return lo, hi


def test_face_on_the_train_range_is_not_a_condition():
    lo, hi = box(MAR=(1, 5))
    assert not condition_mask(lo, hi, train_span(X)).any()
    assert PR(lo, hi) == "any values"


def test_narrow_or_wide_faces_count_iff_they_exclude_rows():
    q03 = float(np.quantile(X[:, 0], 0.03))
    lo, hi = box(income=(q03, HI[0]))
    # 97% of the range: the 95%-width rule missed this real condition
    assert (hi[0] - lo[0]) > 0.95 * (HI[0] - LO[0])
    assert condition_mask(lo, hi, train_span(X))[0]
    assert box_mask(X, lo, hi).sum() < N


def test_face_one_float32_step_inside_counts():
    lo, hi = LO.copy(), HI.copy()
    hi[0] = np.nextafter(np.float32(HI[0]), np.float32(-np.inf))
    assert condition_mask(lo, hi, train_span(X))[0]
    lo2, hi2 = LO.copy(), HI.copy()
    hi2[0] = np.float32(1e30)          # an open face (CART leaf) excludes nothing
    lo2[0] = -np.float32(1e30)
    assert not condition_mask(lo2, hi2, train_span(X)).any()


def test_integer_feature_prints_observed_bounds():
    lo, hi = box(MAR=(1.0, 4.371))
    assert PR(lo, hi) == "MAR <= 4"
    lo, hi = box(MAR=(2.9, 3.2))
    assert PR(lo, hi) == "MAR = 3"


def test_anchors_strict_face_on_a_binary_feature():
    # Anchors' `flag > 0` becomes the smallest float32 above 0; print the value it admits.
    lo, hi = box(flag=(np.nextafter(np.float32(0), np.float32(1)), 1))
    assert PR(lo, hi) == "flag >= 1"


def test_nominal_feature_prints_its_codes_by_name():
    lo, hi = box(workclass=(1, 1))
    assert PR(lo, hi) == "workclass = 'Private'"
    lo, hi = box(workclass=(0, 1.5))
    assert PR(lo, hi) == "workclass in {'Gov', 'Private'}"


def test_one_sided_and_two_sided_continuous():
    lo, hi = box(income=(0.25, HI[0]))
    assert PR(lo, hi) == "income >= 0.25"
    lo, hi = box(income=(0.25, 1.5))
    assert PR(lo, hi) == "0.25 <= income <= 1.5"


def test_printed_length_equals_len():
    for _ in range(200):
        lo, hi = LO.copy(), HI.copy()
        for j in range(X.shape[1]):
            a, b = np.sort(rng.choice(X[:, j], 2))
            if rng.random() < 0.5:
                lo[j] = a
            if rng.random() < 0.5:
                hi[j] = b
        shown = PR(lo, hi)
        n = 0 if shown == "any values" else len(shown.split(" and "))
        assert n == PR.count(lo, hi) == int(condition_mask(lo, hi, train_span(X)).sum())


def test_unit_space_boxes_print_in_original_units():
    Xu = (X - LO) / (HI - LO)
    pr = RulePrinter(Xu, X, NAMES, categorical_indices=[2], categorical_values=CATS,
                     to_orig=lambda b: b * (HI - LO) + LO)
    lo, hi = np.zeros(4, np.float32), np.ones(4, np.float32)
    hi[0] = 0.5
    lo[1] = (2 - LO[1]) / (HI[1] - LO[1])
    assert pr(lo, hi) == f"income <= {fmt_value(0.5 * (HI[0] - LO[0]) + LO[0])} and MAR >= 2"


def test_fmt_value():
    assert fmt_value(3.0) == "3"
    assert fmt_value(5.19) == "5.19"
    assert fmt_value(0.012345) == "0.01235"
    assert fmt_value(1234567.8) == "1234568"
    assert fmt_value(-17.93) == "-17.93"


def test_per_class_block_reprints_and_reports_len():
    y_hat = (X[:, 1] <= 2).astype(int)
    rules = []
    for i, (lo, hi) in enumerate([box(MAR=(1, 2.5), flag=(0, 1)), box(MAR=(1, 2.2), income=(0, 3))]):
        m = box_mask(X, lo, hi)
        met = evaluate_mask(y=y_hat, y_hat=y_hat, mask=m, target_class=1,
                            class_conditional=True, min_support=1)
        rules.append(RankedRule(rule_id=str(i), lower=lo, upper=hi, mask=m, metrics=met,
                                score=float(i), display_rule="MAR in [1.000000, 2.500000] and flag in [0, 1]"))
    union = select_topk_union(rules, y_hat, y_hat, 1, k=2, class_conditional=True,
                              min_support=1, enforce_min_support=False)
    blk = per_class_block(union, printer=PR)
    shown = [r["display_rule"] for r in blk["selected_rules"]]
    assert "MAR <= 2" in shown
    assert all("flag" not in s for s in shown)       # flag in [0, 1] excludes nothing
    assert [r["n_conditions"] for r in blk["selected_rules"]] == [PR.count(r.lower, r.upper) for r in union.individual]
    comp = blk["compactness"]
    assert comp["len_criterion"] == "excludes_train_row"
    assert comp["mean_conditions"] == np.mean([r["n_conditions"] for r in blk["selected_rules"]])
    old = compactness_of_ruleset(union.individual)
    assert "mean_conditions" not in old            # without a span only the old count


def test_strict_bounds_stay_strict_in_original_units():
    # Anchors' `income > 0.5` is the smallest float32 above 0.5 (see revision.baselines._f32_gt)
    lo, hi = box(income=(np.nextafter(np.float32(0.5), np.float32(1)), HI[0]))
    assert PR(lo, hi) == "income > 0.5"
    # a CART left child excludes its float32 threshold: x < t
    lo, hi = box(income=(LO[0], np.nextafter(np.float32(1.25), np.float32(-np.inf))))
    assert PR(lo, hi) == "income < 1.25"
    lo, hi = box(income=(LO[0], np.float32(1.25)))
    assert PR(lo, hi) == "income <= 1.25"


def test_large_integer_feature_rounds_the_face_not_to_a_train_value():
    Xw = np.column_stack([rng.integers(10_000, 1_500_000, 2000)]).astype(np.float32)
    pr = RulePrinter(Xw, Xw, ["fnlwgt"])
    face = np.float32(327389.6)
    assert pr(Xw.min(0), np.array([face], np.float32)) == "fnlwgt <= 327389"


def test_strict_face_on_an_integer_feature_prints_the_admitted_value():
    lo, hi = box(MAR=(np.nextafter(np.float32(3), np.float32(9)), HI[1]))
    assert PR(lo, hi) == "MAR >= 4"
