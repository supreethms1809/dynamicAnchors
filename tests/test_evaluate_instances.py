"""Unit tests for Track B helpers (no trained policies required)."""
from __future__ import annotations

import numpy as np

from revision.evaluate_instances import (
    _json_float,
    _maybe_call,
    box_contains,
    break_even_n,
    class_boxes_from_track_a,
    lookup_fired_classes,
    perturb_precision,
    sample_test_indices,
    summarize_instance_rows,
    unit_to_std,
)


def test_sample_test_indices_stratifies_and_caps():
    y_hat = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
    idx = sample_test_indices(y_hat, max_per_pred=2, seed=0)
    picked = y_hat[idx]
    assert int((picked == 0).sum()) == 2
    assert int((picked == 1).sum()) == 2
    assert int((picked == 2).sum()) == 2
    all_idx = sample_test_indices(y_hat, max_per_pred=None, seed=0)
    assert all_idx.tolist() == list(range(9))


def test_perturb_precision_pins_active_dims():
    X = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], dtype=np.float32)
    x_star = np.array([0.5, 0.4], dtype=np.float32)
    captured = {}

    def predict(X_std):
        captured["z"] = X_std.copy()
        return np.ones(len(X_std), dtype=int)

    rng = np.random.default_rng(0)
    p, n = perturb_precision(
        X_train_unit=X, x_star_unit=x_star,
        lower=np.array([0.4, 0.0]), upper=np.array([0.6, 1.0]),
        active=np.array([True, False]),
        predict_std=predict,
        x_min=np.zeros(2), x_range=np.ones(2),
        y_hat_star=1, n_perturb=4, rng=rng,
    )
    assert n == 3  # cap at train size
    assert abs(p - 1.0) < 1e-9
    assert np.allclose(captured["z"][:, 0], 0.5)


def test_lookup_conflict_and_abstain():
    x = np.array([0.2, 0.2], dtype=np.float32)
    boxes = {
        0: [(np.array([0.0, 0.0]), np.array([0.5, 0.5]))],
        1: [(np.array([0.0, 0.0]), np.array([0.3, 0.3]))],
        2: [(np.array([0.8, 0.8]), np.array([1.0, 1.0]))],
    }
    fired = lookup_fired_classes(x, boxes)
    assert fired == [0, 1]
    assert not box_contains(x, *boxes[2][0])
    empty = lookup_fired_classes(np.array([0.9, 0.1]), boxes)
    assert empty == []


def test_break_even():
    assert abs(break_even_n(1000, 10, 60) - (1000 / 50)) < 1e-9
    assert break_even_n(1000, 80, 60) is None


def test_class_boxes_from_track_a_and_summary():
    track = {
        "per_class": {
            "class_0": {
                "selected_rules": [
                    {"lower_bounds": [0.0, 0.0], "upper_bounds": [0.5, 0.5],
                     "report_metrics": {"fidelity": 1.0}},
                ]
            }
        }
    }
    boxes = class_boxes_from_track_a(track)
    assert 0 in boxes and len(boxes[0]) == 1
    rows = [
        {"perturb_fid": 0.9, "emp_fid": 0.8, "f_correct": True, "contains_x": 1.0,
         "empty_rule": 0.0, "queries": 10, "n_covered": 12},
        {"perturb_fid": 0.5, "emp_fid": 0.4, "f_correct": False, "contains_x": 0.0,
         "empty_rule": 0.0, "queries": 20, "n_covered": 2},
    ]
    s = summarize_instance_rows(rows)
    assert s["all"]["n"] == 2
    assert s["f_correct"]["n"] == 1
    assert s["f_wrong"]["n"] == 1
    assert abs(s["all"]["queries_per_x"] - 15.0) < 1e-9


def test_unit_to_std():
    out = unit_to_std(np.array([[0.0, 1.0]]), np.array([-1.0, 0.0]), np.array([2.0, 4.0]))
    assert np.allclose(out, np.array([[-1.0, 4.0]]))


def test_maybe_call_anchor_precision_methods():
    class _Exp:
        def precision(self, partial_index=None):
            return 0.97
        def coverage(self):
            return 0.18
        names = ["petal width <= 0.8"]

    exp = _Exp()
    assert abs(_json_float(_maybe_call(exp.precision)) - 0.97) < 1e-9
    assert abs(_json_float(_maybe_call(exp.coverage)) - 0.18) < 1e-9
    assert _maybe_call(exp.names) == ["petal width <= 0.8"]
