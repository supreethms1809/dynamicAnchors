"""Tests for revision reporting (C-02, C-11, C-12, C-13, C-26, C-36, C-52)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.eval_harness import (  # noqa: E402
    LEGACY_RULE_TESTER_WARNING,
    audit_selected_unit_rules,
)
from utils.metrics import (  # noqa: E402
    RankedRule,
    break_even_n,
    collect_success_rate,
    compactness_of_box,
    episode_success_rate,
    evaluate_mask,
    paired_wilcoxon,
    ranking_score,
)
from paper.make_tables import (  # noqa: E402
    build_success_table,
    build_table1,
    build_wilcoxon_table,
)
from paper.make_figures import break_even_curves  # noqa: E402


def test_c11_success_rate_counts_episodes_hitting_both_targets():
    anchors = [
        {"precision_rollout_estimated": 0.95, "coverage_class_conditional_rollout_estimated": 0.25, "n_steps": 10},
        {"precision_rollout_estimated": 0.80, "coverage_class_conditional_rollout_estimated": 0.50, "n_steps": 10},
        {"precision_rollout_estimated": 0.99, "coverage_class_conditional_rollout_estimated": 0.10, "n_steps": 10},
    ]
    stats = episode_success_rate(anchors, tau_p=0.90, tau_c=0.20)
    assert stats["n_episodes"] == 3
    assert stats["n_success"] == 1
    assert stats["success_rate"] == pytest.approx(1 / 3)


def test_c11_collect_success_rate_skips_class_based_in_headline():
    rules = {
        "metadata": {"steps_per_episode": 100},
        "per_class_results": {
            "class_0": {
                "anchors": [
                    {"precision_rollout_estimated": 0.95, "coverage_class_conditional_rollout_estimated": 0.3},
                ]
            },
            "class_0_class_based": {
                "anchors": [
                    {"precision_rollout_estimated": 0.99, "coverage_class_conditional_rollout_estimated": 0.9},
                    {"precision_rollout_estimated": 0.99, "coverage_class_conditional_rollout_estimated": 0.9},
                ]
            },
        },
    }
    out = collect_success_rate(rules, 0.9, 0.2)
    assert out["n_episodes"] == 1
    assert out["n_success"] == 1
    assert out["per_class"]["class_0_class_based"]["n_episodes"] == 2


def test_c11_collect_success_rate_walks_mada_per_agent_slots():
    rules = {
        "metadata": {"steps_per_episode": 100},
        "per_class_results": {
            "class_0": {
                "per_agent_results": {
                    "agent_0_0": {
                        "anchors": [
                            {
                                "precision_rollout_estimated": 0.99,
                                "coverage_class_conditional_rollout_estimated": 0.4,
                            }
                        ]
                    }
                }
            }
        },
    }
    out = collect_success_rate(rules, 0.9, 0.2)
    assert out["n_episodes"] == 1
    assert out["n_success"] == 1


def test_c36_compactness_drops_full_range_features():
    c = compactness_of_box(np.array([0.0, 0.2]), np.array([1.0, 0.3]), 0.95)
    assert c["n_active_features"] == 1
    assert c["n_features"] == 2


def test_c12_break_even_n():
    assert break_even_n(fixed_cost=1000, per_instance_cost=1, baseline_per_instance=101) == pytest.approx(10.0)
    assert break_even_n(1000, 10, 5) is None


def test_c26_wilcoxon_effect_size_on_clear_difference():
    a = [0.9, 0.8, 0.85, 0.88, 0.92, 0.87, 0.91, 0.84]
    b = [0.2, 0.3, 0.25, 0.22, 0.28, 0.21, 0.27, 0.24]
    res = paired_wilcoxon(a, b)
    assert res["n"] == 8
    assert res["pvalue"] is not None and res["pvalue"] < 0.05
    assert res["mean_diff"] > 0


def test_print_leg_unwraps_nested_best_metrics():
    from revision.print_leg import _class_summary_line

    line = _class_summary_line(
        "class_0",
        {
            "best": {
                "rule_id": "0:2:...",
                "fidelity": {
                    "fidelity": 1.0,
                    "purity": 1.0,
                    "coverage": 0.3,
                    "n_covered": 3,
                },
            },
            "union": {
                "fidelity": 1.0,
                "purity": 1.0,
                "coverage": 0.5,
                "n_covered": 5,
            },
        },
    )
    assert "best Fid=1.000 Pur=1.000 clsC=0.30 n=3" in line
    assert "union Fid=1.000 Pur=1.000 clsC=0.50 n=5" in line
    assert "—" not in line


def test_make_tables_includes_review_columns(tmp_path):
    block = {
        "k": 5,
        "best": {
            "fidelity": {
                "fidelity": 1.0,
                "purity": 1.0,
                "coverage": 0.5,
                "n_covered": 5,
                "fidelity_ci_low": 0.56,
                "fidelity_ci_high": 1.0,
                "below_min_support": True,
            }
        },
        "union": {
            "fidelity": 1.0,
            "purity": 0.9,
            "coverage": 0.7,
            "n_covered": 8,
            "fidelity_ci_low": 0.60,
            "fidelity_ci_high": 1.0,
            "below_min_support": True,
        },
        "instance": {"coverage": 0.17},
    }
    row = {
        "schema": "rlda_mada_result_v1",
        "dataset": "iris",
        "method": "rlda",
        "seed": 42,
        "tau_p": 0.9,
        "tau_c": 0.2,
        "per_class": {"class_0": block},
        "global_ruleset": {},
        "success_rate": {"n_episodes": 10, "n_success": 4, "success_rate": 0.4},
        "ranking_formula": "precision_coverage",
        "min_support": 10,
    }
    tex = build_table1([row])
    assert r"Cov$_{\mathrm{best}}$" in tex
    assert r"Pur$_\cup$" in tex
    assert "instC" in tex
    assert "0.560" in tex or "0.56" in tex
    assert r"$^\dagger$" in tex
    assert "fid" in tex.lower() or "Ranking" in tex or "C-52" in tex
    succ = build_success_table([row])
    assert "0.400" in succ
    wil = build_wilcoxon_table([row])
    assert "No class-level baseline" in wil


def test_break_even_curves_marks_amortized_methods():
    rows = [
        {
            "schema": "rlda_mada_result_v1",
            "dataset": "iris",
            "method": "rlda",
            "queries": {"n_blackbox_queries": 20000, "n_reporting_queries": 150},
            "classifier_accuracy": {"n": 30},
            "per_class": {},
            "tau_p": 0.9,
            "tau_c": 0.2,
        },
        {
            "schema": "rlda_mada_result_v1",
            "dataset": "iris",
            "method": "sp_anchors",
            "queries": {"n_blackbox_queries": 60000, "n_reporting_queries": 0},
            "classifier_accuracy": {"n": 30},
            "per_class": {},
            "tau_p": 0.9,
            "tau_c": 0.2,
        },
    ]
    curves = break_even_curves(rows, n_max=50)
    assert curves["crossovers"]
    assert curves["crossovers"][0]["n_break_even"] is not None


def test_wine_feature_names_are_chemical_units():
    from BenchMARL.tabular_datasets import TabularDatasetLoader
    loader = TabularDatasetLoader(dataset_name="wine", random_state=0)
    loader.load_dataset()
    assert loader.feature_names[0] != "feature_0"
    assert "alcohol" in [n.lower() for n in loader.feature_names]


def test_legacy_testers_warn_about_revision_harness():
    ma = (REPO / "BenchMARL" / "test_extracted_rules.py").read_text()
    sa = (REPO / "single_agent" / "test_extracted_rules_single.py").read_text()
    assert "revision.evaluate" in ma
    assert "revision.evaluate" in sa
    assert "C-01" in ma and "C-01" in sa
    assert "unique_rules" in LEGACY_RULE_TESTER_WARNING


def test_cart_does_not_reference_undefined_query_budget():
    src = (REPO / "revision" / "baselines.py").read_text()
    # query_budget is a run_anchors_family argument, not run_cart's.
    cart_fn = src.split("def run_cart(")[1].split("def greedy_set_cover")[0]
    assert "query_budget" not in cart_fn


def test_c08_audit_detects_sparsity_coverage_change():
    y = np.array([1, 1, 0])
    y_hat = y.copy()
    X = np.array([[0.1, 0.5], [0.99, 0.5], [0.5, 0.5]], dtype=np.float32)
    lower = np.array([0.0, 0.4], dtype=np.float32)
    upper = np.array([0.96, 0.6], dtype=np.float32)  # dim 0 almost full range
    mask = np.all((X >= lower) & (X <= upper), axis=1)
    metrics = evaluate_mask(y, y_hat, mask, target_class=1, min_support=1)
    rule = RankedRule(
        "r", lower, upper, mask, metrics,
        score=ranking_score(metrics.fidelity, metrics.coverage),
    )
    problems = audit_selected_unit_rules(
        [rule], X,
        sparsity_width_ratio=0.95,
        x_min_std=np.zeros(2),
        x_range_std=np.ones(2),
        scaler_mean=np.zeros(2),
        scaler_scale=np.ones(2),
        feature_min_orig=np.zeros(2),
        feature_max_orig=np.ones(2),
        feature_names=["a", "b"],
    )
    assert problems, "dropping a near-full-range axis should be flagged (C-08)"
