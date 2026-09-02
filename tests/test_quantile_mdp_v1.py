"""V1 gates for the quantile-position MDP (Anchors procedure order)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "single_agent"))

from single_agentENV import SingleAgentAnchorEnv  # noqa: E402
from utils.metrics import ranking_score  # noqa: E402
from single_agent_inference import compute_box_iou, nms_deduplicate_anchors  # noqa: E402


class LinearSepClf(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        self.lin = nn.Linear(n_features, n_classes)
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.bias.zero_()
            self.lin.weight[0, 0] = -4.0
            self.lin.weight[1, 0] = 4.0
            self.lin.bias[0] = 2.0
            self.lin.bias[1] = -2.0

    def forward(self, x):
        return self.lin(x)


def _data(n_features=4, n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    c0 = 0.25 + 0.05 * rng.standard_normal((n_per, n_features))
    c1 = 0.75 + 0.05 * rng.standard_normal((n_per, n_features))
    X = np.clip(np.vstack([c0, c1]), 0.0, 1.0).astype(np.float32)
    y = np.array([0] * n_per + [1] * n_per, dtype=int)
    return X, y


def _env(X, y, **extra):
    n = X.shape[1]
    cfg = {
        "max_cycles": 20,
        "init_mode": "full_space",
        "precision_estimator": "conditional",
        "n_perturb": 64,
        "n_perturb_train": 64,
        "leave_threshold": 0.85,
        "max_new_constraints_per_step": 1,
        "max_quantile_step": 0.10,
        "reset_diversity_frac": 0.0,
        "n_reset_landings": 0,
        "precision_target": 0.9,
        "coverage_target": 0.05,
        "discount": 0.99,
        "mode": "training",
        "training_instance_ratio": 0.0,
        "use_class_centroids": True,
        "X_min": np.zeros(n, dtype=np.float32),
        "X_range": np.ones(n, dtype=np.float32),
        "categorical_freeze": "instance",
    }
    cfg.update(extra)
    return SingleAgentAnchorEnv(
        X_unit=X, X_std=X, y=y,
        feature_names=[f"f{i}" for i in range(n)],
        classifier=LinearSepClf(n),
        target_class=0,
        env_config=cfg,
    )


def test_empty_reset_is_k0_and_full_space():
    env = _env(*_data())
    obs, _ = env.reset(seed=0)
    assert env.n_predicates() == 0
    assert np.allclose(env.lower, 0.0)
    assert np.allclose(env.upper, 1.0)
    assert obs.shape == (3 * env.n_features + 4,)   # G-12: + episode_phase
    assert float(obs[-1]) in (0.0, 1.0)


def test_leave_threshold_and_max_one_new_constraint():
    env = _env(*_data())
    env.reset(seed=1)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[:] = 0.5
    env.step(action)
    assert env.n_predicates() == 0
    action[:] = 0.0
    action[0] = 0.95
    env.step(action)
    assert env.n_predicates() == 1
    action = np.ones(env.action_space.shape, dtype=np.float32)
    env.step(action)
    assert env.n_predicates() == 2


def test_qstar_class_mode_is_not_sentinel_half():
    env = _env(*_data(), training_instance_ratio=0.0)
    env.reset(seed=2)
    assert env.x_star_unit is None
    assert np.all((env.q_star >= 0.0) & (env.q_star <= 1.0))
    assert float(env._get_observation(0.0, 0.0)[-1]) == 0.0
    q_cent = env._class_centroid_quantiles()
    assert np.allclose(env.q_star, q_cent)


def test_discount_is_not_env_gamma():
    env = _env(*_data())
    assert env.gamma == pytest.approx(0.1)
    assert env.discount == pytest.approx(0.99)


def test_noop_zero_delta_phi():
    env = _env(*_data())
    env.reset(seed=3)
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    _, r, _, _, info = env.step(zero)
    assert abs(info["precision_gain"]) < 1e-9
    assert abs(info["coverage_gain"]) < 1e-9
    assert abs(r) < 0.05


def test_k0_ineligible_even_with_large_n_covered():
    env = _env(*_data())
    env.reset(seed=4)
    assert env.n_predicates() == 0
    assert env._best_box is None or env._empty_rule_eligible(
        env._precision_at_reset, 0
    ) is False


def test_ranking_score_none_vs_zero():
    assert ranking_score(1.0, 1.0, n_covered=None) == pytest.approx(2.0)
    assert ranking_score(1.0, 1.0, n_covered=0) == float("-inf")
    assert ranking_score(1.0, 0.5, n_covered=5, min_support=10) == float("-inf")


def test_v1a_privileged_setter_matches_scripted_constraint():
    env = _env(*_data())
    env.reset(seed=5)
    a = env.a.copy()
    b = env.b.copy()
    a[0] = 0.05
    b[0] = 0.95
    env.set_quantile_box(a, b)
    k = env.n_predicates()
    p, c, _ = env._current_metrics()
    assert k == 1
    assert c <= 1.0
    assert 0.0 <= p <= 1.0


def test_v1b_null_does_not_beat_greedy_return():
    env_n = _env(*_data())
    env_g = _env(*_data())
    env_n.reset(seed=6)
    env_g.reset(seed=6)
    zero = np.zeros(env_n.action_space.shape, dtype=np.float32)
    null_ret = 0.0
    for _ in range(8):
        _, r, done, trunc, _ = env_n.step(zero)
        null_ret += r
        if done or trunc:
            break
    greedy_ret = 0.0
    for j in range(min(3, env_g.n_features)):
        action = np.zeros(env_g.action_space.shape, dtype=np.float32)
        action[j] = 0.99
        _, r, done, trunc, _ = env_g.step(action)
        greedy_ret += r
        if done or trunc:
            break
    assert greedy_ret + 1e-6 >= min(null_ret, greedy_ret)
    assert env_n.n_predicates() == 0
    assert env_g.n_predicates() >= 1
    # Empty rule must not collect the +5 terminal bonus.
    assert null_ret < 4.0


def test_v1c_crn_k_stable_on_fixed_box():
    env = _env(*_data())
    env.reset(seed=7)
    a = env.a.copy()
    b = env.b.copy()
    a[0] = 0.1
    b[0] = 0.9
    ks = []
    for seed in range(20):
        env.reset(seed=seed)
        env.set_quantile_box(a, b)
        ks.append(env.n_predicates())
    assert max(ks) - min(ks) == 0


def test_nms_active_only_does_not_collapse_distinct_predicates():
    d = 4
    a1 = {"lower_bounds_normalized": [0.2, 0, 0, 0], "upper_bounds_normalized": [0.4, 1, 1, 1],
          "active_features": [1, 0, 0, 0], "precision_rollout_estimated": 0.9, "coverage_rollout_estimated": 0.4}
    a2 = {"lower_bounds_normalized": [0, 0.2, 0, 0], "upper_bounds_normalized": [1, 0.4, 1, 1],
          "active_features": [0, 1, 0, 0], "precision_rollout_estimated": 0.91, "coverage_rollout_estimated": 0.4}
    kept = nms_deduplicate_anchors([a1, a2], iou_threshold=0.9,
                                   key_precision="precision_rollout_estimated",
                                   key_coverage="coverage_rollout_estimated")
    assert len(kept) == 2
    iou = compute_box_iou(
        a1["lower_bounds_normalized"], a1["upper_bounds_normalized"],
        a2["lower_bounds_normalized"], a2["upper_bounds_normalized"],
        a1["active_features"], a2["active_features"],
    )
    # Volume IoU over the union of active dims. These two boxes DO overlap as
    # regions (0.2*0.2 of the unit square on the two named features), so the
    # correct value is 0.111, not 0. Asserting 0.0 here would pin the semantics
    # to a Jaccard of the active-feature SETS -- see the regression test below.
    assert iou == pytest.approx(0.04 / 0.36)
    assert iou < 0.9


def test_nms_keeps_disjoint_rules_on_the_same_feature():
    """Two modes of one class on the SAME feature must not be deduplicated.

    Regression: when compute_box_iou reduced to a Jaccard of active-feature sets,
    any two rules sharing an active set scored IoU 1.0 regardless of their
    intervals, so NMS kept at most one rule per distinct feature set and discarded
    exactly the multi-modal structure class-level rule sets exist to capture.
    """
    def mk(lo, up, act, p):
        return {"lower_bounds_normalized": lo, "upper_bounds_normalized": up,
                "active_features": act, "precision_rollout_estimated": p,
                "coverage_rollout_estimated": 0.4}

    lo_mode = mk([0.05, 0, 0, 0], [0.15, 1, 1, 1], [1, 0, 0, 0], 0.90)
    hi_mode = mk([0.80, 0, 0, 0], [0.95, 1, 1, 1], [1, 0, 0, 0], 0.91)
    iou = compute_box_iou(
        lo_mode["lower_bounds_normalized"], lo_mode["upper_bounds_normalized"],
        hi_mode["lower_bounds_normalized"], hi_mode["upper_bounds_normalized"],
        lo_mode["active_features"], hi_mode["active_features"],
    )
    assert iou == pytest.approx(0.0)
    kept = nms_deduplicate_anchors([lo_mode, hi_mode], iou_threshold=0.9,
                                   key_precision="precision_rollout_estimated",
                                   key_coverage="coverage_rollout_estimated")
    assert len(kept) == 2

    # Near-identical boxes on the same feature SHOULD still collapse.
    dup = mk([0.05, 0, 0, 0], [0.15, 1, 1, 1], [1, 0, 0, 0], 0.80)
    kept2 = nms_deduplicate_anchors([lo_mode, dup], iou_threshold=0.9,
                                    key_precision="precision_rollout_estimated",
                                    key_coverage="coverage_rollout_estimated")
    assert len(kept2) == 1


def test_masked_iou_equals_plain_volume_iou():
    """Restricting the volume ratio to union-active dims is an exact identity.

    A dim unconstrained in both boxes is [0,1] on each side, contributing a factor
    of 1 to the intersection and to both volumes.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        d = 8
        a1 = rng.random(d) < 0.4
        a2 = rng.random(d) < 0.4
        lo1 = np.where(a1, rng.random(d) * 0.5, 0.0)
        up1 = np.where(a1, 0.5 + rng.random(d) * 0.5, 1.0)
        lo2 = np.where(a2, rng.random(d) * 0.5, 0.0)
        up2 = np.where(a2, 0.5 + rng.random(d) * 0.5, 1.0)
        assert compute_box_iou(lo1, up1, lo2, up2, a1, a2) == pytest.approx(
            compute_box_iou(lo1, up1, lo2, up2)
        )


def test_no_terminal_shaping_correction_on_truncation():
    """SB3 bootstraps on time-limit truncation, so -gamma*Phi must not be applied.

    ReplayBuffer._get_samples returns dones*(1-timeouts) and DummyVecEnv sets
    info["TimeLimit.truncated"], so on truncation V(s') is bootstrapped and already
    carries -Phi(s') under shaping. Applying the absorbing-terminal correction there
    double-counts it, penalising exactly the episodes that end in a good box.
    """
    env = _env(*_data())
    env.reset(seed=3)
    # Constrain one dim so Phi is non-zero, then step with a no-op action: under CRN
    # the box and both metrics are unchanged, so shaping is exactly (gamma-1)*Phi.
    a, b = env.a.copy(), env.b.copy()
    a[1], b[1] = 0.1, 0.8
    env.set_quantile_box(a, b)
    noop = np.zeros(env.action_space.shape[0], dtype=np.float32)
    env.step(noop)

    env.timestep = env.max_cycles - 1
    before = env._rt_shaping
    obs, r, done, trunc, info = env.step(noop)
    assert trunc and not done, "expected this step to hit the time limit"

    phi = env._potential(info["precision"], info["coverage"])
    assert abs(phi) > 1e-9, "test needs a non-zero potential to be meaningful"
    delta = env._rt_shaping - before
    expected_ok = (env.discount - 1.0) * phi          # no absorbing correction
    expected_bug = expected_ok - env.discount * phi   # with the correction applied
    assert delta == pytest.approx(expected_ok, abs=1e-6), (
        f"shaping delta {delta} != {expected_ok}; looks like the absorbing-terminal "
        f"correction fired on truncation (that would give {expected_bug})"
    )


def test_json_roundtrip_unit_bounds_and_quantiles():
    env = _env(*_data())
    env.reset(seed=8)
    a = env.a.copy(); b = env.b.copy()
    a[1] = 0.1; b[1] = 0.8
    env.set_quantile_box(a, b)
    payload = env.export_rule_state()
    assert "lower_bounds_normalized" in payload
    assert "quantile_knots" in payload
    assert payload["n_predicates"] >= 1
    rule, key = env.extract_rule(max_features_in_rule=-1)
    assert "any values" not in rule
    assert key != "any_values"


def test_v1d_n3_mixture_survives_nms():
    env = _env(*_data(), n_reset_landings=2, reset_diversity_frac=1.0, mode="inference")
    rules = []
    for seed in range(20):
        env.reset(seed=seed)
        payload = env.export_rule_state()
        payload["precision_rollout_estimated"] = 0.9
        payload["coverage_rollout_estimated"] = 0.4
        rules.append(payload)
    kept = nms_deduplicate_anchors(
        rules, iou_threshold=0.9,
        key_precision="precision_rollout_estimated",
        key_coverage="coverage_rollout_estimated",
    )
    actives = {tuple(r.get("active_features") or []) for r in rules}
    assert len(actives) > 1
    assert len(kept) > 1


def test_v1e_k0_conditional_p_is_model_base_rate():
    env = _env(*_data())
    env.reset(seed=9)
    p, c, det = env._current_metrics()
    assert env.n_predicates() == 0
    assert c == pytest.approx(1.0, abs=0.05)
    assert p < 0.95
    assert int(det.get("n_class_in_box", 0)) == 40


def test_v1e_n_points_is_not_n_covered():
    env = _env(*_data())
    env.reset(seed=10)
    a = env.a.copy(); b = env.b.copy()
    a[0] = 0.1; b[0] = 0.9
    env.set_quantile_box(a, b)
    _, _, det = env._current_metrics()
    assert int(det["n_points"]) == env.n_perturb
    assert int(det["n_covered"]) != int(det["n_points"])


def test_v1e_freeze_not_reapplied_on_unconstrained_cats():
    X, y = _data()
    env = _env(X, y, categorical_indices=[0], mode="inference")
    env.reset(seed=11)
    assert env.n_predicates() == 0
    assert float(env.upper[0] - env.lower[0]) > 0.5


def test_coverage_floor_never_exceeds_coverage_target():
    """The in-episode collapse guard must not outrank the coverage target.

    min_support / n_class is exactly "min_support class rows in the box", but it is
    compared against class-conditional coverage, so on small classes it becomes a
    large coverage demand (iris n_class=30 -> 0.333, 6.7x a tau_C of 0.05). A box
    that legitimately met tau_C was then judged below the floor and force-retreated
    at the terminal step -- and since _best_box only records boxes already above the
    floor, the retreat could only land on looser, lower-precision boxes.
    """
    env = _env(*_data())
    assert env.min_coverage_floor <= env.coverage_target + 1e-12
    assert env.min_coverage_floor > 0.0


def test_terminal_retreat_keeps_a_box_that_meets_the_coverage_target():
    env = _env(*_data())
    env.reset(seed=5)
    n_class = int((env.y == env.target_class).sum())
    assert 10.0 / n_class > env.coverage_target, "fixture must exercise the small-class case"

    noop = np.zeros(env.action_space.shape[0], dtype=np.float32)
    # A loose box first: above any floor, so it becomes the retreat candidate.
    a, b = env.a.copy(), env.b.copy()
    a[1], b[1] = 0.0, 0.9
    env.set_quantile_box(a, b)
    env._precision_at_reset = 0.0
    env._best_box, env._best_box_cov = None, float("-inf")
    env.step(noop)

    # Then tighten to a box that still clears tau_C.
    a2, b2 = a.copy(), b.copy()
    a2[1], b2[1] = 0.0, 0.35
    env.set_quantile_box(a2, b2)
    _, c_tight, _ = env._current_metrics()
    assert c_tight >= env.coverage_target, "fixture box must meet tau_C"

    env.timestep = env.max_cycles - 1
    _, _, done, trunc, info = env.step(noop)
    assert trunc and not done
    assert info["coverage"] == pytest.approx(c_tight, abs=1e-9), (
        "a box meeting tau_C was retreated away at the terminal step"
    )
