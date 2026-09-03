"""Regression tests for categorical freeze-on-reset, centroid snap, hard Fid."""
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
from revision.evaluate import _pool_class_anchors  # noqa: E402


class LowConfClf(nn.Module):
    """Argmax class 0 with softmax ≈ 0.55 so blend(λ=0.5) is well below hard Fid."""

    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        self.lin = nn.Linear(n_features, n_classes)
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.bias.zero_()
            self.lin.bias[0] = 0.2

    def forward(self, x):
        return self.lin(x)


def _clustered_data(n_features=13, n_per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    c0 = 0.25 + 0.03 * rng.standard_normal((n_per_class, n_features))
    c1 = 0.75 + 0.03 * rng.standard_normal((n_per_class, n_features))
    X = np.clip(np.vstack([c0, c1]), 0.0, 1.0).astype(np.float32)
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=int)
    return X, y


def _make_env(X, y, extra_cfg=None, target_class=0):
    n_features = X.shape[1]
    cfg = {
        "max_cycles": 20,
        "min_width": 0.05,
        "initial_window": 0.1,
        "strict_target_termination": True,
        "require_coverage_gain_to_terminate": True,
        "centroid_snap_threshold": 0.0,
        "precision_target": 0.9,
        "coverage_target": 0.2,
        "precision_estimator": "empirical",
        "precision_blend_lambda": 0.5,
        "categorical_freeze": "instance",
        "mode": "training",
        "use_class_centroids": True,
        "training_instance_ratio": 0.0,
        "X_min": np.zeros(n_features, dtype=np.float32),
        "X_range": np.ones(n_features, dtype=np.float32),
    }
    if extra_cfg:
        cfg.update(extra_cfg)
    env = SingleAgentAnchorEnv(
        X_unit=X,
        X_std=X,
        y=y,
        feature_names=[f"f{i}" for i in range(n_features)],
        classifier=LowConfClf(n_features),
        target_class=target_class,
        env_config=cfg,
    )
    return env


def test_categorical_freeze_on_reset():
    X, y = _clustered_data(n_features=4)
    x_star = X[0].copy()
    env = _make_env(
        X, y,
        extra_cfg={
            "mode": "inference",
            "categorical_indices": [0],
            "precision_estimator": "conditional",
            "reset_diversity_frac": 0.0,
            "n_reset_landings": 0,
        },
    )
    env.x_star_unit = x_star
    env.reset()
    assert env.n_predicates() == 0
    width0 = float(env.upper[0] - env.lower[0])
    assert width0 > 0.5, width0
    a = env.a.copy()
    b = env.b.copy()
    a[0] = 0.2
    b[0] = 0.8
    env.set_quantile_box(a, b)
    assert env.n_predicates() >= 1
    width0 = float(env.upper[0] - env.lower[0])
    assert width0 <= 2.5e-4 or abs((env.a[0] + env.b[0]) / 2 - env.q_star[0]) < 1.0
    assert env.lower[0] <= x_star[0] <= env.upper[0]


def test_mean_centroid_snaps_below_old_0p5_threshold():
    env = _make_env(*_clustered_data())
    class_data = np.array([[0.0, 0.0], [0.0, 0.1], [0.1, 0.0]], dtype=np.float32)
    mean = class_data.mean(axis=0)
    dist = float(np.linalg.norm(class_data - mean, axis=1).min())
    assert 0.0 < dist < 0.5
    snapped = env._snap_to_nearest_class_point(mean, class_data)
    nn = class_data[np.argmin(np.linalg.norm(class_data - mean, axis=1))]
    assert np.allclose(snapped, nn)


def test_control_precision_is_hard_fid_not_softmax_blend():
    X, y = _clustered_data()
    env = _make_env(X, y, extra_cfg={"mode": "inference", "precision_blend_lambda": 0.5})
    env.x_star_unit = X[0]
    obs, info = env.reset()
    precision, coverage, details = env._current_metrics()
    hard = details["hard_precision"]
    proxy = details["precision_proxy"]
    assert hard == pytest.approx(1.0)
    assert proxy < 0.85, proxy
    assert precision == pytest.approx(hard)
    # observation packs hard Fid, not the blend
    # quantile layout is [a, b, q*, P, C, mode, phase] since G-12, so P is at -4
    assert float(obs[-4]) == pytest.approx(hard)


def test_pool_class_anchors_keeps_class_based_sibling():
    pcr = {
        "class_0": {
            "anchors": [{
                "lower_bounds": [0.0, 0.0],
                "upper_bounds": [0.1, 0.1],
                "lower_bounds_normalized": [0.0, 0.0],
                "upper_bounds_normalized": [0.1, 0.1],
            }]
        },
        "class_0_class_based": {
            "anchors": [{
                "lower_bounds": [0.2, 0.2],
                "upper_bounds": [0.8, 0.8],
                "lower_bounds_normalized": [0.2, 0.2],
                "upper_bounds_normalized": [0.8, 0.8],
            }]
        },
    }
    pooled = _pool_class_anchors(pcr, 0)
    assert len(pooled) == 2


def test_instance_coverage_is_class_conditional_not_marginal():
    X, y = _clustered_data(n_per_class=40)
    env = _make_env(X, y, extra_cfg={"mode": "inference"})
    env.x_star_unit = X[0]
    env.reset()
    a = np.zeros(env.n_features, dtype=np.float64)
    b = np.ones(env.n_features, dtype=np.float64)
    a[0], b[0] = 0.0, 0.5
    env.set_quantile_box(a, b)
    precision, coverage, details = env._current_metrics()
    mask = ((X >= env.lower) & (X <= env.upper)).all(axis=1)
    class_mask = y == 0
    expected = float((mask & class_mask).sum() / class_mask.sum())
    marginal = float(mask.mean())
    assert coverage == pytest.approx(expected)
    assert details["coverage_marginal"] == pytest.approx(marginal)
    assert expected != pytest.approx(marginal)


def test_training_and_inference_only_enable_both_targets_met():
    X, y = _clustered_data()
    train_env = _make_env(X, y, extra_cfg={"mode": "training"})
    train_env.reset()
    assert train_env.termination_reason_enabled["both_targets_met"] is True
    assert train_env.termination_reason_enabled["excellent_precision"] is False
    assert train_env.termination_reason_enabled["high_precision_reasonable_coverage"] is False
    assert train_env.termination_reason_enabled["both_reasonably_close"] is False

    inf_env = _make_env(X, y, extra_cfg={"mode": "inference"})
    inf_env.reset()
    assert inf_env.termination_reason_enabled["both_targets_met"] is True
    assert inf_env.termination_reason_enabled["excellent_precision"] is False
    assert inf_env.termination_reason_enabled["both_reasonably_close"] is False


def test_no_terminal_bonus_without_coverage_gain():
    X, y = _clustered_data()
    env = _make_env(
        X, y,
        extra_cfg={
            "mode": "training",
            "precision_target": 0.0,
            "coverage_target": 0.0,
            "min_steps_before_termination": 2,
            "require_coverage_gain_to_terminate": True,
            "require_precision_gain_to_terminate": False,
            "strict_target_termination": True,
            "terminal_bonus": 5.0,
            "precision_estimator": "empirical",
        },
    )
    env.reset()
    env.precision_target_effective = 0.0
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    # Empty start (k=0) is not a rule, so both_targets_met cannot fire.
    done = False
    reward = 0.0
    info = {}
    for _ in range(2):
        _, reward, done, _, info = env.step(action)
    assert done is False
    assert info["n_predicates"] == 0.0
    assert reward < 5.0

    env.reset()
    env.precision_target_effective = 0.0
    env.coverage_target = 0.0
    a = np.zeros(env.n_features, dtype=np.float64)
    b = np.ones(env.n_features, dtype=np.float64)
    a[0], b[0] = 0.0, 0.5
    env.set_quantile_box(a, b)
    env._coverage_at_reset = -1.0
    env._coverage_gain_eps = 0.0
    for _ in range(2):
        _, reward, done, _, info = env.step(action)
    assert done is True
    assert info["termination_reason"] == 1.0
    assert env.n_predicates() >= 1
    assert reward >= 3.0
