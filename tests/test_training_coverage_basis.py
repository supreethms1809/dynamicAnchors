"""Training C must use f̂ when coverage_basis=predicted (same object as Fid)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "single_agent"))
sys.path.insert(0, str(REPO / "BenchMARL"))

from single_agentENV import SingleAgentAnchorEnv  # noqa: E402
from BenchMARL.environment import AnchorEnv  # noqa: E402
from utils.metrics import parse_coverage_basis  # noqa: E402


class AlwaysClass0(nn.Module):
    """Argmax is class 0 on every row, so y and f̂ disagree on class 1."""

    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        self.lin = nn.Linear(n_features, n_classes)
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.bias.zero_()
            self.lin.bias[0] = 2.0
            self.lin.bias[1] = -2.0

    def forward(self, x):
        return self.lin(x)


def _data(n_features=4, n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    c0 = 0.25 + 0.03 * rng.standard_normal((n_per, n_features))
    c1 = 0.75 + 0.03 * rng.standard_normal((n_per, n_features))
    X = np.clip(np.vstack([c0, c1]), 0.0, 1.0).astype(np.float32)
    y = np.array([0] * n_per + [1] * n_per, dtype=int)
    return X, y


def _sa_env(X, y, basis: str, target_class: int = 0):
    n = X.shape[1]
    cfg = dict(yaml.safe_load(open(REPO / "single_agent" / "conf" / "anchor_single.yaml"))["env_config"])
    cfg.update({
        "X_min": np.zeros(n, dtype=np.float32),
        "X_range": np.ones(n, dtype=np.float32),
        "max_cycles": 8,
        "coverage_basis": basis,
        "training_instance_ratio": 0.0,
        "mode": "training",
    })
    return SingleAgentAnchorEnv(
        X_unit=X, X_std=X, y=y,
        feature_names=[f"f{i}" for i in range(n)],
        classifier=AlwaysClass0(n),
        target_class=target_class,
        env_config=cfg,
    )


def _ma_env(X, y, basis: str):
    n = X.shape[1]
    cfg = dict(yaml.safe_load(open(REPO / "BenchMARL" / "conf" / "anchor.yaml"))["env_config"])
    cfg.update({
        "X_min": np.zeros(n, dtype=np.float32),
        "X_range": np.ones(n, dtype=np.float32),
        "agents_per_class": 1,
        "max_cycles": 8,
        "coverage_basis": basis,
        "training_instance_ratio": 0.0,
        "enable_stability_termination": False,
        "mode": "training",
    })
    return AnchorEnv(
        X_unit=X, X_std=X, y=y,
        feature_names=[f"f{i}" for i in range(n)],
        classifier=AlwaysClass0(n),
        env_config=cfg,
    )


def test_parse_coverage_basis_aliases_and_rejects_junk():
    assert parse_coverage_basis("predicted") == "predicted"
    assert parse_coverage_basis("fhat") == "predicted"
    assert parse_coverage_basis("true_label") == "true_label"
    assert parse_coverage_basis("y") == "true_label"
    assert parse_coverage_basis(None) == "predicted"
    with pytest.raises(ValueError, match="coverage_basis"):
        parse_coverage_basis("marginal")


def test_rlda_predicted_coverage_uses_fhat_not_y():
    X, y = _data()
    env_y = _sa_env(X, y, "true_label", target_class=0)
    env_f = _sa_env(X, y, "predicted", target_class=0)
    env_y.reset(seed=0)
    env_f.reset(seed=0)

    def clip(env, a, b):
        env.a, env.b = a, b
        env._sync_unit_bounds_from_quantiles()

    a = np.zeros(env_y.n_features, dtype=np.float64)
    b = np.ones(env_y.n_features, dtype=np.float64)
    a[0], b[0] = 0.0, 0.5
    clip(env_y, a.copy(), b.copy())
    clip(env_f, a.copy(), b.copy())

    _, c_y, det_y = env_y._current_metrics()
    _, c_f, det_f = env_f._current_metrics()
    mask = ((X >= env_y.lower) & (X <= env_y.upper)).all(axis=1)
    preds = np.zeros(len(y), dtype=int)  # AlwaysClass0
    expected_y = float((mask & (y == 0)).sum() / (y == 0).sum())
    expected_f = float((mask & (preds == 0)).sum() / (preds == 0).sum())
    assert c_y == pytest.approx(expected_y)
    assert c_f == pytest.approx(expected_f)
    assert expected_y != pytest.approx(expected_f)
    assert det_y["n_class_samples"] == int((y == 0).sum())
    assert det_f["n_class_samples"] == int((preds == 0).sum())
    # Fid is still on f̂ in both cases.
    assert det_y["hard_precision"] == pytest.approx(det_f["hard_precision"])


def test_mada_predicted_coverage_uses_fhat_not_y():
    X, y = _data()
    env_y = _ma_env(X, y, "true_label")
    env_f = _ma_env(X, y, "predicted")
    env_y.reset(seed=0)
    env_f.reset(seed=0)
    agent = "agent_0"

    def clip(env, a, b):
        env.a[agent], env.b[agent] = a, b
        env._sync_unit_bounds_from_quantiles(agent)

    a = np.zeros(env_y.n_features, dtype=np.float64)
    b = np.ones(env_y.n_features, dtype=np.float64)
    a[0], b[0] = 0.0, 0.5
    clip(env_y, a.copy(), b.copy())
    clip(env_f, a.copy(), b.copy())

    _, c_y, det_y = env_y._current_metrics(agent)
    _, c_f, det_f = env_f._current_metrics(agent)
    mask = ((X >= env_y.lower[agent]) & (X <= env_y.upper[agent])).all(axis=1)
    preds = np.zeros(len(y), dtype=int)
    expected_y = float((mask & (y == 0)).sum() / (y == 0).sum())
    expected_f = float((mask & (preds == 0)).sum() / (preds == 0).sum())
    assert c_y == pytest.approx(expected_y)
    assert c_f == pytest.approx(expected_f)
    assert expected_y != pytest.approx(expected_f)

    union_y = env_y._compute_class_union_metrics()[0]["union_coverage"]
    union_f = env_f._compute_class_union_metrics()[0]["union_coverage"]
    assert union_y == pytest.approx(expected_y)
    assert union_f == pytest.approx(expected_f)


def test_mada_unknown_coverage_basis_is_rejected():
    X, y = _data()
    with pytest.raises(ValueError, match="coverage_basis"):
        _ma_env(X, y, "marginal")


def test_rlda_potential_uses_the_same_c_as_metrics():
    """Φ reads the coverage returned by _current_metrics; no second y-based path."""
    X, y = _data()
    env = _sa_env(X, y, "predicted", target_class=0)
    env.reset(seed=0)
    p, c, _ = env._current_metrics()
    phi = env._potential(p, c)
    # Empty rule: C=1 on f̂ (every row is class 0), gate is off until P clears.
    assert c == pytest.approx(1.0)
    assert np.isfinite(phi)
