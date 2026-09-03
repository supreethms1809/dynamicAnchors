"""Track A empirical Fid is the train done-switch, not conditional CRN P."""
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

from single_agentENV import SingleAgentAnchorEnv  # noqa: E402


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
        "precision_estimator": "empirical",
        "use_perturbation": False,
        "n_perturb": 64,
        "n_perturb_train": 64,
        "leave_threshold": 0.5,
        "max_new_constraints_per_step": 1,
        "precision_target": 0.9,
        "coverage_target": 0.1,
        "min_support": 10,
        "require_min_support_to_terminate": True,
        "require_precision_gain_to_terminate": True,
        "strict_target_termination": True,
        "min_steps_before_termination": 2,
        "mode": "training",
        "training_instance_ratio": 0.0,
        "use_class_centroids": True,
        "X_min": np.zeros(n, dtype=np.float32),
        "X_range": np.ones(n, dtype=np.float32),
    }
    cfg.update(extra)
    return SingleAgentAnchorEnv(
        X_unit=X, X_std=X, y=y,
        feature_names=[f"f{i}" for i in range(n)],
        classifier=LinearSepClf(n),
        target_class=0,
        env_config=cfg,
    )


def test_shipped_yaml_uses_empirical_fid_and_min_support_gate():
    mae = yaml.safe_load(open(REPO / "BenchMARL" / "conf" / "anchor.yaml"))["env_config"]
    sae = yaml.safe_load(open(REPO / "single_agent" / "conf" / "anchor_single.yaml"))["env_config"]
    assert sae["precision_estimator"] == "empirical"
    assert mae["precision_estimator"] == "empirical"
    assert sae["use_perturbation"] is False
    assert mae["require_min_support_to_terminate"] is True
    assert sae["require_min_support_to_terminate"] is True
    assert sae["min_support"] == 10


def test_empirical_p_is_real_row_fid_not_crn_count():
    env = _env(*_data())
    env.reset(seed=0)
    p, c, det = env._current_metrics()
    assert det["sampler"].startswith("empirical_")
    assert int(det["n_points"]) == int(det["n_covered"])
    assert int(det["n_points"]) != env.n_perturb
    assert c == pytest.approx(1.0, abs=0.05)


def test_tiny_box_does_not_terminate_even_if_p_and_c_hit_tau():
    """iris-style: C≥0.1 can be 3 class rows; that is not Track A min_support."""
    n_per = 20
    X0 = np.vstack([
        np.full((3, 4), 0.10, dtype=np.float32),
        np.full((n_per - 3, 4), 0.30, dtype=np.float32),
    ])
    X1 = np.full((n_per, 4), 0.80, dtype=np.float32)
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per + [1] * n_per, dtype=int)
    env = _env(X, y, min_support=10, require_min_support_to_terminate=True)
    env.reset(seed=1)
    env._precision_at_reset = 0.0
    lo = np.full(env.n_features, 0.09, dtype=np.float32)
    hi = np.full(env.n_features, 0.11, dtype=np.float32)
    env.set_quantile_box(env._values_to_q(lo), env._values_to_q(hi))
    p, c, det = env._current_metrics()
    assert p >= 0.9
    assert c >= 0.1
    assert int(det["n_class_in_box"]) == 3
    env.step_count = 5
    env.timestep = 5
    noop = np.zeros(env.action_space.shape[0], dtype=np.float32)
    _, _, done, trunc, info = env.step(noop)
    assert not done
    assert info.get("termination_reason") != "both_targets_met"


def test_supported_box_can_terminate():
    n_per = 20
    # Spread class-0 so a mid quantile band covers ≥10 rows without the whole space.
    x0 = np.linspace(0.05, 0.35, n_per, dtype=np.float32)
    X0 = np.stack([x0] * 4, axis=1)
    X1 = np.full((n_per, 4), 0.80, dtype=np.float32)
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per + [1] * n_per, dtype=int)
    env = _env(X, y, min_support=10, require_min_support_to_terminate=True)
    env.reset(seed=2)
    env._precision_at_reset = 0.0
    env.set_quantile_box(np.zeros(env.n_features), np.full(env.n_features, 0.7))
    p, c, det = env._current_metrics()
    assert p >= 0.9
    assert c >= 0.1
    assert int(det["n_class_in_box"]) >= 10
    env.step_count = 5
    env.timestep = 5
    noop = np.zeros(env.action_space.shape[0], dtype=np.float32)
    _, _, done, trunc, info = env.step(noop)
    assert done
    assert info.get("termination_reason") == 1.0  # both_targets_met
