"""Train–eval contract: extraction fail-hard, train slots, stability, coverage-gain, success dedupe."""
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
sys.path.insert(0, str(REPO / "BenchMARL"))

from single_agentENV import SingleAgentAnchorEnv  # noqa: E402
from environment import AnchorEnv  # noqa: E402
from utils.eval_harness import apply_train_val_slots, resolve_extracted_models_dir  # noqa: E402
from utils.metrics import collect_success_rate  # noqa: E402


class ConstClf(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        self.lin = nn.Linear(n_features, n_classes)
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.bias.zero_()
            self.lin.bias[0] = 1.0

    def forward(self, x):
        return self.lin(x)


def _toy_xy(n_features=4, n_per_class=30, seed=0):
    rng = np.random.default_rng(seed)
    c0 = 0.25 + 0.02 * rng.standard_normal((n_per_class, n_features))
    c1 = 0.75 + 0.02 * rng.standard_normal((n_per_class, n_features))
    X = np.clip(np.vstack([c0, c1]), 0.0, 1.0).astype(np.float32)
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=int)
    return X, y


def _sa_cfg(n_features, **extra):
    cfg = {
        "max_cycles": 40,
        "min_width": 0.05,
        "initial_window": 0.1,
        "precision_target": 0.9,
        "coverage_target": 0.2,
        "require_coverage_gain_to_terminate": True,
        "init_mode": "neighbor_hull",
        "precision_estimator": "empirical",
        "use_perturbation": False,
        "mode": "training",
        "use_class_centroids": True,
        "training_instance_ratio": 1.0,
        "X_min": np.zeros(n_features, dtype=np.float32),
        "X_range": np.ones(n_features, dtype=np.float32),
        "enable_stability_termination": True,
        "stability_window": 20,
        "stability_precision_tol": 1.0,
        "stability_coverage_tol": 1.0,
        "stability_drift_tol": 1.0,
    }
    cfg.update(extra)
    return cfg


def test_resolve_extracted_models_dir_best_missing_raises(tmp_path):
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "individual_models").mkdir()
    with pytest.raises(FileNotFoundError, match="individual_models_best"):
        resolve_extracted_models_dir(str(exp), prefer_model="best")


def test_resolve_extracted_models_dir_best_present(tmp_path):
    exp = tmp_path / "exp"
    best = exp / "individual_models_best"
    best.mkdir(parents=True)
    assert resolve_extracted_models_dir(str(exp), prefer_model="best") == str(best)


def test_apply_train_val_slots_keeps_train_arrays():
    n_train, n_val, n_feat = 80, 20, 3
    env_data = {
        "X_unit": np.zeros((n_train, n_feat), dtype=np.float32),
        "X_std": np.zeros((n_train, n_feat), dtype=np.float32),
        "y": np.zeros(n_train, dtype=int),
        "X_val_unit": np.ones((n_val, n_feat), dtype=np.float32),
        "X_val_std": np.ones((n_val, n_feat), dtype=np.float32),
        "y_val": np.ones(n_val, dtype=int),
    }
    cfg = apply_train_val_slots(env_data, {}, metric_split="val")
    assert env_data["X_unit"].shape[0] == n_train
    assert cfg["eval_split"] == "val"
    assert cfg["X_val_unit"].shape[0] == n_val
    assert "min_coverage_floor" not in cfg
    assert "X_unit" not in cfg


def test_collect_success_rate_does_not_double_count_nested_all_anchors():
    ep = {
        "precision_rollout_estimated": 0.99,
        "coverage_class_conditional_rollout_estimated": 0.4,
    }
    rules = {
        "metadata": {"steps_per_episode": 200},
        "per_class_results": {
            "class_0": {
                "all_anchors": [ep, ep, ep],
                "per_agent_results": {
                    "agent_0_0": {"anchors": [ep, ep, ep]},
                },
            }
        },
    }
    out = collect_success_rate(rules, 0.9, 0.2)
    assert out["n_episodes"] == 3
    assert out["n_success"] == 3


def test_mask_length_condition_off_test_does_not_treat_len_y_as_bool():
    y = np.zeros(10)
    y_test = np.zeros(3)
    mask_ok = np.ones(10, dtype=bool)
    mask_wrong = np.ones(3, dtype=bool)
    eval_on_test_data = False
    buggy = (len(mask_wrong) == len(y_test) if eval_on_test_data else len(y))
    assert buggy == 10
    expected = len(y_test) if eval_on_test_data else len(y)
    assert (len(mask_ok) == expected) is True
    assert (len(mask_wrong) == expected) is False


def test_coverage_improved_true_when_already_at_tau_c():
    X, y = _toy_xy()
    env = SingleAgentAnchorEnv(
        X_unit=X,
        X_std=X,
        y=y,
        feature_names=[f"f{i}" for i in range(X.shape[1])],
        classifier=ConstClf(X.shape[1]),
        target_class=0,
        env_config=_sa_cfg(X.shape[1], coverage_target=0.2, require_coverage_gain_to_terminate=True),
    )
    env._coverage_at_reset = 0.25
    env._coverage_gain_eps = 0.01
    assert env._coverage_improved(0.25) is True
    env._coverage_at_reset = 0.05
    assert env._coverage_improved(0.05) is False
    assert env._coverage_improved(0.07) is True


def test_ma_stability_counter_accumulates_across_steps():
    X, y = _toy_xy()
    n_features = X.shape[1]
    env = AnchorEnv(
        X_unit=X,
        X_std=X,
        y=y,
        feature_names=[f"f{i}" for i in range(n_features)],
        classifier=ConstClf(n_features),
        target_classes=[0],
        env_config=_sa_cfg(
            n_features,
            agents_per_class=1,
            enable_stability_termination=True,
            stability_window=20,
            # Wide tols so a zero action counts as stable.
            stability_precision_tol=1.0,
            stability_coverage_tol=1.0,
            stability_drift_tol=1.0,
            min_steps_before_termination=100,
        ),
    )
    env.reset()
    agent = env.agents[0]
    n_act = int(env.action_space(agent).shape[0])
    zero = np.zeros(n_act, dtype=np.float32)
    for _ in range(25):
        env.step({agent: zero})
    assert env._stable_counts[agent] >= env.stability_window
    assert env._stable_counts[agent] != 0
    assert env._stable_counts[agent] != 1


def test_union_metrics_ignore_idle_agent_boxes():
    X, y = _toy_xy()
    n_features = X.shape[1]
    env = AnchorEnv(
        X_unit=X,
        X_std=X,
        y=y,
        feature_names=[f"f{i}" for i in range(n_features)],
        classifier=ConstClf(n_features),
        target_classes=[0],
        env_config=_sa_cfg(n_features, agents_per_class=2),
    )
    env.reset()
    acting = env.agents[0]
    idle = env.agents[1]
    env.lower[acting] = np.full(n_features, 0.24, dtype=np.float32)
    env.upper[acting] = np.full(n_features, 0.26, dtype=np.float32)
    env.lower[idle] = np.zeros(n_features, dtype=np.float32)
    env.upper[idle] = np.ones(n_features, dtype=np.float32)
    env.agents = [acting, idle]
    cov_with_idle = env._compute_class_union_metrics()[0]["union_coverage"]
    env.agents = [acting]
    cov_acting_only = env._compute_class_union_metrics()[0]["union_coverage"]
    assert cov_with_idle > cov_acting_only + 0.1


def test_mada_overlap_skips_unconstrained_dims():
    lo_a = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    hi_a = np.array([0.4, 1.0, 1.0], dtype=np.float32)
    lo_b = np.array([0.0, 0.2, 0.0], dtype=np.float32)
    hi_b = np.array([1.0, 0.4, 1.0], dtype=np.float32)
    active_a = AnchorEnv._active_dims(lo_a, hi_a, 0.95)
    active_b = AnchorEnv._active_dims(lo_b, hi_b, 0.95)
    assert active_a.tolist() == [True, False, False]
    assert active_b.tolist() == [False, True, False]
    union = active_a | active_b
    pred_jacc = float((active_a & active_b).sum() / max(1, int(union.sum())))
    assert pred_jacc == 0.0
    active_vol_a = float(np.prod(np.maximum((hi_a - lo_a)[active_a], 1e-9)))
    assert active_vol_a == pytest.approx(0.2)


def test_mada_freeze_skips_unconstrained_categorical():
    X, y = _toy_xy()
    n_features = X.shape[1]
    env = AnchorEnv(
        X_unit=X, X_std=X, y=y,
        feature_names=[f"f{i}" for i in range(n_features)],
        classifier=ConstClf(n_features),
        env_config={
            **_sa_cfg(n_features),
            "categorical_indices": [0],
            "categorical_freeze": "instance",
            "agents_per_class": 1,
        },
    )
    env.reset()
    agent = env.agents[0]
    env.lower[agent] = np.zeros(n_features, dtype=np.float32)
    env.upper[agent] = np.ones(n_features, dtype=np.float32)
    env._freeze_categorical_bounds(agent)
    assert float(env.upper[agent][0] - env.lower[agent][0]) > 0.5
    env.lower[agent][0] = 0.3
    env.upper[agent][0] = 0.31
    env._freeze_categorical_bounds(agent)
    assert float(env.upper[agent][0] - env.lower[agent][0]) <= 2.5e-4
