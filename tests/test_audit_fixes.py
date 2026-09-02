"""Regression guards for the 2026-09-01 audit fixes (P-## and G-##).

Each test pins a specific defect from docs/BUGS_pipeline_review.md and
docs/BUGS_external_review_verdicts.md. They are cheap and hermetic: no training,
no checkpoints, no network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "BenchMARL"))
sys.path.insert(0, str(REPO / "single_agent"))

from BenchMARL.environment import AnchorEnv  # noqa: E402
from utils.metrics import collect_success_rate, ranking_score  # noqa: E402
from utils.inference_extract import (  # noqa: E402
    generation_split_arrays, persist_box_from_episode,
)


class _Clf(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.lin = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.lin(x)


def _env(agents_per_class: int = 2, n_classes: int = 2, n_features: int = 4, **extra):
    rng = np.random.default_rng(0)
    X = np.clip(np.vstack([
        (0.25 + 0.4 * c) + 0.05 * rng.standard_normal((40, n_features))
        for c in range(n_classes)
    ]), 0.0, 1.0).astype(np.float32)
    y = np.repeat(np.arange(n_classes), 40).astype(int)
    cfg = dict(yaml.safe_load(
        open(REPO / "BenchMARL" / "conf" / "anchor.yaml"))["env_config"])
    cfg.update({
        "X_min": np.zeros(n_features, dtype=np.float32),
        "X_range": np.ones(n_features, dtype=np.float32),
        "agents_per_class": agents_per_class,
        "max_cycles": 8,
    })
    cfg.update(extra)
    return AnchorEnv(
        X_unit=X, X_std=X, y=y,
        feature_names=[f"f{i}" for i in range(n_features)],
        classifier=_Clf(n_features, n_classes), env_config=cfg,
    )


# --------------------------------------------------------------- P-01

def test_class_union_gate_matches_the_local_gate():
    """P-01: both potentials must price coverage identically.

    The union kept the retired min(1, P / (0.8*target)) form, so at tau_P=0.9 it
    paid FULL coverage credit from union fidelity 0.72 while the local term paid
    nothing below 0.80 -- and the union is the cooperative stream.
    """
    env = _env()
    target = 0.9
    for p in (0.0, 0.5, 0.72, 0.79, 0.85, 0.9, 1.0):
        gate = env._coverage_gate(p, target)
        expected = min(1.0, max(0.0, (p - (target - env.gate_margin)) / env.gate_margin))
        assert gate == pytest.approx(expected)
    # the retired form is strictly more lenient exactly where it mattered
    assert env._coverage_gate(0.72, target) == pytest.approx(0.0)
    assert min(1.0, 0.72 / (target * 0.8)) == pytest.approx(1.0)


def test_class_union_potential_uses_the_shared_gate():
    env = _env()
    cls = 0
    lenient = {"union_precision": 0.72, "union_coverage": 1.0}
    # gate is 0 at fidelity 0.72, so the coverage half contributes nothing
    phi = env._class_union_potential(cls, lenient)
    assert phi == pytest.approx(env.alpha * min(0.72, env._get_effective_precision_target(cls)))


# --------------------------------------------------------------- G-12

def test_quantile_observation_carries_the_episode_clock():
    """G-12: xi_t is part of the state under truncation + per-step costs."""
    env = _env(init_mode="full_space")
    obs, _ = env.reset(seed=0)
    a0 = sorted(obs)[0]
    assert obs[a0].shape == (3 * env.n_features + 4,)
    assert float(obs[a0][-1]) == pytest.approx(0.0)   # t = 0 at reset

    acts = {a: np.zeros(env.action_space(a).shape[0], dtype=np.float32) for a in env.agents}
    obs2, _, _, _, _ = env.step(acts)
    assert float(obs2[a0][-1]) == pytest.approx(1.0 / env.max_cycles)


def test_single_agent_observation_matches_mada_layout():
    """The arms must share the representation, not just the reward."""
    ma = yaml.safe_load(open(REPO / "BenchMARL" / "conf" / "anchor.yaml"))["env_config"]
    sa = yaml.safe_load(open(REPO / "single_agent" / "conf" / "anchor_single.yaml"))["env_config"]
    assert ma["init_mode"] == sa["init_mode"] == "full_space"
    src = (REPO / "single_agent" / "single_agentENV.py").read_text()
    assert "3 * self.n_features + 4" in src, "SA obs must carry the clock too"


# --------------------------------------------------------------- G-02

def test_env_exposes_a_global_state_covering_every_agent():
    """G-02: the cross-class term needs the other classes inside the critic."""
    env = _env(agents_per_class=2, n_classes=2)
    env.reset(seed=0)
    st = env.state()
    assert st.shape == (env.state_size,)
    assert st.shape[0] == 2 * env.n_features * len(env.possible_agents) + 1
    assert np.all(np.isfinite(st))


def test_task_declares_the_global_state():
    from BenchMARL import benchmarl_wrappers as bw
    import inspect
    assert bw.AnchorTaskClass.has_state(None) is True
    src = inspect.getsource(bw.AnchorTaskClass.get_env_fun)
    assert "return_state=True" in src


# --------------------------------------------------------------- G-09

def test_incomplete_action_dict_is_rejected():
    """G-09: silently skipping left the agent out of every returned dict."""
    env = _env(agents_per_class=2, n_classes=2)
    env.reset(seed=0)
    acts = {a: np.zeros(env.action_space(a).shape[0], dtype=np.float32) for a in env.agents}
    acts.pop(sorted(acts)[0])
    with pytest.raises(KeyError, match="no action for live agent"):
        env.step(acts)


def test_complete_action_dict_returns_every_agent():
    env = _env(agents_per_class=2, n_classes=2)
    env.reset(seed=0)
    acts = {a: np.zeros(env.action_space(a).shape[0], dtype=np.float32) for a in env.agents}
    obs, rew, term, trunc, _ = env.step(acts)
    for d in (obs, rew, term, trunc):
        assert set(d) == set(acts)


# --------------------------------------------------------------- G-07

def test_class_start_is_a_real_row_and_differs_per_agent():
    """G-07/B-05: anchor on a real x*, and diversify across same-class agents."""
    env = _env(agents_per_class=3, n_classes=2, init_mode="full_space")
    env.reset(seed=0)
    cls_rows = env.X_unit[env.y == 0]
    starts = []
    for a in [x for x in env.possible_agents if env.agent_to_class[x] == 0]:
        env.x_star_unit.pop(a, None)
        env._class_centroid_quantiles(a)
        pt = env._class_centroid_unit[0]
        assert np.min(np.linalg.norm(cls_rows - pt[None, :], axis=1)) == pytest.approx(0.0, abs=1e-5)
        starts.append(tuple(np.round(pt, 6)))
    assert len(set(starts)) > 1, "same-class agents must not all start at one point"


# --------------------------------------------------------------- G-03

def test_success_rate_denominator_counts_attempted_episodes():
    """G-03: the rate must not be conditioned on 'produced a rule at all'."""
    rules = {
        "metadata": {"max_cycles": 50},
        "per_class_results": {
            "class_0": {
                "class": 0,
                "n_episodes_attempted": 20,
                "anchors": [
                    {"precision": 0.95, "coverage": 0.5, "n_steps": 3},
                    {"precision": 0.95, "coverage": 0.5, "n_steps": 3},
                ],
            }
        },
    }
    out = collect_success_rate(rules, tau_p=0.9, tau_c=0.1)
    assert out["n_episodes"] == 20
    assert out["n_success"] == 2
    assert out["success_rate"] == pytest.approx(2 / 20)
    assert out["per_class"]["class_0"]["n_episodes_no_box"] == 18


def test_success_rate_without_attempt_count_is_unchanged():
    """Older artifacts have no attempt count; behaviour must not change."""
    rules = {
        "metadata": {"max_cycles": 50},
        "per_class_results": {
            "class_0": {"class": 0, "anchors": [{"precision": 0.95, "coverage": 0.5}]},
        },
    }
    out = collect_success_rate(rules, tau_p=0.9, tau_c=0.1)
    assert out["n_episodes"] == 1


# --------------------------------------------------------------- G-08

def test_generation_on_the_test_split_is_refused():
    env_data = {
        "X_unit": np.zeros((2, 3)), "X_std": np.zeros((2, 3)), "y": np.zeros(2, dtype=int),
        "X_test_unit": np.ones((2, 3)), "X_test_std": np.ones((2, 3)), "y_test": np.ones(2, dtype=int),
    }
    with pytest.raises(ValueError, match="not allowed"):
        generation_split_arrays(env_data, {"generation_split": "test"})
    X, _, _, name = generation_split_arrays(env_data, {"generation_split": "train"})
    assert name == "train"


# --------------------------------------------------------------- G-11

def test_quantile_observation_is_never_parsed_as_bounds():
    """G-11: obs[:n] is `a`, a quantile -- not a lower bound."""
    n = 4
    quantile_obs = np.linspace(0, 1, 3 * n + 4).tolist()
    out = persist_box_from_episode(
        {"final_observation": quantile_obs}, {"X_min": np.zeros(n), "X_range": np.ones(n)}, n,
    )
    assert out is None, "a quantile observation must not yield a box"


def test_hull_observation_fallback_still_works():
    n = 4
    hull_obs = ([0.1] * n) + ([0.9] * n) + [0.5, 0.5, 0.0]     # 2n+3
    out = persist_box_from_episode(
        {"final_observation": hull_obs}, {"X_min": np.zeros(n), "X_range": np.ones(n)}, n,
    )
    assert out is not None


# --------------------------------------------------------------- P-02 / P-06

def test_ranking_score_is_support_aware_when_given_support():
    """P-02: without n_covered, lcb_coverage silently becomes the point estimate."""
    hi_fid_no_support = ranking_score(1.0, 0.5, "lcb_coverage", n_covered=1)
    ok_fid_supported = ranking_score(0.92, 0.5, "lcb_coverage", n_covered=25)
    assert ok_fid_supported > hi_fid_no_support, (
        "the Wilson LCB exists to stop fid=1.0-on-n=1 winning"
    )
    # the degraded call is what MADA used to make
    degraded = ranking_score(1.0, 0.5, "lcb_coverage")
    assert degraded == pytest.approx(1.0 * 1.5)
    assert degraded > ok_fid_supported


def test_mada_best_model_scoring_passes_support():
    src = (REPO / "BenchMARL" / "benchmarl_wrappers.py").read_text()
    block = src[src.index("def _extract_eval_score"):src.index("def _save_best_model_if_improved")]
    assert block.count("n_covered=n_cov") >= 2, "both scoring branches must pass support"
    assert "evaluation/box_support_mean" in src


def test_rlda_best_model_scoring_reads_the_configured_formula():
    src = (REPO / "single_agent" / "anchor_trainer_sb3.py").read_text()
    assert "self.ranking_score_formula" in src
    assert "ranking_score_formula=self._get_default_env_config().get(" in src


# --------------------------------------------------------------- G-05

def test_pipelines_never_promote_final_weights_to_best():
    for name in ("run_rlda_pipeline.py", "run_paper_seed_sac.py"):
        src = (REPO / "revision" / name).read_text()
        block = src[src.index("def ensure_best_models"):]
        block = block[:block.index("\ndef ")]
        assert "shutil.copy2" not in block, f"{name} still promotes final -> best"
        assert "SystemExit" in block, f"{name} must fail loudly instead"


# --------------------------------------------------------------- G-10

def test_missing_policy_metadata_fails_with_a_useful_error():
    src = (REPO / "BenchMARL" / "inference.py").read_text()
    i = src.index("metadata_path = metadata_files.get(group)")
    block = src[i:i + 1200]
    assert "FileNotFoundError" in block
    assert 'metadata_path or ""' not in block, "empty path would reach open()"


# --------------------------------------------------------------- G-13

def test_compactness_uses_the_artifact_threshold():
    import inspect
    from utils.eval_harness import per_class_block
    assert "sparsity_width_ratio" in inspect.signature(per_class_block).parameters
    src = (REPO / "revision" / "evaluate.py").read_text()
    assert "sparsity_width_ratio=sparsity" in src


# --------------------------------------------------------------- G-04

def test_break_even_charges_training_and_frees_serving():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mkfig", REPO / "paper" / "make_figures.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    row = {
        "method": "mada",
        "queries": {
            "n_blackbox_queries": 1_000,
            "n_training_queries": 5_000_000,
            "n_reporting_queries": 400,
        },
        "classifier_accuracy": {"n": 100},
    }
    fixed, marginal = m._fixed_and_marginal(row)
    assert fixed == pytest.approx(5_001_000), "training must be in the fixed cost"
    assert marginal == pytest.approx(0.0), "a fixed box costs no queries to apply"


def test_break_even_leaves_per_instance_baselines_alone():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mkfig2", REPO / "paper" / "make_figures.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    method = sorted(m.PER_INSTANCE)[0]
    row = {
        "method": method,
        "queries": {"n_blackbox_queries": 2000, "n_reporting_queries": 400},
        "classifier_accuracy": {"n": 100},
    }
    fixed, marginal = m._fixed_and_marginal(row)
    assert fixed == pytest.approx(0.0)
    assert marginal == pytest.approx(20.0)


# --------------------------------------------------------------- P-05

def _ranked(rule_id, mask, sel_cov, score, y, y_hat, cls):
    """A RankedRule with REAL reporting metrics and a stated selection coverage.

    Nothing is faked: metrics come from evaluate_mask on the given mask, so the
    union >= best invariant holds naturally. Only `selection_metrics` (what the
    validation split saw) is supplied, which is exactly what P-05 is about.
    """
    from utils.metrics import RankedRule, evaluate_mask
    m = evaluate_mask(y=y, y_hat=y_hat, mask=mask, target_class=cls,
                      class_conditional=True, min_support=1)
    return RankedRule(
        rule_id=rule_id, lower=np.zeros(2, dtype=np.float32),
        upper=np.ones(2, dtype=np.float32), mask=mask, metrics=m, score=score,
        extra={"selection_metrics": {"coverage": sel_cov}},
    )


def test_tiebreak_uses_selection_not_reporting_coverage():
    """P-05: `score` is the validation score but `metrics` is the TEST split.

    Breaking ties on r.metrics.coverage let test data pick which rule is called
    `best` -- inside a function whose docstring promises the opposite. Rule A
    was better on validation (0.9 vs 0.1) and is worse on test (0.3 vs 0.6);
    with the scores tied, A must still win.
    """
    from utils.metrics import select_topk_union
    n = 20
    y = np.array([0] * 10 + [1] * 10)
    y_hat = y.copy()
    mask_a = np.zeros(n, dtype=bool); mask_a[:3] = True    # test coverage 0.3
    mask_b = np.zeros(n, dtype=bool); mask_b[:6] = True    # test coverage 0.6
    a = _ranked("A", mask_a, sel_cov=0.9, score=0.5, y=y, y_hat=y_hat, cls=0)
    b = _ranked("B", mask_b, sel_cov=0.1, score=0.5, y=y, y_hat=y_hat, cls=0)
    assert a.metrics.coverage < b.metrics.coverage, "fixture must invert on test"

    for order in ([a, b], [b, a]):
        out = select_topk_union(order, y, y_hat, target_class=0, k=2,
                                class_conditional=True, min_support=1,
                                enforce_min_support=False)
        assert out.best.rule_id == "A", (
            "best must be the validation winner, not the test winner"
        )


def test_tiebreak_is_deterministic_when_everything_ties():
    """Fully tied rules must not depend on input order."""
    from utils.metrics import select_topk_union
    n = 20
    y = np.array([0] * 10 + [1] * 10)
    y_hat = y.copy()
    mask = np.zeros(n, dtype=bool); mask[:6] = True
    a = _ranked("zzz", mask.copy(), 0.5, 0.5, y, y_hat, 0)
    b = _ranked("aaa", mask.copy(), 0.5, 0.5, y, y_hat, 0)
    ids = set()
    for order in ([a, b], [b, a]):
        out = select_topk_union(order, y, y_hat, target_class=0, k=2,
                                class_conditional=True, min_support=1,
                                enforce_min_support=False)
        ids.add(out.best.rule_id)
    assert len(ids) == 1, f"best changed with input order: {ids}"


# --------------------------------------------------------------- G-06

def test_rlda_quality_filter_uses_class_conditional_coverage():
    """G-06: the filter threshold is calibrated for tau_C (class-conditional).

    Source-level: the filter is inline in a 600-line function with no seam to
    call. Asserted on the exact comparison, and on the absence of the marginal
    key from it, since that is the whole defect.
    """
    src = (REPO / "single_agent" / "single_agent_inference.py").read_text()
    i = src.index("if filter_low_quality_rollouts:")
    block = src[i:src.index("if len(kept_rollouts) == 0:", i)]
    assert 'rollout_data["coverage_class_conditional_recomputed"]' in block
    assert 'anchor_coverage = rollout_data["coverage_recomputed"]' not in block, (
        "the marginal value must not be compared against a class-conditional threshold"
    )


def test_recompute_keeps_both_coverages_distinct():
    """The two quantities must stay separately available and NOT be conflated."""
    src = (REPO / "single_agent" / "single_agent_inference.py").read_text()
    assert 'rollout_data["coverage_marginal"] = float(cov_full)' in src
    assert 'rollout_data["coverage"] = float(cov_class_conditional_full)' in src


# --------------------------------------------------------------- G-12 / G-07 (RLDA side, behavioural)

def _sa_env(**extra):
    sys.path.insert(0, str(REPO / "single_agent"))
    from single_agentENV import SingleAgentAnchorEnv  # noqa
    rng = np.random.default_rng(0)
    n = 4
    X = np.clip(np.vstack([
        (0.25 + 0.4 * c) + 0.05 * rng.standard_normal((40, n)) for c in range(2)
    ]), 0.0, 1.0).astype(np.float32)
    y = np.repeat(np.arange(2), 40).astype(int)
    cfg = dict(yaml.safe_load(
        open(REPO / "single_agent" / "conf" / "anchor_single.yaml"))["env_config"])
    cfg.update({
        "X_min": np.zeros(n, dtype=np.float32),
        "X_range": np.ones(n, dtype=np.float32),
        "max_cycles": 8,
    })
    cfg.update(extra)
    return SingleAgentAnchorEnv(
        X_unit=X, X_std=X, y=y, feature_names=[f"f{i}" for i in range(n)],
        classifier=_Clf(n, 2), target_class=0, env_config=cfg,
    )


def test_rlda_observation_carries_the_clock_and_advances():
    """G-12 on the RLDA side, behaviourally -- not just a source string."""
    env = _sa_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3 * env.n_features + 4,)
    assert float(obs[-1]) == pytest.approx(0.0)
    obs2, _, _, _, _ = env.step(np.zeros(env.action_space.shape[0], dtype=np.float32))
    assert float(obs2[-1]) == pytest.approx(1.0 / env.max_cycles)


def test_rlda_class_start_is_a_real_row():
    """G-07 on the RLDA side: anchor on a real x*, not the per-feature median."""
    env = _sa_env()
    env.x_star_unit = None
    env._class_centroid_quantiles()
    pt = env._class_centroid_unit
    rows = env.X_unit[env.y == env.target_class]
    assert np.min(np.linalg.norm(rows - pt[None, :], axis=1)) == pytest.approx(0.0, abs=1e-5)
