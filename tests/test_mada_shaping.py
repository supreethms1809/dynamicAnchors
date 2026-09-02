"""MADA (BenchMARL AnchorEnv) shaping + quantile-MDP port guards.

Paper YAML (`BenchMARL/conf/anchor.yaml`) is `init_mode: full_space`, matching
RLDA. The constructor default stays `neighbor_hull` so CDEA/WyoDOT checkpoints
keep the 2d+3 obs layout — that is plumbing, not the paper method. Shared YAML
knobs are locked by `test_shipped_configs_agree_between_rlda_and_mada`.
"""
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
sys.path.insert(0, str(REPO / "BenchMARL"))

from BenchMARL.environment import AnchorEnv  # noqa: E402
from utils.metrics import ranking_score  # noqa: E402


class LinearSepClf(nn.Module):
    def __init__(self, n_features: int = 4):
        super().__init__()
        self.lin = nn.Linear(n_features, 2)
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


def _env(**extra):
    X, y = _data()
    n = X.shape[1]
    cfg = yaml.safe_load(open(REPO / "BenchMARL" / "conf" / "anchor.yaml"))["env_config"]
    cfg = dict(cfg)
    cfg.update({
        "X_min": np.zeros(n, dtype=np.float32),
        "X_range": np.ones(n, dtype=np.float32),
        "agents_per_class": 1,
        "max_cycles": 8,
        "enable_stability_termination": False,
    })
    cfg.update(extra)
    return AnchorEnv(
        X_unit=X, X_std=X, y=y,
        feature_names=[f"f{i}" for i in range(n)],
        classifier=LinearSepClf(n),
        env_config=cfg,
    )


def _noop(env, agents):
    return {a: np.zeros(env.action_space(a).shape[0], dtype=np.float32) for a in agents}


def test_discount_is_not_the_overlap_gamma():
    env = _env()
    assert env.gamma == pytest.approx(0.1)          # narrow-width penalty weight
    assert env.discount == pytest.approx(0.95)      # Ng shaping discount (horizon 20)
    assert _env(discount=0.5).discount == pytest.approx(0.5)


def test_shaping_is_discounted_on_a_nonterminal_step():
    """F = gamma*Phi(s') - Phi(s). On a no-op step Phi is unchanged, so the
    shaping contribution must be exactly (gamma - 1) * Phi, not 0.

    Hull mode: a no-op there is a genuine non-terminal step with non-zero Phi."""
    env = _env(init_mode="neighbor_hull", precision_estimator="empirical")
    obs, _ = env.reset(seed=0)
    agents = list(obs.keys())
    a0 = agents[0]
    zero = _noop(env, agents)

    _, _, term, trunc, infos = env.step(zero)
    assert not term.get(a0) and not trunc.get(a0), "fixture must not end on step 1"
    info = infos[a0]
    phi = env._potential(info["anchor_precision"], info["anchor_coverage"],
                         env._get_class_for_agent(a0))
    assert abs(phi) > 1e-9
    assert env._rt_shaping[a0] == pytest.approx((env.discount - 1.0) * phi, abs=1e-6)
    assert "shaping_terminal_correction" not in info


def test_absorbing_terminal_correction_applied_on_termination():
    """On the terminating transition the total shaping must be -Phi(s):
    (gamma*Phi(s') - Phi(s)) + (-gamma*Phi(s')) = -Phi(s).

    Pinned to hull mode: there a no-op episode reaches target-based termination so
    the assertion actually runs. Under the quantile MDP a no-op sits at k=0 forever
    (cov_ok=0, no precision gain) and this test would silently skip.
    """
    env = _env(init_mode="neighbor_hull", precision_estimator="empirical")
    obs, _ = env.reset(seed=0)
    agents = list(obs.keys())
    a0 = agents[0]
    zero = _noop(env, agents)

    prev = 0.0
    for _ in range(env.max_cycles):
        _, _, term, trunc, infos = env.step(zero)
        delta = env._rt_shaping[a0] - prev
        prev = env._rt_shaping[a0]
        if term.get(a0):
            info = infos[a0]
            phi = env._potential(info["anchor_precision"], info["anchor_coverage"],
                                 env._get_class_for_agent(a0))
            assert "shaping_terminal_correction" in info
            assert info["shaping_terminal_correction"] == pytest.approx(
                -env.discount * phi, abs=1e-6)
            # (gamma-1)*Phi from the step, then -gamma*Phi from the correction.
            assert delta == pytest.approx(-phi, abs=1e-6)
            return
    pytest.skip("fixture never reached target-based termination")


def test_no_terminal_correction_on_truncation():
    """BenchMARL/TorchRL bootstraps a time-limit truncation, so s' is not
    absorbing there and the correction must not fire -- otherwise every truncated
    episode is penalised in proportion to how good its final box was."""
    # Unreachable coverage target => the episode can only ever truncate.
    env = _env(coverage_target=1.5, max_cycles=3)
    obs, _ = env.reset(seed=1)
    agents = list(obs.keys())
    a0 = agents[0]
    zero = _noop(env, agents)

    prev, info, term, trunc = 0.0, None, None, None
    for _ in range(3):
        prev = env._rt_shaping[a0]
        _, _, term, trunc, infos = env.step(zero)
        info = infos.get(a0, {})
    assert trunc.get(a0) and not term.get(a0), "fixture must end by truncation"
    assert "shaping_terminal_correction" not in info

    phi = env._potential(info["anchor_precision"], info["anchor_coverage"],
                         env._get_class_for_agent(a0))
    delta = env._rt_shaping[a0] - prev
    assert delta == pytest.approx((env.discount - 1.0) * phi, abs=1e-6), (
        "looks like the absorbing-terminal correction fired on truncation"
    )


def test_ranking_support_discriminates_thin_rules():
    """The MADA top-k call site now passes n_covered, so lcb_coverage actually
    applies the Wilson bound. Without it a box covering ONE row scores fid=1.0
    and outranks a well-supported rule."""
    thin = ranking_score(1.0, 0.5, "lcb_coverage", n_covered=1)
    thick = ranking_score(0.95, 0.5, "lcb_coverage", n_covered=50)
    assert thin < thick, "Wilson lower bound must penalise a single-row rule"
    # Absent support keeps the old point-estimate behaviour rather than -inf.
    assert ranking_score(1.0, 0.5, "lcb_coverage", n_covered=None) == pytest.approx(1.5)


def test_stability_termination_also_gets_the_correction():
    """The correction must run AFTER every termination path.

    Stability-based termination is assigned in a post-processing block that runs
    after the per-agent loop; an earlier placement of the correction would silently
    skip those agents and leave the +gamma*Phi(s') un-cancelled.
    """
    env = _env(
        coverage_target=1.5,            # target-based termination impossible
        max_cycles=40,
        enable_stability_termination=True,
        stability_min_steps=2,
        stability_window=2,
    )
    obs, _ = env.reset(seed=0)
    agents = list(obs.keys())
    a0 = agents[0]
    zero = _noop(env, agents)

    for _ in range(env.max_cycles):
        _, _, term, trunc, infos = env.step(zero)
        if term.get(a0):
            info = infos[a0]
            assert info.get("termination_reason_str") == "stabilized"
            phi = env._potential(info["anchor_precision"], info["anchor_coverage"],
                                 env._get_class_for_agent(a0))
            assert "shaping_terminal_correction" in info, (
                "stability termination missed the absorbing-terminal correction"
            )
            assert info["shaping_terminal_correction"] == pytest.approx(
                -env.discount * phi, abs=1e-6)
            return
        if trunc.get(a0):
            pytest.skip("fixture truncated before stabilizing")
    pytest.skip("fixture never stabilized")


# ---------------------------------------------------------------------------
# Quantile-position MDP port (opt-in via init_mode: full_space)
# ---------------------------------------------------------------------------

def test_hull_is_the_CODE_default_so_the_port_is_opt_in():
    """The quantile MDP must be opt-in at the code level.

    WyoDOT/CDEA share this env and hold 2n+3 checkpoints, so an unset init_mode
    has to mean "hull". The shipped anchor.yaml may legitimately opt in -- that is
    a config decision, checked separately below.
    """
    env = _env(init_mode=None) if False else None
    X, y = _data()
    n = X.shape[1]
    from BenchMARL.environment import AnchorEnv as _AE
    bare = _AE(
        X_unit=X, X_std=X, y=y,
        feature_names=[f"f{i}" for i in range(n)],
        classifier=LinearSepClf(n),
        env_config={
            "max_cycles": 8, "agents_per_class": 1,
            "X_min": np.zeros(n, dtype=np.float32),
            "X_range": np.ones(n, dtype=np.float32),
        },
    )
    assert bare.init_mode == "neighbor_hull"
    assert bare._uses_quantile_mdp() is False
    assert bare.observation_space(bare.possible_agents[0]).shape == (2 * n + 3,)


def test_hull_mode_still_works_when_selected():
    env = _env(init_mode="neighbor_hull", precision_estimator="empirical")
    assert env._uses_quantile_mdp() is False
    obs, _ = env.reset(seed=0)
    a0 = list(obs.keys())[0]
    assert obs[a0].shape == (2 * env.n_features + 3,)
    assert env.n_predicates(a0) == env.n_features, "hull init constrains every dim"


def test_shipped_configs_agree_between_rlda_and_mada():
    """Every shared env knob must match, or the RLDA-vs-MADA comparison is invalid.

    The knob list is DERIVED from the two sources (every `env_config.get("...")`
    they both read), not hand-written. An earlier hand-written version checked five
    knobs and passed while beta was 5.0 in MADA and 0.6 in RLDA -- an 8.3x
    difference in the coverage weight of Phi, i.e. the two arms were optimising
    different objectives. Anything genuinely per-method must be named in
    MULTI_AGENT_ONLY below, so adding a divergence is a deliberate act.
    """
    import re
    pat = re.compile(r'env_config\.get\(\s*["\']([A-Za-z0-9_]+)["\']')
    ma_keys = set(pat.findall((REPO / "BenchMARL" / "environment.py").read_text()))
    sa_keys = set(pat.findall((REPO / "single_agent" / "single_agentENV.py").read_text()))

    # Injected at runtime or intentionally per-pipeline.
    PLUMBING = {
        "X_min", "X_range", "scaler_mean", "scaler_scale", "X_val_unit", "X_val_std",
        "y_val", "X_test_unit", "X_test_std", "y_test", "rng", "mode",
        "categorical_indices", "categorical_value_names", "cluster_centroids_per_class",
        "fixed_instances_per_class", "eval_on_test_data", "generation_split",
        "n_perturb", "agents_per_class", "normalize_data", "eval_split", "max_cycles",
    }
    # The multi-agent method itself: no single-agent counterpart exists.
    MULTI_AGENT_ONLY = {
        "inter_class_overlap_weight", "shared_reward_weight", "shared_terminal_bonus",
        "same_class_diversity_weight",
        "class_union_cov_weight", "class_union_prec_weight", "global_coverage_weight",
        "global_coverage_threshold", "nashconv_threshold",
    }
    # Read but never used in either reward (coverage_bonus is hardcoded 0.0).
    DEAD = {
        "coverage_bonus_weight_met", "coverage_bonus_weight_high_prec",
        "coverage_bonus_weight_high_prec_progress",
        "coverage_bonus_weight_high_prec_distance",
        "coverage_bonus_weight_reasonable_prec",
        "coverage_bonus_weight_reasonable_prec_progress",
        "target_class_bonus_weight", "use_random_sampling", "sparsity_weight",
        "js_penalty_weight",
    }

    ma_cfg = yaml.safe_load(open(REPO / "BenchMARL" / "conf" / "anchor.yaml"))
    sa_cfg = yaml.safe_load(open(REPO / "single_agent" / "conf" / "anchor_single.yaml"))
    mae, sae = ma_cfg["env_config"], sa_cfg["env_config"]

    shared = (ma_keys & sa_keys) - PLUMBING - MULTI_AGENT_ONLY - DEAD
    mismatched = {
        k: (mae.get(k, "<absent>"), sae.get(k, "<absent>"))
        for k in sorted(shared)
        if mae.get(k, "<absent>") != sae.get(k, "<absent>")
    }
    assert not mismatched, (
        "shared env knobs differ between MADA and RLDA "
        f"(MADA, RLDA): {mismatched}"
    )

    # max_cycles is excluded above because anchor.yaml carries it twice; check both.
    assert mae.get("max_cycles") == sae.get("max_cycles")
    assert ma_cfg.get("max_cycles") == mae.get("max_cycles"), "anchor.yaml max_cycles out of sync"


def test_beta_is_not_the_hull_era_value():
    """beta=5.0 was justified by precision being pinned at 1.000 under the HULL
    representation. Under the quantile MDP precision is a live term, so that value
    would make a +0.6 coverage gain (+2.24) outweigh the entire precision term.
    0.25 (not 0.6) so shrinking for Fid is not cancelled by the √C term at C≈1.
    """
    mae = yaml.safe_load(open(REPO / "BenchMARL" / "conf" / "anchor.yaml"))["env_config"]
    sae = yaml.safe_load(open(REPO / "single_agent" / "conf" / "anchor_single.yaml"))["env_config"]
    assert mae["beta"] == pytest.approx(0.25)
    assert sae["beta"] == pytest.approx(mae["beta"])


def test_quantile_mode_starts_from_the_empty_rule():
    env = _env(init_mode="full_space", precision_estimator="conditional")
    obs, _ = env.reset(seed=0)
    agents = list(obs.keys())
    a0 = agents[0]
    assert obs[a0].shape == (3 * env.n_features + 4,), \
        "obs is [a, b, q*, P, C, mode, episode_phase] (G-12 added the clock)"
    assert env.action_space(a0).shape == (2 * env.n_features,), "action stays (2d,)"
    for a in agents:
        assert env.n_predicates(a) == 0
        assert np.allclose(env.lower[a], 0.0) and np.allclose(env.upper[a], 1.0)


def test_quantile_mode_adds_predicates_and_keeps_bounds_valid():
    env = _env(init_mode="full_space", precision_estimator="conditional", max_cycles=15)
    obs, _ = env.reset(seed=0)
    agents = list(obs.keys())
    rng = np.random.default_rng(1)
    for _ in range(15):
        if not env.agents:
            break
        acts = {a: rng.uniform(-1, 1, env.action_space(a).shape[0]).astype(np.float32)
                for a in env.agents}
        env.step(acts)
    ks = [env.n_predicates(a) for a in agents]
    assert max(ks) >= 1, "leave-corner must be reachable under exploration"
    for a in agents:
        assert np.all(env.lower[a] <= env.upper[a])
        assert np.all(env.lower[a] >= 0.0) and np.all(env.upper[a] <= 1.0)


def test_export_rule_state_is_authoritative_not_the_observation():
    """obs[:n] is a QUANTILE in this mode, so decoding it as `lower` is wrong."""
    env = _env(init_mode="full_space", precision_estimator="conditional")
    obs, _ = env.reset(seed=2)
    a0 = list(obs.keys())[0]
    env.a[a0][1], env.b[a0][1] = 0.25, 0.75
    env._sync_unit_bounds_from_quantiles(a0)
    st = env.export_rule_state(a0)
    lo = np.asarray(st["lower_bounds_normalized"], dtype=np.float32)
    assert np.all(np.isfinite(lo))
    # The observation's first n entries are `a`, and must NOT equal the bounds.
    o = env._get_observation(a0, 0.5, 0.5)
    assert not np.allclose(o[:env.n_features], lo), (
        "if these matched, obs-decoding would look correct and silently be wrong"
    )
    assert st["n_predicates"] == int(np.sum(st["active_features"]))


def test_quantile_empty_rule_is_not_potential_maximal():
    """k=0 covers everything at the base rate; on a majority class that clears the
    gate, so an ungated Phi would make DOING NOTHING optimal."""
    env = _env(init_mode="full_space", precision_estimator="conditional")
    obs, _ = env.reset(seed=0)
    a0 = list(obs.keys())[0]
    cls = env._get_class_for_agent(a0)
    phi_empty = env._potential(0.95, 1.0, cls, k=0, p_reset=0.0)
    phi_one = env._potential(0.95, 0.5, cls, k=1, p_reset=0.0)
    assert phi_one > phi_empty, "a real rule must beat the empty rule"


def test_quantile_potential_caps_precision_at_target():
    """Precision above target is worth nothing, else the gate slope makes the
    policy shrink to k = d."""
    env = _env(init_mode="full_space", precision_estimator="conditional")
    obs, _ = env.reset(seed=0)
    a0 = list(obs.keys())[0]
    cls = env._get_class_for_agent(a0)
    tgt = env._get_effective_precision_target(cls)
    at = env._potential(tgt, 0.4, cls, k=2, p_reset=0.0)
    above = env._potential(min(1.0, tgt + 0.09), 0.4, cls, k=2, p_reset=0.0)
    assert above == pytest.approx(at, abs=1e-9)


def test_terminal_floor_retreat_rescues_a_collapsed_episode():
    """Quantile mode enforces the coverage floor once, at the end, so a trajectory
    is free to collapse -- and must be able to retreat to the best feasible box it
    found. Measured without this: breast_cancer (d=30) ended collapsed on 100% of
    agent-episodes under a random policy (k=27/30, coverage 0.000); with it, 0%
    (k=3.0, coverage 0.773).
    """
    env = _env(init_mode="full_space", precision_estimator="conditional", max_cycles=6)
    obs, _ = env.reset(seed=0)
    a0 = list(obs.keys())[0]
    assert env.coverage_floor_mode == "terminal"

    # A feasible box the episode "found" earlier.
    good_a, good_b = env.a[a0].copy(), env.b[a0].copy()
    good_a[0], good_b[0] = 0.05, 0.95
    env._best_box[a0] = (env.lower[a0].copy(), env.upper[a0].copy(), good_a, good_b)
    env._best_box_score[a0] = 10.0

    # Now collapse the live box to something empty.
    env.a[a0][:] = 0.49
    env.b[a0][:] = 0.51
    env._sync_unit_bounds_from_quantiles(a0)

    zero = _noop(env, list(obs.keys()))
    for _ in range(env.max_cycles + 1):
        _, _, term, trunc, infos = env.step(zero)
        if term.get(a0) or trunc.get(a0):
            break
    assert np.allclose(env.a[a0], good_a) and np.allclose(env.b[a0], good_b), (
        "collapsed episode did not retreat to the best feasible box"
    )


def test_no_retreat_when_nothing_feasible_was_found():
    """With no feasible box the final one is kept -- never silently fabricated."""
    env = _env(init_mode="full_space", precision_estimator="conditional", max_cycles=4)
    obs, _ = env.reset(seed=1)
    a0 = list(obs.keys())[0]
    env._best_box[a0] = None
    env.a[a0][:] = 0.49
    env.b[a0][:] = 0.51
    env._sync_unit_bounds_from_quantiles(a0)
    kept_a = env.a[a0].copy()
    zero = _noop(env, list(obs.keys()))
    for _ in range(env.max_cycles + 1):
        _, _, term, trunc, _ = env.step(zero)
        if term.get(a0) or trunc.get(a0):
            break
    assert np.allclose(env.a[a0], kept_a)


def _constrain(env, agent, j, a_lo, b_hi):
    env.a[agent][:] = 0.0
    env.b[agent][:] = 1.0
    env.a[agent][j] = float(a_lo)
    env.b[agent][j] = float(b_hi)
    env._sync_unit_bounds_from_quantiles(agent)


def test_paper_yaml_enables_quantile_same_class_diversity():
    mae = yaml.safe_load(open(REPO / "BenchMARL" / "conf" / "anchor.yaml"))["env_config"]
    assert mae["same_class_diversity_weight"] == pytest.approx(0.25)
    assert mae["inter_class_overlap_weight"] == pytest.approx(0.25)
    assert mae["same_class_diversity_weight"] > 0.0


def test_quantile_empty_rule_is_not_a_claim():
    """k=0 is the start state, not a region. Overlap and union ignore it."""
    env = _env(init_mode="full_space", precision_estimator="empirical",
               agents_per_class=2, same_class_diversity_weight=1.0)
    env.reset(seed=0)
    for a in env.possible_agents:
        assert env.n_predicates(a) == 0
        assert env._is_real_rule(a) is False
        assert env._compute_inter_class_overlap_penalty(a) == pytest.approx(0.0)
        assert env._compute_same_class_overlap_penalty(a) == pytest.approx(0.0)
    for cls, m in env._compute_class_union_metrics().items():
        assert m["union_coverage"] == pytest.approx(0.0)
        assert m["union_precision"] == pytest.approx(0.0)


def test_quantile_union_excludes_empty_teammate():
    """A k=0 teammate must not drown Φ^∪ with the full space."""
    env = _env(init_mode="full_space", precision_estimator="empirical",
               agents_per_class=2)
    env.reset(seed=0)
    acting, idle = env.class_to_agents[0]
    _constrain(env, acting, 0, 0.05, 0.55)
    assert env.n_predicates(acting) >= 1
    assert env.n_predicates(idle) == 0
    union = env._compute_class_union_metrics()[0]
    acting_c = float(env._mask_in_box(acting)[env.y == 0].mean())
    assert union["union_coverage"] == pytest.approx(acting_c, abs=1e-9)
    assert union["union_coverage"] < 0.99


def test_quantile_union_keeps_terminated_rule():
    """A finished agent's k>=1 box still covers for teammates."""
    env = _env(init_mode="full_space", precision_estimator="empirical",
               agents_per_class=2)
    env.reset(seed=0)
    done_agent, live = env.class_to_agents[0]
    _constrain(env, done_agent, 0, 0.05, 0.55)
    cov_before = env._compute_class_union_metrics()[0]["union_coverage"]
    env.agents = [a for a in env.agents if a != done_agent]
    cov_after = env._compute_class_union_metrics()[0]["union_coverage"]
    assert cov_after == pytest.approx(cov_before, abs=1e-9)
    assert cov_after > 0.0


def test_quantile_interclass_simpson_zero_when_disjoint():
    env = _env(init_mode="full_space", precision_estimator="empirical",
               agents_per_class=1, inter_class_overlap_weight=1.0)
    env.reset(seed=0)
    a0 = env.class_to_agents[0][0]
    a1 = env.class_to_agents[1][0]
    _constrain(env, a0, 0, 0.01, 0.45)
    _constrain(env, a1, 0, 0.55, 0.99)
    # Unit boxes (not class-CDF images) so the test is about claims, not knots.
    n = env.n_features
    env.lower[a0], env.upper[a0] = np.zeros(n), np.ones(n)
    env.lower[a0][0], env.upper[a0][0] = 0.0, 0.5
    env.lower[a1], env.upper[a1] = np.zeros(n), np.ones(n)
    env.lower[a1][0], env.upper[a1][0] = 0.5, 1.0
    assert env._compute_inter_class_overlap_penalty(a0) == pytest.approx(0.0, abs=1e-9)
    assert env._compute_inter_class_overlap_penalty(a1) == pytest.approx(0.0, abs=1e-9)


def test_quantile_interclass_simpson_high_when_same_rows():
    env = _env(init_mode="full_space", precision_estimator="empirical",
               agents_per_class=1, inter_class_overlap_weight=1.0)
    env.reset(seed=0)
    a0 = env.class_to_agents[0][0]
    a1 = env.class_to_agents[1][0]
    _constrain(env, a0, 0, 0.05, 0.95)
    _constrain(env, a1, 0, 0.05, 0.95)
    n = env.n_features
    for agent in (a0, a1):
        env.lower[agent] = np.zeros(n)
        env.upper[agent] = np.ones(n)
        env.lower[agent][0], env.upper[agent][0] = 0.1, 0.9
    assert env._compute_inter_class_overlap_penalty(a0) > 0.5
    assert env._compute_inter_class_overlap_penalty(a1) > 0.5


def test_quantile_same_class_diversity_penalizes_copies():
    env = _env(init_mode="full_space", precision_estimator="empirical",
               agents_per_class=2, same_class_diversity_weight=1.0,
               inter_class_overlap_weight=0.0)
    env.reset(seed=0)
    a, b = env.class_to_agents[0]
    _constrain(env, a, 0, 0.05, 0.55)
    _constrain(env, b, 0, 0.05, 0.55)
    assert env._compute_same_class_overlap_penalty(a) == pytest.approx(1.0, abs=1e-6)
    n = env.n_features
    env.lower[b], env.upper[b] = np.zeros(n), np.ones(n)
    env.lower[b][0], env.upper[b][0] = 0.6, 1.0
    assert env._compute_same_class_overlap_penalty(a) == pytest.approx(0.0, abs=1e-9)
