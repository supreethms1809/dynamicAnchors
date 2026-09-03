"""MADA agent-grouping guards.

The multi-agent layer used to collapse: `benchmarl_wrappers` built the
`PettingZooWrapper` WITHOUT a `group_map`, so torchrl's default parallel
grouping (agents named "str_int" grouped by "str") decided it instead, and
`share_policy_params: True` then put every agent in a group on one parameter
set. Result: `agents_per_class: 3` trained one network per class replicated
three times, and `agents_per_class: 1` trained ONE network for every class.

Nothing in the suite covered grouping, which is why it survived several
reward-shaping revisions. These tests pin the three pieces that have to agree:
the env's group map, the wrapper passing it through, and the parameter slicing
that makes per-agent policies extractable once params are no longer shared.
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
from BenchMARL.anchor_trainer import _slice_agent_params  # noqa: E402


class _Clf(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.lin = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.lin(x)


def _env(agents_per_class: int, n_classes: int = 3, n_features: int = 4):
    rng = np.random.default_rng(0)
    X = np.clip(
        np.vstack([
            (0.2 + 0.3 * c) + 0.05 * rng.standard_normal((30, n_features))
            for c in range(n_classes)
        ]),
        0.0, 1.0,
    ).astype(np.float32)
    y = np.repeat(np.arange(n_classes), 30).astype(int)

    cfg = dict(yaml.safe_load(
        open(REPO / "BenchMARL" / "conf" / "anchor.yaml")
    )["env_config"])
    cfg.update({
        "X_min": np.zeros(n_features, dtype=np.float32),
        "X_range": np.ones(n_features, dtype=np.float32),
        "agents_per_class": agents_per_class,
        "max_cycles": 4,
    })
    return AnchorEnv(
        X_unit=X, X_std=X, y=y,
        feature_names=[f"f{i}" for i in range(n_features)],
        classifier=_Clf(n_features, n_classes),
        env_config=cfg,
    )


@pytest.mark.parametrize("agents_per_class", [1, 3])
def test_group_map_is_one_group_per_class(agents_per_class):
    """A group is a CLASS, whatever agents_per_class is.

    This is the scope the MADDPG critic centralises over, and the scope the
    same-class shared reward and diversity penalty are defined on.
    """
    env = _env(agents_per_class)
    gm = env.group_map

    assert sorted(gm) == [f"class_{c}" for c in range(3)]
    for cls in range(3):
        agents_c = gm[f"class_{cls}"]
        assert len(agents_c) == agents_per_class
        assert all(env.agent_to_class[a] == cls for a in agents_c)

    # Every agent appears exactly once (torchrl's check_marl_grouping contract).
    flat = [a for agents in gm.values() for a in agents]
    assert sorted(flat) == sorted(env.possible_agents)


def test_group_map_is_never_grouped_by_torchrl_name_parsing():
    """Regression: with agents_per_class == 1 the names are agent_0/agent_1/...,
    which torchrl's default parser collapses into a single group "agent" holding
    every class. The observation carries no class id, so one shared actor would
    be class-blind."""
    gm = _env(agents_per_class=1).group_map
    assert "agent" not in gm
    assert len(gm) == 3, f"classes must not share a group, got {gm}"


def test_wrapper_passes_the_env_group_map_to_torchrl():
    """The wrapper must hand torchrl an explicit group_map. Omitting it is the
    original defect and leaves no trace at runtime."""
    import inspect

    from BenchMARL import benchmarl_wrappers

    src = inspect.getsource(benchmarl_wrappers.AnchorTaskClass.get_env_fun)
    assert "PettingZooWrapper(" in src
    call = src[src.index("PettingZooWrapper("):]
    call = call[: call.index(")")]
    assert "group_map=" in call, (
        "PettingZooWrapper must receive an explicit group_map; without it "
        "torchrl silently regroups agents by name"
    )


def test_share_policy_params_is_off():
    """With a class group holding agents_per_class agents, sharing policy params
    makes them one network -- identical copies that cannot diversify."""
    cfg = yaml.safe_load(
        open(REPO / "BenchMARL" / "conf" / "base_experiment.yaml")
    )
    assert cfg["share_policy_params"] is False


@pytest.mark.parametrize("algo", ["maddpg", "masac"])
def test_share_param_critic_is_off(algo):
    """One Q per group, expanded to every agent, cannot carry per-agent credit:
    the reward is shape (n_agents, 1) and the prediction is shape (1,), so the
    critic regresses to the group mean. Benign only while the agents were
    identical copies."""
    cfg = yaml.safe_load(open(REPO / "BenchMARL" / "conf" / f"{algo}.yaml"))
    assert cfg["share_param_critic"] is False


def _multiagent_actor_state_dict(n_agents: int, share_params: bool):
    from torchrl.modules import MultiAgentMLP

    m = MultiAgentMLP(
        n_agent_inputs=15, n_agent_outputs=8, n_agents=n_agents,
        centralised=False, share_params=share_params,
        depth=2, num_cells=[32, 32],
    )
    # Extraction saves the actor module's state_dict, which nests the MLP.
    return {f"0.mlp.{k}": v for k, v in m.state_dict().items()}


def test_slice_agent_params_yields_distinct_single_agent_actors():
    """share_params=False stores [n_agents, out, in] per layer. Each saved
    policy must be that agent's own slice, in the single-agent layout
    inference.load_policy_model infers dimensions from."""
    n_agents = 3
    sd = _multiagent_actor_state_dict(n_agents, share_params=False)
    assert sd["0.mlp.params.0.weight"].shape == (n_agents, 32, 15)

    firsts = []
    for k in range(n_agents):
        sliced = _slice_agent_params(sd, agent_idx=k, n_agents=n_agents)
        w = sliced["0.mlp.params.0.weight"]
        assert w.shape == (32, 15), "agent dim must be gone"
        assert sliced["0.mlp.params.__batch_size"] == torch.Size([])
        firsts.append(w)

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            assert not torch.equal(firsts[i], firsts[j]), (
                f"agents {i} and {j} share weights -- the collapse is back"
            )


def test_slice_agent_params_passes_shared_layout_through():
    """share_params=True (or a group of one) has no agent dim: leave it alone."""
    sd = _multiagent_actor_state_dict(3, share_params=True)
    assert _slice_agent_params(sd, agent_idx=0, n_agents=3) is sd
    assert _slice_agent_params(sd, agent_idx=2, n_agents=3) is sd


def test_slice_agent_params_strips_a_group_of_one():
    """agents_per_class == 1 still gets a leading dim of 1 under
    share_params=False. Leaving it makes load_policy_model read in_features off
    the wrong axis and build an MLP that cannot load."""
    sd = _multiagent_actor_state_dict(1, share_params=False)
    assert sd["0.mlp.params.0.weight"].shape == (1, 32, 15)

    sliced = _slice_agent_params(sd, agent_idx=0, n_agents=1)
    assert sliced["0.mlp.params.0.weight"].shape == (32, 15)
    assert sliced["0.mlp.params.__batch_size"] == torch.Size([])


def test_slice_agent_params_rejects_an_out_of_range_index():
    sd = _multiagent_actor_state_dict(3, share_params=False)
    with pytest.raises(ValueError, match="out of range"):
        _slice_agent_params(sd, agent_idx=3, n_agents=3)
