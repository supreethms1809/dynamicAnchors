"""A zero-width quantile dim is an EQUALITY predicate, not an invalid box.

Bounds are Q_j(a_j), Q_j(b_j) through the class empirical CDF, so a band falling
between two tied values yields Q_j(a_j) == Q_j(b_j) -- the predicate "f_j = v",
with the real inclusive support of every row tied at v.

Rejecting it pinned `done = False` for entire runs: `both_targets_met` (+5.0)
and all partial terminal credit never fired. Measured collapse at a [0.30, 0.70]
band: uci_credit 5/15 dims before the policy narrows anything; iris 4/4 at width
0.02.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "BenchMARL"))
sys.path.insert(0, str(REPO / "single_agent"))

from utils import quantile_mdp as qmdp  # noqa: E402
from utils.metrics import active_feature_mask, box_mask  # noqa: E402


# ---------------------------------------------------------------- the mechanism

def _tied_column(n_ties: int = 40, n_tail: int = 10) -> np.ndarray:
    """A class column whose middle mass sits on a single repeated value."""
    return np.sort(np.concatenate([
        np.linspace(0.0, 0.2, n_tail),
        np.full(n_ties, 0.5),
        np.linspace(0.8, 1.0, n_tail),
    ]))


def test_ties_collapse_the_unit_interval_to_zero_width():
    v = [_tied_column()]
    lo, up = qmdp.quantile_to_unit_bounds(np.array([0.35]), np.array([0.65]),
                                          v, eps=1e-3)
    assert lo[0] == up[0] == pytest.approx(0.5), (
        "a band inside the tied mass must map to a single value"
    )


def test_the_zero_width_box_still_has_real_support():
    """The equality predicate is not vacuous -- box_mask is inclusive."""
    col = _tied_column(n_ties=40)
    X = col.reshape(-1, 1).astype(np.float32)
    covered = box_mask(X, np.array([0.5]), np.array([0.5]))
    assert covered.sum() == 40, "every tied row satisfies f = 0.5"


def test_zero_width_dim_counts_as_a_predicate():
    """k must be >= 1, or the rule reads as the empty rule downstream."""
    a, b = np.array([0.35, 0.0]), np.array([0.65, 1.0])
    assert qmdp.constrained_mask(a, b, eps=1e-3).tolist() == [True, False]
    lo, up = np.array([0.5, 0.0]), np.array([0.5, 1.0])
    assert active_feature_mask(lo, up).tolist() == [True, False]


# ------------------------------------------------------- the validator itself

def _validator_src(path: Path) -> str:
    src = path.read_text()
    i = src.index("bounds_valid = True")
    return src[i:i + 900]


@pytest.mark.parametrize("rel", [
    "BenchMARL/environment.py",
    "single_agent/single_agentENV.py",
])
def test_validator_is_representation_aware(rel):
    """Quantile MDP admits equality (lower == upper); only inverted boxes fail."""
    block = _validator_src(REPO / rel)
    assert "> agent_upper" in block or "> self.upper" in block
    assert ">= agent_upper" not in block and ">= self.upper" not in block


@pytest.mark.parametrize("lower,upper,expected_valid", [
    ([0.5], [0.5], True),    # equality predicate
    ([0.7], [0.3], False),   # genuinely inverted
    ([0.2], [0.8], True),
])
def test_degenerate_rule_matches_the_env_expression(lower, upper, expected_valid):
    """Pins the truth table both envs now implement."""
    lo, up = np.array(lower), np.array(upper)
    degenerate = lo > up
    assert (not degenerate.any()) is expected_valid


# ------------------------------------------------------------ real datasets

@pytest.mark.parametrize("dataset,min_collapsed", [("uci_credit", 1)])
def test_real_dataset_collapses_dims_at_a_wide_band(dataset, min_collapsed):
    """Guards the claim that this fires before any narrowing takes place."""
    from tabular_datasets import TabularDatasetLoader

    ld = TabularDatasetLoader(dataset_name=dataset, random_state=42)
    ld.load_dataset()
    ld.preprocess_data()
    d = ld.get_anchor_env_data()
    Xu, y = np.asarray(d["X_unit"]), np.asarray(d["y"])

    cd = qmdp.fit_train_cdfs(Xu, y, 0)
    n = Xu.shape[1]
    lo, up = qmdp.quantile_to_unit_bounds(np.full(n, 0.30), np.full(n, 0.70),
                                          cd["v_class"], eps=1e-3)
    collapsed = int((lo >= up).sum())
    assert collapsed >= min_collapsed
    assert int((lo > up).sum()) == 0, "none of them are actually inverted"
