"""MADA explains a class with its agents together: each agent's best rule, OR'd.

`revision.evaluate` used to pool every agent's boxes and keep the class's top-k,
so at k = 1 two of the three policies never contributed. Per-policy selection
picks from each agent's own pool exactly as the single RLDA policy picks from its
pool, and the class rule set is the union of those picks.
"""
import numpy as np

from revision.evaluate import _agent_pools, select_per_policy
from utils.metrics import RANKING_SCORE_LCB_COVERAGE

rng = np.random.default_rng(0)
X = rng.uniform(0, 1, size=(400, 2)).astype(np.float32)
# Class 1 lives in two separate corners; each agent learned one of them.
y_hat = (((X[:, 0] < 0.3) & (X[:, 1] < 0.3)) | ((X[:, 0] > 0.7) & (X[:, 1] > 0.7))).astype(int)


def box(lo, up):
    return {"lower_bounds_normalized": lo, "upper_bounds_normalized": up}


RULES = {
    "class_1": {
        "class": 1,
        "per_agent_results": {
            "agent_1_0": {"anchors": [box([0, 0], [0.3, 0.3]), box([0, 0], [0.2, 0.2])]},
            "agent_1_1": {"anchors": [box([0.7, 0.7], [1, 1])]},
        },
        "class_based_results": {"agent_1_1": {"anchors": [box([0.8, 0.8], [1, 1])]}},
    },
    "class_0": {"class": 0, "anchors": [box([0.3, 0], [0.7, 1])]},  # one policy (RLDA-like)
}


def test_pools_are_per_agent_and_include_class_level_rollouts():
    pools = _agent_pools(RULES, 1)
    assert set(pools) == {"agent_1_0", "agent_1_1"}
    assert len(pools["agent_1_0"]) == 2 and len(pools["agent_1_1"]) == 2
    assert _agent_pools(RULES, 0) is None  # single policy: pooled selection applies


def test_one_rule_per_agent_unioned():
    sel = select_per_policy(_agent_pools(RULES, 1), X, y_hat, y_hat, 1, k=1, min_support=10,
                            ranking_formula=RANKING_SCORE_LCB_COVERAGE)
    agents = sorted(r.extra["agent"] for r in sel.individual)
    assert agents == ["agent_1_0", "agent_1_1"]
    # The union covers both corners, which no single rule does.
    union = np.zeros(len(X), bool)
    for r in sel.individual:
        union |= r.mask
    assert union.sum() > max(r.mask.sum() for r in sel.individual)
    assert sel.union_metrics.coverage > max(r.metrics.coverage for r in sel.individual)


def test_same_box_from_two_agents_counts_once():
    rules = {"class_1": {"class": 1, "per_agent_results": {
        "a": {"anchors": [box([0, 0], [0.3, 0.3])]},
        "b": {"anchors": [box([0, 0], [0.3, 0.3])]},
    }}}
    sel = select_per_policy(_agent_pools(rules, 1), X, y_hat, y_hat, 1, k=1, min_support=10,
                            ranking_formula=RANKING_SCORE_LCB_COVERAGE)
    assert len(sel.individual) == 1


# ---------------------------------------------------------------------------
# Floor: a policy's pick enters the class OR only at D_val Fid >= floor.
# ---------------------------------------------------------------------------

from revision.evaluate import apply_policy_floor  # noqa: E402
from utils.metrics import select_topk_union  # noqa: E402
from utils.eval_harness import rules_from_anchors  # noqa: E402

JUNK = {"class_1": {"class": 1, "per_agent_results": {
    "good": {"anchors": [box([0, 0], [0.3, 0.3])]},          # Fid 1.0 on class 1
    "junk": {"anchors": [box([0, 0], [0.9, 0.9])]},          # most of the space: Fid ~ 0.1
}}}


def test_agent_below_floor_is_left_out_others_still_explain_the_class():
    kw = dict(k=1, min_support=10, ranking_formula=RANKING_SCORE_LCB_COVERAGE)
    both = select_per_policy(_agent_pools(JUNK, 1), X, y_hat, y_hat, 1, **kw)
    assert sorted(r.extra["agent"] for r in both.individual) == ["good", "junk"]
    floored = select_per_policy(_agent_pools(JUNK, 1), X, y_hat, y_hat, 1, floor=0.9, **kw)
    assert [r.extra["agent"] for r in floored.individual] == ["good"]
    assert floored.union_metrics.fidelity >= 0.9


def test_single_policy_below_floor_leaves_the_class_without_a_rule():
    ranked = rules_from_anchors([box([0, 0], [0.9, 0.9])], X, y_hat, y_hat, 1, class_conditional=True,
                                min_support=10, ranking_formula=RANKING_SCORE_LCB_COVERAGE, space="unit")
    sel = select_topk_union(ranked, y_hat, y_hat, 1, k=1)
    assert apply_policy_floor(sel, y_hat, y_hat, 1, None) is sel
    assert apply_policy_floor(sel, y_hat, y_hat, 1, 0.9) is None
