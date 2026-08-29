"""Post-training extraction helpers. Must not change training action mapping."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.inference_extract import (  # noqa: E402
    maddpg_executed_action,
    persist_box_from_episode,
    union_precision,
)
from revision.evaluate import _collect_anchors  # noqa: E402


def test_maddpg_executed_action_is_plain_tanh():
    param = torch.tensor([3.0, -3.0, 0.0])
    got = maddpg_executed_action(param)
    expected = torch.tanh(param)
    assert torch.allclose(got, expected)
    # The old bug was tanh(param / 3e7) ≈ 0 for typical logits.
    assert not torch.allclose(got, torch.tanh(param / 3e7))


def test_persist_box_uses_export_not_obs_prefix():
    n = 4
    a = np.zeros(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    q_star = np.full(n, 0.5, dtype=np.float32)
    obs = np.concatenate([a, b, q_star, np.array([0.9, 0.4, 1.0], dtype=np.float32)])
    true_lo = np.array([0.1, 0.2, 0.2, 0.3], dtype=np.float32)
    true_up = np.array([0.3, 0.8, 0.4, 0.7], dtype=np.float32)
    episode = {
        "final_observation": obs.tolist(),
        "lower_bounds_normalized": true_lo.tolist(),
        "upper_bounds_normalized": true_up.tolist(),
        "active_features": [True, False, True, False],
        "a": a.tolist(),
        "b": b.tolist(),
    }
    box = persist_box_from_episode(
        episode, {"sparsity_width_ratio": 0.95}, n, max_features_in_rule=-1
    )
    assert box is not None
    assert float(box["lower_normalized"][1]) == 0.0
    assert float(box["upper_normalized"][1]) == 1.0
    np.testing.assert_allclose(box["lower_normalized"][0], 0.1, atol=1e-6)
    assert not np.allclose(box["lower_normalized"], obs[:n])


def test_union_precision_prefers_fidelity():
    y = np.array([0, 0, 1, 1])
    y_hat = np.array([0, 1, 1, 1])
    mask = np.array([True, True, True, False])
    assert union_precision(mask, 1, y, y_hat) == 2.0 / 3.0
    assert union_precision(mask, 1, y, None) == 1.0 / 3.0


def test_collect_anchors_admits_unit_bounds_without_original():
    class_data = {
        "anchors": [{
            "lower_bounds_normalized": [0.1, 0.2],
            "upper_bounds_normalized": [0.4, 0.5],
        }]
    }
    got = _collect_anchors(class_data)
    assert len(got) == 1
