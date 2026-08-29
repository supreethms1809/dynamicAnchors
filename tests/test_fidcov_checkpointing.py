"""FidCovEvalCallback checkpoint-selection guards.

Two defects this pins down, both found after wine recorded 0 best_model saves
across all 8 evaluations and inference then died with prefer_model='best':

  1. mean_n used np.min over per-episode support, so ONE empty-box eval episode
     sent the whole checkpoint score to -inf. With best_score initialised to -inf,
     `score > best_score` was then False forever and best_model.zip was never
     written.
  2. A final-weights fallback wrote final_model into best_model.zip, which let the
     job proceed but made "best" mean "final, unselected" -- silently, and on the
     paper's reporting path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "single_agent"))

from anchor_trainer_sb3 import FidCovEvalCallback  # noqa: E402
from utils.metrics import ranking_score  # noqa: E402


class _StubEnv:
    """Replays a scripted list of (precision, coverage, n_covered, k) episodes."""

    def __init__(self, episodes):
        self.episodes = list(episodes)
        self._i = -1

    def reset(self, *a, **kw):
        self._i += 1
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        p, c, n, k = self.episodes[self._i % len(self.episodes)]
        info = {"anchor_precision": p, "anchor_coverage": c, "n_covered": n,
                "n_predicates": k}
        return np.zeros(3, dtype=np.float32), 0.0, True, False, info

    def n_predicates(self):
        p, c, n, k = self.episodes[self._i % len(self.episodes)]
        return int(k)


class _StubModel:
    def __init__(self):
        self.saved = []

    def predict(self, obs, deterministic=True):
        return np.zeros(2, dtype=np.float32), None

    def save(self, path):
        self.saved.append(str(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"stub")


def _cb(episodes, tmp_path):
    cb = FidCovEvalCallback(
        eval_env=_StubEnv(episodes),
        best_model_save_path=str(tmp_path / "best_model" / "class_0"),
        eval_freq=1,
        n_eval_episodes=len(episodes),
    )
    cb.model = _StubModel()
    cb.n_calls = 1
    return cb


def test_one_empty_episode_does_not_poison_the_checkpoint_score(tmp_path):
    """Regression: np.min over support made a single empty box score -inf."""
    episodes = [(0.90, 0.30, 40, 2), (0.85, 0.25, 25, 2),
                (0.80, 0.20, 18, 2), (0.70, 0.00, 0, 2)]
    cb = _cb(episodes, tmp_path)
    mean_p, mean_c, mean_n = cb._evaluate()

    assert np.isfinite(mean_p) and np.isfinite(mean_c)
    assert mean_n == int(round(np.mean([40, 25, 18, 0]))) == 21, (
        "support must be the MEAN across eval episodes, not the min"
    )
    score = ranking_score(mean_p, mean_c, n_covered=mean_n)
    assert np.isfinite(score) and score > 0.0

    cb._on_step()
    assert cb.model.saved, "a finite score must write best_model.zip"
    assert np.isfinite(cb.best_score)


def test_all_empty_episodes_still_rejected(tmp_path):
    """A checkpoint whose boxes are all empty must NOT be selected."""
    cb = _cb([(0.9, 0.0, 0, 2), (0.9, 0.0, 0, 2)], tmp_path)
    _, _, mean_n = cb._evaluate()
    assert mean_n == 0
    cb._on_step()
    assert cb.last_score == float("-inf")
    assert not cb.model.saved, "an all-empty checkpoint must not be saved"


def test_support_is_mean_not_sum_so_score_is_eval_budget_invariant(tmp_path):
    """sum() would make the score grow with n_eval_episodes."""
    one = _cb([(0.9, 0.3, 20, 2)], tmp_path)
    many = _cb([(0.9, 0.3, 20, 2)] * 6, tmp_path)
    assert one._evaluate()[2] == many._evaluate()[2] == 20


def test_zero_predicate_episodes_are_excluded(tmp_path):
    """k < 1 is not an anchor; such episodes must not enter the aggregate."""
    cb = _cb([(1.0, 1.0, 100, 0), (0.9, 0.3, 30, 2)], tmp_path)
    mean_p, _, mean_n = cb._evaluate()
    assert mean_p == pytest.approx(0.9)
    assert mean_n == 30


def test_no_final_weights_fallback_in_trainer_source():
    """The final->best copy must stay deleted.

    It let training 'succeed' while best_model held unselected final weights --
    iris class_1 shipped a byte-identical copy of final_model this way. Downstream
    (C-10 checkpoint selection, revision/evaluate) cannot detect the substitution,
    so this is asserted at the source level.
    """
    src = (REPO / "single_agent" / "anchor_trainer_sb3.py").read_text()
    marker = src[src.index("if os.path.exists(best_model_path):"):]
    marker = marker[:marker.index("TRAINING COMPLETE")]
    assert "Copied final weights" not in marker
    assert "RuntimeError" in marker, (
        "a missing validation-selected checkpoint must fail loudly at the end of "
        "training, not be papered over with final weights"
    )
