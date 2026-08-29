"""Post-training helpers for RLDA/MADA inference. Not used during training."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from utils.metrics import sparsify_box


def maddpg_executed_action(param: torch.Tensor) -> torch.Tensor:
    """Map MADDPG actor params to the action training executed.

    BenchMARL MADDPG is ProbabilisticActor(TanhDelta(low=-1, high=1)), so the
    env action is tanh(param). Do not rescale by a constant first.
    """
    return torch.tanh(param)


def generation_split_arrays(env_data: Dict[str, Any], env_config: Dict[str, Any]):
    """Split used to *generate* candidate boxes (fitting), default train."""
    split = str(env_config.get("generation_split", "train") or "train").lower()
    if split == "val" and env_data.get("X_val_unit") is not None:
        return (env_data["X_val_unit"], env_data["X_val_std"], env_data["y_val"], "validation")
    if split == "test" and env_data.get("X_test_unit") is not None:
        return (env_data["X_test_unit"], env_data["X_test_std"], env_data["y_test"], "test")
    return (env_data["X_unit"], env_data["X_std"], env_data["y"], "train")


def _denormalize_unit_bounds(
    lower_n: np.ndarray,
    upper_n: np.ndarray,
    env_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    x_min = env_config.get("X_min")
    x_range = env_config.get("X_range")
    scaler_mean = env_config.get("scaler_mean")
    scaler_scale = env_config.get("scaler_scale")
    if x_min is None or x_range is None:
        return lower_n.copy(), upper_n.copy()
    lower = (lower_n * x_range) + x_min
    upper = (upper_n * x_range) + x_min
    if scaler_mean is not None and scaler_scale is not None:
        lower = lower * scaler_scale + scaler_mean
        upper = upper * scaler_scale + scaler_mean
    return np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)


def persist_box_from_episode(
    episode_data: Optional[Dict[str, Any]],
    env_config: Dict[str, Any],
    n_features: int,
    max_features_in_rule: int = -1,
) -> Optional[Dict[str, Any]]:
    """Unit box + original-scale box from export_rule_state (not obs quantiles).

    Quantile obs[:n] is `a`, not `lower`. Prefer lower_bounds_normalized from
    export_rule_state / final_lower. Sparsify with the quantile active mask.
    """
    if not episode_data:
        return None
    lo = episode_data.get("lower_bounds_normalized")
    up = episode_data.get("upper_bounds_normalized")
    if lo is None:
        lo = episode_data.get("final_lower")
    if up is None:
        up = episode_data.get("final_upper")
    if lo is None or up is None:
        obs = episode_data.get("final_observation")
        if obs is None:
            return None
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        # Hull-era synthetic final_obs is 2n+2 unit bounds; quantile policy obs is 3n+3.
        if obs.shape[0] < 2 * n_features:
            return None
        lo, up = obs[:n_features], obs[n_features:2 * n_features]
    policy_lo = np.asarray(lo, dtype=np.float32).reshape(-1)
    policy_up = np.asarray(up, dtype=np.float32).reshape(-1)
    if policy_lo.shape[0] != n_features or policy_up.shape[0] != n_features:
        return None

    active = episode_data.get("active_features")
    if active is not None:
        active = np.asarray(active, dtype=bool).reshape(-1)
        if active.shape[0] != n_features:
            active = None

    lower_n, upper_n, active_out = sparsify_box(
        policy_lo,
        policy_up,
        sparsity_width_ratio=float(env_config.get("sparsity_width_ratio", 0.95)),
        max_features=max_features_in_rule,
        active_mask=active,
    )
    # D-01: an episode that added no predicate ends all-corner, i.e. the EMPTY rule.
    # Emitting bounds for it makes [0,1]^d a scorable candidate: it covers every row,
    # so any union containing it reports coverage 1.0 at the class prior (measured on
    # iris class 0: union Fid 0.333 = prior, cov 1.000). The empty rule is a failed
    # episode, not an anchor -- return None so no bounds are persisted for it.
    if not bool(np.any(active_out)):
        return None
    lower, upper = _denormalize_unit_bounds(lower_n, upper_n, env_config)
    a = episode_data.get("a")
    b = episode_data.get("b")
    return {
        "policy_lower_normalized": policy_lo,
        "policy_upper_normalized": policy_up,
        "lower_normalized": np.asarray(lower_n, dtype=np.float32),
        "upper_normalized": np.asarray(upper_n, dtype=np.float32),
        "lower": lower,
        "upper": upper,
        "active_features": active_out,
        "a": None if a is None else np.asarray(a, dtype=np.float64).reshape(-1),
        "b": None if b is None else np.asarray(b, dtype=np.float64).reshape(-1),
    }


def classifier_labels(classifier: Any, X_std: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Argmax labels of `classifier` on standardized features. Not used in training."""
    from utils.networks import predict_proba_torch

    if hasattr(classifier, "eval"):
        classifier.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(np.asarray(X_std, dtype=np.float32)).to(device)
        return predict_proba_torch(classifier, inputs).cpu().numpy().argmax(axis=1)


def union_precision(mask: np.ndarray, target_class: int, y: np.ndarray, y_hat: Optional[np.ndarray]) -> float:
    """Fidelity if y_hat is given, else label purity. Empty mask is 0."""
    if not np.any(mask):
        return 0.0
    src = y_hat if y_hat is not None else y
    return float((src[mask] == target_class).mean())
