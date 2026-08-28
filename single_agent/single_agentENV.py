import functools
import numpy as np
import torch
from typing import Dict, Optional, Tuple, Any, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.device_utils import get_device
from utils.networks import predict_proba_torch
import logging
logger = logging.getLogger(__name__)

# Import gymnasium (or gym for backward compatibility)
try:
    import gymnasium as gym
    from gymnasium import spaces, Env
    GYM_VERSION = "gymnasium"
except ImportError:
    try:
        import gym
        from gym import spaces, Env
        GYM_VERSION = "gym"
    except ImportError:
        raise ImportError("Please install gymnasium: pip install gymnasium")


class SingleAgentAnchorEnv(Env):
    """
    Single-agent Gymnasium environment for finding anchors.
    
    Compatible with Stable-Baselines3 and other single-agent RL libraries.
    
    Observation Space: Box of shape (2 * n_features + 2,)
        - First n_features: lower bounds for each feature
        - Next n_features: upper bounds for each feature
        - Next 1: current precision
        - Next 1: current coverage
    
    Action Space: Box of shape (2 * n_features,)
        - First n_features: delta for lower bounds (clipped to [-1, 1])
        - Next n_features: delta for upper bounds (clipped to [-1, 1])
    """
    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        X_unit: Optional[np.ndarray] = None,
        X_std: Optional[np.ndarray] = None,
        y: np.ndarray = None,
        feature_names: list = None,
        classifier = None,
        device: str = "cpu",
        target_class: Optional[int] = None,
        env_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        if env_config is None:
            env_config = {}
        
        normalize_data = env_config.get("normalize_data", False)
        
        if normalize_data:
            if X_std is None:
                raise ValueError("X_std must be provided when normalize_data=True")
            X_unit_normalized, X_min, X_range = self._normalize_data(X_std, env_config)
            X_unit = X_unit_normalized
            if env_config.get("X_min") is None:
                env_config["X_min"] = X_min
            if env_config.get("X_range") is None:
                env_config["X_range"] = X_range
        else:
            if X_unit is None or X_std is None:
                raise ValueError("Both X_unit and X_std must be provided when normalize_data=False")
        
        self.X_unit = X_unit
        self.X_std = X_std
        self.y = y.astype(int)
        self.feature_names = feature_names
        self.n_features = X_unit.shape[1]
        self.classifier = classifier
        self.device = get_device(device)
        
        # Single agent: determine target class
        if target_class is None:
            # Default to first class if not specified
            unique_classes = sorted(np.unique(y).tolist())
            if len(unique_classes) == 0:
                raise ValueError("No classes found in y")
            target_class = unique_classes[0]
        
        self.target_class = target_class
        
        # Initialize observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            # lower + upper + fidelity + coverage + episode phase
            shape=(2 * self.n_features + 3,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * self.n_features,),  # delta for lower + delta for upper
            dtype=np.float32
        )
        
        step_fracs = env_config.get("step_fracs", (0.005, 0.01, 0.02))
        if step_fracs is None or len(step_fracs) == 0:
            raise ValueError("step_fracs cannot be empty. Provide at least one step fraction value.")
        self.step_fracs = step_fracs
        self.min_width = env_config.get("min_width", 0.05)
        self.alpha = env_config.get("alpha", 0.7)
        self.beta = env_config.get("beta", 0.6)
        self.gamma = env_config.get("gamma", 0.1)

        self.directions = ("shrink_lower", "expand_lower", "shrink_upper", "expand_upper")
        self.precision_target = env_config.get("precision_target", 0.9)
        self.coverage_target = env_config.get("coverage_target", 0.2)
        self.precision_blend_lambda = env_config.get("precision_blend_lambda", 1.0)
        self.drift_penalty_weight = env_config.get("drift_penalty_weight", 0.05)

        self.use_perturbation = env_config.get("use_perturbation", False)
        self.perturbation_mode = env_config.get("perturbation_mode", "bootstrap")
        self.n_perturb = env_config.get("n_perturb", 2048)
        self.X_min = env_config.get("X_min", None)
        self.X_range = env_config.get("X_range", None)
        self.scaler_mean = env_config.get("scaler_mean", None)
        self.scaler_scale = env_config.get("scaler_scale", None)
        if self.scaler_mean is not None:
            self.scaler_mean = np.asarray(self.scaler_mean, dtype=np.float32)
        if self.scaler_scale is not None:
            self.scaler_scale = np.asarray(self.scaler_scale, dtype=np.float32)
        self.rng = env_config.get("rng", None)
        if self.rng is None:
            self.rng = np.random.default_rng(42)
        self.min_coverage_floor = env_config.get("min_coverage_floor", 0.005)
        # "step" = enforce the coverage floor every step (original), "terminal" =
        # only at episode end. See the step() block for the measured rationale.
        self.coverage_floor_mode = str(env_config.get("coverage_floor_mode", "terminal")).lower()
        # Relative floor: a floor of 1 sample is degenerate -- a box holding one point
        # is not a measurement (its Wilson interval spans most of [0,1]). Expressing
        # the floor as min_support samples makes it bind where it is meaningful.
        self.coverage_floor_relative = bool(env_config.get("coverage_floor_relative", True))
        self.coverage_floor_min_support = int(env_config.get("min_support", 10))
        if self.coverage_floor_relative:
            try:
                _n = int(np.asarray(y).shape[0])
            except Exception:
                _n = 0
            if _n > 0:
                _rel = float(self.coverage_floor_min_support) / float(_n)
                if _rel != self.min_coverage_floor:
                    logger.info(
                        f"  coverage floor: {self.min_coverage_floor:.6f} -> {_rel:.6f} "
                        f"({self.coverage_floor_min_support} of {_n} samples; relative floor)"
                    )
                self.min_coverage_floor = _rel
        # B2 fix 1: penalty charged when an action is reverted at the coverage
        # floor. Default 0.0 — the revert already removes the action's effect.
        # Set negative (e.g. -0.05, the old value) to restore the old behaviour.
        self.coverage_floor_penalty_value = float(env_config.get("coverage_floor_penalty", 0.0))
        # B2 fix 2: bisection steps used to project an infeasible action back onto
        # the coverage floor. 0 restores the old hard-revert behaviour.
        self.coverage_floor_projection_steps = int(env_config.get("coverage_floor_projection_steps", 6))
        self.coverage_floor_projection_alpha = 0.0
        # B2 rebalance: width of the precision ramp below target over which coverage
        # credit goes 0 -> 1. Smaller = stricter fidelity-first curriculum.
        self.gate_margin = float(env_config.get("gate_margin", 0.10))
        self.js_penalty_weight = env_config.get("js_penalty_weight", 0.05)
        self.initial_window = env_config.get("initial_window", 0.1)
        # Neighborhood-hull init: cover k nearest class points around the start,
        # not a ±initial_window cube (high-d cubes are singletons and do not transfer).
        self.init_min_neighbors = int(env_config.get("init_min_neighbors", 5))
        self.init_neighbor_frac = float(env_config.get("init_neighbor_frac", 0.1))
        self.init_max_neighbors = int(env_config.get("init_max_neighbors", 20))
        # Cap start hull so class-cond C is at most this fraction of τ_C.
        # Wins over init_min_neighbors (wine n≈35, τ_C=0.2 → k=3, not 5).
        self.init_coverage_frac_of_target = float(
            env_config.get("init_coverage_frac_of_target", 0.5)
        )
        # Only terminate on both_targets_met (disable excellent/high_prec/close).
        self.strict_target_termination = bool(
            env_config.get("strict_target_termination", True)
        )
        # Terminal bonus / target termination require C > C_init + 1/n_class.
        self.require_coverage_gain_to_terminate = bool(
            env_config.get("require_coverage_gain_to_terminate", True)
        )
        self._coverage_at_reset = 0.0
        self._coverage_gain_eps = 0.0
        # 0.0 = always snap a mean/cluster centroid to the nearest class row.
        # A 0.5 threshold left means sitting in empty space at distance 0.3.
        self.centroid_snap_threshold = float(env_config.get("centroid_snap_threshold", 0.0))
        self.fixed_instances_per_class = env_config.get("fixed_instances_per_class", None)
        self.cluster_centroids_per_class = env_config.get("cluster_centroids_per_class", None)
        self.training_instances_per_class = env_config.get("training_instances_per_class", None)
        self.training_instance_ratio = env_config.get("training_instance_ratio", 0.5)  # Base ratio (fallback)
        self.training_instance_ratios_per_class = env_config.get("training_instance_ratios_per_class", None)  # Class-specific ratios
        self.use_random_sampling = env_config.get("use_random_sampling", False)
        self.use_class_centroids = env_config.get("use_class_centroids", True)  # Default: use centroids for initialization
        # Explicit class-based start point (unit space). Set per rollout by inference
        # to cycle through diversified starts (k-means centroids + random class
        # samples) instead of drawing a random centroid every reset.
        self.class_init_point = env_config.get("class_init_point", None)
        
        self.eval_on_test_data = env_config.get("eval_on_test_data", False)
        self.eval_split = env_config.get("eval_split", "test" if self.eval_on_test_data else "train")
        self.X_val_unit = env_config.get("X_val_unit", None)
        self.X_val_std = env_config.get("X_val_std", None)
        self.y_val = None if env_config.get("y_val") is None else np.asarray(env_config["y_val"]).astype(int)
        if self.eval_on_test_data:
            X_test_unit = env_config.get("X_test_unit", None)
            X_test_std = env_config.get("X_test_std", None)
            y_test = env_config.get("y_test", None)
            if X_test_unit is None or X_test_std is None or y_test is None:
                raise ValueError("eval_on_test_data=True requires X_test_unit, X_test_std, and y_test")
            self.X_test_unit = X_test_unit
            self.X_test_std = X_test_std
            self.y_test = y_test.astype(int)
        else:
            self.X_test_unit = None
            self.X_test_std = None
            self.y_test = None

        self.max_action_scale = env_config.get("max_action_scale", 0.1)
        self.min_absolute_step = env_config.get("min_absolute_step", 0.001)
        self.min_steps_before_termination = int(
            env_config.get("min_steps_before_termination", 2)
        )
        
        # Coverage bonus weights (read from config, defaults match reduced values)
        self.coverage_bonus_weight_met = env_config.get("coverage_bonus_weight_met", 0.01)
        self.coverage_bonus_weight_high_prec = env_config.get("coverage_bonus_weight_high_prec", 0.03)
        self.coverage_bonus_weight_high_prec_progress = env_config.get("coverage_bonus_weight_high_prec_progress", 0.07)
        self.coverage_bonus_weight_high_prec_distance = env_config.get("coverage_bonus_weight_high_prec_distance", 0.02)
        self.coverage_bonus_weight_reasonable_prec = env_config.get("coverage_bonus_weight_reasonable_prec", 0.01)
        self.coverage_bonus_weight_reasonable_prec_progress = env_config.get("coverage_bonus_weight_reasonable_prec_progress", 0.02)
        
        # Target class bonus weight (read from config, default matches reduced value)
        self.target_class_bonus_weight = env_config.get("target_class_bonus_weight", 0.02)
        
        # Environment mode: "training", "evaluation", or "inference"
        # Termination counters are reset in reset() for evaluation/inference modes
        self.mode = env_config.get("mode", "training")  # Default to training
        
        # Termination reason counters (diagnostics only). The old max-count
        # mechanism that permanently disabled overused reasons mid-training was
        # removed: it made the MDP non-stationary under the replay buffer. The
        # terminal bonus now provides the incentive it was approximating.
        self._reset_termination_counters()  # This will check self.mode and disable lenient conditions in inference mode
        
        # Multi-agent config options (kept for API compatibility, but not used in single-agent)
        # Single-agent environments are independent (one per class), so these don't apply
        self.inter_class_overlap_weight = env_config.get("inter_class_overlap_weight", 0.1)
        self.shared_reward_weight = env_config.get("shared_reward_weight", 0.2)
        # Optional: class union metrics weights (not used in single-agent, but kept for compatibility)
        self.class_union_cov_weight = env_config.get("class_union_cov_weight", 0.0)
        self.class_union_prec_weight = env_config.get("class_union_prec_weight", 0.0)
        self.same_class_diversity_weight = env_config.get("same_class_diversity_weight", 0.0)
        
        x_star_unit_config = env_config.get("x_star_unit", None)
        if x_star_unit_config is not None:
            if isinstance(x_star_unit_config, dict):
                self.x_star_unit = x_star_unit_config.get("agent_0", x_star_unit_config.get(self.agent_name, None))
            else:
                self.x_star_unit = x_star_unit_config
        else:
            self.x_star_unit = None
        
        # Store original prediction for instance-based anchors (matches original Anchor paper)
        # Value: original prediction (int) for the instance
        self.original_prediction = None

        # Single agent: use direct variables instead of dictionaries
        self.lower = None
        self.upper = None
        self.prev_lower = None
        self.prev_upper = None
        self.box_history = []
        self.coverage_floor_hits = 0
        self.timestep = None
        # Read max_cycles from config - no hardcoded default to ensure YAML settings are respected
        self.max_cycles = env_config.get("max_cycles")
        if self.max_cycles is None:
            raise ValueError("max_cycles must be specified in env_config. Check your YAML config file.")
        self.max_cycles = int(self.max_cycles)

        # --- Potential-based reward shaping state ---
        # Terminal bonus paid when an episode terminates with targets met. It must
        # dominate anything farmable from per-step terms in the remaining steps,
        # otherwise hovering just below the targets beats terminating.
        self.terminal_bonus = env_config.get("terminal_bonus", 5.0)
        # B1: partial terminal credit. The all-or-nothing terminal_bonus requires
        # BOTH targets, which is unreachable above ~13 dimensions -- measured 0%
        # termination on 6 of 10 dataset x class combinations -- so the only
        # non-potential signal left was the movement penalties, whose optimum is
        # "do not move". This pays terminal_bonus * (C / tau_C) at episode end when
        # precision holds, so progress toward coverage is worth something even when
        # the target is missed. Unlike a term inside Phi, a one-off terminal reward
        # does not telescope, so it genuinely changes the optimal policy.
        # 0.0 restores the old all-or-nothing behaviour.
        self.partial_terminal_credit = float(env_config.get("partial_terminal_credit", 1.0))

        # Cached classifier probabilities over the train/test sets (computed lazily,
        # once). Bootstrap perturbation evaluates dataset rows, so probabilities can
        # be looked up instead of re-running the classifier every step.
        self._cached_probs = {"train": None, "val": None, "test": None}
        self.n_blackbox_queries = 0
        self.categorical_indices = list(env_config.get("categorical_indices") or [])
        self.categorical_value_names = {
            int(k): list(v)
            for k, v in (env_config.get("categorical_value_names") or {}).items()
        }
        self.categorical_freeze = env_config.get("categorical_freeze", "instance")
        self.sparsity_width_ratio = float(env_config.get("sparsity_width_ratio", 0.95))
        # B1: weight on the fraction of dimensions released to (near) full range.
        # 0.0 disables dimension release and restores the old potential.
        self.sparsity_weight = float(env_config.get("sparsity_weight", 0.5))
        self.ranking_score_formula = env_config.get("ranking_score_formula", "precision_coverage")
        self.top_k_rules_by_score = env_config.get("top_k_rules_by_score", 5)
        self.min_support = int(env_config.get("min_support", 10))

        # Metrics from the last reset()/step(), reused as prev metrics by the next
        # step(). This halves classifier calls AND makes gains telescope exactly:
        # prev and current come from the same sample draw (common random numbers),
        # so the shaping reward carries no resampling noise.
        self._last_step_metrics = None

        # Class-aware effective precision target: an absolute target (e.g. 0.90) is
        # unreachable for minority/overlapping classes, which both gates the coverage
        # phase of the reward forever and blocks termination. Cap the target by what
        # the classifier itself achieves for this class on training data.
        self.use_class_aware_targets = env_config.get("use_class_aware_targets", True)
        self.precision_target_effective = self._compute_effective_precision_target()
        self._log_effective_config()

    def _log_effective_config(self) -> None:
        """Print the knobs this env actually uses (YAML/CLI after any trainer merge)."""
        bits = [
            f"mode={self.mode}",
            f"precision_target={self.precision_target}",
            f"precision_target_effective={self.precision_target_effective:.4f}",
            f"coverage_target={self.coverage_target}",
            f"max_cycles={self.max_cycles}",
            f"use_perturbation={self.use_perturbation}",
            f"precision_blend_lambda={self.precision_blend_lambda}",
            f"training_instance_ratio={self.training_instance_ratio}",
            f"eval_split={self.eval_split}",
            f"eval_on_test_data={self.eval_on_test_data}",
            f"strict_target_termination={self.strict_target_termination}",
            f"require_coverage_gain_to_terminate={self.require_coverage_gain_to_terminate}",
            f"init_coverage_frac_of_target={self.init_coverage_frac_of_target}",
            f"init_min_neighbors={self.init_min_neighbors}",
            f"centroid_snap_threshold={self.centroid_snap_threshold}",
            f"min_steps_before_termination={self.min_steps_before_termination}",
            f"alpha={self.alpha} beta={self.beta} gamma={self.gamma}",
            f"terminal_bonus={self.terminal_bonus}",
            f"min_coverage_floor={self.min_coverage_floor}",
            f"use_class_aware_targets={self.use_class_aware_targets}",
            f"categorical_freeze={self.categorical_freeze}",
        ]
        logger.info("EFFECTIVE ENV CONFIG (honored YAML/CLI): " + " ".join(bits))

    # SS: This is a helper method to normalize the data. It is used to normalize the data for the perturbation sampling.
    @staticmethod
    def _normalize_data(X_std: np.ndarray, env_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_min = env_config.get("X_min", None)
        X_range = env_config.get("X_range", None)
        
        if X_min is None or X_range is None:
            X_min = X_std.min(axis=0)
            X_max = X_std.max(axis=0)
            X_range = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))
        
        X_unit = (X_std - X_min) / X_range
        X_unit = np.clip(X_unit, 0.0, 1.0).astype(np.float32)
        
        return X_unit, X_min, X_range

    # SS: It is used to mask the data in the box for the perturbation sampling.
    def _mask_in_box(self) -> np.ndarray:
        X_eval_unit, _, _, _ = self._active_data()
        
        # FIX: Ensure we have valid data and box bounds
        if X_eval_unit is None or X_eval_unit.shape[0] == 0:
            logger.warning(f"SingleAgent: X_eval_unit is None or empty in _mask_in_box (eval_on_test_data={self.eval_on_test_data})")
            return np.zeros(0, dtype=bool)
        
        if self.lower is None or self.upper is None:
            logger.warning(f"SingleAgent: Box bounds are None in _mask_in_box")
            return np.zeros(X_eval_unit.shape[0], dtype=bool)
        
        # FIX: Verify that box bounds are in normalized space [0, 1]
        # This ensures the mask computation uses the same normalized space as the data
        lower = self.lower
        upper = self.upper
        
        # Debug: Log if box bounds are outside [0, 1] range (indicates normalization issue)
        if np.any(lower < 0) or np.any(lower > 1) or np.any(upper < 0) or np.any(upper > 1):
            logger.warning(
                f"SingleAgent: Box bounds outside [0, 1] range in _mask_in_box. "
                f"lower range: [{lower.min():.3f}, {lower.max():.3f}], "
                f"upper range: [{upper.min():.3f}, {upper.max():.3f}]. "
                f"This may indicate a normalization mismatch."
            )
        
        # Ensure bounds have correct shape
        if lower.shape[0] != self.n_features or upper.shape[0] != self.n_features:
            logger.error(
                f"SingleAgent: Box bounds shape mismatch in _mask_in_box. "
                f"lower.shape={lower.shape}, upper.shape={upper.shape}, n_features={self.n_features}"
            )
            return np.zeros(X_eval_unit.shape[0], dtype=bool)
        
        conds = []
        for j in range(self.n_features):
            conds.append((X_eval_unit[:, j] >= lower[j]) & (X_eval_unit[:, j] <= upper[j]))
        mask = np.logical_and.reduce(conds) if conds else np.ones(X_eval_unit.shape[0], dtype=bool)
        return mask

    def _get_cached_probs(self, use_test: bool = False, split: Optional[str] = None) -> np.ndarray:
        if split is None:
            split = "test" if use_test else "train"
        if split == "training":
            split = "train"
        key = split
        if self._cached_probs.get(key) is None:
            if key == "test":
                X = self.X_test_std
            elif key == "val":
                X = self.X_val_std
            else:
                X = self.X_std
            if X is None:
                raise ValueError(f"No {key} data available for cached predictions")
            if hasattr(self.classifier, 'eval'):
                self.classifier.eval()
            if hasattr(self.classifier, 'model') and hasattr(self.classifier.model, 'eval'):
                self.classifier.model.eval()
            with torch.no_grad():
                inputs = torch.from_numpy(X.astype(np.float32)).to(self.device)
                self._cached_probs[key] = predict_proba_torch(self.classifier, inputs).cpu().numpy()
            self.n_blackbox_queries += int(X.shape[0])
        return self._cached_probs[key]

    def _compute_effective_precision_target(self) -> float:
        """
        Class-aware precision target: min(precision_target, max(prior, 0.9 x the
        classifier's own precision for this class on training data)). The classifier's
        per-class precision is an upper bound on what any anchor box can achieve, so
        an absolute target above it makes the precision gate and the termination
        conditions unreachable (the Slush/Snow failure mode).
        """
        if not self.use_class_aware_targets:
            return float(self.precision_target)
        try:
            probs = self._get_cached_probs(use_test=False)
        except Exception as e:
            logger.warning(f"SingleAgent: could not compute class-aware precision target ({e}); "
                           f"falling back to absolute target {self.precision_target}")
            return float(self.precision_target)
        preds = probs.argmax(axis=1)
        prior = float((self.y == self.target_class).mean())
        pred_mask = (preds == self.target_class)
        if pred_mask.any():
            clf_class_precision = float((self.y[pred_mask] == self.target_class).mean())
        else:
            clf_class_precision = prior
        ceiling = max(prior, 0.9 * clf_class_precision)
        effective = float(np.clip(min(self.precision_target, ceiling), 0.05, self.precision_target))
        if effective < self.precision_target:
            logger.info(f"SingleAgent: class {self.target_class} effective precision target "
                        f"{effective:.4f} (absolute={self.precision_target}, prior={prior:.4f}, "
                        f"classifier class precision={clf_class_precision:.4f})")
        return effective

    def _potential(self, precision: float, coverage: float) -> float:
        """
        State potential for reward shaping: Phi(s) = alpha * precision +
        beta * sqrt(coverage) * gate(precision). sqrt amplifies the early-coverage
        signal; the gate scales coverage credit by precision quality relative to the
        class-aware threshold, preserving the precision-first curriculum without any
        non-potential bonus terms. Phi depends only on state, so reward shaping with
        Phi(s') - Phi(s) leaves the optimal policy unchanged (Ng et al., 1999).
        """
        precision = max(0.0, float(precision))
        coverage = max(0.0, float(coverage))
        # B2 rebalance: gate coverage credit on precision RELATIVE TO THE TARGET.
        #
        # The old gate reached 1.0 at 0.8*tau_P — i.e. full coverage credit while
        # still below target — which was harmless only because measured precision
        # was pinned at 1.000 in all 10 dataset x class probes. With beta raised so
        # coverage actually drives the policy, that leniency becomes a licence to
        # trade fidelity for coverage. The ramp gives zero credit below
        # (target - gate_margin) and full credit at the target.
        target = max(float(self.precision_target_effective), 1e-6)
        margin = max(float(self.gate_margin), 1e-6)
        gate = (precision - (target - margin)) / margin
        gate = float(min(1.0, max(0.0, gate)))

        # NOTE: no dimension-release / sparsity term here. It was tried and removed:
        # Phi is potential-based shaping, and shaping with Phi(s') - Phi(s) provably
        # leaves the optimal policy unchanged (Ng et al., 1999). Any term added to Phi
        # telescopes over an episode and cancels, which is why adding one produced
        # bit-identical coverage across a 4-dataset x 10-class probe. Incentives that
        # are meant to change behaviour must live in the NON-potential part of the
        # reward -- see the partial terminal credit in step().
        return float(self.alpha * precision + self.beta * np.sqrt(coverage) * gate)

    def _unit_to_std(self, X_unit_samples: np.ndarray) -> np.ndarray:
        if self.X_min is None or self.X_range is None:
            raise ValueError("X_min/X_range must be set for uniform perturbation sampling.")
        return (X_unit_samples * self.X_range) + self.X_min

    def _std_to_orig(self, X_std_samples: np.ndarray) -> np.ndarray:
        if self.scaler_mean is None or self.scaler_scale is None:
            return X_std_samples
        return X_std_samples * self.scaler_scale + self.scaler_mean

    def _unit_to_orig(self, X_unit_samples: np.ndarray) -> np.ndarray:
        return self._std_to_orig(self._unit_to_std(X_unit_samples))

    def _active_split(self) -> str:
        if getattr(self, "mode", "training") == "training":
            return "train"
        split = getattr(self, "eval_split", None)
        if split in ("train", "val", "test"):
            return split
        return "test" if self.eval_on_test_data else "train"

    def _active_data(self):
        split = self._active_split()
        if split == "test":
            return self.X_test_unit, self.X_test_std, self.y_test, "test"
        if split == "val":
            if self.X_val_unit is None:
                return self.X_test_unit, self.X_test_std, self.y_test, "test"
            return self.X_val_unit, self.X_val_std, self.y_val, "val"
        return self.X_unit, self.X_std, self.y, "training"

    def _init_n_neighbors(self, n_class: int) -> int:
        """Size the start hull so class-cond C starts below τ_C.

        k is min(frac·n_class, init_max_neighbors, n_class, cap_tau), then
        floored at init_min_neighbors unless that would exceed cap_tau.
        cap_tau = max(1, int(init_coverage_frac_of_target · τ_C · n_class)),
        so wine n≈35, τ_C=0.2, frac=0.5 → k=3 (C≈0.086), not min_neighbors=5.
        """
        n_class = int(max(1, n_class))
        cap_tau = max(
            1,
            int(self.init_coverage_frac_of_target * float(self.coverage_target) * n_class),
        )
        n = max(1, int(self.init_neighbor_frac * n_class))
        n = max(n, int(self.init_min_neighbors))
        n = min(n, int(self.init_max_neighbors), n_class, cap_tau)
        return int(max(1, n))

    @staticmethod
    def _class_conditional_coverage(
        mask: np.ndarray, y_data: np.ndarray, target_class: int
    ) -> Tuple[float, float, int, int]:
        """Return (class_cond_C, marginal_C, n_class_in_box, n_class)."""
        coverage_marginal = float(mask.mean()) if len(mask) else 0.0
        class_mask = y_data == target_class
        n_class = int(class_mask.sum())
        if n_class == 0 or len(mask) != len(class_mask):
            return 0.0, coverage_marginal, 0, n_class
        n_in = int((mask & class_mask).sum())
        return float(n_in / n_class), coverage_marginal, n_in, n_class

    def _coverage_improved(self, coverage: float) -> bool:
        if not self.require_coverage_gain_to_terminate:
            return True
        c_reset = float(self._coverage_at_reset)
        # Already at τ_C at reset: extra gain is not required to terminate.
        if c_reset >= float(self.coverage_target):
            return True
        return float(coverage) > c_reset + float(self._coverage_gain_eps)

    def _snap_to_nearest_class_point(self, point: np.ndarray, class_data: np.ndarray) -> np.ndarray:
        """Snap a mean/cluster centroid onto the nearest real class row.

        Threshold 0.0 means any positive distance snaps. A 0.5 cutoff left
        high-d means sitting in empty space (distance ~0.3) unsnapped.
        """
        point = np.asarray(point, dtype=np.float32).reshape(-1)
        if class_data is None or len(class_data) == 0:
            return point
        distances = np.linalg.norm(class_data - point, axis=1)
        nearest = class_data[int(np.argmin(distances))]
        if float(distances.min()) > self.centroid_snap_threshold:
            return np.asarray(nearest, dtype=np.float32)
        return point.astype(np.float32)

    def _include_point_in_box(self, point: np.ndarray) -> None:
        point = np.asarray(point, dtype=np.float32).reshape(-1)
        self.lower = np.minimum(self.lower, point).astype(np.float32)
        self.upper = np.maximum(self.upper, point).astype(np.float32)
        self.lower = np.clip(self.lower, 0.0, 1.0).astype(np.float32)
        self.upper = np.clip(self.upper, 0.0, 1.0).astype(np.float32)

    def _window_box_around(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = max(self.initial_window, self.min_width)
        point = np.asarray(point, dtype=np.float32).reshape(-1)
        lower = np.clip(point - w, 0.0, 1.0).astype(np.float32)
        upper = np.clip(point + w, 0.0, 1.0).astype(np.float32)
        return lower, upper

    def _freeze_categorical_bounds(self) -> None:
        if not self.categorical_indices or self.categorical_freeze == "none":
            return
        X_data, _, y_data, _ = self._active_data()
        for j in self.categorical_indices:
            if self.x_star_unit is not None and self.categorical_freeze == "instance":
                v = float(np.asarray(self.x_star_unit).reshape(-1)[j])
            else:
                class_rows = X_data[y_data == self.target_class]
                if class_rows.shape[0] == 0:
                    continue
                vals, counts = np.unique(np.round(class_rows[:, j], 6), return_counts=True)
                v = float(vals[int(np.argmax(counts))])
            self.lower[j] = np.clip(v - 1e-4, 0.0, 1.0)
            self.upper[j] = np.clip(v + 1e-4, 0.0, 1.0)
   
    
    def _get_class_centroid(self) -> Optional[np.ndarray]:
        """
        Get the centroid for the target class.

        Priority:
        0. Use class_init_point if set (explicit diversified start from inference)
        1. Use precomputed cluster_centroids_per_class if available
        2. Use fixed_instances_per_class if available (sample from them)
        3. Compute mean centroid from class data

        Returns:
            Centroid in unit space [0, 1], or None if no data available
        """
        # Priority 0: Explicit start point set by inference for diversified
        # class-based rollouts. Inference snaps these to class data when needed,
        # so use the point as-is.
        if self.class_init_point is not None:
            return np.array(self.class_init_point, dtype=np.float32)

        # Priority 1: Use precomputed cluster centroids
        if self.cluster_centroids_per_class is not None:
            if self.target_class in self.cluster_centroids_per_class:
                centroids = self.cluster_centroids_per_class[self.target_class]
                if len(centroids) > 0:
                    # Sample a random centroid if multiple available
                    centroid_idx = self.rng.integers(0, len(centroids))
                    centroid = np.array(centroids[centroid_idx], dtype=np.float32)
                    
                    # CRITICAL FIX: For scattered data, centroids might be mean centroids (not actual data points)
                    # Check if centroid is close to any actual data point. If not, use the nearest data point instead.
                    # CRITICAL: During training, always use training data to prevent test set leakage
                    if self.mode == "training":
                        X_data = self.X_unit
                        y_data = self.y
                    elif self.eval_on_test_data:
                        X_data = self.X_test_unit
                        y_data = self.y_test
                    else:
                        X_data = self.X_unit
                        y_data = self.y
                    class_mask = (y_data == self.target_class)
                    
                    if class_mask.sum() > 0:
                        return self._snap_to_nearest_class_point(centroid, X_data[class_mask])
                    return centroid
        
        # Priority 2: Use fixed instances (sample one as centroid)
        if self.fixed_instances_per_class is not None:
            if self.target_class in self.fixed_instances_per_class:
                instances = self.fixed_instances_per_class[self.target_class]
                if len(instances) > 0:
                    instance_idx = self.rng.integers(0, len(instances))
                    return np.array(instances[instance_idx], dtype=np.float32)
        
        # Priority 3: Compute mean centroid from class data
        X_data, _, y_data, _ = self._active_data()
        
        class_mask = (y_data == self.target_class)
        if class_mask.sum() == 0:
            logger.warning(f"No instances found for class {self.target_class} to compute centroid")
            return None
        
        class_data = X_data[class_mask]
        mean_centroid = np.mean(class_data, axis=0).astype(np.float32)
        return self._snap_to_nearest_class_point(mean_centroid, class_data)
    
    def _compute_box_from_centroid(self, centroid: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute box bounds from a centroid that are guaranteed to cover at least some points.
        
        This finds points in the class data that are closest to the centroid and computes
        box bounds (min/max) that cover those points, ensuring the box has non-zero coverage.
        
        Args:
            centroid: Centroid point in unit space [0, 1]
            
        Returns:
            Tuple of (lower, upper) bounds, or None if no data available
        """
        # Get class data
        X_data, _, y_data, _ = self._active_data()
        
        class_mask = (y_data == self.target_class)
        if class_mask.sum() == 0:
            return None
        
        class_data = X_data[class_mask]
        
        n_neighbors = self._init_n_neighbors(len(class_data))
        
        # Compute distances to centroid
        distances = np.linalg.norm(class_data - centroid, axis=1)
        nearest_indices = np.argsort(distances)[:n_neighbors]
        nearest_points = class_data[nearest_indices]
        
        # Compute box bounds from nearest points (with some padding)
        # Use min/max of nearest points, then add a small padding (5% of feature range)
        padding = 0.05
        lower = np.maximum(0.0, nearest_points.min(axis=0) - padding)
        upper = np.minimum(1.0, nearest_points.max(axis=0) + padding)
        
        # Ensure minimum width
        widths = upper - lower
        min_width_mask = widths < self.min_width
        if min_width_mask.any():
            # For features with width < min_width, center the box on the centroid
            for f in np.where(min_width_mask)[0]:
                half_width = self.min_width / 2.0
                lower[f] = np.clip(centroid[f] - half_width, 0.0, 1.0 - self.min_width)
                # CRITICAL FIX: Ensure upper >= lower + min_width after clipping
                # If lower + min_width > 1.0, adjust lower instead to maintain min_width
                if lower[f] + self.min_width > 1.0:
                    lower[f] = max(0.0, 1.0 - self.min_width)
                upper[f] = min(1.0, lower[f] + self.min_width)
                # Final validation: ensure width >= min_width
                if upper[f] - lower[f] < self.min_width:
                    upper[f] = min(1.0, lower[f] + self.min_width)
        
        return lower.astype(np.float32), upper.astype(np.float32)

    def _current_metrics(self) -> tuple:
        mask = self._mask_in_box()
        covered = np.where(mask)[0]
        
        _, X_data_std, y_data, data_source = self._active_data()
        
        # Ensure mask and y_data have the same length (safety check)
        if len(mask) != len(y_data):
            logger.error(
                f"SingleAgent: Mask length ({len(mask)}) != y_data length ({len(y_data)}). "
                f"This indicates a data mismatch. eval_on_test_data={self.eval_on_test_data}"
            )
            # Try to fix: use the data source that matches the mask
            expected_len = (
                len(self.y_test) if self.eval_on_test_data and self.y_test is not None
                else len(self.y)
            )
            if len(mask) == expected_len:
                y_data = self.y_test if self.eval_on_test_data else self.y
                logger.warning(f"  Corrected y_data to match mask length")
        
        # Coverage is always class-conditional P(x in box | y = target_class),
        # including instance-based episodes. Marginal P(x in box) is logged only.
        # Precision stays instance prediction-matching when x_star is set.
        is_instance_based = self.x_star_unit is not None
        
        if self.mode == "inference" and not is_instance_based:
            logger.debug(
                f"SingleAgent: Instance-based mode not detected during inference (x_star_unit is None). "
                f"This may indicate x_star_unit was not set before reset() or was cleared during reset()."
            )
        
        # If instance-based and original prediction not stored, compute it now
        if is_instance_based and self.original_prediction is None:
            if isinstance(self.x_star_unit, np.ndarray):
                x_star = self.x_star_unit
            else:
                x_star = np.array(self.x_star_unit, dtype=np.float32)
            x_star_std = self._unit_to_std(x_star.reshape(1, -1))[0]
            
            if hasattr(self.classifier, 'eval'):
                self.classifier.eval()
            if hasattr(self.classifier, 'model') and hasattr(self.classifier.model, 'eval'):
                self.classifier.model.eval()
            
            with torch.no_grad():
                instance_tensor = torch.from_numpy(x_star_std.astype(np.float32)).unsqueeze(0).to(self.device)
                probs = predict_proba_torch(self.classifier, instance_tensor).cpu().numpy()[0]
                self.original_prediction = int(np.argmax(probs))
                logger.debug(f"SingleAgent: Computed original prediction {self.original_prediction} for instance-based anchor")
        
        coverage, coverage_marginal, n_class_in_box, n_class_samples = (
            self._class_conditional_coverage(mask, y_data, self.target_class)
        )
        if n_class_samples == 0:
            logger.warning(
                f"SingleAgent: No samples found for target class {self.target_class} in {data_source} data. "
                f"Total samples: {len(y_data)}, Classes present: {sorted(np.unique(y_data).tolist())}"
            )
        elif coverage == 0.0 and covered.size > 0:
            logger.debug(
                f"SingleAgent: Box covers {int(mask.sum())} total samples but 0 target class samples. "
                f"Target class: {self.target_class}, Class samples in dataset: {n_class_samples}"
            )
        
        if covered.size == 0 and not (self.use_perturbation and self.perturbation_mode in ["uniform", "adaptive"]):
            return 0.0, coverage, {
                "hard_precision": 0.0, 
                "avg_prob": 0.0, 
                "n_points": 0, 
                "sampler": "none",
                "data_source": data_source,
                "coverage_marginal": float(coverage_marginal),
                "n_class_in_box": int(n_class_in_box),
                "n_class_samples": int(n_class_samples),
            }

        # Row indices into the active dataset when evaluation uses real rows
        # (empirical/bootstrap paths). Lets us look up cached classifier
        # probabilities instead of re-running the classifier every step.
        eval_row_idx = None

        if not self.use_perturbation:
            X_eval = X_data_std[covered]
            y_eval = y_data[covered]
            eval_row_idx = covered
            n_points = int(X_eval.shape[0])
            sampler_note = f"empirical_{data_source}"
        else:
            if self.perturbation_mode == "bootstrap":
                if covered.size == 0:
                    data_source = "test" if self.eval_on_test_data else "training"
                    return 0.0, coverage, {
                        "hard_precision": 0.0, 
                        "avg_prob": 0.0, 
                        "n_points": 0, 
                        "sampler": "none",
                        "data_source": data_source,
                        "coverage_marginal": float(coverage_marginal),
                        "n_class_in_box": int(n_class_in_box),
                        "n_class_samples": int(n_class_samples),
                    }
                n_samp = min(self.n_perturb, max(1, covered.size))
                idx = self.rng.choice(covered, size=n_samp, replace=True)
                X_eval = X_data_std[idx]
                y_eval = y_data[idx]
                eval_row_idx = idx
                n_points = int(n_samp)
                sampler_note = f"bootstrap_{data_source}"
            elif self.perturbation_mode == "uniform":
                n_samp = self.n_perturb
                U = np.zeros((n_samp, self.n_features), dtype=np.float32)
                for j in range(self.n_features):
                    low, up = float(self.lower[j]), float(self.upper[j])
                    width = max(up - low, self.min_width)
                    mid = 0.5 * (low + up)
                    low = max(0.0, mid - width / 2.0)
                    up = min(1.0, mid + width / 2.0)
                    U[:, j] = self.rng.uniform(low=low, high=up, size=n_samp).astype(np.float32)
                
                # CRITICAL FIX: For instance-based anchors, always include the original instance
                # This ensures precision calculation includes at least one point (the instance itself)
                # that matches the original prediction, preventing precision from being incorrectly 0.0
                if is_instance_based and self.x_star_unit is not None:
                    x_star_unit = np.array(self.x_star_unit, dtype=np.float32).reshape(1, -1)
                    # Prepend the original instance to the uniform samples
                    U = np.vstack([x_star_unit, U])
                    n_samp = n_samp + 1
                
                X_eval = self._unit_to_std(U)
                y_eval = None
                n_points = int(n_samp)
                sampler_note = f"uniform_{data_source}"
            elif self.perturbation_mode == "adaptive":
                # Adaptive mode: use bootstrap when enough covered points, otherwise use uniform
                min_points_for_bootstrap = max(1, int(0.1 * self.n_perturb))
                
                if covered.size >= min_points_for_bootstrap:
                    # Use bootstrap sampling from real data points
                    n_samp = min(self.n_perturb, covered.size)
                    idx = self.rng.choice(covered, size=n_samp, replace=True)
                    X_eval = X_data_std[idx]
                    y_eval = y_data[idx]
                    eval_row_idx = idx
                    n_points = int(n_samp)
                    sampler_note = f"adaptive_bootstrap_{data_source}"
                else:
                    # Fall back to uniform sampling when not enough covered points
                    n_samp = self.n_perturb
                    U = np.zeros((n_samp, self.n_features), dtype=np.float32)
                    for j in range(self.n_features):
                        low, up = float(self.lower[j]), float(self.upper[j])
                        width = max(up - low, self.min_width)
                        mid = 0.5 * (low + up)
                        low = max(0.0, mid - width / 2.0)
                        up = min(1.0, mid + width / 2.0)
                        U[:, j] = self.rng.uniform(low=low, high=up, size=n_samp).astype(np.float32)
                    
                    # CRITICAL FIX: For instance-based anchors, always include the original instance
                    # This ensures precision calculation includes at least one point (the instance itself)
                    # that matches the original prediction, preventing precision from being incorrectly 0.0
                    if is_instance_based and self.x_star_unit is not None:
                        x_star_unit = np.array(self.x_star_unit, dtype=np.float32).reshape(1, -1)
                        # Prepend the original instance to the uniform samples
                        U = np.vstack([x_star_unit, U])
                        n_samp = n_samp + 1
                    
                    X_eval = self._unit_to_std(U)
                    y_eval = None
                    n_points = int(n_samp)
                    sampler_note = f"adaptive_uniform_{data_source}"
            else:
                raise ValueError(f"Unknown perturbation_mode '{self.perturbation_mode}'. Use 'bootstrap', 'uniform', or 'adaptive'.")

        if eval_row_idx is not None:
            # Dataset rows: look up cached probabilities (computed once per env)
            # instead of running the classifier — this is the hot path in adaptive/
            # bootstrap mode and removes the dominant per-step cost.
            probs = self._get_cached_probs(split=self._active_split())[eval_row_idx]
        else:
            # Fresh uniform samples: must run the classifier
            if hasattr(self.classifier, 'eval'):
                self.classifier.eval()
            if hasattr(self.classifier, 'model') and hasattr(self.classifier.model, 'eval'):
                self.classifier.model.eval()

            with torch.no_grad():
                inputs = torch.from_numpy(X_eval).float().to(self.device)
                probs = predict_proba_torch(self.classifier, inputs).cpu().numpy()
            self.n_blackbox_queries += int(X_eval.shape[0])

        preds = probs.argmax(axis=1)
        positive_idx = (preds == self.target_class)
        
        # Precision calculation depends on mode:
        # - Instance-based: P(prediction matches original instance | anchor conditions hold) - matches original Anchor paper
        # - Class-based: P(y = target_class | x in box) - measures class correctness
        if is_instance_based:
            # Instance-based mode: Match original Anchor paper definition
            # Precision = fraction of samples where prediction matches original instance's prediction
            if self.original_prediction is not None:
                matches_original = (preds == self.original_prediction)
                hard_precision = float(matches_original.mean())
                logger.debug(f"SingleAgent: Instance-based precision (prediction matching): {hard_precision:.4f} "
                           f"(original prediction: {self.original_prediction}, target class: {self.target_class})")
            else:
                # Fallback: if original prediction not stored, use class-based precision
                logger.warning(f"SingleAgent: Original prediction not found for instance-based anchor, "
                             f"falling back to class-based precision calculation")
                if y_eval is None:
                    hard_precision = float(positive_idx.mean())
                else:
                    hard_precision = float((y_eval == self.target_class).mean())
            purity = float((y_eval == (self.original_prediction if self.original_prediction is not None else self.target_class)).mean()) if y_eval is not None else float("nan")
        else:
            # C-09: class-based PRIMARY is model fidelity P(f_hat = c | x in B).
            hard_precision = float(positive_idx.mean())
            purity = float((y_eval == self.target_class).mean()) if y_eval is not None else float("nan")

        # For avg_prob blending, use original_prediction for instance-based, target_class for class-based
        if is_instance_based and self.original_prediction is not None:
            # Instance-based: blend with probability of original_prediction class
            avg_prob = float(probs[:, self.original_prediction].mean())
        else:
            # Class-based: blend with probability of target_class
            avg_prob = float(probs[:, self.target_class].mean())
        
        precision_proxy = (
            self.precision_blend_lambda * hard_precision + (1.0 - self.precision_blend_lambda) * avg_prob
        )
        target_class_fraction = hard_precision  # Same as hard_precision when y_eval is available

        # Control signal is hard Fid. Softmax blend made τ_P unreachable on
        # low-confidence but accurate DNNs (wine p̂_max≈0.37 → proxy 0.68).
        return hard_precision, coverage, {
            "hard_precision": hard_precision,
            "precision_proxy": precision_proxy,
            "purity": purity,
            "avg_prob": avg_prob,
            "n_points": int(n_points),
            "n_covered": int(covered.size),
            "sampler": sampler_note,
            "target_class_fraction": target_class_fraction,
            "data_source": data_source,
            "cov_real": float(coverage),
            "coverage_marginal": float(coverage_marginal),
            "n_class_in_box": int(n_class_in_box),
            "n_class_samples": int(n_class_samples),
        }

    def _reset_termination_counters(self):
        """Reset termination counters. Only both_targets_met is enabled by default."""
        self.termination_reason_counts = {
            "both_targets_met": 0,
            "excellent_precision": 0,
            "high_precision_reasonable_coverage": 0,
            "both_reasonably_close": 0
        }
        self.termination_reason_enabled = {
            "both_targets_met": True,
            "excellent_precision": True,
            "high_precision_reasonable_coverage": True,
            "both_reasonably_close": True
        }
        # Lenient reasons (excellent_precision, high_prec, reasonably_close)
        # paid the +5 bonus on class hulls at step 2 without ever hitting τ_C.
        # strict_target_termination (default) keeps only both_targets_met in
        # training and inference.
        if self.strict_target_termination or self.mode == "inference":
            self.termination_reason_enabled["excellent_precision"] = False
            self.termination_reason_enabled["high_precision_reasonable_coverage"] = False
        if self.strict_target_termination:
            self.termination_reason_enabled["both_reasonably_close"] = False
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to an initial state.
        
        Returns:
            observation: Initial observation (numpy array)
            info: Dictionary with additional information
        """
        if hasattr(self.classifier, 'eval'):
            self.classifier.eval()
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Note: Termination counters are NOT reset here to allow them to accumulate
        # across episodes. They are initialized in __init__ and accumulate throughout
        # the session. In inference/evaluation mode, counters start fresh from __init__,
        # but accumulate across episodes to track usage and disable overused reasons.
        
        self.timestep = 0
        # B2-DIAG: zero the per-episode reward-term accumulators
        self._rt_shaping = 0.0
        self._rt_overlap = 0.0
        self._rt_drift = 0.0
        self._rt_anchor_drift = 0.0
        self._rt_cov_floor = 0.0
        self._rt_terminal = 0.0
        self._rt_total = 0.0
        self._rt_drift_raw = 0.0
        # ACT-DIAG accumulators
        self._act_requested = 0.0
        self._act_applied = 0.0
        self._act_absmean = 0.0
        self._act_maxdelta = 0.0
        self._act_steps = 0
        self._act_proj_loss = 0.0
        self._act_clipped_steps = 0
        self._best_box = None
        self._best_box_cov = -1.0
        
        # CRITICAL: Preserve x_star_unit during inference/evaluation mode
        # During inference, x_star_unit is set externally (e.g., by inference code) to indicate instance-based mode
        # We must preserve it so precision stays instance prediction-matching.
        x_star_unit_preserved = None
        if self.mode in ["inference", "evaluation"] and self.x_star_unit is not None:
            x_star_unit_preserved = self.x_star_unit.copy() if isinstance(self.x_star_unit, np.ndarray) else self.x_star_unit
        
        # Mixed initialization during training: randomly choose between instance-based and centroid-based
        use_instance_based = False
        if self.mode == "training" and self.training_instances_per_class is not None:
            # Get class-specific ratio if available, otherwise use base ratio
            if (self.training_instance_ratios_per_class is not None and 
                self.target_class in self.training_instance_ratios_per_class):
                class_ratio = self.training_instance_ratios_per_class[self.target_class]
            else:
                class_ratio = self.training_instance_ratio
            
            # Randomly decide whether to use instance-based or centroid-based
            use_instance_based = class_ratio > 0.0 and self.rng.random() < class_ratio
            
            if use_instance_based and self.target_class in self.training_instances_per_class:
                # Instance-based: randomly select an instance from training instances
                instances = self.training_instances_per_class[self.target_class]
                if len(instances) > 0:
                    instance_idx = self.rng.integers(0, len(instances))
                    instance = np.array(instances[instance_idx], dtype=np.float32)
                    # CRITICAL: Reset original_prediction when selecting a new instance
                    # This ensures we recompute it for the new instance, not reuse from previous episode
                    self.original_prediction = None
                    self.x_star_unit = instance
                    ratio_info = f" (ratio: {class_ratio:.1%})" if 'class_ratio' in locals() else ""
                    logger.debug(f"Training: Using instance-based initialization (instance {instance_idx}/{len(instances)}{ratio_info})")
                else:
                    use_instance_based = False  # Fall back to centroid-based if no instances
            else:
                # Centroid-based: clear x_star_unit
                self.x_star_unit = None
                # Also clear original prediction when switching to class-based
                self.original_prediction = None
        
        # CRITICAL: Restore x_star_unit during inference/evaluation mode if it was set externally
        # This ensures instance-based mode is preserved and correct coverage (overall) is used for termination
        if self.mode in ["inference", "evaluation"] and x_star_unit_preserved is not None:
            self.x_star_unit = x_star_unit_preserved.copy() if isinstance(x_star_unit_preserved, np.ndarray) else x_star_unit_preserved
            logger.debug(f"SingleAgent: Preserved x_star_unit for instance-based mode during {self.mode}")
        
        # Priority 1: If x_star_unit is explicitly set (for instance-based), use it
        if self.x_star_unit is not None:
            if isinstance(self.x_star_unit, np.ndarray):
                centroid = self.x_star_unit
            else:
                centroid = np.array(self.x_star_unit, dtype=np.float32)
            box_bounds = self._compute_box_from_centroid(centroid)
            if box_bounds is not None:
                self.lower, self.upper = box_bounds
                self._include_point_in_box(centroid)
            else:
                self.lower, self.upper = self._window_box_around(centroid)
            
            # Compute and store original prediction for instance-based anchors (matches original Anchor paper)
            # This is used for precision calculation: P(prediction matches original | anchor conditions hold)
            if self.original_prediction is None:
                # Convert instance from unit space to standardized space for prediction
                x_star_std = self._unit_to_std(centroid.reshape(1, -1))[0]
                
                # Get prediction for original instance
                if hasattr(self.classifier, 'eval'):
                    self.classifier.eval()
                if hasattr(self.classifier, 'model') and hasattr(self.classifier.model, 'eval'):
                    self.classifier.model.eval()
                
                with torch.no_grad():
                    instance_tensor = torch.from_numpy(x_star_std.astype(np.float32)).unsqueeze(0).to(self.device)
                    probs = predict_proba_torch(self.classifier, instance_tensor).cpu().numpy()[0]
                    self.original_prediction = int(np.argmax(probs))
                    # CRITICAL VALIDATION: Verify original_prediction matches target_class
                    # This should be true after filtering instances by prediction, but check as safety
                    if self.original_prediction != self.target_class:
                        logger.warning(
                            f"SingleAgent: original_prediction ({self.original_prediction}) != target_class ({self.target_class})! "
                            f"This may cause precision calculation issues. "
                            f"Instance should have been filtered during training instance selection."
                        )
                    logger.debug(f"SingleAgent: Stored original prediction {self.original_prediction} for instance-based anchor (target_class={self.target_class})")
        # Priority 2: Use class centroid if enabled (for class-based)
        elif self.use_class_centroids:
            centroid = self._get_class_centroid()
            if centroid is not None:
                # Compute box bounds that cover points near the centroid
                # This ensures the box covers at least some points from the cluster
                box_bounds = self._compute_box_from_centroid(centroid)
                if box_bounds is not None:
                    self.lower, self.upper = box_bounds
                    self._include_point_in_box(centroid)
                else:
                    # Fallback: Use fixed window around centroid
                    self.lower, self.upper = self._window_box_around(centroid)
            else:
                # Fallback: Full space initialization
                self.lower = np.zeros(self.n_features, dtype=np.float32)
                self.upper = np.ones(self.n_features, dtype=np.float32)
        # Priority 3: Full space initialization (original behavior)
        else:
            self.lower = np.zeros(self.n_features, dtype=np.float32)
            self.upper = np.ones(self.n_features, dtype=np.float32)

        # Pin categoricals before the first metric computation (not after step 1).
        self._freeze_categorical_bounds()
        
        self.prev_lower = self.lower.copy()
        self.prev_upper = self.upper.copy()
        self.box_history = [(self.lower.copy(), self.upper.copy())]
        self.coverage_floor_hits = 0

        precision, coverage, initial_details = self._current_metrics()
        # Seed the prev-metrics cache for the first step of the episode
        self._last_step_metrics = (precision, coverage, initial_details)
        n_class = int(initial_details.get("n_class_samples") or 0)
        if n_class <= 0:
            n_class = int((self._active_data()[2] == self.target_class).sum())
        self._coverage_at_reset = float(coverage)
        self._coverage_gain_eps = 1.0 / max(n_class, 1)
        
        # Prevent immediate termination: require at least 2 steps before allowing termination
        # This prevents episodes from terminating on the first step due to initial box meeting targets
        self.step_count = 0
        # Read from YAML so RLDA/MADA use the same Markov termination rule.
        
        observation = np.concatenate([
            self.lower,
            self.upper,
            np.array([precision, coverage, 0.0], dtype=np.float32),
        ])
        
        info = {
            "initial_precision": float(precision),
            "initial_coverage": float(coverage),
            "coverage_at_reset": float(self._coverage_at_reset),
            "coverage_gain_eps": float(self._coverage_gain_eps),
        }
        
        return observation, info

    def _apply_action(self, action: int):
        f = action // (len(self.directions) * len(self.step_fracs))
        rem = action % (len(self.directions) * len(self.step_fracs))
        d = rem // len(self.step_fracs)
        m = rem % len(self.step_fracs)

        direction = self.directions[d]
        step = float(self.step_fracs[m])
        cur_width = max(1e-6, self.upper[f] - self.lower[f])
        rel_step = step * cur_width

        if direction == "shrink_lower":
            self.lower[f] = min(self.lower[f] + rel_step, self.upper[f] - self.min_width)
        elif direction == "expand_lower":
            self.lower[f] = max(self.lower[f] - rel_step, 0.0)
        elif direction == "shrink_upper":
            self.upper[f] = max(self.upper[f] - rel_step, self.lower[f] + self.min_width)
        elif direction == "expand_upper":
            self.upper[f] = min(self.upper[f] + rel_step, 1.0)

        if self.upper[f] - self.lower[f] < self.min_width:
            mid = 0.5 * (self.upper[f] + self.lower[f])
            self.lower[f] = max(0.0, mid - self.min_width / 2.0)
            self.upper[f] = min(1.0, mid + self.min_width / 2.0)

    def _apply_continuous_action(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        
        lower_deltas = action[:self.n_features]
        upper_deltas = action[self.n_features:]
        
        widths = np.maximum(self.upper - self.lower, 1e-6)
        max_delta_proportional = self.max_action_scale * widths
        max_delta = np.maximum(max_delta_proportional, self.min_absolute_step)
        
        # Store before state for debugging
        lower_before = self.lower.copy()
        upper_before = self.upper.copy()
        
        lower_changes = lower_deltas * max_delta
        self.lower = np.clip(self.lower + lower_changes, 0.0, self.upper - self.min_width)
        
        upper_changes = upper_deltas * max_delta
        self.upper = np.clip(self.upper + upper_changes, self.lower + self.min_width, 1.0)
        
        # CRITICAL: For instance-based anchors, ensure box always covers x_star_unit FIRST
        # This must happen BEFORE min_width adjustment to prevent the box from moving away
        if self.x_star_unit is not None:
            if isinstance(self.x_star_unit, np.ndarray):
                x_star = self.x_star_unit
            else:
                x_star = np.array(self.x_star_unit, dtype=np.float32)
            
            # Ensure box covers x_star_unit in all dimensions
            for f in range(self.n_features):
                if x_star[f] < self.lower[f]:
                    # Anchor is below lower bound: expand lower to include it
                    self.lower[f] = max(0.0, x_star[f] - self.min_width / 2.0)
                    # Ensure upper is still above lower + min_width
                    if self.upper[f] < self.lower[f] + self.min_width:
                        self.upper[f] = min(1.0, self.lower[f] + self.min_width)
                elif x_star[f] > self.upper[f]:
                    # Anchor is above upper bound: expand upper to include it
                    self.upper[f] = min(1.0, x_star[f] + self.min_width / 2.0)
                    # Ensure lower is still below upper - min_width
                    if self.lower[f] > self.upper[f] - self.min_width:
                        self.lower[f] = max(0.0, self.upper[f] - self.min_width)
                # Ensure x_star_unit is within bounds (safety check)
                if x_star[f] < self.lower[f] or x_star[f] > self.upper[f]:
                    # If still outside, center box around x_star_unit
                    self.lower[f] = max(0.0, x_star[f] - self.min_width / 2.0)
                    self.upper[f] = min(1.0, self.lower[f] + self.min_width)
                    # If upper was clipped, adjust lower
                    if self.upper[f] - self.lower[f] < self.min_width:
                        self.upper[f] = min(1.0, x_star[f] + self.min_width / 2.0)
                        self.lower[f] = max(0.0, self.upper[f] - self.min_width)
        
        # Now ensure min_width constraint is satisfied (AFTER ensuring x_star_unit is covered)
        for f in range(self.n_features):
            if self.upper[f] - self.lower[f] < self.min_width:
                # If x_star_unit exists, try to center around it while maintaining min_width
                if self.x_star_unit is not None:
                    if isinstance(self.x_star_unit, np.ndarray):
                        x_star = self.x_star_unit
                    else:
                        x_star = np.array(self.x_star_unit, dtype=np.float32)
                    # Center around x_star_unit
                    mid = x_star[f]
                else:
                    # No x_star_unit, use box center
                    mid = 0.5 * (self.upper[f] + self.lower[f])
                
                self.lower[f] = max(0.0, mid - self.min_width / 2.0)
                self.upper[f] = min(1.0, mid + self.min_width / 2.0)
                # If upper was clipped, adjust lower
                if self.upper[f] - self.lower[f] < self.min_width:
                    if self.upper[f] >= 1.0:
                        self.lower[f] = max(0.0, 1.0 - self.min_width)
                        self.upper[f] = 1.0
                    elif self.lower[f] <= 0.0:
                        self.lower[f] = 0.0
                        self.upper[f] = min(1.0, self.min_width)
                
                # Final safety check: ensure x_star_unit is still in box
                if self.x_star_unit is not None:
                    if isinstance(self.x_star_unit, np.ndarray):
                        x_star = self.x_star_unit
                    else:
                        x_star = np.array(self.x_star_unit, dtype=np.float32)
                    if x_star[f] < self.lower[f]:
                        self.lower[f] = max(0.0, x_star[f] - self.min_width / 2.0)
                        self.upper[f] = min(1.0, self.lower[f] + self.min_width)
                    elif x_star[f] > self.upper[f]:
                        self.upper[f] = min(1.0, x_star[f] + self.min_width / 2.0)
                        self.lower[f] = max(0.0, self.upper[f] - self.min_width)
        
        # ACT-DIAG: how much of the requested move actually survives.
        #   requested = what the policy asked for (action * max_delta)
        #   applied   = what remains after clip to [0,1], min_width, x_star
        #               containment and categorical freeze
        # A large requested/applied gap means the ENVIRONMENT is blocking movement;
        # a small `requested` means the POLICY is choosing not to move. Several
        # iterations of reward work could not distinguish these two.
        _req = float(np.abs(lower_changes).sum() + np.abs(upper_changes).sum())
        _app = float(np.abs(self.lower - lower_before).sum() + np.abs(self.upper - upper_before).sum())
        self._act_requested = getattr(self, "_act_requested", 0.0) + _req
        self._act_applied = getattr(self, "_act_applied", 0.0) + _app
        self._act_absmean = getattr(self, "_act_absmean", 0.0) + float(np.abs(action).mean())
        self._act_maxdelta = getattr(self, "_act_maxdelta", 0.0) + float(np.mean(max_delta))
        self._act_steps = getattr(self, "_act_steps", 0) + 1

        # Debug: Log if action was applied (only for first call)
        if not hasattr(self, '_action_debug_logged'):
            self._action_debug_logged = False
        
        if not self._action_debug_logged:
            lower_diff = np.abs(self.lower - lower_before).max()
            upper_diff = np.abs(self.upper - upper_before).max()
            logger.debug(f"  _apply_continuous_action: lower_diff={lower_diff:.6f}, upper_diff={upper_diff:.6f}, max_delta={max_delta.max():.6f}, action_mean={action.mean():.4f}")
            if lower_diff < 1e-6 and upper_diff < 1e-6:
                logger.warning(f"  ⚠ Action did not change box! lower_deltas mean={lower_deltas.mean():.4f}, upper_deltas mean={upper_deltas.mean():.4f}, max_delta={max_delta.max():.6f}")
            self._action_debug_logged = True
        self._freeze_categorical_bounds()
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Run one timestep of the environment's dynamics.
        
        Args:
            action: Action to take (numpy array of shape (2 * n_features,))
        
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether the episode has terminated (targets met)
            truncated: Whether the episode was truncated (max steps reached)
            info: Dictionary with additional information
        """
        
        # Convert action to numpy array if needed
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = np.array(action, dtype=np.float32)
        
        # Ensure action is the correct shape
        if action.shape[0] != 2 * self.n_features:
            raise ValueError(f"Action shape {action.shape} does not match expected shape ({2 * self.n_features},)")
        
        # Prev metrics: reuse the metrics computed at the end of the previous
        # step()/reset(). This halves classifier work and, critically, makes prev
        # and current come from the same sample draw, so the shaping gain below
        # measures the action's effect rather than resampling noise.
        if self._last_step_metrics is not None:
            prev_precision, prev_coverage, _ = self._last_step_metrics
        else:
            prev_precision, prev_coverage, _ = self._current_metrics()
        prev_lower = self.lower.copy()
        prev_upper = self.upper.copy()
        
        # Apply continuous action (always continuous for single-agent)
        self._apply_continuous_action(action)
        
        # CRITICAL VALIDATION: Ensure box is valid (lower <= upper in all dimensions)
        # This prevents coverage from being 0.0 due to invalid boxes
        for f in range(self.n_features):
            if self.lower[f] > self.upper[f]:
                logger.warning(
                    f"  ⚠ Invalid box detected: lower[{f}]={self.lower[f]:.6f} > upper[{f}]={self.upper[f]:.6f}. "
                    f"Fixing by centering around x_star_unit if available."
                )
                # Fix invalid box
                if self.x_star_unit is not None:
                    if isinstance(self.x_star_unit, np.ndarray):
                        x_star = self.x_star_unit
                    else:
                        x_star = np.array(self.x_star_unit, dtype=np.float32)
                    # Center around x_star_unit
                    mid = x_star[f]
                else:
                    # No x_star_unit, use midpoint
                    mid = 0.5 * (self.lower[f] + self.upper[f])
                
                self.lower[f] = max(0.0, mid - self.min_width / 2.0)
                self.upper[f] = min(1.0, mid + self.min_width / 2.0)
                if self.upper[f] - self.lower[f] < self.min_width:
                    if self.upper[f] >= 1.0:
                        self.lower[f] = max(0.0, 1.0 - self.min_width)
                        self.upper[f] = 1.0
                    else:
                        self.lower[f] = 0.0
                        self.upper[f] = min(1.0, self.min_width)
        
        precision, coverage, details = self._current_metrics()

        if not np.isfinite(precision):
            precision = 0.0
        if not np.isfinite(coverage):
            coverage = 0.0
        if not np.isfinite(prev_precision):
            prev_precision = 0.0
        if not np.isfinite(prev_coverage):
            prev_coverage = 0.0

        coverage_clipped = False
        coverage_before_revert = None
        coverage_after_revert = None
        # coverage_floor_mode:
        #   "step"     - enforce the floor on every step (original behaviour)
        #   "terminal" - let the box move freely during the episode and enforce the
        #                floor only at the end (default)
        # Measured motivation: with step-wise enforcement the floor bound on 197 of
        # 200 steps and discarded 82% of all requested movement, so no reward change
        # could alter behaviour. An empty box mid-trajectory is recoverable; treating
        # it as forbidden made the environment, not the objective, decide the policy.
        if coverage < self.min_coverage_floor and self.coverage_floor_mode == "step":
            coverage_before_revert = float(coverage)
            _l, _u, _discarded = self._project_to_floor_per_dim(prev_lower, prev_upper)
            self.lower, self.upper = _l, _u
            self._act_proj_loss = getattr(self, "_act_proj_loss", 0.0) + _discarded
            self._act_clipped_steps = getattr(self, "_act_clipped_steps", 0) + 1
            precision, coverage, details = self._current_metrics()
            if not np.isfinite(precision):
                precision = 0.0
            if not np.isfinite(coverage):
                coverage = 0.0
            coverage_after_revert = float(coverage)
            self.coverage_floor_hits += 1
            coverage_clipped = True

        # Track the best feasible box seen this episode. With the floor enforced only
        # at the end, the trajectory is free to explore (and to collapse), so the
        # episode needs a memory of the best anchor it actually found -- otherwise a
        # collapsed final box has nothing to retreat to and the episode returns
        # nothing usable.
        if np.isfinite(coverage) and coverage >= self.min_coverage_floor:
            _score = float(coverage)
            if _score > getattr(self, "_best_box_cov", -1.0):
                self._best_box_cov = _score
                self._best_box = (self.lower.copy(), self.upper.copy())

        precision_gain = precision - prev_precision
        coverage_gain = coverage - prev_coverage

        if not np.isfinite(precision_gain):
            precision_gain = 0.0
        if not np.isfinite(coverage_gain):
            coverage_gain = 0.0

        # Potential-based shaping: reward = Phi(s') - Phi(s) (see _potential).
        # Telescoping makes any A->B->A oscillation net exactly zero. The previous
        # relative-gain normalization (gain / prev value) paid more on the way up
        # than it charged on the way back down, so wiggling a bound forever was the
        # return-maximizing policy. The clips, phase weights, and coverage/target-
        # class bonuses existed to manage that scheme and are retired with it.
        phi_prev = self._potential(prev_precision, prev_coverage)
        phi_curr = self._potential(precision, coverage)
        shaping_gain = phi_curr - phi_prev
        coverage_gain_for_reward = coverage_gain  # raw gain, kept for logging

        widths = self.upper - self.lower
        overlap_penalty = self.gamma * float((widths < (2 * self.min_width)).mean())

        drift = float(np.linalg.norm(self.upper - prev_upper) + np.linalg.norm(self.lower - prev_lower))
        drift_penalty = self.drift_penalty_weight * drift

        anchor_drift_penalty = self._compute_anchor_drift_penalty(prev_lower, prev_upper)

        # Single agent: no inter-class overlap penalty
        inter_class_overlap_penalty = 0.0

        # Retired: the JS proxy was a third movement penalty (consecutive-box volume
        # overlap, not a divergence); drift_penalty already covers box movement.
        js_penalty = 0.0
        # Retired with the relative-gain scheme; keys kept for logging compatibility.
        coverage_bonus = 0.0
        target_class_bonus = 0.0

        # When action is reverted (coverage_clipped), reduce penalties significantly
        coverage_floor_penalty = 0.0
        if coverage_clipped:
            penalty_reduction_factor = 0.1  # Reduce penalties by 90%
            overlap_penalty *= penalty_reduction_factor
            anchor_drift_penalty *= penalty_reduction_factor
            # B2 fix 1: do NOT charge for a reverted action.
            #
            # Measured on breast_cancer (d=30): the box was reverted on ~196 of 200
            # steps, so this -0.05 accounted for 81% of the entire penalty mass and
            # a median episode return of -9.9, while the shaping term that is meant
            # to teach the precision/coverage trade-off contributed -0.04. The
            # revert already removes any benefit of the action; charging for it as
            # well is double jeopardy for a move the environment refused to apply,
            # and it drowns out the only informative part of the reward.
            #
            # Set env_config["coverage_floor_penalty"] to a negative value to
            # restore the old behaviour for ablation.
            coverage_floor_penalty = self.coverage_floor_penalty_value

        reward = (shaping_gain -
                 overlap_penalty -
                 drift_penalty -
                 anchor_drift_penalty +
                 coverage_floor_penalty)

        if not np.isfinite(reward):
            reward = 0.0

        # B2-DIAG: per-episode accumulation of every reward term. The shaping term
        # telescopes (bounded by the range of Phi), while the penalties are charged
        # every step and accumulate with episode length and with sqrt(d) via the L2
        # drift norm. Logging the totals separately shows which term actually
        # dominates the return instead of inferring it from ep_rew_mean.
        self._rt_shaping = getattr(self, "_rt_shaping", 0.0) + float(shaping_gain)
        self._rt_overlap = getattr(self, "_rt_overlap", 0.0) + float(overlap_penalty)
        self._rt_drift = getattr(self, "_rt_drift", 0.0) + float(drift_penalty)
        self._rt_anchor_drift = getattr(self, "_rt_anchor_drift", 0.0) + float(anchor_drift_penalty)
        self._rt_cov_floor = getattr(self, "_rt_cov_floor", 0.0) + float(coverage_floor_penalty)
        self._rt_terminal = getattr(self, "_rt_terminal", 0.0)
        self._rt_total = getattr(self, "_rt_total", 0.0) + float(reward)
        self._rt_drift_raw = getattr(self, "_rt_drift_raw", 0.0) + float(drift)

        self.box_history.append((self.lower.copy(), self.upper.copy()))
        self.prev_lower = prev_lower
        self.prev_upper = prev_upper
        # Cache final metrics for the next step's prev (common random numbers)
        self._last_step_metrics = (precision, coverage, details)
        episode_phase = min(
            1.0, float(self.timestep + 1) / float(self.max_cycles)
        )
        state = np.concatenate([
            self.lower,
            self.upper,
            np.array([precision, coverage, episode_phase], dtype=np.float32),
        ])
        
        ## SS: Target change here:
        # Termination uses the class-aware effective target so minority/overlapping
        # classes have reachable conditions (see _compute_effective_precision_target).
        eps = 1e-12
        precision_target = self.precision_target_effective
        both_targets_met = (
            precision >= precision_target
            and coverage >= self.coverage_target
            and self._coverage_improved(coverage)
        )
        high_precision_with_reasonable_coverage = (
            precision >= 0.95 * precision_target and
            coverage >= 0.7 * self.coverage_target
        )
        both_reasonably_close = (
            precision >= 0.90 * precision_target and
            coverage >= 0.90 * self.coverage_target
        )
        excellent_precision = (
            precision >= precision_target and
            coverage >= 0.5 * self.coverage_target
        )
        # Increment step count
        self.step_count += 1
        
        # Check if termination reasons are enabled (not overused)
        both_targets_met_enabled = self.termination_reason_enabled["both_targets_met"]
        excellent_precision_enabled = self.termination_reason_enabled["excellent_precision"]
        high_precision_enabled = self.termination_reason_enabled["high_precision_reasonable_coverage"]
        both_close_enabled = self.termination_reason_enabled["both_reasonably_close"]
        
        # Only consider conditions that are enabled
        both_targets_met = both_targets_met and both_targets_met_enabled
        excellent_precision = excellent_precision and excellent_precision_enabled
        high_precision_with_reasonable_coverage = high_precision_with_reasonable_coverage and high_precision_enabled
        both_reasonably_close = both_reasonably_close and both_close_enabled
        
        # Validate rule validity before allowing termination
        # Check that bounds are valid: lower < upper for all features, bounds in [0, 1], and finite
        bounds_valid = True
        if np.any(self.lower >= self.upper):
            bounds_valid = False
            invalid_features = np.where(self.lower >= self.upper)[0]
            logger.warning(
                f"Invalid bounds detected: lower >= upper for features {invalid_features[:5]}. "
                f"Preventing termination until bounds are fixed."
            )
        if np.any(self.lower < 0) or np.any(self.upper > 1):
            bounds_valid = False
            logger.warning(
                f"Invalid bounds detected: bounds outside [0, 1] range. "
                f"Preventing termination until bounds are fixed."
            )
        if not np.all(np.isfinite(self.lower)) or not np.all(np.isfinite(self.upper)):
            bounds_valid = False
            logger.warning(
                f"Invalid bounds detected: NaN or Inf values in bounds. "
                f"Preventing termination until bounds are fixed."
            )
        
        # Prevent immediate termination: require minimum steps before allowing termination
        # This prevents episodes from terminating too early due to initial box configuration
        can_terminate = self.step_count >= self.min_steps_before_termination
        
        # Only allow termination if bounds are valid AND targets are met
        done = bool(
            bounds_valid and 
            can_terminate and 
            (both_targets_met or high_precision_with_reasonable_coverage or both_reasonably_close or excellent_precision)
        )
        
        termination_reason = None
        if done:
            # Determine which condition was met (check in priority order)
            if both_targets_met and both_targets_met_enabled:
                termination_reason = "both_targets_met"
            elif excellent_precision and excellent_precision_enabled:
                termination_reason = "excellent_precision"
            elif high_precision_with_reasonable_coverage and high_precision_enabled:
                termination_reason = "high_precision_reasonable_coverage"
            elif both_reasonably_close and both_close_enabled:
                termination_reason = "both_reasonably_close"
            
            # Track counts for diagnostics only. The old mechanism that permanently
            # DISABLED a termination reason after N uses changed the MDP mid-training
            # while stale transitions from the previous dynamics sat in the replay
            # buffer — removed. The terminal bonus below makes terminating at targets
            # strictly better than hovering below them, which is what the disabling
            # was trying (and failing) to enforce.
            if termination_reason:
                self.termination_reason_counts[termination_reason] += 1
                count = self.termination_reason_counts[termination_reason]

                # Terminal bonus: pays once, dwarfs anything farmable from per-step
                # terms in the remaining steps of the episode.
                reward += self.terminal_bonus
                # B2-DIAG: the bonus is added after the reward assembly above, so
                # fold it into the accumulators or the logged decomposition would
                # not sum to the true episode return.
                self._rt_terminal = getattr(self, "_rt_terminal", 0.0) + float(self.terminal_bonus)
                self._rt_total = getattr(self, "_rt_total", 0.0) + float(self.terminal_bonus)

                # Log which termination condition was met
                logger.info(
                    f"Episode terminated for class {self.target_class} (step {self.step_count}): "
                    f"{termination_reason} (count: {count}). "
                    f"Precision: {precision:.4f}, Coverage: {coverage:.4f}. "
                    f"Targets: P>={precision_target:.2f} (effective), C>={self.coverage_target:.4f}"
                )
        
        # Decompose the shaping gain for logging: precision part is exact, the
        # remainder is the gated-coverage part of the potential.
        precision_gain_component = self.alpha * precision_gain
        coverage_gain_component = shaping_gain - precision_gain_component
        
        termination_reason_code = 0.0
        if termination_reason == "both_targets_met":
            termination_reason_code = 1.0
        elif termination_reason == "excellent_precision":
            termination_reason_code = 2.0
        elif termination_reason == "high_precision_reasonable_coverage":
            termination_reason_code = 3.0
        elif termination_reason == "both_reasonably_close":
            termination_reason_code = 4.0
        
        # Info dict structure matches multi-agent for consistency and fair comparison
        # Note: Multi-agent features (inter-class overlap, shared reward, class union metrics)
        # are not applicable for single-agent environments (one agent per class, trained independently)
        info = {
            # Primary metrics (with aliases for consistency with multi-agent)
            "anchor_precision": float(precision),  # Alias for consistency with multi-agent
            "anchor_coverage": float(coverage),    # Alias for consistency with multi-agent
            "precision": float(precision),         # Keep original for backward compatibility
            "coverage": float(coverage),           # Keep original for backward compatibility
            "drift": float(drift),
            "anchor_drift": float(anchor_drift_penalty),
            "js_penalty": float(js_penalty),
            "coverage_clipped": float(1.0 if coverage_clipped else 0.0),
            "termination_reason": termination_reason_code,
            "coverage_floor_hits": float(self.coverage_floor_hits),
            "coverage_before_revert": float(coverage_before_revert) if coverage_before_revert is not None else 0.0,
            "coverage_after_revert": float(coverage_after_revert) if coverage_after_revert is not None else 0.0,
            "precision_gain": float(precision_gain),
            "coverage_gain": float(coverage_gain),
            "coverage_gain_scaled": float(coverage_gain_for_reward),
            "precision_gain_component": float(precision_gain_component),
            "coverage_gain_component": float(coverage_gain_component),
            "coverage_bonus": float(coverage_bonus),
            "target_class_bonus": float(target_class_bonus),
            "overlap_penalty": float(overlap_penalty),
            "drift_penalty": float(drift_penalty),
            "anchor_drift_penalty": float(anchor_drift_penalty),
            "coverage_floor_penalty": float(coverage_floor_penalty),
            "coverage_at_reset": float(self._coverage_at_reset),
            "coverage_improved": float(1.0 if self._coverage_improved(coverage) else 0.0),
            # Multi-agent features (not applicable, set to 0.0 for consistency)
            "inter_class_overlap_penalty": 0.0,  # Not applicable: single agent per environment
            "same_class_overlap_penalty": 0.0,   # Not applicable: single agent per environment
            "shared_reward": 0.0,                # Not applicable: single agent per environment
            "class_union_coverage": 0.0,         # Not applicable: single agent per class
            "class_union_precision": 0.0,        # Not applicable: single agent per class
            "class_union_bonus": 0.0,             # Not applicable: single agent per class
            "total_reward": float(reward),
        }
        
        # Attach classifier-level details if available
        for key, value in details.items():
            if value is not None:
                if isinstance(value, (int, float, np.number)):
                    info[key] = float(value)
                elif isinstance(value, bool):
                    info[key] = float(1.0 if value else 0.0)
                elif isinstance(value, np.ndarray):
                    info[key] = float(value.item()) if value.size == 1 else value.tolist()
                elif isinstance(value, str):
                    # Skip string entries; they are not easily logged numerically
                    continue
        
        # Provide a clearer alias for target_class_fraction, if available (matches multi-agent)
        if "target_class_fraction" in details:
            try:
                info["anchor_class_purity"] = float(details["target_class_fraction"])
            except Exception:
                pass
        
        observation = np.array(state, dtype=np.float32)
        
        self.timestep += 1
        
        max_steps_reached = (self.timestep >= self.max_cycles)
        truncated = max_steps_reached and not done
        
        # Log warning if episode terminates immediately (step_count=1 after step)
        if self.step_count == 1 and done:
            logger.warning(
                f"Episode terminated immediately (step 1) for class {self.target_class}. "
                f"Precision: {precision:.4f}, coverage: {coverage:.4f}, "
                f"Targets: precision>={self.precision_target:.2f}, coverage>={self.coverage_target:.4f}. "
                f"This may indicate initial box is too good or termination conditions too lenient."
            )
        
        if (done or truncated) and self.coverage_floor_mode == "terminal":
            # Enforce the floor once, on the box we are actually going to keep.
            # Per-dimension backoff so only the offending dimensions give up their
            # movement. `prev_lower/prev_upper` is the last box from this step, which
            # is the nearest feasible-ish reference we have.
            _p_end, _c_end, _ = self._current_metrics()
            if np.isfinite(_c_end) and _c_end < self.min_coverage_floor:
                _ref = getattr(self, "_best_box", None)
                if _ref is not None:
                    # Retreat to the best feasible box found this episode, not to the
                    # previous step (which is collapsed too when the box has run down).
                    _discarded = float(
                        np.abs(self.lower - _ref[0]).sum() + np.abs(self.upper - _ref[1]).sum()
                    )
                    _l, _u = _ref[0].copy(), _ref[1].copy()
                else:
                    _l, _u, _discarded = self._project_to_floor_per_dim(prev_lower, prev_upper)
                self.lower, self.upper = _l, _u
                self._act_proj_loss = getattr(self, "_act_proj_loss", 0.0) + _discarded
                self._act_clipped_steps = getattr(self, "_act_clipped_steps", 0) + 1
                self.coverage_floor_hits += 1
                precision, coverage, details = self._current_metrics()
                if not np.isfinite(precision):
                    precision = 0.0
                if not np.isfinite(coverage):
                    coverage = 0.0

        if truncated and not done and self.partial_terminal_credit > 0.0:
            # Same precision gate as the potential: partial credit only accrues to a
            # box that is still faithful, so this cannot be farmed by expanding into
            # garbage. Scales linearly with how far the box got toward tau_C.
            _t = max(float(self.precision_target_effective), 1e-6)
            _m = max(float(self.gate_margin), 1e-6)
            _gate = float(min(1.0, max(0.0, (float(precision) - (_t - _m)) / _m)))
            _frac = float(min(1.0, max(0.0, float(coverage) / max(float(self.coverage_target), 1e-6))))
            _partial = float(self.terminal_bonus) * self.partial_terminal_credit * _frac * _gate
            if _partial > 0.0:
                reward += _partial
                self._rt_terminal = getattr(self, "_rt_terminal", 0.0) + _partial
                self._rt_total = getattr(self, "_rt_total", 0.0) + _partial

        if done or truncated:
            n = max(1, int(self.timestep))
            logger.info(
                "B2-DIAG episode class=%s steps=%d done=%s | return=%.3f = "
                "shaping %+.3f | -overlap %.3f | -drift %.3f | -anchor_drift %.3f | "
                "cov_floor %+.3f | terminal %+.3f || ACT req=%.3f app=%.3f (%.0f%% survived) "
                "proj_loss=%.3f clipped=%d/%d |a|=%.3f maxdelta=%.5f || P=%.4f C=%.4f",
                self.target_class, n, bool(done), self._rt_total,
                self._rt_shaping, self._rt_overlap, self._rt_drift,
                self._rt_anchor_drift, self._rt_cov_floor, self._rt_terminal,
                self._act_requested, self._act_applied,
                (100.0 * self._act_applied / self._act_requested) if self._act_requested > 1e-12 else 0.0,
                self._act_proj_loss, self._act_clipped_steps, max(1, self._act_steps),
                self._act_absmean / max(1, self._act_steps),
                self._act_maxdelta / max(1, self._act_steps),
                float(precision), float(coverage),
            )

        return observation, float(reward), bool(done), truncated, info
    
    def _project_to_floor_per_dim(self, prev_lower, prev_upper):
        """Per-dimension backoff onto the coverage floor.

        The old projection scaled the WHOLE 2d-vector move by a single alpha, so one
        dimension that over-shrank scaled back all the others too. Measured on
        breast_cancer (d=30) that discarded 82% of all requested movement while
        binding on 197 of 200 steps -- the environment, not the policy, was keeping
        the box frozen.

        This reverts dimensions ONE AT A TIME, greedily choosing the dimension whose
        reversion recovers the most coverage, and stops as soon as the floor is met.
        Dimensions that were not the problem keep their movement.

        Returns (lower, upper, discarded_movement).
        """
        prev_lower = np.asarray(prev_lower, dtype=np.float32)
        prev_upper = np.asarray(prev_upper, dtype=np.float32)
        proposed_lower = self.lower.copy()
        proposed_upper = self.upper.copy()

        changed = np.where(
            (np.abs(proposed_lower - prev_lower) > 1e-9) | (np.abs(proposed_upper - prev_upper) > 1e-9)
        )[0]
        if changed.size == 0:
            return proposed_lower, proposed_upper, 0.0

        remaining = list(changed)
        for _ in range(len(changed)):
            _p, c_now, _d = self._current_metrics()
            if np.isfinite(c_now) and c_now >= self.min_coverage_floor:
                break
            best_j, best_c = None, -np.inf
            cur_lower, cur_upper = self.lower.copy(), self.upper.copy()
            for j in remaining:
                self.lower, self.upper = cur_lower.copy(), cur_upper.copy()
                self.lower[j], self.upper[j] = prev_lower[j], prev_upper[j]
                _p2, c2, _d2 = self._current_metrics()
                if np.isfinite(c2) and c2 > best_c:
                    best_c, best_j = c2, j
            self.lower, self.upper = cur_lower, cur_upper
            if best_j is None:
                break
            self.lower[best_j], self.upper[best_j] = prev_lower[best_j], prev_upper[best_j]
            remaining.remove(best_j)

        discarded = float(
            np.abs(proposed_lower - self.lower).sum() + np.abs(proposed_upper - self.upper).sum()
        )
        return self.lower.copy(), self.upper.copy(), discarded

    def _compute_anchor_drift_penalty(self, prev_lower: np.ndarray, prev_upper: np.ndarray) -> float:
        anchor_drift_penalty = 0.0
        if self.x_star_unit is not None:
            box_center = 0.5 * (self.lower + self.upper)
            if isinstance(self.x_star_unit, np.ndarray):
                anchor_point = self.x_star_unit
            else:
                anchor_point = np.array(self.x_star_unit, dtype=np.float32)
            anchor_distance = float(np.linalg.norm(box_center - anchor_point))
            max_allowed_distance = self.initial_window * 2.0
            if anchor_distance > max_allowed_distance:
                excess = anchor_distance - max_allowed_distance
                anchor_drift_penalty = self.drift_penalty_weight * excess * 0.5
        return anchor_drift_penalty
    
    # NOTE: _compute_reward_weights_and_penalties, _compute_coverage_bonus and
    # _compute_target_class_bonus were removed with the move to potential-based
    # reward shaping (see _potential and the reward block in step()).

    def get_anchor_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.lower.copy(), self.upper.copy()
    
    def extract_rule(
        self, 
        max_features_in_rule: Optional[int] = 5,
        initial_lower: Optional[np.ndarray] = None,
        initial_upper: Optional[np.ndarray] = None,
        denormalize: bool = False
    ) -> Tuple[str, str]:
        """
        Extract a human-readable rule string and canonical rule key.
        
        Returns:
            Tuple of (rule_string, canonical_key)
        """
        from utils.metrics import sparsify_box

        lower, upper, active = sparsify_box(
            self.lower,
            self.upper,
            sparsity_width_ratio=self.sparsity_width_ratio,
            max_features=max_features_in_rule,
        )
        lower_unit = lower.copy()
        upper_unit = upper.copy()

        # Denormalize bounds if requested: unit [0,1] -> standardized -> original raw units
        if denormalize:
            if self.X_min is None or self.X_range is None:
                logger.warning("Cannot denormalize: X_min or X_range not available. Using normalized bounds.")
                denormalize = False
            else:
                lower = self._unit_to_orig(lower)
                upper = self._unit_to_orig(upper)

        # Compute canonical key from normalized bounds (before denormalization)
        # This ensures canonicalization works consistently regardless of denormalization
        canonical_key = self._canonicalize_rule_key(lower_unit, upper_unit)
        
        to_show_idx = np.flatnonzero(active)
        if to_show_idx.size == 0:
            rule_str = "any values (no tightened features)"
            return rule_str, canonical_key
        
        cond_parts = []
        for i in to_show_idx:
            if i in self.categorical_indices and denormalize:
                code = int(round(float((lower[i] + upper[i]) / 2.0)))
                labels = self.categorical_value_names.get(int(i), [])
                value = labels[code] if 0 <= code < len(labels) else str(code)
                cond_parts.append(f"{self.feature_names[i]} = {value!r}")
            else:
                cond_parts.append(f"{self.feature_names[i]} ∈ [{lower[i]:.6f}, {upper[i]:.6f}]")
        
        rule_str = " and ".join(cond_parts)
        return rule_str, canonical_key
    
    def _canonicalize_rule_key(
        self,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
    ) -> str:
        """
        Create a canonical rule key from current normalized bounds for deduplication.
        Quantizes bounds to epsilon grid, drops near-full-range features, and sorts.
        
        Returns:
            Canonical rule key as string
        """
        lower = self.lower.copy() if lower is None else np.asarray(lower).copy()
        upper = self.upper.copy() if upper is None else np.asarray(upper).copy()
        min_width = getattr(self, 'min_width', 0.05)
        epsilon = max(1e-3, min_width / 4.0)
        
        # Quantize bounds to epsilon grid
        lower_quantized = np.round(lower / epsilon) * epsilon
        upper_quantized = np.round(upper / epsilon) * epsilon
        
        # Clip to valid range [0, 1]
        lower_quantized = np.clip(lower_quantized, 0.0, 1.0)
        upper_quantized = np.clip(upper_quantized, 0.0, 1.0)
        
        # Drop near-full-range features (features that haven't been tightened)
        # A feature is near-full-range if: lower <= eps AND upper >= 1 - eps
        tightened_mask = ~((lower_quantized <= epsilon) & (upper_quantized >= 1.0 - epsilon))
        tightened_indices = np.where(tightened_mask)[0]
        
        if len(tightened_indices) == 0:
            return "any_values"
        
        # Extract only tightened features and sort by feature index
        tightened_lower = lower_quantized[tightened_indices]
        tightened_upper = upper_quantized[tightened_indices]
        
        # Sort by feature index for canonical ordering
        sort_order = np.argsort(tightened_indices)
        tightened_indices_sorted = tightened_indices[sort_order]
        tightened_lower_sorted = tightened_lower[sort_order]
        tightened_upper_sorted = tightened_upper[sort_order]
        
        # Create canonical key: "f1:l1:u1;f2:l2:u2;..."
        key_parts = [f"{idx}:{lo:.6f}:{hi:.6f}" 
                     for idx, lo, hi in zip(tightened_indices_sorted, tightened_lower_sorted, tightened_upper_sorted)]
        
        return ";".join(key_parts)
    
    def render(self):
        raise NotImplementedError("Render not implemented for AnchorEnv")
    
    def close(self):
        pass

# main function to test the environment compatibility with PettingZoo. 
# The AnchorEnv is inherited from the ParallelEnv class in PettingZoo.
# This is needed for the environment to be compatible with BenchMARL.
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 1000
    n_features = 5
    n_classes = 2
    
    X_raw = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, size=n_samples).astype(int)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw).astype(np.float32)
    
    X_min = X_std.min(axis=0)
    X_max = X_std.max(axis=0)
    X_range = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))
    X_unit = (X_std - X_min) / X_range
    X_unit = np.clip(X_unit, 0.0, 1.0).astype(np.float32)
    
    try:
        from utils.networks import SimpleClassifier
        classifier = SimpleClassifier(input_dim=n_features, num_classes=n_classes, dropout_rate=0.3, use_batch_norm=True)
    except (ImportError, TypeError):
        try:
            from utils.multiagent_networks import SimpleClassifier
            classifier = SimpleClassifier(input_size=n_features, hidden_size=128, output_size=n_classes)
        except ImportError:
            class TestClassifier(torch.nn.Module):
                def __init__(self, input_dim, num_classes):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(input_dim, 64)
                    self.fc2 = torch.nn.Linear(64, 64)
                    self.fc3 = torch.nn.Linear(64, num_classes)
                    self.relu = torch.nn.ReLU()
                
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x
            
            classifier = TestClassifier(input_dim=n_features, num_classes=n_classes)
    
    classifier.eval()
    
    env_config = {
        "precision_target": 0.8,
        "coverage_target": 0.1,
        "use_perturbation": False,
        "X_min": X_min,
        "X_range": X_range,
    }
    
    test_env = SingleAgentAnchorEnv(
        X_unit=X_unit,
        X_std=X_std,
        y=y,
        feature_names=feature_names,
        classifier=classifier,
        device="cpu",
        target_class=0,  # Single agent: one target class
        env_config=env_config
    )
    
    # Test the environment
    print("Testing SingleAgentAnchorEnv...")
    obs, info = test_env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation space: {test_env.observation_space}")
    print(f"Action space: {test_env.action_space}")
    
    # Test a few steps
    for i in range(5):
        action = test_env.action_space.sample()
        obs, reward, terminated, truncated, info = test_env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            obs, info = test_env.reset()
    
    print("Environment test completed successfully!")


if __name__ == "__main__":
    main()
