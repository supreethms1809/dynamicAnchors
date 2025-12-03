import functools
import numpy as np
import torch
from typing import Dict, Optional, Tuple, Any, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.device_utils import get_device
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
            shape=(2 * self.n_features + 2,),  # lower + upper + precision + coverage
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
        self.precision_target = env_config.get("precision_target", 0.8)
        self.coverage_target = env_config.get("coverage_target", 0.02)
        self.precision_blend_lambda = env_config.get("precision_blend_lambda", 0.5)
        self.drift_penalty_weight = env_config.get("drift_penalty_weight", 0.05)

        self.use_perturbation = env_config.get("use_perturbation", False)
        self.perturbation_mode = env_config.get("perturbation_mode", "bootstrap")
        self.n_perturb = env_config.get("n_perturb", 1024)
        self.X_min = env_config.get("X_min", None)
        self.X_range = env_config.get("X_range", None)
        self.rng = env_config.get("rng", None)
        if self.rng is None:
            self.rng = np.random.default_rng(42)
        self.min_coverage_floor = env_config.get("min_coverage_floor", 0.005)
        self.js_penalty_weight = env_config.get("js_penalty_weight", 0.05)
        self.initial_window = env_config.get("initial_window", 0.1)
        self.fixed_instances_per_class = env_config.get("fixed_instances_per_class", None)
        self.cluster_centroids_per_class = env_config.get("cluster_centroids_per_class", None)
        self.use_random_sampling = env_config.get("use_random_sampling", False)
        self.use_class_centroids = env_config.get("use_class_centroids", True)  # Default: use centroids for initialization
        
        self.eval_on_test_data = env_config.get("eval_on_test_data", False)
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
        
        # Termination reason counters: track usage and disable overused reasons
        self.termination_reason_counts = {
            "both_targets_met": 0,
            "excellent_precision": 0,
            "high_precision_reasonable_coverage": 0,
            "both_reasonably_close": 0
        }
        self.termination_reason_max_counts = {
            "both_targets_met": env_config.get("max_termination_count_both_targets", -1),  # -1 = unlimited
            "excellent_precision": env_config.get("max_termination_count_excellent_precision", 10),
            "high_precision_reasonable_coverage": env_config.get("max_termination_count_high_precision", -1),
            "both_reasonably_close": env_config.get("max_termination_count_both_close", -1)
        }
        self.termination_reason_enabled = {
            "both_targets_met": True,
            "excellent_precision": True,
            "high_precision_reasonable_coverage": True,
            "both_reasonably_close": True
        }
        
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

        # Single agent: use direct variables instead of dictionaries
        self.lower = None
        self.upper = None
        self.prev_lower = None
        self.prev_upper = None
        self.box_history = []
        self.coverage_floor_hits = 0
        self.timestep = None
        self.max_cycles = env_config.get("max_cycles", 500)

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
        if self.eval_on_test_data:
            X_eval_unit = self.X_test_unit
        else:
            X_eval_unit = self.X_unit
        
        conds = []
        for j in range(self.n_features):
            conds.append((X_eval_unit[:, j] >= self.lower[j]) & (X_eval_unit[:, j] <= self.upper[j]))
        mask = np.logical_and.reduce(conds) if conds else np.ones(X_eval_unit.shape[0], dtype=bool)
        return mask

    def _unit_to_std(self, X_unit_samples: np.ndarray) -> np.ndarray:
        if self.X_min is None or self.X_range is None:
            raise ValueError("X_min/X_range must be set for uniform perturbation sampling.")
        return (X_unit_samples * self.X_range) + self.X_min
    
    def _get_class_centroid(self) -> Optional[np.ndarray]:
        """
        Get the centroid for the target class.
        
        Priority:
        1. Use precomputed cluster_centroids_per_class if available
        2. Use fixed_instances_per_class if available (sample from them)
        3. Compute mean centroid from class data
            
        Returns:
            Centroid in unit space [0, 1], or None if no data available
        """
        # Priority 1: Use precomputed cluster centroids
        if self.cluster_centroids_per_class is not None:
            if self.target_class in self.cluster_centroids_per_class:
                centroids = self.cluster_centroids_per_class[self.target_class]
                if len(centroids) > 0:
                    # Sample a random centroid if multiple available
                    centroid_idx = self.rng.integers(0, len(centroids))
                    return np.array(centroids[centroid_idx], dtype=np.float32)
        
        # Priority 2: Use fixed instances (sample one as centroid)
        if self.fixed_instances_per_class is not None:
            if self.target_class in self.fixed_instances_per_class:
                instances = self.fixed_instances_per_class[self.target_class]
                if len(instances) > 0:
                    instance_idx = self.rng.integers(0, len(instances))
                    return np.array(instances[instance_idx], dtype=np.float32)
        
        # Priority 3: Compute mean centroid from class data
        X_data = self.X_test_unit if self.eval_on_test_data else self.X_unit
        y_data = self.y_test if self.eval_on_test_data else self.y
        
        class_mask = (y_data == self.target_class)
        if class_mask.sum() == 0:
            logger.warning(f"No instances found for class {self.target_class} to compute centroid")
            return None
        
        class_data = X_data[class_mask]
        centroid = np.mean(class_data, axis=0).astype(np.float32)
        
        return centroid
    
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
        X_data = self.X_test_unit if self.eval_on_test_data else self.X_unit
        y_data = self.y_test if self.eval_on_test_data else self.y
        
        class_mask = (y_data == self.target_class)
        if class_mask.sum() == 0:
            return None
        
        class_data = X_data[class_mask]
        
        # Find points closest to the centroid (use at least 10% of class points, or min 5 points)
        n_neighbors = max(5, int(0.1 * len(class_data)))
        n_neighbors = min(n_neighbors, len(class_data))
        
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
                upper[f] = np.clip(lower[f] + self.min_width, self.min_width, 1.0)
        
        return lower.astype(np.float32), upper.astype(np.float32)

    def _current_metrics(self) -> tuple:
        mask = self._mask_in_box()
        covered = np.where(mask)[0]
        coverage = float(mask.mean())
        
        if covered.size == 0 and not (self.use_perturbation and self.perturbation_mode in ["uniform", "adaptive"]):
            data_source = "test" if self.eval_on_test_data else "training"
            return 0.0, coverage, {
                "hard_precision": 0.0, 
                "avg_prob": 0.0, 
                "n_points": 0, 
                "sampler": "none",
                "data_source": data_source
            }

        if self.eval_on_test_data:
            X_data_std = self.X_test_std
            y_data = self.y_test
            data_source = "test"
        else:
            X_data_std = self.X_std
            y_data = self.y
            data_source = "training"

        if not self.use_perturbation:
            X_eval = X_data_std[covered]
            y_eval = y_data[covered]
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
                        "data_source": data_source
                    }
                n_samp = min(self.n_perturb, max(1, covered.size))
                idx = self.rng.choice(covered, size=n_samp, replace=True)
                X_eval = X_data_std[idx]
                y_eval = y_data[idx]
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
                X_eval = self._unit_to_std(U)
                y_eval = None
                n_points = int(n_samp)
                sampler_note = f"uniform_{data_source}"
            elif self.perturbation_mode == "adaptive":
                min_points_for_bootstrap = max(1, int(0.1 * self.n_perturb))
                
                if covered.size >= min_points_for_bootstrap:
                    n_samp = min(self.n_perturb, covered.size)
                    idx = self.rng.choice(covered, size=n_samp, replace=True)
                    X_eval = X_data_std[idx]
                    y_eval = y_data[idx]
                    n_points = int(n_samp)
                    sampler_note = f"adaptive_bootstrap_{data_source}"
                else:
                    n_samp = self.n_perturb
                    U = np.zeros((n_samp, self.n_features), dtype=np.float32)
                    for j in range(self.n_features):
                        low, up = float(self.lower[j]), float(self.upper[j])
                        width = max(up - low, self.min_width)
                        mid = 0.5 * (low + up)
                        low = max(0.0, mid - width / 2.0)
                        up = min(1.0, mid + width / 2.0)
                        U[:, j] = self.rng.uniform(low=low, high=up, size=n_samp).astype(np.float32)
                    X_eval = self._unit_to_std(U)
                    y_eval = None
                    n_points = int(n_samp)
                    sampler_note = f"adaptive_uniform_{data_source}"
            else:
                raise ValueError(f"Unknown perturbation_mode '{self.perturbation_mode}'. Use 'bootstrap', 'uniform', or 'adaptive'.")

        if hasattr(self.classifier, 'eval'):
            self.classifier.eval()
        if hasattr(self.classifier, 'model') and hasattr(self.classifier.model, 'eval'):
            self.classifier.model.eval()
        
        with torch.no_grad():
            inputs = torch.from_numpy(X_eval).float().to(self.device)
            logits = self.classifier(inputs)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        preds = probs.argmax(axis=1)
        positive_idx = (preds == self.target_class)
        if y_eval is None:
            hard_precision = float(positive_idx.mean())
        else:
            if positive_idx.sum() == 0:
                target_in_box = (y_eval == self.target_class)
                if target_in_box.sum() == 0:
                    hard_precision = 0.0
                else:
                    hard_precision = 0.01
            else:
                hard_precision = float((y_eval[positive_idx] == self.target_class).mean())

        avg_prob = float(probs[:, self.target_class].mean())
        precision_proxy = (
            self.precision_blend_lambda * hard_precision + (1.0 - self.precision_blend_lambda) * avg_prob
        )
        target_class_fraction = 0.0
        if y_eval is not None:
            target_class_fraction = float((y_eval == self.target_class).mean())
        
        return precision_proxy, coverage, {
            "hard_precision": hard_precision,
            "avg_prob": avg_prob,
            "n_points": int(n_points),
            "sampler": sampler_note,
            "target_class_fraction": target_class_fraction,
            "data_source": data_source,
        }

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
        
        self.timestep = 0
        
        # Priority 1: If x_star_unit is explicitly set (for instance-based), use it
        if self.x_star_unit is not None:
            w = self.initial_window
            if isinstance(self.x_star_unit, np.ndarray):
                centroid = self.x_star_unit
            else:
                centroid = np.array(self.x_star_unit, dtype=np.float32)
            self.lower = np.clip(centroid - w, 0.0, 1.0).astype(np.float32)
            self.upper = np.clip(centroid + w, 0.0, 1.0).astype(np.float32)
        # Priority 2: Use class centroid if enabled (for class-based)
        elif self.use_class_centroids:
            centroid = self._get_class_centroid()
            if centroid is not None:
                # Compute box bounds that cover points near the centroid
                # This ensures the box covers at least some points from the cluster
                box_bounds = self._compute_box_from_centroid(centroid)
                if box_bounds is not None:
                    self.lower, self.upper = box_bounds
                else:
                    # Fallback: Use fixed window around centroid
                    w = self.initial_window
                    self.lower = np.clip(centroid - w, 0.0, 1.0).astype(np.float32)
                    self.upper = np.clip(centroid + w, 0.0, 1.0).astype(np.float32)
            else:
                # Fallback: Full space initialization
                self.lower = np.zeros(self.n_features, dtype=np.float32)
                self.upper = np.ones(self.n_features, dtype=np.float32)
        # Priority 3: Full space initialization (original behavior)
        else:
            self.lower = np.zeros(self.n_features, dtype=np.float32)
            self.upper = np.ones(self.n_features, dtype=np.float32)
        
        self.prev_lower = self.lower.copy()
        self.prev_upper = self.upper.copy()
        self.box_history = [(self.lower.copy(), self.upper.copy())]
        self.coverage_floor_hits = 0
        
        precision, coverage, _ = self._current_metrics()
        
        # Prevent immediate termination: require at least 2 steps before allowing termination
        # This prevents episodes from terminating on the first step due to initial box meeting targets
        self.step_count = 0
        self.min_steps_before_termination = 2  # Require at least 2 steps
        
        observation = np.concatenate([self.lower, self.upper, np.array([precision, coverage], dtype=np.float32)])
        
        info = {
            "initial_precision": float(precision),
            "initial_coverage": float(coverage),
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
        
        for f in range(self.n_features):
            if self.upper[f] - self.lower[f] < self.min_width:
                mid = 0.5 * (self.upper[f] + self.lower[f])
                self.lower[f] = max(0.0, mid - self.min_width / 2.0)
                self.upper[f] = min(1.0, mid + self.min_width / 2.0)
        
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
        
        prev_precision, prev_coverage, _ = self._current_metrics()
        prev_lower = self.lower.copy()
        prev_upper = self.upper.copy()
        prev_widths = np.maximum(prev_upper - prev_lower, 1e-9)
        prev_vol = float(np.prod(prev_widths))
        
        # Apply continuous action (always continuous for single-agent)
        self._apply_continuous_action(action)
        
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
        if coverage < self.min_coverage_floor:
            coverage_before_revert = float(coverage)
            logger.debug(f"  Coverage floor hit: coverage={coverage:.6f} < min_coverage_floor={self.min_coverage_floor:.6f}, reverting box bounds")
            self.lower = prev_lower
            self.upper = prev_upper
            precision, coverage, details = self._current_metrics()
            if not np.isfinite(precision):
                precision = 0.0
            if not np.isfinite(coverage):
                coverage = 0.0
            coverage_after_revert = float(coverage)
            self.coverage_floor_hits += 1
            coverage_clipped = True
            logger.debug(f"  Box reverted: coverage after revert={coverage_after_revert:.6f}")

        precision_gain = precision - prev_precision
        coverage_gain = coverage - prev_coverage
        
        if not np.isfinite(precision_gain):
            precision_gain = 0.0
        if not np.isfinite(coverage_gain):
            coverage_gain = 0.0
        
        min_denominator = max(prev_coverage, 1e-6)
        coverage_gain_normalized = coverage_gain / min_denominator
        coverage_gain_scaled = coverage_gain_normalized * 0.5
        coverage_gain_scaled = np.clip(coverage_gain_scaled, -0.5, 0.5)
        coverage_gain_for_reward = coverage_gain_scaled
        
        if not np.isfinite(coverage_gain_for_reward):
            coverage_gain_for_reward = 0.0

        widths = self.upper - self.lower
        overlap_penalty = self.gamma * float((widths < (2 * self.min_width)).mean())

        drift = float(np.linalg.norm(self.upper - prev_upper) + np.linalg.norm(self.lower - prev_lower))
        drift_penalty = self.drift_penalty_weight * drift

        anchor_drift_penalty = self._compute_anchor_drift_penalty(prev_lower, prev_upper)
        
        # Single agent: no inter-class overlap penalty
        inter_class_overlap_penalty = 0.0

        inter_lower = np.maximum(self.lower, prev_lower)
        inter_upper = np.minimum(self.upper, prev_upper)
        inter_widths = np.maximum(inter_upper - inter_lower, 0.0)
        inter_vol = float(np.prod(np.maximum(inter_widths, 0.0)))
        curr_widths = np.maximum(self.upper - self.lower, 1e-9)
        curr_vol = float(np.prod(curr_widths))
        eps = 1e-12
        if inter_vol <= eps:
            js_proxy = 1.0
        else:
            js_proxy = 1.0 - float(inter_vol / (0.5 * (prev_vol + curr_vol) + eps))
            js_proxy = float(np.clip(js_proxy, 0.0, 1.0))
        
        precision_threshold = self.precision_target * 0.8
        precision_weight, coverage_weight, js_penalty = self._compute_reward_weights_and_penalties(
            precision, precision_gain, coverage_gain_for_reward, js_proxy, precision_threshold, eps
        )
        
        coverage_bonus = self._compute_coverage_bonus(
            precision, coverage, coverage_gain_for_reward, precision_threshold, eps
        )
        target_class_bonus = self._compute_target_class_bonus(
            details, precision, precision_threshold, eps
        )

        # When action is reverted (coverage_clipped), reduce penalties significantly
        coverage_floor_penalty = 0.0
        if coverage_clipped:
            penalty_reduction_factor = 0.1  # Reduce penalties by 90%
            overlap_penalty *= penalty_reduction_factor
            anchor_drift_penalty *= penalty_reduction_factor
            js_penalty *= penalty_reduction_factor
            coverage_floor_penalty = -0.05  # Small penalty for violating coverage floor

        # Add small survival bonus to prevent excessive penalty for longer episodes
        # This encourages exploration while still rewarding early termination
        survival_bonus = 0.01  # Small positive reward per step to offset penalties
        
        # Scale penalties based on progress: reduce penalties when making progress
        progress_factor = 1.0
        if precision_gain > 0 or coverage_gain > 0:
            # Reduce penalties when making progress (encourage exploration)
            progress_factor = 0.5
        elif precision >= precision_threshold * 0.8:
            # Reduce penalties when close to target
            progress_factor = 0.7
        
        reward = (self.alpha * precision_weight * precision_gain + 
                 coverage_weight * coverage_gain_for_reward + 
                 coverage_bonus +
                 target_class_bonus +
                 survival_bonus -  # Add survival bonus
                 progress_factor * overlap_penalty -  # Scale penalties
                 progress_factor * drift_penalty - 
                 progress_factor * anchor_drift_penalty - 
                 progress_factor * js_penalty +
                 coverage_floor_penalty)
        
        if not np.isfinite(reward):
            reward = 0.0

        self.box_history.append((self.lower.copy(), self.upper.copy()))
        self.prev_lower = prev_lower
        self.prev_upper = prev_upper
        state = np.concatenate([self.lower, self.upper, np.array([precision, coverage], dtype=np.float32)])
        
        ## SS: Target change here: 
        eps = 1e-12
        both_targets_met = precision >= self.precision_target and coverage >= self.coverage_target
        high_precision_with_reasonable_coverage = (
            precision >= 0.95 * self.precision_target and 
            coverage >= 0.5 * self.coverage_target
        )
        both_reasonably_close = (
            precision >= 0.90 * self.precision_target and 
            coverage >= 0.90 * self.coverage_target
        )
        excellent_precision = (
            precision >= self.precision_target and 
            coverage >= 0.3 * self.coverage_target
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
            
            # Increment counter and check if we should disable this reason
            if termination_reason:
                self.termination_reason_counts[termination_reason] += 1
                count = self.termination_reason_counts[termination_reason]
                max_count = self.termination_reason_max_counts[termination_reason]
                
                # Disable reason if it exceeds max count (unless max_count is -1 for unlimited)
                if max_count > 0 and count >= max_count and self.termination_reason_enabled[termination_reason]:
                    self.termination_reason_enabled[termination_reason] = False
                    logger.warning(
                        f"Termination reason '{termination_reason}' disabled for class {self.target_class} "
                        f"after {count} uses (max: {max_count}). Agent must now meet other conditions."
                    )
                
                # Log which termination condition was met
                logger.info(
                    f"Episode terminated for class {self.target_class} (step {self.step_count}): "
                    f"{termination_reason} (count: {count}/{max_count if max_count > 0 else 'unlimited'}). "
                    f"Precision: {precision:.4f}, Coverage: {coverage:.4f}. "
                    f"Targets: P>={self.precision_target:.2f}, C>={self.coverage_target:.4f}"
                )
        
        precision_gain_component = self.alpha * precision_weight * precision_gain
        coverage_gain_component = coverage_weight * coverage_gain_for_reward
        
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
        
        return observation, float(reward), bool(done), truncated, info
    
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
    
    def _compute_reward_weights_and_penalties(
        self, precision: float, precision_gain: float, coverage_gain: float, 
        js_proxy: float, precision_threshold: float, eps: float
    ) -> tuple:
        if precision >= precision_threshold:
            precision_weight = max(2.0, 1.0 + (precision - precision_threshold) / (1.0 - precision_threshold + eps))
            coverage_weight = self.beta * min(1.0, precision / (precision_threshold + eps))
            
            if precision >= self.precision_target * 0.95 and coverage_gain > 0:
                js_penalty = self.js_penalty_weight * js_proxy * 0.3
            else:
                js_penalty = self.js_penalty_weight * js_proxy
        else:
            precision_weight = 2.0
            coverage_weight = self.beta * (0.5 + 0.5 * (precision / (precision_threshold + eps)))
            if coverage_gain > 0:
                js_penalty = self.js_penalty_weight * js_proxy * 0.5
            else:
                js_penalty = self.js_penalty_weight * js_proxy
        
        return precision_weight, coverage_weight, js_penalty
    
    def _compute_coverage_bonus(
        self, precision: float, coverage: float, coverage_gain: float, 
        precision_threshold: float, eps: float
    ) -> float:
        coverage_bonus = 0.0
        
        if precision >= precision_threshold and coverage >= self.coverage_target:
            coverage_bonus = 0.1 * (coverage / self.coverage_target)
        elif precision >= precision_threshold and coverage_gain > 0:
            progress_to_target = min(1.0, coverage / (self.coverage_target + eps))
            coverage_bonus = (0.3 + 0.7 * progress_to_target) * coverage_gain
            distance_to_target = (self.coverage_target - coverage) / (self.coverage_target + eps)
            coverage_bonus += 0.2 * coverage_gain * (1.0 - distance_to_target)
        elif precision >= precision_threshold * 0.8 and coverage_gain > 0:
            progress_to_target = min(1.0, coverage / (self.coverage_target + eps))
            coverage_bonus = (0.1 + 0.2 * progress_to_target) * coverage_gain
        
        return coverage_bonus
    
    def _compute_target_class_bonus(
        self, details: dict, precision: float, precision_threshold: float, eps: float
    ) -> float:
        target_class_bonus = 0.0
        target_class_fraction = details.get("target_class_fraction", 0.0)
        
        if target_class_fraction > 0.0 and precision < precision_threshold:
            precision_ratio = 1.0 - precision / (precision_threshold + eps)
            target_class_bonus = 0.2 * target_class_fraction * precision_ratio
            target_class_bonus *= max(0.1, precision_ratio)
        
        return target_class_bonus
    


    def get_anchor_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.lower.copy(), self.upper.copy()
    
    def extract_rule(
        self, 
        max_features_in_rule: Optional[int] = 5,
        initial_lower: Optional[np.ndarray] = None,
        initial_upper: Optional[np.ndarray] = None,
        denormalize: bool = False
    ) -> str:
        lower = self.lower.copy()
        upper = self.upper.copy()
        
        # Denormalize bounds if requested (convert from [0, 1] to original feature space)
        if denormalize:
            if self.X_min is None or self.X_range is None:
                logger.warning("Cannot denormalize: X_min or X_range not available. Using normalized bounds.")
                denormalize = False  # Disable denormalization if params not available
            else:
                lower = self._unit_to_std(lower)
                upper = self._unit_to_std(upper)
        
        if initial_lower is None or initial_upper is None:
            # Default initial width is full normalized space [0, 1]
            initial_width_normalized = np.ones(self.n_features, dtype=np.float32)
            if denormalize and self.X_min is not None and self.X_range is not None:
                # Denormalize initial width to match current width scale
                initial_width = initial_width_normalized * self.X_range
            else:
                initial_width = initial_width_normalized
        else:
            # If initial bounds are provided, they should already be in the same space as current bounds
            # But if we're denormalizing current bounds, we need to check if initial bounds are normalized
            initial_width = initial_upper - initial_lower
            # If denormalizing and initial bounds look normalized (all in [0, 1]), denormalize them too
            if denormalize and self.X_min is not None and self.X_range is not None:
                if np.all(initial_lower >= 0) and np.all(initial_lower <= 1) and np.all(initial_upper >= 0) and np.all(initial_upper <= 1):
                    # Initial bounds appear to be normalized, denormalize them
                    initial_lower_denorm = self._unit_to_std(initial_lower)
                    initial_upper_denorm = self._unit_to_std(initial_upper)
                    initial_width = initial_upper_denorm - initial_lower_denorm
        
        current_width = upper - lower
        
        if np.any(initial_width <= 0) or np.any(np.isnan(initial_width)) or np.any(np.isinf(initial_width)):
            initial_width_ref = np.ones_like(initial_width)
        else:
            initial_width_ref = initial_width.copy()
        
        if np.any(current_width <= 0) or np.any(np.isnan(current_width)) or np.any(np.isinf(current_width)):
            tightened = np.array([], dtype=int)
        else:
            # Relative thresholds (work in any scale)
            tightened = np.where(current_width < initial_width_ref * 0.98)[0]
            if tightened.size == 0:
                tightened = np.where(current_width < initial_width_ref * 0.99)[0]
            
            # Absolute thresholds (need to be scale-aware)
            if tightened.size == 0:
                if denormalize and self.X_range is not None:
                    # In denormalized space, use 0.95 * feature range as threshold
                    threshold_95 = 0.95 * self.X_range
                    tightened = np.where(current_width < threshold_95)[0]
                else:
                    # In normalized space, use 0.95 directly
                    tightened = np.where(current_width < 0.95)[0]
            
            if tightened.size == 0:
                if denormalize and self.X_range is not None:
                    # In denormalized space, use 0.9 * feature range as threshold
                    threshold_90 = 0.9 * self.X_range
                    if np.all(initial_width_ref >= threshold_90):
                        tightened = np.where(current_width < threshold_90)[0]
                else:
                    # In normalized space, use 0.9 directly
                    if np.all(initial_width_ref >= 0.9):
                        tightened = np.where(current_width < 0.9)[0]
            
            # Final fallback: any feature that tightened
            if tightened.size == 0:
                tightened = np.where(current_width < initial_width_ref)[0]
        
        if tightened.size == 0:
            return "any values (no tightened features)"
        
        tightened_sorted = np.argsort(current_width[tightened])
        if max_features_in_rule is None or max_features_in_rule == -1 or max_features_in_rule == 0:
            to_show_idx = tightened
        else:
            to_show_idx = tightened[tightened_sorted[:max_features_in_rule]]
        
        if to_show_idx.size == 0:
            return "any values (no tightened features)"
        
        cond_parts = []
        for i in to_show_idx:
            cond_parts.append(f"{self.feature_names[i]} ∈ [{lower[i]:.4f}, {upper[i]:.4f}]")
        
        return " and ".join(cond_parts)
    
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
        "coverage_target": 0.02,
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
