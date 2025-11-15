import functools
from copy import copy
import numpy as np
import torch
from typing import Dict, Optional, Tuple, Any, List
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.device_utils import get_device
import logging
logger = logging.getLogger(__name__)


class AnchorEnv(ParallelEnv):
    metadata = {
        "name": "AnchorEnv",
        "description": "AnchorEnv is a multi-agent environment for finding anchors",
        "keywords": ["multi-agent", "anchor", "environment"],
        "render_modes": None,
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
        target_classes: Optional[List[int]] = None,
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
        
        if target_classes is None:
            if target_class is not None:
                target_classes = [target_class]
            else:
                target_classes = sorted(np.unique(y).tolist())
        
        self.target_classes = target_classes
        self.possible_agents = [f"agent_{cls}" for cls in target_classes]
        self.agent_to_class = {f"agent_{cls}": cls for cls in target_classes}
        
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
        self.inter_class_overlap_weight = env_config.get("inter_class_overlap_weight", 0.1)
        # Shared reward weight for cooperative behavior (applied to all agents)
        self.shared_reward_weight = env_config.get("shared_reward_weight", 0.2)
        
        x_star_unit_config = env_config.get("x_star_unit", None)
        if isinstance(x_star_unit_config, dict):
            self.x_star_unit = x_star_unit_config
        else:
            self.x_star_unit = {}

        self.lower = {}
        self.upper = {}
        self.prev_lower = {}
        self.prev_upper = {}
        self.box_history = {}
        self.coverage_floor_hits = {}
        self.timestep = None
        self.max_cycles = env_config.get("max_cycles", 1000)

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
    def _mask_in_box(self, agent: str) -> np.ndarray:
        if self.eval_on_test_data:
            X_eval_unit = self.X_test_unit
        else:
            X_eval_unit = self.X_unit
        
        conds = []
        for j in range(self.n_features):
            conds.append((X_eval_unit[:, j] >= self.lower[agent][j]) & (X_eval_unit[:, j] <= self.upper[agent][j]))
        mask = np.logical_and.reduce(conds) if conds else np.ones(X_eval_unit.shape[0], dtype=bool)
        return mask

    def _unit_to_std(self, X_unit_samples: np.ndarray) -> np.ndarray:
        if self.X_min is None or self.X_range is None:
            raise ValueError("X_min/X_range must be set for uniform perturbation sampling.")
        return (X_unit_samples * self.X_range) + self.X_min
    
    def _get_class_centroid(self, agent: str) -> Optional[np.ndarray]:
        """
        Get the centroid for the agent's target class.
        
        Priority:
        1. Use precomputed cluster_centroids_per_class if available
        2. Use fixed_instances_per_class if available (sample from them)
        3. Compute mean centroid from class data
        
        Args:
            agent: Agent name (e.g., "agent_0")
            
        Returns:
            Centroid in unit space [0, 1], or None if no data available
        """
        target_class = self.agent_to_class.get(agent)
        if target_class is None:
            return None
        
        # Priority 1: Use precomputed cluster centroids
        if self.cluster_centroids_per_class is not None:
            if target_class in self.cluster_centroids_per_class:
                centroids = self.cluster_centroids_per_class[target_class]
                if len(centroids) > 0:
                    # Sample a random centroid if multiple available
                    centroid_idx = self.rng.integers(0, len(centroids))
                    return np.array(centroids[centroid_idx], dtype=np.float32)
        
        # Priority 2: Use fixed instances (sample one as centroid)
        if self.fixed_instances_per_class is not None:
            if target_class in self.fixed_instances_per_class:
                instances = self.fixed_instances_per_class[target_class]
                if len(instances) > 0:
                    instance_idx = self.rng.integers(0, len(instances))
                    return np.array(instances[instance_idx], dtype=np.float32)
        
        # Priority 3: Compute mean centroid from class data
        X_data = self.X_test_unit if self.eval_on_test_data else self.X_unit
        y_data = self.y_test if self.eval_on_test_data else self.y
        
        class_mask = (y_data == target_class)
        if class_mask.sum() == 0:
            logger.warning(f"No instances found for class {target_class} to compute centroid")
            return None
        
        class_data = X_data[class_mask]
        centroid = np.mean(class_data, axis=0).astype(np.float32)
        
        return centroid
    
    def _compute_box_from_centroid(self, agent: str, centroid: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute box bounds from a centroid that are guaranteed to cover at least some points.
        
        This finds points in the class data that are closest to the centroid and computes
        box bounds (min/max) that cover those points, ensuring the box has non-zero coverage.
        
        Args:
            agent: Agent name
            centroid: Centroid point in unit space [0, 1]
            
        Returns:
            Tuple of (lower, upper) bounds, or None if no data available
        """
        target_class = self.agent_to_class.get(agent)
        if target_class is None:
            return None
        
        # Get class data
        X_data = self.X_test_unit if self.eval_on_test_data else self.X_unit
        y_data = self.y_test if self.eval_on_test_data else self.y
        
        class_mask = (y_data == target_class)
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

    def _current_metrics(self, agent: str) -> tuple:
        target_class = self.agent_to_class[agent]
        mask = self._mask_in_box(agent)
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
                    low, up = float(self.lower[agent][j]), float(self.upper[agent][j])
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
                        low, up = float(self.lower[agent][j]), float(self.upper[agent][j])
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
        positive_idx = (preds == target_class)
        if y_eval is None:
            hard_precision = float(positive_idx.mean())
        else:
            if positive_idx.sum() == 0:
                target_in_box = (y_eval == target_class)
                if target_in_box.sum() == 0:
                    hard_precision = 0.0
                else:
                    hard_precision = 0.01
            else:
                hard_precision = float((y_eval[positive_idx] == target_class).mean())

        avg_prob = float(probs[:, target_class].mean())
        precision_proxy = (
            self.precision_blend_lambda * hard_precision + (1.0 - self.precision_blend_lambda) * avg_prob
        )
        target_class_fraction = 0.0
        if y_eval is not None:
            target_class_fraction = float((y_eval == target_class).mean())
        
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
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        if hasattr(self.classifier, 'eval'):
            self.classifier.eval()
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        
        observations = {}
        infos = {}
        
        for agent in self.agents:
            # Priority 1: If x_star_unit is explicitly set (for instance-based), use it
            if self.x_star_unit.get(agent) is not None:
                w = self.initial_window
                centroid = self.x_star_unit[agent]
                self.lower[agent] = np.clip(centroid - w, 0.0, 1.0)
                self.upper[agent] = np.clip(centroid + w, 0.0, 1.0)
            # Priority 2: Use class centroid if enabled (for class-based)
            elif self.use_class_centroids:
                centroid = self._get_class_centroid(agent)
                if centroid is not None:
                    # Compute box bounds that cover points near the centroid
                    # This ensures the box covers at least some points from the cluster
                    box_bounds = self._compute_box_from_centroid(agent, centroid)
                    if box_bounds is not None:
                        self.lower[agent], self.upper[agent] = box_bounds
                    else:
                        # Fallback: Use fixed window around centroid
                        w = self.initial_window
                        self.lower[agent] = np.clip(centroid - w, 0.0, 1.0)
                        self.upper[agent] = np.clip(centroid + w, 0.0, 1.0)
                else:
                    # Fallback: Full space initialization
                    self.lower[agent] = np.zeros(self.n_features, dtype=np.float32)
                    self.upper[agent] = np.ones(self.n_features, dtype=np.float32)
            # Priority 3: Full space initialization (original behavior)
            else:
                self.lower[agent] = np.zeros(self.n_features, dtype=np.float32)
                self.upper[agent] = np.ones(self.n_features, dtype=np.float32)
            
            self.prev_lower[agent] = self.lower[agent].copy()
            self.prev_upper[agent] = self.upper[agent].copy()
            self.box_history[agent] = [(self.lower[agent].copy(), self.upper[agent].copy())]
            self.coverage_floor_hits[agent] = 0
            
            precision, coverage, _ = self._current_metrics(agent)
            state = np.concatenate([self.lower[agent], self.upper[agent], np.array([precision, coverage], dtype=np.float32)])
            
            observations[agent] = np.array(state, dtype=np.float32)
            infos[agent] = {}
        
        return observations, infos

    def _apply_action(self, agent: str, action: int):
        f = action // (len(self.directions) * len(self.step_fracs))
        rem = action % (len(self.directions) * len(self.step_fracs))
        d = rem // len(self.step_fracs)
        m = rem % len(self.step_fracs)

        direction = self.directions[d]
        step = float(self.step_fracs[m])
        cur_width = max(1e-6, self.upper[agent][f] - self.lower[agent][f])
        rel_step = step * cur_width

        if direction == "shrink_lower":
            self.lower[agent][f] = min(self.lower[agent][f] + rel_step, self.upper[agent][f] - self.min_width)
        elif direction == "expand_lower":
            self.lower[agent][f] = max(self.lower[agent][f] - rel_step, 0.0)
        elif direction == "shrink_upper":
            self.upper[agent][f] = max(self.upper[agent][f] - rel_step, self.lower[agent][f] + self.min_width)
        elif direction == "expand_upper":
            self.upper[agent][f] = min(self.upper[agent][f] + rel_step, 1.0)

        if self.upper[agent][f] - self.lower[agent][f] < self.min_width:
            mid = 0.5 * (self.upper[agent][f] + self.lower[agent][f])
            self.lower[agent][f] = max(0.0, mid - self.min_width / 2.0)
            self.upper[agent][f] = min(1.0, mid + self.min_width / 2.0)

    def _apply_continuous_action(self, agent: str, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        
        lower_deltas = action[:self.n_features]
        upper_deltas = action[self.n_features:]
        
        widths = np.maximum(self.upper[agent] - self.lower[agent], 1e-6)
        max_delta_proportional = self.max_action_scale * widths
        max_delta = np.maximum(max_delta_proportional, self.min_absolute_step)
        
        # Store before state for debugging
        lower_before = self.lower[agent].copy()
        upper_before = self.upper[agent].copy()
        
        lower_changes = lower_deltas * max_delta
        self.lower[agent] = np.clip(self.lower[agent] + lower_changes, 0.0, self.upper[agent] - self.min_width)
        
        upper_changes = upper_deltas * max_delta
        self.upper[agent] = np.clip(self.upper[agent] + upper_changes, self.lower[agent] + self.min_width, 1.0)
        
        for f in range(self.n_features):
            if self.upper[agent][f] - self.lower[agent][f] < self.min_width:
                mid = 0.5 * (self.upper[agent][f] + self.lower[agent][f])
                self.lower[agent][f] = max(0.0, mid - self.min_width / 2.0)
                self.upper[agent][f] = min(1.0, mid + self.min_width / 2.0)
        
        # Debug: Log if action was applied (only for first call per agent)
        if not hasattr(self, '_action_debug_logged'):
            self._action_debug_logged = set()
        
        if agent not in self._action_debug_logged:
            lower_diff = np.abs(self.lower[agent] - lower_before).max()
            upper_diff = np.abs(self.upper[agent] - upper_before).max()
            logger.debug(f"  _apply_continuous_action for {agent}: lower_diff={lower_diff:.6f}, upper_diff={upper_diff:.6f}, max_delta={max_delta.max():.6f}, action_mean={action.mean():.4f}")
            if lower_diff < 1e-6 and upper_diff < 1e-6:
                logger.warning(f"  ⚠ Action did not change box for {agent}! lower_deltas mean={lower_deltas.mean():.4f}, upper_deltas mean={upper_deltas.mean():.4f}, max_delta={max_delta.max():.6f}")
            self._action_debug_logged.add(agent)
    
    def step(
        self, 
        actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict]
    ]:
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        # Compute shared reward once for all agents (cooperative component)
        shared_reward = self._compute_shared_reward() if len(self.agents) > 1 else 0.0
        
        for agent in self.agents:
            if agent not in actions:
                precision, coverage, _ = self._current_metrics(agent)
                state = np.concatenate([self.lower[agent], self.upper[agent], np.array([precision, coverage], dtype=np.float32)])
                observations[agent] = np.array(state, dtype=np.float32)
                rewards[agent] = float(shared_reward)  # Still get shared reward even without action
                terminations[agent] = False
                truncations[agent] = False
                infos[agent] = {
                    "precision": precision,
                    "coverage": coverage,
                    "drift": 0.0,
                    "anchor_drift": 0.0,
                    "js_penalty": 0.0,
                    "coverage_clipped": False,
                    "termination_reason": None,
                    "coverage_floor_hits": self.coverage_floor_hits[agent],
                    "coverage_before_revert": None,
                    "coverage_after_revert": None,
                    "precision_gain": 0.0,
                    "coverage_gain": 0.0,
                    "coverage_gain_scaled": 0.0,
                    "precision_gain_component": 0.0,
                    "coverage_gain_component": 0.0,
                    "coverage_bonus": 0.0,
                    "target_class_bonus": 0.0,
                    "overlap_penalty": 0.0,
                    "drift_penalty": 0.0,
                    "anchor_drift_penalty": 0.0,
                    "inter_class_overlap_penalty": 0.0,
                    "shared_reward": float(shared_reward),
                    "total_reward": float(shared_reward),
                }
                continue
            
            action = actions[agent]
            
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            action = np.array(action, dtype=np.float32)
            
            prev_precision, prev_coverage, _ = self._current_metrics(agent)
            prev_lower = self.lower[agent].copy()
            prev_upper = self.upper[agent].copy()
            prev_widths = np.maximum(prev_upper - prev_lower, 1e-9)
            prev_vol = float(np.prod(prev_widths))
            
            if isinstance(action, np.ndarray) and action.shape[0] == 2 * self.n_features:
                self._apply_continuous_action(agent, action)
            else:
                self._apply_action(agent, int(action))
            
            precision, coverage, details = self._current_metrics(agent)

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
                logger.debug(f"  Coverage floor hit for {agent}: coverage={coverage:.6f} < min_coverage_floor={self.min_coverage_floor:.6f}, reverting box bounds")
                self.lower[agent] = prev_lower
                self.upper[agent] = prev_upper
                precision, coverage, details = self._current_metrics(agent)
                if not np.isfinite(precision):
                    precision = 0.0
                if not np.isfinite(coverage):
                    coverage = 0.0
                coverage_after_revert = float(coverage)
                self.coverage_floor_hits[agent] += 1
                coverage_clipped = True
                logger.debug(f"  Box reverted for {agent}: coverage after revert={coverage_after_revert:.6f}")

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

            widths = self.upper[agent] - self.lower[agent]
            overlap_penalty = self.gamma * float((widths < (2 * self.min_width)).mean())

            drift = float(np.linalg.norm(self.upper[agent] - prev_upper) + np.linalg.norm(self.lower[agent] - prev_lower))
            drift_penalty = self.drift_penalty_weight * drift

            anchor_drift_penalty = self._compute_anchor_drift_penalty(agent, prev_lower, prev_upper)
            
            inter_class_overlap_penalty = self._compute_inter_class_overlap_penalty(agent)
            
            # Use pre-computed shared reward (same for all agents to encourage cooperation)

            inter_lower = np.maximum(self.lower[agent], prev_lower)
            inter_upper = np.minimum(self.upper[agent], prev_upper)
            inter_widths = np.maximum(inter_upper - inter_lower, 0.0)
            inter_vol = float(np.prod(np.maximum(inter_widths, 0.0)))
            curr_widths = np.maximum(self.upper[agent] - self.lower[agent], 1e-9)
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
            # since no actual change occurred, but still give a small negative signal
            coverage_floor_penalty = 0.0
            if coverage_clipped:
                # Reduce all penalties since action didn't actually take effect
                # Drift is already ~0 since we reverted, but reduce other penalties
                penalty_reduction_factor = 0.1  # Reduce penalties by 90%
                overlap_penalty *= penalty_reduction_factor
                anchor_drift_penalty *= penalty_reduction_factor
                js_penalty *= penalty_reduction_factor
                inter_class_overlap_penalty *= penalty_reduction_factor
                # Give a small negative reward for attempting invalid action
                coverage_floor_penalty = -0.05  # Small penalty for violating coverage floor

            reward = (self.alpha * precision_weight * precision_gain + 
                     coverage_weight * coverage_gain_for_reward + 
                     coverage_bonus +
                     target_class_bonus -
                     overlap_penalty - 
                     drift_penalty - 
                     anchor_drift_penalty - 
                     js_penalty -
                     inter_class_overlap_penalty +
                     coverage_floor_penalty +
                     shared_reward)
            
            if not np.isfinite(reward):
                reward = 0.0

            self.box_history[agent].append((self.lower[agent].copy(), self.upper[agent].copy()))
            self.prev_lower[agent] = prev_lower
            self.prev_upper[agent] = prev_upper
            state = np.concatenate([self.lower[agent], self.upper[agent], np.array([precision, coverage], dtype=np.float32)])
            
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
            done = bool(both_targets_met or high_precision_with_reasonable_coverage or both_reasonably_close or excellent_precision)
            
            termination_reason = None
            if done:
                if both_targets_met:
                    termination_reason = "both_targets_met"
                elif excellent_precision:
                    termination_reason = "excellent_precision"
                elif high_precision_with_reasonable_coverage:
                    termination_reason = "high_precision_reasonable_coverage"
                elif both_reasonably_close:
                    termination_reason = "both_reasonably_close"
            
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
            
            info = {
                "precision": float(precision),
                "coverage": float(coverage),
                "drift": float(drift),
                "anchor_drift": float(anchor_drift_penalty),
                "js_penalty": float(js_penalty),
                "coverage_clipped": float(1.0 if coverage_clipped else 0.0),
                "termination_reason": termination_reason_code,
                "coverage_floor_hits": float(self.coverage_floor_hits[agent]),
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
                "inter_class_overlap_penalty": float(inter_class_overlap_penalty),
                "coverage_floor_penalty": float(coverage_floor_penalty),
                "shared_reward": float(shared_reward),
                "total_reward": float(reward),
            }
            
            for key, value in details.items():
                if value is not None:
                    if isinstance(value, (int, float, np.number)):
                        info[key] = float(value)
                    elif isinstance(value, bool):
                        info[key] = float(1.0 if value else 0.0)
                    elif isinstance(value, np.ndarray):
                        info[key] = float(value.item()) if value.size == 1 else value.tolist()
                    elif isinstance(value, str):
                        continue
            
            observations[agent] = np.array(state, dtype=np.float32)
            rewards[agent] = float(reward)
            terminations[agent] = bool(done)
            truncations[agent] = False
            infos[agent] = info
        
        self.timestep += 1
        
        max_steps_reached = (self.timestep >= self.max_cycles)
        
        if max_steps_reached:
            for agent in self.agents:
                if not terminations[agent]:
                    truncations[agent] = True
        
        if any(terminations.values()) or max_steps_reached:
            self.agents = []
        
        return observations, rewards, terminations, truncations, infos
    
    def _compute_inter_class_overlap_penalty(self, agent: str) -> float:
        if len(self.agents) <= 1:
            return 0.0
        
        agent_lower = self.lower[agent]
        agent_upper = self.upper[agent]
        agent_vol = float(np.prod(np.maximum(agent_upper - agent_lower, 1e-9)))
        
        if agent_vol <= 1e-12:
            return 0.0
        
        total_overlap_vol = 0.0
        
        for other_agent in self.agents:
            if other_agent == agent:
                continue
            
            other_lower = self.lower[other_agent]
            other_upper = self.upper[other_agent]
            
            inter_lower = np.maximum(agent_lower, other_lower)
            inter_upper = np.minimum(agent_upper, other_upper)
            inter_widths = np.maximum(inter_upper - inter_lower, 0.0)
            inter_vol = float(np.prod(np.maximum(inter_widths, 0.0)))
            
            if inter_vol > 1e-12:
                overlap_ratio = inter_vol / (agent_vol + 1e-12)
                total_overlap_vol += overlap_ratio
        
        inter_class_overlap_weight = getattr(self, 'inter_class_overlap_weight', 0.1)
        penalty = inter_class_overlap_weight * total_overlap_vol
        
        return float(np.clip(penalty, 0.0, 1.0))
    
    def _compute_anchor_drift_penalty(self, agent: str, prev_lower: np.ndarray, prev_upper: np.ndarray) -> float:
        anchor_drift_penalty = 0.0
        if self.x_star_unit.get(agent) is not None:
            box_center = 0.5 * (self.lower[agent] + self.upper[agent])
            anchor_distance = float(np.linalg.norm(box_center - self.x_star_unit[agent]))
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
    
    def _compute_shared_reward(self) -> float:
        """
        Compute shared reward component for cooperative MARL.
        All agents receive the same shared reward to encourage:
        1. All agents meeting their targets (cooperative success)
        2. Low overall overlap between all agents' rules
        3. Good average performance across all agents
        
        Returns:
            Shared reward value (same for all agents)
        """
        if len(self.agents) <= 1:
            return 0.0
        
        # Collect metrics for all agents
        all_precisions = []
        all_coverages = []
        all_targets_met = []
        
        for agent in self.agents:
            precision, coverage, _ = self._current_metrics(agent)
            all_precisions.append(precision)
            all_coverages.append(coverage)
            both_targets_met = (precision >= self.precision_target and 
                               coverage >= self.coverage_target)
            all_targets_met.append(both_targets_met)
        
        shared_reward = 0.0
        shared_reward_weight = getattr(self, 'shared_reward_weight', 0.2)
        
        # 1. Bonus when ALL agents meet their targets (strong cooperative signal)
        if all(all_targets_met):
            shared_reward += shared_reward_weight * 1.0  # Large bonus for full cooperation
        
        # 2. Bonus for fraction of agents meeting targets (partial cooperation)
        fraction_meeting_targets = sum(all_targets_met) / len(self.agents)
        if fraction_meeting_targets > 0:
            shared_reward += shared_reward_weight * 0.5 * fraction_meeting_targets
        
        # 3. Bonus for low overall overlap (complement to inter_class_overlap_penalty)
        # Compute average overlap across all agent pairs
        total_overlap = 0.0
        n_pairs = 0
        
        for i, agent_i in enumerate(self.agents):
            lower_i = self.lower[agent_i]
            upper_i = self.upper[agent_i]
            vol_i = float(np.prod(np.maximum(upper_i - lower_i, 1e-9)))
            
            if vol_i <= 1e-12:
                continue
            
            for j, agent_j in enumerate(self.agents):
                if i >= j:
                    continue
                
                lower_j = self.lower[agent_j]
                upper_j = self.upper[agent_j]
                
                inter_lower = np.maximum(lower_i, lower_j)
                inter_upper = np.minimum(upper_i, upper_j)
                inter_widths = np.maximum(inter_upper - inter_lower, 0.0)
                inter_vol = float(np.prod(np.maximum(inter_widths, 0.0)))
                
                if inter_vol > 1e-12:
                    # Normalize by average volume of the two boxes
                    vol_j = float(np.prod(np.maximum(upper_j - lower_j, 1e-9)))
                    avg_vol = 0.5 * (vol_i + vol_j)
                    overlap_ratio = inter_vol / (avg_vol + 1e-12)
                    total_overlap += overlap_ratio
                    n_pairs += 1
        
        if n_pairs > 0:
            avg_overlap = total_overlap / n_pairs
            # Reward low overlap (inverse relationship)
            overlap_bonus = shared_reward_weight * 0.3 * (1.0 - min(1.0, avg_overlap))
            shared_reward += overlap_bonus
        
        # 4. Bonus for good average precision/coverage across all agents
        avg_precision = np.mean(all_precisions) if all_precisions else 0.0
        avg_coverage = np.mean(all_coverages) if all_coverages else 0.0
        
        eps = 1e-12
        precision_progress = min(1.0, avg_precision / (self.precision_target + eps))
        coverage_progress = min(1.0, avg_coverage / (self.coverage_target + eps))
        
        # Reward when average performance is good
        if avg_precision >= self.precision_target * 0.8:
            shared_reward += shared_reward_weight * 0.2 * precision_progress
        if avg_coverage >= self.coverage_target * 0.5:
            shared_reward += shared_reward_weight * 0.2 * coverage_progress
        
        return float(np.clip(shared_reward, 0.0, shared_reward_weight * 2.0))  # Cap at reasonable maximum

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * self.n_features + 2,),
            dtype=np.float32
        )
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Box:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * self.n_features,),
            dtype=np.float32
        )

    def get_anchor_bounds(self, agent: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.lower[agent].copy(), self.upper[agent].copy()
    
    def extract_rule(
        self, 
        agent: str, 
        max_features_in_rule: Optional[int] = 5,
        initial_lower: Optional[np.ndarray] = None,
        initial_upper: Optional[np.ndarray] = None
    ) -> str:
        lower = self.lower[agent].copy()
        upper = self.upper[agent].copy()
        
        if initial_lower is None or initial_upper is None:
            initial_width = np.ones(self.n_features, dtype=np.float32)
        else:
            initial_width = initial_upper - initial_lower
        
        current_width = upper - lower
        
        if np.any(initial_width <= 0) or np.any(np.isnan(initial_width)) or np.any(np.isinf(initial_width)):
            initial_width_ref = np.ones_like(initial_width)
        else:
            initial_width_ref = initial_width.copy()
        
        if np.any(current_width <= 0) or np.any(np.isnan(current_width)) or np.any(np.isinf(current_width)):
            tightened = np.array([], dtype=int)
        else:
            tightened = np.where(current_width < initial_width_ref * 0.98)[0]
            if tightened.size == 0:
                tightened = np.where(current_width < initial_width_ref * 0.99)[0]
            if tightened.size == 0:
                tightened = np.where(current_width < 0.95)[0]
            if tightened.size == 0:
                if np.all(initial_width_ref >= 0.9):
                    tightened = np.where(current_width < 0.9)[0]
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
        from trainers.networks import SimpleClassifier
        classifier = SimpleClassifier(input_dim=n_features, num_classes=n_classes, dropout_rate=0.3, use_batch_norm=True)
    except (ImportError, TypeError):
        try:
            from trainers.multiagent_networks import SimpleClassifier
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
    
    test_env = AnchorEnv(
        X_unit=X_unit,
        X_std=X_std,
        y=y,
        feature_names=feature_names,
        classifier=classifier,
        device="cpu",
        target_classes=[0, 1],
        env_config=env_config
    )
    
    from pettingzoo.test import parallel_api_test
    parallel_api_test(test_env, num_cycles=10)


if __name__ == "__main__":
    main()
