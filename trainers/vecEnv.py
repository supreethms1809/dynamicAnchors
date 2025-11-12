"""
Vectorized environment wrapper for Dynamic Anchor RL training compatible with Stable-Baselines3.

This module provides:
1. AnchorEnv - Core dynamic anchor environment
2. DynamicAnchorEnv - A gym.Env compatible wrapper around AnchorEnv
3. make_vec_env - Factory function to create vectorized environments for SB3

This is a STANDALONE module with all dependencies included.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, Any

# Import networks from separate module
try:
    from .networks import SimpleClassifier
except ImportError:
    from trainers.networks import SimpleClassifier

# Try to import sklearn for clustering (optional dependency)
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        raise ImportError("Please install gymnasium or gym: pip install gymnasium")

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    # Create dummy classes for type hints when SB3 is not available
    DummyVecEnv = None
    SubprocVecEnv = None


# ============================================================================
# Helper Functions: Clustering
# ============================================================================

def compute_cluster_centroids_per_class(
    X_unit: np.ndarray,
    y: np.ndarray,
    n_clusters_per_class: int = 10,
    random_state: int = 42,
    min_samples_per_cluster: int = 1
) -> Dict[int, np.ndarray]:
    """
    Compute cluster centroids for each class using KMeans clustering.
    
    This identifies dense regions (clusters) in the data and returns their centroids.
    Starting episodes from cluster centroids can improve training by:
    - Starting from more representative/typical examples
    - Reducing variance by focusing on dense regions
    - Potentially finding better anchors that cover more points
    
    IMPORTANT: This is ONLY for class-based training. For instance-based training,
    set x_star_unit explicitly on the AnchorEnv before calling reset().
    
    Args:
        X_unit: Data in unit space [0, 1], shape (n_samples, n_features)
        y: Class labels, shape (n_samples,)
        n_clusters_per_class: Number of clusters to find per class
        random_state: Random seed for reproducibility
        min_samples_per_cluster: Minimum samples required to form a cluster
        
    Returns:
        Dictionary mapping class -> array of cluster centroids (n_clusters, n_features)
        
    Note:
        Cluster centroids are used when x_star_unit is None (class-based training).
        For instance-based training, set x_star_unit directly on AnchorEnv.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "sklearn is required for cluster-based sampling. "
            "Install it with: pip install scikit-learn"
        )
    
    centroids_per_class = {}
    unique_classes = np.unique(y)
    
    for cls in unique_classes:
        cls_mask = (y == cls)
        cls_indices = np.where(cls_mask)[0]
        X_cls = X_unit[cls_indices]
        
        if len(X_cls) == 0:
            centroids_per_class[cls] = np.array([]).reshape(0, X_unit.shape[1])
            continue
        
        # Determine number of clusters (can't have more clusters than samples)
        n_clusters = min(n_clusters_per_class, len(X_cls))
        
        if n_clusters < min_samples_per_cluster:
            # Not enough samples for clustering, use mean as single centroid
            centroid = X_cls.mean(axis=0, keepdims=True)
            centroids_per_class[cls] = centroid.astype(np.float32)
        else:
            # Perform KMeans clustering
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
                max_iter=300
            )
            kmeans.fit(X_cls)
            centroids = kmeans.cluster_centers_.astype(np.float32)
            centroids_per_class[cls] = centroids
    
    return centroids_per_class


# ============================================================================
# Core Classes: Environment
# ============================================================================


class AnchorEnv:
    """
    Dynamic anchors environment over a hyper-rectangle (bounding box) in feature space.

    - State: concatenation of [lower_bounds, upper_bounds] in normalized feature space (range [0, 1])
             plus current precision, coverage.
    - Actions: choose (feature_idx, direction, magnitude)
        * direction in {shrink_lower, expand_lower, shrink_upper, expand_upper}
        * magnitude in {small, medium, large} -> applied as fraction of feature range
    - Reward: precision_gain * alpha + coverage_gain * beta - overlap_penalty - invalid_penalty
              computed w.r.t. the classifier predictions.
    
    Evaluation Mode:
        By default, precision and coverage are computed on training data (X_unit, X_std, y).
        Set eval_on_test_data=True and provide X_test_unit, X_test_std, y_test to compute
        metrics on test data instead. This allows assessing anchor generalization.
    """

    def __init__(
        self,
        X_unit: np.ndarray,
        X_std: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        classifier,
        device: str = "cpu",
        target_class: int = 1,
        step_fracs=(0.005, 0.01, 0.02),
        min_width: float = 0.05,
        alpha: float = 0.7,
        beta: float = 0.6,
        gamma: float = 0.1,
        precision_target: float = 0.8,
        coverage_target: float = 0.02,
        precision_blend_lambda: float = 0.5,
        drift_penalty_weight: float = 0.05,
        use_perturbation: bool = False,
        perturbation_mode: str = "bootstrap",  # "bootstrap", "uniform", or "adaptive"
        n_perturb: int = 1024,
        X_min: np.ndarray | None = None,
        X_range: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        min_coverage_floor: float = 0.005,
        js_penalty_weight: float = 0.05,
        x_star_unit: np.ndarray | None = None,
        initial_window: float = 0.1,
        # Optional test data for evaluation (if provided, metrics computed on test data)
        X_test_unit: np.ndarray | None = None,
        X_test_std: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        eval_on_test_data: bool = False,
    ):
        self.X_unit = X_unit
        self.X_std = X_std
        self.y = y.astype(int)
        self.feature_names = feature_names
        self.n_features = X_unit.shape[1]
        self.classifier = classifier
        # Standardize device handling
        from trainers.device_utils import get_device
        self.device = get_device(device)
        self.target_class = int(target_class)
        # Validate step_fracs is not empty to prevent division by zero in _apply_action
        if step_fracs is None or len(step_fracs) == 0:
            raise ValueError("step_fracs cannot be empty. Provide at least one step fraction value.")
        self.step_fracs = step_fracs
        self.min_width = min_width
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Actions enumerated: (feature, direction, magnitude_idx)
        self.directions = ("shrink_lower", "expand_lower", "shrink_upper", "expand_upper")
        self.n_actions = self.n_features * len(self.directions) * len(self.step_fracs)

        # Box state
        self.lower = np.zeros(self.n_features, dtype=np.float32)
        self.upper = np.ones(self.n_features, dtype=np.float32)
        self.prev_lower = self.lower.copy()
        self.prev_upper = self.upper.copy()
        self.precision_target = precision_target
        self.coverage_target = coverage_target
        self.precision_blend_lambda = precision_blend_lambda
        self.drift_penalty_weight = drift_penalty_weight

        self.box_history = []
        self.use_perturbation = bool(use_perturbation)
        self.perturbation_mode = str(perturbation_mode)
        self.n_perturb = int(n_perturb)
        self.X_min = X_min
        self.X_range = X_range
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.min_coverage_floor = float(min_coverage_floor)
        self.js_penalty_weight = float(js_penalty_weight)
        self.x_star_unit = x_star_unit.astype(np.float32) if x_star_unit is not None else None
        self.initial_window = float(initial_window)
        # Fixed instance sampling support: store fixed instances per class
        # Format: {class: array of instance indices}
        self.fixed_instances_per_class = None  # Will be set if using fixed instance sampling
        # Cluster centroids support: store cluster centroids per class
        # Format: {class: array of cluster centroids (n_clusters, n_features)}
        self.cluster_centroids_per_class = None  # Will be set if using cluster-based sampling
        # Random sampling flag: if True, randomly sample from pool instead of deterministic cycling
        self.use_random_sampling = False  # Will be set if using random sampling
        
        # Track coverage floor hits (reset per episode)
        self.coverage_floor_hits = 0
        
        # Test data evaluation support
        self.eval_on_test_data = bool(eval_on_test_data)
        if self.eval_on_test_data:
            if X_test_unit is None or X_test_std is None or y_test is None:
                raise ValueError("eval_on_test_data=True requires X_test_unit, X_test_std, and y_test")
            self.X_test_unit = X_test_unit
            self.X_test_std = X_test_std
            self.y_test = y_test.astype(int)
        else:
            self.X_test_unit = None
            self.X_test_std = None
            self.y_test = None

    def _mask_in_box(self) -> np.ndarray:
        """
        Compute mask of data points that fall within the anchor box.
        
        Returns:
            Boolean mask indicating which points are in the box.
            Uses test data if eval_on_test_data=True, otherwise uses training data.
        """
        # Use test data for evaluation if requested, otherwise use training data
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

    def _current_metrics(self) -> tuple:
        """
        Compute precision and coverage metrics.
        
        Returns:
            Tuple of (precision_proxy, coverage, details_dict)
            
        Note:
            - If eval_on_test_data=True: metrics computed on test data
            - If eval_on_test_data=False: metrics computed on training data (default)
            - Coverage: fraction of data points (training or test) that fall within the box
            - Precision: fraction of predictions in the box that are correct (for target class)
        """
        mask = self._mask_in_box()
        covered = np.where(mask)[0]
        coverage = float(mask.mean())
        
        # If no empirical points covered and not using uniform/adaptive perturbation, return early
        if covered.size == 0 and not (self.use_perturbation and self.perturbation_mode in ["uniform", "adaptive"]):
            data_source = "test" if self.eval_on_test_data else "training"
            return 0.0, coverage, {
                "hard_precision": 0.0, 
                "avg_prob": 0.0, 
                "n_points": 0, 
                "sampler": "none",
                "data_source": data_source
            }

        # Select data source based on evaluation mode
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
                # Bootstrap requires empirical points
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
                # Uniform perturbation: generate samples within box bounds, even if box covers 0 empirical points
                # This allows evaluation of boxes that are too small to contain training data
                n_samp = self.n_perturb  # Always use full perturbation budget for uniform
                U = np.zeros((n_samp, self.n_features), dtype=np.float32)
                for j in range(self.n_features):
                    low, up = float(self.lower[j]), float(self.upper[j])
                    # Ensure valid bounds (width >= min_width)
                    width = max(up - low, self.min_width)
                    mid = 0.5 * (low + up)
                    low = max(0.0, mid - width / 2.0)
                    up = min(1.0, mid + width / 2.0)
                    U[:, j] = self.rng.uniform(low=low, high=up, size=n_samp).astype(np.float32)
                X_eval = self._unit_to_std(U)
                y_eval = None  # No ground truth labels for synthetic samples
                n_points = int(n_samp)
                sampler_note = f"uniform_{data_source}"
            elif self.perturbation_mode == "adaptive":
                # Adaptive/hybrid mode: use bootstrap when plenty of points, uniform when sparse
                # Threshold: use bootstrap if we have at least 10% of n_perturb empirical points
                min_points_for_bootstrap = max(1, int(0.1 * self.n_perturb))
                
                if covered.size >= min_points_for_bootstrap:
                    # Plenty of points: use bootstrap (keeps ground truth labels)
                    n_samp = min(self.n_perturb, covered.size)
                    idx = self.rng.choice(covered, size=n_samp, replace=True)
                    X_eval = X_data_std[idx]
                    y_eval = y_data[idx]
                    n_points = int(n_samp)
                    sampler_note = f"adaptive_bootstrap_{data_source}"
                else:
                    # Sparse region: use uniform (synthetic samples)
                    n_samp = self.n_perturb
                    U = np.zeros((n_samp, self.n_features), dtype=np.float32)
                    for j in range(self.n_features):
                        low, up = float(self.lower[j]), float(self.upper[j])
                        # Ensure valid bounds (width >= min_width)
                        width = max(up - low, self.min_width)
                        mid = 0.5 * (low + up)
                        low = max(0.0, mid - width / 2.0)
                        up = min(1.0, mid + width / 2.0)
                        U[:, j] = self.rng.uniform(low=low, high=up, size=n_samp).astype(np.float32)
                    X_eval = self._unit_to_std(U)
                    y_eval = None  # No ground truth labels for synthetic samples
                    n_points = int(n_samp)
                    sampler_note = f"adaptive_uniform_{data_source}"
            else:
                raise ValueError(f"Unknown perturbation_mode '{self.perturbation_mode}'. Use 'bootstrap', 'uniform', or 'adaptive'.")

        # Ensure classifier is in eval mode before forward pass
        # This prevents hanging during evaluation (critical for stable_baselines3 evaluate_policy)
        if hasattr(self.classifier, 'eval'):
            self.classifier.eval()
        # Also ensure underlying model is in eval mode (for UnifiedClassifier wrapper)
        import torch
        if hasattr(self.classifier, 'model') and hasattr(self.classifier.model, 'eval'):
            self.classifier.model.eval()
        
        with torch.no_grad():
            inputs = torch.from_numpy(X_eval).float().to(self.device)
            logits = self.classifier(inputs)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        preds = probs.argmax(axis=1)
        positive_idx = (preds == self.target_class)
        if y_eval is None:
            # Model-consistency precision under synthetic sampling: fraction predicted as target_class
            hard_precision = float(positive_idx.mean())
        else:
            # When we have ground truth labels, calculate precision correctly
            if positive_idx.sum() == 0:
                # Classifier never predicts target class - this is a problem
                # But we should still check if the box contains target class samples
                # Use ground truth to see if box is in the right region
                target_in_box = (y_eval == self.target_class)
                if target_in_box.sum() == 0:
                    # Box doesn't contain any target class samples - precision is 0
                    hard_precision = 0.0
                else:
                    # Box contains target class samples but classifier doesn't predict them
                    # This indicates classifier bias - use a small positive value to encourage exploration
                    # But still penalize heavily (precision is very low)
                    hard_precision = 0.01  # Small positive to provide gradient, but still very low
            else:
                # Standard precision: fraction of positive predictions that are correct
                hard_precision = float((y_eval[positive_idx] == self.target_class).mean())

        avg_prob = float(probs[:, self.target_class].mean())
        precision_proxy = (
            self.precision_blend_lambda * hard_precision + (1.0 - self.precision_blend_lambda) * avg_prob
        )
        # Calculate fraction of samples that are actually target class (for reward bonus)
        target_class_fraction = 0.0
        if y_eval is not None:
            target_class_fraction = float((y_eval == self.target_class).mean())
        
        return precision_proxy, coverage, {
            "hard_precision": hard_precision,
            "avg_prob": avg_prob,
            "n_points": int(n_points),
            "sampler": sampler_note,
            "target_class_fraction": target_class_fraction,  # Fraction of samples that are actually target class
            "data_source": data_source,  # "training" or "test" - indicates which dataset metrics are computed on
        }

    def reset(self):
        # Ensure classifier is in eval mode before reset (critical for preventing hangs)
        if hasattr(self.classifier, 'eval'):
            self.classifier.eval()
        
        if self.x_star_unit is None:
            self.lower[:] = 0.0
            self.upper[:] = 1.0
        else:
            w = self.initial_window
            self.lower = np.clip(self.x_star_unit - w, 0.0, 1.0)
            self.upper = np.clip(self.x_star_unit + w, 0.0, 1.0)
        self.prev_lower = self.lower.copy()
        self.prev_upper = self.upper.copy()
        self.box_history = [(self.lower.copy(), self.upper.copy())]
        # Reset coverage floor hits counter for new episode
        self.coverage_floor_hits = 0
        precision, coverage, _ = self._current_metrics()
        state = np.concatenate([self.lower, self.upper, np.array([precision, coverage], dtype=np.float32)])
        return state

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
        """
        Apply continuous action to modify bounds (for DDPG).
        
        Args:
            action: array of shape (2 * n_features,) with values in [-1, 1]
                   First n_features: deltas for lower bounds
                   Next n_features: deltas for upper bounds
        """
        # Check if continuous actions are enabled
        if not hasattr(self, 'max_action_scale'):
            raise ValueError("Continuous actions not enabled. Set max_action_scale and min_absolute_step.")
        
        # Clip actions to [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        # Split into lower and upper deltas
        lower_deltas = action[:self.n_features]
        upper_deltas = action[self.n_features:]
        
        # Scale actions by current box width for scale-invariant updates
        widths = np.maximum(self.upper - self.lower, 1e-6)
        max_delta_proportional = self.max_action_scale * widths
        max_delta = np.maximum(max_delta_proportional, self.min_absolute_step)
        
        # Apply deltas to lower bounds
        lower_changes = lower_deltas * max_delta
        self.lower = np.clip(self.lower + lower_changes, 0.0, self.upper - self.min_width)
        
        # Apply deltas to upper bounds
        upper_changes = upper_deltas * max_delta
        self.upper = np.clip(self.upper + upper_changes, self.lower + self.min_width, 1.0)
        
        # Ensure valid bounds (enforce minimum width)
        for f in range(self.n_features):
            if self.upper[f] - self.lower[f] < self.min_width:
                mid = 0.5 * (self.upper[f] + self.lower[f])
                self.lower[f] = max(0.0, mid - self.min_width / 2.0)
                self.upper[f] = min(1.0, mid + self.min_width / 2.0)
    
    def step(self, action):
        prev_precision, prev_coverage, _ = self._current_metrics()
        prev_lower = self.lower.copy()
        prev_upper = self.upper.copy()
        prev_widths = np.maximum(prev_upper - prev_lower, 1e-9)
        prev_vol = float(np.prod(prev_widths))
        
        # Check if action is continuous (numpy array) or discrete (int)
        if isinstance(action, np.ndarray) or (hasattr(self, 'n_actions') and hasattr(self, 'max_action_scale') and self.n_actions == 2 * self.n_features):
            # Continuous action
            self._apply_continuous_action(action)
        else:
            # Discrete action
            self._apply_action(int(action))
        
        precision, coverage, details = self._current_metrics()

        # Validate metrics: check for NaN/Inf values
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
            # Store coverage before revert for tracking
            coverage_before_revert = float(coverage)
            # Revert bounds
            self.lower = prev_lower
            self.upper = prev_upper
            # Recompute metrics after revert
            precision, coverage, details = self._current_metrics()
            # Re-validate after reset
            if not np.isfinite(precision):
                precision = 0.0
            if not np.isfinite(coverage):
                coverage = 0.0
            # Store coverage after revert and increment counter
            coverage_after_revert = float(coverage)
            self.coverage_floor_hits += 1
            coverage_clipped = True

        precision_gain = precision - prev_precision
        coverage_gain = coverage - prev_coverage
        
        # Validate gains BEFORE scaling to prevent invalid scaling
        # Note: prev_precision and prev_coverage are already validated above (lines 414-417)
        if not np.isfinite(precision_gain):
            precision_gain = 0.0
        if not np.isfinite(coverage_gain):
            coverage_gain = 0.0
        
        # Normalize coverage gain to match precision gain scale
        # Coverage gains are typically 10-100x smaller than precision gains
        # Use relative scaling with minimum denominator to avoid discontinuity
        min_denominator = max(prev_coverage, 1e-6)  # Smooth transition, no discontinuity
        coverage_gain_normalized = coverage_gain / min_denominator
        # INCREASED scaling to boost coverage learning (was 0.1, now 0.5)
        # This makes coverage gains 5x more impactful in the reward
        coverage_gain_scaled = coverage_gain_normalized * 0.5
        
        # Increased cap to allow larger coverage rewards (was [-0.1, 0.1], now [-0.5, 0.5])
        # This allows coverage improvements to have stronger signal
        coverage_gain_scaled = np.clip(coverage_gain_scaled, -0.5, 0.5)
        
        # Use scaled coverage gain for reward calculation
        coverage_gain_for_reward = coverage_gain_scaled
        
        # Final validation of scaled gain
        if not np.isfinite(coverage_gain_for_reward):
            coverage_gain_for_reward = 0.0

        widths = self.upper - self.lower
        overlap_penalty = self.gamma * float((widths < (2 * self.min_width)).mean())

        drift = float(np.linalg.norm(self.upper - prev_upper) + np.linalg.norm(self.lower - prev_lower))
        drift_penalty = self.drift_penalty_weight * drift

        # Anchor drift penalty: penalize if box drifts away from x_star_unit (instance location)
        anchor_drift_penalty = self._compute_anchor_drift_penalty(prev_lower, prev_upper)

        # Volume/overlap proxy penalty (JS-like): penalize large changes in box volume
        # This is NOT actual JS divergence (which would require log terms), but a simple
        # volume-based proxy that measures how much the box overlaps with its previous state.
        # Formula: 1 - (intersection_volume / average_volume)
        # - 0.0 when boxes fully overlap (no change)
        # - 1.0 when boxes don't overlap (large change)
        inter_lower = np.maximum(self.lower, prev_lower)
        inter_upper = np.minimum(self.upper, prev_upper)
        inter_widths = np.maximum(inter_upper - inter_lower, 0.0)
        inter_vol = float(np.prod(np.maximum(inter_widths, 0.0)))
        curr_widths = np.maximum(self.upper - self.lower, 1e-9)
        curr_vol = float(np.prod(curr_widths))
        eps = 1e-12
        if inter_vol <= eps:
            js_proxy = 1.0  # No overlap, maximum penalty
        else:
            # Compute overlap ratio: intersection / average volume
            js_proxy = 1.0 - float(inter_vol / (0.5 * (prev_vol + curr_vol) + eps))
            js_proxy = float(np.clip(js_proxy, 0.0, 1.0))
        
        # Compute reward weights and penalties
        precision_threshold = self.precision_target * 0.8  # Require 80% of target precision
        precision_weight, coverage_weight, js_penalty = self._compute_reward_weights_and_penalties(
            precision, precision_gain, coverage_gain_for_reward, js_proxy, precision_threshold, eps
        )
        
        # Compute bonuses (use scaled coverage gain for bonus calculation too)
        coverage_bonus = self._compute_coverage_bonus(
            precision, coverage, coverage_gain_for_reward, precision_threshold, eps
        )
        target_class_bonus = self._compute_target_class_bonus(
            details, precision, precision_threshold, eps
        )

        reward = (self.alpha * precision_weight * precision_gain + 
                 coverage_weight * coverage_gain_for_reward + 
                 coverage_bonus +
                 target_class_bonus -
                 overlap_penalty - 
                 drift_penalty - 
                 anchor_drift_penalty - 
                 js_penalty)
        
        # Validate final reward
        if not np.isfinite(reward):
            reward = 0.0

        self.box_history.append((self.lower.copy(), self.upper.copy()))
        self.prev_lower = prev_lower
        self.prev_upper = prev_upper
        state = np.concatenate([self.lower, self.upper, np.array([precision, coverage], dtype=np.float32)])
        
        # Flexible termination conditions:
        # 1. Both targets met (ideal case)
        # 2. Very high precision (>= 95% of target) with reasonable coverage (>= 50% of target)
        # 3. Both metrics reasonably close (>= 90% of each target)
        # 4. Precision exceeds target with any positive coverage (excellent precision compensates for low coverage)
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
        # If precision exceeds target, allow termination even with lower coverage
        # This rewards excellent precision performance
        excellent_precision = (
            precision >= self.precision_target and 
            coverage >= 0.3 * self.coverage_target  # At least 30% of coverage target
        )
        done = bool(both_targets_met or high_precision_with_reasonable_coverage or both_reasonably_close or excellent_precision)
        
        # Track termination reason for debugging
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
        
        # Calculate individual reward components for debugging
        precision_gain_component = self.alpha * precision_weight * precision_gain
        coverage_gain_component = coverage_weight * coverage_gain_for_reward
        
        info = {
            "precision": precision, 
            "coverage": coverage, 
            "drift": drift, 
            "anchor_drift": anchor_drift_penalty,
            "js_penalty": js_penalty, 
            "coverage_clipped": coverage_clipped,
            # Termination reason (for debugging)
            "termination_reason": termination_reason,
            # Coverage floor tracking (for debugging learning stalls)
            "coverage_floor_hits": self.coverage_floor_hits,  # Total hits this episode
            "coverage_before_revert": coverage_before_revert,  # Coverage before revert (if revert occurred)
            "coverage_after_revert": coverage_after_revert,   # Coverage after revert (if revert occurred)
            # Reward components for debugging
            "precision_gain": precision_gain,
            "coverage_gain": coverage_gain,  # Original (unscaled) for logging
            "coverage_gain_scaled": coverage_gain_for_reward,  # Scaled version used in reward
            "precision_gain_component": precision_gain_component,
            "coverage_gain_component": coverage_gain_component,
            "coverage_bonus": coverage_bonus,
            "target_class_bonus": target_class_bonus,
            "overlap_penalty": overlap_penalty,
            "drift_penalty": drift_penalty,
            "anchor_drift_penalty": anchor_drift_penalty,
            "total_reward": reward,
            **details
        }
        return state, reward, done, info
    
    def _compute_anchor_drift_penalty(self, prev_lower: np.ndarray, prev_upper: np.ndarray) -> float:
        """Compute penalty for anchor drifting away from instance location."""
        anchor_drift_penalty = 0.0
        if self.x_star_unit is not None:
            # Compute distance from box center to instance location
            box_center = 0.5 * (self.lower + self.upper)
            anchor_distance = float(np.linalg.norm(box_center - self.x_star_unit))
            # Penalize if distance exceeds initial_window (box has drifted too far)
            max_allowed_distance = self.initial_window * 2.0  # Allow 2x initial window
            if anchor_distance > max_allowed_distance:
                excess = anchor_distance - max_allowed_distance
                anchor_drift_penalty = self.drift_penalty_weight * excess * 0.5  # Penalize drifting away
        return anchor_drift_penalty
    
    def _compute_reward_weights_and_penalties(
        self, precision: float, precision_gain: float, coverage_gain: float, 
        js_proxy: float, precision_threshold: float, eps: float
    ) -> tuple:
        """
        Compute reward weights and volume/overlap proxy penalty based on precision threshold.
        
        Args:
            js_proxy: Volume/overlap proxy value (0.0 = no change, 1.0 = large change)
                     This is NOT actual JS divergence, but a simple volume-based proxy.
            precision_threshold: Precision threshold (typically 0.8 * precision_target)
        
        Returns:
            Tuple of (precision_weight, coverage_weight, js_penalty)
        """
        
        if precision >= precision_threshold:
            # Precision is high enough, now we can optimize coverage
            # Ensure precision_weight doesn't decrease when crossing threshold (fix inconsistency)
            # Scale coverage gain by how much we exceed precision threshold
            precision_weight = max(2.0, 1.0 + (precision - precision_threshold) / (1.0 - precision_threshold + eps))
            coverage_weight = self.beta * min(1.0, precision / (precision_threshold + eps))
            
            # If precision is at target and we're increasing coverage, reduce JS penalty
            # This encourages expanding coverage when precision is already high
            # Note: coverage_gain here is the scaled version (coverage_gain_for_reward)
            if precision >= self.precision_target * 0.95 and coverage_gain > 0:
                # Reduce JS penalty when expanding coverage with high precision
                js_penalty = self.js_penalty_weight * js_proxy * 0.3  # 70% reduction
            else:
                js_penalty = self.js_penalty_weight * js_proxy
        else:
            # Precision is low, focus primarily on precision but still allow coverage learning
            # Penalize coverage gain if precision is not high enough, but less aggressively
            precision_weight = 2.0  # Higher weight for precision when it's low
            # MORE AGGRESSIVE coverage weight: use 0.5 base + 0.5 * precision ratio (was 0.3 + 0.7)
            # This gives coverage 50% weight even when precision is very low, encouraging expansion
            coverage_weight = self.beta * (0.5 + 0.5 * (precision / (precision_threshold + eps)))
            # Reduce JS penalty when expanding coverage (even if precision is low)
            # This encourages exploration and coverage expansion
            if coverage_gain > 0:
                js_penalty = self.js_penalty_weight * js_proxy * 0.5  # 50% reduction when expanding
            else:
                js_penalty = self.js_penalty_weight * js_proxy
        
        return precision_weight, coverage_weight, js_penalty
    
    def _compute_coverage_bonus(
        self, precision: float, coverage: float, coverage_gain: float, 
        precision_threshold: float, eps: float
    ) -> float:
        """Compute bonus for reaching coverage target when precision is high."""
        coverage_bonus = 0.0
        
        if precision >= precision_threshold and coverage >= self.coverage_target:
            # Bonus for meeting both targets - scales with coverage above target
            coverage_bonus = 0.1 * (coverage / self.coverage_target)  # Bonus scales with coverage above target
        elif precision >= precision_threshold and coverage_gain > 0:
            # Progressive bonus for increasing coverage when precision is high
            # The closer we get to target, the larger the bonus becomes
            # This creates a gradient that encourages reaching the target
            progress_to_target = min(1.0, coverage / (self.coverage_target + eps))
            # SIGNIFICANTLY INCREASED multipliers (was 0.1-0.4, now 0.3-1.0)
            # Bonus increases as we approach target (from 0.3 to 1.0) - much stronger signal
            coverage_bonus = (0.3 + 0.7 * progress_to_target) * coverage_gain
            
            # Extra incentive when below target but precision is high
            # Note: We're already in the elif block where coverage < self.coverage_target
            # (otherwise we'd be in the first if block), so this check is redundant but kept for clarity
            # Additional bonus proportional to how far we are from target
            # INCREASED from 0.05 to 0.2 for stronger coverage incentive
            distance_to_target = (self.coverage_target - coverage) / (self.coverage_target + eps)
            coverage_bonus += 0.2 * coverage_gain * (1.0 - distance_to_target)
        
        # NEW: Also give coverage bonus even when precision is below threshold (but not too low)
        # This encourages coverage expansion throughout training, not just when precision is high
        elif precision >= precision_threshold * 0.8 and coverage_gain > 0:  # 80% of precision threshold
            # Smaller bonus but still encourages coverage when precision is reasonable
            progress_to_target = min(1.0, coverage / (self.coverage_target + eps))
            coverage_bonus = (0.1 + 0.2 * progress_to_target) * coverage_gain
        
        return coverage_bonus
    
    def _compute_target_class_bonus(
        self, details: dict, precision: float, precision_threshold: float, eps: float
    ) -> float:
        """Compute bonus for finding regions with target class samples."""
        target_class_bonus = 0.0
        target_class_fraction = details.get("target_class_fraction", 0.0)
        
        if target_class_fraction > 0.0 and precision < precision_threshold:
            # Box contains target class samples but precision is low (classifier bias issue)
            # Give bonus proportional to fraction of target class samples
            # This helps RL agent find good regions even when classifier is biased
            # Compute precision ratio once (quadratic decay: bonus decreases faster as precision improves)
            precision_ratio = 1.0 - precision / (precision_threshold + eps)
            target_class_bonus = 0.2 * target_class_fraction * precision_ratio
            # Scale down as precision improves (less needed when classifier is working)
            # Quadratic decay: bonus decreases faster as precision improves
            target_class_bonus *= max(0.1, precision_ratio)
        
        return target_class_bonus


# ============================================================================
# Gym Wrapper for Stable-Baselines3
# ============================================================================

class DynamicAnchorEnv(gym.Env):
    """
    Gym-compatible wrapper for AnchorEnv to work with Stable-Baselines3.
    
    State Space: Box of shape (2 * n_features + 3,)
        - First n_features: lower bounds for each feature
        - Next n_features: upper bounds for each feature  
        - Next 1: current precision
        - Next 1: current coverage
        - Next 1: target_class (raw value: 0.0 or 1.0 for binary, class index for multi-class)
    
    Action Space: Discrete(n_actions)
        - Actions encode (feature_idx, direction, magnitude)
    """
    
    def __init__(self, anchor_env, seed: Optional[int] = None):
        self.anchor_env = anchor_env
        
        # Include target_class in observation space
        # Using raw class value (0.0 or 1.0) for more prominent class signal
        state_dim = 2 * anchor_env.n_features + 3  # +1 for target_class
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(anchor_env.n_actions)
        self.metadata = {"render_modes": []}
        
        if seed is not None:
            self.seed(seed)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.seed(seed)
        
        # IMPORTANT: If x_star_unit is already set (e.g., for evaluation), use it instead of sampling
        # This allows final evaluation to use specific instances while training can still sample randomly
        if self.anchor_env.x_star_unit is None:
            target_class = self.anchor_env.target_class
            
            # Check if fixed instance sampling is enabled
            if (self.anchor_env.fixed_instances_per_class is not None and 
                target_class in self.anchor_env.fixed_instances_per_class and
                len(self.anchor_env.fixed_instances_per_class[target_class]) > 0):
                # FIXED INSTANCE SAMPLING: Cycle through pre-selected instances
                # Use seed to determine which instance to use (episode number encoded in seed)
                fixed_instances = self.anchor_env.fixed_instances_per_class[target_class]
                
                # Extract episode number from seed if provided (seed = 42 + cls + ep)
                # For cycling, extract episode number: ep = seed - 42 - cls
                # This ensures each class cycles independently starting from instance 0
                if seed is not None and not getattr(self.anchor_env, 'use_random_sampling', False):
                    # Seed format: 42 + cls + ep
                    # Extract episode: ep = seed - 42 - cls
                    base_seed = 42
                    episode_num = seed - base_seed - target_class
                    instance_idx_in_pool = episode_num % len(fixed_instances)
                else:
                    # Random selection from fixed pool (reduces variance by avoiding deterministic patterns)
                    instance_idx_in_pool = self.anchor_env.rng.integers(0, len(fixed_instances))
                
                instance_idx = fixed_instances[instance_idx_in_pool]
                sampled_instance_unit = self.anchor_env.X_unit[instance_idx].astype(np.float32)
                self.anchor_env.x_star_unit = sampled_instance_unit
            else:
                # RANDOM SAMPLING (fallback or when fixed instances not set)
                # Sample an instance from the training data matching the target class
                # This ensures each episode starts with a small box around a relevant instance
                mask_target = (self.anchor_env.y == target_class)
                indices_target = np.where(mask_target)[0]
                
                if len(indices_target) > 0:
                    # Sample a random instance of the target class
                    instance_idx = self.anchor_env.rng.choice(indices_target)
                    sampled_instance_unit = self.anchor_env.X_unit[instance_idx].astype(np.float32)
                    
                    # Set x_star_unit to the sampled instance location
                    self.anchor_env.x_star_unit = sampled_instance_unit
                # If no instances found, x_star_unit remains None (will use full range)
        
        # Use a smaller initial_window for training (0.15) - starts with small box
        # IMPORTANT: Only override if it's still at default (0.1) to allow evaluation to use larger window (0.3)
        # Don't override if it's already been set to a specific value (e.g., 0.3 for evaluation)
        if self.anchor_env.initial_window == 0.1:  # Only override default value
            self.anchor_env.initial_window = 0.15
        # If initial_window is already set to 0.3 (for evaluation), keep it at 0.3
        
        state = self.anchor_env.reset()
        obs = np.array(state, dtype=np.float32)
        
        # Append target_class to observation (raw value 0.0 or 1.0 for better distinction)
        # Using raw class value for more prominent class signal
        target_class_value = float(self.anchor_env.target_class)
        obs = np.concatenate([obs, np.array([target_class_value], dtype=np.float32)])
        
        info = {
            "episode_metrics": {
                "precision": self.anchor_env._current_metrics()[0],
                "coverage": self.anchor_env._current_metrics()[1],
            }
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        next_state, reward, done, info = self.anchor_env.step(action)
        obs = np.array(next_state, dtype=np.float32)
        
        # Append target_class to observation (raw value 0.0 or 1.0 for better distinction)
        # Must match reset() to maintain consistency
        target_class_value = float(self.anchor_env.target_class)
        obs = np.concatenate([obs, np.array([target_class_value], dtype=np.float32)])
        
        terminated = done
        truncated = False
        
        # Copy all info keys to step_info, including reward components
        # This ensures reward breakdown logging works correctly
        step_info = dict(info)  # Copy all keys from info
        # Ensure required keys exist with defaults if missing
        step_info.setdefault("precision", 0.0)
        step_info.setdefault("coverage", 0.0)
        step_info.setdefault("hard_precision", info.get("hard_precision", info.get("precision", 0.0)))
        step_info.setdefault("drift", 0.0)
        step_info.setdefault("js_penalty", 0.0)
        step_info.setdefault("coverage_clipped", False)
        step_info.setdefault("coverage_floor_hits", 0)
        step_info.setdefault("sampler", "unknown")
        step_info.setdefault("n_points", 0)
        
        return obs, float(reward), bool(terminated), bool(truncated), step_info
    
    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.anchor_env.rng = np.random.default_rng(seed)
        return [seed]  # Return seed in list format for gym compatibility
    
    def render(self):
        raise NotImplementedError("Render not implemented for DynamicAnchorEnv")
    
    def close(self):
        pass


# ============================================================================
# Vectorization Functions for Stable-Baselines3
# ============================================================================

def make_dynamic_anchor_env(anchor_env_fn, seed: Optional[int] = None) -> DynamicAnchorEnv:
    """Create a single DynamicAnchorEnv instance."""
    anchor_env = anchor_env_fn()
    return DynamicAnchorEnv(anchor_env, seed=seed)


def make_vec_env(
    anchor_env_fn,
    n_envs: int = 1,
    vec_env_cls: Optional[Any] = None,
    seed: Optional[int] = None,
    start_method: Optional[str] = None,
    **vec_env_kwargs
):
    """
    Create a vectorized environment for Stable-Baselines3.
    
    Args:
        anchor_env_fn: Function that returns an AnchorEnv instance
        n_envs: Number of parallel environments
        vec_env_cls: Vectorized environment class (DummyVecEnv or SubprocVecEnv)
        seed: Base random seed
        start_method: Start method for SubprocVecEnv
        **vec_env_kwargs: Additional arguments
    
    Returns:
        Vectorized environment instance
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable_baselines3 is not installed. Please install it with: pip install stable-baselines3")
    
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    
    if vec_env_cls is None:
        vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    
    env_fns = []
    for i in range(n_envs):
        env_seed = seed + i if seed is not None else None
        env_fns.append(lambda i=i, s=env_seed: make_dynamic_anchor_env(anchor_env_fn, seed=s))
    
    if vec_env_cls == SubprocVecEnv:
        kwargs = {**vec_env_kwargs}
        if start_method is not None:
            kwargs["start_method"] = start_method
        return vec_env_cls(env_fns, **kwargs)
    else:
        return vec_env_cls(env_fns, **vec_env_kwargs)


def make_dummy_vec_env(anchor_env_fn, n_envs: int = 1, seed: Optional[int] = None):
    """Create a DummyVecEnv (synchronous vectorization)."""
    return make_vec_env(anchor_env_fn=anchor_env_fn, n_envs=n_envs, vec_env_cls=DummyVecEnv, seed=seed)


def make_subproc_vec_env(anchor_env_fn, n_envs: int = 4, seed: Optional[int] = None, start_method: Optional[str] = None):
    """Create a SubprocVecEnv (asynchronous parallel vectorization)."""
    return make_vec_env(anchor_env_fn=anchor_env_fn, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=seed, start_method=start_method)


# ============================================================================
# Continuous Actions Wrapper for DDPG
# ============================================================================

class ContinuousAnchorEnv(gym.Env):
    """
    Gym-compatible wrapper for AnchorEnv with continuous actions (for DDPG).
    
    State Space: Box of shape (2 * n_features + 2,)
        - First n_features: lower bounds for each feature
        - Next n_features: upper bounds for each feature  
        - Next 1: current precision
        - Next 1: current coverage
    
    Action Space: Box of shape (2 * n_features,)
        - First n_features: deltas for lower bounds (clipped to [-1, 1])
        - Next n_features: deltas for upper bounds (clipped to [-1, 1])
    """
    
    def __init__(self, anchor_env, seed: Optional[int] = None):
        self.anchor_env = anchor_env
        self.n_features = anchor_env.n_features
        
        # Observation space: [lower_bounds, upper_bounds, precision, coverage]
        state_dim = 2 * anchor_env.n_features + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action space: [lower_deltas, upper_deltas] each in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * anchor_env.n_features,),
            dtype=np.float32
        )
        
        self.metadata = {"render_modes": []}
        
        if seed is not None:
            self.seed(seed)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment and return initial observation."""
        # Ensure classifier is in eval mode before reset (critical for preventing hangs)
        if hasattr(self.anchor_env.classifier, 'eval'):
            self.anchor_env.classifier.eval()
        
        if seed is not None:
            self.seed(seed)
        
        # IMPORTANT: If x_star_unit is already set (e.g., for instance-based evaluation), 
        # use it directly and skip all sampling (cluster centroids, fixed instances, random).
        # Cluster centroids are ONLY used for class-based training, not instance-based training.
        # This allows final evaluation to use specific instances while training can use cluster centroids.
        if self.anchor_env.x_star_unit is None:
            target_class = self.anchor_env.target_class
            
            # Priority order: 1) Cluster centroids, 2) Fixed instances, 3) Random sampling
            # NOTE: Cluster centroids are ONLY for class-based training (when x_star_unit is None).
            # For instance-based training, x_star_unit should be set explicitly before reset().
            # CLUSTER-BASED SAMPLING: Use cluster centroids (dense regions) for better starting points
            if (self.anchor_env.cluster_centroids_per_class is not None and 
                target_class in self.anchor_env.cluster_centroids_per_class and
                len(self.anchor_env.cluster_centroids_per_class[target_class]) > 0):
                # Sample a cluster centroid for this episode
                centroids = self.anchor_env.cluster_centroids_per_class[target_class]
                
                # Extract episode number from seed if provided (seed = 42 + cls + ep)
                # For cycling, extract episode number: ep = seed - 42 - cls
                # This ensures each class cycles independently starting from centroid 0
                if seed is not None and not getattr(self.anchor_env, 'use_random_sampling', False):
                    # Seed format: 42 + cls + ep
                    # Extract episode: ep = seed - 42 - cls
                    base_seed = 42
                    episode_num = seed - base_seed - target_class
                    centroid_idx = episode_num % len(centroids)
                else:
                    # Random selection from centroids (reduces variance by avoiding deterministic patterns)
                    centroid_idx = self.anchor_env.rng.integers(0, len(centroids))
                
                # Use the selected cluster centroid as the starting point
                sampled_instance_unit = centroids[centroid_idx].astype(np.float32)
                self.anchor_env.x_star_unit = sampled_instance_unit
            # Check if fixed instance sampling is enabled
            elif (self.anchor_env.fixed_instances_per_class is not None and 
                target_class in self.anchor_env.fixed_instances_per_class and
                len(self.anchor_env.fixed_instances_per_class[target_class]) > 0):
                # FIXED INSTANCE SAMPLING: Cycle through pre-selected instances
                # Use seed to determine which instance to use (episode number encoded in seed)
                fixed_instances = self.anchor_env.fixed_instances_per_class[target_class]
                
                # Extract episode number from seed if provided (seed = 42 + cls + ep)
                # For cycling, extract episode number: ep = seed - 42 - cls
                # This ensures each class cycles independently starting from instance 0
                if seed is not None and not getattr(self.anchor_env, 'use_random_sampling', False):
                    # Seed format: 42 + cls + ep
                    # Extract episode: ep = seed - 42 - cls
                    base_seed = 42
                    episode_num = seed - base_seed - target_class
                    instance_idx_in_pool = episode_num % len(fixed_instances)
                else:
                    # Random selection from fixed pool (reduces variance by avoiding deterministic patterns)
                    instance_idx_in_pool = self.anchor_env.rng.integers(0, len(fixed_instances))
                
                instance_idx = fixed_instances[instance_idx_in_pool]
                sampled_instance_unit = self.anchor_env.X_unit[instance_idx].astype(np.float32)
                self.anchor_env.x_star_unit = sampled_instance_unit
            else:
                # RANDOM SAMPLING (fallback or when fixed instances/clusters not set)
                # Sample an instance from the training data matching the target class
                mask_target = (self.anchor_env.y == target_class)
                indices_target = np.where(mask_target)[0]
                
                if len(indices_target) > 0:
                    instance_idx = self.anchor_env.rng.choice(indices_target)
                    sampled_instance_unit = self.anchor_env.X_unit[instance_idx].astype(np.float32)
                    self.anchor_env.x_star_unit = sampled_instance_unit
                # If no instances found, x_star_unit remains None (will use full range)
        
        # Use smaller initial_window for training (0.15) unless already set (e.g., 0.3 for evaluation)
        if self.anchor_env.initial_window == 0.1:  # Only override default
            self.anchor_env.initial_window = 0.15
        
        state = self.anchor_env.reset()
        obs = np.array(state, dtype=np.float32)
        
        info = {
            "precision": self.anchor_env._current_metrics()[0],
            "coverage": self.anchor_env._current_metrics()[1],
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Ensure action is numpy array
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = np.clip(action, -1.0, 1.0)
        
        next_state, reward, done, info = self.anchor_env.step(action)
        obs = np.array(next_state, dtype=np.float32)
        
        # Copy all info keys to step_info, including reward components
        # This ensures reward breakdown logging works correctly
        step_info = dict(info)  # Copy all keys from info
        # Ensure required keys exist with defaults if missing
        step_info.setdefault("precision", 0.0)
        step_info.setdefault("coverage", 0.0)
        step_info.setdefault("hard_precision", info.get("hard_precision", info.get("precision", 0.0)))
        step_info.setdefault("drift", 0.0)
        step_info.setdefault("js_penalty", 0.0)
        step_info.setdefault("coverage_clipped", False)
        step_info.setdefault("coverage_floor_hits", 0)
        step_info.setdefault("sampler", "unknown")
        step_info.setdefault("n_points", 0)
        
        terminated = done
        truncated = False
        
        return obs, reward, terminated, truncated, step_info
    
    def seed(self, seed: Optional[int] = None):
        """Set the random seed for the environment."""
        if seed is not None:
            self.anchor_env.rng = np.random.default_rng(seed)
        return [seed]  # Return seed in list format for gym compatibility
    
    def render(self):
        raise NotImplementedError("Render not implemented for ContinuousAnchorEnv")
    
    def close(self):
        pass
