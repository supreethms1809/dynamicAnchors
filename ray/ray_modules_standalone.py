"""
Standalone module for Ray RLlib Dynamic Anchors training.

This module contains all necessary components for training and evaluating
dynamic anchor explanations using Ray RLlib, without dependencies on the
trainers directory.

Includes:
- Device utilities (get_device, get_device_pair)
- AnchorEnv class (core environment)
- ContinuousAnchorEnv wrapper (for continuous actions)
- compute_cluster_centroids_per_class function
- Evaluation functions (greedy_rollout, evaluate_all_classes, etc.)
"""

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import torch
from typing import Dict, Optional, Tuple, Any, List, Union
from collections import Counter

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

# For evaluation functions - these are optional (only needed if using SB3 models)
try:
    from stable_baselines3 import PPO, DDPG, TD3
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    # Create dummy classes for type hints
    class PPO:
        pass
    class DDPG:
        pass
    class TD3:
        pass


# ============================================================================
# NEURAL NETWORK CLASSIFIER
# ============================================================================
import torch.nn as nn


class SimpleClassifier(nn.Module):
    """Simple neural network classifier for tabular data."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3, use_batch_norm: bool = True):
        """
        Initialize the classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization (default: 0.3)
            use_batch_norm: Whether to use batch normalization (default: True)
        """
        super().__init__()
        self.num_classes = num_classes
        
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, 256))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Second layer
        layers.append(nn.Linear(256, 256))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Third layer (optional, for deeper network)
        layers.append(nn.Linear(256, 128))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate * 0.5))  # Less dropout in later layers
        
        # Output layer
        layers.append(nn.Linear(128, num_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.net(x)


# ============================================================================
# DEVICE UTILITIES
# ============================================================================
def _is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.
    
    MPS is Apple's GPU acceleration framework for PyTorch on macOS.
    
    Returns:
        True if MPS is available, False otherwise
    """
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def _get_auto_device() -> torch.device:
    """
    Auto-detect the best available device.
    
    Priority order:
    1. CUDA (if available)
    2. MPS (if available, macOS Apple Silicon)
    3. CPU (fallback)
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif _is_mps_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device(device: Union[str, torch.device, None] = None) -> torch.device:
    """
    Standardize device handling: convert any device input to torch.device.
    
    Args:
        device: Device specification. Can be:
            - None: Auto-detect (cuda > mps > cpu)
            - str: "cpu", "cuda", "cuda:0", "mps", "auto", etc.
            - torch.device: Already a device object
    
    Returns:
        torch.device object
    
    Examples:
        >>> get_device()  # Auto-detect
        device(type='cpu')  # or 'cuda' or 'mps' depending on availability
        >>> get_device("cuda")
        device(type='cuda')
        >>> get_device("mps")
        device(type='mps')
        >>> get_device("auto")
        device(type='cpu')  # or 'cuda' or 'mps' if available
    """
    if device is None:
        # Auto-detect: use best available device
        device = _get_auto_device()
    elif isinstance(device, torch.device):
        # Already a device object, use as-is
        pass
    elif isinstance(device, str):
        if device.lower() == "auto":
            # Auto-detect
            device = _get_auto_device()
        else:
            # Convert string to device
            # Validate MPS availability if requested
            if device.lower() == "mps" and not _is_mps_available():
                raise RuntimeError(
                    "MPS (Metal Performance Shaders) is not available. "
                    "MPS requires macOS with Apple Silicon (M1/M2/M3). "
                    "Falling back to CPU is not automatic - please specify 'cpu' or 'auto'."
                )
            device = torch.device(device)
    else:
        raise ValueError(f"Invalid device type: {type(device)}. Expected str, torch.device, or None.")
    
    return device


def get_device_str(device: Union[str, torch.device, None] = None) -> str:
    """
    Get device as string representation.
    
    Args:
        device: Device specification (same as get_device)
    
    Returns:
        String representation of device (e.g., "cpu", "cuda", "cuda:0")
    """
    device_obj = get_device(device)
    return str(device_obj)


def get_device_pair(device: Union[str, torch.device, None] = None) -> tuple[torch.device, str]:
    """
    Get both device object and string representation.
    
    Useful when you need both formats (e.g., for PyTorch operations and string parameters).
    
    Args:
        device: Device specification (same as get_device)
    
    Returns:
        Tuple of (device_obj, device_str)
    
    Examples:
        >>> device_obj, device_str = get_device_pair("cuda")
        >>> device_obj
        device(type='cuda')
        >>> device_str
        'cuda'
    """
    device_obj = get_device(device)
    device_str = str(device_obj)
    return device_obj, device_str

def set_device(device: Union[str, torch.device, None] = None) -> None:
    """
    Set the device for the current process.
    
    Args:
        device: Device specification (same as get_device)
    """
    torch.cuda.set_device(device)

# ============================================================================
# ENVIRONMENT CLASSES
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
        # get_device is defined above in this file
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


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================
def compute_coverage_on_data(
    lower: np.ndarray,
    upper: np.ndarray,
    X_unit: np.ndarray
) -> float:
    """
    Compute coverage of a box on a dataset.
    
    Args:
        lower: Lower bounds of the box (in unit space [0,1])
        upper: Upper bounds of the box (in unit space [0,1])
        X_unit: Data points in unit space [0,1], shape (n_samples, n_features)
    
    Returns:
        Coverage: fraction of data points that fall within the box
    """
    if X_unit is None or X_unit.shape[0] == 0:
        return 0.0
    
    # Check which points fall within the box
    mask = np.all((X_unit >= lower) & (X_unit <= upper), axis=1)
    coverage = float(mask.mean())
    return coverage


def greedy_rollout(
    env,
    trained_model: Union[PPO, DDPG, TD3],
    steps_per_episode: int = 100,
    max_features_in_rule: Optional[int] = 5,
    device: str = "cpu"
) -> Tuple[Dict[str, Any], str, np.ndarray, np.ndarray]:
    """
    Perform greedy evaluation with a trained policy.
    
    Args:
        env: AnchorEnv, DynamicAnchorEnv, or ContinuousAnchorEnv instance to evaluate on
        trained_model: Trained PPO or DDPG model
        steps_per_episode: Maximum number of steps for rollout
        max_features_in_rule: Maximum number of features to include in rule.
            Use -1 or None to include all tightened features (useful for feature importance).
            Default: 5
    
    Returns:
        Tuple of (info_dict, rule_string, lower_bounds, upper_bounds)
    """
    
    # Detect model type: DDPG/TD3 have actor/critic, PPO has policy
    # TD3 also has actor/critic, so we treat it similarly to DDPG
    is_continuous = (hasattr(trained_model, 'actor') and hasattr(trained_model, 'critic'))
    is_ddpg = is_continuous  # For backward compatibility
    is_td3 = is_continuous  # TD3 has same structure as DDPG
    is_ppo = hasattr(trained_model, 'policy')
    
    # Detect environment type
    is_continuous_env = hasattr(env, 'anchor_env') and hasattr(env, 'action_space') and hasattr(env.action_space, 'shape')
    is_dynamic_env = hasattr(env, 'anchor_env') and not is_continuous_env
    is_anchor_env = not hasattr(env, 'anchor_env')
    
    # Reset environment and get initial state
    reset_result = env.reset()
    
    # Handle both AnchorEnv (returns state) and gym wrappers (returns (obs, info) tuple)
    if isinstance(reset_result, tuple):
        state, reset_info = reset_result
    else:
        state = reset_result
        reset_info = {}
    
    # Ensure state is numpy array
    state = np.array(state, dtype=np.float32)
    
    # Get anchor_env reference and initial bounds
    if is_continuous_env or is_dynamic_env:
        # ContinuousAnchorEnv or DynamicAnchorEnv: access anchor_env for properties
        anchor_env = env.anchor_env
        initial_lower = anchor_env.lower.copy()
        initial_upper = anchor_env.upper.copy()
        # Get initial metrics from anchor_env
        prec, cov, _ = anchor_env._current_metrics()
        # Note: DynamicAnchorEnv already includes target_class in observation space,
        # so we don't need to append it here
    else:
        # AnchorEnv: access properties directly
        anchor_env = env
        initial_lower = env.lower.copy()
        initial_upper = env.upper.copy()
        prec, cov, _ = env._current_metrics()
        # For AnchorEnv (not wrapped), we need to append target_class to state
        # only if the model expects it (check observation space)
        # However, if using PPO with DynamicAnchorEnv wrapper, target_class is already included
        # So we only append for raw AnchorEnv usage
        if is_ppo and not is_ddpg and not is_dynamic_env:
            target_class_value = float(env.target_class)
            state = np.concatenate([state, np.array([target_class_value], dtype=np.float32)])
    
    initial_width = (initial_upper - initial_lower)
    
    last_info = {
        "precision": prec,
        "coverage": cov,
        "hard_precision": prec,
        "avg_prob": prec,
        "sampler": "empirical"
    }
    bounds_changed = False
    
    # Run greedy rollout
    for t in range(steps_per_episode):
        with torch.no_grad():
            # Use trained model to predict action
            action, _states = trained_model.predict(state, deterministic=True)
            
            # Handle action type based on model
            if is_continuous:
                # DDPG/TD3: action is already a numpy array (continuous)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                action = np.clip(action, -1.0, 1.0)
            else:
                # PPO: action is discrete (int)
                action = int(action)
        
        prev_lower = anchor_env.lower.copy()
        prev_upper = anchor_env.upper.copy()
        step_result = env.step(action)
        
        # Handle both gym.Env API (5-tuple) and old API (4-tuple)
        if len(step_result) == 5:
            # gym.Env API: (observation, reward, terminated, truncated, info)
            state, _, done, _, info = step_result
        else:
            # Old API: (state, reward, done, info)
            state, _, done, info = step_result
        
        # Ensure state is numpy array
        state = np.array(state, dtype=np.float32)
        
        # For AnchorEnv (not wrapped), append target_class only if needed
        # DynamicAnchorEnv already includes target_class in observation space
        if is_anchor_env and is_ppo and not is_ddpg and not is_dynamic_env:
            target_class_value = float(env.target_class)
            state = np.concatenate([state, np.array([target_class_value], dtype=np.float32)])
        
        if not np.allclose(prev_lower, anchor_env.lower) or not np.allclose(prev_upper, anchor_env.upper):
            bounds_changed = True
        
        # Extract info properly (handle both dict and gym.Env info format)
        if isinstance(info, dict):
            last_info = info
        else:
            # If info is not a dict, try to get from anchor_env
            prec, cov, det = anchor_env._current_metrics()
            last_info = {
                "precision": prec,
                "coverage": cov,
                "hard_precision": det.get("hard_precision", prec),
                "avg_prob": det.get("avg_prob", prec),
                "sampler": det.get("sampler", "empirical")
            }
        
        if done:
            break
    
    # If box didn't change at all, manually tighten a bit
    if not bounds_changed:
        n_tighten = min(5, anchor_env.n_features)
        idx_perm = anchor_env.rng.permutation(anchor_env.n_features)[:n_tighten]
        
        for j in idx_perm:
            width = anchor_env.upper[j] - anchor_env.lower[j]
            if width > anchor_env.min_width:
                shrink = 0.1 * width
                anchor_env.lower[j] = min(anchor_env.lower[j] + shrink, anchor_env.upper[j] - anchor_env.min_width)
                anchor_env.upper[j] = max(anchor_env.upper[j] - shrink, anchor_env.lower[j] + anchor_env.min_width)
    
    # Always recompute final metrics from anchor_env to ensure accuracy
    # This ensures we get the actual current state metrics, not just the last step's info
    prec_final, cov_final, det_final = anchor_env._current_metrics()
    last_info = {
        "precision": prec_final,
        "coverage": cov_final,
        "hard_precision": det_final.get("hard_precision", prec_final),
        "avg_prob": det_final.get("avg_prob", prec_final),
        "sampler": det_final.get("sampler", "empirical"),
        "n_points": det_final.get("n_points", 0)
    }
    
    # Build rule string
    # Compare final width to initial width to find tightened features
    lw = (anchor_env.upper - anchor_env.lower)
    
    # Ensure initial_width is valid (should be > 0 for all features)
    # If initial_width has zeros or invalid values, something went wrong
    if np.any(initial_width <= 0) or np.any(np.isnan(initial_width)) or np.any(np.isinf(initial_width)):
        # If initial_width is invalid, use full range (1.0) as reference
        # This handles edge cases where reset() might have set invalid bounds
        initial_width_ref = np.ones_like(initial_width)
    else:
        initial_width_ref = initial_width.copy()
    
    # Ensure lw is also valid
    if np.any(lw <= 0) or np.any(np.isnan(lw)) or np.any(np.isinf(lw)):
        # If final width is invalid, no features can be tightened
        tightened = np.array([], dtype=int)
    else:
        # A feature is "tightened" if its width is smaller than initial width
        # Use multiple thresholds to catch different levels of tightening
        
        # First, check for significant tightening (2% reduction)
        tightened = np.where(lw < initial_width_ref * 0.98)[0]
        
        # If no features tightened by 2%, try a more lenient threshold (1%)
        if tightened.size == 0:
            tightened = np.where(lw < initial_width_ref * 0.99)[0]
        
        # If still no tightened features, check absolute thresholds
        # This handles cases where initial_width might be small
        if tightened.size == 0:
            # Check if any feature is significantly smaller than full range (1.0)
            # This catches cases where we started near full range
            tightened = np.where(lw < 0.95)[0]
        
        # Additional check: if initial_width was close to full range (>= 0.9), 
        # any feature with width < 0.9 should be considered tightened
        if tightened.size == 0:
            if np.all(initial_width_ref >= 0.9):
                tightened = np.where(lw < 0.9)[0]
        
        # Final fallback: check if ANY feature has width smaller than its initial width
        # This is the most lenient check - any reduction counts
        if tightened.size == 0:
            tightened = np.where(lw < initial_width_ref)[0]
    
    if tightened.size == 0:
        rule = "any values (no tightened features)"
    else:
        tightened_sorted = np.argsort(lw[tightened])
        # Use all features if max_features_in_rule is -1, None, or 0
        if max_features_in_rule is None or max_features_in_rule == -1 or max_features_in_rule == 0:
            to_show_idx = tightened
        else:
            to_show_idx = tightened[tightened_sorted[:max_features_in_rule]]
        
        if to_show_idx.size == 0:
            rule = "any values (no tightened features)"
        else:
            cond_parts = []
            for i in to_show_idx:
                cond_parts.append(f"{anchor_env.feature_names[i]} ∈ [{anchor_env.lower[i]:.2f}, {anchor_env.upper[i]:.2f}]")
            rule = " and ".join(cond_parts)
    
    return last_info, rule, anchor_env.lower.copy(), anchor_env.upper.copy()


def evaluate_single_instance(
    X_instance: np.ndarray,
    trained_model: Union[PPO, DDPG, TD3],
    make_env_fn,
    feature_names: List[str],
    target_class: int,
    steps_per_episode: int = 100,
    max_features_in_rule: Optional[int] = 5,
    X_min: Optional[np.ndarray] = None,
    X_range: Optional[np.ndarray] = None,
    eval_on_test_data: bool = False,
    X_test_unit: Optional[np.ndarray] = None,
    X_test_std: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    initial_window: Optional[float] = None,
    num_rollouts_per_instance: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate and generate anchor for a single instance.
    
    Args:
        X_instance: Instance to explain (in standardized space, single row)
        trained_model: Trained PPO or DDPG model
        make_env_fn: Function that creates an AnchorEnv instance
        feature_names: List of feature names
        target_class: Target class to explain
        steps_per_episode: Maximum rollout steps
        max_features_in_rule: Maximum features in rule.
            Use -1 or None to include all tightened features (useful for feature importance).
            Default: 5
        X_min: Optional min values for normalization (to unit space)
        X_range: Optional range values for normalization (to unit space)
        eval_on_test_data: If True, compute metrics on test data instead of training data
        X_test_unit: Test data in unit space [0,1] (required if eval_on_test_data=True)
        X_test_std: Test data in standardized space (required if eval_on_test_data=True)
        y_test: Test labels (required if eval_on_test_data=True)
        initial_window: Initial window size for anchor box (default: 0.3 for eval, matches training if None)
        num_rollouts_per_instance: Number of greedy rollouts to run per instance (default: 1)
                                  If > 1, metrics are averaged across rollouts
    
    Returns:
        Dictionary with anchor explanation and metrics.
        Note: 
        - Coverage and precision are computed on training data by default.
        - Set eval_on_test_data=True to compute on test data.
        - local_coverage: coverage on the data used during greedy (micro-set)
        - global_coverage: coverage on full test split (if X_test_unit provided)
        If num_rollouts_per_instance > 1, metrics are averaged across rollouts.
    """
    # Detect model type: DDPG/TD3 have actor/critic, PPO has policy
    # TD3 also has actor/critic, so we treat it similarly to DDPG
    is_continuous = (hasattr(trained_model, 'actor') and hasattr(trained_model, 'critic'))
    is_ddpg = is_continuous  # For backward compatibility
    is_td3 = is_continuous  # TD3 has same structure as DDPG
    is_ppo = hasattr(trained_model, 'policy')
    
    # Create a temporary environment to extract normalization parameters if needed
    temp_env = make_env_fn()
    
    # Get X_min and X_range from environment if not provided
    if X_min is None:
        X_min = temp_env.X_min
    if X_range is None:
        X_range = temp_env.X_range
    
    # Normalize instance to unit space [0, 1] for x_star_unit
    if X_min is not None and X_range is not None:
        X_instance_unit = (X_instance - X_min) / X_range
        X_instance_unit = np.clip(X_instance_unit, 0.0, 1.0).astype(np.float32)
    else:
        X_instance_unit = None
    
    # Create environment with x_star_unit set to the instance location
    # Default initial_window: use training default (0.15) if not specified, or use provided value
    # Note: Previous code used 0.3 for evaluation, but this may not match training conditions
    # AnchorEnv, DynamicAnchorEnv, ContinuousAnchorEnv is defined above in this file
    
    # Set initial_window: use provided value, or default to training value (0.15) for consistency
    if initial_window is None:
        # Default to training value for consistency, unless explicitly overridden
        eval_initial_window = temp_env.initial_window if hasattr(temp_env, 'initial_window') else 0.15
    else:
        eval_initial_window = initial_window
    
    # Prepare test data if evaluation on test data is requested
    if eval_on_test_data:
        if X_test_unit is None or X_test_std is None or y_test is None:
            raise ValueError(
                "eval_on_test_data=True requires X_test_unit, X_test_std, and y_test. "
                "These should be provided when calling evaluate_single_instance."
            )
    
    # Create base AnchorEnv
    anchor_env = AnchorEnv(
        X_unit=temp_env.X_unit,
        X_std=temp_env.X_std,
        y=temp_env.y,
        feature_names=temp_env.feature_names,
        classifier=temp_env.classifier,
        device=str(temp_env.device),
        target_class=target_class,
        step_fracs=temp_env.step_fracs,
        min_width=temp_env.min_width,
        alpha=temp_env.alpha,
        beta=temp_env.beta,
        gamma=temp_env.gamma,
        precision_target=temp_env.precision_target,
        coverage_target=temp_env.coverage_target,
        precision_blend_lambda=temp_env.precision_blend_lambda,
        drift_penalty_weight=temp_env.drift_penalty_weight,
        use_perturbation=temp_env.use_perturbation,
        perturbation_mode=temp_env.perturbation_mode,
        n_perturb=temp_env.n_perturb,
        X_min=temp_env.X_min,
        X_range=temp_env.X_range,
        rng=temp_env.rng,
        min_coverage_floor=temp_env.min_coverage_floor,
        js_penalty_weight=temp_env.js_penalty_weight,
        x_star_unit=X_instance_unit,  # Set to instance location
        initial_window=eval_initial_window,
        # Test data evaluation support
        eval_on_test_data=eval_on_test_data,
        X_test_unit=X_test_unit if eval_on_test_data else None,
        X_test_std=X_test_std if eval_on_test_data else None,
        y_test=y_test if eval_on_test_data else None,
    )
    
    # Wrap with appropriate gym wrapper based on model type
    if is_continuous:
        # DDPG/TD3: Use ContinuousAnchorEnv
        # Enable continuous actions in AnchorEnv
        anchor_env.n_actions = 2 * anchor_env.n_features
        anchor_env.max_action_scale = max(temp_env.step_fracs) if temp_env.step_fracs else 0.02
        anchor_env.min_absolute_step = max(0.05, temp_env.min_width * 0.5)
        env = ContinuousAnchorEnv(anchor_env, seed=42)
    else:
        # PPO: Use DynamicAnchorEnv
        # IMPORTANT: x_star_unit is already set on anchor_env, so DynamicAnchorEnv.reset() will preserve it
        env = DynamicAnchorEnv(anchor_env, seed=42)
    
    # Note: Don't reset here - greedy_rollout will reset internally
    # This avoids double reset and ensures initial_width is captured correctly
    # Check initial coverage will be done inside greedy_rollout after reset
    
    # Run multiple greedy rollouts if requested
    if num_rollouts_per_instance > 1:
        # Collect results from multiple rollouts
        rollout_results = []
        for rollout_idx in range(num_rollouts_per_instance):
            # Create a new environment for each rollout to ensure fresh state
            # Use different seed for each rollout to introduce variation
            if is_ddpg:
                env_rollout = ContinuousAnchorEnv(anchor_env, seed=42 + rollout_idx)
            else:
                env_rollout = DynamicAnchorEnv(anchor_env, seed=42 + rollout_idx)
            
            info, rule, lower, upper = greedy_rollout(
                env_rollout,
                trained_model,
                steps_per_episode=steps_per_episode,
                max_features_in_rule=max_features_in_rule
            )
            rollout_results.append({
                "info": info,
                "rule": rule,
                "lower": lower,
                "upper": upper
            })
        
        # Average metrics across rollouts
        precisions = [r["info"].get("precision", 0.0) for r in rollout_results]
        hard_precisions = [r["info"].get("hard_precision", r["info"].get("precision", 0.0)) for r in rollout_results]
        local_coverages = [r["info"].get("coverage", 0.0) for r in rollout_results]
        
        # Compute global coverage for each rollout (on full test split)
        global_coverages = []
        if X_test_unit is not None:
            for r in rollout_results:
                global_cov = compute_coverage_on_data(r["lower"], r["upper"], X_test_unit)
                global_coverages.append(global_cov)
        
        avg_precision = np.mean(precisions)
        avg_hard_precision = np.mean(hard_precisions)
        avg_local_coverage = np.mean(local_coverages)
        avg_global_coverage = np.mean(global_coverages) if global_coverages else None
        
        # Use the best rollout (by hard precision) for the rule and bounds
        best_idx = np.argmax(hard_precisions)
        best_result = rollout_results[best_idx]
        
        # Also compute std dev for reporting
        std_precision = np.std(precisions) if len(precisions) > 1 else 0.0
        std_hard_precision = np.std(hard_precisions) if len(hard_precisions) > 1 else 0.0
        std_local_coverage = np.std(local_coverages) if len(local_coverages) > 1 else 0.0
        std_global_coverage = np.std(global_coverages) if len(global_coverages) > 1 else None
        
        return {
            "rule": best_result["rule"],
            "precision": float(avg_precision),
            "hard_precision": float(avg_hard_precision),
            "coverage": float(avg_local_coverage),  # Keep for backward compatibility
            "local_coverage": float(avg_local_coverage),  # Coverage on micro-set used during greedy
            "global_coverage": float(avg_global_coverage) if avg_global_coverage is not None else None,  # Coverage on full test split
            "lower_bounds": best_result["lower"].tolist(),
            "upper_bounds": best_result["upper"].tolist(),
            "data_source": best_result["info"].get("data_source", "training"),
            "num_rollouts": num_rollouts_per_instance,
            "std_precision": float(std_precision),
            "std_hard_precision": float(std_hard_precision),
            "std_coverage": float(std_local_coverage),  # Keep for backward compatibility
            "std_local_coverage": float(std_local_coverage),
            "std_global_coverage": float(std_global_coverage) if std_global_coverage is not None else None,
        }
    else:
        # Single rollout (original behavior)
        info, rule, lower, upper = greedy_rollout(
            env,
            trained_model,
            steps_per_episode=steps_per_episode,
            max_features_in_rule=max_features_in_rule
        )
        
        # Get local coverage (from greedy rollout - coverage on data used during greedy)
        local_coverage = info.get("coverage", 0.0)
        
        # Compute global coverage (on full test split) if test data is available
        global_coverage = None
        if X_test_unit is not None:
            global_coverage = compute_coverage_on_data(lower, upper, X_test_unit)
        
        return {
            "rule": rule,
            "precision": info.get("precision", 0.0),
            "hard_precision": info.get("hard_precision", 0.0),
            "coverage": local_coverage,  # Keep for backward compatibility
            "local_coverage": local_coverage,  # Coverage on micro-set used during greedy
            "global_coverage": global_coverage,  # Coverage on full test split (None if not available)
            "lower_bounds": lower.tolist(),
            "upper_bounds": upper.tolist(),
            "data_source": info.get("data_source", "training"),
            "num_rollouts": 1,
        }


def evaluate_class(
    X_test: np.ndarray,
    y_test: np.ndarray,
    trained_model: Union[PPO, DDPG],
    make_env_fn,
    feature_names: List[str],
    target_class: int,
    n_instances: int = 20,
    steps_per_episode: int = 100,
    max_features_in_rule: Optional[int] = 5,
    random_seed: int = 42,
    eval_on_test_data: bool = False,
    X_test_unit: Optional[np.ndarray] = None,
    X_test_std: Optional[np.ndarray] = None,
    initial_window: Optional[float] = None,
    num_rollouts_per_instance: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate and generate anchors for multiple instances of a class.
    
    This function samples n_instances from the target class, generates anchors
    for each, and returns aggregated statistics.
    
    Args:
        X_test: Test instances (used to sample instances to explain)
        y_test: Test labels (used to sample instances to explain)
        trained_model: Trained PPO model
        make_env_fn: Function that creates an AnchorEnv instance
        feature_names: List of feature names
        target_class: Target class to evaluate
        n_instances: Number of instances to evaluate
        steps_per_episode: Maximum rollout steps
        max_features_in_rule: Maximum features in rule.
            Use -1 or None to include all tightened features (useful for feature importance).
            Default: 5
        random_seed: Random seed for sampling
        eval_on_test_data: If True, compute metrics on test data instead of training data
        X_test_unit: Test data in unit space [0,1] (required if eval_on_test_data=True)
        X_test_std: Test data in standardized space (required if eval_on_test_data=True)
        initial_window: Initial window size for anchor box (default: matches training)
        num_rollouts_per_instance: Number of greedy rollouts per instance (default: 1)
                                  If > 1, metrics are averaged across rollouts for each instance
    
    Returns:
        Dictionary with aggregated metrics and individual results.
        Note: By default, coverage and precision are computed on training data.
        Set eval_on_test_data=True to compute on test data.
        If num_rollouts_per_instance > 1, metrics are averaged across rollouts per instance.
    """
    # Sample instances from target class
    rng = np.random.default_rng(random_seed)
    idx_cls = np.where(y_test == target_class)[0]
    
    if idx_cls.size == 0:
        return {
            "avg_precision": 0.0,
            "avg_hard_precision": 0.0,
            "avg_coverage": 0.0,
            "n_instances": 0,
            "individual_results": []
        }
    
    sel = rng.choice(idx_cls, size=min(n_instances, idx_cls.size), replace=False)
    
    # Extract normalization parameters once for all instances
    temp_env = make_env_fn()
    X_min = temp_env.X_min
    X_range = temp_env.X_range
    
    # Prepare test data if evaluation on test data is requested
    if eval_on_test_data:
        if X_test_unit is None or X_test_std is None:
            raise ValueError(
                "eval_on_test_data=True requires X_test_unit and X_test_std. "
                "These should be provided when calling evaluate_class."
            )
    
    # Evaluate each instance
    individual_results = []
    for i, instance_idx in enumerate(sel):
        result = evaluate_single_instance(
            X_instance=X_test[instance_idx],
            trained_model=trained_model,
            make_env_fn=make_env_fn,
            feature_names=feature_names,
            target_class=target_class,
            steps_per_episode=steps_per_episode,
            max_features_in_rule=max_features_in_rule,
            X_min=X_min,
            X_range=X_range,
            eval_on_test_data=eval_on_test_data,
            X_test_unit=X_test_unit,
            X_test_std=X_test_std,
            y_test=y_test,
            initial_window=initial_window,
            num_rollouts_per_instance=num_rollouts_per_instance,
        )
        result["instance_idx"] = int(instance_idx)
        individual_results.append(result)
    
    # Aggregate statistics
    # Check if individual_results is empty to prevent np.mean on empty list (returns nan)
    if len(individual_results) == 0:
        avg_precision = 0.0
        avg_hard_precision = 0.0
        avg_coverage = 0.0
        avg_local_coverage = None
        avg_global_coverage = None
        best_result = None
    else:
        avg_precision = np.mean([r["precision"] for r in individual_results])
        avg_hard_precision = np.mean([r["hard_precision"] for r in individual_results])
        avg_coverage = np.mean([r["coverage"] for r in individual_results])  # Backward compatibility
        
        # Aggregate local and global coverage separately
        local_coverages = [r.get("local_coverage", r.get("coverage", 0.0)) for r in individual_results]
        global_coverages = [r.get("global_coverage") for r in individual_results if r.get("global_coverage") is not None]
        avg_local_coverage = np.mean(local_coverages) if local_coverages else None
        avg_global_coverage = np.mean(global_coverages) if global_coverages else None
        
        # Find best anchor (by hard precision)
        best_result = max(individual_results, key=lambda r: r["hard_precision"])
    
    # Compute union coverage: how many unique test instances are covered by at least one anchor
    union_coverage = None
    if eval_on_test_data and X_test_unit is not None:
        # Check which test instances are covered by at least one anchor
        covered_mask = np.zeros(X_test_unit.shape[0], dtype=bool)
        for result in individual_results:
            lower = np.array(result["lower_bounds"])
            upper = np.array(result["upper_bounds"])
            # Check which test instances fall in this anchor box
            instance_mask = np.all(
                (X_test_unit >= lower) & (X_test_unit <= upper), axis=1
            )
            covered_mask |= instance_mask
        union_coverage = float(covered_mask.mean())
    
    return {
        "avg_precision": float(avg_precision),
        "avg_hard_precision": float(avg_hard_precision),
        "avg_coverage": float(avg_coverage),  # Backward compatibility
        "avg_local_coverage": float(avg_local_coverage) if avg_local_coverage is not None else None,
        "avg_global_coverage": float(avg_global_coverage) if avg_global_coverage is not None else None,
        "union_coverage": union_coverage,  # Union coverage across all anchors (test data only)
        "n_instances": len(individual_results),
        "best_rule": best_result["rule"] if best_result is not None else "",
        "best_precision": best_result["hard_precision"] if best_result is not None else 0.0,
        "individual_results": individual_results,
        "data_source": individual_results[0].get("data_source", "training") if individual_results else "training",
    }


def evaluate_class_level(
    trained_model: Union[PPO, DDPG, TD3, Dict[int, Any]],
    make_env_fn,
    feature_names: List[str],
    target_class: int,
    steps_per_episode: int = 100,
    max_features_in_rule: Optional[int] = 5,
    random_seed: int = 42,
    eval_on_test_data: bool = False,
    X_test_unit: Optional[np.ndarray] = None,
    X_test_std: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    initial_window: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate and generate ONE anchor per class (class-level evaluation).
    
    This creates a single anchor optimized for the entire class, not per-instance.
    This matches how the RL agent was trained (class-level optimization).
    
    Args:
        trained_model: Trained PPO model, DDPG model, or dict of DDPG trainers per class
        make_env_fn: Function that creates AnchorEnv instance
        feature_names: List of feature names
        target_class: Target class to evaluate
        steps_per_episode: Maximum rollout steps
        max_features_in_rule: Maximum features in rule.
            Use -1 or None to include all tightened features (useful for feature importance).
            Default: 5
        random_seed: Random seed for consistency
        eval_on_test_data: If True, compute metrics on test data instead of training data
        X_test_unit: Test data in unit space [0,1] (required if eval_on_test_data=True)
        X_test_std: Test data in standardized space (required if eval_on_test_data=True)
        y_test: Test labels (required if eval_on_test_data=True)
        initial_window: Initial window size for anchor box (default: matches training)
    
    Returns:
        Dictionary with class-level anchor metrics (one anchor per class).
        Coverage is global coverage (fraction of all training/test data covered).
    """
    # Create environment for this class (no x_star_unit - class-level, not instance-level)
    anchor_env = make_env_fn()
    
    # Set initial_window if provided
    if initial_window is not None:
        anchor_env.initial_window = initial_window
    
    # For class-level evaluation, don't set x_star_unit (let it be None)
    # This allows the anchor to optimize for the entire class, not a specific instance
    anchor_env.x_star_unit = None
    
    # CRITICAL: For class-level evaluation, use uniform perturbation if available
    # This allows metrics to be computed even if the box doesn't contain training data points
    # (which can happen when the policy moves the box to a region with no data)
    if hasattr(anchor_env, 'use_perturbation') and hasattr(anchor_env, 'perturbation_mode'):
        # Ensure perturbation is enabled for class-level evaluation
        if not anchor_env.use_perturbation:
            anchor_env.use_perturbation = True
            print(f"  [DEBUG] Enabled perturbation for class-level evaluation")
        # Switch to uniform perturbation (works even when box has no training data)
        if anchor_env.perturbation_mode == "bootstrap":
            anchor_env.perturbation_mode = "uniform"
            print(f"  [DEBUG] Switched to uniform perturbation for class-level evaluation")
    
    # CRITICAL: Reset the environment after setting x_star_unit = None
    # This ensures the box is initialized correctly (full box [0.0, 1.0] for all features)
    # The greedy_rollout will call reset() again, but we need to ensure x_star_unit is None first
    anchor_env.reset()
    
    # Handle different model types
    if isinstance(trained_model, dict):
        # DDPG/TD3: Use per-class trainer
        if target_class not in trained_model:
            return {
                "precision": 0.0,
                "hard_precision": 0.0,
                "coverage": 0.0,
                "rule": "no model",
                "lower_bounds": None,
                "upper_bounds": None,
                "data_source": "training",
                "evaluation_type": "class_level"
            }
        continuous_trainer = trained_model[target_class]
        continuous_model = continuous_trainer.model if hasattr(continuous_trainer, 'model') else continuous_trainer
        
        # Enable continuous actions
        anchor_env.n_actions = 2 * anchor_env.n_features
        anchor_env.max_action_scale = 0.02  # Default step size
        anchor_env.min_absolute_step = 0.05
        
        # Wrap with ContinuousAnchorEnv
        # ContinuousAnchorEnv is defined above in this file
        gym_env = ContinuousAnchorEnv(anchor_env, seed=random_seed)
        
        # Use greedy rollout
        info, rule, lower, upper = greedy_rollout(
            gym_env,
            continuous_model,
            steps_per_episode=steps_per_episode,
            max_features_in_rule=max_features_in_rule
        )
    else:
        # PPO: Use PPO model
        # DynamicAnchorEnv is defined above in this file
        wrapped_env = DynamicAnchorEnv(anchor_env, seed=random_seed)
        
        info, rule, lower, upper = greedy_rollout(
            wrapped_env,
            trained_model,
            steps_per_episode=steps_per_episode,
            max_features_in_rule=max_features_in_rule
        )
    
    # Get metrics from the final anchor
    # Coverage is already global (computed on all training/test data)
    local_coverage = info.get("coverage", 0.0)
    
    # DEBUG: If coverage/precision are 0, check what's in the info dict and recompute metrics
    if local_coverage == 0.0 or info.get("precision", 0.0) == 0.0:
        # Recompute metrics directly from anchor_env to debug
        prec_debug, cov_debug, det_debug = anchor_env._current_metrics()
        print(f"  [DEBUG] Class-level evaluation for class {target_class}:")
        print(f"    Info dict keys: {list(info.keys())}")
        print(f"    Info precision: {info.get('precision', 'NOT FOUND')}")
        print(f"    Info coverage: {info.get('coverage', 'NOT FOUND')}")
        print(f"    Direct recompute - precision: {prec_debug:.6f}, coverage: {cov_debug:.6f}")
        print(f"    Direct recompute - hard_precision: {det_debug.get('hard_precision', 'NOT FOUND')}")
        print(f"    Direct recompute - n_points: {det_debug.get('n_points', 'NOT FOUND')}")
        print(f"    Direct recompute - sampler: {det_debug.get('sampler', 'NOT FOUND')}")
        print(f"    Box bounds - lower: {lower[:3]}..., upper: {upper[:3]}...")
        print(f"    Box width: {(upper - lower)[:3]}...")
        # Use recomputed values if info values are 0
        if local_coverage == 0.0:
            local_coverage = cov_debug
        if info.get("precision", 0.0) == 0.0:
            info["precision"] = prec_debug
            info["hard_precision"] = det_debug.get("hard_precision", prec_debug)
    
    # Compute global coverage on test data if available
    global_coverage = None
    if eval_on_test_data and X_test_unit is not None:
        global_coverage = compute_coverage_on_data(lower, upper, X_test_unit)
    
    return {
        "rule": rule,
        "precision": info.get("precision", 0.0),
        "hard_precision": info.get("hard_precision", 0.0),
        "coverage": local_coverage,  # This is already global coverage (on training/test data)
        "local_coverage": local_coverage,  # For consistency with instance-level API
        "global_coverage": global_coverage if global_coverage is not None else local_coverage,
        "lower_bounds": lower.tolist(),
        "upper_bounds": upper.tolist(),
        "data_source": info.get("data_source", "training"),
        "evaluation_type": "class_level"
    }


def evaluate_all_classes(
    X_test: np.ndarray,
    y_test: np.ndarray,
    trained_model: Union[PPO, DDPG, TD3, Dict[int, Any]],
    make_env_fn,
    feature_names: List[str],
    n_instances_per_class: int = 20,
    steps_per_episode: int = 100,
    max_features_in_rule: int = 5,
    random_seed: int = 42,
    eval_on_test_data: bool = False,
    X_test_unit: Optional[np.ndarray] = None,
    X_test_std: Optional[np.ndarray] = None,
    initial_window: Optional[float] = None,
    num_rollouts_per_instance: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate and generate anchors for all classes.
    
    Args:
        X_test: Test instances (used to sample instances to explain)
        y_test: Test labels (used to sample instances to explain)
        trained_model: Trained PPO model, DDPG model, or dict of DDPG trainers per class
        make_env_fn: Function that creates AnchorEnv instance
        feature_names: List of feature names
        n_instances_per_class: Number of instances per class to evaluate
        steps_per_episode: Maximum rollout steps
        max_features_in_rule: Maximum features in rule
        random_seed: Random seed for sampling
        eval_on_test_data: If True, compute metrics on test data instead of training data
        X_test_unit: Test data in unit space [0,1] (required if eval_on_test_data=True)
        X_test_std: Test data in standardized space (required if eval_on_test_data=True)
        initial_window: Initial window size for anchor box (default: matches training)
        num_rollouts_per_instance: Number of greedy rollouts per instance (default: 1)
                                  If > 1, metrics are averaged across rollouts for each instance
    
    Returns:
        Dictionary with per-class results and overall statistics.
        Note: By default, coverage and precision are computed on training data.
        Set eval_on_test_data=True to compute on test data.
        If num_rollouts_per_instance > 1, metrics are averaged across rollouts per instance.
    """
    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)
    
    # Handle DDPG trainers dict (per-class trainers)
    if isinstance(trained_model, dict):
        # DDPG/TD3: Use per-class trainers
        print(f"Evaluating anchors for {n_classes} classes with {n_instances_per_class} instances each (Continuous action per-class trainers)")
        
        per_class_results = {}
        for cls in unique_classes:
            cls_int = int(cls)
            if cls_int not in trained_model:
                print(f"  Warning: No trainer found for class {cls_int}, skipping...")
                continue
            
            print(f"\nEvaluating class {cls}...")
            # Get the continuous action trainer (DDPG/TD3) for this class
            ddpg_trainer = trained_model[cls_int]
            # Extract the model from the trainer
            cls_model = ddpg_trainer.model if hasattr(ddpg_trainer, 'model') else ddpg_trainer
            
            result = evaluate_class(
                X_test=X_test,
                y_test=y_test,
                trained_model=cls_model,
                make_env_fn=make_env_fn,
                feature_names=feature_names,
                target_class=cls_int,
                n_instances=n_instances_per_class,
                steps_per_episode=steps_per_episode,
                max_features_in_rule=max_features_in_rule,
                random_seed=random_seed,
                eval_on_test_data=eval_on_test_data,
                X_test_unit=X_test_unit,
                X_test_std=X_test_std,
                initial_window=initial_window,
                num_rollouts_per_instance=num_rollouts_per_instance,
            )
            per_class_results[cls_int] = result
            
            print(f"  Avg precision: {result['avg_hard_precision']:.3f}")
            print(f"  Avg coverage: {result['avg_coverage']:.3f}")
            print(f"  Best rule: {result['best_rule']}")
            
            # Show unique rules and their frequencies
            if 'individual_results' in result and len(result['individual_results']) > 0:
                # Extract all rules
                all_rules = [r['rule'] for r in result['individual_results']]
                
                # Count unique rules
                rule_counts = Counter(all_rules)
                n_unique = len(rule_counts)
                
                print(f"  Unique rules: {n_unique} out of {len(all_rules)} instances")
                
                # Show up to 5 most common rules
                if n_unique > 1:
                    print(f"  Top rules:")
                    for rule, count in rule_counts.most_common(5):
                        percentage = (count / len(all_rules)) * 100
                        print(f"    [{count}/{len(all_rules)} ({percentage:.1f}%)] {rule}")
    else:
        # PPO or single continuous action model (DDPG/TD3): Use same model for all classes
        print(f"Evaluating anchors for {n_classes} classes with {n_instances_per_class} instances each")
        
        per_class_results = {}
        for cls in unique_classes:
            print(f"\nEvaluating class {cls}...")
            result = evaluate_class(
                X_test=X_test,
                y_test=y_test,
                trained_model=trained_model,
                make_env_fn=make_env_fn,
                feature_names=feature_names,
                target_class=int(cls),
                n_instances=n_instances_per_class,
                steps_per_episode=steps_per_episode,
                max_features_in_rule=max_features_in_rule,
                random_seed=random_seed,
                eval_on_test_data=eval_on_test_data,
                X_test_unit=X_test_unit,
                X_test_std=X_test_std,
                initial_window=initial_window,
                num_rollouts_per_instance=num_rollouts_per_instance,
            )
            per_class_results[int(cls)] = result
        
        print(f"  Avg precision: {result['avg_hard_precision']:.3f}")
        print(f"  Avg coverage: {result['avg_coverage']:.3f}")
        print(f"  Best rule: {result['best_rule']}")
        
        # Show unique rules and their frequencies
        if 'individual_results' in result and len(result['individual_results']) > 0:
            # Extract all rules
            all_rules = [r['rule'] for r in result['individual_results']]
            
            # Count unique rules
            rule_counts = Counter(all_rules)
            n_unique = len(rule_counts)
            
            print(f"  Unique rules: {n_unique} out of {len(all_rules)} instances")
            
            # Show up to 5 most common rules
            if n_unique > 1:
                print(f"  Top rules:")
                for rule, count in rule_counts.most_common(5):
                    percentage = (count / len(all_rules)) * 100
                    print(f"    [{count}/{len(all_rules)} ({percentage:.1f}%)] {rule}")
    
    # Compute overall statistics
    overall_precision = np.mean([r["avg_hard_precision"] for r in per_class_results.values()])
    overall_coverage = np.mean([r["avg_coverage"] for r in per_class_results.values()])
    
    # Compute overall union coverage if test data evaluation was used
    overall_union_coverage = None
    if eval_on_test_data:
        union_coverages = [r.get("union_coverage") for r in per_class_results.values() if r.get("union_coverage") is not None]
        if union_coverages:
            overall_union_coverage = np.mean(union_coverages)
    
    # Get data source from first result
    data_source = list(per_class_results.values())[0].get("data_source", "training") if per_class_results else "training"
    
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Average precision across all classes: {overall_precision:.3f}")
    print(f"Average coverage across all classes: {overall_coverage:.3f}")
    if overall_union_coverage is not None:
        print(f"Average union coverage across all classes: {overall_union_coverage:.3f}")
    print(f"Metrics computed on: {data_source} data")
    
    return {
        "per_class_results": per_class_results,
        "overall_precision": float(overall_precision),
        "overall_coverage": float(overall_coverage),
        "overall_union_coverage": float(overall_union_coverage) if overall_union_coverage is not None else None,
        "n_classes": n_classes,
        "data_source": data_source,
        "evaluation_type": "instance_level",  # Mark as instance-level evaluation
    }


def evaluate_all_classes_class_level(
    trained_model: Union[PPO, DDPG, TD3, Dict[int, Any]],
    make_env_fn,
    feature_names: List[str],
    target_classes: Optional[List[int]] = None,
    steps_per_episode: int = 100,
    max_features_in_rule: int = 5,
    random_seed: int = 42,
    eval_on_test_data: bool = False,
    X_test_unit: Optional[np.ndarray] = None,
    X_test_std: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    initial_window: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate and generate ONE anchor per class (class-level evaluation for all classes).
    
    This creates a single anchor per class optimized for the entire class, not per-instance.
    This matches how the RL agent was trained (class-level optimization).
    
    Args:
        trained_model: Trained PPO model, DDPG model, or dict of DDPG trainers per class
        make_env_fn: Function that creates AnchorEnv instance (should accept target_class parameter)
        feature_names: List of feature names
        target_classes: List of classes to evaluate (if None, inferred from model)
        steps_per_episode: Maximum rollout steps
        max_features_in_rule: Maximum features in rule
        random_seed: Random seed for consistency
        eval_on_test_data: If True, compute metrics on test data instead of training data
        X_test_unit: Test data in unit space [0,1] (required if eval_on_test_data=True)
        X_test_std: Test data in standardized space (required if eval_on_test_data=True)
        y_test: Test labels (required if eval_on_test_data=True)
        initial_window: Initial window size for anchor box (default: matches training)
    
    Returns:
        Dictionary with per-class class-level results and overall statistics.
        Coverage is global coverage (fraction of all training/test data covered).
    """
    # Determine target classes
    if target_classes is None:
        if isinstance(trained_model, dict):
            target_classes = list(trained_model.keys())
        else:
            # For PPO, we need to infer from make_env_fn or use default
            # Try to get from a temporary environment
            temp_env = make_env_fn()
            if hasattr(temp_env, 'target_class'):
                target_classes = [temp_env.target_class]
            else:
                # Default: assume binary classification (0, 1)
                target_classes = [0, 1]
    
    n_classes = len(target_classes)
    
    # Evaluate each class
    per_class_results = {}
    for cls in target_classes:
        cls_int = int(cls)
        print(f"\nEvaluating class {cls_int} (class-level, one anchor per class)...")
        
        # Create make_env_fn for this specific class
        # Handle both cases: make_env_fn that accepts target_class, or create_anchor_env pattern
        if callable(make_env_fn):
            # Try calling with target_class parameter first
            try:
                # Check if make_env_fn accepts target_class parameter
                import inspect
                sig = inspect.signature(make_env_fn)
                if 'target_class' in sig.parameters or 'target_cls' in sig.parameters:
                    # It accepts target_class/target_cls parameter
                    if 'target_class' in sig.parameters:
                        temp_env = make_env_fn(target_class=cls_int)
                    else:
                        temp_env = make_env_fn(target_cls=cls_int)
                else:
                    # It's a factory that creates envs - try calling it
                    temp_env = make_env_fn()
                    # If it doesn't have target_class set, we need to recreate with target_class
                    if not hasattr(temp_env, 'target_class') or temp_env.target_class != cls_int:
                        # Recreate with target_class if possible
                        if hasattr(make_env_fn, '__name__') and 'create_anchor_env' in str(make_env_fn):
                            # It's create_anchor_env pattern - call with target_cls
                            temp_env = make_env_fn(target_cls=cls_int)
                        else:
                            # Use as-is, but this might not work correctly
                            temp_env = make_env_fn()
            except Exception as e:
                # Fallback: try calling without parameters
                temp_env = make_env_fn()
        else:
            temp_env = make_env_fn
        
        # Create a proper make_env_fn for this class (use closure to capture cls_int)
        def make_env_for_class():
            # Try to call with target_class/target_cls if the function supports it
            try:
                import inspect
                sig = inspect.signature(make_env_fn)
                if 'target_class' in sig.parameters:
                    return make_env_fn(target_class=cls_int)
                elif 'target_cls' in sig.parameters:
                    return make_env_fn(target_cls=cls_int)
                else:
                    # Fallback: call and hope it uses the right class
                    return make_env_fn()
            except:
                # Final fallback
                return make_env_fn()
        
        result = evaluate_class_level(
            trained_model=trained_model,
            make_env_fn=make_env_for_class,
            feature_names=feature_names,
            target_class=cls_int,
            steps_per_episode=steps_per_episode,
            max_features_in_rule=max_features_in_rule,
            random_seed=random_seed,
            eval_on_test_data=eval_on_test_data,
            X_test_unit=X_test_unit,
            X_test_std=X_test_std,
            y_test=y_test,
            initial_window=initial_window,
        )
        per_class_results[cls_int] = result
        
        print(f"  Class-level precision: {result['hard_precision']:.3f}")
        print(f"  Class-level coverage: {result['coverage']:.3f}")
        print(f"  Rule: {result['rule']}")
    
    # Compute overall statistics
    if len(per_class_results) > 0:
        overall_precision = np.mean([r["hard_precision"] for r in per_class_results.values()])
        overall_coverage = np.mean([r["coverage"] for r in per_class_results.values()])
        data_source = list(per_class_results.values())[0].get("data_source", "training")
    else:
        overall_precision = 0.0
        overall_coverage = 0.0
        data_source = "training"
    
    return {
        "per_class_results": per_class_results,
        "overall_precision": float(overall_precision),
        "overall_coverage": float(overall_coverage),
        "n_classes": n_classes,
        "data_source": data_source,
        "evaluation_type": "class_level",  # Mark as class-level evaluation
    }


def load_trained_model(model_path: str, vec_env) -> PPO:
    """
    Load a trained PPO model.
    
    Args:
        model_path: Path to the saved model
        vec_env: Vectorized environment
    
    Returns:
        Loaded PPO model
    """
    model = PPO.load(model_path, env=vec_env)
    print(f"Loaded trained model from {model_path}")
    return model


def prepare_test_data_for_evaluation(
    X_test_scaled: np.ndarray,
    X_min: np.ndarray,
    X_range: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to prepare test data for evaluation.
    
    Converts test data from standardized space to unit space [0, 1] for evaluation.
    
    Args:
        X_test_scaled: Test data in standardized space
        X_min: Min values from training data (for normalization)
        X_range: Range values from training data (for normalization)
    
    Returns:
        Tuple of (X_test_unit, X_test_scaled) ready for evaluation
    """
    X_test_unit = (X_test_scaled - X_min) / X_range
    X_test_unit = np.clip(X_test_unit, 0.0, 1.0).astype(np.float32)
    return X_test_unit, X_test_scaled


def plot_rules_2d(
    eval_results: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    feature_indices: Optional[Tuple[int, int]] = None,
    output_path: str = "./output/visualizations/rules_2d_visualization.png",
    figsize: Tuple[int, int] = (14, 10),
    alpha_anchors: float = 0.3,
    alpha_points: float = 0.6,
    show_instances_used: bool = True,
    X_min: Optional[np.ndarray] = None,
    X_range: Optional[np.ndarray] = None,
) -> str:
    """
    Visualize anchor rules as 2D rectangles.
    
    Creates a 2D plot showing:
    - Data points colored by class
    - Anchor boxes (rules) as rectangles
    - Instances used for evaluation (highlighted)
    
    Args:
        eval_results: Results from evaluate_all_classes() containing rules and anchors
        X_test: Test data (in standardized space, used for plotting)
        y_test: Test labels
        feature_names: List of feature names
        class_names: Optional list of class names (for legend)
        feature_indices: Optional tuple (feat_idx1, feat_idx2) to specify which 2 features to plot
                       If None, auto-selects features that appear most frequently in rules
        output_path: Path to save the plot
        figsize: Figure size (width, height)
        alpha_anchors: Transparency for anchor rectangles (0-1)
        alpha_points: Transparency for data points (0-1)
        show_instances_used: If True, highlight instances used for evaluation
        X_min: Optional min values for converting bounds from unit space to standardized space
        X_range: Optional range values for converting bounds from unit space to standardized space
    
    Returns:
        Path to saved plot file
    """
    
    per_class_results = eval_results.get("per_class_results", {})
    if not per_class_results:
        raise ValueError("eval_results must contain per_class_results with anchors")
    
    # Auto-select features if not specified
    if feature_indices is None:
        # Count feature frequency in rules
        feature_counts = {}
        for cls_result in per_class_results.values():
            if "anchors" in cls_result:
                for anchor in cls_result["anchors"]:
                    lower = np.array(anchor.get("lower_bounds", []))
                    upper = np.array(anchor.get("upper_bounds", []))
                    if len(lower) > 0 and len(upper) > 0:
                        # Find features that were tightened (width < 0.95 of full range)
                        widths = upper - lower
                        tightened = np.where(widths < 0.95)[0]
                        for feat_idx in tightened:
                            feature_counts[feat_idx] = feature_counts.get(feat_idx, 0) + 1
        
        if len(feature_counts) >= 2:
            # Select top 2 most frequently tightened features
            top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:2]
            feature_indices = (top_features[0][0], top_features[1][0])
        elif len(feature_counts) == 1:
            # Use the one tightened feature and the first feature
            feature_indices = (list(feature_counts.keys())[0], 0)
        else:
            # Fallback: use first two features
            feature_indices = (0, min(1, len(feature_names) - 1))
    
    feat_idx1, feat_idx2 = feature_indices
    
    # Validate feature indices
    if feat_idx1 >= len(feature_names) or feat_idx2 >= len(feature_names):
        raise ValueError(f"Feature indices {feature_indices} out of range (max: {len(feature_names)-1})")
    
    # Get feature names for axes
    feat_name1 = feature_names[feat_idx1]
    feat_name2 = feature_names[feat_idx2]
    
    # Create figure with subplots for each class
    # Handle both integer keys (from evaluate_all_classes) and string keys (from JSON)
    unique_classes = []
    for k in per_class_results.keys():
        if isinstance(k, int):
            unique_classes.append(k)
        elif isinstance(k, str):
            # Handle "class_0" format
            if k.startswith("class_"):
                try:
                    unique_classes.append(int(k.split('_')[1]))
                except (ValueError, IndexError):
                    # Try to extract number from string
                    try:
                        unique_classes.append(int(k.replace("class_", "")))
                    except ValueError:
                        pass
            else:
                # Try to convert directly
                try:
                    unique_classes.append(int(k))
                except ValueError:
                    pass
    
    unique_classes = sorted(unique_classes)
    n_classes = len(unique_classes)
    
    if n_classes == 0:
        raise ValueError("No valid class keys found in per_class_results")
    
    # Determine grid layout
    if n_classes <= 2:
        n_rows, n_cols = 1, n_classes
    elif n_classes <= 4:
        n_rows, n_cols = 2, 2
    elif n_classes <= 6:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = 3, 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    # Color map for classes
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_classes))))
    
    # Plot each class
    for plot_idx, cls_int in enumerate(unique_classes):
        if plot_idx >= len(axes):
            break
        
        ax = axes[plot_idx]
        
        # Try to get class result - handle both integer and string keys
        cls_result = None
        if cls_int in per_class_results:
            cls_result = per_class_results[cls_int]
        elif f"class_{cls_int}" in per_class_results:
            cls_result = per_class_results[f"class_{cls_int}"]
        else:
            ax.text(0.5, 0.5, f"No data for class {cls_int}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Class {cls_int}")
            continue
        cls_name = class_names[cls_int] if class_names and cls_int < len(class_names) else f"Class {cls_int}"
        
        # Get anchors for this class
        anchors = cls_result.get("anchors", [])
        instance_indices_used = cls_result.get("instance_indices_used", [])
        
        # Plot all data points (colored by class)
        for other_cls in unique_classes:
            mask = y_test == other_cls
            if np.any(mask):
                ax.scatter(
                    X_test[mask, feat_idx1],
                    X_test[mask, feat_idx2],
                    c=[colors[other_cls % len(colors)]],
                    alpha=alpha_points * 0.5 if other_cls != cls_int else alpha_points,
                    s=20,
                    label=f"Class {other_cls}" if other_cls == cls_int else None,
                    edgecolors='none',
                    zorder=1
                )
        
        # Highlight instances used for evaluation
        if show_instances_used and len(instance_indices_used) > 0:
            used_mask = np.zeros(len(X_test), dtype=bool)
            for idx in instance_indices_used:
                if 0 <= idx < len(X_test):
                    used_mask[idx] = True
            
            if np.any(used_mask):
                ax.scatter(
                    X_test[used_mask, feat_idx1],
                    X_test[used_mask, feat_idx2],
                    c='red',
                    marker='x',
                    s=100,
                    linewidths=2,
                    label='Instances evaluated',
                    zorder=4,
                    alpha=0.8
                )
        
        # Draw anchor rectangles
        anchor_colors = plt.cm.Set3(np.linspace(0, 1, max(len(anchors), 1)))
        for anchor_idx, anchor in enumerate(anchors):
            lower = np.array(anchor.get("lower_bounds", []))
            upper = np.array(anchor.get("upper_bounds", []))
            
            if len(lower) == 0 or len(upper) == 0:
                continue
            
            # Get bounds for the two features we're plotting
            # Bounds are stored in unit space [0, 1], but X_test is in standardized space
            # We need to convert bounds from unit space to standardized space
            # Conversion: standardized = unit * X_range + X_min
            # But we don't have X_min and X_range here, so we'll use the data range
            # For visualization purposes, we can approximate using X_test range
            
            # Get data range for conversion
            # Use provided X_min/X_range if available, otherwise use X_test range
            if X_min is not None and X_range is not None:
                X_min_feat1 = X_min[feat_idx1]
                X_range_feat1 = X_range[feat_idx1]
                X_min_feat2 = X_min[feat_idx2]
                X_range_feat2 = X_range[feat_idx2]
            else:
                # Fallback: use X_test range (approximation)
                X_min_feat1 = X_test[:, feat_idx1].min()
                X_max_feat1 = X_test[:, feat_idx1].max()
                X_range_feat1 = X_max_feat1 - X_min_feat1
                
                X_min_feat2 = X_test[:, feat_idx2].min()
                X_max_feat2 = X_test[:, feat_idx2].max()
                X_range_feat2 = X_max_feat2 - X_min_feat2
            
            # Convert bounds from unit space [0,1] to standardized space
            if feat_idx1 < len(lower) and feat_idx1 < len(upper):
                # Bounds are in [0,1], convert to standardized space
                lower_feat1 = X_min_feat1 + lower[feat_idx1] * X_range_feat1
                upper_feat1 = X_min_feat1 + upper[feat_idx1] * X_range_feat1
            else:
                lower_feat1 = X_test[:, feat_idx1].min()
                upper_feat1 = X_test[:, feat_idx1].max()
            
            if feat_idx2 < len(lower) and feat_idx2 < len(upper):
                lower_feat2 = X_min_feat2 + lower[feat_idx2] * X_range_feat2
                upper_feat2 = X_min_feat2 + upper[feat_idx2] * X_range_feat2
            else:
                lower_feat2 = X_test[:, feat_idx2].min()
                upper_feat2 = X_test[:, feat_idx2].max()
            
            width1 = upper_feat1 - lower_feat1
            width2 = upper_feat2 - lower_feat2
            
            # Draw rectangle
            rect = Rectangle(
                (lower_feat1, lower_feat2),
                width1,
                width2,
                linewidth=2,
                edgecolor=colors[cls_int % len(colors)],
                facecolor=colors[cls_int % len(colors)],
                alpha=alpha_anchors,
                zorder=2
            )
            ax.add_patch(rect)
            
            # Add text label with precision/coverage if space allows
            if width1 > 0 and width2 > 0:
                prec = anchor.get("precision", 0.0)
                cov = anchor.get("coverage", 0.0)
                if prec > 0 or cov > 0:
                    ax.text(
                        lower_feat1 + width1/2,
                        lower_feat2 + width2/2,
                        f"P:{prec:.2f}\nC:{cov:.2f}",
                        ha='center',
                        va='center',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                        zorder=3
                    )
        
        ax.set_xlabel(feat_name1, fontsize=10)
        ax.set_ylabel(feat_name2, fontsize=10)
        ax.set_title(f"{cls_name}\n({len(anchors)} anchors)", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_classes, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(
        f'2D Visualization of Anchor Rules\nFeatures: {feat_name1} vs {feat_name2}',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_rules_2d_from_json(
    json_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    feature_indices: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
    **plot_kwargs
) -> str:
    """
    Visualize anchor rules from saved JSON file.
    
    Convenience function to load rules from JSON and create 2D visualization.
    
    Args:
        json_path: Path to metrics_and_rules.json file
        X_test: Test data (in standardized space)
        y_test: Test labels
        feature_names: List of feature names
        class_names: Optional list of class names
        feature_indices: Optional tuple (feat_idx1, feat_idx2) to specify which 2 features to plot
        output_path: Optional output path (defaults to same directory as JSON with _2d_plot suffix)
        **plot_kwargs: Additional arguments passed to plot_rules_2d()
    
    Returns:
        Path to saved plot file
    """
    import json
    
    # Load JSON
    with open(json_path, 'r') as f:
        metrics_data = json.load(f)
    
    # Convert to eval_results format
    # JSON has string keys like "class_0", need to convert to integer keys
    eval_results = {
        "per_class_results": {}
    }
    
    for cls_key, cls_data in metrics_data.get("per_class_results", {}).items():
        # Extract class integer from key (handle both "class_0" and integer keys)
        if isinstance(cls_key, int):
            cls_int = cls_key
        elif isinstance(cls_key, str) and cls_key.startswith("class_"):
            try:
                cls_int = int(cls_key.split('_')[1])
            except (ValueError, IndexError):
                try:
                    cls_int = int(cls_key.replace("class_", ""))
                except ValueError:
                    continue  # Skip invalid keys
        else:
            try:
                cls_int = int(cls_key)
            except ValueError:
                continue  # Skip invalid keys
        
        eval_results["per_class_results"][cls_int] = cls_data
    
    # Set default output path
    if output_path is None:
        import os
        base_dir = os.path.dirname(json_path)
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_2d_plot.png")
    
    # Create plot
    return plot_rules_2d(
        eval_results=eval_results,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        class_names=class_names,
        feature_indices=feature_indices,
        output_path=output_path,
        **plot_kwargs
    )

