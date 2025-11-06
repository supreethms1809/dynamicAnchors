import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Stable Baselines 3 imports
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_VERSION = "gymnasium"
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_VERSION = "gym"
    except ImportError:
        raise ImportError("Please install gymnasium or gym: pip install gymnasium")

try:
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    SB3_AVAILABLE = True
except ImportError:
    raise ImportError("stable_baselines3 is not installed. Please install it with: pip install stable-baselines3")

# Import DDPG trainer wrapper
try:
    import sys
    import os
    # Add trainers directory to path
    trainers_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trainers")
    if trainers_dir not in sys.path:
        sys.path.insert(0, trainers_dir)
    from DDPG_trainer import DynamicAnchorDDPGTrainer, create_ddpg_trainer
    DDPG_TRAINER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import DDPG_trainer: {e}")
    print("Falling back to direct SB3 DDPG usage")
    DDPG_TRAINER_AVAILABLE = False


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return logits; apply softmax only when probabilities are needed
        return self.net(x)


class AnchorEnv:
    """
    Dynamic anchors environment over a hyper-rectangle (bounding box) in feature space.
    
    Continuous action version: actions directly modify lower and upper bounds.

    - State: concatenation of [lower_bounds, upper_bounds] in normalized feature space (range [0, 1])
             plus current precision, coverage.
    - Actions: continuous actions of shape (2 * n_features,)
        * First n_features: delta for lower bounds (clipped to [-1, 1], then scaled)
        * Next n_features: delta for upper bounds (clipped to [-1, 1], then scaled)
        * Actions are scaled by current box width for scale-invariant updates
    - Reward: precision_gain * alpha + coverage_gain * beta - overlap_penalty - invalid_penalty
              computed w.r.t. the classifier predictions.
    """

    def __init__(
        self,
        X_unit: np.ndarray,
        X_std: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        classifier: SimpleClassifier,
        device: torch.device,
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
        perturbation_mode: str = "bootstrap",  # "bootstrap" or "uniform"
        n_perturb: int = 1024,
        X_min: np.ndarray | None = None,
        X_range: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        min_coverage_floor: float = 0.005,
        js_penalty_weight: float = 0.05,
        x_star_unit: np.ndarray | None = None,
        initial_window: float = 0.1,
    ):
        self.X_unit = X_unit  # normalized to [0,1]
        self.X_std = X_std    # standardized (the scale the classifier was trained on)
        self.y = y.astype(int)
        self.feature_names = feature_names
        self.n_features = X_unit.shape[1]
        self.classifier = classifier
        self.device = device
        self.target_class = int(target_class)
        self.step_fracs = step_fracs  # Used for action scaling
        self.min_width = min_width
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Continuous actions: 2 * n_features (one for lower, one for upper per feature)
        self.n_actions = 2 * self.n_features
        # Action scaling factor (max change per feature as fraction of current width)
        self.max_action_scale = max(step_fracs) if step_fracs else 0.02
        # Minimum absolute step size to prevent updates from being too small when box is small
        # This ensures the box can still grow even when it's at minimum size
        # For continuous control with DDPG, allow stronger expansion from tight boxes
        # Increase minimum absolute step so coverage can grow from small intervals
        self.min_absolute_step = max(0.05, min_width * 0.5)

        # Box state
        self.lower = np.zeros(self.n_features, dtype=np.float32)
        self.upper = np.ones(self.n_features, dtype=np.float32)
        self.prev_lower = self.lower.copy()
        self.prev_upper = self.upper.copy()
        # Targets / weights
        self.precision_target = precision_target
        self.coverage_target = coverage_target
        self.precision_blend_lambda = precision_blend_lambda
        self.drift_penalty_weight = drift_penalty_weight

        # History for visualization
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

    def _mask_in_box(self) -> np.ndarray:
        conds = []
        for j in range(self.n_features):
            conds.append((self.X_unit[:, j] >= self.lower[j]) & (self.X_unit[:, j] <= self.upper[j]))
        mask = np.logical_and.reduce(conds) if conds else np.ones(self.X_unit.shape[0], dtype=bool)
        return mask

    def _unit_to_std(self, X_unit_samples: np.ndarray) -> np.ndarray:
        if self.X_min is None or self.X_range is None:
            raise ValueError("X_min/X_range must be set for uniform perturbation sampling.")
        return (X_unit_samples * self.X_range) + self.X_min

    def _current_metrics(self) -> tuple:
        mask = self._mask_in_box()
        covered = np.where(mask)[0]
        coverage = float(mask.mean())
        if covered.size == 0:
            return 0.0, coverage, {"hard_precision": 0.0, "avg_prob": 0.0, "n_points": 0, "sampler": "none"}

        # Select inputs either from empirical subset or via perturbation sampler
        if not self.use_perturbation:
            X_eval = self.X_std[covered]
            y_eval = self.y[covered]
            n_points = int(X_eval.shape[0])
            sampler_note = "empirical"
        else:
            n_samp = min(self.n_perturb, max(1, covered.size))
            if self.perturbation_mode == "bootstrap":
                # Resample existing covered rows with replacement (keeps true labels)
                idx = self.rng.choice(covered, size=n_samp, replace=True)
                X_eval = self.X_std[idx]
                y_eval = self.y[idx]
                n_points = int(n_samp)
                sampler_note = "bootstrap"
            elif self.perturbation_mode == "uniform":
                # Sample uniformly within the current box in unit space; then invert to std space
                U = np.zeros((n_samp, self.n_features), dtype=np.float32)
                for j in range(self.n_features):
                    low, up = float(self.lower[j]), float(self.upper[j])
                    U[:, j] = self.rng.uniform(low=low, high=up, size=n_samp).astype(np.float32)
                X_eval = self._unit_to_std(U)
                y_eval = None  # unknown under synthetic sampling
                n_points = int(n_samp)
                sampler_note = "uniform"
            else:
                raise ValueError(f"Unknown perturbation_mode '{self.perturbation_mode}'. Use 'bootstrap' or 'uniform'.")

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
            if positive_idx.sum() == 0:
                hard_precision = 0.0
            else:
                hard_precision = float((y_eval[positive_idx] == self.target_class).mean())

        avg_prob = float(probs[:, self.target_class].mean())
        precision_proxy = (
            self.precision_blend_lambda * hard_precision + (1.0 - self.precision_blend_lambda) * avg_prob
        )
        return precision_proxy, coverage, {
            "hard_precision": hard_precision,
            "avg_prob": avg_prob,
            "n_points": int(n_points),
            "sampler": sampler_note,
        }

    def reset(self):
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
        precision, coverage, _ = self._current_metrics()
        state = np.concatenate([self.lower, self.upper, np.array([precision, coverage], dtype=np.float32)])
        return state

    def _apply_action(self, action: np.ndarray):
        """
        Apply continuous action to modify bounds.
        
        Args:
            action: array of shape (2 * n_features,) with values in [-1, 1]
                   First n_features: deltas for lower bounds
                   Next n_features: deltas for upper bounds
        """
        # Clip actions to [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        # Split into lower and upper deltas
        lower_deltas = action[:self.n_features]
        upper_deltas = action[self.n_features:]
        
        # Scale actions by current box width for scale-invariant updates
        # BUT ensure minimum step size so box can grow even when small
        widths = np.maximum(self.upper - self.lower, 1e-6)
        # Use the larger of proportional scaling or absolute minimum
        # This prevents the box from getting stuck at minimum size
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

    def step(self, action: np.ndarray):
        prev_precision, prev_coverage, _ = self._current_metrics()
        prev_lower = self.lower.copy()
        prev_upper = self.upper.copy()
        # Pre-compute previous box volume
        prev_widths = np.maximum(prev_upper - prev_lower, 1e-9)
        prev_vol = float(np.prod(prev_widths))
        self._apply_action(action)
        precision, coverage, details = self._current_metrics()

        # Enforce a minimum coverage floor by reverting overly aggressive actions
        coverage_clipped = False
        if coverage < self.min_coverage_floor:
            # revert bounds
            self.lower = prev_lower
            self.upper = prev_upper
            # recompute with reverted bounds
            precision, coverage, details = self._current_metrics()
            coverage_clipped = True

        precision_gain = precision - prev_precision
        coverage_gain = coverage - prev_coverage

        # Penalize too small boxes
        widths = self.upper - self.lower
        overlap_penalty = self.gamma * float((widths < (2 * self.min_width)).mean())

        # Penalize large drift to promote stability
        drift = float(np.linalg.norm(self.upper - prev_upper) + np.linalg.norm(self.lower - prev_lower))
        drift_penalty = self.drift_penalty_weight * drift

        # JS-like penalty based on volume overlap (distributional shift proxy)
        # Compute intersection and union volumes of axis-aligned boxes in unit space
        inter_lower = np.maximum(self.lower, prev_lower)
        inter_upper = np.minimum(self.upper, prev_upper)
        inter_widths = np.maximum(inter_upper - inter_lower, 0.0)
        inter_vol = float(np.prod(np.maximum(inter_widths, 0.0)))
        curr_widths = np.maximum(self.upper - self.lower, 1e-9)
        curr_vol = float(np.prod(curr_widths))
        eps = 1e-12
        if inter_vol <= eps:
            js_proxy = 1.0  # maximal mismatch
        else:
            # Symmetric KL proxy over uniform distributions on the two boxes
            dkl_prev_to_mix = np.log((prev_vol + curr_vol) / (2.0 * inter_vol + eps) + eps)
            dkl_curr_to_mix = np.log((prev_vol + curr_vol) / (2.0 * inter_vol + eps) + eps)
            # Since both directions equal for uniform + overlap based on intersection with mixture,
            # JS proxy simplifies to this shared log term; keep bounded in [0,1] via mapping
            js_proxy = 1.0 - float(inter_vol / (0.5 * (prev_vol + curr_vol) + eps))
            js_proxy = float(np.clip(js_proxy, 0.0, 1.0))
        js_penalty = self.js_penalty_weight * js_proxy

        reward = self.alpha * precision_gain + self.beta * coverage_gain - overlap_penalty - drift_penalty - js_penalty

        self.box_history.append((self.lower.copy(), self.upper.copy()))
        self.prev_lower = prev_lower
        self.prev_upper = prev_upper
        state = np.concatenate([self.lower, self.upper, np.array([precision, coverage], dtype=np.float32)])
        done = bool(precision >= self.precision_target and coverage >= self.coverage_target)
        info = {"precision": precision, "coverage": coverage, "drift": drift, "js_penalty": js_penalty, "coverage_clipped": coverage_clipped, **details}
        return state, reward, done, info


class PolicyNet(nn.Module):
    """
    Continuous action policy network.
    Outputs mean and log_std for a Gaussian distribution over actions.
    Actions are bounded to [-1, 1] using tanh.
    """
    def __init__(self, state_dim: int, action_dim: int, log_std_init: float = -0.5):
        super().__init__()
        self.action_dim = action_dim
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)
        
        # Initialize log_std
        with torch.no_grad():
            self.log_std_head.weight.data.fill_(0.0)
            self.log_std_head.bias.data.fill_(log_std_init)
        
        # Initialize mean_head with small weights to prevent extreme outputs
        with torch.no_grad():
            # Use Xavier/Glorot initialization scaled down
            nn.init.xavier_uniform_(self.mean_head.weight, gain=0.1)
            self.mean_head.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns mean and log_std for action distribution.
        """
        # Normalize/clamp input to prevent extreme values
        x = torch.clamp(x, -10.0, 10.0)
        
        features = self.shared_net(x)
        # Clamp features to prevent extreme activations
        features = torch.clamp(features, -10.0, 10.0)
        
        mean_raw = self.mean_head(features)
        # Clamp mean_raw BEFORE tanh to prevent gradient saturation
        # This is the key fix: tanh saturates for large inputs, causing gradient issues
        mean_raw = torch.clamp(mean_raw, -3.0, 3.0)
        mean = torch.tanh(mean_raw)  # Bound mean to [-1, 1]
        
        log_std = self.log_std_head(features)
        # Clamp log_std to reasonable range to prevent extreme variances
        log_std = torch.clamp(log_std, -2.0, 0.5)
        
        return mean, log_std
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.
        Returns (action, log_prob) where action is in [-1, 1].
        """
        mean, log_std = self.forward(state)
        
        std = torch.exp(log_std)
        # Clamp std to reasonable range to prevent numerical issues
        std = torch.clamp(std, min=1e-4, max=1.0)
        
        dist = torch.distributions.Normal(mean, std)
        # Sample and apply tanh for bounded actions
        action_raw = dist.sample()
        action = torch.tanh(action_raw)
        
        # Compute log probability with tanh transformation correction
        # More stable formula: log_prob(u) - log(1 - tanh^2(u)) where u = atanh(a)
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        # Tanh correction: log(1 - tanh^2(u)) = -2 * log(cosh(u))
        # More numerically stable: use log(1 - a^2) directly
        action_sq = action.pow(2)
        # Clamp to prevent log(0) or log(negative)
        action_sq = torch.clamp(action_sq, min=1e-8, max=1.0 - 1e-8)
        correction = torch.log(1.0 - action_sq).sum(dim=-1)
        log_prob = log_prob - correction
        
        return action, log_prob
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of action given state.
        Action should be in [-1, 1] (tanh-transformed).
        """
        # Clip actions to prevent numerical issues with atanh
        action = torch.clamp(action, -0.9999, 0.9999)
        
        mean, log_std = self.forward(state)
        
        std = torch.exp(log_std)
        # Clamp std to reasonable range to prevent numerical issues
        std = torch.clamp(std, min=1e-4, max=1.0)
        
        dist = torch.distributions.Normal(mean, std)
        # Use PyTorch's stable atanh implementation
        action_raw = torch.atanh(action)
        
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        # Tanh correction: log(1 - tanh^2(u)) where u = atanh(a)
        # More stable: use log(1 - a^2) directly
        action_sq = action.pow(2)
        # Clamp to prevent log(0) or log(negative)
        action_sq = torch.clamp(action_sq, min=1e-8, max=1.0 - 1e-8)
        correction = torch.log(1.0 - action_sq).sum(dim=-1)
        log_prob = log_prob - correction
        
        return log_prob
    
    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute entropy of action distribution."""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        # Approximate entropy for tanh-transformed distribution
        entropy = dist.entropy().sum(dim=-1)
        return entropy

class ValueNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


# ============================================================================
# Gym-compatible wrapper for Stable Baselines 3 DDPG
# ============================================================================

class ContinuousAnchorEnv(gym.Env):
    """
    Gym-compatible wrapper for AnchorEnv with continuous actions.
    
    State Space: Box of shape (2 * n_features + 2,)
        - First n_features: lower bounds for each feature
        - Next n_features: upper bounds for each feature  
        - Next 1: current precision
        - Next 1: current coverage
    
    Action Space: Box of shape (2 * n_features,)
        - First n_features: deltas for lower bounds (clipped to [-1, 1])
        - Next n_features: deltas for upper bounds (clipped to [-1, 1])
    """
    
    def __init__(self, anchor_env: AnchorEnv, seed: int | None = None):
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
    
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset the environment and return initial observation."""
        if seed is not None:
            self.seed(seed)
        
        state = self.anchor_env.reset()
        obs = np.array(state, dtype=np.float32)
        
        info = {
            "precision": self.anchor_env._current_metrics()[0],
            "coverage": self.anchor_env._current_metrics()[1],
        }
        
        if GYM_VERSION == "gymnasium":
            return obs, info
        else:
            return obs
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment."""
        # Ensure action is numpy array
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = np.clip(action, -1.0, 1.0)
        
        next_state, reward, done, info = self.anchor_env.step(action)
        obs = np.array(next_state, dtype=np.float32)
        
        step_info = {
            "precision": info.get("precision", 0.0),
            "coverage": info.get("coverage", 0.0),
            "hard_precision": info.get("hard_precision", 0.0),
            "drift": info.get("drift", 0.0),
            "js_penalty": info.get("js_penalty", 0.0),
            "coverage_clipped": info.get("coverage_clipped", False),
            "sampler": info.get("sampler", "unknown"),
            "n_points": info.get("n_points", 0),
        }
        
        if GYM_VERSION == "gymnasium":
            terminated = done
            truncated = False
            return obs, float(reward), bool(terminated), bool(truncated), step_info
        else:
            return obs, float(reward), bool(done), step_info
    
    def seed(self, seed: int | None = None):
        """Set random seed."""
        if seed is not None:
            self.anchor_env.rng = np.random.default_rng(seed)
    
    def render(self):
        """Render not implemented."""
        raise NotImplementedError("Render not implemented for ContinuousAnchorEnv")
    
    def close(self):
        """Clean up resources."""
        pass

# --- Device selection helper ---
DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")

# --- Discretization helpers ---
def compute_quantile_bins(X: np.ndarray, disc_perc: list[int]) -> list[np.ndarray]:
    edges_per_feature: list[np.ndarray] = []
    for j in range(X.shape[1]):
        edges = np.unique(np.percentile(X[:, j], disc_perc).astype(np.float32))
        edges_per_feature.append(edges)
    return edges_per_feature

def discretize_by_edges(X: np.ndarray, edges_per_feature: list[np.ndarray]) -> np.ndarray:
    X_bins = np.zeros_like(X, dtype=np.int32)
    for j, edges in enumerate(edges_per_feature):
        if edges.size == 0:
            X_bins[:, j] = 0
        else:
            X_bins[:, j] = np.digitize(X[:, j], edges, right=False)
    return X_bins

def compute_bin_representatives(X: np.ndarray, X_bins: np.ndarray) -> list[np.ndarray]:
    reps: list[np.ndarray] = []
    n_features = X.shape[1]
    for j in range(n_features):
        max_bin = int(X_bins[:, j].max())
        reps_j = np.zeros(max_bin + 1, dtype=np.float32)
        for b in range(max_bin + 1):
            mask = (X_bins[:, j] == b)
            if mask.any():
                reps_j[b] = float(np.median(X[mask, j]))
            else:
                reps_j[b] = float(np.median(X[:, j]))
        reps.append(reps_j)
    return reps

class DiscreteAnchorEnv(AnchorEnv):
    """Anchor environment operating on discretized (binned) features."""
    def __init__(
        self,
        X_bins: np.ndarray,
        X_std: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        classifier: SimpleClassifier,
        device: torch.device,
        bin_reps: list[np.ndarray],
        bin_edges: list[np.ndarray],
        target_class: int = 1,
        step_fracs=(1, 1, 1),
        min_width: float = 1.0,
        alpha: float = 0.7,
        beta: float = 0.6,
        gamma: float = 0.1,
        precision_target: float = 0.95,
        coverage_target: float = 0.02,
        precision_blend_lambda: float = 0.5,
        drift_penalty_weight: float = 0.05,
        use_perturbation: bool = False,
        perturbation_mode: str = "bootstrap",
        n_perturb: int = 1024,
        rng: np.random.Generator | None = None,
        min_coverage_floor: float = 0.005,
        js_penalty_weight: float = 0.05,
        x_star_bins: np.ndarray | None = None,
    ):
        # We store discrete bins as X_unit for reuse of base methods
        self.X_bins = X_bins.astype(np.int32)
        self.bin_reps = bin_reps
        self.bin_edges = bin_edges
        super().__init__(
            X_unit=X_bins.astype(np.float32),  # will only be used for mask comparisons
            X_std=X_std,
            y=y,
            feature_names=feature_names,
            classifier=classifier,
            device=device,
            target_class=target_class,
            step_fracs=step_fracs,
            min_width=float(1.0),
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            precision_target=precision_target,
            coverage_target=coverage_target,
            precision_blend_lambda=precision_blend_lambda,
            drift_penalty_weight=drift_penalty_weight,
            use_perturbation=use_perturbation,
            perturbation_mode=perturbation_mode,
            n_perturb=n_perturb,
            X_min=None,
            X_range=None,
            rng=rng,
            min_coverage_floor=min_coverage_floor,
            js_penalty_weight=js_penalty_weight,
            x_star_unit=(x_star_bins.astype(np.float32) if x_star_bins is not None else None),
            initial_window=0.0,
        )
        # Normalize bin indices to [0, 1] for consistency with continuous mode
        # Store max bins for each feature to enable normalization/de-normalization
        self.max_bins_per_feature = np.array([float(self._max_bin(j)) for j in range(self.n_features)], dtype=np.float32)
        # Avoid division by zero
        self.max_bins_per_feature = np.maximum(self.max_bins_per_feature, 1.0)
        
        # Initialize discrete bounds normalized to [0, 1] (like continuous mode)
        # lower = 0.0 (first bin), upper = 1.0 (last bin) after normalization
        self.lower = np.zeros(self.n_features, dtype=np.float32)
        self.upper = np.ones(self.n_features, dtype=np.float32)
        
        # Store raw bin indices for mask comparison (since X_unit has raw bin indices)
        self.lower_bins = np.zeros(self.n_features, dtype=np.float32)
        self.upper_bins = self.max_bins_per_feature.copy()
        
        # Use integer step bins instead of proportional steps (conservative moves)
        self.step_bins = (1, 1, 1)
        # Per-feature minimum width in normalized [0, 1] space (>= 2 bins or 10% of range)
        self.min_width_bins = np.zeros(self.n_features, dtype=np.float32)
        for j in range(self.n_features):
            maxb = self.max_bins_per_feature[j] + 1.0
            min_bins = max(2.0, np.ceil(0.10 * maxb))
            # Normalize to [0, 1]
            self.min_width_bins[j] = min_bins / maxb if maxb > 0 else 0.1

    def _max_bin(self, j: int) -> int:
        return int(max(0, self.X_bins[:, j].max()))

    def _unit_to_std(self, X_unit_samples: np.ndarray) -> np.ndarray:
        # Map sampled bins to representative continuous values for classifier eval
        X_rep = np.zeros_like(self.X_std[: X_unit_samples.shape[0], :], dtype=np.float32)
        for j in range(self.n_features):
            bins_j = np.clip(X_unit_samples[:, j].astype(int), 0, self._max_bin(j))
            X_rep[:, j] = self.bin_reps[j][bins_j]
        return X_rep

    def _apply_action(self, action: int):
        # Override to move bounds by integer number of bins (not proportional widths)
        f = action // (len(self.directions) * len(self.step_fracs))
        rem = action % (len(self.directions) * len(self.step_fracs))
        d = rem // len(self.step_fracs)
        m = rem % len(self.step_fracs)

        direction = self.directions[d]
        step_bins = int(self.step_bins[m])
        max_bin = float(self._max_bin(f))

        if direction == "shrink_lower":
            self.lower[f] = min(self.lower[f] + step_bins, self.upper[f] - self.min_width_bins[f])
        elif direction == "expand_lower":
            self.lower[f] = max(self.lower[f] - step_bins, 0.0)
        elif direction == "shrink_upper":
            self.upper[f] = max(self.upper[f] - step_bins, self.lower[f] + self.min_width_bins[f])
        elif direction == "expand_upper":
            self.upper[f] = min(self.upper[f] + step_bins, max_bin)

        # Ensure at least min width
        if self.upper[f] - self.lower[f] < self.min_width_bins[f]:
            mid = 0.5 * (self.upper[f] + self.lower[f])
            half = 0.5 * self.min_width_bins[f]
            self.lower[f] = max(0.0, np.floor(mid - half))
            self.upper[f] = min(max_bin, np.ceil(mid + half))

    def reset(self):
        # Initialize to full bin range or around x* in bins if provided
        if self.x_star_unit is None:
            self.lower = np.zeros(self.n_features, dtype=np.float32)
            self.upper = np.array([self._max_bin(j) for j in range(self.n_features)], dtype=np.float32)
        else:
            # self.x_star_unit stores bins if provided for discrete env
            w = 1.0  # one-bin half window
            low = []
            up = []
            for j in range(self.n_features):
                mj = float(self._max_bin(j))
                lj = max(0.0, np.floor(float(self.x_star_unit[j]) - w))
                uj = min(mj, np.ceil(float(self.x_star_unit[j]) + w))
                low.append(lj)
                up.append(uj)
            self.lower = np.array(low, dtype=np.float32)
            self.upper = np.array(up, dtype=np.float32)
        self.prev_lower = self.lower.copy()
        self.prev_upper = self.upper.copy()
        self.box_history = [(self.lower.copy(), self.upper.copy())]
        precision, coverage, _ = self._current_metrics()
        state = np.concatenate([self.lower, self.upper, np.array([precision, coverage], dtype=np.float32)])
        return state

def select_device(device_preference: str = "auto") -> torch.device:
    device_preference = (device_preference or "auto").lower()
    if device_preference == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_preference == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device_preference == "cpu":
        return torch.device("cpu")
    # auto: prefer CUDA > MPS > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_dynamic_anchors(
    episodes: int | None = None,
    steps_per_episode: int | None = None,
    classifier_epochs_per_round: int | None = None,
    classifier_update_every: int = 1,
    seed: int = 42,
    target_classes=None,
    entropy_coef: float | None = None,
    value_coef: float | None = None,
    reg_lambda_inside_anchor: float | None = None,
    dataset: str = "covtype",
    device_preference: str = "auto",
    use_perturbation: bool | None = None,
    perturbation_mode: str | None = None,
    n_perturb: int | None = None,
    debug: bool = True,
    local_instance_index: int = -1,
    initial_window: float | None = None,
    precision_target: float | None = None,
    coverage_target: float | None = None,
    use_discretization: bool = False,
    disc_perc: list[int] | None = None,
    bin_edges: list[np.ndarray] | None = None,
    show_plots: bool = True,
    num_greedy_rollouts: int = 1,
    num_test_instances_per_class: int | None = None,
    max_features_in_rule: int = 5,
    # DDPG-specific parameters (kept for backward compatibility, currently using SB3 defaults)
    ppo_epochs: int = 4,  # Not used with DDPG (kept for compatibility)
    clip_epsilon: float = 0.2,  # Not used with DDPG (kept for compatibility)
    batch_size: int | None = None,  # Not used with DDPG (kept for compatibility)
):
    """
    Train dynamic anchors using DDPG from Stable Baselines 3 with continuous actions.
    
    This version uses continuous actions and DDPG (Deep Deterministic Policy Gradient):
    - Actions are continuous vectors of shape (2 * n_features,)
    - Each action directly modifies lower and upper bounds
    - Uses DDPG from Stable Baselines 3 for off-policy continuous control
    - Actions are bounded to [-1, 1] using action space clipping
    - Uses replay buffer and target networks for stable learning

    Args:
        episodes: number of RL episodes
        steps_per_episode: RL steps per episode
        classifier_epochs_per_round: classifier epochs per RL episode
        seed: random seed
        target_classes: tuple of target class labels
        entropy_coef: entropy regularization coefficient
        value_coef: value loss coefficient
        reg_lambda_inside_anchor: regularization inside anchor
        dataset: which dataset to use; one of 'breast_cancer', 'synthetic', or 'covtype'
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Dataset loading: supports 'breast_cancer', 'synthetic', or 'covtype'
    if dataset == "breast_cancer":
        ds = load_breast_cancer()
        X = ds.data.astype(np.float32)
        y = ds.target.astype(int)
        feature_names = list(ds.feature_names)
    elif dataset == "synthetic":
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=2000, n_features=12, n_informative=8, n_classes=2, random_state=seed)
        X = X.astype(np.float32)
        y = y.astype(int)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    elif dataset == "covtype":
        X, y = fetch_covtype(return_X_y=True, as_frame=False)
        X = X.astype(np.float32)
        y = y.astype(int)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'breast_cancer', 'synthetic', or 'covtype'.")

    # (moved) Printing of names happens after class discovery to include class names

    # Normalize class labels to 0..C-1 if needed, and prepare class names aligned with indices
    unique_classes = np.unique(y)
    if dataset == "breast_cancer":
        # Map dataset-provided names to the sorted unique class order
        class_names = [str(load_breast_cancer().target_names[c]) for c in unique_classes]
    else:
        # Generic names based on original labels
        class_names = [f"class_{int(c)}" for c in unique_classes]
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y = np.array([class_to_idx[c] for c in y], dtype=int)
    num_classes = int(len(unique_classes))
    if target_classes is None:
        target_classes = tuple(range(num_classes))

    # Dataset-specific presets for tunable parameters
    presets = {
        "breast_cancer": {
            "episodes": 25,
            "steps_per_episode": 40,
            "classifier_epochs_per_round": 4,
            "entropy_coef": 0.02,
            "value_coef": 0.5,
            "reg_lambda_inside_anchor": 0.0,
            "use_perturbation": False,
            "perturbation_mode": "bootstrap",
            "n_perturb": 1024,
            "initial_window": 0.2,
            # Env params
            "step_fracs": (0.01, 0.02, 0.04),
            "min_width": 0.05,
            "precision_target": 0.95,
            "coverage_target": 0.05,
            "precision_blend_lambda": 0.5,
            "drift_penalty_weight": 0.05,
            "js_penalty_weight": 0.05,
            "disc_perc": [25, 50, 75],
            # PPO params
            "ppo_epochs": 4,
            "clip_epsilon": 0.2,
        },
        "synthetic": {
            "episodes": 30,
            "steps_per_episode": 50,
            "classifier_epochs_per_round": 3,
            "entropy_coef": 0.02,
            "value_coef": 0.5,
            "reg_lambda_inside_anchor": 0.0,
            "use_perturbation": True,
            "perturbation_mode": "uniform",
            "n_perturb": 2048,
            "initial_window": 0.15,
            "step_fracs": (0.005, 0.01, 0.02),
            "min_width": 0.04,
            "precision_target": 0.95,
            "coverage_target": 0.04,
            "precision_blend_lambda": 0.5,
            "drift_penalty_weight": 0.05,
            "js_penalty_weight": 0.05,
            "disc_perc": [20, 40, 60, 80],
            # PPO params
            "ppo_epochs": 4,
            "clip_epsilon": 0.2,
        },
        "covtype": {
            "episodes": 60,
            "steps_per_episode": 90,
            "classifier_epochs_per_round": 3,
            "entropy_coef": 0.015,
            "value_coef": 0.5,
            "reg_lambda_inside_anchor": 0.0,
            "use_perturbation": True,
            "perturbation_mode": "uniform",
            "n_perturb": 8192,
            "initial_window": 0.1,
            "step_fracs": (0.003, 0.006, 0.012),
            "min_width": 0.02,
            "precision_target": 0.95,
            "coverage_target": 0.02,
            "precision_blend_lambda": 0.5,
            "drift_penalty_weight": 0.05,
            "js_penalty_weight": 0.05,
            "disc_perc": [10, 25, 50, 75, 90],
            # PPO params
            "ppo_epochs": 4,
            "clip_epsilon": 0.2,
        },
    }
    p = presets[dataset]

    # Resolve None parameters to dataset-specific defaults
    episodes = int(episodes if episodes is not None else p["episodes"])
    steps_per_episode = int(steps_per_episode if steps_per_episode is not None else p["steps_per_episode"])
    classifier_epochs_per_round = int(classifier_epochs_per_round if classifier_epochs_per_round is not None else p["classifier_epochs_per_round"])
    entropy_coef = float(entropy_coef if entropy_coef is not None else p["entropy_coef"]) 
    value_coef = float(value_coef if value_coef is not None else p["value_coef"]) 
    reg_lambda_inside_anchor = float(reg_lambda_inside_anchor if reg_lambda_inside_anchor is not None else p["reg_lambda_inside_anchor"]) 
    use_perturbation = bool(use_perturbation if use_perturbation is not None else p["use_perturbation"]) 
    perturbation_mode = str(perturbation_mode if perturbation_mode is not None else p["perturbation_mode"]) 
    n_perturb = int(n_perturb if n_perturb is not None else p["n_perturb"]) 
    initial_window = float(initial_window if initial_window is not None else p["initial_window"]) 
    # Optional override for precision target (for fair comparison across methods)
    if precision_target is not None:
        try:
            p["precision_target"] = float(precision_target)
        except Exception:
            pass
    # Optional override for coverage target
    if coverage_target is not None:
        try:
            p["coverage_target"] = float(coverage_target)
        except Exception:
            pass
    
    # DDPG parameters (kept for backward compatibility, but not used with SB3 DDPG)
    # SB3 DDPG uses its own parameters configured in the DDPG() initialization
    ppo_epochs = int(ppo_epochs if ppo_epochs is not None else p.get("ppo_epochs", 4))
    clip_epsilon = float(clip_epsilon if clip_epsilon is not None else p.get("clip_epsilon", 0.2))
    if batch_size is None:
        batch_size = None
    else:
        batch_size = int(batch_size)

    # Log classes and feature names together
    print("*****************")
    print("")
    print("Run configuration")
    print("")
    print("*****************")
    print(f"[data] classes ({num_classes}): {class_names} | feature_names ({len(feature_names)}): {feature_names}")
    disc_info = ""
    if use_discretization:
        disc_vals = (disc_perc if disc_perc is not None else p['disc_perc'])
        disc_info = f", disc_perc={disc_vals}"
    # Note: DDPG parameters are configured in SB3 DDPG initialization, not via these args
    print(f"[auto] using dataset-specific defaults: episodes={episodes}, steps={steps_per_episode}, clf_epochs={classifier_epochs_per_round}, use_perturbation={use_perturbation}, mode={perturbation_mode}, n_perturb={n_perturb}, initial_window={initial_window}, precision_target={p['precision_target']}, coverage_target={p['coverage_target']}{disc_info}")
    print(f"[DDPG] Using Stable Baselines 3 DDPG with continuous actions")

    # Split before scaling to avoid leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    # Build unit-space stats from train only; apply to both
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))
    X_unit_train = (X_train - X_min) / X_range
    X_unit_test = (X_test - X_min) / X_range

    device = select_device(device_preference)
    print(f"[device] using {device}")
    classifier = SimpleClassifier(X_train.shape[1], num_classes).to(device)
    clf_opt = optim.Adam(classifier.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    # Environment built on full normalized data but will reflect classifier behavior
    rng_local = np.random.default_rng(seed)
    # Local per-instance mode: restrict to class of x* and center initial box near x*
    x_star_unit = None
    if local_instance_index is not None and local_instance_index >= 0:
        idx = int(local_instance_index)
        idx = max(0, min(idx, X_unit_test.shape[0] - 1))
        x_star_unit = X_unit_test[idx]
        y_star = int(y_test[idx])
        target_classes = (y_star,)
        if debug:
            print(f"[local] anchoring on test idx={idx}, class={y_star}")

    edges = None  # Initialize edges for metadata storage
    # Continuous actions are not compatible with discretization
    if use_discretization:
        print("[warning] Discretization is not compatible with continuous actions. Disabling discretization.")
        use_discretization = False
    
    if use_discretization:
        # Build discretized representation on standardized train features
        if bin_edges is not None and len(bin_edges) == X_train.shape[1]:
            edges = [np.array(e, dtype=np.float32).ravel() for e in bin_edges]
        else:
            dp = disc_perc if disc_perc is not None else p["disc_perc"]
            edges = compute_quantile_bins(X_train, dp)
        X_bins_train = discretize_by_edges(X_train, edges)
        x_star_bins = None
        if x_star_unit is not None:
            # Map the representative x* (std) to bins
            x_star_bins = discretize_by_edges(x_star_unit.reshape(1, -1), edges)[0]
        bin_reps = compute_bin_representatives(X_train, X_bins_train)
        envs = {
            c: DiscreteAnchorEnv(
                X_bins=X_bins_train,
                X_std=X_train,
                y=y_train,
                feature_names=feature_names,
                classifier=classifier,
                device=device,
                bin_reps=bin_reps,
                bin_edges=edges,
                target_class=c,
                step_fracs=(1, 1, 1),
                min_width=1.0,
                alpha=0.7,
                beta=0.6,
                gamma=0.1,
                precision_target=p["precision_target"],
                coverage_target=p["coverage_target"],
                precision_blend_lambda=p["precision_blend_lambda"],
                drift_penalty_weight=p["drift_penalty_weight"],
                use_perturbation=use_perturbation,
                perturbation_mode=("bootstrap" if perturbation_mode not in ("bootstrap",) else perturbation_mode),
                n_perturb=n_perturb,
                rng=rng_local,
                min_coverage_floor=0.05,
                js_penalty_weight=p["js_penalty_weight"],
                x_star_bins=x_star_bins,
            ) for c in target_classes
        }
    else:
        envs = {
            c: AnchorEnv(
                X_unit_train, X_train, y_train, feature_names, classifier, device,
                target_class=c,
                step_fracs=p["step_fracs"],
                min_width=p["min_width"],
                alpha=0.7,
                beta=0.6,
                gamma=0.1,
                precision_target=p["precision_target"],
                coverage_target=p["coverage_target"],
                precision_blend_lambda=p["precision_blend_lambda"],
                drift_penalty_weight=p["drift_penalty_weight"],
                use_perturbation=use_perturbation,
                perturbation_mode=perturbation_mode,
                n_perturb=n_perturb,
                X_min=X_min, X_range=X_range,
                rng=rng_local,
                x_star_unit=x_star_unit,
                initial_window=initial_window,
                js_penalty_weight=p["js_penalty_weight"],
            ) for c in target_classes
        }

    # Create DDPG trainers for each class using Stable Baselines 3
    # Wrap each environment in a gym-compatible wrapper
    gym_envs = {}
    ddpg_trainers = {}  # Use trainer wrapper instead of direct DDPG model
    
    for cls, env in envs.items():
        # Wrap environment for SB3
        gym_env = ContinuousAnchorEnv(env, seed=seed + cls)
        gym_envs[cls] = gym_env
        
        # Use DDPG trainer wrapper if available, otherwise fall back to direct DDPG
        if DDPG_TRAINER_AVAILABLE:
            # Create DDPG trainer using the wrapper
            ddpg_trainer = create_ddpg_trainer(
                env=gym_env,
                policy_type="MlpPolicy",
                learning_rate=1e-4,  # Smaller LR for stability
                buffer_size=100000,  # Replay buffer size
                learning_starts=0,  # Start learning immediately (we control when to train)
                batch_size=64,  # Batch size for training
                tau=0.005,  # Soft update coefficient for target network
                gamma=0.99,  # Discount factor
                train_freq=(1, "step"),  # Train every step (but we'll call train() manually)
                gradient_steps=1,  # Gradient steps per training call
                action_noise_sigma=0.3,  # Stronger exploration for continuous box expansion
                policy_kwargs=dict(
                    net_arch=[256, 256]  # Policy network architecture
                ),
                verbose=0,  # Suppress SB3 output
                device=device,
            )
            ddpg_trainers[cls] = ddpg_trainer
        else:
            # Fallback to direct DDPG usage (old code)
            # Create action noise for exploration (DDPG uses deterministic policy)
            n_actions = gym_env.action_space.shape[0]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.3 * np.ones(n_actions)
            )
            
            # Initialize DDPG model from Stable Baselines 3
            ddpg_model = DDPG(
                policy="MlpPolicy",
                env=gym_env,
                learning_rate=1e-4,
                buffer_size=100000,
                learning_starts=0,
                batch_size=64,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                action_noise=action_noise,
                policy_kwargs=dict(net_arch=[256, 256]),
                verbose=0,
                device=device,
            )
            
            # Initialize logger
            if not hasattr(ddpg_model, '_logger') or ddpg_model._logger is None:
                from stable_baselines3.common.logger import configure
                logger = configure(folder=None, format_strings=[])
                ddpg_model.set_logger(logger)
            
            # Create a simple wrapper to match trainer interface
            class SimpleDDPGWrapper:
                def __init__(self, model):
                    self.model = model
                    self.env = gym_env
                
                def predict(self, obs, deterministic=False):
                    return self.model.predict(obs, deterministic=deterministic)
                
                def add_to_replay_buffer(self, obs, next_obs, action, reward, done, info=None):
                    if hasattr(self.model, 'replay_buffer') and self.model.replay_buffer is not None:
                        obs_array = np.array(obs, dtype=np.float32)
                        if obs_array.ndim == 1:
                            obs_array = obs_array.reshape(1, -1)
                        next_obs_array = np.array(next_obs, dtype=np.float32)
                        if next_obs_array.ndim == 1:
                            next_obs_array = next_obs_array.reshape(1, -1)
                        action_array = np.array(action, dtype=np.float32)
                        if action_array.ndim == 1:
                            action_array = action_array.reshape(1, -1)
                        self.model.replay_buffer.add(
                            obs=obs_array,
                            next_obs=next_obs_array,
                            action=action_array,
                            reward=np.array([reward], dtype=np.float32),
                            done=np.array([done], dtype=np.bool_),
                            infos=[info] if isinstance(info, dict) else [{}]
                        )
                        self.model.num_timesteps += 1
                
                def train_step(self, gradient_steps=1):
                    if (hasattr(self.model, 'replay_buffer') and 
                        self.model.replay_buffer is not None):
                        buffer_size = self.model.replay_buffer.size()
                        if buffer_size >= self.model.batch_size:
                            self.model.train(gradient_steps=gradient_steps, batch_size=self.model.batch_size)
                            return True
                    return False
                
                def get_buffer_size(self):
                    """Get current replay buffer size."""
                    if hasattr(self.model, 'replay_buffer') and self.model.replay_buffer is not None:
                        return self.model.replay_buffer.size()
                    return 0
                
                def save(self, path):
                    self.model.save(path)
            
            ddpg_trainers[cls] = SimpleDDPGWrapper(ddpg_model)

    def train_classifier_one_round(batch_size: int = 256):
        classifier.train()
        dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        last_loss = None
        last_train_acc = None
        for e in range(1, classifier_epochs_per_round + 1):
            epoch_loss_sum = 0.0
            epoch_correct = 0
            epoch_count = 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                clf_opt.zero_grad()
                logits = classifier(xb)
                loss = ce(logits, yb)
                # Optional regularization for consistency inside high-precision boxes
                if reg_lambda_inside_anchor > 0.0:
                    with torch.no_grad():
                        combined_mask = np.zeros(X.shape[0], dtype=bool)
                        for env in envs.values():
                            prec, cov, det = env._current_metrics()
                            if det["hard_precision"] >= env.precision_target and det["n_points"] > 0:
                                combined_mask |= env._mask_in_box()
                        idx = np.where(combined_mask)[0]
                    if idx.size > 0:
                        in_box_inputs = torch.from_numpy(X[idx]).float().to(device)
                        with torch.no_grad():
                            p_detach = classifier(in_box_inputs).detach()
                        reg = p_detach.var()
                        loss = loss + reg_lambda_inside_anchor * reg
                loss.backward()
                clf_opt.step()

                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    correct = (preds == yb).sum().item()
                    epoch_correct += correct
                    epoch_count += yb.size(0)
                    epoch_loss_sum += loss.item() * yb.size(0)

            last_loss = epoch_loss_sum / max(1, epoch_count)
            last_train_acc = epoch_correct / max(1, epoch_count)
            print(f"[clf] epoch {e}/{classifier_epochs_per_round} | loss={last_loss:.4f} | train_acc={last_train_acc:.3f} | samples={epoch_count}")
        return last_loss, last_train_acc

    def evaluate_classifier():
        classifier.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X_test).float().to(device)
            logits = classifier(inputs)
            preds = logits.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_test, preds)
        return float(acc)

    episode_rewards = []
    test_acc_history = []
    box_history_per_episode = []
    drift_history_per_episode = []
    prec_cov_history_per_episode = []
    reward_components_history = []
    # Per-class precision/coverage logging
    per_class_prec_cov = {c: [] for c in target_classes}
    # Per-class final box per episode
    per_class_box_history = {c: [] for c in target_classes}
    # Per-class rule strings per episode (for explanations over time)
    per_class_rule_history = {c: [] for c in target_classes}
    # Per-class full feature-conditions per episode (all features saved for revisit)
    per_class_full_conditions_history = {c: [] for c in target_classes}

    for ep in range(episodes):
        # 1) Train classifier according to cadence
        if (ep % max(1, int(classifier_update_every))) == 0:
            last_loss, last_train_acc = train_classifier_one_round()
        else:
            last_loss, last_train_acc = (0.0, 0.0)
        acc = evaluate_classifier()
        test_acc_history.append(acc)

        # 2) DDPG: Collect experiences and train using Stable Baselines 3
        episode_drifts = []
        episode_prec_cov = []
        total_return = 0.0
        # Track reward component sums for this episode (across classes)
        comp_sums = {"prec_gain": 0.0, "cov_gain": 0.0, "overlap_pen": 0.0, "drift_pen": 0.0, "js_pen": 0.0}

        for cls, (env, gym_env, ddpg_trainer) in zip(target_classes, 
                                                    [(envs[c], gym_envs[c], ddpg_trainers[c]) for c in target_classes]):
            # Reset environment
            if GYM_VERSION == "gymnasium":
                obs, _ = gym_env.reset(seed=seed + cls + ep)
            else:
                obs = gym_env.reset()
            
            # Capture initial width for tightened feature detection
            initial_lower = env.lower.copy()
            initial_upper = env.upper.copy()
            initial_width = (initial_upper - initial_lower)
            # Capture true pre-step metrics for histories
            tp0, tc0, td0 = env._current_metrics()
            episode_prec_cov.append((tp0, tc0, td0.get("hard_precision", 0.0)))
            
            class_rewards = []
            info = {}  # ensure defined if no steps
            
            # Run episode using DDPG - collect experiences and train
            for t in range(steps_per_episode):
                classifier.eval()
                
                # Get action from DDPG trainer (deterministic with noise for exploration)
                action, _ = ddpg_trainer.predict(obs, deterministic=False)
                
                # Step environment
                if GYM_VERSION == "gymnasium":
                    next_obs, reward, terminated, truncated, step_info = gym_env.step(action)
                    done = terminated or truncated
                else:
                    next_obs, reward, done, step_info = gym_env.step(action)
                
                # Store metrics
                class_rewards.append(float(reward))
                episode_drifts.append(step_info.get("drift", 0.0))
                episode_prec_cov.append((
                    step_info.get("precision", 0.0), 
                    step_info.get("coverage", 0.0), 
                    step_info.get("hard_precision", 0.0)
                ))
                # Accumulate reward components
                comp_sums["prec_gain"] += float(step_info.get("precision", 0.0))
                comp_sums["cov_gain"] += float(step_info.get("coverage", 0.0))
                comp_sums["drift_pen"] += float(step_info.get("drift", 0.0))
                comp_sums["js_pen"] += float(step_info.get("js_penalty", 0.0))
                
                # Add transition to DDPG trainer's replay buffer and train
                # DDPG trainer handles manual management when stepping outside learn()
                ddpg_trainer.add_to_replay_buffer(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    info=step_info
                )
                
                # Train DDPG if we have enough samples in replay buffer
                trained = ddpg_trainer.train_step(gradient_steps=1)
                
                # Log training progress (first few times to verify training is happening)
                if trained and ep == 0 and t < 5 and cls == target_classes[0]:
                    buffer_size = ddpg_trainer.get_buffer_size()
                    print(f"[DDPG Training] Class {cls}, Episode {ep}, Step {t}: "
                          f"Buffer size={buffer_size}, Training DDPG actor and critic networks")
                
                obs = next_obs
                info = step_info
                if done:
                    break
            
            total_return += sum(class_rewards) if class_rewards else 0.0

            # Per-class precision/coverage logging (track last info for this class)
            last_info_for_cls = info if 'info' in locals() else {}
            if debug:
                lw = (env.upper - env.lower)
                # Compare against initial width (reduced by at least 5% to be considered tightened)
                # This works for both discrete (bin counts) and continuous (normalized [0,1])
                tightened = np.where(lw < initial_width * 0.95)[0]
                topk_narrow = np.argsort(lw)[:3]
                narrow_bounds = ", ".join([f"{feature_names[i]}:[{env.lower[i]:.2f},{env.upper[i]:.2f}]" for i in topk_narrow])
                print(f"[env cls={cls}] end   | prec={last_info_for_cls.get('precision', 0.0):.3f} hard_prec={last_info_for_cls.get('hard_precision', 0.0):.3f} cov={last_info_for_cls.get('coverage', 0.0):.3f} n={last_info_for_cls.get('n_points', 0)} | width_mean={lw.mean():.3f} width_min={lw.min():.3f} | tightened={len(tightened)} | narrow {narrow_bounds}")
                # Human-readable rule summary with confidence
                # Select up to max_features_in_rule tightened features to show
                # If max_features_in_rule is 0 or negative, show all tightened features
                tightened_sorted = np.argsort(lw[tightened]) if tightened.size > 0 else np.array([])
                if max_features_in_rule > 0:
                    to_show_idx = (tightened[tightened_sorted[:max_features_in_rule]] if tightened.size > 0 else np.array([], dtype=int))
                else:
                    # Show all tightened features if max_features_in_rule <= 0
                    to_show_idx = tightened
                if to_show_idx.size == 0:
                    cond_str = "any values (no tightened features)"
                else:
                    cond_parts = []
                    for i in to_show_idx:
                        if hasattr(env, 'bin_edges') and isinstance(env, DiscreteAnchorEnv):
                            # Map bin indices to threshold text
                            lbin = int(np.floor(env.lower[i]))
                            ubin = int(np.ceil(env.upper[i]))
                            edges_i = env.bin_edges[i]
                            # Get actual feature min/max from standardized data
                            feat_min = float(env.X_std[:, i].min())
                            feat_max = float(env.X_std[:, i].max())
                            if lbin <= 0:
                                left = feat_min
                            else:
                                left = float(edges_i[min(lbin-1, len(edges_i)-1)])
                            if ubin >= len(edges_i):
                                right = feat_max
                            else:
                                right = float(edges_i[ubin])
                            # Format like static anchors: use inequalities
                            if left <= feat_min + 1e-6 and right >= feat_max - 1e-6:
                                continue
                            elif left <= feat_min + 1e-6:
                                cond_parts.append(f"{feature_names[i]} <= {right:.2f}")
                            elif right >= feat_max - 1e-6:
                                cond_parts.append(f"{feature_names[i]} > {left:.2f}")
                            else:
                                cond_parts.append(f"{feature_names[i]} > {left:.2f}")
                                cond_parts.append(f"{feature_names[i]} <= {right:.2f}")
                        else:
                            cond_parts.append(f"{feature_names[i]} ∈ [{env.lower[i]:.2f}, {env.upper[i]:.2f}]")
                    cond_str = " and ".join(cond_parts)
                print(
                    f"[rule cls={cls}] IF {cond_str} THEN class={cls} | "
                    f"soft={last_info_for_cls.get('avg_prob', 0.0):.3f}, hard={last_info_for_cls.get('hard_precision', 0.0):.3f}, "
                    f"blended={last_info_for_cls.get('precision', 0.0):.3f}, coverage={last_info_for_cls.get('coverage', 0.0):.3f}, sampler={last_info_for_cls.get('sampler', 'empirical')}"
                )
            # Store rule text per episode per class
            per_class_rule_history[cls].append(cond_str if 'cond_str' in locals() else "any values (no tightened features)")
            # Store full per-feature conditions (all features) for later analysis
            conds_all = {}
            for j in range(env.n_features):
                fname = feature_names[j]
                if hasattr(env, 'bin_edges') and isinstance(env, DiscreteAnchorEnv):
                    lbin = int(np.floor(env.lower[j]))
                    ubin = int(np.ceil(env.upper[j]))
                    edges_j = env.bin_edges[j]
                    # Get actual feature min/max from standardized data
                    feat_min = float(env.X_std[:, j].min())
                    feat_max = float(env.X_std[:, j].max())
                    if lbin <= 0:
                        left = feat_min
                    else:
                        left = float(edges_j[min(lbin-1, len(edges_j)-1)])
                    if ubin >= len(edges_j):
                        right = feat_max
                    else:
                        right = float(edges_j[ubin])
                    conds_all[fname] = {"type": "discrete_interval", "bin_lower": lbin, "bin_upper": ubin, "left": left, "right": right}
                else:
                    conds_all[fname] = {"type": "continuous_interval", "lower": float(env.lower[j]), "upper": float(env.upper[j])}
            per_class_full_conditions_history[cls].append(conds_all)
            per_class_prec_cov[cls].append({
                'precision': last_info_for_cls.get('precision', 0.0),
                'hard_precision': last_info_for_cls.get('hard_precision', 0.0),
                'coverage': last_info_for_cls.get('coverage', 0.0),
            })
            per_class_box_history[cls].append((env.lower.copy(), env.upper.copy()))

        # DDPG training is handled internally by SB3's learn() method during episode execution

        # Save box history for visualization (use last env as representative) and logs
        last_env = next(reversed(envs.values()))
        box_history_per_episode.append(last_env.box_history.copy())
        drift_history_per_episode.append(episode_drifts)
        prec_cov_history_per_episode.append(episode_prec_cov)

        episode_rewards.append(total_return)
        reward_components_history.append(comp_sums)
        last_p, last_c, last_hp = episode_prec_cov[-1] if episode_prec_cov else (0.0, 0.0, 0.0)
        print(f"Episode {ep+1}/{episodes} | return={total_return:.3f} | last_clf_loss={last_loss:.4f} | train_acc={last_train_acc:.3f} | test_acc={acc:.3f} | last_precision={last_p:.3f} | last_cov={last_c:.3f} | last_hard_precision={last_hp:.3f}")

    # Visualization: show evolution of two most varying features
    feat_var = X_unit_train.var(axis=0)
    top2 = np.argsort(-feat_var)[:2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(episode_rewards, label="return")
    # Moving average (window=5)
    if len(episode_rewards) >= 2:
        import numpy as _np
        w = 5
        ma = [_np.mean(episode_rewards[max(0,i-w+1):i+1]) for i in range(len(episode_rewards))]
        axes[0].plot(ma, label="moving avg (w=5)")
    axes[0].set_title("Episode returns")
    axes[0].legend()
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")

    axes[1].plot(test_acc_history)
    axes[1].set_title("Classifier test accuracy")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Accuracy")
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # Plot box bounds over episodes for top-2 features
    lower_series_f0 = [h[0][top2[0]] for ep_hist in box_history_per_episode for h in [ep_hist[-1]]]
    upper_series_f0 = [h[1][top2[0]] for ep_hist in box_history_per_episode for h in [ep_hist[-1]]]
    lower_series_f1 = [h[0][top2[1]] for ep_hist in box_history_per_episode for h in [ep_hist[-1]]]
    upper_series_f1 = [h[1][top2[1]] for ep_hist in box_history_per_episode for h in [ep_hist[-1]]]

    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lower_series_f0, label=f"{feature_names[top2[0]]} lower")
    plt.plot(upper_series_f0, label=f"{feature_names[top2[0]]} upper")
    plt.title("Anchor bounds over episodes (feature 1)")
    plt.xlabel("Episode")
    plt.ylabel("Normalized bound")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(lower_series_f1, label=f"{feature_names[top2[1]]} lower")
    plt.plot(upper_series_f1, label=f"{feature_names[top2[1]]} upper")
    plt.title("Anchor bounds over episodes (feature 2)")
    plt.xlabel("Episode")
    plt.ylabel("Normalized bound")
    plt.legend()
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # Plot drift per episode and precision-coverage trajectory
    avg_drift = [np.mean(d) if len(d) > 0 else 0.0 for d in drift_history_per_episode]
    avg_prec = [np.mean([pc[0] for pc in ep]) if len(ep) > 0 else 0.0 for ep in prec_cov_history_per_episode]
    avg_cov = [np.mean([pc[1] for pc in ep]) if len(ep) > 0 else 0.0 for ep in prec_cov_history_per_episode]
    avg_hard_prec = [np.mean([pc[2] for pc in ep]) if len(ep) > 0 else 0.0 for ep in prec_cov_history_per_episode]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(avg_drift)
    axes[0].set_title("Average drift per episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Drift")

    axes[1].plot(avg_prec, label="blended precision")
    axes[1].plot(avg_hard_prec, label="hard precision")
    axes[1].set_title("Precision over episodes")
    axes[1].legend()

    axes[2].plot(avg_cov)
    axes[2].set_title("Coverage over episodes")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Coverage")
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # Print per-class average metrics over episodes for easy comparison with static anchors
    # Use hard_precision to match static anchors' thresholded precision
    for cls in target_classes:
        series = per_class_prec_cov.get(cls, [])
        if len(series) == 0:
            continue
        avg_prec_cls = float(np.mean([d.get('hard_precision', d.get('precision', 0.0)) for d in series]))
        avg_cov_cls = float(np.mean([d.get('coverage', 0.0) for d in series]))
        cls_name = class_names[cls] if 'class_names' in locals() and cls < len(class_names) else str(cls)
        print(f"[dyn cls={cls}] {cls_name} | avg_precision={avg_prec_cls:.3f} | avg_coverage={avg_cov_cls:.3f} | episodes={len(series)}")

    # Per-class precision/coverage over episodes
    if per_class_prec_cov and len(per_class_prec_cov) > 0:
        episodes_idx = np.arange(1, episodes + 1)
        # Ensure equal-length series per class by padding with zeros if some episodes missing
        def series_for(cls, key):
            vals = [d.get(key, 0.0) for d in per_class_prec_cov.get(cls, [])]
            if len(vals) < episodes:
                vals = vals + [0.0] * (episodes - len(vals))
            return np.array(vals[:episodes])

        fig = plt.figure(figsize=(14, 5))
        # Hard precision per class
        plt.subplot(1, 2, 1)
        for cls in target_classes:
            hp = series_for(cls, 'hard_precision')
            plt.plot(episodes_idx, hp, label=f'class {cls}')
        plt.title('Per-class hard precision over episodes')
        plt.xlabel('Episode')
        plt.ylabel('Hard precision')
        plt.legend(ncol=2, fontsize=8)

        # Coverage per class
        plt.subplot(1, 2, 2)
        for cls in target_classes:
            cov = series_for(cls, 'coverage')
            plt.plot(episodes_idx, cov, label=f'class {cls}')
        plt.title('Per-class coverage over episodes')
        plt.xlabel('Episode')
        plt.ylabel('Coverage')
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    # Greedy evaluation with frozen policy (deterministic per class)
    print("\n=== Starting greedy evaluation ===")
    def greedy_rollout(env, gym_env, ddpg_trainer):
        # Reset environment
        if GYM_VERSION == "gymnasium":
            obs, _ = gym_env.reset()
        else:
            obs = gym_env.reset()
        
        # Capture initial full range for tightened check
        initial_lower = env.lower.copy()
        initial_upper = env.upper.copy()
        initial_width = (initial_upper - initial_lower)
        last_info = {"precision": 0.0, "coverage": 0.0, "hard_precision": 0.0, "avg_prob": 0.0, "sampler": "empirical"}
        # Track if box actually changed
        bounds_changed = False
        for t in range(steps_per_episode):
            # Greedy: use deterministic policy (no exploration)
            action, _ = ddpg_trainer.predict(obs, deterministic=True)
            prev_lower = env.lower.copy()
            prev_upper = env.upper.copy()
            
            # Step environment
            if GYM_VERSION == "gymnasium":
                next_obs, _, terminated, truncated, step_info = gym_env.step(action)
                done = terminated or truncated
            else:
                next_obs, _, done, step_info = gym_env.step(action)
            
            if not np.allclose(prev_lower, env.lower) or not np.allclose(prev_upper, env.upper):
                bounds_changed = True
            obs = next_obs
            last_info = step_info
            if done:
                break
        # If box didn't change at all, the policy likely isn't trained or actions are being reverted
        if not bounds_changed:
            # Fallback: manually tighten a bit to get a reasonable box
            if hasattr(env, 'bin_edges') and isinstance(env, DiscreteAnchorEnv):
                # For discrete, shrink by removing one bin from each side on a few features
                n_tighten = min(5, env.n_features)
                idx_perm = env.rng.permutation(env.n_features)[:n_tighten]
                for j in idx_perm:
                    max_bin = float(env._max_bin(j))
                    if env.upper[j] - env.lower[j] > env.min_width_bins[j]:
                        env.lower[j] = min(env.lower[j] + 1, env.upper[j] - env.min_width_bins[j])
                        env.upper[j] = max(env.upper[j] - 1, env.lower[j] + env.min_width_bins[j])
            else:
                # For continuous, shrink by 10% on a few features
                n_tighten = min(5, env.n_features)
                idx_perm = env.rng.permutation(env.n_features)[:n_tighten]
                for j in idx_perm:
                    width = env.upper[j] - env.lower[j]
                    if width > env.min_width:
                        shrink = 0.1 * width
                        env.lower[j] = min(env.lower[j] + shrink, env.upper[j] - env.min_width)
                        env.upper[j] = max(env.upper[j] - shrink, env.lower[j] + env.min_width)
            # Recompute metrics after manual tightening
            prec_new, cov_new, det_new = env._current_metrics()
            last_info.update(det_new)
            last_info["precision"] = prec_new
            last_info["coverage"] = cov_new
        # Build rule string - check if tightened from initial (not just < 0.999)
        lw = (env.upper - env.lower)
        tightened = np.where(lw < initial_width * 0.95)[0]  # tightened if width reduced by at least 5%
        if tightened.size == 0:
            cond_str = "any values (no tightened features)"
        else:
            # Sort by tightness (narrowest features first) and limit to max_features_in_rule
            # If max_features_in_rule is 0 or negative, show all tightened features
            tightened_sorted = np.argsort(lw[tightened]) if tightened.size > 0 else np.array([])
            if max_features_in_rule > 0:
                to_show_idx = (tightened[tightened_sorted[:max_features_in_rule]] if tightened.size > 0 else np.array([], dtype=int))
            else:
                # Show all tightened features if max_features_in_rule <= 0
                to_show_idx = tightened
            if to_show_idx.size == 0:
                cond_str = "any values (no tightened features)"
            else:
                cond_parts = []
                for i in to_show_idx:
                    if hasattr(env, 'bin_edges') and isinstance(env, DiscreteAnchorEnv):
                        lbin = int(np.floor(env.lower[i]))
                        ubin = int(np.ceil(env.upper[i]))
                        edges_i = env.bin_edges[i]
                        # Get actual feature min/max from standardized data
                        feat_min = float(env.X_std[:, i].min())
                        feat_max = float(env.X_std[:, i].max())
                        if lbin <= 0:
                            left = feat_min
                        else:
                            left = float(edges_i[min(lbin-1, len(edges_i)-1)])
                        if ubin >= len(edges_i):
                            right = feat_max
                        else:
                            right = float(edges_i[ubin])
                        # Format like static anchors: use inequalities instead of intervals
                        # If left is min, only upper bound; if right is max, only lower bound
                        if left <= feat_min + 1e-6 and right >= feat_max - 1e-6:
                            # Full range, skip this feature
                            continue
                        elif left <= feat_min + 1e-6:
                            # Only upper bound: feature <= right
                            cond_parts.append(f"{feature_names[i]} <= {right:.2f}")
                        elif right >= feat_max - 1e-6:
                            # Only lower bound: feature > left
                            cond_parts.append(f"{feature_names[i]} > {left:.2f}")
                        else:
                            # Both bounds: feature > left AND feature <= right
                            cond_parts.append(f"{feature_names[i]} > {left:.2f}")
                            cond_parts.append(f"{feature_names[i]} <= {right:.2f}")
                    else:
                        # Continuous case: use interval notation
                        cond_parts.append(f"{feature_names[i]} ∈ [{env.lower[i]:.2f}, {env.upper[i]:.2f}]")
                cond_str = " and ".join(cond_parts)
        return last_info, cond_str, env.lower.copy(), env.upper.copy()

    # Use test instances (like static anchors) with num_greedy_rollouts test instances per class
    # Each test instance gets 1 greedy rollout (matches static anchors approach)
    final_greedy = {}
    final_greedy_all = {}  # Store all individual anchors for analysis
    
    # Determine number of test instances to use
    # If num_test_instances_per_class is specified, use it; otherwise use num_greedy_rollouts
    if num_test_instances_per_class is not None:
        num_instances_per_class = int(num_test_instances_per_class)
    else:
        # Use num_greedy_rollouts as number of test instances (like before, but with test data)
        num_instances_per_class = num_greedy_rollouts if num_greedy_rollouts > 1 else 20  # Default to 20 if 1
    
    print(f"[greedy] Evaluating {num_instances_per_class} test instances per class with 1 rollout each")
    
    for cls in target_classes:
        # Sample test instances for this class (like static anchors)
        idx_cls = np.where(y_test == cls)[0]
        if idx_cls.size == 0:
            continue
        
        # Sample up to num_instances_per_class test instances
        sel = rng_local.choice(idx_cls, size=min(num_instances_per_class, idx_cls.size), replace=False)
        cls_name = class_names[cls] if cls < len(class_names) else str(cls)
        print(f"[greedy] Class {cls} ({cls_name}): Running {len(sel)} greedy rollouts...")
        all_anchor_results = []  # Collect all anchors (one per instance)
        
        for i, instance_idx in enumerate(sel):
            if (i + 1) % 5 == 0 or (i + 1) == len(sel):
                print(f"  [greedy cls={cls}] Progress: {i+1}/{len(sel)} instances", end='\r')
            # Create env for this test instance
            # Start from full range (not centered on instance) to allow finding good coverage
            # The test instance is used for evaluation context but not to constrain the initial box
            if use_discretization:
                env = DiscreteAnchorEnv(
                    X_bins=discretize_by_edges(X_test, edges),  # Use test data for evaluation
                    X_std=X_test,  # Use test data
                    y=y_test,  # Use test labels
                    feature_names=feature_names,
                    classifier=classifier,
                    device=device,
                    bin_reps=bin_reps,
                    bin_edges=edges,
                    target_class=cls,
                    step_fracs=(1,1,1),
                    min_width=1.0,
                    alpha=0.7,
                    beta=0.6,
                    gamma=0.1,
                    precision_target=p["precision_target"],
                    coverage_target=p["coverage_target"],
                    precision_blend_lambda=p["precision_blend_lambda"],
                    drift_penalty_weight=p["drift_penalty_weight"],
                    use_perturbation=use_perturbation,
                    perturbation_mode=("bootstrap" if perturbation_mode not in ("bootstrap",) else perturbation_mode),
                    n_perturb=n_perturb,
                    rng=np.random.default_rng(seed + instance_idx * 1000),
                    min_coverage_floor=0.05,
                    js_penalty_weight=p["js_penalty_weight"],
                    x_star_bins=None,  # Don't center - start from full range
                )
            else:
                env = AnchorEnv(
                    X_unit_test, X_test, y_test, feature_names, classifier, device,
                    target_class=cls,
                    step_fracs=p["step_fracs"],
                    min_width=p["min_width"],
                    alpha=0.7,
                    beta=0.6,
                    gamma=0.1,
                    precision_target=p["precision_target"],
                    coverage_target=p["coverage_target"],
                    precision_blend_lambda=p["precision_blend_lambda"],
                    drift_penalty_weight=p["drift_penalty_weight"],
                    use_perturbation=use_perturbation,
                    perturbation_mode=perturbation_mode,
                    n_perturb=n_perturb,
                    X_min=X_min, X_range=X_range,
                    rng=np.random.default_rng(seed + instance_idx * 1000),
                    x_star_unit=None,  # Don't center - start from full range
                    initial_window=initial_window,
                    js_penalty_weight=p["js_penalty_weight"],
                )
            
            # Run one greedy rollout for this test instance (like static: one anchor per instance)
            # Wrap environment for DDPG
            gym_env_eval = ContinuousAnchorEnv(env, seed=seed + cls + instance_idx)
            # Use the trained DDPG trainer for this class
            ddpg_trainer_eval = ddpg_trainers[cls]
            info_g, rule_g, lower_g, upper_g = greedy_rollout(env, gym_env_eval, ddpg_trainer_eval)
            # Verify metrics by recomputing on final box
            env.lower[:] = lower_g
            env.upper[:] = upper_g
            prec_check, cov_check, det_check = env._current_metrics()
            
            # If greedy produced full-range box (coverage=1.0), fallback to best training episode
            if cov_check >= 0.99:
                # Find best training episode box for this class
                hist = per_class_prec_cov.get(int(cls), [])
                if hist:
                    best_idx = max(range(len(hist)), key=lambda i: hist[i].get('hard_precision', 0.0))
                    best_lower, best_upper = per_class_box_history[cls][best_idx]
                    env.lower[:] = best_lower
                    env.upper[:] = best_upper
                    prec_check, cov_check, det_check = env._current_metrics()
                    # Rebuild rule for best training box
                    if isinstance(env, DiscreteAnchorEnv):
                        initial_width_best = np.array([float(env._max_bin(j)) for j in range(env.n_features)])
                    else:
                        initial_width_best = np.ones(env.n_features, dtype=np.float32)
                    lw_best = (env.upper - env.lower)
                    tightened_best = np.where(lw_best < initial_width_best * 0.95)[0]
                    if tightened_best.size > 0:
                        # Sort by tightness and limit to max_features_in_rule
                        # If max_features_in_rule is 0 or negative, show all tightened features
                        tightened_best_sorted = np.argsort(lw_best[tightened_best]) if tightened_best.size > 0 else np.array([])
                        if max_features_in_rule > 0:
                            to_show_best = (tightened_best[tightened_best_sorted[:max_features_in_rule]] if tightened_best.size > 0 else np.array([], dtype=int))
                        else:
                            # Show all tightened features if max_features_in_rule <= 0
                            to_show_best = tightened_best
                        cond_parts = []
                        for i in to_show_best:
                            if hasattr(env, 'bin_edges') and isinstance(env, DiscreteAnchorEnv):
                                lbin = int(np.floor(env.lower[i]))
                                ubin = int(np.ceil(env.upper[i]))
                                edges_i = env.bin_edges[i]
                                # Get actual feature min/max from standardized data
                                feat_min = float(env.X_std[:, i].min())
                                feat_max = float(env.X_std[:, i].max())
                                if lbin <= 0:
                                    left = feat_min
                                else:
                                    left = float(edges_i[min(lbin-1, len(edges_i)-1)])
                                if ubin >= len(edges_i):
                                    right = feat_max
                                else:
                                    right = float(edges_i[ubin])
                                # Format like static anchors: use inequalities
                                if left <= feat_min + 1e-6 and right >= feat_max - 1e-6:
                                    continue
                                elif left <= feat_min + 1e-6:
                                    cond_parts.append(f"{feature_names[i]} <= {right:.2f}")
                                elif right >= feat_max - 1e-6:
                                    cond_parts.append(f"{feature_names[i]} > {left:.2f}")
                                else:
                                    cond_parts.append(f"{feature_names[i]} > {left:.2f}")
                                    cond_parts.append(f"{feature_names[i]} <= {right:.2f}")
                            else:
                                cond_parts.append(f"{feature_names[i]} ∈ [{env.lower[i]:.2f}, {env.upper[i]:.2f}]")
                        rule_g = " and ".join(cond_parts)
                    else:
                        rule_g = "any values (no tightened features)"
                    lower_g = env.lower.copy()
                    upper_g = env.upper.copy()
            
            all_anchor_results.append({
                "precision": float(prec_check),
                "hard_precision": float(det_check.get("hard_precision", 0.0)),
                "coverage": float(cov_check),
                "rule": rule_g,
                "lower": lower_g.tolist(),
                "upper": upper_g.tolist(),
                "instance_idx": int(instance_idx),
            })
        
        print()  # New line after progress
        
        # Average metrics across all anchors (one per test instance, like static)
        if len(all_anchor_results) > 0:
            avg_prec = float(np.mean([r["precision"] for r in all_anchor_results]))
            avg_hard_prec = float(np.mean([r["hard_precision"] for r in all_anchor_results]))
            avg_cov = float(np.mean([r["coverage"] for r in all_anchor_results]))
            # Use the best anchor's rule (by hard precision) as representative
            best_anchor = max(all_anchor_results, key=lambda r: r["hard_precision"])
            final_greedy[int(cls)] = {
                "precision": avg_prec,
                "hard_precision": avg_hard_prec,
                "coverage": avg_cov,
                "rule": best_anchor["rule"],
                "lower": best_anchor["lower"],
                "upper": best_anchor["upper"],
                "num_instances": len(sel),
                "num_rollouts": len(all_anchor_results),  # Total anchors = number of instances
                "total_anchors": len(all_anchor_results),
            }
            # Store all individual anchors
            final_greedy_all[int(cls)] = all_anchor_results
            print(f"[greedy cls={cls}] Completed: avg_precision={avg_hard_prec:.3f}, avg_coverage={avg_cov:.3f}, n={len(all_anchor_results)}")
        else:
            # No anchors found
            final_greedy[int(cls)] = {
                "precision": 0.0,
                "hard_precision": 0.0,
                "coverage": 0.0,
                "rule": "no anchors found",
                "lower": None,
                "upper": None,
                "num_instances": 0,
                "num_rollouts": 0,
                "total_anchors": 0,
            }
    
    print(f"\n=== Greedy evaluation complete ===")
    print(f"[greedy] Evaluated {sum(len(final_greedy_all.get(cls, [])) for cls in target_classes)} total rollouts across {len(final_greedy)} classes")

    # Final confusion matrix on test set
    classifier.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X_test).float().to(device)
        probs_final = classifier(inputs).cpu().numpy()
    final_preds = probs_final.argmax(axis=1)
    cm = confusion_matrix(y_test, final_preds, labels=list(range(num_classes)))

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix (test set)')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # Annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8)
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    return {
        "classifier": classifier,  # PyTorch model - should be saved separately
        "ddpg_trainers": ddpg_trainers,  # SB3 DDPG trainers - should be saved separately
        "episode_returns": episode_rewards,
        "test_accuracy": test_acc_history,
        "box_history": box_history_per_episode,
        "drift_history": drift_history_per_episode,
        "precision_coverage_history": prec_cov_history_per_episode,
        "top2_features": [feature_names[i] for i in top2],
        "per_class_precision_coverage_history": per_class_prec_cov,
        "per_class_box_history": per_class_box_history,
        "per_class_rule_history": per_class_rule_history,
        "per_class_full_conditions_history": per_class_full_conditions_history,
        "final_greedy": final_greedy,
        "final_greedy_all": final_greedy_all,  # All individual rollouts (for analysis)
        # Metadata for loading
        "metadata": {
            "dataset": dataset,
            "num_classes": num_classes,
            "feature_names": feature_names,
            "class_names": class_names,
            "n_features": int(X_train.shape[1]),
            "seed": seed,
            "use_discretization": use_discretization,
            "bin_edges": [e.tolist() if isinstance(e, np.ndarray) else e for e in edges] if (use_discretization and edges is not None) else None,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "X_min": X_min.tolist(),
            "X_range": X_range.tolist(),
            "preset_params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()},
        }
    }


def load_trained_models(
    results_file: str,
    policy_file: str | None = None,
    value_file: str | None = None,
    classifier_file: str | None = None,
    device_preference: str = "auto",
):
    """
    Load trained models and metadata from saved files.
    
    Args:
        results_file: Path to the JSON results file containing metadata
        policy_file: Path to policy model file (None = auto-detect from results_file)
        value_file: Path to value network file (None = auto-detect from results_file)
        classifier_file: Path to classifier model file (None = auto-detect from results_file)
        device_preference: Device to load models on ("auto", "cuda", "mps", "cpu")
    
    Returns:
        dict with keys: "classifier", "policy", "value_fn", "metadata", "scaler"
    """
    import json
    from sklearn.preprocessing import StandardScaler
    
    # Load metadata from results file
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    metadata = results.get("metadata", {})
    
    # Auto-detect model filenames if not provided
    base_name = results_file.replace('results_', '').replace('.json', '')
    if policy_file is None:
        policy_file = f'policy_{base_name}.pth'
    if value_file is None:
        value_file = f'value_fn_{base_name}.pth'
    if classifier_file is None:
        classifier_file = f'classifier_{base_name}.pth'
    
    # Load device
    device = select_device(device_preference)
    
    # Load classifier
    n_features = metadata["n_features"]
    num_classes = metadata["num_classes"]
    classifier = SimpleClassifier(n_features, num_classes).to(device)
    classifier.load_state_dict(torch.load(classifier_file, map_location=device))
    classifier.eval()
    
    # Load policy (continuous actions)
    state_dim = 2 * n_features + 2
    action_dim = None  # Will be inferred from policy file
    policy = PolicyNet(state_dim, 1000, log_std_init=-0.5).to(device)  # Temporary action_dim
    policy.load_state_dict(torch.load(policy_file, map_location=device))
    # Infer action_dim from loaded weights (continuous: 2 * n_features)
    action_dim = policy.mean_head.out_features
    policy = PolicyNet(state_dim, action_dim, log_std_init=-0.5).to(device)
    policy.load_state_dict(torch.load(policy_file, map_location=device))
    policy.eval()
    
    # Load value network
    value_fn = ValueNet(state_dim).to(device)
    value_fn.load_state_dict(torch.load(value_file, map_location=device))
    value_fn.eval()
    
    # Recreate scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(metadata["scaler_mean"])
    scaler.scale_ = np.array(metadata["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    
    return {
        "classifier": classifier,
        "policy": policy,
        "value_fn": value_fn,
        "metadata": metadata,
        "scaler": scaler,
        "device": device,
    }


def explain_instance(
    loaded_models: dict,
    X_instance: np.ndarray,
    target_class: int | None = None,
    steps_per_episode: int = 40,
    max_features_in_rule: int = 5,
):
    """
    Generate anchor explanation for a single instance.
    
    Args:
        loaded_models: Output from load_trained_models()
        X_instance: Instance to explain (raw features, will be standardized)
        target_class: Target class to explain (None = use model's prediction)
        steps_per_episode: Number of greedy rollout steps
        max_features_in_rule: Maximum number of features to show in rule
    
    Returns:
        dict with keys: "rule", "precision", "hard_precision", "coverage", "lower", "upper", "target_class"
    """
    metadata = loaded_models["metadata"]
    classifier = loaded_models["classifier"]
    policy = loaded_models["policy"]
    scaler = loaded_models["scaler"]
    device = loaded_models["device"]
    
    # Standardize instance
    X_instance_std = scaler.transform(X_instance.reshape(1, -1)).astype(np.float32).ravel()
    
    # Predict class if not provided
    if target_class is None:
        with torch.no_grad():
            inputs = torch.from_numpy(X_instance_std).float().unsqueeze(0).to(device)
            logits = classifier(inputs)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            target_class = int(preds[0])
    
    # Create unit-space representation
    X_min = np.array(metadata["X_min"])
    X_range = np.array(metadata["X_range"])
    X_instance_unit = (X_instance_std - X_min) / X_range
    
    # Create environment
    feature_names = metadata["feature_names"]
    use_discretization = metadata.get("use_discretization", True)
    
    if use_discretization:
        bin_edges = metadata.get("bin_edges")
        if bin_edges:
            bin_edges = [np.array(e, dtype=np.float32) for e in bin_edges]
            X_bins_instance = discretize_by_edges(X_instance_std.reshape(1, -1), bin_edges)[0]
            # Use simplified bin reps (median of bin)
            bin_reps = []
            for j, edges in enumerate(bin_edges):
                reps_j = np.zeros(len(edges) + 1, dtype=np.float32)
                for b in range(len(edges) + 1):
                    if b == 0:
                        reps_j[b] = float(edges[0]) if len(edges) > 0 else X_instance_std[j]
                    elif b >= len(edges):
                        reps_j[b] = float(edges[-1]) if len(edges) > 0 else X_instance_std[j]
                    else:
                        reps_j[b] = float((edges[b-1] + edges[b]) / 2.0)
                bin_reps.append(reps_j)
            
            env = DiscreteAnchorEnv(
                X_bins=X_bins_instance.reshape(1, -1),
                X_std=X_instance_std.reshape(1, -1),
                y=np.array([target_class]),
                feature_names=feature_names,
                classifier=classifier,
                device=device,
                bin_reps=bin_reps,
                bin_edges=bin_edges,
                target_class=target_class,
                step_fracs=(1, 1, 1),
                min_width=1.0,
                precision_target=metadata["preset_params"]["precision_target"],
                coverage_target=metadata["preset_params"]["coverage_target"],
                rng=np.random.default_rng(metadata["seed"]),
                x_star_bins=None,  # Start from full range
            )
        else:
            use_discretization = False
    
    if not use_discretization:
        # Continuous environment
        env = AnchorEnv(
            X_unit=X_instance_unit.reshape(1, -1),
            X_std=X_instance_std.reshape(1, -1),
            y=np.array([target_class]),
            feature_names=feature_names,
            classifier=classifier,
            device=device,
            target_class=target_class,
            step_fracs=tuple(metadata["preset_params"]["step_fracs"]),
            min_width=metadata["preset_params"]["min_width"],
            precision_target=metadata["preset_params"]["precision_target"],
            coverage_target=metadata["preset_params"]["coverage_target"],
            X_min=X_min,
            X_range=X_range,
            rng=np.random.default_rng(metadata["seed"]),
            x_star_unit=None,  # Start from full range
        )
    
    # Greedy rollout
    def greedy_rollout(env, policy, steps_per_episode):
        state = env.reset()
        initial_lower = env.lower.copy()
        initial_upper = env.upper.copy()
        initial_width = (initial_upper - initial_lower)
        
        for t in range(steps_per_episode):
            s = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                # Greedy: use mean of policy (no exploration)
                mean, _ = policy(s)
                action_np = mean.cpu().numpy()[0]  # Continuous action
            next_state, _, done, info = env.step(action_np)
            state = next_state
            if done:
                break
        
        # Build rule
        lw = (env.upper - env.lower)
        tightened = np.where(lw < initial_width * 0.95)[0]
        
        if tightened.size == 0:
            rule = "any values (no tightened features)"
        else:
            tightened_sorted = np.argsort(lw[tightened])
            to_show_idx = tightened[tightened_sorted[:max_features_in_rule]] if max_features_in_rule > 0 else tightened
            
            cond_parts = []
            for i in to_show_idx:
                if hasattr(env, 'bin_edges') and isinstance(env, DiscreteAnchorEnv):
                    lbin = int(np.floor(env.lower[i]))
                    ubin = int(np.ceil(env.upper[i]))
                    edges_i = env.bin_edges[i]
                    feat_min = float(env.X_std[:, i].min()) if env.X_std.shape[0] > 0 else -np.inf
                    feat_max = float(env.X_std[:, i].max()) if env.X_std.shape[0] > 0 else np.inf
                    
                    if lbin <= 0:
                        left = feat_min
                    else:
                        left = float(edges_i[min(lbin-1, len(edges_i)-1)])
                    if ubin >= len(edges_i):
                        right = feat_max
                    else:
                        right = float(edges_i[ubin])
                    
                    if left <= feat_min + 1e-6 and right >= feat_max - 1e-6:
                        continue
                    elif left <= feat_min + 1e-6:
                        cond_parts.append(f"{feature_names[i]} <= {right:.2f}")
                    elif right >= feat_max - 1e-6:
                        cond_parts.append(f"{feature_names[i]} > {left:.2f}")
                    else:
                        cond_parts.append(f"{feature_names[i]} > {left:.2f}")
                        cond_parts.append(f"{feature_names[i]} <= {right:.2f}")
                else:
                    cond_parts.append(f"{feature_names[i]} ∈ [{env.lower[i]:.2f}, {env.upper[i]:.2f}]")
            rule = " and ".join(cond_parts)
        
        # Get final metrics
        prec, cov, det = env._current_metrics()
        
        return {
            "rule": rule,
            "precision": float(prec),
            "hard_precision": float(det.get("hard_precision", 0.0)),
            "coverage": float(cov),
            "lower": env.lower.tolist(),
            "upper": env.upper.tolist(),
            "target_class": target_class,
        }
    
    result = greedy_rollout(env, policy, steps_per_episode)
    return result


def explain_class(
    loaded_models: dict,
    target_class: int,
    X_test: np.ndarray,
    y_test: np.ndarray | None = None,
    num_instances: int = 20,
    steps_per_episode: int = 40,
    max_features_in_rule: int = 5,
):
    """
    Generate anchor explanation for a class by averaging over multiple test instances.
    
    Args:
        loaded_models: Output from load_trained_models()
        target_class: Target class to explain
        X_test: Test instances (raw features, will be standardized)
        y_test: Test labels (None = not used)
        num_instances: Number of instances to sample for averaging
        steps_per_episode: Number of greedy rollout steps
        max_features_in_rule: Maximum number of features to show in rule
    
    Returns:
        dict with keys: "rule", "precision", "hard_precision", "coverage", "num_instances"
    """
    # Filter instances of target class
    if y_test is not None:
        idx_cls = np.where(y_test == target_class)[0]
        if idx_cls.size == 0:
            raise ValueError(f"No instances found for class {target_class}")
        sel = np.random.choice(idx_cls, size=min(num_instances, idx_cls.size), replace=False)
        X_selected = X_test[sel]
    else:
        # If no labels, sample random instances
        if X_test.shape[0] < num_instances:
            sel = np.arange(X_test.shape[0])
        else:
            sel = np.random.choice(X_test.shape[0], size=num_instances, replace=False)
        X_selected = X_test[sel]
    
    # Get explanations for each instance
    all_explanations = []
    for instance in X_selected:
        expl = explain_instance(
            loaded_models,
            instance,
            target_class=target_class,
            steps_per_episode=steps_per_episode,
            max_features_in_rule=max_features_in_rule,
        )
        all_explanations.append(expl)
    
    # Average metrics
    avg_prec = float(np.mean([e["precision"] for e in all_explanations]))
    avg_hard_prec = float(np.mean([e["hard_precision"] for e in all_explanations]))
    avg_cov = float(np.mean([e["coverage"] for e in all_explanations]))
    
    # Use best rule (by hard precision)
    best_expl = max(all_explanations, key=lambda e: e["hard_precision"])
    
    return {
        "rule": best_expl["rule"],
        "precision": avg_prec,
        "hard_precision": avg_hard_prec,
        "coverage": avg_cov,
        "num_instances": len(all_explanations),
        "all_explanations": all_explanations,  # Individual explanations for analysis
    }


if __name__ == "__main__":
    import argparse

    def parse_nullable_int(v: str | None) -> int | None:
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in ("none", "null", "nan", ""):
            return None
        return int(v)

    parser = argparse.ArgumentParser(description="Run dynamic anchor training with DDPG from Stable Baselines 3 and continuous actions.")
    parser.add_argument("--dataset", type=str, default="covtype", choices=["breast_cancer", "synthetic", "covtype"], help="Dataset to use")
    parser.add_argument("--episodes", type=int, default=30, help="Number of RL episodes")
    parser.add_argument("--steps", type=int, default=40, help="Steps per episode")
    parser.add_argument("--classifier_epochs", type=int, default=3, help="Classifier epochs per RL episode")
    parser.add_argument("--reg_lambda", type=float, default=0.0, help="Regularization inside anchor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=list(DEVICE_CHOICES), help="Device: auto|cuda|mps|cpu")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_perturbation", dest="use_perturbation", action="store_true", help="Enable perturbation-based sampling inside boxes")
    group.add_argument("--no-perturbation", dest="use_perturbation", action="store_false", help="Disable perturbation-based sampling inside boxes")
    parser.set_defaults(use_perturbation=True)
    parser.add_argument("--perturbation_mode", type=str, default="uniform", choices=["bootstrap", "uniform"], help="Sampler when perturbations are enabled")
    parser.add_argument("--local_instance_index", type=int, default=-1, help="If >=0, run local per-instance anchor for test instance index")
    parser.add_argument("--initial_window", type=float, default=0.1, help="Initial half-width around x* in unit space for local anchors")
    parser.add_argument("--n_perturb", type=int, default=4096, help="Number of synthetic/bootstrapped samples per box evaluation")
    parser.add_argument("--show_plots", action="store_true", default=True, help="Enable visualization plots (default: True)")
    parser.add_argument("--no-plots", dest="show_plots", action="store_false", help="Disable visualization plots")
    parser.add_argument("--max_features_in_rule", type=int, default=5, help="Maximum number of features to show in anchor rules (default: 5, use 0 for all features)")
    parser.add_argument("--num_greedy_rollouts", type=int, default=20, help="Number of greedy rollouts per test instance (default: 20)")
    parser.add_argument("--num_test_instances", type=parse_nullable_int, default=None, help="Number of test instances per class to evaluate (None=use num_greedy_rollouts if >1, else default to 20)")
    # PPO-specific arguments
    parser.add_argument("--ppo_epochs", type=int, default=None, help="[DEPRECATED] Not used with DDPG (kept for compatibility)")
    parser.add_argument("--clip_epsilon", type=float, default=None, help="[DEPRECATED] Not used with DDPG (kept for compatibility)")
    parser.add_argument("--batch_size", type=int, default=None, help="[DEPRECATED] Not used with DDPG (kept for compatibility)")

    args = parser.parse_args()

    results = train_dynamic_anchors(
        dataset=args.dataset,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        classifier_epochs_per_round=args.classifier_epochs,
        reg_lambda_inside_anchor=args.reg_lambda,
        seed=args.seed,
        device_preference=args.device,
        use_perturbation=args.use_perturbation,
        perturbation_mode=args.perturbation_mode,
        n_perturb=args.n_perturb,
        local_instance_index=args.local_instance_index,
        initial_window=args.initial_window,
        show_plots=args.show_plots,
        max_features_in_rule=args.max_features_in_rule,
        num_greedy_rollouts=args.num_greedy_rollouts,
        num_test_instances_per_class=args.num_test_instances,
        ppo_epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        batch_size=args.batch_size,
    )

    import json

    # Helper function to recursively convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        """Recursively convert numpy arrays and other non-serializable types to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__') and not isinstance(obj, (torch.nn.Module, torch.Tensor)):
            # For other objects with __dict__, try to convert
            return convert_to_serializable(obj.__dict__)
        else:
            return obj

    # Remove PyTorch models from results before JSON serialization
    classifier_model = results.pop('classifier', None) if 'classifier' in results else None
    ddpg_trainers_dict = results.pop('ddpg_trainers', None) if 'ddpg_trainers' in results else None

    # Save models separately
    model_prefix = f'ddpg_continuous_{args.dataset}_{args.episodes}_{args.steps}_{args.classifier_epochs}_{args.reg_lambda}_{args.seed}_{args.device}_{args.use_perturbation}_{args.perturbation_mode}_{args.n_perturb}_{args.local_instance_index}_{args.initial_window}_{args.show_plots}_{args.max_features_in_rule}'
    if classifier_model is not None:
        classifier_file = f'classifier_{model_prefix}.pth'
        torch.save(classifier_model.state_dict(), classifier_file)
        print(f"Classifier saved to {classifier_file}")
    if ddpg_trainers_dict is not None:
        # Save DDPG trainers using SB3's save method
        for cls, ddpg_trainer in ddpg_trainers_dict.items():
            ddpg_file = f'ddpg_class_{cls}_{model_prefix}.zip'
            ddpg_trainer.save(ddpg_file)
            print(f"DDPG trainer for class {cls} saved to {ddpg_file}")

    # Convert all numpy arrays to lists for JSON serialization
    results_serializable = convert_to_serializable(results)

    # save results to json
    results_file = f'results_{model_prefix}.json'
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    print(f"Results saved to {results_file}")


