"""
Joint training pipeline for Dynamic Anchors: Interleaved classifier and RL training.

This module provides joint training where the classifier and RL policy are
trained in an interleaved manner (matching dyn_anchor_PPO.py):
- RL training happens every episode
- Classifier updates happen every N episodes
- Evaluation is done at the end with frozen policy

Usage example:
    from trainers.tabular_dynAnchors_joint import train_and_evaluate_joint
    
    results = train_and_evaluate_joint(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        target_classes=(0, 1, 2),
        episodes=60,
        steps_per_episode=90,
        classifier_epochs_per_round=4,
        classifier_update_every=1,
    )
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import List, Tuple, Optional, Dict, Any
from functools import partial
from stable_baselines3.common.callbacks import BaseCallback


def compute_bootstrapped_return(
    rewards: list, 
    last_obs: np.ndarray, 
    model, 
    gamma: float = 0.99,
    truncated: bool = False
) -> float:
    """
    Compute episode return with bootstrap for truncated episodes.
    
    If episode ended due to horizon (truncated=True), bootstrap with V(last_state)
    to reduce bias. Otherwise, return sum of rewards.
    
    Args:
        rewards: List of rewards for the episode
        last_obs: Last observation (state) before episode ended
        model: RL model (PPO or DDPG) with value function
        gamma: Discount factor
        truncated: Whether episode ended due to horizon (True) or termination (False)
    
    Returns:
        Bootstrapped return: sum(rewards) + gamma * V(last_state) if truncated, else sum(rewards)
    """
    total_reward = sum(rewards)
    
    if truncated and model is not None:
        try:
            # Get value function estimate for last state
            # For PPO: use policy.value() or extract from model
            # For DDPG: use critic network
            import torch
            
            # Convert observation to tensor
            if isinstance(last_obs, np.ndarray):
                obs_tensor = torch.from_numpy(last_obs).float()
            else:
                obs_tensor = torch.tensor(last_obs, dtype=torch.float32)
            
            # Handle batch dimension
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            with torch.no_grad():
                # Get device from model
                device = model.device if hasattr(model, 'device') else torch.device('cpu')
                obs_tensor = obs_tensor.to(device)
                
                if hasattr(model, 'policy') and hasattr(model.policy, 'predict_values'):
                    # PPO: use policy's value function
                    value_tensor = model.policy.predict_values(obs_tensor)
                    value = value_tensor.item() if value_tensor.numel() == 1 else value_tensor[0].item()
                elif hasattr(model, 'critic'):
                    # DDPG: use critic network (Q-function, need to get action too)
                    # For DDPG, we need action - use actor to get action, then critic
                    if hasattr(model, 'actor'):
                        action = model.actor(obs_tensor)
                        value_tensor = model.critic(obs_tensor, action)
                        value = value_tensor.item() if value_tensor.numel() == 1 else value_tensor[0].item()
                    else:
                        # Fallback: just use critic with zero action (not ideal but better than nothing)
                        action_shape = model.critic.observation_space.shape if hasattr(model.critic, 'observation_space') else (obs_tensor.shape[-1],)
                        zero_action = torch.zeros((obs_tensor.shape[0], action_shape[0] if len(action_shape) > 0 else 1), device=device)
                        value_tensor = model.critic(obs_tensor, zero_action)
                        value = value_tensor.item() if value_tensor.numel() == 1 else value_tensor[0].item()
                elif hasattr(model, 'predict_values'):
                    # Direct value prediction method
                    value_tensor = model.predict_values(obs_tensor)
                    value = value_tensor.item() if value_tensor.numel() == 1 else value_tensor[0].item()
                else:
                    # No value function available
                    value = 0.0
            
            # Bootstrap: add discounted value estimate
            bootstrapped_return = total_reward + gamma * value
            return float(bootstrapped_return)
        except Exception as e:
            # If value function access fails, return sum of rewards
            # This is fine - bootstrapping is only for tracking, not training
            # Don't log to avoid spam, but return sum of rewards
            return total_reward
    
    return total_reward


class RewardCallback(BaseCallback):
    """Callback to track episode rewards during training."""
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_reward_sums = {}  # Track per-environment reward sums
        self.episode_lengths_track = {}  # Track per-environment episode lengths
        self.episode_reward_lists = {}  # Track rewards per environment for bootstrapping
        self.last_observations = {}  # Track last observations for bootstrapping
        
    def _on_training_start(self) -> None:
        # Initialize tracking when training starts
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_reward_sums = {}
        self.episode_lengths_track = {}
        # Get number of environments
        if hasattr(self.training_env, 'num_envs'):
            self.n_envs = self.training_env.num_envs
        elif hasattr(self.training_env, 'envs'):
            self.n_envs = len(self.training_env.envs)
        else:
            self.n_envs = 1
        
        # Initialize tracking for each environment
        for env_idx in range(self.n_envs):
            self.episode_reward_sums[env_idx] = 0.0
            self.episode_lengths_track[env_idx] = 0
            self.episode_reward_lists[env_idx] = []
            self.last_observations[env_idx] = None
    
    def _on_step(self) -> bool:
        # Track rewards from infos (Monitor wrapper provides episode info)
        infos = self.locals.get('infos', [])
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])
        observations = self.locals.get('new_obs', None)  # Get observations for bootstrapping
        
        # First, check if episode info is available from Monitor wrapper
        has_episode_info = False
        for i, info in enumerate(infos):
            if isinstance(info, dict) and 'episode' in info:
                episode_info = info['episode']
                if episode_info is not None:
                    self.episode_rewards.append(episode_info['r'])
                    self.episode_lengths.append(episode_info['l'])
                    has_episode_info = True
        
        # If no episode info from Monitor, manually track by accumulating step rewards
        # This handles cases where Monitor wrapper isn't used or episode info isn't populated
        if not has_episode_info and len(rewards) > 0:
            # Handle both array and list formats
            if isinstance(rewards, np.ndarray):
                rewards_list = rewards.flatten().tolist()
            else:
                rewards_list = rewards if isinstance(rewards, list) else [rewards]
            
            dones_list = []
            if len(dones) > 0:
                if isinstance(dones, np.ndarray):
                    dones_list = dones.flatten().tolist()
                else:
                    dones_list = dones if isinstance(dones, list) else [dones]
            
            # Get truncated flags if available (gymnasium API)
            truncated_list = []
            if 'truncated' in self.locals:
                truncated = self.locals.get('truncated', [])
                if len(truncated) > 0:
                    if isinstance(truncated, np.ndarray):
                        truncated_list = truncated.flatten().tolist()
                    else:
                        truncated_list = truncated if isinstance(truncated, list) else [truncated]
            
            # Track rewards for each environment
            n_track = min(len(rewards_list), len(dones_list) if len(dones_list) > 0 else len(rewards_list), self.n_envs)
            for env_idx in range(n_track):
                reward_val = float(rewards_list[env_idx])
                # Accumulate reward for this environment
                if env_idx not in self.episode_reward_sums:
                    self.episode_reward_sums[env_idx] = 0.0
                    self.episode_lengths_track[env_idx] = 0
                    self.episode_reward_lists[env_idx] = []
                    self.last_observations[env_idx] = None
                
                self.episode_reward_sums[env_idx] += reward_val
                self.episode_reward_lists[env_idx].append(reward_val)
                self.episode_lengths_track[env_idx] += 1
                
                # Store last observation for bootstrapping
                if observations is not None:
                    if isinstance(observations, np.ndarray):
                        if len(observations.shape) > 1:
                            self.last_observations[env_idx] = observations[env_idx].copy()
                        else:
                            self.last_observations[env_idx] = observations.copy()
                    else:
                        self.last_observations[env_idx] = observations[env_idx] if env_idx < len(observations) else None
                
                # If episode is done, compute bootstrapped return if truncated
                if len(dones_list) > env_idx and dones_list[env_idx]:
                    # Check if episode ended due to truncation (horizon) vs termination
                    is_truncated = False
                    if len(truncated_list) > env_idx:
                        is_truncated = bool(truncated_list[env_idx])
                    elif len(infos) > env_idx and isinstance(infos[env_idx], dict):
                        # Try to get truncated flag from info dict
                        is_truncated = infos[env_idx].get('TimeLimit.truncated', False)
                    
                    # Compute return with bootstrap if truncated
                    # NOTE: Bootstrapping is only for tracking/plotting, not for training
                    # SB3 handles rewards internally, so we just track the sum for visualization
                    if is_truncated and self.last_observations[env_idx] is not None:
                        # Try to get model for bootstrapping (optional, fallback to sum if fails)
                        model = getattr(self, 'model', None)
                        if model is None:
                            # Try to get from training environment
                            try:
                                if hasattr(self.training_env, 'get_attr'):
                                    model = self.training_env.get_attr('model', [0])[0]
                            except Exception:
                                model = None
                        
                        # Get gamma from model or use default
                        gamma = 0.99
                        if model is not None and hasattr(model, 'gamma'):
                            gamma = model.gamma
                        
                        # Compute bootstrapped return (for tracking only)
                        try:
                            bootstrapped_return = compute_bootstrapped_return(
                                rewards=self.episode_reward_lists[env_idx],
                                last_obs=self.last_observations[env_idx],
                                model=model,
                                gamma=gamma,
                                truncated=True
                            )
                            self.episode_rewards.append(bootstrapped_return)
                        except Exception as e:
                            # If bootstrapping fails, just use sum of rewards (this is fine)
                            # This doesn't affect training, only affects reward tracking for plots
                            self.episode_rewards.append(self.episode_reward_sums[env_idx])
                    else:
                        # Terminal episode: use sum of rewards
                        self.episode_rewards.append(self.episode_reward_sums[env_idx])
                    
                    self.episode_lengths.append(self.episode_lengths_track[env_idx])
                    # Reset tracking for this environment
                    self.episode_reward_sums[env_idx] = 0.0
                    self.episode_lengths_track[env_idx] = 0
                    self.episode_reward_lists[env_idx] = []
                    self.last_observations[env_idx] = None
        
        return True
    
    def get_episode_rewards(self):
        """Get all episode rewards collected so far."""
        return self.episode_rewards.copy()


def train_and_evaluate_joint(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    target_classes: Tuple[int, ...] = None,
    device: str = "cpu",
    # Joint training parameters (optimized defaults for simple classifiers)
    episodes: int = 60,
    steps_per_episode: int = 90,
    classifier_epochs_per_round: int = 2,  # Default: 2 epochs per update (fewer for simple classifiers)
    classifier_update_every: int = 3,  # Default: update every 3 episodes (allows more RL episodes between updates)
    # Classifier training parameters
    classifier_type: str = "dnn",  # "dnn", "random_forest", or "gradient_boosting"
    classifier_lr: float = 1e-3,
    classifier_batch_size: int = 256,
    classifier_patience: int = 5,
    # Random Forest specific parameters (only used if classifier_type="random_forest")
    rf_n_estimators: int = 100,
    rf_max_depth: int = 10,
    rf_min_samples_split: int = 2,
    rf_min_samples_leaf: int = 1,
    # Gradient Boosting specific parameters (only used if classifier_type="gradient_boosting")
    gb_n_estimators: int = 100,
    gb_max_depth: int = 5,
    gb_learning_rate: float = 0.1,
    gb_min_samples_split: int = 2,
    gb_min_samples_leaf: int = 1,
    # RL training parameters
    use_continuous_actions: bool = False,  # Use continuous actions (DDPG/TD3) instead of PPO (discrete)
    continuous_algorithm: str = "ddpg",  # "ddpg" or "td3" (only used if use_continuous_actions=True)
    n_envs: int = 4,
    learning_rate: float = 3e-4,  # Default for PPO (discrete actions)
    continuous_learning_rate: float = 5e-5,  # Lower LR for TD3/DDPG (continuous actions) for stability
    n_steps: int = None,  # Will default to steps_per_episode
    batch_size: int = 64,
    n_epochs: int = 10,
    # Environment parameters
    use_perturbation: bool = True,
    perturbation_mode: str = "bootstrap",
    n_perturb: int = 1024,
    step_fracs: Tuple[float, ...] = (0.005, 0.01, 0.02),
    min_width: float = 0.05,
    precision_target: float = 0.95,
    coverage_target: float = 0.02,
    # Evaluation parameters
    n_eval_instances_per_class: int = 20,
    max_features_in_rule: int = 5,
    eval_steps_per_episode: int = None,  # Defaults to steps_per_episode if None
    num_rollouts_per_instance: int = 1,  # Number of greedy rollouts per instance (default: 1)
    # Output parameters
    output_dir: str = "./output/joint/",
    save_checkpoints: bool = True,
    checkpoint_freq: int = 10000,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Joint training pipeline: Interleaved classifier and RL training (matching dyn_anchor_PPO.py).
    
    This function:
    1. Prepares data and initializes classifier
    2. Trains RL every episode, updates classifier every N episodes (matching dyn_anchor_PPO.py)
    3. Evaluates on test instances with frozen policy at the end
    4. Returns results and trained models
    
    Note: This matches the behavior of dyn_anchor_PPO.py where:
    - RL training happens every episode
    - Classifier updates happen every classifier_update_every episodes
    - Classifier is trained for classifier_epochs_per_round epochs each update
    
    Args:
        X_train: Training features (will be standardized)
        y_train: Training labels
        X_test: Test features (for evaluation)
        y_test: Test labels
        feature_names: Names of features
        target_classes: Classes to generate anchors for (None = all classes)
        device: Device to use ("cpu" or "cuda")
        episodes: Number of RL episodes (matching dyn_anchor_PPO.py)
        steps_per_episode: Steps per episode
        classifier_epochs_per_round: Classifier epochs when updating (default: 2, fewer epochs for simple classifiers)
        classifier_update_every: Update classifier every N episodes (default: 3, allows more RL episodes between updates)
        classifier_lr: Classifier learning rate
        classifier_batch_size: Classifier batch size
        classifier_patience: Classifier early stopping patience
        n_envs: Number of parallel RL environments
        learning_rate: RL learning rate
        n_steps: RL steps per environment before update
        batch_size: RL batch size for updates
        n_epochs: RL epochs per update
        use_perturbation: Enable perturbation sampling
        perturbation_mode: "bootstrap" or "uniform" sampling
        n_perturb: Number of perturbation samples
        step_fracs: Action step sizes
        min_width: Minimum box width
        precision_target: Target precision threshold
        coverage_target: Target coverage threshold
        n_eval_instances_per_class: Instances per class for evaluation
        max_features_in_rule: Max features to show in rules
        eval_steps_per_episode: Max steps for greedy rollouts (defaults to steps_per_episode)
        num_rollouts_per_instance: Number of greedy rollouts per instance (default: 1)
                                  If > 1, metrics are averaged across rollouts for each instance
        output_dir: Directory for outputs
        save_checkpoints: Save checkpoints during training
        checkpoint_freq: Checkpoint frequency
        verbose: Verbosity level
    
    Returns:
        Dictionary with:
            - trained_model: PPO model (frozen)
            - classifier: Trained classifier (frozen)
            - trainer: Trainer instance
            - eval_results: Per-class evaluation results
            - overall_stats: Overall precision/coverage
            - metadata: Configuration and setup info
            - joint_training_history: History of training metrics per round
    """
    from trainers.networks import SimpleClassifier, UnifiedClassifier
    from trainers.vecEnv import AnchorEnv, make_dummy_vec_env, DummyVecEnv, make_dynamic_anchor_env, ContinuousAnchorEnv
    from trainers.PPO_trainer import train_ppo_model, DynamicAnchorPPOTrainer
    from trainers.DDPG_trainer import DynamicAnchorDDPGTrainer, create_ddpg_trainer
    from trainers.TD3_trainer import DynamicAnchorTD3Trainer, create_td3_trainer
    from trainers.dynAnchors_inference import evaluate_all_classes
    from trainers.device_utils import get_device_pair
    
    # Standardize device handling: get both object and string
    device_obj, device_str = get_device_pair(device)
    
    # Prepare data
    print("\n" + "="*80)
    print("JOINT TRAINING: Classifier + RL Policy")
    print("="*80)
    print("\nPreparing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # Normalize to [0,1] for environment
    X_min = X_train_scaled.min(axis=0)
    X_max = X_train_scaled.max(axis=0)
    X_range = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))
    X_unit_train = (X_train_scaled - X_min) / X_range
    X_unit_test = (X_test_scaled - X_min) / X_range
    
    # Determine target classes
    unique_classes = np.unique(y_train)
    if target_classes is None:
        target_classes = tuple(unique_classes)
    else:
        target_classes = tuple(target_classes)
    
    print(f"Classes: {unique_classes}, Target classes: {target_classes}")
    n_features = X_train.shape[1]
    n_classes = len(unique_classes)
    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    
    # Analyze dataset complexity and recommend ideal training parameters
    def analyze_dataset_complexity(n_train, n_features, n_classes):
        """
        Analyze dataset complexity and recommend ideal training parameters.
        
        Args:
            n_train: Number of training samples
            n_features: Number of features
            n_classes: Number of classes
        
        Returns:
            Dictionary with recommended parameters and analysis
        """
        # Classify dataset size
        if n_train < 500:
            size_category = "small"
            size_factor = 0.5
        elif n_train < 5000:
            size_category = "medium"
            size_factor = 1.0
        elif n_train < 50000:
            size_category = "large"
            size_factor = 1.5
        else:
            size_category = "very_large"
            size_factor = 2.0
        
        # Classify feature complexity
        if n_features < 10:
            feature_category = "low"
            feature_factor = 0.8
        elif n_features < 50:
            feature_category = "medium"
            feature_factor = 1.0
        elif n_features < 200:
            feature_category = "high"
            feature_factor = 1.3
        else:
            feature_category = "very_high"
            feature_factor = 1.5
        
        # Classify class complexity
        if n_classes == 2:
            class_category = "binary"
            class_factor = 0.9
        elif n_classes <= 5:
            class_category = "multi_class"
            class_factor = 1.0
        else:
            class_category = "many_class"
            class_factor = 1.2
        
        # Overall complexity score
        complexity_score = size_factor * feature_factor * class_factor
        
        # Recommend parameters based on complexity
        if complexity_score < 0.6:
            # Very simple dataset
            recommended_episodes = max(15, int(20 * complexity_score))
            recommended_steps = max(30, int(40 * complexity_score))
            recommended_classifier_epochs = 1
            recommended_classifier_update_every = 2
            recommended_n_envs = 2
        elif complexity_score < 1.0:
            # Simple dataset
            recommended_episodes = max(25, int(35 * complexity_score))
            recommended_steps = max(40, int(50 * complexity_score))
            recommended_classifier_epochs = 1
            recommended_classifier_update_every = 2
            recommended_n_envs = 2
        elif complexity_score < 1.5:
            # Medium complexity
            recommended_episodes = max(40, int(50 * complexity_score))
            recommended_steps = max(60, int(70 * complexity_score))
            recommended_classifier_epochs = 2
            recommended_classifier_update_every = 3
            recommended_n_envs = 2
        elif complexity_score < 2.5:
            # High complexity
            recommended_episodes = max(60, int(70 * complexity_score))
            recommended_steps = max(90, int(100 * complexity_score))
            recommended_classifier_epochs = 2
            recommended_classifier_update_every = 4
            recommended_n_envs = 4
        else:
            # Very high complexity
            recommended_episodes = max(80, int(100 * complexity_score))
            recommended_steps = max(120, int(150 * complexity_score))
            recommended_classifier_epochs = 3
            recommended_classifier_update_every = 5
            recommended_n_envs = 4
        
        # Calculate total training budget
        total_rl_timesteps = recommended_episodes * recommended_steps * recommended_n_envs
        total_classifier_epochs = (recommended_episodes // recommended_classifier_update_every) * recommended_classifier_epochs
        
        return {
            "size_category": size_category,
            "feature_category": feature_category,
            "class_category": class_category,
            "complexity_score": complexity_score,
            "recommended_episodes": recommended_episodes,
            "recommended_steps": recommended_steps,
            "recommended_classifier_epochs": recommended_classifier_epochs,
            "recommended_classifier_update_every": recommended_classifier_update_every,
            "recommended_n_envs": recommended_n_envs,
            "total_rl_timesteps": total_rl_timesteps,
            "total_classifier_epochs": total_classifier_epochs,
        }
    
    # Analyze dataset and get recommendations
    complexity_analysis = analyze_dataset_complexity(n_train_samples, n_features, n_classes)
    
    # Log ideal training parameters
    print("\n" + "="*80)
    print("DATASET COMPLEXITY ANALYSIS & IDEAL TRAINING PARAMETERS")
    print("="*80)
    print(f"\nDataset Characteristics:")
    print(f"  Training samples: {n_train_samples:,}")
    print(f"  Test samples: {n_test_samples:,}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_classes}")
    print(f"\nComplexity Classification:")
    print(f"  Size: {complexity_analysis['size_category']} ({n_train_samples:,} samples)")
    print(f"  Features: {complexity_analysis['feature_category']} ({n_features} features)")
    print(f"  Classes: {complexity_analysis['class_category']} ({n_classes} classes)")
    print(f"  Overall Complexity Score: {complexity_analysis['complexity_score']:.2f}")
    print(f"\nRecommended Training Parameters (based on dataset complexity):")
    print(f"  Episodes: {complexity_analysis['recommended_episodes']}")
    print(f"  Steps per episode: {complexity_analysis['recommended_steps']}")
    print(f"  Classifier epochs per update: {complexity_analysis['recommended_classifier_epochs']}")
    print(f"  Classifier update every: {complexity_analysis['recommended_classifier_update_every']} episode(s)")
    print(f"  Parallel environments: {complexity_analysis['recommended_n_envs']}")
    print(f"  Total RL timesteps: {complexity_analysis['total_rl_timesteps']:,}")
    print(f"  Total classifier epochs: {complexity_analysis['total_classifier_epochs']}")
    print(f"\nCurrent Training Parameters:")
    print(f"  Episodes: {episodes}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Classifier epochs per update: {classifier_epochs_per_round}")
    print(f"  Classifier update every: {classifier_update_every} episode(s)")
    print(f"  Parallel environments: {n_envs}")
    current_total_rl_timesteps = episodes * steps_per_episode * n_envs
    current_total_classifier_epochs = (episodes // classifier_update_every) * classifier_epochs_per_round
    print(f"  Total RL timesteps: {current_total_rl_timesteps:,}")
    print(f"  Total classifier epochs: {current_total_classifier_epochs}")
    
    # Check if current training is sufficient
    print(f"\nTraining Sufficiency Analysis:")
    episodes_ratio = episodes / complexity_analysis['recommended_episodes']
    steps_ratio = steps_per_episode / complexity_analysis['recommended_steps']
    total_timesteps_ratio = current_total_rl_timesteps / complexity_analysis['total_rl_timesteps']
    
    if episodes_ratio >= 0.9 and steps_ratio >= 0.9 and total_timesteps_ratio >= 0.9:
        sufficiency_status = "✓ SUFFICIENT"
        sufficiency_color = "green"
        recommendation = "Current training parameters are adequate."
    elif episodes_ratio >= 0.7 and steps_ratio >= 0.7 and total_timesteps_ratio >= 0.7:
        sufficiency_status = "⚠ MARGINAL"
        sufficiency_color = "yellow"
        recommendation = "Consider increasing training for better convergence."
    else:
        sufficiency_status = "✗ INSUFFICIENT"
        sufficiency_color = "red"
        recommendation = "Training may be insufficient. Consider increasing episodes or steps per episode."
    
    print(f"  Status: {sufficiency_status}")
    print(f"  Episodes ratio: {episodes_ratio:.2%} (current: {episodes}, recommended: {complexity_analysis['recommended_episodes']})")
    print(f"  Steps ratio: {steps_ratio:.2%} (current: {steps_per_episode}, recommended: {complexity_analysis['recommended_steps']})")
    print(f"  Total timesteps ratio: {total_timesteps_ratio:.2%} (current: {current_total_rl_timesteps:,}, recommended: {complexity_analysis['total_rl_timesteps']:,})")
    print(f"  Recommendation: {recommendation}")
    print("="*80)
    
    # Initialize classifier (using UnifiedClassifier which supports DNN, RF, and GB)
    print("\n" + "="*80)
    print(f"Initializing Classifier (type: {classifier_type})")
    print("="*80)
    
    clf_type_lower = classifier_type.lower()
    if clf_type_lower == "random_forest":
        classifier = UnifiedClassifier(
            classifier_type="random_forest",
            device=device_str,
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
        # Train Random Forest initially
        print("  Training Random Forest classifier...")
        classifier.fit(X_train_scaled, y_train)
        train_preds = classifier.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_preds)
        print(f"  Random Forest train accuracy: {train_acc:.3f}")
        
        # For Random Forest, we don't need optimizer/criterion
        clf_optimizer = None
        criterion = None
    elif clf_type_lower == "gradient_boosting":
        classifier = UnifiedClassifier(
            classifier_type="gradient_boosting",
            device=device_str,
            n_estimators=gb_n_estimators,
            max_depth=gb_max_depth,
            learning_rate=gb_learning_rate,
            min_samples_split=gb_min_samples_split,
            min_samples_leaf=gb_min_samples_leaf,
            random_state=42,
        )
        # Train Gradient Boosting initially
        print("  Training Gradient Boosting classifier...")
        classifier.fit(X_train_scaled, y_train)
        train_preds = classifier.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_preds)
        print(f"  Gradient Boosting train accuracy: {train_acc:.3f}")
        
        # For Gradient Boosting, we don't need optimizer/criterion
        clf_optimizer = None
        criterion = None
    else:  # DNN
        classifier = UnifiedClassifier(
            classifier_type="dnn",
            input_dim=n_features,
            num_classes=n_classes,
            device=device_str,
        )
        classifier.to(device_obj)
        
        # Initialize optimizer and loss for DNN
        clf_optimizer = optim.Adam(classifier.parameters(), lr=classifier_lr)
        criterion = nn.CrossEntropyLoss()
    
    # Training history
    joint_training_history = []
    
    # Create environment factory function
    def create_anchor_env(target_cls=None):
        """Helper to create AnchorEnv with a specific target class and current classifier."""
        if target_cls is None:
            target_cls = target_classes[0]
        return AnchorEnv(
            X_unit=X_unit_train,
            X_std=X_train_scaled,
            y=y_train,
            feature_names=feature_names,
            classifier=classifier,  # Use current classifier (will be updated)
            device=device_str,
            target_class=target_cls,
            step_fracs=step_fracs,
            min_width=min_width,
            alpha=0.5,  # Reduced from 0.7 to balance with coverage
            beta=1.0,   # Increased from 0.6 to boost coverage learning
            gamma=0.1,
            precision_target=precision_target,
            coverage_target=coverage_target,
            precision_blend_lambda=0.5,
            drift_penalty_weight=0.05,
            use_perturbation=use_perturbation,
            perturbation_mode=perturbation_mode,
            n_perturb=n_perturb,
            X_min=X_min,
            X_range=X_range,
            min_coverage_floor=0.005,
            js_penalty_weight=0.01,  # Reduced from 0.05 to 0.01 to prevent penalty from dominating rewards
        )
    
    # Create default factory (for evaluation)
    def make_anchor_env():
        return create_anchor_env()
    
    # ======================================================================
    # FIXED INSTANCE SAMPLING: Pre-select instances per class for training
    # This reduces variance by using consistent starting points
    # ======================================================================
    n_fixed_instances_per_class = 10  # Number of fixed instances per class
    rng_fixed = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
    fixed_instances_per_class = {}
    
    print(f"\n[Fixed Instance Sampling] Pre-selecting {n_fixed_instances_per_class} instances per class...")
    for cls in target_classes:
        cls_indices = np.where(y_train == cls)[0]
        if len(cls_indices) > 0:
            n_sample = min(n_fixed_instances_per_class, len(cls_indices))
            fixed_indices = rng_fixed.choice(cls_indices, size=n_sample, replace=False)
            fixed_instances_per_class[cls] = fixed_indices
            print(f"  Class {cls}: Selected {len(fixed_indices)} instances (indices: {fixed_indices[:5].tolist()}...)")
        else:
            fixed_instances_per_class[cls] = np.array([], dtype=int)
            print(f"  Class {cls}: No instances available")
    
    # Store fixed instances in a way accessible to environments
    # We'll pass episode number through seed parameter to cycle through instances
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup single tensorboard log directory for all episodes (for unified logging)
    tensorboard_log_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    # Set n_steps default to steps_per_episode if not provided
    if n_steps is None:
        n_steps = steps_per_episode
    
    # Initialize RL trainer(s) based on action type
    if use_continuous_actions:
        # Continuous actions: Create separate trainers for each class (DDPG/TD3 don't use vectorized envs the same way)
        continuous_algorithm_lower = continuous_algorithm.lower()
        if continuous_algorithm_lower == "td3":
            print(f"\n[Continuous Actions] Using TD3 (Twin Delayed DDPG) for continuous action control")
            print(f"  - Creating TD3 trainers for {len(target_classes)} class(es)")
        else:
            print(f"\n[Continuous Actions] Using DDPG (Stable Baselines 3) for continuous action control")
            print(f"  - Creating DDPG trainers for {len(target_classes)} class(es)")
        
        continuous_trainers = {}
        for cls in target_classes:
            anchor_env = create_anchor_env(target_cls=cls)
            # Set fixed instances for this class (for fixed instance sampling)
            anchor_env.fixed_instances_per_class = fixed_instances_per_class
            # Enable continuous actions in AnchorEnv
            anchor_env.n_actions = 2 * anchor_env.n_features
            anchor_env.max_action_scale = max(step_fracs) if step_fracs else 0.02
            anchor_env.min_absolute_step = max(0.05, min_width * 0.5)
            # Wrap with ContinuousAnchorEnv
            gym_env = ContinuousAnchorEnv(anchor_env, seed=42 + cls)
            
            # Create trainer based on algorithm choice
            if continuous_algorithm_lower == "td3":
                # Create TD3 trainer
                trainer = create_td3_trainer(
                    env=gym_env,
                    policy_type="MlpPolicy",
                    learning_rate=continuous_learning_rate,  # Use parameter (default: 5e-5 for stability)
                    buffer_size=100000,
                    learning_starts=0,
                    batch_size=64,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=(1, "step"),
                    gradient_steps=1,
                    action_noise_sigma=0.3,  # Stronger exploration
                    policy_delay=2,  # TD3-specific: delay policy updates
                    policy_kwargs=dict(net_arch=[256, 256]),
                    verbose=0,
                    device=device_str,
                )
            else:
                # Create DDPG trainer (default)
                trainer = create_ddpg_trainer(
                    env=gym_env,
                    policy_type="MlpPolicy",
                    learning_rate=continuous_learning_rate,  # Use parameter (default: 5e-5 for stability)
                    buffer_size=100000,
                    learning_starts=0,
                    batch_size=64,
                    tau=0.005,
                    gamma=0.99,
                    train_freq=(1, "step"),
                    gradient_steps=1,
                    action_noise_sigma=0.3,  # Stronger exploration
                    policy_kwargs=dict(net_arch=[256, 256]),
                    verbose=0,
                    device=device_str,
                )
            continuous_trainers[cls] = trainer
        
        # Keep backward compatibility: ddpg_trainers for existing code
        ddpg_trainers = continuous_trainers
        rl_trainer = None  # Not used for continuous actions
        vec_env = None  # Not used for continuous actions
    else:
        # PPO: Create vectorized environment
        print(f"\n[Discrete Actions] Using PPO (Stable Baselines 3) for discrete action control")
        if len(target_classes) > 1 and n_envs > 1:
            env_fns = []
            for i in range(n_envs):
                target_cls = target_classes[i % len(target_classes)]
                factory_fn = partial(create_anchor_env, target_cls)
                env_fns.append(lambda i=i, f=factory_fn, s=42+i: make_dynamic_anchor_env(f, seed=s))
            vec_env = DummyVecEnv(env_fns)
        else:
            from trainers.vecEnv import make_dummy_vec_env
            vec_env = make_dummy_vec_env(make_anchor_env, n_envs=n_envs, seed=42)
        rl_trainer = None  # Will be created on first episode
        ddpg_trainers = None  # Not used for PPO
    
    # Joint training loop (matching dyn_anchor_PPO.py)
    print("\n" + "="*80)
    print("JOINT TRAINING LOOP (matching dyn_anchor_PPO.py)")
    print("="*80)
    print(f"Number of episodes: {episodes}")
    print(f"  - Steps per episode: {steps_per_episode}")
    print(f"  - Classifier epochs per update: {classifier_epochs_per_round}")
    print(f"  - Classifier update every: {classifier_update_every} episode(s)")
    if use_continuous_actions:
        algorithm_name = continuous_algorithm.upper() if continuous_algorithm.lower() == "td3" else "DDPG"
        print(f"  - RL algorithm: {algorithm_name} (continuous actions)")
    else:
        print(f"  - RL algorithm: PPO (discrete actions)")
        print(f"  - RL n_steps per update: {n_steps}")
    print(f"  - TensorBoard logs: {tensorboard_log_dir}")
    
    # Training history
    test_acc_history = []
    episode_rewards_history = []  # Track rewards per episode
    per_class_precision_history = {cls: [] for cls in target_classes}  # Track precision per class per episode
    per_class_coverage_history = {cls: [] for cls in target_classes}  # Track coverage per class per episode
    
    # Initialize classifier metrics to preserve across episodes
    last_loss = 0.0
    last_train_acc = 0.0
    
    for ep in range(episodes):
        print(f"\n{'='*80}")
        print(f"EPISODE {ep + 1}/{episodes}")
        print(f"{'='*80}")
        
        # ======================================================================
        # STEP 1: Train Classifier (if it's time to update)
        # ======================================================================
        if (ep % max(1, int(classifier_update_every))) == 0:
            print(f"\n[Episode {ep + 1}] Training Classifier...")
            
            clf_type_lower = classifier_type.lower()
            if clf_type_lower == "random_forest":
                # Random Forest: Retrain from scratch
                print("  Retraining Random Forest classifier...")
                classifier.fit(X_train_scaled, y_train)
                
                # Compute training accuracy
                train_preds = classifier.predict(X_train_scaled)
                last_train_acc = accuracy_score(y_train, train_preds)
                last_loss = None  # Random Forest doesn't have loss
                
                if verbose >= 1:
                    print(f"  [clf] Random Forest retrained | train_acc={last_train_acc:.3f}")
            elif clf_type_lower == "gradient_boosting":
                # Gradient Boosting: Retrain from scratch
                print("  Retraining Gradient Boosting classifier...")
                classifier.fit(X_train_scaled, y_train)
                
                # Compute training accuracy
                train_preds = classifier.predict(X_train_scaled)
                last_train_acc = accuracy_score(y_train, train_preds)
                last_loss = None  # Gradient Boosting doesn't have loss
                
                if verbose >= 1:
                    print(f"  [clf] Gradient Boosting retrained | train_acc={last_train_acc:.3f}")
            else:  # DNN
                # DNN: Gradient-based updates
                classifier.train()
                
                dataset = TensorDataset(
                    torch.from_numpy(X_train_scaled).float(),
                    torch.from_numpy(y_train).long()
                )
                loader = DataLoader(dataset, batch_size=classifier_batch_size, shuffle=True)
                
                last_loss = None
                last_train_acc = None
                
                for e in range(1, classifier_epochs_per_round + 1):
                    classifier.train()
                    epoch_loss_sum = 0.0
                    epoch_correct = 0
                    epoch_count = 0
                    
                    for xb, yb in loader:
                        xb = xb.to(device_obj)
                        yb = yb.to(device_obj)
                        clf_optimizer.zero_grad()
                        logits = classifier(xb)
                        loss = criterion(logits, yb)
                        loss.backward()
                        clf_optimizer.step()
                        
                        with torch.no_grad():
                            preds = logits.argmax(dim=1)
                            correct = (preds == yb).sum().item()
                            epoch_correct += correct
                            epoch_count += yb.size(0)
                            epoch_loss_sum += loss.item() * yb.size(0)
                    
                    last_loss = epoch_loss_sum / max(1, epoch_count)
                    last_train_acc = epoch_correct / max(1, epoch_count)
                    
                    if verbose >= 1:
                        print(f"  [clf] epoch {e}/{classifier_epochs_per_round} | loss={last_loss:.4f} | train_acc={last_train_acc:.3f} | samples={epoch_count}")
            
            classifier.eval()  # Set to eval mode for RL training
        # else: Preserve previous loss and accuracy values when classifier is not updated
        # last_loss and last_train_acc are already set from previous iteration, so no need to reset
        
        # Evaluate classifier on test set
        classifier.eval()
        clf_type_lower = classifier_type.lower()
        if clf_type_lower in ["random_forest", "gradient_boosting"]:
            test_preds = classifier.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, test_preds)
        else:  # DNN
            with torch.no_grad():
                test_logits = classifier(torch.from_numpy(X_test_scaled).float().to(device_obj))
                test_preds = test_logits.argmax(dim=1).cpu().numpy()
                test_acc = accuracy_score(y_test, test_preds)
        test_acc_history.append(test_acc)
        
        # ======================================================================
        # STEP 2: Train RL Policy (every episode, matching dyn_anchor_PPO.py)
        # ======================================================================
        print(f"\n[Episode {ep + 1}] Training RL Policy...")
        
        if use_continuous_actions:
            # Continuous actions (DDPG/TD3): Manual training loop (collect experiences, add to replay buffer, train)
            # Calculate timesteps for this episode: steps_per_episode * number of classes
            # (Continuous action algorithms run one episode per class, not using vectorized envs)
            timesteps_per_episode = steps_per_episode * len(target_classes)
            
            episode_rewards_per_class = {}
            training_metrics_per_class = {}  # Store training metrics for each class
            for cls in target_classes:
                # Update environment with current classifier
                anchor_env = create_anchor_env(target_cls=cls)
                # Set fixed instances for this class (for fixed instance sampling)
                anchor_env.fixed_instances_per_class = fixed_instances_per_class
                anchor_env.n_actions = 2 * anchor_env.n_features
                anchor_env.max_action_scale = max(step_fracs) if step_fracs else 0.02
                anchor_env.min_absolute_step = max(0.05, min_width * 0.5)
                # Pass episode number in seed to cycle through fixed instances
                # Seed format: 42 + cls + ep (episode number allows cycling)
                gym_env = ContinuousAnchorEnv(anchor_env, seed=42 + cls + ep)
                ddpg_trainer = ddpg_trainers[cls]
                ddpg_trainer.env = gym_env  # Update environment
                
                # Run episode: collect experiences and train
                try:
                    import gymnasium as gym
                    GYM_VERSION = "gymnasium"
                except ImportError:
                    try:
                        import gym
                        GYM_VERSION = "gym"
                    except ImportError:
                        GYM_VERSION = "gym"
                
                try:
                    if GYM_VERSION == "gymnasium":
                        obs, _ = gym_env.reset(seed=42 + cls + ep)
                    else:
                        obs = gym_env.reset()
                    
                    episode_reward = 0.0
                    # Track reward components for debugging
                    reward_components = {
                        "precision_gain": 0.0,
                        "coverage_gain": 0.0,
                        "coverage_bonus": 0.0,
                        "target_class_bonus": 0.0,
                        "overlap_penalty": 0.0,
                        "drift_penalty": 0.0,
                        "anchor_drift_penalty": 0.0,
                        "js_penalty": 0.0,
                        "total_reward": 0.0,
                        "final_precision": 0.0,
                        "final_coverage": 0.0,
                        "final_n_points": 0,
                        "n_steps": 0,  # Track actual number of steps
                    }
                    
                    for t in range(steps_per_episode):
                        # Get action from DDPG trainer
                        action, _ = ddpg_trainer.predict(obs, deterministic=False)
                        
                        # Step environment
                        if GYM_VERSION == "gymnasium":
                            next_obs, reward, terminated, truncated, step_info = gym_env.step(action)
                            done = terminated or truncated
                        else:
                            step_result = gym_env.step(action)
                            if len(step_result) == 5:
                                next_obs, reward, terminated, truncated, step_info = step_result
                                done = terminated or truncated
                            else:
                                next_obs, reward, done, step_info = step_result
                        
                        episode_reward += reward
                        
                        # Track reward components from step_info (if available)
                        # Note: step_info may have all components from AnchorEnv.step()
                        if isinstance(step_info, dict):
                            # Update final metrics (from last step)
                            if "precision" in step_info:
                                reward_components["final_precision"] = step_info["precision"]
                            if "coverage" in step_info:
                                reward_components["final_coverage"] = step_info["coverage"]
                            if "n_points" in step_info:
                                reward_components["final_n_points"] = step_info["n_points"]
                            
                            # Accumulate reward components (sum over all steps in episode)
                            # Use raw gains for logging (not weighted components) to see actual improvements
                            if "precision_gain" in step_info:
                                reward_components["precision_gain"] += step_info["precision_gain"]
                            if "coverage_gain" in step_info:
                                reward_components["coverage_gain"] += step_info["coverage_gain"]
                            if "coverage_bonus" in step_info:
                                reward_components["coverage_bonus"] += step_info["coverage_bonus"]
                            if "target_class_bonus" in step_info:
                                reward_components["target_class_bonus"] += step_info["target_class_bonus"]
                            if "overlap_penalty" in step_info:
                                reward_components["overlap_penalty"] += abs(step_info["overlap_penalty"])
                            if "drift_penalty" in step_info:
                                reward_components["drift_penalty"] += abs(step_info["drift_penalty"])
                            if "anchor_drift_penalty" in step_info:
                                reward_components["anchor_drift_penalty"] += abs(step_info["anchor_drift_penalty"])
                            if "js_penalty" in step_info:
                                reward_components["js_penalty"] += abs(step_info["js_penalty"])
                        
                        # Add to replay buffer and train
                        ddpg_trainer.add_to_replay_buffer(
                            obs=obs,
                            next_obs=next_obs,
                            action=action,
                            reward=reward,
                            done=done,
                            info=step_info
                        )
                        
                        # Train DDPG if enough samples
                        trained = ddpg_trainer.train_step(gradient_steps=1)
                        
                        obs = next_obs
                        reward_components["n_steps"] += 1
                        if done:
                            break
                    
                    reward_components["total_reward"] = episode_reward
                    episode_rewards_per_class[cls] = episode_reward
                    
                    # Store training metrics for this class (to use when evaluation doesn't run)
                    # These are the final metrics from the training episode
                    training_metrics_per_class[cls] = {
                        "precision": reward_components['final_precision'],
                        "hard_precision": reward_components['final_precision'],  # Use final precision as hard_precision
                        "coverage": reward_components['final_coverage'],
                        "n_points": reward_components['final_n_points'],
                    }
                    
                    # Print detailed reward breakdown for debugging (every 10 episodes or when verbose >= 2)
                    if verbose >= 2 or (ep % 10 == 0 and verbose >= 1):
                        # Calculate average per-step values
                        n_steps_actual = reward_components["n_steps"]
                        avg_js_per_step = reward_components['js_penalty'] / max(1, n_steps_actual)
                        avg_drift_per_step = reward_components['drift_penalty'] / max(1, n_steps_actual)
                        
                        # Use correct algorithm name for logging
                        algo_name = continuous_algorithm.upper() if use_continuous_actions else "PPO"
                        print(f"    [{algo_name} cls={cls}] Reward breakdown:")
                        print(f"      Total reward: {episode_reward:.6f} (over {n_steps_actual} steps)")
                        print(f"      Final precision: {reward_components['final_precision']:.6f} | "
                              f"Final coverage: {reward_components['final_coverage']:.6f} | "
                              f"Final n_points: {reward_components['final_n_points']}")
                        print(f"      Gains: precision={reward_components['precision_gain']:.6f}, "
                              f"coverage={reward_components['coverage_gain']:.6f}")
                        print(f"      Bonuses: coverage={reward_components['coverage_bonus']:.6f}, "
                              f"target_class={reward_components['target_class_bonus']:.6f}")
                        print(f"      Penalties (total): overlap={reward_components['overlap_penalty']:.6f}, "
                              f"drift={reward_components['drift_penalty']:.6f}, "
                              f"anchor_drift={reward_components['anchor_drift_penalty']:.6f}, "
                              f"js={reward_components['js_penalty']:.6f}")
                        print(f"      Penalties (per-step avg): drift={avg_drift_per_step:.6f}, "
                              f"js={avg_js_per_step:.6f}")
                        if avg_js_per_step > 0.04:
                            print(f"      ⚠ WARNING: JS penalty per step is high ({avg_js_per_step:.6f}) - box is changing too much!")
                        if reward_components['precision_gain'] == 0.0 and reward_components['coverage_gain'] == 0.0:
                            print(f"      ⚠ WARNING: No precision/coverage gains - agent isn't improving the box!")
                finally:
                    # Cleanup: close environment
                    try:
                        gym_env.close()
                    except:
                        pass
            
            # Average reward across classes
            avg_reward = np.mean(list(episode_rewards_per_class.values()))
            episode_rewards_history.append(avg_reward)
            if verbose >= 1:
                # Use correct algorithm name for logging
                algo_name = continuous_algorithm.upper() if use_continuous_actions else "PPO"
                print(f"  [{algo_name}] Episode rewards: {episode_rewards_per_class}, avg={avg_reward:.6f}")
        else:
            # PPO: Use vectorized environment and SB3's learn() method
            # Update environment with current classifier (classifier may have changed)
            old_vec_env = None
            if rl_trainer is not None and hasattr(rl_trainer, 'vec_env'):
                old_vec_env = rl_trainer.vec_env
                try:
                    old_vec_env.close()
                except:
                    pass
            
            # Create new vectorized environment with updated classifier
            try:
                if len(target_classes) > 1 and n_envs > 1:
                    env_fns = []
                    for i in range(n_envs):
                        target_cls = target_classes[i % len(target_classes)]
                        factory_fn = partial(create_anchor_env, target_cls)
                        env_fns.append(lambda i=i, f=factory_fn, s=42+i: make_dynamic_anchor_env(f, seed=s))
                    vec_env = DummyVecEnv(env_fns)
                else:
                    from trainers.vecEnv import make_dummy_vec_env
                    vec_env = make_dummy_vec_env(make_anchor_env, n_envs=n_envs, seed=42)
                
                # Calculate timesteps for this episode: n_steps * n_envs (one rollout per episode)
                timesteps_per_episode = n_steps * n_envs
                
                # Create callback to track rewards
                reward_callback = RewardCallback(verbose=0)
                
                if rl_trainer is None:
                    # First episode: create new trainer
                    episode_output_dir = f"{output_dir}/episode_{ep+1}/"
                    os.makedirs(episode_output_dir, exist_ok=True)
                    
                    # Create trainer manually to pass callback
                    rl_trainer = DynamicAnchorPPOTrainer(
                        vec_env=vec_env,
                        policy_type="MlpPolicy",
                        learning_rate=learning_rate,
                        n_steps=n_steps,
                        batch_size=batch_size,
                        n_epochs=n_epochs,
                        verbose=verbose,
                        tensorboard_log=tensorboard_log_dir,
                        device=device_str,
                    )
                    # Train with callback
                    rl_trainer.learn(
                        total_timesteps=timesteps_per_episode,
                        callback=reward_callback,
                        progress_bar=False,
                        log_interval=10,
                        save_checkpoints=save_checkpoints and (ep % 10 == 0),
                        checkpoint_freq=checkpoint_freq,
                        eval_freq=0,
                    )
                    # Save model
                    final_model_path = f"{episode_output_dir}/ppo_model_final"
                    rl_trainer.save(final_model_path)
                    if verbose >= 1:
                        print(f"Training complete! Model saved to {final_model_path}")
                else:
                    # Continue training existing model - update environment and continue learning
                    rl_trainer.model.set_env(vec_env)
                    rl_trainer.vec_env = vec_env  # Update trainer's vec_env reference
                    # Reset callback for this episode
                    reward_callback = RewardCallback(verbose=0)
                    rl_trainer.learn(
                        total_timesteps=timesteps_per_episode,
                        callback=reward_callback,  # Track rewards
                        progress_bar=False,  # Less verbose for per-episode updates
                        log_interval=10,
                        save_checkpoints=save_checkpoints and (ep % 10 == 0),
                        checkpoint_freq=checkpoint_freq,
                        eval_freq=0,  # No evaluation during joint training
                    )
                
                # Extract episode rewards from callback
                episode_rewards = reward_callback.get_episode_rewards()
                if len(episode_rewards) > 0:
                    # Average reward across all completed episodes in this training step
                    avg_reward = np.mean(episode_rewards)
                    episode_rewards_history.append(avg_reward)
                    if verbose >= 2:
                        print(f"  [DEBUG] Episode rewards: {len(episode_rewards)} episodes, avg={avg_reward:.6f}, "
                              f"min={min(episode_rewards):.6f}, max={max(episode_rewards):.6f}")
                else:
                    # If no episodes completed, try to get total reward from accumulated sums
                    if hasattr(reward_callback, 'episode_reward_sums') and len(reward_callback.episode_reward_sums) > 0:
                        total_reward = sum(reward_callback.episode_reward_sums.values())
                        total_length = sum(reward_callback.episode_lengths_track.values())
                        if total_reward != 0 and total_length > 0:
                            avg_reward_per_env = total_reward / max(1, len(reward_callback.episode_reward_sums))
                            episode_rewards_history.append(avg_reward_per_env)
                        else:
                            episode_rewards_history.append(0.0)
                    else:
                        episode_rewards_history.append(0.0)
            except Exception as e:
                # Ensure cleanup even if training fails
                if 'vec_env' in locals():
                    try:
                        vec_env.close()
                    except:
                        pass
                raise e
        
        # ======================================================================
        # STEP 3: Evaluate RL Policy (compute metrics for every episode for plotting)
        # ======================================================================
        per_class_stats = {}
        avg_precision = 0.0
        avg_coverage = 0.0
        
        # Always compute metrics for every episode (for plotting)
        # But only print when classifier is updated (or at the end of training)
        # For debugging: evaluate every episode if verbose >= 2, otherwise every classifier_update_every
        should_evaluate = (verbose >= 2) or (ep % max(1, int(classifier_update_every))) == 0 or (ep == episodes - 1)
        should_print = verbose >= 1 and should_evaluate
        
        # Always compute metrics for plotting (even if not printing)
        # Initialize with training metrics if available (from DDPG training), otherwise defaults
        for cls in target_classes:
            if cls not in per_class_stats:
                # Use training metrics if available (for DDPG, these are computed during training)
                if use_continuous_actions and cls in training_metrics_per_class:
                    training_metrics = training_metrics_per_class[cls]
                    per_class_stats[cls] = {
                        "precision": training_metrics.get("precision", 0.0),
                        "hard_precision": training_metrics.get("hard_precision", training_metrics.get("precision", 0.0)),
                        "coverage": training_metrics.get("coverage", 0.0),
                        "n_points": training_metrics.get("n_points", 0),
                        "rule": "from training metrics",
                        "lower": None,
                        "upper": None
                    }
                else:
                    per_class_stats[cls] = {
                        "precision": 0.0,
                        "hard_precision": 0.0,
                        "coverage": 0.0,
                        "n_points": 0,
                        "rule": "computation pending",
                        "lower": None,
                        "upper": None
                    }
        
        # Always compute metrics for plotting (even if not printing)
        # Evaluation should run every episode for accurate metrics tracking
        try:
            if should_print:
                print(f"\n[Episode {ep + 1}] Evaluating RL Policy (after {classifier_update_every} RL episodes)...")
            elif verbose >= 2:
                print(f"\n[Episode {ep + 1}] Computing RL metrics (silent evaluation)...")
            from trainers.dynAnchors_inference import greedy_rollout
            from trainers.vecEnv import DynamicAnchorEnv
            
            # ALWAYS evaluate (not just when should_evaluate is True)
            # This ensures metrics are tracked for every episode
            for cls in target_classes:
                # Sample a specific training instance for this class to evaluate
                # This matches how final evaluation works (evaluates specific instances)
                cls_indices = np.where(y_train == cls)[0]
                if len(cls_indices) == 0:
                    # No instances of this class, skip
                    per_class_stats[cls] = {
                        "precision": 0.0,
                        "hard_precision": 0.0,
                        "coverage": 0.0,
                        "n_points": 0,
                        "rule": "no instances",
                        "lower": None,
                        "upper": None
                    }
                    continue
                
                # Sample one instance for evaluation (use episode number as seed for consistency)
                rng_eval = np.random.default_rng(seed=42 + cls + ep)
                eval_instance_idx = rng_eval.choice(cls_indices)
                X_eval_instance = X_train_scaled[eval_instance_idx]
                
                # Convert to unit space for x_star_unit
                X_eval_instance_unit = (X_eval_instance - X_min) / X_range
                X_eval_instance_unit = np.clip(X_eval_instance_unit, 0.0, 1.0).astype(np.float32)
                
                # Create environment for this class for evaluation
                # IMPORTANT: Match final evaluation settings exactly
                # 1. Use larger initial_window (0.3) to match evaluate_single_instance
                # 2. Use same perturbation settings as final evaluation
                # 3. Set x_star_unit to the sampled instance location (like final evaluation)
                anchor_env = create_anchor_env(target_cls=cls)
                # Set larger initial_window for evaluation (matches evaluate_single_instance)
                anchor_env.initial_window = 0.3  # Match final evaluation settings
                # Set x_star_unit to the instance location (like final evaluation)
                anchor_env.x_star_unit = X_eval_instance_unit
                # Ensure perturbation settings match (already set in create_anchor_env, but verify)
                # Note: create_anchor_env uses use_perturbation and perturbation_mode from function args
                
                # IMPORTANT: For training evaluation, we want to simulate the final evaluation workflow
                # Final evaluation uses raw AnchorEnv (not DynamicAnchorEnv wrapper) and samples
                # specific instances. However, we still use DynamicAnchorEnv for consistency with training.
                # The key is that initial_window=0.3 ensures the box starts large enough.
                wrapped_env = DynamicAnchorEnv(anchor_env, seed=42)
                # DynamicAnchorEnv.reset() will now respect initial_window=0.3 (not override it)
                
                # Run greedy rollout based on action type
                if use_continuous_actions:
                    # Continuous actions (DDPG/TD3): Use trainer for this class
                    continuous_trainer_eval = ddpg_trainers[cls]
                    # Get the model from the trainer
                    continuous_model = continuous_trainer_eval.model if hasattr(continuous_trainer_eval, 'model') else continuous_trainer_eval
                    
                    # Enable continuous actions in AnchorEnv
                    anchor_env.n_actions = 2 * anchor_env.n_features
                    anchor_env.max_action_scale = max(step_fracs) if step_fracs else 0.02
                    anchor_env.min_absolute_step = max(0.05, min_width * 0.5)
                    
                    # Wrap with ContinuousAnchorEnv
                    gym_env_eval = ContinuousAnchorEnv(anchor_env, seed=42 + cls + ep)
                    
                    # IMPORTANT: Capture initial bounds BEFORE greedy_rollout (which resets internally)
                    # Reset to get initial state and bounds
                    reset_result = gym_env_eval.reset(seed=42 + cls + ep)
                    if isinstance(reset_result, tuple):
                        obs, _ = reset_result
                    else:
                        obs = reset_result
                    obs = np.array(obs, dtype=np.float32)
                    
                    # Capture initial bounds BEFORE greedy_rollout modifies them
                    initial_lower = anchor_env.lower.copy()
                    initial_upper = anchor_env.upper.copy()
                    initial_width = (initial_upper - initial_lower)
                    
                    # Use greedy_rollout function for consistency (it now handles DDPG)
                    # Note: greedy_rollout will reset again internally, but we have initial bounds
                    try:
                        info, rule, lower, upper = greedy_rollout(
                            env=gym_env_eval,
                            trained_model=continuous_model,
                            steps_per_episode=eval_steps_per_episode if eval_steps_per_episode is not None else steps_per_episode,
                            max_features_in_rule=max_features_in_rule,
                            device=device_str
                        )
                        # Rule is already computed correctly by greedy_rollout with improved tightening detection
                    except Exception as e:
                        # Fallback: manual rollout if greedy_rollout fails
                        if verbose >= 2:
                            print(f"    [WARNING] greedy_rollout failed for DDPG, using manual rollout: {e}")
                        try:
                            import gymnasium as gym
                            GYM_VERSION = "gymnasium"
                        except ImportError:
                            try:
                                import gym
                                GYM_VERSION = "gym"
                            except ImportError:
                                GYM_VERSION = "gym"
                        
                        if GYM_VERSION == "gymnasium":
                            obs, _ = gym_env_eval.reset(seed=42 + cls + ep)
                        else:
                            obs = gym_env_eval.reset()
                        obs = np.array(obs, dtype=np.float32)
                        
                        initial_lower = anchor_env.lower.copy()
                        initial_upper = anchor_env.upper.copy()
                        initial_width = (initial_upper - initial_lower)
                        
                        # Run greedy rollout (deterministic for evaluation)
                        for t in range(eval_steps_per_episode if eval_steps_per_episode is not None else steps_per_episode):
                            action, _ = continuous_model.predict(obs, deterministic=True)
                            if isinstance(action, torch.Tensor):
                                action = action.cpu().numpy()
                            action = np.clip(action, -1.0, 1.0)
                            
                            if GYM_VERSION == "gymnasium":
                                next_obs, _, terminated, truncated, step_info = gym_env_eval.step(action)
                                done = terminated or truncated
                            else:
                                step_result = gym_env_eval.step(action)
                                if len(step_result) == 5:
                                    next_obs, _, terminated, truncated, step_info = step_result
                                    done = terminated or truncated
                                else:
                                    next_obs, _, done, step_info = step_result
                            obs = np.array(next_obs, dtype=np.float32)
                            if done:
                                break
                        
                        # Get final metrics after manual rollout
                        prec, cov, det = anchor_env._current_metrics()
                        info = {
                            "precision": prec,
                            "hard_precision": det.get("hard_precision", prec),
                            "coverage": cov,
                            "avg_prob": det.get("avg_prob", prec),
                            "sampler": det.get("sampler", "empirical"),
                            "n_points": det.get("n_points", 0),
                        }
                        # Build rule manually
                        lw = (anchor_env.upper - anchor_env.lower)
                        tightened = np.where(lw < initial_width * 0.95)[0]
                        if tightened.size > 0:
                            tightened_sorted = np.argsort(lw[tightened])
                            to_show = tightened[tightened_sorted[:max_features_in_rule]] if max_features_in_rule > 0 else tightened
                            cond_parts = [f"{feature_names[i]} ∈ [{anchor_env.lower[i]:.2f}, {anchor_env.upper[i]:.2f}]" for i in to_show]
                            rule = " and ".join(cond_parts) if cond_parts else "any values (no tightened features)"
                        else:
                            rule = "any values (no tightened features)"
                        lower = anchor_env.lower.copy()
                        upper = anchor_env.upper.copy()
                else:
                    # PPO: Use PPO trainer
                    # IMPORTANT: Capture initial bounds BEFORE greedy_rollout (which resets internally)
                    # Reset first to get initial state and bounds
                    reset_result = wrapped_env.reset(seed=42 + cls + ep)
                    if isinstance(reset_result, tuple):
                        state, _ = reset_result
                    else:
                        state = reset_result
                    state = np.array(state, dtype=np.float32)
                    
                    # Capture initial bounds after reset (before greedy_rollout modifies them)
                    initial_lower = anchor_env.lower.copy()
                    initial_upper = anchor_env.upper.copy()
                    initial_width = (initial_upper - initial_lower)
                    
                    # Run greedy rollout (greedy_rollout will reset again internally, but we have initial bounds)
                    try:
                        info, rule, lower, upper = greedy_rollout(
                            env=wrapped_env,
                            trained_model=rl_trainer.model,
                            steps_per_episode=eval_steps_per_episode if eval_steps_per_episode is not None else steps_per_episode,
                            max_features_in_rule=max_features_in_rule,
                            device=device_str
                        )
                        # Rule is already computed correctly by greedy_rollout with improved tightening detection
                    except (AttributeError, TypeError) as e:
                        # If greedy_rollout can't access properties directly, run a simple greedy rollout
                        # State already reset above, initial_width already set
                        if verbose >= 2:
                            print(f"    [WARNING] greedy_rollout failed for PPO, using manual rollout: {e}")
                        
                        # Run greedy rollout
                        for t in range(eval_steps_per_episode if eval_steps_per_episode is not None else steps_per_episode):
                            action, _ = rl_trainer.model.predict(state, deterministic=True)
                            step_result = wrapped_env.step(int(action))
                            if len(step_result) == 5:
                                state, _, done, _, info = step_result
                            else:
                                state, _, done, info = step_result
                            state = np.array(state, dtype=np.float32)
                            if done:
                                break
                        
                        # Get final metrics after manual rollout
                        prec, cov, det = anchor_env._current_metrics()
                        info = {
                            "precision": prec,
                            "hard_precision": det.get("hard_precision", prec),
                            "coverage": cov,
                            "avg_prob": det.get("avg_prob", prec),
                            "sampler": det.get("sampler", "empirical"),
                            "n_points": det.get("n_points", 0),
                        }
                        
                        # Build rule manually using improved tightening detection (same as greedy_rollout)
                        lw = (anchor_env.upper - anchor_env.lower)
                        # Use improved tightening detection logic (same as in greedy_rollout)
                        if np.any(initial_width <= 0) or np.any(np.isnan(initial_width)) or np.any(np.isinf(initial_width)):
                            initial_width_ref = np.ones_like(initial_width)
                        else:
                            initial_width_ref = initial_width.copy()
                        
                        if np.any(lw <= 0) or np.any(np.isnan(lw)) or np.any(np.isinf(lw)):
                            tightened = np.array([], dtype=int)
                        else:
                            tightened = np.where(lw < initial_width_ref * 0.98)[0]
                            if tightened.size == 0:
                                tightened = np.where(lw < initial_width_ref * 0.99)[0]
                            if tightened.size == 0:
                                tightened = np.where(lw < 0.95)[0]
                            if tightened.size == 0:
                                if np.all(initial_width_ref >= 0.9):
                                    tightened = np.where(lw < 0.9)[0]
                            if tightened.size == 0:
                                tightened = np.where(lw < initial_width_ref)[0]
                        
                        if tightened.size == 0:
                            rule = "any values (no tightened features)"
                        else:
                            tightened_sorted = np.argsort(lw[tightened])
                            to_show = tightened[tightened_sorted[:max_features_in_rule]] if max_features_in_rule > 0 else tightened
                            if to_show.size == 0:
                                rule = "any values (no tightened features)"
                            else:
                                cond_parts = [f"{feature_names[i]} ∈ [{anchor_env.lower[i]:.2f}, {anchor_env.upper[i]:.2f}]" for i in to_show]
                                rule = " and ".join(cond_parts)
                        lower = anchor_env.lower.copy()
                        upper = anchor_env.upper.copy()
                
                # Get final metrics after greedy rollout (for both PPO and DDPG)
                # IMPORTANT: Always recompute metrics from final state to ensure accuracy
                # (greedy_rollout already returns info, but we recompute to be safe)
                prec, cov, det = anchor_env._current_metrics()
                
                # Update info with recomputed metrics (but keep rule from greedy_rollout)
                info = {
                    "precision": prec,
                    "hard_precision": det.get("hard_precision", prec),
                    "coverage": cov,
                    "avg_prob": det.get("avg_prob", prec),
                    "sampler": det.get("sampler", "empirical"),
                    "n_points": det.get("n_points", 0),
                }
                
                # Debug: Check if box actually expanded
                final_width = (anchor_env.upper - anchor_env.lower)
                initial_vol = np.prod(initial_width) if np.prod(initial_width) > 0 else 1e-9
                final_vol = np.prod(final_width) if np.prod(final_width) > 0 else 1e-9
                expansion_ratio = final_vol / initial_vol if initial_vol > 0 else 1.0
                box_size_ratio = np.mean(final_width) / np.mean(initial_width) if np.mean(initial_width) > 0 else 1.0
                
                # Enhanced debug output
                if verbose >= 2 or should_print:
                    # Use hard_precision for consistency with RL logging (both should show the same metric)
                    hard_prec = det.get("hard_precision", prec)
                    print(f"  [EVAL cls={cls}] prec={hard_prec:.6f} | cov={cov:.6f} | n_pts={det.get('n_points', 0)}")
                    print(f"    Box: initial_vol={initial_vol:.6f} | final_vol={final_vol:.6f} | "
                          f"expansion={expansion_ratio:.3f}x | size_ratio={box_size_ratio:.3f}x")
                    print(f"    Initial box: min={np.min(initial_width):.4f} | max={np.max(initial_width):.4f} | "
                          f"mean={np.mean(initial_width):.4f}")
                    print(f"    Final box: min={np.min(final_width):.4f} | max={np.max(final_width):.4f} | "
                          f"mean={np.mean(final_width):.4f}")
                    if det.get('n_points', 0) == 0:
                        print(f"    ⚠ WARNING: No points found in box! Box may be too small or in wrong region.")
                
                # Rule is already set by greedy_rollout (with improved tightening detection)
                # Don't overwrite it! The rule from greedy_rollout uses the improved logic.
                
                # Store statistics
                # Get detailed metrics for debugging
                n_points = info.get("n_points", 0)
                actual_precision = info.get("precision", 0.0)
                actual_coverage = info.get("coverage", 0.0)
                hard_precision = info.get("hard_precision", actual_precision)
                
                per_class_stats[cls] = {
                    "precision": actual_precision,
                    "hard_precision": hard_precision,
                    "coverage": actual_coverage,
                    "n_points": n_points,
                    "rule": rule,
                    "lower": lower,
                    "upper": upper
                }
                
                # Print per-class statistics (only if printing)
                if should_print:
                    rule_str = per_class_stats[cls]['rule']
                    if len(rule_str) > 80:
                        rule_str = rule_str[:80] + "..."
                    # Show more decimal places and n_points for debugging
                    print(f"  [RL cls={cls}] prec={hard_precision:.6f} | "
                          f"cov={actual_coverage:.6f} | "
                          f"n_pts={n_points} | "
                          f"rule: {rule_str}")
                
                # Store per-class metrics for plotting (every episode)
                # Always store, even if 0.0, to maintain consistent episode tracking
                per_class_precision_history[cls].append(per_class_stats[cls]['hard_precision'])
                per_class_coverage_history[cls].append(per_class_stats[cls]['coverage'])
            
            # Calculate average statistics
            # Use per_class_stats if evaluation ran, otherwise use training_metrics_per_class
            if per_class_stats and any(stats.get("precision", 0.0) > 0 or stats.get("coverage", 0.0) > 0 for stats in per_class_stats.values()):
                # Evaluation ran and computed metrics
                avg_precision = np.mean([stats["hard_precision"] for stats in per_class_stats.values()])
                avg_coverage = np.mean([stats["coverage"] for stats in per_class_stats.values()])
                total_n_points = sum([stats.get("n_points", 0) for stats in per_class_stats.values()])
            elif use_continuous_actions and training_metrics_per_class:
                # Use training metrics for DDPG when evaluation didn't run
                avg_precision = np.mean([metrics.get("hard_precision", metrics.get("precision", 0.0)) 
                                        for metrics in training_metrics_per_class.values()])
                avg_coverage = np.mean([metrics.get("coverage", 0.0) 
                                       for metrics in training_metrics_per_class.values()])
                total_n_points = sum([metrics.get("n_points", 0) for metrics in training_metrics_per_class.values()])
            else:
                # Fallback: compute from per_class_stats even if all zeros
                avg_precision = np.mean([stats.get("hard_precision", 0.0) for stats in per_class_stats.values()]) if per_class_stats else 0.0
                avg_coverage = np.mean([stats.get("coverage", 0.0) for stats in per_class_stats.values()]) if per_class_stats else 0.0
                total_n_points = sum([stats.get("n_points", 0) for stats in per_class_stats.values()]) if per_class_stats else 0
                
                if should_print:
                    print(f"\n  [RL Overall] avg_precision={avg_precision:.6f} | avg_coverage={avg_coverage:.6f} | total_n_pts={total_n_points}")
                    # Debug: Show if any classes have 0 points (explains 0 precision/coverage)
                    zero_pts_classes = [cls for cls, stats in per_class_stats.items() if stats.get("n_points", 0) == 0]
                    if zero_pts_classes:
                        print(f"  [DEBUG] Classes with 0 points covered: {zero_pts_classes} (box may be too small)")
        except Exception as e:
            # If evaluation fails, store 0.0 for all classes to maintain consistent tracking
            if verbose >= 2:
                print(f"  [WARNING] Evaluation failed for episode {ep + 1}: {e}")
            for cls in target_classes:
                if cls not in per_class_stats:
                    per_class_stats[cls] = {
                        "precision": 0.0,
                        "hard_precision": 0.0,
                        "coverage": 0.0,
                        "n_points": 0,
                        "rule": "evaluation failed",
                    }
                # Store 0.0 values to maintain episode count consistency
                per_class_precision_history[cls].append(0.0)
                per_class_coverage_history[cls].append(0.0)
            
            # Set averages to 0.0 if computation failed
            avg_precision = 0.0
            avg_coverage = 0.0
        
        # Ensure we store metrics for every episode, even if computation was skipped
        # This maintains consistent episode tracking for plotting
        for cls in target_classes:
            # Only append if not already appended (in case evaluation was skipped or failed)
            expected_length = ep + 1  # Should have one entry per episode (0-indexed, so ep+1)
            if len(per_class_precision_history[cls]) < expected_length:
                # Use per_class_stats if available, otherwise use training metrics
                if cls in per_class_stats and per_class_stats[cls].get("precision", 0.0) > 0:
                    precision_val = per_class_stats[cls].get("hard_precision", per_class_stats[cls].get("precision", 0.0))
                    coverage_val = per_class_stats[cls].get("coverage", 0.0)
                elif use_continuous_actions and cls in training_metrics_per_class:
                    # Use training metrics for DDPG when evaluation didn't run
                    training_metrics = training_metrics_per_class[cls]
                    precision_val = training_metrics.get("hard_precision", training_metrics.get("precision", 0.0))
                    coverage_val = training_metrics.get("coverage", 0.0)
                else:
                    precision_val = per_class_stats.get(cls, {}).get("hard_precision", 0.0)
                    coverage_val = per_class_stats.get(cls, {}).get("coverage", 0.0)
                
                per_class_precision_history[cls].append(precision_val)
                per_class_coverage_history[cls].append(coverage_val)
                if verbose >= 2:
                    print(f"  [DEBUG] Stored metrics for episode {ep + 1}, class {cls}: "
                          f"prec={precision_val:.6f}, cov={coverage_val:.6f}")
            
            # Verify we have the right number of entries
            if verbose >= 2 and len(per_class_precision_history[cls]) != expected_length:
                print(f"  [WARNING] Episode {ep + 1}: Class {cls} has {len(per_class_precision_history[cls])} "
                      f"precision entries, expected {expected_length}")
        
        # Store history
        joint_training_history.append({
            "episode": ep + 1,
            "classifier_test_acc": float(test_acc),
            "classifier_train_acc": float(last_train_acc) if last_train_acc > 0 else None,
            "classifier_loss": float(last_loss) if last_loss > 0 else None,
            "rl_timesteps": timesteps_per_episode,
            "rl_stats": per_class_stats,
            "rl_avg_precision": float(avg_precision) if per_class_stats else None,
            "rl_avg_coverage": float(avg_coverage) if per_class_stats else None,
        })
        
        if verbose >= 1:
            print(f"\nEpisode {ep + 1} Summary:")
            print(f"  Classifier Test Acc: {test_acc:.3f}")
            if last_train_acc > 0:
                print(f"  Classifier Train Acc: {last_train_acc:.3f} | Loss: {last_loss:.4f}")
            print(f"  RL Timesteps: {timesteps_per_episode}")
            if per_class_stats:
                print(f"  RL Avg Precision: {avg_precision:.3f} | RL Avg Coverage: {avg_coverage:.3f}")
    
    # Final: Set classifier to eval mode and freeze
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    
    print(f"\n{'='*80}")
    print("JOINT TRAINING COMPLETE - Freezing models for evaluation")
    print(f"{'='*80}")
    
    # ======================================================================
    # STEP 3: Final Evaluation with Frozen Policy
    # ======================================================================
    print(f"\nEvaluating on test set with frozen models...")
    print(f"Note: By default, coverage and precision are computed on training data.")
    print(f"      This explains the classifier's behavior on training data.")
    print(f"      To evaluate on test data, set eval_on_test_data=True in evaluate_all_classes.")
    
    # Prepare test data in unit space for optional test-set evaluation
    X_test_unit = (X_test_scaled - X_min) / X_range
    
    if use_continuous_actions:
        # Continuous actions (DDPG/TD3): Use trainers for evaluation
        # Continuous action algorithms have separate trainers per class, so pass the dict of trainers
        if ddpg_trainers and len(ddpg_trainers) > 0:
            # Pass the dict of continuous action trainers (evaluate_all_classes will handle per-class trainers)
            eval_results = evaluate_all_classes(
                X_test=X_test_scaled,
                y_test=y_test,
                trained_model=ddpg_trainers,  # Pass dict of trainers per class
                make_env_fn=make_anchor_env,
                feature_names=feature_names,
                n_instances_per_class=n_eval_instances_per_class,
                max_features_in_rule=max_features_in_rule,
                steps_per_episode=eval_steps_per_episode if eval_steps_per_episode is not None else steps_per_episode,
                random_seed=42,
                # Optional: Set eval_on_test_data=True to compute metrics on test data
                eval_on_test_data=False,  # Default: use training data (backward compatible)
                X_test_unit=X_test_unit if False else None,  # Prepare but don't use unless eval_on_test_data=True
                X_test_std=X_test_scaled if False else None,
                num_rollouts_per_instance=num_rollouts_per_instance,
            )
        else:
            raise ValueError("No DDPG trainers available for evaluation")
    else:
        # PPO: Use PPO trainer's model
        eval_results = evaluate_all_classes(
            X_test=X_test_scaled,
            y_test=y_test,
            trained_model=rl_trainer.model,
            make_env_fn=make_anchor_env,
            feature_names=feature_names,
            n_instances_per_class=n_eval_instances_per_class,
            max_features_in_rule=max_features_in_rule,
            steps_per_episode=eval_steps_per_episode if eval_steps_per_episode is not None else steps_per_episode,
            random_seed=42,
            # Optional: Set eval_on_test_data=True to compute metrics on test data
            eval_on_test_data=False,  # Default: use training data (backward compatible)
            X_test_unit=X_test_unit if False else None,  # Prepare but don't use unless eval_on_test_data=True
            X_test_std=X_test_scaled if False else None,
            num_rollouts_per_instance=num_rollouts_per_instance,
        )
    
    # Close environments
    if not use_continuous_actions and rl_trainer is not None and hasattr(rl_trainer, 'vec_env'):
        try:
            rl_trainer.vec_env.close()
        except:
            pass
    
    # Cleanup continuous action environments if used (DDPG/TD3)
    if use_continuous_actions and ddpg_trainers:
        # Continuous action environments are already closed in the training loop, but ensure cleanup
        for cls, continuous_trainer in ddpg_trainers.items():
            if hasattr(continuous_trainer, 'env') and continuous_trainer.env is not None:
                try:
                    continuous_trainer.env.close()
                except:
                    pass
    
    # ======================================================================
    # STEP 4: Save RL Models (Policy and Value Networks)
    # ======================================================================
    print(f"\n{'='*80}")
    print("SAVING RL MODELS")
    print(f"{'='*80}")
    
    models_dir = f"{output_dir}/models/"
    os.makedirs(models_dir, exist_ok=True)
    
    if use_continuous_actions:
        # Continuous actions (DDPG/TD3): Save each class's model (actor and critic networks)
        algorithm_name = continuous_algorithm.upper() if continuous_algorithm.lower() == "td3" else "DDPG"
        print(f"\nSaving {algorithm_name} models (actor and critic networks)...")
        for cls, continuous_trainer in ddpg_trainers.items():
            model_path = f"{models_dir}/{continuous_algorithm.lower()}_class_{cls}_final"
            continuous_trainer.save(model_path)
            print(f"  Saved {algorithm_name} model for class {cls}: {model_path}")
            
            # Also save actor and critic separately for easier inspection
            # Note: TD3 has twin critics, but SB3 exposes them through unified critic interface
            actor_path = f"{models_dir}/{continuous_algorithm.lower()}_class_{cls}_actor"
            critic_path = f"{models_dir}/{continuous_algorithm.lower()}_class_{cls}_critic"
            try:
                # Save actor network (same for DDPG and TD3)
                torch.save(continuous_trainer.model.actor.state_dict(), f"{actor_path}.pth")
                print(f"    Saved actor network: {actor_path}.pth")
                # Save critic network (for TD3, this saves the unified critic interface)
                torch.save(continuous_trainer.model.critic.state_dict(), f"{critic_path}.pth")
                print(f"    Saved critic network: {critic_path}.pth")
            except Exception as e:
                if verbose >= 1:
                    print(f"    [WARNING] Could not save actor/critic separately: {e}")
    else:
        # PPO: Save PPO model (policy and value networks)
        print(f"\nSaving PPO model (policy and value networks)...")
        model_path = f"{models_dir}/ppo_final"
        rl_trainer.save(model_path)
        print(f"  Saved PPO model: {model_path}")
        
        # Also save policy and value networks separately for easier inspection
        policy_path = f"{models_dir}/ppo_policy"
        value_path = f"{models_dir}/ppo_value"
        try:
            # Save policy network
            torch.save(rl_trainer.model.policy.state_dict(), f"{policy_path}.pth")
            print(f"    Saved policy network: {policy_path}.pth")
            # Save value network (value function)
            torch.save(rl_trainer.model.policy.value_net.state_dict(), f"{value_path}.pth")
            print(f"    Saved value network: {value_path}.pth")
        except Exception as e:
            if verbose >= 1:
                print(f"    [WARNING] Could not save policy/value separately: {e}")
    
    # Save classifier model
    clf_type_lower = classifier_type.lower()
    if clf_type_lower == "random_forest":
        # Save Random Forest using joblib (sklearn's recommended way)
        try:
            import joblib
            classifier_path = f"{models_dir}/classifier_final_rf.joblib"
            joblib.dump(classifier.rf_model, classifier_path)
            print(f"  Saved Random Forest classifier: {classifier_path}")
        except ImportError:
            print(f"  [WARNING] joblib not available, skipping Random Forest save")
    elif clf_type_lower == "gradient_boosting":
        # Save Gradient Boosting using joblib (sklearn's recommended way)
        try:
            import joblib
            classifier_path = f"{models_dir}/classifier_final_gb.joblib"
            joblib.dump(classifier.gb_model, classifier_path)
            print(f"  Saved Gradient Boosting classifier: {classifier_path}")
        except ImportError:
            print(f"  [WARNING] joblib not available, skipping Gradient Boosting save")
    else:  # DNN
        # Save DNN using PyTorch
        classifier_path = f"{models_dir}/classifier_final.pth"
        torch.save(classifier.state_dict(), classifier_path)
        print(f"  Saved DNN classifier: {classifier_path}")
    
    # ======================================================================
    # STEP 5: Save Full Metrics and Rules to JSON
    # ======================================================================
    print(f"\n{'='*80}")
    print("SAVING METRICS AND RULES TO JSON")
    print(f"{'='*80}")
    
    # Prepare comprehensive metrics and rules data
    metrics_data = {
        "training_summary": {
            "episodes": episodes,
            "steps_per_episode": steps_per_episode,
            "classifier_epochs_per_round": classifier_epochs_per_round,
            "classifier_update_every": classifier_update_every,
            "use_continuous_actions": use_continuous_actions,
            "algorithm": continuous_algorithm.upper() if use_continuous_actions else "PPO",
        },
        "overall_statistics": {
            "overall_precision": float(eval_results["overall_precision"]),
            "overall_coverage": float(eval_results["overall_coverage"]),
            "overall_n_points": int(eval_results.get("overall_n_points", 0)),
        },
        "per_class_results": {},
        "training_history": [],
        "evaluation_results": {}
    }
    
    # Add per-class results from evaluation
    for cls_int in target_classes:
        if cls_int in eval_results.get("per_class_results", {}):
            cls_result = eval_results["per_class_results"][cls_int]
            
            # Extract rules and instance information from individual_results
            rules_list = []  # Simple list of rule strings
            rules_with_instances = []  # Rules with instance indices
            anchors_list = []
            instance_indices_used = []  # Track which instances were evaluated
            
            if "individual_results" in cls_result:
                for individual_result in cls_result["individual_results"]:
                    instance_idx = int(individual_result.get("instance_idx", -1))
                    rule = individual_result.get("rule", "")
                    
                    # Track instance index
                    if instance_idx >= 0:
                        instance_indices_used.append(instance_idx)
                    
                    # Add to rules list (all rules, including empty ones)
                    rules_list.append(rule)
                    
                    # Add rule with instance information
                    rules_with_instances.append({
                        "instance_idx": instance_idx,
                        "rule": rule,
                        "precision": float(individual_result.get("precision", 0.0)),
                        "hard_precision": float(individual_result.get("hard_precision", individual_result.get("precision", 0.0))),
                        "coverage": float(individual_result.get("coverage", 0.0)),
                        "n_points": int(individual_result.get("n_points", 0)),
                    })
                    
                    # Add anchor details (lower/upper bounds) for each instance
                    anchors_list.append({
                        "instance_idx": instance_idx,
                        "lower_bounds": individual_result.get("lower_bounds", []),
                        "upper_bounds": individual_result.get("upper_bounds", []),
                        "precision": float(individual_result.get("precision", 0.0)),
                        "hard_precision": float(individual_result.get("hard_precision", individual_result.get("precision", 0.0))),
                        "coverage": float(individual_result.get("coverage", 0.0)),
                        "n_points": int(individual_result.get("n_points", 0)),
                        "rule": rule,
                    })
            
            # Count unique rules
            unique_rules = list(set([r for r in rules_list if r]))  # Only non-empty rules
            unique_rules_count = len(unique_rules)
            
            metrics_data["per_class_results"][f"class_{cls_int}"] = {
                "precision": float(cls_result.get("precision", cls_result.get("avg_precision", 0.0))),
                "hard_precision": float(cls_result.get("hard_precision", cls_result.get("avg_hard_precision", cls_result.get("precision", 0.0)))),
                "coverage": float(cls_result.get("coverage", cls_result.get("avg_coverage", 0.0))),
                "n_points": int(cls_result.get("n_points", 0)),
                "n_instances_evaluated": int(cls_result.get("n_instances", len(anchors_list))),
                "best_rule": cls_result.get("best_rule", ""),
                "best_precision": float(cls_result.get("best_precision", 0.0)),
                # Rules information
                "rules": rules_list,  # All rules (one per instance, in order)
                "rules_with_instances": rules_with_instances,  # Rules with instance indices and metrics
                "unique_rules": unique_rules,  # List of unique rule strings
                "unique_rules_count": unique_rules_count,  # Number of unique rules
                # Instance information
                "instance_indices_used": instance_indices_used,  # Which test instances were evaluated
                # Full anchor details
                "anchors": anchors_list  # Complete anchor details for each instance (includes bounds, metrics, rule)
            }
    
    # Add training history (per episode)
    for hist_entry in joint_training_history:
        episode_data = {
            "episode": int(hist_entry["episode"]),
            "classifier_test_acc": float(hist_entry["classifier_test_acc"]) if hist_entry.get("classifier_test_acc") is not None else None,
            "classifier_train_acc": float(hist_entry["classifier_train_acc"]) if hist_entry.get("classifier_train_acc") is not None else None,
            "classifier_loss": float(hist_entry["classifier_loss"]) if hist_entry.get("classifier_loss") is not None else None,
            "rl_timesteps": int(hist_entry["rl_timesteps"]),
            "rl_avg_precision": float(hist_entry["rl_avg_precision"]) if hist_entry.get("rl_avg_precision") is not None else None,
            "rl_avg_coverage": float(hist_entry["rl_avg_coverage"]) if hist_entry.get("rl_avg_coverage") is not None else None,
            "per_class_rl_stats": {}
        }
        
        # Add per-class RL stats for this episode
        if "rl_stats" in hist_entry and hist_entry["rl_stats"]:
            for cls_int, cls_stats in hist_entry["rl_stats"].items():
                episode_data["per_class_rl_stats"][f"class_{cls_int}"] = {
                    "precision": float(cls_stats.get("precision", 0.0)),
                    "hard_precision": float(cls_stats.get("hard_precision", cls_stats.get("precision", 0.0))),
                    "coverage": float(cls_stats.get("coverage", 0.0)),
                    "n_points": int(cls_stats.get("n_points", 0)),
                    "rule": cls_stats.get("rule", ""),
                }
        
        metrics_data["training_history"].append(episode_data)
    
    # Add full evaluation results
    metrics_data["evaluation_results"] = {
        "overall_precision": float(eval_results.get("overall_precision", 0.0)),
        "overall_coverage": float(eval_results.get("overall_coverage", 0.0)),
        "overall_n_points": int(eval_results.get("overall_n_points", 0)),
        "per_class_precision": {f"class_{k}": float(v) for k, v in eval_results.get("per_class_precision", {}).items()},
        "per_class_coverage": {f"class_{k}": float(v) for k, v in eval_results.get("per_class_coverage", {}).items()},
    }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        """Recursively convert numpy arrays and other non-serializable types to JSON-compatible types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    # Convert all data to JSON-serializable format
    metrics_data = convert_to_serializable(metrics_data)
    
    # Save to JSON file
    json_path = f"{output_dir}/metrics_and_rules.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metrics and rules to: {json_path}")
    
    # ======================================================================
    # STEP 6: Create 2D Visualization of Rules
    # ======================================================================
    print(f"\n{'='*80}")
    print("CREATING 2D VISUALIZATION OF RULES")
    print(f"{'='*80}")
    
    try:
        from trainers.dynAnchors_inference import plot_rules_2d
        
        plot_path = f"{output_dir}/plots/rules_2d_visualization.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        
        # Get X_min and X_range (they're defined earlier in the function)
        # X_min and X_range are computed from X_train_scaled for unit space conversion
        # For plotting, we need them to convert bounds from unit space to standardized space
        
        # eval_results should already have the correct structure from evaluate_all_classes
        # It has per_class_results with integer keys (class indices)
        # Verify structure before plotting
        if "per_class_results" not in eval_results:
            raise ValueError("eval_results missing 'per_class_results'")
        
        # IMPORTANT: evaluate_all_classes returns individual_results, but plot_rules_2d expects anchors
        # Extract anchors from individual_results before plotting
        per_class_results = eval_results.get("per_class_results", {}).copy()
        for cls_key, cls_result in per_class_results.items():
            # If anchors don't exist, extract them from individual_results
            if "anchors" not in cls_result and "individual_results" in cls_result:
                anchors_list = []
                instance_indices_used = []
                for individual_result in cls_result["individual_results"]:
                    instance_idx = int(individual_result.get("instance_idx", -1))
                    if instance_idx >= 0:
                        instance_indices_used.append(instance_idx)
                    anchors_list.append({
                        "instance_idx": instance_idx,
                        "lower_bounds": individual_result.get("lower_bounds", []),
                        "upper_bounds": individual_result.get("upper_bounds", []),
                        "precision": float(individual_result.get("precision", 0.0)),
                        "hard_precision": float(individual_result.get("hard_precision", individual_result.get("precision", 0.0))),
                        "coverage": float(individual_result.get("coverage", 0.0)),
                        "n_points": int(individual_result.get("n_points", 0)),
                        "rule": individual_result.get("rule", ""),
                    })
                cls_result["anchors"] = anchors_list
                cls_result["instance_indices_used"] = instance_indices_used
        
        # Create a copy of eval_results with anchors added
        eval_results_with_anchors = eval_results.copy()
        eval_results_with_anchors["per_class_results"] = per_class_results
        if len(per_class_results) == 0:
            print(f"  [WARNING] No per_class_results found in eval_results, skipping 2D visualization")
            if verbose >= 1:
                print(f"    eval_results keys: {list(eval_results.keys())}")
        else:
            # Debug: Check structure
            if verbose >= 2:
                print(f"  [DEBUG] per_class_results keys: {list(per_class_results.keys())[:5]}")
                sample_key = list(per_class_results.keys())[0]
                sample_result = per_class_results[sample_key]
                print(f"  [DEBUG] Sample class result keys: {list(sample_result.keys())}")
            
            # Check if anchors are present
            has_anchors = False
            anchor_count = 0
            for cls_key, cls_result in per_class_results.items():
                if "anchors" in cls_result:
                    anchors = cls_result.get("anchors", [])
                    if len(anchors) > 0:
                        has_anchors = True
                        anchor_count += len(anchors)
                        if verbose >= 2:
                            print(f"  [DEBUG] Class {cls_key} has {len(anchors)} anchors")
            
            if not has_anchors:
                print(f"  [WARNING] No anchors found in eval_results, skipping 2D visualization")
                print(f"    Hint: This might happen if all rules are 'any values (no tightened features)'")
                if verbose >= 1:
                    print(f"    Total classes checked: {len(per_class_results)}")
                    print(f"    Sample class keys: {list(per_class_results.keys())[:3]}")
            else:
                if verbose >= 1:
                    print(f"  Found {anchor_count} anchors across {len(per_class_results)} classes")
                # Create 2D plot of rules
                try:
                    plot_path = plot_rules_2d(
                        eval_results=eval_results_with_anchors,  # Use eval_results with anchors added
                        X_test=X_test_scaled,  # Use standardized test data
                        y_test=y_test,
                        feature_names=feature_names,
                        class_names=None,  # Can be passed if available, but not required
                        output_path=plot_path,
                        X_min=X_min,  # Available from earlier in function
                        X_range=X_range,  # Available from earlier in function
                    )
                    print(f"  ✓ Saved 2D rules visualization: {plot_path}")
                except Exception as plot_error:
                    print(f"  [ERROR] Failed to create 2D visualization: {plot_error}")
                    import traceback
                    traceback.print_exc()  # Always print traceback for debugging
                    # Try to provide helpful error message
                    print(f"    Check that:")
                    print(f"    - eval_results contains per_class_results with anchors")
                    print(f"    - anchors have lower_bounds and upper_bounds")
                    print(f"    - X_test_scaled, X_min, X_range are valid")
                    print(f"    - X_test_scaled shape: {X_test_scaled.shape if hasattr(X_test_scaled, 'shape') else 'N/A'}")
                    print(f"    - y_test shape: {y_test.shape if hasattr(y_test, 'shape') else 'N/A'}")
                    print(f"    - feature_names length: {len(feature_names) if feature_names else 'N/A'}")
    except Exception as e:
        if verbose >= 1:
            print(f"  [WARNING] Could not create 2D visualization: {e}")
        import traceback
        if verbose >= 2:
            traceback.print_exc()
        elif verbose >= 1:
            # Print a bit more detail even at verbose=1
            print(f"    Error type: {type(e).__name__}")
    
    # Prepare results
    if use_continuous_actions:
        # Continuous actions (DDPG/TD3): Return trainers
        results = {
            "trained_model": ddpg_trainers,  # Dictionary of continuous action trainers per class
            "classifier": classifier,
            "trainer": ddpg_trainers,  # Dictionary of continuous action trainers
            "eval_results": eval_results,
        "overall_stats": {
            "avg_precision": eval_results["overall_precision"],
            "avg_coverage": eval_results["overall_coverage"],
        },
        "joint_training_history": joint_training_history,
        "complexity_analysis": complexity_analysis,  # Store complexity analysis for reference
        "metadata": {
            "n_classes": len(target_classes),
            "n_features": len(feature_names),
            "n_train_samples": n_train_samples,
            "n_test_samples": n_test_samples,
            "target_classes": target_classes,
            "feature_names": feature_names,
            "output_dir": output_dir,
            "episodes": episodes,
            "steps_per_episode": steps_per_episode,
            "classifier_epochs_per_round": classifier_epochs_per_round,
            "classifier_update_every": classifier_update_every,
            "test_acc_history": test_acc_history,
            "current_total_rl_timesteps": current_total_rl_timesteps,
            "current_total_classifier_epochs": current_total_classifier_epochs,
            "recommended_total_rl_timesteps": complexity_analysis['total_rl_timesteps'],
            "recommended_total_classifier_epochs": complexity_analysis['total_classifier_epochs'],
            "training_sufficiency": {
                "episodes_ratio": episodes_ratio,
                "steps_ratio": steps_ratio,
                "total_timesteps_ratio": total_timesteps_ratio,
                "status": sufficiency_status,
                "recommendation": recommendation,
            },
            "use_continuous_actions": use_continuous_actions,
        },
        "model_paths": {
            "classifier": classifier_path,
            "rl_models_dir": models_dir,
        },
        "metrics_json_path": json_path,
    }
    else:
        # PPO: Return PPO trainer
        results = {
            "trained_model": rl_trainer.model,
            "classifier": classifier,
            "trainer": rl_trainer,
            "eval_results": eval_results,
            "overall_stats": {
                "avg_precision": eval_results["overall_precision"],
                "avg_coverage": eval_results["overall_coverage"],
            },
            "joint_training_history": joint_training_history,
            "complexity_analysis": complexity_analysis,
            "metadata": {
                "n_classes": len(target_classes),
                "n_features": len(feature_names),
                "n_train_samples": n_train_samples,
                "n_test_samples": n_test_samples,
                "target_classes": target_classes,
                "feature_names": feature_names,
                "output_dir": output_dir,
                "episodes": episodes,
                "steps_per_episode": steps_per_episode,
                "classifier_epochs_per_round": classifier_epochs_per_round,
                "classifier_update_every": classifier_update_every,
                "test_acc_history": test_acc_history,
                "current_total_rl_timesteps": current_total_rl_timesteps,
                "current_total_classifier_epochs": current_total_classifier_epochs,
                "recommended_total_rl_timesteps": complexity_analysis['total_rl_timesteps'],
                "recommended_total_classifier_epochs": complexity_analysis['total_classifier_epochs'],
                "training_sufficiency": {
                    "episodes_ratio": episodes_ratio,
                    "steps_ratio": steps_ratio,
                    "total_timesteps_ratio": total_timesteps_ratio,
                    "status": sufficiency_status,
                    "recommendation": recommendation,
                },
            "use_continuous_actions": use_continuous_actions,
        },
        "model_paths": {
            "classifier": classifier_path,
            "rl_models_dir": models_dir,
        },
        "metrics_json_path": json_path,
    }
    
    print(f"\nJoint training and evaluation complete!")
    print(f"Overall precision: {eval_results['overall_precision']:.3f}")
    print(f"Overall coverage: {eval_results['overall_coverage']:.3f}")
    print(f"\nModels saved to: {models_dir}")
    print(f"Metrics and rules saved to: {json_path}")
    
    # ======================================================================
    # STEP 5: Analyze Precision and Coverage Maximization
    # ======================================================================
    print(f"\n" + "="*80)
    print("PRECISION AND COVERAGE MAXIMIZATION ANALYSIS")
    print("="*80)
    
    # Analyze precision and coverage trends over episodes
    for cls in target_classes:
        prec_history = per_class_precision_history[cls]
        cov_history = per_class_coverage_history[cls]
        
        if len(prec_history) > 0 and len(cov_history) > 0:
            # Calculate trends
            initial_prec = prec_history[0] if len(prec_history) > 0 else 0.0
            final_prec = prec_history[-1] if len(prec_history) > 0 else 0.0
            max_prec = max(prec_history) if len(prec_history) > 0 else 0.0
            avg_prec = np.mean(prec_history) if len(prec_history) > 0 else 0.0
            
            initial_cov = cov_history[0] if len(cov_history) > 0 else 0.0
            final_cov = cov_history[-1] if len(cov_history) > 0 else 0.0
            max_cov = max(cov_history) if len(cov_history) > 0 else 0.0
            avg_cov = np.mean(cov_history) if len(cov_history) > 0 else 0.0
            
            # Calculate improvement
            prec_improvement = final_prec - initial_prec
            cov_improvement = final_cov - initial_cov
            
            # Check if metrics are improving
            # Consider last 25% of episodes for trend analysis
            n_recent = max(1, len(prec_history) // 4)
            recent_prec = prec_history[-n_recent:] if len(prec_history) >= n_recent else prec_history
            recent_cov = cov_history[-n_recent:] if len(cov_history) >= n_recent else cov_history
            
            recent_avg_prec = np.mean(recent_prec) if len(recent_prec) > 0 else 0.0
            recent_avg_cov = np.mean(recent_cov) if len(recent_cov) > 0 else 0.0
            
            # Check if recent average is better than overall average (improving)
            prec_trending_up = recent_avg_prec > avg_prec * 0.95  # Allow 5% margin
            cov_trending_up = recent_avg_cov > avg_cov * 0.95
            
            # Check if target is being met
            prec_target_met = final_prec >= precision_target * 0.9  # 90% of target
            cov_target_met = final_cov >= coverage_target * 0.9
            
            print(f"\nClass {cls} Analysis:")
            print(f"  Precision:")
            print(f"    Initial: {initial_prec:.6f} → Final: {final_prec:.6f} → Max: {max_prec:.6f}")
            if initial_prec > 0:
                print(f"    Improvement: {prec_improvement:+.6f} ({prec_improvement/initial_prec*100:+.1f}%)")
            else:
                print(f"    Improvement: {prec_improvement:+.6f} (N/A - initial precision was 0)")
            print(f"    Average: {avg_prec:.6f}, Recent average: {recent_avg_prec:.6f}")
            print(f"    Target ({precision_target:.3f}): {'✓ MET' if prec_target_met else '✗ NOT MET'}")
            print(f"    Trend: {'↑ IMPROVING' if prec_trending_up else '→ STABLE' if abs(recent_avg_prec - avg_prec) < 0.01 else '↓ DECLINING'}")
            
            print(f"  Coverage:")
            print(f"    Initial: {initial_cov:.6f} → Final: {final_cov:.6f} → Max: {max_cov:.6f}")
            if initial_cov > 0:
                print(f"    Improvement: {cov_improvement:+.6f} ({cov_improvement/initial_cov*100:+.1f}%)")
            else:
                print(f"    Improvement: {cov_improvement:+.6f} (N/A - initial coverage was 0)")
            print(f"    Average: {avg_cov:.6f}, Recent average: {recent_avg_cov:.6f}")
            print(f"    Target ({coverage_target:.3f}): {'✓ MET' if cov_target_met else '✗ NOT MET'}")
            print(f"    Trend: {'↑ IMPROVING' if cov_trending_up else '→ STABLE' if abs(recent_avg_cov - avg_cov) < 0.01 else '↓ DECLINING'}")
            
            # Overall assessment
            if prec_target_met and cov_target_met:
                status = "✓ EXCELLENT - Both targets met"
            elif prec_target_met and cov_trending_up:
                status = "⚠ GOOD - Precision met, coverage improving"
            elif prec_trending_up and cov_trending_up:
                status = "⚠ MODERATE - Both improving but targets not met"
            elif prec_trending_up or cov_trending_up:
                status = "⚠ WEAK - One metric improving, other stable/declining"
            else:
                status = "✗ POOR - Neither metric improving"
            
            print(f"  Overall Status: {status}")
            
            # Recommendations
            if not prec_trending_up and not cov_trending_up:
                print(f"  Recommendation: Consider increasing training episodes or adjusting reward weights")
            elif not prec_target_met:
                print(f"  Recommendation: Precision not reaching target - may need more episodes or higher precision weight")
            elif not cov_target_met:
                print(f"  Recommendation: Coverage not reaching target - may need more episodes or higher coverage weight")
    
    # Overall assessment
    print(f"\nOverall Assessment:")
    all_prec_met = all([per_class_precision_history[cls][-1] >= precision_target * 0.9 if len(per_class_precision_history[cls]) > 0 else False 
                       for cls in target_classes])
    all_cov_met = all([per_class_coverage_history[cls][-1] >= coverage_target * 0.9 if len(per_class_coverage_history[cls]) > 0 else False 
                      for cls in target_classes])
    
    if all_prec_met and all_cov_met:
        print(f"  ✓ Training is successfully maximizing precision and coverage for all classes")
    elif all_prec_met:
        print(f"  ⚠ Precision targets met, but coverage needs improvement")
    elif all_cov_met:
        print(f"  ⚠ Coverage targets met, but precision needs improvement")
    else:
        print(f"  ✗ Training may not be sufficient - both precision and coverage below targets")
        print(f"    Consider: increasing episodes, adjusting reward weights, or checking reward signal")
    
    # Analyze reward signal quality
    if len(episode_rewards_history) > 0:
        print(f"\nReward Signal Analysis:")
        avg_reward = np.mean(episode_rewards_history)
        final_rewards = episode_rewards_history[-min(10, len(episode_rewards_history)):]  # Last 10 episodes
        recent_avg_reward = np.mean(final_rewards)
        positive_reward_ratio = sum(1 for r in episode_rewards_history if r > 0) / len(episode_rewards_history)
        
        print(f"  Average reward: {avg_reward:.6f}")
        print(f"  Recent average (last {len(final_rewards)} episodes): {recent_avg_reward:.6f}")
        print(f"  Positive reward ratio: {positive_reward_ratio:.2%}")
        
        if recent_avg_reward > avg_reward * 1.1:
            print(f"  ✓ Reward signal is improving - policy is learning")
        elif recent_avg_reward > 0:
            print(f"  ⚠ Reward signal is positive but not improving - may need more training")
        else:
            print(f"  ✗ Reward signal is negative - penalties may be too high or precision/coverage not improving")
            print(f"    Check: reward weights (alpha, beta) and penalty weights (drift, js_penalty, overlap)")
    
    print("="*80)
    
    # ======================================================================
    # STEP 4: Verify Tracking and Create Plots
    # ======================================================================
    # Verify that we have tracked metrics for all episodes
    if verbose >= 1:
        print(f"\nVerifying tracking for {episodes} episodes...")
        for cls in target_classes:
            prec_len = len(per_class_precision_history[cls])
            cov_len = len(per_class_coverage_history[cls])
            if prec_len != episodes or cov_len != episodes:
                print(f"  [WARNING] Class {cls}: precision has {prec_len} entries, coverage has {cov_len} entries, "
                      f"expected {episodes} for each")
            else:
                print(f"  [OK] Class {cls}: tracked {prec_len} episodes for precision and coverage")
    
    print(f"\nCreating training plots...")
    plot_dir = f"{output_dir}/plots/"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot 1: Rewards per episode
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(episode_rewards_history) + 1), episode_rewards_history, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('RL Training: Rewards per Episode', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    reward_plot_path = f"{plot_dir}/rewards_per_episode.png"
    plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {reward_plot_path}")
    
    # Plot 2: Precision per class per episode
    plt.figure(figsize=(12, 6))
    for cls in target_classes:
        if len(per_class_precision_history[cls]) > 0:
            plt.plot(range(1, len(per_class_precision_history[cls]) + 1), 
                    per_class_precision_history[cls], 
                    linewidth=2, marker='o', markersize=3, 
                    label=f'Class {cls}')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('RL Training: Precision per Class per Episode', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    precision_plot_path = f"{plot_dir}/precision_per_class_per_episode.png"
    plt.savefig(precision_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {precision_plot_path}")
    
    # Plot 3: Coverage per class per episode
    plt.figure(figsize=(12, 6))
    for cls in target_classes:
        if len(per_class_coverage_history[cls]) > 0:
            plt.plot(range(1, len(per_class_coverage_history[cls]) + 1), 
                    per_class_coverage_history[cls], 
                    linewidth=2, marker='o', markersize=3, 
                    label=f'Class {cls}')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Coverage', fontsize=12)
    plt.title('RL Training: Coverage per Class per Episode', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    coverage_plot_path = f"{plot_dir}/coverage_per_class_per_episode.png"
    plt.savefig(coverage_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {coverage_plot_path}")
    
    # Add plotting data to results
    results["plotting_data"] = {
        "episode_rewards": episode_rewards_history,
        "per_class_precision": per_class_precision_history,
        "per_class_coverage": per_class_coverage_history,
        "plot_paths": {
            "rewards": reward_plot_path,
            "precision": precision_plot_path,
            "coverage": coverage_plot_path,
        }
    }
    
    return results

