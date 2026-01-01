"""
Stable-Baselines3 Trainer for Single-Agent Dynamic Anchors

This module provides a trainer class for training single-agent dynamic anchor
policies using Stable-Baselines3 (DDPG and SAC algorithms).

Architecture: One policy per class (fair comparison with multi-agent BenchMARL)
- Each class gets its own independent policy
- No class encoding in observation space (each policy only sees its class)
- Policies are trained separately with equal timestep allocation
"""

import numpy as np
import torch
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path
import os
import sys
import logging
import json
from datetime import datetime
import shutil
import yaml

logger = logging.getLogger(__name__)

# Import Stable-Baselines3
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    from stable_baselines3 import DDPG, SAC
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
    WANDB_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    WANDB_AVAILABLE = False
    logger.error("Stable-Baselines3 not available. Please install: pip install stable-baselines3")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_agentENV import SingleAgentAnchorEnv


def _get_algorithm_configs():
    """Get available algorithm configurations."""
    algorithm_map = {}
    
    if SB3_AVAILABLE:
        algorithm_map["ddpg"] = DDPG
        algorithm_map["sac"] = SAC
    
    return algorithm_map


class ResetTerminationCountersCallback(BaseCallback):
    """
    Callback to reset termination reason counters before evaluation.
    This ensures evaluation isn't affected by counters accumulated during training.
    """
    def __init__(self, eval_env, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
    
    def _on_step(self) -> bool:
        return True
    
    def on_evaluation_start(self) -> None:
        """Called before EvalCallback runs evaluation."""
        if hasattr(self.eval_env, '_reset_termination_counters'):
            self.eval_env._reset_termination_counters()
            if self.verbose > 0:
                logger.debug("  Reset termination counters before evaluation")


class LearningRateScheduleCallback(BaseCallback):
    """
    Callback to schedule learning rate during training.
    Reduces learning rate by a factor when evaluation performance plateaus.
    """
    def __init__(self, initial_lr: float, reduction_factor: float = 0.5, min_lr: float = 1e-6, 
                 patience: int = 3, verbose: int = 0):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.reduction_factor = reduction_factor
        self.min_lr = min_lr
        self.patience = patience
        self.best_mean_reward = -float('inf')
        self.patience_counter = 0
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        # Check if we should reduce learning rate based on evaluation
        # This is called after each rollout
        pass
    
    def on_evaluation_end(self, eval_callback: EvalCallback) -> None:
        """
        Called by EvalCallback when evaluation completes.
        Reduces learning rate if performance hasn't improved.
        """
        if hasattr(eval_callback, 'best_mean_reward'):
            current_reward = eval_callback.best_mean_reward
            
            if current_reward > self.best_mean_reward:
                self.best_mean_reward = current_reward
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.patience:
                    # Reduce learning rate
                    new_lr = max(self.current_lr * self.reduction_factor, self.min_lr)
                    if new_lr < self.current_lr:
                        self.current_lr = new_lr
                        # Update learning rate in the model
                        if hasattr(self.model, 'actor') and hasattr(self.model.actor, 'optimizer'):
                            for param_group in self.model.actor.optimizer.param_groups:
                                param_group['lr'] = new_lr
                        if hasattr(self.model, 'critic') and hasattr(self.model.critic, 'optimizer'):
                            for param_group in self.model.critic.optimizer.param_groups:
                                param_group['lr'] = new_lr
                        logger.info(f"  Learning rate reduced to: {new_lr:.2e} (patience: {self.patience_counter}/{self.patience})")
                        self.patience_counter = 0


class AnchorTrainerSB3:
    """
    Stable-Baselines3 trainer for single-agent dynamic anchors.
    
    Supports DDPG and SAC algorithms with one policy per class.
    This provides a fair comparison with multi-agent BenchMARL (when agents_per_class=1).
    """
    
    ALGORITHM_MAP = _get_algorithm_configs()
    
    def __init__(
        self,
        dataset_loader,
        algorithm: str = "ddpg",
        experiment_config: Optional[Dict[str, Any]] = None,
        algorithm_config: Optional[Dict[str, Any]] = None,
        output_dir: str = "./output/single_agent_sb3/",
        seed: int = 42
    ):
        """
        Initialize the SB3 trainer.
        
        Args:
            dataset_loader: TabularDatasetLoader instance
            algorithm: Algorithm to use ("ddpg" or "sac")
            experiment_config: Experiment configuration dictionary
            algorithm_config: Algorithm-specific configuration dictionary
            output_dir: Output directory for logs and checkpoints
            seed: Random seed
        """
        if not SB3_AVAILABLE:
            raise RuntimeError("Stable-Baselines3 is not installed. Please install: pip install stable-baselines3")
        
        self.dataset_loader = dataset_loader
        self.algorithm = algorithm.lower()
        
        if self.algorithm not in self.ALGORITHM_MAP:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Supported: {list(self.ALGORITHM_MAP.keys())}"
            )
        
        self.algorithm_class = self.ALGORITHM_MAP[self.algorithm]
        self.experiment_config = experiment_config or self._get_default_experiment_config()
        self.algorithm_config = algorithm_config or self._get_default_algorithm_config()
        self.output_dir = output_dir
        self.seed = seed
        
        # One model per class
        self.models: Dict[int, Any] = {}  # class -> model
        self.envs: Dict[int, Any] = {}  # class -> training env
        self.eval_envs: Dict[int, Any] = {}  # class -> eval env
        self.target_classes: List[int] = []
        self.experiment_folder = None
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_default_experiment_config(self) -> Dict[str, Any]:
        """Get default experiment configuration."""
        return {
            "total_timesteps": 72_000,
            "eval_freq": 3000,  # More frequent evaluations to catch better models (was 48_000)
            "n_eval_episodes": 20,  # More episodes for better statistics (was 4)
            "checkpoint_freq": 48_000,
            "log_interval": 10,
            "tensorboard_log": True,
        }
    
    def _get_default_algorithm_config(self) -> Dict[str, Any]:
        """Get default algorithm configuration."""
        base_config = {
            "learning_rate": 5e-4,  # Updated to match driver.py default (was 5e-5)
            "buffer_size": 1_000_000,
            "learning_starts": 1000,
            "batch_size": 512,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            "action_noise_sigma": 0.1,
            "policy_kwargs": {
                "net_arch": [256, 256]
            },
        }
        
        if self.algorithm == "sac":
            base_config.update({
                "ent_coef": "auto",
                "target_update_interval": 1,
                "target_entropy": "auto",
            })
        
        return base_config
    
    def setup_experiment(
        self,
        env_config: Optional[Dict[str, Any]] = None,
        target_classes: Optional[List[int]] = None,
        max_cycles: Optional[int] = None,  # If None, will read from env_config
        device: str = "cpu",
        eval_on_test_data: bool = True
    ):
        """
        Set up the training experiment.
        
        Args:
            env_config: Environment configuration dictionary
            target_classes: List of target classes (None = all classes)
            max_cycles: Maximum cycles per episode
            device: Device for training ("cpu", "cuda", "auto")
            eval_on_test_data: Whether to evaluate on test data
        """
        if self.dataset_loader.classifier is None:
            raise ValueError(
                "Classifier not trained yet. "
                "Call dataset_loader.create_classifier() and dataset_loader.train_classifier() first."
            )
        
        if self.dataset_loader.X_train_unit is None:
            raise ValueError(
                "Data not preprocessed yet. "
                "Call dataset_loader.preprocess_data() first."
            )
        
        logger.info("\n" + "="*80)
        logger.info("SETTING UP SINGLE-AGENT SB3 ANCHOR TRAINING EXPERIMENT")
        logger.info("(One Policy Per Class - Fair Comparison with Multi-Agent)")
        logger.info("="*80)
        
        # Get environment data
        env_data = self.dataset_loader.get_anchor_env_data()
        
        # Get target classes
        if target_classes is None:
            target_classes = sorted(np.unique(self.dataset_loader.y_train).tolist())
        
        self.target_classes = target_classes

        # Get default environment configuration
        if env_config is None:
            env_config = self._get_default_env_config()
        
        # Resolve episode length: if not explicitly provided, use env_config.
        if max_cycles is None:
            max_cycles = env_config.get("max_cycles")
            if max_cycles is None:
                raise ValueError("max_cycles must be specified in env_config. Check your YAML config file.")
            max_cycles = int(max_cycles)
        else:
            max_cycles = int(max_cycles)
        
        # Apply logging verbosity early based on config
        verbosity = env_config.get("logging_verbosity", "normal")
        level = logging.WARNING if verbosity == "quiet" else (logging.DEBUG if verbosity == "verbose" else logging.INFO)
        
        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
        
        # Set for all existing loggers
        for name in logging.Logger.manager.loggerDict:
            if isinstance(logging.Logger.manager.loggerDict[name], logging.Logger):
                log = logging.getLogger(name)
                log.setLevel(level)
                for handler in log.handlers:
                    handler.setLevel(level)
        
        # CRITICAL: Set min_coverage_floor dynamically to ensure box always covers at least the anchor instance
        # Use 1/n_samples from the dataset (to ensure at least one point is covered), 
        # or fall back to config default if dataset size unavailable
        # This prevents the coverage floor from being too high and blocking expansion during training
        n_samples = env_data["X_unit"].shape[0] if env_data.get("X_unit") is not None else None
        config_default = env_config.get("min_coverage_floor", 0.005)
        
        if n_samples is not None and n_samples > 0:
            # Use 1/n_samples to ensure at least one point is covered (the anchor instance)
            # For instance-based anchors, initial coverage is typically 0.001-0.002, so we need
            # a floor that's lower than that to allow expansion
            min_coverage_floor = 1.0 / n_samples
            # Use a very small lower bound (1e-6) instead of config_default to avoid blocking expansion
            # The config_default (0.005) is too high for instance-based anchors
            min_coverage_floor = max(min_coverage_floor, 1e-6)
        else:
            # Fall back to config default if dataset size unavailable
            min_coverage_floor = config_default
        
        # Ensure it's non-zero
        min_coverage_floor = max(min_coverage_floor, 1e-6)
        
        # Create environment configuration with data
        env_config_with_data = {
            **env_config,
            "X_min": env_data["X_min"],
            "X_range": env_data["X_range"],
            "max_cycles": max_cycles,
            "min_coverage_floor": min_coverage_floor,  # Override with dynamic value
        }
        
        logger.info(f"  Set min_coverage_floor={min_coverage_floor:.6f} for training (n_samples={n_samples if n_samples is not None else 'unknown'}, ensures box covers at least anchor instance)")
        
        if eval_on_test_data:
            if env_data.get("X_test_unit") is None or env_data.get("X_test_std") is None or env_data.get("y_test") is None:
                raise ValueError(
                    "eval_on_test_data=True requires test data. "
                    "Make sure dataset_loader has test data loaded and preprocessed."
                )
            env_config_with_data.update({
                "eval_on_test_data": True,
                "X_test_unit": env_data["X_test_unit"],
                "X_test_std": env_data["X_test_std"],
                "y_test": env_data["y_test"],
            })
            logger.info(f"  Evaluation configured to use TEST data")
        else:
            env_config_with_data["eval_on_test_data"] = False
            logger.info(f"  Evaluation configured to use TRAINING data")
        
        # Compute k-means centroids for diversity across episodes
        # This ensures each episode can start from a different cluster centroid
        # Using 10 centroids per class for diversity (same as multi-agent when agents_per_class=1)
        # NOTE: Actual number will be adapted based on dataset size (see auto_adapt_clusters)
        n_clusters_per_class = 10
        logger.info(f"\nComputing k-means centroids (k={n_clusters_per_class} max) for each class for diversity...")
        logger.info(f"  Note: Cluster count will be adapted based on dataset size and data distribution")
        
        try:
            from utils.clusters import compute_cluster_centroids_per_class
            
            # Always compute centroids on training data
            X_data = env_data["X_unit"]
            y_data = env_data["y"]
            
            # Use adaptive clustering: adjust cluster count based on dataset size
            # and check for scattered data distribution
            cluster_centroids_per_class = compute_cluster_centroids_per_class(
                X_unit=X_data,
                y=y_data,
                n_clusters_per_class=n_clusters_per_class,
                random_state=self.seed if hasattr(self, 'seed') else 42,
                min_samples_per_cluster=1,
                auto_adapt_clusters=True,  # Adapt cluster count to dataset size
                check_data_scatter=True    # Check if data is scattered (use mean if so)
            )
            
            # Verify we have enough centroids for each class and log class statistics
            for cls in target_classes:
                class_mask = (y_data == cls)
                n_class_samples = class_mask.sum()
                logger.info(f"   Class {cls}: {n_class_samples} training samples")
                
                if cls in cluster_centroids_per_class:
                    n_centroids = len(cluster_centroids_per_class[cls])
                    if n_centroids < n_clusters_per_class:
                        logger.warning(
                            f"   Class {cls}: Only {n_centroids} centroids computed "
                            f"(requested {n_clusters_per_class}). "
                            f"May not have enough samples for k-means."
                        )
                    else:
                        logger.info(f"   Class {cls}: {n_centroids} centroids computed")
                else:
                    logger.warning(f"   Class {cls}: No centroids computed")
            
            # Set cluster centroids in env_config
            env_config_with_data["cluster_centroids_per_class"] = cluster_centroids_per_class
            logger.info("   ✓ Cluster centroids set in environment config")
        except ImportError as e:
            logger.warning(f"   ⚠ Could not compute cluster centroids: {e}")
            logger.warning(f"  Install sklearn: pip install scikit-learn")
            logger.warning(f"  Falling back to mean centroid per class (all episodes will start from same point)")
            env_config_with_data["cluster_centroids_per_class"] = None
        except Exception as e:
            logger.warning(f"   ⚠ Error computing cluster centroids: {e}")
            logger.warning(f"  Falling back to mean centroid per class (all episodes will start from same point)")
            env_config_with_data["cluster_centroids_per_class"] = None
        
        # Sample instances per class for instance-based training (mixed initialization)
        training_instance_ratio = env_config_with_data.get("training_instance_ratio", 0.3)
        use_adaptive_ratios = env_config_with_data.get("use_adaptive_instance_ratios", True)  # Enable by default
        
        if training_instance_ratio > 0.0:
            np.random.seed(self.seed if hasattr(self, 'seed') else 42)
            
            # Compute class-specific ratios based on class imbalance (if adaptive mode enabled)
            training_instance_ratios_per_class = {}
            if use_adaptive_ratios and len(target_classes) > 1:
                # Compute class counts
                class_counts = {cls: (y_data == cls).sum() for cls in target_classes}
                min_count = min(class_counts.values())
                max_count = max(class_counts.values())
                imbalance_ratio = max_count / min_count if min_count > 0 else 1.0
                
                # Use warning level so these important setup messages show even in quiet mode
                logger.warning(f"\nComputing adaptive class-specific training instance ratios...")
                logger.warning(f"  Class imbalance ratio: {imbalance_ratio:.2f}:1 (max/min)")
                logger.warning(f"  Base ratio: {training_instance_ratio:.1%}")
                
                # Only use adaptive ratios if there's significant imbalance (> 1.5:1)
                if imbalance_ratio > 1.5:
                    for cls in target_classes:
                        count = class_counts[cls]
                        # Higher ratio for minority classes (inversely proportional to class size)
                        # Use square root scaling to avoid extreme ratios
                        size_factor = (max_count / count) ** 0.5 if count > 0 else 1.0
                        adaptive_ratio = training_instance_ratio * size_factor
                        # Ensure ratio is at least the base ratio, but don't cap it (respect user's configuration)
                        adaptive_ratio = max(training_instance_ratio, adaptive_ratio)
                        # Cap at 1.0 (100%) maximum since ratio represents probability
                        adaptive_ratio = min(1.0, adaptive_ratio)
                        training_instance_ratios_per_class[cls] = adaptive_ratio
                        
                        minority_status = "minority" if count < min_count * 1.5 else "majority"
                        logger.warning(f"   Class {cls}: {count} samples ({minority_status}) → ratio: {adaptive_ratio:.1%}")
                else:
                    # Balanced dataset - use same ratio for all classes
                    logger.warning(f"  Dataset is relatively balanced - using uniform ratio for all classes")
                    for cls in target_classes:
                        training_instance_ratios_per_class[cls] = training_instance_ratio
            else:
                # Use uniform ratio for all classes
                for cls in target_classes:
                    training_instance_ratios_per_class[cls] = training_instance_ratio
            
            # Sample instances per class (use max ratio to ensure enough instances)
            max_ratio = max(training_instance_ratios_per_class.values())
            n_instances_per_class = max(20, int(10 / max_ratio))  # Ensure enough instances
            training_instances_per_class = {}
            
            logger.info(f"\nSampling instances per class for instance-based training...")
            logger.info(f"  Sampling {n_instances_per_class} instances per class (based on max ratio: {max_ratio:.1%})")
            logger.info(f"  CRITICAL: Filtering instances where classifier prediction matches target_class")
            logger.info(f"  This ensures original_prediction == target_class for instance-based anchors")
            
            # Get classifier to filter instances by prediction
            classifier = self.dataset_loader.get_classifier()
            classifier.eval()
            import torch
            from utils.device_utils import get_device_str
            device_str = get_device_str(device) if device != "auto" else "cpu"
            device_torch = torch.device(device_str)
            
            for cls in target_classes:
                class_mask = (y_data == cls)
                class_indices = np.where(class_mask)[0]
                n_class_samples = len(class_indices)
                class_ratio = training_instance_ratios_per_class[cls]
                
                logger.info(f"   Class {cls}: {n_class_samples} training samples available (ratio: {class_ratio:.1%})")
                
                # CRITICAL FIX: Filter instances where classifier prediction matches target_class
                # This ensures original_prediction == target_class, preventing precision calculation issues
                if len(class_indices) > 0:
                    # Get predictions for all class instances
                    X_class_std = env_data["X_std"][class_indices]
                    with torch.no_grad():
                        X_tensor = torch.from_numpy(X_class_std.astype(np.float32)).to(device_torch)
                        logits = classifier(X_tensor)
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                        predictions = np.argmax(probs, axis=1)
                    
                    # Filter: keep only instances where prediction matches target_class
                    prediction_match_mask = (predictions == cls)
                    matching_indices = class_indices[prediction_match_mask]
                    n_matching = len(matching_indices)
                    
                    logger.info(f"   Class {cls}: {n_matching}/{n_class_samples} instances have prediction matching target_class")
                    
                    if n_matching == 0:
                        logger.warning(
                            f"   Class {cls}: No instances found where classifier prediction matches target_class! "
                            f"This will cause issues with instance-based anchors. "
                            f"Falling back to all instances (may cause low precision)."
                        )
                        matching_indices = class_indices  # Fallback to all instances
                        n_matching = len(matching_indices)
                    
                    if n_matching >= n_instances_per_class:
                        # Randomly sample from matching instances
                        selected_indices = np.random.choice(matching_indices, size=n_instances_per_class, replace=False)
                        training_instances_per_class[cls] = X_data[selected_indices].tolist()
                        logger.info(f"   Class {cls}: {n_instances_per_class} instances sampled from {n_matching} matching instances")
                    elif n_matching > 0:
                        # Use all matching instances if fewer than requested
                        training_instances_per_class[cls] = X_data[matching_indices].tolist()
                        logger.warning(
                            f"   Class {cls}: Only {n_matching} matching instances available "
                            f"(requested {n_instances_per_class}). Using all matching instances."
                        )
                    else:
                        logger.error(f"   Class {cls}: No matching instances available for sampling! This will cause initialization failures.")
                else:
                    logger.error(f"   Class {cls}: No instances available for sampling! This will cause initialization failures.")
            
            # Store training instances and class-specific ratios in env_config
            env_config_with_data["training_instances_per_class"] = training_instances_per_class
            env_config_with_data["training_instance_ratios_per_class"] = training_instance_ratios_per_class
            logger.info("   ✓ Training instances and class-specific ratios set in environment config")
        else:
            env_config_with_data["training_instances_per_class"] = None
            logger.info("   Training instance ratio is 0.0 - using centroid-based initialization only")
        
        # Create experiment folder
        timestamp = datetime.now().strftime("%y_%m_%d-%H_%M_%S")
        experiment_id = f"{self.algorithm}_single_agent_sb3_{timestamp}"
        self.experiment_folder = os.path.join(self.output_dir, experiment_id)
        os.makedirs(self.experiment_folder, exist_ok=True)
        
        # Set up environments
        self._setup_environments(
            env_data=env_data,
            env_config_with_data=env_config_with_data,
            target_classes=target_classes,
            device=device
        )
        
        # Create one model per class
        self._create_models(device=device)
        
        logger.info(f"\nExperiment setup complete:")
        logger.info(f"  Algorithm: {self.algorithm.upper()}")
        logger.info(f"  Target classes: {target_classes}")
        logger.info(f"  Policies: {len(self.models)} (one per class)")
        logger.info(f"  Max cycles per episode: {max_cycles}")
        logger.info(f"  Experiment folder: {self.experiment_folder}")
        logger.info("="*80)
    
    def _get_default_env_config(self) -> Dict[str, Any]:
        """Get default environment configuration from YAML file if available, otherwise use defaults."""
        # Try to load from config file
        config_path = os.path.join(os.path.dirname(__file__), "conf", "anchor_single.yaml")
        env_config = {}
        
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    if config_data and "env_config" in config_data:
                        env_config = config_data["env_config"]
                        # Also load logging_verbosity and max_cycles from top-level YAML if present
                        if "logging_verbosity" in config_data:
                            env_config["logging_verbosity"] = config_data["logging_verbosity"]
                        if "max_cycles" in config_data:
                            env_config["max_cycles"] = config_data["max_cycles"]
                        logger.info(f"Loaded environment config from: {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}. Using defaults.")
        
        # Set defaults (will be overridden by config file values if present)
        defaults = {
            "precision_target": 0.95,
            "coverage_target": 0.1,
            "use_perturbation": True,
            "perturbation_mode": "adaptive",
            "n_perturb": 4096,
            "step_fracs": [0.005, 0.01, 0.02],
            "min_width": 0.05,
            "alpha": 0.7,
            "beta": 0.6,
            "gamma": 0.1,
            "precision_blend_lambda": 0.5,
            "drift_penalty_weight": 0.05,
            "min_coverage_floor": 0.005,
            "js_penalty_weight": 0.05,
            "initial_window": 0.1,
            "max_action_scale": 0.1,
            "min_absolute_step": 0.001,
            "use_class_centroids": True,
            # Coverage bonus weights (defaults match reduced values)
            "coverage_bonus_weight_met": 0.01,
            "coverage_bonus_weight_high_prec": 0.03,
            "coverage_bonus_weight_high_prec_progress": 0.07,
            "coverage_bonus_weight_high_prec_distance": 0.02,
            "coverage_bonus_weight_reasonable_prec": 0.01,
            "coverage_bonus_weight_reasonable_prec_progress": 0.02,
            # Target class bonus weight
            "target_class_bonus_weight": 0.02,
            # Termination reason counters: disable overused reasons
            # Strategy: Higher limits for better outcomes, lower limits for easier/less ideal outcomes
            "max_termination_count_both_targets": -1,
            "max_termination_count_high_precision": 200,
            "max_termination_count_both_close": 50,
            "max_termination_count_excellent_precision": 30,
            "logging_verbosity": "normal",  # Default logging verbosity
        }
        
        # Merge: defaults first, then config file values override
        final_config = {**defaults, **env_config}
        # Ensure logging_verbosity is set (default to "normal" if not in config)
        if "logging_verbosity" not in final_config:
            final_config["logging_verbosity"] = "normal"
        return final_config
    
    def _setup_environments(
        self,
        env_data: Dict[str, Any],
        env_config_with_data: Dict[str, Any],
        target_classes: List[int],
        device: str
    ):
        """Set up training and evaluation environments for each class."""
        logger.info(f"\nCreating environments for {len(target_classes)} classes...")
        
        # Log class data statistics before creating environments
        y_data = env_data["y"]
        for cls in target_classes:
            class_mask = (y_data == cls)
            n_class_samples = class_mask.sum()
            logger.info(f"  Class {cls}: {n_class_samples} training samples")
        
        for target_class in target_classes:
            logger.info(f"  Setting up class {target_class}...")
            
            # Create training environment for this class
            train_env_config = {**env_config_with_data, "mode": "training"}
            train_env = self._create_env_for_class(
                env_data=env_data,
                env_config=train_env_config,
                target_class=target_class,
                device=device
            )
            train_env = Monitor(train_env, filename=None, allow_early_resets=True)
            self.envs[target_class] = train_env
            
            # Create evaluation environment for this class (with evaluation mode)
            eval_env_config = {**env_config_with_data, "mode": "evaluation"}
            eval_env = self._create_env_for_class(
                env_data=env_data,
                env_config=eval_env_config,
                target_class=target_class,
                device=device
            )
            eval_env = Monitor(eval_env, filename=None, allow_early_resets=True)
            self.eval_envs[target_class] = eval_env
    
    def _create_env_for_class(self, env_data, env_config, target_class: int, device):
        """
        Create a SingleAgentAnchorEnv for a specific class.
        
        Args:
            env_data: Environment data dictionary
            env_config: Environment configuration dictionary
            target_class: Target class for this environment
            device: Device for training
        
        Returns:
            SingleAgentAnchorEnv instance for the specified class
        """
        classifier = self.dataset_loader.get_classifier()
        
        env = SingleAgentAnchorEnv(
            X_unit=env_data["X_unit"],
            X_std=env_data["X_std"],
            y=env_data["y"],
            feature_names=env_data["feature_names"],
            classifier=classifier,
            device=device,
            target_class=target_class,
            env_config=env_config
        )
        
        return env
    
    def _create_models(self, device: str = "cpu"):
        """Create one SB3 model per class."""
        from utils.device_utils import get_device_str
        device_str = get_device_str(device) if device != "auto" else "auto"
        
        for target_class in self.target_classes:
            env = self.envs[target_class]
            
            # Create action noise for DDPG
            action_noise = None
            if self.algorithm == "ddpg":
                n_actions = env.action_space.shape[0]
                action_noise = NormalActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=self.algorithm_config["action_noise_sigma"] * np.ones(n_actions)
                )
            
            # Prepare model arguments
            model_kwargs = {
                "policy": "MlpPolicy",
                "env": env,
                "learning_rate": self.algorithm_config["learning_rate"],
                "buffer_size": self.algorithm_config["buffer_size"],
                "learning_starts": self.algorithm_config["learning_starts"],
                "batch_size": self.algorithm_config["batch_size"],
                "tau": self.algorithm_config["tau"],
                "gamma": self.algorithm_config["gamma"],
                "train_freq": self.algorithm_config["train_freq"],
                "gradient_steps": self.algorithm_config["gradient_steps"],
                "policy_kwargs": self.algorithm_config["policy_kwargs"],
                "verbose": 1,
                "device": device_str,
            }
            
            if self.algorithm == "ddpg":
                model_kwargs["action_noise"] = action_noise
            elif self.algorithm == "sac":
                model_kwargs["ent_coef"] = self.algorithm_config.get("ent_coef", "auto")
                model_kwargs["target_update_interval"] = self.algorithm_config.get("target_update_interval", 1)
                model_kwargs["target_entropy"] = self.algorithm_config.get("target_entropy", "auto")
            
            # TensorBoard logging (separate log for each class)
            if self.experiment_config.get("tensorboard_log", True):
                tensorboard_log = os.path.join(self.experiment_folder, "tensorboard", f"class_{target_class}")
                model_kwargs["tensorboard_log"] = tensorboard_log
            
            # Create model for this class
            self.models[target_class] = self.algorithm_class(**model_kwargs)
            
            logger.info(f"  Model for class {target_class} created")
            logger.info(f"    Observation space: {env.observation_space}")
            logger.info(f"    Action space: {env.action_space}")
    
    def train(self):
        """Train all models (one per class) with equal timestep allocation."""
        if not self.models:
            raise ValueError("Models not created. Call setup_experiment() first.")
        
        logger.info("\n" + "="*80)
        logger.info("STARTING SINGLE-AGENT SB3 ANCHOR TRAINING")
        logger.info(f"Training {len(self.models)} policies (one per class)")
        logger.info("="*80)
        
        total_timesteps = self.experiment_config["total_timesteps"]
        n_classes = len(self.target_classes)
        timesteps_per_class = total_timesteps // n_classes
        
        logger.info(f"Total timesteps: {total_timesteps}")
        logger.info(f"Timesteps per class: {timesteps_per_class}")
        logger.info(f"Algorithm: {self.algorithm.upper()}\n")

        # Fix wandb warning: patch tensorboard before init when using multiple event log directories
        # Get the root tensorboard log directory (parent of all class-specific logs)
        if not WANDB_AVAILABLE or os.environ.get("DISABLE_WANDB", "0") == "1":
            if os.environ.get("DISABLE_WANDB", "0") == "1":
                logger.debug("wandb disabled via DISABLE_WANDB environment variable, skipping tensorboard patch")
            else:
                logger.warning("wandb not available, skipping tensorboard patch")
        elif self.experiment_config.get("tensorboard_log", True):
            root_tensorboard_log = os.path.join(self.experiment_folder, "tensorboard")
            try:
                # Import wandb module to ensure it's in local scope
                import wandb as wandb_module
                # Patch tensorboard before wandb.init() to avoid warning about multiple event log directories
                if hasattr(wandb_module, 'tensorboard') and hasattr(wandb_module.tensorboard, 'patch'):
                    wandb_module.tensorboard.patch(root_logdir=root_tensorboard_log)
                    logger.debug(f"Patched wandb tensorboard with root_logdir={root_tensorboard_log}")
                else:
                    # Fallback for older wandb versions
                    import wandb.integration.tensorboard as wandb_tb
                    wandb_tb.patch(root_logdir=root_tensorboard_log)
                    logger.debug(f"Patched wandb tensorboard (using integration) with root_logdir={root_tensorboard_log}")
            except Exception as e:
                logger.warning(f"Could not patch wandb tensorboard: {e}")

        if WANDB_AVAILABLE and os.environ.get("DISABLE_WANDB", "0") != "1":
            wandb.init(project="single-agent-anchor-rl", name=f"single-agent-anchor-rl_{self.dataset_loader.dataset_name}", 
            config=self.algorithm_config, sync_tensorboard=True)
            wandb.config.update({
                "total_timesteps": total_timesteps,
                "timesteps_per_class": timesteps_per_class,
                "algorithm": self.algorithm,
                "learning_rate": self.algorithm_config["learning_rate"],
                "buffer_size": self.algorithm_config["buffer_size"],
            })
        
        # Train each model separately
        for target_class in self.target_classes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training policy for class {target_class}")
            logger.info(f"{'='*60}")
            
            model = self.models[target_class]
            env = self.envs[target_class]
            eval_env = self.eval_envs[target_class]
            
            # Create callbacks for this class
            callbacks = []
            
            # Checkpoint callback (per class)
            if self.experiment_config.get("checkpoint_freq", 0) > 0:
                checkpoint_dir = os.path.join(self.experiment_folder, "checkpoints", f"class_{target_class}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_callback = CheckpointCallback(
                    save_freq=self.experiment_config["checkpoint_freq"],
                    save_path=checkpoint_dir,
                    name_prefix=f"checkpoint_class_{target_class}"
                )
                callbacks.append(checkpoint_callback)
            
            # Evaluation callback (per class)
            # eval_env is already set to "evaluation" mode, so termination counters
            # will be automatically reset in reset() method
            eval_callback = None
            if self.experiment_config.get("eval_freq", 0) > 0:
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=os.path.join(self.experiment_folder, "best_model", f"class_{target_class}"),
                    log_path=os.path.join(self.experiment_folder, "evaluations", f"class_{target_class}"),
                    eval_freq=self.experiment_config["eval_freq"],
                    n_eval_episodes=self.experiment_config.get("n_eval_episodes", 4),
                    deterministic=True,
                    render=False
                )
                callbacks.append(eval_callback)
            
            # Learning rate scheduling callback (optional, reduces LR on plateau)
            use_lr_schedule = self.algorithm_config.get("use_lr_schedule", False)
            if use_lr_schedule and eval_callback is not None:
                initial_lr = self.algorithm_config.get("learning_rate", 5e-4)
                lr_schedule_callback = LearningRateScheduleCallback(
                    initial_lr=initial_lr,
                    reduction_factor=0.5,
                    min_lr=1e-6,
                    patience=3
                )
                # Link LR schedule to eval callback
                lr_schedule_callback.model = model
                # Note: This is a simplified version. For full implementation, 
                # you'd need to properly integrate with EvalCallback's evaluation results
                # callbacks.append(lr_schedule_callback)

            # Train this model
            train_callbacks = callbacks.copy() if callbacks else []
            if WANDB_AVAILABLE and os.environ.get("DISABLE_WANDB", "0") != "1":
                train_callbacks = [WandbCallback(gradient_save_freq=100)] + train_callbacks
            model.learn(
                total_timesteps=timesteps_per_class,
                callback=train_callbacks if train_callbacks else None,
                log_interval=self.experiment_config.get("log_interval", 10),
                progress_bar=True
            )
            
            # Save final model for this class
            final_model_path = os.path.join(self.experiment_folder, "final_model", f"class_{target_class}")
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            model.save(final_model_path)
            logger.info(f"  Final model for class {target_class} saved to: {final_model_path}")
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Results saved to: {self.experiment_folder}")
        
        # Save wandb run URL for later reference
        if WANDB_AVAILABLE and os.environ.get("DISABLE_WANDB", "0") != "1":
            try:
                if wandb.run is not None:
                    wandb_run_url = wandb.run.url if hasattr(wandb.run, 'url') else None
                    if wandb_run_url is None:
                        try:
                            entity = wandb.run.entity if hasattr(wandb.run, 'entity') else None
                            project = wandb.run.project if hasattr(wandb.run, 'project') else None
                            run_id = wandb.run.id if hasattr(wandb.run, 'id') else None
                            if entity and project and run_id:
                                wandb_run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
                        except Exception:
                            pass
                    
                    if wandb_run_url:
                        wandb_url_file = os.path.join(self.experiment_folder, "wandb_run_url.txt")
                        with open(wandb_url_file, 'w') as f:
                            f.write(wandb_run_url)
                        logger.info(f"✓ Wandb run URL saved to: {wandb_url_file}")
                        logger.info(f"  URL: {wandb_run_url}")
            except Exception as e:
                logger.debug(f"Could not save wandb run URL: {e}")
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate all trained models (one per class)."""
        if not self.models:
            raise ValueError("Models not trained. Call train() first.")
        
        logger.info(f"\nEvaluating {len(self.models)} models on {n_episodes} episodes each...")
        
        results = {}
        all_rewards = []
        
        for target_class in self.target_classes:
            model = self.models[target_class]
            eval_env = self.eval_envs[target_class]
            
            # Evaluation environment is already set to "evaluation" mode
            # Termination counters will be reset automatically in reset()
            logger.info(f"\nEvaluating policy for class {target_class}...")
            mean_reward, std_reward = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=n_episodes,
                deterministic=True
            )
            
            results[f"class_{target_class}"] = {
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "n_episodes": n_episodes
            }
            all_rewards.append(mean_reward)
            
            logger.info(f"  Class {target_class}: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Overall statistics
        overall_mean = np.mean(all_rewards)
        overall_std = np.std(all_rewards)
        
        results["overall"] = {
            "mean_reward": overall_mean,
            "std_reward": overall_std,
            "n_classes": len(self.target_classes)
        }
        
        logger.info(f"\nOverall mean reward: {overall_mean:.2f} +/- {overall_std:.2f}")
        
        return results
    
    def get_checkpoint_path(self) -> str:
        """Get the path to the experiment folder."""
        return self.experiment_folder
    
    def reload_experiment(
        self, 
        experiment_dir: str,
        env_config: Optional[Dict[str, Any]] = None,
        target_classes: Optional[List[int]] = None,
        max_cycles: Optional[int] = None,  # If None, will read from env_config
        device: str = "cpu",
        eval_on_test_data: bool = True
    ):
        """
        Reload an experiment from a checkpoint directory.
        
        This will set up environments and then load the saved models.
        If target_classes is not provided, will try to infer from saved models.
        """
        self.experiment_folder = experiment_dir
        
        # First, try to infer target_classes from saved models if not provided
        if target_classes is None:
            # Look for saved models to infer classes
            final_model_dir = os.path.join(experiment_dir, "final_model")
            if os.path.exists(final_model_dir):
                class_dirs = [d for d in os.listdir(final_model_dir) if d.startswith("class_")]
                target_classes = sorted([int(d.split("_")[1]) for d in class_dirs if d.split("_")[1].isdigit()])
                logger.info(f"Inferred target classes from saved models: {target_classes}")
        
        if target_classes is None or len(target_classes) == 0:
            # Fallback: use all unique classes from dataset
            target_classes = sorted(np.unique(self.dataset_loader.y_train).tolist())
            logger.info(f"Using all classes from dataset: {target_classes}")
        
        # Set up environments first (needed for loading models)
        logger.info("Setting up environments for reload...")
        
        # Get environment data
        env_data = self.dataset_loader.get_anchor_env_data()
        
        # Get default environment configuration
        if env_config is None:
            env_config = self._get_default_env_config()
        
        # Resolve episode length: if not explicitly provided, use env_config.
        if max_cycles is None:
            max_cycles = env_config.get("max_cycles")
            if max_cycles is None:
                raise ValueError("max_cycles must be specified in env_config. Check your YAML config file.")
            max_cycles = int(max_cycles)
        else:
            max_cycles = int(max_cycles)
        
        # Create environment configuration with data
        env_config_with_data = {
            **env_config,
            "X_min": env_data["X_min"],
            "X_range": env_data["X_range"],
            "max_cycles": max_cycles,
        }
        
        if eval_on_test_data:
            if env_data.get("X_test_unit") is None or env_data.get("X_test_std") is None or env_data.get("y_test") is None:
                raise ValueError(
                    "eval_on_test_data=True requires test data. "
                    "Make sure dataset_loader has test data loaded and preprocessed."
                )
            env_config_with_data.update({
                "eval_on_test_data": True,
                "X_test_unit": env_data["X_test_unit"],
                "X_test_std": env_data["X_test_std"],
                "y_test": env_data["y_test"],
            })
        else:
            env_config_with_data["eval_on_test_data"] = False
        
        # Compute k-means centroids for diversity (same as setup_experiment)
        # This ensures reloaded experiments also have cluster centroids
        n_clusters_per_class = 10
        logger.info(f"\nComputing k-means centroids (k={n_clusters_per_class}) for each class...")
        try:
            from utils.clusters import compute_cluster_centroids_per_class
            
            # Always compute centroids on training data
            X_data = env_data["X_unit"]
            y_data = env_data["y"]
            
            # Use adaptive clustering: adjust cluster count based on dataset size
            # and check for scattered data distribution
            cluster_centroids_per_class = compute_cluster_centroids_per_class(
                X_unit=X_data,
                y=y_data,
                n_clusters_per_class=n_clusters_per_class,
                random_state=self.seed if hasattr(self, 'seed') else 42,
                min_samples_per_cluster=1,
                auto_adapt_clusters=True,  # Adapt cluster count to dataset size
                check_data_scatter=True    # Check if data is scattered (use mean if so)
            )
            
            # Verify we have enough centroids for each class and log class statistics
            for cls in target_classes:
                class_mask = (y_data == cls)
                n_class_samples = class_mask.sum()
                logger.info(f"   Class {cls}: {n_class_samples} training samples")
                
                if cls in cluster_centroids_per_class:
                    n_centroids = len(cluster_centroids_per_class[cls])
                    if n_centroids < n_clusters_per_class:
                        logger.warning(
                            f"   Class {cls}: Only {n_centroids} centroids computed "
                            f"(requested {n_clusters_per_class}). "
                            f"May not have enough samples for k-means."
                        )
                    else:
                        logger.info(f"   Class {cls}: {n_centroids} centroids computed")
                else:
                    logger.warning(f"   Class {cls}: No centroids computed")
            
            # Set cluster centroids in env_config
            env_config_with_data["cluster_centroids_per_class"] = cluster_centroids_per_class
            logger.info("   ✓ Cluster centroids set in environment config")
        except ImportError as e:
            logger.warning(f"   ⚠ Could not compute cluster centroids: {e}")
            logger.warning(f"  Install sklearn: pip install scikit-learn")
            logger.warning(f"  Falling back to mean centroid per class")
            env_config_with_data["cluster_centroids_per_class"] = None
        except Exception as e:
            logger.warning(f"   ⚠ Error computing cluster centroids: {e}")
            logger.warning(f"  Falling back to mean centroid per class")
            env_config_with_data["cluster_centroids_per_class"] = None
        
        # Set up environments (without creating models)
        self.target_classes = target_classes
        self._setup_environments(
            env_data=env_data,
            env_config_with_data=env_config_with_data,
            target_classes=target_classes,
            device=device
        )
        
        # Now load models for each class
        logger.info(f"\nLoading models from: {experiment_dir}")
        
        for target_class in self.target_classes:
            # Try final model first
            model_path = os.path.join(experiment_dir, "final_model", f"class_{target_class}")
            if not os.path.exists(model_path + ".zip"):
                # Try best model
                model_path = os.path.join(experiment_dir, "best_model", f"class_{target_class}", f"class_{target_class}")
            
            if not os.path.exists(model_path + ".zip"):
                logger.warning(f"Model for class {target_class} not found at {model_path}")
                # Create a new model for this class if not found
                logger.info(f"  Creating new model for class {target_class}")
                continue
            
            env = self.envs[target_class]
            logger.info(f"  Loading model for class {target_class} from: {model_path}")
            self.models[target_class] = self.algorithm_class.load(model_path, env=env)

