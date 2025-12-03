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
except ImportError:
    SB3_AVAILABLE = False
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
            "eval_freq": 48_000,
            "n_eval_episodes": 4,
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
        max_cycles: int = 500,
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
            logger.info(f"  Evaluation configured to use TEST data")
        else:
            env_config_with_data["eval_on_test_data"] = False
            logger.info(f"  Evaluation configured to use TRAINING data")
        
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
        """Get default environment configuration."""
        return {
            "precision_target": 0.8,
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
            # Termination reason counters: disable overused reasons
            "max_termination_count_excellent_precision": 10,  # Disable after 10 uses
            "max_termination_count_both_targets": -1,         # Unlimited (default)
            "max_termination_count_high_precision": -1,       # Unlimited (default)
            "max_termination_count_both_close": -1,           # Unlimited (default)
        }
    
    def _setup_environments(
        self,
        env_data: Dict[str, Any],
        env_config_with_data: Dict[str, Any],
        target_classes: List[int],
        device: str
    ):
        """Set up training and evaluation environments for each class."""
        logger.info(f"\nCreating environments for {len(target_classes)} classes...")
        for target_class in target_classes:
            logger.info(f"  Setting up class {target_class}...")
            
            # Create training environment for this class
            train_env = self._create_env_for_class(
                env_data=env_data,
                env_config=env_config_with_data,
                target_class=target_class,
                device=device
            )
            train_env = Monitor(train_env, filename=None, allow_early_resets=True)
            self.envs[target_class] = train_env
            
            # Create evaluation environment for this class
            eval_env = self._create_env_for_class(
                env_data=env_data,
                env_config=env_config_with_data,
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
            model.learn(
                total_timesteps=timesteps_per_class,
                callback=[WandbCallback(gradient_save_freq=100)] + callbacks if callbacks else None,
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
        max_cycles: int = 500,
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

