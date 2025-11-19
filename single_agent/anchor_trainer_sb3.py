"""
Stable-Baselines3 Trainer for Single-Agent Dynamic Anchors

This module provides a trainer class for training single-agent dynamic anchor
policies using Stable-Baselines3 (DDPG and SAC algorithms).

The key difference from multi-agent BenchMARL:
- Single policy that estimates bounds, precision, and coverage for all classes
- Target class is included in the observation space
- One model handles all classes instead of separate models per class
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


class AnchorTrainerSB3:
    """
    Stable-Baselines3 trainer for single-agent dynamic anchors.
    
    Supports DDPG and SAC algorithms with a single policy for all classes.
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
        
        self.model = None
        self.env = None
        self.eval_env = None
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
            "learning_rate": 5e-5,
            "buffer_size": 1_000_000,
            "learning_starts": 1000,
            "batch_size": 256,
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
        max_cycles: int = 1000,
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
        logger.info("="*80)
        
        # Get environment data
        env_data = self.dataset_loader.get_anchor_env_data()
        
        # Get target classes
        if target_classes is None:
            target_classes = sorted(np.unique(self.dataset_loader.y_train).tolist())
        
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
        
        # Create a multi-class environment wrapper that includes target class in observation
        # For now, we'll create separate environments per class but use a single policy
        # The observation space will include the target class as a one-hot encoding
        
        # Create training environment (we'll cycle through classes)
        self.env = self._create_multi_class_env(
            env_data=env_data,
            env_config=env_config_with_data,
            target_classes=target_classes,
            device=device
        )
        
        # Create evaluation environment
        self.eval_env = self._create_multi_class_env(
            env_data=env_data,
            env_config=env_config_with_data,
            target_classes=target_classes,
            device=device
        )
        
        # Wrap with Monitor for logging
        self.env = Monitor(self.env, filename=None, allow_early_resets=True)
        self.eval_env = Monitor(self.eval_env, filename=None, allow_early_resets=True)
        
        # Create experiment folder
        timestamp = datetime.now().strftime("%y_%m_%d-%H_%M_%S")
        experiment_id = f"{self.algorithm}_single_agent_sb3_{timestamp}"
        self.experiment_folder = os.path.join(self.output_dir, experiment_id)
        os.makedirs(self.experiment_folder, exist_ok=True)
        
        # Create model
        self._create_model(device=device)
        
        logger.info(f"Experiment setup complete:")
        logger.info(f"  Algorithm: {self.algorithm.upper()}")
        logger.info(f"  Target classes: {target_classes}")
        logger.info(f"  Max cycles per episode: {max_cycles}")
        logger.info(f"  Experiment folder: {self.experiment_folder}")
        logger.info("="*80)
    
    def _get_default_env_config(self) -> Dict[str, Any]:
        """Get default environment configuration."""
        return {
            "precision_target": 0.8,
            "coverage_target": 0.02,
            "use_perturbation": False,
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
        }
    
    def _create_multi_class_env(self, env_data, env_config, target_classes, device):
        """
        Create a multi-class environment wrapper.
        
        This wrapper cycles through target classes and includes the target class
        in the observation space so a single policy can handle all classes.
        """
        from gymnasium import spaces
        import gymnasium as gym
        
        # Get classifier from dataset loader
        classifier = self.dataset_loader.get_classifier()
        
        class MultiClassAnchorEnv(gym.Env):
            """Wrapper that cycles through classes and includes class in observation."""
            
            def __init__(self, env_data, env_config, target_classes, device, classifier, seed=None):
                super().__init__()
                self.target_classes = target_classes
                self.n_classes = len(target_classes)
                self.class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}
                self.current_class_idx = 0
                self.fixed_class_idx = None  # For inference: set to specific class index
                self.device = device
                
                # Create base environment for first class
                self.base_env = SingleAgentAnchorEnv(
                    X_unit=env_data["X_unit"],
                    X_std=env_data["X_std"],
                    y=env_data["y"],
                    feature_names=env_data["feature_names"],
                    classifier=classifier,
                    device=device,
                    target_class=target_classes[0],
                    env_config=env_config
                )
                
                # Original observation space: (2 * n_features + 2,)
                # Extended: (2 * n_features + 2 + n_classes,) to include one-hot class encoding
                base_obs_dim = self.base_env.observation_space.shape[0]
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(base_obs_dim + self.n_classes,),
                    dtype=np.float32
                )
                
                self.action_space = self.base_env.action_space
                self.metadata = self.base_env.metadata
                
                if seed is not None:
                    self.seed(seed)
            
            def reset(self, seed=None, options=None):
                # If fixed_class_idx is set (for inference), use it; otherwise cycle
                if hasattr(self, 'fixed_class_idx') and self.fixed_class_idx is not None:
                    self.current_class_idx = self.fixed_class_idx
                else:
                    # Cycle to next class (for training)
                    self.current_class_idx = (self.current_class_idx + 1) % self.n_classes
                
                current_class = self.target_classes[self.current_class_idx]
                
                # Update base environment's target class
                self.base_env.target_class = current_class
                
                # Reset base environment
                obs, info = self.base_env.reset(seed=seed, options=options)
                
                # Add one-hot class encoding to observation
                class_one_hot = np.zeros(self.n_classes, dtype=np.float32)
                class_one_hot[self.current_class_idx] = 1.0
                extended_obs = np.concatenate([obs, class_one_hot])
                
                return extended_obs, info
            
            def step(self, action):
                obs, reward, terminated, truncated, info = self.base_env.step(action)
                
                # Add one-hot class encoding to observation
                class_one_hot = np.zeros(self.n_classes, dtype=np.float32)
                class_one_hot[self.current_class_idx] = 1.0
                extended_obs = np.concatenate([obs, class_one_hot])
                
                return extended_obs, reward, terminated, truncated, info
            
            def seed(self, seed=None):
                if seed is not None:
                    np.random.seed(seed)
                    if hasattr(self.base_env, 'rng'):
                        self.base_env.rng = np.random.default_rng(seed)
                return [seed]
        
        return MultiClassAnchorEnv(env_data, env_config, target_classes, device, classifier, seed=self.seed)
    
    def _create_model(self, device: str = "cpu"):
        """Create the SB3 model."""
        from trainers.device_utils import get_device_str
        device_str = get_device_str(device) if device != "auto" else "auto"
        
        # Create action noise for DDPG
        action_noise = None
        if self.algorithm == "ddpg":
            n_actions = self.env.action_space.shape[0]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=self.algorithm_config["action_noise_sigma"] * np.ones(n_actions)
            )
        
        # Prepare model arguments
        model_kwargs = {
            "policy": "MlpPolicy",
            "env": self.env,
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
        
        # TensorBoard logging
        if self.experiment_config.get("tensorboard_log", True):
            tensorboard_log = os.path.join(self.experiment_folder, "tensorboard")
            model_kwargs["tensorboard_log"] = tensorboard_log
        
        # Create model
        self.model = self.algorithm_class(**model_kwargs)
        
        logger.info(f"Model created: {self.algorithm.upper()}")
        logger.info(f"  Observation space: {self.env.observation_space}")
        logger.info(f"  Action space: {self.env.action_space}")
    
    def train(self):
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not created. Call setup_experiment() first.")
        
        logger.info("\n" + "="*80)
        logger.info("STARTING SINGLE-AGENT SB3 ANCHOR TRAINING")
        logger.info("="*80)
        
        # Create callbacks
        callbacks = []
        
        # Checkpoint callback
        if self.experiment_config.get("checkpoint_freq", 0) > 0:
            checkpoint_dir = os.path.join(self.experiment_folder, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_callback = CheckpointCallback(
                save_freq=self.experiment_config["checkpoint_freq"],
                save_path=checkpoint_dir,
                name_prefix="checkpoint"
            )
            callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        if self.experiment_config.get("eval_freq", 0) > 0:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=os.path.join(self.experiment_folder, "best_model"),
                log_path=os.path.join(self.experiment_folder, "evaluations"),
                eval_freq=self.experiment_config["eval_freq"],
                n_eval_episodes=self.experiment_config.get("n_eval_episodes", 4),
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # Train
        self.model.learn(
            total_timesteps=self.experiment_config["total_timesteps"],
            callback=callbacks if callbacks else None,
            log_interval=self.experiment_config.get("log_interval", 10),
            progress_bar=True
        )
        
        # Save final model
        final_model_path = os.path.join(self.experiment_folder, "final_model")
        self.model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Results saved to: {self.experiment_folder}")
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"\nEvaluating model on {n_episodes} episodes...")
        
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=n_episodes,
            deterministic=True
        )
        
        logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_episodes": n_episodes
        }
    
    def get_checkpoint_path(self) -> str:
        """Get the path to the experiment folder."""
        return self.experiment_folder
    
    def reload_experiment(self, experiment_dir: str):
        """Reload an experiment from a checkpoint directory."""
        self.experiment_folder = experiment_dir
        
        # Load model
        model_path = os.path.join(experiment_dir, "final_model")
        if not os.path.exists(model_path + ".zip"):
            # Try best model
            model_path = os.path.join(experiment_dir, "best_model", "best_model")
        
        if not os.path.exists(model_path + ".zip"):
            raise ValueError(f"Model not found in {experiment_dir}")
        
        # Recreate environment (needed for loading)
        # This is a simplified version - you may need to recreate the full setup
        logger.info(f"Loading model from: {model_path}")
        self.model = self.algorithm_class.load(model_path, env=self.env)

