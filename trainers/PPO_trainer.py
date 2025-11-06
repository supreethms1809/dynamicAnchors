"""
PPO Trainer for Dynamic Anchors using Stable-Baselines3.

This module provides a PPO trainer wrapper around Stable-Baselines3's PPO
for training dynamic anchor policies.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from typing import Optional, List
import os
import warnings

# Suppress the Monitor wrapper warning for DummyVecEnv evaluation
# DummyVecEnv tracks episodes internally, so the warning is safe to ignore
warnings.filterwarnings("ignore", message=".*not wrapped with a.*Monitor.*wrapper.*")


class DynamicAnchorPPOTrainer:
    """
    PPO Trainer for Dynamic Anchor RL using Stable-Baselines3.
    
    This class wraps SB3's PPO algorithm to provide an easy interface for
    training dynamic anchor policies.
    """
    
    def __init__(
        self,
        vec_env,
        policy_type: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.02,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 1,
        tensorboard_log: Optional[str] = "./ppo_tensorboard/",
        device: str = "auto",
    ):
        """
        Initialize the PPO trainer with Stable-Baselines3.
        
        Args:
            vec_env: Vectorized environment from make_vec_env()
            policy_type: Policy architecture ("MlpPolicy", "CnnPolicy", etc.)
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps per environment before update
            batch_size: Minibatch size for updates
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping parameter
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
            verbose: Verbosity level (0, 1, 2)
            tensorboard_log: Directory for tensorboard logs
            device: Device for training ("auto", "cpu", "cuda")
        """
        self.vec_env = vec_env
        
        # Initialize PPO model from Stable-Baselines3
        self.model = PPO(
            policy=policy_type,
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            device=device,
        )
        
        print(f"PPO Trainer initialized with policy: {policy_type}")
        print(f"Learning rate: {learning_rate}, n_steps: {n_steps}, n_envs: {vec_env.num_envs}")
    
    def learn(
        self, 
        total_timesteps: int, 
        callback=None, 
        progress_bar=True,
        log_interval: int = 10,
        save_checkpoints: bool = True,
        checkpoint_freq: int = 10000,
        checkpoint_dir: Optional[str] = None,
        eval_freq: int = 0,
        n_eval_episodes: int = 10,
        eval_log_path: Optional[str] = None,
    ):
        """
        Train the PPO model with optional callbacks for monitoring and checkpoints.
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback: Optional callback(s) for monitoring (can be a list)
            progress_bar: Show progress bar during training
            log_interval: Log training stats every N updates
            save_checkpoints: Whether to save model checkpoints during training
            checkpoint_freq: Frequency of checkpoint saves in timesteps
            checkpoint_dir: Directory to save checkpoints (None = auto-create)
            eval_freq: Frequency of evaluation in timesteps
            n_eval_episodes: Number of episodes for evaluation
            eval_log_path: Path to save evaluation logs
        """
        # Prepare callbacks
        callbacks_list = []
        
        # Add checkpoint callback if requested
        if save_checkpoints:
            if checkpoint_dir is None:
                checkpoint_dir = "./ppo_checkpoints/"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoint_dir,
                name_prefix="ppo_dynamic_anchors",
                verbose=1
            )
            callbacks_list.append(checkpoint_callback)
        
        # Add evaluation callback if requested
        # NOTE: Currently using training env for evaluation to avoid deadlocks
        # A separate eval env would require passing env factory from caller
        if eval_freq > 0:
            if eval_log_path is None:
                eval_log_path = "./ppo_eval_logs/"
            os.makedirs(eval_log_path, exist_ok=True)
            
            eval_callback = EvalCallback(
                self.vec_env,  # Use training environment (will show Monitor warning but works)
                best_model_save_path=eval_log_path,
                log_path=eval_log_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                verbose=1
            )
            callbacks_list.append(eval_callback)
        
        # Add user-provided callbacks
        if callback is not None:
            if isinstance(callback, list):
                callbacks_list.extend(callback)
            else:
                callbacks_list.append(callback)
        
        # Convert callback list to None if empty (better for SB3 API)
        final_callbacks = callbacks_list if callbacks_list else None
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=final_callbacks,
            progress_bar=progress_bar,
            log_interval=log_interval
        )
    
    def save(self, path: str):
        """Save the trained model to disk."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load a trained model from disk."""
        self.model = PPO.load(path, env=self.vec_env)
    
    def predict(self, obs, deterministic: bool = False):
        """Predict action for given observation."""
        return self.model.predict(obs, deterministic=deterministic)
    
    def evaluate(self, n_eval_episodes: int = 10, deterministic: bool = True):
        """Evaluate the policy on the environment."""
        from stable_baselines3.common.evaluation import evaluate_policy
        return evaluate_policy(
            self.model,
            self.vec_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic
        )


def create_ppo_trainer(
    vec_env,
    policy_type: str = "MlpPolicy",
    learning_rate: float = 3e-4,
    **kwargs
):
    """
    Convenience function to create a PPO trainer.
    
    Args:
        vec_env: Vectorized environment
        policy_type: Policy architecture
        learning_rate: Learning rate
        **kwargs: Additional PPO parameters
    
    Returns:
        DynamicAnchorPPOTrainer instance
    """
    return DynamicAnchorPPOTrainer(
        vec_env=vec_env,
        policy_type=policy_type,
        learning_rate=learning_rate,
        **kwargs
    )


def train_ppo_model(
    vec_env,
    total_timesteps: int,
    policy_type: str = "MlpPolicy",
    learning_rate: float = 3e-4,
    output_dir: str = "./ppo_output/",
    save_checkpoints: bool = True,
    checkpoint_freq: int = 10000,
    eval_freq: int = 5000,
    verbose: int = 1,
    tensorboard_log_dir: Optional[str] = None,
    **trainer_kwargs
):
    """
    Complete training pipeline for PPO model.
    
    This function creates a PPO trainer, trains it, and saves the final model.
    
    Args:
        vec_env: Vectorized environment from make_vec_env()
        total_timesteps: Total number of timesteps to train
        policy_type: Policy architecture
        learning_rate: Learning rate for optimizer
        output_dir: Directory to save outputs (model, checkpoints, logs)
        save_checkpoints: Whether to save checkpoints during training
        checkpoint_freq: Frequency of checkpoint saves in timesteps
        eval_freq: Frequency of evaluation in timesteps
        verbose: Verbosity level (0, 1, 2)
        tensorboard_log_dir: Optional unified TensorBoard log directory (for joint training)
        **trainer_kwargs: Additional trainer parameters
    
    Returns:
        Trained DynamicAnchorPPOTrainer instance
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup tensorboard logging
    # Use provided tensorboard_log_dir if specified (for unified logging across rounds),
    # otherwise use per-output directory
    if tensorboard_log_dir is not None:
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        tensorboard_log = tensorboard_log_dir if verbose > 0 else None
    else:
        tensorboard_log = os.path.join(output_dir, "tensorboard") if verbose > 0 else None
    
    # Create trainer
    trainer = DynamicAnchorPPOTrainer(
        vec_env=vec_env,
        policy_type=policy_type,
        learning_rate=learning_rate,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        **trainer_kwargs
    )
    
    # Setup checkpoint and eval directories
    checkpoint_dir = os.path.join(output_dir, "checkpoints") if save_checkpoints else None
    eval_log_path = os.path.join(output_dir, "eval_logs")
    
    # Train the model
    print(f"\nStarting training for {total_timesteps} timesteps...")
    trainer.learn(
        total_timesteps=total_timesteps,
        save_checkpoints=save_checkpoints,
        checkpoint_freq=checkpoint_freq,
        checkpoint_dir=checkpoint_dir,
        eval_freq=eval_freq,
        eval_log_path=eval_log_path,
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, "ppo_model_final")
    trainer.save(final_model_path)
    print(f"\nTraining complete! Model saved to {final_model_path}")
    
    return trainer

