"""
TD3 Trainer for Dynamic Anchors using Stable-Baselines3.

This module provides a TD3 trainer wrapper around Stable-Baselines3's TD3
for training dynamic anchor policies with continuous actions.

TD3 (Twin Delayed DDPG) improves upon DDPG with:
- Twin critics (two Q-networks) for better value estimation
- Delayed policy updates for stability
- Target policy smoothing to reduce overestimation bias
"""

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
from typing import Optional
import numpy as np
import warnings

# Suppress the Monitor wrapper warning for DummyVecEnv evaluation
warnings.filterwarnings("ignore", message=".*not wrapped with a.*Monitor.*wrapper.*")


class DynamicAnchorTD3Trainer:
    """
    TD3 Trainer for Dynamic Anchor RL using Stable-Baselines3.
    
    This class wraps SB3's TD3 algorithm to provide an easy interface for
    training dynamic anchor policies with continuous actions.
    """
    
    def __init__(
        self,
        env,
        policy_type: str = "MlpPolicy",
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        learning_starts: int = 0,  # Start learning immediately for manual control
        batch_size: int = 64,
        tau: float = 0.005,  # Soft update coefficient for target network
        gamma: float = 0.99,
        train_freq: tuple = (1, "step"),  # Train every step (but we'll call train() manually)
        gradient_steps: int = 1,
        action_noise: Optional[NormalActionNoise] = None,
        policy_kwargs: Optional[dict] = None,
        policy_delay: int = 2,  # TD3-specific: delay policy updates
        target_policy_noise: float = 0.2,  # TD3-specific: target policy smoothing
        target_noise_clip: float = 0.5,  # TD3-specific: clip target noise
        verbose: int = 0,
        device: str = "auto",
        tensorboard_log: Optional[str] = None,
    ):
        """
        Initialize the TD3 trainer with Stable-Baselines3.
        
        Args:
            env: Gym-compatible environment (not vectorized for TD3)
            policy_type: Policy architecture ("MlpPolicy", "CnnPolicy", etc.)
            learning_rate: Learning rate for optimizer
            buffer_size: Replay buffer size
            learning_starts: Number of steps before learning starts (0 for manual control)
            batch_size: Batch size for training
            tau: Soft update coefficient for target network
            gamma: Discount factor
            train_freq: Training frequency (we'll call train() manually)
            gradient_steps: Number of gradient steps per training call
            action_noise: Action noise for exploration (None = create default)
            policy_kwargs: Additional policy network arguments
            policy_delay: TD3-specific: delay policy updates (update every N critic updates)
            target_policy_noise: TD3-specific: noise added to target policy
            target_noise_clip: TD3-specific: clip range for target policy noise
            verbose: Verbosity level (0, 1, 2)
            device: Device for training ("auto", "cpu", "cuda")
            tensorboard_log: Directory for TensorBoard logs (None = no logging)
        """
        self.env = env
        
        # Standardize device handling: convert to string for SB3 (SB3 handles "auto" correctly)
        from trainers.device_utils import get_device_str
        device_str = get_device_str(device) if device != "auto" else "auto"
        
        # Create default action noise if not provided
        if action_noise is None:
            n_actions = env.action_space.shape[0]
            # Stronger exploration for continuous box expansion
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.3 * np.ones(n_actions)
            )
        
        # Default policy kwargs
        if policy_kwargs is None:
            policy_kwargs = dict(net_arch=[256, 256])
        
        # Initialize TD3 model from Stable-Baselines3
        self.model = TD3(
            policy=policy_type,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            verbose=verbose,
            device=device_str,
            tensorboard_log=tensorboard_log,
        )
        
        # Initialize logger if not already initialized (needed for train() method)
        if not hasattr(self.model, '_logger') or self.model._logger is None:
            logger = configure(folder=None, format_strings=[])
            self.model.set_logger(logger)
        
        # Verify TD3 model components
        assert hasattr(self.model, 'actor'), "TD3 model missing actor network"
        assert hasattr(self.model, 'critic'), "TD3 model missing critic network"
        assert hasattr(self.model, 'replay_buffer'), "TD3 model missing replay buffer"
        assert hasattr(self.model, 'train'), "TD3 model missing train method"
        
        print(f"TD3 Trainer initialized with policy: {policy_type}")
        print(f"  - Actor network: {type(self.model.actor).__name__}")
        print(f"  - Critic network: {type(self.model.critic).__name__}")
        print(f"  - Replay buffer size: {buffer_size}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Policy delay: {policy_delay} (TD3-specific)")
        noise_sigma = action_noise._sigma[0] if hasattr(action_noise, '_sigma') else 0.3
        print(f"  - Action noise: sigma={noise_sigma:.3f}")
    
    def predict(self, obs, deterministic: bool = False):
        """
        Predict action for given observation.
        
        Args:
            obs: Observation
            deterministic: If True, use deterministic policy (no noise)
            
        Returns:
            Tuple of (action, state) where state is internal state (None for TD3)
        """
        return self.model.predict(obs, deterministic=deterministic)
    
    def add_to_replay_buffer(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        info: Optional[dict] = None
    ):
        """
        Manually add a transition to the replay buffer.
        
        This is needed when stepping the environment manually (outside of learn()).
        
        Args:
            obs: Current observation
            next_obs: Next observation
            action: Action taken
            reward: Reward received
            done: Whether episode is done
            info: Additional info dict
        """
        if hasattr(self.model, 'replay_buffer') and self.model.replay_buffer is not None:
            # Validate input shapes and values
            # CRITICAL: Ensure we have proper numpy arrays and make explicit copies
            # Flatten to 1D first to ensure consistent shape handling
            obs_flat = np.asarray(obs, dtype=np.float32).flatten()
            next_obs_flat = np.asarray(next_obs, dtype=np.float32).flatten()
            
            # Validate lengths match before proceeding
            if len(obs_flat) != len(next_obs_flat):
                print(f"ERROR: Observation length mismatch in add_to_replay_buffer!")
                print(f"  obs type: {type(obs)}, shape: {np.asarray(obs).shape}, flattened len: {len(obs_flat)}")
                print(f"  next_obs type: {type(next_obs)}, shape: {np.asarray(next_obs).shape}, flattened len: {len(next_obs_flat)}")
                print(f"  obs content (first 20): {obs_flat[:20]}")
                print(f"  next_obs content (first 20): {next_obs_flat[:20]}")
                raise ValueError(
                    f"Observation length mismatch: obs has {len(obs_flat)} features, "
                    f"next_obs has {len(next_obs_flat)} features"
                )
            
            # Create proper copies as 2D arrays for replay buffer
            obs_array = obs_flat.copy().reshape(1, -1)
            next_obs_array = next_obs_flat.copy().reshape(1, -1)
            action_array = np.asarray(action, dtype=np.float32)
            if action_array.ndim == 1:
                action_array = action_array.reshape(1, -1)
            
            # Final validation that shapes match
            if obs_array.shape[1] != next_obs_array.shape[1]:
                print(f"ERROR: Observation shape mismatch after reshaping in add_to_replay_buffer!")
                print(f"  obs_array.shape: {obs_array.shape}, next_obs_array.shape: {next_obs_array.shape}")
                raise ValueError(
                    f"Observation shape mismatch: obs has {obs_array.shape[1]} features, "
                    f"next_obs has {next_obs_array.shape[1]} features"
                )
            
            # Validate action dimension matches expected action space
            expected_action_dim = self.env.action_space.shape[0]
            if action_array.shape[1] != expected_action_dim:
                raise ValueError(
                    f"Action shape mismatch: action has {action_array.shape[1]} dimensions, "
                    f"expected {expected_action_dim} (from action_space)"
                )
            
            # Validate values are finite
            if not np.all(np.isfinite(obs_array)):
                raise ValueError("obs contains NaN or Inf values")
            if not np.all(np.isfinite(next_obs_array)):
                raise ValueError("next_obs contains NaN or Inf values")
            if not np.all(np.isfinite(action_array)):
                raise ValueError("action contains NaN or Inf values")
            if not np.isfinite(reward):
                raise ValueError(f"reward is not finite: {reward}")
            
            # Add to replay buffer
            self.model.replay_buffer.add(
                obs=obs_array,
                next_obs=next_obs_array,
                action=action_array,
                reward=np.array([reward], dtype=np.float32),
                done=np.array([done], dtype=np.bool_),
                infos=[info] if isinstance(info, dict) else [{}]
            )
            
            # Update timestep counter
            self.model.num_timesteps += 1
    
    def train_step(self, gradient_steps: int = 1) -> bool:
        """
        Perform one training step if enough samples are in replay buffer.
        
        Args:
            gradient_steps: Number of gradient steps to perform
            
        Returns:
            True if training was performed, False otherwise
        """
        if (hasattr(self.model, 'replay_buffer') and 
            self.model.replay_buffer is not None):
            buffer_size = self.model.replay_buffer.size()
            if buffer_size >= self.model.batch_size:
                # Call TD3's train method - this updates both actor and critic networks
                # TD3 uses twin critics and delayed policy updates
                self.model.train(gradient_steps=gradient_steps, batch_size=self.model.batch_size)
                return True
        return False
    
    def get_buffer_size(self) -> int:
        """Get current replay buffer size."""
        if hasattr(self.model, 'replay_buffer') and self.model.replay_buffer is not None:
            return self.model.replay_buffer.size()
        return 0
    
    def save(self, path: str):
        """Save the trained model to disk."""
        self.model.save(path)
    
    def load(self, path: str, env=None):
        """Load a trained model from disk."""
        if env is None:
            env = self.env
        self.model = TD3.load(path, env=env)
    
    def evaluate(self, n_eval_episodes: int = 10, deterministic: bool = True):
        """Evaluate the policy on the environment."""
        print(f"[DEBUG TD3.evaluate] Starting evaluation with n_eval_episodes={n_eval_episodes}, deterministic={deterministic}")
        
        # Ensure classifier is in eval mode before evaluation
        # This is critical to prevent hanging during forward pass
        print(f"[DEBUG TD3.evaluate] Checking for classifier in environment...")
        if hasattr(self.env, 'anchor_env') and hasattr(self.env.anchor_env, 'classifier'):
            classifier = self.env.anchor_env.classifier
            print(f"[DEBUG TD3.evaluate] Found classifier: {type(classifier).__name__}")
            if hasattr(classifier, 'eval'):
                print(f"[DEBUG TD3.evaluate] Setting classifier to eval mode...")
                classifier.eval()
                # Also ensure underlying model is in eval mode (for UnifiedClassifier wrapper)
                if hasattr(classifier, 'model') and hasattr(classifier.model, 'eval'):
                    classifier.model.eval()
                print(f"[DEBUG TD3.evaluate] Classifier set to eval mode")
            else:
                print(f"[DEBUG TD3.evaluate] Classifier does not have eval() method")
        else:
            print(f"[DEBUG TD3.evaluate] No classifier found in environment (env.anchor_env.classifier)")
        
        print(f"[DEBUG TD3.evaluate] Calling evaluate_policy from stable_baselines3...")
        from stable_baselines3.common.evaluation import evaluate_policy
        try:
            result = evaluate_policy(
                self.model,
                self.env,
                n_eval_episodes=n_eval_episodes,
                deterministic=deterministic
            )
            print(f"[DEBUG TD3.evaluate] evaluate_policy completed successfully, result={result}")
            return result
        except Exception as e:
            print(f"[DEBUG TD3.evaluate] Exception in evaluate_policy: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise


def create_td3_trainer(
    env,
    policy_type: str = "MlpPolicy",
    learning_rate: float = 1e-4,
    action_noise_sigma: float = 0.3,
    policy_delay: int = 2,
    tensorboard_log: Optional[str] = None,
    **kwargs
):
    """
    Convenience function to create a TD3 trainer.
    
    Args:
        env: Gym-compatible environment
        policy_type: Policy architecture
        learning_rate: Learning rate
        action_noise_sigma: Action noise standard deviation
        policy_delay: TD3-specific: delay policy updates
        **kwargs: Additional trainer parameters
        
    Returns:
        DynamicAnchorTD3Trainer instance
    """
    # Create action noise
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=action_noise_sigma * np.ones(n_actions)
    )
    
    return DynamicAnchorTD3Trainer(
        env=env,
        policy_type=policy_type,
        learning_rate=learning_rate,
        action_noise=action_noise,
        policy_delay=policy_delay,
        tensorboard_log=tensorboard_log,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    print("TD3 Trainer for Dynamic Anchors")
    print("=" * 50)
    
    try:
        import gymnasium as gym
        env = gym.make("Pendulum-v1")
    except ImportError:
        try:
            import gym
            env = gym.make("Pendulum-v0")
        except ImportError:
            print("Please install gymnasium or gym to run this example")
            exit(1)
    
    # Create trainer
    trainer = create_td3_trainer(env, learning_rate=1e-4, policy_delay=2)
    
    print(f"\nTD3 Trainer created successfully!")
    print(f"Replay buffer size: {trainer.get_buffer_size()}")
    
    # Test action prediction
    obs, _ = env.reset()
    action, _ = trainer.predict(obs, deterministic=False)
    print(f"Predicted action shape: {action.shape}")
    
    print("\nTD3 Trainer initialized successfully!")

