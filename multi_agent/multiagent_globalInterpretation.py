"""
Load and inspect trained TD3 policy from single agent training output.

This script loads the trained TD3 policy from the best_model directory
and provides utilities to access and inspect the actor and critic networks.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
from pathlib import Path
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

# Import project modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers.networks import UnifiedClassifier
from trainers.vecEnv import AnchorEnv
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_classifier_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    """
    Dynamically create a classifier model from state dict.
    Infers architecture from the state dict keys.
    
    The state dict has layers at indices 0, 2, 4, etc. (with ReLU at 1, 3, etc.)
    This function reconstructs the Sequential model to match exactly.
    
    Args:
        state_dict: State dictionary from saved model
    
    Returns:
        Initialized classifier model matching the architecture
    """
    # Extract layer information from state dict
    # Keys are like: 'net.0.weight', 'net.0.bias', 'net.2.weight', etc.
    # Linear layers are at even indices (0, 2, 4), ReLU at odd indices (1, 3)
    linear_layers = {}
    for key in state_dict.keys():
        if 'weight' in key and 'net.' in key:
            parts = key.split('.')
            if len(parts) >= 2:
                try:
                    layer_idx = int(parts[1])
                    weight_shape = state_dict[key].shape
                    linear_layers[layer_idx] = weight_shape
                except ValueError:
                    continue
    
    # Sort layer indices to get architecture
    sorted_indices = sorted(linear_layers.keys())
    if len(sorted_indices) == 0:
        raise ValueError("Could not infer architecture from state dict")
    
    # Build Sequential model dynamically
    # The pattern is: Linear(0) -> ReLU(1) -> Linear(2) -> ReLU(3) -> Linear(4)
    layers = []
    
    for i, layer_idx in enumerate(sorted_indices):
        weight_shape = linear_layers[layer_idx]
        out_dim, in_dim = weight_shape
        
        # Add linear layer
        layers.append(nn.Linear(in_dim, out_dim))
        
        # Add ReLU after each linear layer except the last one
        if i < len(sorted_indices) - 1:
            layers.append(nn.ReLU())
    
    # Create model
    class DynamicClassifier(nn.Module):
        def __init__(self, layers_list):
            super().__init__()
            self.net = nn.Sequential(*layers_list)
            # Infer num_classes from last layer
            self.num_classes = layers_list[-1].out_features
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
    
    return DynamicClassifier(layers)


def create_minimal_env_for_loading(
    n_features: int,
    n_actions: int,
    observation_space_shape: Tuple[int, ...] = None
) -> gym.Env:
    """
    Create a minimal environment for loading TD3 model.
    
    TD3.load() requires an environment, but we can create a minimal one
    just for loading purposes.
    """
    if observation_space_shape is None:
        # Default: state_dim = 2 * n_features + 2 (lower_bounds, upper_bounds, precision, coverage)
        observation_space_shape = (2 * n_features + 2,)
    
    class MinimalEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=observation_space_shape, dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32
            )
        
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, {}
        
        def step(self, action):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0.0
            terminated = False
            truncated = False
            info = {}
            return obs, reward, terminated, truncated, info
    
    return MinimalEnv()


def load_td3_policy(
    model_path: str,
    output_dir: str,
    target_class: int = 0,
    device: str = "cpu"
) -> Dict:
    """
    Load trained TD3 policy from best_model directory.
    
    Args:
        model_path: Path to the best_model zip file (e.g., "best_model_class_0.zip")
        output_dir: Base output directory (e.g., "output/breast_cancer_td3_joint")
        target_class: Target class for the policy (0 or 1 for binary classification)
        device: Device to load model on ("cpu", "cuda", "auto")
    
    Returns:
        Dictionary containing:
            - "td3_model": Loaded TD3 model
            - "actor": Actor network
            - "critic": Critic network
            - "classifier": Classifier model (if available)
            - "metadata": Training metadata
    """
    output_path = Path(output_dir)
    best_model_path = output_path / "best_model" / model_path
    
    if not best_model_path.exists():
        raise FileNotFoundError(f"Model file not found: {best_model_path}")
    
    print(f"Loading TD3 model from: {best_model_path}")
    
    # Try to load metadata to get environment specs
    metadata_path = output_path / "metrics_and_rules.json"
    n_features = None
    n_actions = None
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            # Try to infer n_features from metadata or results
            # For breast cancer: 30 features
            # State dim = 2 * n_features + 2
            # Action dim = 2 * n_features (for continuous actions: [feature_idx, direction])
            # Actually, for continuous actions in AnchorEnv, action is [delta_lower, delta_upper] per feature
            # So action_dim = 2 * n_features
            n_features = 30  # Breast cancer has 30 features
            n_actions = 2 * n_features  # Continuous actions: delta for lower and upper bounds
    else:
        # Default values for breast cancer dataset
        n_features = 30
        n_actions = 2 * n_features
    
    # Create minimal environment for loading
    # State: [lower_bounds (n_features), upper_bounds (n_features), precision, coverage]
    state_dim = 2 * n_features + 2
    env = create_minimal_env_for_loading(n_features, n_actions, (state_dim,))
    
    # Load TD3 model
    # Note: TD3.load() may require env, but we can pass None or a minimal env
    try:
        td3_model = TD3.load(str(best_model_path), env=env, device=device)
        print(f"✓ Successfully loaded TD3 model")
    except Exception as e:
        print(f"Warning: Could not load with env, trying without env...")
        try:
            td3_model = TD3.load(str(best_model_path), device=device)
            print(f"✓ Successfully loaded TD3 model (without env)")
        except Exception as e2:
            raise RuntimeError(f"Failed to load TD3 model: {e2}")
    
    # Extract actor and critic networks
    actor = td3_model.actor
    critic = td3_model.critic
    
    print(f"✓ Actor network: {type(actor).__name__}")
    print(f"✓ Critic network: {type(critic).__name__}")
    
    # Try to load classifier if available
    classifier = None
    classifier_path = output_path / "models" / "classifier_final.pth"
    if classifier_path.exists():
        try:
            # Load classifier (may contain architecture info or just state dict)
            loaded_data = torch.load(classifier_path, map_location=device)
            
            # Check if architecture info is saved
            if isinstance(loaded_data, dict) and "architecture" in loaded_data:
                # New format: architecture info is saved
                architecture_info = loaded_data["architecture"]
                state_dict = loaded_data["state_dict"]
                
                # Build model from saved architecture
                layers = []
                for layer_info in architecture_info["layers"]:
                    if layer_info["type"] == "Linear":
                        layers.append(nn.Linear(
                            layer_info["in_features"],
                            layer_info["out_features"]
                        ))
                    elif layer_info["type"] == "ReLU":
                        layers.append(nn.ReLU())
                    elif layer_info["type"] == "BatchNorm1d":
                        layers.append(nn.BatchNorm1d(layer_info["num_features"]))
                    elif layer_info["type"] == "Dropout":
                        layers.append(nn.Dropout(layer_info["p"]))
                
                # Create model
                class DynamicClassifier(nn.Module):
                    def __init__(self, layers_list):
                        super().__init__()
                        self.net = nn.Sequential(*layers_list)
                        self.num_classes = architecture_info["num_classes"]
                    
                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return self.net(x)
                
                simple_classifier = DynamicClassifier(layers)
                simple_classifier.load_state_dict(state_dict)
                
                # Print architecture from saved info
                linear_layers = [l for l in architecture_info["layers"] if l["type"] == "Linear"]
                arch_parts = [str(architecture_info["input_dim"])]
                arch_parts.extend([str(l["out_features"]) for l in linear_layers])
                arch_str = " -> ".join(arch_parts)
                print(f"✓ Loaded classifier from: {classifier_path} (with saved architecture)")
                print(f"  Architecture: {arch_str}")
            else:
                # Old format: only state dict, infer architecture
                state_dict = loaded_data if isinstance(loaded_data, dict) else loaded_data
                simple_classifier = create_classifier_from_state_dict(state_dict)
                simple_classifier.load_state_dict(state_dict)
                
                # Print architecture info
                arch_parts = []
                for layer in simple_classifier.net:
                    if isinstance(layer, nn.Linear):
                        if len(arch_parts) == 0:
                            arch_parts.append(f"{layer.in_features} -> {layer.out_features}")
                        else:
                            arch_parts.append(str(layer.out_features))
                arch_str = " -> ".join(arch_parts)
                print(f"✓ Loaded classifier from: {classifier_path} (inferred architecture)")
                print(f"  Architecture: {arch_str}")
            
            simple_classifier.eval()
            simple_classifier.to(device)
            
            # Wrap in UnifiedClassifier for consistency
            classifier = UnifiedClassifier(
                classifier_type="dnn",
                dnn_model=simple_classifier,
                device=device
            )
            classifier.eval()
        except Exception as e:
            print(f"Warning: Could not load classifier: {e}")
            import traceback
            traceback.print_exc()
    
    # Load metadata
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return {
        "td3_model": td3_model,
        "actor": actor,
        "critic": critic,
        "classifier": classifier,
        "metadata": metadata,
        "n_features": n_features,
        "n_actions": n_actions,
        "state_dim": state_dim
    }


def inspect_policy(policy_dict: Dict):
    """
    Inspect the loaded TD3 policy and print network architectures.
    
    Args:
        policy_dict: Dictionary returned by load_td3_policy()
    """
    print("\n" + "="*60)
    print("TD3 Policy Inspection")
    print("="*60)
    
    actor = policy_dict["actor"]
    critic = policy_dict["critic"]
    
    print(f"\nActor Network Architecture:")
    print(f"  Type: {type(actor).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in actor.parameters() if p.requires_grad):,}")
    
    # Print actor structure
    if hasattr(actor, "mlp_extractor"):
        print(f"  MLP Extractor: {actor.mlp_extractor}")
    if hasattr(actor, "action_net"):
        print(f"  Action Net: {actor.action_net}")
    if hasattr(actor, "latent_pi"):
        print(f"  Latent Pi: {actor.latent_pi}")
    
    print(f"\nCritic Network Architecture:")
    print(f"  Type: {type(critic).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in critic.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in critic.parameters() if p.requires_grad):,}")
    
    # TD3 has twin critics
    if hasattr(critic, "q_networks"):
        print(f"  Number of Q-networks: {len(critic.q_networks)}")
        for i, q_net in enumerate(critic.q_networks):
            print(f"    Q-network {i}: {q_net}")
    elif hasattr(critic, "q_net"):
        print(f"  Q-network: {critic.q_net}")
    
    print(f"\nModel Info:")
    print(f"  State dimension: {policy_dict['state_dim']}")
    print(f"  Action dimension: {policy_dict['n_actions']}")
    print(f"  Number of features: {policy_dict['n_features']}")
    
    if policy_dict.get("classifier") is not None:
        print(f"\nClassifier:")
        print(f"  Type: {type(policy_dict['classifier']).__name__}")
        print(f"  Parameters: {sum(p.numel() for p in policy_dict['classifier'].parameters()):,}")


def predict_action(policy_dict: Dict, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
    """
    Predict action using the loaded TD3 policy.
    
    Args:
        policy_dict: Dictionary returned by load_td3_policy()
        observation: Observation array of shape (state_dim,) or (batch_size, state_dim)
        deterministic: If True, use deterministic policy (no noise)
    
    Returns:
        Action array
    """
    td3_model = policy_dict["td3_model"]
    
    # Ensure observation is the right shape
    obs = np.asarray(observation, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    
    action, _ = td3_model.predict(obs, deterministic=deterministic)
    
    return action[0] if action.shape[0] == 1 else action


def main():
    """Main function to load and inspect TD3 policy."""
    # Configuration
    output_dir = "output/breast_cancer_td3_joint"
    model_path = "best_model_class_0.zip"  # Change to "best_model_class_1.zip" for class 1
    target_class = 0
    device = "cpu"
    
    print("="*60)
    print("Loading TD3 Policy from Single Agent Training")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Model path: {model_path}")
    print(f"Target class: {target_class}")
    
    try:
        # Load the policy
        policy_dict = load_td3_policy(
            model_path=model_path,
            output_dir=output_dir,
            target_class=target_class,
            device=device
        )
        
        # Inspect the policy
        inspect_policy(policy_dict)
        
        # Example: Predict action for a sample observation
        print("\n" + "="*60)
        print("Example: Predicting Action")
        print("="*60)
        
        # Create a sample observation (state_dim = 2 * n_features + 2)
        state_dim = policy_dict["state_dim"]
        sample_obs = np.random.rand(state_dim).astype(np.float32)
        print(f"Sample observation shape: {sample_obs.shape}")
        
        action = predict_action(policy_dict, sample_obs, deterministic=True)
        print(f"Predicted action shape: {action.shape}")
        print(f"Predicted action (first 5 values): {action[:5]}")
        
        print("\n✓ Successfully loaded and tested TD3 policy!")
        
        return policy_dict
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    policy_dict = main()
