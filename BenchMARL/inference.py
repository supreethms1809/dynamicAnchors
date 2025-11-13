"""
Simplified inference script for extracting anchor rules using saved policy models.

This script:
1. Loads saved individual policy models from the experiment directory
2. Creates environment and runs rollouts using the loaded policies
3. Extracts rules from the rollouts
4. Saves evaluation data and rules
"""

from tabular_datasets import TabularDatasetLoader
from environment import AnchorEnv
from benchmarl.models.mlp import MlpConfig
from benchmarl_wrappers import AnchorTask
import argparse
import os
import numpy as np
import torch
from typing import Dict, Any, List, Optional
import json
from tensordict import TensorDict

import logging
from logging import INFO, WARNING, ERROR, CRITICAL
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_policy_model(
    policy_path: str,
    metadata_path: str,
    mlp_config_path: str,
    device: str = "cpu"
) -> torch.nn.Module:
    """
    Load a saved policy model.
    
    Args:
        policy_path: Path to saved policy state_dict (.pth file)
        metadata_path: Path to policy metadata JSON file
        mlp_config_path: Path to MLP config YAML
        device: Device to load model on
    
    Returns:
        Loaded policy model (MLP network)
    """
    # Load metadata to understand model structure
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load MLP config
    mlp_config = MlpConfig.get_from_yaml(mlp_config_path)
    
    # Load state dict to infer input/output dimensions
    state_dict = torch.load(policy_path, map_location=device)
    
    # Check if state_dict has nested structure (e.g., "0.mlp.params.0.weight")
    # If so, extract the MLP parameters
    nested_keys = [k for k in state_dict.keys() if '.mlp.params.' in k or '.params.' in k]
    
    if nested_keys:
        # Extract MLP parameters from nested structure
        # Pattern: "0.mlp.params.0.weight" -> "0.weight"
        mlp_state_dict = {}
        for key, value in state_dict.items():
            # Skip TensorDict metadata keys
            if key.startswith('__'):
                continue
            
            # Try to find MLP params in the key
            if '.mlp.params.' in key:
                # Extract the layer part after .mlp.params.
                # "0.mlp.params.0.weight" -> "0.weight"
                parts = key.split('.mlp.params.')
                if len(parts) == 2:
                    new_key = parts[1]  # "0.weight"
                    # Skip if it's still a metadata key
                    if not new_key.startswith('__'):
                        mlp_state_dict[new_key] = value
            elif '.params.' in key and 'mlp' not in key:
                # Alternative pattern: "0.params.0.weight" -> "0.weight"
                parts = key.split('.params.')
                if len(parts) == 2:
                    new_key = parts[1]
                    # Skip if it's still a metadata key
                    if not new_key.startswith('__'):
                        mlp_state_dict[new_key] = value
        
        if not mlp_state_dict:
            raise ValueError(f"Could not extract MLP parameters from nested structure in {policy_path}. Found nested keys: {nested_keys[:5]}")
        
        state_dict = mlp_state_dict
        logger.info(f"  Extracted MLP parameters from nested structure ({len(state_dict)} parameters)")
    
    # Infer input dimension from first layer weight shape
    first_layer_key = None
    for key in sorted(state_dict.keys()):
        if 'weight' in key and first_layer_key is None:
            first_layer_key = key
            break
    
    if first_layer_key is None:
        raise ValueError(f"Could not find weight parameters in {policy_path}. Available keys: {list(state_dict.keys())[:10]}")
    
    input_dim = state_dict[first_layer_key].shape[1]
    output_dim = None
    
    # Find output dimension from last layer
    last_layer_key = None
    for key in sorted(state_dict.keys(), reverse=True):
        if 'weight' in key:
            last_layer_key = key
            output_dim = state_dict[last_layer_key].shape[0]
            break
    
    if output_dim is None:
        raise ValueError(f"Could not infer output dimension from {policy_path}")
    
    logger.info(f"  Inferred model dimensions: input={input_dim}, output={output_dim}")
    
    # Create MLP model using BenchMARL's MLP config
    # We need to create a simple MLP that matches the saved architecture
    from benchmarl.models.common import ModelConfig
    from torchrl.modules import MLP
    
    # Build MLP with the inferred dimensions
    mlp = MLP(
        in_features=input_dim,
        out_features=output_dim,
        num_cells=mlp_config.num_cells,
        activation_class=mlp_config.activation_class,
        activation_kwargs=mlp_config.activation_kwargs or {},
        norm_class=mlp_config.norm_class,
        norm_kwargs=mlp_config.norm_kwargs or {},
        layer_class=mlp_config.layer_class,
    ).to(device)
    
    # Load state dict
    mlp.load_state_dict(state_dict)
    mlp.eval()
    
    return mlp


def run_rollout_with_policy(
    env: AnchorEnv,
    policy: torch.nn.Module,
    group: str,
    max_steps: int = 100,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run a single rollout episode using a loaded policy.
    
    Args:
        env: AnchorEnv environment instance
        policy: Loaded policy model
        group: Agent group name
        max_steps: Maximum steps per episode
        device: Device for tensors
    
    Returns:
        Dictionary with episode data (precision, coverage, observation, etc.)
    """
    # Reset environment
    td = env.reset()
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        # Get observation for this group
        if group not in td.keys():
            break
        
        group_obs = td[group]
        if "observation" not in group_obs.keys():
            break
        
        obs_tensor = group_obs["observation"]
        
        # Move to device and ensure correct shape
        if isinstance(obs_tensor, torch.Tensor):
            obs_tensor = obs_tensor.to(device)
        else:
            obs_tensor = torch.tensor(obs_tensor, device=device)
        
        # Add batch dimension if needed
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # Get action from policy (deterministic for inference)
        with torch.no_grad():
            action = policy(obs_tensor)
            # Remove batch dimension if added
            if action.shape[0] == 1:
                action = action.squeeze(0)
        
        # Set action in TensorDict
        td[group]["action"] = action
        
        # Step environment
        td = env.step(td)
        done = td.get("done", torch.zeros(1, dtype=torch.bool)).any().item()
        step_count += 1
    
    # Extract final metrics from info
    # Try to get from unwrapped environment first (most reliable)
    episode_data = {}
    unwrapped_env = None
    if hasattr(env, 'env') or hasattr(env, '_env'):
        unwrapped_env = getattr(env, 'env', None) or getattr(env, '_env', None)
    
    # Get actual agent name from environment
    actual_agent_name = group
    if unwrapped_env is not None:
        # Check if group matches an agent in the environment
        if hasattr(unwrapped_env, 'agents') and group in unwrapped_env.agents:
            actual_agent_name = group
        elif hasattr(unwrapped_env, 'possible_agents'):
            # Try to find matching agent
            for agent in unwrapped_env.possible_agents:
                if agent == group or group in agent or agent.startswith(group):
                    actual_agent_name = agent
                    break
            # If still not found, use first available agent
            if actual_agent_name == group and len(unwrapped_env.possible_agents) > 0:
                actual_agent_name = unwrapped_env.possible_agents[0]
    
    # Try to get metrics from unwrapped environment
    if unwrapped_env is not None and hasattr(unwrapped_env, '_current_metrics'):
        try:
            precision, coverage, _ = unwrapped_env._current_metrics(actual_agent_name)
            
            # Get final observation (bounds) from environment state
            if hasattr(unwrapped_env, 'lower') and hasattr(unwrapped_env, 'upper'):
                if isinstance(unwrapped_env.lower, dict):
                    if actual_agent_name in unwrapped_env.lower:
                        lower_bounds = unwrapped_env.lower[actual_agent_name]
                        upper_bounds = unwrapped_env.upper[actual_agent_name]
                    else:
                        # Try to find matching key
                        matching_key = None
                        for key in unwrapped_env.lower.keys():
                            if key == actual_agent_name or actual_agent_name in key or key.startswith(actual_agent_name):
                                matching_key = key
                                break
                        if matching_key:
                            lower_bounds = unwrapped_env.lower[matching_key]
                            upper_bounds = unwrapped_env.upper[matching_key]
                        else:
                            # Fall back to first available
                            if len(unwrapped_env.lower) > 0:
                                first_key = list(unwrapped_env.lower.keys())[0]
                                lower_bounds = unwrapped_env.lower[first_key]
                                upper_bounds = unwrapped_env.upper[first_key]
                            else:
                                lower_bounds = None
                                upper_bounds = None
                else:
                    lower_bounds = unwrapped_env.lower
                    upper_bounds = unwrapped_env.upper
                
                if lower_bounds is not None and upper_bounds is not None:
                    final_obs = np.concatenate([lower_bounds, upper_bounds, np.array([precision, coverage], dtype=np.float32)])
                    
                    episode_data = {
                        "precision": float(precision),
                        "coverage": float(coverage),
                        "total_reward": 0.0,
                        "final_observation": final_obs.tolist(),
                    }
        except Exception as e:
            # Fall through to TensorDict extraction
            pass
    
    # Fallback: Try to get from TensorDict structure
    if not episode_data and "next" in td.keys():
        next_td = td["next"]
        
        # Try group name first
        if group in next_td.keys():
            group_data = next_td[group]
            if "info" in group_data.keys():
                info = group_data["info"]
                if hasattr(info, 'shape') and info.shape[0] > 0:
                    final_info = info[-1]
                    
                    def safe_get(key, default=0.0):
                        try:
                            if hasattr(final_info, 'get'):
                                val = final_info.get(key, default)
                            elif hasattr(final_info, 'keys') and key in final_info.keys():
                                val = final_info[key]
                            else:
                                val = getattr(final_info, key, default)
                            
                            if isinstance(val, torch.Tensor):
                                return float(val.item() if val.numel() == 1 else val[-1].item())
                            return float(val)
                        except:
                            return default
                    
                    episode_data = {
                        "precision": safe_get("precision", 0.0),
                        "coverage": safe_get("coverage", 0.0),
                        "total_reward": safe_get("total_reward", 0.0),
                    }
                    
                    # Get final observation (anchor bounds)
                    if "observation" in group_data.keys():
                        obs = group_data["observation"]
                        if hasattr(obs, 'shape') and obs.shape[0] > 0:
                            final_obs = obs[-1]
                            if isinstance(final_obs, torch.Tensor):
                                episode_data["final_observation"] = final_obs.cpu().numpy().tolist()
                            else:
                                episode_data["final_observation"] = np.array(final_obs).tolist()
                        else:
                            # Extract from observation if available
                            final_obs_np = np.array(final_obs) if 'final_obs' in locals() else None
                            if final_obs_np is not None:
                                episode_data["final_observation"] = final_obs_np.tolist()
        
        # If still no data, try actual_agent_name
        if not episode_data and actual_agent_name != group and actual_agent_name in next_td.keys():
            group_data = next_td[actual_agent_name]
            if "info" in group_data.keys():
                info = group_data["info"]
                if hasattr(info, 'shape') and info.shape[0] > 0:
                    final_info = info[-1]
                    
                    def safe_get(key, default=0.0):
                        try:
                            if hasattr(final_info, 'get'):
                                val = final_info.get(key, default)
                            elif hasattr(final_info, 'keys') and key in final_info.keys():
                                val = final_info[key]
                            else:
                                val = getattr(final_info, key, default)
                            
                            if isinstance(val, torch.Tensor):
                                return float(val.item() if val.numel() == 1 else val[-1].item())
                            return float(val)
                        except:
                            return default
                    
                    episode_data = {
                        "precision": safe_get("precision", 0.0),
                        "coverage": safe_get("coverage", 0.0),
                        "total_reward": safe_get("total_reward", 0.0),
                    }
                    
                    # Get final observation
                    if "observation" in group_data.keys():
                        obs = group_data["observation"]
                        if hasattr(obs, 'shape') and obs.shape[0] > 0:
                            final_obs = obs[-1]
                            if isinstance(final_obs, torch.Tensor):
                                episode_data["final_observation"] = final_obs.cpu().numpy().tolist()
                            else:
                                episode_data["final_observation"] = np.array(final_obs).tolist()
    
    return episode_data


def extract_rules_from_policies(
    experiment_dir: str,
    dataset_name: str,
    mlp_config_path: str = "conf/mlp.yaml",
    max_features_in_rule: int = 5,
    steps_per_episode: int = 100,
    n_instances_per_class: int = 20,
    eval_on_test_data: bool = False,
    output_dir: Optional[str] = None,
    seed: int = 42,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Extract anchor rules using saved individual policy models.
    
    Args:
        experiment_dir: Path to BenchMARL experiment directory (contains individual_models/)
        dataset_name: Name of the dataset
        mlp_config_path: Path to MLP config YAML
        max_features_in_rule: Maximum features to include in rules
        steps_per_episode: Maximum steps per rollout
        n_instances_per_class: Number of instances to evaluate per class
        eval_on_test_data: Whether to evaluate on test data
        output_dir: Output directory for results (default: experiment_dir/inference/)
        seed: Random seed
        device: Device to use
    
    Returns:
        Dictionary containing extracted rules and evaluation data
    """
    logger.info("="*80)
    logger.info("ANCHOR RULE EXTRACTION USING SAVED POLICY MODELS")
    logger.info("="*80)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("="*80)
    
    # Find individual_models directory
    individual_models_dir = os.path.join(experiment_dir, "individual_models")
    if not os.path.exists(individual_models_dir):
        raise ValueError(
            f"Individual models directory not found: {individual_models_dir}\n"
            f"Make sure to run extract_and_save_individual_models() after training."
        )
    
    # Find all saved policy models
    policy_files = {}
    metadata_files = {}
    
    for filename in os.listdir(individual_models_dir):
        if filename.startswith("policy_") and filename.endswith(".pth"):
            group = filename.replace("policy_", "").replace(".pth", "")
            policy_files[group] = os.path.join(individual_models_dir, filename)
            
            # Look for corresponding metadata file
            metadata_filename = f"policy_{group}_metadata.json"
            metadata_path = os.path.join(individual_models_dir, metadata_filename)
            if os.path.exists(metadata_path):
                metadata_files[group] = metadata_path
    
    if not policy_files:
        raise ValueError(
            f"No policy models found in {individual_models_dir}\n"
            f"Expected files like: policy_<group>.pth"
        )
    
    logger.info(f"\nFound {len(policy_files)} policy model(s):")
    for group in sorted(policy_files.keys()):
        logger.info(f"  - {group}: {policy_files[group]}")
    
    # Load dataset
    dataset_loader = TabularDatasetLoader(
        dataset_name=dataset_name,
        test_size=0.2,
        random_state=seed
    )
    
    dataset_loader.load_dataset()
    dataset_loader.preprocess_data()
    
    # Load classifier from experiment directory
    classifier_path = os.path.join(experiment_dir, "classifier.pth")
    if os.path.exists(classifier_path):
        logger.info(f"\nLoading classifier from: {classifier_path}")
        classifier = dataset_loader.load_classifier(
            filepath=classifier_path,
            classifier_type="dnn",
            device=device
        )
        dataset_loader.classifier = classifier
        logger.info("Classifier loaded successfully")
    else:
        raise ValueError(
            f"Classifier not found at {classifier_path}\n"
            f"Please ensure the classifier was saved during training."
        )
    
    # Get environment data
    env_data = dataset_loader.get_anchor_env_data()
    target_classes = list(np.unique(dataset_loader.y_train))
    feature_names = env_data["feature_names"]
    n_features = len(feature_names)
    
    # Load all policy models and extract individual agent policies
    logger.info(f"\nLoading policy models...")
    policies = {}
    agent_policies = {}  # Individual agent policies extracted from combined policy
    
    for group in sorted(policy_files.keys()):
        metadata_path = metadata_files.get(group)
        if metadata_path is None:
            logger.warning(f"  ⚠ Warning: No metadata found for {group}, using defaults")
        
        # Load the combined policy
        combined_policy = load_policy_model(
            policy_path=policy_files[group],
            metadata_path=metadata_path or "",
            mlp_config_path=mlp_config_path,
            device=device
        )
        policies[group] = combined_policy
        logger.info(f"  ✓ Loaded combined policy for {group}")
        
        # Try to extract individual agent policies from the combined policy
        # Check if the state_dict has agent-specific keys (e.g., "0.mlp.params..." for agent_0)
        state_dict_path = policy_files[group]
        raw_state_dict = torch.load(state_dict_path, map_location=device)
        
        # Debug: Print some keys to understand structure
        logger.info(f"  Debug: Total keys in state_dict: {len(raw_state_dict.keys())}")
        logger.info(f"  Debug: Sample state_dict keys (first 20): {list(raw_state_dict.keys())[:20]}")
        
        # Check for agent-specific keys in the raw state_dict
        # In MADDPG, the actor module might have separate networks for each agent
        # Pattern could be:
        # - "0.mlp.params.0.weight" (agent 0, layer 0)
        # - "1.mlp.params.0.weight" (agent 1, layer 0)
        # - Or stored in a ModuleList: "policy_nets.0.weight", "policy_nets.1.weight"
        # - Or TensorDict structure with nested keys
        agent_indices = set()
        all_keys = list(raw_state_dict.keys())
        
        # First, try to find all numeric prefixes that could be agent indices
        # We'll be more aggressive in detection
        for key in all_keys:
            # Skip metadata
            if key.startswith('__'):
                continue
            
            parts = key.split('.')
            
            # Pattern 1: Key starts with digit (most common)
            # "0.mlp.params.0.weight" -> agent 0
            if parts[0].isdigit():
                try:
                    agent_idx = int(parts[0])
                    # Additional check: if the next part is not a layer number pattern,
                    # it's more likely to be an agent index
                    # "0.mlp.params.0.weight" -> agent 0 (mlp is not a number)
                    # "0.0.weight" -> could be agent 0 or layer 0
                    if len(parts) > 1:
                        # If second part is not a pure number, likely agent index
                        if not parts[1].isdigit() or parts[1] in ['mlp', 'params', 'net', 'module']:
                            agent_indices.add(agent_idx)
                        # Also add if it's clearly an agent pattern
                        elif 'mlp' in key.lower() or 'params' in key.lower():
                            agent_indices.add(agent_idx)
                    else:
                        agent_indices.add(agent_idx)
                except ValueError:
                    pass
            
            # Pattern 2: Agent index in ModuleList or similar structure
            # "policy_nets.0.weight", "agents.0.mlp.weight", etc.
            for i, part in enumerate(parts):
                if part.isdigit():
                    # Check context around this digit
                    if i > 0:
                        prev_part = parts[i-1].lower()
                        # If previous part suggests agent structure
                        if prev_part in ['policy_nets', 'agents', 'actor', 'actors', 'networks', 'nets']:
                            try:
                                agent_idx = int(part)
                                agent_indices.add(agent_idx)
                            except ValueError:
                                pass
                    # Also check if it's at position 0 and followed by non-numeric
                    elif i == 0 and len(parts) > 1 and not parts[1].isdigit():
                        try:
                            agent_idx = int(part)
                            agent_indices.add(agent_idx)
                        except ValueError:
                            pass
        
        # If no agent indices found in raw state_dict, the policy might be shared
        # In that case, we'll use the same policy for all agents
        if not agent_indices:
            logger.info(f"  Debug: No agent-specific indices found in raw state_dict")
            logger.info(f"  Debug: This might be a shared policy for all agents")
        else:
            logger.info(f"  Debug: Found agent indices: {sorted(agent_indices)}")
        
        # Use raw_state_dict for agent extraction
        state_dict = raw_state_dict
        
        if agent_indices:
            logger.info(f"  Found {len(agent_indices)} agents in combined policy: {sorted(agent_indices)}")
            
            # Extract individual agent policies
            from benchmarl.models.mlp import MlpConfig
            from torchrl.modules import MLP
            mlp_config = MlpConfig.get_from_yaml(mlp_config_path)
            
            for agent_idx in sorted(agent_indices):
                agent_name = f"agent_{agent_idx}"
                
                # Extract state_dict for this agent
                agent_state_dict = {}
                agent_keys_found = []
                
                for key, value in state_dict.items():
                    # Skip TensorDict metadata
                    if key.startswith('__'):
                        continue
                    
                    # Pattern 1: Key starts with agent index
                    # "0.mlp.params.0.weight" -> agent 0
                    if key.startswith(f"{agent_idx}."):
                        agent_keys_found.append(key)
                        # Remove agent prefix: "0.mlp.params.0.weight" -> "mlp.params.0.weight"
                        new_key = key[len(f"{agent_idx}."):]
                        
                        # Further extract MLP params if nested
                        if '.mlp.params.' in new_key:
                            parts = new_key.split('.mlp.params.')
                            if len(parts) == 2:
                                new_key = parts[1]  # "0.weight"
                        elif '.params.' in new_key:
                            parts = new_key.split('.params.')
                            if len(parts) == 2:
                                new_key = parts[1]
                        
                        if not new_key.startswith('__'):
                            agent_state_dict[new_key] = value
                    
                    # Pattern 2: Agent index in different position
                    # "policy_nets.0.weight" or similar
                    elif f".{agent_idx}." in key or key.startswith(f"{agent_idx}_"):
                        agent_keys_found.append(key)
                        # Try to extract the layer part
                        parts = key.split('.')
                        # Find the position of agent index
                        agent_pos = None
                        for i, part in enumerate(parts):
                            if part == str(agent_idx) or part.startswith(f"{agent_idx}_"):
                                agent_pos = i
                                break
                        
                        if agent_pos is not None:
                            # Reconstruct key without agent identifier
                            # This is heuristic - may need adjustment based on actual structure
                            if agent_pos == 0:
                                # "0.mlp.params.0.weight" -> already handled above
                                pass
                            else:
                                # For other patterns, try to extract relevant parts
                                # This might need customization based on actual structure
                                new_key = '.'.join(parts[agent_pos+1:])
                                if new_key and not new_key.startswith('__'):
                                    agent_state_dict[new_key] = value
                
                logger.info(f"  Debug: Found {len(agent_keys_found)} keys for {agent_name}")
                if agent_keys_found:
                    logger.info(f"  Debug: Sample keys for {agent_name}: {agent_keys_found[:5]}")
                
                if agent_state_dict:
                    # Infer dimensions from agent state_dict
                    first_layer_key = None
                    for key in sorted(agent_state_dict.keys()):
                        if 'weight' in key and first_layer_key is None:
                            first_layer_key = key
                            break
                    
                    if first_layer_key:
                        input_dim = agent_state_dict[first_layer_key].shape[1]
                        output_dim = None
                        for key in sorted(agent_state_dict.keys(), reverse=True):
                            if 'weight' in key:
                                output_dim = agent_state_dict[key].shape[0]
                                break
                        
                        if output_dim:
                            # Create MLP for this agent
                            agent_mlp = MLP(
                                in_features=input_dim,
                                out_features=output_dim,
                                num_cells=mlp_config.num_cells,
                                activation_class=mlp_config.activation_class,
                                activation_kwargs=mlp_config.activation_kwargs or {},
                                norm_class=mlp_config.norm_class,
                                norm_kwargs=mlp_config.norm_kwargs or {},
                                layer_class=mlp_config.layer_class,
                            ).to(device)
                            
                            agent_mlp.load_state_dict(agent_state_dict)
                            agent_mlp.eval()
                            agent_policies[agent_name] = agent_mlp
                            logger.info(f"  ✓ Extracted policy for {agent_name}")
                        else:
                            logger.warning(f"  ⚠ Could not infer output dimension for {agent_name}")
                    else:
                        logger.warning(f"  ⚠ Could not find weight parameters for {agent_name}")
                else:
                    logger.warning(f"  ⚠ Could not extract state_dict for {agent_name}")
        else:
            # No agent-specific keys found, policy might be shared
            # We'll handle this after we know the target_classes
            logger.info(f"  Debug: No agent-specific indices found, will use shared policy")
    
    # Check if we have policies for all expected agents
    # Expected agents are agent_0, agent_1, ..., agent_{n_classes-1}
    expected_agent_names = [f"agent_{i}" for i in range(len(target_classes))]
    missing_agents = [name for name in expected_agent_names if name not in agent_policies]
    
    if missing_agents:
        logger.warning(f"\n  ⚠ Warning: Missing policies for agents: {missing_agents}")
        logger.info(f"  Found policies for: {list(agent_policies.keys())}")
        logger.info(f"  Expected policies for: {expected_agent_names}")
        
        # If we have a combined policy but missing some agents, try to use the combined policy
        # for the missing agents (assuming shared policy)
        # Use the first available combined policy from any group
        combined_policy = None
        for g in policies.keys():
            if policies[g] is not None:
                combined_policy = policies[g]
                break
        
        if combined_policy is not None:
            logger.info(f"  Using combined policy for missing agents (assuming shared policy)")
            for missing_agent in missing_agents:
                agent_policies[missing_agent] = combined_policy
                logger.info(f"  ✓ Assigned combined policy to {missing_agent}")
        else:
            logger.warning(f"  ⚠ Error: No combined policy available to assign to missing agents")
    
    # Create environment config (needed before we can determine if policy is shared)
    from anchor_trainer import AnchorTrainer
    trainer = AnchorTrainer(
        dataset_loader=dataset_loader,
        algorithm="maddpg",  # Dummy, not used
        output_dir=output_dir or os.path.join(experiment_dir, "inference"),
        seed=seed
    )
    env_config = trainer._get_default_env_config()
    env_config.update({
        "X_min": env_data["X_min"],
        "X_range": env_data["X_range"],
    })
    
    if eval_on_test_data:
        if env_data.get("X_test_unit") is None:
            raise ValueError("Test data not available for evaluation")
        env_config.update({
            "eval_on_test_data": True,
            "X_test_unit": env_data["X_test_unit"],
            "X_test_std": env_data["X_test_std"],
            "y_test": env_data["y_test"],
        })
    
    # Create task config for environment creation
    anchor_config = {
        "X_unit": env_data["X_unit"],
        "X_std": env_data["X_std"],
        "y": env_data["y"],
        "feature_names": feature_names,
        "classifier": dataset_loader.get_classifier(),
        "device": device,
        "target_classes": target_classes,
        "env_config": env_config,
        "max_cycles": steps_per_episode,
    }
    
    task = AnchorTask.ANCHOR.get_task(config=anchor_config)
    
    # If no agent-specific policies were extracted, or if we only found some agents,
    # use shared policy for missing agents
    if not agent_policies or len(agent_policies) < len(target_classes):
        logger.info(f"  Creating policies for all target classes...")
        for group in sorted(policy_files.keys()):
            combined_policy = policies[group]
            for cls in target_classes:
                agent_name = f"agent_{cls}"
                # Only add if not already extracted
                if agent_name not in agent_policies:
                    agent_policies[agent_name] = combined_policy
                    logger.info(f"  ✓ Using shared policy for {agent_name}")
    
    # Use agent_policies if extracted, otherwise fall back to group policies
    if agent_policies:
        logger.info(f"\nUsing individual agent policies: {list(agent_policies.keys())}")
        # Update policies dict to use agent names
        policies = agent_policies
    else:
        logger.info(f"\nUsing group policies: {list(policies.keys())}")
    
    logger.info(f"\nExtracting rules for classes: {target_classes}")
    logger.info(f"  Instances per class: {n_instances_per_class}")
    logger.info(f"  Steps per episode: {steps_per_episode}")
    logger.info(f"  Max features in rule: {max_features_in_rule}")
    
    # Run rollouts and extract rules
    results = {
        "per_class_results": {},
        "metadata": {
            "dataset": dataset_name,
            "experiment_dir": experiment_dir,
            "target_classes": target_classes,
            "max_features_in_rule": max_features_in_rule,
            "eval_on_test_data": eval_on_test_data,
            "n_instances_per_class": n_instances_per_class,
            "steps_per_episode": steps_per_episode,
        },
    }
    
    # Map agent names to target classes
    # Agent names are like "agent_0", "agent_1", etc., where the number is the class
    agent_to_class = {}
    for agent_name in policies.keys():
        # Extract class from agent name (e.g., "agent_0" -> 0)
        if agent_name.startswith("agent_"):
            try:
                class_num = int(agent_name.split("_")[1])
                agent_to_class[agent_name] = class_num
            except (ValueError, IndexError):
                # If parsing fails, try to map by index
                pass
    
    # If mapping failed, use index-based mapping
    if not agent_to_class:
        sorted_agents = sorted(policies.keys())
        for idx, agent_name in enumerate(sorted_agents):
            if idx < len(target_classes):
                agent_to_class[agent_name] = target_classes[idx]
    
    # Run rollouts for each agent/class
    for agent_name, policy in policies.items():
        target_class = agent_to_class.get(agent_name, target_classes[0] if target_classes else 0)
        class_key = f"class_{target_class}"
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing class {target_class} (agent: {agent_name})")
        logger.info(f"{'='*80}")
        
        # Run multiple rollouts
        anchors_list = []
        rules_list = []
        precisions = []
        coverages = []
        
        for instance_idx in range(n_instances_per_class):
            # Create a single-agent environment for this specific class
            # This ensures we're running rollouts for the correct agent
            rollout_seed = seed + instance_idx if seed is not None else None
            
            # Create environment config for single agent (this class only)
            single_agent_config = anchor_config.copy()
            single_agent_config["target_classes"] = [target_class]  # Only this class
            
            # Create task for single agent
            single_agent_task = AnchorTask.ANCHOR.get_task(config=single_agent_config)
            
            env_fun = single_agent_task.get_env_fun(
                num_envs=1,
                continuous_actions=True,
                seed=rollout_seed,
                device=device
            )
            env = env_fun()
            
            # Get the actual agent name from the environment
            # In single-agent mode, there should be one agent
            unwrapped_env = None
            if hasattr(env, 'env') or hasattr(env, '_env'):
                unwrapped_env = getattr(env, 'env', None) or getattr(env, '_env', None)
            
            # Determine the agent name to use
            actual_agent_name = agent_name
            if unwrapped_env is not None:
                if hasattr(unwrapped_env, 'agents') and len(unwrapped_env.agents) > 0:
                    # Use the first (and only) agent in single-agent environment
                    actual_agent_name = unwrapped_env.agents[0]
                elif hasattr(unwrapped_env, 'possible_agents') and len(unwrapped_env.possible_agents) > 0:
                    # Check if our agent_name is in possible_agents
                    if agent_name in unwrapped_env.possible_agents:
                        actual_agent_name = agent_name
                    else:
                        # Use the first possible agent
                        actual_agent_name = unwrapped_env.possible_agents[0]
            
            # Run rollout
            # Note: The environment uses agent names, so we pass actual_agent_name as the group
            episode_data = run_rollout_with_policy(
                env=env,
                policy=policy,
                group=actual_agent_name,  # Use actual agent name from environment
                max_steps=steps_per_episode,
                device=device
            )
            
            if episode_data:
                precision = episode_data.get("precision", 0.0)
                coverage = episode_data.get("coverage", 0.0)
                
                precisions.append(float(precision))
                coverages.append(float(coverage))
                
                # Extract rule from final observation
                rule = "any values (no tightened features)"
                lower = None
                upper = None
                
                if "final_observation" in episode_data:
                    obs = np.array(episode_data["final_observation"], dtype=np.float32)
                    if len(obs) == 2 * n_features + 2:
                        lower = obs[:n_features].copy()
                        upper = obs[n_features:2*n_features].copy()
                        
                        # Create temporary environment for rule extraction
                        temp_env = AnchorEnv(
                            X_unit=env_data["X_unit"],
                            X_std=env_data["X_std"],
                            y=env_data["y"],
                            feature_names=feature_names,
                            classifier=dataset_loader.get_classifier(),
                            device="cpu",
                            target_classes=[target_class],
                            env_config=env_config
                        )
                        temp_env.lower[group] = lower
                        temp_env.upper[group] = upper
                        
                        rule = temp_env.extract_rule(
                            group,
                            max_features_in_rule=max_features_in_rule
                        )
                
                anchor_data = {
                    "instance_idx": instance_idx,
                    "precision": float(precision),
                    "coverage": float(coverage),
                    "total_reward": float(episode_data.get("total_reward", 0.0)),
                    "rule": rule,
                }
                
                if lower is not None and upper is not None:
                    anchor_data.update({
                        "lower_bounds": lower.tolist(),
                        "upper_bounds": upper.tolist(),
                        "box_widths": (upper - lower).tolist(),
                        "box_volume": float(np.prod(np.maximum(upper - lower, 1e-9))),
                    })
                
                anchors_list.append(anchor_data)
                rules_list.append(rule)
        
        unique_rules = list(set([r for r in rules_list if r and r != "any values (no tightened features)"]))
        
        results["per_class_results"][class_key] = {
            "class": int(target_class),
            "group": group,
            "precision": float(np.mean(precisions)) if precisions else 0.0,
            "coverage": float(np.mean(coverages)) if coverages else 0.0,
            "precision_std": float(np.std(precisions)) if len(precisions) > 1 else 0.0,
            "coverage_std": float(np.std(coverages)) if len(coverages) > 1 else 0.0,
            "n_episodes": len(anchors_list),
            "rules": rules_list,
            "unique_rules": unique_rules,
            "unique_rules_count": len(unique_rules),
            "anchors": anchors_list,
        }
        
        logger.info(f"  Processed {len(anchors_list)} episodes")
        logger.info(f"  Average precision: {results['per_class_results'][class_key]['precision']:.4f}")
        logger.info(f"  Average coverage: {results['per_class_results'][class_key]['coverage']:.4f}")
        logger.info(f"  Unique rules: {len(unique_rules)}")
    
    logger.info("\n" + "="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract anchor rules using saved policy models")
    
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to BenchMARL experiment directory (contains individual_models/)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=["breast_cancer", "wine", "iris", "synthetic", "moons", "circles", "covtype", "housing"],
        help="Dataset name (must match training)"
    )
    
    parser.add_argument(
        "--mlp_config",
        type=str,
        default="conf/mlp.yaml",
        help="Path to MLP config YAML"
    )
    
    parser.add_argument(
        "--max_features_in_rule",
        type=int,
        default=5,
        help="Maximum number of features to include in extracted rules"
    )
    
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=100,
        help="Maximum steps per rollout episode"
    )
    
    parser.add_argument(
        "--n_instances_per_class",
        type=int,
        default=20,
        help="Number of instances to evaluate per class"
    )
    
    parser.add_argument(
        "--eval_on_test_data",
        action="store_true",
        help="Evaluate on test data instead of training data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: experiment_dir/inference/)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference"
    )
    
    args = parser.parse_args()
    
    # Extract rules
    results = extract_rules_from_policies(
        experiment_dir=args.experiment_dir,
        dataset_name=args.dataset,
        mlp_config_path=args.mlp_config,
        max_features_in_rule=args.max_features_in_rule,
        steps_per_episode=args.steps_per_episode,
        n_instances_per_class=args.n_instances_per_class,
        eval_on_test_data=args.eval_on_test_data,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device
    )
    
    # Save results
    output_dir = args.output_dir or os.path.join(args.experiment_dir, "inference")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save rules
    def _convert_to_serializable(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (float, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: _convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    rules_filepath = os.path.join(output_dir, "extracted_rules.json")
    serializable_results = _convert_to_serializable(results)
    
    with open(rules_filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    n_anchors_total = sum(
        len(class_data.get("anchors", []))
        for class_data in serializable_results.get("per_class_results", {}).values()
    )
    n_rules_total = sum(
        len(class_data.get("rules", []))
        for class_data in serializable_results.get("per_class_results", {}).values()
    )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Rule extraction complete!")
    logger.info(f"Results saved to: {rules_filepath}")
    logger.info(f"  Total anchors saved: {n_anchors_total}")
    logger.info(f"  Total rules saved: {n_rules_total}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
