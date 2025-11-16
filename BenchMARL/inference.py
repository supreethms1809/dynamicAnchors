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
import sys
from datetime import datetime
from logging import INFO, WARNING, ERROR, CRITICAL

# Configure logging to write to both console and file
def setup_logging(log_file=None):
    """Setup logging to write to both console and a log file."""
    if log_file is None:
        # Create a temporary log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"/tmp/inference_debug_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)  # Write all debug info to file
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return log_file

# Initialize with default (will be reconfigured in main if needed)
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
    agent_id: str,
    max_steps: int = 100,
    device: str = "cpu",
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run a single rollout episode using a loaded policy on a raw PettingZoo environment.
    
    This bypasses BenchMARL/TorchRL complexity and works directly with the PettingZoo
    ParallelEnv interface, treating policies as plain MLPs.
    
    Args:
        env: Raw AnchorEnv (PettingZoo ParallelEnv) - NOT TorchRL-wrapped
        policy: Loaded policy model (plain PyTorch nn.Module)
        agent_id: Agent ID (e.g., "agent_0") - must match env's agent names
        max_steps: Maximum steps per episode
        device: Device for tensors
    
    Returns:
        Dictionary with episode data (precision, coverage, observation, etc.)
    """
    # Ensure policy is in eval mode
    policy.to(device)
    policy.eval()
    
    # Reset environment (returns dict, not TensorDict)
    obs_dict, infos_dict = env.reset(seed=seed)
    
    if agent_id not in obs_dict:
        # Try to find the actual agent name
        if hasattr(env, 'possible_agents') and len(env.possible_agents) > 0:
            agent_id = env.possible_agents[0]
            logger.warning(f"  Agent '{agent_id}' not found in reset obs, using '{agent_id}' from possible_agents")
        else:
            raise ValueError(f"Agent '{agent_id}' not found in environment. Available agents: {list(obs_dict.keys())}")
    
    # Debug: Check initial box state
    if hasattr(env, 'lower') and hasattr(env, 'upper') and agent_id in env.lower:
        lower_init = env.lower[agent_id]
        upper_init = env.upper[agent_id]
        widths_init = upper_init - lower_init
        precision_init, coverage_init, details_init = env._current_metrics(agent_id)
        logger.info(f"  Initial box state for {agent_id}:")
        logger.info(f"    Box center: {((lower_init + upper_init) / 2).mean():.4f} (mean across features)")
        logger.info(f"    Box widths: min={widths_init.min():.4f}, max={widths_init.max():.4f}, mean={widths_init.mean():.4f}")
        logger.info(f"    Initial precision: {precision_init:.4f}, coverage: {coverage_init:.4f}")
        logger.info(f"    Initial n_points: {details_init.get('n_points', 'N/A')}")
    
    done = False
    step_count = 0
    total_reward = 0.0
    
    # Store box state before first action for debugging
    lower_before = None
    upper_before = None
    if step_count == 0 and hasattr(env, 'lower') and hasattr(env, 'upper') and agent_id in env.lower:
        lower_before = env.lower[agent_id].copy()
        upper_before = env.upper[agent_id].copy()
    
    # Main rollout loop - work directly with PettingZoo dict interface
    while not done and step_count < max_steps:
        # Check if agent is still active
        if agent_id not in obs_dict:
            logger.warning(f"  Agent '{agent_id}' not in observations at step {step_count}")
            break
        
        # Get observation for this agent (raw numpy array from PettingZoo)
        obs_vec = obs_dict[agent_id]  # shape: (2*n_features + 2,) - [lower, upper, precision, coverage]
        
        # Convert to tensor for policy
        obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dim
        
        # Debug: Log observation for first step
        if step_count == 0:
            logger.info(f"  Observation shape: {obs_vec.shape}, first 5 values: {obs_vec[:5]}, last 5: {obs_vec[-5:]}")
            logger.info(f"  Observation stats: mean={obs_vec.mean():.4f}, std={obs_vec.std():.4f}, min={obs_vec.min():.4f}, max={obs_vec.max():.4f}")
        
        # Get action from policy (deterministic for inference)
        with torch.no_grad():
            # Test if policy responds to different inputs (only once)
            if step_count == 0 and not hasattr(run_rollout_with_policy, '_tested_policy_response'):
                zero_obs = torch.zeros_like(obs_tensor)
                zero_output = policy(zero_obs)
                actual_output = policy(obs_tensor)
                
                if isinstance(zero_output, torch.Tensor) and isinstance(actual_output, torch.Tensor):
                    zero_output_np = zero_output.cpu().numpy().flatten()
                    actual_output_np = actual_output.cpu().numpy().flatten()
                    diff = np.abs(actual_output_np - zero_output_np).max()
                    logger.info(f"  Policy response test: max difference between zero obs and actual obs = {diff:.2f}")
                    if diff < 1e-6:
                        logger.warning(f"  ⚠ Policy outputs IDENTICAL values for zero and actual observations!")
                    else:
                        logger.info(f"  ✓ Policy responds differently to different observations (good!)")
                run_rollout_with_policy._tested_policy_response = True
                fwd_outputs = actual_output
            else:
                fwd_outputs = policy(obs_tensor)
            
            # Debug: Log raw policy output before any processing
            if step_count == 0:
                if isinstance(fwd_outputs, torch.Tensor):
                    raw_output = fwd_outputs.cpu().numpy().flatten()
                    logger.info(f"  Raw policy output (before processing): shape={raw_output.shape}, first 5: {raw_output[:5]}, mean={raw_output.mean():.4f}, std={raw_output.std():.4f}, min={raw_output.min():.4f}, max={raw_output.max():.4f}")
                elif isinstance(fwd_outputs, dict):
                    logger.info(f"  Policy output is dict with keys: {list(fwd_outputs.keys())}")
                    for k, v in fwd_outputs.items():
                        if isinstance(v, torch.Tensor):
                            v_np = v.cpu().numpy().flatten()
                            logger.info(f"    {k}: shape={v_np.shape}, first 5: {v_np[:5]}, mean={v_np.mean():.4f}, std={v_np.std():.4f}")
            
            # Handle different policy output formats
            # BenchMARL policies may output action_dist_inputs (logits) that need TanhNormal conversion
            if isinstance(fwd_outputs, dict):
                if "action_dist_inputs" in fwd_outputs:
                    # Policy outputs logits for TanhNormal distribution
                    from torchrl.modules import TanhNormal
                    action_dist_inputs = fwd_outputs["action_dist_inputs"]
                    action_dist = TanhNormal.from_logits(action_dist_inputs)
                    # Use mean for deterministic inference
                    action = action_dist.mean() if hasattr(action_dist, "mean") else action_dist.sample()
                elif "action" in fwd_outputs:
                    action = fwd_outputs["action"]
                else:
                    # Fallback: assume output is action directly
                    action = fwd_outputs
            else:
                # Policy outputs action directly (might be raw logits that need normalization)
                action = fwd_outputs
                # If values are out of [-1, 1] range, they're likely raw logits that need normalization
                if isinstance(action, torch.Tensor):
                    action_vals = action.cpu().numpy().flatten()
                    if len(action_vals) > 0 and (action_vals.min() < -1.1 or action_vals.max() > 1.1):
                        # Values are raw logits that are too large
                        # Use a fixed scaling factor to preserve relative differences between episodes
                        # This prevents identical actions when raw outputs are proportional
                        max_abs = np.abs(action_vals).max()
                        # Use a fixed scaling factor that keeps values in a reasonable range for tanh
                        # Typical outputs are 20-30M, so divide by 30M to get values around 0.7-1.0
                        # This keeps values in tanh's active range (not saturated) while preserving differences
                        fixed_scale = 30000000.0  # 30M - keeps values in reasonable range for tanh
                        # Define sample_idx for logging (used in multiple places)
                        # Dynamically create sample indices based on actual action space size
                        action_size = len(action_vals)
                        if step_count == 0 and action_size > 0:
                            # Sample from beginning, middle, and end of action space
                            sample_idx = [0, 1, 2]  # Always include first few
                            if action_size > 6:
                                mid = action_size // 2
                                sample_idx.extend([mid - 1, mid, mid + 1])
                            if action_size > 10:
                                sample_idx.append(action_size - 1)  # Last element
                            # Ensure all indices are within bounds
                            sample_idx = [i for i in sample_idx if i < action_size]
                        else:
                            sample_idx = []
                        if max_abs > 10.0:
                            if step_count == 0:
                                logger.info(f"  Scaling large logits by fixed factor {fixed_scale:.2f} (max_abs={max_abs:.2f})")
                                logger.info(f"  Before scaling (sample): {[action_vals[i] for i in sample_idx]}")
                            action = action / fixed_scale
                            if step_count == 0:
                                action_scaled_vals = action.cpu().numpy().flatten()
                                logger.info(f"  After scaling (sample): {[action_scaled_vals[i] for i in sample_idx]}")
                        # Now apply tanh to get actions in [-1, 1]
                        action = torch.tanh(action)
                        if step_count == 0 and len(sample_idx) > 0:
                            action_tanh_vals = action.cpu().numpy().flatten()
                            logger.info(f"  After tanh (sample): {[action_tanh_vals[i] for i in sample_idx]}")
            
            # Ensure action has batch dimension to match TensorDict batch size
            if len(action.shape) == 1:
                action = action.unsqueeze(0)
            
            # Convert to numpy for debugging and ensure it's the right shape
            action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
            if step_count == 0:
                # Flatten for comparison
                action_flat = action_np.flatten()
                logger.info(f"  Action shape: {action_np.shape}, mean: {action_flat.mean():.4f}, std: {action_flat.std():.4f}, min: {action_flat.min():.4f}, max: {action_flat.max():.4f}")
                logger.info(f"  Action first 10 values: {action_flat[:10]}")
                
                # Store first action for comparison (to detect if actions are identical across episodes)
                if not hasattr(run_rollout_with_policy, '_first_action'):
                    run_rollout_with_policy._first_action = action_flat.copy()
                    run_rollout_with_policy._action_count = 1
                    logger.info(f"  Stored first action for comparison")
                else:
                    run_rollout_with_policy._action_count += 1
                    # Compare with first action
                    diff = np.abs(action_flat - run_rollout_with_policy._first_action)
                    max_diff = diff.max()
                    mean_diff = diff.mean()
                    logger.info(f"  Action comparison with first episode: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
                    if max_diff < 1e-6:
                        logger.warning(f"  ⚠ Actions are IDENTICAL to first episode! Policy may not be responding to observations.")
                    else:
                        logger.info(f"  ✓ Actions are DIFFERENT from first episode (good!)")
                    
                    # Also compare with previous action to see if there's any variation
                    if hasattr(run_rollout_with_policy, '_prev_action'):
                        prev_diff = np.abs(action_flat - run_rollout_with_policy._prev_action).max()
                        logger.info(f"  Action comparison with previous episode: max_diff={prev_diff:.6f}")
                    run_rollout_with_policy._prev_action = action_flat.copy()
                
                # Check if actions are in valid range
                if action_flat.min() < -1.1 or action_flat.max() > 1.1:
                    logger.warning(f"  ⚠ Actions out of [-1, 1] range! Applying tanh normalization...")
                    action = torch.tanh(action) if isinstance(action, torch.Tensor) else np.tanh(action)
                    action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
                    action_flat = action_np.flatten()
                    logger.info(f"  After tanh - mean: {action_flat.mean():.4f}, min: {action_flat.min():.4f}, max: {action_flat.max():.4f}")
                
                # Debug: Check if actions are all zeros or very small
                action_abs = np.abs(action_flat)
                non_zero_count = np.sum(action_abs > 1e-6)
                logger.info(f"  Action stats: {non_zero_count}/{len(action_abs)} non-zero values, max_abs: {action_abs.max():.6f}")
        
        # Convert action to numpy and remove batch dimension
        action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        if len(action_np.shape) > 1:
            action_np = action_np.squeeze(0)  # Remove batch dimension: [1, 60] -> [60]
        
        # Check if action needs to be sliced (policy might output actions for multiple agents)
        expected_action_dim = 2 * env.n_features  # Should be 60 for 30 features
        if action_np.shape[0] == 2 * expected_action_dim:
            # Policy outputs actions for 2 agents concatenated (120 dims for 2 agents * 60 each)
            # Extract the appropriate slice based on agent_id
            agent_num = int(agent_id.split('_')[-1]) if '_' in agent_id else 0
            start_idx = agent_num * expected_action_dim
            end_idx = start_idx + expected_action_dim
            action_np = action_np[start_idx:end_idx]
            if step_count == 0:
                logger.info(f"  Extracted action slice for {agent_id}: indices [{start_idx}:{end_idx}] from policy output")
        elif action_np.shape[0] != expected_action_dim:
            # Unexpected action dimension
            logger.warning(f"  ⚠ Unexpected action dimension: {action_np.shape[0]} (expected {expected_action_dim})")
            # Try to take first expected_action_dim elements
            if action_np.shape[0] > expected_action_dim:
                action_np = action_np[:expected_action_dim]
                logger.warning(f"  Taking first {expected_action_dim} elements")
        
        # Debug: Log action for first step
        if step_count == 0:
            logger.info(f"  Action before step: shape={action_np.shape}, sample values: {action_np[:5]}")
            logger.info(f"    Action lower_deltas (first {env.n_features}): mean={action_np[:env.n_features].mean():.4f}, std={action_np[:env.n_features].std():.4f}")
            logger.info(f"    Action upper_deltas (last {env.n_features}): mean={action_np[env.n_features:].mean():.4f}, std={action_np[env.n_features:].std():.4f}")
        
        # Step environment directly (raw PettingZoo interface)
        action_dict = {agent_id: action_np}
        obs_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict = env.step(action_dict)
        
        # Accumulate reward
        if agent_id in rewards_dict:
            total_reward += float(rewards_dict[agent_id])
        
        # Check if done
        done = terminations_dict.get(agent_id, False) or truncations_dict.get(agent_id, False)
        
        # Debug: Check if box changed after first action
        if step_count == 0 and lower_before is not None and upper_before is not None:
            if hasattr(env, 'lower') and hasattr(env, 'upper') and agent_id in env.lower:
                lower_after = env.lower[agent_id].copy()
                upper_after = env.upper[agent_id].copy()
                lower_diff = np.abs(lower_after - lower_before).max()
                upper_diff = np.abs(upper_after - upper_before).max()
                logger.info(f"  Box change after step 0: lower_diff={lower_diff:.6f}, upper_diff={upper_diff:.6f}")
                if lower_diff < 1e-6 and upper_diff < 1e-6:
                    logger.warning(f"  ⚠ Box did not change after action! Actions may not be applied correctly.")
        
        step_count += 1
    
    # Extract final metrics directly from environment (raw PettingZoo interface)
    episode_data = {}
    
    try:
        # Get metrics directly from environment
        precision, coverage, details = env._current_metrics(agent_id)
        
        # Get final box bounds
        lower = env.lower[agent_id]
        upper = env.upper[agent_id]
        
        # Construct final observation
        final_obs = np.concatenate([lower, upper, np.array([precision, coverage], dtype=np.float32)])
        
        episode_data = {
            "precision": float(precision),
            "coverage": float(coverage),
            "total_reward": total_reward,
            "final_observation": final_obs.tolist(),
        }
        
        logger.debug(f"  Extracted metrics from environment: precision={precision:.4f}, coverage={coverage:.4f}")
        
    except Exception as e:
        logger.warning(f"  ⚠ Error getting metrics from environment for {agent_id}: {e}")
        import traceback
        logger.debug(f"  Traceback: {traceback.format_exc()}")
        # Return empty episode data
        episode_data = {
            "precision": 0.0,
            "coverage": 0.0,
            "total_reward": total_reward,
        }
    
    return episode_data


def extract_rules_from_policies(
    experiment_dir: str,
    dataset_name: str,
    mlp_config_path: str = "conf/mlp.yaml",
    max_features_in_rule: int = 5,
    steps_per_episode: int = 100,
    n_instances_per_class: int = 20,
    eval_on_test_data: bool = True,  # Always use test data for inference by default
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
        eval_on_test_data: Whether to evaluate on test data (default: True - always uses test data for inference)
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
    # Disable coverage floor check during inference to allow exploration even with low coverage
    env_config["min_coverage_floor"] = 0.0
    env_config.update({
        "X_min": env_data["X_min"],
        "X_range": env_data["X_range"],
    })
    
    # For class-level inference, compute cluster centroids per class
    # This ensures each episode starts from a representative cluster centroid
    # (similar to training), rather than just the mean centroid
    logger.info("\nComputing cluster centroids per class for class-level inference...")
    try:
        from trainers.vecEnv import compute_cluster_centroids_per_class
        # Use a reasonable number of clusters per class (10 is a good default)
        # This allows diversity across episodes while ensuring good coverage
        n_clusters_per_class = min(10, n_instances_per_class)  # At least one cluster per instance
        cluster_centroids_per_class = compute_cluster_centroids_per_class(
            X_unit=env_data["X_unit"],
            y=env_data["y"],
            n_clusters_per_class=n_clusters_per_class,
            random_state=seed if seed is not None else 42
        )
        logger.info(f"  ✓ Cluster centroids computed successfully!")
        for cls in target_classes:
            if cls in cluster_centroids_per_class:
                n_centroids = len(cluster_centroids_per_class[cls])
                logger.info(f"    Class {cls}: {n_centroids} cluster centroids")
            else:
                logger.warning(f"    Class {cls}: No centroids available")
        
        # Set cluster centroids in env_config so environment uses them
        env_config["cluster_centroids_per_class"] = cluster_centroids_per_class
        logger.info("  ✓ Cluster centroids set in environment config")
    except ImportError as e:
        logger.warning(f"  ⚠ Could not compute cluster centroids: {e}")
        logger.warning(f"  Falling back to mean centroid per class. Install sklearn: pip install scikit-learn")
        env_config["cluster_centroids_per_class"] = None
    except Exception as e:
        logger.warning(f"  ⚠ Error computing cluster centroids: {e}")
        logger.warning(f"  Falling back to mean centroid per class")
        env_config["cluster_centroids_per_class"] = None
    
    # Always use test data for inference (unless explicitly overridden)
    if eval_on_test_data:
        if env_data.get("X_test_unit") is None:
            raise ValueError("Test data not available for evaluation. Inference requires test data.")
        env_config.update({
            "eval_on_test_data": True,
            "X_test_unit": env_data["X_test_unit"],
            "X_test_std": env_data["X_test_std"],
            "y_test": env_data["y_test"],
        })
        logger.info("✓ Using test data for inference (default behavior)")
    else:
        logger.warning("⚠ WARNING: Using training data for inference (not recommended!)")
        logger.warning("  This may lead to overoptimistic results. Use test data for proper evaluation.")
    
    # Create config for direct AnchorEnv creation (bypass BenchMARL/TorchRL)
    anchor_config = {
        "X_unit": env_data["X_unit"],
        "X_std": env_data["X_std"],
        "y": env_data["y"],
        "feature_names": feature_names,
        "classifier": dataset_loader.get_classifier(),
        "device": device,
        "target_classes": target_classes,
        "env_config": env_config,
    }
    
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
            
            # Create raw AnchorEnv directly (bypass BenchMARL/TorchRL wrapper)
            env = AnchorEnv(**single_agent_config)
            
            # Get the actual agent name from the environment
            # We need to reset once to initialize agents, but the function will reset again for the rollout
            env.reset(seed=rollout_seed)
            if hasattr(env, 'possible_agents') and len(env.possible_agents) > 0:
                actual_agent_name = env.possible_agents[0]
            elif hasattr(env, 'agents') and len(env.agents) > 0:
                actual_agent_name = env.agents[0]
            else:
                # Fallback to agent_name
                actual_agent_name = agent_name
            
            # Run rollout with raw PettingZoo environment
            episode_data = run_rollout_with_policy(
                env=env,  # Raw AnchorEnv, not TorchRL-wrapped
                policy=policy,
                agent_id=actual_agent_name,  # Use actual agent name from environment
                max_steps=steps_per_episode,
                device=device,
                seed=rollout_seed  # Pass seed for reproducible rollouts
            )
            
            # Debug logging for first episode of each class
            if instance_idx == 0:
                logger.info(f"  Debug class {target_class} episode 0:")
                logger.info(f"    actual_agent_name: {actual_agent_name}")
                logger.info(f"    episode_data keys: {list(episode_data.keys()) if episode_data else 'empty'}")
                if episode_data:
                    logger.info(f"    precision: {episode_data.get('precision', 'missing')}")
                    logger.info(f"    coverage: {episode_data.get('coverage', 'missing')}")
                    logger.info(f"    has final_observation: {'final_observation' in episode_data}")
                else:
                    logger.warning(f"    ⚠ episode_data is empty for class {target_class}!")
            
            if episode_data:
                precision = episode_data.get("precision", 0.0)
                coverage = episode_data.get("coverage", 0.0)
                
                precisions.append(float(precision))
                coverages.append(float(coverage))
            else:
                # Log warning if episode_data is empty
                if instance_idx == 0:
                    logger.warning(f"  ⚠ Warning: Empty episode_data for class {target_class}, episode {instance_idx}")
                precisions.append(0.0)
                coverages.append(0.0)
                precision = 0.0
                coverage = 0.0
            
            # Extract rule from final observation (if available)
            rule = "any values (no tightened features)"
            lower = None
            upper = None
            lower_normalized = None
            upper_normalized = None
            
            if episode_data and "final_observation" in episode_data:
                obs = np.array(episode_data["final_observation"], dtype=np.float32)
                if len(obs) == 2 * n_features + 2:
                    lower_normalized = obs[:n_features].copy()
                    upper_normalized = obs[n_features:2*n_features].copy()
                    
                    # Denormalize bounds to original feature space
                    X_min = env_config.get("X_min")
                    X_range = env_config.get("X_range")
                    if X_min is not None and X_range is not None:
                        lower = (lower_normalized * X_range) + X_min
                        upper = (upper_normalized * X_range) + X_min
                    else:
                        # Fallback to normalized if denormalization params not available
                        lower = lower_normalized
                        upper = upper_normalized
                        logger.warning(f"  ⚠ X_min/X_range not available for denormalization. Using normalized bounds.")
                    
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
                    # Use the agent name that matches the target_class
                    temp_agent_name = f"agent_{target_class}"
                    # Set normalized bounds in temp_env (extract_rule will denormalize internally)
                    temp_env.lower[temp_agent_name] = lower_normalized
                    temp_env.upper[temp_agent_name] = upper_normalized
                    
                    # Extract rule with denormalization enabled
                    rule = temp_env.extract_rule(
                        temp_agent_name,
                        max_features_in_rule=max_features_in_rule,
                        denormalize=True  # Denormalize to original feature space
                    )
            
            anchor_data = {
                "instance_idx": instance_idx,
                "precision": float(precision),
                "coverage": float(coverage),
                "total_reward": float(episode_data.get("total_reward", 0.0)) if episode_data else 0.0,
                "rule": rule,
            }
            
            if lower is not None and upper is not None:
                anchor_data.update({
                    "lower_bounds": lower.tolist(),  # Denormalized bounds in original feature space
                    "upper_bounds": upper.tolist(),  # Denormalized bounds in original feature space
                    "box_widths": (upper - lower).tolist(),
                    "box_volume": float(np.prod(np.maximum(upper - lower, 1e-9))),
                    # Also save normalized bounds for reference
                    "lower_bounds_normalized": lower_normalized.tolist() if lower_normalized is not None else None,
                    "upper_bounds_normalized": upper_normalized.tolist() if upper_normalized is not None else None,
                })
            
            anchors_list.append(anchor_data)
            rules_list.append(rule)
        
        unique_rules = list(set([r for r in rules_list if r and r != "any values (no tightened features)"]))
        
        results["per_class_results"][class_key] = {
            "class": int(target_class),
            "group": agent_name,
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
    
    # Setup logging first (before any other operations)
    log_file = setup_logging()
    logger.info(f"Debug log file: {log_file}")
    logger.info("="*80)
    
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
        "--eval_on_train_data",
        action="store_true",
        help="Override default and evaluate on training data instead of test data (not recommended). Default: uses test data."
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
    
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to debug log file (default: /tmp/inference_debug_<timestamp>.log)"
    )
    
    args = parser.parse_args()
    
    # Reconfigure logging with custom log file if provided
    if args.log_file:
        log_file = setup_logging(args.log_file)
        logger.info(f"Using custom debug log file: {log_file}")
    
    # Always use test data by default (can be overridden with --eval_on_train_data)
    use_test_data = not args.eval_on_train_data
    
    if not use_test_data:
        logger.warning("⚠ WARNING: Running inference on training data is not recommended!")
        logger.warning("  Inference should typically use test data to evaluate generalization.")
    else:
        logger.info("✓ Using test data for inference (default behavior)")
    
    # Extract rules
    results = extract_rules_from_policies(
        experiment_dir=args.experiment_dir,
        dataset_name=args.dataset,
        mlp_config_path=args.mlp_config,
        max_features_in_rule=args.max_features_in_rule,
        steps_per_episode=args.steps_per_episode,
        n_instances_per_class=args.n_instances_per_class,
        eval_on_test_data=use_test_data,
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
    logger.info(f"\nDebug log file saved to: {log_file}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
