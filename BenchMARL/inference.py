"""
Inference script for extracting anchor rules using saved policy models.

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
import time
from tensordict import TensorDict

import logging
import sys
from datetime import datetime
from logging import INFO, WARNING, ERROR, CRITICAL

# Configure logging to write to both console and file
def setup_logging(log_file=None):
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

# Initialize with default
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_policy_model(
    policy_path: str,
    metadata_path: str,
    mlp_config_path: str,
    device: str = "cpu"
) -> torch.nn.Module:
    # Load metadata to understand model structure
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load MLP config
    mlp_config = MlpConfig.get_from_yaml(mlp_config_path)
    
    # Load state dict to infer input/output dimensions
    state_dict = torch.load(policy_path, map_location=device)
    
    # Check if state_dict has nested structure
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
    
    # Infer hidden layer sizes from the state_dict weights
    # Sort all weight keys to get layers in order
    weight_keys = sorted([k for k in state_dict.keys() if 'weight' in k])
    
    # Extract hidden layer sizes from weight matrices
    # For MLP: layer i weight shape is [out_dim, in_dim]
    # Hidden layers are all layers except first (input->hidden) and last (hidden->output)
    hidden_sizes = []
    for i, key in enumerate(weight_keys):
        weight_shape = state_dict[key].shape
        if i == 0:
            # First layer: [hidden1, input_dim]
            hidden_sizes.append(weight_shape[0])
        elif i < len(weight_keys) - 1:
            # Hidden layers: [hidden_i+1, hidden_i]
            hidden_sizes.append(weight_shape[0])
        else:
            # Last layer: [output_dim, hidden_last]
            output_dim = weight_shape[0]
    
    if output_dim is None:
        raise ValueError(f"Could not infer output dimension from {policy_path}")
    
    logger.info(f"  Inferred model dimensions: input={input_dim}, output={output_dim}")
    logger.info(f"  Inferred hidden layer sizes from checkpoint: {hidden_sizes}")
    
    # Create MLP model using the inferred architecture from the checkpoint
    # This ensures we match the actual trained model, not the config file
    from benchmarl.models.common import ModelConfig
    from torchrl.modules import MLP
    
    # Build MLP with the inferred dimensions from the checkpoint
    mlp = MLP(
        in_features=input_dim,
        out_features=output_dim,
        num_cells=hidden_sizes,  # Use inferred hidden sizes, not config file
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


def _should_log_verbose(env_config: Optional[Dict[str, Any]] = None) -> bool:
    """Check if verbose logging is enabled from env_config."""
    if env_config is None:
        return False
    verbosity = env_config.get("logging_verbosity", "normal")
    return verbosity == "verbose"


def _should_log_info(env_config: Optional[Dict[str, Any]] = None) -> bool:
    """Check if info-level logging should be enabled from env_config."""
    if env_config is None:
        return True  # Default to logging info
    verbosity = env_config.get("logging_verbosity", "normal")
    # "quiet" = only warnings/errors, "normal" and "verbose" = info level logging
    return verbosity != "quiet"


def _get_logging_level(env_config: Optional[Dict[str, Any]] = None) -> int:
    """Get the appropriate logging level based on verbosity setting."""
    if env_config is None:
        return logging.INFO
    verbosity = env_config.get("logging_verbosity", "normal")
    if verbosity == "quiet":
        return logging.WARNING  # Only warnings and errors
    elif verbosity == "verbose":
        return logging.DEBUG  # All logs including debug
    else:  # normal
        return logging.INFO  # Standard info level


def _apply_logging_verbosity(env_config: Optional[Dict[str, Any]] = None):
    """Set logging level for all loggers based on verbosity setting."""
    level = _get_logging_level(env_config)
    # Set level for root logger - this affects all loggers unless they have their own level set
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Update all existing handlers to use the new level
    for handler in root_logger.handlers:
        handler.setLevel(level)
    
    # Also set for common module loggers (both full names and short names)
    logger_names = [
        'BenchMARL', 'single_agent', 'tabular_datasets', 'anchor_trainer_sb3', 
        'single_agent_inference', 'test_extracted_rules_single', 'inference',
        'environment', 'benchmarl_wrappers', 'anchor_trainer', 'BenchMARL.environment',
        'BenchMARL.benchmarl_wrappers', 'BenchMARL.anchor_trainer', 'BenchMARL.tabular_datasets',
        'single_agent.anchor_trainer_sb3', 'single_agent.single_agent_inference',
        'single_agent.test_extracted_rules_single'
    ]
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        # Also update handlers for this logger
        for handler in logger.handlers:
            handler.setLevel(level)
    
    # Set level for all existing loggers (walk through the logger hierarchy)
    # This ensures we catch any loggers that were already created
    for name in logging.Logger.manager.loggerDict:
        if isinstance(logging.Logger.manager.loggerDict[name], logging.Logger):
            log = logging.getLogger(name)
            log.setLevel(level)
            for handler in log.handlers:
                handler.setLevel(level)


def run_rollout_with_policy(
    env: AnchorEnv,
    policy: torch.nn.Module,
    agent_id: str,
    max_steps: int = 100,
    device: str = "cpu",
    seed: Optional[int] = None,
    exploration_mode: str = "sample",  # "sample", "mean", or "noisy_mean"
    action_noise_scale: float = 0.05,  # Noise scale for actions (0.0 = no noise)
    verbose_logging: bool = False,  # Enable verbose debug logging
) -> Dict[str, Any]:
    # Ensure policy is in eval mode
    policy.to(device)
    policy.eval()
    
    # Set mode to "inference" so termination counters are reset in reset()
    if hasattr(env, 'mode'):
        env.mode = "inference"
    
    # Reset environment (returns dict, not TensorDict)
    obs_dict, infos_dict = env.reset(seed=seed)
    
    # Start timing the rollout
    rollout_start_time = time.perf_counter()
    
    if agent_id not in obs_dict:
        # Try to find the actual agent name
        if hasattr(env, 'possible_agents') and len(env.possible_agents) > 0:
            agent_id = env.possible_agents[0]
            logger.warning(f"  Agent '{agent_id}' not found in reset obs, using '{agent_id}' from possible_agents")
        else:
            raise ValueError(f"Agent '{agent_id}' not found in environment. Available agents: {list(obs_dict.keys())}")
    
    # Debug: Check initial box state
    if verbose_logging and hasattr(env, 'lower') and hasattr(env, 'upper') and agent_id in env.lower:
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
        
        # Get observation for this agent
        # shape: (2*n_features + 2,) - [lower, upper, precision, coverage]
        obs_vec = obs_dict[agent_id]
        
        # Convert to tensor for policy
        obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Debug: Log observation for first step
        if verbose_logging and step_count == 0:
            logger.info(f"  Observation shape: {obs_vec.shape}, first 5 values: {obs_vec[:5]}, last 5: {obs_vec[-5:]}")
            logger.info(f"  Observation stats: mean={obs_vec.mean():.4f}, std={obs_vec.std():.4f}, min={obs_vec.min():.4f}, max={obs_vec.max():.4f}")
        
        # Get action from policy
        action_has_noise = False
        with torch.no_grad():
            # Test if policy responds to different inputs
            if step_count == 0 and not hasattr(run_rollout_with_policy, '_tested_policy_response'):
                zero_obs = torch.zeros_like(obs_tensor)
                zero_output = policy(zero_obs)
                actual_output = policy(obs_tensor)
                
                if isinstance(zero_output, torch.Tensor) and isinstance(actual_output, torch.Tensor):
                    zero_output_np = zero_output.cpu().numpy().flatten()
                    actual_output_np = actual_output.cpu().numpy().flatten()
                    diff = np.abs(actual_output_np - zero_output_np).max()
                    if verbose_logging:
                        logger.info(f"  Policy response test: max difference between zero obs and actual obs = {diff:.2f}")
                    if diff < 1e-6:
                        logger.warning(f"  Policy outputs IDENTICAL values for zero and actual observations!")
                    elif verbose_logging:
                        logger.info(f"  Policy responds differently to different observations (good!)")
                run_rollout_with_policy._tested_policy_response = True
                fwd_outputs = actual_output
            else:
                fwd_outputs = policy(obs_tensor)
            
            # Debug: Log raw policy output before any processing
            if verbose_logging and step_count == 0:
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
            # BenchMARL policies outputs action_dist_inputs-logits that need TanhNormal conversion
            if isinstance(fwd_outputs, dict):
                if "action_dist_inputs" in fwd_outputs:
                    # Policy outputs logits for TanhNormal distribution
                    from torchrl.modules import TanhNormal
                    action_dist_inputs = fwd_outputs["action_dist_inputs"]
                    action_dist = TanhNormal.from_logits(action_dist_inputs)
                    # Use exploration mode for diversity: sample, mean, or noisy_mean
                    action_has_noise = False  # Track if noise was already added
                    if exploration_mode == "sample":
                        # Sample from distribution for diversity
                        action = action_dist.sample()
                        action_has_noise = True  # Sampling provides randomness
                    elif exploration_mode == "noisy_mean":
                        # Add Gaussian noise to mean action
                        mean_action = action_dist.mean() if hasattr(action_dist, "mean") else action_dist.sample()
                        noise = torch.randn_like(mean_action) * action_noise_scale
                        action = torch.clamp(mean_action + noise, -1.0, 1.0)
                        action_has_noise = True
                    else:
                        action = action_dist.mean() if hasattr(action_dist, "mean") else action_dist.sample()
                        action_has_noise = False
                elif "action" in fwd_outputs:
                    action = fwd_outputs["action"]
                    action_has_noise = False
                else:
                    # Fallback: assume output is action directly
                    action = fwd_outputs
                    action_has_noise = False
            else:
                # Policy outputs action directly
                action = fwd_outputs
                action_has_noise = False
                # If values are out of [-1, 1] range, they're likely raw logits that need normalization
                if isinstance(action, torch.Tensor):
                    action_vals = action.cpu().numpy().flatten()
                    if len(action_vals) > 0 and (action_vals.min() < -1.1 or action_vals.max() > 1.1):
                        # Values are raw logits that are too large
                        # Use a fixed scaling factor to preserve relative differences between episodes
                        # Prevents identical actions when raw outputs are proportional
                        max_abs = np.abs(action_vals).max()
                        # Use a fixed scaling factor that keeps values in a reasonable range for tanh
                        # Typical outputs are 20-30M, so divide by 30M to get values around 0.7-1.0
                        # keeps values in tanh's active range (not saturated) while preserving differences
                        fixed_scale = 30000000.0
                        # Define sample_idx for logging (used in multiple places)
                        # create sample indices based on actual action space size
                        action_size = len(action_vals)
                        if step_count == 0 and action_size > 0:
                            # sample from beginning, middle, and end of action space
                            sample_idx = [0, 1, 2]
                            if action_size > 6:
                                mid = action_size // 2
                                sample_idx.extend([mid - 1, mid, mid + 1])
                            if action_size > 10:
                                sample_idx.append(action_size - 1)
                            # Ensure all indices are within bounds
                            sample_idx = [i for i in sample_idx if i < action_size]
                        else:
                            sample_idx = []
                        if max_abs > 10.0:
                            if verbose_logging and step_count == 0:
                                logger.info(f"  Scaling large logits by fixed factor {fixed_scale:.2f} (max_abs={max_abs:.2f})")
                                logger.info(f"  Before scaling (sample): {[action_vals[i] for i in sample_idx]}")
                            action = action / fixed_scale
                            if verbose_logging and step_count == 0:
                                action_scaled_vals = action.cpu().numpy().flatten()
                                logger.info(f"  After scaling (sample): {[action_scaled_vals[i] for i in sample_idx]}")
                        # Now apply tanh to get actions in [-1, 1]
                        action = torch.tanh(action)
                        if verbose_logging and step_count == 0 and len(sample_idx) > 0:
                            action_tanh_vals = action.cpu().numpy().flatten()
                            logger.info(f"  After tanh (sample): {[action_tanh_vals[i] for i in sample_idx]}")
            
            # Ensure action has batch dimension to match TensorDict batch size
            if len(action.shape) == 1:
                action = action.unsqueeze(0)
            
            # Convert to numpy for debugging and ensure it's the right shape
            action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
            if verbose_logging and step_count == 0:
                # Flatten for comparison
                action_flat = action_np.flatten()
                logger.info(f"  Action shape: {action_np.shape}, mean: {action_flat.mean():.4f}, std: {action_flat.std():.4f}, min: {action_flat.min():.4f}, max: {action_flat.max():.4f}")
                logger.info(f"  Action first 10 values: {action_flat[:10]}")
                
                # Store first action for comparison
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
                        logger.warning(f"  Actions are IDENTICAL to first episode! Policy may not be responding to observations.")
                    else:
                        logger.info(f"  Actions are DIFFERENT from first episode (good!)")
                    
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
        
        # Add action noise if enabled (for actions that don't already have noise)
        # This applies to "mean" mode or non-distribution actions
        if action_noise_scale > 0.0 and not action_has_noise:
                if isinstance(action, torch.Tensor):
                    noise = torch.randn_like(action) * action_noise_scale
                    action = torch.clamp(action + noise, -1.0, 1.0)
                else:
                    # Convert to tensor, add noise, convert back
                    action_tensor = torch.as_tensor(action, dtype=torch.float32, device=device)
                    noise = torch.randn_like(action_tensor) * action_noise_scale
                    action = torch.clamp(action_tensor + noise, -1.0, 1.0)
                    action = action.cpu().numpy()
        
        # Convert action to numpy and remove batch dimension
        action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        if len(action_np.shape) > 1:
            action_np = action_np.squeeze(0)
        
        # Check if action needs to be sliced (policy might output actions for multiple agents)
        expected_action_dim = 2 * env.n_features
        agent_num = int(agent_id.split('_')[-1]) if '_' in agent_id else 0
        
        if action_np.shape[0] == 2 * expected_action_dim:
            # Policy outputs actions for 2 agents concatenated
            # Extract the appropriate slice based on agent_id
            # If agent_num >= 2, wrap around to use agent_0 or agent_1
            num_agents_in_policy = 2
            mapped_agent_num = agent_num % num_agents_in_policy
            start_idx = mapped_agent_num * expected_action_dim
            end_idx = start_idx + expected_action_dim
            action_np = action_np[start_idx:end_idx]
            if verbose_logging and step_count == 0:
                if agent_num != mapped_agent_num:
                    logger.info(f"  Extracted action slice for {agent_id}: mapped agent_{agent_num} -> agent_{mapped_agent_num}, indices [{start_idx}:{end_idx}] from policy output")
                else:
                    logger.info(f"  Extracted action slice for {agent_id}: indices [{start_idx}:{end_idx}] from policy output")
        elif action_np.shape[0] == 4 * expected_action_dim:
            # Policy outputs actions for 4 agents concatenated
            start_idx = agent_num * expected_action_dim
            end_idx = start_idx + expected_action_dim
            action_np = action_np[start_idx:end_idx]
            if verbose_logging and step_count == 0:
                logger.info(f"  Extracted action slice for {agent_id}: indices [{start_idx}:{end_idx}] from policy output")
        elif action_np.shape[0] != expected_action_dim:
            # Unexpected action dimension
            logger.warning(f"  Unexpected action dimension: {action_np.shape[0]} (expected {expected_action_dim})")
            # Try to take first expected_action_dim elements
            if action_np.shape[0] > expected_action_dim:
                action_np = action_np[:expected_action_dim]
                logger.warning(f"  Taking first {expected_action_dim} elements")
        
        # Validate that action is not empty
        if action_np.shape[0] == 0:
            original_action_shape = action.shape if hasattr(action, 'shape') else (action_np.shape[0] if hasattr(action_np, 'shape') else 'unknown')
            raise ValueError(f"Empty action extracted for {agent_id}. Original action shape: {original_action_shape}, agent_num: {agent_num}, expected_action_dim: {expected_action_dim}")
        
        # Debug: Log action for first step (TODO: Remove this)
        if step_count == 0:
            if verbose_logging:
                logger.info(f"  Action before step: shape={action_np.shape}, sample values: {action_np[:5]}")
                logger.info(f"    Action lower_deltas (first {env.n_features}): mean={action_np[:env.n_features].mean():.4f}, std={action_np[:env.n_features].std():.4f}")
                logger.info(f"    Action upper_deltas (last {env.n_features}): mean={action_np[env.n_features:].mean():.4f}, std={action_np[env.n_features:].std():.4f}")
        
        # Step environment directly
        action_dict = {agent_id: action_np}
        obs_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict = env.step(action_dict)
        
        # Accumulate reward
        if agent_id in rewards_dict:
            total_reward += float(rewards_dict[agent_id])
        
        # Check if done
        done = terminations_dict.get(agent_id, False) or truncations_dict.get(agent_id, False)
        
        # Debug: Check if box changed after first action (TODO: Remove this)
        if step_count == 0 and lower_before is not None and upper_before is not None:
            if hasattr(env, 'lower') and hasattr(env, 'upper') and agent_id in env.lower:
                lower_after = env.lower[agent_id].copy()
                upper_after = env.upper[agent_id].copy()
                lower_diff = np.abs(lower_after - lower_before).max()
                upper_diff = np.abs(upper_after - upper_before).max()
                if verbose_logging:
                    logger.info(f"  Box change after step 0: lower_diff={lower_diff:.6f}, upper_diff={upper_diff:.6f}")
                if lower_diff < 1e-6 and upper_diff < 1e-6:
                    logger.warning(f"  Box did not change after action! Actions may not be applied correctly.")
        
        step_count += 1
    
    # End timing the rollout
    rollout_end_time = time.perf_counter()
    rollout_duration = rollout_end_time - rollout_start_time
    
    # Extract final metrics directly from environment
    episode_data = {}
    
    try:
        # Get instance-level metrics directly from environment
        instance_precision, instance_coverage, details = env._current_metrics(agent_id)
        
        # Get class-level metrics (union of all agents for this class)
        target_class = env._get_class_for_agent(agent_id)
        class_union_metrics = {}
        class_precision = 0.0
        class_coverage = 0.0
        
        if target_class is not None:
            # Compute class-level union metrics
            class_union_metrics = env._compute_class_union_metrics()
            if target_class in class_union_metrics:
                class_precision = float(class_union_metrics[target_class].get("union_precision", 0.0))
                class_coverage = float(class_union_metrics[target_class].get("union_coverage", 0.0))
            else:
                # Debug: log why class-level metrics might be missing
                if verbose_logging:
                    logger.debug(f"  Class {target_class} not found in class_union_metrics. Available classes: {list(class_union_metrics.keys())}")
                    logger.debug(f"  Agents with boxes: {list(env.lower.keys())}")
                    logger.debug(f"  Active agents: {list(env.agents) if hasattr(env, 'agents') else 'N/A'}")
        
        # Get final box bounds
        lower = env.lower[agent_id]
        upper = env.upper[agent_id]
        
        # Construct final observation (using instance-level metrics)
        final_obs = np.concatenate([lower, upper, np.array([instance_precision, instance_coverage], dtype=np.float32)])
        
        episode_data = {
            # Instance-level metrics (for this specific agent/instance)
            "instance_precision": float(instance_precision),
            "instance_coverage": float(instance_coverage),
            # Anchor metrics (same as instance-level for single agent)
            "anchor_precision": float(instance_precision),
            "anchor_coverage": float(instance_coverage),
            # Class-level metrics (union of all agents for this class)
            "class_precision": float(class_precision),
            "class_coverage": float(class_coverage),
            "total_reward": total_reward,
            "n_steps": step_count,
            "rollout_time_seconds": float(rollout_duration),
            "final_observation": final_obs.tolist(),
        }
        
        # Debug: Log metrics
        if verbose_logging:
            logger.debug(
                f"  Extracted metrics from environment: "
                f"instance_precision={instance_precision:.4f}, instance_coverage={instance_coverage:.4f}, "
                f"class_precision={class_precision:.4f}, class_coverage={class_coverage:.4f}"
            )
        
    except Exception as e:
        logger.warning(f"  Error getting metrics from environment for {agent_id}: {e}")
        if verbose_logging:
            import traceback
            logger.debug(f"  Traceback: {traceback.format_exc()}")
        # Return empty episode data
        episode_data = {
            "instance_precision": 0.0,
            "instance_coverage": 0.0,
            "anchor_precision": 0.0,
            "anchor_coverage": 0.0,
            "class_precision": 0.0,
            "class_coverage": 0.0,
            "total_reward": total_reward,
            "n_steps": step_count,
            "rollout_time_seconds": float(rollout_duration),
        }
    
    return episode_data


def extract_rules_from_policies(
    experiment_dir: str,
    dataset_name: str,
    mlp_config_path: str = "conf/mlp.yaml",
    max_features_in_rule: int = -1,
    steps_per_episode: int = 500,
    n_instances_per_class: int = 20,
    eval_on_test_data: bool = True,
    output_dir: Optional[str] = None,
    seed: int = 42,
    device: str = "cpu",
    exploration_mode: str = "sample",
    action_noise_scale: float = 0.05,
) -> Dict[str, Any]:
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
    agents_per_class = 1
    
    index_path = os.path.join(individual_models_dir, "policies_index.json")
    if os.path.exists(index_path):
        logger.info(f"Found policies_index.json, using it to locate policies...")
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        agents_per_class = index_data.get("agents_per_class", 1)
        policies_by_class = index_data.get("policies_by_class", {})
        
        for class_key, class_info in policies_by_class.items():
            class_policies = class_info.get("policies", [])
            for policy_info in class_policies:
                group = policy_info.get("group") or policy_info.get("agent")
                policy_file = policy_info.get("policy_file")
                metadata_file = policy_info.get("metadata_file")
                
                if group and policy_file:
                    # Resolve relative paths
                    policy_path = os.path.join(individual_models_dir, policy_file)
                    if os.path.exists(policy_path):
                        policy_files[group] = policy_path
                        
                        if metadata_file:
                            metadata_path = os.path.join(individual_models_dir, metadata_file)
                            if os.path.exists(metadata_path):
                                metadata_files[group] = metadata_path
        
        logger.info(f"  Loaded {len(policy_files)} policies from index (agents_per_class={agents_per_class})")
    else:
        # Fallback: Search for policies in flat structure - older version
        logger.info("No policies_index.json found, searching for policies in flat structure...")
        
        # Search in root directory
        for filename in os.listdir(individual_models_dir):
            if filename.startswith("policy_") and filename.endswith(".pth"):
                group = filename.replace("policy_", "").replace(".pth", "")
                policy_files[group] = os.path.join(individual_models_dir, filename)
                
                # Look for corresponding metadata file
                metadata_filename = f"policy_{group}_metadata.json"
                metadata_path = os.path.join(individual_models_dir, metadata_filename)
                if os.path.exists(metadata_path):
                    metadata_files[group] = metadata_path
        
        # Also search in class subdirectories
        for item in os.listdir(individual_models_dir):
            item_path = os.path.join(individual_models_dir, item)
            if os.path.isdir(item_path) and item.startswith("class_"):
                # This is a class directory
                for filename in os.listdir(item_path):
                    if filename.startswith("policy_") and filename.endswith(".pth"):
                        group = filename.replace("policy_", "").replace(".pth", "")
                        policy_files[group] = os.path.join(item_path, filename)
                        
                        # Look for corresponding metadata file
                        metadata_filename = f"policy_{group}_metadata.json"
                        metadata_path = os.path.join(item_path, metadata_filename)
                        if os.path.exists(metadata_path):
                            metadata_files[group] = metadata_path
    
    if not policy_files:
        raise ValueError(
            f"No policy models found in {individual_models_dir}\n"
            f"Expected files like: policy_<group>.pth or class_*/policy_<group>.pth\n"
            f"Or a policies_index.json file."
        )
    
    logger.info(f"\nFound {len(policy_files)} policy model(s):")
    for group in sorted(policy_files.keys()):
        logger.info(f"  - {group}: {policy_files[group]}")
    
    if agents_per_class > 1:
        logger.info(f"\nNote: Multiple agents per class ({agents_per_class}) detected.")
        logger.info(f"  Policies are organized by class.")
    
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
    
    # Create trainer early to get env_config and check verbosity
    from anchor_trainer import AnchorTrainer
    trainer = AnchorTrainer(
        dataset_loader=dataset_loader,
        algorithm="maddpg",
        output_dir=output_dir or os.path.join(experiment_dir, "inference"),
        seed=seed
    )
    env_config = trainer._get_default_env_config()
    
    # Check logging verbosity from config
    verbose_logging = _should_log_verbose(env_config)
    if verbose_logging:
        logger.info("Verbose logging enabled - showing detailed debug information")
    
    # Load all policy models and extract individual agent policies
    logger.info(f"\nLoading policy models...")
    policies = {}
    agent_policies = {}
    
    for group in sorted(policy_files.keys()):
        metadata_path = metadata_files.get(group)
        if metadata_path is None:
            logger.warning(f"  Warning: No metadata found for {group}, using defaults")
        
        # Load the combined policy
        combined_policy = load_policy_model(
            policy_path=policy_files[group],
            metadata_path=metadata_path or "",
            mlp_config_path=mlp_config_path,
            device=device
        )
        policies[group] = combined_policy
        logger.info(f"  Loaded combined policy for {group}")
        
        # Try to extract individual agent policies from the combined policy
        # Check if the state_dict has agent-specific keys
        state_dict_path = policy_files[group]
        raw_state_dict = torch.load(state_dict_path, map_location=device)
        
        # Debug: Print some keys to understand structure
        # Only log if verbose logging is enabled
        if verbose_logging:
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
        if verbose_logging:
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
                    
                    # Key starts with agent index
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
                    
                    # Agent index in different position
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
                            if agent_pos == 0:
                                # "0.mlp.params.0.weight" -> already handled above
                                pass
                            else:
                                new_key = '.'.join(parts[agent_pos+1:])
                                if new_key and not new_key.startswith('__'):
                                    agent_state_dict[new_key] = value
                
                if verbose_logging:
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
                        
                        # Infer hidden layer sizes from the state_dict weights
                        weight_keys = sorted([k for k in agent_state_dict.keys() if 'weight' in k])
                        hidden_sizes = []
                        for i, key in enumerate(weight_keys):
                            weight_shape = agent_state_dict[key].shape
                            if i == 0:
                                # First layer: [hidden1, input_dim]
                                hidden_sizes.append(weight_shape[0])
                            elif i < len(weight_keys) - 1:
                                # Hidden layers: [hidden_i+1, hidden_i]
                                hidden_sizes.append(weight_shape[0])
                            else:
                                # Last layer: [output_dim, hidden_last]
                                output_dim = weight_shape[0]
                        
                        if output_dim:
                            # Create MLP for this agent using inferred architecture
                            agent_mlp = MLP(
                                in_features=input_dim,
                                out_features=output_dim,
                                num_cells=hidden_sizes,  # Use inferred hidden sizes, not config file
                                activation_class=mlp_config.activation_class,
                                activation_kwargs=mlp_config.activation_kwargs or {},
                                norm_class=mlp_config.norm_class,
                                norm_kwargs=mlp_config.norm_kwargs or {},
                                layer_class=mlp_config.layer_class,
                            ).to(device)
                            
                            agent_mlp.load_state_dict(agent_state_dict)
                            agent_mlp.eval()
                            agent_policies[agent_name] = agent_mlp
                            logger.info(f"  Extracted policy for {agent_name} (hidden sizes: {hidden_sizes})")
                        else:
                            logger.warning(f"  Could not infer output dimension for {agent_name}")
                    else:
                        logger.warning(f"  Could not find weight parameters for {agent_name}")
                else:
                    logger.warning(f"  Could not extract state_dict for {agent_name}")
        else:
            # No agent-specific keys found, policy might be shared
            # We'll handle this after we know the target_classes
            logger.info(f"  Debug: No agent-specific indices found, will use shared policy")
    
    # Check if we have policies for all expected agents
    # Expected agents are agent_0, agent_1, ..., agent_{n_classes-1}
    expected_agent_names = [f"agent_{i}" for i in range(len(target_classes))]
    missing_agents = [name for name in expected_agent_names if name not in agent_policies]
    
    # Check if we have a combined policy available (for shared policy scenarios)
    combined_policy = None
    for g in policies.keys():
        if policies[g] is not None:
            combined_policy = policies[g]
            break
    
    if missing_agents:
        # If we have a combined policy available, this is likely a shared policy scenario
        # (not an error, just informational)
        if combined_policy is not None:
            logger.info(f"\n  Info: Using shared policy for agents: {missing_agents}")
            logger.info(f"  Found individual policies for: {list(agent_policies.keys())}")
            logger.info(f"  Expected policies for: {expected_agent_names}")
            logger.info(f"  Using combined/shared policy for missing agents (this is normal for shared policies)")
            
            # Use the combined policy for the missing agents (assuming shared policy)
            for missing_agent in missing_agents:
                agent_policies[missing_agent] = combined_policy
                logger.info(f"  Assigned shared policy to {missing_agent}")
        else:
            # This is a real problem - no policy available
            logger.warning(f"\n  Warning: Missing policies for agents: {missing_agents}")
            logger.warning(f"  Found policies for: {list(agent_policies.keys())}")
            logger.warning(f"  Expected policies for: {expected_agent_names}")
            logger.warning(f"  Error: No combined policy available to assign to missing agents")
    
    # Disable coverage floor check during inference to allow exploration even with low coverage
    env_config["min_coverage_floor"] = 0.0
    env_config.update({
        "X_min": env_data["X_min"],
        "X_range": env_data["X_range"],
    })
    
    # Apply logging verbosity early (before any logging happens)
    verbose_logging = _should_log_verbose(env_config)
    _apply_logging_verbosity(env_config)
    if verbose_logging:
        logger.info("Verbose logging enabled - showing detailed debug information")
    elif env_config.get("logging_verbosity") == "quiet":
        logger.warning("Quiet logging mode enabled - only warnings and errors will be shown")
    
    # For class-level inference, compute cluster centroids per class
    logger.info("\nComputing cluster centroids per class for class-level inference...")
    try:
        from utils.clusters import compute_cluster_centroids_per_class
        
        if agents_per_class > 1:
            n_clusters_per_class = agents_per_class * 10
            logger.info(f"  Using {n_clusters_per_class} clusters per class ({n_clusters_per_class // agents_per_class} per agent) for k-means clustering")
        else:
            n_clusters_per_class = min(10, n_instances_per_class)
            logger.info(f"  Using {n_clusters_per_class} clusters per class for diversity (matching training and single-agent behavior)")
        
        cluster_centroids_per_class = compute_cluster_centroids_per_class(
            X_unit=env_data["X_unit"],
            y=env_data["y"],
            n_clusters_per_class=n_clusters_per_class,
            random_state=seed if seed is not None else 42
        )
        logger.info(f"  Cluster centroids computed successfully!")
        for cls in target_classes:
            if cls in cluster_centroids_per_class:
                n_centroids = len(cluster_centroids_per_class[cls])
                logger.info(f"    Class {cls}: {n_centroids} cluster centroids")
            else:
                logger.warning(f"    Class {cls}: No centroids available")
        
        # Set cluster centroids in env_config so environment uses them
        env_config["cluster_centroids_per_class"] = cluster_centroids_per_class
        logger.info("  Cluster centroids set in environment config")
    except ImportError as e:
        logger.warning(f"  Could not compute cluster centroids: {e}")
        logger.warning(f"  Falling back to mean centroid per class. Install sklearn: pip install scikit-learn")
        env_config["cluster_centroids_per_class"] = None
    except Exception as e:
        logger.warning(f"  Error computing cluster centroids: {e}")
        logger.warning(f"  Falling back to mean centroid per class")
        env_config["cluster_centroids_per_class"] = None
    
    # Check logging verbosity from config
    verbose_logging = _should_log_verbose(env_config)
    if verbose_logging:
        logger.info("Verbose logging enabled - showing detailed debug information")
    
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
        logger.info("Using test data for inference (default behavior)")
    else:
        logger.warning("WARNING: Using training data for inference (not recommended!)")
        logger.warning("  This may lead to overoptimistic results. Use test data for proper evaluation.")
    
    # Create config for direct AnchorEnv creation
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
    # Note: When agents_per_class > 1, we need to handle multiple agents per class
    if not agent_policies or len(agent_policies) < len(target_classes):
        logger.info(f"  Creating policies for all target classes...")
        for group in sorted(policy_files.keys()):
            combined_policy = policies[group]
            
            # Determine which class this policy belongs to
            target_class = None
            if group in metadata_files:
                try:
                    with open(metadata_files[group], 'r') as f:
                        metadata = json.load(f)
                        target_class = metadata.get("target_class")
                except:
                    pass
            
            # If we can't determine class from metadata, try parsing from group name
            if target_class is None and group.startswith("agent_"):
                try:
                    parts = group.split("_")
                    if len(parts) >= 2 and parts[1].isdigit():
                        target_class = int(parts[1])
                except:
                    pass
            
            # Map policy to all agents of the same class
            if target_class is not None:
                # When agents_per_class > 1, create agents like "agent_0_0", "agent_0_1", etc.
                # When agents_per_class == 1, create agent like "agent_0"
                if agents_per_class > 1:
                    for k in range(agents_per_class):
                        agent_name = f"agent_{target_class}_{k}"
                        if agent_name not in agent_policies:
                            agent_policies[agent_name] = combined_policy
                            logger.info(f"  Using policy for {agent_name} (class {target_class})")
                else:
                    agent_name = f"agent_{target_class}"
                    if agent_name not in agent_policies:
                        agent_policies[agent_name] = combined_policy
                        logger.info(f"  Using policy for {agent_name}")
            else:
                # Fallback: map to all classes if we can't determine class
                for cls in target_classes:
                    agent_name = f"agent_{cls}"
                    if agent_name not in agent_policies:
                        agent_policies[agent_name] = combined_policy
                        logger.info(f"  Using shared policy for {agent_name}")
    
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
    logger.info(f"  Exploration mode: {exploration_mode}")
    logger.info(f"  Action noise scale: {action_noise_scale}")
    
    # Start overall timing
    overall_start_time = time.perf_counter()
    
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
    # Agent names can be:
    # - "agent_0", "agent_1" (when agents_per_class == 1)
    # - "agent_0_0", "agent_0_1", "agent_1_0", "agent_1_1" (when agents_per_class > 1)
    agent_to_class = {}
    for agent_name in policies.keys():
        # Extract class from agent name
        if agent_name.startswith("agent_"):
            try:
                parts = agent_name.split("_")
                # For "agent_0" -> class 0
                # For "agent_0_1" -> class 0 (first number after "agent")
                if len(parts) >= 2 and parts[1].isdigit():
                    class_num = int(parts[1])
                    agent_to_class[agent_name] = class_num
            except (ValueError, IndexError):
                # If parsing fails, try to get from metadata if available
                # Check if we have metadata for this agent
                if agent_name in metadata_files:
                    try:
                        with open(metadata_files[agent_name], 'r') as f:
                            metadata = json.load(f)
                            if "target_class" in metadata and metadata["target_class"] is not None:
                                agent_to_class[agent_name] = int(metadata["target_class"])
                    except:
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
        # Instance-level metrics (per agent/instance)
        instance_precisions = []
        instance_coverages = []
        # Class-level metrics (union of all agents for the class)
        class_precisions = []
        class_coverages = []
        # Legacy lists (for backward compatibility)
        precisions = []
        coverages = []
        # Timing metrics
        rollout_times = []
        
        # Track time for this class
        class_start_time = time.perf_counter()
        
        for instance_idx in range(n_instances_per_class):
            # Create a single-agent environment for this specific class
            # This ensures we're running rollouts for the correct agent
            rollout_seed = seed + instance_idx if seed is not None else None
            
            # Create environment config for single agent (this class only)
            single_agent_config = anchor_config.copy()
            single_agent_config["target_classes"] = [target_class]  # Only this class
            
            # Create raw AnchorEnv directly (bypass BenchMARL/TorchRL wrapper)
            # Set mode to "inference" so termination counters are reset in reset()
            # mode must be inside env_config, not as a direct parameter
            if "env_config" not in single_agent_config:
                single_agent_config["env_config"] = {}
            single_agent_config["env_config"] = single_agent_config["env_config"].copy()
            single_agent_config["env_config"]["mode"] = "inference"
            env = AnchorEnv(**single_agent_config)
            
            # Get the actual agent name from the environment
            # Use possible_agents if available (set during __init__), otherwise fallback
            # Don't reset here - run_rollout_with_policy will reset with the proper seed
            actual_agent_name = None
            
            # First, try to use the agent_name from the outer loop if it exists in the environment
            # This is important when agents_per_class > 1 (e.g., agent_0_0, agent_0_1, etc.)
            if hasattr(env, 'possible_agents') and len(env.possible_agents) > 0:
                if agent_name in env.possible_agents:
                    actual_agent_name = agent_name
                else:
                    # Fallback to first agent if the specific agent_name doesn't exist
                    actual_agent_name = env.possible_agents[0]
            elif hasattr(env, 'agents') and len(env.agents) > 0:
                if agent_name in env.agents:
                    actual_agent_name = agent_name
                else:
                    actual_agent_name = env.agents[0]
            
            # Final fallback: construct agent name based on agents_per_class
            if actual_agent_name is None:
                # If we have agent_name from outer loop, use it (environment should match)
                if agent_name:
                    actual_agent_name = agent_name
                elif agents_per_class == 1:
                    actual_agent_name = f"agent_{target_class}"
                else:
                    # For agents_per_class > 1, default to first agent pattern
                    actual_agent_name = f"agent_{target_class}_0"
                logger.warning(f"  Could not determine agent name from environment, using {actual_agent_name}")
            
            # Run rollout with raw PettingZoo environment
            episode_data = run_rollout_with_policy(
                env=env,  # Raw AnchorEnv, not TorchRL-wrapped
                policy=policy,
                agent_id=actual_agent_name,  # Use actual agent name from environment
                max_steps=steps_per_episode,
                device=device,
                seed=rollout_seed,  # Pass seed for reproducible rollouts
                exploration_mode=exploration_mode,  # Pass exploration mode for diversity
                action_noise_scale=action_noise_scale,  # Pass action noise scale
                verbose_logging=verbose_logging,  # Pass verbosity setting
            )
            
            # Debug logging for first episode of each class
            if verbose_logging and instance_idx == 0:
                logger.info(f"  Debug class {target_class} episode 0:")
                logger.info(f"    actual_agent_name: {actual_agent_name}")
                logger.info(f"    episode_data keys: {list(episode_data.keys()) if episode_data else 'empty'}")
                if episode_data:
                    prec_val = episode_data.get('anchor_precision', 'missing')
                    cov_val = episode_data.get('anchor_coverage', 'missing')
                    logger.info(f"    anchor_precision: {prec_val}")
                    logger.info(f"    anchor_coverage: {cov_val}")
                    logger.info(f"    has final_observation: {'final_observation' in episode_data}")
                else:
                    logger.warning(f"    ⚠ episode_data is empty for class {target_class}!")
            
            # Initialize metrics variables
            instance_precision = 0.0
            instance_coverage = 0.0
            class_precision = 0.0
            class_coverage = 0.0
            precision = 0.0
            coverage = 0.0
            
            # Get rollout time from episode_data
            rollout_time = episode_data.get("rollout_time_seconds", 0.0)
            rollout_times.append(float(rollout_time))
            
            if episode_data:
                # Instance-level metrics
                instance_precision = episode_data.get("anchor_precision", episode_data.get("instance_precision", 0.0))
                instance_coverage = episode_data.get("anchor_coverage", episode_data.get("instance_coverage", 0.0))
                # Class-level metrics
                class_precision = episode_data.get("class_precision", 0.0)
                class_coverage = episode_data.get("class_coverage", 0.0)
                
                # Diagnostic: Log if metrics are zero (might indicate a problem)
                if instance_idx < 3 and (instance_precision == 0.0 or instance_coverage == 0.0):
                    logger.warning(
                        f"  ⚠ Episode {instance_idx} for class {target_class} has zero metrics: "
                        f"precision={instance_precision:.4f}, coverage={instance_coverage:.4f}. "
                        f"This might indicate the policy didn't find any valid boxes or the environment isn't set up correctly."
                    )
                
                instance_precisions.append(float(instance_precision))
                instance_coverages.append(float(instance_coverage))
                class_precisions.append(float(class_precision))
                class_coverages.append(float(class_coverage))
                
                # Legacy fields (for backward compatibility)
                precisions.append(float(instance_precision))
                coverages.append(float(instance_coverage))
                
                precision = float(instance_precision)
                coverage = float(instance_coverage)
            else:
                # Log warning if episode_data is empty
                if instance_idx == 0:
                    logger.warning(f"  ⚠ Warning: Empty episode_data for class {target_class}, episode {instance_idx}")
                instance_precisions.append(0.0)
                instance_coverages.append(0.0)
                class_precisions.append(0.0)
                class_coverages.append(0.0)
                precisions.append(0.0)
                coverages.append(0.0)
            
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
                        logger.warning(f"  X_min/X_range not available for denormalization. Using normalized bounds.")
                    
                    # Create temporary environment for rule extraction
                    # Set mode to "inference" so termination counters are reset in reset()
                    inference_env_config = {**env_config, "mode": "inference"}
                    temp_env = AnchorEnv(
                        X_unit=env_data["X_unit"],
                        X_std=env_data["X_std"],
                        y=env_data["y"],
                        feature_names=feature_names,
                        classifier=dataset_loader.get_classifier(),
                        device="cpu",
                        target_classes=[target_class],
                        env_config=inference_env_config
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
                        denormalize=True  # Denormalize to standardized feature space (mean=0, std=1)
                    )
            
            # Extract metrics for anchor_data (variables are already defined above)
            anchor_instance_precision = float(instance_precision)
            anchor_instance_coverage = float(instance_coverage)
            anchor_class_precision = float(class_precision)
            anchor_class_coverage = float(class_coverage)
            
            anchor_data = {
                "instance_idx": instance_idx,
                # Instance-level metrics
                "instance_precision": anchor_instance_precision,
                "instance_coverage": anchor_instance_coverage,
                # Class-level metrics
                "class_precision": anchor_class_precision,
                "class_coverage": anchor_class_coverage,
                "anchor_precision": float(anchor_instance_precision),
                "anchor_coverage": float(anchor_instance_coverage),
                "total_reward": float(episode_data.get("total_reward", 0.0)) if episode_data else 0.0,
                "rule": rule,
            }
            
            if lower is not None and upper is not None:
                anchor_data.update({
                    "lower_bounds": lower.tolist(),
                    "upper_bounds": upper.tolist(),
                    "box_widths": (upper - lower).tolist(),
                    "box_volume": float(np.prod(np.maximum(upper - lower, 1e-9))),
                    # Also save normalized bounds for reference
                    "lower_bounds_normalized": lower_normalized.tolist() if lower_normalized is not None else None,
                    "upper_bounds_normalized": upper_normalized.tolist() if upper_normalized is not None else None,
                })
            
            anchors_list.append(anchor_data)
            rules_list.append(rule)
        
        unique_rules = list(set([r for r in rules_list if r and r != "any values (no tightened features)"]))
        
        # End timing for this class
        class_end_time = time.perf_counter()
        class_total_time = class_end_time - class_start_time
        
        # Compute instance-level metrics (average across all instances)
        instance_precision = float(np.mean(instance_precisions)) if instance_precisions else 0.0
        instance_coverage = float(np.mean(instance_coverages)) if instance_coverages else 0.0
        avg_rollout_time = float(np.mean(rollout_times)) if rollout_times else 0.0
        total_rollout_time = float(np.sum(rollout_times)) if rollout_times else 0.0
        
        # Compute class-level metrics from union of all anchors across all episodes
        class_precision_union = 0.0
        class_coverage_union = 0.0
        
        # Get the appropriate dataset (test or train) based on eval_on_test_data
        if eval_on_test_data and env_data.get("X_test_unit") is not None:
            X_data = env_data["X_test_unit"]
            y_data = env_data["y_test"]
        else:
            X_data = env_data["X_unit"]
            y_data = env_data["y"]
        
        # Compute union of all anchors for this class
        if X_data is not None and y_data is not None and len(anchors_list) > 0:
            n_samples = X_data.shape[0]
            union_mask = np.zeros(n_samples, dtype=bool)
            
            # Build union mask from all anchors
            for anchor_data in anchors_list:
                if "lower_bounds_normalized" in anchor_data and "upper_bounds_normalized" in anchor_data:
                    lower = np.array(anchor_data["lower_bounds_normalized"], dtype=np.float32)
                    upper = np.array(anchor_data["upper_bounds_normalized"], dtype=np.float32)
                    
                    # Check which points fall in this anchor box
                    in_box = np.all((X_data >= lower) & (X_data <= upper), axis=1)
                    union_mask |= in_box
            
            # Class-level coverage: fraction of class samples that are in the union
            mask_cls = (y_data == target_class)
            if mask_cls.sum() > 0:
                class_coverage_union = float(union_mask[mask_cls].mean())
            else:
                class_coverage_union = 0.0
            
            # Class-level precision: fraction of points in union that belong to target class
            if union_mask.any():
                y_union = y_data[union_mask]
                class_precision_union = float((y_union == target_class).mean())
            else:
                class_precision_union = 0.0
        
        results["per_class_results"][class_key] = {
            "class": int(target_class),
            "group": agent_name,
            # Instance-level metrics (averaged across all instances)
            "instance_precision": instance_precision,
            "instance_coverage": instance_coverage,
            "instance_precision_std": float(np.std(instance_precisions)) if len(instance_precisions) > 1 else 0.0,
            "instance_coverage_std": float(np.std(instance_coverages)) if len(instance_coverages) > 1 else 0.0,
            # Class-level metrics (union of all anchors for this class across all episodes)
            "class_precision": class_precision_union,
            "class_coverage": class_coverage_union,
            # Keep averaged class-level metrics for backward compatibility (but note they're averaged, not union)
            "class_precision_avg": float(np.mean(class_precisions)) if class_precisions else 0.0,
            "class_coverage_avg": float(np.mean(class_coverages)) if class_coverages else 0.0,
            "class_precision_std": float(np.std(class_precisions)) if len(class_precisions) > 1 else 0.0,
            "class_coverage_std": float(np.std(class_coverages)) if len(class_coverages) > 1 else 0.0,
            # Anchor metrics (same as instance-level for single agent)
            "anchor_precision_mean": float(np.mean(precisions)) if precisions else 0.0,
            "anchor_coverage_mean": float(np.mean(coverages)) if coverages else 0.0,
            "anchor_precision_std": float(np.std(precisions)) if len(precisions) > 1 else 0.0,
            "anchor_coverage_std": float(np.std(coverages)) if len(coverages) > 1 else 0.0,
            "n_episodes": len(anchors_list),
            "rules": rules_list,
            "unique_rules": unique_rules,
            "unique_rules_count": len(unique_rules),
            "anchors": anchors_list,
            # Timing metrics
            "avg_rollout_time_seconds": avg_rollout_time,
            "total_rollout_time_seconds": total_rollout_time,
            "class_total_time_seconds": float(class_total_time),
        }
        
        logger.info(f"  Processed {len(anchors_list)} episodes")
        logger.info(f"  Instance-level - Average precision: {instance_precision:.4f}, coverage: {instance_coverage:.4f}")
        logger.info(f"  Class-level (Union) - Precision: {class_precision_union:.4f}, coverage: {class_coverage_union:.4f}")
        logger.info(f"  Unique rules: {len(unique_rules)}")
        logger.info(f"  Average rollout time per episode: {avg_rollout_time:.4f}s")
        logger.info(f"  Total rollout time for class: {total_rollout_time:.4f}s")
        logger.info(f"  Total class processing time: {class_total_time:.4f}s")
    
    # End overall timing
    overall_end_time = time.perf_counter()
    overall_total_time = overall_end_time - overall_start_time
    
    logger.info("\n" + "="*80)
    logger.info(f"Overall inference time: {overall_total_time:.4f}s")
    logger.info("="*80)
    
    # Calculate total rollout time across all classes
    total_rollout_time_all_classes = sum(
        result.get("total_rollout_time_seconds", 0.0)
        for result in results["per_class_results"].values()
    )
    
    # Add timing summary to metadata
    results["metadata"]["total_inference_time_seconds"] = float(overall_total_time)
    results["metadata"]["total_rollout_time_seconds"] = float(total_rollout_time_all_classes)
    
    # Save results if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to serializable format
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
        
        logger.info(f"Results saved to: {rules_filepath}")
    
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
    
    # Build dataset choices dynamically
    dataset_choices = ["breast_cancer", "wine", "iris", "synthetic", "moons", "circles", "covtype", "housing"]
    
    # Add UCIML datasets if available
    try:
        from ucimlrepo import fetch_ucirepo
        dataset_choices.extend([
            "uci_adult", "uci_car", "uci_credit", "uci_nursery", 
            "uci_mushroom", "uci_tic-tac-toe", "uci_vote", "uci_zoo"
        ])
    except ImportError:
        pass
    
    # Add Folktables datasets if available
    try:
        from folktables import ACSDataSource
        # Add common Folktables combinations
        states = ["CA", "NY", "TX", "FL", "IL"]
        years = ["2018", "2019", "2020"]
        tasks = ["income", "coverage", "mobility", "employment", "travel"]
        for task in tasks:
            for state in states[:2]:  # Limit to first 2 states to avoid too many choices
                for year in years[:1]:  # Limit to first year
                    dataset_choices.append(f"folktables_{task}_{state}_{year}")
    except ImportError:
        pass
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=dataset_choices,
        help="Dataset name (must match training). For UCIML: uci_<name_or_id>. For Folktables: folktables_<task>_<state>_<year>"
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
        default=-1,
        help="Maximum number of features to include in extracted rules (use -1 for all features)"
    )
    
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=500,
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
        "--exploration_mode",
        type=str,
        default="sample",
        choices=["sample", "mean", "noisy_mean"],
        help="Exploration mode for rule diversity: 'sample' (sample from policy distribution), 'mean' (deterministic), 'noisy_mean' (mean + noise)"
    )
    
    parser.add_argument(
        "--action_noise_scale",
        type=float,
        default=0.05,
        help="Scale of Gaussian noise to add to actions for exploration (0.0 = no noise, recommended: 0.05-0.1)"
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
        logger.warning("WARNING: Running inference on training data is not recommended!")
        logger.warning("  Inference should typically use test data to evaluate generalization.")
    else:
        logger.info("Using test data for inference (default behavior)")
    
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
        device=args.device,
        exploration_mode=args.exploration_mode,
        action_noise_scale=args.action_noise_scale,
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
