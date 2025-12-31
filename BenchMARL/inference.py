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
from pathlib import Path

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


def _load_nashconv_metrics(experiment_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load NashConv metrics from training_history.json and evaluation_history.json.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary with training and evaluation NashConv metrics
    """
    nashconv_data = {
        "training": None,
        "evaluation": None,
        "available": False
    }
    
    if not experiment_dir:
        return nashconv_data
    
    experiment_path = Path(experiment_dir)
    
    # Load training history
    training_history_path = experiment_path / "training_history.json"
    if training_history_path.exists():
        try:
            with open(training_history_path, 'r') as f:
                training_history = json.load(f)
            
            # Extract NashConv metrics from training history
            training_nashconv = []
            for entry in training_history:
                nashconv_entry = {}
                for key, value in entry.items():
                    if key.startswith("training/nashconv") or key.startswith("training/exploitability"):
                        nashconv_entry[key] = value
                if nashconv_entry:
                    nashconv_entry["step"] = entry.get("step")
                    nashconv_entry["total_frames"] = entry.get("total_frames")
                    training_nashconv.append(nashconv_entry)
            
            if training_nashconv:
                nashconv_data["training"] = training_nashconv
                nashconv_data["available"] = True
        except Exception as e:
            logger.debug(f"Could not load training NashConv metrics: {e}")
    
    # Load evaluation history
    evaluation_history_path = experiment_path / "evaluation_history.json"
    if evaluation_history_path.exists():
        try:
            with open(evaluation_history_path, 'r') as f:
                evaluation_history = json.load(f)
            
            # Extract NashConv metrics from evaluation history
            evaluation_nashconv = []
            for entry in evaluation_history:
                nashconv_entry = {}
                for key, value in entry.items():
                    if key.startswith("evaluation/nashconv") or key.startswith("evaluation/exploitability"):
                        nashconv_entry[key] = value
                if nashconv_entry:
                    nashconv_entry["step"] = entry.get("step")
                    nashconv_entry["total_frames"] = entry.get("total_frames")
                    evaluation_nashconv.append(nashconv_entry)
            
            if evaluation_nashconv:
                nashconv_data["evaluation"] = evaluation_nashconv
                nashconv_data["available"] = True
        except Exception as e:
            logger.debug(f"Could not load evaluation NashConv metrics: {e}")
    
    return nashconv_data


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
    max_steps: Optional[int] = None,  # If None, will read from env_config.max_cycles
    device: str = "cpu",
    seed: Optional[int] = None,
    exploration_mode: str = "sample",  # "sample", "mean", or "noisy_mean"
    action_noise_scale: float = 0.05,  # Noise scale for actions (0.0 = no noise)
    verbose_logging: bool = False,  # Enable verbose debug logging
) -> Dict[str, Any]:
    # Ensure policy is in eval mode
    policy.to(device)
    policy.eval()
    
    # Read max_steps from env if not provided
    if max_steps is None:
        max_steps = env.max_cycles
    
    # Set mode to "inference" so termination counters are reset in reset()
    if hasattr(env, 'mode'):
        env.mode = "inference"
    
    # Verify x_star_unit is set before reset
    # Check if this is a class-based rollout (expected to not have x_star_unit)
    is_class_based = hasattr(env, 'use_class_centroids') and env.use_class_centroids
    
    if agent_id in env.x_star_unit:
        x_star = env.x_star_unit[agent_id]
        if verbose_logging:
            logger.debug(f"  x_star_unit set for {agent_id}: shape={x_star.shape}, mean={x_star.mean():.4f}, range=[{x_star.min():.4f}, {x_star.max():.4f}]")
    else:
        # Only warn if this is supposed to be an instance-based rollout
        if not is_class_based:
            logger.warning(f"  x_star_unit NOT set for {agent_id} before reset! This will cause class-based mode instead of instance-based.")
        elif verbose_logging:
            logger.debug(f"  x_star_unit not set for {agent_id} (expected for class-based rollout using centroids)")
    
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
    
    # Debug: Check initial box state and verify x_star_unit is still set after reset
    if hasattr(env, 'lower') and hasattr(env, 'upper') and agent_id in env.lower:
        lower_init = env.lower[agent_id]
        upper_init = env.upper[agent_id]
        widths_init = upper_init - lower_init
        
        # Check if x_star_unit is still set (reset should preserve it)
        has_x_star = agent_id in env.x_star_unit and env.x_star_unit[agent_id] is not None
        # Only warn if this was supposed to be an instance-based rollout
        if not has_x_star and not is_class_based:
            logger.warning(f"  x_star_unit was lost after reset for {agent_id}! Coverage calculation will be wrong.")
        elif not has_x_star and is_class_based and verbose_logging:
            logger.debug(f"  x_star_unit not set for {agent_id} (expected for class-based rollout)")
        
        precision_init, coverage_init, details_init = env._current_metrics(agent_id)
        
        if verbose_logging:
            logger.info(f"  Initial box state for {agent_id}:")
            logger.info(f"    x_star_unit set: {has_x_star}")
            logger.info(f"    Box center: {((lower_init + upper_init) / 2).mean():.4f} (mean across features)")
            logger.info(f"    Box widths: min={widths_init.min():.4f}, max={widths_init.max():.4f}, mean={widths_init.mean():.4f}")
            logger.info(f"    Initial precision: {precision_init:.4f}, coverage: {coverage_init:.4f}")
            logger.info(f"    Initial n_points: {details_init.get('n_points', 'N/A')}")
        
        # CRITICAL: Log warning if coverage is 0 but box seems reasonable
        if coverage_init == 0.0 and widths_init.mean() > 0.01:
            logger.warning(
                f"  ⚠ Initial coverage is 0 but box width is reasonable (mean={widths_init.mean():.4f}). "
                f"This suggests the box might not contain any data points, or x_star_unit is not set correctly. "
                f"x_star_unit set: {has_x_star}"
            )
    
    done = False
    step_count = 0
    total_reward = 0.0
    last_info_for_agent: Dict[str, Any] = {}
    last_termination = False
    last_truncation = False
    
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
        # Track last infos / done reasons for this agent
        if isinstance(infos_dict, dict) and agent_id in infos_dict and isinstance(infos_dict[agent_id], dict):
            last_info_for_agent = infos_dict[agent_id]
        last_termination = bool(terminations_dict.get(agent_id, False))
        last_truncation = bool(truncations_dict.get(agent_id, False))
        
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
                    # This could indicate actions aren't being applied, but could also be very small actions
                    # Only warn if verbose logging is enabled, otherwise it's too noisy
                    if verbose_logging:
                        logger.warning(f"  Box did not change after first action (diff < 1e-6). Actions may be very small or not applied correctly.")
                    # Note: Very small actions are common, especially early in training or with certain policies
        
        step_count += 1
    
    # End timing the rollout
    rollout_end_time = time.perf_counter()
    rollout_duration = rollout_end_time - rollout_start_time
    
    # Extract final metrics directly from environment
    episode_data = {}
    
    try:
        # Get instance-level metrics directly from environment
        instance_precision, instance_coverage, details = env._current_metrics(agent_id)
        
        # For instance-based anchors, also compute class-conditional coverage for better interpretability
        # This helps understand coverage relative to the target class, not just overall dataset
        instance_coverage_class_conditional = 0.0
        if env.x_star_unit.get(agent_id) is not None:  # Instance-based mode
            target_class = env._get_class_for_agent(agent_id)
            if target_class is not None:
                # Get the mask and class labels
                if env.eval_on_test_data:
                    y_data = env.y_test
                else:
                    y_data = env.y
                mask = env._mask_in_box(agent_id)
                if len(mask) == len(y_data):
                    class_mask = (y_data == target_class)
                    n_class_samples = class_mask.sum()
                    if n_class_samples > 0:
                        n_class_in_box = (mask & class_mask).sum()
                        instance_coverage_class_conditional = float(n_class_in_box / n_class_samples)
        
        # Get class-level metrics (union of all agents for this class)
        target_class = env._get_class_for_agent(agent_id)
        class_union_metrics = {}
        class_precision = 0.0
        class_coverage = 0.0
        
        if target_class is not None:
            # Compute class-level union metrics
            # FIX: When only one agent acts during inference, class-level metrics should only consider that agent's box
            # This avoids issues where other agents' boxes (initialized but never updated) affect the union
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
            
            # FIX: If class-level metrics are 0 but instance-level are > 0, it might be because
            # other agents' boxes (that were never updated) are included in the union.
            # For single-agent rollouts, class-level metrics should match instance-level.
            if class_precision == 0.0 and class_coverage == 0.0 and instance_precision > 0.0:
                # During single-agent inference, class-level should equal instance-level
                # This happens when only one agent exists or only one agent's box is valid
                if verbose_logging:
                    logger.debug(f"  Class-level metrics are 0 but instance-level are > 0. Using instance-level as fallback for single-agent rollout.")
                # For single-agent rollouts, class-level = instance-level
                class_precision = instance_precision
                class_coverage = instance_coverage
        
        # Get final box bounds
        lower = env.lower[agent_id]
        upper = env.upper[agent_id]
        
        # Construct final observation (using instance-level metrics)
        final_obs = np.concatenate([lower, upper, np.array([instance_precision, instance_coverage], dtype=np.float32)])
        
        episode_data = {
            # Instance-level metrics (for this specific agent/instance)
            "instance_precision": float(instance_precision),
            "instance_coverage": float(instance_coverage),  # Overall coverage P(x in box)
            "instance_coverage_class_conditional": float(instance_coverage_class_conditional),  # Class-conditional coverage P(x in box | y = target_class)
            # Anchor metrics (same as instance-level for single agent)
            "anchor_precision": float(instance_precision),
            "anchor_coverage": float(instance_coverage),
            "anchor_coverage_class_conditional": float(instance_coverage_class_conditional),
            # Class-level metrics (union of all agents for this class)
            "class_precision": float(class_precision),
            "class_coverage": float(class_coverage),
            "total_reward": total_reward,
            "n_steps": step_count,
            "rollout_time_seconds": float(rollout_duration),
            "final_observation": final_obs.tolist(),
            # Termination diagnostics
            "terminated": float(1.0 if last_termination else 0.0),
            "truncated": float(1.0 if last_truncation else 0.0),
            "termination_reason_str": str(last_info_for_agent.get("termination_reason_str", "")),
            "stabilized": float(last_info_for_agent.get("stabilized", 0.0)),
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
            "terminated": float(1.0 if last_termination else 0.0),
            "truncated": float(1.0 if last_truncation else 0.0),
            "termination_reason_str": str(last_info_for_agent.get("termination_reason_str", "")),
            "stabilized": float(last_info_for_agent.get("stabilized", 0.0)),
        }
    
    return episode_data


def extract_rules_from_policies(
    experiment_dir: str,
    dataset_name: str,
    mlp_config_path: str = "conf/mlp.yaml",
    max_features_in_rule: int = -1,
    steps_per_episode: Optional[int] = None,
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
        
        # Try to load seed from policies_index.json (to match training seed)
        training_seed = index_data.get("seed")
        if training_seed is not None:
            logger.info(f"Found training seed in policies_index.json: {training_seed}")
            # Use training seed instead of provided seed to ensure reproducibility
            if seed != training_seed:
                logger.info(f"Using training seed ({training_seed}) instead of provided seed ({seed}) to ensure reproducibility")
            seed = training_seed
        else:
            logger.info(f"Using provided seed: {seed} (training seed not found in policies_index.json)")
        
        agents_per_class = index_data.get("agents_per_class", 1)
        policies_by_class = index_data.get("policies_by_class", {})
        
        # IMPORTANT:
        # - anchor_trainer.extract_and_save_individual_models() saves one file per *agent*
        #   (e.g., policy_agent_0_0.pth, policy_agent_0_1.pth, ...), along with metadata that
        #   includes both the group (agent_0) and the specific agent name (agent_0_0, etc.).
        # - The original loader keyed policy_files by "group", which caused later entries for
        #   the same group to overwrite earlier ones (only the last agent per class survived).
        # - Here we key by the *agent* name so that all per-agent policies are visible, while
        #   still retaining backward compatibility if "agent" is missing.
        
        for class_key, class_info in policies_by_class.items():
            class_policies = class_info.get("policies", [])
            for policy_info in class_policies:
                agent_name = policy_info.get("agent")
                group = policy_info.get("group") or agent_name
                policy_file = policy_info.get("policy_file")
                metadata_file = policy_info.get("metadata_file")
                
                # Use agent_name as the primary key when available, otherwise fall back to group.
                key = agent_name or group
                
                if key and policy_file:
                    # Resolve relative paths
                    policy_path = os.path.join(individual_models_dir, policy_file)
                    if os.path.exists(policy_path):
                        policy_files[key] = policy_path
                        
                        if metadata_file:
                            metadata_path = os.path.join(individual_models_dir, metadata_file)
                            if os.path.exists(metadata_path):
                                metadata_files[key] = metadata_path
        
        logger.info(f"  Loaded {len(policy_files)} policies from index (agents_per_class={agents_per_class})")
    else:
        # Fallback: Search for policies in flat structure - older version
        logger.info("No policies_index.json found, searching for policies in flat structure...")
        logger.info(f"Using provided seed: {seed} (training seed not available - policies_index.json not found)")
        
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
    # Prefer YAML-based env config so inference matches training settings.
    try:
        env_config = trainer._load_env_config_from_yaml()
    except Exception:
        env_config = trainer._get_default_env_config()

    # Support YAML layouts that include a nested `env_config:` section.
    if isinstance(env_config, dict) and isinstance(env_config.get("env_config", None), dict):
        nested = env_config.get("env_config", {})
        top = {k: v for k, v in env_config.items() if k != "env_config"}
        env_config = {**nested, **top}

    # Resolve rollout length: if not explicitly provided, use env_config.max_cycles.
    if steps_per_episode is None:
        steps_per_episode = int(env_config.get("max_cycles"))
        if steps_per_episode is None:
            raise ValueError("max_cycles must be specified in env_config. Check your YAML config file.")
    
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
    # Expected agent names depend on agents_per_class:
    # - If agents_per_class == 1: agent_0, agent_1, ..., agent_{n_classes-1}
    # - If agents_per_class > 1: agent_0_0, agent_0_1, ..., agent_{class}_{agent_idx}
    # NOTE: This check is done BEFORE the mapping section below, so we skip it if agents_per_class > 1
    # because the mapping section (lines 1300+) will handle individual agent policies correctly
    if agents_per_class == 1:
        expected_agent_names = [f"agent_{cls}" for cls in target_classes]
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
    else:
        # When agents_per_class > 1, skip the old naming convention check
        # The mapping section below will handle individual agent policies correctly
        logger.debug(f"  Skipping old naming convention check (agents_per_class={agents_per_class} > 1)")
    
    # Set min_coverage_floor to ensure box always covers at least the anchor instance
    # Use 1/n_samples from the dataset (to ensure at least one point is covered), 
    # or fall back to config default (0.005) if dataset size unavailable
    if eval_on_test_data and env_data.get("X_test_unit") is not None:
        n_samples = env_data["X_test_unit"].shape[0]
    elif env_data.get("X_unit") is not None:
        n_samples = env_data["X_unit"].shape[0]
    else:
        n_samples = None
    
    config_default = env_config.get("min_coverage_floor", 0.005)
    
    if n_samples is not None and n_samples > 0:
        # Use 1/n_samples to ensure at least one point is covered (the anchor instance)
        min_coverage_floor = 1.0 / n_samples
        # Ensure it's not smaller than a reasonable minimum (use config default as lower bound)
        # This prevents extremely small values for very large datasets
        min_coverage_floor = max(min_coverage_floor, config_default)
    else:
        # Fall back to config default if dataset size unavailable
        min_coverage_floor = config_default
    
    # Ensure it's non-zero
    min_coverage_floor = max(min_coverage_floor, 1e-6)
    
    env_config["min_coverage_floor"] = min_coverage_floor
    logger.info(f"  Set min_coverage_floor={min_coverage_floor:.6f} (n_samples={n_samples if n_samples is not None else 'unknown'}, ensures box covers at least anchor instance)")
    
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
    
    # Log perturbation settings to verify they're loaded correctly
    logger.info(f"\nPerturbation settings loaded from config:")
    logger.info(f"  use_perturbation: {env_config.get('use_perturbation', 'NOT SET')}")
    logger.info(f"  perturbation_mode: {env_config.get('perturbation_mode', 'NOT SET')}")
    logger.info(f"  n_perturb: {env_config.get('n_perturb', 'NOT SET')}")
    if env_config.get('use_perturbation') and env_config.get('perturbation_mode') == 'adaptive':
        min_points_threshold = max(1, int(0.1 * env_config.get('n_perturb', 4096)))
        logger.info(f"  (Adaptive mode will use uniform sampling if covered points < {min_points_threshold})")
    
    # Respect the perturbation_mode from config (no longer overriding for inference)
    # Users can set perturbation_mode in config to control behavior during inference
    original_perturbation_mode = env_config.get('perturbation_mode')
    original_use_perturbation = env_config.get('use_perturbation')
    
    if original_use_perturbation:
        logger.info(f"\n✓ Using perturbation mode for inference: {original_perturbation_mode}")
        if original_perturbation_mode == 'adaptive':
            min_points_threshold = max(1, int(0.1 * env_config.get('n_perturb', 4096)))
            logger.info(f"  (Adaptive mode will use uniform sampling if covered points < {min_points_threshold})")
    else:
        logger.info(f"\n✓ Perturbation disabled for inference (use_perturbation={original_use_perturbation})")
    
    # For class-level inference, compute cluster centroids per class
    logger.info("\nComputing cluster centroids per class for class-level inference...")
    try:
        from utils.clusters import compute_cluster_centroids_per_class
        
        # Use full dataset (train + test) for clustering when test data is available
        # This provides better cluster representation and stability
        # Clustering is initialization, not training, so using test data is acceptable
        if eval_on_test_data and env_data.get("X_test_unit") is not None:
            # Combine train and test data for clustering
            X_cluster = np.vstack([env_data["X_unit"], env_data["X_test_unit"]])
            y_cluster = np.concatenate([env_data["y"], env_data["y_test"]])
            logger.info("  Using FULL dataset (train + test) for clustering")
            logger.info(f"    Training samples: {len(env_data['X_unit'])}, Test samples: {len(env_data['X_test_unit'])}, Total: {len(X_cluster)}")
        else:
            # Fallback to training data only if test data not available
            X_cluster = env_data["X_unit"]
            y_cluster = env_data["y"]
            logger.info("  Using TRAINING data for clustering (test data not available)")
        
        if agents_per_class > 1:
            n_clusters_per_class = agents_per_class * 10
            logger.info(f"  Using {n_clusters_per_class} clusters per class ({n_clusters_per_class // agents_per_class} per agent) for k-means clustering")
        else:
            n_clusters_per_class = min(10, n_instances_per_class)
            logger.info(f"  Using {n_clusters_per_class} clusters per class for diversity (matching training and single-agent behavior)")
        
        cluster_centroids_per_class = compute_cluster_centroids_per_class(
            X_unit=X_cluster,
            y=y_cluster,
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
    # Use test data if eval_on_test_data=True, otherwise use training data
    if eval_on_test_data and env_data.get("X_test_unit") is not None:
        anchor_X_unit = env_data["X_test_unit"]
        anchor_X_std = env_data["X_test_std"]
        anchor_y = env_data["y_test"]
    else:
        anchor_X_unit = env_data["X_unit"]
        anchor_X_std = env_data["X_std"]
        anchor_y = env_data["y"]
    
    # Ensure normalize_data=False to avoid double normalization
    # get_anchor_env_data() already returns X_unit (normalized to [0,1]) and X_std (standardized)
    # AnchorEnv.__init__ should use these directly without re-normalizing
    env_config["normalize_data"] = False
    
    anchor_config = {
        "X_unit": anchor_X_unit,
        "X_std": anchor_X_std,
        "y": anchor_y,
        "feature_names": feature_names,
        "classifier": dataset_loader.get_classifier(),
        "device": device,
        "target_classes": target_classes,
        "env_config": env_config,
    }
    
    # If no agent-specific policies were extracted, or if we only found some agents,
    # map policies to agents properly
    # IMPORTANT: When agents_per_class > 1, we should use individual policies for each agent,
    # not map the same policy to all agents of a class
    if not agent_policies or len(agent_policies) < len(target_classes) * agents_per_class:
        logger.info(f"  Mapping policies to agents (agents_per_class={agents_per_class})...")
        
        # First, try to map policies directly by agent name from policy_files
        # The policy_files dict should already have agent names as keys (e.g., "agent_0_0", "agent_0_1")
        for agent_name in sorted(policy_files.keys()):
            if agent_name not in agent_policies and agent_name in policies:
                agent_policies[agent_name] = policies[agent_name]
                logger.info(f"  Mapped policy for {agent_name}")
        
        # If we still have missing agents, try to map from group policies
        # This handles cases where policies are stored by group name instead of agent name
        for group in sorted(policy_files.keys()):
            if group not in agent_policies:
                combined_policy = policies.get(group)
                if combined_policy is None:
                    continue
                
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
                            # If group name has format "agent_0_1", extract both class and agent index
                            if len(parts) >= 3 and parts[2].isdigit():
                                # This is already an agent name, use it directly
                                agent_name = group
                                if agent_name not in agent_policies:
                                    agent_policies[agent_name] = combined_policy
                                    logger.info(f"  Mapped policy for {agent_name} (from group)")
                                    continue
                    except:
                        pass
                
                # Map policy to agents based on agents_per_class
                if target_class is not None:
                    if agents_per_class > 1:
                        # Try to find which specific agent this policy belongs to
                        # Check if we can determine agent index from group name or metadata
                        agent_idx = None
                        if group.startswith("agent_"):
                            parts = group.split("_")
                            if len(parts) >= 3 and parts[2].isdigit():
                                agent_idx = int(parts[2])
                        
                        if agent_idx is not None:
                            # This is a specific agent's policy
                            agent_name = f"agent_{target_class}_{agent_idx}"
                            if agent_name not in agent_policies:
                                agent_policies[agent_name] = combined_policy
                                logger.info(f"  Mapped individual policy for {agent_name} (class {target_class}, agent {agent_idx})")
                        else:
                            # Can't determine specific agent, map to all agents of this class as fallback
                            # (This should be rare - individual policies should have agent indices in their names)
                            for k in range(agents_per_class):
                                agent_name = f"agent_{target_class}_{k}"
                                if agent_name not in agent_policies:
                                    agent_policies[agent_name] = combined_policy
                                    logger.info(f"  Mapped shared policy for {agent_name} (class {target_class}, fallback)")
                    else:
                        # agents_per_class == 1, use simple agent name
                        agent_name = f"agent_{target_class}"
                        if agent_name not in agent_policies:
                            agent_policies[agent_name] = combined_policy
                            logger.info(f"  Mapped policy for {agent_name}")
                else:
                    # Fallback: map to all classes if we can't determine class
                    for cls in target_classes:
                        if agents_per_class > 1:
                            for k in range(agents_per_class):
                                agent_name = f"agent_{cls}_{k}"
                                if agent_name not in agent_policies:
                                    agent_policies[agent_name] = combined_policy
                                    logger.info(f"  Mapped shared policy for {agent_name} (fallback)")
                        else:
                            agent_name = f"agent_{cls}"
                            if agent_name not in agent_policies:
                                agent_policies[agent_name] = combined_policy
                                logger.info(f"  Mapped shared policy for {agent_name} (fallback)")
    
    # Use agent_policies if extracted, otherwise fall back to group policies
    if agent_policies:
        # CRITICAL FIX: Remove old naming convention agents (agent_0, agent_1) when agents_per_class > 1
        # These are fallback mappings that shouldn't be present when we have individual agent policies
        if agents_per_class > 1:
            old_naming_agents = [f"agent_{cls}" for cls in target_classes]
            agents_to_remove = [name for name in old_naming_agents if name in agent_policies]
            # Only remove if we have the corresponding individual agents (e.g., agent_0_0, agent_0_1, etc.)
            for old_name in agents_to_remove:
                # Extract class from old name
                class_num = int(old_name.split("_")[1])
                # Check if we have individual agents for this class
                has_individual_agents = any(
                    name.startswith(f"agent_{class_num}_") for name in agent_policies.keys()
                )
                if has_individual_agents:
                    del agent_policies[old_name]
                    logger.info(f"  Removed old naming convention agent: {old_name} (have individual agents for class {class_num})")
        
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
        
        # CRITICAL FIX: Sample instances from the target class data (instance-based rollouts)
        # This matches training's extract_rules behavior where x_star_unit is set per instance
        # Determine which dataset to sample from (test or training)
        if eval_on_test_data:
            class_mask = (anchor_config["y"] == target_class)
            class_instances = np.where(class_mask)[0]
            X_data_unit = anchor_config["X_unit"]
            data_source_name = "test"
        else:
            # If not using test data, use training data (should not happen in normal inference)
            if env_data.get("X_unit") is not None:
                class_mask = (env_data["y"] == target_class)
                class_instances = np.where(class_mask)[0]
                X_data_unit = env_data["X_unit"]
            else:
                # Fallback: use anchor_config data
                class_mask = (anchor_config["y"] == target_class)
                class_instances = np.where(class_mask)[0]
                X_data_unit = anchor_config["X_unit"]
            data_source_name = "training"
        
        if len(class_instances) == 0:
            logger.warning(f"  No instances found for class {target_class} in {data_source_name} data, skipping...")
            continue
        
        # Sample instances for THIS AGENT (different agents should get different instances)
        # When agents_per_class > 1, each agent should process different instances to ensure diversity
        n_samples = min(n_instances_per_class, len(class_instances))
        
        # Extract agent index from agent_name (e.g., "agent_0_1" -> agent_idx=1)
        agent_idx = 0
        if agents_per_class > 1 and "_" in agent_name:
            parts = agent_name.split("_")
            if len(parts) >= 3 and parts[2].isdigit():
                agent_idx = int(parts[2])
        
        # Use deterministic sampling with agent-specific seed to ensure different agents get different instances
        # This ensures reproducibility while maintaining diversity across agents
        agent_specific_seed = (seed if seed is not None else 42) + agent_idx * 1000
        rng_for_sampling = np.random.default_rng(agent_specific_seed)
        
        # Sample instances for this agent
        # If we have enough instances, each agent gets a different subset
        # Otherwise, all agents share the same instances (but this should be rare)
        if len(class_instances) >= n_samples * agents_per_class:
            # Enough instances: each agent gets its own subset
            instances_per_agent = len(class_instances) // agents_per_class
            start_idx = agent_idx * instances_per_agent
            end_idx = start_idx + instances_per_agent
            agent_class_instances = class_instances[start_idx:end_idx]
            sampled_indices = rng_for_sampling.choice(agent_class_instances, size=min(n_samples, len(agent_class_instances)), replace=False)
            logger.info(f"  Agent {agent_name} (idx={agent_idx}): Sampling {len(sampled_indices)} instances from subset [{start_idx}:{end_idx}] of {data_source_name} data for class {target_class}")
        else:
            # Not enough instances: all agents share (but use different seed for randomization)
            sampled_indices = rng_for_sampling.choice(class_instances, size=n_samples, replace=False)
            logger.info(f"  Agent {agent_name} (idx={agent_idx}): Sampling {n_samples} instances from {data_source_name} data for class {target_class} (shared pool - not enough instances for separate subsets)")
        
        for instance_idx_in_range, data_instance_idx in enumerate(sampled_indices):
            # Get the actual instance from the dataset
            x_instance = X_data_unit[data_instance_idx]
            
            # Create environment for this specific class and agent
            # IMPORTANT: When agents_per_class > 1, we create an environment with all agents for the class
            # but only run rollouts for the specific agent (agent_name)
            rollout_seed = seed + instance_idx_in_range if seed is not None else None
            
            # Create environment config for this class
            single_agent_config = anchor_config.copy()
            single_agent_config["target_classes"] = [target_class]  # Only this class
            
            # Create raw AnchorEnv directly (bypass BenchMARL/TorchRL wrapper)
            # Set mode to "inference" so termination counters are reset in reset()
            # mode must be inside env_config, not as a direct parameter
            if "env_config" not in single_agent_config:
                single_agent_config["env_config"] = {}
            single_agent_config["env_config"] = single_agent_config["env_config"].copy()
            single_agent_config["env_config"]["mode"] = "inference"
            # Ensure normalize_data=False to avoid double normalization (data is already normalized from get_anchor_env_data)
            single_agent_config["env_config"]["normalize_data"] = False
            
            # IMPORTANT: Ensure agents_per_class matches the training configuration
            # This allows the environment to create the correct agent names (agent_0_0, agent_0_1, etc.)
            # The env_config should already have agents_per_class from the config file, but we ensure it's set
            if "agents_per_class" not in single_agent_config["env_config"]:
                single_agent_config["env_config"]["agents_per_class"] = agents_per_class
            
            # CRITICAL FIX: Ensure X_min and X_range are set in env_config for proper normalization
            # These are needed for _unit_to_std conversion and must match training normalization
            if "X_min" not in single_agent_config["env_config"]:
                single_agent_config["env_config"]["X_min"] = env_config.get("X_min")
            if "X_range" not in single_agent_config["env_config"]:
                single_agent_config["env_config"]["X_range"] = env_config.get("X_range")
            
            env = AnchorEnv(**single_agent_config)
            
            # Get the actual agent name from the environment
            # Use possible_agents if available (set during __init__), otherwise fallback
            # Don't reset here - run_rollout_with_policy will reset with the proper seed
            actual_agent_name = None
            
            # First, try to use the agent_name from the outer loop if it exists in the environment
            # This is important when agents_per_class > 1 (e.g., agent_0_0, agent_0_1, etc.)
            # The environment should have been created with agents_per_class, so it should have the correct agent names
            if hasattr(env, 'possible_agents') and len(env.possible_agents) > 0:
                if agent_name in env.possible_agents:
                    actual_agent_name = agent_name
                    if verbose_logging:
                        logger.info(f"  Found agent {agent_name} in environment possible_agents")
                else:
                    # If the specific agent doesn't exist, log available agents for debugging
                    if verbose_logging:
                        logger.warning(
                            f"  Agent {agent_name} not found in environment. "
                            f"Available agents: {env.possible_agents}. "
                            f"Using first available agent."
                        )
                    # Fallback to first agent if the specific agent_name doesn't exist
                    actual_agent_name = env.possible_agents[0]
            elif hasattr(env, 'agents') and len(env.agents) > 0:
                if agent_name in env.agents:
                    actual_agent_name = agent_name
                else:
                    if verbose_logging:
                        logger.warning(
                            f"  Agent {agent_name} not found in environment.agents. "
                            f"Available: {env.agents}. Using first available."
                        )
                    actual_agent_name = env.agents[0]
            
            # Final fallback: construct agent name based on agents_per_class
            if actual_agent_name is None:
                # If we have agent_name from outer loop, use it (environment should match)
                if agent_name:
                    actual_agent_name = agent_name
                    if verbose_logging:
                        logger.info(f"  Using agent_name from outer loop: {actual_agent_name}")
                elif agents_per_class == 1:
                    actual_agent_name = f"agent_{target_class}"
                else:
                    # For agents_per_class > 1, try to extract agent index from agent_name
                    # e.g., "agent_0_1" -> use "agent_0_1", not "agent_0_0"
                    if agent_name and "_" in agent_name:
                        parts = agent_name.split("_")
                        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                            # agent_name is already in correct format (agent_0_1)
                            actual_agent_name = agent_name
                        else:
                            # Default to first agent pattern
                            actual_agent_name = f"agent_{target_class}_0"
                    else:
                        actual_agent_name = f"agent_{target_class}_0"
                logger.warning(f"  Could not determine agent name from environment, using {actual_agent_name}")
            
            # CRITICAL FIX: Set x_star_unit for instance-based rollout (matches training behavior)
            # This must be set BEFORE reset() is called (which happens inside run_rollout_with_policy)
            # Setting x_star_unit makes the rollout instance-based, using overall coverage P(x in box)
            # instead of class-conditional coverage P(x in box | y = target_class)
            # The environment initializes x_star_unit as a dict in __init__, so we can directly assign
            # IMPORTANT: Make a copy to avoid reference issues
            env.x_star_unit[actual_agent_name] = x_instance.copy()
            
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
            if verbose_logging and instance_idx_in_range == 0:
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
            instance_coverage_class_conditional = 0.0
            class_precision = 0.0
            class_coverage = 0.0
            precision = 0.0
            coverage = 0.0
            
            if episode_data:
                # Get rollout time from episode_data
                rollout_time = episode_data.get("rollout_time_seconds", 0.0)
                rollout_times.append(float(rollout_time))
                
                # Instance-level metrics
                instance_precision = episode_data.get("anchor_precision", episode_data.get("instance_precision", 0.0))
                instance_coverage = episode_data.get("anchor_coverage", episode_data.get("instance_coverage", 0.0))
                # Class-level metrics
                class_precision = episode_data.get("class_precision", 0.0)
                class_coverage = episode_data.get("class_coverage", 0.0)
                
                # Get class-conditional coverage if available
                instance_coverage_class_conditional = episode_data.get("instance_coverage_class_conditional", episode_data.get("anchor_coverage_class_conditional", 0.0))
                
                # Diagnostic: Log if metrics are zero (might indicate a problem)
                if instance_idx_in_range < 3 and (instance_precision == 0.0 or instance_coverage == 0.0):
                    logger.warning(
                        f"  ⚠ Episode {instance_idx_in_range} for class {target_class} has zero metrics: "
                        f"precision={instance_precision:.4f}, coverage={instance_coverage:.4f} "
                        f"(class-conditional: {instance_coverage_class_conditional:.4f}). "
                        f"This might indicate the policy didn't find any valid boxes or the environment isn't set up correctly."
                    )
                
                # Log coverage diagnostics if overall coverage is low but class-conditional is reasonable
                if instance_coverage < 0.05 and instance_coverage_class_conditional > 0.1:
                    logger.info(
                        f"  Episode {instance_idx_in_range}: Low overall coverage ({instance_coverage:.4f}) but "
                        f"reasonable class-conditional coverage ({instance_coverage_class_conditional:.4f}). "
                        f"This is expected if the dataset has many samples from other classes."
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
                if instance_idx_in_range == 0:
                    logger.warning(f"  ⚠ Warning: Empty episode_data for class {target_class}, episode {instance_idx_in_range}")
                rollout_times.append(0.0)
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
                    # Ensure normalize_data=False to avoid double normalization
                    inference_env_config = {**env_config, "mode": "inference", "normalize_data": False}
                    
                    # Use test data if eval_on_test_data=True, otherwise use training data
                    # This ensures the coordinate space matches the rollout environment
                    if eval_on_test_data and env_data.get("X_test_unit") is not None:
                        temp_X_unit = env_data["X_test_unit"]
                        temp_X_std = env_data["X_test_std"]
                        temp_y = env_data["y_test"]
                    else:
                        temp_X_unit = env_data["X_unit"]
                        temp_X_std = env_data["X_std"]
                        temp_y = env_data["y"]
                    
                    temp_env = AnchorEnv(
                        X_unit=temp_X_unit,
                        X_std=temp_X_std,
                        y=temp_y,
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
                    
                    # Compute initial bounds from x_instance and initial_window to get correct reference
                    # This matches how the environment initializes bounds in reset()
                    initial_window = env_config.get("initial_window", 0.1)
                    initial_lower_normalized = np.clip(x_instance - initial_window, 0.0, 1.0)
                    initial_upper_normalized = np.clip(x_instance + initial_window, 0.0, 1.0)
                    
                    # Extract rule with denormalization enabled and correct initial bounds
                    rule = temp_env.extract_rule(
                        temp_agent_name,
                        max_features_in_rule=max_features_in_rule,
                        initial_lower=initial_lower_normalized,
                        initial_upper=initial_upper_normalized,
                        denormalize=True  # Denormalize to standardized feature space (mean=0, std=1)
                    )
            
            # Extract metrics for anchor_data (variables are already defined above)
            anchor_instance_precision = float(instance_precision)
            anchor_instance_coverage = float(instance_coverage)
            anchor_instance_coverage_class_conditional = float(instance_coverage_class_conditional) if 'instance_coverage_class_conditional' in locals() else 0.0
            anchor_class_precision = float(class_precision)
            anchor_class_coverage = float(class_coverage)
            
            anchor_data = {
                "instance_idx": instance_idx_in_range,
                "data_instance_idx": int(data_instance_idx),
                # Instance-level metrics
                "instance_precision": anchor_instance_precision,
                "instance_coverage": anchor_instance_coverage,  # Overall coverage P(x in box)
                "instance_coverage_class_conditional": anchor_instance_coverage_class_conditional,  # Class-conditional coverage P(x in box | y = target_class)
                # Class-level metrics
                "class_precision": anchor_class_precision,
                "class_coverage": anchor_class_coverage,
                "anchor_precision": float(anchor_instance_precision),
                "anchor_coverage": float(anchor_instance_coverage),
                "anchor_coverage_class_conditional": float(anchor_instance_coverage_class_conditional),
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
        
        # Compute average class-conditional coverage if available
        instance_coverages_class_conditional = []
        for anchor_data in anchors_list:
            if "instance_coverage_class_conditional" in anchor_data:
                instance_coverages_class_conditional.append(anchor_data["instance_coverage_class_conditional"])
            elif "anchor_coverage_class_conditional" in anchor_data:
                instance_coverages_class_conditional.append(anchor_data["anchor_coverage_class_conditional"])
        instance_coverage_class_conditional = float(np.mean(instance_coverages_class_conditional)) if instance_coverages_class_conditional else 0.0
        
        avg_rollout_time = float(np.mean(rollout_times)) if rollout_times else 0.0
        total_rollout_time = float(np.sum(rollout_times)) if rollout_times else 0.0
        
        # Compute class-level metrics from union of all anchors across all episodes FOR THIS AGENT
        # NOTE: This is per-agent union (union of this agent's anchors), not class-level union across all agents
        class_precision_union = 0.0
        class_coverage_union = 0.0
        n_class_samples = 0  # Store class sample count for weighted averaging
        
        # Get the appropriate dataset (test or train) based on eval_on_test_data
        if eval_on_test_data and env_data.get("X_test_unit") is not None:
            X_data = env_data["X_test_unit"]
            y_data = env_data["y_test"]
        else:
            X_data = env_data["X_unit"]
            y_data = env_data["y"]
        
        # Compute union of all anchors FOR THIS AGENT ONLY
        if X_data is not None and y_data is not None and len(anchors_list) > 0:
            n_samples = X_data.shape[0]
            union_mask = np.zeros(n_samples, dtype=bool)
            
            # Build union mask from all anchors for this agent
            # Each agent should have different anchors, so the union should be different
            for anchor_data in anchors_list:
                if "lower_bounds_normalized" in anchor_data and "upper_bounds_normalized" in anchor_data:
                    lower = np.array(anchor_data["lower_bounds_normalized"], dtype=np.float32)
                    upper = np.array(anchor_data["upper_bounds_normalized"], dtype=np.float32)
                    
                    # Check which points fall in this anchor box
                    in_box = np.all((X_data >= lower) & (X_data <= upper), axis=1)
                    union_mask |= in_box
            
            # Class-level coverage: fraction of class samples that are in the union
            mask_cls = (y_data == target_class)
            n_class_samples = int(mask_cls.sum())  # Store for weighted averaging
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
            
            # Debug: Log anchor diversity and coverage diagnostics for this agent
            if len(anchors_list) > 1:
                # Check if anchors are actually different
                anchor_centers = []
                anchor_widths = []
                for anchor_data in anchors_list:
                    if "lower_bounds_normalized" in anchor_data and "upper_bounds_normalized" in anchor_data:
                        lower = np.array(anchor_data["lower_bounds_normalized"], dtype=np.float32)
                        upper = np.array(anchor_data["upper_bounds_normalized"], dtype=np.float32)
                        center = (lower + upper) / 2
                        width = upper - lower
                        anchor_centers.append(center)
                        anchor_widths.append(width)
                
                if anchor_centers:
                    centers_array = np.array(anchor_centers)
                    widths_array = np.array(anchor_widths)
                    center_std = centers_array.std(axis=0).mean()
                    width_std = widths_array.std(axis=0).mean()
                    avg_width = widths_array.mean(axis=0).mean()
                    # Log if anchors are very similar (std < 0.001) which might indicate a problem
                    if center_std < 0.001 or width_std < 0.001:
                        logger.warning(f"  Agent {agent_name}: Anchors are very similar! center_std={center_std:.6f}, width_std={width_std:.6f}")
                    elif verbose_logging:
                        logger.debug(f"  Agent {agent_name}: {len(anchors_list)} anchors, center_std={center_std:.6f}, width_std={width_std:.6f}")
                    
                    # Log coverage diagnostics if coverage is low
                    if class_coverage_union < 0.05:
                        logger.warning(
                            f"  Agent {agent_name}: Low per-agent union coverage ({class_coverage_union:.4f}). "
                            f"Average box width: {avg_width:.4f}, "
                            f"Total samples: {n_samples}, Class samples: {n_class_samples}, "
                            f"Samples in union: {union_mask.sum()}"
                        )
        
        # When agents_per_class > 1, we need to aggregate results from all agents of the same class
        # Store per-agent results first, then aggregate at class level
        agent_result = {
            "class": int(target_class),
            "agent": agent_name,  # Store which agent this result is from
            "group": agent_name,
            # Instance-level metrics (averaged across all instances for this agent)
            "instance_precision": instance_precision,
            "instance_coverage": instance_coverage,  # Overall coverage P(x in box)
            "instance_coverage_class_conditional": instance_coverage_class_conditional,  # Class-conditional coverage P(x in box | y = target_class)
            "instance_precision_std": float(np.std(instance_precisions)) if len(instance_precisions) > 1 else 0.0,
            "instance_coverage_std": float(np.std(instance_coverages)) if len(instance_coverages) > 1 else 0.0,
            # Class-level metrics (union of all anchors for this agent across all episodes)
            # Note: This is per-agent union, not class-level union across all agents
            "class_precision": class_precision_union,
            "class_coverage": class_coverage_union,
            "n_class_samples": n_class_samples,  # Store for weighted averaging in global metrics
            # Keep averaged class-level metrics for backward compatibility
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
        
        # Aggregate results at class level when agents_per_class > 1
        if class_key not in results["per_class_results"]:
            # First agent for this class - initialize class-level results
            results["per_class_results"][class_key] = {
                "class": int(target_class),
                "agents": [agent_name],  # Track which agents contributed
                "per_agent_results": {agent_name: agent_result},  # Store per-agent results
                # Initialize aggregated metrics (will be updated as we process more agents)
                "instance_precision": instance_precision,
                "instance_coverage": instance_coverage,
                "class_precision": class_precision_union,  # Temporary - will be overwritten with class-based union only
                "class_coverage": class_coverage_union,  # Temporary - will be overwritten with class-based union only
                "n_class_samples": n_class_samples,  # Store for weighted averaging in global metrics
                "all_anchors": anchors_list.copy(),  # Collect instance-based anchors from all agents
                "all_rules": rules_list.copy(),  # Collect instance-based rules from all agents
                "total_rollout_time_seconds": total_rollout_time,
                "class_based_results": {},  # Will store class-based results per agent
            }
        else:
            # Additional agent for this class - aggregate results
            existing = results["per_class_results"][class_key]
            existing["agents"].append(agent_name)
            existing["per_agent_results"][agent_name] = agent_result
            
            # Aggregate instance-level metrics (average across all agents)
            n_agents = len(existing["agents"])
            existing["instance_precision"] = (
                (existing["instance_precision"] * (n_agents - 1) + instance_precision) / n_agents
            )
            existing["instance_coverage"] = (
                (existing["instance_coverage"] * (n_agents - 1) + instance_coverage) / n_agents
            )
            
            # Aggregate anchors and rules
            existing["all_anchors"].extend(anchors_list)
            existing["all_rules"].extend(rules_list)
            existing["total_rollout_time_seconds"] += total_rollout_time
            
            # Recompute class-level union metrics from ALL anchors (instance-based + class-based) across ALL agents
            # Get the appropriate dataset (test or train) based on eval_on_test_data
            if eval_on_test_data and env_data.get("X_test_unit") is not None:
                X_data = env_data["X_test_unit"]
                y_data = env_data["y_test"]
            else:
                X_data = env_data["X_unit"]
                y_data = env_data["y"]
            
            # Collect ALL anchors: instance-based + class-based
            all_anchors_for_union = existing["all_anchors"].copy()  # Instance-based anchors
            
            # Add class-based anchors if they exist (they're stored in class_based_results nested under class_key)
            if "class_based_results" in existing:
                # Collect anchors from all agents' class-based results
                for agent_cb_result in existing["class_based_results"].values():
                    if "anchors" in agent_cb_result:
                        all_anchors_for_union.extend(agent_cb_result["anchors"])
            
            # Compute union of ALL anchors (instance-based + class-based) from ALL agents for this class
            if X_data is not None and y_data is not None and len(all_anchors_for_union) > 0:
                n_samples = X_data.shape[0]
                union_mask = np.zeros(n_samples, dtype=bool)
                
                # Build union mask from all anchors (instance-based + class-based) across all agents
                for anchor_data in all_anchors_for_union:
                    if "lower_bounds_normalized" in anchor_data and "upper_bounds_normalized" in anchor_data:
                        lower = np.array(anchor_data["lower_bounds_normalized"], dtype=np.float32)
                        upper = np.array(anchor_data["upper_bounds_normalized"], dtype=np.float32)
                        
                        # Check which points fall in this anchor box
                        in_box = np.all((X_data >= lower) & (X_data <= upper), axis=1)
                        union_mask |= in_box
                
                # Class-level coverage: fraction of class samples that are in the union
                mask_cls = (y_data == target_class)
                n_class_samples_aggregated = int(mask_cls.sum())  # Store for weighted averaging
                if mask_cls.sum() > 0:
                    existing["class_coverage"] = float(union_mask[mask_cls].mean())
                else:
                    existing["class_coverage"] = 0.0
                existing["n_class_samples"] = n_class_samples_aggregated  # Update for weighted averaging
                
                # Class-level precision: fraction of points in union that belong to target class
                if union_mask.any():
                    y_union = y_data[union_mask]
                    existing["class_precision"] = float((y_union == target_class).mean())
                else:
                    existing["class_precision"] = 0.0
        
        # For backward compatibility and simpler access, also store the aggregated result
        # using the same structure as before (but now aggregated across all agents)
        if agents_per_class == 1:
            # Single agent per class - use the agent result directly
            results["per_class_results"][class_key] = agent_result
        else:
            # Multiple agents per class - use aggregated result
            # The aggregated result is already stored above, but add some convenience fields
            aggregated = results["per_class_results"][class_key]
            aggregated["unique_rules"] = list(set([
                r for r in aggregated["all_rules"] 
                if r and r != "any values (no tightened features)"
            ]))
            aggregated["unique_rules_count"] = len(aggregated["unique_rules"])
            aggregated["n_episodes"] = len(aggregated["all_anchors"])
            aggregated["rules"] = aggregated["all_rules"]  # All rules from all agents
            aggregated["anchors"] = aggregated["all_anchors"]  # All anchors from all agents
        
        logger.info(f"  Processed {len(anchors_list)} episodes for agent {agent_name}")
        # Log in order: Instance metrics, Class-based metrics (will be logged later), then Union metrics
        logger.info(f"  Instance-level - Average precision: {instance_precision:.4f}, coverage: {instance_coverage:.4f}")
        if instance_coverage_class_conditional > 0.0:
            logger.info(f"    Class-conditional coverage: {instance_coverage_class_conditional:.4f}")
            if instance_coverage < 0.05 and instance_coverage_class_conditional > 0.1:
                logger.info(f"    Note: Low overall coverage is expected when dataset has many samples from other classes.")
        logger.info(f"  Unique rules (instance-based): {len(unique_rules)}")
        logger.info(f"  Average rollout time per episode: {avg_rollout_time:.4f}s")
        # Union metrics will be logged after all agents are processed (for multi-agent) or here (for single-agent)
        if agents_per_class > 1:
            # Log per-agent union metrics with anchor count for debugging
            logger.info(f"  Per-agent union (this agent's {len(anchors_list)} episodes) - Precision: {class_precision_union:.4f}, coverage: {class_coverage_union:.4f}")
            # Class-level union (across all agents) will be logged after all agents are processed
        else:
            logger.info(f"  Class-level (Union) - Precision: {class_precision_union:.4f}, coverage: {class_coverage_union:.4f}")
        # logger.info(f"  Total rollout time for agent: {total_rollout_time:.4f}s")
        # logger.info(f"  Total agent processing time: {class_total_time:.4f}s")
        
        # If this is the last agent for this class, log aggregated instance-level metrics
        # Union metrics will be recomputed after class-based rollouts using ONLY class-based anchors
        if agents_per_class > 1 and class_key in results["per_class_results"]:
            aggregated = results["per_class_results"][class_key]
            if len(aggregated.get("agents", [])) == agents_per_class:
                # All agents for this class have been processed (instance-based rollouts)
                logger.info(f"\n  {'='*60}")
                logger.info(f"  Class {target_class} - Instance-Based Results (all {agents_per_class} agents):")
                logger.info(f"    Instance-level (avg across agents): "
                          f"precision={aggregated['instance_precision']:.4f}, "
                          f"coverage={aggregated['instance_coverage']:.4f}")
                logger.info(f"    Per-agent union (instance-based anchors only): "
                          f"precision={aggregated['class_precision']:.4f}, "
                          f"coverage={aggregated['class_coverage']:.4f}")
                logger.info(f"    Total unique rules (instance-based, across all agents): {aggregated.get('unique_rules_count', 0)}")
                logger.info(f"    Total episodes (instance-based, across all agents): {aggregated.get('n_episodes', 0)}")
                logger.info(f"  {'='*60}")
                logger.info(f"  Note: Final union metrics (using class-based anchors only) will be computed after class-based rollouts.")
    
    # ============================================================================
    # CLASS-BASED ROLLOUTS (using k-means cluster centroids)
    # ============================================================================
    logger.info(f"\n{'='*80}")
    logger.info("Starting CLASS-BASED rollouts (initialized from k-means cluster centroids)")
    logger.info(f"{'='*80}")
    
    # Determine number of class-based rollouts per agent
    # Use cluster centroids when available, otherwise limit to a reasonable number
    n_class_based_rollouts_per_agent = None
    if env_config.get("cluster_centroids_per_class") is not None:
        # Use number of cluster centroids as guidance
        max_centroids = 0
        for cls in target_classes:
            if cls in env_config["cluster_centroids_per_class"]:
                max_centroids = max(max_centroids, len(env_config["cluster_centroids_per_class"][cls]))
        n_class_based_rollouts_per_agent = max(5, min(max_centroids, 10))  # Between 5 and 10 rollouts
        logger.info(f"  Using {n_class_based_rollouts_per_agent} class-based rollouts per agent (based on cluster centroids)")
    else:
        n_class_based_rollouts_per_agent = 5
        logger.info(f"  Using {n_class_based_rollouts_per_agent} class-based rollouts per agent (default, no cluster centroids)")
    
    # Run class-based rollouts for each agent/class
    for agent_name, policy in policies.items():
        target_class = agent_to_class.get(agent_name, target_classes[0] if target_classes else 0)
        class_key = f"class_{target_class}"
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Class-based rollouts for class {target_class} (agent: {agent_name})")
        logger.info(f"{'='*80}")
        
        # Initialize lists for class-based rollouts
        class_based_anchors_list = []
        class_based_rules_list = []
        class_based_instance_precisions = []
        class_based_instance_coverages = []
        class_based_class_precisions = []
        class_based_class_coverages = []
        class_based_rollout_times = []
        
        class_based_start_time = time.perf_counter()
        
        # Run class-based rollouts (NOT setting x_star_unit, so environment uses cluster centroids)
        for rollout_idx in range(n_class_based_rollouts_per_agent):
            rollout_seed = seed + 10000 + rollout_idx if seed is not None else None  # Use different seed range
            
            # CRITICAL: Use FULL dataset (train + test) for class-based rollouts
            # Class-based rules should express a particular class of the full dataset
            # This ensures rules are evaluated on the complete class distribution
            if env_data.get("X_test_unit") is not None:
                # Combine train and test data for class-based rollouts (full dataset)
                full_X_unit = np.vstack([env_data["X_unit"], env_data["X_test_unit"]])
                full_X_std = np.vstack([env_data["X_std"], env_data["X_test_std"]])
                full_y = np.concatenate([env_data["y"], env_data["y_test"]])
                use_full_dataset = True
                if rollout_idx == 0:  # Log once per class
                    logger.info(f"  Using FULL dataset (train + test) for class-based rollouts")
                    logger.info(f"    Training samples: {len(env_data['X_unit'])}, Test samples: {len(env_data['X_test_unit'])}, Total: {len(full_y)}")
                    logger.info(f"    Note: Class-based rules should express the class across the full dataset")
            else:
                # Fallback to training data only if test data not available
                full_X_unit = env_data["X_unit"]
                full_X_std = env_data["X_std"]
                full_y = env_data["y"]
                use_full_dataset = False
                if rollout_idx == 0:  # Log once per class
                    logger.info(f"  Using TRAINING data only for class-based rollouts (test data not available)")
            
            # Create environment config for this class with full dataset
            class_based_config = anchor_config.copy()
            class_based_config["target_classes"] = [target_class]
            # Override with full dataset
            class_based_config["X_unit"] = full_X_unit
            class_based_config["X_std"] = full_X_std
            class_based_config["y"] = full_y
            
            # Ensure env_config is properly set up
            if "env_config" not in class_based_config:
                class_based_config["env_config"] = {}
            class_based_config["env_config"] = class_based_config["env_config"].copy()
            class_based_config["env_config"]["mode"] = "inference"
            class_based_config["env_config"]["normalize_data"] = False
            # When using full dataset, set eval_on_test_data=False so metrics are computed on full dataset
            class_based_config["env_config"]["eval_on_test_data"] = False if use_full_dataset else eval_on_test_data
            
            # Ensure cluster centroids are available
            if "cluster_centroids_per_class" not in class_based_config["env_config"]:
                class_based_config["env_config"]["cluster_centroids_per_class"] = env_config.get("cluster_centroids_per_class")
            
            # Ensure other required configs
            if "agents_per_class" not in class_based_config["env_config"]:
                class_based_config["env_config"]["agents_per_class"] = agents_per_class
            if "X_min" not in class_based_config["env_config"]:
                class_based_config["env_config"]["X_min"] = env_config.get("X_min")
            if "X_range" not in class_based_config["env_config"]:
                class_based_config["env_config"]["X_range"] = env_config.get("X_range")
            
            # Create environment
            env = AnchorEnv(**class_based_config)
            
            # Get the actual agent name
            actual_agent_name = None
            if hasattr(env, 'possible_agents') and len(env.possible_agents) > 0:
                if agent_name in env.possible_agents:
                    actual_agent_name = agent_name
                else:
                    actual_agent_name = env.possible_agents[0]
            elif hasattr(env, 'agents') and len(env.agents) > 0:
                actual_agent_name = env.agents[0]
            else:
                if agents_per_class == 1:
                    actual_agent_name = f"agent_{target_class}"
                else:
                    actual_agent_name = f"agent_{target_class}_0"
            
            # IMPORTANT: DO NOT set x_star_unit - this triggers class-based initialization
            # The environment will use cluster_centroids_per_class when available,
            # otherwise it will use mean centroid of the class
            
            # Run rollout
            episode_data = run_rollout_with_policy(
                env=env,
                policy=policy,
                agent_id=actual_agent_name,
                max_steps=steps_per_episode,
                device=device,
                seed=rollout_seed,
                exploration_mode=exploration_mode,
                action_noise_scale=action_noise_scale,
                verbose_logging=verbose_logging,
            )
            
            # Extract metrics
            instance_precision = 0.0
            instance_coverage = 0.0
            class_precision = 0.0
            class_coverage = 0.0
            
            if episode_data:
                instance_precision = episode_data.get("anchor_precision", episode_data.get("instance_precision", 0.0))
                instance_coverage = episode_data.get("anchor_coverage", episode_data.get("instance_coverage", 0.0))
                class_precision = episode_data.get("class_precision", 0.0)
                class_coverage = episode_data.get("class_coverage", 0.0)
                
                class_based_instance_precisions.append(float(instance_precision))
                class_based_instance_coverages.append(float(instance_coverage))
                class_based_class_precisions.append(float(class_precision))
                class_based_class_coverages.append(float(class_coverage))
                
                rollout_time = episode_data.get("rollout_time_seconds", 0.0)
                class_based_rollout_times.append(float(rollout_time))
            
            # Extract rule and bounds
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
                    
                    # Denormalize bounds
                    X_min = env_config.get("X_min")
                    X_range = env_config.get("X_range")
                    if X_min is not None and X_range is not None:
                        lower = (lower_normalized * X_range) + X_min
                        upper = (upper_normalized * X_range) + X_min
                    else:
                        lower = lower_normalized
                        upper = upper_normalized
                    
                    # Extract rule - use the box center as reference for class-based initialization
                    temp_env = AnchorEnv(
                        X_unit=anchor_config["X_unit"],
                        X_std=anchor_config["X_std"],
                        y=anchor_config["y"],
                        feature_names=feature_names,
                        classifier=dataset_loader.get_classifier(),
                        device="cpu",
                        target_classes=[target_class],
                        env_config={**env_config, "mode": "inference", "normalize_data": False}
                    )
                    temp_agent_name = f"agent_{target_class}"
                    temp_env.lower[temp_agent_name] = lower_normalized
                    temp_env.upper[temp_agent_name] = upper_normalized
                    
                    # For class-based, use the center of the final box as reference for initial bounds
                    # This approximates what the initial box might have been
                    initial_window = env_config.get("initial_window", 0.1)
                    box_center = (lower_normalized + upper_normalized) / 2.0
                    initial_lower_normalized = np.clip(box_center - initial_window, 0.0, 1.0)
                    initial_upper_normalized = np.clip(box_center + initial_window, 0.0, 1.0)
                    
                    rule = temp_env.extract_rule(
                        temp_agent_name,
                        max_features_in_rule=max_features_in_rule,
                        initial_lower=initial_lower_normalized,
                        initial_upper=initial_upper_normalized,
                        denormalize=True
                    )
            
            # Store anchor data
            anchor_data = {
                "rollout_type": "class_based",  # Flag to distinguish from instance-based
                "rollout_idx": rollout_idx,
                "agent": agent_name,
                "instance_precision": float(instance_precision),
                "instance_coverage": float(instance_coverage),
                "class_precision": float(class_precision),
                "class_coverage": float(class_coverage),
                "anchor_precision": float(instance_precision),
                "anchor_coverage": float(instance_coverage),
                "total_reward": float(episode_data.get("total_reward", 0.0)) if episode_data else 0.0,
                "rule": rule,
            }
            
            if lower is not None and upper is not None:
                anchor_data.update({
                    "lower_bounds": lower.tolist(),
                    "upper_bounds": upper.tolist(),
                    "box_widths": (upper - lower).tolist(),
                    "box_volume": float(np.prod(np.maximum(upper - lower, 1e-9))),
                    "lower_bounds_normalized": lower_normalized.tolist() if lower_normalized is not None else None,
                    "upper_bounds_normalized": upper_normalized.tolist() if upper_normalized is not None else None,
                })
            
            class_based_anchors_list.append(anchor_data)
            class_based_rules_list.append(rule)
        
        # Compute aggregated metrics for class-based rollouts
        class_based_unique_rules = list(set([r for r in class_based_rules_list if r and r != "any values (no tightened features)"]))
        class_based_end_time = time.perf_counter()
        class_based_total_time = class_based_end_time - class_based_start_time
        
        # CRITICAL: Filter low-precision anchors before computing average
        # We only trust high-precision anchors, so average should reflect quality of trusted anchors only
        # This matches the filtering used for union computation
        precision_target = env_config.get("precision_target", 0.95)
        precision_threshold = precision_target * 0.8
        
        # Get dataset for recomputing precision (same as used for union)
        if env_data.get("X_test_unit") is not None:
            X_data_filter = np.vstack([env_data["X_unit"], env_data["X_test_unit"]])
            y_data_filter = np.concatenate([env_data["y"], env_data["y_test"]])
        else:
            X_data_filter = env_data["X_unit"]
            y_data_filter = env_data["y"]
        
        # Filter anchors by recomputing precision on actual dataset
        filtered_class_based_instance_precisions = []
        filtered_class_based_instance_coverages = []
        filtered_class_based_class_precisions = []
        filtered_class_based_class_coverages = []
        filtered_class_based_anchors = []
        
        if X_data_filter is not None and y_data_filter is not None:
            for anchor_data in class_based_anchors_list:
                if "lower_bounds_normalized" in anchor_data and "upper_bounds_normalized" in anchor_data:
                    lower = np.array(anchor_data["lower_bounds_normalized"], dtype=np.float32)
                    upper = np.array(anchor_data["upper_bounds_normalized"], dtype=np.float32)
                    
                    # Check which points fall in this anchor box
                    in_box = np.all((X_data_filter >= lower) & (X_data_filter <= upper), axis=1)
                    
                    if in_box.sum() > 0:
                        # Compute actual precision on the dataset
                        y_in_box = y_data_filter[in_box]
                        actual_precision = float((y_in_box == target_class).mean())
                        
                        # Only include high-precision anchors in average
                        if actual_precision >= precision_threshold:
                            # Use stored precisions/coverages from anchor data
                            filtered_class_based_instance_precisions.append(anchor_data.get("instance_precision", 0.0))
                            filtered_class_based_instance_coverages.append(anchor_data.get("instance_coverage", 0.0))
                            filtered_class_based_class_precisions.append(actual_precision)  # Use recomputed precision
                            filtered_class_based_class_coverages.append(anchor_data.get("class_coverage", 0.0))
                            filtered_class_based_anchors.append(anchor_data)
        
        # Compute average of filtered high-precision anchors only
        # This represents the quality of anchors we actually trust and use
        if filtered_class_based_class_precisions:
            class_based_instance_precision = float(np.mean(filtered_class_based_instance_precisions))
            class_based_instance_coverage = float(np.mean(filtered_class_based_instance_coverages))
            class_based_class_precision = float(np.mean(filtered_class_based_class_precisions))
            class_based_class_coverage = float(np.mean(filtered_class_based_class_coverages))
            logger.info(f"  Agent {agent_name} (Class {target_class}): Filtered {len(filtered_class_based_class_precisions)}/{len(class_based_class_precisions)} "
                       f"high-precision anchors (threshold={precision_threshold:.3f})")
            logger.info(f"  Average precision of filtered anchors: {class_based_class_precision:.4f}")
        else:
            # Fallback: if no anchors pass filter, use all anchors (but log warning)
            class_based_instance_precision = float(np.mean(class_based_instance_precisions)) if class_based_instance_precisions else 0.0
            class_based_instance_coverage = float(np.mean(class_based_instance_coverages)) if class_based_instance_coverages else 0.0
            class_based_class_precision = float(np.mean(class_based_class_precisions)) if class_based_class_precisions else 0.0
            class_based_class_coverage = float(np.mean(class_based_class_coverages)) if class_based_class_coverages else 0.0
            logger.warning(f"  Agent {agent_name} (Class {target_class}): No anchors passed precision filter (threshold={precision_threshold:.3f}), "
                          f"using average of all {len(class_based_class_precisions)} anchors: {class_based_class_precision:.4f}")
        
        class_based_avg_rollout_time = float(np.mean(class_based_rollout_times)) if class_based_rollout_times else 0.0
        class_based_total_rollout_time = float(np.sum(class_based_rollout_times)) if class_based_rollout_times else 0.0
        
        # Store class-based results in per_class_results
        if class_key not in results["per_class_results"]:
            results["per_class_results"][class_key] = {}
        
        # Add class-based results (append to existing or create new)
        if "class_based_results" not in results["per_class_results"][class_key]:
            results["per_class_results"][class_key]["class_based_results"] = {}
        
        results["per_class_results"][class_key]["class_based_results"][agent_name] = {
            "class": int(target_class),
            "agent": agent_name,
            "rollout_type": "class_based",
            "n_episodes": len(class_based_anchors_list),
            "instance_precision": class_based_instance_precision,
            "instance_coverage": class_based_instance_coverage,
            "class_precision": class_based_class_precision,
            "class_coverage": class_based_class_coverage,
            "instance_precision_std": float(np.std(filtered_class_based_instance_precisions if filtered_class_based_instance_precisions else class_based_instance_precisions)) if len(filtered_class_based_instance_precisions if filtered_class_based_instance_precisions else class_based_instance_precisions) > 1 else 0.0,
            "instance_coverage_std": float(np.std(filtered_class_based_instance_coverages if filtered_class_based_instance_coverages else class_based_instance_coverages)) if len(filtered_class_based_instance_coverages if filtered_class_based_instance_coverages else class_based_instance_coverages) > 1 else 0.0,
            "unique_rules": class_based_unique_rules,
            "unique_rules_count": len(class_based_unique_rules),
            "rules": class_based_rules_list,
            "anchors": class_based_anchors_list,
            "avg_rollout_time_seconds": class_based_avg_rollout_time,
            "total_rollout_time_seconds": class_based_total_rollout_time,
            "total_processing_time_seconds": float(class_based_total_time),
            # Store count of filtered vs total anchors for transparency
            "n_filtered": len(filtered_class_based_class_precisions) if filtered_class_based_class_precisions else 0,
            "n_total": len(class_based_class_precisions),
        }
        
        logger.info(f"  Class-based rollouts completed: {len(class_based_anchors_list)} episodes")
        logger.info(f"  Class-based - Average precision: {class_based_instance_precision:.4f}, coverage: {class_based_instance_coverage:.4f}")
        logger.info(f"  Unique rules (class-based): {len(class_based_unique_rules)}")
    
    logger.info(f"\n{'='*80}")
    logger.info("Class-based rollouts completed")
    logger.info(f"{'='*80}")
    
    # Recompute union metrics for all classes using ONLY class-based anchors
    # This represents the smallest set of general rules that explain the class structure
    logger.info(f"\n{'='*80}")
    logger.info("Recomputing Class Union Metrics (Class-based anchors only)")
    logger.info(f"{'='*80}")
    
    # CRITICAL: Use FULL dataset (train + test) for class union metrics
    # Class union metrics represent rules that express a particular class of the full dataset
    # This ensures consistency with class-based rollouts which also use full dataset
    if env_data.get("X_test_unit") is not None:
        # Use full dataset (train + test) for class union metrics
        X_data_union = np.vstack([env_data["X_unit"], env_data["X_test_unit"]])
        y_data_union = np.concatenate([env_data["y"], env_data["y_test"]])
        logger.info(f"  Using FULL dataset (train + test) for class union metrics")
        logger.info(f"    Training samples: {len(env_data['X_unit'])}, Test samples: {len(env_data['X_test_unit'])}, Total: {len(y_data_union)}")
    else:
        # Fallback to training data only if test data not available
        X_data_union = env_data["X_unit"]
        y_data_union = env_data["y"]
        logger.info(f"  Using TRAINING data only for class union metrics (test data not available)")
    
    # Recompute union metrics for each class
    for class_key, class_data in results["per_class_results"].items():
        target_class = class_data.get("class")
        if target_class is None:
            continue
        
        # Collect ONLY class-based anchors (represent class structure, not instance-specific patterns)
        all_anchors_for_union = []
        class_based_unique_rules = []
        
        # Get class-based anchors and rules if they exist
        if "class_based_results" in class_data:
            for agent_cb_result in class_data["class_based_results"].values():
                if "anchors" in agent_cb_result:
                    all_anchors_for_union.extend(agent_cb_result["anchors"])
                if "unique_rules" in agent_cb_result:
                    # Collect unique rules from all agents (will deduplicate later)
                    class_based_unique_rules.extend(agent_cb_result["unique_rules"])
        
        # Deduplicate rules across all agents
        if class_based_unique_rules:
            class_based_unique_rules = list(set([
                r for r in class_based_unique_rules 
                if r and r != "any values (no tightened features)"
            ]))
        
        # Filter anchors by precision threshold before computing union
        # This ensures we only include high-quality rules in the union
        # Use the same precision threshold as used during training/inference
        # precision_threshold = precision_target * 0.8 (same as in environment.py)
        precision_target = env_config.get("precision_target", 0.95)
        precision_threshold = precision_target * 0.8
        filtered_anchors_for_union = []
        filtered_rules_for_union = []
        
        # CRITICAL: Recompute precision on actual dataset for filtering
        # The stored instance_precision uses perturbation sampling which may not match actual dataset precision
        if X_data_union is not None and y_data_union is not None:
            for anchor_data in all_anchors_for_union:
                if "lower_bounds_normalized" in anchor_data and "upper_bounds_normalized" in anchor_data:
                    lower = np.array(anchor_data["lower_bounds_normalized"], dtype=np.float32)
                    upper = np.array(anchor_data["upper_bounds_normalized"], dtype=np.float32)
                    
                    # Check which points fall in this anchor box
                    in_box = np.all((X_data_union >= lower) & (X_data_union <= upper), axis=1)
                    
                    if in_box.sum() > 0:
                        # Compute actual precision on the dataset
                        y_in_box = y_data_union[in_box]
                        actual_precision = float((y_in_box == target_class).mean())
                        
                        # Use actual precision for filtering, not stored instance_precision
                        if actual_precision >= precision_threshold:
                            filtered_anchors_for_union.append(anchor_data)
                            # Also track the corresponding rule if available
                            rule = anchor_data.get("rule", "")
                            if rule and rule != "any values (no tightened features)":
                                filtered_rules_for_union.append(rule)
                    else:
                        # No samples in box - skip this anchor
                        continue
        else:
            # Fallback: use stored precision if dataset not available
            logger.warning(f"  Class {target_class} - Cannot recompute precision on dataset, using stored values")
            for anchor_data in all_anchors_for_union:
                # Get precision from anchor data (prefer instance_precision, fallback to anchor_precision)
                anchor_precision = anchor_data.get("instance_precision", anchor_data.get("anchor_precision", 0.0))
                
                if anchor_precision >= precision_threshold:
                    filtered_anchors_for_union.append(anchor_data)
                    # Also track the corresponding rule if available
                    rule = anchor_data.get("rule", "")
                    if rule and rule != "any values (no tightened features)":
                        filtered_rules_for_union.append(rule)
        
        # Update class_based_unique_rules to only include rules from high-precision anchors
        if filtered_rules_for_union:
            class_based_unique_rules = list(set(filtered_rules_for_union))
        
        # Log filtering statistics
        n_total_anchors = len(all_anchors_for_union)
        n_filtered_anchors = len(filtered_anchors_for_union)
        if n_total_anchors > 0:
            logger.info(f"  Class {target_class} - Filtered {n_total_anchors} anchors to {n_filtered_anchors} high-precision anchors (precision >= {precision_threshold:.2f})")
        
        # Compute union of class-based anchors only (smallest set of general rules)
        # Use filtered anchors instead of all anchors
        if X_data_union is not None and y_data_union is not None and len(filtered_anchors_for_union) > 0:
            n_samples = X_data_union.shape[0]
            union_mask = np.zeros(n_samples, dtype=bool)
            
            # Build union mask from filtered high-precision anchors only
            for anchor_data in filtered_anchors_for_union:
                if "lower_bounds_normalized" in anchor_data and "upper_bounds_normalized" in anchor_data:
                    lower = np.array(anchor_data["lower_bounds_normalized"], dtype=np.float32)
                    upper = np.array(anchor_data["upper_bounds_normalized"], dtype=np.float32)
                    
                    # Check which points fall in this anchor box
                    in_box = np.all((X_data_union >= lower) & (X_data_union <= upper), axis=1)
                    union_mask |= in_box
            
            # Class-level coverage: fraction of class samples that are in the union
            mask_cls = (y_data_union == target_class)
            if mask_cls.sum() > 0:
                class_coverage_combined = float(union_mask[mask_cls].mean())
            else:
                class_coverage_combined = 0.0
            
            # Class-level precision: fraction of points in union that belong to target class
            if union_mask.any():
                y_union = y_data_union[union_mask]
                class_precision_combined = float((y_union == target_class).mean())
            else:
                class_precision_combined = 0.0
            
            # Update class-level metrics with class-based union only
            class_data["class_precision"] = class_precision_combined
            class_data["class_coverage"] = class_coverage_combined
            
            # CRITICAL: Store union rules in class_data so they can be accessed later
            # These are the deduplicated class-based rules that form the union
            class_data["class_level_unique_rules"] = class_based_unique_rules
            class_data["class_union_unique_rules"] = class_based_unique_rules  # Alias for clarity
            
            # Log the final union metrics
            n_class_based_anchors = len(filtered_anchors_for_union)
            n_unique_rules = len(class_based_unique_rules)
            logger.info(f"\n  Class {target_class} - Final Union Metrics (High-Precision Class-Based Anchors Only):")
            logger.info(f"    Union precision: {class_precision_combined:.4f}, Union coverage: {class_coverage_combined:.4f}")
            logger.info(f"    Anchors used: {n_class_based_anchors} high-precision class-based anchors (precision >= {precision_threshold:.2f})")
            logger.info(f"    Unique Rules: {n_unique_rules}")
            logger.info(f"    Note: Only includes anchors with precision >= {precision_threshold:.2f} to ensure high-quality union")
            
            # Log the actual rules
            if class_based_unique_rules:
                logger.info(f"\n  Class {target_class} - Class-Based Union Rules:")
                for i, rule in enumerate(class_based_unique_rules, 1):
                    logger.info(f"    Rule {i}: {rule}")
            else:
                logger.info(f"\n  Class {target_class} - No unique rules found in class-based anchors")
        elif X_data_union is not None and y_data_union is not None and len(filtered_anchors_for_union) == 0 and len(all_anchors_for_union) > 0:
            # All anchors were filtered out due to low precision
            logger.warning(f"  Class {target_class} - All {len(all_anchors_for_union)} anchors filtered out (precision < {precision_threshold:.2f})")
            class_data["class_precision"] = 0.0
            class_data["class_coverage"] = 0.0
            class_data["class_level_unique_rules"] = []
            class_data["class_union_unique_rules"] = []
        else:
            # No class-based anchors available - set union metrics to 0.0
            if len(all_anchors_for_union) == 0:
                logger.warning(f"  Class {target_class}: No class-based anchors found. Class union metrics set to 0.0")
                class_data["class_precision"] = 0.0
                class_data["class_coverage"] = 0.0
    
    # End overall timing
    overall_end_time = time.perf_counter()
    overall_total_time = overall_end_time - overall_start_time
    
    logger.info("\n" + "="*80)
    logger.info(f"Overall inference time: {overall_total_time:.4f}s")
    logger.info("="*80)
    
    # Load and log NashConv metrics if available
    # IMPORTANT: This is purely for logging and does NOT affect inference results.
    # If NashConv loading fails, inference continues normally.
    # Use experiment_dir parameter (already available in function scope)
    try:
        nashconv_data = _load_nashconv_metrics(experiment_dir)
        if nashconv_data.get("available", False):
            logger.info("\n" + "="*80)
            logger.info("NASH EQUILIBRIUM CONVERGENCE METRICS")
            logger.info("="*80)
            
            has_any_metrics = False
            
            # Training metrics
            if nashconv_data.get("training"):
                training_data = nashconv_data["training"]
                if training_data:
                    final_training = training_data[-1]
                    nashconv_sum = final_training.get('training/nashconv_sum')
                    exploitability_max = final_training.get('training/exploitability_max')
                    
                    if nashconv_sum is not None or exploitability_max is not None:
                        has_any_metrics = True
                        logger.info("Training NashConv (final):")
                        if nashconv_sum is not None:
                            logger.info(f"  NashConv sum: {nashconv_sum:.6f}")
                        else:
                            logger.info(f"  NashConv sum: N/A (not found in training history)")
                        if exploitability_max is not None:
                            logger.info(f"  Max exploitability: {exploitability_max:.6f}")
                        else:
                            logger.info(f"  Max exploitability: N/A (not found in training history)")
                        if 'training/class_nashconv_sum' in final_training:
                            logger.info(f"  Class NashConv sum: {final_training.get('training/class_nashconv_sum'):.6f}")
                        logger.info(f"  Training step: {final_training.get('step', 'N/A')}")
            
            # Evaluation metrics
            if nashconv_data.get("evaluation"):
                eval_data = nashconv_data["evaluation"]
                if eval_data:
                    final_eval = eval_data[-1]
                    nashconv_sum = final_eval.get('evaluation/nashconv_sum')
                    exploitability_max = final_eval.get('evaluation/exploitability_max')
                    
                    if nashconv_sum is not None or exploitability_max is not None:
                        has_any_metrics = True
                        logger.info("\nEvaluation NashConv (final):")
                        if nashconv_sum is not None:
                            logger.info(f"  NashConv sum: {nashconv_sum:.6f}")
                        else:
                            logger.info(f"  NashConv sum: N/A (not found in evaluation history)")
                        if exploitability_max is not None:
                            logger.info(f"  Max exploitability: {exploitability_max:.6f}")
                        else:
                            logger.info(f"  Max exploitability: N/A (not found in evaluation history)")
                        if 'evaluation/class_nashconv_sum' in final_eval:
                            logger.info(f"  Class NashConv sum: {final_eval.get('evaluation/class_nashconv_sum'):.6f}")
                        logger.info(f"  Evaluation step: {final_eval.get('step', 'N/A')}")
            
            if not has_any_metrics:
                logger.info("  NashConv metrics files found but no metrics extracted.")
                logger.info("  This may indicate the training/evaluation history files don't contain NashConv metrics.")
                logger.info(f"  Experiment directory: {experiment_dir}")
            
            logger.info("="*80)
        else:
            logger.debug(f"NashConv metrics not available. Experiment dir: {experiment_dir}")
    except Exception as e:
        # NashConv loading/logging failures should NOT break inference
        logger.warning(f"Failed to load/log NashConv metrics (non-fatal): {e}")
        logger.debug(f"NashConv error details: {type(e).__name__}: {e}", exc_info=True)
    
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
        default=100,
        help="Maximum steps per rollout episode (default: 100, matches config max_cycles)"
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
