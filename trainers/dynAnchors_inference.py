"""
Post-training evaluation and inference for Dynamic Anchors.

This module provides functions to evaluate trained PPO/DDPG policies and generate
dynamic anchor explanations for individual instances and classes.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter
from stable_baselines3 import PPO, DDPG, TD3


def compute_coverage_on_data(
    lower: np.ndarray,
    upper: np.ndarray,
    X_unit: np.ndarray
) -> float:
    """
    Compute coverage of a box on a dataset.
    
    Args:
        lower: Lower bounds of the box (in unit space [0,1])
        upper: Upper bounds of the box (in unit space [0,1])
        X_unit: Data points in unit space [0,1], shape (n_samples, n_features)
    
    Returns:
        Coverage: fraction of data points that fall within the box
    """
    if X_unit is None or X_unit.shape[0] == 0:
        return 0.0
    
    # Check which points fall within the box
    mask = np.all((X_unit >= lower) & (X_unit <= upper), axis=1)
    coverage = float(mask.mean())
    return coverage


def greedy_rollout(
    env,
    trained_model: Union[PPO, DDPG, TD3],
    steps_per_episode: int = 100,
    max_features_in_rule: int = 5,
    device: str = "cpu"
) -> Tuple[Dict[str, Any], str, np.ndarray, np.ndarray]:
    """
    Perform greedy evaluation with a trained policy.
    
    Args:
        env: AnchorEnv, DynamicAnchorEnv, or ContinuousAnchorEnv instance to evaluate on
        trained_model: Trained PPO or DDPG model
        steps_per_episode: Maximum number of steps for rollout
        max_features_in_rule: Maximum number of features to include in rule
    
    Returns:
        Tuple of (info_dict, rule_string, lower_bounds, upper_bounds)
    """
    
    # Detect model type: DDPG/TD3 have actor/critic, PPO has policy
    # TD3 also has actor/critic, so we treat it similarly to DDPG
    is_continuous = (hasattr(trained_model, 'actor') and hasattr(trained_model, 'critic'))
    is_ddpg = is_continuous  # For backward compatibility
    is_td3 = is_continuous  # TD3 has same structure as DDPG
    is_ppo = hasattr(trained_model, 'policy')
    
    # Detect environment type
    is_continuous_env = hasattr(env, 'anchor_env') and hasattr(env, 'action_space') and hasattr(env.action_space, 'shape')
    is_dynamic_env = hasattr(env, 'anchor_env') and not is_continuous_env
    is_anchor_env = not hasattr(env, 'anchor_env')
    
    # Reset environment and get initial state
    reset_result = env.reset()
    
    # Handle both AnchorEnv (returns state) and gym wrappers (returns (obs, info) tuple)
    if isinstance(reset_result, tuple):
        state, reset_info = reset_result
    else:
        state = reset_result
        reset_info = {}
    
    # Ensure state is numpy array
    state = np.array(state, dtype=np.float32)
    
    # Get anchor_env reference and initial bounds
    if is_continuous_env or is_dynamic_env:
        # ContinuousAnchorEnv or DynamicAnchorEnv: access anchor_env for properties
        anchor_env = env.anchor_env
        initial_lower = anchor_env.lower.copy()
        initial_upper = anchor_env.upper.copy()
        # Get initial metrics from anchor_env
        prec, cov, _ = anchor_env._current_metrics()
        # Note: DynamicAnchorEnv already includes target_class in observation space,
        # so we don't need to append it here
    else:
        # AnchorEnv: access properties directly
        anchor_env = env
        initial_lower = env.lower.copy()
        initial_upper = env.upper.copy()
        prec, cov, _ = env._current_metrics()
        # For AnchorEnv (not wrapped), we need to append target_class to state
        # only if the model expects it (check observation space)
        # However, if using PPO with DynamicAnchorEnv wrapper, target_class is already included
        # So we only append for raw AnchorEnv usage
        if is_ppo and not is_ddpg and not is_dynamic_env:
            target_class_value = float(env.target_class)
            state = np.concatenate([state, np.array([target_class_value], dtype=np.float32)])
    
    initial_width = (initial_upper - initial_lower)
    
    last_info = {
        "precision": prec,
        "coverage": cov,
        "hard_precision": prec,
        "avg_prob": prec,
        "sampler": "empirical"
    }
    bounds_changed = False
    
    # Run greedy rollout
    for t in range(steps_per_episode):
        with torch.no_grad():
            # Use trained model to predict action
            action, _states = trained_model.predict(state, deterministic=True)
            
            # Handle action type based on model
            if is_continuous:
                # DDPG/TD3: action is already a numpy array (continuous)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                action = np.clip(action, -1.0, 1.0)
            else:
                # PPO: action is discrete (int)
                action = int(action)
        
        prev_lower = anchor_env.lower.copy()
        prev_upper = anchor_env.upper.copy()
        step_result = env.step(action)
        
        # Handle both gym.Env API (5-tuple) and old API (4-tuple)
        if len(step_result) == 5:
            # gym.Env API: (observation, reward, terminated, truncated, info)
            state, _, done, _, info = step_result
        else:
            # Old API: (state, reward, done, info)
            state, _, done, info = step_result
        
        # Ensure state is numpy array
        state = np.array(state, dtype=np.float32)
        
        # For AnchorEnv (not wrapped), append target_class only if needed
        # DynamicAnchorEnv already includes target_class in observation space
        if is_anchor_env and is_ppo and not is_ddpg and not is_dynamic_env:
            target_class_value = float(env.target_class)
            state = np.concatenate([state, np.array([target_class_value], dtype=np.float32)])
        
        if not np.allclose(prev_lower, anchor_env.lower) or not np.allclose(prev_upper, anchor_env.upper):
            bounds_changed = True
        
        # Extract info properly (handle both dict and gym.Env info format)
        if isinstance(info, dict):
            last_info = info
        else:
            # If info is not a dict, try to get from anchor_env
            prec, cov, det = anchor_env._current_metrics()
            last_info = {
                "precision": prec,
                "coverage": cov,
                "hard_precision": det.get("hard_precision", prec),
                "avg_prob": det.get("avg_prob", prec),
                "sampler": det.get("sampler", "empirical")
            }
        
        if done:
            break
    
    # If box didn't change at all, manually tighten a bit
    if not bounds_changed:
        n_tighten = min(5, anchor_env.n_features)
        idx_perm = anchor_env.rng.permutation(anchor_env.n_features)[:n_tighten]
        
        for j in idx_perm:
            width = anchor_env.upper[j] - anchor_env.lower[j]
            if width > anchor_env.min_width:
                shrink = 0.1 * width
                anchor_env.lower[j] = min(anchor_env.lower[j] + shrink, anchor_env.upper[j] - anchor_env.min_width)
                anchor_env.upper[j] = max(anchor_env.upper[j] - shrink, anchor_env.lower[j] + anchor_env.min_width)
    
    # Always recompute final metrics from anchor_env to ensure accuracy
    # This ensures we get the actual current state metrics, not just the last step's info
    prec_final, cov_final, det_final = anchor_env._current_metrics()
    last_info = {
        "precision": prec_final,
        "coverage": cov_final,
        "hard_precision": det_final.get("hard_precision", prec_final),
        "avg_prob": det_final.get("avg_prob", prec_final),
        "sampler": det_final.get("sampler", "empirical"),
        "n_points": det_final.get("n_points", 0)
    }
    
    # Build rule string
    # Compare final width to initial width to find tightened features
    lw = (anchor_env.upper - anchor_env.lower)
    
    # Ensure initial_width is valid (should be > 0 for all features)
    # If initial_width has zeros or invalid values, something went wrong
    if np.any(initial_width <= 0) or np.any(np.isnan(initial_width)) or np.any(np.isinf(initial_width)):
        # If initial_width is invalid, use full range (1.0) as reference
        # This handles edge cases where reset() might have set invalid bounds
        initial_width_ref = np.ones_like(initial_width)
    else:
        initial_width_ref = initial_width.copy()
    
    # Ensure lw is also valid
    if np.any(lw <= 0) or np.any(np.isnan(lw)) or np.any(np.isinf(lw)):
        # If final width is invalid, no features can be tightened
        tightened = np.array([], dtype=int)
    else:
        # A feature is "tightened" if its width is smaller than initial width
        # Use multiple thresholds to catch different levels of tightening
        
        # First, check for significant tightening (2% reduction)
        tightened = np.where(lw < initial_width_ref * 0.98)[0]
        
        # If no features tightened by 2%, try a more lenient threshold (1%)
        if tightened.size == 0:
            tightened = np.where(lw < initial_width_ref * 0.99)[0]
        
        # If still no tightened features, check absolute thresholds
        # This handles cases where initial_width might be small
        if tightened.size == 0:
            # Check if any feature is significantly smaller than full range (1.0)
            # This catches cases where we started near full range
            tightened = np.where(lw < 0.95)[0]
        
        # Additional check: if initial_width was close to full range (>= 0.9), 
        # any feature with width < 0.9 should be considered tightened
        if tightened.size == 0:
            if np.all(initial_width_ref >= 0.9):
                tightened = np.where(lw < 0.9)[0]
        
        # Final fallback: check if ANY feature has width smaller than its initial width
        # This is the most lenient check - any reduction counts
        if tightened.size == 0:
            tightened = np.where(lw < initial_width_ref)[0]
    
    if tightened.size == 0:
        rule = "any values (no tightened features)"
    else:
        tightened_sorted = np.argsort(lw[tightened])
        to_show_idx = tightened[tightened_sorted[:max_features_in_rule]] if max_features_in_rule > 0 else tightened
        
        if to_show_idx.size == 0:
            rule = "any values (no tightened features)"
        else:
            cond_parts = []
            for i in to_show_idx:
                cond_parts.append(f"{anchor_env.feature_names[i]} ∈ [{anchor_env.lower[i]:.2f}, {anchor_env.upper[i]:.2f}]")
            rule = " and ".join(cond_parts)
    
    return last_info, rule, anchor_env.lower.copy(), anchor_env.upper.copy()


def evaluate_single_instance(
    X_instance: np.ndarray,
    trained_model: Union[PPO, DDPG, TD3],
    make_env_fn,
    feature_names: List[str],
    target_class: int,
    steps_per_episode: int = 100,
    max_features_in_rule: int = 5,
    X_min: Optional[np.ndarray] = None,
    X_range: Optional[np.ndarray] = None,
    eval_on_test_data: bool = False,
    X_test_unit: Optional[np.ndarray] = None,
    X_test_std: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    initial_window: Optional[float] = None,
    num_rollouts_per_instance: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate and generate anchor for a single instance.
    
    Args:
        X_instance: Instance to explain (in standardized space, single row)
        trained_model: Trained PPO or DDPG model
        make_env_fn: Function that creates an AnchorEnv instance
        feature_names: List of feature names
        target_class: Target class to explain
        steps_per_episode: Maximum rollout steps
        max_features_in_rule: Maximum features in rule
        X_min: Optional min values for normalization (to unit space)
        X_range: Optional range values for normalization (to unit space)
        eval_on_test_data: If True, compute metrics on test data instead of training data
        X_test_unit: Test data in unit space [0,1] (required if eval_on_test_data=True)
        X_test_std: Test data in standardized space (required if eval_on_test_data=True)
        y_test: Test labels (required if eval_on_test_data=True)
        initial_window: Initial window size for anchor box (default: 0.3 for eval, matches training if None)
        num_rollouts_per_instance: Number of greedy rollouts to run per instance (default: 1)
                                  If > 1, metrics are averaged across rollouts
    
    Returns:
        Dictionary with anchor explanation and metrics.
        Note: 
        - Coverage and precision are computed on training data by default.
        - Set eval_on_test_data=True to compute on test data.
        - local_coverage: coverage on the data used during greedy (micro-set)
        - global_coverage: coverage on full test split (if X_test_unit provided)
        If num_rollouts_per_instance > 1, metrics are averaged across rollouts.
    """
    # Detect model type: DDPG/TD3 have actor/critic, PPO has policy
    # TD3 also has actor/critic, so we treat it similarly to DDPG
    is_continuous = (hasattr(trained_model, 'actor') and hasattr(trained_model, 'critic'))
    is_ddpg = is_continuous  # For backward compatibility
    is_td3 = is_continuous  # TD3 has same structure as DDPG
    is_ppo = hasattr(trained_model, 'policy')
    
    # Create a temporary environment to extract normalization parameters if needed
    temp_env = make_env_fn()
    
    # Get X_min and X_range from environment if not provided
    if X_min is None:
        X_min = temp_env.X_min
    if X_range is None:
        X_range = temp_env.X_range
    
    # Normalize instance to unit space [0, 1] for x_star_unit
    if X_min is not None and X_range is not None:
        X_instance_unit = (X_instance - X_min) / X_range
        X_instance_unit = np.clip(X_instance_unit, 0.0, 1.0).astype(np.float32)
    else:
        X_instance_unit = None
    
    # Create environment with x_star_unit set to the instance location
    # Default initial_window: use training default (0.15) if not specified, or use provided value
    # Note: Previous code used 0.3 for evaluation, but this may not match training conditions
    from trainers.vecEnv import AnchorEnv, DynamicAnchorEnv, ContinuousAnchorEnv
    
    # Set initial_window: use provided value, or default to training value (0.15) for consistency
    if initial_window is None:
        # Default to training value for consistency, unless explicitly overridden
        eval_initial_window = temp_env.initial_window if hasattr(temp_env, 'initial_window') else 0.15
    else:
        eval_initial_window = initial_window
    
    # Prepare test data if evaluation on test data is requested
    if eval_on_test_data:
        if X_test_unit is None or X_test_std is None or y_test is None:
            raise ValueError(
                "eval_on_test_data=True requires X_test_unit, X_test_std, and y_test. "
                "These should be provided when calling evaluate_single_instance."
            )
    
    # Create base AnchorEnv
    anchor_env = AnchorEnv(
        X_unit=temp_env.X_unit,
        X_std=temp_env.X_std,
        y=temp_env.y,
        feature_names=temp_env.feature_names,
        classifier=temp_env.classifier,
        device=str(temp_env.device),
        target_class=target_class,
        step_fracs=temp_env.step_fracs,
        min_width=temp_env.min_width,
        alpha=temp_env.alpha,
        beta=temp_env.beta,
        gamma=temp_env.gamma,
        precision_target=temp_env.precision_target,
        coverage_target=temp_env.coverage_target,
        precision_blend_lambda=temp_env.precision_blend_lambda,
        drift_penalty_weight=temp_env.drift_penalty_weight,
        use_perturbation=temp_env.use_perturbation,
        perturbation_mode=temp_env.perturbation_mode,
        n_perturb=temp_env.n_perturb,
        X_min=temp_env.X_min,
        X_range=temp_env.X_range,
        rng=temp_env.rng,
        min_coverage_floor=temp_env.min_coverage_floor,
        js_penalty_weight=temp_env.js_penalty_weight,
        x_star_unit=X_instance_unit,  # Set to instance location
        initial_window=eval_initial_window,
        # Test data evaluation support
        eval_on_test_data=eval_on_test_data,
        X_test_unit=X_test_unit if eval_on_test_data else None,
        X_test_std=X_test_std if eval_on_test_data else None,
        y_test=y_test if eval_on_test_data else None,
    )
    
    # Wrap with appropriate gym wrapper based on model type
    if is_continuous:
        # DDPG/TD3: Use ContinuousAnchorEnv
        # Enable continuous actions in AnchorEnv
        anchor_env.n_actions = 2 * anchor_env.n_features
        anchor_env.max_action_scale = max(temp_env.step_fracs) if temp_env.step_fracs else 0.02
        anchor_env.min_absolute_step = max(0.05, temp_env.min_width * 0.5)
        env = ContinuousAnchorEnv(anchor_env, seed=42)
    else:
        # PPO: Use DynamicAnchorEnv
        # IMPORTANT: x_star_unit is already set on anchor_env, so DynamicAnchorEnv.reset() will preserve it
        env = DynamicAnchorEnv(anchor_env, seed=42)
    
    # Note: Don't reset here - greedy_rollout will reset internally
    # This avoids double reset and ensures initial_width is captured correctly
    # Check initial coverage will be done inside greedy_rollout after reset
    
    # Run multiple greedy rollouts if requested
    if num_rollouts_per_instance > 1:
        # Collect results from multiple rollouts
        rollout_results = []
        for rollout_idx in range(num_rollouts_per_instance):
            # Create a new environment for each rollout to ensure fresh state
            # Use different seed for each rollout to introduce variation
            if is_ddpg:
                env_rollout = ContinuousAnchorEnv(anchor_env, seed=42 + rollout_idx)
            else:
                env_rollout = DynamicAnchorEnv(anchor_env, seed=42 + rollout_idx)
            
            info, rule, lower, upper = greedy_rollout(
                env_rollout,
                trained_model,
                steps_per_episode=steps_per_episode,
                max_features_in_rule=max_features_in_rule
            )
            rollout_results.append({
                "info": info,
                "rule": rule,
                "lower": lower,
                "upper": upper
            })
        
        # Average metrics across rollouts
        precisions = [r["info"].get("precision", 0.0) for r in rollout_results]
        hard_precisions = [r["info"].get("hard_precision", r["info"].get("precision", 0.0)) for r in rollout_results]
        local_coverages = [r["info"].get("coverage", 0.0) for r in rollout_results]
        
        # Compute global coverage for each rollout (on full test split)
        global_coverages = []
        if X_test_unit is not None:
            for r in rollout_results:
                global_cov = compute_coverage_on_data(r["lower"], r["upper"], X_test_unit)
                global_coverages.append(global_cov)
        
        avg_precision = np.mean(precisions)
        avg_hard_precision = np.mean(hard_precisions)
        avg_local_coverage = np.mean(local_coverages)
        avg_global_coverage = np.mean(global_coverages) if global_coverages else None
        
        # Use the best rollout (by hard precision) for the rule and bounds
        best_idx = np.argmax(hard_precisions)
        best_result = rollout_results[best_idx]
        
        # Also compute std dev for reporting
        std_precision = np.std(precisions) if len(precisions) > 1 else 0.0
        std_hard_precision = np.std(hard_precisions) if len(hard_precisions) > 1 else 0.0
        std_local_coverage = np.std(local_coverages) if len(local_coverages) > 1 else 0.0
        std_global_coverage = np.std(global_coverages) if len(global_coverages) > 1 else None
        
        return {
            "rule": best_result["rule"],
            "precision": float(avg_precision),
            "hard_precision": float(avg_hard_precision),
            "coverage": float(avg_local_coverage),  # Keep for backward compatibility
            "local_coverage": float(avg_local_coverage),  # Coverage on micro-set used during greedy
            "global_coverage": float(avg_global_coverage) if avg_global_coverage is not None else None,  # Coverage on full test split
            "lower_bounds": best_result["lower"].tolist(),
            "upper_bounds": best_result["upper"].tolist(),
            "data_source": best_result["info"].get("data_source", "training"),
            "num_rollouts": num_rollouts_per_instance,
            "std_precision": float(std_precision),
            "std_hard_precision": float(std_hard_precision),
            "std_coverage": float(std_local_coverage),  # Keep for backward compatibility
            "std_local_coverage": float(std_local_coverage),
            "std_global_coverage": float(std_global_coverage) if std_global_coverage is not None else None,
        }
    else:
        # Single rollout (original behavior)
        info, rule, lower, upper = greedy_rollout(
            env,
            trained_model,
            steps_per_episode=steps_per_episode,
            max_features_in_rule=max_features_in_rule
        )
        
        # Get local coverage (from greedy rollout - coverage on data used during greedy)
        local_coverage = info.get("coverage", 0.0)
        
        # Compute global coverage (on full test split) if test data is available
        global_coverage = None
        if X_test_unit is not None:
            global_coverage = compute_coverage_on_data(lower, upper, X_test_unit)
        
        return {
            "rule": rule,
            "precision": info.get("precision", 0.0),
            "hard_precision": info.get("hard_precision", 0.0),
            "coverage": local_coverage,  # Keep for backward compatibility
            "local_coverage": local_coverage,  # Coverage on micro-set used during greedy
            "global_coverage": global_coverage,  # Coverage on full test split (None if not available)
            "lower_bounds": lower.tolist(),
            "upper_bounds": upper.tolist(),
            "data_source": info.get("data_source", "training"),
            "num_rollouts": 1,
        }


def evaluate_class(
    X_test: np.ndarray,
    y_test: np.ndarray,
    trained_model: Union[PPO, DDPG],
    make_env_fn,
    feature_names: List[str],
    target_class: int,
    n_instances: int = 20,
    steps_per_episode: int = 100,
    max_features_in_rule: int = 5,
    random_seed: int = 42,
    eval_on_test_data: bool = False,
    X_test_unit: Optional[np.ndarray] = None,
    X_test_std: Optional[np.ndarray] = None,
    initial_window: Optional[float] = None,
    num_rollouts_per_instance: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate and generate anchors for multiple instances of a class.
    
    This function samples n_instances from the target class, generates anchors
    for each, and returns aggregated statistics.
    
    Args:
        X_test: Test instances (used to sample instances to explain)
        y_test: Test labels (used to sample instances to explain)
        trained_model: Trained PPO model
        make_env_fn: Function that creates an AnchorEnv instance
        feature_names: List of feature names
        target_class: Target class to evaluate
        n_instances: Number of instances to evaluate
        steps_per_episode: Maximum rollout steps
        max_features_in_rule: Maximum features in rule
        random_seed: Random seed for sampling
        eval_on_test_data: If True, compute metrics on test data instead of training data
        X_test_unit: Test data in unit space [0,1] (required if eval_on_test_data=True)
        X_test_std: Test data in standardized space (required if eval_on_test_data=True)
        initial_window: Initial window size for anchor box (default: matches training)
        num_rollouts_per_instance: Number of greedy rollouts per instance (default: 1)
                                  If > 1, metrics are averaged across rollouts for each instance
    
    Returns:
        Dictionary with aggregated metrics and individual results.
        Note: By default, coverage and precision are computed on training data.
        Set eval_on_test_data=True to compute on test data.
        If num_rollouts_per_instance > 1, metrics are averaged across rollouts per instance.
    """
    # Sample instances from target class
    rng = np.random.default_rng(random_seed)
    idx_cls = np.where(y_test == target_class)[0]
    
    if idx_cls.size == 0:
        return {
            "avg_precision": 0.0,
            "avg_hard_precision": 0.0,
            "avg_coverage": 0.0,
            "n_instances": 0,
            "individual_results": []
        }
    
    sel = rng.choice(idx_cls, size=min(n_instances, idx_cls.size), replace=False)
    
    # Extract normalization parameters once for all instances
    temp_env = make_env_fn()
    X_min = temp_env.X_min
    X_range = temp_env.X_range
    
    # Prepare test data if evaluation on test data is requested
    if eval_on_test_data:
        if X_test_unit is None or X_test_std is None:
            raise ValueError(
                "eval_on_test_data=True requires X_test_unit and X_test_std. "
                "These should be provided when calling evaluate_class."
            )
    
    # Evaluate each instance
    individual_results = []
    for i, instance_idx in enumerate(sel):
        result = evaluate_single_instance(
            X_instance=X_test[instance_idx],
            trained_model=trained_model,
            make_env_fn=make_env_fn,
            feature_names=feature_names,
            target_class=target_class,
            steps_per_episode=steps_per_episode,
            max_features_in_rule=max_features_in_rule,
            X_min=X_min,
            X_range=X_range,
            eval_on_test_data=eval_on_test_data,
            X_test_unit=X_test_unit,
            X_test_std=X_test_std,
            y_test=y_test,
            initial_window=initial_window,
            num_rollouts_per_instance=num_rollouts_per_instance,
        )
        result["instance_idx"] = int(instance_idx)
        individual_results.append(result)
    
    # Aggregate statistics
    avg_precision = np.mean([r["precision"] for r in individual_results])
    avg_hard_precision = np.mean([r["hard_precision"] for r in individual_results])
    avg_coverage = np.mean([r["coverage"] for r in individual_results])  # Backward compatibility
    
    # Aggregate local and global coverage separately
    local_coverages = [r.get("local_coverage", r.get("coverage", 0.0)) for r in individual_results]
    global_coverages = [r.get("global_coverage") for r in individual_results if r.get("global_coverage") is not None]
    avg_local_coverage = np.mean(local_coverages) if local_coverages else None
    avg_global_coverage = np.mean(global_coverages) if global_coverages else None
    
    # Find best anchor (by hard precision)
    best_result = max(individual_results, key=lambda r: r["hard_precision"])
    
    # Compute union coverage: how many unique test instances are covered by at least one anchor
    union_coverage = None
    if eval_on_test_data and X_test_unit is not None:
        # Check which test instances are covered by at least one anchor
        covered_mask = np.zeros(X_test_unit.shape[0], dtype=bool)
        for result in individual_results:
            lower = np.array(result["lower_bounds"])
            upper = np.array(result["upper_bounds"])
            # Check which test instances fall in this anchor box
            instance_mask = np.all(
                (X_test_unit >= lower) & (X_test_unit <= upper), axis=1
            )
            covered_mask |= instance_mask
        union_coverage = float(covered_mask.mean())
    
    return {
        "avg_precision": float(avg_precision),
        "avg_hard_precision": float(avg_hard_precision),
        "avg_coverage": float(avg_coverage),  # Backward compatibility
        "avg_local_coverage": float(avg_local_coverage) if avg_local_coverage is not None else None,
        "avg_global_coverage": float(avg_global_coverage) if avg_global_coverage is not None else None,
        "union_coverage": union_coverage,  # Union coverage across all anchors (test data only)
        "n_instances": len(individual_results),
        "best_rule": best_result["rule"],
        "best_precision": best_result["hard_precision"],
        "individual_results": individual_results,
        "data_source": individual_results[0].get("data_source", "training") if individual_results else "training",
    }


def evaluate_all_classes(
    X_test: np.ndarray,
    y_test: np.ndarray,
    trained_model: Union[PPO, DDPG, TD3, Dict[int, Any]],
    make_env_fn,
    feature_names: List[str],
    n_instances_per_class: int = 20,
    steps_per_episode: int = 100,
    max_features_in_rule: int = 5,
    random_seed: int = 42,
    eval_on_test_data: bool = False,
    X_test_unit: Optional[np.ndarray] = None,
    X_test_std: Optional[np.ndarray] = None,
    initial_window: Optional[float] = None,
    num_rollouts_per_instance: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate and generate anchors for all classes.
    
    Args:
        X_test: Test instances (used to sample instances to explain)
        y_test: Test labels (used to sample instances to explain)
        trained_model: Trained PPO model, DDPG model, or dict of DDPG trainers per class
        make_env_fn: Function that creates AnchorEnv instance
        feature_names: List of feature names
        n_instances_per_class: Number of instances per class to evaluate
        steps_per_episode: Maximum rollout steps
        max_features_in_rule: Maximum features in rule
        random_seed: Random seed for sampling
        eval_on_test_data: If True, compute metrics on test data instead of training data
        X_test_unit: Test data in unit space [0,1] (required if eval_on_test_data=True)
        X_test_std: Test data in standardized space (required if eval_on_test_data=True)
        initial_window: Initial window size for anchor box (default: matches training)
        num_rollouts_per_instance: Number of greedy rollouts per instance (default: 1)
                                  If > 1, metrics are averaged across rollouts for each instance
    
    Returns:
        Dictionary with per-class results and overall statistics.
        Note: By default, coverage and precision are computed on training data.
        Set eval_on_test_data=True to compute on test data.
        If num_rollouts_per_instance > 1, metrics are averaged across rollouts per instance.
    """
    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)
    
    # Handle DDPG trainers dict (per-class trainers)
    if isinstance(trained_model, dict):
        # DDPG/TD3: Use per-class trainers
        print(f"Evaluating anchors for {n_classes} classes with {n_instances_per_class} instances each (Continuous action per-class trainers)")
        
        per_class_results = {}
        for cls in unique_classes:
            cls_int = int(cls)
            if cls_int not in trained_model:
                print(f"  Warning: No trainer found for class {cls_int}, skipping...")
                continue
            
            print(f"\nEvaluating class {cls}...")
            # Get the continuous action trainer (DDPG/TD3) for this class
            ddpg_trainer = trained_model[cls_int]
            # Extract the model from the trainer
            cls_model = ddpg_trainer.model if hasattr(ddpg_trainer, 'model') else ddpg_trainer
            
            result = evaluate_class(
                X_test=X_test,
                y_test=y_test,
                trained_model=cls_model,
                make_env_fn=make_env_fn,
                feature_names=feature_names,
                target_class=cls_int,
                n_instances=n_instances_per_class,
                steps_per_episode=steps_per_episode,
                max_features_in_rule=max_features_in_rule,
                random_seed=random_seed,
                eval_on_test_data=eval_on_test_data,
                X_test_unit=X_test_unit,
                X_test_std=X_test_std,
                initial_window=initial_window,
                num_rollouts_per_instance=num_rollouts_per_instance,
            )
            per_class_results[cls_int] = result
            
            print(f"  Avg precision: {result['avg_hard_precision']:.3f}")
            print(f"  Avg coverage: {result['avg_coverage']:.3f}")
            print(f"  Best rule: {result['best_rule']}")
            
            # Show unique rules and their frequencies
            if 'individual_results' in result and len(result['individual_results']) > 0:
                # Extract all rules
                all_rules = [r['rule'] for r in result['individual_results']]
                
                # Count unique rules
                rule_counts = Counter(all_rules)
                n_unique = len(rule_counts)
                
                print(f"  Unique rules: {n_unique} out of {len(all_rules)} instances")
                
                # Show up to 5 most common rules
                if n_unique > 1:
                    print(f"  Top rules:")
                    for rule, count in rule_counts.most_common(5):
                        percentage = (count / len(all_rules)) * 100
                        print(f"    [{count}/{len(all_rules)} ({percentage:.1f}%)] {rule}")
    else:
        # PPO or single continuous action model (DDPG/TD3): Use same model for all classes
        print(f"Evaluating anchors for {n_classes} classes with {n_instances_per_class} instances each")
        
        per_class_results = {}
        for cls in unique_classes:
            print(f"\nEvaluating class {cls}...")
            result = evaluate_class(
                X_test=X_test,
                y_test=y_test,
                trained_model=trained_model,
                make_env_fn=make_env_fn,
                feature_names=feature_names,
                target_class=int(cls),
                n_instances=n_instances_per_class,
                steps_per_episode=steps_per_episode,
                max_features_in_rule=max_features_in_rule,
                random_seed=random_seed,
                eval_on_test_data=eval_on_test_data,
                X_test_unit=X_test_unit,
                X_test_std=X_test_std,
                initial_window=initial_window,
                num_rollouts_per_instance=num_rollouts_per_instance,
            )
            per_class_results[int(cls)] = result
        
        print(f"  Avg precision: {result['avg_hard_precision']:.3f}")
        print(f"  Avg coverage: {result['avg_coverage']:.3f}")
        print(f"  Best rule: {result['best_rule']}")
        
        # Show unique rules and their frequencies
        if 'individual_results' in result and len(result['individual_results']) > 0:
            # Extract all rules
            all_rules = [r['rule'] for r in result['individual_results']]
            
            # Count unique rules
            rule_counts = Counter(all_rules)
            n_unique = len(rule_counts)
            
            print(f"  Unique rules: {n_unique} out of {len(all_rules)} instances")
            
            # Show up to 5 most common rules
            if n_unique > 1:
                print(f"  Top rules:")
                for rule, count in rule_counts.most_common(5):
                    percentage = (count / len(all_rules)) * 100
                    print(f"    [{count}/{len(all_rules)} ({percentage:.1f}%)] {rule}")
    
    # Compute overall statistics
    overall_precision = np.mean([r["avg_hard_precision"] for r in per_class_results.values()])
    overall_coverage = np.mean([r["avg_coverage"] for r in per_class_results.values()])
    
    # Compute overall union coverage if test data evaluation was used
    overall_union_coverage = None
    if eval_on_test_data:
        union_coverages = [r.get("union_coverage") for r in per_class_results.values() if r.get("union_coverage") is not None]
        if union_coverages:
            overall_union_coverage = np.mean(union_coverages)
    
    # Get data source from first result
    data_source = list(per_class_results.values())[0].get("data_source", "training") if per_class_results else "training"
    
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Average precision across all classes: {overall_precision:.3f}")
    print(f"Average coverage across all classes: {overall_coverage:.3f}")
    if overall_union_coverage is not None:
        print(f"Average union coverage across all classes: {overall_union_coverage:.3f}")
    print(f"Metrics computed on: {data_source} data")
    
    return {
        "per_class_results": per_class_results,
        "overall_precision": float(overall_precision),
        "overall_coverage": float(overall_coverage),
        "overall_union_coverage": float(overall_union_coverage) if overall_union_coverage is not None else None,
        "n_classes": n_classes,
        "data_source": data_source,
    }


def load_trained_model(model_path: str, vec_env) -> PPO:
    """
    Load a trained PPO model.
    
    Args:
        model_path: Path to the saved model
        vec_env: Vectorized environment
    
    Returns:
        Loaded PPO model
    """
    model = PPO.load(model_path, env=vec_env)
    print(f"Loaded trained model from {model_path}")
    return model


def prepare_test_data_for_evaluation(
    X_test_scaled: np.ndarray,
    X_min: np.ndarray,
    X_range: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to prepare test data for evaluation.
    
    Converts test data from standardized space to unit space [0, 1] for evaluation.
    
    Args:
        X_test_scaled: Test data in standardized space
        X_min: Min values from training data (for normalization)
        X_range: Range values from training data (for normalization)
    
    Returns:
        Tuple of (X_test_unit, X_test_scaled) ready for evaluation
    """
    X_test_unit = (X_test_scaled - X_min) / X_range
    X_test_unit = np.clip(X_test_unit, 0.0, 1.0).astype(np.float32)
    return X_test_unit, X_test_scaled


def plot_rules_2d(
    eval_results: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    feature_indices: Optional[Tuple[int, int]] = None,
    output_path: str = "./output/visualizations/rules_2d_visualization.png",
    figsize: Tuple[int, int] = (14, 10),
    alpha_anchors: float = 0.3,
    alpha_points: float = 0.6,
    show_instances_used: bool = True,
    X_min: Optional[np.ndarray] = None,
    X_range: Optional[np.ndarray] = None,
) -> str:
    """
    Visualize anchor rules as 2D rectangles.
    
    Creates a 2D plot showing:
    - Data points colored by class
    - Anchor boxes (rules) as rectangles
    - Instances used for evaluation (highlighted)
    
    Args:
        eval_results: Results from evaluate_all_classes() containing rules and anchors
        X_test: Test data (in standardized space, used for plotting)
        y_test: Test labels
        feature_names: List of feature names
        class_names: Optional list of class names (for legend)
        feature_indices: Optional tuple (feat_idx1, feat_idx2) to specify which 2 features to plot
                       If None, auto-selects features that appear most frequently in rules
        output_path: Path to save the plot
        figsize: Figure size (width, height)
        alpha_anchors: Transparency for anchor rectangles (0-1)
        alpha_points: Transparency for data points (0-1)
        show_instances_used: If True, highlight instances used for evaluation
        X_min: Optional min values for converting bounds from unit space to standardized space
        X_range: Optional range values for converting bounds from unit space to standardized space
    
    Returns:
        Path to saved plot file
    """
    
    per_class_results = eval_results.get("per_class_results", {})
    if not per_class_results:
        raise ValueError("eval_results must contain per_class_results with anchors")
    
    # Auto-select features if not specified
    if feature_indices is None:
        # Count feature frequency in rules
        feature_counts = {}
        for cls_result in per_class_results.values():
            if "anchors" in cls_result:
                for anchor in cls_result["anchors"]:
                    lower = np.array(anchor.get("lower_bounds", []))
                    upper = np.array(anchor.get("upper_bounds", []))
                    if len(lower) > 0 and len(upper) > 0:
                        # Find features that were tightened (width < 0.95 of full range)
                        widths = upper - lower
                        tightened = np.where(widths < 0.95)[0]
                        for feat_idx in tightened:
                            feature_counts[feat_idx] = feature_counts.get(feat_idx, 0) + 1
        
        if len(feature_counts) >= 2:
            # Select top 2 most frequently tightened features
            top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:2]
            feature_indices = (top_features[0][0], top_features[1][0])
        elif len(feature_counts) == 1:
            # Use the one tightened feature and the first feature
            feature_indices = (list(feature_counts.keys())[0], 0)
        else:
            # Fallback: use first two features
            feature_indices = (0, min(1, len(feature_names) - 1))
    
    feat_idx1, feat_idx2 = feature_indices
    
    # Validate feature indices
    if feat_idx1 >= len(feature_names) or feat_idx2 >= len(feature_names):
        raise ValueError(f"Feature indices {feature_indices} out of range (max: {len(feature_names)-1})")
    
    # Get feature names for axes
    feat_name1 = feature_names[feat_idx1]
    feat_name2 = feature_names[feat_idx2]
    
    # Create figure with subplots for each class
    # Handle both integer keys (from evaluate_all_classes) and string keys (from JSON)
    unique_classes = []
    for k in per_class_results.keys():
        if isinstance(k, int):
            unique_classes.append(k)
        elif isinstance(k, str):
            # Handle "class_0" format
            if k.startswith("class_"):
                try:
                    unique_classes.append(int(k.split('_')[1]))
                except (ValueError, IndexError):
                    # Try to extract number from string
                    try:
                        unique_classes.append(int(k.replace("class_", "")))
                    except ValueError:
                        pass
            else:
                # Try to convert directly
                try:
                    unique_classes.append(int(k))
                except ValueError:
                    pass
    
    unique_classes = sorted(unique_classes)
    n_classes = len(unique_classes)
    
    if n_classes == 0:
        raise ValueError("No valid class keys found in per_class_results")
    
    # Determine grid layout
    if n_classes <= 2:
        n_rows, n_cols = 1, n_classes
    elif n_classes <= 4:
        n_rows, n_cols = 2, 2
    elif n_classes <= 6:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = 3, 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    # Color map for classes
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_classes))))
    
    # Plot each class
    for plot_idx, cls_int in enumerate(unique_classes):
        if plot_idx >= len(axes):
            break
        
        ax = axes[plot_idx]
        
        # Try to get class result - handle both integer and string keys
        cls_result = None
        if cls_int in per_class_results:
            cls_result = per_class_results[cls_int]
        elif f"class_{cls_int}" in per_class_results:
            cls_result = per_class_results[f"class_{cls_int}"]
        else:
            ax.text(0.5, 0.5, f"No data for class {cls_int}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Class {cls_int}")
            continue
        cls_name = class_names[cls_int] if class_names and cls_int < len(class_names) else f"Class {cls_int}"
        
        # Get anchors for this class
        anchors = cls_result.get("anchors", [])
        instance_indices_used = cls_result.get("instance_indices_used", [])
        
        # Plot all data points (colored by class)
        for other_cls in unique_classes:
            mask = y_test == other_cls
            if np.any(mask):
                ax.scatter(
                    X_test[mask, feat_idx1],
                    X_test[mask, feat_idx2],
                    c=[colors[other_cls % len(colors)]],
                    alpha=alpha_points * 0.5 if other_cls != cls_int else alpha_points,
                    s=20,
                    label=f"Class {other_cls}" if other_cls == cls_int else None,
                    edgecolors='none',
                    zorder=1
                )
        
        # Highlight instances used for evaluation
        if show_instances_used and len(instance_indices_used) > 0:
            used_mask = np.zeros(len(X_test), dtype=bool)
            for idx in instance_indices_used:
                if 0 <= idx < len(X_test):
                    used_mask[idx] = True
            
            if np.any(used_mask):
                ax.scatter(
                    X_test[used_mask, feat_idx1],
                    X_test[used_mask, feat_idx2],
                    c='red',
                    marker='x',
                    s=100,
                    linewidths=2,
                    label='Instances evaluated',
                    zorder=4,
                    alpha=0.8
                )
        
        # Draw anchor rectangles
        anchor_colors = plt.cm.Set3(np.linspace(0, 1, max(len(anchors), 1)))
        for anchor_idx, anchor in enumerate(anchors):
            lower = np.array(anchor.get("lower_bounds", []))
            upper = np.array(anchor.get("upper_bounds", []))
            
            if len(lower) == 0 or len(upper) == 0:
                continue
            
            # Get bounds for the two features we're plotting
            # Bounds are stored in unit space [0, 1], but X_test is in standardized space
            # We need to convert bounds from unit space to standardized space
            # Conversion: standardized = unit * X_range + X_min
            # But we don't have X_min and X_range here, so we'll use the data range
            # For visualization purposes, we can approximate using X_test range
            
            # Get data range for conversion
            # Use provided X_min/X_range if available, otherwise use X_test range
            if X_min is not None and X_range is not None:
                X_min_feat1 = X_min[feat_idx1]
                X_range_feat1 = X_range[feat_idx1]
                X_min_feat2 = X_min[feat_idx2]
                X_range_feat2 = X_range[feat_idx2]
            else:
                # Fallback: use X_test range (approximation)
                X_min_feat1 = X_test[:, feat_idx1].min()
                X_max_feat1 = X_test[:, feat_idx1].max()
                X_range_feat1 = X_max_feat1 - X_min_feat1
                
                X_min_feat2 = X_test[:, feat_idx2].min()
                X_max_feat2 = X_test[:, feat_idx2].max()
                X_range_feat2 = X_max_feat2 - X_min_feat2
            
            # Convert bounds from unit space [0,1] to standardized space
            if feat_idx1 < len(lower) and feat_idx1 < len(upper):
                # Bounds are in [0,1], convert to standardized space
                lower_feat1 = X_min_feat1 + lower[feat_idx1] * X_range_feat1
                upper_feat1 = X_min_feat1 + upper[feat_idx1] * X_range_feat1
            else:
                lower_feat1 = X_test[:, feat_idx1].min()
                upper_feat1 = X_test[:, feat_idx1].max()
            
            if feat_idx2 < len(lower) and feat_idx2 < len(upper):
                lower_feat2 = X_min_feat2 + lower[feat_idx2] * X_range_feat2
                upper_feat2 = X_min_feat2 + upper[feat_idx2] * X_range_feat2
            else:
                lower_feat2 = X_test[:, feat_idx2].min()
                upper_feat2 = X_test[:, feat_idx2].max()
            
            width1 = upper_feat1 - lower_feat1
            width2 = upper_feat2 - lower_feat2
            
            # Draw rectangle
            rect = Rectangle(
                (lower_feat1, lower_feat2),
                width1,
                width2,
                linewidth=2,
                edgecolor=colors[cls_int % len(colors)],
                facecolor=colors[cls_int % len(colors)],
                alpha=alpha_anchors,
                zorder=2
            )
            ax.add_patch(rect)
            
            # Add text label with precision/coverage if space allows
            if width1 > 0 and width2 > 0:
                prec = anchor.get("precision", 0.0)
                cov = anchor.get("coverage", 0.0)
                if prec > 0 or cov > 0:
                    ax.text(
                        lower_feat1 + width1/2,
                        lower_feat2 + width2/2,
                        f"P:{prec:.2f}\nC:{cov:.2f}",
                        ha='center',
                        va='center',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                        zorder=3
                    )
        
        ax.set_xlabel(feat_name1, fontsize=10)
        ax.set_ylabel(feat_name2, fontsize=10)
        ax.set_title(f"{cls_name}\n({len(anchors)} anchors)", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_classes, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(
        f'2D Visualization of Anchor Rules\nFeatures: {feat_name1} vs {feat_name2}',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_rules_2d_from_json(
    json_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    feature_indices: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
    **plot_kwargs
) -> str:
    """
    Visualize anchor rules from saved JSON file.
    
    Convenience function to load rules from JSON and create 2D visualization.
    
    Args:
        json_path: Path to metrics_and_rules.json file
        X_test: Test data (in standardized space)
        y_test: Test labels
        feature_names: List of feature names
        class_names: Optional list of class names
        feature_indices: Optional tuple (feat_idx1, feat_idx2) to specify which 2 features to plot
        output_path: Optional output path (defaults to same directory as JSON with _2d_plot suffix)
        **plot_kwargs: Additional arguments passed to plot_rules_2d()
    
    Returns:
        Path to saved plot file
    """
    import json
    
    # Load JSON
    with open(json_path, 'r') as f:
        metrics_data = json.load(f)
    
    # Convert to eval_results format
    # JSON has string keys like "class_0", need to convert to integer keys
    eval_results = {
        "per_class_results": {}
    }
    
    for cls_key, cls_data in metrics_data.get("per_class_results", {}).items():
        # Extract class integer from key (handle both "class_0" and integer keys)
        if isinstance(cls_key, int):
            cls_int = cls_key
        elif isinstance(cls_key, str) and cls_key.startswith("class_"):
            try:
                cls_int = int(cls_key.split('_')[1])
            except (ValueError, IndexError):
                try:
                    cls_int = int(cls_key.replace("class_", ""))
                except ValueError:
                    continue  # Skip invalid keys
        else:
            try:
                cls_int = int(cls_key)
            except ValueError:
                continue  # Skip invalid keys
        
        eval_results["per_class_results"][cls_int] = cls_data
    
    # Set default output path
    if output_path is None:
        import os
        base_dir = os.path.dirname(json_path)
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_2d_plot.png")
    
    # Create plot
    return plot_rules_2d(
        eval_results=eval_results,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        class_names=class_names,
        feature_indices=feature_indices,
        output_path=output_path,
        **plot_kwargs
    )

