"""
Post-training evaluation and inference for Dynamic Anchors.

This module provides functions to evaluate trained PPO/DDPG policies and generate
dynamic anchor explanations for individual instances and classes.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter
from stable_baselines3 import PPO, DDPG


def greedy_rollout(
    env,
    trained_model: Union[PPO, DDPG],
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
    
    # Detect model type: DDPG has actor/critic, PPO has policy
    is_ddpg = hasattr(trained_model, 'actor') and hasattr(trained_model, 'critic')
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
            if is_ddpg:
                # DDPG: action is already a numpy array (continuous)
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
    
    # A feature is "tightened" if its width is at least 2% smaller than initial width
    # This threshold is more lenient than 5% to catch smaller tightenings
    # Also consider features that are significantly smaller than full range (1.0)
    tightened = np.where((lw < initial_width * 0.98) | (lw < 0.98))[0]
    
    # If no features tightened by 2%, try a more lenient threshold (1%)
    if tightened.size == 0:
        tightened = np.where((lw < initial_width * 0.99) | (lw < 0.99))[0]
    
    # If still no tightened features, check if any features are smaller than full range
    # This handles cases where box started at full range and was tightened
    if tightened.size == 0:
        # Check if any feature is significantly smaller than 1.0 (full range)
        tightened = np.where(lw < 0.95)[0]
    
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
    trained_model: Union[PPO, DDPG],
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
    
    Returns:
        Dictionary with anchor explanation and metrics.
        Note: Coverage and precision are computed on training data by default.
        Set eval_on_test_data=True to compute on test data.
    """
    # Detect model type: DDPG has actor/critic, PPO has policy
    is_ddpg = hasattr(trained_model, 'actor') and hasattr(trained_model, 'critic')
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
    if is_ddpg:
        # DDPG: Use ContinuousAnchorEnv
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
    
    # Run greedy evaluation
    info, rule, lower, upper = greedy_rollout(
        env,
        trained_model,
        steps_per_episode=steps_per_episode,
        max_features_in_rule=max_features_in_rule
    )
    
    return {
        "rule": rule,
        "precision": info.get("precision", 0.0),
        "hard_precision": info.get("hard_precision", 0.0),
        "coverage": info.get("coverage", 0.0),
        "lower_bounds": lower.tolist(),
        "upper_bounds": upper.tolist(),
        "data_source": info.get("data_source", "training"),  # Indicates which dataset metrics are computed on
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
    
    Returns:
        Dictionary with aggregated metrics and individual results.
        Note: By default, coverage and precision are computed on training data.
        Set eval_on_test_data=True to compute on test data.
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
        )
        result["instance_idx"] = int(instance_idx)
        individual_results.append(result)
    
    # Aggregate statistics
    avg_precision = np.mean([r["precision"] for r in individual_results])
    avg_hard_precision = np.mean([r["hard_precision"] for r in individual_results])
    avg_coverage = np.mean([r["coverage"] for r in individual_results])
    
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
        "avg_coverage": float(avg_coverage),
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
    trained_model: Union[PPO, DDPG, Dict[int, Any]],
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
    
    Returns:
        Dictionary with per-class results and overall statistics.
        Note: By default, coverage and precision are computed on training data.
        Set eval_on_test_data=True to compute on test data.
    """
    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)
    
    # Handle DDPG trainers dict (per-class trainers)
    if isinstance(trained_model, dict):
        # DDPG: Use per-class trainers
        print(f"Evaluating anchors for {n_classes} classes with {n_instances_per_class} instances each (DDPG per-class trainers)")
        
        per_class_results = {}
        for cls in unique_classes:
            cls_int = int(cls)
            if cls_int not in trained_model:
                print(f"  Warning: No trainer found for class {cls_int}, skipping...")
                continue
            
            print(f"\nEvaluating class {cls}...")
            # Get the DDPG trainer for this class
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
        # PPO or single DDPG model: Use same model for all classes
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

