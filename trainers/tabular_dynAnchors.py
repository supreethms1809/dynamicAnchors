"""
Complete Dynamic Anchors pipeline for tabular data using Stable-Baselines3.

This module provides an end-to-end pipeline for training and evaluating
dynamic anchor explanations on tabular classification data.

Usage example:
    from trainers.tabular_dynAnchors import train_and_evaluate_dynamic_anchors
    
    results = train_and_evaluate_dynamic_anchors(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        classifier=classifier,
        target_classes=(0, 1, 2),
        n_envs=4,
        total_timesteps=50000,
    )
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
import os
import json
from functools import partial


def train_and_evaluate_dynamic_anchors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    classifier: torch.nn.Module,
    target_classes: Tuple[int, ...] = None,
    device: str = "cpu",
    # Training parameters
    n_envs: int = 4,
    total_timesteps: int = 50000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    # Continuous action parameters
    use_continuous_actions: bool = False,
    continuous_algorithm: str = "ddpg",
    continuous_learning_rate: float = 5e-5,
    # Environment parameters
    use_perturbation: bool = True,
    perturbation_mode: str = "bootstrap",  # "bootstrap", "uniform", or "adaptive"
    n_perturb: int = 1024,
    step_fracs: Tuple[float, ...] = (0.005, 0.01, 0.02),
    min_width: float = 0.05,
    precision_target: float = 0.95,
    coverage_target: float = 0.02,
    # Evaluation parameters
    n_eval_instances_per_class: int = 20,
    max_features_in_rule: int = 5,
    steps_per_episode: int = 100,
    eval_steps_per_episode: int = None,  # Defaults to steps_per_episode if None
    num_rollouts_per_instance: int = 1,  # Number of greedy rollouts per instance (default: 1)
    # Output parameters
    output_dir: str = "./output/anchors/",
    save_checkpoints: bool = True,
    checkpoint_freq: int = 10000,
    eval_freq: int = 5000,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Complete pipeline for training and evaluating dynamic anchors.
    
    This function:
    1. Prepares data and creates environments
    2. Trains RL policy (PPO for discrete actions, TD3/DDPG for continuous actions) to generate anchors
    3. Evaluates on test instances to compute precision/coverage
    4. Returns results and trained models
    
    Args:
        X_train: Training features (will be standardized)
        y_train: Training labels
        X_test: Test features (for evaluation)
        y_test: Test labels
        feature_names: Names of features
        classifier: Trained PyTorch classifier
        target_classes: Classes to generate anchors for (None = all classes)
        device: Device to use ("cpu", "cuda", "auto")
        n_envs: Number of parallel training environments (only used for PPO/discrete actions)
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO (discrete actions)
        n_steps: Steps per environment before update (PPO only)
        batch_size: Batch size for PPO updates (PPO only)
        n_epochs: PPO epochs per update (PPO only)
        use_continuous_actions: If True, use continuous actions (TD3/DDPG) instead of discrete (PPO)
        continuous_algorithm: "ddpg" or "td3" (only used if use_continuous_actions=True)
        continuous_learning_rate: Learning rate for TD3/DDPG (continuous actions)
        use_perturbation: Enable perturbation sampling in environment
        perturbation_mode: "bootstrap", "uniform", or "adaptive" sampling.
            - "bootstrap": Resample empirical points with replacement (requires points in box)
            - "uniform": Generate uniform samples within box bounds (works even with 0 points)
            - "adaptive": Use bootstrap when plenty of points, uniform when sparse (recommended)
        n_perturb: Number of perturbation samples
        step_fracs: Action step sizes
        min_width: Minimum box width
        precision_target: Target precision threshold
        coverage_target: Target coverage threshold
        n_eval_instances_per_class: Instances per class for evaluation
        max_features_in_rule: Max features to show in rules
        steps_per_episode: Max steps for greedy rollouts
        output_dir: Directory for outputs
        save_checkpoints: Save checkpoints during training
        checkpoint_freq: Checkpoint frequency
        eval_freq: Evaluation frequency
        verbose: Verbosity level
    
    Returns:
        Dictionary with:
            - trained_model: RL model (PPO model or dict of TD3/DDPG trainers per class)
            - trainer: Trainer instance
            - eval_results: Per-class evaluation results
            - overall_stats: Overall precision/coverage
            - metadata: Configuration and setup info
    """
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from trainers.vecEnv import AnchorEnv, make_dummy_vec_env, ContinuousAnchorEnv
    from trainers.PPO_trainer import train_ppo_model
    from trainers.dynAnchors_inference import evaluate_all_classes, evaluate_all_classes_class_level
    from trainers.device_utils import get_device_pair
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Standardize device handling: get both object and string
    device_obj, device_str = get_device_pair(device)
    
    # Prepare data
    print("Preparing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # Normalize to [0,1] for environment
    X_min = X_train_scaled.min(axis=0)
    X_max = X_train_scaled.max(axis=0)
    X_range = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))
    X_unit_train = (X_train_scaled - X_min) / X_range
    X_unit_test = (X_test_scaled - X_min) / X_range
    
    # Prepare test data in unit space for optional test-set evaluation (consistent with joint training)
    X_test_unit = (X_test_scaled - X_min) / X_range
    
    # Determine target classes
    unique_classes = np.unique(y_train)
    if target_classes is None:
        target_classes = tuple(unique_classes)
    else:
        target_classes = tuple(target_classes)
    
    print(f"Classes: {unique_classes}, Target classes: {target_classes}")
    
    # ======================================================================
    # CLUSTER-BASED SAMPLING: Compute cluster centroids per class
    # This identifies dense regions and uses their centroids as starting points
    # This can improve training by starting from more representative examples
    # ======================================================================
    from trainers.vecEnv import compute_cluster_centroids_per_class
    
    n_clusters_per_class = 10  # Number of cluster centroids per class
    print(f"\n[Cluster-Based Sampling] Computing {n_clusters_per_class} cluster centroids per class...")
    try:
        cluster_centroids_per_class = compute_cluster_centroids_per_class(
            X_unit=X_unit_train,
            y=y_train,
            n_clusters_per_class=n_clusters_per_class,
            random_state=42
        )
        print(f"  Cluster centroids computed successfully!")
        for cls in target_classes:
            if cls in cluster_centroids_per_class:
                n_centroids = len(cluster_centroids_per_class[cls])
                print(f"  Class {cls}: {n_centroids} cluster centroids")
            else:
                print(f"  Class {cls}: No centroids available")
    except ImportError as e:
        print(f"  WARNING: Could not compute cluster centroids: {e}")
        print(f"  Falling back to random sampling. Install sklearn: pip install scikit-learn")
        cluster_centroids_per_class = None
    
    # ======================================================================
    # FIXED INSTANCE SAMPLING: Pre-select instances per class for training
    # This reduces variance by using consistent starting points (fallback)
    # ======================================================================
    n_fixed_instances_per_class = 10  # Number of fixed instances per class
    rng_fixed = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
    fixed_instances_per_class = {}
    
    print(f"\n[Fixed Instance Sampling] Pre-selecting {n_fixed_instances_per_class} instances per class (fallback)...")
    for cls in target_classes:
        cls_indices = np.where(y_train == cls)[0]
        if len(cls_indices) > 0:
            n_sample = min(n_fixed_instances_per_class, len(cls_indices))
            fixed_indices = rng_fixed.choice(cls_indices, size=n_sample, replace=False)
            fixed_instances_per_class[cls] = fixed_indices
            print(f"  Class {cls}: Selected {len(fixed_indices)} instances (indices: {fixed_indices[:5].tolist()}...)")
        else:
            fixed_instances_per_class[cls] = np.array([], dtype=int)
            print(f"  Class {cls}: No instances available")
    
    # Create environment factory function
    # For multi-class training, distribute classes across environments
    def create_anchor_env(target_cls=None):
        """Helper to create AnchorEnv with a specific target class."""
        if target_cls is None:
            target_cls = target_classes[0]  # Default to first class
        env = AnchorEnv(
            X_unit=X_unit_train,
            X_std=X_train_scaled,
            y=y_train,
            feature_names=feature_names,
            classifier=classifier,
            device=device_str,  # Pass string to environment
            target_class=target_cls,
            step_fracs=step_fracs,
            min_width=min_width,
            alpha=0.7,
            beta=0.6,
            gamma=0.1,
            precision_target=precision_target,
            coverage_target=coverage_target,
            precision_blend_lambda=0.5,
            drift_penalty_weight=0.05,
            use_perturbation=use_perturbation,
            perturbation_mode=perturbation_mode,
            n_perturb=n_perturb,
            X_min=X_min,
            X_range=X_range,
            min_coverage_floor=0.005,
            js_penalty_weight=0.05,
        )
        # Set cluster centroids for this class (priority: cluster centroids > fixed instances > random)
        # Note: cluster_centroids_per_class and fixed_instances_per_class are set in outer scope
        if cluster_centroids_per_class is not None:
            env.cluster_centroids_per_class = cluster_centroids_per_class
        env.fixed_instances_per_class = fixed_instances_per_class
        # Enable continuous actions if requested
        if use_continuous_actions:
            env.max_action_scale = max(step_fracs) if step_fracs else 0.02
            env.min_absolute_step = max(0.05, min_width * 0.5)
        return env
    
    # Create default factory (for evaluation and single-class training)
    def make_anchor_env():
        return create_anchor_env()
    
    # Create vectorized environment with different target classes if multi-class
    print(f"\nCreating {n_envs} parallel environments...")
    if len(target_classes) > 1 and n_envs > 1:
        # For multi-class, create separate factories for each target class
        from trainers.vecEnv import DummyVecEnv, make_dynamic_anchor_env
        from functools import partial
        
        env_fns = []
        for i in range(n_envs):
            target_cls = target_classes[i % len(target_classes)]
            # Create factory for this specific target class using partial
            factory_fn = partial(create_anchor_env, target_cls)
            # Capture factory_fn correctly with default argument
            env_fns.append(lambda i=i, f=factory_fn, s=42+i: make_dynamic_anchor_env(f, seed=s))
        vec_env = DummyVecEnv(env_fns)
    else:
        # Single class or single env - use standard approach
        vec_env = make_dummy_vec_env(make_anchor_env, n_envs=n_envs, seed=42)
    
    # Train model (PPO for discrete actions, TD3/DDPG for continuous actions)
    continuous_trainers = None  # Initialize for use in evaluation
    if use_continuous_actions:
        continuous_algorithm_lower = continuous_algorithm.lower()
        if continuous_algorithm_lower == "td3":
            algorithm_name = "TD3"
            from trainers.TD3_trainer import DynamicAnchorTD3Trainer, create_td3_trainer
            print(f"\nTraining TD3 for {total_timesteps} timesteps...")
        else:
            algorithm_name = "DDPG"
            from trainers.DDPG_trainer import DynamicAnchorDDPGTrainer, create_ddpg_trainer
            print(f"\nTraining DDPG for {total_timesteps} timesteps...")
        
        # For continuous actions, we need to train per class (DDPG/TD3 don't use vectorized envs the same way)
        # Create trainers for each target class
        continuous_trainers = {}
        gym_envs = {}  # Store gym_envs separately to use directly (not through trainer.env which is wrapped)
        for cls in target_classes:
            # Create AnchorEnv for this class
            anchor_env = create_anchor_env(target_cls=cls)
            # Set cluster centroids for this class (priority: cluster centroids > fixed instances > random)
            if cluster_centroids_per_class is not None:
                anchor_env.cluster_centroids_per_class = cluster_centroids_per_class
            # Set fixed instances for this class (for fixed instance sampling fallback)
            anchor_env.fixed_instances_per_class = fixed_instances_per_class
            
            # Wrap with ContinuousAnchorEnv for TD3/DDPG (provides action_space attribute)
            gym_env = ContinuousAnchorEnv(anchor_env, seed=42 + cls)
            gym_envs[cls] = gym_env  # Store for direct use
            
            if continuous_algorithm_lower == "td3":
                cls_trainer = create_td3_trainer(
                    env=gym_env,
                    learning_rate=continuous_learning_rate,
                    device=device_str,
                    verbose=verbose,
                )
            else:
                cls_trainer = create_ddpg_trainer(
                    env=gym_env,
                    learning_rate=continuous_learning_rate,
                    device=device_str,
                    verbose=verbose,
                )
            continuous_trainers[cls] = cls_trainer
        
        # Manual training loop for continuous actions (similar to joint training)
        # Calculate steps per class
        steps_per_class = total_timesteps // len(target_classes)
        
        for cls in target_classes:
            print(f"\nTraining {algorithm_name} for class {cls} ({steps_per_class} timesteps)...")
            cls_trainer = continuous_trainers[cls]
            gym_env = gym_envs[cls]  # Use the unwrapped ContinuousAnchorEnv directly
            
            # Handle gymnasium vs gym reset() return format
            try:
                import gymnasium as gym
                GYM_VERSION = "gymnasium"
            except ImportError:
                try:
                    import gym
                    GYM_VERSION = "gym"
                except ImportError:
                    GYM_VERSION = "gym"
            
            # Manual training loop
            # Reset environment - reset() returns observation that already includes [lower, upper, precision, coverage]
            if GYM_VERSION == "gymnasium":
                reset_result, reset_info = gym_env.reset(seed=42 + cls)
            else:
                reset_result = gym_env.reset(seed=42 + cls)
                if isinstance(reset_result, tuple):
                    reset_result, reset_info = reset_result
                else:
                    reset_info = {}
            
            # Use reset_result directly - it already has the correct shape: (2 * n_features + 2,)
            # This matches how we use anchor_state from step()
            obs = np.array(reset_result, dtype=np.float32).flatten()
            
            # Calculate expected observation dimensions once
            expected_shape = (2 * gym_env.n_features + 2,)
            expected_len = 2 * gym_env.n_features + 2
            
            # Validate initial observation shape
            if obs.shape != expected_shape or len(obs) != expected_len:
                raise ValueError(
                    f"Initial observation shape mismatch: obs shape={obs.shape}, len={len(obs)}, "
                    f"expected shape={expected_shape}, len={expected_len}"
                )
            
            # Ensure classifier is in eval mode
            if hasattr(gym_env.anchor_env.classifier, 'eval'):
                gym_env.anchor_env.classifier.eval()
            
            for step in range(steps_per_class):
                # Ensure obs is the right shape before predict
                obs = np.array(obs, dtype=np.float32).flatten()
                
                # Validate obs length before predict
                if len(obs) != expected_len:
                    print(f"ERROR: obs has wrong length before predict at step {step}!")
                    print(f"  obs length: {len(obs)}, expected: {expected_len}")
                    print(f"  obs shape: {obs.shape}")
                    raise ValueError(
                        f"Observation length mismatch before predict at step {step}: "
                        f"obs len={len(obs)}, expected len={expected_len}"
                    )
                
                # Reshape to expected shape
                obs = obs.reshape(expected_shape)
                
                # Get action - ensure we pass flattened obs to predict
                obs_for_predict = np.array(obs, dtype=np.float32).flatten()
                action, _ = cls_trainer.predict(obs_for_predict, deterministic=False)
                
                # Step environment using the wrapper's step() method (same as joint training)
                # This ensures observations are properly formatted with correct shape
                if GYM_VERSION == "gymnasium":
                    next_obs, reward, terminated, truncated, step_info = gym_env.step(action)
                    done = terminated or truncated
                else:
                    step_result = gym_env.step(action)
                    if len(step_result) == 5:
                        next_obs, reward, terminated, truncated, step_info = step_result
                        done = terminated or truncated
                    else:
                        next_obs, reward, done, step_info = step_result
                
                # next_obs from gym_env.step() should already have correct shape (2 * n_features + 2,)
                # because ContinuousAnchorEnv.step() calls anchor_env.step() and returns the state
                # which includes [lower, upper, precision, coverage]
                next_obs = np.array(next_obs, dtype=np.float32).flatten()
                
                # CRITICAL: Validate observation length immediately after step()
                if len(next_obs) != expected_len:
                    print(f"ERROR: next_obs from gym_env.step() has wrong length at step {step}!")
                    print(f"  Expected length: {expected_len}")
                    print(f"  Actual length: {len(next_obs)}")
                    print(f"  next_obs shape: {next_obs.shape if hasattr(next_obs, 'shape') else 'no shape'}")
                    print(f"  n_features: {gym_env.n_features}")
                    print(f"  next_obs content (first 20): {next_obs[:20]}")
                    # Fallback: construct manually from bounds if wrapper returns wrong shape
                    precision = step_info.get('precision', 0.0) if isinstance(step_info, dict) else 0.0
                    coverage = step_info.get('coverage', 0.0) if isinstance(step_info, dict) else 0.0
                    lower = np.array(gym_env.anchor_env.lower.copy(), dtype=np.float32).flatten()
                    upper = np.array(gym_env.anchor_env.upper.copy(), dtype=np.float32).flatten()
                    next_obs = np.concatenate([
                        lower,
                        upper,
                        np.array([precision, coverage], dtype=np.float32)
                    ]).astype(np.float32).flatten()
                    print(f"  Fallback construction: len={len(next_obs)}")
                    if len(next_obs) != expected_len:
                        raise ValueError(
                            f"Failed to construct next_obs with correct shape at step {step}: "
                            f"expected {expected_len}, got {len(next_obs)}"
                        )
                
                # Ensure next_obs has the correct shape tuple
                next_obs = next_obs.reshape(expected_shape)
                
                # Validate observation shapes match before adding to replay buffer
                if obs.shape != next_obs.shape or len(obs.flatten()) != len(next_obs.flatten()):
                    print(f"ERROR: Observation shape mismatch at step {step}!")
                    print(f"  obs shape: {obs.shape}, len: {len(obs.flatten())}")
                    print(f"  next_obs shape: {next_obs.shape}, len: {len(next_obs.flatten())}")
                    print(f"  Expected shape: {expected_shape}, len: {expected_len}")
                    raise ValueError(
                        f"Observation shape mismatch at step {step}: "
                        f"obs shape={obs.shape} len={len(obs.flatten())}, "
                        f"next_obs shape={next_obs.shape} len={len(next_obs.flatten())}. "
                        f"Expected shape: {expected_shape} len: {expected_len}"
                    )
                
                # Final validation right before adding to replay buffer
                # Ensure both are 1D arrays with correct length
                obs_final = np.array(obs, dtype=np.float32).flatten()
                next_obs_final = np.array(next_obs, dtype=np.float32).flatten()
                
                # CRITICAL: Double-check lengths match before calling add_to_replay_buffer
                if len(obs_final) != expected_len:
                    raise ValueError(
                        f"obs_final has wrong length: {len(obs_final)}, expected {expected_len}"
                    )
                if len(next_obs_final) != expected_len:
                    raise ValueError(
                        f"next_obs_final has wrong length: {len(next_obs_final)}, expected {expected_len}"
                    )
                if len(obs_final) != len(next_obs_final):
                    raise ValueError(
                        f"Length mismatch: obs_final len={len(obs_final)}, "
                        f"next_obs_final len={len(next_obs_final)}"
                    )
                
                # DEBUG: Print shapes right before add_to_replay_buffer (only first few steps)
                if step < 3:
                    print(f"DEBUG step {step}: obs_final.shape={obs_final.shape}, len={len(obs_final)}, "
                          f"next_obs_final.shape={next_obs_final.shape}, len={len(next_obs_final)}, "
                          f"expected_len={expected_len}")
                
                # Add to replay buffer - ensure arrays are fresh copies with explicit copy()
                # CRITICAL: Use .copy() to ensure we have independent arrays that won't be modified
                obs_final_copy = np.array(obs_final.copy(), dtype=np.float32).flatten()
                next_obs_final_copy = np.array(next_obs_final.copy(), dtype=np.float32).flatten()
                
                # Final check on copies - validate lengths match
                if len(obs_final_copy) != expected_len:
                    raise ValueError(
                        f"Copy validation failed: obs_final_copy len={len(obs_final_copy)}, expected={expected_len}"
                    )
                if len(next_obs_final_copy) != expected_len:
                    raise ValueError(
                        f"Copy validation failed: next_obs_final_copy len={len(next_obs_final_copy)}, expected={expected_len}"
                    )
                if len(obs_final_copy) != len(next_obs_final_copy):
                    raise ValueError(
                        f"Copy validation failed: obs_final_copy len={len(obs_final_copy)} != "
                        f"next_obs_final_copy len={len(next_obs_final_copy)}"
                    )
                
                # Ensure arrays are contiguous and have correct dtype before passing
                obs_final_copy = np.ascontiguousarray(obs_final_copy, dtype=np.float32)
                next_obs_final_copy = np.ascontiguousarray(next_obs_final_copy, dtype=np.float32)
                
                cls_trainer.add_to_replay_buffer(
                    obs=obs_final_copy,
                    next_obs=next_obs_final_copy,
                    action=action,
                    reward=reward,
                    done=done,
                    info=step_info
                )
                
                # Train if enough samples
                if cls_trainer.model.replay_buffer.size() > cls_trainer.model.learning_starts:
                    cls_trainer.train_step(gradient_steps=1)
                
                if done:
                    # Reset environment - reset() returns observation that already includes [lower, upper, precision, coverage]
                    if GYM_VERSION == "gymnasium":
                        reset_result, reset_info = gym_env.reset(seed=42 + cls + step)
                    else:
                        reset_result = gym_env.reset(seed=42 + cls + step)
                        if isinstance(reset_result, tuple):
                            reset_result, reset_info = reset_result
                        else:
                            reset_info = {}
                    
                    # Use reset_result directly - it already has the correct shape: (2 * n_features + 2,)
                    obs = np.array(reset_result, dtype=np.float32).flatten()
                    
                    # Validate shape
                    if len(obs) != expected_len:
                        raise ValueError(
                            f"Reset observation length mismatch: obs len={len(obs)}, "
                            f"expected len={expected_len}"
                        )
                    obs = obs.reshape(expected_shape)
                else:
                    # Use next_obs as-is, but ensure it has correct shape
                    obs = np.array(next_obs, dtype=np.float32).flatten()
                    if len(obs) != expected_len:
                        raise ValueError(
                            f"obs from next_obs has wrong length: {len(obs)}, expected {expected_len}"
                        )
                    obs = obs.reshape(expected_shape)
                
                if (step + 1) % (steps_per_class // 10) == 0:
                    print(f"  Progress: {step + 1}/{steps_per_class} steps")
        
        # For evaluation, we'll use the continuous_trainers dict
        trainer = type('TrainerWrapper', (), {
            'model': continuous_trainers,  # Dict of trainers per class
        })()
    else:
        print(f"\nTraining PPO for {total_timesteps} timesteps...")
        # Note: PPO will use the same device as the classifier
        trainer = train_ppo_model(
            vec_env=vec_env,
            total_timesteps=total_timesteps,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            output_dir=output_dir,
            save_checkpoints=save_checkpoints,
            checkpoint_freq=checkpoint_freq,
            eval_freq=eval_freq,
            verbose=verbose,
            device=device_str,  # Pass device to PPO trainer
        )
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    print(f"Note: By default, coverage and precision are computed on training data.")
    print(f"      This explains the classifier's behavior on training data.")
    print(f"      To evaluate on test data, set eval_on_test_data=True in evaluate_all_classes.")
    
    # Instance-level evaluation (one anchor per test instance, like static anchors)
    print(f"\n[Instance-Level Evaluation] Creating one anchor per test instance...")
    
    # For continuous actions, pass the dict of trainers; for PPO, pass the model
    trained_model_for_eval = continuous_trainers if use_continuous_actions else trainer.model
    
    # Use eval_steps_per_episode if provided, otherwise use steps_per_episode (consistent with joint training)
    eval_steps = eval_steps_per_episode if eval_steps_per_episode is not None else steps_per_episode
    
    eval_results_instance = evaluate_all_classes(
        X_test=X_test_scaled,
        y_test=y_test,
        trained_model=trained_model_for_eval,
        make_env_fn=create_anchor_env,  # Use create_anchor_env which accepts target_cls
        feature_names=feature_names,
        n_instances_per_class=n_eval_instances_per_class,
        max_features_in_rule=max_features_in_rule,
        steps_per_episode=eval_steps,
        random_seed=42,
        eval_on_test_data=False,  # Default: use training data (consistent with joint training)
        X_test_unit=X_test_unit if False else None,  # Consistent with joint training pattern
        X_test_std=X_test_scaled if False else None,  # Consistent with joint training pattern
        num_rollouts_per_instance=num_rollouts_per_instance,  # Consistent with joint training
    )
    
    # Class-level evaluation (one anchor per class)
    print(f"\n[Class-Level Evaluation] Creating one anchor per class...")
    eval_results_class = evaluate_all_classes_class_level(
        trained_model=trained_model_for_eval,
        make_env_fn=create_anchor_env,  # Use create_anchor_env which accepts target_cls
        feature_names=feature_names,
        target_classes=list(target_classes),
        steps_per_episode=eval_steps,
        max_features_in_rule=max_features_in_rule,
        random_seed=42,
        eval_on_test_data=False,  # Default: use training data (consistent with joint training)
        X_test_unit=X_test_unit if False else None,  # Consistent with joint training pattern
        X_test_std=X_test_scaled if False else None,  # Consistent with joint training pattern
        y_test=y_test if False else None,  # Consistent with joint training pattern
    )
    
    # Combine both evaluation results
    eval_results = {
        "instance_level": eval_results_instance,
        "class_level": eval_results_class,
    }
    
    # Close vectorized environment only for PPO (discrete actions)
    if not use_continuous_actions:
        vec_env.close()
    
    # Prepare results
    results = {
        "trained_model": trainer.model,
        "trainer": trainer,
        "eval_results": eval_results,  # Contains both instance_level and class_level
        "overall_stats": {
            "instance_level": {
                "avg_precision": eval_results_instance.get("overall_precision", 0.0),
                "avg_coverage": eval_results_instance.get("overall_coverage", 0.0),
            },
            "class_level": {
                "avg_precision": eval_results_class.get("overall_precision", 0.0),
                "avg_coverage": eval_results_class.get("overall_coverage", 0.0),
            },
        },
        "metadata": {
            "n_classes": len(target_classes),
            "n_features": len(feature_names),
            "target_classes": target_classes,
            "feature_names": feature_names,
            "output_dir": output_dir,
        }
    }
    
    print(f"\nTraining and evaluation complete!")
    print(f"\n{'='*80}")
    print("FINAL EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"\n[Instance-Level Evaluation] (One anchor per test instance, like static anchors):")
    print(f"  Overall Precision: {eval_results_instance.get('overall_precision', 0.0):.3f}")
    print(f"  Overall Coverage: {eval_results_instance.get('overall_coverage', 0.0):.3f}")
    print(f"\n[Class-Level Evaluation] (One anchor per class, dynamic anchors advantage):")
    print(f"  Overall Precision: {eval_results_class.get('overall_precision', 0.0):.3f}")
    print(f"  Overall Coverage: {eval_results_class.get('overall_coverage', 0.0):.3f}")
    
    # ======================================================================
    # Save metrics and rules to JSON (for non-joint training)
    # ======================================================================
    print(f"\n{'='*80}")
    print("SAVING METRICS AND RULES TO JSON")
    print(f"{'='*80}")
    
    # Prepare comprehensive metrics and rules data (without training_history for non-joint)
    metrics_data = {
        "overall_statistics": {
            # Instance-level (one anchor per instance, like static anchors)
            "instance_level": {
                "overall_precision": float(eval_results_instance.get("overall_precision", 0.0)),
                "overall_coverage": float(eval_results_instance.get("overall_coverage", 0.0)),
                "overall_n_points": int(eval_results_instance.get("overall_n_points", 0)),
            },
            # Class-level (one anchor per class, dynamic anchors advantage)
            "class_level": {
                "overall_precision": float(eval_results_class.get("overall_precision", 0.0)),
                "overall_coverage": float(eval_results_class.get("overall_coverage", 0.0)),
            },
        },
        "per_class_results": {
            "instance_level": {},
            "class_level": {},
        },
        "metadata": {
            "n_classes": len(target_classes),
            "n_features": len(feature_names),
            "target_classes": target_classes,
            "feature_names": feature_names,
            "output_dir": output_dir,
            "total_timesteps": total_timesteps,
            "use_continuous_actions": use_continuous_actions,
            "algorithm": continuous_algorithm.upper() if use_continuous_actions else "PPO",
        },
    }
    
    # Add per-class results from instance-level evaluation
    for cls_int in target_classes:
        if cls_int in eval_results_instance.get("per_class_results", {}):
            cls_result = eval_results_instance["per_class_results"][cls_int]
            
            # Extract rules and instance information from individual_results
            rules_list = []
            rules_with_instances = []
            anchors_list = []
            instance_indices_used = []
            
            if "individual_results" in cls_result:
                for individual_result in cls_result["individual_results"]:
                    instance_idx = int(individual_result.get("instance_idx", -1))
                    rule = individual_result.get("rule", "")
                    
                    if instance_idx >= 0:
                        instance_indices_used.append(instance_idx)
                    
                    rules_list.append(rule)
                    
                    rules_with_instances.append({
                        "instance_idx": instance_idx,
                        "rule": rule,
                        "precision": float(individual_result.get("precision", 0.0)),
                        "hard_precision": float(individual_result.get("hard_precision", individual_result.get("precision", 0.0))),
                        "coverage": float(individual_result.get("coverage", 0.0)),
                        "n_points": int(individual_result.get("n_points", 0)),
                    })
                    
                    anchors_list.append({
                        "instance_idx": instance_idx,
                        "lower_bounds": individual_result.get("lower_bounds", []),
                        "upper_bounds": individual_result.get("upper_bounds", []),
                        "precision": float(individual_result.get("precision", 0.0)),
                        "hard_precision": float(individual_result.get("hard_precision", individual_result.get("precision", 0.0))),
                        "coverage": float(individual_result.get("coverage", 0.0)),
                        "n_points": int(individual_result.get("n_points", 0)),
                        "rule": rule,
                    })
            
            unique_rules = list(set([r for r in rules_list if r]))
            unique_rules_count = len(unique_rules)
            
            metrics_data["per_class_results"]["instance_level"][f"class_{cls_int}"] = {
                "precision": float(cls_result.get("precision", cls_result.get("avg_precision", 0.0))),
                "hard_precision": float(cls_result.get("hard_precision", cls_result.get("avg_hard_precision", cls_result.get("precision", 0.0)))),
                "coverage": float(cls_result.get("coverage", cls_result.get("avg_coverage", 0.0))),
                "n_points": int(cls_result.get("n_points", 0)),
                "n_instances_evaluated": int(cls_result.get("n_instances", len(anchors_list))),
                "best_rule": cls_result.get("best_rule", ""),
                "best_precision": float(cls_result.get("best_precision", 0.0)),
                "rules": rules_list,
                "rules_with_instances": rules_with_instances,
                "unique_rules": unique_rules,
                "unique_rules_count": unique_rules_count,
                "instance_indices_used": instance_indices_used,
                "anchors": anchors_list,
            }
    
    # Add per-class results from class-level evaluation
    for cls_int in target_classes:
        if cls_int in eval_results_class.get("per_class_results", {}):
            cls_result = eval_results_class["per_class_results"][cls_int]
            
            metrics_data["per_class_results"]["class_level"][f"class_{cls_int}"] = {
                "precision": float(cls_result.get("precision", 0.0)),
                "hard_precision": float(cls_result.get("hard_precision", cls_result.get("precision", 0.0))),
                "coverage": float(cls_result.get("coverage", 0.0)),
                "global_coverage": float(cls_result.get("global_coverage", cls_result.get("coverage", 0.0))),
                "rule": cls_result.get("rule", ""),
                "lower_bounds": cls_result.get("lower_bounds", []),
                "upper_bounds": cls_result.get("upper_bounds", []),
                "evaluation_type": cls_result.get("evaluation_type", "class_level"),
            }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        """Recursively convert numpy arrays and other non-serializable types to JSON-compatible types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    # Convert all data to JSON-serializable format
    metrics_data = convert_to_serializable(metrics_data)
    
    # Save to JSON file
    os.makedirs(output_dir, exist_ok=True)
    json_path = f"{output_dir}/metrics_and_rules.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metrics and rules to: {json_path}")
    
    # Update results to include JSON path
    results["metrics_json_path"] = json_path
    
    return results


def explain_individual_instance(
    X_instance: np.ndarray,
    trained_model: Any,
    feature_names: List[str],
    target_class: int,
    X_train_data: np.ndarray,
    X_unit_data: np.ndarray,
    y_train_data: np.ndarray,
    classifier: torch.nn.Module,
    device: torch.device,
    scaler: Any,
    X_min: np.ndarray,
    X_range: np.ndarray,
    steps_per_episode: int = 100,
    max_features_in_rule: int = 5
) -> Dict[str, Any]:
    """
    Explain a single instance using trained dynamic anchor policy.
    
    Args:
        X_instance: Instance to explain (raw features)
        trained_model: Trained PPO model
        feature_names: Feature names
        target_class: Target class to explain
        X_train_data: Training data (scaled)
        X_unit_data: Training data (normalized to [0,1])
        y_train_data: Training labels
        classifier: Classifier model
        device: Device
        scaler: Fitted StandardScaler
        X_min: Min values for normalization
        X_range: Range values for normalization
        steps_per_episode: Max rollout steps
        max_features_in_rule: Max features in rule
    
    Returns:
        Dictionary with anchor explanation
    """
    from trainers.vecEnv import AnchorEnv
    from trainers.dynAnchors_inference import evaluate_single_instance
    
    # Scale instance
    X_instance_scaled = scaler.transform(X_instance.reshape(1, -1)).astype(np.float32).ravel()
    
    # Create environment factory
    def make_env():
        return AnchorEnv(
            X_unit=X_unit_data,
            X_std=X_train_data,
            y=y_train_data,
            feature_names=feature_names,
            classifier=classifier,
            device=device,
            target_class=target_class,
            step_fracs=(0.005, 0.01, 0.02),
            min_width=0.05,
            alpha=0.7,
            beta=0.6,
            gamma=0.1,
            precision_target=0.95,
            coverage_target=0.02,
            precision_blend_lambda=0.5,
            drift_penalty_weight=0.05,
            use_perturbation=False,
            perturbation_mode="bootstrap",
            n_perturb=1024,
            X_min=X_min,
            X_range=X_range,
            min_coverage_floor=0.005,
            js_penalty_weight=0.05,
        )
    
    # Evaluate
    result = evaluate_single_instance(
        X_instance=X_instance_scaled,
        trained_model=trained_model,
        make_env_fn=make_env,
        feature_names=feature_names,
        target_class=target_class,
        steps_per_episode=steps_per_episode,
        max_features_in_rule=max_features_in_rule
    )
    
    return result

