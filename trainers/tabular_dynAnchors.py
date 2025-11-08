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
    # Environment parameters
    use_perturbation: bool = True,
    perturbation_mode: str = "bootstrap",
    n_perturb: int = 1024,
    step_fracs: Tuple[float, ...] = (0.005, 0.01, 0.02),
    min_width: float = 0.05,
    precision_target: float = 0.95,
    coverage_target: float = 0.02,
    # Evaluation parameters
    n_eval_instances_per_class: int = 20,
    max_features_in_rule: int = 5,
    steps_per_episode: int = 100,
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
    2. Trains PPO policy to generate anchors
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
        device: Device to use ("cpu")
        n_envs: Number of parallel training environments
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO
        n_steps: Steps per environment before update
        batch_size: Batch size for PPO updates
        n_epochs: PPO epochs per update
        use_perturbation: Enable perturbation sampling in environment
        perturbation_mode: "bootstrap" or "uniform" sampling
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
            - trained_model: PPO model
            - trainer: Trainer instance
            - eval_results: Per-class evaluation results
            - overall_stats: Overall precision/coverage
            - metadata: Configuration and setup info
    """
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from trainers.vecEnv import AnchorEnv, make_dummy_vec_env
    from trainers.PPO_trainer import train_ppo_model
    from trainers.dynAnchors_inference import evaluate_all_classes
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
    
    # Determine target classes
    unique_classes = np.unique(y_train)
    if target_classes is None:
        target_classes = tuple(unique_classes)
    else:
        target_classes = tuple(target_classes)
    
    print(f"Classes: {unique_classes}, Target classes: {target_classes}")
    
    # Create environment factory function
    # For multi-class training, distribute classes across environments
    def create_anchor_env(target_cls=None):
        """Helper to create AnchorEnv with a specific target class."""
        if target_cls is None:
            target_cls = target_classes[0]  # Default to first class
        return AnchorEnv(
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
    
    # Train PPO model
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
    
    eval_results = evaluate_all_classes(
        X_test=X_test_scaled,
        y_test=y_test,
        trained_model=trainer.model,
        make_env_fn=make_anchor_env,
        feature_names=feature_names,
        n_instances_per_class=n_eval_instances_per_class,
        max_features_in_rule=max_features_in_rule,
        steps_per_episode=steps_per_episode,
        random_seed=42,
        # Optional: Set eval_on_test_data=True to compute metrics on test data
        eval_on_test_data=False,  # Default: use training data (backward compatible)
    )
    
    vec_env.close()
    
    # Prepare results
    results = {
        "trained_model": trainer.model,
        "trainer": trainer,
        "eval_results": eval_results,
        "overall_stats": {
            "avg_precision": eval_results["overall_precision"],
            "avg_coverage": eval_results["overall_coverage"],
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
    print(f"Overall precision: {eval_results['overall_precision']:.3f}")
    print(f"Overall coverage: {eval_results['overall_coverage']:.3f}")
    
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

