"""
Single-Agent Anchor Inference Script

This script loads a trained single-agent Stable-Baselines3 policy and performs
anchor inference. It can compare results with multi-agent inference.

Usage:
python single_agent/single_agent_inference.py \
    --experiment_dir <path_to_sb3_experiment> \
    --dataset breast_cancer \
    --compare_with_multiagent <path_to_multiagent_experiment> \
    --n_instances_per_class 20
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add single_agent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from BenchMARL.tabular_datasets import TabularDatasetLoader
from single_agentENV import SingleAgentAnchorEnv
from anchor_trainer_sb3 import AnchorTrainerSB3

# Import SB3
try:
    from stable_baselines3 import DDPG, SAC
    from stable_baselines3.common.evaluation import evaluate_policy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    raise ImportError("Stable-Baselines3 not installed. Please install: pip install stable-baselines3")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_single_agent_rollout(
    env: SingleAgentAnchorEnv,
    model,
    max_steps: Optional[int] = None,  # If None, will read from env.max_cycles
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run a single rollout episode using a trained SB3 model.
    
    Args:
        env: SingleAgentAnchorEnv environment (already configured for target class)
        model: Trained SB3 model (DDPG or SAC) for this class
        max_steps: Maximum steps per episode
        seed: Random seed
    
    Returns:
        Dictionary with episode data (precision, coverage, bounds, etc.)
    """
    # Reset environment
    obs, info = env.reset(seed=seed)
    
    # Start timing the rollout
    rollout_start_time = time.perf_counter()
    
    done = False
    step_count = 0
    total_reward = 0.0
    
    # Read max_steps from env if not provided
    if max_steps is None:
        max_steps = env.max_cycles
    
    # Store initial state
    initial_lower = env.lower.copy()
    initial_upper = env.upper.copy()
    
    # Main rollout loop
    while not done and step_count < max_steps:
        # Get action from model (deterministic for inference)
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += float(reward)
        done = terminated or truncated
        step_count += 1
    
    # End timing the rollout
    rollout_end_time = time.perf_counter()
    rollout_duration = rollout_end_time - rollout_start_time
    
    # Get final metrics
    precision, coverage, details = env._current_metrics()
    
    # For instance-based anchors, also compute class-conditional coverage for better interpretability
    coverage_class_conditional = 0.0
    if env.x_star_unit is not None:  # Instance-based mode
        # Get the mask and class labels
        if env.eval_on_test_data:
            y_data = env.y_test
        else:
            y_data = env.y
        mask = env._mask_in_box()
        if len(mask) == len(y_data):
            class_mask = (y_data == env.target_class)
            n_class_samples = class_mask.sum()
            if n_class_samples > 0:
                n_class_in_box = (mask & class_mask).sum()
                coverage_class_conditional = float(n_class_in_box / n_class_samples)
    
    # Get final bounds
    final_lower = env.lower.copy()
    final_upper = env.upper.copy()
    
    episode_data = {
        "target_class": int(env.target_class),
        "precision": float(precision),
        "coverage": float(coverage),  # Overall coverage P(x in box)
        "coverage_class_conditional": float(coverage_class_conditional),  # Class-conditional coverage P(x in box | y = target_class)
        "total_reward": float(total_reward),
        "n_steps": step_count,
        "rollout_time_seconds": float(rollout_duration),
        "final_observation": obs.tolist(),
        "initial_lower": initial_lower.tolist(),
        "initial_upper": initial_upper.tolist(),
        "final_lower": final_lower.tolist(),
        "final_upper": final_upper.tolist(),
    }
    
    # Add details from metrics
    if details:
        for key, value in details.items():
            if isinstance(value, (int, float, np.number)):
                episode_data[f"metric_{key}"] = float(value)
    
    return episode_data


def extract_rules_single_agent(
    experiment_dir: str,
    dataset_name: str,
    max_features_in_rule: int = -1,
    steps_per_episode: Optional[int] = None,  # If None, will read from env_config.max_cycles
    n_instances_per_class: int = 20,
    eval_on_test_data: bool = True,
    output_dir: Optional[str] = None,
    seed: int = 42,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Extract anchor rules using a trained single-agent SB3 model.
    
    Args:
        experiment_dir: Path to SB3 experiment directory
        dataset_name: Name of the dataset
        max_features_in_rule: Maximum features to include in rules
        steps_per_episode: Maximum steps per rollout
        n_instances_per_class: Number of instances to evaluate per class
        eval_on_test_data: Whether to evaluate on test data
        output_dir: Output directory for results
        seed: Random seed
        device: Device to use
    
    Returns:
        Dictionary containing extracted rules and evaluation data
    """
    logger.info("="*80)
    logger.info("SINGLE-AGENT ANCHOR RULE EXTRACTION (Stable-Baselines3)")
    logger.info("(One Policy Per Class Architecture)")
    logger.info("="*80)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("="*80)
    
    # Determine algorithm from experiment folder name
    experiment_name = os.path.basename(experiment_dir)
    if "ddpg" in experiment_name.lower():
        algorithm = "ddpg"
        model_class = DDPG
    elif "sac" in experiment_name.lower():
        algorithm = "sac"
        model_class = SAC
    else:
        logger.warning("Could not determine algorithm from path, defaulting to DDPG")
        algorithm = "ddpg"
        model_class = DDPG
    
    logger.info(f"Algorithm: {algorithm.upper()}")
    
    # Load dataset
    dataset_loader = TabularDatasetLoader(
        dataset_name=dataset_name,
        test_size=0.2,
        random_state=seed
    )
    
    dataset_loader.load_dataset()
    dataset_loader.preprocess_data()
    
    # Resolve experiment directory - handle case where training directory or base output directory is passed
    # instead of the actual experiment directory
    resolved_experiment_dir = experiment_dir
    experiment_path = Path(experiment_dir)
    
    # Check if this is the actual experiment directory (has models)
    has_models = (experiment_path / "final_model").exists() or (experiment_path / "best_model").exists()
    has_classifier = (experiment_path / "classifier.pth").exists()
    
    # If it's not an experiment directory (no models), look for experiment subdirectories
    if not has_models:
        # Look for experiment directories within this directory
        if experiment_path.is_dir():
            experiment_dirs = [
                d for d in experiment_path.iterdir()
                if d.is_dir() and (
                    (d / "final_model").exists() or
                    (d / "best_model").exists() or
                    (d / "classifier.pth").exists()
                )
            ]
            if experiment_dirs:
                # Prefer directories with models, then by modification time
                dirs_with_models = [d for d in experiment_dirs if (d / "final_model").exists() or (d / "best_model").exists()]
                if dirs_with_models:
                    resolved_experiment_dir = str(max(dirs_with_models, key=lambda p: p.stat().st_mtime))
                else:
                    resolved_experiment_dir = str(max(experiment_dirs, key=lambda p: p.stat().st_mtime))
                logger.info(f"Found experiment directory within: {resolved_experiment_dir}")
                has_models = True  # Mark as found so we don't check parent
    
    # If still not found and we have a classifier but no models, try parent directory
    if not has_models and has_classifier:
        parent_dir = experiment_path.parent
        # Look for experiment directories in parent
        if parent_dir.is_dir():
            experiment_dirs = [
                d for d in parent_dir.iterdir()
                if d.is_dir() and (
                    (d / "final_model").exists() or
                    (d / "best_model").exists() or
                    (d / "classifier.pth").exists()
                )
            ]
            if experiment_dirs:
                # Prefer directories with models, then by modification time
                dirs_with_models = [d for d in experiment_dirs if (d / "final_model").exists() or (d / "best_model").exists()]
                if dirs_with_models:
                    resolved_experiment_dir = str(max(dirs_with_models, key=lambda p: p.stat().st_mtime))
                else:
                    resolved_experiment_dir = str(max(experiment_dirs, key=lambda p: p.stat().st_mtime))
                logger.info(f"Found experiment directory in parent: {resolved_experiment_dir}")
    
    # Update experiment_dir to the resolved path
    experiment_dir = resolved_experiment_dir
    logger.info(f"Using experiment directory: {experiment_dir}")
    
    # Load classifier
    # The classifier can be in several locations:
    # 1. In the experiment directory itself (copied during training)
    # 2. In the training/ subdirectory (original location)
    # 3. In the parent of training/ (if experiment_dir is a subdirectory)
    checked_paths = []
    
    # First, try the experiment_dir itself (most common case after training)
    classifier_path = os.path.join(experiment_dir, "classifier.pth")
    checked_paths.append(classifier_path)
    
    if not os.path.exists(classifier_path):
        # Try the parent directory (in case experiment_dir is a subdirectory within training)
        experiment_parent = os.path.dirname(experiment_dir)
        classifier_path = os.path.join(experiment_parent, "classifier.pth")
        checked_paths.append(classifier_path)
    
    if not os.path.exists(classifier_path):
        # Try looking for training/ subdirectory and check there
        # This handles the case where experiment_dir is the base output directory
        training_dir = os.path.join(experiment_dir, "training")
        if os.path.isdir(training_dir):
            training_classifier = os.path.join(training_dir, "classifier.pth")
            checked_paths.append(training_classifier)
            if os.path.exists(training_classifier):
                classifier_path = training_classifier
    
    if not os.path.exists(classifier_path):
        # Try parent/training/classifier.pth (if experiment_dir is a subdirectory)
        experiment_parent = os.path.dirname(experiment_dir)
        training_dir = os.path.join(experiment_parent, "training")
        if os.path.isdir(training_dir):
            training_classifier = os.path.join(training_dir, "classifier.pth")
            checked_paths.append(training_classifier)
            if os.path.exists(training_classifier):
                classifier_path = training_classifier
    
    if os.path.exists(classifier_path):
        logger.info(f"Loading classifier from: {classifier_path}")
        classifier = dataset_loader.load_classifier(
            filepath=classifier_path,
            classifier_type="dnn",
            device=device
        )
        dataset_loader.classifier = classifier
    else:
        # Provide helpful error message with all checked paths
        raise ValueError(
            f"Classifier not found. Checked the following paths:\n" +
            "\n".join(f"  - {path}" for path in checked_paths) +
            f"\nPlease ensure the classifier was trained and saved during the training phase."
        )
    
    # Get environment data
    env_data = dataset_loader.get_anchor_env_data()
    target_classes = sorted(np.unique(dataset_loader.y_train).tolist())
    feature_names = env_data["feature_names"]
    n_features = len(feature_names)
    
    # Create environment config
    trainer = AnchorTrainerSB3(
        dataset_loader=dataset_loader,
        algorithm=algorithm,
        output_dir=experiment_dir,
        seed=seed
    )
    env_config = trainer._get_default_env_config()
    env_config.update({
        "X_min": env_data["X_min"],
        "X_range": env_data["X_range"],
    })
    
    # Check logging verbosity from config and apply to loggers
    verbosity = env_config.get("logging_verbosity", "normal")
    verbose_logging = (verbosity == "verbose")
    
    # Apply logging verbosity to all loggers
    import logging
    level = logging.WARNING if verbosity == "quiet" else (logging.DEBUG if verbosity == "verbose" else logging.INFO)
    
    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)
    
    # Set for all existing loggers
    for name in logging.Logger.manager.loggerDict:
        if isinstance(logging.Logger.manager.loggerDict[name], logging.Logger):
            log = logging.getLogger(name)
            log.setLevel(level)
            for handler in log.handlers:
                handler.setLevel(level)
    
    if verbose_logging:
        logger.info("Verbose logging enabled - showing detailed debug information")
    elif verbosity == "quiet":
        logger.warning("Quiet logging mode enabled - only warnings and errors will be shown")
    
    # Always use test data for inference
    if eval_on_test_data:
        if env_data.get("X_test_unit") is None:
            raise ValueError("Test data not available for evaluation.")
        env_config.update({
            "eval_on_test_data": True,
            "X_test_unit": env_data["X_test_unit"],
            "X_test_std": env_data["X_test_std"],
            "y_test": env_data["y_test"],
        })
        logger.info("✓ Using test data for inference")
    
    # Compute cluster centroids per class for diversity across episodes
    # This ensures each episode starts from a representative cluster centroid
    # (similar to training), rather than just the mean centroid
    logger.info("\nComputing cluster centroids per class for diversity...")
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
        
        # Use multiple clusters for diversity across episodes
        # Use at least one cluster per instance, but cap at 10 for efficiency
        n_clusters_per_class = min(10, n_instances_per_class)
        logger.info(f"  Using {n_clusters_per_class} clusters per class for diversity across episodes")
        
        cluster_centroids_per_class = compute_cluster_centroids_per_class(
            X_unit=X_cluster,
            y=y_cluster,
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
    
    # Load models for each class (one model per class)
    logger.info(f"\nLoading models for {len(target_classes)} classes...")
    models = {}  # class -> model
    
    for target_class in target_classes:
        # Try best model first (prefer best model over final model)
        # Best model is saved as best_model.zip in best_model/class_X/ directory
        model_path = os.path.join(experiment_dir, "best_model", f"class_{target_class}", "best_model.zip")
        if not os.path.exists(model_path):
            # Also try the old naming convention for backward compatibility
            model_path = os.path.join(experiment_dir, "best_model", f"class_{target_class}", f"class_{target_class}.zip")
        if not os.path.exists(model_path):
            # Fallback to final model if best model doesn't exist
            model_path = os.path.join(experiment_dir, "final_model", f"class_{target_class}.zip")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model for class {target_class} not found")
            logger.warning(f"  Tried: best_model/class_{target_class}/best_model.zip")
            logger.warning(f"  Tried: best_model/class_{target_class}/class_{target_class}.zip")
            logger.warning(f"  Tried: final_model/class_{target_class}.zip")
            continue
        
        # Create environment for this class (needed for model loading)
        env = SingleAgentAnchorEnv(
            X_unit=env_data["X_unit"],
            X_std=env_data["X_std"],
            y=env_data["y"],
            feature_names=feature_names,
            classifier=dataset_loader.get_classifier(),
            device=device,
            target_class=target_class,
            env_config=env_config
        )
        
        # Load model
        logger.info(f"  Loading model for class {target_class} from: {model_path}")
        models[target_class] = model_class.load(model_path, env=env, device=device)
    
    if not models:
        raise ValueError(f"No models found in {experiment_dir}. Expected models in final_model/class_*/ or best_model/class_*/ directories.")
    
    logger.info(f"✓ Loaded {len(models)} model(s) successfully")
    
    # Extract rules for each class
    logger.info(f"\nExtracting rules for classes: {target_classes}")
    logger.info(f"  Instances per class: {n_instances_per_class}")
    logger.info(f"  Steps per episode: {steps_per_episode}")
    logger.info(f"  Max features in rule: {max_features_in_rule}")
    
    # Start overall timing
    overall_start_time = time.perf_counter()
    
    results = {
        "per_class_results": {},
        "metadata": {
            "dataset": dataset_name,
            "experiment_dir": experiment_dir,
            "algorithm": algorithm,
            "target_classes": target_classes,
            "max_features_in_rule": max_features_in_rule,
            "eval_on_test_data": eval_on_test_data,
            "n_instances_per_class": n_instances_per_class,
            "steps_per_episode": steps_per_episode,
            "model_type": "single_agent_sb3",
        },
    }
    
    # Run rollouts for each class
    for target_class in target_classes:
        class_key = f"class_{target_class}"
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing class {target_class}")
        logger.info(f"{'='*80}")
        
        # Get model for this class
        if target_class not in models:
            logger.warning(f"  Model for class {target_class} not found, skipping...")
            continue
        
        model = models[target_class]
        
        anchors_list = []
        rules_list = []
        precisions = []
        coverages = []
        rollout_times = []
        
        # Track time for this class
        class_start_time = time.perf_counter()
        
        # ========================================================================
        # INSTANCE-BASED ROLLOUTS (boxes around specific instances)
        # ========================================================================
        logger.info(f"  Starting instance-based rollouts...")
        
        # Determine which dataset to sample from
        if eval_on_test_data and env_data.get("X_test_unit") is not None:
            class_mask = (env_data["y_test"] == target_class)
            class_instances = np.where(class_mask)[0]
            X_data_unit = env_data["X_test_unit"]
            env_X_unit = env_data["X_test_unit"]
            env_X_std = env_data["X_test_std"]
            env_y = env_data["y_test"]
            data_source_name = "test"
        else:
            class_mask = (env_data["y"] == target_class)
            class_instances = np.where(class_mask)[0]
            X_data_unit = env_data["X_unit"]
            env_X_unit = env_data["X_unit"]
            env_X_std = env_data["X_std"]
            env_y = env_data["y"]
            data_source_name = "training"
        
        if len(class_instances) == 0:
            logger.warning(f"  No instances found for class {target_class} in {data_source_name} data, skipping instance-based rollouts...")
        else:
            # Sample instances (same approach as multi-agent)
            n_samples = min(n_instances_per_class, len(class_instances))
            rng_for_sampling = np.random.default_rng(seed if seed is not None else 42)
            sampled_indices = rng_for_sampling.choice(class_instances, size=n_samples, replace=False)
            
            logger.info(f"  Sampling {n_samples} instances from {data_source_name} data for instance-based rollouts")
            
            for instance_idx_in_range, data_instance_idx in enumerate(sampled_indices):
                # Get the actual instance from the dataset
                x_instance = X_data_unit[data_instance_idx]
                
                # Create environment for this class
                rollout_seed = seed + instance_idx_in_range if seed is not None else None
                
                # Set mode to "inference" for rule extraction
                inference_env_config = {**env_config, "mode": "inference"}
                env = SingleAgentAnchorEnv(
                    X_unit=env_X_unit,
                    X_std=env_X_std,
                    y=env_y,
                    feature_names=feature_names,
                    classifier=dataset_loader.get_classifier(),
                    device=device,
                    target_class=target_class,
                    env_config=inference_env_config
                )
                
                # CRITICAL: Set x_star_unit for instance-based rollout
                # This creates a box around the instance (instance-based mode)
                env.x_star_unit = x_instance.copy()
                
                # Run rollout
                episode_data = run_single_agent_rollout(
                    env=env,
                    model=model,
                    max_steps=steps_per_episode,
                    seed=rollout_seed
                )
                
                precision = episode_data.get("precision", 0.0)
                coverage = episode_data.get("coverage", 0.0)
                coverage_class_conditional = episode_data.get("coverage_class_conditional", 0.0)
                rollout_time = episode_data.get("rollout_time_seconds", 0.0)
                
                precisions.append(float(precision))
                coverages.append(float(coverage))
                rollout_times.append(float(rollout_time))
                
                # Log coverage diagnostics if overall coverage is low but class-conditional is reasonable
                if instance_idx_in_range < 3 and coverage < 0.05 and coverage_class_conditional > 0.1:
                    logger.info(
                        f"  Episode {instance_idx_in_range}: Low overall coverage ({coverage:.4f}) but "
                        f"reasonable class-conditional coverage ({coverage_class_conditional:.4f}). "
                        f"This is expected if the dataset has many samples from other classes."
                    )
                
                # Extract rule from final bounds
                rule = "any values (no tightened features)"
                lower = None
                upper = None
                lower_normalized = None
                upper_normalized = None
                
                if "final_lower" in episode_data and "final_upper" in episode_data:
                    lower_normalized = np.array(episode_data["final_lower"], dtype=np.float32)
                    upper_normalized = np.array(episode_data["final_upper"], dtype=np.float32)
                    
                    # Denormalize bounds
                    X_min = env_config.get("X_min")
                    X_range = env_config.get("X_range")
                    if X_min is not None and X_range is not None:
                        lower = (lower_normalized * X_range) + X_min
                        upper = (upper_normalized * X_range) + X_min
                    else:
                        lower = lower_normalized
                        upper = upper_normalized
                    
                    # Extract rule using environment's extract_rule method
                    # Create temporary env for rule extraction (use same data as rollout env)
                    temp_env = SingleAgentAnchorEnv(
                        X_unit=env_X_unit,
                        X_std=env_X_std,
                        y=env_y,
                        feature_names=feature_names,
                        classifier=dataset_loader.get_classifier(),
                        device="cpu",
                        target_class=target_class,
                        env_config=env_config
                    )
                    temp_env.lower = lower_normalized
                    temp_env.upper = upper_normalized
                    
                    # Compute initial bounds from x_instance and initial_window for correct reference
                    initial_window = env_config.get("initial_window", 0.1)
                    initial_lower_normalized = np.clip(x_instance - initial_window, 0.0, 1.0)
                    initial_upper_normalized = np.clip(x_instance + initial_window, 0.0, 1.0)
                    
                    rule = temp_env.extract_rule(
                        max_features_in_rule=max_features_in_rule,
                        initial_lower=initial_lower_normalized,
                        initial_upper=initial_upper_normalized,
                        denormalize=True
                    )
                
                anchor_data = {
                    "instance_idx": instance_idx_in_range,
                    "data_instance_idx": int(data_instance_idx),
                    "rollout_type": "instance_based",  # Flag to distinguish from class-based
                    "precision": float(precision),
                    "coverage": float(coverage),  # Overall coverage P(x in box)
                    "coverage_class_conditional": float(coverage_class_conditional),  # Class-conditional coverage P(x in box | y = target_class)
                    "total_reward": float(episode_data.get("total_reward", 0.0)),
                    "n_steps": int(episode_data.get("n_steps", 0)),
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
                
                anchors_list.append(anchor_data)
                rules_list.append(rule)
        
        unique_rules = list(set([r for r in rules_list if r and r != "any values (no tightened features)"]))
        
        # End timing for this class
        class_end_time = time.perf_counter()
        class_total_time = class_end_time - class_start_time
        
        # Compute instance-level metrics (average across all instances)
        instance_precision = float(np.mean(precisions)) if precisions else 0.0
        instance_coverage = float(np.mean(coverages)) if coverages else 0.0
        
        # Compute average class-conditional coverage if available
        coverages_class_conditional = []
        for anchor_data in anchors_list:
            if "coverage_class_conditional" in anchor_data:
                coverages_class_conditional.append(anchor_data["coverage_class_conditional"])
        instance_coverage_class_conditional = float(np.mean(coverages_class_conditional)) if coverages_class_conditional else 0.0
        
        avg_rollout_time = float(np.mean(rollout_times)) if rollout_times else 0.0
        total_rollout_time = float(np.sum(rollout_times)) if rollout_times else 0.0
        
        # Compute class-level metrics (union of all anchors for this class)
        class_precision = 0.0
        class_coverage = 0.0
        anchors_with_bounds = 0  # Initialize to track anchors used in union computation
        
        # Get the appropriate dataset (test or train) based on eval_on_test_data
        # IMPORTANT: Must use the same dataset that was used during rollouts for consistency
        if eval_on_test_data and env_data.get("X_test_unit") is not None:
            X_data = env_data["X_test_unit"]
            y_data = env_data["y_test"]
            data_source = "test"
            logger.info(f"  Computing class-level metrics on TEST data (eval_on_test_data=True)")
        else:
            X_data = env_data["X_unit"]
            y_data = env_data["y"]
            data_source = "train"
            if eval_on_test_data:
                logger.warning(f"  WARNING: eval_on_test_data=True but X_test_unit is None, using TRAIN data instead!")
            else:
                logger.info(f"  Computing class-level metrics on TRAIN data (eval_on_test_data=False)")
        
        # Compute union of all anchors for this class
        # NOTE: This uses ALL anchors (including duplicates), not just unique rules.
        # The test script uses unique rules, which may give slightly different results
        # due to denormalization rounding. Both are valid but measure different things:
        # - Inference (all anchors): Shows coverage of all generated anchors
        # - Test script (unique rules): Shows coverage of deduplicated rules after denormalization
        if X_data is not None and y_data is not None and len(anchors_list) > 0:
            n_samples = X_data.shape[0]
            union_mask = np.zeros(n_samples, dtype=bool)
            
            # Count how many anchors have normalized bounds
            anchors_with_bounds = 0
            
            # Debug: Track coverage per anchor
            anchor_coverages = []
            
            # Build union mask from all anchors
            # TODO: Consider using unique rules instead to match test script results
            for anchor_idx, anchor_data in enumerate(anchors_list):
                if "lower_bounds_normalized" in anchor_data and "upper_bounds_normalized" in anchor_data:
                    lower_norm = anchor_data["lower_bounds_normalized"]
                    upper_norm = anchor_data["upper_bounds_normalized"]
                    
                    # Skip if bounds are None
                    if lower_norm is None or upper_norm is None:
                        logger.warning(f"  Anchor {anchor_idx} has None normalized bounds, skipping")
                        continue
                    
                    lower = np.array(lower_norm, dtype=np.float32)
                    upper = np.array(upper_norm, dtype=np.float32)
                    
                    # Validate bounds shape matches data
                    if lower.shape[0] != X_data.shape[1] or upper.shape[0] != X_data.shape[1]:
                        logger.warning(f"  Anchor {anchor_idx} bounds shape mismatch: lower.shape={lower.shape}, upper.shape={upper.shape}, X_data.shape[1]={X_data.shape[1]}, skipping")
                        continue
                    
                    # Validate bounds are in [0, 1] range
                    if np.any(lower < 0) or np.any(upper > 1):
                        logger.warning(f"  Anchor {anchor_idx} has bounds outside [0,1]: lower min={lower.min():.4f}, upper max={upper.max():.4f}")
                    
                    # Check which points fall in this anchor box
                    in_box = np.all((X_data >= lower) & (X_data <= upper), axis=1)
                    n_in_box = in_box.sum()
                    anchor_coverages.append(n_in_box)
                    union_mask |= in_box
                    anchors_with_bounds += 1
                else:
                    logger.warning(f"  Anchor {anchor_idx} missing normalized bounds keys")
            
            logger.debug(f"  Anchor individual coverages: {anchor_coverages[:5]}..." if len(anchor_coverages) > 5 else f"  Anchor individual coverages: {anchor_coverages}")
            
            # Class-level (class-union) metrics: union of all anchors for this class
            # NOTE: "class-level" here refers to "class-union" metrics computed from the union of all anchors.
            # This is different from "instance-level" metrics which are averaged across individual anchors.
            # 
            # IMPORTANT: This computation uses ALL anchors (including duplicates), not just unique rules.
            # The test script (test_extracted_rules_single.py) uses unique rules after deduplication and denormalization,
            # which may give slightly different results due to:
            #   1. Deduplication: Only unique rules are tested (fewer anchors)
            #   2. Denormalization: Rules are converted to standardized space, which may introduce rounding differences
            # 
            # Both metrics are valid but measure different things:
            #   - Inference (all anchors): Shows coverage of all generated anchors with normalized bounds
            #   - Test script (unique rules): Shows coverage of deduplicated rules after denormalization
            # 
            # For consistency with the test script, consider using unique rules for class-level metrics.
            mask_cls = (y_data == target_class)
            n_class_samples = mask_cls.sum()
            n_total_samples = len(y_data)
            if n_class_samples > 0:
                n_covered_class_samples = union_mask[mask_cls].sum()
                n_covered_total_samples = union_mask.sum()
                class_coverage = float(n_covered_class_samples / n_class_samples)
                logger.info(f"  Class {target_class} class-union coverage on {data_source} data: {n_covered_class_samples}/{n_class_samples} = {class_coverage:.4f}")
                logger.info(f"    (using {anchors_with_bounds} anchors with normalized bounds out of {len(anchors_list)} total anchors)")
                logger.info(f"    NOTE: This uses ALL anchors. Test script uses unique rules, which may differ slightly.")
                logger.info(f"    Total samples in {data_source} data: {n_total_samples}, Class {target_class} samples: {n_class_samples}, Total covered by union: {n_covered_total_samples}")
                
                # Debug: Check if union covers all samples (which would indicate a problem)
                if n_covered_total_samples == n_total_samples:
                    logger.warning(f"    WARNING: Union covers ALL {n_total_samples} samples in {data_source} data! This might indicate anchors are too wide.")
                elif n_covered_class_samples == n_class_samples:
                    logger.info(f"    Union covers all {n_class_samples} class {target_class} samples (perfect coverage for this class)")
            else:
                class_coverage = 0.0
                logger.warning(f"  Class {target_class} has no samples in {data_source} data")
            
            # Class-level (class-union) precision: fraction of points in union that belong to target class
            n_union_samples = union_mask.sum()
            if n_union_samples > 0:
                n_union_class_samples = (y_data[union_mask] == target_class).sum()
                class_precision = float(n_union_class_samples / n_union_samples)
                logger.info(f"  Class {target_class} class-union precision on {data_source} data: {n_union_class_samples}/{n_union_samples} = {class_precision:.4f}")
            else:
                class_precision = 0.0
                logger.warning(f"  Class {target_class} union covers no samples in {data_source} data")
        else:
            if X_data is None or y_data is None:
                logger.warning(f"  Cannot compute class-level metrics: X_data or y_data is None")
            if len(anchors_list) == 0:
                logger.warning(f"  Cannot compute class-level metrics: no anchors found")
        
        results["per_class_results"][class_key] = {
            "class": int(target_class),
            # Instance-level metrics (averaged across all instances)
            "instance_precision": instance_precision,
            "instance_coverage": instance_coverage,  # Overall coverage P(x in box)
            "instance_coverage_class_conditional": instance_coverage_class_conditional,  # Class-conditional coverage P(x in box | y = target_class)
            "instance_precision_std": float(np.std(precisions)) if len(precisions) > 1 else 0.0,
            "instance_coverage_std": float(np.std(coverages)) if len(coverages) > 1 else 0.0,
            # Class-level metrics (union of all anchors for this class)
            "class_precision": class_precision,
            "class_coverage": class_coverage,
            # Legacy fields for backward compatibility (same as instance-level)
            "precision": instance_precision,
            "coverage": instance_coverage,
            "precision_std": float(np.std(precisions)) if len(precisions) > 1 else 0.0,
            "coverage_std": float(np.std(coverages)) if len(coverages) > 1 else 0.0,
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
        
        logger.info(f"  Processed {len(anchors_list)} instance-based episodes")
        # Log in order: Instance metrics, then Union metrics (will be recomputed with class-based anchors later)
        logger.info(f"\n  {'='*60}")
        logger.info(f"  Class {target_class} - INSTANCE-BASED Results:")
        logger.info(f"  {'='*60}")
        logger.info(f"  Instance-Level Metrics (averaged across all {len(anchors_list)} instance-based anchors):")
        logger.info(f"    Precision: {instance_precision:.4f}")
        logger.info(f"    Coverage:  {instance_coverage:.4f}")
        if instance_coverage_class_conditional > 0.0:
            logger.info(f"    Class-conditional coverage: {instance_coverage_class_conditional:.4f}")
            if instance_coverage < 0.05 and instance_coverage_class_conditional > 0.1:
                logger.info(f"    Note: Low overall coverage is expected when dataset has many samples from other classes.")
        logger.info(f"  Unique rules (instance-based, after deduplication): {len(unique_rules)}")
        logger.info(f"  Average rollout time per episode: {avg_rollout_time:.4f}s")
        # Union metrics (instance-based only) - temporary, will be overwritten with class-based union after class-based rollouts
        n_anchors_for_union = anchors_with_bounds if anchors_with_bounds > 0 else len(anchors_list)
        logger.info(f"  Class Union Metrics (temporary - union of {n_anchors_for_union} instance-based anchors):")
        logger.info(f"    Precision: {class_precision:.4f}")
        logger.info(f"    Coverage:  {class_coverage:.4f}")
        logger.info(f"    Note: These will be overwritten with class-based union metrics after class-based rollouts.")
        logger.info(f"  {'='*60}")
        # logger.info(f"  Total rollout time for class: {total_rollout_time:.4f}s")
        # logger.info(f"  Total class processing time: {class_total_time:.4f}s")
    
    # ========================================================================
    # CLASS-BASED ROLLOUTS (using k-means cluster centroids)
    # ========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("Starting CLASS-BASED rollouts (initialized from k-means cluster centroids)")
    logger.info(f"{'='*80}")
    
    # Determine number of class-based rollouts per class
    n_class_based_rollouts_per_class = None
    if env_config.get("cluster_centroids_per_class") is not None:
        max_centroids = 0
        for cls in target_classes:
            if cls in env_config["cluster_centroids_per_class"]:
                max_centroids = max(max_centroids, len(env_config["cluster_centroids_per_class"][cls]))
        n_class_based_rollouts_per_class = max(5, min(max_centroids, 10))
        logger.info(f"  Using {n_class_based_rollouts_per_class} class-based rollouts per class (based on cluster centroids)")
    else:
        n_class_based_rollouts_per_class = 5
        logger.info(f"  Using {n_class_based_rollouts_per_class} class-based rollouts per class (default, no cluster centroids)")
    
    # Run class-based rollouts for each class
    for target_class in target_classes:
        class_key = f"class_{target_class}"
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Class-based rollouts for class {target_class}")
        logger.info(f"{'='*80}")
        
        # Get model for this class
        if target_class not in models:
            logger.warning(f"  Model for class {target_class} not found, skipping...")
            continue
        
        model = models[target_class]
        
        # Initialize lists for class-based rollouts
        class_based_anchors_list = []
        class_based_rules_list = []
        class_based_precisions = []
        class_based_coverages = []
        class_based_rollout_times = []
        
        class_based_start_time = time.perf_counter()
        
        # Run class-based rollouts (NOT setting x_star_unit, so environment uses cluster centroids)
        for rollout_idx in range(n_class_based_rollouts_per_class):
            rollout_seed = seed + 10000 + rollout_idx if seed is not None else None  # Use different seed range
            
            # CRITICAL: Use FULL dataset (train + test) for class-based rollouts
            # Class-based rules should express a particular class of the full dataset
            # This ensures rules are evaluated on the complete class distribution
            if eval_on_test_data and env_data.get("X_test_unit") is not None:
                # Combine train and test data for class-based rollouts (full dataset)
                env_X_unit = np.vstack([env_data["X_unit"], env_data["X_test_unit"]])
                env_X_std = np.vstack([env_data["X_std"], env_data["X_test_std"]])
                env_y = np.concatenate([env_data["y"], env_data["y_test"]])
                use_full_dataset = True
                if rollout_idx == 0:  # Log once per class
                    logger.info(f"  Using FULL dataset (train + test) for class-based rollouts")
                    logger.info(f"    Training samples: {len(env_data['X_unit'])}, Test samples: {len(env_data['X_test_unit'])}, Total: {len(env_y)}")
                    logger.info(f"    Note: Class-based rules should express the class across the full dataset")
            else:
                # Fallback to training data only if test data not available
                env_X_unit = env_data["X_unit"]
                env_X_std = env_data["X_std"]
                env_y = env_data["y"]
                use_full_dataset = False
                if rollout_idx == 0:  # Log once per class
                    logger.info(f"  Using TRAINING data only for class-based rollouts (test data not available)")
            
            # Set mode to "inference" for rule extraction
            # CRITICAL: Ensure use_class_centroids is True for class-based rollouts
            # When using full dataset, set eval_on_test_data=False so metrics are computed on full dataset
            inference_env_config = {
                **env_config, 
                "mode": "inference",
                "use_class_centroids": True,  # Explicitly enable class-based initialization
                "eval_on_test_data": False if use_full_dataset else eval_on_test_data  # Use full dataset for metrics when using full dataset
            }
            
            env = SingleAgentAnchorEnv(
                X_unit=env_X_unit,
                X_std=env_X_std,
                y=env_y,
                feature_names=feature_names,
                classifier=dataset_loader.get_classifier(),
                device=device,
                target_class=target_class,
                env_config=inference_env_config
            )
            
            # IMPORTANT: DO NOT set x_star_unit - this triggers class-based initialization
            # The environment will use cluster_centroids_per_class when available,
            # otherwise it will use mean centroid of the class
            # Verify that x_star_unit is None (should be by default)
            if env.x_star_unit is not None:
                logger.warning(f"  WARNING: x_star_unit is set for class-based rollout {rollout_idx}! This will cause instance-based mode instead.")
                logger.warning(f"    x_star_unit shape: {env.x_star_unit.shape if isinstance(env.x_star_unit, np.ndarray) else type(env.x_star_unit)}")
                # Force it to None for class-based mode
                env.x_star_unit = None
            
            # Verify use_class_centroids is enabled
            if not env.use_class_centroids:
                logger.warning(f"  WARNING: use_class_centroids is False for class-based rollout {rollout_idx}! This will cause full-space initialization.")
                env.use_class_centroids = True
            
            # Debug: Log centroid selection
            if rollout_idx == 0:  # Only log for first rollout to avoid spam
                centroid = env._get_class_centroid()
                if centroid is not None:
                    logger.debug(f"  Class-based rollout {rollout_idx}: Using centroid (shape={centroid.shape}, mean={centroid.mean():.4f})")
                else:
                    logger.warning(f"  Class-based rollout {rollout_idx}: No centroid available! Will use full-space initialization.")
            
            # Run rollout
            episode_data = run_single_agent_rollout(
                env=env,
                model=model,
                max_steps=steps_per_episode,
                seed=rollout_seed
            )
            
            precision = episode_data.get("precision", 0.0)
            coverage = episode_data.get("coverage", 0.0)
            rollout_time = episode_data.get("rollout_time_seconds", 0.0)
            
            # Debug logging for first few rollouts
            if rollout_idx < 3:
                logger.debug(f"  Class-based rollout {rollout_idx}: precision={precision:.4f}, coverage={coverage:.4f}")
                if precision == 0.0 or coverage == 0.0:
                    logger.debug(f"    Final bounds: lower={env.lower[:3] if hasattr(env, 'lower') else 'N/A'}, upper={env.upper[:3] if hasattr(env, 'upper') else 'N/A'} (first 3 dims)")
                    logger.debug(f"    Box volume: {np.prod(env.upper - env.lower) if hasattr(env, 'lower') and hasattr(env, 'upper') else 'N/A'}")
            
            class_based_precisions.append(float(precision))
            class_based_coverages.append(float(coverage))
            class_based_rollout_times.append(float(rollout_time))
            
            # Extract rule from final bounds
            rule = "any values (no tightened features)"
            lower = None
            upper = None
            lower_normalized = None
            upper_normalized = None
            
            if "final_lower" in episode_data and "final_upper" in episode_data:
                lower_normalized = np.array(episode_data["final_lower"], dtype=np.float32)
                upper_normalized = np.array(episode_data["final_upper"], dtype=np.float32)
                
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
                temp_env = SingleAgentAnchorEnv(
                    X_unit=env_X_unit,
                    X_std=env_X_std,
                    y=env_y,
                    feature_names=feature_names,
                    classifier=dataset_loader.get_classifier(),
                    device="cpu",
                    target_class=target_class,
                    env_config=env_config
                )
                temp_env.lower = lower_normalized
                temp_env.upper = upper_normalized
                
                # For class-based, use the center of the final box as reference for initial bounds
                initial_window = env_config.get("initial_window", 0.1)
                box_center = (lower_normalized + upper_normalized) / 2.0
                initial_lower_normalized = np.clip(box_center - initial_window, 0.0, 1.0)
                initial_upper_normalized = np.clip(box_center + initial_window, 0.0, 1.0)
                
                rule = temp_env.extract_rule(
                    max_features_in_rule=max_features_in_rule,
                    initial_lower=initial_lower_normalized,
                    initial_upper=initial_upper_normalized,
                    denormalize=True
                )
            
            # Store anchor data
            anchor_data = {
                "rollout_type": "class_based",  # Flag to distinguish from instance-based
                "rollout_idx": rollout_idx,
                "precision": float(precision),
                "coverage": float(coverage),
                "total_reward": float(episode_data.get("total_reward", 0.0)),
                "n_steps": int(episode_data.get("n_steps", 0)),
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
        
        class_based_precision = float(np.mean(class_based_precisions)) if class_based_precisions else 0.0
        class_based_coverage = float(np.mean(class_based_coverages)) if class_based_coverages else 0.0
        class_based_avg_rollout_time = float(np.mean(class_based_rollout_times)) if class_based_rollout_times else 0.0
        class_based_total_rollout_time = float(np.sum(class_based_rollout_times)) if class_based_rollout_times else 0.0
        
        # Store class-based results in per_class_results
        if class_key not in results["per_class_results"]:
            results["per_class_results"][class_key] = {}
        
        # CRITICAL: Add class-based metrics to the main class entry (for consistency with multi-agent)
        # This allows the summary script to find them easily
        results["per_class_results"][class_key]["class_level_precision"] = class_based_precision
        results["per_class_results"][class_key]["class_level_coverage"] = class_based_coverage
        # Legacy names for backward compatibility
        results["per_class_results"][class_key]["class_based_precision"] = class_based_precision
        results["per_class_results"][class_key]["class_based_coverage"] = class_based_coverage
        
        # Also store class-based results in a separate key for easy access (kept for backward compatibility)
        class_based_key = f"class_{target_class}_class_based"
        results["per_class_results"][class_based_key] = {
            "class": int(target_class),
            "rollout_type": "class_based",
            "n_episodes": len(class_based_anchors_list),
            "precision": class_based_precision,
            "coverage": class_based_coverage,
            "precision_std": float(np.std(class_based_precisions)) if len(class_based_precisions) > 1 else 0.0,
            "coverage_std": float(np.std(class_based_coverages)) if len(class_based_coverages) > 1 else 0.0,
            "unique_rules": class_based_unique_rules,
            "unique_rules_count": len(class_based_unique_rules),
            "rules": class_based_rules_list,
            "anchors": class_based_anchors_list,
            "avg_rollout_time_seconds": class_based_avg_rollout_time,
            "total_rollout_time_seconds": class_based_total_rollout_time,
            "total_processing_time_seconds": float(class_based_total_time),
        }
        
        logger.info(f"\n  {'='*60}")
        logger.info(f"  Class {target_class} - CLASS-BASED Results (Centroid-Based Rollouts):")
        logger.info(f"  {'='*60}")
        logger.info(f"  Class-Based Metrics (averaged across all {len(class_based_anchors_list)} class-based anchors):")
        logger.info(f"    Precision: {class_based_precision:.4f}")
        logger.info(f"    Coverage:  {class_based_coverage:.4f}")
        logger.info(f"  Unique rules (class-based, after deduplication): {len(class_based_unique_rules)}")
        logger.info(f"  Average rollout time per episode: {class_based_avg_rollout_time:.4f}s")
        logger.info(f"  {'='*60}")
    
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
        
        # Get class-based anchors and rules (stored separately)
        class_based_key = f"class_{target_class}_class_based"
        if class_based_key in results.get("per_class_results", {}):
            class_based_data = results["per_class_results"][class_based_key]
            if "anchors" in class_based_data:
                all_anchors_for_union.extend(class_based_data["anchors"])
            if "unique_rules" in class_based_data:
                class_based_unique_rules = class_based_data["unique_rules"]
        
        # Compute union of class-based anchors only (smallest set of general rules)
        if X_data_union is not None and y_data_union is not None and len(all_anchors_for_union) > 0:
            n_samples = X_data_union.shape[0]
            union_mask = np.zeros(n_samples, dtype=bool)
            
            # Build union mask from all anchors
            for anchor_data in all_anchors_for_union:
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
            n_class_based_anchors = len(all_anchors_for_union)
            n_unique_rules = len(class_based_unique_rules)
            logger.info(f"\n  {'='*60}")
            logger.info(f"  Class {target_class} - CLASS UNION Results (Union of Class-Based Anchors Only):")
            logger.info(f"  {'='*60}")
            logger.info(f"  Class Union Metrics (union of {n_class_based_anchors} class-based anchors):")
            logger.info(f"    Precision: {class_precision_combined:.4f}")
            logger.info(f"    Coverage:  {class_coverage_combined:.4f}")
            logger.info(f"    Unique Rules: {n_unique_rules}")
            logger.info(f"    Note: Represents the smallest set of general rules that explain the class structure")
            
            # Log the actual rules
            if class_based_unique_rules:
                logger.info(f"\n  Class {target_class} - Class-Based Union Rules:")
                for i, rule in enumerate(class_based_unique_rules, 1):
                    logger.info(f"    Rule {i}: {rule}")
            else:
                logger.info(f"\n  Class {target_class} - No unique rules found in class-based anchors")
            
            logger.info(f"  {'='*60}")
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
        def _convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int_)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
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
        
        rules_filepath = os.path.join(output_dir, "extracted_rules_single_agent.json")
        serializable_results = _convert_to_serializable(results)
        
        with open(rules_filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {rules_filepath}")
    
    return results


def compare_with_multiagent(
    single_agent_results: Dict[str, Any],
    multiagent_experiment_dir: str,
    dataset_name: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Compare single-agent results with multi-agent results.
    
    Args:
        single_agent_results: Results from single-agent inference
        multiagent_experiment_dir: Path to multi-agent experiment directory
        dataset_name: Dataset name
        output_dir: Output directory for comparison results
    
    Returns:
        Dictionary with comparison metrics
    """
    logger.info("\n" + "="*80)
    logger.info("COMPARING SINGLE-AGENT vs MULTI-AGENT RESULTS")
    logger.info("="*80)
    
    # Load multi-agent results
    multiagent_inference_path = os.path.join(multiagent_experiment_dir, "inference", "extracted_rules.json")
    if not os.path.exists(multiagent_inference_path):
        logger.warning(f"Multi-agent inference results not found at: {multiagent_inference_path}")
        logger.warning("Skipping comparison. Run multi-agent inference first.")
        return {}
    
    with open(multiagent_inference_path, 'r') as f:
        multiagent_results = json.load(f)
    
    logger.info(f"Loaded multi-agent results from: {multiagent_inference_path}")
    
    # Compare per-class metrics
    comparison = {
        "single_agent": {},
        "multi_agent": {},
        "differences": {},
        "summary": {}
    }
    
    single_agent_classes = single_agent_results.get("per_class_results", {})
    multiagent_classes = multiagent_results.get("per_class_results", {})
    
    all_classes = set()
    for key in single_agent_classes.keys():
        all_classes.add(key)
    for key in multiagent_classes.keys():
        all_classes.add(key)
    
    for class_key in sorted(all_classes):
        sa_data = single_agent_classes.get(class_key, {})
        ma_data = multiagent_classes.get(class_key, {})
        
        # Use instance-level metrics (explicit fields preferred, fallback to legacy)
        sa_precision = sa_data.get("instance_precision", sa_data.get("precision", 0.0))
        sa_coverage = sa_data.get("instance_coverage", sa_data.get("coverage", 0.0))
        sa_class_precision = sa_data.get("class_precision", sa_precision)
        sa_class_coverage = sa_data.get("class_coverage", sa_coverage)
        sa_unique_rules = sa_data.get("unique_rules_count", 0)
        
        ma_precision = ma_data.get("instance_precision", ma_data.get("precision", 0.0))
        ma_coverage = ma_data.get("instance_coverage", ma_data.get("coverage", 0.0))
        ma_class_precision = ma_data.get("class_precision", ma_precision)
        ma_class_coverage = ma_data.get("class_coverage", ma_coverage)
        ma_unique_rules = ma_data.get("unique_rules_count", 0)
        
        comparison["single_agent"][class_key] = {
            "instance_precision": sa_precision,
            "instance_coverage": sa_coverage,
            "class_precision": sa_class_precision,
            "class_coverage": sa_class_coverage,
            "unique_rules": sa_unique_rules,
            # Legacy fields for backward compatibility
            "precision": sa_precision,
            "coverage": sa_coverage,
        }
        
        comparison["multi_agent"][class_key] = {
            "instance_precision": ma_precision,
            "instance_coverage": ma_coverage,
            "class_precision": ma_class_precision,
            "class_coverage": ma_class_coverage,
            "unique_rules": ma_unique_rules,
            # Legacy fields for backward compatibility
            "precision": ma_precision,
            "coverage": ma_coverage,
        }
        
        comparison["differences"][class_key] = {
            "instance_precision_diff": sa_precision - ma_precision,
            "instance_coverage_diff": sa_coverage - ma_coverage,
            "class_precision_diff": sa_class_precision - ma_class_precision,
            "class_coverage_diff": sa_class_coverage - ma_class_coverage,
            "unique_rules_diff": sa_unique_rules - ma_unique_rules,
            # Legacy fields for backward compatibility
            "precision_diff": sa_precision - ma_precision,
            "coverage_diff": sa_coverage - ma_coverage,
        }
        
        logger.info(f"\n{class_key}:")
        logger.info(f"  Instance-Level:")
        logger.info(f"    Precision:  Single={sa_precision:.4f}, Multi={ma_precision:.4f}, Diff={sa_precision - ma_precision:.4f}")
        logger.info(f"    Coverage:   Single={sa_coverage:.4f}, Multi={ma_coverage:.4f}, Diff={sa_coverage - ma_coverage:.4f}")
        logger.info(f"  Class-Level:")
        logger.info(f"    Precision:  Single={sa_class_precision:.4f}, Multi={ma_class_precision:.4f}, Diff={sa_class_precision - ma_class_precision:.4f}")
        logger.info(f"    Coverage:   Single={sa_class_coverage:.4f}, Multi={ma_class_coverage:.4f}, Diff={sa_class_coverage - ma_class_coverage:.4f}")
        logger.info(f"  Unique Rules: Single={sa_unique_rules}, Multi={ma_unique_rules}, Diff={sa_unique_rules - ma_unique_rules}")
    
    # Summary statistics
    sa_precisions = [v["precision"] for v in comparison["single_agent"].values()]
    ma_precisions = [v["precision"] for v in comparison["multi_agent"].values()]
    sa_coverages = [v["coverage"] for v in comparison["single_agent"].values()]
    ma_coverages = [v["coverage"] for v in comparison["multi_agent"].values()]
    
    comparison["summary"] = {
        "single_agent": {
            "mean_precision": float(np.mean(sa_precisions)) if sa_precisions else 0.0,
            "mean_coverage": float(np.mean(sa_coverages)) if sa_coverages else 0.0,
        },
        "multi_agent": {
            "mean_precision": float(np.mean(ma_precisions)) if ma_precisions else 0.0,
            "mean_coverage": float(np.mean(ma_coverages)) if ma_coverages else 0.0,
        },
        "overall_differences": {
            "precision_diff": float(np.mean(sa_precisions) - np.mean(ma_precisions)) if sa_precisions and ma_precisions else 0.0,
            "coverage_diff": float(np.mean(sa_coverages) - np.mean(ma_coverages)) if sa_coverages and ma_coverages else 0.0,
        }
    }
    
    logger.info(f"\nSummary:")
    logger.info(f"  Mean Precision:  Single={comparison['summary']['single_agent']['mean_precision']:.4f}, "
                f"Multi={comparison['summary']['multi_agent']['mean_precision']:.4f}")
    logger.info(f"  Mean Coverage:   Single={comparison['summary']['single_agent']['mean_coverage']:.4f}, "
                f"Multi={comparison['summary']['multi_agent']['mean_coverage']:.4f}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Single-Agent Anchor Inference (SB3)")
    
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to SB3 experiment directory (contains final_model.zip or best_model/)"
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
        "--max_features_in_rule",
        type=int,
        default=-1,
        help="Maximum number of features to include in extracted rules (use -1 for all features)"
    )
    
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=100,
        help="Maximum steps per rollout episode (default: 100)"
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
        help="Evaluate on training data instead of test data (not recommended)"
    )
    
    parser.add_argument(
        "--compare_with_multiagent",
        type=str,
        default=None,
        help="Path to multi-agent experiment directory for comparison"
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
        choices=["cpu", "cuda", "auto"],
        help="Device to use for inference"
    )
    
    args = parser.parse_args()
    
    # Extract rules
    results = extract_rules_single_agent(
        experiment_dir=args.experiment_dir,
        dataset_name=args.dataset,
        max_features_in_rule=args.max_features_in_rule,
        steps_per_episode=args.steps_per_episode,
        n_instances_per_class=args.n_instances_per_class,
        eval_on_test_data=not args.eval_on_train_data,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device
    )
    
    # Save results
    output_dir = args.output_dir or os.path.join(args.experiment_dir, "inference")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to serializable format
    def _convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
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
    
    # Save single-agent results
    rules_filepath = os.path.join(output_dir, "extracted_rules_single_agent.json")
    serializable_results = _convert_to_serializable(results)
    
    with open(rules_filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Single-agent inference complete!")
    logger.info(f"Results saved to: {rules_filepath}")
    
    # Compare with multi-agent if requested
    if args.compare_with_multiagent:
        comparison = compare_with_multiagent(
            single_agent_results=results,
            multiagent_experiment_dir=args.compare_with_multiagent,
            dataset_name=args.dataset,
            output_dir=output_dir
        )
        
        if comparison:
            # Save comparison results
            comparison_filepath = os.path.join(output_dir, "comparison_single_vs_multi.json")
            serializable_comparison = _convert_to_serializable(comparison)
            
            with open(comparison_filepath, 'w') as f:
                json.dump(serializable_comparison, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Comparison results saved to: {comparison_filepath}")
    
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()

