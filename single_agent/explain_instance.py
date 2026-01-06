"""
Helper function to explain a new instance using trained Single-Agent Dynamic Anchors models.

This module provides functions to explain a new instance (unknown class) by checking
it against all trained classes and returning anchor explanations for each class.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
import json
from pathlib import Path

from BenchMARL.tabular_datasets import TabularDatasetLoader
from single_agentENV import SingleAgentAnchorEnv
from single_agent_inference import run_single_agent_rollout

logger = logging.getLogger(__name__)


def normalize_instance(
    instance: np.ndarray,
    X_min: np.ndarray,
    X_range: np.ndarray
) -> np.ndarray:
    """
    Normalize a new instance to unit space [0, 1].
    
    Args:
        instance: Instance in standardized space (mean=0, std=1), shape (n_features,)
        X_min: Minimum values per feature from training data
        X_range: Range values per feature from training data
    
    Returns:
        Normalized instance in unit space [0, 1]
    """
    instance_unit = (instance - X_min) / X_range
    instance_unit = np.clip(instance_unit, 0.0, 1.0).astype(np.float32)
    return instance_unit


def predict_class(
    instance: np.ndarray,
    classifier: torch.nn.Module,
    device: str = "cpu"
) -> Tuple[int, np.ndarray]:
    """
    Predict the class of a new instance.
    
    Args:
        instance: Instance in standardized space, shape (n_features,)
        classifier: Trained classifier model
        device: Device to run inference on
    
    Returns:
        Tuple of (predicted_class, class_probabilities)
    """
    classifier.eval()
    with torch.no_grad():
        # Reshape to (1, n_features) for batch dimension
        instance_tensor = torch.from_numpy(instance.astype(np.float32)).unsqueeze(0).to(device)
        logits = classifier(instance_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probs))
    
    return predicted_class, probs


def explain_new_instance_single_agent(
    new_instance: np.ndarray,
    experiment_dir: str,
    dataset_name: str,
    instance_in_standardized_space: bool = True,
    steps_per_episode: Optional[int] = None,
    max_features_in_rule: int = -1,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Explain a new instance by checking it against all trained classes.
    
    This function:
    1. Normalizes the new instance
    2. Predicts its class (for reference)
    3. Runs instance-based inference for all classes
    4. Returns anchor explanations for all classes
    
    Args:
        new_instance: New instance to explain, shape (n_features,)
                     If instance_in_standardized_space=True, expects standardized space (mean=0, std=1)
                     If False, expects original feature space (will be standardized first)
        experiment_dir: Path to experiment directory with trained models
        dataset_name: Name of the dataset
        instance_in_standardized_space: Whether instance is already in standardized space
        steps_per_episode: Maximum steps per rollout (None = use from config)
        max_features_in_rule: Maximum features to include in rule (-1 = all)
        device: Device to run inference on
        seed: Random seed
    
    Returns:
        Dictionary with:
        - 'predicted_class': Predicted class for the instance
        - 'class_probabilities': Probability distribution over classes
        - 'explanations': Dict mapping class_id -> explanation results
            Each explanation contains:
            - 'precision': Precision of the anchor
            - 'coverage': Coverage of the anchor
            - 'rule': Extracted anchor rule
            - 'bounds': Final anchor bounds (normalized and standardized)
            - 'rollout_time': Time taken for rollout
    """
    logger.info("="*80)
    logger.info("EXPLAINING NEW INSTANCE (Single-Agent)")
    logger.info("="*80)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Instance shape: {new_instance.shape}")
    logger.info("="*80)
    
    # Load dataset and normalization parameters
    dataset_loader = TabularDatasetLoader(
        dataset_name=dataset_name,
        test_size=0.2,
        random_state=seed
    )
    dataset_loader.load_dataset()
    dataset_loader.preprocess_data()
    
    # Get normalization parameters
    X_min = dataset_loader.X_min
    X_range = dataset_loader.X_range
    
    # Standardize instance if needed
    if not instance_in_standardized_space:
        logger.info("Standardizing instance from original feature space...")
        instance_std = dataset_loader.scaler.transform(new_instance.reshape(1, -1))[0]
    else:
        instance_std = new_instance.astype(np.float32)
    
    # Normalize to unit space
    instance_unit = normalize_instance(instance_std, X_min, X_range)
    logger.info(f"Instance normalized to unit space: min={instance_unit.min():.4f}, max={instance_unit.max():.4f}")
    
    # Load classifier and predict class
    classifier_path = os.path.join(experiment_dir, "classifier.pth")
    if not os.path.exists(classifier_path):
        raise ValueError(f"Classifier not found at {classifier_path}")
    
    classifier = dataset_loader.load_classifier(
        filepath=classifier_path,
        classifier_type="dnn",
        device=device
    )
    
    predicted_class, class_probs = predict_class(instance_std, classifier, device)
    logger.info(f"\nPredicted class: {predicted_class}")
    logger.info(f"Class probabilities: {class_probs}")
    
    # Load environment data
    env_data = dataset_loader.get_anchor_env_data()
    target_classes = list(np.unique(dataset_loader.y_train))
    feature_names = env_data["feature_names"]
    
    # Load env config
    from anchor_trainer_sb3 import AnchorTrainerSB3
    trainer = AnchorTrainerSB3(
        dataset_loader=dataset_loader,
        algorithm="ddpg",
        output_dir=os.path.join(experiment_dir, "inference"),
        seed=seed
    )
    try:
        env_config = trainer._load_env_config_from_yaml()
    except Exception:
        env_config = trainer._get_default_env_config()
    
    if isinstance(env_config, dict) and isinstance(env_config.get("env_config", None), dict):
        nested = env_config.get("env_config", {})
        top = {k: v for k, v in env_config.items() if k != "env_config"}
        env_config = {**nested, **top}
    
    if steps_per_episode is None:
        steps_per_episode = int(env_config.get("max_cycles", 100))
    
    # Load models - SB3 stores models per class
    # Check for model files in experiment directory
    from stable_baselines3 import DDPG, SAC
    
    # Find model files (SB3 naming convention)
    model_files = {}
    for target_class in target_classes:
        # Try different naming patterns
        model_patterns = [
            f"class_{target_class}_model.zip",
            f"class_{target_class}_ddpg.zip",
            f"class_{target_class}_sac.zip",
            f"model_class_{target_class}.zip"
        ]
        
        for pattern in model_patterns:
            model_path = os.path.join(experiment_dir, pattern)
            if os.path.exists(model_path):
                model_files[target_class] = model_path
                break
    
    # If not found, check subdirectories
    if not model_files:
        for item in os.listdir(experiment_dir):
            item_path = os.path.join(experiment_dir, item)
            if os.path.isdir(item_path):
                for target_class in target_classes:
                    for pattern in [f"class_{target_class}_model.zip", f"model_class_{target_class}.zip"]:
                        model_path = os.path.join(item_path, pattern)
                        if os.path.exists(model_path):
                            model_files[target_class] = model_path
                            break
    
    if not model_files:
        raise ValueError(
            f"No model files found in {experiment_dir}\n"
            f"Expected files like: class_0_model.zip, class_1_model.zip, etc."
        )
    
    logger.info(f"Found {len(model_files)} model file(s) for {len(target_classes)} classes")
    
    # Prepare results
    results = {
        'predicted_class': int(predicted_class),
        'class_probabilities': class_probs.tolist(),
        'explanations': {}
    }
    
    # Run inference for each class
    for target_class in target_classes:
        if target_class not in model_files:
            logger.warning(f"No model found for class {target_class}, skipping...")
            continue
        
        model_path = model_files[target_class]
        logger.info(f"\n{'='*80}")
        logger.info(f"Explaining for Class {target_class}")
        logger.info(f"Model: {model_path}")
        logger.info(f"{'='*80}")
        
        # Load SB3 model
        # Try to determine algorithm from filename or use DDPG as default
        if "sac" in model_path.lower():
            model = SAC.load(model_path, device=device)
        else:
            model = DDPG.load(model_path, device=device)
        
        # Create environment for this class
        env_config_inference = {**env_config, "mode": "inference", "normalize_data": False}
        if "X_min" not in env_config_inference:
            env_config_inference["X_min"] = X_min
        if "X_range" not in env_config_inference:
            env_config_inference["X_range"] = X_range
        
        env = SingleAgentAnchorEnv(
            X_unit=env_data["X_unit"],
            X_std=env_data["X_std"],
            y=env_data["y"],
            feature_names=feature_names,
            classifier=classifier,
            target_class=target_class,
            env_config=env_config_inference
        )
        
        # CRITICAL: Set x_star_unit BEFORE reset() for instance-based mode
        env.x_star_unit = instance_unit.copy()
        
        # Run rollout
        import time
        rollout_start = time.perf_counter()
        
        episode_data = run_single_agent_rollout(
            env=env,
            model=model,
            max_steps=steps_per_episode,
            seed=seed
        )
        
        rollout_time = time.perf_counter() - rollout_start
        
        # Extract rule
        if "final_lower" in episode_data and "final_upper" in episode_data:
            lower_normalized = np.array(episode_data["final_lower"], dtype=np.float32)
            upper_normalized = np.array(episode_data["final_upper"], dtype=np.float32)
            
            # Denormalize to standardized space
            lower_std = (lower_normalized * X_range) + X_min
            upper_std = (upper_normalized * X_range) + X_min
            
            # Create temp env for rule extraction
            temp_env = SingleAgentAnchorEnv(
                X_unit=env_data["X_unit"],
                X_std=env_data["X_std"],
                y=env_data["y"],
                feature_names=feature_names,
                classifier=classifier,
                target_class=target_class,
                env_config=env_config_inference
            )
            temp_env.lower = lower_normalized
            temp_env.upper = upper_normalized
            
            # Compute initial bounds from instance
            initial_window = env_config.get("initial_window", 0.1)
            initial_lower_normalized = np.clip(instance_unit - initial_window, 0.0, 1.0)
            initial_upper_normalized = np.clip(instance_unit + initial_window, 0.0, 1.0)
            
            rule, canonical_key = temp_env.extract_rule(
                max_features_in_rule=max_features_in_rule,
                initial_lower=initial_lower_normalized,
                initial_upper=initial_upper_normalized,
                denormalize=True
            )
        else:
            rule = "No rule extracted"
            lower_std = None
            upper_std = None
            lower_normalized = None
            upper_normalized = None
        
        # Store results
        results['explanations'][target_class] = {
            'precision': float(episode_data.get('anchor_precision', 0.0)),
            'coverage': float(episode_data.get('anchor_coverage', 0.0)),
            'rule': rule,
            'bounds_normalized': {
                'lower': lower_normalized.tolist() if lower_normalized is not None else None,
                'upper': upper_normalized.tolist() if upper_normalized is not None else None
            },
            'bounds_standardized': {
                'lower': lower_std.tolist() if lower_std is not None else None,
                'upper': upper_std.tolist() if upper_std is not None else None
            },
            'rollout_time_seconds': rollout_time
        }
        
        logger.info(f"  Precision: {results['explanations'][target_class]['precision']:.4f}")
        logger.info(f"  Coverage: {results['explanations'][target_class]['coverage']:.4f}")
        logger.info(f"  Rule: {rule[:100]}..." if len(rule) > 100 else f"  Rule: {rule}")
    
    logger.info(f"\n{'='*80}")
    logger.info("Explanation complete!")
    logger.info(f"{'='*80}")
    
    return results

