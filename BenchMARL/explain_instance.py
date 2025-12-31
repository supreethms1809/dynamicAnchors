"""
Helper function to explain a new instance using trained Dynamic Anchors models.

This module provides functions to explain a new instance (unknown class) by checking
it against all trained classes/agents and returning anchor explanations for each class.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
import json
from pathlib import Path

from BenchMARL.tabular_datasets import TabularDatasetLoader
from BenchMARL.environment import AnchorEnv
from BenchMARL.inference import run_rollout_with_policy
from BenchMARL.inference import load_policy_model

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


def explain_new_instance_multi_agent(
    new_instance: np.ndarray,
    experiment_dir: str,
    dataset_name: str,
    instance_in_standardized_space: bool = True,
    steps_per_episode: Optional[int] = None,
    max_features_in_rule: int = -1,
    device: str = "cpu",
    seed: int = 42,
    exploration_mode: str = "sample",
    action_noise_scale: float = 0.05,
    mlp_config_path: str = "conf/mlp.yaml",
) -> Dict[str, Any]:
    """
    Explain a new instance by checking it against all trained classes/agents.
    
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
        exploration_mode: Exploration mode for rollouts ("sample", "mean", "noisy_mean")
        action_noise_scale: Action noise scale for exploration
        mlp_config_path: Path to MLP config file
    
    Returns:
        Dictionary with:
        - 'predicted_class': Predicted class for the instance
        - 'class_probabilities': Probability distribution over classes
        - 'explanations': Dict mapping class_id -> explanation results
            Each explanation contains:
            - 'agent_name': Agent name used
            - 'precision': Precision of the anchor
            - 'coverage': Coverage of the anchor
            - 'rule': Extracted anchor rule
            - 'bounds': Final anchor bounds (normalized and standardized)
            - 'rollout_time': Time taken for rollout
    """
    logger.info("="*80)
    logger.info("EXPLAINING NEW INSTANCE (Multi-Agent)")
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
    from BenchMARL.anchor_trainer import AnchorTrainer
    trainer = AnchorTrainer(
        dataset_loader=dataset_loader,
        algorithm="maddpg",
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
    
    # Load policies
    individual_models_dir = os.path.join(experiment_dir, "individual_models")
    index_path = os.path.join(individual_models_dir, "policies_index.json")
    
    if not os.path.exists(index_path):
        raise ValueError(f"policies_index.json not found at {index_path}")
    
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    agents_per_class = index_data.get("agents_per_class", 1)
    policies_by_class = index_data.get("policies_by_class", {})
    
    # Prepare results
    results = {
        'predicted_class': int(predicted_class),
        'class_probabilities': class_probs.tolist(),
        'explanations': {}
    }
    
    # Run inference for each class
    for target_class in target_classes:
        class_key = str(target_class)
        if class_key not in policies_by_class:
            logger.warning(f"No policies found for class {target_class}, skipping...")
            continue
        
        class_policies = policies_by_class[class_key].get("policies", [])
        if not class_policies:
            logger.warning(f"No policies in class {target_class}, skipping...")
            continue
        
        # Use first agent for this class (or you could try all agents)
        policy_info = class_policies[0]
        agent_name = policy_info.get("agent") or policy_info.get("group")
        policy_file = policy_info.get("policy_file")
        
        if not agent_name or not policy_file:
            logger.warning(f"Invalid policy info for class {target_class}, skipping...")
            continue
        
        policy_path = os.path.join(individual_models_dir, policy_file)
        if not os.path.exists(policy_path):
            logger.warning(f"Policy file not found: {policy_path}, skipping...")
            continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Explaining for Class {target_class} using agent {agent_name}")
        logger.info(f"{'='*80}")
        
        # Load policy
        metadata_path = policy_info.get("metadata_file", "")
        if metadata_path:
            metadata_path = os.path.join(individual_models_dir, metadata_path)
            if not os.path.exists(metadata_path):
                metadata_path = ""
        
        policy = load_policy_model(
            policy_path=policy_path,
            metadata_path=metadata_path,
            mlp_config_path=mlp_config_path,
            device=device
        )
        
        # Create environment for this class
        single_agent_config = {
            "X_unit": env_data["X_unit"],
            "X_std": env_data["X_std"],
            "y": env_data["y"],
            "feature_names": feature_names,
            "classifier": classifier,
            "target_classes": [target_class],
            "env_config": {**env_config, "mode": "inference", "normalize_data": False}
        }
        
        # Ensure normalization parameters are in env_config
        if "X_min" not in single_agent_config["env_config"]:
            single_agent_config["env_config"]["X_min"] = X_min
        if "X_range" not in single_agent_config["env_config"]:
            single_agent_config["env_config"]["X_range"] = X_range
        
        env = AnchorEnv(**single_agent_config)
        
        # CRITICAL: Set x_star_unit BEFORE reset() for instance-based mode
        env.x_star_unit[agent_name] = instance_unit.copy()
        
        # Run rollout
        import time
        rollout_start = time.perf_counter()
        
        episode_data = run_rollout_with_policy(
            env=env,
            policy=policy,
            agent_id=agent_name,
            max_steps=steps_per_episode,
            device=device,
            seed=seed,
            exploration_mode=exploration_mode,
            action_noise_scale=action_noise_scale,
            verbose_logging=False
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
            temp_env = AnchorEnv(
                X_unit=env_data["X_unit"],
                X_std=env_data["X_std"],
                y=env_data["y"],
                feature_names=feature_names,
                classifier=classifier,
                target_classes=[target_class],
                env_config={**env_config, "mode": "inference", "normalize_data": False}
            )
            temp_env.lower[agent_name] = lower_normalized
            temp_env.upper[agent_name] = upper_normalized
            
            # Compute initial bounds from instance
            initial_window = env_config.get("initial_window", 0.1)
            initial_lower_normalized = np.clip(instance_unit - initial_window, 0.0, 1.0)
            initial_upper_normalized = np.clip(instance_unit + initial_window, 0.0, 1.0)
            
            rule = temp_env.extract_rule(
                agent_name,
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
            'agent_name': agent_name,
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

