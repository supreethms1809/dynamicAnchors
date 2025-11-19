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
from datetime import datetime

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
    env,
    model,
    target_class: int,
    target_classes: List[int],
    max_steps: int = 100,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run a single rollout episode using a trained SB3 model.
    
    Args:
        env: Environment (SingleAgentAnchorEnv or MultiClassAnchorEnv wrapper)
        model: Trained SB3 model (DDPG or SAC)
        target_class: Target class for this rollout
        target_classes: List of all target classes (for one-hot encoding)
        max_steps: Maximum steps per episode
        seed: Random seed
    
    Returns:
        Dictionary with episode data (precision, coverage, bounds, etc.)
    """
    # Check if this is the MultiClassAnchorEnv wrapper
    if hasattr(env, 'base_env'):
        # MultiClassAnchorEnv wrapper - set target class and reset
        env.base_env.target_class = target_class
        obs, info = env.reset(seed=seed)
        
        # Extract base environment for metrics
        base_env = env.base_env
    else:
        # Direct SingleAgentAnchorEnv
        env.target_class = target_class
        obs, info = env.reset(seed=seed)
        base_env = env
    
    done = False
    step_count = 0
    total_reward = 0.0
    
    # Store initial state
    initial_lower = base_env.lower.copy()
    initial_upper = base_env.upper.copy()
    
    # Main rollout loop
    while not done and step_count < max_steps:
        # Get action from model (deterministic for inference)
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += float(reward)
        done = terminated or truncated
        step_count += 1
    
    # Get final metrics from base environment
    precision, coverage, details = base_env._current_metrics()
    
    # Get final bounds
    final_lower = base_env.lower.copy()
    final_upper = base_env.upper.copy()
    
    # Extract base observation (remove class encoding if present)
    if hasattr(env, 'base_env'):
        # Remove one-hot class encoding from observation
        base_obs_dim = base_env.observation_space.shape[0]
        base_obs = obs[:base_obs_dim]
    else:
        base_obs = obs
    
    episode_data = {
        "target_class": int(target_class),
        "precision": float(precision),
        "coverage": float(coverage),
        "total_reward": float(total_reward),
        "n_steps": step_count,
        "final_observation": base_obs.tolist(),
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
    steps_per_episode: int = 100,
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
    logger.info("="*80)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("="*80)
    
    # Find model file
    model_path = os.path.join(experiment_dir, "final_model.zip")
    if not os.path.exists(model_path):
        # Try best model
        best_model_path = os.path.join(experiment_dir, "best_model", "best_model.zip")
        if os.path.exists(best_model_path):
            model_path = best_model_path
            logger.info(f"Using best model: {best_model_path}")
        else:
            raise ValueError(f"Model not found in {experiment_dir}. Expected: final_model.zip or best_model/best_model.zip")
    
    # Determine algorithm from model path or experiment folder name
    experiment_name = os.path.basename(experiment_dir)
    if "ddpg" in experiment_name.lower():
        algorithm = "ddpg"
        model_class = DDPG
    elif "sac" in experiment_name.lower():
        algorithm = "sac"
        model_class = SAC
    else:
        # Try to load and infer from model
        logger.warning("Could not determine algorithm from path, trying to infer from model...")
        # Default to DDPG, will fail if wrong
        algorithm = "ddpg"
        model_class = DDPG
    
    logger.info(f"Algorithm: {algorithm.upper()}")
    logger.info(f"Loading model from: {model_path}")
    
    # Load dataset
    dataset_loader = TabularDatasetLoader(
        dataset_name=dataset_name,
        test_size=0.2,
        random_state=seed
    )
    
    dataset_loader.load_dataset()
    dataset_loader.preprocess_data()
    
    # Load classifier
    classifier_path = os.path.join(experiment_dir, "classifier.pth")
    if os.path.exists(classifier_path):
        logger.info(f"Loading classifier from: {classifier_path}")
        classifier = dataset_loader.load_classifier(
            filepath=classifier_path,
            classifier_type="dnn",
            device=device
        )
        dataset_loader.classifier = classifier
    else:
        raise ValueError(f"Classifier not found at {classifier_path}")
    
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
    
    # Create MultiClassAnchorEnv wrapper (same as training) for model loading
    # The model was trained with this wrapper, so we need to use it for inference too
    trainer = AnchorTrainerSB3(
        dataset_loader=dataset_loader,
        algorithm=algorithm,
        output_dir=experiment_dir,
        seed=seed
    )
    
    # Create the same wrapper environment used during training
    env_wrapper = trainer._create_multi_class_env(
        env_data=env_data,
        env_config=env_config,
        target_classes=target_classes,
        device=device
    )
    
    # Load model (SB3 requires an env to load)
    logger.info("Loading SB3 model...")
    model = model_class.load(model_path, env=env_wrapper, device=device)
    logger.info("✓ Model loaded successfully")
    
    # Extract rules for each class
    logger.info(f"\nExtracting rules for classes: {target_classes}")
    logger.info(f"  Instances per class: {n_instances_per_class}")
    logger.info(f"  Steps per episode: {steps_per_episode}")
    logger.info(f"  Max features in rule: {max_features_in_rule}")
    
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
        
        anchors_list = []
        rules_list = []
        precisions = []
        coverages = []
        
        for instance_idx in range(n_instances_per_class):
            # Create MultiClassAnchorEnv wrapper for this rollout
            # The model expects the extended observation space with class encoding
            rollout_seed = seed + instance_idx if seed is not None else None
            
            env = trainer._create_multi_class_env(
                env_data=env_data,
                env_config=env_config,
                target_classes=target_classes,
                device=device
            )
            
            # Set the target class index for this rollout (for inference)
            # Find the index of target_class in target_classes
            target_class_idx = target_classes.index(target_class)
            env.fixed_class_idx = target_class_idx  # Set fixed class for this rollout
            env.current_class_idx = target_class_idx
            env.base_env.target_class = target_class
            
            # Run rollout
            episode_data = run_single_agent_rollout(
                env=env,
                model=model,
                target_class=target_class,
                target_classes=target_classes,
                max_steps=steps_per_episode,
                seed=rollout_seed
            )
            
            # Clear fixed_class_idx after rollout
            env.fixed_class_idx = None
            
            precision = episode_data.get("precision", 0.0)
            coverage = episode_data.get("coverage", 0.0)
            
            precisions.append(float(precision))
            coverages.append(float(coverage))
            
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
                # Create temporary env for rule extraction
                temp_env = SingleAgentAnchorEnv(
                    X_unit=env_data["X_unit"],
                    X_std=env_data["X_std"],
                    y=env_data["y"],
                    feature_names=feature_names,
                    classifier=dataset_loader.get_classifier(),
                    device="cpu",
                    target_class=target_class,
                    env_config=env_config
                )
                temp_env.lower = lower_normalized
                temp_env.upper = upper_normalized
                
                rule = temp_env.extract_rule(
                    max_features_in_rule=max_features_in_rule,
                    denormalize=True
                )
            
            anchor_data = {
                "instance_idx": instance_idx,
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
            
            anchors_list.append(anchor_data)
            rules_list.append(rule)
        
        unique_rules = list(set([r for r in rules_list if r and r != "any values (no tightened features)"]))
        
        results["per_class_results"][class_key] = {
            "class": int(target_class),
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
        
        sa_precision = sa_data.get("precision", 0.0)
        sa_coverage = sa_data.get("coverage", 0.0)
        sa_unique_rules = sa_data.get("unique_rules_count", 0)
        
        ma_precision = ma_data.get("precision", 0.0)
        ma_coverage = ma_data.get("coverage", 0.0)
        ma_unique_rules = ma_data.get("unique_rules_count", 0)
        
        comparison["single_agent"][class_key] = {
            "precision": sa_precision,
            "coverage": sa_coverage,
            "unique_rules": sa_unique_rules,
        }
        
        comparison["multi_agent"][class_key] = {
            "precision": ma_precision,
            "coverage": ma_coverage,
            "unique_rules": ma_unique_rules,
        }
        
        comparison["differences"][class_key] = {
            "precision_diff": sa_precision - ma_precision,
            "coverage_diff": sa_coverage - ma_coverage,
            "unique_rules_diff": sa_unique_rules - ma_unique_rules,
        }
        
        logger.info(f"\n{class_key}:")
        logger.info(f"  Precision:  Single={sa_precision:.4f}, Multi={ma_precision:.4f}, Diff={sa_precision - ma_precision:.4f}")
        logger.info(f"  Coverage:   Single={sa_coverage:.4f}, Multi={ma_coverage:.4f}, Diff={sa_coverage - ma_coverage:.4f}")
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
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=["breast_cancer", "wine", "iris", "synthetic", "moons", "circles", "covtype", "housing"],
        help="Dataset name (must match training)"
    )
    
    parser.add_argument(
        "--max_features_in_rule",
        type=int,
        default=-1,
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

