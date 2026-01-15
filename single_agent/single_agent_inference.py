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


def canonicalize_rule_key(
    lower_bounds_normalized: np.ndarray,
    upper_bounds_normalized: np.ndarray,
    min_width: float = 0.05,
    epsilon: Optional[float] = None
) -> str:
    """
    Create a canonical rule key from normalized bounds for deduplication.
    
    Steps:
    1. Quantize bounds to epsilon grid
    2. Drop near-full-range features (lower <= eps & upper >= 1 - eps)
    3. Sort by feature index
    4. Return string key
    
    Args:
        lower_bounds_normalized: Lower bounds in [0, 1] normalized space
        upper_bounds_normalized: Upper bounds in [0, 1] normalized space
        min_width: Minimum width (used to compute epsilon if not provided)
        epsilon: Quantization step size (default: max(1e-3, min_width/4))
    
    Returns:
        Canonical rule key as string
    """
    lower = np.array(lower_bounds_normalized, dtype=np.float32).flatten()
    upper = np.array(upper_bounds_normalized, dtype=np.float32).flatten()
    
    # Compute epsilon for quantization
    if epsilon is None:
        epsilon = max(1e-3, min_width / 4.0)
    
    # Quantize bounds to epsilon grid
    lower_quantized = np.round(lower / epsilon) * epsilon
    upper_quantized = np.round(upper / epsilon) * epsilon
    
    # Clip to valid range [0, 1]
    lower_quantized = np.clip(lower_quantized, 0.0, 1.0)
    upper_quantized = np.clip(upper_quantized, 0.0, 1.0)
    
    # Drop near-full-range features (features that haven't been tightened)
    # A feature is near-full-range if: lower <= eps AND upper >= 1 - eps
    tightened_mask = ~((lower_quantized <= epsilon) & (upper_quantized >= 1.0 - epsilon))
    tightened_indices = np.where(tightened_mask)[0]
    
    if len(tightened_indices) == 0:
        # No tightened features - return special key
        return "any_values"
    
    # Extract only tightened features and sort by feature index
    tightened_lower = lower_quantized[tightened_indices]
    tightened_upper = upper_quantized[tightened_indices]
    
    # Sort by feature index for canonical ordering
    sort_order = np.argsort(tightened_indices)
    tightened_indices_sorted = tightened_indices[sort_order]
    tightened_lower_sorted = tightened_lower[sort_order]
    tightened_upper_sorted = tightened_upper[sort_order]
    
    # Create canonical key: "f1:l1:u1;f2:l2:u2;..."
    key_parts = [f"{idx}:{lo:.6f}:{hi:.6f}" 
                 for idx, lo, hi in zip(tightened_indices_sorted, tightened_lower_sorted, tightened_upper_sorted)]
    
    return ";".join(key_parts)


def compute_box_iou(
    lower1: np.ndarray,
    upper1: np.ndarray,
    lower2: np.ndarray,
    upper2: np.ndarray
) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes in normalized [0,1] space.
    
    Args:
        lower1: Lower bounds of first box (n_features,)
        upper1: Upper bounds of first box (n_features,)
        lower2: Lower bounds of second box (n_features,)
        upper2: Upper bounds of second box (n_features,)
    
    Returns:
        IoU value in [0, 1]
    """
    lower1 = np.array(lower1, dtype=np.float32).flatten()
    upper1 = np.array(upper1, dtype=np.float32).flatten()
    lower2 = np.array(lower2, dtype=np.float32).flatten()
    upper2 = np.array(upper2, dtype=np.float32).flatten()
    
    if len(lower1) != len(upper1) or len(lower2) != len(upper2) or len(lower1) != len(lower2):
        return 0.0
    
    # Compute intersection: max of lower bounds, min of upper bounds
    intersect_lower = np.maximum(lower1, lower2)
    intersect_upper = np.minimum(upper1, upper2)
    
    # Intersection volume (only where intersect_lower < intersect_upper)
    intersect_widths = np.maximum(0.0, intersect_upper - intersect_lower)
    intersection_volume = np.prod(intersect_widths)
    
    # Compute union volumes
    volume1 = np.prod(np.maximum(0.0, upper1 - lower1))
    volume2 = np.prod(np.maximum(0.0, upper2 - lower2))
    union_volume = volume1 + volume2 - intersection_volume
    
    if union_volume <= 0.0:
        return 0.0
    
    iou = intersection_volume / union_volume
    return float(iou)


def nms_deduplicate_anchors(
    anchors_list: List[Dict[str, Any]],
    iou_threshold: float = 0.9,
    key_precision: str = "precision_recomputed",
    key_coverage: str = "coverage_recomputed",
    fallback_precision: str = "precision",
    fallback_coverage: str = "coverage"
) -> List[Dict[str, Any]]:
    """
    Non-Maximum Suppression (NMS) for anchor deduplication using IoU.
    
    Ranks anchors by precision (tie-break by coverage), then keeps a rule only if
    its box IoU with all kept rules is below the threshold.
    
    Args:
        anchors_list: List of anchor dictionaries with bounds and metrics
        iou_threshold: IoU threshold for considering anchors as duplicates (default: 0.9)
        key_precision: Key for precision metric (prefer recomputed)
        key_coverage: Key for coverage metric (prefer recomputed)
        fallback_precision: Fallback key for precision if primary not found
        fallback_coverage: Fallback key for coverage if primary not found
    
    Returns:
        Filtered list of anchors after NMS
    """
    if len(anchors_list) == 0:
        return []
    
    # Extract bounds and metrics for each anchor
    anchor_data = []
    for i, anchor in enumerate(anchors_list):
        lower = anchor.get("lower_bounds_normalized")
        upper = anchor.get("upper_bounds_normalized")
        
        if lower is None or upper is None:
            # Skip anchors without bounds
            continue
        
        lower = np.array(lower, dtype=np.float32)
        upper = np.array(upper, dtype=np.float32)
        
        # Get precision and coverage (prefer recomputed, fallback to primary)
        precision = anchor.get(key_precision, anchor.get(fallback_precision, 0.0))
        coverage = anchor.get(key_coverage, anchor.get(fallback_coverage, 0.0))
        
        anchor_data.append({
            "index": i,
            "anchor": anchor,
            "lower": lower,
            "upper": upper,
            "precision": float(precision),
            "coverage": float(coverage),
            "score": float(precision)  # Primary sort by precision
        })
    
    if len(anchor_data) == 0:
        return []
    
    # Sort by precision (descending), tie-break by coverage (descending)
    anchor_data.sort(key=lambda x: (x["precision"], x["coverage"]), reverse=True)
    
    # Greedy NMS: keep anchors with IoU < threshold with all previously kept anchors
    kept = []
    for current in anchor_data:
        is_duplicate = False
        
        # Check IoU with all previously kept anchors
        for kept_anchor in kept:
            iou = compute_box_iou(
                current["lower"], current["upper"],
                kept_anchor["lower"], kept_anchor["upper"]
            )
            
            if iou >= iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept.append(current)
    
    # Return original anchor dictionaries in order
    return [anchor_data[i]["anchor"] for i in sorted(a["index"] for a in kept)]


def compute_anchor_metrics_on_full_dataset(
    lower_bounds_normalized: np.ndarray,
    upper_bounds_normalized: np.ndarray,
    X_full_unit: np.ndarray,
    X_full_std: np.ndarray,
    original_instance_unit: np.ndarray,
    original_instance_std: np.ndarray,
    original_prediction: int,
    full_predictions: np.ndarray,
    classifier=None,
    device: str = "cpu",
    target_class: int = None,
    validate_with_classifier: bool = False
) -> Tuple[float, float]:
    """
    Compute precision and coverage for a single anchor on the full dataset.
    
    This follows the original Anchors paper methodology:
    - Precision: Of all instances in the full dataset that satisfy the anchor,
                  what fraction have the same prediction as the original instance?
                  Formula: P(f(x) = f(x_original) | x satisfies anchor)
    - Coverage: What fraction of instances in the full dataset satisfy the anchor?
                Formula: P(x satisfies anchor)
    
    Args:
        lower_bounds_normalized: Lower bounds of anchor box in normalized [0,1] space
        upper_bounds_normalized: Upper bounds of anchor box in normalized [0,1] space
        X_full_unit: Full dataset (train + test combined) in normalized [0,1] space (for box matching)
        X_full_std: Full dataset (train + test combined) in standardized space (for classifier predictions)
        original_instance_unit: The instance that this anchor explains (in normalized [0,1] space)
        original_instance_std: The instance that this anchor explains (in standardized space)
        original_prediction: The model's prediction for the original instance
        full_predictions: Pre-computed predictions for all instances in X_full_std (numpy array of shape (n_samples,))
        classifier: Trained classifier model (optional, only used for validation if validate_with_classifier=True)
        device: Device to run classifier on (only used for validation)
        target_class: Target class for this anchor (optional, only used for validation/logging)
        validate_with_classifier: If True, validate predictions using classifier (expensive, for debugging only)
    
    Returns:
        Tuple of (precision, coverage, n_in_box) on the full dataset
        - precision: P(f(x) = f(x_original) | x satisfies anchor) for instance-based
        - coverage: P(x satisfies anchor)
        - n_in_box: Number of instances that satisfy the anchor (support count for weighting)
    """
    # Determine which instances satisfy the anchor
    # An instance satisfies the anchor if it's within all feature bounds
    lower = np.array(lower_bounds_normalized, dtype=np.float32)
    upper = np.array(upper_bounds_normalized, dtype=np.float32)
    
    # Ensure bounds are valid
    if lower.shape[0] != X_full_unit.shape[1] or upper.shape[0] != X_full_unit.shape[1]:
        logger.warning(f"  Bounds shape mismatch: lower.shape={lower.shape}, upper.shape={upper.shape}, X_full_unit.shape[1]={X_full_unit.shape[1]}")
        return 0.0, 0.0
    
    # Check which samples satisfy the anchor (all features must be within bounds)
    anchor_mask = np.all((X_full_unit >= lower) & (X_full_unit <= upper), axis=1)
    
    # Coverage: fraction of instances in full dataset that satisfy the anchor
    n_total = len(X_full_unit)
    n_in_anchor = anchor_mask.sum()
    if n_total == 0:
        coverage = 0.0
    else:
        coverage = float(n_in_anchor / n_total)
    
    # Precision: fraction of instances that satisfy anchor and have same prediction as original
    if n_in_anchor == 0:
        # No instances satisfy the anchor
        precision = 0.0
    else:
        # Use pre-computed predictions (much faster than running classifier per anchor)
        # Extract predictions for samples that satisfy the anchor
        predictions_in_anchor = full_predictions[anchor_mask]
        
        # VALIDATION: Check if original instance is in the anchor box
        # The original instance should always be in the box (environment ensures this)
        original_in_box = np.all((original_instance_unit >= lower) & (original_instance_unit <= upper))
        if not original_in_box:
            logger.warning(
                f"  WARNING: Original instance is NOT in anchor box! This should not happen. "
                f"Original instance bounds check: lower={lower[:3]} (first 3), upper={upper[:3]} (first 3), "
                f"original_instance_unit={original_instance_unit[:3]} (first 3)"
            )
        elif validate_with_classifier and classifier is not None:
            # Optional validation: verify predictions using classifier (expensive, for debugging only)
            # Verify original instance's prediction matches stored original_prediction
            original_tensor = torch.from_numpy(original_instance_std.astype(np.float32)).unsqueeze(0).to(device)
            classifier.eval()
            with torch.no_grad():
                original_logits = classifier(original_tensor)
                original_pred_computed = int(original_logits.argmax(dim=1).cpu().numpy()[0])
            
            if original_pred_computed != original_prediction:
                logger.warning(
                    f"  WARNING: Original instance's computed prediction ({original_pred_computed}) "
                    f"!= stored original_prediction ({original_prediction}). This may cause precision calculation issues."
                )
            
            # Use target_class for diagnostics: verify original prediction matches target class (if expected)
            if target_class is not None and original_pred_computed != target_class:
                logger.debug(
                    f"  Note: Original instance prediction ({original_pred_computed}) != target_class ({target_class}). "
                    f"This is expected when using prediction routing or when instance belongs to different class."
                )
        
        # Count how many have the same prediction as the original instance
        n_matching_pred = (predictions_in_anchor == original_prediction).sum()
        precision = float(n_matching_pred / n_in_anchor)
        
        # DEBUG: Log precision calculation details when precision is exactly 1.0 with multiple instances
        if precision == 1.0 and n_in_anchor > 1:
            unique_preds, counts = np.unique(predictions_in_anchor, return_counts=True)
            logger.debug(
                f"  DEBUG: Precision=1.0 with n_in_anchor={n_in_anchor}, n_matching_pred={n_matching_pred}, "
                f"coverage={coverage:.4f}. Unique predictions in box: {dict(zip(unique_preds, counts))}, "
                f"original_prediction={original_prediction}"
            )
        
        # VALIDATION: If precision is 0.0 but coverage > 0, log a warning
        # At minimum, if original instance is in box and its prediction matches, precision should be >= 1/n_in_anchor
        if precision == 0.0 and coverage > 0.0:
            logger.warning(
                f"  WARNING: Precision is 0.0 but coverage is {coverage:.4f} (n_in_anchor={n_in_anchor}). "
                f"This means no samples in the anchor box match the original prediction ({original_prediction}). "
                f"Original instance in box: {original_in_box}"
            )
    
    # Sanity checks
    assert 0.0 <= precision <= 1.0, f"Precision {precision} out of range [0, 1]"
    assert 0.0 <= coverage <= 1.0, f"Coverage {coverage} out of range [0, 1]"
    
    return precision, coverage, int(n_in_anchor)


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


def _process_instances_for_class(
    sampled_indices: np.ndarray,
    target_class: int,
    model,
    X_data_unit: np.ndarray,
    X_data_std: np.ndarray,
    X_full_unit: np.ndarray,
    X_full_std: np.ndarray,
    full_predictions: np.ndarray,
    env_data: Dict[str, Any],
    env_config: Dict[str, Any],
    feature_names: List[str],
    dataset_loader,
    classifier,
    n_rollouts_per_instance: int,
    steps_per_episode: Optional[int],
    max_features_in_rule: int,
    coverage_on_all_data: bool,
    eval_on_test_data: bool,
    use_full_for_sampling: bool,
    filter_low_quality_rollouts: bool,
    min_precision_threshold: Optional[float],
    min_coverage_threshold: float,
    use_weighted_average: bool,
    device: str,
    seed: Optional[int]
) -> Dict[str, Any]:
    """
    Process instances for a given class: run rollouts, recompute metrics, filter anchors, compute averages.
    
    This is a helper function extracted from extract_rules_single_agent to be reused by both
    traditional mode (ground truth routing) and prediction routing mode.
    
    Returns a dictionary with processed results including anchors, metrics, and instance-level statistics.
    """
    anchors_list = []
    rules_list = []
    precisions = []
    coverages = []
    rollout_times = []
    per_instance_results = []
    
    n_samples = len(sampled_indices)
    
    # For each sampled instance, run multiple rollouts
    for instance_idx_in_range, data_instance_idx in enumerate(sampled_indices):
        # Get the actual instance from the dataset
        x_instance = X_data_unit[data_instance_idx]
        x_instance_std = X_data_std[data_instance_idx]
        
        # Compute original prediction directly from the instance vector (avoid fragile index mapping)
        classifier.eval()
        with torch.no_grad():
            instance_tensor = torch.from_numpy(x_instance_std.astype(np.float32)).unsqueeze(0).to(device)
            logits = classifier(instance_tensor)
            original_prediction = int(logits.argmax(dim=1).cpu().numpy()[0])
        
        logger.info(f"\n  Instance {instance_idx_in_range + 1}/{n_samples} (data index {data_instance_idx}): Running {n_rollouts_per_instance} rollouts")
        
        # Store rollouts for this instance
        instance_rollouts = []
        
        # Set mode to "inference" for rule extraction
        if coverage_on_all_data:
            # Option: Use full dataset (train + test) if explicitly requested
            if env_data.get("X_test_unit") is not None:
                X_all_unit = np.vstack([env_data["X_unit"], env_data.get("X_test_unit", [])])
                X_all_std = np.vstack([env_data["X_std"], env_data.get("X_test_std", [])])
                y_all = np.concatenate([env_data["y"], env_data.get("y_test", [])])
                env_X_unit = X_all_unit
                env_X_std = X_all_std
                env_y = y_all
                env_eval_on_test = False  # Use combined data
                use_full_dataset = True
                n_samples_for_coverage = len(env_y)
            else:
                env_X_unit = env_data["X_unit"]
                env_X_std = env_data["X_std"]
                env_y = env_data["y"]
                env_eval_on_test = False
                use_full_dataset = False
                n_samples_for_coverage = len(env_y) if env_data.get("X_unit") is not None else None
        elif eval_on_test_data and env_data.get("X_test_unit") is not None:
            # Default: Use test data for metrics computation (follows original Anchor paper)
            env_X_unit = env_data["X_test_unit"]
            env_X_std = env_data["X_test_std"]
            env_y = env_data["y_test"]
            env_eval_on_test = True  # Use test data for evaluation
            use_full_dataset = False
            n_samples_for_coverage = len(env_y) if env_data.get("X_test_unit") is not None else None
        else:
            # Fallback: Use training data if test data not available
            env_X_unit = env_data["X_unit"]
            env_X_std = env_data["X_std"]
            env_y = env_data["y"]
            env_eval_on_test = False
            use_full_dataset = False
            n_samples_for_coverage = len(env_y) if env_data.get("X_unit") is not None else None
        
        # Set min_coverage_floor dynamically
        config_default = env_config.get("min_coverage_floor", 0.005)
        if n_samples_for_coverage is not None and n_samples_for_coverage > 0:
            min_coverage_floor = 1.0 / n_samples_for_coverage
            min_coverage_floor = max(min_coverage_floor, 1e-6)
        else:
            min_coverage_floor = config_default
        min_coverage_floor = max(min_coverage_floor, 1e-6)
        
        inference_env_config = {
            **env_config, 
            "mode": "inference",
            "eval_on_test_data": env_eval_on_test,
            "min_coverage_floor": min_coverage_floor
        }
        logger.debug(f"  Set min_coverage_floor={min_coverage_floor:.6f} (n_samples={n_samples_for_coverage if n_samples_for_coverage is not None else 'unknown'})")
        
        if not use_full_dataset and env_eval_on_test:
            inference_env_config.update({
                "X_test_unit": env_data.get("X_test_unit"),
                "X_test_std": env_data.get("X_test_std"),
                "y_test": env_data.get("y_test")
            })
        
        # Track anchors seen so far for this instance (for early stopping)
        seen_canonical_keys = set()
        seen_anchors_for_instance = []  # Store anchors with bounds for IoU checking
        
        # Track anchors seen so far for this instance (for early stopping duplicates)
        seen_canonical_keys = set()
        seen_anchors_for_instance = []  # Store anchors with bounds for IoU checking
        
        # Run multiple rollouts for this instance
        for rollout_idx in range(n_rollouts_per_instance):
            rollout_seed = seed + (instance_idx_in_range * n_rollouts_per_instance) + rollout_idx if seed is not None else None
            
            env = SingleAgentAnchorEnv(
                X_unit=env_X_unit,
                X_std=env_X_std,
                y=env_y,
                feature_names=feature_names,
                classifier=classifier,
                device=device,
                target_class=target_class,
                env_config=inference_env_config
            )
            
            # CRITICAL: Set x_star_unit for instance-based rollout
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
            
            logger.info(f"    Rollout {rollout_idx + 1}/{n_rollouts_per_instance} for instance {data_instance_idx}: "
                      f"Precision={precision:.4f}, Coverage={coverage:.4f}, "
                      f"Class-Conditional Coverage={coverage_class_conditional:.4f}")
        
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
                temp_env = SingleAgentAnchorEnv(
                    X_unit=env_X_unit,
                    X_std=env_X_std,
                    y=env_y,
                    feature_names=feature_names,
                    classifier=classifier,
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
                
                rule, canonical_key = temp_env.extract_rule(
                    max_features_in_rule=max_features_in_rule,
                    initial_lower=initial_lower_normalized,
                    initial_upper=initial_upper_normalized,
                    denormalize=True
                )
            else:
                # No bounds - use default canonical key
                canonical_key = "any_values"
            
            anchor_data = {
                "instance_idx": instance_idx_in_range,
                "rollout_idx": rollout_idx,
                "data_instance_idx": int(data_instance_idx),
                "rollout_type": "instance_based",
                # Rollout-estimated metrics (computed during rollout using perturbation samples)
                "precision_rollout_estimated": float(precision),
                "coverage_rollout_estimated": float(coverage),
                "coverage_class_conditional_rollout_estimated": float(coverage_class_conditional),
                "total_reward": float(episode_data.get("total_reward", 0.0)),
                "n_steps": int(episode_data.get("n_steps", 0)),
                "rollout_time_seconds": float(rollout_time),
                "rule": rule,
                "canonical_key": canonical_key,  # Canonical key for deduplication
                "original_prediction": int(original_prediction),
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
            
            # Check for early stopping: if anchor is too similar to existing ones, skip adding it
            # (This reduces duplicates at source)
            early_stop_iou_threshold = env_config.get("early_stop_iou_threshold", 0.95)  # Default: 0.95 (very similar)
            is_duplicate = False
            
            if lower_normalized is not None and upper_normalized is not None:
                # Check canonical key first (fast check)
                if canonical_key in seen_canonical_keys:
                    is_duplicate = True
                    logger.debug(f"    Rollout {rollout_idx + 1}: Skipping duplicate (canonical key match)")
                else:
                    # Check IoU with existing anchors (more expensive but catches near-duplicates)
                    for seen_anchor in seen_anchors_for_instance:
                        seen_lower = seen_anchor["lower"]
                        seen_upper = seen_anchor["upper"]
                        iou = compute_box_iou(lower_normalized, upper_normalized, seen_lower, seen_upper)
                        if iou >= early_stop_iou_threshold:
                            is_duplicate = True
                            logger.debug(f"    Rollout {rollout_idx + 1}: Skipping near-duplicate (IoU={iou:.4f} >= {early_stop_iou_threshold})")
                            break
            
            if not is_duplicate:
                instance_rollouts.append(anchor_data)
                if canonical_key not in seen_canonical_keys:
                    seen_canonical_keys.add(canonical_key)
                if lower_normalized is not None and upper_normalized is not None:
                    seen_anchors_for_instance.append({
                        "lower": lower_normalized.copy(),
                        "upper": upper_normalized.copy()
                    })
        
        # After all rollouts for this instance are done:
        # 0. (Optional) Reduce duplicates at source: keep top-K anchors per instance by precision/coverage score
        #    This reduces the number of anchors to recompute and deduplicate later
        top_k_per_instance = env_config.get("top_k_anchors_per_instance", None)
        if top_k_per_instance is not None and top_k_per_instance > 0 and len(instance_rollouts) > top_k_per_instance:
            # Score anchors by precision * coverage (simple product score)
            scores = []
            for rollout_data in instance_rollouts:
                prec = rollout_data.get("precision_rollout_estimated", 0.0)
                cov = rollout_data.get("coverage_rollout_estimated", 0.0)
                scores.append(prec * cov)
            
            # Sort by score (descending) and keep top-K
            sorted_indices = np.argsort(scores)[::-1]
            instance_rollouts = [instance_rollouts[i] for i in sorted_indices[:top_k_per_instance]]
            logger.debug(f"  Kept top {top_k_per_instance} anchors per instance (from {len(sorted_indices)} rollouts)")
        
        # After all rollouts for this instance are done:
        # 1. Recompute precision and coverage on full dataset for each anchor
        # 2. Filter out bad anchors for this instance
        # 3. Average filtered anchors for this instance
        logger.info(f"\n  Processing {len(instance_rollouts)} rollouts for instance {data_instance_idx}...")
        
        # x_instance_std was already computed directly from X_data_std[data_instance_idx] above
        # This avoids fragile index mapping - we use the instance vector directly rather than looking it up
        # The instance is the same regardless of whether it came from train/test/full dataset
        
        # Step 1: Recompute metrics on appropriate dataset based on flags
        valid_rollouts = []
        # Select dataset for recomputation based on eval_on_test_data and coverage_on_all_data flags
        # This ensures metrics match training logs and user expectations
        if coverage_on_all_data:
            # Use full dataset (train + test) if explicitly requested
            if env_data.get("X_test_unit") is not None:
                X_recompute_unit = np.vstack([env_data["X_unit"], env_data["X_test_unit"]])
                X_recompute_std = np.vstack([env_data["X_std"], env_data["X_test_std"]])
                y_recompute = np.concatenate([env_data["y"], env_data["y_test"]])
                recompute_data_source = "full (train + test)"
            else:
                X_recompute_unit = env_data["X_unit"]
                X_recompute_std = env_data["X_std"]
                y_recompute = env_data["y"]
                recompute_data_source = "training"
        elif eval_on_test_data and env_data.get("X_test_unit") is not None:
            # Use test data only (matches training evaluation and original Anchor paper)
            X_recompute_unit = env_data["X_test_unit"]
            X_recompute_std = env_data["X_test_std"]
            y_recompute = env_data["y_test"]
            recompute_data_source = "test"
        else:
            # Fallback: Use training data if test data not available
            X_recompute_unit = env_data["X_unit"]
            X_recompute_std = env_data["X_std"]
            y_recompute = env_data["y"]
            recompute_data_source = "training"
        
        logger.debug(f"  Recomputing metrics on {recompute_data_source} data (eval_on_test_data={eval_on_test_data}, coverage_on_all_data={coverage_on_all_data})")
        
        # Compute predictions for recompute dataset once (much faster than per-anchor)
        # This is a critical performance optimization - avoids running classifier per anchor
        # Check if passed-in full_predictions aligns with recompute dataset (can reuse if it matches)
        use_passed_predictions = False
        if full_predictions is not None and len(full_predictions) == len(X_recompute_std):
            # Check if X_full_std matches X_recompute_std (they should be the same arrays or same content)
            if (X_full_std is X_recompute_std) or (np.array_equal(X_full_std, X_recompute_std)):
                full_predictions_recompute = full_predictions
                use_passed_predictions = True
                logger.debug(f"  Reusing passed-in full_predictions for recompute dataset ({len(full_predictions)} samples)")
        
        if not use_passed_predictions:
            # Compute predictions for recompute dataset
            classifier.eval()
            with torch.no_grad():
                X_recompute_tensor = torch.from_numpy(X_recompute_std.astype(np.float32)).to(device)
                full_logits_recompute = classifier(X_recompute_tensor)
                full_predictions_recompute = full_logits_recompute.argmax(dim=1).cpu().numpy()
        
        for rollout_data in instance_rollouts:
            if rollout_data.get("lower_bounds_normalized") is not None and rollout_data.get("upper_bounds_normalized") is not None:
                lower_norm = np.array(rollout_data["lower_bounds_normalized"], dtype=np.float32)
                upper_norm = np.array(rollout_data["upper_bounds_normalized"], dtype=np.float32)
                
                # Recompute precision and coverage on selected dataset (respects eval_on_test_data and coverage_on_all_data)
                # Use pre-computed predictions for performance (avoids running classifier per anchor)
                prec_full, cov_full, n_in_box = compute_anchor_metrics_on_full_dataset(
                    lower_bounds_normalized=lower_norm,
                    upper_bounds_normalized=upper_norm,
                    X_full_unit=X_recompute_unit,
                    X_full_std=X_recompute_std,
                    original_instance_unit=x_instance,
                    original_instance_std=x_instance_std,
                    original_prediction=rollout_data["original_prediction"],
                    full_predictions=full_predictions_recompute,
                    classifier=classifier,
                    device=device,
                    target_class=target_class,
                    validate_with_classifier=False
                )
                
                # Recompute class-conditional coverage on selected dataset (for consistency)
                # P(x in box | y = target_class) on selected dataset
                in_box_mask = np.all((X_recompute_unit >= lower_norm) & (X_recompute_unit <= upper_norm), axis=1)
                class_mask = (y_recompute == target_class)
                n_class_samples = class_mask.sum()
                if n_class_samples > 0:
                    n_class_in_box = (in_box_mask & class_mask).sum()
                    cov_class_conditional_full = float(n_class_in_box / n_class_samples)
                else:
                    cov_class_conditional_full = 0.0
                
                # DEBUG: Log precision change when significant
                prec_rollout = rollout_data.get("precision_rollout_estimated", 0.0)
                if abs(prec_full - prec_rollout) > 0.05:  # Log if difference > 5%
                    logger.info(
                        f"    Rollout {rollout_data['rollout_idx']}: Precision changed from {prec_rollout:.4f} "
                        f"(rollout-estimated, using perturbation samples) to {prec_full:.4f} "
                        f"(recomputed on full dataset with actual instances)"
                    )
                
                # Store recomputed metrics (computed on full dataset)
                rollout_data["precision_recomputed"] = float(prec_full)
                rollout_data["coverage_recomputed"] = float(cov_full)
                rollout_data["coverage_class_conditional_recomputed"] = float(cov_class_conditional_full)
                rollout_data["n_in_box"] = int(n_in_box)  # Support count for weighting
                rollout_data["n_class_in_box"] = int(n_class_in_box) if n_class_samples > 0 else 0  # Class-conditional support count
                
                # Set primary metrics to recomputed values (these are the metrics we use for evaluation)
                rollout_data["precision"] = float(prec_full)
                rollout_data["coverage"] = float(cov_full)
                rollout_data["coverage_class_conditional"] = float(cov_class_conditional_full)
                
                # Append to valid_rollouts (has recomputed metrics)
                valid_rollouts.append(rollout_data)
            else:
                logger.warning(f"    Rollout {rollout_data['rollout_idx']} has no normalized bounds, skipping")
                continue
        
        # Step 2: Filter bad anchors for this instance (from valid_rollouts)
        kept_rollouts = []
        if filter_low_quality_rollouts:
            if min_precision_threshold is None:
                precision_target = env_config.get("precision_target", 0.95)
                min_precision_threshold = precision_target * 0.8
            
            for rollout_data in valid_rollouts:
                anchor_precision = rollout_data["precision_recomputed"]
                anchor_coverage = rollout_data["coverage_recomputed"]
                
                if anchor_precision >= min_precision_threshold and anchor_coverage >= min_coverage_threshold:
                    kept_rollouts.append(rollout_data)
                    logger.debug(f"    ✓ Kept rollout {rollout_data['rollout_idx']}: precision={anchor_precision:.4f}, coverage={anchor_coverage:.4f}")
                else:
                    logger.debug(f"    ✗ Discarded rollout {rollout_data['rollout_idx']}: precision={anchor_precision:.4f} (< {min_precision_threshold:.3f}) or "
                               f"coverage={anchor_coverage:.4f} (< {min_coverage_threshold:.3f})")
            
            if len(kept_rollouts) == 0:
                logger.warning(f"  WARNING: All {len(valid_rollouts)} valid rollouts for instance {data_instance_idx} were filtered out! Using all valid rollouts as fallback.")
                kept_rollouts = valid_rollouts
                if len(kept_rollouts) == 0:
                    logger.warning(f"  WARNING: No valid rollouts found! Using all rollouts (may have missing metrics).")
                    kept_rollouts = instance_rollouts
        else:
            kept_rollouts = valid_rollouts
        
        # Average kept anchors for this instance (use recomputed metrics)
        instance_precisions = [r["precision_recomputed"] for r in kept_rollouts if "precision_recomputed" in r and r.get("precision_recomputed") is not None]
        instance_coverages = [r["coverage_recomputed"] for r in kept_rollouts if "coverage_recomputed" in r and r.get("coverage_recomputed") is not None]
        
        if len(instance_precisions) != len(kept_rollouts) or len(instance_coverages) != len(kept_rollouts):
            logger.warning(f"  WARNING: Instance {instance_idx_in_range + 1}: Only {len(instance_precisions)}/{len(kept_rollouts)} rollouts have valid precision metrics, "
                         f"{len(instance_coverages)}/{len(kept_rollouts)} have valid coverage metrics")
        
        if instance_precisions and instance_coverages:
            avg_instance_precision = float(np.mean(instance_precisions))
            avg_instance_coverage = float(np.mean(instance_coverages))
            # Calculate median and IQR for robustness (helps avoid being dominated by outliers)
            instance_precision_median = float(np.median(instance_precisions))
            instance_precision_q25 = float(np.percentile(instance_precisions, 25))
            instance_precision_q75 = float(np.percentile(instance_precisions, 75))
            instance_precision_iqr = instance_precision_q75 - instance_precision_q25
            
            instance_coverage_median = float(np.median(instance_coverages))
            instance_coverage_q25 = float(np.percentile(instance_coverages, 25))
            instance_coverage_q75 = float(np.percentile(instance_coverages, 75))
            instance_coverage_iqr = instance_coverage_q75 - instance_coverage_q25
            
            logger.info(f"  Instance {instance_idx_in_range + 1} average (from {len(kept_rollouts)} kept rollouts): "
                      f"Precision={avg_instance_precision:.4f} (median={instance_precision_median:.4f}, IQR=[{instance_precision_q25:.4f}, {instance_precision_q75:.4f}]), "
                      f"Coverage={avg_instance_coverage:.4f} (median={instance_coverage_median:.4f}, IQR=[{instance_coverage_q25:.4f}, {instance_coverage_q75:.4f}])")
            
            per_instance_results.append({
                "instance_idx": instance_idx_in_range,
                "data_instance_idx": int(data_instance_idx),
                "n_rollouts_total": len(instance_rollouts),
                "n_rollouts_valid": len(valid_rollouts),
                "n_rollouts_kept": len(kept_rollouts),
                "avg_precision": avg_instance_precision,
                "avg_coverage": avg_instance_coverage,
                "precision_median": instance_precision_median,
                "precision_q25": instance_precision_q25,
                "precision_q75": instance_precision_q75,
                "precision_iqr": instance_precision_iqr,
                "coverage_median": instance_coverage_median,
                "coverage_q25": instance_coverage_q25,
                "coverage_q75": instance_coverage_q75,
                "coverage_iqr": instance_coverage_iqr,
            })
        else:
            logger.warning(f"  No valid metrics for instance {instance_idx_in_range + 1}, skipping")
        
        # Add kept rollouts to main lists (use recomputed metrics)
        # Note: We'll deduplicate using canonical keys and NMS after all instances are processed
        for rollout_data in kept_rollouts:
            anchors_list.append(rollout_data)
            rules_list.append(rollout_data["rule"])
            if "precision_recomputed" in rollout_data:
                precisions.append(rollout_data["precision_recomputed"])
                coverages.append(rollout_data["coverage_recomputed"])
            rollout_times.append(rollout_data.get("rollout_time_seconds", 0.0))
    
    # Compute instance-level metrics from per-instance results
    if per_instance_results:
        instance_level_precisions = [r["avg_precision"] for r in per_instance_results]
        instance_level_coverages = [r["avg_coverage"] for r in per_instance_results]
        
        # Support-weighted means: weight by n_in_box (support count) instead of coverage values
        # This follows the metrics strategy: weight by actual number of samples covered
        # CRITICAL FIX: Build weights from anchors_list (all anchors across all instances), not just kept_rollouts from last instance
        if use_weighted_average:
            # Extract n_in_box from anchors_list (support count for each anchor across ALL instances)
            # For instance-based: weight by n_in_box (number of samples covered by each anchor)
            # We need to aggregate n_in_box per instance to match instance_level_precisions/coverages
            # Since instance_level_precisions/coverages are per-instance averages, we should weight by
            # the total n_in_box for each instance (sum of n_in_box for all anchors from that instance)
            n_in_box_per_instance = []
            instance_idx_to_n_in_box = {}
            
            # Aggregate n_in_box by instance_idx
            for anchor in anchors_list:
                instance_idx = anchor.get("instance_idx", None)
                n_in_box = anchor.get("n_in_box", None)
                if instance_idx is not None and n_in_box is not None:
                    if instance_idx not in instance_idx_to_n_in_box:
                        instance_idx_to_n_in_box[instance_idx] = 0
                    instance_idx_to_n_in_box[instance_idx] += int(n_in_box)
            
            # Build weights list matching per_instance_results order
            for result in per_instance_results:
                instance_idx = result.get("instance_idx", None)
                if instance_idx is not None and instance_idx in instance_idx_to_n_in_box:
                    n_in_box_per_instance.append(instance_idx_to_n_in_box[instance_idx])
                else:
                    # Fallback: estimate from coverage if n_in_box not available
                    cov = result.get("avg_coverage", 0.0)
                    n_in_box_per_instance.append(max(1, int(cov * 1000)))  # Rough estimate
            
            if n_in_box_per_instance and len(n_in_box_per_instance) == len(instance_level_precisions):
                weights = np.array(n_in_box_per_instance, dtype=np.float64)
                weights = np.maximum(weights, 0.0)
                weights_sum = weights.sum()
                if weights_sum > 0:
                    weights = weights / weights_sum
                    instance_precision = float(np.average(instance_level_precisions, weights=weights))
                    instance_coverage = float(np.average(instance_level_coverages, weights=weights))
                else:
                    instance_precision = float(np.mean(instance_level_precisions)) if instance_level_precisions else 0.0
                    instance_coverage = float(np.mean(instance_level_coverages)) if instance_level_coverages else 0.0
            else:
                # Fallback to simple mean if n_in_box not available or lengths don't match
                logger.warning(f"  Support-weighted average: n_in_box list length ({len(n_in_box_per_instance) if n_in_box_per_instance else 0}) "
                             f"!= instance metrics length ({len(instance_level_precisions)}). Using unweighted mean.")
                instance_precision = float(np.mean(instance_level_precisions)) if instance_level_precisions else 0.0
                instance_coverage = float(np.mean(instance_level_coverages)) if instance_level_coverages else 0.0
        else:
            instance_precision = float(np.mean(instance_level_precisions)) if instance_level_precisions else 0.0
            instance_coverage = float(np.mean(instance_level_coverages)) if instance_level_coverages else 0.0
    else:
        # Fallback: Compute from all rollouts
        # Extract metrics directly from anchors_list (same source as precisions/coverages lists)
        # Prefer recomputed metrics, fallback to primary metrics (which should be recomputed)
        fallback_precisions = [a.get("precision_recomputed", a.get("precision", 0.0)) for a in anchors_list if "precision_recomputed" in a or "precision" in a]
        fallback_coverages = [a.get("coverage_recomputed", a.get("coverage", 0.0)) for a in anchors_list if "coverage_recomputed" in a or "coverage" in a]
        
        # Support-weighted means: weight by n_in_box (support count) instead of coverage values
        if use_weighted_average and fallback_precisions and fallback_coverages:
            # Extract n_in_box from anchors_list (support count for weighting)
            n_in_box_list = []
            for anchor in anchors_list:
                n_in_box = anchor.get("n_in_box", None)
                if n_in_box is not None:
                    n_in_box_list.append(int(n_in_box))
                else:
                    # Fallback: use coverage as estimate if n_in_box not available
                    cov = anchor.get("coverage_recomputed", anchor.get("coverage", 0.0))
                    n_in_box_list.append(max(1, int(cov * 1000)))  # Rough estimate
            
            if n_in_box_list and len(n_in_box_list) == len(fallback_precisions):
                weights = np.array(n_in_box_list, dtype=np.float64)
                weights = np.maximum(weights, 0.0)
                weights_sum = weights.sum()
                if weights_sum > 0:
                    weights = weights / weights_sum
                    instance_precision = float(np.average(fallback_precisions, weights=weights))
                    instance_coverage = float(np.average(fallback_coverages, weights=weights))
                else:
                    instance_precision = float(np.mean(fallback_precisions)) if fallback_precisions else 0.0
                    instance_coverage = float(np.mean(fallback_coverages)) if fallback_coverages else 0.0
            else:
                instance_precision = float(np.mean(fallback_precisions)) if fallback_precisions else 0.0
                instance_coverage = float(np.mean(fallback_coverages)) if fallback_coverages else 0.0
        else:
            instance_precision = float(np.mean(fallback_precisions)) if fallback_precisions else 0.0
            instance_coverage = float(np.mean(fallback_coverages)) if fallback_coverages else 0.0
    
    # Compute average class-conditional coverage
    # For class-conditional coverage: weight by class-conditional covered count (n_class_in_box)
    coverages_class_conditional = []
    n_class_in_box_list = []  # Class-conditional support counts for weighting
    for anchor_data in anchors_list:
        if "coverage_class_conditional" in anchor_data:
            coverages_class_conditional.append(anchor_data["coverage_class_conditional"])
            # For class-conditional coverage, weight by class-conditional covered count
            n_class_in_box = anchor_data.get("n_class_in_box", None)
            if n_class_in_box is not None:
                n_class_in_box_list.append(int(n_class_in_box))
            else:
                # Fallback: estimate from n_in_box and coverage_class_conditional ratio
                n_in_box = anchor_data.get("n_in_box", None)
                if n_in_box is not None:
                    cov_class_cond = anchor_data.get("coverage_class_conditional", 0.0)
                    cov_overall = anchor_data.get("coverage_recomputed", anchor_data.get("coverage", 1e-6))
                    if cov_overall > 0:
                        n_class_in_box_est = max(1, int(n_in_box * (cov_class_cond / cov_overall)))
                    else:
                        n_class_in_box_est = max(1, n_in_box)
                    n_class_in_box_list.append(n_class_in_box_est)
                else:
                    # Final fallback: use coverage_class_conditional as estimate
                    cov_class_cond = anchor_data.get("coverage_class_conditional", 0.0)
                    n_class_in_box_list.append(max(1, int(cov_class_cond * 1000)))  # Rough estimate
    
    if use_weighted_average and coverages_class_conditional and n_class_in_box_list and len(n_class_in_box_list) == len(coverages_class_conditional):
        weights = np.array(n_class_in_box_list, dtype=np.float64)
        weights = np.maximum(weights, 0.0)
        weights_sum = weights.sum()
        if weights_sum > 0:
            weights = weights / weights_sum
            instance_coverage_class_conditional = float(np.average(coverages_class_conditional, weights=weights))
        else:
            instance_coverage_class_conditional = float(np.mean(coverages_class_conditional)) if coverages_class_conditional else 0.0
    else:
        instance_coverage_class_conditional = float(np.mean(coverages_class_conditional)) if coverages_class_conditional else 0.0
    
    # Calculate overall median and IQR if we have per-instance results
    instance_precision_median = None
    instance_precision_iqr = None
    instance_coverage_median = None
    instance_coverage_iqr = None
    if per_instance_results:
        instance_level_precisions = [r.get("avg_precision", 0.0) for r in per_instance_results]
        instance_level_coverages = [r.get("avg_coverage", 0.0) for r in per_instance_results]
        if instance_level_precisions:
            instance_precision_median = float(np.median(instance_level_precisions))
            instance_precision_q25 = float(np.percentile(instance_level_precisions, 25))
            instance_precision_q75 = float(np.percentile(instance_level_precisions, 75))
            instance_precision_iqr = instance_precision_q75 - instance_precision_q25
        if instance_level_coverages:
            instance_coverage_median = float(np.median(instance_level_coverages))
            instance_coverage_q25 = float(np.percentile(instance_level_coverages, 25))
            instance_coverage_q75 = float(np.percentile(instance_level_coverages, 75))
            instance_coverage_iqr = instance_coverage_q75 - instance_coverage_q25
    
    return {
        "anchors_list": anchors_list,
        "rules_list": rules_list,
        "precisions": precisions,
        "coverages": coverages,
        "rollout_times": rollout_times,
        "per_instance_results": per_instance_results,
        "instance_precision": instance_precision,
        "instance_coverage": instance_coverage,
        "instance_coverage_class_conditional": instance_coverage_class_conditional,
        "instance_precision_median": instance_precision_median,
        "instance_precision_iqr": instance_precision_iqr,
        "instance_coverage_median": instance_coverage_median,
        "instance_coverage_iqr": instance_coverage_iqr,
    }


def extract_rules_single_agent(
    experiment_dir: str,
    dataset_name: str,
    max_features_in_rule: int = -1,
    steps_per_episode: Optional[int] = None,  # If None, will read from env_config.max_cycles
    n_instances_per_class: int = 10,  # Number of instances to sample (default: 10)
    n_rollouts_per_instance: int = 10,  # Number of rollouts per instance (default: 20)
    eval_on_test_data: bool = True,
    coverage_on_all_data: bool = False,  # If True, compute coverage on all data (train+test combined, matches baseline)
    sample_from_full_dataset: bool = False,  # If True, sample instances from full dataset (train+test) instead of just test/train
    filter_by_prediction: bool = True,  # If True, filter instances where classifier prediction matches target_class (for fair comparison with baseline, set to False)
    use_prediction_routing: bool = True,  # If True (default), route instances to policies based on classifier predictions (realistic evaluation). If False, use ground truth labels (traditional evaluation).
    use_weighted_average: bool = False,  # If True, use coverage-weighted average instead of simple arithmetic mean
    filter_low_quality_rollouts: bool = True,  # If True, filter out low-precision/low-coverage rollouts before averaging
    min_precision_threshold: Optional[float] = None,  # Minimum precision to keep (default: precision_target * 0.8)
    min_coverage_threshold: float = 0.01,  # Minimum coverage to keep (default: 0.01)
    output_dir: Optional[str] = None,
    seed: int = 42,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Extract anchor rules using a trained single-agent SB3 model.
    
    IMPORTANT NOTE ON POLICY SELECTION:
    
    When use_prediction_routing=True (default, realistic evaluation):
    - Samples instances from ALL classes
    - Gets classifier predictions for all instances
    - Routes each instance to the policy corresponding to its predicted class
    - This provides a more realistic evaluation scenario where we don't know the ground truth class
    
    When use_prediction_routing=False (traditional evaluation):
    - Processes each class separately
    - For class 0: Samples instances with ground truth label == 0, uses class 0's policy
    - For class 1: Samples instances with ground truth label == 1, uses class 1's policy
    - This measures "how well does each policy explain instances of its own class?"
    
    The prediction routing approach is more realistic but may show different metrics
    (e.g., lower precision if classifier makes mistakes). Use traditional mode for
    per-class performance evaluation, and prediction routing for realistic system evaluation.
    
    Args:
        experiment_dir: Path to SB3 experiment directory
        dataset_name: Name of the dataset
        max_features_in_rule: Maximum features to include in rules
        steps_per_episode: Maximum steps per rollout
        n_instances_per_class: Number of instances to sample per class (default: 10)
        n_rollouts_per_instance: Number of rollouts to run per instance (default: 20)
        eval_on_test_data: Whether to evaluate on test data
        coverage_on_all_data: If True, compute coverage on all data (train+test combined)
        sample_from_full_dataset: If True, sample instances from full dataset (train+test) instead of just test/train
        filter_by_prediction: If True, filter instances where classifier prediction matches target_class.
                             Set to False for fair comparison with baseline (baseline doesn't filter by prediction).
        use_weighted_average: If True, compute weighted average using coverage as weights (anchors with higher coverage get more weight).
                             If False, use simple arithmetic mean (default, for comparison with baseline).
        filter_low_quality_rollouts: If True, filter out low-precision/low-coverage rollouts before averaging and rule extraction.
        min_precision_threshold: Minimum precision to keep (default: precision_target * 0.8, or 0.76 if precision_target=0.95).
        min_coverage_threshold: Minimum coverage to keep (default: 0.01).
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
    
    # Use the classifier instance loaded above everywhere (env + recompute)
    # This ensures we use the same classifier instance throughout
    
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
        
        # Use adaptive clustering: adjust cluster count based on dataset size
        # and check for scattered data distribution
        cluster_centroids_per_class = compute_cluster_centroids_per_class(
            X_unit=X_cluster,
            y=y_cluster,
            n_clusters_per_class=n_clusters_per_class,
            random_state=seed if seed is not None else 42,
            auto_adapt_clusters=True,  # Adapt cluster count to dataset size
            check_data_scatter=True    # Check if data is scattered (use mean if so)
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
    
    # Classifier is already loaded earlier (line ~892), use the same instance everywhere
    # This ensures consistency between environment and recompute operations
    
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
        # Use the same classifier instance loaded earlier
        env = SingleAgentAnchorEnv(
            X_unit=env_data["X_unit"],
            X_std=env_data["X_std"],
            y=env_data["y"],
            feature_names=feature_names,
            classifier=classifier,
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
    logger.info(f"  Instances per class: {n_instances_per_class} (will sample this many instances)")
    logger.info(f"  Rollouts per instance: {n_rollouts_per_instance} (will run this many rollouts for each instance)")
    logger.info(f"  Total rollouts per class: {n_instances_per_class * n_rollouts_per_instance}")
    logger.info(f"  Steps per episode: {steps_per_episode}")
    logger.info(f"  Max features in rule: {max_features_in_rule}")
    
    # Start overall timing
    overall_start_time = time.perf_counter()
    
    # Classifier is already loaded earlier (before model loading), use the same instance everywhere
    
    results = {
        "per_class_results": {},
        "metadata": {
            "dataset": dataset_name,
            "experiment_dir": experiment_dir,
            "algorithm": algorithm,
            "target_classes": target_classes,
            "max_features_in_rule": max_features_in_rule,
            "eval_on_test_data": eval_on_test_data,
            "coverage_on_all_data": coverage_on_all_data,
            "sample_from_full_dataset": sample_from_full_dataset,
            "filter_by_prediction": filter_by_prediction,
            "use_prediction_routing": use_prediction_routing,
            "n_instances_per_class": n_instances_per_class,
            "n_rollouts_per_instance": n_rollouts_per_instance,
            "steps_per_episode": steps_per_episode,
            "model_type": "single_agent_sb3",
        },
    }
    
    # Prepare dataset for recomputation (respects eval_on_test_data and coverage_on_all_data)
    # This will be used when calling _process_instances_for_class, which will select the appropriate
    # dataset for recomputation based on the flags passed to it
    if coverage_on_all_data:
        # Use full dataset (train + test) if explicitly requested
        if env_data.get("X_test_unit") is not None:
            X_full_unit = np.vstack([env_data["X_unit"], env_data["X_test_unit"]])
            X_full_std = np.vstack([env_data["X_std"], env_data["X_test_std"]])
            y_full = np.concatenate([env_data["y"], env_data["y_test"]])
        else:
            X_full_unit = env_data["X_unit"]
            X_full_std = env_data["X_std"]
            y_full = env_data["y"]
    elif eval_on_test_data and env_data.get("X_test_unit") is not None:
        # Use test data only (matches training evaluation)
        X_full_unit = env_data["X_test_unit"]
        X_full_std = env_data["X_test_std"]
        y_full = env_data["y_test"]
    else:
        # Fallback: Use training data
        X_full_unit = env_data["X_unit"]
        X_full_std = env_data["X_std"]
        y_full = env_data["y"]
    
    # Branch based on prediction routing mode
    if use_prediction_routing:
        logger.info("\n" + "="*80)
        logger.info("USING PREDICTION ROUTING MODE (Realistic Evaluation)")
        logger.info("="*80)
        logger.info("  - Samples instances from ALL classes")
        logger.info("  - Routes each instance to policy based on classifier prediction")
        logger.info("  - More realistic for deployment scenarios")
        logger.info("="*80)
        
        # PREDICTION ROUTING MODE: Sample from all classes and route by predictions
        # Determine dataset to sample from
        # Respect eval_on_test_data and coverage_on_all_data flags to avoid mixing instance source and evaluation set
        use_full_for_sampling = sample_from_full_dataset
        # Only force full dataset if explicitly requested (coverage_on_all_data) or user-set (sample_from_full_dataset)
        # Don't force it based on n_rollouts_per_instance to avoid mixing train+test sampling with test-only evaluation
        if coverage_on_all_data and env_data.get("X_test_unit") is not None:
            use_full_for_sampling = True
        
        if use_full_for_sampling and env_data.get("X_test_unit") is not None:
            X_data_unit = X_full_unit
            X_data_std = X_full_std
            data_source_name = "full (train + test)"
        elif eval_on_test_data and env_data.get("X_test_unit") is not None:
            X_data_unit = env_data["X_test_unit"]
            X_data_std = env_data["X_test_std"]
            data_source_name = "test"
        else:
            X_data_unit = env_data["X_unit"]
            X_data_std = env_data["X_std"]
            data_source_name = "training"
        
        # Sample instances from ALL classes (not filtered by class)
        n_total_instances = len(X_data_unit)
        total_samples_needed = n_instances_per_class * len(target_classes)
        n_samples_to_take = min(total_samples_needed, n_total_instances)
        
        rng_for_sampling = np.random.default_rng(seed if seed is not None else 42)
        all_sampled_indices = rng_for_sampling.choice(n_total_instances, size=n_samples_to_take, replace=False)
        
        logger.info(f"  Sampled {n_samples_to_take} instances from {data_source_name} data (from all classes)")
        
        # Get classifier predictions for all sampled instances
        X_sampled_std = X_data_std[all_sampled_indices]
        classifier.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_sampled_std.astype(np.float32)).to(device)
            logits = classifier(X_tensor)
            predictions = logits.argmax(dim=1).cpu().numpy()
        
        # Group instances by predicted class
        predicted_classes_to_indices = {}
        for idx, pred_class in zip(all_sampled_indices, predictions):
            if pred_class not in predicted_classes_to_indices:
                predicted_classes_to_indices[pred_class] = []
            predicted_classes_to_indices[pred_class].append(idx)
        
        logger.info(f"  Instances routed to classes: {[f'{cls}: {len(indices)} instances' for cls, indices in predicted_classes_to_indices.items()]}")
        
        # Initialize aggregation variables for prediction routing mode (aggregate across all predicted classes)
        anchors_list = []
        rules_list = []
        precisions = []
        coverages = []
        rollout_times = []
        per_instance_results = []
        
        # Process each predicted class
        for predicted_class in sorted(predicted_classes_to_indices.keys()):
            if predicted_class not in target_classes:
                logger.warning(f"  Predicted class {predicted_class} not in target_classes, skipping...")
                continue
            
            if predicted_class not in models:
                logger.warning(f"  Model for predicted class {predicted_class} not found, skipping...")
                continue
            
            # Use predicted_class as the target_class for processing
            target_class = predicted_class
            class_key = f"class_{target_class}"
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing instances routed to predicted class {target_class} ({len(predicted_classes_to_indices[predicted_class])} instances)")
            logger.info(f"{'='*80}")
            
            model = models[target_class]
            sampled_indices = np.array(predicted_classes_to_indices[predicted_class])
            n_samples = len(sampled_indices)
            
            # Set up variables needed for instance processing
            # Use the data source we sampled from
            env_X_unit = X_data_unit
            env_X_std = X_data_std
            
            # Determine env_y based on data source
            if use_full_for_sampling and env_data.get("X_test_unit") is not None:
                env_y = y_full
                use_full_dataset_for_env = True
            elif eval_on_test_data and env_data.get("X_test_unit") is not None:
                env_y = env_data["y_test"]
                use_full_dataset_for_env = False
            else:
                env_y = env_data["y"]
                use_full_dataset_for_env = False
            
            class_start_time = time.perf_counter()
            
            logger.info(f"  Processing {n_samples} instances routed to predicted class {target_class}")
            logger.info(f"  Methodology (following original Anchors paper):")
            logger.info(f"    - For each instance: run {n_rollouts_per_instance} rollouts")
            logger.info(f"    - Recompute precision/coverage on FULL dataset (train + test) for each anchor")
            logger.info(f"    - Filter out bad anchors for each instance (precision/coverage thresholds)")
            logger.info(f"    - Average filtered anchors per instance -> per-instance metrics")
            logger.info(f"    - Average across instances -> final instance-level metrics")
            
            # Store per-instance results
            per_instance_results = []
            
            # For prediction routing, we need to process instances using the same logic as traditional mode
            # The instance processing code (starting around line 974 in traditional mode) needs to be executed here
            # TODO: Extract instance processing into a helper function to avoid code duplication
            # For now, we'll set up the variables and note that the processing logic should be the same
            # The key difference is in index mapping: in prediction routing, sampled_indices are already 
            # indices into X_data_unit, and we need to map them to full dataset indices correctly
            
            # Prepare dataset for recomputing metrics (respects eval_on_test_data and coverage_on_all_data)
            # Note: _process_instances_for_class will compute predictions for the recompute dataset internally
            # We don't need to compute full_predictions here - it will be computed once inside _process_instances_for_class
            if coverage_on_all_data:
                # Use full dataset (train + test) if explicitly requested
                if env_data.get("X_test_unit") is not None:
                    X_full_unit = np.vstack([env_data["X_unit"], env_data["X_test_unit"]])
                    X_full_std = np.vstack([env_data["X_std"], env_data["X_test_std"]])
                else:
                    X_full_unit = env_data["X_unit"]
                    X_full_std = env_data["X_std"]
            elif eval_on_test_data and env_data.get("X_test_unit") is not None:
                # Use test data only (matches training evaluation)
                X_full_unit = env_data["X_test_unit"]
                X_full_std = env_data["X_test_std"]
            else:
                # Fallback: Use training data
                X_full_unit = env_data["X_unit"]
                X_full_std = env_data["X_std"]
            
            # Note: full_predictions will be computed inside _process_instances_for_class for X_recompute_std
            # We pass None here to indicate it should be computed (it already does this internally)
            # This avoids duplicate computation - predictions are computed once per instance inside _process_instances_for_class
            
            # Process instances using helper function (same as traditional mode)
            process_results = _process_instances_for_class(
                sampled_indices=sampled_indices,
                target_class=target_class,
                model=model,
                X_data_unit=X_data_unit,
                X_data_std=X_data_std,
                X_full_unit=X_full_unit,
                X_full_std=X_full_std,
                full_predictions=None,  # Will be computed inside _process_instances_for_class for recompute dataset
                env_data=env_data,
                env_config=env_config,
                feature_names=feature_names,
                dataset_loader=dataset_loader,
                classifier=classifier,
                n_rollouts_per_instance=n_rollouts_per_instance,
                steps_per_episode=steps_per_episode,
                max_features_in_rule=max_features_in_rule,
                coverage_on_all_data=coverage_on_all_data,
                eval_on_test_data=eval_on_test_data,
                use_full_for_sampling=use_full_for_sampling,
                filter_low_quality_rollouts=filter_low_quality_rollouts,
                min_precision_threshold=min_precision_threshold,
                min_coverage_threshold=min_coverage_threshold,
                use_weighted_average=use_weighted_average,
                device=device,
                seed=seed
            )
            
            # Extract results for this predicted class
            anchors_list_per_class = process_results["anchors_list"]
            rules_list_per_class = process_results["rules_list"]
            precisions_per_class = process_results["precisions"]
            coverages_per_class = process_results["coverages"]
            rollout_times_per_class = process_results["rollout_times"]
            per_instance_results_per_class = process_results["per_instance_results"]
            instance_precision_per_class = process_results["instance_precision"]
            instance_coverage_per_class = process_results["instance_coverage"]
            instance_coverage_class_conditional_per_class = process_results["instance_coverage_class_conditional"]
            
            logger.info(f"  Instance-level metrics for predicted class {target_class}: "
                      f"Precision={instance_precision_per_class:.4f}, "
                      f"Coverage={instance_coverage_per_class:.4f}")
            
            # Compute unique rules for this class
            # Deduplicate using canonical keys (more robust than rule strings)
            # Step 1: Deduplicate by canonical key
            canonical_keys_seen = set()
            unique_anchors_by_key = {}
            for anchor in anchors_list_per_class:
                canonical_key = anchor.get("canonical_key")
                if canonical_key is None:
                    # Fallback: use rule string if canonical key not available
                    rule_str = anchor.get("rule", "")
                    if rule_str and rule_str != "any values (no tightened features)":
                        # Use rule string as fallback key
                        canonical_key = f"rule:{rule_str}"
                    else:
                        canonical_key = "any_values"
                
                # Keep the anchor with best precision (tie-break by coverage) for each canonical key
                if canonical_key not in canonical_keys_seen:
                    canonical_keys_seen.add(canonical_key)
                    unique_anchors_by_key[canonical_key] = anchor
                else:
                    # Compare with existing anchor - keep the one with better precision
                    existing = unique_anchors_by_key[canonical_key]
                    existing_prec = existing.get("precision_recomputed", existing.get("precision", 0.0))
                    new_prec = anchor.get("precision_recomputed", anchor.get("precision", 0.0))
                    if new_prec > existing_prec:
                        unique_anchors_by_key[canonical_key] = anchor
                    elif new_prec == existing_prec:
                        # Tie-break by coverage
                        existing_cov = existing.get("coverage_recomputed", existing.get("coverage", 0.0))
                        new_cov = anchor.get("coverage_recomputed", anchor.get("coverage", 0.0))
                        if new_cov > existing_cov:
                            unique_anchors_by_key[canonical_key] = anchor
            
            # Step 2: Apply NMS for near-duplicate suppression (IoU-based)
            nms_iou_threshold = env_config.get("nms_iou_threshold", 0.9)  # Default: 0.9
            anchors_after_nms = nms_deduplicate_anchors(
                list(unique_anchors_by_key.values()),
                iou_threshold=nms_iou_threshold
            )
            
            # Update anchors_list_per_class with deduplicated anchors
            anchors_list_per_class = anchors_after_nms
            
            # Extract unique rules from deduplicated anchors (for backward compatibility)
            unique_rules_per_class = list(set([
                anchor.get("rule", "") 
                for anchor in anchors_list_per_class 
                if anchor.get("rule") and anchor.get("rule") != "any values (no tightened features)"
            ]))
            
            # Calculate duplicate ratio: (total_rules - unique_rules) / total_rules
            total_rules_per_class = len(rules_list_per_class) if rules_list_per_class else 0
            unique_rules_count_per_class = len(unique_rules_per_class)
            duplicate_ratio_per_class = (total_rules_per_class - unique_rules_count_per_class) / total_rules_per_class if total_rules_per_class > 0 else 0.0
            
            # Compute top-K rules by score: precision * (1 + coverage)
            # This provides a clear ranking of best rules
            top_k_rules_count = env_config.get("top_k_rules_by_score", 10)  # Default: top 10
            rule_scores = []
            for anchor in anchors_list_per_class:
                precision = anchor.get("precision_recomputed", anchor.get("precision", 0.0))
                coverage = anchor.get("coverage_recomputed", anchor.get("coverage", 0.0))
                score = precision * (1 + coverage)  # Score formula: precision * (1 + coverage)
                rule_scores.append({
                    "rule": anchor.get("rule", ""),
                    "precision": float(precision),
                    "coverage": float(coverage),
                    "score": float(score),
                    "canonical_key": anchor.get("canonical_key", "")
                })
            
            # Sort by score (descending) and take top-K
            rule_scores.sort(key=lambda x: x["score"], reverse=True)
            top_k_rules_per_class = rule_scores[:top_k_rules_count]
            
            # Compute timing metrics for this class
            avg_rollout_time_per_class = float(np.mean(rollout_times_per_class)) if rollout_times_per_class else 0.0
            total_rollout_time_per_class = float(np.sum(rollout_times_per_class)) if rollout_times_per_class else 0.0
            class_total_time_per_class = time.perf_counter() - class_start_time
            
            # Save instance-based results for this predicted class to per_class_results
            # NOTE: class_precision and class_coverage (union metrics) will be set later from class-based anchors
            results["per_class_results"][class_key] = {
                "class": int(target_class),
                # Instance-level metrics (averaged across all instances for this predicted class)
                "instance_precision": instance_precision_per_class,
                "instance_coverage": instance_coverage_per_class,
                "instance_coverage_class_conditional": instance_coverage_class_conditional_per_class,
                "instance_precision_std": float(np.std(precisions_per_class)) if len(precisions_per_class) > 1 else 0.0,
                "instance_coverage_std": float(np.std(coverages_per_class)) if len(coverages_per_class) > 1 else 0.0,
                # Class-level metrics (union of class-based anchors, will be set later)
                "class_precision": 0.0,  # Will be set later from class-based union
                "class_coverage": 0.0,   # Will be set later from class-based union
                # Legacy fields for backward compatibility (same as instance-level)
                "precision": instance_precision_per_class,
                "coverage": instance_coverage_per_class,
                "precision_std": float(np.std(precisions_per_class)) if len(precisions_per_class) > 1 else 0.0,
                "coverage_std": float(np.std(coverages_per_class)) if len(coverages_per_class) > 1 else 0.0,
                "n_episodes": len(anchors_list_per_class),
                "rules": rules_list_per_class,
                "unique_rules": unique_rules_per_class,
                "unique_rules_count": unique_rules_count_per_class,
                "total_rules_count": total_rules_per_class,
                "duplicate_ratio": float(duplicate_ratio_per_class),
                "top_k_rules_by_score": top_k_rules_per_class,
                "top_k_rules_count": len(top_k_rules_per_class),
                "anchors": anchors_list_per_class,
                # Timing metrics
                "avg_rollout_time_seconds": avg_rollout_time_per_class,
                "total_rollout_time_seconds": total_rollout_time_per_class,
                "class_total_time_seconds": float(class_total_time_per_class),
            }
            
            # Also aggregate for overall summary (optional, for debugging/logging)
            anchors_list.extend(anchors_list_per_class)
            rules_list.extend(rules_list_per_class)
            precisions.extend(precisions_per_class)
            coverages.extend(coverages_per_class)
            rollout_times.extend(rollout_times_per_class)
            per_instance_results.extend(per_instance_results_per_class)
    
    else:
        # TRADITIONAL MODE: Process each class separately using ground truth labels
        logger.info("\n" + "="*80)
        logger.info("USING TRADITIONAL MODE (Per-Class Evaluation)")
        logger.info("="*80)
        logger.info("  - Processes each class separately")
        logger.info("  - Samples instances with ground truth label == target_class")
        logger.info("  - Uses that class's policy to explain those instances")
        logger.info("="*80)
        
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
            # Respect eval_on_test_data and coverage_on_all_data flags to avoid mixing instance source and evaluation set
            # When eval_on_test_data=True, sample from test data to match evaluation set (avoids depressing coverage)
            # When coverage_on_all_data=True, sample from full dataset to match evaluation set
            use_full_for_sampling = sample_from_full_dataset
            # Only force full dataset if explicitly requested (coverage_on_all_data) or user-set (sample_from_full_dataset)
            # Don't force it based on n_rollouts_per_instance to avoid mixing train+test sampling with test-only evaluation
            if coverage_on_all_data and env_data.get("X_test_unit") is not None:
                use_full_for_sampling = True
                if not sample_from_full_dataset:
                    logger.info(f"  Using full dataset (train + test) for instance sampling (coverage_on_all_data=True)")
            
            # Check if we need more instances than available in test data
            if eval_on_test_data and env_data.get("X_test_unit") is not None and not use_full_for_sampling:
                test_class_mask = (env_data["y_test"] == target_class)
                n_test_instances = test_class_mask.sum()
                # If test data doesn't have enough instances, use full dataset
                if n_test_instances < n_instances_per_class:
                    logger.info(f"  Test data has only {n_test_instances} instances for class {target_class}, "
                              f"but {n_instances_per_class} instances requested. Using full dataset (train + test) for sampling.")
                    use_full_for_sampling = True
            
            if use_full_for_sampling and env_data.get("X_test_unit") is not None:
                # Sample from full dataset (train + test combined)
                class_mask_train = (env_data["y"] == target_class)
                class_mask_test = (env_data["y_test"] == target_class)
                class_instances_train = np.where(class_mask_train)[0]
                class_instances_test = np.where(class_mask_test)[0]
                # Combine train and test instances (test indices offset by train size)
                n_train = len(env_data["y"])
                class_instances_test_offset = class_instances_test + n_train
                class_instances = np.concatenate([class_instances_train, class_instances_test_offset])
                # Combine data
                X_data_unit = np.vstack([env_data["X_unit"], env_data["X_test_unit"]])
                X_data_std = np.vstack([env_data["X_std"], env_data["X_test_std"]])
                env_X_unit = X_data_unit
                env_X_std = X_data_std
                env_y = np.concatenate([env_data["y"], env_data["y_test"]])
                data_source_name = "full (train + test)"
            elif eval_on_test_data and env_data.get("X_test_unit") is not None:
                class_mask = (env_data["y_test"] == target_class)
                class_instances = np.where(class_mask)[0]
                X_data_unit = env_data["X_test_unit"]
                X_data_std = env_data["X_test_std"]
                env_X_unit = env_data["X_test_unit"]
                env_X_std = env_data["X_test_std"]
                env_y = env_data["y_test"]
                data_source_name = "test"
            else:
                class_mask = (env_data["y"] == target_class)
                class_instances = np.where(class_mask)[0]
                X_data_unit = env_data["X_unit"]
                X_data_std = env_data["X_std"]
                env_X_unit = env_data["X_unit"]
                env_X_std = env_data["X_std"]
                env_y = env_data["y"]
                data_source_name = "training"
            
            if len(class_instances) == 0:
                logger.warning(f"  No instances found for class {target_class} in {data_source_name} data, skipping instance-based rollouts...")
            else:
                # Filter instances where classifier prediction matches target_class (optional, for fair comparison)
                # NOTE: Baseline (Static Anchors) does NOT filter by prediction, so set filter_by_prediction=False for fair comparison
                # When filter_by_prediction=True, this ensures original_prediction == target_class, preventing precision calculation issues
                # This matches the filtering done during training (anchor_trainer_sb3.py lines 483-500)
                if filter_by_prediction:
                    # Use the classifier instance already loaded at the top of extract_rules_single_agent
                    if classifier is not None and len(class_instances) > 0:
                        # Get predictions for all class instances
                        X_class_std = X_data_std[class_instances]
                        # torch is already imported at module level, no need to import again
                        with torch.no_grad():
                            X_tensor = torch.from_numpy(X_class_std.astype(np.float32)).to(device)
                            logits = classifier(X_tensor)
                            probs = torch.softmax(logits, dim=-1).cpu().numpy()
                            predictions = np.argmax(probs, axis=1)
                        
                        # Filter: keep only instances where prediction matches target_class
                        prediction_match_mask = (predictions == target_class)
                        matching_indices = class_instances[prediction_match_mask]
                        n_matching = len(matching_indices)
                        n_total = len(class_instances)
                        
                        logger.info(f"  Filtered instances: {n_matching}/{n_total} instances have prediction matching target_class {target_class}")
                        
                        if n_matching == 0:
                            logger.warning(f"  No instances found where classifier prediction matches target_class {target_class}, skipping instance-based rollouts...")
                            class_instances = np.array([], dtype=int)
                        else:
                            class_instances = matching_indices
                    else:
                        logger.warning(f"  Classifier not available, skipping prediction-based filtering (may cause precision issues)")
                else:
                    logger.info(f"  Prediction filtering disabled (filter_by_prediction=False) for fair comparison with baseline")
                    logger.info(f"  Using all {len(class_instances)} instances with ground truth label matching target_class {target_class}")
                
                if len(class_instances) > 0:
                    # Sample instances (same approach as multi-agent)
                    n_samples = min(n_instances_per_class, len(class_instances))
                    rng_for_sampling = np.random.default_rng(seed if seed is not None else 42)
                    sampled_indices = rng_for_sampling.choice(class_instances, size=n_samples, replace=False)
                    
                    logger.info(f"  Sampling {n_samples} instances from {data_source_name} data for instance-based rollouts")
                    logger.info(f"  Methodology (following original Anchors paper):")
                    logger.info(f"    - For each instance: run {n_rollouts_per_instance} rollouts")
                    # Log which dataset will be used for recomputation
                    if coverage_on_all_data:
                        recompute_dataset_name = "FULL dataset (train + test)"
                    elif eval_on_test_data:
                        recompute_dataset_name = "TEST dataset"
                    else:
                        recompute_dataset_name = "TRAINING dataset"
                    logger.info(f"    - Recompute precision/coverage on {recompute_dataset_name} for each anchor (eval_on_test_data={eval_on_test_data}, coverage_on_all_data={coverage_on_all_data})")
                    logger.info(f"    - Filter out bad anchors for each instance (precision/coverage thresholds)")
                    logger.info(f"    - Average filtered anchors per instance -> per-instance metrics")
                    logger.info(f"    - Average across instances -> final instance-level metrics")
                    if n_samples < n_instances_per_class:
                        logger.warning(f"    WARNING: Only {len(class_instances)} instances available for class {target_class} in {data_source_name} data, "
                                    f"limiting to {n_samples} instances (requested {n_instances_per_class})")
                    else:
                        logger.info(f"    Using {n_samples} instances as requested (found {len(class_instances)} instances available)")
                    
                    # Prepare dataset for recomputing metrics (respects eval_on_test_data and coverage_on_all_data)
                    if coverage_on_all_data:
                        # Use full dataset (train + test) if explicitly requested
                        if env_data.get("X_test_unit") is not None:
                            X_full_unit = np.vstack([env_data["X_unit"], env_data["X_test_unit"]])
                            X_full_std = np.vstack([env_data["X_std"], env_data["X_test_std"]])
                        else:
                            X_full_unit = env_data["X_unit"]
                            X_full_std = env_data["X_std"]
                    elif eval_on_test_data and env_data.get("X_test_unit") is not None:
                        # Use test data only (matches training evaluation)
                        X_full_unit = env_data["X_test_unit"]
                        X_full_std = env_data["X_test_std"]
                    else:
                        # Fallback: Use training data
                        X_full_unit = env_data["X_unit"]
                        X_full_std = env_data["X_std"]
                    
                    # Use the classifier instance already loaded at the top of extract_rules_single_agent
                    classifier.eval()
                    with torch.no_grad():
                        X_full_tensor = torch.from_numpy(X_full_std.astype(np.float32)).to(device)
                        full_logits = classifier(X_full_tensor)
                        full_predictions = full_logits.argmax(dim=1).cpu().numpy()
                    
                    # Process instances using helper function
                    process_results = _process_instances_for_class(
                        sampled_indices=sampled_indices,
                        target_class=target_class,
                        model=model,
                        X_data_unit=X_data_unit,
                        X_data_std=X_data_std,
                        X_full_unit=X_full_unit,
                        X_full_std=X_full_std,
                        full_predictions=full_predictions,
                        env_data=env_data,
                        env_config=env_config,
                        feature_names=feature_names,
                        dataset_loader=dataset_loader,
                        classifier=classifier,
                        n_rollouts_per_instance=n_rollouts_per_instance,
                        steps_per_episode=steps_per_episode,
                        max_features_in_rule=max_features_in_rule,
                        coverage_on_all_data=coverage_on_all_data,
                        eval_on_test_data=eval_on_test_data,
                        use_full_for_sampling=use_full_for_sampling,
                        filter_low_quality_rollouts=filter_low_quality_rollouts,
                        min_precision_threshold=min_precision_threshold,
                        min_coverage_threshold=min_coverage_threshold,
                        use_weighted_average=use_weighted_average,
                        device=device,
                        seed=seed
                    )
                    
                    # Extract results
                    anchors_list = process_results["anchors_list"]
                    rules_list = process_results["rules_list"]
                    precisions = process_results["precisions"]
                    coverages = process_results["coverages"]
                    rollout_times = process_results["rollout_times"]
                    per_instance_results = process_results["per_instance_results"]
                    instance_precision = process_results["instance_precision"]
                    instance_coverage = process_results["instance_coverage"]
                    instance_coverage_class_conditional = process_results["instance_coverage_class_conditional"]
                    
                    logger.info(f"  Instance-level metrics (average across {len(per_instance_results)} instances, "
                              f"each averaged from filtered rollouts, computed on FULL dataset):")
                    logger.info(f"    Precision: {instance_precision:.4f}")
                    logger.info(f"    Coverage:  {instance_coverage:.4f}")
                    
                    # Deduplicate using canonical keys (more robust than rule strings)
                    # Step 1: Deduplicate by canonical key
                    canonical_keys_seen = set()
                    unique_anchors_by_key = {}
                    for anchor in anchors_list:
                        canonical_key = anchor.get("canonical_key")
                        if canonical_key is None:
                            # Fallback: use rule string if canonical key not available
                            rule_str = anchor.get("rule", "")
                            if rule_str and rule_str != "any values (no tightened features)":
                                canonical_key = f"rule:{rule_str}"
                            else:
                                canonical_key = "any_values"
                        
                        # Keep the anchor with best precision (tie-break by coverage) for each canonical key
                        if canonical_key not in canonical_keys_seen:
                            canonical_keys_seen.add(canonical_key)
                            unique_anchors_by_key[canonical_key] = anchor
                        else:
                            existing = unique_anchors_by_key[canonical_key]
                            existing_prec = existing.get("precision_recomputed", existing.get("precision", 0.0))
                            new_prec = anchor.get("precision_recomputed", anchor.get("precision", 0.0))
                            if new_prec > existing_prec:
                                unique_anchors_by_key[canonical_key] = anchor
                            elif new_prec == existing_prec:
                                existing_cov = existing.get("coverage_recomputed", existing.get("coverage", 0.0))
                                new_cov = anchor.get("coverage_recomputed", anchor.get("coverage", 0.0))
                                if new_cov > existing_cov:
                                    unique_anchors_by_key[canonical_key] = anchor
                    
                    # Step 2: Apply NMS for near-duplicate suppression (IoU-based)
                    nms_iou_threshold = env_config.get("nms_iou_threshold", 0.9)
                    anchors_after_nms = nms_deduplicate_anchors(
                        list(unique_anchors_by_key.values()),
                        iou_threshold=nms_iou_threshold
                    )
                    
                    # Update anchors_list with deduplicated anchors
                    anchors_list = anchors_after_nms
                    
                    # Extract unique rules from deduplicated anchors (for backward compatibility)
                    unique_rules = list(set([
                        anchor.get("rule", "") 
                        for anchor in anchors_list 
                        if anchor.get("rule") and anchor.get("rule") != "any values (no tightened features)"
                    ]))
                    
                    # End timing for this class
                    class_end_time = time.perf_counter()
                    class_total_time = class_end_time - class_start_time
            
            # Note: instance_precision, instance_coverage, instance_coverage_class_conditional are already computed by helper function
            
            avg_rollout_time = float(np.mean(rollout_times)) if rollout_times else 0.0
            total_rollout_time = float(np.sum(rollout_times)) if rollout_times else 0.0
            
            # NOTE: Class-level (class-union) metrics will be computed later using class-based anchors only
            # We do NOT compute union of instance-based anchors here - that's not one of our three metrics:
            # 1. Instance-based = average of instance-based rollouts
            # 2. Class-based = average of class-based rollouts  
            # 3. Class-union = union of class-based rules (computed later, no new rollouts)
            class_precision = 0.0  # Will be set later from class-based union
            class_coverage = 0.0   # Will be set later from class-based union
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
            # SKIPPED: We do NOT compute union of instance-based anchors here.
            # The union of instance-based anchors is NOT one of our three metrics.
            # Class-union metrics will be computed later from class-based rules only.
            if False:  # Disabled - union of instance-based anchors is not one of our metrics
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
                "instance_coverage": instance_coverage,  # Class-conditional coverage P(x in box | y = target_class) for instance-based anchors (more meaningful than overall coverage)
                "instance_coverage_class_conditional": instance_coverage_class_conditional,  # Same as instance_coverage (kept for backward compatibility)
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
                "total_rules_count": len(rules_list) if rules_list else 0,
                "duplicate_ratio": (len(rules_list) - len(unique_rules)) / len(rules_list) if rules_list and len(rules_list) > 0 else 0.0,
                # Top-K rules by score: precision * (1 + coverage)
                "top_k_rules_by_score": [
                    {
                        "rule": anchor.get("rule", ""),
                        "precision": float(anchor.get("precision_recomputed", anchor.get("precision", 0.0))),
                        "coverage": float(anchor.get("coverage_recomputed", anchor.get("coverage", 0.0))),
                        "score": float((anchor.get("precision_recomputed", anchor.get("precision", 0.0)) * (1 + anchor.get("coverage_recomputed", anchor.get("coverage", 0.0)))))
                    }
                    for anchor in sorted(
                        anchors_list,
                        key=lambda a: (a.get("precision_recomputed", a.get("precision", 0.0)) * (1 + a.get("coverage_recomputed", a.get("coverage", 0.0)))),
                        reverse=True
                    )[:env_config.get("top_k_rules_by_score", 10)]
                ],
                "top_k_rules_count": min(env_config.get("top_k_rules_by_score", 10), len(anchors_list)),
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
            # Determine data source for logging
            # Instance-based rollouts now use full dataset for metrics (for consistency with class-based)
            coverage_data_source = "all (train+test)" if env_data.get("X_test_unit") is not None else data_source
            logger.info(f"  Instance-Level Metrics (averaged across all {len(anchors_list)} instance-based anchors):")
            logger.info(f"    Precision: {instance_precision:.4f}")
            logger.info(f"    Coverage:  {instance_coverage:.4f} (overall: P(x in box) on {coverage_data_source} data)")
            if instance_coverage_class_conditional > 0.0:
                logger.info(f"    Class-conditional coverage: {instance_coverage_class_conditional:.4f} (P(x in box | y = target_class))")
            
            # Debug: Log box sizes to understand why coverage is low
            if anchors_list and len(anchors_list) > 0:
                box_volumes = [a.get("box_volume", 0.0) for a in anchors_list if "box_volume" in a]
                if box_volumes:
                    avg_box_volume = float(np.mean(box_volumes))
                    min_box_volume = float(np.min(box_volumes))
                    max_box_volume = float(np.max(box_volumes))
                    logger.info(f"    Box volumes (normalized space): avg={avg_box_volume:.6f}, min={min_box_volume:.6f}, max={max_box_volume:.6f}")
                    
                    # Check if boxes are suspiciously small
                    if avg_box_volume < 1e-6:
                        logger.warning(f"    WARNING: Box volumes are extremely small! This suggests anchors may be collapsing.")
                        # Log first anchor's bounds for debugging
                        if "lower_bounds_normalized" in anchors_list[0] and "upper_bounds_normalized" in anchors_list[0]:
                            lower = np.array(anchors_list[0]["lower_bounds_normalized"])
                            upper = np.array(anchors_list[0]["upper_bounds_normalized"])
                            widths = upper - lower
                            logger.warning(f"    Sample anchor widths (first 5 features): {widths[:5]}")
                            logger.warning(f"    Sample anchor widths (min/max): {widths.min():.6f} / {widths.max():.6f}")
            logger.info(f"  Unique rules (instance-based, after deduplication): {len(unique_rules)}")
            logger.info(f"  Average rollout time per episode: {avg_rollout_time:.4f}s")
            # Union metrics (instance-based only) - temporary, will be overwritten with class-based union after class-based rollouts
            # NOTE: class_precision and class_coverage are set to 0.0 here
            # They will be properly computed later from class-based union metrics (lines 1331-1360)
            # The three metrics are:
            # 1. Instance-based = average (computed above)
            # 2. Class-based = average (will be computed from class-based rollouts)
            # 3. Class-union = union of class-based rules (will be computed later)
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
        n_class_based_rollouts_per_class = max(20, min(max_centroids, 50))  # Increased from 5-10 to 20-50
        logger.info(f"  Using {n_class_based_rollouts_per_class} class-based rollouts per class (based on cluster centroids)")
    else:
        n_class_based_rollouts_per_class = 20  # Increased from 5 to 20
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
        class_based_rollout_times = []
        
        class_based_start_time = time.perf_counter()
        
        # Run class-based rollouts (NOT setting x_star_unit, so environment uses cluster centroids)
        for rollout_idx in range(n_class_based_rollouts_per_class):
            rollout_seed = seed + 10000 + rollout_idx if seed is not None else None  # Use different seed range
            
            # CRITICAL FIX: Respect eval_on_test_data and coverage_on_all_data flags for class-based rollouts
            # This ensures consistent evaluation sets between instance-based and class-based metrics
            # Previously, class-based always used full dataset when test data existed, mixing evaluation sets
            if coverage_on_all_data and env_data.get("X_test_unit") is not None:
                # Use full dataset (train + test) if explicitly requested via coverage_on_all_data
                env_X_unit = np.vstack([env_data["X_unit"], env_data["X_test_unit"]])
                env_X_std = np.vstack([env_data["X_std"], env_data["X_test_std"]])
                env_y = np.concatenate([env_data["y"], env_data["y_test"]])
                use_full_dataset = True
                if rollout_idx == 0:  # Log once per class
                    logger.info(f"  Using FULL dataset (train + test) for class-based rollouts (coverage_on_all_data=True)")
                    logger.info(f"    Training samples: {len(env_data['X_unit'])}, Test samples: {len(env_data['X_test_unit'])}, Total: {len(env_y)}")
            elif eval_on_test_data and env_data.get("X_test_unit") is not None:
                # Use test data only (matches instance-based evaluation when eval_on_test_data=True)
                env_X_unit = env_data["X_test_unit"]
                env_X_std = env_data["X_test_std"]
                env_y = env_data["y_test"]
                use_full_dataset = False
                if rollout_idx == 0:  # Log once per class
                    logger.info(f"  Using TEST dataset for class-based rollouts (eval_on_test_data=True, matches instance-based evaluation)")
                    logger.info(f"    Test samples: {len(env_y)}")
            else:
                # Fallback to training data only if test data not available or eval_on_test_data=False
                env_X_unit = env_data["X_unit"]
                env_X_std = env_data["X_std"]
                env_y = env_data["y"]
                use_full_dataset = False
                if rollout_idx == 0:  # Log once per class
                    logger.info(f"  Using TRAINING dataset for class-based rollouts (test data not available or eval_on_test_data=False)")
                    logger.info(f"    Training samples: {len(env_y)}")
            
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
                classifier=classifier,
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
            
            # Log precision and coverage for each class-based rollout
            logger.info(f"  Class-based rollout {rollout_idx + 1}/{n_class_based_rollouts_per_class}: "
                      f"Precision={precision:.4f}, Coverage={coverage:.4f}")
            
            # Additional debug info for problematic rollouts
            if precision == 0.0 or coverage == 0.0:
                logger.debug(f"    Final bounds: lower={env.lower[:3] if hasattr(env, 'lower') else 'N/A'}, upper={env.upper[:3] if hasattr(env, 'upper') else 'N/A'} (first 3 dims)")
                logger.debug(f"    Box volume: {np.prod(env.upper - env.lower) if hasattr(env, 'lower') and hasattr(env, 'upper') else 'N/A'}")
            
            # Append rollout_time only (doesn't change after recompute)
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
                    classifier=classifier,
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
                
                rule, canonical_key = temp_env.extract_rule(
                    max_features_in_rule=max_features_in_rule,
                    initial_lower=initial_lower_normalized,
                    initial_upper=initial_upper_normalized,
                    denormalize=True
                )
            else:
                # No bounds - use default canonical key
                canonical_key = "any_values"
            
            # Store anchor data
            anchor_data = {
                "rollout_type": "class_based",  # Flag to distinguish from instance-based
                "rollout_idx": rollout_idx,
                # Rollout-estimated metrics (computed during rollout using perturbation samples)
                "precision_rollout_estimated": float(precision),
                "coverage_rollout_estimated": float(coverage),
                "total_reward": float(episode_data.get("total_reward", 0.0)),
                "n_steps": int(episode_data.get("n_steps", 0)),
                "rule": rule,
                "canonical_key": canonical_key,  # Canonical key for deduplication
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
        
        # Deduplicate class-based anchors using canonical keys + NMS
        # Step 1: Deduplicate by canonical key
        canonical_keys_seen = set()
        unique_class_based_anchors_by_key = {}
        for anchor in class_based_anchors_list:
            canonical_key = anchor.get("canonical_key")
            if canonical_key is None:
                # Fallback: use rule string if canonical key not available
                rule_str = anchor.get("rule", "")
                if rule_str and rule_str != "any values (no tightened features)":
                    canonical_key = f"rule:{rule_str}"
                else:
                    canonical_key = "any_values"
            
            # Keep the anchor with best precision (tie-break by coverage) for each canonical key
            if canonical_key not in canonical_keys_seen:
                canonical_keys_seen.add(canonical_key)
                unique_class_based_anchors_by_key[canonical_key] = anchor
            else:
                existing = unique_class_based_anchors_by_key[canonical_key]
                existing_prec = existing.get("precision_rollout_estimated", existing.get("precision", 0.0))
                new_prec = anchor.get("precision_rollout_estimated", anchor.get("precision", 0.0))
                if new_prec > existing_prec:
                    unique_class_based_anchors_by_key[canonical_key] = anchor
                elif new_prec == existing_prec:
                    existing_cov = existing.get("coverage_rollout_estimated", existing.get("coverage", 0.0))
                    new_cov = anchor.get("coverage_rollout_estimated", anchor.get("coverage", 0.0))
                    if new_cov > existing_cov:
                        unique_class_based_anchors_by_key[canonical_key] = anchor
        
        # Step 2: Apply NMS for near-duplicate suppression (IoU-based)
        nms_iou_threshold = env_config.get("nms_iou_threshold", 0.9)
        class_based_anchors_after_nms = nms_deduplicate_anchors(
            list(unique_class_based_anchors_by_key.values()),
            iou_threshold=nms_iou_threshold,
            key_precision="precision_rollout_estimated",
            key_coverage="coverage_rollout_estimated"
        )
        
        # Update class_based_anchors_list with deduplicated anchors
        class_based_anchors_list = class_based_anchors_after_nms
        
        # Extract unique rules from deduplicated anchors (for backward compatibility)
        class_based_unique_rules = list(set([
            anchor.get("rule", "") 
            for anchor in class_based_anchors_list 
            if anchor.get("rule") and anchor.get("rule") != "any values (no tightened features)"
        ]))
        class_based_end_time = time.perf_counter()
        class_based_total_time = class_based_end_time - class_based_start_time
        
        # CRITICAL: Filter low-precision anchors before computing average
        # Following original Anchors paper methodology:
        # 1. Recompute precision and coverage on FULL dataset for each anchor
        # 2. Filter out bad anchors (based on precision threshold)
        # 3. Average filtered anchors
        # 
        # Precision for class-based: P(y = target_class | x in box)
        # Coverage for class-based: P(x in box | y = target_class) - class-conditional coverage
        #   (fraction of target class samples that fall in the anchor box)
        # 
        # We only trust high-precision anchors, so average should reflect quality of trusted anchors only
        precision_target = env_config.get("precision_target", 0.95)
        precision_threshold = precision_target * 0.8
        
        # CRITICAL FIX: Respect eval_on_test_data and coverage_on_all_data flags for class-based recomputation
        # This ensures consistent evaluation sets between instance-based and class-based metrics
        # Previously, class-based always used full dataset, mixing evaluation sets
        if coverage_on_all_data and env_data.get("X_test_unit") is not None:
            # Use full dataset (train + test) if explicitly requested via coverage_on_all_data
            X_data_filter = np.vstack([env_data["X_unit"], env_data["X_test_unit"]])
            y_data_filter = np.concatenate([env_data["y"], env_data["y_test"]])
            recompute_data_source = "FULL dataset (train + test)"
        elif eval_on_test_data and env_data.get("X_test_unit") is not None:
            # Use test data only (matches instance-based evaluation when eval_on_test_data=True)
            X_data_filter = env_data["X_test_unit"]
            y_data_filter = env_data["y_test"]
            recompute_data_source = "TEST dataset"
        else:
            # Fallback to training data only if test data not available or eval_on_test_data=False
            X_data_filter = env_data["X_unit"]
            y_data_filter = env_data["y"]
            recompute_data_source = "TRAINING dataset"
        
        logger.info(f"  Recomputing precision and coverage on {recompute_data_source} for {len(class_based_anchors_list)} class-based anchors")
        logger.info(f"    Total samples: {len(X_data_filter)} (eval_on_test_data={eval_on_test_data}, coverage_on_all_data={coverage_on_all_data})")
        
        # Filter anchors by recomputing precision on actual dataset
        filtered_class_based_precisions = []
        filtered_class_based_coverages = []
        filtered_class_based_anchors = []
        
        if X_data_filter is not None and y_data_filter is not None:
            for anchor_data in class_based_anchors_list:
                if "lower_bounds_normalized" in anchor_data and "upper_bounds_normalized" in anchor_data:
                    lower = np.array(anchor_data["lower_bounds_normalized"], dtype=np.float32)
                    upper = np.array(anchor_data["upper_bounds_normalized"], dtype=np.float32)
                    
                    # Check which points fall in this anchor box
                    in_box = np.all((X_data_filter >= lower) & (X_data_filter <= upper), axis=1)
                    
                    if in_box.sum() > 0:
                        # Recompute precision and coverage on full dataset (following original Anchors paper methodology)
                        # Precision: P(y = target_class | x in box)
                        y_in_box = y_data_filter[in_box]
                        actual_precision = float((y_in_box == target_class).mean())
                        
                        # Coverage: P(x in box | y = target_class) - class-conditional coverage
                        # Fraction of target class samples that fall in the anchor box
                        class_mask = (y_data_filter == target_class)
                        n_class_samples = class_mask.sum()
                        n_class_in_box = (in_box & class_mask).sum()
                        actual_coverage = float(n_class_in_box / n_class_samples) if n_class_samples > 0 else 0.0
                        
                        # Store recomputed metrics (computed on full dataset)
                        anchor_data["precision_recomputed"] = actual_precision
                        anchor_data["coverage_recomputed"] = actual_coverage
                        
                        # Set primary metrics to recomputed values (these are the metrics we use for evaluation)
                        anchor_data["precision"] = actual_precision
                        anchor_data["coverage"] = actual_coverage
                        
                        # Only include high-precision anchors in average
                        if actual_precision >= precision_threshold:
                            filtered_class_based_precisions.append(actual_precision)
                            filtered_class_based_coverages.append(actual_coverage)
                            filtered_class_based_anchors.append(anchor_data)
        
        # Compute average of filtered high-precision anchors only
        # This represents the quality of anchors we actually trust and use
        # Metrics are computed on FULL dataset (following original Anchors paper methodology)
        if filtered_class_based_precisions:
            class_based_precision = float(np.mean(filtered_class_based_precisions))
            class_based_coverage = float(np.mean(filtered_class_based_coverages))
            logger.info(f"  Class {target_class}: Filtered {len(filtered_class_based_precisions)}/{len(class_based_anchors_list)} "
                       f"high-precision anchors (threshold={precision_threshold:.3f})")
            logger.info(f"  Average precision (on full dataset): {class_based_precision:.4f}")
            logger.info(f"  Average coverage (class-conditional, on full dataset): {class_based_coverage:.4f}")
        else:
            # Fallback: if no anchors pass filter, recompute metrics for all anchors and use those
            logger.warning(f"  Class {target_class}: No anchors passed precision filter (threshold={precision_threshold:.3f})")
            
            # Recompute metrics for all anchors and average them
            all_precisions_full = []
            all_coverages_full = []
            for anchor_data in class_based_anchors_list:
                if "precision_recomputed" in anchor_data:
                    all_precisions_full.append(anchor_data["precision_recomputed"])
                    all_coverages_full.append(anchor_data["coverage_recomputed"])
                else:
                    # Fallback to primary metrics (should be recomputed) or rollout-estimated if not recomputed yet
                    all_precisions_full.append(anchor_data.get("precision", anchor_data.get("precision_rollout_estimated", 0.0)))
                    all_coverages_full.append(anchor_data.get("coverage", anchor_data.get("coverage_rollout_estimated", 0.0)))
            
            if all_precisions_full:
                class_based_precision = float(np.mean(all_precisions_full))
                class_based_coverage = float(np.mean(all_coverages_full))
                logger.warning(f"    Using average of all {len(all_precisions_full)} anchors (on full dataset): "
                             f"precision={class_based_precision:.4f}, coverage={class_based_coverage:.4f}")
            else:
                # Last resort: use recomputed metrics from anchor_data
                all_precisions_recomputed = [a.get("precision_recomputed", a.get("precision", 0.0)) for a in class_based_anchors_list if "precision_recomputed" in a or "precision" in a]
                all_coverages_recomputed = [a.get("coverage_recomputed", a.get("coverage", 0.0)) for a in class_based_anchors_list if "coverage_recomputed" in a or "coverage" in a]
                if all_precisions_recomputed:
                    class_based_precision = float(np.mean(all_precisions_recomputed))
                    class_based_coverage = float(np.mean(all_coverages_recomputed))
                    logger.warning(f"    Using recomputed metrics from all anchors (fallback): "
                                 f"precision={class_based_precision:.4f}, coverage={class_based_coverage:.4f}")
                else:
                    class_based_precision = 0.0
                    class_based_coverage = 0.0
                    logger.warning(f"    No valid metrics found for any anchor")
        
        class_based_avg_rollout_time = float(np.mean(class_based_rollout_times)) if class_based_rollout_times else 0.0
        class_based_total_rollout_time = float(np.sum(class_based_rollout_times)) if class_based_rollout_times else 0.0
        
        # Store class-based results in per_class_results
        if class_key not in results["per_class_results"]:
            results["per_class_results"][class_key] = {}
        
        # CRITICAL: Store class-based metrics (average of individual anchors) separately from union metrics
        # These are DIFFERENT from class union metrics:
        # - class_based_precision/coverage = AVERAGE of individual class-based anchors (each anchor is one rollout)
        # - class_precision/coverage = UNION metrics (computed from union of multiple anchors joined together)
        # 
        # Store average of FILTERED high-precision anchors only
        results["per_class_results"][class_key]["class_level_precision"] = class_based_precision  # Average of individual anchors
        results["per_class_results"][class_key]["class_level_coverage"] = class_based_coverage   # Average of individual anchors
        # Legacy names for backward compatibility
        results["per_class_results"][class_key]["class_based_precision"] = class_based_precision  # Average of individual anchors
        results["per_class_results"][class_key]["class_based_coverage"] = class_based_coverage    # Average of individual anchors
        # NOTE: class_precision and class_coverage (union metrics) will be set later in the union computation section
        # Store count of filtered vs total anchors for transparency
        results["per_class_results"][class_key]["class_based_n_filtered"] = len(filtered_class_based_precisions)
        results["per_class_results"][class_key]["class_based_n_total"] = len(class_based_anchors_list)
        
        # Also store class-based results in a separate key for easy access (kept for backward compatibility)
        class_based_key = f"class_{target_class}_class_based"
        results["per_class_results"][class_based_key] = {
            "class": int(target_class),
            "rollout_type": "class_based",
            "n_episodes": len(class_based_anchors_list),
            "precision": class_based_precision,
            "coverage": class_based_coverage,
            "precision_std": float(np.std(filtered_class_based_precisions)) if len(filtered_class_based_precisions) > 1 else 0.0,
            "coverage_std": float(np.std(filtered_class_based_coverages)) if len(filtered_class_based_coverages) > 1 else 0.0,
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
        logger.info(f"  Class-Based Metrics (AVERAGE of individual anchors, computed on full dataset):")
        logger.info(f"    Precision: {class_based_precision:.4f} [Average of {len(filtered_class_based_precisions) if filtered_class_based_precisions else len(class_based_anchors_list)} individual anchors]")
        logger.info(f"    Coverage:  {class_based_coverage:.4f} [Average class-conditional coverage]")
        logger.info(f"    NOTE: These are averages of individual anchors, NOT union metrics (union metrics computed separately)")
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
        
        # Filter anchors by precision threshold before computing union
        # This ensures we only include high-quality rules in the union
        # Use the same precision threshold as used during training/inference
        # precision_threshold = precision_target * 0.8 (same as in environment.py)
        precision_target = env_config.get("precision_target", 0.95)
        precision_threshold = precision_target * 0.8
        filtered_anchors_for_union = []
        filtered_rules_for_union = []
        
        # CRITICAL: Filter anchors using precision computed on full dataset
        # Use already-computed precision_recomputed if available (from class-based rollouts),
        # otherwise recompute precision on full dataset for filtering
        # Following original Anchors paper methodology: all metrics computed on full dataset
        if X_data_union is not None and y_data_union is not None:
            for anchor_data in all_anchors_for_union:
                if "lower_bounds_normalized" in anchor_data and "upper_bounds_normalized" in anchor_data:
                    # Prefer using already-computed precision_recomputed if available
                    if "precision_recomputed" in anchor_data:
                        actual_precision = anchor_data["precision_recomputed"]
                    else:
                        # Recompute precision on full dataset if not already computed
                        lower = np.array(anchor_data["lower_bounds_normalized"], dtype=np.float32)
                        upper = np.array(anchor_data["upper_bounds_normalized"], dtype=np.float32)
                        
                        # Check which points fall in this anchor box
                        in_box = np.all((X_data_union >= lower) & (X_data_union <= upper), axis=1)
                        
                        if in_box.sum() > 0:
                            # Compute actual precision on the dataset: P(y = target_class | x in box)
                            y_in_box = y_data_union[in_box]
                            actual_precision = float((y_in_box == target_class).mean())
                        else:
                            # No samples in box - skip this anchor
                            continue
                    
                    # Use precision computed on full dataset for filtering
                    if actual_precision >= precision_threshold:
                        filtered_anchors_for_union.append(anchor_data)
                        # Also track the corresponding rule if available
                        rule = anchor_data.get("rule", "")
                        if rule and rule != "any values (no tightened features)":
                            filtered_rules_for_union.append(rule)
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
            
            # Class-level union metrics (computed on full dataset, following original Anchors paper methodology)
            # Precision: P(y = target_class | x in union) - fraction of points in union that belong to target class
            # Coverage: P(x in union | y = target_class) - class-conditional coverage, fraction of class samples in union
            mask_cls = (y_data_union == target_class)
            n_class_samples = mask_cls.sum()
            
            if union_mask.any():
                # Precision: P(y = target_class | x in union)
                y_union = y_data_union[union_mask]
                class_precision_combined = float((y_union == target_class).mean())
                
                # Coverage: P(x in union | y = target_class) - class-conditional coverage
                if n_class_samples > 0:
                    n_class_in_union = (union_mask & mask_cls).sum()
                    class_coverage_combined = float(n_class_in_union / n_class_samples)
                else:
                    class_coverage_combined = 0.0
            else:
                # Union covers no samples
                class_precision_combined = 0.0
                class_coverage_combined = 0.0
            
            # Update class union metrics (computed from union of multiple anchors)
            # NOTE: These are DIFFERENT from class_based_precision/coverage:
            # - class_based_precision/coverage = AVERAGE of individual class-based anchors
            # - class_precision/coverage = UNION metrics (union of multiple anchors joined together with OR operation)
            class_data["class_precision"] = class_precision_combined  # Union metrics
            class_data["class_coverage"] = class_coverage_combined    # Union metrics (class-conditional)
            
            # CRITICAL: Store union rules in class_data so they can be accessed later
            # These are the deduplicated class-based rules that form the union
            class_data["class_level_unique_rules"] = class_based_unique_rules
            class_data["class_union_unique_rules"] = class_based_unique_rules  # Alias for clarity
            
            # Log the final union metrics
            n_class_based_anchors = len(filtered_anchors_for_union)
            n_unique_rules = len(class_based_unique_rules)
            logger.info(f"\n  {'='*60}")
            logger.info(f"  Class {target_class} - CLASS UNION Results (Union of Class-Based Anchors Only):")
            logger.info(f"  {'='*60}")
            logger.info(f"  Class Union Metrics (UNION of {n_class_based_anchors} class-based anchors joined together, computed on FULL dataset):")
            logger.info(f"    Precision: {class_precision_combined:.4f} [P(y = target_class | x in union)]")
            logger.info(f"    Coverage:  {class_coverage_combined:.4f} [P(x in union | y = target_class) - class-conditional]")
            logger.info(f"    Unique Rules: {n_unique_rules}")
            logger.info(f"    NOTE: These are UNION metrics (multiple anchors joined with OR), DIFFERENT from class_based_precision/coverage")
            logger.info(f"          which are averages of individual anchors. Union = x in (anchor1 OR anchor2 OR ...)")
            logger.info(f"          Represents the smallest set of general rules that explain the class structure")
            
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
            "uci_mushroom", "uci_tic-tac-toe", "uci_vote", "uci_zoo", "uci_default-credit-card-clients"
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
        "--n_rollouts_per_instance",
        type=int,
        default=10,
        help="Number of rollouts to run per instance (default: 20). "
             "Each instance will be rolled out this many times, and metrics will be averaged across rollouts."
    )
    
    parser.add_argument(
        "--eval_on_train_data",
        action="store_true",
        help="Evaluate on training data instead of test data (not recommended)"
    )
    
    parser.add_argument(
        "--coverage_on_all_data",
        action="store_true",
        help="If True, compute coverage on all data (train+test combined, matches baseline anchor-exp behavior). "
             "Baseline uses all training data for coverage (no train/test split), so this ensures fair comparison."
    )
    
    parser.add_argument(
        "--sample_from_full_dataset",
        action="store_true",
        help="If True, sample instances from full dataset (train+test combined) instead of just test/train data. "
             "This ensures more instances are available for instance-based rollouts."
    )
    
    parser.add_argument(
        "--filter_by_prediction",
        action="store_true",
        default=True,
        help="If True, filter instances where classifier prediction matches target_class (default: True). "
             "Set to False for fair comparison with baseline (baseline doesn't filter by prediction)."
    )
    
    parser.add_argument(
        "--no_filter_by_prediction",
        action="store_false",
        dest="filter_by_prediction",
        help="Disable prediction filtering (equivalent to --filter_by_prediction=False). "
             "Use this for fair comparison with baseline methods."
    )
    
    parser.add_argument(
        "--use_weighted_average",
        action="store_true",
        help="If True, use coverage-weighted average for instance-level metrics instead of simple arithmetic mean. "
             "Anchors with higher coverage will have more weight in the average. "
             "Default: False (simple average for comparison with baseline)."
    )
    
    parser.add_argument(
        "--filter_low_quality_rollouts",
        action="store_true",
        default=True,
        help="If True, filter out low-precision/low-coverage rollouts before averaging (default: True). "
             "Only high-quality rollouts are used for metrics and rule extraction."
    )
    
    parser.add_argument(
        "--no_filter_low_quality_rollouts",
        action="store_false",
        dest="filter_low_quality_rollouts",
        help="Disable filtering of low-quality rollouts (use all rollouts regardless of quality)."
    )
    
    parser.add_argument(
        "--min_precision_threshold",
        type=float,
        default=None,
        help="Minimum precision threshold for keeping rollouts (default: precision_target * 0.8). "
             "Rollouts with precision below this threshold will be discarded."
    )
    
    parser.add_argument(
        "--min_coverage_threshold",
        type=float,
        default=0.01,
        help="Minimum coverage threshold for keeping rollouts (default: 0.01). "
             "Rollouts with coverage below this threshold will be discarded."
    )
    
    parser.add_argument(
        "--use_prediction_routing",
        action="store_true",
        default=True,
        help="If True (default), route instances to policies based on classifier predictions (realistic evaluation). "
             "If False, use ground truth labels (traditional evaluation). "
             "When enabled, instances are sampled from all classes and routed to policies based on their predicted class."
    )
    
    parser.add_argument(
        "--no_prediction_routing",
        action="store_false",
        dest="use_prediction_routing",
        help="Disable prediction routing (equivalent to --use_prediction_routing=False). "
             "Use traditional evaluation mode with ground truth labels for routing."
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
        n_rollouts_per_instance=args.n_rollouts_per_instance,
        eval_on_test_data=not args.eval_on_train_data,
        coverage_on_all_data=args.coverage_on_all_data,
        sample_from_full_dataset=args.sample_from_full_dataset,
        filter_by_prediction=args.filter_by_prediction,
        use_prediction_routing=args.use_prediction_routing,
        use_weighted_average=args.use_weighted_average,
        filter_low_quality_rollouts=args.filter_low_quality_rollouts,
        min_precision_threshold=args.min_precision_threshold,
        min_coverage_threshold=args.min_coverage_threshold,
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

