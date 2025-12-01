#!/usr/bin/env python3
"""
Test Single-Agent Extracted Rules Script

This script reads extracted rules from a single-agent SB3 inference JSON file and tests them
against a dataset to find all samples that satisfy each rule.

Usage:
    python single_agent/test_extracted_rules_single.py --rules_file <path_to_extracted_rules_single_agent.json> --dataset <dataset_name> [--use_train_data]

Example:
    python single_agent/test_extracted_rules_single.py \
        --rules_file output/single_agent_sb3_breast_cancer_ddpg/training/.../inference/extracted_rules_single_agent.json \
        --dataset breast_cancer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import re
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict
import logging
from datetime import datetime

# Import directly to avoid importing environment module which requires pettingzoo
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'BenchMARL'))
from tabular_datasets import TabularDatasetLoader

# Set up basic logging (will be reconfigured in main)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_file_logging(log_file_path: str):
    """
    Setup logging to write to both console and a log file.
    
    Args:
        log_file_path: Path to the log file
    """
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (create directory if needed)
    log_dir = os.path.dirname(log_file_path)
    if log_dir:  # Only create if there's a directory component
        os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return log_file_path


def get_experiment_dir_from_rules_file(rules_file: str) -> str:
    """
    Extract experiment directory from rules file path.
    
    Rules file is typically at: experiment_dir/inference/extracted_rules.json
    Returns: experiment_dir
    """
    rules_file = os.path.abspath(rules_file)
    
    # If rules_file is in an 'inference' subdirectory, go up one level
    parts = rules_file.split(os.sep)
    if 'inference' in parts:
        inference_idx = parts.index('inference')
        experiment_dir = os.sep.join(parts[:inference_idx])
        return experiment_dir
    
    # Otherwise, assume rules_file is directly in experiment_dir
    return os.path.dirname(rules_file)


def parse_rule(rule_str: str) -> List[Tuple[str, float, float]]:
    """
    Parse a rule string into a list of (feature_name, lower_bound, upper_bound) tuples.
    
    Example:
        "mean concavity ∈ [0.0367, 0.9744] and worst concave points ∈ [0.2905, 1.3520]"
        -> [("mean concavity", 0.0367, 0.9744), ("worst concave points", 0.2905, 1.3520)]
    
    Args:
        rule_str: Rule string in format "feature ∈ [lower, upper] and ..."
    
    Returns:
        List of (feature_name, lower_bound, upper_bound) tuples
    """
    if rule_str == "any values (no tightened features)":
        return []
    
    conditions = []
    
    # Split by " and " to get individual conditions
    condition_strings = rule_str.split(" and ")
    
    # Pattern to match: "feature_name ∈ [lower, upper]"
    pattern = r'(.+?)\s*∈\s*\[([-\d.]+),\s*([-\d.]+)\]'
    
    for condition_str in condition_strings:
        condition_str = condition_str.strip()
        match = re.search(pattern, condition_str)
        if match:
            feature_name = match.group(1).strip()
            lower = float(match.group(2))
            upper = float(match.group(3))
            conditions.append((feature_name, lower, upper))
        else:
            logger.warning(f"Could not parse condition: {condition_str}")
    
    return conditions


def check_rule_satisfaction(
    X: np.ndarray,
    feature_names: List[str],
    rule_conditions: List[Tuple[str, float, float]]
) -> np.ndarray:
    """
    Check which samples in X satisfy the rule conditions.
    
    Args:
        X: Data matrix (n_samples, n_features) in original feature space
        feature_names: List of feature names corresponding to columns in X
        rule_conditions: List of (feature_name, lower_bound, upper_bound) tuples
    
    Returns:
        Boolean array of shape (n_samples,) indicating which samples satisfy the rule
    """
    if len(rule_conditions) == 0:
        # Empty rule means all samples satisfy it
        return np.ones(X.shape[0], dtype=bool)
    
    # Create feature name to index mapping
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    
    # Start with all samples satisfying
    mask = np.ones(X.shape[0], dtype=bool)
    
    # Apply each condition (all must be satisfied)
    for feature_name, lower, upper in rule_conditions:
        if feature_name not in feature_to_idx:
            logger.warning(f"Feature '{feature_name}' not found in dataset. Available features: {feature_names[:5]}...")
            # If feature not found, no samples satisfy this condition
            mask = np.zeros(X.shape[0], dtype=bool)
            break
        
        feature_idx = feature_to_idx[feature_name]
        feature_values = X[:, feature_idx]
        
        # Check if values are within bounds [lower, upper] (inclusive)
        condition_mask = (feature_values >= lower) & (feature_values <= upper)
        mask = mask & condition_mask
    
    return mask


def analyze_rule_overlaps_detailed(
    results: Dict,
    X_data: np.ndarray,
    y_data: np.ndarray,
    feature_names: List[str],
    unique_classes: List[int]
) -> Dict:
    """
    Analyze how rules from different classes overlap in detail.
    
    Args:
        results: Results dictionary from test_rules_from_json
        X_data: Data matrix (n_samples, n_features) in standardized space
        y_data: Class labels (n_samples,)
        feature_names: List of feature names
        unique_classes: List of unique class labels
    
    Returns:
        Dictionary containing detailed overlap analysis
    """
    logger.info("\n" + "="*80)
    logger.info("DETAILED RULE OVERLAP ANALYSIS")
    logger.info("="*80)
    
    per_class_results = results.get("per_class_results", {})
    rule_results = results.get("rule_results", [])
    
    # Build mapping of rules to their source classes from per_class_results
    rule_to_source_classes = {}
    for class_key, class_data in per_class_results.items():
        target_class = class_data.get("class")
        unique_rules = class_data.get("unique_rules", [])
        for rule_str in unique_rules:
            if rule_str not in rule_to_source_classes:
                rule_to_source_classes[rule_str] = []
            rule_to_source_classes[rule_str].append(target_class)
    
    # Analyze overlaps
    overlap_analysis = {
        "rule_overlaps": [],
        "class_pair_overlaps": {},
        "summary": {
            "total_unique_rules": len(rule_to_source_classes),
            "rules_with_overlaps": 0,
            "total_overlap_pairs": 0
        }
    }
    
    # Use rule_results which already has overlap information
    for rule_result in rule_results:
        rule_str = rule_result.get("rule")
        if rule_str == "any values (no tightened features)":
            continue
        
        if rule_result.get("satisfies_multiple_classes", False):
            classes_satisfied = rule_result.get("classes_satisfied", [])
            source_classes = rule_to_source_classes.get(rule_str, [])
            
            overlap_analysis["summary"]["rules_with_overlaps"] += 1
            
            # Get class sample counts
            class_sample_counts = {}
            total_samples_satisfying = 0
            for target_class in classes_satisfied:
                class_key = f"class_{target_class}"
                if class_key in rule_result.get("per_class_results", {}):
                    class_res = rule_result["per_class_results"][class_key]
                    n_satisfying = class_res.get("n_satisfying_class_samples", 0)
                    class_sample_counts[target_class] = n_satisfying
                    total_samples_satisfying += n_satisfying
            
            # Record overlap details
            overlap_info = {
                "rule": rule_str,
                "rule_index": rule_result.get("rule_index"),
                "source_classes": source_classes,
                "satisfies_classes": classes_satisfied,
                "class_sample_counts": class_sample_counts,
                "total_samples_satisfying": total_samples_satisfying
            }
            overlap_analysis["rule_overlaps"].append(overlap_info)
            
            # Record class pair overlaps
            for i, class1 in enumerate(classes_satisfied):
                for class2 in classes_satisfied[i+1:]:
                    pair_key = f"{min(class1, class2)}_{max(class1, class2)}"
                    if pair_key not in overlap_analysis["class_pair_overlaps"]:
                        overlap_analysis["class_pair_overlaps"][pair_key] = {
                            "class1": int(min(class1, class2)),
                            "class2": int(max(class1, class2)),
                            "overlapping_rules": [],
                            "n_overlapping_rules": 0
                        }
                    overlap_analysis["class_pair_overlaps"][pair_key]["overlapping_rules"].append(rule_str)
                    overlap_analysis["class_pair_overlaps"][pair_key]["n_overlapping_rules"] += 1
                    overlap_analysis["summary"]["total_overlap_pairs"] += 1
    
    # Identify unique rules per class (rules that don't overlap)
    per_class_results = results.get("per_class_results", {})
    unique_rules_per_class = {}
    for class_key, class_data in per_class_results.items():
        target_class = class_data.get("class")
        unique_rules = class_data.get("unique_rules", [])
        # Filter out overlapping rules
        non_overlapping = [
            rule for rule in unique_rules 
            if rule not in [overlap["rule"] for overlap in overlap_analysis["rule_overlaps"]]
        ]
        unique_rules_per_class[target_class] = non_overlapping
    
    # Log summary
    logger.info(f"Total unique rules: {overlap_analysis['summary']['total_unique_rules']}")
    logger.info(f"Rules with overlaps: {overlap_analysis['summary']['rules_with_overlaps']}")
    logger.info(f"Total overlap pairs: {overlap_analysis['summary']['total_overlap_pairs']}")
    
    # Log overlapping rules in detail
    if overlap_analysis["rule_overlaps"]:
        logger.info(f"{'='*80}")
        logger.info("OVERLAPPING RULES (satisfy multiple classes):")
        logger.info(f"{'='*80}")
        for idx, overlap_info in enumerate(overlap_analysis["rule_overlaps"], 1):
            logger.info(f"Overlapping Rule {idx}:")
            logger.info(f"  Rule: {overlap_info['rule'][:100]}{'...' if len(overlap_info['rule']) > 100 else ''}")
            logger.info(f"  Source classes (where rule was extracted): {overlap_info['source_classes']}")
            logger.info(f"  Satisfies classes: {overlap_info['satisfies_classes']}")
            logger.info(f"  Total samples satisfying: {overlap_info['total_samples_satisfying']}")
            logger.info(f"  Class sample counts:")
            for cls, count in overlap_info['class_sample_counts'].items():
                logger.info(f"Class {cls}: {count} samples")
    else:
        logger.info(f"No overlapping rules found.")
    
    # Log class pair overlaps
    if overlap_analysis["class_pair_overlaps"]:
        logger.info(f"{'='*80}")
        logger.info("CLASS PAIR OVERLAPS:")
        logger.info(f"{'='*80}")
        for pair_key, pair_info in overlap_analysis["class_pair_overlaps"].items():
            logger.info(f"Classes {pair_info['class1']} & {pair_info['class2']}:")
            logger.info(f"  Number of overlapping rules: {pair_info['n_overlapping_rules']}")
            logger.info(f"  Overlapping rules:")
            for rule_idx, rule_str in enumerate(pair_info['overlapping_rules'], 1):
                logger.info(f"    {rule_idx}. {rule_str[:80]}{'...' if len(rule_str) > 80 else ''}")
    
    # Log unique rules per class
    logger.info(f"{'='*80}")
    logger.info("UNIQUE RULES (class-specific, no overlaps):")
    logger.info(f"{'='*80}")
    for target_class in sorted(unique_rules_per_class.keys()):
        unique_rules = unique_rules_per_class[target_class]
        logger.info(f"Class {target_class}:")
        if unique_rules:
            logger.info(f"  {len(unique_rules)} unique rule(s):")
            for rule_idx, rule_str in enumerate(unique_rules, 1):
                logger.info(f"    {rule_idx}. {rule_str[:80]}{'...' if len(rule_str) > 80 else ''}")
        else:
            logger.info(f"  No unique rules (all rules overlap with other classes)")
    
    # Add unique rules to analysis results
    overlap_analysis["unique_rules_per_class"] = {
        f"class_{cls}": {
            "class": cls,
            "unique_rules": rules,
            "n_unique_rules": len(rules)
        }
        for cls, rules in unique_rules_per_class.items()
    }
    
    return overlap_analysis


def analyze_missed_samples(
    results: Dict,
    X_data: np.ndarray,
    y_data: np.ndarray,
    feature_names: List[str],
    unique_classes: List[int]
) -> Dict:
    """
    Analyze which samples are missed (not covered by any rule) in each class.
    
    Args:
        results: Results dictionary from test_rules_from_json
        X_data: Data matrix (n_samples, n_features) in standardized space
        y_data: Class labels (n_samples,)
        feature_names: List of feature names
        unique_classes: List of unique class labels
    
    Returns:
        Dictionary containing missed samples analysis
    """
    logger.info("\n" + "="*80)
    logger.info("MISSED SAMPLES ANALYSIS")
    logger.info("="*80)
    
    per_class_results = results.get("per_class_results", {})
    
    missed_samples_analysis = {
        "per_class_analysis": {},
        "summary": {}
    }
    
    # For each class, collect all rules and check coverage
    for class_key, class_data in per_class_results.items():
        target_class = class_data.get("class")
        unique_rules = class_data.get("unique_rules", [])
        
        # Get all samples for this class
        class_mask = (y_data == target_class)
        class_indices = np.where(class_mask)[0]
        n_class_samples = len(class_indices)
        
        # Check which samples are covered by at least one rule
        covered_mask = np.zeros(n_class_samples, dtype=bool)
        
        for rule_str in unique_rules:
            if rule_str == "any values (no tightened features)":
                # Empty rule covers all samples
                covered_mask = np.ones(n_class_samples, dtype=bool)
                break
            
            rule_conditions = parse_rule(rule_str)
            if len(rule_conditions) == 0:
                covered_mask = np.ones(n_class_samples, dtype=bool)
                break
            
            # Check which samples satisfy this rule
            satisfying_mask = check_rule_satisfaction(X_data, feature_names, rule_conditions)
            # Only consider samples from this class
            class_satisfying = satisfying_mask[class_mask]
            covered_mask = covered_mask | class_satisfying
        
        # Find missed samples
        missed_indices = class_indices[~covered_mask]
        n_missed = len(missed_indices)
        n_covered = np.sum(covered_mask)
        coverage_ratio = n_covered / n_class_samples if n_class_samples > 0 else 0.0
        
        class_analysis = {
            "class": int(target_class),
            "n_class_samples": int(n_class_samples),
            "n_covered_samples": int(n_covered),
            "n_missed_samples": int(n_missed),
            "coverage_ratio": float(coverage_ratio),
            "missed_sample_indices": missed_indices.tolist()
        }
        
        missed_samples_analysis["per_class_analysis"][f"class_{target_class}"] = class_analysis
        
        logger.info(f"Class {target_class}:")
        logger.info(f"  Total samples: {n_class_samples}")
        logger.info(f"  Covered samples: {n_covered} ({100*coverage_ratio:.2f}%)")
        logger.info(f"  Missed samples: {n_missed} ({100*(1-coverage_ratio):.2f}%)")
    
    # Summary
    total_samples = len(y_data)
    total_missed = sum(
        class_analysis["n_missed_samples"]
        for class_analysis in missed_samples_analysis["per_class_analysis"].values()
    )
    total_covered = total_samples - total_missed
    
    missed_samples_analysis["summary"] = {
        "total_samples": int(total_samples),
        "total_covered_samples": int(total_covered),
        "total_missed_samples": int(total_missed),
        "overall_coverage_ratio": float(total_covered / total_samples) if total_samples > 0 else 0.0
    }
    
    logger.info(f"Overall Summary:")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Covered samples: {total_covered} ({100*missed_samples_analysis['summary']['overall_coverage_ratio']:.2f}%)")
    logger.info(f"  Missed samples: {total_missed} ({100*(1-missed_samples_analysis['summary']['overall_coverage_ratio']):.2f}%)")
    
    return missed_samples_analysis


# Helper to select global rules per class for post-hoc explanations
def select_global_rules_per_class(
    results: Dict,
    X_data: np.ndarray,
    y_data: np.ndarray,
    feature_names: List[str],
    unique_classes: List[int],
    precision_threshold: float = 0.9,
    max_rules_per_class: int = 5,
    overlap_penalty_weight: float = 0.0,
) -> Dict:
    """Select a small, high-precision set of rules per class to form global explanations.

    This operates post-hoc on the tested rules:
      * Uses rule-level precision/coverage computed in this script.
      * For each class, greedily picks rules that cover new class samples,
        optionally penalizing overlap between rules.
      * Computes class-union coverage and class-union precision for the
        selected subset, similar to the environment's class-level metrics.
    
    Uses the same source as missed_samples_analysis: unique_rules from per_class_results
    (original rules file), not rule_results (test results), to ensure consistency.
    """
    per_class_results = results.get("per_class_results", {})
    rule_results = results.get("rule_results", [])
    n_samples = X_data.shape[0]

    # Create a mapping from rule strings to rule indices in rule_results for lookup
    rule_str_to_idx = {}
    rule_str_to_global_indices = {}
    for rule_idx, rr in enumerate(rule_results):
        rule_str = rr.get("rule", "")
        rule_str_to_idx[rule_str] = rule_idx
        # Precompute global satisfying index sets for each rule to support union precision
        conditions = rr.get("conditions", [])
        rule_conditions = [
            (cond["feature"], float(cond["lower"]), float(cond["upper"]))
            for cond in conditions
        ]
        if len(rule_conditions) == 0:
            satisfying_mask = np.ones(n_samples, dtype=bool)
        else:
            satisfying_mask = check_rule_satisfaction(X_data, feature_names, rule_conditions)
        indices = set(np.where(satisfying_mask)[0].tolist())
        rule_str_to_global_indices[rule_str] = indices

    global_explanations: Dict[str, Any] = {
        "settings": {
            "precision_threshold": float(precision_threshold),
            "max_rules_per_class": int(max_rules_per_class) if max_rules_per_class != -1 else -1,
            "overlap_penalty_weight": float(overlap_penalty_weight),
        },
        "per_class": {},
    }

    for cls in unique_classes:
        class_key = f"class_{cls}"
        # Indices of all samples of this class
        class_mask = (y_data == cls)
        class_indices = set(np.where(class_mask)[0].tolist())
        n_class_samples = len(class_indices)

        if n_class_samples == 0:
            global_explanations["per_class"][class_key] = {
                "class": int(cls),
                "n_class_samples": 0,
                "n_selected_rules": 0,
                "selected_rule_indices": [],
                "selected_rules": [],
                "class_union_coverage": 0.0,
                "class_union_precision": 0.0,
                "n_covered_class_samples": 0,
                "n_union_samples_total": 0,
            }
            continue

        # Build candidate list for this class using unique_rules from per_class_results
        # (same source as missed_samples_analysis to ensure consistency)
        # Compute metrics directly from rule strings, matching missed_samples_analysis approach
        candidates: List[Dict[str, Any]] = []
        if class_key not in per_class_results:
            # Fallback: if class_key not in per_class_results, try to get from rule_results
            # This shouldn't happen if per_class_results is properly populated
            logger.warning(f"  Class key {class_key} not found in per_class_results. Available keys: {list(per_class_results.keys())}")
        else:
            unique_rules = per_class_results[class_key].get("unique_rules", [])
            if not unique_rules:
                logger.debug(f"  No unique_rules found for {class_key} in per_class_results")
            
            for rule_str in unique_rules:
                if rule_str == "any values (no tightened features)":
                    continue
                
                # Parse rule and compute metrics directly (same as missed_samples_analysis)
                rule_conditions = parse_rule(rule_str)
                if len(rule_conditions) == 0:
                    # Empty rule covers all samples
                    satisfying_mask = np.ones(n_samples, dtype=bool)
                else:
                    satisfying_mask = check_rule_satisfaction(X_data, feature_names, rule_conditions)
                
                # Get class-specific samples (same approach as missed_samples_analysis)
                class_satisfying = satisfying_mask[class_mask]  # Index to get only class samples
                n_satisfying_class = int(np.sum(class_satisfying))
                if n_satisfying_class <= 0:
                    continue
                
                # Compute precision: fraction of satisfying samples that belong to this class
                n_satisfying_total = int(np.sum(satisfying_mask))
                if n_satisfying_total > 0:
                    rule_prec = float(n_satisfying_class / n_satisfying_total)
                else:
                    rule_prec = 0.0
                
                if rule_prec < precision_threshold:
                    continue
                
                # Get indices of satisfying class samples (in full dataset indices)
                class_satisfying_mask = satisfying_mask & class_mask
                satisfying_class_indices = set(np.where(class_satisfying_mask)[0].tolist())
                
                # Find rule index in rule_results for union precision calculation
                rule_idx = rule_str_to_idx.get(rule_str, -1)
                
                candidates.append({
                    "rule_idx": rule_idx,
                    "rule_str": rule_str,
                    "precision": rule_prec,
                    "class_indices": satisfying_class_indices,
                })

        selected: List[Dict[str, Any]] = []
        covered_class_indices: Set[int] = set()

        # If max_rules_per_class is -1, use ALL candidates (all rules that satisfy the class)
        # Otherwise, use greedy selection to maximize new class coverage
        use_all_rules = (max_rules_per_class == -1)
        
        if use_all_rules:
            # Use all candidates to match missed samples analysis
            selected = candidates.copy()
            for cand in selected:
                covered_class_indices |= cand["class_indices"]
        else:
            # Greedy selection: maximize new class coverage, optionally penalizing overlap
            while len(selected) < max_rules_per_class and candidates:
                best_candidate = None
                best_score = 0.0

                for cand in candidates:
                    new_cover = cand["class_indices"] - covered_class_indices
                    gain = len(new_cover)
                    if gain <= 0:
                        continue
                    if overlap_penalty_weight > 0.0:
                        overlap = len(cand["class_indices"] & covered_class_indices)
                        score = gain - overlap_penalty_weight * overlap
                    else:
                        score = float(gain)
                    if score > best_score:
                        best_score = score
                        best_candidate = cand

                if best_candidate is None or best_score <= 0.0:
                    break

                selected.append(best_candidate)
                covered_class_indices |= best_candidate["class_indices"]
                # Remove by rule_str to avoid issues if rule_idx is -1
                candidates = [c for c in candidates if c["rule_str"] != best_candidate["rule_str"]]

        # Class-union coverage over this class (on true labels)
        n_covered_class = len(covered_class_indices)
        class_union_coverage = n_covered_class / n_class_samples if n_class_samples > 0 else 0.0

        # Union precision: among all samples covered by selected rules, fraction belonging to this class
        union_indices_global: Set[int] = set()
        for s in selected:
            rule_str = s["rule_str"]
            if rule_str in rule_str_to_global_indices:
                union_indices_global |= rule_str_to_global_indices[rule_str]
            else:
                # Compute global indices directly if not found in lookup
                rule_conditions = parse_rule(rule_str)
                if len(rule_conditions) == 0:
                    satisfying_mask = np.ones(n_samples, dtype=bool)
                else:
                    satisfying_mask = check_rule_satisfaction(X_data, feature_names, rule_conditions)
                indices = set(np.where(satisfying_mask)[0].tolist())
                union_indices_global |= indices
        n_union_total = len(union_indices_global)
        if n_union_total > 0:
            n_union_class = sum(1 for idx in union_indices_global if y_data[idx] == cls)
            class_union_precision = n_union_class / n_union_total
        else:
            class_union_precision = 0.0

        global_explanations["per_class"][class_key] = {
            "class": int(cls),
            "n_class_samples": int(n_class_samples),
            "n_selected_rules": len(selected),
            "selected_rule_indices": [s["rule_idx"] for s in selected],
            "selected_rules": [s["rule_str"] for s in selected],
            "class_union_coverage": float(class_union_coverage),
            "class_union_precision": float(class_union_precision),
            "n_covered_class_samples": int(n_covered_class),
            "n_union_samples_total": int(n_union_total),
        }

    return global_explanations


def test_rules_from_json(
    rules_file: str,
    dataset_name: str,
    use_test_data: bool = True,
    seed: int = 42,
    precision_threshold: float = 0.9,
    max_rules_per_class: int = -1,
    overlap_penalty_weight: float = 0.0,
) -> Dict:
    """
    Test extracted rules against a dataset.
    
    Args:
        rules_file: Path to extracted_rules_single_agent.json file
        dataset_name: Name of the dataset to test against
        use_test_data: If True, test on test data; if False, test on training data
        seed: Random seed for dataset loading
        precision_threshold: Minimum rule-level precision for a rule to be considered in global explanations
        max_rules_per_class: Maximum number of rules to select per class for global explanations (-1 = no limit)
        overlap_penalty_weight: Penalty weight for selecting highly overlapping rules in global explanations
    
    Returns:
        Dictionary containing test results for each rule
    """
    logger.info("="*80)
    logger.info("TESTING SINGLE-AGENT EXTRACTED RULES")
    logger.info("="*80)
    logger.info(f"Rules file: {rules_file}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Data split: {'test' if use_test_data else 'training'}")
    logger.info("="*80)
    
    # Load rules from JSON
    logger.info("Loading rules from JSON file...")
    with open(rules_file, 'r') as f:
        rules_data = json.load(f)
    
    logger.info(f"✓ Loaded rules file with {len(rules_data.get('per_class_results', {}))} classes")
    
    # Check if the rules file contains instance-level and class-level metrics
    has_instance_metrics = False
    has_class_metrics = False
    per_class_results = rules_data.get("per_class_results", {})
    for class_data in per_class_results.values():
        if "instance_precision" in class_data:
            has_instance_metrics = True
        if "class_precision" in class_data:
            has_class_metrics = True
        if has_instance_metrics and has_class_metrics:
            break
    
    if has_instance_metrics or has_class_metrics:
        logger.info("✓ Found instance-level and/or class-level precision metrics in rules file")
    
    # Check if this is a single-agent results file
    metadata = rules_data.get("metadata", {})
    model_type = metadata.get("model_type", "unknown")
    if model_type == "single_agent_sb3":
        logger.info(f"✓ Detected single-agent SB3 results (algorithm: {metadata.get('algorithm', 'unknown')})")
    else:
        logger.warning(f"⚠ Model type: {model_type} (expected 'single_agent_sb3')")
    
    # Determine the data source for metrics labels
    # Check if inference was run on test data
    inference_eval_on_test_data = metadata.get("eval_on_test_data", False)
    metrics_data_source = "test data" if inference_eval_on_test_data else "training data"
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    dataset_loader = TabularDatasetLoader(
        dataset_name=dataset_name,
        test_size=0.2,
        random_state=seed
    )
    dataset_loader.load_dataset()
    dataset_loader.preprocess_data()
    
    # Get data (in standardized feature space, matching the denormalized rules)
    # Rules are denormalized from [0, 1] to standardized space (mean=0, std=1)
    if use_test_data:
        X_data = dataset_loader.X_test_scaled  # Standardized feature space (matches rule space)
        y_data = dataset_loader.y_test
        data_type = "test"
    else:
        X_data = dataset_loader.X_train_scaled  # Standardized feature space (matches rule space)
        y_data = dataset_loader.y_train
        data_type = "training"
    
    feature_names = dataset_loader.feature_names
    
    logger.info(f"✓ Loaded {data_type} data: {X_data.shape[0]} samples, {X_data.shape[1]} features")
    logger.info(f"  Class distribution: {np.bincount(y_data)}")
    
    # Collect all unique rules from all classes
    per_class_results = rules_data.get("per_class_results", {})
    all_unique_rules = set()
    rule_to_source_classes = defaultdict(list)  # Track which classes each rule came from
    
    for class_key, class_data in per_class_results.items():
        target_class = class_data.get("class")
        unique_rules = class_data.get("unique_rules", [])
        for rule_str in unique_rules:
            all_unique_rules.add(rule_str)
            rule_to_source_classes[rule_str].append(target_class)
    
    all_unique_rules = sorted(list(all_unique_rules))  # Sort for consistent ordering
    
    logger.info(f"{'='*80}")
    logger.info(f"Testing {len(all_unique_rules)} unique rules against all classes")
    logger.info(f"{'='*80}")
    
    # Get all unique classes in the dataset
    unique_classes = sorted(list(np.unique(y_data)))
    logger.info(f"Classes in dataset: {unique_classes}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y_data, return_counts=True)))}")
    
    # Process each rule and test against all classes
    results = {
        "dataset": dataset_name,
        "data_type": data_type,
        "model_type": model_type,
        "algorithm": metadata.get("algorithm", "unknown"),
        "n_samples": X_data.shape[0],
        "n_features": X_data.shape[1],
        "classes": unique_classes,
        "rules_tested": len(all_unique_rules),
        "rule_results": [],
        # Include original per_class_results from rules file for consistency with missed_samples_analysis
        # Use the same reference (not a copy) to ensure we're using the exact same data structure
        "per_class_results": per_class_results
    }
    
    rules_satisfying_both_classes = []
    
    for rule_idx, rule_str in enumerate(all_unique_rules):
        logger.info(f"{'='*80}")
        logger.info(f"Rule {rule_idx + 1}/{len(all_unique_rules)}: {rule_str[:100]}...")
        logger.info(f"{'='*80}")
        logger.info(f"  Source classes: {rule_to_source_classes[rule_str]}")
        
        # Parse rule
        rule_conditions = parse_rule(rule_str)
        
        if len(rule_conditions) == 0:
            logger.info(f"  Empty rule - all samples satisfy it")
            satisfying_mask = np.ones(X_data.shape[0], dtype=bool)
        else:
            # Check which samples satisfy the rule
            satisfying_mask = check_rule_satisfaction(X_data, feature_names, rule_conditions)
        
        n_satisfying = np.sum(satisfying_mask)
        logger.info(f"  Total samples satisfying rule: {n_satisfying}/{X_data.shape[0]} ({100*n_satisfying/X_data.shape[0]:.2f}%)")
        
        # Test against each class
        rule_result = {
            "rule": rule_str,
            "rule_index": rule_idx,
            "source_classes": rule_to_source_classes[rule_str],
            "n_conditions": len(rule_conditions),
            "conditions": [
                {"feature": feat, "lower": lower, "upper": upper}
                for feat, lower, upper in rule_conditions
            ],
            "n_satisfying_samples": int(n_satisfying),
            "per_class_results": {}
        }
        
        classes_satisfied = []
        
        for target_class in unique_classes:
            n_class_samples = np.sum(y_data == target_class)
            n_satisfying_class = np.sum(satisfying_mask & (y_data == target_class))
            
            # Calculate precision and coverage for this class
            if n_satisfying > 0:
                precision = n_satisfying_class / n_satisfying
            else:
                precision = 0.0
            
            if n_class_samples > 0:
                coverage = n_satisfying_class / n_class_samples
            else:
                coverage = 0.0
            
            # Get indices of satisfying samples for this class
            satisfying_class_indices = np.where(satisfying_mask & (y_data == target_class))[0].tolist()
            
            class_result = {
                "class": int(target_class),
                "n_class_samples": int(n_class_samples),
                "n_satisfying_class_samples": int(n_satisfying_class),
                "rule_precision": float(precision),  # Rule-level precision
                "rule_coverage": float(coverage),    # Rule-level coverage
                "satisfying_sample_indices": satisfying_class_indices
            }
            
            # Try to get instance-level and class-level metrics from the rules file if available
            class_key = f"class_{target_class}"
            if class_key in per_class_results:
                class_data = per_class_results[class_key]
                # Instance-level metrics (from training/inference)
                if "instance_precision" in class_data:
                    class_result["instance_precision"] = float(class_data.get("instance_precision", 0.0))
                    class_result["instance_coverage"] = float(class_data.get("instance_coverage", 0.0))
                # Class-level metrics (from training/inference)
                if "class_precision" in class_data:
                    class_result["class_precision"] = float(class_data.get("class_precision", 0.0))
                    class_result["class_coverage"] = float(class_data.get("class_coverage", 0.0))
            
            rule_result["per_class_results"][f"class_{target_class}"] = class_result
            
            logger.info(f"  Class {target_class}:")
            logger.info(f"    Samples satisfying: {n_satisfying_class}/{n_class_samples} ({100*coverage:.2f}% coverage)")
            logger.info(f"    Rule-level precision: {precision:.4f} (calculated from testing)")
            
            # Only display instance-level and class-level metrics if:
            # 1. The rule matches samples from this class (n_satisfying_class > 0), OR
            # 2. This class is a source class for this rule (where it was extracted from)
            is_source_class = target_class in rule_to_source_classes[rule_str]
            should_show_metrics = n_satisfying_class > 0 or is_source_class
            
            if should_show_metrics:
                # Display instance-level and class-level metrics if available
                if "instance_precision" in class_result:
                    logger.info(f"    Instance-level precision: {class_result['instance_precision']:.4f} (from inference on {metrics_data_source})")
                    logger.info(f"    Instance-level coverage: {class_result['instance_coverage']:.4f} (from inference on {metrics_data_source})")
                if "class_precision" in class_result:
                    logger.info(f"    Class-level precision: {class_result['class_precision']:.4f} (from inference on {metrics_data_source})")
                    logger.info(f"    Class-level coverage: {class_result['class_coverage']:.4f} (from inference on {metrics_data_source})")
            
            if n_satisfying_class > 0:
                classes_satisfied.append(target_class)
        
        # Check if rule satisfies multiple classes
        if len(classes_satisfied) > 1:
            rule_result["satisfies_multiple_classes"] = True
            rule_result["classes_satisfied"] = classes_satisfied
            rules_satisfying_both_classes.append({
                "rule_index": rule_idx,
                "rule": rule_str,
                "classes_satisfied": classes_satisfied,
                "per_class_results": rule_result["per_class_results"]
            })
            logger.info(f"✓ Rule satisfies {len(classes_satisfied)} classes: {classes_satisfied}")
        else:
            rule_result["satisfies_multiple_classes"] = False
            rule_result["classes_satisfied"] = classes_satisfied
        
        results["rule_results"].append(rule_result)
    
    # Summary
    results["summary"] = {
        "total_rules": len(all_unique_rules),
        "rules_satisfying_multiple_classes": len(rules_satisfying_both_classes),
        "rules_satisfying_both_classes": rules_satisfying_both_classes
    }
    
    logger.info(f"{'='*80}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total unique rules tested: {len(all_unique_rules)}")
    logger.info(f"Rules satisfying multiple classes: {len(rules_satisfying_both_classes)}")
    
    if rules_satisfying_both_classes:
        logger.info(f"Rules that satisfy multiple classes:")
        for rule_info in rules_satisfying_both_classes:
            logger.info(f"  Rule {rule_info['rule_index'] + 1}: {rule_info['rule'][:80]}...")
            logger.info(f"    Classes satisfied: {rule_info['classes_satisfied']}")
            for class_val in rule_info['classes_satisfied']:
                class_key = f"class_{class_val}"
                class_res = rule_info['per_class_results'][class_key]
                # Try rule_precision/rule_coverage first (calculated in this script), then anchor_precision/anchor_coverage
                rule_prec = class_res.get('rule_precision', class_res.get('anchor_precision', 0.0))
                rule_cov = class_res.get('rule_coverage', class_res.get('anchor_coverage', 0.0))
                logger.info(f"      Class {class_val}: rule_precision={rule_prec:.4f}, rule_coverage={rule_cov:.4f}")
                if "instance_precision" in class_res:
                    logger.info(f"        Instance-level: precision={class_res['instance_precision']:.4f}, coverage={class_res['instance_coverage']:.4f}")
                if "class_precision" in class_res:
                    logger.info(f"        Class-level: precision={class_res['class_precision']:.4f}, coverage={class_res['class_coverage']:.4f}")
    else:
        logger.info(f"No rules satisfy multiple classes.")
    
    # Analyze rule overlaps in detail
    overlap_analysis = analyze_rule_overlaps_detailed(
        results=results,
        X_data=X_data,
        y_data=y_data,
        feature_names=feature_names,
        unique_classes=unique_classes
    )
    results["overlap_analysis"] = overlap_analysis
    
    # Analyze missed samples per class
    missed_samples_analysis = analyze_missed_samples(
        results=results,
        X_data=X_data,
        y_data=y_data,
        feature_names=feature_names,
        unique_classes=unique_classes
    )
    results["missed_samples_analysis"] = missed_samples_analysis

    # Build post-hoc global explanations using the requested selection parameters
    global_explanations = select_global_rules_per_class(
        results=results,
        X_data=X_data,
        y_data=y_data,
        feature_names=feature_names,
        unique_classes=unique_classes,
        precision_threshold=precision_threshold,
        max_rules_per_class=max_rules_per_class,
        overlap_penalty_weight=overlap_penalty_weight,
    )
    results["global_explanations"] = global_explanations
    
    # Log global explanations results
    logger.info("\n" + "="*80)
    logger.info("GLOBAL EXPLANATIONS (All Available High-Precision Rules per Class)")
    logger.info("="*80)
    settings = global_explanations.get("settings", {})
    max_rules_setting = settings.get('max_rules_per_class', -1)
    max_rules_display = "all available" if max_rules_setting == -1 else str(max_rules_setting)
    logger.info(f"Settings: precision_threshold={settings.get('precision_threshold', 0.9):.2f}, "
                f"max_rules_per_class={max_rules_display}, "
                f"overlap_penalty_weight={settings.get('overlap_penalty_weight', 0.0):.2f}")
    logger.info("")
    
    for class_key, class_data in global_explanations.get("per_class", {}).items():
        cls = class_data.get("class", -1)
        n_selected = class_data.get("n_selected_rules", 0)
        n_class_samples = class_data.get("n_class_samples", 0)
        n_covered = class_data.get("n_covered_class_samples", 0)
        union_cov = class_data.get("class_union_coverage", 0.0)
        union_prec = class_data.get("class_union_precision", 0.0)
        selected_rules = class_data.get("selected_rules", [])
        selected_indices = class_data.get("selected_rule_indices", [])
        
        logger.info(f"Class {cls}:")
        logger.info(f"  Class samples: {n_class_samples}")
        logger.info(f"  Selected rules: {n_selected}")
        logger.info(f"  Class-union coverage: {union_cov:.4f} ({n_covered}/{n_class_samples} samples covered)")
        logger.info(f"  Class-union precision: {union_prec:.4f}")
        
        if n_selected > 0:
            logger.info(f"  Selected rule indices: {selected_indices}")
            for idx, rule_str in enumerate(selected_rules, 1):
                # Truncate long rules for readability
                rule_display = rule_str[:100] + "..." if len(rule_str) > 100 else rule_str
                logger.info(f"    Rule {idx} (index {selected_indices[idx-1]}): {rule_display}")
        else:
            logger.info(f"  No rules selected (no rules met precision threshold {settings.get('precision_threshold', 0.9):.2f})")
        logger.info("")
    
    logger.info("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test single-agent extracted rules against a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test rules on test data (default)
  python single_agent/test_extracted_rules.py --rules_file path/to/extracted_rules_single_agent.json --dataset breast_cancer
  
  # Test rules on training data
  python single_agent/test_extracted_rules.py --rules_file path/to/extracted_rules_single_agent.json --dataset breast_cancer --use_train_data
        """
    )
    
    parser.add_argument(
        "--rules_file",
        type=str,
        required=True,
        help="Path to extracted_rules_single_agent.json file"
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
        required=True,
        choices=dataset_choices,
        help="Dataset name. For UCIML: uci_<name_or_id>. For Folktables: folktables_<task>_<state>_<year>"
    )
    
    parser.add_argument(
        "--use_train_data",
        action="store_true",
        help="Test on training data instead of test data (default: test data)"
    )
    
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset loading (default: 42)"
    )
    
    parser.add_argument(
        "--precision_threshold",
        type=float,
        default=0.9,
        help="Minimum rule-level precision for a rule to be considered in global explanations (default: 0.9)"
    )

    parser.add_argument(
        "--max_rules_per_class",
        type=int,
        default=-1,
        help="Maximum number of rules to select per class for global explanations (-1 = no limit, use all)"
    )

    parser.add_argument(
        "--overlap_penalty_weight",
        type=float,
        default=0.0,
        help="Penalty weight for selecting highly overlapping rules in global explanations (default: 0.0)"
    )

    args = parser.parse_args()

    # Determine experiment directory and set up logging
    experiment_dir = get_experiment_dir_from_rules_file(args.rules_file)
    
    # Create log file in experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(experiment_dir, f"test_rules_{timestamp}.log")
    
    # Setup logging to both console and file
    setup_file_logging(log_file_path)
    logger.info(f"{'='*80}")
    logger.info(f"Logging to file: {log_file_path}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"{'='*80}")
    
    try:
        # Load rules_data to access metadata and per_class_results
        logger.info("Loading rules file for metadata...")
        with open(args.rules_file, 'r') as f:
            rules_data = json.load(f)
        
        # Get metadata to determine data source for metrics labels
        metadata = rules_data.get("metadata", {})
        inference_eval_on_test_data = metadata.get("eval_on_test_data", False)
        metrics_data_source = "test data" if inference_eval_on_test_data else "training data"
        
        # Test rules
        results = test_rules_from_json(
            rules_file=args.rules_file,
            dataset_name=args.dataset,
            use_test_data=not args.use_train_data,
            seed=args.seed,
            precision_threshold=args.precision_threshold,
            max_rules_per_class=args.max_rules_per_class,
            overlap_penalty_weight=args.overlap_penalty_weight,
        )
        
        # Results are logged to file automatically via logging
        logger.info(f"{'='*80}")
        logger.info(f"✓ Log file saved to: {log_file_path}")
        
        # Print per-class summary
        logger.info(f"{'='*80}")
        logger.info("PER-CLASS SUMMARY")
        logger.info(f"{'='*80}")
        
        unique_classes = results.get("classes", [])
        # Use inference results (from rules_data) for instance/class-level metrics, not test results
        inference_per_class_results = rules_data.get("per_class_results", {})
        for target_class in unique_classes:
            logger.info(f"Class {target_class}:")

            rule_precisions = []
            rule_coverages = []

            for rule_result in results["rule_results"]:
                class_key = f"class_{target_class}"
                if class_key in rule_result["per_class_results"]:
                    class_res = rule_result["per_class_results"][class_key]
                    # Rule-level metrics (calculated by this test script)
                    rule_prec = class_res.get("rule_precision", class_res.get("anchor_precision", 0.0))
                    rule_cov = class_res.get("rule_coverage", class_res.get("anchor_coverage", 0.0))
                    rule_precisions.append(rule_prec)
                    rule_coverages.append(rule_cov)

            if rule_precisions:
                logger.info(f"  Rules tested: {len(rule_precisions)}")
                logger.info(f"  Rule-level metrics (from testing):")
                logger.info(f"    Mean precision: {np.mean(rule_precisions):.4f} ± {np.std(rule_precisions):.4f}")
                logger.info(f"    Mean coverage: {np.mean(rule_coverages):.4f} ± {np.std(rule_coverages):.4f}")
                logger.info(f"    Best precision: {np.max(rule_precisions):.4f}")
                logger.info(f"    Best coverage: {np.max(rule_coverages):.4f}")

            # Instance/class-level metrics from inference (stored in rules_data, not test results)
            class_key = f"class_{target_class}"
            if class_key in inference_per_class_results:
                inference_data = inference_per_class_results[class_key]
                if "instance_precision" in inference_data:
                    logger.info(f"  Instance-level metrics (from inference on {metrics_data_source}):")
                    logger.info(f"    Precision: {inference_data['instance_precision']:.4f}")
                    logger.info(f"    Coverage: {inference_data['instance_coverage']:.4f}")
                if "class_precision" in inference_data:
                    logger.info(f"  Class-level metrics (from inference on {metrics_data_source}):")
                    logger.info(f"    Precision: {inference_data['class_precision']:.4f}")
                    logger.info(f"    Coverage: {inference_data['class_coverage']:.4f}")
        
        logger.info(f"{'='*80}")
        logger.info("Rule testing complete!")
        logger.info(f"{'='*80}")
    
    except Exception as e:
        logger.error(f"Error during rule testing: {str(e)}", exc_info=True)
        logger.error(f"{'='*80}")
        logger.error(f"Log file saved to: {log_file_path}")
        logger.error(f"{'='*80}")
        raise
    
    finally:
        # Ensure all log handlers are flushed and closed
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.flush()
            if hasattr(handler, 'close'):
                handler.close()
        
        # Print log file location to console (even if logging fails)
        print(f"\n{'='*80}")
        print(f"Log file saved to: {log_file_path}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

