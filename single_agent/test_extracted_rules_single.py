#!/usr/bin/env python3
"""
Test Single-Agent Extracted Rules Script

This script reads extracted rules from a single-agent SB3 inference JSON file and tests them
against a dataset to find all samples that satisfy each rule.

Usage:
    python single_agent/test_extracted_rules_single.py --rules_file <path_to_extracted_rules_single_agent.json> --dataset <dataset_name> [--use_train_data] [--use_full_dataset]

Example:
    python single_agent/test_extracted_rules_single.py \
        --rules_file output/single_agent_sb3_breast_cancer_ddpg/training/.../inference/extracted_rules_single_agent.json \
        --dataset breast_cancer
    
    # Test on full dataset (train + test combined)
    python single_agent/test_extracted_rules_single.py \
        --rules_file output/single_agent_sb3_breast_cancer_ddpg/training/.../inference/extracted_rules_single_agent.json \
        --dataset breast_cancer --use_full_dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import re
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
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
    Destructive: replaces all existing handlers. Use for the CLI entry point
    where this script owns the logger. For in-process callers that already
    have a logger configured (e.g. the wyodot pipeline), prefer
    `add_file_log_handler` which is additive.
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


def add_file_log_handler(log_file_path: str) -> logging.FileHandler:
    """Append a FileHandler to the root logger without disturbing existing
    handlers. Returns the handler so the caller can remove it afterward
    (use logging.getLogger().removeHandler(handler)). Use this from
    in-process callers like wyodot/run_pipeline.py that have their own
    pipeline-level logging already set up."""
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)
    return fh


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
        # Skip if target_class is None (invalid class data)
        if target_class is None:
            continue
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
            logger.info(f"  Rule: {overlap_info['rule']}")
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
                logger.info(f"    {rule_idx}. {rule_str}")
    
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
                logger.info(f"    {rule_idx}. {rule_str}")
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
        # Skip class-based result keys (they're stored separately, not as actual classes)
        if class_key.endswith("_class_based") or class_data.get("rollout_type") == "class_based":
            continue
        
        target_class = class_data.get("class")
        # Skip if target_class is None (invalid class data)
        if target_class is None:
            continue
        
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
        # (same source as missed_samples_analysis to ensure consistency).
        # We build candidates_all (everything that touches at least one class sample)
        # AND candidates (the subset meeting precision_threshold). If candidates is
        # empty but candidates_all is not, we fall back to candidates_all ranked by
        # precision so the global-explanation report can show *something* instead
        # of a misleading 0/0 — and we flag the fallback so the caller knows.
        candidates: List[Dict[str, Any]] = []
        candidates_all: List[Dict[str, Any]] = []
        if class_key not in per_class_results:
            # Fallback: if class_key not in per_class_results, try to get from rule_results
            # This shouldn't happen if per_class_results is properly populated
            logger.warning(f"  Class key {class_key} not found in per_class_results. Available keys: {list(per_class_results.keys())}")
        else:
            unique_rules = per_class_results[class_key].get("unique_rules", [])
            # Fallback: if the instance-based slot is empty (e.g. the classifier
            # never predicted this class for any sampled instance, so no
            # instance-based rollouts ran), borrow the class-based rules from
            # the sibling "class_<N>_class_based" entry. These rules came from
            # centroid-seeded class-based rollouts and don't depend on
            # prediction routing — so they're available even when instance
            # routing produces zero hits for the class.
            cb_key = f"{class_key}_class_based"
            if not unique_rules and cb_key in per_class_results:
                cb_rules = per_class_results[cb_key].get("unique_rules", [])
                if cb_rules:
                    unique_rules = cb_rules
                    logger.info(
                        f"  Class {cls}: instance-based rules empty; falling back to "
                        f"{len(cb_rules)} class-based rules from {cb_key} for global explanation."
                    )
            if not unique_rules:
                logger.debug(f"  No unique_rules found for {class_key} or {cb_key} in per_class_results")

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

                # Get indices of satisfying class samples (in full dataset indices)
                class_satisfying_mask = satisfying_mask & class_mask
                satisfying_class_indices = set(np.where(class_satisfying_mask)[0].tolist())

                # Find rule index in rule_results for union precision calculation
                rule_idx = rule_str_to_idx.get(rule_str, -1)

                cand = {
                    "rule_idx": rule_idx,
                    "rule_str": rule_str,
                    "precision": rule_prec,
                    "class_indices": satisfying_class_indices,
                }
                candidates_all.append(cand)
                if rule_prec >= precision_threshold:
                    candidates.append(cand)

        # Fallback: if no rule met precision_threshold but rules do exist that
        # cover this class, surface the top-K-by-precision so the user can see
        # what the policy actually produced. The report layer can flag these
        # via "fallback_used" / "max_candidate_precision".
        used_fallback = False
        if not candidates and candidates_all:
            used_fallback = True
            # Sort by precision desc; tie-break by class coverage desc
            candidates_all.sort(
                key=lambda c: (c["precision"], len(c["class_indices"])),
                reverse=True,
            )
            fallback_k = max_rules_per_class if (max_rules_per_class and max_rules_per_class > 0) else len(candidates_all)
            candidates = candidates_all[:fallback_k]
            logger.warning(
                f"  Class {cls}: no rule met precision_threshold={precision_threshold:.2f}; "
                f"falling back to top {len(candidates)} of {len(candidates_all)} rules by precision "
                f"(max precision available: {candidates_all[0]['precision']:.4f})"
            )

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

        max_candidate_precision = max((c["precision"] for c in candidates_all), default=0.0)
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
            # Honesty fields: did we have to drop below the precision bar to show anything?
            "fallback_used": bool(used_fallback),
            "n_candidates_total": int(len(candidates_all)),
            "max_candidate_precision": float(max_candidate_precision),
        }

    return global_explanations


def test_rules_from_json(
    rules_file: str,
    dataset_name: str,
    use_test_data: bool = True,
    use_full_dataset: bool = False,
    seed: int = 42,
    precision_threshold: float = 0.9,
    max_rules_per_class: int = -1,
    overlap_penalty_weight: float = 0.0,
    report_md_path: Optional[str] = None,
) -> Dict:
    """
    Test extracted rules against a dataset.
    
    Args:
        rules_file: Path to extracted_rules_single_agent.json file
        dataset_name: Name of the dataset to test against
        use_test_data: If True, test on test data; if False, test on training data
        use_full_dataset: If True, test on full dataset (train + test combined); overrides use_test_data
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
    if use_full_dataset:
        logger.info(f"Data split: full (train + test combined)")
    else:
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
    # Check if inference was run on test data and coverage_on_all_data flag
    inference_eval_on_test_data = metadata.get("eval_on_test_data", False)
    inference_coverage_on_all_data = metadata.get("coverage_on_all_data", False)
    if inference_coverage_on_all_data:
        metrics_data_source = "full dataset (train + test)"
    elif inference_eval_on_test_data:
        metrics_data_source = "test data"
    else:
        metrics_data_source = "training data"
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    dataset_loader = TabularDatasetLoader(
        dataset_name=dataset_name,
        test_size=0.2,
        random_state=seed
    )
    dataset_loader.load_dataset()
    dataset_loader.preprocess_data()
    
    # Get data in original (raw) feature space, matching the denormalized rule bounds.
    # Rules are denormalized all the way back to raw feature units (inverse StandardScaler).
    if use_full_dataset:
        X_data = np.vstack([dataset_loader.X_train, dataset_loader.X_test]).astype(np.float32)
        y_data = np.concatenate([dataset_loader.y_train, dataset_loader.y_test])
        data_type = "full (train + test)"
        logger.info(f"✓ Loaded full dataset: {len(dataset_loader.y_train)} training + {len(dataset_loader.y_test)} test = {X_data.shape[0]} total samples")
    elif use_test_data:
        X_data = dataset_loader.X_test.astype(np.float32)  # Raw feature space (matches rule space)
        y_data = dataset_loader.y_test
        data_type = "test"
    else:
        X_data = dataset_loader.X_train.astype(np.float32)  # Raw feature space (matches rule space)
        y_data = dataset_loader.y_train
        data_type = "training"
    
    feature_names = dataset_loader.feature_names
    
    logger.info(f"✓ Loaded {data_type} data: {X_data.shape[0]} samples, {X_data.shape[1]} features")
    logger.info(f"  Class distribution: {np.bincount(y_data)}")
    
    # Collect all unique rules from all classes
    # Separate instance-based and class-based rules for separate testing
    per_class_results = rules_data.get("per_class_results", {})
    all_unique_rules = set()
    instance_based_rules = set()
    class_based_rules = set()
    rule_to_source_classes = defaultdict(list)  # Track which classes each rule came from
    rule_to_rollout_type = {}  # Track whether rule is instance-based or class-based
    rule_to_original_predictions = defaultdict(list)  # Track original predictions for instance-based rules
    
    for class_key, class_data in per_class_results.items():
        # Skip separate class-based keys (we'll process them separately)
        if class_key.endswith("_class_based") or class_data.get("rollout_type") == "class_based":
            continue
        
        target_class = class_data.get("class")
        if target_class is None:
            continue
        
        # Collect instance-based rules
        unique_rules = class_data.get("unique_rules", [])
        anchors_list = class_data.get("anchors", [])  # Get anchors to extract original predictions
        rule_to_orig_pred_for_class = defaultdict(set)  # Track original predictions per rule for this class
        
        # Extract original predictions from anchors for this class
        for anchor in anchors_list:
            if anchor.get("rollout_type") == "instance_based" and "rule" in anchor and "original_prediction" in anchor:
                rule_str = anchor["rule"]
                orig_pred = anchor["original_prediction"]
                rule_to_orig_pred_for_class[rule_str].add(orig_pred)
                rule_to_original_predictions[rule_str].append(orig_pred)
        
        for rule_str in unique_rules:
            all_unique_rules.add(rule_str)
            instance_based_rules.add(rule_str)
            rule_to_source_classes[rule_str].append(target_class)
            # CRITICAL FIX: Always set instance_based, never overwrite it with class_based later
            # This ensures prediction-match precision is used for instance-based rules even if they also appear in class-based sets
            rule_to_rollout_type[rule_str] = "instance_based"
        
        # Collect class-based rules from nested dict (legacy format)
        class_based_results = class_data.get("class_based_results", {})
        if isinstance(class_based_results, dict):
            # Check if it's a dict of results or a single result dict
            if "unique_rules" in class_based_results:
                # Single result dict
                cb_unique_rules = class_based_results.get("unique_rules", [])
                for rule_str in cb_unique_rules:
                    all_unique_rules.add(rule_str)
                    class_based_rules.add(rule_str)
                    rule_to_source_classes[rule_str].append(target_class)
                    # CRITICAL FIX: Only set class_based if rule doesn't already exist as instance_based
                    # This preserves instance-based classification for overlapping rules
                    if rule_str not in rule_to_rollout_type:
                        rule_to_rollout_type[rule_str] = "class_based"
            else:
                # Dict of agent results (for multi-agent compatibility)
                for agent_name, agent_results in class_based_results.items():
                    cb_unique_rules = agent_results.get("unique_rules", [])
                    for rule_str in cb_unique_rules:
                        all_unique_rules.add(rule_str)
                        class_based_rules.add(rule_str)
                        rule_to_source_classes[rule_str].append(target_class)
                        # CRITICAL FIX: Only set class_based if rule doesn't already exist as instance_based
                        if rule_str not in rule_to_rollout_type:
                            rule_to_rollout_type[rule_str] = "class_based"
    
    # CRITICAL: Also collect class-based rules from separate keys (current format)
    # Inference saves class-based rules in separate class_{class}_class_based keys
    for class_key, class_data in per_class_results.items():
        if class_key.endswith("_class_based") or class_data.get("rollout_type") == "class_based":
            target_class = class_data.get("class")
            if target_class is None:
                # Try to extract from key name
                if class_key.endswith("_class_based"):
                    try:
                        target_class = int(class_key.replace("class_", "").replace("_class_based", ""))
                    except ValueError:
                        continue
            
            # Collect class-based rules from separate key
            cb_unique_rules = class_data.get("unique_rules", [])
            for rule_str in cb_unique_rules:
                all_unique_rules.add(rule_str)
                class_based_rules.add(rule_str)
                rule_to_source_classes[rule_str].append(target_class)
                # CRITICAL FIX: Only set class_based if rule doesn't already exist as instance_based
                # This preserves instance-based classification for overlapping rules
                # Instance-based rules should use prediction-match precision, which is more accurate for fidelity
                if rule_str not in rule_to_rollout_type:
                    rule_to_rollout_type[rule_str] = "class_based"
    
    all_unique_rules = sorted(list(all_unique_rules))  # Sort for consistent ordering
    instance_based_rules = sorted(list(instance_based_rules))
    class_based_rules = sorted(list(class_based_rules))
    
    # Count overlapping rules (rules that appear in both instance-based and class-based sets)
    # Convert to sets for intersection operation, then back to sorted list
    overlapping_rules = sorted(list(set(instance_based_rules) & set(class_based_rules)))
    
    logger.info(f"{'='*80}")
    logger.info(f"Found {len(all_unique_rules)} total unique rules")
    logger.info(f"  Instance-based rules: {len(instance_based_rules)}")
    logger.info(f"  Class-based rules: {len(class_based_rules)}")
    if overlapping_rules:
        logger.info(f"  Overlapping rules (in both sets): {len(overlapping_rules)}")
        logger.info(f"    Note: Overlapping rules will use instance-based classification (prediction-match precision)")
        logger.debug(f"    Overlapping rules: {overlapping_rules[:5]}{'...' if len(overlapping_rules) > 5 else ''}")
    logger.info(f"  Overlapping rules (in both): {len(set(instance_based_rules) & set(class_based_rules))}")
    logger.info(f"{'='*80}")
    logger.info(f"Testing all {len(all_unique_rules)} unique rules against all classes")
    logger.info(f"{'='*80}")
    
    # Get all unique classes in the dataset
    unique_classes = sorted(list(np.unique(y_data)))
    logger.info(f"Classes in dataset: {unique_classes}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y_data, return_counts=True)))}")
    
    # For instance-based rules, track original predictions per rule
    # CRITICAL FIX: Store all original predictions per rule, not just most common
    # This allows computing precision per source class when a rule has multiple original predictions
    from collections import Counter
    rule_to_most_common_orig_pred = {}  # For backward compatibility / fallback
    rule_to_all_orig_preds = {}  # Store all original predictions per rule
    rule_to_orig_pred_by_source_class = {}  # Store original predictions per rule per source class
    
    for rule_str, orig_preds in rule_to_original_predictions.items():
        if orig_preds:
            # Store all original predictions for this rule
            rule_to_all_orig_preds[rule_str] = list(orig_preds)
            # Use most common as fallback (for backward compatibility)
            counter = Counter(orig_preds)
            rule_to_most_common_orig_pred[rule_str] = counter.most_common(1)[0][0]
    
    # Build mapping from rule to original predictions by source class
    # This allows computing precision per source class when rules span multiple classes
    for class_key, class_data in per_class_results.items():
        if class_key.endswith("_class_based") or class_data.get("rollout_type") == "class_based":
            continue
        target_class = class_data.get("class")
        if target_class is None:
            continue
        anchors_list = class_data.get("anchors", [])
        for anchor in anchors_list:
            if anchor.get("rollout_type") == "instance_based" and "rule" in anchor and "original_prediction" in anchor:
                rule_str = anchor["rule"]
                orig_pred = anchor["original_prediction"]
                if rule_str not in rule_to_orig_pred_by_source_class:
                    rule_to_orig_pred_by_source_class[rule_str] = {}
                if target_class not in rule_to_orig_pred_by_source_class[rule_str]:
                    rule_to_orig_pred_by_source_class[rule_str][target_class] = []
                rule_to_orig_pred_by_source_class[rule_str][target_class].append(orig_pred)
    
    # Load classifier for prediction-match precision calculation (instance-based rules only)
    classifier = None
    device = "cpu"
    try:
        import torch
        if hasattr(dataset_loader, 'classifier') and dataset_loader.classifier is not None:
            classifier = dataset_loader.classifier
            logger.info("✓ Using classifier from dataset_loader for prediction-match precision (instance-based rules)")
        else:
            # Search likely on-disk locations relative to the rules file.
            # Single-agent layout typically: <exp_dir>/inference/extracted_rules_*.json
            # with classifier at <exp_dir>/classifier.pth or <exp_dir>/training/classifier.pth.
            rules_parent = os.path.dirname(os.path.abspath(rules_file))
            exp_dir = os.path.dirname(rules_parent)
            candidates = [
                os.path.join(exp_dir, "classifier.pth"),
                os.path.join(exp_dir, "training", "classifier.pth"),
                os.path.join(rules_parent, "classifier.pth"),
                os.path.join(os.path.dirname(exp_dir), "training", "classifier.pth"),
                os.path.join("models", f"classifier_{dataset_name}_seed{seed}.pth"),
            ]
            classifier_path = next((p for p in candidates if os.path.exists(p)), None)
            if classifier_path is not None:
                logger.info(f"Loading classifier from {classifier_path}...")
                classifier = dataset_loader.load_classifier(filepath=classifier_path, device=device)
                dataset_loader.classifier = classifier
                logger.info("✓ Classifier loaded from file for prediction-match precision (instance-based rules)")
            else:
                logger.warning(f"Classifier not found in any of: {candidates}")
                logger.warning("Will use class-label precision for all rules (including instance-based)")
    except Exception as e:
        logger.warning(f"Could not load classifier for prediction-match precision: {e}")
        logger.warning("Will use class-label precision for all rules (including instance-based)")

    # Precompute classifier predictions on every row of X_data once. The per-rule
    # loop below indexes into all_predictions via a boolean mask (see prediction-
    # match precision branch around line 1073). Without this, that branch raises
    # NameError: name 'all_predictions' is not defined. We standardize first
    # because the classifier (RF or DNN) was trained on scaler-normalized features.
    all_predictions = None
    if classifier is not None:
        try:
            X_for_pred = X_data
            if hasattr(dataset_loader, "scaler") and dataset_loader.scaler is not None:
                X_for_pred = dataset_loader.scaler.transform(X_data).astype(np.float32)
            if hasattr(classifier, "predict"):
                # sklearn estimators (RandomForest, GradientBoosting, ...)
                all_predictions = np.asarray(classifier.predict(X_for_pred))
            else:
                # torch nn.Module (DNN classifier)
                import torch as _torch
                if hasattr(classifier, "eval"):
                    classifier.eval()
                with _torch.no_grad():
                    X_tensor = _torch.from_numpy(np.asarray(X_for_pred, dtype=np.float32))
                    logits = classifier(X_tensor)
                    all_predictions = logits.argmax(dim=1).cpu().numpy()
            logger.info(f"✓ Precomputed classifier predictions on all {len(all_predictions)} samples for prediction-match precision")
        except Exception as e:
            logger.warning(f"Could not precompute classifier predictions: {e}")
            logger.warning("Will fall back to class-label precision for instance-based rules")
            all_predictions = None

    # Process each rule and test against all classes
    results = {
        "dataset": dataset_name,
        "data_type": data_type,
        "model_type": model_type,
        "algorithm": metadata.get("algorithm", "unknown"),
        "n_samples": X_data.shape[0],
        "n_features": X_data.shape[1],
        "classes": unique_classes,
        "class_names": {
            f"class_{int(cls)}": (
                dataset_loader.class_names[int(cls)]
                if getattr(dataset_loader, "class_names", None)
                and int(cls) < len(dataset_loader.class_names)
                else f"class_{int(cls)}"
            )
            for cls in unique_classes
        },
        "rules_tested": len(all_unique_rules),
        "rule_results": [],
        # Include original per_class_results from rules file for consistency with missed_samples_analysis
        # Use the same reference (not a copy) to ensure we're using the exact same data structure
        "per_class_results": per_class_results
    }
    
    rules_satisfying_both_classes = []
    
    for rule_idx, rule_str in enumerate(all_unique_rules):
        rollout_type = rule_to_rollout_type.get(rule_str, "unknown")
        logger.info(f"{'='*80}")
        logger.info(f"Rule {rule_idx + 1}/{len(all_unique_rules)}: {rule_str}")
        logger.info(f"{'='*80}")
        logger.info(f"  Rollout type: {rollout_type}")
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
            "rollout_type": rule_to_rollout_type.get(rule_str, "unknown"),  # Track instance vs class-based
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
            
            # Calculate precision based on rollout type to match rollout/recomputation metrics
            # - Instance-based: Use prediction-match precision (P(pred(x) = original_pred | x satisfies rule))
            #   This matches the recomputation metric used in inference
            # - Class-based: Use class-label precision (P(y = target_class | x satisfies rule))
            #   This matches the class-based precision used in inference
            if rollout_type == "instance_based" and classifier is not None and all_predictions is not None:
                # Instance-based rule: Compute prediction-match precision
                # CRITICAL FIX: Use original prediction for this specific source class if available,
                # otherwise fall back to most common across all source classes
                # This handles rules extracted from multiple classes with different original predictions
                original_prediction = None
                if rule_str in rule_to_orig_pred_by_source_class and target_class in rule_to_orig_pred_by_source_class[rule_str]:
                    # Use most common original prediction for this specific source class
                    source_class_preds = rule_to_orig_pred_by_source_class[rule_str][target_class]
                    if source_class_preds:
                        counter = Counter(source_class_preds)
                        original_prediction = counter.most_common(1)[0][0]
                        logger.debug(f"  Using original prediction {original_prediction} for source class {target_class} (from {len(source_class_preds)} instances)")
                
                if original_prediction is None and rule_str in rule_to_most_common_orig_pred:
                    # Fallback: use most common original prediction across all source classes
                    original_prediction = rule_to_most_common_orig_pred[rule_str]
                    logger.debug(f"  Using most common original prediction {original_prediction} across all source classes (fallback)")
                
                if original_prediction is not None and n_satisfying > 0:
                    # Use pre-computed predictions (performance optimization)
                    predictions_satisfying = all_predictions[satisfying_mask]
                    # Precision = fraction of satisfying samples with prediction matching original
                    n_matching_pred = (predictions_satisfying == original_prediction).sum()
                    precision = float(n_matching_pred / n_satisfying)
                else:
                    precision = 0.0
            else:
                # Class-based rule (or instance-based without classifier): Use class-label precision
                # This is P(y = target_class | x satisfies rule)
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
            if rollout_type == "instance_based" and rule_str in rule_to_most_common_orig_pred and classifier is not None:
                logger.info(f"    Rule-level precision: {precision:.4f} (prediction-match, matches rollout/recomputation metrics)")
            else:
                logger.info(f"    Rule-level precision: {precision:.4f} (class-label, calculated from testing)")
            
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
    
    # Rank rules per class based on precision and coverage
    logger.info(f"\n{'='*80}")
    logger.info("RANKING RULES PER CLASS")
    logger.info(f"{'='*80}")
    
    # For each class, collect all rules with their precision and coverage, then rank them
    # CRITICAL FIX: Separate rankings for instance-based and class-based rules to avoid mixing precision semantics
    # Instance-based uses prediction-match precision, class-based uses class-label precision
    ranked_rules_per_class = {}
    for target_class in unique_classes:
        instance_based_rules_with_metrics = []
        class_based_rules_with_metrics = []
        
        for rule_result in results["rule_results"]:
            class_key = f"class_{target_class}"
            if class_key in rule_result["per_class_results"]:
                class_res = rule_result["per_class_results"][class_key]
                rule_str = rule_result["rule"]
                rollout_type = rule_result.get("rollout_type", "unknown")
                
                # Get rule-level precision and coverage (from testing)
                rule_precision = class_res.get("rule_precision", 0.0)
                rule_coverage = class_res.get("rule_coverage", 0.0)
                
                # Calculate a combined score for ranking (weighted: precision more important)
                # Using: precision * (1 + coverage)
                if rule_precision > 0:
                    combined_score = rule_precision * (1.0 + rule_coverage)
                else:
                    combined_score = 0.0
                
                rule_info = {
                    "rule": rule_str,
                    "rule_precision": rule_precision,
                    "rule_coverage": rule_coverage,
                    "combined_score": combined_score,
                    "rule_index": rule_result.get("rule_index", -1),
                    "precision_type": "prediction-match" if rollout_type == "instance_based" else "class-label"
                }
                
                # Separate by rollout type to avoid mixing precision semantics
                if rollout_type == "instance_based":
                    instance_based_rules_with_metrics.append(rule_info)
                elif rollout_type == "class_based":
                    class_based_rules_with_metrics.append(rule_info)
                else:
                    # Unknown type - add to both for backward compatibility
                    instance_based_rules_with_metrics.append(rule_info)
                    class_based_rules_with_metrics.append(rule_info)
        
        # Sort each group separately by combined score (descending), then by precision, then by coverage
        instance_based_rules_with_metrics.sort(
            key=lambda x: (x["combined_score"], x["rule_precision"], x["rule_coverage"]),
            reverse=True
        )
        class_based_rules_with_metrics.sort(
            key=lambda x: (x["combined_score"], x["rule_precision"], x["rule_coverage"]),
            reverse=True
        )
        
        # Store both rankings separately
        ranked_rules_per_class[target_class] = {
            "instance_based": instance_based_rules_with_metrics,
            "class_based": class_based_rules_with_metrics,
            "all": instance_based_rules_with_metrics + class_based_rules_with_metrics  # Combined for backward compatibility
        }
        
        total_rules = len(instance_based_rules_with_metrics) + len(class_based_rules_with_metrics)
        logger.info(f"Class {target_class}: Ranked {total_rules} rules ({len(instance_based_rules_with_metrics)} instance-based, {len(class_based_rules_with_metrics)} class-based)")
        
        # Log top rules separately by type to avoid mixing precision semantics
        if instance_based_rules_with_metrics:
            logger.info(f"  Top 3 instance-based rules (prediction-match precision):")
            for rank, rule_info in enumerate(instance_based_rules_with_metrics[:3], 1):
                logger.info(f"    Rank {rank}: precision={rule_info['rule_precision']:.4f}, "
                          f"coverage={rule_info['rule_coverage']:.4f}, "
                          f"score={rule_info['combined_score']:.4f}")
                logger.info(f"      Rule: {rule_info['rule']}")
        
        if class_based_rules_with_metrics:
            logger.info(f"  Top 3 class-based rules (class-label precision):")
            for rank, rule_info in enumerate(class_based_rules_with_metrics[:3], 1):
                logger.info(f"    Rank {rank}: precision={rule_info['rule_precision']:.4f}, "
                          f"coverage={rule_info['rule_coverage']:.4f}, "
                          f"score={rule_info['combined_score']:.4f}")
                logger.info(f"      Rule: {rule_info['rule']}")
    
    # Store ranked rules in results
    results["ranked_rules_per_class"] = ranked_rules_per_class
    
    # Update per_class_results to include ranked rules with metrics
    for class_key, class_data in results["per_class_results"].items():
        target_class = class_data.get("class", -1)
        if target_class in ranked_rules_per_class:
            ranked_rules = ranked_rules_per_class[target_class]
            # Store ranked rules with their metrics (now a dict with instance_based, class_based, all)
            class_data["ranked_rules"] = ranked_rules
            # Also update unique_rules to be in ranked order (top rules first)
            # Use "all" for backward compatibility (combined list)
            if isinstance(ranked_rules, dict) and "all" in ranked_rules:
                class_data["ranked_unique_rules"] = [r["rule"] for r in ranked_rules["all"]]
            elif isinstance(ranked_rules, list):
                # Backward compatibility: if it's still a list, use it directly
                class_data["ranked_unique_rules"] = [r["rule"] for r in ranked_rules]
            else:
                class_data["ranked_unique_rules"] = []
    
    # Compute statistics by rollout type
    instance_based_stats = {
        "count": len(instance_based_rules),
        "avg_precision": [],
        "avg_coverage": []
    }
    class_based_stats = {
        "count": len(class_based_rules),
        "avg_precision": [],
        "avg_coverage": []
    }
    
    for rule_result in results["rule_results"]:
        rollout_type = rule_result.get("rollout_type", "unknown")
        for class_key, class_res in rule_result.get("per_class_results", {}).items():
            precision = class_res.get("rule_precision", 0.0)
            coverage = class_res.get("rule_coverage", 0.0)
            if rollout_type == "instance_based":
                instance_based_stats["avg_precision"].append(precision)
                instance_based_stats["avg_coverage"].append(coverage)
            elif rollout_type == "class_based":
                class_based_stats["avg_precision"].append(precision)
                class_based_stats["avg_coverage"].append(coverage)
    
    # Summary
    results["summary"] = {
        "total_rules": len(all_unique_rules),
        "instance_based_rules": len(instance_based_rules),
        "class_based_rules": len(class_based_rules),
        "instance_based_stats": {
            "count": instance_based_stats["count"],
            "avg_precision": float(np.mean(instance_based_stats["avg_precision"])) if instance_based_stats["avg_precision"] else 0.0,
            "avg_coverage": float(np.mean(instance_based_stats["avg_coverage"])) if instance_based_stats["avg_coverage"] else 0.0,
        },
        "class_based_stats": {
            "count": class_based_stats["count"],
            "avg_precision": float(np.mean(class_based_stats["avg_precision"])) if class_based_stats["avg_precision"] else 0.0,
            "avg_coverage": float(np.mean(class_based_stats["avg_coverage"])) if class_based_stats["avg_coverage"] else 0.0,
        },
        "rules_satisfying_multiple_classes": len(rules_satisfying_both_classes),
        "rules_satisfying_both_classes": rules_satisfying_both_classes
    }
    
    logger.info(f"{'='*80}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total unique rules tested: {len(all_unique_rules)}")
    logger.info(f"  Instance-based rules: {len(instance_based_rules)}")
    logger.info(f"  Class-based rules: {len(class_based_rules)}")
    logger.info(f"  Overlapping (in both): {len(set(instance_based_rules) & set(class_based_rules))}")
    logger.info(f"")
    logger.info(f"Instance-based rule statistics:")
    logger.info(f"  Average precision: {results['summary']['instance_based_stats']['avg_precision']:.4f}")
    logger.info(f"  Average coverage: {results['summary']['instance_based_stats']['avg_coverage']:.4f}")
    logger.info(f"")
    logger.info(f"Class-based rule statistics:")
    logger.info(f"  Average precision: {results['summary']['class_based_stats']['avg_precision']:.4f}")
    logger.info(f"  Average coverage: {results['summary']['class_based_stats']['avg_coverage']:.4f}")
    logger.info(f"")
    logger.info(f"Rules satisfying multiple classes: {len(rules_satisfying_both_classes)}")
    
    if rules_satisfying_both_classes:
        logger.info(f"Rules that satisfy multiple classes:")
        for rule_info in rules_satisfying_both_classes:
            logger.info(f"  Rule {rule_info['rule_index'] + 1}: {rule_info['rule']}")
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
        
        fallback_used = class_data.get("fallback_used", False)
        n_candidates_total = class_data.get("n_candidates_total", 0)
        max_cand_prec = class_data.get("max_candidate_precision", 0.0)

        logger.info(f"Class {cls}:")
        logger.info(f"  Class samples: {n_class_samples}")
        if fallback_used:
            logger.info(
                f"  Selected rules: {n_selected}  "
                f"[FALLBACK — no rule met threshold {settings.get('precision_threshold', 0.9):.2f}; "
                f"showing top-{n_selected} by precision (max precision = {max_cand_prec:.4f})]"
            )
        else:
            logger.info(f"  Selected rules: {n_selected}")
        logger.info(f"  Class-union coverage: {union_cov:.4f} ({n_covered}/{n_class_samples} samples covered)")
        logger.info(f"  Class-union precision: {union_prec:.4f}")

        if n_selected > 0:
            logger.info(f"  Selected rule indices: {selected_indices}")
            for idx, rule_str in enumerate(selected_rules, 1):
                logger.info(f"    Rule {idx} (index {selected_indices[idx-1]}): {rule_str}")
        elif n_candidates_total == 0:
            logger.info(f"  No rules in unique_rules for this class (nothing to evaluate).")
        else:
            logger.info(
                f"  No rules selected (no rules met precision threshold "
                f"{settings.get('precision_threshold', 0.9):.2f}, max available precision = {max_cand_prec:.4f})"
            )
        logger.info("")

    logger.info("="*80)

    # ------------------------------------------------------------------
    # Post-hoc analysis enrichments (added 2026-05-18): feature_importance
    # and lift over base rate. Both are derived from rule_results and the
    # already-computed class distribution; no extra dataset passes needed.
    # ------------------------------------------------------------------

    # Feature importance: how often does each feature constrain a rule?
    # Counts both raw occurrences and the number of distinct rules that use
    # each feature. Useful for "what is the model paying attention to?".
    feature_total_occurrences = Counter()
    feature_distinct_rules = Counter()
    for rr in results.get("rule_results", []):
        seen_in_rule = set()
        for cond in rr.get("conditions", []):
            f = cond.get("feature")
            if not f:
                continue
            feature_total_occurrences[f] += 1
            seen_in_rule.add(f)
        for f in seen_in_rule:
            feature_distinct_rules[f] += 1
    n_total_rules = len(results.get("rule_results", []))
    results["feature_importance"] = {
        "n_rules": int(n_total_rules),
        "by_feature": [
            {
                "feature": f,
                "n_conditions": int(feature_total_occurrences[f]),
                "n_rules_using_it": int(feature_distinct_rules[f]),
                "fraction_of_rules": float(feature_distinct_rules[f] / n_total_rules) if n_total_rules else 0.0,
            }
            for f in sorted(feature_total_occurrences, key=lambda k: -feature_distinct_rules[k])
        ],
    }

    # Lift = rule_precision / class_base_rate. Lift > 1 means the rule
    # carries signal above the prior; lift ≈ 1 means it's just predicting
    # the class's base rate; lift < 1 is actively anti-predictive.
    total_samples = X_data.shape[0]
    class_base_rates = {
        int(c): (float(np.sum(y_data == c) / total_samples) if total_samples else 0.0)
        for c in unique_classes
    }
    results["class_base_rates"] = class_base_rates

    lifts_by_class: Dict[int, list] = {int(c): [] for c in unique_classes}
    for rr in results.get("rule_results", []):
        for class_key, class_res in rr.get("per_class_results", {}).items():
            cls_int = int(class_res.get("class", -1))
            base_rate = class_base_rates.get(cls_int, 0.0)
            prec = float(class_res.get("rule_precision", 0.0))
            lift = float(prec / base_rate) if base_rate > 0 else 0.0
            class_res["base_rate"] = base_rate
            class_res["lift"] = lift
            if class_res.get("n_satisfying_class_samples", 0) > 0:
                lifts_by_class.setdefault(cls_int, []).append(lift)

    # Per-class average lift, surfaced into per_class_results for at-a-glance use
    for cls_int, lifts in lifts_by_class.items():
        key = f"class_{cls_int}"
        if key in results.get("per_class_results", {}):
            results["per_class_results"][key]["avg_lift_over_base_rate"] = (
                float(np.mean(lifts)) if lifts else 0.0
            )
            results["per_class_results"][key]["n_rules_scored_for_lift"] = int(len(lifts))

    # ------------------------------------------------------------------
    # Optional markdown report — the TL;DR a human can actually read.
    # ------------------------------------------------------------------
    if report_md_path:
        try:
            _write_test_report_markdown(results, report_md_path)
            logger.info(f"✓ Test report (markdown) written to: {report_md_path}")
        except Exception as e:
            logger.warning(f"Could not write markdown report to {report_md_path}: {e}")

    return results


def _write_test_report_markdown(results: Dict, output_path: str) -> None:
    """Render a compact human-readable markdown summary of a test_rules_from_json
    result. Designed for skim-reading after a pipeline run; the .json next to it
    is the authoritative artifact."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    lines = []
    lines.append(f"# Test report — {results.get('dataset', 'unknown')}")
    lines.append("")
    lines.append(f"- **Algorithm**: {results.get('algorithm', 'unknown')}")
    lines.append(f"- **Model type**: {results.get('model_type', 'unknown')}")
    lines.append(f"- **Data split**: {results.get('data_type', 'unknown')}")
    lines.append(f"- **Total samples**: {results.get('n_samples', '?')}")
    lines.append(f"- **Total rules tested**: {len(results.get('rule_results', []))}")
    lines.append("")

    # Per-class summary table
    lines.append("## Per-class summary")
    lines.append("")
    lines.append("| Class | n samples | base rate | rules | inst. prec | inst. cov | class-union prec | class-union cov | avg lift |")
    lines.append("|-------|----------:|----------:|------:|-----------:|----------:|-----------------:|----------------:|---------:|")
    base_rates = results.get("class_base_rates", {})
    for class_key in sorted(results.get("per_class_results", {}).keys()):
        cd = results["per_class_results"][class_key]
        cls = cd.get("class", "?")
        n = cd.get("n_class_samples", 0) or cd.get("class_total_samples", 0)
        br = base_rates.get(int(cls), 0.0) if isinstance(cls, int) else 0.0
        n_rules = cd.get("unique_rules_count", len(cd.get("unique_rules", [])) if cd.get("unique_rules") else 0)
        ip = cd.get("instance_precision", cd.get("precision", 0.0))
        ic = cd.get("instance_coverage", cd.get("coverage", 0.0))
        cp = cd.get("class_precision", 0.0)
        cc = cd.get("class_coverage", 0.0)
        lift = cd.get("avg_lift_over_base_rate", 0.0)
        lines.append(
            f"| {cls} | {n} | {br:.3f} | {n_rules} | {ip:.4f} | {ic:.4f} | {cp:.4f} | {cc:.4f} | {lift:.2f}× |"
        )
    lines.append("")

    # Global explanations summary (with fallback markers)
    ge = results.get("global_explanations", {})
    if ge:
        lines.append("## Global explanations (per-class union of selected rules)")
        lines.append("")
        s = ge.get("settings", {})
        lines.append(f"_settings: precision_threshold={s.get('precision_threshold', '?')}, max_rules_per_class={s.get('max_rules_per_class', '?')}_")
        lines.append("")
        for class_key in sorted(ge.get("per_class", {}).keys()):
            d = ge["per_class"][class_key]
            cls = d.get("class", "?")
            n_sel = d.get("n_selected_rules", 0)
            cov = d.get("class_union_coverage", 0.0)
            prec = d.get("class_union_precision", 0.0)
            fb = d.get("fallback_used", False)
            max_cp = d.get("max_candidate_precision", 0.0)
            n_cands = d.get("n_candidates_total", 0)

            tag = " **[FALLBACK below threshold]**" if fb else ""
            lines.append(f"### Class {cls}{tag}")
            lines.append(f"- Selected rules: **{n_sel}** of {n_cands} candidates  (max candidate precision: {max_cp:.4f})")
            lines.append(f"- Class-union precision: **{prec:.4f}**")
            lines.append(f"- Class-union coverage: **{cov:.4f}**  ({d.get('n_covered_class_samples', 0)}/{d.get('n_class_samples', 0)} class samples)")
            if d.get("selected_rules"):
                lines.append("")
                lines.append("Selected rules:")
                for i, r in enumerate(d["selected_rules"], 1):
                    lines.append(f"  {i}. `{r}`")
            lines.append("")

    # Feature importance
    fi = results.get("feature_importance", {})
    by_feature = fi.get("by_feature", [])
    if by_feature:
        lines.append("## Feature importance across rules")
        lines.append("")
        lines.append("| Feature | rules using it | % of rules | total conditions |")
        lines.append("|---------|---------------:|-----------:|-----------------:|")
        for row in by_feature:
            lines.append(
                f"| {row['feature']} | {row['n_rules_using_it']} | "
                f"{100*row['fraction_of_rules']:.1f}% | {row['n_conditions']} |"
            )
        lines.append("")

    # Top-K rules by lift per class (the most informative rules)
    lines.append("## Top rules by lift (most informative per class)")
    lines.append("")
    rule_results = results.get("rule_results", [])
    classes = sorted({int(c.get("class", -1))
                      for rr in rule_results for c in rr.get("per_class_results", {}).values()})
    TOPK = 5
    for cls in classes:
        items = []
        for rr in rule_results:
            cr = rr.get("per_class_results", {}).get(f"class_{cls}")
            if not cr or cr.get("n_satisfying_class_samples", 0) <= 0:
                continue
            items.append({
                "rule": rr.get("rule", ""),
                "precision": cr.get("rule_precision", 0.0),
                "coverage": cr.get("rule_coverage", 0.0),
                "lift": cr.get("lift", 0.0),
                "n_satisfying_class": cr.get("n_satisfying_class_samples", 0),
            })
        items.sort(key=lambda r: (r["lift"], r["precision"]), reverse=True)
        if not items:
            continue
        lines.append(f"### Class {cls}")
        lines.append("")
        lines.append("| # | lift | prec | cov | n class samples in box | rule |")
        lines.append("|---|-----:|-----:|----:|----------------------:|------|")
        for i, it in enumerate(items[:TOPK], 1):
            lines.append(
                f"| {i} | {it['lift']:.2f}× | {it['precision']:.3f} | {it['coverage']:.3f} | "
                f"{it['n_satisfying_class']} | `{it['rule']}` |"
            )
        lines.append("")

    # Overlap summary
    oa = results.get("overlap_analysis", {}).get("summary", {})
    if oa:
        lines.append("## Cross-class rule overlap")
        lines.append("")
        lines.append(f"- Total unique rules: {oa.get('total_unique_rules', '?')}")
        lines.append(f"- Rules satisfying ≥2 classes: {oa.get('rules_with_overlaps', '?')}")
        lines.append(f"- Total overlap pairs: {oa.get('total_overlap_pairs', '?')}")
        lines.append("")

    # Missed samples summary
    ms = results.get("missed_samples_analysis", {}).get("summary", {})
    if ms:
        lines.append("## Missed-sample summary")
        lines.append("")
        cov = ms.get("overall_coverage_ratio", 0.0)
        lines.append(f"- Overall coverage: **{cov:.4f}**  ({ms.get('total_covered_samples', 0)}/{ms.get('total_samples', 0)})")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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
  
  # Test rules on full dataset (train + test combined)
  python single_agent/test_extracted_rules.py --rules_file path/to/extracted_rules_single_agent.json --dataset breast_cancer --use_full_dataset
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
        "--use_full_dataset",
        action="store_true",
        help="Test on full dataset (train + test combined) instead of just test or train data (default: False)"
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
        # CRITICAL FIX: Account for coverage_on_all_data flag to match test_rules_from_json logic
        metadata = rules_data.get("metadata", {})
        inference_eval_on_test_data = metadata.get("eval_on_test_data", False)
        inference_coverage_on_all_data = metadata.get("coverage_on_all_data", False)
        if inference_coverage_on_all_data:
            metrics_data_source = "full dataset (train + test)"
        elif inference_eval_on_test_data:
            metrics_data_source = "test data"
        else:
            metrics_data_source = "training data"
        
        # Test rules
        results = test_rules_from_json(
            rules_file=args.rules_file,
            dataset_name=args.dataset,
            use_test_data=not args.use_train_data if not args.use_full_dataset else False,
            use_full_dataset=args.use_full_dataset,
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
                
                logger.info(f"\n  {'='*60}")
                logger.info(f"  Class {target_class} - Inference Metrics Summary:")
                logger.info(f"  {'='*60}")
                
                # Instance-level metrics (averaged across instance-based anchors)
                if "instance_precision" in inference_data:
                    logger.info(f"  Instance-Level Metrics (averaged across instance-based anchors, from inference on {metrics_data_source}):")
                    logger.info(f"    Precision: {inference_data['instance_precision']:.4f}")
                    logger.info(f"    Coverage:  {inference_data['instance_coverage']:.4f}")
                
                # Class-based metrics (averaged across class-based anchors from centroid-based rollouts)
                if "class_level_precision" in inference_data or "class_based_precision" in inference_data:
                    class_based_prec = inference_data.get("class_level_precision", inference_data.get("class_based_precision", 0.0))
                    class_based_cov = inference_data.get("class_level_coverage", inference_data.get("class_based_coverage", 0.0))
                    if class_based_prec > 0.0 or class_based_cov > 0.0:
                        logger.info(f"  Class-Based Metrics (averaged across class-based anchors, from inference on {metrics_data_source}):")
                        logger.info(f"    Precision: {class_based_prec:.4f}")
                        logger.info(f"    Coverage:  {class_based_cov:.4f}")
                
                # Class union metrics (union of all anchors: instance-based + class-based)
                if "class_precision" in inference_data or "class_union_precision" in inference_data:
                    union_prec = inference_data.get("class_union_precision", inference_data.get("class_precision", 0.0))
                    union_cov = inference_data.get("class_union_coverage", inference_data.get("class_coverage", 0.0))
                    logger.info(f"  Class Union Metrics (union of all anchors, from inference on {metrics_data_source}):")
                    logger.info(f"    Precision: {union_prec:.4f}")
                    logger.info(f"    Coverage:  {union_cov:.4f}")
                
                logger.info(f"  {'='*60}")
        
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

