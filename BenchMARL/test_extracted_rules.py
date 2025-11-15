#!/usr/bin/env python3
"""
Test Extracted Rules Script

This script reads extracted rules from a JSON file and tests them against a dataset
to find all samples that satisfy each rule.

Usage:
    python test_extracted_rules.py --rules_file <path_to_extracted_rules.json> --dataset <dataset_name> [--use_test_data]

Example:
    python test_extracted_rules.py --rules_file masac_anchor_mlp__a4a120a7_25_11_14-23_12_28/inference/extracted_rules.json --dataset breast_cancer
"""

import json
import argparse
import numpy as np
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import logging

from tabular_datasets import TabularDatasetLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def test_rules_from_json(
    rules_file: str,
    dataset_name: str,
    use_test_data: bool = True,
    seed: int = 42
) -> Dict:
    """
    Test extracted rules against a dataset.
    
    Args:
        rules_file: Path to extracted_rules.json file
        dataset_name: Name of the dataset to test against
        use_test_data: If True, test on test data; if False, test on training data
        seed: Random seed for dataset loading
    
    Returns:
        Dictionary containing test results for each rule
    """
    logger.info("="*80)
    logger.info("TESTING EXTRACTED RULES")
    logger.info("="*80)
    logger.info(f"Rules file: {rules_file}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Data split: {'test' if use_test_data else 'training'}")
    logger.info("="*80)
    
    # Load rules from JSON
    logger.info("\nLoading rules from JSON file...")
    with open(rules_file, 'r') as f:
        rules_data = json.load(f)
    
    logger.info(f"✓ Loaded rules file with {len(rules_data.get('per_class_results', {}))} classes")
    
    # Load dataset
    logger.info(f"\nLoading dataset: {dataset_name}")
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
    
    logger.info(f"\n{'='*80}")
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
        "n_samples": X_data.shape[0],
        "n_features": X_data.shape[1],
        "classes": unique_classes,
        "rules_tested": len(all_unique_rules),
        "rule_results": []
    }
    
    rules_satisfying_both_classes = []
    
    for rule_idx, rule_str in enumerate(all_unique_rules):
        logger.info(f"\n{'='*80}")
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
                "precision": float(precision),
                "coverage": float(coverage),
                "satisfying_sample_indices": satisfying_class_indices
            }
            
            rule_result["per_class_results"][f"class_{target_class}"] = class_result
            
            logger.info(f"\n  Class {target_class}:")
            logger.info(f"    Samples satisfying: {n_satisfying_class}/{n_class_samples} ({100*coverage:.2f}% coverage)")
            logger.info(f"    Precision: {precision:.4f}")
            
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
            logger.info(f"\n  ✓ Rule satisfies {len(classes_satisfied)} classes: {classes_satisfied}")
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
    
    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total unique rules tested: {len(all_unique_rules)}")
    logger.info(f"Rules satisfying multiple classes: {len(rules_satisfying_both_classes)}")
    
    if rules_satisfying_both_classes:
        logger.info(f"\nRules that satisfy multiple classes:")
        for rule_info in rules_satisfying_both_classes:
            logger.info(f"  Rule {rule_info['rule_index'] + 1}: {rule_info['rule'][:80]}...")
            logger.info(f"    Classes satisfied: {rule_info['classes_satisfied']}")
            for class_val in rule_info['classes_satisfied']:
                class_key = f"class_{class_val}"
                class_res = rule_info['per_class_results'][class_key]
                logger.info(f"      Class {class_val}: precision={class_res['precision']:.4f}, coverage={class_res['coverage']:.4f}")
    else:
        logger.info(f"\nNo rules satisfy multiple classes.")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test extracted rules against a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test rules on test data (default)
  python test_extracted_rules.py --rules_file path/to/extracted_rules.json --dataset breast_cancer
  
  # Test rules on training data
  python test_extracted_rules.py --rules_file path/to/extracted_rules.json --dataset breast_cancer --use_train_data
  
  # Save results to file
  python test_extracted_rules.py --rules_file path/to/extracted_rules.json --dataset breast_cancer --output results.json
        """
    )
    
    parser.add_argument(
        "--rules_file",
        type=str,
        required=True,
        help="Path to extracted_rules.json file"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., breast_cancer, wine, iris)"
    )
    
    parser.add_argument(
        "--use_train_data",
        action="store_true",
        help="Test on training data instead of test data (default: test data)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path to save results (JSON format). If not specified, results are only printed."
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset loading (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Test rules
    results = test_rules_from_json(
        rules_file=args.rules_file,
        dataset_name=args.dataset,
        use_test_data=not args.use_train_data,
        seed=args.seed
    )
    
    # Save results if output path specified
    if args.output:
        logger.info(f"\n{'='*80}")
        logger.info(f"Saving results to: {args.output}")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("✓ Results saved successfully")
    
    # Print per-class summary
    logger.info(f"\n{'='*80}")
    logger.info("PER-CLASS SUMMARY")
    logger.info(f"{'='*80}")
    
    unique_classes = results.get("classes", [])
    for target_class in unique_classes:
        logger.info(f"\nClass {target_class}:")
        class_precisions = []
        class_coverages = []
        
        for rule_result in results["rule_results"]:
            class_key = f"class_{target_class}"
            if class_key in rule_result["per_class_results"]:
                class_res = rule_result["per_class_results"][class_key]
                class_precisions.append(class_res["precision"])
                class_coverages.append(class_res["coverage"])
        
        if class_precisions:
            logger.info(f"  Rules tested: {len(class_precisions)}")
            logger.info(f"  Mean precision: {np.mean(class_precisions):.4f} ± {np.std(class_precisions):.4f}")
            logger.info(f"  Mean coverage: {np.mean(class_coverages):.4f} ± {np.std(class_coverages):.4f}")
            logger.info(f"  Best precision: {np.max(class_precisions):.4f}")
            logger.info(f"  Best coverage: {np.max(class_coverages):.4f}")
    
    logger.info(f"\n{'='*80}")
    logger.info("Rule testing complete!")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()

