#!/usr/bin/env python3
"""
Plot Comparison: Single-Agent vs Multi-Agent

This script creates comparison plots between single-agent and multi-agent results.

Usage:
    python plot_comparison.py --single_agent_summary <path_to_single_agent_summary.json> --multi_agent_summary <path_to_multi_agent_summary.json> --dataset <dataset_name> --output_dir <output_directory>

Example:
    python plot_comparison.py --single_agent_summary comparison_results/wine_maddpg_20251203_181507/single_agent/summary.json --multi_agent_summary comparison_results/wine_maddpg_20251203_181507/multi_agent/summary.json --dataset wine --output_dir comparison_results/wine_maddpg_20251203_181507
"""

import json
import argparse
import numpy as np
import os
import sys
import re
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
from pathlib import Path
import logging

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib not available. Plotting will be skipped.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if HAS_PLOTTING:
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10


def _load_nashconv_metrics(experiment_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load NashConv metrics from training_history.json and evaluation_history.json.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary with training and evaluation NashConv metrics
    """
    nashconv_data = {
        "training": None,
        "evaluation": None,
        "available": False
    }
    
    if not experiment_dir:
        return nashconv_data
    
    experiment_path = Path(experiment_dir)
    
    # Load training history
    training_history_path = experiment_path / "training_history.json"
    if training_history_path.exists():
        try:
            with open(training_history_path, 'r') as f:
                training_history = json.load(f)
            
            # Extract NashConv metrics from training history
            training_nashconv = []
            for entry in training_history:
                nashconv_entry = {}
                for key, value in entry.items():
                    if key.startswith("training/nashconv") or key.startswith("training/exploitability"):
                        nashconv_entry[key] = value
                if nashconv_entry:
                    nashconv_entry["step"] = entry.get("step")
                    nashconv_entry["total_frames"] = entry.get("total_frames")
                    training_nashconv.append(nashconv_entry)
            
            if training_nashconv:
                nashconv_data["training"] = training_nashconv
                nashconv_data["available"] = True
        except Exception as e:
            logger.debug(f"Could not load training NashConv metrics: {e}")
    
    # Load evaluation history
    evaluation_history_path = experiment_path / "evaluation_history.json"
    if evaluation_history_path.exists():
        try:
            with open(evaluation_history_path, 'r') as f:
                evaluation_history = json.load(f)
            
            # Extract NashConv metrics from evaluation history
            evaluation_nashconv = []
            for entry in evaluation_history:
                nashconv_entry = {}
                for key, value in entry.items():
                    if key.startswith("evaluation/nashconv") or key.startswith("evaluation/exploitability"):
                        nashconv_entry[key] = value
                if nashconv_entry:
                    nashconv_entry["step"] = entry.get("step")
                    nashconv_entry["total_frames"] = entry.get("total_frames")
                    evaluation_nashconv.append(nashconv_entry)
            
            if evaluation_nashconv:
                nashconv_data["evaluation"] = evaluation_nashconv
                nashconv_data["available"] = True
        except Exception as e:
            logger.debug(f"Could not load evaluation NashConv metrics: {e}")
    
    return nashconv_data


def extract_feature_intervals_from_rule(rule_str: str) -> List[Tuple[str, float, float]]:
    """Extract feature names and intervals (lower, upper) from a rule string."""
    if rule_str == "any values (no tightened features)" or not rule_str:
        return []
    
    intervals = []
    seen_features = set()  # Track duplicate features
    
    # Pattern to match: "feature_name ∈ [lower, upper]"
    pattern = r'(.+?)\s*∈\s*\[([-\d.]+),\s*([-\d.]+)\]'
    
    # Split by " and " to get individual conditions
    conditions = rule_str.split(" and ")
    
    for condition in conditions:
        condition = condition.strip()
        if not condition:
            continue
        match = re.search(pattern, condition)
        if match:
            feature_name = match.group(1).strip()
            try:
                lower = float(match.group(2))
                upper = float(match.group(3))
                
                # Validate interval: ensure lower <= upper
                if lower > upper:
                    logger.warning(
                        f"Invalid interval in rule '{rule_str[:100]}...': "
                        f"feature '{feature_name}' has lower={lower} > upper={upper}. Swapping."
                    )
                    lower, upper = upper, lower
                
                # Check for duplicate features (keep first occurrence)
                if feature_name in seen_features:
                    logger.debug(
                        f"Duplicate feature '{feature_name}' in rule '{rule_str[:100]}...'. "
                        f"Keeping first occurrence."
                    )
                    continue
                
                seen_features.add(feature_name)
                intervals.append((feature_name, lower, upper))
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Failed to parse interval from condition '{condition}' in rule '{rule_str[:100]}...': {e}"
                )
                continue
    
    return intervals


def calculate_equilibrium_metrics(per_class_summary: Dict, precision_target: float = 0.95, coverage_target: float = 0.1) -> Dict:
    """
    Calculate equilibrium metrics from per_class_summary.
    
    Equilibrium means all classes meet both precision and coverage targets.
    This is a fair comparison metric between single-agent and multi-agent:
    - Single-agent: one agent per class, equilibrium = all classes meet targets
    - Multi-agent: multiple agents per class, equilibrium = all classes meet targets (union metrics)
    
    Returns:
        Dict with equilibrium metrics:
        - equilibrium_reached: bool
        - equilibrium_fraction: float (0.0 to 1.0)
        - classes_meeting_targets: int
        - total_classes: int
        - per_class_status: Dict[class_id, {"meets_targets": bool, "precision_gap": float, "coverage_gap": float}]
    """
    if not per_class_summary:
        return {
            "equilibrium_reached": False,
            "equilibrium_fraction": 0.0,
            "classes_meeting_targets": 0,
            "total_classes": 0,
            "per_class_status": {}
        }
    
    classes_meeting_targets = 0
    per_class_status = {}
    
    for class_key, class_data in per_class_summary.items():
        # Use class-level metrics (union) if available, otherwise instance-level
        # CRITICAL: Prefer class-level metrics for equilibrium - they measure the union explanation
        # Instance-level metrics measure individual anchors, which is different
        class_precision_raw = class_data.get("class_precision")
        class_coverage_raw = class_data.get("class_coverage")
        instance_precision_raw = class_data.get("instance_precision", 0.0)
        instance_coverage_raw = class_data.get("instance_coverage", 0.0)
        
        # Handle None values: convert to 0.0 if None, otherwise use the value
        # If class-level metrics exist (even if 0.0), use them
        # Only fallback to instance-level if class-level key doesn't exist (for backward compatibility)
        if class_precision_raw is not None:
            precision = float(class_precision_raw)
        else:
            # Class-level precision missing: fallback to instance-level (for backward compatibility)
            precision = float(instance_precision_raw) if instance_precision_raw is not None else 0.0
            # Note: This fallback is for backward compatibility with old results or baseline methods
            # For Dynamic Anchors, class-level metrics should always be present
        
        if class_coverage_raw is not None:
            coverage = float(class_coverage_raw)
        else:
            # Class-level coverage missing: fallback to instance-level (for backward compatibility)
            coverage = float(instance_coverage_raw) if instance_coverage_raw is not None else 0.0
            # Note: This fallback is for backward compatibility with old results or baseline methods
            # For Dynamic Anchors, class-level metrics should always be present
        
        # Validate that we have numeric values
        if not isinstance(precision, (int, float)) or not isinstance(coverage, (int, float)):
            logger.warning(
                f"Equilibrium calculation: Invalid precision/coverage values for class {class_key}. "
                f"precision={precision}, coverage={coverage}. Using 0.0"
            )
            precision = 0.0
            coverage = 0.0
        
        meets_precision = precision >= precision_target
        meets_coverage = coverage >= coverage_target
        meets_both = meets_precision and meets_coverage
        
        if meets_both:
            classes_meeting_targets += 1
        
        class_id = class_data.get("class", -1)
        per_class_status[class_id] = {
            "meets_targets": meets_both,
            "precision": precision,
            "coverage": coverage,
            "precision_gap": max(0.0, precision_target - precision),
            "coverage_gap": max(0.0, coverage_target - coverage)
        }
    
    total_classes = len(per_class_summary)
    equilibrium_fraction = classes_meeting_targets / total_classes if total_classes > 0 else 0.0
    equilibrium_reached = (classes_meeting_targets == total_classes) and (total_classes > 0)
    
    return {
        "equilibrium_reached": equilibrium_reached,
        "equilibrium_fraction": equilibrium_fraction,
        "classes_meeting_targets": classes_meeting_targets,
        "total_classes": total_classes,
        "per_class_status": per_class_status,
        "precision_target": precision_target,
        "coverage_target": coverage_target
    }


def sort_rules_for_plotting(class_data: Dict, rules: List[str]) -> List[str]:
    """
    Sort rules for consistent plotting order.
    Priority: 1) Use ranked_unique_rules if available (already sorted)
              2) Sort by metrics from ranked_rules if available
              3) Sort by rule length (simpler first), then alphabetically
    """
    # If rules are already ranked, use them as-is
    ranked_unique = class_data.get("ranked_unique_rules", [])
    if ranked_unique and set(rules) == set(ranked_unique):
        return ranked_unique
    
    # Try to get metrics from ranked_rules
    ranked_rules_with_metrics = class_data.get("ranked_rules", [])
    if ranked_rules_with_metrics:
        # Create a mapping from rule string to metrics
        rule_to_metrics = {r["rule"]: r for r in ranked_rules_with_metrics if isinstance(r, dict) and "rule" in r}
        
        # Sort by combined_score (descending), then precision, then coverage
        def sort_key(rule_str: str) -> tuple:
            if rule_str in rule_to_metrics:
                metrics = rule_to_metrics[rule_str]
                return (
                    -metrics.get("combined_score", 0.0),  # Negative for descending
                    -metrics.get("rule_precision", 0.0),
                    -metrics.get("rule_coverage", 0.0)
                )
            # Fallback: sort by rule length (shorter/simpler first), then alphabetically
            return (len(extract_feature_intervals_from_rule(rule_str)), rule_str)
        
        return sorted(rules, key=sort_key)
    
    # Fallback: sort by rule length (simpler rules first), then alphabetically
    def sort_key(rule_str: str) -> tuple:
        intervals = extract_feature_intervals_from_rule(rule_str)
        return (len(intervals), rule_str)  # Shorter rules first, then alphabetical
    
    return sorted(rules, key=sort_key)


def load_summary_file(summary_file: str) -> Optional[Dict]:
    """Load summary JSON file."""
    if not os.path.exists(summary_file):
        logger.warning(f"Summary file does not exist: {summary_file}")
        return None
    
    try:
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        # Handle nested structure: {"summary": {...}, "test_results": {...}}
        if "summary" in data:
            summary = data["summary"]
            # Check if summary is actually empty or None
            if not summary:
                logger.warning(f"Summary file {summary_file} contains empty 'summary' field")
                return None
            return summary
        # If already flat structure, return as-is
        if not data:
            logger.warning(f"Summary file {summary_file} is empty")
            return None
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {summary_file}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load summary file {summary_file}: {e}")
        return None


def extract_baseline_metrics(baseline_results_file: str) -> Optional[Dict]:
    """
    Extract baseline metrics from baseline results JSON in a format compatible with comparison plots.
    
    Args:
        baseline_results_file: Path to baseline_results JSON file
    
    Returns:
        Dictionary with baseline metrics in summary format, or None if not available
    """
    try:
        with open(baseline_results_file, 'r') as f:
            baseline_data = json.load(f)
        
        # Extract static anchors results
        methods = baseline_data.get("methods", {})
        static_anchors = methods.get("static_anchors", {})
        
        if "error" in static_anchors or not static_anchors:
            logger.warning("No static anchors results in baseline data")
            return None
        
        per_class_results = static_anchors.get("per_class_results", {})
        if not per_class_results:
            logger.warning("No per-class results in static anchors")
            return None
        
        # Convert to summary format
        per_class_summary = {}
        for class_key, class_data in per_class_results.items():
            # Extract instance-level and class-level metrics
            instance_precision = class_data.get("instance_precision", class_data.get("avg_precision", 0.0))
            instance_coverage = class_data.get("instance_coverage", class_data.get("avg_coverage", 0.0))
            class_precision = class_data.get("class_precision", 0.0)
            class_coverage = class_data.get("class_coverage", 0.0)
            
            per_class_summary[class_key] = {
                "class": int(class_key),
                "instance_precision": float(instance_precision),
                "instance_coverage": float(instance_coverage),
                "class_precision": float(class_precision),
                "class_coverage": float(class_coverage),
                # Legacy fields
                "precision": float(instance_precision),
                "coverage": float(instance_coverage),
            }
        
        # Calculate overall stats
        instance_precisions = [v["instance_precision"] for v in per_class_summary.values()]
        instance_coverages = [v["instance_coverage"] for v in per_class_summary.values()]
        class_precisions = [v["class_precision"] for v in per_class_summary.values() if v["class_precision"] > 0]
        class_coverages = [v["class_coverage"] for v in per_class_summary.values() if v["class_coverage"] > 0]
        
        overall_stats = {
            "mean_instance_precision": float(np.mean(instance_precisions)) if instance_precisions else 0.0,
            "mean_instance_coverage": float(np.mean(instance_coverages)) if instance_coverages else 0.0,
            "mean_class_precision": float(np.mean(class_precisions)) if class_precisions else 0.0,
            "mean_class_coverage": float(np.mean(class_coverages)) if class_coverages else 0.0,
            "std_instance_precision": float(np.std(instance_precisions)) if len(instance_precisions) > 1 else 0.0,
            "std_instance_coverage": float(np.std(instance_coverages)) if len(instance_coverages) > 1 else 0.0,
            "std_class_precision": float(np.std(class_precisions)) if len(class_precisions) > 1 else 0.0,
            "std_class_coverage": float(np.std(class_coverages)) if len(class_coverages) > 1 else 0.0,
            # Legacy fields
            "mean_precision": float(np.mean(instance_precisions)) if instance_precisions else 0.0,
            "mean_coverage": float(np.mean(instance_coverages)) if instance_coverages else 0.0,
        }
        
        return {
            "per_class_summary": per_class_summary,
            "overall_stats": overall_stats,
            "method": "Static Anchors (Baseline)"
        }
    except Exception as e:
        logger.error(f"Error extracting baseline metrics: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_precision_coverage_comparison(
    single_agent_summary: Optional[Dict],
    multi_agent_summary: Optional[Dict],
    output_dir: str,
    dataset_name: str = "",
    baseline_summary: Optional[Dict] = None
):
    """Plot precision and coverage comparison using grouped bar charts for clarity."""
    if not HAS_PLOTTING:
        return
    
    # Get data
    single_per_class = single_agent_summary.get("per_class_summary", {}) if single_agent_summary else {}
    multi_per_class = multi_agent_summary.get("per_class_summary", {}) if multi_agent_summary else {}
    baseline_per_class = baseline_summary.get("per_class_summary", {}) if baseline_summary else {}
    
    # Debug: Log structure
    if single_per_class:
        sample_key = list(single_per_class.keys())[0]
        sample_data = single_per_class[sample_key]
        logger.info(f"Single-agent sample class data keys: {list(sample_data.keys())}")
        logger.info(f"Single-agent sample class data: class={sample_data.get('class')}, class_precision={sample_data.get('class_precision')}, class_coverage={sample_data.get('class_coverage')}")
    
    if multi_per_class:
        sample_key = list(multi_per_class.keys())[0]
        sample_data = multi_per_class[sample_key]
        logger.info(f"Multi-agent sample class data keys: {list(sample_data.keys())}")
        logger.info(f"Multi-agent sample class data: class={sample_data.get('class')}, class_precision={sample_data.get('class_precision')}, class_coverage={sample_data.get('class_coverage')}")
    
    if not single_per_class and not multi_per_class:
        logger.warning("No per_class_summary data found in summaries. Plots will be empty.")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f"{dataset_name.upper()}: Precision vs Coverage Comparison (No Data)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_coverage_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Get all unique classes and sort them
    all_classes = set()
    for class_data in single_per_class.values():
        all_classes.add(class_data.get("class"))
    for class_data in multi_per_class.values():
        all_classes.add(class_data.get("class"))
    for class_data in baseline_per_class.values():
        all_classes.add(class_data.get("class"))
    
    if not all_classes:
        logger.warning("No classes found in summaries. Plots will be empty.")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No class data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f"{dataset_name.upper()}: Precision vs Coverage Comparison (No Data)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_coverage_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    classes = sorted(all_classes)
    
    # Extract data for each class
    # Instance-level metrics (all three methods)
    single_instance_precisions = []
    single_instance_coverages = []
    multi_instance_precisions = []
    multi_instance_coverages = []
    baseline_instance_precisions = []
    baseline_instance_coverages = []
    
    # Class-level metrics (only single-agent and multi-agent, NOT baseline)
    single_class_precisions = []
    single_class_coverages = []
    multi_class_precisions = []
    multi_class_coverages = []
    
    # Class-level metrics (centroid-based rollouts, only single-agent and multi-agent, NOT baseline)
    single_class_level_precisions = []
    single_class_level_coverages = []
    multi_class_level_precisions = []
    multi_class_level_coverages = []
    # Legacy names for backward compatibility
    single_class_based_precisions = []
    single_class_based_coverages = []
    multi_class_based_precisions = []
    multi_class_based_coverages = []
    
    logger.info(f"Extracting data for classes: {classes}")
    
    for cls in classes:
        # Find single-agent data for this class
        single_data = None
        single_class_based_data = None
        if single_per_class:
            for class_key, class_data in single_per_class.items():
                # Skip class_based entries when looking for main class data
                if class_key.endswith("_class_based"):
                    continue
                if class_data.get("class") == cls:
                    single_data = class_data
                    break
            
            # Also look for separate class-based entry (format: class_{cls}_class_based) as fallback
            class_based_key = f"class_{cls}_class_based"
            if class_based_key in single_per_class:
                single_class_based_data = single_per_class[class_based_key]
                logger.debug(f"Found separate class-based entry for single-agent class {cls}: {class_based_key}")
        
        if single_data:
            # Instance-level
            single_inst_prec = single_data.get("instance_precision", 0.0)
            single_inst_cov = single_data.get("instance_coverage", 0.0)
            single_instance_precisions.append(single_inst_prec)
            single_instance_coverages.append(single_inst_cov)
            # Class Union (union of class-based anchors only)
            single_union_prec = single_data.get("class_union_precision", single_data.get("class_precision", 0.0))
            single_union_cov = single_data.get("class_union_coverage", single_data.get("class_coverage", 0.0))
            single_class_precisions.append(single_union_prec)
            single_class_coverages.append(single_union_cov)
            
            # Class-level (centroid-based rollouts) - check main class entry first (current format)
            single_cl_prec = single_data.get("class_level_precision", single_data.get("class_based_precision", 0.0))
            single_cl_cov = single_data.get("class_level_coverage", single_data.get("class_based_coverage", 0.0))
            
            # If not found in main class data, check separate class_based entry (fallback format)
            if (single_cl_prec == 0.0 and single_cl_cov == 0.0) and single_class_based_data:
                single_cl_prec = single_class_based_data.get("precision", single_class_based_data.get("class_level_precision", single_class_based_data.get("class_based_precision", 0.0)))
                single_cl_cov = single_class_based_data.get("coverage", single_class_based_data.get("class_level_coverage", single_class_based_data.get("class_based_coverage", 0.0)))
                logger.debug(f"  Using class-based data from separate entry: prec={single_cl_prec:.3f}, cov={single_cl_cov:.3f}")
            
            # Also check class_based_results nested dict (legacy format)
            if (single_cl_prec == 0.0 and single_cl_cov == 0.0) and "class_based_results" in single_data:
                class_based_results = single_data.get("class_based_results", {})
                if isinstance(class_based_results, dict):
                    # Handle both single dict and nested dict formats
                    if "precision" in class_based_results:
                        single_cl_prec = class_based_results.get("precision", class_based_results.get("class_level_precision", 0.0))
                        single_cl_cov = class_based_results.get("coverage", class_based_results.get("class_level_coverage", 0.0))
                    else:
                        # Nested dict format (multi-agent style)
                        for agent_result in class_based_results.values():
                            if isinstance(agent_result, dict) and "instance_precision" in agent_result:
                                single_cl_prec = agent_result.get("instance_precision", agent_result.get("precision", 0.0))
                                single_cl_cov = agent_result.get("instance_coverage", agent_result.get("coverage", 0.0))
                                break
                    logger.debug(f"  Using class-based data from nested class_based_results: prec={single_cl_prec:.3f}, cov={single_cl_cov:.3f}")
            
            single_class_level_precisions.append(single_cl_prec)
            single_class_level_coverages.append(single_cl_cov)
            # Legacy names
            single_class_based_precisions.append(single_cl_prec)
            single_class_based_coverages.append(single_cl_cov)
            logger.info(f"Single-agent C{cls}: inst_prec={single_inst_prec:.3f}, inst_cov={single_inst_cov:.3f}, class_union_prec={single_union_prec:.3f}, class_union_cov={single_union_cov:.3f}, class_based_prec={single_cl_prec:.3f}, class_based_cov={single_cl_cov:.3f}")
            # Debug: Check if class-based keys exist
            if single_cl_prec > 0.0 or single_cl_cov > 0.0:
                logger.debug(f"  Single-agent C{cls}: Found class-based data (prec={single_cl_prec:.3f}, cov={single_cl_cov:.3f})")
            elif "class_level_precision" in single_data or "class_based_precision" in single_data or single_class_based_data:
                logger.debug(f"  Single-agent C{cls}: Class-based keys exist but values are 0.0")
            else:
                logger.debug(f"  Single-agent C{cls}: No class-based keys found (keys: {list(single_data.keys())})")
        else:
            logger.warning(f"Single-agent: No data found for class {cls}")
            single_instance_precisions.append(0.0)
            single_instance_coverages.append(0.0)
            single_class_precisions.append(0.0)
            single_class_coverages.append(0.0)
            single_class_based_precisions.append(0.0)
            single_class_based_coverages.append(0.0)
            # Also append to class_level lists for consistency
            single_class_level_precisions.append(0.0)
            single_class_level_coverages.append(0.0)
        
        # Find multi-agent data for this class
        multi_data = None
        if multi_per_class:
            for class_key, class_data in multi_per_class.items():
                # Skip class_based entries when looking for main class data
                if class_key.endswith("_class_based") or class_data.get("rollout_type") == "class_based":
                    continue
                if class_data.get("class") == cls:
                    multi_data = class_data
                    break
        if multi_data:
            # Instance-level
            multi_inst_prec = multi_data.get("instance_precision", 0.0)
            multi_inst_cov = multi_data.get("instance_coverage", 0.0)
            multi_instance_precisions.append(multi_inst_prec)
            multi_instance_coverages.append(multi_inst_cov)
            # Class Union (union of class-based anchors only)
            multi_union_prec = multi_data.get("class_union_precision", multi_data.get("class_precision", 0.0))
            multi_union_cov = multi_data.get("class_union_coverage", multi_data.get("class_coverage", 0.0))
            multi_class_precisions.append(multi_union_prec)
            multi_class_coverages.append(multi_union_cov)
            # Class-level (centroid-based rollouts)
            multi_cl_prec = multi_data.get("class_level_precision", multi_data.get("class_based_precision", 0.0))
            multi_cl_cov = multi_data.get("class_level_coverage", multi_data.get("class_based_coverage", 0.0))
            multi_class_level_precisions.append(multi_cl_prec)
            multi_class_level_coverages.append(multi_cl_cov)
            # Legacy names
            multi_class_based_precisions.append(multi_cl_prec)
            multi_class_based_coverages.append(multi_cl_cov)
            logger.info(f"Multi-agent C{cls}: inst_prec={multi_inst_prec:.3f}, inst_cov={multi_inst_cov:.3f}, class_union_prec={multi_union_prec:.3f}, class_union_cov={multi_union_cov:.3f}, class_based_prec={multi_cl_prec:.3f}, class_based_cov={multi_cl_cov:.3f}")
        else:
            logger.warning(f"Multi-agent: No data found for class {cls}")
            multi_instance_precisions.append(0.0)
            multi_instance_coverages.append(0.0)
            multi_class_precisions.append(0.0)
            multi_class_coverages.append(0.0)
            multi_class_based_precisions.append(0.0)
            multi_class_based_coverages.append(0.0)
        
        # Find baseline data for this class (INSTANCE-LEVEL ONLY)
        baseline_data = None
        if baseline_per_class:
            for class_key, class_data in baseline_per_class.items():
                if class_data.get("class") == cls:
                    baseline_data = class_data
                    break
        if baseline_data:
            # Instance-level only (baseline class-level is union of 20 anchors - not fair comparison)
            baseline_inst_prec = baseline_data.get("instance_precision", baseline_data.get("avg_precision", 0.0))
            baseline_inst_cov = baseline_data.get("instance_coverage", baseline_data.get("avg_coverage", 0.0))
            baseline_instance_precisions.append(baseline_inst_prec)
            baseline_instance_coverages.append(baseline_inst_cov)
            logger.info(f"Baseline C{cls}: inst_prec={baseline_inst_prec:.3f}, inst_cov={baseline_inst_cov:.3f} (class-level excluded - union not fair)")
        else:
            baseline_instance_precisions.append(0.0)
            baseline_instance_coverages.append(0.0)
    
    # Format title with dataset name
    title_prefix = f"{dataset_name.upper()}" if dataset_name else ""
    
    # Check if we have class-level metrics (centroid-based rollouts)
    # Show if we have summary data and extracted values (even if some are 0.0)
    has_class_level_single = single_agent_summary is not None and len(single_class_level_precisions) == len(classes)
    has_class_level_multi = multi_agent_summary is not None and len(multi_class_level_precisions) == len(classes)
    has_class_level = has_class_level_single or has_class_level_multi
    
    # Debug logging
    if single_agent_summary:
        logger.info(f"Single-agent class-based data: {len(single_class_level_precisions)} precisions, {len(single_class_level_coverages)} coverages for {len(classes)} classes")
        logger.info(f"  Values: prec={single_class_level_precisions}, cov={single_class_level_coverages}")
        logger.info(f"  Will show: {has_class_level_single}")
    
    # Create subplots: instance-based (row 1), class-based (row 2), union-based (row 3)
    # Reordered as requested: instance -> class-based -> union
    if has_class_level:
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        # Row 1: Instance-based, Row 2: Class-based, Row 3: Union
        ax1_inst, ax2_inst, ax1_cl, ax2_cl, ax1_union, ax2_union = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        # Row 1: Instance-based, Row 2: Union (no class-based available)
        ax1_inst, ax2_inst, ax1_union, ax2_union = axes.flatten()
    
    x = np.arange(len(classes))
    
    # Top row: Instance-level comparison (all three methods)
    n_methods_inst = sum([bool(single_agent_summary), bool(multi_agent_summary), bool(baseline_summary)])
    width_inst = 0.8 / n_methods_inst if n_methods_inst > 0 else 0.35
    offset_inst = -0.4 + width_inst/2
    
    # Bottom row: Class-level comparison (only single-agent and multi-agent)
    n_methods_cls = sum([bool(single_agent_summary), bool(multi_agent_summary)])
    width_cls = 0.8 / n_methods_cls if n_methods_cls > 0 else 0.35
    offset_cls = -0.4 + width_cls/2
    
    # Top-left: Instance-level Precision
    bar_idx = 0
    if single_agent_summary:
        ax1_inst.bar(x + offset_inst + bar_idx*width_inst, single_instance_precisions, width_inst, 
                    label='Single-Agent', alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
        for i, val in enumerate(single_instance_precisions):
            if val > 0.01:
                ax1_inst.text(x[i] + offset_inst + bar_idx*width_inst + width_inst/2, val + 0.02,
                             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        bar_idx += 1
    if multi_agent_summary:
        ax1_inst.bar(x + offset_inst + bar_idx*width_inst, multi_instance_precisions, width_inst, 
                    label='Multi-Agent', alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
        for i, val in enumerate(multi_instance_precisions):
            if val > 0.01:
                ax1_inst.text(x[i] + offset_inst + bar_idx*width_inst + width_inst/2, val + 0.02,
                             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        bar_idx += 1
    if baseline_summary:
        ax1_inst.bar(x + offset_inst + bar_idx*width_inst, baseline_instance_precisions, width_inst, 
                    label='Baseline (Static Anchors)', alpha=0.8, color='green', edgecolor='black', linewidth=1.5)
        for i, val in enumerate(baseline_instance_precisions):
            if val > 0.01:
                ax1_inst.text(x[i] + offset_inst + bar_idx*width_inst + width_inst/2, val + 0.02,
                             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1_inst.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1_inst.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1_inst.set_title(f'{title_prefix}: Instance-Level Precision (All Methods)', fontsize=13, fontweight='bold')
    ax1_inst.set_xticks(x)
    ax1_inst.set_xticklabels([f'C{cls}' for cls in classes])
    ax1_inst.set_ylim([0, 1.1])
    ax1_inst.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1_inst.legend(fontsize=10, framealpha=0.9)
    
    # Top-right: Instance-level Coverage
    bar_idx = 0
    if single_agent_summary:
        ax2_inst.bar(x + offset_inst + bar_idx*width_inst, single_instance_coverages, width_inst, 
                    label='Single-Agent', alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
        for i, val in enumerate(single_instance_coverages):
            if val > 0.01:
                ax2_inst.text(x[i] + offset_inst + bar_idx*width_inst + width_inst/2, val + 0.02,
                             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        bar_idx += 1
    if multi_agent_summary:
        ax2_inst.bar(x + offset_inst + bar_idx*width_inst, multi_instance_coverages, width_inst, 
                    label='Multi-Agent', alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
        for i, val in enumerate(multi_instance_coverages):
            if val > 0.01:
                ax2_inst.text(x[i] + offset_inst + bar_idx*width_inst + width_inst/2, val + 0.02,
                             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        bar_idx += 1
    if baseline_summary:
        ax2_inst.bar(x + offset_inst + bar_idx*width_inst, baseline_instance_coverages, width_inst, 
                    label='Baseline (Static Anchors)', alpha=0.8, color='green', edgecolor='black', linewidth=1.5)
        for i, val in enumerate(baseline_instance_coverages):
            if val > 0.01:
                ax2_inst.text(x[i] + offset_inst + bar_idx*width_inst + width_inst/2, val + 0.02,
                             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2_inst.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2_inst.set_ylabel('Coverage', fontsize=12, fontweight='bold')
    ax2_inst.set_title(f'{title_prefix}: Instance-Level Coverage (All Methods)', fontsize=13, fontweight='bold')
    ax2_inst.set_xticks(x)
    ax2_inst.set_xticklabels([f'C{cls}' for cls in classes])
    ax2_inst.set_ylim([0, 1.1])
    ax2_inst.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2_inst.legend(fontsize=10, framealpha=0.9)
    
    # Row 2: Class-Based Precision and Coverage (centroid-based rollouts, if available)
    # This is now row 2 (middle row) after reordering
    if has_class_level:
        bar_idx = 0
        if single_agent_summary and has_class_level_single:
            ax1_cl.bar(x + offset_cls + bar_idx*width_cls, single_class_level_precisions, width_cls, 
                      label='Single-Agent', alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
            for i, val in enumerate(single_class_level_precisions):
                if val > 0.01:
                    ax1_cl.text(x[i] + offset_cls + bar_idx*width_cls + width_cls/2, val + 0.02,
                               f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            bar_idx += 1
        if multi_agent_summary and has_class_level_multi:
            ax1_cl.bar(x + offset_cls + bar_idx*width_cls, multi_class_level_precisions, width_cls, 
                      label='Multi-Agent', alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
            for i, val in enumerate(multi_class_level_precisions):
                if val > 0.01:
                    ax1_cl.text(x[i] + offset_cls + bar_idx*width_cls + width_cls/2, val + 0.02,
                               f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1_cl.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax1_cl.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax1_cl.set_title(f'{title_prefix}: Class-Based Precision (Centroid-Based Rollouts)', fontsize=13, fontweight='bold')
        ax1_cl.set_xticks(x)
        ax1_cl.set_xticklabels([f'C{cls}' for cls in classes])
        ax1_cl.set_ylim([0, 1.1])
        ax1_cl.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1_cl.legend(fontsize=10, framealpha=0.9)
        
        # Row 2-right: Class-Based Coverage (centroid-based rollouts)
        bar_idx = 0
        if single_agent_summary and has_class_level_single:
            ax2_cl.bar(x + offset_cls + bar_idx*width_cls, single_class_level_coverages, width_cls, 
                      label='Single-Agent', alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
            for i, val in enumerate(single_class_level_coverages):
                if val > 0.01:
                    ax2_cl.text(x[i] + offset_cls + bar_idx*width_cls + width_cls/2, val + 0.02,
                               f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            bar_idx += 1
        if multi_agent_summary and has_class_level_multi:
            ax2_cl.bar(x + offset_cls + bar_idx*width_cls, multi_class_level_coverages, width_cls, 
                      label='Multi-Agent', alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
            for i, val in enumerate(multi_class_level_coverages):
                if val > 0.01:
                    ax2_cl.text(x[i] + offset_cls + bar_idx*width_cls + width_cls/2, val + 0.02,
                               f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2_cl.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax2_cl.set_ylabel('Coverage', fontsize=12, fontweight='bold')
        ax2_cl.set_title(f'{title_prefix}: Class-Based Coverage (Centroid-Based Rollouts)', fontsize=13, fontweight='bold')
        ax2_cl.set_xticks(x)
        ax2_cl.set_xticklabels([f'C{cls}' for cls in classes])
        ax2_cl.set_ylim([0, 1.1])
        ax2_cl.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2_cl.legend(fontsize=10, framealpha=0.9)
    
    # Row 3: Class Union Precision and Coverage (union of instance-based anchors, only single-agent and multi-agent)
    # This is now row 3 (bottom row) after reordering
    bar_idx = 0
    if single_agent_summary:
        ax1_union.bar(x + offset_cls + bar_idx*width_cls, single_class_precisions, width_cls, 
                   label='Single-Agent', alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
        for i, val in enumerate(single_class_precisions):
            if val > 0.01:
                ax1_union.text(x[i] + offset_cls + bar_idx*width_cls + width_cls/2, val + 0.02,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        bar_idx += 1
    if multi_agent_summary:
        ax1_union.bar(x + offset_cls + bar_idx*width_cls, multi_class_precisions, width_cls, 
                   label='Multi-Agent', alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
        for i, val in enumerate(multi_class_precisions):
            if val > 0.01:
                ax1_union.text(x[i] + offset_cls + bar_idx*width_cls + width_cls/2, val + 0.02,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1_union.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1_union.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1_union.set_title(f'{title_prefix}: Class Union Precision (Union of Class-Based Anchors Only)', fontsize=13, fontweight='bold')
    ax1_union.set_xticks(x)
    ax1_union.set_xticklabels([f'C{cls}' for cls in classes])
    ax1_union.set_ylim([0, 1.1])
    ax1_union.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1_union.legend(fontsize=10, framealpha=0.9)
    
    # Row 3-right: Class Union Coverage (union of instance-based anchors, only single-agent and multi-agent)
    bar_idx = 0
    if single_agent_summary:
        ax2_union.bar(x + offset_cls + bar_idx*width_cls, single_class_coverages, width_cls, 
                   label='Single-Agent', alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
        for i, val in enumerate(single_class_coverages):
            if val > 0.01:
                ax2_union.text(x[i] + offset_cls + bar_idx*width_cls + width_cls/2, val + 0.02,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        bar_idx += 1
    if multi_agent_summary:
        ax2_union.bar(x + offset_cls + bar_idx*width_cls, multi_class_coverages, width_cls, 
                   label='Multi-Agent', alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
        for i, val in enumerate(multi_class_coverages):
            if val > 0.01:
                ax2_union.text(x[i] + offset_cls + bar_idx*width_cls + width_cls/2, val + 0.02,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2_union.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2_union.set_ylabel('Coverage', fontsize=12, fontweight='bold')
    ax2_union.set_title(f'{title_prefix}: Class Union Coverage (Union of Class-Based Anchors Only)', fontsize=13, fontweight='bold')
    ax2_union.set_xticks(x)
    ax2_union.set_xticklabels([f'C{cls}' for cls in classes])
    ax2_union.set_ylim([0, 1.1])
    ax2_union.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2_union.legend(fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_coverage_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved precision-coverage comparison plot to {output_dir}/precision_coverage_comparison.png")


def plot_feature_importance_subplot(ax, summary: Optional[Dict], title: str, top_n: int = 10):
    """
    Plot feature importance on a given axis with class-wise breakdown.
    
    Importance Score = frequency / (average_interval_width + ε)
    Normalized to percentages (sum to 100%) for easy interpretation.
    Shows overall importance with stacked bars indicating per-class contributions.
    """
    if summary is None:
        ax.text(0.5, 0.5, 'No summary data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11, fontweight='bold')
        return
    
    per_class_full = summary.get("per_class_summary", {})
    # Filter out _class_based entries - only use main class entries
    per_class = {
        k: v for k, v in per_class_full.items() 
        if not k.endswith('_class_based') and v.get('rollout_type') != 'class_based'
    }
    
    # Collect feature intervals per class and globally
    feature_intervals_global: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    feature_intervals_per_class: Dict[int, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    feature_frequency_global = Counter()
    feature_frequency_per_class: Dict[int, Counter] = defaultdict(Counter)
    
    for class_key, class_data in per_class.items():
        target_class = class_data.get("class", -1)
        # Collect rules: instance-based plus union-level class-based rules (preferred),
        # falling back to per-agent class-based rules for backward compatibility.
        unique_rules = class_data.get("unique_rules", [])
        union_rules = class_data.get("class_union_unique_rules", [])
        class_based_unique_rules = class_data.get("class_based_unique_rules", [])
        rules_for_class_based = union_rules if union_rules else class_based_unique_rules
        all_unique_rules = unique_rules + rules_for_class_based
        
        for rule_str in all_unique_rules:
            intervals = extract_feature_intervals_from_rule(rule_str)
            for feature_name, lower, upper in intervals:
                # Global collection
                feature_intervals_global[feature_name].append((lower, upper))
                feature_frequency_global[feature_name] += 1
                
                # Per-class collection
                feature_intervals_per_class[target_class][feature_name].append((lower, upper))
                feature_frequency_per_class[target_class][feature_name] += 1
    
    if not feature_intervals_global:
        ax.text(0.5, 0.5, 'No feature data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11, fontweight='bold')
        return
    
    # Calculate global importance scores
    feature_importance_global = {}
    for feature_name, intervals_list in feature_intervals_global.items():
        # Calculate interval widths, filtering out invalid (negative or zero) widths
        interval_widths = []
        for lower, upper in intervals_list:
            width = float(upper - lower)
            # Validate width: should be non-negative, and finite
            if width >= 0 and np.isfinite(width):
                interval_widths.append(width)
            else:
                logger.warning(
                    f"Invalid interval width for feature '{feature_name}': "
                    f"lower={lower}, upper={upper}, width={width}. Skipping."
                )
        
        # Calculate average width, handling edge cases
        if not interval_widths:
            # No valid intervals: use default width
            avg_width = 1.0
            logger.debug(f"Feature '{feature_name}' has no valid intervals, using default avg_width=1.0")
        else:
            avg_width = float(np.mean(interval_widths))
            # Ensure avg_width is finite and positive
            if not np.isfinite(avg_width) or avg_width <= 0:
                logger.warning(
                    f"Invalid avg_width for feature '{feature_name}': {avg_width}. Using default 1.0."
                )
                avg_width = 1.0
        
        frequency = feature_frequency_global[feature_name]
        # Importance = frequency / avg_width (higher frequency and narrower intervals = more important)
        # Use small epsilon to avoid division by zero
        importance_score = float(frequency) / (avg_width + 1e-6)
        
        # Validate importance score
        if not np.isfinite(importance_score) or importance_score < 0:
            logger.warning(
                f"Invalid importance score for feature '{feature_name}': {importance_score}. "
                f"Setting to 0.0."
            )
            importance_score = 0.0
        
        feature_importance_global[feature_name] = {
            "importance": importance_score,
            "frequency": frequency,
            "avg_interval_width": avg_width
        }
    
    # Calculate per-class importance scores
    feature_importance_per_class: Dict[int, Dict[str, Dict]] = {}
    for target_class, class_intervals in feature_intervals_per_class.items():
        class_importance = {}
        for feature_name, intervals_list in class_intervals.items():
            # Calculate interval widths, filtering out invalid widths
            interval_widths = []
            for lower, upper in intervals_list:
                width = float(upper - lower)
                # Validate width: should be non-negative, and finite
                if width >= 0 and np.isfinite(width):
                    interval_widths.append(width)
                else:
                    logger.debug(
                        f"Invalid interval width for feature '{feature_name}' in class {target_class}: "
                        f"lower={lower}, upper={upper}, width={width}. Skipping."
                    )
            
            # Calculate average width, handling edge cases
            if not interval_widths:
                avg_width = 1.0
            else:
                avg_width = float(np.mean(interval_widths))
                # Ensure avg_width is finite and positive
                if not np.isfinite(avg_width) or avg_width <= 0:
                    avg_width = 1.0
            
            frequency = feature_frequency_per_class[target_class][feature_name]
            importance_score = float(frequency) / (avg_width + 1e-6)
            
            # Validate importance score
            if not np.isfinite(importance_score) or importance_score < 0:
                importance_score = 0.0
            
            class_importance[feature_name] = {
                "importance": importance_score,
                "frequency": frequency,
                "avg_interval_width": avg_width
            }
        feature_importance_per_class[target_class] = class_importance
    
    # Get all features sorted by importance, then take top N (or all if fewer than top_n)
    all_features_sorted = sorted(feature_importance_global.items(), key=lambda x: x[1]["importance"], reverse=True)
    n_available = len(all_features_sorted)
    n_to_show = min(top_n, n_available)  # Show top N, or all if fewer than N
    top_features = all_features_sorted[:n_to_show]
    
    if not top_features:
        ax.text(0.5, 0.5, 'No features found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11, fontweight='bold')
        return
    
    # Normalize importance scores to percentages (sum to 100%)
    # Calculate total importance across ALL features (not just top N) for proper normalization
    total_importance = sum(f["importance"] for f in feature_importance_global.values())
    
    # Validate total_importance
    if not np.isfinite(total_importance) or total_importance <= 0:
        logger.warning(
            f"Invalid total_importance: {total_importance}. "
            f"All feature importance percentages will be set to 0."
        )
        total_importance = 1.0  # Use 1.0 to avoid division by zero, but all percentages will be 0
    
    features = [f[0] for f in top_features]
    importances_raw = [f[1]["importance"] for f in top_features]
    importances_pct = []
    for imp in importances_raw:
        if total_importance > 0 and np.isfinite(imp):
            pct = (imp / total_importance * 100)
            # Validate percentage
            if np.isfinite(pct) and pct >= 0:
                importances_pct.append(pct)
            else:
                logger.debug(f"Invalid percentage calculated: {pct} from imp={imp}, total={total_importance}")
                importances_pct.append(0.0)
        else:
            importances_pct.append(0.0)
    
    # Also normalize per-class importance scores to percentages
    feature_importance_per_class_pct: Dict[int, Dict[str, float]] = {}
    for target_class, class_importance in feature_importance_per_class.items():
        class_total = sum(imp["importance"] for imp in class_importance.values())
        
        # Validate class_total
        if not np.isfinite(class_total) or class_total <= 0:
            logger.debug(
                f"Invalid class_total for class {target_class}: {class_total}. "
                f"All percentages for this class will be set to 0."
            )
            class_total = 1.0  # Avoid division by zero
        
        class_pct = {}
        for feat_name, feat_data in class_importance.items():
            raw_imp = feat_data["importance"]
            if class_total > 0 and np.isfinite(raw_imp):
                pct_imp = (raw_imp / class_total * 100)
                # Validate percentage
                if np.isfinite(pct_imp) and pct_imp >= 0:
                    class_pct[feat_name] = pct_imp
                else:
                    class_pct[feat_name] = 0.0
            else:
                class_pct[feat_name] = 0.0
        feature_importance_per_class_pct[target_class] = class_pct
    
    # Get all classes
    classes = sorted(feature_importance_per_class_pct.keys())
    n_classes = len(classes)
    
    if n_classes == 0:
        # If no class-wise data, show overall bars
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances_pct, alpha=0.8, color='teal', edgecolor='black', linewidth=1, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=7)
        ax.set_xlabel('Importance (%)', fontsize=9, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.set_xlim([0, max(importances_pct) * 1.1 if importances_pct else 100])
        for i, imp_pct in enumerate(importances_pct):
            ax.text(imp_pct + max(importances_pct) * 0.02 if importances_pct else 0, i, f'{imp_pct:.1f}%', va='center', fontsize=6)
    else:
        # Prepare data for heatmap: feature (rows) x class (columns)
        heatmap_data = []
        for feat in features:
            row = []
            for cls in classes:
                class_imp_pct = feature_importance_per_class_pct[cls].get(feat, 0.0)
                row.append(class_imp_pct)
            heatmap_data.append(row)
        
        heatmap_array = np.array(heatmap_data)
        
        # Create heatmap
        im = ax.imshow(heatmap_array, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels([f'C{cls}' for cls in classes], fontsize=6)
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features, fontsize=6)
        
        # Add text annotations with percentages for all cells
        for i in range(len(features)):
            for j in range(len(classes)):
                value = heatmap_array[i, j]
                if value > 0:  # Only show non-zero values
                    # Adjust font size based on grid size
                    max_dim = max(len(features), len(classes))
                    font_size = 6 if max_dim <= 5 else 5 if max_dim <= 8 else 4
                    text_color = 'white' if value > np.max(heatmap_array) * 0.5 else 'black'
                    ax.text(j, i, f'{value:.0f}%', ha='center', va='center', 
                            color=text_color, fontsize=font_size, fontweight='bold')
        
        ax.set_xlabel('Class', fontsize=8, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=8, fontweight='bold')
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Importance (%)', fontsize=7, fontweight='bold')


def plot_equilibrium_comparison(ax, single_agent_summary: Optional[Dict], multi_agent_summary: Optional[Dict], 
                                precision_target: float = 0.95, coverage_target: float = 0.1):
    """Plot target achievement comparison between single-agent and multi-agent."""
    if not HAS_PLOTTING:
        return
    
    single_eq = None
    multi_eq = None
    
    if single_agent_summary:
        single_per_class = single_agent_summary.get("per_class_summary", {})
        if single_per_class:
            single_eq = calculate_equilibrium_metrics(single_per_class, precision_target, coverage_target)
    
    if multi_agent_summary:
        multi_per_class = multi_agent_summary.get("per_class_summary", {})
        if multi_per_class:
            multi_eq = calculate_equilibrium_metrics(multi_per_class, precision_target, coverage_target)
    
    if not single_eq and not multi_eq:
        ax.text(0.5, 0.5, 'No target achievement data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Target Achievement Comparison', fontsize=11, fontweight='bold')
        return
    
    # Prepare data for plotting
    methods = []
    equilibrium_fractions = []
    equilibrium_reached = []
    colors = []
    
    if single_eq:
        methods.append('Single-Agent')
        equilibrium_fractions.append(single_eq["equilibrium_fraction"])
        equilibrium_reached.append(single_eq["equilibrium_reached"])
        colors.append('steelblue' if single_eq["equilibrium_reached"] else 'lightblue')
    
    if multi_eq:
        methods.append('Multi-Agent')
        equilibrium_fractions.append(multi_eq["equilibrium_fraction"])
        equilibrium_reached.append(multi_eq["equilibrium_reached"])
        colors.append('darkgreen' if multi_eq["equilibrium_reached"] else 'lightgreen')
    
    if not methods:
        return
    
    x = np.arange(len(methods))
    bars = ax.bar(x, equilibrium_fractions, alpha=0.8, color=colors, edgecolor='black', linewidth=2)
    
    # Add target achievement threshold line
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Target Achievement (100%)', alpha=0.7)
    
    # Add value labels and status
    for i, (frac, reached) in enumerate(zip(equilibrium_fractions, equilibrium_reached)):
        label = f'{frac:.1%}'
        if reached:
            label += '\n✓ TARGETS MET'
        ax.text(i, frac + 0.05, label, ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='darkgreen' if reached else 'black')
    
    # Add class counts
    if single_eq:
        ax.text(0, -0.15, f'{single_eq["classes_meeting_targets"]}/{single_eq["total_classes"]} classes',
                ha='center', va='top', fontsize=8, transform=ax.get_xaxis_transform())
    if multi_eq:
        idx = 1 if single_eq else 0
        ax.text(idx, -0.15, f'{multi_eq["classes_meeting_targets"]}/{multi_eq["total_classes"]} classes',
                ha='center', va='top', fontsize=8, transform=ax.get_xaxis_transform())
    
    ax.set_ylabel('Target Achievement Fraction', fontsize=10, fontweight='bold')
    
    # Extract actual target values from equilibrium metrics (they're stored there)
    # Use values from single_eq or multi_eq, whichever is available
    actual_precision_target = precision_target
    actual_coverage_target = coverage_target
    if single_eq:
        actual_precision_target = single_eq.get("precision_target", precision_target)
        actual_coverage_target = single_eq.get("coverage_target", coverage_target)
    elif multi_eq:
        actual_precision_target = multi_eq.get("precision_target", precision_target)
        actual_coverage_target = multi_eq.get("coverage_target", coverage_target)
    
    ax.set_title(f'Target Achievement Comparison\n(All classes meet targets: P≥{actual_precision_target:.2f}, C≥{actual_coverage_target:.2f})', 
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10, fontweight='bold')
    ax.set_ylim([0, 1.2])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='upper right', fontsize=8)


def plot_global_metrics_subplot(ax, summary: Optional[Dict], title: str):
    """Plot global metrics on a given axis."""
    if summary is None:
        ax.text(0.5, 0.5, 'No summary data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11, fontweight='bold')
        return
    
    per_class = summary.get("per_class_summary", {})
    overall_stats = summary.get("overall_stats", {})
    
    if not per_class:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11, fontweight='bold')
        return
    
    # Calculate global metrics
    global_instance_precision = overall_stats.get("mean_instance_precision") or overall_stats.get("mean_precision",
        np.mean([per_class[c]["instance_precision"] for c in per_class]))
    global_instance_coverage = overall_stats.get("mean_instance_coverage") or overall_stats.get("mean_coverage",
        np.mean([per_class[c]["instance_coverage"] for c in per_class]))
    global_class_precision = overall_stats.get("mean_class_precision",
        np.mean([per_class[c]["class_precision"] for c in per_class]))
    global_class_coverage = overall_stats.get("mean_class_coverage",
        np.mean([per_class[c]["class_coverage"] for c in per_class]))
    # REMOVED: Global metrics (averages across classes) - not meaningful per user request
    # This function is kept for backward compatibility but will show empty/placeholder
    ax.text(0.5, 0.5, 'Global metrics removed\n(not meaningful for comparison)', 
            ha='center', va='center', transform=ax.transAxes, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    return
    
    # OLD CODE (commented out - kept for reference):
    # global_class_level_precision = overall_stats.get("mean_class_level_precision", overall_stats.get("mean_class_based_precision", 0.0))
    # global_class_level_coverage = overall_stats.get("mean_class_level_coverage", overall_stats.get("mean_class_based_coverage", 0.0))
    # 
    # # Calculate standard deviations
    # instance_precisions = [per_class[c]["instance_precision"] for c in per_class]
    # instance_coverages = [per_class[c]["instance_coverage"] for c in per_class]
    # class_union_precisions = [per_class[c].get("class_union_precision", per_class[c].get("class_precision", 0.0)) for c in per_class]
    # class_union_coverages = [per_class[c].get("class_union_coverage", per_class[c].get("class_coverage", 0.0)) for c in per_class]
    # class_level_precisions = [per_class[c].get("class_level_precision", per_class[c].get("class_based_precision", 0.0)) for c in per_class if per_class[c].get("class_level_precision", per_class[c].get("class_based_precision", 0.0)) > 0]
    # class_level_coverages = [per_class[c].get("class_level_coverage", per_class[c].get("class_based_coverage", 0.0)) for c in per_class if per_class[c].get("class_level_coverage", per_class[c].get("class_based_coverage", 0.0)) > 0]
    
    x = np.arange(len(metrics))
    bars = ax.bar(x, values, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5,
                  yerr=stds, capsize=8, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    
    ax.set_ylabel('Value', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for i, (val, std) in enumerate(zip(values, stds)):
        label = f'{val:.3f}'
        if std > 0:
            label += f'\n±{std:.3f}'
        ax.text(i, val + std + 0.05, label, ha='center', va='bottom', fontsize=8, fontweight='bold')


def plot_rule_overlap_subplot(ax, summary: Dict, title: str, max_rules: int = 5):
    """
    Plot rule overlap heatmap for a given summary on a subplot axis.
    
    Shows rule-to-rule overlap for the class with the most rules.
    """
    per_class = summary.get("per_class_summary", {})
    
    if not per_class:
        ax.text(0.5, 0.5, 'No class data available', ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return
    
    # Calculate rule overlap for each class
    def calculate_rule_overlap(rule1_intervals: List[Tuple[str, float, float]], 
                               rule2_intervals: List[Tuple[str, float, float]]) -> float:
        """
        Calculate overlap score between two rules based on their intervals.
        
        Since rules are AND conditions, we use the minimum overlap across all common features.
        This correctly reflects that ALL features must overlap for rules to overlap.
        """
        if not rule1_intervals or not rule2_intervals:
            return 0.0
        
        # Create feature to interval mapping (handle duplicate features by keeping first occurrence)
        rule1_dict = {}
        for feat, lower, upper in rule1_intervals:
            # Validate interval: lower <= upper
            if lower > upper:
                # Invalid interval: swap or skip
                logger.warning(f"Invalid interval in rule1: {feat} has lower={lower} > upper={upper}. Swapping.")
                lower, upper = upper, lower
            # Only keep first occurrence if duplicate features exist
            if feat not in rule1_dict:
                rule1_dict[feat] = (lower, upper)
        
        rule2_dict = {}
        for feat, lower, upper in rule2_intervals:
            # Validate interval: lower <= upper
            if lower > upper:
                # Invalid interval: swap or skip
                logger.warning(f"Invalid interval in rule2: {feat} has lower={lower} > upper={upper}. Swapping.")
                lower, upper = upper, lower
            # Only keep first occurrence if duplicate features exist
            if feat not in rule2_dict:
                rule2_dict[feat] = (lower, upper)
        
        # Find common features
        common_features = set(rule1_dict.keys()) & set(rule2_dict.keys())
        if not common_features:
            return 0.0
        
        # Calculate overlap for each common feature
        feature_overlaps = []
        for feat in common_features:
            lower1, upper1 = rule1_dict[feat]
            lower2, upper2 = rule2_dict[feat]
            
            # Ensure valid intervals (should be guaranteed by validation above, but double-check)
            if lower1 > upper1 or lower2 > upper2:
                feature_overlaps.append(0.0)
                continue
            
            # Calculate intersection
            intersect_lower = max(lower1, lower2)
            intersect_upper = min(upper1, upper2)
            
            if intersect_lower <= intersect_upper:
                # Calculate Jaccard-like overlap: intersection / union
                intersection = intersect_upper - intersect_lower
                union_lower = min(lower1, lower2)
                union_upper = max(upper1, upper2)
                union = union_upper - union_lower
                
                if union > 0:
                    feature_overlaps.append(intersection / union)
                else:
                    # Zero-width intervals: if they're the same point, perfect overlap (1.0)
                    # Otherwise, different points, no overlap (0.0)
                    if abs(lower1 - upper1) < 1e-10 and abs(lower2 - upper2) < 1e-10:
                        # Both are zero-width intervals
                        if abs(lower1 - lower2) < 1e-10:
                            feature_overlaps.append(1.0)  # Same point
                        else:
                            feature_overlaps.append(0.0)  # Different points
                    else:
                        feature_overlaps.append(0.0)
            else:
                # No intersection for this feature
                feature_overlaps.append(0.0)
        
        # For AND conditions, use minimum (all features must overlap)
        # This is more accurate than averaging, which can overestimate overlap
        return min(feature_overlaps) if feature_overlaps else 0.0
    
    # Find class with most rules
    # Prefer ranked rules if available (from test_extracted_rules), otherwise use unique_rules
    class_with_most_rules = None
    max_rules_count = 0
    
    for class_key, class_data in per_class.items():
        target_class = class_data.get("class", -1)
        # Check for ranked rules first (from test_extracted_rules ranking)
        ranked_rules = class_data.get("ranked_unique_rules", [])
        if not ranked_rules:
            # Fallback to unique_rules if ranked rules not available
            ranked_rules = class_data.get("unique_rules", [])
        
        # Sort rules for consistent plotting
        ranked_rules = sort_rules_for_plotting(class_data, ranked_rules)
        
        if len(ranked_rules) > max_rules_count:
            max_rules_count = len(ranked_rules)
            class_with_most_rules = target_class
    
    if class_with_most_rules is None or max_rules_count <= 1:
        ax.text(0.5, 0.5, 'No rules found for overlap analysis', 
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return
    
    # Get rules for this class
    class_data = None
    for class_key, cd in per_class.items():
        if cd.get("class", -1) == class_with_most_rules:
            class_data = cd
            break
    
    if class_data is None:
        ax.text(0.5, 0.5, f'Could not find data for class {class_with_most_rules}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return
    
    # Prefer ranked rules if available (from test_extracted_rules), otherwise use unique_rules
    class_rules = class_data.get("ranked_unique_rules", [])
    if not class_rules:
        class_rules = class_data.get("unique_rules", [])
    
    # Sort rules for consistent plotting
    class_rules = sort_rules_for_plotting(class_data, class_rules)
    
    if len(class_rules) <= 1:
        ax.text(0.5, 0.5, f'Class {class_with_most_rules} has only 1 rule\n(no overlap to show)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return
    
    # Always show top 5 rules for overlap analysis (already sorted)
    n_rules_to_show = min(5, len(class_rules))
    class_rules_subset = class_rules[:n_rules_to_show]
    
    # Calculate overlap matrix
    overlap_matrix = np.zeros((n_rules_to_show, n_rules_to_show))
    rule_intervals = [extract_feature_intervals_from_rule(rule) for rule in class_rules_subset]
    
    for i in range(n_rules_to_show):
        for j in range(n_rules_to_show):
            if i == j:
                overlap_matrix[i, j] = 1.0  # Self-overlap
            else:
                overlap_matrix[i, j] = calculate_rule_overlap(rule_intervals[i], rule_intervals[j])
    
    # Create heatmap
    im = ax.imshow(overlap_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_rules_to_show))
    ax.set_yticks(np.arange(n_rules_to_show))
    ax.set_xticklabels([f"R{i+1}" for i in range(n_rules_to_show)], fontsize=6, rotation=45, ha='right')
    ax.set_yticklabels([f"R{i+1}" for i in range(n_rules_to_show)], fontsize=6)
    
    # Add text annotations for all cells
    for i in range(n_rules_to_show):
        for j in range(n_rules_to_show):
            value = overlap_matrix[i, j]
            # Adjust font size based on number of rules
            font_size = 6 if n_rules_to_show <= 6 else 5 if n_rules_to_show <= 8 else 4
            text_color = 'white' if value > 0.5 else 'black'
            # For diagonal (self-overlap = 1.0), show as "1.0", otherwise show 2 decimals
            if i == j:
                text = '1.0'
            else:
                text = f'{value:.2f}' if value > 0.01 else ''  # Skip very small values
            if text:
                ax.text(j, i, text, ha='center', va='center', 
                        color=text_color, fontsize=font_size, fontweight='bold')
    
    ax.set_xlabel('Rule', fontsize=8, fontweight='bold')
    ax.set_ylabel('Rule', fontsize=8, fontweight='bold')
    ax.set_title(f'{title}\n(Class {class_with_most_rules}, {max_rules_count} rules, showing {n_rules_to_show})', 
                 fontsize=9, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Overlap', fontsize=7, fontweight='bold')


def plot_rule_overlap_comparison_per_class(
    single_agent_summary: Optional[Dict],
    multi_agent_summary: Optional[Dict],
    output_dir: str,
    dataset_name: str = "",
    max_rules: int = 5
):
    """Plot rule overlap comparison class-by-class (single vs multi)."""
    if not HAS_PLOTTING:
        return
    
    # Check for None summaries
    if single_agent_summary is None and multi_agent_summary is None:
        logger.warning("Both summaries are None, skipping rule overlap comparison.")
        return
    
    # Calculate rule overlap helper function
    def calculate_rule_overlap(rule1_intervals: List[Tuple[str, float, float]], 
                               rule2_intervals: List[Tuple[str, float, float]]) -> float:
        """
        Calculate overlap score between two rules based on their intervals.
        
        Since rules are AND conditions, we use the minimum overlap across all common features.
        This correctly reflects that ALL features must overlap for rules to overlap.
        """
        if not rule1_intervals or not rule2_intervals:
            return 0.0
        
        # Create feature to interval mapping (handle duplicate features by keeping first occurrence)
        rule1_dict = {}
        for feat, lower, upper in rule1_intervals:
            # Validate interval: lower <= upper
            if lower > upper:
                # Invalid interval: swap or skip
                logger.warning(f"Invalid interval in rule1: {feat} has lower={lower} > upper={upper}. Swapping.")
                lower, upper = upper, lower
            # Only keep first occurrence if duplicate features exist
            if feat not in rule1_dict:
                rule1_dict[feat] = (lower, upper)
        
        rule2_dict = {}
        for feat, lower, upper in rule2_intervals:
            # Validate interval: lower <= upper
            if lower > upper:
                # Invalid interval: swap or skip
                logger.warning(f"Invalid interval in rule2: {feat} has lower={lower} > upper={upper}. Swapping.")
                lower, upper = upper, lower
            # Only keep first occurrence if duplicate features exist
            if feat not in rule2_dict:
                rule2_dict[feat] = (lower, upper)
        
        # Find common features
        common_features = set(rule1_dict.keys()) & set(rule2_dict.keys())
        if not common_features:
            return 0.0
        
        # Calculate overlap for each common feature
        feature_overlaps = []
        for feat in common_features:
            lower1, upper1 = rule1_dict[feat]
            lower2, upper2 = rule2_dict[feat]
            
            # Ensure valid intervals (should be guaranteed by validation above, but double-check)
            if lower1 > upper1 or lower2 > upper2:
                feature_overlaps.append(0.0)
                continue
            
            # Calculate intersection
            intersect_lower = max(lower1, lower2)
            intersect_upper = min(upper1, upper2)
            
            if intersect_lower <= intersect_upper:
                intersection = intersect_upper - intersect_lower
                union_lower = min(lower1, lower2)
                union_upper = max(upper1, upper2)
                union = union_upper - union_lower
                
                if union > 0:
                    feature_overlaps.append(intersection / union)
                else:
                    # Zero-width intervals: if they're the same point, perfect overlap (1.0)
                    # Otherwise, different points, no overlap (0.0)
                    if abs(lower1 - upper1) < 1e-10 and abs(lower2 - upper2) < 1e-10:
                        # Both are zero-width intervals
                        if abs(lower1 - lower2) < 1e-10:
                            feature_overlaps.append(1.0)  # Same point
                        else:
                            feature_overlaps.append(0.0)  # Different points
                    else:
                        feature_overlaps.append(0.0)
            else:
                # No intersection for this feature
                feature_overlaps.append(0.0)
        
        # For AND conditions, use minimum (all features must overlap)
        # This is more accurate than averaging, which can overestimate overlap
        return min(feature_overlaps) if feature_overlaps else 0.0
    
    def plot_overlap_for_class(ax, class_rules: List[str], class_num: int, title: str, per_class_data: Optional[Dict] = None):
        """Plot overlap heatmap for a class on given axis."""
        if len(class_rules) <= 1:
            ax.text(0.5, 0.5, f'Class {class_num} has only 1 rule\n(no overlap to show)', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')
            return
        
        # Sort rules for consistent plotting if per_class_data is available
        if per_class_data:
            class_rules = sort_rules_for_plotting(per_class_data, class_rules)
        
        # Always show top 5 rules for overlap analysis (already sorted)
        n_rules_to_show = min(5, len(class_rules))
        class_rules_subset = class_rules[:n_rules_to_show]
        
        # Calculate overlap matrix
        overlap_matrix = np.zeros((n_rules_to_show, n_rules_to_show))
        rule_intervals = [extract_feature_intervals_from_rule(rule) for rule in class_rules_subset]
        
        for i in range(n_rules_to_show):
            for j in range(n_rules_to_show):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                else:
                    overlap_matrix[i, j] = calculate_rule_overlap(rule_intervals[i], rule_intervals[j])
        
        # Create heatmap
        im = ax.imshow(overlap_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n_rules_to_show))
        ax.set_yticks(np.arange(n_rules_to_show))
        ax.set_xticklabels([f"R{i+1}" for i in range(n_rules_to_show)], fontsize=6, rotation=45, ha='right')
        ax.set_yticklabels([f"R{i+1}" for i in range(n_rules_to_show)], fontsize=6)
        
        # Add text annotations for all cells
        for i in range(n_rules_to_show):
            for j in range(n_rules_to_show):
                value = overlap_matrix[i, j]
                # Adjust font size based on number of rules
                font_size = 6 if n_rules_to_show <= 6 else 5 if n_rules_to_show <= 8 else 4
                text_color = 'white' if value > 0.5 else 'black'
                # For diagonal (self-overlap = 1.0), show as "1.0", otherwise show 2 decimals
                if i == j:
                    text = '1.0'
                else:
                    text = f'{value:.2f}' if value > 0.01 else ''  # Skip very small values
                if text:
                    ax.text(j, i, text, ha='center', va='center', 
                            color=text_color, fontsize=font_size, fontweight='bold')
        
        ax.set_xlabel('Rule', fontsize=8, fontweight='bold')
        ax.set_ylabel('Rule', fontsize=8, fontweight='bold')
        ax.set_title(title, fontsize=9, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Overlap', fontsize=7, fontweight='bold')
    
    # Get classes with multiple rules from both summaries
    sa_per_class = single_agent_summary.get("per_class_summary", {}) if single_agent_summary else {}
    ma_per_class = multi_agent_summary.get("per_class_summary", {}) if multi_agent_summary else {}
    
    # Find all classes that have multiple rules in either summary
    all_classes = set()
    for class_data in sa_per_class.values():
        target_class = class_data.get("class", -1)
        if len(class_data.get("unique_rules", [])) > 1:
            all_classes.add(target_class)
    for class_data in ma_per_class.values():
        target_class = class_data.get("class", -1)
        if len(class_data.get("unique_rules", [])) > 1:
            all_classes.add(target_class)
    
    if not all_classes:
        logger.warning("No classes with multiple rules found for overlap comparison.")
        return
    
    classes_sorted = sorted(all_classes)
    n_classes = len(classes_sorted)
    
    # Format title with dataset name
    title_prefix = f"{dataset_name.upper()}" if dataset_name else ""
    
    # Create figure: 2 columns (single vs multi), n_classes rows
    fig, axes = plt.subplots(n_classes, 2, figsize=(16, 4 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, target_class in enumerate(classes_sorted):
        # Get rules for this class from both summaries (prefer ranked rules)
        sa_class_rules = []
        sa_class_data = None
        ma_class_rules = []
        ma_class_data = None
        
        for class_data in sa_per_class.values():
            if class_data.get("class", -1) == target_class:
                sa_class_data = class_data
                # Prefer ranked rules if available
                sa_class_rules = class_data.get("ranked_unique_rules", [])
                if not sa_class_rules:
                    sa_class_rules = class_data.get("unique_rules", [])
                # Sort rules for consistent plotting
                sa_class_rules = sort_rules_for_plotting(class_data, sa_class_rules)
                break
        
        for class_data in ma_per_class.values():
            if class_data.get("class", -1) == target_class:
                ma_class_data = class_data
                # Prefer ranked rules if available
                ma_class_rules = class_data.get("ranked_unique_rules", [])
                if not ma_class_rules:
                    ma_class_rules = class_data.get("unique_rules", [])
                # Sort rules for consistent plotting
                ma_class_rules = sort_rules_for_plotting(class_data, ma_class_rules)
                break
        
        # Plot single-agent overlap
        ax_sa = axes[row_idx, 0]
        plot_overlap_for_class(ax_sa, sa_class_rules, target_class, 
                              f'{title_prefix}: Single-Agent - Class {target_class}',
                              per_class_data=sa_class_data)
        
        # Plot multi-agent overlap
        ax_ma = axes[row_idx, 1]
        plot_overlap_for_class(ax_ma, ma_class_rules, target_class, 
                              f'{title_prefix}: Multi-Agent - Class {target_class}',
                              per_class_data=ma_class_data)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rule_overlap_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved rule overlap comparison plot to {output_dir}/rule_overlap_comparison.png")


def plot_nashconv_comparison(
    multi_agent_summary: Optional[Dict],
    output_dir: str,
    dataset_name: str = ""
):
    """
    Plot NashConv convergence comparison (only for multi-agent, as single-agent doesn't have NashConv).
    
    Improved version that:
    - Shows informative messages when data is missing
    - Uses flexible layout based on available data
    - Combines training and evaluation on same plots for comparison
    
    Args:
        multi_agent_summary: Multi-agent summary dictionary (may contain nashconv_metrics)
        output_dir: Output directory for the plot
        dataset_name: Dataset name for title
    """
    if not HAS_PLOTTING:
        return
    
    if not multi_agent_summary:
        logger.debug("No multi-agent summary, skipping NashConv plot")
        return
    
    nashconv_data = multi_agent_summary.get("nashconv_metrics", {})
    if not nashconv_data.get("available", False):
        logger.debug("NashConv data not available in multi-agent summary, skipping plot")
        return
    
    training_data = nashconv_data.get("training", []) or []
    eval_data = nashconv_data.get("evaluation", []) or []
    
    # Log what data is available
    logger.info(f"NashConv data available: Training={len(training_data)} points, Evaluation={len(eval_data)} points")
    if not training_data:
        logger.warning("⚠ Training NashConv data is missing. This could mean:")
        logger.warning("  1. Training history was not logged properly")
        logger.warning("  2. training_history.json file is missing or empty")
        logger.warning("  3. NashConv metrics were not computed during training")
    
    title_prefix = f"{dataset_name.upper()}" if dataset_name else ""
    
    # Determine what data is available
    has_training = len(training_data) > 0
    has_eval = len(eval_data) > 0
    
    if not has_training and not has_eval:
        logger.warning("⚠ No NashConv data available (neither training nor evaluation), skipping plot")
        return
    
    # Only create the training vs evaluation comparison plot (more useful than separate 2x2 grid)
    # This shows both training and evaluation on the same axes for direct comparison
    if has_training and has_eval:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{title_prefix}: Training vs Evaluation NashConv Comparison", fontsize=14, fontweight='bold')
        
        # Combined NashConv Sum plot
        ax1 = axes[0]
        
        # Prepare training data
        train_x = []
        for i, e in enumerate(training_data):
            if "total_frames" in e and e["total_frames"] is not None:
                train_x.append(e["total_frames"])
            elif "step" in e and e["step"] is not None:
                train_x.append(e["step"])
            else:
                train_x.append(i)
        train_nashconv = [e.get("training/nashconv_sum", 0.0) for e in training_data]
        
        # Prepare evaluation data
        eval_x = []
        for i, e in enumerate(eval_data):
            if "total_frames" in e and e["total_frames"] is not None:
                eval_x.append(e["total_frames"])
            elif "step" in e and e["step"] is not None:
                eval_x.append(e["step"])
            else:
                eval_x.append(i)
        eval_nashconv = [e.get("evaluation/nashconv_sum", 0.0) for e in eval_data]
        
        # Plot both on same axes
        if len(train_nashconv) == 1:
            ax1.scatter(train_x, train_nashconv, s=100, color='b', marker='o', label='Training', zorder=3)
        else:
            ax1.plot(train_x, train_nashconv, 'b-', marker='o', linewidth=2, markersize=4, label='Training NashConv')
        
        if len(eval_nashconv) == 1:
            ax1.scatter(eval_x, eval_nashconv, s=100, color='r', marker='^', label='Evaluation', zorder=3)
        else:
            ax1.plot(eval_x, eval_nashconv, 'r-', marker='^', linewidth=2, markersize=4, label='Evaluation NashConv')
        
        ax1.set_xlabel('Step/Frames')
        ax1.set_ylabel('NashConv Sum')
        ax1.set_title('NashConv Sum: Training vs Evaluation')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='ε=0.01 threshold')
        
        # Combined Exploitability plot
        ax2 = axes[1]
        
        train_exploit = [e.get("training/exploitability_max", 0.0) for e in training_data]
        eval_exploit = [e.get("evaluation/exploitability_max", 0.0) for e in eval_data]
        
        if len(train_exploit) == 1:
            ax2.scatter(train_x, train_exploit, s=100, color='g', marker='s', label='Training', zorder=3)
        else:
            ax2.plot(train_x, train_exploit, 'g-', marker='s', linewidth=2, markersize=4, label='Training Max Exploitability')
        
        if len(eval_exploit) == 1:
            ax2.scatter(eval_x, eval_exploit, s=100, color='m', marker='d', label='Evaluation', zorder=3)
        else:
            ax2.plot(eval_x, eval_exploit, 'm-', marker='d', linewidth=2, markersize=4, label='Evaluation Max Exploitability')
        
        ax2.set_xlabel('Step/Frames')
        ax2.set_ylabel('Max Exploitability')
        ax2.set_title('Max Exploitability: Training vs Evaluation')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        combined_path = Path(output_dir) / "nashconv_training_vs_evaluation.png"
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Training vs Evaluation NashConv plot to {combined_path}")
    else:
        # If only one type of data is available, create a simple plot
        logger.info(f"Only {'training' if has_training else 'evaluation'} NashConv data available, creating single plot")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{title_prefix}: {'Training' if has_training else 'Evaluation'} NashConv Metrics", fontsize=14, fontweight='bold')
        
        if has_training:
            # Plot training data only
            train_x = []
            for i, e in enumerate(training_data):
                if "total_frames" in e and e["total_frames"] is not None:
                    train_x.append(e["total_frames"])
                elif "step" in e and e["step"] is not None:
                    train_x.append(e["step"])
                else:
                    train_x.append(i)
            train_nashconv = [e.get("training/nashconv_sum", 0.0) for e in training_data]
            train_exploit = [e.get("training/exploitability_max", 0.0) for e in training_data]
            
            ax1, ax2 = axes[0], axes[1]
            if len(train_nashconv) == 1:
                ax1.scatter(train_x, train_nashconv, s=100, color='b', marker='o', label='Training NashConv', zorder=3)
            else:
                ax1.plot(train_x, train_nashconv, 'b-', marker='o', linewidth=2, markersize=4, label='Training NashConv')
            ax1.set_xlabel('Training Frames' if any("total_frames" in e for e in training_data) else 'Training Step')
            ax1.set_ylabel('NashConv Sum')
            ax1.set_title('Training NashConv Sum')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='ε=0.01 threshold')
            
            if len(train_exploit) == 1:
                ax2.scatter(train_x, train_exploit, s=100, color='g', marker='s', label='Training Max Exploitability', zorder=3)
            else:
                ax2.plot(train_x, train_exploit, 'g-', marker='s', linewidth=2, markersize=4, label='Training Max Exploitability')
            ax2.set_xlabel('Training Frames' if any("total_frames" in e for e in training_data) else 'Training Step')
            ax2.set_ylabel('Max Exploitability')
            ax2.set_title('Training Max Exploitability')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        elif has_eval:
            # Plot evaluation data only
            eval_x = []
            for i, e in enumerate(eval_data):
                if "total_frames" in e and e["total_frames"] is not None:
                    eval_x.append(e["total_frames"])
                elif "step" in e and e["step"] is not None:
                    eval_x.append(e["step"])
                else:
                    eval_x.append(i)
            eval_nashconv = [e.get("evaluation/nashconv_sum", 0.0) for e in eval_data]
            eval_exploit = [e.get("evaluation/exploitability_max", 0.0) for e in eval_data]
            
            ax1, ax2 = axes[0], axes[1]
            if len(eval_nashconv) == 1:
                ax1.scatter(eval_x, eval_nashconv, s=100, color='r', marker='^', label='Evaluation NashConv', zorder=3)
            else:
                ax1.plot(eval_x, eval_nashconv, 'r-', marker='^', linewidth=2, markersize=4, label='Evaluation NashConv')
            ax1.set_xlabel('Evaluation Frames' if any("total_frames" in e for e in eval_data) else 'Evaluation Step')
            ax1.set_ylabel('NashConv Sum')
            ax1.set_title('Evaluation NashConv Sum')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='ε=0.01 threshold')
            
            if len(eval_exploit) == 1:
                ax2.scatter(eval_x, eval_exploit, s=100, color='m', marker='d', label='Evaluation Max Exploitability', zorder=3)
            else:
                ax2.plot(eval_x, eval_exploit, 'm-', marker='d', linewidth=2, markersize=4, label='Evaluation Max Exploitability')
            ax2.set_xlabel('Evaluation Frames' if any("total_frames" in e for e in eval_data) else 'Evaluation Step')
            ax2.set_ylabel('Max Exploitability')
            ax2.set_title('Evaluation Max Exploitability')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        single_path = Path(output_dir) / "nashconv_training_vs_evaluation.png"
        plt.savefig(single_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {'Training' if has_training else 'Evaluation'} NashConv plot to {single_path}")


def save_comparison_metrics_json(
    single_agent_summary: Optional[Dict],
    multi_agent_summary: Optional[Dict],
    baseline_summary: Optional[Dict],
    output_dir: str,
    dataset_name: str = ""
):
    """
    Save consolidated comparison metrics to JSON file for easy Excel copying.
    
    Saves metrics in a flat, easy-to-copy format:
    - Baseline: average instance precision/coverage
    - Single-agent: instance, class (union), class-based metrics, NashConv, unique rules, timing
    - Multi-agent: instance, class (union), class-based metrics, NashConv, unique rules, timing
    """
    consolidated = {
        "dataset": dataset_name,
        "baseline": {},
        "single_agent": {},
        "multi_agent": {}
    }
    
    # Extract baseline metrics
    if baseline_summary:
        baseline_overall = baseline_summary.get("overall_stats", {})
        consolidated["baseline"] = {
            "avg_instance_precision": baseline_overall.get("mean_instance_precision", baseline_overall.get("mean_precision", 0.0)),
            "avg_instance_coverage": baseline_overall.get("mean_instance_coverage", baseline_overall.get("mean_coverage", 0.0)),
            "std_instance_precision": baseline_overall.get("std_instance_precision", 0.0),
            "std_instance_coverage": baseline_overall.get("std_instance_coverage", 0.0),
        }
    
    # Extract single-agent metrics
    if single_agent_summary:
        sa_overall = single_agent_summary.get("overall_stats", {})
        sa_per_class = single_agent_summary.get("per_class_summary", {})
        
        # Compute averages from per_class_summary (overall_stats no longer has these)
        instance_precisions = [c.get("instance_precision", 0.0) for c in sa_per_class.values() if c.get("instance_precision", 0.0) > 0]
        instance_coverages = [c.get("instance_coverage", 0.0) for c in sa_per_class.values() if c.get("instance_coverage", 0.0) > 0]
        class_precisions = [c.get("class_union_precision", c.get("class_precision", 0.0)) for c in sa_per_class.values() if c.get("class_union_precision", c.get("class_precision", 0.0)) > 0]
        class_coverages = [c.get("class_union_coverage", c.get("class_coverage", 0.0)) for c in sa_per_class.values() if c.get("class_union_coverage", c.get("class_coverage", 0.0)) > 0]
        
        # For class-based metrics, check if keys exist (even if values are 0.0)
        # Also check for class_{class}_class_based entries in per_class_results
        class_based_precisions = []
        class_based_coverages = []
        has_class_based_data = False
        
        # Check per_class_summary for class-based metrics
        for class_data in sa_per_class.values():
            # Check for class_level_precision/coverage or class_based_precision/coverage
            if "class_level_precision" in class_data or "class_based_precision" in class_data:
                has_class_based_data = True
                prec = class_data.get("class_level_precision", class_data.get("class_based_precision", 0.0))
                cov = class_data.get("class_level_coverage", class_data.get("class_based_coverage", 0.0))
                class_based_precisions.append(prec)
                class_based_coverages.append(cov)
        
        # Also check if there are separate class_{class}_class_based entries
        # These might be in per_class_results if the summary structure is different
        if not has_class_based_data:
            # Try checking overall_stats for class-based rule counts as indicator
            if sa_overall.get("total_unique_rules_class_based", 0) > 0:
                has_class_based_data = True
                # If we have class-based rules but no metrics, set to 0.0
                if not class_based_precisions:
                    class_based_precisions = [0.0]
                    class_based_coverages = [0.0]
        
        # Calculate average rollout time
        avg_rollout_times = []
        total_rollout_times = []
        for class_data in sa_per_class.values():
            avg_time = class_data.get("avg_rollout_time_seconds", 0.0)
            total_time = class_data.get("total_rollout_time_seconds", 0.0)
            if avg_time > 0:
                avg_rollout_times.append(avg_time)
            if total_time > 0:
                total_rollout_times.append(total_time)
        
        avg_rollout_time = float(np.mean(avg_rollout_times)) if avg_rollout_times else 0.0
        total_rollout_time = float(np.sum(total_rollout_times)) if total_rollout_times else 0.0
        
        # Extract NashConv (typically not available for single-agent, but check anyway)
        nashconv_metrics = single_agent_summary.get("nashconv_metrics", {})
        nashconv_data = {}
        if nashconv_metrics.get("available", False):
            training_data = nashconv_metrics.get("training", [])
            eval_data = nashconv_metrics.get("evaluation", [])
            if training_data:
                latest_training = training_data[-1] if training_data else {}
                nashconv_data["training_nashconv_sum"] = latest_training.get("training/nashconv_sum", 0.0)
                nashconv_data["training_exploitability_max"] = latest_training.get("training/exploitability_max", 0.0)
            if eval_data:
                latest_eval = eval_data[-1] if eval_data else {}
                nashconv_data["evaluation_nashconv_sum"] = latest_eval.get("evaluation/nashconv_sum", 0.0)
                nashconv_data["evaluation_exploitability_max"] = latest_eval.get("evaluation/exploitability_max", 0.0)
        
        consolidated["single_agent"] = {
            "avg_instance_precision": float(np.mean(instance_precisions)) if instance_precisions else 0.0,
            "avg_instance_coverage": float(np.mean(instance_coverages)) if instance_coverages else 0.0,
            "avg_class_precision": float(np.mean(class_precisions)) if class_precisions else 0.0,  # Union precision
            "avg_class_coverage": float(np.mean(class_coverages)) if class_coverages else 0.0,  # Union coverage
            "avg_class_based_precision": float(np.mean(class_based_precisions)) if class_based_precisions else 0.0,
            "avg_class_based_coverage": float(np.mean(class_based_coverages)) if class_based_coverages else 0.0,
            "has_class_based_data": has_class_based_data,  # Flag to indicate if class-based data exists
            "unique_rules_count": sa_overall.get("total_unique_rules_instance_based", sa_overall.get("total_unique_rules", 0)),
            "unique_rules_class_based_count": sa_overall.get("total_unique_rules_class_based", 0),
            "total_unique_rules": sa_overall.get("total_unique_rules", 0),
            "avg_rollout_time_seconds": avg_rollout_time,
            "total_rollout_time_seconds": total_rollout_time,
            "nashconv": nashconv_data if nashconv_data else None
        }
    
    # Extract multi-agent metrics
    if multi_agent_summary:
        ma_overall = multi_agent_summary.get("overall_stats", {})
        ma_per_class = multi_agent_summary.get("per_class_summary", {})
        
        # Compute averages from per_class_summary (overall_stats no longer has these)
        instance_precisions = [c.get("instance_precision", 0.0) for c in ma_per_class.values() if c.get("instance_precision", 0.0) > 0]
        instance_coverages = [c.get("instance_coverage", 0.0) for c in ma_per_class.values() if c.get("instance_coverage", 0.0) > 0]
        class_precisions = [c.get("class_union_precision", c.get("class_precision", 0.0)) for c in ma_per_class.values() if c.get("class_union_precision", c.get("class_precision", 0.0)) > 0]
        class_coverages = [c.get("class_union_coverage", c.get("class_coverage", 0.0)) for c in ma_per_class.values() if c.get("class_union_coverage", c.get("class_coverage", 0.0)) > 0]
        class_based_precisions = [c.get("class_level_precision", c.get("class_based_precision", 0.0)) for c in ma_per_class.values() if c.get("class_level_precision", c.get("class_based_precision", 0.0)) > 0]
        class_based_coverages = [c.get("class_level_coverage", c.get("class_based_coverage", 0.0)) for c in ma_per_class.values() if c.get("class_level_coverage", c.get("class_based_coverage", 0.0)) > 0]
        
        # Calculate average rollout time
        # For multi-agent, rollout times are stored in per_agent_results
        avg_rollout_times = []
        total_rollout_times = []
        for class_data in ma_per_class.values():
            # Check per_agent_results first (for multi-agent with agents_per_class > 1)
            if "per_agent_results" in class_data:
                per_agent_times = []
                per_agent_total_times = []
                for agent_result in class_data["per_agent_results"].values():
                    agent_avg_time = agent_result.get("avg_rollout_time_seconds", 0.0)
                    agent_total_time = agent_result.get("total_rollout_time_seconds", 0.0)
                    if agent_avg_time > 0:
                        per_agent_times.append(agent_avg_time)
                    if agent_total_time > 0:
                        per_agent_total_times.append(agent_total_time)
                if per_agent_times:
                    avg_rollout_times.extend(per_agent_times)  # Collect all agent times
                if per_agent_total_times:
                    total_rollout_times.extend(per_agent_total_times)
            else:
                # Fallback: check if rollout time is directly in class_data (for single agent per class)
                avg_time = class_data.get("avg_rollout_time_seconds", 0.0)
                total_time = class_data.get("total_rollout_time_seconds", 0.0)
                if avg_time > 0:
                    avg_rollout_times.append(avg_time)
                if total_time > 0:
                    total_rollout_times.append(total_time)
        
        avg_rollout_time = float(np.mean(avg_rollout_times)) if avg_rollout_times else 0.0
        total_rollout_time = float(np.sum(total_rollout_times)) if total_rollout_times else 0.0
        
        # Extract NashConv
        nashconv_metrics = multi_agent_summary.get("nashconv_metrics", {})
        nashconv_data = {}
        if nashconv_metrics.get("available", False):
            training_data = nashconv_metrics.get("training", [])
            eval_data = nashconv_metrics.get("evaluation", [])
            if training_data:
                latest_training = training_data[-1] if training_data else {}
                nashconv_data["training_nashconv_sum"] = latest_training.get("training/nashconv_sum", 0.0)
                nashconv_data["training_exploitability_max"] = latest_training.get("training/exploitability_max", 0.0)
            if eval_data:
                latest_eval = eval_data[-1] if eval_data else {}
                nashconv_data["evaluation_nashconv_sum"] = latest_eval.get("evaluation/nashconv_sum", 0.0)
                nashconv_data["evaluation_exploitability_max"] = latest_eval.get("evaluation/exploitability_max", 0.0)
        
        consolidated["multi_agent"] = {
            "avg_instance_precision": float(np.mean(instance_precisions)) if instance_precisions else 0.0,
            "avg_instance_coverage": float(np.mean(instance_coverages)) if instance_coverages else 0.0,
            "avg_class_precision": float(np.mean(class_precisions)) if class_precisions else 0.0,  # Union precision
            "avg_class_coverage": float(np.mean(class_coverages)) if class_coverages else 0.0,  # Union coverage
            "avg_class_based_precision": float(np.mean(class_based_precisions)) if class_based_precisions else 0.0,
            "avg_class_based_coverage": float(np.mean(class_based_coverages)) if class_based_coverages else 0.0,
            "unique_rules_count": ma_overall.get("total_unique_rules_instance_based", ma_overall.get("total_unique_rules", 0)),
            "unique_rules_class_based_count": ma_overall.get("total_unique_rules_class_based", 0),
            "total_unique_rules": ma_overall.get("total_unique_rules", 0),
            "avg_rollout_time_seconds": avg_rollout_time,
            "total_rollout_time_seconds": total_rollout_time,
            "nashconv": nashconv_data if nashconv_data else None
        }
    
    # Save to file
    output_file = os.path.join(output_dir, 'comparison_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    logger.info(f"Saved comparison metrics to {output_file}")
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON METRICS SUMMARY (for Excel copying):")
    logger.info(f"{'='*80}")
    
    if consolidated["baseline"]:
        logger.info(f"\nBASELINE (Static Anchors):")
        logger.info(f"  Avg Instance Precision: {consolidated['baseline']['avg_instance_precision']:.4f}")
        logger.info(f"  Avg Instance Coverage:  {consolidated['baseline']['avg_instance_coverage']:.4f}")
    
    if consolidated["single_agent"]:
        sa = consolidated["single_agent"]
        logger.info(f"\nSINGLE-AGENT:")
        logger.info(f"  Avg Instance Precision: {sa['avg_instance_precision']:.4f}")
        logger.info(f"  Avg Instance Coverage:  {sa['avg_instance_coverage']:.4f}")
        logger.info(f"  Avg Class Precision (Union): {sa['avg_class_precision']:.4f}")
        logger.info(f"  Avg Class Coverage (Union):  {sa['avg_class_coverage']:.4f}")
        # Always log class-based metrics if data exists (even if values are 0.0)
        if sa.get('has_class_based_data', False) or sa['unique_rules_class_based_count'] > 0:
            logger.info(f"  Avg Class-Based Precision: {sa['avg_class_based_precision']:.4f}")
            logger.info(f"  Avg Class-Based Coverage:  {sa['avg_class_based_coverage']:.4f}")
        else:
            logger.info(f"  Avg Class-Based Precision: N/A (no class-based data)")
            logger.info(f"  Avg Class-Based Coverage:  N/A (no class-based data)")
        logger.info(f"  Unique Rules (Instance-Based): {sa['unique_rules_count']}")
        logger.info(f"  Unique Rules (Class-Based): {sa['unique_rules_class_based_count']}")
        logger.info(f"  Avg Rollout Time (seconds): {sa['avg_rollout_time_seconds']:.4f}")
        if sa.get('nashconv'):
            nc = sa['nashconv']
            logger.info(f"  NashConv (Training): {nc.get('training_nashconv_sum', 'N/A')}")
            logger.info(f"  NashConv (Evaluation): {nc.get('evaluation_nashconv_sum', 'N/A')}")
    
    if consolidated["multi_agent"]:
        ma = consolidated["multi_agent"]
        logger.info(f"\nMULTI-AGENT:")
        logger.info(f"  Avg Instance Precision: {ma['avg_instance_precision']:.4f}")
        logger.info(f"  Avg Instance Coverage:  {ma['avg_instance_coverage']:.4f}")
        logger.info(f"  Avg Class Precision (Union): {ma['avg_class_precision']:.4f}")
        logger.info(f"  Avg Class Coverage (Union):  {ma['avg_class_coverage']:.4f}")
        if ma['avg_class_based_precision'] > 0 or ma['avg_class_based_coverage'] > 0:
            logger.info(f"  Avg Class-Based Precision: {ma['avg_class_based_precision']:.4f}")
            logger.info(f"  Avg Class-Based Coverage:  {ma['avg_class_based_coverage']:.4f}")
        logger.info(f"  Unique Rules (Instance-Based): {ma['unique_rules_count']}")
        logger.info(f"  Unique Rules (Class-Based): {ma['unique_rules_class_based_count']}")
        logger.info(f"  Avg Rollout Time (seconds): {ma['avg_rollout_time_seconds']:.4f}")
        if ma.get('nashconv'):
            nc = ma['nashconv']
            logger.info(f"  NashConv (Training): {nc.get('training_nashconv_sum', 'N/A')}")
            logger.info(f"  NashConv (Evaluation): {nc.get('evaluation_nashconv_sum', 'N/A')}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Full metrics saved to: {output_file}")
    logger.info(f"{'='*80}\n")


def plot_comprehensive_comparison(
    single_agent_summary: Optional[Dict],
    multi_agent_summary: Optional[Dict],
    output_dir: str,
    dataset_name: str = "",
    baseline_summary: Optional[Dict] = None,
    precision_target: float = 0.95,
    coverage_target: float = 0.1
):
    """Plot comprehensive comparison with 5 subplots (2x3 grid: 2 rows, 3 columns)."""
    if not HAS_PLOTTING:
        return
    
    # Format title with dataset name
    title_prefix = f"{dataset_name.upper()}" if dataset_name else ""
    
    # Create 1x3 grid: Feature importance + Target achievement (removed global metrics)
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    
    # Create axes
    ax1 = fig.add_subplot(gs[0, 0])  # Single-agent feature importance
    ax2 = fig.add_subplot(gs[0, 1])  # Multi-agent feature importance
    ax_eq = fig.add_subplot(gs[0, 2])  # Equilibrium comparison
    
    # Row 1: Feature importance + Target achievement comparison
    plot_feature_importance_subplot(ax1, single_agent_summary, f'{title_prefix}: Single-Agent Feature Importance', top_n=10)
    plot_feature_importance_subplot(ax2, multi_agent_summary, f'{title_prefix}: Multi-Agent Feature Importance', top_n=10)
    plot_equilibrium_comparison(ax_eq, single_agent_summary, multi_agent_summary, precision_target, coverage_target)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comprehensive comparison plot to {output_dir}/comprehensive_comparison.png")
    
    # Also create separate rule overlap comparison plot
    plot_rule_overlap_comparison_per_class(single_agent_summary, multi_agent_summary, output_dir, dataset_name)
    
    # Create NashConv convergence plot (multi-agent only)
    plot_nashconv_comparison(multi_agent_summary, output_dir, dataset_name)


def main():
    parser = argparse.ArgumentParser(
        description="Plot comparison between single-agent and multi-agent results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python plot_comparison.py --single_agent_summary path/to/single_agent/summary.json --multi_agent_summary path/to/multi_agent/summary.json --dataset wine --output_dir output/
        """
    )
    
    parser.add_argument(
        "--single_agent_summary",
        type=str,
        default=None,
        help="Path to single-agent summary JSON file"
    )
    
    parser.add_argument(
        "--multi_agent_summary",
        type=str,
        default=None,
        help="Path to multi-agent summary JSON file"
    )
    
    parser.add_argument(
        "--baseline_results",
        type=str,
        default=None,
        help="Path to baseline results JSON file"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., wine, iris, housing)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load summaries
    single_agent_summary = None
    multi_agent_summary = None
    baseline_summary = None
    
    if args.single_agent_summary:
        logger.info(f"Loading single-agent summary from {args.single_agent_summary}")
        single_agent_summary = load_summary_file(args.single_agent_summary)
        if single_agent_summary:
            logger.info(f"✓ Single-agent summary loaded successfully")
            logger.info(f"  Summary keys: {list(single_agent_summary.keys())}")
            per_class_count = len(single_agent_summary.get('per_class_summary', {}))
            logger.info(f"  Per-class summary entries: {per_class_count}")
            if per_class_count == 0:
                logger.warning("  ⚠ WARNING: Single-agent summary has no per_class_summary data!")
        else:
            logger.warning("⚠ Failed to load single-agent summary (returned None/empty)")
            single_agent_summary = None
    
    if args.multi_agent_summary:
        logger.info(f"Loading multi-agent summary from {args.multi_agent_summary}")
        multi_agent_summary = load_summary_file(args.multi_agent_summary)
        if multi_agent_summary:
            logger.info(f"✓ Multi-agent summary loaded successfully")
            logger.info(f"  Summary keys: {list(multi_agent_summary.keys())}")
            per_class_count = len(multi_agent_summary.get('per_class_summary', {}))
            logger.info(f"  Per-class summary entries: {per_class_count}")
            if per_class_count == 0:
                logger.warning("  ⚠ WARNING: Multi-agent summary has no per_class_summary data!")
                logger.warning("  This could mean:")
                logger.warning("    1. The summary.json file was not generated correctly")
                logger.warning("    2. The inference/test step did not produce any rules")
                logger.warning("    3. The file structure is different than expected")
            
            # Load NashConv metrics if available
            if 'nashconv_metrics' not in multi_agent_summary:
                # Try to load from experiment directory
                ma_summary_path = Path(args.multi_agent_summary)
                if "inference" in str(ma_summary_path):
                    ma_experiment_dir = str(ma_summary_path.parent.parent)
                else:
                    ma_experiment_dir = str(ma_summary_path.parent)
                
                nashconv_data = _load_nashconv_metrics(ma_experiment_dir)
                if nashconv_data.get("available", False):
                    multi_agent_summary["nashconv_metrics"] = nashconv_data
                    logger.info("✓ Loaded NashConv metrics for multi-agent")
        else:
            logger.warning("⚠ Failed to load multi-agent summary (returned None/empty)")
            logger.warning(f"  File path was: {args.multi_agent_summary}")
            logger.warning("  This could mean:")
            logger.warning("    1. The file does not exist")
            logger.warning("    2. The file is empty or invalid JSON")
            logger.warning("    3. The file structure is different than expected")
            multi_agent_summary = None
    
    if args.baseline_results:
        logger.info(f"Loading baseline results from {args.baseline_results}")
        baseline_summary = extract_baseline_metrics(args.baseline_results)
        if baseline_summary:
            logger.info(f"Baseline summary keys: {list(baseline_summary.keys())}")
            logger.info(f"Baseline per_class_summary entries: {len(baseline_summary.get('per_class_summary', {}))}")
        else:
            logger.warning("Could not extract baseline metrics")
    
    # Check that we have at least one summary
    if not any([single_agent_summary, multi_agent_summary, baseline_summary]):
        logger.error("At least one summary file (single-agent, multi-agent, or baseline) must be provided")
        sys.exit(1)
    
    # Generate plots
    if HAS_PLOTTING:
        logger.info("Generating comparison plots...")
        plot_precision_coverage_comparison(
            single_agent_summary, multi_agent_summary, args.output_dir, args.dataset, baseline_summary
        )
        plot_comprehensive_comparison(
            single_agent_summary, multi_agent_summary, args.output_dir, args.dataset, baseline_summary
        )
        logger.info(f"Plots saved to {args.output_dir}")
    else:
        logger.warning("Matplotlib not available. Skipping plot generation.")
    
    # Save consolidated comparison metrics to JSON for easy Excel copying
    logger.info("\nSaving comparison metrics to JSON...")
    save_comparison_metrics_json(
        single_agent_summary, multi_agent_summary, baseline_summary, args.output_dir, args.dataset
    )


if __name__ == "__main__":
    main()

