#!/usr/bin/env python3
"""
Summarize and Plot Single-Agent Rule Results

This script analyzes extracted rules and test results from single-agent dynamic anchors
experiments and generates comprehensive summaries and visualizations.

Usage:
    python summarize_and_plot_rules_single.py --rules_file <path_to_extracted_rules_single_agent.json> --dataset <dataset_name> [options]
    python summarize_and_plot_rules_single.py --experiment_dir <path_to_experiment_dir> --dataset <dataset_name> [options]

Example:
    python summarize_and_plot_rules_single.py --experiment_dir output/single_agent_sb3_wine_ddpg/training/ddpg_single_agent_sb3_25_11_29-17_42_44 --dataset wine
    python summarize_and_plot_rules_single.py --rules_file path/to/extracted_rules_single_agent.json --dataset wine --run_tests
"""

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'BenchMARL'))

import json
import argparse
import numpy as np
import re
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
from pathlib import Path
import logging

# Import test function from single-agent test script
from test_extracted_rules_single import test_rules_from_json, parse_rule

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    logger.warning("matplotlib or seaborn not available. Plotting will be disabled.")
    HAS_PLOTTING = False

# Set style
if HAS_PLOTTING:
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10


def extract_features_from_rule(rule_str: str) -> List[str]:
    """Extract feature names from a rule string."""
    if rule_str == "any values (no tightened features)":
        return []
    
    # Split by " and " to get individual conditions
    conditions = rule_str.split(" and ")
    features = []
    
    # Pattern to match: "feature_name ∈ [lower, upper]"
    pattern = r'^(.+?)\s*∈\s*\['
    
    for condition in conditions:
        condition = condition.strip()
        match = re.match(pattern, condition)
        if match:
            feature_name = match.group(1).strip()
            features.append(feature_name)
    
    return features


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


def load_rules_file(rules_file: str) -> Dict:
    """Load extracted rules from JSON file."""
    with open(rules_file, 'r') as f:
        return json.load(f)


def summarize_rules_from_json(rules_data: Dict) -> Dict:
    """Extract summary statistics from single-agent rules JSON."""
    per_class_results = rules_data.get("per_class_results", {})
    
    summary = {
        "n_classes": len(per_class_results),
        "classes": [],
        "per_class_summary": {},
        "overall_stats": {},
        "model_type": "single_agent_sb3"
    }
    
    # Get metadata if available
    metadata = rules_data.get("metadata", {})
    if metadata:
        summary["algorithm"] = metadata.get("algorithm", "unknown")
        summary["model_type"] = metadata.get("model_type", "single_agent_sb3")
    
    all_precisions = []
    all_coverages = []
    all_class_precisions = []
    all_class_coverages = []
    all_rule_counts = []
    all_unique_rule_counts = []
    feature_frequency = Counter()
    
    for class_key, class_data in per_class_results.items():
        target_class = class_data.get("class", -1)
        summary["classes"].append(target_class)
        
        # Extract instance-level metrics (average across all instances)
        instance_precision = class_data.get("instance_precision", class_data.get("precision", 0.0))
        instance_coverage = class_data.get("instance_coverage", class_data.get("coverage", 0.0))
        
        # Extract class-level metrics (union of all anchors for this class)
        class_precision = class_data.get("class_precision", 0.0)
        class_coverage = class_data.get("class_coverage", 0.0)
        
        # CRITICAL: Do NOT fallback class-level metrics to instance-level metrics!
        # These are fundamentally different:
        # - Instance-level: average precision/coverage across individual anchors
        # - Class-level: precision/coverage of the UNION of all anchors
        # If class-level metrics are missing (None), default to 0.0, not instance-level values.
        # If class-level metrics are 0.0, it means the union covers no samples or has issues,
        # and we should report this accurately.
        
        # Warn if class-level metrics are 0.0 but instance-level are > 0.0 (potential bug)
        if (class_precision == 0.0 or class_coverage == 0.0) and (instance_precision > 0.0 or instance_coverage > 0.0):
            n_anchors = class_data.get("n_episodes", class_data.get("n_instances", 0))
            logger.warning(
                f"  ⚠ Class {target_class}: Class-level metrics are 0.0 but instance-level are > 0.0. "
                f"This may indicate the union computation has issues. "
                f"(instance_prec={instance_precision:.3f}, instance_cov={instance_coverage:.3f}, "
                f"class_prec={class_precision:.3f}, class_cov={class_coverage:.3f}, n_anchors={n_anchors})"
            )
        
        # Extract rules
        unique_rules = class_data.get("unique_rules", [])
        all_rules = class_data.get("rules", [])
        
        # Count features in rules
        for rule_str in unique_rules:
            features = extract_features_from_rule(rule_str)
            feature_frequency.update(features)
        
        class_summary = {
            "class": int(target_class),
            # Instance-level metrics
            "instance_precision": float(instance_precision),
            "instance_coverage": float(instance_coverage),
            # Class-level metrics
            "class_precision": float(class_precision),
            "class_coverage": float(class_coverage),
            # Legacy fields for backward compatibility
            "precision": float(instance_precision),
            "coverage": float(instance_coverage),
            "n_total_rules": len(all_rules),
            "n_unique_rules": len(unique_rules),
            "unique_rules": unique_rules
        }
        
        # Add stds if available
        if "instance_precision_std" in class_data:
            class_summary["instance_precision_std"] = float(class_data.get("instance_precision_std", 0.0))
        if "instance_coverage_std" in class_data:
            class_summary["instance_coverage_std"] = float(class_data.get("instance_coverage_std", 0.0))
        if "class_precision_std" in class_data:
            class_summary["class_precision_std"] = float(class_data.get("class_precision_std", 0.0))
        if "class_coverage_std" in class_data:
            class_summary["class_coverage_std"] = float(class_data.get("class_coverage_std", 0.0))
        # Legacy std fields
        if "precision_std" in class_data:
            class_summary["precision_std"] = float(class_data.get("precision_std", 0.0))
        if "coverage_std" in class_data:
            class_summary["coverage_std"] = float(class_data.get("coverage_std", 0.0))
        
        # Add n_episodes if available
        if "n_episodes" in class_data:
            class_summary["n_episodes"] = int(class_data.get("n_episodes", 0))
        
        summary["per_class_summary"][class_key] = class_summary
        
        # Collect for overall stats (using instance-level for overall stats)
        all_precisions.append(instance_precision)
        all_coverages.append(instance_coverage)
        all_class_precisions.append(class_precision)
        all_class_coverages.append(class_coverage)
        all_rule_counts.append(len(all_rules))
        all_unique_rule_counts.append(len(unique_rules))
    
    summary["overall_stats"] = {
        # Instance-level overall stats
        "mean_precision": float(np.mean(all_precisions)) if all_precisions else 0.0,
        "mean_coverage": float(np.mean(all_coverages)) if all_coverages else 0.0,
        "std_precision": float(np.std(all_precisions)) if len(all_precisions) > 1 else 0.0,
        "std_coverage": float(np.std(all_coverages)) if len(all_coverages) > 1 else 0.0,
        # Class-level overall stats
        "mean_class_precision": float(np.mean(all_class_precisions)) if all_class_precisions else 0.0,
        "mean_class_coverage": float(np.mean(all_class_coverages)) if all_class_coverages else 0.0,
        "std_class_precision": float(np.std(all_class_precisions)) if len(all_class_precisions) > 1 else 0.0,
        "std_class_coverage": float(np.std(all_class_coverages)) if len(all_class_coverages) > 1 else 0.0,
        "total_unique_rules": int(sum(all_unique_rule_counts)),
        "mean_unique_rules_per_class": float(np.mean(all_unique_rule_counts)) if all_unique_rule_counts else 0.0,
        "feature_frequency": dict(feature_frequency.most_common())
    }
    
    summary["classes"] = sorted(summary["classes"])
    
    return summary


def plot_metrics_comparison(summary: Dict, output_dir: str, dataset_name: str = ""):
    """Plot all metrics (precision and coverage, instance and class-level) in a single grouped bar chart."""
    if not HAS_PLOTTING:
        return
    
    per_class = summary["per_class_summary"]
    classes = sorted(per_class.keys(), key=lambda x: per_class[x]["class"])
    
    if not classes:
        logger.warning("No class data available for metrics comparison.")
        return
    
    # Extract data
    class_nums = [per_class[c]["class"] for c in classes]
    instance_prec = [per_class[c]["instance_precision"] for c in classes]
    instance_cov = [per_class[c]["instance_coverage"] for c in classes]
    class_prec = [per_class[c]["class_precision"] for c in classes]
    class_cov = [per_class[c]["class_coverage"] for c in classes]
    
    # Format title with dataset name
    title_prefix = f"Single-Agent - {dataset_name.upper()}" if dataset_name else "Single-Agent"
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(class_nums))
    width = 0.2  # Thin bars for 4 metrics per class
    
    # Plot all 4 metrics side by side for each class
    bars1 = ax.bar(x - 1.5*width, instance_prec, width, label='Instance Precision', 
                   alpha=0.8, color='steelblue', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x - 0.5*width, instance_cov, width, label='Instance Coverage', 
                   alpha=0.8, color='coral', edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + 0.5*width, class_prec, width, label='Class Precision (Union)', 
                   alpha=0.8, color='darkgreen', edgecolor='black', linewidth=1)
    bars4 = ax.bar(x + 1.5*width, class_cov, width, label='Class Coverage (Union)', 
                   alpha=0.8, color='purple', edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'{title_prefix}: Metrics Comparison per Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_nums)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(fontsize=10, loc='best', framealpha=0.9, ncol=2)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:  # Only label if bar is tall enough
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved metrics comparison plot to {output_dir}/metrics_comparison.png")


def plot_precision_coverage_tradeoff(summary: Dict, output_dir: str, dataset_name: str = ""):
    """Plot precision vs coverage trade-off scatter plot for all classes."""
    if not HAS_PLOTTING:
        return
    
    per_class = summary["per_class_summary"]
    classes = sorted(per_class.keys(), key=lambda x: per_class[x]["class"])
    
    if not classes:
        logger.warning("No class data available for precision-coverage trade-off plot.")
        return
    
    # Format title with dataset name
    title_prefix = f"Single-Agent - {dataset_name.upper()}" if dataset_name else "Single-Agent"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Get colors for classes
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    # Left plot: Instance-level precision vs coverage
    for idx, class_key in enumerate(classes):
        class_data = per_class[class_key]
        cls = class_data["class"]
        ax1.scatter(
            class_data["instance_coverage"],
            class_data["instance_precision"],
            s=300, alpha=0.8, color=colors[idx], 
            marker='o', label=f'Class {cls}',
            edgecolors='black', linewidths=2, zorder=3
        )
        # Add class label
        ax1.annotate(f'C{cls}', 
                    (class_data["instance_coverage"], class_data["instance_precision"]),
                    xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black'))
    
    ax1.set_xlabel('Coverage', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title_prefix}: Instance-Level Precision vs Coverage Trade-off', fontsize=14, fontweight='bold')
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([0.7, 1.05])
    ax1.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    # Add diagonal reference line (ideal: high precision, high coverage)
    ax1.plot([0, 1], [1, 1], 'k--', alpha=0.3, linewidth=1, label='Ideal (Precision=1.0)')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Right plot: Class-level (union) precision vs coverage
    for idx, class_key in enumerate(classes):
        class_data = per_class[class_key]
        cls = class_data["class"]
        ax2.scatter(
            class_data["class_coverage"],
            class_data["class_precision"],
            s=300, alpha=0.8, color=colors[idx],
            marker='s', label=f'Class {cls}',
            edgecolors='black', linewidths=2, zorder=3
        )
        # Add class label
        ax2.annotate(f'C{cls}', 
                    (class_data["class_coverage"], class_data["class_precision"]),
                    xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black'))
    
    ax2.set_xlabel('Coverage', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title(f'{title_prefix}: Class-Level (Union) Precision vs Coverage Trade-off', fontsize=14, fontweight='bold')
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([0.7, 1.05])
    ax2.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    # Add diagonal reference line (ideal: high precision, high coverage)
    ax2.plot([0, 1], [1, 1], 'k--', alpha=0.3, linewidth=1, label='Ideal (Precision=1.0)')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_coverage_tradeoff.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved precision-coverage trade-off plot to {output_dir}/precision_coverage_tradeoff.png")


def plot_global_metrics(summary: Dict, output_dir: str, dataset_name: str = ""):
    """Plot all global metrics (precision and coverage, instance and class-level) in a single plot."""
    if not HAS_PLOTTING:
        return
    
    per_class = summary["per_class_summary"]
    overall_stats = summary.get("overall_stats", {})
    
    if not per_class:
        logger.warning("No class data available for global metrics.")
        return
    
    # Format title with dataset name
    title_prefix = f"Single-Agent - {dataset_name.upper()}" if dataset_name else "Single-Agent"
    
    # Calculate global metrics (mean across classes)
    global_instance_precision = overall_stats.get("mean_precision", 
        np.mean([per_class[c]["instance_precision"] for c in per_class]))
    global_instance_coverage = overall_stats.get("mean_coverage",
        np.mean([per_class[c]["instance_coverage"] for c in per_class]))
    global_class_precision = overall_stats.get("mean_class_precision",
        np.mean([per_class[c]["class_precision"] for c in per_class]))
    global_class_coverage = overall_stats.get("mean_class_coverage",
        np.mean([per_class[c]["class_coverage"] for c in per_class]))
    
    # Calculate standard deviations
    instance_precisions = [per_class[c]["instance_precision"] for c in per_class]
    instance_coverages = [per_class[c]["instance_coverage"] for c in per_class]
    class_precisions = [per_class[c]["class_precision"] for c in per_class]
    class_coverages = [per_class[c]["class_coverage"] for c in per_class]
    
    std_instance_precision = np.std(instance_precisions) if len(instance_precisions) > 1 else 0.0
    std_instance_coverage = np.std(instance_coverages) if len(instance_coverages) > 1 else 0.0
    std_class_precision = np.std(class_precisions) if len(class_precisions) > 1 else 0.0
    std_class_coverage = np.std(class_coverages) if len(class_coverages) > 1 else 0.0
    
    # Single plot with all 4 metrics
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metrics = ['Instance\nPrecision', 'Instance\nCoverage', 'Class\nPrecision\n(Union)', 'Class\nCoverage\n(Union)']
    values = [global_instance_precision, global_instance_coverage, global_class_precision, global_class_coverage]
    stds = [std_instance_precision, std_instance_coverage, std_class_precision, std_class_coverage]
    colors = ['steelblue', 'coral', 'darkgreen', 'purple']
    
    x = np.arange(len(metrics))
    bars = ax.bar(x, values, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5,
                  yerr=stds, capsize=10, error_kw={'elinewidth': 2, 'capthick': 2})
    
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'{title_prefix}: Global Metrics Summary', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for i, (val, std) in enumerate(zip(values, stds)):
        label = f'{val:.3f}'
        if std > 0:
            label += f'\n±{std:.3f}'
        ax.text(i, val + std + 0.05, label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'global_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved global metrics plot to {output_dir}/global_metrics.png")


def plot_feature_importance(summary: Dict, output_dir: str, dataset_name: str = "", top_n: int = 15):
    """
    Plot feature importance based on frequency and interval selectivity.
    
    Importance Score Formula: 
    - Raw Importance = frequency / (average_interval_width + ε)
    - Percentage = (Raw Importance / Total Importance) × 100%
    
    Interpretation:
    - Higher frequency = feature appears in more rules (more commonly used)
    - Narrower intervals (lower avg_width) = more selective/precise feature usage
    - Raw importance combines both: features that are frequently used AND selective score higher
    - Percentage: Normalized to sum to 100% across all features for easy interpretation
      (e.g., "Feature X accounts for 25% of total importance")
    
    Example:
    - Feature A: freq=10, width=0.05 → raw=200 → 25%
    - Feature B: freq=20, width=0.20 → raw=100 → 12.5%
    - Feature A is more important (more selective) and accounts for 25% of total importance
    """
    if not HAS_PLOTTING:
        return
    
    per_class = summary["per_class_summary"]
    
    # Collect feature intervals per class and globally
    feature_intervals_global: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    feature_intervals_per_class: Dict[int, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    feature_frequency_global = Counter()
    feature_frequency_per_class: Dict[int, Counter] = defaultdict(Counter)
    
    for class_key, class_data in per_class.items():
        target_class = class_data.get("class", -1)
        unique_rules = class_data.get("unique_rules", [])
        
        for rule_str in unique_rules:
            intervals = extract_feature_intervals_from_rule(rule_str)
            for feature_name, lower, upper in intervals:
                # Global collection
                feature_intervals_global[feature_name].append((lower, upper))
                feature_frequency_global[feature_name] += 1
                
                # Per-class collection
                feature_intervals_per_class[target_class][feature_name].append((lower, upper))
                feature_frequency_per_class[target_class][feature_name] += 1
    
    if not feature_intervals_global:
        logger.warning("No feature intervals found for importance analysis.")
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
            "avg_interval_width": avg_width,
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
                "avg_interval_width": avg_width,
            }
        feature_importance_per_class[target_class] = class_importance
    
    # Get all features sorted by importance, then take top N (or all if fewer than top_n)
    all_features_sorted = sorted(feature_importance_global.items(), key=lambda x: x[1]["importance"], reverse=True)
    n_available = len(all_features_sorted)
    n_to_show = min(top_n, n_available)  # Show top N, or all if fewer than N
    top_features = all_features_sorted[:n_to_show]
    
    if not top_features:
        return
    
    # Log how many features we're showing
    if n_available <= top_n:
        logger.info(f"Showing all {n_available} available features in feature importance plot")
    else:
        logger.info(f"Showing top {top_n} of {n_available} available features in feature importance plot")
    
    # Normalize importance scores to percentages (sum to 100%)
    # Calculate sum of all importance scores (not just top N, for proper normalization)
    total_importance = sum(f["importance"] for f in feature_importance_global.values())
    
    # Validate total_importance
    if not np.isfinite(total_importance) or total_importance <= 0:
        logger.warning(
            f"Invalid total_importance: {total_importance}. "
            f"All feature importance percentages will be set to 0."
        )
        total_importance = 1.0  # Use 1.0 to avoid division by zero, but all percentages will be 0
    
    # Normalize top features to percentages
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
    frequencies = [f[1]["frequency"] for f in top_features]
    avg_widths = [f[1]["avg_interval_width"] for f in top_features]
    
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
    
    # Format title with dataset name
    title_prefix = f"Single-Agent - {dataset_name.upper()}" if dataset_name else "Single-Agent"
    
    # Helper function to sort rules for consistent ordering in plots
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
    
    # Get classes with multiple rules for overlap visualization
    # Prefer ranked rules if available (from test_extracted_rules), otherwise use unique_rules
    classes_with_rules = []
    for class_key, class_data in per_class.items():
        target_class = class_data.get("class", -1)
        # Check for ranked rules first (from test_extracted_rules ranking)
        ranked_rules = class_data.get("ranked_unique_rules", [])
        if not ranked_rules:
            # Fallback to unique_rules if ranked rules not available
            ranked_rules = class_data.get("unique_rules", [])
        
        # Sort rules for consistent plotting
        ranked_rules = sort_rules_for_plotting(class_data, ranked_rules)
        
        if len(ranked_rules) > 1:
            classes_with_rules.append((target_class, ranked_rules))
    
    n_overlap_classes = len(classes_with_rules)
    
    # Create figure: 2x2 for main plots, then additional rows for rule overlap per class
    if n_overlap_classes == 0:
        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    elif n_overlap_classes <= 2:
        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    else:
        # Add extra rows for additional classes
        n_extra_rows = (n_overlap_classes - 2 + 1) // 2  # How many extra 2-class rows needed
        fig = plt.figure(figsize=(24, 12 + 6 * n_extra_rows))
        gs = fig.add_gridspec(2 + n_extra_rows, 2, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Overall feature importance (scatter + bar)
    ax1 = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(features))
    
    # Scatter plot: frequency vs selectivity (use raw importance for color/size)
    scatter = ax1.scatter(frequencies, [1.0/(w + 1e-6) for w in avg_widths], 
                          s=[imp * 50 for imp in importances_raw], alpha=0.6, c=importances_pct, 
                          cmap='viridis', edgecolors='black', linewidths=1)
    
    # Add feature labels
    for i, (feat, freq, inv_width) in enumerate(zip(features, frequencies, [1.0/(w + 1e-6) for w in avg_widths])):
        ax1.annotate(feat, (freq, inv_width), fontsize=7, ha='left', va='center')
    
    ax1.set_xlabel('Frequency (Number of Rules)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Interval Selectivity (1/Avg Width)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{title_prefix}: Overall Feature Importance\n(Frequency vs Selectivity)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(scatter, ax=ax1, label='Importance (%)')
    
    # Subplot 2: Per-class importance breakdown (heatmap) - in percentages
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Get all classes
    classes = sorted(feature_importance_per_class_pct.keys())
    n_classes = len(classes)
    
    if n_classes == 0:
        ax2.text(0.5, 0.5, 'No class-wise data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'{title_prefix}: Class-Wise Feature Importance', fontsize=12, fontweight='bold')
    else:
        # Prepare data for heatmap: feature (rows) x class (columns)
        # Each cell shows the importance % of that feature for that class
        heatmap_data = []
        for feat in features:
            row = []
            for cls in classes:
                class_imp_pct = feature_importance_per_class_pct[cls].get(feat, 0.0)
                row.append(class_imp_pct)
            heatmap_data.append(row)
        
        heatmap_array = np.array(heatmap_data)
        
        # Create heatmap
        im = ax2.imshow(heatmap_array, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        # Set ticks and labels
        ax2.set_xticks(np.arange(len(classes)))
        ax2.set_xticklabels([f'Class {cls}' for cls in classes], fontsize=9)
        ax2.set_yticks(np.arange(len(features)))
        ax2.set_yticklabels(features, fontsize=8)
        
        # Add text annotations with percentages
        for i in range(len(features)):
            for j in range(len(classes)):
                value = heatmap_array[i, j]
                text_color = 'white' if value > np.max(heatmap_array) * 0.5 else 'black'
                ax2.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                        color=text_color, fontsize=7, fontweight='bold')
        
        ax2.set_xlabel('Class', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Feature', fontsize=11, fontweight='bold')
        ax2.set_title(f'{title_prefix}: Class-Wise Feature Importance\n(% per Class)', fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Importance (%)', fontsize=10, fontweight='bold')
    
    # Subplot 3: Overall importance bar chart with details (percentages)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.barh(y_pos, importances_pct, alpha=0.8, color='teal', edgecolor='black', linewidth=1)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(features, fontsize=9)
    ax3.set_xlabel('Global Importance (%)', fontsize=11, fontweight='bold')
    if n_available <= top_n:
        ax3.set_title(f'{title_prefix}: All {n_available} Features (Overall, %)', fontsize=12, fontweight='bold')
    else:
        ax3.set_title(f'{title_prefix}: Top {top_n} Features (Overall, %)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax3.set_xlim([0, max(importances_pct) * 1.1 if importances_pct else 100])
    
    # Add value labels with percentage
    for i, (imp_pct, freq, width) in enumerate(zip(importances_pct, frequencies, avg_widths)):
        label = f'{imp_pct:.1f}%\n(f:{freq}, w:{width:.3f})'
        ax3.text(imp_pct + max(importances_pct) * 0.02 if importances_pct else 0, i, label, va='center', fontsize=7)
    
    # Subplot 4: Rule overlap per class - create a grid for all classes
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
                logger.warning(f"Invalid interval in rule1: {feat} has lower={lower} > upper={upper}. Swapping.")
                lower, upper = upper, lower
            # Only keep first occurrence if duplicate features exist
            if feat not in rule1_dict:
                rule1_dict[feat] = (lower, upper)
        
        rule2_dict = {}
        for feat, lower, upper in rule2_intervals:
            # Validate interval: lower <= upper
            if lower > upper:
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
    
    # Plot rule overlap for each class
    if n_overlap_classes == 0:
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.text(0.5, 0.5, 'No classes with multiple rules\n(no overlap to show)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        ax4.set_title(f'{title_prefix}: Rule Overlap', fontsize=11, fontweight='bold')
    else:
        # Create overlap axes: first 2 classes in bottom right, rest in additional rows
        overlap_axes = []
        if n_overlap_classes >= 1:
            if n_overlap_classes == 1:
                overlap_axes.append(fig.add_subplot(gs[1, 1]))
            else:
                # First 2 classes in bottom right (split vertically)
                gs_overlap1 = gs[1, 1].subgridspec(2, 1, hspace=0.4)
                overlap_axes.append(fig.add_subplot(gs_overlap1[0]))
                overlap_axes.append(fig.add_subplot(gs_overlap1[1]))
        
        # Additional classes in new rows (2 per row)
        for extra_idx in range(2, n_overlap_classes):
            row = 2 + (extra_idx - 2) // 2
            col = (extra_idx - 2) % 2
            overlap_axes.append(fig.add_subplot(gs[row, col]))
        
        # Plot overlap for each class
        for idx, (target_class, class_rules) in enumerate(classes_with_rules):
            if idx >= len(overlap_axes):
                break
            ax_overlap = overlap_axes[idx]
            
            # Always show top 5 rules for overlap analysis
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
            im = ax_overlap.imshow(overlap_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
            
            # Set ticks and labels
            ax_overlap.set_xticks(np.arange(n_rules_to_show))
            ax_overlap.set_yticks(np.arange(n_rules_to_show))
            ax_overlap.set_xticklabels([f"R{i+1}" for i in range(n_rules_to_show)], fontsize=6, rotation=45, ha='right')
            ax_overlap.set_yticklabels([f"R{i+1}" for i in range(n_rules_to_show)], fontsize=6)
            
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
                        ax_overlap.text(j, i, text, ha='center', va='center', 
                                color=text_color, fontsize=font_size, fontweight='bold')
            
            ax_overlap.set_xlabel('Rule', fontsize=8, fontweight='bold')
            ax_overlap.set_ylabel('Rule', fontsize=8, fontweight='bold')
            ax_overlap.set_title(f'Class {target_class} ({len(class_rules)} rules, showing {n_rules_to_show})', 
                             fontsize=9, fontweight='bold')
            
            # Add colorbar only for first subplot
            if idx == 0:
                cbar = plt.colorbar(im, ax=ax_overlap)
                cbar.set_label('Overlap', fontsize=7, fontweight='bold')
    
    plt.suptitle(f'{title_prefix}: Feature Importance Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature importance plot to {output_dir}/feature_importance.png")


def plot_test_results_overlap(test_results: Dict, output_dir: str, dataset_name: str = ""):
    """Plot overlap analysis from test results."""
    if not HAS_PLOTTING:
        return
    
    overlap_analysis = test_results.get("overlap_analysis", {})
    if not overlap_analysis:
        logger.warning("No overlap analysis data available.")
        return
    
    class_pair_overlaps = overlap_analysis.get("class_pair_overlaps", {})
    if not class_pair_overlaps:
        logger.info("No class pair overlaps found.")
        return
    
    # Create overlap matrix
    classes = sorted(test_results.get("classes", []))
    n_classes = len(classes)
    overlap_matrix = np.zeros((n_classes, n_classes))
    
    for pair_key, pair_info in class_pair_overlaps.items():
        class1 = pair_info["class1"]
        class2 = pair_info["class2"]
        n_overlaps = pair_info["n_overlapping_rules"]
        
        idx1 = classes.index(class1)
        idx2 = classes.index(class2)
        overlap_matrix[idx1, idx2] = n_overlaps
        overlap_matrix[idx2, idx1] = n_overlaps
    
    # Format title with dataset name
    title_prefix = f"Single-Agent - {dataset_name.upper()}" if dataset_name else "Single-Agent"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(overlap_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, int(overlap_matrix[i, j]),
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Class', fontsize=12)
    ax.set_title(f'{title_prefix}: Rule Overlap Between Classes', fontsize=14)
    plt.colorbar(im, ax=ax, label='Number of Overlapping Rules')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rule_overlap_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved overlap matrix plot to {output_dir}/rule_overlap_matrix.png")


def plot_test_results_coverage(test_results: Dict, output_dir: str, dataset_name: str = ""):
    """Plot coverage analysis from test results."""
    if not HAS_PLOTTING:
        return
    
    missed_analysis = test_results.get("missed_samples_analysis", {})
    if not missed_analysis:
        logger.warning("No missed samples analysis available.")
        return
    
    per_class_analysis = missed_analysis.get("per_class_analysis", {})
    if not per_class_analysis:
        return
    
    classes = sorted([int(k.split('_')[1]) for k in per_class_analysis.keys()])
    coverage_ratios = []
    missed_counts = []
    total_counts = []
    
    for cls in classes:
        class_key = f"class_{cls}"
        if class_key in per_class_analysis:
            analysis = per_class_analysis[class_key]
            coverage_ratios.append(analysis.get("coverage_ratio", 0.0))
            missed_counts.append(analysis.get("n_missed_samples", 0))
            total_counts.append(analysis.get("n_class_samples", 0))
    
    # Format title with dataset name
    title_prefix = f"Single-Agent - {dataset_name.upper()}" if dataset_name else "Single-Agent"
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Coverage ratio
    axes[0].bar(classes, coverage_ratios, alpha=0.7, color='green')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Coverage Ratio', fontsize=12)
    axes[0].set_title(f'{title_prefix}: Test Data Coverage Ratio per Class', fontsize=14)
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(True, alpha=0.3, axis='y')
    for cls, ratio in zip(classes, coverage_ratios):
        axes[0].text(cls, ratio + 0.02, f'{ratio:.3f}', ha='center', va='bottom')
    
    # Missed samples
    axes[1].bar(classes, missed_counts, alpha=0.7, color='red')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Number of Missed Samples', fontsize=12)
    axes[1].set_title(f'{title_prefix}: Missed Samples per Class (Test Data)', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    for cls, missed, total in zip(classes, missed_counts, total_counts):
        axes[1].text(cls, missed + max(missed_counts) * 0.02 if missed_counts else 0.1, 
                    f'{missed}/{total}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_coverage_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved test coverage analysis plot to {output_dir}/test_coverage_analysis.png")


def plot_rule_precision_vs_coverage_test(test_results: Dict, output_dir: str, dataset_name: str = ""):
    """Plot rule-level precision vs coverage from test results."""
    if not HAS_PLOTTING:
        return
    
    rule_results = test_results.get("rule_results", [])
    if not rule_results:
        logger.warning("No rule results available.")
        return
    
    classes = sorted(test_results.get("classes", []))
    
    # Format title with dataset name
    title_prefix = f"Single-Agent - {dataset_name.upper()}" if dataset_name else "Single-Agent"
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each class with different color
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    for class_idx, target_class in enumerate(classes):
        precisions = []
        coverages = []
        
        for rule_result in rule_results:
            class_key = f"class_{target_class}"
            if class_key in rule_result.get("per_class_results", {}):
                class_res = rule_result["per_class_results"][class_key]
                prec = class_res.get("rule_precision", 0.0)
                cov = class_res.get("rule_coverage", 0.0)
                if prec > 0 or cov > 0:  # Only plot if rule matches any samples
                    precisions.append(prec)
                    coverages.append(cov)
        
        if precisions:
            ax.scatter(coverages, precisions, s=100, alpha=0.6, 
                      color=colors[class_idx], label=f'Class {target_class}')
    
    ax.set_xlabel('Rule Coverage', fontsize=12)
    ax.set_ylabel('Rule Precision', fontsize=12)
    ax.set_title(f'{title_prefix}: Rule-Level Precision vs Coverage (Test Data)', fontsize=14)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rule_precision_vs_coverage_test.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved rule precision vs coverage test plot to {output_dir}/rule_precision_vs_coverage_test.png")


def generate_summary_report(summary: Dict, test_results: Optional[Dict] = None, output_file: str = "summary_report.txt"):
    """Generate a text summary report."""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SINGLE-AGENT DYNAMIC ANCHORS RULE ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Model info
        model_type = summary.get("model_type", "single_agent_sb3")
        algorithm = summary.get("algorithm", "unknown")
        f.write("MODEL INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Algorithm: {algorithm.upper()}\n\n")
        
        # Basic stats
        f.write("BASIC STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of classes: {summary['n_classes']}\n")
        f.write(f"Classes: {summary['classes']}\n")
        f.write(f"Total unique rules across all classes: {summary['overall_stats']['total_unique_rules']}\n")
        f.write(f"Mean unique rules per class: {summary['overall_stats']['mean_unique_rules_per_class']:.2f}\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-"*80 + "\n")
        f.write("Instance-Level (Average Across Instances):\n")
        f.write(f"  Mean precision: {summary['overall_stats']['mean_precision']:.4f} ± {summary['overall_stats']['std_precision']:.4f}\n")
        f.write(f"  Mean coverage: {summary['overall_stats']['mean_coverage']:.4f} ± {summary['overall_stats']['std_coverage']:.4f}\n")
        f.write("Class-Level (Union of All Anchors):\n")
        f.write(f"  Mean precision: {summary['overall_stats']['mean_class_precision']:.4f} ± {summary['overall_stats']['std_class_precision']:.4f}\n")
        f.write(f"  Mean coverage: {summary['overall_stats']['mean_class_coverage']:.4f} ± {summary['overall_stats']['std_class_coverage']:.4f}\n\n")
        
        # Per-class details
        f.write("PER-CLASS DETAILS\n")
        f.write("-"*80 + "\n")
        for class_key in sorted(summary['per_class_summary'].keys(), 
                               key=lambda x: summary['per_class_summary'][x]['class']):
            class_data = summary['per_class_summary'][class_key]
            f.write(f"\nClass {class_data['class']}:\n")
            f.write(f"  Instance-Level (Average Across Instances):\n")
            f.write(f"    Precision: {class_data['instance_precision']:.4f}")
            if "instance_precision_std" in class_data:
                f.write(f" ± {class_data['instance_precision_std']:.4f}")
            f.write("\n")
            f.write(f"    Coverage: {class_data['instance_coverage']:.4f}")
            if "instance_coverage_std" in class_data:
                f.write(f" ± {class_data['instance_coverage_std']:.4f}")
            f.write("\n")
            f.write(f"  Class-Level (Union of All Anchors):\n")
            f.write(f"    Precision: {class_data['class_precision']:.4f}")
            if "class_precision_std" in class_data:
                f.write(f" ± {class_data['class_precision_std']:.4f}")
            f.write("\n")
            f.write(f"    Coverage: {class_data['class_coverage']:.4f}")
            if "class_coverage_std" in class_data:
                f.write(f" ± {class_data['class_coverage_std']:.4f}")
            f.write("\n")
            f.write(f"  Total rules: {class_data['n_total_rules']}\n")
            f.write(f"  Unique rules: {class_data['n_unique_rules']}\n")
            if "n_episodes" in class_data:
                f.write(f"  Episodes evaluated: {class_data['n_episodes']}\n")
        
        # Feature frequency
        f.write("\n\nTOP 10 MOST FREQUENTLY USED FEATURES\n")
        f.write("-"*80 + "\n")
        feature_freq = summary['overall_stats']['feature_frequency']
        top_features = sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (feature, count) in enumerate(top_features, 1):
            f.write(f"{i:2d}. {feature}: {count} rules\n")
        
        # Test results if available
        if test_results:
            f.write("\n\n" + "="*80 + "\n")
            f.write("TEST RESULTS ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Dataset: {test_results.get('dataset', 'unknown')}\n")
            f.write(f"Data type: {test_results.get('data_type', 'unknown')}\n")
            f.write(f"Number of test samples: {test_results.get('n_samples', 0)}\n")
            f.write(f"Rules tested: {test_results.get('rules_tested', 0)}\n\n")
            
            # Overlap analysis
            overlap_analysis = test_results.get("overlap_analysis", {})
            if overlap_analysis:
                f.write("RULE OVERLAP ANALYSIS\n")
                f.write("-"*80 + "\n")
                summary_overlap = overlap_analysis.get("summary", {})
                f.write(f"Total unique rules: {summary_overlap.get('total_unique_rules', 0)}\n")
                f.write(f"Rules with overlaps: {summary_overlap.get('rules_with_overlaps', 0)}\n")
                f.write(f"Total overlap pairs: {summary_overlap.get('total_overlap_pairs', 0)}\n\n")
            
            # Missed samples
            missed_analysis = test_results.get("missed_samples_analysis", {})
            if missed_analysis:
                f.write("MISSED SAMPLES ANALYSIS\n")
                f.write("-"*80 + "\n")
                summary_missed = missed_analysis.get("summary", {})
                f.write(f"Total samples: {summary_missed.get('total_samples', 0)}\n")
                f.write(f"Covered samples: {summary_missed.get('total_covered_samples', 0)}\n")
                f.write(f"Missed samples: {summary_missed.get('total_missed_samples', 0)}\n")
                f.write(f"Overall coverage ratio: {summary_missed.get('overall_coverage_ratio', 0.0):.4f}\n\n")
                
                # Per-class missed samples
                per_class_missed = missed_analysis.get("per_class_analysis", {})
                for class_key in sorted(per_class_missed.keys(), 
                                       key=lambda x: per_class_missed[x]['class']):
                    class_analysis = per_class_missed[class_key]
                    cls = class_analysis['class']
                    f.write(f"Class {cls}:\n")
                    f.write(f"  Coverage ratio: {class_analysis['coverage_ratio']:.4f}\n")
                    f.write(f"  Covered: {class_analysis['n_covered_samples']}/{class_analysis['n_class_samples']}\n")
                    f.write(f"  Missed: {class_analysis['n_missed_samples']}\n")
    
    logger.info(f"Saved summary report to {output_file}")


def save_consolidated_metrics(summary: Dict, dataset_name: str, output_file: str, experiment_dir: Optional[str] = None):
    """
    Save consolidated metrics JSON for easy copying.
    Includes: dataset name, instance metrics, class metrics, global metrics, and wandb run link.
    """
    overall_stats = summary.get("overall_stats", {})
    
    # Try to get wandb run link
    wandb_run_url = None
    
    # First try: Check if wandb.run is active
    try:
        import wandb
        if wandb.run is not None:
            # Get wandb run URL
            wandb_run_url = wandb.run.url if hasattr(wandb.run, 'url') else None
            if wandb_run_url is None:
                # Alternative method to get URL
                try:
                    entity = wandb.run.entity if hasattr(wandb.run, 'entity') else None
                    project = wandb.run.project if hasattr(wandb.run, 'project') else None
                    run_id = wandb.run.id if hasattr(wandb.run, 'id') else None
                    if entity and project and run_id:
                        wandb_run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
                except Exception:
                    pass
    except (ImportError, AttributeError):
        pass
    
    # Second try: Read from saved file if experiment_dir is provided
    if wandb_run_url is None and experiment_dir:
        try:
            wandb_url_file = os.path.join(experiment_dir, "wandb_run_url.txt")
            if os.path.exists(wandb_url_file):
                with open(wandb_url_file, 'r') as f:
                    wandb_run_url = f.read().strip()
                logger.debug(f"Read wandb URL from file: {wandb_url_file}")
        except Exception:
            pass
    
    # Build consolidated metrics
    consolidated = {
        "dataset": dataset_name,
        "wandb_run_url": wandb_run_url,
        "model_type": summary.get("model_type", "single_agent_sb3"),
        "algorithm": summary.get("algorithm", "unknown"),
        "n_classes": summary.get("n_classes", 0),
        "classes": summary.get("classes", []),
        "metrics": {
            "instance_level": {
                "mean_precision": overall_stats.get("mean_precision", overall_stats.get("mean_instance_precision", 0.0)),
                "mean_coverage": overall_stats.get("mean_coverage", overall_stats.get("mean_instance_coverage", 0.0)),
                "std_precision": overall_stats.get("std_precision", overall_stats.get("std_instance_precision", 0.0)),
                "std_coverage": overall_stats.get("std_coverage", overall_stats.get("std_instance_coverage", 0.0)),
            },
            "class_level": {
                "mean_precision": overall_stats.get("mean_class_precision", 0.0),
                "mean_coverage": overall_stats.get("mean_class_coverage", 0.0),
                "std_precision": overall_stats.get("std_class_precision", 0.0),
                "std_coverage": overall_stats.get("std_class_coverage", 0.0),
            },
            "global": {
                "total_unique_rules": overall_stats.get("total_unique_rules", 0),
                "mean_unique_rules_per_class": overall_stats.get("mean_unique_rules_per_class", 0.0),
            }
        },
        "per_class": {}
    }
    
    # Add per-class metrics
    per_class_summary = summary.get("per_class_summary", {})
    for class_key, class_data in per_class_summary.items():
        cls = class_data.get("class", -1)
        consolidated["per_class"][f"class_{cls}"] = {
            "instance_precision": class_data.get("instance_precision", 0.0),
            "instance_coverage": class_data.get("instance_coverage", 0.0),
            "class_precision": class_data.get("class_precision", 0.0),
            "class_coverage": class_data.get("class_coverage", 0.0),
            "n_unique_rules": class_data.get("n_unique_rules", 0),
        }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    logger.info(f"Saved consolidated metrics to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize and plot single-agent rule results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze from experiment directory
  python summarize_and_plot_rules_single.py --experiment_dir output/single_agent_sb3_wine_ddpg/training/ddpg_single_agent_sb3_25_11_29-17_42_44 --dataset wine
  
  # Analyze from rules file directly
  python summarize_and_plot_rules_single.py --rules_file path/to/extracted_rules_single_agent.json --dataset wine
  
  # Run tests and then analyze
  python summarize_and_plot_rules_single.py --rules_file path/to/extracted_rules_single_agent.json --dataset wine --run_tests
        """
    )
    
    parser.add_argument(
        "--rules_file",
        type=str,
        default=None,
        help="Path to extracted_rules_single_agent.json file"
    )
    
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Path to experiment directory (will look for inference/extracted_rules_single_agent.json)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., wine, iris, breast_cancer)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots and summary (default: same as experiment_dir or current dir)"
    )
    
    parser.add_argument(
        "--run_tests",
        action="store_true",
        help="Run test_extracted_rules_single.py before generating plots"
    )
    
    parser.add_argument(
        "--test_results_file",
        type=str,
        default=None,
        help="Path to saved test results JSON file (alternative to running tests)"
    )
    
    parser.add_argument(
        "--use_train_data",
        action="store_true",
        help="Use training data for testing (if --run_tests is used)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset loading (default: 42)"
    )
    
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip generating plots (only generate summary report)"
    )
    
    args = parser.parse_args()
    
    # Determine rules file path
    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
        if not experiment_dir.is_absolute():
            # Try relative to current directory and single_agent directory
            if Path("single_agent").exists():
                experiment_dir = Path.cwd() / experiment_dir
        rules_file = experiment_dir / "inference" / "extracted_rules_single_agent.json"
        if not rules_file.exists():
            raise FileNotFoundError(f"Rules file not found: {rules_file}")
    elif args.rules_file:
        rules_file = Path(args.rules_file)
        if not rules_file.is_absolute():
            rules_file = Path.cwd() / rules_file
        experiment_dir = rules_file.parent.parent  # Assume inference/extracted_rules_single_agent.json structure
    else:
        raise ValueError("Must provide either --rules_file or --experiment_dir")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(experiment_dir) if args.experiment_dir else Path.cwd()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store experiment_dir as string for later use (if available)
    # experiment_dir is defined in both code paths above
    experiment_dir_str = str(experiment_dir) if 'experiment_dir' in locals() and experiment_dir else None
    
    logger.info(f"Rules file: {rules_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load rules
    logger.info("Loading extracted rules...")
    rules_data = load_rules_file(str(rules_file))
    
    # Summarize rules
    logger.info("Summarizing rules...")
    summary = summarize_rules_from_json(rules_data)
    # Add dataset name to summary
    summary["dataset"] = args.dataset
    
    # Optionally run tests
    test_results = None
    if args.run_tests:
        logger.info("Running test_extracted_rules_single.py...")
        test_results = test_rules_from_json(
            rules_file=str(rules_file),
            dataset_name=args.dataset,
            use_test_data=not args.use_train_data,
            seed=args.seed
        )
        # Save test results
        test_results_file = output_dir / "test_results.json"
        with open(test_results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Saved test results to {test_results_file}")
    elif args.test_results_file:
        logger.info(f"Loading test results from {args.test_results_file}...")
        with open(args.test_results_file, 'r') as f:
            test_results = json.load(f)
    
    # Generate plots
    if not args.no_plots and HAS_PLOTTING:
        logger.info("Generating plots...")
        plot_metrics_comparison(summary, str(output_dir), args.dataset)
        plot_precision_coverage_tradeoff(summary, str(output_dir), args.dataset)
        plot_global_metrics(summary, str(output_dir), args.dataset)
        plot_feature_importance(summary, str(output_dir), args.dataset)
        
        if test_results:
            plot_test_results_overlap(test_results, str(output_dir), args.dataset)
            plot_test_results_coverage(test_results, str(output_dir), args.dataset)
            plot_rule_precision_vs_coverage_test(test_results, str(output_dir), args.dataset)
    
    # Generate summary report
    logger.info("Generating summary report...")
    report_file = output_dir / "summary_report.txt"
    generate_summary_report(summary, test_results, str(report_file))
    
    # Save summary JSON
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "summary": summary,
            "test_results": test_results
        }, f, indent=2)
    logger.info(f"Saved summary JSON to {summary_file}")
    
    # Save consolidated metrics JSON (easy to copy)
    consolidated_metrics_file = output_dir / "consolidated_metrics.json"
    # experiment_dir_str already defined above
    save_consolidated_metrics(summary, args.dataset, str(consolidated_metrics_file), experiment_dir_str)
    logger.info(f"Saved consolidated metrics to {consolidated_metrics_file}")
    
    logger.info(f"\n{'='*80}")
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()

