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


def extract_feature_intervals_from_rule(rule_str: str) -> List[Tuple[str, float, float]]:
    """Extract feature names and intervals (lower, upper) from a rule string."""
    if rule_str == "any values (no tightened features)":
        return []
    
    intervals = []
    # Pattern to match: "feature_name ∈ [lower, upper]"
    pattern = r'(.+?)\s*∈\s*\[([-\d.]+),\s*([-\d.]+)\]'
    
    # Split by " and " to get individual conditions
    conditions = rule_str.split(" and ")
    
    for condition in conditions:
        condition = condition.strip()
        match = re.search(pattern, condition)
        if match:
            feature_name = match.group(1).strip()
            lower = float(match.group(2))
            upper = float(match.group(3))
            intervals.append((feature_name, lower, upper))
    
    return intervals


def load_summary_file(summary_file: str) -> Dict:
    """Load summary JSON file."""
    with open(summary_file, 'r') as f:
        data = json.load(f)
        # Handle nested structure: {"summary": {...}, "test_results": {...}}
        if "summary" in data:
            return data["summary"]
        # If already flat structure, return as-is
        return data


def plot_precision_coverage_comparison(
    single_agent_summary: Dict,
    multi_agent_summary: Dict,
    output_dir: str,
    dataset_name: str = ""
):
    """Plot precision and coverage comparison using grouped bar charts for clarity."""
    if not HAS_PLOTTING:
        return
    
    # Get data
    single_per_class = single_agent_summary.get("per_class_summary", {})
    multi_per_class = multi_agent_summary.get("per_class_summary", {})
    
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
    single_precisions = []
    single_coverages = []
    multi_precisions = []
    multi_coverages = []
    
    logger.info(f"Extracting data for classes: {classes}")
    
    for cls in classes:
        # Find single-agent data for this class
        single_data = None
        for class_key, class_data in single_per_class.items():
            if class_data.get("class") == cls:
                single_data = class_data
                logger.debug(f"Single-agent class {cls}: found in key {class_key}, class_precision={class_data.get('class_precision')}, class_coverage={class_data.get('class_coverage')}")
                break
        if single_data:
            single_prec = single_data.get("class_precision", 0.0)
            single_cov = single_data.get("class_coverage", 0.0)
            single_precisions.append(single_prec)
            single_coverages.append(single_cov)
            logger.info(f"Single-agent C{cls}: precision={single_prec:.3f}, coverage={single_cov:.3f}")
        else:
            logger.warning(f"Single-agent: No data found for class {cls}")
            single_precisions.append(0.0)
            single_coverages.append(0.0)
        
        # Find multi-agent data for this class
        multi_data = None
        for class_key, class_data in multi_per_class.items():
            if class_data.get("class") == cls:
                multi_data = class_data
                logger.debug(f"Multi-agent class {cls}: found in key {class_key}, class_precision={class_data.get('class_precision')}, class_coverage={class_data.get('class_coverage')}")
                break
        if multi_data:
            multi_prec = multi_data.get("class_precision", 0.0)
            multi_cov = multi_data.get("class_coverage", 0.0)
            multi_precisions.append(multi_prec)
            multi_coverages.append(multi_cov)
            logger.info(f"Multi-agent C{cls}: precision={multi_prec:.3f}, coverage={multi_cov:.3f}")
        else:
            logger.warning(f"Multi-agent: No data found for class {cls}")
            multi_precisions.append(0.0)
            multi_coverages.append(0.0)
    
    # Format title with dataset name
    title_prefix = f"{dataset_name.upper()}" if dataset_name else ""
    
    # Create two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    x = np.arange(len(classes))
    width = 0.35
    
    # Left plot: Precision comparison
    bars1 = ax1.bar(x - width/2, single_precisions, width, label='Single-Agent', 
                    alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, multi_precisions, width, label='Multi-Agent', 
                    alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title_prefix}: Class-Level Precision Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'C{cls}' for cls in classes])
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.legend(fontsize=11, framealpha=0.9)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Only label if bar is tall enough
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Right plot: Coverage comparison
    bars3 = ax2.bar(x - width/2, single_coverages, width, label='Single-Agent', 
                    alpha=0.8, color='darkgreen', edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x + width/2, multi_coverages, width, label='Multi-Agent', 
                    alpha=0.8, color='purple', edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Coverage', fontsize=12, fontweight='bold')
    ax2.set_title(f'{title_prefix}: Class-Level Coverage Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'C{cls}' for cls in classes])
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.legend(fontsize=11, framealpha=0.9)
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Only label if bar is tall enough
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_coverage_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved precision-coverage comparison plot to {output_dir}/precision_coverage_comparison.png")


def plot_feature_importance_subplot(ax, summary: Dict, title: str, top_n: int = 10):
    """Plot feature importance on a given axis."""
    per_class = summary.get("per_class_summary", {})
    
    # Collect all feature intervals from all rules
    feature_intervals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    feature_frequency = Counter()
    
    for class_data in per_class.values():
        unique_rules = class_data.get("unique_rules", [])
        for rule_str in unique_rules:
            intervals = extract_feature_intervals_from_rule(rule_str)
            for feature_name, lower, upper in intervals:
                feature_intervals[feature_name].append((lower, upper))
                feature_frequency[feature_name] += 1
    
    if not feature_intervals:
        ax.text(0.5, 0.5, 'No feature data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11, fontweight='bold')
        return
    
    # Calculate importance score
    feature_importance = {}
    for feature_name, intervals_list in feature_intervals.items():
        interval_widths = [upper - lower for lower, upper in intervals_list]
        avg_width = np.mean(interval_widths) if interval_widths else 1.0
        frequency = feature_frequency[feature_name]
        importance_score = frequency / (avg_width + 1e-6)
        feature_importance[feature_name] = {
            "importance": importance_score,
            "frequency": frequency,
            "avg_interval_width": avg_width
        }
    
    # Get top N features
    top_features = sorted(feature_importance.items(), key=lambda x: x[1]["importance"], reverse=True)[:top_n]
    
    if not top_features:
        ax.text(0.5, 0.5, 'No features found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11, fontweight='bold')
        return
    
    features = [f[0] for f in top_features]
    importances = [f[1]["importance"] for f in top_features]
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, alpha=0.8, color='teal', edgecolor='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    ax.set_xlabel('Importance Score', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels
    for i, imp in enumerate(importances):
        ax.text(imp + max(importances) * 0.02, i, f'{imp:.2f}', va='center', fontsize=7)


def plot_global_metrics_subplot(ax, summary: Dict, title: str):
    """Plot global metrics on a given axis."""
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
    
    # Calculate standard deviations
    instance_precisions = [per_class[c]["instance_precision"] for c in per_class]
    instance_coverages = [per_class[c]["instance_coverage"] for c in per_class]
    class_precisions = [per_class[c]["class_precision"] for c in per_class]
    class_coverages = [per_class[c]["class_coverage"] for c in per_class]
    
    std_instance_precision = np.std(instance_precisions) if len(instance_precisions) > 1 else 0.0
    std_instance_coverage = np.std(instance_coverages) if len(instance_coverages) > 1 else 0.0
    std_class_precision = np.std(class_precisions) if len(class_precisions) > 1 else 0.0
    std_class_coverage = np.std(class_coverages) if len(class_coverages) > 1 else 0.0
    
    metrics = ['Instance\nPrecision', 'Instance\nCoverage', 'Class\nPrecision\n(Union)', 'Class\nCoverage\n(Union)']
    values = [global_instance_precision, global_instance_coverage, global_class_precision, global_class_coverage]
    stds = [std_instance_precision, std_instance_coverage, std_class_precision, std_class_coverage]
    colors = ['steelblue', 'coral', 'darkgreen', 'purple']
    
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


def plot_comprehensive_comparison(
    single_agent_summary: Dict,
    multi_agent_summary: Dict,
    output_dir: str,
    dataset_name: str = ""
):
    """Plot comprehensive comparison with 4 subplots."""
    if not HAS_PLOTTING:
        return
    
    # Format title with dataset name
    title_prefix = f"{dataset_name.upper()}" if dataset_name else ""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # Subplot 1: Single-agent feature importance
    plot_feature_importance_subplot(ax1, single_agent_summary, f'{title_prefix}: Single-Agent Feature Importance', top_n=10)
    
    # Subplot 2: Multi-agent feature importance
    plot_feature_importance_subplot(ax2, multi_agent_summary, f'{title_prefix}: Multi-Agent Feature Importance', top_n=10)
    
    # Subplot 3: Single-agent global metrics
    plot_global_metrics_subplot(ax3, single_agent_summary, f'{title_prefix}: Single-Agent Global Metrics')
    
    # Subplot 4: Multi-agent global metrics
    plot_global_metrics_subplot(ax4, multi_agent_summary, f'{title_prefix}: Multi-Agent Global Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comprehensive comparison plot to {output_dir}/comprehensive_comparison.png")


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
        required=True,
        help="Path to single-agent summary JSON file"
    )
    
    parser.add_argument(
        "--multi_agent_summary",
        type=str,
        required=True,
        help="Path to multi-agent summary JSON file"
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
    logger.info(f"Loading single-agent summary from {args.single_agent_summary}")
    single_agent_summary = load_summary_file(args.single_agent_summary)
    logger.info(f"Single-agent summary keys: {list(single_agent_summary.keys())}")
    logger.info(f"Single-agent per_class_summary entries: {len(single_agent_summary.get('per_class_summary', {}))}")
    
    logger.info(f"Loading multi-agent summary from {args.multi_agent_summary}")
    multi_agent_summary = load_summary_file(args.multi_agent_summary)
    logger.info(f"Multi-agent summary keys: {list(multi_agent_summary.keys())}")
    logger.info(f"Multi-agent per_class_summary entries: {len(multi_agent_summary.get('per_class_summary', {}))}")
    
    # Generate plots
    if HAS_PLOTTING:
        logger.info("Generating comparison plots...")
        plot_precision_coverage_comparison(single_agent_summary, multi_agent_summary, args.output_dir, args.dataset)
        plot_comprehensive_comparison(single_agent_summary, multi_agent_summary, args.output_dir, args.dataset)
        logger.info(f"Plots saved to {args.output_dir}")
    else:
        logger.warning("Matplotlib not available. Skipping plot generation.")


if __name__ == "__main__":
    main()

