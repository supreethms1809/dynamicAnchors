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
    single_precisions = []
    single_coverages = []
    multi_precisions = []
    multi_coverages = []
    baseline_precisions = []
    baseline_coverages = []
    
    logger.info(f"Extracting data for classes: {classes}")
    
    for cls in classes:
        # Find single-agent data for this class
        single_data = None
        if single_per_class:
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
        if multi_per_class:
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
        
        # Find baseline data for this class
        baseline_data = None
        if baseline_per_class:
            for class_key, class_data in baseline_per_class.items():
                if class_data.get("class") == cls:
                    baseline_data = class_data
                    break
        if baseline_data:
            baseline_prec = baseline_data.get("class_precision", 0.0)
            baseline_cov = baseline_data.get("class_coverage", 0.0)
            baseline_precisions.append(baseline_prec)
            baseline_coverages.append(baseline_cov)
            logger.info(f"Baseline C{cls}: precision={baseline_prec:.3f}, coverage={baseline_cov:.3f}")
        else:
            baseline_precisions.append(0.0)
            baseline_coverages.append(0.0)
    
    # Format title with dataset name
    title_prefix = f"{dataset_name.upper()}" if dataset_name else ""
    
    # Create two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    x = np.arange(len(classes))
    # Adjust width based on number of methods
    n_methods = sum([bool(single_agent_summary), bool(multi_agent_summary), bool(baseline_summary)])
    width = 0.8 / n_methods if n_methods > 0 else 0.35
    offset = -0.4 + width/2
    
    # Left plot: Precision comparison
    bar_idx = 0
    if single_agent_summary:
        ax1.bar(x + offset + bar_idx*width, single_precisions, width, label='Single-Agent', 
                alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
        bar_idx += 1
    if multi_agent_summary:
        ax1.bar(x + offset + bar_idx*width, multi_precisions, width, label='Multi-Agent', 
                alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
        bar_idx += 1
    if baseline_summary:
        ax1.bar(x + offset + bar_idx*width, baseline_precisions, width, label='Baseline (Static Anchors)', 
                alpha=0.8, color='green', edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title_prefix}: Class-Level Precision Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'C{cls}' for cls in classes])
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.legend(fontsize=11, framealpha=0.9)
    
    # Add value labels on bars
    bar_idx = 0
    if single_agent_summary:
        for i, val in enumerate(single_precisions):
            if val > 0.01:  # Only label if bar is tall enough
                ax1.text(x[i] + offset + bar_idx*width + width/2, val + 0.02,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        bar_idx += 1
    if multi_agent_summary:
        for i, val in enumerate(multi_precisions):
            if val > 0.01:
                ax1.text(x[i] + offset + bar_idx*width + width/2, val + 0.02,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        bar_idx += 1
    if baseline_summary:
        for i, val in enumerate(baseline_precisions):
            if val > 0.01:
                ax1.text(x[i] + offset + bar_idx*width + width/2, val + 0.02,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Right plot: Coverage comparison
    bar_idx = 0
    if single_agent_summary:
        ax2.bar(x + offset + bar_idx*width, single_coverages, width, label='Single-Agent', 
                alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
        bar_idx += 1
    if multi_agent_summary:
        ax2.bar(x + offset + bar_idx*width, multi_coverages, width, label='Multi-Agent', 
                alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
        bar_idx += 1
    if baseline_summary:
        ax2.bar(x + offset + bar_idx*width, baseline_coverages, width, label='Baseline (Static Anchors)', 
                alpha=0.8, color='green', edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Coverage', fontsize=12, fontweight='bold')
    ax2.set_title(f'{title_prefix}: Class-Level Coverage Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'C{cls}' for cls in classes])
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.legend(fontsize=11, framealpha=0.9)
    
    # Add value labels on bars
    bar_idx = 0
    if single_agent_summary:
        for i, val in enumerate(single_coverages):
            if val > 0.01:  # Only label if bar is tall enough
                ax2.text(x[i] + offset + bar_idx*width + width/2, val + 0.02,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        bar_idx += 1
    if multi_agent_summary:
        for i, val in enumerate(multi_coverages):
            if val > 0.01:
                ax2.text(x[i] + offset + bar_idx*width + width/2, val + 0.02,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        bar_idx += 1
    if baseline_summary:
        for i, val in enumerate(baseline_coverages):
            if val > 0.01:
                ax2.text(x[i] + offset + bar_idx*width + width/2, val + 0.02,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_coverage_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved precision-coverage comparison plot to {output_dir}/precision_coverage_comparison.png")


def plot_feature_importance_subplot(ax, summary: Dict, title: str, top_n: int = 10):
    """
    Plot feature importance on a given axis with class-wise breakdown.
    
    Importance Score = frequency / (average_interval_width + ε)
    Normalized to percentages (sum to 100%) for easy interpretation.
    Shows overall importance with stacked bars indicating per-class contributions.
    """
    per_class = summary.get("per_class_summary", {})
    
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
        ax.text(0.5, 0.5, 'No feature data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11, fontweight='bold')
        return
    
    # Calculate global importance scores
    feature_importance_global = {}
    for feature_name, intervals_list in feature_intervals_global.items():
        interval_widths = [upper - lower for lower, upper in intervals_list]
        avg_width = np.mean(interval_widths) if interval_widths else 1.0
        frequency = feature_frequency_global[feature_name]
        importance_score = frequency / (avg_width + 1e-6)
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
            interval_widths = [upper - lower for lower, upper in intervals_list]
            avg_width = np.mean(interval_widths) if interval_widths else 1.0
            frequency = feature_frequency_per_class[target_class][feature_name]
            importance_score = frequency / (avg_width + 1e-6)
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
    total_importance = sum(f["importance"] for f in feature_importance_global.values())
    
    features = [f[0] for f in top_features]
    importances_raw = [f[1]["importance"] for f in top_features]
    importances_pct = [(imp / total_importance * 100) if total_importance > 0 else 0.0 for imp in importances_raw]
    
    # Also normalize per-class importance scores to percentages
    feature_importance_per_class_pct: Dict[int, Dict[str, float]] = {}
    for target_class, class_importance in feature_importance_per_class.items():
        class_total = sum(imp["importance"] for imp in class_importance.values())
        class_pct = {}
        for feat_name, feat_data in class_importance.items():
            raw_imp = feat_data["importance"]
            pct_imp = (raw_imp / class_total * 100) if class_total > 0 else 0.0
            class_pct[feat_name] = pct_imp
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
        """Calculate overlap score between two rules based on their intervals."""
        if not rule1_intervals or not rule2_intervals:
            return 0.0
        
        # Create feature to interval mapping
        rule1_dict = {feat: (lower, upper) for feat, lower, upper in rule1_intervals}
        rule2_dict = {feat: (lower, upper) for feat, lower, upper in rule2_intervals}
        
        # Find common features
        common_features = set(rule1_dict.keys()) & set(rule2_dict.keys())
        if not common_features:
            return 0.0
        
        # Calculate overlap for each common feature
        total_overlap = 0.0
        for feat in common_features:
            lower1, upper1 = rule1_dict[feat]
            lower2, upper2 = rule2_dict[feat]
            
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
                    total_overlap += intersection / union
        
        # Normalize by number of common features
        return total_overlap / len(common_features) if common_features else 0.0
    
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
    
    if len(class_rules) <= 1:
        ax.text(0.5, 0.5, f'Class {class_with_most_rules} has only 1 rule\n(no overlap to show)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return
    
    # Always show top 5 rules for overlap analysis
    # Always show top 5 rules for overlap analysis (already ranked if available)
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
    single_agent_summary: Dict,
    multi_agent_summary: Dict,
    output_dir: str,
    dataset_name: str = "",
    max_rules: int = 5
):
    """Plot rule overlap comparison class-by-class (single vs multi)."""
    if not HAS_PLOTTING:
        return
    
    # Calculate rule overlap helper function
    def calculate_rule_overlap(rule1_intervals: List[Tuple[str, float, float]], 
                               rule2_intervals: List[Tuple[str, float, float]]) -> float:
        """Calculate overlap score between two rules based on their intervals."""
        if not rule1_intervals or not rule2_intervals:
            return 0.0
        
        rule1_dict = {feat: (lower, upper) for feat, lower, upper in rule1_intervals}
        rule2_dict = {feat: (lower, upper) for feat, lower, upper in rule2_intervals}
        
        common_features = set(rule1_dict.keys()) & set(rule2_dict.keys())
        if not common_features:
            return 0.0
        
        total_overlap = 0.0
        for feat in common_features:
            lower1, upper1 = rule1_dict[feat]
            lower2, upper2 = rule2_dict[feat]
            
            intersect_lower = max(lower1, lower2)
            intersect_upper = min(upper1, upper2)
            
            if intersect_lower <= intersect_upper:
                intersection = intersect_upper - intersect_lower
                union_lower = min(lower1, lower2)
                union_upper = max(upper1, upper2)
                union = union_upper - union_lower
                
                if union > 0:
                    total_overlap += intersection / union
        
        return total_overlap / len(common_features) if common_features else 0.0
    
    def plot_overlap_for_class(ax, class_rules: List[str], class_num: int, title: str, per_class_data: Optional[Dict] = None):
        """Plot overlap heatmap for a class on given axis."""
        if len(class_rules) <= 1:
            ax.text(0.5, 0.5, f'Class {class_num} has only 1 rule\n(no overlap to show)', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')
            return
        
        # Always show top 5 rules for overlap analysis (already ranked if per_class_data has ranked_rules)
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
    sa_per_class = single_agent_summary.get("per_class_summary", {})
    ma_per_class = multi_agent_summary.get("per_class_summary", {})
    
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
                break
        
        for class_data in ma_per_class.values():
            if class_data.get("class", -1) == target_class:
                ma_class_data = class_data
                # Prefer ranked rules if available
                ma_class_rules = class_data.get("ranked_unique_rules", [])
                if not ma_class_rules:
                    ma_class_rules = class_data.get("unique_rules", [])
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


def plot_comprehensive_comparison(
    single_agent_summary: Optional[Dict],
    multi_agent_summary: Optional[Dict],
    output_dir: str,
    dataset_name: str = "",
    baseline_summary: Optional[Dict] = None
):
    """Plot comprehensive comparison with 4 subplots (2x2 grid)."""
    if not HAS_PLOTTING:
        return
    
    # Format title with dataset name
    title_prefix = f"{dataset_name.upper()}" if dataset_name else ""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2 = axes[0, 0], axes[0, 1]  # Row 1: Feature importance
    ax3, ax4 = axes[1, 0], axes[1, 1]  # Row 2: Global metrics
    
    # Row 1: Feature importance
    plot_feature_importance_subplot(ax1, single_agent_summary, f'{title_prefix}: Single-Agent Feature Importance', top_n=10)
    plot_feature_importance_subplot(ax2, multi_agent_summary, f'{title_prefix}: Multi-Agent Feature Importance', top_n=10)
    
    # Row 2: Global metrics
    plot_global_metrics_subplot(ax3, single_agent_summary, f'{title_prefix}: Single-Agent Global Metrics')
    plot_global_metrics_subplot(ax4, multi_agent_summary, f'{title_prefix}: Multi-Agent Global Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comprehensive comparison plot to {output_dir}/comprehensive_comparison.png")
    
    # Also create separate rule overlap comparison plot
    plot_rule_overlap_comparison_per_class(single_agent_summary, multi_agent_summary, output_dir, dataset_name)


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
        logger.info(f"Single-agent summary keys: {list(single_agent_summary.keys())}")
        logger.info(f"Single-agent per_class_summary entries: {len(single_agent_summary.get('per_class_summary', {}))}")
    
    if args.multi_agent_summary:
        logger.info(f"Loading multi-agent summary from {args.multi_agent_summary}")
        multi_agent_summary = load_summary_file(args.multi_agent_summary)
        logger.info(f"Multi-agent summary keys: {list(multi_agent_summary.keys())}")
        logger.info(f"Multi-agent per_class_summary entries: {len(multi_agent_summary.get('per_class_summary', {}))}")
    
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


if __name__ == "__main__":
    main()

