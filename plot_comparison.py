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


def calculate_equilibrium_metrics(per_class_summary: Dict, precision_target: float = 0.95, coverage_target: float = 0.5) -> Dict:
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
        precision = class_data.get("class_precision", class_data.get("instance_precision", 0.0))
        coverage = class_data.get("class_coverage", class_data.get("instance_coverage", 0.0))
        
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
    
    logger.info(f"Extracting data for classes: {classes}")
    
    for cls in classes:
        # Find single-agent data for this class
        single_data = None
        if single_per_class:
            for class_key, class_data in single_per_class.items():
                if class_data.get("class") == cls:
                    single_data = class_data
                    break
        if single_data:
            # Instance-level
            single_inst_prec = single_data.get("instance_precision", 0.0)
            single_inst_cov = single_data.get("instance_coverage", 0.0)
            single_instance_precisions.append(single_inst_prec)
            single_instance_coverages.append(single_inst_cov)
            # Class-level
            single_cls_prec = single_data.get("class_precision", 0.0)
            single_cls_cov = single_data.get("class_coverage", 0.0)
            single_class_precisions.append(single_cls_prec)
            single_class_coverages.append(single_cls_cov)
            logger.info(f"Single-agent C{cls}: inst_prec={single_inst_prec:.3f}, inst_cov={single_inst_cov:.3f}, class_prec={single_cls_prec:.3f}, class_cov={single_cls_cov:.3f}")
        else:
            logger.warning(f"Single-agent: No data found for class {cls}")
            single_instance_precisions.append(0.0)
            single_instance_coverages.append(0.0)
            single_class_precisions.append(0.0)
            single_class_coverages.append(0.0)
        
        # Find multi-agent data for this class
        multi_data = None
        if multi_per_class:
            for class_key, class_data in multi_per_class.items():
                if class_data.get("class") == cls:
                    multi_data = class_data
                    break
        if multi_data:
            # Instance-level
            multi_inst_prec = multi_data.get("instance_precision", 0.0)
            multi_inst_cov = multi_data.get("instance_coverage", 0.0)
            multi_instance_precisions.append(multi_inst_prec)
            multi_instance_coverages.append(multi_inst_cov)
            # Class-level
            multi_cls_prec = multi_data.get("class_precision", 0.0)
            multi_cls_cov = multi_data.get("class_coverage", 0.0)
            multi_class_precisions.append(multi_cls_prec)
            multi_class_coverages.append(multi_cls_cov)
            logger.info(f"Multi-agent C{cls}: inst_prec={multi_inst_prec:.3f}, inst_cov={multi_inst_cov:.3f}, class_prec={multi_cls_prec:.3f}, class_cov={multi_cls_cov:.3f}")
        else:
            logger.warning(f"Multi-agent: No data found for class {cls}")
            multi_instance_precisions.append(0.0)
            multi_instance_coverages.append(0.0)
            multi_class_precisions.append(0.0)
            multi_class_coverages.append(0.0)
        
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
    
    # Create four subplots: instance-level (top row), class-level (bottom row)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1_inst, ax2_inst, ax1_cls, ax2_cls = axes.flatten()
    
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
    
    # Bottom-left: Class-level Precision (only single-agent and multi-agent)
    bar_idx = 0
    if single_agent_summary:
        ax1_cls.bar(x + offset_cls + bar_idx*width_cls, single_class_precisions, width_cls, 
                   label='Single-Agent', alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
        for i, val in enumerate(single_class_precisions):
            if val > 0.01:
                ax1_cls.text(x[i] + offset_cls + bar_idx*width_cls + width_cls/2, val + 0.02,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        bar_idx += 1
    if multi_agent_summary:
        ax1_cls.bar(x + offset_cls + bar_idx*width_cls, multi_class_precisions, width_cls, 
                   label='Multi-Agent', alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
        for i, val in enumerate(multi_class_precisions):
            if val > 0.01:
                ax1_cls.text(x[i] + offset_cls + bar_idx*width_cls + width_cls/2, val + 0.02,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1_cls.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1_cls.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1_cls.set_title(f'{title_prefix}: Class-Level Precision (Dynamic Anchors Only)', fontsize=13, fontweight='bold')
    ax1_cls.set_xticks(x)
    ax1_cls.set_xticklabels([f'C{cls}' for cls in classes])
    ax1_cls.set_ylim([0, 1.1])
    ax1_cls.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1_cls.legend(fontsize=10, framealpha=0.9)
    
    # Bottom-right: Class-level Coverage (only single-agent and multi-agent)
    bar_idx = 0
    if single_agent_summary:
        ax2_cls.bar(x + offset_cls + bar_idx*width_cls, single_class_coverages, width_cls, 
                   label='Single-Agent', alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
        for i, val in enumerate(single_class_coverages):
            if val > 0.01:
                ax2_cls.text(x[i] + offset_cls + bar_idx*width_cls + width_cls/2, val + 0.02,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        bar_idx += 1
    if multi_agent_summary:
        ax2_cls.bar(x + offset_cls + bar_idx*width_cls, multi_class_coverages, width_cls, 
                   label='Multi-Agent', alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
        for i, val in enumerate(multi_class_coverages):
            if val > 0.01:
                ax2_cls.text(x[i] + offset_cls + bar_idx*width_cls + width_cls/2, val + 0.02,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2_cls.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2_cls.set_ylabel('Coverage', fontsize=12, fontweight='bold')
    ax2_cls.set_title(f'{title_prefix}: Class-Level Coverage (Dynamic Anchors Only)', fontsize=13, fontweight='bold')
    ax2_cls.set_xticks(x)
    ax2_cls.set_xticklabels([f'C{cls}' for cls in classes])
    ax2_cls.set_ylim([0, 1.1])
    ax2_cls.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2_cls.legend(fontsize=10, framealpha=0.9)
    
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


def plot_equilibrium_comparison(ax, single_agent_summary: Optional[Dict], multi_agent_summary: Optional[Dict], 
                                precision_target: float = 0.95, coverage_target: float = 0.5):
    """Plot equilibrium comparison between single-agent and multi-agent."""
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
        ax.text(0.5, 0.5, 'No equilibrium data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Equilibrium Comparison', fontsize=11, fontweight='bold')
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
    
    # Add equilibrium threshold line
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Equilibrium (100%)', alpha=0.7)
    
    # Add value labels and status
    for i, (frac, reached) in enumerate(zip(equilibrium_fractions, equilibrium_reached)):
        label = f'{frac:.1%}'
        if reached:
            label += '\n✓ EQUILIBRIUM'
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
    
    ax.set_ylabel('Equilibrium Fraction', fontsize=10, fontweight='bold')
    ax.set_title('Equilibrium Comparison\n(All classes meet targets: P≥0.95, C≥0.50)', 
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
        """
        Calculate overlap score between two rules based on their intervals.
        
        Since rules are AND conditions, we use the minimum overlap across all common features.
        This correctly reflects that ALL features must overlap for rules to overlap.
        """
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
        feature_overlaps = []
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
                    feature_overlaps.append(intersection / union)
                else:
                    # Zero-width intervals: if they're the same point, perfect overlap (1.0)
                    # Otherwise, different points, no overlap (0.0)
                    if lower1 == upper1 == lower2 == upper2:
                        feature_overlaps.append(1.0)
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
        
        rule1_dict = {feat: (lower, upper) for feat, lower, upper in rule1_intervals}
        rule2_dict = {feat: (lower, upper) for feat, lower, upper in rule2_intervals}
        
        common_features = set(rule1_dict.keys()) & set(rule2_dict.keys())
        if not common_features:
            return 0.0
        
        # Calculate overlap for each common feature
        feature_overlaps = []
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
                    feature_overlaps.append(intersection / union)
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


def plot_comprehensive_comparison(
    single_agent_summary: Optional[Dict],
    multi_agent_summary: Optional[Dict],
    output_dir: str,
    dataset_name: str = "",
    baseline_summary: Optional[Dict] = None
):
    """Plot comprehensive comparison with 5 subplots (2x3 grid: 2 rows, 3 columns)."""
    if not HAS_PLOTTING:
        return
    
    # Format title with dataset name
    title_prefix = f"{dataset_name.upper()}" if dataset_name else ""
    
    # Create 2x3 grid: add equilibrium comparison as 5th subplot
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Create axes
    axes = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    ]
    ax1, ax2, ax_eq = axes[0, 0], axes[0, 1], axes[0, 2]  # Row 1: Feature importance + Equilibrium
    ax3, ax4, ax5 = axes[1, 0], axes[1, 1], axes[1, 2]  # Row 2: Global metrics + empty
    
    # Row 1: Feature importance + Equilibrium comparison
    plot_feature_importance_subplot(ax1, single_agent_summary, f'{title_prefix}: Single-Agent Feature Importance', top_n=10)
    plot_feature_importance_subplot(ax2, multi_agent_summary, f'{title_prefix}: Multi-Agent Feature Importance', top_n=10)
    plot_equilibrium_comparison(ax_eq, single_agent_summary, multi_agent_summary)
    
    # Row 2: Global metrics
    plot_global_metrics_subplot(ax3, single_agent_summary, f'{title_prefix}: Single-Agent Global Metrics')
    plot_global_metrics_subplot(ax4, multi_agent_summary, f'{title_prefix}: Multi-Agent Global Metrics')
    # Hide 6th subplot (bottom right)
    ax5.axis('off')
    
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


if __name__ == "__main__":
    main()

