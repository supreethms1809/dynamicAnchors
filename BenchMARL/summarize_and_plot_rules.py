#!/usr/bin/env python3
"""
Summarize and Plot Multi-Agent Rule Results

This script analyzes extracted rules and test results from multi-agent dynamic anchors
experiments and generates comprehensive summaries and visualizations.

Usage:
    python summarize_and_plot_rules.py --rules_file <path_to_extracted_rules.json> --dataset <dataset_name> [options]
    python summarize_and_plot_rules.py --experiment_dir <path_to_experiment_dir> --dataset <dataset_name> [options]

Example:
    python summarize_and_plot_rules.py --experiment_dir winemaddpg_anchor_mlp__149590d2_25_11_28-11_33_03 --dataset wine
    python summarize_and_plot_rules.py --rules_file path/to/extracted_rules.json --dataset wine --run_tests
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

# Import test function
from test_extracted_rules import test_rules_from_json, parse_rule

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


def load_rules_file(rules_file: str) -> Dict:
    """Load extracted rules from JSON file."""
    with open(rules_file, 'r') as f:
        return json.load(f)


def summarize_rules_from_json(rules_data: Dict) -> Dict:
    """Extract summary statistics from rules JSON."""
    per_class_results = rules_data.get("per_class_results", {})
    
    summary = {
        "n_classes": len(per_class_results),
        "classes": [],
        "per_class_summary": {},
        "overall_stats": {}
    }
    
    all_precisions = []
    all_coverages = []
    all_rule_counts = []
    all_unique_rule_counts = []
    feature_frequency = Counter()
    
    for class_key, class_data in per_class_results.items():
        target_class = class_data.get("class", -1)
        summary["classes"].append(target_class)
        
        # Extract metrics - handle both formats:
        # Multi-agent: instance_precision, instance_coverage, class_precision, class_coverage
        # Single-agent: precision, coverage (no class-level metrics)
        instance_precision = class_data.get("instance_precision") or class_data.get("precision", 0.0)
        instance_coverage = class_data.get("instance_coverage") or class_data.get("coverage", 0.0)
        class_precision = class_data.get("class_precision", 0.0)
        class_coverage = class_data.get("class_coverage", 0.0)
        
        # For single-agent, if class-level metrics are missing, use instance-level as fallback
        if class_precision == 0.0 and instance_precision > 0.0:
            class_precision = instance_precision
        if class_coverage == 0.0 and instance_coverage > 0.0:
            class_coverage = instance_coverage
        
        # Extract rules
        unique_rules = class_data.get("unique_rules", [])
        all_rules = class_data.get("rules", [])
        
        # Count features in rules
        for rule_str in unique_rules:
            features = extract_features_from_rule(rule_str)
            feature_frequency.update(features)
        
        class_summary = {
            "class": int(target_class),
            "agent": class_data.get("group", "unknown"),
            "instance_precision": float(instance_precision),
            "instance_coverage": float(instance_coverage),
            "class_precision": float(class_precision),
            "class_coverage": float(class_coverage),
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
        
        summary["per_class_summary"][class_key] = class_summary
        
        # Collect for overall stats
        all_precisions.extend([instance_precision, class_precision])
        all_coverages.extend([instance_coverage, class_coverage])
        all_rule_counts.append(len(all_rules))
        all_unique_rule_counts.append(len(unique_rules))
    
    summary["overall_stats"] = {
        "mean_instance_precision": float(np.mean([s["instance_precision"] for s in summary["per_class_summary"].values()])),
        "mean_instance_coverage": float(np.mean([s["instance_coverage"] for s in summary["per_class_summary"].values()])),
        "mean_class_precision": float(np.mean([s["class_precision"] for s in summary["per_class_summary"].values()])),
        "mean_class_coverage": float(np.mean([s["class_coverage"] for s in summary["per_class_summary"].values()])),
        "total_unique_rules": int(sum([s["n_unique_rules"] for s in summary["per_class_summary"].values()])),
        "mean_unique_rules_per_class": float(np.mean(all_unique_rule_counts)),
        "feature_frequency": dict(feature_frequency.most_common())
    }
    
    summary["classes"] = sorted(summary["classes"])
    
    return summary


def plot_metrics_comparison(summary: Dict, output_dir: str, dataset_name: str = ""):
    """Plot precision and coverage metrics comparison."""
    if not HAS_PLOTTING:
        return
    
    per_class = summary["per_class_summary"]
    classes = sorted(per_class.keys(), key=lambda x: per_class[x]["class"])
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    class_nums = [per_class[c]["class"] for c in classes]
    instance_prec = [per_class[c]["instance_precision"] for c in classes]
    instance_cov = [per_class[c]["instance_coverage"] for c in classes]
    class_prec = [per_class[c]["class_precision"] for c in classes]
    class_cov = [per_class[c]["class_coverage"] for c in classes]
    
    x = np.arange(len(class_nums))
    width = 0.35
    
    # Format title with dataset name
    title_prefix = f"Multi-Agent - {dataset_name.upper()}" if dataset_name else "Multi-Agent"
    
    # Instance-level: precision and coverage in one plot
    axes[0].bar(x - width/2, instance_prec, width, label='Precision', alpha=0.7, color='steelblue')
    axes[0].bar(x + width/2, instance_cov, width, label='Coverage', alpha=0.7, color='coral')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].set_title(f'{title_prefix}: Instance-Level Precision and Coverage per Class', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_nums)
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].legend(fontsize=11)
    # Add value labels
    for i, (prec, cov) in enumerate(zip(instance_prec, instance_cov)):
        axes[0].text(i - width/2, prec + 0.02, f'{prec:.3f}', ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, cov + 0.02, f'{cov:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Class-level: precision and coverage in one plot
    axes[1].bar(x - width/2, class_prec, width, label='Precision', alpha=0.7, color='darkgreen')
    axes[1].bar(x + width/2, class_cov, width, label='Coverage', alpha=0.7, color='purple')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title(f'{title_prefix}: Class-Level Precision and Coverage per Class', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_nums)
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend(fontsize=11)
    # Add value labels
    for i, (prec, cov) in enumerate(zip(class_prec, class_cov)):
        axes[1].text(i - width/2, prec + 0.02, f'{prec:.3f}', ha='center', va='bottom', fontsize=9)
        axes[1].text(i + width/2, cov + 0.02, f'{cov:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved metrics comparison plot to {output_dir}/metrics_comparison.png")


def plot_precision_vs_coverage(summary: Dict, output_dir: str, dataset_name: str = ""):
    """Plot precision vs coverage scatter plot."""
    if not HAS_PLOTTING:
        return
    
    per_class = summary["per_class_summary"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get colors for classes - sort by class number for consistent coloring
    classes = sorted(per_class.keys(), key=lambda x: per_class[x]["class"])
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    # Format title with dataset name
    title_prefix = f"Multi-Agent - {dataset_name.upper()}" if dataset_name else "Multi-Agent"
    
    # Instance-level
    for idx, class_key in enumerate(classes):
        class_data = per_class[class_key]
        cls = class_data["class"]
        ax.scatter(
            class_data["instance_coverage"],
            class_data["instance_precision"],
            s=200, alpha=0.7, color=colors[idx], 
            marker='o', label=f'Instance-Level Class {cls}',
            edgecolors='black', linewidths=1.5
        )
    
    # Class-level
    for idx, class_key in enumerate(classes):
        class_data = per_class[class_key]
        cls = class_data["class"]
        ax.scatter(
            class_data["class_coverage"],
            class_data["class_precision"],
            s=200, alpha=0.7, color=colors[idx],
            marker='s', label=f'Class-Level Class {cls}',
            edgecolors='black', linewidths=1.5
        )
    
    ax.set_xlabel('Coverage', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'{title_prefix}: Precision vs Coverage (Instance-Level and Class-Level)', fontsize=14)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([0.7, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best', ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_vs_coverage.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved precision vs coverage plot to {output_dir}/precision_vs_coverage.png")


def plot_rule_counts(summary: Dict, output_dir: str, dataset_name: str = ""):
    """Plot rule counts per class."""
    if not HAS_PLOTTING:
        return
    
    per_class = summary["per_class_summary"]
    classes = sorted(per_class.keys(), key=lambda x: per_class[x]["class"])
    
    class_nums = [per_class[c]["class"] for c in classes]
    unique_counts = [per_class[c]["n_unique_rules"] for c in classes]
    total_counts = [per_class[c]["n_total_rules"] for c in classes]
    
    x = np.arange(len(class_nums))
    width = 0.35
    
    # Format title with dataset name
    title_prefix = f"Multi-Agent - {dataset_name.upper()}" if dataset_name else "Multi-Agent"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, unique_counts, width, label='Unique Rules', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, total_counts, width, label='Total Rules', alpha=0.8, color='coral')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Rules', fontsize=12)
    ax.set_title(f'{title_prefix}: Rule Counts per Class', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_nums)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rule_counts.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved rule counts plot to {output_dir}/rule_counts.png")


def plot_feature_frequency(summary: Dict, output_dir: str, dataset_name: str = "", top_n: int = 15):
    """Plot most frequently used features in rules."""
    if not HAS_PLOTTING:
        return
    
    feature_freq = summary["overall_stats"]["feature_frequency"]
    if not feature_freq:
        logger.warning("No feature frequency data available.")
        return
    
    # Get top N features
    top_features = sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, counts = zip(*top_features) if top_features else ([], [])
    
    if not features:
        return
    
    # Format title with dataset name
    title_prefix = f"Multi-Agent - {dataset_name.upper()}" if dataset_name else "Multi-Agent"
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(features))
    
    ax.barh(y_pos, counts, alpha=0.7, color='teal')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Frequency (Number of Rules)', fontsize=12)
    ax.set_title(f'{title_prefix}: Top {top_n} Most Frequently Used Features in Rules', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, count in enumerate(counts):
        ax.text(count + 0.5, i, f'{count}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature frequency plot to {output_dir}/feature_frequency.png")


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
    title_prefix = f"Multi-Agent - {dataset_name.upper()}" if dataset_name else "Multi-Agent"
    
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
    title_prefix = f"Multi-Agent - {dataset_name.upper()}" if dataset_name else "Multi-Agent"
    
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
        axes[1].text(cls, missed + max(missed_counts) * 0.02, f'{missed}/{total}', ha='center', va='bottom')
    
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
    title_prefix = f"Multi-Agent - {dataset_name.upper()}" if dataset_name else "Multi-Agent"
    
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
        f.write("MULTI-AGENT DYNAMIC ANCHORS RULE ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
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
        f.write(f"Mean instance-level precision: {summary['overall_stats']['mean_instance_precision']:.4f}\n")
        f.write(f"Mean instance-level coverage: {summary['overall_stats']['mean_instance_coverage']:.4f}\n")
        f.write(f"Mean class-level precision: {summary['overall_stats']['mean_class_precision']:.4f}\n")
        f.write(f"Mean class-level coverage: {summary['overall_stats']['mean_class_coverage']:.4f}\n\n")
        
        # Per-class details
        f.write("PER-CLASS DETAILS\n")
        f.write("-"*80 + "\n")
        for class_key in sorted(summary['per_class_summary'].keys(), 
                               key=lambda x: summary['per_class_summary'][x]['class']):
            class_data = summary['per_class_summary'][class_key]
            f.write(f"\nClass {class_data['class']} (Agent: {class_data.get('agent', 'unknown')}):\n")
            f.write(f"  Instance-level precision: {class_data['instance_precision']:.4f}\n")
            f.write(f"  Instance-level coverage: {class_data['instance_coverage']:.4f}\n")
            f.write(f"  Class-level precision: {class_data['class_precision']:.4f}\n")
            f.write(f"  Class-level coverage: {class_data['class_coverage']:.4f}\n")
            f.write(f"  Total rules: {class_data['n_total_rules']}\n")
            f.write(f"  Unique rules: {class_data['n_unique_rules']}\n")
        
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


def main():
    parser = argparse.ArgumentParser(
        description="Summarize and plot multi-agent rule results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze from experiment directory
  python summarize_and_plot_rules.py --experiment_dir winemaddpg_anchor_mlp__149590d2_25_11_28-11_33_03 --dataset wine
  
  # Analyze from rules file directly
  python summarize_and_plot_rules.py --rules_file path/to/extracted_rules.json --dataset wine
  
  # Run tests and then analyze
  python summarize_and_plot_rules.py --rules_file path/to/extracted_rules.json --dataset wine --run_tests
        """
    )
    
    parser.add_argument(
        "--rules_file",
        type=str,
        default=None,
        help="Path to extracted_rules.json file"
    )
    
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Path to experiment directory (will look for inference/extracted_rules.json)"
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
        help="Run test_extracted_rules.py before generating plots"
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
            # Try relative to current directory and BenchMARL directory
            if Path("BenchMARL").exists():
                experiment_dir = Path("BenchMARL") / experiment_dir
        rules_file = experiment_dir / "inference" / "extracted_rules.json"
        if not rules_file.exists():
            raise FileNotFoundError(f"Rules file not found: {rules_file}")
    elif args.rules_file:
        rules_file = Path(args.rules_file)
        if not rules_file.is_absolute():
            rules_file = Path.cwd() / rules_file
        experiment_dir = rules_file.parent.parent  # Assume inference/extracted_rules.json structure
    else:
        raise ValueError("Must provide either --rules_file or --experiment_dir")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(experiment_dir) if args.experiment_dir else Path.cwd()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Rules file: {rules_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load rules
    logger.info("Loading extracted rules...")
    rules_data = load_rules_file(str(rules_file))
    
    # Summarize rules
    logger.info("Summarizing rules...")
    summary = summarize_rules_from_json(rules_data)
    
    # Optionally run tests
    test_results = None
    if args.run_tests:
        logger.info("Running test_extracted_rules.py...")
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
        plot_precision_vs_coverage(summary, str(output_dir), args.dataset)
        plot_rule_counts(summary, str(output_dir), args.dataset)
        plot_feature_frequency(summary, str(output_dir), args.dataset)
        
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
    
    logger.info(f"\n{'='*80}")
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()

