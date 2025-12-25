#!/usr/bin/env python3
"""
Analyze Classifier Rules: Extract General Rules for Class Classification

This script answers the question:
"Given the dataset and trained classifier, what is the set of general rules 
that is classifying the samples as a particular class?"

Usage:
    python analyze_classifier_rules.py \
        --rules_file extracted_rules.json \
        --dataset breast_cancer \
        --output_dir analysis_output
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClassifierRulesAnalyzer:
    """Analyze extracted rules to answer: What rules classify samples as a particular class?"""
    
    def __init__(self, rules_file: str, dataset_name: str, output_dir: str):
        self.rules_file = Path(rules_file)
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load rules
        logger.info(f"Loading rules from {rules_file}")
        with open(self.rules_file, 'r') as f:
            self.rules_data = json.load(f)
        
        self.per_class_results = self.rules_data.get("per_class_results", {})
    
    def extract_rules_for_class(self, target_class: int) -> Dict[str, Any]:
        """Extract all rules for a specific class."""
        class_key = f"class_{target_class}"
        
        if class_key not in self.per_class_results:
            logger.warning(f"No rules found for class {target_class}")
            return {}
        
        class_data = self.per_class_results[class_key]
        
        # Get instance-based rules
        instance_rules = class_data.get("unique_rules", [])
        
        # Get class-based rules if available
        class_based_key = f"class_{target_class}_class_based"
        class_based_rules = []
        if class_based_key in self.per_class_results:
            class_based_data = self.per_class_results[class_based_key]
            class_based_rules = class_based_data.get("unique_rules", [])
        
        # Get class union rules (union of class-based anchors - smallest set of general rules)
        # These are the rules that form the class_precision/class_coverage union metrics
        class_union_rules = class_data.get("class_level_unique_rules", [])
        # Fallback: if not in main entry, check separate class_based entry
        if not class_union_rules and class_based_key in self.per_class_results:
            class_based_data = self.per_class_results[class_based_key]
            class_union_rules = class_based_data.get("unique_rules", [])
        
        # For multi-agent: also check class_based_results nested structure
        if not class_union_rules and "class_based_results" in class_data:
            union_rules_set = set()
            for agent_result in class_data["class_based_results"].values():
                agent_rules = agent_result.get("unique_rules", [])
                union_rules_set.update(agent_rules)
            class_union_rules = list(union_rules_set)
        
        # Get metrics
        instance_precision = class_data.get("instance_precision", 0.0)
        instance_coverage = class_data.get("instance_coverage", 0.0)
        class_precision = class_data.get("class_precision", 0.0)  # Union precision (class-based union)
        class_coverage = class_data.get("class_coverage", 0.0)    # Union coverage (class-based union)
        
        return {
            "class": target_class,
            "instance_based_rules": instance_rules,
            "class_based_rules": class_based_rules,
            "class_union_rules": class_union_rules,  # Union of class-based rules (smallest general set)
            "all_rules": list(set(instance_rules + class_based_rules)),
            "n_instance_rules": len(instance_rules),
            "n_class_based_rules": len(class_based_rules),
            "n_class_union_rules": len(class_union_rules),
            "n_total_rules": len(set(instance_rules + class_based_rules)),
            "instance_precision": instance_precision,
            "instance_coverage": instance_coverage,
            "class_precision": class_precision,  # Union precision (from class-based union)
            "class_coverage": class_coverage,    # Union coverage (from class-based union)
        }
    
    def analyze_rule_features(self, rules: List[str]) -> Dict[str, Any]:
        """
        Analyze which features are most important in rules.
        
        Importance is based on:
        1. Frequency: How often the feature appears in rules
        2. Interval width: Narrower intervals indicate more selective/important features
        
        Formula: importance = frequency / (average_interval_width + ε)
        """
        feature_counts = {}
        feature_intervals = {}  # Store all intervals for each feature
        
        for rule in rules:
            if not rule or rule == "any values (no tightened features)":
                continue
            
            # Parse rule (simple parsing - assumes format "feature ∈ [lower, upper]")
            parts = rule.split(" and ")
            for part in parts:
                part = part.strip()
                if "∈" in part or "in" in part.lower():
                    # Extract feature name and range
                    if "∈" in part:
                        feature_part, range_part = part.split("∈", 1)
                    else:
                        feature_part, range_part = part.split("in", 1)
                    
                    feature_name = feature_part.strip()
                    feature_counts[feature_name] = feature_counts.get(feature_name, 0) + 1
                    
                    # Extract range
                    if "[" in range_part and "]" in range_part:
                        range_str = range_part.split("[")[1].split("]")[0]
                        if "," in range_str:
                            try:
                                lower, upper = map(float, range_str.split(","))
                                interval_width = upper - lower
                                if feature_name not in feature_intervals:
                                    feature_intervals[feature_name] = []
                                feature_intervals[feature_name].append({
                                    "lower": lower,
                                    "upper": upper,
                                    "width": interval_width
                                })
                            except ValueError:
                                pass
        
        # Compute importance scores based on frequency and interval width
        feature_importance = {}
        epsilon = 1e-6  # Small constant to prevent division by zero
        
        for feature_name in feature_counts.keys():
            frequency = feature_counts[feature_name]
            
            if feature_name in feature_intervals and feature_intervals[feature_name]:
                # Calculate average interval width
                widths = [interval["width"] for interval in feature_intervals[feature_name]]
                avg_width = np.mean(widths)
                min_width = np.min(widths)
                max_width = np.max(widths)
                
                # Calculate importance: frequency / (avg_width + epsilon)
                # Narrower intervals (smaller width) = higher importance
                raw_importance = frequency / (avg_width + epsilon)
                
                feature_importance[feature_name] = {
                    "frequency": frequency,
                    "avg_interval_width": float(avg_width),
                    "min_interval_width": float(min_width),
                    "max_interval_width": float(max_width),
                    "raw_importance": float(raw_importance),
                    "n_intervals": len(feature_intervals[feature_name])
                }
            else:
                # Feature appears but no valid intervals found
                feature_importance[feature_name] = {
                    "frequency": frequency,
                    "avg_interval_width": None,
                    "min_interval_width": None,
                    "max_interval_width": None,
                    "raw_importance": 0.0,
                    "n_intervals": 0
                }
        
        # Normalize importance scores to percentages
        total_importance = sum(f["raw_importance"] for f in feature_importance.values())
        if total_importance > 0:
            for feature_name in feature_importance:
                feature_importance[feature_name]["importance_percentage"] = (
                    feature_importance[feature_name]["raw_importance"] / total_importance * 100
                )
        else:
            for feature_name in feature_importance:
                feature_importance[feature_name]["importance_percentage"] = 0.0
        
        # Sort by importance (not just frequency)
        most_important_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1]["raw_importance"],
            reverse=True
        )[:10]
        
        return {
            "feature_counts": feature_counts,
            "feature_importance": feature_importance,
            "most_important_features": most_important_features
        }
    
    def generate_class_report(self, target_class: int, class_name: Optional[str] = None) -> str:
        """Generate a human-readable report for a class."""
        rules_info = self.extract_rules_for_class(target_class)
        
        if not rules_info:
            return f"No rules found for class {target_class}"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"GENERAL RULES FOR CLASS {target_class}" + 
                          (f" ({class_name})" if class_name else ""))
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary
        report_lines.append("## SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total unique rules: {rules_info['n_total_rules']}")
        report_lines.append(f"  - Instance-based rules: {rules_info['n_instance_rules']}")
        report_lines.append(f"  - Class-based rules: {rules_info['n_class_based_rules']}")
        report_lines.append(f"  - Class Union rules (smallest general set): {rules_info['n_class_union_rules']}")
        report_lines.append("")
        
        # Quality metrics
        report_lines.append("## RULE QUALITY METRICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Instance-Based Precision: {rules_info['instance_precision']:.4f}")
        report_lines.append(f"  → Average precision across individual instance-based rules")
        report_lines.append("")
        report_lines.append(f"Instance-Based Coverage: {rules_info['instance_coverage']:.4f}")
        report_lines.append(f"  → Average coverage across individual instance-based rules")
        report_lines.append("")
        report_lines.append(f"Class Union Precision (Union of Class-Based Anchors Only): {rules_info['class_precision']:.4f}")
        report_lines.append(f"  → {rules_info['class_precision']*100:.1f}% of samples satisfying union rules are actually class {target_class}")
        report_lines.append(f"  → Based on union of class-based rules (smallest general set)")
        report_lines.append("")
        report_lines.append(f"Class Union Coverage (Union of Class-Based Anchors Only): {rules_info['class_coverage']:.4f}")
        report_lines.append(f"  → {rules_info['class_coverage']*100:.1f}% of class {target_class} samples satisfy at least one union rule")
        report_lines.append(f"  → Based on union of class-based rules (smallest general set)")
        report_lines.append("")
        
        # Interpretation (based on class union metrics)
        precision = rules_info['class_precision']  # Class union precision (class-based union)
        coverage = rules_info['class_coverage']    # Class union coverage (class-based union)
        
        if precision > 0.9 and coverage > 0.8:
            quality = "Excellent"
            interpretation = "Class union rules (smallest general set) accurately and completely explain the class"
        elif precision > 0.9:
            quality = "Accurate but Incomplete"
            interpretation = "Class union rules are accurate but miss some samples"
        elif coverage > 0.8:
            quality = "Complete but Less Accurate"
            interpretation = "Class union rules cover most samples but include some false positives"
        else:
            quality = "Poor"
            interpretation = "Class union rules need improvement"
        
        report_lines.append(f"Quality Assessment (Class Union): {quality}")
        report_lines.append(f"Interpretation: {interpretation}")
        report_lines.append("")
        
        # Feature analysis for class union rules (smallest general set)
        class_union_rules = rules_info.get('class_union_rules', [])
        if class_union_rules:
            union_feature_analysis = self.analyze_rule_features(class_union_rules)
            report_lines.append("## KEY FEATURES IN CLASS UNION RULES (Smallest General Set)")
            report_lines.append("-" * 80)
            report_lines.append("Feature importance in the class union rules (smallest general set):")
            report_lines.append("  - Frequency: How often the feature appears in union rules")
            report_lines.append("  - Interval selectivity: Narrower intervals indicate more important features")
            report_lines.append("  - Formula: importance = frequency / (average_interval_width + ε)")
            report_lines.append("")
            report_lines.append("Most important features in union rules:")
            for i, (feature, importance_data) in enumerate(union_feature_analysis['most_important_features'][:10], 1):
                freq = importance_data['frequency']
                avg_width = importance_data.get('avg_interval_width')
                importance_pct = importance_data.get('importance_percentage', 0.0)
                
                if avg_width is not None:
                    report_lines.append(
                        f"  {i}. {feature}: "
                        f"frequency={freq}/{len(class_union_rules)}, "
                        f"avg_interval_width={avg_width:.4f}, "
                        f"importance={importance_pct:.1f}%"
                    )
                else:
                    report_lines.append(
                        f"  {i}. {feature}: "
                        f"frequency={freq}/{len(class_union_rules)}, "
                        f"importance={importance_pct:.1f}%"
                    )
            report_lines.append("")
        
        # Feature analysis for all rules
        all_rules = rules_info['all_rules']
        feature_analysis = self.analyze_rule_features(all_rules)
        
        report_lines.append("## KEY FEATURES (All Rules - Ranked by Importance)")
        report_lines.append("-" * 80)
        report_lines.append("Feature importance across all rules (instance-based + class-based):")
        report_lines.append("  - Frequency: How often the feature appears in rules")
        report_lines.append("  - Interval selectivity: Narrower intervals indicate more important features")
        report_lines.append("  - Formula: importance = frequency / (average_interval_width + ε)")
        report_lines.append("")
        report_lines.append("Most important features:")
        for i, (feature, importance_data) in enumerate(feature_analysis['most_important_features'], 1):
            freq = importance_data['frequency']
            avg_width = importance_data.get('avg_interval_width')
            importance_pct = importance_data.get('importance_percentage', 0.0)
            
            if avg_width is not None:
                report_lines.append(
                    f"  {i}. {feature}: "
                    f"frequency={freq}/{len(all_rules)}, "
                    f"avg_interval_width={avg_width:.4f}, "
                    f"importance={importance_pct:.1f}%"
                )
            else:
                report_lines.append(
                    f"  {i}. {feature}: "
                    f"frequency={freq}/{len(all_rules)}, "
                    f"importance={importance_pct:.1f}%"
                )
        report_lines.append("")
        
        # Class Union Rules (smallest general set)
        class_union_rules = rules_info.get('class_union_rules', [])
        if class_union_rules:
            report_lines.append("## CLASS UNION RULES (Smallest General Set)")
            report_lines.append("-" * 80)
            report_lines.append(f"These are the class-based union rules that form the class-level explanation.")
            report_lines.append(f"Total union rules: {len(class_union_rules)}")
            report_lines.append(f"Precision: {rules_info['class_precision']:.4f}, Coverage: {rules_info['class_coverage']:.4f}")
            report_lines.append("")
            report_lines.append("Union Rules:")
            for i, rule in enumerate(class_union_rules, 1):
                rule_display = rule[:150] + "..." if len(rule) > 150 else rule
                report_lines.append(f"  {i}. {rule_display}")
            report_lines.append("")
        else:
            report_lines.append("## CLASS UNION RULES")
            report_lines.append("-" * 80)
            report_lines.append("No class union rules found (class-based rollouts may not have been run)")
            report_lines.append("")
        
        # Sample rules (all rules)
        report_lines.append("## SAMPLE RULES (All Types)")
        report_lines.append("-" * 80)
        report_lines.append(f"Showing first 5 rules (out of {len(all_rules)} total):")
        report_lines.append("")
        
        for i, rule in enumerate(all_rules[:5], 1):
            # Truncate very long rules
            rule_display = rule[:150] + "..." if len(rule) > 150 else rule
            report_lines.append(f"Rule {i}:")
            report_lines.append(f"  {rule_display}")
            report_lines.append("")
        
        if len(all_rules) > 5:
            report_lines.append(f"... and {len(all_rules) - 5} more rules")
            report_lines.append("")
        
        # Complete rule set
        report_lines.append("## COMPLETE RULE SET")
        report_lines.append("-" * 80)
        report_lines.append(f"The classifier identifies class {target_class} samples using the following rule set:")
        report_lines.append("")
        report_lines.append("IF (Rule 1) OR (Rule 2) OR ... OR (Rule N)")
        report_lines.append(f"THEN classifier predicts class {target_class}")
        report_lines.append("")
        report_lines.append(f"Where N = {len(all_rules)} unique rules")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def generate_summary_report(self, class_names: Optional[Dict[int, str]] = None) -> str:
        """Generate a summary report for all classes."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CLASSIFIER RULES ANALYSIS")
        report_lines.append(f"Dataset: {self.dataset_name}")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("RESEARCH QUESTION:")
        report_lines.append('"Given the dataset and trained classifier, what is the set of')
        report_lines.append('general rules that is classifying the samples as a particular class?"')
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Get all classes
        classes = []
        for class_key in self.per_class_results.keys():
            if class_key.startswith("class_") and not class_key.endswith("_class_based"):
                try:
                    class_num = int(class_key.split("_")[1])
                    classes.append(class_num)
                except ValueError:
                    pass
        
        classes = sorted(set(classes))
        
        # Summary table
        report_lines.append("## SUMMARY TABLE")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Class':<10} {'Rules':<10} {'Precision':<12} {'Coverage':<12} {'Quality':<20}")
        report_lines.append("-" * 80)
        
        for cls in classes:
            rules_info = self.extract_rules_for_class(cls)
            if not rules_info:
                continue
            
            precision = rules_info['class_precision']
            coverage = rules_info['class_coverage']
            n_rules = rules_info['n_total_rules']
            
            if precision > 0.9 and coverage > 0.8:
                quality = "Excellent"
            elif precision > 0.9:
                quality = "Accurate"
            elif coverage > 0.8:
                quality = "Complete"
            else:
                quality = "Needs Improvement"
            
            class_label = class_names.get(cls, f"Class {cls}") if class_names else f"Class {cls}"
            report_lines.append(
                f"{class_label:<10} {n_rules:<10} {precision:<12.4f} {coverage:<12.4f} {quality:<20}"
            )
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Detailed reports for each class
        for cls in classes:
            class_label = class_names.get(cls, None) if class_names else None
            class_report = self.generate_class_report(cls, class_label)
            report_lines.append(class_report)
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_analysis(self, class_names: Optional[Dict[int, str]] = None):
        """Save analysis reports to files."""
        # Generate summary report
        summary_report = self.generate_summary_report(class_names)
        
        summary_file = self.output_dir / "classifier_rules_analysis.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_report)
        logger.info(f"Saved summary report to {summary_file}")
        
        # Save structured data
        structured_data = {}
        classes = []
        for class_key in self.per_class_results.keys():
            if class_key.startswith("class_") and not class_key.endswith("_class_based"):
                try:
                    class_num = int(class_key.split("_")[1])
                    classes.append(class_num)
                except ValueError:
                    pass
        
        for cls in sorted(set(classes)):
            rules_info = self.extract_rules_for_class(cls)
            if rules_info:
                feature_analysis = self.analyze_rule_features(rules_info['all_rules'])
                structured_data[f"class_{cls}"] = {
                    **rules_info,
                    "feature_analysis": feature_analysis
                }
        
        json_file = self.output_dir / "classifier_rules_analysis.json"
        with open(json_file, 'w') as f:
            json.dump(structured_data, f, indent=2, default=str)
        logger.info(f"Saved structured analysis to {json_file}")
        
        # Print summary to console
        print("\n" + summary_report)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze classifier rules to answer: What rules classify samples as a particular class?"
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
        help="Dataset name (e.g., 'breast_cancer')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="classifier_rules_analysis",
        help="Output directory for analysis results"
    )
    
    args = parser.parse_args()
    
    analyzer = ClassifierRulesAnalyzer(
        rules_file=args.rules_file,
        dataset_name=args.dataset,
        output_dir=args.output_dir
    )
    
    # Get class names if available (for better reporting)
    class_names = None
    if args.dataset == "breast_cancer":
        class_names = {0: "benign", 1: "malignant"}
    elif args.dataset == "wine":
        class_names = {0: "class_0", 1: "class_1", 2: "class_2"}
    # Add more datasets as needed
    
    analyzer.save_analysis(class_names)


if __name__ == "__main__":
    main()

