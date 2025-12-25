#!/usr/bin/env python3
"""
Analyze Explainability Results with Exploratory Data Analysis

This script connects EDA findings to explainability results, providing
interpretations based on data characteristics.

Usage:
    python analyze_results_with_eda.py \
        --dataset breast_cancer \
        --results_dir comparison_results/breast_cancer_maddpg_20251222_201640 \
        --output_dir eda_analysis_output
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from BenchMARL.tabular_datasets import TabularDatasetLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert keys and values
        # For dictionary keys, convert numpy integers to native int (JSON accepts int keys)
        converted_dict = {}
        for k, v in obj.items():
            if isinstance(k, np.integer):
                converted_key = int(k)
            elif isinstance(k, np.floating):
                # JSON doesn't accept float keys, convert to string
                converted_key = str(float(k))
            else:
                converted_key = k
            converted_dict[converted_key] = convert_numpy_types(v)
        return converted_dict
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (bool, type(None))):
        return obj
    elif isinstance(obj, (int, float, str)):
        return obj
    else:
        # For other types, try to convert to string as fallback
        try:
            return str(obj)
        except:
            return obj


class EDAResultAnalyzer:
    """Analyze explainability results in the context of EDA findings."""
    
    def __init__(self, dataset_name: str, results_dir: str, output_dir: str):
        self.dataset_name = dataset_name
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        logger.info(f"Loading dataset: {dataset_name}")
        self.loader = TabularDatasetLoader(
            dataset_name=dataset_name,
            test_size=0.2,
            random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test, \
            self.feature_names, self.class_names = self.loader.load_dataset()
        
        # Try to find existing EDA results first
        existing_eda_dir = self._find_existing_eda_results()
        
        if existing_eda_dir:
            logger.info(f"Found existing EDA results in: {existing_eda_dir}")
            logger.info("  Using existing EDA results (skipping recomputation)")
            eda_output_dir = str(existing_eda_dir)
        else:
            logger.info("No existing EDA results found. Performing EDA analysis...")
            eda_output_dir = str(self.output_dir / "eda/")
        
        # Use perform_eda_analysis which will load existing results if available
        self.eda_results = self.loader.perform_eda_analysis(
            output_dir=eda_output_dir,
            verbose=False,
            use_ydata_profiling=False,
            force_rerun=False  # Don't recompute if results exist
        )
        
        # Load explainability results
        self.results = self._load_results()
    
    def _find_existing_eda_results(self) -> Optional[Path]:
        """Look for existing EDA results in the results directory structure."""
        # Check common locations where EDA might be stored
        possible_eda_paths = [
            # Directly in results_dir (multi-agent or single-agent subdirs)
            self.results_dir / "multi_agent" / "eda",
            self.results_dir / "single_agent" / "eda",
            # Parent directory (if results_dir is inside experiment folder)
            self.results_dir.parent / "multi_agent" / "eda",
            self.results_dir.parent / "single_agent" / "eda",
            # BenchMARL experiment structure (check parent directories)
            self.results_dir.parent.parent / "eda",
        ]
        
        # Check fixed paths first
        for path in possible_eda_paths:
            if path.exists() and (path / "eda_summary.json").exists():
                logger.debug(f"Found EDA results at: {path}")
                return path
        
        # Try to find eda directories by searching in results_dir and parent directories
        # Limit search depth to avoid scanning entire filesystem
        search_dirs = [
            self.results_dir,
            self.results_dir.parent,
            self.results_dir.parent.parent,
        ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            try:
                # Look for eda subdirectories (limit depth to 3 levels)
                for item in search_dir.rglob("eda"):
                    if item.is_dir() and (item / "eda_summary.json").exists():
                        # Check depth (count path parts relative to search_dir)
                        depth = len(item.relative_to(search_dir).parts)
                        if depth <= 3:  # Limit depth to avoid deep searches
                            logger.debug(f"Found EDA results at: {item}")
                            return item
            except (ValueError, PermissionError) as e:
                # Skip if we can't access the directory
                logger.debug(f"Could not search {search_dir}: {e}")
                continue
        
        return None
    
    def _load_results(self) -> Dict[str, Any]:
        """Load explainability results from JSON files."""
        results = {}
        
        # Try to load comparison summary
        comparison_file = self.results_dir / "comparison_summary.json"
        if comparison_file.exists():
            with open(comparison_file, 'r') as f:
                results["comparison"] = json.load(f)
        
        # Try to load method-specific summaries
        for method in ["multi_agent", "single_agent", "baseline"]:
            method_file = self.results_dir / method / "summary.json"
            if method_file.exists():
                with open(method_file, 'r') as f:
                    results[method] = json.load(f)
        
        return results
    
    def analyze_class_distribution_impact(self) -> Dict[str, Any]:
        """Analyze how class distribution affects results."""
        class_dist = self.eda_results["class_distribution"]
        is_balanced = class_dist["is_balanced"]
        train_dist = class_dist["train"]
        
        analysis = {
            "is_balanced": is_balanced,
            "class_sizes": {},
            "impact_on_results": []
        }
        
        # Get class sizes (convert numpy int64 keys to native int)
        for cls, info in train_dist.items():
            cls_key = int(cls) if isinstance(cls, (np.integer, np.int64)) else cls
            analysis["class_sizes"][cls_key] = {
                "count": int(info["count"]) if isinstance(info["count"], (np.integer, np.int64)) else info["count"],
                "percentage": float(info["percentage"]) if isinstance(info["percentage"], (np.floating, np.float64)) else info["percentage"]
            }
        
        # Analyze impact
        if not is_balanced:
            analysis["impact_on_results"].append(
                "Imbalanced classes: Multi-agent may help coordinate coverage"
            )
        
        # Check if results show imbalance effects
        for method_name in ["multi_agent", "single_agent"]:
            if method_name in self.results:
                method_results = self.results[method_name].get("summary", {})
                per_class = method_results.get("per_class_summary", {})
                
                if per_class:
                    coverages = [c.get("instance_coverage", 0) for c in per_class.values()]
                    if len(coverages) > 1 and max(coverages) - min(coverages) > 0.3:
                        analysis["impact_on_results"].append(
                            f"{method_name}: Large coverage differences between classes "
                            f"(likely due to class imbalance)"
                        )
        
        return analysis
    
    def analyze_separability_impact(self) -> Dict[str, Any]:
        """Analyze how class separability affects precision/coverage."""
        separability = self.eda_results["class_separability"]
        top_features = separability.get("top_features", [])
        overlap_metrics = separability.get("class_overlap_metrics", {})
        
        analysis = {
            "top_separable_features": top_features[:10],
            "class_overlap": overlap_metrics,
            "precision_explanations": {},
            "coverage_explanations": {}
        }
        
        # Analyze precision based on separability
        for method_name in ["multi_agent", "single_agent"]:
            if method_name in self.results:
                method_results = self.results[method_name].get("summary", {})
                per_class = method_results.get("per_class_summary", {})
                
                for class_key, class_results in per_class.items():
                    precision = class_results.get("instance_precision", 0)
                    
                    if precision > 0.9:
                        analysis["precision_explanations"][f"{method_name}_class_{class_key}"] = \
                            "High precision: Classes are well-separated"
                    elif precision < 0.7:
                        analysis["precision_explanations"][f"{method_name}_class_{class_key}"] = \
                            "Low precision: Classes overlap significantly"
        
        return analysis
    
    def analyze_feature_correlation_impact(self) -> Dict[str, Any]:
        """Analyze how feature correlations affect rule complexity."""
        correlations = self.eda_results["feature_correlations"]
        high_corr = correlations.get("high_correlations", [])
        
        analysis = {
            "high_correlations_count": len(high_corr),
            "top_correlations": sorted(
                high_corr,
                key=lambda x: abs(x["correlation"]),
                reverse=True
            )[:10],
            "impact_on_rules": []
        }
        
        if len(high_corr) > 5:
            analysis["impact_on_rules"].append(
                "Many correlated features: Rules may include groups of correlated features"
            )
        
        # Check if rules include correlated features
        for method_name in ["multi_agent", "single_agent"]:
            if method_name in self.results:
                method_results = self.results[method_name].get("summary", {})
                per_class = method_results.get("per_class_summary", {})
                
                for class_key, class_results in per_class.items():
                    rules = class_results.get("unique_rules", [])
                    if rules:
                        # Simple check: count features in first rule
                        first_rule = rules[0] if rules else ""
                        rule_features = [f for f in self.feature_names if f in first_rule]
                        
                        # Check if correlated features appear together
                        for corr in high_corr[:5]:
                            feat1, feat2 = corr["feature1"], corr["feature2"]
                            if feat1 in rule_features and feat2 in rule_features:
                                analysis["impact_on_rules"].append(
                                    f"{method_name} class {class_key}: Rules include "
                                    f"correlated features ({feat1}, {feat2})"
                                )
                                break
        
        return analysis
    
    def analyze_precision_coverage_tradeoff(self) -> Dict[str, Any]:
        """Analyze precision-coverage trade-off based on data characteristics."""
        separability = self.eda_results["class_separability"]
        feature_stats = self.eda_results["feature_statistics"]
        
        analysis = {
            "tradeoff_explanations": {},
            "recommendations": []
        }
        
        for method_name in ["multi_agent", "single_agent"]:
            if method_name in self.results:
                method_results = self.results[method_name].get("summary", {})
                per_class = method_results.get("per_class_summary", {})
                
                for class_key, class_results in per_class.items():
                    precision = class_results.get("instance_precision", 0)
                    coverage = class_results.get("instance_coverage", 0)
                    
                    explanation = []
                    
                    if precision > 0.9 and coverage < 0.3:
                        explanation.append(
                            "High precision, low coverage: Well-separated but concentrated distribution"
                        )
                    elif precision < 0.7 and coverage > 0.7:
                        explanation.append(
                            "Low precision, high coverage: Overlapping classes require wide anchors"
                        )
                    elif precision > 0.8 and coverage > 0.6:
                        explanation.append(
                            "Good balance: Classes are separable and well-distributed"
                        )
                    
                    if explanation:
                        analysis["tradeoff_explanations"][f"{method_name}_class_{class_key}"] = explanation
        
        return analysis
    
    def compare_methods_with_eda(self) -> Dict[str, Any]:
        """Compare methods in the context of EDA findings."""
        comparison = {
            "method_comparisons": {},
            "eda_context": {}
        }
        
        # Get EDA context
        class_dist = self.eda_results["class_distribution"]
        correlations = self.eda_results["feature_correlations"]
        separability = self.eda_results["class_separability"]
        
        comparison["eda_context"] = {
            "is_balanced": class_dist["is_balanced"],
            "high_correlations_count": len(correlations.get("high_correlations", [])),
            "top_separable_features": separability.get("top_features", [])[:5]
        }
        
        # Compare methods
        methods_data = {}
        for method_name in ["multi_agent", "single_agent", "baseline"]:
            if method_name in self.results:
                method_results = self.results[method_name].get("summary", {})
                per_class = method_results.get("per_class_summary", {})
                
                if per_class:
                    precisions = [c.get("instance_precision", 0) for c in per_class.values()]
                    coverages = [c.get("instance_coverage", 0) for c in per_class.values()]
                    
                    methods_data[method_name] = {
                        "avg_precision": float(np.mean(precisions)),
                        "avg_coverage": float(np.mean(coverages)),
                        "std_precision": float(np.std(coverages)),
                        "std_coverage": float(np.std(coverages))
                    }
        
        comparison["method_comparisons"] = methods_data
        
        # Add interpretations
        if "multi_agent" in methods_data and "single_agent" in methods_data:
            ma_coverage = methods_data["multi_agent"]["avg_coverage"]
            sa_coverage = methods_data["single_agent"]["avg_coverage"]
            
            if ma_coverage > sa_coverage + 0.1:
                comparison["interpretations"] = [
                    "Multi-agent achieves higher coverage: Benefits from coordination",
                    "Especially beneficial for:" + (
                        " imbalanced classes" if not class_dist["is_balanced"] else ""
                    ) + (
                        " correlated features" if len(correlations.get("high_correlations", [])) > 5 else ""
                    )
                ]
        
        return comparison
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("EDA-INFORMED RESULT ANALYSIS")
        report_lines.append(f"Dataset: {self.dataset_name}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 1. Dataset Overview
        report_lines.append("## 1. DATASET OVERVIEW")
        report_lines.append("-" * 80)
        report_lines.append(f"Training samples: {len(self.X_train):,}")
        report_lines.append(f"Test samples: {len(self.X_test):,}")
        report_lines.append(f"Features: {len(self.feature_names)}")
        report_lines.append(f"Classes: {len(self.class_names)}")
        report_lines.append("")
        
        # 2. Class Distribution Impact
        report_lines.append("## 2. CLASS DISTRIBUTION ANALYSIS")
        report_lines.append("-" * 80)
        class_dist_analysis = self.analyze_class_distribution_impact()
        report_lines.append(f"Balanced: {class_dist_analysis['is_balanced']}")
        report_lines.append("Class sizes:")
        for cls, info in class_dist_analysis["class_sizes"].items():
            cls_name = self.class_names[int(cls)] if int(cls) < len(self.class_names) else f"Class {cls}"
            report_lines.append(f"  {cls_name}: {info['count']:,} ({info['percentage']:.1f}%)")
        
        if class_dist_analysis["impact_on_results"]:
            report_lines.append("\nImpact on results:")
            for impact in class_dist_analysis["impact_on_results"]:
                report_lines.append(f"  • {impact}")
        report_lines.append("")
        
        # 3. Separability Impact
        report_lines.append("## 3. CLASS SEPARABILITY ANALYSIS")
        report_lines.append("-" * 80)
        separability_analysis = self.analyze_separability_impact()
        report_lines.append("Top 5 most separable features:")
        for feat in separability_analysis["top_separable_features"][:5]:
            report_lines.append(f"  • {feat}")
        
        if separability_analysis["precision_explanations"]:
            report_lines.append("\nPrecision explanations:")
            for key, explanation in separability_analysis["precision_explanations"].items():
                report_lines.append(f"  • {key}: {explanation}")
        report_lines.append("")
        
        # 4. Feature Correlation Impact
        report_lines.append("## 4. FEATURE CORRELATION ANALYSIS")
        report_lines.append("-" * 80)
        corr_analysis = self.analyze_feature_correlation_impact()
        report_lines.append(f"High correlations (|r| > 0.7): {corr_analysis['high_correlations_count']}")
        
        if corr_analysis["top_correlations"]:
            report_lines.append("\nTop correlations:")
            for corr in corr_analysis["top_correlations"][:5]:
                report_lines.append(
                    f"  • {corr['feature1']} <-> {corr['feature2']}: {corr['correlation']:.3f}"
                )
        
        if corr_analysis["impact_on_rules"]:
            report_lines.append("\nImpact on rules:")
            for impact in corr_analysis["impact_on_rules"][:5]:
                report_lines.append(f"  • {impact}")
        report_lines.append("")
        
        # 5. Precision-Coverage Trade-off
        report_lines.append("## 5. PRECISION-COVERAGE TRADE-OFF ANALYSIS")
        report_lines.append("-" * 80)
        tradeoff_analysis = self.analyze_precision_coverage_tradeoff()
        
        for key, explanations in tradeoff_analysis["tradeoff_explanations"].items():
            report_lines.append(f"{key}:")
            for explanation in explanations:
                report_lines.append(f"  • {explanation}")
        report_lines.append("")
        
        # 6. Method Comparison
        report_lines.append("## 6. METHOD COMPARISON WITH EDA CONTEXT")
        report_lines.append("-" * 80)
        comparison = self.compare_methods_with_eda()
        
        report_lines.append("EDA Context:")
        eda_ctx = comparison["eda_context"]
        report_lines.append(f"  • Balanced classes: {eda_ctx['is_balanced']}")
        report_lines.append(f"  • High correlations: {eda_ctx['high_correlations_count']}")
        report_lines.append(f"  • Top separable features: {', '.join(eda_ctx['top_separable_features'][:3])}")
        report_lines.append("")
        
        report_lines.append("Method Performance:")
        for method_name, metrics in comparison["method_comparisons"].items():
            report_lines.append(f"  {method_name}:")
            report_lines.append(f"    Avg Precision: {metrics['avg_precision']:.3f} ± {metrics['std_precision']:.3f}")
            report_lines.append(f"    Avg Coverage: {metrics['avg_coverage']:.3f} ± {metrics['std_coverage']:.3f}")
        
        if "interpretations" in comparison:
            report_lines.append("\nInterpretations:")
            for interpretation in comparison["interpretations"]:
                report_lines.append(f"  • {interpretation}")
        report_lines.append("")
        
        # 7. Recommendations
        report_lines.append("## 7. RECOMMENDATIONS")
        report_lines.append("-" * 80)
        
        if not class_dist_analysis["is_balanced"]:
            report_lines.append("• Consider multi-agent for imbalanced classes")
        
        if corr_analysis["high_correlations_count"] > 5:
            report_lines.append("• Many correlated features: Rules may be simplified")
        
        if len(self.feature_names) > 20:
            report_lines.append("• High dimensionality: Consider feature selection")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_analysis(self):
        """Save analysis results to files."""
        # Generate and save report
        report = self.generate_report()
        report_file = self.output_dir / "eda_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved analysis report to {report_file}")
        
        # Save structured analysis as JSON
        analysis_data = {
            "class_distribution_impact": self.analyze_class_distribution_impact(),
            "separability_impact": self.analyze_separability_impact(),
            "feature_correlation_impact": self.analyze_feature_correlation_impact(),
            "precision_coverage_tradeoff": self.analyze_precision_coverage_tradeoff(),
            "method_comparison": self.compare_methods_with_eda()
        }
        
        analysis_file = self.output_dir / "eda_analysis.json"
        # Convert numpy types to native Python types before JSON serialization
        analysis_data_serializable = convert_numpy_types(analysis_data)
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data_serializable, f, indent=2, default=str)
        logger.info(f"Saved structured analysis to {analysis_file}")
        
        # Print report to console
        print("\n" + report)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze explainability results with EDA context"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'breast_cancer', 'circles')"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to results directory containing summary.json files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eda_analysis_output",
        help="Output directory for analysis results"
    )
    
    args = parser.parse_args()
    
    analyzer = EDAResultAnalyzer(
        dataset_name=args.dataset,
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    
    analyzer.save_analysis()


if __name__ == "__main__":
    main()

