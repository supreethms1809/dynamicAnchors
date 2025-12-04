#!/usr/bin/env python3
"""
Analysis script for baseline explainability methods results.

This script analyzes results from baseline.establish_baseline.py and generates:
- Summary statistics for each method
- Feature importance comparisons
- Method comparison visualizations
- Detailed analysis plots

Usage:
    python -m baseline.analyze_baseline baseline_results.json
    python -m baseline.analyze_baseline output/breast_cancer_baseline/baseline_results_*.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import sys
import os
from pathlib import Path

def load_baseline_results(json_path):
    """Load baseline results JSON file."""
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_baseline(json_path):
    """Analyze baseline explainability results and generate plots."""
    
    data = load_baseline_results(json_path)
    
    print("="*80)
    print("BASELINE EXPLAINABILITY METHODS ANALYSIS")
    print("="*80)
    
    # Extract metadata
    dataset = data.get('dataset', 'unknown')
    test_accuracy = data.get('test_accuracy', 0.0)
    n_features = data.get('n_features', 0)
    n_classes = data.get('n_classes', 0)
    class_names = data.get('class_names', [])
    feature_names = data.get('feature_names', [])
    methods = data.get('methods', {})
    
    print(f"\nDataset: {dataset.upper().replace('_', ' ')}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Features: {n_features}, Classes: {n_classes}")
    print(f"Methods run: {list(methods.keys())}")
    
    # ========================================================================
    # 1. METHOD-BY-METHOD ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("METHOD-BY-METHOD ANALYSIS")
    print("="*80)
    
    # LIME Analysis
    if 'lime' in methods and 'error' not in methods['lime']:
        print("\n" + "-"*80)
        print("LIME (Local Interpretable Model-agnostic Explanations)")
        print("-"*80)
        lime_results = methods['lime']
        per_class = lime_results.get('per_class_results', {})
        
        if per_class:
            print(f"Successfully explained {len(per_class)} classes")
            successful_classes = lime_results.get('successful_classes', [])
            failed_classes = lime_results.get('failed_classes', [])
            if failed_classes:
                print(f"Failed classes: {failed_classes}")
            
            # Aggregate feature importance across classes
            all_importance = []
            for cls_key in sorted(per_class.keys()):
                cls_data = per_class[cls_key]
                avg_importance = cls_data.get('avg_feature_importance', [])
                if avg_importance:
                    all_importance.append(avg_importance)
                    n_instances = cls_data.get('n_instances', 0)
                    print(f"\nClass {cls_key} ({class_names[int(cls_key)] if int(cls_key) < len(class_names) else cls_key}):")
                    print(f"  Instances explained: {n_instances}")
                    
                    # Top features
                    importance_arr = np.array(avg_importance)
                    top_indices = np.argsort(np.abs(importance_arr))[-5:][::-1]
                    print(f"  Top 5 features:")
                    for idx in top_indices:
                        print(f"    {feature_names[idx]}: {importance_arr[idx]:.4f}")
            
            if all_importance:
                global_importance = np.mean(all_importance, axis=0)
                top_global = np.argsort(np.abs(global_importance))[-10:][::-1]
                print(f"\nGlobal Top 10 Features (avg across classes):")
                for idx in top_global:
                    print(f"  {feature_names[idx]}: {global_importance[idx]:.4f}")
        else:
            print("No LIME results available")
    
    # Static Anchors Analysis
    if 'static_anchors' in methods and 'error' not in methods['static_anchors']:
        print("\n" + "-"*80)
        print("STATIC ANCHORS")
        print("-"*80)
        anchor_results = methods['static_anchors']
        per_class = anchor_results.get('per_class_results', {})
        
        if per_class:
            instance_precisions = []
            instance_coverages = []
            class_precisions = []
            class_coverages = []
            for cls_key in sorted(per_class.keys()):
                cls_data = per_class[cls_key]
                # Instance-level metrics
                instance_prec = cls_data.get('instance_precision', cls_data.get('avg_precision', 0.0))
                instance_cov = cls_data.get('instance_coverage', cls_data.get('avg_coverage', 0.0))
                # Class-level metrics
                class_prec = cls_data.get('class_precision', 0.0)
                class_cov = cls_data.get('class_coverage', 0.0)
                n_instances = cls_data.get('n_instances', 0)
                
                instance_precisions.append(instance_prec)
                instance_coverages.append(instance_cov)
                if class_prec > 0 or class_cov > 0:
                    class_precisions.append(class_prec)
                    class_coverages.append(class_cov)
                
                print(f"\nClass {cls_key} ({class_names[int(cls_key)] if int(cls_key) < len(class_names) else cls_key}):")
                print(f"  Instance-level (avg across {n_instances} instances):")
                print(f"    Avg Precision: {instance_prec:.4f}")
                print(f"    Avg Coverage:  {instance_cov:.4f}")
                if class_prec > 0 or class_cov > 0:
                    print(f"  Class-level (union of all anchors):")
                    print(f"    Union Precision: {class_prec:.4f}")
                    print(f"    Union Coverage:  {class_cov:.4f}")
            
            print(f"\nOverall Statistics:")
            print(f"  Instance-level (avg across instances):")
            print(f"    Mean Precision: {np.mean(instance_precisions):.4f} (+/- {np.std(instance_precisions):.4f})")
            print(f"    Mean Coverage:  {np.mean(instance_coverages):.4f} (+/- {np.std(instance_coverages):.4f})")
            if class_precisions:
                print(f"  Class-level (union of all anchors):")
                print(f"    Mean Union Precision: {np.mean(class_precisions):.4f} (+/- {np.std(class_precisions):.4f})")
                print(f"    Mean Union Coverage:  {np.mean(class_coverages):.4f} (+/- {np.std(class_coverages):.4f})")
        else:
            print("No Static Anchors results available")
    
    # SHAP Analysis
    if 'shap' in methods and 'error' not in methods['shap']:
        print("\n" + "-"*80)
        print("SHAP (SHapley Additive exPlanations)")
        print("-"*80)
        shap_results = methods['shap']
        per_class = shap_results.get('per_class_results', {})
        
        if per_class:
            print(f"Successfully explained {len(per_class)} classes")
            
            # Aggregate feature importance across classes
            all_importance = []
            for cls_key in sorted(per_class.keys()):
                cls_data = per_class[cls_key]
                avg_importance = cls_data.get('avg_feature_importance', [])
                if avg_importance:
                    all_importance.append(avg_importance)
                    n_instances = cls_data.get('n_instances', 0)
                    print(f"\nClass {cls_key} ({class_names[int(cls_key)] if int(cls_key) < len(class_names) else cls_key}):")
                    print(f"  Instances explained: {n_instances}")
                    
                    # Top features
                    importance_arr = np.array(avg_importance)
                    top_indices = np.argsort(importance_arr)[-5:][::-1]
                    print(f"  Top 5 features (avg |SHAP|):")
                    for idx in top_indices:
                        print(f"    {feature_names[idx]}: {importance_arr[idx]:.4f}")
            
            if all_importance:
                global_importance = np.mean(all_importance, axis=0)
                top_global = np.argsort(global_importance)[-10:][::-1]
                print(f"\nGlobal Top 10 Features (avg across classes):")
                for idx in top_global:
                    print(f"  {feature_names[idx]}: {global_importance[idx]:.4f}")
        else:
            print("No SHAP results available")
    
    # Feature Importance Analysis
    if 'feature_importance' in methods and 'error' not in methods['feature_importance']:
        print("\n" + "-"*80)
        print("FEATURE IMPORTANCE (Permutation Importance)")
        print("-"*80)
        fi_results = methods['feature_importance']
        
        sorted_features = fi_results.get('sorted_features', [])
        sorted_importance = fi_results.get('sorted_importance', [])
        baseline_acc = fi_results.get('baseline_accuracy', test_accuracy)
        
        print(f"Baseline accuracy: {baseline_acc:.4f}")
        print(f"\nTop 10 Most Important Features:")
        for i, (feat, imp) in enumerate(zip(sorted_features[:10], sorted_importance[:10]), 1):
            print(f"  {i}. {feat}: {imp:.4f}")
    
    # ========================================================================
    # 2. METHOD COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("METHOD COMPARISON")
    print("="*80)
    
    # Collect feature importance from all methods
    method_feature_importance = {}
    
    # LIME
    if 'lime' in methods and 'error' not in methods['lime']:
        lime_per_class = methods['lime'].get('per_class_results', {})
        if lime_per_class:
            all_lime_importance = []
            for cls_data in lime_per_class.values():
                avg_imp = cls_data.get('avg_feature_importance', [])
                if avg_imp:
                    all_lime_importance.append(avg_imp)
            if all_lime_importance:
                method_feature_importance['LIME'] = np.mean(all_lime_importance, axis=0)
    
    # SHAP
    if 'shap' in methods and 'error' not in methods['shap']:
        shap_per_class = methods['shap'].get('per_class_results', {})
        if shap_per_class:
            all_shap_importance = []
            for cls_data in shap_per_class.values():
                avg_imp = cls_data.get('avg_feature_importance', [])
                if avg_imp:
                    all_shap_importance.append(avg_imp)
            if all_shap_importance:
                method_feature_importance['SHAP'] = np.mean(all_shap_importance, axis=0)
    
    # Feature Importance
    if 'feature_importance' in methods and 'error' not in methods['feature_importance']:
        fi_importance = methods['feature_importance'].get('feature_importance', [])
        if fi_importance:
            # Normalize to [0, 1] for comparison
            fi_arr = np.array(fi_importance)
            if fi_arr.max() > 0:
                fi_arr = (fi_arr - fi_arr.min()) / (fi_arr.max() - fi_arr.min())
            method_feature_importance['Permutation Importance'] = fi_arr
    
    # Compare feature rankings
    if len(method_feature_importance) > 1:
        print("\nFeature Importance Correlation:")
        method_names = list(method_feature_importance.keys())
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                imp1 = method_feature_importance[method1]
                imp2 = method_feature_importance[method2]
                if len(imp1) == len(imp2):
                    corr = np.corrcoef(imp1, imp2)[0, 1]
                    print(f"  {method1} vs {method2}: {corr:.3f}")
    
    # ========================================================================
    # 3. GENERATE PLOTS
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    # Create single focused plot: Precision and Coverage by Class
    dataset_display_name = dataset.upper().replace('_', ' ')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if 'static_anchors' in methods and 'error' not in methods['static_anchors']:
        anchor_per_class = methods['static_anchors'].get('per_class_results', {})
        if anchor_per_class:
            classes = sorted(anchor_per_class.keys())
            # Instance-level metrics
            instance_precisions = [anchor_per_class[cls].get('instance_precision', anchor_per_class[cls].get('avg_precision', 0.0)) for cls in classes]
            instance_coverages = [anchor_per_class[cls].get('instance_coverage', anchor_per_class[cls].get('avg_coverage', 0.0)) for cls in classes]
            # Class-level metrics
            class_precisions = [anchor_per_class[cls].get('class_precision', 0.0) for cls in classes]
            class_coverages = [anchor_per_class[cls].get('class_coverage', 0.0) for cls in classes]
            
            x = np.arange(len(classes))
            width = 0.2
            
            # Plot instance-level metrics
            bars1 = ax.bar(x - 1.5*width, instance_precisions, width, label='Instance Precision', alpha=0.8, color='blue', edgecolor='black', linewidth=1.5)
            bars2 = ax.bar(x - 0.5*width, instance_coverages, width, label='Instance Coverage', alpha=0.8, color='lightblue', edgecolor='black', linewidth=1.5)
            
            # Plot class-level metrics if available
            if any(cp > 0 or cc > 0 for cp, cc in zip(class_precisions, class_coverages)):
                bars3 = ax.bar(x + 0.5*width, class_precisions, width, label='Class Union Precision', alpha=0.8, color='orange', edgecolor='black', linewidth=1.5)
                bars4 = ax.bar(x + 1.5*width, class_coverages, width, label='Class Union Coverage', alpha=0.8, color='red', edgecolor='black', linewidth=1.5)
            
            ax.set_xlabel('Class', fontsize=12, fontweight='bold')
            ax.set_ylabel('Precision / Coverage', fontsize=12, fontweight='bold')
            ax.set_title(f'{dataset_display_name} - Baseline Static Anchors: Precision & Coverage by Class', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels([class_names[int(c)] if int(c) < len(class_names) else f'Class {c}' for c in classes], 
                               rotation=45, ha='right', fontsize=11)
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0.01:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            if any(cp > 0 or cc > 0 for cp, cc in zip(class_precisions, class_coverages)):
                for bars in [bars3, bars4]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0.01:
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Static Anchors data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{dataset_display_name} - Baseline Static Anchors', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Static Anchors data available\n(Static anchors method not run or encountered error)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{dataset_display_name} - Baseline Static Anchors', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.dirname(json_path)
    output_path = os.path.join(output_dir, 'baseline_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved baseline analysis plot to: {output_path}")
    
    # ========================================================================
    # 4. SAVE SUMMARY STATISTICS
    # ========================================================================
    summary_output = os.path.join(output_dir, 'baseline_summary.json')
    summary_data = {
        'dataset': dataset,
        'test_accuracy': test_accuracy,
        'n_features': n_features,
        'n_classes': n_classes,
        'methods_summary': {}
    }
    
    for method_name in ['lime', 'static_anchors', 'shap', 'feature_importance']:
        if method_name in methods:
            if 'error' in methods[method_name]:
                summary_data['methods_summary'][method_name] = {
                    'status': 'error',
                    'error': methods[method_name]['error']
                }
            else:
                per_class = methods[method_name].get('per_class_results', {})
                summary_data['methods_summary'][method_name] = {
                    'status': 'success',
                    'n_classes': len(per_class) if per_class else 0
                }
                
                # Add method-specific stats
                if method_name == 'static_anchors' and per_class:
                    instance_precisions = [per_class[cls].get('instance_precision', per_class[cls].get('avg_precision', 0.0)) for cls in per_class.keys()]
                    instance_coverages = [per_class[cls].get('instance_coverage', per_class[cls].get('avg_coverage', 0.0)) for cls in per_class.keys()]
                    class_precisions = [per_class[cls].get('class_precision', 0.0) for cls in per_class.keys()]
                    class_coverages = [per_class[cls].get('class_coverage', 0.0) for cls in per_class.keys()]
                    summary_data['methods_summary'][method_name]['instance_avg_precision'] = float(np.mean(instance_precisions))
                    summary_data['methods_summary'][method_name]['instance_avg_coverage'] = float(np.mean(instance_coverages))
                    summary_data['methods_summary'][method_name]['avg_precision'] = float(np.mean(instance_precisions))  # Legacy
                    summary_data['methods_summary'][method_name]['avg_coverage'] = float(np.mean(instance_coverages))  # Legacy
                    if any(cp > 0 or cc > 0 for cp, cc in zip(class_precisions, class_coverages)):
                        summary_data['methods_summary'][method_name]['class_union_precision'] = float(np.mean(class_precisions))
                        summary_data['methods_summary'][method_name]['class_union_coverage'] = float(np.mean(class_coverages))
    
    with open(summary_output, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary statistics to: {summary_output}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

def main():
    """Main entry point for the baseline analysis script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze baseline explainability methods results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m baseline.analyze_baseline output/breast_cancer_baseline/baseline_results_20251110_163644.json
  python -m baseline.analyze_baseline output/breast_cancer_baseline/baseline_results_*.json
  python -m baseline.analyze_baseline --json output/breast_cancer_baseline/baseline_results_*.json
        """
    )
    parser.add_argument(
        'json_path',
        type=str,
        nargs='?',
        default=None,
        help='Path to baseline results JSON file'
    )
    parser.add_argument(
        '--json',
        type=str,
        dest='json_path_alt',
        default=None,
        help='Path to baseline results JSON file (alternative flag)'
    )
    
    args = parser.parse_args()
    
    # Determine which path to use
    json_path = args.json_path_alt or args.json_path
    
    if json_path is None:
        parser.print_help()
        print("\nError: No JSON file path provided.")
        print("\nUsage examples:")
        print("  python -m baseline.analyze_baseline output/breast_cancer_baseline/baseline_results_*.json")
        print("  python -m baseline.analyze_baseline --json output/breast_cancer_baseline/baseline_results_*.json")
        sys.exit(1)
    
    # Handle wildcards
    import glob
    if '*' in json_path or '?' in json_path:
        matches = glob.glob(json_path)
        if not matches:
            print(f"Error: No files found matching pattern: {json_path}")
            sys.exit(1)
        if len(matches) > 1:
            print(f"Warning: Multiple files match pattern. Using first match: {matches[0]}")
        json_path = matches[0]
    
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    try:
        analyze_baseline(json_path)
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing baseline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

