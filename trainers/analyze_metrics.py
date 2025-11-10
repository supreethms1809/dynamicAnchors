#!/usr/bin/env python3
"""Quick analysis script for metrics_and_rules.json"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import re
import sys
from itertools import combinations

def analyze_metrics(json_path):
    """Analyze metrics_and_rules.json file and generate plots."""
    
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Helper function to get the appropriate per_class_results structure
    def get_results_structure():
        """Returns (instance_results, class_results, is_new_structure)"""
        per_class_results = data.get('per_class_results', {})
        if 'instance_level' in per_class_results:
            return per_class_results['instance_level'], per_class_results.get('class_level', {}), True
        else:
            return per_class_results, {}, False
    
    instance_results, class_results, is_new_structure = get_results_structure()
    
    print("="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    summary = data['training_summary']
    print(f"Algorithm: {summary['algorithm']}")
    print(f"Episodes: {summary['episodes']}")
    print(f"Steps per episode: {summary['steps_per_episode']}")
    print(f"Classifier update every: {summary['classifier_update_every']} episodes")
    print(f"Classifier epochs per round: {summary['classifier_epochs_per_round']}")
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    overall = data['overall_statistics']
    
    # Handle both new structure (instance_level/class_level) and old structure (backward compatibility)
    if 'instance_level' in overall:
        # New structure: separate instance-level and class-level
        print("\n[Instance-Level Evaluation] (One anchor per test instance, like static anchors):")
        instance_overall = overall['instance_level']
        print(f"  Overall Precision: {instance_overall.get('overall_precision', 0.0):.4f}")
        print(f"  Overall Coverage: {instance_overall.get('overall_coverage', 0.0):.4f}")
        if 'overall_n_points' in instance_overall:
            print(f"  Overall N Points: {instance_overall.get('overall_n_points', 0)}")
        
        print("\n[Class-Level Evaluation] (One anchor per class, dynamic anchors advantage):")
        class_overall = overall['class_level']
        print(f"  Overall Precision: {class_overall.get('overall_precision', 0.0):.4f}")
        print(f"  Overall Coverage: {class_overall.get('overall_coverage', 0.0):.4f}")
        
        # Show comparison
        print("\n[Comparison]:")
        cov_improvement = class_overall.get('overall_coverage', 0.0) - instance_overall.get('overall_coverage', 0.0)
        if cov_improvement > 0:
            improvement_pct = (cov_improvement / instance_overall.get('overall_coverage', 1e-10)) * 100 if instance_overall.get('overall_coverage', 0) > 0 else 0
            print(f"  Coverage Improvement: {cov_improvement:+.4f} ({improvement_pct:+.1f}% increase)")
            print(f"    → Dynamic anchors (class-level) achieve higher coverage!")
        else:
            print(f"  Coverage Difference: {cov_improvement:.4f}")
    else:
        # Old structure: backward compatibility
        print(f"Overall Precision: {overall.get('overall_precision', 0.0):.4f}")
        print(f"Overall Coverage: {overall.get('overall_coverage', 0.0):.4f}")
        if 'overall_n_points' in overall:
            print(f"Overall N Points: {overall.get('overall_n_points', 0)}")
    
    # Handle both new structure (instance_level/class_level) and old structure
    per_class_results = data.get('per_class_results', {})
    
    if 'instance_level' in per_class_results:
        # New structure: separate instance-level and class-level
        instance_results = per_class_results['instance_level']
        class_results = per_class_results['class_level']
    
    print("\n" + "="*80)
    print("TRAINING PROGRESS")
    print("="*80)
    history = data['training_history']
    episodes = [e['episode'] for e in history]
    classifier_acc = [e['classifier_test_acc'] for e in history]
    classifier_loss = [e['classifier_loss'] for e in history]
    rl_precision = [e['rl_avg_precision'] for e in history]
    rl_coverage = [e['rl_avg_coverage'] for e in history]
    
    # Check if history is empty to prevent IndexError
    if len(classifier_acc) == 0 or len(rl_precision) == 0 or len(rl_coverage) == 0:
        print("⚠ WARNING: No history data found in metrics file. Cannot compute statistics.")
        return
    
    print(f"Initial classifier accuracy: {classifier_acc[0]:.4f}")
    print(f"Final classifier accuracy: {classifier_acc[-1]:.4f}")
    print(f"Classifier improvement: {classifier_acc[-1] - classifier_acc[0]:+.4f}")
    
    print(f"\nInitial RL precision: {rl_precision[0]:.4f}")
    print(f"Final RL precision: {rl_precision[-1]:.4f}")
    print(f"RL improvement: {rl_precision[-1] - rl_precision[0]:+.4f}")
    
    print(f"\nInitial RL coverage: {rl_coverage[0]:.4f}")
    print(f"Final RL coverage: {rl_coverage[-1]:.4f}")
    print(f"RL coverage change: {rl_coverage[-1] - rl_coverage[0]:+.4f}")
    
    # Check for issues
    print("\n" + "="*80)
    print("ISSUE DETECTION")
    print("="*80)
    zero_prec_episodes = [e['episode'] for e in history if e['rl_avg_precision'] == 0]
    zero_cov_episodes = [e['episode'] for e in history if e['rl_avg_coverage'] == 0]
    
    if zero_prec_episodes:
        print(f"⚠ Warning: {len(zero_prec_episodes)} episodes with zero precision")
        if len(zero_prec_episodes) <= 10:
            print(f"  Episodes: {zero_prec_episodes}")
    
    if zero_cov_episodes:
        print(f"⚠ Warning: {len(zero_cov_episodes)} episodes with zero coverage")
        if len(zero_cov_episodes) <= 10:
            print(f"  Episodes: {zero_cov_episodes[:10]}")
    
    # Per-class analysis over time
    classes = sorted(set([k for e in history for k in e['per_class_rl_stats'].keys()]))
    if classes:
        print("\n" + "="*80)
        print("PER-CLASS TRAINING ANALYSIS")
        print("="*80)
        for cls in classes:
            cls_precisions = []
            cls_coverages = []
            for e in history:
                cls_stats = e['per_class_rl_stats'].get(cls, {})
                cls_precisions.append(cls_stats.get('hard_precision', 0.0))
                cls_coverages.append(cls_stats.get('coverage', 0.0))
            
            print(f"\n{cls}:")
            if len(cls_precisions) > 0 and len(cls_coverages) > 0:
                print(f"  Initial precision: {cls_precisions[0]:.4f} → Final: {cls_precisions[-1]:.4f}")
                print(f"  Initial coverage: {cls_coverages[0]:.4f} → Final: {cls_coverages[-1]:.4f}")
                print(f"  Max precision: {max(cls_precisions):.4f} (episode {episodes[cls_precisions.index(max(cls_precisions))]})")
                print(f"  Max coverage: {max(cls_coverages):.4f} (episode {episodes[cls_coverages.index(max(cls_coverages))]})")
            else:
                print(f"  No precision/coverage history for this class")
    
    # Create plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    fig = plt.figure(figsize=(18, 10))
    
    # Plot 1: Classifier performance
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(episodes, classifier_acc, 'b-', label='Test Accuracy', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Classifier Test Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Classifier loss
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(episodes, classifier_loss, 'r-', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Classifier Loss')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RL Precision
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(episodes, rl_precision, 'g-', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Precision')
    ax3.set_title('RL Average Precision')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: RL Coverage
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(episodes, rl_coverage, 'm-', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Coverage')
    ax4.set_title('RL Average Coverage')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Per-class precision
    ax5 = plt.subplot(2, 3, 5)
    for cls in classes:
        cls_precisions = [e['per_class_rl_stats'].get(cls, {}).get('hard_precision', 0.0) 
                          for e in history]
        ax5.plot(episodes, cls_precisions, label=cls, linewidth=1.5, alpha=0.7)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Hard Precision')
    ax5.set_title('Per-Class Precision Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Per-class coverage
    ax6 = plt.subplot(2, 3, 6)
    for cls in classes:
        cls_coverages = [e['per_class_rl_stats'].get(cls, {}).get('coverage', 0.0) 
                         for e in history]
        ax6.plot(episodes, cls_coverages, label=cls, linewidth=1.5, alpha=0.7)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Coverage')
    ax6.set_title('Per-Class Coverage Over Time')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = json_path.replace('.json', '_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved analysis plots to: {output_path}")
    
    # Feature frequency analysis
    print("\n" + "="*80)
    print("FEATURE FREQUENCY ANALYSIS")
    print("="*80)
    
    # Get the appropriate per_class_results structure for analysis
    instance_results, class_results, is_new_structure = get_results_structure()
    results_to_analyze = instance_results  # Use instance-level for rule-based analyses
    
    for cls_key in sorted(results_to_analyze.keys()):
        cls_data = results_to_analyze[cls_key]
        rules = cls_data.get('rules', [])
        features = []
        for rule in rules:
            matches = re.findall(r'(\w+) ∈', rule)
            features.extend(matches)
        
        if features:
            feature_counts = Counter(features)
            print(f"\n{cls_key} - Most common features:")
            for feat, count in feature_counts.most_common(5):
                print(f"  {feat}: {count} times ({count/len(rules)*100:.1f}% of rules)")
    
    # ========== ADDITIONAL INTERPRETATION ANALYSES ==========
    
    print("\n" + "="*80)
    print("ADDITIONAL INTERPRETATION ANALYSES")
    print("="*80)
    
    # 1. Rule Complexity Analysis
    print("\n" + "-"*80)
    print("1. RULE COMPLEXITY ANALYSIS")
    print("-"*80)
    instance_results, class_results, is_new_structure = get_results_structure()
    results_to_analyze = instance_results  # Use instance-level for rule-based analyses
    for cls_key in sorted(results_to_analyze.keys()):
        rules = results_to_analyze[cls_key].get('rules', [])
        rule_complexities = []
        for rule in rules:
            if rule and rule != "any values (no tightened features)":
                matches = re.findall(r'(\w+) ∈', rule)
                rule_complexities.append(len(matches))
            else:
                rule_complexities.append(0)
        
        if rule_complexities:
            print(f"\n{cls_key}:")
            print(f"  Mean features per rule: {np.mean(rule_complexities):.2f}")
            print(f"  Std features per rule: {np.std(rule_complexities):.2f}")
            print(f"  Min features: {np.min(rule_complexities)}, Max features: {np.max(rule_complexities)}")
            print(f"  Rules with 0 features: {rule_complexities.count(0)} ({rule_complexities.count(0)/len(rule_complexities)*100:.1f}%)")
            print(f"  Rules with 1-2 features: {sum(1 for c in rule_complexities if 1 <= c <= 2)} ({sum(1 for c in rule_complexities if 1 <= c <= 2)/len(rule_complexities)*100:.1f}%)")
            print(f"  Rules with 3+ features: {sum(1 for c in rule_complexities if c >= 3)} ({sum(1 for c in rule_complexities if c >= 3)/len(rule_complexities)*100:.1f}%)")
    
    # 2. Precision-Coverage Tradeoff Analysis
    print("\n" + "-"*80)
    print("2. PRECISION-COVERAGE TRADEOFF ANALYSIS")
    print("-"*80)
    instance_results, class_results, is_new_structure = get_results_structure()
    results_to_analyze = instance_results  # Use instance-level for rule-based analyses
    for cls_key in sorted(results_to_analyze.keys()):
        rules_data = results_to_analyze[cls_key].get('rules_with_instances', [])
        if rules_data:
            precisions = [r['hard_precision'] for r in rules_data if r.get('hard_precision', 0) > 0]
            coverages = [r['coverage'] for r in rules_data if r.get('coverage', 0) > 0]
            
            if len(precisions) > 1 and len(coverages) > 1:
                # Correlation
                if len(precisions) == len(coverages):
                    corr = np.corrcoef(precisions, coverages)[0, 1]
                    print(f"\n{cls_key}:")
                    print(f"  Precision-Coverage correlation: {corr:.3f}")
                    if corr < -0.3:
                        print(f"    → Strong negative tradeoff (more precise = less coverage)")
                    elif corr > 0.3:
                        print(f"    → Positive relationship (more precise = more coverage)")
                    else:
                        print(f"    → Weak relationship")
                
                # Pareto frontier analysis (high precision, reasonable coverage)
                high_prec_rules = [r for r in rules_data if r.get('hard_precision', 0) > 0.8]
                if high_prec_rules:
                    high_prec_covs = [r['coverage'] for r in high_prec_rules]
                    print(f"  Rules with precision >0.8: {len(high_prec_rules)}")
                    print(f"    Mean coverage: {np.mean(high_prec_covs):.4f}")
                    print(f"    Max coverage: {np.max(high_prec_covs):.4f}")
    
    # 3. Comprehensive Feature Importance Analysis
    print("\n" + "-"*80)
    print("3. COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print("-"*80)
    
    # Store feature importance data for all classes
    all_class_feature_importance = {}
    global_feature_importance = defaultdict(lambda: {
        'count': 0, 
        'total_precision': 0.0, 
        'total_coverage': 0.0,
        'classes': set(),
        'best_precision': 0.0,
        'best_coverage': 0.0
    })
    global_scores = []  # Initialize for later use in plotting
    
    instance_results, class_results, is_new_structure = get_results_structure()
    results_to_analyze = instance_results  # Use instance-level for rule-based analyses
    for cls_key in sorted(results_to_analyze.keys()):
        rules_data = results_to_analyze[cls_key].get('rules_with_instances', [])
        if rules_data:
            feature_importance = defaultdict(lambda: {
                'count': 0, 
                'total_precision': 0.0, 
                'total_coverage': 0.0,
                'precisions': [],
                'coverages': []
            })
            
            for rule_data in rules_data:
                rule = rule_data.get('rule', '')
                precision = rule_data.get('hard_precision', 0.0)
                coverage = rule_data.get('coverage', 0.0)
                
                if rule and rule != "any values (no tightened features)":
                    matches = re.findall(r'(\w+) ∈', rule)
                    for feat in matches:
                        feature_importance[feat]['count'] += 1
                        feature_importance[feat]['total_precision'] += precision
                        feature_importance[feat]['total_coverage'] += coverage
                        feature_importance[feat]['precisions'].append(precision)
                        feature_importance[feat]['coverages'].append(coverage)
                        
                        # Update global stats
                        global_feature_importance[feat]['count'] += 1
                        global_feature_importance[feat]['total_precision'] += precision
                        global_feature_importance[feat]['total_coverage'] += coverage
                        global_feature_importance[feat]['classes'].add(cls_key)
                        global_feature_importance[feat]['best_precision'] = max(
                            global_feature_importance[feat]['best_precision'], precision
                        )
                        global_feature_importance[feat]['best_coverage'] = max(
                            global_feature_importance[feat]['best_coverage'], coverage
                        )
            
            if feature_importance:
                # Calculate multiple importance scores
                feat_scores = []
                for feat, stats_dict in feature_importance.items():
                    count = stats_dict['count']
                    avg_prec = stats_dict['total_precision'] / count if count > 0 else 0
                    avg_cov = stats_dict['total_coverage'] / count if count > 0 else 0
                    max_prec = max(stats_dict['precisions']) if stats_dict['precisions'] else 0
                    max_cov = max(stats_dict['coverages']) if stats_dict['coverages'] else 0
                    
                    # Multiple scoring methods
                    score_frequency = count  # Simple frequency
                    score_precision_weighted = count * avg_prec  # Frequency × avg precision
                    score_coverage_weighted = count * avg_cov  # Frequency × avg coverage
                    score_combined = count * avg_prec * avg_cov  # Frequency × precision × coverage
                    score_max_precision = count * max_prec  # Frequency × max precision
                    
                    feat_scores.append({
                        'feature': feat,
                        'count': count,
                        'avg_precision': avg_prec,
                        'avg_coverage': avg_cov,
                        'max_precision': max_prec,
                        'max_coverage': max_cov,
                        'score_frequency': score_frequency,
                        'score_precision_weighted': score_precision_weighted,
                        'score_coverage_weighted': score_coverage_weighted,
                        'score_combined': score_combined,
                        'score_max_precision': score_max_precision
                    })
                
                all_class_feature_importance[cls_key] = feat_scores
                
                # Print top features by different metrics
                print(f"\n{cls_key} - Top features by frequency:")
                feat_scores_sorted = sorted(feat_scores, key=lambda x: x['score_frequency'], reverse=True)
                for feat_data in feat_scores_sorted[:5]:
                    print(f"  {feat_data['feature']}: {feat_data['count']} appearances "
                          f"(avg prec={feat_data['avg_precision']:.3f}, avg cov={feat_data['avg_coverage']:.4f})")
                
                print(f"\n{cls_key} - Top features by precision-weighted importance:")
                feat_scores_sorted = sorted(feat_scores, key=lambda x: x['score_precision_weighted'], reverse=True)
                for feat_data in feat_scores_sorted[:5]:
                    print(f"  {feat_data['feature']}: score={feat_data['score_precision_weighted']:.2f} "
                          f"(count={feat_data['count']}, avg prec={feat_data['avg_precision']:.3f})")
                
                print(f"\n{cls_key} - Top features by combined score (freq × prec × cov):")
                feat_scores_sorted = sorted(feat_scores, key=lambda x: x['score_combined'], reverse=True)
                for feat_data in feat_scores_sorted[:5]:
                    print(f"  {feat_data['feature']}: score={feat_data['score_combined']:.4f} "
                          f"(count={feat_data['count']}, prec={feat_data['avg_precision']:.3f}, cov={feat_data['avg_coverage']:.4f})")
    
    # Global feature importance (across all classes)
    print("\n" + "-"*80)
    print("GLOBAL FEATURE IMPORTANCE (Across All Classes)")
    print("-"*80)
    if global_feature_importance:
        global_scores = []
        for feat, stats_dict in global_feature_importance.items():
            count = stats_dict['count']
            avg_prec = stats_dict['total_precision'] / count if count > 0 else 0
            avg_cov = stats_dict['total_coverage'] / count if count > 0 else 0
            n_classes = len(stats_dict['classes'])
            
            global_score = count * avg_prec * avg_cov * n_classes  # Weight by class diversity too
            
            global_scores.append({
                'feature': feat,
                'count': count,
                'avg_precision': avg_prec,
                'avg_coverage': avg_cov,
                'n_classes': n_classes,
                'best_precision': stats_dict['best_precision'],
                'best_coverage': stats_dict['best_coverage'],
                'global_score': global_score
            })
        
        global_scores.sort(key=lambda x: x['global_score'], reverse=True)
        print("\nTop 10 globally important features:")
        for i, feat_data in enumerate(global_scores[:10], 1):
            print(f"  {i}. {feat_data['feature']}: "
                  f"count={feat_data['count']}, "
                  f"classes={feat_data['n_classes']}, "
                  f"avg_prec={feat_data['avg_precision']:.3f}, "
                  f"avg_cov={feat_data['avg_coverage']:.4f}, "
                  f"best_prec={feat_data['best_precision']:.3f}, "
                  f"score={feat_data['global_score']:.4f}")
        
        # Save feature importance to JSON
        importance_output = json_path.replace('.json', '_feature_importance.json')
        importance_data = {
            'global_importance': global_scores,
            'per_class_importance': {}
        }
        for cls_key, feat_scores in all_class_feature_importance.items():
            importance_data['per_class_importance'][cls_key] = feat_scores
        
        with open(importance_output, 'w') as f:
            json.dump(importance_data, f, indent=2, default=str)  # default=str handles any non-serializable types
        print(f"\nSaved feature importance data to: {importance_output}")
    
    # 4. Feature Co-occurrence Analysis
    print("\n" + "-"*80)
    print("4. FEATURE CO-OCCURRENCE ANALYSIS")
    print("-"*80)
    instance_results, class_results, is_new_structure = get_results_structure()
    results_to_analyze = instance_results  # Use instance-level for rule-based analyses
    for cls_key in sorted(results_to_analyze.keys()):
        rules = results_to_analyze[cls_key].get('rules', [])
        feature_pairs = defaultdict(int)
        
        for rule in rules:
            if rule and rule != "any values (no tightened features)":
                matches = re.findall(r'(\w+) ∈', rule)
                # Count pairs
                for pair in combinations(sorted(matches), 2):
                    feature_pairs[pair] += 1
        
        if feature_pairs:
            top_pairs = sorted(feature_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n{cls_key} - Most common feature pairs:")
            for (feat1, feat2), count in top_pairs:
                print(f"  ({feat1}, {feat2}): {count} times ({count/len([r for r in rules if r])*100:.1f}% of rules)")
    
    # 5. Rule Overlap/Sharing Analysis
    print("\n" + "-"*80)
    print("5. RULE OVERLAP ANALYSIS")
    print("-"*80)
    instance_results, class_results, is_new_structure = get_results_structure()
    results_to_analyze = instance_results  # Use instance-level for rule-based analyses
    for cls_key in sorted(results_to_analyze.keys()):
        rules = results_to_analyze[cls_key].get('rules', [])
        rule_counts = Counter(rules)
        
        if rules:
            unique_rules = len(set(rules))
            total_rules = len(rules)
            sharing_ratio = total_rules / unique_rules if unique_rules > 0 else 0
            
            print(f"\n{cls_key}:")
            print(f"  Total rules: {total_rules}")
            print(f"  Unique rules: {unique_rules}")
            print(f"  Sharing ratio: {sharing_ratio:.2f} (avg instances per unique rule)")
            
            # Most shared rules
            if rule_counts:
                most_shared = rule_counts.most_common(3)
                print(f"  Most shared rules:")
                for rule, count in most_shared:
                    if rule and rule != "any values (no tightened features)":
                        rule_preview = rule[:60] + "..." if len(rule) > 60 else rule
                        print(f"    '{rule_preview}': shared by {count} instances")
    
    # 6. Rule Efficiency Analysis (Precision per Feature)
    print("\n" + "-"*80)
    print("6. RULE EFFICIENCY ANALYSIS (Precision per Feature)")
    print("-"*80)
    instance_results, class_results, is_new_structure = get_results_structure()
    results_to_analyze = instance_results  # Use instance-level for rule-based analyses
    for cls_key in sorted(results_to_analyze.keys()):
        rules_data = results_to_analyze[cls_key].get('rules_with_instances', [])
        if rules_data:
            efficiencies = []
            for rule_data in rules_data:
                rule = rule_data.get('rule', '')
                precision = rule_data.get('hard_precision', 0.0)
                
                if rule and rule != "any values (no tightened features)":
                    matches = re.findall(r'(\w+) ∈', rule)
                    n_features = len(matches)
                    if n_features > 0:
                        efficiency = precision / n_features
                        efficiencies.append((n_features, precision, efficiency))
            
            if efficiencies:
                features_list, precisions_list, eff_list = zip(*efficiencies)
                print(f"\n{cls_key}:")
                print(f"  Mean efficiency (precision/feature): {np.mean(eff_list):.4f}")
                print(f"  Best efficiency: {np.max(eff_list):.4f} (precision={precisions_list[eff_list.index(np.max(eff_list))]:.3f}, features={features_list[eff_list.index(np.max(eff_list))]})")
                
                # Efficiency by complexity
                for n_feat in sorted(set(features_list)):
                    effs_for_n = [eff for n, _, eff in efficiencies if n == n_feat]
                    if effs_for_n:
                        print(f"  {n_feat}-feature rules: mean efficiency={np.mean(effs_for_n):.4f} (n={len(effs_for_n)})")
    
    # 7. Convergence Analysis
    print("\n" + "-"*80)
    print("7. CONVERGENCE ANALYSIS")
    print("-"*80)
    history = data['training_history']
    if len(history) >= 10:
        # Last 25% of episodes
        last_quarter = max(1, len(history) // 4)
        last_episodes = history[-last_quarter:]
        first_quarter = history[:last_quarter]
        
        # RL Precision convergence
        last_precisions = [e['rl_avg_precision'] for e in last_episodes]
        first_precisions = [e['rl_avg_precision'] for e in first_quarter]
        
        last_prec_mean = np.mean(last_precisions)
        last_prec_std = np.std(last_precisions)
        first_prec_mean = np.mean(first_precisions)
        
        print(f"\nRL Precision:")
        print(f"  First quarter mean: {first_prec_mean:.4f}")
        print(f"  Last quarter mean: {last_prec_mean:.4f}")
        print(f"  Last quarter std: {last_prec_std:.4f}")
        if last_prec_std < 0.05:
            print(f"  → Converged (low variance in final quarter)")
        else:
            print(f"  → Still varying (std={last_prec_std:.4f})")
        
        # RL Coverage convergence
        last_coverages = [e['rl_avg_coverage'] for e in last_episodes]
        first_coverages = [e['rl_avg_coverage'] for e in first_quarter]
        
        last_cov_mean = np.mean(last_coverages)
        last_cov_std = np.std(last_coverages)
        first_cov_mean = np.mean(first_coverages)
        
        print(f"\nRL Coverage:")
        print(f"  First quarter mean: {first_cov_mean:.4f}")
        print(f"  Last quarter mean: {last_cov_mean:.4f}")
        print(f"  Last quarter std: {last_cov_std:.4f}")
        if last_cov_std < 0.01:
            print(f"  → Converged (low variance in final quarter)")
        else:
            print(f"  → Still varying (std={last_cov_std:.4f})")
    
    # 8. Classifier-RL Correlation Analysis
    print("\n" + "-"*80)
    print("8. CLASSIFIER-RL CORRELATION ANALYSIS")
    print("-"*80)
    history = data['training_history']
    classifier_acc = [e['classifier_test_acc'] for e in history]
    rl_precision = [e['rl_avg_precision'] for e in history]
    rl_coverage = [e['rl_avg_coverage'] for e in history]
    
    if len(classifier_acc) > 1:
        corr_prec = np.corrcoef(classifier_acc, rl_precision)[0, 1]
        corr_cov = np.corrcoef(classifier_acc, rl_coverage)[0, 1]
        
        print(f"\nCorrelation between classifier accuracy and RL metrics:")
        print(f"  Classifier vs RL Precision: {corr_prec:.3f}")
        if abs(corr_prec) > 0.5:
            print(f"    → Strong correlation ({'positive' if corr_prec > 0 else 'negative'})")
        elif abs(corr_prec) > 0.3:
            print(f"    → Moderate correlation")
        else:
            print(f"    → Weak correlation")
        
        print(f"  Classifier vs RL Coverage: {corr_cov:.3f}")
        if abs(corr_cov) > 0.5:
            print(f"    → Strong correlation ({'positive' if corr_cov > 0 else 'negative'})")
        elif abs(corr_cov) > 0.3:
            print(f"    → Moderate correlation")
        else:
            print(f"    → Weak correlation")
    
    # 9. Class Comparison Summary
    print("\n" + "-"*80)
    print("9. CLASS COMPARISON SUMMARY")
    print("-"*80)
    class_summaries = []
    instance_results, class_results, is_new_structure = get_results_structure()
    results_to_analyze = instance_results  # Use instance-level for rule-based analyses
    for cls_key in sorted(results_to_analyze.keys()):
        cls_data = results_to_analyze[cls_key]
        rules_data = cls_data.get('rules_with_instances', [])
        
        summary = {
            'class': cls_key,
            'precision': cls_data['hard_precision'],
            'coverage': cls_data['coverage'],
            'best_precision': cls_data['best_precision'],
            'n_rules': len(cls_data['rules']),
            'unique_rules': len(set(cls_data['rules'])),
        }
        
        if rules_data:
            precisions = [r['hard_precision'] for r in rules_data if r.get('hard_precision', 0) > 0]
            coverages = [r['coverage'] for r in rules_data if r.get('coverage', 0) > 0]
            summary['mean_rule_precision'] = np.mean(precisions) if precisions else 0
            summary['mean_rule_coverage'] = np.mean(coverages) if coverages else 0
        
        class_summaries.append(summary)
        
        # Add class-level comparison if available
        if is_new_structure and cls_key in class_results:
            cls_class_data = class_results[cls_key]
            summary['class_level_precision'] = cls_class_data.get('hard_precision', 0.0)
            summary['class_level_coverage'] = cls_class_data.get('coverage', 0.0)
            summary['coverage_improvement'] = summary['class_level_coverage'] - summary['coverage']
    
    # Sort by precision
    class_summaries.sort(key=lambda x: x['precision'], reverse=True)
    print("\nClasses ranked by precision (Instance-Level):")
    for i, summary in enumerate(class_summaries, 1):
        print(f"  {i}. {summary['class']}: precision={summary['precision']:.3f}, coverage={summary['coverage']:.4f}, "
              f"best={summary['best_precision']:.3f}, rules={summary['n_rules']}/{summary['unique_rules']} unique")
        if 'class_level_coverage' in summary:
            print(f"      → Class-Level Coverage: {summary['class_level_coverage']:.4f} "
                  f"(improvement: {summary.get('coverage_improvement', 0):+.4f})")
    
    # 10. Feature Range Analysis (if bounds available)
    print("\n" + "-"*80)
    print("10. FEATURE RANGE ANALYSIS")
    print("-"*80)
    instance_results, class_results, is_new_structure = get_results_structure()
    results_to_analyze = instance_results  # Use instance-level for anchor analysis
    for cls_key in sorted(results_to_analyze.keys()):
        anchors = results_to_analyze[cls_key].get('anchors', [])
        if anchors and len(anchors) > 0:
            # Check if we have bounds
            first_anchor = anchors[0]
            if 'lower_bounds' in first_anchor and 'upper_bounds' in first_anchor:
                lower_bounds = first_anchor.get('lower_bounds', [])
                upper_bounds = first_anchor.get('upper_bounds', [])
                
                if lower_bounds and upper_bounds:
                    # Calculate average range widths across all anchors
                    all_ranges = []
                    for anchor in anchors:
                        if 'lower_bounds' in anchor and 'upper_bounds' in anchor:
                            lb = anchor['lower_bounds']
                            ub = anchor['upper_bounds']
                            if len(lb) == len(ub):
                                ranges = [ub[i] - lb[i] for i in range(len(lb)) if ub[i] > lb[i]]
                                all_ranges.extend(ranges)
                    
                    if all_ranges:
                        print(f"\n{cls_key}:")
                        print(f"  Mean feature range width: {np.mean(all_ranges):.4f}")
                        print(f"  Std feature range width: {np.std(all_ranges):.4f}")
                        print(f"  Min range: {np.min(all_ranges):.4f}, Max range: {np.max(all_ranges):.4f}")
    
    # Generate additional plots
    print("\n" + "="*80)
    print("GENERATING ADDITIONAL PLOTS")
    print("="*80)
    
    # Create additional analysis plots (including feature importance)
    fig2 = plt.figure(figsize=(24, 16))
    
    # Plot 1: Precision-Coverage scatter
    ax1 = plt.subplot(3, 4, 1)
    instance_results, class_results, is_new_structure = get_results_structure()
    results_to_analyze = instance_results  # Use instance-level for rule-based analyses
    for cls_key in sorted(results_to_analyze.keys()):
        rules_data = results_to_analyze[cls_key].get('rules_with_instances', [])
        if rules_data:
            precisions = [r['hard_precision'] for r in rules_data]
            coverages = [r['coverage'] for r in rules_data]
            ax1.scatter(coverages, precisions, label=cls_key, alpha=0.6, s=30)
    ax1.set_xlabel('Coverage')
    ax1.set_ylabel('Hard Precision')
    ax1.set_title('Precision vs Coverage Tradeoff')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rule complexity distribution
    ax2 = plt.subplot(3, 4, 2)
    all_complexities = []
    instance_results, class_results, is_new_structure = get_results_structure()
    results_to_analyze = instance_results  # Use instance-level for rule-based analyses
    for cls_key in sorted(results_to_analyze.keys()):
        rules = results_to_analyze[cls_key].get('rules', [])
        for rule in rules:
            if rule and rule != "any values (no tightened features)":
                matches = re.findall(r'(\w+) ∈', rule)
                all_complexities.append(len(matches))
    if all_complexities:
        ax2.hist(all_complexities, bins=range(max(all_complexities)+2), edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Number of Features per Rule')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Rule Complexity Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Precision distribution
    ax3 = plt.subplot(3, 4, 3)
    instance_results, class_results, is_new_structure = get_results_structure()
    results_to_analyze = instance_results  # Use instance-level for rule-based analyses
    for cls_key in sorted(results_to_analyze.keys()):
        rules_data = results_to_analyze[cls_key].get('rules_with_instances', [])
        if rules_data:
            precisions = [r['hard_precision'] for r in rules_data if r.get('hard_precision', 0) > 0]
            if precisions:
                ax3.hist(precisions, bins=20, alpha=0.5, label=cls_key, edgecolor='black')
    ax3.set_xlabel('Hard Precision')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Precision Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Coverage distribution
    ax4 = plt.subplot(3, 4, 4)
    instance_results, class_results, is_new_structure = get_results_structure()
    results_to_analyze = instance_results  # Use instance-level for rule-based analyses
    for cls_key in sorted(results_to_analyze.keys()):
        rules_data = results_to_analyze[cls_key].get('rules_with_instances', [])
        if rules_data:
            coverages = [r['coverage'] for r in rules_data if r.get('coverage', 0) > 0]
            if coverages:
                ax4.hist(coverages, bins=20, alpha=0.5, label=cls_key, edgecolor='black')
    ax4.set_xlabel('Coverage')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Coverage Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Classifier vs RL Precision over time
    ax5 = plt.subplot(3, 4, 5)
    ax5_twin = ax5.twinx()
    episodes = [e['episode'] for e in history]
    ax5.plot(episodes, classifier_acc, 'b-', label='Classifier Acc', linewidth=2)
    ax5_twin.plot(episodes, rl_precision, 'g-', label='RL Precision', linewidth=2)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Classifier Accuracy', color='b')
    ax5_twin.set_ylabel('RL Precision', color='g')
    ax5.set_title('Classifier vs RL Precision')
    ax5.tick_params(axis='y', labelcolor='b')
    ax5_twin.tick_params(axis='y', labelcolor='g')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Rule efficiency (precision per feature)
    ax6 = plt.subplot(3, 4, 6)
    instance_results, class_results, is_new_structure = get_results_structure()
    results_to_analyze = instance_results  # Use instance-level for rule-based analyses
    for cls_key in sorted(results_to_analyze.keys()):
        rules_data = results_to_analyze[cls_key].get('rules_with_instances', [])
        if rules_data:
            complexities = []
            precisions = []
            for rule_data in rules_data:
                rule = rule_data.get('rule', '')
                precision = rule_data.get('hard_precision', 0.0)
                if rule and rule != "any values (no tightened features)":
                    matches = re.findall(r'(\w+) ∈', rule)
                    n_features = len(matches)
                    if n_features > 0:
                        complexities.append(n_features)
                        precisions.append(precision)
            if complexities:
                ax6.scatter(complexities, precisions, label=cls_key, alpha=0.6, s=30)
    ax6.set_xlabel('Number of Features')
    ax6.set_ylabel('Hard Precision')
    ax6.set_title('Precision vs Rule Complexity')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Convergence analysis (last quarter)
    ax7 = plt.subplot(3, 4, 7)
    if len(history) >= 10:
        last_quarter = max(1, len(history) // 4)
        last_episodes_data = history[-last_quarter:]
        last_episodes = [e['episode'] for e in last_episodes_data]
        last_precisions = [e['rl_avg_precision'] for e in last_episodes_data]
        last_coverages = [e['rl_avg_coverage'] for e in last_episodes_data]
        
        ax7_twin = ax7.twinx()
        ax7.plot(last_episodes, last_precisions, 'g-', label='Precision', linewidth=2)
        ax7_twin.plot(last_episodes, last_coverages, 'm-', label='Coverage', linewidth=2)
        ax7.set_xlabel('Episode (Last Quarter)')
        ax7.set_ylabel('RL Precision', color='g')
        ax7_twin.set_ylabel('RL Coverage', color='m')
        ax7.set_title('Convergence Analysis (Last Quarter)')
        ax7.tick_params(axis='y', labelcolor='g')
        ax7_twin.tick_params(axis='y', labelcolor='m')
        ax7.grid(True, alpha=0.3)
    
    # Plot 8: Class comparison bar chart
    ax8 = plt.subplot(3, 4, 8)
    classes = [s['class'] for s in class_summaries]
    precisions = [s['precision'] for s in class_summaries]
    coverages = [s['coverage'] for s in class_summaries]
    
    x = np.arange(len(classes))
    width = 0.35
    ax8.bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
    ax8_twin = ax8.twinx()
    ax8_twin.bar(x + width/2, coverages, width, label='Coverage', alpha=0.8, color='orange')
    ax8.set_xlabel('Class')
    ax8.set_ylabel('Precision', color='blue')
    ax8_twin.set_ylabel('Coverage', color='orange')
    ax8.set_title('Class Performance Comparison')
    ax8.set_xticks(x)
    ax8.set_xticklabels(classes, rotation=45, ha='right')
    ax8.tick_params(axis='y', labelcolor='blue')
    ax8_twin.tick_params(axis='y', labelcolor='orange')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Plot 9: Global Feature Importance (Top 15)
    ax9 = plt.subplot(3, 4, 9)
    if global_scores:
        global_scores_sorted = sorted(global_scores, key=lambda x: x['global_score'], reverse=True)[:15]
        features = [f['feature'] for f in global_scores_sorted]
        scores = [f['global_score'] for f in global_scores_sorted]
        ax9.barh(range(len(features)), scores, alpha=0.7)
        ax9.set_yticks(range(len(features)))
        ax9.set_yticklabels(features)
        ax9.set_xlabel('Global Importance Score')
        ax9.set_title('Top 15 Global Feature Importance')
        ax9.invert_yaxis()
        ax9.grid(True, alpha=0.3, axis='x')
    
    # Plot 10: Feature Importance by Class (Top 10 features per class)
    ax10 = plt.subplot(3, 4, 10)
    if all_class_feature_importance:
        # Collect top features from each class
        top_features_by_class = {}
        for cls_key, feat_scores in all_class_feature_importance.items():
            sorted_feats = sorted(feat_scores, key=lambda x: x['score_precision_weighted'], reverse=True)[:10]
            top_features_by_class[cls_key] = {f['feature']: f['score_precision_weighted'] for f in sorted_feats}
        
        # Create heatmap data
        all_features = set()
        for cls_feats in top_features_by_class.values():
            all_features.update(cls_feats.keys())
        all_features = sorted(list(all_features))[:15]  # Limit to top 15 features
        
        if all_features and top_features_by_class:
            heatmap_data = []
            class_labels = []
            for cls_key in sorted(top_features_by_class.keys()):
                class_labels.append(cls_key)
                row = [top_features_by_class[cls_key].get(feat, 0) for feat in all_features]
                heatmap_data.append(row)
            
            im = ax10.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax10.set_xticks(range(len(all_features)))
            ax10.set_xticklabels(all_features, rotation=45, ha='right', fontsize=8)
            ax10.set_yticks(range(len(class_labels)))
            ax10.set_yticklabels(class_labels)
            ax10.set_title('Feature Importance Heatmap\n(Precision-Weighted)')
            plt.colorbar(im, ax=ax10, label='Importance Score')
    
    # Plot 11: Feature Frequency vs Average Precision
    ax11 = plt.subplot(3, 4, 11)
    if global_scores:
        frequencies = [f['count'] for f in global_scores]
        avg_precs = [f['avg_precision'] for f in global_scores]
        feature_names = [f['feature'] for f in global_scores]
        
        scatter = ax11.scatter(frequencies, avg_precs, alpha=0.6, s=100, c=range(len(frequencies)), cmap='viridis')
        ax11.set_xlabel('Feature Frequency (Count)')
        ax11.set_ylabel('Average Precision')
        ax11.set_title('Feature Frequency vs Precision')
        ax11.grid(True, alpha=0.3)
        
        # Annotate top features
        top_indices = sorted(range(len(global_scores)), key=lambda i: global_scores[i]['global_score'], reverse=True)[:5]
        for idx in top_indices:
            ax11.annotate(feature_names[idx], 
                         (frequencies[idx], avg_precs[idx]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 12: Instance-Level vs Class-Level Coverage Comparison
    ax12 = plt.subplot(3, 4, 12)
    instance_results, class_results, is_new_structure = get_results_structure()
    if is_new_structure and class_results:
        classes = sorted(set(instance_results.keys()) & set(class_results.keys()))
        if classes:
            instance_covs = [instance_results[cls].get('coverage', 0.0) for cls in classes]
            class_covs = [class_results[cls].get('coverage', 0.0) for cls in classes]
            
            x = np.arange(len(classes))
            width = 0.35
            ax12.bar(x - width/2, instance_covs, width, label='Instance-Level', alpha=0.8, color='skyblue')
            ax12.bar(x + width/2, class_covs, width, label='Class-Level', alpha=0.8, color='coral')
            ax12.set_xlabel('Class')
            ax12.set_ylabel('Coverage')
            ax12.set_title('Coverage: Instance vs Class Level\n(Dynamic Anchors Advantage)')
            ax12.set_xticks(x)
            ax12.set_xticklabels(classes, rotation=45, ha='right')
            ax12.legend()
            ax12.grid(True, alpha=0.3, axis='y')
            
            # Add improvement annotations
            for i, cls in enumerate(classes):
                improvement = class_covs[i] - instance_covs[i]
                if improvement > 0:
                    ax12.annotate(f'+{improvement:.3f}', 
                                xy=(i + width/2, class_covs[i]),
                                xytext=(0, 5), textcoords='offset points',
                                ha='center', fontsize=8, color='green', weight='bold')
        else:
            ax12.text(0.5, 0.5, 'No class-level data\navailable', 
                     ha='center', va='center', transform=ax12.transAxes)
            ax12.set_title('Coverage Comparison (N/A)')
    else:
        # Fallback: Feature Importance Comparison (if no class-level data)
        if global_scores:
            top_10_features = sorted(global_scores, key=lambda x: x['global_score'], reverse=True)[:10]
            features = [f['feature'] for f in top_10_features]
            
            # Calculate scores from available data
            prec_weighted_scores = [f['count'] * f['avg_precision'] for f in top_10_features]
            combined_scores = [f['count'] * f['avg_precision'] * f['avg_coverage'] for f in top_10_features]
            freq_scores = [f['count'] for f in top_10_features]
            
            # Normalize scores for comparison
            max_freq = max(freq_scores) if freq_scores else 1
            max_prec_weighted = max(prec_weighted_scores) if prec_weighted_scores else 1
            max_combined = max(combined_scores) if combined_scores else 1
            
            x_pos = np.arange(len(features))
            width_bar = 0.25
            
            # Normalize and plot
            freq_norm = [f / max_freq for f in freq_scores]
            prec_norm = [f / max_prec_weighted if max_prec_weighted > 0 else 0 for f in prec_weighted_scores]
            comb_norm = [f / max_combined if max_combined > 0 else 0 for f in combined_scores]
            
            ax12.bar(x_pos - width_bar, freq_norm, width_bar, label='Frequency (norm)', alpha=0.7)
            ax12.bar(x_pos, prec_norm, width_bar, label='Precision-Weighted (norm)', alpha=0.7)
            ax12.bar(x_pos + width_bar, comb_norm, width_bar, label='Combined Score (norm)', alpha=0.7)
            
            ax12.set_xlabel('Feature')
            ax12.set_ylabel('Normalized Score')
            ax12.set_title('Top 10 Features: Multiple Metrics')
            ax12.set_xticks(x_pos)
            ax12.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
            ax12.legend(fontsize=7)
            ax12.grid(True, alpha=0.3, axis='y')
        else:
            ax12.text(0.5, 0.5, 'No data available', 
                     ha='center', va='center', transform=ax12.transAxes)
            ax12.set_title('Plot 12 (N/A)')
    
    plt.tight_layout()
    
    # Save additional plots
    output_path2 = json_path.replace('.json', '_detailed_analysis.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved detailed analysis plots to: {output_path2}")
    
    # Create dedicated feature importance visualization
    if global_scores and all_class_feature_importance:
        print("\nGenerating feature importance visualizations...")
        fig3 = plt.figure(figsize=(20, 12))
        
        # Plot 1: Global feature importance bar chart
        ax1 = plt.subplot(2, 3, 1)
        top_15_global = sorted(global_scores, key=lambda x: x['global_score'], reverse=True)[:15]
        features = [f['feature'] for f in top_15_global]
        scores = [f['global_score'] for f in top_15_global]
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        ax1.barh(range(len(features)), scores, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features)
        ax1.set_xlabel('Global Importance Score')
        ax1.set_title('Top 15 Global Feature Importance')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Feature importance by class (stacked or grouped)
        ax2 = plt.subplot(2, 3, 2)
        if all_class_feature_importance:
            # Get top 10 features globally
            top_features = [f['feature'] for f in sorted(global_scores, key=lambda x: x['global_score'], reverse=True)[:10]]
            
            x = np.arange(len(top_features))
            width = 0.8 / len(all_class_feature_importance)
            offset = -0.4 + width/2
            
            for i, (cls_key, feat_scores) in enumerate(sorted(all_class_feature_importance.items())):
                scores = []
                for feat in top_features:
                    feat_data = next((f for f in feat_scores if f['feature'] == feat), None)
                    scores.append(feat_data['score_precision_weighted'] if feat_data else 0)
                ax2.bar(x + offset + i*width, scores, width, label=cls_key, alpha=0.7)
            
            ax2.set_xlabel('Feature')
            ax2.set_ylabel('Precision-Weighted Importance')
            ax2.set_title('Feature Importance by Class')
            ax2.set_xticks(x)
            ax2.set_xticklabels(top_features, rotation=45, ha='right', fontsize=8)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Feature frequency distribution
        ax3 = plt.subplot(2, 3, 3)
        frequencies = [f['count'] for f in global_scores]
        ax3.hist(frequencies, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax3.set_xlabel('Feature Frequency (Count)')
        ax3.set_ylabel('Number of Features')
        ax3.set_title('Feature Frequency Distribution')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Average precision vs average coverage scatter
        ax4 = plt.subplot(2, 3, 4)
        avg_precs = [f['avg_precision'] for f in global_scores]
        avg_covs = [f['avg_coverage'] for f in global_scores]
        counts = [f['count'] for f in global_scores]
        feature_names = [f['feature'] for f in global_scores]
        
        scatter = ax4.scatter(avg_precs, avg_covs, s=[c*5 for c in counts], alpha=0.6, c=counts, cmap='plasma')
        ax4.set_xlabel('Average Precision')
        ax4.set_ylabel('Average Coverage')
        ax4.set_title('Feature Quality: Precision vs Coverage\n(Size = Frequency)')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Frequency')
        
        # Annotate top features
        top_indices = sorted(range(len(global_scores)), key=lambda i: global_scores[i]['global_score'], reverse=True)[:5]
        for idx in top_indices:
            ax4.annotate(feature_names[idx], 
                        (avg_precs[idx], avg_covs[idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        # Plot 5: Number of classes each feature appears in
        ax5 = plt.subplot(2, 3, 5)
        n_classes_list = [f['n_classes'] for f in global_scores]
        features_sorted = sorted(global_scores, key=lambda x: x['n_classes'], reverse=True)[:15]
        features_names_sorted = [f['feature'] for f in features_sorted]
        n_classes_sorted = [f['n_classes'] for f in features_sorted]
        
        ax5.barh(range(len(features_names_sorted)), n_classes_sorted, alpha=0.7, color='coral')
        ax5.set_yticks(range(len(features_names_sorted)))
        ax5.set_yticklabels(features_names_sorted)
        ax5.set_xlabel('Number of Classes')
        ax5.set_title('Top 15 Features by Class Diversity')
        ax5.invert_yaxis()
        ax5.grid(True, alpha=0.3, axis='x')
        
        # Plot 6: Best precision achieved by each feature
        ax6 = plt.subplot(2, 3, 6)
        best_prec_sorted = sorted(global_scores, key=lambda x: x['best_precision'], reverse=True)[:15]
        features_best = [f['feature'] for f in best_prec_sorted]
        best_precs = [f['best_precision'] for f in best_prec_sorted]
        
        ax6.barh(range(len(features_best)), best_precs, alpha=0.7, color='lightgreen')
        ax6.set_yticks(range(len(features_best)))
        ax6.set_yticklabels(features_best)
        ax6.set_xlabel('Best Precision Achieved')
        ax6.set_title('Top 15 Features by Best Precision')
        ax6.invert_yaxis()
        ax6.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save feature importance plots
        output_path3 = json_path.replace('.json', '_feature_importance.png')
        plt.savefig(output_path3, dpi=150, bbox_inches='tight')
        print(f"Saved feature importance visualizations to: {output_path3}")

if __name__ == '__main__':
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'output/housing_joint/metrics_and_rules.json'
    try:
        analyze_metrics(json_path)
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        print("Usage: python analyze_metrics.py <path_to_metrics_and_rules.json>")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing metrics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

