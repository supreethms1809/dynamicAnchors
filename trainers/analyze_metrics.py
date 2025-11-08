#!/usr/bin/env python3
"""Quick analysis script for metrics_and_rules.json"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import sys

def analyze_metrics(json_path):
    """Analyze metrics_and_rules.json file and generate plots."""
    
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
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
    print(f"Overall Precision: {overall['overall_precision']:.4f}")
    print(f"Overall Coverage: {overall['overall_coverage']:.4f}")
    
    print("\n" + "="*80)
    print("PER-CLASS RESULTS (Final Evaluation)")
    print("="*80)
    for cls_key in sorted(data['per_class_results'].keys()):
        cls_data = data['per_class_results'][cls_key]
        print(f"\n{cls_key}:")
        print(f"  Hard Precision: {cls_data['hard_precision']:.4f}")
        print(f"  Coverage: {cls_data['coverage']:.4f}")
        print(f"  Best Rule Precision: {cls_data['best_precision']:.4f}")
        print(f"  Number of rules: {len(cls_data['rules'])}")
        print(f"  Unique rules: {len(set(cls_data['rules']))}")
        
        # Analyze rule quality
        rules_data = cls_data.get('rules_with_instances', [])
        if rules_data:
            precisions = [r['hard_precision'] for r in rules_data if r.get('hard_precision', 0) > 0]
            coverages = [r['coverage'] for r in rules_data if r.get('coverage', 0) > 0]
            if precisions:
                print(f"  Rule Precision - Mean: {np.mean(precisions):.3f}, "
                      f"Std: {np.std(precisions):.3f}, "
                      f"Min: {np.min(precisions):.3f}, "
                      f"Max: {np.max(precisions):.3f}")
            if coverages:
                print(f"  Rule Coverage - Mean: {np.mean(coverages):.3f}, "
                      f"Std: {np.std(coverages):.3f}")
    
    print("\n" + "="*80)
    print("TRAINING PROGRESS")
    print("="*80)
    history = data['training_history']
    episodes = [e['episode'] for e in history]
    classifier_acc = [e['classifier_test_acc'] for e in history]
    classifier_loss = [e['classifier_loss'] for e in history]
    rl_precision = [e['rl_avg_precision'] for e in history]
    rl_coverage = [e['rl_avg_coverage'] for e in history]
    
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
            print(f"  Initial precision: {cls_precisions[0]:.4f} → Final: {cls_precisions[-1]:.4f}")
            print(f"  Initial coverage: {cls_coverages[0]:.4f} → Final: {cls_coverages[-1]:.4f}")
            print(f"  Max precision: {max(cls_precisions):.4f} (episode {episodes[cls_precisions.index(max(cls_precisions))]})")
            print(f"  Max coverage: {max(cls_coverages):.4f} (episode {episodes[cls_coverages.index(max(cls_coverages))]})")
    
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
    for cls_key in sorted(data['per_class_results'].keys()):
        rules = data['per_class_results'][cls_key]['rules']
        features = []
        for rule in rules:
            matches = re.findall(r'(\w+) ∈', rule)
            features.extend(matches)
        
        if features:
            feature_counts = Counter(features)
            print(f"\n{cls_key} - Most common features:")
            for feat, count in feature_counts.most_common(5):
                print(f"  {feat}: {count} times ({count/len(rules)*100:.1f}% of rules)")

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

