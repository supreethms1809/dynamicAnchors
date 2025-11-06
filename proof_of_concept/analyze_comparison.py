#!/usr/bin/env python3
"""
Analysis script to understand why dynamic anchors show better metrics than static anchors.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_agnostic.dynamic_anchors.static_anchor import run_static_anchors
from model_agnostic.dynamic_anchors.dyn_anchor import train_dynamic_anchors


def analyze_differences():
    """
    Analyze key differences between static and dynamic anchors that explain the metric differences.
    """
    
    print("=" * 80)
    print("ANALYSIS: Why Dynamic Anchors Show Better Metrics")
    print("=" * 80)
    
    print("\n1. FUNDAMENTAL APPROACH DIFFERENCE:")
    print("-" * 80)
    print("Static Anchors:")
    print("  - Per-instance explanations (20 different anchors per class)")
    print("  - Each anchor optimized for ONE specific test instance")
    print("  - Coverage = per-instance coverage (how many perturbation samples satisfy anchor)")
    print("  - Each anchor can be very tight for that instance → lower average coverage")
    
    print("\nDynamic Anchors:")
    print("  - Per-class explanations (1 anchor per class)")
    print("  - ONE anchor optimized to work for ALL training instances of that class")
    print("  - Coverage = fraction of training instances in box")
    print("  - Must cover many instances → naturally broader → higher coverage")
    
    print("\n2. DATASET USED FOR METRICS:")
    print("-" * 80)
    print("Static Anchors:")
    print("  - Uses TEST instances (20 instances from test set)")
    print("  - Coverage calculated per test instance")
    print("  - Precision calculated on perturbation samples for each instance")
    
    print("\nDynamic Anchors:")
    print("  - Uses TRAINING data (X_train) for coverage calculation")
    print("  - Coverage = fraction of training instances in the box")
    print("  - Precision calculated on training instances in the box")
    print("  ⚠️  CRITICAL: This means comparison is NOT apples-to-apples!")
    print("     Dynamic uses train data, static uses test instances")
    
    print("\n3. OPTIMIZATION OBJECTIVE:")
    print("-" * 80)
    print("Static Anchors:")
    print("  - Bandit algorithm optimizes per instance")
    print("  - Goal: Find minimal anchor satisfying precision threshold for THIS instance")
    print("  - May sacrifice coverage for precision on that instance")
    print("  - Each anchor optimized independently")
    
    print("\nDynamic Anchors:")
    print("  - RL optimizes globally across all training instances")
    print("  - Reward = α·precision_gain + β·coverage_gain")
    print("     where α=0.7, β=0.6 (explicitly optimizes for coverage!)")
    print("  - Goal: Maximize both precision AND coverage globally")
    print("  - Co-training: Policy and classifier adapt together")
    
    print("\n4. WHY DYNAMIC SHOWS BETTER COVERAGE:")
    print("-" * 80)
    reasons = [
        "Per-class vs per-instance: One anchor must cover all instances → broader anchor",
        "Explicit coverage optimization: β=0.6 weight on coverage_gain in reward",
        "Global optimization: RL learns to balance precision and coverage globally",
        "Training data: May have different distribution than test instances",
        "RL training: Policy trained over many episodes to find optimal trade-off"
    ]
    for i, reason in enumerate(reasons, 1):
        print(f"  {i}. {reason}")
    
    print("\n5. WHY DYNAMIC SHOWS SIMILAR/HIGHER PRECISION:")
    print("-" * 80)
    reasons = [
        "RL training: Policy trained to maintain precision while expanding coverage",
        "Co-training: Classifier and policy adapt together, maintaining precision",
        "Greedy rollout: After training, selects actions maintaining precision",
        "Fallback mechanism: Uses best training episode if greedy fails",
        "Hard precision: Uses thresholded precision (≥0.95) matching static"
    ]
    for i, reason in enumerate(reasons, 1):
        print(f"  {i}. {reason}")
    
    print("\n6. POTENTIAL ISSUES IN CURRENT COMPARISON:")
    print("-" * 80)
    issues = [
        "Different datasets: Dynamic uses train, static uses test → unfair comparison",
        "Per-instance vs per-class: Fundamentally different explanation types",
        "Averaging mismatch: Check if dynamic is using 20 rollouts (episodes=1 suggests not)",
        "Coverage definition: Static uses per-instance coverage, dynamic uses global coverage"
    ]
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. ⚠️  {issue}")
    
    print("\n7. RECOMMENDATIONS FOR FAIR COMPARISON:")
    print("-" * 80)
    recommendations = [
        "Use same dataset: Make dynamic compute metrics on test data",
        "Ensure averaging: Verify num_greedy_rollouts=20 in dynamic",
        "Separate metrics: Report per-instance and per-class metrics separately",
        "Understand goal: Per-instance (static) vs per-class (dynamic) are different use cases",
        "Same coverage definition: Both should use same definition (global vs per-instance)"
    ]
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. ✓ {rec}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("Dynamic anchors show better coverage because:")
    print("  1. They find per-class rules (broader by design)")
    print("  2. They explicitly optimize for coverage (β=0.6 in reward)")
    print("  3. They use training data which may have different distribution")
    print("  4. They optimize globally vs per-instance")
    print("\nThe comparison may not be apples-to-apples due to:")
    print("  - Different datasets (train vs test)")
    print("  - Different explanation types (per-class vs per-instance)")
    print("  - Different coverage definitions")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_differences()

