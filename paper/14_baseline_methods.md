# Baseline Methods

## Overview

This document describes the baseline explainability methods used for comparison with Dynamic Anchors, including Static Anchors, LIME, SHAP, and Feature Importance.

---

## Baseline Methods

### 1. Static Anchors

**Method**: Original Anchor algorithm (from anchor-exp library)

**Location**: `baseline/establish_baseline.py` `run_static_anchors()`

**Process**:
1. For each instance, generate one anchor explanation
2. Uses beam search to find minimal anchor
3. Computes precision and coverage per anchor

**Key Features**:
- Instance-level explanations (one per instance)
- Deterministic (no learning)
- Fast generation

**Metrics**:
- `instance_precision`: Average precision across anchors
- `instance_coverage`: Average coverage across anchors
- `class_precision`: Union precision (all anchors combined)
- `class_coverage`: Union coverage (all anchors combined)

**Code Reference**: `baseline/establish_baseline.py` lines 800-1100

---

### 2. LIME (Local Interpretable Model-agnostic Explanations)

**Method**: Local linear approximation

**Location**: `baseline/establish_baseline.py` `run_lime()`

**Process**:
1. Sample perturbed instances around target
2. Train local linear model
3. Extract feature importance from linear model

**Key Features**:
- Instance-level explanations
- Model-agnostic
- Linear approximations

**Metrics**:
- Feature importance scores per instance
- Average feature importance per class

**Code Reference**: `baseline/establish_baseline.py` lines 1100-1328

---

### 3. SHAP (SHapley Additive exPlanations)

**Method**: Shapley values from game theory

**Location**: `baseline/establish_baseline.py` `run_shap()`

**Process**:
1. Compute Shapley values for each feature
2. Uses KernelSHAP or TreeSHAP
3. Aggregates across instances

**Key Features**:
- Instance-level or global explanations
- Theoretically grounded (game theory)
- Additive feature contributions

**Metrics**:
- SHAP values per feature per instance
- Average SHAP values per class

**Code Reference**: `baseline/establish_baseline.py` lines 1329-1567

---

### 4. Feature Importance (Permutation Importance)

**Method**: Permute features and measure accuracy drop

**Location**: `baseline/establish_baseline.py` `run_feature_importance()`

**Process**:
1. Get baseline accuracy
2. For each feature:
   - Permute feature values
   - Measure accuracy drop
   - Importance = baseline_accuracy - permuted_accuracy
3. Repeat multiple times for stability

**Key Features**:
- Global method (one score per feature)
- Model-specific
- Measures feature impact on classifier

**Metrics**:
- Feature importance scores (global)
- Sorted features by importance

**Code Reference**: `baseline/establish_baseline.py` lines 1570-1654

---

## Comparison Methodology

### Fair Comparison

**Instance-Level Metrics**:
- Compare average precision/coverage across instances
- Both Dynamic Anchors and Static Anchors produce instance-level explanations
- Fair comparison: same granularity

**Class-Level Metrics**:
- Compare union precision/coverage
- Dynamic Anchors optimizes for class-level
- Static Anchors: union of instance-level anchors

---

## Metric Computation

### Static Anchors Metrics

**Per Instance**:
- `precision`: Precision of anchor for this instance
- `coverage`: Coverage of anchor for this instance

**Aggregated**:
- `instance_precision`: Mean precision across instances
- `instance_coverage`: Mean coverage across instances
- `class_precision`: Union precision (all anchors)
- `class_coverage`: Union coverage (all anchors)

**Code Reference**: `baseline/establish_baseline.py` `compute_class_union_metrics()`

---

### LIME Metrics

**Per Instance**:
- Feature importance scores
- Local model coefficients

**Aggregated**:
- Average feature importance per class
- Most important features per class

**Code Reference**: `baseline/establish_baseline.py` `run_lime()`

---

### SHAP Metrics

**Per Instance**:
- SHAP values per feature
- Feature contributions

**Aggregated**:
- Average SHAP values per class
- Feature importance rankings

**Code Reference**: `baseline/establish_baseline.py` `run_shap()`

---

### Feature Importance Metrics

**Global**:
- One importance score per feature
- Sorted by importance

**Not Instance-Level**: Global method, not comparable to instance-level methods

**Code Reference**: `baseline/establish_baseline.py` `run_feature_importance()`

---

## Comparison Plots

### Precision-Coverage Comparison

**Shows**:
- Instance-level precision/coverage
- Class-level precision/coverage
- Comparison across methods

**Methods Compared**:
- Dynamic Anchors (single-agent)
- Dynamic Anchors (multi-agent)
- Static Anchors
- (LIME/SHAP not shown - different metric type)

**Code Reference**: `plot_comparison.py` `plot_precision_coverage_comparison()`

---

### Feature Importance Comparison

**Shows**:
- Top features by importance
- Comparison across methods

**Methods Compared**:
- Dynamic Anchors (from rules)
- LIME (average importance)
- SHAP (average values)
- Permutation Importance (global)

**Code Reference**: `plot_comparison.py` `plot_feature_importance_subplot()`

---

## Baseline Implementation

### Running Baselines

**Script**: `baseline/establish_baseline.py`

**Process**:
1. Load dataset
2. Train classifier
3. Run each baseline method
4. Save results to JSON

**Output**: `baseline_results.json` with metrics for each method

---

### Baseline Analysis

**Script**: `baseline/analyze_baseline.py`

**Process**:
1. Load baseline results
2. Generate plots
3. Print summary statistics

**Output**: Plots and summary tables

---

## Key Differences

### Dynamic Anchors vs Static Anchors

| Aspect | Dynamic Anchors | Static Anchors |
|--------|----------------|----------------|
| **Method** | RL-based optimization | Beam search |
| **Learning** | Yes (trained policies) | No (deterministic) |
| **Optimization** | Class-level or instance-level | Instance-level only |
| **Coverage** | Optimized for coverage | Minimal anchor |
| **Speed** | Slower (training) | Faster (no training) |

---

### Dynamic Anchors vs LIME/SHAP

| Aspect | Dynamic Anchors | LIME/SHAP |
|--------|----------------|-----------|
| **Explanation Type** | Rules (intervals) | Feature importance |
| **Granularity** | Instance or class | Instance |
| **Interpretability** | Human-readable rules | Feature scores |
| **Coverage** | Explicit coverage metric | Not applicable |

---

## Code References

- **Static Anchors**: `baseline/establish_baseline.py` `run_static_anchors()`
- **LIME**: `baseline/establish_baseline.py` `run_lime()`
- **SHAP**: `baseline/establish_baseline.py` `run_shap()`
- **Feature Importance**: `baseline/establish_baseline.py` `run_feature_importance()`
- **Comparison**: `plot_comparison.py`

---

## Summary

**Baseline Methods**:

1. **Static Anchors**: Original Anchor algorithm (instance-level)
2. **LIME**: Local linear approximations (feature importance)
3. **SHAP**: Shapley values (feature contributions)
4. **Feature Importance**: Permutation importance (global)

**Comparison Strategy**:

- **Instance-Level**: Compare with Static Anchors (same granularity)
- **Feature Importance**: Compare with LIME/SHAP (different but related)
- **Class-Level**: Dynamic Anchors advantage (optimized for class)

**For Paper Writing**:

- Explain why each baseline is chosen
- Describe comparison methodology
- Show precision/coverage comparisons
- Discuss advantages of Dynamic Anchors

