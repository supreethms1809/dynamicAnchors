# Metrics Definitions

## Overview

This document provides comprehensive definitions of all metrics computed in the Dynamic Anchors system, with mathematical formulations and interpretations.

---

## 1. Precision Metrics

### 1.1 Hard Precision (`hard_precision`)

**Definition**: Precision depends on the mode (instance-based vs class-based).

#### Instance-Based Mode (when `x_star_unit` is set)

**Definition**: Fraction of samples in the anchor box where the model prediction matches the original instance's prediction.

**Formula**:
```
hard_precision = P(prediction matches original instance | x ∈ box)
              = (predictions == original_prediction).mean()
```

**Where**:
- `predictions`: Model predictions for samples in the box
- `original_prediction`: Model's prediction for the original instance (`x_star_unit`)

**Interpretation**:
- **1.0**: All samples in box get the same prediction as the original instance (perfect precision)
- **0.5**: Half of samples match the original prediction
- **0.0**: No samples match the original prediction

**Purpose**: Matches the original Anchor paper definition - measures prediction stability/consistency for a specific instance.

**Computation**:
- Uses model predictions (no ground truth labels needed)
- Original prediction is computed once when `x_star_unit` is set

#### Class-Based Mode (when `x_star_unit` is not set)

**Definition**: Fraction of samples in the anchor box that belong to the target class.

**Formula**:
```
hard_precision = P(y = target_class | x ∈ box)
              = (y_eval == target_class).mean()
```

**Where**:
- `y_eval`: Labels for samples in the box (from perturbation or full dataset)
- `target_class`: The class being explained

**Interpretation**:
- **1.0**: All samples in box are target class (perfect precision)
- **0.5**: Half of samples are target class
- **0.0**: No samples are target class

**Computation**:
- **With labels**: Uses ground truth labels
- **Without labels**: Uses model predictions as proxy

**Purpose**: Measures correctness - how well the anchor captures the target class.

**Code Location**: `BenchMARL/environment.py` `_current_metrics()` method

**Key Difference**: 
- **Instance-based**: Measures prediction matching (stability) - matches original Anchor paper
- **Class-based**: Measures class correctness (accuracy) - appropriate for class-level explanations

---

### 1.2 Average Probability (`avg_prob`)

**Definition**: Average predicted probability of target class for samples in the box.

**Formula**:
```
avg_prob = mean(probs[:, target_class])
```

**Where**:
- `probs`: Softmax outputs from classifier
- Shape: `[n_samples_in_box, n_classes]`

**Interpretation**:
- **1.0**: Model is 100% confident in target class
- **0.5**: Model is 50% confident
- **0.0**: Model predicts other classes

**Purpose**: Soft precision metric that accounts for model uncertainty

**Code Location**: `BenchMARL/environment.py` `_current_metrics()` method

---

### 1.3 Precision Proxy (`precision_proxy`)

**Definition**: Blended precision metric used for rewards and training.

**Formula**:
```
precision_proxy = λ * hard_precision + (1 - λ) * avg_prob
```

**Where**:
- `λ` = `precision_blend_lambda` (default: 0.5)

**Interpretation**:
- Balances hard precision (ground truth) with model confidence
- More stable for training than pure hard precision
- Accounts for both correctness and confidence

**Purpose**: Primary precision metric for reward computation

**Code Location**: `BenchMARL/environment.py` `_current_metrics()` method

---

### 1.4 Instance-Level Precision (`instance_precision`)

**Definition**: Average precision across all individual anchors/episodes for a class.

**Formula**:
```
instance_precision = mean([hard_precision_1, hard_precision_2, ..., hard_precision_N])
```

**Where**:
- `N`: Number of episodes/anchors for the class
- Each `hard_precision_i`: Precision of the i-th anchor

**Interpretation**:
- Average quality of individual anchor explanations
- Each anchor explains one instance/episode
- Comparable to baseline methods (Static Anchors)

**Use Case**: Compare with baseline methods that produce instance-level explanations

**Code Location**: `BenchMARL/inference.py` lines 2010-2012

---

### 1.5 Class-Level Precision (`class_precision` or `class_precision_union`)

**Definition**: Precision of the union of **class-based anchors only** for a class.

**Formula**:
```
class_precision = P(y = target_class | x ∈ union_of_class_based_boxes)
                = (y_union == target_class).mean()
```

**Where**:
- `union_mask`: Boolean mask indicating samples in ANY class-based anchor box
- `y_union`: Labels for samples in the union
- **Source**: Only class-based anchors (initialized from cluster centroids), NOT instance-based anchors

**Interpretation**:
- When combining ALL class-based anchors (OR operation), what fraction of samples are target class?
- Measures quality of the smallest set of general rules that explain the class structure
- Represents the most general explanation for the class (not instance-specific)
- Typically lower than instance-level (union includes more samples)

**Key Difference from Instance-Level**:
- Instance-level: Average of individual precisions (instance-based anchors)
- Class-level: Precision of combined class-based explanation (general rules only)

**Semantic Meaning**: 
- Represents the "smallest set of general rules that explain a class"
- Uses only class-based rules to capture class-level patterns, not instance-specific details

**Use Case**: Measure quality of complete class explanation using general rules

**Code Location**: `BenchMARL/inference.py` lines 2556-2626, `single_agent/single_agent_inference.py` lines 1133-1208

---

## 2. Coverage Metrics

### 2.1 Coverage (Instance-Based Mode)

**Definition**: Fraction of ALL samples that fall in the anchor box.

**Formula**:
```
coverage = P(x ∈ box)
         = mask.mean()
```

**Where**:
- `mask`: Boolean array indicating which samples are in box
- Computed over entire dataset (all classes)

**Interpretation**:
- **1.0**: Box covers all samples (very broad)
- **0.1**: Box covers 10% of samples
- **0.0**: Box covers no samples

**When Used**: When `x_star_unit` is set (instance-based mode)

**Code Location**: `BenchMARL/environment.py` `_current_metrics()` method

---

### 2.2 Coverage (Class-Based Mode)

**Definition**: Fraction of target class samples that fall in the anchor box.

**Formula**:
```
coverage = P(x ∈ box | y = target_class)
         = (mask & class_mask).sum() / n_class_samples
```

**Where**:
- `mask`: Boolean array for samples in box
- `class_mask`: Boolean array for target class samples
- `n_class_samples`: Total number of target class samples

**Interpretation**:
- **1.0**: Box covers all target class samples (perfect coverage)
- **0.5**: Box covers 50% of target class samples
- **0.0**: Box covers no target class samples

**When Used**: When `x_star_unit` is NOT set (class-based mode)

**Code Location**: `BenchMARL/environment.py` `_current_metrics()` method

---

### 2.3 Instance-Level Coverage (`instance_coverage`)

**Definition**: Average coverage across all individual anchors/episodes for a class.

**Formula**:
```
instance_coverage = mean([coverage_1, coverage_2, ..., coverage_N])
```

**Where**:
- `N`: Number of episodes/anchors
- Each `coverage_i`: Coverage of the i-th anchor

**Interpretation**:
- Average breadth of individual anchor explanations
- Each anchor covers some fraction of samples
- Comparable to baseline methods

**Use Case**: Compare with baseline methods

**Code Location**: `BenchMARL/inference.py` lines 2010-2012

**Note**: In single-agent inference, instance metrics are recomputed on the selected dataset before averaging; in multi-agent inference, instance metrics come directly from rollout estimates.

---

### 2.4 Class-Level Coverage (`class_coverage` or `class_coverage_union`)

**Definition**: Class-conditional coverage of the union of **class-based anchors only**.

**Formula**:
```
class_coverage = P(x ∈ union_of_class_based | y = target_class)
               = union_mask[mask_cls].mean()
```

**Where**:
- `union_mask`: Boolean mask for samples in ANY class-based anchor box
- `mask_cls`: Boolean mask for target class samples
- **Source**: Only class-based anchors (initialized from cluster centroids), NOT instance-based anchors

**Interpretation**:
- What fraction of target class samples are covered by at least one class-based anchor?
- Measures completeness of class explanation using general rules
- Represents coverage of the smallest set of general rules that explain the class structure
- Typically higher than instance-level (union covers more)

**Key Difference from Instance-Level**:
- Instance-level: Average coverage of individual instance-based anchors
- Class-level: Coverage of the union (OR) of all class-based anchors (general rules only)

**Semantic Meaning**:
- Represents the "smallest set of general rules that explain a class"
- Uses only class-based rules to capture class-level patterns

**Use Case**: Measure completeness of class explanation using general rules

**Code Location**: `BenchMARL/inference.py` lines 2556-2626, `single_agent/single_agent_inference.py` lines 1133-1208

---

### 2.5 Instance Coverage Class-Conditional (`instance_coverage_class_conditional`)

**Definition**: Class-conditional coverage for instance-based anchors.

**Formula**:
```
instance_coverage_class_conditional = P(x ∈ box | y = target_class)
                                      = (mask & class_mask).sum() / n_class_samples
```

**Purpose**: Provides class-conditional coverage even for instance-based anchors

**Interpretation**:
- For instance-based anchors, shows coverage relative to target class
- More meaningful than overall coverage when dataset is imbalanced
- Helps understand: "Of the target class samples, how many does this anchor cover?"

**Use Case**: Better interpretability for instance-based anchors in imbalanced datasets

**Code Location**: `BenchMARL/inference.py` lines 907-972, `single_agent/single_agent_inference.py` lines 433-457

---

## 3. Union Metrics

### 3.1 Per-Agent Union Precision/Coverage

**Definition**: Union metrics computed from one agent's anchors only.

**Formula**:
```
union_mask = anchor_1 OR anchor_2 OR ... OR anchor_N  (for one agent)
per_agent_precision = P(y = target | x ∈ union_mask)
per_agent_coverage = P(x ∈ union_mask | y = target)
```

**Purpose**: Measures quality of one agent's complete set of anchors

**When Used**: During inference, computed per agent before aggregation

**Code Location**: `BenchMARL/inference.py` lines 2026-2069

---

### 3.2 Class-Level Union Precision/Coverage

**Definition**: Union metrics computed from ALL agents' anchors.

**Formula**:
```
union_mask = agent_1_anchors OR agent_2_anchors OR ... OR agent_K_anchors
class_precision = P(y = target | x ∈ union_mask)
class_coverage = P(x ∈ union_mask | y = target)
```

**Purpose**: Measures quality of complete class explanation (all agents combined)

**When Used**: After all agents processed, final class-level metrics

**Code Location**: `BenchMARL/inference.py` lines 2181-2219

---

### 3.3 Class Union Metrics (Class-Based Only)

**Definition**: Union metrics computed from **class-based anchors only** (not instance-based), with precision filtering applied.

**Formula**:
```
# Step 1: Filter anchors by precision threshold
precision_threshold = precision_target * 0.8  # Default: 0.95 * 0.8 = 0.76
filtered_anchors = {anchor | actual_precision(anchor) >= precision_threshold}

# Step 2: Compute union of filtered anchors
class_based_union_mask = union of filtered_anchors
class_precision = P(y = target | x ∈ class_based_union_mask)
class_coverage = P(x ∈ class_based_union_mask | y = target)
```

**Precision Filtering**:
- **Threshold**: `precision_threshold = precision_target * 0.8`
  - Uses the same threshold as training/inference (80% of target precision)
  - Default: `0.95 * 0.8 = 0.76` when `precision_target = 0.95`

- **Precision Recalculation**: For each anchor, precision is **recomputed on the actual dataset** before filtering
  - Uses the same dataset selected by `coverage_on_all_data`/`eval_on_test_data` (full train+test when enabled, otherwise test/train)
  - **Not** the stored `instance_precision` from perturbation sampling
  - Ensures consistency: filtering uses same data and method as final union computation

- **Rationale**: The stored `instance_precision` comes from perturbation sampling during rollouts, which may not match the actual dataset precision. Recomputing ensures accurate filtering.

**Purpose**: Represents the smallest set of **high-quality general rules** that explain the class structure. Uses only class-based rules to capture class-level patterns, excluding instance-specific details and low-precision anchors.

**Rationale**: 
- Class-based anchors are initialized from cluster centroids, representing general class patterns
- Instance-based anchors are specific to individual samples and may not generalize
- Precision filtering ensures only high-quality anchors are included
- Using only class-based rules provides a more general, interpretable explanation

**When Used**: After class-based rollouts complete, as the final class-level union metrics

**Code Location**: `BenchMARL/inference.py` lines 2556-2680, `single_agent/single_agent_inference.py` lines 1133-1208

**Note**: This replaces the previous approach that included both instance-based and class-based anchors. The current implementation uses only class-based anchors for class union metrics to provide a cleaner, more general explanation. The precision filtering was added to ensure only high-quality rules are included in the union.

---

## 4. Class-Based Metrics

### 4.1 Class-Based Precision (`class_based_precision`)

**Definition**: Average precision across class-based rollouts.

**Formula**:
```
class_based_precision = mean([precision_cb1, precision_cb2, ..., precision_cbM])
```

**Where**:
- `M`: Number of class-based rollouts
- Each `precision_cbi`: Precision from i-th class-based rollout

**Source**: Rollouts initialized from cluster centroids (not specific instances)

**Interpretation**: Average quality of class-structure-based anchors

**Code Location**: `BenchMARL/inference.py` lines 2690-2796, `single_agent/single_agent_inference.py` lines 2657-2796

---

### 4.2 Class-Based Coverage (`class_based_coverage`)

**Definition**: Average coverage across class-based rollouts.

**Formula**:
```
class_based_coverage = mean([coverage_cb1, coverage_cb2, ..., coverage_cbM])
```

**Interpretation**: Average breadth of class-structure-based anchors

**Code Location**: `BenchMARL/inference.py` lines 2690-2796, `single_agent/single_agent_inference.py` lines 2657-2796

---

## 5. Standard Deviation Metrics

### 5.1 Instance Precision/Coverage Std

**Definition**: Standard deviation of instance-level metrics.

**Formula**:
```
instance_precision_std = std([precision_1, precision_2, ..., precision_N])
instance_coverage_std = std([coverage_1, coverage_2, ..., coverage_N])
```

**Purpose**: Measures variability in anchor quality across episodes

**Interpretation**:
- **Low std**: Consistent anchor quality
- **High std**: Variable anchor quality (may indicate instability)

**Code Location**: `BenchMARL/inference.py` lines 2116-2117

---

## 6. Rule Metrics

### 6.1 Unique Rules Count

**Definition**: Number of unique rule strings (after deduplication).

**Formula**:
```
unique_rules = set([rule_1, rule_2, ..., rule_N])
unique_rules_count = len(unique_rules)
```

**Purpose**: Measures diversity of extracted rules

**Interpretation**:
- **High count**: Diverse rules (good for coverage)
- **Low count**: Similar rules (may indicate lack of diversity)

**Code Location**: `BenchMARL/inference.py` line 2004

---

## 7. Timing Metrics

### 7.1 Average Rollout Time

**Definition**: Average time per rollout episode.

**Formula**:
```
avg_rollout_time = mean([time_1, time_2, ..., time_N])
```

**Unit**: Seconds

**Purpose**: Measures inference efficiency

**Code Location**: `BenchMARL/inference.py` line 2023

---

### 7.2 Total Rollout Time

**Definition**: Sum of all rollout times.

**Formula**:
```
total_rollout_time = sum([time_1, time_2, ..., time_N])
```

**Purpose**: Total inference time for all episodes

**Code Location**: `BenchMARL/inference.py` line 2024

---

## 8. Box Metrics

### 8.1 Box Widths

**Definition**: Width of box along each feature dimension.

**Formula**:
```
box_widths = upper - lower
```

**Shape**: `[n_features]`

**Purpose**: Measures how tight/loose the anchor is per feature

**Code Location**: `BenchMARL/inference.py` line 1994

---

### 8.2 Box Volume

**Definition**: Hypervolume of the anchor box.

**Formula**:
```
box_volume = prod(max(upper - lower, 1e-9))
```

**Purpose**: Overall measure of box size

**Interpretation**:
- **Large volume**: Broad anchor (covers more samples)
- **Small volume**: Tight anchor (covers fewer samples)

**Code Location**: `BenchMARL/inference.py` line 1995

---

## 9. Metric Relationships

### Instance-Level vs Class-Level

| Metric Type | Computation | What It Measures |
|-------------|-------------|------------------|
| **Instance-Level** | Average across episodes | Average quality of individual anchors |
| **Class-Level** | Union of all anchors | Quality of combined explanation |

**Key Insight**:
- Instance-level ≤ Class-level precision (union includes more samples)
- Instance-level ≤ Class-level coverage (union covers more)

### Precision vs Coverage Trade-off

**Relationship**:
- **High precision**: Box is tight, covers mostly target class
- **High coverage**: Box is broad, covers many samples
- **Trade-off**: Cannot maximize both simultaneously

**Targets**:
- `precision_target`: 0.95 (high precision)
- `coverage_target`: 0.5 (multi-agent), 0.3 (single-agent)

---

## 10. Metric Computation Locations

### During Training

- **Per-episode**: `_current_metrics()` in `environment.py`
- **Class union**: `_compute_class_union_metrics()` in `environment.py`
- **Logged**: `AnchorMetricsCallback` in `benchmarl_wrappers.py`

### During Inference

- **Per-episode**: `run_rollout_with_policy()` in `inference.py`
- **Aggregation**: `extract_rules_from_policies()` in `inference.py`
- **Union computation**: Lines 2040-2069, 2181-2219, 2556-2611

---

## 11. Metric Interpretation Guide

### Good Performance Indicators

1. **High Instance Precision** (>0.9): Individual anchors are accurate
2. **High Class Coverage** (>0.5): Complete explanation covers most of class
3. **Low Std** (<0.1): Consistent anchor quality
4. **Many Unique Rules**: Diverse explanations

### Warning Signs

1. **Low Coverage** (<0.1): Anchors are too narrow
2. **High Std** (>0.3): Unstable anchor generation
3. **Few Unique Rules**: Lack of diversity
4. **Zero Metrics**: Policy or environment issues

---

## Code References

- **Per-Episode Metrics**: `BenchMARL/environment.py` `_current_metrics()`
- **Class Union Metrics**: `BenchMARL/environment.py` `_compute_class_union_metrics()`
- **Inference Aggregation**: `BenchMARL/inference.py` lines 2010-2611
- **Metrics Documentation**: `docs/PRECISION_COVERAGE_METRICS.md`
