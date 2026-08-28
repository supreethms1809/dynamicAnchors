# Precision and Coverage Metrics Explained

This document explains all the different precision and coverage metrics computed in the Dynamic Anchors codebase.

## Overview

The codebase computes multiple precision and coverage metrics at different levels of granularity. Understanding the differences is crucial for interpreting results correctly.

---

## 1. **Per-Episode/Per-Anchor Metrics** (During Training/Inference)

These are computed for each individual anchor/episode in `_current_metrics()`.

### 1.1 `precision_proxy`
- **Definition**: Primary precision metric used for rewards and training
- **Formula**: `λ * hard_precision + (1 - λ) * avg_prob`
- **Components**:
  - `hard_precision`: Ground truth precision (see below)
  - `avg_prob`: Average predicted probability of target class for samples in box
  - `precision_blend_lambda`: Weighting factor (default 0.5)
- **Purpose**: Balances hard precision with model confidence for training stability
- **Used for**: Reward calculation, termination conditions, training signal

### 1.2 `hard_precision`
- **Definition**: True precision based on ground truth labels
- **Formula**: 
  - When labels available: `P(y = target_class | x in box) = (y_eval == target_class).mean()`
  - When labels unavailable (uniform sampling): `P(model predicts target | x in box) = positive_idx.mean()`
- **What it measures**: 
  - **With labels**: Fraction of samples in box that are actually target class
  - **Without labels**: Fraction of samples in box where model predicts target class (proxy)
- **Purpose**: Ground truth measurement of anchor quality

### 1.3 `avg_prob`
- **Definition**: Average predicted probability of target class
- **Formula**: `mean(probs[:, target_class])` where `probs` are softmax outputs
- **What it measures**: Average confidence the model has in predicting target class for samples in box
- **Purpose**: Soft precision metric that accounts for model uncertainty

### 1.4 `target_class_fraction`
- **Definition**: Alias for `hard_precision` when labels are available
- **Formula**: Same as `hard_precision` when `y_eval` is not None
- **Purpose**: Stores the fraction of samples in box that belong to target class
- **Used for**: Bonus rewards when precision is below threshold

### 1.5 `coverage`
- **Definition**: Coverage of the anchor box (context-dependent)
- **Formula** (Instance-based mode, when `x_star_unit` is set):
  - `P(x in box) = mask.mean()` (Overall coverage - matches original Anchor)
- **Formula** (Class-based mode, when `x_star_unit` is None):
  - `P(x in box | y = target_class) = (mask & class_mask).sum() / n_class_samples` (Class-conditional coverage)
- **What it measures**:
  - **Instance-based**: Fraction of ALL samples that fall in box
  - **Class-based**: Fraction of target class samples that fall in box
- **Purpose**: Measures how general/broad the anchor explanation is

---

## 2. **Instance-Level Metrics** (During Inference)

These are computed for each episode/rollout and then averaged across all episodes for a class.

### 2.1 `instance_precision` (also `anchor_precision`)
- **Definition**: Average precision across all individual anchors/episodes for a class
- **Formula**: `mean([hard_precision_1, hard_precision_2, ..., hard_precision_N])`
- **What it measures**: Average precision of individual anchors (each explaining one instance/episode)
- **Purpose**: Measures quality of individual anchor explanations
- **Used for**: Comparing with baseline methods (Static Anchors), which also produce instance-level explanations

### 2.2 `instance_coverage` (also `anchor_coverage`)
- **Definition**: Average coverage across all individual anchors/episodes for a class
- **Formula**: `mean([coverage_1, coverage_2, ..., coverage_N])`
- **What it measures**: Average coverage of individual anchors
- **Purpose**: Measures average breadth of individual anchor explanations
- **Used for**: Comparing with baseline methods

### 2.3 `instance_precision_std` / `instance_coverage_std`
- **Definition**: Standard deviation of instance-level metrics
- **Formula**: `std([metric_1, metric_2, ..., metric_N])`
- **Purpose**: Measures variability in anchor quality across episodes

---

## 3. **Class-Level Metrics** (Union Metrics)

These are computed from the **union of class-based anchors only** for a class.

### 3.1 `class_precision` (also `class_precision_union`)
- **Definition**: Precision of the union of **class-based anchors only** for a class
- **Formula**: `P(y = target_class | x in union_of_class_based_boxes) = (y_union == target_class).mean()`
- **What it measures**: When you take ALL class-based anchors for a class and form their union (OR operation), what fraction of samples in that union are actually target class?
- **Source**: Only class-based anchors (initialized from cluster centroids), NOT instance-based anchors
- **Purpose**: Measures quality of the smallest set of general rules that explain the class structure
- **Semantic Meaning**: Represents the most general explanation for the class (not instance-specific)
- **Key difference from instance-level**: 
  - Instance-level: Average of individual instance-based anchor precisions
  - Class-level: Precision of the combined class-based explanation (general rules only)

### 3.2 `class_coverage` (also `class_coverage_union`)
- **Definition**: Class-conditional coverage of the union of **class-based anchors only**
- **Formula**: `P(x in union_of_class_based | y = target_class) = union_mask[mask_cls].mean()`
- **What it measures**: What fraction of target class samples are covered by at least one class-based anchor?
- **Source**: Only class-based anchors (initialized from cluster centroids), NOT instance-based anchors
- **Purpose**: Measures completeness of class explanation using general rules
- **Semantic Meaning**: Represents coverage of the smallest set of general rules that explain the class structure
- **Key difference from instance-level**:
  - Instance-level: Average coverage of individual instance-based anchors
  - Class-level: Coverage of the union (OR) of all class-based anchors (general rules only)

### 3.3 `class_precision_avg` / `class_coverage_avg` (Legacy)
- **Definition**: Average of class-level metrics across episodes (NOT union!)
- **Warning**: This is misleading - it's the average of class-level metrics, not the union metric
- **Note**: Kept for backward compatibility but `class_precision`/`class_coverage` are the correct union metrics

---

## 4. **Baseline Metrics** (Static Anchors)

Static Anchors (from anchor-exp library) also compute similar metrics:

### 4.1 `instance_precision` / `instance_coverage` (Baseline)
- **Definition**: Average precision/coverage across all individual instance explanations
- **From**: `anchor-exp` library's `exp.precision()` and `exp.coverage()`
- **Note**: Static Anchors generates one anchor per instance, so these are instance-level

### 4.2 `class_precision` / `class_coverage` (Baseline)
- **Definition**: Union metrics computed from all instance-level anchors for a class
- **Formula**: Same as class-level union metrics in Dynamic Anchors
- **Purpose**: Can be computed but typically not used for fair comparison (union of 20 instance anchors vs single optimized class anchor)

---

## 5. **During Training Metrics** (Real-time)

These are computed during training steps:

### 5.1 `anchor_precision` / `anchor_coverage` (in `infos`)
- **Definition**: Current precision/coverage for an agent's anchor box
- **Source**: From `_current_metrics()` return values
- **Used for**: Logging, callbacks, monitoring training progress

### 5.2 `class_union_precision` / `class_union_coverage` (in `infos`)
- **Definition**: Real-time union metrics for the agent's class
- **Source**: From `_compute_class_union_metrics()`
- **Used for**: Class-level rewards, shared rewards, multi-agent cooperation

### 5.3 `global_coverage`
- **Definition**: Overall dataset coverage by union of ALL agents' anchors
- **Formula**: `P(x in union_of_all_agents_boxes)`
- **Purpose**: Measures global explainability coverage across all classes

---

## 6. **Key Distinctions**

### Instance-Level vs Class-Level

| Aspect | Instance-Level | Class-Level (Union) |
|--------|----------------|---------------------|
| **What it measures** | Average quality of individual anchors | Quality of combined explanation |
| **Computation** | Average across episodes | Union (OR) of all anchors |
| **Use case** | Compare with Static Anchors | Measure complete class explanation |
| **For Dynamic Anchors** | 20+ anchors per class | One unified explanation per class |

### Precision vs Coverage

| Metric | What it measures | Formula (with labels) |
|--------|------------------|----------------------|
| **Precision** | Accuracy: Are samples in box correct? | `P(y = target \| x in box)` |
| **Coverage** | Completeness: How much does box cover? | `P(x in box \| y = target)` (class-based) |

### Instance-Based vs Class-Based Coverage

| Mode | Coverage Definition | When Used |
|------|---------------------|-----------|
| **Instance-based** | `P(x in box)` - Overall coverage | When `x_star_unit` is set (explaining specific instance) |
| **Class-based** | `P(x in box \| y = target)` - Class-conditional | When `x_star_unit` is None (explaining entire class) |

---

## 7. **Which Metrics to Use for Comparison**

### For Fair Comparison with Static Anchors:
- Use **instance-level** metrics: `instance_precision`, `instance_coverage`
- Both methods produce multiple anchors per class (one per instance)
- Average them for fair comparison

### For Evaluating Dynamic Anchors' Class Explanation:
- Use **class-level** metrics: `class_precision`, `class_coverage`
- These measure the union of **class-based anchors only** (the smallest set of general rules)
- Represents the most general explanation for the class structure
- Uses only class-based rules (not instance-based) to provide a cleaner, more interpretable explanation

### For Training Monitoring:
- Use **per-episode** metrics: `anchor_precision`, `anchor_coverage`
- Monitor `precision_proxy` for reward signals
- Track `class_union_precision`/`class_union_coverage` for multi-agent cooperation

---

## 8. **Example Interpretation**

Suppose for class 0:
- `instance_precision = 0.92`: Average of 20 individual anchors, each has ~92% precision
- `class_precision = 0.88`: When combining all 20 anchors (union), 88% of samples in union are class 0
- `instance_coverage = 0.15`: Each anchor covers ~15% of target class samples on average
- `class_coverage = 0.65`: The union of all 20 anchors covers 65% of target class samples

This means:
- Individual anchors are very precise (92%) but narrow (15% coverage each)
- Combined explanation is still precise (88%) and covers more of the class (65%)

---

## 9. **Code Locations**

- **Per-episode metrics**: `_current_metrics()` in `Environment.py` and `single_agentENV.py`
- **Class union metrics**: `_compute_class_union_metrics()` in `Environment.py`
- **Inference aggregation**: `extract_rules_from_policies()` in `inference.py`
- **Baseline metrics**: `compute_class_union_metrics()` in `establish_baseline.py`
