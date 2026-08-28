# Metrics Explanation - What Each Metric Actually Measures

## Overview

There are **4 distinct types of metrics** being computed and displayed. Understanding the difference is crucial:

---

## 1. **Instance-Level Precision/Coverage** (Row 1 in plots)

**What it measures**: Average precision/coverage across individual anchor boxes

**How it's computed**:
- For each rollout/episode, compute precision/coverage of that single anchor box
- Average across all rollouts: `mean([precision_1, precision_2, ..., precision_N])`

**Source**: 
- **Instance-based rollouts**: Anchors initialized around specific data instances
- Each anchor explains one instance
- Average shows: "On average, how good are individual instance explanations?"

**Formula**:
- Precision: `P(y = target_class | x in box)` for each box, then average
- Coverage: `P(x in box)` for instance-based, or `P(x in box | y = target_class)` for class-based, then average

**Use case**: Compare with baseline methods (Static Anchors) which also produce instance-level explanations

---

## 2. **Class-Level Precision/Coverage (Union)** (Row 2 in plots)

**What it measures**: Precision/coverage of the **UNION** of **class-based anchors only**

**How it's computed**:
1. Take ALL **class-based anchors** for a class (initialized from cluster centroids)
2. Form their union: `union_mask = class_based_anchor_1 OR class_based_anchor_2 OR ... OR class_based_anchor_M`
3. Compute precision/coverage on this union

**Source**: Only from **class-based rollouts** (boxes around cluster centroids), NOT instance-based rollouts

**Formula**:
- Precision: `P(y = target_class | x in union_of_class_based_boxes)`
- Coverage: `P(x in union_of_class_based | y = target_class)` (class-conditional)

**Semantic Meaning**: 
- Represents the "smallest set of general rules that explain the class structure"
- Uses only class-based rules to capture class-level patterns, excluding instance-specific details

**Key difference**: 
- Instance-level = average of individual instance-based boxes
- Class-level (union) = combined explanation from class-based boxes only (general rules)

**Use case**: Measures how well the smallest set of general rules explains the entire class

---

## 3. **Class-Based Precision/Coverage** (Row 3 in plots, if available)

**What it measures**: Precision/coverage from **separate rollouts** initialized around class centroids

**How it's computed**:
- Run separate rollouts initialized around k-means cluster centroids (NOT specific instances)
- For each class-based rollout, compute precision/coverage
- Average across class-based rollouts: `mean([precision_cb1, precision_cb2, ...])`

**Source**: **Class-based rollouts** (boxes initialized from cluster centroids)

**Formula**:
- Precision: `P(y = target_class | x in box)` for each class-based box, then average
- Coverage: `P(x in box | y = target_class)` for each class-based box, then average

**Key difference from instance-based**:
- Instance-based: Boxes around specific data instances
- Class-based: Boxes around cluster centroids (representative class points)

**Use case**: Measures how well anchors initialized from class structure (centroids) explain the class

**NOTE**: Class-based rollouts ALSO compute union metrics (`class_precision`/`class_coverage` in the class_based_results), but these are currently NOT being extracted/displayed. We're only showing the average metrics.

---

## 4. **Global Metrics** (Summary across all classes)

**What it measures**: Average metrics across all classes

**How it's computed**:
- Take per-class metrics (instance-level, class-level, or class-based)
- Average across all classes: `mean([metric_class1, metric_class2, ...])`

**Types**:
- `mean_instance_precision`: Average of instance-level precisions across classes
- `mean_instance_coverage`: Average of instance-level coverages across classes
- `mean_class_precision`: Average of class-level (union) precisions across classes
- `mean_class_coverage`: Average of class-level (union) coverages across classes
- `mean_class_based_precision`: Average of class-based precisions across classes
- `mean_class_based_coverage`: Average of class-based coverages across classes

**Use case**: Overall performance summary across all classes

---

## Summary Table

| Metric Type | Source Rollouts | Computation | What It Measures |
|------------|----------------|-------------|------------------|
| **Instance-Level** | Instance-based (around instances) | Average across individual anchors | Average quality of individual instance explanations |
| **Class-Level (Union)** | Class-based (around centroids) | Union of all class-based anchors | Quality of smallest set of general rules that explain class |
| **Class-Based** | Class-based (around centroids) | Average across class-based anchors | Average quality of class-structure-based explanations |
| **Global** | All of the above | Average across classes | Overall performance summary |

---

## Current Issue: Class-Based Metrics Confusion

**Problem**: Class-based rollouts compute TWO sets of metrics:
1. `instance_precision`/`instance_coverage` - Average across class-based rollouts
2. `class_precision`/`class_coverage` - Union of class-based anchors

**Current behavior**: We're extracting `instance_precision`/`instance_coverage` from class-based results and calling them "class_based_precision"/"class_based_coverage", which is correct.

**However**: We're NOT extracting the union metrics from class-based rollouts (`class_precision`/`class_coverage` in class_based_results), which would show the union of all class-based anchors.

**Recommendation**: 
- Keep current naming: "class_based_precision"/"class_based_coverage" = average across class-based rollouts
- Optionally add: "class_based_union_precision"/"class_based_union_coverage" = union of class-based anchors (if needed)

---

## What Each Row in the Plot Shows

**Row 1 (Instance-Level)**:
- Single-Agent: Average precision/coverage of instance-based anchors
- Multi-Agent: Average precision/coverage of instance-based anchors
- Baseline: Average precision/coverage of static anchors

**Row 2 (Class-Level Union)**:
- Single-Agent: Union of all class-based anchors (general rules only)
- Multi-Agent: Union of all class-based anchors (general rules only)
- Baseline: NOT shown (not fair comparison)
- **Note**: Uses only class-based rules, not instance-based, to represent the smallest set of general rules

**Row 3 (Class-Based)**:
- Single-Agent: Average precision/coverage of class-based anchors (from centroids)
- Multi-Agent: Average precision/coverage of class-based anchors (from centroids)
- Baseline: NOT shown (baseline doesn't do class-based rollouts)

---

## Key Insight

The confusion comes from the fact that:
1. **Instance-based rollouts** produce both instance-level (average) AND class-level (union) metrics
2. **Class-based rollouts** ALSO produce both instance-level (average) AND class-level (union) metrics
3. But we're only displaying:
   - Instance-based → instance-level (average) AND class-level (union)
   - Class-based → only instance-level (average), NOT the union

This is actually correct for comparison purposes, but the naming could be clearer!

