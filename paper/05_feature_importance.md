# Feature Importance

## Overview

Feature importance measures which features are most critical for the classifier's predictions. The Dynamic Anchors system computes feature importance based on rule frequency and interval selectivity.

---

## Feature Importance Computation

### Method: Rule-Based Feature Importance

**Location**: `BenchMARL/summarize_and_plot_rules.py` lines 674-995  
**Single-Agent**: `single_agent/summarize_and_plot_rules_single.py` lines 476-844

### Formula

**Raw Importance Score**:
```
raw_importance = frequency / (average_interval_width + ε)
```

**Where**:
- `frequency`: Number of times feature appears in rules
- `average_interval_width`: Mean width of intervals for this feature across all rules
- `ε`: Small constant (1e-6) to prevent division by zero

**Normalized Importance (Percentage)**:
```
importance_percentage = (raw_importance / total_raw_importance) × 100%
```

**Interpretation**:
- **Higher frequency**: Feature appears in more rules (more commonly used)
- **Narrower intervals**: More selective/precise feature usage
- **Raw importance**: Combines both - features that are frequently used AND selective score higher
- **Percentage**: Normalized to sum to 100% for easy interpretation

---

## Computation Process

### Step 1: Extract Feature Intervals from Rules

**Rule Format**:
```
"feature_name ∈ [lower, upper] and feature_name2 ∈ [lower2, upper2]"
```

**Extraction**:
- Parse rule string using regex pattern
- Extract: `(feature_name, lower, upper)` tuples
- Handle duplicate features (keep first occurrence)

**Code**: `extract_feature_intervals_from_rule()` in `plot_comparison.py`

### Step 2: Collect Intervals Per Feature

**Per Class**:
```python
feature_intervals_per_class[class][feature] = [
    (lower1, upper1),
    (lower2, upper2),
    ...
]
```

**Global**:
```python
feature_intervals_global[feature] = [
    (lower1, upper1),
    (lower2, upper2),
    ...
]
```

**Purpose**: Aggregate intervals across all rules

### Step 3: Compute Frequency

**Per Feature**:
```python
frequency = len(feature_intervals[feature])
```

**Interpretation**: How many rules mention this feature

### Step 4: Compute Average Interval Width

**Per Feature**:
```python
widths = [upper - lower for (lower, upper) in feature_intervals[feature]]
average_width = mean(widths)
```

**Interpretation**: Average selectivity of this feature across rules

### Step 5: Compute Raw Importance

```python
raw_importance = frequency / (average_width + 1e-6)
```

**Interpretation**: 
- High frequency + narrow width → High importance
- Low frequency or wide width → Low importance

### Step 6: Normalize to Percentages

```python
total_importance = sum(all_raw_importances)
importance_percentage = (raw_importance / total_importance) × 100
```

**Purpose**: Makes importance scores comparable and interpretable

---

## Feature Importance Visualization

### Plot Components

1. **Frequency Bar Chart**:
   - X-axis: Features (sorted by importance)
   - Y-axis: Number of rules containing feature
   - Shows: How often feature appears

2. **Average Width Bar Chart**:
   - X-axis: Features
   - Y-axis: Average interval width
   - Shows: How selective feature usage is

3. **Raw Importance Bar Chart**:
   - X-axis: Features
   - Y-axis: Raw importance score
   - Shows: Combined frequency and selectivity

4. **Percentage Importance Bar Chart**:
   - X-axis: Features
   - Y-axis: Percentage of total importance
   - Shows: Relative importance (sums to 100%)

### Example Plot

```
Feature Importance (Top 10)
┌─────────────────────────────────────┐
│ Feature A: ████████████ 25.0%      │
│ Feature B: ████████ 12.5%           │
│ Feature C: ██████ 10.0%             │
│ ...                                 │
└─────────────────────────────────────┘
```

---

## Comparison with Baseline Methods

### Baseline: Permutation Importance

**Method**: Permute feature values and measure accuracy drop

**Formula**:
```
importance = baseline_accuracy - accuracy_with_permuted_feature
```

**Location**: `baseline/establish_baseline.py` lines 1570-1654

**Key Differences**:
- **Baseline**: Global method (one score per feature)
- **Dynamic Anchors**: Rule-based (from extracted rules)
- **Baseline**: Measures feature impact on classifier
- **Dynamic Anchors**: Measures feature usage in explanations

### Baseline: SHAP Values

**Method**: Shapley values from game theory

**Location**: `baseline/establish_baseline.py` lines 1329-1567

**Key Differences**:
- **SHAP**: Instance-level or global feature contributions
- **Dynamic Anchors**: Rule-based feature importance
- **SHAP**: Explains model predictions
- **Dynamic Anchors**: Explains extracted rules

---

## Per-Class vs Global Importance

### Per-Class Importance

**Computation**: Feature importance computed separately for each class

**Purpose**: Identifies features important for specific classes

**Example**:
- Class 0: `age` is most important (25%)
- Class 1: `income` is most important (30%)

### Global Importance

**Computation**: Feature importance aggregated across all classes

**Purpose**: Identifies features important across all classes

**Example**:
- Global: `age` is most important (20%), `income` second (15%)

---

## Feature Importance Metrics

### Metrics Computed

1. **Frequency**: Number of rules containing feature
2. **Average Width**: Mean interval width for feature
3. **Raw Importance**: `frequency / (average_width + ε)`
4. **Percentage**: Normalized importance (0-100%)

### Additional Statistics

- **Min/Max Width**: Range of interval widths
- **Width Std**: Variability in interval widths
- **Rule Coverage**: Fraction of rules containing feature

---

## Interpretation Guidelines

### High Importance Features

**Characteristics**:
- High frequency (appears in many rules)
- Narrow intervals (selective usage)
- Consistent across episodes

**Interpretation**: Feature is critical for explanations

### Low Importance Features

**Characteristics**:
- Low frequency (rarely appears)
- Wide intervals (non-selective)
- Inconsistent usage

**Interpretation**: Feature is less critical or redundant

### Feature Selection Insights

**Narrow Intervals**:
- Feature is used precisely
- Indicates feature is discriminative
- Good for interpretability

**Wide Intervals**:
- Feature is used broadly
- May indicate less discriminative feature
- Or feature is important but with wide range

---

## Code References

- **Multi-Agent Plotting**: `BenchMARL/summarize_and_plot_rules.py` lines 674-995
- **Single-Agent Plotting**: `single_agent/summarize_and_plot_rules_single.py` lines 476-844
- **Rule Parsing**: `plot_comparison.py` `extract_feature_intervals_from_rule()`
- **Baseline Methods**: `baseline/establish_baseline.py`

---

## Example Output

### Top 10 Features (Example)

```
Feature Importance (Top 10)
─────────────────────────────────────────────
1. age:           25.0%  (freq=50, width=0.05)
2. income:        20.0%  (freq=40, width=0.08)
3. education:     15.0%  (freq=30, width=0.10)
4. marital_status: 12.0%  (freq=24, width=0.12)
5. workclass:     10.0%  (freq=20, width=0.15)
...
```

**Interpretation**:
- `age` is most important (25% of total importance)
- Appears in 50 rules with average width 0.05 (very selective)
- `income` is second most important (20%)
- Appears in 40 rules with average width 0.08 (selective)

