# Explaining Results with Exploratory Data Analysis

This guide explains how to interpret your Dynamic Anchors results in the context of exploratory data analysis (EDA). Understanding the underlying data characteristics helps explain why certain methods perform better or worse, and provides insights into the explainability results.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Data Characteristics to Analyze](#key-data-characteristics-to-analyze)
3. [Connecting EDA to Explainability Metrics](#connecting-eda-to-explainability-metrics)
4. [Interpreting Results Based on Data Properties](#interpreting-results-based-on-data-properties)
5. [Step-by-Step Analysis Workflow](#step-by-step-analysis-workflow)
6. [Example Interpretations](#example-interpretations)
7. [Common Patterns and Explanations](#common-patterns-and-explanations)

---

## Overview

Exploratory Data Analysis (EDA) provides crucial context for understanding explainability results. By analyzing data characteristics, you can:

- **Explain performance differences** between methods (baseline, single-agent, multi-agent)
- **Understand precision/coverage trade-offs** based on class separability
- **Interpret feature importance** in the context of feature distributions
- **Explain rule complexity** based on data dimensionality and correlations
- **Understand convergence behavior** based on class balance and overlap

---

## Key Data Characteristics to Analyze

### 1. **Class Distribution**

**What to measure:**
- Class balance (balanced vs. imbalanced)
- Number of samples per class
- Train/test distribution consistency

**How it affects results:**
- **Balanced classes**: All methods should perform similarly; easier to achieve high coverage
- **Imbalanced classes**: Minority classes may have lower coverage; multi-agent may help by coordinating coverage
- **Small class sizes**: Lower coverage expected; precision may be higher due to tighter regions

**Code location**: `BenchMARL/tabular_datasets.py` → `_analyze_class_distribution()`

### 2. **Feature Statistics**

**What to measure:**
- Mean, std, min, max per feature
- Feature scales and ranges
- Outliers

**How it affects results:**
- **Large feature ranges**: Anchors may be wider; harder to achieve high precision
- **Normalized features**: Easier to interpret anchor bounds; consistent scales across features
- **Outliers**: May cause anchors to expand unnecessarily; can affect precision

**Code location**: `BenchMARL/tabular_datasets.py` → `_analyze_feature_statistics()`

### 3. **Feature Correlations**

**What to measure:**
- Pairwise correlations between features
- Highly correlated feature groups

**How it affects results:**
- **High correlations**: Rules may include redundant features; can simplify explanations
- **Low correlations**: Each feature contributes independently; rules may be more complex
- **Correlated groups**: Multi-agent may find complementary features within groups

**Code location**: `BenchMARL/tabular_datasets.py` → `_analyze_feature_correlations()`

### 4. **Class Separability**

**What to measure:**
- Distance between class means
- Overlap between class distributions
- Feature-level separability scores

**How it affects results:**
- **High separability**: Easier to achieve high precision; clear decision boundaries
- **Low separability**: Harder to achieve high precision; may need wider anchors (lower precision)
- **Feature-level separability**: Explains which features appear in rules

**Code location**: `BenchMARL/tabular_datasets.py` → `_analyze_class_separability()`

### 5. **Dimensionality**

**What to measure:**
- Number of features
- Effective dimensionality (after accounting for correlations)

**How it affects results:**
- **Low dimensionality**: Simpler rules; easier to achieve high coverage
- **High dimensionality**: More complex rules; sparser coverage; curse of dimensionality
- **Effective dimensionality**: More relevant than raw feature count

---

## Connecting EDA to Explainability Metrics

### Precision Metrics

**What affects precision:**
1. **Class separability**: Higher separability → higher precision
2. **Feature correlations**: Correlated features may reduce precision (wider anchors needed)
3. **Class balance**: Imbalanced classes may have lower precision for minority classes
4. **Outliers**: Outliers can force anchors to expand, reducing precision

**Interpretation:**
- **High precision (>0.9)**: Classes are well-separated; clear decision boundaries
- **Low precision (<0.7)**: Classes overlap significantly; may need to trade precision for coverage
- **Precision differences between classes**: Check class separability metrics

### Coverage Metrics

**What affects coverage:**
1. **Class size**: Larger classes → easier to achieve high coverage
2. **Dimensionality**: Higher dimensionality → harder to achieve high coverage
3. **Class distribution**: Balanced classes → more uniform coverage
4. **Feature ranges**: Wider ranges → easier to cover more samples

**Interpretation:**
- **High coverage (>0.8)**: Data is concentrated in feature space; good class representation
- **Low coverage (<0.3)**: Data is sparse; high-dimensional or complex distributions
- **Coverage differences**: Check class sizes and separability

### Rule Complexity

**What affects rule complexity:**
1. **Feature correlations**: Correlated features → simpler rules (fewer unique features)
2. **Dimensionality**: Higher dimensionality → more features in rules
3. **Class separability**: Well-separated classes → simpler rules (fewer features needed)

**Interpretation:**
- **Simple rules (few features)**: High feature correlations or clear separability
- **Complex rules (many features)**: Low correlations, high dimensionality, or overlapping classes

### NashConv Convergence

**What affects convergence:**
1. **Class balance**: Balanced classes → faster convergence
2. **Class overlap**: Overlapping classes → slower convergence (harder to coordinate)
3. **Number of agents**: More agents → potentially slower convergence

**Interpretation:**
- **Fast convergence**: Well-separated, balanced classes
- **Slow convergence**: Overlapping classes, imbalanced distribution, or many agents

---

## Interpreting Results Based on Data Properties

### Scenario 1: High Precision, Low Coverage

**Data characteristics:**
- High class separability
- Concentrated class distributions
- Low dimensionality

**Explanation:**
- Classes are well-separated, so anchors can be precise
- But data is concentrated in small regions, so coverage is limited
- **Action**: This is expected; consider if coverage is sufficient for your use case

### Scenario 2: Low Precision, High Coverage

**Data characteristics:**
- Low class separability
- Overlapping class distributions
- Wide feature ranges

**Explanation:**
- Classes overlap significantly, requiring wider anchors
- Wide anchors cover more samples but include samples from other classes
- **Action**: Check if precision is acceptable; may need feature engineering

### Scenario 3: Multi-Agent Outperforms Single-Agent

**Data characteristics:**
- High feature correlations
- Multiple distinct regions per class
- Imbalanced classes

**Explanation:**
- Multiple agents can cover different regions/features
- Coordination helps balance coverage across classes
- **Action**: Multi-agent is beneficial for complex, multi-modal distributions

### Scenario 4: Single-Agent Outperforms Multi-Agent

**Data characteristics:**
- Low dimensionality
- Well-separated classes
- Balanced distribution

**Explanation:**
- Simple problem doesn't need multi-agent coordination
- Single agent can efficiently cover the space
- **Action**: Use single-agent for simpler problems

### Scenario 5: High Rule Overlap Between Classes

**Data characteristics:**
- High feature correlations
- Overlapping class distributions
- Low separability

**Explanation:**
- Classes share similar feature ranges
- Rules naturally overlap in feature space
- **Action**: Expected behavior; consider if overlap is problematic

---

## Step-by-Step Analysis Workflow

### Step 1: Load and Analyze Dataset

```python
from BenchMARL.tabular_datasets import TabularDatasetLoader

loader = TabularDatasetLoader(
    dataset_name="breast_cancer",
    test_size=0.2,
    random_state=42
)

X_train, X_test, y_train, y_test, feature_names, class_names = loader.load_dataset()

# Perform EDA
eda_results = loader.perform_eda(
    output_dir="./eda_output/",
    verbose=True,
    use_ydata_profiling=False
)
```

### Step 2: Extract Key EDA Metrics

```python
# Class distribution
class_dist = eda_results["class_distribution"]
is_balanced = class_dist["is_balanced"]
train_dist = class_dist["train"]

# Feature statistics
feature_stats = eda_results["feature_statistics"]

# Correlations
correlations = eda_results["feature_correlations"]
high_corr = correlations["high_correlations"]

# Separability
separability = eda_results["class_separability"]
top_features = separability["top_features"]
```

### Step 3: Load Explainability Results

```python
import json

# Load results
with open("comparison_results/breast_cancer_maddpg_20251222_201640/comparison_summary.json") as f:
    results = json.load(f)

multi_agent_summary = results.get("multi_agent", {})
single_agent_summary = results.get("single_agent", {})
baseline_summary = results.get("baseline", {})
```

### Step 4: Connect EDA to Results

```python
# For each class, analyze:
for class_key, class_results in multi_agent_summary.get("per_class_summary", {}).items():
    precision = class_results["instance_precision"]
    coverage = class_results["instance_coverage"]
    
    # Check class size
    class_size = class_results.get("n_class_samples", 0)
    class_pct = (class_size / len(y_test)) * 100
    
    # Check separability
    class_separability = separability["class_overlap_metrics"].get(f"Class_{class_key}_vs_...")
    
    # Interpret
    print(f"\nClass {class_key}:")
    print(f"  Precision: {precision:.3f}, Coverage: {coverage:.3f}")
    print(f"  Class size: {class_size} ({class_pct:.1f}%)")
    print(f"  Separability: {class_separability}")
    
    # Explain based on EDA
    if precision > 0.9 and coverage < 0.3:
        print("  → High precision, low coverage: Well-separated but concentrated distribution")
    elif precision < 0.7 and coverage > 0.7:
        print("  → Low precision, high coverage: Overlapping classes, wide anchors needed")
```

### Step 5: Compare Methods with EDA Context

```python
# Compare methods
methods = {
    "baseline": baseline_summary,
    "single_agent": single_agent_summary,
    "multi_agent": multi_agent_summary
}

for method_name, method_results in methods.items():
    print(f"\n{method_name.upper()}:")
    
    # Check if method benefits from data characteristics
    if method_name == "multi_agent":
        if len(high_corr) > 5:
            print("  → Multi-agent benefits from feature correlations (can coordinate)")
        if not is_balanced:
            print("  → Multi-agent helps with imbalanced classes (coordination)")
    
    # Check precision/coverage trade-off
    avg_precision = np.mean([c["instance_precision"] for c in method_results.get("per_class_summary", {}).values()])
    avg_coverage = np.mean([c["instance_coverage"] for c in method_results.get("per_class_summary", {}).values()])
    
    print(f"  Avg Precision: {avg_precision:.3f}, Avg Coverage: {avg_coverage:.3f}")
    
    # Explain based on separability
    if avg_precision > 0.9:
        print("  → High precision: Classes are well-separated")
    elif avg_precision < 0.7:
        print("  → Low precision: Classes overlap significantly")
```

---

## Example Interpretations

### Example 1: Breast Cancer Dataset

**EDA Findings:**
- 2 classes (malignant, benign)
- Balanced distribution (~63% benign, ~37% malignant)
- 30 features (high dimensionality)
- Some highly correlated features (radius, perimeter, area)
- Good class separability

**Results:**
- Multi-agent: Precision=0.96, Coverage=0.01 (instance-level)
- Multi-agent: Class precision=1.0, Class coverage=0.98 (union)

**Interpretation:**
- **High precision**: Classes are well-separated (confirmed by EDA)
- **Low instance coverage**: High dimensionality makes individual anchors narrow
- **High class coverage**: Union of many anchors covers most of the class
- **Feature correlations**: Rules may include correlated features (radius, perimeter, area together)

**Why multi-agent works well:**
- Multiple agents can explore different feature combinations
- Coordination helps achieve high class-level coverage
- Well-separated classes allow high precision

### Example 2: Circles Dataset (Synthetic)

**EDA Findings:**
- 2 classes (inner circle, outer circle)
- Balanced distribution
- 2 features (low dimensionality)
- Non-linear decision boundary
- High separability (clear boundary)

**Results:**
- Single-agent: Precision=0.95, Coverage=0.85
- Multi-agent: Precision=0.96, Coverage=0.88

**Interpretation:**
- **High precision**: Clear separation (confirmed by EDA)
- **High coverage**: Low dimensionality makes coverage easier
- **Similar performance**: Simple problem doesn't need multi-agent coordination

**Why single-agent is sufficient:**
- Low dimensionality → easy to cover
- Clear boundary → simple rules suffice
- Multi-agent adds little value

### Example 3: Imbalanced Dataset

**EDA Findings:**
- 2 classes
- Imbalanced (90% class 0, 10% class 1)
- Moderate separability
- High dimensionality

**Results:**
- Single-agent: Class 0 precision=0.92, coverage=0.85; Class 1 precision=0.88, coverage=0.15
- Multi-agent: Class 0 precision=0.93, coverage=0.90; Class 1 precision=0.90, coverage=0.45

**Interpretation:**
- **Class 0 (majority)**: Both methods perform well (large class, easier)
- **Class 1 (minority)**: Multi-agent significantly improves coverage
- **Imbalance effect**: Single-agent struggles with minority class

**Why multi-agent helps:**
- Coordination ensures minority class gets adequate coverage
- Multiple agents can focus on different regions of minority class
- Shared rewards encourage balanced coverage

---

## Common Patterns and Explanations

### Pattern 1: Precision-Coverage Trade-off

**Observation**: High precision often comes with low coverage, and vice versa.

**EDA Explanation**:
- **High precision, low coverage**: Well-separated classes → narrow, precise anchors → limited coverage
- **Low precision, high coverage**: Overlapping classes → wide anchors needed → more coverage but lower precision

**Action**: Check class separability metrics. If separability is low, consider feature engineering.

### Pattern 2: Multi-Agent Improves Coverage

**Observation**: Multi-agent achieves higher class-level coverage than single-agent.

**EDA Explanation**:
- Multiple agents can cover different regions of feature space
- Coordination helps balance coverage across classes
- Especially beneficial for imbalanced or multi-modal distributions

**Action**: Check class balance and distribution shape. Multi-agent is most beneficial for complex distributions.

### Pattern 3: Rule Overlap Between Classes

**Observation**: Rules from different classes overlap in feature space.

**EDA Explanation**:
- High feature correlations → similar feature ranges across classes
- Overlapping class distributions → natural overlap
- Low separability → boundaries are fuzzy

**Action**: Check feature correlations and class separability. Overlap is expected for correlated features.

### Pattern 4: Feature Importance Mismatch

**Observation**: Features in rules don't match feature importance scores.

**EDA Explanation**:
- Feature importance measures global importance
- Rules focus on local regions where features matter
- Correlated features may appear together in rules

**Action**: Check feature correlations. Rules may use correlated features as a group.

### Pattern 5: Slow NashConv Convergence

**Observation**: NashConv decreases slowly or plateaus.

**EDA Explanation**:
- Overlapping classes → harder to coordinate
- Imbalanced classes → minority class agents struggle
- Many agents → more coordination needed

**Action**: Check class separability and balance. Consider reducing number of agents or adjusting rewards.

---

## Summary

**Key Takeaways:**

1. **Always perform EDA first** to understand data characteristics
2. **Connect EDA metrics to explainability results** to explain performance
3. **Use EDA to guide method selection** (single-agent vs. multi-agent)
4. **Interpret precision/coverage trade-offs** based on separability
5. **Explain rule complexity** using feature correlations and dimensionality
6. **Understand convergence behavior** through class balance and overlap

**Next Steps:**

1. Run EDA on your dataset using `TabularDatasetLoader.perform_eda()`
2. Load your explainability results
3. Connect EDA findings to results using the patterns above
4. Document your interpretations for your paper/report

---

## References

- **EDA Code**: `BenchMARL/tabular_datasets.py`
- **Metrics Documentation**: `docs/PRECISION_COVERAGE_METRICS.md`
- **NashConv Documentation**: `docs/NASHCONV_INTERPRETATION.md`
- **Feature Importance**: `docs/PRECISION_COVERAGE_METRICS.md` (Section 5)

