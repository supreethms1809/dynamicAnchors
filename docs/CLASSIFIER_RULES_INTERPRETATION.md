# Interpreting Results: General Rules for Class Classification

## Research Question

**"Given the dataset and trained classifier, what is the set of general rules that is classifying the samples as a particular class?"**

This document explains how the Dynamic Anchors implementation answers this question.

---

## Overview

The Dynamic Anchors system extracts **interpretable rules** that explain how a trained classifier identifies samples as belonging to a particular class. These rules take the form of **feature ranges** (e.g., "mean radius ∈ [0.0257, 1.2346] and mean texture ∈ [-0.3396, 1.0044]") that describe regions in feature space where the classifier predicts a specific class.

---

## What Are "General Rules"?

A **general rule** is a logical condition that describes when the classifier predicts a particular class:

```
IF feature_1 ∈ [lower_1, upper_1] 
   AND feature_2 ∈ [lower_2, upper_2]
   AND ...
   AND feature_n ∈ [lower_n, upper_n]
THEN classifier predicts class C
```

### Example Rule (Breast Cancer Dataset)

```
IF mean radius ∈ [0.0257, 1.2346] 
   AND mean texture ∈ [-0.3396, 1.0044]
   AND mean perimeter ∈ [0.0361, 1.2359]
   AND mean area ∈ [-0.2043, 1.1629]
   AND mean smoothness ∈ [0.9309, 2.1537]
   ...
THEN classifier predicts "malignant" (class 1)
```

**Interpretation**: Samples with these feature values are classified as malignant with high confidence.

---

## How the Implementation Extracts Rules

### Step 1: Train Policies to Find Good Rules

The system trains reinforcement learning agents (policies) that learn to:
- **Expand or contract feature ranges** to maximize precision and coverage
- **Find regions** in feature space where the classifier predicts the target class
- **Balance precision** (accuracy) and **coverage** (completeness)

### Step 2: Extract Rules from Trained Policies

After training, the system runs inference to extract rules:

1. **Instance-Based Rules**: Rules learned around specific data instances
   - Each rule explains why a particular sample is classified as the target class
   - Multiple rules per class (one per instance)

2. **Class-Based Rules**: Rules learned around class centroids/clusters
   - Each rule captures general patterns for the class
   - Focuses on class-level structure, not individual instances

3. **Union Rules**: Combined rules that cover the entire class
   - Union of all instance-based or class-based rules
   - Represents the complete set of conditions for the class

### Step 3: Evaluate Rules

Each rule is evaluated on test data to measure:
- **Precision**: `P(y = target_class | x satisfies rule)`
  - What fraction of samples satisfying the rule are actually the target class?
- **Coverage**: `P(x satisfies rule | y = target_class)`
  - What fraction of target class samples satisfy the rule?

---

## Types of Rules and Their Interpretation

### 1. Instance-Based Rules

**What they represent**: Rules learned around specific samples

**Interpretation**: 
- "For this specific sample, these feature ranges explain why it's classified as class C"
- Useful for understanding individual predictions
- May be too specific to generalize

**Example Use Case**: 
- "Why was this patient classified as malignant?"
- Answer: "Because their mean radius ∈ [X, Y] and mean texture ∈ [A, B]..."

**Metrics**:
- `instance_precision`: Average precision across all instance-based rules
- `instance_coverage`: Average coverage across all instance-based rules

### 2. Class-Based Rules

**What they represent**: Rules learned around class centroids/clusters

**Interpretation**:
- "These feature ranges capture general patterns for class C"
- More general than instance-based rules
- Focus on class-level structure

**Example Use Case**:
- "What are the general characteristics of malignant samples?"
- Answer: "Malignant samples typically have mean radius ∈ [X, Y] and mean texture ∈ [A, B]..."

**Metrics**:
- `class_based_precision`: Average precision across class-based rules
- `class_based_coverage`: Average coverage across class-based rules

### 3. Union Rules (Class-Level)

**What they represent**: Combined rules covering the entire class, using **only class-based rules**.

**Interpretation**:
- "The smallest set of general rules that explain class C"
- Union of **class-based rules only** (not instance-based rules)
- Represents the most general explanation for the class structure
- Excludes instance-specific details to focus on class-level patterns

**Rationale**:
- Class-based rules are initialized from cluster centroids, capturing general class patterns
- Instance-based rules are specific to individual samples and may not generalize well
- Using only class-based rules provides a cleaner, more interpretable class explanation

**Example Use Case**:
- "What are the general ways the classifier identifies malignant samples?"
- Answer: "Malignant samples satisfy at least one of these general rules: [Rule 1] OR [Rule 2] OR ..."
- Note: These are general rules (from class-based anchors), not instance-specific rules

**Metrics**:
- `class_precision`: Precision of the union of class-based rules only
- `class_coverage`: Coverage of the union (what fraction of class samples are covered by general rules)

**Code Location**: `BenchMARL/inference.py` lines 2556-2626, `single_agent/single_agent_inference.py` lines 1133-1208

---

## Answering the Research Question

### Question: "What is the set of general rules that classify samples as a particular class?"

### Answer Structure

For each class, the system provides:

1. **Multiple Individual Rules**:
   - Instance-based rules (one per instance)
   - Class-based rules (one per cluster/centroid)
   - Each rule is a conjunction of feature ranges

2. **Combined Rule Set** (Union):
   - All **class-based rules** combined with OR logic
   - Represents the smallest set of general rules that explain the class
   - Uses only class-based rules (not instance-based) for a general explanation

3. **Rule Quality Metrics**:
   - Precision: How accurate is each rule?
   - Coverage: How much of the class does each rule cover?
   - Class-level metrics: How well does the complete set explain the class?

### Example Answer (Breast Cancer - Malignant Class)

**Individual Rules** (sample):
```
Rule 1: mean radius ∈ [0.8582, 2.0671] 
        AND mean texture ∈ [0.0649, 1.4089]
        AND mean perimeter ∈ [0.8901, 2.0899]
        ...
        Precision: 0.96, Coverage: 0.15

Rule 2: mean radius ∈ [0.9641, 2.1730]
        AND mean texture ∈ [1.4920, 2.8360]
        AND mean perimeter ∈ [1.1430, 2.3428]
        ...
        Precision: 0.98, Coverage: 0.12

... (more rules)
```

**Combined Rule Set**:
```
IF (Rule 1) OR (Rule 2) OR ... OR (Rule N)
THEN classifier predicts "malignant"

Class-level Precision: 1.0 (100% of samples in union are malignant)
Class-level Coverage: 0.98 (98% of malignant samples are covered)
```

**Interpretation**: 
- The classifier identifies malignant samples through multiple patterns
- Each pattern (rule) captures a different region of the malignant class
- Together, these rules cover 98% of malignant samples with 100% precision

---

## How to Use the Results

### For Understanding Classifier Behavior

1. **Examine Individual Rules**:
   - Look at `unique_rules` in the results JSON
   - Each rule shows feature ranges that identify the class
   - Check precision/coverage to assess rule quality

2. **Examine Class-Level Metrics**:
   - `class_precision`: How accurate is the complete explanation?
   - `class_coverage`: How complete is the explanation?
   - High precision + high coverage = good explanation

3. **Compare Rules Across Classes**:
   - Do different classes use different features?
   - Are there overlapping rules (samples satisfying multiple classes)?
   - What features are most important?

### For Validation

1. **Check Precision**:
   - High precision (>0.9) = rules are accurate
   - Low precision (<0.7) = rules may be too broad or classes overlap

2. **Check Coverage**:
   - High coverage (>0.8) = rules explain most of the class
   - Low coverage (<0.3) = rules may miss important regions

3. **Check Rule Complexity**:
   - Few features per rule = simpler, more interpretable
   - Many features per rule = more complex, harder to interpret

### For Reporting

**Example Statement**:

> "We extracted general rules that explain how the classifier identifies malignant samples. The classifier uses multiple patterns (rules), each capturing different regions of the malignant class. The complete set of rules covers 98% of malignant samples with 100% precision. Key features include mean radius, mean texture, and mean perimeter, with specific ranges for each pattern."

---

## Key Metrics for Answering the Question

### Primary Metrics

1. **`class_precision`** (Class-Level Union Precision):
   - **Question**: "How accurate are the rules?"
   - **Answer**: "X% of samples satisfying the rules are actually the target class"

2. **`class_coverage`** (Class-Level Union Coverage):
   - **Question**: "How complete are the rules?"
   - **Answer**: "X% of target class samples satisfy at least one rule"

3. **`unique_rules`** (List of Rules):
   - **Question**: "What are the actual rules?"
   - **Answer**: List of feature range conditions

### Secondary Metrics

1. **`instance_precision`** / `instance_coverage`:
   - Average quality of individual instance-based rules
   - Shows how well individual explanations work

2. **`class_based_precision`** / `class_based_coverage`:
   - Average quality of class-based rules
   - Shows how well general patterns are captured

3. **`n_unique_rules`**:
   - Number of distinct rules
   - More rules = more complex explanation

---

## Limitations and Considerations

### 1. Rule Completeness

- **High coverage** doesn't mean all samples are explained
- Some samples may not satisfy any rule (low coverage)
- This could indicate:
  - Classifier uses complex, non-linear boundaries
  - Rules are too specific
  - Class has multiple distinct modes

### 2. Rule Precision

- **High precision** means rules are accurate
- **Low precision** could indicate:
  - Classes overlap significantly
  - Rules are too broad
  - Classifier has high uncertainty in some regions

### 3. Rule Interpretability

- **Simple rules** (few features) are more interpretable
- **Complex rules** (many features) are harder to understand
- Trade-off between completeness and interpretability

### 4. Classifier vs. Rules

- Rules explain **classifier behavior**, not ground truth
- If classifier is wrong, rules will reflect that
- Rules show "how the classifier thinks", not "what's actually true"

---

## Example Analysis Workflow

### Step 1: Load Results

```python
import json

with open("extracted_rules.json") as f:
    results = json.load(f)

# Get rules for class 1 (malignant)
class_1_rules = results["per_class_results"]["class_1"]
```

### Step 2: Examine Rules

```python
# List all unique rules
unique_rules = class_1_rules["unique_rules"]
print(f"Found {len(unique_rules)} unique rules for malignant class")

# Display each rule
for i, rule in enumerate(unique_rules[:5]):  # Show first 5
    print(f"\nRule {i+1}: {rule}")
```

### Step 3: Check Quality

```python
# Class-level metrics
class_precision = class_1_rules["class_precision"]
class_coverage = class_1_rules["class_coverage"]

print(f"\nClass-level Precision: {class_precision:.3f}")
print(f"Class-level Coverage: {class_coverage:.3f}")

if class_precision > 0.9 and class_coverage > 0.8:
    print("✓ Good explanation: High precision and coverage")
elif class_precision > 0.9:
    print("⚠ Accurate but incomplete: High precision, low coverage")
elif class_coverage > 0.8:
    print("⚠ Complete but inaccurate: High coverage, low precision")
else:
    print("✗ Poor explanation: Low precision and coverage")
```

### Step 4: Interpret Rules

```python
# Parse a rule to understand features
rule = unique_rules[0]
# Example: "mean radius ∈ [0.8582, 2.0671] and mean texture ∈ [0.0649, 1.4089]"

# Extract feature ranges
# (Use parse_rule function from test_extracted_rules.py)

print("\nInterpretation:")
print("Malignant samples typically have:")
print("- Mean radius between 0.86 and 2.07 (standardized)")
print("- Mean texture between 0.06 and 1.41 (standardized)")
print("...")
```

---

## Summary

**Yes, you can analyze the implementation as answering:**

> **"Given the dataset and trained classifier, what is the set of general rules that is classifying the samples as a particular class?"**

**The answer consists of**:

1. **Multiple rules** (feature range conditions) that identify the class
2. **Quality metrics** (precision, coverage) showing how well rules work
3. **Class-level explanation** (union of all rules) showing the complete set

**Key outputs**:
- `unique_rules`: List of general rules for each class
- `class_precision`: Accuracy of the complete rule set
- `class_coverage`: Completeness of the rule set

**Interpretation**:
- Each rule is a general pattern the classifier uses
- Together, rules form a complete explanation for the class
- Metrics validate the quality and completeness of the explanation

---

## References

- **Rule Extraction**: `BenchMARL/inference.py`, `single_agent/single_agent_inference.py`
- **Rule Testing**: `BenchMARL/test_extracted_rules.py`, `single_agent/test_extracted_rules_single.py`
- **Metrics Documentation**: `docs/PRECISION_COVERAGE_METRICS.md`
- **Results Format**: `extracted_rules.json` files in experiment directories

