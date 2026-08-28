# Rule Overlap Metrics

## Overview

Rule overlap metrics measure the similarity between different anchor rules. This helps assess diversity and redundancy in the extracted explanations.

---

## Rule Overlap Computation

### Method: Interval-Based Overlap

**Location**: 
- `BenchMARL/summarize_and_plot_rules.py` lines 999-1038
- `single_agent/summarize_and_plot_rules_single.py` lines 850-889
- `plot_comparison.py` lines 1223-1303

### Formula

**For Two Rules**:
```
overlap_score = min(feature_overlaps)
```

**Where**:
- `feature_overlaps`: List of overlap scores for each common feature
- Each feature overlap: `intersection / union` (Jaccard-like)

**Feature Overlap** (for one feature):
```
intersection = max(0, min(upper1, upper2) - max(lower1, lower2))
union = max(upper1, upper2) - min(lower1, lower2)
feature_overlap = intersection / union  (if union > 0)
```

**Key Insight**: Uses **minimum** across features because rules are AND conditions (all features must overlap for rules to overlap).

---

## Computation Process

### Step 1: Parse Rules into Intervals

**Input Rule**:
```
"age ∈ [25.0, 35.0] and income ∈ [50000.0, 75000.0]"
```

**Extracted Intervals**:
```python
[
    ("age", 25.0, 35.0),
    ("income", 50000.0, 75000.0)
]
```

**Code**: `extract_feature_intervals_from_rule()` function

### Step 2: Find Common Features

**Rule 1**: `[("age", 25, 35), ("income", 50k, 75k)]`  
**Rule 2**: `[("age", 30, 40), ("income", 60k, 80k)]`

**Common Features**: `{"age", "income"}`

**If No Common Features**: `overlap = 0.0`

### Step 3: Compute Feature Overlaps

**For Each Common Feature**:

1. **Get Intervals**:
   - Rule 1: `(lower1, upper1) = (25, 35)`
   - Rule 2: `(lower2, upper2) = (30, 40)`

2. **Compute Intersection**:
   ```
   intersect_lower = max(lower1, lower2) = max(25, 30) = 30
   intersect_upper = min(upper1, upper2) = min(35, 40) = 35
   intersection = intersect_upper - intersect_lower = 35 - 30 = 5
   ```

3. **Compute Union**:
   ```
   union_lower = min(lower1, lower2) = min(25, 30) = 25
   union_upper = max(upper1, upper2) = max(35, 40) = 40
   union = union_upper - union_lower = 40 - 25 = 15
   ```

4. **Compute Overlap**:
   ```
   feature_overlap = intersection / union = 5 / 15 = 0.333
   ```

### Step 4: Take Minimum Across Features

**Why Minimum?**
- Rules are AND conditions: `feature1 ∈ [a,b] AND feature2 ∈ [c,d]`
- For rules to overlap, ALL features must overlap
- Minimum ensures all features overlap

**Example**:
- Feature "age": overlap = 0.8
- Feature "income": overlap = 0.3
- **Rule overlap = min(0.8, 0.3) = 0.3**

### Step 5: Handle Edge Cases

**Zero-Width Intervals**:
- If both intervals are points (zero width)
- If points are same: `overlap = 1.0`
- If points differ: `overlap = 0.0`

**No Intersection**:
- If `intersect_lower > intersect_upper`: `overlap = 0.0`

---

## Rule Overlap Matrix

### Computation

**For N rules**, compute N×N overlap matrix:

```python
overlap_matrix[i, j] = calculate_rule_overlap(rule_i, rule_j)
```

**Properties**:
- **Symmetric**: `overlap_matrix[i, j] = overlap_matrix[j, i]`
- **Diagonal**: `overlap_matrix[i, i] = 1.0` (rule overlaps with itself)

### Visualization

**Heatmap**:
- Rows/Columns: Rules
- Colors: Overlap scores (0.0 = no overlap, 1.0 = perfect overlap)
- Red = high overlap, Blue = low overlap

**Example**:
```
      Rule1  Rule2  Rule3
Rule1  1.00   0.30   0.00
Rule2  0.30   1.00   0.50
Rule3  0.00   0.50   1.00
```

---

## Overlap Statistics

### Per-Class Statistics

**Computed**:
- **Mean Overlap**: Average overlap across all rule pairs
- **Max Overlap**: Maximum overlap (most similar pair)
- **Min Overlap**: Minimum overlap (most different pair)
- **Std Overlap**: Standard deviation of overlaps

**Formula**:
```python
all_overlaps = [overlap_matrix[i, j] for i < j]  # Upper triangle
mean_overlap = mean(all_overlaps)
max_overlap = max(all_overlaps)
min_overlap = min(all_overlaps)
std_overlap = std(all_overlaps)
```

### Interpretation

**High Mean Overlap** (>0.7):
- Rules are very similar
- May indicate lack of diversity
- Could suggest overfitting to similar patterns

**Low Mean Overlap** (<0.3):
- Rules are diverse
- Good for coverage (different regions)
- May indicate good exploration

**High Max Overlap** (>0.9):
- At least one pair of rules is nearly identical
- May indicate redundancy

**Low Min Overlap** (<0.1):
- At least one pair is very different
- Good for diversity

---

## Rule Overlap Visualization

### Heatmap Plot

**Components**:
1. **Color Scale**: 0.0 (blue) to 1.0 (red)
2. **Rule Labels**: Rule indices or rule strings (truncated)
3. **Value Annotations**: Overlap scores in cells

**Purpose**: Visual inspection of rule similarity

### Statistics Summary

**Displayed**:
- Mean overlap
- Max overlap
- Min overlap
- Number of rules

---

## Use Cases

### 1. Diversity Assessment

**Question**: Are extracted rules diverse?

**Metric**: Mean overlap
- **Low** (<0.3): Diverse rules ✓
- **High** (>0.7): Similar rules (may need more diversity)

### 2. Redundancy Detection

**Question**: Are there duplicate or near-duplicate rules?

**Metric**: Max overlap
- **High** (>0.9): Redundant rules detected
- **Low** (<0.5): No significant redundancy

### 3. Coverage Analysis

**Question**: Do rules cover different regions?

**Metric**: Min overlap
- **Low** (<0.2): Rules cover different regions ✓
- **High** (>0.5): Rules may overlap significantly

### 4. Quality Control

**Question**: Are rules consistent?

**Metric**: Std overlap
- **Low std**: Consistent overlap patterns
- **High std**: Variable overlap (some similar, some different)

---

## Comparison: Single-Agent vs Multi-Agent

### Single-Agent Overlap

**Characteristics**:
- Rules from one agent per class
- Typically lower overlap (one agent, diverse instances)
- More consistent patterns

### Multi-Agent Overlap

**Characteristics**:
- Rules from multiple agents per class
- Can have higher overlap (agents may converge)
- More variable patterns

**Analysis**:
- **Within-agent overlap**: Overlap of one agent's rules
- **Between-agent overlap**: Overlap across different agents
- **Overall overlap**: All rules combined

---

## Code References

- **Overlap Function**: `plot_comparison.py` lines 1223-1303
- **Multi-Agent Plotting**: `BenchMARL/summarize_and_plot_rules.py` lines 997-1038
- **Single-Agent Plotting**: `single_agent/summarize_and_plot_rules_single.py` lines 850-889
- **Rule Parsing**: `plot_comparison.py` `extract_feature_intervals_from_rule()`

---

## Example Interpretation

### Example Overlap Matrix

```
      R1    R2    R3    R4    R5
R1   1.00  0.85  0.20  0.15  0.10
R2   0.85  1.00  0.25  0.20  0.15
R3   0.20  0.25  1.00  0.30  0.25
R4   0.15  0.20  0.30  1.00  0.40
R5   0.10  0.15  0.25  0.40  1.00
```

**Observations**:
- **R1 and R2**: High overlap (0.85) - very similar, may be redundant
- **R3-R5**: Lower overlap (0.20-0.40) - more diverse
- **Mean**: ~0.35 - moderate diversity
- **Max**: 0.85 - some redundancy detected

**Recommendation**: Consider removing R1 or R2 (redundant), keep R3-R5 (diverse)

---

## Mathematical Formulation

### Complete Formula

For rules `R1` and `R2` with intervals:

```
R1 = {f1: [a1, b1], f2: [a2, b2], ..., fn: [an, bn]}
R2 = {f1: [c1, d1], f2: [c2, d2], ..., fm: [cm, dm]}
```

**Common features**: `F_common = {f | f ∈ R1 and f ∈ R2}`

**For each feature f ∈ F_common**:
```
intersection_f = max(0, min(b_f, d_f) - max(a_f, c_f))
union_f = max(b_f, d_f) - min(a_f, c_f)
overlap_f = intersection_f / union_f  (if union_f > 0)
```

**Rule overlap**:
```
overlap(R1, R2) = min({overlap_f | f ∈ F_common})
```

**If F_common is empty**: `overlap(R1, R2) = 0.0`

---

## Edge Cases Handled

### 1. Invalid Intervals

**Problem**: `lower > upper`

**Solution**: Swap or skip interval

### 2. Duplicate Features

**Problem**: Feature appears multiple times in rule

**Solution**: Keep first occurrence

### 3. Zero-Width Intervals

**Problem**: `lower == upper` (point interval)

**Solution**: 
- If both are points and same: `overlap = 1.0`
- If both are points and different: `overlap = 0.0`

### 4. No Common Features

**Problem**: Rules have no features in common

**Solution**: `overlap = 0.0`

---

## Integration with Other Metrics

### Rule Overlap vs Coverage

**Relationship**:
- **High overlap + High coverage**: Rules are similar but cover well
- **Low overlap + High coverage**: Rules are diverse and cover well ✓
- **High overlap + Low coverage**: Rules are similar and don't cover well
- **Low overlap + Low coverage**: Rules are diverse but don't cover well

### Rule Overlap vs Feature Importance

**Relationship**:
- Features with high importance may appear in overlapping rules
- Overlapping rules may share important features
- Can identify redundant important features

---

## Code References

- **Overlap Computation**: `plot_comparison.py` lines 1223-1303
- **Multi-Agent**: `BenchMARL/summarize_and_plot_rules.py` lines 997-1038
- **Single-Agent**: `single_agent/summarize_and_plot_rules_single.py` lines 850-889
- **Visualization**: `plot_comparison.py` `plot_rule_overlap_subplot()`

