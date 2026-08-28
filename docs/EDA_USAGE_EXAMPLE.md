# EDA Result Analysis - Usage Example

This document provides a quick example of how to use the EDA-informed result analysis tools.

## Quick Start

### Step 1: Run the Analysis Script

```bash
python analyze_results_with_eda.py \
    --dataset breast_cancer \
    --results_dir comparison_results/breast_cancer_maddpg_20251222_201640 \
    --output_dir eda_analysis_output
```

### Step 2: Review the Output

The script generates two files:

1. **`eda_analysis_report.txt`**: Human-readable report with interpretations
2. **`eda_analysis.json`**: Structured data for programmatic access

### Step 3: Use in Your Paper/Report

The report provides ready-to-use explanations for your results, such as:

- "High precision, low coverage: Well-separated but concentrated distribution"
- "Multi-agent achieves higher coverage: Benefits from coordination"
- "Many correlated features: Rules may include groups of correlated features"

## Example Output

```
================================================================================
EDA-INFORMED RESULT ANALYSIS
Dataset: breast_cancer
================================================================================

## 1. DATASET OVERVIEW
--------------------------------------------------------------------------------
Training samples: 455
Test samples: 114
Features: 30
Classes: 2

## 2. CLASS DISTRIBUTION ANALYSIS
--------------------------------------------------------------------------------
Balanced: True
Class sizes:
  benign: 286 (62.9%)
  malignant: 169 (37.1%)

## 3. CLASS SEPARABILITY ANALYSIS
--------------------------------------------------------------------------------
Top 5 most separable features:
  • worst radius
  • worst perimeter
  • worst area
  • mean radius
  • mean perimeter

Precision explanations:
  • multi_agent_class_0: High precision: Classes are well-separated
  • multi_agent_class_1: High precision: Classes are well-separated

## 4. FEATURE CORRELATION ANALYSIS
--------------------------------------------------------------------------------
High correlations (|r| > 0.7): 12

Top correlations:
  • mean radius <-> mean perimeter: 0.998
  • mean radius <-> mean area: 0.987
  • mean perimeter <-> mean area: 0.993

Impact on rules:
  • multi_agent class 0: Rules include correlated features (mean radius, mean perimeter)

## 5. PRECISION-COVERAGE TRADE-OFF ANALYSIS
--------------------------------------------------------------------------------
multi_agent_class_0:
  • High precision, low coverage: Well-separated but concentrated distribution

## 6. METHOD COMPARISON WITH EDA CONTEXT
--------------------------------------------------------------------------------
EDA Context:
  • Balanced classes: True
  • High correlations: 12
  • Top separable features: worst radius, worst perimeter, worst area

Method Performance:
  multi_agent:
    Avg Precision: 0.962 ± 0.015
    Avg Coverage: 0.009 ± 0.003
  single_agent:
    Avg Precision: 0.958 ± 0.018
    Avg Coverage: 0.008 ± 0.002

Interpretations:
  • Multi-agent achieves higher coverage: Benefits from coordination
```

## Integration with Your Workflow

### For Paper Writing

Use the report to explain your results:

```markdown
## Results

Our multi-agent approach achieved precision of 0.96 and coverage of 0.98 
(class-level union). The EDA analysis reveals that:

1. **High Precision**: The classes are well-separated (confirmed by 
   separability analysis), allowing for precise anchors.

2. **High Class Coverage**: Despite low instance-level coverage (0.01), 
   the union of multiple anchors achieves 98% coverage. This is explained 
   by the high dimensionality (30 features) which makes individual anchors 
   narrow, but their union covers the class effectively.

3. **Feature Correlations**: The dataset contains 12 highly correlated 
   feature pairs (|r| > 0.7), which explains why rules often include groups 
   of correlated features (e.g., mean radius, mean perimeter, mean area).
```

### For Method Selection

Use EDA to guide method selection:

- **Balanced, well-separated classes**: Single-agent may be sufficient
- **Imbalanced classes**: Multi-agent helps coordinate coverage
- **High correlations**: Multi-agent can exploit feature groups
- **High dimensionality**: Multi-agent benefits from multiple anchor exploration

## Advanced Usage

### Custom Analysis

You can also use the analyzer programmatically:

```python
from analyze_results_with_eda import EDAResultAnalyzer

analyzer = EDAResultAnalyzer(
    dataset_name="breast_cancer",
    results_dir="comparison_results/breast_cancer_maddpg_20251222_201640",
    output_dir="custom_analysis"
)

# Get specific analyses
class_dist_impact = analyzer.analyze_class_distribution_impact()
separability_impact = analyzer.analyze_separability_impact()
corr_impact = analyzer.analyze_feature_correlation_impact()

# Generate custom report
report = analyzer.generate_report()
```

## Tips

1. **Run EDA first**: Always perform EDA before interpreting results
2. **Check all sections**: Each section provides different insights
3. **Compare methods**: Use the method comparison section to explain differences
4. **Document patterns**: Note common patterns across datasets

