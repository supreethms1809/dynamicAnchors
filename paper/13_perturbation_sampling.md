# Perturbation Sampling

## Overview

Perturbation sampling is used to efficiently estimate precision and coverage without evaluating all data points. This document describes the three sampling modes: bootstrap, uniform, and adaptive.

---

## Purpose

### Why Perturbation Sampling?

**Problem**: Evaluating precision/coverage on full dataset is expensive

**Solution**: Sample subset of points for evaluation

**Benefits**:
- Faster computation
- Scales to large datasets
- Good approximation with sufficient samples

**Trade-off**: Approximation vs. exact computation

---

## Sampling Modes

### 1. Bootstrap Sampling

**Method**: Sample with replacement from covered points

**Process**:
1. Identify points in box: `covered = mask == True`
2. Sample with replacement: `idx = rng.choice(covered, size=n_perturb, replace=True)`
3. Evaluate on sampled points

**Formula**:
```python
n_samp = min(n_perturb, covered.size)
idx = rng.choice(covered, size=n_samp, replace=True)
X_eval = X_data_std[idx]
y_eval = y_data[idx]
```

**When Used**: When `perturbation_mode == "bootstrap"`

**Advantages**:
- Uses real data points
- Preserves data distribution
- Good for precision estimation

**Disadvantages**:
- Requires sufficient covered points
- May miss uncovered regions

**Code Location**: `BenchMARL/environment.py` lines 570-577

---

### 2. Uniform Sampling

**Method**: Uniform random sampling within box bounds

**Process**:
1. For each feature, sample uniformly from `[lower, upper]`
2. Generate `n_perturb` synthetic samples
3. Evaluate on synthetic points

**Formula**:
```python
for j in range(n_features):
    low, up = lower[j], upper[j]
    width = max(up - low, min_width)
    mid = 0.5 * (low + up)
    low = max(0.0, mid - width / 2.0)
    up = min(1.0, mid + width / 2.0)
    U[:, j] = rng.uniform(low=low, high=up, size=n_perturb)
X_eval = unit_to_std(U)  # Convert to standardized space
```

**When Used**: When `perturbation_mode == "uniform"`

**Advantages**:
- Works even with few/no covered points
- Uniform coverage of box
- Good for coverage estimation

**Disadvantages**:
- Synthetic points (not real data)
- May include points outside data distribution
- Less accurate for precision (no labels)

**Code Location**: `BenchMARL/environment.py` lines 579-599

---

### 3. Adaptive Sampling

**Method**: Bootstrap when enough covered points, otherwise uniform

**Process**:
1. Check if enough covered points: `covered.size >= min_points_for_bootstrap`
2. **If yes**: Use bootstrap sampling
3. **If no**: Use uniform sampling

**Threshold**:
```python
min_points_for_bootstrap = max(1, int(0.1 * n_perturb))
```

**Default**: `n_perturb = 4096` → threshold = 410 points

**When Used**: When `perturbation_mode == "adaptive"` (default)

**Advantages**:
- Best of both worlds
- Bootstrap when possible (real data)
- Uniform fallback (always works)

**Disadvantages**:
- More complex logic
- May switch modes during episode

**Code Location**: `BenchMARL/environment.py` lines 584-620

---

## Sampling Parameters

### n_perturb

**Default**: 4096

**Purpose**: Number of samples to generate

**Effect**:
- **More samples**: Better approximation, slower
- **Fewer samples**: Faster, less accurate

**Typical Range**: 1024-8192

---

### min_points_for_bootstrap

**Formula**: `max(1, int(0.1 * n_perturb))`

**Default**: 410 (for `n_perturb=4096`)

**Purpose**: Minimum covered points needed for bootstrap

**Effect**: Controls when adaptive mode switches to uniform

---

## Data Source Selection

### Training Data vs Test Data

**Training Mode**: Uses training data (`X_train_std`, `y_train`)

**Evaluation Mode**: Uses test data (`X_test_std`, `y_test`) if `eval_on_test_data=True`

**Inference Mode**: Uses test data by default; if `coverage_on_all_data=True`, metrics use the combined train+test dataset

**Code Location**: `BenchMARL/environment.py` `_current_metrics()` method

---

## Sampling Process Flow

### Step 1: Identify Covered Points

**Process**:
```python
mask = (X_data_unit >= lower) & (X_data_unit <= upper)
covered = np.where(mask.all(axis=1))[0]
coverage = covered.size / len(X_data_unit)
```

**Purpose**: Find which points are in box

---

### Step 2: Select Sampling Mode

**Bootstrap**:
- Always use bootstrap
- Requires `covered.size > 0`

**Uniform**:
- Always use uniform
- Works even with `covered.size == 0`

**Adaptive**:
- Bootstrap if `covered.size >= threshold`
- Uniform otherwise

---

### Step 3: Generate Samples

**Bootstrap**:
```python
n_samp = min(n_perturb, covered.size)
idx = rng.choice(covered, size=n_samp, replace=True)
X_eval = X_data_std[idx]
y_eval = y_data[idx]
```

**Uniform**:
```python
U = uniform_samples_in_box(lower, upper, n_perturb)
X_eval = unit_to_std(U)
y_eval = None  # No labels for synthetic points
```

---

### Step 4: Evaluate Precision/Coverage

**With Labels** (bootstrap):
```python
precision = (y_eval == target_class).mean()
```

**Without Labels** (uniform):
```python
# Use model predictions as proxy
probs = classifier(X_eval)
precision_proxy = probs[:, target_class].mean()
```

---

## Mode Selection Logic

### Adaptive Mode Decision

```python
if perturbation_mode == "adaptive":
    min_points = max(1, int(0.1 * n_perturb))
    
    if covered.size >= min_points:
        # Use bootstrap
        sampler_note = "adaptive_bootstrap"
    else:
        # Use uniform
        sampler_note = "adaptive_uniform"
```

**Logging**: Records which mode was used

**Code Location**: `BenchMARL/environment.py` lines 584-620

---

## Coverage Estimation

### Bootstrap Mode

**Coverage**: Uses actual coverage from covered points

**Formula**:
```python
coverage = covered.size / total_samples
```

**Accuracy**: Exact (within sampling variance)

---

### Uniform Mode

**Coverage**: Always computed from the actual dataset mask (not from synthetic samples)

**Note**: Uniform sampling affects precision estimation only; coverage is still `mask.mean()` on real data

---

## Precision Estimation

### Bootstrap Mode (With Labels)

**Precision**: Uses ground truth labels

**Formula**:
```python
precision = (y_eval == target_class).mean()
```

**Accuracy**: Exact (within sampling variance)

---

### Uniform Mode (Without Labels)

**Precision**: Uses model predictions as proxy

**Formula**:
```python
probs = classifier(X_eval)
precision_proxy = probs[:, target_class].mean()
```

**Accuracy**: Approximation (model confidence, not ground truth)

**Instance-Based Fix**:
- For instance-based anchors, the original instance is prepended to uniform/adaptive samples so precision is anchored to the correct prediction.

**Note**: May overestimate precision (model may be overconfident)

---

## Configuration

### Environment Config

**From**: `BenchMARL/conf/anchor.yaml`

```yaml
env_config:
  use_perturbation: True
  perturbation_mode: "adaptive"  # "bootstrap", "uniform", or "adaptive"
  n_perturb: 4096
```

---

## Code References

- **Bootstrap Sampling**: `BenchMARL/environment.py` lines 570-577
- **Uniform Sampling**: `BenchMARL/environment.py` lines 579-599
- **Adaptive Sampling**: `BenchMARL/environment.py` lines 584-620
- **Single-Agent**: `single_agent/single_agentENV.py` similar implementation

---

## Summary

**Sampling Modes**:

1. **Bootstrap**: Sample with replacement from covered points (real data)
2. **Uniform**: Uniform random sampling within box (synthetic data)
3. **Adaptive**: Bootstrap when possible, uniform otherwise (default)

**Key Parameters**:

- `n_perturb`: Number of samples (default: 4096)
- `perturbation_mode`: Sampling strategy (default: "adaptive")
- `min_points_for_bootstrap`: Threshold for adaptive mode (10% of n_perturb)

**Trade-offs**:

- **Bootstrap**: Real data, requires covered points
- **Uniform**: Always works, synthetic data
- **Adaptive**: Best of both worlds

**For Paper Writing**:

- Explain why perturbation sampling is needed
- Describe three modes and their trade-offs
- Discuss adaptive mode benefits
- Mention precision/coverage estimation differences

---

## When Perturbation Sampling is Used vs. Actual Dataset

### During Training and Inference Rollouts

**Perturbation Sampling IS Used**:
- **Precision computation** during training steps (`_current_metrics()`)
- **Precision computation** during inference rollouts
- Stored as `instance_precision` in anchor data
- Purpose: Efficient training signal and rollout evaluation

**Coverage Computation**:
- **Always uses actual dataset** (never perturbation sampling)
- Formula: `coverage = mask.mean()` where `mask` is computed on actual data points
- Purpose: Coverage measures fraction of actual dataset covered

### Final Union Computation

**Actual Dataset IS Used**:
- **Precision filtering**: Recomputes precision on actual dataset before filtering
- **Union computation**: Uses actual dataset (`X_data_union`, `y_data_union`)
- **Final metrics**: Computed on actual dataset, not perturbation samples

**Why Recompute Precision?**:
- Stored `instance_precision` comes from perturbation sampling (may be inaccurate)
- Final union computation uses actual dataset (accurate)
- Recomputing ensures filtering uses same data and method as final computation
- Prevents mismatch between stored precision and actual precision

**Key Insight**: Perturbation sampling is used for **efficiency during training/rollouts**, but **final evaluation always uses the actual dataset** for accuracy.
