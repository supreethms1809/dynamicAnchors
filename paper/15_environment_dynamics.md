# Environment Dynamics

## Overview

This document describes the detailed dynamics of the AnchorEnv environment, including the step-by-step execution flow, metric computation, and episode progression.

---

## Episode Flow

### 1. Episode Start (Reset)

**Process**:
1. Select initial instance/centroid per agent
2. Initialize box bounds around anchor point
3. Set `x_star_unit` for instance-based mode
4. Reset state variables
5. Compute initial metrics
6. Return initial observation

**Code Location**: `BenchMARL/environment.py` `reset()` method (lines 705-762)

---

### 2. Episode Loop (Step)

**For each step until termination**:

#### Step 2.1: Receive Actions

**Input**: `actions: Dict[str, np.ndarray]`
- One action per agent
- Action shape: `(2 * n_features,)`
- Components: `[delta_lower (n), delta_upper (n)]`

**Code Location**: `BenchMARL/environment.py` `step()` method (line 833)

---

#### Step 2.2: Process Each Agent

**For each agent**:

1. **Extract Action**:
   ```python
   action = actions[agent]
   delta_lower = action[:n_features]
   delta_upper = action[n_features:]
   ```

2. **Store Previous State**:
   ```python
   prev_lower = self.lower[agent].copy()
   prev_upper = self.upper[agent].copy()
   prev_precision, prev_coverage, _ = self._current_metrics(agent)
   ```

3. **Apply Action**:
   - Scale actions by `max_action_scale`
   - Compute step sizes relative to box width
   - Update bounds: `new_lower = lower + step_sizes_lower`
   - Clip to [0, 1]
   - Enforce minimum width

4. **Check Coverage Floor**:
   - If `coverage < min_coverage_floor`:
     - Revert action (restore `prev_lower`, `prev_upper`)
     - Set `coverage_clipped = True`
     - Reset stability counter

5. **Compute New Metrics**:
   ```python
   precision, coverage, details = self._current_metrics(agent)
   ```

6. **Compute Gains**:
   ```python
   precision_gain = precision - prev_precision
   coverage_gain = coverage - prev_coverage
   ```

7. **Compute Rewards**:
   - Local reward components
   - Penalties
   - Bonuses
   - Store in `reward_without_shared[agent]`

8. **Track Stability**:
   - If metrics stable: increment `_stable_counts[agent]`
   - Else: reset to 0

9. **Check Termination**:
   - Target-based conditions
   - Stabilization condition (if enabled)

**Code Location**: `BenchMARL/environment.py` `step()` method (lines 854-1200)

---

#### Step 2.3: Compute Shared Rewards (Multi-Agent)

**After all agents processed**:

```python
mean_local_reward = mean(reward_without_shared.values())
shared_reward = shared_reward_weight * mean_local_reward

for agent in agents:
    reward[agent] = reward_without_shared[agent] + shared_reward
```

**Purpose**: Encourages cooperation

**Code Location**: `BenchMARL/environment.py` `step()` method (lines 1201-1355)

---

#### Step 2.4: Build Observations

**For each agent**:
```python
observation = concatenate([
    self.lower[agent],      # n_features
    self.upper[agent],      # n_features
    precision,              # 1
    coverage                # 1
])
```

**Shape**: `(2 * n_features + 2,)`

**Code Location**: `BenchMARL/environment.py` `step()` method (lines 1119-1121)

---

#### Step 2.5: Check Termination

**Conditions checked**:
1. Target-based termination (if targets met)
2. Stabilization termination (if enabled and stabilized)
3. Maximum steps (truncation)

**Code Location**: `BenchMARL/environment.py` `step()` method (lines 1124-1394)

---

## Metric Computation (`_current_metrics`)

### Process Flow

**Location**: `BenchMARL/environment.py` `_current_metrics()` method (lines 457-682)

#### Step 1: Determine Data Source

**Training Mode**:
- `X_data_unit = self.X_unit` (training data)
- `X_data_std = self.X_std`
- `y_data = self.y`

**Evaluation Mode** (if `eval_on_test_data=True`):
- `X_data_unit = self.X_test_unit` (test data)
- `X_data_std = self.X_test_std`
- `y_data = self.y_test`

---

#### Step 2: Compute Coverage

**Instance-Based Mode** (when `x_star_unit[agent]` is set):
```python
mask = self._mask_in_box(agent)  # Points in box
coverage = mask.mean()  # Overall coverage: P(x in box)
```

**Class-Based Mode** (when `x_star_unit[agent]` is not set):
```python
mask = self._mask_in_box(agent)
class_mask = (y_data == target_class)
coverage = (mask & class_mask).sum() / class_mask.sum()  # Class-conditional: P(x in box | y = target)
```

**Code Location**: `BenchMARL/environment.py` `_mask_in_box()` method

---

#### Step 3: Sample Points for Precision Estimation

**Perturbation Sampling**:

**Bootstrap Mode**:
```python
covered = np.where(mask)[0]
n_samp = min(n_perturb, covered.size)
idx = rng.choice(covered, size=n_samp, replace=True)
X_eval = X_data_std[idx]
y_eval = y_data[idx]  # Ground truth labels available
```

**Uniform Mode**:
```python
# Sample uniformly within box bounds
U = uniform_samples_in_box(lower, upper, n_perturb)
X_eval = unit_to_std(U)
y_eval = None  # No labels (synthetic points)
```

**Adaptive Mode**:
- Bootstrap if `covered.size >= threshold`
- Uniform otherwise

**Code Location**: `BenchMARL/environment.py` `_current_metrics()` method (lines 570-624)

---

#### Step 4: Evaluate Classifier

```python
classifier.eval()
with torch.no_grad():
    inputs = torch.from_numpy(X_eval).float().to(device)
    logits = classifier(inputs)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
```

**Output**: `probs` shape `[n_samples, n_classes]`

**Code Location**: `BenchMARL/environment.py` `_current_metrics()` method (lines 626-634)

---

#### Step 5: Compute Precision

**With Labels** (bootstrap mode):
```python
hard_precision = (y_eval == target_class).mean()
```

**Without Labels** (uniform mode):
```python
preds = probs.argmax(axis=1)
hard_precision = (preds == target_class).mean()  # Proxy using predictions
```

**Precision Proxy**:
```python
avg_prob = probs[:, target_class].mean()
precision_proxy = (
    precision_blend_lambda * hard_precision +
    (1 - precision_blend_lambda) * avg_prob
)
```

**Code Location**: `BenchMARL/environment.py` `_current_metrics()` method (lines 636-655)

---

#### Step 6: Return Metrics

```python
return precision_proxy, coverage, {
    "hard_precision": hard_precision,
    "avg_prob": avg_prob,
    "n_points": n_points,
    "sampler": sampler_note,
    "target_class_fraction": target_class_fraction,
    "data_source": data_source,
}
```

---

## Class Union Metrics Computation

### Process Flow

**Location**: `BenchMARL/environment.py` `_compute_class_union_metrics()` method (lines 1501-1570)

#### Step 1: Initialize Union Mask

**For each class**:
```python
union_mask = np.zeros(n_samples, dtype=bool)
```

---

#### Step 2: Build Union from All Agents

**For each agent in class**:
```python
mask_agent = self._mask_in_box(agent)
union_mask |= mask_agent  # OR operation
```

**Result**: `union_mask[i] = True` if point `i` is in ANY agent's box

---

#### Step 3: Compute Class-Conditional Coverage

```python
mask_cls = (y_data == cls)  # Points belonging to class
cov_union = union_mask[mask_cls].mean()  # P(x in union | y = cls)
```

**Interpretation**: Fraction of class samples covered by union

---

#### Step 4: Compute Union Precision

```python
y_union = y_data[union_mask]  # Labels of points in union
prec_union = (y_union == cls).mean()  # P(y = cls | x in union)
```

**Interpretation**: Fraction of union samples that belong to class

---

#### Step 5: Return Metrics

```python
return {
    cls: {
        "union_coverage": cov_union,
        "union_precision": prec_union,
    }
    for cls in target_classes
}
```

---

## Box Update Mechanics

### Action Application (`_apply_continuous_action`)

**Location**: `BenchMARL/environment.py` `_apply_continuous_action()` method (lines 792-829)

#### Step 1: Extract Deltas

```python
delta_lower = action[:n_features]
delta_upper = action[n_features:]
```

---

#### Step 2: Scale Actions

```python
delta_lower_scaled = delta_lower * max_action_scale
delta_upper_scaled = delta_upper * max_action_scale
```

**Default**: `max_action_scale = 0.1`

---

#### Step 3: Compute Step Sizes

**Relative to Box Width**:
```python
widths = upper - lower
step_sizes_lower = delta_lower_scaled * widths
step_sizes_upper = delta_upper_scaled * widths
```

**Absolute Minimum**:
```python
step_sizes_lower = max(step_sizes_lower, min_absolute_step * sign(delta_lower_scaled))
step_sizes_upper = max(step_sizes_upper, min_absolute_step * sign(delta_upper_scaled))
```

---

#### Step 4: Update Bounds

```python
new_lower = lower + step_sizes_lower
new_upper = upper + step_sizes_upper
```

---

#### Step 5: Enforce Constraints

**Bounds Clipping**:
```python
new_lower = np.clip(new_lower, 0.0, 1.0)
new_upper = np.clip(new_upper, 0.0, 1.0)
```

**Minimum Width**:
```python
widths = new_upper - new_lower
if (widths < min_width).any():
    # Adjust bounds to maintain minimum width
```

**Ordering**:
```python
if new_lower >= new_upper:
    # Swap or adjust to maintain valid box
```

---

#### Step 6: Update State

```python
self.lower[agent] = new_lower
self.upper[agent] = new_upper
```

---

## State Representation

### Observation Space

**Components**:
1. **Lower Bounds**: `self.lower[agent]` (shape: `[n_features]`)
2. **Upper Bounds**: `self.upper[agent]` (shape: `[n_features]`)
3. **Precision**: Current precision (scalar)
4. **Coverage**: Current coverage (scalar)

**Concatenated**:
```python
observation = np.concatenate([
    self.lower[agent],
    self.upper[agent],
    np.array([precision, coverage], dtype=np.float32)
])
```

**Shape**: `(2 * n_features + 2,)`

**Range**: 
- Bounds: `[0, 1]` (normalized)
- Precision: `[0, 1]`
- Coverage: `[0, 1]`

---

### Internal State

**Per Agent**:
- `self.lower[agent]`: Lower bounds (normalized [0, 1])
- `self.upper[agent]`: Upper bounds (normalized [0, 1])
- `self.x_star_unit[agent]`: Anchor instance (if instance-based)
- `self.prev_lower[agent]`: Previous lower bounds
- `self.prev_upper[agent]`: Previous upper bounds
- `self.box_history[agent]`: List of box states

**Global**:
- `self.X_unit`: Normalized data [0, 1]
- `self.X_std`: Standardized data (mean=0, std=1)
- `self.y`: Class labels
- `self.classifier`: Trained classifier

---

## Episode Termination

### Termination Conditions

**1. Target-Based**:
- Both targets met: `precision >= precision_target AND coverage >= coverage_target`
- High precision: `precision >= 0.95 * precision_target AND coverage >= 0.7 * coverage_target`
- Both close: `precision >= 0.90 * precision_target AND coverage >= 0.90 * coverage_target`
- Excellent precision: `precision >= 0.99`

**2. Stabilization-Based**:
- Metrics stable for `stability_window` consecutive steps
- Requires minimum `stability_min_steps` before allowing termination

**3. Maximum Steps**:
- Episode truncated if `timestep >= max_cycles`

**Code Location**: `BenchMARL/environment.py` `step()` method (lines 1124-1394)

---

## Multi-Agent Coordination

### Shared Rewards

**Computation**:
```python
# After all agents act
mean_local = mean([reward_without_shared[agent] for agent in agents])
shared_reward = shared_reward_weight * mean_local

# Add to each agent
for agent in agents:
    reward[agent] = reward_without_shared[agent] + shared_reward
```

**Purpose**: Encourages cooperation

---

### Class Union Rewards (Optional)

**Computation**:
```python
class_union_metrics = self._compute_class_union_metrics()
class_precision_gain = class_union_metrics[class]["union_precision"] - prev_class_precision
class_coverage_gain = class_union_metrics[class]["union_coverage"] - prev_class_coverage

class_bonus = (
    class_union_prec_weight * class_precision_gain +
    class_union_cov_weight * class_coverage_gain
)
```

**Purpose**: Aligns agents toward class-level objectives

---

## Code References

- **Step Function**: `BenchMARL/environment.py` `step()` method (lines 831-1394)
- **Metric Computation**: `BenchMARL/environment.py` `_current_metrics()` method (lines 457-682)
- **Class Union**: `BenchMARL/environment.py` `_compute_class_union_metrics()` method (lines 1501-1570)
- **Action Application**: `BenchMARL/environment.py` `_apply_continuous_action()` method (lines 792-829)
- **Box Masking**: `BenchMARL/environment.py` `_mask_in_box()` method

---

## Summary

**Episode Flow**:

1. **Reset**: Initialize boxes, compute initial metrics
2. **Step Loop**:
   - Receive actions
   - Apply actions to boxes
   - Compute metrics
   - Compute rewards
   - Check termination
3. **Termination**: When targets met, stabilized, or max steps reached

**Key Processes**:

- **Metric Computation**: Perturbation sampling → classifier evaluation → precision/coverage
- **Box Updates**: Action scaling → step computation → constraint enforcement
- **Multi-Agent**: Shared rewards, class union metrics

**For Paper Writing**:

- Describe episode flow
- Explain metric computation process
- Detail box update mechanics
- Discuss multi-agent coordination
