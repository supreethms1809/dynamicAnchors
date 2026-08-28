# Action Space

## Overview

This document describes the action space used in the Dynamic Anchors environment, including action representation, application, and constraints.

---

## Action Representation

### Continuous Action Space

**Type**: Continuous (Box space)

**Dimension**: `2 * n_features`

**Components**:
- First `n_features`: Deltas for lower bounds
- Next `n_features`: Deltas for upper bounds

**Range**: `[-1.0, 1.0]` (before scaling)

**Code**:
```python
action_space = Box(
    low=-1.0,
    high=1.0,
    shape=(2 * n_features,),
    dtype=np.float32
)
```

**Code Location**: `BenchMARL/environment.py` (observation/action space definition)

---

## Action Application

### Step 1: Extract Deltas

**From Action**:
```python
delta_lower = action[:n_features]
delta_upper = action[n_features:]
```

**Range**: `[-1.0, 1.0]`

---

### Step 2: Scale Actions

**Scaling Factor**: `max_action_scale` (default: 0.1)

**Formula**:
```python
delta_lower_scaled = delta_lower * max_action_scale
delta_upper_scaled = delta_upper * max_action_scale
```

**Result**: Actions scaled to `[-max_action_scale, max_action_scale]`

**Purpose**: Controls step size, prevents large jumps

---

### Step 3: Compute Step Sizes

**Method**: Relative to current box width

**Formula**:
```python
widths = upper - lower
step_sizes_lower = delta_lower_scaled * widths
step_sizes_upper = delta_upper_scaled * widths
```

**Purpose**: Makes steps relative to current box size

**Alternative**: Absolute steps (if `min_absolute_step` is set)

---

### Step 4: Apply Steps

**Update Bounds**:
```python
new_lower = lower + step_sizes_lower
new_upper = upper + step_sizes_upper
```

**Constraints**:
- `new_lower >= 0` (clip to 0)
- `new_upper <= 1` (clip to 1)
- `new_lower < new_upper` (maintain valid box)

---

## Action Constraints

### 1. Bounds Clipping

**Lower Bound**:
```python
new_lower = clip(new_lower, 0.0, 1.0)
```

**Upper Bound**:
```python
new_upper = clip(new_upper, 0.0, 1.0)
```

**Purpose**: Keeps box within [0, 1] normalized space

---

### 2. Minimum Width Constraint

**Constraint**: `upper - lower >= min_width` per feature

**Default**: `min_width = 0.05`

**Enforcement**: Applied after action, may revert if violated

**Purpose**: Prevents boxes from collapsing to zero width

---

### 3. Coverage Floor Constraint

**Constraint**: Coverage must be ≥ `min_coverage_floor`

**Default**: `min_coverage_floor = 0.005` (dynamically overridden to `1 / n_samples` during training/inference)

**Enforcement**: If violated, action is reverted

**Purpose**: Ensures box always covers anchor instance

---

### 4. Minimum Absolute Step

**Constraint**: Steps must be ≥ `min_absolute_step`

**Default**: `min_absolute_step = 0.001`

**Formula**:
```python
step_size = max(step_size, min_absolute_step * sign(step_size))
```

**Purpose**: Prevents tiny steps that don't change box

---

## Action Scaling Parameters

### max_action_scale

**Default**: 0.1-0.15

**Purpose**: Controls maximum step size relative to box width

**Effect**:
- **Small (0.05)**: Smaller steps, slower learning
- **Large (0.2)**: Larger steps, faster learning but less stable

**Code Location**: `BenchMARL/environment.py` line 185

---

### min_absolute_step

**Default**: 0.001-0.01

**Purpose**: Ensures steps are meaningful

**Effect**: Prevents actions that don't change box

**Code Location**: `BenchMARL/environment.py` line 186

---

## Discrete Action Space (Legacy)

### Step Fractions

**Definition**: `step_fracs = [0.005, 0.01, 0.02]`

**Purpose**: Defines relative step sizes for discrete actions

**Note**: Currently using continuous actions, discrete mode not actively used

**Code Location**: `BenchMARL/environment.py` line 108

---

## Action Application Code

### Multi-Agent

**Location**: `BenchMARL/environment.py` `_apply_continuous_action()` method

**Process**:
1. Extract deltas from action
2. Scale by `max_action_scale`
3. Compute step sizes relative to box width
4. Apply steps to bounds
5. Clip to [0, 1]
6. Enforce minimum width
7. Check coverage floor

---

### Single-Agent

**Location**: `single_agent/single_agentENV.py` `step()` method

**Process**: Similar to multi-agent, but for single agent

---

## Action Noise (Inference)

### Exploration Modes

**Modes**:
- `"sample"`: Sample from policy distribution
- `"mean"`: Use mean action (deterministic)
- `"noisy_mean"`: Add noise to mean action

**Noise Scale**: `action_noise_scale` (default: 0.05)

**Purpose**: Adds diversity during inference

**Code Location**: `BenchMARL/inference.py` `run_rollout_with_policy()` method

---

## Action Interpretation

### Positive Deltas

**Lower Bound**: `delta_lower > 0` → Expand lower bound (box grows downward)

**Upper Bound**: `delta_upper > 0` → Expand upper bound (box grows upward)

**Result**: Box expands

---

### Negative Deltas

**Lower Bound**: `delta_lower < 0` → Shrink lower bound (box shrinks from bottom)

**Upper Bound**: `delta_upper < 0` → Shrink upper bound (box shrinks from top)

**Result**: Box shrinks

---

### Mixed Deltas

**Example**: `delta_lower > 0, delta_upper < 0`

**Result**: Box shifts (lower expands, upper shrinks)

**Purpose**: Allows box to move/center around anchor point

---

## Code References

- **Action Space Definition**: `BenchMARL/environment.py` (observation/action spaces)
- **Action Application**: `BenchMARL/environment.py` `_apply_continuous_action()`
- **Constraints**: `BenchMARL/environment.py` `step()` method
- **Single-Agent**: `single_agent/single_agentENV.py` `step()` method

---

## Summary

**Action Space**:

- **Type**: Continuous Box space
- **Dimension**: `2 * n_features` (deltas for lower and upper bounds)
- **Range**: `[-1.0, 1.0]` before scaling

**Action Application**:

1. Extract deltas
2. Scale by `max_action_scale`
3. Compute relative step sizes
4. Apply to bounds
5. Enforce constraints

**Constraints**:

- Bounds in [0, 1]
- Minimum width per feature
- Coverage floor
- Minimum absolute step

**For Paper Writing**:

- Describe continuous action space
- Explain action scaling and application
- Discuss constraints (bounds, width, coverage)
- Mention action noise for exploration
