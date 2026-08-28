# Termination Conditions

## Overview

This document describes the early termination conditions used in the Dynamic Anchors environment to end episodes when targets are met or when the system stabilizes.

---

## Termination Types

### 1. Target-Based Termination

**Purpose**: Terminate when precision/coverage targets are achieved

**Conditions**:

#### 1.1 Both Targets Met

**Formula**:
```python
both_targets_met = (
    precision >= precision_target and 
    coverage >= coverage_target
)
```

**Thresholds**:
- `precision_target`: 0.95 (default)
- `coverage_target`: 0.5 (multi-agent), 0.3 (single-agent)

**Termination**: `True` if both conditions met

**Counter Limit**: `max_termination_count_both_targets` (default: -1, unlimited)

**Code Location**: `BenchMARL/environment.py` lines 1124-1125

---

#### 1.2 High Precision with Reasonable Coverage

**Formula**:
```python
high_precision_with_reasonable_coverage = (
    precision >= 0.95 * precision_target and
    coverage >= 0.7 * coverage_target
)
```

**Thresholds**:
- Precision: ≥ 0.9025 (0.95 × 0.95)
- Coverage: ≥ 0.21-0.35 (0.7 × coverage_target)

**Purpose**: Allows termination when precision is high but coverage is slightly below target

**Counter Limit**: `max_termination_count_high_precision` (default: 200)

**Code Location**: `BenchMARL/environment.py` lines 1126-1128

---

#### 1.3 Both Reasonably Close

**Formula**:
```python
both_reasonably_close = (
    precision >= 0.90 * precision_target and
    coverage >= 0.90 * coverage_target
)
```

**Thresholds**:
- Precision: ≥ 0.855 (0.90 × 0.95)
- Coverage: ≥ 0.27-0.45 (0.90 × coverage_target)

**Purpose**: Allows termination when both metrics are close to targets

**Counter Limit**: `max_termination_count_both_close` (default: 100 in current configs)

**Code Location**: `BenchMARL/environment.py` lines 1129-1131

---

#### 1.4 Excellent Precision

**Formula**:
```python
excellent_precision = precision >= 0.99
```

**Threshold**: Precision ≥ 0.99

**Purpose**: Terminate early when precision is very high (even if coverage is low)

**Counter Limit**: `max_termination_count_excellent_precision` (default: 100 in current configs)

**Code Location**: `BenchMARL/environment.py` lines 1132-1133

---

### 2. Stabilization-Based Termination

**Purpose**: Terminate when metrics stabilize (no significant changes)

**Location**: `BenchMARL/environment.py` lines 1370-1378

**Process**:

1. **Track Stability**: Count consecutive stable steps
2. **Stability Criteria**:
   ```python
   stable = (
       abs(precision - prev_precision) <= stability_precision_tol and
       abs(coverage - prev_coverage) <= stability_coverage_tol and
       anchor_drift <= stability_drift_tol
   )
   ```
3. **Increment Counter**: If stable, increment `_stable_counts[agent]`
4. **Reset Counter**: If not stable, reset to 0
5. **Terminate**: If counter ≥ `stability_window`

**Parameters**:
- `stability_window`: 20 (number of consecutive stable steps)
- `stability_min_steps`: 50 (minimum steps before allowing termination)
- `stability_precision_tol`: 1e-3 (tolerance for precision changes)
- `stability_coverage_tol`: 1e-3 (tolerance for coverage changes)
- `stability_drift_tol`: 1e-3 (tolerance for box drift)

**Code Location**: `BenchMARL/environment.py` lines 960-983, 1370-1378

---

### 3. Maximum Steps Termination

**Purpose**: Truncate episodes that exceed maximum length

**Formula**:
```python
max_steps_reached = timestep >= max_cycles
```

**Default**: `max_cycles = 500`

**Result**: Episode truncated (not terminated)

**Code Location**: `BenchMARL/environment.py` lines 1365, 1381-1394

---

## Termination Priority

### Order of Checks

1. **Target-Based Termination**: Check if targets met
2. **Stabilization Termination**: Check if stabilized (if enabled)
3. **Maximum Steps**: Truncate if exceeded

**Note**: Multiple conditions can be true, but termination happens on first match

---

## Termination Reason Tracking

### Counters

**Purpose**: Prevent overuse of termination conditions

**Counters**:
- `termination_reason_max_counts`: Dict mapping reason → max count
- Tracks how many times each reason was used
- Disables reason if max count exceeded

**Default Limits**:
```python
{
    "both_targets_met": -1,  # Unlimited
    "excellent_precision": 100,
    "high_precision_reasonable_coverage": 200,
    "both_reasonably_close": 100
}
```

**Code Location**: `BenchMARL/environment.py` lines 131-136

---

## Reset Behavior

### Training Mode

**Behavior**: Counters persist across episodes

**Purpose**: Prevents overuse of easy termination conditions

### Evaluation/Inference Mode

**Behavior**: Counters persist across episodes; they reset at environment initialization or via explicit reset calls.

**Purpose**: Prevent overuse of easy termination conditions across a full evaluation run.

**Notes**:
- In SB3 evaluation, a callback resets counters before evaluation starts.
- In single-agent inference mode, lenient termination reasons are disabled at env initialization.

---

## Single-Agent Termination

### Similar Conditions

**Location**: `single_agent/single_agentENV.py` lines 846-913

**Key Differences**:
- Single agent (no loop over agents)
- No stabilization-based termination in single-agent
- Minimum steps before termination is enforced (`min_steps_before_termination = 2`)
- In inference mode, lenient termination reasons (`excellent_precision`, `high_precision_reasonable_coverage`) are disabled

**Code Reference**: `single_agent/single_agentENV.py` `step()` method

---

## Stabilization Tracking Details

### Stability Computation

**Per Step**:
```python
dp = abs(precision - prev_precision)
dc = abs(coverage - prev_coverage)
anchor_drift = max(
    abs(lower - prev_lower).max(),
    abs(upper - prev_upper).max()
)

stable = (
    dp <= stability_precision_tol and
    dc <= stability_coverage_tol and
    anchor_drift <= stability_drift_tol
)
```

**Counter Update**:
```python
if stable:
    _stable_counts[agent] += 1
else:
    _stable_counts[agent] = 0
```

**Termination Check**:
```python
if timestep >= stability_min_steps:
    if _stable_counts[agent] >= stability_window:
        terminate = True
```

**Code Location**: `BenchMARL/environment.py` lines 960-983

---

## Configuration

### Environment Config

**From**: `BenchMARL/conf/anchor.yaml`

```yaml
env_config:
  # Stabilization-based early termination
  enable_stability_termination: true
  stability_window: 20
  stability_min_steps: 50
  stability_precision_tol: 1e-3
  stability_coverage_tol: 1e-3
  stability_drift_tol: 1e-3
  
  # Termination reason counters
  max_termination_count_both_targets: -1
  max_termination_count_high_precision: 200
  max_termination_count_both_close: 100
  max_termination_count_excellent_precision: 100
```

---

## Edge Cases

### Coverage Floor Violation

**Behavior**: Action reverted, termination prevented

**Code**:
```python
if coverage < min_coverage_floor:
    # Revert action
    # Reset stability counter
    _stable_counts[agent] = 0
```

**Purpose**: Prevents termination when box is invalid

---

### Invalid Bounds

**Behavior**: Termination prevented until bounds fixed

**Checks**:
- Bounds outside [0, 1]
- NaN or Inf values

**Code Location**: `single_agent/single_agentENV.py` lines 886-897

---

## Termination Information

### Info Dictionary

**Stored Information**:
- `termination_reason`: Integer code
- `termination_reason_str`: String description
- `stabilized`: 1.0 if stabilized, 0.0 otherwise

**Purpose**: Logging and analysis

**Code Location**: `BenchMARL/environment.py` lines 1135-1200

---

## Code References

- **Termination Conditions**: `BenchMARL/environment.py` lines 1124-1133
- **Stabilization Tracking**: `BenchMARL/environment.py` lines 960-983, 1370-1378
- **Counters**: `BenchMARL/environment.py` lines 131-136
- **Single-Agent**: `single_agent/single_agentENV.py` lines 846-913

---

## Summary

**Termination Types**:

1. **Target-Based**: When precision/coverage targets met
2. **Stabilization-Based**: When metrics stabilize
3. **Maximum Steps**: When episode length exceeded

**Key Features**:

- **Multiple conditions**: Flexible termination criteria
- **Counters**: Prevent overuse of easy conditions
- **Stabilization**: Early termination when converged
- **Mode-dependent**: Different behavior for training vs evaluation

**For Paper Writing**:

- Explain early termination benefits (efficiency)
- Describe stabilization-based termination
- Discuss termination condition priorities
- Mention counter limits for fair evaluation
