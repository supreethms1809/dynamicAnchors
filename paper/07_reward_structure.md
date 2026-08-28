# Reward Structure

## Overview

The reward structure guides agents to learn policies that generate high-precision, high-coverage anchor boxes. The system uses a combination of local rewards (per-agent) and shared rewards (multi-agent cooperation).

---

## Single-Agent Reward Structure

### Location: `single_agent/single_agentENV.py` lines 682-836

### Complete Reward Formula

```python
reward = (
    alpha * precision_weight * precision_gain_for_reward +
    beta * coverage_weight * coverage_gain_for_reward +
    coverage_bonus +
    target_class_bonus -
    progress_factor * overlap_penalty -
    progress_factor * drift_penalty -
    progress_factor * anchor_drift_penalty -
    progress_factor * js_penalty +
    coverage_floor_penalty
)
```

### Reward Components

#### 1. Precision Gain Component

**Formula**:
```python
precision_gain = precision - prev_precision
precision_gain_normalized = precision_gain / max(prev_precision, 1e-6)
precision_gain_scaled = precision_gain_normalized * 0.5
precision_gain_scaled = clip(precision_gain_scaled, -0.5, 0.5)
precision_gain_for_reward = precision_gain_scaled
```

**Weight**: `alpha * precision_weight` (default: `0.7 * precision_weight`)

**Purpose**: Rewards increases in precision

**Interpretation**:
- **Positive**: Precision increased → positive reward
- **Negative**: Precision decreased → negative reward
- **Clipped**: Prevents reward explosions

#### 2. Coverage Gain Component

**Formula**:
```python
coverage_gain = coverage - prev_coverage
coverage_gain_normalized = coverage_gain / max(prev_coverage, 1e-6)
coverage_gain_scaled = coverage_gain_normalized * 1.0  # Increased from 0.5
coverage_gain_scaled = clip(coverage_gain_scaled, -1.0, 1.0)
coverage_gain_for_reward = coverage_gain_scaled
```

**Weight**: `beta * coverage_weight` (default: `0.6 * coverage_weight`)

**Purpose**: Rewards increases in coverage

**Key Feature**: Scaling increased from 0.5 to 1.0 to give stronger signal for coverage expansion

#### 3. Coverage Bonus

**Formula**:
```python
coverage_bonus = (
    coverage_bonus_weight_met * (coverage / coverage_target)  # If targets met
    + coverage_bonus_weight_high_prec * base_weight  # If precision >= threshold
    + coverage_bonus_weight_high_prec_progress * progress_multiplier  # Progress bonus
    + coverage_bonus_weight_high_prec_distance * distance_multiplier  # Distance bonus
    + coverage_bonus_weight_reasonable_prec * base_weight  # If precision >= 0.8 * threshold
    + coverage_bonus_weight_reasonable_prec_progress * progress_multiplier
)
```

**Weights** (single-agent defaults):
- `coverage_bonus_weight_met`: 0.01
- `coverage_bonus_weight_high_prec`: 0.03
- `coverage_bonus_weight_high_prec_progress`: 0.07
- `coverage_bonus_weight_high_prec_distance`: 0.02
- `coverage_bonus_weight_reasonable_prec`: 0.01
- `coverage_bonus_weight_reasonable_prec_progress`: 0.02

**Multi-agent note**: In `BenchMARL/conf/anchor.yaml`, `coverage_bonus_weight_met` and `coverage_bonus_weight_high_prec` are higher (0.1) to provide a stronger coverage signal.

**Purpose**: Structured rewards for good coverage behavior

**Conditions**:
- **Targets met**: When precision ≥ target AND coverage ≥ target
- **High precision**: When precision ≥ threshold (0.8 * precision_target)
- **Reasonable precision**: When precision ≥ 0.8 * threshold

#### 4. Target Class Bonus

**Formula**:
```python
target_class_bonus = (
    target_class_bonus_weight * 
    (target_class_fraction / precision_threshold) *
    (precision < precision_threshold)  # Only if precision below threshold
)
```

**Weight**: `target_class_bonus_weight` (default: 0.02)

**Purpose**: Bonus when precision is below threshold but target class fraction is high

**Interpretation**: Encourages maintaining target class samples even when precision is low

#### 5. Overlap Penalty

**Formula**:
```python
overlap_penalty = gamma * mean(widths < 2 * min_width)
```

**Weight**: `gamma` (default: 0.1)

**Purpose**: Penalizes boxes that are too narrow (may collapse)

**Interpretation**: Encourages boxes to maintain minimum width

#### 6. Drift Penalty

**Formula**:
```python
drift = ||upper - prev_upper|| + ||lower - prev_lower||
drift_penalty = drift_penalty_weight * drift
```

**Weight**: `drift_penalty_weight` (default: 0.05)

**Purpose**: Penalizes large box movements

**Interpretation**: Encourages stable, gradual adjustments

#### 7. Anchor Drift Penalty

**Formula**:
```python
anchor_drift = max(
    ||lower - x_star_unit||,
    ||upper - x_star_unit||
)
anchor_drift_penalty = penalty_weight * anchor_drift
```

**Purpose**: Penalizes boxes that drift away from anchor instance

**Interpretation**: Keeps boxes centered around anchor point (for instance-based mode)

#### 8. JS Penalty (Jaccard-like)

**Formula**:
```python
inter_vol = intersection_volume(current_box, prev_box)
curr_vol = volume(current_box)
prev_vol = volume(prev_box)
js_proxy = 1.0 - (inter_vol / (0.5 * (prev_vol + curr_vol)))
js_penalty = js_penalty_weight * js_proxy
```

**Weight**: `js_penalty_weight` (default: 0.05)

**Purpose**: Penalizes large changes in box volume

**Interpretation**: Encourages gradual volume changes

#### 9. Coverage Floor Penalty

**Formula**:
```python
if coverage < min_coverage_floor:
    coverage_floor_penalty = -0.05  # Small negative reward
    # Action is reverted, penalties reduced by 90%
```

**Purpose**: Penalty for attempting invalid actions (coverage too low)

**Interpretation**: Prevents boxes from collapsing below minimum coverage

### Progress Factor

**Formula**:
```python
if precision_gain > 0 or coverage_gain > 0:
    progress_factor = 0.5  # Reduce penalties when making progress
elif precision >= precision_threshold * 0.8:
    progress_factor = 0.7  # Reduce penalties when close to target
else:
    progress_factor = 1.0  # Full penalties
```

**Purpose**: Reduces penalties when agent is making progress or close to target

**Interpretation**: Encourages exploration when making progress

---

## Multi-Agent Reward Structure

### Location: `BenchMARL/environment.py` lines 1100-1200

### Complete Reward Formula

```python
# Local reward (per-agent)
reward_local = (
    alpha * precision_weight * precision_gain_for_reward +
    beta * coverage_weight * coverage_gain_for_reward +
    coverage_bonus +
    target_class_bonus -
    progress_factor * overlap_penalty -
    progress_factor * drift_penalty -
    progress_factor * anchor_drift_penalty -
    progress_factor * js_penalty -
    progress_factor * inter_class_overlap_penalty -
    progress_factor * same_class_overlap_penalty +
    coverage_floor_penalty
)

# Shared reward (cooperation)
reward_shared = shared_reward_weight * mean(reward_local across all agents)

# Final reward
reward_final = reward_local + reward_shared
```

### Additional Multi-Agent Components

#### 1. Inter-Class Overlap Penalty

**Formula**:
```python
inter_class_overlap_penalty = inter_class_overlap_weight * overlap_score
```

**Weight**: `inter_class_overlap_weight` (default: 0.1)

**Purpose**: Penalizes overlap between agents of different classes

**Interpretation**: Prevents agents from different classes from interfering

**Code**: `_compute_inter_class_overlap_penalty()` in `environment.py`

#### 2. Same-Class Overlap Penalty

**Formula**:
```python
same_class_overlap_penalty = same_class_diversity_weight * overlap_score
```

**Weight**: `same_class_diversity_weight` (default: 0.0, can be enabled)

**Purpose**: Encourages diversity within same class (when enabled)

**Interpretation**: Prevents agents of same class from converging to same box

**Code**: `_compute_same_class_overlap_penalty()` in `environment.py`

#### 3. Shared Reward

**Formula**:
```python
reward_shared = shared_reward_weight * mean([reward_local_1, reward_local_2, ..., reward_local_K])
```

**Weight**: `shared_reward_weight` (default: 0.5)

**Purpose**: Encourages cooperation between agents

**Interpretation**: All agents receive bonus based on average performance

**Key Feature**: Computed AFTER all agents have acted

#### 4. Class Union Rewards (Optional)

**Formula**:
```python
class_union_reward = (
    class_union_cov_weight * class_union_coverage_gain +
    class_union_prec_weight * class_union_precision_gain
)
```

**Weights** (defaults in current config):
- `class_union_cov_weight`: 0.01
- `class_union_prec_weight`: 0.01

**Purpose**: Rewards improvements in class-level union metrics

**Interpretation**: Aligns agents toward class-level objectives

#### 5. Global Coverage Reward (Optional)

**Formula**:
```python
if global_coverage >= global_coverage_threshold:
    global_coverage_reward = global_coverage_weight * global_coverage
```

**Weights** (defaults in current config):
- `global_coverage_weight`: 0.01
- `global_coverage_threshold`: 0.1

**Purpose**: Rewards when all agents together cover the dataset

**Interpretation**: Encourages global explainability

---

## Reward Weight Configuration

### Single-Agent Defaults

**From**: `single_agent/conf/anchor_single.yaml`

```yaml
alpha: 0.7              # Precision gain weight
beta: 0.6               # Coverage gain weight
gamma: 0.1              # Overlap penalty weight
drift_penalty_weight: 0.05
js_penalty_weight: 0.05
coverage_bonus_weight_met: 0.01
coverage_bonus_weight_high_prec: 0.03
coverage_bonus_weight_high_prec_progress: 0.07
coverage_bonus_weight_high_prec_distance: 0.02
coverage_bonus_weight_reasonable_prec: 0.01
coverage_bonus_weight_reasonable_prec_progress: 0.02
target_class_bonus_weight: 0.02
```

### Multi-Agent Defaults

**From**: `BenchMARL/conf/anchor.yaml`

```yaml
alpha: 0.7              # Precision gain weight
beta: 0.6               # Coverage gain weight (aligned with single-agent)
gamma: 0.1              # Overlap penalty weight
drift_penalty_weight: 0.05
js_penalty_weight: 0.05
inter_class_overlap_weight: 0.1
shared_reward_weight: 0.5
class_union_cov_weight: 0.01
class_union_prec_weight: 0.01
global_coverage_weight: 0.01
global_coverage_threshold: 0.1
```

**Key Difference**: Multi-agent uses a higher shared reward weight and small class-union/global bonuses to encourage cooperation

---

## Reward Computation Flow

### Single-Agent Flow

1. **Compute Gains**:
   - `precision_gain = precision - prev_precision`
   - `coverage_gain = coverage - prev_coverage`

2. **Normalize & Scale**:
   - Normalize by previous values
   - Scale and clip gains

3. **Compute Bonuses**:
   - Coverage bonus (structured)
   - Target class bonus

4. **Compute Penalties**:
   - Overlap, drift, anchor drift, JS penalties
   - Apply progress factor

5. **Combine**:
   - `reward = gains + bonuses - penalties`

### Multi-Agent Flow

1. **For Each Agent**:
   - Compute local reward (same as single-agent)
   - Store in `reward_without_shared[agent]`

2. **After All Agents Act**:
   - Compute shared reward: `mean(reward_without_shared)`
   - Add to each agent's reward

3. **Optional Class/Global Rewards**:
   - Compute class union metrics
   - Add class union rewards if enabled
   - Add global coverage reward if enabled

---

## Reward Shaping Strategies

### 1. Progress-Based Penalty Reduction

**Strategy**: Reduce penalties when making progress

**Implementation**: `progress_factor` scales penalties

**Purpose**: Encourages exploration when improving

### 2. Coverage Floor Protection

**Strategy**: Revert actions that violate coverage floor

**Implementation**: Action reverted, penalties reduced by 90%

**Purpose**: Prevents boxes from collapsing

### 3. Structured Coverage Bonuses

**Strategy**: Different bonuses for different precision levels

**Implementation**: Multiple `coverage_bonus_weight_*` parameters

**Purpose**: Rewards coverage improvements at appropriate precision levels

### 4. Shared Rewards for Cooperation

**Strategy**: All agents receive average performance bonus

**Implementation**: `shared_reward_weight * mean(local_rewards)`

**Purpose**: Encourages multi-agent cooperation

---

## Reward Interpretation

### Positive Rewards

**Sources**:
- Precision/coverage gains
- Coverage bonuses
- Target class bonuses
- Shared rewards (multi-agent)

**Interpretation**: Agent is improving or performing well

### Negative Rewards

**Sources**:
- Precision/coverage losses
- Overlap penalties
- Drift penalties
- Coverage floor penalties

**Interpretation**: Agent is regressing or violating constraints

### Reward Magnitude

**Typical Range**: `[-1.0, 1.0]` (after clipping)

**High Rewards** (>0.5):
- Large improvements in precision/coverage
- Targets met with bonuses

**Low Rewards** (<-0.5):
- Large regressions
- Multiple penalties applied

---

## Reward Design Rationale

### 1. Gain-Based Rewards

**Rationale**: Rewards improvements, not absolute values

**Benefit**: Encourages continuous improvement

**Trade-off**: May cause instability if gains are noisy

### 2. Normalized Gains

**Rationale**: Makes rewards scale-invariant

**Benefit**: Works across different precision/coverage ranges

**Trade-off**: Small changes when values are small may be overemphasized

### 3. Clipped Gains

**Rationale**: Prevents reward explosions

**Benefit**: Stable training

**Trade-off**: May limit learning signal for large improvements

### 4. Progress Factor

**Rationale**: Reduces penalties when making progress

**Benefit**: Encourages exploration

**Trade-off**: May allow some constraint violations

### 5. Shared Rewards

**Rationale**: Encourages cooperation

**Benefit**: Aligns multi-agent objectives

**Trade-off**: May reduce individual agent incentives

---

## Code References

- **Single-Agent Reward**: `single_agent/single_agentENV.py` lines 682-836
- **Multi-Agent Reward**: `BenchMARL/environment.py` lines 1100-1200
- **Coverage Bonus**: `BenchMARL/environment.py` `_compute_coverage_bonus()`
- **Target Class Bonus**: `BenchMARL/environment.py` `_compute_target_class_bonus()`
- **Overlap Penalties**: `BenchMARL/environment.py` `_compute_inter_class_overlap_penalty()`, `_compute_same_class_overlap_penalty()`
