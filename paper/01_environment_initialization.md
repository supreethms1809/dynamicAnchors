# Environment Initialization

## Overview

The anchor generation environment (`AnchorEnv` for multi-agent, `SingleAgentAnchorEnv` for single-agent) is a reinforcement learning environment where agents learn to find interpretable anchor boxes (hyperrectangles) that explain classifier predictions.

---

## Multi-Agent Environment (`AnchorEnv`)

### Class: `AnchorEnv` (PettingZoo ParallelEnv)

**Location**: `BenchMARL/environment.py`

### Initialization Parameters

```python
AnchorEnv(
    X_unit: np.ndarray,           # Normalized features [0, 1]
    X_std: np.ndarray,            # Standardized features (mean=0, std=1)
    y: np.ndarray,                # Class labels
    feature_names: List[str],     # Feature names
    classifier: nn.Module,         # Trained classifier
    device: str = "cpu",
    target_class: Optional[int] = None,
    target_classes: Optional[List[int]] = None,
    env_config: Dict[str, Any]    # Environment configuration
)
```

### Key Initialization Steps

#### 1. **Data Normalization** (Lines 45-58)

- **If `normalize_data=True`**: Normalizes `X_std` to `[0, 1]` range
  - Computes `X_min`, `X_max`, `X_range`
  - Formula: `X_unit = (X_std - X_min) / X_range`
- **If `normalize_data=False`**: Uses provided `X_unit` directly (already normalized)

**Purpose**: Ensures features are in `[0, 1]` range for consistent box bounds

#### 2. **Agent Creation** (Lines 76-94)

**Agent Naming Convention**:
- **`agents_per_class == 1`**: `agent_0`, `agent_1`, ..., `agent_{class}`
- **`agents_per_class > 1`**: `agent_0_0`, `agent_0_1`, ..., `agent_{class}_{agent_idx}`

**Mappings Created**:
- `self.possible_agents`: List of all agent names
- `self.agent_to_class`: Maps agent name → target class
- `self.class_to_agents`: Maps class → list of agent names

**Example**:
```python
# For 2 classes, 3 agents per class:
possible_agents = ['agent_0_0', 'agent_0_1', 'agent_0_2', 
                   'agent_1_0', 'agent_1_1', 'agent_1_2']
agent_to_class = {'agent_0_0': 0, 'agent_0_1': 0, ..., 'agent_1_2': 1}
class_to_agents = {0: ['agent_0_0', 'agent_0_1', 'agent_0_2'],
                   1: ['agent_1_0', 'agent_1_1', 'agent_1_2']}
```

#### 3. **Group Mapping** (Line 106)

- Maps group names to agent lists for BenchMARL compatibility
- Default: Each agent is its own group

#### 4. **Reward Parameters** (Lines 108-128)

**Core Reward Weights**:
- `alpha` (default: 0.7): Weight for precision gain component
- `beta` (default: 0.6): Weight for coverage gain component
- `gamma` (default: 0.1): Weight for overlap penalty

**Target Thresholds**:
- `precision_target` (default: 0.95): Target precision threshold
- `coverage_target` (default: 0.5 multi-agent, 0.3 single-agent): Target coverage threshold
- `precision_blend_lambda` (default: 0.5): Blends hard precision with avg probability

**Penalty Weights**:
- `drift_penalty_weight` (default: 0.05): Penalty for box movement
- `js_penalty_weight` (default: 0.05): Jaccard-like penalty for volume changes

#### 5. **Action Space Configuration** (Lines 108-112)

**Step Fractions** (`step_fracs`):
- Default: `[0.005, 0.01, 0.02]`
- Defines relative step sizes for box adjustments
- Used in discrete action mode (currently using continuous)

**Min Width** (`min_width`):
- Default: 0.05
- Minimum allowed box width per feature
- Prevents boxes from collapsing to zero width

#### 6. **Perturbation Settings** (Lines 139-150)

**Perturbation Modes**:
- `bootstrap`: Sample with replacement from covered points
- `uniform`: Uniform random sampling
- `adaptive`: Bootstrap if enough covered points, otherwise uniform

**Parameters**:
- `use_perturbation` (default: True): Enable perturbation sampling
- `perturbation_mode` (default: "adaptive"): Sampling strategy
- `n_perturb` (default: 4096): Number of perturbed samples per step

**Purpose**: Efficiently estimate precision/coverage without evaluating all data points

#### 7. **Coverage Constraints** (Lines 156-158)

- `min_coverage_floor` (default: 0.005, dynamically overridden during training/inference)
- During training/inference, `min_coverage_floor` is set to `1 / n_samples` to ensure the box covers at least one sample (the anchor instance)
- If coverage drops below floor, action is reverted

#### 8. **Initialization Strategy** (Lines 159-162)

**Instance Selection**:
- `fixed_instances_per_class`: Pre-computed instances per class
- `cluster_centroids_per_class`: K-means cluster centroids
- `use_class_centroids` (default: True): Use centroids for initialization
- `use_random_sampling` (default: False): Random instance selection

**Purpose**: Determines starting point for each episode

#### 9. **Multi-Agent Specific Settings** (Lines 191-204)

**Cooperation Rewards**:
- `shared_reward_weight` (default: 0.5): Weight for shared reward component
- `inter_class_overlap_weight` (default: 0.1): Penalty for overlap between different classes
- `same_class_diversity_weight` (default: 0.0): Encourages diversity within same class

**Class-Level Rewards**:
- `class_union_cov_weight` (default: 0.01): Reward for class union coverage
- `class_union_prec_weight` (default: 0.01): Reward for class union precision
- `global_coverage_weight` (default: 0.01): Reward for global dataset coverage

#### 10. **Coverage Bonus Weights** (Lines 207-215)

**Structured Rewards for Coverage**:
- `coverage_bonus_weight_met`: When targets are met
- `coverage_bonus_weight_high_prec`: When precision ≥ threshold
- `coverage_bonus_weight_high_prec_progress`: Progress multiplier
- `coverage_bonus_weight_high_prec_distance`: Distance to target multiplier
- `coverage_bonus_weight_reasonable_prec`: When precision ≥ 0.8 * threshold
- `coverage_bonus_weight_reasonable_prec_progress`: Progress multiplier
- `target_class_bonus_weight`: Bonus for target class fraction

#### 11. **State Initialization** (Lines 217-227)

**Box State**:
- `self.lower`: Dict mapping agent → lower bounds (normalized [0,1])
- `self.upper`: Dict mapping agent → upper bounds (normalized [0,1])
- `self.x_star_unit`: Dict mapping agent → anchor instance (if instance-based)
- `self.prev_lower`, `self.prev_upper`: Previous state for drift computation
- `self.box_history`: List of box states per agent (for tracking)

**Termination Counters**:
- `self.termination_reason_max_counts`: Limits per termination reason
- Prevents overuse of early termination conditions

#### 12. **Test Data Support** (Lines 164-183)

**Evaluation Mode**:
- `eval_on_test_data` (default: False in multi-agent config, True in single-agent config): Use test data for metrics
- If True: Requires `X_test_unit`, `X_test_std`, `y_test`
- Metrics computed on test set instead of training set (training envs still force train-only to avoid leakage)

**Purpose**: Enables evaluation on held-out test data

---

## Single-Agent Environment (`SingleAgentAnchorEnv`)

### Class: `SingleAgentAnchorEnv` (Gymnasium Env)

**Location**: `single_agent/single_agentENV.py`

### Key Differences from Multi-Agent

1. **Single Agent**: Only one agent per environment instance
2. **Single Target Class**: `target_class` instead of `target_classes`
3. **Direct Variables**: `self.lower`, `self.upper` (not dictionaries)
4. **No Multi-Agent Rewards**: No shared rewards, inter-class penalties, etc.
5. **Gymnasium Interface**: Compatible with Stable-Baselines3

### Observation Space

```python
observation_space = Box(
    low=-inf, high=inf,
    shape=(2 * n_features + 2,),  # [lower (n), upper (n), precision, coverage]
    dtype=np.float32
)
```

**Components**:
- First `n_features`: Lower bounds for each feature
- Next `n_features`: Upper bounds for each feature
- Next 1: Current precision
- Next 1: Current coverage

### Action Space

```python
action_space = Box(
    low=-1.0, high=1.0,
    shape=(2 * n_features,),  # [delta_lower (n), delta_upper (n)]
    dtype=np.float32
)
```

**Components**:
- First `n_features`: Deltas for lower bounds
- Next `n_features`: Deltas for upper bounds
- Actions are clipped to `[-1, 1]` and scaled by `max_action_scale`

---

## Reset Process

### Multi-Agent Reset (`AnchorEnv.reset()`)

**Steps**:

1. **Select Initial Instance** (Lines 705-762):
   - **If `x_star_unit[agent]` is set**: Use that instance (instance-based mode)
   - **If `cluster_centroids_per_class` available**: Select centroid for agent's class
   - **Otherwise**: Use mean of class or random instance

2. **Initialize Box Bounds**:
   - Center: Selected instance or centroid
   - Width: `initial_window` (default: 0.1) on each side
   - Formula: `lower = clip(center - initial_window, 0, 1)`
   - Formula: `upper = clip(center + initial_window, 0, 1)`

3. **Reset State**:
   - Clear `box_history`
   - Reset `prev_lower`, `prev_upper`
   - Termination counters persist across episodes; they are initialized at env creation and can be reset explicitly (e.g., SB3 eval callback)

4. **Compute Initial Metrics**:
   - Precision, coverage, details from `_current_metrics()`

5. **Return Observation**:
   - Concatenate: `[lower, upper, precision, coverage]`

### Single-Agent Reset (`SingleAgentAnchorEnv.reset()`)

**Similar process** but:
- Single agent (no loop)
- Direct variable assignment (not dictionary)
- Returns single observation (not dict)

---

## Key Design Decisions

### 1. **Dual Normalization**

- **`X_unit`**: Normalized to `[0, 1]` for box bounds
- **`X_std`**: Standardized (mean=0, std=1) for classifier input
- **Purpose**: Box bounds in `[0,1]` space, classifier uses standardized features

### 2. **Instance-Based vs Class-Based Modes**

- **Instance-based**: `x_star_unit` set → coverage = `P(x in box)` (overall)
- **Class-based**: `x_star_unit` not set → coverage = `P(x in box | y = target)` (class-conditional)
  - During training, `training_instance_ratio` controls the mix of instance-based vs centroid-based episodes (defaults to 1.0 in current configs)

### 3. **Multi-Agent Architecture**

- **`agents_per_class == 1`**: One agent per class (standard setup)
- **`agents_per_class > 1`**: Multiple agents per class (cooperation/diversity)
- Each agent learns its own policy but shares class-level rewards

### 4. **Early Termination**

- **Stabilization-based**: Terminates when metrics stabilize
- **Target-based**: Terminates when targets are met
- **Counters**: Prevent overuse of termination conditions

---

## Configuration File Structure

**Example**: `BenchMARL/conf/anchor.yaml`

```yaml
env_config:
  agents_per_class: 3
  precision_target: 0.95
  coverage_target: 0.5
  use_perturbation: True
  perturbation_mode: "adaptive"
  n_perturb: 4096
  step_fracs: [0.005, 0.01, 0.02]
  min_width: 0.05
  alpha: 0.7
  beta: 0.6
  gamma: 0.1
  max_cycles: 500
  initial_window: 0.1
  max_action_scale: 0.1
  # ... more parameters
```

---

## Code References

- **Multi-Agent Init**: `BenchMARL/environment.py` lines 28-227
- **Single-Agent Init**: `single_agent/single_agentENV.py` lines 46-215
- **Reset**: `BenchMARL/environment.py` lines 705-762
- **Config Loading**: `BenchMARL/anchor_trainer.py` lines 200-260
