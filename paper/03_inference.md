# Inference Process

## Overview

Inference extracts interpretable anchor rules from trained policies. The process runs rollouts with trained policies to generate anchors and computes metrics for evaluation.

---

## Multi-Agent Inference

### Main Function: `extract_rules_from_policies()`

**Location**: `BenchMARL/inference.py` lines 823-2729

### Phase 1: Policy Loading & Setup

#### Step 1.1: Locate Policy Files (Lines 844-951)

**Process**:
1. **Check for `policies_index.json`** (preferred):
   - Loads structured index with agent-to-policy mappings
   - Extracts `agents_per_class` from index
   - Uses training seed from index (ensures reproducibility)

2. **Fallback: Flat Structure** (legacy):
   - Searches for `policy_*.pth` files
   - Less reliable for multi-agent scenarios

**Output**: `policy_files` dict mapping agent names → file paths

#### Step 1.2: Load Dataset & Classifier (Lines 957-982)

- Loads dataset with same seed as training
- Loads classifier from `classifier.pth`
- Required for computing precision/coverage

#### Step 1.3: Load Policy Models (Lines 1021-1247)

**Process**:

1. **Load Combined Policies**:
   - Calls `load_policy_model()` for each policy file
   - Infers MLP architecture from state_dict weights
   - Creates MLP with inferred dimensions
   - Loads weights

2. **Extract Individual Agent Policies**:
   - Detects agent-specific keys in state_dict
   - Patterns: `"0.mlp.params.0.weight"`, `"policy_nets.0.weight"`, etc.
   - Extracts per-agent state_dicts
   - Creates separate MLP for each agent

**Key Feature**: Architecture inferred from checkpoint, not config file

#### Step 1.4: Map Policies to Agents (Lines 1436-1554)

- Ensures each agent has a policy assigned
- Handles shared policies vs individual policies
- Maps group policies to agent names

### Phase 2: Instance-Based Rollouts

#### Step 2.1: Sample Instances Per Agent (Lines 1641-1696)

**Key Feature**: **Different agents get different instances**

**Process**:
1. Get class instances from the correct dataset:
   - If `coverage_on_all_data=True`, sample from combined train+test
   - Else if `eval_on_test_data=True`, sample from test; otherwise training
2. (Optional) If `filter_by_prediction=True`, keep only instances where classifier prediction matches target class
3. Extract agent index from agent name (`agent_0_1` → idx=1)
4. **If enough instances**: Divide into subsets, each agent gets its own
5. **Otherwise**: All agents share pool (with different seeds)

**Purpose**: Ensures diversity across agents of same class

**Default CLI behavior**: Prediction filtering is enabled unless `--no_filter_by_prediction` is passed (used in baseline comparisons).

#### Step 2.2: Run Rollout for Each Instance (Lines 1698-1813)

**For each sampled instance**:

1. **Create Environment**:
   ```python
   env = AnchorEnv(
       target_classes=[target_class],
       env_config=env_config
   )
   ```

2. **Set Anchor Instance**:
   ```python
   env.x_star_unit[agent_name] = x_instance.copy()
   ```
   - **CRITICAL**: Makes rollout instance-based (not class-based)

3. **Run Rollout**:
   ```python
   episode_data = run_rollout_with_policy(
       env=env,
       policy=policy,
       agent_id=agent_name,
       max_steps=steps_per_episode,
       seed=rollout_seed
   )
   ```

**Rollout Process** (`run_rollout_with_policy()`):
- Resets environment
- Runs policy for `max_steps` iterations
- Returns metrics and final state

#### Step 2.3: Extract Metrics (Lines 1815-1891)

**From `episode_data`**:
- `instance_precision`: Precision for this anchor
- `instance_coverage`: Overall coverage `P(x in box)`
- `instance_coverage_class_conditional`: `P(x in box | y = target_class)`
- `class_precision`, `class_coverage`: Union metrics (from environment)
- `rollout_time_seconds`: Time taken for rollout

#### Step 2.4: Extract Rule (Lines 1893-1964)

**Process**:
1. Get final observation: `[lower, upper, precision, coverage]`
2. Denormalize bounds: From `[0,1]` to standardized space
3. Create temporary environment for rule extraction
4. Call `env.extract_rule()`:
   - Compares final bounds to initial bounds
   - Identifies tightened features
   - Formats as: `"feature_name ∈ [lower, upper] and ..."`

**Example Rule**:
```
"age ∈ [25.0, 35.0] and income ∈ [50000.0, 75000.0]"
```

#### Step 2.5: Store Anchor Data (Lines 1966-2002)

**Stored per episode**:
- Instance-level metrics
- Class-level metrics
- Rule string
- Box bounds (normalized and denormalized)
- Box widths, volume

#### Step 2.6: Deduplicate & Rank Rules (Post-Processing)

**Purpose**: Reduce near-duplicate anchors and keep the most informative rules.

**Process**:
1. **Canonical key**: Quantize bounds to an epsilon grid (`epsilon = max(1e-3, min_width / 4)`), drop near-full-range features, and build a canonical key.
2. **Early duplicate suppression**: Skip rollouts with identical canonical keys or high IoU overlap (default IoU threshold = 0.9).
3. **Post-pass dedup**: Keep best anchor per canonical key (precision tie-breaks by coverage), then apply IoU-based NMS.
4. **Top-K selection** (optional): Keep the highest-scoring rules (`score = precision * (1 + coverage)`).

### Phase 3: Aggregate Metrics Per Agent

#### Step 3.1: Compute Averages (Lines 2010-2024)

- `instance_precision = mean(instance_precisions)`
- `instance_coverage = mean(instance_coverages)`
- `avg_rollout_time = mean(rollout_times)`

**Note**: Instance-level metrics come directly from rollout estimates (perturbation-based); class-union metrics later recompute precision on the actual dataset for filtering.

#### Step 3.2: Compute Per-Agent Union (Lines 2026-2069)

**Union of all anchors for THIS AGENT**:
- Builds union mask from all agent's anchors
- Computes union precision/coverage
- **Note**: This is per-agent union, not class-level union

### Phase 4: Aggregate Across Agents

#### Step 4.1: Combine Agent Results (Lines 2144-2220)

**When `agents_per_class > 1`**:
- Collects anchors from all agents
- Computes class-level union (union of ALL agents' anchors)
- Updates aggregated metrics

**Final Class Metrics**:
- `class_precision`: Union precision across all agents
- `class_coverage`: Union coverage across all agents
- `all_anchors`: All anchors from all agents

**Note**: These instance-based union metrics are intermediate; the final class-union metrics are recomputed later using class-based anchors only.

### Phase 5: Class-Based Rollouts

#### Step 5.1: Run Class-Based Rollouts (Lines 2273-2535)

**Purpose**: Generate anchors from class structure (not specific instances)

**Process**:
1. Create environment with `use_class_centroids=True`
2. **Do NOT set `x_star_unit`** (makes it class-based)
3. Run rollouts initialized from cluster centroids
4. Extract metrics and rules

**Key Difference**:
- Instance-based: `x_star_unit` set → overall coverage
- Class-based: `x_star_unit` not set → class-conditional coverage

### Phase 6: Final Union Metrics

#### Step 6.1: Recompute Union with Class-Based Anchors (Lines 2537-2611)

**Final Union Computation**:
- Collects **ONLY class-based anchors** (not instance-based)
- **Filters anchors by precision threshold** before computing union
- Combines filtered anchors into union
- Computes final union precision/coverage on actual dataset

**Precision Filtering Process**:
1. **Threshold Calculation**: `precision_threshold = precision_target * 0.8`
   - Default: `0.95 * 0.8 = 0.76`
   - Uses same threshold as training/inference

2. **Precision Recalculation**: For each anchor, recomputes precision on **actual dataset** (not perturbation samples)
   - Uses `X_data_union`/`y_data_union` from the same data source as the rollouts
     (full train+test when `coverage_on_all_data=True`, otherwise test/train based on `eval_on_test_data`)
   - Ensures filtering uses same data as final union computation
   - Avoids mismatch between stored `instance_precision` (from perturbation) and actual precision

3. **Filtering**: Only anchors with `actual_precision >= precision_threshold` are included in union

**Purpose**: Provides the smallest set of **high-quality general rules** that explain the class structure. Uses only class-based anchors (initialized from cluster centroids) to capture class-level patterns, excluding instance-specific details.

**Key Improvement**: The precision filtering ensures that only high-quality anchors are included in the union, preventing low-precision rules from degrading the overall class union precision.

---

## Single-Agent Inference

### Main Function: `extract_rules_single_agent()`

**Location**: `single_agent/single_agent_inference.py`

### Key Differences

1. **One Environment Per Class**: Creates separate environment for each class
2. **No Agent Aggregation**: No need to combine multiple agents
3. **Simpler Structure**: Direct rollout → metrics → rule extraction

### Process

1. **Load Model**: Loads SB3 model for each class
2. **Sample Instances**: Samples instances from the appropriate dataset (test/train/full based on flags)
3. **Run Rollouts**: Multiple rollouts per instance (instance-based)
4. **Extract Rules**: Same rule extraction process
5. **Aggregate**: Recompute precision/coverage on selected dataset, filter low-quality rollouts, then average

**Important Options**:
- `coverage_on_all_data`: Use combined train+test for coverage/precision recomputation
- `sample_from_full_dataset`: Draw instances from train+test pool
- `filter_by_prediction`: Keep only instances where classifier prediction matches target class (default True in CLI)
- `use_prediction_routing`: Route instances to policies using classifier predictions (realistic evaluation)
- `filter_low_quality_rollouts`: Drop low-precision/low-coverage rollouts before averaging
- `use_weighted_average`: Weight averages by support/coverage instead of simple mean

---

## Rollout Function Details

### `run_rollout_with_policy()` (Multi-Agent)

**Location**: `BenchMARL/inference.py` lines 324-820

**Process**:

1. **Reset Environment**:
   - Verifies `x_star_unit` is set (for instance-based)
   - Resets environment with seed
   - Checks initial box state

2. **Main Loop** (up to `max_steps`):
   - **Observe**: Get `[lower, upper, precision, coverage]`
   - **Act**: Policy outputs action
   - **Process Action**:
     - Handle different policy output formats (dict, tensor)
     - Extract action for specific agent (if multi-agent policy)
     - Apply action noise if enabled
   - **Step Environment**: Update box bounds
   - **Check Termination**: Early termination if targets met

3. **Extract Final Metrics**:
   - `instance_precision`, `instance_coverage`: From environment
   - `class_precision`, `class_coverage`: Union metrics
   - `rollout_time_seconds`: Duration

**Key Features**:
- Handles different policy output formats
- Supports exploration modes (sample, mean, noisy_mean)
- Action noise for diversity
- Verbose logging for debugging

---

## Rule Extraction Details

### `env.extract_rule()`

**Location**: `BenchMARL/environment.py` (method of AnchorEnv)

**Process**:

1. **Compare Bounds**:
   - Final bounds vs initial bounds
   - Identifies features that were tightened

2. **Format Rule**:
   - For each tightened feature: `"feature_name ∈ [lower, upper]"`
   - Combine with `" and "`
   - If no tightened features: `"any values (no tightened features)"`

3. **Denormalization**:
   - Converts from normalized `[0,1]` to standardized space
   - Uses `X_min`, `X_range` for conversion

**Example Output**:
```
"age ∈ [25.0, 35.0] and income ∈ [50000.0, 75000.0]"
```

---

## Inference Outputs

### Saved Files

1. **`extracted_rules.json`**:
   - All anchors with metrics
   - Rules per episode
   - Per-class and per-agent summaries

2. **Metrics Summary**:
   - Instance-level averages
   - Class-level union metrics
   - Class-based metrics
   - Rollout times

### Metrics Computed

**Per Episode**:
- Instance precision/coverage
- Class precision/coverage (union)
- Rule string
- Box bounds

**Per Agent**:
- Average instance metrics
- Per-agent union metrics
- Unique rules count

**Per Class**:
- Aggregated instance metrics
- Class-level union metrics (all agents)
- Total unique rules

---

## Key Design Decisions

### 1. Instance Sampling Strategy

- **Different agents get different instances** (when possible)
- Ensures diversity across agents
- Uses deterministic seeding for reproducibility

### 2. Architecture Inference

- **Inferred from checkpoint**, not config file
- Handles changes in config between training and inference
- Ensures compatibility

### 3. Union Metrics Computation

- **Per-agent union**: Union of one agent's anchors
- **Class-level union**: Union of ALL agents' anchors
- **Final union**: Uses class-based anchors only (after precision filtering)

### 4. Rule Extraction

- Uses temporary environment with same data source
- Denormalizes to standardized space
- Compares final to initial bounds

---

## Code References

- **Multi-Agent Inference**: `BenchMARL/inference.py` lines 823-2729
- **Single-Agent Inference**: `single_agent/single_agent_inference.py`
- **Rollout Function**: `BenchMARL/inference.py` lines 324-820
- **Rule Extraction**: `BenchMARL/environment.py` (extract_rule method)
