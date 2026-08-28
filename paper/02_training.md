# Training Process

## Overview

Training uses reinforcement learning to learn policies that generate interpretable anchor boxes. The system supports both single-agent (Stable-Baselines3) and multi-agent (BenchMARL) training.

---

## Multi-Agent Training

### Architecture: BenchMARL + MADDPG/MASAC

**Location**: `BenchMARL/anchor_trainer.py`

### Training Setup (`setup_experiment()`)

#### Step 1: Load Configuration (Lines 200-260)

1. **Load Environment Config**:
   - Reads `anchor.yaml` config file
   - Extracts `env_config` dictionary
   - Sets default values for missing parameters

2. **Compute Cluster Centroids**:
   - Uses k-means clustering per class
   - Number of centroids: `agents_per_class * 10` (if >1 agent) or `10` (if 1 agent)
   - Purpose: Provides diverse starting points for episodes
   - Stored in: `env_config["cluster_centroids_per_class"]`

3. **Prepare Environment Data**:
   - Gets normalized/standardized features from dataset loader
   - Sets `X_min`, `X_range` for denormalization
   - Configures test data if `eval_on_test_data=True`

#### Step 2: Create Task (Lines 261-274)

```python
anchor_config = {
    "X_unit": env_data["X_unit"],
    "X_std": env_data["X_std"],
    "y": env_data["y"],
    "feature_names": env_data["feature_names"],
    "classifier": self.dataset_loader.get_classifier(),
    "device": device,
    "target_classes": target_classes,
    "env_config": env_config_with_data,
}

self.task = AnchorTask.ANCHOR.get_task(config=anchor_config)
```

**Purpose**: Creates BenchMARL task wrapper around `AnchorEnv`

#### Step 3: Setup Callback (Lines 276-282)

```python
self.callback = AnchorMetricsCallback(
    log_training_metrics=True,
    log_evaluation_metrics=True,
    save_to_file=True,
    collect_anchor_data=True,
    compute_nashconv=True,  # Compute NashConv metrics
    nashconv_compute_frequency=10  # Every 10 batches
)
```

**Callback Functions**:
- `on_batch_collected()`: Logs metrics, computes NashConv
- `on_evaluation_end()`: Logs evaluation metrics
- Saves training/evaluation history to JSON files

#### Step 4: Create Experiment (Lines 283-291)

```python
self.experiment = Experiment(
    config=self.experiment_config,
    task=self.task,
    algorithm_config=self.algorithm_config,  # MADDPG or MASAC
    model_config=self.model_config,  # MLP architecture
    critic_model_config=self.critic_model_config,
    seed=self.seed,
    callbacks=[self.callback]
)
```

**Components**:
- **Algorithm**: MADDPG (Multi-Agent DDPG) or MASAC (Multi-Agent SAC)
- **Model**: MLP (Multi-Layer Perceptron) for actor/critic
- **Critic**: Centralized critic (observes all agents' states)

### Training Execution (`train()`)

#### Step 1: Run Experiment (Line 317)

```python
self.experiment.run()
```

**What Happens**:
1. **Environment Reset**: Each episode starts with reset
2. **Agent Actions**: All agents act in parallel
3. **Reward Computation**: Local + shared rewards computed
4. **Policy Update**: Actor/critic networks updated via MADDPG/MASAC
5. **Metrics Logging**: Callback logs metrics every batch

#### Step 2: Policy Updates

**MADDPG Algorithm**:
- **Actor**: Policy network for each agent
- **Critic**: Centralized Q-function (observes all agents)
- **Update**: DDPG-style updates with multi-agent extensions

**Key Features**:
- Centralized training, decentralized execution
- Agents can observe other agents during training
- Policies are independent at inference time

#### Step 3: Metrics Collection

**Training Metrics** (logged every batch):
- `anchor_precision`, `anchor_coverage`: Per-agent metrics
- `class_union_precision`, `class_union_coverage`: Class-level union metrics
- `nashconv_sum`, `exploitability_max`: Nash equilibrium convergence
- `total_reward`: Sum of rewards per episode

**Evaluation Metrics** (logged periodically):
- Same as training metrics but on evaluation episodes
- Computed from separate evaluation rollouts

### Training Loop Details

#### Episode Structure

1. **Reset Environment**:
   - Selects initial instance/centroid per agent
   - Initializes box bounds around anchor point
   - Sets `x_star_unit` for instance-based mode
   - During training, instance vs centroid initialization is randomized using `training_instance_ratio` (optionally adaptive per class) and instance samples are filtered so the classifier prediction matches the target class

2. **Step Loop** (up to `max_cycles`):
   - **Observe**: Agent receives `[lower, upper, precision, coverage]`
   - **Act**: Policy outputs action `[delta_lower, delta_upper]`
   - **Apply Action**: Updates box bounds
   - **Compute Reward**: Local + shared components
   - **Check Termination**: Early termination if targets met or stabilized

3. **Episode End**:
   - Metrics logged to callback
   - Policy updated via algorithm
   - Next episode begins

#### Reward Computation (During Step)

**Local Reward** (`R_local`):
```python
R_local = (
    alpha * precision_weight * precision_gain +
    beta * coverage_weight * coverage_gain +
    coverage_bonus +
    target_class_bonus -
    progress_factor * (overlap_penalty + drift_penalty + ...)
)
```

**Shared Reward** (`R_shared`):
```python
R_shared = shared_reward_weight * mean(R_local across all agents)
```

**Final Reward**:
```python
R_final = R_local + R_shared
```

**Purpose**: Encourages both individual performance and cooperation

---

## Single-Agent Training

### Architecture: Stable-Baselines3 + DDPG/SAC

**Location**: `single_agent/anchor_trainer_sb3.py`

### Training Setup

#### Step 1: Create Environment Per Class

```python
for target_class in target_classes:
    env = SingleAgentAnchorEnv(
        X_unit=X_unit,
        X_std=X_std,
        y=y,
        feature_names=feature_names,
        classifier=classifier,
        target_class=target_class,
        env_config=env_config
    )
```

**Key Difference**: One environment per class (independent training)

#### Step 2: Create Model

```python
if algorithm == "ddpg":
    model = DDPG("MlpPolicy", env, ...)
elif algorithm == "sac":
    model = SAC("MlpPolicy", env, ...)
```

**Algorithms**:
- **DDPG**: Deep Deterministic Policy Gradient
- **SAC**: Soft Actor-Critic

#### Step 3: Train

```python
model.learn(total_timesteps=total_timesteps)
```

**Training Process**:
- Standard single-agent RL loop
- No shared rewards or multi-agent coordination
- Each class trained independently

---

## Key Training Differences

### Multi-Agent vs Single-Agent

| Aspect | Multi-Agent | Single-Agent |
|--------|-------------|--------------|
| **Framework** | BenchMARL | Stable-Baselines3 |
| **Algorithm** | MADDPG/MASAC | DDPG/SAC |
| **Environment** | One env with all agents | One env per class |
| **Rewards** | Local + Shared | Local only |
| **Coordination** | Centralized critic | Independent |
| **NashConv** | Computed | Not computed |

### Training Hyperparameters

**Common Parameters**:
- `max_cycles`: Max steps per episode (default: 500)
- `precision_target`: Target precision (default: 0.95)
- `coverage_target`: Target coverage (default: 0.5 multi-agent, 0.3 single-agent)
- `alpha`, `beta`, `gamma`: Reward weights

**Multi-Agent Specific**:
- `agents_per_class`: Number of agents per class (default: 1-4)
- `shared_reward_weight`: Cooperation weight (default: 0.5)
- `nashconv_compute_frequency`: How often to compute NashConv (default: 10)

---

## Training Outputs

### Saved Files

1. **Policy Models**:
   - `checkpoints/`: Periodic checkpoints during training
   - `individual_models/`: Extracted per-agent policies (post-training)

2. **Metrics Files**:
   - `training_history.json`: Training metrics over time
   - `evaluation_history.json`: Evaluation metrics
   - `evaluation_anchor_data.json`: Anchor data from evaluations

3. **Configuration**:
   - `config.pkl`: Experiment configuration
   - `classifier.pth`: Trained classifier (if saved)

### Metrics Logged

**Per Batch**:
- Precision, coverage (per agent)
- Class union metrics
- NashConv (periodically)
- Rewards, losses

**Per Evaluation**:
- Evaluation precision/coverage
- Evaluation NashConv
- Anchor data (bounds, metrics)

---

## Training Challenges & Solutions

### Challenge 1: Coverage vs Precision Trade-off

**Problem**: Agents may optimize precision at expense of coverage

**Solution**:
- `beta` weight set to 0.6 to balance coverage without over-expansion
- Coverage bonuses tuned for gradual, stable expansion
- `coverage_target` aligned with configs (0.5 multi-agent, 0.3 single-agent)

### Challenge 2: Multi-Agent Coordination

**Problem**: Agents may interfere with each other

**Solution**:
- Shared rewards encourage cooperation
- Inter-class overlap penalties prevent conflicts
- Class union rewards align objectives

### Challenge 3: Early Termination Overuse

**Problem**: Agents may terminate too early

**Solution**:
- Termination reason counters limit usage
- Multi-agent: stabilization requirements (20 stable steps) and `stability_min_steps=50`
- Single-agent: no stabilization termination; minimum steps before termination is 2

### Challenge 4: Box Collapse

**Problem**: Boxes may shrink to zero width

**Solution**:
- `min_coverage_floor` prevents coverage from dropping too low
- `min_coverage_floor` is dynamically set to `1 / n_samples` during training to ensure the box covers at least one point
- Action reversion if coverage violates floor
- `min_width` constraint on box dimensions

---

## Code References

- **Multi-Agent Setup**: `BenchMARL/anchor_trainer.py` lines 200-305
- **Multi-Agent Train**: `BenchMARL/anchor_trainer.py` lines 307-373
- **Single-Agent Train**: `single_agent/anchor_trainer_sb3.py`
- **Reward Computation**: `BenchMARL/environment.py` lines 1100-1113
- **Callback**: `BenchMARL/benchmarl_wrappers.py` lines 136-1877
