# Dynamic Anchors Paper Documentation Index

This directory contains comprehensive documentation to help write a research paper on the Dynamic Anchors system. The documentation covers both single-agent and multi-agent implementations.

---

## Core Components

### 1. [Environment Initialization](01_environment_initialization.md)
- Multi-agent (`AnchorEnv`) and single-agent (`SingleAgentAnchorEnv`) initialization
- Data normalization and standardization
- Agent creation and mapping
- Configuration parameters
- Reset process

### 2. [Training Process](02_training.md)
- Multi-agent training (BenchMARL + MADDPG/MASAC)
- Single-agent training (Stable-Baselines3 + DDPG/SAC)
- Training setup and execution
- Policy updates and metrics collection
- Training outputs and challenges

### 3. [Inference Process](03_inference.md)
- Policy loading and extraction
- Instance-based rollouts
- Class-based rollouts
- Rule extraction
- Metric aggregation

### 4. [Metrics Definitions](04_metrics_definitions.md)
- Precision metrics (hard precision, precision proxy, instance-level, class-level)
- Coverage metrics (instance-based, class-based, union)
- Standard deviation metrics
- Rule metrics
- Timing metrics

### 5. [Feature Importance](05_feature_importance.md)
- Rule-based feature importance computation
- Frequency and interval selectivity
- Per-class vs global importance
- Visualization
- Comparison with baseline methods

### 6. [Rule Overlap Metrics](06_rule_overlap_metrics.md)
- Interval-based overlap computation
- Overlap matrix and statistics
- Diversity assessment
- Redundancy detection

### 7. [Reward Structure](07_reward_structure.md)
- Single-agent reward components
- Multi-agent reward components (local + shared)
- Reward weights and configuration
- Reward computation flow
- Reward shaping strategies

---

## Advanced Topics

### 8. [NashConv and Exploitability](08_nashconv_exploitability.md)
- Nash Equilibrium concepts
- Exploitability computation
- NashConv sum and max
- Epsilon-Nash Equilibrium
- Convergence interpretation

### 9. [Network Architectures](09_network_architectures.md)
- Policy networks (MLP)
- Critic networks (centralized)
- Classifier networks
- Architecture configuration
- Weight initialization and regularization

### 10. [Data Preprocessing](10_data_preprocessing.md)
- Dataset loading
- Train/test splitting
- Standardization (for classifier)
- Normalization (for anchor boxes)
- Dual representation

### 11. [Termination Conditions](11_termination_conditions.md)
- Target-based termination
- Stabilization-based termination
- Maximum steps termination
- Termination reason tracking
- Configuration

### 12. [Action Space](12_action_space.md)
- Continuous action representation
- Action application process
- Action constraints
- Scaling parameters
- Action noise

### 13. [Perturbation Sampling](13_perturbation_sampling.md)
- Bootstrap sampling
- Uniform sampling
- Adaptive sampling
- Sampling parameters
- Precision/coverage estimation

### 14. [Baseline Methods](14_baseline_methods.md)
- Static Anchors
- LIME
- SHAP
- Feature Importance
- Comparison methodology

### 15. [Environment Dynamics](15_environment_dynamics.md)
- Episode flow and step-by-step execution
- Metric computation process
- Box update mechanics
- State representation
- Multi-agent coordination
- Termination conditions

---

## Quick Reference

### For Writing Methods Section

1. **Environment**: See [01_environment_initialization.md](01_environment_initialization.md) and [15_environment_dynamics.md](15_environment_dynamics.md)
2. **Training**: See [02_training.md](02_training.md)
3. **Inference**: See [03_inference.md](03_inference.md)
4. **Rewards**: See [07_reward_structure.md](07_reward_structure.md)
5. **Environment Execution**: See [15_environment_dynamics.md](15_environment_dynamics.md) for step-by-step flow

### For Writing Metrics Section

1. **Definitions**: See [04_metrics_definitions.md](04_metrics_definitions.md)
2. **Feature Importance**: See [05_feature_importance.md](05_feature_importance.md)
3. **Rule Overlap**: See [06_rule_overlap_metrics.md](06_rule_overlap_metrics.md)
4. **NashConv**: See [08_nashconv_exploitability.md](08_nashconv_exploitability.md)

### For Writing Experimental Setup

1. **Data Preprocessing**: See [10_data_preprocessing.md](10_data_preprocessing.md)
2. **Network Architectures**: See [09_network_architectures.md](09_network_architectures.md)
3. **Baseline Methods**: See [14_baseline_methods.md](14_baseline_methods.md)
4. **Termination**: See [11_termination_conditions.md](11_termination_conditions.md)

### For Writing Implementation Details

1. **Action Space**: See [12_action_space.md](12_action_space.md)
2. **Perturbation**: See [13_perturbation_sampling.md](13_perturbation_sampling.md)
3. **Environment Execution**: See [15_environment_dynamics.md](15_environment_dynamics.md)
4. **Training**: See [02_training.md](02_training.md)
5. **Inference**: See [03_inference.md](03_inference.md)

---

## Code Locations

### Main Files

- **Multi-Agent Environment**: `BenchMARL/environment.py`
- **Single-Agent Environment**: `single_agent/single_agentENV.py`
- **Multi-Agent Training**: `BenchMARL/anchor_trainer.py`
- **Single-Agent Training**: `single_agent/anchor_trainer_sb3.py`
- **Multi-Agent Inference**: `BenchMARL/inference.py`
- **Single-Agent Inference**: `single_agent/single_agent_inference.py`
- **Baselines**: `baseline/establish_baseline.py`
- **Comparison**: `plot_comparison.py`

### Configuration Files

- **Multi-Agent Config**: `BenchMARL/conf/anchor.yaml`
- **Single-Agent Config**: `single_agent/conf/anchor_single.yaml`
- **MLP Config**: `BenchMARL/conf/mlp.yaml`

---

## Key Concepts

### Multi-Agent vs Single-Agent

| Aspect | Multi-Agent | Single-Agent |
|--------|-------------|--------------|
| **Framework** | BenchMARL | Stable-Baselines3 |
| **Algorithm** | MADDPG/MASAC | DDPG/SAC |
| **Rewards** | Local + Shared | Local only |
| **NashConv** | Computed | Not computed |
| **Cooperation** | Centralized critic | Independent |

### Instance-Level vs Class-Level

| Aspect | Instance-Level | Class-Level |
|--------|----------------|-------------|
| **What** | Average of individual anchors | Union of all anchors |
| **Computation** | Mean across episodes | Union (OR) operation |
| **Use Case** | Compare with baselines | Measure complete explanation |

### Precision vs Coverage

| Metric | What It Measures | Formula |
|--------|------------------|---------|
| **Precision** | Accuracy: Are samples correct? | `P(y = target \| x in box)` |
| **Coverage** | Completeness: How much covered? | `P(x in box \| y = target)` |

---

## Paper Structure Suggestions

### Abstract
- Brief overview of Dynamic Anchors
- Key contributions (multi-agent, class-level optimization)
- Main results

### Introduction
- Problem: Interpretability for ML models
- Related work: Anchor, LIME, SHAP
- Contribution: RL-based anchor generation

### Methods
1. **Environment Design**: [01_environment_initialization.md](01_environment_initialization.md)
2. **Reward Structure**: [07_reward_structure.md](07_reward_structure.md)
3. **Training**: [02_training.md](02_training.md)
4. **Inference**: [03_inference.md](03_inference.md)

### Metrics
1. **Definitions**: [04_metrics_definitions.md](04_metrics_definitions.md)
2. **Feature Importance**: [05_feature_importance.md](05_feature_importance.md)
3. **Rule Overlap**: [06_rule_overlap_metrics.md](06_rule_overlap_metrics.md)
4. **NashConv**: [08_nashconv_exploitability.md](08_nashconv_exploitability.md)

### Experimental Setup
1. **Datasets**: [10_data_preprocessing.md](10_data_preprocessing.md)
2. **Baselines**: [14_baseline_methods.md](14_baseline_methods.md)
3. **Architectures**: [09_network_architectures.md](09_network_architectures.md)
4. **Hyperparameters**: Config files

### Results
- Precision/coverage comparisons
- Feature importance comparisons
- Rule overlap analysis
- NashConv convergence

### Discussion
- Advantages of Dynamic Anchors
- Multi-agent vs single-agent
- Limitations and future work

---

## Additional Resources

### Existing Documentation

- `docs/PRECISION_COVERAGE_METRICS.md`: Detailed metric explanations
- `docs/METRICS_EXPLANATION.md`: Metric overview
- `docs/NASHCONV_INTERPRETATION.md`: NashConv guide
- `BenchMARL/docs/MULTI_AGENT_INFERENCE_REVIEW.md`: Inference review

### Configuration Examples

- `BenchMARL/conf/anchor.yaml`: Multi-agent config
- `single_agent/conf/anchor_single.yaml`: Single-agent config
- `BenchMARL/conf/mlp.yaml`: Network architecture

---

## Notes

- All documentation includes code references for verification
- Mathematical formulations provided where applicable
- Both single-agent and multi-agent covered
- Comparison with baselines included
- Practical guidelines for paper writing included

---

**Last Updated**: Documentation created for paper writing support

