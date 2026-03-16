# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Dynamic Anchors** is a research project that extracts explainable decision rules ("anchors") from tabular datasets using reinforcement learning — both multi-agent (BenchMARL/MARL) and single-agent (Stable-Baselines3) approaches. The two pipelines are designed for direct comparison under the same reward structure.

## Setup

```bash
conda create -n dynamic-anchors python=3.12
conda activate dynamic-anchors
pip install -r BenchMARL/requirements.txt
```

## Common Commands

### Multi-Agent Training (BenchMARL)
```bash
cd BenchMARL
python driver.py --dataset breast_cancer --algorithm maddpg --seed 42
# Algorithms: maddpg, masac
# Datasets: breast_cancer, wine, iris, synthetic, moons, circles, covtype, housing
#           + optional UCIML (uci_adult, uci_car, ...) and Folktables datasets
```

### Multi-Agent Inference & Evaluation
```bash
cd BenchMARL
python inference.py --experiment_dir <path_to_experiment_folder> --dataset breast_cancer
python test_extracted_rules.py --rules_file <path_to_extracted_rules.json> --dataset breast_cancer
```

### Single-Agent Training (SB3)
```bash
python single_agent/driver.py --dataset breast_cancer --algorithm ddpg --seed 42
# Algorithms: ddpg, sac
```

### Single-Agent Inference & Evaluation
```bash
python single_agent/single_agent_inference.py --experiment_dir <path> --dataset breast_cancer
cd single_agent
python test_extracted_rules_single.py --rules_file <path_to_extracted_rules.json> --dataset breast_cancer
```

### Batch Experiments
```bash
python run_batch_comparisons.py --dataset breast_cancer --algorithm maddpg
python run_comparison_pipeline.py --dataset wine --algorithm masac
python BenchMARL/run_production_experiments.py --datasets breast_cancer wine --algorithms maddpg
```

### Tests
```bash
cd tests
python test_multi_agent.py
python test_single_agent.py
```

## Architecture

### Dual Pipeline Design

Both pipelines share the same reward structure and dataset handling to ensure fair comparison:

| Component | Multi-Agent (BenchMARL/) | Single-Agent (single_agent/) |
|---|---|---|
| Entry point | `driver.py` | `driver.py` |
| Trainer | `anchor_trainer.py` | `anchor_trainer_sb3.py` |
| Environment | `environment.py` (PettingZoo) | `single_agentENV.py` (Gymnasium) |
| Inference | `inference.py` | `single_agent_inference.py` |
| Rule evaluation | `test_extracted_rules.py` | `test_extracted_rules_single.py` |
| Algorithm | MADDPG / MASAC | DDPG / SAC |
| Agents per class | Multiple (configurable) | One per class |

### Data Pipeline (`BenchMARL/tabular_datasets.py`)

1. Load dataset → train/test split (80/20)
2. `StandardScaler` normalization for classification
3. Min-max unit normalization to [0, 1] for RL feature space
4. Train classifier: `dnn`, `random_forest`, or `gradient_boosting`
5. Expose `get_anchor_env_data()` for the RL environment

### RL Environment Design

Agents learn to expand axis-aligned boxes (anchor rules) in the unit-normalized feature space:
- **Action space:** δ adjustments to box bounds (continuous, [-1, 1] per feature bound)
- **Observation space:** lower bounds, upper bounds, current precision, current coverage
- **Reward:** Weighted sum of precision gain (α), coverage gain (β), overlap penalty (γ), JS-divergence penalty, and progressive coverage bonuses

Multi-agent setting: multiple agents per class compete/cooperate to find diverse, non-overlapping rules. Convergence uses NashConv threshold (0.01).

### Configuration (YAML)

All key hyperparameters are in YAML configs — do not hardcode them:
- `BenchMARL/conf/base_experiment.yaml` — training loop parameters (lr, batch size, frames, checkpointing, device)
- `BenchMARL/conf/anchor.yaml` — environment and reward parameters (`precision_target`, `coverage_target`, `alpha`, `beta`, `gamma`, `agents_per_class`, etc.)
- `BenchMARL/conf/maddpg.yaml` / `masac.yaml` — algorithm-specific settings
- `BenchMARL/conf/mlp.yaml` — network architecture
- `single_agent/conf/anchor_single.yaml` — single-agent equivalent (multi-agent fields set to 0)

### Shared Utilities (`utils/`)
- `device_utils.py` — auto-selects CUDA > MPS (Apple Silicon) > CPU
- `networks.py` — classifier and RL policy network architectures
- `clusters.py` — clustering utilities for rule analysis
- `multiagent_networks.py` — multi-agent specific network variants

### Output Structure

```
BenchMARL/output/{dataset}_{algorithm}/training/
    checkpoint.pt, individual_models/, classifier.pth, training_history.json

output/single_agent_sb3_{dataset}_{algorithm}/training/
    SB3 checkpoints, classifier.pth, TensorBoard logs

experiment_folder/inference/
    extracted_rules.json, evaluation_metrics.json, test_results.json
```

### Baseline (`baseline/`)

`establish_baseline.py` implements the traditional Anchor (Ribeiro et al.) baseline for comparison. `generate_anchor_for_instance.py` generates per-instance baseline rules.

## Key Design Decisions

- **Two normalizations:** StandardScaler for the classifier; min-max [0,1] for the RL agent. Don't conflate them.
- **Fair comparison:** Single-agent pipeline uses identical reward weights to multi-agent so results are directly comparable.
- **Individual models:** After multi-agent training, `AnchorTrainer` extracts individual per-agent policies from the BenchMARL checkpoint for standalone inference.
- **Perturbation-based sampling:** Environments use adaptive perturbation (`n_perturb: 4096`) to robustly estimate precision/coverage during training.
