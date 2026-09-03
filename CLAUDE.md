# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Dynamic Anchors** extracts explainable decision rules ("anchors") from tabular
datasets using reinforcement learning — multi-agent (BenchMARL/MADA) and
single-agent (Stable-Baselines3/RLDA). Both arms share the quantile-position MDP
and the same reward so results are comparable. Paper numbers come from
`revision.evaluate` (rank on D_val, Fid/Pur on D_test), not from printed rule strings.

## Setup

```bash
conda create -n dynamic-anchors python=3.12
conda activate dynamic-anchors
pip install -r BenchMARL/requirements.txt
```

## Common Commands

### Results pipeline (today's collector)

```bash
python revision/run_overnight_sweep.py
python revision/run_paper_seed.py --datasets iris wine --seed 42
python revision/run_rlda_pipeline.py --algo ddpg --datasets iris --seed 42
python revision/run_mada_pipeline.py --algo maddpg --datasets iris --seed 42
python wyodot/run_pipeline.py --dataset wyodot_kvdw_labeled --algorithm maddpg --seed 42
python -m revision.baselines --dataset iris --seed 42
python -m revision.evaluate --rules_file <extracted_rules.json> --dataset iris --method rlda --seed 42
```

### Multi-agent training / inference (BenchMARL)

```bash
cd BenchMARL
python driver.py --dataset breast_cancer --algorithm maddpg --seed 42
# Algorithms: maddpg, masac
python inference.py --experiment_dir <path_to_experiment_folder> --dataset breast_cancer
```

### Single-agent training / inference (SB3)

```bash
python single_agent/driver.py --dataset breast_cancer --algorithm ddpg --seed 42
# Algorithms: ddpg, sac
python single_agent/single_agent_inference.py --experiment_dir <path> --dataset breast_cancer
```

### Tests

```bash
cd tests
python -m pytest
```

## Architecture

Both pipelines share the same reward and dataset handling:

| Component | Multi-Agent (`BenchMARL/`) | Single-Agent (`single_agent/`) |
|---|---|---|
| Entry point | `driver.py` | `driver.py` |
| Trainer | `anchor_trainer.py` | `anchor_trainer_sb3.py` |
| Environment | `environment.py` (PettingZoo) | `single_agentENV.py` (Gymnasium) |
| Inference | `inference.py` | `single_agent_inference.py` |
| Evaluation | `python -m revision.evaluate` | same |
| Algorithm | MADDPG / MASAC | DDPG / SAC |
| Agents per class | Multiple (configurable) | One per class |

### Data Pipeline (`BenchMARL/tabular_datasets.py`)

1. Load dataset → train/test split (80/20)
2. `StandardScaler` normalization for classification
3. Min-max unit normalization to [0, 1] for RL feature space
4. Train classifier: `dnn`, `random_forest`, or `gradient_boosting`
5. Expose `get_anchor_env_data()` for the RL environment

### RL Environment Design

Quantile-position MDP. Agents start from the empty rule (`k=0`) and add
predicates by leave-corner actions on class-conditional quantile bounds.

- **Action space:** leave-corner adjustments in quantile space
- **Observation space:** `3n+4` = `a`, `b`, `q*`, precision, coverage, mode, episode phase
- **Reward:** `shaping_gain - overlap_penalty - drift_penalty - anchor_drift_penalty + coverage_floor_penalty`
  - `shaping_gain = discount * Phi' - Phi` (`discount: 0.95`, must match algo gamma)
  - overlap weight is YAML **`gamma: 0.1`** (not MDP discount)
  - `anchor_drift_penalty` uses **`initial_window: 0.1`**

Multi-agent: multiple agents per class. Checkpoint selection uses FidCov + NashConv.

### Configuration (YAML)

Do not hardcode hyperparameters:

- `BenchMARL/conf/base_experiment.yaml` — training loop (lr, batch, frames, device)
- `BenchMARL/conf/anchor.yaml` — env/reward (`precision_target`, `coverage_target`, `alpha`, `beta`, `gamma`, `discount`, `initial_window`, `agents_per_class`, `precision_estimator: empirical`)
- `BenchMARL/conf/maddpg.yaml` / `masac.yaml` — algorithm
- `BenchMARL/conf/mlp.yaml` — network
- `single_agent/conf/anchor_single.yaml` — single-agent equivalent (multi-agent fields 0)

### Shared Utilities (`utils/`)

- `quantile_mdp.py` — CDF knots, leave-corner actions, unit-bound sync
- `metrics.py`, `eval_harness.py`, `inference_extract.py` — revision scoring
- `dataset_factory.py`, `networks.py`, `device_utils.py`, `clusters.py`

### Output Structure

```
BenchMARL/output/{dataset}_{algorithm}/training/
    checkpoint.pt, individual_models/, classifier.pth, training_history.json

output/single_agent_sb3_{dataset}_{algorithm}/training/
    SB3 checkpoints, classifier.pth, TensorBoard logs

experiment_folder/inference/
    extracted_rules.json, evaluation_metrics.json

revision/ / runs/  — evaluate JSON artifacts consumed by paper/make_tables.py
```

### Baselines

`revision/baselines.py` (`python -m revision.baselines`): Anchors, CART, greedy,
random search, same schema as `revision.evaluate`.

## Key Design Decisions

- **Two normalizations:** StandardScaler for the classifier; min-max [0,1] for the RL agent. Don't conflate them.
- **Fair comparison:** Identical live reward weights in both YAML files.
- **Empty-rule start:** `k=0` covers the whole space but is not a rule; union/overlap ignore it until a predicate is added.
- **Track A vs Track B:** training termination uses empirical Fid; conditional CRN is Track B / instance evaluation.
- **Individual models:** After multi-agent training, `AnchorTrainer` extracts per-agent policies for standalone inference.
- **Inference JSON:** keep `lower_bounds_normalized` / `lower_bounds`; `revision.evaluate` reads those keys.
