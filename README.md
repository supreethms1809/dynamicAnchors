# Dynamic Anchors

Research code that extracts explainable decision rules ("anchors") from tabular
datasets with reinforcement learning. Two pipelines share the same quantile-position
MDP, reward, and held-out evaluation so results are comparable:

| Arm | Code | Algorithm |
|---|---|---|
| RLDA (single-agent) | `single_agent/` | DDPG / SAC |
| MADA (multi-agent) | `BenchMARL/` | MADDPG / MASAC |

Agents grow axis-aligned boxes in unit-normalized feature space. Training uses
empirical fidelity (Fid) as the done-switch. Paper cells come from
`revision.evaluate` (rank on validation, report Fid/Pur on test).

## Setup

```bash
conda create -n dynamic-anchors python=3.12
conda activate dynamic-anchors
pip install -r BenchMARL/requirements.txt
```

## Results pipeline (what produces today's numbers)

Overnight sweep and paper-seed runs use the same stack:

```bash
# Full overnight sweep (RLDA + MADA, multiple seeds)
python revision/run_overnight_sweep.py

# One-seed paper run
python revision/run_paper_seed.py --datasets iris wine breast_cancer --seed 42

# Individual arms
python revision/run_rlda_pipeline.py --algo ddpg --datasets iris --seed 42
python revision/run_mada_pipeline.py --algo maddpg --datasets iris --seed 42

# WyoDOT (same train → infer → revision.evaluate path)
python wyodot/run_pipeline.py --dataset wyodot_kvdw_labeled --algorithm maddpg --seed 42
```

`revision.evaluate` is the scorer (τ_P=0.90, τ_C=0.10 by default). Baselines
(Anchors, CART, greedy, random search) are `python -m revision.baselines`.
Tables/figures: `paper/make_tables.py`, `paper/make_figures.py`.

Generated writeups: [`docs/RESULTS_comparison.md`](docs/RESULTS_comparison.md),
[`docs/RULES.md`](docs/RULES.md).

## Direct training / inference

```bash
# Multi-agent
cd BenchMARL
python driver.py --dataset breast_cancer --algorithm maddpg --seed 42
python inference.py --experiment_dir <experiment_folder> --dataset breast_cancer
python -m revision.evaluate --rules_file <experiment_folder>/inference/extracted_rules.json --dataset breast_cancer --method mada

# Single-agent
python single_agent/driver.py --dataset breast_cancer --algorithm ddpg --seed 42
python single_agent/single_agent_inference.py --experiment_dir <path> --dataset breast_cancer
python -m revision.evaluate --rules_file <path>/inference/extracted_rules.json --dataset breast_cancer --method rlda
```

Algorithms: MADDPG / MASAC (multi), DDPG / SAC (single).
Datasets: `breast_cancer`, `wine`, `iris`, `synthetic`, `moons`, `circles`,
`covtype`, `housing`, plus optional UCIML and Folktables names.

## Reward (live)

```
reward = shaping_gain - overlap_penalty - drift_penalty - anchor_drift_penalty + coverage_floor_penalty
```

- `shaping_gain` = `discount * Phi' - Phi` (`discount: 0.95`, must match the algo gamma)
- `overlap_penalty` uses YAML **`gamma: 0.1`** (narrow-width weight, not MDP discount)
- `drift_penalty` from `drift_penalty_weight: 0.05`
- `anchor_drift_penalty` from **`initial_window: 0.1`**

Config: `BenchMARL/conf/anchor.yaml`, `single_agent/conf/anchor_single.yaml`.

## Tests

```bash
cd tests
python -m pytest
```

## Architecture

| Component | Multi-agent (`BenchMARL/`) | Single-agent (`single_agent/`) |
|---|---|---|
| Entry point | `driver.py` | `driver.py` |
| Trainer | `anchor_trainer.py` | `anchor_trainer_sb3.py` |
| Environment | `environment.py` (PettingZoo) | `single_agentENV.py` (Gymnasium) |
| Inference | `inference.py` | `single_agent_inference.py` |
| Evaluation | `python -m revision.evaluate` | same |
| Agents per class | Multiple (YAML `agents_per_class`) | One per class |

Observation is `3n+4`: quantile bounds `a`, `b`, instance quantiles `q*`,
precision, coverage, mode, episode phase. Action is leave-corner quantile
adjustment. The environment starts from the empty rule (`k=0`).

Data (`BenchMARL/tabular_datasets.py`): train/test split, StandardScaler for the
classifier, min-max `[0,1]` for the RL box. Do not conflate the two spaces.
