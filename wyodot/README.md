# WyoDOT Dynamic Anchors

Road surface condition classification (Dry, Wet, Snow, Slush, Ice) using Dynamic Anchors with a Random Forest classifier.

## Datasets

| Name | File | Rows | Features | Description |
|------|------|------|----------|-------------|
| `wyodot_kvdw_labeled` | `KVDW_labeled.csv` | 39,858 | 5 (air_temp, humidity, dewpoint, road_temp, wind_speed) | Synoptic weather station data, labeled via rules learned from testbed |
| `wyodot_testbed` | `merged_dataset_new.csv` | 691 | 7 (mean_voltage_V, std_voltage_V, Temperature_C, Humidity_pct, DewPoint_C, AvgSurfaceTemp_C, WindSpeed_mps) | Outdoor testbed with SurfaceVue10 verified labels |

### Preprocessing

- **NaN rows** are dropped automatically.
- **Testbed label remapping**: `Snow/Frost` -> `Snow`, `Moist` -> `Wet`, `Error` rows dropped.

## Random Forest Classifier (from paper)

```
n_estimators: 300
max_depth: 20
min_samples_split: 2
min_samples_leaf: 1
max_features: sqrt
class_weight: balanced
random_state: 42
n_jobs: -1
```

| Dataset | Accuracy |
|---------|----------|
| wyodot_kvdw_labeled | ~98.9% |
| wyodot_testbed | ~92.7% |

## Usage

All commands run from the `wyodot/` directory.

### Multi-Agent Training (BenchMARL)

```bash
# MADDPG
python driver_multi_agent.py --dataset wyodot_kvdw_labeled --algorithm maddpg --seed 42
python driver_multi_agent.py --dataset wyodot_testbed --algorithm maddpg --seed 42

# MASAC
python driver_multi_agent.py --dataset wyodot_kvdw_labeled --algorithm masac --seed 42
```

### Single-Agent Training (Stable-Baselines3)

```bash
# DDPG
python driver_single_agent.py --dataset wyodot_kvdw_labeled --algorithm ddpg --seed 42
python driver_single_agent.py --dataset wyodot_testbed --algorithm ddpg --seed 42

# SAC
python driver_single_agent.py --dataset wyodot_testbed --algorithm sac --seed 42
```

### Common Options

| Flag | Default | Description |
|------|---------|-------------|
| `--classifier_type` | `random_forest` | `random_forest`, `dnn`, or `gradient_boosting` |
| `--seed` | `42` | Random seed |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` (multi) / `auto` (single) |
| `--skip_eda` | off | Skip EDA analysis |
| `--max_cycles` | from YAML | Max steps per episode |
| `--max_n_frames` | from YAML | Total training frames (multi-agent only) |
| `--total_timesteps` | `60000` | Total training timesteps (single-agent only) |
| `--target_classes` | all | Specific class indices to train on |
| `--load_checkpoint` | none | Path to checkpoint for evaluation only |

### Full Pipeline (recommended)

Runs training, inference, rule testing, and comparison plots end-to-end:

```bash
# Full pipeline for both single-agent and multi-agent
python run_pipeline.py --dataset wyodot_kvdw_labeled --algorithm maddpg --seed 42

# Skip training (reuse existing models), only run inference + testing + plots
python run_pipeline.py --dataset wyodot_testbed --algorithm maddpg --skip_training

# Only multi-agent pipeline
python run_pipeline.py --dataset wyodot_kvdw_labeled --algorithm maddpg --skip_single_agent

# Only single-agent pipeline
python run_pipeline.py --dataset wyodot_kvdw_labeled --algorithm maddpg --skip_multi_agent
```

### Pipeline Options

| Flag | Description |
|------|-------------|
| `--skip_training` | Skip training, use existing models |
| `--skip_inference` | Skip inference, use existing rules |
| `--skip_testing` | Skip rule testing |
| `--skip_single_agent` | Skip single-agent pipeline |
| `--skip_multi_agent` | Skip multi-agent pipeline |
| `--force_retrain` | Force retraining even if models exist |
| `--n_instances_per_class` | Instances per class for inference (default: 20) |
| `--steps_per_episode` | Steps per episode for inference (default: 500) |
| `--max_n_frames` | Override multi-agent training frames |
| `--total_timesteps` | Override single-agent training timesteps |

## Output Structure

```
wyodot/output/
    single_agent_sb3_wyodot_kvdw_labeled_ddpg/training/   # Single-agent models
    wyodot_kvdw_labeled_maddpg/training/                   # Multi-agent models

wyodot/comparison_results/wyodot_results/
    wyodot_kvdw_labeled_maddpg_<timestamp>/
        pipeline_run.log                     # Full pipeline log
        comparison_summary.json              # Side-by-side metrics
        single_agent/summary.json            # Single-agent summary + plots
        multi_agent/summary.json             # Multi-agent summary + plots
        classifier_rules_analysis/           # Rule analysis
```
