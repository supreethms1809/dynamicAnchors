Dynamic Anchors: BenchMARL (Multi‑Agent) and Single‑Agent Pipelines

### Conda environment (Python 3.12) and dependencies

```bash
conda create -n dynamic-anchors python=3.12
conda activate dynamic-anchors

# Install core dependencies (covers BenchMARL and single_agent)
pip install -r BenchMARL/requirements.txt
```

### BenchMARL: multi‑agent training

- **Basic training**

```bash
cd BenchMARL

# Example: breast_cancer with MADDPG
python driver.py --dataset breast_cancer --algorithm maddpg --seed 42
```

- **Supported options (high level)**
  - **Datasets**: `breast_cancer`, `wine`, `iris`, `synthetic`, `moons`, `circles`, `covtype`, `housing`, plus optional UCIML and Folktables datasets if those packages are installed.
  - **Algorithms**: `maddpg`, `masac` (continuous‑action MARL).
  - **Key flags** (see `BenchMARL/driver.py` for full list):
    - `--experiment_config conf/base_experiment.yaml`
    - `--algorithm_config conf/<algo>.yaml` (defaults if omitted)
    - `--mlp_config conf/mlp.yaml`
    - `--classifier_type {dnn,random_forest,gradient_boosting}`

- **Outputs**
  - Training runs and classifiers are written under `BenchMARL/output/{dataset}_{algorithm}/training/`.
  - BenchMARL checkpoints (used for later evaluation/inference) are in the experiment folder printed at the end of training.

### BenchMARL: evaluation / inference and rule extraction

- **Evaluate a trained checkpoint (no further training)**

```bash
cd BenchMARL

python driver.py \
  --dataset breast_cancer \
  --algorithm maddpg \
  --load_checkpoint <path_to_experiment_folder>
```

This reloads the experiment, runs evaluation, and (when possible) extracts individual models for standalone use.

- **Rule / anchor extraction**

```bash
cd BenchMARL

python inference.py \
  --experiment_dir <path_to_experiment_folder> \
  --dataset breast_cancer
```

This uses the saved BenchMARL checkpoint (and classifier) to extract final dynamic anchor rules.

- **Test rule extraction**
```bash
cd BenchMARL

python test_extracted_rules.py \
  --rules_file <path_to_extracted_rules.json> \
  --dataset breast_cancer
```

Metrics
- Precision and Coverage of each rule per class
- Missed samples
- Rule overlap analysis

### Single‑agent (Stable‑Baselines3) training

- **Basic training**

```bash
# From repo root
python single_agent/driver.py --dataset breast_cancer --algorithm ddpg --seed 42

# SAC variant
python single_agent/driver.py --dataset breast_cancer --algorithm sac --seed 42
```

- **Advanced configuration (examples)**

```bash
python single_agent/driver.py \
  --dataset breast_cancer \
  --algorithm ddpg \
  --seed 42 \
  --total_timesteps 3072000 \
  --learning_rate 5e-5 \
  --max_cycles 1000 \
  --device cuda \
  --eval_on_test_data
```

- **Outputs**
  - Results are written under `output/single_agent_sb3_{dataset}_{algorithm}/`.
  - Inside `training/`, you will find SB3 checkpoints, the best/final models, TensorBoard logs, and `classifier.pth`.

### Single‑agent evaluation / inference

- **Evaluate or run inference from a saved experiment**

```bash
python single_agent/driver.py \
  --dataset breast_cancer \
  --algorithm ddpg \
  --load_checkpoint <path_to_experiment_folder>
```

This reloads the saved SB3 policy and classifier, runs evaluation over multiple episodes, and logs aggregate performance metrics.

- **Test rule extraction**
```bash
cd single_agent

python test_extracted_rules_single.py \
  --rules_file <path_to_extracted_rules.json> \
  --dataset breast_cancer
```

Metrics
- Precision and Coverage of each rule per class
- Missed samples
- Rule overlap analysis
