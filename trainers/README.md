# Dynamic Anchors with Stable-Baselines3

A complete implementation of Dynamic Anchor explanations for tabular classification using reinforcement learning, fully compatible with Stable-Baselines3.

## Overview

This module provides a pipeline for training PPO agents to generate high-precision, high-coverage anchor explanations for tabular classification models. It uses Stable-Baselines3 for robust RL training and vectorized environments for efficient parallel training.

## Quick Start

### Installation

```bash
pip install stable-baselines3 gymnasium numpy torch scikit-learn
```

### Basic Usage

```python
from trainers.tabular_dynAnchors import train_and_evaluate_dynamic_anchors
from trainers.networks import SimpleClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Prepare your data
X_train, y_train, X_test, y_test = ...  # Your data
feature_names = [...]  # Feature names

# Train a classifier
classifier = SimpleClassifier(n_features, n_classes)
# ... train classifier ...

# Train dynamic anchors
results = train_and_evaluate_dynamic_anchors(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    classifier=classifier,
    target_classes=(0, 1, 2),  # Classes to explain
    n_envs=4,                  # Parallel environments
    total_timesteps=50000,     # Training budget
)

print(f"Overall precision: {results['overall_stats']['avg_precision']:.3f}")
print(f"Overall coverage: {results['overall_stats']['avg_coverage']:.3f}")
```

## Module Structure

### Core Modules

- **`vecEnv.py`** - Environment implementation with Gym wrapper
  - `AnchorEnv` - Core dynamic anchor environment
  - `SimpleClassifier` - Neural network classifier
  - `DynamicAnchorEnv` - Gym-compatible wrapper
  - `make_vec_env` - Vectorization helpers

- **`networks.py`** - Neural network architectures
  - `SimpleClassifier` - Classification network
  - `PolicyNet` - Policy network (optional, SB3 uses its own)
  - `ValueNet` - Value network (optional)

- **`PPO_trainer.py`** - PPO training infrastructure
  - `DynamicAnchorPPOTrainer` - SB3 PPO wrapper
  - `train_ppo_model` - Complete training pipeline

- **`dynAnchors_inference.py`** - Post-training evaluation
  - `greedy_rollout` - Greedy policy evaluation
  - `evaluate_single_instance` - Instance-level explanations
  - `evaluate_class` - Class-level evaluations
  - `evaluate_all_classes` - Multi-class evaluation

- **`tabular_dynAnchors.py`** - End-to-end pipeline
  - `train_and_evaluate_dynamic_anchors` - Complete workflow

### Test Modules

- **`test_vecEnv.py`** - Environment tests
- **`test_PPO_trainer.py`** - PPO trainer tests
- **`test_inference.py`** - Inference tests

## Testing

Run all tests:

```bash
# Environment tests
python trainers/test_vecEnv.py

# PPO trainer tests
python trainers/test_PPO_trainer.py --test-training

# Inference tests
python trainers/test_inference.py --test-all

# All tests
python trainers/test_vecEnv.py
python trainers/test_PPO_trainer.py --test-envs --test-training
python trainers/test_inference.py --test-all
```

## Features

✅ **Stable-Baselines3 Integration**
- Full Gymnasium/Gym interface
- DummyVecEnv and SubprocVecEnv support
- Compatible with all SB3 algorithms

✅ **Efficient Training**
- Vectorized environments for parallel training
- Checkpointing during training
- Automatic evaluation callbacks
- Tensorboard integration

✅ **Flexible Evaluation**
- Single instance explanations
- Per-class aggregate statistics
- Multi-class evaluation
- Configurable rollout parameters

✅ **Production Ready**
- Comprehensive test suite
- Proper error handling
- Save/load functionality
- Well-documented APIs

## Environment Details

### State Space
- `Box(2 * n_features + 2,)`
- Lower bounds for each feature
- Upper bounds for each feature
- Current precision
- Current coverage

### Action Space
- `Discrete(n_features * 4 * n_step_sizes)`
- Actions encode (feature_idx, direction, magnitude)
- Directions: shrink_lower, expand_lower, shrink_upper, expand_upper

### Reward
- `α * precision_gain + β * coverage_gain - penalties`
- Penalties for overlap, drift, JS-divergence
- Coverage floor enforcement

## License

Part of the Dynamic Anchors project.

