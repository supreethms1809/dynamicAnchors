# Network Architectures

## Overview

This document describes the neural network architectures used in the Dynamic Anchors system, including policy networks, critic networks, and classifier networks.

---

## Policy Networks

### Multi-Agent: MLP Architecture

**Location**: `BenchMARL/conf/mlp.yaml` (config-driven)

**Architecture**: Multi-Layer Perceptron (MLP)

**Structure**:
```
Input (state_dim) 
  → Linear(hidden_size) 
  → ReLU 
  → Linear(hidden_size) 
  → ReLU 
  → Linear(action_dim) 
  → Tanh
```

**Default Configuration**:
- **Hidden Size**: 256 (configurable)
- **Activation**: ReLU for hidden layers, Tanh for output
- **Output**: Bounded to `[-1, 1]` via Tanh

**Input Dimension**:
```
state_dim = 2 * n_features + 2
```
- `n_features`: Lower bounds
- `n_features`: Upper bounds  
- `1`: Current precision
- `1`: Current coverage

**Output Dimension**:
```
action_dim = 2 * n_features
```
- `n_features`: Deltas for lower bounds
- `n_features`: Deltas for upper bounds

**Code Reference**: BenchMARL uses config-driven MLP creation

---

### Single-Agent: Stable-Baselines3 MLP

**Location**: Stable-Baselines3 `MlpPolicy`

**Architecture**: Similar MLP structure

**Key Differences**:
- Uses SB3's policy network structure
- Supports both DDPG and SAC algorithms
- Action distribution: `TanhNormal` for SAC, deterministic for DDPG

---

## Critic Networks (Multi-Agent)

### Centralized Critic

**Location**: `BenchMARL/benchmarl_wrappers.py` (MADDPG/MASAC)

**Architecture**: Centralized Q-function

**Key Feature**: Observes all agents' states during training

**Structure**:
```
Input (concatenated states of all agents)
  → Linear(hidden_size)
  → ReLU
  → Linear(hidden_size)
  → ReLU
  → Linear(1)  # Q-value
```

**Purpose**:
- Enables centralized training
- Agents can observe other agents during training
- Policies remain decentralized at inference time

**Code Reference**: BenchMARL's MADDPG/MASAC implementations

---

## Classifier Networks

### SimpleClassifier

**Location**: `utils/networks.py` lines 61-120

**Architecture**: Multi-layer MLP with batch normalization and dropout

**Default Structure** (for small datasets):
```
Input (n_features)
  → Linear(256)
  → BatchNorm1d(256)
  → ReLU
  → Dropout(0.3)
  → Linear(256)
  → BatchNorm1d(256)
  → ReLU
  → Dropout(0.3)
  → Linear(128)
  → BatchNorm1d(128)
  → ReLU
  → Dropout(0.15)
  → Linear(n_classes)
```

**Configurable Hidden Sizes**:
- Small datasets: `[256, 256, 128]`
- Large datasets: `[512, 512, 256]` or `[256, 256, 256, 128]`

**Features**:
- Batch normalization for stability
- Dropout for regularization (0.3 for early layers, 0.15 for later)
- ReLU activation
- No activation on output (logits)

**Code Reference**: `utils/networks.py` `SimpleClassifier` class

---

### BigClassifier

**Location**: `utils/networks.py` lines 13-59

**Architecture**: Larger MLP for complex datasets

**Structure**:
```
Input (n_features)
  → Linear(256)
  → BatchNorm1d(256)
  → ReLU
  → Dropout(0.3)
  → Linear(256)
  → BatchNorm1d(256)
  → ReLU
  → Dropout(0.3)
  → Linear(128)
  → BatchNorm1d(128)
  → ReLU
  → Dropout(0.15)
  → Linear(64)
  → BatchNorm1d(64)
  → ReLU
  → Dropout(0.075)
  → Linear(n_classes)
```

**Features**:
- Deeper network (4 hidden layers)
- Progressive dropout reduction
- More parameters for complex patterns

**Code Reference**: `utils/networks.py` `bigClassifier` class

---

## Legacy Policy Networks

### PolicyNet (Legacy)

**Location**: `utils/multiagent_networks.py` lines 106-135

**Architecture**: Custom policy network

**Structure**:
```
Input (input_size)
  → Linear(hidden_size)
  → ReLU
  → BatchNorm1d(hidden_size)
  → Linear(hidden_size)
  → ReLU
  → BatchNorm1d(hidden_size)
  → Dropout(0.3)
  → Linear(output_size)
  → Tanh
```

**Features**:
- Batch normalization
- Dropout regularization
- Tanh output activation
- Xavier uniform initialization

**Note**: Legacy implementation, BenchMARL uses config-driven MLP

---

## Architecture Configuration

### MLP Config File

**Location**: `BenchMARL/conf/mlp.yaml`

**Example Configuration**:
```yaml
num_cells: [256, 256]  # Hidden layer sizes
activation: "relu"
layer_class: "torch.nn.Linear"
```

**Interpretation**:
- `num_cells`: List of hidden layer sizes
- `activation`: Activation function (ReLU)
- `layer_class`: Type of layer (Linear)

---

## Weight Initialization

### Default Initialization

**Policy Networks**:
- Xavier uniform initialization (default in PyTorch)
- Biases initialized to zero

**Critic Networks**:
- Xavier uniform initialization
- Ensures stable gradients

**Classifier Networks**:
- Xavier uniform initialization
- Standard PyTorch defaults

---

## Activation Functions

### ReLU (Hidden Layers)

**Formula**: `f(x) = max(0, x)`

**Purpose**: Non-linearity, prevents vanishing gradients

**Used In**: All hidden layers

### Tanh (Policy Output)

**Formula**: `f(x) = tanh(x)`

**Purpose**: Bounds output to `[-1, 1]`

**Used In**: Policy network outputs (actions)

**Note**: Actions are then scaled by `max_action_scale`

### No Activation (Classifier Output)

**Purpose**: Outputs logits (before softmax)

**Used In**: Classifier final layer

---

## Regularization Techniques

### Batch Normalization

**Purpose**: Stabilizes training, reduces internal covariate shift

**Used In**: 
- Classifier networks (all hidden layers)
- Legacy policy networks

**Parameters**:
- `eps=1e-5`: Small constant for numerical stability
- `momentum=0.1`: Moving average momentum

### Dropout

**Purpose**: Prevents overfitting

**Rates**:
- Early layers: 0.3 (30% dropout)
- Later layers: 0.15-0.075 (reduced dropout)

**Used In**: Classifier networks

---

## Network Sizes

### Typical Dimensions

**Small Dataset** (e.g., Breast Cancer, Iris):
- Features: 10-30
- State: 20-60 dimensions
- Action: 20-60 dimensions
- Hidden: 256 units

**Medium Dataset** (e.g., Wine, Housing):
- Features: 10-15
- State: 20-30 dimensions
- Action: 20-30 dimensions
- Hidden: 256 units

**Large Dataset** (e.g., Folktables):
- Features: 50-100+
- State: 100-200+ dimensions
- Action: 100-200+ dimensions
- Hidden: 256-512 units

---

## Code References

- **MLP Config**: `BenchMARL/conf/mlp.yaml`
- **Classifier**: `utils/networks.py` `SimpleClassifier`, `bigClassifier`
- **Legacy Policy**: `utils/multiagent_networks.py` `PolicyNet`
- **BenchMARL**: Uses config-driven architecture creation

---

## Architecture Selection

### For Classifiers

**SimpleClassifier**: Default for most datasets
- Good balance of capacity and regularization
- Works well for small to medium datasets

**BigClassifier**: For complex datasets
- More capacity for complex patterns
- Use when SimpleClassifier underfits

**Custom Hidden Sizes**: For very large datasets
- Increase hidden sizes: `[512, 512, 256]`
- Or add more layers: `[256, 256, 256, 128]`

### For Policies

**MLP**: Default for all scenarios
- Configurable via YAML
- Works well for continuous action spaces
- Standard architecture for RL

---

## Summary

**Key Architectures**:

1. **Policy Networks**: MLP with ReLU hidden layers, Tanh output
2. **Critic Networks**: Centralized MLP (multi-agent only)
3. **Classifier Networks**: MLP with batch norm and dropout

**Design Principles**:

- **Simplicity**: Standard MLP architectures
- **Regularization**: Batch norm and dropout for classifiers
- **Bounded Actions**: Tanh activation for policy outputs
- **Configurability**: YAML-based configuration for flexibility

**For Paper Writing**:

- Describe MLP architecture for policies
- Explain centralized critic for multi-agent
- Mention classifier architecture
- Discuss choice of activation functions and regularization

