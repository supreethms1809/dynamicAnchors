# Interpreting NashConv and Exploitability Metrics

## Overview

**NashConv** (Nash Convergence) and **Exploitability** are metrics used to measure how close a multi-agent policy is to a **Nash Equilibrium**. These metrics help you understand whether your agents have converged to stable, optimal strategies.

## Key Concepts

### 1. **Nash Equilibrium**
A Nash Equilibrium is a state where no agent can improve their expected reward by unilaterally changing their strategy, assuming all other agents keep their strategies fixed.

### 2. **Exploitability (per agent)**
For each agent `i`, exploitability measures:
```
Exploitability_i = max_{a_i} Q_i(s, a_i, a_{-i}) - Q_i(s, a^π)
```

Where:
- `Q_i(s, a_i, a_{-i})` is the Q-value (expected reward) for agent `i` taking action `a_i` while opponents take actions `a_{-i}`
- `a^π` is the action taken by the current policy
- The `max` is over all possible actions for agent `i`

**Interpretation:**
- **Low exploitability (close to 0)**: The agent's current policy is close to optimal given opponents' strategies
- **High exploitability**: The agent could significantly improve by deviating from its current policy

### 3. **NashConv Sum**
The sum of all agents' exploitabilities:
```
NashConv = Σ_i Exploitability_i
```

**Interpretation:**
- **NashConv ≈ 0**: All agents are playing near-optimal strategies → **Nash Equilibrium achieved**
- **NashConv > 0**: At least one agent can improve → **Not at equilibrium**

### 4. **Exploitability Max**
The maximum exploitability across all agents:
```
Exploitability_max = max_i Exploitability_i
```

**Interpretation:**
- Shows which agent is furthest from optimal play
- Useful for identifying which agent needs more training

## Epsilon-Nash Equilibrium

An **ε-Nash Equilibrium** is achieved when:
```
NashConv ≤ ε
```

Where `ε` is a small threshold (e.g., 0.01, 0.001).

**Example:**
- If `NashConv = 0.015` and `ε = 0.01`, you're close but not quite at equilibrium
- If `NashConv = 0.005` and `ε = 0.01`, you've achieved an ε-Nash equilibrium

## Interpreting Your Metrics

### From Your Training Output:

**Evaluation NashConv (from your logs):**
- `NashConv sum: 0.015784, Exploitability max: 0.008841`
- `NashConv sum: 0.020481, Exploitability max: 0.011087`
- `NashConv sum: 0.014379, Exploitability max: 0.007300`
- `NashConv sum: 0.019986, Exploitability max: 0.010838`

**What this means:**
1. **NashConv values (~0.015-0.020)**: Your agents are relatively close to equilibrium, but there's still room for improvement
2. **Exploitability max (~0.007-0.011)**: The worst-performing agent could improve their expected reward by about 0.7-1.1% by deviating
3. **Trend**: If these values are decreasing over training, your agents are converging toward equilibrium

### Good vs. Bad Values

**Good (converging):**
- NashConv decreasing over time: `0.05 → 0.02 → 0.01 → 0.005`
- Exploitability max decreasing: `0.02 → 0.01 → 0.005 → 0.001`
- Values stabilize at low levels

**Bad (not converging):**
- NashConv increasing or staying high: `0.01 → 0.02 → 0.03`
- Exploitability max staying high: `0.01 → 0.01 → 0.01` (not improving)
- High variance (values jumping around)

### Typical Thresholds

- **ε = 0.01 (1%)**: Reasonable for most applications
- **ε = 0.001 (0.1%)**: Very strict, for high-precision applications
- **ε = 0.1 (10%)**: Loose, acceptable for exploratory training

## Class-Level Metrics

If you have multiple agents per class, you'll also see:
- **class_nashconv_sum**: Sum of exploitabilities for all agents in a class
- **class_exploitability_max**: Maximum exploitability within a class

These help you understand:
- Which class of agents is performing better/worse
- Whether agents within the same class are converging similarly

## What to Look For During Training

1. **Decreasing trend**: NashConv should generally decrease as training progresses
2. **Stability**: Once converged, values should stabilize at low levels
3. **Comparison**: Training NashConv vs. Evaluation NashConv should be similar (if different, may indicate overfitting)
4. **Per-agent breakdown**: Check individual exploitabilities to see which agents need more training

## Example Interpretation

```
Training NashConv (iter 100, batch 50): sum=0.015784, max=0.008841
Training NashConv (iter 200, batch 100): sum=0.012345, max=0.006789
Training NashConv (iter 300, batch 150): sum=0.009876, max=0.005432
Training NashConv (iter 400, batch 200): sum=0.008234, max=0.004567
```

**Analysis:**
- ✅ NashConv is decreasing: `0.0158 → 0.0123 → 0.0099 → 0.0082`
- ✅ Exploitability max is decreasing: `0.0088 → 0.0068 → 0.0054 → 0.0046`
- ✅ Convergence is happening, approaching ε-Nash equilibrium (if ε = 0.01)
- The agents are learning stable strategies

## References

- **Nash Equilibrium**: A fundamental concept in game theory
- **Exploitability**: Measures deviation from optimal play
- **NashConv**: Standard metric in multi-agent RL for measuring convergence
