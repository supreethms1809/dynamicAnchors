# How Policy Loading Works in Multi-Agent Inference

## Overview

When you train with `agents_per_class: 3`, BenchMARL creates **6 agents total**:
- Class 0: `agent_0_0`, `agent_0_1`, `agent_0_2`
- Class 1: `agent_1_0`, `agent_1_1`, `agent_1_2`

Each agent gets its **own individual policy network** during training, and these are saved separately.

## What Happens During Inference

### Step 1: Policy Loading (lines 929-1145)

The inference code loads **all individual policy files**:
```
Loaded combined policy for agent_0_0
Loaded combined policy for agent_0_1
Loaded combined policy for agent_0_2
Loaded combined policy for agent_1_0
Loaded combined policy for agent_1_1
Loaded combined policy for agent_1_2
```

**Note:** The term "combined policy" is misleading here. These are actually **individual policy files** saved during training. Each file contains one agent's policy network.

### Step 2: Policy Extraction (lines 1034-1145)

The code tries to extract individual agent policies from each loaded file. If the state_dict has agent-specific keys (like `"0.mlp.params.0.weight"`), it extracts them. Otherwise, it uses the policy as-is.

### Step 3: Policy Mapping (lines 1284-1332)

**This is the key part!** The code maps policies to agents:

```python
if agents_per_class > 1:
    for k in range(agents_per_class):
        agent_name = f"agent_{target_class}_{k}"
        if agent_name not in agent_policies:
            agent_policies[agent_name] = combined_policy  # Same policy for all!
```

**Problem:** Even though individual policies are loaded (agent_0_0, agent_0_1, agent_0_2), the code maps the **same `combined_policy` object** to all agents of the same class. This means:
- `agent_policies["agent_0_0"]` = policy from agent_0_0 file
- `agent_policies["agent_0_1"]` = **same policy** (from agent_0_0 file)
- `agent_policies["agent_0_2"]` = **same policy** (from agent_0_0 file)

### Step 4: Environment Creation (lines 1432-1443)

When creating the environment for rollouts:

```python
single_agent_config = anchor_config.copy()
single_agent_config["target_classes"] = [target_class]  # Only this class
env = AnchorEnv(**single_agent_config)
```

The environment uses `env_config` which has `agents_per_class: 1` (from your config change). So the environment only creates:
- `agent_0` (for class 0)
- `agent_1` (for class 1)

**Not** `agent_0_0`, `agent_0_1`, `agent_0_2`, etc.

### Step 5: Policy Usage (lines 1401-1487)

The code iterates over `policies.items()`, which contains:
- `"agent_0"` → policy (from agent_0_0 file)
- `"agent_1"` → policy (from agent_1_0 file)

When it runs a rollout:
1. Creates environment with only `agent_0` (because `agents_per_class: 1`)
2. Uses the policy mapped to `"agent_0"` (which is actually the policy from agent_0_0 file)
3. The policies for agent_0_1 and agent_0_2 are **never used**

## The Issue

**Mismatch between training and inference:**

1. **Training:** `agents_per_class: 3` → 3 agents per class, each with its own policy
2. **Inference:** `agents_per_class: 1` → 1 agent per class, only uses one policy

**Result:**
- Only 1 out of 3 policies per class is actually used
- The other 2 policies are loaded but ignored
- This is why you see "Using individual agent policies: ['agent_0', 'agent_1']" - only 2 agents total, not 6

## How to Fix

### Option 1: Use All Policies (Recommended for Multi-Agent)

To actually use all 3 policies per class, you need to:

1. **Keep `agents_per_class: 3` in config** during inference
2. **Fix the policy mapping** to use individual policies instead of mapping the same policy to all agents
3. **Run rollouts for each agent** (agent_0_0, agent_0_1, agent_0_2 separately)

### Option 2: Use Single Policy (Current Behavior)

If you want to use only one policy per class (current behavior):
1. Keep `agents_per_class: 1` in config
2. The code will use the first policy it finds for each class
3. This is simpler but doesn't leverage the diversity of multiple policies

## Current Behavior Summary

**What's happening:**
- ✅ All 6 individual policies are loaded
- ✅ Policies are extracted correctly
- ❌ Policy mapping assigns the same policy to all agents of a class
- ❌ Environment only creates 1 agent per class (due to config)
- ❌ Only 1 policy per class is actually used

**Why it works:**
- The code falls back to using the first policy it finds for each class
- Since all agents of a class are mapped to the same policy anyway, it doesn't matter
- The environment only needs one agent, so it works

**What's lost:**
- The diversity of having 3 different policies per class
- The ability to extract multiple anchors per class using different policies
- The full benefit of multi-agent training

## Recommendation

If you want to use all 3 policies per class:
1. Set `agents_per_class: 3` in the config for inference
2. Modify the inference code to properly map individual policies to individual agents
3. Run separate rollouts for each agent (agent_0_0, agent_0_1, agent_0_2)

If you just want one policy per class (simpler):
1. Keep `agents_per_class: 1` in config
2. The current code will work fine
3. You'll get one anchor per class, using the first policy it finds

