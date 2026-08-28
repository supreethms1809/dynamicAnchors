# Multi-Agent Performance Issues - Analysis & Recommendations

## Current Performance Gap

**Single-Agent:**
- Instance Precision: **0.970 ± 0.022** (excellent, stable)
- Instance Coverage: **0.019 ± 0.005** (low but stable)
- Class Precision (Union): **1.000** (perfect)
- Class Coverage (Union): **0.151 ± 0.040**
- Equilibrium: **100%** (2/2 classes meet targets)

**Multi-Agent:**
- Instance Precision: **0.501 ± 0.498** (very poor, extremely high variance)
- Instance Coverage: **0.002 ± 0.002** (extremely low, 10x worse than single-agent)
- Class Precision (Union): **0.500 ± 0.500** (very poor, extremely high variance)
- Class Coverage (Union): **0.036 ± 0.036** (very low)
- Equilibrium: **0%** (0/2 classes meet targets)

## Key Issues Identified

### 1. **Extremely High Variance** (Critical)
The huge error bars (0.501 ± 0.498) indicate the multi-agent model is **extremely unstable**. This suggests:
- Training instability
- Poor convergence
- Coordination failures between agents
- Reward structure problems

### 2. **Coverage Target Mismatch** (Potential Issue)
- Config file (`BenchMARL/conf/anchor.yaml`): `coverage_target: 0.3`
- Environment default (`BenchMARL/environment.py` line 116): `0.1` (fallback)
- Single-agent config: `coverage_target: 0.3`

**Action:** Verify that `coverage_target: 0.3` is actually being passed to the environment during training.

### 3. **Multiple Agents Per Class** (Coordination Challenge)
- Multi-agent: `agents_per_class: 3` (3 agents per class)
- Single-agent: 1 agent per class

With 3 agents per class, coordination becomes much harder:
- Agents may interfere with each other
- Shared rewards may not be sufficient to encourage cooperation
- Nash equilibrium is harder to reach

### 4. **Reward Structure Analysis**

**Multi-agent reward components:**
- Local reward (per-agent precision/coverage gains)
- Shared reward (`shared_reward_weight: 0.5`) - encourages cooperation
- Class union bonus (`class_union_cov_weight: 0.03`, `class_union_prec_weight: 0.01`)
- Global coverage bonus (`global_coverage_weight: 0.03`)
- Inter-class overlap penalty (`inter_class_overlap_weight: 0.1`)

**Potential issues:**
- Shared reward might be too weak or not working correctly
- Class union bonuses might be too small to encourage coordination
- Penalties might be overwhelming the rewards

### 5. **Training Configuration Differences**

**Single-agent:**
- Algorithm: DDPG/SAC (Stable-Baselines3)
- Total timesteps: 24,000
- Learning rate: Adaptive based on dataset size
- Network architecture: Adaptive (256-512 units based on dataset)

**Multi-agent:**
- Algorithm: MADDPG/MASAC (BenchMARL)
- Training iterations: 60 (from terminal logs)
- Uses BenchMARL's default hyperparameters

**Potential issues:**
- Multi-agent might need more training iterations
- Learning rates might not be optimal for multi-agent setting
- Network architectures might need adjustment

## Recommendations

### Immediate Actions (High Priority)

1. **Verify Coverage Target is Passed Correctly**
   ```python
   # In BenchMARL/environment.py, add logging:
   logger.info(f"coverage_target from config: {env_config.get('coverage_target', 'NOT SET')}")
   logger.info(f"coverage_target used: {self.coverage_target}")
   ```

2. **Reduce Agents Per Class for Testing**
   - Change `agents_per_class: 3` → `agents_per_class: 1` in `BenchMARL/conf/anchor.yaml`
   - This makes multi-agent equivalent to single-agent in terms of coordination complexity
   - If this improves performance, the issue is coordination-related

3. **Increase Training Iterations**
   - Multi-agent training might need more iterations to converge
   - Current: 60 iterations
   - Try: 100-200 iterations

4. **Check Shared Reward Computation**
   - Verify `_compute_shared_reward()` is working correctly
   - Add logging to see if shared rewards are being computed and applied
   - Ensure shared reward is significant enough to encourage cooperation

### Medium Priority

5. **Tune Reward Weights**
   - Increase `shared_reward_weight` from 0.5 to 0.7-0.8
   - Increase `class_union_cov_weight` from 0.03 to 0.05-0.1
   - Reduce `inter_class_overlap_weight` if it's too punitive

6. **Add Reward Normalization**
   - Multi-agent rewards might have different scales than single-agent
   - Consider normalizing rewards to similar ranges

7. **Improve Training Stability**
   - Add gradient clipping
   - Use learning rate scheduling
   - Increase batch size if possible
   - Use more stable optimizers (e.g., AdamW instead of Adam)

### Long-term Improvements

8. **Curriculum Learning**
   - Start with `agents_per_class: 1` and gradually increase
   - Or start with easier tasks and increase difficulty

9. **Better Coordination Mechanisms**
   - Use communication between agents
   - Implement hierarchical policies
   - Use centralized training with decentralized execution (CTDE) more effectively

10. **Hyperparameter Search**
    - Perform systematic hyperparameter tuning for multi-agent setting
    - Focus on: learning rates, reward weights, network architectures

## Diagnostic Commands

### Check if coverage_target is being passed:
```bash
grep -r "coverage_target" BenchMARL/conf/anchor.yaml
grep -r "coverage_target" BenchMARL/environment.py
```

### Check training logs for reward values:
```bash
# Look for reward values in training logs
grep "total_reward\|shared_reward\|class_union" <training_log_file>
```

### Compare NashConv convergence:
```bash
# Check if NashConv is converging
grep "NashConv metrics" <training_log_file> | tail -20
```

## Expected Outcomes After Fixes

If fixes are successful, we should see:
- **Precision**: 0.85-0.95 (stable, low variance)
- **Coverage**: 0.01-0.05 (improved from 0.002)
- **Error bars**: Much smaller (indicating stability)
- **Equilibrium**: >50% (at least 1/2 classes meet targets)

## Next Steps

1. Start with reducing `agents_per_class` to 1 and verify coverage_target
2. If performance improves, gradually increase agents_per_class
3. If performance doesn't improve, focus on reward structure and training stability
4. Consider running hyperparameter search for multi-agent setting

