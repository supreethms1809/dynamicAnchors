# NashConv Computation Frequency Guide

## Overview

The `nashconv_compute_frequency` parameter controls how often NashConv is computed during training. Since NashConv computation is computationally expensive, you need to balance:

1. **Tracking convergence**: Need enough data points to see trends
2. **Computational cost**: Each computation takes time and resources
3. **Log clarity**: Too frequent = log spam, too sparse = missing trends

## Recommended Values

### By Training Length

| Training Length | Iterations | Recommended Frequency | Rationale |
|----------------|------------|----------------------|-----------|
| **Short/Testing** | < 20 | `2-3` batches | More frequent for visibility during quick tests |
| **Medium** | 20-100 | `5-10` batches | Balanced - good coverage without excessive cost |
| **Long** | 100-500 | `10-15` batches | Less frequent, still sufficient data points |
| **Very Long** | > 500 | `15-20` batches | Minimal frequency, focus on convergence trends |

### By Training Duration

| Expected Duration | Recommended Frequency | Notes |
|------------------|----------------------|-------|
| < 1 hour | `2-5` batches | Can afford more frequent computation |
| 1-4 hours | `5-10` batches | Standard production setting |
| 4-12 hours | `10-15` batches | Balance between tracking and cost |
| > 12 hours | `15-20` batches | Minimize overhead, track major trends |

## Example Calculations

### Example 1: Medium Training (60 iterations)
- **Frequency = 10**: 6 NashConv computations
  - Data points at: batches 10, 20, 30, 40, 50, 60
  - Good coverage, reasonable cost

### Example 2: Long Training (200 iterations)
- **Frequency = 10**: 20 NashConv computations
  - Data points at: batches 10, 20, 30, ..., 200
  - Good tracking, moderate cost

- **Frequency = 20**: 10 NashConv computations
  - Data points at: batches 20, 40, 60, ..., 200
  - Less frequent, lower cost, still sufficient

### Example 3: Very Long Training (1000 iterations)
- **Frequency = 20**: 50 NashConv computations
  - Good coverage, reasonable cost
- **Frequency = 50**: 20 NashConv computations
  - Sparse but tracks major convergence milestones

## Configuration

### Default Value
The default is set to **`10`** batches, which works well for most production training scenarios.

### How to Change

#### Option 1: In Code (AnchorTrainer)
```python
callback = AnchorMetricsCallback(
    compute_nashconv=True,
    nashconv_compute_frequency=10  # Change this value
)
```

#### Option 2: Via Config File
You can add this to your experiment configuration and pass it through to the callback.

## Computational Cost Considerations

**NashConv computation involves:**
1. Extracting observations and actions from batches
2. Computing best responses for each agent (gradient ascent or random search)
3. Evaluating Q-values using the critic network
4. Aggregating metrics

**Estimated overhead per computation:**
- Small environment (3 agents, 4 features): ~0.1-0.5 seconds
- Medium environment (10 agents, 20 features): ~0.5-2 seconds
- Large environment (50+ agents, 100+ features): ~2-10 seconds

**Total overhead for training:**
- Frequency = 2: ~5-10% overhead (very frequent)
- Frequency = 10: ~1-2% overhead (recommended)
- Frequency = 20: ~0.5-1% overhead (minimal)

## Best Practices

1. **Start with default (10)**: Works well for most cases
2. **Reduce for testing (2-3)**: Better visibility during short test runs
3. **Increase for very long training (15-20)**: Minimize overhead
4. **Monitor trends**: If you see smooth convergence, you can increase frequency
5. **Check logs**: If logs are too verbose, increase frequency

## Example: Tracking Convergence

With `frequency = 10` over 100 iterations, you'll get:
```
Batch 10:  NashConv = 0.025
Batch 20:  NashConv = 0.020
Batch 30:  NashConv = 0.018
Batch 40:  NashConv = 0.015
Batch 50:  NashConv = 0.012
Batch 60:  NashConv = 0.010
Batch 70:  NashConv = 0.009
Batch 80:  NashConv = 0.008
Batch 90:  NashConv = 0.007
Batch 100: NashConv = 0.006
```

This gives you **10 data points** to visualize convergence, which is sufficient for most analysis.

## When to Adjust

**Decrease frequency (compute more often) if:**
- Training is short and you want detailed tracking
- You're debugging convergence issues
- You have excess computational resources

**Increase frequency (compute less often) if:**
- Training is very long (> 500 iterations)
- Computational resources are limited
- Logs are becoming too verbose
- You're confident convergence is stable

## Summary

**Recommended default: `10` batches**

This provides:
- ✅ Good balance between tracking and cost
- ✅ Sufficient data points for convergence analysis
- ✅ Reasonable computational overhead (~1-2%)
- ✅ Works well for most training scenarios

Adjust based on your specific needs:
- **Testing**: `2-3`
- **Production**: `10` (default)
- **Very long training**: `15-20`
