# NashConv-Based Model Selection Implementation

## Summary

This document describes the implementation of NashConv-based model selection for multi-agent training. NashConv (Nash Convergence) measures how close a multi-agent policy is to Nash equilibrium, and is now used as a criterion for selecting best models during training.

---

## Changes Made

### 1. Configuration File (`BenchMARL/conf/anchor.yaml`)

**Added**:
```yaml
# NashConv threshold for model selection (ε-Nash equilibrium)
# Models with NashConv <= this threshold are considered close to Nash equilibrium
# Lower values = stricter equilibrium requirement (default: 0.01 = 1% exploitability)
nashconv_threshold: 0.01  # ε-Nash threshold for best model selection
```

**Location**: Line 72-75

**Purpose**: Configurable threshold for ε-Nash equilibrium. Models with NashConv ≤ threshold are preferred.

---

### 2. Callback Initialization (`BenchMARL/benchmarl_wrappers.py`)

**Added Parameters**:
- `nashconv_threshold: float = 0.01` to `__init__()` method

**Added Tracking Variables**:
- `self.best_equilibrium_nashconv = float('inf')` - Tracks best NashConv for equilibrium models
- `self.best_eval_nashconv = float('inf')` - Tracks best NashConv for aggregate models

**Location**: Lines 138, 175, 183

---

### 3. Model Selection Logic (`BenchMARL/benchmarl_wrappers.py`)

#### Strategy 1: Equilibrium-Based (Enhanced with NashConv)

**Previous Behavior**:
- Saved model if all classes meet targets and `min_class_score > best_equilibrium_score`

**New Behavior**:
1. **If NashConv ≤ threshold**:
   - Saves model if score improved OR NashConv improved (when scores are similar)
   - Logs NashConv value

2. **If NashConv > threshold**:
   - Only saves if NashConv improved AND no good equilibrium model exists yet
   - Helps track progress toward equilibrium

**Code Location**: Lines 1648-1697

**Key Logic**:
```python
if nashconv_sum <= self.nashconv_threshold:
    # NashConv acceptable - save if score improved or NashConv improved
    score_improved = min_class_score > self.best_equilibrium_score
    nashconv_improved = nashconv_sum < self.best_equilibrium_nashconv
    
    if score_improved or (scores_similar and nashconv_improved):
        save_model = True
        # ... save model
```

#### Strategy 2: Global Aggregate (Enhanced with NashConv Tiebreaker)

**Previous Behavior**:
- Saved model if `eval_score > best_eval_score`

**New Behavior**:
1. **Primary**: Save if score improved (unchanged)
2. **Tiebreaker**: If scores are similar (within 0.01), prefer model with lower NashConv

**Code Location**: Lines 1699-1720

**Key Logic**:
```python
if score_improved:
    save_model = True
elif scores_similar and nashconv_improved:
    # Tiebreaker: prefer lower NashConv
    save_model = True
    save_reason = "aggregate_nashconv"
```

---

### 4. Logging Updates (`BenchMARL/benchmarl_wrappers.py`)

**Enhanced Log Messages**:
- Include NashConv values in all best model save messages
- Show NashConv threshold when saving equilibrium models
- Indicate when NashConv tiebreaker is used

**Examples**:
```
✓ New best model saved (EQUILIBRIUM)!
  All classes meet targets.
  Global: P=0.96, C=0.28, NashConv: 0.008
  Class 0: P=0.98, C=0.30, Score=1.28
  Class 1: P=0.95, C=0.26, Score=1.21
```

```
✓ New best model saved (NashConv tiebreaker)!
  Score: 1.20 (similar to 1.19), NashConv: 0.005 (improved from 0.012)
```

**Location**: Lines 1722-1758

---

### 5. Trainer Integration (`BenchMARL/anchor_trainer.py`)

**Updated Callback Instantiation**:
```python
# Get NashConv threshold from config (default: 0.01)
nashconv_threshold = env_config.get("nashconv_threshold", 0.01)

self.callback = AnchorMetricsCallback(
    log_training_metrics=True, 
    log_evaluation_metrics=True,
    save_to_file=True,
    collect_anchor_data=True,
    nashconv_threshold=nashconv_threshold
)
```

**Location**: Lines 313-319

**Purpose**: Reads `nashconv_threshold` from config file and passes it to callback.

---

## How It Works

### Model Selection Flow

1. **During Evaluation**:
   - NashConv is computed (if `compute_nashconv=True`)
   - NashConv metrics are added to `aggregated` dictionary

2. **Equilibrium Check**:
   - If all classes meet targets:
     - Check if NashConv ≤ threshold
     - If yes: Save if score improved or NashConv improved
     - If no: Only save if NashConv improved and no good model exists yet

3. **Aggregate Check** (if equilibrium not reached):
   - Save if score improved
   - OR if scores similar and NashConv improved (tiebreaker)

### Example Scenarios

#### Scenario 1: Equilibrium with Good NashConv

**Model A**:
- All classes meet targets: ✓
- Min class score: 1.20
- NashConv: 0.008 (≤ 0.01 threshold)
- **Result**: Saved (equilibrium with good NashConv)

**Model B** (later):
- All classes meet targets: ✓
- Min class score: 1.22
- NashConv: 0.015 (> 0.01 threshold)
- **Result**: Not saved (NashConv too high, even though score is better)

#### Scenario 2: NashConv Tiebreaker

**Model A**:
- Score: 1.20
- NashConv: 0.012
- **Result**: Saved (best score so far)

**Model B** (later):
- Score: 1.21 (similar to 1.20, within 0.01)
- NashConv: 0.005 (better than 0.012)
- **Result**: Saved (NashConv tiebreaker - similar score but better equilibrium)

---

## Configuration

### Default Values

- **`nashconv_threshold`**: 0.01 (1% exploitability)
  - This is a reasonable ε-Nash threshold for most applications
  - Lower values (e.g., 0.001) = stricter equilibrium requirement
  - Higher values (e.g., 0.05) = looser requirement

### How to Change

**Option 1: Edit Config File**
```yaml
# In BenchMARL/conf/anchor.yaml
env_config:
  nashconv_threshold: 0.005  # Stricter (0.5% exploitability)
```

**Option 2: Pass via Code**
```python
env_config = {
    "nashconv_threshold": 0.005,
    # ... other config
}
trainer.setup_experiment(env_config=env_config)
```

---

## Benefits

1. **Ensures Equilibrium**: Models selected are closer to Nash equilibrium
2. **Better Multi-Agent Solutions**: Prefers stable, coordinated strategies
3. **Configurable**: Threshold can be adjusted based on requirements
4. **Backward Compatible**: Falls back gracefully if NashConv not available
5. **Transparent**: Logs show NashConv values and selection reasons

---

## Testing

### Verification Steps

1. **Check Config**: Verify `nashconv_threshold` is in config file
2. **Check Logs**: Look for NashConv values in best model save messages
3. **Check Behavior**: 
   - Models with NashConv ≤ threshold should be preferred
   - Models with similar scores but lower NashConv should be preferred

### Expected Log Output

```
Evaluation - Precision: 0.9600, Coverage: 0.2800 (n=10)
  ✓ EQUILIBRIUM: All 3 classes meet targets!
  ✓ New equilibrium checkpoint! Min class score: 1.2000, NashConv: 0.008000 (≤ 0.010)
  ✓ New best model saved (EQUILIBRIUM)!
    All classes meet targets.
    Global: P=0.9600, C=0.2800, NashConv: 0.008000
    Class 0: P=0.9800, C=0.3000, Score=1.2800
    Class 1: P=0.9500, C=0.2600, Score=1.2100
    Class 2: P=0.9600, C=0.2800, Score=1.2400
    Best model path: .../best_model/best_checkpoint.pt
```

---

## Future Improvements

1. **Per-Class NashConv**: Consider per-class exploitability in selection
2. **Adaptive Threshold**: Adjust threshold based on training progress
3. **NashConv Weighting**: Weight NashConv vs score in selection (currently equal priority)
4. **Visualization**: Plot NashConv trends in model selection decisions

---

## Related Documentation

- `docs/MODEL_SELECTION_REVIEW.md` - Original review and recommendations
- `docs/NASHCONV_INTERPRETATION.md` - Understanding NashConv metrics
- `docs/NASHCONV_FREQUENCY_GUIDE.md` - NashConv computation frequency

---

## Conclusion

NashConv-based model selection ensures that multi-agent training selects models that are both high-performing (precision+coverage) AND close to Nash equilibrium (low NashConv). This leads to more stable, coordinated multi-agent solutions.

