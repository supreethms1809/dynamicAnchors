# Data Preprocessing

## Overview

This document describes the data preprocessing pipeline used in the Dynamic Anchors system, including normalization, standardization, and train/test splitting.

---

## Preprocessing Pipeline

### Step 1: Dataset Loading

**Location**: `BenchMARL/tabular_datasets.py` `TabularDatasetLoader`

**Process**:
1. Load raw dataset (CSV, sklearn datasets, etc.)
2. Extract features and labels
3. Handle missing values (if any)
4. Encode categorical features (if needed)

**Supported Datasets**:
- Breast Cancer (Wisconsin)
- Iris
- Wine
- Housing (Boston)
- Circles (synthetic)
- Moons (synthetic)
- Folktables (income, employment)
- UCI Credit Card

**Code Reference**: `BenchMARL/tabular_datasets.py` `load_dataset()` method

---

### Step 2: Train/Test Split

**Method**: Stratified random split

**Parameters**:
- `test_size`: 0.2 (20% test, 80% train)
- `random_state`: 42 (for reproducibility)
- `stratify`: True (maintains class distribution)

**Formula**:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
```

**Purpose**: 
- Ensures consistent evaluation
- Maintains class balance in train/test sets
- Reproducible splits

**Code Reference**: `BenchMARL/tabular_datasets.py` lines 74-397

---

### Step 3: Standardization

**Method**: StandardScaler (sklearn)

**Formula**:
```
X_scaled = (X - mean(X)) / std(X)
```

**Process**:
1. Fit scaler on training data: `scaler.fit(X_train)`
2. Transform training data: `X_train_scaled = scaler.transform(X_train)`
3. Transform test data: `X_test_scaled = scaler.transform(X_test)`

**Result**:
- Mean = 0
- Standard deviation = 1
- Preserves relative relationships

**Purpose**:
- Normalizes feature scales
- Required for neural network classifiers
- Prevents features with large scales from dominating

**Code Reference**: `BenchMARL/tabular_datasets.py` lines 399-426

---

### Step 4: Normalization to [0, 1]

**Method**: Min-Max normalization

**Formula**:
```
X_min = min(X_scaled, axis=0)
X_max = max(X_scaled, axis=0)
X_range = X_max - X_min  # Handle zero range: set to 1.0
X_unit = (X_scaled - X_min) / X_range
X_unit = clip(X_unit, 0.0, 1.0)
```

**Process**:
1. Compute min/max from training data
2. Compute range (handle zero range)
3. Normalize to [0, 1]
4. Clip to ensure bounds

**Result**:
- All features in [0, 1] range
- Required for anchor box bounds
- Consistent scale across features

**Purpose**:
- Anchor boxes operate in [0, 1] space
- Simplifies box initialization and updates
- Enables consistent step sizes

**Code Reference**: `BenchMARL/tabular_datasets.py` lines 406-414

---

## Dual Representation

### Why Two Representations?

**Standardized (`X_std`)**:
- Used for classifier input
- Mean=0, Std=1
- Better for neural network training

**Normalized (`X_unit`)**:
- Used for anchor box bounds
- Range [0, 1]
- Required for environment operations

**Conversion**:
- `X_unit` can be converted back to `X_std` using `X_min` and `X_range`
- `X_std` can be converted to `X_unit` using normalization

**Code Reference**: `BenchMARL/environment.py` `_normalize_data()` method

---

## Preprocessing Parameters

### Stored Values

**`X_min`**: Minimum values per feature (from standardized data)  
**`X_max`**: Maximum values per feature (from standardized data)  
**`X_range`**: Range per feature (handles zero range)  
**`scaler`**: Fitted StandardScaler object

**Purpose**: 
- Used for denormalization during inference
- Converts anchor boxes from [0,1] back to original scale
- Required for rule extraction

---

## Handling Edge Cases

### Zero Range Features

**Problem**: Some features may have zero range (constant values)

**Solution**:
```python
X_range = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))
```

**Result**: Sets range to 1.0 for constant features (prevents division by zero)

### Out-of-Range Values

**Problem**: Test data may have values outside training range

**Solution**: Clipping to [0, 1] after normalization

**Code**:
```python
X_unit = np.clip(X_unit, 0.0, 1.0)
```

---

## Data Statistics

### Logged Information

**After Preprocessing**:
- Dataset shape (train/test)
- Feature names
- Class names and distribution
- Normalized range: `[min, max]` per dataset

**Example Output**:
```
Data preprocessing complete:
  Scaled train shape: (455, 30)
  Scaled test shape: (114, 30)
  Unit train range: [0.000, 1.000]
  Unit test range: [0.000, 1.000]
```

---

## Classifier Training Data

### Preprocessing for Classifier

**Input**: `X_train_scaled` (standardized)  
**Output**: Trained classifier

**Process**:
1. Train classifier on standardized training data
2. Evaluate on standardized test data
3. Save classifier for use in environment

**Purpose**: Classifier must use standardized data (matches training)

---

## Environment Data

### Preprocessing for Environment

**Inputs**:
- `X_unit`: Normalized to [0, 1] (for box bounds)
- `X_std`: Standardized (for classifier evaluation)
- `y`: Class labels

**Process**:
1. Environment uses `X_unit` for box operations
2. When evaluating precision/coverage:
   - Samples in box (from `X_unit`)
   - Convert to `X_std` for classifier
   - Get predictions from classifier

**Code Reference**: `BenchMARL/environment.py` `_current_metrics()` method

---

## Inference-Time Preprocessing

### Rule Extraction

**Process**:
1. Anchor boxes are in [0, 1] space
2. Denormalize to standardized space:
   ```python
   X_std = X_unit * X_range + X_min
   ```
3. Format rules in standardized space (more interpretable)

**Purpose**: Rules are more interpretable in original scale

**Code Reference**: `BenchMARL/inference.py` rule extraction functions

---

## Code References

- **Dataset Loading**: `BenchMARL/tabular_datasets.py` `load_dataset()`
- **Preprocessing**: `BenchMARL/tabular_datasets.py` `preprocess_data()`
- **Normalization**: `BenchMARL/environment.py` `_normalize_data()`
- **Denormalization**: Used in inference for rule extraction

---

## Summary

**Preprocessing Steps**:

1. **Load Dataset**: Raw features and labels
2. **Split**: 80/20 train/test (stratified)
3. **Standardize**: Mean=0, Std=1 (for classifier)
4. **Normalize**: Range [0, 1] (for anchor boxes)

**Key Points**:

- **Dual representation**: Standardized for classifier, normalized for boxes
- **Reproducibility**: Fixed random seed (42)
- **Stratification**: Maintains class balance
- **Edge cases**: Handles zero range, out-of-range values

**For Paper Writing**:

- Explain why dual representation is needed
- Describe standardization for classifier
- Describe normalization for anchor boxes
- Mention train/test split strategy
- Discuss handling of edge cases

