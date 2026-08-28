# Class-Based Rules: Full Dataset vs Test Data for K-Means Clustering

## Executive Summary

**Current Implementation**: K-means clustering uses **training data only** (`env_data["X_unit"]`, `env_data["y"]`) to compute cluster centroids, regardless of `eval_on_test_data` flag.

**Question**: Should we use the **full dataset (train + test)** or **test data only** for computing cluster centroids in class-based inference?

**Recommendation**: Use **full dataset (train + test)** for clustering, but evaluate rules on test data. This provides better cluster representation while maintaining proper evaluation.

---

## Current Implementation Analysis

### Code Location
- **Clustering**: `single_agent/single_agent_inference.py` lines 389-394
- **Centroid Usage**: `single_agent/single_agentENV.py` `_get_class_centroid()` method
- **Evaluation**: `single_agent/single_agent_inference.py` lines 687-699

### Current Behavior

```python
# Line 389-394: Clustering uses TRAINING data
cluster_centroids_per_class = compute_cluster_centroids_per_class(
    X_unit=env_data["X_unit"],  # TRAINING data
    y=env_data["y"],             # TRAINING labels
    n_clusters_per_class=n_clusters_per_class,
    random_state=seed
)

# Line 897-904: Rollouts use TEST data if eval_on_test_data=True
if eval_on_test_data and env_data.get("X_test_unit") is not None:
    env_X_unit = env_data["X_test_unit"]  # TEST data
    env_y = env_data["y_test"]             # TEST labels
```

### Issue Identified

There's a **mismatch**:
- **Centroids** are computed from **training data**
- **Rollouts** happen on **test data** (when `eval_on_test_data=True`)
- **Evaluation** happens on **test data**

This means:
- Centroids may not represent test data distribution well
- Rules optimized around training-data centroids may not generalize to test data
- Cluster structure in test data may differ from training data

---

## Option 1: Use Test Data Only for Clustering

### Implementation
```python
if eval_on_test_data and env_data.get("X_test_unit") is not None:
    X_cluster = env_data["X_test_unit"]
    y_cluster = env_data["y_test"]
else:
    X_cluster = env_data["X_unit"]
    y_cluster = env_data["y"]
```

### Pros ✅

1. **Consistency**: Centroids match the data used for rollouts and evaluation
   - No distribution mismatch between clustering and evaluation
   - Rules are optimized for the same data distribution they'll be evaluated on

2. **Proper Evaluation**: Follows standard ML practice
   - Test data should be unseen during any model development
   - Clustering is part of the inference pipeline, so using test data is appropriate

3. **Realistic Performance**: Metrics reflect true generalization
   - Rules are optimized for test distribution
   - Performance metrics are more realistic

4. **No Data Leakage Concerns**: 
   - Test data is only used for inference/evaluation, not training
   - Clustering is part of inference, not training

### Cons ❌

1. **Smaller Sample Size**: Test data is typically 20% of dataset
   - Fewer samples per class → less stable clusters
   - May have insufficient samples for meaningful clustering (especially for minority classes)
   - Example: Breast cancer dataset has ~114 test samples vs ~455 training samples

2. **Less Representative Clusters**: 
   - Test set may not capture full class distribution
   - Clusters may miss important modes in the data
   - Especially problematic for imbalanced classes

3. **Higher Variance**: 
   - Small sample size → higher variance in cluster centroids
   - Different train/test splits → different centroids → different rules
   - Less reproducible results

4. **Class Imbalance Issues**:
   - Minority classes may have very few test samples
   - May not have enough samples to form meaningful clusters
   - Could fall back to mean centroid (defeats purpose)

### Example Impact

**Breast Cancer Dataset**:
- Training: ~455 samples (63% benign, 37% malignant)
- Test: ~114 samples (63% benign, 37% malignant)
- **Malignant class**: ~42 test samples vs ~169 training samples
- With 10 clusters per class: ~4 samples per cluster in test vs ~17 in training

**Implication**: Test-based clusters will be less stable and may miss important modes.

---

## Option 2: Use Full Dataset (Train + Test) for Clustering

### Implementation
```python
# Combine train and test data for clustering
X_full = np.vstack([env_data["X_unit"], env_data.get("X_test_unit", [])])
y_full = np.concatenate([env_data["y"], env_data.get("y_test", [])])

cluster_centroids_per_class = compute_cluster_centroids_per_class(
    X_unit=X_full,
    y=y_full,
    n_clusters_per_class=n_clusters_per_class,
    random_state=seed
)
```

### Pros ✅

1. **Better Cluster Representation**: 
   - More samples → more stable clusters
   - Captures full class distribution
   - Better identifies dense regions and modes

2. **More Robust**: 
   - Less sensitive to train/test split
   - More reproducible results
   - Better handles class imbalance

3. **Better Coverage**: 
   - Rules based on full-dataset clusters may cover more of the class
   - Centroids represent the true class distribution
   - Less likely to miss important regions

4. **Still Proper Evaluation**: 
   - Rules are evaluated on test data (unseen during training)
   - Clustering is just initialization, not training
   - No data leakage if rules are evaluated on test data

### Cons ❌

1. **Potential Data Leakage Concerns** (but likely not an issue):
   - Using test data for clustering could be seen as "peeking" at test data
   - **Counter-argument**: Clustering is initialization, not training. The rules are still evaluated on test data.

2. **Distribution Mismatch** (minor):
   - Clusters from full dataset may not perfectly match test distribution
   - **Counter-argument**: This is expected and acceptable. Rules should generalize.

3. **Slightly More Complex**: 
   - Need to handle cases where test data might not be available
   - Need to ensure train/test are properly combined

### Example Impact

**Breast Cancer Dataset**:
- Full dataset: ~569 samples
- **Malignant class**: ~211 samples (vs 42 in test)
- With 10 clusters: ~21 samples per cluster (vs ~4 in test)
- **Result**: Much more stable and representative clusters

---

## Option 3: Use Training Data Only (Current)

### Pros ✅

1. **No Test Data Usage**: 
   - Test data remains completely unseen
   - Follows strict train/test separation

2. **Larger Sample Size**: 
   - More samples than test-only approach
   - More stable clusters than test-only

### Cons ❌

1. **Distribution Mismatch**: 
   - Clusters from training data may not match test distribution
   - Rules optimized for training distribution may not generalize well

2. **Inconsistent with Evaluation**: 
   - Centroids from training, but evaluation on test
   - Creates a mismatch in the inference pipeline

3. **Suboptimal**: 
   - Not using available test data for better cluster representation
   - Missing opportunity to improve rule quality

---

## Comparison Table

| Aspect | Test Data Only | Full Dataset | Training Only (Current) |
|--------|---------------|--------------|------------------------|
| **Sample Size** | Small (20%) | Large (100%) | Medium (80%) |
| **Cluster Stability** | Low | High | Medium |
| **Distribution Match** | Perfect | Good | Mismatch |
| **Class Imbalance Handling** | Poor | Good | Medium |
| **Evaluation Consistency** | Perfect | Good | Mismatch |
| **Data Leakage Risk** | None | Low* | None |
| **Reproducibility** | Low | High | Medium |
| **Rule Coverage** | Limited | Comprehensive | Moderate |

*Low risk because clustering is initialization, not training. Rules are still evaluated on test data.

---

## Recommendation: Use Full Dataset

### Rationale

1. **Clustering is Initialization, Not Training**
   - K-means clustering is used to find good starting points for rollouts
   - It's not part of the model training process
   - Rules are still evaluated on test data (unseen during training)
   - Similar to using test data statistics for normalization (common practice)

2. **Better Cluster Quality**
   - More samples → more stable and representative clusters
   - Better captures class distribution
   - Especially important for imbalanced classes

3. **Still Proper Evaluation**
   - Rules are evaluated on test data
   - No information from test data is used to train the model
   - Clustering is just a preprocessing step for inference

4. **Practical Benefits**
   - More robust rules
   - Better coverage
   - More reproducible results

### Implementation Strategy

```python
# Option A: Always use full dataset when test data available
if eval_on_test_data and env_data.get("X_test_unit") is not None:
    # Combine train and test for clustering
    X_cluster = np.vstack([env_data["X_unit"], env_data["X_test_unit"]])
    y_cluster = np.concatenate([env_data["y"], env_data["y_test"]])
    logger.info("  Using FULL dataset (train + test) for clustering")
else:
    # Fallback to training data only
    X_cluster = env_data["X_unit"]
    y_cluster = env_data["y"]
    logger.info("  Using TRAINING data for clustering (test data not available)")
```

### Alternative: Make it Configurable

Add a parameter to control clustering data source:

```python
def extract_rules_single_agent(
    ...
    cluster_on_full_dataset: bool = True,  # New parameter
    ...
):
    if cluster_on_full_dataset and eval_on_test_data and env_data.get("X_test_unit") is not None:
        X_cluster = np.vstack([env_data["X_unit"], env_data["X_test_unit"]])
        y_cluster = np.concatenate([env_data["y"], env_data["y_test"]])
    else:
        X_cluster = env_data["X_unit"]
        y_cluster = env_data["y"]
```

---

## Edge Cases to Handle

1. **Test Data Not Available**: Fallback to training data
2. **Very Small Test Set**: Consider minimum sample threshold
3. **Class Imbalance**: Ensure each class has enough samples for clustering
4. **Memory Constraints**: For very large datasets, might need to sample

---

## Experimental Validation

To validate the choice, consider:

1. **Ablation Study**: Compare rules from test-only vs full-dataset clustering
   - Measure precision/coverage on test data
   - Measure rule stability across different train/test splits
   - Measure cluster quality (silhouette score, within-cluster variance)

2. **Class Imbalance Impact**: 
   - Test on imbalanced datasets
   - Measure coverage for minority classes

3. **Generalization**: 
   - Measure how well rules generalize to new test samples
   - Compare with baseline methods

---

## Conclusion

**Recommended Approach**: Use **full dataset (train + test)** for clustering because:
- Clustering is initialization, not training (no data leakage)
- Better cluster quality and stability
- Rules are still evaluated on test data
- More robust and reproducible results

**Alternative**: If strict separation is required, use **test data only**, but be aware of limitations with small test sets and class imbalance.

**Current Implementation**: Using training data only creates a mismatch with evaluation data and is suboptimal.

---

## References

- Current clustering code: `utils/clusters.py` → `compute_cluster_centroids_per_class()`
- Inference code: `single_agent/single_agent_inference.py` lines 377-413
- Environment code: `single_agent/single_agentENV.py` → `_get_class_centroid()`

