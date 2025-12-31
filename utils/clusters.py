import numpy as np
import torch
from typing import Dict, Optional, Tuple, Any

# Try to import sklearn for clustering (optional dependency)
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ============================================================================
# Helper Functions: Clustering
# ============================================================================

def compute_cluster_centroids_per_class(
    X_unit: np.ndarray,
    y: np.ndarray,
    n_clusters_per_class: int = 10,
    random_state: int = 42,
    min_samples_per_cluster: int = 1,
    auto_adapt_clusters: bool = True,
    check_data_scatter: bool = True
) -> Dict[int, np.ndarray]:
    """
    Compute cluster centroids for each class using KMeans clustering.
    
    This identifies dense regions (clusters) in the data and returns their centroids.
    Starting episodes from cluster centroids can improve training by:
    - Starting from more representative/typical examples
    - Reducing variance by focusing on dense regions
    - Potentially finding better anchors that cover more points
    
    IMPORTANT: This is ONLY for class-based training. For instance-based training,
    set x_star_unit explicitly on the AnchorEnv before calling reset().
    
    Args:
        X_unit: Data in unit space [0, 1], shape (n_samples, n_features)
        y: Class labels, shape (n_samples,)
        n_clusters_per_class: Number of clusters to find per class (max, will be adapted for small datasets)
        random_state: Random seed for reproducibility
        min_samples_per_cluster: Minimum samples required to form a cluster
        auto_adapt_clusters: If True, adapt cluster count based on dataset size (recommended)
        check_data_scatter: If True, check if data is scattered and use mean if so (recommended)
        
    Returns:
        Dictionary mapping class -> array of cluster centroids (n_clusters, n_features)
        
    Note:
        Cluster centroids are used when x_star_unit is None (class-based training).
        For instance-based training, set x_star_unit directly on AnchorEnv.
        
        Adaptive clustering:
        - Small datasets (< 20 samples): Use mean centroid only
        - Small-medium (20-50): Use 1-2 clusters
        - Medium (50-200): Use sqrt(n_samples) or n_samples/20
        - Large (> 200): Use requested number (default 10)
        
        Scatter detection:
        - If data is uniformly distributed (low variance), uses mean centroid
        - Prevents meaningless clustering on scattered data
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "sklearn is required for cluster-based sampling. "
            "Install it with: pip install scikit-learn"
        )
    
    centroids_per_class = {}
    unique_classes = np.unique(y)
    
    for cls in unique_classes:
        cls_mask = (y == cls)
        cls_indices = np.where(cls_mask)[0]
        X_cls = X_unit[cls_indices]
        
        if len(X_cls) == 0:
            centroids_per_class[cls] = np.array([]).reshape(0, X_unit.shape[1])
            continue
        
        # Data validation: Check for NaN, Inf, or invalid values
        if np.any(np.isnan(X_cls)) or np.any(np.isinf(X_cls)):
            import warnings
            warnings.warn(
                f"Class {cls}: Found NaN or Inf values in data. "
                f"Using mean centroid as fallback."
            )
            # Use mean as fallback
            centroid = np.nanmean(X_cls, axis=0, keepdims=True)
            # Replace any remaining NaN/Inf with 0.5 (middle of [0,1] range)
            centroid = np.nan_to_num(centroid, nan=0.5, posinf=1.0, neginf=0.0)
            centroids_per_class[cls] = centroid.astype(np.float32)
            continue
        
        # Ensure data is in valid range [0, 1] and clip if necessary
        # This prevents numerical issues in k-means
        X_cls = np.clip(X_cls, 0.0, 1.0).astype(np.float32)
        
        # Check for constant features (zero variance) - these can cause k-means issues
        feature_vars = np.var(X_cls, axis=0)
        constant_features = feature_vars == 0
        
        if np.any(constant_features):
            # For constant features, use the constant value
            # This is safe and won't cause k-means issues
            pass  # KMeans can handle constant features, but we'll use a more robust approach
        
        # ADAPTIVE CLUSTER COUNT: Adjust based on dataset size
        # For small datasets, using too many clusters doesn't make sense
        # For scattered data, clustering may not be meaningful
        n_samples = len(X_cls)
        
        if auto_adapt_clusters:
            # Adaptive cluster count based on dataset size
            if n_samples < 20:
                # Very small dataset: use mean centroid only
                n_clusters = 1
            elif n_samples < 50:
                # Small dataset: use 1-2 clusters
                n_clusters = min(2, n_clusters_per_class)
            elif n_samples < 100:
                # Medium-small: use sqrt(n_samples) or max 5 clusters
                n_clusters = min(int(np.sqrt(n_samples)), 5, n_clusters_per_class)
            elif n_samples < 200:
                # Medium: use n_samples/20 or max 10 clusters
                n_clusters = min(max(5, n_samples // 20), n_clusters_per_class)
            else:
                # Large dataset: use requested number (default 10)
                n_clusters = n_clusters_per_class
        else:
            # Use requested number, but cap at number of samples
            n_clusters = min(n_clusters_per_class, n_samples)
        
        # Ensure we have enough samples per cluster
        n_clusters = min(n_clusters, n_samples // max(1, min_samples_per_cluster))
        n_clusters = max(1, n_clusters)  # At least 1 cluster
        
        # CHECK DATA SCATTER: If data is uniformly distributed, clustering may not be meaningful
        # Use coefficient of variation of distances to centroid as a proxy for scatter
        use_clustering = True
        if check_data_scatter and n_clusters > 1 and n_samples >= 10:
            try:
                # Quick check: compute mean centroid and check variance of distances
                mean_centroid = X_cls.mean(axis=0, keepdims=True)
                distances_to_mean = np.linalg.norm(X_cls - mean_centroid, axis=1)
                cv_distances = np.std(distances_to_mean) / (np.mean(distances_to_mean) + 1e-10)
                
                # If coefficient of variation is very low (< 0.1), data is very uniform/scattered
                # This suggests clustering may not find meaningful structure
                if cv_distances < 0.1:
                    import warnings
                    warnings.warn(
                        f"Class {cls}: Data appears uniformly distributed (CV={cv_distances:.4f} < 0.1). "
                        f"Clustering may not be meaningful. Using mean centroid instead."
                    )
                    use_clustering = False
            except Exception:
                # If scatter check fails, proceed with clustering
                pass
        
        if n_clusters < min_samples_per_cluster or not use_clustering:
            # Not enough samples for clustering, use mean as single centroid
            centroid = X_cls.mean(axis=0, keepdims=True)
            centroids_per_class[cls] = centroid.astype(np.float32)
        else:
            # Perform KMeans clustering with error handling
            try:
                import warnings
                # Suppress sklearn warnings during k-means fitting
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    warnings.filterwarnings("ignore", message=".*divide by zero.*")
                    warnings.filterwarnings("ignore", message=".*overflow.*")
                    warnings.filterwarnings("ignore", message=".*invalid value.*")
                    
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        random_state=random_state,
                        n_init=10,
                        max_iter=300,
                        algorithm='lloyd'  # Use Lloyd algorithm (more stable)
                    )
                    kmeans.fit(X_cls)
                    centroids = kmeans.cluster_centers_.astype(np.float32)
                    
                    # Validate centroids (check for NaN/Inf)
                    if np.any(np.isnan(centroids)) or np.any(np.isinf(centroids)):
                        # Fallback to mean if k-means produced invalid centroids
                        warnings.warn(
                            f"Class {cls}: K-means produced invalid centroids. "
                            f"Using mean centroid as fallback."
                        )
                        centroid = X_cls.mean(axis=0, keepdims=True)
                        centroids_per_class[cls] = centroid.astype(np.float32)
                    else:
                        centroids_per_class[cls] = centroids
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Class {cls}: K-means clustering failed: {e}. "
                    f"Using mean centroid as fallback."
                )
                # Fallback to mean centroid
                centroid = X_cls.mean(axis=0, keepdims=True)
                centroids_per_class[cls] = centroid.astype(np.float32)
    
    return centroids_per_class