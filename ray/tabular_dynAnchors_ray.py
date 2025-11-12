"""
Complete Dynamic Anchors pipeline for tabular data using Ray RLlib (SAC for continuous actions).

This module provides an end-to-end pipeline for training and evaluating
dynamic anchor explanations on tabular classification data using Ray RLlib.

Post-hoc training only: Train classifier first, then train RL policy.

Usage example:
    from ray.tabular_dynAnchors_ray import train_and_evaluate_dynamic_anchors_ray
    
    results = train_and_evaluate_dynamic_anchors_ray(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        classifier=classifier,
        target_classes=(0, 1, 2),
        total_timesteps=50000,
    )
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
import os
import json
from functools import partial


def train_and_evaluate_dynamic_anchors_ray(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    classifier: torch.nn.Module,
    target_classes: Tuple[int, ...] = None,
    device: str = "cpu",
    # Training parameters
    total_timesteps: int = 50000,
    learning_rate: float = 3e-4,
    # Environment parameters
    use_perturbation: bool = True,
    perturbation_mode: str = "adaptive",  # "bootstrap", "uniform", or "adaptive"
    n_perturb: int = 2048,
    step_fracs: Tuple[float, ...] = (0.005, 0.01, 0.02),
    min_width: float = 0.05,
    precision_target: float = 0.95,
    coverage_target: float = 0.02,
    # Evaluation parameters
    n_eval_instances_per_class: int = 20,
    max_features_in_rule: Optional[int] = 5,
    steps_per_episode: int = 100,
    eval_steps_per_episode: int = None,
    num_rollouts_per_instance: int = 1,
    # Training sampling parameters
    n_clusters_per_class: Optional[int] = None,
    n_fixed_instances_per_class: Optional[int] = None,
    use_random_sampling: bool = False,
    # Output parameters
    output_dir: str = "./output/anchors_ray/",
    save_checkpoints: bool = True,
    checkpoint_freq: int = 10000,
    verbose: int = 1,
    # Ray RLlib specific parameters
    num_workers: int = 0,  # Number of parallel workers (0 = single process)
    num_gpus: int = 0,  # Number of GPUs to use
    num_cpus_per_worker: int = 1,
    num_envs_per_env_runner: int = 1,  # Number of parallel environments per worker (increase for GPU)
    # SAC specific parameters
    tau: float = 0.005,  # Soft update coefficient
    target_network_update_freq: int = 1,
    buffer_size: int = 1000000,
    learning_starts: int = 1000,
    train_batch_size: int = 256,  # Will be auto-increased for GPU if None
    target_entropy: Optional[float] = None,  # Auto if None
) -> Dict[str, Any]:
    """
    Complete pipeline for training and evaluating dynamic anchors using Ray RLlib SAC.
    
    This function:
    1. Prepares data and creates environments
    2. Trains RL policy (SAC) to generate anchors
    3. Evaluates on test instances to compute precision/coverage
    4. Returns results and trained models
    
    Args:
        X_train: Training features (will be standardized)
        y_train: Training labels
        X_test: Test features (for evaluation)
        y_test: Test labels
        feature_names: Names of features
        classifier: Trained PyTorch classifier
        target_classes: Classes to generate anchors for (None = all classes)
        device: Device to use ("cpu", "cuda", "auto")
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for SAC
        use_perturbation: Enable perturbation sampling in environment
        perturbation_mode: "bootstrap", "uniform", or "adaptive" sampling
        n_perturb: Number of perturbation samples
        step_fracs: Action step sizes (for continuous actions, max step_fracs is used)
        min_width: Minimum box width
        precision_target: Target precision threshold
        coverage_target: Target coverage threshold
        n_eval_instances_per_class: Instances per class for evaluation
        max_features_in_rule: Max features to show in rules (-1 or None = all)
        steps_per_episode: Max steps for greedy rollouts
        eval_steps_per_episode: Steps per episode for evaluation (defaults to steps_per_episode)
        num_rollouts_per_instance: Number of greedy rollouts per instance
        n_clusters_per_class: Number of cluster centroids per class (None = use all instances)
        n_fixed_instances_per_class: Number of fixed instances per class for fallback
        use_random_sampling: If True, randomly sample from pool each episode
        output_dir: Directory for outputs
        save_checkpoints: Save checkpoints during training
        checkpoint_freq: Checkpoint frequency
        verbose: Verbosity level
        num_workers: Number of Ray workers (0 = single process)
        num_gpus: Number of GPUs to use (will auto-optimize batch size and parallel envs)
        num_cpus_per_worker: CPUs per worker
        num_envs_per_env_runner: Number of parallel environments per worker (auto-optimized for GPU)
        tau: Soft update coefficient for SAC
        target_network_update_freq: Frequency of target network updates
        buffer_size: Replay buffer size
        learning_starts: Steps before learning starts
        train_batch_size: Batch size for training (auto-increased for GPU if < 512)
        target_entropy: Target entropy for SAC (auto if None)
    
    Returns:
        Dictionary with:
            - trained_model: Dict of Ray RLlib SAC trainers per class
            - eval_results: Per-class evaluation results
            - overall_stats: Overall precision/coverage
            - metadata: Configuration and setup info
    """
    try:
        import ray
        from ray.rllib.algorithms.sac import SACConfig
        from ray.rllib.algorithms.callbacks import DefaultCallbacks
    except ImportError as e:
        raise ImportError(
            "Ray RLlib is not installed. Please install it with: "
            "pip install 'ray[rllib]'"
        ) from e
    
    from sklearn.preprocessing import StandardScaler
    
    # Handle imports when running as script vs module
    try:
        # Try absolute import (when running as module: python -m ray.tabular_dynAnchors_ray)
        from ray.ray_modules_standalone import (
            AnchorEnv, 
            ContinuousAnchorEnv, 
            compute_cluster_centroids_per_class,
            evaluate_all_classes, 
            evaluate_all_classes_class_level,
            get_device_pair
        )
    except ImportError:
        # Fallback to relative import (when running directly: python ray/tabular_dynAnchors_ray.py)
        from ray_modules_standalone import (
            AnchorEnv, 
            ContinuousAnchorEnv, 
            compute_cluster_centroids_per_class,
            evaluate_all_classes, 
            evaluate_all_classes_class_level,
            get_device_pair
        )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Standardize device handling
    device_obj, device_str = get_device_pair(device)
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            num_cpus=num_workers * num_cpus_per_worker + 1,
            num_gpus=num_gpus,
            ignore_reinit_error=True,
        )
    
    # Prepare data
    print("Preparing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # Normalize to [0,1] for environment
    X_min = X_train_scaled.min(axis=0)
    X_max = X_train_scaled.max(axis=0)
    X_range = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))
    X_unit_train = (X_train_scaled - X_min) / X_range
    X_unit_test = (X_test_scaled - X_min) / X_range
    
    # Prepare test data in unit space
    X_test_unit = (X_test_scaled - X_min) / X_range
    
    # Determine target classes
    unique_classes = np.unique(y_train)
    if target_classes is None:
        target_classes = tuple(unique_classes)
    else:
        target_classes = tuple(target_classes)
    
    print(f"Classes: {unique_classes}, Target classes: {target_classes}")
    
    # ======================================================================
    # CLUSTER-BASED SAMPLING: Compute cluster centroids per class
    # ======================================================================
    cluster_centroids_per_class = None
    if n_clusters_per_class is None:
        print(f"\n[Cluster-Based Sampling] Using all training instances per class (n_clusters_per_class=None)...")
    else:
        print(f"\n[Cluster-Based Sampling] Computing {n_clusters_per_class} cluster centroids per class...")
        try:
            cluster_centroids_per_class = compute_cluster_centroids_per_class(
                X_unit=X_unit_train,
                y=y_train,
                n_clusters_per_class=n_clusters_per_class,
                random_state=42
            )
            print(f"  Cluster centroids computed successfully!")
            for cls in target_classes:
                if cls in cluster_centroids_per_class:
                    n_centroids = len(cluster_centroids_per_class[cls])
                    print(f"  Class {cls}: {n_centroids} centroids")
        except Exception as e:
            print(f"  WARNING: Could not compute cluster centroids: {e}")
            print(f"  Falling back to random sampling.")
            cluster_centroids_per_class = None
    
    # Fixed instance sampling (fallback)
    fixed_instances_per_class = None
    if n_fixed_instances_per_class is not None:
        print(f"\n[Fixed Instance Sampling] Selecting {n_fixed_instances_per_class} fixed instances per class...")
        fixed_instances_per_class = {}
        for cls in target_classes:
            cls_indices = np.where(y_train == cls)[0]
            if len(cls_indices) > 0:
                n_fixed = min(n_fixed_instances_per_class, len(cls_indices))
                rng = np.random.default_rng(42 + cls)
                fixed_indices = rng.choice(cls_indices, size=n_fixed, replace=False)
                fixed_instances_per_class[cls] = fixed_indices.tolist()
                print(f"  Class {cls}: {len(fixed_indices)} fixed instances")
    
    # Create environment factory function
    def create_anchor_env(target_cls=None, use_test_centroids=False):
        """Helper to create AnchorEnv with a specific target class."""
        if target_cls is None:
            target_cls = target_classes[0]
        
        env = AnchorEnv(
            X_unit=X_unit_train,
            X_std=X_train_scaled,
            y=y_train,
            feature_names=feature_names,
            classifier=classifier,
            device=device_str,
            target_class=target_cls,
            step_fracs=step_fracs,
            min_width=min_width,
            alpha=0.7,
            beta=0.6,
            gamma=0.1,
            precision_target=precision_target,
            coverage_target=coverage_target,
            precision_blend_lambda=0.5,
            drift_penalty_weight=0.05,
            use_perturbation=use_perturbation,
            perturbation_mode=perturbation_mode,
            n_perturb=n_perturb,
            X_min=X_min,
            X_range=X_range,
            min_coverage_floor=0.005,
            js_penalty_weight=0.05,
        )
        
        # Set cluster centroids
        if use_test_centroids and test_cluster_centroids_per_class is not None:
            env.cluster_centroids_per_class = test_cluster_centroids_per_class
        elif cluster_centroids_per_class is not None:
            env.cluster_centroids_per_class = cluster_centroids_per_class
        
        env.fixed_instances_per_class = fixed_instances_per_class
        env.use_random_sampling = use_random_sampling
        
        # Enable continuous actions
        env.max_action_scale = max(step_fracs) if step_fracs else 0.02
        env.min_absolute_step = max(0.05, min_width * 0.5)
        
        return env
    
    # Custom callback to track precision and coverage metrics
    class PrecisionCoverageCallback(DefaultCallbacks):
        """
        Custom callback to track precision and coverage metrics from environment info dict.
        
        Ray RLlib doesn't automatically aggregate custom metrics from the environment's info dict,
        so this callback extracts precision and coverage from each episode's info and aggregates them.
        """
        # Class-level storage to share metrics across callback instances (workers vs driver)
        _shared_precisions = []
        _shared_coverages = []
        _lock = None
        _metrics_file = None  # File-based storage for cross-process sharing
        
        def __init__(self):
            super().__init__()
            self.episode_precisions = []
            self.episode_coverages = []
            # Track which episodes we've already stored metrics for (to avoid duplicates)
            self.processed_episodes = set()
            
            # Initialize lock for thread-safe access (if needed)
            if PrecisionCoverageCallback._lock is None:
                import threading
                PrecisionCoverageCallback._lock = threading.Lock()
            
            # Initialize metrics file for cross-process sharing (fallback)
            if PrecisionCoverageCallback._metrics_file is None:
                import tempfile
                PrecisionCoverageCallback._metrics_file = os.path.join(
                    tempfile.gettempdir(), 
                    f"ray_metrics_{os.getpid()}.json"
                )
        
        def on_episode_step(self, *, episode, env_index, env_runner=None, metrics_logger=None, env=None, rl_module=None, worker=None, base_env=None, policies=None, **kwargs):
            """Called at each step of an episode."""
            # Extract precision and coverage from info dict if available
            # Since episodes rarely reach done=True (they're truncated at max_steps), we need to extract
            # metrics from every step, not just wait for episode end
            try:
                info = None
                
                # Method 1: Try get_infos() first (most reliable for new API stack)
                if hasattr(episode, 'get_infos'):
                    try:
                        infos = episode.get_infos()
                        if infos:
                            # Get last info dict (most recent step)
                            info = infos[-1] if isinstance(infos, list) else infos
                    except Exception:
                        pass
                
                # Method 2: Try last_info_for with agent_id (for single-agent, use default agent)
                if info is None and hasattr(episode, 'last_info_for'):
                    try:
                        # Try with agent_id first (for multi-agent compatibility)
                        try:
                            agent_id = episode.agent_id_for(env_index) if hasattr(episode, 'agent_id_for') else None
                            if agent_id is not None:
                                info = episode.last_info_for(agent_id)
                            else:
                                info = episode.last_info_for(env_index)
                        except (TypeError, AttributeError, KeyError):
                            # Try without arguments (single-agent)
                            try:
                                info = episode.last_info_for()
                            except Exception:
                                # Try with default agent ID
                                try:
                                    info = episode.last_info_for("default_agent")
                                except Exception:
                                    pass
                    except Exception:
                        pass
                
                # Method 2: Try accessing from episode's observations/steps (new API stack)
                if info is None and hasattr(episode, 'get_observations'):
                    try:
                        obs_data = episode.get_observations()
                        if obs_data is not None:
                            # Check if info is stored in observations
                            if isinstance(obs_data, dict):
                                if 'infos' in obs_data:
                                    infos = obs_data['infos']
                                    if infos and len(infos) > 0:
                                        info = infos[-1] if isinstance(infos, list) else infos
                            elif isinstance(obs_data, list) and len(obs_data) > 0:
                                # If observations is a list, try to get info from last element
                                last_obs = obs_data[-1]
                                if isinstance(last_obs, dict) and 'info' in last_obs:
                                    info = last_obs['info']
                    except Exception:
                        pass
                
                # Method 3: Try accessing from episode's steps directly
                if info is None and hasattr(episode, 'get_steps'):
                    try:
                        steps = episode.get_steps()
                        if steps and len(steps) > 0:
                            last_step = steps[-1]
                            if isinstance(last_step, dict) and 'info' in last_step:
                                info = last_step['info']
                    except Exception:
                        pass
                
                # Method 4: Try accessing via env_runner (new API stack)
                if info is None and env_runner is not None:
                    try:
                        # Check if env_runner has access to last step info
                        if hasattr(env_runner, 'last_info'):
                            info = env_runner.last_info
                        elif hasattr(env_runner, '_last_info'):
                            info = env_runner._last_info
                        # Try accessing from env_runner's episode
                        elif hasattr(env_runner, 'episode') and env_runner.episode is not None:
                            if hasattr(env_runner.episode, 'last_info_for'):
                                try:
                                    info = env_runner.episode.last_info_for()
                                except Exception:
                                    pass
                    except Exception:
                        pass
                
                # Method 5: Try accessing directly from base_env (contains actual environment instances)
                if info is None and base_env is not None:
                    try:
                        # base_env might be a vectorized env or have envs attribute
                        if hasattr(base_env, 'envs') and len(base_env.envs) > env_index:
                            actual_env = base_env.envs[env_index]
                            # Unwrap if it's wrapped (e.g., by Ray's wrappers)
                            while hasattr(actual_env, 'env') or hasattr(actual_env, 'gym_env'):
                                actual_env = getattr(actual_env, 'env', None) or getattr(actual_env, 'gym_env', None)
                                if actual_env is None:
                                    break
                            
                            # Try to get metrics from ContinuousAnchorEnv
                            if hasattr(actual_env, 'anchor_env'):
                                anchor_env = actual_env.anchor_env
                                if hasattr(anchor_env, '_current_metrics'):
                                    prec, cov, _ = anchor_env._current_metrics()
                                    info = {"precision": prec, "coverage": cov}
                        elif hasattr(base_env, 'get_unwrapped'):
                            # Try to get unwrapped environment
                            unwrapped = base_env.get_unwrapped()
                            if unwrapped and len(unwrapped) > env_index:
                                actual_env = unwrapped[env_index]
                                if hasattr(actual_env, 'anchor_env'):
                                    anchor_env = actual_env.anchor_env
                                    if hasattr(anchor_env, '_current_metrics'):
                                        prec, cov, _ = anchor_env._current_metrics()
                                        info = {"precision": prec, "coverage": cov}
                    except Exception:
                        pass
                
                # Method 6: Try accessing directly from env parameter (if available)
                if info is None and env is not None:
                    try:
                        # If env is ContinuousAnchorEnv, try to get info from anchor_env
                        if hasattr(env, 'anchor_env'):
                            anchor_env = env.anchor_env
                            if hasattr(anchor_env, '_current_metrics'):
                                prec, cov, _ = anchor_env._current_metrics()
                                info = {"precision": prec, "coverage": cov}
                    except Exception:
                        pass
                
                # Extract precision and coverage from info if found
                # Only store metrics at the last step of each episode (to avoid duplicates)
                # Check if this is the last step (episode is done or truncated)
                is_last_step = False
                if hasattr(episode, 'is_done'):
                    # is_done might be a property or method
                    try:
                        is_last_step = episode.is_done() if callable(episode.is_done) else episode.is_done
                    except (TypeError, AttributeError):
                        pass
                if not is_last_step and hasattr(episode, 'is_terminated'):
                    try:
                        is_last_step = episode.is_terminated() if callable(episode.is_terminated) else episode.is_terminated
                    except (TypeError, AttributeError):
                        pass
                if not is_last_step and hasattr(episode, 'is_truncated'):
                    try:
                        is_last_step = episode.is_truncated() if callable(episode.is_truncated) else episode.is_truncated
                    except (TypeError, AttributeError):
                        pass
                # Also check episode length to detect truncation
                if not is_last_step and hasattr(episode, 'len'):
                    # If episode length is close to max_steps, it might be truncated
                    # We'll store metrics at every step for now, but deduplicate in on_sample_end
                    is_last_step = False  # Store at every step, deduplicate later
                
                if info is not None and isinstance(info, dict):
                    precision = info.get("precision", None)
                    coverage = info.get("coverage", None)
                    
                    # Store in episode custom_metrics for later retrieval
                    if precision is not None:
                        if not hasattr(episode, 'custom_metrics'):
                            episode.custom_metrics = {}
                        episode.custom_metrics["precision"] = float(precision)
                    if coverage is not None:
                        if not hasattr(episode, 'custom_metrics'):
                            episode.custom_metrics = {}
                        episode.custom_metrics["coverage"] = float(coverage)
                    
                    # Store metrics at last step or every step (we'll deduplicate in on_sample_end)
                    # Using episode ID to avoid duplicates
                    episode_id = id(episode)  # Use object ID as unique identifier
                    if episode_id not in self.processed_episodes or is_last_step:
                        if precision is not None:
                            self.episode_precisions.append(float(precision))
                        if coverage is not None:
                            self.episode_coverages.append(float(coverage))
                        if is_last_step:
                            self.processed_episodes.add(episode_id)
                
                # Debug: Print what we found (only first few times to avoid spam)
                if not hasattr(self, '_step_debug_count'):
                    self._step_debug_count = 0
                if self._step_debug_count < 3:
                    print(f"[DEBUG on_episode_step] info found: {info is not None}")
                    if info is not None:
                        print(f"  Info keys: {list(info.keys()) if isinstance(info, dict) else 'N/A'}")
                        print(f"  Precision: {info.get('precision', 'NOT FOUND') if isinstance(info, dict) else 'N/A'}")
                        print(f"  Coverage: {info.get('coverage', 'NOT FOUND') if isinstance(info, dict) else 'N/A'}")
                    else:
                        print(f"  Episode type: {type(episode)}")
                        print(f"  Episode has last_info_for: {hasattr(episode, 'last_info_for')}")
                        print(f"  base_env type: {type(base_env) if base_env is not None else None}")
                    self._step_debug_count += 1
                    
            except Exception as e:
                # If info extraction fails, print error for debugging (only first few times)
                if not hasattr(self, '_step_error_count'):
                    self._step_error_count = 0
                if self._step_error_count < 2:
                    print(f"[DEBUG on_episode_step] Error: {e}")
                    import traceback
                    traceback.print_exc()
                    self._step_error_count += 1
        
        def on_episode_end(self, *, episode, env_index, env_runner=None, metrics_logger=None, env=None, rl_module=None, worker=None, base_env=None, policies=None, **kwargs):
            """Called at the end of each episode."""
            # Extract final precision and coverage from episode's custom metrics
            precision = None
            coverage = None
            
            # First try to get from custom_metrics (set during on_episode_step)
            if hasattr(episode, 'custom_metrics'):
                precision = episode.custom_metrics.get("precision", None)
                coverage = episode.custom_metrics.get("coverage", None)
            
            # Also try to get from episode infos directly
            if (precision is None or coverage is None) and hasattr(episode, 'get_infos'):
                try:
                    infos = episode.get_infos()
                    if infos and len(infos) > 0:
                        last_info = infos[-1] if isinstance(infos, list) else infos
                        if isinstance(last_info, dict):
                            if precision is None:
                                precision = last_info.get("precision", None)
                            if coverage is None:
                                coverage = last_info.get("coverage", None)
                except Exception:
                    pass
            
            # Store metrics in episode for aggregation
            # Also store in instance lists and shared storage
            if precision is not None:
                self.episode_precisions.append(float(precision))
                try:
                    with PrecisionCoverageCallback._lock:
                        PrecisionCoverageCallback._shared_precisions.append(float(precision))
                    self._write_metrics_to_file(float(precision), None)
                except Exception:
                    pass
            if coverage is not None:
                self.episode_coverages.append(float(coverage))
                try:
                    with PrecisionCoverageCallback._lock:
                        PrecisionCoverageCallback._shared_coverages.append(float(coverage))
                    self._write_metrics_to_file(None, float(coverage))
                except Exception:
                    pass
        
        def _write_metrics_to_file(self, precision=None, coverage=None):
            """Write metrics to file for cross-process sharing."""
            try:
                import json
                metrics_file = PrecisionCoverageCallback._metrics_file
                if metrics_file:
                    # Read existing metrics
                    precisions = []
                    coverages = []
                    if os.path.exists(metrics_file):
                        try:
                            with open(metrics_file, 'r') as f:
                                data = json.load(f)
                                precisions = data.get("precisions", [])
                                coverages = data.get("coverages", [])
                        except Exception:
                            pass
                    
                    # Add new metrics
                    if precision is not None:
                        precisions.append(float(precision))
                    if coverage is not None:
                        coverages.append(float(coverage))
                    
                    # Write back
                    with open(metrics_file, 'w') as f:
                        json.dump({"precisions": precisions, "coverages": coverages}, f)
            except Exception:
                pass
        
        def _read_metrics_from_file(self):
            """Read metrics from file for cross-process sharing."""
            try:
                import json
                metrics_file = PrecisionCoverageCallback._metrics_file
                if metrics_file and os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        precisions = data.get("precisions", [])
                        coverages = data.get("coverages", [])
                        # Clear file after reading
                        with open(metrics_file, 'w') as f:
                            json.dump({"precisions": [], "coverages": []}, f)
                        return precisions, coverages
            except Exception:
                pass
            return [], []
        
        def on_sample_end(self, *, env_runner=None, metrics_logger=None, samples=None, worker=None, **kwargs):
            """Called after sampling is complete. Extract info from sample batches."""
            # In Ray RLlib's new API stack, samples is a list of episodes
            if samples is None:
                return  # No samples to process
            
            try:
                # samples is typically a list of Episode objects in the new API stack
                if isinstance(samples, list):
                    for episode in samples:
                        # Use episode ID to avoid duplicate extraction
                        episode_id = id(episode)
                        if episode_id in self.processed_episodes:
                            continue  # Already processed this episode
                        
                        precision = None
                        coverage = None
                        
                        # Method 1: Try get_infos() (most reliable for new API stack)
                        if hasattr(episode, 'get_infos'):
                            try:
                                # Get all info dicts from episode
                                infos = episode.get_infos()
                                if infos:
                                    # Get last info dict (most recent step) - this captures truncated episodes
                                    last_info = infos[-1] if isinstance(infos, list) else infos
                                    if isinstance(last_info, dict):
                                        precision = last_info.get("precision", None)
                                        coverage = last_info.get("coverage", None)
                            except Exception:
                                pass
                        
                        # Method 2: Try custom_metrics (set during on_episode_step)
                        if (precision is None or coverage is None) and hasattr(episode, 'custom_metrics'):
                            if precision is None:
                                precision = episode.custom_metrics.get("precision", None)
                            if coverage is None:
                                coverage = episode.custom_metrics.get("coverage", None)
                        
                        # Method 3: Try last_info_for as fallback
                        if (precision is None or coverage is None) and hasattr(episode, 'last_info_for'):
                            try:
                                info = episode.last_info_for()
                                if isinstance(info, dict):
                                    if precision is None:
                                        precision = info.get("precision", None)
                                    if coverage is None:
                                        coverage = info.get("coverage", None)
                            except Exception:
                                pass
                        
                        # Method 4: Try accessing internal episode data (fallback)
                        if (precision is None or coverage is None) and hasattr(episode, '_episode'):
                            try:
                                internal_episode = episode._episode
                                if hasattr(internal_episode, 'infos') and internal_episode.infos:
                                    last_info = internal_episode.infos[-1] if isinstance(internal_episode.infos, list) else internal_episode.infos
                                    if isinstance(last_info, dict):
                                        if precision is None:
                                            precision = last_info.get("precision", None)
                                        if coverage is None:
                                            coverage = last_info.get("coverage", None)
                            except Exception:
                                pass
                        
                        # Store metrics if found (after trying all methods)
                        if precision is not None or coverage is not None:
                            if precision is not None:
                                self.episode_precisions.append(float(precision))
                                # Also store in shared storage for cross-instance access
                                try:
                                    with PrecisionCoverageCallback._lock:
                                        PrecisionCoverageCallback._shared_precisions.append(float(precision))
                                    # Also write to file for cross-process sharing
                                    self._write_metrics_to_file(float(precision), None)
                                except Exception:
                                    pass
                            if coverage is not None:
                                self.episode_coverages.append(float(coverage))
                                # Also store in shared storage for cross-instance access
                                try:
                                    with PrecisionCoverageCallback._lock:
                                        PrecisionCoverageCallback._shared_coverages.append(float(coverage))
                                    # Also write to file for cross-process sharing
                                    self._write_metrics_to_file(None, float(coverage))
                                except Exception:
                                    pass
                            # Mark episode as processed
                            self.processed_episodes.add(episode_id)
                            
                            # Debug: print first few extractions
                            if not hasattr(self, '_extraction_debug_count'):
                                self._extraction_debug_count = 0
                            if self._extraction_debug_count < 5:
                                print(f"[DEBUG on_sample_end] Extracted precision: {precision}, coverage: {coverage}")
                                print(f"  Total precisions collected: {len(self.episode_precisions)}")
                                print(f"  Total coverages collected: {len(self.episode_coverages)}")
                                self._extraction_debug_count += 1
                
                # If samples is a SampleBatch or similar object
                elif hasattr(samples, 'infos'):
                    # Extract from SampleBatch.infos (list of info dicts)
                    if samples.infos is not None:
                        for info in samples.infos:
                            if isinstance(info, dict):
                                precision = info.get("precision", None)
                                coverage = info.get("coverage", None)
                                if precision is not None:
                                    self.episode_precisions.append(float(precision))
                                if coverage is not None:
                                    self.episode_coverages.append(float(coverage))
                
                # Debug: Print sample structure (only first time)
                if not hasattr(self, '_sample_debug_count'):
                    self._sample_debug_count = 0
                if self._sample_debug_count < 2:
                    print(f"[DEBUG on_sample_end] samples type: {type(samples)}")
                    if isinstance(samples, list) and len(samples) > 0:
                        print(f"  First episode type: {type(samples[0])}")
                        print(f"  First episode has get_infos: {hasattr(samples[0], 'get_infos')}")
                        print(f"  First episode has last_info_for: {hasattr(samples[0], 'last_info_for')}")
                        if hasattr(samples[0], 'get_infos'):
                            try:
                                infos = samples[0].get_infos()
                                print(f"  get_infos() returned: {type(infos)}, length: {len(infos) if isinstance(infos, list) else 'N/A'}")
                                if infos and len(infos) > 0:
                                    last_info = infos[-1] if isinstance(infos, list) else infos
                                    print(f"  Last info keys: {list(last_info.keys()) if isinstance(last_info, dict) else 'N/A'}")
                            except Exception as e:
                                print(f"  get_infos() error: {e}")
                    self._sample_debug_count += 1
                    
            except Exception as e:
                # If extraction fails, print error for debugging
                if not hasattr(self, '_sample_error_count'):
                    self._sample_error_count = 0
                if self._sample_error_count < 2:
                    print(f"[DEBUG on_sample_end] Error extracting metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    self._sample_error_count += 1
        
        def on_train_result(self, *, algorithm, result, **kwargs):
            """Called at the end of each training iteration."""
            # Ensure custom_metrics dict exists
            if "custom_metrics" not in result:
                result["custom_metrics"] = {}
            
            # Extract metrics from shared storage (works across callback instances)
            precision_mean = 0.0
            coverage_mean = 0.0
            
            # Method 1: Aggregate from shared storage (works within same process)
            try:
                with PrecisionCoverageCallback._lock:
                    if PrecisionCoverageCallback._shared_precisions:
                        precision_mean = np.mean(PrecisionCoverageCallback._shared_precisions)
                        PrecisionCoverageCallback._shared_precisions = []  # Clear for next iteration
                    if PrecisionCoverageCallback._shared_coverages:
                        coverage_mean = np.mean(PrecisionCoverageCallback._shared_coverages)
                        PrecisionCoverageCallback._shared_coverages = []  # Clear for next iteration
            except Exception:
                pass
            
            # Method 2: Read from file (works across processes)
            if precision_mean == 0.0 or coverage_mean == 0.0:
                file_precisions, file_coverages = self._read_metrics_from_file()
                if file_precisions and precision_mean == 0.0:
                    precision_mean = np.mean(file_precisions)
                if file_coverages and coverage_mean == 0.0:
                    coverage_mean = np.mean(file_coverages)
            
            # Method 3: Fallback to instance lists (for same-process callbacks)
            if precision_mean == 0.0 and self.episode_precisions:
                precision_mean = np.mean(self.episode_precisions)
                self.episode_precisions = []
            if coverage_mean == 0.0 and self.episode_coverages:
                coverage_mean = np.mean(self.episode_coverages)
                self.episode_coverages = []
            
            # Store in result
            result["custom_metrics"]["precision_mean"] = float(precision_mean)
            result["custom_metrics"]["coverage_mean"] = float(coverage_mean)
            
            # Debug output
            if not hasattr(self, '_train_result_debug_count'):
                self._train_result_debug_count = 0
            if self._train_result_debug_count < 5:
                shared_prec_count = len(PrecisionCoverageCallback._shared_precisions) if PrecisionCoverageCallback._shared_precisions else 0
                shared_cov_count = len(PrecisionCoverageCallback._shared_coverages) if PrecisionCoverageCallback._shared_coverages else 0
                print(f"[DEBUG on_train_result] Final metrics:")
                print(f"  Shared precisions: {shared_prec_count}")
                print(f"  Shared coverages: {shared_cov_count}")
                print(f"  Instance precisions: {len(self.episode_precisions)}")
                print(f"  Instance coverages: {len(self.episode_coverages)}")
                print(f"  precision_mean: {precision_mean}")
                print(f"  coverage_mean: {coverage_mean}")
                if shared_prec_count > 0:
                    print(f"  Sample shared precisions: {PrecisionCoverageCallback._shared_precisions[:3]}")
                if shared_cov_count > 0:
                    print(f"  Sample shared coverages: {PrecisionCoverageCallback._shared_coverages[:3]}")
                self._train_result_debug_count += 1
            
            # Clear processed episodes set for next iteration
            self.processed_episodes.clear()
    
    # Train SAC model for each class
    print(f"\nTraining SAC for {len(target_classes)} classes...")
    
    sac_trainers = {}
    training_history = []
    per_class_precision_history = {cls: [] for cls in target_classes}
    per_class_coverage_history = {cls: [] for cls in target_classes}
    
    steps_per_class = total_timesteps // len(target_classes)
    
    for cls in target_classes:
        print(f"\n{'='*60}")
        print(f"Training SAC for class {cls} ({steps_per_class} timesteps)")
        print(f"{'='*60}")
        
        # Configure SAC according to Ray RLlib documentation
        # Reference: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#soft-actor-critic-sac
        # Create an environment class wrapper for Ray RLlib (required for new API stack)
        # Must inherit from gymnasium.Env for Ray RLlib to recognize it
        import gymnasium as gym
        
        class AnchorEnvWrapper(gym.Env):
            """Wrapper class for Ray RLlib environment registration."""
            def __init__(self, env_config=None):
                """Initialize environment with config."""
                super().__init__()
                if env_config is None:
                    env_config = {}
                anchor_env = create_anchor_env(target_cls=cls)
                worker_idx = env_config.get("worker_index", 0)
                self.gym_env = ContinuousAnchorEnv(anchor_env, seed=42 + cls + worker_idx)
                # Expose observation and action spaces
                self.observation_space = self.gym_env.observation_space
                self.action_space = self.gym_env.action_space
            
            def reset(self, seed=None, options=None):
                """Reset environment."""
                return self.gym_env.reset(seed=seed, options=options)
            
            def step(self, action):
                """Step environment."""
                return self.gym_env.step(action)
            
            def close(self):
                """Close environment."""
                return self.gym_env.close()
        
        # Create a test environment to get spaces for logging
        test_env = AnchorEnvWrapper()
        obs_space = test_env.observation_space
        action_space = test_env.action_space
        
        print(f"  Observation space: {obs_space}")
        print(f"  Action space: {action_space}")
        test_env.close()
        
        # Optimize batch size for GPU utilization
        # Larger batches better utilize GPU parallelism
        optimized_train_batch_size = train_batch_size
        if num_gpus > 0:
            # Increase batch size for GPU to better utilize compute
            # GPU can handle larger batches efficiently
            if train_batch_size < 512:
                optimized_train_batch_size = max(512, train_batch_size * 2)
                if verbose >= 1:
                    print(f"  Optimizing batch size for GPU: {train_batch_size} -> {optimized_train_batch_size}")
        
        # Optimize number of parallel environments for GPU
        # More parallel environments = more data collection = better GPU utilization
        optimized_num_envs = num_envs_per_env_runner
        if num_gpus > 0:
            # Increase parallel environments to keep GPU busy
            # With GPU, we can handle more parallel data collection
            if num_envs_per_env_runner < 4:
                optimized_num_envs = max(4, num_envs_per_env_runner * 2)
                if verbose >= 1:
                    print(f"  Optimizing parallel environments for GPU: {num_envs_per_env_runner} -> {optimized_num_envs}")
        
        config = SACConfig()
        config = config.environment(env=AnchorEnvWrapper)  # Pass class, not function
        # Add custom callback to track precision and coverage metrics
        config = config.callbacks(PrecisionCoverageCallback)
        config = config.training(
            actor_lr=learning_rate,
            critic_lr=learning_rate,
            tau=tau,
            target_network_update_freq=target_network_update_freq,
            num_steps_sampled_before_learning_starts=learning_starts,  # Correct parameter name
            train_batch_size=optimized_train_batch_size,
            target_entropy=target_entropy,
            replay_buffer_config={
                "capacity": buffer_size,
                "type": "EpisodeReplayBuffer",  # Required for new EnvRunner API
            },
        )
        config = config.env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=optimized_num_envs,  # Use optimized number of parallel environments
            num_cpus_per_env_runner=num_cpus_per_worker,
        )
        config = config.resources(
            num_gpus=num_gpus if num_workers == 0 else 0,  # Only use GPU for local training
        )
        # Configure GPU allocation for learners (new API stack)
        if num_gpus > 0 and num_workers == 0:
            config = config.learners(
                num_gpus_per_learner=1,  # Allocate GPU to learner
            )
        # Enable GPU for inference as well (not just training)
        # This allows GPU to be used for action computation during rollouts
        if num_gpus > 0:
            config = config.framework(
                framework="torch",
                # Ensure tensors are on GPU
                torch_compile_learner=False,  # Can enable for faster training if PyTorch 2.0+
            )
        else:
            config = config.framework(framework="torch")
        
        # Create SAC trainer (use build_algo instead of deprecated build)
        trainer = config.build_algo()
        
        # Training loop using Ray RLlib's train() method
        # Ray RLlib handles rollout collection internally
        # Account for parallel environments in iteration calculation
        effective_envs = optimized_num_envs * max(1, num_workers + 1)
        num_iterations = max(1, steps_per_class // (steps_per_episode * effective_envs))
        
        print(f"  Training for {num_iterations} iterations (target: {steps_per_class} timesteps)...")
        if num_gpus > 0:
            print(f"  GPU Optimization: {optimized_num_envs} parallel envs, batch size {optimized_train_batch_size}")
        
        for iteration in range(num_iterations):
            # Train one iteration (Ray handles rollout collection internally)
            result = trainer.train()
            
            # Extract metrics from training result
            timesteps = result.get("timesteps_total", 0)
            episode_reward_mean = result.get("episode_reward_mean", 0.0)
            episode_len_mean = result.get("episode_len_mean", 0)
            
            # Try to extract precision/coverage from custom metrics if available
            # Note: Ray RLlib doesn't automatically aggregate custom metrics from env info dict
            # These metrics are returned by the environment but need a callback to be tracked
            # For now, we'll default to 0.0 if not available
            custom_metrics = result.get("custom_metrics", {})
            episode_precision = custom_metrics.get("precision_mean", 0.0)
            episode_coverage = custom_metrics.get("coverage_mean", 0.0)
            
            # Also try checking in info dict (some Ray versions store metrics there)
            if episode_precision == 0.0 or episode_coverage == 0.0:
                info_dict = result.get("info", {})
                if isinstance(info_dict, dict):
                    # Check for metrics in various possible locations
                    if "precision_mean" in info_dict:
                        episode_precision = info_dict.get("precision_mean", episode_precision)
                    if "coverage_mean" in info_dict:
                        episode_coverage = info_dict.get("coverage_mean", episode_coverage)
            
            # Track episode
            training_history.append({
                "class": cls,
                "episode": iteration + 1,
                "timestep": timesteps,
                "reward": float(episode_reward_mean),
                "precision": float(episode_precision),
                "coverage": float(episode_coverage),
                "steps": int(episode_len_mean),
            })
            per_class_precision_history[cls].append(float(episode_precision))
            per_class_coverage_history[cls].append(float(episode_coverage))
            
            # Progress update
            if (iteration + 1) % max(1, num_iterations // 10) == 0 or iteration == num_iterations - 1:
                print(f"  Iteration {iteration + 1}/{num_iterations} | "
                      f"Timesteps: {timesteps} | "
                      f"Reward: {episode_reward_mean:.3f} | "
                      f"Precision: {episode_precision:.3f} | "
                      f"Coverage: {episode_coverage:.3f}")
            
            # Save checkpoint
            if save_checkpoints and (timesteps % checkpoint_freq == 0 or iteration == num_iterations - 1):
                checkpoint_dir = os.path.join(output_dir, "checkpoints", f"sac_class_{cls}_iter_{iteration}")
                checkpoint_dir = os.path.abspath(checkpoint_dir)  # Convert to absolute path
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = trainer.save(checkpoint_dir)
                if verbose >= 1:
                    print(f"  Saved checkpoint: {checkpoint_path}")
            
            # Stop if we've reached target timesteps
            if timesteps >= steps_per_class:
                print(f"  Reached target timesteps ({steps_per_class}), stopping training.")
                break
        
        sac_trainers[cls] = trainer
    
    # Compute test cluster centroids for evaluation
    print(f"\n[Class-Level Evaluation] Computing cluster centroids from test data...")
    test_cluster_centroids_per_class = None
    try:
        test_cluster_centroids_per_class = compute_cluster_centroids_per_class(
            X_unit=X_test_unit,
            y=y_test,
            n_clusters_per_class=10,
            random_state=42
        )
        print(f"  Test cluster centroids computed successfully!")
    except Exception as e:
        print(f"  WARNING: Could not compute test cluster centroids: {e}")
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    
    # Create wrapper class to make Ray RLlib SAC trainers compatible with evaluation functions
    class RaySACTrainerWrapper:
        """
        Wrapper to make Ray RLlib SAC trainers compatible with Stable-Baselines3 evaluation API.
        
        The evaluation functions expect models with a .predict() method that returns (action, state).
        Ray RLlib uses .compute_single_action() instead, so this wrapper bridges the gap.
        
        Also provides actor/critic attributes to be detected as a continuous action model.
        """
        def __init__(self, trainer):
            """
            Initialize wrapper with a Ray RLlib SAC trainer.
            
            Args:
                trainer: Ray RLlib SAC Algorithm instance
            """
            self.trainer = trainer
            # Set actor/critic attributes so evaluation functions detect this as continuous model
            # Try to access actual networks, but if that fails, set dummy attributes
            # (evaluation functions only check for existence, not actual usage)
            try:
                # New API stack: access via learner_group
                if hasattr(trainer, 'learner_group'):
                    learner_group = trainer.learner_group
                    if hasattr(learner_group, 'get_learner'):
                        learner = learner_group.get_learner()
                        if hasattr(learner, 'module'):
                            module = learner.module
                            # SAC module has actor and critic networks
                            if hasattr(module, 'actor'):
                                self.actor = module.actor
                            if hasattr(module, 'critic'):
                                self.critic = module.critic
            except Exception:
                pass
            
            # Always set actor/critic attributes (even if None) so evaluation functions
            # detect this as a continuous action model
            # The evaluation functions check: hasattr(trained_model, 'actor') and hasattr(trained_model, 'critic')
            if not hasattr(self, 'actor'):
                self.actor = None  # Dummy attribute to mark as continuous
            if not hasattr(self, 'critic'):
                self.critic = None  # Dummy attribute to mark as continuous
        
        def predict(self, obs, deterministic=False):
            """
            Predict action for given observation (compatible with SB3 API).
            
            Args:
                obs: Observation (numpy array)
                deterministic: If True, use deterministic policy (explore=False)
            
            Returns:
                Tuple of (action, None) to match SB3 API
            """
            # Ensure obs is numpy array
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().numpy()
            obs = np.array(obs, dtype=np.float32)
            
            # Use new API: get_module() and forward_inference() instead of deprecated compute_single_action()
            try:
                # Get the RL module (new API stack)
                module = self.trainer.get_module()
                
                # Convert observation to tensor and add batch dimension
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.from_numpy(obs).float()
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                
                # Add batch dimension if needed
                if len(obs_tensor.shape) == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                
                # Compute action using forward_inference
                with torch.no_grad():
                    # Forward inference returns action_dist_inputs
                    fwd_outputs = module.forward_inference({"obs": obs_tensor})
                    
                    # Get the action distribution class
                    action_dist_class = module.get_inference_action_dist_cls()
                    
                    # Create the action distribution from logits
                    action_dist = action_dist_class.from_logits(fwd_outputs["action_dist_inputs"])
                    
                    # For deterministic, use mean; for stochastic, sample
                    if deterministic:
                        # Use mean action (deterministic) - try mean() first, fallback to sample()
                        if hasattr(action_dist, 'mean'):
                            action = action_dist.mean()[0]
                        else:
                            # For TanhNormal, deterministic_sample() or mode() might be available
                            if hasattr(action_dist, 'deterministic_sample'):
                                action = action_dist.deterministic_sample()[0]
                            elif hasattr(action_dist, 'mode'):
                                action = action_dist.mode()[0]
                            else:
                                # Fallback: use sample (not ideal but works)
                                action = action_dist.sample()[0]
                    else:
                        # Sample from distribution (stochastic)
                        action = action_dist.sample()[0]
                    
                    # Convert to numpy
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    else:
                        action = np.array(action)
                
                # Ensure correct dtype and shape
                action = np.array(action, dtype=np.float32)
                if len(action.shape) > 1:
                    action = action.flatten()
                
            except Exception as e:
                # If new API fails, try alternative approach
                # Some versions might have different method names
                try:
                    # Try using compute_actions_from_input_dict (if available)
                    if hasattr(self.trainer, 'compute_actions_from_input_dict'):
                        obs_dict = {"obs": np.array(obs).reshape(1, -1)}
                        action_batch = self.trainer.compute_actions_from_input_dict(obs_dict, explore=not deterministic)
                        action = action_batch[0] if isinstance(action_batch, (list, tuple)) else action_batch
                        action = np.array(action, dtype=np.float32)
                    else:
                        raise RuntimeError(f"New API failed: {e}")
                except Exception as alt_error:
                    raise RuntimeError(
                        f"Failed to compute action using new API ({e}) and alternative ({alt_error})"
                    ) from alt_error
            
            return action, None
    
    # Create wrapped trainers dict for evaluation
    wrapped_trainers = {cls: RaySACTrainerWrapper(trainer) for cls, trainer in sac_trainers.items()}
    
    # Instance-level evaluation
    print(f"\n[Instance-Level Evaluation] Creating one anchor per test instance...")
    eval_steps = eval_steps_per_episode if eval_steps_per_episode is not None else steps_per_episode
    
    # Use wrapped trainers for evaluation (compatible with evaluate_all_classes)
    eval_results_instance = evaluate_all_classes(
        X_test=X_test_scaled,
        y_test=y_test,
        trained_model=wrapped_trainers,  # Pass dict of wrapped trainers
        make_env_fn=create_anchor_env,  # Use create_anchor_env which accepts target_cls
        feature_names=feature_names,
        n_instances_per_class=n_eval_instances_per_class,
        max_features_in_rule=max_features_in_rule,
        steps_per_episode=eval_steps,
        random_seed=42,
        eval_on_test_data=False,  # Default: use training data
        X_test_unit=X_test_unit if False else None,
        X_test_std=X_test_scaled if False else None,
        num_rollouts_per_instance=num_rollouts_per_instance,
    )
    
    # Class-level evaluation (one anchor per class)
    print(f"\n[Class-Level Evaluation] Creating one anchor per class (using test centroids)...")
    def create_anchor_env_for_class_eval(target_cls=None):
        return create_anchor_env(target_cls=target_cls, use_test_centroids=True)
    
    eval_results_class = evaluate_all_classes_class_level(
        trained_model=wrapped_trainers,  # Pass dict of wrapped trainers
        make_env_fn=create_anchor_env_for_class_eval,
        feature_names=feature_names,
        target_classes=list(target_classes),
        steps_per_episode=eval_steps,
        max_features_in_rule=max_features_in_rule,
        random_seed=42,
        eval_on_test_data=False,  # Default: use training data
        X_test_unit=X_test_unit if False else None,
        X_test_std=X_test_scaled if False else None,
        y_test=y_test if False else None,
    )
    
    # Prepare results
    results = {
        "trained_model": sac_trainers,
        "eval_results": {
            "instance_level": eval_results_instance,
            "class_level": eval_results_class,
        },
        "overall_stats": {
            "instance_level": {
                "avg_precision": eval_results_instance.get("overall_precision", 0.0),
                "avg_coverage": eval_results_instance.get("overall_coverage", 0.0),
            },
            "class_level": {
                "avg_precision": eval_results_class.get("overall_precision", 0.0),
                "avg_coverage": eval_results_class.get("overall_coverage", 0.0),
            },
        },
        "metadata": {
            "n_classes": len(target_classes),
            "n_features": len(feature_names),
            "target_classes": target_classes,
            "feature_names": feature_names,
            "output_dir": output_dir,
            "X_test_scaled": X_test_scaled,
            "X_min": X_min,
            "X_range": X_range,
        },
        "training_history": training_history,
    }
    
    # Save models
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"\nSaving SAC models...")
    for cls, trainer in sac_trainers.items():
        model_dir = os.path.join(models_dir, f"sac_class_{cls}_final")
        model_dir = os.path.abspath(model_dir)  # Convert to absolute path
        os.makedirs(model_dir, exist_ok=True)
        model_path = trainer.save(model_dir)
        print(f"  Saved SAC model for class {cls}: {model_path}")
    
    print(f"\nTraining complete!")
    print(f"Models saved to: {models_dir}")
    
    # Print evaluation results summary
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n[Instance-Level Evaluation] (One anchor per test instance):")
    print(f"  Overall Precision: {eval_results_instance.get('overall_precision', 0.0):.3f}")
    print(f"  Overall Coverage:  {eval_results_instance.get('overall_coverage', 0.0):.3f}")
    
    print(f"\n[Class-Level Evaluation] (One anchor per class):")
    print(f"  Overall Precision: {eval_results_class.get('overall_precision', 0.0):.3f}")
    print(f"  Overall Coverage:  {eval_results_class.get('overall_coverage', 0.0):.3f}")
    
    print(f"\n{'='*80}")
    
    # ======================================================================
    # Save metrics and rules to JSON (similar to trainers code)
    # ======================================================================
    print(f"\n{'='*80}")
    print("SAVING METRICS AND RULES TO JSON")
    print(f"{'='*80}")
    
    # Prepare comprehensive metrics and rules data
    metrics_data = {
        "overall_statistics": {
            "instance_level": {
                "overall_precision": float(eval_results_instance.get("overall_precision", 0.0)),
                "overall_coverage": float(eval_results_instance.get("overall_coverage", 0.0)),
                "overall_n_points": int(eval_results_instance.get("overall_n_points", 0)),
            },
            "class_level": {
                "overall_precision": float(eval_results_class.get("overall_precision", 0.0)),
                "overall_coverage": float(eval_results_class.get("overall_coverage", 0.0)),
            },
        },
        "per_class_results": {
            "instance_level": {},
            "class_level": {},
        },
        "training_history": [],
        "metadata": {
            "n_classes": len(target_classes),
            "n_features": len(feature_names),
            "target_classes": target_classes,
            "feature_names": feature_names,
            "output_dir": output_dir,
            "total_timesteps": total_timesteps,
            "algorithm": "SAC",
            "use_continuous_actions": True,
        },
    }
    
    # Add per-class results from instance-level evaluation
    for cls_int in target_classes:
        if cls_int in eval_results_instance.get("per_class_results", {}):
            cls_result = eval_results_instance["per_class_results"][cls_int]
            
            rules_list = []
            rules_with_instances = []
            anchors_list = []
            instance_indices_used = []
            
            if "individual_results" in cls_result:
                for individual_result in cls_result["individual_results"]:
                    instance_idx = int(individual_result.get("instance_idx", -1))
                    rule = individual_result.get("rule", "")
                    
                    if instance_idx >= 0:
                        instance_indices_used.append(instance_idx)
                    
                    rules_list.append(rule)
                    rules_with_instances.append({
                        "instance_idx": instance_idx,
                        "rule": rule,
                        "precision": float(individual_result.get("precision", 0.0)),
                        "hard_precision": float(individual_result.get("hard_precision", individual_result.get("precision", 0.0))),
                        "coverage": float(individual_result.get("coverage", 0.0)),
                        "n_points": int(individual_result.get("n_points", 0)),
                    })
                    anchors_list.append({
                        "instance_idx": instance_idx,
                        "lower_bounds": individual_result.get("lower_bounds", []),
                        "upper_bounds": individual_result.get("upper_bounds", []),
                        "precision": float(individual_result.get("precision", 0.0)),
                        "hard_precision": float(individual_result.get("hard_precision", individual_result.get("precision", 0.0))),
                        "coverage": float(individual_result.get("coverage", 0.0)),
                        "n_points": int(individual_result.get("n_points", 0)),
                        "rule": rule,
                    })
            
            unique_rules = list(set([r for r in rules_list if r]))
            
            metrics_data["per_class_results"]["instance_level"][f"class_{cls_int}"] = {
                "precision": float(cls_result.get("precision", cls_result.get("avg_precision", 0.0))),
                "hard_precision": float(cls_result.get("hard_precision", cls_result.get("avg_hard_precision", cls_result.get("precision", 0.0)))),
                "coverage": float(cls_result.get("coverage", cls_result.get("avg_coverage", 0.0))),
                "n_points": int(cls_result.get("n_points", 0)),
                "n_instances_evaluated": int(cls_result.get("n_instances", len(anchors_list))),
                "best_rule": cls_result.get("best_rule", ""),
                "best_precision": float(cls_result.get("best_precision", 0.0)),
                "rules": rules_list,
                "rules_with_instances": rules_with_instances,
                "unique_rules": unique_rules,
                "unique_rules_count": len(unique_rules),
                "instance_indices_used": instance_indices_used,
                "anchors": anchors_list,
            }
    
    # Add per-class results from class-level evaluation
    for cls_int in target_classes:
        if cls_int in eval_results_class.get("per_class_results", {}):
            cls_result = eval_results_class["per_class_results"][cls_int]
            
            metrics_data["per_class_results"]["class_level"][f"class_{cls_int}"] = {
                "precision": float(cls_result.get("precision", 0.0)),
                "hard_precision": float(cls_result.get("hard_precision", cls_result.get("precision", 0.0))),
                "coverage": float(cls_result.get("coverage", 0.0)),
                "global_coverage": float(cls_result.get("global_coverage", cls_result.get("coverage", 0.0))),
                "rule": cls_result.get("rule", ""),
                "lower_bounds": cls_result.get("lower_bounds", []),
                "upper_bounds": cls_result.get("upper_bounds", []),
                "evaluation_type": cls_result.get("evaluation_type", "class_level"),
            }
    
    # Add training history
    if training_history:
        for hist_entry in training_history:
            episode_data = {
                "episode": int(hist_entry.get("episode", 0)),
                "timestep": int(hist_entry.get("timestep", 0)),
                "reward": float(hist_entry.get("reward", 0.0)),
                "precision": float(hist_entry.get("precision", 0.0)),
                "coverage": float(hist_entry.get("coverage", 0.0)),
                "steps": int(hist_entry.get("steps", 0)),
            }
            if "class" in hist_entry:
                episode_data["class"] = int(hist_entry["class"])
            metrics_data["training_history"].append(episode_data)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        """Recursively convert numpy arrays and other non-serializable types to JSON-compatible types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    # Convert all data to JSON-serializable format
    metrics_data = convert_to_serializable(metrics_data)
    
    # Save to JSON file
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "metrics_and_rules.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metrics and rules to: {json_path}")
    
    # Update results to include JSON path
    results["metrics_json_path"] = json_path
    
    print(f"\n{'='*80}")
    
    return results

