import numpy as np
import torch
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path
import os
import sys
import logging
import yaml
logger = logging.getLogger(__name__)

from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarl_wrappers import AnchorTask, AnchorMetricsCallback, obs_precision_coverage
from environment import AnchorEnv


def _get_algorithm_configs():
    algorithm_map = {}
    
    try:
        from benchmarl.algorithms import MaddpgConfig
        algorithm_map["maddpg"] = (MaddpgConfig, "conf/maddpg.yaml")
    except ImportError:
        pass
    
    try:
        from benchmarl.algorithms import MasacConfig
        algorithm_map["masac"] = (MasacConfig, "conf/masac.yaml")
    except ImportError:
        pass
    
    return algorithm_map


def _slice_agent_params(
    state_dict: Dict[str, Any], agent_idx: int, n_agents: int
) -> Dict[str, Any]:
    """Extract one agent's actor parameters from a group state_dict.

    torchrl's MultiAgentMLP stores its parameters in a TensorDict whose
    batch_size is empty when share_params=True and ``(n_agents,)`` when it is
    False. In the latter case every tensor carries a leading agent dim, and the
    serialized dict marks it with a ``<prefix>.__batch_size`` entry:

        share_params=True    params.0.weight  (256, 15)   __batch_size ()
        share_params=False   params.0.weight  (3, 256, 15) __batch_size (3,)

    We locate each such prefix, index it at ``agent_idx``, and clear the marker,
    so the result is byte-compatible with the shared layout that
    ``inference.load_policy_model`` already knows how to read. A state_dict with
    no agent dim (share_params=True, or a group of one) is returned unchanged.
    """
    batched_prefixes = []
    for key, value in state_dict.items():
        if not key.endswith("__batch_size"):
            continue
        if isinstance(value, torch.Size) and tuple(value) == (n_agents,):
            batched_prefixes.append(key[: -len("__batch_size")])

    # No marker means share_params=True: no agent dim to strip.
    # A group of one still carries a leading dim of 1 under share_params=False.
    if not batched_prefixes:
        return state_dict

    if not 0 <= agent_idx < n_agents:
        raise ValueError(
            f"agent_idx {agent_idx} out of range for a group of {n_agents} agents"
        )

    sliced: Dict[str, Any] = {}
    for key, value in state_dict.items():
        prefix = next((p for p in batched_prefixes if key.startswith(p)), None)
        if prefix is None:
            sliced[key] = value
        elif key.endswith("__batch_size"):
            sliced[key] = torch.Size([])
        elif torch.is_tensor(value) and value.shape[:1] == torch.Size([n_agents]):
            sliced[key] = value[agent_idx].clone()
        else:
            sliced[key] = value

    return sliced


class AnchorTrainer:
    
    ALGORITHM_MAP = _get_algorithm_configs()
    
    def __init__(
        self,
        dataset_loader,
        algorithm: str = "maddpg",
        algorithm_config_path: Optional[str] = None,
        experiment_config_path: str = "conf/base_experiment.yaml",
        mlp_config_path: str = "conf/mlp.yaml",
        anchor_config_path: str = "conf/anchor.yaml",
        output_dir: str = "./output/anchor_training/",
        seed: int = 0
    ):
        self.dataset_loader = dataset_loader
        self.algorithm = algorithm.lower()
        
        if self.algorithm not in self.ALGORITHM_MAP:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Supported: {list(self.ALGORITHM_MAP.keys())}"
            )
        
        self.algorithm_config_class, default_algorithm_path = self.ALGORITHM_MAP[self.algorithm]
        self.algorithm_config_path = algorithm_config_path or default_algorithm_path
        self.experiment_config_path = experiment_config_path
        self.mlp_config_path = mlp_config_path
        self.anchor_config_path = anchor_config_path
        self.output_dir = output_dir
        self.seed = seed
        
        self.experiment = None
        self.experiment_config = None
        self.algorithm_config = None
        self.model_config = None
        self.critic_model_config = None
        self.task = None
        self._anchor_env_config = None
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setup_experiment(
        self,
        env_config: Optional[Dict[str, Any]] = None,
        target_classes: Optional[List[int]] = None,
        max_cycles: Optional[int] = None,
        device: str = "cpu",
        eval_on_test_data: Optional[bool] = None,
        max_n_frames: Optional[int] = None
    ) -> Experiment:
        if self.dataset_loader.classifier is None:
            raise ValueError(
                "Classifier not trained yet. "
                "Call dataset_loader.create_classifier() and dataset_loader.train_classifier() first."
            )
        
        if self.dataset_loader.X_train_unit is None:
            raise ValueError(
                "Data not preprocessed yet. "
                "Call dataset_loader.preprocess_data() first."
            )
        
        logger.info("\n" + "="*80)
        logger.info("SETTING UP ANCHOR TRAINING EXPERIMENT")
        logger.info("="*80)
        
        # Main configurations that controls the training process in BenchMARL.
        # Loaded from the conf/*.yaml files.
        self.experiment_config = ExperimentConfig.get_from_yaml(self.experiment_config_path)
        self.algorithm_config = self.algorithm_config_class.get_from_yaml(self.algorithm_config_path)
        self.model_config = MlpConfig.get_from_yaml(self.mlp_config_path)
        self.critic_model_config = MlpConfig.get_from_yaml(self.mlp_config_path)

        # Apply per-dataset training budget override before Experiment() is created.
        if max_n_frames is not None:
            logger.info(f"  Overriding max_n_frames: {self.experiment_config.max_n_frames} → {max_n_frames}")
            self.experiment_config.max_n_frames = max_n_frames

        # Pin the BenchMARL experiment folder under our output_dir. Without this,
        # save_folder stays null and BenchMARL writes the experiment to the
        # current working directory (the project root). That left run_pipeline's
        # skip-check — which rglobs for individual_models *inside* output_dir —
        # unable to find finished experiments, so MA training always retrained.
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.experiment_config.save_folder = str(Path(self.output_dir).resolve())
            logger.info(f"  Experiment save_folder pinned to: {self.experiment_config.save_folder}")
        except Exception as e:
            logger.warning(f"  Could not pin experiment save_folder ({e}); "
                           f"BenchMARL will default to the current working directory.")

        # Adjust network sizes based on dataset size for larger datasets
        n_train_samples = len(self.dataset_loader.y_train) if hasattr(self.dataset_loader, 'y_train') and self.dataset_loader.y_train is not None else 0
        if n_train_samples > 10000:
            # Large datasets (housing, etc.): use larger policy network
            new_num_cells = [512, 512]
            logger.info(f"  Large dataset detected ({n_train_samples} samples), using larger policy network: {new_num_cells}")
            # Update model configs
            self.model_config.num_cells = new_num_cells
            self.critic_model_config.num_cells = new_num_cells
        elif n_train_samples > 5000:
            # Medium-large datasets: slightly larger
            new_num_cells = [256, 256, 256]
            logger.info(f"  Medium-large dataset detected ({n_train_samples} samples), using medium policy network: {new_num_cells}")
            self.model_config.num_cells = new_num_cells
            self.critic_model_config.num_cells = new_num_cells
        else:
            logger.info(f"  Small dataset detected ({n_train_samples} samples), using default policy network: {self.model_config.num_cells}")
        
        # Get the anchor environment data.
        env_data = self.dataset_loader.get_anchor_env_data()
        
        # Get the environment configuration from YAML file or use defaults.
        if env_config is None:
            env_config = self._load_env_config_from_yaml()

        # Support YAML layouts that include a nested `env_config:` section.
        # If present, merge nested keys first, then let any top-level keys override.
        if isinstance(env_config, dict) and isinstance(env_config.get("env_config", None), dict):
            nested = env_config.get("env_config", {})
            top = {k: v for k, v in env_config.items() if k != "env_config"}
            env_config = {**nested, **top}
        
        # Apply logging verbosity early based on config
        verbosity = env_config.get("logging_verbosity", "normal")
        level = logging.WARNING if verbosity == "quiet" else (logging.DEBUG if verbosity == "verbose" else logging.INFO)
        
        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
        
        # Set for all existing loggers
        for name in logging.Logger.manager.loggerDict:
            if isinstance(logging.Logger.manager.loggerDict[name], logging.Logger):
                log = logging.getLogger(name)
                log.setLevel(level)
                for handler in log.handlers:
                    handler.setLevel(level)
        
        # Get the target classes.
        if target_classes is None:
            target_classes = list(np.unique(self.dataset_loader.y_train))
        
        # Resolve episode length: if not explicitly provided, use env_config.
        if max_cycles is None:
            max_cycles = env_config.get("max_cycles")
            if max_cycles is None:
                raise ValueError("max_cycles must be specified in env_config. Check your YAML config file.")
            max_cycles = int(max_cycles)
        else:
            max_cycles = int(max_cycles)

        # CRITICAL: Set min_coverage_floor dynamically to ensure box always covers at least the anchor instance
        # Use 1/n_samples from the dataset (to ensure at least one point is covered), 
        # or fall back to config default if dataset size unavailable
        # This prevents the coverage floor from being too high and blocking expansion during training
        n_samples = env_data["X_unit"].shape[0] if env_data.get("X_unit") is not None else None
        config_default = env_config.get("min_coverage_floor", 0.005)
        
        if n_samples is not None and n_samples > 0:
            # Use 1/n_samples to ensure at least one point is covered (the anchor instance)
            # For instance-based anchors, initial coverage is typically 0.001-0.002, so we need
            # a floor that's lower than that to allow expansion
            min_coverage_floor = 1.0 / n_samples
            # Use a very small lower bound (1e-6) instead of config_default to avoid blocking expansion
            # The config_default (0.005) is too high for instance-based anchors
            min_coverage_floor = max(min_coverage_floor, 1e-6)
        else:
            # Fall back to config default if dataset size unavailable
            min_coverage_floor = config_default
        
        # Ensure it's non-zero
        min_coverage_floor = max(min_coverage_floor, 1e-6)
        
        # Create the environment configuration with the data.
        env_config_with_data = {
            **env_config,
            "X_min": env_data["X_min"],
            "X_range": env_data["X_range"],
            "scaler_mean": env_data.get("scaler_mean"),
            "scaler_scale": env_data.get("scaler_scale"),
            "max_cycles": max_cycles,  # Ensure max_cycles is in env_config for the environment
            "min_coverage_floor": min_coverage_floor,  # Override with dynamic value
            # C-07 / C-10: pass categorical freeze info and the val split through.
            "categorical_indices": env_data.get("categorical_indices") or [],
            "categorical_value_names": env_data.get("categorical_value_names") or {},
            "X_val_unit": env_data.get("X_val_unit"),
            "X_val_std": env_data.get("X_val_std"),
            "y_val": env_data.get("y_val"),
            "X_test_unit": env_data.get("X_test_unit"),
            "X_test_std": env_data.get("X_test_std"),
            "y_test": env_data.get("y_test"),
        }
        
        logger.info(f"  Set min_coverage_floor={min_coverage_floor:.6f} for training (n_samples={n_samples if n_samples is not None else 'unknown'}, ensures box covers at least anchor instance)")
        
        # CRITICAL: Prevent test data leak into training - multi-agent uses same env for training/eval
        # Unlike single-agent which has separate train/eval envs, multi-agent must force training data during training
        # The eval_on_test_data parameter should only affect evaluation, not training
        # Always use training data for training environment (mode="training")
        # Store eval_on_test_data for later use in evaluation, but force False for training env
        env_mode = env_config.get("mode", "training")
        is_training_mode = (env_mode == "training")
        
        # Respect YAML config for eval_on_test_data if parameter not explicitly provided
        if eval_on_test_data is None:
            eval_on_test_data = env_config.get("eval_on_test_data", False)
            logger.info(f"  Using eval_on_test_data={eval_on_test_data} from YAML config (anchor.yaml)")
        else:
            # Parameter was explicitly provided, use it (but log a warning if it overrides YAML)
            yaml_value = env_config.get("eval_on_test_data", False)
            if eval_on_test_data != yaml_value:
                logger.warning(
                    f"  WARNING: eval_on_test_data={eval_on_test_data} (explicit parameter) "
                    f"overrides YAML config value ({yaml_value}). "
                    f"Note: This will only affect evaluation, not training (training always uses training data)."
                )
        
        # CRITICAL: Force eval_on_test_data=False for training to prevent test data leakage
        # Store the original value for evaluation use, but training environment must use training data
        if is_training_mode:
            if eval_on_test_data:
                logger.warning(
                    f"  CRITICAL: eval_on_test_data=True was requested, but forcing False for TRAINING environment "
                    f"to prevent test data leakage. Training will use training data only. "
                    f"Evaluation can use test data separately."
                )
            eval_on_test_data_for_env = False
            logger.info(f"  Training environment configured to use TRAINING data (test data leak prevented)")
        else:
            # For evaluation/inference modes, use the requested eval_on_test_data value
            eval_on_test_data_for_env = eval_on_test_data
            logger.info(f"  {'Evaluation' if env_mode == 'evaluation' else 'Inference'} environment configured to use {'TEST' if eval_on_test_data_for_env else 'TRAINING'} data")
        
        # Store original eval_on_test_data for potential use in evaluation
        self._eval_on_test_data_for_evaluation = eval_on_test_data
        
        if eval_on_test_data_for_env:
            if env_data.get("X_test_unit") is None or env_data.get("X_test_std") is None or env_data.get("y_test") is None:
                raise ValueError(
                    "eval_on_test_data=True requires test data. "
                    "Make sure dataset_loader has test data loaded and preprocessed."
                )
            env_config_with_data.update({
                "eval_on_test_data": True,
                "X_test_unit": env_data["X_test_unit"],
                "X_test_std": env_data["X_test_std"],
                "y_test": env_data["y_test"],
            })
        else:
            env_config_with_data["eval_on_test_data"] = False
        
        # Compute k-means centroids for multiple agents per class
        # Compute k-means centroids for diversity across episodes
        # This ensures each episode can start from a different cluster centroid
        # For agents_per_class > 1: use agents_per_class * 10 centroids (10 per agent for diversity)
        # For agents_per_class == 1: use 10 centroids per class
        agents_per_class = env_config.get("agents_per_class", 1)
        
        # Determine number of centroids to compute
        if agents_per_class > 1:
            # Use 10 centroids per agent for diversity
            n_clusters_per_class = agents_per_class * 10
            logger.info(f"\nComputing k-means centroids (k={n_clusters_per_class}) for each class...")
            logger.info(f"  Note: agents_per_class={agents_per_class}, using {n_clusters_per_class} centroids ({n_clusters_per_class // agents_per_class} per agent) for episode diversity")
        else:
            # For single agent per class, use 10 centroids for diversity
            n_clusters_per_class = 10
            logger.info(f"\nComputing k-means centroids (k={n_clusters_per_class}) for each class for diversity...")
            logger.info(f"  Note: agents_per_class={agents_per_class}, using {n_clusters_per_class} centroids for episode diversity")
        
        try:
            from utils.clusters import compute_cluster_centroids_per_class
            
            # Always compute centroids on training data
            X_data = env_data["X_unit"]
            y_data = env_data["y"]
            
            # Use adaptive clustering: adjust cluster count based on dataset size
            # and check for scattered data distribution
            cluster_centroids_per_class = compute_cluster_centroids_per_class(
                X_unit=X_data,
                y=y_data,
                n_clusters_per_class=n_clusters_per_class,
                random_state=self.seed if hasattr(self, 'seed') else 42,
                min_samples_per_cluster=1,
                auto_adapt_clusters=True,  # Adapt cluster count to dataset size
                check_data_scatter=True    # Check if data is scattered (use mean if so)
            )
            
            # Verify we have enough centroids for each class
            for cls in target_classes:
                if cls in cluster_centroids_per_class:
                    n_centroids = len(cluster_centroids_per_class[cls])
                    if n_centroids < n_clusters_per_class:
                        logger.warning(
                            f"   Class {cls}: Only {n_centroids} centroids computed "
                            f"(requested {n_clusters_per_class}). "
                            f"May not have enough samples for k-means."
                        )
                    else:
                        logger.info(f"   Class {cls}: {n_centroids} centroids computed")
                else:
                    logger.warning(f"   Class {cls}: No centroids computed")
            
            # Set cluster centroids in env_config
            env_config_with_data["cluster_centroids_per_class"] = cluster_centroids_per_class
            logger.info("   Cluster centroids set in environment config")
        except ImportError as e:
            logger.warning(f"   Could not compute cluster centroids: {e}")
            logger.warning(f"  Install sklearn: pip install scikit-learn")
            logger.warning(f"  Falling back to mean centroid per class (all episodes will start from same point)")
            env_config_with_data["cluster_centroids_per_class"] = None
        except Exception as e:
            logger.warning(f"   Error computing cluster centroids: {e}")
            logger.warning(f"  Falling back to mean centroid per class (all episodes will start from same point)")
            env_config_with_data["cluster_centroids_per_class"] = None
        
        # Sample instances per class for instance-based training (mixed initialization)
        training_instance_ratio = env_config_with_data.get("training_instance_ratio", 0.3)
        use_adaptive_ratios = env_config_with_data.get("use_adaptive_instance_ratios", True)  # Enable by default
        
        if training_instance_ratio > 0.0:
            np.random.seed(self.seed if hasattr(self, 'seed') else 42)
            
            # Compute class-specific ratios based on class imbalance (if adaptive mode enabled)
            training_instance_ratios_per_class = {}
            if use_adaptive_ratios and len(target_classes) > 1:
                # Compute class counts
                class_counts = {cls: (y_data == cls).sum() for cls in target_classes}
                min_count = min(class_counts.values())
                max_count = max(class_counts.values())
                imbalance_ratio = max_count / min_count if min_count > 0 else 1.0
                
                # Use warning level so these important setup messages show even in quiet mode
                logger.warning(f"\nComputing adaptive class-specific training instance ratios...")
                logger.warning(f"  Class imbalance ratio: {imbalance_ratio:.2f}:1 (max/min)")
                logger.warning(f"  Base ratio: {training_instance_ratio:.1%}")
                
                # Only use adaptive ratios if there's significant imbalance (> 1.5:1)
                if imbalance_ratio > 1.5:
                    for cls in target_classes:
                        count = class_counts[cls]
                        # Higher ratio for minority classes (inversely proportional to class size)
                        # Use square root scaling to avoid extreme ratios
                        size_factor = (max_count / count) ** 0.5 if count > 0 else 1.0
                        adaptive_ratio = training_instance_ratio * size_factor
                        # Ensure ratio is at least the base ratio, but don't cap it (respect user's configuration)
                        adaptive_ratio = max(training_instance_ratio, adaptive_ratio)
                        # Cap at 1.0 (100%) maximum since ratio represents probability
                        adaptive_ratio = min(1.0, adaptive_ratio)
                        training_instance_ratios_per_class[cls] = adaptive_ratio
                        
                        minority_status = "minority" if count < min_count * 1.5 else "majority"
                        logger.warning(f"   Class {cls}: {count} samples ({minority_status}) → ratio: {adaptive_ratio:.1%}")
                else:
                    # Balanced dataset - use same ratio for all classes
                    logger.warning(f"  Dataset is relatively balanced - using uniform ratio for all classes")
                    for cls in target_classes:
                        training_instance_ratios_per_class[cls] = training_instance_ratio
            else:
                # Use uniform ratio for all classes
                for cls in target_classes:
                    training_instance_ratios_per_class[cls] = training_instance_ratio
            
            # Sample instances per class (use max ratio to ensure enough instances)
            max_ratio = max(training_instance_ratios_per_class.values())
            n_instances_per_class = max(20, int(10 / max_ratio))  # Ensure enough instances
            training_instances_per_class = {}
            
            logger.info(f"\nSampling instances per class for instance-based training...")
            logger.info(f"  Sampling {n_instances_per_class} instances per class (based on max ratio: {max_ratio:.1%})")
            logger.info(f"  CRITICAL: Filtering instances where classifier prediction matches target_class")
            logger.info(f"  This ensures original_prediction == target_class for instance-based anchors")
            
            # Get classifier to filter instances by prediction
            classifier = self.dataset_loader.get_classifier()
            classifier.eval()
            from utils.device_utils import get_device_str
            device_str = get_device_str(device) if device != "auto" else "cpu"
            device_torch = torch.device(device_str)
            
            for cls in target_classes:
                class_mask = (y_data == cls)
                class_indices = np.where(class_mask)[0]
                class_ratio = training_instance_ratios_per_class[cls]
                
                logger.info(f"   Class {cls}: {len(class_indices)} training samples available (ratio: {class_ratio:.1%})")
                
                # Filter instances where classifier prediction matches target_class
                # This ensures original_prediction == target_class, preventing precision calculation issues
                if len(class_indices) > 0:
                    # Get predictions for all class instances
                    X_class_std = env_data["X_std"][class_indices]
                    with torch.no_grad():
                        from utils.networks import predict_proba_torch
                        X_tensor = torch.from_numpy(X_class_std.astype(np.float32)).to(device_torch)
                        probs = predict_proba_torch(classifier, X_tensor).cpu().numpy()
                        predictions = np.argmax(probs, axis=1)
                    
                    # Filter: keep only instances where prediction matches target_class
                    prediction_match_mask = (predictions == cls)
                    matching_indices = class_indices[prediction_match_mask]
                    n_matching = len(matching_indices)
                    
                    logger.info(f"   Class {cls}: {n_matching}/{len(class_indices)} instances have prediction matching target_class")
                    
                    if n_matching == 0:
                        logger.warning(
                            f"   Class {cls}: No instances found where classifier prediction matches target_class! "
                            f"This will cause issues with instance-based anchors. "
                            f"Falling back to all instances (may cause low precision)."
                        )
                        matching_indices = class_indices  # Fallback to all instances
                        n_matching = len(matching_indices)
                    
                    if n_matching >= n_instances_per_class:
                        # Randomly sample from matching instances
                        selected_indices = np.random.choice(matching_indices, size=n_instances_per_class, replace=False)
                        training_instances_per_class[cls] = X_data[selected_indices].tolist()
                        logger.info(f"   Class {cls}: {n_instances_per_class} instances sampled from {n_matching} matching instances (ratio: {class_ratio:.1%})")
                    elif n_matching > 0:
                        # Use all matching instances if fewer than requested
                        training_instances_per_class[cls] = X_data[matching_indices].tolist()
                        logger.warning(
                            f"   Class {cls}: Only {n_matching} matching instances available "
                            f"(requested {n_instances_per_class}). Using all matching instances (ratio: {class_ratio:.1%})."
                        )
                    else:
                        logger.error(f"   Class {cls}: No matching instances available for sampling! This will cause initialization failures.")
                else:
                    logger.warning(f"   Class {cls}: No instances available for sampling")
            
            # Store training instances and class-specific ratios in env_config
            env_config_with_data["training_instances_per_class"] = training_instances_per_class
            env_config_with_data["training_instance_ratios_per_class"] = training_instance_ratios_per_class
            logger.info("   ✓ Training instances and class-specific ratios set in environment config")
        else:
            env_config_with_data["training_instances_per_class"] = None
            env_config_with_data["training_instance_ratios_per_class"] = None
            logger.info("   Training instance ratio is 0.0 - using centroid-based initialization only")
        
        # Create the anchor configuration.
        anchor_config = {
            "X_unit": env_data["X_unit"],
            "X_std": env_data["X_std"],
            "y": env_data["y"],
            "feature_names": env_data["feature_names"],
            "classifier": self.dataset_loader.get_classifier(),
            "device": device,
            "target_classes": target_classes,
            "env_config": env_config_with_data,
            "max_cycles": max_cycles,
        }
        
        self.task = AnchorTask.ANCHOR.get_task(config=anchor_config)
        
        # SS Bug: Check if the collec_anchor_data is actually working
        # Get NashConv threshold from config (default: 0.01)
        nashconv_threshold = env_config.get("nashconv_threshold", 0.01)

        # NashConv is OFF by default (`compute_nashconv: false` in conf/anchor.yaml).
        #
        # Two independent reasons:
        #   - MASAC: its critic API (twin Q-nets, stochastic TanhNormal actor)
        #     doesn't match the MADDPG-shaped exploitability proxy this callback
        #     implements, so it returns empty metrics anyway.
        #   - MADDPG: the gradient best-response frequently reports
        #     `q_has_action_grad == False` and the random-search fallback then
        #     evaluates no candidate, so whole evaluations come back
        #     "exploitability not measurable" -- a per-eval warning storm and a
        #     large amount of wasted compute for a number we cannot report.
        #
        # Best-model selection does not depend on it: with NashConv unavailable
        # the callback falls back to pure score-based selection (equilibrium min
        # class score, then aggregate precision+coverage). Set
        # `compute_nashconv: true` in the env config to re-enable.
        compute_nashconv = bool(env_config.get("compute_nashconv", False))
        if compute_nashconv and self.algorithm == "masac":
            compute_nashconv = False
            logger.info("  NashConv disabled for MASAC (incompatible critic API; would return empty metrics).")
        elif not compute_nashconv:
            logger.info("  NashConv disabled (compute_nashconv=false); model selection is score-based.")

        self.callback = AnchorMetricsCallback(
            log_training_metrics=True,
            log_evaluation_metrics=True,
            save_to_file=True,
            collect_anchor_data=True,
            compute_nashconv=compute_nashconv,
            nashconv_threshold=nashconv_threshold,
            ranking_score_formula=env_config.get(
                "ranking_score_formula", "precision_coverage"
            ),
        )
        self.experiment = Experiment(
            config=self.experiment_config,
            task=self.task,
            algorithm_config=self.algorithm_config,
            model_config=self.model_config,
            critic_model_config=self.critic_model_config,
            seed=self.seed,
            callbacks=[self.callback]
        )
        # Extract individual_models_best from in-memory actors whenever eval
        # improves. Experiment.state_dict() can fail after training (collector
        # has no .env), so this is the reliable path to BEST weights.
        self.callback.extract_best_fn = lambda: self.extract_and_save_individual_models(
            save_policies=True,
            save_critics=False,
            models_subdir="individual_models_best",
        )

        # BenchMARL constructs a dedicated ``test_env`` for periodic evaluation,
        # but both environments originate from the same task config. Mark only
        # that realized environment as evaluation so checkpoint selection reads
        # D_val while collectors continue to read D_train.
        self._set_benchmarl_eval_split(
            self.experiment.test_env,
            split=str(env_config.get("eval_split", "val")),
        )
        
        # Set experiment folder on callback for periodic saving during training
        if hasattr(self.callback, 'set_experiment_folder'):
            self.callback.set_experiment_folder(str(self.experiment.folder_name))
        
        logger.info(f"Experiment setup complete:")
        logger.info(f"  Algorithm: {self.algorithm.upper()}")
        logger.info(f"  Model: MLP")
        logger.info(f"  Target classes: {target_classes}")
        logger.info(f"  Max cycles per episode: {max_cycles}")
        logger.info(f"  Experiment folder: {self.experiment.folder_name}")
        logger.info("="*80)
        
        return self.experiment

    @staticmethod
    def _set_benchmarl_eval_split(env: Any, split: str = "val") -> None:
        if split != "val":
            raise ValueError(
                "Training-time BenchMARL evaluation must use split='val'; "
                f"got {split!r}"
            )
        seen = set()

        def _visit(obj: Any) -> bool:
            if obj is None or id(obj) in seen:
                return False
            seen.add(id(obj))
            configured = False
            if isinstance(obj, AnchorEnv):
                if obj.X_val_unit is None or obj.X_val_std is None or obj.y_val is None:
                    raise ValueError(
                        "BenchMARL validation environment is missing X_val/y_val"
                    )
                obj.mode = "evaluation"
                obj.eval_split = "val"
                obj.eval_on_test_data = False
                return True
            for attr in ("base_env", "_env", "env"):
                try:
                    configured = _visit(getattr(obj, attr, None)) or configured
                except (AttributeError, RuntimeError):
                    continue
            return configured

        if not _visit(env):
            raise RuntimeError(
                "Could not locate AnchorEnv inside BenchMARL test_env; "
                "validation checkpoint selection cannot be guaranteed"
            )
        logger.info("BenchMARL periodic evaluation configured on VALIDATION split")
    
    def train(self) -> Experiment:
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        
        logger.info("\n" + "="*80)
        logger.info("STARTING ANCHOR TRAINING")
        logger.info("="*80)
        
        self.experiment.run()
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Results saved to: {self.experiment.folder_name}")
        
        # Save wandb run URL for later reference (if available)
        if os.environ.get("DISABLE_WANDB", "0") == "1":
            logger.debug("Wandb disabled via DISABLE_WANDB environment variable")
        else:
            try:
                import wandb
                if wandb.run is not None:
                    wandb_run_url = wandb.run.url if hasattr(wandb.run, 'url') else None
                    if wandb_run_url is None:
                        try:
                            entity = wandb.run.entity if hasattr(wandb.run, 'entity') else None
                            project = wandb.run.project if hasattr(wandb.run, 'project') else None
                            run_id = wandb.run.id if hasattr(wandb.run, 'id') else None
                            if entity and project and run_id:
                                wandb_run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
                        except Exception:
                            pass
                    
                    if wandb_run_url:
                        wandb_url_file = os.path.join(str(self.experiment.folder_name), "wandb_run_url.txt")
                        os.makedirs(os.path.dirname(wandb_url_file), exist_ok=True)
                        with open(wandb_url_file, 'w') as f:
                            f.write(wandb_run_url)
                        logger.info(f"✓ Wandb run URL saved to: {wandb_url_file}")
                        logger.info(f"  URL: {wandb_run_url}")
            except Exception as e:
                logger.debug(f"Could not save wandb run URL: {e}")
        
        # Flush any remaining training metrics before final save
        if hasattr(self, 'callback') and self.callback is not None:
            if hasattr(self.callback, '_flush_remaining_metrics'):
                try:
                    self.callback._flush_remaining_metrics()
                except Exception as e:
                    logger.warning(f"   Warning: Could not flush remaining metrics: {e}")
        
        # Save callback data to files
        if hasattr(self, 'callback') and self.callback is not None:
            if hasattr(self.callback, 'save_data_to_files'):
                logger.info("\nSaving callback data to files...")
                try:
                    saved_files = self.callback.save_data_to_files(str(self.experiment.folder_name))
                    if saved_files:
                        logger.info(f"   Saved {len(saved_files)} data files")
                    else:
                        logger.info("  No callback data to save")
                except Exception as e:
                    logger.warning(f"   Warning: Could not save callback data: {e}")
        
        return self.experiment

    def evaluate(
        self,
        n_eval_episodes: Optional[int] = None,
        collect_anchor_data: bool = True
    ) -> Dict[str, Any]:
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        
        logger.info("\n" + "="*80)
        logger.info("RUNNING EVALUATION")
        logger.info("="*80)
        
        # Get anchor data collected during training evaluations
        # BenchMARL's periodic evaluations during training DO pass rollouts to callbacks
        evaluation_anchor_data = []
        if hasattr(self.callback, 'get_evaluation_anchor_data'):
            evaluation_anchor_data = self.callback.get_evaluation_anchor_data()
            logger.info(f"  Found {len(evaluation_anchor_data)} episodes of anchor data from training evaluations")
        
        try:
            self.experiment.evaluate()
        except Exception as e:
            # Handle wandb run finished error gracefully
            if "wandb" in str(type(e)).lower() or "finished" in str(e).lower() or "UsageError" in str(type(e)):
                logger.warning(f"Warning: Evaluation completed but could not log to wandb (run may be finished): {e}")
                logger.warning("Evaluation metrics are still saved to CSV and other loggers.")
            else:
                # Re-raise if it's a different error
                raise
        
        # IMPORTANT: BenchMARL's evaluate() doesn't pass rollouts to callbacks after training
        # Manually run rollouts to collect anchor data. BenchMARL evaluate does do rollouts.
        if collect_anchor_data:
            logger.info(f"\n  Running manual rollouts to collect anchor data...")
            logger.info(f"  (BenchMARL's evaluate() doesn't pass rollouts to callbacks after training)")
            
            # Get number of episodes from config if not provided
            if n_eval_episodes is None:
                n_eval_episodes = self.experiment_config.evaluation_episodes if hasattr(self.experiment_config, 'evaluation_episodes') else 2
            
            # Manually run rollouts using the environment
            try:
                from tensordict import TensorDict
                
                algorithm = self.experiment.algorithm
                
                # Get device from policy (to ensure environment and tensors are on same device)
                policy_device = None
                try:
                    for group in algorithm.group_map.keys():
                        policy = algorithm.get_policy_for_loss(group)
                        if hasattr(policy, 'parameters'):
                            for param in policy.parameters():
                                if param is not None:
                                    policy_device = param.device
                                    break
                        if policy_device is not None:
                            break
                except Exception:
                    pass
                
                # Default to CPU if device not found
                if policy_device is None:
                    policy_device = torch.device("cpu")
                
                # Create environment instance on the same device as policy
                # Set mode to "evaluation" so termination counters are reset in reset()
                env = self._create_env_instance(device=str(policy_device))
                if hasattr(env, 'env') or hasattr(env, '_env'):
                    unwrapped = getattr(env, 'env', None) or getattr(env, '_env', None)
                    if unwrapped is not None:
                        unwrapped.mode = "evaluation"
                elif hasattr(env, 'mode'):
                    env.mode = "evaluation"
                
                logger.info(f"  Debug: algorithm.group_map = {algorithm.group_map}")
                logger.info(f"  Debug: group_map keys = {list(algorithm.group_map.keys())}")
                for group, agents in algorithm.group_map.items():
                    logger.info(f"    Group '{group}' contains agents: {agents}")
                
                # Get unwrapped environment to check actual agents
                unwrapped_env = None
                if hasattr(env, 'env') or hasattr(env, '_env'):
                    unwrapped_env = getattr(env, 'env', None) or getattr(env, '_env', None)
                    if unwrapped_env is not None:
                        if hasattr(unwrapped_env, 'possible_agents'):
                            logger.info(f"  Debug: Environment has {len(unwrapped_env.possible_agents)} agents: {unwrapped_env.possible_agents}")
                        if hasattr(unwrapped_env, 'agent_to_class'):
                            logger.info(f"  Debug: Agent to class mapping: {unwrapped_env.agent_to_class}")
                
                # Run evaluation episodes
                manual_rollouts = []
                import time
                eval_start_time = time.perf_counter()
                
                for episode in range(n_eval_episodes):
                    # Reset environment
                    td = env.reset()
                    done = False
                    episode_data = {}
                    
                    # Start timing this episode
                    episode_start_time = time.perf_counter()
                    
                    # Run episode
                    step_count = 0
                    # Read max_steps from environment config, not hardcoded default
                    max_steps = self.task.max_steps(env) if hasattr(self.task, 'max_steps') else env.max_cycles
                    
                    if episode == 0:
                        logger.info(f"  Debug: Initial td keys: {list(td.keys()) if hasattr(td, 'keys') else 'N/A'}")
                        if hasattr(td, 'keys'):
                            for key in td.keys():
                                if hasattr(td[key], 'keys'):
                                    logger.info(f"    {key} keys: {list(td[key].keys())}")
                    
                    while not done and step_count < max_steps:
                        # Get action from policy (deterministic for evaluation)
                        with torch.no_grad():
                            # Get policy for each group
                            for group in algorithm.group_map.keys():
                                policy = algorithm.get_policy_for_loss(group)
                                
                                # Create input TensorDict
                                if group in td.keys():
                                    group_obs = td[group]
                                    if "observation" in group_obs.keys():
                                        obs_tensor = group_obs["observation"]
                                        
                                        # Move observation to policy device
                                        if isinstance(obs_tensor, torch.Tensor):
                                            obs_tensor = obs_tensor.to(policy_device)
                                        
                                        input_td = TensorDict(
                                            {group: {"observation": obs_tensor}},
                                            batch_size=obs_tensor.shape[:1],
                                            device=policy_device
                                        )
                                        
                                        # Get action
                                        if hasattr(policy, "forward_inference"):
                                            action_output = policy.forward_inference(input_td)
                                        else:
                                            action_output = policy(input_td)
                                        
                                        # Extract action and move back to environment device if needed
                                        if isinstance(action_output, TensorDict):
                                            # Try to get action from nested structure
                                            # TensorDict doesn't support tuple key checks without include_nested=True
                                            action = None
                                            
                                            # Method 1: Try nested key access directly (most common structure)
                                            try:
                                                action = action_output[(group, "action")]
                                            except (KeyError, TypeError):
                                                pass
                                            
                                            # Method 2: Try flat nested structure (group -> action)
                                            if action is None:
                                                try:
                                                    if group in action_output.keys():
                                                        group_data = action_output[group]
                                                        if isinstance(group_data, TensorDict):
                                                            if "action" in group_data.keys():
                                                                action = group_data["action"]
                                                        elif hasattr(group_data, 'get'):
                                                            action = group_data.get("action", None)
                                                except (KeyError, TypeError, AttributeError):
                                                    pass
                                            
                                            # Method 3: Try direct "action" key (some policies return flat structure)
                                            if action is None:
                                                try:
                                                    if "action" in action_output.keys():
                                                        action = action_output["action"]
                                                except (KeyError, TypeError):
                                                    pass
                                            
                                            if action is not None:
                                                # Move action to environment device
                                                # The environment expects actions on its device (usually same as observations)
                                                if isinstance(td, TensorDict) and hasattr(td, 'device'):
                                                    # Move action to match the TensorDict device (environment device)
                                                    action = action.to(td.device)
                                                elif hasattr(env, 'device'):
                                                    action = action.to(env.device)
                                                td[group]["action"] = action
                        
                        # Step environment
                        td = env.step(td)
                        done = td.get("done", torch.zeros(1, dtype=torch.bool)).any().item()
                        step_count += 1
                    
                    # End timing this episode
                    episode_end_time = time.perf_counter()
                    episode_duration = episode_end_time - episode_start_time
                    
                    # Collect final metrics from info
                    unwrapped_env = None
                    if hasattr(env, 'env') or hasattr(env, '_env'):
                        unwrapped_env = getattr(env, 'env', None) or getattr(env, '_env', None)
                    
                    if episode == 0 and unwrapped_env is not None:
                        if hasattr(unwrapped_env, 'lower') and hasattr(unwrapped_env, 'upper'):
                            if isinstance(unwrapped_env.lower, dict):
                                for agent_name in unwrapped_env.agents:
                                    if agent_name in unwrapped_env.lower:
                                        lower = unwrapped_env.lower[agent_name]
                                        upper = unwrapped_env.upper[agent_name]
                                        logger.info(f"  Debug: After episode, {agent_name} bounds:")
                                        logger.info(f"    Lower range: [{lower.min():.4f}, {lower.max():.4f}]")
                                        logger.info(f"    Upper range: [{upper.min():.4f}, {upper.max():.4f}]")
                                        # Get metrics to verify
                                        try:
                                            prec, cov, _ = unwrapped_env._current_metrics(agent_name)
                                            logger.info(f"    Precision: {prec:.4f}, Coverage: {cov:.4f}")
                                        except Exception as e:
                                            logger.info(f"    Could not get metrics: {e}")
                    
                    # Try multiple ways to access the final state
                    # After the episode loop, td should contain the final state
                    # Check both td and td["next"] if it exists
                    final_td = td
                    if "next" in td.keys():
                        next_td = td["next"]
                        # Use next_td as it contains the state after the last step
                        final_td = next_td
                    else:
                        # If no "next" key, use td directly (it's the final state)
                        next_td = td
                        final_td = td
                    
                    # Collect data for each agent separately
                    # If group_map has agents listed per group, iterate over those agents
                    # Otherwise, iterate over groups and find matching agents
                    agents_to_process = []
                    for group in algorithm.group_map.keys():
                        # Get agents in this group
                        agents_in_group = algorithm.group_map[group]
                        if isinstance(agents_in_group, list) and len(agents_in_group) > 0:
                            # Group contains multiple agents - process each separately
                            for agent in agents_in_group:
                                agents_to_process.append((group, agent))
                        else:
                            # Group name might be the agent name, or we need to find matching agents
                            agents_to_process.append((group, group))
                    
                    # If no agents found from group_map, try to get from environment
                    if not agents_to_process and unwrapped_env is not None:
                        if hasattr(unwrapped_env, 'agents') and len(unwrapped_env.agents) > 0:
                            # Use actual agent names from environment
                            for agent in unwrapped_env.agents:
                                # Try to find which group this agent belongs to
                                group = None
                                for g, agents_list in algorithm.group_map.items():
                                    if agent in agents_list or (isinstance(agents_list, str) and agent == agents_list):
                                        group = g
                                        break
                                if group is None:
                                    # Use agent name as group if no match found
                                    group = agent
                                agents_to_process.append((group, agent))
                    
                    # Process each agent separately
                    for group, agent_name in agents_to_process:
                        if episode == 0:
                            if unwrapped_env is None:
                                logger.info(f"  Debug: unwrapped_env is None for {agent_name}")
                            elif not hasattr(unwrapped_env, 'agents'):
                                logger.info(f"  Debug: unwrapped_env doesn't have 'agents' attribute for {agent_name}")
                            elif len(unwrapped_env.agents) == 0:
                                logger.info(f"  Debug: unwrapped_env.agents is empty for {agent_name}")
                                if hasattr(unwrapped_env, 'possible_agents'):
                                    logger.info(f"  Debug: Using possible_agents instead: {unwrapped_env.possible_agents}")
                            else:
                                logger.info(f"  Debug: unwrapped_env.agents = {unwrapped_env.agents} for {agent_name}")
                        
                        # First, try to get info from unwrapped environment (most reliable)
                        # After episode, agents might be removed, so check possible_agents too
                        agent_in_env = False
                        if unwrapped_env is not None:
                            # Check if agent is in current agents list
                            if hasattr(unwrapped_env, 'agents') and agent_name in unwrapped_env.agents:
                                agent_in_env = True
                            # If not, check possible_agents (agents that can exist)
                            elif hasattr(unwrapped_env, 'possible_agents') and agent_name in unwrapped_env.possible_agents:
                                agent_in_env = True
                                # Try to find matching agent if name doesn't match exactly
                                if agent_name not in unwrapped_env.possible_agents:
                                    matching_agent = None
                                    for possible_agent in unwrapped_env.possible_agents:
                                        if possible_agent == agent_name or (agent_name in possible_agent or possible_agent.startswith(agent_name)):
                                            matching_agent = possible_agent
                                            break
                                    if matching_agent:
                                        agent_name = matching_agent
                                        agent_in_env = True
                            
                            if agent_in_env and hasattr(unwrapped_env, '_current_metrics'):
                                try:
                                    # Get current metrics directly from environment for this specific agent
                                    precision, coverage, _ = unwrapped_env._current_metrics(agent_name)
                                    
                                    # Debug output for first episode
                                    if episode == 0:
                                        logger.info(f"  Debug: Got metrics from unwrapped_env for {agent_name}: precision={precision:.4f}, coverage={coverage:.4f}")
                                    
                                    # Get final observation (bounds) from environment state
                                    if hasattr(unwrapped_env, 'lower') and hasattr(unwrapped_env, 'upper'):
                                        # lower and upper are dictionaries keyed by agent name
                                        if isinstance(unwrapped_env.lower, dict):
                                            # Try agent_name first
                                            if agent_name in unwrapped_env.lower:
                                                lower_bounds = unwrapped_env.lower[agent_name]
                                                upper_bounds = unwrapped_env.upper[agent_name]
                                            else:
                                                # If agent_name not found, try to find a matching key
                                                matching_key = None
                                                for key in unwrapped_env.lower.keys():
                                                    if key == agent_name or agent_name in key or key.startswith(agent_name):
                                                        matching_key = key
                                                        break
                                                
                                                if matching_key:
                                                    if episode == 0:
                                                        logger.info(f"  Debug: Using matching key '{matching_key}' for agent {agent_name}")
                                                    lower_bounds = unwrapped_env.lower[matching_key]
                                                    upper_bounds = unwrapped_env.upper[matching_key]
                                                else:
                                                    if episode == 0:
                                                        logger.info(f"  Debug: Agent {agent_name} not in lower/upper dict. Available keys: {list(unwrapped_env.lower.keys())}")
                                                    continue  # Skip this agent if not in dict
                                        else:
                                            # If not a dict, might be a single array (single agent case)
                                            lower_bounds = unwrapped_env.lower
                                            upper_bounds = unwrapped_env.upper
                                        
                                        if episode == 0:
                                            logger.info(f"  Debug: {agent_name} bounds - lower range: [{lower_bounds.min():.4f}, {lower_bounds.max():.4f}], upper range: [{upper_bounds.min():.4f}, {upper_bounds.max():.4f}]")
                                        
                                        final_obs = np.concatenate([lower_bounds, upper_bounds, np.array([precision, coverage], dtype=np.float32)])
                                        
                                        # Store data keyed by agent name (not group) to distinguish between agents
                                        episode_data[agent_name] = {
                                            "anchor_precision": float(precision),
                                            "anchor_coverage": float(coverage),
                                            "total_reward": 0.0,  # Will try to get from info if available
                                            "final_observation": final_obs.tolist(),
                                            "group": group,  # Keep track of which group this agent belongs to
                                            "target_class": unwrapped_env.agent_to_class.get(agent_name, None) if hasattr(unwrapped_env, 'agent_to_class') else None,
                                        }
                                        
                                        if episode == 0:
                                            logger.info(f"  Debug: Stored data for {agent_name} with precision={precision:.4f}, coverage={coverage:.4f}")
                                        
                                        # Try to get total_reward from last step's info if available
                                        # (This is a fallback - we already have precision/coverage from env)
                                        continue  # Skip to next agent since we got data from env
                                    else:
                                        if episode == 0:
                                            logger.info(f"  Debug: unwrapped_env doesn't have lower/upper attributes")
                                except Exception as e:
                                    if episode == 0:
                                        logger.info(f"  Debug: Could not get metrics from unwrapped env for agent {agent_name}: {e}")
                                        import traceback
                                        traceback.print_exc()
                        
                        # Fallback: Try to get from TensorDict structure
                        # Try both final_td (which might be next_td) and td directly
                        group_data = None
                        
                        # First try final_td (state after last step)
                        if group in final_td.keys():
                            group_data = final_td[group]
                        elif hasattr(final_td, 'get'):
                            group_data = final_td.get(group, None)
                        
                        # If not found, try next_td
                        if group_data is None:
                            if group in next_td.keys():
                                group_data = next_td[group]
                            elif hasattr(next_td, 'get'):
                                group_data = next_td.get(group, None)
                        
                        # If still not found, try td directly (current state)
                        if group_data is None:
                            if group in td.keys():
                                group_data = td[group]
                            elif hasattr(td, 'get'):
                                group_data = td.get(group, None)
                        
                        if group_data is not None:
                            # Try to get info first
                            info = None
                            if isinstance(group_data, TensorDict):
                                if "info" in group_data.keys():
                                    info = group_data["info"]
                            elif hasattr(group_data, 'get'):
                                info = group_data.get("info", None)
                            elif hasattr(group_data, 'keys') and "info" in group_data.keys():
                                info = group_data["info"]
                            
                            # Get final observation (anchor bounds) - this is always available
                            obs = None
                            if isinstance(group_data, TensorDict):
                                if "observation" in group_data.keys():
                                    obs = group_data["observation"]
                            elif hasattr(group_data, 'get'):
                                obs = group_data.get("observation", None)
                            
                            # Extract P,C from the 3n+4 observation via obs_precision_coverage.
                            # Observation structure: [a, b, q*, P, C, mode, phase]
                            if obs is not None:
                                if hasattr(obs, 'shape') and obs.shape[0] > 0:
                                    final_obs = obs[-1]
                                elif isinstance(obs, (list, tuple)) and len(obs) > 0:
                                    final_obs = obs[-1]
                                else:
                                    final_obs = obs
                                
                                # Convert to numpy
                                if isinstance(final_obs, torch.Tensor):
                                    final_obs_np = final_obs.cpu().numpy()
                                elif isinstance(final_obs, np.ndarray):
                                    final_obs_np = final_obs
                                else:
                                    final_obs_np = np.array(final_obs)
                                
                                # Extract precision and coverage from observation
                                obs_len = len(final_obs_np) if hasattr(final_obs_np, '__len__') else final_obs_np.shape[0] if hasattr(final_obs_np, 'shape') else 0
                                
                                if obs_len >= 5:  # n>=1 features + P, C, mode, phase
                                    n_features = (obs_len - 4) // 3
                                    if n_features > 0:
                                        # P,C come from obs_precision_coverage (quantile layout).
                                        precision, coverage = obs_precision_coverage(final_obs_np)
                                        
                                        # Store data keyed by agent name to distinguish between agents
                                        episode_data[agent_name] = {
                                            "anchor_precision": precision,
                                            "anchor_coverage": coverage,
                                            "total_reward": 0.0,  # Not available from observation
                                            "final_observation": final_obs_np.tolist(),
                                            "group": group,  # Keep track of which group this agent belongs to
                                            "target_class": unwrapped_env.agent_to_class.get(agent_name, None) if unwrapped_env is not None and hasattr(unwrapped_env, 'agent_to_class') else None,
                                        }
                                        
                                        # If we also have info, try to get total_reward from it
                                        if info is not None:
                                            try:
                                                if hasattr(info, 'shape') and info.shape[0] > 0:
                                                    final_info = info[-1]
                                                elif isinstance(info, (list, tuple)) and len(info) > 0:
                                                    final_info = info[-1]
                                                elif isinstance(info, dict):
                                                    final_info = info
                                                else:
                                                    final_info = info
                                                
                                                def safe_get(key, default=0.0):
                                                    try:
                                                        if isinstance(final_info, dict):
                                                            return float(final_info.get(key, default))
                                                        elif hasattr(final_info, 'get'):
                                                            val = final_info.get(key, default)
                                                        elif hasattr(final_info, 'keys') and key in final_info.keys():
                                                            val = final_info[key]
                                                        else:
                                                            val = getattr(final_info, key, default)
                                                        
                                                        if isinstance(val, torch.Tensor):
                                                            return float(val.item() if val.numel() == 1 else val[-1].item())
                                                        return float(val)
                                                    except:
                                                        return default
                                                
                                                episode_data[group]["total_reward"] = safe_get("total_reward", 0.0)
                                            except:
                                                pass  # Keep default 0.0 if info extraction fails
                            
                            # If we didn't get data from observation, try info only
                            elif info is not None:
                                # Get final info (last step)
                                if hasattr(info, 'shape') and info.shape[0] > 0:
                                    final_info = info[-1]
                                elif isinstance(info, (list, tuple)) and len(info) > 0:
                                    final_info = info[-1]
                                elif isinstance(info, dict):
                                    final_info = info
                                else:
                                    final_info = info
                                
                                # Extract metrics
                                def safe_get(key, default=0.0):
                                    try:
                                        if isinstance(final_info, dict):
                                            val = final_info.get(key, default)
                                        elif hasattr(final_info, 'get'):
                                            val = final_info.get(key, default)
                                        elif hasattr(final_info, 'keys') and key in final_info.keys():
                                            val = final_info[key]
                                        else:
                                            val = getattr(final_info, key, default)
                                        
                                        if isinstance(val, torch.Tensor):
                                            return float(val.item() if val.numel() == 1 else val[-1].item())
                                        return float(val)
                                    except:
                                        return default
                                
                                # Store data keyed by agent name to distinguish between agents
                                precision_val = safe_get("anchor_precision", 0.0)
                                coverage_val = safe_get("anchor_coverage", 0.0)
                                episode_data[agent_name] = {
                                    "anchor_precision": precision_val,
                                    "anchor_coverage": coverage_val,
                                    "total_reward": safe_get("total_reward", 0.0),
                                    "group": group,  # Keep track of which group this agent belongs to
                                    "target_class": unwrapped_env.agent_to_class.get(agent_name, None) if unwrapped_env is not None and hasattr(unwrapped_env, 'agent_to_class') else None,
                                }
                                
                                # Try to get observation for bounds
                                if obs is not None:
                                    if hasattr(obs, 'shape') and obs.shape[0] > 0:
                                        final_obs = obs[-1]
                                    elif isinstance(obs, (list, tuple)) and len(obs) > 0:
                                        final_obs = obs[-1]
                                    else:
                                        final_obs = obs
                                    
                                    if isinstance(final_obs, torch.Tensor):
                                        episode_data[agent_name]["final_observation"] = final_obs.cpu().numpy().tolist()
                                    elif isinstance(final_obs, np.ndarray):
                                        episode_data[agent_name]["final_observation"] = final_obs.tolist()
                                    else:
                                        episode_data[agent_name]["final_observation"] = list(final_obs) if hasattr(final_obs, '__iter__') else [final_obs]
                    
                    if episode == 0:
                        logger.info(f"  Debug: Episode {episode} completed, step_count={step_count}, done={done}")
                        logger.info(f"  Debug: episode_data keys: {list(episode_data.keys())}")
                        if not episode_data:
                            logger.info(f"  Debug: td keys: {list(td.keys()) if hasattr(td, 'keys') else 'N/A'}")
                            if "next" in td.keys():
                                next_td = td["next"]
                                logger.info(f"  Debug: next_td keys: {list(next_td.keys()) if hasattr(next_td, 'keys') else 'N/A'}")
                                # Check what's in the agent group
                                for group in algorithm.group_map.keys():
                                    if group in next_td.keys():
                                        group_data = next_td[group]
                                        logger.info(f"  Debug: next_td['{group}'] type: {type(group_data)}")
                                        if hasattr(group_data, 'keys'):
                                            logger.info(f"  Debug: next_td['{group}'] keys: {list(group_data.keys())}")
                                        elif isinstance(group_data, dict):
                                            logger.info(f"  Debug: next_td['{group}'] dict keys: {list(group_data.keys())}")
                            # Also check td directly (not just next)
                            for group in algorithm.group_map.keys():
                                if group in td.keys():
                                    group_data = td[group]
                                    logger.info(f"  Debug: td['{group}'] type: {type(group_data)}")
                                    if hasattr(group_data, 'keys'):
                                        logger.info(f"  Debug: td['{group}'] keys: {list(group_data.keys())}")
                                        # Check if info is nested deeper
                                        for key in group_data.keys():
                                            if hasattr(group_data[key], 'keys'):
                                                logger.info(f"  Debug: td['{group}']['{key}'] keys: {list(group_data[key].keys())}")
                                    elif isinstance(group_data, dict):
                                        logger.info(f"  Debug: td['{group}'] dict keys: {list(group_data.keys())}")
                    
                    # Add timing to episode_data
                    if episode_data:
                        # Add timing to each agent's episode data
                        for agent_name in episode_data.keys():
                            if isinstance(episode_data[agent_name], dict):
                                episode_data[agent_name]["rollout_time_seconds"] = float(episode_duration)
                        
                        manual_rollouts.append(episode_data)
                        if hasattr(self.callback, 'evaluation_anchor_data'):
                            self.callback.evaluation_anchor_data.append(episode_data)
                
                eval_end_time = time.perf_counter()
                eval_total_time = eval_end_time - eval_start_time
                avg_episode_time = eval_total_time / n_eval_episodes if n_eval_episodes > 0 else 0.0
                
                logger.info(f"   Collected {len(manual_rollouts)} episodes from manual rollouts")
                logger.info(f"   Evaluation time: {eval_total_time:.4f}s total, {avg_episode_time:.4f}s per episode")
                evaluation_anchor_data.extend(manual_rollouts)
                
            except Exception as e:
                logger.warning(f"   Warning: Could not run manual rollouts: {e}")
                logger.warning(f"    Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
        
        # Final summary
        if not evaluation_anchor_data:
            logger.warning("\n   Warning: No anchor data collected from any evaluation.")
            logger.warning("  This may happen if:")
            logger.warning("    1. Training evaluations didn't run (check evaluation_interval in config)")
            logger.warning("    2. Manual rollouts failed (see error above)")
            logger.warning("  Consider using inference.py to extract rules directly from the trained model.")
        else:
            logger.info(f"\n   Total episodes collected: {len(evaluation_anchor_data)}")
        
        # Save callback data to files (including any new evaluation data)
        if hasattr(self, 'callback') and self.callback is not None:
            if hasattr(self.callback, 'save_data_to_files'):
                logger.info("\nSaving callback data to files...")
                try:
                    saved_files = self.callback.save_data_to_files(str(self.experiment.folder_name))
                    if saved_files:
                        logger.info(f"   Saved {len(saved_files)} data files")
                    else:
                        logger.info("  No callback data to save")
                except Exception as e:
                    logger.warning(f"   Warning: Could not save callback data: {e}")
        
        return {
            "experiment_folder": str(self.experiment.folder_name),
            "total_frames": self.experiment.total_frames,
            "n_iters_performed": self.experiment.n_iters_performed,
            "evaluation_anchor_data": evaluation_anchor_data,
            "evaluation_history": self.callback.get_evaluation_history() if hasattr(self.callback, 'get_evaluation_history') else []
        }
    
    def get_checkpoint_path(self) -> str:
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        
        experiment_folder = str(self.experiment.folder_name)
        logger.info(f"BenchMARL experiment folder: {experiment_folder}")
        logger.info(f"  This folder contains all checkpoints, logs, and model states")
        logger.info(f"  Use this path to load checkpoints later with BenchMARL's load_state_dict()")
        
        return experiment_folder
    
    # This is needed for easier inference. Inference with BenchMARL is complicated and couldn't get it to work
    def extract_best_and_final_models(
        self,
        save_policies: bool = True,
        save_critics: bool = False,
    ) -> Dict[str, Dict[str, str]]:
        """Extract per-agent individual models from BOTH the final training
        state and the best checkpoint.

        Produces, under the experiment folder:
          individual_models/        <- final training state (always)
          individual_models_best/   <- best checkpoint state (only if
                                       best_model/best_checkpoint.pt exists)

        Gives the multi-agent pipeline the same best/final choice the
        single-agent pipeline has. Returns {"final": {...}, "best": {...}}.
        """
        results: Dict[str, Dict[str, str]] = {}

        # 1. Final state — whatever the experiment currently holds.
        self.save_training_query_count()   # G-04
        results["final"] = self.extract_and_save_individual_models(
            save_policies=save_policies, save_critics=save_critics,
            models_subdir="individual_models",
        )

        # 2. Best actors. Prefer the copy extracted at save-best time (in-memory
        # actors, no Experiment.state_dict). Fall back to reloading the .pt.
        best_dir = os.path.join(str(self.experiment.folder_name), "individual_models_best")
        best_index = os.path.join(best_dir, "policies_index.json")
        if os.path.isfile(best_index):
            logger.info(
                "individual_models_best already extracted at save-best time; keeping %s",
                best_dir,
            )
            results["best"] = {"existing": best_dir}
            return results

        best_ckpt = os.path.join(str(self.experiment.folder_name), "best_model", "best_checkpoint.pt")
        if not os.path.exists(best_ckpt):
            logger.info("No best_model/best_checkpoint.pt found — only final individual_models extracted.")
            results["best"] = {}
            return results

        logger.info(f"Best checkpoint found — extracting individual_models_best from: {best_ckpt}")
        final_state = None
        try:
            final_state = self.experiment.state_dict()
            best_state = torch.load(best_ckpt, map_location="cpu")
            self.experiment.load_state_dict(best_state)
            results["best"] = self.extract_and_save_individual_models(
                save_policies=save_policies, save_critics=save_critics,
                models_subdir="individual_models_best",
            )
        except Exception as e:
            logger.error(
                "Could not extract individual_models_best from checkpoint (%s). "
                "Inference with prefer_model=best will fail until this is fixed.",
                e,
            )
            results["best"] = {}
        finally:
            if final_state is not None:
                try:
                    self.experiment.load_state_dict(final_state)
                except Exception as e:
                    logger.warning(f"Could not restore final experiment state after best extraction: {e}")

        return results

    def save_training_query_count(self) -> Optional[str]:
        """Write training_queries.json: black-box calls spent TRAINING.

        G-04: the break-even figure treated construction cost as extraction
        only. Policy training issues n_perturb_train classifier calls per env
        step across the whole frame budget -- the dominant term, and the only
        reason a break-even point exists. AnchorEnv already counts these in
        n_blackbox_queries; nothing persisted them.

        The collector holds its own env copies, so the count is read from every
        reachable env instance and summed.
        """
        if self.experiment is None:
            return None
        total = 0
        seen = set()

        def _harvest(obj, depth=0):
            nonlocal total
            if obj is None or depth > 6 or id(obj) in seen:
                return
            seen.add(id(obj))
            n = getattr(obj, "n_blackbox_queries", None)
            if isinstance(n, (int, float)):
                total += int(n)
                return
            for attr in ("env", "_env", "base_env", "_envs", "envs"):
                inner = getattr(obj, attr, None)
                if isinstance(inner, (list, tuple)):
                    for e in inner:
                        _harvest(e, depth + 1)
                elif inner is not None:
                    _harvest(inner, depth + 1)

        for holder in ("env_func", "train_env", "test_env", "collector"):
            _harvest(getattr(self.experiment, holder, None))

        # Two regimes, and the honest figure differs between them:
        #
        #  precision_estimator == "conditional": every agent-step draws
        #    n_perturb_train perturbations and classifies them, so the cost is
        #    max_n_frames (the PER-AGENT step budget) x n_agents x n_perturb_train.
        #    The live counters cannot see this because the collector holds env
        #    copies in worker processes, so the analytic figure is primary.
        #
        #  precision_estimator == "empirical" (the shipped config): precision is
        #    measured on real rows via cached split predictions, so there is NO
        #    per-step perturbation cost and the analytic formula would overstate
        #    by orders of magnitude. The observed counter is the right basis.
        #
        # Getting this wrong in either direction misstates the break-even point,
        # so the regime is recorded alongside the number.
        env_cfg = self._anchor_env_config or {}
        if "env_config" in env_cfg:
            env_cfg = env_cfg["env_config"]
        n_perturb_train = int(env_cfg.get("n_perturb_train", 0) or 0)
        n_agents = 0
        try:
            n_agents = sum(len(v) for v in self.experiment.group_map.values())
        except Exception:
            pass
        if n_agents <= 0:
            n_agents = int(env_cfg.get("agents_per_class", 1) or 1)
        frames = int(getattr(self.experiment_config, "max_n_frames", 0) or 0)
        estimator = str(env_cfg.get("precision_estimator", "empirical")).lower()
        perturbation_based = estimator == "conditional"
        analytic = int(frames * n_agents * n_perturb_train) if perturbation_based else 0

        path = os.path.join(str(self.experiment.folder_name), "training_queries.json")
        payload = {
            "n_training_queries": int(analytic if perturbation_based else total),
            "n_training_queries_observed": int(total),
            "n_training_queries_analytic": int(analytic),
            "precision_estimator": estimator,
            "basis": (
                "analytic: max_n_frames x n_agents x n_perturb_train "
                "(perturbation-based estimator)"
                if perturbation_based else
                "observed: AnchorEnv.n_blackbox_queries; the empirical estimator "
                "scores real rows from cached split predictions, so there is no "
                "per-step perturbation cost"
            ),
            "max_n_frames": frames,
            "n_agents": int(n_agents),
            "n_perturb_train": n_perturb_train,
            "complete": bool(analytic > 0 if perturbation_based else total > 0),
        }
        if not perturbation_based:
            payload["note"] = (
                "LOWER BOUND: the collector holds env copies in worker processes "
                "that this process cannot read. Under the empirical estimator the "
                "per-step cost is ~0, so the shortfall is small, but the figure is "
                "not exact."
            )
        import json as _json
        with open(path, "w") as f:
            _json.dump(payload, f, indent=2)
        logger.info(
            f"Training black-box queries: {payload['n_training_queries']} "
            f"({payload['basis'].split(':')[0]}, estimator={estimator}) -> {path}"
        )
        return path

    def extract_and_save_individual_models(
        self,
        output_dir: Optional[str] = None,
        save_policies: bool = True,
        save_critics: bool = False,
        models_subdir: str = "individual_models",
    ) -> Dict[str, str]:
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )

        algorithm = self.experiment.algorithm

        # Save in the experiment's run log directory. models_subdir lets the
        # caller target a separate folder (e.g. "individual_models_best" when
        # extracting from the best checkpoint — see extract_best_and_final_models).
        output_dir = os.path.join(str(self.experiment.folder_name), models_subdir)

        os.makedirs(output_dir, exist_ok=True)
        
        saved_models = {}
        
        logger.info("\n" + "="*80)
        logger.info("EXTRACTING INDIVIDUAL MODELS FOR STANDALONE INFERENCE")
        logger.info("="*80)
        
        # Get environment to access agent-to-class mapping
        env = self._create_env_instance()
        # Set mode to "inference" for rule extraction
        unwrapped_env = None
        if hasattr(env, 'env') or hasattr(env, '_env'):
            unwrapped_env = getattr(env, 'env', None) or getattr(env, '_env', None)
        if unwrapped_env is not None and hasattr(unwrapped_env, 'mode'):
            unwrapped_env.mode = "inference"
        elif hasattr(env, 'mode'):
            env.mode = "inference"
        
        if hasattr(env, 'env') or hasattr(env, '_env'):
            unwrapped_env = getattr(env, 'env', None) or getattr(env, '_env', None)
        
        # Determine if we have multiple agents per class
        agents_per_class = 1
        if unwrapped_env is not None and hasattr(unwrapped_env, 'agents_per_class'):
            agents_per_class = unwrapped_env.agents_per_class
        
        # Debug: Log environment and algorithm group information
        logger.info(f"\nDebug: Environment agents_per_class = {agents_per_class}")
        if unwrapped_env is not None:
            if hasattr(unwrapped_env, 'possible_agents'):
                logger.info(f"Debug: Environment possible_agents = {unwrapped_env.possible_agents}")
            if hasattr(unwrapped_env, 'agent_to_class'):
                logger.info(f"Debug: Environment agent_to_class = {unwrapped_env.agent_to_class}")
            if hasattr(unwrapped_env, 'group_map'):
                logger.info(f"Debug: Environment group_map keys = {list(unwrapped_env.group_map.keys())}")
        
        logger.info(f"Debug: Algorithm group_map keys = {list(algorithm.group_map.keys())}")
        logger.info(f"Debug: Algorithm group_map = {algorithm.group_map}")
        
        # Track policies by class for organization
        policies_by_class = {}
        
        # Check if algorithm group_map matches expected number of agents
        expected_num_groups = len(unwrapped_env.possible_agents) if unwrapped_env and hasattr(unwrapped_env, 'possible_agents') else None
        actual_num_groups = len(algorithm.group_map.keys())
        
        # Determine agents_per_class to provide context
        agents_per_class = 1
        if unwrapped_env is not None and hasattr(unwrapped_env, 'agents_per_class'):
            agents_per_class = unwrapped_env.agents_per_class
        
        if expected_num_groups is not None and actual_num_groups < expected_num_groups:
            # This is expected when agents_per_class > 1 (agents are grouped by class)
            if agents_per_class > 1:
                logger.info(f"\n Info: Algorithm groups agents by class (agents_per_class={agents_per_class}).")
                logger.info(f"  Algorithm has {actual_num_groups} groups (one per class) but environment has {expected_num_groups} agents ({agents_per_class} per class).")
                logger.info(f"  Algorithm groups: {list(algorithm.group_map.keys())}")
                logger.info(f"  Environment agents: {unwrapped_env.possible_agents if unwrapped_env else 'N/A'}")
                logger.info(f"  This is expected - BenchMARL groups agents by class when agents_per_class > 1.")
            else:
                logger.warning(f"\n Warning: Algorithm has {actual_num_groups} groups but environment has {expected_num_groups} agents.")
                logger.warning(f"  This suggests agents may be grouped together. Each group may contain multiple agents.")
                logger.warning(f"  Algorithm groups: {list(algorithm.group_map.keys())}")
                logger.warning(f"  Environment agents: {unwrapped_env.possible_agents if unwrapped_env else 'N/A'}")
        
        for group in algorithm.group_map.keys():
            # Get all agents in this group
            agents_in_group = algorithm.group_map.get(group, [group])
            logger.info(f"\nExtracting models for group: {group} (contains {len(agents_in_group)} agent(s): {agents_in_group})")
            
            # If group contains multiple agents, we need to handle each separately
            # However, in MADDPG, each agent typically has its own policy even if grouped
            # So we'll extract one policy per group, but save it with proper agent names
            
            # Determine which class this agent/group belongs to
            # Use the first agent in the group to determine class
            primary_agent = agents_in_group[0] if agents_in_group else group
            target_class = None
            agent_name = primary_agent  # Use primary agent name
            
            if unwrapped_env is not None and hasattr(unwrapped_env, 'agent_to_class'):
                target_class = unwrapped_env.agent_to_class.get(primary_agent, None)
                if target_class is None:
                    # Try to parse from agent name (e.g., "agent_0" -> 0, "agent_0_1" -> 0)
                    try:
                        parts = primary_agent.split("_")
                        if len(parts) >= 2 and parts[1].isdigit():
                            target_class = int(parts[1])
                    except:
                        pass
            
            # Convert numpy types to native Python types for JSON serialization
            def _convert_to_serializable(obj: Any) -> Any:
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.int_, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: _convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [_convert_to_serializable(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool)) or obj is None:
                    return obj
                else:
                    return str(obj)
            
            if save_policies:
                try:
                    # Get policy using BenchMARL's official API
                    policy = algorithm.get_policy_for_loss(group)
                    
                    # Extract the underlying neural network module
                    # Policy is ProbabilisticActor wrapping actor_module
                    actor_module = None
                    if hasattr(policy, "module"):
                        actor_module = policy.module
                    elif hasattr(policy, "actor_network"):
                        actor_module = policy.actor_network
                    elif hasattr(policy, "net"):
                        actor_module = policy.net
                    
                    if actor_module is None:
                        logger.warning(f"   Warning: Could not extract actor module from policy for group {group}")
                        logger.warning(f"    Policy type: {type(policy)}")
                        logger.warning(f"    Policy attributes: {dir(policy)}")
                    else:
                        # If group contains multiple agents, save a policy for each agent
                        # (In most cases, each agent has its own policy even if grouped)
                        agents_to_save = agents_in_group if len(agents_in_group) > 1 else [primary_agent]
                        
                        for agent_to_save in agents_to_save:
                            # Determine class for this specific agent
                            agent_target_class = target_class
                            if unwrapped_env is not None and hasattr(unwrapped_env, 'agent_to_class'):
                                agent_target_class = unwrapped_env.agent_to_class.get(agent_to_save, target_class)
                            
                            # Determine save location based on agents_per_class
                            if agents_per_class > 1 and agent_target_class is not None:
                                # Organize by class when multiple agents per class
                                class_dir = os.path.join(output_dir, f"class_{agent_target_class}")
                                os.makedirs(class_dir, exist_ok=True)
                                policy_path = os.path.join(class_dir, f"policy_{agent_to_save}.pth")
                                metadata_path = os.path.join(class_dir, f"policy_{agent_to_save}_metadata.json")
                            else:
                                # Flat structure when one agent per class
                                policy_path = os.path.join(output_dir, f"policy_{agent_to_save}.pth")
                                metadata_path = os.path.join(output_dir, f"policy_{agent_to_save}_metadata.json")
                            
                            # With share_policy_params: False the group's
                            # MultiAgentMLP stores a LEADING AGENT DIM on every
                            # parameter (params.0.weight is [n_agents, out, in]).
                            # Saving the raw group state_dict would hand every
                            # agent the whole stack, and load_policy_model --
                            # which infers in_features from weight.shape[1] --
                            # would build the wrong MLP and fail to load.
                            # Slice out this agent's own parameters so each file
                            # is a standalone single-agent actor.
                            agent_idx_in_group = (
                                agents_in_group.index(agent_to_save)
                                if agent_to_save in agents_in_group else 0
                            )
                            agent_state_dict = _slice_agent_params(
                                actor_module.state_dict(),
                                agent_idx=agent_idx_in_group,
                                n_agents=len(agents_in_group),
                            )
                            torch.save(agent_state_dict, policy_path)
                            saved_models[f"policy_{agent_to_save}"] = policy_path
                            logger.info(f"   Saved policy model to: {policy_path}")
                            
                            # Track by class
                            if agent_target_class is not None:
                                if agent_target_class not in policies_by_class:
                                    policies_by_class[agent_target_class] = []
                                policies_by_class[agent_target_class].append({
                                    "agent": agent_to_save,
                                    "group": group,
                                    "policy_path": policy_path,
                                    "metadata_path": metadata_path
                                })
                            
                            # Also save metadata about the model structure with class information
                            metadata = {
                                "group": group,
                                "agent": agent_to_save,
                                "target_class": agent_target_class,
                                "agents_per_class": agents_per_class,
                                "agents_in_group": agents_in_group,
                                "model_type": "policy",
                                "algorithm": self.algorithm,
                                "input_spec": str(getattr(actor_module, "input_spec", None)),
                                "output_spec": str(getattr(actor_module, "output_spec", None)),
                            }
                            
                            import json
                            serializable_metadata = _convert_to_serializable(metadata)
                            with open(metadata_path, 'w') as f:
                                json.dump(serializable_metadata, f, indent=2)
                            logger.info(f"   Saved policy metadata to: {metadata_path}")
                            if agent_target_class is not None:
                                logger.info(f"    Class: {agent_target_class}, Agent: {agent_to_save}")
                        
                except Exception as e:
                    logger.warning(f"  ✗ Error extracting policy for group {group}: {e}")
            
            if save_critics:
                try:
                    if hasattr(algorithm, "get_value_module"):
                        critic = algorithm.get_value_module(group)
                        
                        # Extract the underlying neural network module
                        critic_module = None
                        if hasattr(critic, "module"):
                            critic_module = critic.module
                        elif hasattr(critic, "value_network"):
                            critic_module = critic.value_network
                        elif hasattr(critic, "net"):
                            critic_module = critic.net
                        
                        if critic_module is None:
                            logger.warning(f"   Warning: Could not extract critic module for group {group}")
                        else:
                            # Save the critic module
                            critic_path = os.path.join(output_dir, f"critic_{group}.pth")
                            torch.save(critic_module.state_dict(), critic_path)
                            saved_models[f"critic_{group}"] = critic_path
                            logger.info(f"   Saved critic model to: {critic_path}")
                    else:
                        logger.warning(f"   Algorithm {self.algorithm} does not have get_value_module() method")
                        
                except Exception as e:
                    logger.warning(f"  ✗ Error extracting critic for group {group}: {e}")
        
        # Save an index file mapping classes to policies
        if policies_by_class:
            import json
            index_data = {
                "agents_per_class": agents_per_class,
                "seed": self.seed,  # Save seed to match training seed during inference
                "policies_by_class": {}
            }
            for class_id, policies in sorted(policies_by_class.items()):
                index_data["policies_by_class"][f"class_{class_id}"] = {
                    "class": int(class_id),
                    "n_agents": len(policies),
                    "policies": [
                        {
                            "agent": p["agent"],
                            "group": p["group"],
                            "policy_file": os.path.relpath(p["policy_path"], output_dir),
                            "metadata_file": os.path.relpath(p["metadata_path"], output_dir)
                        }
                        for p in policies
                    ]
                }
            
            index_path = os.path.join(output_dir, "policies_index.json")
            serializable_index_data = _convert_to_serializable(index_data)
            with open(index_path, 'w') as f:
                json.dump(serializable_index_data, f, indent=2)
            logger.info(f"\n   Saved policies index to: {index_path}")
            logger.info(f"    Classes: {len(policies_by_class)}, Total policies: {sum(len(p) for p in policies_by_class.values())}")
        
        logger.info("\n" + "="*80)
        logger.info(f"Individual models saved to: {output_dir}")
        logger.info("="*80)
        if agents_per_class > 1:
            logger.info(f"\nNote: Multiple agents per class ({agents_per_class}) detected.")
            logger.info(f"  Policies are organized in class-specific directories: class_0/, class_1/, etc.")
        logger.info("\nTo use these models for inference:")
        logger.info("  1. Load the model state_dict")
        logger.info("  2. Reconstruct the model architecture")
        logger.info("  3. Load state_dict into the model")
        logger.info("  4. Use model.eval() and model(observation) for inference")
        logger.info("\n  See policies_index.json for a mapping of classes to policy files.")
        
        return saved_models
    
    
    def reload_experiment(self, checkpoint_file: str):
        from benchmarl.hydra_config import reload_experiment_from_file
        
        # If directory provided, find the checkpoint file
        if os.path.isdir(checkpoint_file):
            experiment_dir = checkpoint_file
            # Check for checkpoints in the checkpoints subdirectory (BenchMARL standard location)
            checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
            if os.path.exists(checkpoints_dir):
                checkpoint_files = [
                    f for f in os.listdir(checkpoints_dir)
                    if (f.endswith('.pt') or f.endswith('.pth'))
                ]
                if checkpoint_files:
                    # Use the most recent checkpoint
                    checkpoint_file = os.path.join(
                        checkpoints_dir,
                        max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoints_dir, f)))
                    )
            
            # If not found in checkpoints subdirectory, check root directory
            if not os.path.isfile(checkpoint_file):
                all_files = os.listdir(experiment_dir)
                checkpoint_files = [
                    f for f in all_files 
                    if (f.endswith('.pt') or f.endswith('.pth')) 
                    and f != 'classifier.pth'
                    and not f.startswith('classifier')
                ]
                if checkpoint_files:
                    checkpoint_file = os.path.join(
                        experiment_dir, 
                        max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(experiment_dir, f)))
                    )
        
        if not os.path.isfile(checkpoint_file):
            raise ValueError(
                f"Checkpoint file not found: {checkpoint_file}. "
                f"Please provide a valid checkpoint file path."
            )
        
        logger.info(f"\n{'='*80}")
        logger.info("RELOADING ENTIRE EXPERIMENT")
        logger.info(f"{'='*80}")
        logger.info(f"Checkpoint file: {checkpoint_file}")
        logger.info("  Reference: https://benchmarl.readthedocs.io/en/latest/concepts/features.html#reloading")
        
        # Reload experiment from checkpoint (simple and direct as per documentation)
        experiment = reload_experiment_from_file(checkpoint_file)
        
        # Assign the reloaded experiment to the trainer
        self.experiment = experiment
        
        # Extract task from the reloaded experiment
        if hasattr(experiment, 'task'):
            self.task = experiment.task
        
        # Find the callback if it exists in the experiment
        if hasattr(experiment, 'callbacks') and experiment.callbacks:
            for cb in experiment.callbacks:
                if hasattr(cb, 'get_evaluation_anchor_data'):
                    self.callback = cb
                    break
        
        logger.info("   Experiment reloaded successfully")
        logger.info(f"  Resuming from iteration: {self.experiment.n_iters_performed}")
        logger.info(f"  Total frames: {self.experiment.total_frames}")
    
    def get_experiment(self) -> Experiment:
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        return self.experiment
    
    def _create_env_instance(self, device=None):
        if self.experiment is None or self.task is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        
        # Get device from parameter, algorithm, experiment config, or use default
        if device is None:
            device = "cpu"
            # Try to get device from algorithm (most reliable)
            if hasattr(self.experiment, 'algorithm') and self.experiment.algorithm is not None:
                algorithm = self.experiment.algorithm
                # Try to get device from policy parameters
                try:
                    for group in algorithm.group_map.keys():
                        policy = algorithm.get_policy_for_loss(group)
                        if hasattr(policy, 'parameters'):
                            # Get device from first parameter
                            for param in policy.parameters():
                                if param is not None:
                                    device = str(param.device)
                                    break
                        if device != "cpu":
                            break
                except Exception:
                    pass
            
            # Fallback to config if algorithm device not found
            if device == "cpu":
                if hasattr(self.experiment_config, 'device'):
                    device = self.experiment_config.device
                elif hasattr(self.experiment, 'device'):
                    device = self.experiment.device
        
        # Convert device to string if it's a torch.device
        if hasattr(device, 'type'):
            device = str(device)
        
        # Get seed from experiment or use None
        seed = getattr(self.experiment, 'seed', None)
        
        # Create environment using task's get_env_fun
        env_fun = self.task.get_env_fun(
            num_envs=1,
            continuous_actions=True,
            seed=seed,
            device=device
        )
        
        # Call the factory function to create the environment instance
        return env_fun()
    
    def _load_env_config_from_yaml(self) -> Dict[str, Any]:
        # Return cached config if already loaded
        if self._anchor_env_config is not None:
            return self._anchor_env_config.copy()
        
        # Try to load from YAML file
        config_path = self.anchor_config_path
        if not os.path.isabs(config_path):
            # If relative path, make it relative to the BenchMARL directory
            benchmarl_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(benchmarl_dir, config_path)
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                
                if yaml_data and "env_config" in yaml_data:
                    env_config = yaml_data["env_config"]
                    # Convert boolean strings to actual booleans if needed
                    for key, value in env_config.items():
                        if isinstance(value, str):
                            if value.lower() == "true":
                                env_config[key] = True
                            elif value.lower() == "false":
                                env_config[key] = False
                    
                    # Also load logging_verbosity from top-level YAML if present
                    if "logging_verbosity" in yaml_data:
                        env_config["logging_verbosity"] = yaml_data["logging_verbosity"]
                    
                    logger.info(f"  Loaded environment config from: {config_path}")
                    self._anchor_env_config = env_config.copy()
                    return env_config
                else:
                    logger.warning(f"  Warning: {config_path} exists but doesn't contain 'env_config' key.")
                    raise ValueError(f"{config_path} is missing the env_config mapping.")
            except ValueError:
                raise
            except Exception as e:
                logger.warning(f"  Warning: Could not load config from {config_path}: {e}.")
                raise ValueError(f"Could not load {config_path}: {e}") from e
        else:
            logger.warning(f"  Warning: Anchor config file not found at {config_path}.")
            raise FileNotFoundError(
                f"Environment YAML not found: {config_path}. "
                "YAML is the source of truth; refusing to train on hardcoded defaults."
            )
    
    def _get_default_env_config(self) -> Dict[str, Any]:
        config = self._load_env_config_from_yaml()
        if "logging_verbosity" not in config:
            config["logging_verbosity"] = "normal"
        return config
