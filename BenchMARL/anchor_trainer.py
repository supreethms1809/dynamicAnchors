import numpy as np
import torch
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path
import os
import sys

from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarl_wrappers import AnchorTask, AnchorMetricsCallback
from Environment import AnchorEnv


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


class AnchorTrainer:
    
    ALGORITHM_MAP = _get_algorithm_configs()
    
    def __init__(
        self,
        dataset_loader,
        algorithm: str = "maddpg",
        algorithm_config_path: Optional[str] = None,
        experiment_config_path: str = "conf/base_experiment.yaml",
        mlp_config_path: str = "conf/mlp.yaml",
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
        self.output_dir = output_dir
        self.seed = seed
        
        self.experiment = None
        self.experiment_config = None
        self.algorithm_config = None
        self.model_config = None
        self.critic_model_config = None
        self.task = None
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setup_experiment(
        self,
        env_config: Optional[Dict[str, Any]] = None,
        target_classes: Optional[List[int]] = None,
        max_cycles: int = 100,
        device: str = "cpu",
        eval_on_test_data: bool = False
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
        
        print("\n" + "="*80)
        print("SETTING UP ANCHOR TRAINING EXPERIMENT")
        print("="*80)
        
        self.experiment_config = ExperimentConfig.get_from_yaml(self.experiment_config_path)
        self.algorithm_config = self.algorithm_config_class.get_from_yaml(self.algorithm_config_path)
        self.model_config = MlpConfig.get_from_yaml(self.mlp_config_path)
        self.critic_model_config = MlpConfig.get_from_yaml(self.mlp_config_path)
        
        env_data = self.dataset_loader.get_anchor_env_data()
        
        if env_config is None:
            env_config = self._get_default_env_config()
        
        if target_classes is None:
            target_classes = list(np.unique(self.dataset_loader.y_train))
        
        env_config_with_data = {
            **env_config,
            "X_min": env_data["X_min"],
            "X_range": env_data["X_range"],
        }
        
        if eval_on_test_data:
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
            print(f"  Evaluation configured to use TEST data")
        else:
            env_config_with_data["eval_on_test_data"] = False
            print(f"  Evaluation configured to use TRAINING data")
        
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
        
        self.callback = AnchorMetricsCallback(
            log_training_metrics=True, 
            log_evaluation_metrics=True,
            save_to_file=True
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
        
        print(f"Experiment setup complete:")
        print(f"  Algorithm: {self.algorithm.upper()}")
        print(f"  Model: MLP")
        print(f"  Target classes: {target_classes}")
        print(f"  Max cycles per episode: {max_cycles}")
        print(f"  Output directory: {self.output_dir}")
        print("="*80)
        
        return self.experiment
    
    def train(self) -> Experiment:
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        
        print("\n" + "="*80)
        print("STARTING ANCHOR TRAINING")
        print("="*80)
        
        self.experiment.run()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"Results saved to: {self.experiment.folder_name}")
        
        return self.experiment
    
    def evaluate(self) -> Dict[str, Any]:
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        
        print("\n" + "="*80)
        print("RUNNING EVALUATION")
        print("="*80)
        
        self.experiment.evaluate()
        
        return {
            "experiment_folder": str(self.experiment.folder_name),
            "total_frames": self.experiment.total_frames,
            "n_iters_performed": self.experiment.n_iters_performed
        }
    
    def save_checkpoint(self, filepath: Optional[str] = None):
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        
        if filepath is None:
            filepath = os.path.join(
                self.output_dir,
                f"checkpoint_{self.experiment.total_frames}.pt"
            )
        
        checkpoint_dir = os.path.dirname(filepath)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        state_dict = self.experiment.state_dict()
        torch.save(state_dict, filepath)
        
        print(f"Checkpoint saved to: {filepath}")
        return filepath
    
    def load_checkpoint(self, filepath: str):
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        
        state_dict = torch.load(filepath, map_location="cpu")
        self.experiment.load_state_dict(state_dict)
        
        print(f"Checkpoint loaded from: {filepath}")
        print(f"Resuming from iteration: {self.experiment.n_iters_performed}")
        print(f"Total frames: {self.experiment.total_frames}")
    
    def get_experiment(self) -> Experiment:
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        return self.experiment
    
    def extract_rules(
        self,
        max_features_in_rule: Optional[int] = 5,
        steps_per_episode: int = 100,
        n_instances_per_class: int = 20,
        eval_on_test_data: bool = False
    ) -> Dict[str, Any]:
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        
        print("\n" + "="*80)
        print("EXTRACTING ANCHOR RULES")
        print("="*80)
        
        env_data = self.dataset_loader.get_anchor_env_data()
        target_classes = list(np.unique(self.dataset_loader.y_train))
        
        results = {
            "per_class_results": {},
            "metadata": {
                "dataset": self.dataset_loader.dataset_name,
                "algorithm": self.algorithm,
                "target_classes": target_classes,
                "max_features_in_rule": max_features_in_rule,
            }
        }
        
        for target_class in target_classes:
            agent = f"agent_{target_class}"
            class_key = f"class_{target_class}"
            
            print(f"\nExtracting rules for class {target_class} (agent: {agent})...")
            
            env_config = self._get_default_env_config()
            env_config.update({
                "X_min": env_data["X_min"],
                "X_range": env_data["X_range"],
            })
            
            if eval_on_test_data:
                if env_data.get("X_test_unit") is None or env_data.get("X_test_std") is None or env_data.get("y_test") is None:
                    raise ValueError(
                        "eval_on_test_data=True requires test data. "
                        "Make sure dataset_loader has test data loaded and preprocessed."
                    )
                env_config.update({
                    "eval_on_test_data": True,
                    "X_test_unit": env_data["X_test_unit"],
                    "X_test_std": env_data["X_test_std"],
                    "y_test": env_data["y_test"],
                })
            else:
                env_config["eval_on_test_data"] = False
            
            anchor_env = AnchorEnv(
                X_unit=env_data["X_unit"],
                X_std=env_data["X_std"],
                y=env_data["y"],
                feature_names=env_data["feature_names"],
                classifier=self.dataset_loader.get_classifier(),
                device="cpu",
                target_classes=[target_class],
                env_config=env_config
            )
            
            if eval_on_test_data:
                class_mask = (env_data["y_test"] == target_class)
                class_instances = np.where(class_mask)[0]
                X_data_unit = env_data["X_test_unit"]
                data_source_name = "test"
            else:
                class_mask = (env_data["y"] == target_class)
                class_instances = np.where(class_mask)[0]
                X_data_unit = env_data["X_unit"]
                data_source_name = "training"
            
            if len(class_instances) == 0:
                print(f"  No instances found for class {target_class} in {data_source_name} data")
                continue
            
            n_samples = min(n_instances_per_class, len(class_instances))
            sampled_indices = np.random.choice(class_instances, size=n_samples, replace=False)
            
            print(f"  Sampling {n_samples} instances from {data_source_name} data for class {target_class}")
            
            rules_list = []
            anchors_list = []
            precisions = []
            coverages = []
            
            for instance_idx in sampled_indices:
                x_instance = X_data_unit[instance_idx]
                
                anchor_env.x_star_unit = {agent: x_instance}
                obs, info = anchor_env.reset(seed=42 + instance_idx)
                
                initial_lower = anchor_env.lower[agent].copy()
                initial_upper = anchor_env.upper[agent].copy()
                
                for step in range(steps_per_episode):
                    obs_tensor = torch.tensor(obs[agent], dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        action_tensor = self.experiment.algorithm.get_action(
                            {agent: obs_tensor},
                            explore=False,
                            step=self.experiment.n_iters_performed
                        )
                        action = action_tensor[agent].squeeze(0).cpu().numpy()
                    
                    obs, rewards, terminations, truncations, infos = anchor_env.step({agent: action})
                    
                    if terminations[agent] or truncations[agent]:
                        break
                
                lower, upper = anchor_env.get_anchor_bounds(agent)
                precision, coverage, _ = anchor_env._current_metrics(agent)
                
                rule = anchor_env.extract_rule(
                    agent,
                    max_features_in_rule=max_features_in_rule,
                    initial_lower=initial_lower,
                    initial_upper=initial_upper
                )
                
                rules_list.append(rule)
                
                anchor_data = {
                    "instance_idx": int(instance_idx),
                    "lower_bounds": lower.tolist(),
                    "upper_bounds": upper.tolist(),
                    "precision": float(precision),
                    "coverage": float(coverage),
                    "rule": rule,
                    "initial_lower_bounds": initial_lower.tolist(),
                    "initial_upper_bounds": initial_upper.tolist(),
                    "final_lower_bounds": lower.tolist(),
                    "final_upper_bounds": upper.tolist(),
                    "box_widths": (upper - lower).tolist(),
                    "box_volume": float(np.prod(np.maximum(upper - lower, 1e-9))),
                }
                anchors_list.append(anchor_data)
                precisions.append(float(precision))
                coverages.append(float(coverage))
            
            unique_rules = list(set([r for r in rules_list if r and r != "any values (no tightened features)"]))
            
            overlap_info = self._check_class_overlaps(target_class, anchors_list, results["per_class_results"])
            
            results["per_class_results"][class_key] = {
                "class": int(target_class),
                "agent_id": agent,
                "precision": float(np.mean(precisions)) if precisions else 0.0,
                "coverage": float(np.mean(coverages)) if coverages else 0.0,
                "precision_std": float(np.std(precisions)) if len(precisions) > 1 else 0.0,
                "coverage_std": float(np.std(coverages)) if len(coverages) > 1 else 0.0,
                "precision_min": float(np.min(precisions)) if precisions else 0.0,
                "precision_max": float(np.max(precisions)) if precisions else 0.0,
                "coverage_min": float(np.min(coverages)) if coverages else 0.0,
                "coverage_max": float(np.max(coverages)) if coverages else 0.0,
                "n_instances_evaluated": len(anchors_list),
                "rules": rules_list,
                "unique_rules": unique_rules,
                "unique_rules_count": len(unique_rules),
                "anchors": anchors_list,
                "overlap_info": overlap_info,
            }
            
            print(f"  Extracted {len(anchors_list)} anchors")
            print(f"  Average precision: {results['per_class_results'][class_key]['precision']:.4f}")
            print(f"  Average coverage: {results['per_class_results'][class_key]['coverage']:.4f}")
            print(f"  Unique rules: {len(unique_rules)}")
            if overlap_info["has_overlaps"]:
                print(f"  ⚠️  Overlaps detected: {overlap_info['n_overlaps']} anchors overlap with other classes")
            else:
                print(f"  ✓ No overlaps with other classes")
        
        print("="*80)
        
        results["training_history"] = self.callback.get_training_history() if hasattr(self, 'callback') else []
        results["evaluation_history"] = self.callback.get_evaluation_history() if hasattr(self, 'callback') else []
        
        return results
    
    def _check_class_overlaps(
        self, 
        target_class: int, 
        anchors_list: List[Dict[str, Any]], 
        existing_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        overlap_info = {
            "has_overlaps": False,
            "n_overlaps": 0,
            "overlapping_anchors": [],
        }
        
        if len(existing_results) == 0:
            return overlap_info
        
        for anchor in anchors_list:
            lower = np.array(anchor["lower_bounds"])
            upper = np.array(anchor["upper_bounds"])
            anchor_vol = float(np.prod(np.maximum(upper - lower, 1e-9)))
            
            if anchor_vol <= 1e-12:
                continue
            
            overlaps_with = []
            
            for other_class_key, other_class_data in existing_results.items():
                if "anchors" not in other_class_data:
                    continue
                
                for other_anchor in other_class_data["anchors"]:
                    other_lower = np.array(other_anchor["lower_bounds"])
                    other_upper = np.array(other_anchor["upper_bounds"])
                    
                    inter_lower = np.maximum(lower, other_lower)
                    inter_upper = np.minimum(upper, other_upper)
                    inter_widths = np.maximum(inter_upper - inter_lower, 0.0)
                    inter_vol = float(np.prod(np.maximum(inter_widths, 0.0)))
                    
                    if inter_vol > 1e-12:
                        overlap_ratio = inter_vol / (anchor_vol + 1e-12)
                        overlaps_with.append({
                            "class": other_class_key,
                            "instance_idx": other_anchor.get("instance_idx", -1),
                            "overlap_ratio": float(overlap_ratio),
                            "overlap_volume": float(inter_vol),
                        })
            
            if overlaps_with:
                overlap_info["has_overlaps"] = True
                overlap_info["n_overlaps"] += 1
                overlap_info["overlapping_anchors"].append({
                    "instance_idx": anchor.get("instance_idx", -1),
                    "overlaps_with": overlaps_with,
                })
        
        return overlap_info
    
    def save_rules(self, results: Dict[str, Any], filepath: Optional[str] = None):
        import json
        
        if filepath is None:
            filepath = os.path.join(
                self.output_dir,
                "extracted_rules.json"
            )
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        serializable_results = self._convert_to_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        n_anchors_total = sum(
            len(class_data.get("anchors", []))
            for class_data in serializable_results.get("per_class_results", {}).values()
        )
        n_rules_total = sum(
            len(class_data.get("rules", []))
            for class_data in serializable_results.get("per_class_results", {}).values()
        )
        
        print(f"Rules and anchors saved to: {filepath}")
        print(f"  Total anchors saved: {n_anchors_total}")
        print(f"  Total rules saved: {n_rules_total}")
        return filepath
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    def _get_default_env_config(self) -> Dict[str, Any]:
        return {
            "precision_target": 0.8,
            "coverage_target": 0.02,
            "use_perturbation": False,
            "perturbation_mode": "bootstrap",
            "n_perturb": 1024,
            "step_fracs": (0.005, 0.01, 0.02),
            "min_width": 0.05,
            "alpha": 0.7,
            "beta": 0.6,
            "gamma": 0.1,
            "precision_blend_lambda": 0.5,
            "drift_penalty_weight": 0.05,
            "min_coverage_floor": 0.005,
            "js_penalty_weight": 0.05,
            "initial_window": 0.1,
            "max_action_scale": 0.1,
            "min_absolute_step": 0.001,
            "inter_class_overlap_weight": 0.1,
        }

