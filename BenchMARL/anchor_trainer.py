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
        max_cycles: int = 500,
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
            "max_cycles": max_cycles,  # Ensure max_cycles is in env_config for the environment
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
            save_to_file=True,
            collect_anchor_data=True  # Collect anchor data during evaluation for rule extraction
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
    
    def evaluate(
        self,
        n_eval_episodes: Optional[int] = None,
        collect_anchor_data: bool = True
    ) -> Dict[str, Any]:
        """
        Run BenchMARL evaluation and collect anchor data.
        
        Note: BenchMARL's evaluate() method doesn't pass rollouts to callbacks after training.
        This method manually runs rollouts to collect evaluation data.
        
        Args:
            n_eval_episodes: Number of episodes to run for evaluation (default: from config)
            collect_anchor_data: Whether to collect anchor data during evaluation
        
        Returns:
            Dictionary containing evaluation results and collected anchor data
        """
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        
        print("\n" + "="*80)
        print("RUNNING EVALUATION")
        print("="*80)
        
        # Get anchor data collected during training evaluations (before clearing)
        # BenchMARL's periodic evaluations during training DO pass rollouts to callbacks
        evaluation_anchor_data = []
        if hasattr(self.callback, 'get_evaluation_anchor_data'):
            evaluation_anchor_data = self.callback.get_evaluation_anchor_data()
            print(f"  Found {len(evaluation_anchor_data)} episodes of anchor data from training evaluations")
        
        # Try BenchMARL's standard evaluate() for metrics logging
        try:
            self.experiment.evaluate()
        except Exception as e:
            # Handle wandb run finished error gracefully
            if "wandb" in str(type(e)).lower() or "finished" in str(e).lower() or "UsageError" in str(type(e)):
                print(f"Warning: Evaluation completed but could not log to wandb (run may be finished): {e}")
                print("Evaluation metrics are still saved to CSV and other loggers.")
            else:
                # Re-raise if it's a different error
                raise
        
        # IMPORTANT: BenchMARL's evaluate() doesn't pass rollouts to callbacks after training
        # So we need to manually run rollouts to collect anchor data
        if collect_anchor_data:
            print(f"\n  Running manual rollouts to collect anchor data...")
            print(f"  (BenchMARL's evaluate() doesn't pass rollouts to callbacks after training)")
            
            # Get number of episodes from config if not provided
            if n_eval_episodes is None:
                n_eval_episodes = self.experiment_config.evaluation_episodes if hasattr(self.experiment_config, 'evaluation_episodes') else 2
            
            # Manually run rollouts using the environment
            try:
                from tensordict import TensorDict
                
                env = self.experiment.env
                algorithm = self.experiment.algorithm
                
                # Run evaluation episodes
                manual_rollouts = []
                for episode in range(n_eval_episodes):
                    # Reset environment
                    td = env.reset()
                    done = False
                    episode_data = {}
                    
                    # Run episode
                    step_count = 0
                    max_steps = self.task.max_steps(env) if hasattr(self.task, 'max_steps') else 100
                    
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
                                        input_td = TensorDict(
                                            {group: {"observation": obs_tensor}},
                                            batch_size=obs_tensor.shape[:1]
                                        )
                                        
                                        # Get action
                                        if hasattr(policy, "forward_inference"):
                                            action_output = policy.forward_inference(input_td)
                                        else:
                                            action_output = policy(input_td)
                                        
                                        # Extract action
                                        if isinstance(action_output, TensorDict):
                                            if (group, "action") in action_output.keys():
                                                action = action_output[(group, "action")]
                                                td[group]["action"] = action
                        
                        # Step environment
                        td = env.step(td)
                        done = td.get("done", torch.zeros(1, dtype=torch.bool)).any().item()
                        step_count += 1
                    
                    # Collect final metrics from info
                    if "next" in td.keys():
                        next_td = td["next"]
                        for group in algorithm.group_map.keys():
                            if group in next_td.keys() and "info" in next_td[group].keys():
                                info = next_td[group]["info"]
                                if info.shape[0] > 0:
                                    final_info = info[-1]
                                    
                                    # Extract metrics
                                    def safe_get(key, default=0.0):
                                        try:
                                            if hasattr(final_info, 'get'):
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
                                    
                                    episode_data[group] = {
                                        "precision": safe_get("precision", 0.0),
                                        "coverage": safe_get("coverage", 0.0),
                                        "total_reward": safe_get("total_reward", 0.0),
                                    }
                                    
                                    # Get final observation (anchor bounds)
                                    if "observation" in next_td[group].keys():
                                        final_obs = next_td[group]["observation"][-1]
                                        if isinstance(final_obs, torch.Tensor):
                                            episode_data[group]["final_observation"] = final_obs.cpu().numpy().tolist()
                    
                    if episode_data:
                        manual_rollouts.append(episode_data)
                        if hasattr(self.callback, 'evaluation_anchor_data'):
                            self.callback.evaluation_anchor_data.append(episode_data)
                
                print(f"  ✓ Collected {len(manual_rollouts)} episodes from manual rollouts")
                evaluation_anchor_data.extend(manual_rollouts)
                
            except Exception as e:
                print(f"  ⚠ Warning: Could not run manual rollouts: {e}")
                print(f"    Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
        
        # Final summary
        if not evaluation_anchor_data:
            print("\n  ⚠ Warning: No anchor data collected from any evaluation.")
            print("  This may happen if:")
            print("    1. Training evaluations didn't run (check evaluation_interval in config)")
            print("    2. Manual rollouts failed (see error above)")
            print("  Consider using inference.py to extract rules directly from the trained model.")
        else:
            print(f"\n  ✓ Total episodes collected: {len(evaluation_anchor_data)}")
        
        return {
            "experiment_folder": str(self.experiment.folder_name),
            "total_frames": self.experiment.total_frames,
            "n_iters_performed": self.experiment.n_iters_performed,
            "evaluation_anchor_data": evaluation_anchor_data,
            "evaluation_history": self.callback.get_evaluation_history() if hasattr(self.callback, 'get_evaluation_history') else []
        }
    
    def get_checkpoint_path(self) -> str:
        """
        Get the path to BenchMARL's experiment folder where checkpoints are saved.
        
        BenchMARL automatically saves checkpoints in the experiment folder.
        This method returns the path to that folder for later use.
        
        Returns:
            Path to BenchMARL experiment folder containing checkpoints
        """
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        
        experiment_folder = str(self.experiment.folder_name)
        print(f"BenchMARL experiment folder: {experiment_folder}")
        print(f"  This folder contains all checkpoints, logs, and model states")
        print(f"  Use this path to load checkpoints later with BenchMARL's load_state_dict()")
        
        return experiment_folder
    
    def extract_and_save_individual_models(
        self,
        output_dir: Optional[str] = None,
        save_policies: bool = True,
        save_critics: bool = False
    ) -> Dict[str, str]:
        """
        Extract and save individual policy/critic models for easier standalone inference.
        
        Based on MADDPG source code structure:
        - get_policy_for_loss(group) returns ProbabilisticActor wrapping actor_module
        - get_value_module(group) returns critic module
        - We extract the underlying neural network modules for standalone use
        
        Reference: https://benchmarl.readthedocs.io/en/latest/_modules/benchmarl/algorithms/maddpg.html
        
        Args:
            output_dir: Directory to save models (default: experiment folder / individual_models/)
            save_policies: Whether to save policy models
            save_critics: Whether to save critic models
        
        Returns:
            Dictionary mapping group names to saved model file paths
        """
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        
        algorithm = self.experiment.algorithm
        
        if output_dir is None:
            output_dir = os.path.join(str(self.experiment.folder_name), "individual_models")
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_models = {}
        
        print("\n" + "="*80)
        print("EXTRACTING INDIVIDUAL MODELS FOR STANDALONE INFERENCE")
        print("="*80)
        
        for group in algorithm.group_map.keys():
            print(f"\nExtracting models for group: {group}")
            
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
                        print(f"  ⚠ Warning: Could not extract actor module from policy for group {group}")
                        print(f"    Policy type: {type(policy)}")
                        print(f"    Policy attributes: {dir(policy)}")
                    else:
                        # Save the actor module
                        policy_path = os.path.join(output_dir, f"policy_{group}.pth")
                        torch.save(actor_module.state_dict(), policy_path)
                        saved_models[f"policy_{group}"] = policy_path
                        print(f"  ✓ Saved policy model to: {policy_path}")
                        
                        # Also save metadata about the model structure
                        metadata = {
                            "group": group,
                            "model_type": "policy",
                            "algorithm": self.algorithm,
                            "input_spec": str(getattr(actor_module, "input_spec", None)),
                            "output_spec": str(getattr(actor_module, "output_spec", None)),
                        }
                        metadata_path = os.path.join(output_dir, f"policy_{group}_metadata.json")
                        import json
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        print(f"  ✓ Saved policy metadata to: {metadata_path}")
                        
                except Exception as e:
                    print(f"  ✗ Error extracting policy for group {group}: {e}")
            
            if save_critics:
                try:
                    # Get critic using BenchMARL's official API (if available)
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
                            print(f"  ⚠ Warning: Could not extract critic module for group {group}")
                        else:
                            # Save the critic module
                            critic_path = os.path.join(output_dir, f"critic_{group}.pth")
                            torch.save(critic_module.state_dict(), critic_path)
                            saved_models[f"critic_{group}"] = critic_path
                            print(f"  ✓ Saved critic model to: {critic_path}")
                    else:
                        print(f"  ⚠ Algorithm {self.algorithm} does not have get_value_module() method")
                        
                except Exception as e:
                    print(f"  ✗ Error extracting critic for group {group}: {e}")
        
        print("\n" + "="*80)
        print(f"Individual models saved to: {output_dir}")
        print("="*80)
        print("\nTo use these models for inference:")
        print("  1. Load the model state_dict")
        print("  2. Reconstruct the model architecture")
        print("  3. Load state_dict into the model")
        print("  4. Use model.eval() and model(observation) for inference")
        
        return saved_models
    
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a BenchMARL checkpoint from the experiment folder.
        
        Args:
            checkpoint_path: Path to BenchMARL experiment folder or specific checkpoint file
        """
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        
        # If it's a directory, look for checkpoint files
        if os.path.isdir(checkpoint_path):
            # BenchMARL typically saves checkpoints in the experiment folder
            # Look for checkpoint files, but exclude classifier.pth
            all_files = os.listdir(checkpoint_path)
            checkpoint_files = [
                f for f in all_files 
                if (f.endswith('.pt') or f.endswith('.pth')) 
                and f != 'classifier.pth'  # Exclude classifier
                and not f.startswith('classifier')  # Exclude any classifier files
            ]
            
            # Also check subdirectories for checkpoints
            for root, dirs, files in os.walk(checkpoint_path):
                # Skip wandb and other log directories
                if 'wandb' in root or 'logs' in root.lower():
                    continue
                for f in files:
                    if (f.endswith('.pt') or f.endswith('.pth')) and f != 'classifier.pth':
                        rel_path = os.path.relpath(os.path.join(root, f), checkpoint_path)
                        if rel_path not in checkpoint_files:
                            checkpoint_files.append(rel_path)
            
            if checkpoint_files:
                # Use the most recent checkpoint
                checkpoint_file = max(
                    checkpoint_files, 
                    key=lambda f: os.path.getmtime(os.path.join(checkpoint_path, f))
                )
                checkpoint_path = os.path.join(checkpoint_path, checkpoint_file)
                print(f"Found checkpoint file: {checkpoint_file}")
            else:
                # No checkpoint files found - BenchMARL might save state differently
                # BenchMARL saves checkpoints automatically, but they might be in a different location
                # or checkpointing might be disabled. Try using restore_file mechanism.
                print(f"Warning: No checkpoint files found in {checkpoint_path}")
                print("  BenchMARL checkpoints may be disabled (checkpoint_interval=0 in config)")
                print("  Attempting to use BenchMARL's restore_file mechanism...")
                
                # Check if there's a checkpoint in a standard BenchMARL location
                possible_checkpoint_locations = [
                    os.path.join(checkpoint_path, "checkpoint.pt"),
                    os.path.join(checkpoint_path, "checkpoint.pth"),
                    os.path.join(checkpoint_path, "model.pt"),
                    os.path.join(checkpoint_path, "model.pth"),
                ]
                
                found_checkpoint = None
                for loc in possible_checkpoint_locations:
                    if os.path.exists(loc):
                        found_checkpoint = loc
                        break
                
                if found_checkpoint:
                    checkpoint_path = found_checkpoint
                    print(f"  Found checkpoint at: {checkpoint_path}")
                else:
                    # No checkpoint found - this means checkpointing was disabled
                    # We can't load a checkpoint that doesn't exist
                    raise ValueError(
                        f"No BenchMARL checkpoint found in {checkpoint_path}. "
                        f"Checkpoints may be disabled (checkpoint_interval=0). "
                        f"To enable checkpointing, set checkpoint_interval > 0 in base_experiment.yaml, "
                        f"or set checkpoint_at_end: True to save at the end of training."
                    )
        
        # Only try to load if we found a checkpoint file
        if os.path.isfile(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                self.experiment.load_state_dict(state_dict)
                print(f"Checkpoint loaded from: {checkpoint_path}")
                print(f"Resuming from iteration: {self.experiment.n_iters_performed}")
                print(f"Total frames: {self.experiment.total_frames}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint from {checkpoint_path}: {e}")
                print("  The checkpoint file may be corrupted or incompatible.")
                print("  Continuing with current experiment state...")
                raise
        else:
            raise ValueError(
                f"Checkpoint file not found: {checkpoint_path}. "
                f"Make sure checkpoint_at_end: True is set in the config to save checkpoints."
            )
    
    def get_experiment(self) -> Experiment:
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        return self.experiment
    
    def extract_rules_from_evaluation(
        self,
        evaluation_data: Dict[str, Any],
        max_features_in_rule: Optional[int] = 5,
        eval_on_test_data: bool = False
    ) -> Dict[str, Any]:
        """
        Extract anchor rules from BenchMARL evaluation data.
        
        Args:
            evaluation_data: Dictionary returned from evaluate() containing evaluation_anchor_data
            max_features_in_rule: Maximum number of features to include in extracted rules
            eval_on_test_data: Whether evaluation was done on test data
        
        Returns:
            Dictionary containing extracted rules and anchor data
        """
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        
        print("\n" + "="*80)
        print("EXTRACTING ANCHOR RULES FROM EVALUATION DATA")
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
                "eval_on_test_data": eval_on_test_data,
            },
            "evaluation_data": evaluation_data
        }
        
        # Get anchor data from evaluation
        evaluation_anchor_data = evaluation_data.get("evaluation_anchor_data", [])
        
        if not evaluation_anchor_data:
            print("Warning: No anchor data found in evaluation results.")
            print("  This may happen if BenchMARL's evaluation doesn't generate rollouts.")
            print("  Falling back to extracting rules from trained model directly.")
            print("  Using deprecated extract_rules() method as fallback...")
            
            # Fallback: use the old method to extract rules directly
            try:
                return self.extract_rules(
                    max_features_in_rule=max_features_in_rule,
                    steps_per_episode=100,  # Default steps
                    n_instances_per_class=20,  # Default instances
                    eval_on_test_data=eval_on_test_data
                )
            except Exception as e:
                print(f"  Error in fallback rule extraction: {e}")
                print("  Returning empty results structure.")
                return results
        
        # Group evaluation episodes by agent/class
        for target_class in target_classes:
            agent = f"agent_{target_class}"
            class_key = f"class_{target_class}"
            
            print(f"\nExtracting rules for class {target_class} (agent: {agent})...")
            
            # Collect all episodes for this agent
            agent_episodes = []
            for episode_data in evaluation_anchor_data:
                if agent in episode_data:
                    agent_episodes.append(episode_data[agent])
            
            if not agent_episodes:
                print(f"  No evaluation data found for {agent}")
                continue
            
            print(f"  Found {len(agent_episodes)} evaluation episodes for {agent}")
            
            # Extract rules from evaluation data
            # Note: We need to reconstruct anchor bounds from observations if available
            # For now, we'll use the metrics from evaluation
            rules_list = []
            anchors_list = []
            precisions = []
            coverages = []
            
            # Get feature names for rule extraction
            feature_names = env_data["feature_names"]
            n_features = len(feature_names)
            
            # Create a temporary environment to use its extract_rule method
            # We'll extract rules from the observation data
            temp_env_config = self._get_default_env_config()
            temp_env_config.update({
                "X_min": env_data["X_min"],
                "X_range": env_data["X_range"],
            })
            
            temp_anchor_env = AnchorEnv(
                X_unit=env_data["X_unit"],
                X_std=env_data["X_std"],
                y=env_data["y"],
                feature_names=feature_names,
                classifier=self.dataset_loader.get_classifier(),
                device="cpu",
                target_classes=[target_class],
                env_config=temp_env_config
            )
            
            for episode_idx, episode in enumerate(agent_episodes):
                precision = episode.get("precision", 0.0)
                coverage = episode.get("coverage", 0.0)
                
                precisions.append(float(precision))
                coverages.append(float(coverage))
                
                # Extract anchor bounds from observation
                # Observation structure: [lower_bounds (n_features), upper_bounds (n_features), precision, coverage]
                lower = None
                upper = None
                initial_lower = None
                initial_upper = None
                rule = "any values (no tightened features)"
                
                if "final_observation" in episode:
                    obs = np.array(episode["final_observation"], dtype=np.float32)
                    if len(obs) == 2 * n_features + 2:
                        # Extract lower and upper bounds from observation
                        lower = obs[:n_features].copy()
                        upper = obs[n_features:2*n_features].copy()
                        
                        # Set anchor bounds in temp environment for rule extraction
                        temp_anchor_env.lower[agent] = lower
                        temp_anchor_env.upper[agent] = upper
                        
                        # Extract rule using the environment's method
                        rule = temp_anchor_env.extract_rule(
                            agent,
                            max_features_in_rule=max_features_in_rule,
                            initial_lower=initial_lower,
                            initial_upper=initial_upper
                        )
                
                anchor_data = {
                    "episode_idx": episode_idx,
                    "precision": float(precision),
                    "coverage": float(coverage),
                    "total_reward": float(episode.get("total_reward", 0.0)),
                    "rule": rule,
                }
                
                if lower is not None and upper is not None:
                    anchor_data.update({
                        "lower_bounds": lower.tolist(),
                        "upper_bounds": upper.tolist(),
                        "box_widths": (upper - lower).tolist(),
                        "box_volume": float(np.prod(np.maximum(upper - lower, 1e-9))),
                    })
                    if initial_lower is not None and initial_upper is not None:
                        anchor_data.update({
                            "initial_lower_bounds": initial_lower.tolist(),
                            "initial_upper_bounds": initial_upper.tolist(),
                        })
                
                anchors_list.append(anchor_data)
                rules_list.append(rule)
            
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
                "n_episodes": len(anchors_list),
                "rules": rules_list,
                "unique_rules": unique_rules,
                "unique_rules_count": len(unique_rules),
                "anchors": anchors_list,
                "overlap_info": overlap_info,
            }
            
            print(f"  Processed {len(anchors_list)} episodes")
            print(f"  Average precision: {results['per_class_results'][class_key]['precision']:.4f}")
            print(f"  Average coverage: {results['per_class_results'][class_key]['coverage']:.4f}")
            print(f"  Unique rules: {len(unique_rules)}")
            if overlap_info["has_overlaps"]:
                print(f"  ⚠️  Overlaps detected: {overlap_info['n_overlaps']} anchors overlap with other classes")
            else:
                print(f"  ✓ No overlaps with other classes")
        
        print("="*80)
        
        results["training_history"] = self.callback.get_training_history() if hasattr(self, 'callback') else []
        results["evaluation_history"] = evaluation_data.get("evaluation_history", [])
        
        return results
    
    def extract_rules(
        self,
        max_features_in_rule: Optional[int] = 5,
        steps_per_episode: int = 100,
        n_instances_per_class: int = 20,
        eval_on_test_data: bool = False
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Use extract_rules_from_evaluation() instead.
        This method is kept for backward compatibility but may not work with all algorithms.
        """
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
                        # Use BenchMARL's official API to get policy for this agent/group
                        # Reference: https://benchmarl.readthedocs.io/en/latest/generated/benchmarl.algorithms.Algorithm.html#benchmarl.algorithms.Algorithm
                        algorithm = self.experiment.algorithm
                        
                        # Find the group for this agent
                        agent_group = None
                        if hasattr(algorithm, "group_map") and isinstance(algorithm.group_map, dict):
                            for group, agents_list in algorithm.group_map.items():
                                if agent in agents_list:
                                    agent_group = group
                                    break
                        
                        # If no group found, try using agent name directly as group
                        if agent_group is None:
                            # Check if agent name itself is a valid group
                            if hasattr(algorithm, "group_map") and isinstance(algorithm.group_map, dict):
                                if agent in algorithm.group_map:
                                    agent_group = agent
                                else:
                                    # Try to infer group from agent name (e.g., "agent_0" -> might be in group "agent")
                                    possible_groups = list(algorithm.group_map.keys())
                                    if len(possible_groups) == 1:
                                        agent_group = possible_groups[0]
                                    else:
                                        # Try to match agent to group by checking if agent is in any group's agent list
                                        for group in possible_groups:
                                            if agent in algorithm.group_map[group]:
                                                agent_group = group
                                                break
                        
                        if agent_group is None:
                            raise ValueError(
                                f"Could not find group for agent {agent}. "
                                f"Available groups: {list(algorithm.group_map.keys()) if hasattr(algorithm, 'group_map') else 'N/A'}"
                            )
                        
                        # Use BenchMARL's official API: get_policy_for_loss(group)
                        # This returns a TensorDictModule representing the policy
                        try:
                            policy = algorithm.get_policy_for_loss(agent_group)
                        except Exception as e:
                            raise ValueError(
                                f"Could not get policy for group {agent_group} (agent {agent}) using algorithm.get_policy_for_loss(). "
                                f"Error: {e}"
                            )
                        
                        # Get action from policy using forward_inference (BenchMARL standard)
                        # The policy is a TensorDictModule, so we can call it directly
                        action = None
                        
                        # Create input TensorDict with observation
                        # BenchMARL policies expect nested structure: {group: {"observation": obs}}
                        from tensordict import TensorDict
                        
                        # Method 1: Use policy module's input spec to determine correct structure
                        # Based on MADDPG source: actor_input_spec = Composite({group: observation_spec[group]})
                        # Reference: https://benchmarl.readthedocs.io/en/latest/_modules/benchmarl/algorithms/maddpg.html
                        input_td = None
                        last_error = None
                        
                        try:
                            # Try to get input spec from the policy module itself
                            policy_module = policy
                            if hasattr(policy, "module"):
                                policy_module = policy.module
                            
                            # Check if policy has input_spec or in_keys that tell us the structure
                            if hasattr(policy_module, "input_spec") and policy_module.input_spec is not None:
                                input_spec = policy_module.input_spec
                                # Input spec should be Composite({group: {"observation": ...}})
                                if agent_group in input_spec.keys():
                                    group_spec = input_spec[agent_group]
                                    if "observation" in group_spec.keys():
                                        # Structure confirmed: {group: {"observation": obs}}
                                        input_td = TensorDict(
                                            {agent_group: {"observation": obs_tensor}},
                                            batch_size=obs_tensor.shape[:1]
                                        )
                        except Exception as e:
                            # If spec method fails, continue to fallback methods
                            pass
                        
                        # Method 1b: Use experiment's observation spec as fallback
                        if input_td is None:
                            try:
                                # Get observation spec from the experiment's task
                                if hasattr(self.experiment, 'task') and hasattr(self.experiment.task, 'observation_spec'):
                                    obs_spec = self.experiment.task.observation_spec(self.experiment.env)
                                    if obs_spec is not None and agent_group in obs_spec.keys():
                                        # Use the spec to create the correct structure
                                        group_obs_spec = obs_spec[agent_group]
                                        if "observation" in group_obs_spec.keys():
                                            # Structure: {group: {"observation": obs}}
                                            input_td = TensorDict(
                                                {agent_group: {"observation": obs_tensor}},
                                                batch_size=obs_tensor.shape[:1]
                                            )
                            except Exception as e:
                                # If spec method fails, continue to fallback methods
                                pass
                        
                        # Method 2: Use policy's in_key if available
                        if input_td is None:
                            policy_in_key = None
                            if hasattr(policy, "in_key"):
                                policy_in_key = policy.in_key
                            elif hasattr(policy, "module") and hasattr(policy.module, "in_key"):
                                policy_in_key = policy.module.in_key
                            
                            if policy_in_key is not None:
                                if isinstance(policy_in_key, tuple) and len(policy_in_key) > 1:
                                    # Nested key like (group, "observation")
                                    input_td = TensorDict({policy_in_key[0]: {policy_in_key[1]: obs_tensor}}, batch_size=obs_tensor.shape[:1])
                                elif isinstance(policy_in_key, tuple) and len(policy_in_key) == 1:
                                    # Single-level key
                                    input_td = TensorDict({policy_in_key[0]: obs_tensor}, batch_size=obs_tensor.shape[:1])
                                elif isinstance(policy_in_key, str):
                                    # String key - might need nesting
                                    input_td = TensorDict({agent_group: {policy_in_key: obs_tensor}}, batch_size=obs_tensor.shape[:1])
                        
                        # Method 3: Try common structures (fallback)
                        if input_td is None:
                            structures_to_try = [
                                ({agent_group: {"observation": obs_tensor}}, "nested {group: {observation: ...}}"),
                                ({"observation": obs_tensor}, "flat {observation: ...}"),
                                ({agent_group: obs_tensor}, "flat {group: ...}"),
                            ]
                            
                            for struct_dict, struct_name in structures_to_try:
                                try:
                                    test_td = TensorDict(struct_dict, batch_size=obs_tensor.shape[:1])
                                    # Try forward_inference to see if structure is correct
                                    if hasattr(policy, "forward_inference"):
                                        _ = policy.forward_inference(test_td.clone())
                                    else:
                                        _ = policy(test_td.clone())
                                    # If we get here, the structure works!
                                    input_td = test_td
                                    break
                                except Exception as e:
                                    last_error = e
                                    continue
                            
                            if input_td is None:
                                raise ValueError(
                                    f"Could not determine correct input TensorDict structure for policy. "
                                    f"Agent_group: {agent_group}, agent: {agent}. "
                                    f"Tried {len(structures_to_try)} different structures, last error: {last_error}"
                                )
                        
                        # Use forward_inference for deterministic evaluation
                        if hasattr(policy, "forward_inference"):
                            fwd_outputs = policy.forward_inference(input_td)
                        else:
                            # Fallback: use regular forward
                            fwd_outputs = policy(input_td)
                        
                        # Extract action from outputs
                        if isinstance(fwd_outputs, TensorDict):
                            if "action_dist_inputs" in fwd_outputs.keys():
                                # Create action distribution and get deterministic action
                                action_dist_inputs = fwd_outputs["action_dist_inputs"]
                                
                                # Try to get action distribution class
                                if hasattr(policy, "get_inference_action_dist_cls"):
                                    action_dist_class = policy.get_inference_action_dist_cls()
                                    action_dist = action_dist_class.from_logits(action_dist_inputs)
                                else:
                                    # Fallback: try TanhNormal for continuous actions
                                    from torchrl.modules import TanhNormal
                                    action_dist = TanhNormal.from_logits(action_dist_inputs)
                                
                                # Get deterministic action (mean for evaluation)
                                if hasattr(action_dist, "mean"):
                                    action = action_dist.mean()
                                elif hasattr(action_dist, "deterministic_sample"):
                                    action = action_dist.deterministic_sample()
                                elif hasattr(action_dist, "mode"):
                                    action = action_dist.mode()
                                else:
                                    action = action_dist.sample()
                            elif "action" in fwd_outputs.keys():
                                action = fwd_outputs["action"]
                            else:
                                raise ValueError(f"Could not extract action from policy output. Keys: {fwd_outputs.keys()}")
                        elif isinstance(fwd_outputs, dict):
                            if "action_dist_inputs" in fwd_outputs:
                                action_dist_inputs = fwd_outputs["action_dist_inputs"]
                                from torchrl.modules import TanhNormal
                                action_dist = TanhNormal.from_logits(action_dist_inputs)
                                action = action_dist.mean() if hasattr(action_dist, "mean") else action_dist.sample()
                            elif "action" in fwd_outputs:
                                action = fwd_outputs["action"]
                            else:
                                raise ValueError(f"Could not extract action from policy output. Keys: {list(fwd_outputs.keys())}")
                        else:
                            # Output might be the action directly
                            action = fwd_outputs
                        
                        if action is None:
                            raise ValueError("Could not extract action from policy")
                        
                        # If it's a distribution, get deterministic action
                        if hasattr(action, "mean"):
                            action = action.mean()
                        elif hasattr(action, "sample") and not isinstance(action, torch.Tensor):
                            action = action.sample()
                        
                        # Convert to numpy array
                        if isinstance(action, torch.Tensor):
                            action = action.cpu().numpy()
                        action = np.array(action, dtype=np.float32)
                        if len(action.shape) > 1:
                            action = action.flatten()
                        # Ensure correct shape (should be 1D array)
                        if action.shape[0] != 2 * anchor_env.n_features:
                            # If action is wrong size, pad or truncate
                            expected_size = 2 * anchor_env.n_features
                            if action.shape[0] < expected_size:
                                action = np.pad(action, (0, expected_size - action.shape[0]), mode='constant')
                            else:
                                action = action[:expected_size]
                    
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
            if self.output_dir is None:
                raise ValueError("output_dir must be set to save rules")
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
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (float, np.float64, np.float32)):  # Handle NumPy 2.0 compatibility
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
            "shared_reward_weight": 0.2,  # Weight for shared cooperative reward
        }

