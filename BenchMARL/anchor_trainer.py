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
        self._anchor_env_config = None  # Cache for loaded env_config from YAML
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setup_experiment(
        self,
        env_config: Optional[Dict[str, Any]] = None,
        target_classes: Optional[List[int]] = None,
        max_cycles: int = 1000,
        device: str = "cpu",
        eval_on_test_data: bool = True
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
        
        # Get the anchor environment data.
        env_data = self.dataset_loader.get_anchor_env_data()
        
        # Get the environment configuration from YAML file or use defaults.
        if env_config is None:
            env_config = self._load_env_config_from_yaml()
        
        # Get the target classes.
        if target_classes is None:
            target_classes = list(np.unique(self.dataset_loader.y_train))
        
        # Create the environment configuration with the data.
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
            logger.info(f"  Evaluation configured to use TEST data")
        else:
            env_config_with_data["eval_on_test_data"] = False
            logger.info(f"  Evaluation configured to use TRAINING data")
        
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
        self.callback = AnchorMetricsCallback(
            log_training_metrics=True, 
            log_evaluation_metrics=True,
            save_to_file=True,
            collect_anchor_data=True
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
        
        logger.info(f"Experiment setup complete:")
        logger.info(f"  Algorithm: {self.algorithm.upper()}")
        logger.info(f"  Model: MLP")
        logger.info(f"  Target classes: {target_classes}")
        logger.info(f"  Max cycles per episode: {max_cycles}")
        logger.info(f"  Experiment folder: {self.experiment.folder_name}")
        logger.info("="*80)
        
        return self.experiment
    
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
        
        # Save callback data to files
        if hasattr(self, 'callback') and self.callback is not None:
            if hasattr(self.callback, 'save_data_to_files'):
                logger.info("\nSaving callback data to files...")
                try:
                    saved_files = self.callback.save_data_to_files(str(self.experiment.folder_name))
                    if saved_files:
                        logger.info(f"  ✓ Saved {len(saved_files)} data files")
                    else:
                        logger.info("  No callback data to save")
                except Exception as e:
                    logger.warning(f"  ⚠ Warning: Could not save callback data: {e}")
        
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
        # BenchMARL's periodic evaluations during training DO pass rollouts to callbacks - SS: If we get info from callbacks, we don't need to run manual rollouts
        evaluation_anchor_data = []
        if hasattr(self.callback, 'get_evaluation_anchor_data'):
            evaluation_anchor_data = self.callback.get_evaluation_anchor_data()
            logger.info(f"  Found {len(evaluation_anchor_data)} episodes of anchor data from training evaluations")
        
        # BenchMARL's standard evaluate() for metrics logging (This is not working right now - SS)
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
                env = self._create_env_instance(device=str(policy_device))
                
                # Debug: Check group_map structure (before episode loop) - SS: REMOVE THIS DEBUG LATER
                logger.info(f"  Debug: algorithm.group_map = {algorithm.group_map}")
                logger.info(f"  Debug: group_map keys = {list(algorithm.group_map.keys())}")
                for group, agents in algorithm.group_map.items():
                    logger.info(f"    Group '{group}' contains agents: {agents}")
                
                # Get unwrapped environment to check actual agents - SS: REMOVE THIS DEBUG LATER
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
                for episode in range(n_eval_episodes):
                    # Reset environment
                    td = env.reset()
                    done = False
                    episode_data = {}
                    
                    # Run episode
                    step_count = 0
                    max_steps = self.task.max_steps(env) if hasattr(self.task, 'max_steps') else 100
                    
                    # Debug: Check initial td structure - SS: REMOVE THIS DEBUG LATER
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
                    
                    # Collect final metrics from info - SS: REMOVE THIS DEBUG LATER
                    unwrapped_env = None
                    if hasattr(env, 'env') or hasattr(env, '_env'):
                        unwrapped_env = getattr(env, 'env', None) or getattr(env, '_env', None)
                    
                    # Debug: Check unwrapped environment state - SS: REMOVE THIS DEBUG LATER
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
                    
                    # Try multiple ways to access the final state - SS: REMOVE THIS DEBUG LATER
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
                        # Debug: Check unwrapped_env state - SS: REMOVE THIS DEBUG LATER
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
                                        
                                        # Debug: Check bounds values
                                        if episode == 0:
                                            logger.info(f"  Debug: {agent_name} bounds - lower range: [{lower_bounds.min():.4f}, {lower_bounds.max():.4f}], upper range: [{upper_bounds.min():.4f}, {upper_bounds.max():.4f}]")
                                        
                                        final_obs = np.concatenate([lower_bounds, upper_bounds, np.array([precision, coverage], dtype=np.float32)])
                                        
                                        # Store data keyed by agent name (not group) to distinguish between agents
                                        episode_data[agent_name] = {
                                            "precision": float(precision),
                                            "coverage": float(coverage),
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
                                        traceback.print_exc()  # SS: REMOVE THIS DEBUG LATER
                        
                        # Fallback: Try to get from TensorDict structure - SS: ONLY THIS PART IS CURRENTLY WORKING
                        # Try both final_td (which might be next_td) and td directly - SS: ONLY THIS PART IS CURRENTLY WORKING
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
                            
                            # Extract data from observation if available
                            # Observation structure: [lower_bounds (n_features), upper_bounds (n_features), precision, coverage]
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
                                # obs_len = 2*n_features + 2 (precision + coverage)
                                obs_len = len(final_obs_np) if hasattr(final_obs_np, '__len__') else final_obs_np.shape[0] if hasattr(final_obs_np, 'shape') else 0
                                
                                if obs_len >= 4:  # At least 2 features + precision + coverage
                                    n_features = (obs_len - 2) // 2
                                    if n_features > 0:
                                        # Extract precision and coverage from last two elements
                                        precision = float(final_obs_np[-2])
                                        coverage = float(final_obs_np[-1])
                                        
                                        # Store data keyed by agent name to distinguish between agents
                                        episode_data[agent_name] = {
                                            "precision": precision,
                                            "coverage": coverage,
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
                                episode_data[agent_name] = {
                                    "precision": safe_get("precision", 0.0),
                                    "coverage": safe_get("coverage", 0.0),
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
                    
                    # Debug first episode - SS: REMOVE THIS DEBUG LATER
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
                    
                    if episode_data:
                        manual_rollouts.append(episode_data)
                        if hasattr(self.callback, 'evaluation_anchor_data'):
                            self.callback.evaluation_anchor_data.append(episode_data)
                
                logger.info(f"  ✓ Collected {len(manual_rollouts)} episodes from manual rollouts")
                evaluation_anchor_data.extend(manual_rollouts)
                
            except Exception as e:
                logger.warning(f"  ⚠ Warning: Could not run manual rollouts: {e}")
                logger.warning(f"    Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
        
        # Final summary
        if not evaluation_anchor_data:
            logger.warning("\n  ⚠ Warning: No anchor data collected from any evaluation.")
            logger.warning("  This may happen if:")
            logger.warning("    1. Training evaluations didn't run (check evaluation_interval in config)")
            logger.warning("    2. Manual rollouts failed (see error above)")
            logger.warning("  Consider using inference.py to extract rules directly from the trained model.")
        else:
            logger.info(f"\n  ✓ Total episodes collected: {len(evaluation_anchor_data)}")
        
        # Save callback data to files (including any new evaluation data)
        if hasattr(self, 'callback') and self.callback is not None:
            if hasattr(self.callback, 'save_data_to_files'):
                logger.info("\nSaving callback data to files...")
                try:
                    saved_files = self.callback.save_data_to_files(str(self.experiment.folder_name))
                    if saved_files:
                        logger.info(f"  ✓ Saved {len(saved_files)} data files")
                except Exception as e:
                    logger.warning(f"  ⚠ Warning: Could not save callback data: {e}")
        
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
            output_dir: Directory to save models (ignored - always saves in experiment run log directory)
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
        
        # Always save in the experiment's run log directory (where BenchMARL saves logs/checkpoints)
        # SS: Models are saved alongside the run logs, not in a separate location
        output_dir = os.path.join(str(self.experiment.folder_name), "individual_models")
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_models = {}
        
        logger.info("\n" + "="*80)
        logger.info("EXTRACTING INDIVIDUAL MODELS FOR STANDALONE INFERENCE")
        logger.info("="*80)
        
        for group in algorithm.group_map.keys():
            logger.info(f"\nExtracting models for group: {group}")
            
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
                        logger.warning(f"  ⚠ Warning: Could not extract actor module from policy for group {group}")
                        logger.warning(f"    Policy type: {type(policy)}")
                        logger.warning(f"    Policy attributes: {dir(policy)}")
                    else:
                        # Save the actor module
                        policy_path = os.path.join(output_dir, f"policy_{group}.pth")
                        torch.save(actor_module.state_dict(), policy_path)
                        saved_models[f"policy_{group}"] = policy_path
                        logger.info(f"  ✓ Saved policy model to: {policy_path}")
                        
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
                        logger.info(f"  ✓ Saved policy metadata to: {metadata_path}")
                        
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
                            logger.warning(f"  ⚠ Warning: Could not extract critic module for group {group}")
                        else:
                            # Save the critic module
                            critic_path = os.path.join(output_dir, f"critic_{group}.pth")
                            torch.save(critic_module.state_dict(), critic_path)
                            saved_models[f"critic_{group}"] = critic_path
                            logger.info(f"  ✓ Saved critic model to: {critic_path}")
                    else:
                        logger.warning(f"  ⚠ Algorithm {self.algorithm} does not have get_value_module() method")
                        
                except Exception as e:
                    logger.warning(f"  ✗ Error extracting critic for group {group}: {e}")
        
        logger.info("\n" + "="*80)
        logger.info(f"Individual models saved to: {output_dir}")
        logger.info("="*80)
        logger.info("\nTo use these models for inference:")
        logger.info("  1. Load the model state_dict")
        logger.info("  2. Reconstruct the model architecture")
        logger.info("  3. Load state_dict into the model")
        logger.info("  4. Use model.eval() and model(observation) for inference")
        
        return saved_models
    
    
    def reload_experiment(self, checkpoint_file: str):
        """
        Reload the entire BenchMARL Experiment from a checkpoint file.
        Uses BenchMARL's official reload_experiment_from_file() method.
        
        Reference: https://benchmarl.readthedocs.io/en/latest/concepts/features.html#reloading
        
        Args:
            checkpoint_file: Path to BenchMARL checkpoint file (.pt or .pth file)
                            If a directory is provided, will search for checkpoint files
        """
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
        
        logger.info("  ✓ Experiment reloaded successfully")
        logger.info(f"  Resuming from iteration: {self.experiment.n_iters_performed}")
        logger.info(f"  Total frames: {self.experiment.total_frames}")
    
    def get_experiment(self) -> Experiment:
        if self.experiment is None:
            raise ValueError(
                "Experiment not set up yet. Call setup_experiment() first."
            )
        return self.experiment
    
    # SS: This is the environment instance that is used to collect anchor data. If we remove evaluate()
    # then we won't need this method.
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
    
    # SS: This another method which is currently very messy. It is used to extract rules from the evaluation data.
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
        
        logger.info("\n" + "="*80)
        logger.info("EXTRACTING ANCHOR RULES FROM EVALUATION DATA")
        logger.info("="*80)
        
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
            logger.warning("Warning: No anchor data found in evaluation results.")
            logger.warning("  This may happen if BenchMARL's evaluation doesn't generate rollouts.")
            logger.warning("  Falling back to extracting rules from trained model directly.")
            logger.warning("  Using deprecated extract_rules() method as fallback...")
            
            # Fallback: use the old method to extract rules directly
            try:
                return self.extract_rules(
                    max_features_in_rule=max_features_in_rule,
                    steps_per_episode=100,  # Default steps
                    n_instances_per_class=20,  # Default instances
                    eval_on_test_data=eval_on_test_data
                )
            except Exception as e:
                logger.warning(f"  Error in fallback rule extraction: {e}")
                logger.warning("  Returning empty results structure.")
                return results
        
        # Group evaluation episodes by agent/class
        for target_class in target_classes:
            agent = f"agent_{target_class}"
            class_key = f"class_{target_class}"
            
            logger.info(f"\nExtracting rules for class {target_class} (agent: {agent})...")
            
            # Collect all episodes for this agent
            agent_episodes = []
            for episode_data in evaluation_anchor_data:
                if agent in episode_data:
                    agent_episodes.append(episode_data[agent])
            
            if not agent_episodes:
                logger.warning(f"  No evaluation data found for {agent}")
                continue
            
            logger.info(f"  Found {len(agent_episodes)} evaluation episodes for {agent}")
            
            # Extract rules from evaluation data
            # Note: We need to reconstruct anchor bounds from observations if available
            # For now, we'll use the metrics from evaluation
            rules_list = []
            anchors_list = []
            # Instance-level metrics (per agent/instance)
            instance_precisions = []
            instance_coverages = []
            # Class-level metrics (union of all agents for the class)
            class_precisions = []
            class_coverages = []
            # Legacy lists (for backward compatibility)
            precisions = []
            coverages = []
            
            # Get feature names for rule extraction
            feature_names = env_data["feature_names"]
            n_features = len(feature_names)
            
            # Create a temporary environment to use its extract_rule method
            # We'll extract rules from the observation data
            temp_env_config = self._load_env_config_from_yaml()
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
                # Instance-level metrics
                instance_precision = episode.get("instance_precision", episode.get("precision", 0.0))
                instance_coverage = episode.get("instance_coverage", episode.get("coverage", 0.0))
                # Class-level metrics (may not be in episode data, will compute if needed)
                class_precision = episode.get("class_precision", 0.0)
                class_coverage = episode.get("class_coverage", 0.0)
                
                instance_precisions.append(float(instance_precision))
                instance_coverages.append(float(instance_coverage))
                class_precisions.append(float(class_precision))
                class_coverages.append(float(class_coverage))
                
                # Legacy fields (for backward compatibility)
                precisions.append(float(instance_precision))
                coverages.append(float(instance_coverage))
                
                precision = float(instance_precision)
                coverage = float(instance_coverage)
                
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
                    # Instance-level metrics
                    "instance_precision": float(instance_precision),
                    "instance_coverage": float(instance_coverage),
                    # Class-level metrics
                    "class_precision": float(class_precision),
                    "class_coverage": float(class_coverage),
                    # Legacy fields (for backward compatibility)
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
                # Instance-level metrics (averaged across all instances)
                "instance_precision": float(np.mean(instance_precisions)) if instance_precisions else 0.0,
                "instance_coverage": float(np.mean(instance_coverages)) if instance_coverages else 0.0,
                "instance_precision_std": float(np.std(instance_precisions)) if len(instance_precisions) > 1 else 0.0,
                "instance_coverage_std": float(np.std(instance_coverages)) if len(instance_coverages) > 1 else 0.0,
                "instance_precision_min": float(np.min(instance_precisions)) if instance_precisions else 0.0,
                "instance_precision_max": float(np.max(instance_precisions)) if instance_precisions else 0.0,
                "instance_coverage_min": float(np.min(instance_coverages)) if instance_coverages else 0.0,
                "instance_coverage_max": float(np.max(instance_coverages)) if instance_coverages else 0.0,
                # Class-level metrics (union of all agents for this class)
                "class_precision": float(np.mean(class_precisions)) if class_precisions else 0.0,
                "class_coverage": float(np.mean(class_coverages)) if class_coverages else 0.0,
                "class_precision_std": float(np.std(class_precisions)) if len(class_precisions) > 1 else 0.0,
                "class_coverage_std": float(np.std(class_coverages)) if len(class_coverages) > 1 else 0.0,
                "class_precision_min": float(np.min(class_precisions)) if class_precisions else 0.0,
                "class_precision_max": float(np.max(class_precisions)) if class_precisions else 0.0,
                "class_coverage_min": float(np.min(class_coverages)) if class_coverages else 0.0,
                "class_coverage_max": float(np.max(class_coverages)) if class_coverages else 0.0,
                # Legacy fields (for backward compatibility, using instance-level)
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
            
            logger.info(f"  Processed {len(anchors_list)} episodes")
            logger.info(f"  Instance-level - Average precision: {results['per_class_results'][class_key]['instance_precision']:.4f}, coverage: {results['per_class_results'][class_key]['instance_coverage']:.4f}")
            logger.info(f"  Class-level - Average precision: {results['per_class_results'][class_key]['class_precision']:.4f}, coverage: {results['per_class_results'][class_key]['class_coverage']:.4f}")
            logger.info(f"  Unique rules: {len(unique_rules)}")
            if overlap_info["has_overlaps"]:
                logger.warning(f"  ⚠️  Overlaps detected: {overlap_info['n_overlaps']} anchors overlap with other classes")
            else:
                logger.info(f"  ✓ No overlaps with other classes")
        
        logger.info("="*80)
        
        results["training_history"] = self.callback.get_training_history() if hasattr(self, 'callback') else []
        results["evaluation_history"] = evaluation_data.get("evaluation_history", [])
        
        return results
    
    # SS: This is a duplicate of the method above. Get rid of this if we don't need it.
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
        
        logger.info("\n" + "="*80)
        logger.info("EXTRACTING ANCHOR RULES")
        logger.info("="*80)
        
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
            
            logger.info(f"\nExtracting rules for class {target_class} (agent: {agent})...")
            
            env_config = self._load_env_config_from_yaml()
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
                logger.warning(f"  No instances found for class {target_class} in {data_source_name} data")
                continue
            
            n_samples = min(n_instances_per_class, len(class_instances))
            sampled_indices = np.random.choice(class_instances, size=n_samples, replace=False)
            
            logger.info(f"  Sampling {n_samples} instances from {data_source_name} data for class {target_class}")
            
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
                                    # Create environment instance (Experiment doesn't have direct env attribute)
                                    env = self._create_env_instance()
                                    obs_spec = self.experiment.task.observation_spec(env)
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
            
            logger.info(f"  Extracted {len(anchors_list)} anchors")
            logger.info(f"  Average precision: {results['per_class_results'][class_key]['precision']:.4f}")
            logger.info(f"  Average coverage: {results['per_class_results'][class_key]['coverage']:.4f}")
            logger.info(f"  Unique rules: {len(unique_rules)}")
            if overlap_info["has_overlaps"]:
                logger.warning(f"  ⚠️  Overlaps detected: {overlap_info['n_overlaps']} anchors overlap with other classes")
            else:
                logger.info(f"  ✓ No overlaps with other classes")
        
        logger.info("="*80)
        
        results["training_history"] = self.callback.get_training_history() if hasattr(self, 'callback') else []
        results["evaluation_history"] = self.callback.get_evaluation_history() if hasattr(self, 'callback') else []
        
        return results
    
    # SS: This method is needed for the metrics and I forgot if i am using it in the reward function.
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
        
        logger.info(f"Rules and anchors saved to: {filepath}")
        logger.info(f"  Total anchors saved: {n_anchors_total}")
        logger.info(f"  Total rules saved: {n_rules_total}")
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
    
    def _load_env_config_from_yaml(self) -> Dict[str, Any]:
        """
        Load environment configuration from anchor.yaml file.
        Falls back to hardcoded defaults if file is not found or doesn't contain env_config.
        
        Returns:
            Dictionary containing environment configuration parameters
        """
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
                    # Convert list to tuple for step_fracs if it's a list (YAML loads lists as lists)
                    if "step_fracs" in env_config and isinstance(env_config["step_fracs"], list):
                        env_config["step_fracs"] = tuple(env_config["step_fracs"])
                    # Convert boolean strings to actual booleans if needed
                    for key, value in env_config.items():
                        if isinstance(value, str):
                            if value.lower() == "true":
                                env_config[key] = True
                            elif value.lower() == "false":
                                env_config[key] = False
                    
                    logger.info(f"  Loaded environment config from: {config_path}")
                    self._anchor_env_config = env_config.copy()
                    return env_config
                else:
                    logger.warning(f"  Warning: {config_path} exists but doesn't contain 'env_config' key. Using defaults.")
            except Exception as e:
                logger.warning(f"  Warning: Could not load config from {config_path}: {e}. Using defaults.")
        else:
            logger.warning(f"  Warning: Anchor config file not found at {config_path}. Using defaults.")
        
        # Fallback to hardcoded defaults
        default_config = self._get_default_env_config()
        self._anchor_env_config = default_config.copy()
        return default_config
    
    def _get_default_env_config(self) -> Dict[str, Any]:
        """
        Get hardcoded default environment configuration.
        This is used as a fallback if YAML file is not found or doesn't contain env_config.
        """
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
            "use_class_centroids": True,  # Use class centroids for initial window (default: True)
        }

