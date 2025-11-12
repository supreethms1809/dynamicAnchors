import copy
from typing import Callable, Dict, List, Optional, Any
import numpy as np
import torch

from torchrl.data import Composite
from torchrl.envs import EnvBase, PettingZooEnv
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from tensordict import TensorDictBase

from benchmarl.environments.common import Task, TaskClass
from benchmarl.experiment.callback import Callback
from benchmarl.utils import DEVICE_TYPING

import sys
import os
# Add BenchMARL directory to path to import AnchorEnv
benchmarl_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, benchmarl_dir)
from environment import AnchorEnv

class AnchorTaskClass(TaskClass):
    
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        
        env_config = {k: v for k, v in config.items() if k != "max_cycles"}
        
        def _make_env():
            anchor_env = AnchorEnv(**env_config)
            
            return PettingZooWrapper(
                env=anchor_env,
                return_state=False,
                categorical_actions=False,
                seed=seed,
                device=device,
            )
        
        return _make_env
    
    def supports_continuous_actions(self) -> bool:
        return True
    
    def supports_discrete_actions(self) -> bool:
        return False
    
    def has_state(self) -> bool:
        return False
    
    def has_render(self, env: EnvBase) -> bool:
        return False
    
    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("max_cycles", 100)
    
    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map
    
    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None
    
    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None
    
    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        return observation_spec
    
    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        if observation_spec.is_empty():
            return None
        return observation_spec
    
    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec
    
    @staticmethod
    def env_name() -> str:
        return "anchor"


class AnchorTask(Task):
    
    ANCHOR = None
    
    @staticmethod
    def associated_class():
        return AnchorTaskClass


class AnchorMetricsCallback(Callback):
    
    def __init__(self, log_training_metrics: bool = True, log_evaluation_metrics: bool = True, save_to_file: bool = True):
        super().__init__()
        self.log_training_metrics = log_training_metrics
        self.log_evaluation_metrics = log_evaluation_metrics
        self.save_to_file = save_to_file
        self.training_metrics = []
        self.training_history = []
        self.evaluation_history = []
    
    def _extract_metrics_from_info(self, info: TensorDictBase, prefix: str = "") -> Dict[str, float]:
        metrics = {}
        
        if info is None:
            return metrics
        
        metric_keys = [
            "precision", "coverage", "drift", "anchor_drift", "js_penalty",
            "precision_gain", "coverage_gain", "coverage_bonus", "target_class_bonus",
            "overlap_penalty", "drift_penalty", "anchor_drift_penalty", "total_reward",
            "coverage_floor_hits", "coverage_clipped"
        ]
        
        for key in metric_keys:
            if key in info.keys():
                value = info[key]
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        metrics[f"{prefix}{key}"] = value.item()
                    elif value.numel() > 0:
                        metrics[f"{prefix}{key}_mean"] = value.float().mean().item()
                        metrics[f"{prefix}{key}_std"] = value.float().std().item()
                elif isinstance(value, (int, float, np.number)):
                    metrics[f"{prefix}{key}"] = float(value)
        
        return metrics
    
    def on_batch_collected(self, batch: TensorDictBase):
        if not self.log_training_metrics:
            return
        
        all_metrics = {}
        
        for group in self.experiment.group_map.keys():
            group_key = ("next", group, "info")
            if group_key in batch.keys(include_nested=True):
                info = batch[group_key]
                
                if info.shape[0] > 0:
                    final_info = info[-1]
                    group_metrics = self._extract_metrics_from_info(
                        final_info, 
                        prefix=f"training/{group}/"
                    )
                    all_metrics.update(group_metrics)
        
        if all_metrics:
            self.training_metrics.append(all_metrics)
            
            if len(self.training_metrics) >= 10:
                aggregated = {}
                for key in self.training_metrics[0].keys():
                    values = [m[key] for m in self.training_metrics if key in m]
                    if values:
                        aggregated[key] = sum(values) / len(values)
                
                self.experiment.logger.log(
                    aggregated,
                    step=self.experiment.n_iters_performed
                )
                
                if self.save_to_file:
                    aggregated["step"] = self.experiment.n_iters_performed
                    aggregated["total_frames"] = self.experiment.total_frames
                    self.training_history.append(aggregated.copy())
                
                self.training_metrics = []
    
    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        if not self.log_evaluation_metrics:
            return
        
        all_metrics = {}
        n_episodes = 0
        
        for rollout in rollouts:
            n_episodes += 1
            
            for group in self.experiment.group_map.keys():
                if group in rollout.keys():
                    group_rollout = rollout[group]
                    
                    if "info" in group_rollout.keys():
                        info = group_rollout["info"]
                        
                        if info.shape[0] > 0:
                            final_info = info[-1]
                            group_metrics = self._extract_metrics_from_info(
                                final_info,
                                prefix=f"evaluation/{group}/"
                            )
                            
                            for key, value in group_metrics.items():
                                if key not in all_metrics:
                                    all_metrics[key] = []
                                all_metrics[key].append(value)
        
        if all_metrics:
            aggregated = {}
            for key, values in all_metrics.items():
                if values:
                    aggregated[f"{key}_mean"] = sum(values) / len(values)
                    aggregated[f"{key}_std"] = np.std(values) if len(values) > 1 else 0.0
                    aggregated[f"{key}_min"] = min(values)
                    aggregated[f"{key}_max"] = max(values)
            
            aggregated["evaluation/n_episodes"] = n_episodes
            
            self.experiment.logger.log(
                aggregated,
                step=self.experiment.n_iters_performed
            )
            
            if self.save_to_file:
                aggregated["step"] = self.experiment.n_iters_performed
                aggregated["total_frames"] = self.experiment.total_frames
                self.evaluation_history.append(aggregated.copy())
            
            precision_key = [k for k in aggregated.keys() if "precision" in k and "mean" in k and "evaluation" in k]
            coverage_key = [k for k in aggregated.keys() if "coverage" in k and "mean" in k and "evaluation" in k]
            if precision_key and coverage_key:
                print(
                    f"Evaluation - Precision: {aggregated[precision_key[0]]:.4f}, "
                    f"Coverage: {aggregated[coverage_key[0]]:.4f} "
                    f"(n={n_episodes})"
                )
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        return self.training_history.copy()
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        return self.evaluation_history.copy()

