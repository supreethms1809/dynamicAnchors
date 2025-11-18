import torch
import os
import sys
import torch.nn as nn
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import PettingZooTask
from benchmarl.experiment import ExperimentConfig, Experiment
from benchmarl.models.mlp import MlpConfig

experiment_config = ExperimentConfig.get_from_yaml("conf/base_experiment.yaml")
task = PettingZooTask(1)
algorithm_config = MaddpgConfig.get_from_yaml("conf/maddpg.yaml")
model_config = MlpConfig.get_from_yaml("conf/mlp.yaml")
critic_model_config = MlpConfig.get_from_yaml("conf/mlp.yaml")

experiment_config.max_n_frames = 12_000
experiment_config.loggers = []

experiment = Experiment(config=experiment_config, 
                        task=task, 
                        algorithm_config=algorithm_config, 
                        model_config=model_config, 
                        critic_model_config=critic_model_config,
                        seed=42)
print(experiment)


