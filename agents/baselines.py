"""Defines agents that can be used as baselines for experiments"""
from typing import Any, Dict, Tuple

import gym
import numpy as np
import torch
from stable_baselines3 import PPO as SBPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from tqdm import tqdm

from .agent import Agent


class PPO(Agent):
    """A standard PPO agent"""

    def __init__(self, name: str, env: gym.Env, seed: int = None) -> None:
        super().__init__(name, env, seed)

        self._model = SBPPO(
            policy="CnnPolicy",
            env=self._env,
            tensorboard_log="/tmp/tensorboard/",
            policy_kwargs=dict(features_extractor_class=_CustomCNN, features_extractor_kwargs=dict(features_dim=128)),
            seed=self._seed,
        )

    def learn(self, **kwargs: Dict) -> None:
        self._model.learn(**kwargs, tb_log_name=self._name, callback=_TqdmCallback())

    def predict(self, observation: np.ndarray, **kwargs: Dict) -> Tuple[np.ndarray, Any]:
        return self._model.predict(observation, deterministic=True)

    def save(self, path: str, **kwargs: Dict) -> None:
        self._model.save(path)

    def load(self, path: str, **kwargs: Dict) -> None:
        self._model = SBPPO.load(path, env=self._env)


class _CustomCNN(BaseFeaturesExtractor):
    """A custom CNN for extracting features from the input images

    :param observation_space: The observation space of the environment
    :type observation_space: gym.spaces.Box
    :param features_dim: The size of the output feature vector, defaults to 256
    :type features_dim: int, optional
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        self._cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=observation_space.shape[0],
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding="valid",
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        # We need do to this because we don't know what the original input is, so we can't compute the sizes manually
        with torch.no_grad():
            n_flatten = self._cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self._linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Generates feature vectors from the given observations

        :param observations: The input observations
        :type observations: torch.Tensor
        :return: Feature vectors
        :rtype: torch.Tensor
        """
        return self._linear(self._cnn(observations))


class _TqdmCallback(BaseCallback):
    """A callback that displays a progress bar during training

    Taken from: https://github.com/hill-a/stable-baselines/issues/297
    """

    def __init__(self):
        super().__init__()

        self._progress_bar = None

    def _on_training_start(self):
        self._progress_bar = tqdm(total=self.locals["total_timesteps"])

    def _on_step(self):
        self._progress_bar.update(1)
        return True

    def _on_training_end(self):
        self._progress_bar.close()
        self._progress_bar = None
