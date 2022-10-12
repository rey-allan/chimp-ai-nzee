"""Defines base functionality shared by all agents."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import gym
import numpy as np


class Agent(ABC):
    """A base agent modeled after the StableBaselines API"""

    def __init__(self, env: gym.Env, seed: int = None) -> None:
        super().__init__()

        self._env = env
        self._seed = seed

    @abstractmethod
    def learn(self, **kwargs: Dict) -> None:
        """Runs a learning procedure to train the agent"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, observation: np.ndarray, **kwargs: Dict) -> Tuple[np.ndarray, Any]:
        """Predicts the action(s) to take given the input observation

        :param observation: The input observation to predict for
        :type observation: np.ndarray
        :return: The action(s) to take, and any other extra information (optional)
        :rtype: Tuple[np.ndarray, Any]
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str, **kwargs: Dict) -> None:
        """Saves the agent's parameters and related artifacts to the given path

        :param path: The path to save to
        :type path: str
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str, **kwargs: Dict) -> None:
        """Loads the agent's artifacts from the given path

        :param path: The path to load from
        :type path: str
        """
        raise NotImplementedError
