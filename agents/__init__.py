from typing import Dict, Union

from stable_baselines3.common.base_class import BaseAlgorithm

from .agent import Agent
from .baselines import PPO


def make_agent(agent_type: str, **kwargs: Dict) -> Union[Agent, BaseAlgorithm]:
    """Makes an agent of the specified typ

    :param agent_type: The type of agent to make
    :type agent_type: str
    :raises ValueError: If the agent type is not supported
    :return: An agent instance
    :rtype: Union[Agent, BaseAlgorithm]
    """
    if agent_type == "PPO":
        return PPO(**kwargs)

    raise ValueError(f"Agent of type {agent_type} is not supported!")
