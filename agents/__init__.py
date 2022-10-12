from typing import Union

from stable_baselines3.common.base_class import BaseAlgorithm

from .agent import Agent


def make_agent(agent_type: str) -> Union[Agent, BaseAlgorithm]:
    """Makes an agent of the specified typ

    :param agent_type: The type of agent to make
    :type agent_type: str
    :raises ValueError: If the agent type is not supported
    :return: An agent instance
    :rtype: Union[Agent, BaseAlgorithm]
    """
    raise ValueError(f"Agent of type {agent_type} is not supported!")
