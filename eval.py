"""Script for evaluating subordinate RL agents."""
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from agents import make_agent
from envs import ExperimentSettings, make_env


def _plot_rewards(rewards: Dict[int, List[float]], output_path: str) -> None:
    plt.close()

    for setting, rews in rewards.items():
        plt.plot(rews, label=f"{ExperimentSettings.name(setting)}")

    plt.title("Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Avg. Reward")
    plt.legend(loc="best")
    plt.savefig(f"{output_path}/rewards.png", bbox_inches="tight")


def _plot_collected(collected: Dict[int, Dict[str, int]], output_path: str) -> None:
    plt.close()

    groups = list(collected[ExperimentSettings.NO_BARRIER].keys())
    x_axis = np.arange(len(groups))
    bar_width = 0.4

    for i, (setting, collect) in enumerate(collected.items()):
        plt.bar(x_axis + bar_width * i, list(collect.values()), bar_width, label=f"{ExperimentSettings.name(setting)}")

    plt.title("Frequency of Items Collected")
    plt.xlabel("Item")
    plt.xticks(x_axis + (bar_width / 2.0), groups)
    plt.ylabel("No. of times collected")
    plt.legend(loc="best")
    plt.savefig(f"{output_path}/items_collected.png", bbox_inches="tight")


def _main(agent_type: str, episodes: int, n_frames: int, render: bool = False) -> None:
    rewards_per_experiment = {setting: [] for setting in ExperimentSettings.list()}
    collected_per_experiment = {setting: {"Left": 0, "Right": 0, "N/A": 0} for setting in ExperimentSettings.list()}

    for experiment_setting in ExperimentSettings.list():
        print(f"Evaluating agent {agent_type} in experiment {experiment_setting} for {episodes} episodes")

        seed = 24 * (experiment_setting + 1)
        env = make_env(n_frames=n_frames, experiment_setting=experiment_setting, seed=seed)
        model = make_agent(agent_type, env=env, seed=seed)
        model.load(f"./models/{agent_type}")

        max_timesteps = 100
        for _ in tqdm(range(episodes)):
            rewards = []
            obs = env.reset()

            for _ in range(max_timesteps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

                rewards.append(reward)

                if render:
                    env.render()

                if done:
                    break

            rewards_per_experiment[experiment_setting].append(np.mean(rewards))
            collected_per_experiment[experiment_setting][info[0]["Item Collected"] or "N/A"] += 1

            if render:
                # Close the environment so the renderer window gets destroyed as well
                env.close()

    output_path = f"results/{agent_type}"
    Path(output_path).mkdir(exist_ok=True)
    _plot_rewards(rewards_per_experiment, output_path)
    _plot_collected(collected_per_experiment, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluates RL agents that represent the subordinate subject")
    parser.add_argument("-a", type=str, required=True, help="The type of the trained agent to use")
    parser.add_argument("-e", type=int, default=10, help="The number of evaluation episodes. Defaults to 10")
    parser.add_argument("--n-frames", type=int, default=1, help="Number of frames to stack as input to the agent")
    parser.add_argument("--render", action="store_true", help="Whether to render the environment or not")

    args = parser.parse_args()

    _main(args.a, args.e, args.n_frames, args.render)
