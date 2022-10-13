"""Script for training an RL agents that represent the subordinate subject of the experiment."""
import argparse

from agents import make_agent
from envs import make_env


def _main(agent_type: str, timesteps: int, n_frames: int, n_envs: int, setting: int) -> None:
    seed = 24
    env = make_env(n_frames, n_envs, setting, seed)

    model = make_agent(agent_type, env=env, seed=seed)

    model.learn(total_timesteps=timesteps)
    model.save(f"models/{agent_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains RL agents that represent the subordinate subject")
    parser.add_argument("-a", type=str, required=True, help="The type of the agent to use")
    parser.add_argument("-t", type=int, default=500, help="The total training timesteps. Defaults to 500")
    parser.add_argument("--n-frames", type=int, default=1, help="Frames to stack as input to the agent. Defaults to 1")
    parser.add_argument("--n-envs", type=int, default=1, help="Environments to use for training. Defaults to 1")
    parser.add_argument("--setting", type=int, default=None, help="Experiment setting to train on. Defaults to None")

    args = parser.parse_args()

    print(f"Training RL agent using the following args: {args}")
    _main(args.a, args.t, args.n_frames, args.n_envs, args.setting)
