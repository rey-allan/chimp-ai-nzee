"""Script for training an RL agents that represent the subordinate subject of the experiment."""
import argparse

from agents import make_agent
from envs import make_env


def _main(agent_type: str, timesteps: int, n_frames: int, n_envs: int) -> None:
    print(f"Training RL agent {agent_type} for {timesteps} timesteps using {n_frames} frames and {n_envs} envs")

    seed = 24
    env = make_env(n_frames, n_envs, seed)

    model = make_agent(agent_type, env=env, seed=seed)

    model.learn(total_timesteps=timesteps)
    model.save(f"models/{agent_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains RL agents that represent the subordinate subject")
    parser.add_argument("-a", type=str, required=True, help="The type of the agent to use")
    parser.add_argument("-t", type=int, default=500, help="The total training timesteps. Defaults to 500")
    parser.add_argument("--n-frames", type=int, default=1, help="Number of frames to stack as input to the agent")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of concurrent environments to use for training")

    args = parser.parse_args()

    _main(args.a, args.t, args.n_frames, args.n_envs)
