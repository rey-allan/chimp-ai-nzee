"""Script for demoing subordinate RL agents."""
import argparse

from agents import make_agent
from envs import make_env


def _main(agent_type: str, model_name: str, episodes: int, experiment_setting: int, n_frames: int) -> None:
    print(f"Demoing agent {model_name} in experiment {experiment_setting} for {episodes} episodes")

    seed = 24 * (experiment_setting + 1)
    env = make_env(n_frames=n_frames, experiment_setting=experiment_setting, seed=seed)
    model = make_agent(agent_type, name=model_name, env=env, seed=seed)
    model.load(f"./models/{model_name}")

    max_timesteps = 100
    for _ in range(episodes):
        obs = env.reset()

        for _ in range(max_timesteps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            env.render()

            if done:
                break

        # Close the environment so the renderer window gets destroyed as well
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demoes RL agents that represent the subordinate subject")
    parser.add_argument("-a", type=str, required=True, help="The type of the trained agent to use")
    parser.add_argument("-m", type=str, required=True, help="Name of the model to use for loading artifacts")
    parser.add_argument("-e", type=int, default=10, help="The number of demoing episodes. Defaults to 10")
    parser.add_argument("--setting", type=int, required=True, help="Experiment setting to train on")
    parser.add_argument("--n-frames", type=int, default=1, help="Frames to stack as input to the agent. Defaults to 1")

    args = parser.parse_args()

    _main(args.a, args.m, args.e, args.setting, args.n_frames)
