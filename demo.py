"""Script for demoing subordinate RL agents."""
import argparse
from pathlib import Path

import imageio

from agents import make_agent
from envs import ExperimentSettings, make_env


def _main(
    agent_type: str,
    model_name: str,
    episodes: int,
    setting: int,
    n_frames: int,
    save_gif: bool,
) -> None:
    print(f"Demoing agent {model_name} in experiment {ExperimentSettings.name(setting)} for {episodes} episodes")

    seed = 24 * (setting + 1)
    env = make_env(n_frames=n_frames, experiment_setting=setting, seed=seed)
    model = make_agent(agent_type, name=model_name, env=env, seed=seed)
    model.load(f"./models/{model_name}")

    if save_gif:
        demo_path = f"results/{model_name}/demos"
        Path(demo_path).mkdir(exist_ok=True, parents=True)

    max_timesteps = 100
    images = []
    render_mode = "human" if not save_gif else "rgb_array"

    for _ in range(episodes):
        obs = env.reset()
        img = env.render(mode=render_mode)

        for _ in range(max_timesteps):
            images.append(img)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            img = env.render(mode=render_mode)

            if done:
                break

        # Close the environment so the renderer window gets destroyed as well
        env.close()

    # Generate gif, if requested, of all episodes together
    if save_gif:
        imageio.mimsave(
            f"{demo_path}/experiment_{ExperimentSettings.name(setting).replace(' ', '_')}.gif",
            images,
            fps=15,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demoes RL agents that represent the subordinate subject")
    parser.add_argument("-a", type=str, required=True, help="The type of the trained agent to use")
    parser.add_argument("-m", type=str, required=True, help="Name of the model to use for loading artifacts")
    parser.add_argument("-e", type=int, default=10, help="The number of demoing episodes. Defaults to 10")
    parser.add_argument("--setting", type=int, required=True, help="Experiment setting to train on")
    parser.add_argument("--n-frames", type=int, default=1, help="Frames to stack as input to the agent. Defaults to 1")
    parser.add_argument("--save-gif", action="store_true", help="Whether to save gifs for the episodes")

    args = parser.parse_args()

    _main(args.a, args.m, args.e, args.setting, args.n_frames, args.save_gif)
