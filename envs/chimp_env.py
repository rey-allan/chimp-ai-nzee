"""Wrapper of chimpanzee Theory of Mind Pycolab experiment as a Gym environment."""
import random
import time
import tkinter
from collections import defaultdict
from itertools import product
from typing import List, Tuple

import gym
import numpy as np
import torch
from gym import spaces
from pycolab.cropping import ObservationCropper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

from .chimp_experiment import Actions, Characters, ExperimentSettings, make_game, make_subordinate_cropper

# Colors for rendering
RENDERING_COLORS = {
    Characters.SUBORDINATE: "#00FFFF",
    Characters.DOMINANT: "#FF00FF",
    Characters.WALL: "#DCDCDC",
    Characters.COLLECTIBLE_LEFT: "#FFFF6E",
    Characters.COLLECTIBLE_RIGHT: "#FFFF6E",
}
# Colors for producing the observation image (state)
OBSERVATION_COLORS = {
    Characters.SUBORDINATE: (0, 255, 255),
    Characters.DOMINANT: (255, 0, 255),
    Characters.WALL: (220, 220, 220),
    Characters.COLLECTIBLE_LEFT: (255, 255, 110),
    Characters.COLLECTIBLE_RIGHT: (255, 255, 110),
}


# pylint: disable=too-many-instance-attributes
class ChimpTheoryOfMindEnv(gym.Env):
    """A Gym environment for the chimpanzee Theory of Mind Pycolab experiment

    :param seed: The random seed to use, defaults to None
    :type seed: int, optional
    """

    def __init__(self, seed: int = None) -> None:
        super().__init__()

        random.seed(seed)
        torch.manual_seed(seed)

        self._game = None
        self._obs_cropper = make_subordinate_cropper()
        # Use a default observation cropper for rendering
        # This default cropper will actually display the whole game
        self._render_cropper = ObservationCropper()
        self._done = False
        self._raw_obs = None
        self._renderer = None
        self._dominant_agent = _Dominant()

        # The observation is the RBG viewbox of the subordinate perspective
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self._obs_cropper.rows, self._obs_cropper.cols),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(Actions.list()))
        self.action_space.seed(seed)

    def reset(self) -> np.ndarray:
        """Resets the environment

        :return: The first state for the subordinate subject
        :rtype: np.ndarray
        """
        # Reset to a random experiment setting
        setting = random.choice(ExperimentSettings.list())
        self._game = make_game(setting)
        self._obs_cropper.set_engine(self._game)
        self._render_cropper.set_engine(self._game)
        self._renderer = _Renderer(cell_size=25, colors=RENDERING_COLORS, croppers=[self._render_cropper])

        self._dominant_agent.reset(setting)

        observation, _, _ = self._game.its_showtime()
        self._raw_obs = observation
        self._done = self._game.game_over

        return self._generate_obs(observation)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Steps into the environment executing the given action for the subordinate agent

        :param action: The action to execute for the subordinate
        :type action: int
        :return: A tuple of subordinate observation, reward, done?, info
        :rtype: Tuple[np.ndarray, float, bool, dict]
        """
        assert self._game is not None, "Game is not initialized, call reset function before step"
        assert self._done is False, "Step can not be called after the game is terminated!"

        observation, reward, _ = self._game.play(
            {
                Characters.SUBORDINATE: action,
                Characters.DOMINANT: self._dominant_agent.action(),
            }
        )
        self._raw_obs = observation
        self._done = self._game.game_over

        return self._generate_obs(observation), reward, self._done, {}

    def render(self, mode="human") -> None:
        """Renders the environment in a separate window"""
        self._renderer(self._raw_obs)

    def close(self) -> None:
        """Closes the environment"""
        self._renderer.close()

    def _generate_obs(self, observation: np.ndarray) -> np.ndarray:
        obs = self._obs_cropper.crop(observation)
        img = np.zeros(self.observation_space.shape)

        # Generate an RGB image of the cropped observation
        for character in Characters.list():
            value = ord(character)
            color = OBSERVATION_COLORS[character]
            img[0, obs.board == value] = color[0]
            img[1, obs.board == value] = color[1]
            img[2, obs.board == value] = color[2]

        return img.astype(np.uint8)


def make_env(n_frames: int = 4, n_envs: int = 1, seed: int = None) -> ChimpTheoryOfMindEnv:
    """Builds a chimp Theory of Mind environment

    :param n_frames: The number of frames to stack as input to the agent, defaults to 4
    :type n_frames: int, optional
    :param n_envs: The number of concurrent environments to use, defaults to 1
    :type n_envs: int, optional
    :param seed: The seed to use for the random number generator, defaults to None
    :type seed: int, optional
    :return: An instance of the environment
    :rtype: ChimpTheoryOfMindEnv
    """
    env = make_vec_env(ChimpTheoryOfMindEnv, n_envs=n_envs, seed=seed, env_kwargs=dict(seed=seed))
    env = VecFrameStack(env, n_stack=n_frames)

    return env


class _Dominant:
    """Defines the deterministic behavior of the dominant agent"""

    PATH_TO_LEFT = [
        Actions.LEFT,
        Actions.LEFT,
        Actions.LEFT,
        Actions.LEFT,
        Actions.LEFT,
        Actions.UP,
        Actions.UP,
        Actions.LEFT,
        Actions.LEFT,
        Actions.LEFT,
    ]

    PATH_TO_RIGHT = [
        Actions.LEFT,
        Actions.LEFT,
        Actions.LEFT,
        Actions.LEFT,
        Actions.LEFT,
        Actions.DOWN,
        Actions.DOWN,
        Actions.LEFT,
        Actions.LEFT,
        Actions.LEFT,
    ]

    def __init__(self) -> None:
        # Track the direction the subject is going
        self._path = None
        # Tracks the next action to take
        self._action_idx = 0

    def reset(self, setting: int) -> None:
        """Resets the agent

        :param setting: The experiment setting being used
        :type setting: int
        """
        # No barrier setting, choose a random direction to go
        if setting == ExperimentSettings.NO_BARRIER:
            self._path = random.choice([self.PATH_TO_LEFT, self.PATH_TO_RIGHT])
        else:
            # For the barrier setting, always choose the right (the one not behind the barrier)
            self._path = self.PATH_TO_RIGHT
        self._action_idx = 0

    def action(self) -> int:
        """Returns the next action to take

        :return: The next action
        :rtype: int
        """
        assert self._path is not None, "Dominant subject hasn't been initialized! Make sure to call `reset()` first."

        try:
            action = self._path[self._action_idx]
            self._action_idx += 1
        except IndexError:
            # The dominant agent has reached the end of its chosen path, just stay in place thereafter
            action = Actions.STAY

        return action


class _Renderer:
    """Renders the environment in a TKinter window as a series of colored "cells".

    Modified from: https://github.com/TolgaOk/gymcolab/blob/master/gymcolab/renderer/window.py
    """

    DEFAULT_COLOR = "#000000"

    def __init__(
        self, cell_size: int, colors: dict[str, Tuple[int, int, int]], croppers: List[ObservationCropper]
    ) -> None:
        self._root = tkinter.Tk()
        self._root.title("Chimpanzee Theory of Mind Experiment")

        self._croppers = sorted(croppers, key=lambda x: x.rows, reverse=True)
        width = (sum(cropper.cols for cropper in croppers) + len(croppers) - 1) * cell_size
        height = max(cropper.rows for cropper in croppers) * cell_size

        self._canvas = tkinter.Canvas(self._root, width=width, height=height, bg="gray")
        self._canvas.pack()
        self._canvas_height = height
        self._canvas_widht = width
        self._cell_size = cell_size
        self._border_ratio = 0.05
        self._all_cells = None

        self._colors = defaultdict(lambda: self.DEFAULT_COLOR)
        for key, value in colors.items():
            self._colors[ord(key)] = value

    def __call__(self, board: np.ndarray) -> None:
        self._all_cells = self._all_cells or self._init_render()

        for cropper, cells in zip(self._croppers, self._all_cells):
            cropped_board = cropper.crop(board).board

            for i, value in enumerate(cropped_board.flatten("F")):
                self._canvas.itemconfig(cells[i], fill=self._colors[value])

        self._root.update()
        # Add a small delay for better visualization
        time.sleep(0.2)

    def close(self) -> None:
        """Closes the renderer window"""
        self._root.destroy()

    def _init_render(self) -> List:
        all_cells = []
        global_col = 0

        for cropper in self._croppers:
            rows = cropper.rows
            cols = cropper.cols

            b_w = int(self._cell_size * self._border_ratio)
            b_h = int(self._cell_size * self._border_ratio)

            cells = [
                self._canvas.create_rectangle(
                    x * self._cell_size + b_w,
                    y * self._cell_size + b_h,
                    (x + 1) * self._cell_size - b_w,
                    (y + 1) * self._cell_size - b_h,
                )
                for x, y in product(range(global_col, cols + global_col), range(rows))
            ]
            all_cells.append(cells)
            global_col += 1 + cropper.cols

        return all_cells


def _main() -> None:
    env = ChimpTheoryOfMindEnv(seed=42)

    _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        env.render()
        total_reward += reward if reward else 0.0

    print(f"Total reward: {total_reward:.4f}")


if __name__ == "__main__":
    _main()
