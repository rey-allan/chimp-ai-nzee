"""Wrapper of chimpanzee Theory of Mind Pycolab experiment as a Gym environment."""
import sys
import tkinter
from collections import defaultdict
from itertools import product
from typing import List, Tuple

import gym
import numpy as np

# pylint: disable=import-error
from chimp_experiment import Actions, Characters, make_game, make_subordinate_cropper
from gym import spaces
from pycolab.cropping import ObservationCropper
from pycolab.rendering import ObservationToFeatureArray

# Colors for rendering only
COLORS = {
    Characters.SUBORDINATE: "#00FFFF",
    Characters.DOMINANT: "#FF00FF",
    Characters.WALL: "#DCDCDC",
    Characters.COLLECTIBLE_LEFT: "#FFFF6E",
    Characters.COLLECTIBLE_RIGHT: "#FFFF6E",
}


# pylint: disable=too-many-instance-attributes
class ChimpTheoryOfMindEnv(gym.Env):
    """A Gym environment for the chimpanzee Theory of Mind Pycolab experiment

    :param seed: The random seed to use, defaults to None
    :type seed: int, optional
    """

    def __init__(self, seed: int = None) -> None:
        super().__init__()

        self._game = None
        self._obs_cropper = make_subordinate_cropper()
        # Use a default observation cropper for rendering
        # This default cropper will actually display the whole game
        self._render_cropper = ObservationCropper()
        self._to_feature = ObservationToFeatureArray(Characters.SUBORDINATE)
        self._done = False
        self._raw_obs = None
        self._renderer = None

        # The observation is the viewbox of the subordinate perspective
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, self._obs_cropper.rows, self._obs_cropper.cols))
        self.action_space = spaces.Discrete(len(Actions.list()))
        self.action_space.seed(seed)

    # pylint: disable=arguments-differ
    def reset(self, setting: int = 0) -> np.ndarray:
        """Resets the environment to the given experiment setting

        :param setting: The experiment setting to use, defaults to 0
        :type setting: int, optional
        :return: The first observation for the subordinate subject
        :rtype: np.ndarray
        """
        self._game = make_game(setting)
        self._obs_cropper.set_engine(self._game)
        self._render_cropper.set_engine(self._game)
        self._renderer = _Renderer(cell_size=25, colors=COLORS, croppers=[self._render_cropper])

        observation, _, _ = self._game.its_showtime()
        self._raw_obs = observation
        self._done = self._game.game_over

        return self._wrap_observation(observation)

    # pylint: disable=arguments-renamed
    def step(self, actions: dict[str, int]) -> Tuple[np.ndarray, float, bool, dict]:
        """Steps into the environment executing the given actions for both subjects

        :param actions: The actions to execute per subject: {"S": subordinate_action, "D": dominant_action}
        :type actions: dict[str, int]
        :return: A tuple of subordiante observation, reward, done?, info
        :rtype: Tuple[np.ndarray, float, bool, dict]
        """
        assert self._game is not None, "Game is not initialized, call reset function before step"
        assert self._done is False, "Step can not be called after the game is terminated!"

        observation, reward, _ = self._game.play(actions)
        self._raw_obs = observation
        self._done = self._game.game_over

        return self._wrap_observation(observation), reward, self._done, {}

    def render(self) -> None:
        """Renders the environment in a separate window"""
        self._renderer(self._raw_obs)

    def _wrap_observation(self, observation: np.ndarray) -> np.ndarray:
        return self._to_feature(self._obs_cropper.crop(observation))


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

    def __del__(self):
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


def _main(argv=()) -> None:
    setting = int(argv[1]) if len(argv) > 1 else 0
    env = ChimpTheoryOfMindEnv(seed=42)

    _ = env.reset(setting)
    done = False
    total_reward = 0

    while not done:
        actions = {Characters.SUBORDINATE: env.action_space.sample(), Characters.DOMINANT: env.action_space.sample()}
        _, reward, done, _ = env.step(actions)
        env.render()
        total_reward += reward if reward else 0.0

    print(f"Total reward: {total_reward:.4f}")


if __name__ == "__main__":
    _main(sys.argv)
