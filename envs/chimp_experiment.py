"""Implementation of chimpanzee Theory of Mind experiment as a Pycolab game."""
import curses
import sys

from pycolab import ascii_art, human_ui
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

ENV_ART = [
    # Legend:
    #   "#": impassable walls
    #   "S": subordinate location
    #   "D": dominant location
    #   "l": collectible left (w.r.t the subordinate)
    #   "r": collectible right (w.r.t the subordinate)
    # No barrier setting
    [
        "####################",
        "#                  #",
        "#                  #",
        "#        l         #",
        "#                  #",
        "#S                D#",
        "#                  #",
        "#        r         #",
        "#                  #",
        "#                  #",
        "####################",
    ],
    # Barrier setting
    [
        "####################",
        "#                  #",
        "#         #        #",
        "#        l#        #",
        "#         #        #",
        "#S                D#",
        "#                  #",
        "#        r         #",
        "#                  #",
        "#                  #",
        "####################",
    ],
]

# These colors are only for humans to see in the CursesUi
COLOR_FG = {
    # Default black background
    " ": (0, 0, 0),
    # Walls
    "#": (220, 220, 220),
    # Subordinate
    "S": (0, 999, 999),
    # Dominant
    "D": (999, 0, 780),
    # Collectables
    "l": (999, 862, 110),
    "r": (999, 862, 110),
}


def make_game(setting):
    """Builds and returns a setting for the chimpanzee Theory of Mind experiment"""
    return ascii_art.ascii_art_to_game(
        ENV_ART[setting],
        what_lies_beneath=" ",
        sprites={"S": SubjectSprite, "D": SubjectSprite},
        drapes={
            "l": ascii_art.Partial(CollectibleDrape, name="Left"),
            "r": ascii_art.Partial(CollectibleDrape, name="Right"),
        },
        update_schedule=["S", "D", "l", "r"],
        z_order="lrSD",
    )


class SubjectSprite(prefab_sprites.MazeWalker):
    """A `Sprite` for the subjects of the experiment, the subordinate and the dominant."""

    def __init__(self, corner, position, character):
        super().__init__(corner, position, character, impassable="#")

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers

        # Actions are sent as a `dict`: {"S": action_subordinate, "D": action_dominant}
        action = 4 if actions is None else actions.get(self.character, 4)

        if action == 0:
            self._north(board, the_plot)
        elif action == 1:
            self._south(board, the_plot)
        elif action == 2:
            self._west(board, the_plot)
        elif action == 3:
            self._east(board, the_plot)
        elif action == 4:
            self._stay(board, the_plot)


class CollectibleDrape(plab_things.Drape):
    """A `Drape` handling the collectibles.

    This Drape detects when a subject reaches a collectible, removing it and
    penalizing or rewarding the subordinate appropriately.
    """

    def __init__(self, curtain, character, name):
        super().__init__(curtain, character)
        self._name = name

    def update(self, actions, board, layers, backdrop, things, the_plot):
        subordinate_position = things["S"].position
        dominant_position = things["D"].position

        # If the subordinate has reached a collectible
        if self.curtain[subordinate_position]:
            # But it is the same location the dominant went for, it gets penalized
            if subordinate_position == dominant_position:
                the_plot.log("Subordinate and dominant reached the same collectible!")
                the_plot.add_reward(-100.0)
            # If it reaches that collectible by itself, then it received the reward
            else:
                the_plot.log("Subordinate has reached a collectible!")
                the_plot.add_reward(100.0)

            # Track information of which collectible was reached by the subordinate
            the_plot["Collected"] = self._name
            # Experiment is done
            the_plot.terminate_episode()

        # If the dominant has reached a collectible, we simply remove it, without any reward
        if self.curtain[dominant_position]:
            the_plot.log("Dominant has reached a collectible!")
            self.curtain[dominant_position] = False


def _main(argv=()):
    setting = int(argv[1]) if len(argv) > 1 else 0

    game = make_game(setting)

    # Make a CursesUi to play it with
    # pylint: disable=invalid-name
    ui = human_ui.CursesUi(
        keys_to_actions={
            "w": {"S": 0, "D": 4},
            "s": {"S": 1, "D": 4},
            "a": {"S": 2, "D": 4},
            "d": {"S": 3, "D": 4},
            curses.KEY_UP: {"S": 4, "D": 0},
            curses.KEY_DOWN: {"S": 4, "D": 1},
            curses.KEY_LEFT: {"S": 4, "D": 2},
            curses.KEY_RIGHT: {"S": 4, "D": 3},
        },
        delay=100,
        colour_fg=COLOR_FG,
    )

    ui.play(game)


if __name__ == "__main__":
    _main(sys.argv)
