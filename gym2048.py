from __future__ import print_function

import itertools
import logging
import sys

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from six import StringIO


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class IllegalMove(Exception):
    pass


def stack(flat, layers=16):
    larray = []
    for i in range(1, layers + 1):
        ii = 2**i
        layer = np.copy(flat)
        layer[layer != ii] = 0
        layer[layer == ii] = 1
        # print("Layer")
        # print(layer)
        # print(layer.shape)
        larray.append(layer)

    newstack = np.stack(larray, axis=-1)
    return newstack


class Game2048Env(gym.Env):  # directions 0, 1, 2, 3 are up, right, down, left
    metadata = {"render.modes": ["human", "ansi"]}
    max_steps = 1000000

    def __init__(self):
        self.logger = logging.getLogger("dqn")
        # Definitions for game. Board must be square.
        self.size = 4
        self.w = self.size
        self.h = self.size
        self.squares = self.size * self.size

        # Maintain own idea of game score, separate from rewards
        self.score = 0

        # Members for gym implementation
        self.action_space = spaces.Discrete(4)
        # Suppose that the maximum tile is as if you have powers of 2 across the board.
        layers = self.squares
        self.observation_space = spaces.Box(
            0, 1, (self.w, self.h, layers), dtype=np.int32
        )
        self.set_illegal_move_reward(0.0)
        self.set_max_tile(None)

        self.max_illegal = 5  # max number of illegal actions
        self.num_illegal = 0

        # Initialise seed
        self.seed()

        # # Reset ready for a game
        # self.reset()

    def _get_info(self, info=None):
        if not info:
            info = {}
        else:
            assert type(info) == dict, "info should be of type dict!"

        info["highest"] = self.highest
        info["score"] = self.score
        info["steps"] = self.steps
        return info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_illegal_move_reward(self, reward):
        """Define the reward/penalty for performing an illegal move. Also need
        to update the reward range for this."""
        # Guess that the maximum reward is also 2**squares though you'll probably never get that.
        # (assume that illegal move reward is the lowest value that can be returned
        self.illegal_move_reward = reward
        self.reward_range = (self.illegal_move_reward, float(2**self.squares))

    def set_max_tile(self, max_tile):
        """Define the maximum tile that will end the game (e.g. 2048). None means no limit.
        This does not affect the state returned."""
        assert max_tile is None or isinstance(max_tile, int)
        self.max_tile = max_tile

    # Implement gym interface
    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        logging.debug("Action {}".format(action))
        self.steps += 1
        score = 0
        done = None
        info = {
            "illegal_move": False,
        }
        try:
            last_highest = self.highest
            last_num_tiles = self.num_tiles

            score = float(self.__move(action))
            self.score += score
            assert score <= 2 ** (self.w * self.h)
            self.__add_tile()
            done = self.is_end()
            reward = float(score) * (
                1.0
                + (
                    float((self.highest - last_highest) / last_highest)
                    - float((self.num_tiles - last_num_tiles) / last_num_tiles)
                )
            )
        except IllegalMove as e:
            logging.debug("Illegal move")
            info["illegal_move"] = True
            if self.steps > self.max_steps:
                done = True
            else:
                done = False
            reward = self.illegal_move_reward
            self.num_illegal += 1
            if (
                self.num_illegal >= self.max_illegal
            ):  # exceed the maximum number of illegal actions
                done = True

        info = self._get_info(info)

        # Return observation (board state), reward, done and info dict
        return self.Matrix, reward, done, info

    def reset(self):
        self.Matrix = np.zeros((self.h, self.w), np.int32)
        self.score = 0
        self.steps = 0
        self.num_illegal = 0

        logging.debug("Adding tiles")
        self.__add_tile()
        self.__add_tile()

        return self.Matrix, 0, False, self._get_info()

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout
        s = "Score: {}\n".format(self.score)
        s += "Highest: {}\n".format(self.highest)
        npa = np.array(self.Matrix)
        grid = npa.reshape((self.size, self.size))
        s += "{}\n\n".format(grid)
        outfile.write(s)
        return outfile

    # Implement 2048 game logic
    def __add_tile(self):
        """Add a tile, probably a 2 but maybe a 4"""
        possible_tiles = np.array([2, 4])
        tile_probabilities = np.array([0.9, 0.1])
        val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        empties = self.__empties()
        assert empties.shape[0]
        empty_idx = self.np_random.choice(empties.shape[0])
        empty = empties[empty_idx]
        logging.debug("Adding %s at %s", val, (empty[0], empty[1]))
        self.set(empty[0], empty[1], val)

    def get(self, x, y):
        """Return the value of one square."""
        return self.Matrix[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.Matrix[x, y] = val

    def __empties(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.Matrix == 0)

    @property
    def num_tiles(self):
        return np.count_nonzero(self.Matrix)

    @property
    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.Matrix)

    def __move(self, direction, trial=False):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the score that [would have] got."""
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = (
            dir_mod_two ^ dir_div_two
        )  # 0 for towards up left, 1 for towards bottom right

        # Construct a range for extracting row/column into a list
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = [self.get(x, y) for x in rx]
                (new, ms) = self.__shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [self.get(x, y) for y in ry]
                (new, ms) = self.__shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])
        if changed != True:
            raise IllegalMove

        return move_score

    def __combine(self, shifted_row):
        """Combine same tiles when moving to one side. This function always
        shifts towards the left. Also count the score of combined tiles."""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # Skip the next thing in the list.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def __shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]

        # Reverse list to handle shifting to the right
        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.__combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def is_end(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
        or there are no legal moves. If there are empty spaces then there
        must be legal moves."""

        if self.max_tile is not None and self.highest == self.max_tile:
            return True

        if self.steps >= self.max_steps:
            return True

        for direction in range(4):
            try:
                self.__move(direction, trial=True)
                # Not the end if we can do any move
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        """Retrieve the whole board, useful for testing."""
        return self.Matrix

    def set_board(self, new_board):
        """Retrieve the whole board, useful for testing."""
        self.Matrix = new_board


def main():
    gym = Game2048Env()
    gym.reset()
    gym.render()
    while True:
        act = int(input("input action:"))
        gym.step(act)
        gym.render()


if __name__ == "__main__":
    main()
