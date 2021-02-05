"""
State
"""

from math import ceil, log
from random import choice
from typing import List, Tuple

from ...base import State as BaseState, Action
from .direction import Direction

class State(BaseState):
    """
    State. Represents the board contains tiles.
    """

    _EMPTY: int = 0

    def __init__(self, board: List[List[int]] = None, size: int = None, unit: int = None):
        """
        # Arguments
            board: List[List[int]] = None. Default board.
            size: int = None. The size of the board.
            unit: int = None. Unit value for tile, other valid values are powers of this unit value.
        """
        if board is not None:
            self.size = size or len(board)
            self.unit = unit or min(tile for row in board for tile in row if tile != self._EMPTY)
            self._board = board
        else:
            self.size = size
            self.unit = unit
            self._cleared()

    def __eq__(self, other: "State") -> bool:
        return self._board == other._board

    def __str__(self):
        width = ceil(log(self._max, 10))
        return "\n".join([
            "".join([
                f"{tile:{width}}" if tile != self._EMPTY else " " * width for tile in row
            ]) for row in self._board
        ])

    def reset(self):
        self._cleared()
        self._seeded()

    def executed(self, action: Action) -> float:
        is_changed, total_merged_value = self._collapsed(Direction(action.data))
        if is_changed:
            self._seeded()
        return 0 if total_merged_value == 0 else log(total_merged_value) / log(self._max)

    def is_ended(self):
        return not self._is_collapsible()

    @property
    def data(self) -> List[float]:
        """
        Flattens then normalizes the values of board
        # Returns list value of the flattened board.
        """
        return [
            tile if tile == self._EMPTY else log(tile) / log(self._max)
            for row in self._board for tile in row
        ]

    @property
    def _max(self) -> int:
        """
        # Returns the maximum achievable tile.
        """
        return self.unit ** (self.size ** 2)

    def _cleared(self):
        """
        Clears the board.
        """
        # Initialize a `size` x `size` board filled with empty values
        self._board = self._create_empty_board()

    def _seeded(self) -> int:
        """
        Randomly seeds a new tile in an empty spot on the board with unit value.
        # Returns the value of the new tile if it is successfully seeded,
            otherwise returns `0` if the board has no spot for seeding new tile.
        # Examples
            0   2   0   0           0   2   0   0
            0   0   0   0      →    0   0   0   0
            0   0   0   0           0   0   0   2 ← new tile
            0   0   2   0           0   0   2   0
        """
        flattened_board = [tile for row in self._board for tile in row]
        empty_indices = [index for index, tile in enumerate(flattened_board) if tile == self._EMPTY]
        if len(empty_indices) == 0:
            return 0
        index = choice(empty_indices)
        self._board[index // self.size][index % self.size] = self.unit
        return self.unit

    def _collapsed(self, direction: Direction) -> Tuple[bool, int]:
        """
        Collapse the entire board in a given direction.
        # Arguments
            direction: Direction. Collapsing direction.
        # Returns a flag indicates whether the board is changed after collapsing or not
            and merged value.
        # Examples
            0   0   0   4                                   0   0   2   4
            0   0   0   0       direction:  UP         →    0   0   0   2
            0   0   0   2                                   0   0   0   0
            0   0   2   0                                   0   0   0   0
        """
        old_state = self.clone()
        total_merged_value = 0
        for i in range(self.size):
            collapsed_array, merged_value = self._collapse(self._peel(direction, i), self._EMPTY)
            self._paved(direction, i, collapsed_array)
            total_merged_value += merged_value
        return (self != old_state, total_merged_value)

    def _is_collapsible(self) -> bool:
        """
        # Returns a flag indicates whether the board is collapsible or not.
        """
        for i in range(self.size):
            row = self._board[i]
            column = [r[i] for r in self._board]
            for j in range(self.size):
                if (row[j] == self._EMPTY or column[j] == self._EMPTY
                        or j > 0 and (row[j] == row[j - 1] or column[j] == column[j - 1])):
                    return True
        return False

    def _peel(self, direction: Direction, index: int) -> List[int]:
        """
        Peel off an array from the board by a given direction and index.
        # Arguments
            direction: Direction. Peeling direction.
            index: int. Index of row or column.
        # Returns the array peeled off from the board.
        # Examples
            0   2   4   0       direction:  RIGHT
            0   0   0   0       index: 0               →    0   4   2   0
            0   0   2   0
            0   0   0   0
        """
        if direction in (Direction.LEFT, Direction.RIGHT):
            array = self._board[index]
        else:
            array = [row[index] for row in self._board]
        if direction in (Direction.RIGHT, Direction.DOWN):
            array = list(reversed(array))
        return array

    def _paved(self, direction: Direction, index: int, array: List[int]):
        """
        Copy entire array elements into specific row or column.
        # Arguments
            direction: Direction. Paving direction.
            index: int. Index of row or column.
            array: List[int]. Array to be copied.
        # Examples
            0   2   0   0       direction:  UP                  0   2   4   0
            0   0   0   0       index: 3                   →    0   0   0   0
            0   0   0   0       array:  4   0   2   0           0   0   2   0
            0   0   2   0                                       0   0   0   0
                                                                        ↑
                                                                        paved column
        """
        array = array[:self.size]
        if direction in (Direction.RIGHT, Direction.DOWN):
            array = reversed(array)
        for i, tile in enumerate(array):
            if direction in (Direction.LEFT, Direction.RIGHT):
                self._board[index][i] = tile
            else:
                self._board[i][index] = tile

    def _create_empty_board(self) -> List[List[int]]:
        """
        # Returns new empty board.
        """
        return [row[:] for row in [[self._EMPTY] * self.size] * self.size]

    @staticmethod
    def _collapse(array: List[int], empty_element: int) -> Tuple[List[int], int]:
        """
        Slide entire elements of an array as far as possible to the left side.
        If two elements of the same number collide while moving,
            they will merge into an element with the total value of the two elements that collided.
        The resulting element cannot merge with another element again in the same move.
        # Arguments
            array: List[int]. Original array.
            empty_element: int. Value of empty element.
        # Returns the collapsed array and sum of merged values.
        # Examples
            1. No elements merged
                0   2   0   0      →    2   0   0   0
            2. One element merged
                0   2   2   4      →    4   4   0   0
                                        ↑
                                        2+2
            3. Two elements merged
                2   2   2   2      →    4   4   0   0
                                        ↑   ↑
                                        2+2 2+2
        """
        collapsed_array = []
        is_merging = False
        merged_values = []
        for element in array:
            if element == empty_element:
                continue
            last_element = None if not collapsed_array else collapsed_array[-1]
            if element == last_element and not is_merging:
                collapsed_array[-1] += element
                merged_values.append(last_element + element)
                is_merging = True
            else:
                collapsed_array.append(element)
                is_merging = False
        collapsed_array += [empty_element] * (len(array) - len(collapsed_array))
        return (collapsed_array, sum(merged_values))
