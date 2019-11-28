"""
Action
"""

from __future__ import annotations

from ...base import Action as BaseAction
from .direction import Direction

class Action(BaseAction):
    """
    Action
    """

    def __init__(self, direction: Direction):
        """
        # Arguments
            direction: Direction. Direction of action.
        """
        self.direction = direction

    def __str__(self):
        return self.direction.__str__()

    @property
    def data(self) -> int:
        """
        # Returns int value of the direction.
        """
        return self.direction.value

    def rotate_left(self) -> Action:
        """
        # Return newly left-rotating action.
        """
        if self.direction == Direction.UP:
            direction = Direction.LEFT
        elif self.direction == Direction.RIGHT:
            direction = Direction.UP
        elif self.direction == Direction.DOWN:
            direction = Direction.RIGHT
        else: # LEFT
            direction = Direction.DOWN
        return Action(direction)

    def rotate_right(self) -> Action:
        """
        # Return newly right-rotating action.
        """
        if self.direction == Direction.UP:
            direction = Direction.RIGHT
        elif self.direction == Direction.RIGHT:
            direction = Direction.DOWN
        elif self.direction == Direction.DOWN:
            direction = Direction.LEFT
        else: # LEFT
            direction = Direction.UP
        return Action(direction)

    def turn(self) -> Action:
        """
        # Return newly turning action.
        """
        if self.direction == Direction.UP:
            direction = Direction.DOWN
        elif self.direction == Direction.RIGHT:
            direction = Direction.LEFT
        elif self.direction == Direction.DOWN:
            direction = Direction.UP
        else: # LEFT
            direction = Direction.RIGHT
        return Action(direction)

    def flip(self) -> Action:
        """
        # Return newly flipping action.
        """
        if self.direction == Direction.UP or self.direction == Direction.DOWN:
            direction = self.direction
        elif self.direction == Direction.RIGHT:
            direction = Direction.LEFT
        else: # LEFT
            direction = Direction.RIGHT
        return Action(direction)
