"""
State builder
"""

from ...base import StateBuilder as BaseStateBuilder
from .state import State

class StateBuilder(BaseStateBuilder):
    """
    State builder
    """

    def __init__(self):
        self.size = 0
        self.unit = 0

    def set_size(self, size: int):
        """
        # Arguments
            size: int. The size of the board.
        """
        self.size = size
        return self

    def set_unit(self, unit: int):
        """
        # Arguments
            unit: int. Unit value for tile, other valid values are powers of this unit value.
        """
        self.unit = unit
        return self

    def build(self) -> State:
        return State(size=self.size, unit=self.unit)
