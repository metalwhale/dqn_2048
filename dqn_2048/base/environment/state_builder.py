"""
State builder
"""

from abc import abstractmethod

from .state import State

class StateBuilder:
    """
    State builder
    """

    @abstractmethod
    def build(self) -> State:
        """
        # Returns newly state.
        """
