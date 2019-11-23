"""
State
"""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy

from .action import Action

class State:
    """
    State
    """

    @abstractmethod
    def reset(self):
        """
        Resets state.
        """

    @abstractmethod
    def executed(self, action: Action) -> float:
        """
        # Arguments
            action: Action. Action to be executed with purpose of changing the state.
        # Returns the reward achieved after executing action.
        """

    @abstractmethod
    def is_ended(self) -> bool:
        """
        # Returns a flag indicates whether the state is ended yet.
        """

    @abstractmethod
    def build(self) -> State:
        """
        # Returns newly state.
        """

    def clone(self) -> State:
        """
        # Returns newly cloned state.
        """
        return deepcopy(self)
