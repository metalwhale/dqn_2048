"""
Transition
"""

from __future__ import annotations

from .state import State
from .action import Action

class Transition:
    """
    Transition
    """

    def __init__(self, old_state: State, action: Action, reward: float, state: State):
        """
        # Arguments
            old_state: State. Old state.
            action: Action. Executed action.
            reward: float. Observed reward.
            state: State. Current state.
        """
        self.old_state = old_state
        self.action = action
        self.reward = reward
        self.state = state

    def __eq__(self, other: Transition) -> bool:
        return self.state == other.state
