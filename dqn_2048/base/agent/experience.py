"""
Experience
"""

from ..environment.state import State
from ..environment.action import Action

class Experience:
    """
    Experience
    """

    def __init__(self, state: State, action: Action, value: float):
        """
        # Arguments
            state: State. Experience state.
            action: Action. Executed action.
            value: float. Target value.
        """
        self.state = state
        self.action = action
        self.value = value
