"""
Environment
"""

from .state import State
from .state_builder import StateBuilder
from .action import Action
from .transition import Transition

class Environment:
    """
    Environment
    """

    def __init__(self, state_builder: StateBuilder):
        """
        # Arguments
            state_builder: StateBuilder. State builder.
        """
        self._state = self._create(state_builder)
        self.reset()

    def reset(self) -> State:
        """
        Resets to initial state.
        # Returns reset state.
        """
        self._state.reset()
        return self.current_state

    def execute(self, action: Action) -> Transition:
        """
        # Arguments
            action: Action. Action to be executed.
        # Returns a transition represents the result after executing given action.
        """
        old_state = self.current_state
        reward = self._state.executed(action)
        # Passing the cloned state instead of the original one as a parameter to prevent the value
        # from being accidentally changed due to the environment's state updating
        return Transition(old_state, action, reward, self.current_state)

    @property
    def current_state(self) -> State:
        """
        # Returns the current state.
        """
        return self._state.clone()

    @staticmethod
    def _create(state_builder: StateBuilder) -> State:
        """
        # Arguments
            state_builder: StateBuilder. Builder of the state.
        # Returns newly state.
        """
        return state_builder.build()
