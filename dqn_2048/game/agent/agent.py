"""
Agent
"""

from random import uniform

from ...base import Decision
from ...base import Agent as BaseAgent

class Agent(BaseAgent):
    """
    Agent
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.0
        self.epsilon_decay_rate = 0

    def set_epsilons(self, start: float, end: float, decay_rate: int):
        """
        # Arguments
            start: float. Starting positive epsilon value.
            end: float. Ending positive epsilon value, less than `epsilon_start`.
            decay_rate: int. The rate of epsilon decay schedule.
        """
        self.epsilon_start = start
        self.epsilon_end = end
        self.epsilon_decay_rate = decay_rate

    def _make_decision(self) -> Decision:
        # With probability ε, select a random action, otherwise use quality model to act
        epsilon = max(
            self.epsilon_start - self._step / self.epsilon_decay_rate,
            self.epsilon_end
        )
        is_exploring = uniform(0, self.epsilon_start) < epsilon
        return Decision.EXPLORE if is_exploring else Decision.EXPLOIT
