"""
Quality
"""

from __future__ import annotations

from abc import abstractmethod
from typing import List, Tuple

from ..environment.state import State
from ..environment.action import Action
from ..environment.transition import Transition
from .experience import Experience

class Quality:
    """
    Quality. The Q in 'Q-learning', the soul of DQN.
    """

    def __init__(self, gamma: float):
        """
        # Arguments
            gamma: float. The discount factor, used for Bellman approximation.
        """
        self.gamma = gamma

    @abstractmethod
    def learn(self, batch: List[Experience]):
        """
        Calculate loss: L = (Qs,a - y) ^ 2
            then update Q(s, a) using the SGD algorithm by minimizing the loss.
        # Arguments
            batch: List[Experience]. Batch of experience replay to be trained.
        """

    @abstractmethod
    def copied(self, training_quality: Quality):
        """
        Copy weights from "training quality model".
        Used by the "target quality model" Qˆ.
        # Arguments
            training_quality: Quality. The training quality model.
        """

    @abstractmethod
    def save(self, dir_path: str):
        """
        # Arguments
            dir_path: str. Path of directory to save the quality model.
        """

    def predict(self, state: State) -> Action:
        """
        Predict action to be executed.
        Used by the "training quality model" Q.
        # Arguments
            state: State. Observed state.
        # Returns action with max value.
        """
        return self._select_action(state)[0]

    def calculate(self, transition: Transition) -> float:
        """
        Calculate target y = r if the episode has ended at this step,
            or y = r + γ * maxa'∈A(Qˆs',a') otherwise.
        Used by the "target quality model" Qˆ.
        # Arguments
            transition: Transition. The transition in the buffer.
        # Returns the discounted cumulative reward.
        """
        next_state = transition.state
        reward = transition.reward
        if next_state.is_ended():
            return reward
        value = self._select_action(next_state)[1]
        return reward + self.gamma * value

    @abstractmethod
    def _select_action(self, state: State) -> Tuple[Action, float]:
        """
        # Arguments
            state: State. Observed state.
        # Returns a tuple contains action with a = argmaxa(Qs,a) and the corresponding value.
        """
