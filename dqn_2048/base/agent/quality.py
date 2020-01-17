"""
Quality
"""

from __future__ import annotations

from abc import abstractmethod
from random import choice
from typing import List, Tuple

from numpy import ndarray

from ..environment.state import State
from ..environment.action import Action
from ..environment.transition import Transition
from .experience import Experience

class Quality:
    """
    Quality. The Q in 'Q-learning', the soul of DQN.
    """

    def __init__(self, gamma: float, output_size: int):
        """
        # Arguments
            gamma: float. The discount factor, used for Bellman approximation.
            output_size: int. Size of the action output space.
        """
        self.gamma = gamma
        self.output_size = output_size

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

    def act(self, state: State) -> Action:
        """
        Select an action to execute.
        Used by the "training quality model" Q.
        # Arguments
            state: State. Observed state.
        # Returns action with max value.
        """
        return self._select(state)[0]

    def randomly_act(self) -> Action:
        """
        # Returns random action.
        """
        return Action(choice(range(self.output_size)))

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
        value = self._select(next_state)[1]
        return reward + self.gamma * value

    def _select(self, state: State) -> Tuple[Action, float]:
        """
        # Arguments
            state: State. Used for selecting best action.
        # Returns best action with a = argmaxa(Qs,a) and the corresponding value.
        """
        values = self._predict(state)
        index = values.argmax()
        return (Action(index), values[index])

    @abstractmethod
    def _predict(self, state: State) -> ndarray:
        """
        # Arguments
            state: State. Observed state.
        # Returns action values for given state.
        """
