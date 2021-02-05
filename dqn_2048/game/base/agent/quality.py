"""
Quality
"""

from abc import abstractmethod
from random import choice
from typing import List, Tuple

import numpy as np

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
    def copied(self, training_quality: "Quality"):
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
        actions, _ = self._select([state])
        return actions[0]

    def randomly_act(self) -> Action:
        """
        # Returns random action.
        """
        return Action(choice(range(self.output_size)))

    def calculate(self, transitions: List[Transition]) -> List[float]:
        """
        Calculate target y = r if the episode has ended at this step,
            or y = r + γ * maxa'∈A(Qˆs',a') otherwise.
        Used by the "target quality model" Qˆ.
        # Arguments
            transitions: List[Transition]. Sample of transitions in the buffer.
        # Returns list of discounted cumulative rewards.
        """
        next_states = [t.state for t in transitions]
        rewards = [t.reward for t in transitions]
        _, values = self._select(next_states)
        return [
            r if s.is_ended() else r + self.gamma * v
            for s, r, v in zip(next_states, rewards, values)
        ]

    def _select(self, states: List[State]) -> Tuple[List[Action], List[float]]:
        """
        # Arguments
            states: List[State]. Used for selecting best actions.
        # Returns list of best actions with a = argmaxa(Qs,a) and the corresponding values.
        """
        values = self._predict(states)
        indices = values.argmax(axis=1)
        return (
            [Action(index) for index in indices],
            [values[i][index] for i, index in enumerate(indices)]
        )

    @abstractmethod
    def _predict(self, states: List[State]) -> np.ndarray:
        """
        # Arguments
            states: List[State]. List of observed states.
        # Returns list of action values for given states.
        """
