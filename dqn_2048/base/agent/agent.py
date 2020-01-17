"""
Agent
"""

from abc import abstractmethod
from collections import deque
from random import sample
from typing import Tuple

from ..environment.state import State
from ..environment.transition import Transition
from ..environment.environment import Environment
from .quality_builder import QualityBuilder
from .experience import Experience
from .decision import Decision

class Agent:
    """
    Agent. The body of DQN.
    """

    def __init__(
            self,
            quality_builder: QualityBuilder,
            batch_size: int,
            experiences_count: int,
            starting_step: int,
            target_syncing_frequency: int
        ):
        """
        # Arguments
            quality_builder: QualityBuilder. Quality builder.
            batch_size: int. The batch size sampled from the transition buffer.
            experiences_count: int. The maximum capacity of the buffer.
            starting_step: int. The count of steps we wait for before starting training
                to populate the transition buffer.
            target_syncing_frequency: int. How frequently we sync model weights
                from the training model to the target model,
                which is used for getting the value of the next state in the Bellman approximation.
        """
        self.batch_size = batch_size
        self.starting_step = starting_step
        self.target_syncing_frequency = target_syncing_frequency
        # Initialize parameters for Q(s, a) and Qˆ(s, a) with random weights
        # and empty transition buffer
        self._training_quality = quality_builder.build()
        self._target_quality = quality_builder.build()
        self._transitions = deque(maxlen=experiences_count)
        self._step = 0

    def observe(self, environment: Environment):
        """
        # Arguments
            environment: Environment. The environment to observe.
        """
        # Copy weights from Q to Qˆ
        if self._step % self.target_syncing_frequency == 0:
            self._target_quality.copied(self._training_quality)
        # Get current state of the environment
        if environment.current_state.is_ended():
            environment.reset()
        transition = self._transit(environment, True)
        # Store transition in the transition buffer
        self._transitions.append(transition)
        if self._step >= self.starting_step and len(self._transitions) >= self.batch_size:
            # Sample a random batch from the buffer
            batch = [
                Experience(t.old_state, t.action, self._target_quality.calculate(t))
                for t in sample(self._transitions, self.batch_size)
            ]
            # Update Q(s, a)
            self._training_quality.learn(batch)
        self._step += 1

    def play(self, environment: Environment) -> Tuple[State, int, float]:
        """
        # Arguments
            environment: Environment. The environment to play inside.
        # Returns the last state, number of transitions, and the cumulative reward.
        """
        environment.reset()
        transitions_count = 0
        reward = 0
        while True:
            transition = self._transit(environment, False)
            transitions_count += 1
            reward += transition.reward
            next_state = transition.state
            if next_state.is_ended():
                return (next_state, transitions_count, reward)

    def save(self, dir_path: str):
        """
        # Arguments
            dir_path: str. Path of directory to save the training quality model.
        """
        self._training_quality.save(dir_path)

    def _transit(self, environment: Environment, is_learning: bool) -> Transition:
        """
        # Arguments
            state: State. The state from which the action is selected.
            environment: Environment. Observing environment state to execute action.
            is_learning: bool. Flag indicates whether is learning or not.
        # Returns new transition.
        """
        state = environment.current_state
        if not is_learning or self._make_decision() == Decision.EXPLOIT:
            action = self._training_quality.act(state)
        else:
            action = self._training_quality.randomly_act()
        # Execute action a in an emulator and observe reward r and the next state s'
        transition = environment.execute(action)
        if is_learning:
            return transition
        # If the state has not been changed, keep randomly transiting until it is updated
        if transition.state == state:
            while True:
                action = self._training_quality.randomly_act()
                transition = environment.execute(action)
                if transition.state != state:
                    break
        return transition

    @abstractmethod
    def _make_decision(self) -> Decision:
        """
        Decide whether to exploit (keep doing things based on experiment)
            or explore (try something new)
        # Returns decision.
        """
