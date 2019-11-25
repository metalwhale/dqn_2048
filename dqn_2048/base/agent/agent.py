"""
Agent
"""

from typing import Tuple

from collections import deque
from random import uniform, sample

from ..environment.state import State
from ..environment.action import Action
from ..environment.action_builder import ActionBuilder
from ..environment.environment import Environment
from .quality_builder import QualityBuilder
from .experience import Experience

class Agent:
    """
    Agent. The body of DQN.
    """

    def __init__(
            self,
            quality_builder: QualityBuilder,
            action_builder: ActionBuilder,
            batch_size: int,
            experience_counts: int,
            starting_step: int,
            target_syncing_frequency: int,
            epsilon_start: float,
            epsilon_end: float,
            epsilon_decay_rate: int
        ):
        """
        # Arguments
            quality_builder: QualityBuilder. Quality builder.
            action_builder: ActionBuilder. Action builder.
            batch_size: int. The batch size sampled from the transition buffer.
            experience_counts: int. The maximum capacity of the buffer.
            starting_step: int. The count of steps we wait for before starting training
                to populate the transition buffer.
            target_syncing_frequency: int. How frequently we sync model weights
                from the training model to the target model,
                which is used to get the value of the next state in the Bellman approximation.
            epsilon_start: float. Starting positive epsilon value.
            epsilon_end: float. Ending positive epsilon value, less than `epsilon_start`.
            epsilon_decay_rate: int. The rate of epsilon decay schedule.
        """
        self.action_builder = action_builder
        self.batch_size = batch_size
        self.starting_step = starting_step
        self.target_syncing_frequency = target_syncing_frequency
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        # Initialize parameters for Q(s, a) and Qˆ(s, a) with random weights
        # and empty transition buffer
        self._training_quality = quality_builder.build()
        self._target_quality = quality_builder.build()
        self._transitions = deque(maxlen=experience_counts)
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
        state = environment.current_state
        if state.is_ended():
            state = environment.reset()
        action = self._select_action(state)
        # Execute action a in an emulator and observe reward r and the next state s'
        transition = environment.execute(action)
        # Store transition in the transition buffer
        self._transitions.append(transition)
        if self._step >= self.starting_step:
            # Sample a random batch from the buffer
            batch = [
                Experience(t.old_state, t.action, self._target_quality.calculate(t))
                for t in sample(self._transitions, self.batch_size)
            ]
            # Update Q(s, a)
            self._training_quality.learn(batch)
        self._step += 1

    def play(self, environment: Environment) -> Tuple[State, int, int, float]:
        """
        # Arguments
            environment: Environment. The environment to play inside.
        # Returns the last state, the count of total actions and wasted actions,
            and the cumulative reward.
        """
        environment.reset()
        total_actions_count = 0
        wasted_actions_count = 0
        reward = 0
        while True:
            state = environment.current_state
            action = self._select_action(state)
            transition = environment.execute(action)
            total_actions_count += 1
            if transition.reward <= 0:
                wasted_actions_count += 1
            reward += transition.reward
            next_state = transition.state
            if next_state.is_ended():
                return (next_state, total_actions_count, wasted_actions_count, reward)

    def _select_action(self, state: State) -> Action:
        """
        # Arguments
            state: State. The state from which the action is selected.
        # Returns selected action.
        """
        epsilon = max(
            self.epsilon_start - self._step / self.epsilon_decay_rate,
            self.epsilon_end
        )
        # With probability ε, select a random action, otherwise use quality model to predict
        if uniform(0, self.epsilon_start) < epsilon:
            action = self.action_builder.randomly_build()
        else:
            action = self._training_quality.predict(state)
        return action
