"""
Agent
"""

from collections import deque
from random import uniform, sample

from ..environment.action import Action
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
            batch_size: int,
            experience_counts: int,
            starting_step: int,
            target_syncing_frequency: int,
            epsilon_start: float,
            epsilon_end: float,
            epsilon_decay_rate: int,
            steps_count: int
        ):
        """
        # Arguments
            quality_builder: QualityBuilder. Quality builder.
            batch_size: int. The batch size sampled from the experience buffer.
            experience_counts: int. The maximum capacity of the buffer.
            starting_step: int. The count of steps we wait for before starting training
                to populate the experience buffer.
            target_syncing_frequency: int. How frequently we sync model weights
                from the training model to the target model,
                which is used to get the value of the next state in the Bellman approximation.
            epsilon_start: float. Starting positive epsilon value.
            epsilon_end: float. Ending positive epsilon value, less than `epsilon_start`.
            epsilon_decay_rate: int. The rate of epsilon decay schedule.
            steps_count: int. Number of steps to be run.
        """
        self.batch_size = batch_size
        self.starting_step = starting_step
        self.target_syncing_frequency = target_syncing_frequency
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        self.steps_count = steps_count
        # Initialize parameters for Q(s, a) and Qˆ(s, a) with random weights
        # and empty experience buffer
        self._training_quality = quality_builder.build()
        self._target_quality = quality_builder.build()
        self._experiences = deque(maxlen=experience_counts)
        self._step = 0

    def observe(self, environment: Environment):
        """
        # Arguments
            environment: Environment. The environment to observe.
        """
        while self._step < self.steps_count:
            state = environment.current_state
            if state.is_ended():
                state = environment.reset()
            epsilon = max(
                self.epsilon_start - self._step / self.epsilon_decay_rate,
                self.epsilon_end
            )
            # With probability ε, select a random action, otherwise use quality model to predict
            if uniform(0, self.epsilon_start) < epsilon:
                action = Action.random()
            else:
                action = self._training_quality.predict(state)
            # Execute action a in an emulator and observe reward r and the next state s'
            transition = environment.execute(action)
            # Calculate target
            target_value = self._target_quality.calculate(transition)
            # Store experience in the experience buffer
            experience = Experience(transition.old_state, transition.action, target_value)
            self._experiences.append(experience)
            if self._step >= self.starting_step:
                # Sample a random batch of experiences from the experience buffer
                batch = sample(self._experiences, self.batch_size)
                # Update Q(s, a)
                self._training_quality.learn(batch)
            # Copy weights from Q to Qˆ
            if self._step % self.target_syncing_frequency == 0:
                self._target_quality.copied(self._training_quality)
            self._step += 1
