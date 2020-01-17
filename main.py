# pylint: skip-file

# %%
"""
Main
"""

from sys import stdout
from dqn_2048 import Direction, StateBuilder, Environment, QualityBuilder, Agent

BOARD_SIZE = 4
BOARD_UNIT = 2

GAMMA = 0.99
LEARNING_RATE = 1e-4

BATCH_SIZE = EXPERIENCES_COUNT = STARTING_STEP = 400
TARGET_SYNCING_FREQUENCY = 100
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY_RATE = 50000

STEPS_COUNT = 100000

state_builder = StateBuilder().set_size(BOARD_SIZE).set_unit(BOARD_UNIT)
environment = Environment(state_builder)

quality_builder = QualityBuilder().set_gamma(GAMMA) \
    .set_input_size(BOARD_SIZE ** 2) \
    .set_output_size(len(Direction)) \
    .set_learning_rate(LEARNING_RATE)
agent = Agent(
    quality_builder,
    BATCH_SIZE, EXPERIENCES_COUNT, STARTING_STEP, TARGET_SYNCING_FREQUENCY
)
agent.set_epsilons(EPSILON_START, EPSILON_END, EPSILON_DECAY_RATE)

result = open("result.txt", "w+")
for step in range(STEPS_COUNT):
    agent.observe(environment)
    if step % 100 == 0:
        stdout.write(str(step) + "\n")
        stdout.flush()
    if step % 10000 == 0:
        last_state, transitions_count, reward = agent.play(Environment(state_builder))
        result.write("\n".join([
            f"STEP: {step}.",
            f"Total transitions: {transitions_count}. Achieved reward: {reward}.",
            str(last_state)
        ]) + "\n" * 2)
        result.flush()
        if step > 0:
            agent.save("result")
result.close()

# %%
